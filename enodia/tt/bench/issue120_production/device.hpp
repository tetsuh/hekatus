// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/dataflow_api.h"
#include "protocol.hpp"

namespace issue120_production {
struct Clock {
    uint64_t extended = 0;
    uint32_t previous = 0;
    bool ambiguous = false;
    void initialize() { previous = get_timestamp_32b(); extended = previous; }
    uint64_t read() {
        const uint32_t low = get_timestamp_32b();
        const uint32_t delta = low - previous;
        ambiguous |= delta >= 0x80000000u;
        extended += delta; previous = low; return extended;
    }
};
inline void reads_complete(Clock& clock) {
    while (!ncrisc_noc_reads_flushed(noc_index)) { clock.read(); }
    invalidate_l1_cache();
}
inline void writes_complete(Clock& clock) {
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) { clock.read(); }
}
inline void atomics_complete(Clock& clock) {
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_index)) { clock.read(); }
}
inline void wait_start(Clock& clock, uint64_t budget) {
    auto* start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(start_semaphore));
    const uint64_t opened = clock.read();
    const uint64_t limit = budget < 1350000000ULL ? budget : 1350000000ULL;
    while (*start == 0) {
        invalidate_l1_cache();
        if (clock.read() - opened >= limit) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3)) = 1;
            asm volatile("fence" ::: "memory");
            break;
        }
    }
}
inline void write_ring(uint32_t slot, uint32_t ring, uint32_t source) {
    noc_async_write(source, get_noc_addr(get_arg_val<uint32_t>(producer_arg::peer_x),
        get_arg_val<uint32_t>(producer_arg::peer_y), ring + slot * page_bytes), page_bytes);
}
struct Progress {
    uint32_t sequence = 0, sum = 0;
    uint64_t start = 0, end = 0;
    bool valid = false;
    uint32_t challenge() const { return sum - sequence; }
};
struct FrameObservation { uint64_t admission; Progress right; };
inline void put64(volatile tt_l1_ptr uint32_t* row, uint32_t offset, uint64_t value) {
    row[offset] = uint32_t(value); row[offset + 1] = uint32_t(value >> 32);
}

template<class Accessor>
struct Observer {
    const Accessor& progress;
    volatile tt_l1_ptr uint32_t* mailbox;
    uint32_t scratch, frames, repeats, period, token = 2;
    uint64_t startup_issued = 0, issued = 0, start_sent = 0;
    Progress startup_left, startup_right, left, tail;
    bool gated = false;

    Progress once(Clock& clock) {
        const uint64_t begin = clock.read();
        noc_async_read_page(0, progress, scratch);
        reads_complete(clock);
        const auto* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        const Wire wire{data[0], data[1]};
        bool valid = wire.sequence > 0 && wire.sequence <= repeats &&
            wire.sum >= wire.sequence && wire.sum - wire.sequence <= frames + 2;
        for (uint32_t i = 0; i < page_words; ++i) { valid &= data[i] == word(wire, i); }
        return {wire.sequence, wire.sum, begin, clock.read(), valid};
    }
    Progress observe(Clock& clock) {
        Progress selected = once(clock);
        for (uint32_t i = 1; i < 3; ++i) {
            const auto candidate = once(clock);
            if (candidate.valid) { selected = candidate; }
        }
        return selected;
    }
    uint64_t issue(uint32_t challenge, Clock& clock) {
        const uint64_t tick = clock.read();
        mailbox[0] = challenge;
        asm volatile("fence" ::: "memory");
        return tick;
    }
    Observer(const Accessor& accessor, uint32_t count, Clock& clock) :
        progress(accessor), mailbox(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_arg_val<uint32_t>(consumer_arg::mailbox))), frames(count),
        repeats(get_arg_val<uint32_t>(consumer_arg::repeats)),
        period(get_arg_val<uint32_t>(consumer_arg::period)) {
        cb_reserve_back(26, 1); scratch = get_write_ptr(26);
        startup_issued = issue(1, clock);
        const uint64_t deadline = startup_issued + 135000000;
        // Same bounded startup duration/policy in control and loaded. Never
        // retry for evidence in ordinary frame processing.
        while (clock.read() + period < deadline) {
            const auto candidate = observe(clock);
            if (candidate.valid && candidate.challenge() == 1) {
                if (!startup_left.valid) { startup_left = candidate; }
                if (candidate.sequence > startup_left.sequence) { startup_right = candidate; }
            }
        }
        gated = startup_left.valid && startup_right.valid;
        left = startup_right;
        issued = issue(token, clock);
        // Reserve the final startup period for the distinct first-frame token.
        // The resident producer still releases frame0 immediately after start.
        while (clock.read() < deadline) {}
        start_sent = clock.read();
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(consumer_arg::peer_x),
            get_arg_val<uint32_t>(consumer_arg::peer_y), get_semaphore(start_semaphore)), 1);
        atomics_complete(clock);
    }
    FrameObservation before_frame(Clock& clock) {
        const uint64_t admission = clock.read();
        return {admission, observe(clock)};
    }
    void record(volatile tt_l1_ptr uint32_t* row, uint32_t index,
                FrameObservation observed, uint64_t work_start, Clock& clock) {
        // Original Issue118 completion/checksum/work fields0-3 are untouched.
        for (uint32_t i = 4; i < 32; ++i) { row[i] = 0; }
        row[4] = index; put64(row, 5, observed.admission);
        row[7] = left.sequence; row[8] = left.sum;
        row[9] = observed.right.sequence; row[10] = observed.right.sum;
        put64(row, 11, issued); put64(row, 13, observed.right.end);
        row[15] = left.valid && observed.right.valid;
        row[16] = clock.ambiguous; row[17] = token;
        row[18] = get_arg_val<uint32_t>(6);
        put64(row, 19, left.start); put64(row, 21, left.end);
        put64(row, 23, observed.right.start); put64(row, 25, work_start);
    }
    void after_frame(uint32_t consumed, Clock& clock) {
        left = observe(clock);
        token = consumed + 2;
        issued = issue(token, clock);
    }
    void finish(Clock& clock) {
        tail = left;
        const uint64_t deadline = clock.read() + period;
        while (clock.read() < deadline) {
            const auto candidate = observe(clock);
            if (candidate.valid && candidate.sequence > left.sequence &&
                candidate.challenge() == token) { tail = candidate; }
        }
    }
    void summary(volatile tt_l1_ptr uint32_t* row, uint32_t consumed, Clock& clock) {
        for (uint32_t i = 32; i < 68; ++i) { row[i] = 0; }
        row[32] = consumed + 2; row[33] = frames + 2; row[34] = mailbox[0];
        row[35] = clock.ambiguous;
        row[36] = startup_left.sequence; row[37] = startup_left.sum;
        row[38] = startup_right.sequence; row[39] = startup_right.sum;
        put64(row, 40, startup_issued); put64(row, 42, startup_right.end);
        put64(row, 44, startup_right.start);
        row[46] = left.sequence; row[47] = left.sum;
        row[48] = tail.sequence; row[49] = tail.sum;
        put64(row, 50, issued); put64(row, 52, tail.end);
        row[54] = left.valid && tail.valid;
        row[55] = startup_left.valid && startup_right.valid; row[56] = gated;
        put64(row, 57, start_sent);
        row[62] = get_arg_val<uint32_t>(consumer_arg::peer_x);
        row[63] = get_arg_val<uint32_t>(consumer_arg::peer_y);
        row[64] = my_x[0]; row[65] = my_y[0];
    }
    void completed(Clock& clock) {
        const uint64_t tick = clock.read();
        mailbox[2] = uint32_t(tick); mailbox[3] = uint32_t(tick >> 32);
        asm volatile("fence" ::: "memory");
        mailbox[1] = 1;
        asm volatile("fence" ::: "memory");
        cb_push_back(26, 1); cb_wait_front(26, 1); cb_pop_front(26, 1);
    }
};
} // namespace issue120_production
