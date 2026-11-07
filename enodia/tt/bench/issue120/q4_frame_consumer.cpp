// SPDX-License-Identifier: Apache-2.0
// Real ring-frame work and transfer observation share this low-word-only clock.
#include "q4_clock.hpp"

using namespace issue120_q4;

namespace {
struct Progress {
    uint32_t sequence;
    uint32_t challenge;
    uint64_t start;
    uint64_t end;
    bool valid;
};

template <typename Accessor>
Progress observe_one(const Accessor& progress, uint32_t scratch, uint32_t limit, Clock& clock) {
    const uint64_t start = clock.read();
    noc_async_read_page(0, progress, scratch);
    clock.reads_complete();
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t encoded = words[0];
    const uint32_t sequence = protocol::unpack_sequence(encoded);
    const uint32_t challenge = protocol::unpack_challenge(encoded);
    bool valid = sequence > 0 && sequence <= limit;
    for (uint32_t word = 1; word < page_words; ++word) {
        valid &= words[word] == protocol::word(encoded, word);
    }
    return {sequence, challenge, start, clock.read(), valid};
}

template <typename Accessor>
Progress observe(const Accessor& progress, uint32_t scratch, uint32_t limit, Clock& clock) {
    Progress result = observe_one(progress, scratch, limit, clock);
    // Always do three reads, never retry-until-success in the frame path.
    for (uint32_t attempt = 1; attempt < 3; ++attempt) {
        const Progress candidate = observe_one(progress, scratch, limit, clock);
        if (candidate.valid) { result = candidate; }
    }
    return result;
}

template <typename Accessor>
Progress observe_change(const Accessor& progress, uint32_t scratch, uint32_t limit,
                        const Progress& left, uint32_t challenge, uint64_t deadline, Clock& clock) {
    Progress right = left;
    while (clock.read() < deadline) {
        right = observe(progress, scratch, limit, clock);
        if (left.valid && right.valid && right.sequence > left.sequence && right.challenge == challenge) { break; }
    }
    return right;
}

uint64_t issue_challenge(volatile tt_l1_ptr uint32_t* mailbox, uint32_t token, Clock& clock) {
    const uint64_t issued = clock.read();
    *mailbox = token;
    asm volatile("fence" ::: "memory");
    return issued;
}
}

void kernel_main() {
    const uint32_t ring_address = get_arg_val<uint32_t>(0);
    const uint32_t anchor_address = get_arg_val<uint32_t>(1);
    const uint32_t progress_address = get_arg_val<uint32_t>(2);
    const uint32_t frames_address = get_arg_val<uint32_t>(3);
    const uint32_t stats_address = get_arg_val<uint32_t>(4);
    const uint32_t frame_count = get_arg_val<uint32_t>(5);
    const uint32_t period = get_arg_val<uint32_t>(6);
    const uint32_t ring_pages = get_arg_val<uint32_t>(7);
    const uint32_t repeat_count = get_arg_val<uint32_t>(8);
    auto* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(11));
    const uint64_t budget = uint64_t(get_arg_val<uint32_t>(9)) |
        (uint64_t(get_arg_val<uint32_t>(10)) << 32);
    constexpr auto ring_args = TensorAccessorArgs<0>();
    constexpr auto anchor_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr auto progress_args = TensorAccessorArgs<anchor_args.next_compile_time_args_offset()>();
    constexpr auto frames_args = TensorAccessorArgs<progress_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<frames_args.next_compile_time_args_offset()>();
    const auto anchor = TensorAccessor(anchor_args, anchor_address);
    const auto progress = TensorAccessor(progress_args, progress_address);
    const auto frames = TensorAccessor(frames_args, frames_address);
    const auto stats = TensorAccessor(stats_args, stats_address);
    const uint64_t producer = anchor.get_noc_addr(0) & coordinate_mask;
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_id));
    auto* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_id));
    cb_reserve_back(0, 2);
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t record_address = scratch + page_bytes;
    auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record_address);
    Clock clock;
    clock.initialize();
    const uint64_t opened = clock.read();
    // Startup/tail observation is bounded and outside frame processing. The
    // normal frame path never waits for transfer progress to improve.
    issue_challenge(mailbox, 1, clock);
    Progress left = observe(progress, scratch, repeat_count, clock);
    const uint64_t startup_deadline = clock.read() + 135000000;
    while ((!left.valid || left.challenge != 1) && clock.read() < startup_deadline) {
        left = observe(progress, scratch, repeat_count, clock);
    }
    const Progress startup = observe_change(progress, scratch, repeat_count, left,
        1, startup_deadline, clock);
    const bool gated = left.valid && startup.valid && startup.sequence > left.sequence && startup.challenge == 1;
    left = startup;
    const uint64_t load_start = left.start;
    uint32_t token = 2;
    uint64_t challenge_issued = issue_challenge(mailbox, token, clock);
    const uint64_t start_target = producer | get_semaphore(start_id);
    const uint64_t start_inc_ticks = clock.read();
    noc_semaphore_inc(start_target, 1);
    clock.atomics_complete();
    uint32_t consumed = 0;
    uint32_t aborted = 0;
    for (; consumed < frame_count; ++consumed) {
        while (*ready <= consumed) {
            invalidate_l1_cache();
            if (*done != 0) { break; }
            if (clock.read() - opened >= budget) { aborted = 1; break; }
        }
        invalidate_l1_cache();
        if (*ready <= consumed || aborted != 0) { break; }
        const uint64_t frame_start = clock.read();
        const Progress right = observe(progress, scratch, repeat_count, clock);
        const auto* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            ring_address + (consumed % ring_pages) * page_bytes);
        const uint32_t payload_frame_id = payload[0];
        uint32_t frame_checksum = 0;
        for (uint32_t work = 0; work < 64; ++work) {
            frame_checksum = (frame_checksum * 33u) ^ payload[work] ^ (work + consumed);
        }
        const uint64_t frame_end = clock.read();
        const uint32_t left_sequence = left.sequence;
        const uint32_t right_sequence = right.sequence;
        for (uint32_t word = 0; word < page_words; ++word) { row[word] = 0; }
        row[0] = consumed;
        row[1] = uint32_t(frame_start); row[2] = uint32_t(frame_start >> 32);
        row[3] = uint32_t(frame_end); row[4] = uint32_t(frame_end >> 32);
        row[5] = left_sequence; row[6] = right_sequence;
        row[7] = uint32_t(challenge_issued); row[8] = uint32_t(challenge_issued >> 32);
        row[9] = uint32_t(right.end); row[10] = uint32_t(right.end >> 32);
        row[11] = left.valid && right.valid;
        row[12] = frame_checksum;
        row[13] = payload_frame_id;
        row[14] = 64;
        row[15] = clock.ambiguous;
        row[16] = token;
        row[17] = right.challenge;
        noc_async_write_page(consumed, frames, record_address);
        clock.writes_complete();
        noc_semaphore_inc(producer | get_semaphore(free_id), 1);
        clock.atomics_complete();
        // Observe the old generation, then issue a fresh CPU challenge after
        // this frame endpoint. A matching later response proves CPU work and
        // data-visible DMA after this same-clock challenge-issue timestamp.
        left = observe(progress, scratch, repeat_count, clock);
        ++token;
        challenge_issued = issue_challenge(mailbox, token, clock);
    }
    const Progress tail = observe_change(progress, scratch, repeat_count, left,
        token, clock.read() + period, clock);
    const uint64_t load_end = tail.end;
    for (uint32_t word = 0; word < page_words; ++word) { row[word] = 0; }
    row[0] = consumed;
    row[1] = clock.ambiguous;
    row[2] = gated;
    row[3] = uint32_t(load_start); row[4] = uint32_t(load_start >> 32);
    row[5] = uint32_t(load_end); row[6] = uint32_t(load_end >> 32);
    row[7] = left.sequence; row[8] = tail.sequence;
    row[9] = uint32_t(challenge_issued); row[10] = uint32_t(challenge_issued >> 32);
    row[11] = uint32_t(tail.end); row[12] = uint32_t(tail.end >> 32);
    row[13] = left.valid && tail.valid;
    row[14] = aborted;
    row[15] = startup.challenge;
    row[16] = token;
    row[17] = tail.challenge;
    row[18] = uint32_t(start_target); row[19] = uint32_t(start_target >> 32);
    row[20] = uint32_t(start_inc_ticks); row[21] = uint32_t(start_inc_ticks >> 32);
    row[22] = uint32_t(get_semaphore(0));
    row[23] = uint32_t(get_arg_val<uint32_t>(11));
    row[24] = *mailbox;
    noc_async_write_page(0, stats, record_address);
    clock.writes_complete();
    cb_push_back(0, 2);
    cb_wait_front(0, 2);
    cb_pop_front(0, 2);
}
