// SPDX-License-Identifier: Apache-2.0
// Q1 polls the complete fixed H2D payload on one resident observer core.
// Read readiness is polled while sampling the low 32-bit card clock.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_scratch = 0;
constexpr uint32_t page_bytes = 4096;
constexpr uint32_t page_words = page_bytes / sizeof(uint32_t);
constexpr uint32_t max_sample_gap = 0x7fffffff;
constexpr uint32_t stats_word_count = 24;

struct WrapTrackedClock {
    uint32_t previous = 0;
    uint64_t extended = 0;
    uint32_t ambiguous = 0;

    void initialize() {
        previous = get_timestamp_32b();
        extended = previous;
    }

    uint64_t sample() {
        const uint32_t low = get_timestamp_32b();
        const uint32_t delta = low - previous;
        if (delta > max_sample_gap) {
            ambiguous = 1;
        }
        extended += delta;
        previous = low;
        return extended;
    }
};

struct ReadPage {
    bool matches;
    uint64_t start_ticks;
    uint64_t end_ticks;
};

template <typename Accessor>
ReadPage read_and_check_page(
    const Accessor& payload,
    uint32_t page,
    uint32_t marker,
    WrapTrackedClock& clock) {
    cb_reserve_back(cb_scratch, 1);
    const uint32_t local = get_write_ptr(cb_scratch);
    const uint64_t start = clock.sample();
    noc_async_read_page(page, payload, local);
    while (!ncrisc_noc_reads_flushed(noc_index)) {
        clock.sample();
    }
    invalidate_l1_cache();
    const auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local);
    bool matches = true;
    for (uint32_t word = 0; word < page_words; ++word) {
        const uint32_t index = page * page_words + word;
        const uint32_t expected = word == page_words - 1 ? marker ^ page : 0x12000000u ^ index;
        matches &= words[word] == expected;
    }
    const uint64_t end = clock.sample();
    cb_push_back(cb_scratch, 1);
    cb_wait_front(cb_scratch, 1);
    cb_pop_front(cb_scratch, 1);
    return {matches, start, end};
}

template <typename Accessor>
void write_stats_page(
    const Accessor& stats,
    uint32_t words[stats_word_count],
    WrapTrackedClock& clock) {
    cb_reserve_back(cb_scratch, 1);
    const uint32_t local = get_write_ptr(cb_scratch);
    auto* output = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local);
    for (uint32_t index = 0; index < page_words; ++index) {
        output[index] = 0;
    }
    for (uint32_t index = 0; index < stats_word_count; ++index) {
        output[index] = words[index];
    }
    noc_async_write_page(0, stats, local);
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
        clock.sample();
    }
    cb_push_back(cb_scratch, 1);
    cb_wait_front(cb_scratch, 1);
    cb_pop_front(cb_scratch, 1);
}

struct ScanResult {
    bool complete;
    uint64_t last_negative_read_start;
    uint64_t scan_end;
    uint32_t pages_read;
};

template <typename Accessor>
ScanResult scan_payload(
    const Accessor& payload,
    uint32_t page_count,
    uint32_t marker,
    bool verify_full_payload,
    WrapTrackedClock& clock,
    uint64_t last_negative_read_start,
    uint32_t& total_pages_read) {
    const uint32_t witness_page = page_count - 1;
    const ReadPage witness = read_and_check_page(payload, witness_page, marker, clock);
    ++total_pages_read;
    if (!witness.matches) {
        return {false, witness.start_ticks, witness.end_ticks, 1};
    }
    if (!verify_full_payload) {
        return {true, last_negative_read_start, witness.end_ticks, 1};
    }
    for (uint32_t page = 0; page < page_count; ++page) {
        const ReadPage result = read_and_check_page(payload, page, marker, clock);
        ++total_pages_read;
        if (!result.matches) {
            return {false, result.start_ticks, clock.sample(), page_count + 1};
        }
    }
    return {true, last_negative_read_start, clock.sample(), page_count + 1};
}
}  // namespace

void kernel_main() {
    const uint32_t payload_address = get_arg_val<uint32_t>(0);
    const uint32_t stats_address = get_arg_val<uint32_t>(1);
    const uint32_t marker = get_arg_val<uint32_t>(2);
    const uint32_t page_count = get_arg_val<uint32_t>(3);
    const uint64_t resident_ticks = static_cast<uint64_t>(get_arg_val<uint32_t>(4)) |
        (static_cast<uint64_t>(get_arg_val<uint32_t>(5)) << 32);
    const uint64_t tail_ticks = static_cast<uint64_t>(get_arg_val<uint32_t>(6)) |
        (static_cast<uint64_t>(get_arg_val<uint32_t>(7)) << 32);

    constexpr auto payload_args = TensorAccessorArgs<0>();
    constexpr auto stats_args = TensorAccessorArgs<payload_args.next_compile_time_args_offset()>();
    const auto payload = TensorAccessor(payload_args, payload_address);
    const auto stats = TensorAccessor(stats_args, stats_address);

    WrapTrackedClock clock;
    clock.initialize();
    const uint64_t program_start = clock.extended;
    uint32_t total_pages_read = 0;
    const ScanResult before = scan_payload(payload, page_count, marker, true, clock, 0, total_pages_read);
    const uint64_t resident_start = clock.sample();
    uint64_t resident_window_end = resident_start;
    const uint64_t primary_deadline = resident_start + resident_ticks;
    const uint64_t observer_deadline = primary_deadline + tail_ticks;
    bool payload_seen = before.complete;
    uint64_t completion_lower = before.complete ? 0 : before.last_negative_read_start;
    uint64_t completion_upper = before.complete ? resident_start : 0;
    uint64_t last_negative_read_start = before.last_negative_read_start;
    uint64_t observer_end = resident_start;

    // The witness page is polled cheaply. Only after it matches do we read and
    // verify every page, so completion is never inferred from an unrelated
    // control write or from a single page of a multi-page DMA.
    while (clock.extended < observer_deadline) {
        const uint64_t poll_deadline = clock.extended + 135'000'000;
        while (clock.extended < poll_deadline) {
            clock.sample();
        }
        const ScanResult scan = scan_payload(
            payload, page_count, marker, !payload_seen, clock, last_negative_read_start, total_pages_read);
        last_negative_read_start = scan.last_negative_read_start;
        if (scan.complete && !payload_seen) {
            payload_seen = true;
            completion_lower = last_negative_read_start;
            completion_upper = scan.scan_end;
        }
        if (resident_window_end == resident_start && scan.scan_end >= primary_deadline) {
            resident_window_end = scan.scan_end;
        }
        observer_end = scan.scan_end;
    }
    if (resident_window_end == resident_start) {
        resident_window_end = clock.sample();
    }
    if (observer_end == resident_start) {
        observer_end = clock.sample();
    }

    // 0=not seen, 1=inside primary window, 2=after primary window but still
    // observed in the active CQ0 program, 3=before window, 4=boundary/gap.
    uint32_t primary_relation = 0;
    if (payload_seen && clock.ambiguous == 0) {
        if (completion_upper <= resident_start) {
            primary_relation = 3;
        } else if (completion_lower >= resident_window_end) {
            primary_relation = 2;
        } else if (completion_lower >= resident_start && completion_upper <= resident_window_end) {
            primary_relation = 1;
        } else {
            primary_relation = 4;
        }
    }
    uint32_t words[stats_word_count] = {
        static_cast<uint32_t>(resident_start),
        static_cast<uint32_t>(resident_start >> 32),
        static_cast<uint32_t>(resident_window_end),
        static_cast<uint32_t>(resident_window_end >> 32),
        static_cast<uint32_t>(completion_lower),
        static_cast<uint32_t>(completion_lower >> 32),
        static_cast<uint32_t>(completion_upper),
        static_cast<uint32_t>(completion_upper >> 32),
        payload_seen,
        primary_relation,
        clock.ambiguous,
        static_cast<uint32_t>(observer_end),
        static_cast<uint32_t>(observer_end >> 32),
        marker,
        page_count,
        total_pages_read,
        static_cast<uint32_t>(primary_deadline),
        static_cast<uint32_t>(primary_deadline >> 32),
        static_cast<uint32_t>(observer_deadline),
        static_cast<uint32_t>(observer_deadline >> 32),
        static_cast<uint32_t>(program_start),
        static_cast<uint32_t>(program_start >> 32),
        0,
        0,
    };
    write_stats_page(stats, words, clock);
}
