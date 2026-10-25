// SPDX-License-Identifier: Apache-2.0
// Resident consumer: fixed work and one designated-core timestamp stream.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "resident_tables.h"

namespace {
constexpr std::uint32_t cb_timestamp = 21;
constexpr std::uint32_t page_words = 32 * 32;
constexpr std::uint32_t ready_semaphore_id = 0;
constexpr std::uint32_t free_semaphore_id = 1;
constexpr std::uint32_t done_semaphore_id = 2;
constexpr std::uint32_t error_semaphore_id = 3;
constexpr std::uint32_t failure_run_wide_budget = 1;
constexpr std::uint32_t failure_consumer_empty_wait = 3;
constexpr std::uint32_t failure_consumer_fixed_work_budget = 4;

struct WrapTrackedClock {
    std::uint64_t extended = 0;
    void initialize() { extended = get_timestamp_32b(); }
    std::uint64_t read() {
        const std::uint32_t low = get_timestamp_32b();
        extended += static_cast<std::uint32_t>(low - static_cast<std::uint32_t>(extended));
        return extended;
    }
};
}

void kernel_main() {
    const std::uint32_t ring_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t producer_anchor_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t timestamp_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t stats_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t frame_count = get_arg_val<std::uint32_t>(4);
    const std::uint32_t ring_pages = get_arg_val<std::uint32_t>(5);
    const std::uint32_t work_per_frame = get_arg_val<std::uint32_t>(6);
    const std::uint32_t per_frame_work_budget_ticks = get_arg_val<std::uint32_t>(7);
    const std::uint64_t run_budget_ticks =
        static_cast<std::uint64_t>(get_arg_val<std::uint32_t>(8))
        | (static_cast<std::uint64_t>(get_arg_val<std::uint32_t>(9)) << 32);
    const std::uint32_t tables_enabled = get_arg_val<std::uint32_t>(10);
    const std::uint32_t table_pages = get_arg_val<std::uint32_t>(13);
    const std::uint32_t prefetch = get_arg_val<std::uint32_t>(14);

    constexpr auto ring_args = TensorAccessorArgs<0>();
    constexpr auto producer_anchor_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr auto timestamp_args = TensorAccessorArgs<producer_anchor_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<timestamp_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    constexpr auto tables_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();
    const auto ring = TensorAccessor(ring_args, ring_address);
    const auto producer_anchor = TensorAccessor(producer_anchor_args, producer_anchor_address);
    const auto timestamps = TensorAccessor(timestamp_args, timestamp_address);
    const auto stats = TensorAccessor(stats_args, stats_address);
    const auto metadata_accessor = TensorAccessor(metadata_args, get_arg_val<std::uint32_t>(11));
    const auto tables = TensorAccessor(tables_args, get_arg_val<std::uint32_t>(12));

    const std::uint64_t producer_anchor_noc = producer_anchor.get_noc_addr(0);
    const std::uint64_t noc_coord_mask = ~((std::uint64_t(1) << NOC_ADDR_COORD_SHIFT) - 1);
    const std::uint64_t free_noc =
        (producer_anchor_noc & noc_coord_mask) | get_semaphore(free_semaphore_id);
    const std::uint64_t error_noc =
        (producer_anchor_noc & noc_coord_mask) | get_semaphore(error_semaphore_id);
    auto* ready_sem = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_semaphore(ready_semaphore_id));
    auto* done_sem = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_semaphore(done_semaphore_id));
    std::uint32_t consumer_empty_count = 0;
    std::uint32_t frames_consumed = 0;
    std::uint32_t frames_retired = 0;
    std::uint32_t discard_count = 0;
    std::uint32_t error_flag = 0;
    std::uint32_t accumulator = 0;
    std::uint32_t failure_code = 0;
    std::uint64_t failure_elapsed_ticks = 0;
    std::uint64_t failure_limit_ticks = 0;
    std::uint64_t startup_ticks = 0;
    std::uint64_t work_min_ticks = UINT64_MAX;
    std::uint64_t work_max_ticks = 0;
    std::uint32_t work_valid = 0;
    std::uint32_t startup_valid = 0;
    std::uint64_t previous_completion = 0;
    bool error_sent = false;
    WrapTrackedClock clock;
    clock.initialize();
    const std::uint64_t run_start = clock.read();

    volatile tt_l1_ptr std::uint32_t* metadata = nullptr;
    volatile tt_l1_ptr std::uint32_t* switch_stats = nullptr;
    std::uint32_t table_base = 0;
    std::uint32_t active_slot = 0;
    std::uint32_t loaded[2] = {resident_tables::no_table, resident_tables::no_table};
    if (tables_enabled) {
        cb_reserve_back(resident_tables::metadata_consumer_cb, 1);
        const auto address = get_write_ptr(resident_tables::metadata_consumer_cb);
        noc_async_read_page(0, metadata_accessor, address);
        noc_async_read_barrier();
        metadata = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(address);
        cb_reserve_back(resident_tables::tables_cb, table_pages * (prefetch ? 2 : 1));
        table_base = get_write_ptr(resident_tables::tables_cb);
        cb_reserve_back(resident_tables::switch_stats_cb, 1);
        switch_stats = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            get_write_ptr(resident_tables::switch_stats_cb));
        for (std::uint32_t word = 0; word < page_words; ++word) {
            switch_stats[word] = 0;
        }
    }

    while (true) {
        invalidate_l1_cache();
        const std::uint32_t required = frames_retired + 1;
        const std::uint32_t ready_count = *ready_sem;
        if (*done_sem != 0) {
            invalidate_l1_cache();
            const std::uint32_t fresh_ready_count = *ready_sem;
            if (fresh_ready_count == frames_retired) {
                break;
            }
        }
        auto* payload = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            ring_address + (frames_retired % ring_pages) * page_words * sizeof(std::uint32_t));
        if (ready_count < required) {
            consumer_empty_count += 1;
        }
        while (*ready_sem < required) {
            invalidate_l1_cache();
            if (*done_sem != 0) {
                invalidate_l1_cache();
                const std::uint32_t fresh_ready_count = *ready_sem;
                if (fresh_ready_count == frames_retired) {
                    break;
                }
            }
            if (clock.read() - run_start >= run_budget_ticks) {
                error_flag = 1;
                failure_code = failure_consumer_empty_wait;
                failure_elapsed_ticks = clock.read() - run_start;
                failure_limit_ticks = run_budget_ticks;
                if (!error_sent) {
                    noc_semaphore_inc(error_noc, 1);
                    noc_async_atomic_barrier();
                    error_sent = true;
                }
                break;
            }
        }
        invalidate_l1_cache();
        if (*ready_sem < required) {
            if (*done_sem != 0) {
                invalidate_l1_cache();
                const std::uint32_t fresh_ready_count = *ready_sem;
                if (fresh_ready_count == frames_retired) {
                    break;
                }
            }
            if (error_flag != 0) {
                break;
            }
            continue;
        }

        const std::uint64_t start = clock.read();
        std::uint32_t frame = frames_consumed;
        std::uint32_t segment = 0;
        bool discard = false;
        resident_tables::Identity identity{0, 0};
        if (tables_enabled) {
            identity = {payload[0], payload[1]};
            frame = payload[2];
            segment = payload[3];
            const auto index = resident_tables::ready_table(metadata, identity, frame);
            discard = index == resident_tables::no_table;
            if (!discard) {
                if (loaded[active_slot] != index) {
                    if (prefetch && loaded[1 - active_slot] == index) {
                        active_slot = 1 - active_slot;
                    } else {
                        resident_tables::copy_table(tables, index, table_pages,
                            table_base + active_slot * table_pages * resident_tables::bytes_per_page);
                        loaded[active_slot] = index;
                    }
                }
                accumulator = 0;
            }
            // One-frame lookahead. Skipped/unready lookahead uses demand copy, never fallback.
            if (!discard && prefetch && segment < metadata[1]) {
                const std::uint32_t offset = resident_tables::schedule_offset(metadata) + 3 * segment;
                if (frame + 1 == metadata[offset]) {
                    const resident_tables::Identity next{metadata[offset + 1], metadata[offset + 2]};
                    const auto next_index = resident_tables::ready_table(metadata, next, frame + 1);
                    if (next_index != resident_tables::no_table && loaded[1 - active_slot] != next_index) {
                        resident_tables::copy_table(tables, next_index, table_pages,
                            table_base + (1 - active_slot) * table_pages * resident_tables::bytes_per_page);
                        loaded[1 - active_slot] = next_index;
                    }
                }
            }
        }
        if (!discard) {
            auto* selected_table = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
                table_base + active_slot * table_pages * resident_tables::bytes_per_page);
            for (std::uint32_t work = 0; work < work_per_frame; ++work) {
                const std::uint32_t word = work % page_words;
                if (tables_enabled) {
                    const auto input = word < 4 ? resident_tables::synthetic_word(frame, word) : payload[word];
                    accumulator = (accumulator * 33u) ^ input ^ (work + frame)
                        ^ selected_table[work % (table_pages * page_words)];
                } else {
                    accumulator = (accumulator * 33u) ^ payload[word] ^ (work + frames_consumed);
                }
            }
        }
        const std::uint64_t end = clock.read();
        const std::uint64_t elapsed = end - start;
        if (!discard) {
            work_min_ticks = elapsed < work_min_ticks ? elapsed : work_min_ticks;
            work_max_ticks = elapsed > work_max_ticks ? elapsed : work_max_ticks;
            work_valid = 1;
            if (elapsed >= static_cast<std::uint64_t>(per_frame_work_budget_ticks)) {
                error_flag = 1;
                failure_code = failure_consumer_fixed_work_budget;
                failure_elapsed_ticks = elapsed;
                failure_limit_ticks = per_frame_work_budget_ticks;
            }
        }
        if (error_flag == 0 && end - run_start >= run_budget_ticks) {
            error_flag = 1;
            failure_code = failure_run_wide_budget;
            failure_elapsed_ticks = end - run_start;
            failure_limit_ticks = run_budget_ticks;
        }
        if (error_flag != 0 && !error_sent) {
            noc_semaphore_inc(error_noc, 1);
            noc_async_atomic_barrier();
            error_sent = true;
        }

        if (discard) {
            discard_count += 1;
            switch_stats[4 * segment] += 1;
            frames_retired += 1;
            noc_semaphore_inc(free_noc, 1);
            noc_async_atomic_barrier();
            if (error_flag != 0) {
                break;
            }
            continue;
        }
        if (tables_enabled && switch_stats[4 * segment + 3] == 0) {
            const auto interval = frames_consumed ? end - previous_completion : 0;
            switch_stats[4 * segment + 1] = static_cast<std::uint32_t>(interval);
            switch_stats[4 * segment + 2] = static_cast<std::uint32_t>(interval >> 32);
            switch_stats[4 * segment + 3] = frames_consumed ? 2 : 1;
        }
        previous_completion = end;
        cb_reserve_back(cb_timestamp, 1);
        auto* timestamp_page = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            get_write_ptr(cb_timestamp));
        timestamp_page[0] = static_cast<std::uint32_t>(end);
        timestamp_page[1] = static_cast<std::uint32_t>(end >> 32);
        timestamp_page[2] = accumulator;
        timestamp_page[3] = static_cast<std::uint32_t>(elapsed);
        if (tables_enabled) {
            timestamp_page[4] = frame;
            timestamp_page[5] = identity.config;
            timestamp_page[6] = identity.generation;
            timestamp_page[7] = segment;
        }
        cb_push_back(cb_timestamp, 1);
        cb_wait_front(cb_timestamp, 1);
        noc_async_write_page(frames_consumed, timestamps, get_read_ptr(cb_timestamp));
        noc_async_write_barrier();
        cb_pop_front(cb_timestamp, 1);
        if (frames_consumed == 0) {
            startup_ticks = end - run_start;
            startup_valid = 1;
        }
        frames_consumed += 1;
        frames_retired += 1;
        noc_semaphore_inc(free_noc, 1);
        noc_async_atomic_barrier();
        if (error_flag != 0) {
            break;
        }
    }

    cb_reserve_back(cb_timestamp, 1);
    auto* summary = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_write_ptr(cb_timestamp));
    summary[0] = consumer_empty_count;
    summary[1] = frames_consumed;
    summary[2] = error_flag;
    summary[3] = accumulator;
    summary[4] = static_cast<std::uint32_t>(startup_ticks);
    summary[5] = static_cast<std::uint32_t>(startup_ticks >> 32);
    summary[6] = startup_valid;
    summary[7] = failure_code;
    summary[8] = static_cast<std::uint32_t>(failure_elapsed_ticks);
    summary[9] = static_cast<std::uint32_t>(failure_elapsed_ticks >> 32);
    summary[10] = static_cast<std::uint32_t>(failure_limit_ticks);
    summary[11] = static_cast<std::uint32_t>(failure_limit_ticks >> 32);
    summary[12] = failure_code != 0;
    summary[13] = static_cast<std::uint32_t>(work_min_ticks);
    summary[14] = static_cast<std::uint32_t>(work_min_ticks >> 32);
    summary[15] = static_cast<std::uint32_t>(work_max_ticks);
    summary[16] = static_cast<std::uint32_t>(work_max_ticks >> 32);
    summary[17] = work_valid;
    if (tables_enabled) {
        summary[18] = discard_count;
        summary[19] = frames_retired;
        for (std::uint32_t word = 0; word < 4 * (metadata[1] + 1); ++word) {
            summary[20 + word] = switch_stats[word];
        }
    }
    cb_push_back(cb_timestamp, 1);
    cb_wait_front(cb_timestamp, 1);
    noc_async_write_page(0, stats, get_read_ptr(cb_timestamp));
    noc_async_write_barrier();
    cb_pop_front(cb_timestamp, 1);
}
