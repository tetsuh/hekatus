// SPDX-License-Identifier: Apache-2.0
// Issue #12 Stage 1 consumer: fixed work, one designated-core timestamp stream.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_timestamp = 1;
constexpr std::uint32_t page_words = 32 * 32;
constexpr std::uint32_t ready_word = page_words - 3;
constexpr std::uint32_t free_word = page_words - 2;
constexpr std::uint32_t control_error_word = 0;
constexpr std::uint32_t control_done_word = 1;
constexpr std::uint32_t control_produced_word = 3;
constexpr std::uint32_t failure_run_wide_budget = 1;
constexpr std::uint32_t failure_consumer_empty_wait = 3;
constexpr std::uint32_t failure_consumer_fixed_work_budget = 4;
constexpr std::uint32_t failure_other_check = 5;
}

void kernel_main() {
    const std::uint32_t ring_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t control_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t timestamp_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t stats_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t frame_count = get_arg_val<std::uint32_t>(4);
    const std::uint32_t ring_pages = get_arg_val<std::uint32_t>(5);
    const std::uint32_t work_per_frame = get_arg_val<std::uint32_t>(6);
    const std::uint32_t per_frame_work_budget_ticks = get_arg_val<std::uint32_t>(7);
    const std::uint64_t run_budget_ticks =
        static_cast<std::uint64_t>(get_arg_val<std::uint32_t>(8))
        | (static_cast<std::uint64_t>(get_arg_val<std::uint32_t>(9)) << 32);

    constexpr auto ring_args = TensorAccessorArgs<0>();
    constexpr auto control_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr auto timestamp_args = TensorAccessorArgs<control_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<timestamp_args.next_compile_time_args_offset()>();
    const auto ring = TensorAccessor(ring_args, ring_address);
    const auto control = TensorAccessor(control_args, control_address);
    const auto timestamps = TensorAccessor(timestamp_args, timestamp_address);
    const auto stats = TensorAccessor(stats_args, stats_address);

    auto* control_local = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(control_address);
    std::uint32_t consumer_empty_count = 0;
    std::uint32_t frames_consumed = 0;
    std::uint32_t error_flag = 0;
    std::uint32_t accumulator = 0;
    std::uint32_t failure_code = 0;
    std::uint64_t failure_elapsed_ticks = 0;
    std::uint64_t failure_limit_ticks = 0;
    std::uint64_t startup_ticks = 0;
    std::uint32_t startup_valid = 0;
    const std::uint64_t run_start = get_timestamp();

    while (true) {
        if (control_local[control_error_word] != 0) {
            error_flag = 1;
            failure_code = failure_other_check;
            failure_elapsed_ticks = get_timestamp() - run_start;
            failure_limit_ticks = run_budget_ticks;
            break;
        }
        const std::uint32_t required = frames_consumed + 1;
        auto* payload = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            ring_address + (frames_consumed % ring_pages) * page_words * sizeof(std::uint32_t));
        if (payload[ready_word] < required) {
            consumer_empty_count += 1;
        }
        while (payload[ready_word] < required && control_local[control_error_word] == 0) {
            invalidate_l1_cache();
            if (control_local[control_done_word] != 0
                && frames_consumed >= control_local[control_produced_word]) {
                break;
            }
            if (get_timestamp() - run_start >= run_budget_ticks) {
                error_flag = 1;
                failure_code = failure_consumer_empty_wait;
                failure_elapsed_ticks = get_timestamp() - run_start;
                failure_limit_ticks = run_budget_ticks;
                control_local[control_error_word] = 1;
                break;
            }
        }
        if (control_local[control_error_word] != 0) {
            error_flag = 1;
            if (failure_code == 0) {
                failure_code = failure_other_check;
                failure_elapsed_ticks = get_timestamp() - run_start;
                failure_limit_ticks = run_budget_ticks;
            }
            break;
        }
        if (payload[ready_word] < required) {
            if (control_local[control_done_word] != 0
                && frames_consumed >= control_local[control_produced_word]) {
                break;
            }
            continue;
        }

        const std::uint64_t start = get_timestamp();
        for (std::uint32_t work = 0; work < work_per_frame; ++work) {
            const std::uint32_t word = work % ready_word;
            accumulator = (accumulator * 33u) ^ payload[word] ^ (work + frames_consumed);
        }
        const std::uint64_t end = get_timestamp();
        const std::uint64_t elapsed = end - start;
        if (elapsed >= static_cast<std::uint64_t>(per_frame_work_budget_ticks)) {
            error_flag = 1;
            failure_code = failure_consumer_fixed_work_budget;
            failure_elapsed_ticks = elapsed;
            failure_limit_ticks = per_frame_work_budget_ticks;
            control_local[control_error_word] = 1;
        } else if (end - run_start >= run_budget_ticks) {
            error_flag = 1;
            failure_code = failure_run_wide_budget;
            failure_elapsed_ticks = end - run_start;
            failure_limit_ticks = run_budget_ticks;
            control_local[control_error_word] = 1;
        }

        cb_reserve_back(cb_timestamp, 1);
        auto* timestamp_page = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            get_write_ptr(cb_timestamp));
        timestamp_page[0] = static_cast<std::uint32_t>(end);
        timestamp_page[1] = static_cast<std::uint32_t>(end >> 32);
        timestamp_page[2] = accumulator;
        timestamp_page[3] = static_cast<std::uint32_t>(elapsed);
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
        payload[free_word] = frames_consumed;
        if (error_flag != 0) {
            break;
        }
        if (control_local[control_done_word] != 0
            && frames_consumed >= control_local[control_produced_word]) {
            break;
        }
        if (frames_consumed >= frame_count) {
            error_flag = 1;
            failure_code = failure_other_check;
            failure_elapsed_ticks = get_timestamp() - run_start;
            failure_limit_ticks = run_budget_ticks;
            control_local[control_error_word] = 1;
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
    summary[7] = failure_code;
    summary[8] = static_cast<std::uint32_t>(failure_elapsed_ticks);
    summary[9] = static_cast<std::uint32_t>(failure_elapsed_ticks >> 32);
    summary[10] = static_cast<std::uint32_t>(failure_limit_ticks);
    summary[11] = static_cast<std::uint32_t>(failure_limit_ticks >> 32);
    summary[12] = failure_code != 0;
    cb_push_back(cb_timestamp, 1);
    cb_wait_front(cb_timestamp, 1);
    noc_async_write_page(0, stats, get_read_ptr(cb_timestamp));
    noc_async_write_barrier();
    cb_pop_front(cb_timestamp, 1);
}
