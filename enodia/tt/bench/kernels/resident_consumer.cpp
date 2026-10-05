// SPDX-License-Identifier: Apache-2.0
// Issue #12 Stage 1 consumer: fixed work, one designated-core timestamp stream.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_timestamp = 1;
constexpr std::uint32_t page_words = 32 * 32;
constexpr std::uint32_t ready_word = page_words - 3;
constexpr std::uint32_t free_word = page_words - 2;
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
    const std::uint64_t run_start = get_timestamp();

    for (std::uint32_t frame = 0; frame < frame_count; ++frame) {
        const std::uint32_t required = frame + 1;
        auto* payload = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            ring_address + (frame % ring_pages) * page_words * sizeof(std::uint32_t));
        if (payload[ready_word] < required) {
            consumer_empty_count += 1;
        }
        while (payload[ready_word] < required && control_local[0] == 0) {
            invalidate_l1_cache();
            if (get_timestamp() - run_start >= run_budget_ticks) {
                error_flag = 1;
                control_local[0] = 1;
                break;
            }
        }
        if (error_flag != 0 || control_local[0] != 0) {
            error_flag = 1;
            break;
        }

        const std::uint64_t start = get_timestamp();
        for (std::uint32_t work = 0; work < work_per_frame; ++work) {
            const std::uint32_t word = work % ready_word;
            accumulator = (accumulator * 33u) ^ payload[word] ^ (work + frame);
        }
        const std::uint64_t end = get_timestamp();
        const std::uint64_t elapsed = end - start;
        if (elapsed >= static_cast<std::uint64_t>(per_frame_work_budget_ticks)
            || end - run_start >= run_budget_ticks) {
            error_flag = 1;
            control_local[0] = 1;
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
        noc_async_write_page(frame, timestamps, get_read_ptr(cb_timestamp));
        noc_async_write_barrier();
        cb_pop_front(cb_timestamp, 1);

        frames_consumed = frame + 1;
        payload[free_word] = frame + 1;
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
    cb_push_back(cb_timestamp, 1);
    cb_wait_front(cb_timestamp, 1);
    noc_async_write_page(0, stats, get_read_ptr(cb_timestamp));
    noc_async_write_barrier();
    cb_pop_front(cb_timestamp, 1);
}
