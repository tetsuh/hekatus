// SPDX-License-Identifier: Apache-2.0
// Issue #12 Stage 1 producer: cadence-paced synthetic int16-complex frames.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_scratch = 0;
constexpr std::uint32_t page_words = 32 * 32;
constexpr std::uint32_t page_bytes = page_words * sizeof(std::uint32_t);
constexpr std::uint32_t control_error_word = 0;
constexpr std::uint32_t control_done_word = 1;
constexpr std::uint32_t control_attempted_word = 2;
constexpr std::uint32_t control_produced_word = 3;
constexpr std::uint32_t control_dropped_word = 4;
constexpr std::uint32_t ready_semaphore_id = 0;
constexpr std::uint32_t free_semaphore_id = 1;
constexpr std::uint32_t done_semaphore_id = 2;
constexpr std::uint32_t error_semaphore_id = 3;
constexpr std::uint32_t failure_run_wide_budget = 1;
constexpr std::uint32_t failure_producer_pacing_wait = 2;
constexpr std::uint32_t failure_other_check = 5;

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
    const std::uint32_t control_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t stats_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t frame_count = get_arg_val<std::uint32_t>(3);
    const std::uint32_t frame_interval_ticks = get_arg_val<std::uint32_t>(4);
    const std::uint32_t ring_pages = get_arg_val<std::uint32_t>(5);
    const std::uint64_t run_budget_ticks =
        static_cast<std::uint64_t>(get_arg_val<std::uint32_t>(6))
        | (static_cast<std::uint64_t>(get_arg_val<std::uint32_t>(7)) << 32);

    constexpr auto ring_args = TensorAccessorArgs<0>();
    constexpr auto control_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<control_args.next_compile_time_args_offset()>();
    const auto ring = TensorAccessor(ring_args, ring_address);
    const auto control = TensorAccessor(control_args, control_address);
    const auto stats = TensorAccessor(stats_args, stats_address);

    const std::uint64_t consumer_noc = control.get_noc_addr(0);
    const std::uint64_t noc_coord_mask = ~((std::uint64_t(1) << NOC_ADDR_COORD_SHIFT) - 1);
    const std::uint64_t ready_noc = (consumer_noc & noc_coord_mask) | get_semaphore(ready_semaphore_id);
    const std::uint64_t done_noc = (consumer_noc & noc_coord_mask) | get_semaphore(done_semaphore_id);
    auto* free_sem = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_semaphore(free_semaphore_id));
    auto* error_sem = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_semaphore(error_semaphore_id));

    std::uint32_t producer_full_count = 0;
    std::uint32_t ready_count = 0;
    std::uint32_t frames_dropped = 0;
    std::uint32_t attempts_started = 0;
    std::uint32_t error_flag = 0;
    std::uint32_t failure_code = 0;
    std::uint64_t failure_elapsed_ticks = 0;
    std::uint64_t failure_limit_ticks = 0;
    WrapTrackedClock clock;
    clock.initialize();
    const std::uint64_t run_start = clock.read();
    std::uint64_t next_release = run_start;

    for (std::uint32_t attempted = 0; attempted < frame_count; ++attempted) {
        invalidate_l1_cache();
        if (*error_sem != 0) {
            error_flag = 1;
            failure_code = failure_other_check;
            failure_elapsed_ticks = clock.read() - run_start;
            failure_limit_ticks = run_budget_ticks;
            break;
        }
        attempts_started = attempted + 1;
        if (attempted != 0) {
            next_release += static_cast<std::uint64_t>(frame_interval_ticks);
        }
        while (static_cast<std::int64_t>(clock.read() - next_release) < 0) {
            invalidate_l1_cache();
            if (*error_sem != 0) {
                error_flag = 1;
                failure_code = failure_other_check;
                failure_elapsed_ticks = clock.read() - run_start;
                failure_limit_ticks = run_budget_ticks;
                break;
            }
            if (clock.read() - run_start >= run_budget_ticks) {
                error_flag = 1;
                failure_code = failure_producer_pacing_wait;
                failure_elapsed_ticks = clock.read() - run_start;
                failure_limit_ticks = run_budget_ticks;
                break;
            }
        }
        if (error_flag != 0) {
            break;
        }

        invalidate_l1_cache();
        if (*error_sem != 0) {
            error_flag = 1;
            failure_code = failure_other_check;
            failure_elapsed_ticks = clock.read() - run_start;
            failure_limit_ticks = run_budget_ticks;
            break;
        }
        const std::uint32_t consumed_required =
            ready_count + 1 > ring_pages ? ready_count + 1 - ring_pages : 0;
        bool slot_full = false;
        if (consumed_required != 0) {
            invalidate_l1_cache();
            if (*error_sem != 0) {
                error_flag = 1;
                failure_code = failure_other_check;
                failure_elapsed_ticks = clock.read() - run_start;
                failure_limit_ticks = run_budget_ticks;
                break;
            }
            slot_full = *free_sem < consumed_required;
            if (clock.read() - run_start >= run_budget_ticks) {
                error_flag = 1;
                failure_code = failure_run_wide_budget;
                failure_elapsed_ticks = clock.read() - run_start;
                failure_limit_ticks = run_budget_ticks;
            }
        }
        if (error_flag != 0) {
            break;
        }
        if (slot_full) {
            // Drop-new policy: cadence advances without waiting or overwriting.
            producer_full_count += 1;
            frames_dropped += 1;
            continue;
        }

        cb_reserve_back(cb_scratch, 1);
        auto* source = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            get_write_ptr(cb_scratch));
        for (std::uint32_t word = 0; word < page_words; ++word) {
            const std::uint16_t i = static_cast<std::uint16_t>((attempted + word) & 0x07FFu);
            const std::uint16_t q = static_cast<std::uint16_t>((3 * attempted + 5 * word) & 0x07FFu);
            source[word] = static_cast<std::uint32_t>(i) | (static_cast<std::uint32_t>(q) << 16);
        }
        cb_push_back(cb_scratch, 1);
        cb_wait_front(cb_scratch, 1);
        noc_async_write_page(ready_count % ring_pages, ring, get_read_ptr(cb_scratch));
        noc_async_write_barrier();
        cb_pop_front(cb_scratch, 1);
        noc_semaphore_inc(ready_noc, 1);
        noc_async_atomic_barrier();
        ready_count += 1;
    }

    cb_reserve_back(cb_scratch, 1);
    auto* control_page = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_write_ptr(cb_scratch));
    control_page[control_error_word] = error_flag;
    control_page[control_done_word] = 1;
    control_page[control_attempted_word] = attempts_started;
    control_page[control_produced_word] = ready_count;
    control_page[control_dropped_word] = frames_dropped;
    cb_push_back(cb_scratch, 1);
    cb_wait_front(cb_scratch, 1);
    noc_async_write_page(0, control, get_read_ptr(cb_scratch));
    noc_async_write_barrier();
    cb_pop_front(cb_scratch, 1);

    cb_reserve_back(cb_scratch, 1);
    auto* summary = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_write_ptr(cb_scratch));
    summary[0] = producer_full_count;
    summary[1] = ready_count;
    summary[2] = error_flag;
    summary[3] = attempts_started;
    summary[4] = frames_dropped;
    summary[5] = 1;
    summary[6] = failure_code;
    summary[7] = static_cast<std::uint32_t>(failure_elapsed_ticks);
    summary[8] = static_cast<std::uint32_t>(failure_elapsed_ticks >> 32);
    summary[9] = static_cast<std::uint32_t>(failure_limit_ticks);
    summary[10] = static_cast<std::uint32_t>(failure_limit_ticks >> 32);
    summary[11] = failure_code != 0;
    cb_push_back(cb_scratch, 1);
    cb_wait_front(cb_scratch, 1);
    noc_async_write_page(0, stats, get_read_ptr(cb_scratch));
    noc_async_write_barrier();
    cb_pop_front(cb_scratch, 1);
    noc_semaphore_inc(done_noc, 1);
    noc_async_atomic_barrier();
}
