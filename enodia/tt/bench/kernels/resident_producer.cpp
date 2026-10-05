// SPDX-License-Identifier: Apache-2.0
// Issue #12 Stage 1 producer: synthetic int16-complex pages into a remote L1 ring.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_scratch = 0;
constexpr std::uint32_t page_words = 32 * 32;
constexpr std::uint32_t page_bytes = page_words * sizeof(std::uint32_t);
}

void kernel_main() {
    const std::uint32_t ring_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t stats_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t consumer_x = get_arg_val<std::uint32_t>(2);
    const std::uint32_t consumer_y = get_arg_val<std::uint32_t>(3);
    const std::uint32_t ready_semaphore_id = get_arg_val<std::uint32_t>(4);
    const std::uint32_t free_semaphore_id = get_arg_val<std::uint32_t>(5);
    const std::uint32_t error_semaphore_id = get_arg_val<std::uint32_t>(6);
    const std::uint32_t frame_count = get_arg_val<std::uint32_t>(7);
    const std::uint32_t frame_interval_ticks = get_arg_val<std::uint32_t>(8);
    const std::uint32_t ring_pages = get_arg_val<std::uint32_t>(9);

    constexpr auto ring_args = TensorAccessorArgs<0>();
    constexpr auto stats_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    const auto ring = TensorAccessor(ring_args, ring_address);
    const auto stats = TensorAccessor(stats_args, stats_address);

    auto* free_semaphore = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_semaphore(free_semaphore_id));
    auto* error_semaphore = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_semaphore(error_semaphore_id));
    const std::uint64_t ready_noc_address = get_noc_addr(
        consumer_x,
        consumer_y,
        get_semaphore(ready_semaphore_id));

    std::uint32_t producer_full_count = 0;
    std::uint32_t frames_produced = 0;
    std::uint32_t error_flag = 0;
    std::uint64_t next_release = get_timestamp();

    for (std::uint32_t frame = 0; frame < frame_count; ++frame) {
        if (frame != 0) {
            next_release += static_cast<std::uint64_t>(frame_interval_ticks);
        }
        while (static_cast<std::int64_t>(get_timestamp() - next_release) < 0) {
            if (*error_semaphore != 0) {
                error_flag = 1;
                break;
            }
        }
        if (error_flag != 0) {
            break;
        }

        const std::uint32_t consumed_required =
            frame + 1 > ring_pages ? frame + 1 - ring_pages : 0;
        if (consumed_required != 0 && *free_semaphore < consumed_required) {
            producer_full_count += 1;
        }
        while (consumed_required != 0 && *free_semaphore < consumed_required) {
            invalidate_l1_cache();
            if (*error_semaphore != 0) {
                error_flag = 1;
                break;
            }
        }
        if (error_flag != 0) {
            break;
        }

        cb_reserve_back(cb_scratch, 1);
        auto* source = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
            get_write_ptr(cb_scratch));
        for (std::uint32_t word = 0; word < page_words; ++word) {
            // Two packed int16 values (I in the low half, Q in the high half).
            const std::uint16_t i = static_cast<std::uint16_t>((frame + word) & 0x07FFu);
            const std::uint16_t q = static_cast<std::uint16_t>((3 * frame + 5 * word) & 0x07FFu);
            source[word] = static_cast<std::uint32_t>(i) | (static_cast<std::uint32_t>(q) << 16);
        }
        cb_push_back(cb_scratch, 1);
        cb_wait_front(cb_scratch, 1);
        noc_async_write_page(frame % ring_pages, ring, get_read_ptr(cb_scratch));
        noc_async_write_barrier();
        cb_pop_front(cb_scratch, 1);

        // The payload barrier above precedes this release semaphore update.
        noc_inline_dw_write<InlineWriteDst::L1>(ready_noc_address, frame + 1);
        frames_produced = frame + 1;
    }

    cb_reserve_back(cb_scratch, 1);
    auto* summary = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_write_ptr(cb_scratch));
    summary[0] = producer_full_count;
    summary[1] = frames_produced;
    summary[2] = error_flag;
    summary[3] = *error_semaphore;
    cb_push_back(cb_scratch, 1);
    cb_wait_front(cb_scratch, 1);
    noc_async_write_page(0, stats, get_read_ptr(cb_scratch));
    noc_async_write_barrier();
    cb_pop_front(cb_scratch, 1);
}
