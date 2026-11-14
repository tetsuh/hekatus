// SPDX-License-Identifier: Apache-2.0
// Actual fixed-size frames, paced independently of the transfer progress path.
#include "q4_clock.hpp"
#include "q4_runtime_args.hpp"
#include "q4_accessor_diagnostics.hpp"

using namespace issue120_q4;

void kernel_main() {
    const uint32_t ring_address = get_arg_val<uint32_t>(0);
    const uint32_t stats_address = get_arg_val<uint32_t>(1);
    const uint32_t frame_count = get_arg_val<uint32_t>(2);
    const uint32_t frame_interval_ticks = get_arg_val<uint32_t>(3);
    const uint32_t ring_pages = get_arg_val<uint32_t>(4);
    const uint64_t budget = uint64_t(get_arg_val<uint32_t>(5)) |
        (uint64_t(get_arg_val<uint32_t>(6)) << 32);
    constexpr auto ring_args = TensorAccessorArgs<0>();
    constexpr auto stats_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    const auto ring = TensorAccessor(ring_args, ring_address);
    const auto stats = TensorAccessor(stats_args, stats_address);
    const uint32_t peer_x = get_arg_val<uint32_t>(issue120_q4_args::Producer::peer_x);
    const uint32_t peer_y = get_arg_val<uint32_t>(issue120_q4_args::Producer::peer_y);
    const uint64_t ready_target = get_noc_addr(peer_x, peer_y, get_semaphore(ready_id));
    auto* start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(start_id));
    auto* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(free_id));
    Clock clock;
    clock.initialize();
    const uint64_t opened = clock.read();
    uint32_t attempted = 0;
    uint32_t produced = 0;
    uint32_t overflow = 0;
    uint32_t aborted = 0;
    while (*start == 0) {
        invalidate_l1_cache();
        if (clock.read() - opened >= budget) { aborted = 1; break; }
    }
    const uint64_t epoch = clock.read();
    for (uint32_t frame = 0; frame < frame_count && aborted == 0; ++frame) {
        const uint64_t release = epoch + uint64_t(frame + 1) * frame_interval_ticks;
        while (clock.read() < release) {
            if (clock.read() - opened >= budget) { aborted = 1; break; }
        }
        if (aborted != 0) { break; }
        ++attempted;
        invalidate_l1_cache();
        if (produced - *freed >= ring_pages) { ++overflow; continue; }
        cb_reserve_back(0, 1);
        const uint32_t scratch = get_write_ptr(0);
        auto* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t word = 0; word < page_words; ++word) {
            payload[word] = frame_word(frame, word);
        }
        noc_async_write(scratch,
            get_noc_addr(peer_x, peer_y, ring_address + (produced % ring_pages) * page_bytes), page_bytes);
        clock.writes_complete();
        cb_push_back(0, 1);
        cb_wait_front(0, 1);
        cb_pop_front(0, 1);
        noc_semaphore_inc(ready_target, 1);
        clock.atomics_complete();
        ++produced;
    }
    const uint32_t exit_start_raw = *start;
    const uint64_t exited = clock.read();
    cb_reserve_back(0, 1);
    const uint32_t scratch = get_write_ptr(0);
    auto* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t word = 0; word < page_words; ++word) { result[word] = 0; }
    result[0] = attempted;
    result[1] = produced;
    result[2] = overflow;
    result[3] = aborted;
    result[4] = clock.ambiguous;
    result[5] = exit_start_raw;
    result[6] = uint32_t(get_semaphore(start_id));
    result[7] = uint32_t(opened); result[8] = uint32_t(opened >> 32);
    result[9] = uint32_t(exited); result[10] = uint32_t(exited >> 32);
    result[11] = uint32_t(get_semaphore(0));
    // Snapshot only: accessor coordinates never determine a peer destination.
    constexpr auto coordinate_offset = issue120_q4_args::producer_coordinate_stats;
    const uint64_t accessor_raw = ring.get_noc_addr(0);
    result[coordinate_offset] = accessor_bank_slot<decltype(ring_args)>();
    result[coordinate_offset + 1] = accessor_bank_word<decltype(ring_args)>();
    result[coordinate_offset + 2] = uint32_t(accessor_raw);
    result[coordinate_offset + 3] = uint32_t(accessor_raw >> 32);
    result[coordinate_offset + 4] = my_x[0]; result[coordinate_offset + 5] = my_y[0];
    result[coordinate_offset + 6] = peer_x; result[coordinate_offset + 7] = peer_y;
    result[coordinate_offset + 8] = uint32_t(ready_target);
    result[coordinate_offset + 9] = uint32_t(ready_target >> 32);
    result[coordinate_offset + 10] = get_compile_time_arg_val(0);
    result[coordinate_offset + 11] = decltype(ring_args)::is_sharded;
    noc_async_write_page(0, stats, scratch);
    clock.writes_complete();
    cb_push_back(0, 1);
    cb_wait_front(0, 1);
    cb_pop_front(0, 1);
    noc_semaphore_inc(get_noc_addr(peer_x, peer_y, get_semaphore(done_id)), 1);
    clock.atomics_complete();
}
