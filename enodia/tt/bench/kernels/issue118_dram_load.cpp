// SPDX-License-Identifier: Apache-2.0
// Issue #118 fixed read/write stream over all eight DRAM banks.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t dram_banks = 8;
constexpr std::uint32_t pages_per_bank = 256;
constexpr std::uint32_t page_words = 4096 / sizeof(std::uint32_t);
constexpr std::uint32_t cb_stream = 22;
constexpr std::uint32_t cb_interval = 23;

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
    static_assert(dram_banks == 8, "the DRAM load is fixed to all eight banks");
    const std::uint32_t read_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t write_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t bank_stride = get_arg_val<std::uint32_t>(2);
    const std::uint32_t stats_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t repeat_count = get_arg_val<std::uint32_t>(4);
    const std::uint32_t bank_index = get_arg_val<std::uint32_t>(5);
    const std::uint32_t stats_page = get_arg_val<std::uint32_t>(6);
    constexpr auto read_args = TensorAccessorArgs<0>();
    constexpr auto write_args = TensorAccessorArgs<read_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<write_args.next_compile_time_args_offset()>();
    const auto read = TensorAccessor(read_args, read_address);
    const auto write = TensorAccessor(write_args, write_address);
    const auto stats = TensorAccessor(stats_args, stats_address);
    volatile std::uint32_t checksum = 0;
    std::uint32_t read_pages = 0;
    std::uint32_t write_pages = 0;
    WrapTrackedClock clock;
    clock.initialize();
    const std::uint64_t load_start = clock.read();

    // Each bank worker repeats the complete fixed page stream.  Clock reads
    // occur on every page so the software extension observes 32-bit wraps.
    for (std::uint32_t repetition = 0; repetition < repeat_count; ++repetition) {
        for (std::uint32_t page = 0; page < pages_per_bank; ++page) {
            const std::uint32_t page_index = bank_index * bank_stride + page;
            cb_reserve_back(cb_stream, 1);
            auto* local = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
                get_write_ptr(cb_stream));
            noc_async_read_page(page_index, read, local);
            noc_async_read_barrier();
            checksum ^= local[page_words - 1] ^ page_index ^ repetition;
            local[0] = checksum;
            noc_async_write_page(page_index, write, local);
            noc_async_write_barrier();
            cb_push_back(cb_stream, 1);
            cb_wait_front(cb_stream, 1);
            cb_pop_front(cb_stream, 1);
            ++read_pages;
            ++write_pages;
            clock.read();
        }
    }

    const std::uint64_t load_end = clock.read();
    cb_reserve_back(cb_interval, 1);
    auto* interval = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_write_ptr(cb_interval));
    interval[0] = static_cast<std::uint32_t>(load_start);
    interval[1] = static_cast<std::uint32_t>(load_start >> 32);
    interval[2] = static_cast<std::uint32_t>(load_end);
    interval[3] = static_cast<std::uint32_t>(load_end >> 32);
    interval[4] = read_pages;
    interval[5] = write_pages;
    interval[6] = repeat_count;
    interval[7] = checksum;
    cb_push_back(cb_interval, 1);
    cb_wait_front(cb_interval, 1);
    noc_async_write_page(stats_page, stats, get_read_ptr(cb_interval));
    noc_async_write_barrier();
    cb_pop_front(cb_interval, 1);
}
