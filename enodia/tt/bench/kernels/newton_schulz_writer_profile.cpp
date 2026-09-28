// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_output_real = 15;
constexpr std::uint32_t cb_output_imag = 16;
constexpr std::uint32_t cb_profile_reader = 17;
constexpr std::uint32_t cb_profile_compute = 18;
constexpr std::uint32_t cb_profile_writer = 19;
constexpr std::uint32_t profile_magic = 0x5052464C;
constexpr std::uint32_t profile_words = 32 * 32;
constexpr std::uint32_t profile_total_offset = 0;
constexpr std::uint32_t profile_write_offset = 1;
constexpr std::uint32_t profile_count_offset = 2;
constexpr std::uint32_t profile_ready_offset = 31;
constexpr std::uint32_t profile_slot_stride = 32;

void fill_profile_writer(std::uint32_t cycles, std::uint32_t count) {
    cb_reserve_back(cb_profile_writer, 1);
    volatile tt_l1_ptr std::uint32_t* profile =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_write_ptr(cb_profile_writer));
    for (std::uint32_t index = 0; index < profile_words; ++index) {
        profile[index] = 0;
    }
    profile[profile_total_offset] = cycles;
    profile[profile_write_offset] = cycles;
    profile[profile_count_offset] = count;
    profile[profile_ready_offset] = profile_magic;
    cb_push_back(cb_profile_writer, 1);
}
}  // namespace

void kernel_main() {
    const std::uint32_t real_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t imag_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t profile_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t profile_index = get_arg_val<std::uint32_t>(3);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(4);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(5);

    constexpr auto real_args = TensorAccessorArgs<0>();
    constexpr auto imag_args = TensorAccessorArgs<real_args.next_compile_time_args_offset()>();
    constexpr auto profile_args = TensorAccessorArgs<imag_args.next_compile_time_args_offset()>();
    const auto real = TensorAccessor(real_args, real_address);
    const auto imag = TensorAccessor(imag_args, imag_address);
    const auto profile = TensorAccessor(profile_args, profile_address);
    // Core 0 alone consumes the profile CB triplet and writes the three
    // uint32 pages to DRAM; nonzero cores only drain normal output CBs.
    const bool measure_core = start_tile == 0;
    const std::uint32_t write_start = measure_core ? get_timestamp_32b() : 0;
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        cb_wait_front(cb_output_real, 1);
        cb_wait_front(cb_output_imag, 1);
        noc_async_write_page(tile, real, get_read_ptr(cb_output_real));
        noc_async_write_barrier();
        noc_async_write_page(tile, imag, get_read_ptr(cb_output_imag));
        noc_async_write_barrier();
        cb_pop_front(cb_output_real, 1);
        cb_pop_front(cb_output_imag, 1);
    }
    const std::uint32_t write_cycles = measure_core ? get_timestamp_32b() - write_start : 0;
    if (!measure_core) {
        return;
    }

    cb_wait_front(cb_profile_reader, 1);
    cb_wait_front(cb_profile_compute, 1);
    fill_profile_writer(write_cycles, tile_count);
    cb_wait_front(cb_profile_writer, 1);
    noc_async_write_page(profile_index, profile, get_read_ptr(cb_profile_reader));
    noc_async_write_barrier();
    noc_async_write_page(profile_index + 1, profile, get_read_ptr(cb_profile_compute));
    noc_async_write_barrier();
    noc_async_write_page(profile_index + 2, profile, get_read_ptr(cb_profile_writer));
    noc_async_write_barrier();
    // Clear completion flags after the DRAM copy so the next launch cannot
    // mistake the previous triplet for a newly completed compute page.
    volatile tt_l1_ptr std::uint32_t* compute_profile = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_read_ptr(cb_profile_compute));
    compute_profile[profile_ready_offset] = 0;
    compute_profile[profile_slot_stride + profile_ready_offset] = 0;
    compute_profile[2 * profile_slot_stride + profile_ready_offset] = 0;
    cb_pop_front(cb_profile_reader, 1);
    cb_pop_front(cb_profile_compute, 1);
    cb_pop_front(cb_profile_writer, 1);
}
