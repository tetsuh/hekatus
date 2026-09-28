// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_r_real = 0;
constexpr std::uint32_t cb_r_negative_imag = 1;
constexpr std::uint32_t cb_r_imag = 2;
constexpr std::uint32_t cb_x0_real = 3;
constexpr std::uint32_t cb_x0_imag = 4;
constexpr std::uint32_t cb_identity = 5;
constexpr std::uint32_t cb_zero = 6;
constexpr std::uint32_t cb_profile_reader = 17;
constexpr std::uint32_t profile_magic = 0x5052464C;
constexpr std::uint32_t profile_total_offset = 0;
constexpr std::uint32_t profile_read_offset = 1;
constexpr std::uint32_t profile_constant_offset = 2;
constexpr std::uint32_t profile_count_offset = 3;
constexpr std::uint32_t profile_ready_offset = 31;
constexpr std::uint32_t profile_words = 32 * 32;

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile_id, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile_id, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

void write_profile(std::uint32_t read_cycles, std::uint32_t constant_cycles, std::uint32_t count) {
    cb_reserve_back(cb_profile_reader, 1);
    volatile tt_l1_ptr std::uint32_t* profile =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_write_ptr(cb_profile_reader));
    for (std::uint32_t index = 0; index < profile_words; ++index) {
        profile[index] = 0;
    }
    profile[profile_total_offset] = read_cycles + constant_cycles;
    profile[profile_read_offset] = read_cycles;
    profile[profile_constant_offset] = constant_cycles;
    profile[profile_count_offset] = count;
    profile[profile_ready_offset] = profile_magic;
    cb_push_back(cb_profile_reader, 1);
}
}  // namespace

void kernel_main() {
    const std::uint32_t r_real_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t r_negative_imag_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t r_imag_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t x0_real_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t x0_imag_address = get_arg_val<std::uint32_t>(4);
    const std::uint32_t identity_address = get_arg_val<std::uint32_t>(5);
    const std::uint32_t zero_address = get_arg_val<std::uint32_t>(6);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(7);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(8);

    constexpr auto r_real_args = TensorAccessorArgs<1>();
    constexpr auto r_negative_imag_args =
        TensorAccessorArgs<r_real_args.next_compile_time_args_offset()>();
    constexpr auto r_imag_args =
        TensorAccessorArgs<r_negative_imag_args.next_compile_time_args_offset()>();
    constexpr auto x0_real_args =
        TensorAccessorArgs<r_imag_args.next_compile_time_args_offset()>();
    constexpr auto x0_imag_args =
        TensorAccessorArgs<x0_real_args.next_compile_time_args_offset()>();
    constexpr auto identity_args =
        TensorAccessorArgs<x0_imag_args.next_compile_time_args_offset()>();
    constexpr auto zero_args =
        TensorAccessorArgs<identity_args.next_compile_time_args_offset()>();

    const auto r_real = TensorAccessor(r_real_args, r_real_address);
    const auto r_negative_imag = TensorAccessor(r_negative_imag_args, r_negative_imag_address);
    const auto r_imag = TensorAccessor(r_imag_args, r_imag_address);
    const auto x0_real = TensorAccessor(x0_real_args, x0_real_address);
    const auto x0_imag = TensorAccessor(x0_imag_args, x0_imag_address);
    const auto identity = TensorAccessor(identity_args, identity_address);
    const auto zero = TensorAccessor(zero_args, zero_address);

    // Only core 0 owns the three-page profile transport; other cores stream
    // their assigned tiles without sampling or touching the profile CB.
    const bool measure_core = start_tile == 0;
    std::uint32_t constant_cycles = 0;
    if (measure_core) {
        const std::uint32_t constant_start = get_timestamp_32b();
        read_tile(cb_identity, 0, identity);
        read_tile(cb_zero, 0, zero);
        constant_cycles = get_timestamp_32b() - constant_start;
    } else {
        read_tile(cb_identity, 0, identity);
        read_tile(cb_zero, 0, zero);
    }

    std::uint32_t read_cycles = 0;
    const std::uint32_t read_start = measure_core ? get_timestamp_32b() : 0;
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        read_tile(cb_r_real, tile, r_real);
        read_tile(cb_r_negative_imag, tile, r_negative_imag);
        read_tile(cb_r_imag, tile, r_imag);
        read_tile(cb_x0_real, tile, x0_real);
        read_tile(cb_x0_imag, tile, x0_imag);
    }
    if (measure_core) {
        read_cycles = get_timestamp_32b() - read_start;
        write_profile(read_cycles, constant_cycles, tile_count);
    }
}
