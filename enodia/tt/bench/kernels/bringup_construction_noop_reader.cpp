// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The host probe supplies the active CB index set.  This source deliberately
// performs no reads so it can be used as the construction-only control.
void kernel_main() {
    const std::uint32_t r_real_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t x_real_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t r_imag_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t x_imag_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t identity_address = get_arg_val<std::uint32_t>(4);
    const std::uint32_t zero_address = get_arg_val<std::uint32_t>(5);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(6);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(7);
    constexpr auto r_real_args = TensorAccessorArgs<0>();
    constexpr auto x_real_args = TensorAccessorArgs<r_real_args.next_compile_time_args_offset()>();
    constexpr auto r_imag_args = TensorAccessorArgs<x_real_args.next_compile_time_args_offset()>();
    constexpr auto x_imag_args = TensorAccessorArgs<r_imag_args.next_compile_time_args_offset()>();
    constexpr auto identity_args = TensorAccessorArgs<x_imag_args.next_compile_time_args_offset()>();
    constexpr auto zero_args = TensorAccessorArgs<identity_args.next_compile_time_args_offset()>();
    const auto r_real = TensorAccessor(r_real_args, r_real_address);
    const auto x_real = TensorAccessor(x_real_args, x_real_address);
    const auto r_imag = TensorAccessor(r_imag_args, r_imag_address);
    const auto x_imag = TensorAccessor(x_imag_args, x_imag_address);
    const auto identity = TensorAccessor(identity_args, identity_address);
    const auto zero = TensorAccessor(zero_args, zero_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        (void)tile;
        (void)r_real;
        (void)x_real;
        (void)r_imag;
        (void)x_imag;
        (void)identity;
        (void)zero;
    }
}
