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

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile_id, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile_id, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
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

    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
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

    // Identity and zero are immutable per-core inputs.  They remain at the
    // front of their queues for every matrix and every residual operation.
    read_tile(cb_identity, 0, identity);
    read_tile(cb_zero, 0, zero);

    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        read_tile(cb_r_real, tile, r_real);
        read_tile(cb_r_negative_imag, tile, r_negative_imag);
        read_tile(cb_r_imag, tile, r_imag);
        read_tile(cb_x0_real, tile, x0_real);
        read_tile(cb_x0_imag, tile, x0_imag);

        // The compute kernel holds the three R pages until all fixed
        // iterations for this matrix are complete.  The reader only streams
        // external inputs; it never routes products or state.
    }
    (void)iterations;
}
