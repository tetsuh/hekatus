// SPDX-License-Identifier: Apache-2.0
// Reader for the minimal Issue #63 two-tile a/b/c probe.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

namespace {
constexpr std::uint32_t cb_x_real = 3;
constexpr std::uint32_t cb_x_imag = 4;
constexpr std::uint32_t cb_identity = 5;
constexpr std::uint32_t cb_zero = 6;
constexpr std::uint32_t cb_negative_x_imag = 13;
constexpr std::uint32_t cb_two_tile_r = 20;
constexpr std::uint32_t two_tile_r_pages = 4;

template <typename Accessor>
void read_page(
    std::uint32_t cb,
    std::uint32_t tile,
    const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    const std::uint32_t r_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t x_real_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t x_imag_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t negative_x_imag_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t identity_address = get_arg_val<std::uint32_t>(4);
    const std::uint32_t zero_address = get_arg_val<std::uint32_t>(5);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(6);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(7);

    constexpr auto r_args = TensorAccessorArgs<1>();
    constexpr auto x_real_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto x_imag_args = TensorAccessorArgs<x_real_args.next_compile_time_args_offset()>();
    constexpr auto negative_x_imag_args =
        TensorAccessorArgs<x_imag_args.next_compile_time_args_offset()>();
    constexpr auto identity_args =
        TensorAccessorArgs<negative_x_imag_args.next_compile_time_args_offset()>();
    constexpr auto zero_args = TensorAccessorArgs<identity_args.next_compile_time_args_offset()>();

    const auto r = TensorAccessor(r_args, r_address);
    const auto x_real = TensorAccessor(x_real_args, x_real_address);
    const auto x_imag = TensorAccessor(x_imag_args, x_imag_address);
    const auto negative_x_imag = TensorAccessor(negative_x_imag_args, negative_x_imag_address);
    const auto identity = TensorAccessor(identity_args, identity_address);
    const auto zero = TensorAccessor(zero_args, zero_address);

    // Identity and zero remain resident for every matrix in stages b and c.
    read_page(cb_identity, 0, identity);
    read_page(cb_zero, 0, zero);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        cb_reserve_back(cb_two_tile_r, two_tile_r_pages);
        for (std::uint32_t face = 0; face < two_tile_r_pages; ++face) {
            noc_async_read_page(
                tile * two_tile_r_pages + face,
                r,
                get_write_ptr(cb_two_tile_r) + face * get_tile_size(cb_two_tile_r));
            noc_async_read_barrier();
        }
        cb_push_back(cb_two_tile_r, two_tile_r_pages);
        read_page(cb_x_real, tile, x_real);
        read_page(cb_x_imag, tile, x_imag);
        read_page(cb_negative_x_imag, tile, negative_x_imag);
    }
}
