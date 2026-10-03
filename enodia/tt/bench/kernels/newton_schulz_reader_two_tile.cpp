// SPDX-License-Identifier: Apache-2.0
// Reader for the opt-in two-tile complex-product path.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

namespace {
constexpr std::uint32_t cb_x0_real = 3;
constexpr std::uint32_t cb_x0_imag = 4;
constexpr std::uint32_t cb_identity = 5;
constexpr std::uint32_t cb_zero = 6;
constexpr std::uint32_t cb_two_tile_r = 20;

// Four signed R pages plus the seed-free K=3 [2I, 0] column.
constexpr std::uint32_t two_tile_r_pages = 6;

template <typename Accessor>
void read_one(std::uint32_t cb, std::uint32_t tile_id, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile_id, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

template <typename RAccessor, typename XAccessor>
void read_matrix_block(
    std::uint32_t tile_id,
    std::uint32_t block_count,
    const RAccessor& r_block,
    const XAccessor& x0_real,
    const XAccessor& x0_imag,
    bool batch_reads) {
    cb_reserve_back(cb_two_tile_r, two_tile_r_pages * block_count);
    cb_reserve_back(cb_x0_real, block_count);
    cb_reserve_back(cb_x0_imag, block_count);

    for (std::uint32_t index = 0; index < block_count; ++index) {
        const std::uint32_t tile = tile_id + index;
        const std::uint32_t r_ptr = get_write_ptr(cb_two_tile_r) +
                                     index * two_tile_r_pages * get_tile_size(cb_two_tile_r);
        if (batch_reads) {
            for (std::uint32_t face = 0; face < two_tile_r_pages; ++face) {
                noc_async_read_page(
                    tile * two_tile_r_pages + face,
                    r_block,
                    r_ptr + face * get_tile_size(cb_two_tile_r));
            }
            noc_async_read_page(
                tile,
                x0_real,
                get_write_ptr(cb_x0_real) + index * get_tile_size(cb_x0_real));
            noc_async_read_page(
                tile,
                x0_imag,
                get_write_ptr(cb_x0_imag) + index * get_tile_size(cb_x0_imag));
        } else {
            for (std::uint32_t face = 0; face < two_tile_r_pages; ++face) {
                noc_async_read_page(
                    tile * two_tile_r_pages + face,
                    r_block,
                    r_ptr + face * get_tile_size(cb_two_tile_r));
                noc_async_read_barrier();
            }
            noc_async_read_page(
                tile,
                x0_real,
                get_write_ptr(cb_x0_real) + index * get_tile_size(cb_x0_real));
            noc_async_read_barrier();
            noc_async_read_page(
                tile,
                x0_imag,
                get_write_ptr(cb_x0_imag) + index * get_tile_size(cb_x0_imag));
            noc_async_read_barrier();
        }
    }
    if (batch_reads) {
        noc_async_read_barrier();
    }
    cb_push_back(cb_two_tile_r, two_tile_r_pages * block_count);
    cb_push_back(cb_x0_real, block_count);
    cb_push_back(cb_x0_imag, block_count);
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
    constexpr bool batch_reads = get_compile_time_arg_val(1) != 0;
    constexpr std::uint32_t matrix_block = get_compile_time_arg_val(2);
    (void)iterations;

    const std::uint32_t r_block_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t x0_real_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t x0_imag_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t identity_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t zero_address = get_arg_val<std::uint32_t>(4);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(5);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(6);

    constexpr auto r_block_args = TensorAccessorArgs<3>();
    constexpr auto x0_real_args = TensorAccessorArgs<r_block_args.next_compile_time_args_offset()>();
    constexpr auto x0_imag_args = TensorAccessorArgs<x0_real_args.next_compile_time_args_offset()>();
    constexpr auto identity_args = TensorAccessorArgs<x0_imag_args.next_compile_time_args_offset()>();
    constexpr auto zero_args = TensorAccessorArgs<identity_args.next_compile_time_args_offset()>();

    const auto r_block = TensorAccessor(r_block_args, r_block_address);
    const auto x0_real = TensorAccessor(x0_real_args, x0_real_address);
    const auto x0_imag = TensorAccessor(x0_imag_args, x0_imag_address);
    const auto identity = TensorAccessor(identity_args, identity_address);
    const auto zero = TensorAccessor(zero_args, zero_address);

    read_one(cb_identity, 0, identity);
    read_one(cb_zero, 0, zero);
    for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
        const std::uint32_t block_count =
            (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
        read_matrix_block(
            start_tile + offset,
            block_count,
            r_block,
            x0_real,
            x0_imag,
            batch_reads);
    }
}
