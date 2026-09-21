// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_matmul_a = 0;
constexpr std::uint32_t cb_matmul_b = 1;
constexpr std::uint32_t cb_product_rr = 2;
constexpr std::uint32_t cb_product_ii = 3;
constexpr std::uint32_t cb_product_ri = 4;
constexpr std::uint32_t cb_product_ir = 5;
constexpr std::uint32_t cb_product = 16;

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

void copy_tile(std::uint32_t destination, std::uint32_t source_address) {
    cb_reserve_back(destination, 1);
    volatile tt_l1_ptr std::uint32_t* source =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(source_address);
    volatile tt_l1_ptr std::uint32_t* target =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_write_ptr(destination));
    for (std::uint32_t word = 0; word < (32 * 32 * 2) / sizeof(std::uint32_t); ++word) {
        target[word] = source[word];
    }
    cb_push_back(destination, 1);
}

void route_product(std::uint32_t destination) {
    cb_wait_front(cb_product, 1);
    copy_tile(destination, get_read_ptr(cb_product));
    cb_pop_front(cb_product, 1);
}

void route_complex_products() {
    route_product(cb_product_rr);
    route_product(cb_product_ii);
    route_product(cb_product_ri);
    route_product(cb_product_ir);
}

template <typename RealAccessor, typename ImagAccessor>
void stream_complex_pair(
    std::uint32_t tile,
    const RealAccessor& a_real,
    const ImagAccessor& a_imag,
    const RealAccessor& b_real,
    const ImagAccessor& b_imag) {
    read_tile(cb_matmul_a, tile, a_real);
    read_tile(cb_matmul_b, tile, b_real);
    read_tile(cb_matmul_a, tile, a_imag);
    read_tile(cb_matmul_b, tile, b_imag);
    read_tile(cb_matmul_a, tile, a_real);
    read_tile(cb_matmul_b, tile, b_imag);
    read_tile(cb_matmul_a, tile, a_imag);
    read_tile(cb_matmul_b, tile, b_real);
}
}  // namespace

void kernel_main() {
    const std::uint32_t a_real_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t b_real_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t a_imag_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t b_imag_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(6);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(7);
    constexpr auto a_real_args = TensorAccessorArgs<0>();
    constexpr auto b_real_args = TensorAccessorArgs<a_real_args.next_compile_time_args_offset()>();
    constexpr auto a_imag_args = TensorAccessorArgs<b_real_args.next_compile_time_args_offset()>();
    constexpr auto b_imag_args = TensorAccessorArgs<a_imag_args.next_compile_time_args_offset()>();
    const auto a_real = TensorAccessor(a_real_args, a_real_address);
    const auto b_real = TensorAccessor(b_real_args, b_real_address);
    const auto a_imag = TensorAccessor(a_imag_args, a_imag_address);
    const auto b_imag = TensorAccessor(b_imag_args, b_imag_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        stream_complex_pair(tile, a_real, a_imag, b_real, b_imag);
        route_complex_products();
        stream_complex_pair(tile, a_real, a_imag, b_real, b_imag);
        route_complex_products();
    }
}
