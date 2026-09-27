// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_r = 0;
constexpr std::uint32_t cb_x_bfloat16 = 1;
constexpr std::uint32_t cb_product_rr = 2;
constexpr std::uint32_t cb_product_ii = 3;
constexpr std::uint32_t cb_product_ri = 4;
constexpr std::uint32_t cb_product_ir = 5;
constexpr std::uint32_t cb_identity = 6;
constexpr std::uint32_t cb_zero = 7;
constexpr std::uint32_t cb_s_float32_real = 10;
constexpr std::uint32_t cb_s_float32_imag = 11;
constexpr std::uint32_t cb_diagnostic_real = 12;
constexpr std::uint32_t cb_diagnostic_imag = 13;
constexpr std::uint32_t cb_product = 16;

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

void copy_float32_tile(std::uint32_t destination, std::uint32_t source_address) {
    cb_reserve_back(destination, 1);
    volatile tt_l1_ptr std::uint32_t* source =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(source_address);
    volatile tt_l1_ptr std::uint32_t* target =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_write_ptr(destination));
    for (std::uint32_t word = 0; word < (32 * 32 * 4) / sizeof(std::uint32_t); ++word) {
        target[word] = source[word];
    }
    cb_push_back(destination, 1);
}

template <typename RealAccessor, typename ImagAccessor>
void stream_external_complex(
    std::uint32_t destination,
    std::uint32_t tile,
    const RealAccessor& real,
    const ImagAccessor& imag,
    bool left_operand) {
    if (left_operand) {
        read_tile(destination, tile, real);
        read_tile(destination, tile, imag);
        read_tile(destination, tile, real);
        read_tile(destination, tile, imag);
    } else {
        read_tile(destination, tile, real);
        read_tile(destination, tile, imag);
        read_tile(destination, tile, imag);
        read_tile(destination, tile, real);
    }
}

void route_product(std::uint32_t destination) {
    cb_wait_front(cb_product, 1);
    copy_float32_tile(destination, get_read_ptr(cb_product));
    cb_pop_front(cb_product, 1);
}

void route_complex_products() {
    route_product(cb_product_rr);
    route_product(cb_product_ii);
    route_product(cb_product_ri);
    route_product(cb_product_ir);
}

void drain_s_to_diagnostic(std::uint32_t source, std::uint32_t destination) {
    cb_wait_front(source, 1);
    copy_float32_tile(destination, get_read_ptr(source));
    cb_pop_front(source, 1);
}
}  // namespace

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
        stream_external_complex(cb_r, tile, r_real, r_imag, true);
        stream_external_complex(cb_x_bfloat16, tile, x_real, x_imag, false);
        route_complex_products();
        read_tile(cb_identity, 0, identity);
        read_tile(cb_zero, 0, zero);
        stream_external_complex(cb_x_bfloat16, tile, x_real, x_imag, true);
        drain_s_to_diagnostic(cb_s_float32_real, cb_diagnostic_real);
        drain_s_to_diagnostic(cb_s_float32_imag, cb_diagnostic_imag);
    }
}
