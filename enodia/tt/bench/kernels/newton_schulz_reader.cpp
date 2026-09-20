// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t tile_bytes = 32 * 32 * 2;
constexpr std::uint32_t cb_matmul_a = 0;
constexpr std::uint32_t cb_matmul_b = 1;
constexpr std::uint32_t cb_product_rr = 2;
constexpr std::uint32_t cb_product_ii = 3;
constexpr std::uint32_t cb_product_ri = 4;
constexpr std::uint32_t cb_product_ir = 5;
constexpr std::uint32_t cb_identity = 6;
constexpr std::uint32_t cb_zero = 7;
constexpr std::uint32_t cb_matmul_product = 16;
constexpr std::uint32_t cb_rx_real = 17;
constexpr std::uint32_t cb_rx_imag = 18;
constexpr std::uint32_t cb_s_real = 19;
constexpr std::uint32_t cb_s_imag = 20;
constexpr std::uint32_t cb_state_real = 21;
constexpr std::uint32_t cb_state_imag = 22;

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile_id, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile_id, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

void copy_tile(std::uint32_t destination, std::uint32_t source_address) {
    cb_reserve_back(destination, 1);
    volatile tt_l1_ptr std::uint32_t* source =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(source_address);
    volatile tt_l1_ptr std::uint32_t* target =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_write_ptr(destination));
    for (std::uint32_t word = 0; word < tile_bytes / sizeof(std::uint32_t); ++word) {
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

void stream_cb_complex(
    std::uint32_t destination,
    std::uint32_t real_cb,
    std::uint32_t imag_cb,
    bool left_operand,
    bool consume) {
    cb_wait_front(real_cb, 1);
    cb_wait_front(imag_cb, 1);
    const std::uint32_t real = get_read_ptr(real_cb);
    const std::uint32_t imag = get_read_ptr(imag_cb);
    if (left_operand) {
        copy_tile(destination, real);
        copy_tile(destination, imag);
        copy_tile(destination, real);
        copy_tile(destination, imag);
    } else {
        copy_tile(destination, real);
        copy_tile(destination, imag);
        copy_tile(destination, imag);
        copy_tile(destination, real);
    }
    if (consume) {
        cb_pop_front(real_cb, 1);
        cb_pop_front(imag_cb, 1);
    }
}

void route_tile(std::uint32_t source, std::uint32_t destination) {
    cb_wait_front(source, 1);
    copy_tile(destination, get_read_ptr(source));
    cb_pop_front(source, 1);
}

void route_complex_products() {
    route_tile(cb_matmul_product, cb_product_rr);
    route_tile(cb_matmul_product, cb_product_ii);
    route_tile(cb_matmul_product, cb_product_ri);
    route_tile(cb_matmul_product, cb_product_ir);
}
}  // namespace

void kernel_main() {
    const std::uint32_t r_real_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t r_imag_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t x_real_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t x_imag_address = get_arg_val<std::uint32_t>(3);
    const std::uint32_t identity_address = get_arg_val<std::uint32_t>(4);
    const std::uint32_t zero_address = get_arg_val<std::uint32_t>(5);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(6);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(7);

    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
    constexpr auto r_real_args = TensorAccessorArgs<1>();
    constexpr auto r_imag_args = TensorAccessorArgs<r_real_args.next_compile_time_args_offset()>();
    constexpr auto x_real_args = TensorAccessorArgs<r_imag_args.next_compile_time_args_offset()>();
    constexpr auto x_imag_args = TensorAccessorArgs<x_real_args.next_compile_time_args_offset()>();
    constexpr auto identity_args = TensorAccessorArgs<x_imag_args.next_compile_time_args_offset()>();
    constexpr auto zero_args = TensorAccessorArgs<identity_args.next_compile_time_args_offset()>();

    const auto r_real = TensorAccessor(r_real_args, r_real_address);
    const auto r_imag = TensorAccessor(r_imag_args, r_imag_address);
    const auto x_real = TensorAccessor(x_real_args, x_real_address);
    const auto x_imag = TensorAccessor(x_imag_args, x_imag_address);
    const auto identity = TensorAccessor(identity_args, identity_address);
    const auto zero = TensorAccessor(zero_args, zero_address);

    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            stream_external_complex(cb_matmul_a, tile, r_real, r_imag, true);
            if (iteration == 0) {
                stream_external_complex(cb_matmul_b, tile, x_real, x_imag, false);
            } else {
                stream_cb_complex(cb_matmul_b, cb_state_real, cb_state_imag, false, false);
            }
            route_complex_products();

            route_tile(cb_rx_real, cb_product_rr);
            route_tile(cb_rx_imag, cb_product_ii);
            read_tile(cb_identity, 0, identity);
            read_tile(cb_zero, 0, zero);

            if (iteration == 0) {
                stream_external_complex(cb_matmul_a, tile, x_real, x_imag, true);
            } else {
                stream_cb_complex(cb_matmul_a, cb_state_real, cb_state_imag, true, true);
            }
            stream_cb_complex(cb_matmul_b, cb_s_real, cb_s_imag, false, true);
            route_complex_products();
        }
    }
}
