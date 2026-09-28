// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

namespace {
constexpr std::uint32_t cb_r = 0;
constexpr std::uint32_t cb_x = 17;
constexpr std::uint32_t cb_s = 18;
constexpr std::uint32_t cb_product_rr = 2;
constexpr std::uint32_t cb_product_ii = 3;
constexpr std::uint32_t cb_product_ri = 4;
constexpr std::uint32_t cb_product_ir = 5;
constexpr std::uint32_t cb_identity = 6;
constexpr std::uint32_t cb_zero = 7;
constexpr std::uint32_t cb_product = 16;
constexpr std::uint32_t cb_rx_real = 8;
constexpr std::uint32_t cb_rx_imag = 9;
constexpr std::uint32_t cb_s_real = 19;
constexpr std::uint32_t cb_s_imag = 20;
constexpr std::uint32_t cb_state_real = 21;
constexpr std::uint32_t cb_state_imag = 22;
constexpr std::uint32_t cb_out_real = 23;
constexpr std::uint32_t cb_out_imag = 24;

void matmul_one(std::uint32_t left, std::uint32_t right) {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(cb_product, 1);
    tile_regs_acquire();
    matmul_block(left, right, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_product);
    tile_regs_release();
    cb_push_back(cb_product, 1);
    cb_pop_front(left, 1);
    cb_pop_front(right, 1);
}

void complex_matmul_products(std::uint32_t left, std::uint32_t right) {
    matmul_one(left, right);
    matmul_one(left, right);
    matmul_one(left, right);
    matmul_one(left, right);
}

template <std::uint32_t left, std::uint32_t right, std::uint32_t output>
void subtract_one() {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    binary_op_init_common(left, right, output);
    sub_tiles_init(left, right);
    tile_regs_acquire();
    sub_tiles(left, right, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output);
    tile_regs_release();
    cb_push_back(output, 1);
    cb_pop_front(left, 1);
    cb_pop_front(right, 1);
}

template <std::uint32_t left, std::uint32_t right, std::uint32_t output>
void add_one() {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    binary_op_init_common(left, right, output);
    add_tiles_init(left, right);
    tile_regs_acquire();
    add_tiles(left, right, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output);
    tile_regs_release();
    cb_push_back(output, 1);
    cb_pop_front(left, 1);
    cb_pop_front(right, 1);
}

void switch_to_float32_left() {
    reconfig_data_format_srcb(cb_r, cb_x);
    matmul_block_init(cb_x, cb_s, false, 1, 1, 1);
}

void switch_to_bfloat16_left() {
    reconfig_data_format_srcb(cb_x, cb_r);
    matmul_block_init(cb_r, cb_x, false, 1, 1, 1);
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_r, cb_x, cb_product);
    matmul_block_init(cb_r, cb_x, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        for (std::uint32_t iteration = 0; iteration < 8; ++iteration) {
            if (iteration != 0) {
                switch_to_bfloat16_left();
            }
            complex_matmul_products(cb_r, cb_x);
            subtract_one<cb_product_rr, cb_product_ii, cb_rx_real>();
            add_one<cb_product_ri, cb_product_ir, cb_rx_imag>();
            subtract_one<cb_identity, cb_rx_real, cb_s_real>();
            subtract_one<cb_zero, cb_rx_imag, cb_s_imag>();
            switch_to_float32_left();
            complex_matmul_products(cb_x, cb_s);
            if (iteration + 1 == 8) {
                subtract_one<cb_product_rr, cb_product_ii, cb_out_real>();
                add_one<cb_product_ri, cb_product_ir, cb_out_imag>();
            } else {
                subtract_one<cb_product_rr, cb_product_ii, cb_state_real>();
                add_one<cb_product_ri, cb_product_ir, cb_state_imag>();
            }
        }
    }
}
