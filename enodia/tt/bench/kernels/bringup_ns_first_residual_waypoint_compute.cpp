// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/debug/waypoint.h"

namespace {
constexpr std::uint32_t cb_r = 0;
constexpr std::uint32_t cb_x_bfloat16 = 1;
constexpr std::uint32_t cb_product_rr = 2;
constexpr std::uint32_t cb_product_ii = 3;
constexpr std::uint32_t cb_product_ri = 4;
constexpr std::uint32_t cb_product_ir = 5;
constexpr std::uint32_t cb_identity = 6;
constexpr std::uint32_t cb_zero = 7;
constexpr std::uint32_t cb_rx_real = 8;
constexpr std::uint32_t cb_rx_imag = 9;
constexpr std::uint32_t cb_s_bfloat16_real = 10;
constexpr std::uint32_t cb_s_bfloat16_imag = 11;
constexpr std::uint32_t cb_product = 16;

// Stage 69 markers are intentionally distinct from built-in CWFW/UABD/etc.:
// M69I matmul init, M69R/M69X matmul waits, M69M matmul, M69P/M69B pack/push
// S69L/S69R subtract waits, S69I/S69T subtract inits, S69P/S69B pack/push
// A69L/A69R add waits, A69I/A69T add inits, A69P/A69B pack/push

void matmul_one() {
    WAYPOINT("M69R");
    cb_wait_front(cb_r, 1);
    WAYPOINT("M69X");
    cb_wait_front(cb_x_bfloat16, 1);
    cb_reserve_back(cb_product, 1);
    tile_regs_acquire();
    WAYPOINT("M69M");
    matmul_block(cb_r, cb_x_bfloat16, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    WAYPOINT("M69P");
    pack_tile(0, cb_product);
    tile_regs_release();
    WAYPOINT("M69B");
    cb_push_back(cb_product, 1);
    cb_pop_front(cb_r, 1);
    cb_pop_front(cb_x_bfloat16, 1);
}

void complex_matmul_products() {
    matmul_one();
    matmul_one();
    matmul_one();
    matmul_one();
}

template <std::uint32_t left, std::uint32_t right, std::uint32_t output>
void subtract_one() {
    WAYPOINT("S69L");
    cb_wait_front(left, 1);
    WAYPOINT("S69R");
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    WAYPOINT("S69I");
    binary_op_init_common(left, right, output);
    WAYPOINT("S69T");
    sub_tiles_init(left, right);
    tile_regs_acquire();
    sub_tiles(left, right, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    WAYPOINT("S69P");
    pack_tile(0, output);
    tile_regs_release();
    WAYPOINT("S69B");
    cb_push_back(output, 1);
    cb_pop_front(left, 1);
    cb_pop_front(right, 1);
}

template <std::uint32_t left, std::uint32_t right, std::uint32_t output>
void add_one() {
    WAYPOINT("A69L");
    cb_wait_front(left, 1);
    WAYPOINT("A69R");
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    WAYPOINT("A69I");
    binary_op_init_common(left, right, output);
    WAYPOINT("A69T");
    add_tiles_init(left, right);
    tile_regs_acquire();
    add_tiles(left, right, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    WAYPOINT("A69P");
    pack_tile(0, output);
    tile_regs_release();
    WAYPOINT("A69B");
    cb_push_back(output, 1);
    cb_pop_front(left, 1);
    cb_pop_front(right, 1);
}
}  // namespace

// This stage stops after the first residual S; it does not start the second matmul group.
void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_r, cb_x_bfloat16, cb_product);
    WAYPOINT("M69I");
    matmul_block_init(cb_r, cb_x_bfloat16, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        complex_matmul_products();
        subtract_one<cb_product_rr, cb_product_ii, cb_rx_real>();
        add_one<cb_product_ri, cb_product_ir, cb_rx_imag>();
        subtract_one<cb_identity, cb_rx_real, cb_s_bfloat16_real>();
        subtract_one<cb_zero, cb_rx_imag, cb_s_bfloat16_imag>();
    }
}
