// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr std::uint32_t cb_matmul_a = 0;
constexpr std::uint32_t cb_matmul_b = 1;
constexpr std::uint32_t cb_product_rr = 2;
constexpr std::uint32_t cb_product_ii = 3;
constexpr std::uint32_t cb_product_ri = 4;
constexpr std::uint32_t cb_product_ir = 5;
constexpr std::uint32_t cb_identity = 6;
constexpr std::uint32_t cb_zero = 7;
constexpr std::uint32_t cb_x_real = 17;
constexpr std::uint32_t cb_x_imag = 18;
constexpr std::uint32_t cb_product = 16;
constexpr std::uint32_t cb_s_real = 19;
constexpr std::uint32_t cb_s_imag = 20;
constexpr std::uint32_t cb_out_real = 23;
constexpr std::uint32_t cb_out_imag = 24;

void matmul_one() {
    cb_wait_front(cb_matmul_a, 1);
    cb_wait_front(cb_matmul_b, 1);
    cb_reserve_back(cb_product, 1);
    tile_regs_acquire();
    matmul_block(cb_matmul_a, cb_matmul_b, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_product);
    tile_regs_release();
    cb_push_back(cb_product, 1);
    cb_pop_front(cb_matmul_a, 1);
    cb_pop_front(cb_matmul_b, 1);
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

void complex_matmul() {
    matmul_one();
    matmul_one();
    matmul_one();
    matmul_one();
}

void copy_source_to_operand(std::uint32_t source, std::uint32_t destination, bool consume) {
    cb_wait_front(source, 1);
    cb_reserve_back(destination, 1);
    copy_tile_init(source);
    tile_regs_acquire();
    copy_tile(source, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, destination);
    tile_regs_release();
    cb_push_back(destination, 1);
    if (consume) {
        cb_pop_front(source, 1);
    }
}

void stage_second_operands() {
    copy_source_to_operand(cb_x_real, cb_matmul_a, true);
    copy_source_to_operand(cb_x_imag, cb_matmul_a, true);
    copy_source_to_operand(cb_x_real, cb_matmul_a, true);
    copy_source_to_operand(cb_x_imag, cb_matmul_a, true);
    copy_source_to_operand(cb_s_real, cb_matmul_b, false);
    copy_source_to_operand(cb_s_imag, cb_matmul_b, false);
    copy_source_to_operand(cb_s_imag, cb_matmul_b, false);
    copy_source_to_operand(cb_s_real, cb_matmul_b, false);
    cb_pop_front(cb_s_real, 1);
    cb_pop_front(cb_s_imag, 1);
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_matmul_a, cb_matmul_b, cb_product);
    matmul_block_init(cb_matmul_a, cb_matmul_b, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        complex_matmul();
        subtract_one<cb_identity, cb_product_rr, cb_s_real>();
        subtract_one<cb_zero, cb_product_ii, cb_s_imag>();
        stage_second_operands();
        matmul_block_init(cb_matmul_a, cb_matmul_b, false, 1, 1, 1);
        complex_matmul();
        subtract_one<cb_product_rr, cb_product_ii, cb_out_real>();
        add_one<cb_product_ri, cb_product_ir, cb_out_imag>();
    }
}
