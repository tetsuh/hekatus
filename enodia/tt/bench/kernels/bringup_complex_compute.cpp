// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"

namespace {
constexpr std::uint32_t cb_operand_a = 0;
constexpr std::uint32_t cb_operand_b = 1;
constexpr std::uint32_t cb_rr = 2;
constexpr std::uint32_t cb_ii = 3;
constexpr std::uint32_t cb_ri = 4;
constexpr std::uint32_t cb_ir = 5;
constexpr std::uint32_t cb_out_real = 23;
constexpr std::uint32_t cb_out_imag = 24;
constexpr std::uint32_t cb_product = 16;

void matmul_one() {
    cb_wait_front(cb_operand_a, 1);
    cb_wait_front(cb_operand_b, 1);
    cb_reserve_back(cb_product, 1);
    tile_regs_acquire();
    matmul_block(cb_operand_a, cb_operand_b, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_product);
    tile_regs_release();
    cb_push_back(cb_product, 1);
    cb_pop_front(cb_operand_a, 1);
    cb_pop_front(cb_operand_b, 1);
}

template <std::uint32_t left, std::uint32_t right, std::uint32_t output>
void binary_one(bool add) {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    binary_op_init_common(left, right, output);
    if (add) {
        add_tiles_init(left, right);
    } else {
        sub_tiles_init(left, right);
    }
    tile_regs_acquire();
    if (add) {
        add_tiles(left, right, 0, 0, 0);
    } else {
        sub_tiles(left, right, 0, 0, 0);
    }
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
    binary_one<cb_rr, cb_ii, cb_out_real>(false);
    binary_one<cb_ri, cb_ir, cb_out_imag>(true);
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    mm_block_init(cb_operand_a, cb_operand_b, cb_product, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        complex_matmul();
    }
}
