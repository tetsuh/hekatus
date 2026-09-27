// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

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

void matmul_one() {
    cb_wait_front(cb_r, 1);
    cb_wait_front(cb_x_bfloat16, 1);
    cb_reserve_back(cb_product, 1);
    tile_regs_acquire();
    matmul_block(cb_r, cb_x_bfloat16, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_product);
    tile_regs_release();
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

template <
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output>
void subtract_one() {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    // The four-operand overload records current SrcA/SrcB -> next SrcA/SrcB.
    reconfig_data_format(current_srca, left, current_srcb, right);
    // The one-operand overload is forced, including same-format output changes.
    pack_reconfig_data_format(output);
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

template <
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output>
void add_one() {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    // The four-operand overload records current SrcA/SrcB -> next SrcA/SrcB.
    reconfig_data_format(current_srca, left, current_srcb, right);
    // The one-operand overload is forced, including same-format output changes.
    pack_reconfig_data_format(output);
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
}  // namespace

// This stage stops after the first residual S; it does not start the second matmul group.
void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_r, cb_x_bfloat16, cb_product);
    matmul_block_init(cb_r, cb_x_bfloat16, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        complex_matmul_products();
        // Matmul leaves SrcA/SrcB at cb_x_bfloat16/cb_r; switch to CB2/CB3.
        subtract_one<cb_x_bfloat16, cb_r, cb_product_rr, cb_product_ii, cb_rx_real>();
        // RX-real leaves CB2/CB3 active; switch to CB4/CB5.
        add_one<cb_product_rr, cb_product_ii, cb_product_ri, cb_product_ir, cb_rx_imag>();
        // RX-imag leaves CB4/CB5 active; switch to identity/RX-real.
        subtract_one<cb_product_ri, cb_product_ir, cb_identity, cb_rx_real, cb_s_bfloat16_real>();
        // The first residual leaves identity/RX-real active; switch to zero/RX-imag.
        subtract_one<cb_identity, cb_rx_real, cb_zero, cb_rx_imag, cb_s_bfloat16_imag>();
    }
}
