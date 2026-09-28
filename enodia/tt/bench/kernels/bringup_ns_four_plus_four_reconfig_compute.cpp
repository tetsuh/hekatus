// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr std::uint32_t split_iteration = 4;
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
constexpr std::uint32_t cb_s_bfloat16_operand = 12;
constexpr std::uint32_t cb_product = 16;
constexpr std::uint32_t cb_x_float32 = 17;
constexpr std::uint32_t cb_s_float32_real = 18;
constexpr std::uint32_t cb_s_float32_imag = 19;
constexpr std::uint32_t cb_s_float32_operand = 22;
constexpr std::uint32_t cb_state_bfloat16_real = 20;
constexpr std::uint32_t cb_state_bfloat16_imag = 21;
constexpr std::uint32_t cb_state_float32_intermediate_real = 14;
constexpr std::uint32_t cb_state_float32_intermediate_imag = 15;
constexpr std::uint32_t cb_state_float32_real = 23;
constexpr std::uint32_t cb_state_float32_imag = 24;

void matmul_one(std::uint32_t srcb, std::uint32_t srca) {
    cb_wait_front(srcb, 1);
    cb_wait_front(srca, 1);
    cb_reserve_back(cb_product, 1);
    tile_regs_acquire();
    matmul_block(srcb, srca, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_product);
    tile_regs_release();
    cb_push_back(cb_product, 1);
    cb_pop_front(srcb, 1);
    cb_pop_front(srca, 1);
}

void complex_matmul_products(std::uint32_t srcb, std::uint32_t srca) {
    matmul_one(srcb, srca);
    matmul_one(srcb, srca);
    matmul_one(srcb, srca);
    matmul_one(srcb, srca);
}

// The template arguments record the unpacker and packer state left by the
// preceding operation.  The short binary init is intentionally the only
// operation init here; full destination/synchronization setup is not repeated.
template <
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t current_pack,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output>
void subtract_one() {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    reconfig_data_format(current_srca, left, current_srcb, right);
    // The forced overload records every new output CB, including same-format transitions.
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
    std::uint32_t current_pack,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output>
void add_one() {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    reconfig_data_format(current_srca, left, current_srcb, right);
    // The forced overload records every new output CB, including same-format transitions.
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

void convert_state_to_float32() {
    cb_wait_front(cb_state_bfloat16_real, 1);
    cb_wait_front(cb_state_bfloat16_imag, 1);

    // The preceding BF16 state-imag add left SrcA=CB4 and SrcB=CB5.  copy_tile_init
    // does not change unpack formats, so each BF16 state source is explicit.
    cb_reserve_back(cb_state_float32_intermediate_real, 1);
    reconfig_data_format_srca(cb_product_ri, cb_state_bfloat16_real);
    copy_tile_init(cb_state_bfloat16_real);
    // The current pack CB is BF16 state-imag CB21; the destination is Float32 CB14.
    pack_reconfig_data_format(cb_state_bfloat16_imag, cb_state_float32_intermediate_real);
    tile_regs_acquire();
    copy_tile(cb_state_bfloat16_real, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_float32_intermediate_real);
    tile_regs_release();
    cb_push_back(cb_state_float32_intermediate_real, 1);

    cb_reserve_back(cb_state_float32_intermediate_imag, 1);
    reconfig_data_format_srca(cb_state_bfloat16_real, cb_state_bfloat16_imag);
    copy_tile_init(cb_state_bfloat16_imag);
    // CB14 and CB15 are both Float32, but the forced overload keeps the new CB identity explicit.
    pack_reconfig_data_format(cb_state_float32_intermediate_imag);
    tile_regs_acquire();
    copy_tile(cb_state_bfloat16_imag, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_float32_intermediate_imag);
    tile_regs_release();
    cb_push_back(cb_state_float32_intermediate_imag, 1);
    cb_pop_front(cb_state_bfloat16_real, 1);
    cb_pop_front(cb_state_bfloat16_imag, 1);
}

void switch_to_bfloat16_first_group_from_state() {
    // State-imag leaves SrcA=CB4, SrcB=CB5, pack=CB21; restore SrcA=CB1,
    // SrcB=CB0, and the Float32 product pack CB16 for the next iteration.
    reconfig_data_format(cb_product_ri, cb_x_bfloat16, cb_product_ir, cb_r);
    pack_reconfig_data_format(cb_product);
    matmul_block_init(cb_r, cb_x_bfloat16, false, 1, 1, 1);
}

void switch_to_float32_first_group_from_bfloat16() {
    convert_state_to_float32();
    // Conversion leaves SrcA=CB21, SrcB=CB5, pack=CB15.  Matmul in1 is
    // SrcA, while in0 is SrcB: move SrcA to X CB17 and SrcB to R CB0.
    reconfig_data_format_srca(cb_state_bfloat16_imag, cb_x_float32);
    reconfig_data_format_srcb(cb_product_ir, cb_r);
    pack_reconfig_data_format(cb_product);
    matmul_block_init(cb_r, cb_x_float32, false, 1, 1, 1);
}

void switch_to_float32_first_group_from_state() {
    // State-imag leaves SrcA=CB4, SrcB=CB5, pack=CB15.  Restore the
    // Float32 X/R pair and the Float32 product pack for the next iteration.
    reconfig_data_format(cb_product_ri, cb_x_float32, cb_product_ir, cb_r);
    pack_reconfig_data_format(cb_product);
    matmul_block_init(cb_r, cb_x_float32, false, 1, 1, 1);
}

void switch_to_bfloat16_second_group() {
    // S-imag leaves SrcA=CB7, SrcB=CB9, pack=CB11.  The reader's BF16
    // second-group operand is CB12 and X is CB1 (matmul in1 is SrcA).
    reconfig_data_format(cb_zero, cb_s_bfloat16_operand, cb_rx_imag, cb_x_bfloat16);
    pack_reconfig_data_format(cb_product);
    matmul_block_init(cb_x_bfloat16, cb_s_bfloat16_operand, false, 1, 1, 1);
}

void switch_to_float32_second_group() {
    // S-imag leaves SrcA=CB7, SrcB=CB9, pack=CB19.  The reader's Float32
    // second-group operand is CB22 and X is CB17 (matmul in1 is SrcA).
    reconfig_data_format(cb_zero, cb_s_float32_operand, cb_rx_imag, cb_x_float32);
    pack_reconfig_data_format(cb_product);
    matmul_block_init(cb_x_float32, cb_s_float32_operand, false, 1, 1, 1);
}
}  // namespace

// FP32 destination accumulation is enabled globally; only X/S state storage changes at iteration 4.
void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_r, cb_x_bfloat16, cb_product);
    // Startup establishes SrcA=CB1 BF16, SrcB=CB0 BF16, and pack=CB16 Float32.
    matmul_block_init(cb_r, cb_x_bfloat16, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        for (std::uint32_t iteration = 0; iteration < 8; ++iteration) {
            if (iteration == split_iteration) {
                switch_to_float32_first_group_from_bfloat16();
            } else if (iteration == 0) {
                // Startup already configured the first BF16 matmul boundary.
            } else if (iteration < split_iteration) {
                switch_to_bfloat16_first_group_from_state();
            } else {
                switch_to_float32_first_group_from_state();
            }

            const bool float32_phase = iteration >= split_iteration;
            const std::uint32_t first_srca = float32_phase ? cb_x_float32 : cb_x_bfloat16;
            complex_matmul_products(cb_r, first_srca);

            if (float32_phase) {
                subtract_one<
                    cb_x_float32,
                    cb_r,
                    cb_product,
                    cb_product_rr,
                    cb_product_ii,
                    cb_rx_real>();
                add_one<
                    cb_product_rr,
                    cb_product_ii,
                    cb_rx_real,
                    cb_product_ri,
                    cb_product_ir,
                    cb_rx_imag>();
                subtract_one<
                    cb_product_ri,
                    cb_product_ir,
                    cb_rx_imag,
                    cb_identity,
                    cb_rx_real,
                    cb_s_float32_real>();
                subtract_one<
                    cb_identity,
                    cb_rx_real,
                    cb_s_float32_real,
                    cb_zero,
                    cb_rx_imag,
                    cb_s_float32_imag>();
                switch_to_float32_second_group();
                complex_matmul_products(cb_x_float32, cb_s_float32_operand);
                if (iteration + 1 == 8) {
                    subtract_one<
                        cb_s_float32_operand,
                        cb_x_float32,
                        cb_product,
                        cb_product_rr,
                        cb_product_ii,
                        cb_state_float32_real>();
                    add_one<
                        cb_product_rr,
                        cb_product_ii,
                        cb_state_float32_real,
                        cb_product_ri,
                        cb_product_ir,
                        cb_state_float32_imag>();
                } else {
                    subtract_one<
                        cb_s_float32_operand,
                        cb_x_float32,
                        cb_product,
                        cb_product_rr,
                        cb_product_ii,
                        cb_state_float32_intermediate_real>();
                    add_one<
                        cb_product_rr,
                        cb_product_ii,
                        cb_state_float32_intermediate_real,
                        cb_product_ri,
                        cb_product_ir,
                        cb_state_float32_intermediate_imag>();
                }
            } else {
                subtract_one<
                    cb_x_bfloat16,
                    cb_r,
                    cb_product,
                    cb_product_rr,
                    cb_product_ii,
                    cb_rx_real>();
                add_one<
                    cb_product_rr,
                    cb_product_ii,
                    cb_rx_real,
                    cb_product_ri,
                    cb_product_ir,
                    cb_rx_imag>();
                subtract_one<
                    cb_product_ri,
                    cb_product_ir,
                    cb_rx_imag,
                    cb_identity,
                    cb_rx_real,
                    cb_s_bfloat16_real>();
                subtract_one<
                    cb_identity,
                    cb_rx_real,
                    cb_s_bfloat16_real,
                    cb_zero,
                    cb_rx_imag,
                    cb_s_bfloat16_imag>();
                switch_to_bfloat16_second_group();
                complex_matmul_products(cb_x_bfloat16, cb_s_bfloat16_operand);
                subtract_one<
                    cb_s_bfloat16_operand,
                    cb_x_bfloat16,
                    cb_product,
                    cb_product_rr,
                    cb_product_ii,
                    cb_state_bfloat16_real>();
                add_one<
                    cb_product_rr,
                    cb_product_ii,
                    cb_state_bfloat16_real,
                    cb_product_ri,
                    cb_product_ir,
                    cb_state_bfloat16_imag>();
            }
        }
    }
}
