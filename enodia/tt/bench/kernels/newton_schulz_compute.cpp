// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

namespace {
constexpr std::uint32_t cb_r_real = 0;
constexpr std::uint32_t cb_r_negative_imag = 1;
constexpr std::uint32_t cb_r_imag = 2;
constexpr std::uint32_t cb_x0_real = 3;
constexpr std::uint32_t cb_x0_imag = 4;
constexpr std::uint32_t cb_identity = 5;
constexpr std::uint32_t cb_zero = 6;
constexpr std::uint32_t cb_state_real = 7;
constexpr std::uint32_t cb_state_imag = 8;
constexpr std::uint32_t cb_s_real = 9;
constexpr std::uint32_t cb_s_imag = 10;
constexpr std::uint32_t cb_product_real = 11;
constexpr std::uint32_t cb_product_imag = 12;
constexpr std::uint32_t cb_negative_x_imag = 13;
constexpr std::uint32_t cb_output_real = 15;
constexpr std::uint32_t cb_output_imag = 16;

void pack_one(std::uint32_t output) {
    tile_regs_wait();
    pack_tile(0, output);
    tile_regs_release();
    cb_push_back(output, 1);
}

void complex_matmul(
    std::uint32_t left_real,
    std::uint32_t left_imag_for_real,
    std::uint32_t left_imag_for_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output_real,
    std::uint32_t output_imag,
    bool resident_left,
    bool consume_left,
    bool consume_right) {
    // With SrcOrder::Reverse the second CB is SrcA and the first CB is SrcB.
    // Each half keeps both signed terms in one acquired DEST tile.  The two
    // left-imaginary CBs are intentionally separate: R supplies -R_im for
    // the real half and +R_im for the imaginary half; X uses the same pattern
    // through the compute-owned negated-X CB below.
    matmul_block_init(left_real, right_real, false, 1, 1, 1);

    if (!resident_left) {
        cb_wait_front(left_real, 1);
        cb_wait_front(left_imag_for_real, 1);
        cb_wait_front(left_imag_for_imag, 1);
    }
    cb_wait_front(right_real, 1);
    cb_wait_front(right_imag, 1);
    cb_reserve_back(output_real, 1);

    // Order 1: each complex half owns one DEST section.  matmul_block is
    // DST += C, so the two terms for one half share dst0, then the second
    // half starts a separate acquire/commit/pack/release section.
    tile_regs_acquire();
    matmul_block(left_real, right_real, 0, 0, 0, false, 1, 1, 1);
    matmul_block(left_imag_for_real, right_imag, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    pack_reconfig_data_format(output_real);
    pack_one(output_real);

    cb_reserve_back(output_imag, 1);
    tile_regs_acquire();
    matmul_block(left_real, right_imag, 0, 0, 0, false, 1, 1, 1);
    matmul_block(left_imag_for_imag, right_real, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    pack_reconfig_data_format(output_imag);
    pack_one(output_imag);

    if (consume_left) {
        cb_pop_front(left_real, 1);
        cb_pop_front(left_imag_for_real, 1);
        if (left_imag_for_imag != left_imag_for_real) {
            cb_pop_front(left_imag_for_imag, 1);
        }
    }
    if (consume_right) {
        cb_pop_front(right_real, 1);
        cb_pop_front(right_imag, 1);
    }
}

void subtract_one(
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output,
    bool consume_left,
    bool consume_right) {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    reconfig_data_format(current_srca, left, current_srcb, right);
    pack_reconfig_data_format(output);
    sub_tiles_init(left, right);
    tile_regs_acquire();
    sub_tiles(left, right, 0, 0, 0);
    tile_regs_commit();
    pack_one(output);
    if (consume_left) {
        cb_pop_front(left, 1);
    }
    if (consume_right) {
        cb_pop_front(right, 1);
    }
}

void negate_state_imag(std::uint32_t x_imag) {
    cb_wait_front(x_imag, 1);
    cb_wait_front(cb_zero, 1);
    cb_reserve_back(cb_negative_x_imag, 1);
    // Keep the sign conversion on the known binary/reconfiguration path.  The
    // zero and X tiles are both retained: zero is the per-core constant and X
    // must remain available for the following X*S product.
    reconfig_data_format(cb_zero, cb_zero, cb_product_imag, x_imag);
    pack_reconfig_data_format(cb_negative_x_imag);
    sub_tiles_init(cb_zero, x_imag);
    tile_regs_acquire();
    sub_tiles(cb_zero, x_imag, 0, 0, 0);
    tile_regs_commit();
    pack_one(cb_negative_x_imag);
}

void residual_format_transition_to_matmul(std::uint32_t x_real, std::uint32_t x_imag) {
    // Negation leaves zero/x_imag as the active SrcA/SrcB pair.  The next
    // matmul consumes BF16 S/X operands and writes FP32 DEST products.
    reconfig_data_format(cb_zero, cb_s_real, x_imag, x_real);
}

void stream_initial_or_state(
    std::uint32_t iteration,
    std::uint32_t& x_real,
    std::uint32_t& x_imag) {
    if (iteration == 0) {
        x_real = cb_x0_real;
        x_imag = cb_x0_imag;
    } else {
        x_real = cb_state_real;
        x_imag = cb_state_imag;
    }
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
    constexpr bool state_fp32 = get_compile_time_arg_val(1) != 0;
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(0);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(1);
    static_assert(iterations == 8, "the throughput kernel has a fixed eight-iteration count");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_r_real, cb_x0_real, cb_product_real);
    matmul_block_init(cb_r_real, cb_x0_real, false, 1, 1, 1);

    (void)start_tile;
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        // R is resident for this matrix: wait once, reuse its three pages for
        // all eight iterations, and pop them only after the final update.
        cb_wait_front(cb_r_real, 1);
        cb_wait_front(cb_r_negative_imag, 1);
        cb_wait_front(cb_r_imag, 1);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            std::uint32_t x_real;
            std::uint32_t x_imag;
            stream_initial_or_state(iteration, x_real, x_imag);

            if constexpr (state_fp32) {
                if (iteration != 0) {
                    // The preceding X*S leaves SrcA=S_real and SrcB=X_imag.
                    // Restore the mixed BF16-R / FP32-X matmul formats before
                    // the next iteration; BF16 state needs no transition.
                    reconfig_data_format(cb_s_real, x_real, x_imag, cb_r_real);
                }
            }

            // T = R * X.  The host supplies -R_im, so the real accumulation is
            // R_re*X_re + (-R_im)*X_im and the imaginary accumulation uses +R_im.
            // X is retained here because the following product also consumes X.
            complex_matmul(
                cb_r_real,
                cb_r_negative_imag,
                cb_r_imag,
                x_real,
                x_imag,
                cb_product_real,
                cb_product_imag,
                true,
                false,
                false);

            // S = 2I - T.  Its BF16 outputs are the operands of X*S.
            // The final R*X term leaves x_real/r_imag as the active
            // SrcA/SrcB pair under SrcOrder::Reverse.
            subtract_one(
                x_real,
                cb_r_imag,
                cb_identity,
                cb_product_real,
                cb_s_real,
                false,
                true);
            subtract_one(
                cb_identity,
                cb_product_real,
                cb_zero,
                cb_product_imag,
                cb_s_imag,
                false,
                true);

            // X*S needs -X_im for its real half and +X_im for its imaginary
            // half.  Generate the sign-only operand in a compute-owned CB;
            // no extra reader stream or product-combination CB is required.
            negate_state_imag(x_imag);
            residual_format_transition_to_matmul(x_real, x_imag);
            const std::uint32_t output_real =
                iteration + 1 == iterations ? cb_output_real : cb_state_real;
            const std::uint32_t output_imag =
                iteration + 1 == iterations ? cb_output_imag : cb_state_imag;
            complex_matmul(
                x_real,
                cb_negative_x_imag,
                x_imag,
                cb_s_real,
                cb_s_imag,
                output_real,
                output_imag,
                false,
                true,
                true);
        }

        // R is resident for all eight iterations of this matrix and is only
        // released after the final X*S product has been published.
        cb_pop_front(cb_r_real, 1);
        cb_pop_front(cb_r_negative_imag, 1);
        cb_pop_front(cb_r_imag, 1);
    }
}
