// SPDX-License-Identifier: Apache-2.0
// Minimal two-tile a/b/c probe.  This is diagnostic-only and is not the
// throughput kernel: one 2-tile product, one iteration, matrix_block=1.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr std::uint32_t cb_x_real = 3;
constexpr std::uint32_t cb_x_imag = 4;
constexpr std::uint32_t cb_identity = 5;
constexpr std::uint32_t cb_zero = 6;
constexpr std::uint32_t cb_negative_x_imag = 13;
constexpr std::uint32_t cb_output_real = 15;
constexpr std::uint32_t cb_output_imag = 16;
constexpr std::uint32_t cb_two_tile_r = 20;
constexpr std::uint32_t cb_two_tile_x = 21;
constexpr std::uint32_t cb_two_tile_s = 22;

void build_x_column() {
    cb_reserve_back(cb_two_tile_s, 2);
    reconfig_data_format_srca(cb_two_tile_s, cb_x_real);
    tile_regs_acquire();
    copy_tile_init(cb_x_real);
    copy_tile(cb_x_real, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_x_real, cb_x_imag);
    copy_tile(cb_x_imag, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_output_real, cb_two_tile_s);
    pack_tile(0, cb_two_tile_s);
    pack_tile(1, cb_two_tile_s);
    tile_regs_release();
    cb_push_back(cb_two_tile_s, 2);
}

void build_x_block() {
    cb_wait_front(cb_x_real, 1);
    cb_wait_front(cb_x_imag, 1);
    cb_wait_front(cb_negative_x_imag, 1);
    cb_reserve_back(cb_two_tile_x, 4);
    // The seed-free K=2 call leaves SrcA on BF16 CB_IDENTITY. Reconfigure
    // from that format before unpacking the FP32/state X pages.
    reconfig_data_format_srca(cb_identity, cb_x_real);
    tile_regs_acquire();
    copy_tile_init(cb_x_real);
    copy_tile(cb_x_real, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_x_real, cb_x_imag);
    copy_tile(cb_x_imag, 0, 1);
    copy_tile_to_dst_init_short_with_dt(cb_x_imag, cb_negative_x_imag);
    copy_tile(cb_negative_x_imag, 0, 2);
    copy_tile_to_dst_init_short_with_dt(cb_negative_x_imag, cb_x_real);
    copy_tile(cb_x_real, 0, 3);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_two_tile_s, cb_two_tile_x);
    pack_tile(0, cb_two_tile_x);
    pack_tile(1, cb_two_tile_x);
    pack_tile(2, cb_two_tile_x);
    pack_tile(3, cb_two_tile_x);
    tile_regs_release();
    cb_push_back(cb_two_tile_x, 4);
}

// Compute [Sr; Si] = [2I; 0] - R*X, or a selected partial product.
// The R CB carries the six-page column-major [−Rr, −Ri, Ri, −Rr, 2I, 0]
// block.  Xr/Xi remain in the state-format CB and the exact BF16 I page stays
// resident in CB_IDENTITY for the third K term.
void r_times_x(
    bool pack_to_s,
    bool run_k0,
    bool run_k1,
    bool run_identity) {
    cb_wait_front(cb_two_tile_r, 6);
    cb_wait_front(cb_x_real, 1);
    cb_wait_front(cb_x_imag, 1);
    build_x_column();
    if (!pack_to_s) {
        cb_reserve_back(cb_output_real, 1);
        cb_reserve_back(cb_output_imag, 1);
    }

    // Matmul maps in0 to SrcB and in1 to SrcA. Configure each source at
    // the operation boundary before the matching rt=2, ct=1, kt=1 init.
    reconfig_data_format_srca(cb_two_tile_s);
    reconfig_data_format_srcb(cb_two_tile_r);
    matmul_block_init(cb_two_tile_r, cb_two_tile_s, false, 1, 2, 1);
    tile_regs_acquire();
    if (run_k0) {
        // k=0: (−Rr)*Xr and (−Ri)*Xr.  One rt=2 call writes both rows.
        matmul_block(cb_two_tile_r, cb_two_tile_s, 0, 0, 0, false, 1, 2, 1);
    }
    if (run_k1) {
        // k=1: (+Ri)*Xi and (−Rr)*Xi.  Accumulate into the same two rows.
        matmul_block(cb_two_tile_r, cb_two_tile_s, 2, 1, 0, false, 1, 2, 1);
    }
    if (run_identity) {
        // k=2: [2I; 0]*I.  The logical X/S-column offset is 2, but the
        // resident BF16 identity has physical offset 0 in its one-page CB.
        // Both constants are BF16 and no DEST seed copy is performed.
        cb_wait_front(cb_identity, 1);
        reconfig_data_format_srca(cb_identity);
        reconfig_data_format_srcb(cb_two_tile_r);
        matmul_block(cb_two_tile_r, cb_identity, 4, 0, 0, false, 1, 2, 1);
    }
    tile_regs_commit();
    tile_regs_wait();

    if (pack_to_s) {
        // Recycle the input column only after all matmul reads complete.
        cb_pop_front(cb_two_tile_s, 2);
        cb_reserve_back(cb_two_tile_s, 2);
        pack_reconfig_data_format(cb_output_real, cb_two_tile_s);
        pack_tile(0, cb_two_tile_s);
        pack_tile(1, cb_two_tile_s);
        cb_push_back(cb_two_tile_s, 2);
    } else {
        // The K=3 input column is not an output in a/b/a* stages.  Release
        // it before the next matrix so the two-page CB cannot fill.
        cb_pop_front(cb_two_tile_s, 2);
        pack_reconfig_data_format(cb_output_real);
        pack_tile(0, cb_output_real);
        pack_reconfig_data_format(cb_output_imag);
        pack_tile(1, cb_output_imag);
        cb_push_back(cb_output_real, 1);
        cb_push_back(cb_output_imag, 1);
    }
    tile_regs_release();
}

void x_times_s() {
    cb_wait_front(cb_two_tile_x, 4);
    cb_wait_front(cb_two_tile_s, 2);
    cb_reserve_back(cb_output_real, 1);
    cb_reserve_back(cb_output_imag, 1);
    // b-prime's K=2 call leaves SrcA=BF16 I and SrcB=BF16 R. Configure the
    // c operands independently so S and X/state are unpacked as FP32.
    reconfig_data_format_srca(cb_two_tile_s);
    reconfig_data_format_srcb(cb_two_tile_x);
    matmul_block_init(cb_two_tile_x, cb_two_tile_s, false, 1, 2, 1);
    tile_regs_acquire();
    // k=0: [Xr; Xi]*Sr writes the real and imaginary output rows.
    matmul_block(cb_two_tile_x, cb_two_tile_s, 0, 0, 0, false, 1, 2, 1);
    // k=1: [-Xi; Xr]*Si accumulates into those same two output rows.
    matmul_block(cb_two_tile_x, cb_two_tile_s, 2, 1, 0, false, 1, 2, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_two_tile_s, cb_output_real);
    pack_tile(0, cb_output_real);
    pack_reconfig_data_format(cb_two_tile_s, cb_output_imag);
    pack_tile(1, cb_output_imag);
    tile_regs_release();
    cb_push_back(cb_output_real, 1);
    cb_push_back(cb_output_imag, 1);

    cb_pop_front(cb_two_tile_x, 4);
    cb_pop_front(cb_two_tile_s, 2);
}

void pop_input_pages() {
    cb_pop_front(cb_two_tile_r, 6);
    cb_pop_front(cb_x_real, 1);
    cb_pop_front(cb_x_imag, 1);
    cb_pop_front(cb_negative_x_imag, 1);
}
}  // namespace

void kernel_main() {
    // 0=a, 1=a1, 2=a2, 3=a3, 4=b_prime (b alias), 5=c.
    constexpr std::uint32_t stage = get_compile_time_arg_val(0);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(0);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(1);
    (void)start_tile;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_two_tile_r, cb_two_tile_s, cb_output_real);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        if (stage == 0) {
            r_times_x(false, true, true, false);
            pop_input_pages();
        } else if (stage == 1) {
            r_times_x(false, true, false, false);
            pop_input_pages();
        } else if (stage == 2) {
            r_times_x(false, false, true, false);
            pop_input_pages();
        } else if (stage == 3) {
            r_times_x(false, true, true, false);
            pop_input_pages();
        } else if (stage == 4) {
            // b_prime: seed-free S=2I-RX, with 2I/0 supplied as K=2.
            r_times_x(false, true, true, true);
            pop_input_pages();
        } else if (stage == 5) {
            r_times_x(true, true, true, true);
            cb_pop_front(cb_two_tile_r, 6);
            build_x_block();
            x_times_s();
        }
    }
}
