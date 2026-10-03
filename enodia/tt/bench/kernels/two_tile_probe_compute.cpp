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
    pack_reconfig_data_format(cb_two_tile_s);
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
    reconfig_data_format_srca(cb_two_tile_x, cb_x_real);
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
    pack_reconfig_data_format(cb_two_tile_x);
    pack_tile(0, cb_two_tile_x);
    pack_tile(1, cb_two_tile_x);
    pack_tile(2, cb_two_tile_x);
    pack_tile(3, cb_two_tile_x);
    tile_regs_release();
    cb_push_back(cb_two_tile_x, 4);
}

void seed_dest_slots() {
    cb_wait_front(cb_identity, 1);
    cb_wait_front(cb_zero, 1);
    reconfig_data_format_srca(cb_two_tile_s, cb_identity);
    copy_tile_init(cb_identity);
    copy_tile(cb_identity, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_identity, cb_zero);
    copy_tile(cb_zero, 0, 1);
}

// Compute [Sr; Si] = [2I; 0] - R*X, or -R*X when seed_dest is false.
// The R and X CBs are the two operands: in0 is SrcB and in1 is SrcA.
void r_times_x(bool seed_dest, bool pack_to_s) {
    cb_wait_front(cb_two_tile_r, 4);
    cb_wait_front(cb_x_real, 1);
    cb_wait_front(cb_x_imag, 1);
    build_x_column();
    if (!pack_to_s) {
        cb_reserve_back(cb_output_real, 1);
        cb_reserve_back(cb_output_imag, 1);
    }

    reconfig_data_format(cb_two_tile_s, cb_two_tile_r);
    matmul_block_init(cb_two_tile_r, cb_two_tile_s, false, 1, 2, 1);
    tile_regs_acquire();
    if (seed_dest) {
        seed_dest_slots();
    }
    // Real output: (-Rr)*Xr + (+Ri)*Xi.
    matmul_block(cb_two_tile_r, cb_two_tile_s, 0, 0, 0, false, 1, 2, 1);
    matmul_block(cb_two_tile_r, cb_two_tile_s, 2, 1, 0, false, 1, 2, 1);
    // Imaginary output: (-Rr)*Xi + (-Ri)*Xr.
    matmul_block(cb_two_tile_r, cb_two_tile_s, 0, 1, 1, false, 1, 2, 1);
    matmul_block(cb_two_tile_r, cb_two_tile_s, 2, 0, 1, false, 1, 2, 1);
    tile_regs_commit();
    tile_regs_wait();

    if (pack_to_s) {
        // Recycle the input column only after all matmul reads complete.
        cb_pop_front(cb_two_tile_s, 2);
        cb_reserve_back(cb_two_tile_s, 2);
        pack_reconfig_data_format(cb_two_tile_s);
        pack_tile(0, cb_two_tile_s);
        pack_tile(1, cb_two_tile_s);
        cb_push_back(cb_two_tile_s, 2);
    } else {
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
    reconfig_data_format(cb_two_tile_s, cb_two_tile_x);
    matmul_block_init(cb_two_tile_x, cb_two_tile_s, false, 1, 2, 1);
    tile_regs_acquire();
    // Real output: Xr*Sr + (-Xi)*Si.
    matmul_block(cb_two_tile_x, cb_two_tile_s, 0, 0, 0, false, 1, 2, 1);
    matmul_block(cb_two_tile_x, cb_two_tile_s, 2, 1, 0, false, 1, 2, 1);
    // Imaginary output: Xr*Si + Xi*Sr.
    matmul_block(cb_two_tile_x, cb_two_tile_s, 0, 1, 1, false, 1, 2, 1);
    matmul_block(cb_two_tile_x, cb_two_tile_s, 1, 0, 1, false, 1, 2, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_output_real);
    pack_tile(0, cb_output_real);
    pack_reconfig_data_format(cb_output_imag);
    pack_tile(1, cb_output_imag);
    tile_regs_release();
    cb_push_back(cb_output_real, 1);
    cb_push_back(cb_output_imag, 1);

    cb_pop_front(cb_two_tile_x, 4);
    cb_pop_front(cb_two_tile_s, 2);
}

void pop_input_pages() {
    cb_pop_front(cb_two_tile_r, 4);
    cb_pop_front(cb_x_real, 1);
    cb_pop_front(cb_x_imag, 1);
    cb_pop_front(cb_negative_x_imag, 1);
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t stage = get_compile_time_arg_val(0);
    constexpr char probe_stage = stage == 0 ? 'a' : (stage == 1 ? 'b' : 'c');
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(0);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(1);
    (void)start_tile;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_two_tile_r, cb_two_tile_s, cb_output_real);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        if (probe_stage == 'a') {
            r_times_x(false, false);
            pop_input_pages();
        } else if (probe_stage == 'b') {
            r_times_x(true, false);
            pop_input_pages();
        } else if (probe_stage == 'c') {
            r_times_x(true, true);
            cb_pop_front(cb_two_tile_r, 4);
            build_x_block();
            x_times_s();
            pop_input_pages();
        }
    }
}
