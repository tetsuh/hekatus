// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <type_traits>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

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
constexpr std::uint32_t cb_r_negative_real = 14;
constexpr std::uint32_t cb_output_real = 15;
constexpr std::uint32_t cb_output_imag = 16;
constexpr std::uint32_t cb_profile_compute = 18;
constexpr std::uint32_t profile_magic = 0x5052464C;
constexpr std::uint32_t profile_ready_offset = 31;
constexpr std::uint32_t profile_warmup_ready_offset = 63;
constexpr std::uint32_t profile_slot_stride = 64;
constexpr std::uint32_t profile_warmup_base = 32;
constexpr std::uint32_t profile_total_offset = 0;
constexpr std::uint32_t profile_r_wait_offset = 1;
constexpr std::uint32_t profile_x_wait_offset = 2;
constexpr std::uint32_t profile_complex_real_offset = 3;
constexpr std::uint32_t profile_complex_imag_offset = 4;
constexpr std::uint32_t profile_s_binary_offset = 5;
constexpr std::uint32_t profile_pack_push_offset = 6;
constexpr std::uint32_t profile_state_handoff_offset = 7;
constexpr std::uint32_t profile_section_sum_offset = 8;
constexpr std::uint32_t profile_residual_offset = 9;
constexpr std::uint32_t profile_event_count_offset = 10;
constexpr std::uint32_t profile_warmup_event_count_offset = 11;

// Counters use the lower 32-bit wall-clock API. All sections accumulate over
// the assigned tiles and eight iterations; the warmup fields retain the first
// tile/iteration separately. Timestamp reads, CB pushes/pops, and setup that
// is outside a named section remain visible as an explicit residual.
struct ProfileCounters {
    std::uint32_t total_start = 0;
    std::uint32_t total_end = 0;
    std::uint32_t warmup_start = 0;
    std::uint32_t warmup_end = 0;
    std::uint32_t event_count = 0;
    std::uint32_t warmup_event_count = 0;
    std::uint32_t r_wait = 0;
    std::uint32_t x_wait = 0;
    std::uint32_t complex_real = 0;
    std::uint32_t complex_imag = 0;
    std::uint32_t s_binary = 0;
    std::uint32_t pack_push = 0;
    std::uint32_t state_handoff = 0;
    std::uint32_t warmup_r_wait = 0;
    std::uint32_t warmup_x_wait = 0;
    std::uint32_t warmup_complex_real = 0;
    std::uint32_t warmup_complex_imag = 0;
    std::uint32_t warmup_s_binary = 0;
    std::uint32_t warmup_pack_push = 0;
    std::uint32_t warmup_state_handoff = 0;
};

void add_profile_cycles(std::uint32_t& total, std::uint32_t& warmup, std::uint32_t cycles, bool is_warmup) {
    total += cycles;
    if (is_warmup) {
        warmup += cycles;
    }
}
struct EmptyProfileCounters {};

void pack_one(std::uint32_t output) {
    tile_regs_wait();
    pack_tile(0, output);
    tile_regs_release();
    cb_push_back(output, 1);
}

void pack_one_profiled(std::uint32_t output, ProfileCounters& counters, bool warmup) {
    DeviceZoneScopedN("NS-COMPUTE-PACK-PUSH");
    const std::uint32_t start = get_timestamp_32b();
    pack_one(output);
    add_profile_cycles(
        counters.pack_push,
        counters.warmup_pack_push,
        get_timestamp_32b() - start,
        warmup);
}

void wait_complex_inputs(
    std::uint32_t left_real,
    std::uint32_t left_imag_for_real,
    std::uint32_t left_imag_for_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    bool resident_left) {
    if (!resident_left) {
        cb_wait_front(left_real, 1);
        cb_wait_front(left_imag_for_real, 1);
        cb_wait_front(left_imag_for_imag, 1);
    }
    cb_wait_front(right_real, 1);
    cb_wait_front(right_imag, 1);
}

void wait_complex_inputs_profiled(
    std::uint32_t left_real,
    std::uint32_t left_imag_for_real,
    std::uint32_t left_imag_for_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    bool resident_left,
    ProfileCounters& counters,
    bool warmup) {
    DeviceZoneScopedN("NS-COMPUTE-X-CB-WAIT");
    const std::uint32_t start = get_timestamp_32b();
    wait_complex_inputs(
        left_real,
        left_imag_for_real,
        left_imag_for_imag,
        right_real,
        right_imag,
        resident_left);
    add_profile_cycles(
        counters.x_wait,
        counters.warmup_x_wait,
        get_timestamp_32b() - start,
        warmup);
}

void complex_real_impl(
    std::uint32_t left_real,
    std::uint32_t left_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output,
    bool profile_pack,
    ProfileCounters* counters = nullptr,
    bool warmup = false) {
    cb_reserve_back(output, 1);
    tile_regs_acquire();
    matmul_block(left_real, right_real, 0, 0, 0, false, 1, 1, 1);
    matmul_block(left_imag, right_imag, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    pack_reconfig_data_format(output);
    if (profile_pack) {
        pack_one_profiled(output, *counters, warmup);
    } else {
        pack_one(output);
    }
}

void complex_real_profiled(
    std::uint32_t left_real,
    std::uint32_t left_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output,
    ProfileCounters& counters,
    bool warmup) {
    DeviceZoneScopedN("NS-COMPUTE-COMPLEX-REAL");
    const std::uint32_t start = get_timestamp_32b();
    complex_real_impl(left_real, left_imag, right_real, right_imag, output, true, &counters, warmup);
    add_profile_cycles(
        counters.complex_real,
        counters.warmup_complex_real,
        get_timestamp_32b() - start,
        warmup);
}

void complex_imag_impl(
    std::uint32_t left_real,
    std::uint32_t left_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output,
    bool profile_pack,
    ProfileCounters* counters = nullptr,
    bool warmup = false) {
    cb_reserve_back(output, 1);
    tile_regs_acquire();
    matmul_block(left_real, right_imag, 0, 0, 0, false, 1, 1, 1);
    matmul_block(left_imag, right_real, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    pack_reconfig_data_format(output);
    if (profile_pack) {
        pack_one_profiled(output, *counters, warmup);
    } else {
        pack_one(output);
    }
}

void complex_imag_profiled(
    std::uint32_t left_real,
    std::uint32_t left_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output,
    ProfileCounters& counters,
    bool warmup) {
    DeviceZoneScopedN("NS-COMPUTE-COMPLEX-IMAG");
    const std::uint32_t start = get_timestamp_32b();
    complex_imag_impl(left_real, left_imag, right_real, right_imag, output, true, &counters, warmup);
    add_profile_cycles(
        counters.complex_imag,
        counters.warmup_complex_imag,
        get_timestamp_32b() - start,
        warmup);
}

template <bool profile_sample>
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
    bool consume_right,
    bool profile_enabled,
    bool warmup = false,
    ProfileCounters* counters = nullptr) {
    // With SrcOrder::Reverse the second CB is SrcA and the first CB is SrcB.
    // Each half keeps both signed terms in one acquired DEST tile.  Order 1
    // deliberately uses one DEST section per half; this fidelity sweep must
    // not mix the later two-DEST experiment into its timings.
    matmul_block_init(left_real, right_real, false, 1, 1, 1);
    if constexpr (profile_sample) {
        if (profile_enabled) {
            wait_complex_inputs_profiled(
                left_real,
                left_imag_for_real,
                left_imag_for_imag,
                right_real,
                right_imag,
                resident_left,
                *counters,
                warmup);
        } else {
            wait_complex_inputs(
                left_real,
                left_imag_for_real,
                left_imag_for_imag,
                right_real,
                right_imag,
                resident_left);
        }
    } else {
        wait_complex_inputs(
            left_real,
            left_imag_for_real,
            left_imag_for_imag,
            right_real,
            right_imag,
            resident_left);
    }

    if constexpr (profile_sample) {
        if (profile_enabled) {
            complex_real_profiled(
                left_real,
                left_imag_for_real,
                right_real,
                right_imag,
                output_real,
                *counters,
                warmup);
        } else {
            complex_real_impl(left_real, left_imag_for_real, right_real, right_imag, output_real, false);
        }
    } else {
        complex_real_impl(left_real, left_imag_for_real, right_real, right_imag, output_real, false);
    }

    if constexpr (profile_sample) {
        if (profile_enabled) {
            complex_imag_profiled(
                left_real,
                left_imag_for_imag,
                right_real,
                right_imag,
                output_imag,
                *counters,
                warmup);
        } else {
            complex_imag_impl(left_real, left_imag_for_imag, right_real, right_imag, output_imag, false);
        }
    } else {
        complex_imag_impl(left_real, left_imag_for_imag, right_real, right_imag, output_imag, false);
    }

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

void wait_complex_inputs_block(
    std::uint32_t left_real,
    std::uint32_t left_imag_for_real,
    std::uint32_t left_imag_for_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t block_count,
    bool resident_left) {
    if (!resident_left) {
        cb_wait_front(left_real, block_count);
        cb_wait_front(left_imag_for_real, block_count);
        if (left_imag_for_imag != left_imag_for_real) {
            cb_wait_front(left_imag_for_imag, block_count);
        }
    }
    cb_wait_front(right_real, block_count);
    cb_wait_front(right_imag, block_count);
}

template <bool one_dest_half>
void complex_matmul_block(
    std::uint32_t left_real,
    std::uint32_t left_imag_for_real,
    std::uint32_t left_imag_for_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output_real,
    std::uint32_t output_imag,
    std::uint32_t block_count,
    bool resident_left,
    bool consume_left,
    bool consume_right) {
    wait_complex_inputs_block(
        left_real,
        left_imag_for_real,
        left_imag_for_imag,
        right_real,
        right_imag,
        block_count,
        resident_left);

    const std::uint32_t block_left_real = left_real;
    const std::uint32_t block_right_real = right_real;
    const std::uint32_t block_left_imag = left_imag_for_real;
    const std::uint32_t block_right_imag = right_imag;
    const std::uint32_t block_left_imag_for_imag = left_imag_for_imag;
    if constexpr (one_dest_half) {
        // Block 8 has only eight FP32/full-sync DEST tiles.  Its state CBs
        // retain two block windows so the next iteration can reserve output
        // while the current state remains available for both DEST passes.
        cb_reserve_back(output_real, block_count);
        cb_reserve_back(output_imag, block_count);

        // Accumulate and pack the real and imaginary halves in separate DEST
        // passes, using one tile per matrix in each pass.
        matmul_block_init(left_real, right_real, false, 1, 1, 1);
        tile_regs_acquire();
        for (std::uint32_t index = 0; index < block_count; ++index) {
            matmul_block(block_left_real, block_right_real, index, index, index, false, 1, 1, 1);
            matmul_block(block_left_imag, block_right_imag, index, index, index, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(output_real);
        pack_tile_block(0, output_real, block_count);
        tile_regs_release();

        matmul_block_init(left_real, right_real, false, 1, 1, 1);
        tile_regs_acquire();
        for (std::uint32_t index = 0; index < block_count; ++index) {
            matmul_block(block_left_real, block_right_imag, index, index, index, false, 1, 1, 1);
            matmul_block(block_left_imag_for_imag, block_right_real, index, index, index, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(output_imag);
        pack_tile_block(0, output_imag, block_count);
        tile_regs_release();
        cb_push_back(output_real, block_count);
        cb_push_back(output_imag, block_count);
    } else {
        // Blocks 2/4 retain the fast two-half path.  Delay output reservation
        // until after the DEST pass because state CBs are also its inputs.
        matmul_block_init(left_real, right_real, false, 1, 1, 1);
        tile_regs_acquire();
        for (std::uint32_t index = 0; index < block_count; ++index) {
            matmul_block(block_left_real, block_right_real, index, index, index, false, 1, 1, 1);
            matmul_block(block_left_imag, block_right_imag, index, index, index, false, 1, 1, 1);
            matmul_block(
                block_left_real,
                block_right_imag,
                index,
                index,
                block_count + index,
                false,
                1,
                1,
                1);
            matmul_block(
                block_left_imag_for_imag,
                block_right_real,
                index,
                index,
                block_count + index,
                false,
                1,
                1,
                1);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (consume_left) {
            // All input reads have completed, so releasing the input slots
            // before reserve_back breaks the state-input/output cycle.
            cb_pop_front(left_real, block_count);
            cb_pop_front(left_imag_for_real, block_count);
            if (left_imag_for_imag != left_imag_for_real) {
                cb_pop_front(left_imag_for_imag, block_count);
            }
        }
        cb_reserve_back(output_real, block_count);
        cb_reserve_back(output_imag, block_count);
        pack_reconfig_data_format(output_real);
        pack_tile_block(0, output_real, block_count);
        pack_reconfig_data_format(output_imag);
        pack_tile_block(block_count, output_imag, block_count);
        tile_regs_release();
        cb_push_back(output_real, block_count);
        cb_push_back(output_imag, block_count);
    }

    if constexpr (one_dest_half) {
        if (consume_left) {
            cb_pop_front(left_real, block_count);
            cb_pop_front(left_imag_for_real, block_count);
            if (left_imag_for_imag != left_imag_for_real) {
                cb_pop_front(left_imag_for_imag, block_count);
            }
        }
    }
    if (consume_right) {
        cb_pop_front(right_real, block_count);
        cb_pop_front(right_imag, block_count);
    }
}

template <bool one_dest_half>
void fused_s_matmul_block(
    std::uint32_t negative_r_real,
    std::uint32_t negative_r_imag,
    std::uint32_t positive_r_imag,
    std::uint32_t x_real,
    std::uint32_t x_imag,
    std::uint32_t block_count) {
    cb_wait_front(cb_identity, 1);
    cb_wait_front(cb_zero, 1);
    cb_wait_front(x_real, block_count);
    cb_wait_front(x_imag, block_count);
    cb_wait_front(negative_r_real, block_count);
    cb_wait_front(negative_r_imag, block_count);
    cb_wait_front(positive_r_imag, block_count);
    cb_reserve_back(cb_s_real, block_count);
    cb_reserve_back(cb_s_imag, block_count);

    if constexpr (one_dest_half) {
        // Block 8 cannot hold real and imaginary S tiles simultaneously.
        // Build, pack, and release each half before acquiring the next one.
        reconfig_data_format_srca(x_real, cb_identity);
        copy_tile_init(cb_identity);
        tile_regs_acquire();
        for (std::uint32_t index = 0; index < block_count; ++index) {
            copy_tile(cb_identity, 0, index);
        }
        reconfig_data_format(x_real, negative_r_real);
        matmul_block_init(negative_r_real, x_real, false, 1, 1, 1);
        for (std::uint32_t index = 0; index < block_count; ++index) {
            // S_re = 2I + (-R_re)X_re + (+R_im)X_im.
            matmul_block(negative_r_real, x_real, index, index, index, false, 1, 1, 1);
            matmul_block(positive_r_imag, x_imag, index, index, index, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s_real);
        pack_tile_block(0, cb_s_real, block_count);
        tile_regs_release();

        reconfig_data_format_srca(x_real, cb_zero);
        copy_tile_init(cb_zero);
        tile_regs_acquire();
        for (std::uint32_t index = 0; index < block_count; ++index) {
            copy_tile(cb_zero, 0, index);
        }
        reconfig_data_format(x_real, negative_r_real);
        matmul_block_init(negative_r_real, x_real, false, 1, 1, 1);
        for (std::uint32_t index = 0; index < block_count; ++index) {
            // S_im = (-R_re)X_im + (-R_im)X_re.
            matmul_block(negative_r_real, x_imag, index, index, index, false, 1, 1, 1);
            matmul_block(negative_r_imag, x_real, index, index, index, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s_imag);
        pack_tile_block(0, cb_s_imag, block_count);
        tile_regs_release();
    } else {
        // Blocks 1/2/4 retain the two-half fused path.
        reconfig_data_format_srca(x_real, cb_identity);
        copy_tile_init(cb_identity);
        tile_regs_acquire();
        for (std::uint32_t index = 0; index < block_count; ++index) {
            copy_tile(cb_identity, 0, index);
        }
        copy_tile_to_dst_init_short_with_dt(cb_identity, cb_zero);
        for (std::uint32_t index = 0; index < block_count; ++index) {
            copy_tile(cb_zero, 0, block_count + index);
        }
        reconfig_data_format(x_real, negative_r_real);
        matmul_block_init(negative_r_real, x_real, false, 1, 1, 1);
        for (std::uint32_t index = 0; index < block_count; ++index) {
            const std::uint32_t imag_slot = block_count + index;
            // S_re = 2I + (-R_re)X_re + (+R_im)X_im.
            matmul_block(negative_r_real, x_real, index, index, index, false, 1, 1, 1);
            matmul_block(positive_r_imag, x_imag, index, index, index, false, 1, 1, 1);
            // S_im = (-R_re)X_im + (-R_im)X_re.
            matmul_block(negative_r_real, x_imag, index, index, imag_slot, false, 1, 1, 1);
            matmul_block(negative_r_imag, x_real, index, index, imag_slot, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s_real);
        pack_tile_block(0, cb_s_real, block_count);
        pack_reconfig_data_format(cb_s_imag);
        pack_tile_block(block_count, cb_s_imag, block_count);
        tile_regs_release();
    }
    cb_push_back(cb_s_real, block_count);
    cb_push_back(cb_s_imag, block_count);
}

void subtract_block(
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output,
    std::uint32_t block_count,
    bool left_resident,
    bool consume_right) {
    if (!left_resident) {
        cb_wait_front(left, block_count);
    } else {
        cb_wait_front(left, 1);
    }
    cb_wait_front(right, block_count);
    cb_reserve_back(output, block_count);
    reconfig_data_format(current_srca, left, current_srcb, right);
    pack_reconfig_data_format(output);
    sub_tiles_init(left, right);
    tile_regs_acquire();
    for (std::uint32_t index = 0; index < block_count; ++index) {
        sub_tiles(left, right, left_resident ? 0 : index, index, index);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(0, output, block_count);
    tile_regs_release();
    cb_push_back(output, block_count);
    if (consume_right) {
        cb_pop_front(right, block_count);
    }
    if (!left_resident) {
        cb_pop_front(left, block_count);
    }
}

void negate_state_imag_block(
    std::uint32_t x_imag,
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t block_count) {
    cb_wait_front(x_imag, block_count);
    cb_wait_front(cb_zero, 1);
    cb_reserve_back(cb_negative_x_imag, block_count);
    reconfig_data_format(current_srca, cb_zero, current_srcb, x_imag);
    pack_reconfig_data_format(cb_negative_x_imag);
    sub_tiles_init(cb_zero, x_imag);
    tile_regs_acquire();
    for (std::uint32_t index = 0; index < block_count; ++index) {
        sub_tiles(cb_zero, x_imag, 0, index, index);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(0, cb_negative_x_imag, block_count);
    tile_regs_release();
    cb_push_back(cb_negative_x_imag, block_count);
}

// Build S directly in DEST: start S_re at BF16 2I and S_im at FP32 zero,
// then accumulate the signed BF16 R terms against X.  The output CB is the
// only pack boundary for S; RX never makes a product CB round trip in the
// fused path.
void fused_s_matmul(
    std::uint32_t negative_r_real,
    std::uint32_t negative_r_imag,
    std::uint32_t positive_r_imag,
    std::uint32_t x_real,
    std::uint32_t x_imag) {
    cb_wait_front(cb_identity, 1);
    cb_wait_front(cb_zero, 1);
    cb_wait_front(x_real, 1);
    cb_wait_front(x_imag, 1);
    cb_reserve_back(cb_s_real, 1);
    cb_reserve_back(cb_s_imag, 1);

    // copy_tile_init changes only the copy operation, not SrcA's data format.
    // Select the BF16 identity format before loading it; this is essential when
    // X/state is FP32.  The BF16-state path safely treats x_real as the same
    // format even when the preceding operation left SrcA on S_real.
    reconfig_data_format_srca(x_real, cb_identity);
    copy_tile_init(cb_identity);
    // Reconfigure both operands before the short matmul init so BF16 X and
    // mixed BF16 R/FP32 X use their CB formats.
    tile_regs_acquire();
    copy_tile(cb_identity, 0, 0);
    // S_im has no identity term.  This short transition changes SrcA from the
    // BF16 identity CB to the FP32 zero CB without a full binary initializer.
    copy_tile_to_dst_init_short_with_dt(cb_identity, cb_zero);
    copy_tile(cb_zero, 0, 1);
    reconfig_data_format(x_real, negative_r_real);
    matmul_block_init(negative_r_real, x_real, false, 1, 1, 1);

    // S_re = 2I + (-R_re)X_re + (+R_im)X_im.
    matmul_block(negative_r_real, x_real, 0, 0, 0, false, 1, 1, 1);
    matmul_block(positive_r_imag, x_imag, 0, 0, 0, false, 1, 1, 1);
    // S_im = (-R_re)X_im + (-R_im)X_re.
    matmul_block(negative_r_real, x_imag, 0, 0, 1, false, 1, 1, 1);
    matmul_block(negative_r_imag, x_real, 0, 0, 1, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_s_real);
    pack_tile(0, cb_s_real);
    pack_reconfig_data_format(cb_s_imag);
    pack_tile(1, cb_s_imag);
    tile_regs_release();
    cb_push_back(cb_s_real, 1);
    cb_push_back(cb_s_imag, 1);
}

void subtract_one_impl(
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output,
    bool consume_left,
    bool consume_right,
    bool profile_pack,
    ProfileCounters* counters = nullptr,
    bool warmup = false) {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    reconfig_data_format(current_srca, left, current_srcb, right);
    pack_reconfig_data_format(output);
    sub_tiles_init(left, right);
    tile_regs_acquire();
    sub_tiles(left, right, 0, 0, 0);
    tile_regs_commit();
    if (profile_pack) {
        pack_one_profiled(output, *counters, warmup);
    } else {
        pack_one(output);
    }
    if (consume_left) {
        cb_pop_front(left, 1);
    }
    if (consume_right) {
        cb_pop_front(right, 1);
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
    subtract_one_impl(current_srca, current_srcb, left, right, output, consume_left, consume_right, false);
}

void subtract_one_profiled(
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output,
    bool consume_left,
    bool consume_right,
    ProfileCounters& counters,
    bool warmup) {
    DeviceZoneScopedN("NS-COMPUTE-S-BINARY");
    const std::uint32_t start = get_timestamp_32b();
    subtract_one_impl(
        current_srca,
        current_srcb,
        left,
        right,
        output,
        consume_left,
        consume_right,
        true,
        &counters,
        warmup);
    add_profile_cycles(
        counters.s_binary,
        counters.warmup_s_binary,
        get_timestamp_32b() - start,
        warmup);
}

void negate_state_imag_impl(
    std::uint32_t x_imag,
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    bool profile_pack,
    ProfileCounters* counters = nullptr,
    bool warmup = false) {
    cb_wait_front(x_imag, 1);
    cb_wait_front(cb_zero, 1);
    cb_reserve_back(cb_negative_x_imag, 1);
    reconfig_data_format(current_srca, cb_zero, current_srcb, x_imag);
    pack_reconfig_data_format(cb_negative_x_imag);
    sub_tiles_init(cb_zero, x_imag);
    tile_regs_acquire();
    sub_tiles(cb_zero, x_imag, 0, 0, 0);
    tile_regs_commit();
    if (profile_pack) {
        pack_one_profiled(cb_negative_x_imag, *counters, warmup);
    } else {
        pack_one(cb_negative_x_imag);
    }
}

void residual_format_transition_to_matmul(std::uint32_t x_real, std::uint32_t x_imag) {
    reconfig_data_format(cb_zero, cb_s_real, x_imag, x_real);
}

void state_handoff(
    std::uint32_t x_real,
    std::uint32_t x_imag,
    std::uint32_t current_srca,
    std::uint32_t current_srcb) {
    negate_state_imag_impl(x_imag, current_srca, current_srcb, false);
    residual_format_transition_to_matmul(x_real, x_imag);
}

void state_handoff_profiled(
    std::uint32_t x_real,
    std::uint32_t x_imag,
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    ProfileCounters& counters,
    bool warmup) {
    DeviceZoneScopedN("NS-COMPUTE-STATE-HANDOFF");
    const std::uint32_t start = get_timestamp_32b();
    negate_state_imag_impl(x_imag, current_srca, current_srcb, true, &counters, warmup);
    residual_format_transition_to_matmul(x_real, x_imag);
    add_profile_cycles(
        counters.state_handoff,
        counters.warmup_state_handoff,
        get_timestamp_32b() - start,
        warmup);
}

template <bool fuse_s>
void wait_r_inputs_block(std::uint32_t block_count) {
    if constexpr (!fuse_s) {
        cb_wait_front(cb_r_real, block_count);
    }
    cb_wait_front(cb_r_negative_imag, block_count);
    cb_wait_front(cb_r_imag, block_count);
    if constexpr (fuse_s) {
        cb_wait_front(cb_r_negative_real, block_count);
    }
}

void stream_initial_or_state(
    std::uint32_t iteration,
    std::uint32_t& x_real,
    std::uint32_t& x_imag);

constexpr std::uint32_t cb_two_tile_r = 20;
constexpr std::uint32_t cb_two_tile_x = 21;
constexpr std::uint32_t cb_two_tile_s = 22;

// Build the physical column-major pages for [[Xr, -Xi], [Xi, Xr]].
// The two K=1 calls consume [Xr, Xi] then [-Xi, Xr].
void build_two_tile_x_block(
    std::uint32_t x_real,
    std::uint32_t x_imag,
    std::uint32_t negative_x_imag,
    std::uint32_t block_count) {
    cb_wait_front(x_real, block_count);
    cb_wait_front(x_imag, block_count);
    cb_wait_front(negative_x_imag, block_count);
    cb_reserve_back(cb_two_tile_x, 4 * block_count);
    reconfig_data_format_srca(cb_two_tile_x, x_real);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        tile_regs_acquire();
        copy_tile_init(x_real);
        copy_tile(x_real, index, 0);
        copy_tile_to_dst_init_short_with_dt(x_real, x_imag);
        copy_tile(x_imag, index, 1);
        copy_tile_to_dst_init_short_with_dt(x_imag, negative_x_imag);
        copy_tile(negative_x_imag, index, 2);
        copy_tile_to_dst_init_short_with_dt(negative_x_imag, x_real);
        copy_tile(x_real, index, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s_imag, cb_two_tile_x);
        pack_tile_block(0, cb_two_tile_x, 4);
        tile_regs_release();
        cb_push_back(cb_two_tile_x, 4);
    }
}

// Build the column block [Xr; Xi] used as the K=2 right input of R*X.
// CB_TWO_TILE_S is recycled after this product for [Sr; Si].
void build_two_tile_x_column(
    std::uint32_t x_real,
    std::uint32_t x_imag,
    std::uint32_t block_count) {
    cb_wait_front(x_real, block_count);
    cb_wait_front(x_imag, block_count);
    cb_reserve_back(cb_two_tile_s, 2 * block_count);
    reconfig_data_format_srca(cb_two_tile_s, x_real);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        tile_regs_acquire();
        copy_tile_init(x_real);
        copy_tile(x_real, index, 0);
        copy_tile_to_dst_init_short_with_dt(x_real, x_imag);
        copy_tile(x_imag, index, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s_imag, cb_two_tile_s);
        pack_tile_block(0, cb_two_tile_s, 2);
        tile_regs_release();
        cb_push_back(cb_two_tile_s, 2);
    }
}

// Copy the two S output halves into the [Sr; Si] input block for X*S.
void build_two_tile_s_block(std::uint32_t block_count) {
    cb_wait_front(cb_s_real, block_count);
    cb_wait_front(cb_s_imag, block_count);
    cb_reserve_back(cb_two_tile_s, 2 * block_count);
    reconfig_data_format_srca(cb_two_tile_s, cb_s_real);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        tile_regs_acquire();
        copy_tile_init(cb_s_real);
        copy_tile(cb_s_real, index, 0);
        copy_tile_to_dst_init_short_with_dt(cb_s_real, cb_s_imag);
        copy_tile(cb_s_imag, index, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s_imag, cb_two_tile_s);
        pack_tile_block(0, cb_two_tile_s, 2);
        tile_regs_release();
        cb_push_back(cb_two_tile_s, 2);
    }
    cb_pop_front(cb_s_real, block_count);
    cb_pop_front(cb_s_imag, block_count);
}

// Seed the two output DEST tiles for S with BF16-hosted 2I and zero.  The
// caller owns the DEST acquisition so the copies and the K=2 matmul share one
// acquired register window.
void initialize_two_tile_s_dest(std::uint32_t block_count) {
    cb_wait_front(cb_identity, 1);
    cb_wait_front(cb_zero, 1);
    reconfig_data_format_srca(cb_two_tile_x, cb_identity);
    copy_tile_init(cb_identity);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        copy_tile(cb_identity, 0, 2 * index);
    }
    copy_tile_to_dst_init_short_with_dt(cb_identity, cb_zero);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        copy_tile(cb_zero, 0, 2 * index + 1);
    }
}

void two_tile_s_matmul_block(std::uint32_t block_count) {
    cb_wait_front(cb_two_tile_r, 4 * block_count);
    cb_wait_front(cb_two_tile_s, 2 * block_count);
    cb_reserve_back(cb_s_real, block_count);
    cb_reserve_back(cb_s_imag, block_count);

    tile_regs_acquire();
    initialize_two_tile_s_dest(block_count);
    reconfig_data_format(cb_two_tile_s, cb_two_tile_r);
    matmul_block_init(cb_two_tile_r, cb_two_tile_s, false, 1, 2, 1);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        // S = 2I - R*X, with A=[[-Rr, Ri], [-Ri, -Rr]] and B=[Xr; Xi].
        // The physical A pages are [A00, A10, A01, A11], so the two K=1
        // calls accumulate both terms into the two DEST output rows.
        matmul_block(
            cb_two_tile_r,
            cb_two_tile_s,
            4 * index,
            2 * index,
            2 * index,
            false,
            1,
            2,
            1);
        matmul_block(
            cb_two_tile_r,
            cb_two_tile_s,
            4 * index + 2,
            2 * index + 1,
            2 * index,
            false,
            1,
            2,
            1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_s_imag, cb_s_real);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        pack_tile<true>(2 * index, cb_s_real, index);
    }
    pack_reconfig_data_format(cb_s_real, cb_s_imag);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        pack_tile<true>(2 * index + 1, cb_s_imag, index);
    }
    cb_push_back(cb_s_real, block_count);
    cb_push_back(cb_s_imag, block_count);
    tile_regs_release();
}

void two_tile_x_matmul_block(
    std::uint32_t x_real,
    std::uint32_t x_imag,
    std::uint32_t block_count,
    std::uint32_t output_real,
    std::uint32_t output_imag) {
    cb_wait_front(cb_two_tile_x, 4 * block_count);
    cb_wait_front(cb_two_tile_s, 2 * block_count);
    reconfig_data_format(cb_two_tile_s, cb_two_tile_x);
    matmul_block_init(cb_two_tile_x, cb_two_tile_s, false, 1, 2, 1);
    tile_regs_acquire();
    for (std::uint32_t index = 0; index < block_count; ++index) {
        // X*S uses A=[[Xr, -Xi], [Xi, Xr]] and B=[Sr; Si].
        matmul_block(
            cb_two_tile_x,
            cb_two_tile_s,
            4 * index,
            2 * index,
            2 * index,
            false,
            1,
            2,
            1);
        matmul_block(
            cb_two_tile_x,
            cb_two_tile_s,
            4 * index + 2,
            2 * index + 1,
            2 * index,
            false,
            1,
            2,
            1);
    }
    tile_regs_commit();
    tile_regs_wait();

    // State inputs may alias state outputs. Release every input page before
    // reserving the next state/output queue window.
    cb_pop_front(x_real, block_count);
    cb_pop_front(x_imag, block_count);
    cb_pop_front(cb_negative_x_imag, block_count);
    cb_pop_front(cb_two_tile_x, 4 * block_count);
    cb_pop_front(cb_two_tile_s, 2 * block_count);
    cb_reserve_back(output_real, block_count);
    cb_reserve_back(output_imag, block_count);
    pack_reconfig_data_format(cb_s_imag, output_real);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        pack_tile<true>(2 * index, output_real, index);
    }
    pack_reconfig_data_format(output_real, output_imag);
    for (std::uint32_t index = 0; index < block_count; ++index) {
        pack_tile<true>(2 * index + 1, output_imag, index);
    }
    cb_push_back(output_real, block_count);
    cb_push_back(output_imag, block_count);
    tile_regs_release();
}

template <std::uint32_t iterations, bool state_fp32>
void process_two_tile_matrix_block(std::uint32_t block_count) {
    (void)state_fp32;
    cb_wait_front(cb_two_tile_r, 4 * block_count);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        std::uint32_t x_real;
        std::uint32_t x_imag;
        stream_initial_or_state(iteration, x_real, x_imag);
        // The existing CB_NEG_X_IMAG path supplies -Xi for the X block.
        negate_state_imag_block(x_imag, cb_zero, x_imag, block_count);
        build_two_tile_x_block(x_real, x_imag, cb_negative_x_imag, block_count);
        build_two_tile_x_column(x_real, x_imag, block_count);
        two_tile_s_matmul_block(block_count);
        cb_pop_front(cb_two_tile_s, 2 * block_count);
        build_two_tile_s_block(block_count);
        const std::uint32_t output_real =
            iteration + 1 == iterations ? cb_output_real : cb_state_real;
        const std::uint32_t output_imag =
            iteration + 1 == iterations ? cb_output_imag : cb_state_imag;
        two_tile_x_matmul_block(x_real, x_imag, block_count, output_real, output_imag);
    }
    cb_pop_front(cb_two_tile_r, 4 * block_count);
}

// The block path is selected only for matrix_block > 1. Keeping it separate
// leaves the original single-matrix path and its profiling scopes untouched.
void stream_initial_or_state(
    std::uint32_t iteration,
    std::uint32_t& x_real,
    std::uint32_t& x_imag);

template <std::uint32_t iterations, bool state_fp32, bool fuse_s, bool one_dest_half>
void process_matrix_block(std::uint32_t block_count) {
    wait_r_inputs_block<fuse_s>(block_count);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        std::uint32_t x_real;
        std::uint32_t x_imag;
        stream_initial_or_state(iteration, x_real, x_imag);

        if constexpr (state_fp32) {
            if (iteration != 0) {
                if constexpr (fuse_s) {
                    reconfig_data_format(cb_s_real, x_real, x_imag, cb_r_negative_real);
                } else {
                    reconfig_data_format(cb_s_real, x_real, x_imag, cb_r_real);
                }
            }
        }

        if constexpr (fuse_s) {
            fused_s_matmul_block<one_dest_half>(
                cb_r_negative_real,
                cb_r_negative_imag,
                cb_r_imag,
                x_real,
                x_imag,
                block_count);
        } else {
            complex_matmul_block<one_dest_half>(
                cb_r_real,
                cb_r_negative_imag,
                cb_r_imag,
                x_real,
                x_imag,
                cb_product_real,
                cb_product_imag,
                block_count,
                true,
                false,
                false);
            subtract_block(
                x_real,
                cb_r_imag,
                cb_identity,
                cb_product_real,
                cb_s_real,
                block_count,
                true,
                true);
            subtract_block(
                cb_identity,
                cb_product_real,
                cb_zero,
                cb_product_imag,
                cb_s_imag,
                block_count,
                true,
                true);
        }

        const std::uint32_t handoff_srca = fuse_s ? x_real : cb_zero;
        const std::uint32_t handoff_srcb = fuse_s ? cb_r_negative_imag : cb_product_imag;
        negate_state_imag_block(x_imag, handoff_srca, handoff_srcb, block_count);
        residual_format_transition_to_matmul(x_real, x_imag);

        const std::uint32_t output_real =
            iteration + 1 == iterations ? cb_output_real : cb_state_real;
        const std::uint32_t output_imag =
            iteration + 1 == iterations ? cb_output_imag : cb_state_imag;
        complex_matmul_block<one_dest_half>(
            x_real,
            cb_negative_x_imag,
            x_imag,
            cb_s_real,
            cb_s_imag,
            output_real,
            output_imag,
            block_count,
            false,
            true,
            true);
    }
    if constexpr (!fuse_s) {
        cb_pop_front(cb_r_real, block_count);
    }
    cb_pop_front(cb_r_negative_imag, block_count);
    cb_pop_front(cb_r_imag, block_count);
    if constexpr (fuse_s) {
        cb_pop_front(cb_r_negative_real, block_count);
    }
}

template <bool fuse_s>
void wait_r_inputs() {
    if constexpr (!fuse_s) {
        cb_wait_front(cb_r_real, 1);
    }
    cb_wait_front(cb_r_negative_imag, 1);
    cb_wait_front(cb_r_imag, 1);
    if constexpr (fuse_s) {
        cb_wait_front(cb_r_negative_real, 1);
    }
}

template <bool fuse_s>
void wait_r_inputs_profiled(ProfileCounters& counters, bool warmup) {
    DeviceZoneScopedN("NS-COMPUTE-R-CB-WAIT");
    const std::uint32_t start = get_timestamp_32b();
    wait_r_inputs<fuse_s>();
    add_profile_cycles(
        counters.r_wait,
        counters.warmup_r_wait,
        get_timestamp_32b() - start,
        warmup);
}

// The compute page is one 32x32 uint32 L1 tile: three 32-word stage slots,
// with word 31 of each slot carrying the completion magic.  Each TRISC owns
// one slot.  TRISC2 publishes the CB page only after the other two slots are
// complete, so the writer never observes a partially written page.
void clear_profile_ready(std::uint32_t start_tile) {
    if (start_tile != 0) {
        return;
    }
    volatile tt_l1_ptr std::uint32_t* profile = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_tile_address(cb_profile_compute, 0));
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    profile[profile_ready_offset] = 0;
    profile[profile_warmup_ready_offset] = 0;
#elif defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1
    profile[profile_slot_stride + profile_ready_offset] = 0;
    profile[profile_slot_stride + profile_warmup_ready_offset] = 0;
#elif defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 2
    profile[2 * profile_slot_stride + profile_ready_offset] = 0;
    profile[2 * profile_slot_stride + profile_warmup_ready_offset] = 0;
#endif
}

void write_profile_slot(
    volatile tt_l1_ptr std::uint32_t* profile,
    std::uint32_t base,
    std::uint32_t total,
    std::uint32_t warmup_total,
    std::uint32_t event_count,
    std::uint32_t warmup_event_count,
    std::uint32_t r_wait,
    std::uint32_t x_wait,
    std::uint32_t complex_real,
    std::uint32_t complex_imag,
    std::uint32_t s_binary,
    std::uint32_t pack_push,
    std::uint32_t state_handoff,
    std::uint32_t warmup_r_wait,
    std::uint32_t warmup_x_wait,
    std::uint32_t warmup_complex_real,
    std::uint32_t warmup_complex_imag,
    std::uint32_t warmup_s_binary,
    std::uint32_t warmup_pack_push,
    std::uint32_t warmup_state_handoff) {
    const std::uint32_t section_sum =
        r_wait + x_wait + complex_real + complex_imag + s_binary + pack_push + state_handoff;
    const std::uint32_t warmup_section_sum = warmup_r_wait + warmup_x_wait + warmup_complex_real +
                                               warmup_complex_imag + warmup_s_binary + warmup_pack_push +
                                               warmup_state_handoff;
    const std::uint32_t warmup_base = base + profile_warmup_base;
    profile[base + profile_total_offset] = total;
    profile[warmup_base + profile_total_offset] = warmup_total;
    profile[base + profile_r_wait_offset] = r_wait;
    profile[base + profile_x_wait_offset] = x_wait;
    profile[base + profile_complex_real_offset] = complex_real;
    profile[base + profile_complex_imag_offset] = complex_imag;
    profile[base + profile_s_binary_offset] = s_binary;
    profile[base + profile_pack_push_offset] = pack_push;
    profile[base + profile_state_handoff_offset] = state_handoff;
    profile[warmup_base + profile_r_wait_offset] = warmup_r_wait;
    profile[warmup_base + profile_x_wait_offset] = warmup_x_wait;
    profile[warmup_base + profile_complex_real_offset] = warmup_complex_real;
    profile[warmup_base + profile_complex_imag_offset] = warmup_complex_imag;
    profile[warmup_base + profile_s_binary_offset] = warmup_s_binary;
    profile[warmup_base + profile_pack_push_offset] = warmup_pack_push;
    profile[warmup_base + profile_state_handoff_offset] = warmup_state_handoff;
    profile[base + profile_section_sum_offset] = section_sum;
    profile[warmup_base + profile_section_sum_offset] = warmup_section_sum;
    profile[base + profile_residual_offset] = total - section_sum;
    profile[warmup_base + profile_residual_offset] = warmup_total - warmup_section_sum;
    profile[base + profile_event_count_offset] = event_count;
    profile[warmup_base + profile_warmup_event_count_offset] = warmup_event_count;
    profile[base + profile_ready_offset] = profile_magic;
    profile[warmup_base + profile_ready_offset] = profile_magic;
}

void write_profile_counters(ProfileCounters& counters) {
    volatile tt_l1_ptr std::uint32_t* profile = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_tile_address(cb_profile_compute, 0));
    const std::uint32_t total = counters.total_end - counters.total_start;
    const std::uint32_t warmup_total = counters.warmup_end - counters.warmup_start;
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    write_profile_slot(
        profile,
        0,
        total,
        warmup_total,
        counters.event_count,
        counters.warmup_event_count,
        counters.r_wait,
        counters.x_wait,
        0,
        0,
        0,
        0,
        0,
        counters.warmup_r_wait,
        counters.warmup_x_wait,
        0,
        0,
        0,
        0,
        0);
#elif defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1
    write_profile_slot(
        profile,
        profile_slot_stride,
        total,
        warmup_total,
        counters.event_count,
        counters.warmup_event_count,
        0,
        0,
        counters.complex_real,
        counters.complex_imag,
        counters.s_binary,
        0,
        counters.state_handoff,
        0,
        0,
        counters.warmup_complex_real,
        counters.warmup_complex_imag,
        counters.warmup_s_binary,
        0,
        counters.warmup_state_handoff);
#elif defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 2
    cb_reserve_back(cb_profile_compute, 1);
    write_profile_slot(
        profile,
        2 * profile_slot_stride,
        total,
        warmup_total,
        counters.event_count,
        counters.warmup_event_count,
        0,
        0,
        0,
        0,
        0,
        counters.pack_push,
        0,
        0,
        0,
        0,
        0,
        0,
        counters.warmup_pack_push,
        0);
    while (profile[profile_ready_offset] != profile_magic ||
           profile[profile_warmup_ready_offset] != profile_magic ||
           profile[profile_slot_stride + profile_ready_offset] != profile_magic ||
           profile[profile_slot_stride + profile_warmup_ready_offset] != profile_magic) {
    }
    cb_push_back(cb_profile_compute, 1);
#endif
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

template <bool profile_sample>
void kernel_main_impl() {
    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
    constexpr bool state_fp32 = get_compile_time_arg_val(1) != 0;
    constexpr bool fuse_s = get_compile_time_arg_val(3) != 0;
    constexpr std::uint32_t matrix_block = get_compile_time_arg_val(4);
    constexpr bool two_tile_complex = get_compile_time_arg_val(5) != 0;
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(0);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(1);
    static_assert(iterations == 8, "the throughput kernel has a fixed eight-iteration count");
    using CounterState = std::conditional_t<profile_sample, ProfileCounters, EmptyProfileCounters>;
    CounterState counters{};

    DeviceZoneScopedN("NS-COMPUTE-TOTAL");
    if constexpr (two_tile_complex) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_two_tile_r, cb_two_tile_x, cb_s_real);
        matmul_block_init(cb_two_tile_r, cb_two_tile_x, false, 1, 2, 1);
    } else if constexpr (fuse_s) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_r_negative_real, cb_x0_real, cb_product_real);
        matmul_block_init(cb_r_negative_real, cb_x0_real, false, 1, 1, 1);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_r_real, cb_x0_real, cb_product_real);
        matmul_block_init(cb_r_real, cb_x0_real, false, 1, 1, 1);
    }
    if constexpr (profile_sample) {
        clear_profile_ready(start_tile);
        if (start_tile == 0) {
            counters.total_start = get_timestamp_32b();
            counters.warmup_start = counters.total_start;
        }
    }

    if constexpr (two_tile_complex) {
        for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
            const std::uint32_t block_count =
                (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
            process_two_tile_matrix_block<iterations, state_fp32>(block_count);
            if constexpr (profile_sample) {
                if (start_tile == 0) {
                    counters.event_count += iterations;
                    if (offset == 0) {
                        counters.warmup_event_count += 1;
                        counters.warmup_end = get_timestamp_32b();
                    }
                }
            }
        }
    } else if constexpr (matrix_block == 1) {
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        if constexpr (profile_sample) {
            if (start_tile == 0) {
                wait_r_inputs_profiled<fuse_s>(counters, tile == 0);
            } else {
                wait_r_inputs<fuse_s>();
            }
        } else {
            wait_r_inputs<fuse_s>();
        }
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            std::uint32_t x_real;
            std::uint32_t x_imag;
            stream_initial_or_state(iteration, x_real, x_imag);
            const bool profile_core = profile_sample && start_tile == 0;
            const bool warmup = profile_core && tile == 0 && iteration == 0;
            if constexpr (profile_sample) {
                if (profile_core) {
                    ++counters.event_count;
                }
                if (warmup) {
                    ++counters.warmup_event_count;
                }
            }

            if constexpr (state_fp32) {
                if (iteration != 0) {
                    if constexpr (fuse_s) {
                        reconfig_data_format(cb_s_real, x_real, x_imag, cb_r_negative_real);
                    } else {
                        reconfig_data_format(cb_s_real, x_real, x_imag, cb_r_real);
                    }
                }
            }

            if constexpr (fuse_s) {
                fused_s_matmul(
                    cb_r_negative_real,
                    cb_r_negative_imag,
                    cb_r_imag,
                    x_real,
                    x_imag);
            } else if constexpr (profile_sample) {
                complex_matmul<true>(
                    cb_r_real,
                    cb_r_negative_imag,
                    cb_r_imag,
                    x_real,
                    x_imag,
                    cb_product_real,
                    cb_product_imag,
                    true,
                    false,
                    false,
                    profile_core,
                    warmup,
                    &counters);
            } else {
                complex_matmul<false>(
                    cb_r_real,
                    cb_r_negative_imag,
                    cb_r_imag,
                    x_real,
                    x_imag,
                    cb_product_real,
                    cb_product_imag,
                    true,
                    false,
                    false,
                    false);
            }

            // The last fused term leaves SrcA=X and SrcB=-R_im.  The
            // baseline binary path leaves SrcA=zero and SrcB=product_imag.
            const std::uint32_t handoff_srca = fuse_s ? x_real : cb_zero;
            const std::uint32_t handoff_srcb = fuse_s ? cb_r_negative_imag : cb_product_imag;
            if constexpr (profile_sample) {
                if (profile_core) {
                    if constexpr (!fuse_s) {
                        subtract_one_profiled(
                            x_real,
                            cb_r_imag,
                            cb_identity,
                            cb_product_real,
                            cb_s_real,
                            false,
                            true,
                            counters,
                            warmup);
                        subtract_one_profiled(
                            cb_identity,
                            cb_product_real,
                            cb_zero,
                            cb_product_imag,
                            cb_s_imag,
                            false,
                            true,
                            counters,
                            warmup);
                    }
                    state_handoff_profiled(
                        x_real,
                        x_imag,
                        handoff_srca,
                        handoff_srcb,
                        counters,
                        warmup);
                } else {
                    if constexpr (!fuse_s) {
                        subtract_one(x_real, cb_r_imag, cb_identity, cb_product_real, cb_s_real, false, true);
                        subtract_one(cb_identity, cb_product_real, cb_zero, cb_product_imag, cb_s_imag, false, true);
                    }
                    state_handoff(x_real, x_imag, handoff_srca, handoff_srcb);
                }
            } else {
                if constexpr (!fuse_s) {
                    subtract_one(x_real, cb_r_imag, cb_identity, cb_product_real, cb_s_real, false, true);
                    subtract_one(cb_identity, cb_product_real, cb_zero, cb_product_imag, cb_s_imag, false, true);
                }
                state_handoff(x_real, x_imag, handoff_srca, handoff_srcb);
            }

            const std::uint32_t output_real =
                iteration + 1 == iterations ? cb_output_real : cb_state_real;
            const std::uint32_t output_imag =
                iteration + 1 == iterations ? cb_output_imag : cb_state_imag;
            if constexpr (profile_sample) {
                complex_matmul<true>(
                    x_real,
                    cb_negative_x_imag,
                    x_imag,
                    cb_s_real,
                    cb_s_imag,
                    output_real,
                    output_imag,
                    false,
                    true,
                    true,
                    profile_core,
                    warmup,
                    &counters);
            } else {
                complex_matmul<false>(
                    x_real,
                    cb_negative_x_imag,
                    x_imag,
                    cb_s_real,
                    cb_s_imag,
                    output_real,
                    output_imag,
                    false,
                    true,
                    true,
                    false,
                    false);
            }
            if constexpr (profile_sample) {
                if (warmup) {
                    counters.warmup_end = get_timestamp_32b();
                }
            }
        }

        if constexpr (!fuse_s) {
            cb_pop_front(cb_r_real, 1);
        }
        cb_pop_front(cb_r_negative_imag, 1);
        cb_pop_front(cb_r_imag, 1);
        if constexpr (fuse_s) {
            cb_pop_front(cb_r_negative_real, 1);
        }
    }
    } else {
        for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
            const std::uint32_t block_count =
                (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
            process_matrix_block<iterations, state_fp32, fuse_s, (matrix_block == 8)>(block_count);
            if constexpr (profile_sample) {
                if (start_tile == 0) {
                    counters.event_count += iterations;
                    if (offset == 0) {
                        counters.warmup_event_count += 1;
                        counters.warmup_end = get_timestamp_32b();
                    }
                }
            }
        }
    }
    // Only the core whose range starts at tile 0 publishes the compute page;
    // all other cores retain the normal math/output path without profile CB IO.
    if constexpr (profile_sample) {
        if (start_tile == 0) {
            counters.total_end = get_timestamp_32b();
            write_profile_counters(counters);
        }
    }
}

void kernel_main() {
    constexpr bool profile_sample = get_compile_time_arg_val(2) != 0;
    if constexpr (profile_sample) {
        kernel_main_impl<true>();
    } else {
        kernel_main_impl<false>();
    }
}
