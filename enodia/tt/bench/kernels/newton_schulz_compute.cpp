// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
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
constexpr std::uint32_t cb_output_real = 15;
constexpr std::uint32_t cb_output_imag = 16;

void pack_one(std::uint32_t output) {
    tile_regs_wait();
    pack_tile(0, output);
    tile_regs_release();
    cb_push_back(output, 1);
}

void pack_one_profiled(std::uint32_t output) {
    DeviceZoneScopedN("NS-COMPUTE-PACK-PUSH");
    pack_one(output);
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
    bool resident_left) {
    DeviceZoneScopedN("NS-COMPUTE-X-CB-WAIT");
    wait_complex_inputs(
        left_real,
        left_imag_for_real,
        left_imag_for_imag,
        right_real,
        right_imag,
        resident_left);
}

void complex_real_impl(
    std::uint32_t left_real,
    std::uint32_t left_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output,
    bool profile_pack) {
    cb_reserve_back(output, 1);
    tile_regs_acquire();
    matmul_block(left_real, right_real, 0, 0, 0, false, 1, 1, 1);
    matmul_block(left_imag, right_imag, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    if (profile_pack) {
        pack_reconfig_data_format(output);
        pack_one_profiled(output);
    } else {
        pack_reconfig_data_format(output);
        pack_one(output);
    }
}

void complex_real_profiled(
    std::uint32_t left_real,
    std::uint32_t left_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output) {
    DeviceZoneScopedN("NS-COMPUTE-COMPLEX-REAL");
    complex_real_impl(left_real, left_imag, right_real, right_imag, output, true);
}

void complex_imag_impl(
    std::uint32_t left_real,
    std::uint32_t left_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output,
    bool profile_pack) {
    cb_reserve_back(output, 1);
    tile_regs_acquire();
    matmul_block(left_real, right_imag, 0, 0, 0, false, 1, 1, 1);
    matmul_block(left_imag, right_real, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    if (profile_pack) {
        pack_reconfig_data_format(output);
        pack_one_profiled(output);
    } else {
        pack_reconfig_data_format(output);
        pack_one(output);
    }
}

void complex_imag_profiled(
    std::uint32_t left_real,
    std::uint32_t left_imag,
    std::uint32_t right_real,
    std::uint32_t right_imag,
    std::uint32_t output) {
    DeviceZoneScopedN("NS-COMPUTE-COMPLEX-IMAG");
    complex_imag_impl(left_real, left_imag, right_real, right_imag, output, true);
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
    bool sample) {
    // With SrcOrder::Reverse the second CB is SrcA and the first CB is SrcB.
    // Each half keeps both signed terms in one acquired DEST tile.  Order 1
    // deliberately uses one DEST section per half; this fidelity sweep must
    // not mix the later two-DEST experiment into its timings.
    matmul_block_init(left_real, right_real, false, 1, 1, 1);
    if constexpr (profile_sample) {
        if (sample) {
            wait_complex_inputs_profiled(
                left_real,
                left_imag_for_real,
                left_imag_for_imag,
                right_real,
                right_imag,
                resident_left);
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
        if (sample) {
            complex_real_profiled(left_real, left_imag_for_real, right_real, right_imag, output_real);
        } else {
            complex_real_impl(left_real, left_imag_for_real, right_real, right_imag, output_real, false);
        }
    } else {
        complex_real_impl(left_real, left_imag_for_real, right_real, right_imag, output_real, false);
    }

    if constexpr (profile_sample) {
        if (sample) {
            complex_imag_profiled(left_real, left_imag_for_imag, right_real, right_imag, output_imag);
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

void subtract_one_impl(
    std::uint32_t current_srca,
    std::uint32_t current_srcb,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t output,
    bool consume_left,
    bool consume_right,
    bool profile_pack) {
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
        pack_one_profiled(output);
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
    bool consume_right) {
    DeviceZoneScopedN("NS-COMPUTE-S-BINARY");
    subtract_one_impl(current_srca, current_srcb, left, right, output, consume_left, consume_right, true);
}

void negate_state_imag_impl(std::uint32_t x_imag, bool profile_pack) {
    cb_wait_front(x_imag, 1);
    cb_wait_front(cb_zero, 1);
    cb_reserve_back(cb_negative_x_imag, 1);
    reconfig_data_format(cb_zero, cb_zero, cb_product_imag, x_imag);
    pack_reconfig_data_format(cb_negative_x_imag);
    sub_tiles_init(cb_zero, x_imag);
    tile_regs_acquire();
    sub_tiles(cb_zero, x_imag, 0, 0, 0);
    tile_regs_commit();
    if (profile_pack) {
        pack_one_profiled(cb_negative_x_imag);
    } else {
        pack_one(cb_negative_x_imag);
    }
}

void residual_format_transition_to_matmul(std::uint32_t x_real, std::uint32_t x_imag) {
    reconfig_data_format(cb_zero, cb_s_real, x_imag, x_real);
}

void state_handoff(std::uint32_t x_real, std::uint32_t x_imag) {
    negate_state_imag_impl(x_imag, false);
    residual_format_transition_to_matmul(x_real, x_imag);
}

void state_handoff_profiled(std::uint32_t x_real, std::uint32_t x_imag) {
    DeviceZoneScopedN("NS-COMPUTE-STATE-HANDOFF");
    negate_state_imag_impl(x_imag, true);
    residual_format_transition_to_matmul(x_real, x_imag);
}

void wait_r_inputs() {
    cb_wait_front(cb_r_real, 1);
    cb_wait_front(cb_r_negative_imag, 1);
    cb_wait_front(cb_r_imag, 1);
}

void wait_r_inputs_profiled() {
    DeviceZoneScopedN("NS-COMPUTE-R-CB-WAIT");
    wait_r_inputs();
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
    constexpr bool profile_sample = get_compile_time_arg_val(2) != 0;
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(0);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(1);
    static_assert(iterations == 8, "the throughput kernel has a fixed eight-iteration count");

    DeviceZoneScopedN("NS-COMPUTE-TOTAL");
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_r_real, cb_x0_real, cb_product_real);
    matmul_block_init(cb_r_real, cb_x0_real, false, 1, 1, 1);

    (void)start_tile;
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        if constexpr (profile_sample) {
            if (tile == 0) {
                wait_r_inputs_profiled();
            } else {
                wait_r_inputs();
            }
        } else {
            wait_r_inputs();
        }
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            std::uint32_t x_real;
            std::uint32_t x_imag;
            stream_initial_or_state(iteration, x_real, x_imag);
            const bool sample = profile_sample && tile == 0 && iteration == 0;

            if constexpr (state_fp32) {
                if (iteration != 0) {
                    reconfig_data_format(cb_s_real, x_real, x_imag, cb_r_real);
                }
            }

            if constexpr (profile_sample) {
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
                    sample);
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

            if constexpr (profile_sample) {
                if (sample) {
                    subtract_one_profiled(x_real, cb_r_imag, cb_identity, cb_product_real, cb_s_real, false, true);
                    subtract_one_profiled(cb_identity, cb_product_real, cb_zero, cb_product_imag, cb_s_imag, false, true);
                    state_handoff_profiled(x_real, x_imag);
                } else {
                    subtract_one(x_real, cb_r_imag, cb_identity, cb_product_real, cb_s_real, false, true);
                    subtract_one(cb_identity, cb_product_real, cb_zero, cb_product_imag, cb_s_imag, false, true);
                    state_handoff(x_real, x_imag);
                }
            } else {
                subtract_one(x_real, cb_r_imag, cb_identity, cb_product_real, cb_s_real, false, true);
                subtract_one(cb_identity, cb_product_real, cb_zero, cb_product_imag, cb_s_imag, false, true);
                state_handoff(x_real, x_imag);
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
                    sample);
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
                    false);
            }
        }

        cb_pop_front(cb_r_real, 1);
        cb_pop_front(cb_r_negative_imag, 1);
        cb_pop_front(cb_r_imag, 1);
    }
}
