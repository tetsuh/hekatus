// SPDX-License-Identifier: Apache-2.0
//
// Opt-in fidelity-split wrapper for newton_schulz_compute.cpp.  The legacy
// source is included unchanged so its no-split ABI and operation ordering stay
// discoverable and stable.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {

enum class SplitFidelity : std::uint32_t {
    hifi2,
    hifi3,
};

// The included implementation starts with one matmul init for hardware
// startup.  Every Newton-Schulz iteration then has exactly two operation
// boundaries (RX and X*S), including the fused and matrix-block paths.
SplitFidelity active_fidelity = SplitFidelity::hifi3;
std::uint32_t matmul_init_calls = 0;

template <MathFidelity fidelity>
void split_matmul_block_init_impl(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t transpose = 0,
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1,
    std::uint32_t call_line = __builtin_LINE()) {
    // This is the public matmul_block_init sequence with only the math LLK
    // specialization changed.  The unpacker remains paired with the same CBs.
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<fidelity, MM_THROTTLE>(
        in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
}

template <MathFidelity fidelity>
void split_matmul_block_impl(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    std::uint32_t call_line = __builtin_LINE()) {
    // Reinitialize and execute with one identical compile-time fidelity.  A
    // public matmul_block fidelity argument cannot express this pair.
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul<fidelity, MM_THROTTLE>(idst, ct_dim, rt_dim)));
}

void split_matmul_block_init(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t transpose = 0,
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1,
    std::uint32_t call_line = __builtin_LINE()) {
    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
    constexpr std::uint32_t split_iteration = get_compile_time_arg_val(5);
    static_assert(iterations == 8, "the throughput kernel has a fixed eight-iteration count");
    static_assert(split_iteration <= iterations, "fidelity split exceeds iteration count");

    const std::uint32_t init_index = matmul_init_calls++;
    const std::uint32_t operation_index = init_index == 0 ? 0 : (init_index - 1) / 2;
    const std::uint32_t iteration = operation_index % iterations;
    active_fidelity = iteration < split_iteration ? SplitFidelity::hifi2 : SplitFidelity::hifi3;

    if (active_fidelity == SplitFidelity::hifi2) {
        split_matmul_block_init_impl<MathFidelity::HiFi2>(
            in0_cb_id,
            in1_cb_id,
            transpose,
            ct_dim,
            rt_dim,
            kt_dim,
            call_line);
    } else {
        split_matmul_block_init_impl<MathFidelity::HiFi3>(
            in0_cb_id,
            in1_cb_id,
            transpose,
            ct_dim,
            rt_dim,
            kt_dim,
            call_line);
    }
}

void split_matmul_block(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    std::uint32_t call_line = __builtin_LINE()) {
    if (active_fidelity == SplitFidelity::hifi2) {
        split_matmul_block_impl<MathFidelity::HiFi2>(
            in0_cb_id,
            in1_cb_id,
            in0_tile_index,
            in1_tile_index,
            idst,
            transpose,
            ct_dim,
            rt_dim,
            kt_dim,
            call_line);
    } else {
        split_matmul_block_impl<MathFidelity::HiFi3>(
            in0_cb_id,
            in1_cb_id,
            in0_tile_index,
            in1_tile_index,
            idst,
            transpose,
            ct_dim,
            rt_dim,
            kt_dim,
            call_line);
    }
}

}  // namespace

// Intercept only the included implementation's public calls.  The wrappers
// above are direct LLK init/execute pairs; no fidelity argument is added to
// the public matmul_block API.
#define matmul_block_init(...) split_matmul_block_init(__VA_ARGS__)
#define matmul_block(...) split_matmul_block(__VA_ARGS__)
#include "newton_schulz_compute.cpp"
#undef matmul_block
#undef matmul_block_init
