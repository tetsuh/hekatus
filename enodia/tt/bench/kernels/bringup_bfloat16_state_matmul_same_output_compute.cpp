// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr std::uint32_t cb_r = 0;
constexpr std::uint32_t cb_warmup_right = 1;
constexpr std::uint32_t cb_warmup_output = 16;
constexpr std::uint32_t cb_state_bfloat16 = 20;
constexpr std::uint32_t cb_state_float32_intermediate = 14;
constexpr std::uint32_t cb_float32_operand = 17;
constexpr std::uint32_t cb_final_output = 16;

void matmul_one(std::uint32_t left, std::uint32_t right, std::uint32_t output) {
    cb_wait_front(left, 1);
    cb_wait_front(right, 1);
    cb_reserve_back(output, 1);
    tile_regs_acquire();
    matmul_block(left, right, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output);
    tile_regs_release();
    cb_push_back(output, 1);
    cb_pop_front(left, 1);
    cb_pop_front(right, 1);
}

void convert_state_to_float32() {
    cb_wait_front(cb_state_bfloat16, 1);
    cb_reserve_back(cb_state_float32_intermediate, 1);
    pack_reconfig_data_format(cb_state_bfloat16, cb_state_float32_intermediate);
    copy_tile_init(cb_state_bfloat16);
    tile_regs_acquire();
    copy_tile(cb_state_bfloat16, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_float32_intermediate);
    tile_regs_release();
    cb_push_back(cb_state_float32_intermediate, 1);
    cb_pop_front(cb_state_bfloat16, 1);
}
}  // namespace

// The warm-up result is drained before converting state and using it as SrcA.
void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_r, cb_warmup_right, cb_warmup_output);
    matmul_block_init(cb_r, cb_warmup_right, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        matmul_one(cb_r, cb_warmup_right, cb_warmup_output);
        convert_state_to_float32();
        // Matmul maps its second CB argument to SrcA; this changes the BF16 right operand to Float32.
        reconfig_data_format_srca(cb_warmup_right, cb_float32_operand);
        matmul_block_init(cb_r, cb_float32_operand, false, 1, 1, 1);
        matmul_one(cb_r, cb_float32_operand, cb_final_output);
    }
}
