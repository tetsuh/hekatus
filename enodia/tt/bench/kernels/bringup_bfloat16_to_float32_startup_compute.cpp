// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr std::uint32_t cb_bfloat16_input = 20;
constexpr std::uint32_t cb_float32_output = 23;

void convert_tile() {
    cb_wait_front(cb_bfloat16_input, 1);
    cb_reserve_back(cb_float32_output, 1);
    pack_reconfig_data_format(cb_bfloat16_input, cb_float32_output);
    copy_tile_init(cb_bfloat16_input);
    tile_regs_acquire();
    copy_tile(cb_bfloat16_input, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_float32_output);
    tile_regs_release();
    cb_push_back(cb_float32_output, 1);
    cb_pop_front(cb_bfloat16_input, 1);
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_bfloat16_input, cb_bfloat16_input, cb_float32_output);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        convert_tile();
    }
}
