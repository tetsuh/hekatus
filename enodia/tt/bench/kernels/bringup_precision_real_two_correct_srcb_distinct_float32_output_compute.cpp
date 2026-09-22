// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

namespace {
constexpr std::uint32_t cb_first_a = 0;
constexpr std::uint32_t cb_first_b = 1;
constexpr std::uint32_t cb_product = 16;
constexpr std::uint32_t cb_second_a = 17;
constexpr std::uint32_t cb_second_b = 18;
constexpr std::uint32_t cb_second_product = 19;

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
}  // namespace

// Correctly reconfigure SrcB, then switch to a distinct Float32 output CB.
void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_first_a, cb_first_b, cb_product);
    matmul_block_init(cb_first_a, cb_first_b, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        matmul_one(cb_first_a, cb_first_b, cb_product);
        reconfig_data_format_srcb(cb_first_a, cb_second_a);
        matmul_block_init(cb_second_a, cb_second_b, false, 1, 1, 1);
        pack_reconfig_data_format(cb_product, cb_second_product);
        matmul_one(cb_second_a, cb_second_b, cb_second_product);
    }
}
