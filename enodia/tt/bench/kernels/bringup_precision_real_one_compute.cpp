// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"

namespace {
constexpr std::uint32_t cb_a = 0;
constexpr std::uint32_t cb_b = 1;
constexpr std::uint32_t cb_product = 16;

void matmul_one() {
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_product, 1);
    tile_regs_acquire();
    matmul_block(cb_a, cb_b, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_product);
    tile_regs_release();
    cb_push_back(cb_product, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_product);
    matmul_block_init(cb_a, cb_b, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        matmul_one();
    }
}
