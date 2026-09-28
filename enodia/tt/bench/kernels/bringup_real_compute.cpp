// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr std::uint32_t cb_left = 0;
constexpr std::uint32_t cb_right = 1;
constexpr std::uint32_t cb_out_real = 14;

void matmul_one() {
    cb_wait_front(cb_left, 1);
    cb_wait_front(cb_right, 1);
    cb_reserve_back(cb_out_real, 1);
    tile_regs_acquire();
    matmul_block(cb_left, cb_right, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_real);
    tile_regs_release();
    cb_push_back(cb_out_real, 1);
    cb_pop_front(cb_left, 1);
    cb_pop_front(cb_right, 1);

}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(0);
    mm_block_init(cb_left, cb_right, cb_out_real, false, 1, 1, 1);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        matmul_one();
    }
}
