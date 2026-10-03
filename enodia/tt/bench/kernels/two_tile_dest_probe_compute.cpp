// SPDX-License-Identifier: Apache-2.0
// One real rt=2, ct=1, kt=1 call for DEST-slot and in0-stride diagnosis.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#if __has_include("debug/dprint.h")
#include "debug/dprint.h"
#define DEST_PROBE_DPRINT(slot, cb) \
    do { \
        DPRINT << "dest_probe before pack dst_index=" << (slot) \
               << " output_cb=" << (cb) << ENDL(); \
    } while (0)
#else
#define DEST_PROBE_DPRINT(slot, cb) \
    do { \
    } while (0)
#endif

namespace {
constexpr std::uint32_t cb_in0 = 20;
constexpr std::uint32_t cb_in1 = 22;
constexpr std::uint32_t cb_output_slot0 = 15;
constexpr std::uint32_t cb_output_slot1 = 16;
constexpr std::uint32_t in0_pages_per_matrix = 4;
}

void kernel_main() {
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(0);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(1);
    (void)start_tile;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_output_slot0);
    matmul_block_init(cb_in0, cb_in1, false, 1, 2, 1);
    for (std::uint32_t matrix = 0; matrix < tile_count; ++matrix) {
        cb_wait_front(cb_in0, in0_pages_per_matrix);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_output_slot0, 1);
        cb_reserve_back(cb_output_slot1, 1);

        tile_regs_acquire();
        matmul_block(cb_in0, cb_in1, 0, 0, 0, false, 1, 2, 1);
        tile_regs_commit();
        tile_regs_wait();

        DEST_PROBE_DPRINT(0, cb_output_slot0);
        pack_reconfig_data_format(cb_output_slot0);
        pack_tile(0, cb_output_slot0);
        DEST_PROBE_DPRINT(1, cb_output_slot1);
        pack_reconfig_data_format(cb_output_slot1);
        pack_tile(1, cb_output_slot1);
        tile_regs_release();

        cb_push_back(cb_output_slot0, 1);
        cb_push_back(cb_output_slot1, 1);
        cb_pop_front(cb_in0, in0_pages_per_matrix);
        cb_pop_front(cb_in1, 1);
    }
}
