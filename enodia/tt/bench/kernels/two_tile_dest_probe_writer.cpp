// SPDX-License-Identifier: Apache-2.0
// Write the two DEST slots through independent output CBs.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

namespace {
constexpr std::uint32_t cb_output_slot0 = 15;
constexpr std::uint32_t cb_output_slot1 = 16;
}

void kernel_main() {
    const std::uint32_t slot0_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t slot1_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(2);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(3);

    constexpr auto slot0_args = TensorAccessorArgs<1>();
    constexpr auto slot1_args = TensorAccessorArgs<slot0_args.next_compile_time_args_offset()>();
    const auto slot0 = TensorAccessor(slot0_args, slot0_address);
    const auto slot1 = TensorAccessor(slot1_args, slot1_address);

    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        cb_wait_front(cb_output_slot0, 1);
        cb_wait_front(cb_output_slot1, 1);
        noc_async_write_page(tile, slot0, get_read_ptr(cb_output_slot0));
        noc_async_write_page(tile, slot1, get_read_ptr(cb_output_slot1));
        noc_async_write_barrier();
        cb_pop_front(cb_output_slot0, 1);
        cb_pop_front(cb_output_slot1, 1);
    }
}
