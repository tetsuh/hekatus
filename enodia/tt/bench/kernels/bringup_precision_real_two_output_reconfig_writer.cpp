// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const std::uint32_t output_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(1);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(2);
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(output_args, output_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        cb_wait_front(19, 1);
        noc_async_write_page(tile, output, get_read_ptr(19));
        noc_async_write_barrier();
        cb_pop_front(19, 1);
    }
}
