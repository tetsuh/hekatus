// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_bfloat16_input = 20;

template <typename Accessor>
void read_tile(std::uint32_t destination, std::uint32_t tile, const Accessor& accessor) {
    cb_reserve_back(destination, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(destination));
    noc_async_read_barrier();
    cb_push_back(destination, 1);
}
}  // namespace

void kernel_main() {
    const std::uint32_t input_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(1);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(2);
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(input_args, input_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        read_tile(cb_bfloat16_input, start_tile + offset, input);
    }
}
