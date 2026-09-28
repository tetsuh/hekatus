// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_left = 0;
constexpr std::uint32_t cb_right = 1;

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    const std::uint32_t left_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t right_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(2);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(3);
    constexpr auto left_args = TensorAccessorArgs<0>();
    constexpr auto right_args = TensorAccessorArgs<left_args.next_compile_time_args_offset()>();
    const auto left = TensorAccessor(left_args, left_address);
    const auto right = TensorAccessor(right_args, right_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        read_tile(cb_left, tile, left);
        read_tile(cb_right, tile, right);
    }
}
