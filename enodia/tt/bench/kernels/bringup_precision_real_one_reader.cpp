// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_a = 0;
constexpr std::uint32_t cb_b = 1;

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    const std::uint32_t a_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t b_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(2);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(3);
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, a_address);
    const auto b = TensorAccessor(b_args, b_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        read_tile(cb_a, tile, a);
        read_tile(cb_b, tile, b);
    }
}
