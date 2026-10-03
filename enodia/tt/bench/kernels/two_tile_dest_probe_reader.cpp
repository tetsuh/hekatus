// SPDX-License-Identifier: Apache-2.0
// Reader for the one-call DEST-slot diagnostic.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

namespace {
constexpr std::uint32_t cb_in0 = 20;
constexpr std::uint32_t cb_in1 = 22;
constexpr std::uint32_t in0_pages_per_matrix = 4;

template <typename Accessor>
void read_page(
    std::uint32_t cb,
    std::uint32_t tile,
    const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    const std::uint32_t in0_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t in1_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(2);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(3);

    constexpr auto in0_args = TensorAccessorArgs<1>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto in0 = TensorAccessor(in0_args, in0_address);
    const auto in1 = TensorAccessor(in1_args, in1_address);

    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        cb_reserve_back(cb_in0, in0_pages_per_matrix);
        for (std::uint32_t page = 0; page < in0_pages_per_matrix; ++page) {
            noc_async_read_page(
                tile * in0_pages_per_matrix + page,
                in0,
                get_write_ptr(cb_in0) + page * get_tile_size(cb_in0));
            noc_async_read_barrier();
        }
        cb_push_back(cb_in0, in0_pages_per_matrix);
        read_page(cb_in1, tile, in1);
    }
}
