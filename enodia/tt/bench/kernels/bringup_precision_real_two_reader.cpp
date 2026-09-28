// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_first_a = 0;
constexpr std::uint32_t cb_first_b = 1;
constexpr std::uint32_t cb_product = 16;
constexpr std::uint32_t cb_second_a = 17;
constexpr std::uint32_t cb_second_b = 18;

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

void copy_float32_tile(std::uint32_t destination, std::uint32_t source_address) {
    cb_reserve_back(destination, 1);
    volatile tt_l1_ptr std::uint32_t* source =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(source_address);
    volatile tt_l1_ptr std::uint32_t* target =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_write_ptr(destination));
    for (std::uint32_t word = 0; word < (32 * 32 * 4) / sizeof(std::uint32_t); ++word) {
        target[word] = source[word];
    }
    cb_push_back(destination, 1);
}

void reuse_first_product() {
    cb_wait_front(cb_product, 1);
    copy_float32_tile(cb_second_a, get_read_ptr(cb_product));
    cb_pop_front(cb_product, 1);
}
}  // namespace

// The diagnostic computes (A_bf16 @ B_bf16)_fp32 @ C_bf16, reusing the first product as A.
void kernel_main() {
    const std::uint32_t a_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t b_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t c_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(3);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(4);
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, a_address);
    const auto b = TensorAccessor(b_args, b_address);
    const auto c = TensorAccessor(c_args, c_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        read_tile(cb_first_a, tile, a);
        read_tile(cb_first_b, tile, b);
        reuse_first_product();
        read_tile(cb_second_b, tile, c);
    }
}
