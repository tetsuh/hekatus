// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_r = 0;
constexpr std::uint32_t cb_warmup_right = 1;
constexpr std::uint32_t cb_warmup_output = 16;
constexpr std::uint32_t cb_writer_ready = 13;
constexpr std::uint32_t cb_state_bfloat16 = 20;
constexpr std::uint32_t cb_state_float32_intermediate = 14;
constexpr std::uint32_t cb_float32_operand = 17;

template <typename Accessor>
void read_tile(std::uint32_t destination, std::uint32_t tile, const Accessor& accessor) {
    cb_reserve_back(destination, 1);
    noc_async_read_page(tile, accessor, get_write_ptr(destination));
    noc_async_read_barrier();
    cb_push_back(destination, 1);
}

void discard_warmup_output() {
    cb_wait_front(cb_warmup_output, 1);
    cb_pop_front(cb_warmup_output, 1);
    cb_reserve_back(cb_writer_ready, 1);
    cb_push_back(cb_writer_ready, 1);
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

void route_converted_state() {
    cb_wait_front(cb_state_float32_intermediate, 1);
    copy_float32_tile(cb_float32_operand, get_read_ptr(cb_state_float32_intermediate));
    cb_pop_front(cb_state_float32_intermediate, 1);
}
}  // namespace

// Queue two R tiles, discard the warm-up result, then route the converted state.
void kernel_main() {
    const std::uint32_t r_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t warmup_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t state_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(3);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(4);
    constexpr auto r_args = TensorAccessorArgs<0>();
    constexpr auto warmup_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<warmup_args.next_compile_time_args_offset()>();
    const auto r = TensorAccessor(r_args, r_address);
    const auto warmup = TensorAccessor(warmup_args, warmup_address);
    const auto state = TensorAccessor(state_args, state_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        read_tile(cb_r, tile, r);
        read_tile(cb_r, tile, r);
        read_tile(cb_warmup_right, tile, warmup);
        discard_warmup_output();
        read_tile(cb_state_bfloat16, tile, state);
        route_converted_state();
    }
}
