// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr std::uint32_t cb_output_real = 15;
constexpr std::uint32_t cb_output_imag = 16;
}

void kernel_main() {
    const std::uint32_t real_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t imag_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(2);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(3);
    constexpr std::uint32_t matrix_block = get_compile_time_arg_val(0);

    constexpr auto real_args = TensorAccessorArgs<1>();
    constexpr auto imag_args = TensorAccessorArgs<real_args.next_compile_time_args_offset()>();
    const auto real = TensorAccessor(real_args, real_address);
    const auto imag = TensorAccessor(imag_args, imag_address);

    {
        DeviceZoneScopedN("NS-WRITER-WRITES");
        if constexpr (matrix_block == 1) {
            for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
                const std::uint32_t tile = start_tile + offset;
                cb_wait_front(cb_output_real, 1);
                cb_wait_front(cb_output_imag, 1);
                noc_async_write_page(tile, real, get_read_ptr(cb_output_real));
                noc_async_write_barrier();
                noc_async_write_page(tile, imag, get_read_ptr(cb_output_imag));
                noc_async_write_barrier();
                cb_pop_front(cb_output_real, 1);
                cb_pop_front(cb_output_imag, 1);
            }
        } else {
            for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
                const std::uint32_t block_count =
                    (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
                cb_wait_front(cb_output_real, block_count);
                cb_wait_front(cb_output_imag, block_count);
                for (std::uint32_t index = 0; index < block_count; ++index) {
                    const std::uint32_t tile = start_tile + offset + index;
                    const std::uint32_t real_ptr =
                        get_read_ptr(cb_output_real) + index * get_tile_size(cb_output_real);
                    const std::uint32_t imag_ptr =
                        get_read_ptr(cb_output_imag) + index * get_tile_size(cb_output_imag);
                    noc_async_write_page(tile, real, real_ptr);
                    noc_async_write_page(tile, imag, imag_ptr);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_real, block_count);
                cb_pop_front(cb_output_imag, block_count);
            }
        }
    }
}
