// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The host probe supplies the active CB index set.  This source deliberately
// performs no writes so it can be used as the construction-only control.
void kernel_main() {
    const std::uint32_t real_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t imag_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(2);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(3);
    constexpr auto real_args = TensorAccessorArgs<0>();
    constexpr auto imag_args = TensorAccessorArgs<real_args.next_compile_time_args_offset()>();
    const auto real = TensorAccessor(real_args, real_address);
    const auto imag = TensorAccessor(imag_args, imag_address);
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        (void)tile;
        (void)real;
        (void)imag;
    }
}
