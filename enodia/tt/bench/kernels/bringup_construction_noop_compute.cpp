// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/common.h"

// The host probe supplies the active CB index set.  This source deliberately
// performs no compute work; the compile-time argument shape remains [1].
void kernel_main() {
    constexpr std::uint32_t tiles_per_core = get_compile_time_arg_val(0);
    (void)tiles_per_core;
}
