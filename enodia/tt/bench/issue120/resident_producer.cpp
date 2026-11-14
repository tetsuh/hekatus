// SPDX-License-Identifier: Apache-2.0
// Companion fixed integer worker for the synthetic CQ0 resident pair.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint64_t window_ticks = static_cast<uint64_t>(get_arg_val<uint32_t>(0)) |
        (static_cast<uint64_t>(get_arg_val<uint32_t>(1)) << 32);
    const uint32_t work_count = get_arg_val<uint32_t>(2);
    uint32_t previous = get_timestamp_32b();
    uint64_t elapsed = 0;
    volatile uint32_t accumulator = 1;
    while (elapsed < window_ticks) {
        for (uint32_t index = 0; index < work_count; ++index) {
            accumulator = accumulator * 33u + index;
        }
        const uint32_t low = get_timestamp_32b();
        elapsed += static_cast<uint32_t>(low - previous);
        previous = low;
    }
    // Keep the fixed work observable to the compiler without a host readback.
    (void)accumulator;
}
