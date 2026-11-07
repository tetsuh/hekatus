// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "q4_protocol.hpp"

namespace issue120_q4 {
struct Clock {
    uint64_t extended = 0;
    uint32_t previous = 0;
    bool ambiguous = false;

    void initialize() {
        previous = get_timestamp_32b();
        extended = previous;
    }
    uint64_t read() {
        const uint32_t low = get_timestamp_32b();
        const uint32_t delta = low - previous;
        ambiguous |= delta >= 0x80000000u;
        extended += delta;
        previous = low;
        return extended;
    }
    void reads_complete() {
        while (!ncrisc_noc_reads_flushed(noc_index)) { read(); }
        invalidate_l1_cache();
    }
    void writes_complete() {
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) { read(); }
    }
    void atomics_complete() {
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_index)) { read(); }
    }
};
constexpr uint32_t page_bytes = 4096;
constexpr uint32_t page_words = page_bytes / sizeof(uint32_t);
constexpr uint32_t ready_id = 0;
constexpr uint32_t free_id = 1;
constexpr uint32_t start_id = 2;
constexpr uint32_t done_id = 3;
namespace protocol = issue120_q4_protocol;
constexpr uint64_t coordinate_mask = ~((uint64_t(1) << NOC_ADDR_COORD_SHIFT) - 1);

inline uint32_t frame_word(uint32_t frame, uint32_t word) {
    if (word == 0) { return frame; }
    return ((frame + word) & 0x7ffu) | (((3 * frame + 5 * word) & 0x7ffu) << 16);
}
}
