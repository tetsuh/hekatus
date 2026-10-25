// SPDX-License-Identifier: Apache-2.0
// Issue #119 wire format; mirrored by resident_tables.py.
#pragma once

namespace resident_tables {
constexpr std::uint32_t metadata_producer_cb = 22;
constexpr std::uint32_t metadata_consumer_cb = 23;
constexpr std::uint32_t tables_cb = 24;
constexpr std::uint32_t switch_stats_cb = 25;
constexpr std::uint32_t words_per_page = 1024;
constexpr std::uint32_t bytes_per_page = 4096;
constexpr std::uint32_t no_table = UINT32_MAX;

struct Identity {
    std::uint32_t config;
    std::uint32_t generation;
};

inline bool same(Identity a, Identity b) {
    return a.config == b.config && a.generation == b.generation;
}

inline std::uint32_t synthetic_word(std::uint32_t frame, std::uint32_t word) {
    const std::uint32_t i = (frame + word) & 0x07ffu;
    const std::uint32_t q = (3 * frame + 5 * word) & 0x07ffu;
    return i | (q << 16);
}

inline std::uint32_t schedule_offset(volatile tt_l1_ptr std::uint32_t* metadata) {
    return 4 + 4 * metadata[0];
}

inline std::uint32_t ready_table(
    volatile tt_l1_ptr std::uint32_t* metadata, Identity identity, std::uint32_t frame) {
    std::uint32_t result = no_table;
    for (std::uint32_t index = 0; index < metadata[0]; ++index) {
        const std::uint32_t base = 4 + 4 * index;
        if (same(identity, {metadata[base], metadata[base + 1]}) && frame >= metadata[base + 2]) {
            result = index;
        }
    }
    return result;
}

// Fixed configured page count and size, independent of table values.
template <typename Accessor>
inline void copy_table(const Accessor& tables, std::uint32_t index,
                       std::uint32_t pages, std::uint32_t destination) {
    for (std::uint32_t page = 0; page < pages; ++page) {
        noc_async_read_page(index * pages + page, tables, destination + page * bytes_per_page);
        // Bound outstanding requests independently of the swept table size.
        noc_async_read_barrier();
    }
}
}
