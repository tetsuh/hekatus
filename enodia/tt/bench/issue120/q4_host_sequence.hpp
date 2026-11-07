// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>

namespace issue120_q4_host {
struct SequenceTimes {
    uint64_t submitted;
    uint64_t joined;
    uint64_t resident_finished;
    uint64_t drain_started;
    uint64_t drained;
};

// All CQs of one MeshDevice share the API mutex. Never finish while another
// thread is still enqueueing: finish would prevent that thread from progressing.
template <typename Enqueue, typename Join, typename Finish0, typename Finish1, typename Clock>
SequenceTimes execute_sequence(Enqueue enqueue, Join join, Finish0 finish0, Finish1 finish1, Clock clock) {
    const auto submitted = clock();
    enqueue();
    join();
    const auto joined = clock();
    finish0();
    const auto resident_finished = clock();
    const auto drain_started = clock();
    finish1();
    return {submitted, joined, resident_finished, drain_started, clock()};
}

// Blackhole's NOC_XY_ADDR uses 36 local-address bits and 6 bits per coordinate.
constexpr uint64_t coordinate_mask = ~((uint64_t(1) << 36) - 1);
constexpr uint64_t start_target(uint32_t x, uint32_t y, uint32_t semaphore_base, uint32_t alignment) {
    return (uint64_t(y) << 42) | (uint64_t(x) << 36) | (semaphore_base + 2 * alignment);
}

template <typename Read>
std::array<uint32_t, 4> semaphore_words(uint32_t base, uint32_t alignment, Read read) {
    std::array<uint32_t, 4> result{};
    for (uint32_t i = 0; i < result.size(); ++i) { result[i] = read(base + i * alignment); }
    return result;
}
}
