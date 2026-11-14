// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "q4_runtime_args.hpp"

namespace issue120_q4_host {
struct CoreXY { uint32_t x; uint32_t y; };
struct PeerCoordinates { CoreXY producer; CoreXY consumer; };

template <typename Translate>
PeerCoordinates resolve_peer_coordinates(CoreXY producer, CoreXY consumer, Translate translate) {
    return {translate(producer), translate(consumer)};
}

constexpr uint32_t pack_coordinate(CoreXY core) {
    return ((core.x & 0xff) << 8) | (core.y & 0xff);
}

inline uint32_t validate_single_core_accessor(const std::vector<uint32_t>& args, CoreXY core, uint32_t pages) {
    // Default compile-time, one-core L1 distribution: Sharded only, no CRTA/DRAM.
    if (args.size() < 4 || args[0] != 1 || args[1] != 4096 || args[2] < 1 || args[2] > 4 ||
        (args[3] & 0x7fffffffu) != 1 || pages == 0) {
        throw std::invalid_argument("Q4 accessor is not a static single-core sharded L1 layout");
    }
    const uint32_t rank = args[2];
    const uint32_t slot = 4 + 2 * rank;
    if (args.size() != slot + 1 || args[slot] != pack_coordinate(core)) {
        throw std::invalid_argument("Q4 accessor bank slot or virtual coordinate differs");
    }
    uint64_t tensor_pages = 1, shard_pages = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (args[4 + i] == 0 || args[4 + i] > pages || args[4 + rank + i] == 0 || args[4 + rank + i] > pages) {
            throw std::invalid_argument("Q4 accessor shape exceeds its fixed allocation");
        }
        tensor_pages *= args[4 + i];
        shard_pages *= args[4 + rank + i];
    }
    if (tensor_pages != pages || shard_pages != pages) {
        throw std::invalid_argument("Q4 accessor shape does not match its page allocation");
    }
    return slot;
}

constexpr const char* coordinate_classification(uint32_t kernel_word, CoreXY accessor, CoreXY expected) {
    if (kernel_word != pack_coordinate(expected)) { return "C"; }
    return accessor.x == expected.x && accessor.y == expected.y ? "A" : "B";
}

template <typename Resolve>
PeerCoordinates append_peer_runtime_args(std::vector<uint32_t>& producer, std::vector<uint32_t>& consumer,
                                         Resolve resolve) {
    using namespace issue120_q4_args;
    if (producer.size() != Producer::peer_x || consumer.size() != Consumer::peer_x) {
        throw std::invalid_argument("Q4 runtime argument prefix changed");
    }
    const auto coordinates = resolve();
    producer.insert(producer.end(), {coordinates.consumer.x, coordinates.consumer.y});
    consumer.insert(consumer.end(), {coordinates.producer.x, coordinates.producer.y});
    return coordinates;
}

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
