// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace issue120_production {
constexpr std::uint32_t page_words = 1024;
constexpr std::uint32_t page_bytes = 4096;
constexpr std::uint32_t pattern = 0x120C0000;
struct Wire { std::uint32_t sequence; std::uint32_t sum; };
constexpr bool valid_domain(std::uint32_t frames, std::uint32_t repeats) {
    return frames > 0 && frames <= 500000 && repeats > 0 &&
        std::uint64_t(repeats) + frames + 2 <= 0xffffffffULL;
}
// Callers admit the domain before dispatch; seed(0,0) is initialization only.
constexpr Wire encode(std::uint32_t sequence, std::uint32_t challenge) {
    return {sequence, std::uint32_t(std::uint64_t(sequence) + challenge)};
}
constexpr std::uint32_t word(Wire wire, std::uint32_t index) {
    return index == 0 ? wire.sequence : index == 1 ? wire.sum :
        ((index % 2 == 0 ? wire.sequence : wire.sum) ^ index ^ pattern);
}
constexpr std::uint32_t checksum_step(std::uint32_t accumulator, std::uint32_t frame,
                                      std::uint32_t work) {
    const std::uint32_t index = work % page_words;
    const std::uint32_t payload = ((frame + index) & 0x7ffu) |
        (((3 * frame + 5 * index) & 0x7ffu) << 16);
    return (accumulator * 33u) ^ payload ^ (work + frame);
}
namespace producer_arg { constexpr auto prefix = 10u, peer_x = 10u, peer_y = 11u; }
namespace consumer_arg {
constexpr auto progress = 15u, mailbox = 16u, repeats = 17u, period = 18u;
constexpr auto prefix = 19u, peer_x = 19u, peer_y = 20u;
}
constexpr auto start_semaphore = 4u;
static_assert(valid_domain(500000, 24439669));
static_assert(encode(24439669, 500002).sum == 24939671);
static_assert(!valid_domain(500000, 0xffffffffu));
} // namespace issue120_production
