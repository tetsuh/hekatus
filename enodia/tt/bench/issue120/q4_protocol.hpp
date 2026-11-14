// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace issue120_q4_protocol {
constexpr std::uint32_t generation_mask = 0x7ffffu;
constexpr std::uint32_t max_challenge = 0x1fffu;
constexpr std::uint32_t challenge_shift = 19;
constexpr std::uint32_t pattern = 0x120B0000u;

constexpr std::uint32_t pack(std::uint32_t sequence, std::uint32_t challenge) {
    return (challenge << challenge_shift) | sequence;
}
constexpr std::uint32_t unpack_sequence(std::uint32_t encoded) {
    return encoded & generation_mask;
}
constexpr std::uint32_t unpack_challenge(std::uint32_t encoded) {
    return encoded >> challenge_shift;
}
constexpr std::uint32_t word(std::uint32_t encoded, std::uint32_t index) {
    return index == 0 ? encoded : encoded ^ index ^ pattern;
}
static_assert(pack(2, 2) != pack(3, 3), "generation/challenge must not cancel as XOR would");
static_assert(pack(generation_mask, max_challenge) == 0xffffffffu);
static_assert(unpack_sequence(pack(generation_mask, max_challenge)) == generation_mask);
static_assert(unpack_challenge(pack(generation_mask, max_challenge)) == max_challenge);
}
