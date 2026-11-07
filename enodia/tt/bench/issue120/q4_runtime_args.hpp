// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace issue120_q4_args {
// Append peer coordinates; never move an existing runtime argument.
struct Producer {
    static constexpr uint32_t peer_x = 7;
    static constexpr uint32_t peer_y = 8;
    static constexpr uint32_t count = 9;
};
struct Consumer {
    static constexpr uint32_t mailbox = 11;
    static constexpr uint32_t peer_x = 12;
    static constexpr uint32_t peer_y = 13;
    static constexpr uint32_t count = 14;
};
// Identical fixed diagnostic layout appended to each final stats page.
constexpr uint32_t producer_coordinate_stats = 12;
constexpr uint32_t consumer_coordinate_stats = 25;
}
