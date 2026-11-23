// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "protocol.hpp"
#include <array>
#include <stdexcept>
#include <vector>

namespace issue120_production {
struct Transition { std::uint32_t first_sequence = 0, challenge = 0; };
struct Traffic {
    std::vector<Transition> transitions;
    std::array<std::uint32_t, page_words> payload{};
    std::uint32_t emitted = 0, transition_count = 0;
    explicit Traffic(std::uint32_t frames) : transitions(frames + 3) {}
};
template<class Read, class Cpu, class Write>
void submit_fixed(std::uint32_t repeats, std::uint32_t frames, Traffic& facts,
                  Read&& read, Cpu&& cpu, Write&& write) {
    if (!valid_domain(frames, repeats) || facts.transitions.size() != frames + 3 || facts.emitted != 0) {
        throw std::invalid_argument("production wire/transition capacity differs");
    }
    std::uint32_t previous = 0;
    for (std::uint32_t repetition = 0; repetition < repeats; ++repetition) {
        const std::uint32_t challenge = read();
        if (challenge < previous || challenge > frames + 2) {
            throw std::runtime_error("nonmonotonic or out-of-domain card challenge");
        }
        cpu(); // Do not read/retag the challenge after this fixed work.
        const std::uint32_t sequence = repetition + 1;
        const auto wire = encode(sequence, challenge);
        for (std::uint32_t i = 0; i < page_words; ++i) { facts.payload[i] = word(wire, i); }
        write(facts.payload); // Ordinary API copies inline before returning.
        if (facts.transition_count == 0 || challenge != previous) {
            if (facts.transition_count >= facts.transitions.size()) {
                throw std::runtime_error("transition capacity exceeded");
            }
            facts.transitions[facts.transition_count++] = {sequence, challenge};
        }
        // An enqueue failure cannot leave a phantom committed suffix.
        facts.emitted = sequence;
        previous = challenge;
    }
}
} // namespace issue120_production
