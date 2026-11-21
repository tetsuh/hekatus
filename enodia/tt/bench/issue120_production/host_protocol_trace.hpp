// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "host_protocol.hpp"
#include "host_trace.hpp"

namespace issue120_production {
template<class Read, class Cpu, class Write, class Clock>
void submit_fixed_traced(std::uint32_t repeats, std::uint32_t frames, Traffic& facts,
                         Read&& read, Cpu&& cpu, Write&& write, HostTrace& trace, Clock&& clock) {
    // Observe the preceding fully committed prefix before the next original read.
    // The last commit is observed after return. Payload/CPU/transition code is reused.
    auto observed_read = [&] {
        if(facts.emitted != 0) trace.observe_success(facts.emitted, clock);
        return read();
    };
    submit_fixed(repeats, frames, facts, observed_read, cpu, write);
    trace.observe_success(facts.emitted, clock);
}
} // namespace issue120_production
