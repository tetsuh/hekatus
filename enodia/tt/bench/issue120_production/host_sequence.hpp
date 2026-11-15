// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

namespace issue120_production {
struct XY { std::uint32_t x; std::uint32_t y; };

struct NoStop { void operator()() const {} };
template<class Start, class Enqueue, class Join, class Finish0, class Finish1, class Stop = NoStop>
void host_sequence(Start&& start, Enqueue&& enqueue, Join&& join,
                   Finish0&& finish0, Finish1&& finish1, Stop stop = {}) {
    std::exception_ptr enqueue_error;
    try { start(); enqueue(); }
    catch (...) { enqueue_error = std::current_exception(); stop(); }
    // Join callbacks leave every submitter inactive even when reporting a
    // worker error. Drain queued work before releasing its buffers/program.
    try { join(); }
    catch (...) { if (!enqueue_error) { enqueue_error = std::current_exception(); } stop(); }
    try { finish0(); finish1(); }
    catch (...) { if (!enqueue_error) { enqueue_error = std::current_exception(); } }
    if (enqueue_error) { std::rethrow_exception(enqueue_error); }
}

template<class Translate>
void append_peers(std::vector<std::uint32_t>& producer,
                  std::vector<std::uint32_t>& consumer,
                  std::size_t producer_prefix, std::size_t consumer_prefix,
                  Translate&& translate) {
    if (producer.size() != producer_prefix || consumer.size() != consumer_prefix) {
        throw std::invalid_argument("production runtime argument prefix differs");
    }
    const XY producer_xy = translate(XY{0, 0});
    const XY consumer_xy = translate(XY{1, 0});
    producer.insert(producer.end(), {consumer_xy.x, consumer_xy.y});
    consumer.insert(consumer.end(), {producer_xy.x, producer_xy.y});
}
} // namespace issue120_production
