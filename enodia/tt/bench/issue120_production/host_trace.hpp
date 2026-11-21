// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>

namespace issue120_production {
struct HostSample { std::uint64_t ns = 0; std::uint32_t completed = 0, reserved = 0; };
static_assert(sizeof(HostSample) == 16);
class HostTrace {
public:
    static constexpr std::uint32_t period = 65536, capacity = 65536;
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
    static constexpr std::uint64_t sample_storage_bytes = std::uint64_t(capacity) * sizeof(HostSample);
    HostTrace() : samples_(std::make_unique<std::array<HostSample, capacity>>()) {}
    template<class Clock> void observe_success(std::uint32_t sequence, Clock&& clock) {
        // Only this scalar is observed before join; no sample data is published.
        counter_.value.store(sequence, std::memory_order_relaxed);
        if(sequence == 1 || (sequence & (period - 1)) == 0) {
            if(sample_count_ == capacity) throw std::runtime_error("host trace capacity exceeded");
            const auto ns = clock();
            (*samples_)[sample_count_++] = {ns, sequence, 0};
        }
    }
    template<class Clock> void capture_enqueue_return(std::uint64_t before, Clock&& clock) {
        main_before_ = before;
        main_count_ = counter_.value.load(std::memory_order_relaxed);
        main_after_ = clock();
    }
    std::uint32_t published_count() const { return counter_.value.load(std::memory_order_relaxed); }
    std::uint32_t main_count() const { return main_count_; }
    std::uint64_t main_before() const { return main_before_; }
    std::uint64_t main_after() const { return main_after_; }
    // Worker-owned array/count may be inspected only after loader join.
    std::span<const HostSample> samples() const { return {samples_->data(), sample_count_}; }
    template<class Json> Json describe(bool loaded, std::uint64_t start, std::uint64_t end,
                                       std::uint64_t joined) const {
        Json out = {{"schema","issue120-host-submission-trace-v1"},{"clock","CLOCK_MONOTONIC"},
            {"applicable",loaded},{"period_writes",period},{"capacity",capacity},
            {"sample_storage_bytes",sample_storage_bytes},{"worker_start_ns",start},
            {"worker_end_ns",end},{"join_ns",joined},{"final_count",published_count()},
            {"first_committed_enqueue_observed_ns",sample_count_ ? (*samples_)[0].ns : 0},
            {"main_enqueue_return_observation",{{"before_ns",main_before_},{"after_ns",main_after_},
                {"committed_count",main_count_}}},{"samples",Json::array()}};
        for(const auto& sample : samples()) out["samples"].push_back(
            {{"monotonic_ns",sample.ns},{"committed_count",sample.completed}});
        return out;
    }
private:
    struct alignas(64) Published { std::atomic<std::uint32_t> value{0}; } counter_;
    std::unique_ptr<std::array<HostSample, capacity>> samples_;
    std::uint32_t sample_count_ = 0, main_count_ = 0;
    std::uint64_t main_before_ = 0, main_after_ = 0;
};
} // namespace issue120_production
