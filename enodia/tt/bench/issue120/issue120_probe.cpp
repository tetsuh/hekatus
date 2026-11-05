// SPDX-License-Identifier: Apache-2.0
// Probe-only native host runner for Issue #120, pinned to tt-metalium v0.75.0.
#include "q4_frame_probe.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

#ifndef ISSUE120_SOURCE_DIR
#error ISSUE120_SOURCE_DIR must be set by the probe CMake project
#endif

namespace {
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using Json = nlohmann::json;
constexpr uint32_t page_bytes = 4096;
constexpr uint32_t page_words = page_bytes / sizeof(uint32_t);
constexpr uint32_t transfer_bytes = 64 * 1024;
constexpr uint32_t marker_value = 0x120C0A17;
constexpr uint32_t observer_window_ticks_low = 2'700'000'000;
constexpr uint32_t tail_ticks_low = 2'700'000'000;
constexpr uint32_t watcher_first_run_cap_seconds = 60;
constexpr uint32_t normal_cap_seconds = 30;
constexpr uint32_t resident_x = 0;
constexpr uint32_t resident_y = 0;
constexpr uint32_t observer_x = 1;
constexpr uint32_t observer_y = 0;
constexpr uint32_t load_x = 2;
constexpr uint32_t load_y = 0;
using SteadyClock = std::chrono::steady_clock;

std::string kernel_source(const char* name) {
    return (std::filesystem::current_path() / ISSUE120_SOURCE_DIR / name).string();
}

struct Arguments {
    std::string probe;
    std::string stall_group;
    uint32_t repeat_count = 256;
    uint32_t cap_seconds = normal_cap_seconds;
    bool execute = false;
    bool watcher = false;
};

uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        SteadyClock::now().time_since_epoch()).count());
}

struct FixedCpuWork {
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t checksum;
};

FixedCpuWork run_fixed_cpu_work() {
    volatile uint32_t checksum = 0;
    const uint64_t start = monotonic_ns();
    for (uint32_t round = 0; round < 1024; ++round) {
        for (uint32_t word = 0; word < 1024; ++word) {
            checksum = ((checksum * 33u) ^ word) & 0xffffffffu;
        }
    }
    return {start, monotonic_ns(), static_cast<uint32_t>(checksum)};
}

Arguments parse_args(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        auto value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            return argv[++index];
        };
        if (name == "--execute") {
            result.execute = true;
        } else if (name == "--probe") {
            result.probe = value();
        } else if (name == "--stall-group") {
            result.stall_group = value();
        } else if (name == "--repeat-count") {
            result.repeat_count = static_cast<uint32_t>(std::stoul(value()));
        } else if (name == "--cap-seconds") {
            result.cap_seconds = static_cast<uint32_t>(std::stoul(value()));
        } else if (name == "--watcher") {
            result.watcher = value() == "1";
        } else {
            throw std::invalid_argument("unknown argument: " + name);
        }
    }
    if (!result.execute) {
        throw std::invalid_argument("native device access requires --execute");
    }
    if (result.probe.empty() || result.repeat_count == 0 || result.cap_seconds == 0 ||
        result.cap_seconds > watcher_first_run_cap_seconds) {
        throw std::invalid_argument("probe, positive repeat count, and cap <= 60 are required");
    }
    if (result.watcher && result.cap_seconds != watcher_first_run_cap_seconds) {
        throw std::invalid_argument("the first kernel run requires Watcher and a 60-second cap");
    }
    if (result.probe == "q1-cq1-completion" && result.stall_group.empty()) {
        throw std::invalid_argument("Q1 requires an explicit stall-group variant");
    }
    return result;
}

std::shared_ptr<MeshBuffer> make_dram_buffer(MeshDevice& mesh, uint32_t bytes) {
    DeviceLocalBufferConfig local{
        .page_size = page_bytes,
        .buffer_type = BufferType::DRAM,
    };
    return MeshBuffer::create(ReplicatedBufferConfig{.size = bytes}, local, &mesh);
}

void configure_q1_stall_group(MeshDevice& mesh, const std::string& variant) {
    if (variant != "resident-and-transfer" && variant != "transfer-only") {
        throw std::invalid_argument("unsupported Q1 stall-group variant");
    }
    std::vector<CoreRangeSet> resident_ranges{
        CoreRangeSet({CoreRange{CoreCoord{resident_x, resident_y}, CoreCoord{observer_x, observer_y}}})};
    std::vector<CoreRangeSet> transfer_ranges{
        CoreRangeSet({CoreRange{CoreCoord{load_x, load_y}, CoreCoord{load_x, load_y}}})};
    std::array<SubDevice, 2> sub_devices{
        SubDevice(ttsl::Span<const CoreRangeSet>(resident_ranges.data(), resident_ranges.size())),
        SubDevice(ttsl::Span<const CoreRangeSet>(transfer_ranges.data(), transfer_ranges.size()))};
    const auto manager = mesh.create_sub_device_manager(
        ttsl::Span<const SubDevice>(sub_devices.data(), sub_devices.size()), 0);
    mesh.load_sub_device_manager(manager);
    if (variant == "resident-and-transfer") {
        const std::array<SubDeviceId, 2> ids{SubDeviceId{0}, SubDeviceId{1}};
        mesh.set_sub_device_stall_group(ttsl::Span<const SubDeviceId>(ids.data(), ids.size()));
    } else {
        const SubDeviceId transfer_id{1};
        mesh.set_sub_device_stall_group(ttsl::Span<const SubDeviceId>(&transfer_id, 1));
    }
}

void add_resident_pair_kernels(
    Program& program,
    const std::shared_ptr<MeshBuffer>& marker,
    const std::shared_ptr<MeshBuffer>& stats) {
    const CoreCoord producer{resident_x, resident_y};
    const CoreCoord observer{observer_x, observer_y};
    const CoreRangeSet observer_range({CoreRange{observer, observer}});
    CreateCircularBuffer(
        program,
        observer_range,
        CircularBufferConfig(2 * page_bytes, {{0, tt::DataFormat::UInt32}}).set_page_size(0, page_bytes));
    const auto marker_args = TensorAccessorArgs(*marker).get_compile_time_args();
    const auto stats_args = TensorAccessorArgs(*stats).get_compile_time_args();
    std::vector<uint32_t> observer_compile = marker_args;
    observer_compile.insert(observer_compile.end(), stats_args.begin(), stats_args.end());
    const auto observer_kernel = CreateKernel(
        program,
        kernel_source("resident_observer.cpp"),
        observer,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = observer_compile});
    SetRuntimeArgs(
        program,
        observer_kernel,
        observer,
        {static_cast<uint32_t>(marker->address()),
         static_cast<uint32_t>(stats->address()),
         marker_value,
         transfer_bytes / page_bytes,
         observer_window_ticks_low,
         0,
         tail_ticks_low,
         0});

    const auto producer_kernel = CreateKernel(
        program,
        kernel_source("resident_producer.cpp"),
        producer,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {}});
    const uint64_t producer_ticks = static_cast<uint64_t>(observer_window_ticks_low) + tail_ticks_low;
    SetRuntimeArgs(
        program,
        producer_kernel,
        producer,
        {static_cast<uint32_t>(producer_ticks), static_cast<uint32_t>(producer_ticks >> 32), 16'384});
}

void add_resident_pair_program(
    MeshWorkload& workload,
    MeshDevice& mesh,
    const std::shared_ptr<MeshBuffer>& marker,
    const std::shared_ptr<MeshBuffer>& stats) {
    Program program = CreateProgram();
    add_resident_pair_kernels(program, marker, stats);
    workload.add_program(MeshCoordinateRange(mesh.shape()), std::move(program));
}

std::vector<uint32_t> read_stats(MeshCommandQueue& queue, const std::shared_ptr<MeshBuffer>& stats) {
    std::vector<uint32_t> words(page_words, 0);
    queue.enqueue_read_mesh_buffer(words.data(), stats, true);
    return words;
}

uint64_t join_words(const std::vector<uint32_t>& words, size_t index) {
    return static_cast<uint64_t>(words.at(index)) |
        (static_cast<uint64_t>(words.at(index + 1)) << 32);
}

std::vector<std::shared_ptr<MeshBuffer>> make_data_buffer_pool(MeshDevice& mesh, uint32_t count) {
    std::vector<std::shared_ptr<MeshBuffer>> buffers;
    buffers.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        buffers.push_back(make_dram_buffer(mesh, transfer_bytes));
    }
    return buffers;
}

Json run_host_transfer_probe(MeshDevice& mesh, const Arguments& args) {
    const bool q1 = args.probe == "q1-cq1-completion";
    if (q1) {
        configure_q1_stall_group(mesh, args.stall_group);
    }
    MeshCommandQueue& cq0 = mesh.mesh_command_queue(0);
    MeshCommandQueue& cq1 = mesh.mesh_command_queue(1);
    auto stats = make_dram_buffer(mesh, page_bytes);
    const uint32_t pool_size = 4;
    auto data_pool = make_data_buffer_pool(mesh, pool_size);
    std::vector<uint32_t> source(transfer_bytes / sizeof(uint32_t));
    for (uint32_t index = 0; index < source.size(); ++index) {
        source[index] = 0x12000000u ^ index;
    }
    for (uint32_t page = 0; page < transfer_bytes / page_bytes; ++page) {
        source[(page + 1) * page_words - 1] = marker_value ^ page;
    }
    std::vector<uint32_t> initial_payload(source.size(), 0);
    cq1.enqueue_write_mesh_buffer(data_pool[0], initial_payload.data(), true);
    std::vector<uint32_t>().swap(initial_payload);

    MeshWorkload workload;
    add_resident_pair_program(workload, mesh, data_pool[0], stats);
    const uint64_t resident_enqueue_start_ns = monotonic_ns();
    const uint32_t transfer_count = args.probe == "q1-cq1-completion" ? 1 : args.repeat_count;
    const uint64_t host_load_start_ns = monotonic_ns();
    EnqueueMeshWorkload(cq0, workload, false);
    FixedCpuWork cpu_work{host_load_start_ns, host_load_start_ns, 0};
    if (args.probe == "q4-fixed-window") {
        cpu_work = run_fixed_cpu_work();
    }
    // Injection delay is scheduling only, never completion/timing evidence.
    // A too-early payload remains a before-window or ambiguous observation.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    for (uint32_t repetition = 0; repetition < transfer_count; ++repetition) {
        const auto& buffer = data_pool[repetition % pool_size];
        cq1.enqueue_write_mesh_buffer(buffer, source.data(), false);
    }
    const uint64_t host_load_submission_end_ns = monotonic_ns();

    // The repeat loop has no completion read or payload readback. One final
    // observer/drain phase is timed separately from the submission interval.
    const uint64_t drain_start_ns = monotonic_ns();
    if (q1) {
        const SubDeviceId resident_id{0};
        cq0.finish(ttsl::Span<const SubDeviceId>(&resident_id, 1));
    } else {
        cq0.finish();
    }
    const uint64_t resident_host_end_ns = monotonic_ns();
    if (q1) {
        const SubDeviceId transfer_id{1};
        cq1.finish(ttsl::Span<const SubDeviceId>(&transfer_id, 1));
    } else {
        cq1.finish();
    }
    const uint64_t drain_end_ns = monotonic_ns();
    const auto words = read_stats(cq0, stats);
    const bool payload_seen = words.at(8) == 1;
    const uint32_t relation_code = words.at(9);
    const bool clock_ambiguous = words.at(10) != 0;
    const char* relation = relation_code == 1 ? "inside_resident_window" :
        relation_code == 2 ? "after_resident_window" :
        relation_code == 3 ? "before_resident_window" :
        relation_code == 4 ? "not_evaluable" : "not_evaluable";
    const uint64_t resident_start = join_words(words, 0);
    const uint64_t resident_end = join_words(words, 2);
    const uint64_t completion_lower = join_words(words, 4);
    const uint64_t completion_upper = join_words(words, 6);
    const uint64_t observer_end = join_words(words, 11);
    const uint64_t program_start = join_words(words, 20);
    const uint32_t data_count = transfer_count;
    const uint64_t data_bytes = static_cast<uint64_t>(data_count) * transfer_bytes;
    const uint32_t allocation_bytes = transfer_bytes + pool_size * transfer_bytes + 3 * page_bytes;
    const char* program_relation = !payload_seen || clock_ambiguous ? "not_evaluable" :
        completion_upper <= program_start ? "before_resident_program" :
        (completion_lower >= program_start && completion_upper <= observer_end)
            ? "resident_program_active" : "not_evaluable";
    const bool observer_evidence_valid = payload_seen && !clock_ambiguous;

    Json result = {
        {"probe_id", args.probe},
        {"status", observer_evidence_valid ? "complete" : "not_evaluable"},
        {"not_evaluable_reason", observer_evidence_valid ? Json(nullptr) :
            Json("payload not fully witnessed during the observer window or clock gap was ambiguous")},
        {"resident_interval", {{"start_ticks", resident_start}, {"end_ticks", resident_end},
                                {"clock", "resident_observer_low32_wrap_tracked"}}},
        {"resident_program_interval", {{"start_ticks", program_start}, {"observer_end_ticks", observer_end},
                                       {"clock", "resident_observer_low32_wrap_tracked"}}},
        {"resident_host_interval", {{"start_ns", resident_enqueue_start_ns}, {"end_ns", resident_host_end_ns}}},
        {"setup", {{"payload_buffer_zeroed", true}, {"setup_queue_id", 1},
                    {"setup_bytes", transfer_bytes}, {"setup_blocking", true},
                    {"setup_outside_load_interval", true}}},
        {"host_interval", {{"start_ns", host_load_start_ns}, {"end_ns", host_load_submission_end_ns},
                            {"scope", "submission_and_fixed_host_work_only"}}},
        {"final_drain_interval", {{"start_ns", drain_start_ns}, {"end_ns", drain_end_ns}}},
        {"work", {{"repeat_count", data_count}, {"transfer_count", data_count},
                   {"bytes_per_transfer", transfer_bytes}, {"bytes_total", data_bytes},
                   {"queue_id", 1}, {"payload_pages", transfer_bytes / page_bytes},
                   {"observer_page_reads", words.at(15)}, {"payload_words_per_page", page_words}}},
        {"memory", {{"buffer_pool_size", pool_size}, {"allocated_bytes", allocation_bytes},
                     {"host_payload_bytes", transfer_bytes},
                     {"device_buffer_bytes", pool_size * transfer_bytes},
                     {"auxiliary_bytes", 3 * page_bytes},
                     {"observer_l1_scratch_bytes", 2 * page_bytes},
                     {"max_outstanding_transfers", q1 ? Json(1) : Json(nullptr)},

                     {"outstanding_bound", q1 ? "one fixed CQ1 payload submission" :
                         "fixed CQ1 ring with enqueue backpressure; occupancy not sampled"},

                     {"host_payload_immutable", true}, {"host_payload_pinned", false}}},
        {"host_explicit_cq_reads_during_submission", false},
        {"host_enqueue_may_backpressure", !q1},
        {"device_wait_stream_may_stall", true},
        {"outer_cap_seconds", args.cap_seconds},
        {"watcher", args.watcher},
        {"errors", observer_evidence_valid ? Json::array() :
            Json::array({"payload witness or card clock evidence is unavailable"})},
        {"transfer_completion_ticks", payload_seen ? Json(completion_upper) : Json(nullptr)},
        {"transfer_completion_lower_ticks", payload_seen ? Json(completion_lower) : Json(nullptr)},
        {"transfer_completion_upper_ticks", payload_seen ? Json(completion_upper) : Json(nullptr)},
        {"transfer_completion_relation", relation},
        {"transfer_program_relation", program_relation},
        {"transfer_payload_visible", payload_seen},
        {"transfer_payload_pages_verified", payload_seen ? transfer_bytes / page_bytes : 0},
        {"clock_gap_ambiguous", clock_ambiguous},
    };
    if (q1) {
        result["stall_group"] = args.stall_group;
    }
    if (args.probe == "q2-h2d-write-only") {
        result["host_buffer_mode"] = "ordinary_unpinned_immutable_raw_host_pointer";
        result["explicit_completion_queue_reads_during_submission"] = false;
        result["completion_reads_during_final_drain"] = true;
        result["host_readback_during_submission"] = false;
        result["device_witness_reads_during_submission"] = true;
        result["device_wait_stream_may_stall"] = true;
        result["host_enqueue_may_backpressure"] = true;
        result["completion_wait_scope"] = "one_final_queue_finish_after_resident_observer";
    }
    if (args.probe == "q4-fixed-window") {
        const bool cpu_envelope_covers = cpu_work.start_ns <= resident_enqueue_start_ns &&
            cpu_work.end_ns >= resident_host_end_ns;
        result["coverage"] = cpu_envelope_covers
            ? "not_evaluable_pcie_progress_not_observed" : "not_covered";
        result["payload_witness_inside_window"] = payload_seen && relation_code == 1 && !clock_ambiguous;
        result["full_window_pcie_progress_evidence"] = false;
        result["coverage_basis"] = "fixed_payload_witness_bracket_on_resident_observer_clock";
        result["coverage_scope"] = "one_payload_witness_in_short_synthetic_observer_window_not_frame_stream";
        result["coverage_submission_count"] = transfer_count;
        result["coverage_witness_slot"] = 0;
        result["later_count_calibration"] = "precommit_a_count_and_repeat_the_bounded_probe;never_loop_until_covered";
        result["host_cpu_work"] = {{"rounds", 1024}, {"words_per_round", 1024},
                                    {"operations", 1'048'576}, {"checksum", cpu_work.checksum},
                                    {"start_ns", cpu_work.start_ns}, {"end_ns", cpu_work.end_ns}};
    }
    return result;
}

Json run_pinned_noc_probe(MeshDevice& mesh, const Arguments& args) {
    MeshCommandQueue& cq0 = mesh.mesh_command_queue(0);
    auto marker = make_dram_buffer(mesh, transfer_bytes);
    auto observer_stats = make_dram_buffer(mesh, page_bytes);
    std::vector<uint32_t> empty_marker(transfer_bytes / sizeof(uint32_t), 0);
    cq0.enqueue_write_mesh_buffer(marker, empty_marker.data(), true);
    std::vector<uint32_t>().swap(empty_marker);
    constexpr uint32_t host_buffer_bytes = 3 * page_bytes;
    std::shared_ptr<void> host_storage(
        std::aligned_alloc(page_bytes, host_buffer_bytes), std::free);
    if (!host_storage) {
        throw std::bad_alloc();
    }
    auto* host_words = static_cast<uint32_t*>(host_storage.get());
    std::fill_n(host_words, host_buffer_bytes / sizeof(uint32_t), 0);
    for (uint32_t index = 0; index < page_words; ++index) {
        host_words[index] = 0x1200A000u ^ index;
    }
    HostBuffer host_buffer(
        ttsl::Span<uint32_t>(host_words, host_buffer_bytes / sizeof(uint32_t)),
        MemoryPin(host_storage));
    const auto parameters = experimental::GetMemoryPinningParameters(mesh);
    const long system_page_size = sysconf(_SC_PAGESIZE);
    if (system_page_size <= 0) {
        throw std::runtime_error("host page size is unavailable for pinned-memory accounting");
    }
    const auto host_address = reinterpret_cast<uintptr_t>(host_buffer.view_bytes().data());
    const size_t host_offset = host_address % static_cast<size_t>(system_page_size);
    const size_t required_pin_bytes = host_buffer_bytes + host_offset;
    if (parameters.max_pins < 1 || parameters.max_total_pin_size < required_pin_bytes) {
        throw std::runtime_error("pinned memory capacity preflight rejects the aligned host allocation");
    }
    if (!parameters.can_map_to_noc) {
        throw std::runtime_error("pinned memory preflight reports NoC mapping unsupported");
    }
    auto pinned = experimental::PinnedMemory::Create(
        mesh,
        MeshCoordinateRangeSet(MeshCoordinateRange(mesh.shape())),
        host_buffer,
        true);
    const auto device_ids = mesh.get_device_ids();
    if (device_ids.size() != 1) {
        throw std::runtime_error("Q3 requires one MMIO device in the unit mesh");
    }
    const auto noc_addr = pinned->get_noc_addr(device_ids.at(0));
    if (!noc_addr.has_value()) {
        throw std::runtime_error("pinned host allocation has no NOC address for device 0");
    }
    if (!pinned->has_device(device_ids.at(0)) || noc_addr->device_id != device_ids.at(0) ||
        pinned->get_buffer_size() < host_buffer_bytes) {
        throw std::runtime_error("pinned allocation does not cover the selected MMIO device and full host range");
    }

    Program program = CreateProgram();
    add_resident_pair_kernels(program, marker, observer_stats);
    const CoreCoord load_core{load_x, load_y};
    const CoreRangeSet load_range({CoreRange{load_core, load_core}});
    CreateCircularBuffer(
        program,
        load_range,
        CircularBufferConfig(2 * page_bytes, {{0, tt::DataFormat::UInt32}}).set_page_size(0, page_bytes));
    const auto kernel = CreateKernel(
        program,
        kernel_source("pinned_noc_load.cpp"),
        load_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {}});
    SetRuntimeArgs(
        program,
        kernel,
        load_core,
        {noc_addr->pcie_xy_enc,
         static_cast<uint32_t>(noc_addr->addr),
         static_cast<uint32_t>(noc_addr->addr >> 32),
         args.repeat_count});
    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(mesh.shape()), std::move(program));
    const uint64_t resident_submit_ns = monotonic_ns();
    EnqueueMeshWorkload(cq0, workload, false);
    const FixedCpuWork cpu_work = run_fixed_cpu_work();
    const uint64_t drain_start_ns = monotonic_ns();
    cq0.finish();
    const uint64_t host_end_ns = monotonic_ns();

    void* locked_host = pinned->lock();
    const auto* output = static_cast<const uint32_t*>(locked_host);
    const size_t stats_offset = 2 * page_bytes / sizeof(uint32_t);
    const std::array<uint32_t, 9> load_words{
        output[stats_offset], output[stats_offset + 1], output[stats_offset + 2],
        output[stats_offset + 3], output[stats_offset + 4], output[stats_offset + 5],
        output[stats_offset + 6], output[stats_offset + 7], output[stats_offset + 8]};
    const uint64_t expected_checksum = [&]() {
        uint32_t checksum = 0;
        for (uint32_t repetition = 0; repetition < args.repeat_count; ++repetition) {
            checksum ^= (0x1200A000u ^ (repetition % page_words)) ^ repetition;
        }
        return checksum;
    }();
    const auto* final_host = static_cast<const uint32_t*>(locked_host);
    const size_t output_page = page_bytes / sizeof(uint32_t);
    bool read_payload_matches = true;
    for (size_t word = 0; word < output_page; ++word) {
        read_payload_matches &= final_host[word] == (0x1200A000u ^ word);
    }
    bool write_payload_matches = final_host[output_page] == expected_checksum;
    for (size_t word = 1; word < output_page; ++word) {
        write_payload_matches &= final_host[output_page + word] == (0x1200A000u ^ word);
    }
    const bool counters_match = load_words[4] == args.repeat_count &&
        load_words[5] == args.repeat_count && load_words[6] == page_bytes &&
        load_words[7] == expected_checksum;
    const bool load_observation_valid = read_payload_matches && write_payload_matches && counters_match;
    pinned->unlock();
    const auto observer_words = read_stats(cq0, observer_stats);
    const uint64_t resident_start_ticks = join_words(observer_words, 0);
    const uint64_t resident_end_ticks = join_words(observer_words, 2);
    const uint64_t load_start_ticks = static_cast<uint64_t>(load_words[0]) |
        (static_cast<uint64_t>(load_words[1]) << 32);
    const uint64_t load_end_ticks = static_cast<uint64_t>(load_words[2]) |
        (static_cast<uint64_t>(load_words[3]) << 32);
    const uint64_t bytes_each_direction = static_cast<uint64_t>(args.repeat_count) * page_bytes;
    Json result = {
        {"probe_id", args.probe},
        {"status", load_observation_valid && load_words[8] == 0 && observer_words.at(10) == 0
                       ? "complete" : "not_evaluable"},
        {"not_evaluable_reason", load_observation_valid && load_words[8] == 0 && observer_words.at(10) == 0
                       ? Json(nullptr) : Json("pinned payload/counter verification or continuous card-clock sampling failed")},

        {"resident_interval", {{"start_ticks", resident_start_ticks}, {"end_ticks", resident_end_ticks},
                                {"clock", "resident_observer_low32_wrap_tracked"}}},
        {"card_load_interval", {{"start_ticks", load_start_ticks}, {"end_ticks", load_end_ticks},
                                {"clock", "load_worker_low32_wrap_tracked"}}},
        {"resident_host_interval", {{"start_ns", resident_submit_ns}, {"end_ns", host_end_ns}}},
        {"setup", {{"observer_payload_buffer_zeroed", true}, {"setup_queue_id", 0},
                    {"setup_bytes", transfer_bytes}, {"setup_blocking", true},
                    {"setup_outside_load_interval", true}}},
        {"host_interval", {{"start_ns", cpu_work.start_ns}, {"end_ns", cpu_work.end_ns},
                            {"scope", "fixed_host_cpu_work_during_pinned_noc_probe"}}},
        {"final_drain_interval", {{"start_ns", drain_start_ns}, {"end_ns", host_end_ns}}},
        {"work", {{"repeat_count", args.repeat_count}, {"transfer_count", 2ULL * args.repeat_count},
                   {"bytes_per_transfer", page_bytes}, {"bytes_total", 2 * bytes_each_direction},
                   {"read_bytes", bytes_each_direction}, {"write_bytes", bytes_each_direction},
                   {"worker_core", {load_x, load_y}}, {"load_worker_reads", load_words[4]},
                   {"load_worker_writes", load_words[5]}, {"clock_gap_ambiguous", load_words[8] != 0}}},
        {"memory", {{"buffer_pool_size", 1}, {"allocated_bytes", host_buffer_bytes + 21 * page_bytes},
                     {"host_payload_bytes", host_buffer_bytes}, {"device_buffer_bytes", 0},
                     {"auxiliary_bytes", 21 * page_bytes},
                     {"max_outstanding_transfers", 1}, {"pinned_host_bytes", host_buffer_bytes},
                     {"device_l1_scratch_bytes", 4 * page_bytes},
                     {"device_dram_buffers_bytes", 17 * page_bytes}}},


        {"outer_cap_seconds", args.cap_seconds},
        {"watcher", args.watcher},
        {"errors", load_observation_valid && load_words[8] == 0 && observer_words.at(10) == 0
                       ? Json::array() : Json::array({"pinned payload/counter verification or card-clock sampling failed"})},

        {"host_cpu_work", {{"rounds", 1024}, {"words_per_round", 1024},
                            {"operations", 1'048'576}, {"checksum", cpu_work.checksum},
                            {"start_ns", cpu_work.start_ns}, {"end_ns", cpu_work.end_ns}}},
        {"host_readback_during_submission", false},
        {"host_explicit_cq_reads_during_submission", false},

        {"load_payload_validation", {{"valid", load_observation_valid},
                                     {"read_payload_matches", read_payload_matches},
                                     {"write_payload_matches", write_payload_matches},
                                     {"counters_match", counters_match}}},
        {"noc_preflight", {{"max_pins", parameters.max_pins},
                           {"requested_pin_bytes", host_buffer_bytes},
                           {"page_alignment_offset_bytes", host_offset},
                           {"required_aligned_pin_bytes", required_pin_bytes},
                           {"pinned_buffer_bytes", pinned->get_buffer_size()},

                           {"device_id", device_ids.at(0)},
                           {"noc_address_device_id", noc_addr->device_id},
                           {"max_total_pin_size", parameters.max_total_pin_size},
                           {"can_map_to_noc", parameters.can_map_to_noc},
                           {"usable_from_noc", pinned->usable_from_noc(device_ids.at(0))},
                           {"noc_address_available", true}}},
        {"cross_core_tick_comparison", "not_assumed"},
        {"coverage", "not_evaluable_cross_core_ticks_not_compared"},
        {"coverage_basis", "fixed_q3_worker_and_resident_pair_share_one_cq0_program"},
        {"coverage_scope", "short_mechanism_probe_only;not_frame_coverage"},
        {"later_count_calibration", "precommit_repeat_count_and_repeat_bounded_probe;never_loop_until_covered"},
    };
    return result;
}

Json run_probe(const Arguments& args) {
    if (args.probe != "q1-cq1-completion" && args.probe != "q2-h2d-write-only" &&
        args.probe != "q3-pinned-noc" && args.probe != "q4-fixed-window" &&
        args.probe != "q4-frame-coverage") {
        throw std::invalid_argument("unknown Issue #120 probe");
    }
    auto mesh = MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 2);
    Json result;
    if (args.probe == "q4-frame-coverage") {
        result = run_frame_coverage_probe(*mesh, args.repeat_count, args.watcher, args.cap_seconds);
    } else if (args.probe == "q3-pinned-noc") {
        result = run_pinned_noc_probe(*mesh, args);
    } else {
        result = run_host_transfer_probe(*mesh, args);
    }
    mesh->close();
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        result["host_process_max_rss_kib"] = usage.ru_maxrss;
    }
    result["memory_scope"] = "explicit_payload_dram_and_cb_buffers;runtime_rings_and_allocator_metadata_excluded";
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        const char* pinned = std::getenv("HEKATUS_TT_PINNED_CONTAINER");
        if (pinned == nullptr || std::string(pinned) != "1") {
            throw std::invalid_argument("device access requires the pinned-container wrapper");
        }
        const char* watcher = std::getenv("TT_METAL_WATCHER");
        const bool watcher_enabled = watcher != nullptr && std::string(watcher) == "1";
        if (watcher_enabled != args.watcher) {
            throw std::invalid_argument("Watcher CLI and environment must agree");
        }
        const Json result = run_probe(args);
        std::cout << result.dump() << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Issue #120 probe failed: " << error.what() << std::endl;
        return 2;
    }
}
