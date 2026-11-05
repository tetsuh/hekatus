// SPDX-License-Identifier: Apache-2.0
// Q4 uses real ring frames and independently running fixed CPU/H2D work.
#include "q4_frame_probe.hpp"
#include "q4_protocol.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include <span>
#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using Json = nlohmann::json;
constexpr uint32_t page_bytes = 4096;
constexpr uint32_t page_words = 1024;
constexpr uint32_t frame_count = 4096;
constexpr uint32_t frame_period = 1350000;
constexpr uint32_t ring_pages = 4;
constexpr uint64_t run_budget = 13500000000ULL;
constexpr uint32_t cpu_rounds = 32;
constexpr uint32_t cpu_words = 1024;
namespace protocol = issue120_q4_protocol;
static_assert(frame_count + 2 <= protocol::max_challenge);
const CoreCoord producer_core{0, 0};
const CoreCoord consumer_core{1, 0};

uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string source(const char* filename) {
    return (std::filesystem::current_path() / ISSUE120_SOURCE_DIR / filename).string();
}

std::shared_ptr<MeshBuffer> dram(MeshDevice& mesh, uint32_t pages) {
    return MeshBuffer::create(ReplicatedBufferConfig{.size = uint64_t(pages) * page_bytes},
        DeviceLocalBufferConfig{.page_size = page_bytes, .buffer_type = BufferType::DRAM}, &mesh);
}

std::shared_ptr<MeshBuffer> l1(MeshDevice& mesh, CoreCoord core, uint32_t pages) {
    const CoreRangeSet range({CoreRange{core, core}});
    ShardSpecBuffer shard(range, {pages, page_words}, ShardOrientation::ROW_MAJOR,
                         {1, page_words}, {pages, 1});
    DeviceLocalBufferConfig local{
        .page_size = page_bytes, .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED)};
    return MeshBuffer::create(ReplicatedBufferConfig{.size = uint64_t(pages) * page_bytes}, local, &mesh);
}

void partition(MeshDevice& mesh) {
    const CoreRangeSet resident({CoreRange{producer_core, consumer_core}});
    const CoreRangeSet transfer({CoreRange{CoreCoord{2, 0}, CoreCoord{2, 0}}});
    const std::array<CoreRangeSet, 1> resident_ranges{resident};
    const std::array<CoreRangeSet, 1> transfer_ranges{transfer};
    const std::array<SubDevice, 2> sub_devices{
        SubDevice(ttsl::Span<const CoreRangeSet>(resident_ranges)),
        SubDevice(ttsl::Span<const CoreRangeSet>(transfer_ranges))};
    mesh.load_sub_device_manager(mesh.create_sub_device_manager(ttsl::Span<const SubDevice>(sub_devices), 0));
    const std::array<SubDeviceId, 1> transfer_ids{SubDeviceId{1}};
    mesh.set_sub_device_stall_group(ttsl::Span<const SubDeviceId>(transfer_ids));
}

struct Resources {
    std::shared_ptr<MeshBuffer> progress;
    std::shared_ptr<MeshBuffer> ring;
    std::shared_ptr<MeshBuffer> anchor;
    std::shared_ptr<MeshBuffer> mailbox;
    std::shared_ptr<MeshBuffer> frames;
    std::shared_ptr<MeshBuffer> producer_stats;
    std::shared_ptr<MeshBuffer> consumer_stats;
};

Resources resources(MeshDevice& mesh) {
    return {dram(mesh, 1), l1(mesh, consumer_core, ring_pages), l1(mesh, producer_core, 1),
            l1(mesh, consumer_core, 1), dram(mesh, frame_count), dram(mesh, 1), dram(mesh, 1)};
}

void append_accessor(std::vector<uint32_t>& args, const std::shared_ptr<MeshBuffer>& buffer) {
    const auto extra = TensorAccessorArgs(*buffer).get_compile_time_args();
    args.insert(args.end(), extra.begin(), extra.end());
}

void scratch(Program& program, CoreCoord core, uint32_t pages) {
    CreateCircularBuffer(program, CoreRangeSet({CoreRange{core, core}}),
        CircularBufferConfig(pages * page_bytes, {{0, tt::DataFormat::UInt32}}).set_page_size(0, page_bytes));
}

MeshWorkload workload(MeshDevice& mesh, const Resources& buffers, uint32_t repeats) {
    Program program = CreateProgram();
    const CoreRangeSet pair({CoreRange{producer_core, consumer_core}});
    for (uint32_t index = 0; index < 4; ++index) { CreateSemaphore(program, pair, 0); }
    scratch(program, producer_core, 1);
    scratch(program, consumer_core, 2);
    std::vector<uint32_t> producer_args;
    append_accessor(producer_args, buffers.ring);
    append_accessor(producer_args, buffers.producer_stats);
    std::vector<uint32_t> consumer_args;
    for (const auto& buffer : {buffers.ring, buffers.anchor, buffers.progress, buffers.frames, buffers.consumer_stats}) {
        append_accessor(consumer_args, buffer);
    }
    const auto producer = CreateKernel(program, source("q4_frame_producer.cpp"), producer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default, .compile_args = producer_args});
    const auto consumer = CreateKernel(program, source("q4_frame_consumer.cpp"), consumer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default, .compile_args = consumer_args});
    SetRuntimeArgs(program, producer, producer_core,
        {uint32_t(buffers.ring->address()), uint32_t(buffers.producer_stats->address()),
         frame_count, frame_period, ring_pages, uint32_t(run_budget), uint32_t(run_budget >> 32)});
    SetRuntimeArgs(program, consumer, consumer_core,
        {uint32_t(buffers.ring->address()), uint32_t(buffers.anchor->address()),
         uint32_t(buffers.progress->address()), uint32_t(buffers.frames->address()),
         uint32_t(buffers.consumer_stats->address()), frame_count, frame_period, ring_pages,
         repeats, uint32_t(run_budget), uint32_t(run_budget >> 32), uint32_t(buffers.mailbox->address())});
    MeshWorkload result;
    result.add_program(MeshCoordinateRange(mesh.shape()), std::move(program));
    return result;
}

struct HostLoad {
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    uint32_t checksum = 0;
    std::exception_ptr error;
};

void read_challenge(IDevice* device, uint32_t mailbox, std::span<uint8_t> destination) {
    // The mailbox is device-owned and fenced by its writer. No device-wide
    // host-write flush or CQ completion is needed for this targeted read.
    const auto core = device->virtual_core_from_logical_core(consumer_core, tt::CoreType::WORKER);
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        destination.data(), uint32_t(destination.size()), tt_cxy_pair(device->id(), core), mailbox);
}

void traffic(MeshCommandQueue& queue, const std::shared_ptr<MeshBuffer>& progress,
             IDevice* device, uint32_t mailbox, uint32_t repeats,
             std::vector<uint32_t>& payload, HostLoad& load) {
    load.start_ns = now_ns();
    volatile uint32_t checksum = 0;
    uint32_t challenge = 0;
    try {
        for (uint32_t repetition = 0; repetition < repeats; ++repetition) {
            // This direct four-byte L1 read does not finish/read a CQ. A fresh
            // card token causally binds the subsequent CPU work to its interval.
            read_challenge(device, mailbox,
                std::span<uint8_t>(reinterpret_cast<uint8_t*>(&challenge), sizeof(challenge)));
            for (uint32_t round = 0; round < cpu_rounds; ++round) {
                for (uint32_t word = 0; word < cpu_words; ++word) {
                    checksum = (checksum * 33u) ^ word;
                }
            }
            const uint32_t sequence = repetition + 1;
            if (challenge > protocol::max_challenge) {
                throw std::runtime_error("card challenge exceeds its bounded wire field");
            }
            const uint32_t encoded = protocol::pack(sequence, challenge);
            for (uint32_t word = 0; word < page_words; ++word) {
                payload[word] = protocol::word(encoded, word);
            }
            // Ordinary raw-pointer writes copy inline before returning. Reuse
            // the one staging page only after that copy, never after a CQ wait.
            queue.enqueue_write_mesh_buffer(progress, payload.data(), false);
        }
        load.end_ns = now_ns();
        load.checksum = checksum;
    } catch (...) {
        load.error = std::current_exception();
    }
}

std::vector<uint32_t> download(MeshCommandQueue& queue, const std::shared_ptr<MeshBuffer>& buffer, uint32_t pages) {
    std::vector<uint32_t> words(uint64_t(pages) * page_words);
    queue.enqueue_read_mesh_buffer(words.data(), buffer, true);
    return words;
}

uint64_t ticks(const uint32_t* words, uint32_t index) {
    return uint64_t(words[index]) | (uint64_t(words[index + 1]) << 32);
}

Json frame_rows(const std::vector<uint32_t>& words, uint32_t consumed) {
    if (consumed > frame_count) { throw std::runtime_error("invalid consumer frame count"); }
    Json rows = Json::array();
    for (uint32_t index = 0; index < consumed; ++index) {
        const auto* row = words.data() + uint64_t(index) * page_words;
        rows.push_back({
            {"frame_index", row[0]}, {"start_ticks", ticks(row, 1)}, {"end_ticks", ticks(row, 3)},
            {"left_sequence", row[5]}, {"right_sequence", row[6]},
            {"left_ticks", ticks(row, 7)}, {"right_ticks", ticks(row, 9)},
            {"progress_valid", row[11] != 0}, {"checksum", row[12]},
            {"issued_challenge", row[16]}, {"response_challenge", row[17]},
            {"payload_frame_id", row[13]}, {"work_iterations", row[14]}});
    }
    return rows;
}
}

nlohmann::json run_frame_coverage_probe(
    tt::tt_metal::distributed::MeshDevice& mesh, uint32_t repeats, bool watcher, uint32_t cap) {
    if (repeats == 0 || repeats > protocol::generation_mask) {
        throw std::invalid_argument("Q4 repeat count exceeds the unique progress wire field");
    }
    partition(mesh);
    auto buffers = resources(mesh);
    auto program = workload(mesh, buffers, repeats);
    auto& cq0 = mesh.mesh_command_queue(0);
    auto& cq1 = mesh.mesh_command_queue(1);
    std::vector<uint32_t> payload(page_words, 0);
    cq1.enqueue_write_mesh_buffer(buffers.progress, payload.data(), true);
    cq0.enqueue_write_mesh_buffer(buffers.mailbox, payload.data(), true);
    HostLoad load;
    // This scoped thread joins before any buffer, queue or device is released,
    // including exceptional exits. It has a fixed count, not a duration loop.
    auto* physical_device = mesh.get_devices().at(0);
    std::jthread load_thread([&] {
        traffic(cq1, buffers.progress, physical_device, uint32_t(buffers.mailbox->address()), repeats, payload, load);
    });
    const uint64_t submitted = now_ns();
    EnqueueMeshWorkload(cq0, program, false);
    const std::array<SubDeviceId, 1> resident_ids{SubDeviceId{0}};
    cq0.finish(ttsl::Span<const SubDeviceId>(resident_ids));
    const uint64_t resident_finished = now_ns();
    load_thread.join();
    if (load.error) { std::rethrow_exception(load.error); }
    const uint64_t drain_started = now_ns();
    const std::array<SubDeviceId, 1> transfer_ids{SubDeviceId{1}};
    cq1.finish(ttsl::Span<const SubDeviceId>(transfer_ids));
    const uint64_t drained = now_ns();
    const auto producer = download(cq0, buffers.producer_stats, 1);
    const auto consumer = download(cq0, buffers.consumer_stats, 1);
    const auto raw_frames = download(cq0, buffers.frames, frame_count);
    const Json rows = frame_rows(raw_frames, consumer[0]);
    Json evidence = {
        {"clock", "consumer_low32_wrap_tracked"},
        {"clock_gap_ambiguous", producer[4] != 0 || consumer[1] != 0},
        {"frame_count", frame_count}, {"attempted", producer[0]}, {"produced", producer[1]},
        {"consumed", consumer[0]}, {"overflow", producer[2]}, {"aborted", producer[3] + consumer[14]},
        {"producer_gated_after_load_observation", consumer[2] != 0},
        {"startup_response_challenge", consumer[15]}, {"frame_period_ticks", frame_period},
        {"load_start_ticks", ticks(consumer.data(), 3)}, {"load_end_ticks", ticks(consumer.data(), 5)},
        {"transfer_bytes", page_bytes}, {"repeat_count", repeats}, {"frames", rows},
        {"tail", {{"left_sequence", consumer[7]}, {"right_sequence", consumer[8]},
                  {"left_ticks", ticks(consumer.data(), 9)}, {"right_ticks", ticks(consumer.data(), 11)},
                  {"progress_valid", consumer[13] != 0},
                  {"issued_challenge", consumer[16]}, {"response_challenge", consumer[17]}}}};
    uint64_t first = ticks(consumer.data(), 3);
    uint64_t last = ticks(consumer.data(), 5);
    if (!rows.empty()) { first = rows.front()["start_ticks"]; last = rows.back()["end_ticks"]; }
    const uint64_t auxiliary = uint64_t(frame_count + 11) * page_bytes + 32;
    return {
        {"probe_id", "q4-frame-coverage"}, {"status", "not_evaluable"},
        {"not_evaluable_reason", "card-side frame coverage requires validation"}, {"errors", Json::array()},
        {"resident_interval", {{"start_ticks", first}, {"end_ticks", last}, {"clock", "consumer_low32_wrap_tracked"}}},
        {"resident_host_interval", {{"start_ns", submitted}, {"end_ns", resident_finished}}},
        {"host_interval", {{"start_ns", load.start_ns}, {"end_ns", load.end_ns}}},
        {"final_drain_interval", {{"start_ns", drain_started}, {"end_ns", drained}}},
        {"work", {{"repeat_count", repeats}, {"transfer_count", repeats},
                  {"bytes_per_transfer", page_bytes}, {"bytes_total", uint64_t(repeats) * page_bytes},
                  {"host_mmio_reads", repeats}, {"host_mmio_payload_bytes", uint64_t(repeats) * sizeof(uint32_t)},
                  {"payload_words_per_transfer", page_words}}},
        {"host_cpu_work", {{"rounds", cpu_rounds}, {"words_per_round", cpu_words},
                           {"operations", uint64_t(repeats) * cpu_rounds * cpu_words},
                           {"outer_repetitions", repeats}, {"checksum", load.checksum},
                           {"start_ns", load.start_ns}, {"end_ns", load.end_ns}}},
        {"memory", {{"buffer_pool_size", 1}, {"allocated_bytes", auxiliary + 2 * page_bytes},
                    {"host_payload_bytes", page_bytes}, {"device_buffer_bytes", page_bytes},
                    {"auxiliary_bytes", auxiliary}, {"max_outstanding_transfers", nullptr},
                    {"outstanding_bound", "fixed runtime ring with backpressure; occupancy not sampled"}}},
        {"memory_scope", "explicit host, DRAM, L1 and semaphore buffers; retrieval/runtime metadata excluded"},
        {"stall_group", "transfer-only"}, {"frame_coverage", evidence},
        {"coverage_scope", "all_real_frame_completion_intervals_consumer_clock"},
        {"outer_cap_seconds", cap}, {"watcher", watcher},
        {"cpu_challenge_protocol", "direct_l1_span_read_then_fixed_cpu_work_then_cq1_write"},
        {"host_buffer_mode", "ordinary_unpinned_inline_copy_before_reuse"}};
}
