// SPDX-License-Identifier: Apache-2.0
// Q4 uses real ring frames and independently running fixed CPU/H2D work.
#include "q4_frame_probe.hpp"
#include "q4_protocol.hpp"
#include "q4_host_sequence.hpp"
#include "distributed/mesh_workload_impl.hpp"
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

struct RoutingRecord {
    issue120_q4_host::PeerCoordinates passed;
    std::vector<uint32_t> producer_ring_compile_args;
    std::vector<uint32_t> consumer_anchor_compile_args;
    uint32_t consumer_anchor_bank_slot = 0;
};

issue120_q4_host::PeerCoordinates peer_coordinates(IDevice* device) {
    using issue120_q4_host::CoreXY;
    return issue120_q4_host::resolve_peer_coordinates(
        CoreXY{producer_core.x, producer_core.y}, CoreXY{consumer_core.x, consumer_core.y},
        [device](CoreXY logical) {
            const auto core = device->virtual_core_from_logical_core({logical.x, logical.y}, tt::CoreType::WORKER);
            return CoreXY{core.x, core.y};
        });
}

void append_accessor(std::vector<uint32_t>& args, const std::shared_ptr<MeshBuffer>& buffer,
                     std::vector<uint32_t>* captured = nullptr) {
    const auto extra = TensorAccessorArgs(*buffer).get_compile_time_args();
    if (extra.empty()) { throw std::runtime_error("empty Q4 accessor compile arguments"); }
    if (captured != nullptr) { *captured = extra; }
    args.insert(args.end(), extra.begin(), extra.end());
}

void scratch(Program& program, CoreCoord core, uint32_t pages) {
    CreateCircularBuffer(program, CoreRangeSet({CoreRange{core, core}}),
        CircularBufferConfig(pages * page_bytes, {{0, tt::DataFormat::UInt32}}).set_page_size(0, page_bytes));
}

MeshWorkload workload(MeshDevice& mesh, const Resources& buffers, uint32_t repeats,
                      std::array<uint32_t, 4>& semaphore_ids, RoutingRecord& routing) {
    Program program = CreateProgram();
    const CoreRangeSet pair({CoreRange{producer_core, consumer_core}});
    for (uint32_t index = 0; index < semaphore_ids.size(); ++index) {
        semaphore_ids[index] = CreateSemaphore(program, pair, 0);
    }
    scratch(program, producer_core, 1);
    scratch(program, consumer_core, 2);
    std::vector<uint32_t> producer_args;
    append_accessor(producer_args, buffers.ring, &routing.producer_ring_compile_args);
    append_accessor(producer_args, buffers.producer_stats);
    std::vector<uint32_t> consumer_args;
    append_accessor(consumer_args, buffers.ring);
    append_accessor(consumer_args, buffers.anchor, &routing.consumer_anchor_compile_args);
    routing.consumer_anchor_bank_slot = uint32_t(consumer_args.size() - 1);
    for (const auto& buffer : {buffers.progress, buffers.frames, buffers.consumer_stats}) {
        append_accessor(consumer_args, buffer);
    }
    const auto producer = CreateKernel(program, source("q4_frame_producer.cpp"), producer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default, .compile_args = producer_args});
    const auto consumer = CreateKernel(program, source("q4_frame_consumer.cpp"), consumer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default, .compile_args = consumer_args});
    std::vector<uint32_t> producer_runtime{
        uint32_t(buffers.ring->address()), uint32_t(buffers.producer_stats->address()),
        frame_count, frame_period, ring_pages, uint32_t(run_budget), uint32_t(run_budget >> 32)};
    std::vector<uint32_t> consumer_runtime{
        uint32_t(buffers.ring->address()), uint32_t(buffers.anchor->address()),
        uint32_t(buffers.progress->address()), uint32_t(buffers.frames->address()),
        uint32_t(buffers.consumer_stats->address()), frame_count, frame_period, ring_pages,
        repeats, uint32_t(run_budget), uint32_t(run_budget >> 32), uint32_t(buffers.mailbox->address())};
    routing.passed = issue120_q4_host::append_peer_runtime_args(producer_runtime, consumer_runtime,
        [&] { return peer_coordinates(mesh.get_devices().at(0)); });
    SetRuntimeArgs(program, producer, producer_core, producer_runtime);
    SetRuntimeArgs(program, consumer, consumer_core, consumer_runtime);
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

void read_l1(IDevice* device, CoreCoord logical, uint32_t address, std::span<uint8_t> destination) {
    const auto core = device->virtual_core_from_logical_core(logical, tt::CoreType::WORKER);
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        destination.data(), uint32_t(destination.size()), tt_cxy_pair(device->id(), core), address);
}

void read_challenge(IDevice* device, uint32_t mailbox, std::span<uint8_t> destination) {
    // Targeted read: no CQ finish and no device-wide host-write flush.
    read_l1(device, consumer_core, mailbox, destination);
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

struct CoordinateExpectation {
    issue120_q4_host::CoreXY self;
    issue120_q4_host::CoreXY peer;
    issue120_q4_host::CoreXY passed;
    const std::vector<uint32_t>& host_cta;
    uint32_t host_bank_slot;
};

Json coordinate_direction(const std::vector<uint32_t>& stats, uint32_t offset,
                          const CoordinateExpectation& expected) {
    const uint32_t packed = issue120_q4_host::pack_coordinate(expected.peer);
    const uint64_t accessor = ticks(stats.data(), offset + 2);
    const std::array<uint32_t, 2> peer{expected.peer.x, expected.peer.y};
    const std::array<uint32_t, 2> passed{expected.passed.x, expected.passed.y};
    const std::array<uint32_t, 2> runtime{stats[offset + 6], stats[offset + 7]};
    const std::array<uint32_t, 2> accessor_xy{uint32_t((accessor >> 36) & 63), uint32_t((accessor >> 42) & 63)};
    const uint64_t target = ticks(stats.data(), offset + 8);
    const std::array<uint32_t, 2> target_xy{uint32_t((target >> 36) & 63), uint32_t((target >> 42) & 63)};
    const bool word_matches = stats[offset + 1] == packed;
    return {
        {"host_expected_peer_virtual", peer}, {"host_passed_peer_runtime", passed},
        {"host_expected_packed_word", packed}, {"host_accessor_compile_args", expected.host_cta},
        {"host_actual_bank_word", expected.host_cta.back()}, {"host_global_bank_cta_slot", expected.host_bank_slot},
        {"kernel_bank_cta_slot", stats[offset]}, {"kernel_bank_word", stats[offset + 1]},
        {"kernel_word_matches_host_expected_pack", word_matches},
        {"host_actual_word_matches_expected_pack", expected.host_cta.back() == packed},
        {"kernel_word_matches_actual_host_cta", stats[offset + 1] == expected.host_cta.back()},
        {"kernel_cta_slot_matches_host", stats[offset] == expected.host_bank_slot},
        {"accessor_raw_noc_address", accessor}, {"accessor_coordinate", accessor_xy},
        {"accessor_matches_virtual", accessor_xy == peer},
        {"kernel_self_noc0", {stats[offset + 4], stats[offset + 5]}},
        {"host_expected_self_virtual", {expected.self.x, expected.self.y}},
        {"self_matches_virtual", stats[offset + 4] == expected.self.x && stats[offset + 5] == expected.self.y},
        {"kernel_peer_runtime", runtime}, {"runtime_matches_virtual", runtime == peer},
        {"runtime_matches_passed", runtime == passed}, {"passed_matches_diagnostic_virtual", passed == peer},
        {"actual_target", target}, {"actual_target_coordinate", target_xy},
        {"target_coordinate_matches_virtual", target_xy == peer},
        {"accessor_diagnostic_branch", issue120_q4_host::coordinate_classification(
            stats[offset + 1], {accessor_xy[0], accessor_xy[1]}, expected.peer)}};
}

Json startup_diagnostics(MeshDevice& mesh, MeshWorkload& program, const Resources& buffers,
                         const std::vector<uint32_t>& producer, const std::vector<uint32_t>& consumer,
                         const std::array<uint32_t, 4>& semaphore_ids, const RoutingRecord& routing) {
    auto* device = mesh.get_devices().at(0);
    auto shared_mesh = mesh.shared_from_this();
    const auto& hal = MetalContext::instance().hal();
    const uint32_t alignment = hal.get_alignment(HalMemType::L1);
    // In v0.75 this is a relocated kernel-config base plus program sem_offset,
    // not a static HAL SEMAPHORE address. Query the workload's actual CQ slot.
    const uint32_t base = program.impl().get_sem_base_addr(shared_mesh, producer_core, tt::CoreType::WORKER);
    uint32_t reads = 0;
    auto word = [&](CoreCoord core, uint32_t address) {
        uint32_t value = 0;
        read_l1(device, core, address, std::span<uint8_t>(reinterpret_cast<uint8_t*>(&value), sizeof(value)));
        ++reads;
        return value;
    };
    const auto producer_words = issue120_q4_host::semaphore_words(base, alignment,
        [&](uint32_t address) { return word(producer_core, address); });
    const auto consumer_words = issue120_q4_host::semaphore_words(base, alignment,
        [&](uint32_t address) { return word(consumer_core, address); });
    const uint32_t mailbox = word(consumer_core, uint32_t(buffers.mailbox->address()));
    const auto diagnostic_coordinates = peer_coordinates(device);
    const auto core = diagnostic_coordinates.producer;
    const uint64_t expected = issue120_q4_host::start_target(core.x, core.y, base, alignment);
    const uint64_t emitted = ticks(consumer.data(), 18);
    const uint64_t l1_base = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    const uint64_t l1_size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    const bool reported_valid = producer[6] >= l1_base && uint64_t(producer[6]) + 4 <= l1_base + l1_size;
    Json reported_word = nullptr;
    if (reported_valid) {
        reported_word = producer[6] == base + 2 * alignment ? producer_words[2] : word(producer_core, producer[6]);
    }
    const auto final_progress = download(mesh.mesh_command_queue(1), buffers.progress, 1);
    const uint32_t encoded = final_progress[0];
    bool valid = protocol::unpack_sequence(encoded) > 0;
    for (uint32_t i = 1; i < page_words; ++i) { valid &= final_progress[i] == protocol::word(encoded, i); }
    return {
        {"schema", "issue178-startup-diagnostics-v1"}, {"scope", "post_join_and_both_finishes_not_coverage"},
        {"coordinate_diagnostics", {
            {"schema", "issue178-coordinate-diagnostics-v2"}, {"scope", "final_stats_only_not_coverage"},
            {"consumer_to_producer", coordinate_direction(consumer, issue120_q4_args::consumer_coordinate_stats,
                {diagnostic_coordinates.consumer, diagnostic_coordinates.producer, routing.passed.producer,
                 routing.consumer_anchor_compile_args, routing.consumer_anchor_bank_slot})},
            {"producer_to_consumer", coordinate_direction(producer, issue120_q4_args::producer_coordinate_stats,
                {diagnostic_coordinates.producer, diagnostic_coordinates.consumer, routing.passed.consumer,
                 routing.producer_ring_compile_args, uint32_t(routing.producer_ring_compile_args.size() - 1)})},
            {"producer_ready_target_used", producer[1] > 0},
            {"runtime_arg_indices", {{"producer_peer", {issue120_q4_args::Producer::peer_x, issue120_q4_args::Producer::peer_y}},
                                     {"consumer_peer", {issue120_q4_args::Consumer::peer_x, issue120_q4_args::Consumer::peer_y}}}}}},
        {"semaphore_ids", semaphore_ids}, {"kernel_assumed_ids", {0, 1, 2, 3}},
        {"hal_kernel_config_base", hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG)},
        {"resolved_semaphore_base", base}, {"l1_alignment", alignment},
        {"producer_virtual_core", {core.x, core.y}},
        {"expected_start_target", expected}, {"consumer_emitted_start_target", emitted},
        {"start_target_matches", expected == emitted},
        {"bank_coordinate_matches", (expected & issue120_q4_host::coordinate_mask) == (emitted & issue120_q4_host::coordinate_mask)},
        {"producer_semaphores", producer_words}, {"consumer_semaphores", consumer_words},
        {"producer_exit_start_raw", producer[5]}, {"producer_start_address", producer[6]},
        {"producer_start_address_matches", producer[6] == base + 2 * alignment},
        {"producer_reported_address_valid", reported_valid}, {"producer_reported_start_post_finish", reported_word},
        {"producer_opened_ticks", ticks(producer.data(), 7)}, {"producer_exit_ticks", ticks(producer.data(), 9)},
        {"producer_semaphore_base", producer[11]}, {"producer_abort", producer[3]},
        {"consumer_semaphore_base", consumer[22]}, {"consumer_abort", consumer[14]},
        {"consumer_start_inc_ticks", ticks(consumer.data(), 20)},
        {"consumer_mailbox_address", consumer[23]}, {"consumer_exit_mailbox_readback", consumer[24]},
        {"mailbox_post_finish", mailbox}, {"mailbox_expected", 2},
        {"progress_post_drain", {{"sequence", protocol::unpack_sequence(encoded)},
                                 {"challenge", protocol::unpack_challenge(encoded)}, {"valid", valid}}},
        {"diagnostic_mmio_reads", reads}, {"diagnostic_mmio_bytes", 4 * reads},
        {"diagnostic_progress_read_bytes", page_bytes},
        {"interpretation", "post-finish 1 means present, not proof of arrival time; compare exit value/ticks; 0 requires target comparison, not a guessed cause"}};
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
    std::array<uint32_t, 4> semaphore_ids{};
    RoutingRecord routing{};
    auto program = workload(mesh, buffers, repeats, semaphore_ids, routing);
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
    const std::array<SubDeviceId, 1> resident_ids{SubDeviceId{0}};
    const std::array<SubDeviceId, 1> transfer_ids{SubDeviceId{1}};
    const auto schedule = issue120_q4_host::execute_sequence(
        [&] { EnqueueMeshWorkload(cq0, program, false); },
        [&] { load_thread.join(); if (load.error) { std::rethrow_exception(load.error); } },
        [&] { cq0.finish(ttsl::Span<const SubDeviceId>(resident_ids)); },
        [&] { cq1.finish(ttsl::Span<const SubDeviceId>(transfer_ids)); },
        now_ns);
    const auto producer = download(cq0, buffers.producer_stats, 1);
    const auto consumer = download(cq0, buffers.consumer_stats, 1);
    const auto diagnostics = startup_diagnostics(mesh, program, buffers, producer, consumer, semaphore_ids, routing);
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
        {"resident_host_interval", {{"start_ns", schedule.submitted}, {"end_ns", schedule.resident_finished},
                                    {"scope", "enqueue_to_post_join_finish_return_upper_bound"}}},
        {"host_scheduling", {{"policy", "join_before_any_finish"}, {"load_join_completed_ns", schedule.joined},
                             {"resident_finish_returned_ns", schedule.resident_finished}}},
        {"startup_diagnostics", diagnostics},
        {"host_interval", {{"start_ns", load.start_ns}, {"end_ns", load.end_ns}}},
        {"final_drain_interval", {{"start_ns", schedule.drain_started}, {"end_ns", schedule.drained}}},
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
