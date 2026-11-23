// SPDX-License-Identifier: Apache-2.0
#include "protocol.hpp"
#include "host_protocol.hpp"
#include "host_sequence.hpp"
#include "../issue120/native_result_io.hpp"
#include "../issue120/q4_host_sequence.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <thread>
#include <time.h>

namespace {
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
namespace p = issue120_production;
using Json = nlohmann::json;
static_assert(std::endian::native == std::endian::little);
const CoreCoord producer_core{0,0}, consumer_core{1,0};
constexpr auto root_source = "enodia/tt/bench/issue120_production";
constexpr auto kernel_root = "out/issue120-production-build/kernels";
uint64_t clock_ns(clockid_t id) {
    timespec value{};
    if (::clock_gettime(id, &value) != 0) { throw std::runtime_error("host clock unavailable"); }
    return uint64_t(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}
uint64_t now() { return clock_ns(CLOCK_MONOTONIC); }
struct Stamp { uint64_t mono = 0, utc = 0; };
Stamp stamp() { const auto mono = now(); return {mono, clock_ns(CLOCK_REALTIME)}; }
uint64_t u64(const uint32_t* data, uint32_t offset) {
    return uint64_t(data[offset]) | (uint64_t(data[offset+1]) << 32);
}
struct Args {
    bool loaded; uint32_t frames, repeats, cap; uint64_t budget;
    std::string identity; std::filesystem::path result;
};
uint64_t decimal(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos ||
        (value.size() > 1 && value[0] == '0')) { throw std::invalid_argument("invalid numeric argument"); }
    size_t used = 0; const auto parsed = std::stoull(value, &used);
    if (used != value.size()) { throw std::invalid_argument("invalid numeric argument"); }
    return parsed;
}
Args parse(int argc, char** argv) {
    std::map<std::string,std::string> values;
    const std::array<std::string,7> keys{"--condition","--frames","--repeats","--cap",
        "--budget","--attempt-identity","--result-file"};
    for (int i=1; i<argc; i+=2) {
        if (i+1 >= argc || std::find(keys.begin(), keys.end(), argv[i]) == keys.end() ||
            !values.emplace(argv[i], argv[i+1]).second) { throw std::invalid_argument("unsupported/duplicate argument"); }
    }
    if (values.size() != keys.size()) { throw std::invalid_argument("all production arguments required"); }
    const auto frames = decimal(values.at("--frames")), repeats = decimal(values.at("--repeats"));
    const auto cap = decimal(values.at("--cap")), budget = decimal(values.at("--budget"));
    const auto condition = values.at("--condition");
    const bool bounded = (frames==4096 && repeats==262144) ||
        (frames==16384 && repeats==1048576) || (frames==32768 && repeats==1572864);
    const bool production = frames==500000; // Exact N is bound by the committed Python admission.
    if (frames > 0xffffffffULL || repeats > 0xffffffffULL ||
        !p::valid_domain(uint32_t(frames), uint32_t(repeats)) ||
        (!bounded && !production) || cap != (production ? 600 : 60) ||
        (condition!="control" && condition!="loaded") ||
        budget != (frames * (1350000ULL+100000ULL) * 110 + 99)/100 + 135000000ULL) {
        throw std::invalid_argument("configuration differs from the precommitted production domain");
    }
    Args a{condition=="loaded",uint32_t(frames),uint32_t(repeats),uint32_t(cap),budget,
        values.at("--attempt-identity"),values.at("--result-file")};
    issue120::validate_result_target("/out", a.result, a.identity);
    return a;
}
std::shared_ptr<MeshBuffer> dram(MeshDevice& mesh, uint32_t pages) {
    return MeshBuffer::create(ReplicatedBufferConfig{.size=uint64_t(pages)*p::page_bytes},
        DeviceLocalBufferConfig{.page_size=p::page_bytes,.buffer_type=BufferType::DRAM}, &mesh);
}
std::shared_ptr<MeshBuffer> l1(MeshDevice& mesh, CoreCoord core, uint32_t pages) {
    const CoreRangeSet range({CoreRange{core,core}});
    BufferDistributionSpec distribution(Shape{pages,1},Shape{pages,1},range,ShardOrientation::ROW_MAJOR);
    auto b = MeshBuffer::create(ReplicatedBufferConfig{.size=uint64_t(pages)*p::page_bytes},
        DeviceLocalBufferConfig{.page_size=p::page_bytes,.buffer_type=BufferType::L1,
            .sharding_args=BufferShardingArgs(std::move(distribution))}, &mesh);
    const auto* ref=b->get_reference_buffer(); const auto& d=ref->buffer_distribution_spec();
    if (!d || d->cores().size()!=1 || d->cores().front()!=core ||
        d->num_dev_pages_per_core(0)!=pages || ref->aligned_size_per_bank()<uint64_t(pages)*p::page_bytes) {
        throw std::runtime_error("explicit production L1 distribution differs");
    }
    return b;
}
void partition(MeshDevice& mesh) {
    const std::array<CoreRangeSet,1> r{CoreRangeSet({CoreRange{producer_core,consumer_core}})};
    const std::array<CoreRangeSet,1> t{CoreRangeSet({CoreRange{CoreCoord{2,0},CoreCoord{2,0}}})};
    const std::array<SubDevice,2> parts{SubDevice(ttsl::Span<const CoreRangeSet>(r)),
                                      SubDevice(ttsl::Span<const CoreRangeSet>(t))};
    mesh.load_sub_device_manager(mesh.create_sub_device_manager(ttsl::Span<const SubDevice>(parts),0));
    const std::array<SubDeviceId,1> ids{SubDeviceId{1}};
    mesh.set_sub_device_stall_group(ttsl::Span<const SubDeviceId>(ids));
}
void cb(Program& program, CoreCoord core, uint32_t index) {
    CreateCircularBuffer(program,CoreRangeSet({CoreRange{core,core}}),
        CircularBufferConfig(p::page_bytes,{{index,tt::DataFormat::UInt32}}).set_page_size(index,p::page_bytes));
}
struct Buffers {
    std::shared_ptr<MeshBuffer> ring,control,anchor,mailbox,progress,frames,producer_stats,consumer_stats;
    Buffers(MeshDevice& mesh, uint32_t count) : ring(l1(mesh,consumer_core,4)),
        control(l1(mesh,consumer_core,1)), anchor(l1(mesh,producer_core,1)),
        mailbox(l1(mesh,consumer_core,1)), progress(dram(mesh,1)), frames(dram(mesh,count)),
        producer_stats(dram(mesh,1)), consumer_stats(dram(mesh,1)) {}
};
void append(std::vector<uint32_t>& words, const std::shared_ptr<MeshBuffer>& buffer,
            IDevice* device, std::optional<CoreCoord> core={}, uint32_t pages=0) {
    const auto extra=TensorAccessorArgs(*buffer).get_compile_time_args();
    if (core) {
        const auto xy=device->virtual_core_from_logical_core(*core,tt::CoreType::WORKER);
        if(xy.x>255 || xy.y>255) throw std::runtime_error("virtual coordinate exceeds wire domain");
        issue120_q4_host::validate_single_core_accessor(extra,{uint32_t(xy.x),uint32_t(xy.y)},pages);
    }
    words.insert(words.end(),extra.begin(),extra.end());
}
MeshWorkload workload(MeshDevice& mesh, Buffers& b, const Args& a, Json& routing) {
    auto* device=mesh.get_devices().at(0); Program program=CreateProgram();
    for (uint32_t i=0;i<5;++i) {
        if (CreateSemaphore(program,CoreRangeSet({CoreRange{producer_core,consumer_core}}),0)!=i) {
            throw std::runtime_error("production semaphore IDs differ");
        }
    }
    cb(program,producer_core,20); cb(program,consumer_core,21); cb(program,consumer_core,26);
    std::vector<uint32_t> pc,cc;
    append(pc,b.ring,device,consumer_core,4); append(pc,b.control,device,consumer_core,1);
    append(pc,b.producer_stats,device); append(pc,b.producer_stats,device);
    append(cc,b.ring,device,consumer_core,4); append(cc,b.anchor,device,producer_core,1);
    for(const auto& buffer:{b.frames,b.consumer_stats,b.producer_stats,b.consumer_stats,b.progress}) {
        append(cc,buffer,device);
    }
    Json manifest; std::ifstream(std::filesystem::path(root_source)/"compile_args.json") >> manifest;
    if (manifest.at("schema")!="issue120-production-compile-args-v1" ||
        pc!=manifest.at("producer").get<std::vector<uint32_t>>() ||
        cc!=manifest.at("consumer").get<std::vector<uint32_t>>()) {
        throw std::runtime_error("production actual CTA differs from offline manifest: "+Json{{"producer",pc},{"consumer",cc}}.dump());
    }
    const auto pk=CreateKernel(program,(std::filesystem::path(kernel_root)/"production_producer.cpp").string(),
        producer_core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
            .noc=NOC::RISCV_0_default,.compile_args=pc});
    const auto ck=CreateKernel(program,(std::filesystem::path(kernel_root)/"production_consumer.cpp").string(),
        consumer_core,DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
            .noc=NOC::RISCV_0_default,.compile_args=cc});
    std::vector<uint32_t> pr{uint32_t(b.ring->address()),uint32_t(b.control->address()),
        uint32_t(b.producer_stats->address()),a.frames,1350000,4,uint32_t(a.budget),uint32_t(a.budget>>32),
        0,uint32_t(b.producer_stats->address())};
    std::vector<uint32_t> cr{uint32_t(b.ring->address()),uint32_t(b.anchor->address()),
        uint32_t(b.frames->address()),uint32_t(b.consumer_stats->address()),a.frames,4,64,100000,
        uint32_t(a.budget),uint32_t(a.budget>>32),0,uint32_t(b.producer_stats->address()),
        uint32_t(b.consumer_stats->address()),0,0,uint32_t(b.progress->address()),
        uint32_t(b.mailbox->address()),a.repeats,1350000};
    p::append_peers(pr,cr,p::producer_arg::prefix,p::consumer_arg::prefix,[&](p::XY c) {
        const auto xy=device->virtual_core_from_logical_core({c.x,c.y},tt::CoreType::WORKER);
        if(xy.x>255 || xy.y>255) throw std::runtime_error("virtual coordinate exceeds wire domain");
        return p::XY{uint32_t(xy.x),uint32_t(xy.y)};
    });
    SetRuntimeArgs(program,pk,producer_core,pr); SetRuntimeArgs(program,ck,consumer_core,cr);
    routing={{"producer_cta",pc},{"consumer_cta",cc},{"producer_runtime",pr},{"consumer_runtime",cr}};
    MeshWorkload result; result.add_program(MeshCoordinateRange(mesh.shape()),std::move(program)); return result;
}
void read_mailbox(IDevice* device, uint32_t address, void* target, uint32_t bytes) {
    const auto xy=device->virtual_core_from_logical_core(consumer_core,tt::CoreType::WORKER);
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(target,bytes,
        tt_cxy_pair(device->id(),xy),address);
}
struct Header { char magic[8]; uint32_t version=1,words=0,count=0,bound=0; };
static_assert(sizeof(Header)==24 && sizeof(p::Transition)==8);
class Binary {
    int fd;
public:
    explicit Binary(const std::filesystem::path& path) :
        fd(::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600)) {
        if(fd<0) throw std::runtime_error("cannot claim binary evidence");
    }
    ~Binary() { if(fd>=0) ::close(fd); }
    void write(const void* ptr,size_t bytes) {
        const auto* data=static_cast<const char*>(ptr); size_t at=0;
        while(at<bytes) {
            const auto n=::write(fd,data+at,bytes-at);
            if(n<0 && errno==EINTR) continue;
            if(n<=0) throw std::runtime_error("binary evidence write failed");
            at+=size_t(n);
        }
    }
    void finish() {
        if(::fsync(fd)!=0) throw std::runtime_error("binary evidence fsync failed");
        const int old=fd; fd=-1;
        if(::close(old)!=0) throw std::runtime_error("binary evidence close failed");
    }
};
std::vector<uint32_t> download(MeshCommandQueue& queue,const std::shared_ptr<MeshBuffer>& buffer) {
    std::vector<uint32_t> words(p::page_words); queue.enqueue_read_mesh_buffer(words.data(),buffer,true); return words;
}
Json run(MeshDevice& mesh,const Args& a) {
    partition(mesh); Buffers b(mesh,a.frames); auto& q0=mesh.mesh_command_queue(0); auto& q1=mesh.mesh_command_queue(1);
    std::vector<uint32_t> zeros(4*p::page_words,0);
    for(const auto& buffer:{b.ring,b.control,b.anchor,b.mailbox}) q0.enqueue_write_mesh_buffer(buffer,zeros.data(),true);
    std::array<uint32_t,p::page_words> seed{};
    for(uint32_t i=0;i<p::page_words;++i) seed[i]=p::word(p::encode(0,0),i);
    q0.enqueue_write_mesh_buffer(b.progress,seed.data(),true);
    Json routing; auto work=workload(mesh,b,a,routing);
    p::Traffic traffic(a.frames); std::atomic<bool> cancel{false}; std::exception_ptr load_error,observe_error;
    uint64_t load_start=0,load_end=0,cpu_operations=0,load_reads=0,observer_reads=0;
    uint64_t joined=0,observed_joined=0,enqueue_return=0,finish_begin=0,finish_end=0,drain_end=0;
    uint32_t cpu_checksum=0; Stamp program_start{},program_end{};
    std::thread loader,observer;
    auto* device=mesh.get_devices().at(0); const uint32_t mailbox=uint32_t(b.mailbox->address());
    const std::array<SubDeviceId,1> resident_ids{SubDeviceId{0}},transfer_ids{SubDeviceId{1}};
    program_start=stamp();
    p::host_sequence([&] {
        observer=std::thread([&] {
            try {
                const auto begin=now(); std::array<uint32_t,4> words{};
                while(!cancel.load()) {
                    read_mailbox(device,mailbox,words.data(),16); ++observer_reads;
                    if(words[1]==1) { program_end=stamp(); return; }
                    if(now()-begin>=uint64_t(a.cap)*1000000000ULL) throw std::runtime_error("terminal observation cap");
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            } catch(...) { observe_error=std::current_exception(); cancel=true; }
        });
        if(a.loaded) loader=std::thread([&] {
            load_start=now(); volatile uint32_t checksum=0;
            try {
                p::submit_fixed(a.repeats,a.frames,traffic,[&] {
                    if(cancel.load()) throw std::runtime_error("failed execution cancelled submitter");
                    uint32_t challenge=0; read_mailbox(device,mailbox,&challenge,4); ++load_reads; return challenge;
                },[&] {
                    for(uint32_t round=0;round<32;++round) for(uint32_t word=0;word<1024;++word) {
                        checksum=(checksum*33u)^word;
                    }
                    cpu_operations+=32768;
                },[&](const auto& payload) { q1.enqueue_write_mesh_buffer(b.progress,payload.data(),false); });
            } catch(...) { load_error=std::current_exception(); cancel=true; }
            cpu_checksum=checksum; load_end=now();
        });
    },[&] {
        try { EnqueueMeshWorkload(q0,work,false); enqueue_return=now(); }
        catch(...) { cancel=true; throw; }
    },[&] {
        if(loader.joinable()) loader.join(); joined=now();
        if(observer.joinable()) observer.join(); observed_joined=now();
        if(load_error) std::rethrow_exception(load_error);
        if(observe_error) std::rethrow_exception(observe_error);
    },[&] { finish_begin=now(); q0.finish(ttsl::Span<const SubDeviceId>(resident_ids)); finish_end=now(); },
       [&] { q1.finish(ttsl::Span<const SubDeviceId>(transfer_ids)); drain_end=now(); },
       [&] { cancel=true; });
    if(program_end.mono==0) throw std::runtime_error("resident completion observation absent");
    const auto ps=download(q1,b.producer_stats),cs=download(q1,b.consumer_stats);
    if(cs[1]>a.frames) throw std::runtime_error("consumer count exceeds fixed frame capacity");
    const auto evidence=a.result.parent_path().parent_path()/"evidence";
    if(!std::filesystem::create_directory(evidence)) throw std::runtime_error("evidence directory must be new");
    Binary frame_file(evidence/"frames.bin"); Header fh{}; std::memcpy(fh.magic,"IP120F1",7);
    fh.words=32;fh.count=cs[1];fh.bound=a.frames;frame_file.write(&fh,sizeof(fh));
    std::vector<uint32_t> chunk(256*p::page_words); uint32_t expected_checksum=0,verified=0;
    for(uint32_t first=0;first<cs[1];first+=256) {
        const uint32_t count=std::min(256u,cs[1]-first);
        std::vector<tt::tt_metal::distributed::ShardDataTransfer> transfers{
            tt::tt_metal::distributed::ShardDataTransfer(MeshCoordinate(0,0)).host_data(chunk.data()).region(
                BufferRegion(uint64_t(first)*p::page_bytes,uint64_t(count)*p::page_bytes))};
        q1.enqueue_read_shards(transfers,b.frames,true);
        for(uint32_t i=0;i<count;++i) {
            const auto* row=chunk.data()+i*p::page_words;
            for(uint32_t unit=0;unit<64;++unit) expected_checksum=p::checksum_step(expected_checksum,first+i,unit);
            verified+=row[2]==expected_checksum;
            frame_file.write(row,32*sizeof(uint32_t));
        }
    }
    frame_file.finish(); Binary map_file(evidence/"transitions.bin"); Header mh{};
    std::memcpy(mh.magic,"IP120T1",7);mh.words=2;mh.count=traffic.transition_count;mh.bound=traffic.emitted;
    map_file.write(&mh,sizeof(mh));map_file.write(traffic.transitions.data(),traffic.transition_count*sizeof(p::Transition));map_file.finish();
    std::array<uint32_t,4> final_mailbox{}; read_mailbox(device,mailbox,final_mailbox.data(),16);
    Json host=nullptr,cpu={{"operations",0},{"reads",0},{"transfers",0}};
    if(a.loaded) {
        host={{"start_ns",load_start},{"end_ns",load_end}};
        cpu={{"start_ns",load_start},{"end_ns",load_end},{"operations",cpu_operations},
             {"reads",load_reads},{"transfers",traffic.emitted},{"checksum",cpu_checksum}};
    }
    return {{"schema","issue120-production-native-v1"},{"condition",a.loaded?"loaded":"control"},
        {"frame_count",a.frames},{"repeat_count",a.repeats},{"host_interval",host},{"host_cpu_work",cpu},
        {"host_scheduling",{{"enqueue_begin_ns",program_start.mono},{"enqueue_return_ns",enqueue_return},
            {"load_join_completed_ns",joined},{"completion_observer_joined_ns",observed_joined},
            {"finish_begin_ns",finish_begin},{"finish_end_ns",finish_end},{"drain_end_ns",drain_end}}},
        {"program_host_interval",{{"start_ns",program_start.mono},{"end_ns",program_end.mono},
            {"start_utc_ns",program_start.utc},{"end_utc_ns",program_end.utc},
            {"scope","enqueue_to_l1_terminal_observed_upper_bound"}}},
        {"resident_host_interval",{{"start_ns",program_start.mono},{"end_ns",finish_end},
            {"scope","enqueue_to_post_join_finish_return_upper_bound"}}},
        {"producer_stats",ps},{"consumer_stats",cs},{"checksum_verified_frames",verified},
        {"startup_diagnostics",{{"expected_after_consumed_frames",cs[1]+2},
            {"expected_after_full_run",a.frames+2},{"mailbox_post_finish",final_mailbox[0]},
            {"consumer_reported_expectation",cs[32]},{"producer_peer_runtime_xy",{routing["producer_runtime"][10],routing["producer_runtime"][11]}},
            {"consumer_peer_runtime_xy",{cs[62],cs[63]}},{"consumer_self_noc0",{cs[64],cs[65]}}}},
        {"observer_reads",observer_reads},{"routing",routing},
        {"evidence",{{"frames","frames.bin"},{"transitions","transitions.bin"},{"frame_words",32}}}};
}
}
int main(int argc,char** argv) {
    try {
        const auto args=parse(argc,argv);
        auto mesh=MeshDevice::create_unit_mesh(0,DEFAULT_L1_SMALL_SIZE,DEFAULT_TRACE_REGION_SIZE,2);
        const Json result=run(*mesh,args); mesh->close();
        issue120::publish_result("/out",args.result,args.identity,result.dump()); return 0;
    } catch(const std::exception& error) {
        std::cerr<<"Issue120 production failed: "<<error.what()<<'\n'; return 2;
    }
}
