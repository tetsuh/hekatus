"""CPU-only host trace ordering, fixed cadence and cross-thread count seams."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "enodia/tt/bench/issue120_production"


@pytest.fixture(scope="module")
def host_trace_executable(tmp_path_factory):
    directory = tmp_path_factory.mktemp("host-trace")
    source = directory / "trace.cpp"
    source.write_text(r'''
#include "host_trace.hpp"
#include "host_protocol_trace.hpp"
#include <atomic>
#include <cassert>
#include <stdexcept>
#include <thread>
#include <string>
namespace p=issue120_production;
int main(int argc,char** argv) {
    assert(argc==2);p::HostTrace trace;
    static_assert(p::HostTrace::capacity==65536);
    static_assert(p::HostTrace::period==65536);
    static_assert(std::atomic<uint32_t>::is_always_lock_free);
    if(std::string(argv[1])=="cadence") {
        uint32_t operations=0,writes=0,clocks=0;
        auto clock=[&]{return uint64_t(1000+10*++clocks);};
        p::Traffic facts(4);
        p::submit_fixed_traced(65537,4,facts,[]{return 1u;},[&]{++operations;},[&](const auto& data){
            assert(data[0]==++writes);assert(trace.published_count()==data[0]-1);
        },trace,clock);
        trace.capture_enqueue_return(1025,clock);
        assert(operations==65537&&writes==65537&&facts.emitted==65537);
        assert(clocks==3&&trace.samples().size()==2);
        assert(trace.samples()[0].completed==1&&trace.samples()[0].ns==1010);
        assert(trace.samples()[1].completed==65536&&trace.samples()[1].ns==1020);
        assert(trace.main_count()==65537&&trace.main_before()==1025&&trace.main_after()==1030);
    } else if(std::string(argv[1])=="concurrent") {
        std::atomic<bool> ready{false},release{false};std::atomic<uint64_t> ns{100};
        auto clock=[&]{return ns.fetch_add(10);};
        std::thread worker([&]{
            for(uint32_t s=1;s<=131072;++s) {
                trace.observe_success(s,clock);
                if(s==65536){ready.store(true);while(!release.load())std::this_thread::yield();}
            }
        });
        while(!ready.load())std::this_thread::yield();
        trace.capture_enqueue_return(115,clock);
        assert(trace.main_count()==65536);
        release=true;worker.join();
        assert(trace.published_count()==131072&&trace.samples().size()==3);
        assert(trace.samples()[2].completed==131072);
    } else if(std::string(argv[1])=="failure") {
        p::Traffic facts(4);bool failed=false;
        try {
            p::submit_fixed_traced(8,4,facts,[]{return 1u;},[]{},[&](const auto& data){
                if(data[0]==2)throw std::runtime_error("injected enqueue failure");
            },trace,[]{return uint64_t(100);});
        } catch(const std::runtime_error&){failed=true;}
        assert(failed&&facts.emitted==1&&trace.published_count()==1);
        assert(trace.samples().size()==1);
    } else if(std::string(argv[1])=="clock-failure") {
        p::Traffic facts(4);bool failed=false;
        try {
            p::submit_fixed_traced(8,4,facts,[]{return 1u;},[]{},[](const auto&){},trace,
                []()->uint64_t{throw std::runtime_error("clock unavailable");});
        } catch(const std::runtime_error&){failed=true;}
        assert(failed&&facts.emitted==1&&trace.published_count()==1);
        assert(trace.samples().empty());
        static_assert(1+uint64_t(0xffffffffu)/p::HostTrace::period==p::HostTrace::capacity);
    } else return 2;
}
''')
    binary = directory / "trace"
    subprocess.run(["c++", "-std=c++20", "-O2", "-pthread", "-I", str(SOURCE), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("scenario", ["cadence", "concurrent", "failure", "clock-failure"])
def test_actual_host_trace_has_fixed_cadence_and_no_phantom_suffix(host_trace_executable, scenario):
    subprocess.run([str(host_trace_executable), scenario], check=True, timeout=20)


def test_host_only_native_contract_rejects_device_source_change():
    from enodia.tt.bench.issue120_production import host_trace_contract

    text = host_trace_contract.derive_native((SOURCE / "native.cpp").read_text())
    assert host_trace_contract.verify_native_source(text)["device_dispatch_unchanged"] is True
    altered = text.replace("const CoreCoord producer_core{0,0}", "const CoreCoord producer_core{3,0}")
    with pytest.raises(ValueError):
        host_trace_contract.verify_native_source(altered)
