"""Execute the production scheduling seam with fake queues; never open devices."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sequence_binary(tmp_path_factory):
    directory = tmp_path_factory.mktemp("q4-sequence")
    source = directory / "sequence.cpp"
    source.write_text(r'''
#include "q4_host_sequence.hpp"
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
int main() {
    using namespace issue120_q4_host;
    std::vector<std::string> events;
    uint64_t clock = 0;
    auto times = execute_sequence(
        [&] { events.push_back("enqueue"); },
        [&] { events.push_back("join"); },
        [&] { events.push_back("finish0"); },
        [&] { events.push_back("finish1"); },
        [&] { return ++clock; });
    assert((events == std::vector<std::string>{"enqueue", "join", "finish0", "finish1"}));
    assert(times.submitted < times.joined && times.joined < times.resident_finished);
    assert(times.resident_finished <= times.drain_started && times.drain_started < times.drained);

    std::mutex api, state;
    std::condition_variable ready;
    bool submitted = false, writes_complete = false;
    std::thread load([&] {
        { std::unique_lock lock(state); ready.wait(lock, [&] { return submitted; }); }
        std::lock_guard lock(api);
        writes_complete = true;
    });
    execute_sequence(
        [&] {
            std::lock_guard lock(api);
            { std::lock_guard signal(state); submitted = true; }
            ready.notify_one();
        },
        [&] { load.join(); },
        [&] { std::lock_guard lock(api); assert(writes_complete); },
        [&] { std::lock_guard lock(api); assert(writes_complete); },
        [&] { return ++clock; });

    constexpr uint32_t x = 1, y = 2, semaphore_base = 0x2000, alignment = 16;
    const auto target = start_target(x, y, semaphore_base, alignment);
    assert(target == ((uint64_t(y) << 42) | (uint64_t(x) << 36) | 0x2020));
    const auto bank = (uint64_t(y) << 42) | (uint64_t(x) << 36) | 0x123456;
    assert((bank & coordinate_mask) == (target & coordinate_mask));
    assert((target & ~coordinate_mask) == semaphore_base + 2 * alignment);
    std::vector<uint32_t> addresses;
    auto words = semaphore_words(semaphore_base, alignment, [&](uint32_t address) {
        addresses.push_back(address); return (address - semaphore_base) / alignment;
    });
    assert((addresses == std::vector<uint32_t>{0x2000,0x2010,0x2020,0x2030}));
    assert((words == std::array<uint32_t,4>{0,1,2,3}));
}
''')
    binary = directory / "cpu-sequence"
    subprocess.run(["g++", "-std=c++20", "-pthread", "-I", "enodia/tt/bench/issue120", str(source), "-o", str(binary)], check=True)
    return binary


def test_join_precedes_all_finishes_and_fixed_snapshot_addresses(sequence_binary):
    subprocess.run([str(sequence_binary)], check=True, timeout=10)


def test_start_target_matches_pinned_blackhole_address_macro(tmp_path):
    header = Path("out/issue120-deps/source/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h")
    # CI does not stage the pinned image tree; an exact public-source macro is
    # also frozen below, so dependency staging is not required by this test.
    macro = "#define NOC_XY_ADDR(x,y,a) (((uint64_t)(y)<<42)|((uint64_t)(x)<<36)|(uint64_t)(a))"
    if header.exists():
        text = header.read_text()
        start = text.index("#define NOC_XY_ADDR(")
        macro = text[start:text.index("\n\n", start)]
        macro = "#define NOC_ADDR_LOCAL_BITS 36\n#define NOC_ADDR_NODE_ID_BITS 6\n" + macro
    cpp = tmp_path / "address.cpp"
    cpp.write_text('#include "q4_host_sequence.hpp"\n' + macro + r'''
int main() {
    constexpr auto actual = issue120_q4_host::start_target(1, 2, 0x2000, 16);
    static_assert(actual == NOC_XY_ADDR(1,2,0x2000+2*16));
    static_assert((NOC_XY_ADDR(1,2,0x1234) & issue120_q4_host::coordinate_mask) ==
                  (actual & issue120_q4_host::coordinate_mask));
}
''')
    binary = tmp_path / "address"
    subprocess.run(["g++", "-std=c++20", "-I", "enodia/tt/bench/issue120", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
