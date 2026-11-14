"""CPU-only characterization of the existing host loop, not a device diagnosis.

Compile the actual traffic body with in-memory queue/mailbox seams. No Metalium
headers/libraries are loaded; the full startup/NoC path is deliberately not faked
into a successful device validation. No production behavior changes here.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

HOST = Path("enodia/tt/bench/issue120/q4_frame_probe.cpp")
BASE = Path("docs/measurements/2026-11-07-p150a-issue120-p6")


def _definition(source: str, signature: str) -> str:
    """Extract a brace-balanced definition from this controlled source fixture."""
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[start:index + 1]
    raise AssertionError("unterminated host definition")


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    source = HOST.read_text()
    constants = "\n".join(
        re.search(rf"constexpr uint32_t {name} = \d+;", source)[0]
        for name in ("page_words", "cpu_rounds", "cpu_words")
    )
    directory = tmp_path_factory.mktemp("startup-replay")
    cpp = directory / "replay.cpp"
    cpp.write_text(r'''
#include "q4_protocol.hpp"
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace protocol = issue120_q4_protocol;
struct IDevice {};
struct MeshBuffer {};
static std::string mode;
static uint32_t reads = 0;
static uint32_t published = 0;
static std::vector<uint32_t> echoes;
static std::vector<uint32_t> sequences;
static bool valid = true;
uint64_t now_ns() { static uint64_t n = 0; return ++n; }
void read_challenge(IDevice*, uint32_t address, std::span<uint8_t> bytes) {
    if (address != 1234 || bytes.size() != 4) { throw std::runtime_error("bad mock read"); }
    if (mode == "delayed" && reads > 0) { published = reads; }
    const uint32_t value = mode == "wrong_target" ? 0 : published;
    std::memcpy(bytes.data(), &value, sizeof(value));
    ++reads;
}
struct MeshCommandQueue {
    void enqueue_write_mesh_buffer(const std::shared_ptr<MeshBuffer>&,
                                   const uint32_t* data, bool blocking) {
        valid &= !blocking;
        sequences.push_back(protocol::unpack_sequence(data[0]));
        echoes.push_back(protocol::unpack_challenge(data[0]));
        for (uint32_t i = 0; i < 1024; ++i) {
            valid &= data[i] == protocol::word(data[0], i);
        }
    }
};
''' + constants + "\n" + _definition(source, "struct HostLoad") + ";\n" +
        _definition(source, "void traffic(") + r'''
int main(int argc, char** argv) {
    if (argc != 2) { return 2; }
    mode = argv[1];
    if (mode == "wrong_target" || mode == "ready") { published = 1; }
    if (mode == "invalid") { published = protocol::max_challenge + 1; }
    MeshCommandQueue queue;
    auto progress = std::make_shared<MeshBuffer>();
    IDevice device;
    std::vector<uint32_t> payload(page_words);
    HostLoad load;
    traffic(queue, progress, &device, 1234, 3, payload, load);
    std::cout << "{\"reads\":" << reads << ",\"error\":" << bool(load.error)
              << ",\"published\":" << published << ",\"valid\":" << valid
              << ",\"echoes\":[";
    for (uint32_t i = 0; i < echoes.size(); ++i) {
        if (i) { std::cout << ','; }
        std::cout << echoes[i];
    }
    std::cout << "],\"sequences\":[";
    for (uint32_t i = 0; i < sequences.size(); ++i) {
        if (i) { std::cout << ','; }
        std::cout << sequences[i];
    }
    std::cout << "]}";
}
''')
    binary = directory / "cpu-startup-replay"
    subprocess.run(["g++", "-std=c++20", "-I", "enodia/tt/bench/issue120", str(cpp), "-o", str(binary)], check=True)
    return binary


def _run(replay, mode):
    result = subprocess.run([str(replay), mode], capture_output=True, check=True, timeout=10)
    return json.loads(result.stdout)


def test_initial_zero_then_publication_is_echoed_by_the_real_host_loop(replay):
    result = _run(replay, "delayed")
    assert result["echoes"] == [0, 1, 2]
    assert result["sequences"] == [1, 2, 3]
    assert result["reads"] == 3
    assert result["valid"] == 1
    assert result["error"] == 0


def test_zero_echoes_do_not_distinguish_unpublished_from_wrong_target(replay):
    unpublished = _run(replay, "unpublished")
    wrong_target = _run(replay, "wrong_target")
    assert unpublished["published"] == 0
    assert wrong_target["published"] == 1
    assert unpublished["echoes"] == wrong_target["echoes"] == [0, 0, 0]
    assert unpublished["valid"] == wrong_target["valid"] == 1


def test_published_startup_challenge_is_not_lost_by_encoding(replay):
    result = _run(replay, "ready")
    assert result["echoes"] == [1, 1, 1]
    assert result["reads"] == 3
    assert result["valid"] == 1


def test_out_of_range_mailbox_is_not_silently_accepted(replay):
    result = _run(replay, "invalid")
    assert result["reads"] == 1
    assert result["error"] == 1
    assert result["echoes"] == []


def test_saved_evidence_and_source_do_not_resolve_the_device_wait_stage():
    native = json.loads(BASE.with_name(BASE.name + "-native-result.json").read_bytes())["result"]
    frames = native["frame_coverage"]
    assert frames["aborted"] == 2
    assert frames["attempted"] == frames["produced"] == frames["consumed"] == 0
    assert frames["startup_response_challenge"] == 0
    source = HOST.read_text()
    assert '{"aborted", producer[3] + consumer[14]}' in source
    assert source.index("std::jthread load_thread") < source.index("EnqueueMeshWorkload(cq0")
    assert "startup_diagnostics" not in native
