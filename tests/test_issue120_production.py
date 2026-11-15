"""Board-free contracts for the additive production instrument."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from enodia.tt.bench.issue120_production import plan, protocol

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "enodia/tt/bench/issue120_production"


def test_long_wire_domain_and_transition_binding():
    entries = [(1, 0), (2, 1), (3, 500002)]
    protocol.validate_transitions(entries, frame_count=500000, repeat_count=24439669)
    for sequence, challenge in [(1, 0), (2, 1), (24439669, 500002)]:
        encoded = protocol.encode(sequence, challenge)
        assert protocol.decode(*encoded) == (sequence, challenge)
        words = protocol.payload(sequence, challenge)
        assert protocol.observe(words, entries, frame_count=500000,
                                repeat_count=24439669) == (sequence, challenge)
    assert protocol.encode(524288, 8192) != protocol.encode(1, 0)


def test_torn_multiword_tuple_never_uses_declared_valid_flag():
    entries = [(1, 0), (3, 1)]
    # Each limb is individually plausible; (sequence1,challenge1) was not sent.
    torn = protocol.payload(1, 1)
    with pytest.raises(ValueError, match="association"):
        protocol.observe(torn, entries, frame_count=3, repeat_count=100)
    mixed = protocol.payload(3, 1)
    mixed[17] = protocol.payload(2, 0)[17]
    with pytest.raises(ValueError, match="torn"):
        protocol.observe(mixed, entries, frame_count=3, repeat_count=100)


@pytest.mark.parametrize("entries", [[], [(2, 0)], [(1, 0), (1, 1)],
                                     [(1, 2), (2, 1)], [(1, 0), (2, 6)]])
def test_invalid_transition_maps(entries):
    with pytest.raises(ValueError):
        protocol.validate_transitions(entries, frame_count=3, repeat_count=100)


def test_wire_overflow_and_fixed_memory_formula():
    with pytest.raises(ValueError):
        protocol.encode(2**32 - 1, 1)
    with pytest.raises(ValueError):
        protocol.admit_domain(frame_count=500000, repeat_count=2**32 - 1)
    a = plan.memory_bounds(frame_count=500000, repeat_count=1000000)
    b = plan.memory_bounds(frame_count=500000, repeat_count=24439669)
    assert a == b
    assert a["resident_timestamp_bytes"] == 2048000000
    assert a["transition_capacity"] == 500003


def test_precommitted_estimate_and_control_configuration():
    estimate = plan.work_estimate()
    assert estimate["candidate_repetitions"] == 24439669
    assert estimate["operations"] == 800839073792
    assert estimate["submitted_bytes"] == 100104884224
    assert estimate["slow_planning_seconds"] < 580
    control = plan.run_spec("production-v1-control-watch")
    loaded = plan.run_spec("production-v1-load-watch")
    assert plan.program_configuration(control) == plan.program_configuration(loaded)
    assert control.condition == "control"
    assert loaded.condition == "loaded"
    assert control.watcher and loaded.watcher
    assert control.cap_seconds == 60
    assert plan.run_spec("production-v1-loaded-500000").cap_seconds == 600


def test_cumulative_resident_checksum_is_not_q4_reset_checksum():
    accumulator = 0
    for frame in range(4):
        expected = accumulator
        for work in range(64):
            word = work % 1024
            i = (frame + word) & 0x7FF
            q = (3 * frame + 5 * word) & 0x7FF
            expected = (((expected * 33) & 0xFFFFFFFF) ^ (i | (q << 16))
                        ^ (work + frame))
        accumulator = protocol.resident_checksum(accumulator, frame, 64)
        assert accumulator == expected


def test_cpp_protocol_and_actual_callable_host_sequence(tmp_path):
    unit = tmp_path / "seam.cpp"
    unit.write_text(r'''
#include "protocol.hpp"
#include "host_sequence.hpp"
#include "host_protocol.hpp"
#include <cassert>
#include <stdexcept>
#include <vector>
int main() {
    namespace p = issue120_production;
    static_assert(p::encode(24439669,500002).sum == 24939671);
    std::vector<int> order;
    bool active = false;
    auto start = [&] { active = true; order.push_back(0); };
    auto enqueue = [&] { assert(active); order.push_back(1); };
    auto join = [&] { assert(active); active = false; order.push_back(2); };
    auto finish0 = [&] { assert(!active); order.push_back(3); };
    auto finish1 = [&] { assert(!active); order.push_back(4); };
    p::host_sequence(start, enqueue, join, finish0, finish1);
    assert((order == std::vector<int>{0,1,2,3,4}));
    order.clear();
    try { p::host_sequence(start, [&] { throw std::runtime_error("enqueue"); },
                          join, finish0, finish1); assert(false); }
    catch(const std::runtime_error&) { assert(!active); }
    assert((order == std::vector<int>{0,2,3,4}));
    order.clear(); bool stopped = false;
    try { p::host_sequence([&] { start(); throw std::runtime_error("partial start"); },
                          enqueue, [&] { assert(stopped); join(); }, finish0, finish1,
                          [&] { stopped = true; order.push_back(9); }); assert(false); }
    catch(const std::runtime_error&) { assert(!active); }
    assert((order == std::vector<int>{0,9,2,3,4}));
    order.clear();
    try { p::host_sequence(start, enqueue, [&] { join(); throw std::runtime_error("worker"); },
                          finish0, finish1); assert(false); }
    catch(const std::runtime_error&) { assert(!active); }
    assert((order == std::vector<int>{0,1,2,3,4}));
    std::vector<unsigned> producer{10,11}, consumer{20,21,22};
    auto translate = [](p::XY c) { return p::XY{c.x+1,c.y+2}; };
    p::append_peers(producer, consumer, 2, 3, translate);
    assert((producer == std::vector<unsigned>{10,11,2,2}));
    assert((consumer == std::vector<unsigned>{20,21,22,1,2}));
    unsigned challenge = 0, read_value = 0, writes = 0;
    p::Traffic facts(3);
    p::submit_fixed(3,3, facts,
        [&] { read_value = challenge; return challenge; },
        [&] { ++challenge; },
        [&](const auto& page) { assert(page[1]-page[0] == read_value); ++writes; });
    assert(facts.emitted == 3 && writes == 3 && facts.transition_count == 3);
    p::Traffic failed(3); challenge = 0; writes = 0;
    try { p::submit_fixed(3,3, failed,
        [&] { return challenge; }, [&] { ++challenge; },
        [&](const auto&) { if (++writes == 2) throw std::runtime_error("write"); });
        assert(false);
    } catch(const std::runtime_error&) {}
    assert(failed.emitted == 1 && failed.transition_count == 1);
}
''')
    exe = tmp_path / "seam"
    subprocess.run(["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    "-pthread", "-I", str(SOURCE), str(unit), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=5)


def test_versioned_kernel_adapter_preserves_base_bytes(tmp_path):
    from enodia.tt.bench.issue120_production import kernel_adapter

    paths = [ROOT / "enodia/tt/bench/kernels" / ("resident_" + role + ".cpp")
             for role in ("producer", "consumer")]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    outputs = kernel_adapter.generate(tmp_path)
    assert set(outputs) == {"producer", "consumer"}
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    # Kernel derivation is a source contract; host execution order is tested above.
    for role in outputs:
        text = outputs[role].read_text()
        assert "get_arg_val" in text
        assert "issue120_production" in text
    checksum = "accumulator = (accumulator * 33u) ^ payload[word] ^ (work + frames_consumed);"
    assert checksum in outputs["consumer"].read_text()


def test_runner_default_is_board_free_and_unauthorized_run_stops(monkeypatch, capsys):
    from enodia.tt.bench.issue120_production import runner

    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: called.append(a))
    assert runner.main([]) == 0
    assert json.loads(capsys.readouterr().out)["board_access"] is False
    assert runner.main(["--run", "production-v1-load-watch"]) == 2
    assert called == []
