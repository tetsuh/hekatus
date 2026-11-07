"""Every real frame interval needs card-clock-bracketed transfer progress."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from enodia.tt.bench.issue120_coverage import (
    FRAME_COUNT,
    evaluate_frame_coverage,
    expected_frame_checksum,
)
from enodia.tt.bench.issue120_probe import ISSUE120_PROBES, planned_probe


def evidence():
    frames = []
    for index in range(3):
        frames.append({
            "frame_index": index,
            "start_ticks": 100 + 10 * index,
            "end_ticks": 105 + 10 * index,
            "left_ticks": 90 if index == 0 else 96 + 10 * index,
            "right_ticks": 102 + 10 * index,
            "left_sequence": 5 + index,
            "right_sequence": 6 + index,
            "progress_valid": True,
            "issued_challenge": index + 2,
            "response_challenge": index + 2,
            "payload_frame_id": index,
            "checksum": expected_frame_checksum(index),
            "work_iterations": 64,
        })
    return {
        "clock": "consumer_low32_wrap_tracked",
        "clock_gap_ambiguous": False,
        "frame_count": 3,
        "attempted": 3, "produced": 3, "consumed": 3,
        "overflow": 0, "aborted": 0,
        "producer_gated_after_load_observation": True,
        "startup_response_challenge": 1,
        "load_start_ticks": 90, "load_end_ticks": 130,
        "transfer_bytes": 4096, "repeat_count": 100,
        "frames": frames,
        "tail": {"left_ticks": 126, "right_ticks": 130,
                 "left_sequence": 8, "right_sequence": 9, "progress_valid": True,
                 "issued_challenge": 5, "response_challenge": 5},
    }


def test_progress_encoding_keeps_generation_and_challenge_in_disjoint_bits():
    from enodia.tt.bench.issue120_coverage import pack_progress

    assert pack_progress(2, 2) != pack_progress(3, 3)
    assert pack_progress(1, 0) == 1
    assert pack_progress(0x7FFFF, 0x1FFF) == 0xFFFFFFFF
    with pytest.raises(ValueError):
        pack_progress(0x80000, 1)
    with pytest.raises(ValueError):
        planned_probe("q4-frame-coverage", repeat_count=0x80000)


def test_all_frame_intervals_and_both_endpoints_are_required():
    result = evaluate_frame_coverage(evidence())
    assert result["coverage"] == "covered"
    assert result["verified_frames"] == 3
    assert result["completed_bytes_lower_bound"] == 4 * 4096


def test_generation_jumps_are_not_turned_into_unwitnessed_transfer_bytes():
    value = evidence()
    for frame, left, right in zip(value["frames"], (5, 9, 20), (9, 20, 21), strict=True):
        frame["left_sequence"] = left
        frame["right_sequence"] = right
    value["tail"]["left_sequence"] = 21
    value["tail"]["right_sequence"] = 30
    result = evaluate_frame_coverage(value)
    assert result["coverage"] == "covered"
    assert result["completed_bytes_lower_bound"] == 4 * 4096
    assert result["completed_bytes_during_frames_lower_bound"] == 2 * 4096


@pytest.mark.parametrize("index", [0, 1, 2])
def test_one_stalled_frame_cannot_be_rescued_by_host_envelope(index):
    value = evidence()
    value["frames"][index]["right_sequence"] = value["frames"][index]["left_sequence"]
    value["host_envelope"] = {"start_ns": 0, "end_ns": 10**20}
    value["queued_transfers"] = 10**9
    assert evaluate_frame_coverage(value)["coverage"] != "covered"


@pytest.mark.parametrize("mutation", [
    "missing_frame", "duplicate_frame", "reordered_frame", "overflow", "abort",
    "torn_payload", "bad_checksum", "late_start", "early_end", "tail_stall",
    "wrong_clock", "clock_gap", "ungated", "out_of_interval", "wrong_count",
])
def test_coverage_rejects_incomplete_or_inconsistent_evidence(mutation):
    value = copy.deepcopy(evidence())
    if mutation == "missing_frame":
        value["frames"].pop()
    elif mutation == "duplicate_frame":
        value["frames"][1]["frame_index"] = 0
    elif mutation == "reordered_frame":
        value["frames"].reverse()
    elif mutation == "overflow":
        value["overflow"] = 1
    elif mutation == "abort":
        value["aborted"] = 1
    elif mutation == "torn_payload":
        value["frames"][1]["progress_valid"] = False
    elif mutation == "bad_checksum":
        value["frames"][1]["checksum"] ^= 1
    elif mutation == "late_start":
        value["load_start_ticks"] = 101
    elif mutation == "early_end":
        value["load_end_ticks"] = 124
    elif mutation == "tail_stall":
        value["tail"]["right_sequence"] = value["tail"]["left_sequence"]
    elif mutation == "wrong_clock":
        value["clock"] = "host_monotonic"
    elif mutation == "clock_gap":
        value["clock_gap_ambiguous"] = True
    elif mutation == "ungated":
        value["producer_gated_after_load_observation"] = False
    elif mutation == "out_of_interval":
        value["frames"][1]["left_ticks"] = 104
    else:
        value["consumed"] = 2
    assert evaluate_frame_coverage(value)["coverage"] != "covered"


def test_progress_cannot_exceed_the_fixed_number_of_submissions():
    value = evidence()
    value["repeat_count"] = 7
    assert evaluate_frame_coverage(value)["coverage"] != "covered"


def test_frame_clock_cannot_move_backwards_or_observe_response_before_admission():
    value = evidence()
    value["frames"][1]["start_ticks"] = 104
    assert evaluate_frame_coverage(value)["coverage"] != "covered"
    value = evidence()
    value["frames"][1]["right_ticks"] = 109
    assert evaluate_frame_coverage(value)["coverage"] != "covered"
    value = evidence()
    value["startup_response_challenge"] = True
    assert evaluate_frame_coverage(value)["coverage"] != "covered"


def test_front_loaded_cpu_work_cannot_pass_even_if_dma_progresses():
    value = evidence()
    value["frames"][1]["response_challenge"] = 1
    value["host_envelope"] = {"start_ns": 0, "end_ns": 10**20}
    assert evaluate_frame_coverage(value)["coverage"] != "covered"
    value = evidence()
    value["tail"]["response_challenge"] = 1
    assert evaluate_frame_coverage(value)["coverage"] != "covered"


@pytest.mark.parametrize("value", [None, [], "covered"])
def test_non_object_evidence_is_not_evaluable(value):
    assert evaluate_frame_coverage(value)["coverage"] == "not_evaluable"


def test_no_frame_stream_means_no_coverage_even_with_a_witness():
    assert evaluate_frame_coverage({"payload_witness": True, "coverage": "covered"})["coverage"] != "covered"


def test_extended_card_clock_handles_real_wrap_without_high_register_reads():
    from enodia.tt.bench.issue120_probe import allocate_low_word_clock

    value = evidence()
    offset = (1 << 32) - 110
    clock = allocate_low_word_clock()
    assert clock.observe(90 + offset) == 90 + offset
    assert clock.observe((130 + offset) & 0xFFFFFFFF) == 130 + offset
    value["load_start_ticks"] += offset
    value["load_end_ticks"] += offset
    for frame in value["frames"]:
        for field in ("start_ticks", "end_ticks", "left_ticks", "right_ticks"):
            frame[field] += offset
    for field in ("left_ticks", "right_ticks"):
        value["tail"][field] += offset
    assert evaluate_frame_coverage(value)["coverage"] == "covered"


def test_new_coverage_probe_is_additive_fixed_work_and_bounded_memory():
    assert len(ISSUE120_PROBES) == 4
    small = planned_probe("q4-frame-coverage", repeat_count=1, first_kernel_run=True)
    large = planned_probe("q4-frame-coverage", repeat_count=262144, first_kernel_run=True)
    assert small.allocated_bytes == large.allocated_bytes
    assert small.resident_cores == ((0, 0), (1, 0))
    assert large.watcher is True
    assert large.outer_cap_seconds == 60
    assert FRAME_COUNT == 4096


def test_native_coverage_uses_real_ring_and_same_core_progress_not_host_time():
    root = Path(__file__).parents[1] / "enodia/tt/bench/issue120"
    producer = (root / "q4_frame_producer.cpp").read_text()
    consumer = (root / "q4_frame_consumer.cpp").read_text()
    host = (root / "q4_frame_probe.cpp").read_text()
    clock = (root / "q4_clock.hpp").read_text()
    assert "noc_semaphore_inc" in producer
    assert "frame_count" in producer
    assert "frame_interval_ticks" in producer
    assert "get_timestamp_32b" in clock
    assert "get_timestamp()" not in consumer + clock
    assert "attempt < 3" in consumer
    assert "left_sequence" in consumer
    assert "right_sequence" in consumer
    assert "frame_checksum" in consumer
    frame_loop = consumer.split("for (; consumed < frame_count", 1)[1].split("const Progress tail", 1)[0]
    assert "observe_change" not in frame_loop
    assert "issue_challenge" in frame_loop
    assert "read_core" in host
    assert "ReadFromDeviceL1" not in host
    assert "l1_barrier" not in host
    assert "std::span<uint8_t>" in host
    assert "response_challenge" in consumer + host
    assert "protocol::word" in consumer
    assert "protocol::pack" in host
    assert "protocol::word" in host
    assert "std::jthread" in host
    assert "enqueue_write_mesh_buffer" in host
    assert "synchronize" not in host.split("for (uint32_t repetition", 1)[1].split("load.end_ns", 1)[0]
