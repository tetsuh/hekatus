"""The full probe record must bind the every-frame proof to its fixed job."""

from __future__ import annotations

import copy

import pytest

from enodia.tt.bench import issue120_probe as probe
from enodia.tt.bench.issue120_coverage import (
    FRAME_COUNT,
    FRAME_PERIOD_TICKS,
    expected_frame_checksum,
)


@pytest.fixture(scope="module")
def valid_record():
    spec = probe.planned_probe("q4-frame-coverage", first_kernel_run=True)
    frames = [{
        "frame_index": index, "payload_frame_id": index,
        "start_ticks": 100 + 10 * index, "end_ticks": 105 + 10 * index,
        "left_ticks": 90 if index == 0 else 96 + 10 * index,
        "right_ticks": 102 + 10 * index,
        "left_sequence": 5 + index, "right_sequence": 6 + index,
        "issued_challenge": index + 2, "response_challenge": index + 2,
        "progress_valid": True, "checksum": expected_frame_checksum(index),
        "work_iterations": 64,
    } for index in range(FRAME_COUNT)]
    end = frames[-1]["end_ticks"]
    return {
        "schema": probe.PROBE_SCHEMA, "probe_id": spec.probe_id,
        "status": "complete", "errors": [], "coverage": "covered",
        "environment": {
            "board_model": "p150a", "serial": "fixture-serial", "pci": "0000:00:00.0",
            "firmware": "fixture", "kmd": "fixture", "image_digest": probe.PINNED_IMAGE_DIGEST,
            "cpu_model": "fixture CPU", "commit": "b" * 40,
        },
        "source": {"tt_metal_tag": "v0.75.0", "tt_metal_commit": probe.PINNED_SOURCE_REVISION,
                   "libtt_metal_sha256": probe.PINNED_LIBRARY_HASH},
        "resident_interval": {"start_ticks": 100, "end_ticks": end},
        "host_interval": {"start_ns": 1, "end_ns": 2},
        "work": {"repeat_count": spec.repeat_count, "transfer_count": spec.repeat_count,
                 "bytes_per_transfer": spec.transfer_bytes,
                 "bytes_total": spec.repeat_count * spec.transfer_bytes,
                 "host_mmio_reads": spec.repeat_count, "host_mmio_payload_bytes": spec.repeat_count * 4},
        "host_cpu_work": {"rounds": spec.host_cpu_rounds, "words_per_round": spec.host_cpu_words_per_round,
                          "outer_repetitions": spec.repeat_count,
                          "operations": spec.repeat_count * spec.host_cpu_rounds * spec.host_cpu_words_per_round},
        "memory": {"buffer_pool_size": spec.buffer_pool_size, "allocated_bytes": spec.allocated_bytes,
                   "host_payload_bytes": spec.transfer_bytes, "device_buffer_bytes": spec.transfer_bytes,
                   "auxiliary_bytes": spec.auxiliary_bytes, "max_outstanding_transfers": None,
                   "outstanding_bound": "fixed runtime ring with backpressure"},
        "outer_cap_seconds": 60, "watcher": True, "first_kernel_run": True,
        "plan": probe._plan_record(spec, first_kernel_run=True),
        "cpu_challenge_protocol": "direct_l1_span_read_then_fixed_cpu_work_then_cq1_write",
        "coverage_scope": "all_real_frame_completion_intervals_consumer_clock",
        "frame_coverage": {
            "clock": "consumer_low32_wrap_tracked", "clock_gap_ambiguous": False,
            "frame_count": FRAME_COUNT, "frame_period_ticks": FRAME_PERIOD_TICKS,
            "repeat_count": spec.repeat_count, "attempted": FRAME_COUNT,
            "produced": FRAME_COUNT, "consumed": FRAME_COUNT, "overflow": 0, "aborted": 0,
            "producer_gated_after_load_observation": True, "startup_response_challenge": 1,
            "load_start_ticks": 90, "load_end_ticks": end + 5,
            "transfer_bytes": spec.transfer_bytes, "frames": frames,
            "tail": {"left_ticks": end + 1, "right_ticks": end + 5,
                     "left_sequence": FRAME_COUNT + 5, "right_sequence": FRAME_COUNT + 6,
                     "issued_challenge": FRAME_COUNT + 2, "response_challenge": FRAME_COUNT + 2,
                     "progress_valid": True},
        },
    }


def test_full_record_requires_all_frames_and_ignores_host_envelope(valid_record):
    assert probe.validate_probe_record(valid_record) == []
    value = copy.deepcopy(valid_record)
    value["frame_coverage"]["frames"][FRAME_COUNT // 2]["response_challenge"] = 1
    with pytest.raises(probe.ProbeRecordError, match="coverage"):
        probe.validate_probe_record(value)


@pytest.mark.parametrize("field", ["operations", "rounds", "words_per_round", "outer_repetitions"])
def test_cpu_work_cannot_be_changed_without_changing_the_plan(valid_record, field):
    value = copy.deepcopy(valid_record)
    value["host_cpu_work"][field] += 1
    with pytest.raises(probe.ProbeRecordError, match="CPU"):
        probe.validate_probe_record(value)


def test_cpu_work_must_be_an_object(valid_record):
    value = copy.deepcopy(valid_record)
    value["host_cpu_work"] = None
    with pytest.raises(probe.ProbeRecordError, match="CPU"):
        probe.validate_probe_record(value)


def test_challenge_read_accounting_cannot_be_omitted(valid_record):
    value = copy.deepcopy(valid_record)
    del value["work"]["host_mmio_reads"]
    with pytest.raises(probe.ProbeRecordError, match="challenge"):
        probe.validate_probe_record(value)


def test_frame_counter_limit_must_match_the_fixed_repeat_count(valid_record):
    value = copy.deepcopy(valid_record)
    value["frame_coverage"]["repeat_count"] += 1
    with pytest.raises(probe.ProbeRecordError, match="counter limit"):
        probe.validate_probe_record(value)
