"""Board-free acceptance tests for the resident producer/consumer harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from enodia.tt.bench.resident_harness import (
    ResidentConfig,
    ResidentPreflightError,
    RingAccounting,
    build_measurement_record,
    cycle_budget_exceeded,
    frame_interval_statistics,
    interval_ticks_for_microseconds,
    required_samples_for_percentile,
    run_budget_breakdown,
    run_budget_exceeded,
    split_u64,
    termination_reason,
    timestamp_digest,
    validate_configuration,
    wrap_delta,
)


def _config(**overrides) -> ResidentConfig:
    values = {
        "frame_count": 100,
        "frame_interval_ticks": 1_000,
        "producer_core": (0, 0),
        "consumer_core": (1, 0),
        "ring_pages": 4,
        "work_per_frame": 64,
        "designated_timestamp_core": (1, 0),
        "cycle_budget": 10_000,
        "outer_timeout_seconds": 60,
        "fixed_work_ticks_per_frame": 100,
    }
    values.update(overrides)
    return ResidentConfig(**values)


def test_ring_accounting_counts_full_and_empty_transitions():
    ring = RingAccounting(capacity=2)

    assert ring.reserve_producer() == 0
    ring.publish()
    assert ring.reserve_producer() == 1
    ring.publish()
    assert ring.reserve_producer() is None
    assert ring.producer_full_count == 1
    assert ring.attempted == 3
    assert ring.produced == 2
    assert ring.dropped == 1

    assert ring.consume() == 0
    assert ring.consume() == 1
    assert ring.consume() is None
    assert ring.consumer_empty_count == 1

    assert ring.reserve_producer() == 0
    ring.publish()
    assert ring.consume() == 0


def test_wrap_delta_is_unsigned_and_wrap_safe():
    assert wrap_delta(0x00000005, 0xFFFFFFFE, bits=32) == 7
    assert wrap_delta(0x0000000000000005, 0xFFFFFFFFFFFFFFFE, bits=64) == 7
    with pytest.raises(ValueError):
        wrap_delta(1, 0, bits=31)


def test_cycle_budget_and_termination_are_explicit():
    assert not cycle_budget_exceeded(elapsed_ticks=99, cycle_budget=100)
    assert cycle_budget_exceeded(elapsed_ticks=100, cycle_budget=100)
    assert termination_reason(frame_count_reached=True, cycle_budget_hit=False, outer_timeout=False) == (
        "frame_count"
    )
    assert termination_reason(frame_count_reached=False, cycle_budget_hit=True, outer_timeout=False) == (
        "cycle_budget"
    )
    assert termination_reason(frame_count_reached=False, cycle_budget_hit=False, outer_timeout=True) == (
        "outer_timeout"
    )


def test_run_budget_covers_n_frames_interval_work_and_margin():
    config = _config(
        frame_count=2_001,
        frame_interval_ticks=1_350_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
    )
    breakdown = run_budget_breakdown(config)
    assert breakdown == {
        "frame_count": 2_001,
        "frame_interval_ticks": 1_350_000,
        "pacing_ticks": 2_701_350_000,
        "per_frame_work_budget_ticks": 100_000,
        "cycle_budget_ticks_per_frame": 10_000_000,
        "fixed_work_ticks": 200_100_000,
        "safety_margin_percent": 10,
        "safety_margin_ticks": 290_145_000,
        "run_budget_ticks": 3_191_595_000,
    }
    assert not run_budget_exceeded(
        elapsed_ticks=4 * config.frame_interval_ticks + 4 * config.cycle_budget,
        run_budget_ticks=breakdown["run_budget_ticks"],
    )
    assert interval_ticks_for_microseconds(microseconds=1_000, aiclk_mhz=800) == 800_000
    assert interval_ticks_for_microseconds(microseconds=1_000, aiclk_mhz=1_350) == 1_350_000
    assert split_u64(breakdown["run_budget_ticks"]) == (3_191_595_000, 0)


def test_run_budget_uses_64_bit_overflow_checks_and_scopes_errors():
    config = _config(
        frame_count=2_001,
        frame_interval_ticks=800_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
    )
    breakdown = run_budget_breakdown(config)
    assert breakdown["pacing_ticks"] == 1_600_800_000
    assert breakdown["run_budget_ticks"] == 1_980_990_000
    assert run_budget_exceeded(
        elapsed_ticks=breakdown["run_budget_ticks"],
        run_budget_ticks=breakdown["run_budget_ticks"],
    )
    with pytest.raises(ValueError):
        split_u64(1 << 64)
    with pytest.raises(ValueError):
        interval_ticks_for_microseconds(microseconds=0, aiclk_mhz=1_350)
    huge = _config(frame_count=2**63, frame_interval_ticks=2**63)
    with pytest.raises(ResidentPreflightError, match="64-bit"):
        run_budget_breakdown(huge)


def test_ring_drop_policy_drains_without_producer_wait():
    ring = RingAccounting(capacity=2)
    assert ring.reserve_producer() == 0
    ring.publish()
    assert ring.reserve_producer() == 1
    ring.publish()
    assert ring.reserve_producer() is None
    assert ring.drain_complete(producer_done=False) is False
    assert ring.consume() == 0
    assert ring.consume() == 1
    assert ring.drain_complete(producer_done=True) is True


def test_fixed_work_preflight_uses_half_interval_boundary():
    assert validate_configuration(
        _config(frame_interval_ticks=800_000, fixed_work_ticks_per_frame=400_000)
    )
    assert validate_configuration(
        _config(frame_interval_ticks=1_350_000, fixed_work_ticks_per_frame=675_000)
    )
    with pytest.raises(ValueError, match="half the frame interval"):
        validate_configuration(
            _config(frame_interval_ticks=800_000, fixed_work_ticks_per_frame=400_001)
        )
    with pytest.raises(ValueError, match="half the frame interval"):
        validate_configuration(
            _config(frame_interval_ticks=1_350_000, fixed_work_ticks_per_frame=675_001)
        )


def test_configuration_rejects_outer_cap_and_core_clock_mismatch():
    with pytest.raises(ValueError, match="600"):
        validate_configuration(_config(outer_timeout_seconds=601))
    with pytest.raises(ValueError, match="designated_timestamp_core"):
        validate_configuration(_config(designated_timestamp_core=(0, 0)))
    with pytest.raises(ValueError, match="different cores"):
        validate_configuration(_config(consumer_core=(0, 0)))


def test_histogram_percentiles_have_exact_sample_thresholds():
    assert required_samples_for_percentile(0.50) == 40
    assert required_samples_for_percentile(0.99) == 2_000
    assert required_samples_for_percentile(0.999) == 20_000
    assert required_samples_for_percentile(0.9999) == 200_000

    short = frame_interval_statistics([10, 20, 30], bin_width_ticks=1)
    assert short["N"] == 3
    assert short["min_ticks"] == 10
    assert short["max_ticks"] == 30
    for name in ("p50", "p99", "p99_9", "p99_99"):
        assert short["percentiles"][name]["status"] == "insufficient"
        assert "value_ticks" not in short["percentiles"][name]

    enough = frame_interval_statistics(list(range(40)), bin_width_ticks=5)
    assert enough["percentiles"]["p50"]["status"] == "ok"
    assert enough["percentiles"]["p99"]["status"] == "insufficient"
    assert enough["histogram"]["bin_width_ticks"] == 5
    assert sum(item["count"] for item in enough["histogram"]["bins"]) == 40


def test_timestamp_digest_redacts_raw_values():
    timestamps = [0x100000000, 0x100000007, 0x10000000F]
    redacted = timestamp_digest(timestamps)
    expected = b"".join(value.to_bytes(8, "little") for value in timestamps)
    assert redacted == {"count": 3, "sha256": hashlib.sha256(expected).hexdigest()}
    assert "timestamps" not in redacted


def test_record_schema_is_strict_and_excludes_raw_timestamps():
    config = _config()
    record = build_measurement_record(
        config=config,
        aiclk_mhz=1_350,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=2,
        consumer_empty_count=1,
        cycle_budget_hit=False,
        kernel_error_flag=0,
        harness_commit="0123456789abcdef",
        environment={
            "board": {"serial": "redacted-board-serial"},
            "firmware": {"bundle": "19.6.0.0"},
            "kmd_version": "2.11.0",
            "image": "sha256:example",
        },
        power_trace="resident-power.csv",
    )
    encoded = json.dumps(record, allow_nan=False)
    parsed = json.loads(encoded)
    assert parsed["schema"] == "issue-12-stage-1-resident-v1"
    assert parsed["parameters"]["frame_interval_is_not_acquisition_rate_claim"] is True
    assert parsed["raw_timestamps"]["count"] == 3
    assert '"timestamps":' not in encoded
    assert parsed["histogram"]["N"] == 2
    assert parsed["ring"]["producer_full_count"] == 2
    assert parsed["ring"]["consumer_empty_count"] == 1
    assert parsed["ring"]["dropped_frame_count"] == 97
    assert parsed["ring"]["full_ring_policy"] == "drop_new_frame_without_waiting_or_overwriting"
    assert parsed["parameters"]["full_ring_policy"] == "drop_new_frame_without_waiting_or_overwriting"
    assert parsed["parameters"]["attempted_frame_count"] == 100
    assert parsed["environment"]["image"] == "sha256:example"


def test_kernel_protocol_uses_accessor_ring_metadata_and_budgeted_waits():
    producer = Path("enodia/tt/bench/kernels/resident_producer.cpp").read_text()
    consumer = Path("enodia/tt/bench/kernels/resident_consumer.cpp").read_text()

    assert "noc_async_write_page" in producer
    assert "noc_async_read_page" in producer
    assert "get_noc_addr" not in producer
    assert "noc_inline_dw_write" not in producer
    assert "noc_semaphore" not in producer
    assert "while (consumed_required" not in producer
    assert "Drop-new policy" in producer
    assert "ready_word" in producer and "free_word" in producer
    assert "run_budget_ticks" in producer
    assert "get_timestamp() - run_start >= run_budget_ticks" in producer

    assert "ready_word" in consumer and "free_word" in consumer
    assert "control_local[control_error_word] = 1" in consumer
    assert "get_timestamp() - run_start >= run_budget_ticks" in consumer
    assert "per_frame_work_budget_ticks" in consumer
    assert "noc_inline_dw_write" not in consumer
    assert "noc_semaphore" not in consumer


def test_record_rejects_missing_environment_provenance():
    with pytest.raises(ValueError, match="environment"):
        build_measurement_record(
            config=_config(),
            aiclk_mhz=1_350,
            timestamps=[1, 2],
            producer_full_count=0,
            consumer_empty_count=0,
            cycle_budget_hit=False,
            kernel_error_flag=0,
            harness_commit="0123456789abcdef",
            environment={},
            power_trace="resident-power.csv",
        )
