"""Board-free acceptance tests for the resident producer/consumer harness."""

from __future__ import annotations

import hashlib
import json

import pytest

from enodia.tt.bench.resident_harness import (
    ResidentConfig,
    RingAccounting,
    build_measurement_record,
    cycle_budget_exceeded,
    frame_interval_statistics,
    required_samples_for_percentile,
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
    assert parsed["environment"]["image"] == "sha256:example"


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
