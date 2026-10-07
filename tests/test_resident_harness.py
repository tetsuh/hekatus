"""Board-free acceptance tests for the resident producer/consumer harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from enodia.tt.bench.resident_harness import (
    RESIDENT_SEMAPHORE_COUNT,
    SEMAPHORE_BYTES,
    UINT32_MAX,
    ResidentConfig,
    ResidentPreflightError,
    RingAccounting,
    build_measurement_record,
    cycle_budget_exceeded,
    failure_name,
    frame_interval_statistics,
    interval_ticks_for_microseconds,
    periodic_gap_decomposition,
    required_samples_for_percentile,
    run_budget_breakdown,
    run_budget_exceeded,
    select_failure_check,
    split_u64,
    startup_allowance_ticks,
    termination_reason,
    ticks_to_seconds,
    timestamp_digest,
    validate_configuration,
    validate_pinned_environment,
    validate_run_budget_fits_outer_cap,
    wrap_delta,
)
from enodia.tt.bench.run_resident import _parser, _resolve_watcher_mode, _runtime_u32


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


def test_failure_codes_and_precedence_are_explicit():
    assert failure_name(0) == "none"
    assert failure_name(2) == "producer_pacing_wait"
    producer = {
        "code": 2,
        "name": "producer_pacing_wait",
        "source": "producer",
        "elapsed_ticks": 12,
        "limit_ticks": 10,
    }
    consumer = {
        "code": 4,
        "name": "consumer_fixed_work_budget",
        "source": "consumer",
        "elapsed_ticks": 100,
        "limit_ticks": 90,
    }
    assert select_failure_check(producer, consumer)["name"] == "consumer_fixed_work_budget"
    assert select_failure_check()["name"] == "none"
    with pytest.raises(ValueError):
        failure_name(99)


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
        "critical_path_ticks": 2_701_350_000,
        "schedule_ticks": 2_836_350_000,
        "overlap_model": "producer pacing and consumer fixed work overlap; critical path is max",
        "startup_allowance_ms": 100,
        "budget_aiclk_mhz": 1_350,
        "startup_allowance_ticks": 135_000_000,
        "safety_margin_percent": 10,
        "watcher_overhead_margin_percent": 0,
        "total_margin_percent": 10,
        "safety_margin_ticks": 283_635_000,
        "run_budget_ticks": 3_119_985_000,
    }
    assert not run_budget_exceeded(
        elapsed_ticks=4 * config.frame_interval_ticks + 4 * config.cycle_budget,
        run_budget_ticks=breakdown["run_budget_ticks"],
    )
    assert interval_ticks_for_microseconds(microseconds=1_000, aiclk_mhz=800) == 800_000
    assert interval_ticks_for_microseconds(microseconds=1_000, aiclk_mhz=1_350) == 1_350_000
    assert split_u64(breakdown["run_budget_ticks"]) == (3_119_985_000, 0)
    watcher_breakdown = run_budget_breakdown(config, watcher=True)
    assert watcher_breakdown["watcher_overhead_margin_percent"] == 100
    assert watcher_breakdown["total_margin_percent"] == 110
    assert watcher_breakdown["run_budget_ticks"] == 5_956_335_000


def test_wrap_tracked_low_word_extension_is_monotonic_across_wrap():
    extended = 0xFFFFFFFE
    values = []
    for low in (0xFFFFFFFF, 0x00000000, 0x00000001, 0x00000010):
        extended += (low - (extended & 0xFFFFFFFF)) & 0xFFFFFFFF
        values.append(extended)
    assert values == [0xFFFFFFFF, 0x1_0000_0000, 0x1_0000_0001, 0x1_0000_0010]
    assert values == sorted(values)


def test_ticks_to_seconds_uses_aiclk_hz_conversion():
    assert ticks_to_seconds(ticks=1_350_000, aiclk_mhz=1_350) == pytest.approx(0.001)
    assert ticks_to_seconds(ticks=80_000_000, aiclk_mhz=800) == pytest.approx(0.1)


def test_periodic_gap_decomposition_preserves_quotients_and_remainders():
    result = periodic_gap_decomposition([6_363, 3_182, 6_363], period_frames=6_363)
    assert result["entries"] == [
        {"gap_frames": 6_363, "quotient": 1, "remainder": 0},
        {"gap_frames": 3_182, "quotient": 0, "remainder": 3_182},
        {"gap_frames": 6_363, "quotient": 1, "remainder": 0},
    ]
    assert result["aggregate_by_quotient_remainder"] == {"1,0": 2, "0,3182": 1}


def test_startup_allowance_is_100_ms_at_both_supported_clocks():
    assert startup_allowance_ticks(aiclk_mhz=800) == 80_000_000
    assert startup_allowance_ticks(aiclk_mhz=1_350) == 135_000_000


def test_run_budget_uses_64_bit_overflow_checks_and_scopes_errors():
    config = _config(
        frame_count=2_001,
        frame_interval_ticks=800_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
        budget_aiclk_mhz=800,
    )
    breakdown = run_budget_breakdown(config)
    assert breakdown["pacing_ticks"] == 1_600_800_000
    assert breakdown["run_budget_ticks"] == 1_848_880_000
    assert run_budget_exceeded(
        elapsed_ticks=breakdown["run_budget_ticks"],
        run_budget_ticks=breakdown["run_budget_ticks"],
    )
    with pytest.raises(ValueError):
        split_u64(1 << 64)
    with pytest.raises(ValueError):
        interval_ticks_for_microseconds(microseconds=0, aiclk_mhz=1_350)
    huge = _config(frame_count=UINT32_MAX, frame_interval_ticks=UINT32_MAX)
    with pytest.raises(ResidentPreflightError, match="64-bit"):
        run_budget_breakdown(huge)


def test_outer_cap_rejects_60000_frames_and_reports_safe_alternative():
    config = _config(
        frame_count=60_000,
        frame_interval_ticks=1_350_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
        outer_timeout_seconds=60,
        budget_aiclk_mhz=1_350,
    )
    with pytest.raises(ResidentPreflightError, match="outer cap"):
        validate_run_budget_fits_outer_cap(config, watcher=False)
    safe = _config(
        frame_count=54_445,
        frame_interval_ticks=1_350_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
        outer_timeout_seconds=60,
        budget_aiclk_mhz=1_350,
    )
    breakdown = validate_run_budget_fits_outer_cap(safe, watcher=False)
    assert breakdown["run_budget_ticks"] == 80_999_325_000
    longest = _config(
        frame_count=600_000,
        frame_interval_ticks=1_350_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
        outer_timeout_seconds=600,
        budget_aiclk_mhz=1_350,
    )
    with pytest.raises(ResidentPreflightError, match="outer cap"):
        validate_run_budget_fits_outer_cap(longest, watcher=False)


def test_resident_semaphore_l1_bytes_are_in_preflight_accounting():
    source = Path("enodia/tt/bench/resident_harness.py").read_text()
    assert RESIDENT_SEMAPHORE_COUNT == 4
    assert SEMAPHORE_BYTES == 4
    assert "RESIDENT_SEMAPHORE_COUNT * SEMAPHORE_BYTES" in source


@pytest.mark.parametrize("aiclk_mhz", [800, 1_350])
def test_600_second_timing_cap_maximum_safe_frame_boundary(aiclk_mhz):
    interval_ticks = aiclk_mhz * 1_000
    safe = _config(
        frame_count=545_354,
        frame_interval_ticks=interval_ticks,
        outer_timeout_seconds=600,
        budget_aiclk_mhz=aiclk_mhz,
    )
    breakdown = validate_run_budget_fits_outer_cap(safe)
    cap_ticks = 600 * aiclk_mhz * 1_000_000
    assert breakdown["run_budget_ticks"] <= cap_ticks
    expected = 479_999_520_000 if aiclk_mhz == 800 else 809_999_190_000
    assert breakdown["run_budget_ticks"] == expected

    rejected = _config(
        frame_count=545_355,
        frame_interval_ticks=interval_ticks,
        outer_timeout_seconds=600,
        budget_aiclk_mhz=aiclk_mhz,
    )
    with pytest.raises(ResidentPreflightError, match="outer cap"):
        validate_run_budget_fits_outer_cap(rejected)


def test_runtime_uint32_bounds_cover_kernel_arguments():
    assert validate_configuration(
        _config(frame_interval_ticks=UINT32_MAX)
    ).frame_interval_ticks == UINT32_MAX
    invalid_interval = _config(frame_interval_ticks=UINT32_MAX + 1)
    with pytest.raises(ResidentPreflightError, match="frame_interval_ticks.*uint32"):
        validate_configuration(invalid_interval)

    invalid_configs = {
        field: _config(**{field: UINT32_MAX + 1})
        for field in ("frame_count", "ring_pages", "work_per_frame", "cycle_budget")
    }
    for field, invalid_config in invalid_configs.items():
        with pytest.raises(ResidentPreflightError, match=f"{field}.*uint32"):
            validate_configuration(invalid_config)


def test_watcher_mode_requires_matching_cli_and_environment():
    assert _resolve_watcher_mode(False, None) is False
    assert _resolve_watcher_mode(True, "1") is True
    with pytest.raises(ResidentPreflightError, match="same mode"):
        _resolve_watcher_mode(True, None)
    with pytest.raises(ResidentPreflightError, match="same mode"):
        _resolve_watcher_mode(False, "1")
    with pytest.raises(ResidentPreflightError, match="unset or exactly 1"):
        _resolve_watcher_mode(False, "0")
    with pytest.raises(ResidentPreflightError, match="unset or exactly 1"):
        _resolve_watcher_mode(False, "")


def test_resident_parser_rejects_abbreviated_watcher_flag():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--out", "record.json", "--wat"])


def test_runtime_addresses_are_checked_at_uint32_boundary():
    assert _runtime_u32(UINT32_MAX, "address") == UINT32_MAX
    with pytest.raises(ResidentPreflightError, match="address.*uint32"):
        _runtime_u32(UINT32_MAX + 1, "address")


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
    too_much_work_at_800 = _config(frame_interval_ticks=800_000, fixed_work_ticks_per_frame=400_001)
    with pytest.raises(ValueError, match="half the frame interval"):
        validate_configuration(too_much_work_at_800)
    too_much_work_at_1350 = _config(frame_interval_ticks=1_350_000, fixed_work_ticks_per_frame=675_001)
    with pytest.raises(ValueError, match="half the frame interval"):
        validate_configuration(too_much_work_at_1350)


def test_configuration_rejects_outer_cap_and_core_clock_mismatch():
    overlong_timing = _config(outer_timeout_seconds=601)
    with pytest.raises(ValueError, match="600"):
        validate_configuration(overlong_timing)
    assert validate_configuration(_config(outer_timeout_seconds=600))
    overlong_watcher = _config(outer_timeout_seconds=61)
    with pytest.raises(ValueError, match="60"):
        validate_configuration(overlong_watcher, watcher=True)
    assert validate_configuration(_config(outer_timeout_seconds=60), watcher=True)
    mismatched_timestamp_core = _config(designated_timestamp_core=(0, 0))
    with pytest.raises(ValueError, match="designated_timestamp_core"):
        validate_configuration(mismatched_timestamp_core)
    same_cores = _config(consumer_core=(0, 0))
    with pytest.raises(ValueError, match="different cores"):
        validate_configuration(same_cores)


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
        startup_ticks=123,
        startup_ticks_valid=True,
        work_min_ticks=10,
        work_max_ticks=20,
        failure_check={
            "code": 4,
            "name": "consumer_fixed_work_budget",
            "source": "consumer",
            "elapsed_ticks": 123,
            "limit_ticks": 100,
            "unit": "device_clock_ticks",
        },
        cycle_budget_hit=False,
        kernel_error_flag=0,
        harness_commit="0123456789abcdef",
        environment={
            "board": {"serial": "redacted-board-serial"},
            "firmware": {"bundle": "19.6.0.0"},
            "kmd_version": "2.11.0",
            "image": "registry.example/tt@sha256:" + "a" * 64,
            "image_pinned": True,
        },
        power_trace="resident-power.csv",
    )
    encoded = json.dumps(record, allow_nan=False)
    parsed = json.loads(encoded)
    assert parsed["schema"] == "issue-12-stage-1-resident-v1"
    assert parsed["parameters"]["frame_interval_is_not_acquisition_rate_claim"] is True
    assert parsed["clock"]["timestamp_api"] == "get_timestamp_32b"
    assert parsed["clock"]["timestamp_semantics"] == "32-bit low word, software-extended (wrap-tracked)"
    assert parsed["raw_timestamps"]["count"] == 3
    assert '"timestamps":' not in encoded
    assert parsed["histogram"]["N"] == 2
    assert parsed["ring"]["producer_full_count"] == 2
    assert parsed["ring"]["consumer_empty_count"] == 1
    assert parsed["ring"]["dropped_frame_count"] == 97
    assert parsed["startup"]["observed_ticks"] == 123
    assert parsed["startup"]["observed_valid"] is True
    assert parsed["work_ticks"] == {
        "minimum": 10,
        "maximum": 20,
        "valid": True,
        "unit": "device_clock_ticks",
    }
    assert parsed["failure_check"]["name"] == "consumer_fixed_work_budget"
    assert parsed["failure_check"]["elapsed_ticks"] == 123
    assert parsed["failure_check"]["limit_ticks"] == 100
    assert parsed["timing_evidence"] is False
    assert parsed["ring"]["full_ring_policy"] == "drop_new_frame_without_waiting_or_overwriting"
    assert parsed["parameters"]["full_ring_policy"] == "drop_new_frame_without_waiting_or_overwriting"
    assert parsed["parameters"]["attempted_frame_count"] == 100
    assert parsed["environment"]["image"] == "registry.example/tt@sha256:" + "a" * 64


def test_kernel_protocol_uses_accessor_ring_metadata_and_budgeted_waits():
    producer = Path("enodia/tt/bench/kernels/resident_producer.cpp").read_text()
    consumer = Path("enodia/tt/bench/kernels/resident_consumer.cpp").read_text()

    assert "get_timestamp()" not in producer
    assert "WALL_CLOCK_H" not in producer
    assert "get_timestamp_32b()" in producer
    assert "extended += static_cast<std::uint32_t>(low - static_cast<std::uint32_t>(extended))" in producer
    assert "noc_async_write_page" in producer
    assert "noc_async_read_page" not in producer
    assert "control.get_noc_addr(0)" in producer
    assert "consumer_x" not in producer
    assert "noc_inline_dw_write" not in producer
    assert "noc_semaphore_inc" in producer
    assert "get_semaphore(free_semaphore_id)" in producer
    assert "while (consumed_required" not in producer
    assert "Drop-new policy" in producer
    assert "run_budget_ticks" in producer
    assert "failure_producer_pacing_wait" in producer
    assert "failure_run_wide_budget" in producer
    assert "failure_elapsed_ticks" in producer
    assert "control_probe" not in producer
    assert "noc_async_read_page(0, control" not in producer
    assert "error_semaphore_id" in producer
    assert "*error_sem" in producer
    assert "noc_semaphore_inc(done_noc, 1)" in producer
    assert "clock.read() - run_start >= run_budget_ticks" in producer

    assert "get_timestamp()" not in consumer
    assert "WALL_CLOCK_H" not in consumer
    assert "get_timestamp_32b()" in consumer
    assert "extended += static_cast<std::uint32_t>(low - static_cast<std::uint32_t>(extended))" in consumer
    assert "noc_semaphore_inc" in consumer
    assert "get_semaphore(ready_semaphore_id)" in consumer
    assert "get_semaphore(done_semaphore_id)" in consumer
    assert "error_semaphore_id" in consumer
    assert "noc_semaphore_inc(error_noc, 1)" in consumer
    assert "*done_sem" in consumer
    assert "control_local" not in consumer
    assert "control_done_word" not in consumer
    assert "control_produced_word" not in consumer
    assert "clock.read() - run_start >= run_budget_ticks" in consumer
    assert "failure_consumer_empty_wait" in consumer
    assert "failure_consumer_fixed_work_budget" in consumer
    assert "failure_run_wide_budget" in consumer
    assert "failure_elapsed_ticks" in consumer
    assert "per_frame_work_budget_ticks" in consumer
    assert "noc_inline_dw_write" not in consumer
    assert "invalidate_l1_cache" in consumer


def test_control_page_is_consumer_l1_and_passed_to_both_accessors():
    runner = Path("enodia/tt/bench/run_resident.py").read_text()
    assert "control = ttnn.zeros" in runner
    assert "memory_config=_sharded_pages_config(ttnn, config, 1)" in runner
    assert "control.buffer_address()" in runner
    assert "control_compile" in runner
    assert "producer_anchor" in runner
    assert "SemaphoreDescriptor(0" in runner
    assert "SemaphoreDescriptor(1" in runner
    assert "SemaphoreDescriptor(2" in runner
    assert "SemaphoreDescriptor(3" in runner
    assert "consumer_compile = [\n        *ring_compile,\n        *producer_anchor_compile" in runner
    assert "allow-budget-margin-over-cap" not in runner
    assert "_resolve_watcher_mode(args.watcher" in runner
    assert "validate_configuration(config, watcher=watcher)" in runner


def test_consumer_budget_failure_cancels_producer_without_control_sync():
    consumer = Path("enodia/tt/bench/kernels/resident_consumer.cpp").read_text()
    producer = Path("enodia/tt/bench/kernels/resident_producer.cpp").read_text()
    assert "error_sent" in consumer
    assert "if (error_flag != 0)" in consumer
    assert "*error_sem != 0" in producer
    assert "attempts_started = attempted + 1" in producer
    assert "ready_count += 1" in producer
    assert "frames_dropped += 1" in producer


def test_every_resident_noc_signal_and_write_has_a_matching_barrier():
    producer = Path("enodia/tt/bench/kernels/resident_producer.cpp").read_text()
    consumer = Path("enodia/tt/bench/kernels/resident_consumer.cpp").read_text()

    def assert_signal_barriers(source, signal_count):
        assert source.count("noc_semaphore_inc(") == signal_count
        positions = []
        cursor = 0
        while (position := source.find("noc_semaphore_inc(", cursor)) >= 0:
            positions.append(position)
            cursor = position + 1
        for position in positions:
            barrier = source.find("noc_async_atomic_barrier()", position)
            assert barrier >= 0
            next_signal = source.find("noc_semaphore_inc(", position + 1)
            assert next_signal < 0 or barrier < next_signal

    def assert_write_barriers(source, write_count):
        assert source.count("noc_async_write_page(") == write_count
        positions = []
        cursor = 0
        while (position := source.find("noc_async_write_page(", cursor)) >= 0:
            positions.append(position)
            cursor = position + 1
        for position in positions:
            barrier = source.find("noc_async_write_barrier()", position)
            assert barrier >= 0
            next_write = source.find("noc_async_write_page(", position + 1)
            assert next_write < 0 or barrier < next_write

    assert "noc_async_read_page(" not in producer + consumer
    assert_signal_barriers(producer, 2)
    assert_signal_barriers(consumer, 4)
    assert_write_barriers(producer, 3)
    assert_write_barriers(consumer, 2)
    assert producer.index("noc_async_write_page(ready_count") < producer.index(
        "noc_semaphore_inc(ready_noc"
    )
    assert producer.index("noc_async_write_page(0, stats") < producer.index(
        "noc_semaphore_inc(done_noc"
    )
    assert "failure_consumer_empty_wait" in consumer
    assert "failure_consumer_fixed_work_budget" in consumer
    assert "failure_run_wide_budget" in consumer
    assert "Drop-new policy" in producer


def test_resident_environment_requires_verified_digest_image():
    valid = {
        "image": "registry.example/tt@sha256:" + "a" * 64,
        "image_pinned": True,
    }
    validate_pinned_environment(valid)
    with pytest.raises(ResidentPreflightError, match="image_pinned"):
        validate_pinned_environment({"image": "registry.example/tt:latest", "image_pinned": False})
    with pytest.raises(ResidentPreflightError, match="image_pinned"):
        validate_pinned_environment({"image": "registry.example/tt@sha256:" + "A" * 64, "image_pinned": True})


def test_record_rejects_missing_environment_provenance():
    config = _config()
    with pytest.raises(ValueError, match="environment"):
        build_measurement_record(
            config=config,
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
