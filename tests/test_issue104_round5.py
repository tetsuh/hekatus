"""Round-five shared resident-record catalog and runner regressions."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from enodia.tt.bench import run_resident
from enodia.tt.bench.resident_harness import (
    AUDITED_CLOCK_SOURCE_IMAGE,
    ResidentConfig,
    build_measurement_record,
    build_rejection_record,
)
from enodia.tt.bench.resident_record import (
    REQUIRED_PAIR_FIELDS,
    REQUIRED_SAMPLER_GAP_FIELDS,
    RESIDENT_INVARIANT_CATALOG,
    RESIDENT_RECORD_FIELD_MATRIX,
    RESIDENT_RECORD_SCHEMA_MARKER,
    TIMING_EVIDENCE_COMPATIBILITY_BRANCHES,
    TIMING_EVIDENCE_UNVERIFIED_AICLK_SOURCES,
    build_outlier_analysis,
    validate_resident_record,
)
from tools.analyze_issue104_wrap import validate_record_invariants

RUN_START = "2026-01-01T00:00:00.500000+00:00"
RUN_END = "2026-01-01T00:00:01.500000+00:00"


def _config(frame_count: int = 3) -> ResidentConfig:
    return ResidentConfig(
        frame_count=frame_count,
        frame_interval_ticks=1_000_000,
        producer_core=(0, 0),
        consumer_core=(1, 0),
        ring_pages=4,
        work_per_frame=64,
        designated_timestamp_core=(1, 0),
        cycle_budget=10_000,
        outer_timeout_seconds=60,
        fixed_work_ticks_per_frame=100,
        budget_aiclk_mhz=1_350,
    )


def _environment(*, mode: str = "explicit") -> dict:
    return {
        "board": {"serial": "board", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kmd_version": "2.11.0",
        "image": AUDITED_CLOCK_SOURCE_IMAGE,
        "image_pinned": True,
        "harness_commit": "a" * 40,
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": {
            "mode": mode,
            "interval_seconds": 1.0 if mode == "explicit" else None,
            "power_trace": "required" if mode != "off" else "absent_by_design",
            "timing_evidence": "available" if mode != "off" else "diagnostic_only",
        },
        "aiclk_mhz_observed": [1_350],
    }


def _trace(path: Path) -> None:
    path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,75,800,60\n"
        "2026-01-01T00:00:01+00:00,76,1350,61\n"
        "2026-01-01T00:00:02+00:00,75,1350,61\n"
    )


def _valid_record(
    tmp_path: Path, timestamps: list[int] | None = None
) -> tuple[dict, Path, list[int]]:
    trace = tmp_path / "power.csv"
    _trace(trace)
    timestamps = [1_000, 2_000, 3_000] if timestamps is None else timestamps
    record = build_measurement_record(
        config=_config(frame_count=len(timestamps)),
        aiclk_mhz=1_350,
        timestamps=timestamps,
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=_environment(),
        power_trace=trace.name,
        power_trace_path=trace,
        run_start=RUN_START,
        run_end=RUN_END,
        timing_evidence=True,
    )
    report = validate_resident_record(
        record, timestamps=timestamps, power_trace_path=trace
    )
    assert report["valid"] is True
    assert report["out_of_scope"] is not True
    assert record["resident_record_schema"] == RESIDENT_RECORD_SCHEMA_MARKER
    return record, trace, timestamps


def _set_sampler_pair(record: dict, mode: str, interval):
    for sampler in (
        record["telemetry_sampler"],
        record["environment"]["telemetry_sampler"],
    ):
        sampler.update(mode=mode, interval_seconds=interval)


def _mutations(record: dict, trace: Path, timestamps: list[int]):
    del trace, timestamps
    return {
        "parameters.frame_count": lambda r: r["parameters"].update(frame_count=4),
        "parameters.attempted_frame_count": lambda r: r["parameters"].update(attempted_frame_count=2),
        "parameters.produced_frame_count": lambda r: r["parameters"].update(produced_frame_count=2),
        "parameters.dropped_frame_count": lambda r: r["parameters"].update(dropped_frame_count=1),
        "parameters.aborted_attempts": lambda r: r["parameters"].update(aborted_attempts=1),
        "raw_timestamps.count": lambda r: r["raw_timestamps"].update(count=4),
        "histogram.N": lambda r: r["histogram"].update(N=1),
        "histogram.bin_total": lambda r: r["histogram"]["histogram"]["bins"][0].update(count=3),
        "ring.attempted_frame_count": lambda r: r["ring"].update(attempted_frame_count=2),
        "ring.produced_frame_count": lambda r: r["ring"].update(produced_frame_count=2),
        "ring.consumed_frame_count": lambda r: r["ring"].update(consumed_frame_count=2),
        "ring.dropped_frame_count": lambda r: r["ring"].update(dropped_frame_count=1),
        "clock.aiclk_mhz": lambda r: r["clock"].update(aiclk_mhz=1_349),
        "power_trace.sample_count": lambda r: r["power_clock_provenance"].update(sample_count=4),
        "power_trace.csv_row_count": lambda r: r["power_clock_provenance"].update(csv_row_count=4),
        "power_trace.valid_row_count": lambda r: r["power_clock_provenance"].update(valid_row_count=0),
        "power_trace.in_run_valid_row_count": lambda r: r["power_clock_provenance"].update(in_run_valid_row_count=0),
        "power_trace.first_timestamp": lambda r: r["power_clock_provenance"].update(first_timestamp="2025-01-01T00:00:00+00:00"),
        "power_trace.last_timestamp": lambda r: r["power_clock_provenance"].update(last_timestamp="2025-01-01T00:00:00+00:00"),
        "power_trace.run_start": lambda r: r["power_clock_provenance"].update(run_start="2026-01-01T00:00:01.500000+00:00"),
        "power_trace.coverage_complete": lambda r: r["power_clock_provenance"].update(coverage_complete=False),
        "power_trace.columns": lambda r: r["power_clock_provenance"].update(columns=["wrong"]),
        "power_trace.sha256": lambda r: r.update(power_trace_sha256="0" * 64),
        "clock.aiclk_source": lambda r: r["clock"].update(aiclk_source="snapshot_only"),
        "sampler.trace_presence": lambda r: r["telemetry_sampler"].update(power_trace="absent_by_design"),
        "sampler.default_interval": lambda r: _set_sampler_pair(r, "default", 5.0),
        "sampler.off_interval": lambda r: _set_sampler_pair(r, "off", 2.0),
        "sampler.explicit_zero_interval": lambda r: _set_sampler_pair(r, "explicit", 0),
        "sampler.explicit_bool_interval": lambda r: _set_sampler_pair(r, "explicit", True),
        "sampler.explicit_string_interval": lambda r: _set_sampler_pair(r, "explicit", "1"),
        "sampler.explicit_nan_interval": lambda r: _set_sampler_pair(r, "explicit", float("nan")),
        "sampler.explicit_infinite_interval": lambda r: _set_sampler_pair(r, "explicit", float("inf")),
        "sampler.default_missing_interval": lambda r: _set_sampler_pair(r, "default", None),
        "timing_evidence.reason": lambda r: r.update(timing_evidence=False, timing_evidence_reason="power_trace_run_samples"),
    }


def _delete_field(record: dict, path: str) -> None:
    components = path.split(".")
    target = record
    for component in components[:-1]:
        target = target[component]
    del target[components[-1]]


def _set_field(record: dict, path: str, value) -> None:
    components = path.split(".")
    target = record
    for component in components[:-1]:
        target = target[component]
    target[components[-1]] = value


def _matrix_record(kind: str, tmp_path: Path) -> dict:
    if kind == "sampled_timing":
        record, _trace, _timestamps = _valid_record(tmp_path)
        return record
    if kind == "sampler_off":
        return build_measurement_record(
            config=_config(),
            aiclk_mhz=1_350,
            timestamps=[1_000, 2_000, 3_000],
            producer_full_count=0,
            consumer_empty_count=0,
            kernel_error_flag=0,
            harness_commit="a" * 40,
            environment=_environment(mode="off"),
            power_trace=None,
        )
    if kind == "error":
        classification = 2
        return build_measurement_record(
            config=_config(),
            aiclk_mhz=1_350,
            timestamps=[],
            producer_full_count=0,
            consumer_empty_count=0,
            kernel_error_flag=1,
            harness_commit="a" * 40,
            environment=_environment(mode="off"),
            power_trace=None,
            attempted_frame_count=1,
            produced_frame_count=0,
            dropped_frame_count=0,
            aborted_attempts=1,
            failure_check={
                "code": classification,
                "name": "producer_pacing_wait",
                "source": "producer",
                "elapsed_ticks": 1,
                "limit_ticks": 2,
                "unit": "device_clock_ticks",
                "valid": True,
            },
        )
    if kind == "rejected":
        return build_rejection_record(
            config=_config(), reason="preflight rejected", environment=_environment(mode="off")
        )
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", tuple(RESIDENT_RECORD_FIELD_MATRIX))
def test_required_field_matrix_deletion_mutations_are_rejected(kind: str, tmp_path: Path):
    record = _matrix_record(kind, tmp_path)
    for path in RESIDENT_RECORD_FIELD_MATRIX[kind]:
        candidate = copy.deepcopy(record)
        _delete_field(candidate, path)
        report = validate_resident_record(candidate)
        assert not report["valid"], (kind, path, report["failures"])
        assert any(path in mismatch["fields"] for mismatch in report["mismatches"]), (
            kind,
            path,
            report["mismatches"],
        )


@pytest.mark.parametrize("kind", tuple(RESIDENT_RECORD_FIELD_MATRIX))
def test_required_field_matrix_type_and_range_mutations_are_rejected(
    kind: str, tmp_path: Path
):
    record = _matrix_record(kind, tmp_path)
    for field in RESIDENT_RECORD_FIELD_MATRIX[kind]:
        for mutation, value in (
            ("type", field.invalid_type),
            ("range", field.invalid_range),
        ):
            candidate = copy.deepcopy(record)
            _set_field(candidate, field.path, value)
            report = validate_resident_record(candidate)
            assert not report["valid"], (kind, field.path, mutation, report["failures"])
            assert any(
                field.path in mismatch["fields"]
                for mismatch in report["mismatches"]
            ), (kind, field.path, mutation, report["mismatches"])
            if (
                kind == "sampled_timing"
                and field.path != "timing_evidence"
                and record["timing_evidence"] is True
            ):
                builder_candidate = copy.deepcopy(candidate)
                validate_resident_record(
                    builder_candidate,
                    builder=True,
                    raise_on_error=False,
                )
                assert builder_candidate["timing_evidence"] is False, (
                    field.path,
                    mutation,
                )


def test_required_list_item_fields_are_checked_from_the_same_matrix(tmp_path: Path):
    timestamps = [0, 1_000_000, 2_700_000, 4_050_000, 5_050_000, 6_750_000]
    record, trace, _timestamps = _valid_record(tmp_path, timestamps=timestamps)
    for field in REQUIRED_PAIR_FIELDS:
        candidate = copy.deepcopy(record)
        del candidate["outlier_analysis"]["pairs"][0][field]
        assert not validate_resident_record(
            candidate, timestamps=timestamps, power_trace_path=trace
        )["valid"]
    for field in REQUIRED_SAMPLER_GAP_FIELDS:
        candidate = copy.deepcopy(record)
        del candidate["outlier_analysis"]["sampler_interval_comparison"]["adjacent_gaps"][0][field]
        assert not validate_resident_record(
            candidate, timestamps=timestamps, power_trace_path=trace
        )["valid"]


def test_unknown_modern_fields_are_warnings_and_do_not_reject_the_record(
    tmp_path: Path,
):
    record, trace, timestamps = _valid_record(tmp_path)
    candidate = copy.deepcopy(record)
    candidate["unexpected_field"] = True
    report = validate_resident_record(
        candidate, timestamps=timestamps, power_trace_path=trace
    )
    assert report["valid"]
    assert "unexpected_field" in report["warning_fields"]
    assert report["checks"]["record.unknown_fields"]["ok"]

    candidate = copy.deepcopy(record)
    candidate["environment"]["unexpected_field"] = True
    report = validate_resident_record(
        candidate, timestamps=timestamps, power_trace_path=trace
    )
    assert report["valid"]
    assert "environment.unexpected_field" in report["warning_fields"]


@pytest.mark.parametrize(
    "path",
    [
        Path("docs/measurements/2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005.json"),
        Path("docs/measurements/2026-10-06-p150a-issue104-sampler-off.json"),
        Path("docs/measurements/2026-10-06-p150a-issue104-sampler-default.json"),
        Path("docs/measurements/2026-10-06-p150a-issue104-sampler-5s.json"),
    ],
    ids=["issue12-final", "issue104-sampler-off", "issue104-sampler-default", "issue104-sampler-5s"],
)
def test_historical_records_are_out_of_scope_without_rewriting_the_record(path: Path):
    record = json.loads(path.read_text())

    report = validate_resident_record(record)

    assert report["out_of_scope"] is True
    assert report["classification"] == "out_of_scope"
    assert report["valid"] is None
    assert report["timing_evidence"] is None
    assert report["mismatches"] == []


def test_missing_or_different_runner_marker_is_out_of_scope(tmp_path: Path):
    record, _trace, _timestamps = _valid_record(tmp_path)

    missing = copy.deepcopy(record)
    del missing["resident_record_schema"]
    different = copy.deepcopy(record)
    different["resident_record_schema"] = RESIDENT_RECORD_SCHEMA_MARKER + 1

    for candidate in (missing, different):
        report = validate_resident_record(candidate)
        assert report["out_of_scope"] is True
        assert report["classification"] == "out_of_scope"
        assert report["valid"] is None
        assert report["timing_evidence"] is None


def test_failure_code_zero_is_canonical_and_failure_mutations_force_non_timing(
    tmp_path: Path,
):
    record, trace, timestamps = _valid_record(tmp_path)
    assert validate_resident_record(
        record, timestamps=timestamps, power_trace_path=trace
    )["valid"]
    mutations = {
        "failure_check.code.type": lambda r: r["failure_check"].update(code="0"),
        "failure_check.code.range": lambda r: r["failure_check"].update(code=99),
        "failure_check.name": lambda r: r["failure_check"].update(name="wrong"),
        "failure_check.source": lambda r: r["failure_check"].update(source="unknown"),
        "cycle_budget.error_flag.type": lambda r: r["cycle_budget"].update(error_flag=True),
        "cycle_budget.error_flag.range": lambda r: r["cycle_budget"].update(error_flag=2),
    }
    for name, mutate in mutations.items():
        candidate = copy.deepcopy(record)
        mutate(candidate)
        report = validate_resident_record(
            candidate, timestamps=timestamps, power_trace_path=trace
        )
        assert not report["valid"], (name, report["failures"])
        builder_candidate = copy.deepcopy(candidate)
        validate_resident_record(
            builder_candidate,
            timestamps=timestamps,
            power_trace_path=trace,
            builder=True,
            raise_on_error=False,
        )
        assert builder_candidate["timing_evidence"] is False, name
        assert builder_candidate["power_clock_provenance"]["timing_evidence"] is False


def test_catalog_is_nonempty_and_executable():
    assert RESIDENT_INVARIANT_CATALOG
    assert all(callable(entry.check) for entry in RESIDENT_INVARIANT_CATALOG)
    assert set(TIMING_EVIDENCE_COMPATIBILITY_BRANCHES) == {
        "basename_only_trace_metadata",
        "sampler_off_trace_absent",
        "missing_timing_evidence_field",
    }
    assert {
        "legacy_unverified",
        "snapshot_only",
        "sampler_off_diagnostic",
        "no_valid_in_run_samples",
    } <= set(TIMING_EVIDENCE_UNVERIFIED_AICLK_SOURCES)


@pytest.mark.parametrize(
    ("state", "expected_reason"),
    [
        ("legacy", "legacy_power_trace_unverified"),
        ("unverified", "power_trace_aiclk_unverified"),
        ("unavailable", "power_trace_bytes_unavailable"),
    ],
    ids=["legacy-basename", "unverified-provenance", "unavailable-bytes"],
)
def test_builder_forces_unverified_timing_requests_false(
    tmp_path: Path, state, expected_reason
):
    environment = _environment()
    trace_name = f"{state}.csv"
    trace_metadata = None
    if state == "legacy":
        environment.pop("telemetry_sampler")
    elif state == "unverified":
        trace_metadata = {"file": trace_name, "aiclk_source": "unverified"}

    record = build_measurement_record(
        config=_config(),
        aiclk_mhz=1_350,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=environment,
        power_trace=trace_name,
        power_trace_metadata=trace_metadata,
        timing_evidence=True,
    )

    assert record["timing_evidence"] is False
    assert record["timing_evidence_reason"] == expected_reason
    assert record["power_clock_provenance"]["timing_evidence"] is False
    assert record["power_clock_provenance"]["timing_evidence_reason"] == expected_reason


def test_marker_one_historical_schema_mutation_cannot_skip_modern_checks(tmp_path: Path):
    record, _trace, timestamps = _valid_record(tmp_path)
    record = copy.deepcopy(record)
    assert record["resident_record_schema"] == RESIDENT_RECORD_SCHEMA_MARKER
    record["schema"] = "adr-0005-issue-104-paired-outliers-v1"
    record["timing_evidence"] = True
    record["timing_evidence_reason"] = "power_trace_run_samples"
    record["power_clock_provenance"]["timing_evidence"] = True
    record["power_clock_provenance"]["timing_evidence_reason"] = "power_trace_run_samples"
    del record["parameters"]["aborted_attempts"]
    for field in (
        "pair_sum_target_ticks",
        "pair_sums",
        "event_frame_positions",
        "event_elapsed_seconds",
        "event_elapsed_ticks",
        "event_elapsed_ticks_mod_period",
        "frame_gaps",
        "gap_counts",
        "gap_histogram",
        "gcd_frame_gap",
        "interval_count",
        "pair_order_counts",
    ):
        record["outlier_analysis"].pop(field, None)

    report = validate_resident_record(record, timestamps=timestamps)

    assert report["valid"] is False
    assert any(
        "parameters.aborted_attempts" in mismatch["fields"]
        for mismatch in report["mismatches"]
    )
    assert any(
        "outlier_analysis.pair_sums" in mismatch["fields"]
        for mismatch in report["mismatches"]
    )
    assert any(
        mismatch["invariant"] == "clock.timing_evidence"
        for mismatch in report["mismatches"]
    )
    assert any(
        "bytes are unavailable" in failure or "not permitted" in failure
        for failure in report["failures"]
    )


def test_table_driven_catalog_mutations_report_the_changed_relationship(tmp_path):
    record, trace, timestamps = _valid_record(tmp_path)
    mutations = _mutations(record, trace, timestamps)
    assert {entry.name for entry in RESIDENT_INVARIANT_CATALOG} >= {
        "parameters.counts",
        "raw_timestamps",
        "histogram",
        "outlier_analysis",
        "ring.counts",
        "sampler.trace_contract",
        "power_trace.facts",
        "clock.timing_evidence",
    }
    for field, mutate in mutations.items():
        candidate = copy.deepcopy(record)
        mutate(candidate)
        report = validate_resident_record(
            candidate, timestamps=timestamps, power_trace_path=trace
        )
        assert report["valid"] is False, field
        assert report["mismatches"], field
        if field.startswith("sampler."):
            assert any(
                mismatch["invariant"] == "sampler.trace_contract"
                for mismatch in report["mismatches"]
            ), field


def test_pair_catalog_mutations_are_checked_by_the_shared_validator(tmp_path):
    trace = tmp_path / "power.csv"
    _trace(trace)
    timestamps = [0, 1_000_000, 2_700_000]
    record = build_measurement_record(
        config=_config(),
        aiclk_mhz=1_350,
        timestamps=timestamps,
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=_environment(),
        power_trace=trace.name,
        power_trace_path=trace,
        run_start=RUN_START,
        run_end=RUN_END,
    )
    for field, mutate in {
        "pair_count": lambda out: out.update(pair_count=0),
        "frame_start_indices": lambda out: out["frame_start_indices"].__setitem__(0, 2),
        "pair_sums": lambda out: out["pair_sums"].__setitem__(0, out["pair_sums"][0] + 1),
        "pair_object_sum": lambda out: out["pairs"][0].update(pair_sum_ticks=1),
        "pair_endpoints": lambda out: out["pairs"][0].update(interval_end_frame_indices=[2, 3]),
        "event_frame_positions": lambda out: out.update(event_frame_positions=[1]),
        "event_elapsed_ticks": lambda out: out.update(event_elapsed_ticks=[1, 2]),
        "pair_order": lambda out: out["pairs"][0].update(order="long-first"),
    }.items():
        candidate = copy.deepcopy(record)
        mutate(candidate["outlier_analysis"])
        report = validate_resident_record(
            candidate, timestamps=timestamps, power_trace_path=trace
        )
        assert report["valid"] is False, field


def test_builder_and_analyzer_use_the_same_shared_validator(tmp_path):
    record, trace, timestamps = _valid_record(tmp_path)
    builder_report = validate_resident_record(
        record, timestamps=timestamps, power_trace_path=trace
    )
    analyzer_report = validate_record_invariants(
        record, timestamps=timestamps, power_trace_path=trace
    )
    assert builder_report["valid"] is analyzer_report["valid"] is True
    assert validate_record_invariants is validate_resident_record


@pytest.mark.parametrize(
    ("timestamps", "expected_pairs"),
    [
        ([0, 1_350_000, 2_700_000, 4_050_000], 0),
        ([2**32 - 1_000, 2**32 + 100, 2**32 + 2_699_000], 1),
        ([
            2**32 - 1_000,
            2**32 + 100,
            2**32 + 2_699_000,
            2 * 2**32 - 1_200,
            2 * 2**32 - 100,
            2 * 2**32 + 2_698_800,
        ], 2),
    ],
)
def test_shared_pair_helper_preserves_zero_one_two_pair_cases(timestamps, expected_pairs):
    result = build_outlier_analysis(timestamps, sampler_mode="off")
    assert result["pair_count"] == expected_pairs
    assert len(result["pairs"]) == expected_pairs
    assert len(result["event_frame_positions"]) == expected_pairs * 2
    assert len(result["pair_sums"]) == expected_pairs


def test_normal_runner_result_contains_outlier_analysis_without_hardware(tmp_path, monkeypatch):
    environment = _environment(mode="off")
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(environment))
    output = tmp_path / "record.json"

    def fake_run_device(_ttnn, _device, config, *, watcher):
        del watcher
        count = config.frame_count
        return {
            "timestamps": [1_000 + config.frame_interval_ticks * index for index in range(count)],
            "producer_full_count": 0,
            "consumer_empty_count": 0,
            "kernel_error_flag": 0,
            "frames_attempted": count,
            "frames_produced": count,
            "frames_dropped": 0,
            "frames_aborted": 0,
            "frames_consumed": count,
            "startup_ticks": 0,
            "startup_ticks_valid": False,
            "work_min_ticks": None,
            "work_max_ticks": None,
            "work_ticks_valid": False,
            "failure_check": {"code": 0, "name": "none", "source": "none"},
        }

    monkeypatch.setattr(run_resident, "_run_device", fake_run_device)
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(open_device=lambda **_: object(), close_device=lambda _device: None),
    )
    assert run_resident.main(
        [
            "--out",
            str(output),
            "--env-json",
            str(environment_path),
            "--frame-count",
            "3",
        ]
    ) == 0
    record = json.loads(output.read_text())
    assert record["outlier_analysis"]["pair_count"] == 0
    assert record["outlier_analysis"]["sampler_interval_comparison"]["mode"] == "off"
    assert record["telemetry_sampler"] == record["environment"]["telemetry_sampler"]
    assert record["outlier_analysis"]["sampler_interval_comparison"]["interval_seconds"] is None


def test_runner_completes_legacy_default_sampler_once_and_reuses_it(tmp_path, monkeypatch):
    environment = _environment()
    environment.pop("telemetry_sampler")
    environment["aiclk_mhz_observed"] = [1_350]
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(environment))
    output = tmp_path / "record.json"

    def fake_run_device(_ttnn, _device, config, *, watcher):
        del watcher
        count = config.frame_count
        return {
            "timestamps": [1_000 + config.frame_interval_ticks * index for index in range(count)],
            "producer_full_count": 0,
            "consumer_empty_count": 0,
            "kernel_error_flag": 0,
            "frames_attempted": count,
            "frames_produced": count,
            "frames_dropped": 0,
            "frames_aborted": 0,
            "frames_consumed": count,
            "startup_ticks": 0,
            "startup_ticks_valid": False,
            "work_min_ticks": None,
            "work_max_ticks": None,
            "work_ticks_valid": False,
            "failure_check": {"code": 0, "name": "none", "source": "none"},
        }

    monkeypatch.setattr(run_resident, "_run_device", fake_run_device)
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(open_device=lambda **_: object(), close_device=lambda _device: None),
    )
    assert run_resident.main(
        [
            "--out",
            str(output),
            "--env-json",
            str(environment_path),
            "--power-trace",
            "legacy-default.csv",
            "--frame-count",
            "3",
        ]
    ) == 0

    record = json.loads(output.read_text())
    expected_sampler = {
        "mode": "default",
        "interval_seconds": 2.0,
        "power_trace": "required",
        "timing_evidence": "available",
    }
    assert record["telemetry_sampler"] == expected_sampler
    assert record["environment"]["telemetry_sampler"] == expected_sampler
    assert record["outlier_analysis"]["sampler_interval_comparison"]["mode"] == "default"
    assert record["outlier_analysis"]["sampler_interval_comparison"]["interval_seconds"] == 2.0
