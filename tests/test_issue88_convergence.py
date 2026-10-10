"""Board-free coverage for Issue #88 convergence and publication gates."""

import copy
import datetime
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import newton_schulz_issue88 as runner

RUN_START = datetime.datetime(2026, 1, 1, 0, 0, 1, tzinfo=datetime.UTC)
RUN_END = datetime.datetime(2026, 1, 1, 0, 0, 1, 500000, tzinfo=datetime.UTC)
SOURCE_RECORD = Path(__file__).parents[1] / (
    "docs/measurements/2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json"
)
DERIVED_SUMMARY = Path(__file__).parents[1] / (
    "docs/measurements/2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-summary.json"
)


def _environment(run_id: str = "run-1") -> dict:
    return runner._normalize_environment(
        {
            "image": "ghcr.io/example/image@sha256:" + "a" * 64,
            "run_id": run_id,
            "image_pinned": True,
            "kernel": "Linux 6.8.0-test",
            "kmd_version": "2.11.0",
            "tt_env_active_release": "0.75.0",
            "harness_commit": "b" * 40,
            "harness_dirty": False,
            "board_info": {
                "board_type": "p150a",
                "board_id": "board-serial",
                "bus_id": "0000:09:00.0",
            },
            "firmwares": {"fw_bundle_version": "19.6.0.0"},
        },
        run_id,
    )


def _power_trace(tmp_path: Path, run_id: str = "run-1") -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"power-{run_id}.csv"
    path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,75,1350,60\n"
        "2026-01-01T00:00:02+00:00,76,1350,61\n"
    )
    snapshot = runner._atomic_power_trace_snapshot(
        path,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
    )
    trace = runner._read_power_trace(
        snapshot,
        run_id=run_id,
        source_path=path,
        run_start=RUN_START.isoformat(),
        run_end=RUN_END.isoformat(),
    )
    trace["poll_complete"] = True
    return trace


def _passing_run(run_id: str = "run-1") -> dict:
    rows = []
    artifacts = []
    for config in runner.ISSUE88_COMPARISON_ROWS:
        if config.get("expected_preflight_rejection"):
            rows.append(runner._preflight_rejected_row(config))
            continue
        artifact = {
            "row": config["name"],
            "run_id": run_id,
            "file": f"issue88-fp32-r-inverse-{run_id}-{config['name']}.npy",
            "format": "NumPy .npy",
            "matrix_count": 1,
            "element_count": 4,
            "shape": [1, 2, 2],
            "dtype": "<c8",
            "byte_count": 32,
            "sha256": "a" * 64,
            "hash_definition": runner.INVERSE_ARRAY_HASH_DEFINITION,
            "npy_file_byte_count": 160,
            "npy_file_sha256": "b" * 64,
            "npy_file_hash_definition": runner.INVERSE_NPY_HASH_DEFINITION,
        }
        artifacts.append(artifact)
        rows.append(
            {
                "row": config["name"],
                "status": "ok",
                "device_inverse_artifact": artifact,
            }
        )
    return {
        "status": "pass",
        "run_id": run_id,
        "comparison_rows": rows,
        "device_inverse_artifacts": artifacts,
        "rows_completed": len(rows),
        "rows_requested": len(rows),
        "stopped_on_failure": False,
    }


def _passing_parts(tmp_path: Path) -> tuple[dict, dict, dict]:
    run = _passing_run()
    telemetry = {
        "status": "complete",
        "normalized_environment": _environment(),
        "power_trace": _power_trace(tmp_path),
        "failures": [],
    }
    cleanup = {
        "device_opened": True,
        "close_attempted": True,
        "close_succeeded": True,
    }
    return run, telemetry, cleanup


def _actual_producer_run(
    tmp_path: Path,
    run_id: str,
    *,
    mode: str = "success",
    first_launch: bool = False,
    time_fn=None,
):
    selected = (runner.ISSUE88_COMPARISON_ROWS[0],) if first_launch or mode != "success" else runner.ISSUE88_COMPARISON_ROWS
    matrices_by_size = {
        size: np.broadcast_to(
            np.eye(size, dtype=np.complex64), (2, size, size)
        ).copy()
        for size in (16, 32)
    }

    class FakeKernel:
        @classmethod
        def prepare(cls, ttnn, device, input_matrices, **kwargs):
            if mode == "prepare_error":
                raise RuntimeError(r"C:\Users\alice\kernel error")
            kernel = cls()
            kernel.launch_count = 0
            kernel.actual = runner.reference_context(
                input_matrices, kwargs["variant"]
            )["fixed_reference"]
            if mode == "undefined":
                kernel.actual = np.zeros_like(kernel.actual)
            elif mode == "nonfinite":
                kernel.actual = np.full_like(kernel.actual, np.nan)
            return kernel

        def launch(self):
            self.launch_count += 1
            if mode == "row_error" and self.launch_count == 4:
                raise RuntimeError(r"C:\Users\alice\timed launch error")

        def result(self):
            return self.actual

        def close(self):
            if mode == "cleanup_error":
                raise RuntimeError(r"C:\Users\alice\close error")

    if time_fn is None:
        clock = [0.0]

        def time_fn():
            clock[0] += 0.00001
            return clock[0]

    if mode == "timeout":
        timeout_clock = [0.0]

        def time_fn():
            timeout_clock[0] += 0.0001
            return timeout_clock[0]

        timeout_s = 0.0005
    else:
        timeout_s = runner.ROW_TIMEOUT_S
    return runner.run_comparison(
        SimpleNamespace(synchronize_device=lambda device: None),
        object(),
        artifact_dir=tmp_path / "inverse",
        run_id=run_id,
        rows=selected,
        first_launch=first_launch or mode in {"undefined", "nonfinite"},
        launches=1 if first_launch or mode in {"undefined", "nonfinite"} else runner.LAUNCHES_PER_ROW,
        timeout_s=timeout_s,
        kernel_class=FakeKernel,
        time_fn=time_fn,
        matrices_factory=lambda batch, size, **kwargs: matrices_by_size[size],
    )


def _publication_telemetry(tmp_path: Path, run_id: str) -> dict:
    normalized = _environment(run_id)
    return {
        "status": "complete",
        "environment": normalized,
        "normalized_environment": normalized,
        "power_trace": _power_trace(tmp_path / "trace", run_id),
        "failures": [],
    }


def _publish_producer_output(
    tmp_path: Path, run: dict, run_id: str, telemetry: dict | None = None
) -> tuple[dict, dict]:
    telemetry = telemetry or _publication_telemetry(tmp_path, run_id)
    cleanup = {
        "device_opened": True,
        "close_attempted": True,
        "close_succeeded": True,
    }
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)
    raw_path = tmp_path / f"raw-{run_id}.json"
    record = runner._record_payload(
        run,
        telemetry=telemetry,
        run_id=run_id,
        raw_path=raw_path,
        cleanup=cleanup,
        status_components=statuses,
    )
    raw = runner._raw_payload(
        run,
        telemetry=telemetry,
        run_id=run_id,
        raw_path=raw_path,
        cleanup=cleanup,
        status_components=statuses,
    )
    runner._atomic_json_write(tmp_path / f"record-{run_id}.json", record)
    runner._atomic_json_write(raw_path, raw)
    assert json.loads((tmp_path / f"record-{run_id}.json").read_text()) == json.loads(
        runner._serialize_record(record)
    )
    assert json.loads(raw_path.read_text()) == json.loads(runner._serialize_record(raw))
    return record, raw


def test_issue88_actual_runner_success_rejection_and_first_launch_paths_publish(tmp_path):
    full = _actual_producer_run(tmp_path, "run-actual-full")
    assert full["status"] == "pass"
    assert [row["status"] for row in full["comparison_rows"]] == [
        runner.ISSUE88_ROW_STATUS_OK,
        runner.ISSUE88_ROW_STATUS_OK,
        runner.ISSUE88_ROW_STATUS_OK,
        runner.ISSUE88_ROW_STATUS_OK,
        runner.ISSUE88_ROW_STATUS_OK,
        runner.ISSUE88_ROW_STATUS_PREFLIGHT_REJECTED,
    ]
    full_record, full_raw = _publish_producer_output(
        tmp_path, full, "run-actual-full"
    )
    assert full_record["status"] == "pass"
    assert full_raw["run"]["rows_requested"] == 6
    assert len(full_record["measurement"]["device_inverse_artifacts"]) == 5
    rows = full_record["measurement"]["comparison_rows"]
    assert tuple(row["row"] for row in rows) == tuple(
        config["name"] for config in runner.ISSUE88_COMPARISON_ROWS
    )
    assert tuple(
        row["correctness"]["quality_vs_matching_reference"]["metric_reference"]
        for row in rows[:-1]
    ) == (
        runner.ISSUE88_BF16_MATCHING_METRIC_REFERENCE,
        runner.ISSUE88_FP32_MATCHING_METRIC_REFERENCE,
        runner.ISSUE88_BF16_MATCHING_METRIC_REFERENCE,
        runner.ISSUE88_BF16_MATCHING_METRIC_REFERENCE,
        runner.ISSUE88_FP32_MATCHING_METRIC_REFERENCE,
    )
    assert tuple(
        row["correctness"]["quality_vs_true_inverse"]["metric_reference"]
        for row in rows[:-1]
    ) == (runner.ISSUE88_TRUE_INVERSE_METRIC_REFERENCE,) * 5

    first = _actual_producer_run(
        tmp_path, "run-actual-first", first_launch=True
    )
    assert first["status"] == "pass"
    assert first["comparison_rows"][0]["timing"] == runner.ISSUE88_FIRST_LAUNCH_TIMING
    first_record, first_raw = _publish_producer_output(
        tmp_path, first, "run-actual-first"
    )
    assert first_record["measurement"]["comparison_rows"][0]["timing"] == (
        runner.ISSUE88_FIRST_LAUNCH_TIMING
    )
    assert first_raw["run"]["comparison_rows"][0]["status"] == (
        runner.ISSUE88_ROW_STATUS_CORRECTNESS_ONLY
    )


@pytest.mark.parametrize(
    ("mode", "expected_code", "expected_stage"),
    [
        ("prepare_error", "row_execution_failed", "row"),
        ("row_error", "row_execution_failed", "row"),
        ("timeout", "row_timeout", "row_timeout"),
        ("cleanup_error", "row_cleanup_failed", "row_cleanup"),
        ("nonfinite", "correctness_failed", "correctness"),
    ],
)
def test_issue88_actual_runner_failure_paths_publish_private_diagnostics_safely(
    tmp_path, monkeypatch, mode, expected_code, expected_stage
):
    run_id = f"run-actual-{mode}"
    monkeypatch.setattr(runner, "ISSUE88_RAW_LOG_ROOT", tmp_path / "private-logs")
    monkeypatch.setattr(runner, "_ACTIVE_ISSUE88_RUN_ID", run_id)
    run = _actual_producer_run(
        tmp_path,
        run_id,
        mode=mode,
    )
    assert run["status"] == "failed"
    row = run["comparison_rows"][0]
    assert row["failure_stage"] == expected_stage
    assert row["error"]["code"] == expected_code
    if mode == "prepare_error":
        assert row["reference_available"] is True
    if mode == "timeout":
        assert row["stop_condition"] == runner.ISSUE88_STOP_CONDITION_TIMEOUT
    if mode == "row_error":
        assert row["stop_condition"] == runner.ISSUE88_STOP_CONDITION_ROW_ERROR
    record, raw = _publish_producer_output(tmp_path, run, run_id)
    assert record["status"] == "failed"
    assert raw["run"]["comparison_rows"][0]["error"]["code"] == expected_code
    serialized = (tmp_path / f"record-{run_id}.json").read_text()
    assert "C:\\Users\\alice" not in serialized
    assert "kernel error" not in serialized
    log_text = (tmp_path / "private-logs" / f"issue88-{run_id}.log").read_text()
    if mode == "prepare_error":
        assert r"C:\Users\alice\kernel error" in log_text
    if mode == "cleanup_error":
        assert r"C:\Users\alice\close error" in log_text


def test_issue88_actual_runner_undefined_diagnostic_shapes_publish(tmp_path):
    run_id = "run-actual-undefined"
    run = _actual_producer_run(tmp_path, run_id, mode="undefined")
    assert run["comparison_rows"][0]["status"] == runner.ISSUE88_ROW_STATUS_CORRECTNESS_ONLY
    record, _raw = _publish_producer_output(tmp_path, run, run_id)
    beam = record["measurement"]["comparison_rows"][0]["correctness"][
        "quality_vs_true_inverse"
    ]["beam_pattern"]
    assert beam["response_metric_undefined_values"] > 0
    assert beam["phase_aligned_complex_response_relative_frobenius_error"] is None


def test_issue88_missing_telemetry_and_power_trace_failure_shape_publishes(tmp_path):
    run_id = "run-missing-telemetry"
    run = _actual_producer_run(tmp_path, run_id, first_launch=True)
    telemetry = {"status": "failed", "failures": []}
    record, raw = _publish_producer_output(tmp_path, run, run_id, telemetry)
    assert record["status"] == "failed"
    assert record["status_components"]["power_trace"]["ok"] is False
    assert "power_trace" not in raw["telemetry"]


def test_issue88_actual_telemetry_reader_missing_environment_and_trace_publish(tmp_path):
    run_id = "run-missing-artifacts"
    missing_dir = tmp_path / "missing"
    monotonic = [0.0]
    telemetry = runner._read_telemetry(
        missing_dir,
        run_id=run_id,
        environment_path=missing_dir / f"env-{run_id}.json",
        power_path=missing_dir / f"power-{run_id}.csv",
        run_start=RUN_START,
        run_end=RUN_END,
        monotonic_fn=lambda: monotonic[0],
        sleep_fn=lambda seconds: monotonic.__setitem__(0, monotonic[0] + seconds),
    )
    run = _actual_producer_run(tmp_path, run_id, first_launch=True)
    record, raw = _publish_producer_output(tmp_path, run, run_id, telemetry)

    assert record["status"] == "failed"
    assert telemetry["normalized_environment"] is None
    assert telemetry["power_trace"]["readable"] is False
    assert raw["telemetry"]["power_trace"]["readable"] is False
    assert telemetry["failures"]


def test_issue88_shared_publication_vocabularies_are_exact():
    assert tuple(runner.ISSUE88_PUBLIC_STRING_VALUES) == (
        "status",
        "artifact_status",
        "finite_status",
        "timing",
        "stop_condition",
    )
    assert tuple(sorted(runner.ISSUE88_PUBLIC_STRING_VALUES["status"])) == (
        "admitted",
        "complete",
        "correctness_only",
        "failed",
        "ok",
        "pass",
        "passed before allocation",
        "preflight_rejected",
        "rejected_before_allocation",
    )
    assert tuple(sorted(runner.ISSUE88_PUBLIC_STRING_VALUES["timing"])) == (
        "not run (--first-launch)",
    )
    assert tuple(sorted(runner.ISSUE88_PUBLIC_STRING_VALUES["stop_condition"])) == (
        "row_error",
        "row_timeout_s_exceeded",
    )
    assert tuple(runner.ISSUE88_STATUS_COMPONENT_CODES.items()) == (
        ("rows", ("rows_succeeded", "rows_failed")),
        ("inverse_outputs", ("inverse_outputs_succeeded", "inverse_outputs_failed")),
        ("board_selection", ("board_selection_succeeded", "board_selection_failed")),
        ("image_toolchain", ("image_toolchain_succeeded", "image_toolchain_failed")),
        ("harness", ("harness_succeeded", "harness_failed")),
        ("power_trace", ("power_trace_succeeded", "power_trace_failed")),
        ("device_close", ("device_close_succeeded", "device_close_failed")),
    )
    assert tuple(sorted(runner.ISSUE88_ERROR_CODES)) == (
        "artifact_binding_failed",
        "board_selection_failed",
        "board_selection_succeeded",
        "correctness_failed",
        "device_close_failed",
        "device_close_succeeded",
        "device_open_failed",
        "device_session_failed",
        "environment_artifact_unreadable",
        "environment_run_id_missing",
        "harness_failed",
        "harness_succeeded",
        "host_preparation_failed",
        "image_toolchain_failed",
        "image_toolchain_succeeded",
        "inverse_artifact_persistence_failed",
        "inverse_outputs_failed",
        "inverse_outputs_succeeded",
        "power_sample_invalid",
        "power_timestamp_invalid",
        "power_trace_failed",
        "power_trace_incomplete",
        "power_trace_invalid",
        "power_trace_no_usable_rows",
        "power_trace_succeeded",
        "power_trace_timeout",
        "power_trace_unreadable",
        "preflight_rejected",
        "record_publication_failed",
        "row_cleanup_failed",
        "row_execution_failed",
        "row_timeout",
        "rows_failed",
        "rows_succeeded",
        "run_id_mismatch",
    )


def test_issue88_derived_summary_maps_values_to_source_rows_and_references():
    source_bytes = SOURCE_RECORD.read_bytes()
    source = json.loads(source_bytes)
    summary = json.loads(DERIVED_SUMMARY.read_text())
    measured_rows = [
        row for row in source["measurement"]["comparison_rows"] if row["status"] == "ok"
    ]

    assert summary["source_record"] == {
        "file": SOURCE_RECORD.name,
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    assert summary["source_record"]["sha256"] == (
        "34d3db5f1185548b7415dac2166ee7783d02c82e0ea8b0659e4ec1b500121d88"
    )
    assert len(measured_rows) == 5

    true_table, matching_table = summary["tables"]
    true_reference = summary["metric_reference_definition"]["original_R_true_inverse"]
    matching_reference = summary["metric_reference_definition"][
        "variant_matching_fixed_N_reference"
    ]
    assert true_table["metric_reference"] == true_reference["metric_reference"]
    for table, group, error_field in (
        (
            true_table,
            "quality_vs_true_inverse",
            "relative_frobenius_error_vs_true_inverse",
        ),
        (
            matching_table,
            "quality_vs_matching_reference",
            "relative_frobenius_error_vs_matching_reference",
        ),
    ):
        assert [item["row"] for item in table["rows"]] == [
            row["row"] for row in measured_rows
        ]
        for item, source_row in zip(table["rows"], measured_rows, strict=True):
            correctness = source_row["correctness"]
            quality = correctness[group]
            assert item["variant"] == ("BF16-R" if source_row["variant"] == "bf16" else "FP32-R")
            assert item["L"] == source_row["size"]
            assert item["R_placement"] == source_row["r_memory"].upper()
            expected_metric_reference = (
                true_reference["metric_reference"]
                if group == "quality_vs_true_inverse"
                else matching_reference[
                    "bf16_R" if source_row["variant"] == "bf16" else "fp32_R"
                ]
            )
            assert item["metric_reference"] == expected_metric_reference
            assert item["inverse_relative_frobenius_error"] == correctness[error_field]
            assert item["mv_direction_max_cosine_deficit"] == (
                quality["mv_weight_direction"]["max_cosine_deficit"]
            )
            assert item["beam_pattern_relative_frobenius_error"] == (
                quality["beam_pattern"]["relative_frobenius_error"]
            )
            if group == "quality_vs_matching_reference":
                assert table["metric_reference_by_variant"][source_row["variant"]] == (
                    matching_reference[
                        "bf16_R" if source_row["variant"] == "bf16" else "fp32_R"
                    ]
                )

    owner_comparisons = summary["owner_facing_comparisons_true_inverse_reference"]
    assert all(
        item["metric_reference"] == true_reference["metric_reference"]
        for item in owner_comparisons
    )
    source_by_name = {row["row"]: row for row in measured_rows}
    assert [item["rows"] for item in owner_comparisons] == [
        ["bf16-r-L16", "fp32-r-L16"],
        ["bf16-r-L32-r-dram", "fp32-r-L32-r-dram"],
        ["bf16-r-L32", "bf16-r-L32-r-dram"],
        ["bf16-r-L32", "fp32-r-L32-r-dram"],
    ]
    for comparison in owner_comparisons:
        for row_name in comparison["rows"]:
            source_row = source_by_name[row_name]
            context = comparison["performance_context"][row_name]
            assert context["p50_latency_seconds"] == source_row["seconds_per_launch_p50"]
            assert context["p50_tflops"] == source_row["tflops_p50_derived"]
            assert context["launches_measured"] == source_row["launches_measured"]

    true_inverse_rows = {item["row"]: item for item in true_table["rows"]}
    assert all(
        true_inverse_rows["bf16-r-L32"][metric]
        == true_inverse_rows["bf16-r-L32-r-dram"][metric]
        for metric in (
            "inverse_relative_frobenius_error",
            "mv_direction_max_cosine_deficit",
            "beam_pattern_relative_frobenius_error",
        )
    )
    outcomes = summary["fp32_r_outcome_under_common_true_inverse_reference"]
    for bf16_name, fp32_name in (
        ("bf16-r-L16", "fp32-r-L16"),
        ("bf16-r-L32-r-dram", "fp32-r-L32-r-dram"),
    ):
        bf16 = true_inverse_rows[bf16_name]
        fp32 = true_inverse_rows[fp32_name]
        assert outcomes["inverse_error_improves"] == (
            fp32["inverse_relative_frobenius_error"]
            < bf16["inverse_relative_frobenius_error"]
        )
        assert outcomes["mv_direction_deficit_improves"] == (
            fp32["mv_direction_max_cosine_deficit"]
            < bf16["mv_direction_max_cosine_deficit"]
        )
        assert outcomes["beam_pattern_error_improves"] == (
            fp32["beam_pattern_relative_frobenius_error"]
            < bf16["beam_pattern_relative_frobenius_error"]
        )
    assert outcomes["beam_pattern_error_improves"] is False
    assert outcomes["beam_pattern_outcome"] == (
        "Beam-pattern error is higher (worse) for FP32-R in both same-placement "
        "comparisons: L=16 with R in L1 and L=32 with R in DRAM."
    )
    assert summary["preflight_rejection"]["row"] == "fp32-r-L32"
    assert summary["preflight_rejection"]["status"] == "preflight_rejected"


def test_issue88_catalogue_has_six_rows_and_host_only_rejection(tmp_path):
    runner.validate_comparison_rows()
    assert len(runner.ISSUE88_COMPARISON_ROWS) == 6
    rejected = next(
        row for row in runner.ISSUE88_COMPARISON_ROWS if row["name"] == "fp32-r-L32"
    )
    assert rejected["name"] == "fp32-r-L32"
    assert rejected["expected_preflight_rejection"] is True
    assert rejected["preflight_status"] == "rejected_before_allocation"
    assert rejected["preflight_bytes"] == 1_884_928
    assert rejected["preflight_over_budget_bytes"] == 312_064

    selected = runner._select_rows(runner._build_parser().parse_args([]))
    assert [row["name"] for row in selected] == [
        "bf16-r-L16",
        "fp32-r-L16",
        "bf16-r-L32",
        "bf16-r-L32-r-dram",
        "fp32-r-L32-r-dram",
        "fp32-r-L32",
    ]
    assert selected[-1]["expected_preflight_rejection"] is True

    result = runner.run_comparison(
        SimpleNamespace(),
        object(),
        artifact_dir=tmp_path,
        run_id="run-rejection-only",
        rows=(rejected,),
    )

    assert result["status"] == "pass"
    assert result["rows_completed"] == 1
    assert result["comparison_rows"] == [runner._preflight_rejected_row(rejected)]
    assert result["comparison_rows"][0]["preflight"]["allocation_attempted"] is False
    assert result["comparison_rows"][0]["preflight"]["launches"] == 0
    assert "device_inverse_artifact" not in result["comparison_rows"][0]
    assert result["device_inverse_artifacts"] == []


def test_issue88_status_component_table_is_the_complete_publication_gate(tmp_path):
    run, telemetry, cleanup = _passing_parts(tmp_path)
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)

    assert tuple(statuses) == tuple(name for name, _description in runner.ISSUE88_STATUS_COMPONENTS)
    assert all(set(component) == {"ok", "code"} for component in statuses.values())
    assert all(component["code"] in runner.ISSUE88_ERROR_CODES for component in statuses.values())
    assert runner._status_components_pass(statuses)

    record = runner._record_payload(
        run,
        telemetry=telemetry,
        run_id="run-1",
        raw_path=tmp_path / "raw-run-1.json",
        cleanup=cleanup,
    )
    assert record["status"] == "pass"
    assert record["status_components"] == statuses
    trace = telemetry["power_trace"]
    provenance = record["measurement"]["power_clock_provenance"]
    assert record["power_trace"] == trace["file"]
    assert record["power_trace_sha256"] == trace["sha256"]
    assert provenance["trace"] == trace["file"]
    assert provenance["trace_sha256"] == trace["sha256"]
    assert provenance["sample_count"] == trace["sample_count"]
    assert provenance["coverage"] == trace["coverage"]
    assert provenance["coverage_definition"] == runner.POWER_TRACE_COVERAGE_DEFINITION
    assert provenance["immutable_snapshot"] is True
    assert record["measurement"]["comparison_conclusion_reference_policy"] == (
        "Each conclusion uses one declared metric_reference; metrics with "
        "different references are reported in separate conclusions."
    )
    assert record["measurement"]["device_inverse_artifacts"] == run[
        "device_inverse_artifacts"
    ]
    assert record["measurement"]["comparison_rows"][0][
        "device_inverse_artifact"
    ] == run["comparison_rows"][0]["device_inverse_artifact"]
    raw = runner._raw_payload(
        run,
        telemetry=telemetry,
        run_id="run-1",
        raw_path=tmp_path / "raw-run-1.json",
        cleanup=cleanup,
        status_components=statuses,
    )
    assert raw["run"]["device_inverse_artifacts"] == run["device_inverse_artifacts"]
    assert raw["rows"][0]["device_inverse_artifact"] == run["comparison_rows"][0][
        "device_inverse_artifact"
    ]
    assert "/home/" not in json.dumps(record)
    assert "/home/" not in json.dumps(raw, default=str)


@pytest.mark.parametrize(
    ("component", "mutate"),
    [
        (
            "rows",
            lambda run, telemetry: run["comparison_rows"][0].update(
                status="failed",
                error={"code": "row_execution_failed", "exception_type": "RuntimeError"}
            ),
        ),
        (
            "inverse_outputs",
            lambda run, telemetry: run["comparison_rows"][0].pop(
                "device_inverse_artifact"
            ),
        ),
        ("board_selection", lambda run, telemetry: telemetry["normalized_environment"]["board"].pop("board_type")),
        ("image_toolchain", lambda run, telemetry: telemetry["normalized_environment"].update(image_pinned=False)),
        ("harness", lambda run, telemetry: telemetry["normalized_environment"].update(harness_dirty=True)),
        (
            "power_trace",
            lambda run, telemetry: telemetry["power_trace"].update(
                coverage_complete=False, covers_run_end=False
            ),
        ),
    ],
)
def test_issue88_each_provenance_failure_forces_failed_record(
    tmp_path, component, mutate
):
    run, telemetry, cleanup = _passing_parts(tmp_path)
    mutate(run, telemetry)

    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)
    assert statuses[component]["ok"] is False
    assert statuses[component]["code"] in runner.ISSUE88_ERROR_CODES
    assert not runner._status_components_pass(statuses)

    record = runner._record_payload(
        run,
        telemetry=telemetry,
        run_id="run-1",
        raw_path=tmp_path / "raw-run-1.json",
        cleanup=cleanup,
        status_components=statuses,
    )
    assert record["status"] == "failed"
    assert set(record["failure"]) == {"code", "exception_type"}
    assert record["failure"]["code"] == statuses[component]["code"]


def test_issue88_close_failure_is_a_status_component_failure(tmp_path):
    run, telemetry, cleanup = _passing_parts(tmp_path)
    cleanup["close_succeeded"] = False
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)

    assert statuses["device_close"] == {
        "ok": False,
        "code": "device_close_failed",
    }
    assert not runner._status_components_pass(statuses)


def _write_main_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    environment_path = tmp_path / "env-run-1.json"
    power_path = tmp_path / "power-run-1.csv"
    result_path = tmp_path / "result.json"
    environment_path.write_text(
        json.dumps(
            {
                "image": "ghcr.io/example/image@sha256:" + "a" * 64,
                "run_id": "run-1",
                "image_pinned": True,
                "kernel": "Linux 6.8.0-test",
                "kmd_version": "2.11.0",
                "tt_env_active_release": "0.75.0",
                "harness_commit": "b" * 40,
                "harness_dirty": False,
                "board_info": {
                    "board_type": "p150a",
                    "board_id": "board-serial",
                    "bus_id": "0000:09:00.0",
                },
                "firmwares": {"fw_bundle_version": "19.6.0.0"},
            }
        )
    )
    power_path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,75,1350,60\n"
        "2026-01-01T00:00:02+00:00,76,1350,61\n"
    )
    return environment_path, power_path, result_path


def _run_main_with_close(tmp_path, monkeypatch, close_device):
    monkeypatch.setattr(runner, "ISSUE88_RAW_LOG_ROOT", tmp_path / "raw-log")
    environment_path, power_path, result_path = _write_main_artifacts(tmp_path)
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "run-1")
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(
            open_device=lambda **kwargs: object(),
            close_device=close_device,
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_comparison",
        lambda *args, **kwargs: _passing_run(run_id=kwargs["run_id"]),
    )
    return runner.main(
        [
            "--out",
            str(result_path),
            "--env-json",
            str(environment_path),
            "--power-trace",
            str(power_path),
        ],
        now_fn=iter((RUN_START, RUN_END)).__next__,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    ), result_path


def test_issue88_close_success_is_recorded_after_normal_return(tmp_path, monkeypatch):
    result, result_path = _run_main_with_close(tmp_path, monkeypatch, lambda device: None)

    assert result == 0
    record = json.loads(result_path.read_text())
    assert record["run_protocol"]["cleanup"]["close_succeeded"] is True
    assert record["status_components"]["device_close"]["ok"] is True


def test_issue88_close_exception_keeps_false_cleanup_metadata(tmp_path, monkeypatch):
    def fail_close(device):
        raise RuntimeError("close injected")

    result, result_path = _run_main_with_close(tmp_path, monkeypatch, fail_close)

    assert result == 1
    record = json.loads(result_path.read_text())
    assert record["run_protocol"]["cleanup"]["close_succeeded"] is False
    assert record["status_components"]["device_close"]["ok"] is False
    assert record["failure"] == {
        "code": "device_close_failed",
        "exception_type": "RuntimeError",
    }
    assert "close injected" not in json.dumps(record)
    raw_log = next((tmp_path / "raw-log").glob("issue88-*.log"))
    assert "close injected" in raw_log.read_text()


def _power_csv(timestamps: list[str]) -> str:
    return "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n" + "".join(
        f"{timestamp},75,1350,60\n" for timestamp in timestamps
    )


def _trace_for_rows(tmp_path: Path, rows: list[str], *, run_id: str = "run-1") -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / f"power-{run_id}.csv"
    source.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n" + "".join(rows)
    )
    snapshot = runner._atomic_power_trace_snapshot(
        source,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
    )
    trace = runner._read_power_trace(
        snapshot,
        run_id=run_id,
        source_path=source,
        run_start=RUN_START.isoformat(),
        run_end=RUN_END.isoformat(),
    )
    trace["poll_complete"] = True
    return trace


def test_issue88_power_trace_excludes_each_nonfinite_or_non_numeric_field(tmp_path):
    invalid_values = {
        "power_w": ("", "NaN", "inf", "-inf", "garbage"),
        "aiclk_mhz": ("", "NaN", "inf", "-inf", "garbage", "0"),
        "asic_temp_c": ("", "NaN", "inf", "-inf", "garbage"),
    }
    for field, values in invalid_values.items():
        column = {"power_w": 1, "aiclk_mhz": 2, "asic_temp_c": 3}[field]
        for value in values:
            fields = ["2026-01-01T00:00:01+00:00", "75", "1350", "60"]
            fields[column] = value
            trace = _trace_for_rows(
                tmp_path / f"{field}-{value or 'empty'}".replace("/", "-"),
                [
                    "2026-01-01T00:00:00+00:00,75,1350,60\n",
                    ",".join(fields) + "\n",
                    "2026-01-01T00:00:02+00:00,76,1350,61\n",
                ],
            )
            assert trace["sample_count"] == 2
            assert trace["csv_row_count"] == 3
            assert trace["valid_row_count"] == 2
            assert trace["invalid_row_count"] == 1
            assert trace["coverage_complete"] is True
            assert trace["first_timestamp"] == "2026-01-01T00:00:00+00:00"
            assert trace["last_timestamp"] == "2026-01-01T00:00:02+00:00"
            assert all(
                sample[field] != value for sample in trace["samples"]
            )


def test_issue88_power_trace_coverage_uses_only_usable_rows(tmp_path):
    trace = _trace_for_rows(
        tmp_path,
        [
            "2026-01-01T00:00:00+00:00,75,1350,60\n",
            "2026-01-01T00:00:02+00:00,NaN,1350,61\n",
        ],
    )

    assert trace["sample_count"] == 1
    assert trace["invalid_row_count"] == 1
    assert trace["last_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert trace["covers_run_end"] is False
    assert trace["coverage_complete"] is False
    assert runner._power_trace_failure_reason(trace) == "power_trace_incomplete"

    run, telemetry, cleanup = _passing_parts(tmp_path / "status")
    telemetry["power_trace"] = trace
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)
    assert statuses["power_trace"]["ok"] is False
    assert not runner._status_components_pass(statuses)
    record = runner._record_payload(
        run,
        telemetry=telemetry,
        run_id="run-1",
        raw_path=tmp_path / "raw-run-1.json",
        cleanup=cleanup,
        status_components=statuses,
    )
    assert record["status"] == "failed"


def test_issue88_power_trace_with_no_usable_rows_fails_closed(tmp_path):
    trace = _trace_for_rows(
        tmp_path,
        [
            "2026-01-01T00:00:00+00:00,NaN,1350,60\n",
            "2026-01-01T00:00:02+00:00,76,garbage,61\n",
        ],
    )

    assert trace["sample_count"] == 0
    assert trace["valid_row_count"] == 0
    assert trace["nonempty"] is False
    assert trace["coverage_complete"] is False
    assert trace["error"]["code"] == "power_trace_no_usable_rows"
    assert runner._power_trace_failure_reason(trace) == "power_trace_no_usable_rows"


def test_issue88_valid_power_trace_rows_remain_publishable(tmp_path):
    trace = _trace_for_rows(
        tmp_path,
        [
            "2026-01-01T00:00:00+00:00,75,1350,60\n",
            "2026-01-01T00:00:02+00:00,76,1350,61\n",
        ],
    )
    run, telemetry, cleanup = _passing_parts(tmp_path / "status")
    telemetry["power_trace"] = trace
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)

    assert trace["coverage_complete"] is True
    assert statuses["power_trace"]["ok"] is True
    assert runner._status_components_pass(statuses)


def test_issue88_alternate_valid_numeric_and_timestamp_encodings_publish(tmp_path):
    trace = _trace_for_rows(
        tmp_path,
        [
            "2025-12-31T19:00:00-05:00,+7.5e1,1.35E3,6e1\n",
            "2026-01-01T00:00:02Z,76.0,1350.0,61.0\n",
        ],
    )

    assert trace["coverage_complete"] is True
    assert trace["samples"] == [
        {
            "timestamp_utc": "2025-12-31T19:00:00-05:00",
            "power_w": "+7.5e1",
            "aiclk_mhz": "1.35E3",
            "asic_temp_c": "6e1",
        },
        {
            "timestamp_utc": "2026-01-01T00:00:02Z",
            "power_w": "76.0",
            "aiclk_mhz": "1350.0",
            "asic_temp_c": "61.0",
        },
    ]
    assert runner._serialize_record(
        {"power_trace": runner._public_power_trace(trace)}
    )


def _write_environment(path: Path, *, run_id: str, embedded_run_id: str | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "image": "ghcr.io/example/image@sha256:" + "a" * 64,
                "run_id": run_id if embedded_run_id is None else embedded_run_id,
                "image_pinned": True,
                "kernel": "Linux 6.8.0-test",
                "kmd_version": "2.11.0",
                "tt_env_active_release": "0.75.0",
                "harness_commit": "b" * 40,
                "harness_dirty": False,
                "board_info": {
                    "board_type": "p150a",
                    "board_id": "board-serial",
                    "bus_id": "0000:09:00.0",
                },
                "firmwares": {"fw_bundle_version": "19.6.0.0"},
            }
        )
    )


def _assert_failed_publication(tmp_path: Path, run_id: str, telemetry: dict) -> None:
    run = _passing_run(run_id=run_id)
    cleanup = {
        "device_opened": True,
        "close_attempted": True,
        "close_succeeded": True,
    }
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)
    assert not runner._status_components_pass(statuses)
    record = runner._record_payload(
        run,
        telemetry=telemetry,
        run_id=run_id,
        raw_path=tmp_path / f"raw-{run_id}.json",
        cleanup=cleanup,
        status_components=statuses,
    )
    assert record["status"] == "failed"


def test_issue88_exact_current_pair_is_valid_and_publishable(tmp_path):
    run_id = "run-current"
    _write_environment(tmp_path / f"env-{run_id}.json", run_id=run_id)
    (tmp_path / f"power-{run_id}.csv").write_text(
        _power_csv(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:02+00:00",
            ]
        )
    )

    telemetry = runner._read_telemetry(
        tmp_path,
        run_id=run_id,
        run_start=RUN_START,
        run_end=RUN_END,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    assert telemetry["status"] == "complete"
    assert telemetry["normalized_environment"]["run_id"] == run_id
    assert telemetry["power_trace"]["run_id"] == run_id
    assert telemetry["power_trace"]["source_file"] == f"power-{run_id}.csv"
    run = _passing_run(run_id=run_id)
    cleanup = {
        "device_opened": True,
        "close_attempted": True,
        "close_succeeded": True,
    }
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)
    assert runner._status_components_pass(statuses)
    record = runner._record_payload(
        run,
        telemetry=telemetry,
        run_id=run_id,
        raw_path=tmp_path / f"raw-{run_id}.json",
        cleanup=cleanup,
        status_components=statuses,
    )
    assert record["status"] == "pass"


def test_issue88_old_environment_is_not_paired_with_current_power(tmp_path):
    run_id = "run-current"
    _write_environment(tmp_path / "env-run-old.json", run_id="run-old")
    (tmp_path / f"power-{run_id}.csv").write_text(
        _power_csv(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:02+00:00",
            ]
        )
    )

    telemetry = runner._read_telemetry(
        tmp_path,
        run_id=run_id,
        run_start=RUN_START,
        run_end=RUN_END,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    assert telemetry["normalized_environment"] is None
    assert any(
        failure["stage"] == "telemetry.environment"
        and failure["error"]["code"] == "environment_artifact_unreadable"
        for failure in telemetry["failures"]
    )
    _assert_failed_publication(tmp_path, run_id, telemetry)


def test_issue88_explicit_artifact_path_must_be_current_run_path(tmp_path):
    run_id = "run-current"
    old_environment = tmp_path / "env-run-old.json"
    current_environment = tmp_path / f"env-{run_id}.json"
    _write_environment(old_environment, run_id="run-old")
    _write_environment(current_environment, run_id=run_id)
    current_power = tmp_path / f"power-{run_id}.csv"
    current_power.write_text(
        _power_csv(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:02+00:00",
            ]
        )
    )

    telemetry = runner._read_telemetry(
        tmp_path,
        run_id=run_id,
        environment_path=old_environment,
        power_path=current_power,
        run_start=RUN_START,
        run_end=RUN_END,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    assert telemetry["environment"] is None
    assert any(
        failure["stage"] == "telemetry.environment"
        and failure["error"]["code"] == "artifact_binding_failed"
        for failure in telemetry["failures"]
    )
    _assert_failed_publication(tmp_path, run_id, telemetry)


def test_issue88_embedded_environment_run_id_mismatch_is_not_repaired(tmp_path):
    run_id = "run-current"
    environment_path = tmp_path / f"env-{run_id}.json"
    _write_environment(
        environment_path,
        run_id=run_id,
        embedded_run_id="run-old",
    )
    raw_before = json.loads(environment_path.read_text())
    (tmp_path / f"power-{run_id}.csv").write_text(
        _power_csv(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:02+00:00",
            ]
        )
    )

    telemetry = runner._read_telemetry(
        tmp_path,
        run_id=run_id,
        run_start=RUN_START,
        run_end=RUN_END,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    assert telemetry["environment"]["run_id"] == "run-old"
    assert telemetry["normalized_environment"] is None
    assert json.loads(environment_path.read_text()) == raw_before
    assert any(
        failure["stage"] == "telemetry.environment"
        and failure["error"]["code"] == "run_id_mismatch"
        for failure in telemetry["failures"]
    )
    _assert_failed_publication(tmp_path, run_id, telemetry)


def test_issue88_missing_environment_run_id_is_rejected(tmp_path):
    run_id = "run-current"
    environment_path = tmp_path / f"env-{run_id}.json"
    _write_environment(environment_path, run_id=run_id)
    raw = json.loads(environment_path.read_text())
    raw.pop("run_id")
    environment_path.write_text(json.dumps(raw))
    (tmp_path / f"power-{run_id}.csv").write_text(
        _power_csv(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:02+00:00",
            ]
        )
    )

    telemetry = runner._read_telemetry(
        tmp_path,
        run_id=run_id,
        run_start=RUN_START,
        run_end=RUN_END,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    assert telemetry["environment"]["image"]
    assert telemetry["normalized_environment"] is None
    assert any(
        failure["stage"] == "telemetry.environment"
        and failure["error"]["code"] == "environment_run_id_missing"
        for failure in telemetry["failures"]
    )
    _assert_failed_publication(tmp_path, run_id, telemetry)


def test_issue88_power_source_run_id_mismatch_rejects_snapshot_and_publish(tmp_path):
    run_id = "run-current"
    _write_environment(tmp_path / f"env-{run_id}.json", run_id=run_id)
    old_power = tmp_path / "power-run-old.csv"
    old_power.write_text(
        _power_csv(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:02+00:00",
            ]
        )
    )

    telemetry = runner._read_telemetry(
        tmp_path,
        run_id=run_id,
        power_path=old_power,
        run_start=RUN_START,
        run_end=RUN_END,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    assert telemetry["power_trace"]["immutable_snapshot"] is False
    assert any(
        failure["stage"] == "telemetry.power"
        and failure["error"]["code"] == "artifact_binding_failed"
        for failure in telemetry["failures"]
    )
    _assert_failed_publication(tmp_path, run_id, telemetry)


def test_issue88_glob_only_artifacts_are_rejected_without_publication(tmp_path):
    run_id = "run-current"
    _write_environment(tmp_path / "env-run-old.json", run_id="run-old")
    (tmp_path / "power-run-old.csv").write_text(
        _power_csv(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:02+00:00",
            ]
        )
    )
    missing_environment = tmp_path / f"env-{run_id}.json"
    missing_power = tmp_path / f"power-{run_id}.csv"

    telemetry = runner._read_telemetry(
        tmp_path,
        run_id=run_id,
        environment_path=missing_environment,
        power_path=missing_power,
        run_start=RUN_START,
        run_end=RUN_END,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    assert len(telemetry["failures"]) == 2
    assert {failure["error"]["code"] for failure in telemetry["failures"]} == {
        "environment_artifact_unreadable",
        "power_trace_unreadable",
    }
    _assert_failed_publication(tmp_path, run_id, telemetry)


def test_issue88_snapshot_with_mismatched_run_id_is_rejected(tmp_path):
    old_run_id = "run-old"
    source = tmp_path / f"power-{old_run_id}.csv"
    source.write_text(
        _power_csv(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:02+00:00",
            ]
        )
    )
    snapshot = runner._atomic_power_trace_snapshot(
        source,
        tmp_path,
        run_id=old_run_id,
        snapshot_number=1,
    )

    with pytest.raises(ValueError, match="does not match the current run"):
        runner._read_power_trace(
            snapshot,
            run_id="run-current",
            run_start=RUN_START.isoformat(),
            run_end=RUN_END.isoformat(),
        )


def test_issue88_power_poll_waits_for_sample_covering_run_end(tmp_path):
    path = tmp_path / "power-run-1.csv"
    path.write_text(_power_csv(["2026-01-01T00:00:01+00:00"]))
    monotonic = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        monotonic[0] += seconds
        path.write_text(
            _power_csv(
                [
                    "2026-01-01T00:00:01+00:00",
                    "2026-01-01T00:00:02+00:00",
                ]
            )
        )

    trace = runner._wait_for_power_trace(
        tmp_path,
        run_id="run-1",
        explicit=path,
        run_start=RUN_START,
        run_end=RUN_END,
        timeout_s=2.0,
        interval_s=1.0,
        monotonic_fn=lambda: monotonic[0],
        sleep_fn=sleep,
    )

    assert sleeps == [1.0]
    assert trace["poll_complete"] is True
    assert trace["coverage_complete"] is True
    assert trace["sample_count"] == 2
    assert trace["first_timestamp"] <= trace["run_start"]
    assert trace["run_end"] <= trace["last_timestamp"]


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2026-01-01T00:00:01+00:00"],
        ["not-a-timestamp"],
    ],
)
def test_issue88_power_poll_records_incomplete_or_invalid_timestamp_failure(
    tmp_path, timestamps
):
    path = tmp_path / "power-run-1.csv"
    path.write_text(_power_csv(timestamps))
    monotonic = [0.0]

    def sleep(seconds):
        monotonic[0] += seconds

    trace = runner._wait_for_power_trace(
        tmp_path,
        run_id="run-1",
        explicit=path,
        run_start=RUN_START,
        run_end=RUN_END,
        timeout_s=1.0,
        interval_s=0.5,
        monotonic_fn=lambda: monotonic[0],
        sleep_fn=sleep,
    )

    assert trace["poll_complete"] is False
    assert trace["coverage_complete"] is False
    assert trace["poll_error"]
    if timestamps[0] == "not-a-timestamp":
        assert trace["timestamps_parse"] is False
    else:
        assert trace["covers_run_end"] is False


def test_issue88_power_trace_snapshot_is_run_bound_hashed_and_immutable(tmp_path):
    live_path = tmp_path / "power-run-1.csv"
    live_path.write_text(_power_csv(["2026-01-01T00:00:01+00:00"]))
    monotonic = [0.0]

    def sleep(seconds):
        monotonic[0] += seconds
        live_path.write_text(
            _power_csv(
                [
                    "2026-01-01T00:00:01+00:00",
                    "2026-01-01T00:00:02+00:00",
                ]
            )
        )

    trace = runner._wait_for_power_trace(
        tmp_path,
        run_id="run-1",
        explicit=live_path,
        run_start=RUN_START,
        run_end=RUN_END,
        timeout_s=2.0,
        interval_s=1.0,
        monotonic_fn=lambda: monotonic[0],
        sleep_fn=sleep,
    )

    snapshot_path = tmp_path / trace["file"]
    snapshot_bytes = snapshot_path.read_bytes()
    assert trace["file"].startswith("issue88-fp32-r-power-snapshot-run-1-")
    assert trace["file"] != live_path.name
    assert trace["run_id"] == "run-1"
    assert trace["immutable_snapshot"] is True
    assert trace["sha256"] == hashlib.sha256(snapshot_bytes).hexdigest()
    assert trace["sample_count"] == 2
    assert trace["coverage_definition"] == runner.POWER_TRACE_COVERAGE_DEFINITION
    assert trace["coverage"]["last_at_or_after_run_end"] is True

    live_path.write_text(_power_csv(["2026-01-01T00:00:03+00:00"]))
    assert snapshot_path.read_bytes() == snapshot_bytes
    assert trace["sha256"] == hashlib.sha256(snapshot_path.read_bytes()).hexdigest()


def test_issue88_power_trace_snapshot_preserves_hash_on_partial_csv_failure(tmp_path):
    live_path = tmp_path / "power-run-1.csv"
    live_path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:01+00:00,75\n"
    )

    trace = runner._wait_for_power_trace(
        tmp_path,
        run_id="run-1",
        explicit=live_path,
        run_start=RUN_START,
        run_end=RUN_END,
        timeout_s=0.1,
        interval_s=0.1,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    assert trace["file"].startswith("issue88-fp32-r-power-snapshot-run-1-")
    assert trace["immutable_snapshot"] is True
    assert trace["sha256"] == hashlib.sha256((tmp_path / trace["file"]).read_bytes()).hexdigest()
    assert trace["readable"] is False
    assert trace["coverage_complete"] is False
    assert trace["coverage"]["last_at_or_after_run_end"] is False


def test_issue88_power_poll_configuration_stays_within_row_cap():
    assert runner.POWER_TRACE_POLL_TIMEOUT_S <= runner.ROW_TIMEOUT_S


PATH_PRIVACY_BOUNDARY_CASES = (
    r"C:\Users\alice\secret.json",
    "~alice/private/secret.json",
    r"~alice\private\secret.json",
    "/alice(private)",
    "/(alice)",
    "/home/Alice Smith/log.json",
    "/home/Alice;Smith/log.json",
    "/home/Alice,Smith/log.json",
)


def _privacy_test_environment(tmp_path: Path, run_id: str = "run-1") -> dict:
    environment_path = tmp_path / f"env-{run_id}.json"
    _write_environment(environment_path, run_id=run_id)
    return json.loads(environment_path.read_text())


def _record_with_raw_environment(tmp_path: Path, raw_environment: dict) -> dict:
    run, telemetry, cleanup = _passing_parts(tmp_path / "parts")
    telemetry["environment"] = runner._sanitize_metadata(raw_environment)
    telemetry["normalized_environment"] = runner._normalize_environment(
        raw_environment, "run-1"
    )
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)
    return runner._record_payload(
        run,
        telemetry=telemetry,
        run_id="run-1",
        raw_path=tmp_path / "raw-run-1.json",
        cleanup=cleanup,
        status_components=statuses,
    )


@pytest.mark.parametrize("path_text", PATH_PRIVACY_BOUNDARY_CASES)
@pytest.mark.parametrize("injection", ("value", "nested-key"))
def test_issue88_structural_metadata_projection_omits_path_inputs(
    tmp_path, path_text, injection
):
    raw = _privacy_test_environment(tmp_path)
    if injection == "value":
        raw["kernel"] = path_text
    else:
        raw["board_info"][path_text] = path_text

    sanitized_environment = runner._sanitize_metadata(raw)
    normalized_environment = runner._normalize_environment(raw, "run-1")
    assert path_text not in json.dumps(sanitized_environment)
    assert path_text not in json.dumps(normalized_environment)

    record = _record_with_raw_environment(tmp_path, raw)
    output = tmp_path / "record.json"
    runner._atomic_json_write(output, record)
    serialized = output.read_text()
    assert path_text not in serialized
    assert runner._ABSOLUTE_PATH_RE.search(serialized) is None


@pytest.mark.parametrize("path_text", PATH_PRIVACY_BOUNDARY_CASES)
@pytest.mark.parametrize("injection", ("nested-value", "nested-key"))
def test_issue88_final_allowlist_rejects_unsanitized_nested_paths(
    tmp_path, path_text, injection
):
    record = _record_with_raw_environment(
        tmp_path, _privacy_test_environment(tmp_path)
    )
    if injection == "nested-value":
        record["notes"][0] = path_text
    else:
        record["environment"]["board"][path_text] = "p150a"
    output = tmp_path / f"rejected-{injection}.json"

    with pytest.raises(ValueError, match="record serialization rejected"):
        runner._atomic_json_write(output, record)

    assert not output.exists()


@pytest.mark.parametrize(
    "ordinary_text",
    [
        "ordinary non-path text",
        "relative/path.txt",
        "./relative/path.txt",
        "ratio 1/2 remains readable",
        "device-1-retake.json",
    ],
)
def test_issue88_sanitize_text_preserves_ordinary_and_basename_text(ordinary_text):
    assert runner._sanitize_text(ordinary_text) == ordinary_text


def test_issue88_final_allowlist_rejects_unknown_fields_and_error_members(tmp_path):
    record = _record_with_raw_environment(
        tmp_path, _privacy_test_environment(tmp_path)
    )
    invalid_key = copy.deepcopy(record)
    invalid_key["measurement"]["unknown-path"] = "relative"
    invalid_error = copy.deepcopy(record)
    invalid_error["failure"] = {
        "code": "row_timeout",
        "exception_type": "TimeoutError",
        "message": "unapproved text",
    }
    invalid_public_text = copy.deepcopy(record)
    invalid_public_text["measurement"]["description"] = "unapproved ordinary text"

    for payload in (invalid_key, invalid_error, invalid_public_text):
        with pytest.raises(ValueError, match="record serialization rejected"):
            runner._serialize_record(payload)


def test_issue88_error_entries_use_fixed_vocabulary_and_private_raw_log(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ISSUE88_RAW_LOG_ROOT", tmp_path / "raw-log")
    message = r"C:\Users\alice\secret.json; /home/Alice Smith/log.json"
    error = runner._error_entry(
        "row_execution_failed", RuntimeError(message), run_id="run-private"
    )

    assert error == {
        "code": "row_execution_failed",
        "exception_type": "RuntimeError",
    }
    serialized = runner._serialize_record({"error": error})
    assert message not in serialized
    raw_log = tmp_path / "raw-log" / "issue88-run-private.log"
    assert message in raw_log.read_text()
    assert {
        "power_trace_unreadable",
        "run_id_mismatch",
        "preflight_rejected",
    } <= runner.ISSUE88_ERROR_CODES


def test_issue88_error_arrays_validate_closed_entry_shapes():
    safe = {
        "errors": [{"code": "row_execution_failed", "exception_type": "RuntimeError"}],
        "failures": [
            {
                "stage": "row",
                "error": {"code": "row_execution_failed", "exception_type": "RuntimeError"},
            }
        ],
    }
    serialized = runner._serialize_record(safe)
    assert json.loads(serialized) == safe

    invalid = copy.deepcopy(safe)
    invalid["failures"][0]["error"] = {
        "code": "row_execution_failed",
        "exception_type": "RuntimeError",
        "message": r"C:\Users\alice\private.log",
    }
    with pytest.raises(ValueError, match="record serialization rejected"):
        runner._serialize_record(invalid)


def test_issue88_safe_record_passes_full_publication_validation(tmp_path):
    record = _record_with_raw_environment(
        tmp_path, _privacy_test_environment(tmp_path)
    )
    serialized = runner._serialize_record(record)

    assert json.loads(serialized) == record
    assert runner._ABSOLUTE_PATH_RE.search(serialized) is None


@pytest.mark.parametrize(
    "filename",
    [
        "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json",
        "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-summary.json",
        "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv",
    ],
)
def test_issue88_device1_retake_artifacts_have_no_absolute_paths(filename, tmp_path):
    measurements = Path(__file__).parents[1] / "docs/measurements"
    artifact = measurements / filename
    original_bytes = artifact.read_bytes()

    assert runner._ABSOLUTE_PATH_RE.search(artifact.read_text()) is None
    if filename.endswith(".json"):
        runner._serialize_record(json.loads(original_bytes))
    else:
        record = json.loads(
            (measurements / "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json").read_text()
        )
        provenance = record["measurement"]["power_clock_provenance"]
        snapshot = tmp_path / (
            f"{runner.POWER_TRACE_SNAPSHOT_PREFIX}-{record['run_id']}-1.csv"
        )
        snapshot.write_bytes(original_bytes)
        trace = runner._read_power_trace(
            snapshot,
            run_id=record["run_id"],
            run_start=provenance["run_start"],
            run_end=provenance["run_end"],
        )
        runner._validate_record_node(runner._public_power_trace(trace))
        assert trace["coverage_complete"] is True
    assert artifact.read_bytes() == original_bytes
