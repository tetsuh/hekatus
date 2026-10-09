"""Board-free tests for the Issue #88 comparison runner."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from enodia.tt.bench.newton_schulz_reference import (
    bf16_round_complex,
    initial_value,
    newton_schulz_reference,
    random_hpd_batch,
)
from tools import newton_schulz_issue88 as runner

PLAN = Path(__file__).parents[1] / "docs/measurements/2026-10-07-host-newton-schulz-issue88-fp32-r-plan.json"


def test_issue88_catalogue_pins_every_execution_control_and_placement():
    runner.validate_comparison_rows()

    rows = runner.ISSUE88_COMPARISON_ROWS
    assert [(row["variant"], row["size"], row["r_memory"]) for row in rows] == [
        ("bf16", 16, "l1"),
        ("fp32-r", 16, "l1"),
        ("bf16", 32, "l1"),
        ("bf16", 32, "dram"),
        ("fp32-r", 32, "dram"),
        ("fp32-r", 32, "l1"),
    ]
    assert {
        key: rows[0][key]
        for key in (
            "batch",
            "iterations",
            "initial_value",
            "state_format",
            "destination_format",
            "math_fidelity",
            "fuse_s",
            "matrix_block",
            "double_buffer",
            "dst_full_sync_en",
            "x0_memory",
            "output_memory",
        )
    } == {
        "batch": 8192,
        "iterations": 12,
        "initial_value": "I/||R||inf",
        "state_format": "BF16",
        "destination_format": "FP32",
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "x0_memory": "l1",
        "output_memory": "dram",
    }


def test_issue88_placement_pairs_keep_controls_identical():
    rows = {row["name"]: row for row in runner.ISSUE88_COMPARISON_ROWS}
    fixed_controls = (
        "batch",
        "iterations",
        "initial_value",
        "state_format",
        "destination_format",
        "fp32_dest_acc_en",
        "math_fidelity",
        "fuse_s",
        "input_memory",
        "matrix_block",
        "double_buffer",
        "dst_full_sync_en",
        "output_memory",
        "x0_memory",
    )
    for bf16_name, fp32_name, r_memory in (
        ("bf16-r-L16", "fp32-r-L16", "l1"),
        ("bf16-r-L32-r-dram", "fp32-r-L32-r-dram", "dram"),
    ):
        bf16 = rows[bf16_name]
        fp32 = rows[fp32_name]
        assert bf16["r_memory"] == fp32["r_memory"] == r_memory
        assert (bf16["input_memory"], bf16["x0_memory"], bf16["output_memory"]) == (
            fp32["input_memory"],
            fp32["x0_memory"],
            fp32["output_memory"],
        )
        assert all(bf16[key] == fp32[key] for key in fixed_controls)
        assert bf16["variant"] == "bf16"
        assert fp32["variant"] == "fp32-r"
        assert bf16["r_format"] == "BF16"
        assert fp32["r_format"] == "FP32"


def test_issue88_timing_summary_retains_samples_and_all_derivations():
    summary = runner.timing_summary([1.0, 2.0, 4.0], flops_per_launch=8)

    assert summary["seconds_per_launch_samples"] == [1.0, 2.0, 4.0]
    assert summary["seconds_per_launch_min"] == 1.0
    assert summary["seconds_per_launch_p50"] == 2.0
    assert summary["seconds_per_launch_p99"] == pytest.approx(3.96)
    assert summary["seconds_per_launch_p99_9"] == pytest.approx(3.996)
    assert summary["tflops_min_derived"] == pytest.approx(8e-12)
    assert summary["tflops_p50_derived"] == pytest.approx(4e-12)


def test_issue88_reference_context_uses_original_x0_for_both_r_formats():
    matrices = random_hpd_batch(2, 4, condition_number=100.0, seed=runner.INPUT_SEED)
    x0 = initial_value(matrices)
    bf16_context = runner.reference_context(matrices, "bf16")
    fp32_context = runner.reference_context(matrices, "fp32-r")

    np.testing.assert_array_equal(bf16_context["x0"], x0)
    np.testing.assert_array_equal(fp32_context["x0"], x0)
    np.testing.assert_array_equal(bf16_context["reference_r"], bf16_round_complex(matrices))
    np.testing.assert_array_equal(fp32_context["reference_r"], matrices)
    np.testing.assert_allclose(
        bf16_context["fixed_reference"],
        newton_schulz_reference(bf16_round_complex(matrices), x0=x0),
    )
    np.testing.assert_allclose(
        fp32_context["fixed_reference"],
        newton_schulz_reference(matrices, x0=x0),
    )


def test_issue88_quality_metrics_label_each_reference_and_submetric():
    matrices = random_hpd_batch(2, 4, condition_number=100.0, seed=runner.INPUT_SEED)
    for variant, group, metric_reference in (
        (
            "bf16",
            "quality_vs_matching_reference",
            (
                "fixed-N=12 Newton-Schulz reference using BF16-rounded R and "
                "X0=I/||original FP32 R||_infinity"
            ),
        ),
        (
            "fp32-r",
            "quality_vs_matching_reference",
            (
                "fixed-N=12 Newton-Schulz reference using original FP32 R and "
                "X0=I/||original FP32 R||_infinity"
            ),
        ),
    ):
        context = runner.reference_context(matrices, variant)
        metrics = runner.correctness_metrics(context["fixed_reference"], context)
        quality = metrics[group]
        assert quality["metric_reference"] == metric_reference
        assert quality["mv_weight_direction"]["metric_reference"] == metric_reference
        assert quality["beam_pattern"]["metric_reference"] == metric_reference
        assert metrics["metric_references"]["relative_error"] == metric_reference
        assert metrics["device_test_gate"]["metric_reference"] == metric_reference

    context = runner.reference_context(matrices, "fp32-r")
    metrics = runner.correctness_metrics(context["fixed_reference"], context)
    true_inverse = metrics["quality_vs_true_inverse"]
    assert true_inverse["metric_reference"] == runner.TRUE_INVERSE_METRIC_REFERENCE
    assert (
        true_inverse["mv_weight_direction"]["metric_reference"]
        == runner.TRUE_INVERSE_METRIC_REFERENCE
    )
    assert (
        true_inverse["beam_pattern"]["metric_reference"]
        == runner.TRUE_INVERSE_METRIC_REFERENCE
    )


def test_issue88_same_array_metrics_are_zero_for_identical_inverses():
    inverse = np.broadcast_to(np.eye(4, dtype=np.complex128), (2, 4, 4)).copy()
    metrics = runner.same_array_metrics(inverse, inverse)

    assert metrics["mv_weight_direction"]["max_cosine_deficit"] == pytest.approx(0.0)
    assert metrics["beam_pattern"][
        "phase_sensitive_complex_response_relative_frobenius_error"
    ] == pytest.approx(0.0)
    assert metrics["beam_pattern"][
        "phase_sensitive_complex_response_max_absolute_error"
    ] == pytest.approx(0.0)


def test_issue88_inverse_artifact_metadata_hashes_c_order_payload(tmp_path):
    inverse = np.arange(16, dtype=np.float32).astype(np.complex64).reshape(2, 2, 4)
    inverse = inverse[:, :, :2].copy()
    metadata = runner._atomic_inverse_artifact_write(
        tmp_path,
        run_id="run-hash-test",
        row_name="bf16-r-L16",
        downloaded_inverse=inverse,
    )
    path = tmp_path / metadata["file"]
    expected_payload = np.ascontiguousarray(inverse).view(np.uint8)

    assert metadata["row"] == "bf16-r-L16"
    assert metadata["file"] == "issue88-fp32-r-inverse-run-hash-test-bf16-r-L16.npy"
    assert metadata["matrix_count"] == 2
    assert metadata["element_count"] == 8
    assert metadata["shape"] == [2, 2, 2]
    assert metadata["dtype"] == inverse.dtype.str
    assert metadata["byte_count"] == inverse.nbytes
    assert metadata["sha256"] == hashlib.sha256(expected_payload).hexdigest()
    assert metadata["hash_definition"] == runner.INVERSE_ARRAY_HASH_DEFINITION
    assert metadata["npy_file_byte_count"] == path.stat().st_size
    assert metadata["npy_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["npy_file_hash_definition"] == runner.INVERSE_NPY_HASH_DEFINITION


def test_issue88_inverse_artifact_round_trips_with_numpy_load(tmp_path):
    inverse = np.asarray(
        [[[1.0 + 2.0j, 3.0 - 4.0j], [5.5 + 0.0j, -6.0 + 0.25j]]],
        dtype=np.complex64,
    )
    metadata = runner._atomic_inverse_artifact_write(
        tmp_path,
        run_id="run-roundtrip",
        row_name="fp32-r-L16",
        downloaded_inverse=inverse,
    )
    loaded = np.load(tmp_path / metadata["file"], allow_pickle=False)

    assert loaded.shape == inverse.shape
    assert loaded.dtype == inverse.dtype
    np.testing.assert_array_equal(loaded, inverse)


def test_issue88_runner_has_no_spec_reference_and_matches_plan():
    source = Path(runner.__file__).read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert all(not module.startswith("enodia.spec") for module in imported_modules)

    plan = json.loads(PLAN.read_text())
    assert plan["comparison_plan"]["runner"] == runner.ISSUE88_RUNNER
    assert plan["comparison_plan"]["watcher"] is False
    assert plan["comparison_plan"]["input_generator"].endswith("random_hpd_batch")
    assert plan["comparison_plan"]["condition_number"] == runner.CONDITION_NUMBER
    assert plan["comparison_plan"]["seed"] == runner.INPUT_SEED
    parser = runner._build_parser()
    for row in plan["comparison_plan"]["rows"]:
        args = parser.parse_args(row["runner_args"])
        runner._validate_args(parser, args)


def test_issue88_runner_selects_the_l32_dram_pair_in_one_session(tmp_path):
    parser = runner._build_parser()
    args = parser.parse_args(
        [
            "--row",
            "bf16-r-L32-r-dram",
            "--row",
            "fp32-r-L32-r-dram",
        ]
    )
    runner._validate_args(parser, args)
    selected = runner._select_rows(args)
    assert [row["name"] for row in selected] == [
        "bf16-r-L32-r-dram",
        "fp32-r-L32-r-dram",
    ]

    matrices = np.broadcast_to(
        np.eye(2, dtype=np.complex64),
        (2, 2, 2),
    ).copy()
    factory_calls = []
    prepared = []
    synchronized = []

    def matrices_factory(batch, size, **kwargs):
        factory_calls.append((batch, size, kwargs))
        return matrices

    class FakeKernel:
        @classmethod
        def prepare(cls, ttnn, device, input_matrices, **kwargs):
            kernel = cls()
            kernel.actual = runner.reference_context(
                input_matrices, kwargs["variant"]
            )["fixed_reference"]
            kernel.launches = 0
            prepared.append((input_matrices, kernel, kwargs))
            return kernel

        def launch(self):
            self.launches += 1

        def result(self):
            return self.actual

        def close(self):
            return None

    clock = [0.0]

    def time_fn():
        clock[0] += 0.001
        return clock[0]

    result = runner.run_comparison(
        SimpleNamespace(synchronize_device=lambda device: synchronized.append(device)),
        device=object(),
        artifact_dir=tmp_path,
        run_id="run-issue88-pair",
        rows=selected,
        kernel_class=FakeKernel,
        time_fn=time_fn,
        matrices_factory=matrices_factory,
    )

    assert result["status"] == "pass"
    assert len(factory_calls) == 1
    assert factory_calls[0] == (
        8192,
        32,
        {"condition_number": 100.0, "seed": 6300},
    )
    assert all(item[0] is matrices for item in prepared)
    assert prepared[0][0] is prepared[1][0] is matrices
    assert [item[2]["r_memory"] for item in prepared] == ["dram", "dram"]
    assert [item[1].launches for item in prepared] == [1002, 1002]
    assert len(synchronized) == 2 * (1 + 1 + runner.LAUNCHES_PER_ROW)
    assert [row["launches_measured"] for row in result["comparison_rows"]] == [1000, 1000]
    artifacts = result["device_inverse_artifacts"]
    assert len(artifacts) == 2
    assert [artifact["row"] for artifact in artifacts] == [row["name"] for row in selected]
    for artifact in artifacts:
        assert artifact["file"] == (
            f"issue88-fp32-r-inverse-run-issue88-pair-{artifact['row']}.npy"
        )
        assert (tmp_path / artifact["file"]).is_file()
        assert runner._inverse_artifact_metadata_error(
            artifact, row_name=artifact["row"], run_id="run-issue88-pair"
        ) is None


def test_issue88_full_selection_retains_measured_outputs_not_preflight_output(tmp_path):
    selected = runner._select_rows(runner._build_parser().parse_args([]))
    matrices = np.broadcast_to(np.eye(2, dtype=np.complex64), (2, 2, 2)).copy()

    class FakeKernel:
        @classmethod
        def prepare(cls, ttnn, device, input_matrices, **kwargs):
            kernel = cls()
            kernel.actual = runner.reference_context(
                input_matrices, kwargs["variant"]
            )["fixed_reference"]
            kernel.launches = 0
            return kernel

        def launch(self):
            self.launches += 1

        def result(self):
            return self.actual

        def close(self):
            return None

    clock = [0.0]

    def time_fn():
        clock[0] += 0.001
        return clock[0]

    result = runner.run_comparison(
        SimpleNamespace(synchronize_device=lambda device: None),
        object(),
        artifact_dir=tmp_path,
        run_id="run-six-row",
        rows=selected,
        kernel_class=FakeKernel,
        time_fn=time_fn,
        matrices_factory=lambda *args, **kwargs: matrices,
    )

    assert result["status"] == "pass"
    assert [row["row"] for row in result["comparison_rows"]] == [
        row["name"] for row in selected
    ]
    measured_rows = result["comparison_rows"][:-1]
    rejected_row = result["comparison_rows"][-1]
    assert all(row["status"] == "ok" for row in measured_rows)
    assert all("device_inverse_artifact" in row for row in measured_rows)
    expected_accounting = {
        "bf16-r-L16": (885504, 0, 687360, 358400, 415744, "admitted", "L1"),
        "fp32-r-L16": (1229568, 0, 343296, 456704, 661504, "admitted", "L1"),
        "bf16-r-L32": (1295104, 0, 277760, 358400, 825344, "admitted", "L1"),
        "bf16-r-L32-r-dram": (803584, 0, 769280, 358400, 333824, "admitted", "DRAM"),
        "fp32-r-L32-r-dram": (901888, 0, 670976, 456704, 333824, "admitted", "DRAM"),
        "fp32-r-L32": (1884928, 312064, 0, 456704, 1316864, "rejected_before_allocation", "L1"),
    }
    for row in result["comparison_rows"]:
        preflight = row["preflight"]
        expected = expected_accounting[row["row"]]
        assert (
            preflight["total_bytes"],
            preflight["overage_bytes"],
            preflight["headroom_bytes"],
            preflight["cb_bytes"],
            preflight["tensor_bytes"],
            preflight["status"],
            preflight["placement"]["R"],
        ) == expected
        assert preflight["bytes"] == preflight["total_bytes"]
        assert preflight["breakdown"]["circular_buffers"] == preflight["cb_bytes_by_name"]
        assert preflight["breakdown"]["tensor_items"] == preflight["tensor_items"]
    assert rejected_row["status"] == "preflight_rejected"
    assert "device_inverse_artifact" not in rejected_row
    assert rejected_row["launches_requested"] == 0
    assert rejected_row["launches_measured"] == 0
    assert [artifact["row"] for artifact in result["device_inverse_artifacts"]] == [
        row["row"] for row in measured_rows
    ]
    assert len(list(tmp_path.glob("*.npy"))) == 5


def test_issue88_l1_accounting_artifact_matches_runner_and_immutable_source():
    artifact_path = (
        Path(__file__).parents[1]
        / "docs/measurements/2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-l1-accounting.json"
    )
    source_path = artifact_path.with_name(
        "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json"
    )
    artifact = json.loads(artifact_path.read_text())
    assert artifact["harness_commit"] == "79d215141e4d64752c8cdc048dcaccc73be1e013"
    assert artifact["derivation"]["accounting_module"] == (
        "enodia.tt.bench.newton_schulz_kernel"
    )
    assert artifact["derivation"]["board_run_performed"] is False
    assert artifact["source_record"]["sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()

    source_rows = {row["row"]: row for row in artifact["rows"]}
    for config in runner.ISSUE88_COMPARISON_ROWS:
        serialized = runner._preflight_accounting(config)
        source = source_rows[config["name"]]
        breakdown = source["byte_breakdown"]
        assert breakdown["circular_buffers"] == serialized["cb_bytes_by_name"]
        assert breakdown["tensor_items"] == serialized["tensor_items"]
        for key in ("cb_bytes", "tensor_bytes", "total_bytes", "budget_bytes"):
            assert breakdown[key] == serialized[key]
        for key in ("overage_bytes", "headroom_bytes"):
            assert breakdown[key] == serialized[key]
        assert source["placement"] == serialized["placement"]
        assert source["preflight"]["status"] == serialized["status"]
        assert source["status"] == ("accepted" if serialized["accepted"] else "rejected")

    assert [item["name"] for item in artifact["matched_configuration_differences"]] == [
        "BF16-R/L16 vs FP32-R/L16 (same R-in-L1)",
        "BF16-R/L32/R-in-DRAM vs FP32-R/L32/R-in-DRAM",
        "BF16-R/L32/R-in-L1 vs rejected FP32-R/L32/R-in-L1",
    ]
    assert [
        item["byte_differences_right_minus_left"]["total"]
        for item in artifact["matched_configuration_differences"]
    ] == [344064, 98304, 589824]


def test_issue88_inverse_persistence_failure_fails_run_before_pass(tmp_path):
    blocked_output = tmp_path / "not-a-directory"
    blocked_output.write_text("file blocks artifact directory")
    matrices = np.broadcast_to(np.eye(2, dtype=np.complex64), (2, 2, 2)).copy()

    class FakeKernel:
        @classmethod
        def prepare(cls, ttnn, device, input_matrices, **kwargs):
            kernel = cls()
            kernel.actual = runner.reference_context(
                input_matrices, kwargs["variant"]
            )["fixed_reference"]
            return kernel

        def launch(self):
            return None

        def result(self):
            return self.actual

        def close(self):
            return None

    config = next(
        row for row in runner.ISSUE88_COMPARISON_ROWS if row["name"] == "bf16-r-L16"
    )
    result = runner.run_comparison(
        SimpleNamespace(synchronize_device=lambda device: None),
        object(),
        artifact_dir=blocked_output,
        run_id="run-persist-failure",
        rows=(config,),
        launches=1,
        first_launch=True,
        kernel_class=FakeKernel,
        matrices_factory=lambda *args, **kwargs: matrices,
    )

    assert result["status"] == "failed"
    assert result["comparison_rows"][0]["status"] == "failed"
    assert result["comparison_rows"][0]["failure_stage"] == "inverse_artifact_persistence"
    assert result["device_inverse_artifacts"] == []


def test_issue88_runner_parser_rejects_the_forbidden_fp32_l32_l1_placement():
    parser = runner._build_parser()
    args = parser.parse_args(
        [
            "--only",
            "newton_schulz_L32_b8192",
            "--custom-variant",
            "fp32-r",
            "--r-memory",
            "l1",
        ]
    )
    with pytest.raises(SystemExit):
        runner._validate_args(parser, args)
