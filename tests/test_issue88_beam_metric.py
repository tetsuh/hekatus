"""Board-free regression tests for Issue #88 beam-response metrics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from enodia.tt.bench.newton_schulz_reference import random_hpd_batch
from tools import newton_schulz_issue88 as runner

SOURCE_RECORD = Path(__file__).parents[1] / (
    "docs/measurements/2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json"
)
BEAM_SUMMARY = Path(__file__).parents[1] / (
    "docs/measurements/2026-10-08-p150a-newton-schulz-issue88-fp32-r-beam-metric-summary.json"
)


def test_issue88_true_inverse_health_and_mv_normalization():
    matrices = random_hpd_batch(8, 16, condition_number=100.0, seed=6300)
    true_inverse = np.linalg.inv(matrices.astype(np.complex128))
    metrics = runner.same_array_metrics(true_inverse, true_inverse)

    assert metrics["mv_weight_direction"]["max_cosine_deficit"] == pytest.approx(0.0)
    assert metrics["beam_pattern"]["phase_sensitive_complex_response_relative_frobenius_error"] == pytest.approx(
        0.0
    )
    assert metrics["beam_pattern"]["magnitude_response_relative_frobenius_error"] == pytest.approx(
        0.0
    )
    for direction in runner.LOOK_DIRECTIONS_DEG:
        steering = runner.steering_vector(16, direction)
        weights = runner._weight(true_inverse, steering)
        np.testing.assert_allclose(
            np.einsum("bi,i->b", weights.conj(), steering), 1.0, atol=2e-14
        )
    for size in (16, 32):
        for direction in (-30.0, 30.0):
            assert abs(np.sum(runner.steering_vector(size, direction))) < 1e-12


def test_issue88_device_metrics_compare_original_R_true_inverse_separately():
    matrices = random_hpd_batch(2, 4, condition_number=100.0, seed=runner.INPUT_SEED)
    context = runner.reference_context(matrices, "bf16")
    metrics = runner.correctness_metrics(context["true_inverse"], context)

    true_quality = metrics["quality_vs_true_inverse"]
    matching_quality = metrics["quality_vs_matching_reference"]
    assert true_quality["metric_reference"] == runner.TRUE_INVERSE_METRIC_REFERENCE
    assert true_quality["beam_pattern"]["metric_reference"] == (
        runner.TRUE_INVERSE_METRIC_REFERENCE
    )
    assert true_quality["beam_pattern"][
        "phase_sensitive_complex_response_relative_frobenius_error"
    ] == pytest.approx(0.0)
    assert matching_quality["metric_reference"] == context[
        "matching_reference_metric_reference"
    ]
    assert matching_quality["beam_pattern"][
        "phase_sensitive_complex_response_relative_frobenius_error"
    ] > 0.0
    assert true_quality["beam_pattern"]["metric_definitions"] == (
        runner.BEAM_RESPONSE_METRIC_DEFINITIONS
    )


def test_issue88_complex_scalar_alignment_separates_phase_from_pattern_shape():
    reference = np.asarray([[[1.0 + 0.0j, 0.4 - 0.2j, -0.1 + 0.3j]]])
    candidate = 1.7 * np.exp(0.73j) * reference
    metrics = runner.beam_response_metrics(candidate, reference, look_indices=(0,))

    assert metrics["phase_sensitive_complex_response_relative_frobenius_error"] > 0.5
    assert metrics["phase_aligned_complex_response_relative_frobenius_error"] < 1e-14
    assert metrics["best_complex_scalar"]["magnitude"] == pytest.approx(1.0 / 1.7)
    assert metrics["best_complex_scalar"]["phase_degrees"] == pytest.approx(
        -np.rad2deg(0.73)
    )
    assert metrics["magnitude_response_relative_frobenius_error"] == pytest.approx(0.7)
    assert metrics["normalized_magnitude_pattern_relative_frobenius_error"] < 1e-14
    assert metrics["db_pattern"]["rms_absolute_error_db"] == pytest.approx(0.0)


def test_issue88_db_pattern_floor_is_finite_and_reports_floored_values():
    reference = np.asarray([[[1.0 + 0.0j, 0.0j]]])
    candidate = np.asarray([[[1.0 + 0.0j, 1e-4 + 0.0j]]])
    metrics = runner.beam_response_metrics(candidate, reference, look_indices=(0,))

    db = metrics["db_pattern"]
    assert db["floor_db"] == -120.0
    assert db["reference_floored_values"] == 1
    assert db["candidate_floored_values"] == 0
    assert db["rms_absolute_error_db"] == pytest.approx(40.0 / np.sqrt(2.0))
    assert np.isfinite(db["max_absolute_error_db"])


def test_issue88_all_zero_response_is_reported_as_undefined_not_nonfinite_json():
    zeros = np.zeros((1, 1, 2), dtype=np.complex128)
    metrics = runner.beam_response_metrics(zeros, zeros, look_indices=(0,))

    assert metrics["phase_sensitive_complex_response_relative_frobenius_error"] is None
    assert metrics["phase_aligned_complex_response_relative_frobenius_error"] is None
    assert metrics["alignment_undefined"] is True
    assert metrics["db_pattern"]["undefined_values"] == 2
    assert metrics["db_pattern"]["candidate_undefined_look_peaks"] == 1
    assert metrics["db_pattern"]["reference_undefined_look_peaks"] == 1
    json.dumps(metrics, allow_nan=False)


def test_issue88_beam_summary_maps_rows_without_relabeling_device_metrics():
    source_bytes = SOURCE_RECORD.read_bytes()
    source = json.loads(source_bytes)
    summary = json.loads(BEAM_SUMMARY.read_text())

    assert summary["source_record"] == {
        "file": SOURCE_RECORD.name,
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    rows = summary["host_reference_diagnostics"]["rows"]
    source_rows = source["measurement"]["comparison_rows"]
    assert [row["row"] for row in rows] == [row["name"] for row in source_rows]
    assert [row["source_status"] for row in rows] == [row["status"] for row in source_rows]
    for row, source_row in zip(rows, source_rows, strict=True):
        assert row["source_row_provenance"]["input_fingerprint"] == source_row.get(
            "input_fingerprint"
        )
        assert row["source_row_provenance"]["reference_fingerprint"] == source_row.get(
            "reference_fingerprint"
        )
        assert row["device_inverse_output_available"] is False
        beam = row["host_reference_metrics_vs_true_inverse"]["beam_pattern"]
        assert "phase_sensitive_complex_response_relative_frobenius_error" in beam
        assert "phase_aligned_complex_response_relative_frobenius_error" in beam
        assert "magnitude_response_relative_frobenius_error" in beam
        assert "normalized_magnitude_pattern_relative_frobenius_error" in beam
        assert beam["db_pattern"]["floor_db"] == -120.0
        assert "host_weight_transform_assessment" in row
    assert summary["conclusion"]["device_level_fp32_r_beam_ranking_supported"] is False
    assert summary["conclusion"]["kernel_or_reconstruction_conjugation_transpose_defect_found"] is False
    assert summary["historical_metric_handling"]["phase_sensitive_complex_response_retained"] is True
