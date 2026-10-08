"""Rebuild Issue #88 host-only beam metrics without touching device artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from enodia.tt.bench.newton_schulz_reference import (
    bf16_round_complex,
    initial_value,
    newton_schulz_reference,
    random_hpd_batch,
)
from tools import newton_schulz_issue88 as runner

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RECORD = ROOT / (
    "docs/measurements/2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json"
)
OUTPUT_RECORD = ROOT / (
    "docs/measurements/2026-10-08-p150a-newton-schulz-issue88-fp32-r-beam-metric-summary.json"
)
SCHEMA = "issue88-beam-metric-diagnostic-v1"


def _fingerprinted_reference(
    matrices: np.ndarray,
    variant: str,
    true_inverse: np.ndarray,
    x0: np.ndarray,
) -> dict[str, np.ndarray]:
    reference_r = bf16_round_complex(matrices) if variant == "bf16" else matrices
    fixed_reference = newton_schulz_reference(reference_r, iterations=12, x0=x0)
    return {
        "reference_r": reference_r,
        "x0": x0,
        "fixed_reference": fixed_reference,
        "true_inverse": true_inverse,
    }


def _matrix_transform_diagnostics(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, Any]:
    transforms = {
        "same_orientation": reference,
        "conjugated_reference": reference.conj(),
        "transposed_reference": reference.transpose(0, 2, 1),
    }
    diagnostics = {
        name: runner.best_complex_scalar_alignment(candidate, transformed)
        for name, transformed in transforms.items()
    }
    residuals = {
        name: details["aligned_relative_frobenius_error"]
        for name, details in diagnostics.items()
    }
    same_orientation_error = residuals["same_orientation"]
    conjugated_error = residuals["conjugated_reference"]
    transposed_error = residuals["transposed_reference"]
    diagnostics["classification"] = {
        "classification_tolerance_relative_residual": 1e-12,
        "best_transform_by_scalar_residual": min(
            (name for name, value in residuals.items() if value is not None),
            key=lambda name: residuals[name],
            default=None,
        ),
        "global_complex_scalar_multiple": bool(
            same_orientation_error is not None and same_orientation_error <= 1e-12
        ),
        "conjugated_reference_match": bool(
            conjugated_error is not None
            and conjugated_error <= 1e-12
            and (same_orientation_error is None or conjugated_error < same_orientation_error)
        ),
        "transposed_reference_match": bool(
            transposed_error is not None
            and transposed_error <= 1e-12
            and (same_orientation_error is None or transposed_error < same_orientation_error)
        ),
        "interpretation": (
            "Best scalar alignment is a diagnostic, not a matrix correctness metric; "
            "a normalized MV weight may remove an inverse-wide scalar."
        ),
    }
    return diagnostics


def _source_row_diagnostics(
    source_row: dict[str, Any],
    context: dict[str, np.ndarray],
    *,
    local_matrix_sha256: str,
) -> dict[str, Any]:
    candidate = context["fixed_reference"]
    true_inverse = context["true_inverse"]
    quality = runner.same_array_metrics(candidate, true_inverse)
    inverse_transforms = _matrix_transform_diagnostics(candidate, true_inverse)
    correctness = source_row.get("correctness") or {}
    source_quality = correctness.get("quality_vs_true_inverse") or {}
    source_beam = source_quality.get("beam_pattern") or {}
    weight_transforms = quality["mv_weight_direction"]["weight_ratio_diagnostics"]
    weight_orientation_error = weight_transforms["relative_residual_after_best_global_scalar"]
    weight_conjugate_error = weight_transforms[
        "candidate_vs_conjugate_reference_relative_error_after_scalar"
    ]
    weight_transpose_error = weight_transforms[
        "candidate_vs_transpose_reference_relative_error_after_scalar"
    ]
    return {
        "row": source_row["name"],
        "source_status": source_row["status"],
        "variant": source_row["variant"],
        "L": source_row["size"],
        "R_format": source_row["r_format"],
        "R_placement": source_row["r_memory"],
        "source_row_provenance": {
            "batch": source_row["batch"],
            "iterations": source_row["iterations"],
            "initial_value": source_row["initial_value"],
            "state_format": source_row["state_format"],
            "destination_format": source_row["destination_format"],
            "fp32_dest_acc_en": source_row["fp32_dest_acc_en"],
            "math_fidelity": source_row["math_fidelity"],
            "fuse_s": source_row["fuse_s"],
            "matrix_block": source_row["matrix_block"],
            "double_buffer": source_row["double_buffer"],
            "dst_full_sync_en": source_row["dst_full_sync_en"],
            "input_memory": source_row["input_memory"],
            "x0_memory": source_row["x0_memory"],
            "output_memory": source_row["output_memory"],
            "packing": source_row["packing"],
            "R_placement": source_row["r_memory"],
            "input_fingerprint": source_row.get("input_fingerprint"),
            "reference_fingerprint": source_row.get("reference_fingerprint"),
            "preflight": source_row.get("preflight"),
        },
        "candidate_definition": (
            "Host fixed-N=12 NumPy reference for this row's R representation and original-R X0; "
            "not the device inverse output."
        ),
        "device_inverse_output_available": False,
        "host_regenerated_matrix_sha256": local_matrix_sha256,
        "historical_device_inverse_relative_error_vs_true_inverse": correctness.get(
            "relative_frobenius_error_vs_true_inverse"
        ),
        "historical_device_phase_sensitive_complex_response_error": source_beam.get(
            "relative_frobenius_error"
        ),
        "historical_device_metric_status": (
            "invalidated for beam-pattern interpretation: the original weight denominator "
            "contracted unrelated einsum indices and was not aᴴPa"
            if correctness
            else "not measured: source row was rejected before device allocation"
        ),
        "host_reference_metrics_vs_true_inverse": quality,
        "host_weight_transform_assessment": {
            "classification_tolerance_relative_residual": 1e-12,
            "best_transform_by_scalar_residual": min(
                (
                    name
                    for name, value in (
                        ("same_orientation", weight_orientation_error),
                        ("conjugated_reference", weight_conjugate_error),
                        ("transposed_reference", weight_transpose_error),
                    )
                    if value is not None
                ),
                key=lambda name: {
                    "same_orientation": weight_orientation_error,
                    "conjugated_reference": weight_conjugate_error,
                    "transposed_reference": weight_transpose_error,
                }[name],
                default=None,
            ),
            "global_complex_scale_or_phase_only": bool(
                weight_orientation_error is not None and weight_orientation_error <= 1e-12
            ),
            "conjugated_reference_match": bool(
                weight_conjugate_error is not None
                and weight_conjugate_error <= 1e-12
                and (weight_orientation_error is None or weight_conjugate_error < weight_orientation_error)
            ),
            "transposed_reference_match": bool(
                weight_transpose_error is not None
                and weight_transpose_error <= 1e-12
                and (weight_orientation_error is None or weight_transpose_error < weight_orientation_error)
            ),
            "same_orientation_residual_after_scalar": weight_orientation_error,
            "conjugate_residual_after_scalar": weight_conjugate_error,
            "transpose_residual_after_scalar": weight_transpose_error,
            "ratio_summary": weight_transforms["ratio_summary"],
            "interpretation": "Residuals are for host fixed-N references only; no device inverse arrays were persisted in the source record.",
        },
        "host_inverse_transform_diagnostics": inverse_transforms,
    }


def build_summary(source_path: Path = SOURCE_RECORD) -> dict[str, Any]:
    """Build a reproducible host-only metric correction from the source record."""
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    source_rows = source["measurement"]["comparison_rows"]
    input_generation = source["input_generation"]
    batch = int(source_rows[0]["batch"])
    condition_number = float(input_generation["condition_number"])
    seed = int(input_generation["seed"])
    generated: dict[int, dict[str, Any]] = {}
    regenerated_fingerprints: dict[str, Any] = {}

    for size_text, source_fingerprint in input_generation["fingerprints"].items():
        size = int(size_text)
        matrices = random_hpd_batch(
            batch,
            size,
            condition_number=condition_number,
            seed=seed,
        )
        x0 = initial_value(matrices)
        true_inverse = np.linalg.inv(matrices.astype(np.complex128))
        local_matrix_sha256 = runner._array_sha256(matrices)
        true_inverse_sha256 = runner._array_sha256(true_inverse)
        variant_contexts: dict[str, dict[str, np.ndarray]] = {}
        reference_fingerprints: dict[str, Any] = {}
        for variant in ("bf16", "fp32-r"):
            context = _fingerprinted_reference(matrices, variant, true_inverse, x0)
            variant_contexts[variant] = context
            reference_fingerprints[variant] = {
                "reference_R_sha256": runner._array_sha256(context["reference_r"]),
                "X0_sha256": runner._array_sha256(x0),
                "fixed_N_reference_sha256": runner._array_sha256(
                    context["fixed_reference"]
                ),
                "true_inverse_sha256": true_inverse_sha256,
            }
        generated[size] = {
            "matrices": matrices,
            "local_matrix_sha256": local_matrix_sha256,
            "variant_contexts": variant_contexts,
        }
        source_reference_fingerprints: dict[str, Any] = {}
        reference_hash_matches: dict[str, Any] = {}
        for variant in ("bf16", "fp32-r"):
            source_row = next(
                row
                for row in source_rows
                if int(row["size"]) == size and row["variant"] == variant
            )
            source_reference = source_row.get("reference_fingerprint")
            source_reference_fingerprints[variant] = {
                "input_fingerprint": source_row.get("input_fingerprint"),
                "reference_fingerprint": source_reference,
            }
            local_reference = reference_fingerprints[variant]
            if source_reference is None:
                reference_hash_matches[variant] = None
            else:
                reference_hash_matches[variant] = {
                    "reference_R": source_reference.get("reference_r_sha256")
                    == local_reference["reference_R_sha256"],
                    "X0": source_reference.get("x0_sha256")
                    == local_reference["X0_sha256"],
                    "fixed_N_reference": source_reference.get("fixed_reference_sha256")
                    == local_reference["fixed_N_reference_sha256"],
                    "true_inverse": source_reference.get("true_inverse_sha256")
                    == local_reference["true_inverse_sha256"],
                }
        regenerated_fingerprints[size_text] = {
            "source": source_fingerprint,
            "source_row_fingerprints_by_variant": source_reference_fingerprints,
            "host_regenerated": {
                "size": size,
                "batch": batch,
                "condition_number": condition_number,
                "seed": seed,
                "matrix_sha256": local_matrix_sha256,
                "true_inverse_sha256": true_inverse_sha256,
                "references_by_variant": reference_fingerprints,
            },
            "matrix_sha256_matches_source": (
                local_matrix_sha256 == source_fingerprint["matrix_sha256"]
            ),
            "reference_fingerprints_match_source_by_variant": reference_hash_matches,
        }

    rows = []
    for source_row in source_rows:
        context = generated[int(source_row["size"])]["variant_contexts"][
            source_row["variant"]
        ]
        rows.append(
            _source_row_diagnostics(
                source_row,
                context,
                local_matrix_sha256=generated[int(source_row["size"])][
                    "local_matrix_sha256"
                ],
            )
        )

    return {
        "summary_schema": SCHEMA,
        "issue": "#88",
        "source_record": {
            "file": source_path.name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "source_power_trace": {
            "file": "2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv",
            "sha256": hashlib.sha256(
                (source_path.parent / "2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv").read_bytes()
            ).hexdigest(),
            "modified": False,
        },
        "host_only": True,
        "hardware_or_device_accessed": False,
        "host_environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_reproduction": {
            "generator": input_generation["generator"],
            "condition_number": condition_number,
            "seed": seed,
            "same_matrix_for_matching_size_variants": input_generation[
                "same_matrix_for_matching_size_variants"
            ],
            "source_fingerprints": regenerated_fingerprints,
            "limitation": (
                "The immutable source record stores matrix and reference SHA-256 fingerprints, "
                "not the arrays or downloaded device inverse matrices. The seed/condition "
                "generator was rerun here, but regenerated matrix-byte fingerprints do not "
                "match the source fingerprints under this host NumPy environment. Thus these "
                "host-reference diagnostics are not represented as exact recomputations of "
                "the measured device rows."
            ),
        },
        "metric_definitions": {
            "MV weight": "w = P a / (aᴴ P a), with P the inverse and steering a; normalization is checked as wᴴa=1.",
            "direction cosine deficit": "1 - |w_trueᴴ w_candidate| / (||w_true||₂ ||w_candidate||₂).",
            "complex ratios": "candidate weight / true-inverse weight where |w_true| > 1e-12 times that vector's maximum component; undefined counts and ratio range/phase summaries are retained.",
            "phase_sensitive_complex_response": "||w_candidateᴴa(theta)-w_trueᴴa(theta)||_F / ||w_trueᴴa(theta)||_F; the complex phase is retained.",
            "best complex scalar alignment": "alpha = <candidate_response, true_response> / ||candidate_response||²; report alpha and the relative Frobenius residual of alpha*candidate_response.",
            "magnitude response": "Relative Frobenius error between |candidate response| and |true response|, before per-pattern peak normalization; phase is discarded but scale is retained.",
            "normalized magnitude pattern": "Relative Frobenius error after each look's sampled main response magnitude normalizes its pattern.",
            "dB pattern": "20*log10(max(|response|/|response_at_look|, 1e-6)); candidate and reference normalized separately; -120 dB floor applies to exact/sub-floor zeros and counts are recorded.",
            "nonfinite and undefined values": "Metric helpers retain finite/total/undefined counts; dB metrics also count undefined look peaks and floor hits.",
        },
        "kernel_reconstruction_audit": {
            "complex_representation": "Separate real and imaginary float32 planes; L=16 uses diagonal 16x16 blocks packed into 32x32 tiles and unpacks the same diagonal corners.",
            "complex_multiply_signs": "R*X real=R_re*X_re-R_im*X_im and imag=R_re*X_im+R_im*X_re; X*S uses the negative X_imag plane for the real term.",
            "download_reconstruction": "Each output plane is downloaded as row-major float32, unpacked with matching packed corners, then reconstructed as real + 1j*imag.",
            "recorded_device_inverse_relative_error_vs_true_inverse_by_row": [
                {
                    "row": row["name"],
                    "relative_frobenius_error": row.get("correctness", {}).get(
                        "relative_frobenius_error_vs_true_inverse"
                    ),
                }
                for row in source_rows
                if row.get("correctness") is not None
            ],
            "conjugation_or_transpose_defect_found": False,
            "basis": "Board-free source audit of the input pack/unpack helpers, complex multiply signs in newton_schulz_compute.cpp, tensor download, and result reconstruction. Recorded inverse errors are also far below a wholesale conjugate/transpose mismatch; no accelerator runtime validation was performed.",
        },
        "historical_metric_handling": {
            "phase_sensitive_complex_response_retained": True,
            "role": "Secondary diagnostic alongside phase-aligned complex, magnitude, and dB-pattern metrics; useful for coherent response phase errors but not by itself a pattern-shape metric.",
            "historical_values": "The source JSON and its reference-labeled summary remain unchanged. Their beam values came from the erroneous normalization contraction and are invalid as beam-pattern evidence.",
        },
        "host_reference_diagnostics": {
            "candidate_scope": "Fixed-N=12 NumPy references for each source row's BF16-R or FP32-R variant, compared with the host-regenerated complex128 inverse; not downloaded accelerator outputs.",
            "rows": rows,
        },
        "conclusion": {
            "metric_definition_defect_found": True,
            "defect": "_weight used einsum labels 'j,bi->b', which independently summed steering and weights instead of contracting aᴴPa. At ±30 degrees for the 16/32-element half-wavelength array, the steering-vector sum is zero (up to floating residual), creating singular normalization and enormous phase-sensitive errors.",
            "kernel_or_reconstruction_conjugation_transpose_defect_found": False,
            "device_level_fp32_r_beam_ranking_supported": False,
            "corrected_device_beam_ranking": "Not available: the source record retains correctness summaries and fingerprints, not the downloaded inverse matrices needed to recompute corrected device-level response metrics.",
            "corrected_host_only_conclusion": "The old source beam-pattern conclusion that FP32-R is worse is unsupported and must not be used: its metric normalization is defective. Corrected host-reference metrics are diagnostic only and do not replace device-output evidence. Keep the R representation comparison open pending retained device outputs or an authorized rerun.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_RECORD)
    parser.add_argument("--output", type=Path, default=OUTPUT_RECORD)
    parser.add_argument("--write", action="store_true", help="write the derived JSON summary")
    args = parser.parse_args()
    summary = build_summary(args.source)
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.write:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
