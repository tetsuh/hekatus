"""Measure the Issue #88 BF16-R versus FP32-R catalogue in one device session.

The runner is intentionally separate from the generic catalogue driver.  It
feeds the accelerator the deterministic HPD matrices used by the independent
NumPy reference, performs one synchronized correctness launch for every row,
and then retains one synchronized sample for each of the 1,000 timed launches.
No accelerator-independent specification module is imported here: the board
result is compared with ``enodia.tt.bench.newton_schulz_reference`` only.

The script is used as a custom runner by ``run_in_container.sh``.  The wrapper
captures environment and power artifacts beside the result path; this runner
copies only sanitized, basename-bound telemetry into its raw audit artifact.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import itertools
import json
import math
import os
import re
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.newton_schulz_reference import (
    NEWTON_SCHULZ_ITERATIONS,
    bf16_round_complex,
    initial_value,
    newton_schulz_reference,
    random_hpd_batch,
)

DEVICE_ID = 0
LAUNCHES_PER_ROW = 1_000
ROW_TIMEOUT_S = 60.0
# The sibling sampler remains alive until the wrapper observes the runner exit.
# Polling is deliberately bounded below the existing 60-second row cap.
POWER_TRACE_POLL_TIMEOUT_S = 30.0
POWER_TRACE_POLL_INTERVAL_S = 0.5
CONDITION_NUMBER = 100.0
INPUT_SEED = 6300
FIXED_ITERATIONS = NEWTON_SCHULZ_ITERATIONS
# Public aliases follow the naming used by the other board-side runners.
ISSUE88_DEVICE_ID = DEVICE_ID
ISSUE88_LAUNCHES = LAUNCHES_PER_ROW
ISSUE88_CONDITION_NUMBER = CONDITION_NUMBER
ISSUE88_SEED = INPUT_SEED
ISSUE88_ITERATIONS = FIXED_ITERATIONS
DEVICE_TEST_RELATIVE_ERROR_GATE = 1e-2
FP32_R_L32_L1_PREFLIGHT_BYTES = 1_884_928
FP32_R_L32_L1_PREFLIGHT_OVERAGE_BYTES = 312_064
FP32_R_L32_L1_PREFLIGHT_STATUS = "rejected_before_allocation"
SUCCESSFUL_ROW_STATUSES = frozenset({"ok", "correctness_only", "preflight_rejected"})

LOOK_DIRECTIONS_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
PATTERN_DIRECTIONS_DEG = (
    -60.0,
    -45.0,
    -30.0,
    -15.0,
    0.0,
    15.0,
    30.0,
    45.0,
    60.0,
)

ISSUE88_RECORD_SCHEMA = "adr-0005-issue88-fp32-r-v1"
ISSUE88_RAW_SCHEMA = "adr-0005-issue88-fp32-r-raw-v1"
ISSUE88_RUNNER = "tools/newton_schulz_issue88.py"
TRUE_INVERSE_METRIC_REFERENCE = "NumPy complex128 inverse of original FP32 R"
POWER_TRACE_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")
_PCI_BUS_ID_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", re.IGNORECASE)
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)
_HARNESS_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)

# This is the one source of truth for the publication gate.  The builder
# iterates this table when it emits status_components, and board-free tests
# iterate it rather than maintaining a second list of required checks.
ISSUE88_STATUS_COMPONENTS = (
    ("rows", "every selected comparison row succeeds"),
    (
        "board_selection",
        "telemetry board selection is verifiable and has board type, serial, PCI identity, and firmware",
    ),
    (
        "image_toolchain",
        "the image is digest-pinned and kernel-driver/toolchain fields are present",
    ),
    ("harness", "harness_commit is valid and harness_dirty is false"),
    ("power_trace", "the power trace is readable, nonempty, and covers the device run"),
    ("device_close", "device close succeeds"),
)
_PRIVATE_METADATA_KEYS = frozenset(
    {
        "hostname",
        "host_name",
        "machine",
        "machine_name",
        "node_name",
        "username",
        "user_name",
        "user",
        "home",
        "home_directory",
        "cwd",
        "working_directory",
        "password",
        "secret",
        "token",
        "access_token",
        "api_key",
        "private_key",
    }
)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:])/(?:home|Users|tmp|var/tmp|workspace|workspaces|work|out|mnt|opt|root|run/user|dev|build|src)/[^\s,;\"']+"
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


# The tuple is built by a function so callers and tests receive independent
# dictionaries and cannot mutate the catalogue used by a later row.
def comparison_rows() -> tuple[dict[str, Any], ...]:
    """Return the placement-explicit Issue #88 comparison rows."""
    common = {
        "batch": 8192,
        "iterations": FIXED_ITERATIONS,
        "initial_value": "I/||R||inf",
        "state_format": "BF16",
        "destination_format": "FP32",
        "fp32_dest_acc_en": True,
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "input_memory": "l1",
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "output_memory": "dram",
        "x0_memory": "l1",
    }
    rows = (
        {
            **common,
            "name": "bf16-r-L16",
            "variant": "bf16",
            "r_format": "BF16",
            "size": 16,
            "packing": "diagonal_pairs_32x32",
            "r_memory": "l1",
        },
        {
            **common,
            "name": "fp32-r-L16",
            "variant": "fp32-r",
            "r_format": "FP32",
            "size": 16,
            "packing": "diagonal_pairs_32x32",
            "r_memory": "l1",
        },
        {
            **common,
            "name": "bf16-r-L32",
            "variant": "bf16",
            "r_format": "BF16",
            "size": 32,
            "packing": "native_32x32",
            "r_memory": "l1",
        },
        {
            **common,
            "name": "bf16-r-L32-r-dram",
            "variant": "bf16",
            "r_format": "BF16",
            "size": 32,
            "packing": "native_32x32",
            "r_memory": "dram",
            "configuration_note": (
                "R in DRAM is explicit to match the L=32 FP32-R placement-control row; "
                "no placement fallback is performed."
            ),
        },
        {
            **common,
            "name": "fp32-r-L32-r-dram",
            "variant": "fp32-r",
            "r_format": "FP32",
            "size": 32,
            "packing": "native_32x32",
            "r_memory": "dram",
            "configuration_note": (
                "R in DRAM is explicit because the selected block-8 R-in-L1 "
                "preflight rejects before allocation."
            ),
        },
        {
            **common,
            "name": "fp32-r-L32",
            "variant": "fp32-r",
            "r_format": "FP32",
            "size": 32,
            "packing": "native_32x32",
            "r_memory": "l1",
            "configuration_note": (
                "Expected host-only preflight rejection; block 8 R-in-L1 exceeds "
                "the L1 budget and no placement fallback is performed."
            ),
            "preflight_status": FP32_R_L32_L1_PREFLIGHT_STATUS,
            "preflight_bytes": FP32_R_L32_L1_PREFLIGHT_BYTES,
            "preflight_over_budget_bytes": FP32_R_L32_L1_PREFLIGHT_OVERAGE_BYTES,
            "expected_preflight_rejection": True,
        },
    )
    return tuple(dict(row) for row in rows)


ISSUE88_COMPARISON_ROWS = comparison_rows()


def validate_comparison_rows(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> None:
    """Validate the full Issue #88 catalogue or an exact selected subset."""
    selected = ISSUE88_COMPARISON_ROWS if rows is None else tuple(rows)
    if not selected:
        raise ValueError("Issue #88 requires at least one comparison row")

    canonical = {row["name"]: row for row in ISSUE88_COMPARISON_ROWS}
    names = [row.get("name") for row in selected]
    unknown = [name for name in names if name not in canonical]
    if unknown:
        raise ValueError(f"Issue #88 comparison row names are unknown: {unknown!r}")
    if len(set(names)) != len(names):
        raise ValueError("Issue #88 comparison rows must not contain duplicate names")
    if rows is None and set(names) != set(canonical):
        raise ValueError("Issue #88 full comparison catalogue is incomplete")

    expected = {
        "batch": 8192,
        "iterations": 12,
        "initial_value": "I/||R||inf",
        "state_format": "BF16",
        "destination_format": "FP32",
        "fp32_dest_acc_en": True,
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "input_memory": "l1",
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "output_memory": "dram",
        "x0_memory": "l1",
    }
    for row in selected:
        name = row["name"]
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"{name} changes required {key}: expected {value!r}, "
                    f"got {row.get(key)!r}"
                )
        for key in ("variant", "r_format", "size", "packing", "r_memory"):
            if row.get(key) != canonical[name][key]:
                raise ValueError(
                    f"{name} changes required {key}: "
                    f"expected {canonical[name][key]!r}, got {row.get(key)!r}"
                )


def shape_name(row: dict[str, Any]) -> str:
    """Return the kernel catalogue shape name represented by one row."""
    return f"newton_schulz_L{row['size']}_b{row['batch']}"


def percentile(samples: list[float] | tuple[float, ...], quantile: float) -> float:
    """Return the linearly interpolated percentile used by board records."""
    if not samples:
        raise ValueError("at least one timing sample is required")
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be between 0 and 1, got {quantile!r}")
    ordered = sorted(float(value) for value in samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def timing_summary(samples: list[float], flops_per_launch: int) -> dict[str, Any]:
    """Derive every retained timing and TFLOPS statistic from launch samples."""
    if not samples:
        raise ValueError("timing_summary requires at least one launch sample")
    if flops_per_launch < 1:
        raise ValueError("flops_per_launch must be positive")
    values = [float(sample) for sample in samples]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timing samples must be finite and positive")
    minimum = min(values)
    p50 = percentile(values, 0.50)
    p99 = percentile(values, 0.99)
    p99_9 = percentile(values, 0.999)

    def tflops(seconds: float) -> float:
        return flops_per_launch / seconds / 1e12

    return {
        "launches_measured": len(values),
        "seconds_per_launch_samples": values,
        "seconds_per_launch_min": minimum,
        "seconds_per_launch_p50": p50,
        "seconds_per_launch_p99": p99,
        "seconds_per_launch_p99_9": p99_9,
        "tflops_min_derived": tflops(minimum),
        "tflops_p50_derived": tflops(p50),
        "tflops_p99_derived": tflops(p99),
        "tflops_p99_9_derived": tflops(p99_9),
        "tflops_derivation": "flops_per_launch / seconds_per_launch_statistic / 1e12",
        # Keep the name used by the existing custom runner as an explicit alias.
        "achieved_tflops": tflops(minimum),
        "tflops_fastest_launch_derived": tflops(minimum),
    }


def steering_vector(
    aperture_size: int,
    direction_deg: float,
    *,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    """Build the half-wavelength linear-array steering vector locally."""
    if not isinstance(aperture_size, (int, np.integer)) or aperture_size < 1:
        raise ValueError(f"aperture_size must be positive, got {aperture_size}")
    if not np.isfinite(direction_deg):
        raise ValueError(f"direction_deg must be finite, got {direction_deg}")
    positions = np.arange(aperture_size, dtype=np.float64) - (aperture_size - 1.0) / 2.0
    phase = np.pi * positions * np.sin(np.deg2rad(direction_deg))
    return np.exp(1j * phase).astype(dtype)


def direction_cosine_deficit(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return ``1 - |uᴴv|/(||u|| ||v||)`` for two weight directions."""
    denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
    if not denominator > 0.0 or not np.isfinite(denominator):
        return float("nan")
    cosine = abs(np.vdot(reference, candidate)) / denominator
    return 1.0 - float(np.clip(cosine, 0.0, 1.0))


def _weight(inverse: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """Return MV weights ``P a / (aᴴ P a)`` for batched inverse matrices P."""
    weights = np.einsum("bij,j->bi", inverse, steering, optimize=True)
    # The second index must contract with the steering index.  Leaving these
    # labels distinct computes (sum(conj(a))) * (sum(Pa)), not aᴴPa; for the
    # Issue #88 ±30° steering vectors that erroneous sum is exactly zero.
    denominator = np.einsum("j,bj->b", steering.conj(), weights, optimize=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return weights / denominator[:, None]


def best_complex_scalar_alignment(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, Any]:
    """Fit one least-squares complex scalar and report its aligned residual."""
    candidate = np.asarray(candidate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have matching shapes")
    finite = np.isfinite(candidate) & np.isfinite(reference)
    candidate_values = candidate[finite]
    reference_values = reference[finite]
    denominator = float(np.vdot(candidate_values, candidate_values).real)
    reference_norm = float(np.linalg.norm(reference_values))
    if (
        candidate_values.size == 0
        or not math.isfinite(denominator)
        or denominator <= 0.0
        or not math.isfinite(reference_norm)
        or reference_norm <= 0.0
    ):
        return {
            "best_complex_scalar": None,
            "aligned_relative_frobenius_error": None,
            "finite_values": int(candidate_values.size),
            "total_values": int(candidate.size),
            "undefined": True,
        }
    scalar = np.vdot(candidate_values, reference_values) / denominator
    aligned_error = float(
        np.linalg.norm(scalar * candidate_values - reference_values) / reference_norm
    )
    return {
        "best_complex_scalar": {
            "real": float(scalar.real),
            "imag": float(scalar.imag),
            "magnitude": float(abs(scalar)),
            "phase_radians": float(np.angle(scalar)),
            "phase_degrees": float(np.rad2deg(np.angle(scalar))),
        },
        "aligned_relative_frobenius_error": aligned_error,
        "finite_values": int(candidate_values.size),
        "total_values": int(candidate.size),
        "undefined": False,
    }


def beam_response_metrics(
    candidate_response: np.ndarray,
    reference_response: np.ndarray,
    *,
    look_indices: tuple[int, ...],
    floor_db: float = -120.0,
) -> dict[str, Any]:
    """Compare complex, phase-aligned, magnitude, and dB response patterns.

    Responses have shape ``(batch, look, pattern_direction)``.  The dB pattern
    is normalized to each response's sampled look-direction magnitude and uses
    a fixed amplitude floor (default -120 dB); exact and sub-floor zeros are
    counted rather than converted to infinities.
    """
    candidate = np.asarray(candidate_response, dtype=np.complex128)
    reference = np.asarray(reference_response, dtype=np.complex128)
    if candidate.ndim != 3 or reference.shape != candidate.shape:
        raise ValueError("beam responses must have matching (batch, look, direction) shapes")
    if len(look_indices) != candidate.shape[1] or any(
        index < 0 or index >= candidate.shape[2] for index in look_indices
    ):
        raise ValueError("look_indices must identify one in-range pattern direction per look")
    if not math.isfinite(floor_db) or floor_db >= 0.0:
        raise ValueError("floor_db must be finite and below zero")

    finite = np.isfinite(candidate) & np.isfinite(reference)
    candidate_values = candidate[finite]
    reference_values = reference[finite]
    reference_norm = float(np.linalg.norm(reference_values))
    if candidate_values.size and reference_norm > 0.0 and math.isfinite(reference_norm):
        phase_sensitive_error = float(
            np.linalg.norm(candidate_values - reference_values) / reference_norm
        )
    else:
        phase_sensitive_error = None
    if reference_norm > 0.0 and math.isfinite(reference_norm):
        magnitude_error = float(
            np.linalg.norm(np.abs(candidate_values) - np.abs(reference_values))
            / np.linalg.norm(np.abs(reference_values))
        )
    else:
        magnitude_error = None
    alignment = best_complex_scalar_alignment(candidate, reference)

    batch, look_count, _direction_count = candidate.shape
    batch_indices = np.arange(batch)[:, None]
    look_indices_array = np.asarray(look_indices, dtype=np.intp)[None, :]
    candidate_peak = np.abs(candidate[batch_indices, np.arange(look_count)[None, :], look_indices_array])
    reference_peak = np.abs(reference[batch_indices, np.arange(look_count)[None, :], look_indices_array])
    candidate_peak_valid = np.isfinite(candidate_peak) & (candidate_peak > 0.0)
    reference_peak_valid = np.isfinite(reference_peak) & (reference_peak > 0.0)
    candidate_db_valid = finite & candidate_peak_valid[..., None]
    reference_db_valid = finite & reference_peak_valid[..., None]
    both_db_valid = candidate_db_valid & reference_db_valid
    candidate_magnitude = np.abs(candidate)
    reference_magnitude = np.abs(reference)
    candidate_normalized = np.full(candidate.shape, np.nan, dtype=np.float64)
    reference_normalized = np.full(reference.shape, np.nan, dtype=np.float64)
    np.divide(
        candidate_magnitude,
        candidate_peak[..., None],
        out=candidate_normalized,
        where=candidate_peak_valid[..., None] & np.isfinite(candidate_magnitude),
    )
    np.divide(
        reference_magnitude,
        reference_peak[..., None],
        out=reference_normalized,
        where=reference_peak_valid[..., None] & np.isfinite(reference_magnitude),
    )
    amplitude_floor = 10.0 ** (floor_db / 20.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate_db = 20.0 * np.log10(np.maximum(candidate_normalized, amplitude_floor))
        reference_db = 20.0 * np.log10(np.maximum(reference_normalized, amplitude_floor))
    candidate_floored = candidate_db_valid & (candidate_normalized < amplitude_floor)
    reference_floored = reference_db_valid & (reference_normalized < amplitude_floor)
    db_difference = candidate_db[both_db_valid] - reference_db[both_db_valid]
    db_rms = float(np.sqrt(np.mean(np.square(db_difference)))) if db_difference.size else None
    db_max = float(np.max(np.abs(db_difference))) if db_difference.size else None
    magnitude_pattern_valid = candidate_db_valid & reference_db_valid
    candidate_magnitude_pattern = candidate_normalized[magnitude_pattern_valid]
    reference_magnitude_pattern = reference_normalized[magnitude_pattern_valid]
    reference_magnitude_pattern_norm = float(np.linalg.norm(reference_magnitude_pattern))
    normalized_magnitude_pattern_error = (
        float(
            np.linalg.norm(candidate_magnitude_pattern - reference_magnitude_pattern)
            / reference_magnitude_pattern_norm
        )
        if reference_magnitude_pattern_norm > 0.0
        and math.isfinite(reference_magnitude_pattern_norm)
        else None
    )

    return {
        "phase_sensitive_complex_response_relative_frobenius_error": phase_sensitive_error,
        "phase_sensitive_complex_response_max_absolute_error": (
            float(np.max(np.abs(candidate_values - reference_values)))
            if candidate_values.size
            else None
        ),
        "phase_aligned_complex_response_relative_frobenius_error": alignment[
            "aligned_relative_frobenius_error"
        ],
        "best_complex_scalar": alignment["best_complex_scalar"],
        "alignment_finite_values": alignment["finite_values"],
        "alignment_total_values": alignment["total_values"],
        "alignment_undefined": alignment["undefined"],
        "magnitude_response_relative_frobenius_error": magnitude_error,
        "normalized_magnitude_pattern_relative_frobenius_error": normalized_magnitude_pattern_error,
        "magnitude_response_max_absolute_error": (
            float(np.max(np.abs(np.abs(candidate_values) - np.abs(reference_values))))
            if candidate_values.size
            else None
        ),
        "db_pattern": {
            "definition": (
                "20*log10(max(abs(response)/abs(response_at_look), "
                "10**(floor_db/20))); candidate and reference normalized separately"
            ),
            "floor_db": float(floor_db),
            "amplitude_floor": amplitude_floor,
            "rms_absolute_error_db": db_rms,
            "max_absolute_error_db": db_max,
            "finite_values": int(db_difference.size),
            "total_values": int(candidate.size),
            "undefined_values": int(candidate.size - np.count_nonzero(both_db_valid)),
            "candidate_floored_values": int(np.count_nonzero(candidate_floored)),
            "reference_floored_values": int(np.count_nonzero(reference_floored)),
            "candidate_undefined_look_peaks": int(np.count_nonzero(~candidate_peak_valid)),
            "reference_undefined_look_peaks": int(np.count_nonzero(~reference_peak_valid)),
        },
        "finite_values": int(candidate_values.size),
        "total_values": int(candidate.size),
        "undefined_values": int(candidate.size - candidate_values.size),
    }


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else None


def _finite_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else None


def _weight_ratio_diagnostics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Summarize complex candidate/reference weight ratios above a relative floor."""
    candidate = np.asarray(candidate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if candidate.shape != reference.shape or candidate.ndim < 1:
        raise ValueError("candidate and reference weights must have matching shapes")
    reference_scale = np.max(np.abs(reference), axis=-1, keepdims=True)
    threshold = reference_scale * 1e-12
    defined = np.isfinite(candidate) & np.isfinite(reference) & (np.abs(reference) > threshold)
    ratios = candidate[defined] / reference[defined]
    finite_ratios = ratios[np.isfinite(ratios)]
    alignment = best_complex_scalar_alignment(candidate, reference)
    if finite_ratios.size:
        magnitudes = np.abs(finite_ratios)
        phases = np.angle(finite_ratios)
        phase_vector = np.mean(np.exp(1j * phases))
        ratio_summary = {
            "magnitude_min": float(np.min(magnitudes)),
            "magnitude_p05": float(np.percentile(magnitudes, 5)),
            "magnitude_median": float(np.median(magnitudes)),
            "magnitude_p95": float(np.percentile(magnitudes, 95)),
            "magnitude_max": float(np.max(magnitudes)),
            "phase_circular_mean_radians": float(np.angle(phase_vector)),
            "phase_circular_resultant_length": float(abs(phase_vector)),
        }
    else:
        ratio_summary = None
    return {
        "definition": "candidate_weight/reference_weight where |reference_weight| > 1e-12 * max(|reference_weight|) per vector",
        "defined_ratios": int(np.count_nonzero(defined)),
        "total_ratios": int(candidate.size),
        "nonfinite_ratios": int(np.count_nonzero(defined) - finite_ratios.size),
        "ratio_summary": ratio_summary,
        "best_global_complex_scalar_candidate_from_reference": alignment["best_complex_scalar"],
        "relative_residual_after_best_global_scalar": alignment[
            "aligned_relative_frobenius_error"
        ],
        "global_scalar_alignment_undefined": alignment["undefined"],
    }


def same_array_metrics(
    candidate_inverse: np.ndarray,
    true_inverse: np.ndarray,
    *,
    look_directions_deg: tuple[float, ...] = LOOK_DIRECTIONS_DEG,
    pattern_directions_deg: tuple[float, ...] = PATTERN_DIRECTIONS_DEG,
) -> dict[str, Any]:
    """Compare normalized MV weights and explicitly labeled beam responses.

    The complex response error is retained as a phase-sensitive secondary
    diagnostic.  Phase-aligned complex, magnitude-only, and dB-pattern values
    separate overall response phase/scale from pattern-shape differences.
    """
    candidate = np.asarray(candidate_inverse)
    reference = np.asarray(true_inverse)
    if candidate.ndim == 2:
        candidate = candidate[None, ...]
    if reference.ndim == 2:
        reference = reference[None, ...]
    if (
        candidate.ndim != 3
        or reference.shape != candidate.shape
        or candidate.shape[1] != candidate.shape[2]
    ):
        raise ValueError("candidate_inverse and true_inverse must have matching batched square shapes")
    batch, aperture_size, _ = candidate.shape
    pattern = np.stack(
        [
            steering_vector(aperture_size, direction, dtype=np.complex128)
            for direction in pattern_directions_deg
        ],
        axis=1,
    )
    look_indices = tuple(
        pattern_directions_deg.index(direction) if direction in pattern_directions_deg else -1
        for direction in look_directions_deg
    )
    if -1 in look_indices:
        raise ValueError("pattern_directions_deg must include every look direction for dB normalization")

    deficits: list[np.ndarray] = []
    normalization_errors_candidate: list[np.ndarray] = []
    normalization_errors_reference: list[np.ndarray] = []
    true_responses: list[np.ndarray] = []
    candidate_responses: list[np.ndarray] = []
    true_weights: list[np.ndarray] = []
    candidate_weights: list[np.ndarray] = []
    conjugate_true_weights: list[np.ndarray] = []
    transpose_true_weights: list[np.ndarray] = []
    for direction in look_directions_deg:
        steering = steering_vector(aperture_size, direction, dtype=np.complex128)
        true_weight = _weight(reference.astype(np.complex128, copy=False), steering)
        candidate_weight = _weight(candidate.astype(np.complex128, copy=False), steering)
        denominator_norm = np.linalg.norm(true_weight, axis=1) * np.linalg.norm(candidate_weight, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = np.abs(
                np.einsum("bi,bi->b", true_weight.conj(), candidate_weight, optimize=True)
            ) / denominator_norm
        cosine = np.clip(cosine, 0.0, 1.0)
        deficits.append(1.0 - cosine)
        true_main_response = np.einsum("bi,i->b", true_weight.conj(), steering, optimize=True)
        candidate_main_response = np.einsum(
            "bi,i->b", candidate_weight.conj(), steering, optimize=True
        )
        normalization_errors_reference.append(np.abs(true_main_response - 1.0))
        normalization_errors_candidate.append(np.abs(candidate_main_response - 1.0))
        true_responses.append(np.einsum("bi,ij->bj", true_weight.conj(), pattern, optimize=True))
        candidate_responses.append(
            np.einsum("bi,ij->bj", candidate_weight.conj(), pattern, optimize=True)
        )
        true_weights.append(true_weight)
        candidate_weights.append(candidate_weight)
        conjugate_true_weights.append(true_weight.conj())
        transpose_inverse = reference.astype(np.complex128, copy=False).transpose(0, 2, 1)
        transpose_true_weights.append(_weight(transpose_inverse, steering))

    deficit_array = np.stack(deficits, axis=1)
    true_response_array = np.stack(true_responses, axis=1)
    candidate_response_array = np.stack(candidate_responses, axis=1)
    response_metrics = beam_response_metrics(
        candidate_response_array,
        true_response_array,
        look_indices=look_indices,
    )
    phase_sensitive_error = response_metrics[
        "phase_sensitive_complex_response_relative_frobenius_error"
    ]
    max_absolute_error = response_metrics[
        "phase_sensitive_complex_response_max_absolute_error"
    ]
    response_difference = candidate_response_array - true_response_array
    response_difference_norm = np.linalg.norm(response_difference, axis=2)
    response_reference_norm = np.linalg.norm(true_response_array, axis=2)
    response_error_array = np.full(response_reference_norm.shape, np.nan, dtype=np.float64)
    response_error_valid = (
        np.isfinite(response_difference_norm)
        & np.isfinite(response_reference_norm)
        & (response_reference_norm > 0.0)
    )
    np.divide(
        response_difference_norm,
        response_reference_norm,
        out=response_error_array,
        where=response_error_valid,
    )
    stacked_true_weights = np.stack(true_weights, axis=1)
    stacked_candidate_weights = np.stack(candidate_weights, axis=1)
    stacked_conjugate_weights = np.stack(conjugate_true_weights, axis=1)
    stacked_transpose_weights = np.stack(transpose_true_weights, axis=1)
    ratio_metrics = _weight_ratio_diagnostics(stacked_candidate_weights, stacked_true_weights)
    ratio_metrics["candidate_vs_conjugate_reference_relative_error_after_scalar"] = (
        best_complex_scalar_alignment(stacked_candidate_weights, stacked_conjugate_weights)[
            "aligned_relative_frobenius_error"
        ]
    )
    ratio_metrics["candidate_vs_transpose_reference_relative_error_after_scalar"] = (
        best_complex_scalar_alignment(stacked_candidate_weights, stacked_transpose_weights)[
            "aligned_relative_frobenius_error"
        ]
    )
    normalization_candidate = np.stack(normalization_errors_candidate, axis=1)
    normalization_reference = np.stack(normalization_errors_reference, axis=1)

    return {
        "steering_array": {
            "array": "equal-spaced linear array",
            "element_spacing": "one-half wavelength",
            "look_directions_deg": list(look_directions_deg),
            "pattern_directions_deg": list(pattern_directions_deg),
        },
        "mv_weight_direction": {
            "definition": "1 - absolute normalized direction cosine",
            "by_look_direction": [_finite_max(values) for values in deficits],
            "mean_by_look_direction": [_finite_mean(values) for values in deficits],
            "mean_cosine_deficit": _finite_mean(deficit_array),
            "max_cosine_deficit": _finite_max(deficit_array),
            "finite_values": int(np.count_nonzero(np.isfinite(deficit_array))),
            "total_values": int(batch * len(look_directions_deg)),
            "weight_ratio_diagnostics": ratio_metrics,
        },
        "weight_normalization": {
            "definition": "w = P a / (aᴴ P a); measure |wᴴ a - 1|",
            "candidate_max_absolute_error": _finite_max(normalization_candidate),
            "reference_max_absolute_error": _finite_max(normalization_reference),
            "candidate_finite_values": int(np.count_nonzero(np.isfinite(normalization_candidate))),
            "reference_finite_values": int(np.count_nonzero(np.isfinite(normalization_reference))),
            "total_values": int(batch * len(look_directions_deg)),
        },
        "beam_pattern": {
            "metric_definition": "beam response is wᴴ a(theta); reported response-pattern comparisons are over batch, look, and pattern directions",
            "phase_sensitive_complex_response_relative_frobenius_error": phase_sensitive_error,
            "phase_sensitive_complex_response_max_absolute_error": max_absolute_error,
            "phase_aligned_complex_response_relative_frobenius_error": response_metrics[
                "phase_aligned_complex_response_relative_frobenius_error"
            ],
            "best_complex_scalar": response_metrics["best_complex_scalar"],
            "magnitude_response_relative_frobenius_error": response_metrics[
                "magnitude_response_relative_frobenius_error"
            ],
            "normalized_magnitude_pattern_relative_frobenius_error": response_metrics[
                "normalized_magnitude_pattern_relative_frobenius_error"
            ],
            "magnitude_response_max_absolute_error": response_metrics[
                "magnitude_response_max_absolute_error"
            ],
            "db_pattern": response_metrics["db_pattern"],
            "phase_sensitive_complex_response_relative_error_by_look_direction": [
                _finite_mean(values) for values in response_error_array.T
            ],
            "phase_sensitive_complex_response_max_relative_error_by_look_direction": [
                _finite_max(values) for values in response_error_array.T
            ],
            "finite_values": int(np.count_nonzero(np.isfinite(response_error_array))),
            "total_values": int(batch * len(look_directions_deg)),
            "response_metric_finite_values": response_metrics["finite_values"],
            "response_metric_total_values": response_metrics["total_values"],
            "response_metric_undefined_values": response_metrics["undefined_values"],
        },
    }


# Aliases keep the local metric seam easy to exercise without importing the
# specification sweep module.
beam_pattern_metrics = same_array_metrics
_same_array_metrics = same_array_metrics
_direction_cosine_deficit = direction_cosine_deficit


def _relative_frobenius_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    denominator = np.linalg.norm(expected)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.linalg.norm(actual - expected) / denominator)


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def reference_context(
    matrices: np.ndarray,
    variant: str,
    *,
    true_inverse: np.ndarray | None = None,
    x0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the matching fixed-N reference and original-R true inverse."""
    matrices = np.asarray(matrices, dtype=np.complex64)
    if variant not in {"bf16", "fp32-r"}:
        raise ValueError(f"unsupported Issue #88 variant {variant!r}")
    original_x0 = initial_value(matrices) if x0 is None else x0
    reference_r = bf16_round_complex(matrices) if variant == "bf16" else matrices
    fixed_reference = newton_schulz_reference(
        reference_r,
        iterations=FIXED_ITERATIONS,
        x0=original_x0,
    )
    if true_inverse is None:
        true_inverse = np.linalg.inv(matrices.astype(np.complex128))
    matching_reference = (
        "fixed-N=12 Newton-Schulz reference using BF16-rounded R and "
        "X0=I/||original FP32 R||_infinity"
        if variant == "bf16"
        else "fixed-N=12 Newton-Schulz reference using original FP32 R and "
        "X0=I/||original FP32 R||_infinity"
    )
    return {
        "reference_r": reference_r,
        "x0": original_x0,
        "fixed_reference": fixed_reference,
        "true_inverse": true_inverse,
        "matching_reference_metric_reference": matching_reference,
        "true_inverse_metric_reference": TRUE_INVERSE_METRIC_REFERENCE,
    }


def _quality_metrics_with_reference(
    quality_metrics: dict[str, Any], metric_reference: str
) -> dict[str, Any]:
    labeled = dict(quality_metrics)
    labeled["metric_reference"] = metric_reference
    for name in ("mv_weight_direction", "beam_pattern"):
        labeled[name] = {
            **quality_metrics[name],
            "metric_reference": metric_reference,
        }
    return labeled


def correctness_metrics(
    actual: np.ndarray,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Compare a downloaded device result with both required references."""
    actual = np.asarray(actual)
    reference = context["fixed_reference"]
    true_inverse = context["true_inverse"]
    matching_metric_reference = context["matching_reference_metric_reference"]
    true_inverse_metric_reference = context["true_inverse_metric_reference"]
    finite = bool(np.all(np.isfinite(actual)))
    reference_finite = bool(np.all(np.isfinite(reference)))
    true_inverse_finite = bool(np.all(np.isfinite(true_inverse)))
    matching_error = (
        _relative_frobenius_error(actual, reference)
        if finite and reference_finite
        else float("nan")
    )
    true_error = (
        _relative_frobenius_error(actual, true_inverse)
        if finite and true_inverse_finite
        else float("nan")
    )
    metrics: dict[str, Any] = {
        "finite": finite,
        "finite_status": "pass" if finite else "failed",
        "reference_finite": reference_finite,
        "true_inverse_finite": true_inverse_finite,
        "relative_frobenius_error_vs_matching_reference": matching_error,
        "relative_frobenius_error_vs_true_inverse": true_error,
        "metric_references": {
            "relative_frobenius_error_vs_matching_reference": matching_metric_reference,
            "relative_frobenius_error_vs_true_inverse": true_inverse_metric_reference,
            "relative_error": matching_metric_reference,
            "device_test_gate": matching_metric_reference,
        },
        # This alias makes the matching-reference comparison easy to find next
        # to the equivalent field in older board records.
        "relative_error": matching_error,
        "device_test_gate": {
            "metric_reference": matching_metric_reference,
            "relative_error_max": DEVICE_TEST_RELATIVE_ERROR_GATE,
            "unchanged": True,
            "pass": bool(
                finite
                and reference_finite
                and np.isfinite(matching_error)
                and matching_error <= DEVICE_TEST_RELATIVE_ERROR_GATE
            ),
            "application": "recorded metric/gate only; existing device-test assertion is unchanged",
        },
    }
    if finite and reference_finite:
        metrics["quality_vs_matching_reference"] = _quality_metrics_with_reference(
            same_array_metrics(actual, reference), matching_metric_reference
        )
    else:
        metrics["quality_vs_matching_reference"] = None
    if finite and true_inverse_finite:
        metrics["quality_vs_true_inverse"] = _quality_metrics_with_reference(
            same_array_metrics(actual, true_inverse), true_inverse_metric_reference
        )
    else:
        metrics["quality_vs_true_inverse"] = None
    return metrics


def _sanitize_text(value: Any) -> str:
    text = str(value)
    return _ABSOLUTE_PATH_RE.sub("<redacted-path>", text)


def _sanitize_metadata(value: Any, *, key: str | None = None) -> Any:
    """Remove private metadata and host paths before JSON publication."""
    if key is not None and key.lower() in _PRIVATE_METADATA_KEYS:
        return None
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if str(child_key).lower() in _PRIVATE_METADATA_KEYS:
                continue
            result[str(child_key)] = _sanitize_metadata(child_value, key=str(child_key))
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return _ABSOLUTE_PATH_RE.sub("<redacted-path>", value)
    return value


def _validated_run_id(value: Any) -> str:
    if not isinstance(value, str) or not _RUN_ID_RE.fullmatch(value):
        raise ValueError("run_id must be a safe artifact identifier")
    return value


def _new_run_id() -> str:
    supplied = os.environ.get("HEKATUS_TT_RUN_ID")
    if supplied:
        return _validated_run_id(supplied)
    return f"local-{uuid.uuid4().hex}"


def _read_one_artifact(
    output_dir: Path,
    pattern: str,
    *,
    run_id: str,
    explicit: Path | None = None,
) -> Path:
    if explicit is not None:
        return explicit
    exact = output_dir / pattern.format(run_id=run_id)
    if exact.exists():
        return exact
    matches = sorted(output_dir.glob(pattern.format(run_id="*")))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no {pattern} telemetry artifact")
    raise RuntimeError(f"ambiguous {pattern} telemetry artifacts")


def _normalize_environment(raw: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Normalize wrapper metadata while retaining no path-bearing raw values."""
    environment = _sanitize_metadata(raw)
    if not isinstance(environment, dict):
        raise TypeError("environment artifact must contain an object")
    image = environment.get("image")
    if isinstance(image, str) and "@" in image:
        environment["image_digest"] = image.rsplit("@", 1)[1]
    if "kernel" in environment:
        environment["host_kernel"] = environment["kernel"]
    if "kmd_version" in environment:
        environment["kernel_driver_version"] = environment["kmd_version"]
    environment.setdefault("python", sys.version.split()[0])
    if environment.get("tt_env_active_release"):
        environment.setdefault("toolchain_release", environment["tt_env_active_release"])

    board = environment.get("board") or environment.get("board_info")
    if isinstance(board, dict):
        board = dict(board)
        board.setdefault("device_id", DEVICE_ID)
        serial = board.get("serial") or board.get("board_serial") or board.get("board_id")
        if serial is not None:
            board["serial"] = serial
            board.setdefault("board_id", serial)
        environment["board"] = board
        raw_board = raw.get("board") or raw.get("board_info")
        explicit_serial = isinstance(raw_board, dict) and bool(raw_board.get("serial"))
        environment["board_serial_identity"] = {
            "serial": board.get("serial"),
            "board_id": board.get("board_id"),
            "source": "explicit_serial" if explicit_serial else "board_id_alias",
        }
    firmware = environment.get("firmware") or environment.get("firmwares")
    if isinstance(firmware, dict):
        environment["firmware"] = dict(firmware)
    environment["run_id"] = run_id
    return _sanitize_metadata(environment)


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(datetime.UTC)


def _timestamp_text(value: datetime.datetime | str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        parsed = value
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{field} must include a timezone")
        return parsed.astimezone(datetime.UTC).isoformat()
    return _parse_utc_timestamp(value, field=field).isoformat()


def _power_trace_base(
    path: Path | None,
    *,
    run_id: str,
    run_start: str | None,
    run_end: str | None,
) -> dict[str, Any]:
    return {
        "file": path.name if path is not None else None,
        "run_id": run_id,
        "columns": list(POWER_TRACE_COLUMNS),
        "samples": [],
        "sampling_source": "board snapshot captured by run_in_container.sh",
        "sample_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "run_start": run_start,
        "run_end": run_end,
        "readable": False,
        "nonempty": False,
        "timestamps_parse": False,
        "timestamps_ordered": False,
        "covers_run_start": False,
        "covers_run_end": False,
        "coverage_complete": False,
        "poll_complete": False,
        "coverage": {
            "nonempty": False,
            "timestamps_parse": False,
            "timestamps_ordered": False,
            "first_at_or_before_run_start": False,
            "last_at_or_after_run_end": False,
            "readable": False,
            "complete": False,
        },
    }


def _read_power_trace(
    path: Path,
    *,
    run_id: str,
    run_start: str | None,
    run_end: str | None,
    read_text_fn: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Read one sibling CSV and retain honest coverage metadata.

    A partially written sampler file is represented as an incomplete trace
    rather than being promoted to a successful artifact.  That distinction is
    what lets the bounded poll retry while the wrapper is still appending.
    """
    trace = _power_trace_base(
        path,
        run_id=run_id,
        run_start=run_start,
        run_end=run_end,
    )
    try:
        text = path.read_text() if read_text_fn is None else read_text_fn(path)
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != POWER_TRACE_COLUMNS:
            raise ValueError(f"power trace has unexpected columns {fieldnames!r}")
        samples = [dict(row) for row in reader]
    except Exception as exc:  # noqa: BLE001 - expose unreadable traces as failures
        trace["error"] = _sanitize_text(exc)
        return trace

    trace["readable"] = True
    trace["samples"] = _sanitize_metadata(samples)
    trace["sample_count"] = len(samples)
    trace["nonempty"] = bool(samples)
    if samples:
        trace["first_timestamp"] = _sanitize_text(samples[0].get("timestamp_utc"))
        trace["last_timestamp"] = _sanitize_text(samples[-1].get("timestamp_utc"))

    parsed_timestamps: list[datetime.datetime] = []
    try:
        parsed_timestamps = [
            _parse_utc_timestamp(
                sample.get("timestamp_utc"),
                field=f"power sample {index} timestamp_utc",
            )
            for index, sample in enumerate(samples)
        ]
    except ValueError as exc:
        trace["error"] = _sanitize_text(exc)
    else:
        trace["timestamps_parse"] = True
        if parsed_timestamps:
            trace["first_timestamp"] = parsed_timestamps[0].isoformat()
            trace["last_timestamp"] = parsed_timestamps[-1].isoformat()
            trace["timestamps_ordered"] = all(
                left <= right
                for left, right in itertools.pairwise(parsed_timestamps)
            )
            trace["_last_timestamp_datetime"] = parsed_timestamps[-1]
        try:
            start = _parse_utc_timestamp(run_start, field="run_start")
            end = _parse_utc_timestamp(run_end, field="run_end")
        except ValueError as exc:
            trace["error"] = _sanitize_text(exc)
        else:
            if parsed_timestamps:
                trace["covers_run_start"] = (
                    parsed_timestamps[0] <= start <= end
                )
                trace["covers_run_end"] = parsed_timestamps[-1] >= end
                trace["coverage_complete"] = all(
                    (
                        trace["readable"],
                        trace["nonempty"],
                        trace["timestamps_parse"],
                        trace["timestamps_ordered"],
                        parsed_timestamps[0] <= start,
                        start <= end,
                        end <= parsed_timestamps[-1],
                    )
                )

    trace["coverage"] = {
        "nonempty": trace["nonempty"],
        "timestamps_parse": trace["timestamps_parse"],
        "timestamps_ordered": trace["timestamps_ordered"],
        "first_at_or_before_run_start": trace["covers_run_start"],
        "last_at_or_after_run_end": trace["covers_run_end"],
        "readable": trace["readable"],
        "complete": trace["coverage_complete"],
    }
    return trace


def _public_power_trace(trace: dict[str, Any]) -> dict[str, Any]:
    public = dict(trace)
    public.pop("_last_timestamp_datetime", None)
    return _sanitize_metadata(public)


def _wait_for_power_trace(
    output_dir: Path,
    *,
    run_id: str,
    run_end: datetime.datetime | str,
    run_start: datetime.datetime | str | None = None,
    explicit: Path | None = None,
    timeout_s: float = POWER_TRACE_POLL_TIMEOUT_S,
    interval_s: float = POWER_TRACE_POLL_INTERVAL_S,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    read_text_fn: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Poll the sibling sampler until its final sample covers ``run_end``."""
    if not math.isfinite(timeout_s) or timeout_s <= 0.0 or timeout_s > ROW_TIMEOUT_S:
        raise ValueError("power trace poll timeout must be between 0 and 60 seconds")
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("power trace poll interval must be finite and positive")
    start_text = _timestamp_text(run_start, field="run_start")
    end_text = _timestamp_text(run_end, field="run_end")
    end_value = _parse_utc_timestamp(end_text, field="run_end")
    poll_started = monotonic_fn()
    deadline = poll_started + timeout_s
    poll_count = 0
    max_polls = max(1, math.ceil(timeout_s / interval_s) + 1)
    last_trace: dict[str, Any] | None = None
    last_error = "no power trace was readable"

    while poll_count < max_polls:
        poll_count += 1
        try:
            selected = _read_one_artifact(
                output_dir,
                "power-{run_id}.csv",
                run_id=run_id,
                explicit=explicit,
            )
        except Exception as exc:  # noqa: BLE001 - sampler may not have created it yet
            last_error = _sanitize_text(exc)
        else:
            trace = _read_power_trace(
                selected,
                run_id=run_id,
                run_start=start_text,
                run_end=end_text,
                read_text_fn=read_text_fn,
            )
            trace["poll_count"] = poll_count
            last_trace = trace
            final_timestamp = trace.get("_last_timestamp_datetime")
            if isinstance(final_timestamp, datetime.datetime) and final_timestamp >= end_value:
                trace["poll_complete"] = True
                return _public_power_trace(trace)
            last_error = trace.get("error") or "power trace final sample is before run_end"

        remaining = deadline - monotonic_fn()
        if remaining <= 0.0:
            break
        sleep_fn(min(interval_s, remaining))

    if last_trace is None:
        last_trace = _power_trace_base(
            explicit,
            run_id=run_id,
            run_start=start_text,
            run_end=end_text,
        )
    last_trace["poll_count"] = poll_count
    last_trace["poll_error"] = _sanitize_text(
        f"power trace did not reach run_end within {timeout_s:g} seconds: {last_error}"
    )
    return _public_power_trace(last_trace)


def _power_trace_failure_reason(trace: Any) -> str:
    if not isinstance(trace, dict):
        return "power trace is missing"
    if not trace.get("readable"):
        return trace.get("error") or "power trace is not readable"
    if not trace.get("nonempty"):
        return "power trace contains no samples"
    if not trace.get("timestamps_parse"):
        return trace.get("error") or "power trace timestamps do not parse"
    if not trace.get("timestamps_ordered"):
        return "power trace timestamps are not ordered"
    if not trace.get("covers_run_start"):
        return "power trace starts after run_start"
    if not trace.get("covers_run_end"):
        return trace.get("poll_error") or "power trace ends before run_end"
    if not trace.get("coverage_complete"):
        return "power trace coverage is incomplete"
    return "power trace coverage is complete"


def _read_telemetry(
    output_dir: Path,
    *,
    run_id: str,
    environment_path: Path | None = None,
    power_path: Path | None = None,
    run_start: datetime.datetime | str | None = None,
    run_end: datetime.datetime | str | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    read_text_fn: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Read environment and, after the device run, a complete power trace."""
    start_text = _timestamp_text(run_start, field="run_start") if run_start is not None else None
    end_text = _timestamp_text(run_end, field="run_end") if run_end is not None else None
    telemetry: dict[str, Any] = {
        "status": "failed",
        "run_id": run_id,
        "run_start": start_text,
        "run_end": end_text,
        "environment_file": None,
        "environment": None,
        "normalized_environment": None,
        "power_trace": None,
        "failures": [],
    }
    try:
        selected_environment = _read_one_artifact(
            output_dir,
            "env-{run_id}.json",
            run_id=run_id,
            explicit=environment_path,
        )
        environment_text = (
            selected_environment.read_text()
            if read_text_fn is None
            else read_text_fn(selected_environment)
        )
        raw_environment = json.loads(environment_text)
        if not isinstance(raw_environment, dict):
            raise TypeError("environment artifact must contain an object")
        telemetry["environment_file"] = selected_environment.name
        telemetry["environment"] = _sanitize_metadata(raw_environment)
        telemetry["normalized_environment"] = _normalize_environment(raw_environment, run_id)
    except Exception as exc:  # noqa: BLE001 - publish a truthful partial artifact
        telemetry["failures"].append(
            {"stage": "telemetry.environment", "error": _sanitize_text(exc)}
        )

    try:
        if end_text is None:
            raise ValueError("run_end is required before reading power telemetry")
        power_trace = _wait_for_power_trace(
            output_dir,
            run_id=run_id,
            run_start=start_text,
            run_end=end_text,
            explicit=power_path,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            read_text_fn=read_text_fn,
        )
        telemetry["power_trace"] = power_trace
        if not power_trace.get("coverage_complete"):
            telemetry["failures"].append(
                {
                    "stage": "telemetry.power",
                    "error": _power_trace_failure_reason(power_trace),
                }
            )
    except Exception as exc:  # noqa: BLE001 - preserve environment failure separately
        telemetry["failures"].append(
            {"stage": "telemetry.power", "error": _sanitize_text(exc)}
        )
    if not telemetry["failures"]:
        telemetry["status"] = "complete"
    return telemetry


def _failure(stage: str, error: Any) -> dict[str, str]:
    return {"stage": stage, "error": _sanitize_text(error)}


def _failed_row(
    config: dict[str, Any],
    stage: str,
    error: Any,
    *,
    samples: list[float] | None = None,
    flops_per_launch: int | None = None,
) -> dict[str, Any]:
    row = dict(config)
    row.update({"status": "failed", "failure_stage": stage, "error": _sanitize_text(error)})
    if samples and flops_per_launch is not None:
        row.update(timing_summary(samples, flops_per_launch))
    return row


def _preflight_rejected_row(config: dict[str, Any]) -> dict[str, Any]:
    """Represent the expected FP32-R L=32 L1 rejection without a launch."""
    if not config.get("expected_preflight_rejection"):
        raise ValueError("row is not an expected preflight rejection")
    result = dict(config)
    result.update(
        {
            "row": config["name"],
            "shape_name": shape_name(config),
            "status": "preflight_rejected",
            "failure_stage": None,
            "launches_requested": 0,
            "launches_measured": 0,
            "preflight": {
                "status": config["preflight_status"],
                "bytes": config["preflight_bytes"],
                "over_budget_bytes": config["preflight_over_budget_bytes"],
                "variant": config["variant"],
                "size": config["size"],
                "matrix_block": config["matrix_block"],
                "r_memory": config["r_memory"],
                "fallback": False,
                "allocation_attempted": False,
                "launches": 0,
            },
        }
    )
    return result


def _run_row(
    ttnn: Any,
    device: Any,
    matrices: np.ndarray,
    config: dict[str, Any],
    *,
    reference: dict[str, np.ndarray] | None = None,
    launches: int = LAUNCHES_PER_ROW,
    timeout_s: float = ROW_TIMEOUT_S,
    first_launch: bool = False,
    kernel_class: Any | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run correctness, warm-up, and timed launches for one exact row."""
    if launches != LAUNCHES_PER_ROW and not first_launch:
        raise ValueError(f"Issue #88 requires exactly {LAUNCHES_PER_ROW} timed launches")
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    flops_per_launch = 8192 * 4 * 2 * config["size"] ** 3 * (2 * FIXED_ITERATIONS)
    result: dict[str, Any] = dict(config)
    result["row"] = config["name"]
    result["shape_name"] = shape_name(config)
    result["flops_per_launch"] = flops_per_launch
    result["launches_requested"] = 0 if first_launch else launches
    result["row_timeout_s"] = timeout_s
    result["status"] = "failed"
    kernel = None
    samples: list[float] = []
    row_started = time_fn()
    try:
        if reference is None:
            reference = reference_context(matrices, config["variant"])
        if time_fn() - row_started > timeout_s:
            result = _failed_row(config, "row_timeout", "row exceeded 60 seconds before device launch")
            result.update(
                {
                    "row": config["name"],
                    "shape_name": shape_name(config),
                    "flops_per_launch": flops_per_launch,
                    "launches_requested": 0 if first_launch else launches,
                    "row_timeout_s": timeout_s,
                }
            )
            return result

        if kernel_class is None:
            from enodia.tt.bench.newton_schulz_kernel import NewtonSchulzKernel

            kernel_class = NewtonSchulzKernel
        kernel = kernel_class.prepare(
            ttnn,
            device,
            matrices,
            variant=config["variant"],
            math_fidelity=config["math_fidelity"],
            fuse_s=config["fuse_s"],
            batch_reads=False,
            matrix_block=config["matrix_block"],
            double_buffer=config["double_buffer"],
            input_memory=config["input_memory"],
            r_memory=config["r_memory"],
            x0_memory=config["x0_memory"],
            output_memory=config["output_memory"],
            fp32_dest_acc_en=config["fp32_dest_acc_en"],
            dst_full_sync_en=config["dst_full_sync_en"],
            iterations=FIXED_ITERATIONS,
        )

        kernel.launch()
        ttnn.synchronize_device(device)
        actual = kernel.result()
        correctness = correctness_metrics(actual, reference)
        result["correctness"] = correctness
        result["correctness_launch"] = {
            "synchronized": True,
            "downloaded": True,
            "finite": correctness["finite"],
        }
        if not correctness["finite"]:
            result.update(
                _failed_row(config, "correctness", "correctness launch returned non-finite values")
            )
            result["correctness"] = correctness
            result["correctness_launch"] = {
                "synchronized": True,
                "downloaded": True,
                "finite": False,
            }
            return result

        if first_launch:
            result["status"] = "correctness_only"
            result["launches_measured"] = 0
            result["timing"] = "not run (--first-launch)"
            return result

        # Correctness is not counted as the warm-up.  This makes the timing
        # protocol explicit: one additional synchronized warm-up, then exactly
        # 1,000 synchronized samples.
        kernel.launch()
        ttnn.synchronize_device(device)
        result["warmup"] = {"launches": 1, "synchronized": True}

        if time_fn() - row_started > timeout_s:
            result.update(
                _failed_row(config, "row_timeout", "row exceeded 60 seconds before timed launches")
            )
            result["correctness"] = correctness
            result["warmup"] = {"launches": 1, "synchronized": True}
            return result

        for _ in range(launches):
            launch_started = time_fn()
            kernel.launch()
            ttnn.synchronize_device(device)
            samples.append(time_fn() - launch_started)
            if time_fn() - row_started > timeout_s:
                result.update(
                    _failed_row(
                        config,
                        "row_timeout",
                        "row exceeded 60 seconds; timed launches stopped",
                        samples=samples,
                        flops_per_launch=flops_per_launch,
                    )
                )
                result["correctness"] = correctness
                result["warmup"] = {"launches": 1, "synchronized": True}
                result["stop_condition"] = "row_timeout_s_exceeded"
                return result

        result["status"] = "ok"
        result.update(timing_summary(samples, flops_per_launch))
        result["warmup"] = {"launches": 1, "synchronized": True}
        result["timing_protocol"] = (
            "one synchronized warm-up followed by exactly 1,000 synchronized launches"
        )
        return result
    except Exception as exc:  # noqa: BLE001 - preserve row-local device failures
        result = _failed_row(
            config,
            "row",
            exc,
            samples=samples,
            flops_per_launch=flops_per_launch,
        )
        result["row"] = config["name"]
        result["shape_name"] = shape_name(config)
        result["flops_per_launch"] = flops_per_launch
        result["launches_requested"] = 0 if first_launch else launches
        result["row_timeout_s"] = timeout_s
        if reference is not None:
            result["reference_available"] = True
        if samples:
            result["stop_condition"] = "row_error"
        return result
    finally:
        if kernel is not None:
            try:
                kernel.close()
            except Exception as exc:  # noqa: BLE001 - cleanup is part of the row result
                if result.get("status") == "ok":
                    result["status"] = "failed"
                    result["failure_stage"] = "row_cleanup"
                    result["error"] = _sanitize_text(exc)
                else:
                    result.setdefault("cleanup_error", _sanitize_text(exc))


def _prepare_comparison_input(
    config: dict[str, Any],
    *,
    matrices_by_size: dict[int, np.ndarray],
    x0_by_size: dict[int, np.ndarray],
    true_inverse_by_size: dict[int, np.ndarray],
    fingerprints: dict[str, dict[str, Any]],
    matrices_factory: Callable[..., np.ndarray],
) -> tuple[int, np.ndarray, dict[str, Any]]:
    """Prepare one cached HPD batch and its variant-specific reference."""
    size = int(config["size"])
    if size not in matrices_by_size:
        matrices_by_size[size] = matrices_factory(
            config["batch"],
            size,
            condition_number=CONDITION_NUMBER,
            seed=INPUT_SEED,
        )
        x0_by_size[size] = initial_value(matrices_by_size[size])
        true_inverse_by_size[size] = np.linalg.inv(
            matrices_by_size[size].astype(np.complex128)
        )
        fingerprints[str(size)] = {
            "size": size,
            "batch": config["batch"],
            "condition_number": CONDITION_NUMBER,
            "seed": INPUT_SEED,
            "matrix_sha256": _array_sha256(matrices_by_size[size]),
        }
    matrices = matrices_by_size[size]
    reference = reference_context(
        matrices,
        config["variant"],
        true_inverse=true_inverse_by_size[size],
        x0=x0_by_size[size],
    )
    return size, matrices, reference


def run_comparison(
    ttnn: Any,
    device: Any,
    *,
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    launches: int = LAUNCHES_PER_ROW,
    timeout_s: float = ROW_TIMEOUT_S,
    first_launch: bool = False,
    kernel_class: Any | None = None,
    time_fn: Callable[[], float] = time.perf_counter,
    matrices_factory: Callable[..., np.ndarray] = random_hpd_batch,
) -> dict[str, Any]:
    """Run selected rows on one already-open device and stop on first failure."""
    selected = tuple(ISSUE88_COMPARISON_ROWS if rows is None else rows)
    if not selected:
        raise ValueError("at least one Issue #88 row is required")
    validate_comparison_rows(selected)
    if not first_launch:
        if launches != LAUNCHES_PER_ROW:
            raise ValueError(f"Issue #88 requires exactly {LAUNCHES_PER_ROW} timed launches")
    elif launches < 1:
        raise ValueError("first-launch mode requires a positive launch setting")

    remaining = collections.Counter(row["size"] for row in selected)
    matrices_by_size: dict[int, np.ndarray] = {}
    x0_by_size: dict[int, np.ndarray] = {}
    true_inverse_by_size: dict[int, np.ndarray] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    comparison: list[dict[str, Any]] = []
    stopped = False
    for config in selected:
        if config.get("expected_preflight_rejection"):
            comparison.append(_preflight_rejected_row(config))
            continue
        try:
            size, matrices, reference = _prepare_comparison_input(
                config,
                matrices_by_size=matrices_by_size,
                x0_by_size=x0_by_size,
                true_inverse_by_size=true_inverse_by_size,
                fingerprints=fingerprints,
                matrices_factory=matrices_factory,
            )
        except Exception as exc:  # noqa: BLE001 - retain a host-preparation failure as a row
            row = _failed_row(config, "host_preparation", exc)
            row["row"] = config["name"]
            row["shape_name"] = shape_name(config)
            comparison.append(row)
            stopped = True
            break
        row = _run_row(
            ttnn,
            device,
            matrices,
            config,
            reference=reference,
            launches=launches,
            timeout_s=timeout_s,
            first_launch=first_launch,
            kernel_class=kernel_class,
            time_fn=time_fn,
        )
        row["input_fingerprint"] = dict(fingerprints[str(size)])
        row["reference_fingerprint"] = {
            "reference_r_sha256": _array_sha256(reference["reference_r"]),
            "x0_sha256": _array_sha256(reference["x0"]),
            "fixed_reference_sha256": _array_sha256(reference["fixed_reference"]),
            "true_inverse_sha256": _array_sha256(reference["true_inverse"]),
        }
        comparison.append(row)
        remaining[size] -= 1
        if remaining[size] == 0:
            matrices_by_size.pop(size, None)
            x0_by_size.pop(size, None)
            true_inverse_by_size.pop(size, None)
        if row.get("status") not in SUCCESSFUL_ROW_STATUSES:
            stopped = True
            break
        if first_launch:
            break

    passed = bool(comparison) and len(comparison) == len(selected) and all(
        row.get("status") in SUCCESSFUL_ROW_STATUSES for row in comparison
    )
    return {
        "status": "pass" if passed and not stopped else "failed",
        "comparison_rows": comparison,
        "rows_completed": len(comparison),
        "rows_requested": len(selected),
        "stopped_on_failure": stopped,
        "first_launch_only": first_launch,
        "input_generation": {
            "generator": "enodia.tt.bench.newton_schulz_reference.random_hpd_batch",
            "condition_number": CONDITION_NUMBER,
            "seed": INPUT_SEED,
            "same_matrix_for_matching_size_variants": True,
            "fingerprints": fingerprints,
            "fp32_r_l32_default_l1_preflight": {
                "status": FP32_R_L32_L1_PREFLIGHT_STATUS,
                "bytes": FP32_R_L32_L1_PREFLIGHT_BYTES,
                "over_budget_bytes": FP32_R_L32_L1_PREFLIGHT_OVERAGE_BYTES,
                "variant": "fp32-r",
                "size": 32,
                "matrix_block": 8,
                "r_memory": "l1",
                "reason": "selected block-8 L1 budget rejects before allocation; no fallback",
            },
        },
    }


# Name the session driver explicitly for callers that mirror the Issue #101 API.
run_issue88_comparison = run_comparison


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(strict_json_dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _raw_path(result_path: Path, run_id: str) -> Path:
    return result_path.parent / f"issue88-fp32-r-raw-{run_id}.json"


def _status_components(
    run: dict[str, Any],
    *,
    telemetry: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Evaluate every publication gate without deriving missing provenance."""
    rows = run.get("comparison_rows")
    requested = run.get("rows_requested")
    successful_statuses = SUCCESSFUL_ROW_STATUSES
    row_failures: list[str] = []
    if not isinstance(rows, list) or not rows:
        row_failures.append("no selected rows completed")
    else:
        if run.get("status") != "pass":
            row_failures.append(f"run status is {run.get('status')!r}")
        if not isinstance(requested, int) or len(rows) != requested:
            row_failures.append(
                f"completed {len(rows)} of {requested!r} selected rows"
            )
        for row in rows:
            if row.get("status") not in successful_statuses:
                row_failures.append(
                    f"{row.get('row', row.get('name', '<unknown>'))} status is "
                    f"{row.get('status')!r}: {row.get('error', 'row did not succeed')}"
                )
        if run.get("stopped_on_failure"):
            row_failures.append("run stopped on a row failure")
    row_ok = not row_failures

    environment = telemetry.get("normalized_environment")
    board_failures: list[str] = []
    board = environment.get("board") if isinstance(environment, dict) else None
    if not isinstance(board, dict):
        board_failures.append("normalized telemetry board is missing")
    else:
        board_type = board.get("board_type")
        if not isinstance(board_type, str) or not board_type.strip():
            board_failures.append("board_type is missing")
        elif board_type.lower() != "p150a":
            board_failures.append(f"board_type is not p150a: {board_type!r}")
        serial = board.get("serial")
        board_id = board.get("board_id")
        if not isinstance(serial, str) or not serial.strip():
            board_failures.append("board serial is missing")
        if not isinstance(board_id, str) or not board_id.strip():
            board_failures.append("board_id is missing")
        pci = next(
            (
                board.get(key)
                for key in ("bus_id", "pci_bus_id", "pci_address", "pci_bdf")
                if board.get(key) is not None
            ),
            None,
        )
        if not isinstance(pci, str) or _PCI_BUS_ID_RE.fullmatch(pci.strip()) is None:
            board_failures.append("verifiable PCI identity is missing")
        serial_identity = (
            environment.get("board_serial_identity")
            if isinstance(environment, dict)
            else None
        )
        if not isinstance(serial_identity, dict):
            board_failures.append("board serial identity provenance is missing")
        elif serial_identity.get("serial") != serial or serial_identity.get("board_id") != board_id:
            board_failures.append("board serial identity does not match board metadata")
        firmware = environment.get("firmware") if isinstance(environment, dict) else None
        if not isinstance(firmware, dict) or not isinstance(
            firmware.get("fw_bundle_version"), str
        ) or not firmware.get("fw_bundle_version", "").strip():
            board_failures.append("firmware bundle version is missing")

    image_failures: list[str] = []
    if not isinstance(environment, dict):
        image_failures.append("normalized environment is missing")
    else:
        image = environment.get("image")
        if not isinstance(image, str) or not image.strip():
            image_failures.append("image is missing")
        if environment.get("image_pinned") is not True:
            image_failures.append("image_pinned is not true")
        digest = environment.get("image_digest")
        if not isinstance(digest, str) or _IMAGE_DIGEST_RE.fullmatch(digest.strip()) is None:
            image_failures.append("image_digest is not a valid sha256 digest")
        for field in ("kernel_driver_version", "toolchain_release"):
            if not isinstance(environment.get(field), str) or not environment[field].strip():
                image_failures.append(f"{field} is missing")

    harness_failures: list[str] = []
    if not isinstance(environment, dict):
        harness_failures.append("normalized environment is missing")
    else:
        commit = environment.get("harness_commit")
        if not isinstance(commit, str) or _HARNESS_COMMIT_RE.fullmatch(commit.strip()) is None:
            harness_failures.append("harness_commit is not a full hexadecimal commit")
        if environment.get("harness_dirty") is not False:
            harness_failures.append("harness_dirty is not false")

    power_trace = telemetry.get("power_trace")
    power_failures: list[str] = []
    if not isinstance(power_trace, dict):
        power_failures.append("power trace is missing")
    else:
        if not isinstance(power_trace.get("samples"), list) or not power_trace["samples"]:
            power_failures.append("power trace contains no samples")
        if power_trace.get("sample_count") != len(power_trace.get("samples", [])):
            power_failures.append("power trace sample count is inconsistent")
        for field in ("readable", "timestamps_parse", "timestamps_ordered", "poll_complete"):
            if power_trace.get(field) is not True:
                power_failures.append(f"power trace {field} is not true")
        try:
            first = _parse_utc_timestamp(
                power_trace.get("first_timestamp"), field="power first_timestamp"
            )
            start = _parse_utc_timestamp(power_trace.get("run_start"), field="power run_start")
            end = _parse_utc_timestamp(power_trace.get("run_end"), field="power run_end")
            last = _parse_utc_timestamp(
                power_trace.get("last_timestamp"), field="power last_timestamp"
            )
        except ValueError as exc:
            power_failures.append(_sanitize_text(exc))
        else:
            if not first <= start <= end <= last:
                power_failures.append(
                    "power timestamps do not satisfy first <= run_start <= run_end <= last"
                )
        if power_trace.get("coverage_complete") is not True:
            power_failures.append("power trace coverage_complete is not true")
    power_ok = not power_failures
    power_reason = (
        "power trace coverage is complete"
        if power_ok
        else "; ".join(power_failures)
    )
    close_ok = cleanup.get("close_succeeded") is True
    close_reason = (
        "device close returned normally"
        if close_ok
        else "device close was not completed successfully"
    )
    checks = {
        "rows": (
            row_ok,
            "all selected rows succeeded" if row_ok else "; ".join(row_failures),
        ),
        "board_selection": (
            not board_failures,
            "board selection and firmware are verifiable"
            if not board_failures
            else "; ".join(board_failures),
        ),
        "image_toolchain": (
            not image_failures,
            "digest-pinned image and toolchain metadata are present"
            if not image_failures
            else "; ".join(image_failures),
        ),
        "harness": (
            not harness_failures,
            "harness commit is valid and the tree is clean"
            if not harness_failures
            else "; ".join(harness_failures),
        ),
        "power_trace": (power_ok, power_reason),
        "device_close": (close_ok, close_reason),
    }
    return {
        name: {"ok": bool(checks[name][0]), "reason": _sanitize_text(checks[name][1])}
        for name, _description in ISSUE88_STATUS_COMPONENTS
    }


def _status_components_pass(status_components: dict[str, dict[str, Any]]) -> bool:
    return all(
        status_components.get(name, {}).get("ok") is True
        for name, _description in ISSUE88_STATUS_COMPONENTS
    )


def _record_payload(
    run: dict[str, Any],
    *,
    telemetry: dict[str, Any],
    run_id: str,
    raw_path: Path,
    cleanup: dict[str, Any],
    status_components: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    environment = telemetry.get("normalized_environment")
    power_trace = telemetry.get("power_trace") or {}
    rows = run.get("comparison_rows", [])
    if status_components is None:
        status_components = _status_components(run, telemetry=telemetry, cleanup=cleanup)
    overall_pass = _status_components_pass(status_components)
    timed_out = any(row.get("failure_stage") == "row_timeout" for row in rows)
    failure = None
    if run.get("failure") is not None:
        failure = run["failure"]
    elif telemetry.get("failures"):
        failure = telemetry["failures"][0]
    elif not overall_pass:
        failed_component = next(
            (
                (name, component)
                for name, component in status_components.items()
                if not component.get("ok")
            ),
            ("unknown", {"reason": "record publication gate failed"}),
        )
        failure = {
            "stage": "status_component",
            "component": failed_component[0],
            "error": failed_component[1].get("reason", "component failed"),
        }
    record: dict[str, Any] = {
        "record_schema": ISSUE88_RECORD_SCHEMA,
        "status": "pass" if overall_pass else "failed",
        "status_components": status_components,
        "issue": "#88",
        "adr": "ADR-0005",
        "captured_at": (environment or {}).get(
            "captured_at", datetime.datetime.now(datetime.UTC).isoformat()
        ),
        "run_id": run_id,
        "harness_commit": (environment or {}).get("harness_commit"),
        "environment": environment,
        "algorithm": {
            "update": "X[k+1] = X[k] @ (2I - R @ X[k])",
            "iterations": FIXED_ITERATIONS,
            "initial_value": "I / ||R||_infinity",
            "state_format": "BF16",
            "destination_format": "FP32",
            "math_fidelity": "HiFi3",
            "fuse_s": True,
            "matrix_block": 8,
            "double_buffer": True,
            "dst_full_sync_en": True,
            "output_memory": "dram",
        },
        "input_generation": run.get("input_generation"),
        "measurement": {
            "description": (
                "Issue #88 BF16-R versus FP32-R fixed-N=12 comparison with one "
                "correctness launch and 1,000 synchronized launches per row."
            ),
            "device_id": DEVICE_ID,
            "same_device_session": True,
            "same_python_process": True,
            "watcher": bool(os.environ.get("TT_METAL_WATCHER")),
            "launches_per_row": LAUNCHES_PER_ROW,
            "first_launch_only": bool(run.get("first_launch_only")),
            "row_timeout_s": ROW_TIMEOUT_S,
            "correctness_reference": (
                "enodia.tt.bench.newton_schulz_reference.newton_schulz_reference; fixed N=12"
            ),
            "true_inverse": TRUE_INVERSE_METRIC_REFERENCE,
            "comparison_conclusion_reference_policy": (
                "Each conclusion uses one declared metric_reference; metrics with "
                "different references are reported in separate conclusions."
            ),
            "correctness_device_test_gate": {
                "relative_error_max": DEVICE_TEST_RELATIVE_ERROR_GATE,
                "unchanged": True,
                "application": "recorded metric/gate only",
            },
            "comparison_rows": rows,
            "power_trace": power_trace.get("file"),
            "power_clock_provenance": {
                "trace": power_trace.get("file"),
                "columns": power_trace.get("columns"),
                "samples": power_trace.get("sample_count", len(power_trace.get("samples", []))),
                "sample_count": power_trace.get(
                    "sample_count", len(power_trace.get("samples", []))
                ),
                "first_timestamp": power_trace.get("first_timestamp"),
                "last_timestamp": power_trace.get("last_timestamp"),
                "run_start": power_trace.get("run_start"),
                "run_end": power_trace.get("run_end"),
                "coverage": power_trace.get("coverage"),
                "coverage_complete": power_trace.get("coverage_complete", False),
                "sampling_source": power_trace.get("sampling_source"),
            },
            "commands": {
                "wrapper": "enodia/tt/bench/run_in_container.sh",
                "runner": ISSUE88_RUNNER,
                "device_id": DEVICE_ID,
            },
        },
        "metric_definitions": {
            "quality_metric_reference_policy": (
                "Each quality metric group and its MV/beam submetrics carry a "
                "metric_reference; conclusions must not combine distinct references."
            ),
            "relative_frobenius_error_vs_matching_reference": (
                "||X_device - X_reference||_F / ||X_reference||_F"
            ),
            "relative_frobenius_error_vs_true_inverse": (
                "||X_device - R_original^-1||_F / ||R_original^-1||_F"
            ),
            "mv_weight_direction": (
                "1 - |w_true^H w_device| / (||w_true||_2 ||w_device||_2), "
                "w = R^-1 a / (a^H R^-1 a)"
            ),
            "beam_pattern": (
                "relative Frobenius error between same-array w^H a(theta) response "
                "arrays over look and pattern directions"
            ),
            "steering_array": {
                "array": "equal-spaced linear array",
                "element_spacing": "one-half wavelength",
                "look_directions_deg": list(LOOK_DIRECTIONS_DEG),
                "pattern_directions_deg": list(PATTERN_DIRECTIONS_DEG),
            },
        },
        "rows": rows,
        "results": rows,
        "power_trace": power_trace.get("file"),
        "raw_artifact": {
            "schema": ISSUE88_RAW_SCHEMA,
            "file": raw_path.name,
            "run_id": run_id,
            "external_temporary": True,
            "not_committed": True,
        },
        "run_protocol": {
            "timeout": timed_out,
            "abnormal_exit": not overall_pass,
            "reset_performed": False,
            "cleanup": cleanup,
            "partial_rows_retained": True,
        },
        "notes": [
            "BF16-R reference uses BF16-rounded R and X0 computed from original R.",
            "FP32-R reference uses original FP32 R and the same X0 computed from original R.",
            "FP32-R L=32 keeps block 8 and moves only R to explicit DRAM placement.",
            "Raw launch samples, fingerprints, correctness metrics, and sanitized telemetry are retained.",
            "No host name or user-specific absolute path is included in this record.",
        ],
    }
    if failure is not None:
        record["failure"] = failure
    return _sanitize_metadata(record)


def _raw_payload(
    run: dict[str, Any],
    *,
    telemetry: dict[str, Any],
    run_id: str,
    raw_path: Path,
    cleanup: dict[str, Any],
    status_components: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture all result-builder inputs while retaining only safe metadata."""
    if status_components is None:
        status_components = _status_components(run, telemetry=telemetry, cleanup=cleanup)
    record_status = "pass" if _status_components_pass(status_components) else "failed"
    payload = {
        "raw_schema": ISSUE88_RAW_SCHEMA,
        "captured_at": _utc_now().isoformat(),
        "run_id": run_id,
        "artifact_file": raw_path.name,
        "artifact_status": record_status,
        "status_components": status_components,
        "run": run,
        "comparison_results": run.get("comparison_rows", []),
        "rows": run.get("comparison_rows", []),
        "results": run.get("comparison_rows", []),
        "configs": {
            "rows": [dict(row) for row in ISSUE88_COMPARISON_ROWS],
            "launches_per_row": LAUNCHES_PER_ROW,
            "fp32_r_l32_default_l1_preflight": {
                "status": FP32_R_L32_L1_PREFLIGHT_STATUS,
                "bytes": FP32_R_L32_L1_PREFLIGHT_BYTES,
                "over_budget_bytes": FP32_R_L32_L1_PREFLIGHT_OVERAGE_BYTES,
                "matrix_block": 8,
                "r_memory": "l1",
            },
            "row_timeout_s": ROW_TIMEOUT_S,
            "iterations": FIXED_ITERATIONS,
            "condition_number": CONDITION_NUMBER,
            "seed": INPUT_SEED,
        },
        "telemetry": telemetry,
        "cleanup": cleanup,
    }
    if run.get("failure") is not None:
        payload["failure"] = run["failure"]
    elif record_status != "pass":
        failed_component = next(
            (
                (name, component)
                for name, component in status_components.items()
                if not component.get("ok")
            ),
            ("unknown", {"reason": "record publication gate failed"}),
        )
        payload["failure"] = {
            "stage": "status_component",
            "component": failed_component[0],
            "error": failed_component[1].get("reason", "component failed"),
        }
    return _sanitize_metadata(payload)


def _mark_failed(run: dict[str, Any], stage: str, error: Any) -> dict[str, Any]:
    updated = dict(run)
    details = _failure(stage, error)
    existing = updated.get("failure")
    if existing is not None:
        secondary = list(updated.get("secondary_failures", []))
        secondary.append(details)
        updated["secondary_failures"] = secondary
    else:
        updated["failure"] = details
        updated["failure_stage"] = stage
        updated["error"] = details["error"]
    updated["status"] = "failed"
    return updated


def _select_rows(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    rows = list(ISSUE88_COMPARISON_ROWS)
    if args.only:
        exact_selectors = {
            selector
            for selector in args.only
            if any(row["name"] == selector for row in rows)
        }
        rows = [
            row
            for row in rows
            if any(
                (
                    selector in exact_selectors
                    and row["name"] == selector
                )
                or (
                    selector not in exact_selectors
                    and (selector in row["name"] or selector in shape_name(row))
                )
                for selector in args.only
            )
        ]
    if args.custom_variant is not None:
        rows = [row for row in rows if row["variant"] == args.custom_variant]
    if args.row:
        requested = set(args.row)
        rows = [row for row in rows if row["name"] in requested]
    if not rows:
        raise ValueError("no Issue #88 comparison row matches the requested selection")
    return tuple(rows[:1] if args.first_launch else rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-launch", action="store_true")
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--row", action="append", default=None)
    parser.add_argument("--custom-variant", choices=("bf16", "fp32-r"), default=None)
    parser.add_argument("--custom-math-fidelity", default="HiFi3")
    parser.add_argument("--matrix-block", type=int, default=8)
    parser.add_argument("--r-memory", choices=("l1", "dram"), default=None)
    parser.add_argument("--x0-memory", choices=("l1", "dram"), default=None)
    parser.add_argument("--output-memory", choices=("l1", "dram"), default="dram")
    parser.add_argument("--repeats", type=int, default=LAUNCHES_PER_ROW)
    # These compatibility options let a plan row be pasted into the custom
    # runner without making the dedicated runner depend on run_matmul.
    parser.add_argument("--kind", action="append", default=None)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--input-memory", choices=("l1", "dram"), default="l1")
    parser.add_argument("--fuse-s", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--double-buffer", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dst-full-sync", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fp32-dest-acc", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--env-json", type=Path, default=None)
    parser.add_argument("--power-trace", type=Path, default=None)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.device_id != DEVICE_ID:
        parser.error("Issue #88 must run on visible device_id=0 inside the container")
    if args.kind and any(kind != "custom_newton_schulz" for kind in args.kind):
        parser.error("the Issue #88 runner accepts only custom_newton_schulz")
    if args.custom_math_fidelity != "HiFi3":
        parser.error("Issue #88 requires HiFi3")
    if args.matrix_block != 8:
        parser.error("Issue #88 requires matrix_block=8; no block fallback is allowed")
    if args.input_memory != "l1":
        parser.error("Issue #88 requires compute and identity inputs in L1")
    if args.output_memory != "dram":
        parser.error("Issue #88 requires output_memory=dram")
    if not args.fuse_s:
        parser.error("Issue #88 requires fused S")
    if not args.double_buffer:
        parser.error("Issue #88 requires double_buffer=true")
    if not args.dst_full_sync:
        parser.error("Issue #88 requires full DEST synchronization")
    if not args.fp32_dest_acc:
        parser.error("Issue #88 requires FP32 DEST accumulation")
    if args.iters != 1:
        parser.error("the custom runner counts one fixed-N=12 kernel launch per sample")
    if args.first_launch:
        if args.repeats < 1:
            parser.error("--first-launch requires a positive launch setting")
    elif args.repeats != LAUNCHES_PER_ROW:
        parser.error(f"Issue #88 requires exactly {LAUNCHES_PER_ROW} timed launches")
    selected = _select_rows(args)
    if args.r_memory is not None and any(row["r_memory"] != args.r_memory for row in selected):
        parser.error("--r-memory changes a requested row placement; no fallback is allowed")
    if args.x0_memory is not None and any(row["x0_memory"] != args.x0_memory for row in selected):
        parser.error("--x0-memory changes a requested row placement")


def main(
    argv: list[str] | None = None,
    *,
    now_fn: Callable[[], datetime.datetime] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    read_text_fn: Callable[[Path], str] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    selected_rows = _select_rows(args)
    result_path = Path(
        os.environ.get("HEKATUS_TT_RESULT_PATH")
        or args.out
        or "/out/issue88-fp32-r.json"
    )
    run_id = _new_run_id()
    output_dir = result_path.parent
    now = _utc_now if now_fn is None else now_fn
    monotonic = time.monotonic if monotonic_fn is None else monotonic_fn
    sleep = time.sleep if sleep_fn is None else sleep_fn
    run_start = now()
    run: dict[str, Any] = {
        "status": "failed",
        "comparison_rows": [],
        "rows_completed": 0,
        "rows_requested": len(selected_rows),
        "stopped_on_failure": False,
        "first_launch_only": args.first_launch,
        "input_generation": {
            "generator": "enodia.tt.bench.newton_schulz_reference.random_hpd_batch",
            "condition_number": CONDITION_NUMBER,
            "seed": INPUT_SEED,
        },
    }
    device = None
    device_opened = False
    close_error: Exception | None = None
    try:
        try:
            import ttnn
        except Exception as exc:  # noqa: BLE001 - publish an open-stage record
            run = _mark_failed(run, "open", exc)
        else:
            try:
                device = ttnn.open_device(device_id=DEVICE_ID)
                device_opened = True
            except Exception as exc:  # noqa: BLE001 - publish an open-stage record
                run = _mark_failed(run, "open", exc)
            else:
                try:
                    run = run_comparison(
                        ttnn, device, rows=selected_rows, first_launch=args.first_launch
                    )
                except Exception as exc:  # noqa: BLE001 - preserve device-session failures
                    partial = getattr(exc, "partial_rows", None)
                    if isinstance(partial, list):
                        run["comparison_rows"] = list(partial)
                        run["rows_completed"] = len(partial)
                    run = _mark_failed(run, "device_session", exc)
    finally:
        cleanup = {
            "device_opened": device_opened,
            "close_attempted": device_opened,
            "close_succeeded": False,
        }
        if device_opened:
            cleanup["close_attempted"] = True
            try:
                import ttnn

                ttnn.close_device(device)
                cleanup["close_succeeded"] = True
            except Exception as exc:  # noqa: BLE001 - retain rows before returning failure
                close_error = exc
                run = _mark_failed(run, "close", exc)
                cleanup["close_succeeded"] = False
                cleanup["close_error"] = _sanitize_text(exc)
        run["cleanup"] = cleanup

    run_end = now()
    run["run_start"] = _timestamp_text(run_start, field="run_start")
    run["run_end"] = _timestamp_text(run_end, field="run_end")
    if close_error is not None:
        # The failure is represented in JSON; do not discard the completed rows.
        run["stopped_on_failure"] = True

    # The sampler is still alive at this point.  Reading power telemetry here,
    # after every row and device close operation, lets the poll wait for a
    # final sample at or beyond run_end instead of freezing a partial trace.
    telemetry = _read_telemetry(
        output_dir,
        run_id=run_id,
        environment_path=args.env_json,
        power_path=args.power_trace,
        run_start=run_start,
        run_end=run_end,
        monotonic_fn=monotonic,
        sleep_fn=sleep,
        read_text_fn=read_text_fn,
    )
    status_components = _status_components(run, telemetry=telemetry, cleanup=run["cleanup"])
    raw_path = _raw_path(result_path, run_id)
    raw = _raw_payload(
        run,
        telemetry=telemetry,
        run_id=run_id,
        raw_path=raw_path,
        cleanup=run["cleanup"],
        status_components=status_components,
    )
    record = _record_payload(
        run,
        telemetry=telemetry,
        run_id=run_id,
        raw_path=raw_path,
        cleanup=run["cleanup"],
        status_components=status_components,
    )
    _atomic_json_write(raw_path, raw)
    _atomic_json_write(result_path, record)
    print(f"issue88 result -> {result_path}", flush=True)
    print(f"issue88 raw -> {raw_path.name}", flush=True)
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
