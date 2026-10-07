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
FP32_R_L32_L1_PREFLIGHT_STATUS = "rejected_before_allocation"

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
    """Return the four requested, placement-explicit comparison rows."""
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
    )
    return tuple(dict(row) for row in rows)


ISSUE88_COMPARISON_ROWS = comparison_rows()


def validate_comparison_rows(rows: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None) -> None:
    """Validate the non-negotiable Issue #88 execution catalogue."""
    selected = ISSUE88_COMPARISON_ROWS if rows is None else tuple(rows)
    if len(selected) != 4:
        raise ValueError(f"Issue #88 requires four comparison rows, got {len(selected)}")
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
    names = {row.get("name") for row in selected}
    if names != {row["name"] for row in ISSUE88_COMPARISON_ROWS}:
        raise ValueError("Issue #88 comparison row names do not match the required catalogue")
    canonical = {row["name"]: row for row in ISSUE88_COMPARISON_ROWS}
    for row in selected:
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"{row.get('name', '<unnamed>')} changes required {key}: "
                    f"expected {value!r}, got {row.get(key)!r}"
                )
        for key in ("variant", "r_format", "size", "packing", "r_memory"):
            if row.get(key) != canonical[row["name"]][key]:
                raise ValueError(
                    f"{row['name']} changes required {key}: "
                    f"expected {canonical[row['name']][key]!r}, got {row.get(key)!r}"
                )
    placements = {
        (row["variant"], row["size"]): row["r_memory"] for row in selected
    }
    if placements != {
        ("bf16", 16): "l1",
        ("fp32-r", 16): "l1",
        ("bf16", 32): "l1",
        ("fp32-r", 32): "dram",
    }:
        raise ValueError("Issue #88 R placements do not match the required catalogue")


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
    """Form normalized MV weights for a batch of inverse matrices."""
    weights = np.einsum("bij,j->bi", inverse, steering, optimize=True)
    denominator = np.einsum("j,bi->b", steering.conj(), weights, optimize=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return weights / denominator[:, None]


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def same_array_metrics(
    candidate_inverse: np.ndarray,
    true_inverse: np.ndarray,
    *,
    look_directions_deg: tuple[float, ...] = LOOK_DIRECTIONS_DEG,
    pattern_directions_deg: tuple[float, ...] = PATTERN_DIRECTIONS_DEG,
) -> dict[str, Any]:
    """Compute MV-direction and same-array pattern errors without spec imports.

    A leading batch dimension is accepted so the aggregate values describe all
    8,192 matrices in a board row.  Per-look-direction values are conservative
    maxima across the batch, with means retained alongside them for audit.
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

    deficits: list[np.ndarray] = []
    true_responses: list[np.ndarray] = []
    candidate_responses: list[np.ndarray] = []
    response_errors: list[np.ndarray] = []
    for direction in look_directions_deg:
        steering = steering_vector(aperture_size, direction, dtype=np.complex128)
        true_weight = _weight(reference.astype(np.complex128, copy=False), steering)
        candidate_weight = _weight(candidate.astype(np.complex128, copy=False), steering)
        denominators = np.linalg.norm(true_weight, axis=1) * np.linalg.norm(candidate_weight, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = np.abs(
                np.einsum("bi,bi->b", true_weight.conj(), candidate_weight, optimize=True)
            ) / denominators
        cosine = np.clip(cosine, 0.0, 1.0)
        deficits.append(1.0 - cosine)
        true_response = np.einsum("bi,ij->bj", true_weight.conj(), pattern, optimize=True)
        candidate_response = np.einsum(
            "bi,ij->bj", candidate_weight.conj(), pattern, optimize=True
        )
        true_responses.append(true_response)
        candidate_responses.append(candidate_response)
        response_difference = np.linalg.norm(candidate_response - true_response, axis=1)
        true_norm = np.linalg.norm(true_response, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            response_errors.append(response_difference / np.maximum(true_norm, np.finfo(float).tiny))

    deficit_array = np.stack(deficits, axis=1)
    true_response_array = np.stack(true_responses, axis=1)
    candidate_response_array = np.stack(candidate_responses, axis=1)
    response_error_array = np.stack(response_errors, axis=1)
    difference = candidate_response_array - true_response_array
    true_norm = np.linalg.norm(true_response_array)
    pattern_error = float(np.linalg.norm(difference) / max(true_norm, np.finfo(float).tiny))
    pattern_max = float(np.max(np.abs(difference))) if difference.size else float("nan")

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
        },
        "beam_pattern": {
            "relative_frobenius_error": pattern_error,
            "beam_pattern_relative_error": pattern_error,
            "max_absolute_error": pattern_max,
            "beam_response_max_absolute_error": pattern_max,
            "relative_error_by_look_direction": [
                _finite_mean(values) for values in response_error_array.T
            ],
            "beam_response_relative_error_by_look_direction": [
                _finite_mean(values) for values in response_error_array.T
            ],
            "max_relative_error_by_look_direction": [
                _finite_max(values) for values in response_error_array.T
            ],
            "finite_values": int(np.count_nonzero(np.isfinite(response_error_array))),
            "total_values": int(batch * len(look_directions_deg)),
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
) -> dict[str, np.ndarray]:
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
    return {
        "reference_r": reference_r,
        "x0": original_x0,
        "fixed_reference": fixed_reference,
        "true_inverse": true_inverse,
    }


def correctness_metrics(
    actual: np.ndarray,
    context: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare a downloaded device result with both required references."""
    actual = np.asarray(actual)
    reference = context["fixed_reference"]
    true_inverse = context["true_inverse"]
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
        # This alias makes the matching-reference comparison easy to find next
        # to the equivalent field in older board records.
        "relative_error": matching_error,
        "device_test_gate": {
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
        metrics["quality_vs_matching_reference"] = same_array_metrics(actual, reference)
    else:
        metrics["quality_vs_matching_reference"] = None
    if finite and true_inverse_finite:
        metrics["quality_vs_true_inverse"] = same_array_metrics(actual, true_inverse)
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


def _read_telemetry(
    output_dir: Path,
    *,
    run_id: str,
    environment_path: Path | None = None,
    power_path: Path | None = None,
) -> dict[str, Any]:
    """Read the wrapper's sibling artifacts, retaining raw samples for audit."""
    telemetry: dict[str, Any] = {
        "status": "failed",
        "run_id": run_id,
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
        raw_environment = json.loads(selected_environment.read_text())
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
        selected_power = _read_one_artifact(
            output_dir,
            "power-{run_id}.csv",
            run_id=run_id,
            explicit=power_path,
        )
        with selected_power.open(newline="") as handle:
            reader = csv.DictReader(handle)
            expected = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")
            if tuple(reader.fieldnames or ()) != expected:
                raise ValueError(f"power trace has unexpected columns {reader.fieldnames!r}")
            samples = [dict(row) for row in reader]
        if not samples:
            raise ValueError("power trace contains no samples")
        telemetry["power_trace"] = {
            "file": selected_power.name,
            "run_id": run_id,
            "columns": list(expected),
            "samples": _sanitize_metadata(samples),
            "sampling_source": "board snapshot captured by run_in_container.sh",
        }
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
) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
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
    if rows is None:
        validate_comparison_rows()
    if not selected:
        raise ValueError("at least one Issue #88 row is required")
    if not first_launch:
        validate_comparison_rows(selected if len(selected) == 4 else ISSUE88_COMPARISON_ROWS)
        if launches != LAUNCHES_PER_ROW:
            raise ValueError(f"Issue #88 requires exactly {LAUNCHES_PER_ROW} timed launches")
    else:
        if launches < 1:
            raise ValueError("first-launch mode requires a positive launch setting")

    remaining = collections.Counter(row["size"] for row in selected)
    matrices_by_size: dict[int, np.ndarray] = {}
    x0_by_size: dict[int, np.ndarray] = {}
    true_inverse_by_size: dict[int, np.ndarray] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    comparison: list[dict[str, Any]] = []
    stopped = False
    for config in selected:
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
        if row.get("status") not in {"ok", "correctness_only"}:
            stopped = True
            break
        if first_launch:
            break

    passed = bool(comparison) and all(
        row.get("status") in {"ok", "correctness_only"} for row in comparison
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


def _record_payload(
    run: dict[str, Any],
    *,
    telemetry: dict[str, Any],
    run_id: str,
    raw_path: Path,
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    environment = telemetry.get("normalized_environment")
    power_trace = telemetry.get("power_trace") or {}
    rows = run.get("comparison_rows", [])
    overall_pass = run.get("status") == "pass" and telemetry.get("status") == "complete"
    timed_out = any(row.get("failure_stage") == "row_timeout" for row in rows)
    failure = None
    if run.get("failure") is not None:
        failure = run["failure"]
    elif telemetry.get("failures"):
        failure = telemetry["failures"][0]
    record: dict[str, Any] = {
        "record_schema": ISSUE88_RECORD_SCHEMA,
        "status": "pass" if overall_pass else "failed",
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
            "true_inverse": "NumPy complex128 inverse of original FP32 R",
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
                "samples": len(power_trace.get("samples", [])),
                "sampling_source": power_trace.get("sampling_source"),
            },
            "commands": {
                "wrapper": "enodia/tt/bench/run_in_container.sh",
                "runner": ISSUE88_RUNNER,
                "device_id": DEVICE_ID,
            },
        },
        "metric_definitions": {
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
            "abnormal_exit": run.get("status") != "pass",
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
) -> dict[str, Any]:
    """Capture all result-builder inputs while retaining only safe metadata."""
    payload = {
        "raw_schema": ISSUE88_RAW_SCHEMA,
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "run_id": run_id,
        "artifact_file": raw_path.name,
        "artifact_status": run.get("status"),
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
        rows = [
            row
            for row in rows
            if any(
                selector in row["name"] or selector in shape_name(row)
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


def main(argv: list[str] | None = None) -> int:
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
    telemetry = _read_telemetry(
        output_dir,
        run_id=run_id,
        environment_path=args.env_json,
        power_path=args.power_trace,
    )
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
                    run = run_comparison(ttnn, device, rows=selected_rows, first_launch=args.first_launch)
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
            "close_succeeded": not device_opened,
        }
        if device_opened:
            try:
                import ttnn

                ttnn.close_device(device)
            except Exception as exc:  # noqa: BLE001 - retain rows before returning failure
                close_error = exc
                run = _mark_failed(run, "close", exc)
                cleanup["close_succeeded"] = False
                cleanup["close_error"] = _sanitize_text(exc)
        run["cleanup"] = cleanup

    if close_error is not None:
        # The failure is represented in JSON; do not discard the completed rows.
        run["stopped_on_failure"] = True

    raw_path = _raw_path(result_path, run_id)
    raw = _raw_payload(run, telemetry=telemetry, run_id=run_id, raw_path=raw_path, cleanup=run["cleanup"])
    record = _record_payload(
        run,
        telemetry=telemetry,
        run_id=run_id,
        raw_path=raw_path,
        cleanup=run["cleanup"],
    )
    _atomic_json_write(raw_path, raw)
    _atomic_json_write(result_path, record)
    print(f"issue88 result -> {result_path}", flush=True)
    print(f"issue88 raw -> {raw_path.name}", flush=True)
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
