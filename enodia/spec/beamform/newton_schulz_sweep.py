"""Offline Newton-Schulz reference sweep (design.md §9, issue #85).

Run the complete host-side sweep and write an ADR-0005 record with:

``uv run python -m enodia.spec.beamform.newton_schulz_sweep \
    --record docs/measurements/2026-10-03-host-newton-schulz-reference-sweep.json``

The matrices, steering vectors, and all sweep inputs are generated locally and
from a fixed seed.  No accelerator implementation is involved.  The sweep
keeps inverse and MV-weight direction metrics separate from the same-array
beam-pattern metric: the former two are the provisional stage-1 selection
basis, while the latter is measured evidence rather than a gate.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from enodia.spec.beamform.newton_schulz import (
    X0_CHOICES,
    newton_schulz_inverse,
)
from enodia.strict_json import dumps as strict_json_dumps
from enodia.strict_json import normalize_json

CONDITION_NUMBERS: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0)
ITERATIONS: tuple[int, ...] = tuple(range(8, 17))
APERTURE_SIZES: tuple[int, ...] = (16, 32, 64)
DTYPES: tuple[str, ...] = ("float64", "float32")
LOOK_DIRECTIONS_DEG: tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
PATTERN_DIRECTIONS_DEG: tuple[float, ...] = (
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
DEFAULT_SEED = 85
DEFAULT_RECORD = Path(
    "docs/measurements/2026-10-03-host-newton-schulz-reference-sweep.json"
)

# The design's L0 discussion uses 1e-2 as a per-stage relative-error warning
# level.  The direction metric below is a cosine deficit, so it uses the same
# dimensionless threshold for the provisional host-side recommendation.
INVERSE_ERROR_GATE = 1e-2
MV_DIRECTION_ERROR_GATE = 1e-2


@dataclass(frozen=True)
class SweepResult:
    """Metrics for one (dtype, X0, N, condition number, aperture) point."""

    condition_number: float
    realized_condition_number: float
    aperture_size: int
    x0: str
    iterations: int
    dtype: str
    matrix_dtype: str
    inverse_relative_frobenius_error: float
    mv_direction_error: tuple[float, ...]
    mv_direction_error_deg: tuple[float, ...]
    beam_pattern_relative_error: float
    beam_response_max_absolute_error: float
    beam_response_relative_error: tuple[float, ...]


def _complex_dtype(dtype: str | np.dtype | type) -> np.dtype:
    """Map a real or complex NumPy dtype to the corresponding complex dtype."""
    value = np.dtype(dtype)
    if value.kind == "f" and value.itemsize in (4, 8):
        return np.dtype(np.complex64 if value.itemsize == 4 else np.complex128)
    if value.kind == "c" and value.itemsize in (8, 16):
        return value
    raise ValueError("dtype must be float32/float64 or complex64/complex128")


def _dtype_name(dtype: str | np.dtype | type) -> str:
    """Return the real precision name used in the measurement record."""
    value = _complex_dtype(dtype)
    return "float32" if value == np.dtype(np.complex64) else "float64"


def _matrix_seed(aperture_size: int, condition_number: float, seed: int) -> int:
    """Derive an independent deterministic stream for one matrix shape."""
    return int(seed) + 1009 * aperture_size + round(condition_number * 100.0)


def deterministic_hpd(
    aperture_size: int,
    condition_number: float,
    *,
    seed: int = DEFAULT_SEED,
    dtype: str | np.dtype | type = np.complex128,
) -> np.ndarray:
    """Construct a deterministic complex Hermitian positive-definite matrix.

    A seeded complex Gaussian QR factor supplies a unitary eigenbasis, and
    geometric eigenvalues span the requested condition number.  The matrix is
    formed first in complex128 and then cast, so float32 and float64 sweep
    points use the same underlying construction.
    """
    if not isinstance(aperture_size, (int, np.integer)) or aperture_size < 1:
        raise ValueError(f"aperture_size must be positive, got {aperture_size}")
    if not np.isfinite(condition_number) or condition_number < 1.0:
        raise ValueError(f"condition_number must be finite and at least 1, got {condition_number}")

    rng = np.random.default_rng(_matrix_seed(aperture_size, condition_number, seed))
    gaussian = rng.standard_normal((aperture_size, aperture_size))
    gaussian = gaussian + 1j * rng.standard_normal((aperture_size, aperture_size))
    basis, _ = np.linalg.qr(gaussian)
    eigenvalues = np.geomspace(1.0, float(condition_number), aperture_size)
    matrix = (basis * eigenvalues) @ basis.conj().T
    matrix = (matrix + matrix.conj().T) * 0.5
    return matrix.astype(_complex_dtype(dtype))


def steering_vector(
    aperture_size: int,
    direction_deg: float,
    *,
    dtype: str | np.dtype | type = np.complex128,
) -> np.ndarray:
    """Return an equal-spaced half-wavelength linear-array steering vector.

    Element zero is at the centre-relative coordinate ``-(L-1)/2``.  The
    phase convention is ``exp(j*pi*n*sin(theta))`` for direction ``theta`` in
    degrees; using the same convention for the true and approximate weights
    makes the pattern comparison independent of a global weight phase.
    """
    if not isinstance(aperture_size, (int, np.integer)) or aperture_size < 1:
        raise ValueError(f"aperture_size must be positive, got {aperture_size}")
    if not np.isfinite(direction_deg):
        raise ValueError(f"direction_deg must be finite, got {direction_deg}")
    positions = np.arange(aperture_size, dtype=np.float64) - (aperture_size - 1.0) / 2.0
    phase = np.pi * positions * np.sin(np.deg2rad(direction_deg))
    return np.exp(1j * phase).astype(_complex_dtype(dtype))


def _weight(inverse: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """Form the normalized MV weight for one steering vector."""
    weight = inverse @ steering
    denominator = steering.conj() @ weight
    return weight / denominator


def _direction_cosine_deficit(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return ``1 - |uᴴv|/(||u|| ||v||)`` for two weight directions."""
    denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
    if denominator == 0.0:
        return float("nan")
    cosine = abs(np.vdot(reference, candidate)) / denominator
    cosine = float(np.clip(cosine, 0.0, 1.0))
    return 1.0 - cosine


def _direction_angle_deg(cosine_deficit: float) -> float:
    """Convert a cosine deficit to its acute direction angle for readability."""
    if not np.isfinite(cosine_deficit):
        return float("nan")
    cosine = np.clip(1.0 - cosine_deficit, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def evaluate_point(
    matrix: np.ndarray,
    *,
    condition_number: float,
    aperture_size: int,
    x0: str,
    iterations: int,
    dtype: str | np.dtype | type,
    look_directions_deg: tuple[float, ...] = LOOK_DIRECTIONS_DEG,
    pattern_directions_deg: tuple[float, ...] = PATTERN_DIRECTIONS_DEG,
) -> SweepResult:
    """Measure inverse, MV direction, and same-array pattern errors."""
    matrix = np.asarray(matrix)
    work_dtype = _complex_dtype(dtype)
    if matrix.shape != (aperture_size, aperture_size):
        raise ValueError(
            f"matrix shape {matrix.shape} does not match aperture_size={aperture_size}"
        )
    matrix = matrix.astype(work_dtype, copy=False)
    true_matrix = matrix.astype(np.complex128)
    true_inverse = np.linalg.inv(true_matrix)
    approximate_inverse = newton_schulz_inverse(matrix, iterations, x0=x0)
    inverse_error = np.linalg.norm(
        approximate_inverse.astype(np.complex128) - true_inverse, ord="fro"
    ) / np.linalg.norm(true_inverse, ord="fro")

    pattern = np.stack(
        [
            steering_vector(aperture_size, direction, dtype=np.complex128)
            for direction in pattern_directions_deg
        ],
        axis=1,
    )
    direction_errors: list[float] = []
    direction_angles: list[float] = []
    response_errors: list[float] = []
    true_responses: list[np.ndarray] = []
    approximate_responses: list[np.ndarray] = []
    for direction in look_directions_deg:
        steering_true = steering_vector(aperture_size, direction, dtype=np.complex128)
        steering_work = steering_true.astype(work_dtype)
        true_weight = _weight(true_inverse, steering_true)
        approximate_weight = _weight(approximate_inverse, steering_work).astype(np.complex128)
        direction_error = _direction_cosine_deficit(true_weight, approximate_weight)
        direction_errors.append(direction_error)
        direction_angles.append(_direction_angle_deg(direction_error))
        true_response = true_weight.conj() @ pattern
        approximate_response = approximate_weight.conj() @ pattern
        true_responses.append(true_response)
        approximate_responses.append(approximate_response)
        response_errors.append(
            float(np.linalg.norm(approximate_response - true_response)
                  / max(np.linalg.norm(true_response), np.finfo(float).tiny))
        )

    true_response_array = np.stack(true_responses)
    approximate_response_array = np.stack(approximate_responses)
    pattern_error = np.linalg.norm(approximate_response_array - true_response_array) / max(
        np.linalg.norm(true_response_array), np.finfo(float).tiny
    )
    max_response_error = np.max(np.abs(approximate_response_array - true_response_array))
    realized_condition_number = np.linalg.cond(true_matrix)
    return SweepResult(
        condition_number=float(condition_number),
        realized_condition_number=float(realized_condition_number),
        aperture_size=aperture_size,
        x0=x0,
        iterations=iterations,
        dtype=_dtype_name(work_dtype),
        matrix_dtype=np.dtype(work_dtype).name,
        inverse_relative_frobenius_error=float(inverse_error),
        mv_direction_error=tuple(float(value) for value in direction_errors),
        mv_direction_error_deg=tuple(float(value) for value in direction_angles),
        beam_pattern_relative_error=float(pattern_error),
        beam_response_max_absolute_error=float(max_response_error),
        beam_response_relative_error=tuple(float(value) for value in response_errors),
    )


def sweep(
    *,
    condition_numbers: tuple[float, ...] = CONDITION_NUMBERS,
    iterations: tuple[int, ...] = ITERATIONS,
    aperture_sizes: tuple[int, ...] = APERTURE_SIZES,
    x0_choices: tuple[str, ...] = X0_CHOICES,
    dtypes: tuple[str, ...] = DTYPES,
    seed: int = DEFAULT_SEED,
) -> tuple[tuple[SweepResult, ...], float]:
    """Run every requested sweep point and return results plus elapsed seconds."""
    if not x0_choices or any(choice not in X0_CHOICES for choice in x0_choices):
        raise ValueError(f"x0_choices must be drawn from {X0_CHOICES}")
    if not iterations or any(
        not isinstance(value, (int, np.integer)) or value < 0 for value in iterations
    ):
        raise ValueError("iterations must contain non-negative integers")
    if not dtypes or any(_dtype_name(value) not in DTYPES for value in dtypes):
        raise ValueError(f"dtypes must be drawn from {DTYPES}")

    started = time.perf_counter()
    results: list[SweepResult] = []
    for dtype in dtypes:
        matrix_dtype = _complex_dtype(dtype)
        for condition_number in condition_numbers:
            for aperture_size in aperture_sizes:
                matrix = deterministic_hpd(
                    aperture_size, condition_number, seed=seed, dtype=matrix_dtype
                )
                for x0 in x0_choices:
                    for iteration_count in iterations:
                        results.append(
                            evaluate_point(
                                matrix,
                                condition_number=condition_number,
                                aperture_size=aperture_size,
                                x0=x0,
                                iterations=int(iteration_count),
                                dtype=matrix_dtype,
                            )
                        )
    return tuple(results), time.perf_counter() - started


def provisional_recommendation(results: tuple[SweepResult, ...] | list[SweepResult]) -> dict:
    """Select the smallest fixed N meeting the inverse and direction gates."""
    results = tuple(results)
    if not results:
        raise ValueError("cannot recommend from an empty sweep")
    by_dtype: dict[str, dict] = {}
    available_dtypes = [dtype for dtype in DTYPES if any(r.dtype == dtype for r in results)]
    for dtype in available_dtypes:
        candidates: list[dict] = []
        for x0 in X0_CHOICES:
            for iteration_count in sorted({r.iterations for r in results if r.dtype == dtype}):
                points = [
                    r
                    for r in results
                    if r.dtype == dtype and r.x0 == x0 and r.iterations == iteration_count
                ]
                if not points:
                    continue
                worst_inverse = max(r.inverse_relative_frobenius_error for r in points)
                worst_direction = max(max(r.mv_direction_error) for r in points)
                passes = (
                    worst_inverse <= INVERSE_ERROR_GATE
                    and worst_direction <= MV_DIRECTION_ERROR_GATE
                )
                candidates.append(
                    {
                        "x0": x0,
                        "iterations": iteration_count,
                        "worst_inverse_relative_frobenius_error": worst_inverse,
                        "worst_mv_direction_cosine_deficit": worst_direction,
                        "passes_stage_1_gate": passes,
                    }
                )
        passing = [candidate for candidate in candidates if candidate["passes_stage_1_gate"]]
        if passing:
            selected = min(
                passing,
                key=lambda candidate: (
                    candidate["iterations"],
                    candidate["worst_inverse_relative_frobenius_error"],
                    candidate["worst_mv_direction_cosine_deficit"],
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda candidate: (
                    max(
                        candidate["worst_inverse_relative_frobenius_error"] / INVERSE_ERROR_GATE,
                        candidate["worst_mv_direction_cosine_deficit"] / MV_DIRECTION_ERROR_GATE,
                    ),
                    candidate["iterations"],
                ),
            )
        by_dtype[dtype] = selected

    return {
        "selection": (
            "For each dtype, choose the smallest fixed iteration count whose worst-case inverse "
            "and MV direction metrics pass both provisional gates; if none pass, choose the "
            "lowest normalized worst-case score."
        ),
        "gates": {
            "inverse_relative_frobenius_error_max": INVERSE_ERROR_GATE,
            "mv_direction_cosine_deficit_max": MV_DIRECTION_ERROR_GATE,
        },
        "by_dtype": by_dtype,
        "note": (
            "Provisional stage-1 recommendation is based on inverse and MV-weight direction "
            "metrics. Same-array beam response/pattern error is measured for evidence but is "
            "not a stage-1 decision gate."
        ),
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment() -> dict:
    """Return host reproducibility data without a hostname or machine name."""
    import numpy
    import scipy

    revision = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all")
    return {
        "captured_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "platform": f"{platform.system()}-{platform.release()}",
        "architecture": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "harness_revision": None if revision == "unknown" else revision,
        "harness_dirty": None if status == "unknown" else bool(status),
        "board": None,
        "note": "host-side NumPy reference sweep; no accelerator involved",
    }


def _result_record(result: SweepResult) -> dict:
    """Convert one result to the explicit JSON record shape."""
    return {
        "condition_number_requested": result.condition_number,
        "condition_number_realized": result.realized_condition_number,
        "aperture_size": result.aperture_size,
        "x0": result.x0,
        "iterations": result.iterations,
        "dtype": result.dtype,
        "matrix_dtype": result.matrix_dtype,
        "metrics": {
            "inverse_relative_frobenius_error": result.inverse_relative_frobenius_error,
            "mv_weight_direction_error": {
                "definition": "1 - absolute normalized direction cosine",
                "by_look_direction": list(result.mv_direction_error),
                "by_look_direction_deg": list(result.mv_direction_error_deg),
                "mean_cosine_deficit": float(np.mean(result.mv_direction_error)),
                "max_cosine_deficit": max(result.mv_direction_error),
            },
            "beam_pattern_relative_error": result.beam_pattern_relative_error,
            "beam_response_max_absolute_error": result.beam_response_max_absolute_error,
            "beam_response_relative_error_by_look_direction": list(
                result.beam_response_relative_error
            ),
        },
    }


def measurement_record(
    results: tuple[SweepResult, ...] | list[SweepResult],
    elapsed_seconds: float,
    *,
    condition_numbers: tuple[float, ...] = CONDITION_NUMBERS,
    iterations: tuple[int, ...] = ITERATIONS,
    aperture_sizes: tuple[int, ...] = APERTURE_SIZES,
    x0_choices: tuple[str, ...] = X0_CHOICES,
    dtypes: tuple[str, ...] = DTYPES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Build an ADR-0005-compliant strict-JSON-ready host record."""
    results = tuple(results)
    return normalize_json(
        {
            "record_schema": "adr-0005-host-sweep-v1",
            "environment": environment(),
            "what": (
                "NumPy Newton-Schulz inverse reference sweep for issue #85; fixed iteration "
                "counts and two selectable initial values"
            ),
            "algorithm": {
                "update": "X[k+1] = X[k] @ (2I - R @ X[k])",
                "x0_choices": list(x0_choices),
                "x0_definitions": {
                    "r_h_norms": "R^H / (||R||_1 ||R||_infinity)",
                    "identity_norminf": "I / ||R||_infinity",
                },
                "complex_precision": {
                    "float64": "complex128",
                    "float32": "complex64",
                },
                "early_exit": False,
            },
            "sweep_parameters": {
                "condition_numbers_requested": list(condition_numbers),
                "iterations": list(iterations),
                "aperture_sizes": list(aperture_sizes),
                "dtypes": list(dtypes),
                "seed": seed,
                "steering_vector": {
                    "array": "equal-spaced linear array",
                    "element_spacing": "one-half wavelength",
                    "look_directions_deg": list(LOOK_DIRECTIONS_DEG),
                    "pattern_directions_deg": list(PATTERN_DIRECTIONS_DEG),
                },
            },
            "metric_definitions": {
                "inverse_relative_frobenius_error": (
                    "||X_N - R^-1||_F / ||R^-1||_F, with the true inverse computed in complex128"
                ),
                "mv_weight_direction_error": (
                    "1 - |w_true^H w_N| / (||w_true||_2 ||w_N||_2), for "
                    "w = R^-1 a / (a^H R^-1 a); reported over look directions"
                ),
                "beam_pattern_relative_error": (
                    "relative Frobenius error between w^H a(theta) response matrices over "
                    "look and pattern directions"
                ),
                "beam_response_max_absolute_error": (
                    "maximum absolute complex response difference over the same array and "
                    "direction grid"
                ),
            },
            "results": [_result_record(result) for result in results],
            "elapsed_seconds": elapsed_seconds,
            "provisional_recommendation": provisional_recommendation(results),
            "notes": (
                "This is stage-1 host evidence only. The provisional X0/N recommendation is "
                "derived from inverse and MV-direction metrics; beam-pattern evidence is "
                "measured but is not the stage-1 decision gate. No final design or budget "
                "decision is recorded here."
            ),
        }
    )


def _write_record(record: dict, output: Path) -> Path:
    """Write only below the current repository, returning a relative path."""
    root = Path.cwd().resolve()
    destination = output.resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"--record must point beneath the current directory {root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(strict_json_dumps(record, indent=2) + "\n")
    return destination.relative_to(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the offline Newton-Schulz reference sweep")
    parser.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_RECORD,
        help="write the ADR-0005 JSON record beneath the current directory",
    )
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    args = parser.parse_args(argv)

    dtypes = DTYPES if args.dtype == "both" else (args.dtype,)
    results, elapsed = sweep(dtypes=dtypes)
    record = measurement_record(results, elapsed, dtypes=dtypes)
    relative = _write_record(record, args.record)
    recommendation = record["provisional_recommendation"]["by_dtype"]
    print(f"sweep points: {len(results)}; elapsed: {elapsed:.2f} s")
    for dtype, selected in recommendation.items():
        print(
            f"{dtype}: x0={selected['x0']} N={selected['iterations']} "
            f"inverse={selected['worst_inverse_relative_frobenius_error']:.3e} "
            f"direction={selected['worst_mv_direction_cosine_deficit']:.3e} "
            f"gate={selected['passes_stage_1_gate']}"
        )
    print(f"record: {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
