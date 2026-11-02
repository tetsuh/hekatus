"""Board-free truncation sweep for Issue #165.

Run with ``uv run python -m enodia.spec.sim.pulse_support_sweep --output-dir
 docs/measurements/issue165-pulse-support/YYYY-MM-DD`` from a clean committed
checkout. The result directory is write-once.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from enodia.demo import DEFAULT_SCATTERERS
from enodia.spec.probe import linear_5mhz
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim import (
    DEFAULT_PULSE_SUPPORT_SIGMAS,
    PointScatterer,
    _simulate_bmode_float64_frame,
)

CANDIDATE_SIGMAS = (4.0, 5.0, 5.5, 6.0, 7.0, 8.0)
MAX_RELATIVE_ERROR = 1e-6
TIMING_REPEATS = 3
OUTPUT_JSON = "pulse-support-sweep.json"
OUTPUT_README = "README.md"


def select_default_sigmas(
    candidates: list[dict[str, float | None]], *, max_relative_error: float = MAX_RELATIVE_ERROR
) -> float:
    """Choose the fastest measured k that satisfies the frame-error bound."""
    passing = [
        row
        for row in candidates
        if row["k"] is not None and row["max_relative_error"] <= max_relative_error
    ]
    if not passing:
        raise ValueError(f"no pulse support candidate meets the {max_relative_error:g} error bound")
    return min(passing, key=lambda row: (row["seconds"], row["k"]))["k"]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _cpu_model() -> str:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _uptime_load_average() -> dict[str, str]:
    raw = subprocess.run(
        ["uptime"], check=True, capture_output=True, text=True
    ).stdout.strip()
    match = re.search(r"load averages?:\s*(.+)$", raw, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError("uptime output did not include a load average")
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "load_average": match.group(1),
        "uptime_output": raw,
    }


def _timed_frame(profile, events, scatterers, support_sigmas):
    before = _uptime_load_average()
    started = time.perf_counter()
    frame = _simulate_bmode_float64_frame(
        profile,
        events,
        scatterers,
        pulse_support_sigmas=support_sigmas,
    )
    seconds = time.perf_counter() - started
    after = _uptime_load_average()
    return frame, seconds, {"before": before, "after": after}


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    peak = float(np.max(np.abs(reference)))
    difference = float(np.max(np.abs(candidate - reference)))
    if peak == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / peak


def _random_field(seed: int, count: int) -> list[PointScatterer]:
    rng = np.random.default_rng(seed)
    x_m = rng.uniform(-10e-3, 10e-3, count)
    z_m = rng.uniform(20e-3, 40e-3, count)
    amplitude = rng.uniform(0.25, 1.5, count)
    return [
        PointScatterer(float(x), float(z), float(a))
        for x, z, a in zip(x_m, z_m, amplitude, strict=True)
    ]


def _sweep_case(profile, events, name, scatterers, supports):
    reference, reference_seconds, reference_load = _timed_frame(
        profile, events, scatterers, None
    )
    rows = [
        {
            "k": None,
            "max_relative_error": 0.0,
            "max_relative_error_pct": 0.0,
            "seconds": reference_seconds,
            "seconds_per_scatterer": reference_seconds / len(scatterers),
            "uptime_load_average": reference_load,
        }
    ]
    for support in supports:
        errors = []
        timing_samples = []
        load_average_samples = []
        for _ in range(TIMING_REPEATS):
            candidate, seconds, load_average = _timed_frame(
                profile, events, scatterers, support
            )
            errors.append(_relative_error(candidate, reference))
            timing_samples.append(seconds)
            load_average_samples.append(load_average)
        error = max(errors)
        median_seconds = float(np.median(timing_samples))
        rows.append(
            {
                "k": support,
                "max_relative_error": error,
                "max_relative_error_pct": 100.0 * error,
                "seconds": median_seconds,
                "seconds_per_scatterer": median_seconds / len(scatterers),
                "seconds_samples": timing_samples,
                "uptime_load_average": load_average_samples,
            }
        )
    return {
        "name": name,
        "scatterer_count": len(scatterers),
        "rows": rows,
    }


def _summary_by_k(cases):
    summaries = []
    keys = [None, *CANDIDATE_SIGMAS]
    for k in keys:
        rows = [row for case in cases for row in case["rows"] if row["k"] == k]
        max_error = max(row["max_relative_error"] for row in rows)
        summaries.append(
            {
                "k": k,
                "max_relative_error": max_error,
                "max_relative_error_pct": 100.0 * max_error,
                "seconds": sum(row["seconds"] for row in rows),
            }
        )
    return summaries


def _render_readme(record: dict) -> str:
    summary = record["sweep_summary"]
    lines = [
        "# Issue #165 pulse-support sweep",
        "",
        "Board-free measurement generated by `enodia.spec.sim.pulse_support_sweep`; this file and its JSON companion are committed unchanged from the script output.",
        "",
        f"- Harness commit: `{record['environment']['commit']}`",
        f"- CPU: {record['environment']['cpu_model']}",
        f"- Python / NumPy: {record['environment']['python_version']} / {record['environment']['numpy_version']}",
        f"- Profile / events: `{record['profile']['name']}` / {record['profile']['event_count']}",
        f"- Seed: {record['scatterers']['seed']}",
        f"- Selected default: **{record['recommended_default_sigmas']}σ** (maximum relative error ≤ {record['selection_error_limit']:.1e})",
        "",
        "| k (σ) | Maximum relative error (%) | Total time across cases (s) |",
        "| ---: | ---: | ---: |",
    ]
    for row in summary:
        k_label = "None" if row["k"] is None else f"{row['k']:g}"
        lines.append(
            f"| {k_label} | {row['max_relative_error_pct']:.9g} | {row['seconds']:.6f} |"
        )
    default_case = next(case for case in record["cases"] if case["name"] == "seeded_random_field")
    none_row = next(row for row in default_case["rows"] if row["k"] is None)
    default_row = next(
        row for row in default_case["rows"] if row["k"] == record["recommended_default_sigmas"]
    )
    lines.extend(
        [
            "",
            f"For the seeded {default_case['scatterer_count']}-scatterer field, the measured reference time was {none_row['seconds']:.6f} s ({none_row['seconds_per_scatterer']:.6f} s/scatterer); the selected default took {default_row['seconds']:.6f} s ({default_row['seconds_per_scatterer']:.6f} s/scatterer).",
            "",
            "Load averages were captured with `uptime` immediately before and after each timed frame; the JSON preserves those observations.",
            "",
            "Reproduce with:",
            "",
            "```bash",
            "uv run python -m enodia.spec.sim.pulse_support_sweep --output-dir docs/measurements/issue165-pulse-support/YYYY-MM-DD",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_sweep(output_dir: Path, *, seed: int = 165) -> tuple[Path, Path]:
    output_json = output_dir / OUTPUT_JSON
    output_readme = output_dir / OUTPUT_README
    if output_json.exists() or output_readme.exists():
        raise FileExistsError(f"refusing to overwrite measurement output in {output_dir}")
    if _git("status", "--porcelain"):
        raise RuntimeError("pulse-support measurements require a clean committed checkout")

    profile = linear_5mhz()
    config = make_bmode_config(profile)
    events = list(config.events)
    cases = [
        _sweep_case(profile, events, "existing_single_scatterer", [PointScatterer(0.0, 20e-3)], CANDIDATE_SIGMAS),
        _sweep_case(profile, events, "golden_path_scatterers", list(DEFAULT_SCATTERERS), CANDIDATE_SIGMAS),
        _sweep_case(profile, events, "seeded_random_field", _random_field(seed, 200), CANDIDATE_SIGMAS),
    ]
    summary = _summary_by_k(cases)
    recommended = select_default_sigmas(summary, max_relative_error=MAX_RELATIVE_ERROR)
    current_default_row = next(row for row in summary if row["k"] == DEFAULT_PULSE_SUPPORT_SIGMAS)
    record = {
        "issue": 165,
        "measurement_date_utc": datetime.now(UTC).date().isoformat(),
        "measurement_type": "board-free CPU simulator sweep",
        "environment": {
            "commit": _git("rev-parse", "HEAD"),
            "cpu_model": _cpu_model(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "profile": {
            "name": profile.name,
            "f0_hz": profile.f0_hz,
            "bandwidth_frac": profile.bandwidth_frac,
            "sampling_rate_hz": profile.fs_hz,
            "event_count": len(events),
        },
        "scatterers": {
            "counts": [case["scatterer_count"] for case in cases],
            "seed": seed,
            "seeded_random_region_m": {"x": [-10e-3, 10e-3], "z": [20e-3, 40e-3]},
        },
        "selection_error_limit": MAX_RELATIVE_ERROR,
        "timing_repeats_per_candidate": TIMING_REPEATS,
        "implementation_default_sigmas_before_sweep": DEFAULT_PULSE_SUPPORT_SIGMAS,
        "recommended_default_sigmas": recommended,
        "recommended_default_max_relative_error": current_default_row["max_relative_error"]
        if recommended == DEFAULT_PULSE_SUPPORT_SIGMAS
        else next(row["max_relative_error"] for row in summary if row["k"] == recommended),
        "cases": cases,
        "sweep_summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_readme.write_text(_render_readme(record), encoding="utf-8")
    return output_json, output_readme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=165)
    args = parser.parse_args()
    json_path, readme_path = run_sweep(args.output_dir, seed=args.seed)
    print(json_path.as_posix())
    print(readme_path.as_posix())


if __name__ == "__main__":
    main()
