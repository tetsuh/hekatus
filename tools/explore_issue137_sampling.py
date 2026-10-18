"""Board-free, post-hoc sample diagnostics; never a limiter verdict.

Emit JSON to stdout. Unlike the preregistered left-held duration-weighted
estimator, these statistics use only sample points timestamped in [60, 120).
The reported s/sqrt(n) is a naive IID standard error, not a validated error
bound for serially sampled telemetry or for the duration-weighted estimator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

DEFAULT_ARCHIVE = (
    Path(__file__).resolve().parents[1] / "docs/measurements/issue137-board-power/2026-10-18"
)
FIELDS = ("board_power_w", "current_a", "power_w", "aiclk_mhz")
BUSY_THRESHOLD_A = 150


def archive_file(root: Path, reference: str) -> Path:
    """Confine a source reference to the operator-selected archive."""
    base = root.resolve()
    file = (base / reference).resolve()
    if not file.is_relative_to(base):
        raise ValueError("source reference is outside the archive")
    return file


def _finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite selected {field}")
    return value


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _spread(values: list[float]) -> dict:
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "mean": statistics.fmean(values),
        "sample_sd": sd,
        "naive_se": sd / math.sqrt(len(values)) if sd is not None else None,
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_samples(rows: list[dict[str, str]], *, workload_start_ns: int) -> dict:
    """Summarize exact sample points, with no held boundary point or padding."""
    selected = [
        (index, row)
        for index, row in enumerate(rows)
        if workload_start_ns + 60_000_000_000
        <= int(row["timestamp_monotonic_ns"])
        < workload_start_ns + 120_000_000_000
    ]
    if not selected:
        raise ValueError("plateau contains no samples")
    values = [{field: _finite(row, field) for field in FIELDS} for _, row in selected]
    busy = [value for value in values if value["current_a"] >= BUSY_THRESHOLD_A]
    clock = _spread([value["aiclk_mhz"] for value in values])
    return {
        "sample_count": len(values),
        "selected_trace_row_indices": [index for index, _ in selected],
        "selected_offsets_ns": [
            int(row["timestamp_monotonic_ns"]) - workload_start_ns for _, row in selected
        ],
        "board_power_w": _spread([value["board_power_w"] for value in values]),
        "busy": {
            "threshold_current_a": BUSY_THRESHOLD_A,
            "sample_count": len(busy),
            "fraction": len(busy) / len(values),
            "means": {field: _mean([value[field] for value in busy]) for field in FIELDS[:3]},
        },
        "aiclk_mhz": {"mean": clock["mean"], "sample_sd": clock["sample_sd"]},
    }


def _source(root: Path, reference: str, inventory: dict) -> tuple[bytes, dict]:
    data = archive_file(root, reference).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    entry = inventory[reference]
    if digest != entry["sha256"] or len(data) != entry["byte_count"]:
        raise ValueError("source artifact does not match the immutable manifest")
    return data, {"path": reference, "sha256": digest, "byte_count": len(data)}


def _explore_run(root: Path, entry: dict, inventory: dict) -> dict:
    data, record_source = _source(root, entry["record"], inventory)
    record = json.loads(data)
    trace_reference = (Path(entry["record"]).parent / record["trace"]["path"]).as_posix()
    trace, trace_source = _source(root, trace_reference, inventory)
    rows = list(csv.DictReader(io.StringIO(trace.decode("utf-8"))))
    return {
        "run_number": entry["run_number"],
        "configuration": entry["configuration"],
        "sources": {"record": record_source, "trace": trace_source},
        **summarize_samples(rows, workload_start_ns=record["workload_start_monotonic_ns"]),
        "duration_weighted_reference": entry["plateau_means"],
    }


def explore_archive(root: Path) -> dict:
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    inventory = {entry["path"]: entry for entry in manifest["files"]}
    runs = [_explore_run(root, entry, inventory) for entry in manifest["runs"]]
    high = [run for run in runs if run["configuration"] in ("C081", "C108")]
    return {
        "schema": "issue137-exploratory-sampling-v1",
        "verdict_bearing": False,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selection": {
            "clock": "timestamp_monotonic_ns relative to workload_start_monotonic_ns",
            "window_seconds": [60, 120],
            "lower_inclusive": True,
            "upper_exclusive": True,
            "estimator": "unweighted arithmetic sample statistics",
            "sd_denominator": "n-1",
            "naive_se": "sample_sd / sqrt(n); IID assumption only, no correlation correction",
            "busy": "current_a >= 150 A; post-hoc proxy, not measured utilization",
            "boundary": "no pre-window held sample, interpolation, padding, or extrapolation",
        },
        "runs": runs,
        "summary": {
            "high_run_numbers": [run["run_number"] for run in high],
            "high_run_duration_weighted_board_power_mean_w": statistics.fmean(
                run["duration_weighted_reference"]["board_power_w"] for run in high
            ),
            "high_run_arithmetic_board_power_mean_w": statistics.fmean(
                run["board_power_w"]["mean"] for run in high
            ),
            "interpretation": "exploratory only; neither estimator replaces the preregistered verdict",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    print(json.dumps(explore_archive(args.archive), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
