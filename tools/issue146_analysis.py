"""Pure statistics and telemetry helpers for the Issue #146 analyzer."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from scipy.stats import t as student_t

from enodia.tt.bench import power_trace_issue146

ARBITER_FIELDS = power_trace_issue146.ISSUE146_ARB_COLUMNS
GDDR_FIELDS = power_trace_issue146.ISSUE137_GDDR_COLUMNS
EXPECTED_ENABLED_MIN = 0x3
EXPECTED_ENABLED_MAX = 0x1D1
EXPECTED_BUSY_MIN = 0x10546
EXPECTED_CONTROL_MAX = 0x80546
_CONFIDENCE_QUANTILES = {0.99: 0.995, 0.95: 0.975}


class Issue146AnalysisError(ValueError):
    """Raised when a pure Issue #146 analysis input is not representable."""


def finite(value: Any, *, field: str) -> float:
    """Return a finite numeric value, rejecting booleans and non-finite text."""
    if isinstance(value, bool):
        raise Issue146AnalysisError(f"{field} must be finite numeric data")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Issue146AnalysisError(f"{field} must be finite numeric data") from exc
    if not math.isfinite(number):
        raise Issue146AnalysisError(f"{field} must be finite numeric data")
    return number


def integer(value: Any, *, field: str) -> int:
    """Return a non-negative integer from a JSON scalar."""
    if isinstance(value, bool):
        raise Issue146AnalysisError(f"{field} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Issue146AnalysisError(f"{field} must be a non-negative integer") from exc
    if number < 0 or (isinstance(value, float) and number != value):
        raise Issue146AnalysisError(f"{field} must be a non-negative integer")
    return number


def _correlation(values: Sequence[float], mean: float) -> float:
    centered = [value - mean for value in values]
    denominator = math.fsum(value * value for value in centered)
    if not denominator:
        return 0.0
    numerator = math.fsum(left * right for left, right in pairwise(centered))
    return numerator / denominator


def _statistics(
    values: Sequence[Any],
    *,
    confidence: float,
    within_run_pairs: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    if not values:
        raise Issue146AnalysisError("statistics require at least one sample")
    numbers = [finite(value, field="sample") for value in values]
    count = len(numbers)
    mean = math.fsum(numbers) / count
    sample_sd = statistics.stdev(numbers) if count >= 2 else 0.0
    if within_run_pairs is None:
        r1 = _correlation(numbers, mean)
        pair_count = max(0, count - 1)
    else:
        pair_count = 0
        numerator = 0.0
        centered = [number - mean for number in numbers]
        denominator = math.fsum(value * value for value in centered)
        for run_values in within_run_pairs:
            run_numbers = [finite(value, field="pooled sample") for value in run_values]
            for left, right in pairwise(run_numbers):
                numerator += (left - mean) * (right - mean)
                pair_count += 1
        r1 = numerator / denominator if denominator else 0.0
    n_eff = count * (1.0 - r1) / (1.0 + r1) if r1 > 0.0 else float(count)
    if not math.isfinite(n_eff) or n_eff <= 0.0:
        n_eff = float(count)
    degrees_of_freedom = max(2, math.floor(n_eff) - 1)
    quantile = _CONFIDENCE_QUANTILES[confidence]
    critical = float(student_t.ppf(quantile, degrees_of_freedom))
    half_width = critical * sample_sd / math.sqrt(n_eff)
    return {
        "n": count,
        "mean": mean,
        "sample_sd": sample_sd,
        "r1": r1,
        "n_eff": n_eff,
        "df": degrees_of_freedom,
        "confidence": confidence,
        "t_quantile": quantile,
        "t_critical": critical,
        "half_width": half_width,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "pair_count": pair_count,
        "precision_pass": half_width <= 10.0,
    }


def point_statistics(values: Sequence[Any], *, confidence: float = 0.99) -> dict[str, Any]:
    """Compute the registered arithmetic estimator and its serial correction."""
    if confidence not in {0.99, 0.95}:
        raise Issue146AnalysisError("confidence must be 0.99 or 0.95")
    return _statistics(values, confidence=confidence)


def pooled_statistics(runs: Sequence[Sequence[Any]], *, confidence: float = 0.95) -> dict[str, Any]:
    """Compute pooled statistics without a pair crossing a run boundary."""
    if confidence not in {0.95}:
        raise Issue146AnalysisError("pooled Issue #146 confidence must be 0.95")
    if not runs or any(not run for run in runs):
        raise Issue146AnalysisError("pooled statistics require non-empty runs")
    concatenated = [value for run in runs for value in run]
    return _statistics(concatenated, confidence=confidence, within_run_pairs=runs)


def select_plateau_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    first_launch_start_monotonic_ns: int,
    start_seconds: float = 60.0,
    end_seconds: float = 180.0,
) -> list[Mapping[str, Any]]:
    """Select raw points in the half-open plateau, without holding or interpolation."""
    start_ns = first_launch_start_monotonic_ns + round(start_seconds * 1_000_000_000.0)
    end_ns = first_launch_start_monotonic_ns + round(end_seconds * 1_000_000_000.0)
    return [
        row
        for row in rows
        if isinstance(row.get("timestamp_monotonic_ns"), int)
        and start_ns <= row["timestamp_monotonic_ns"] < end_ns
    ]


def _raw_word(value: Any, *, field: str) -> int:
    if not isinstance(value, str) or not value.startswith(("0x", "0X")):
        raise Issue146AnalysisError(f"{field} must be a raw hexadecimal word")
    try:
        number = int(value, 16)
    except (TypeError, ValueError) as exc:
        raise Issue146AnalysisError(f"{field} must be a raw hexadecimal word") from exc
    if number < 0 or number > 0xFFFFFFFF:
        raise Issue146AnalysisError(f"{field} must be a 32-bit raw hexadecimal word")
    return number


def decode_arbiter_row(row: Mapping[str, Any]) -> dict[str, int]:
    """Decode raw arbiter words while preserving the row's raw values elsewhere."""
    minimum = _raw_word(row.get("aiclk_arb_min_raw"), field="aiclk_arb_min_raw")
    maximum = _raw_word(row.get("aiclk_arb_max_raw"), field="aiclk_arb_max_raw")
    enabled_min = _raw_word(row.get("enabled_min_arb_raw"), field="enabled_min_arb_raw")
    enabled_max = _raw_word(row.get("enabled_max_arb_raw"), field="enabled_max_arb_raw")
    return {
        "aiclk_arb_min_frequency_mhz": minimum & 0xFFFF,
        "aiclk_arb_min_index": minimum >> 16,
        "aiclk_arb_max_frequency_mhz": maximum & 0xFFFF,
        "aiclk_arb_max_index": maximum >> 16,
        "enabled_min_arb": enabled_min,
        "enabled_max_arb": enabled_max,
        "aiclk_arb_min_raw_int": minimum,
        "aiclk_arb_max_raw_int": maximum,
    }


def decode_gddr_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Decode all sixteen GDDR die bytes using the pinned #137 provenance."""
    try:
        decoded = power_trace_issue146.decode_issue146_gddr_row(row, source_established=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise Issue146AnalysisError("GDDR row cannot be decoded") from exc
    values: dict[str, float] = {}
    for item in decoded.values():
        channels = item.get("channels_c") if isinstance(item, Mapping) else None
        if not isinstance(channels, Mapping):
            raise Issue146AnalysisError("GDDR row has no decoded channels")
        for channel, die in channels.items():
            if isinstance(die, Mapping):
                values[f"{channel}.bottom_c"] = finite(die.get("bottom_c"), field=channel)
                values[f"{channel}.top_c"] = finite(die.get("top_c"), field=channel)
    return {"values": values, "max_c": max(values.values()) if values else None}


def gddr_zero_word(row: Mapping[str, Any]) -> bool:
    """Return whether any retained packed GDDR word is numerically zero."""
    return any(_raw_word(row.get(field), field=field) == 0 for field in GDDR_FIELDS)


def group_timing_facts(
    groups: Sequence[Mapping[str, Any]],
    *,
    first_launch_start_monotonic_ns: int,
    plateau_start_seconds: float = 60.0,
    plateau_end_seconds: float = 180.0,
    workload_duration_seconds: float | None = None,
    idle_fraction_limit: float = 0.02,
    overrun_limit_seconds: float = 10.0,
) -> dict[str, Any]:
    """Recompute group count, bounds, clipped idle gaps, and overrun."""
    if not groups:
        raise Issue146AnalysisError("group timings are required")
    normalized: list[tuple[int, int]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise Issue146AnalysisError(f"group {index} is not an object")
        start = integer(group.get("start_monotonic_ns"), field=f"group {index} start")
        end = integer(group.get("end_monotonic_ns"), field=f"group {index} end")
        if end < start:
            raise Issue146AnalysisError(f"group {index} ends before it starts")
        normalized.append((start, end))
    if normalized[0][0] != first_launch_start_monotonic_ns:
        raise Issue146AnalysisError("first group does not bind first_launch_start_monotonic_ns")
    if any(left[1] > right[0] for left, right in pairwise(normalized)):
        raise Issue146AnalysisError("group timings overlap or are out of order")
    plateau_start = first_launch_start_monotonic_ns + round(plateau_start_seconds * 1_000_000_000.0)
    plateau_end = first_launch_start_monotonic_ns + round(plateau_end_seconds * 1_000_000_000.0)
    gap_ns = 0
    gaps: list[dict[str, Any]] = []
    for index, (previous, current) in enumerate(pairwise(normalized)):
        raw_start, raw_end = previous[1], current[0]
        clipped_start = max(raw_start, plateau_start)
        clipped_end = min(raw_end, plateau_end)
        clipped = max(0, clipped_end - clipped_start)
        if clipped:
            gap_ns += clipped
            gaps.append(
                {
                    "between_groups": [index, index + 1],
                    "raw_start_monotonic_ns": raw_start,
                    "raw_end_monotonic_ns": raw_end,
                    "clipped_start_monotonic_ns": clipped_start,
                    "clipped_end_monotonic_ns": clipped_end,
                    "duration_seconds": clipped / 1_000_000_000.0,
                }
            )
    plateau_ns = plateau_end - plateau_start
    idle_fraction = gap_ns / plateau_ns if plateau_ns > 0 else math.inf
    last_end = normalized[-1][1]
    workload_duration = (
        plateau_end_seconds if workload_duration_seconds is None else workload_duration_seconds
    )
    workload_end = first_launch_start_monotonic_ns + round(workload_duration * 1_000_000_000.0)
    overrun_seconds = (last_end - workload_end) / 1_000_000_000.0
    coverage_pass = last_end >= workload_end
    return {
        "group_count": len(normalized),
        "first_group_start_monotonic_ns": normalized[0][0],
        "last_group_end_monotonic_ns": last_end,
        "workload_end_monotonic_ns": workload_end,
        "overrun_seconds": overrun_seconds,
        "overrun_limit_seconds": overrun_limit_seconds,
        "overrun_pass": coverage_pass and overrun_seconds <= overrun_limit_seconds,
        "coverage_pass": coverage_pass,
        "plateau_start_monotonic_ns": plateau_start,
        "plateau_end_monotonic_ns": plateau_end,
        "idle_gap_ns": gap_ns,
        "idle_fraction": idle_fraction,
        "idle_fraction_limit": idle_fraction_limit,
        "idle_fraction_pass": idle_fraction <= idle_fraction_limit,
        "gaps": gaps,
        "groups": [
            {"start_monotonic_ns": start, "end_monotonic_ns": end} for start, end in normalized
        ],
    }


def current_dip_facts(
    rows: Sequence[Mapping[str, Any]],
    *,
    first_launch_start_monotonic_ns: int,
    window_start_seconds: float,
    window_end_seconds: float,
    group_timings: Sequence[Mapping[str, Any]] = (),
    sample_interval_seconds: float | None = None,
) -> dict[str, Any]:
    """Report strict current dips and their raw group-boundary correspondence.

    The threshold comparison is deliberately strict: a sample exactly equal to
    0.8 times the median is not a dip.  Boundary correspondence is descriptive;
    this helper does not classify dips as clustered or infer a cause.
    """
    start_ns = first_launch_start_monotonic_ns + round(window_start_seconds * 1_000_000_000.0)
    end_ns = first_launch_start_monotonic_ns + round(window_end_seconds * 1_000_000_000.0)
    selected = [
        row
        for row in rows
        if isinstance(row.get("timestamp_monotonic_ns"), int)
        and start_ns <= row["timestamp_monotonic_ns"] < end_ns
    ]
    currents: list[float] = []
    for row in selected:
        currents.append(finite(row.get("current_a"), field="current_a"))
    if not currents:
        return {
            "status": "not_evaluable",
            "window": {
                "start_seconds": window_start_seconds,
                "end_seconds": window_end_seconds,
                "selection": "strict half-open window",
            },
            "sample_count": 0,
            "median_current_a": None,
            "threshold_current_a": None,
            "threshold_operator": "strictly below (<)",
            "dip_count": 0,
            "dip_fraction": None,
            "dips": [],
            "sample_capture_interval_seconds": sample_interval_seconds,
            "capture_interval_limitation": (
                "No current samples were available in the selected window."
            ),
        }
    median = statistics.median(currents)
    threshold = 0.8 * median
    normalized_groups = [
        {
            "start_monotonic_ns": integer(group.get("start_monotonic_ns"), field="group start"),
            "end_monotonic_ns": integer(group.get("end_monotonic_ns"), field="group end"),
        }
        for group in group_timings
        if isinstance(group, Mapping)
    ]
    boundaries = [
        ("start", index, group["start_monotonic_ns"])
        for index, group in enumerate(normalized_groups)
    ] + [
        ("end", index, group["end_monotonic_ns"])
        for index, group in enumerate(normalized_groups)
    ]
    boundaries.sort(key=lambda item: item[2])
    dips: list[dict[str, Any]] = []
    for row, current in zip(selected, currents):
        if not current < threshold:
            continue
        timestamp = int(row["timestamp_monotonic_ns"])
        containing_group = next(
            (
                index
                for index, group in enumerate(normalized_groups)
                if group["start_monotonic_ns"] <= timestamp < group["end_monotonic_ns"]
            ),
            None,
        )
        containing_gap = next(
            (
                index
                for index, (previous, following) in enumerate(
                    pairwise(normalized_groups)
                )
                if previous["end_monotonic_ns"] <= timestamp < following["start_monotonic_ns"]
            ),
            None,
        )
        nearest = sorted(
            (
                {
                    "kind": kind,
                    "group_index": index,
                    "offset_seconds": (boundary - first_launch_start_monotonic_ns)
                    / 1_000_000_000.0,
                    "distance_seconds": abs(boundary - timestamp) / 1_000_000_000.0,
                }
                for kind, index, boundary in boundaries
            ),
            key=lambda item: (item["distance_seconds"], item["offset_seconds"]),
        )[:2]
        asic_power = row.get("power_w")
        dips.append(
            {
                "timestamp_monotonic_ns": timestamp,
                "offset_seconds": (timestamp - first_launch_start_monotonic_ns)
                / 1_000_000_000.0,
                "current_a": current,
                "board_power_w": finite(row.get("board_power_w"), field="board_power_w"),
                "asic_power_w": (
                    finite(asic_power, field="power_w") if asic_power is not None else None
                ),
                "group_correspondence": {
                    "containing_group_index": containing_group,
                    "containing_gap_index": containing_gap,
                    "nearest_group_boundaries": nearest,
                },
            }
        )
    return {
        "status": "reported",
        "window": {
            "start_seconds": window_start_seconds,
            "end_seconds": window_end_seconds,
            "selection": "strict half-open window",
        },
        "sample_count": len(currents),
        "median_current_a": median,
        "threshold_current_a": threshold,
        "threshold_operator": "strictly below (<)",
        "dip_count": len(dips),
        "dip_fraction": len(dips) / len(currents),
        "dips": dips,
        "sample_capture_interval_seconds": sample_interval_seconds,
        "capture_interval_limitation": (
            "Capture timestamps are discrete samples; boundary correspondence is limited "
            "by the configured sampler interval and does not establish causality."
        ),
    }


def left_held_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    first_launch_start_monotonic_ns: int,
    start_seconds: float = 60.0,
    end_seconds: float = 180.0,
) -> dict[str, Any]:
    """Compute the #137 duration-weighted left-held cross-check only."""
    start = first_launch_start_monotonic_ns + round(start_seconds * 1_000_000_000.0)
    end = first_launch_start_monotonic_ns + round(end_seconds * 1_000_000_000.0)
    ordered = sorted(
        (
            row
            for row in rows
            if isinstance(row.get("timestamp_monotonic_ns"), int) and field in row
        ),
        key=lambda row: int(row["timestamp_monotonic_ns"]),
    )
    if not ordered or int(ordered[-1]["timestamp_monotonic_ns"]) < end:
        return {
            "status": "not_evaluable",
            "reason": "left-held cross-check lacks a terminal sample",
            "field": field,
            "segment_count": 0,
        }
    segments: list[tuple[float, int]] = []
    for index, row in enumerate(ordered):
        timestamp = int(row["timestamp_monotonic_ns"])
        next_timestamp = (
            int(ordered[index + 1]["timestamp_monotonic_ns"]) if index + 1 < len(ordered) else end
        )
        segment_start = max(timestamp, start)
        segment_end = min(next_timestamp, end)
        if segment_end > segment_start:
            segments.append((finite(row[field], field=field), segment_end - segment_start))
    duration = sum(item[1] for item in segments)
    if duration != end - start or not segments:
        return {
            "status": "not_evaluable",
            "reason": "left-held cross-check lacks complete plateau coverage",
            "field": field,
            "segment_count": len(segments),
        }
    return {
        "status": "ok",
        "field": field,
        "duration_ns": duration,
        "segment_count": len(segments),
        "time_weighted_mean": math.fsum(value * span for value, span in segments) / duration,
        "min": min(value for value, _span in segments),
        "max": max(value for value, _span in segments),
    }


__all__ = [
    "ARBITER_FIELDS",
    "EXPECTED_BUSY_MIN",
    "EXPECTED_CONTROL_MAX",
    "EXPECTED_ENABLED_MAX",
    "EXPECTED_ENABLED_MIN",
    "GDDR_FIELDS",
    "Issue146AnalysisError",
    "current_dip_facts",
    "decode_arbiter_row",
    "decode_gddr_row",
    "finite",
    "gddr_zero_word",
    "group_timing_facts",
    "integer",
    "left_held_mean",
    "point_statistics",
    "pooled_statistics",
    "select_plateau_rows",
]
