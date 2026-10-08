"""Analyze resident timestamp phases against the 32-bit wall-clock wrap.

Record validation and the complete paired-outlier catalog are shared with
the resident builder in :mod:`enodia.tt.bench.resident_record`.  This tool
only performs phase-window presentation and file I/O around that canonical
catalog; it does not carry a second invariant catalog.  The input files are
external run artifacts and this tool emits only scalar/list analysis, never
raw arrays into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from enodia.tt.bench.resident_record import (
    CLOCK_MODULUS_TICKS,
    DEFAULT_AICLK_MHZ,
    DEFAULT_PHASE_WINDOW_MS,
    LONG_THRESHOLD_TICKS,  # noqa: F401 - preserve analyzer module API
    PAIR_ORDERS,  # noqa: F401 - preserve analyzer module API
    PAIR_TARGET_TICKS,  # noqa: F401 - preserve analyzer module API
    PAIR_TOLERANCE_TICKS,  # noqa: F401 - preserve analyzer module API
    SHORT_THRESHOLD_TICKS,  # noqa: F401 - preserve analyzer module API
    _count_orders,
    _detect_pairs,  # noqa: F401 - preserve analyzer module API
    _is_paired_outlier,
    _pair_order,
    _record_correspondence,
    build_outlier_analysis,
    normalize_sampler_metadata,
    raw_metadata_matches,
    validate_resident_record,
)

# Compatibility aliases retain the analyzer's board-free names while keeping
# their implementation in the shared module.
validate_record_invariants = validate_resident_record
_raw_metadata_matches = raw_metadata_matches

RUN_SPECS = {
    "sampler-off": (
        "off/raw-timestamps.bin",
        "2026-10-06-p150a-issue104-sampler-off.json",
    ),
    "sampler-default": (
        "default/raw-timestamps.bin",
        "2026-10-06-p150a-issue104-sampler-default.json",
    ),
    "sampler-5s": (
        "explicit-5s/raw-timestamps.bin",
        "2026-10-06-p150a-issue104-sampler-5s.json",
    ),
}


def _phase_window_ticks(aiclk_mhz: int, width_ms: float) -> int:
    if isinstance(aiclk_mhz, bool) or not isinstance(aiclk_mhz, int) or aiclk_mhz <= 0:
        raise ValueError("aiclk_mhz must be a positive integer")
    if not math.isfinite(width_ms) or width_ms <= 0:
        raise ValueError("phase window must be a positive finite number of milliseconds")
    return math.ceil(width_ms * aiclk_mhz * 1_000)


def _read_uint64_le(path: Path) -> tuple[list[int], dict[str, Any]]:
    data = path.read_bytes()
    if len(data) % 8:
        raise ValueError(f"{path.name} is not a whole number of 8-byte records")
    values = [value[0] for value in struct.iter_unpack("<Q", data)]
    if any(current <= previous for previous, current in itertools.pairwise(values)):
        raise ValueError(f"{path.name} timestamps must be strictly increasing absolute values")
    return values, {
        "count": len(values),
        "sha256": hashlib.sha256(data).hexdigest(),
        "encoding": "unsigned 64-bit absolute timestamp",
        "endianness": "little",
        "packing": "<Q",
        "bytes_per_value": 8,
    }



def _crossing_endpoint_indices(timestamps: list[int]) -> list[int]:
    return [
        index
        for index, (previous, current) in enumerate(itertools.pairwise(timestamps), start=1)
        if current // CLOCK_MODULUS_TICKS > previous // CLOCK_MODULUS_TICKS
    ]


def _circular_mean(phases: Iterable[int]) -> float | None:
    values = list(phases)
    if not values:
        return None
    angles = [2.0 * math.pi * phase / CLOCK_MODULUS_TICKS for phase in values]
    sine = sum(math.sin(angle) for angle in angles)
    cosine = sum(math.cos(angle) for angle in angles)
    return (math.atan2(sine, cosine) % (2.0 * math.pi)) * CLOCK_MODULUS_TICKS / (2.0 * math.pi)


def _circular_arc_width(phases: Iterable[int]) -> int | None:
    ordered = sorted(phases)
    if not ordered:
        return None
    if len(ordered) == 1:
        return 0
    gaps = [current - previous for previous, current in itertools.pairwise(ordered)]
    gaps.append(CLOCK_MODULUS_TICKS - ordered[-1] + ordered[0])
    return CLOCK_MODULUS_TICKS - max(gaps)


def _rayleigh_r(phases: Iterable[int]) -> float | None:
    values = list(phases)
    if not values:
        return None
    angles = [2.0 * math.pi * phase / CLOCK_MODULUS_TICKS for phase in values]
    return math.hypot(
        sum(math.cos(angle) for angle in angles),
        sum(math.sin(angle) for angle in angles),
    ) / len(values)


def _record_sampler_metadata(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    sampler = record.get("telemetry_sampler")
    if isinstance(sampler, Mapping):
        return normalize_sampler_metadata(sampler)
    environment = record.get("environment")
    nested = environment.get("telemetry_sampler") if isinstance(environment, Mapping) else None
    if isinstance(nested, Mapping):
        return normalize_sampler_metadata(nested)
    return None


def _record_aiclk_mhz(record: Mapping[str, Any]) -> int | None:
    clock = record.get("clock")
    value = clock.get("aiclk_mhz") if isinstance(clock, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def analyze_run(
    timestamps: list[int],
    *,
    raw_metadata: Mapping[str, Any],
    record: Mapping[str, Any],
    aiclk_mhz: int | None = None,
    phase_window_ms: float = DEFAULT_PHASE_WINDOW_MS,
) -> dict[str, Any]:
    """Return board-free wrap, pair-order, and record-correspondence analysis."""
    if len(timestamps) < 3:
        raise ValueError("at least three timestamps are required")
    if raw_metadata.get("count") != len(timestamps):
        raise ValueError("raw metadata count does not match timestamps")
    if not _raw_metadata_matches(record, raw_metadata):
        raise ValueError("raw timestamp count/hash does not match committed record")
    invariant_validation = validate_resident_record(
        record, raw_metadata=raw_metadata, timestamps=timestamps
    )
    if aiclk_mhz is None:
        aiclk_mhz = _record_aiclk_mhz(record) or DEFAULT_AICLK_MHZ
    phase_window_ticks = _phase_window_ticks(aiclk_mhz, phase_window_ms)
    if phase_window_ticks >= CLOCK_MODULUS_TICKS:
        raise ValueError("phase window must be less than one clock period")

    intervals = [current - previous for previous, current in itertools.pairwise(timestamps)]
    outlier_analysis = build_outlier_analysis(
        timestamps,
        aiclk_mhz=aiclk_mhz,
        sampler_metadata=_record_sampler_metadata(record),
    )
    pairs = outlier_analysis["pairs"]
    crossing_indices = _crossing_endpoint_indices(timestamps)
    wrap_candidate_indices = [
        index
        for index in crossing_indices
        if timestamps[index] % CLOCK_MODULUS_TICKS < phase_window_ticks
    ]
    pair_starts = outlier_analysis["frame_start_indices"]
    pair_start_set = set(pair_starts)
    candidate_set = set(wrap_candidate_indices)
    event_phases_raw = [timestamps[index] % CLOCK_MODULUS_TICKS for index in pair_starts]
    event_phases_elapsed = [
        (timestamps[index] - timestamps[0]) % CLOCK_MODULUS_TICKS for index in pair_starts
    ]
    candidate_order_entries: list[dict[str, Any]] = []
    for index in wrap_candidate_indices:
        first = intervals[index - 1]
        second = intervals[index] if index < len(intervals) else None
        order = _pair_order(first, second) if second is not None else "other"
        candidate_order_entries.append(
            {
                "frame_index": index,
                "raw_phase_ticks": timestamps[index] % CLOCK_MODULUS_TICKS,
                "first_interval_ticks": first,
                "second_interval_ticks": second,
                "pair_sum_ticks": first + second if second is not None else None,
                "order": order if second is not None and _is_paired_outlier(first, second) else "other",
                "is_event_pair_start": index in pair_start_set,
            }
        )
    candidate_order_counts = _count_orders(candidate_order_entries)
    record_match = _record_correspondence(pairs, record)
    phase_width_ticks = _circular_arc_width(event_phases_raw)
    period_seconds = CLOCK_MODULUS_TICKS / (aiclk_mhz * 1_000_000)
    raw_phase_min_ticks = min(event_phases_raw) if event_phases_raw else None
    raw_phase_max_ticks = max(event_phases_raw) if event_phases_raw else None
    elapsed_phase_min_ticks = min(event_phases_elapsed) if event_phases_elapsed else None
    elapsed_phase_max_ticks = max(event_phases_elapsed) if event_phases_elapsed else None
    return {
        "frame_count": len(timestamps),
        "raw_timestamps": dict(raw_metadata),
        "raw_timestamps_match_record": True,
        "invariant_validation": invariant_validation,
        "aiclk_mhz": aiclk_mhz,
        "wall_clock_period_ticks": CLOCK_MODULUS_TICKS,
        "wall_clock_period_seconds": period_seconds,
        "interval_count": len(intervals),
        "pair_count": len(pairs),
        "pairs": pairs,
        "pair_order_counts": outlier_analysis["pair_order_counts"],
        "pair_sum_min_ticks": outlier_analysis["pair_sum_min_ticks"],
        "pair_sum_max_ticks": outlier_analysis["pair_sum_max_ticks"],
        "pair_sum_delta_min_ticks": outlier_analysis["sum_delta_min_ticks"],
        "pair_sum_delta_max_ticks": outlier_analysis["sum_delta_max_ticks"],
        "outlier_analysis": outlier_analysis,
        "record_correspondence": record_match,
        "clock_wrap_crossings": {
            "definition": (
                "an adjacent absolute-timestamp pair whose integer quotient by 2^32 increases"
            ),
            "count": len(crossing_indices),
            "endpoint_frame_indices": crossing_indices,
        },
        "phase_window": {
            "definition": "post-wrap raw timestamp modulo 2^32 in [0, window_ticks)",
            "window_start_ticks": 0,
            "window_end_exclusive_ticks": phase_window_ticks,
            "window_width_ms": phase_window_ms,
            "wrap_candidate_count": len(wrap_candidate_indices),
            "wrap_candidate_frame_indices": wrap_candidate_indices,
            "wrap_candidate_order_counts": candidate_order_counts,
            "wrap_candidate_order_entries": candidate_order_entries,
            "event_pair_starts_in_window": len(pair_start_set & candidate_set),
            "event_pair_starts_outside_window": sorted(pair_start_set - candidate_set),
            "non_event_wrap_candidate_count": len(candidate_set - pair_start_set),
            "non_event_wrap_candidate_frame_indices": sorted(candidate_set - pair_start_set),
            "pair_to_wrap_candidate_ratio": {
                "numerator_event_pair_starts_in_window": len(pair_start_set & candidate_set),
                "denominator_wrap_candidates_in_window": len(wrap_candidate_indices),
                "fraction": (
                    len(pair_start_set & candidate_set) / len(wrap_candidate_indices)
                    if wrap_candidate_indices
                    else None
                ),
            },
        },
        "phase_analysis": {
            "definition": (
                "the first interval endpoint of each detected short/long pair; absolute phase is "
                "timestamp modulo 2^32 and elapsed phase is (timestamp - first_timestamp) modulo 2^32"
            ),
            "event_count": len(pair_starts),
            "event_frame_indices": pair_starts,
            "raw_event_phase_ticks_mod_period": event_phases_raw,
            "first_interval_end_elapsed_ticks": [
                timestamps[index] - timestamps[0] for index in pair_starts
            ],
            "first_interval_end_elapsed_ticks_mod_period": event_phases_elapsed,
            "raw_phase_min_ticks": raw_phase_min_ticks,
            "raw_phase_max_ticks": raw_phase_max_ticks,
            "elapsed_phase_min_ticks": elapsed_phase_min_ticks,
            "elapsed_phase_max_ticks": elapsed_phase_max_ticks,
            "rayleigh_R": _rayleigh_r(event_phases_elapsed),
            "circular_mean_raw_phase_ticks": _circular_mean(event_phases_raw),
            "circular_arc_width_ticks": phase_width_ticks,
            "circular_arc_width_ms": (
                phase_width_ticks / (aiclk_mhz * 1_000)
                if phase_width_ticks is not None
                else None
            ),
            "all_event_phases_in_window": (
                all(phase < phase_window_ticks for phase in event_phases_raw)
                if event_phases_raw
                else None
            ),
        },
        "run_elapsed_ticks": outlier_analysis["run_elapsed_ticks"],
        "run_elapsed_seconds": outlier_analysis["run_elapsed_seconds"],
    }


def analyze_file(
    raw_path: Path,
    record_path: Path,
    *,
    aiclk_mhz: int = DEFAULT_AICLK_MHZ,
    phase_window_ms: float = DEFAULT_PHASE_WINDOW_MS,
) -> dict[str, Any]:
    """Read one raw artifact and its committed record, then analyze it."""
    timestamps, metadata = _read_uint64_le(raw_path)
    record = json.loads(record_path.read_text())
    analysis = analyze_run(
        timestamps,
        raw_metadata=metadata,
        record=record,
        aiclk_mhz=aiclk_mhz,
        phase_window_ms=phase_window_ms,
    )
    trace_name = record.get("power_trace") if isinstance(record, Mapping) else None
    trace_path = record_path.parent / trace_name if isinstance(trace_name, str) else None
    analysis["invariant_validation"] = validate_resident_record(
        record,
        raw_metadata=metadata,
        timestamps=timestamps,
        power_trace_path=trace_path,
    )
    return analysis


def analyze_issue104(
    runs_dir: Path,
    records_dir: Path,
    *,
    issue12_raw: Path | None = None,
    issue12_record: Path | None = None,
    phase_window_ms: float = DEFAULT_PHASE_WINDOW_MS,
) -> dict[str, Any]:
    """Analyze the three Issue #104 runs and an optional Issue #12 artifact."""
    runs: dict[str, Any] = {}
    for name, (relative_raw, record_name) in RUN_SPECS.items():
        runs[name] = analyze_file(
            runs_dir / relative_raw,
            records_dir / record_name,
            phase_window_ms=phase_window_ms,
        )
    if issue12_raw is not None:
        if issue12_record is None:
            issue12_record = records_dir / "2026-10-06-p150a-issue12-stage1-final-500000-adr0005.json"
        runs["issue12-final-500k"] = analyze_file(
            issue12_raw,
            issue12_record,
            phase_window_ms=phase_window_ms,
        )
    return {
        "schema": "issue-104-wrap-analysis-v1",
        "clock": {
            "modulus_ticks": CLOCK_MODULUS_TICKS,
            "aiclk_mhz": DEFAULT_AICLK_MHZ,
            "period_seconds": CLOCK_MODULUS_TICKS / (DEFAULT_AICLK_MHZ * 1_000_000),
            "phase_window_ms": phase_window_ms,
        },
        "runs": runs,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, default=Path("docs/measurements"))
    parser.add_argument("--issue12-raw", type=Path, default=None)
    parser.add_argument("--issue12-record", type=Path, default=None)
    parser.add_argument("--phase-window-ms", type=float, default=DEFAULT_PHASE_WINDOW_MS)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_issue104(
        args.runs_dir,
        args.records_dir,
        issue12_raw=args.issue12_raw,
        issue12_record=args.issue12_record,
        phase_window_ms=args.phase_window_ms,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out is None:
        print(encoded, end="")
    else:
        args.json_out.write_text(encoded)
        print(f"analysis -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
