"""Analyze resident timestamp phases against the 32-bit wall-clock wrap.

The input files are external run artifacts.  This tool reads their absolute
uint64 timestamps, verifies the committed record metadata, and emits only
scalar/list analysis; it never copies raw arrays into the repository.  Pair
correspondence is checked by length, duplicate-aware multiset, and ordered
comparisons so a truncated or reordered record cannot pass silently.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CLOCK_MODULUS_TICKS = 1 << 32
DEFAULT_AICLK_MHZ = 1_350
DEFAULT_PHASE_WINDOW_MS = 0.1
PAIR_TARGET_TICKS = 2 * 1_350_000
SHORT_THRESHOLD_TICKS = 1_250_000
LONG_THRESHOLD_TICKS = 1_450_000
PAIR_TOLERANCE_TICKS = 1_000
PAIR_ORDERS = ("short-first", "long-first", "other")

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


def _record_pair_starts(record: Mapping[str, Any]) -> list[int]:
    outlier = record.get("outlier_analysis")
    if not isinstance(outlier, Mapping):
        raise TypeError("record is missing outlier_analysis")
    if "frame_start_indices" not in outlier:
        # Zero-pair records omit pair-detail arrays but declare an empty pair_sums list.
        pair_count = outlier.get("pair_count")
        pair_sums = outlier.get("pair_sums")
        if (
            type(pair_count) is int
            and pair_count == 0
            and isinstance(pair_sums, list)
            and not pair_sums
        ):
            return []
        raise ValueError("record outlier_analysis.frame_start_indices must be non-negative integers")
    starts = outlier["frame_start_indices"]
    if not isinstance(starts, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in starts
    ):
        raise ValueError("record outlier_analysis.frame_start_indices must be non-negative integers")
    return list(starts)


def _record_pairs(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outlier = record["outlier_analysis"]
    pairs = outlier.get("pairs", [])
    if not isinstance(pairs, list) or any(not isinstance(pair, Mapping) for pair in pairs):
        raise ValueError("record outlier_analysis.pairs must be a list of objects")
    return pairs


def _pair_order(first: int, second: int) -> str:
    if first < SHORT_THRESHOLD_TICKS and second > LONG_THRESHOLD_TICKS:
        return "short-first"
    if first > LONG_THRESHOLD_TICKS and second < SHORT_THRESHOLD_TICKS:
        return "long-first"
    return "other"


def _is_paired_outlier(first: int, second: int) -> bool:
    return (
        abs(first + second - PAIR_TARGET_TICKS) <= PAIR_TOLERANCE_TICKS
        and _pair_order(first, second) != "other"
    )


def _detect_pairs(timestamps: list[int]) -> list[dict[str, Any]]:
    intervals = [current - previous for previous, current in itertools.pairwise(timestamps)]
    pairs: list[dict[str, Any]] = []
    for interval_index, (first, second) in enumerate(itertools.pairwise(intervals)):
        if not _is_paired_outlier(first, second):
            continue
        first_endpoint = interval_index + 1
        pairs.append(
            {
                "interval_end_frame_indices": [first_endpoint, first_endpoint + 1],
                "short_ticks": min(first, second),
                "long_ticks": max(first, second),
                "first_interval_ticks": first,
                "second_interval_ticks": second,
                "pair_sum_ticks": first + second,
                "sum_delta_ticks": first + second - PAIR_TARGET_TICKS,
                "order": _pair_order(first, second),
                "first_interval_end_elapsed_ticks": timestamps[first_endpoint] - timestamps[0],
                "first_interval_end_elapsed_ticks_mod_period": (
                    timestamps[first_endpoint] - timestamps[0]
                )
                % CLOCK_MODULUS_TICKS,
            }
        )
    return pairs


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


def _count_orders(pairs: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter({order: 0 for order in PAIR_ORDERS})
    for pair in pairs:
        counts[str(pair["order"])] += 1
    return {order: counts[order] for order in PAIR_ORDERS}


def _duplicate_values(values: Iterable[int]) -> list[int]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _counter_difference(left: Iterable[int], right: Iterable[int]) -> list[int]:
    return sorted((Counter(left) - Counter(right)).elements())


def _ordered_mismatches(left: list[Any], right: list[Any]) -> list[int]:
    return [
        index
        for index in range(min(len(left), len(right)))
        if left[index] != right[index]
    ]


def _record_correspondence(
    detected_pairs: list[Mapping[str, Any]], record: Mapping[str, Any]
) -> dict[str, Any]:
    detected_endpoints = [pair["interval_end_frame_indices"] for pair in detected_pairs]
    record_starts = _record_pair_starts(record)
    record_pairs = _record_pairs(record)
    record_endpoints = [
        pair.get("interval_end_frame_indices") for pair in record_pairs if isinstance(pair, Mapping)
    ]
    detected_starts = [endpoints[0] for endpoints in detected_endpoints]

    missing = _counter_difference(record_starts, detected_starts)
    unexpected = _counter_difference(detected_starts, record_starts)
    record_duplicate_starts = _duplicate_values(record_starts)
    raw_duplicate_starts = _duplicate_values(detected_starts)
    frame_start_length_match = len(record_starts) == len(detected_starts)
    frame_start_multiset_match = not missing and not unexpected
    frame_start_order_mismatches = _ordered_mismatches(record_starts, detected_starts)
    frame_start_order_match = (
        frame_start_length_match and not frame_start_order_mismatches
    )
    exact_frame_start_match = (
        frame_start_length_match
        and frame_start_multiset_match
        and frame_start_order_match
        and not record_duplicate_starts
    )

    endpoint_mismatches = _ordered_mismatches(detected_endpoints, record_endpoints)
    endpoint_length_match = len(detected_endpoints) == len(record_endpoints)
    missing_endpoint_indices = list(range(len(record_endpoints), len(detected_endpoints)))
    unexpected_endpoint_indices = list(range(len(detected_endpoints), len(record_endpoints)))

    common_pair_count = min(len(detected_pairs), len(record_pairs))
    order_mismatches: list[int] = []
    value_mismatches: list[int] = []
    for index in range(common_pair_count):
        detected = detected_pairs[index]
        recorded = record_pairs[index]
        if recorded.get("order") is not None and recorded.get("order") != detected["order"]:
            order_mismatches.append(index)
        if (
            recorded.get("short_ticks") != detected["short_ticks"]
            or recorded.get("long_ticks") != detected["long_ticks"]
        ):
            value_mismatches.append(index)
    missing_pair_indices = list(range(len(record_pairs), len(detected_pairs)))
    unexpected_pair_indices = list(range(len(detected_pairs), len(record_pairs)))

    return {
        "record_pair_count": len(record_starts),
        "raw_derived_pair_count": len(detected_pairs),
        "record_frame_start_indices": record_starts,
        "raw_derived_frame_start_indices": detected_starts,
        "exact_frame_start_match": exact_frame_start_match,
        "missing_raw_pair_starts": missing,
        "unexpected_raw_pair_starts": unexpected,
        "frame_start_length_match": frame_start_length_match,
        "frame_start_length_mismatch": not frame_start_length_match,
        "frame_start_lengths": {
            "record": len(record_starts),
            "raw_derived": len(detected_starts),
        },
        "record_duplicate_frame_start_indices": record_duplicate_starts,
        "raw_derived_duplicate_frame_start_indices": raw_duplicate_starts,
        "frame_start_multiset_match": frame_start_multiset_match,
        "frame_start_order_match": frame_start_order_match,
        "frame_start_order_mismatches": frame_start_order_mismatches,
        "record_pair_list_count": len(record_pairs),
        "record_pair_list_length_match": len(record_pairs) == len(detected_pairs),
        "record_pair_list_length_mismatch": len(record_pairs) != len(detected_pairs),
        "record_pair_endpoints_present": (
            len(record_endpoints) == len(record_pairs)
            and all(endpoint is not None for endpoint in record_endpoints)
        ),
        "pair_endpoint_list_length_match": endpoint_length_match,
        "pair_endpoint_list_length_mismatch": not endpoint_length_match,
        "pair_endpoint_list_lengths": {
            "record": len(record_endpoints),
            "raw_derived": len(detected_endpoints),
        },
        "pair_endpoint_missing_record_indices": missing_endpoint_indices,
        "pair_endpoint_unexpected_record_indices": unexpected_endpoint_indices,
        "pair_endpoint_mismatches": endpoint_mismatches,
        "pair_endpoint_order_mismatches": endpoint_mismatches,
        "pair_order_mismatches": order_mismatches,
        "pair_order_missing_record_indices": missing_pair_indices,
        "pair_order_unexpected_record_indices": unexpected_pair_indices,
        "pair_value_mismatches": value_mismatches,
        "pair_value_missing_record_indices": missing_pair_indices,
        "pair_value_unexpected_record_indices": unexpected_pair_indices,
    }


def _raw_metadata_matches(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    expected = record.get("raw_timestamps")
    if not isinstance(expected, Mapping):
        return False
    return expected.get("count") == metadata["count"] and expected.get("sha256") == metadata["sha256"]


def analyze_run(
    timestamps: list[int],
    *,
    raw_metadata: Mapping[str, Any],
    record: Mapping[str, Any],
    aiclk_mhz: int = DEFAULT_AICLK_MHZ,
    phase_window_ms: float = DEFAULT_PHASE_WINDOW_MS,
) -> dict[str, Any]:
    """Return board-free wrap, pair-order, and record-correspondence analysis."""
    if len(timestamps) < 3:
        raise ValueError("at least three timestamps are required")
    if raw_metadata.get("count") != len(timestamps):
        raise ValueError("raw metadata count does not match timestamps")
    if not _raw_metadata_matches(record, raw_metadata):
        raise ValueError("raw timestamp count/hash does not match committed record")
    phase_window_ticks = _phase_window_ticks(aiclk_mhz, phase_window_ms)
    if phase_window_ticks >= CLOCK_MODULUS_TICKS:
        raise ValueError("phase window must be less than one clock period")

    intervals = [current - previous for previous, current in itertools.pairwise(timestamps)]
    pairs = _detect_pairs(timestamps)
    crossing_indices = _crossing_endpoint_indices(timestamps)
    wrap_candidate_indices = [
        index
        for index in crossing_indices
        if timestamps[index] % CLOCK_MODULUS_TICKS < phase_window_ticks
    ]
    pair_starts = [pair["interval_end_frame_indices"][0] for pair in pairs]
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
    pair_sum_values = [pair["pair_sum_ticks"] for pair in pairs]
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
        "aiclk_mhz": aiclk_mhz,
        "wall_clock_period_ticks": CLOCK_MODULUS_TICKS,
        "wall_clock_period_seconds": period_seconds,
        "interval_count": len(intervals),
        "pair_count": len(pairs),
        "pairs": pairs,
        "pair_order_counts": _count_orders(pairs),
        "pair_sum_min_ticks": min(pair_sum_values) if pair_sum_values else None,
        "pair_sum_max_ticks": max(pair_sum_values) if pair_sum_values else None,
        "pair_sum_delta_min_ticks": min(pair["sum_delta_ticks"] for pair in pairs)
        if pairs
        else None,
        "pair_sum_delta_max_ticks": max(pair["sum_delta_ticks"] for pair in pairs)
        if pairs
        else None,
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
        "run_elapsed_ticks": timestamps[-1] - timestamps[0],
        "run_elapsed_seconds": (timestamps[-1] - timestamps[0]) / (aiclk_mhz * 1_000_000),
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
    return analyze_run(
        timestamps,
        raw_metadata=metadata,
        record=record,
        aiclk_mhz=aiclk_mhz,
        phase_window_ms=phase_window_ms,
    )


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
