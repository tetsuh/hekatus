"""Analyze resident timestamp phases against the 32-bit wall-clock wrap.

Invariant catalog (``validate_record_invariants`` is the board-free analyzer
entry point and mirrors the resident builder):

* every declared count agrees with its sequence: raw timestamp count, interval
  count, histogram ``N`` and bin sum, pair count, frame starts, pair objects,
  pair sums, event arrays, and power sample count versus CSV rows and valid
  rows;
* sampler mode agrees with trace presence, trace filename, and the explicit
  absent-by-design reason;
* a named power trace is checked against its actual bytes (SHA-256), exact
  columns, readability, valid-row count, timestamp parsing/order, first/last
  values, and PR #109 coverage
  ``first_timestamp <= run_start <= run_end <= last_timestamp``;
* sampled timing evidence requires AICLK from valid in-run trace samples.  A
  pre-run environment snapshot, a sampler-off run, missing in-run rows, or
  incomplete coverage cannot support ``timing_evidence=true`` and is reported
  with a machine-readable reason;
* pair endpoints, starts, values, orders, duplicate-aware multisets, lengths,
  and order are compared against pairs derived from the raw timestamp stream;
* raw timestamp metadata must match the actual little-endian bytes.  Strict
  count/hash checks remain errors, while other mismatches are returned in the
  validation report instead of being silently accepted.

The input files are external run artifacts.  This tool emits only scalar/list
analysis; it never copies raw arrays into the repository.  The catalog is kept
in this durable docstring as well as enforced by the builder and analyzer.
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

from enodia.tt.bench.telemetry import parse_power_trace

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


def _failure(failures: list[str], checks: dict[str, Any], name: str, ok: bool, reason: str) -> None:
    checks[name] = {"ok": bool(ok), "reason": reason}
    if not ok:
        failures.append(reason)


def _declared_power_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = record.get("power_clock_provenance")
    if isinstance(provenance, Mapping):
        return dict(provenance)
    return {
        "file": record.get("power_trace"),
        "sample_count": record.get("power_trace_sample_count"),
        "csv_row_count": record.get("power_trace_csv_row_count"),
        "valid_row_count": record.get("power_trace_valid_row_count"),
        "in_run_valid_row_count": record.get("power_trace_in_run_valid_row_count"),
        "sha256": record.get("power_trace_sha256"),
        "coverage_complete": record.get("power_trace_coverage", {}).get("complete")
        if isinstance(record.get("power_trace_coverage"), Mapping)
        else None,
        "run_start": record.get("run_start"),
        "run_end": record.get("run_end"),
        "aiclk_source": None,
    }


def _validate_pair_catalog(
    record: Mapping[str, Any],
    *,
    timestamps: list[int] | None,
    failures: list[str],
    checks: dict[str, Any],
) -> None:
    outlier = record.get("outlier_analysis")
    if not isinstance(outlier, Mapping):
        _failure(failures, checks, "pair_catalog", True, "pair catalog not present")
        return
    pair_count = outlier.get("pair_count")
    if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count < 0:
        _failure(failures, checks, "pair_catalog", False, "pair_count is not a non-negative integer")
        return
    starts = outlier.get("frame_start_indices", [])
    pairs = outlier.get("pairs", [])
    pair_sums = outlier.get("pair_sums")
    pair_failures: list[str] = []
    if not isinstance(starts, list):
        pair_failures.append("frame_start_indices is not a list")
        starts = []
    if len(starts) != pair_count:
        pair_failures.append("pair_count does not equal frame_start_indices length")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in starts):
        pair_failures.append("frame_start_indices must contain positive integers")
    if not isinstance(pairs, list):
        pair_failures.append("pairs is not a list")
        pairs = []
    if len(pairs) != pair_count:
        pair_failures.append("pair_count does not equal pair list length")
    if pair_sums is not None:
        if not isinstance(pair_sums, list):
            pair_failures.append("pair_sums is not a list")
            pair_sums = []
        if len(pair_sums) != pair_count:
            pair_failures.append("pair_count does not equal pair_sums length")
    duplicates = _duplicate_values(value for value in starts if isinstance(value, int))
    if duplicates:
        pair_failures.append(f"duplicate frame_start_indices: {duplicates!r}")
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            pair_failures.append(f"pair {index} is not an object")
            continue
        endpoints = pair.get("interval_end_frame_indices")
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            pair_failures.append(f"pair {index} endpoints are not a two-item list")
            continue
        if (
            index < len(starts)
            and isinstance(starts[index], int)
            and endpoints != [starts[index], starts[index] + 1]
        ):
            pair_failures.append(f"pair {index} endpoints do not match frame starts")
        if (
            pair_sums is not None
            and index < len(pair_sums)
            and pair.get("pair_sum_ticks") != pair_sums[index]
        ):
            pair_failures.append(f"pair {index} sum does not match pair_sums")
    positions = outlier.get("event_frame_positions")
    if positions is not None:
        expected_positions = [endpoint for pair in pairs if isinstance(pair, Mapping)
                              for endpoint in pair.get("interval_end_frame_indices", [])]
        if positions != expected_positions:
            pair_failures.append("event_frame_positions do not match pair endpoints")
    for field in ("event_elapsed_seconds", "event_elapsed_ticks", "event_elapsed_ticks_mod_period"):
        values = outlier.get(field)
        if values is not None and (not isinstance(values, list) or len(values) != pair_count * 2):
            pair_failures.append(f"{field} length does not equal pair endpoint count")
    gaps = outlier.get("frame_gaps")
    if gaps is not None:
        expected_gaps = [right - left for left, right in itertools.pairwise(starts)]
        if gaps != expected_gaps:
            pair_failures.append("frame_gaps do not match ordered frame starts")

    correspondence: dict[str, Any] | None = None
    if timestamps is not None:
        for index, pair in enumerate(pairs):
            if not isinstance(pair, Mapping):
                continue
            endpoints = pair.get("interval_end_frame_indices")
            if (
                not isinstance(endpoints, list)
                or len(endpoints) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in endpoints)
                or endpoints[0] < 1
                or endpoints[1] >= len(timestamps)
            ):
                continue
            first_ticks = timestamps[endpoints[0]] - timestamps[endpoints[0] - 1]
            second_ticks = timestamps[endpoints[1]] - timestamps[endpoints[0]]
            if pair.get("pair_sum_ticks") is not None and pair["pair_sum_ticks"] != first_ticks + second_ticks:
                pair_failures.append(f"pair {index} sum does not match raw timestamps")
        try:
            correspondence = _record_correspondence(_detect_pairs(timestamps), record)
        except (TypeError, ValueError, KeyError) as exc:
            pair_failures.append(f"pair correspondence cannot be computed: {exc}")
        else:
            if not correspondence["exact_frame_start_match"]:
                pair_failures.append("derived pair starts do not match by length, multiset, and order")
            if not correspondence["record_pair_list_length_match"]:
                pair_failures.append("derived pair list length does not match record")
            if correspondence["pair_endpoint_mismatches"]:
                pair_failures.append("derived pair endpoints do not match record")
            if correspondence["pair_order_mismatches"]:
                pair_failures.append("derived pair orders do not match record")
            if correspondence["pair_value_mismatches"]:
                pair_failures.append("derived pair values do not match record")
    checks["pair_correspondence"] = correspondence
    if pair_failures:
        failures.extend(f"pair_catalog: {failure}" for failure in pair_failures)
    checks["pair_catalog"] = {
        "ok": not pair_failures,
        "reason": "pair catalog is internally and, when available, raw-stream consistent"
        if not pair_failures
        else "; ".join(pair_failures),
    }


def validate_record_invariants(
    record: Mapping[str, Any],
    *,
    raw_metadata: Mapping[str, Any] | None = None,
    timestamps: list[int] | None = None,
    power_trace_path: Path | None = None,
) -> dict[str, Any]:
    """Return a board-free report for the complete Issue #104 invariant catalog."""
    failures: list[str] = []
    checks: dict[str, Any] = {}
    if not isinstance(record, Mapping):
        return {"valid": False, "failures": ["record is not an object"], "checks": {}}

    raw = record.get("raw_timestamps")
    raw_count = raw.get("count") if isinstance(raw, Mapping) else None
    raw_hash = raw.get("sha256") if isinstance(raw, Mapping) else None
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
        _failure(failures, checks, "raw_count", False, "raw_timestamps.count is not a non-negative integer")
    elif timestamps is not None and raw_count != len(timestamps):
        _failure(failures, checks, "raw_count", False, "raw_timestamps.count does not equal timestamp rows")
    elif raw_metadata is not None and raw_count != raw_metadata.get("count"):
        _failure(failures, checks, "raw_count", False, "raw_timestamps.count does not equal actual raw bytes")
    else:
        _failure(failures, checks, "raw_count", True, "raw timestamp count matches available evidence")
    if (
        not isinstance(raw_hash, str)
        or len(raw_hash) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in raw_hash)
    ):
        _failure(failures, checks, "raw_hash", False, "raw_timestamps.sha256 is not a 64-character digest")
    elif raw_metadata is not None:
        if raw_hash != raw_metadata.get("sha256"):
            _failure(failures, checks, "raw_hash", False, "raw_timestamps.sha256 does not match actual raw bytes")
        else:
            _failure(failures, checks, "raw_hash", True, "raw timestamp SHA-256 matches actual raw bytes")
    else:
        _failure(failures, checks, "raw_hash", True, "raw timestamp SHA-256 is declared; bytes were not supplied")

    histogram = record.get("histogram")
    histogram_n = histogram.get("N") if isinstance(histogram, Mapping) else None
    if raw_count is not None and isinstance(histogram_n, int) and histogram_n != max(0, raw_count - 1):
        _failure(failures, checks, "histogram_n_plus_one", False, "histogram N must equal raw timestamp count minus one")
    elif timestamps is not None and isinstance(histogram_n, int) and histogram_n != max(0, len(timestamps) - 1):
        _failure(failures, checks, "histogram_n_plus_one", False, "histogram N must equal actual interval count")
    elif isinstance(histogram_n, int):
        _failure(failures, checks, "histogram_n_plus_one", True, "histogram N equals raw timestamp count minus one")
    else:
        _failure(failures, checks, "histogram_n_plus_one", False, "histogram.N is missing or invalid")
    if isinstance(histogram, Mapping):
        nested = histogram.get("histogram")
        bins = nested.get("bins") if isinstance(nested, Mapping) else None
        if isinstance(bins, list) and isinstance(histogram_n, int):
            bin_sum = sum(item.get("count", 0) for item in bins if isinstance(item, Mapping))
            if bin_sum != histogram_n:
                failures.append("histogram bins do not sum to histogram N")
                checks["histogram_bins"] = {"ok": False, "bin_sum": bin_sum, "N": histogram_n}
            else:
                checks["histogram_bins"] = {"ok": True, "bin_sum": bin_sum, "N": histogram_n}

    ring = record.get("ring")
    if isinstance(ring, Mapping):
        consumed = ring.get("consumed_frame_count")
        if isinstance(consumed, int) and raw_count is not None and consumed != raw_count:
            failures.append("ring consumed_frame_count does not equal raw timestamp count")
        if isinstance(consumed, int) and histogram_n is not None and histogram_n != max(0, consumed - 1):
            failures.append("ring consumed_frame_count does not equal histogram N plus one")

    environment = record.get("environment")
    sampler = environment.get("telemetry_sampler") if isinstance(environment, Mapping) else None
    mode = sampler.get("mode") if isinstance(sampler, Mapping) else None
    trace_name = record.get("power_trace")
    if mode == "off":
        sampler_ok = (
            trace_name is None
            and sampler.get("power_trace") == "absent_by_design"
            and record.get("power_trace_absent_reason") == "sampler_off_by_design"
        )
        _failure(
            failures,
            checks,
            "sampler_trace",
            sampler_ok,
            "sampler-off trace is absent by design"
            if sampler_ok
            else "sampler-off mode requires no trace and absent-by-design reason",
        )
    elif mode in {"default", "explicit"}:
        sampler_ok = (
            isinstance(trace_name, str)
            and bool(trace_name.strip())
            and sampler.get("power_trace") == "required"
            and Path(trace_name).name == trace_name
        )
        _failure(
            failures,
            checks,
            "sampler_trace",
            sampler_ok,
            "sampled sampler mode names a power trace"
            if sampler_ok
            else "sampled sampler mode requires a power trace filename",
        )
    else:
        checks["sampler_trace"] = {"ok": True, "reason": "sampler metadata is legacy or absent"}

    declared_power = _declared_power_metadata(record)
    power_report: dict[str, Any] | None = None
    provenance = record.get("power_clock_provenance")
    if isinstance(provenance, Mapping):
        duplicate_fields = (
            ("file", "power_trace"),
            ("sample_count", "power_trace_sample_count"),
            ("csv_row_count", "power_trace_csv_row_count"),
            ("valid_row_count", "power_trace_valid_row_count"),
            ("in_run_valid_row_count", "power_trace_in_run_valid_row_count"),
            ("sha256", "power_trace_sha256"),
        )
        for nested_name, top_level_name in duplicate_fields:
            nested_value = provenance.get(nested_name)
            top_level_value = record.get(top_level_name)
            if nested_value is not None and top_level_value is not None and nested_value != top_level_value:
                label = "SHA-256" if nested_name == "sha256" else nested_name
                failures.append(
                    f"power provenance {label} does not match {top_level_name}"
                )
    if mode in {"default", "explicit"} and trace_name:
        if power_trace_path is None:
            failures.append("power trace actual bytes are unavailable for SHA-256 and row validation")
            checks["power_trace"] = {"ok": False, "reason": "power_trace_bytes_unavailable"}
        else:
            power_report = parse_power_trace(
                power_trace_path,
                run_start=declared_power.get("run_start") or record.get("run_start"),
                run_end=declared_power.get("run_end") or record.get("run_end"),
            )
            power_failures: list[str] = []
            if power_report.get("file") != trace_name:
                power_failures.append("trace filename does not match record")
            if declared_power.get("sha256") is not None and declared_power.get("sha256") != power_report.get("sha256"):
                power_failures.append("power trace SHA-256 does not match actual bytes")
            if not isinstance(declared_power.get("sample_count"), int):
                power_failures.append("power trace sample count is not declared")
            elif declared_power.get("sample_count") != power_report.get("sample_count"):
                power_failures.append("power trace sample count does not match CSV rows")
            declared_valid = declared_power.get("valid_row_count")
            if declared_valid is None:
                power_failures.append("power trace valid row count is not declared")
            if declared_valid is not None and declared_valid != power_report.get("valid_row_count"):
                power_failures.append("power trace valid row count does not match actual rows")
            declared_csv_rows = declared_power.get("csv_row_count")
            if declared_csv_rows is not None and declared_csv_rows != power_report.get("csv_row_count"):
                power_failures.append("power trace CSV row count does not match actual rows")
            declared_in_run = declared_power.get("in_run_valid_row_count")
            if declared_in_run is not None and declared_in_run != power_report.get("in_run_valid_row_count"):
                power_failures.append("power trace in-run row count does not match actual rows")
            for field in ("readable", "nonempty", "timestamps_parse", "timestamps_ordered"):
                if power_report.get(field) is not True:
                    power_failures.append(f"power trace {field} is not true")
            if power_report.get("coverage_complete") is not True:
                power_failures.append("power trace coverage_complete is not true")
            checks["power_trace"] = {
                "ok": not power_failures,
                "reason": "power trace bytes, rows, timestamps, and coverage are valid"
                if not power_failures
                else "; ".join(power_failures),
                "metadata": power_report,
            }
            failures.extend(f"power_trace: {failure}" for failure in power_failures)
    elif mode == "off":
        checks["power_trace"] = {"ok": True, "reason": "sampler-off has no power trace by design"}

    timing = record.get("timing_evidence") is True
    aiclk_source = declared_power.get("aiclk_source")
    if timing:
        timing_failures: list[str] = []
        if mode == "off":
            timing_failures.append("sampler-off records are diagnostic-only")
        if aiclk_source != "run_trace_samples":
            timing_failures.append("timing AICLK is not sourced from valid in-run trace samples")
        if power_report is not None:
            if power_report.get("coverage_complete") is not True:
                timing_failures.append("timing trace coverage is incomplete")
            if power_report.get("in_run_valid_row_count", 0) < 1:
                timing_failures.append("timing trace has no valid in-run rows")
            if power_report.get("aiclk_source") != "run_trace_samples":
                timing_failures.append("timing trace AICLK samples are not in-run")
        if timing_failures:
            failures.extend(f"timing_evidence: {failure}" for failure in timing_failures)
            checks["timing_evidence"] = {"ok": False, "reason": "; ".join(timing_failures)}
        else:
            checks["timing_evidence"] = {"ok": True, "reason": "AICLK comes from complete in-run trace samples"}
    else:
        reason = record.get("timing_evidence_reason")
        if mode in {"default", "explicit"} and not isinstance(reason, str):
            failures.append("timing_evidence: sampled record is false without an explicit reason")
            checks["timing_evidence"] = {
                "ok": False,
                "reason": "sampled record is false without an explicit reason",
            }
        else:
            checks["timing_evidence"] = {
                "ok": True,
                "reason": reason
                or record.get("power_trace_absent_reason")
                or "timing evidence is false",
            }

    _validate_pair_catalog(record, timestamps=timestamps, failures=failures, checks=checks)
    return {
        "valid": not failures,
        "failures": failures,
        "checks": checks,
        "power_trace": power_report,
    }


# Descriptive aliases make the board-free validation seam discoverable to callers.
validate_record_power_invariants = validate_record_invariants
validate_issue104_record = validate_record_invariants


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
    invariant_validation = validate_record_invariants(
        record, raw_metadata=raw_metadata, timestamps=timestamps
    )
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
        "invariant_validation": invariant_validation,
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
    analysis = analyze_run(
        timestamps,
        raw_metadata=metadata,
        record=record,
        aiclk_mhz=aiclk_mhz,
        phase_window_ms=phase_window_ms,
    )
    trace_name = record.get("power_trace") if isinstance(record, Mapping) else None
    trace_path = record_path.parent / trace_name if isinstance(trace_name, str) else None
    analysis["invariant_validation"] = validate_record_invariants(
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
