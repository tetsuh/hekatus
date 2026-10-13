"""Generate the board-free Issue #122 AICLK audit from committed measurements.

The analyzer intentionally reads only ``docs/measurements`` (or the directory
specified by ``--measurements-root``).  It never opens a device or invokes an
external command.  The pure helpers are also used by the board-free tests.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UTC = dt.UTC
POWER_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")
AICLK_ENDPOINTS = frozenset((800.0, 1350.0))
INTERMEDIATE_POSITION_ORDER = ("beginning", "middle", "end")
_CONFIGURED_BUDGET_AICLK_LABEL = "configured budget AICLK"
# This absolute tolerance covers only floating-point serialization differences
# between the CSV recomputation and persisted JSON, not a changed clock basis.
_EXACT_INTERVAL_AICLK_TOLERANCE_MHZ = 1e-9
_EXACT_INTERVAL_AICLK_VALUE_FIELDS = frozenset(
    {
        "aiclk_mhz_for_elapsed_seconds",
        "aiclk_mhz_load_interval_weighted",
        "load_interval_aiclk_mhz",
        "actual_duration_aiclk_mhz",
    }
)
_EXACT_INTERVAL_AICLK_SOURCE_FIELDS = frozenset(
    {
        "aiclk_source_for_elapsed_seconds",
        "aiclk_mhz_load_interval_weighted_source",
        "load_interval_aiclk_source",
        "actual_duration_aiclk_source",
    }
)
_EXACT_INTERVAL_AICLK_SELECTION_FIELDS = frozenset(
    {
        "aiclk_selection_for_elapsed_seconds",
        "load_interval_aiclk_selection",
        "actual_duration_aiclk_selection",
    }
)
_SEMANTIC_SOURCE_FIELDS = frozenset(
    {
        "aiclk_mhz_for_elapsed_seconds",
        "aiclk_source_for_elapsed_seconds",
        "aiclk_selection_for_elapsed_seconds",
        "aiclk_mhz_load_interval_weighted",
        "aiclk_mhz_load_interval_weighted_source",
        "load_interval_aiclk_mhz",
        "load_interval_aiclk_source",
        "load_interval_aiclk_selection",
        "actual_duration_aiclk_mhz",
        "actual_duration_aiclk_source",
        "actual_duration_aiclk_selection",
        "predicted_load_time_aiclk_mhz",
        "predicted_load_time_aiclk_source",
        "predicted_load_time_aiclk_selection",
    }
)
_SEMANTIC_DERIVED_FIELDS = frozenset(
    {
        "run_elapsed_seconds",
        "event_elapsed_seconds",
        "pair_elapsed_seconds",
        "sampler_gap_seconds",
        "actual_duration_seconds",
        "predicted_load_time_seconds",
    }
)
_FIXED_REFERENCE_FIELDS = frozenset(
    {"reference_aiclk_mhz", "throughput_reference_aiclk_mhz"}
)
_FIXED_REFERENCE_METRIC_FIELDS = frozenset(
    {"achieved_throughput", "repeats_per_card_second"}
)
_CONFIGURED_AICLK_FIELDS = frozenset({"budget_aiclk_mhz"})
_HOST_TIMING_FIELDS = frozenset(
    {
        "seconds_per_iteration",
        "seconds_per_launch_min",
        "seconds_per_launch_p50",
        "seconds_per_launch_p99",
        "seconds_per_launch_p99_9",
        "tflops_min_derived",
        "tflops_p50_derived",
        "tflops_p99_derived",
        "tflops_p99_9_derived",
        "tflops_fastest_launch_derived",
        "achieved_tflops",
        "tflops_derivation",
    }
)




@dataclass(frozen=True)
class TraceSample:
    """One valid timestamp/AICLK row from a committed CSV."""

    timestamp: dt.datetime
    aiclk_mhz: float
    row_number: int


@dataclass(frozen=True)
class _LeftHeldSegment:
    """One positive-duration left-held segment clipped to an interval."""

    row_index: int
    timestamp: dt.datetime
    aiclk_mhz: float
    start: dt.datetime
    end: dt.datetime
    seconds: float


@dataclass(frozen=True)
class TraceData:
    """Parsed CSV facts retained for deterministic report generation."""

    path: Path
    samples: tuple[TraceSample, ...]
    csv_row_count: int
    invalid_row_count: int
    sha256: str
    headers: tuple[str, ...]

    @property
    def timestamps_ordered(self) -> bool:
        return all(
            left.timestamp <= right.timestamp
            for left, right in zip(self.samples, self.samples[1:])
        )


@dataclass(frozen=True)
class AiclkClassification:
    """Classification of the valid AICLK values in one CSV."""

    label: str
    unique_mhz: tuple[float, ...]
    minimum_intermediate_mhz: float | None
    intermediate_position: str | None

    @property
    def kind(self) -> str:
        """Alias that keeps the classification convenient for callers."""

        return self.label


@dataclass(frozen=True)
class TraceResolution:
    """The committed CSV selected for one JSON record."""

    path: Path | None
    method: str
    reference: str | None
    sha256: str | None


@dataclass(frozen=True)
class IntervalBounds:
    """Timestamp bounds and their provenance inside one JSON record."""

    start: dt.datetime
    end: dt.datetime
    kind: str
    source: str


@dataclass(frozen=True)
class RecordAnalysis:
    """All reportable facts for one committed JSON record."""

    path: Path
    issue: str
    status: str
    resolution: TraceResolution
    trace: TraceData | None
    classification: AiclkClassification | None
    interval: IntervalBounds | None
    evidence: str
    minimum_mhz: float | None
    maximum_mhz: float | None
    weighted_mhz: float | None
    weighted_is_exact: bool
    timing_fields: bool
    conversion_fields: bool
    source_tag: str


@dataclass(frozen=True)
class ConversionImpact:
    """Per-record AICLK conversion facts retained by the audit."""

    category: str
    metrics: str
    selected_value: str
    conclusion_change: str


@dataclass(frozen=True)
class RowTimingBound:
    """Measured row duration and its order-derived start lower bound."""

    name: str
    status: str
    sample_count: int | None
    sample_seconds: float | None
    start_offset_seconds: float | None
    start_timestamp: dt.datetime | None


@dataclass(frozen=True)
class HoldSegment:
    """A low-clock positive-duration segment inside an explicit run interval."""

    aiclk_mhz: float
    start_offset_seconds: float
    end_offset_seconds: float
    start: dt.datetime
    end: dt.datetime


def parse_timestamp(value: str) -> dt.datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value!r}")
    return parsed.astimezone(UTC)


def _sample_parts(row: TraceSample | Sequence[Any] | Mapping[str, Any]) -> tuple[dt.datetime, float]:
    if isinstance(row, TraceSample):
        return row.timestamp, row.aiclk_mhz
    if isinstance(row, Mapping):
        timestamp = row.get("timestamp") or row.get("timestamp_utc")
        value = row.get("aiclk_mhz")
    else:
        if len(row) < 2:
            raise ValueError("a sample must contain timestamp and AICLK")
        timestamp, value = row[0], row[1]
    if isinstance(timestamp, str):
        timestamp = parse_timestamp(timestamp)
    if not isinstance(timestamp, dt.datetime):
        raise TypeError("sample timestamp must be datetime or ISO text")
    if timestamp.tzinfo is None:
        raise ValueError("sample timestamp has no timezone")
    return timestamp.astimezone(UTC), float(value)


def left_held_time_weighted_aiclk(
    rows: Sequence[TraceSample | Sequence[Any] | Mapping[str, Any]],
    start: dt.datetime,
    end: dt.datetime,
) -> float | None:
    """Return a left-held AICLK average over an explicitly covered interval.

    A sample owns the interval from its timestamp up to (but not including) the
    next sample.  The first usable row must be at or before ``start`` and the
    last usable row must be at or after ``end``; refusing extrapolation is what
    separates exact evidence from a CSV-capture diagnostic.
    """

    if start.tzinfo is None or end.tzinfo is None or start > end or start == end:
        return None
    samples = [_sample_parts(row) for row in rows]
    if not samples or any(
        left[0] > right[0] for left, right in itertools.pairwise(samples)
    ):
        return None
    if samples[0][0] > start or samples[-1][0] < end:
        return None
    duration = (end - start).total_seconds()
    segments = _left_held_positive_segments(rows, start, end)
    weighted = sum(segment.aiclk_mhz * segment.seconds for segment in segments)
    covered = sum(segment.seconds for segment in segments)
    if not math.isclose(covered, duration, rel_tol=0.0, abs_tol=1e-9):
        return None
    return weighted / covered


def _left_held_positive_segments(
    rows: Sequence[TraceSample | Sequence[Any] | Mapping[str, Any]],
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[_LeftHeldSegment, ...]:
    """Return the positive-duration left-held segments in ``[start, end]``."""

    if start.tzinfo is None or end.tzinfo is None or start >= end:
        return ()
    samples = [_sample_parts(row) for row in rows]
    if any(
        left[0] > right[0] for left, right in itertools.pairwise(samples)
    ):
        return ()
    segments: list[_LeftHeldSegment] = []
    for index, (timestamp, aiclk) in enumerate(samples):
        segment_start = max(timestamp, start)
        next_timestamp = samples[index + 1][0] if index + 1 < len(samples) else end
        segment_end = min(next_timestamp, end)
        seconds = (segment_end - segment_start).total_seconds()
        if seconds > 0.0:
            segments.append(
                _LeftHeldSegment(
                    index,
                    timestamp,
                    aiclk,
                    segment_start,
                    segment_end,
                    seconds,
                )
            )
    return tuple(segments)


# Short name for callers that prefer the terminology used by the report.
time_weighted_aiclk = left_held_time_weighted_aiclk


def classify_intermediate_position(indices: int | Iterable[int], length: int) -> str | None:
    """Classify an index (or indices) in the beginning, middle, or end thirds."""

    if length <= 0:
        return None
    if isinstance(indices, int):
        positions = (indices,)
    else:
        positions = tuple(indices)
    if not positions:
        return None
    labels: set[str] = set()
    for index in positions:
        if index < 0 or index >= length:
            raise ValueError("sample index is outside the CSV")
        fraction = index / length
        if fraction < 1.0 / 3.0:
            labels.add("beginning")
        elif fraction < 2.0 / 3.0:
            labels.add("middle")
        else:
            labels.add("end")
    return "+".join(label for label in INTERMEDIATE_POSITION_ORDER if label in labels)


def classify_aiclk_values(values: Sequence[float]) -> AiclkClassification:
    """Classify endpoint-only versus intermediate AICLK samples."""

    unique = tuple(sorted({float(value) for value in values}))
    if not unique:
        return AiclkClassification("no valid AICLK values", (), None, None)
    intermediates = [value for value in unique if value not in AICLK_ENDPOINTS]
    if not intermediates and set(unique) == set(AICLK_ENDPOINTS):
        return AiclkClassification(
            "two-valued {800,1350}", unique, None, None
        )
    if intermediates:
        minimum = min(intermediates)
        minimum_indices = [index for index, value in enumerate(values) if value == minimum]
        # Report the first occurrence when a minimum spans more than one third;
        # the report asks for one beginning/middle/end location, not a range.
        return AiclkClassification(
            "contains intermediate values",
            unique,
            minimum,
            classify_intermediate_position(minimum_indices[0], len(values)),
        )
    if len(unique) == 1:
        return AiclkClassification(
            f"single-valued {_format_number(unique[0])}", unique, None, None
        )
    return AiclkClassification("other endpoint values", unique, None, None)


def _parse_aiclk(value: Any) -> float:
    parsed = float(str(value).strip())
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("AICLK is not finite and positive")
    return parsed


def read_trace(path: Path) -> TraceData:
    """Read one CSV without changing it."""

    samples: list[TraceSample] = []
    csv_rows = 0
    invalid_rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = tuple(reader.fieldnames or ())
        for row_number, row in enumerate(reader, start=2):
            csv_rows += 1
            try:
                timestamp = parse_timestamp(str(row["timestamp_utc"]))
                aiclk = _parse_aiclk(row["aiclk_mhz"])
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
                continue
            samples.append(TraceSample(timestamp, aiclk, row_number))
    return TraceData(
        path=path,
        samples=tuple(samples),
        csv_row_count=csv_rows,
        invalid_row_count=invalid_rows,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        headers=headers,
    )


def _walk_strings(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            if isinstance(child, str):
                yield child_path, child
            yield from _walk_strings(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, (*path, str(index)))


def _candidate_priority(path: tuple[str, ...]) -> int:
    joined = ".".join(path).lower()
    leaf = path[-1].lower()
    if leaf == "power_trace":
        return 0
    if "power_clock_provenance" in joined and leaf in {"trace", "file", "source_file"}:
        return 1
    if "power_snapshot" in joined:
        return 2
    if leaf in {"trace", "file", "source_file"} and "power" in joined:
        return 3
    return 4


def power_trace_references(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return CSV references ordered by provenance priority, not filename stem."""

    candidates: list[tuple[int, tuple[str, ...], str]] = []
    for path, value in _walk_strings(record):
        if not value.lower().endswith(".csv"):
            continue
        joined = ".".join(path).lower()
        if not any(token in joined for token in ("power", "trace", "snapshot")):
            continue
        candidates.append((_candidate_priority(path), path, value))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    result: list[str] = []
    for _priority, _path, value in candidates:
        if value not in result:
            result.append(value)
    return tuple(result)


def _trace_sha256(record: Mapping[str, Any]) -> str | None:
    candidates: list[tuple[int, str]] = []
    for path, value in _walk_strings(record):
        leaf = path[-1].lower()
        if leaf not in {"power_trace_sha256", "power_snapshot_sha256", "trace_sha256", "sha256"}:
            continue
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            continue
        joined = ".".join(path).lower()
        if "power" not in joined and "trace" not in joined and "snapshot" not in joined:
            continue
        candidates.append((_candidate_priority(path), value.lower()))
    if not candidates:
        return None
    return min(candidates)[1]


def _path_from_reference(root: Path, reference: str) -> Path:
    value = Path(reference)
    # External artifact paths are intentionally reduced to a corpus basename;
    # the audit never reaches outside the committed measurement directory.
    return root / value.name


def _resolve_explicit_trace(
    root: Path,
    references: Sequence[str],
    expected_hash: str | None,
) -> TraceResolution | None:
    for reference in references:
        candidate = _path_from_reference(root, reference)
        if candidate.is_file():
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if expected_hash is None or actual == expected_hash:
                return TraceResolution(
                    candidate,
                    "explicit reference",
                    reference,
                    expected_hash or actual,
                )
    return None


def _resolve_hash_trace(
    csv_by_hash: Mapping[str, tuple[Path, ...]],
    references: Sequence[str],
    expected_hash: str | None,
) -> TraceResolution | None:
    if expected_hash and expected_hash in csv_by_hash:
        return TraceResolution(
            min(csv_by_hash[expected_hash]),
            "SHA-256 provenance",
            references[0] if references else None,
            expected_hash,
        )
    return None


def _record_trace_stems(record_path: Path) -> list[str]:
    stems = [record_path.stem]
    for suffix in ("-summary", "-l1-accounting"):
        if stems[0].endswith(suffix):
            stems.append(stems[0][: -len(suffix)])
    return stems


def _resolve_stem_trace(
    root: Path,
    record_path: Path,
    references: Sequence[str],
    expected_hash: str | None,
) -> TraceResolution | None:
    for stem in _record_trace_stems(record_path):
        candidate = root / f"{stem}-power.csv"
        if not candidate.is_file():
            continue
        if expected_hash is not None:
            actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                continue
        return TraceResolution(
            candidate,
            "record-stem fallback",
            references[0] if references else None,
            expected_hash,
        )
    return None


def resolve_trace(
    root: Path,
    record_path: Path,
    record: Mapping[str, Any],
    csv_by_hash: Mapping[str, tuple[Path, ...]],
) -> TraceResolution:
    """Resolve a trace by explicit reference, hash provenance, then safe fallback."""

    references = power_trace_references(record)
    expected_hash = _trace_sha256(record)
    resolution = _resolve_explicit_trace(root, references, expected_hash)
    if resolution is not None:
        return resolution
    resolution = _resolve_hash_trace(csv_by_hash, references, expected_hash)
    if resolution is not None:
        return resolution
    resolution = _resolve_stem_trace(root, record_path, references, expected_hash)
    if resolution is not None:
        return resolution
    return TraceResolution(
        None,
        "unresolved" if references else "no reference",
        references[0] if references else None,
        expected_hash,
    )


def _mapping_interval(
    value: Mapping[str, Any],
    lower_keys: Mapping[str, Any],
    path: tuple[str, ...],
    start_key: str,
    end_key: str,
    kind: str,
    rank: int,
) -> tuple[int, IntervalBounds] | None:
    if start_key not in lower_keys or end_key not in lower_keys:
        return None
    try:
        start = parse_timestamp(str(value[lower_keys[start_key]]))
        end = parse_timestamp(str(value[lower_keys[end_key]]))
    except (TypeError, ValueError):
        return None
    joined = ".".join(path).lower()
    if "power_clock_provenance" in joined or "power_snapshot" in joined:
        rank -= 1
    return rank, IntervalBounds(start, end, kind, ".".join((*path, start_key, end_key)))


def _mapping_intervals(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[int, IntervalBounds]]:
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _mapping_intervals(child, (*path, str(index)))
        return
    if not isinstance(value, Mapping):
        return
    lower_keys = {str(key).lower(): key for key in value}
    for start_key, end_key, kind, rank in (
        ("load_interval_start", "load_interval_end", "load", 0),
        ("run_start", "run_end", "run", 1),
    ):
        interval = _mapping_interval(
            value, lower_keys, path, start_key, end_key, kind, rank
        )
        if interval is not None:
            yield interval
    for key, child in value.items():
        yield from _mapping_intervals(child, (*path, str(key)))


def find_interval(record: Mapping[str, Any]) -> IntervalBounds | None:
    """Find the highest-provenance explicit wall-time interval in a record."""

    candidates = list(_mapping_intervals(record))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].source))
    return candidates[0][1]


def _coverage_rows(trace: TraceData, interval: IntervalBounds) -> tuple[list[TraceSample], bool]:
    """Return samples owning positive-duration pieces of an explicit interval.

    This deliberately follows :func:`left_held_time_weighted_aiclk`: a row
    before the interval start is included when it holds through that start,
    while a row timestamped exactly at the interval end owns no duration and
    is excluded.  Capture-span diagnostics do not use this helper.
    """

    samples = list(trace.samples)
    segments = _left_held_positive_segments(samples, interval.start, interval.end)
    inside = [samples[segment.row_index] for segment in segments]
    complete = (
        bool(samples)
        and trace.invalid_row_count == 0
        and trace.timestamps_ordered
        and samples[0].timestamp <= interval.start
        and samples[-1].timestamp >= interval.end
    )
    return inside, complete


def _recursive_has_key(value: Any, keys: set[str]) -> bool:
    if isinstance(value, Mapping):
        if any(str(key).lower() in keys for key in value):
            return True
        return any(_recursive_has_key(child, keys) for child in value.values())
    if isinstance(value, list):
        return any(_recursive_has_key(child, keys) for child in value)
    return False


def _source_tag(record: Mapping[str, Any]) -> str:
    text = json.dumps(record, sort_keys=True).lower()
    if "run_matmul.py" in text:
        return "run_matmul.py"
    if "newton_schulz_issue88.py" in text:
        return "newton_schulz_issue88.py"
    if "newton_schulz_runner.py" in text:
        return "newton_schulz_runner.py (record reference)"
    if "issue101_combined.py" in text:
        return "newton_schulz_issue101_combined.py"
    if "newton_schulz" in text:
        return "other Newton-Schulz source"
    return "not stated"


def _issue_label(record_path: Path, record: Mapping[str, Any]) -> str:
    filename_match = re.search(r"issue(\d+)", record_path.name, re.IGNORECASE)
    if filename_match:
        return f"#{filename_match.group(1)}"
    explicit: list[str] = []
    for path, value in _walk_strings(record):
        if path[-1].lower() != "issue":
            continue
        if value.startswith("#") and value[1:].isdigit():
            explicit.append(value)
        elif value.isdigit():
            explicit.append(f"#{value}")
    for key, value in record.items():
        if str(key).lower() == "issue" and isinstance(value, int) and not isinstance(value, bool):
            explicit.append(f"#{value}")
    return min(set(explicit), key=lambda item: (len(item), item)) if explicit else "—"


def _status(record: Mapping[str, Any]) -> str:
    for key in ("status",):
        value = record.get(key)
        if isinstance(value, str):
            return value
    for container_name in ("measurement", "load", "result", "performance", "correctness"):
        container = record.get(container_name)
        if isinstance(container, Mapping) and isinstance(container.get("status"), str):
            return str(container["status"])
    return "—"


def _format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return f"{value:.{digits}f}"


def _format_values(values: Sequence[float]) -> str:
    return ", ".join(_format_number(value) for value in values) if values else "—"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _record_field_values(
    record: Mapping[str, Any], field_names: frozenset[str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return deterministic scalar values for selected record field names."""

    values: dict[str, set[str]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key).lower()
                if name in field_names and not isinstance(child, (Mapping, list)):
                    if child is None:
                        rendered = "—"
                    elif isinstance(child, bool):
                        rendered = str(child).lower()
                    elif isinstance(child, (int, float)):
                        rendered = _format_number(float(child))
                    else:
                        rendered = str(child)
                    values.setdefault(name, set()).add(rendered)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(record)
    return tuple(
        (name, tuple(sorted(field_values)))
        for name, field_values in sorted(values.items())
    )


def _field_inventory(
    values: Sequence[tuple[str, Sequence[str]]], include_values: bool = True
) -> str:
    """Render selected record fields without exposing repeated nested paths."""

    rendered: list[str] = []
    for name, field_values in values:
        if not include_values:
            rendered.append(f"`{name}`")
            continue
        rendered_values = ", ".join(field_values)
        rendered.append(f"`{name}={rendered_values}`")
    return ", ".join(rendered) if rendered else "—"


def _record_scalar_fields(
    record: Mapping[str, Any], field_names: frozenset[str]
) -> tuple[tuple[str, Any], ...]:
    """Return scalar semantic fields with paths and unformatted values."""

    values: list[tuple[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = (*path, str(key))
                if (
                    str(key).lower() in field_names
                    and not isinstance(child, (Mapping, list))
                ):
                    values.append((".".join(child_path), child))
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(record)
    return tuple(sorted(values))


def _format_aiclk_metadata_value(value: Any) -> str:
    if value is None:
        return "—"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(parsed):
        return str(value)
    return _format_number(parsed, digits=9)


def _exact_interval_aiclk_mismatch_details(
    analysis: RecordAnalysis, record: Mapping[str, Any]
) -> tuple[tuple[tuple[str, Any], ...], tuple[str, ...]]:
    """Compare exact trace weighting with persisted semantic AICLK metadata."""

    assert analysis.weighted_mhz is not None
    persisted_values = _record_scalar_fields(record, _EXACT_INTERVAL_AICLK_VALUE_FIELDS)
    persisted_sources = _record_scalar_fields(record, _EXACT_INTERVAL_AICLK_SOURCE_FIELDS)
    persisted_selections = _record_scalar_fields(
        record, _EXACT_INTERVAL_AICLK_SELECTION_FIELDS
    )
    mismatches: list[str] = []
    if not persisted_values:
        mismatches.append("no persisted weighted/elapsed AICLK value is present")
    for path, raw_value in persisted_values:
        try:
            persisted_value = float(raw_value)
        except (TypeError, ValueError):
            mismatches.append(
                f"`{path}` is not a finite AICLK value ({_format_aiclk_metadata_value(raw_value)})"
            )
            continue
        if not math.isfinite(persisted_value) or not math.isclose(
            persisted_value,
            analysis.weighted_mhz,
            rel_tol=0.0,
            abs_tol=_EXACT_INTERVAL_AICLK_TOLERANCE_MHZ,
        ):
            discrepancy = persisted_value - analysis.weighted_mhz
            mismatches.append(
                f"`{path}` persisted {_format_aiclk_metadata_value(raw_value)} MHz, "
                f"a discrepancy of {discrepancy:+.9f} MHz"
            )

    expected_source = "load_interval_trace_samples"
    for path, raw_source in persisted_sources:
        if raw_source != expected_source:
            mismatches.append(
                f"`{path}` is {_format_aiclk_metadata_value(raw_source)!r}, "
                f"not {expected_source!r}"
            )
    expected_selection = "time_weighted_load_interval"
    for path, raw_selection in persisted_selections:
        if raw_selection != expected_selection:
            mismatches.append(
                f"`{path}` is {_format_aiclk_metadata_value(raw_selection)!r}, "
                f"not {expected_selection!r}"
            )
    return persisted_values, tuple(mismatches)


def _exact_interval_selection_statement(
    analysis: RecordAnalysis, record: Mapping[str, Any]
) -> str:
    """Describe whether persisted exact-interval metadata matches the trace."""

    assert analysis.weighted_mhz is not None
    persisted_values, mismatches = _exact_interval_aiclk_mismatch_details(
        analysis, record
    )
    if not mismatches:
        return (
            f"Selected {_format_number(analysis.weighted_mhz)} MHz from the exact "
            f"{analysis.interval.kind} interval; the persisted semantic selection "
            "is the same measured conversion basis."
        )

    persisted_text = "; ".join(
        f"`{path}={_format_aiclk_metadata_value(value)} MHz`"
        for path, value in persisted_values
    ) or "—"
    discrepancy_text = "; ".join(mismatches)
    return (
        f"Computed {_format_aiclk_metadata_value(analysis.weighted_mhz)} MHz from the "
        f"exact {analysis.interval.kind} interval, but persisted weighted/elapsed "
        f"AICLK value(s) are {persisted_text}; discrepancy: {discrepancy_text}. "
        "The persisted selection does not match the exact interval basis and is not "
        "verifiable as corrected."
    )


def _trace_weighted_statement(analysis: RecordAnalysis) -> str:
    if analysis.weighted_mhz is not None:
        evidence = "exact" if analysis.weighted_is_exact else "capture-span diagnostic only"
        return f"{_format_number(analysis.weighted_mhz)} MHz ({evidence})"
    if analysis.interval is not None:
        return f"not verifiable from retained interval evidence ({analysis.evidence})"
    return "not verifiable from retained interval evidence (no explicit interval bounds)"


def _fixed_policy_statement(record: Mapping[str, Any]) -> str:
    fixed = _record_field_values(record, _FIXED_REFERENCE_FIELDS)
    configured = _record_field_values(record, _CONFIGURED_AICLK_FIELDS)
    statements: list[str] = []
    if fixed:
        statements.append(
            f"{_field_inventory(fixed)} is fixed 1350-MHz normalization, not observed AICLK"
        )
    if configured:
        statements.append(
            f"{_field_inventory(configured)} is configured budget policy, not measured elapsed time"
        )
    return "; ".join(statements)


def _record_date(path: Path) -> str:
    return path.name[:10]


def _uses_host_perf_counter(
    analysis: RecordAnalysis, record: Mapping[str, Any]
) -> bool:
    if not analysis.timing_fields or analysis.conversion_fields:
        return False
    if analysis.issue == "#88":
        return True
    if _record_field_values(record, frozenset({"tflops_derivation"})):
        return True
    if analysis.source_tag in {
        "run_matmul.py",
        "newton_schulz_issue88.py",
        "newton_schulz_issue101_combined.py",
    }:
        return True
    return (
        "2026-09-27" <= _record_date(analysis.path) <= "2026-10-06"
        and "newton-schulz" in analysis.path.name
        and "issue88" not in analysis.path.name
    )


def _conclusion_change_statement(
    analysis: RecordAnalysis, record: Mapping[str, Any]
) -> str:
    if analysis.issue == "#88":
        return (
            "No; the existing #88 quality/L1 and row-order clock-confounding "
            "conclusions are unchanged."
        )
    if analysis.issue == "#118":
        return (
            "No; the existing #118 weighted comparisons and conclusion are "
            "unchanged."
        )
    if analysis.issue == "#104":
        return (
            "No; the retained #104 conclusion (no causal sampler hypothesis) "
            "is unchanged."
        )
    conclusion_fields = _record_field_values(
        record, frozenset({"conclusion", "headline_conclusion"})
    )
    if conclusion_fields:
        return "No; the retained conclusion is unchanged."
    if analysis.status.startswith("blocked") or analysis.status == "fail":
        return "No published conclusion is retained in this blocked/failed record."
    return "No published conclusion is retained; no change is asserted."


def _record_conversion_impact(
    analysis: RecordAnalysis, record: Mapping[str, Any]
) -> ConversionImpact:
    semantic_source = _record_field_values(record, _SEMANTIC_SOURCE_FIELDS)
    semantic_derived = _record_field_values(record, _SEMANTIC_DERIVED_FIELDS)
    semantic = tuple(sorted((*semantic_source, *semantic_derived)))
    fixed_policy = _fixed_policy_statement(record)
    configured = _record_field_values(record, _CONFIGURED_AICLK_FIELDS)
    if semantic_source:
        category_parts = ["persisted semantic conversion"]
        if _record_field_values(record, _FIXED_REFERENCE_FIELDS):
            category_parts.append("fixed 1350-MHz normalization")
        if configured:
            category_parts.append("configured-budget policy")
        metric_parts = [
            "Persisted semantic conversion fields: " + _field_inventory(semantic) + "."
        ]
        if fixed_policy:
            metric_parts.append(f"{fixed_policy}.")
        fixed_metrics = _record_field_values(record, _FIXED_REFERENCE_METRIC_FIELDS)
        if fixed_metrics:
            metric_parts.append(
                "Fixed-reference derived metrics retained: "
                f"{_field_inventory(fixed_metrics, include_values=False)}."
            )
        if configured and analysis.issue == "#104":
            metric_parts.append(
                "The legacy analyzer's 1350 MHz selection is retained rather than "
                "rewritten."
            )
        if analysis.weighted_is_exact:
            selected = _exact_interval_selection_statement(analysis, record)
        else:
            selected = (
                "The persisted semantic selection is retained, but a corrected value is "
                "not verifiable from retained interval evidence; the trace weighted "
                f"value is {_trace_weighted_statement(analysis)}."
            )
        return ConversionImpact(
            "; ".join(category_parts),
            " ".join(metric_parts),
            selected,
            _conclusion_change_statement(analysis, record),
        )

    host_metrics = _record_field_values(record, _HOST_TIMING_FIELDS)
    if _uses_host_perf_counter(analysis, record):
        category = "no AICLK conversion (host perf_counter)"
        metric_text = (
            "Host perf_counter-derived timing/TFLOPS metrics retained: "
            f"{_field_inventory(host_metrics, include_values=False)}."
            if host_metrics
            else "Host perf_counter timing is the retained source; no AICLK conversion field is present."
        )
        if fixed_policy:
            metric_text += f" {fixed_policy}."
        selected = (
            "Not applicable: the retained host metrics do not use trace AICLK; "
            f"the trace weighted value {_trace_weighted_statement(analysis)} is context only."
        )
        return ConversionImpact(
            category, metric_text, selected, _conclusion_change_statement(analysis, record)
        )

    if analysis.issue == "#88":
        metric_text = (
            "Retained #88 quality/L1 comparison fields; no cycle-to-time or "
            "AICLK-derived metric is present."
        )
        if fixed_policy:
            metric_text += f" {fixed_policy}."
        return ConversionImpact(
            "no AICLK conversion (quality/L1 record)",
            metric_text,
            (
                "Not applicable: quality/L1 values do not use trace AICLK; the trace "
                f"weighted value {_trace_weighted_statement(analysis)} is context only."
            ),
            _conclusion_change_statement(analysis, record),
        )

    category_parts = ["legacy/unverifiable"]
    if _record_field_values(record, _FIXED_REFERENCE_FIELDS):
        category_parts.append("fixed 1350-MHz normalization")
    if configured:
        category_parts.append("configured-budget policy")
    if host_metrics:
        metric_text = (
            "Retained timing/throughput fields without a persisted AICLK source: "
            f"{_field_inventory(host_metrics, include_values=False)}."
        )
    elif semantic_derived:
        metric_text = (
            "Retained derived timing fields without a persisted AICLK source: "
            f"{_field_inventory(semantic_derived, include_values=False)}."
        )
    else:
        metric_text = "No retained derived elapsed/throughput metric is tied to trace AICLK."
    if fixed_policy:
        metric_text += f" {fixed_policy}."
    return ConversionImpact(
        "; ".join(category_parts),
        metric_text,
        (
            "No corrected conversion value is verifiable from retained interval evidence; "
            f"the trace weighted value is {_trace_weighted_statement(analysis)}."
        ),
        _conclusion_change_statement(analysis, record),
    )


def _conversion_impact_row(
    analysis: RecordAnalysis, record: Mapping[str, Any]
) -> str:
    trace = analysis.trace
    classification = analysis.classification
    if trace is None or classification is None:
        trace_text = "—"
    else:
        minmax = (
            f"{_format_number(analysis.minimum_mhz)} / "
            f"{_format_number(analysis.maximum_mhz)}"
        )
        trace_text = (
            f"{_format_values(classification.unique_mhz)}; min/max {minmax}; "
            f"weighted {_weighted_cell(analysis)}"
        )
    impact = _record_conversion_impact(analysis, record)
    return "| " + " | ".join(
        _cell(value)
        for value in (
            analysis.path.name,
            trace_text,
            impact.category,
            impact.metrics,
            impact.selected_value,
            impact.conclusion_change,
        )
    ) + " |"


def _append_conversion_impact_inventory(
    lines: list[str],
    analyses: Sequence[RecordAnalysis],
    records: Mapping[Path, Mapping[str, Any]],
) -> None:
    varying = [
        analysis
        for analysis in analyses
        if analysis.classification is not None
        and len(analysis.classification.unique_mhz) > 1
    ]
    lines.extend(
        [
            "",
            "## Per-record conversion impact for varying AICLK traces",
            "",
            (
                f"This section covers every one of the {len(varying)} committed JSON "
                "records whose resolved trace has more than one valid AICLK value. "
                "The trace min/max and weighted value are audit facts; a weighted value "
                "marked `capture-span diagnostic` is not a corrected conversion."
            ),
            "",
            "| record | trace AICLK (MHz; unique/min-max/weighted) | evidence category | retained derived values/metrics | corrected/selected value | published conclusion change |",
            "|---|---|---|---|---|---|",
        ]
    )
    for analysis in varying:
        record = records.get(analysis.path, {})
        lines.append(_conversion_impact_row(analysis, record))



def _record_trace_facts(
    trace: TraceData | None,
    interval: IntervalBounds | None,
) -> tuple[str, float | None, float | None, float | None, bool]:
    evidence = "no CSV trace"
    minimum = maximum = weighted = None
    weighted_is_exact = False
    if trace is None or not trace.samples:
        return evidence, minimum, maximum, weighted, weighted_is_exact
    if interval is not None:
        inside, complete = _coverage_rows(trace, interval)
        values = [sample.aiclk_mhz for sample in inside]
        if values:
            minimum, maximum = min(values), max(values)
        if complete:
            weighted = left_held_time_weighted_aiclk(
                trace.samples, interval.start, interval.end
            )
            weighted_is_exact = weighted is not None
        evidence = (
            f"exact {interval.kind} interval"
            if weighted_is_exact
            else f"bounds present; trace coverage incomplete ({interval.kind})"
        )
        return evidence, minimum, maximum, weighted, weighted_is_exact
    values = [sample.aiclk_mhz for sample in trace.samples]
    minimum, maximum = min(values), max(values)
    if trace.timestamps_ordered and len(trace.samples) > 1:
        weighted = left_held_time_weighted_aiclk(
            trace.samples,
            trace.samples[0].timestamp,
            trace.samples[-1].timestamp,
        )
    return "CSV capture span diagnostic", minimum, maximum, weighted, weighted_is_exact


def _record_field_presence(record: Mapping[str, Any]) -> tuple[bool, bool]:
    timing_fields = _recursive_has_key(
        record,
        {
            "seconds_per_iteration",
            "seconds_per_launch_p50",
            "elapsed_s",
            "achieved_tflops",
            "tflops_p50_derived",
        },
    )
    conversion_fields = _recursive_has_key(
        record,
        {
            "aiclk_mhz_for_elapsed_seconds",
            "aiclk_mhz_load_interval_weighted",
            "load_interval_aiclk_mhz",
            "ticks_to_seconds",
        },
    )
    return timing_fields, conversion_fields


def _record_analysis(
    root: Path,
    record_path: Path,
    record: Mapping[str, Any],
    csv_by_hash: Mapping[str, tuple[Path, ...]],
    trace_cache: dict[Path, TraceData],
) -> RecordAnalysis:
    resolution = resolve_trace(root, record_path, record, csv_by_hash)
    trace = (
        trace_cache.setdefault(resolution.path, read_trace(resolution.path))
        if resolution.path is not None
        else None
    )
    classification = (
        classify_aiclk_values([sample.aiclk_mhz for sample in trace.samples])
        if trace
        else None
    )
    interval = find_interval(record)
    evidence, minimum, maximum, weighted, weighted_is_exact = _record_trace_facts(
        trace, interval
    )
    timing_fields, conversion_fields = _record_field_presence(record)
    return RecordAnalysis(
        path=record_path,
        issue=_issue_label(record_path, record),
        status=_status(record),
        resolution=resolution,
        trace=trace,
        classification=classification,
        interval=interval,
        evidence=evidence,
        minimum_mhz=minimum,
        maximum_mhz=maximum,
        weighted_mhz=weighted,
        weighted_is_exact=weighted_is_exact,
        timing_fields=timing_fields,
        conversion_fields=conversion_fields,
        source_tag=_source_tag(record),
    )


def _interval_cell(analysis: RecordAnalysis) -> str:
    if analysis.interval is None:
        return analysis.evidence
    return analysis.evidence


def _weighted_cell(analysis: RecordAnalysis) -> str:
    if analysis.weighted_mhz is None:
        return "—"
    label = "exact" if analysis.weighted_is_exact else "diagnostic"
    return f"{_format_number(analysis.weighted_mhz)} ({label})"


def _analysis_row(analysis: RecordAnalysis) -> str:
    trace = analysis.trace
    classification = analysis.classification
    if trace is None:
        rows = unique = "—"
    else:
        rows = f"{len(trace.samples)}/{trace.csv_row_count}"
        unique = _format_values(classification.unique_mhz if classification else ())
    if classification is None:
        label = minimum_intermediate = position = "—"
    else:
        label = classification.label
        minimum_intermediate = _format_number(classification.minimum_intermediate_mhz)
        position = classification.intermediate_position or "—"
    minmax = (
        f"{_format_number(analysis.minimum_mhz)} / {_format_number(analysis.maximum_mhz)}"
        if analysis.minimum_mhz is not None
        else "—"
    )
    reference = analysis.resolution.reference or "—"
    resolved = analysis.resolution.path.name if analysis.resolution.path else "—"
    return "| " + " | ".join(
        _cell(value)
        for value in (
            analysis.path.name,
            analysis.issue,
            analysis.status,
            reference,
            resolved,
            analysis.resolution.method,
            rows,
            unique,
            label,
            minmax,
            minimum_intermediate,
            position,
            _interval_cell(analysis),
            _weighted_cell(analysis),
        )
    ) + " |"


def _csv_row(path: Path, trace: TraceData, users: Sequence[RecordAnalysis]) -> str:
    classification = classify_aiclk_values([sample.aiclk_mhz for sample in trace.samples])
    references = "; ".join(analysis.path.name for analysis in users) or "—"
    return "| " + " | ".join(
        _cell(value)
        for value in (
            path.name,
            references,
            f"{len(trace.samples)}/{trace.csv_row_count}",
            _format_values(classification.unique_mhz),
            classification.label,
            _format_number(classification.minimum_intermediate_mhz),
            classification.intermediate_position or "—",
            trace.invalid_row_count,
        )
    ) + " |"


def _focus_row(analysis: RecordAnalysis) -> str:
    trace = analysis.trace
    classification = analysis.classification
    minmax = (
        f"{_format_number(analysis.minimum_mhz)} / {_format_number(analysis.maximum_mhz)}"
        if analysis.minimum_mhz is not None
        else "—"
    )
    return "| " + " | ".join(
        _cell(value)
        for value in (
            analysis.path.name,
            analysis.issue,
            analysis.interval.kind if analysis.interval else "—",
            analysis.evidence,
            minmax,
            classification.label if classification else "—",
            _weighted_cell(analysis),
            trace.path.name if trace else "—",
        )
    ) + " |"


def _find_record(analyses: Sequence[RecordAnalysis], needle: str) -> RecordAnalysis | None:
    for analysis in analyses:
        if needle in analysis.path.name:
            return analysis
    return None


def _format_percentage(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100.0:.{digits}f}%"


class _ConversionCallVisitor(ast.NodeVisitor):
    """Collect production conversion calls with their enclosing function."""

    def __init__(self) -> None:
        self.function_stack: list[str] = []
        self.calls: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Name) and function.id == "ticks_to_seconds":
            self.calls.append((node.lineno, self.function_stack[-1] if self.function_stack else "module"))
        self.generic_visit(node)


def _conversion_calls(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _ConversionCallVisitor()
    visitor.visit(tree)
    return tuple(sorted(visitor.calls))


def _line_containing(path: Path, needle: str) -> int:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return line_number
    raise ValueError(f"conversion inventory marker is missing: {path}:{needle}")


_InventoryEntry = tuple[str, int, str, str, str, str]


def _add_inventory_entry(
    entries: list[_InventoryEntry],
    repository_root: Path,
    path: Path,
    line: int,
    conversion: str,
    value: str,
    source: str,
    contract: str,
) -> None:
    entries.append(
        (path.relative_to(repository_root).as_posix(), line, conversion, value, source, contract)
    )


def _append_resident_record_inventory(
    entries: list[_InventoryEntry], repository_root: Path, path: Path
) -> None:
    for line, function in _conversion_calls(path):
        if function == "_sampler_comparison":
            value = "elapsed_aiclk_selection"
            source = (
                "resident elapsed-time source (loaded DRAM/Newton-Schulz: weighted; "
                "sampled/unloaded: in-run maximum or configured)"
            )
            contract = "sampler gap seconds"
        elif function == "build_outlier_analysis":
            value = "elapsed_selection['aiclk_mhz']"
            source = "aiclk_source_for_elapsed_seconds"
            contract = "resident event/pair/run elapsed seconds"
        else:
            value = "record elapsed selection"
            source = "persisted aiclk_source_for_elapsed_seconds"
            contract = "validator recomputation of resident elapsed seconds"
        _add_inventory_entry(
            entries,
            repository_root,
            path,
            line,
            f"{function} ticks → seconds",
            value,
            source,
            contract,
        )


def _append_issue104_inventory(
    entries: list[_InventoryEntry], repository_root: Path, path: Path
) -> None:
    for line, function in _conversion_calls(path):
        if function == "analyze_issue104":
            value = "configured 1350 MHz default selection"
            source = "configured budget"
        else:
            value = "aiclk_selection"
            source = "legacy_unverified for historical sampled records; trace selection otherwise"
        _add_inventory_entry(
            entries,
            repository_root,
            path,
            line,
            f"{function} ticks → seconds",
            value,
            source,
            "historical wrap analysis preserves numeric results",
        )


def _append_run_resident_inventory(
    entries: list[_InventoryEntry], repository_root: Path, path: Path
) -> None:
    for line, function in _conversion_calls(path):
        if function == "_load_achieved_throughput":
            conversion = "throughput reference ticks → seconds"
            value = "1350 MHz fixed reference"
            source = "fixed_reference_1350_mhz"
            contract = "normalization denominator, not observed AICLK"
        else:
            conversion = "calibration actual duration ticks → seconds"
            value = "throughput_aiclk_mhz"
            source = "actual_duration_aiclk_source"
            contract = "weighted for loaded records; selected run clock otherwise"
        _add_inventory_entry(
            entries, repository_root, path, line, conversion, value, source, contract
        )


def _append_load_harness_inventory(
    entries: list[_InventoryEntry], repository_root: Path, path: Path
) -> None:
    for line, function in _conversion_calls(path):
        if function == "predicted_load_time_seconds":
            conversion = "predicted load card ticks → seconds"
            value = "min(weighted load interval, in-run minimum) when minimum is present"
            source = "predicted_load_time_aiclk_source"
            contract = "time_weighted_load_interval or in_run_minimum"
        else:
            conversion = "persisted throughput reference ticks → seconds"
            value = "1350 MHz fixed reference"
            source = "fixed_reference_1350_mhz"
            contract = "historical repeats_per_card_second normalization"
        _add_inventory_entry(
            entries, repository_root, path, line, conversion, value, source, contract
        )


def _append_inverse_conversion_inventory(
    entries: list[_InventoryEntry], repository_root: Path, source_root: Path, tools_root: Path
) -> None:
    # Inverse seconds-to-ticks calculations are listed separately so a budget
    # clock cannot be mistaken for measured elapsed-time evidence.
    resident_harness = source_root / "resident_harness.py"
    _add_inventory_entry(
        entries,
        repository_root,
        resident_harness,
        _line_containing(resident_harness, "def interval_ticks_for_microseconds("),
        "inverse wall time → ticks",
        _CONFIGURED_BUDGET_AICLK_LABEL,
        "configured_budget",
        "startup/budget policy, not measured elapsed time",
    )
    _add_inventory_entry(
        entries,
        repository_root,
        resident_harness,
        _line_containing(resident_harness, "cap_ticks = config.outer_timeout_seconds"),
        "inverse outer timeout seconds → ticks",
        _CONFIGURED_BUDGET_AICLK_LABEL,
        "configured_budget",
        "outer-cap policy, not measured elapsed time",
    )
    run_resident = source_root / "run_resident.py"
    _add_inventory_entry(
        entries,
        repository_root,
        run_resident,
        _line_containing(run_resident, "target_ticks = int(target_seconds"),
        "inverse calibration target seconds → ticks",
        _CONFIGURED_BUDGET_AICLK_LABEL,
        "configured_budget",
        "calibration budget policy",
    )
    _add_inventory_entry(
        entries,
        repository_root,
        run_resident,
        _line_containing(run_resident, "startup_ticks = 100_000"),
        "inverse startup allowance → ticks",
        _CONFIGURED_BUDGET_AICLK_LABEL,
        "configured_budget",
        "calibration budget policy",
    )
    issue104 = tools_root / "analyze_issue104_wrap.py"
    _add_inventory_entry(
        entries,
        repository_root,
        issue104,
        _line_containing(issue104, "def _phase_window_ticks("),
        "inverse phase-window milliseconds → ticks",
        "selected analyzer AICLK",
        "legacy_unverified for historical sampled records; configured off-mode default otherwise",
        "analysis window policy, not elapsed-time evidence",
    )


def _conversion_inventory() -> list[_InventoryEntry]:
    """Return the post-#124 production tick/time conversion inventory.

    The line numbers come from the current AST/source text on every report
    generation, so the committed Markdown cannot silently retain pre-#124
    locations after a source edit.
    """
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "enodia" / "tt" / "bench"
    tools_root = repository_root / "tools"
    entries: list[_InventoryEntry] = []

    resident_clock = source_root / "resident_clock.py"
    _add_inventory_entry(
        entries,
        repository_root,
        resident_clock,
        _line_containing(resident_clock, "def ticks_to_seconds("),
        "shared ticks → seconds helper",
        "selected AICLK (exact when weighted)",
        "selection['aiclk_source'] / semantic selection",
        "all cycle→time callers must pass a selected value",
    )
    _append_resident_record_inventory(
        entries, repository_root, source_root / "resident_record.py"
    )
    _append_issue104_inventory(entries, repository_root, tools_root / "analyze_issue104_wrap.py")
    _append_run_resident_inventory(
        entries, repository_root, source_root / "run_resident.py"
    )
    _append_load_harness_inventory(
        entries, repository_root, source_root / "load_harness.py"
    )
    _append_inverse_conversion_inventory(entries, repository_root, source_root, tools_root)
    return entries


def _conversion_inventory_section() -> list[str]:
    lines = [
        "## Post-#124 cycle-ticks-to-seconds conversion inventory",
        "",
        "Line numbers are generated from the current production source. The source column retains the machine-readable record tag where one exists and the semantic selection beside it.",
        "",
        "| path | line | conversion path | selected value | chosen source | interpretation |",
        "|---|---:|---|---|---|---|",
    ]
    lines.extend(
        "| " + " | ".join(_cell(value) for value in entry) + " |"
        for entry in _conversion_inventory()
    )
    lines.extend(
        [
            "",
            (
                "`clock.aiclk_mhz` remains the in-run maximum public observation. "
                "For loaded DRAM/Newton-Schulz records, elapsed conversion uses the "
                "persisted time-weighted load-interval AICLK, while conservative "
                "prediction may select the in-run minimum. Sampled or unloaded "
                "resident records may retain `in_run_maximum` or `configured_budget` "
                "as their elapsed-time selection; the persisted source/selection "
                "pair identifies that choice. The 1,350 MHz throughput reference is "
                "fixed normalization. Configured budget and outer-cap conversions "
                "are inverse policy calculations."
            ),
        ]
    )
    return lines


def _final_p50_values(record: Mapping[str, Any]) -> dict[str, float]:
    rows = record.get("rows")
    if not isinstance(rows, list):
        return {}
    p50_by_row: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = row.get("row")
        timing = row.get("timing")
        if not isinstance(name, str) or not isinstance(timing, Mapping):
            continue
        value = timing.get("tflops_p50_derived")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            p50_by_row[name] = parsed
    return p50_by_row


def _final_p50_change(
    comparison: Any, p50_by_row: Mapping[str, float]
) -> tuple[str, float] | None:
    if not isinstance(comparison, Mapping):
        return None
    row_names = comparison.get("rows")
    if not isinstance(row_names, list) or len(row_names) != 2:
        return None
    left, right = row_names
    if not isinstance(left, str) or not isinstance(right, str):
        return None
    if left not in p50_by_row or right not in p50_by_row:
        return None
    change = p50_by_row[right] / p50_by_row[left] - 1.0
    if not math.isfinite(change):
        return None
    placement = str(comparison.get("placement") or f"{left} vs {right}")
    return placement, change


def _final_p50_changes(record: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    p50_by_row = _final_p50_values(record)
    comparisons = record.get("matched_placement_differences")
    if not isinstance(comparisons, list):
        return ()
    changes = []
    for comparison in comparisons:
        change = _final_p50_change(comparison, p50_by_row)
        if change is not None:
            changes.append(change)
    return tuple(changes)


def _source_audit_section(analyses: Sequence[RecordAnalysis]) -> list[str]:
    family = [
        analysis
        for analysis in analyses
        if "newton-schulz" in analysis.path.name
        and "2026-09-27" <= analysis.path.name[:10] <= "2026-10-06"
        and "issue88" not in analysis.path.name
    ]
    dates = sorted({analysis.path.name[:10] for analysis in family})
    timing_count = sum(analysis.timing_fields for analysis in family)
    conversion_count = sum(analysis.conversion_fields for analysis in family)
    rows = [
        "## #63-family Newton-Schulz timing/source audit",
        "",
        (
            f"The committed #63-family slice contains {len(family)} JSON records dated "
            f"{dates[0] if dates else '—'} through {dates[-1] if dates else '—'}. "
            f"{timing_count} records contain published timing/TFLOPS fields; "
            f"{conversion_count} contain an explicit AICLK-to-elapsed conversion field."
        ),
        "",
        "| record | source/runner named by record | published timing fields | AICLK conversion fields |",
        "|---|---|---|---|",
    ]
    for analysis in family:
        rows.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    analysis.path.name,
                    analysis.source_tag,
                    "yes" if analysis.timing_fields else "no",
                    "yes" if analysis.conversion_fields else "no",
                )
            )
            + " |"
        )
    rows.extend(
        [
            "",
            (
                "Source inspection result: `enodia/tt/bench/run_matmul.py` functions "
                "`run_stock` and `run_custom_newton_schulz` bracket device launches plus "
                "`ttnn.synchronize_device` with host `time.perf_counter()`. They publish "
                "`seconds_per_iteration`, `seconds_per_launch_*`, `achieved_tflops`, and "
                "`tflops_*_derived`; the latter are FLOPs divided by those host elapsed "
                "seconds. `tools/newton_schulz_issue100_same_run.py:run_issue100_comparison` "
                "delegates to `run_custom_newton_schulz`, and "
                "`tools/newton_schulz_issue101_combined.py:_run_performance_with_capture` "
                "uses the same path."
            ),
            "",
            (
                "The records dated 2026-09-27 through 2026-10-06 contain no persisted "
                "`aiclk_mhz_load_interval_weighted`, `aiclk_mhz_for_elapsed_seconds`, "
                "or equivalent cycle-to-time conversion for their published TFLOPS/seconds. "
                "Their AICLK columns are telemetry context, not the denominator used by "
                "the published host timings. The 2026-10-03 correctness records reference "
                "`tools/newton_schulz_runner.py`, which is not present in this checkout and "
                "does not publish timing fields in the committed records; that gap does not "
                "support an AICLK conversion claim. The later "
                "`enodia/tt/bench/power_trace_snapshot.py:_time_weighted_aiclk` helper is "
                "the explicit-interval path used by #118, not by these older #63 records."
            ),
        ]
    )
    rows.append("")
    return rows


def _row_field(row: Any, field: str) -> Any:
    return row.get(field) if isinstance(row, Mapping) else None


def _row_name(row: Any, index: int) -> str:
    value = _row_field(row, "row")
    return value if isinstance(value, str) else f"row-{index}"


def _row_status(row: Any) -> str:
    value = _row_field(row, "status")
    return value if isinstance(value, str) else "—"


def _row_samples(row: Any) -> list[Any] | None:
    value = _row_field(row, "seconds_per_launch_samples")
    return value if isinstance(value, list) else None


def _row_sample_count(raw_samples: list[Any] | None) -> int | None:
    return len(raw_samples) if raw_samples is not None else None


def _parse_sample_seconds(raw_samples: list[Any] | None) -> float | None:
    if not raw_samples:
        return None
    parsed: list[float] = []
    for value in raw_samples:
        try:
            parsed_value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed_value) or parsed_value < 0.0:
            return None
        parsed.append(parsed_value)
    return sum(parsed)


def _row_start_bounds(
    prefix_seconds: float,
    prefix_known: bool,
    sample_seconds: float | None,
    run_start: dt.datetime | None,
) -> tuple[float | None, dt.datetime | None]:
    if not prefix_known or sample_seconds is None:
        return None, None
    start_timestamp = (
        run_start + dt.timedelta(seconds=prefix_seconds)
        if run_start is not None
        else None
    )
    return prefix_seconds, start_timestamp


def _advance_timing_prefix(
    prefix_seconds: float, prefix_known: bool, sample_seconds: float | None
) -> tuple[float, bool]:
    if sample_seconds is None:
        return prefix_seconds, False
    if prefix_known:
        return prefix_seconds + sample_seconds, True
    return prefix_seconds, False


def _build_row_timing_bound(
    row: Any,
    index: int,
    run_start: dt.datetime | None,
    prefix_seconds: float,
    prefix_known: bool,
) -> tuple[RowTimingBound, float, bool]:
    raw_samples = _row_samples(row)
    sample_seconds = _parse_sample_seconds(raw_samples)
    start_offset, start_timestamp = _row_start_bounds(
        prefix_seconds, prefix_known, sample_seconds, run_start
    )
    bound = RowTimingBound(
        _row_name(row, index),
        _row_status(row),
        _row_sample_count(raw_samples),
        sample_seconds,
        start_offset,
        start_timestamp,
    )
    next_prefix_seconds, next_prefix_known = _advance_timing_prefix(
        prefix_seconds, prefix_known, sample_seconds
    )
    return bound, next_prefix_seconds, next_prefix_known


def _row_timing_bounds(
    record: Mapping[str, Any], run_start: dt.datetime | None
) -> tuple[RowTimingBound, ...]:
    """Sum measured per-launch samples in persisted block order."""

    rows = record.get("rows")
    if not isinstance(rows, list):
        return ()
    bounds: list[RowTimingBound] = []
    prefix_seconds = 0.0
    prefix_known = True
    for index, row in enumerate(rows):
        bound, prefix_seconds, prefix_known = _build_row_timing_bound(
            row, index, run_start, prefix_seconds, prefix_known
        )
        bounds.append(bound)
    return tuple(bounds)


def _hold_segments(
    analysis: RecordAnalysis, threshold_mhz: float = 1343.0
) -> tuple[HoldSegment, ...]:
    """Return merged positive-duration segments below the audit threshold."""

    if analysis.trace is None or analysis.interval is None:
        return ()
    interval = analysis.interval
    segments: list[HoldSegment] = []
    for segment in _left_held_positive_segments(
        analysis.trace.samples, interval.start, interval.end
    ):
        if segment.aiclk_mhz >= threshold_mhz:
            continue
        start_offset = (segment.start - interval.start).total_seconds()
        end_offset = (segment.end - interval.start).total_seconds()
        current = HoldSegment(
            segment.aiclk_mhz,
            start_offset,
            end_offset,
            segment.start,
            segment.end,
        )
        if (
            segments
            and segments[-1].aiclk_mhz == current.aiclk_mhz
            and segments[-1].end == current.start
        ):
            previous = segments[-1]
            segments[-1] = HoldSegment(
                previous.aiclk_mhz,
                previous.start_offset_seconds,
                current.end_offset_seconds,
                previous.start,
                current.end,
            )
        else:
            segments.append(current)
    return tuple(segments)


def _format_timestamp(value: dt.datetime | None) -> str:
    return value.isoformat(timespec="microseconds") if value is not None else "—"


def _format_seconds(value: float | None) -> str:
    return f"{value:.9f} s" if value is not None else "—"


def _append_issue88_record_timing(
    lines: list[str], analysis: RecordAnalysis, record: Mapping[str, Any]
) -> None:
    run_start = analysis.interval.start if analysis.interval is not None else None
    run_end = analysis.interval.end if analysis.interval is not None else None
    bounds = _row_timing_bounds(record, run_start)
    lines.extend(
        [
            f"#### `{analysis.path.name}`",
            "",
            (
                f"`run_start` = `{_format_timestamp(run_start)}`, "
                f"`run_end` = `{_format_timestamp(run_end)}`. "
                "Rows are consumed in persisted block order. For row *i*, the start "
                "lower bound is `run_start` plus the sum of measured "
                "`seconds_per_launch_samples` from earlier rows; no p50 or synthetic "
                "duration is used."
            ),
            "",
            "| row order/name | status | sample count | sample length sum (s) | cumulative start lower bound (s) | lower-bound timestamp |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for index, bound in enumerate(bounds):
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    f"{index}: {bound.name}",
                    bound.status,
                    bound.sample_count if bound.sample_count is not None else "—",
                    _format_seconds(bound.sample_seconds),
                    _format_seconds(bound.start_offset_seconds),
                    _format_timestamp(bound.start_timestamp),
                )
            )
            + " |"
        )
    if not bounds:
        lines.append("| — | — | — | — | — | — |")

    holds = _hold_segments(analysis)
    lines.extend(
        [
            "",
            (
                "Positive-duration trace holds below 1343 MHz are listed below; this "
                "includes the 800 MHz hold and excludes zero-duration rows at the interval "
                "end. The end offset is measured from `run_start`."
            ),
            "",
            "| AICLK (MHz) | hold start offset | hold end offset | hold end timestamp |",
            "|---:|---:|---:|---|",
        ]
    )
    if holds:
        for hold in holds:
            lines.append(
                f"| {_format_number(hold.aiclk_mhz)} | "
                f"{_format_seconds(hold.start_offset_seconds)} | "
                f"{_format_seconds(hold.end_offset_seconds)} | "
                f"{_format_timestamp(hold.end)} |"
            )
    else:
        lines.append("| — | — | — | — |")


def _find_row_bound(
    bounds: Sequence[RowTimingBound], name: str
) -> RowTimingBound | None:
    return next((bound for bound in bounds if bound.name == name), None)


def _issue88_bound_overlaps_hold(
    bound: RowTimingBound | None, hold_end: float
) -> bool:
    return (
        bound is not None
        and bound.start_offset_seconds is not None
        and bound.start_offset_seconds <= hold_end
    )


def _append_issue88_l16_conclusion(
    lines: list[str],
    bf16_l16: RowTimingBound | None,
    fp32_l16: RowTimingBound | None,
    hold_end: float,
) -> None:
    bf16_overlaps = _issue88_bound_overlaps_hold(bf16_l16, hold_end)
    fp32_overlaps = _issue88_bound_overlaps_hold(fp32_l16, hold_end)
    if bf16_overlaps and not fp32_overlaps:
        lines.append(
            "For L16, only BF16-R can overlap the initial low-clock hold under these "
            "lower bounds; that would make BF16-R look slower and make the FP32-R "
            "difference look smaller."
        )
    elif bf16_overlaps and fp32_overlaps:
        lines.append(
            "For L16, both BF16-R and FP32-R can overlap the initial low-clock hold "
            "under these lower bounds, so the direction of the clock confounding cannot "
            "be determined."
        )
    else:
        lines.append(
            "For L16, neither row's lower bound overlaps the initial low-clock hold; "
            "this audit makes no additional directional claim."
        )


def _issue88_l32_rows_after_hold(
    bounds: tuple[RowTimingBound | None, RowTimingBound | None], hold_end: float
) -> bool:
    return all(
        bound is not None
        and bound.start_offset_seconds is not None
        and bound.start_offset_seconds > hold_end
        for bound in bounds
    )


def _append_issue88_l32_conclusion(
    lines: list[str],
    bf16_l32_dram: RowTimingBound | None,
    fp32_l32_dram: RowTimingBound | None,
    hold_end: float,
) -> None:
    if _issue88_l32_rows_after_hold((bf16_l32_dram, fp32_l32_dram), hold_end):
        lines.append(
            "Both L32/DRAM rows have start lower bounds after the low-clock hold; a "
            "condition-specific AICLK upper-bound claim is therefore supported by this "
            "ordering evidence."
        )
    else:
        lines.append(
            "The BF16-R and FP32-R L32/DRAM rows are not both after the low-clock hold "
            "under their start lower bounds; this audit cannot claim a condition-specific "
            "AICLK upper bound."
        )


def _append_issue88_order_conclusions(
    lines: list[str], analysis: RecordAnalysis, record: Mapping[str, Any]
) -> None:
    bounds = _row_timing_bounds(
        record, analysis.interval.start if analysis.interval is not None else None
    )
    holds = _hold_segments(analysis)
    if not holds:
        lines.append("No condition-specific clock-bound conclusion is available because no low-clock hold segment was measured.")
        return
    first_hold_end = holds[0].end_offset_seconds
    _append_issue88_l16_conclusion(
        lines,
        _find_row_bound(bounds, "bf16-r-L16"),
        _find_row_bound(bounds, "fp32-r-L16"),
        first_hold_end,
    )
    _append_issue88_l32_conclusion(
        lines,
        _find_row_bound(bounds, "bf16-r-L32-r-dram"),
        _find_row_bound(bounds, "fp32-r-L32-r-dram"),
        first_hold_end,
    )


def _follow_up_section(
    analyses: Sequence[RecordAnalysis],
    records: Mapping[Path, Mapping[str, Any]],
) -> list[str]:
    lines = ["## Follow-up findings", ""]
    convergence = _find_record(analyses, "2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json")
    device1 = _find_record(analyses, "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json")
    final = _find_record(analyses, "2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json")
    final_summary = _find_record(
        analyses,
        "2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json",
    )
    targets = tuple(analysis for analysis in (convergence, device1, final) if analysis is not None)
    lines.extend(
        [
            "### #88 R-format ordering and confounding",
            "",
            (
                "The old `0.521%` complete-run-width claim is withdrawn. It treated the "
                "whole run as available to every condition and did not use the host timing "
                "row order. The corrected lower-bound audit below uses each target record's "
                "persisted `rows` in block order and sums the measured "
                "`seconds_per_launch_samples` for each row. It is an ordering lower bound, "
                "not a reconstructed row-to-power timestamp alignment."
            ),
            "",
            (
                "The explicit-interval inventory above reports extrema and weighted values "
                "from the same positive-duration left-held trace segments. The three #88 "
                "records below additionally report every measured row's sample-length sum, "
                "cumulative start lower bound, and the 800 MHz / below-1343 MHz hold ends."
            ),
        ]
    )
    for analysis in targets:
        record = records.get(analysis.path)
        if record is None:
            continue
        lines.append("")
        _append_issue88_record_timing(lines, analysis, record)
        lines.append("")
        _append_issue88_order_conclusions(lines, analysis, record)

    final_changes = (
        _final_p50_changes(records[final_summary.path])
        if final_summary is not None and final_summary.path in records
        else ()
    )
    lines.extend(
        [
            "",
            "### Corrected conclusion boundary",
            "",
            (
                "The three records retain corrected quality metrics and measured "
                "throughput/L1 cost context. Their host timings remain valid observations, "
                "but the prior conclusion that a complete-run AICLK width could not explain "
                "the throughput differences is withdrawn: the row-order lower bounds above "
                "do not establish synchronized condition-specific clock values."
            ),
            "",
            (
                "The final summary's retained P50 TFLOPS comparisons are descriptive only "
                "and were not used to construct the timing bounds. "
                + (
                    "Recorded comparisons: "
                    + "; ".join(
                        f"{placement}: {_format_percentage(change)}"
                        for placement, change in final_changes
                    )
                    + "."
                    if final_changes
                    else "No usable matched P50 comparison is available."
                )
            ),
            "",
            (
                "Accordingly, this audit supports reporting the corrected quality metrics "
                "and the observed throughput/L1 trade-off only within the evidence retained "
                "by the records. It does not make a stronger FP32-R/BF16-R causal or "
                "clock-independent conclusion."
            ),
            "",
            "### No-bound old records",
            "",
            (
                "A trace classified as `two-valued {800,1350}` contains only the two endpoint "
                "values. A trace classified as `contains intermediate values` reports the "
                "minimum value outside that set and the thirds-based CSV position of its "
                "occurrence. Without explicit covered bounds, all such min/max/weighted "
                "values are capture-span diagnostics, never in-run timing evidence."
            ),
            "",
            (
                "Operator-requested assumption (not proof): for a trace whose only values "
                "are 800 and 1350, treat 800 as idle outside the measured interval and 1350 "
                "as the measured clock. Under that assumption, using maximum 1350 then has "
                "zero AICLK conversion error. The trace alone does not prove that assumption "
                "or the interval coverage."
            ),
            "",
            (
                "Issue #12 resident records include both endpoint-only traces and historical "
                "intermediate-value traces: the current-wrap capture has minimum intermediate "
                "1343 MHz at the beginning of its CSV, while the final corrected-drain and "
                "board-id-alias timing captures are endpoint-only; watcher validations and the "
                "preflight rejection have no intermediate evidence or no trace. Issue #104's "
                "sampler-default and sampler-5s resident traces are endpoint-only; its "
                "sampler-off record has no CSV by design. See the focused tables above for "
                "every committed #12 and #104 JSON record."
            ),
            "",
            "### #118 weighted comparisons",
            "",
            "| condition | prior published weighted AICLK (MHz) | audit result | evidence |",
            "|---|---:|---:|---|",
            "| unloaded control | 1,350 reference | unavailable | bounds are not covered through the final CSV sample; no weighted evidence |",
            "| DRAM | 1,349.842 | 1,349.842 | exact left-held load interval |",
            "| Newton-Schulz | 1,050.914 | 1,050.914 | exact left-held load interval |",
            "",
            (
                "These values reproduce the prior audit's exact #118 comparisons to three "
                "decimal MHz without rewriting any measurement leaf. The control remains a "
                "1,350 MHz reference rather than a computed weighted load interval. The "
                "corrected extrema are not weighted averages: for example, the DRAM interval "
                "now reports the positive-duration 800 MHz pre-start hold in its min/max while "
                "its weighted value remains 1,349.842 MHz."
            ),
        ]
    )
    return lines


def _report_input_data(
    root: Path,
) -> tuple[
    list[Path],
    list[Path],
    dict[Path, TraceData],
    list[RecordAnalysis],
    dict[Path, Mapping[str, Any]],
]:
    json_paths = sorted(root.glob("*.json"))
    csv_paths = sorted(root.glob("*.csv"))
    trace_cache: dict[Path, TraceData] = {}
    csv_by_hash: dict[str, list[Path]] = {}
    for path in csv_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        csv_by_hash.setdefault(digest, []).append(path)
    hash_index = {digest: tuple(sorted(paths)) for digest, paths in csv_by_hash.items()}
    analyses: list[RecordAnalysis] = []
    records: dict[Path, Mapping[str, Any]] = {}
    for path in json_paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record = {"status": f"unreadable: {type(exc).__name__}"}
        records[path] = record
        analyses.append(_record_analysis(root, path, record, hash_index, trace_cache))
    return json_paths, csv_paths, trace_cache, analyses, records


def _users_by_trace(
    csv_paths: Sequence[Path], analyses: Sequence[RecordAnalysis]
) -> dict[Path, list[RecordAnalysis]]:
    users_by_trace: dict[Path, list[RecordAnalysis]] = {path: [] for path in csv_paths}
    for analysis in analyses:
        if analysis.resolution.path in users_by_trace:
            users_by_trace[analysis.resolution.path].append(analysis)
    return users_by_trace


def _report_header(root: Path, json_paths: Sequence[Path], csv_paths: Sequence[Path]) -> list[str]:
    return [
        "# Issue #122 AICLK audit (generated)",
        "",
        f"- Input root: `{root.as_posix()}`",
        "- Generation command: `python3 tools/audit_issue122_aiclk.py`",
        "- JSON records: " + str(len(json_paths)),
        "- CSV files: " + str(len(csv_paths)),
        (
            "- Interval policy: valid timestamped rows are left-held until the next row; "
            "an explicit interval is exact only when ordered trace rows cover both bounds. "
            "Missing bounds produce a `CSV capture span diagnostic`; they do not become "
            "in-run evidence. Intermediate means any valid AICLK outside `{800,1350}`; "
            "its position uses the beginning/middle/end thirds of the CSV."
        ),
        (
            "- Exact semantic AICLK checks compare persisted weighted/elapsed values "
            "with the recomputed interval value using an absolute 1e-9 MHz tolerance "
            "for representation-level differences only; persisted source/selection "
            "tags are compared when present."
        ),
        "",
    ]


def _append_json_inventory(lines: list[str], analyses: Sequence[RecordAnalysis]) -> None:
    lines.extend(
        [
            "",
            "## JSON record inventory",
            "",
            "| record | issue | status | trace reference/provenance | resolved CSV | pairing | rows valid/CSV | unique AICLK (MHz) | classification | min/max (MHz) | min intermediate (MHz) | position | interval evidence | weighted AICLK (MHz) |",
            "|---|---|---|---|---|---|---:|---|---|---:|---:|---|---|---:|",
        ]
    )
    lines.extend(_analysis_row(analysis) for analysis in analyses)


def _append_csv_inventory(
    lines: list[str],
    csv_paths: Sequence[Path],
    trace_cache: dict[Path, TraceData],
    users_by_trace: Mapping[Path, Sequence[RecordAnalysis]],
) -> None:
    lines.extend(
        [
            "",
            "## Committed CSV inventory",
            "",
            "| CSV | JSON records using it | rows valid/CSV | unique AICLK (MHz) | classification | min intermediate (MHz) | position | invalid rows |",
            "|---|---|---:|---|---|---:|---|---:|",
        ]
    )
    for path in csv_paths:
        trace = trace_cache.setdefault(path, read_trace(path))
        lines.append(_csv_row(path, trace, users_by_trace[path]))


def _append_explicit_interval_inventory(
    lines: list[str], analyses: Sequence[RecordAnalysis]
) -> None:
    exact_or_bounded = [analysis for analysis in analyses if analysis.interval is not None]
    lines.extend(
        [
            "",
            "## Explicit interval records",
            "",
            (
                f"The inventory contains {len(exact_or_bounded)} explicit-interval JSON rows. "
                "The six logical target runs are the three #88 retakes and the three #118 "
                "conditions; the count is eight because the device-1 and final-runner "
                "retakes each have both a source and summary JSON row using the same trace. "
                "All eight rows are retained below."
            ),
            "",
            "| record | issue | interval kind | evidence | min/max (MHz) | classification | weighted AICLK (MHz) | resolved CSV |",
            "|---|---|---|---|---:|---|---:|---|",
        ]
    )
    lines.extend(_focus_row(analysis) for analysis in exact_or_bounded)


def _append_focused_inventory(lines: list[str], analyses: Sequence[RecordAnalysis]) -> None:
    focused = [
        analysis
        for analysis in analyses
        if analysis.issue in {"#12", "#104"}
    ]
    lines.extend(
        [
            "",
            "## Issue #12 and #104 resident records",
            "",
            "| record | issue | classification | min/max (MHz) | interval evidence | weighted AICLK (MHz) | resolved CSV |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )
    for analysis in focused:
        classification = analysis.classification.label if analysis.classification else "no trace"
        minmax = (
            f"{_format_number(analysis.minimum_mhz)} / {_format_number(analysis.maximum_mhz)}"
            if analysis.minimum_mhz is not None
            else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    analysis.path.name,
                    analysis.issue,
                    classification,
                    minmax,
                    analysis.evidence,
                    _weighted_cell(analysis),
                    analysis.trace.path.name if analysis.trace else "—",
                )
            )
            + " |"
        )


def _append_intermediate_inventory(
    lines: list[str], csv_paths: Sequence[Path], trace_cache: dict[Path, TraceData]
) -> None:
    intermediate_csvs: list[tuple[Path, AiclkClassification]] = []
    for path in csv_paths:
        trace = trace_cache.setdefault(path, read_trace(path))
        classification = classify_aiclk_values([sample.aiclk_mhz for sample in trace.samples])
        if classification.minimum_intermediate_mhz is not None:
            intermediate_csvs.append((path, classification))
    lines.extend(
        [
            "",
            "## Intermediate-value CSV diagnostics",
            "",
            "| CSV | minimum intermediate (MHz) | position | unique AICLK (MHz) |",
            "|---|---:|---|---|",
        ]
    )
    for path, classification in intermediate_csvs:
        lines.append(
            f"| {path.name} | {_format_number(classification.minimum_intermediate_mhz)} | "
            f"{classification.intermediate_position} | {_format_values(classification.unique_mhz)} |"
        )


def render_report(root: Path) -> str:
    """Render the deterministic Markdown report used by the committed audit."""

    json_paths, csv_paths, trace_cache, analyses, records = _report_input_data(root)
    users_by_trace = _users_by_trace(csv_paths, analyses)
    lines = _report_header(root, json_paths, csv_paths)
    lines.extend(_conversion_inventory_section())
    _append_conversion_impact_inventory(lines, analyses, records)
    _append_json_inventory(lines, analyses)
    _append_csv_inventory(lines, csv_paths, trace_cache, users_by_trace)
    _append_explicit_interval_inventory(lines, analyses)

    _append_focused_inventory(lines, analyses)
    _append_intermediate_inventory(lines, csv_paths, trace_cache)
    lines.extend(_source_audit_section(analyses))
    lines.extend(_follow_up_section(analyses, records))
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements-root",
        type=Path,
        default=Path("docs/measurements"),
        help="committed measurement corpus root (default: docs/measurements)",
    )
    args = parser.parse_args(argv)
    if not args.measurements_root.is_dir():
        parser.error(f"measurements root is not a directory: {args.measurements_root}")
    print(render_report(args.measurements_root), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
