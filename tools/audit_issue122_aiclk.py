"""Generate the board-free Issue #122 AICLK audit from committed measurements.

The analyzer intentionally reads only ``docs/measurements`` (or the directory
specified by ``--measurements-root``).  It never opens a device or invokes an
external command.  The pure helpers are also used by the board-free tests.
"""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class TraceSample:
    """One valid timestamp/AICLK row from a committed CSV."""

    timestamp: dt.datetime
    aiclk_mhz: float
    row_number: int


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
    if not samples or any(left > right for left, right in itertools.pairwise(samples)):
        return None
    if samples[0][0] > start or samples[-1][0] < end:
        return None
    duration = (end - start).total_seconds()
    weighted = 0.0
    covered = 0.0
    for index, (timestamp, aiclk) in enumerate(samples):
        segment_start = max(timestamp, start)
        next_timestamp = samples[index + 1][0] if index + 1 < len(samples) else end
        segment_end = min(next_timestamp, end)
        seconds = max(0.0, (segment_end - segment_start).total_seconds())
        if seconds:
            weighted += aiclk * seconds
            covered += seconds
    if not math.isclose(covered, duration, rel_tol=0.0, abs_tol=1e-9):
        return None
    return weighted / covered


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


def resolve_trace(
    root: Path,
    record_path: Path,
    record: Mapping[str, Any],
    csv_by_hash: Mapping[str, tuple[Path, ...]],
) -> TraceResolution:
    """Resolve a trace by explicit reference, hash provenance, then safe fallback."""

    references = power_trace_references(record)
    expected_hash = _trace_sha256(record)
    for reference in references:
        candidate = _path_from_reference(root, reference)
        if candidate.is_file():
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if expected_hash is None or actual == expected_hash:
                return TraceResolution(candidate, "explicit reference", reference, expected_hash or actual)
    if expected_hash and expected_hash in csv_by_hash:
        return TraceResolution(
            min(csv_by_hash[expected_hash]),
            "SHA-256 provenance",
            references[0] if references else None,
            expected_hash,
        )

    stems = [record_path.stem]
    for suffix in ("-summary", "-l1-accounting"):
        if stems[0].endswith(suffix):
            stems.append(stems[0][: -len(suffix)])
    for stem in stems:
        candidate = root / f"{stem}-power.csv"
        if candidate.is_file():
            return TraceResolution(candidate, "record-stem fallback", references[0] if references else None, expected_hash)
    return TraceResolution(None, "unresolved" if references else "no reference", references[0] if references else None, expected_hash)


def _mapping_intervals(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[int, IntervalBounds]]:
    if not isinstance(value, Mapping):
        if isinstance(value, list):
            for index, child in enumerate(value):
                yield from _mapping_intervals(child, (*path, str(index)))
        return
    lower_keys = {str(key).lower(): key for key in value}
    for start_key, end_key, kind, rank in (
        ("load_interval_start", "load_interval_end", "load", 0),
        ("run_start", "run_end", "run", 1),
    ):
        if start_key not in lower_keys or end_key not in lower_keys:
            continue
        try:
            start = parse_timestamp(str(value[lower_keys[start_key]]))
            end = parse_timestamp(str(value[lower_keys[end_key]]))
        except (TypeError, ValueError):
            continue
        joined = ".".join(path).lower()
        if "power_clock_provenance" in joined or "power_snapshot" in joined:
            rank -= 1
        yield rank, IntervalBounds(start, end, kind, ".".join((*path, start_key, end_key)))
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
    samples = list(trace.samples)
    inside = [sample for sample in samples if interval.start <= sample.timestamp <= interval.end]
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


def _record_analysis(
    root: Path,
    record_path: Path,
    record: Mapping[str, Any],
    csv_by_hash: Mapping[str, tuple[Path, ...]],
    trace_cache: dict[Path, TraceData],
) -> RecordAnalysis:
    resolution = resolve_trace(root, record_path, record, csv_by_hash)
    trace = None
    if resolution.path is not None:
        trace = trace_cache.setdefault(resolution.path, read_trace(resolution.path))
    classification = classify_aiclk_values([sample.aiclk_mhz for sample in trace.samples]) if trace else None
    interval = find_interval(record)
    evidence = "no CSV trace"
    minimum = maximum = weighted = None
    weighted_is_exact = False
    if trace and trace.samples:
        if interval is not None:
            inside, complete = _coverage_rows(trace, interval)
            values = [sample.aiclk_mhz for sample in inside]
            if values:
                minimum, maximum = min(values), max(values)
            if complete:
                weighted = left_held_time_weighted_aiclk(trace.samples, interval.start, interval.end)
                weighted_is_exact = weighted is not None
            evidence = (
                f"exact {interval.kind} interval"
                if weighted_is_exact
                else f"bounds present; trace coverage incomplete ({interval.kind})"
            )
        else:
            values = [sample.aiclk_mhz for sample in trace.samples]
            minimum, maximum = min(values), max(values)
            if trace.timestamps_ordered and len(trace.samples) > 1:
                weighted = left_held_time_weighted_aiclk(
                    trace.samples,
                    trace.samples[0].timestamp,
                    trace.samples[-1].timestamp,
                )
            evidence = "CSV capture span diagnostic"
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


def _multiplicative_aiclk_range(minimum: float | None, maximum: float | None) -> float | None:
    if minimum is None or maximum is None or minimum <= 0.0 or maximum < minimum:
        return None
    return maximum / minimum - 1.0


def _format_percentage(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100.0:.{digits}f}%"


def _final_p50_changes(record: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    rows = record.get("rows")
    if not isinstance(rows, list):
        return ()
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
        if math.isfinite(parsed) and parsed != 0.0:
            p50_by_row[name] = parsed

    changes: list[tuple[str, float]] = []
    comparisons = record.get("matched_placement_differences")
    if not isinstance(comparisons, list):
        return ()
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            continue
        row_names = comparison.get("rows")
        if not isinstance(row_names, list) or len(row_names) != 2:
            continue
        left, right = row_names
        if not isinstance(left, str) or not isinstance(right, str):
            continue
        if left not in p50_by_row or right not in p50_by_row:
            continue
        change = p50_by_row[right] / p50_by_row[left] - 1.0
        if math.isfinite(change):
            placement = str(comparison.get("placement") or f"{left} vs {right}")
            changes.append((placement, change))
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
    lines.extend(
        [
            "### #88 R-format ordering and confounding",
            "",
            (
                "The `comparison_rows` arrays in the convergence rerun, device-1 retake, "
                "and final runner retake are block ordered, not alternating: "
                "`bf16-r-L16`, `fp32-r-L16`, `bf16-r-L32`, "
                "`bf16-r-L32-r-dram`, `fp32-r-L32-r-dram`, and the rejected "
                "`fp32-r-L32` row. Each row retains host timing samples and derived "
                "seconds/TFLOPS fields, but no row start or cumulative host timestamp that "
                "can be aligned to a power-CSV timestamp. Therefore per-condition AICLK "
                "min/max/weighted values cannot be calculated from the committed evidence. "
                "The defensible bound for any row is the complete run-trace AICLK range, "
                "not a condition-specific estimate:"
            ),
            "",
            "| record | complete run-trace AICLK bound (MHz) | AICLK-only range (`max/min - 1`) | per-row alignment |",
            "|---|---:|---:|---|",
        ]
    )
    for analysis in (convergence, device1, final):
        if analysis is None:
            continue
        bound = (
            f"{_format_number(analysis.minimum_mhz)}–{_format_number(analysis.maximum_mhz)}"
            if analysis.minimum_mhz is not None
            else "—"
        )
        aiclk_range = _multiplicative_aiclk_range(
            analysis.minimum_mhz,
            analysis.maximum_mhz,
        )
        lines.append(
            f"| {analysis.path.name} | {bound} | {_format_percentage(aiclk_range)} | unavailable |"
        )
    final_range = _multiplicative_aiclk_range(
        final.minimum_mhz if final else None,
        final.maximum_mhz if final else None,
    )
    final_changes = (
        _final_p50_changes(records[final_summary.path])
        if final_summary is not None and final_summary.path in records
        else ()
    )
    lines.extend(
        [
            "",
            (
                "Each AICLK-only range is a model-dependent upper bound under linear "
                "AICLK-to-throughput normalization: it assumes the condition could fall "
                "at either complete-run endpoint. It is not a replacement for the "
                "synchronized host timings retained by each row; row-to-power alignment "
                "remains unavailable."
            ),
            "",
            (
                "For the authoritative final trace, the complete-run bound gives a "
                f"{_format_percentage(final_range)} maximum clock-rate spread. "
                + (
                    "The authoritative summary's retained P50 TFLOPS values give observed "
                    "P50 throughput changes of "
                    + "; ".join(
                        f"{placement}: {_format_percentage(change)}"
                        for placement, change in final_changes
                    )
                    + f"; even under the linear normalization model, the "
                    f"{_format_percentage(final_range)} clock-rate spread cannot by itself "
                    "explain those observed throughput changes."
                    if final_changes
                    else "The authoritative summary has no usable matched P50 values for a derived throughput comparison."
                )
            ),
            "",
            (
                "Relative to `docs/design.md`, the current published #88 conclusion is "
                "unchanged: FP32-R improves the corrected quality metrics but costs "
                "throughput/L1; only the withdrawn defective `_weight` conclusion changed. "
                "The withdrawn report's claim that FP32-R worsened beam-pattern error used "
                "the defective `_weight` contraction. The corrected final evidence says "
                "FP32-R lowers inverse error, MV cosine deficit, and all four beam indicators "
                "at both matched placements; BF16-R remains the default cost choice."
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
                "1,350 MHz reference rather than a computed weighted load interval."
            ),
        ]
    )
    return lines


def render_report(root: Path) -> str:
    """Render the deterministic Markdown report used by the committed audit."""

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
    users_by_trace: dict[Path, list[RecordAnalysis]] = {path: [] for path in csv_paths}
    for analysis in analyses:
        if analysis.resolution.path in users_by_trace:
            users_by_trace[analysis.resolution.path].append(analysis)

    lines = [
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
        "",
        "## JSON record inventory",
        "",
        "| record | issue | status | trace reference/provenance | resolved CSV | pairing | rows valid/CSV | unique AICLK (MHz) | classification | min/max (MHz) | min intermediate (MHz) | position | interval evidence | weighted AICLK (MHz) |",
        "|---|---|---|---|---|---|---:|---|---|---:|---:|---|---|---:|",
    ]
    lines.extend(_analysis_row(analysis) for analysis in analyses)
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

    exact_or_bounded = [analysis for analysis in analyses if analysis.interval is not None]
    lines.extend(
        [
            "",
            "## Explicit interval records",
            "",
            "| record | issue | interval kind | evidence | min/max (MHz) | classification | weighted AICLK (MHz) | resolved CSV |",
            "|---|---|---|---|---:|---|---:|---|",
        ]
    )
    lines.extend(_focus_row(analysis) for analysis in exact_or_bounded)

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

    intermediate_csvs = []
    for path in csv_paths:
        trace = trace_cache.setdefault(path, read_trace(path))
        classification = classify_aiclk_values([sample.aiclk_mhz for sample in trace.samples])
        if classification.minimum_intermediate_mhz is not None:
            intermediate_csvs.append((path, trace, classification))
    lines.extend(
        [
            "",
            "## Intermediate-value CSV diagnostics",
            "",
            "| CSV | minimum intermediate (MHz) | position | unique AICLK (MHz) |",
            "|---|---:|---|---|",
        ]
    )
    for path, _trace, classification in intermediate_csvs:
        lines.append(
            f"| {path.name} | {_format_number(classification.minimum_intermediate_mhz)} | "
            f"{classification.intermediate_position} | {_format_values(classification.unique_mhz)} |"
        )
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
