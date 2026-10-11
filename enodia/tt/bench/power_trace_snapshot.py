"""Run-bound immutable power-trace snapshots and coverage evaluation.

The wrapper-owned sampler writes a live CSV while a runner is executing.  A
runner must copy the bytes it judges into a never-overwritten, run-bound
snapshot before constructing its record.  This module is the shared
snapshot/read/hash lifecycle for board runners; callers select the coverage
contract without reimplementing CSV parsing or artifact ownership.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import io
import itertools
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enodia.tt.bench.run_binding import (
    RUN_ID_PATTERN,
    RunBindingError,
    require_run_artifact,
    validate_run_id,
)
from enodia.tt.bench.sampler_contract import (
    COVERAGE_DEFINITIONS,
    EXPLICIT_FINAL_SAMPLE,
    ONE_INTERVAL_BOUND,
)
from enodia.tt.bench.telemetry import (
    POWER_TRACE_COLUMNS,
    _parse_trace_timestamp,
    _trace_number,
)

DEFAULT_SNAPSHOT_PREFIX = "power-trace-snapshot"
POWER_TRACE_SHA256_DEFINITION = (
    "SHA-256 of the complete immutable per-run power CSV snapshot bytes"
)


@dataclass(frozen=True)
class PowerTraceReadOptions:
    """Parser settings that apply to the CSV rows rather than run identity."""

    row_label: str = "power sample"
    tolerate_invalid_run_bounds: bool = False

# These descriptions were emitted by Issue #88 before coverage definitions
# became named contract values.  They remain readable as explicit-final-
# sample records, but new records use the canonical name above.
_LEGACY_EXPLICIT_FINAL_SAMPLE_DESCRIPTIONS = frozenset(
    {
        (
            "Coverage is complete when the immutable per-run snapshot has at least one "
            "usable finite telemetry row, usable timestamps parse in order, its first "
            "usable sample is at or before the recorded run start, and its last usable "
            "sample timestamp is at or after the recorded run end; the run interval is "
            "covered through its end. Rows with unusable power, AICLK, or temperature "
            "values are excluded from these checks."
        ),
        (
            "Coverage is complete when the immutable per-run snapshot is readable and "
            "nonempty, timestamps parse in order, its first sample is at or before the "
            "recorded run start, and its last sample timestamp is at or after the "
            "recorded run end; the run interval is covered through its end."
        ),
    }
)


def canonical_coverage_definition(value: object | None) -> str:
    """Return a named coverage definition, accepting historical Issue #88 text."""
    if value is None:
        return EXPLICIT_FINAL_SAMPLE
    if value in COVERAGE_DEFINITIONS:
        return str(value)
    if value in _LEGACY_EXPLICIT_FINAL_SAMPLE_DESCRIPTIONS:
        return EXPLICIT_FINAL_SAMPLE
    raise ValueError(
        "coverage_definition must be one of "
        f"{', '.join(COVERAGE_DEFINITIONS)}"
    )


def is_coverage_definition(value: object) -> bool:
    """Return whether a persisted value names a supported coverage contract."""
    try:
        canonical_coverage_definition(value)
    except (TypeError, ValueError):
        return False
    return True


def _snapshot_name_pattern(prefix: str) -> re.Pattern[str]:
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError("snapshot_prefix must be a nonempty basename")
    return re.compile(
        rf"^{re.escape(prefix)}-(?P<run_id>{RUN_ID_PATTERN})-"
        rf"(?P<number>[1-9][0-9]*)\.csv$"
    )


def validate_power_trace_snapshot_path(
    path: Path,
    *,
    run_id: str,
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX,
    output_dir: Path | None = None,
) -> Path:
    """Require a non-symlink snapshot basename that encodes the current run."""
    path = Path(path)
    run_id = validate_run_id(run_id)
    pattern = _snapshot_name_pattern(snapshot_prefix)
    if output_dir is not None:
        expected_dir = Path(output_dir).resolve()
        try:
            same_directory = path.resolve().parent == expected_dir
        except (OSError, RuntimeError):
            same_directory = False
        if not same_directory or path.is_symlink():
            raise RunBindingError(
                "power trace snapshot is not in the current run output directory"
            )
    match = pattern.fullmatch(path.name)
    if match is None:
        raise RunBindingError(
            "power trace snapshot filename does not encode a run-bound snapshot"
        )
    if match.group("run_id") != run_id:
        raise RunBindingError(
            "power trace snapshot run_id does not match the current run "
            f"({match.group('run_id')!r} != {run_id!r})"
        )
    return path


def atomic_power_trace_snapshot(
    source_path: Path,
    output_dir: Path,
    *,
    run_id: str,
    snapshot_number: int,
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX,
    read_text_fn: Callable[[Path], str] | None = None,
) -> Path:
    """Copy one live sampler read to an atomic, never-overwritten snapshot."""
    run_id = validate_run_id(run_id)
    if (
        isinstance(snapshot_number, bool)
        or not isinstance(snapshot_number, int)
        or snapshot_number < 1
    ):
        raise ValueError("snapshot_number must be a positive integer")
    output_dir = Path(output_dir).resolve()
    source_path = require_run_artifact(
        output_dir,
        prefix="power-",
        suffix=".csv",
        run_id=run_id,
        explicit=Path(source_path),
        kind="power trace source",
    )
    payload = (
        source_path.read_bytes()
        if read_text_fn is None
        else read_text_fn(source_path).encode()
    )
    _snapshot_name_pattern(snapshot_prefix)
    snapshot_name = f"{snapshot_prefix}-{run_id}-{snapshot_number}.csv"
    if Path(snapshot_name).name != snapshot_name:
        raise ValueError("power trace snapshot name must be a basename")
    snapshot_path = output_dir / snapshot_name
    if snapshot_path.parent != output_dir:
        raise ValueError("power trace snapshot must stay in the output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{snapshot_path.name}.", suffix=".tmp", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # The temporary file is complete before this same-directory hard-link
        # creates the destination.  Link creation is atomic and exclusive:
        # competing writers get FileExistsError instead of replacing bytes.
        os.link(temporary, snapshot_path)
    finally:
        temporary.unlink(missing_ok=True)
    return snapshot_path


def _timestamp_text(value: datetime.datetime | str | None, *, field: str) -> str | None:
    if value is None:
        return None
    return _parse_trace_timestamp(value, field=field).isoformat()


def _error_value(
    error_factory: Callable[[str, Exception | None], Any] | None,
    kind: str,
    error: Exception | None = None,
) -> Any:
    if error_factory is not None:
        return error_factory(kind, error)
    return {"code": kind}


def _trace_base(
    path: Path | None,
    *,
    run_id: str | None,
    run_start: str | None,
    run_end: str | None,
    coverage_definition: str,
    source_file: str | None,
    immutable_snapshot: bool = True,
) -> dict[str, Any]:
    return {
        "file": path.name if path is not None else None,
        "source_file": source_file,
        "run_id": run_id,
        "columns": list(POWER_TRACE_COLUMNS),
        "samples": [],
        "sampling_source": "immutable per-run CSV snapshot captured by run_in_container.sh",
        "sha256": None,
        "sha256_definition": POWER_TRACE_SHA256_DEFINITION,
        "byte_count": None,
        "immutable_snapshot": immutable_snapshot,
        "csv_row_count": 0,
        "sample_count": 0,
        "valid_row_count": 0,
        "invalid_row_count": 0,
        "in_run_valid_row_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "run_start": run_start,
        "run_end": run_end,
        "readable": False,
        "nonempty": False,
        "timestamps_parse": False,
        "timestamps_ordered": False,
        "covers_run_start": False,
        "covers_run_end": False,
        "coverage_complete": False,
        "coverage_definition": coverage_definition,
        "poll_complete": False,
        "errors": [],
        "aiclk_source": "no_valid_in_run_samples",
        "aiclk_mhz": None,
        "aiclk_mhz_in_run": [],
        "aiclk_mhz_in_run_min": None,
        "aiclk_mhz_in_run_max": None,
        "aiclk_mhz_load_interval_weighted": None,
        "aiclk_mhz_load_interval_weighted_source": "unavailable",
        "aiclk_mhz_load_interval_weighted_seconds": 0.0,
        "load_interval_start": None,
        "load_interval_end": None,
        "load_interval_source": None,
        "in_run_samples": [],
        "aiclk_trace_in_run": [],
        "coverage": {
            "definition": coverage_definition,
            "nonempty": False,
            "timestamps_parse": False,
            "timestamps_ordered": False,
            "first_at_or_before_run_start": False,
            "last_at_or_after_run_end": False,
            "last_within_interval_bound": False,
            "readable": False,
            "complete": False,
        },
    }


def _valid_interval_seconds(value: float | None) -> float | None:
    try:
        interval = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return interval if math.isfinite(interval) and interval > 0.0 else None


def timestamp_within_interval_bound(
    sample_timestamp: datetime.datetime,
    run_end: datetime.datetime,
    sampler_interval_seconds: float | None,
) -> bool:
    """Return whether a sample is inside the inclusive one-interval bound."""
    interval = _valid_interval_seconds(sampler_interval_seconds)
    if interval is None:
        return False
    interval_start = run_end - datetime.timedelta(seconds=interval)
    return interval_start <= sample_timestamp <= run_end


def _run_timestamp_text(
    value: datetime.datetime | str | None,
    *,
    field: str,
    tolerate_invalid: bool,
) -> tuple[str | None, ValueError | None]:
    try:
        return _timestamp_text(value, field=field), None
    except ValueError as exc:
        if not tolerate_invalid:
            raise
        return None, exc


def _read_trace_samples(
    raw_bytes: bytes, *, reject_incomplete_rows: bool
) -> list[dict[str, Any]]:
    text = raw_bytes.decode()
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = tuple(reader.fieldnames or ())
    if fieldnames != POWER_TRACE_COLUMNS:
        raise ValueError(f"power trace has unexpected columns {fieldnames!r}")
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(reader):
        incomplete = set(row) != set(POWER_TRACE_COLUMNS) or any(
            row.get(column) is None for column in POWER_TRACE_COLUMNS
        )
        if reject_incomplete_rows and incomplete:
            raise ValueError(f"power trace sample {index} is an incomplete CSV row")
        samples.append(dict(row))
    return samples


def _usable_trace_rows(
    samples: list[dict[str, Any]],
    *,
    error_factory: Callable[[str, Exception | None], Any] | None,
    row_label: str,
) -> tuple[
    list[tuple[dict[str, Any], datetime.datetime, float]],
    list[Any],
    Any | None,
]:
    errors: list[Any] = []
    timestamp_error: Any | None = None
    usable_rows: list[tuple[dict[str, Any], datetime.datetime, float]] = []
    for index, sample in enumerate(samples):
        if None in sample or any(column not in sample for column in POWER_TRACE_COLUMNS):
            message = f"{row_label} {index} does not have exactly the declared columns"
            errors.append(
                _error_value(error_factory, "power_row_incomplete", ValueError(message))
            )
            continue
        try:
            timestamp = _parse_trace_timestamp(
                sample.get("timestamp_utc"),
                field=f"{row_label} {index} timestamp_utc",
            )
        except ValueError as exc:
            timestamp_error = timestamp_error or _error_value(
                error_factory, "power_timestamp_invalid", exc
            )
            errors.append(_error_value(error_factory, "power_timestamp_invalid", exc))
            continue
        try:
            _trace_number(sample.get("power_w"), field=f"{row_label} {index} power_w")
            aiclk = _trace_number(
                sample.get("aiclk_mhz"),
                field=f"{row_label} {index} aiclk_mhz",
                positive=True,
            )
            _trace_number(
                sample.get("asic_temp_c"), field=f"{row_label} {index} asic_temp_c"
            )
        except ValueError as exc:
            errors.append(_error_value(error_factory, "power_sample_invalid", exc))
        else:
            usable_rows.append((sample, timestamp, aiclk))
    return usable_rows, errors, timestamp_error


def _set_trace_row_facts(
    trace: dict[str, Any],
    samples: list[dict[str, Any]],
    usable_rows: list[tuple[dict[str, Any], datetime.datetime, float]],
    errors: list[Any],
    *,
    require_all_rows_valid: bool,
) -> bool:
    usable_samples = [sample for sample, _timestamp, _aiclk in usable_rows]
    trace["samples"] = usable_samples
    trace["sample_count"] = (
        len(samples) if require_all_rows_valid else len(usable_samples)
    )
    trace["valid_row_count"] = len(usable_samples)
    trace["invalid_row_count"] = len(samples) - len(usable_samples)
    trace["nonempty"] = bool(samples) if require_all_rows_valid else bool(usable_samples)
    trace["errors"] = errors
    return trace["valid_row_count"] == trace["csv_row_count"]


def _set_trace_timestamp_facts(
    trace: dict[str, Any], parsed_timestamps: list[datetime.datetime]
) -> None:
    trace["timestamps_parse"] = True
    trace["first_timestamp"] = parsed_timestamps[0].isoformat()
    trace["last_timestamp"] = parsed_timestamps[-1].isoformat()
    trace["timestamps_ordered"] = all(
        left <= right for left, right in itertools.pairwise(parsed_timestamps)
    )
    trace["_last_timestamp_datetime"] = parsed_timestamps[-1]


def _time_weighted_aiclk(
    usable_rows: list[tuple[dict[str, Any], datetime.datetime, float]],
    *,
    start: datetime.datetime,
    end: datetime.datetime,
) -> tuple[float | None, float, str]:
    """Return a left-held AICLK average over one explicit trace interval.

    A sample owns the time from its timestamp through the next sample.  The
    final sample owns the remainder through the requested endpoint.  The
    interval must be bounded by usable samples; extrapolating a clock outside
    the immutable trace would turn a diagnostic snapshot into timing evidence.
    """
    if start > end:
        return None, 0.0, "load_interval_end_precedes_start"
    duration = (end - start).total_seconds()
    if duration <= 0.0:
        return None, 0.0, "load_interval_is_empty"
    if not usable_rows:
        return None, 0.0, "load_interval_has_no_valid_samples"
    if usable_rows[0][1] > start or usable_rows[-1][1] < end:
        return None, 0.0, "load_interval_not_bounded_by_trace"

    weighted = 0.0
    covered = 0.0
    for index, (_sample, timestamp, aiclk) in enumerate(usable_rows):
        segment_start = max(timestamp, start)
        next_timestamp = (
            usable_rows[index + 1][1] if index + 1 < len(usable_rows) else end
        )
        segment_end = min(next_timestamp, end)
        seconds = max(0.0, (segment_end - segment_start).total_seconds())
        if seconds:
            weighted += aiclk * seconds
            covered += seconds
    if not math.isclose(covered, duration, rel_tol=0.0, abs_tol=1e-9):
        return None, covered, "load_interval_trace_coverage_incomplete"
    return weighted / covered, covered, "load_interval_trace_samples"


def _set_trace_run_facts(
    trace: dict[str, Any],
    usable_rows: list[tuple[dict[str, Any], datetime.datetime, float]],
    parsed_timestamps: list[datetime.datetime],
    *,
    start_text: str | None,
    end_text: str | None,
    load_interval_start_text: str | None,
    load_interval_end_text: str | None,
    load_interval_source: str | None,
    definition: str,
    interval: float | None,
    require_all_rows_valid: bool,
    all_rows_valid: bool,
    error_factory: Callable[[str, Exception | None], Any] | None,
    append_run_errors: bool,
) -> None:
    try:
        start = _parse_trace_timestamp(start_text, field="run_start")
        end = _parse_trace_timestamp(end_text, field="run_end")
    except ValueError as exc:
        error = _error_value(error_factory, "power_timestamp_invalid", exc)
        trace["error"] = error
        if append_run_errors:
            trace["errors"].append(error)
        return

    if start > end and append_run_errors:
        trace["errors"].append("run_start must be at or before run_end")
    trace["covers_run_start"] = parsed_timestamps[0] <= start <= end
    trace["covers_run_end"] = parsed_timestamps[-1] >= end
    in_run = [
        aiclk
        for _sample, timestamp, aiclk in usable_rows
        if start <= timestamp <= end
    ]
    trace["in_run_valid_row_count"] = len(in_run)
    trace["aiclk_mhz_in_run"] = in_run
    trace["in_run_samples"] = [
        dict(sample)
        for sample, timestamp, _aiclk in usable_rows
        if start <= timestamp <= end
    ]
    trace["aiclk_trace_in_run"] = list(trace["in_run_samples"])
    if in_run:
        trace["aiclk_source"] = "run_trace_samples"
        trace["aiclk_mhz"] = max(in_run)
        trace["aiclk_mhz_in_run_min"] = min(in_run)
        trace["aiclk_mhz_in_run_max"] = max(in_run)

    load_start_text = load_interval_start_text or start_text
    load_end_text = load_interval_end_text or end_text
    trace["load_interval_start"] = load_start_text
    trace["load_interval_end"] = load_end_text
    trace["load_interval_source"] = load_interval_source or (
        "run_bounds" if load_start_text is not None and load_end_text is not None else None
    )
    try:
        load_start = _parse_trace_timestamp(load_start_text, field="load_interval_start")
        load_end = _parse_trace_timestamp(load_end_text, field="load_interval_end")
    except ValueError as exc:
        trace["aiclk_mhz_load_interval_weighted_source"] = "unavailable"
        trace["aiclk_mhz_load_interval_weighted_seconds"] = 0.0
        trace["errors"].append(_error_value(error_factory, "load_interval_invalid", exc))
    else:
        weighted, covered, reason = _time_weighted_aiclk(
            usable_rows, start=load_start, end=load_end
        )
        trace["aiclk_mhz_load_interval_weighted_seconds"] = covered
        if weighted is not None:
            trace["aiclk_mhz_load_interval_weighted"] = weighted
            trace["aiclk_mhz_load_interval_weighted_source"] = (
                "load_interval_trace_samples"
            )
        else:
            trace["aiclk_mhz_load_interval_weighted_source"] = reason

    interval_bound = False
    if interval is not None:
        interval_bound = timestamp_within_interval_bound(
            parsed_timestamps[-1], end, interval
        )
        trace["sampler_interval_seconds"] = interval
    trace["coverage_complete"] = all(
        (
            trace["readable"],
            trace["nonempty"],
            trace["valid_row_count"] > 0,
            not require_all_rows_valid or all_rows_valid,
            trace["timestamps_parse"],
            trace["timestamps_ordered"],
            parsed_timestamps[0] <= start,
            start <= end,
            (
                end <= parsed_timestamps[-1]
                if definition == EXPLICIT_FINAL_SAMPLE
                else interval_bound
            ),
        )
    )
    trace["coverage"]["last_within_interval_bound"] = interval_bound


def _set_trace_coverage(trace: dict[str, Any], definition: str) -> dict[str, Any]:
    trace["coverage"] = {
        **trace["coverage"],
        "definition": definition,
        "nonempty": trace["nonempty"],
        "timestamps_parse": trace["timestamps_parse"],
        "timestamps_ordered": trace["timestamps_ordered"],
        "first_at_or_before_run_start": trace["covers_run_start"],
        "last_at_or_after_run_end": trace["covers_run_end"],
        "readable": trace["readable"],
        "usable_row_count": trace["valid_row_count"],
        "invalid_row_count": trace["invalid_row_count"],
        "complete": trace["coverage_complete"],
    }
    return trace


def _evaluate_trace_rows(
    trace: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    start_text: str | None,
    end_text: str | None,
    load_interval_start_text: str | None,
    load_interval_end_text: str | None,
    load_interval_source: str | None,
    definition: str,
    interval: float | None,
    require_all_rows_valid: bool,
    error_factory: Callable[[str, Exception | None], Any] | None,
    row_label: str,
    initial_errors: list[Any],
    append_run_errors: bool,
) -> dict[str, Any]:
    trace["readable"] = True
    trace["csv_row_count"] = len(samples)
    usable_rows, row_errors, timestamp_error = _usable_trace_rows(
        samples, error_factory=error_factory, row_label=row_label
    )
    errors = [*initial_errors, *row_errors]
    parsed_timestamps = [timestamp for _sample, timestamp, _aiclk in usable_rows]
    all_rows_valid = _set_trace_row_facts(
        trace,
        samples,
        usable_rows,
        errors,
        require_all_rows_valid=require_all_rows_valid,
    )
    if not usable_rows:
        trace["error"] = timestamp_error or _error_value(
            error_factory, "power_trace_no_usable_rows"
        )
    elif timestamp_error is not None:
        trace["error"] = timestamp_error
    elif require_all_rows_valid and not all_rows_valid:
        trace["error"] = errors[0]
    else:
        _set_trace_timestamp_facts(trace, parsed_timestamps)
        _set_trace_run_facts(
            trace,
            usable_rows,
            parsed_timestamps,
            start_text=start_text,
            end_text=end_text,
            definition=definition,
            interval=interval,
            load_interval_start_text=load_interval_start_text,
            load_interval_end_text=load_interval_end_text,
            load_interval_source=load_interval_source,
            require_all_rows_valid=require_all_rows_valid,
            all_rows_valid=all_rows_valid,
            error_factory=error_factory,
            append_run_errors=append_run_errors,
        )
    return _set_trace_coverage(trace, definition)


def read_power_trace_csv(
    path: Path,
    *,
    run_id: str | None = None,
    run_start: datetime.datetime | str | None,
    run_end: datetime.datetime | str | None,
    source_file: str | None = None,
    sampler_interval_seconds: float | None = None,
    coverage_definition: str = EXPLICIT_FINAL_SAMPLE,
    load_interval_start: datetime.datetime | str | None = None,
    load_interval_end: datetime.datetime | str | None = None,
    load_interval_source: str | None = None,
    require_all_rows_valid: bool = False,
    read_text_fn: Callable[[Path], str] | None = None,
    error_factory: Callable[[str, Exception | None], Any] | None = None,
    immutable_snapshot: bool = False,
    reject_incomplete_rows: bool = True,
    read_options: PowerTraceReadOptions | None = None,
) -> dict[str, Any]:
    """Read power CSV bytes and evaluate one named run-coverage contract."""
    options = read_options or PowerTraceReadOptions()
    definition = canonical_coverage_definition(coverage_definition)
    path = Path(path)
    start_text, start_error = _run_timestamp_text(
        run_start, field="run_start", tolerate_invalid=options.tolerate_invalid_run_bounds
    )
    end_text, end_error = _run_timestamp_text(
        run_end, field="run_end", tolerate_invalid=options.tolerate_invalid_run_bounds
    )
    load_start_text, load_start_error = _run_timestamp_text(
        load_interval_start if load_interval_start is not None else run_start,
        field="load_interval_start",
        tolerate_invalid=options.tolerate_invalid_run_bounds,
    )
    load_end_text, load_end_error = _run_timestamp_text(
        load_interval_end if load_interval_end is not None else run_end,
        field="load_interval_end",
        tolerate_invalid=options.tolerate_invalid_run_bounds,
    )
    initial_errors = [
        _error_value(error_factory, "power_timestamp_invalid", error)
        for error in (
            start_error,
            end_error,
            load_start_error,
            load_end_error,
        )
        if error is not None
    ]
    trace = _trace_base(
        path,
        run_id=run_id,
        run_start=start_text,
        run_end=end_text,
        coverage_definition=definition,
        source_file=source_file,
        immutable_snapshot=immutable_snapshot,
    )
    trace["errors"] = initial_errors
    trace["load_interval_start"] = load_start_text
    trace["load_interval_end"] = load_end_text
    trace["load_interval_source"] = load_interval_source or (
        "run_bounds"
        if load_start_text is not None and load_end_text is not None
        else None
    )
    interval = (
        _valid_interval_seconds(sampler_interval_seconds)
        if definition == ONE_INTERVAL_BOUND
        else None
    )
    if definition == ONE_INTERVAL_BOUND:
        trace["sampler_interval_seconds"] = interval
    try:
        raw_bytes = (
            path.read_bytes() if read_text_fn is None else read_text_fn(path).encode()
        )
        trace["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        trace["byte_count"] = len(raw_bytes)
        samples = _read_trace_samples(
            raw_bytes, reject_incomplete_rows=reject_incomplete_rows
        )
    except Exception as exc:  # noqa: BLE001 - return a diagnostic trace
        kind = "power_trace_unreadable" if isinstance(exc, OSError) else "power_trace_invalid"
        trace["error"] = _error_value(error_factory, kind, exc)
        return trace
    return _evaluate_trace_rows(
        trace,
        samples,
        start_text=start_text,
        end_text=end_text,
        load_interval_start_text=load_start_text,
        load_interval_end_text=load_end_text,
        load_interval_source=load_interval_source,
        definition=definition,
        interval=interval,
        require_all_rows_valid=require_all_rows_valid,
        error_factory=error_factory,
        row_label=options.row_label,
        initial_errors=initial_errors,
        append_run_errors=options.tolerate_invalid_run_bounds,
    )


def empty_power_trace_metadata(
    path: Path | None = None,
    *,
    run_id: str | None = None,
    run_start: datetime.datetime | str | None = None,
    run_end: datetime.datetime | str | None = None,
    source_file: str | None = None,
    sampler_interval_seconds: float | None = None,
    coverage_definition: str = EXPLICIT_FINAL_SAMPLE,
    immutable_snapshot: bool = False,
    aiclk_source: str = "no_valid_in_run_samples",
    errors: list[Any] | None = None,
    tolerate_invalid_run_bounds: bool = False,
    include_issue118_metadata: bool = False,
) -> dict[str, Any]:
    """Return shared diagnostic metadata when no CSV bytes can be read."""
    definition = canonical_coverage_definition(coverage_definition)
    start_text, _start_error = _run_timestamp_text(
        run_start, field="run_start", tolerate_invalid=tolerate_invalid_run_bounds
    )
    end_text, _end_error = _run_timestamp_text(
        run_end, field="run_end", tolerate_invalid=tolerate_invalid_run_bounds
    )
    trace = _trace_base(
        Path(path) if path is not None else None,
        run_id=run_id,
        run_start=start_text,
        run_end=end_text,
        coverage_definition=definition,
        source_file=source_file,
        immutable_snapshot=immutable_snapshot,
    )
    if definition == ONE_INTERVAL_BOUND:
        trace["sampler_interval_seconds"] = _valid_interval_seconds(
            sampler_interval_seconds
        )
    if tolerate_invalid_run_bounds:
        trace["run_start"] = run_start
        trace["run_end"] = run_end
    trace["aiclk_source"] = aiclk_source
    trace["errors"] = list(errors or [])
    if not include_issue118_metadata:
        for field in (
            "aiclk_mhz_in_run_min",
            "aiclk_mhz_in_run_max",
            "aiclk_mhz_load_interval_weighted",
            "aiclk_mhz_load_interval_weighted_source",
            "aiclk_mhz_load_interval_weighted_seconds",
            "load_interval_start",
            "load_interval_end",
            "load_interval_source",
            "in_run_samples",
            "aiclk_trace_in_run",
        ):
            trace.pop(field, None)
    return trace


def read_power_trace_snapshot(
    path: Path,
    *,
    run_id: str,
    run_start: datetime.datetime | str | None,
    run_end: datetime.datetime | str | None,
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX,
    source_path: Path | None = None,
    sampler_interval_seconds: float | None = None,
    coverage_definition: str = EXPLICIT_FINAL_SAMPLE,
    load_interval_start: datetime.datetime | str | None = None,
    load_interval_end: datetime.datetime | str | None = None,
    load_interval_source: str | None = None,
    include_issue118_metadata: bool = False,
    require_all_rows_valid: bool = False,
    read_text_fn: Callable[[Path], str] | None = None,
    error_factory: Callable[[str, Exception | None], Any] | None = None,
) -> dict[str, Any]:
    """Read one immutable snapshot through the shared CSV lifecycle.

    ``explicit_final_sample`` requires a usable sample at or after ``run_end``.
    ``one_interval_bound`` accepts a final usable sample in the inclusive
    interval ``run_end - sampler_interval_seconds <= sample_timestamp <=
    run_end``.  The interval is an explicit caller/record setting and is never
    inferred from CSV timestamps.  The default retains the Issue #88 usable-row
    compatibility behavior; resident callers opt into all-row validity. The
    ``include_issue118_metadata`` switch preserves the historical #111/#12
    public snapshot shape unless the Issue #118 fields are explicitly requested.
    """
    definition = canonical_coverage_definition(coverage_definition)
    path = validate_power_trace_snapshot_path(
        path, run_id=run_id, snapshot_prefix=snapshot_prefix
    )
    source_file = None
    if source_path is not None:
        source_path = require_run_artifact(
            path.parent,
            prefix="power-",
            suffix=".csv",
            run_id=run_id,
            explicit=Path(source_path),
            kind="power trace source",
        )
        source_file = source_path.name
    trace = read_power_trace_csv(
        path,
        run_id=run_id,
        run_start=run_start,
        run_end=run_end,
        source_file=source_file,
        sampler_interval_seconds=sampler_interval_seconds,
        coverage_definition=definition,
        load_interval_start=load_interval_start,
        load_interval_end=load_interval_end,
        load_interval_source=load_interval_source,
        require_all_rows_valid=require_all_rows_valid,
        read_text_fn=read_text_fn,
        error_factory=error_factory,
        immutable_snapshot=True,
    )
    if not include_issue118_metadata:
        for field in (
            "aiclk_mhz_in_run_min",
            "aiclk_mhz_in_run_max",
            "aiclk_mhz_load_interval_weighted",
            "aiclk_mhz_load_interval_weighted_source",
            "aiclk_mhz_load_interval_weighted_seconds",
            "load_interval_start",
            "load_interval_end",
            "load_interval_source",
            "in_run_samples",
            "aiclk_trace_in_run",
        ):
            trace.pop(field, None)
    return trace


_POWER_TRACE_METADATA_FIELDS = (
    "file",
    "columns",
    "sha256",
    "readable",
    "nonempty",
    "csv_row_count",
    "sample_count",
    "valid_row_count",
    "in_run_valid_row_count",
    "timestamps_parse",
    "timestamps_ordered",
    "first_timestamp",
    "last_timestamp",
    "run_start",
    "run_end",
    "covers_run_start",
    "covers_run_end",
    "coverage_complete",
    "coverage_definition",
    "sampler_interval_seconds",
    "coverage",
    "source_file",
    "immutable_snapshot",
    "byte_count",
    "sha256_definition",
    "aiclk_source",
    "aiclk_mhz",
    "aiclk_mhz_in_run",
    "aiclk_mhz_in_run_min",
    "aiclk_mhz_in_run_max",
    "aiclk_mhz_load_interval_weighted",
    "aiclk_mhz_load_interval_weighted_source",
    "aiclk_mhz_load_interval_weighted_seconds",
    "load_interval_start",
    "load_interval_end",
    "load_interval_source",
    "in_run_samples",
    "aiclk_trace_in_run",
    "errors",
)


def public_power_trace_metadata(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Keep shared trace facts while excluding parser-only row arrays."""
    return {key: trace.get(key) for key in _POWER_TRACE_METADATA_FIELDS}


def public_power_trace_snapshot(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Drop parser-only state while retaining immutable snapshot metadata."""
    public = dict(trace)
    public.pop("_last_timestamp_datetime", None)
    return public


__all__ = [
    "COVERAGE_DEFINITIONS",
    "DEFAULT_SNAPSHOT_PREFIX",
    "EXPLICIT_FINAL_SAMPLE",
    "ONE_INTERVAL_BOUND",
    "POWER_TRACE_COLUMNS",
    "POWER_TRACE_SHA256_DEFINITION",
    "PowerTraceReadOptions",
    "atomic_power_trace_snapshot",
    "canonical_coverage_definition",
    "empty_power_trace_metadata",
    "is_coverage_definition",
    "public_power_trace_metadata",
    "public_power_trace_snapshot",
    "read_power_trace_csv",
    "read_power_trace_snapshot",
    "timestamp_within_interval_bound",
    "validate_power_trace_snapshot_path",
]
