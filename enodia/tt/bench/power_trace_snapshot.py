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
    path: Path,
    *,
    run_id: str,
    run_start: str | None,
    run_end: str | None,
    coverage_definition: str,
    source_file: str | None,
) -> dict[str, Any]:
    return {
        "file": path.name,
        "source_file": source_file,
        "run_id": run_id,
        "columns": list(POWER_TRACE_COLUMNS),
        "samples": [],
        "sampling_source": "immutable per-run CSV snapshot captured by run_in_container.sh",
        "sha256": None,
        "sha256_definition": POWER_TRACE_SHA256_DEFINITION,
        "byte_count": None,
        "immutable_snapshot": True,
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
    require_all_rows_valid: bool = False,
    read_text_fn: Callable[[Path], str] | None = None,
    error_factory: Callable[[str, Exception | None], Any] | None = None,
) -> dict[str, Any]:
    """Read one immutable snapshot and evaluate its named coverage contract.

    ``explicit_final_sample`` requires a usable sample at or after ``run_end``.
    ``one_interval_bound`` accepts a final usable sample in the inclusive
    interval ``run_end - sampler_interval_seconds <= sample_timestamp <=
    run_end``.  The interval is an explicit caller/record setting and is never
    inferred from CSV timestamps.  When ``require_all_rows_valid`` is true,
    invalid rows do not contribute timestamp, coverage, or in-run AICLK facts;
    this is the resident-run contract.  The default retains the Issue #88
    usable-row compatibility behavior.
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
    start_text = _timestamp_text(run_start, field="run_start")
    end_text = _timestamp_text(run_end, field="run_end")
    trace = _trace_base(
        path,
        run_id=run_id,
        run_start=start_text,
        run_end=end_text,
        coverage_definition=definition,
        source_file=source_file,
    )
    if definition == ONE_INTERVAL_BOUND:
        try:
            interval = float(sampler_interval_seconds)
        except (TypeError, ValueError, OverflowError):
            interval = float("nan")
        trace["sampler_interval_seconds"] = (
            interval if math.isfinite(interval) and interval > 0.0 else None
        )
    try:
        raw_bytes = (
            path.read_bytes() if read_text_fn is None else read_text_fn(path).encode()
        )
        trace["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        trace["byte_count"] = len(raw_bytes)
        text = raw_bytes.decode()
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != POWER_TRACE_COLUMNS:
            raise ValueError(f"power trace has unexpected columns {fieldnames!r}")
        samples = []
        for index, row in enumerate(reader):
            if set(row) != set(POWER_TRACE_COLUMNS) or any(
                row.get(column) is None for column in POWER_TRACE_COLUMNS
            ):
                raise ValueError(f"power trace sample {index} is an incomplete CSV row")
            samples.append(dict(row))
    except Exception as exc:  # noqa: BLE001 - return a diagnostic trace
        kind = "power_trace_unreadable" if isinstance(exc, OSError) else "power_trace_invalid"
        trace["error"] = _error_value(error_factory, kind, exc)
        return trace

    trace["readable"] = True
    trace["csv_row_count"] = len(samples)
    errors: list[Any] = []
    timestamp_error: Any | None = None
    usable_rows: list[tuple[dict[str, str], datetime.datetime, float]] = []
    for index, sample in enumerate(samples):
        try:
            timestamp = _parse_trace_timestamp(
                sample.get("timestamp_utc"),
                field=f"power sample {index} timestamp_utc",
            )
        except ValueError as exc:
            timestamp_error = timestamp_error or _error_value(
                error_factory, "power_timestamp_invalid", exc
            )
            errors.append(_error_value(error_factory, "power_timestamp_invalid", exc))
            continue
        try:
            _trace_number(sample.get("power_w"), field=f"power sample {index} power_w")
            aiclk = _trace_number(
                sample.get("aiclk_mhz"),
                field=f"power sample {index} aiclk_mhz",
                positive=True,
            )
            _trace_number(
                sample.get("asic_temp_c"), field=f"power sample {index} asic_temp_c"
            )
        except ValueError as exc:
            errors.append(_error_value(error_factory, "power_sample_invalid", exc))
        else:
            usable_rows.append((sample, timestamp, aiclk))

    usable_samples = [sample for sample, _timestamp, _aiclk in usable_rows]
    parsed_timestamps = [timestamp for _sample, timestamp, _aiclk in usable_rows]
    trace["samples"] = usable_samples
    trace["sample_count"] = len(usable_samples)
    trace["valid_row_count"] = len(usable_samples)
    trace["invalid_row_count"] = len(samples) - len(usable_samples)
    trace["nonempty"] = bool(samples) if require_all_rows_valid else bool(usable_samples)
    trace["errors"] = errors
    all_rows_valid = trace["valid_row_count"] == trace["csv_row_count"]
    if not usable_samples:
        trace["error"] = timestamp_error or _error_value(
            error_factory, "power_trace_no_usable_rows"
        )
    elif timestamp_error is not None:
        trace["error"] = timestamp_error
    elif require_all_rows_valid and not all_rows_valid:
        # Resident timing metadata must have the same all-row validity gate as
        # telemetry.parse_power_trace.  Leave all facts at their diagnostic
        # defaults instead of deriving them from only the usable rows.
        pass
    else:
        trace["timestamps_parse"] = True
        trace["first_timestamp"] = parsed_timestamps[0].isoformat()
        trace["last_timestamp"] = parsed_timestamps[-1].isoformat()
        trace["timestamps_ordered"] = all(
            left <= right for left, right in itertools.pairwise(parsed_timestamps)
        )
        trace["_last_timestamp_datetime"] = parsed_timestamps[-1]
        try:
            start = _parse_trace_timestamp(start_text, field="run_start")
            end = _parse_trace_timestamp(end_text, field="run_end")
        except ValueError as exc:
            trace["error"] = _error_value(error_factory, "power_timestamp_invalid", exc)
        else:
            trace["covers_run_start"] = parsed_timestamps[0] <= start <= end
            trace["covers_run_end"] = parsed_timestamps[-1] >= end
            in_run = [
                aiclk
                for _sample, timestamp, aiclk in usable_rows
                if start <= timestamp <= end
            ]
            trace["in_run_valid_row_count"] = len(in_run)
            trace["aiclk_mhz_in_run"] = in_run
            if in_run:
                trace["aiclk_source"] = "run_trace_samples"
                trace["aiclk_mhz"] = max(in_run)
            interval_bound = False
            if definition == ONE_INTERVAL_BOUND:
                try:
                    interval = float(sampler_interval_seconds)
                except (TypeError, ValueError, OverflowError):
                    interval = float("nan")
                if math.isfinite(interval) and interval > 0.0:
                    interval_start = end - datetime.timedelta(seconds=interval)
                    interval_bound = interval_start <= parsed_timestamps[-1] <= end
                trace["sampler_interval_seconds"] = (
                    interval if math.isfinite(interval) and interval > 0.0 else None
                )
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
    "atomic_power_trace_snapshot",
    "canonical_coverage_definition",
    "is_coverage_definition",
    "public_power_trace_snapshot",
    "read_power_trace_snapshot",
    "validate_power_trace_snapshot_path",
]
