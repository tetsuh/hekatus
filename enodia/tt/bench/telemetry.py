"""Board telemetry and environment capture.

Kept in Python rather than embedded in the wrapper script, because the first
run produced a power trace containing nothing but its header: the sampler
had been written inline and its quoting was wrong, and nothing tested it.
Parsing lives here so it can fail in a test instead of in a measurement.

The power trace is not incidental. Whether the board enforces the limit its
firmware reports is an open question (#28), and the answer decides whether a
throughput figure measured on one board is a lower bound for another. Under
sustained load, this trace is the answer.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.strict_json import dumps as strict_json_dumps

SNAPSHOT_COMMAND = ("tt-smi", "-s", "--snapshot_no_tty")
CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c"
DEFAULT_SAMPLER_INTERVAL_SECONDS = 2.0
SAMPLER_MODES = ("off", "default", "explicit")
POWER_TRACE_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")


def _parse_trace_timestamp(value: Any, *, field: str) -> datetime.datetime:
    """Parse one timezone-bearing ISO-8601 trace timestamp."""
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field} must include a timezone")
        return value.astimezone(datetime.UTC)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(datetime.UTC)


def _trace_timestamp_text(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _parse_trace_timestamp(value, field=field).isoformat()


def _trace_number(value: Any, *, field: str, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{field} must be {requirement}")
    return parsed


def _power_trace_base(path: Path, *, run_start: Any, run_end: Any) -> dict[str, Any]:
    errors: list[str] = []
    try:
        start_text = _trace_timestamp_text(run_start, field="run_start") if run_start is not None else None
    except ValueError as exc:
        start_text = None
        errors.append(str(exc))
    try:
        end_text = _trace_timestamp_text(run_end, field="run_end") if run_end is not None else None
    except ValueError as exc:
        end_text = None
        errors.append(str(exc))
    return {
        "file": path.name,
        "columns": list(POWER_TRACE_COLUMNS),
        "sha256": None,
        "readable": False,
        "nonempty": False,
        "csv_row_count": 0,
        "sample_count": 0,
        "valid_row_count": 0,
        "in_run_valid_row_count": 0,
        "timestamps_parse": False,
        "timestamps_ordered": False,
        "first_timestamp": None,
        "last_timestamp": None,
        "run_start": start_text,
        "run_end": end_text,
        "covers_run_start": False,
        "covers_run_end": False,
        "coverage_complete": False,
        "coverage": {
            "nonempty": False,
            "timestamps_parse": False,
            "timestamps_ordered": False,
            "first_at_or_before_run_start": False,
            "last_at_or_after_run_end": False,
            "readable": False,
            "complete": False,
        },
        "aiclk_source": "no_valid_in_run_samples",
        "aiclk_mhz": None,
        "aiclk_mhz_in_run": [],
        "errors": errors,
    }


def parse_power_trace(
    path: Path,
    *,
    run_start: datetime.datetime | str | None = None,
    run_end: datetime.datetime | str | None = None,
) -> dict[str, Any]:
    """Parse a power CSV and validate PR #109 complete run coverage.

    The returned metadata is board-free and deliberately separates physical
    file facts (byte hash and row counts), timestamp facts, run coverage, and
    AICLK provenance.  A usable trace is nonempty, readable, fully parseable,
    ordered, and satisfies ``first_timestamp <= run_start <= run_end <=
    last_timestamp``.  ``aiclk_mhz_in_run`` contains only valid rows whose
    timestamps fall inside the explicit run interval; it never falls back to a
    pre-run environment snapshot.
    """
    if not isinstance(path, Path):
        path = Path(path)
    trace = _power_trace_base(path, run_start=run_start, run_end=run_end)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        trace["errors"] = [
            f"power trace is not readable: {type(exc).__name__}"
        ]
        return trace
    trace["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != POWER_TRACE_COLUMNS:
            raise ValueError(f"power trace has unexpected columns {fieldnames!r}")
        rows = [dict(row) for row in reader]
    except (UnicodeDecodeError, csv.Error, TypeError, ValueError) as exc:
        trace["errors"] = [str(exc)]
        return trace

    trace["readable"] = True
    trace["csv_row_count"] = len(rows)
    trace["sample_count"] = len(rows)
    trace["nonempty"] = bool(rows)
    parsed_rows: list[tuple[datetime.datetime, float]] = []
    errors: list[str] = list(trace.get("errors", []))
    for index, row in enumerate(rows):
        if None in row or any(column not in row for column in POWER_TRACE_COLUMNS):
            errors.append(f"power row {index} does not have exactly the declared columns")
            continue
        try:
            timestamp = _parse_trace_timestamp(
                row["timestamp_utc"], field=f"power row {index} timestamp_utc"
            )
            _trace_number(row["power_w"], field=f"power row {index} power_w")
            aiclk = _trace_number(
                row["aiclk_mhz"], field=f"power row {index} aiclk_mhz", positive=True
            )
            _trace_number(row["asic_temp_c"], field=f"power row {index} asic_temp_c")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            parsed_rows.append((timestamp, aiclk))

    trace["valid_row_count"] = len(parsed_rows)
    trace["timestamps_parse"] = bool(rows) and len(parsed_rows) == len(rows)
    if trace["timestamps_parse"]:
        timestamps = [timestamp for timestamp, _aiclk in parsed_rows]
        trace["first_timestamp"] = timestamps[0].isoformat()
        trace["last_timestamp"] = timestamps[-1].isoformat()
        trace["timestamps_ordered"] = all(
            left <= right for left, right in itertools.pairwise(timestamps)
        )
        try:
            start = _parse_trace_timestamp(trace["run_start"], field="run_start")
            end = _parse_trace_timestamp(trace["run_end"], field="run_end")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if start > end:
                errors.append("run_start must be at or before run_end")
            trace["covers_run_start"] = timestamps[0] <= start <= end
            trace["covers_run_end"] = timestamps[-1] >= end
            in_run = [
                aiclk
                for timestamp, aiclk in parsed_rows
                if start <= timestamp <= end
            ]
            trace["in_run_valid_row_count"] = len(in_run)
            trace["aiclk_mhz_in_run"] = in_run
            if in_run:
                trace["aiclk_source"] = "run_trace_samples"
                trace["aiclk_mhz"] = max(in_run)
            trace["coverage_complete"] = all(
                (
                    trace["readable"],
                    trace["nonempty"],
                    trace["valid_row_count"] == trace["csv_row_count"],
                    trace["timestamps_parse"],
                    trace["timestamps_ordered"],
                    timestamps[0] <= start,
                    start <= end,
                    end <= timestamps[-1],
                )
            )
    if not trace["timestamps_parse"] and rows:
        trace["aiclk_source"] = "no_valid_in_run_samples"
    trace["errors"] = errors
    trace["coverage"] = {
        "nonempty": trace["nonempty"],
        "timestamps_parse": trace["timestamps_parse"],
        "timestamps_ordered": trace["timestamps_ordered"],
        "first_at_or_before_run_start": trace["covers_run_start"],
        "last_at_or_after_run_end": trace["covers_run_end"],
        "readable": trace["readable"],
        "complete": trace["coverage_complete"],
    }
    return trace


def validate_power_trace_coverage(
    path: Path,
    *,
    run_start: datetime.datetime | str | None = None,
    run_end: datetime.datetime | str | None = None,
) -> dict[str, Any]:
    """Board-free alias for :func:`parse_power_trace` used by record builders."""
    return parse_power_trace(path, run_start=run_start, run_end=run_end)


# Keep both names discoverable for callers that describe the seam as parsing
# or validation; they intentionally share one implementation and one catalog.
read_power_trace = parse_power_trace
validate_power_trace = validate_power_trace_coverage


def _device_info(snapshot: str) -> dict | None:
    """The first device, or None for anything this cannot read.

    Every shape the snapshot might arrive in is checked rather than assumed.
    An exception raised here would propagate out of the sampling loop and end
    it, and a dead sampler says nothing at all — which is how the first run
    produced a trace containing only its header.
    """
    try:
        devices = json.loads(snapshot)["device_info"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(devices, list) or not devices:
        return None
    return devices[0] if isinstance(devices[0], dict) else None


def parse_telemetry(snapshot: str) -> dict[str, str] | None:
    """Extract the sampled quantities, or None if the snapshot is unusable."""
    device = _device_info(snapshot)
    if device is None:
        return None
    try:
        telemetry = device["telemetry"]
        return {
            "power_w": str(telemetry["power"]).strip(),
            "aiclk_mhz": str(telemetry["aiclk"]).strip(),
            "asic_temp_c": str(telemetry["asic_temperature"]).strip(),
        }
    except (KeyError, TypeError):
        return None


def telemetry_csv_row(snapshot: str, *, timestamp: str) -> str | None:
    """One CSV row in the order of CSV_HEADER, or None if nothing was read."""
    reading = parse_telemetry(snapshot)
    if reading is None:
        return None
    return f"{timestamp},{reading['power_w']},{reading['aiclk_mhz']},{reading['asic_temp_c']}"


def parse_environment(snapshot: str) -> dict:
    """The board identity that every result has to carry with it."""
    device = _device_info(snapshot)
    if device is None:
        return {"board_snapshot_error": "no device information in the snapshot"}
    environment = {
        key: device[key] for key in ("board_info", "firmwares", "limits") if key in device
    } | {
        "board": device.get("board_info"),
        "firmware": device.get("firmwares"),
        "limits": device.get("limits"),
    }
    reading = parse_telemetry(snapshot)
    if reading is not None:
        try:
            environment["aiclk_mhz_observed"] = [int(float(reading["aiclk_mhz"]))]
        except (TypeError, ValueError):
            pass
    return environment


def sampler_metadata(mode: str, interval: float | None = None) -> dict[str, object]:
    """Validate and describe the wrapper's telemetry-sampler mode."""
    if mode not in SAMPLER_MODES:
        raise ValueError(f"sampler mode must be one of {SAMPLER_MODES}, got {mode!r}")
    if mode == "off":
        if interval is not None:
            raise ValueError("sampler-off mode cannot carry an interval")
        return {
            "mode": "off",
            "interval_seconds": None,
            "power_trace": "absent_by_design",
            "timing_evidence": "diagnostic_only",
        }
    if interval is None:
        interval = DEFAULT_SAMPLER_INTERVAL_SECONDS
    if not (math.isfinite(interval) and interval > 0):
        raise ValueError(f"sampler interval must be positive and finite, got {interval}")
    if mode == "default" and interval != DEFAULT_SAMPLER_INTERVAL_SECONDS:
        raise ValueError("default sampler mode must use the existing 2-second interval")
    return {
        "mode": mode,
        "interval_seconds": interval,
        "power_trace": "required",
        "timing_evidence": "available",
    }


def _run(command: tuple[str, ...] | str) -> str:
    try:
        completed = subprocess.run(
            command,
            shell=isinstance(command, str),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return completed.stdout
    except Exception as exc:  # noqa: BLE001 - a missing tool must not stop a measurement
        return f"<unavailable: {exc}>"


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _git(repo: Path, *args: str) -> tuple[str, bool]:
    """Output of one git query, and whether it answered at all.

    `_run` returns whatever landed on standard output and discards the exit
    status, which is right for the environment block: a tool that is missing
    costs one field. It is wrong for provenance. `git rev-parse` outside a
    repository — an ownership check refusing the bind-mounted tree is the
    realistic case here — exits non-zero with nothing on stdout, and empty
    stdout is exactly what a clean `git status` produces. Read through `_run`,
    a harness nobody could identify would be recorded as an unnamed commit in
    a clean tree, which is the one thing this must never claim.
    """
    try:
        completed = subprocess.run(
            ("git", "-C", str(repo), *args),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - a missing git must not stop a measurement
        return f"git {' '.join(args)} could not run: {type(exc).__name__}: {exc}", False
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        return f"git {' '.join(args)} failed: {detail}", False
    return completed.stdout.strip(), True


def harness_identity() -> dict:
    """Which revision of this tree computed the result.

    The FLOP accounting behind every efficiency figure lives here, so a
    measurement that cannot name the harness that produced it cannot be
    reproduced or compared against a later one. A modified tree is recorded
    as modified rather than silently attributed to its last commit, and a
    tree git could not read at all is recorded as unknown rather than clean.
    A transferred source tree may deliberately omit ``.git``; the wrapper can
    then provide the already-verified source revision explicitly.
    """
    override_commit = os.environ.get("HEKATUS_TT_HARNESS_COMMIT")
    if override_commit:
        override_dirty = os.environ.get("HEKATUS_TT_HARNESS_DIRTY", "")
        if override_dirty not in {"0", "1", "false", "true"}:
            return {
                "harness_commit": None,
                "harness_dirty": None,
                "harness_identity_error": (
                    "HEKATUS_TT_HARNESS_DIRTY must be one of 0, 1, false, true"
                ),
            }
        return {
            "harness_commit": override_commit,
            "harness_dirty": override_dirty in {"1", "true"},
            "harness_identity_source": "HEKATUS_TT_HARNESS_COMMIT",
        }
    repo = Path(__file__).resolve().parents[3]
    commit, commit_ok = _git(repo, "rev-parse", "HEAD")
    # Tracked changes only. The toolchain writes its build output into the
    # tree on every run, so counting untracked files would make the flag true
    # always, which says nothing about the code that computed the result.
    status, status_ok = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if not (commit_ok and status_ok):
        return {
            "harness_commit": None,
            "harness_dirty": None,
            "harness_identity_error": commit if not commit_ok else status,
        }
    return {"harness_commit": commit, "harness_dirty": bool(status)}


def capture_environment(
    image: str,
    image_pinned: bool,
    *,
    sampler_mode: str = "default",
    sampler_interval: float | None = None,
) -> dict:
    """Everything needed to name the environment a measurement came from."""
    environment = {
        "captured_at": _now(),
        "image": image,
        "image_pinned": image_pinned,
        "telemetry_sampler": sampler_metadata(sampler_mode, sampler_interval),
        "kernel": _run("uname -sr").strip(),
        "kmd_version": _run("modinfo tenstorrent 2>/dev/null | awk '/^version:/{print $2}'").strip(),
        "tt_env_active_release": _run(
            "tt-env status 2>/dev/null | awk '/Active release:/{print $3}'"
        ).strip(),
    }
    environment.update(harness_identity())
    environment.update(parse_environment(_run(SNAPSHOT_COMMAND)))
    run_id = os.environ.get("HEKATUS_TT_RUN_ID")
    if run_id:
        # The outer wrapper creates this identity once and exports it to both
        # environment capture and the benchmark process.  Do not invent one
        # here: an unset value is useful to callers that only exercise this
        # parser module.
        environment["run_id"] = run_id
    return environment


def _sample_loop(handle, interval: float) -> None:
    """Sample until terminated, surviving a bad reading but not a bad writer.

    The sampler observes a run it must not disturb, so a reading that fails —
    an unresponsive tool, a snapshot that will not parse — costs one row and
    the loop continues.

    Failing to *record* is a different thing and is not survived. Retrying a
    write that cannot succeed would turn a broken run into a quietly short
    trace, which is precisely the failure this module was extracted to stop
    happening silently.
    """
    handle.write(CSV_HEADER + "\n")
    while True:
        try:
            row = telemetry_csv_row(_run(SNAPSHOT_COMMAND), timestamp=_now())
        except Exception as exc:  # noqa: BLE001 - one lost row beats a lost trace
            print(f"telemetry sample failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            row = None
        if row is not None:
            handle.write(row + "\n")
        time.sleep(interval)


def _sample_forever(out_path: Path, interval: float) -> None:
    with out_path.open("w", buffering=1) as handle:
        _sample_loop(handle, interval)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture board environment or sample telemetry")
    sub = parser.add_subparsers(dest="mode", required=True)

    capture = sub.add_parser("capture-env", help="write the environment as JSON and exit")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--image", required=True)
    capture.add_argument("--image-pinned", action="store_true")
    capture.add_argument("--sampler-mode", choices=SAMPLER_MODES, default="default")
    capture.add_argument("--sampler-interval", type=float, default=None)

    sample = sub.add_parser("sample", help="append telemetry rows until terminated")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--interval", type=float, default=DEFAULT_SAMPLER_INTERVAL_SECONDS)

    args = parser.parse_args(argv)
    if args.mode == "sample" and not (math.isfinite(args.interval) and args.interval > 0):
        # Zero turns the loop into a busy wait on the snapshot tool; negative
        # and non-finite values reach sleep and end the sampler outright.
        parser.error(f"--interval must be positive and finite, got {args.interval}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "capture-env":
        try:
            if args.sampler_mode == "default" and args.sampler_interval is None:
                # Keep the two-argument call compatible with board-free callers
                # that replace capture_environment in tests.
                environment = capture_environment(args.image, args.image_pinned)
            else:
                environment = capture_environment(
                    args.image,
                    args.image_pinned,
                    sampler_mode=args.sampler_mode,
                    sampler_interval=args.sampler_interval,
                )
        except ValueError as exc:
            parser.error(str(exc))
        args.out.write_text(strict_json_dumps(environment, indent=2) + "\n")
        print(f"environment -> {args.out}")
        return

    try:
        _sample_forever(args.out, args.interval)
    except KeyboardInterrupt:
        # The wrapper terminates the sampler when the run finishes; that is
        # the normal way this ends, not a failure.
        pass


if __name__ == "__main__":
    main()
