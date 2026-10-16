"""Issue #121 Q2 one-run harness and read-only telemetry capture.

The module is intentionally importable on a board-free host.  It does not
open a device or run ``tt-smi`` during import.  A real invocation launches the
existing Issue #118 resident runner through the repository wrapper, while the
pre-run, in-run, and post-run snapshots all use one bounded subprocess path.
"""

from __future__ import annotations

import argparse
import codecs
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.board_identity import normalize_board_id, normalize_pci_bus_id
from enodia.tt.bench.power_trace_extended import (
    EXTENDED_POWER_TRACE_COLUMNS,
    EXTENDED_POWER_TRACE_SCHEMA,
    extended_telemetry_csv_row,
    read_extended_power_trace_csv,
    select_extended_board,
)
from enodia.tt.bench.run_binding import validate_run_id
from enodia.tt.bench.telemetry import SNAPSHOT_COMMAND

ISSUE121_Q2_PLAN = "docs/measurements/2026-10-16-host-newton-schulz-issue121-q2-plan.json"
ISSUE121_Q2_RECORD_SCHEMA = "issue121-q2-measurement-v1"
ISSUE121_Q2_SNAPSHOT_SCHEMA = "issue121-q2-tt-smi-snapshot-v1"
ISSUE121_Q2_TRACE_SCHEMA = EXTENDED_POWER_TRACE_SCHEMA
ISSUE121_Q2_TRACE_SCHEMA_VERSION = 1
ISSUE121_Q2_IMAGE_NAME = (
    "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64"
)

# The plan's device_id identifies the host device-node number.  A single
# device bind is visible as device_id 0 inside the resident container.
DEVICE_ID = 1
DEVICE_NODE = "/dev/tenstorrent/1"
CONTAINER_VISIBLE_DEVICE_ID = 0
RUN_DURATION_SECONDS = 285.0
RUN_GUARD_SECONDS = 330.0
WORKLOAD_OUTPUT_TAIL_LINES = 40
WORKLOAD_OUTPUT_TAIL_MAX_CHARS = 8_192
WORKLOAD_STDOUT_LOG_NAME = "workload-stdout.log"
WORKLOAD_STDERR_LOG_NAME = "workload-stderr.log"
WORKLOAD_LOG_CHUNK_BYTES = 64 * 1024
WORKLOAD_LOG_FIELD = "workload log"
WORKLOAD_PATH_OMISSION_REASON = "not_saved: path_detected"
COOLDOWN_WAIT_CAP_SECONDS = 600.0
SNAPSHOT_TIMEOUT_SECONDS = 5.0
TRACE_INTERVAL_SECONDS = 2.0
EARLY_RAW_WINDOW_SECONDS = 10.0
RAW_WINDOW_COLUMNS = (
    "timestamp_monotonic_ns",
    "seconds_from_run_start",
    "aiclk_mhz",
    "power_w",
    "current_a",
    "asic_temp_c",
)

LEGACY_POWER_TRACE_COLUMNS = (
    "timestamp_utc",
    "power_w",
    "aiclk_mhz",
    "asic_temp_c",
)
EXTENDED_TRACE_COLUMNS = tuple(EXTENDED_POWER_TRACE_COLUMNS)
SNAPSHOT_COMMAND_TUPLE = tuple(SNAPSHOT_COMMAND)

COOLDOWN_POWER_TOLERANCE_W = 5.0
COOLDOWN_TEMPERATURE_TOLERANCE_C = 3.0

# The fixed-card-tick observation is the board-free calibration source already
# retained by Issue #118.  It is used only to build a repeat-count plan; it is
# not treated as a new Q2 measurement.
ISSUE118_NS_CARD_TICKS = 742_492_282_048
ISSUE118_NS_REPEAT_COUNT = 333_688
ISSUE118_NS_WEIGHTED_AICLK_MHZ = 1050.9138856647014
ISSUE118_NS_MIN_AICLK_MHZ = 1018.0
ISSUE118_NS_THROUGHPUT = 606.704754341449
Q2_PREDICTION_AICLK_MHZ = ISSUE118_NS_MIN_AICLK_MHZ

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
_FORBIDDEN_KEYS = frozenset(
    {
        "hostname",
        "host_name",
        "username",
        "user_name",
        "worktree_path",
        "repository_path",
        "working_directory",
        "cwd",
    }
)
_FORBIDDEN_PATH_TEXT = re.compile(
    r"(?<![A-Za-z0-9])(?:/home/|/Users/|/root/|/tmp/|~[/\\]|[A-Za-z]:[\\/])"
)


class Issue121Q2Error(ValueError):
    """Raised when a Q2 harness contract cannot be represented safely."""


class SnapshotCaptureError(Issue121Q2Error):
    """Raised by strict callers when a snapshot is required but unusable."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _finite_number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise Issue121Q2Error(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Issue121Q2Error(f"{field} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise Issue121Q2Error(f"{field} must be {requirement}")
    return number


def _safe_run_id(value: str) -> str:
    try:
        return validate_run_id(value)
    except ValueError as exc:
        raise Issue121Q2Error(str(exc)) from exc


def _validate_container_visible_device_id(value: Any | None = None) -> int:
    """Require the sole device bind to remain visible as device 0."""
    if CONTAINER_VISIBLE_DEVICE_ID != 0:
        raise Issue121Q2Error("Q2 container-visible device_id must be configured as 0")
    selected = CONTAINER_VISIBLE_DEVICE_ID if value is None else value
    if isinstance(selected, bool) or not isinstance(selected, int) or selected != 0:
        raise Issue121Q2Error("Q2 container-visible device_id must be 0")
    return selected


def _trusted_output_roots() -> tuple[Path, ...]:
    roots = (
        Path.cwd(),
        Path(tempfile.gettempdir()),
        Path("/out"),
        Path(__file__).resolve().parents[1],
    )
    return tuple(root.resolve(strict=False) for root in roots)


def _safe_output_path(path: Path, *, field: str) -> Path:
    """Resolve a host path only when it remains below a trusted output root."""
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=False)
        if candidate.is_symlink():
            raise Issue121Q2Error(f"{field} must not be a symbolic link")
        trusted_roots = _trusted_output_roots()
    except (OSError, RuntimeError, ValueError) as exc:
        raise Issue121Q2Error(f"{field} cannot be resolved safely") from exc
    if not any(resolved.is_relative_to(root) for root in trusted_roots):
        raise Issue121Q2Error(f"{field} must be beneath a trusted output directory")
    return resolved


def _safe_output_dir(value: Path) -> Path:
    output_dir = Path(value)
    text = str(output_dir)
    if "\n" in text or "\\" in text or ".." in output_dir.parts:
        raise Issue121Q2Error("output_dir must not contain path traversal")
    _safe_output_path(output_dir, field="output_dir")
    return output_dir


def _safe_artifact_path(output_dir: Path, name: str, *, field: str) -> Path:
    safe_name = _safe_basename(name, field=field)
    return _safe_output_path(output_dir / safe_name, field=field)


def _safe_basename(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise Issue121Q2Error(f"{field} must be a basename")
    return value


def _safe_identity(board_id: str, bus_id: str) -> tuple[str, str]:
    normalized_board = normalize_board_id(board_id)
    normalized_bus = normalize_pci_bus_id(bus_id)
    if normalized_board is None:
        raise Issue121Q2Error("board_id must be a non-empty string")
    if normalized_bus is None:
        raise Issue121Q2Error("bus_id must be a valid PCI BDF")
    return normalized_board, normalized_bus


def _safe_digest(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.strip()) is None:
        raise Issue121Q2Error("image_digest must be sha256:<64 lowercase hexadecimal digits>")
    return value.strip().lower()


def _safe_commit(value: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value.strip()) is None:
        raise Issue121Q2Error("harness_commit must be a hexadecimal commit identifier")
    return value.strip().lower()


def _json_safe_snapshot(value: Any) -> Any:
    """Retain device telemetry while dropping host identity and process data."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_KEYS:
                continue
            result[key_text] = _json_safe_snapshot(child)
        return result
    if isinstance(value, list):
        return [_json_safe_snapshot(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe_snapshot(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _device_only_snapshot(root: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only the snapshot fields needed by the extended reader."""
    result: dict[str, Any] = {}
    if isinstance(root.get("host_sw_vers"), Mapping):
        result["host_sw_vers"] = _json_safe_snapshot(root["host_sw_vers"])
    devices = root.get("device_info")
    if not isinstance(devices, list):
        raise SnapshotCaptureError("snapshot device_info must be an array")
    safe_devices: list[Any] = []
    for index, device in enumerate(devices):
        if not isinstance(device, Mapping):
            raise SnapshotCaptureError(f"snapshot device_info[{index}] must be an object")
        safe_device: dict[str, Any] = {}
        for key in ("board_info", "firmwares", "limits", "smbus_telem", "telemetry"):
            if key in device:
                safe_device[key] = _json_safe_snapshot(device[key])
        safe_devices.append(safe_device)
    if not safe_devices:
        raise SnapshotCaptureError("snapshot device_info must not be empty")
    result["device_info"] = safe_devices
    return result


def _stdout_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def _run_snapshot_command(
    command_runner: Callable[..., Any], timeout: float
) -> tuple[str, int | None, str, str | None]:
    stdout = ""
    returncode: int | None = None
    status = "ok"
    error_type: str | None = None
    try:
        completed = command_runner(
            SNAPSHOT_COMMAND_TUPLE,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = _stdout_text(getattr(completed, "stdout", ""))
        raw_returncode = getattr(completed, "returncode", 0)
        returncode = int(raw_returncode) if raw_returncode is not None else None
        if returncode != 0:
            status = "command_failed"
            error_type = "nonzero_returncode"
    except subprocess.TimeoutExpired:
        status = "timeout"
        error_type = "TimeoutExpired"
    except (OSError, subprocess.SubprocessError) as exc:
        status = "command_failed"
        error_type = type(exc).__name__
    except Exception as exc:  # noqa: BLE001 - a mocked command can fail arbitrarily
        status = "command_failed"
        error_type = type(exc).__name__
    return stdout, returncode, status, error_type


def _parse_snapshot_output(
    stdout: str, status: str, error_type: str | None
) -> tuple[dict[str, Any] | None, str, str | None]:
    parsed_snapshot: dict[str, Any] | None = None
    if stdout:
        try:
            decoded = json.loads(stdout)
            if not isinstance(decoded, Mapping):
                raise SnapshotCaptureError("snapshot JSON must be an object")
            parsed_snapshot = _device_only_snapshot(decoded)
        except (json.JSONDecodeError, SnapshotCaptureError, RecursionError, TypeError):
            if status == "ok":
                status = "invalid_json"
            if error_type is None:
                error_type = "invalid_snapshot"
    elif status == "ok":
        status = "empty_output"
        error_type = "empty_output"
    return parsed_snapshot, status, error_type


def capture_readonly_snapshot(
    *,
    stage: str,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    utc_now: Callable[[], str] = _utc_now,
    run_command: Callable[..., Any] | None = None,
    timeout_seconds: float = SNAPSHOT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Capture one bounded read-only snapshot without retaining host fields.

    ``run_command`` is an injection seam for board-free tests.  When omitted,
    the exact ``tt-smi -s --snapshot_no_tty`` command is executed with no shell.
    """
    if not isinstance(stage, str) or not stage.strip():
        raise Issue121Q2Error("snapshot stage must be a non-empty string")
    timeout = _finite_number(timeout_seconds, field="snapshot timeout", positive=True)
    start_ns = monotonic_ns()
    command_runner = subprocess.run if run_command is None else run_command
    stdout, returncode, status, error_type = _run_snapshot_command(command_runner, timeout)
    end_ns = monotonic_ns()
    parsed_snapshot, status, error_type = _parse_snapshot_output(
        stdout, status, error_type
    )

    return {
        "schema": ISSUE121_Q2_SNAPSHOT_SCHEMA,
        "stage": stage,
        "command": list(SNAPSHOT_COMMAND_TUPLE),
        "timeout_seconds": timeout,
        "captured_at_utc": utc_now(),
        "timestamp_monotonic_start_ns": int(start_ns),
        "timestamp_monotonic_ns": int(end_ns),
        "snapshot_elapsed_ns": max(0, int(end_ns) - int(start_ns)),
        "status": status,
        "returncode": returncode,
        "error_type": error_type,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "snapshot": parsed_snapshot,
    }


def _snapshot_device(capture: Mapping[str, Any], board_id: str, bus_id: str) -> Mapping[str, Any]:
    snapshot = capture.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise SnapshotCaptureError("snapshot capture has no structured snapshot")
    try:
        return select_extended_board(snapshot, board_id=board_id, bus_id=bus_id)
    except ValueError as exc:
        raise SnapshotCaptureError(str(exc)) from exc


def _selected_snapshot_identity(
    capture: Mapping[str, Any], *, board_id: str, bus_id: str
) -> dict[str, str]:
    """Return safe serial and PCI identity for the explicitly selected board."""
    normalized_board, normalized_bus = _safe_identity(board_id, bus_id)
    device = _snapshot_device(capture, normalized_board, normalized_bus)
    board_info = device.get("board_info")
    if not isinstance(board_info, Mapping):
        raise SnapshotCaptureError("selected snapshot device has no board_info")
    serial = normalize_board_id(board_info.get("serial")) or normalize_board_id(
        board_info.get("board_id")
    )
    if serial is None:
        raise SnapshotCaptureError("selected snapshot device has no board serial")
    if serial != normalized_board:
        raise SnapshotCaptureError("selected snapshot board serial does not match board_id")
    selected_bus = normalize_pci_bus_id(board_info.get("bus_id"))
    if selected_bus != normalized_bus:
        raise SnapshotCaptureError("selected snapshot device has an unexpected PCI bus")
    return {
        "serial": serial,
        "board_id": normalized_board,
        "bus_id": normalized_bus,
    }


def _snapshot_capture_is_successful(capture: Mapping[str, Any]) -> bool:
    """Require an integer zero return code before using a snapshot."""
    return (
        capture.get("status") == "ok"
        and isinstance(capture.get("returncode"), int)
        and not isinstance(capture.get("returncode"), bool)
        and capture.get("returncode") == 0
    )


def _require_successful_snapshot_capture(
    capture: Mapping[str, Any], *, label: str
) -> None:
    """Reject failed commands before consuming any retained snapshot JSON."""
    if _snapshot_capture_is_successful(capture):
        return
    status = _redact_capture_value(str(capture.get("status", "unknown")))
    returncode = capture.get("returncode")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        returncode = None
    raise SnapshotCaptureError(
        f"{label} snapshot capture did not complete successfully: "
        f"status={status!r}, returncode={returncode!r}"
    )


def _record_snapshot_identity(
    capture: Mapping[str, Any], *, board_id: str, bus_id: str
) -> dict[str, Any]:
    """Record selected snapshot identity without hiding an unavailable read."""
    if not _snapshot_capture_is_successful(capture):
        return {
            "status": "unavailable",
            "reason": (
                f"snapshot capture status is {capture.get('status')!r} "
                f"with returncode {capture.get('returncode')!r}"
            ),
        }
    try:
        return {
            "status": "selected",
            **_selected_snapshot_identity(capture, board_id=board_id, bus_id=bus_id),
        }
    except SnapshotCaptureError as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _snapshot_identity_is_valid(
    identity: Mapping[str, Any], *, board_id: str, bus_id: str
) -> bool:
    """Require a selected identity to match the board targeted by this run."""
    if identity.get("status") != "selected":
        return False
    return (
        normalize_board_id(identity.get("serial")) == board_id
        and normalize_board_id(identity.get("board_id")) == board_id
        and normalize_pci_bus_id(identity.get("bus_id")) == bus_id
    )


_SNAPSHOT_LIMIT_SPECS = {
    "tdp_limit_w": {
        "unit": "W",
        "sources": (
            ("limits", ("tdp_limit_w", "tdp_limit", "TDP_LIMIT_MAX")),
            ("smbus_telem", ("TDP_LIMIT_MAX", "tdp_limit")),
        ),
    },
    "board_power_limit_w": {
        "unit": "W",
        "sources": (
            ("limits", ("board_power_limit_w", "board_power_limit", "BOARD_POWER_LIMIT")),
            ("smbus_telem", ("BOARD_POWER_LIMIT", "board_power_limit")),
        ),
    },
    "tdc_limit": {
        "unit": "A",
        "sources": (
            ("limits", ("tdc_limit", "TDC_LIMIT_MAX")),
            ("smbus_telem", ("TDC_LIMIT_MAX", "tdc_limit")),
        ),
    },
    "therm_trip_l1_limit_c": {
        "unit": "degC",
        "sources": (
            ("limits", ("therm_trip_l1_limit_c", "therm_trip_l1_limit")),
            ("smbus_telem", ("THERM_TRIP_L1_LIMIT", "therm_trip_l1_limit")),
        ),
    },
    "thm_shutdown_limit_c": {
        "unit": "degC",
        "sources": (
            (
                "limits",
                ("thm_shutdown_limit_c", "thm_shutdown_limit", "thm_limit", "THM_LIMIT_SHUTDOWN"),
            ),
            ("smbus_telem", ("THM_LIMIT_SHUTDOWN", "thm_shutdown_limit")),
        ),
    },
    "asic_fmax_mhz": {
        "unit": "MHz",
        "sources": (
            ("limits", ("asic_fmax_mhz", "asic_fmax", "ASIC_FMAX")),
            ("smbus_telem", ("ASIC_FMAX", "asic_fmax")),
        ),
    },
    "thm_limit_throttle": {
        "unit": "degC",
        "sources": (
            ("smbus_telem", ("THM_LIMIT_THROTTLE", "thm_limit_throttle")),
            ("limits", ("THM_LIMIT_THROTTLE", "thm_limit_throttle")),
        ),
    },
}


def _recognized_snapshot_limit(
    name: str, spec: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    for source_name, keys in spec["sources"]:
        source = sources.get(source_name)
        if source is None:
            continue
        for key in keys:
            if key in source:
                return {
                    "value": _json_safe_snapshot(source[key]),
                    "unit": spec["unit"],
                    "source_key": f"{source_name}.{key}",
                    "status": "known",
                }
    return {
        "value": None,
        "unit": None,
        "source_key": (
            "smbus_telem.THM_LIMIT_THROTTLE" if name == "thm_limit_throttle" else None
        ),
        "status": "unknown",
    }


def _snapshot_raw_smbus(source: Mapping[str, Any]) -> dict[str, Any]:
    recognized_names = {
        "board_power_limit",
        "tdc_limit_max",
        "thm_limit_throttle",
        "thm_limit_shutdown",
        "asic_fmax",
    }
    return {
        str(key): _json_safe_snapshot(value)
        for key, value in source.items()
        if str(key).casefold() in recognized_names or "limit" in str(key).casefold()
    }


def _extract_snapshot_limits(
    capture: Mapping[str, Any], *, board_id: str, bus_id: str
) -> dict[str, Any]:
    """Persist raw pre/post limit facts without deriving absent values."""
    try:
        device = _snapshot_device(capture, board_id, bus_id)
    except SnapshotCaptureError as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "recognized": {},
            "raw": {},
            "raw_smbus": {},
        }
    recognized_sources = {
        source_name: source
        for source_name in ("limits", "smbus_telem")
        if isinstance(source := device.get(source_name), Mapping)
    }
    recognized = {
        name: _recognized_snapshot_limit(name, spec, recognized_sources)
        for name, spec in _SNAPSHOT_LIMIT_SPECS.items()
    }
    raw_smbus = _snapshot_raw_smbus(recognized_sources.get("smbus_telem", {}))
    return {
        "status": "available",
        "stage": capture.get("stage"),
        "recognized": recognized,
        # Keep both names discoverable: raw is the record-facing spelling;
        # raw_smbus names the source section explicitly.
        "raw": raw_smbus,
        "raw_smbus": raw_smbus,
        "THM_LIMIT_THROTTLE": raw_smbus.get(
            "THM_LIMIT_THROTTLE", raw_smbus.get("thm_limit_throttle")
        ),
    }


def _raw_window_from_trace(
    trace_path: Path, *, run_start_ns: int, run_end_ns: int
) -> dict[str, Any]:
    """Extract the first ten seconds from the persisted trace rows."""
    trace = read_extended_power_trace_csv(trace_path)
    rows: list[dict[str, Any]] = []
    end_ns = min(
        run_end_ns,
        run_start_ns + round(EARLY_RAW_WINDOW_SECONDS * 1_000_000_000.0),
    )
    for row in trace.get("samples", []):
        timestamp = row.get("timestamp_monotonic_ns")
        if not isinstance(timestamp, int) or not run_start_ns <= timestamp <= end_ns:
            continue
        rows.append(
            {
                "timestamp_monotonic_ns": timestamp,
                "seconds_from_run_start": (timestamp - run_start_ns) / 1_000_000_000.0,
                "aiclk_mhz": float(row["aiclk_mhz"]),
                "power_w": float(row["power_w"]),
                "current_a": float(row["current_a"]),
                "asic_temp_c": float(row["asic_temp_c"]),
            }
        )
    return {
        "start_seconds": 0.0,
        "end_seconds": EARLY_RAW_WINDOW_SECONDS,
        "columns": list(RAW_WINDOW_COLUMNS),
        "rows": rows,
    }


def _redact_capture_value(value: Any) -> Any:
    """Keep capture diagnostics record-safe without retaining host paths."""
    if isinstance(value, Mapping):
        return {str(key): _redact_capture_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_capture_value(child) for child in value]
    if isinstance(value, tuple):
        return [_redact_capture_value(child) for child in value]
    if isinstance(value, str) and _FORBIDDEN_PATH_TEXT.search(value):
        return "<redacted-path>"
    return value


def _safe_capture_details(capture: Any) -> dict[str, Any]:
    """Retain bounded capture facts while excluding host snapshot sections."""
    if not isinstance(capture, Mapping):
        return {
            "stage": "in_run",
            "status": "capture_failed",
            "error_type": "non_mapping_capture",
        }
    safe_capture = _json_safe_snapshot(capture)
    if not isinstance(safe_capture, dict):
        return {
            "stage": "in_run",
            "status": "capture_failed",
            "error_type": "invalid_capture",
        }
    for key in ("host_info", "processes", "cmdline"):
        safe_capture.pop(key, None)
    snapshot = capture.get("snapshot")
    if isinstance(snapshot, Mapping):
        try:
            safe_capture["snapshot"] = _device_only_snapshot(snapshot)
        except SnapshotCaptureError:
            safe_capture["snapshot"] = None
    return _redact_capture_value(safe_capture)


def _in_run_sampler_error(capture: Any, reason: str) -> str:
    """Build a safe diagnostic from capture status without copying output text."""
    if isinstance(capture, Mapping):
        status = capture.get("status", "unknown")
        returncode = capture.get("returncode")
    else:
        status = "capture_failed"
        returncode = None
    safe_status = _redact_capture_value(str(status))
    return (
        f"in-run snapshot unusable: {reason}; status={safe_status!r}; "
        f"returncode={returncode!r}"
    )


def _in_run_capture_failure_reason(capture: Any) -> str:
    if not isinstance(capture, Mapping):
        return "capture result is not an object"
    if not _snapshot_capture_is_successful(capture):
        return "capture command did not complete successfully"
    if not isinstance(capture.get("snapshot"), Mapping):
        return "snapshot is missing or invalid"
    return "telemetry row conversion failed"


def _append_extended_row(
    handle: Any,
    capture: Mapping[str, Any],
    *,
    board_id: str,
    bus_id: str,
) -> bool:
    snapshot = capture.get("snapshot")
    timestamp_ns = capture.get("timestamp_monotonic_ns")
    elapsed_ns = capture.get("snapshot_elapsed_ns")
    if not isinstance(snapshot, Mapping) or not isinstance(timestamp_ns, int):
        return False
    row = extended_telemetry_csv_row(
        snapshot,
        board_id=board_id,
        bus_id=bus_id,
        timestamp_utc=str(capture.get("captured_at_utc")),
        timestamp_monotonic_ns=timestamp_ns,
        snapshot_elapsed_ns=elapsed_ns if isinstance(elapsed_ns, int) else None,
    )
    if row is None:
        return False
    handle.write(row + "\n")
    handle.flush()
    return True


def _sample_one_extended_capture(
    handle: Any,
    *,
    board_id: str,
    bus_id: str,
    captures: list[dict[str, Any]],
    errors: list[str],
    capture_fn: Callable[..., dict[str, Any]],
) -> bool:
    """Capture, validate, and write one in-run sample."""
    capture: Any = None
    try:
        capture = capture_fn(stage="in_run")
        captures.append(_safe_capture_details(capture))
        if not isinstance(capture, Mapping) or not _snapshot_capture_is_successful(capture):
            errors.append(
                _in_run_sampler_error(capture, _in_run_capture_failure_reason(capture))
            )
            return False
        row_failure_reported = False
        try:
            row_written = _append_extended_row(
                handle, capture, board_id=board_id, bus_id=bus_id
            )
        except Exception as exc:  # noqa: BLE001 - preserve a bounded error type
            row_failure_reported = True
            row_written = False
            errors.append(
                _in_run_sampler_error(
                    capture, f"row write failed ({type(exc).__name__})"
                )
            )
        if not row_written and not row_failure_reported:
            errors.append(
                _in_run_sampler_error(capture, _in_run_capture_failure_reason(capture))
            )
        return row_written
    except Exception as exc:  # noqa: BLE001 - preserve an incomplete trace fact
        captures.append(
            {
                "stage": "in_run",
                "status": "capture_failed",
                "error_type": type(exc).__name__,
            }
        )
        errors.append(f"in-run snapshot capture failed: {type(exc).__name__}")
        return False


def _sample_extended_trace(
    stop_event: threading.Event,
    handle: Any,
    *,
    board_id: str,
    bus_id: str,
    interval_seconds: float,
    captures: list[dict[str, Any]],
    errors: list[str],
    capture_fn: Callable[..., dict[str, Any]],
    usable_rows: list[bool] | None = None,
) -> None:
    """Sample until the workload ends; the stop wait avoids an extra row."""
    interval = _finite_number(interval_seconds, field="trace interval", positive=True)
    while not stop_event.wait(interval):
        row_written = _sample_one_extended_capture(
            handle,
            board_id=board_id,
            bus_id=bus_id,
            captures=captures,
            errors=errors,
            capture_fn=capture_fn,
        )
        if usable_rows is not None:
            usable_rows.append(row_written)


def build_workload_command(
    *,
    output_dir: Path,
    run_id: str,
    result_name: str | None = None,
    calibration_name: str = "issue121-q2-calibration.json",
    container_device_id: int | None = None,
) -> tuple[str, ...]:
    """Build the workload command for the host directory mounted as ``/out``."""
    visible_device_id = _validate_container_visible_device_id(container_device_id)
    output_dir = _safe_output_dir(output_dir)
    safe_run_id = _safe_run_id(run_id)
    if result_name is None:
        result_name = f"runner-result-{safe_run_id}.json"
    result_name = _safe_basename(result_name, field="result_name")
    calibration_name = _safe_basename(calibration_name, field="calibration_name")
    return (
        "enodia/tt/bench/run_in_container.sh",
        str(output_dir),
        "--",
        "--out",
        f"/out/{result_name}",
        "--device-id",
        str(visible_device_id),
        "--condition",
        "newton_schulz",
        "--calibration-json",
        f"/out/{calibration_name}",
        "--outer-timeout-seconds",
        str(int(RUN_GUARD_SECONDS)),
    )


def build_workload_environment(
    *,
    run_id: str,
    image_digest: str,
    device_node: str = DEVICE_NODE,
    container_device_id: int | None = None,
) -> dict[str, str]:
    """Return wrapper environment values without retaining a host path."""
    visible_device_id = _validate_container_visible_device_id(container_device_id)
    safe_run_id = _safe_run_id(run_id)
    digest = _safe_digest(image_digest)
    if device_node != DEVICE_NODE:
        raise Issue121Q2Error(f"device_node must be {DEVICE_NODE}")
    return {
        "HEKATUS_TT_RUN_ID": safe_run_id,
        "HEKATUS_TT_RUNNER": "enodia/tt/bench/run_resident.py",
        "HEKATUS_TT_IMAGE": f"{ISSUE121_Q2_IMAGE_NAME}@{digest}",
        "HEKATUS_TT_DEVICE_NODE": device_node,
        "HEKATUS_TT_CONTAINER_DEVICE_ID": str(visible_device_id),
        # The existing resident runner requires sampler-on provenance.  The
        # Q2 thread additionally captures the opt-in identity-bound extended
        # trace; the legacy sampler remains unchanged and is not analyzed.
        "HEKATUS_TT_TELEMETRY_MODE": "default",
        "HEKATUS_TT_CONTAINER_TIMEOUT_S": str(int(RUN_GUARD_SECONDS)),
    }


def build_calibration_payload(
    *,
    target_seconds: float = RUN_DURATION_SECONDS,
    predicted_aiclk_mhz: float = Q2_PREDICTION_AICLK_MHZ,
) -> dict[str, Any]:
    """Build a board-free fixed-repeat calibration contract for the run."""
    target = _finite_number(target_seconds, field="target_seconds", positive=True)
    if target > RUN_DURATION_SECONDS:
        raise Issue121Q2Error("target_seconds must not exceed the Q2 run duration")
    predicted_clock = _finite_number(
        predicted_aiclk_mhz, field="predicted_aiclk_mhz", positive=True
    )
    ticks_per_repeat = ISSUE118_NS_CARD_TICKS / ISSUE118_NS_REPEAT_COUNT
    repeat_count = max(
        1,
        math.ceil(target * predicted_clock * 1_000_000.0 / ticks_per_repeat),
    )
    load_card_ticks = round(ticks_per_repeat * repeat_count)
    weighted_clock = ISSUE118_NS_WEIGHTED_AICLK_MHZ
    minimum_clock = min(predicted_clock, ISSUE118_NS_MIN_AICLK_MHZ)
    calibration = {
        "condition": "newton_schulz",
        "duration_seconds": 60.0,
        "achieved_throughput": ISSUE118_NS_THROUGHPUT,
        "throughput_unit": "repeats_per_card_second",
        "repeat_count": repeat_count,
        "watcher_enabled": False,
        "source": "watcher_off_calibration",
        "target_seconds": target,
        "safety_factor": 1.0,
        "throughput_reference_aiclk_mhz": 1350,
        "throughput_reference_aiclk_source": "fixed_reference_1350_mhz",
        "throughput_unit_definition": (
            "load repeats per 1.35e9 card-clock ticks (historical unit name "
            "repeats_per_card_second); convert those ticks to real seconds with the "
            "observed load-interval AICLK"
        ),
        "load_card_ticks": load_card_ticks,
        "load_measurement_repeat_count": repeat_count,
        "load_interval_aiclk_mhz": weighted_clock,
        "load_interval_aiclk_min_mhz": minimum_clock,
        "load_interval_aiclk_source": "load_interval_trace_samples",
        "load_interval_aiclk_selection": "time_weighted_load_interval",
        "predicted_load_time_aiclk_mhz": minimum_clock,
        "predicted_load_time_aiclk_source": "in_run_minimum",
        "predicted_load_time_aiclk_selection": "in_run_minimum",
        "actual_duration_aiclk_mhz": weighted_clock,
        "actual_duration_aiclk_source": "load_interval_trace_samples",
        "actual_duration_aiclk_selection": "time_weighted_load_interval",
    }
    return {
        "schema": "issue121-q2-calibration-v1",
        "source": "Issue #118 board-free calibration reference",
        "workload": "adr-0013-108-core-newton-schulz-surrogate",
        "calibration": calibration,
    }


def evaluate_cooldown(
    observed_power_w: float,
    observed_temperature_c: float,
    *,
    baseline_power_w: float,
    baseline_temperature_c: float,
    observed_idle_aiclk_mhz: float | None = None,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
) -> dict[str, Any]:
    """Apply the cooldown gate using an explicit preceding-run baseline."""
    power = _finite_number(observed_power_w, field="observed_power_w")
    temperature = _finite_number(observed_temperature_c, field="observed_temperature_c")
    baseline_power = _finite_number(baseline_power_w, field="baseline_power_w")
    baseline_temperature = _finite_number(
        baseline_temperature_c, field="baseline_temperature_c"
    )
    power_tolerance = _finite_number(power_tolerance_w, field="power_tolerance_w", positive=True)
    temperature_tolerance = _finite_number(
        temperature_tolerance_c, field="temperature_tolerance_c", positive=True
    )
    power_limit = baseline_power + power_tolerance
    temperature_limit = baseline_temperature + temperature_tolerance
    power_pass = power <= power_limit
    temperature_pass = temperature <= temperature_limit
    return {
        "baseline_source": "explicit preceding selected-device pre-run snapshot",
        "baseline_power_w": baseline_power,
        "baseline_temperature_c": baseline_temperature,
        "power_tolerance_w": power_tolerance,
        "temperature_tolerance_c": temperature_tolerance,
        "power_limit_w": power_limit,
        "temperature_limit_c": temperature_limit,
        "observed_power_w": power,
        "observed_temperature_c": temperature,
        "observed_idle_aiclk_mhz": (
            None
            if observed_idle_aiclk_mhz is None
            else _finite_number(observed_idle_aiclk_mhz, field="observed_idle_aiclk_mhz", positive=True)
        ),
        "aiclk_gate": False,
        "power_gate": power_pass,
        "temperature_gate": temperature_pass,
        "gate": "pass" if power_pass and temperature_pass else "fail",
        "gate_result": "pass" if power_pass and temperature_pass else "fail",
        "verified": power_pass and temperature_pass,
    }


def _cooldown_reading(capture: Mapping[str, Any], board_id: str, bus_id: str) -> tuple[float, float, float | None]:
    device = _snapshot_device(capture, board_id, bus_id)
    telemetry = device.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise SnapshotCaptureError("snapshot selected device has no telemetry")
    try:
        # The approved cooldown observable is selected-device ASIC power.  Keep
        # board_power in the snapshot and extended trace for reporting, but do
        # not substitute it for telemetry.power in this gate.
        power = _finite_number(telemetry["power"], field="telemetry.power")
        temperature = _finite_number(
            telemetry["asic_temperature"], field="telemetry.asic_temperature"
        )
    except KeyError as exc:
        raise SnapshotCaptureError(f"snapshot telemetry is missing {exc.args[0]}") from exc
    aiclk = telemetry.get("aiclk")
    return power, temperature, None if aiclk is None else _finite_number(aiclk, field="telemetry.aiclk", positive=True)


def evaluate_cooldown_transition(
    preceding_pre_snapshot: Mapping[str, Any],
    cooldown_snapshot: Mapping[str, Any],
    *,
    board_id: str,
    bus_id: str,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
) -> dict[str, Any]:
    """Evaluate one transition from its selected-device pre-run baseline."""
    _require_successful_snapshot_capture(
        preceding_pre_snapshot, label="preceding cooldown baseline"
    )
    if preceding_pre_snapshot.get("stage") != "pre_run":
        raise SnapshotCaptureError("preceding cooldown baseline must be a pre_run snapshot")
    baseline_power, baseline_temperature, _baseline_aiclk = _cooldown_reading(
        preceding_pre_snapshot, board_id, bus_id
    )
    _require_successful_snapshot_capture(cooldown_snapshot, label="cooldown")
    observed_power, observed_temperature, observed_aiclk = _cooldown_reading(
        cooldown_snapshot, board_id, bus_id
    )
    return evaluate_cooldown(
        observed_power,
        observed_temperature,
        baseline_power_w=baseline_power,
        baseline_temperature_c=baseline_temperature,
        observed_idle_aiclk_mhz=observed_aiclk,
        power_tolerance_w=power_tolerance_w,
        temperature_tolerance_c=temperature_tolerance_c,
    )


def _build_environment(
    pre_snapshot: Mapping[str, Any],
    *,
    board_id: str,
    bus_id: str,
    image_digest: str,
    harness_commit: str,
    kernel: str,
    kernel_driver_version: str,
    snapshot_name: str,
    container_device_id: int | None = None,
) -> dict[str, Any]:
    visible_device_id = _validate_container_visible_device_id(container_device_id)
    normalized_board, normalized_bus = _safe_identity(board_id, bus_id)
    device = _snapshot_device(pre_snapshot, normalized_board, normalized_bus)
    selected_identity = _selected_snapshot_identity(
        pre_snapshot, board_id=normalized_board, bus_id=normalized_bus
    )
    board_info = device.get("board_info")
    if not isinstance(board_info, Mapping):
        raise SnapshotCaptureError("selected snapshot device has no board_info")
    firmwares = device.get("firmwares")
    if not isinstance(firmwares, Mapping):
        raise SnapshotCaptureError("selected snapshot device has no firmware metadata")
    firmware_bundle = firmwares.get("fw_bundle_version")
    if not isinstance(firmware_bundle, (str, int, float)) or not str(firmware_bundle).strip():
        raise SnapshotCaptureError("selected snapshot device has no firmware bundle")
    if not isinstance(kernel, str) or not kernel.strip():
        raise Issue121Q2Error("kernel must be a non-empty string")
    if not isinstance(kernel_driver_version, str) or not kernel_driver_version.strip():
        raise Issue121Q2Error("kernel_driver_version must be a non-empty string")
    safe_snapshot_name = Path(snapshot_name).name
    if safe_snapshot_name != snapshot_name:
        raise Issue121Q2Error("snapshot_name must be a basename")
    return {
        "board": {
            "serial": selected_identity["serial"],
            "board_type": board_info.get("board_type"),
            "identity": {
                "board_id": selected_identity["board_id"],
                "bus_id": selected_identity["bus_id"],
            },
        },
        "firmware": {"bundle": str(firmware_bundle)},
        "kernel": kernel.strip(),
        "kernel_driver_version": kernel_driver_version.strip(),
        "image_digest": _safe_digest(image_digest),
        "harness_commit": _safe_commit(harness_commit),
        "device_id": DEVICE_ID,
        "device_node": DEVICE_NODE,
        "container_device_id": visible_device_id,
        "pre_snapshot": safe_snapshot_name,
    }


_WORKLOAD_LINE_ENDINGS = frozenset(
    {"\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"}
)


class _WorkloadTailCollector:
    """Collect the existing bounded text-tail contract without buffering a log."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._lines: deque[str] = deque(maxlen=WORKLOAD_OUTPUT_TAIL_LINES)
        self._pending = ""

    def feed(self, payload: bytes) -> None:
        self._consume(self._decoder.decode(payload, final=False))

    def finish(self) -> str:
        self._consume(self._decoder.decode(b"", final=True))
        if self._pending:
            self._lines.append(self._pending[-WORKLOAD_OUTPUT_TAIL_MAX_CHARS:])
            self._pending = ""
        tail = "".join(self._lines)
        return tail[-WORKLOAD_OUTPUT_TAIL_MAX_CHARS:]

    def _consume(self, text: str) -> None:
        if not text:
            return
        self._pending += text
        parts = self._pending.splitlines(keepends=True)
        if parts and parts[-1][-1:] not in _WORKLOAD_LINE_ENDINGS:
            self._pending = parts.pop()
        else:
            self._pending = ""
        for line in parts:
            self._lines.append(line[-WORKLOAD_OUTPUT_TAIL_MAX_CHARS:])
        if len(self._pending) > WORKLOAD_OUTPUT_TAIL_MAX_CHARS:
            self._pending = self._pending[-WORKLOAD_OUTPUT_TAIL_MAX_CHARS:]


class _WorkloadPathDetector:
    """Find host-specific path text while reading a raw log in chunks."""

    def __init__(self) -> None:
        self.detected = False
        self._context = ""

    def feed(self, payload: bytes) -> None:
        if self.detected:
            return
        text = payload.decode("utf-8", errors="replace")
        candidate = self._context + text
        if _FORBIDDEN_PATH_TEXT.search(candidate):
            self.detected = True
            return
        self._context = candidate[-128:]


def _bounded_workload_output_tail(value: Any) -> tuple[str | None, str | None]:
    """Return a bounded output tail unless it contains a host-specific path."""
    text = _stdout_text(value)
    if _FORBIDDEN_PATH_TEXT.search(text):
        return None, WORKLOAD_PATH_OMISSION_REASON
    lines = text.splitlines(keepends=True)
    tail = "".join(lines[-WORKLOAD_OUTPUT_TAIL_LINES:])
    if len(tail) > WORKLOAD_OUTPUT_TAIL_MAX_CHARS:
        tail = tail[-WORKLOAD_OUTPUT_TAIL_MAX_CHARS:]
    return tail, None


def _workload_stream_field(
    result: Any, stream: str, omissions: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    if isinstance(result, Mapping):
        if stream in result:
            value = result[stream]
        elif f"{stream}_tail" in result:
            value = result[f"{stream}_tail"]
        elif stream in omissions and omissions[stream] == WORKLOAD_PATH_OMISSION_REASON:
            return None, omissions[stream]
        else:
            value = ""
    else:
        value = getattr(result, stream, "")
    return _bounded_workload_output_tail(value)


def _workload_output_fields(result: Any) -> dict[str, Any]:
    """Retain only safe bounded tails from one workload process result."""
    existing_omissions = result.get("output_omissions") if isinstance(result, Mapping) else None
    omissions = existing_omissions if isinstance(existing_omissions, Mapping) else {}
    fields: dict[str, Any] = {}
    omitted: dict[str, str] = {}
    for stream in ("stdout", "stderr"):
        tail, reason = _workload_stream_field(result, stream, omissions)
        if reason is not None:
            omitted[stream] = reason
        else:
            fields[f"{stream}_tail"] = tail
    if omitted:
        fields["output_omissions"] = omitted
    return fields


def _normalize_workload_result(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        normalized = {
            "returncode": result.get("returncode"),
            "status": str(result.get("status", "completed")),
            "timed_out": bool(result.get("timed_out", False)),
        }
    else:
        normalized = {
            "returncode": getattr(result, "returncode", result if isinstance(result, int) else None),
            "status": "completed",
            "timed_out": False,
        }
    normalized.update(_workload_output_fields(result))
    return normalized


def _workload_stream_bytes(value: Any) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    if value is None:
        return b""
    return None


def _stream_value(result: Any, stream: str) -> tuple[Any, bool]:
    if isinstance(result, Mapping):
        if stream in result:
            return result[stream], True
        return None, False
    value = getattr(result, stream, None)
    return value, hasattr(result, stream)


def _write_workload_log(path: Path, payload: bytes) -> str:
    """Write bytes in bounded chunks; return a record-safe write status."""
    try:
        safe_path = _safe_output_path(path, field=WORKLOAD_LOG_FIELD)
        with safe_path.open("wb") as handle:
            for offset in range(0, len(payload), WORKLOAD_LOG_CHUNK_BYTES):
                handle.write(payload[offset : offset + WORKLOAD_LOG_CHUNK_BYTES])
    except (OSError, TypeError, ValueError):
        return "write_failed"
    return "saved"


def _inspect_workload_log(
    path: Path, *, write_status: str
) -> tuple[dict[str, Any], str | None, str | None]:
    """Hash and tail a local log without putting its contents in the record."""
    metadata = {
        "path": path.name,
        "byte_count": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "status": write_status,
    }
    if write_status not in {"saved", "write_failed"}:
        return metadata, "", None

    digest = hashlib.sha256()
    byte_count = 0
    tail_collector = _WorkloadTailCollector()
    path_detector = _WorkloadPathDetector()
    read_ok = False
    try:
        safe_path = _safe_output_path(path, field=WORKLOAD_LOG_FIELD)
        with safe_path.open("rb") as handle:
            while True:
                chunk = handle.read(WORKLOAD_LOG_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
                tail_collector.feed(chunk)
                path_detector.feed(chunk)
        read_ok = True
    except (OSError, TypeError, ValueError):
        metadata["status"] = "read_failed" if write_status == "saved" else write_status

    metadata["byte_count"] = byte_count
    metadata["sha256"] = digest.hexdigest()
    if not read_ok:
        return metadata, "", None
    if path_detector.detected:
        return metadata, None, WORKLOAD_PATH_OMISSION_REASON
    return metadata, tail_collector.finish(), None


def _workload_fields_from_logs(
    paths: Mapping[str, Path], statuses: Mapping[str, str], *, existing_omissions: Any = None
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    fields: dict[str, Any] = {"local_diagnostics": diagnostics}
    omissions = existing_omissions if isinstance(existing_omissions, Mapping) else {}
    omitted: dict[str, str] = {}
    for stream in ("stdout", "stderr"):
        metadata, tail, reason = _inspect_workload_log(
            paths[stream], write_status=statuses[stream]
        )
        diagnostics[stream] = metadata
        if reason is not None or (
            tail == ""
            and stream in omissions
            and omissions[stream] == WORKLOAD_PATH_OMISSION_REASON
        ):
            omitted[stream] = reason or omissions[stream]
        else:
            fields[f"{stream}_tail"] = tail
    if omitted:
        fields["output_omissions"] = omitted
    return fields


def _persist_workload_streams(output_dir: Path, result: Any) -> dict[str, Any]:
    """Persist callback-provided streams and return record-safe diagnostics."""
    paths = {
        "stdout": _safe_artifact_path(
            output_dir, WORKLOAD_STDOUT_LOG_NAME, field="workload stdout log"
        ),
        "stderr": _safe_artifact_path(
            output_dir, WORKLOAD_STDERR_LOG_NAME, field="workload stderr log"
        ),
    }
    statuses: dict[str, str] = {}
    for stream, path in paths.items():
        value, present = _stream_value(result, stream)
        payload = _workload_stream_bytes(value) if present else b""
        if payload is None:
            statuses[stream] = "unavailable"
            continue
        try:
            statuses[stream] = _write_workload_log(path, payload)
        except (OSError, TypeError, ValueError):
            statuses[stream] = "write_failed"
    existing_omissions = result.get("output_omissions") if isinstance(result, Mapping) else None
    return _workload_fields_from_logs(paths, statuses, existing_omissions=existing_omissions)


def _run_workload_without_logs(
    command: Sequence[str], environment: Mapping[str, str], *, repo_root: Path
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=repo_root,
            env={**os.environ, **dict(environment)},
            capture_output=True,
            text=False,
            timeout=RUN_GUARD_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _normalize_workload_result(
            {
                "returncode": None,
                "status": "timeout",
                "timed_out": True,
                "stdout": getattr(exc, "stdout", ""),
                "stderr": getattr(exc, "stderr", ""),
            }
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "returncode": None,
            "status": type(exc).__name__,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    return _normalize_workload_result(completed)


def _open_workload_logs(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, str]]:
    handles: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for stream, path in paths.items():
        try:
            safe_path = _safe_output_path(path, field=WORKLOAD_LOG_FIELD)
            handles[stream] = safe_path.open("wb")
            statuses[stream] = "saved"
        except (OSError, TypeError, ValueError):
            statuses[stream] = "write_failed"
    return handles, statuses


def _run_workload_with_logs(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    repo_root: Path,
    handles: Mapping[str, Any],
    statuses: dict[str, str],
) -> tuple[dict[str, Any], Any]:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=repo_root,
            env={**os.environ, **dict(environment)},
            stdout=handles.get("stdout", subprocess.DEVNULL),
            stderr=handles.get("stderr", subprocess.DEVNULL),
            text=False,
            timeout=RUN_GUARD_SECONDS,
            check=False,
        )
        process_result = {
            "returncode": getattr(completed, "returncode", None),
            "status": "completed",
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        process_result = {
            "returncode": None,
            "status": "timeout",
            "timed_out": True,
        }
        completed = exc
    except (OSError, subprocess.SubprocessError) as exc:
        process_result = {
            "returncode": None,
            "status": type(exc).__name__,
            "timed_out": False,
        }
        completed = exc
    finally:
        _close_workload_logs(handles, statuses)
    return process_result, completed


def _close_workload_logs(handles: Mapping[str, Any], statuses: dict[str, str]) -> None:
    for stream, handle in handles.items():
        try:
            handle.close()
        except (OSError, ValueError):
            statuses[stream] = "write_failed"


def _persist_subprocess_output(
    paths: Mapping[str, Path],
    statuses: dict[str, str],
    completed: Any,
) -> None:
    """Use returned output only when a test seam left a saved log empty."""
    for stream, path in paths.items():
        if statuses[stream] != "saved":
            continue
        value = getattr(completed, stream, None)
        payload = _workload_stream_bytes(value)
        if not payload:
            continue
        try:
            is_empty = _safe_output_path(path, field=WORKLOAD_LOG_FIELD).stat().st_size == 0
        except OSError:
            is_empty = False
        if not is_empty:
            continue
        try:
            statuses[stream] = _write_workload_log(path, payload)
        except (OSError, TypeError, ValueError):
            statuses[stream] = "write_failed"


def _run_workload(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if output_dir is None:
        return _run_workload_without_logs(command, environment, repo_root=repo_root)

    safe_output_dir = _safe_output_path(output_dir, field="workload output directory")
    paths = {
        "stdout": _safe_artifact_path(
            safe_output_dir, WORKLOAD_STDOUT_LOG_NAME, field="workload stdout log"
        ),
        "stderr": _safe_artifact_path(
            safe_output_dir, WORKLOAD_STDERR_LOG_NAME, field="workload stderr log"
        ),
    }
    handles, statuses = _open_workload_logs(paths)
    process_result, completed = _run_workload_with_logs(
        command,
        environment,
        repo_root=repo_root,
        handles=handles,
        statuses=statuses,
    )
    _persist_subprocess_output(paths, statuses, completed)
    process_result.update(_workload_fields_from_logs(paths, statuses))
    return process_result


def _safe_record_value(value: Any, *, field: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_KEYS:
                raise Issue121Q2Error(f"persisted record contains forbidden field {field}.{key_text}")
            _safe_record_value(child, field=f"{field}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _safe_record_value(child, field=f"{field}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_PATH_TEXT.search(value):
        raise Issue121Q2Error(f"persisted record contains a host-specific path in {field}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    safe_path = _safe_output_path(path, field="JSON output")
    safe_path.write_text(strict_json_dumps(value, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _RunOptions:
    run_id: str
    run_number: int | None
    board_id: str
    bus_id: str
    image_digest: str
    harness_commit: str
    kernel: str
    kernel_driver_version: str
    container_device_id: int | None
    trace_interval_seconds: float
    workload_runner: Callable[[Sequence[str], Mapping[str, str], float], Any] | None
    capture_fn: Callable[..., dict[str, Any]] | None
    monotonic_ns: Callable[[], int]
    utc_now: Callable[[], str]


_RUN_OPTION_DEFAULTS = {
    "run_number": None,
    "container_device_id": None,
    "trace_interval_seconds": TRACE_INTERVAL_SECONDS,
    "workload_runner": None,
    "capture_fn": None,
    "monotonic_ns": time.monotonic_ns,
    "utc_now": _utc_now,
}
_RUN_OPTION_REQUIRED = (
    "run_id",
    "board_id",
    "bus_id",
    "image_digest",
    "harness_commit",
    "kernel",
    "kernel_driver_version",
)
_RUN_OPTION_NAMES = frozenset((*_RUN_OPTION_REQUIRED, *_RUN_OPTION_DEFAULTS))


def _run_options(values: Mapping[str, Any]) -> _RunOptions:
    unknown = set(values).difference(_RUN_OPTION_NAMES)
    if unknown:
        name = min(unknown)
        raise TypeError(f"run_one() got an unexpected keyword argument {name!r}")
    missing = [name for name in _RUN_OPTION_REQUIRED if name not in values]
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise TypeError(f"run_one() missing required keyword-only arguments: {names}")
    return _RunOptions(
        **{name: values.get(name, default) for name, default in _RUN_OPTION_DEFAULTS.items()},
        **{name: values[name] for name in _RUN_OPTION_REQUIRED},
    )


def _write_boundary_extended_row(
    handle: Any,
    capture: Mapping[str, Any],
    *,
    board_id: str,
    bus_id: str,
    stage: str,
    errors: list[str],
) -> bool:
    try:
        row_written = _append_extended_row(
            handle,
            capture,
            board_id=board_id,
            bus_id=bus_id,
        )
    except Exception as exc:  # noqa: BLE001 - preserve a bounded error type
        row_written = False
        errors.append(f"{stage} telemetry row write failed: {type(exc).__name__}")
    if not row_written:
        errors.append(f"{stage} telemetry row was not written")
    return row_written


def _q2_run_is_ok(
    workload: Mapping[str, Any],
    *,
    guard_pass: bool,
    sampler_errors: Sequence[str],
    pre_capture_ok: bool,
    pre_identity_valid: bool,
    pre_row: bool,
    post_capture_ok: bool,
    post_identity_valid: bool,
    post_row: bool,
    in_run_row_results: Sequence[bool],
) -> bool:
    required_evidence = (
        not sampler_errors,
        pre_capture_ok,
        pre_identity_valid,
        pre_row,
        post_capture_ok,
        post_identity_valid,
        post_row,
        any(in_run_row_results),
    )
    return workload.get("returncode") == 0 and guard_pass and all(required_evidence)


@dataclass
class _PreflightResult:
    pre_capture: dict[str, Any]
    pre_path: Path
    trace_path: Path
    safe_trace_path: Path
    pre_identity: dict[str, Any]
    capture_successful: bool
    identity_valid: bool
    pre_row: bool
    trace_created: bool
    reason: str | None
    interval: float
    sampler_errors: list[str]


def _failed_preflight_record(
    *,
    safe_output_dir: Path,
    safe_run_id: str,
    run_number: int | None,
    normalized_board: str,
    normalized_bus: str,
    preflight: _PreflightResult,
) -> dict[str, Any]:
    """Persist a safe failed attempt without starting workload setup."""
    post_reason = "post-run snapshot not captured after pre-run preflight failure"
    post_identity = {"status": "unavailable", "reason": post_reason}
    trace_bytes = b""
    trace_sample_count = 0
    trace_sha256: str | None = None
    trace_path_value: str | None = None
    if preflight.trace_created:
        trace_path_value = preflight.trace_path.name
        try:
            trace_bytes = preflight.trace_path.read_bytes()
            trace_sample_count = max(
                0,
                sum(1 for line in trace_bytes.decode("utf-8").splitlines()[1:] if line),
            )
            trace_sha256 = hashlib.sha256(trace_bytes).hexdigest()
        except (OSError, UnicodeDecodeError):
            preflight.sampler_errors.append("pre-run trace could not be read")
    record_name = f"run-{safe_run_id}.json"
    record_path = _safe_artifact_path(
        safe_output_dir, record_name, field="measurement record"
    )
    output_paths: dict[str, Any] = {
        "record": record_name,
        "pre_snapshot": preflight.pre_path.name,
    }
    if trace_path_value is not None:
        output_paths["extended_trace"] = trace_path_value
    record: dict[str, Any] = {
        "record_schema": ISSUE121_Q2_RECORD_SCHEMA,
        "issue": "#121",
        "run_id": safe_run_id,
        "run_number": run_number,
        "status": "failed",
        "environment": None,
        "preflight": {
            "status": "failed",
            "capture_successful": preflight.capture_successful,
            "identity_valid": preflight.identity_valid,
            "pre_row_written": preflight.pre_row,
            "reason": _redact_capture_value(preflight.reason),
        },
        "workload": {
            "name": "adr-0013-108-core-newton-schulz-surrogate",
            "condition": "newton_schulz",
            "device_id": DEVICE_ID,
            "container_device_id": CONTAINER_VISIBLE_DEVICE_ID,
            "device_node": DEVICE_NODE,
            "worker_core_count": 108,
            "data_independent": True,
            "run_duration_target_seconds": RUN_DURATION_SECONDS,
            "run_guard_seconds": RUN_GUARD_SECONDS,
            "started": False,
        },
        "run_timing": {
            "workload_start_monotonic_ns": None,
            "workload_end_monotonic_ns": None,
            "workload_duration_seconds": None,
            "target_duration_seconds": RUN_DURATION_SECONDS,
            "guard_seconds": RUN_GUARD_SECONDS,
            "within_guard": False,
            "timeout_seconds": RUN_GUARD_SECONDS,
        },
        "workload_result": {
            "status": "not_started",
            "returncode": None,
            "timed_out": False,
        },
        "snapshot_limits": {
            "pre": _extract_snapshot_limits(
                preflight.pre_capture, board_id=normalized_board, bus_id=normalized_bus
            ),
            "post": {"status": "unavailable", "reason": post_reason},
        },
        "snapshot_identities": {
            "pre": dict(preflight.pre_identity),
            "post": post_identity,
        },
        "raw_window": {
            "start_seconds": 0.0,
            "end_seconds": EARLY_RAW_WINDOW_SECONDS,
            "columns": list(RAW_WINDOW_COLUMNS),
            "rows": [],
        },
        "limitations": [
            "The workload was not started because pre-run evidence failed closed.",
        ],
        "snapshots": {
            "pre": {
                "path": preflight.pre_path.name,
                "status": preflight.pre_capture.get("status"),
                "timestamp_monotonic_ns": preflight.pre_capture.get(
                    "timestamp_monotonic_ns"
                ),
                "stdout_sha256": preflight.pre_capture.get("stdout_sha256"),
                "identity": dict(preflight.pre_identity),
            },
            "post": {
                "path": None,
                "status": "not_started",
                "timestamp_monotonic_ns": None,
                "stdout_sha256": None,
                "identity": post_identity,
            },
            "command": list(SNAPSHOT_COMMAND_TUPLE),
            "timeout_seconds": SNAPSHOT_TIMEOUT_SECONDS,
            "read_only": True,
        },
        "trace": {
            "path": trace_path_value,
            "schema": ISSUE121_Q2_TRACE_SCHEMA,
            "schema_version": ISSUE121_Q2_TRACE_SCHEMA_VERSION,
            "columns": list(EXTENDED_TRACE_COLUMNS),
            "legacy_columns_unchanged": list(LEGACY_POWER_TRACE_COLUMNS),
            "board_id": normalized_board,
            "bus_id": normalized_bus,
            "timestamp_axis": "timestamp_monotonic_ns",
            "run_start_monotonic_ns": None,
            "run_end_monotonic_ns": None,
            "sample_count": trace_sample_count,
            "pre_row_written": preflight.pre_row,
            "post_row_written": False,
            "usable_in_run_row_count": 0,
            "in_run_captures": [],
            "sha256": trace_sha256,
            "sampler_interval_seconds": preflight.interval,
            "sampler_errors": list(
                _redact_capture_value(list(preflight.sampler_errors))
            ),
        },
        "output_paths": output_paths,
        "provenance": {
            "source_snapshot": "pre-run tt-smi snapshot",
            "trace_source": "extended-v1 snapshots captured by this harness",
            "analysis_source": "tools/analyze_issue121_q2.py",
            "limit_provenance": "snapshot_limits.pre",
            "host_identity_persisted": False,
        },
    }
    _safe_record_value(record)
    _write_json(record_path, record)
    return record


def _capture_preflight(
    *,
    safe_output_dir: Path,
    safe_run_id: str,
    capture: Callable[..., dict[str, Any]],
    monotonic_ns: Callable[[], int],
    utc_now: Callable[[], str],
    normalized_board: str,
    normalized_bus: str,
    interval: float,
) -> _PreflightResult:
    """Capture and validate pre-run evidence before creating workload setup."""
    try:
        raw_pre_capture = capture(
            stage="pre_run", monotonic_ns=monotonic_ns, utc_now=utc_now
        )
    except Exception as exc:  # noqa: BLE001 - persist a bounded preflight failure
        raw_pre_capture = {
            "stage": "pre_run",
            "status": "capture_failed",
            "returncode": None,
            "error_type": type(exc).__name__,
            "snapshot": None,
        }
    pre_capture = _safe_capture_details(raw_pre_capture)
    pre_capture["stage"] = "pre_run"
    pre_path = _safe_artifact_path(
        safe_output_dir, f"pre-snapshot-{safe_run_id}.json", field="pre snapshot"
    )
    _write_json(pre_path, pre_capture)
    trace_path = _safe_artifact_path(
        safe_output_dir, f"extended-trace-{safe_run_id}.csv", field="extended trace"
    )
    safe_trace_path = _safe_output_path(trace_path, field="extended trace")
    sampler_errors: list[str] = []
    capture_successful = _snapshot_capture_is_successful(pre_capture)
    pre_identity = _record_snapshot_identity(
        pre_capture, board_id=normalized_board, bus_id=normalized_bus
    )
    identity_valid = _snapshot_identity_is_valid(
        pre_identity, board_id=normalized_board, bus_id=normalized_bus
    )
    pre_row = False
    trace_created = False
    reason: str | None = None
    if not capture_successful:
        try:
            _require_successful_snapshot_capture(pre_capture, label="pre-run")
        except SnapshotCaptureError as exc:
            reason = str(exc)
    elif not identity_valid:
        reason = (
            "pre-run snapshot identity does not match the requested board/bus: "
            + str(pre_identity.get("reason", "identity unavailable"))
        )
    else:
        try:
            with safe_trace_path.open("w", encoding="utf-8", newline="") as trace_handle:
                trace_created = True
                trace_handle.write(",".join(EXTENDED_TRACE_COLUMNS) + "\n")
                pre_row = _write_boundary_extended_row(
                    trace_handle,
                    pre_capture,
                    board_id=normalized_board,
                    bus_id=normalized_bus,
                    stage="pre-run",
                    errors=sampler_errors,
                )
        except (OSError, TypeError, ValueError) as exc:
            sampler_errors.append(f"pre-run trace setup failed: {type(exc).__name__}")
            pre_row = False
        if not pre_row:
            reason = "; ".join(sampler_errors) or "pre-run telemetry row was not written"
    return _PreflightResult(
        pre_capture=pre_capture,
        pre_path=pre_path,
        trace_path=trace_path,
        safe_trace_path=safe_trace_path,
        pre_identity=pre_identity,
        capture_successful=capture_successful,
        identity_valid=identity_valid,
        pre_row=pre_row,
        trace_created=trace_created,
        reason=reason,
        interval=interval,
        sampler_errors=sampler_errors,
    )


def _run_one(output_dir: Path, options: _RunOptions) -> dict[str, Any]:
    """Run one fixed Q2 attempt; hardware execution is never implicit on import."""
    run_id = options.run_id
    run_number = options.run_number
    board_id = options.board_id
    bus_id = options.bus_id
    image_digest = options.image_digest
    harness_commit = options.harness_commit
    kernel = options.kernel
    kernel_driver_version = options.kernel_driver_version
    container_device_id = options.container_device_id
    trace_interval_seconds = options.trace_interval_seconds
    workload_runner = options.workload_runner
    capture_fn = options.capture_fn
    monotonic_ns = options.monotonic_ns
    utc_now = options.utc_now
    visible_device_id = _validate_container_visible_device_id(container_device_id)
    output_dir = _safe_output_dir(output_dir)
    safe_output_dir = _safe_output_path(output_dir, field="output_dir")
    safe_output_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = _safe_run_id(run_id)
    if run_number is not None and run_number not in {1, 2, 3}:
        raise Issue121Q2Error("run_number must be 1, 2, or 3")
    normalized_board, normalized_bus = _safe_identity(board_id, bus_id)
    interval = _finite_number(trace_interval_seconds, field="trace_interval_seconds", positive=True)
    capture = capture_readonly_snapshot if capture_fn is None else capture_fn

    preflight = _capture_preflight(
        safe_output_dir=safe_output_dir,
        safe_run_id=safe_run_id,
        capture=capture,
        monotonic_ns=monotonic_ns,
        utc_now=utc_now,
        normalized_board=normalized_board,
        normalized_bus=normalized_bus,
        interval=interval,
    )
    if preflight.reason is not None:
        return _failed_preflight_record(
            safe_output_dir=safe_output_dir,
            safe_run_id=safe_run_id,
            run_number=run_number,
            normalized_board=normalized_board,
            normalized_bus=normalized_bus,
            preflight=preflight,
        )
    pre_capture = preflight.pre_capture
    pre_path = preflight.pre_path
    trace_path = preflight.trace_path
    safe_trace_path = preflight.safe_trace_path
    sampler_errors = preflight.sampler_errors
    pre_row = preflight.pre_row

    environment = _build_environment(
        pre_capture,
        board_id=normalized_board,
        bus_id=normalized_bus,
        image_digest=image_digest,
        harness_commit=harness_commit,
        kernel=kernel,
        kernel_driver_version=kernel_driver_version,
        snapshot_name=pre_path.name,
        container_device_id=visible_device_id,
    )
    calibration_path = _safe_artifact_path(
        safe_output_dir, "issue121-q2-calibration.json", field="calibration metadata"
    )
    _write_json(calibration_path, build_calibration_payload())
    result_name = f"runner-result-{safe_run_id}.json"
    command = build_workload_command(
        output_dir=output_dir,
        run_id=safe_run_id,
        result_name=result_name,
        calibration_name=calibration_path.name,
        container_device_id=visible_device_id,
    )
    workload_environment = build_workload_environment(
        run_id=safe_run_id,
        image_digest=image_digest,
        container_device_id=visible_device_id,
    )

    in_run_captures: list[dict[str, Any]] = []
    in_run_row_results: list[bool] = []
    stop_event = threading.Event()
    with safe_trace_path.open("a", encoding="utf-8", newline="") as trace_handle:
        sampler_thread = threading.Thread(
            target=_sample_extended_trace,
            args=(stop_event, trace_handle),
            kwargs={
                "board_id": normalized_board,
                "bus_id": normalized_bus,
                "interval_seconds": interval,
                "captures": in_run_captures,
                "errors": sampler_errors,
                "capture_fn": capture,
                "usable_rows": in_run_row_results,
            },
            name="issue121-q2-extended-sampler",
            daemon=True,
        )
        sampler_thread.start()
        workload_start_ns = monotonic_ns()
        if workload_runner is None:
            workload_result = _run_workload(
                command,
                workload_environment,
                repo_root=Path(__file__).resolve().parents[1],
                output_dir=safe_output_dir,
            )
        else:
            callback_result = workload_runner(command, workload_environment, RUN_GUARD_SECONDS)
            workload_result = _normalize_workload_result(callback_result)
            workload_result.update(_persist_workload_streams(safe_output_dir, callback_result))
        workload_end_ns = monotonic_ns()
        stop_event.set()
        sampler_thread.join(timeout=interval + 1.0)
        if sampler_thread.is_alive():
            sampler_errors.append("extended sampler did not stop within its bounded join")
        post_capture = capture(stage="post_run", monotonic_ns=monotonic_ns, utc_now=utc_now)
        post_row = _write_boundary_extended_row(
            trace_handle,
            post_capture,
            board_id=normalized_board,
            bus_id=normalized_bus,
            stage="post-run",
            errors=sampler_errors,
        )

    if not any(in_run_row_results):
        sampler_errors.append("extended sampler captured no usable in-run telemetry rows")

    post_path = _safe_artifact_path(
        safe_output_dir, f"post-snapshot-{safe_run_id}.json", field="post snapshot"
    )
    _write_json(post_path, post_capture)
    try:
        trace_bytes = safe_trace_path.read_bytes()
    except OSError as exc:
        raise Issue121Q2Error("extended trace could not be read after capture") from exc
    snapshot_limits = {
        "pre": _extract_snapshot_limits(
            pre_capture, board_id=normalized_board, bus_id=normalized_bus
        ),
        "post": _extract_snapshot_limits(
            post_capture, board_id=normalized_board, bus_id=normalized_bus
        ),
    }
    snapshot_identities = {
        "pre": _record_snapshot_identity(
            pre_capture, board_id=normalized_board, bus_id=normalized_bus
        ),
        "post": _record_snapshot_identity(
            post_capture, board_id=normalized_board, bus_id=normalized_bus
        ),
    }
    raw_window = _raw_window_from_trace(
        safe_trace_path,
        run_start_ns=int(workload_start_ns),
        run_end_ns=int(workload_end_ns),
    )
    workload = _normalize_workload_result(workload_result)
    if isinstance(workload_result, Mapping) and "local_diagnostics" in workload_result:
        workload["local_diagnostics"] = workload_result["local_diagnostics"]
    workload_duration = max(0.0, (int(workload_end_ns) - int(workload_start_ns)) / 1_000_000_000.0)
    trace_sample_count = max(0, sum(1 for line in trace_bytes.decode("utf-8").splitlines()[1:] if line))
    guard_pass = workload_duration <= RUN_GUARD_SECONDS and not workload.get("timed_out", False)
    pre_identity_valid = _snapshot_identity_is_valid(
        snapshot_identities["pre"], board_id=normalized_board, bus_id=normalized_bus
    )
    post_capture_ok = _snapshot_capture_is_successful(post_capture)
    post_identity_valid = _snapshot_identity_is_valid(
        snapshot_identities["post"], board_id=normalized_board, bus_id=normalized_bus
    )
    pre_capture_ok = _snapshot_capture_is_successful(pre_capture)
    status = (
        "ok"
        if _q2_run_is_ok(
            workload,
            guard_pass=guard_pass,
            sampler_errors=sampler_errors,
            pre_capture_ok=pre_capture_ok,
            pre_identity_valid=pre_identity_valid,
            pre_row=pre_row,
            post_capture_ok=post_capture_ok,
            post_identity_valid=post_identity_valid,
            post_row=post_row,
            in_run_row_results=in_run_row_results,
        )
        else "failed"
    )
    record: dict[str, Any] = {
        "record_schema": ISSUE121_Q2_RECORD_SCHEMA,
        "issue": "#121",
        "run_id": safe_run_id,
        "run_number": run_number,
        "status": status,
        "environment": environment,
        "workload": {
            "name": "adr-0013-108-core-newton-schulz-surrogate",
            "condition": "newton_schulz",
            "device_id": DEVICE_ID,
            "container_device_id": visible_device_id,
            "device_node": DEVICE_NODE,
            "worker_core_count": 108,
            "data_independent": True,
            "run_duration_target_seconds": RUN_DURATION_SECONDS,
            "run_guard_seconds": RUN_GUARD_SECONDS,
            "wrapper": "enodia/tt/bench/run_in_container.sh",
            "runner": "enodia/tt/bench/run_resident.py",
            "result_path": result_name,
            "calibration_path": calibration_path.name,
        },
        "run_timing": {
            "workload_start_monotonic_ns": int(workload_start_ns),
            "workload_end_monotonic_ns": int(workload_end_ns),
            "workload_duration_seconds": workload_duration,
            "target_duration_seconds": RUN_DURATION_SECONDS,
            "guard_seconds": RUN_GUARD_SECONDS,
            "within_guard": guard_pass,
            "timeout_seconds": RUN_GUARD_SECONDS,
        },
        "workload_result": workload,
        "snapshot_limits": snapshot_limits,
        "snapshot_identities": snapshot_identities,
        "raw_window": raw_window,
        "limitations": [
            (
                "The extended trace samples at a 2 second cadence; onset ordering "
                "has approximately 2 second temporal resolution."
            ),
            "Raw window rows are sample values; no onset interpolation is performed.",
        ],
        "snapshots": {
            "pre": {
                "path": pre_path.name,
                "status": pre_capture.get("status"),
                "timestamp_monotonic_ns": pre_capture.get("timestamp_monotonic_ns"),
                "stdout_sha256": pre_capture.get("stdout_sha256"),
                "identity": snapshot_identities["pre"],
            },
            "post": {
                "path": post_path.name,
                "status": post_capture.get("status"),
                "timestamp_monotonic_ns": post_capture.get("timestamp_monotonic_ns"),
                "stdout_sha256": post_capture.get("stdout_sha256"),
                "identity": snapshot_identities["post"],
            },
            "command": list(SNAPSHOT_COMMAND_TUPLE),
            "timeout_seconds": SNAPSHOT_TIMEOUT_SECONDS,
            "read_only": True,
        },
        "trace": {
            "path": trace_path.name,
            "schema": ISSUE121_Q2_TRACE_SCHEMA,
            "schema_version": ISSUE121_Q2_TRACE_SCHEMA_VERSION,
            "columns": list(EXTENDED_TRACE_COLUMNS),
            "legacy_columns_unchanged": list(LEGACY_POWER_TRACE_COLUMNS),
            "board_id": normalized_board,
            "bus_id": normalized_bus,
            "timestamp_axis": "timestamp_monotonic_ns",
            "run_start_monotonic_ns": int(workload_start_ns),
            "run_end_monotonic_ns": int(workload_end_ns),
            "sample_count": trace_sample_count,
            "pre_row_written": pre_row,
            "post_row_written": post_row,
            "usable_in_run_row_count": sum(in_run_row_results),
            "in_run_captures": list(in_run_captures),
            "sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "sampler_interval_seconds": interval,
            "sampler_errors": list(sampler_errors),
        },
        "output_paths": {
            "record": f"run-{safe_run_id}.json",
            "pre_snapshot": pre_path.name,
            "post_snapshot": post_path.name,
            "extended_trace": trace_path.name,
            "calibration": calibration_path.name,
            "workload_result": result_name,
            "workload_stdout": WORKLOAD_STDOUT_LOG_NAME,
            "workload_stderr": WORKLOAD_STDERR_LOG_NAME,
        },
        "provenance": {
            "source_snapshot": "pre-run tt-smi snapshot",
            "trace_source": "extended-v1 snapshots captured by this harness",
            "analysis_source": "tools/analyze_issue121_q2.py",
            "limit_provenance": "snapshot_limits.pre and snapshot_limits.post",
            "host_identity_persisted": False,
        },
    }
    record_path = _safe_artifact_path(
        safe_output_dir, f"run-{safe_run_id}.json", field="measurement record"
    )
    _safe_record_value(record)
    _write_json(record_path, record)
    return record


def run_one(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    """Run one fixed Q2 attempt with the existing keyword-only interface."""
    return _run_one(output_dir, _run_options(kwargs))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-number", type=int, default=None)
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--bus-id", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--kernel-driver-version", required=True)
    parser.add_argument("--trace-interval-seconds", type=float, default=TRACE_INTERVAL_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        record = run_one(
            args.out_dir,
            run_id=args.run_id,
            run_number=args.run_number,
            board_id=args.board_id,
            bus_id=args.bus_id,
            image_digest=args.image_digest,
            harness_commit=args.harness_commit,
            kernel=args.kernel,
            kernel_driver_version=args.kernel_driver_version,
            trace_interval_seconds=args.trace_interval_seconds,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Issue #121 Q2 run rejected: {exc}", file=os.sys.stderr)
        return 2
    print(strict_json_dumps({"status": record["status"], "run_id": record["run_id"]}))
    return 0


__all__ = [
    "CONTAINER_VISIBLE_DEVICE_ID",
    "COOLDOWN_POWER_TOLERANCE_W",
    "COOLDOWN_TEMPERATURE_TOLERANCE_C",
    "DEVICE_ID",
    "DEVICE_NODE",
    "EARLY_RAW_WINDOW_SECONDS",
    "EXTENDED_TRACE_COLUMNS",
    "ISSUE121_Q2_IMAGE_NAME",
    "ISSUE121_Q2_PLAN",
    "ISSUE121_Q2_RECORD_SCHEMA",
    "ISSUE121_Q2_TRACE_SCHEMA",
    "LEGACY_POWER_TRACE_COLUMNS",
    "RAW_WINDOW_COLUMNS",
    "RUN_DURATION_SECONDS",
    "RUN_GUARD_SECONDS",
    "SNAPSHOT_COMMAND_TUPLE",
    "SNAPSHOT_TIMEOUT_SECONDS",
    "WORKLOAD_LOG_CHUNK_BYTES",
    "WORKLOAD_OUTPUT_TAIL_LINES",
    "WORKLOAD_OUTPUT_TAIL_MAX_CHARS",
    "WORKLOAD_STDERR_LOG_NAME",
    "WORKLOAD_STDOUT_LOG_NAME",
    "SnapshotCaptureError",
    "build_calibration_payload",
    "build_workload_command",
    "build_workload_environment",
    "capture_readonly_snapshot",
    "evaluate_cooldown",
    "evaluate_cooldown_transition",
    "run_one",
]


if __name__ == "__main__":
    raise SystemExit(main())
