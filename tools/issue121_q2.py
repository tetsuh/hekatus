"""Issue #121 Q2 one-run harness and read-only telemetry capture.

The module is intentionally importable on a board-free host.  It does not
open a device or run ``tt-smi`` during import.  A real invocation launches the
existing Issue #118 resident runner through the repository wrapper, while the
pre-run, in-run, and post-run snapshots all use one bounded subprocess path.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
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

DEVICE_ID = 1
DEVICE_NODE = "/dev/tenstorrent/1"
RUN_DURATION_SECONDS = 300.0
RUN_GUARD_SECONDS = 330.0
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

COOLDOWN_BASELINE_POWER_W = 47.0
COOLDOWN_BASELINE_TEMPERATURE_C = 58.5
COOLDOWN_POWER_TOLERANCE_W = 5.0
COOLDOWN_TEMPERATURE_TOLERANCE_C = 3.0
COOLDOWN_POWER_LIMIT_W = COOLDOWN_BASELINE_POWER_W + COOLDOWN_POWER_TOLERANCE_W
COOLDOWN_TEMPERATURE_LIMIT_C = (
    COOLDOWN_BASELINE_TEMPERATURE_C + COOLDOWN_TEMPERATURE_TOLERANCE_C
)

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
    r"(?:^|[\s'\"(])(?:/home/|/Users/|/root/|/tmp/|~[/\\]|[A-Za-z]:[\\/])"
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


def _safe_output_dir(value: Path) -> Path:
    output_dir = Path(value)
    text = str(output_dir)
    if "\n" in text or "\\" in text or ".." in output_dir.parts:
        raise Issue121Q2Error("output_dir must not contain path traversal")
    return output_dir


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
    end_ns = monotonic_ns()

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
        "unit": "reported snapshot unit",
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
    recognized_sources: dict[str, Mapping[str, Any]] = {}
    for source_name in ("limits", "smbus_telem"):
        source = device.get(source_name)
        if isinstance(source, Mapping):
            recognized_sources[source_name] = source
    recognized: dict[str, Any] = {}
    for name, spec in _SNAPSHOT_LIMIT_SPECS.items():
        found = False
        for source_name, keys in spec["sources"]:
            source = recognized_sources.get(source_name)
            if source is None:
                continue
            for key in keys:
                if key not in source:
                    continue
                recognized[name] = {
                    "value": _json_safe_snapshot(source[key]),
                    "unit": spec["unit"],
                    "source_key": f"{source_name}.{key}",
                    "status": "known",
                }
                found = True
                break
            if found:
                break
        if not found:
            recognized[name] = {
                "value": None,
                "unit": None,
                "source_key": (
                    "smbus_telem.THM_LIMIT_THROTTLE"
                    if name == "thm_limit_throttle"
                    else None
                ),
                "status": "unknown",
            }
    smbus = recognized_sources.get("smbus_telem", {})
    raw_smbus = {
        str(key): _json_safe_snapshot(value)
        for key, value in smbus.items()
        if str(key).casefold() in {
            "board_power_limit",
            "tdc_limit_max",
            "thm_limit_throttle",
            "thm_limit_shutdown",
            "asic_fmax",
        }
        or "limit" in str(key).casefold()
    }
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
) -> None:
    """Sample until the workload ends; the stop wait avoids an extra row."""
    interval = _finite_number(interval_seconds, field="trace interval", positive=True)
    while not stop_event.wait(interval):
        try:
            capture = capture_fn(stage="in_run")
            captures.append(capture)
            _append_extended_row(handle, capture, board_id=board_id, bus_id=bus_id)
        except Exception as exc:  # noqa: BLE001 - preserve an incomplete trace fact
            errors.append(f"{type(exc).__name__}: in-run snapshot unavailable")


def build_workload_command(
    *,
    output_dir: Path,
    run_id: str,
    result_name: str | None = None,
    calibration_name: str = "issue121-q2-calibration.json",
) -> tuple[str, ...]:
    """Build the workload command for the host directory mounted as ``/out``."""
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
        str(DEVICE_ID),
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
) -> dict[str, str]:
    """Return wrapper environment values without retaining a host path."""
    safe_run_id = _safe_run_id(run_id)
    digest = _safe_digest(image_digest)
    if device_node != DEVICE_NODE:
        raise Issue121Q2Error(f"device_node must be {DEVICE_NODE}")
    return {
        "HEKATUS_TT_RUN_ID": safe_run_id,
        "HEKATUS_TT_RUNNER": "enodia/tt/bench/run_resident.py",
        "HEKATUS_TT_IMAGE": f"{ISSUE121_Q2_IMAGE_NAME}@{digest}",
        "HEKATUS_TT_DEVICE_NODE": device_node,
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
    observed_idle_aiclk_mhz: float | None = None,
    baseline_power_w: float = COOLDOWN_BASELINE_POWER_W,
    baseline_temperature_c: float = COOLDOWN_BASELINE_TEMPERATURE_C,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
) -> dict[str, Any]:
    """Apply the #127 c006a70 power/temperature cooldown gate."""
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
        "baseline_source": "preceding run pre-run snapshot",
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
        power = _finite_number(telemetry["board_power"], field="telemetry.board_power")
        temperature = _finite_number(
            telemetry["asic_temperature"], field="telemetry.asic_temperature"
        )
    except KeyError as exc:
        raise SnapshotCaptureError(f"snapshot telemetry is missing {exc.args[0]}") from exc
    aiclk = telemetry.get("aiclk")
    return power, temperature, None if aiclk is None else _finite_number(aiclk, field="telemetry.aiclk", positive=True)


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
) -> dict[str, Any]:
    normalized_board, normalized_bus = _safe_identity(board_id, bus_id)
    device = _snapshot_device(pre_snapshot, normalized_board, normalized_bus)
    board_info = device.get("board_info")
    if not isinstance(board_info, Mapping):
        raise SnapshotCaptureError("selected snapshot device has no board_info")
    serial = normalize_board_id(board_info.get("serial")) or normalize_board_id(
        board_info.get("board_id")
    )
    if serial is None:
        raise SnapshotCaptureError("selected snapshot device has no board serial")
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
            "serial": serial,
            "board_type": board_info.get("board_type"),
            "identity": {"board_id": normalized_board, "bus_id": normalized_bus},
        },
        "firmware": {"bundle": str(firmware_bundle)},
        "kernel": kernel.strip(),
        "kernel_driver_version": kernel_driver_version.strip(),
        "image_digest": _safe_digest(image_digest),
        "harness_commit": _safe_commit(harness_commit),
        "device_node": DEVICE_NODE,
        "pre_snapshot": safe_snapshot_name,
    }


def _normalize_workload_result(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return {
            "returncode": result.get("returncode"),
            "status": str(result.get("status", "completed")),
            "timed_out": bool(result.get("timed_out", False)),
        }
    return {
        "returncode": getattr(result, "returncode", result if isinstance(result, int) else None),
        "status": "completed",
        "timed_out": False,
    }


def _run_workload(
    command: Sequence[str], environment: Mapping[str, str], *, repo_root: Path
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=repo_root,
            env={**os.environ, **dict(environment)},
            capture_output=True,
            text=True,
            timeout=RUN_GUARD_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"returncode": None, "status": "timeout", "timed_out": True}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "status": type(exc).__name__, "timed_out": False}
    return _normalize_workload_result(completed)


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
    path.write_text(strict_json_dumps(value, indent=2) + "\n", encoding="utf-8")


def run_one(
    output_dir: Path,
    *,
    run_id: str,
    run_number: int | None = None,
    board_id: str,
    bus_id: str,
    image_digest: str,
    harness_commit: str,
    kernel: str,
    kernel_driver_version: str,
    trace_interval_seconds: float = TRACE_INTERVAL_SECONDS,
    workload_runner: Callable[[Sequence[str], Mapping[str, str], float], Any] | None = None,
    capture_fn: Callable[..., dict[str, Any]] | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    utc_now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Run one fixed Q2 attempt; hardware execution is never implicit on import."""
    output_dir = _safe_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = _safe_run_id(run_id)
    if run_number is not None and run_number not in {1, 2, 3}:
        raise Issue121Q2Error("run_number must be 1, 2, or 3")
    normalized_board, normalized_bus = _safe_identity(board_id, bus_id)
    interval = _finite_number(trace_interval_seconds, field="trace_interval_seconds", positive=True)
    capture = capture_readonly_snapshot if capture_fn is None else capture_fn

    pre_capture = capture(stage="pre_run", monotonic_ns=monotonic_ns, utc_now=utc_now)
    pre_path = output_dir / f"pre-snapshot-{safe_run_id}.json"
    _write_json(pre_path, pre_capture)
    environment = _build_environment(
        pre_capture,
        board_id=normalized_board,
        bus_id=normalized_bus,
        image_digest=image_digest,
        harness_commit=harness_commit,
        kernel=kernel,
        kernel_driver_version=kernel_driver_version,
        snapshot_name=pre_path.name,
    )

    calibration_path = output_dir / "issue121-q2-calibration.json"
    _write_json(calibration_path, build_calibration_payload())
    trace_path = output_dir / f"extended-trace-{safe_run_id}.csv"
    result_name = f"runner-result-{safe_run_id}.json"
    command = build_workload_command(
        output_dir=output_dir,
        run_id=safe_run_id,
        result_name=result_name,
        calibration_name=calibration_path.name,
    )
    workload_environment = build_workload_environment(
        run_id=safe_run_id,
        image_digest=image_digest,
    )

    in_run_captures: list[dict[str, Any]] = []
    sampler_errors: list[str] = []
    stop_event = threading.Event()
    with trace_path.open("w", encoding="utf-8", newline="") as trace_handle:
        trace_handle.write(",".join(EXTENDED_TRACE_COLUMNS) + "\n")
        pre_row = _append_extended_row(
            trace_handle,
            pre_capture,
            board_id=normalized_board,
            bus_id=normalized_bus,
        )
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
            )
        else:
            workload_result = _normalize_workload_result(
                workload_runner(command, workload_environment, RUN_GUARD_SECONDS)
            )
        workload_end_ns = monotonic_ns()
        stop_event.set()
        sampler_thread.join(timeout=interval + 1.0)
        if sampler_thread.is_alive():
            sampler_errors.append("extended sampler did not stop within its bounded join")
        post_capture = capture(stage="post_run", monotonic_ns=monotonic_ns, utc_now=utc_now)
        post_row = _append_extended_row(
            trace_handle,
            post_capture,
            board_id=normalized_board,
            bus_id=normalized_bus,
        )

    post_path = output_dir / f"post-snapshot-{safe_run_id}.json"
    _write_json(post_path, post_capture)
    try:
        trace_bytes = trace_path.read_bytes()
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
    raw_window = _raw_window_from_trace(
        trace_path,
        run_start_ns=int(workload_start_ns),
        run_end_ns=int(workload_end_ns),
    )
    workload = _normalize_workload_result(workload_result)
    workload_duration = max(0.0, (int(workload_end_ns) - int(workload_start_ns)) / 1_000_000_000.0)
    trace_sample_count = max(0, sum(1 for line in trace_bytes.decode("utf-8").splitlines()[1:] if line))
    guard_pass = workload_duration <= RUN_GUARD_SECONDS and not workload.get("timed_out", False)
    status = "ok" if workload.get("returncode") == 0 and guard_pass and not sampler_errors else "failed"
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
            },
            "post": {
                "path": post_path.name,
                "status": post_capture.get("status"),
                "timestamp_monotonic_ns": post_capture.get("timestamp_monotonic_ns"),
                "stdout_sha256": post_capture.get("stdout_sha256"),
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
        },
        "provenance": {
            "source_snapshot": "pre-run tt-smi snapshot",
            "trace_source": "extended-v1 snapshots captured by this harness",
            "analysis_source": "tools/analyze_issue121_q2.py",
            "limit_provenance": "snapshot_limits.pre and snapshot_limits.post",
            "host_identity_persisted": False,
        },
    }
    record_path = output_dir / f"run-{safe_run_id}.json"
    _safe_record_value(record)
    _write_json(record_path, record)
    return record


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
    except (Issue121Q2Error, OSError, TypeError, ValueError) as exc:
        print(f"Issue #121 Q2 run rejected: {exc}", file=os.sys.stderr)
        return 2
    print(strict_json_dumps({"status": record["status"], "run_id": record["run_id"]}))
    return 0


__all__ = [
    "COOLDOWN_BASELINE_POWER_W",
    "COOLDOWN_BASELINE_TEMPERATURE_C",
    "COOLDOWN_POWER_LIMIT_W",
    "COOLDOWN_TEMPERATURE_LIMIT_C",
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
    "SnapshotCaptureError",
    "build_calibration_payload",
    "build_workload_command",
    "build_workload_environment",
    "capture_readonly_snapshot",
    "evaluate_cooldown",
    "run_one",
]


if __name__ == "__main__":
    raise SystemExit(main())
