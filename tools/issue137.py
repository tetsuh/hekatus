"""Issue #137 board-power runner and trace provenance.

This module keeps planning board-free and makes one explicitly approved
smoke or long invocation.  The Issue #137 load is host-duration-bounded:
each device callback performs one fixed Newton-Schulz launch and returns only
after device completion.  The observed launch count is never an input.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench import power_trace_issue137, run_resident
from enodia.tt.bench.duration_load import (
    DEFAULT_OVERRUN_LIMIT_SECONDS,
    validate_duration_bounds,
)
from enodia.tt.bench.load_harness import (
    NewtonSchulzCorePlan,
    build_newton_schulz_core_plan,
)
from enodia.tt.bench.resident_harness import validate_harness_preflight
from enodia.tt.bench.run_binding import validate_run_id
from tools import issue121_q2

DEFAULT_PLAN_PATH = Path(
    "docs/measurements/2026-10-18-host-newton-schulz-issue137-plan.json"
)
ISSUE137_RECORD_SCHEMA = "issue137-measurement-v1"
ISSUE137_DEVICE_RESULT_SCHEMA = "issue137-device-result-v1"
ISSUE137_SNAPSHOT_SCHEMA = "issue137-tt-smi-snapshot-v1"
ISSUE137_LOAD_MODE = "host-duration-bounded"
ISSUE137_DEVICE_RUNNER = "tools/issue137_device.py"
WRAPPER_SCRIPT = "enodia/tt/bench/run_in_container.sh"
DEVICE_NODE = "/dev/tenstorrent/1"
CONTAINER_VISIBLE_DEVICE_ID = 0
COOLDOWN_CAP_SECONDS = 600.0
COOLDOWN_POWER_TOLERANCE_W = 5.0
COOLDOWN_TEMPERATURE_TOLERANCE_C = 3.0
WRAPPER_CLEANUP_MARGIN_SECONDS = 5.0
SAMPLER_SHUTDOWN_MARGIN_SECONDS = 1.0

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
_FORBIDDEN_PATH_MARKERS = ("/home/", "/Users/", "/root/", "/tmp/")
_RESULT_MARKER_RE = re.compile(
    r"(?:results?|issue137 result)\s*[-=]*>\s*(\S+)", re.IGNORECASE
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


class Issue137Error(ValueError):
    """Raised when the Issue #137 plan, execution, or record is unsafe."""


@dataclass(frozen=True)
class Issue137Configuration:
    """One immutable owner-plan configuration and its exact core plan."""

    identifier: str
    worker_core_count: int
    role: str
    verdict_bearing: bool
    core_plan: NewtonSchulzCorePlan

    @property
    def total_blocks(self) -> int:
        return self.core_plan.total_blocks

    @property
    def batch(self) -> int:
        return self.core_plan.batch

    def as_record(self) -> dict[str, Any]:
        core = self.core_plan.as_record()
        return {
            "id": self.identifier,
            "worker_core_count": self.worker_core_count,
            "role": self.role,
            "verdict_bearing": self.verdict_bearing,
            "total_blocks": self.total_blocks,
            "batch": self.batch,
            "matrix_block": self.core_plan.matrix_block,
            "worker_core_coordinates": [list(core) for core in self.core_plan.worker_cores],
            "worker_pool": [list(core) for core in self.core_plan.worker_pool],
            "block_distribution": [dict(row) for row in self.core_plan.block_distribution],
            "work_distribution": [dict(row) for row in self.core_plan.work_distribution],
            "l1_ledger": dict(self.core_plan.l1_ledger),
            "selection_rule": core["selection_rule"],
            "core_plan": core,
        }


class _CapturePublicationState:
    """Permit bounded sampler shutdown without publishing after finalization."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.revoked = False
        self.late_captures: list[Any] = []

    def revoke(self) -> None:
        with self.lock:
            self.revoked = True

    def retain_late_capture(self, capture: Any) -> None:
        with self.lock:
            self.late_captures.append(capture)

    def drain_late_captures(self) -> list[Any]:
        with self.lock:
            captures = list(self.late_captures)
            self.late_captures.clear()
            return captures


def _safe_json_value(value: Any) -> Any:
    """Copy JSON data while dropping host identity and absolute host paths."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_KEYS:
                continue
            result[key_text] = _safe_json_value(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(child) for child in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if any(marker in value for marker in _FORBIDDEN_PATH_MARKERS):
            return "<redacted-path>"
        return value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_plan_path(path: Path) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Issue137Error("plan path cannot be resolved") from exc
    if not resolved.is_file():
        raise Issue137Error("plan path is not a file")
    return resolved


def _plan_hash(plan: Mapping[str, Any]) -> str:
    try:
        encoded = strict_json_dumps(plan, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Issue137Error("Issue #137 plan cannot be hashed") from exc
    return hashlib.sha256(encoded).hexdigest()


def load_plan(path: Path = DEFAULT_PLAN_PATH) -> dict[str, Any]:
    """Load and validate the immutable owner plan without rewriting it."""
    safe_path = _safe_plan_path(Path(path))
    try:
        payload = json.loads(safe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Issue137Error(f"cannot read Issue #137 plan: {safe_path.name}") from exc
    if not isinstance(payload, dict):
        raise Issue137Error("Issue #137 plan must be a JSON object")
    if payload.get("record_schema") != "issue137-premeasurement-plan-v1":
        raise Issue137Error("unsupported Issue #137 plan schema")
    if payload.get("status") != "planned" or payload.get("board_run") is not False:
        raise Issue137Error("Issue #137 plan is not the immutable planned record")
    execution = payload.get("load_execution")
    if not isinstance(execution, Mapping) or execution.get("mode") != ISSUE137_LOAD_MODE:
        raise Issue137Error("Issue #137 plan does not select host-duration-bounded execution")
    return payload


def plan_configurations(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    configurations = plan.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise Issue137Error("Issue #137 plan configurations are missing")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in configurations:
        if not isinstance(item, Mapping):
            raise Issue137Error("Issue #137 configuration must be an object")
        identifier = item.get("id")
        count = item.get("worker_core_count")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in seen
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(item.get("role"), str)
            or not isinstance(item.get("verdict_bearing"), bool)
        ):
            raise Issue137Error("Issue #137 configuration fields are invalid")
        seen.add(identifier)
        result.append(dict(item))
    return result


def configuration(plan: Mapping[str, Any], identifier: str) -> Issue137Configuration:
    """Build one board-free configuration and its exact core provenance."""
    matches = [item for item in plan_configurations(plan) if item["id"] == identifier]
    if len(matches) != 1:
        raise Issue137Error(f"unknown Issue #137 configuration {identifier!r}")
    item = matches[0]
    core_plan = build_newton_schulz_core_plan(
        core_count=item["worker_core_count"],
        matrix_block=8,
    )
    return Issue137Configuration(
        identifier=identifier,
        worker_core_count=item["worker_core_count"],
        role=item["role"],
        verdict_bearing=item["verdict_bearing"],
        core_plan=core_plan,
    )


def long_run_order(plan: Mapping[str, Any]) -> list[str]:
    scope = plan.get("measurement_scope")
    order = scope.get("order") if isinstance(scope, Mapping) else None
    if not isinstance(order, list) or not all(isinstance(item, str) for item in order):
        raise Issue137Error("Issue #137 long-run order is missing")
    expected = {item["id"] for item in plan_configurations(plan)}
    if (
        len(order) != 8
        or set(order) != expected
        or any(order.count(identifier) != 2 for identifier in expected)
    ):
        raise Issue137Error("Issue #137 long-run order does not match the plan")
    return list(order)


def _timing_from_scope(plan: Mapping[str, Any], mode: str) -> tuple[float, float, bool]:
    scope = plan.get("measurement_scope")
    if not isinstance(scope, Mapping):
        raise Issue137Error("Issue #137 measurement scope is missing")
    if mode == "smoke":
        values = scope.get("smoke_runs")
        target_key = "workload_seconds"
    elif mode == "long":
        values = scope
        target_key = "run_duration_seconds"
    else:
        raise Issue137Error("Issue #137 has no calibration mode")
    if not isinstance(values, Mapping):
        raise Issue137Error(f"Issue #137 {mode} timing is missing")
    guard_key = "guard_seconds" if mode == "smoke" else "run_guard_seconds"
    try:
        target = float(values[target_key])
        guard = float(values[guard_key])
    except (KeyError, TypeError, ValueError) as exc:
        raise Issue137Error(f"Issue #137 {mode} timing is invalid") from exc
    try:
        validate_duration_bounds(
            target_seconds=target,
            guard_seconds=guard,
            overrun_limit_seconds=DEFAULT_OVERRUN_LIMIT_SECONDS,
        )
    except (TypeError, ValueError) as exc:
        raise Issue137Error(f"Issue #137 {mode} timing is invalid: {exc}") from exc
    return target, guard, mode != "long"


def run_timing(plan: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Return the preregistered target and fixed guard for one mode."""
    target, guard, diagnostic_only = _timing_from_scope(plan, mode)
    return {
        "load_seconds": target,
        "guard_seconds": guard,
        "diagnostic_only": diagnostic_only,
    }


def _safe_basename(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise Issue137Error(f"{field} must be a basename")
    return value


def _safe_output_dir(path: Path) -> Path:
    try:
        resolved = run_resident._safe_output_path(
            Path(path), description="Issue #137 output directory"
        )
        if Path(path).is_symlink():
            raise Issue137Error("Issue #137 output directory must not be a symlink")
        resolved.mkdir(parents=True, exist_ok=True)
        if not resolved.is_dir():
            raise Issue137Error("Issue #137 output directory is not a directory")
        return resolved
    except Issue137Error:
        raise
    except Exception as exc:
        raise Issue137Error(str(exc)) from exc


def _safe_artifact_path(output_dir: Path, name: str, *, field: str) -> Path:
    safe_name = _safe_basename(name, field=field)
    path = output_dir / safe_name
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise Issue137Error(f"{field} cannot be resolved safely") from exc
    if resolved.parent != output_dir.resolve():
        raise Issue137Error(f"{field} must remain in the Issue #137 output directory")
    return resolved


def resolve_result_path(
    *, cli_path: Path | None = None, environment_path: Path | None = None
) -> Path:
    """Apply the resident result-path equality contract without opening a device."""
    if cli_path is None and environment_path is None:
        raise Issue137Error("Issue #137 result path is required")
    try:
        if cli_path is not None and environment_path is not None:
            cli = run_resident._safe_output_path(cli_path, description="--out destination")
            environment = run_resident._safe_output_path(
                environment_path, description="HEKATUS_TT_RESULT_PATH destination"
            )
            if cli != environment:
                raise Issue137Error("--out and HEKATUS_TT_RESULT_PATH must identify one result path")
            return cli
        selected = cli_path if cli_path is not None else environment_path
        assert selected is not None
        return run_resident._safe_output_path(
            selected, description="Issue #137 result destination"
        )
    except Issue137Error:
        raise
    except Exception as exc:
        raise Issue137Error(str(exc)) from exc


def build_run_command(
    *,
    plan: Mapping[str, Any],
    configuration: Issue137Configuration,
    mode: str,
    output_dir: Path,
    run_id: str,
    calibration_name: str | None = None,
    approve_long_run: bool = False,
    result_name: str | None = None,
) -> tuple[str, ...]:
    """Build one wrapper invocation; output identity is wrapper-owned."""
    timing = run_timing(plan, mode)
    if mode == "long" and not approve_long_run:
        raise Issue137Error("long Issue #137 runs require per-instance owner approval")
    if calibration_name is not None:
        raise Issue137Error("Issue #137 duration mode does not accept calibration artifacts")
    if result_name is not None:
        raise Issue137Error("Issue #137 result path is wrapper-owned; do not pass result_name")
    try:
        safe_run_id = validate_run_id(run_id)
    except ValueError as exc:
        raise Issue137Error(str(exc)) from exc
    del safe_run_id
    safe_output = _safe_output_dir(output_dir)
    return (
        WRAPPER_SCRIPT,
        str(safe_output),
        "--",
        "--device-id",
        str(CONTAINER_VISIBLE_DEVICE_ID),
        "--condition",
        "newton_schulz",
        "--load-core-count",
        str(configuration.worker_core_count),
        "--load-total-blocks",
        str(configuration.total_blocks),
        "--load-matrix-block",
        str(configuration.core_plan.matrix_block),
        "--target-seconds",
        str(int(timing["load_seconds"])),
        "--overrun-limit-seconds",
        str(int(DEFAULT_OVERRUN_LIMIT_SECONDS)),
        "--plan-hash",
        _plan_hash(plan),
        "--mode",
        mode,
        "--outer-timeout-seconds",
        str(int(timing["guard_seconds"])),
    )


def build_planning_record(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build a board-free plan artifact with every core distribution."""
    configurations = [configuration(plan, item["id"]) for item in plan_configurations(plan)]
    return {
        "record_schema": "issue137-planning-v1",
        "record_revision": 2,
        "issue": "#137",
        "board_run": False,
        "plan_sha256": _plan_hash(plan),
        "long_run_order": long_run_order(plan),
        "ordered_long_runs": [
            {"order_index": index, "configuration_id": identifier}
            for index, identifier in enumerate(long_run_order(plan))
        ],
        "timing": {
            "smoke": run_timing(plan, "smoke"),
            "long": run_timing(plan, "long"),
            "plateau": dict(plan["plateau"]),
            "cooldown": dict(plan["cooldown"]),
        },
        "load_execution": _safe_json_value(plan["load_execution"]),
        "configurations": [item.as_record() for item in configurations],
        "trace": {
            "schema": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA,
            "schema_version": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA_VERSION,
            "columns": list(power_trace_issue137.ISSUE137_POWER_TRACE_COLUMNS),
            "unit_provenance": power_trace_issue137.ISSUE137_UNIT_PROVENANCE,
        },
    }


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise Issue137Error(f"{field} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Issue137Error(f"{field} must be numeric") from exc
    if not math.isfinite(converted):
        raise Issue137Error(f"{field} must be finite")
    return converted


def _cooldown_parameters(plan: Mapping[str, Any]) -> dict[str, float]:
    cooldown = plan.get("cooldown")
    if not isinstance(cooldown, Mapping):
        raise Issue137Error("Issue #137 cooldown plan is missing")
    criterion = str(cooldown.get("criterion", ""))
    power_match = re.search(r"power_w\s*<=\s*baseline\s*\+\s*([0-9]+(?:\.[0-9]+)?)", criterion)
    temperature_match = re.search(
        r"asic_temp_c\s*<=\s*baseline\s*\+\s*([0-9]+(?:\.[0-9]+)?)", criterion
    )
    try:
        cap = _finite(cooldown["cap_seconds"], field="cooldown cap")
    except KeyError as exc:
        raise Issue137Error("Issue #137 cooldown cap is missing") from exc
    return {
        "cap_seconds": cap,
        "power_tolerance_w": (
            float(power_match.group(1)) if power_match else COOLDOWN_POWER_TOLERANCE_W
        ),
        "temperature_tolerance_c": (
            float(temperature_match.group(1))
            if temperature_match
            else COOLDOWN_TEMPERATURE_TOLERANCE_C
        ),
    }


def evaluate_cooldown(
    baseline: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
) -> dict[str, Any]:
    """Evaluate only plan power and ASIC-temperature cooldown tolerances."""
    try:
        baseline_power = _finite(baseline["power_w"], field="baseline power_w")
        baseline_temperature = _finite(
            baseline["asic_temp_c"], field="baseline asic_temp_c"
        )
        observed_power = _finite(observed["power_w"], field="observed power_w")
        observed_temperature = _finite(
            observed["asic_temp_c"], field="observed asic_temp_c"
        )
        power_tolerance = _finite(power_tolerance_w, field="power tolerance")
        temperature_tolerance = _finite(
            temperature_tolerance_c, field="temperature tolerance"
        )
    except KeyError as exc:
        raise Issue137Error(f"cooldown reading is missing {exc.args[0]}") from exc
    power_limit = baseline_power + power_tolerance
    temperature_limit = baseline_temperature + temperature_tolerance
    power_pass = observed_power <= power_limit
    temperature_pass = observed_temperature <= temperature_limit
    return {
        "baseline_source": "preceding selected-device pre-run snapshot",
        "baseline_power_w": baseline_power,
        "baseline_temperature_c": baseline_temperature,
        "power_tolerance_w": power_tolerance,
        "temperature_tolerance_c": temperature_tolerance,
        "power_limit_w": power_limit,
        "temperature_limit_c": temperature_limit,
        "observed_power_w": observed_power,
        "observed_temperature_c": observed_temperature,
        "power_gate": power_pass,
        "temperature_gate": temperature_pass,
        "aiclk_gate": False,
        "verified": power_pass and temperature_pass,
        "gate_result": "pass" if power_pass and temperature_pass else "fail",
    }


def _snapshot_baseline(
    capture: Mapping[str, Any], *, board_id: str | None = None, bus_id: str | None = None
) -> dict[str, float]:
    if capture.get("stage") not in {None, "pre_run", "cooldown"}:
        raise Issue137Error("cooldown reading has an unsupported snapshot stage")
    if capture.get("status") != "ok" or capture.get("returncode") != 0:
        raise Issue137Error("selected-device pre-run snapshot capture failed")
    snapshot = capture.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise Issue137Error("selected-device pre-run snapshot is unavailable")
    try:
        if board_id is not None or bus_id is not None:
            from enodia.tt.bench.power_trace_extended import select_extended_board

            device = select_extended_board(snapshot, board_id=board_id, bus_id=bus_id)
        else:
            devices = snapshot.get("device_info")
            if not isinstance(devices, list) or len(devices) != 1:
                raise Issue137Error("selected-device snapshot must identify one device")
            device = devices[0]
    except (ValueError, KeyError) as exc:
        raise Issue137Error("selected-device pre-run snapshot identity is invalid") from exc
    telemetry = device.get("telemetry") if isinstance(device, Mapping) else None
    if not isinstance(telemetry, Mapping):
        raise Issue137Error("selected-device pre-run snapshot telemetry is unavailable")
    try:
        return {
            "power_w": _finite(telemetry["power"], field="snapshot power"),
            "asic_temp_c": _finite(
                telemetry["asic_temperature"], field="snapshot ASIC temperature"
            ),
        }
    except KeyError as exc:
        raise Issue137Error(f"selected-device snapshot is missing {exc.args[0]}") from exc


def wait_for_cooldown(
    preceding_pre_snapshot: Mapping[str, Any],
    *,
    capture_fn: Callable[..., dict[str, Any]],
    board_id: str | None = None,
    bus_id: str | None = None,
    cap_seconds: float = COOLDOWN_CAP_SECONDS,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll the selected device until the fixed cooldown cap or a pass."""
    baseline = _snapshot_baseline(
        preceding_pre_snapshot, board_id=board_id, bus_id=bus_id
    )
    cap = _finite(cap_seconds, field="cooldown cap")
    started = monotonic_fn()
    readings: list[Mapping[str, Any]] = []
    while True:
        elapsed = _finite(monotonic_fn() - started, field="cooldown elapsed")
        if elapsed >= cap:
            return cooldown_gate(
                baseline,
                readings=readings,
                elapsed_seconds=cap,
                cap_seconds=cap,
                power_tolerance_w=power_tolerance_w,
                temperature_tolerance_c=temperature_tolerance_c,
            )
        capture = capture_fn(stage="cooldown")
        after_capture = _finite(monotonic_fn() - started, field="cooldown elapsed")
        if after_capture >= cap:
            return cooldown_gate(
                baseline,
                readings=readings,
                elapsed_seconds=cap,
                cap_seconds=cap,
                power_tolerance_w=power_tolerance_w,
                temperature_tolerance_c=temperature_tolerance_c,
            )
        reading = _snapshot_baseline(capture, board_id=board_id, bus_id=bus_id)
        readings.append(reading)
        evaluation = evaluate_cooldown(
            baseline,
            reading,
            power_tolerance_w=power_tolerance_w,
            temperature_tolerance_c=temperature_tolerance_c,
        )
        if evaluation["verified"]:
            evaluation["poll_count"] = len(readings)
            evaluation["cooldown_wait_seconds"] = after_capture
            evaluation["wait_cap_seconds"] = cap
            return evaluation
        delay = min(1.0, max(0.0, cap - after_capture))
        if delay <= 0.0:
            break
        sleep_fn(delay)
    return cooldown_gate(
        baseline,
        readings=readings,
        elapsed_seconds=cap,
        cap_seconds=cap,
        power_tolerance_w=power_tolerance_w,
        temperature_tolerance_c=temperature_tolerance_c,
    )


def cooldown_gate(
    baseline: Mapping[str, Any],
    *,
    readings: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    cap_seconds: float = COOLDOWN_CAP_SECONDS,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
) -> dict[str, Any]:
    """Evaluate mocked cooldown polls and fail closed at the cap."""
    elapsed = _finite(elapsed_seconds, field="cooldown elapsed_seconds")
    cap = _finite(cap_seconds, field="cooldown cap")
    for index, reading in enumerate(readings):
        evaluation = evaluate_cooldown(
            baseline,
            reading,
            power_tolerance_w=power_tolerance_w,
            temperature_tolerance_c=temperature_tolerance_c,
        )
        evaluation["poll_count"] = index + 1
        evaluation["cooldown_wait_seconds"] = elapsed
        if evaluation["verified"] and elapsed < cap:
            evaluation["wait_cap_seconds"] = cap
            return evaluation
    return {
        "baseline_source": "preceding selected-device pre-run snapshot",
        "poll_count": len(readings),
        "cooldown_wait_seconds": elapsed,
        "wait_cap_seconds": cap,
        "power_tolerance_w": power_tolerance_w,
        "temperature_tolerance_c": temperature_tolerance_c,
        "verified": False,
        "gate_result": "fail",
        "aiclk_gate": False,
        "failure_reason": "cooldown wait cap expired" if elapsed >= cap else "cooldown gate failed",
    }


def _image_digest(environment: Mapping[str, Any]) -> str | None:
    if "image_digest" in environment:
        value = environment.get("image_digest")
        return value if isinstance(value, str) else None
    image = environment.get("image")
    if isinstance(image, str) and "@sha256:" in image:
        return image.rsplit("@", 1)[1]
    return None


def build_environment_record(
    environment: Mapping[str, Any], configuration: Issue137Configuration
) -> dict[str, Any]:
    """Extract required environment facts without persisting host identity."""
    if not isinstance(environment, Mapping):
        raise Issue137Error("environment must be an object")
    safe = _safe_json_value(environment)
    board = environment.get("board") or environment.get("board_info")
    if not isinstance(board, Mapping):
        raise Issue137Error("environment board identity is missing")
    identity = board.get("identity") if isinstance(board.get("identity"), Mapping) else {}
    serial = board.get("serial") or board.get("board_id") or identity.get("board_id")
    board_id = board.get("board_id") or identity.get("board_id") or serial
    bus_id = board.get("bus_id") or board.get("pci_bus_id") or identity.get("bus_id")
    firmware = environment.get("firmware") or environment.get("firmwares")
    if not isinstance(firmware, Mapping):
        raise Issue137Error("environment firmware provenance is missing")
    firmware_bundle = firmware.get("bundle") or firmware.get("fw_bundle_version")
    kernel_driver = environment.get("kernel_driver_version") or environment.get("kmd_version")
    image_digest = _image_digest(environment)
    harness_commit = environment.get("harness_commit")
    device_node = environment.get("device_node")
    required = {
        "serial": serial,
        "board_id": board_id,
        "bus_id": bus_id,
        "firmware_bundle": firmware_bundle,
        "kernel": environment.get("kernel"),
        "kernel_driver_version": kernel_driver,
        "image_digest": image_digest,
        "harness_commit": harness_commit,
        "device_node": device_node,
    }
    missing = [name for name, value in required.items() if not isinstance(value, str) or not value]
    if missing:
        raise Issue137Error("environment provenance is missing: " + ", ".join(missing))
    if _SHA256_RE.fullmatch(image_digest) is None:
        raise Issue137Error("environment image_digest must be sha256:<64 lowercase hexadecimal digits>")
    if _COMMIT_RE.fullmatch(harness_commit) is None:
        raise Issue137Error("environment harness_commit must be a hexadecimal commit identifier")
    record = {
        "board": {
            "serial": serial,
            "identity": {"board_id": board_id, "bus_id": bus_id},
        },
        "firmware": {"bundle": firmware_bundle},
        "kernel": environment["kernel"],
        "kernel_driver_version": kernel_driver,
        "image_digest": image_digest,
        "harness_commit": harness_commit,
        "device_node": device_node,
        "worker_core_count": configuration.worker_core_count,
        "worker_core_coordinates": [list(core) for core in configuration.core_plan.worker_cores],
        "total_blocks": configuration.total_blocks,
        "core_block_distribution": [
            dict(row) for row in configuration.core_plan.block_distribution
        ],
        "safe_environment_digest": hashlib.sha256(
            strict_json_dumps(safe, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    host_sw_vers = environment.get("host_sw_vers")
    if isinstance(host_sw_vers, Mapping):
        record["host_sw_vers"] = _safe_json_value(host_sw_vers)
    return record


def _environment_identity_matches(
    environment_record: Mapping[str, Any], *, board_id: str | None, bus_id: str | None
) -> bool:
    if board_id is None or bus_id is None:
        return False
    board = environment_record.get("board")
    identity = board.get("identity") if isinstance(board, Mapping) else None
    if not isinstance(identity, Mapping):
        return False
    return (
        str(board.get("serial")) == str(board_id)
        and str(identity.get("board_id")) == str(board_id)
        and str(identity.get("bus_id")) == str(bus_id)
    )


def validate_environment_for_run(environment: Mapping[str, Any]) -> None:
    """Apply clean/pushed preflight before any device open."""
    try:
        validate_harness_preflight(
            environment,
            pushed=environment.get("harness_pushed"),
        )
    except Exception as exc:
        raise Issue137Error(str(exc)) from exc
    if environment.get("device_node", DEVICE_NODE) != DEVICE_NODE:
        raise Issue137Error(f"Issue #137 requires device_node {DEVICE_NODE}")
    sampler = environment.get("telemetry_sampler")
    if sampler is not None and not isinstance(sampler, Mapping):
        raise Issue137Error("telemetry_sampler provenance must be an object")


def capture_readonly_snapshot(*, stage: str, **kwargs: Any) -> dict[str, Any]:
    """Capture one read-only snapshot through the established Q2 seam."""
    from tools.issue121_q2 import capture_readonly_snapshot as capture_q2

    capture = dict(capture_q2(stage=stage, **kwargs))
    capture["schema"] = ISSUE137_SNAPSHOT_SCHEMA
    return capture


def _validate_capture_identity(
    capture: Mapping[str, Any], *, stage: str, board_id: str | None, bus_id: str | None
) -> None:
    if capture.get("schema") not in {None, ISSUE137_SNAPSHOT_SCHEMA}:
        raise Issue137Error(f"{stage} snapshot schema is unsupported")
    if capture.get("stage") not in {stage, None}:
        raise Issue137Error(f"{stage} snapshot stage is invalid")
    if capture.get("status") != "ok" or capture.get("returncode") != 0:
        raise Issue137Error(f"{stage} snapshot capture did not complete successfully")
    snapshot = capture.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise Issue137Error(f"{stage} snapshot payload is missing")
    try:
        from enodia.tt.bench.power_trace_extended import select_extended_board

        select_extended_board(snapshot, board_id=board_id, bus_id=bus_id)
    except (ValueError, KeyError, TypeError) as exc:
        raise Issue137Error(f"{stage} snapshot selected-device identity is invalid") from exc
    timestamp = capture.get("timestamp_monotonic_ns")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise Issue137Error(f"{stage} snapshot monotonic timestamp is invalid")


def trace_rows_from_captures(
    captures: Sequence[Mapping[str, Any]],
    *,
    board_id: str | None,
    bus_id: str | None,
) -> list[dict[str, Any]]:
    """Convert captures strictly; malformed or missing rows fail closed."""
    rows: list[dict[str, Any]] = []
    saw_in_run = False
    for index, capture in enumerate(captures):
        if not isinstance(capture, Mapping):
            raise Issue137Error(f"Issue #137 capture {index} is not an object")
        stage = capture.get("stage")
        if stage not in {"pre_run", "in_run", "post_run"}:
            continue
        _validate_capture_identity(
            capture, stage=str(stage), board_id=board_id, bus_id=bus_id
        )
        if stage == "in_run":
            saw_in_run = True
        snapshot = capture.get("snapshot")
        try:
            row = power_trace_issue137.parse_issue137_snapshot(
                snapshot,
                board_id=board_id,
                bus_id=bus_id,
                timestamp_utc=capture.get("captured_at_utc"),
                timestamp_monotonic_ns=capture.get("timestamp_monotonic_ns"),
                snapshot_elapsed_ns=capture.get("snapshot_elapsed_ns"),
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise Issue137Error(
                f"Issue #137 capture {index} could not produce a complete trace row"
            ) from exc
        rows.append(row)
    if not saw_in_run:
        raise Issue137Error("Issue #137 trace has no in-run capture")
    if not rows:
        raise Issue137Error("no usable Issue #137 telemetry captures were recorded")
    timestamps = [row.get("timestamp_monotonic_ns") for row in rows]
    if any(
        isinstance(left, int) and isinstance(right, int) and right < left
        for left, right in itertools.pairwise(timestamps)
    ):
        raise Issue137Error("Issue #137 trace capture timestamps are not monotonic")
    return rows


def write_trace_from_captures(
    path: Path,
    captures: Sequence[Mapping[str, Any]],
    *,
    board_id: str | None,
    bus_id: str | None,
) -> Path:
    """Write one strict Issue #137 trace from injected or live captures."""
    destination = Path(path)
    rows = trace_rows_from_captures(captures, board_id=board_id, bus_id=bus_id)
    try:
        destination = run_resident._safe_output_path(
            destination, description="Issue #137 trace destination"
        )
        if destination.exists() or destination.is_symlink():
            raise Issue137Error("Issue #137 trace destination collides with an existing artifact")
    except Issue137Error:
        raise
    except Exception as exc:
        raise Issue137Error(str(exc)) from exc
    return power_trace_issue137.write_issue137_power_trace(destination, rows)


def sample_issue137_trace(
    stop_event: threading.Event,
    captures: list[dict[str, Any]],
    *,
    capture_fn: Callable[..., dict[str, Any]],
    interval_seconds: float,
    publication_state: _CapturePublicationState | None = None,
) -> None:
    """Capture bounded snapshots until the workload stops."""
    interval = _finite(interval_seconds, field="trace interval")
    if interval <= 0.0:
        raise Issue137Error("trace interval must be positive")
    state = publication_state or _CapturePublicationState()
    while not stop_event.wait(interval):
        try:
            capture = capture_fn(stage="in_run")
        except Exception as exc:  # noqa: BLE001 - preserve a structured failure
            capture = {
                "stage": "in_run",
                "status": "capture_failed",
                "returncode": None,
                "error_type": type(exc).__name__,
            }
        with state.lock:
            retained = dict(capture) if isinstance(capture, Mapping) else capture
            if state.revoked:
                state.late_captures.append(retained)
            else:
                captures.append(retained)


def _artifact_names(output_dir: Path) -> set[str]:
    try:
        return {entry.name for entry in output_dir.iterdir()}
    except OSError as exc:
        raise Issue137Error("Issue #137 output directory cannot be inspected") from exc


def _reject_output_collisions(
    output_dir: Path, *, run_id: str, trace_path: Path | None
) -> None:
    names = {
        f"run-{run_id}.json",
        f"pre-snapshot-{run_id}.json",
        f"post-snapshot-{run_id}.json",
    }
    if trace_path is not None:
        names.add(Path(trace_path).name)
    try:
        existing_entries = list(output_dir.iterdir())
    except OSError as exc:
        raise Issue137Error("Issue #137 output collision check failed") from exc
    for entry in existing_entries:
        if entry.is_symlink() and (
            entry.name.startswith(("runner-result", "env-", "power-", "run-", "pre-", "post-"))
            or entry.name in names
        ):
            raise Issue137Error(f"Issue #137 output artifact is a symlink: {entry.name}")
    for name in names:
        path = output_dir / _safe_basename(name, field="artifact name")
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise Issue137Error("Issue #137 output collision check failed") from exc
        raise Issue137Error(f"Issue #137 output collision: {name}")


def _write_artifact(path: Path, payload: Any) -> None:
    try:
        if path.exists() or path.is_symlink():
            raise Issue137Error(f"Issue #137 artifact already exists: {path.name}")
        path.write_text(strict_json_dumps(_safe_json_value(payload), indent=2) + "\n", encoding="utf-8")
    except Issue137Error:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise Issue137Error(f"Issue #137 artifact could not be written: {path.name}") from exc


def _extract_result_name(stdout: Any) -> str | None:
    if isinstance(stdout, bytes):
        text = stdout.decode("utf-8", errors="replace")
    elif isinstance(stdout, str):
        text = stdout
    else:
        return None
    matches = _RESULT_MARKER_RE.findall(text)
    if not matches:
        return None
    try:
        return _safe_basename(Path(matches[-1]).name, field="runner result name")
    except Issue137Error:
        return None


def _resolve_runner_result_artifact(
    output_dir: Path, *, before_names: set[str], completed: Any
) -> tuple[Path, str]:
    """Bind the result to a returned marker or one newly-created artifact."""
    marker_name = _extract_result_name(getattr(completed, "stdout", None))
    after_names = _artifact_names(output_dir)
    new_names = sorted(
        name
        for name in after_names - before_names
        if name.startswith("runner-result") and name.endswith(".json")
    )
    candidates: list[str] = []
    if marker_name is not None:
        candidates.append(marker_name)
    candidates.extend(name for name in new_names if name not in candidates)
    if not candidates:
        raise Issue137Error("runner result artifact was not returned")
    if len(candidates) != 1:
        raise Issue137Error("runner result artifact identity is ambiguous")
    name = candidates[0]
    if marker_name is not None and name not in new_names and name in before_names:
        raise Issue137Error("runner result artifact is stale")
    path = _safe_artifact_path(output_dir, name, field="runner result")
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise Issue137Error("runner result artifact cannot be inspected") from exc
    if path.is_symlink() or not path.is_file() or stat_result.st_nlink != 1:
        raise Issue137Error("runner result artifact identity is unsafe")
    return path, name


def _read_runner_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Issue137Error("runner result artifact is malformed") from exc
    if not isinstance(payload, dict):
        raise Issue137Error("runner result artifact must be an object")
    return payload


def _integer_field(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Issue137Error(f"runner result field {field} is invalid")
    return value


def validate_runner_result(
    result: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    configuration: Issue137Configuration,
    mode: str,
) -> dict[str, Any]:
    """Validate actual device output, not merely its process return code."""
    if result.get("record_schema") != ISSUE137_DEVICE_RESULT_SCHEMA:
        raise Issue137Error("runner result schema is unsupported")
    if result.get("status") != "ok":
        raise Issue137Error("runner result did not report successful execution")
    if result.get("mode") != mode or result.get("load_mode") != ISSUE137_LOAD_MODE:
        raise Issue137Error("runner result mode binding is invalid")
    timing = run_timing(plan, mode)
    if _finite(result.get("target_seconds"), field="runner target") != timing["load_seconds"]:
        raise Issue137Error("runner result target duration does not match the plan")
    if _finite(result.get("guard_seconds"), field="runner guard") != timing["guard_seconds"]:
        raise Issue137Error("runner result guard does not match the plan")
    if _finite(result.get("overrun_limit_seconds"), field="runner overrun limit") != DEFAULT_OVERRUN_LIMIT_SECONDS:
        raise Issue137Error("runner result overrun limit does not match the plan")
    if not isinstance(result.get("run_id"), str) or not result["run_id"]:
        raise Issue137Error("runner result wrapper run identity is missing")
    expected_plan = configuration.core_plan.as_record()
    actual_plan = result.get("core_plan")
    if actual_plan != expected_plan:
        raise Issue137Error("runner result core plan does not match the selected configuration")
    launch_count = _integer_field(result, "launch_count")
    if launch_count < 1:
        raise Issue137Error("runner result has no completed launches")
    first = _integer_field(result, "first_launch_start_monotonic_ns")
    workload_end = _integer_field(result, "workload_end_monotonic_ns")
    last = _integer_field(result, "last_launch_end_monotonic_ns")
    expected_end = first + round(timing["load_seconds"] * 1_000_000_000.0)
    if workload_end != expected_end or last < workload_end:
        raise Issue137Error("runner result monotonic workload bounds are invalid")
    overrun = _finite(result.get("overrun_seconds"), field="runner overrun")
    expected_overrun = (last - workload_end) / 1_000_000_000.0
    if not math.isclose(overrun, expected_overrun, rel_tol=0.0, abs_tol=1e-9):
        raise Issue137Error("runner result overrun does not match monotonic bounds")
    if overrun > DEFAULT_OVERRUN_LIMIT_SECONDS:
        raise Issue137Error("runner result exceeds the configured overrun limit")
    if not isinstance(result.get("environment"), Mapping):
        raise Issue137Error("runner result environment provenance is missing")
    if result.get("plan_sha256") != _plan_hash(plan):
        raise Issue137Error("runner result plan hash does not match the owner plan")
    return {
        "launch_count": launch_count,
        "first_launch_start_monotonic_ns": first,
        "workload_end_monotonic_ns": workload_end,
        "last_launch_end_monotonic_ns": last,
        "overrun_seconds": overrun,
    }


def _observed_unit_provenance_verified(
    environment: Mapping[str, Any] | None, captures: Sequence[Mapping[str, Any]]
) -> bool:
    candidates: list[Any] = []
    if isinstance(environment, Mapping):
        candidates.append(environment.get("host_sw_vers"))
    for capture in captures:
        snapshot = capture.get("snapshot") if isinstance(capture, Mapping) else None
        if isinstance(snapshot, Mapping):
            candidates.append(snapshot.get("host_sw_vers"))
    observed = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    return bool(observed) and all(
        power_trace_issue137.source_versions_verified(candidate) for candidate in observed
    )


def _execution_environment(
    environment: Mapping[str, Any] | None, *, guard_seconds: float
) -> dict[str, str]:
    values = dict(os.environ)
    values["HEKATUS_TT_RUNNER"] = ISSUE137_DEVICE_RUNNER
    values["HEKATUS_TT_DEVICE_NODE"] = DEVICE_NODE
    values["HEKATUS_TT_CONTAINER_DEVICE_ID"] = str(CONTAINER_VISIBLE_DEVICE_ID)
    values["HEKATUS_TT_CONTAINER_TIMEOUT_S"] = str(int(guard_seconds))
    sampler = environment.get("telemetry_sampler") if isinstance(environment, Mapping) else None
    if isinstance(sampler, Mapping):
        sampler_mode = sampler.get("mode")
        if isinstance(sampler_mode, str) and sampler_mode:
            values["HEKATUS_TT_TELEMETRY_MODE"] = sampler_mode
        values.pop("HEKATUS_TT_TELEMETRY_INTERVAL_S", None)
        interval = sampler.get("interval_seconds")
        if interval is not None and sampler_mode != "off":
            values["HEKATUS_TT_TELEMETRY_INTERVAL_S"] = str(interval)
    else:
        values["HEKATUS_TT_TELEMETRY_MODE"] = "default"
        values.pop("HEKATUS_TT_TELEMETRY_INTERVAL_S", None)
    if isinstance(environment, Mapping):
        digest = _image_digest(environment)
        if digest:
            values["HEKATUS_TT_IMAGE"] = f"ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@{digest}"
    return values


def _sampler_join_timeout_seconds(trace_interval_seconds: float) -> float:
    capture_timeout = float(issue121_q2.SNAPSHOT_TIMEOUT_SECONDS)
    interval = _finite(trace_interval_seconds, field="trace interval")
    return max(interval + SAMPLER_SHUTDOWN_MARGIN_SECONDS, capture_timeout + SAMPLER_SHUTDOWN_MARGIN_SECONDS)


def _run_issue137_workload(
    command: Sequence[str],
    *,
    capture_output: bool = True,
    text: bool = True,
    timeout: float,
    check: bool = False,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    del check
    process = subprocess.Popen(
        tuple(command),
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        env=None if env is None else dict(env),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = issue121_q2._cleanup_workload_process_group(
            process,
            stdout=getattr(exc, "stdout", None),
            stderr=getattr(exc, "stderr", None),
        )
        raise subprocess.TimeoutExpired(
            tuple(command), timeout, output=stdout, stderr=stderr
        ) from exc
    except BaseException:
        issue121_q2._cleanup_workload_process_group(process)
        raise
    if issue121_q2._workload_process_group_exists(process):
        stdout, stderr = issue121_q2._cleanup_workload_process_group(
            process, stdout=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(tuple(command), process.returncode, stdout, stderr)


def _base_record(
    *,
    plan: Mapping[str, Any],
    configuration: Issue137Configuration,
    mode: str,
    run_id: str,
    run_number: int | None,
    command: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    timing = run_timing(plan, mode)
    safe_output = str(output_dir)
    record_command = [
        "<run-output-dir>" if argument == safe_output else argument for argument in command
    ]
    return {
        "record_schema": ISSUE137_RECORD_SCHEMA,
        "record_revision": 2,
        "issue": "#137",
        "status": "planned",
        "plan_sha256": _plan_hash(plan),
        "run": {
            "run_id": run_id,
            "run_number": run_number,
            "mode": mode,
            "load_mode": ISSUE137_LOAD_MODE,
            "timing": timing,
            "target_seconds": timing["load_seconds"],
            "guard_seconds": timing["guard_seconds"],
            "overrun_limit_seconds": DEFAULT_OVERRUN_LIMIT_SECONDS,
            "launch_count": None,
            "workload_start_monotonic_ns": None,
            "workload_end_monotonic_ns": None,
            "last_launch_end_monotonic_ns": None,
            "overrun_seconds": None,
        },
        "configuration": configuration.as_record(),
        "command": record_command,
        "cooldown_gate": None,
        "verdict_evidence": False,
        "diagnostic_only": mode != "long",
        "workload_started": False,
        "environment": None,
        "provenance": {
            "wrapper_entrypoint": WRAPPER_SCRIPT,
            "runner_entrypoint": ISSUE137_DEVICE_RUNNER,
            "runner_result_path": None,
            "plan_sha256": _plan_hash(plan),
            "trace_schema": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA,
            "unit_provenance": None,
            "unit_provenance_status": "unknown",
            "host_identity_persisted": False,
        },
    }


def _failure_record(
    record: dict[str, Any], *, kind: str, message: str, returncode: int | None = None
) -> dict[str, Any]:
    record = dict(record)
    record["status"] = "failed"
    record["verdict_evidence"] = False
    record["workload_started"] = bool(record.get("workload_started"))
    failure: dict[str, Any] = {"kind": kind, "message": message}
    if returncode is not None:
        failure["returncode"] = returncode
    record["failure"] = failure
    return record


def _persist_record(output_dir: Path, run_id: str, record: Mapping[str, Any]) -> Path:
    destination = _safe_artifact_path(output_dir, f"run-{run_id}.json", field="record")
    _write_artifact(destination, record)
    return destination


def _store_capture(
    output_dir: Path, *, run_id: str, stage: str, capture: Mapping[str, Any]
) -> str:
    name = f"{stage.replace('_', '-')}-snapshot-{run_id}.json"
    path = _safe_artifact_path(output_dir, name, field=f"{stage} snapshot")
    _write_artifact(path, capture)
    return name


def run_one(
    *,
    plan: Mapping[str, Any],
    configuration: Issue137Configuration,
    mode: str,
    output_dir: Path,
    run_id: str,
    run_number: int | None = None,
    calibration: Mapping[str, Any] | None = None,
    approve_long_run: bool = False,
    command_runner: Callable[..., Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    preceding_pre_snapshot: Mapping[str, Any] | None = None,
    cooldown_readings: Sequence[Mapping[str, Any]] = (),
    cooldown_elapsed_seconds: float = 0.0,
    board_id: str | None = None,
    bus_id: str | None = None,
    trace_path: Path | None = None,
    trace_interval_seconds: float = 2.0,
    capture_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare or execute exactly one Issue #137 instance."""
    if calibration is not None:
        raise Issue137Error("Issue #137 duration mode does not accept calibration artifacts")
    try:
        safe_run_id = validate_run_id(run_id)
    except ValueError as exc:
        raise Issue137Error(str(exc)) from exc
    timing = run_timing(plan, mode)
    scope = plan.get("measurement_scope")
    effective_board_id = board_id
    effective_bus_id = bus_id
    if isinstance(scope, Mapping):
        effective_board_id = effective_board_id or scope.get("board_serial")
        effective_bus_id = effective_bus_id or scope.get("pci_bus_id")
    if mode == "smoke":
        smoke = scope.get("smoke_runs") if isinstance(scope, Mapping) else None
        allowed = smoke.get("configurations") if isinstance(smoke, Mapping) else None
        if not isinstance(allowed, list) or configuration.identifier not in allowed:
            raise Issue137Error(
                f"configuration {configuration.identifier} is not in the preregistered smoke set"
            )
    if mode == "long":
        if run_number is None or run_number < 1 or run_number > 8:
            raise Issue137Error("long runs require run_number from 1 through 8")
        ordered = long_run_order(plan)
        if ordered[run_number - 1] != configuration.identifier:
            raise Issue137Error("run_number does not match the preregistered Issue #137 order")
        if not approve_long_run:
            raise Issue137Error("long Issue #137 runs require per-instance owner approval")
    resolved_output = _safe_output_dir(output_dir)
    resolved_trace: Path | None = None
    if trace_path is not None:
        try:
            resolved_trace = run_resident._safe_output_path(
                Path(trace_path), description="Issue #137 trace destination"
            )
        except Exception as exc:
            raise Issue137Error(str(exc)) from exc
        if resolved_trace.parent != resolved_output:
            raise Issue137Error("Issue #137 trace must be beneath the run output directory")
    _reject_output_collisions(resolved_output, run_id=safe_run_id, trace_path=resolved_trace)
    command = build_run_command(
        plan=plan,
        configuration=configuration,
        mode=mode,
        output_dir=resolved_output,
        run_id=safe_run_id,
        approve_long_run=approve_long_run,
    )
    record = _base_record(
        plan=plan,
        configuration=configuration,
        mode=mode,
        run_id=safe_run_id,
        run_number=run_number,
        command=command,
        output_dir=resolved_output,
    )
    if environment is not None:
        validate_environment_for_run(environment)
        record["environment"] = build_environment_record(environment, configuration)
    if mode == "long" and run_number and run_number > 1 and preceding_pre_snapshot is None:
        raise Issue137Error("preceding selected-device pre-run baseline is required")
    if preceding_pre_snapshot is not None and mode == "long" and run_number and run_number > 1:
        cooldown = _cooldown_parameters(plan)
        baseline = _snapshot_baseline(
            preceding_pre_snapshot,
            board_id=effective_board_id,
            bus_id=effective_bus_id,
        )
        if not cooldown_readings and capture_fn is not None:
            gate = wait_for_cooldown(
                preceding_pre_snapshot,
                capture_fn=capture_fn,
                board_id=effective_board_id,
                bus_id=effective_bus_id,
                cap_seconds=cooldown["cap_seconds"],
                power_tolerance_w=cooldown["power_tolerance_w"],
                temperature_tolerance_c=cooldown["temperature_tolerance_c"],
            )
        else:
            gate = cooldown_gate(
                baseline,
                readings=cooldown_readings,
                elapsed_seconds=cooldown_elapsed_seconds,
                cap_seconds=cooldown["cap_seconds"],
                power_tolerance_w=cooldown["power_tolerance_w"],
                temperature_tolerance_c=cooldown["temperature_tolerance_c"],
            )
        record["cooldown_gate"] = gate
        if not gate["verified"]:
            failed = _failure_record(
                record,
                kind="cooldown_gate_failure",
                message="cooldown gate failed closed; workload was not started",
            )
            _persist_record(resolved_output, safe_run_id, failed)
            return failed
    else:
        record["cooldown_gate"] = {
            "required": False,
            "verified": True,
            "gate_result": "not_applicable",
            "aiclk_gate": False,
        }
    if command_runner is None:
        return record

    # The wrapper/device entrypoint performs the same preflight from its actual
    # env artifact.  A supplied fixture is validated here before any command.
    sampler = capture_fn or capture_readonly_snapshot
    captures: list[dict[str, Any]] = []
    publication_state = _CapturePublicationState()
    trace_stop = threading.Event()
    trace_thread: threading.Thread | None = None
    result: dict[str, Any] | None = None
    result_name: str | None = None
    returncode: int | None = None
    timed_out = False
    cleanup_failure: str | None = None
    try:
        pre_capture = sampler(stage="pre_run")
        _validate_capture_identity(
            pre_capture,
            stage="pre_run",
            board_id=effective_board_id,
            bus_id=effective_bus_id,
        )
        record["pre_run_snapshot"] = {
            "path": _store_capture(
                resolved_output, run_id=safe_run_id, stage="pre_run", capture=pre_capture
            ),
            "selected_device": {
                "board_id": effective_board_id,
                "bus_id": effective_bus_id,
            },
        }
        record["pre_run_snapshot"]["baseline"] = _snapshot_baseline(
            pre_capture, board_id=effective_board_id, bus_id=effective_bus_id
        )
        captures.append(dict(pre_capture))
    except (Issue137Error, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        failed = _failure_record(
            record, kind="pre_run_snapshot_failure", message=type(exc).__name__
        )
        _persist_record(resolved_output, safe_run_id, failed)
        return failed
    if resolved_trace is not None:
        trace_interval = _finite(trace_interval_seconds, field="trace interval")
        if trace_interval <= 0.0:
            raise Issue137Error("trace interval must be positive")
        trace_thread = threading.Thread(
            target=sample_issue137_trace,
            args=(trace_stop, captures),
            kwargs={
                "capture_fn": sampler,
                "interval_seconds": trace_interval,
                "publication_state": publication_state,
            },
            name="issue137-power-trace",
            daemon=True,
        )
        trace_thread.start()
    before_result_names = _artifact_names(resolved_output)
    try:
        completed = command_runner(
            command,
            capture_output=True,
            text=True,
            # The fixed plan guard is the normal workload deadline.  The
            # process-group cleanup helper applies TERM/KILL grace only after
            # this communicate timeout expires.
            timeout=timing["guard_seconds"],
            check=False,
            env=_execution_environment(environment, guard_seconds=timing["guard_seconds"]),
        )
        raw_returncode = getattr(completed, "returncode", None)
        returncode = (
            raw_returncode
            if isinstance(raw_returncode, int) and not isinstance(raw_returncode, bool)
            else None
        )
    except subprocess.TimeoutExpired:
        completed = SimpleNamespace(returncode=None, stdout="")
        timed_out = True
    except (OSError, subprocess.SubprocessError) as exc:
        completed = SimpleNamespace(returncode=None, stdout="")
        record["runner_error_type"] = type(exc).__name__
    finally:
        publication_state.revoke()
        trace_stop.set()
        if trace_thread is not None:
            trace_thread.join(timeout=_sampler_join_timeout_seconds(trace_interval))
            captures.extend(publication_state.drain_late_captures())
            if trace_thread.is_alive():
                cleanup_failure = "power sampler thread did not stop within the bounded cleanup interval"
    try:
        post_capture = sampler(stage="post_run")
        _validate_capture_identity(
            post_capture,
            stage="post_run",
            board_id=effective_board_id,
            bus_id=effective_bus_id,
        )
        captures.append(dict(post_capture))
        record["post_run_snapshot"] = {
            "path": _store_capture(
                resolved_output, run_id=safe_run_id, stage="post_run", capture=post_capture
            ),
            "selected_device": {
                "board_id": effective_board_id,
                "bus_id": effective_bus_id,
            },
        }
    except (Issue137Error, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        record["post_run_snapshot"] = {"status": "unavailable", "reason": type(exc).__name__}
        cleanup_failure = cleanup_failure or f"post-run snapshot failure: {type(exc).__name__}"

    try:
        result_path, result_name = _resolve_runner_result_artifact(
            resolved_output,
            before_names=before_result_names,
            completed=completed,
        )
        result = _read_runner_result(result_path)
    except Issue137Error as exc:
        result = None
        result_error = str(exc)
    else:
        result_error = None
    if result is not None:
        try:
            observed = validate_runner_result(
                result, plan=plan, configuration=configuration, mode=mode
            )
            record["runner_result"] = _safe_json_value(result)
            actual_environment = build_environment_record(
                result["environment"], configuration
            )
            if not _environment_identity_matches(
                actual_environment,
                board_id=effective_board_id,
                bus_id=effective_bus_id,
            ):
                raise Issue137Error(
                    "runner result environment identity does not match selected device"
                )
            record["environment"] = actual_environment
            record["provenance"]["runner_result_path"] = result_name
            record["provenance"]["wrapper_run_id"] = result["run_id"]
            record["run"].update(observed)
            record["run"]["workload_start_monotonic_ns"] = observed[
                "first_launch_start_monotonic_ns"
            ]
            record["run"]["workload_end_monotonic_ns"] = observed[
                "workload_end_monotonic_ns"
            ]
            record["run"]["last_launch_end_monotonic_ns"] = observed[
                "last_launch_end_monotonic_ns"
            ]
            record["workload_start_monotonic_ns"] = observed[
                "first_launch_start_monotonic_ns"
            ]
            record["workload_end_monotonic_ns"] = observed["workload_end_monotonic_ns"]
            record["run_end_monotonic_ns"] = observed["workload_end_monotonic_ns"]
            record["workload_actual_end_monotonic_ns"] = observed[
                "last_launch_end_monotonic_ns"
            ]
            record["workload_started"] = True
        except Issue137Error as exc:
            result_error = str(exc)
    source_provenance_verified = _observed_unit_provenance_verified(
        record.get("environment") if isinstance(record.get("environment"), Mapping) else environment,
        captures,
    )
    if source_provenance_verified:
        record["provenance"]["unit_provenance"] = power_trace_issue137.ISSUE137_UNIT_PROVENANCE
        record["provenance"]["unit_provenance_status"] = "established"
    else:
        record["provenance"]["unit_provenance"] = None
        record["provenance"]["unit_provenance_status"] = "unknown"
    if resolved_trace is not None:
        if result is None or result_error is not None:
            record["trace_error"] = "runner result unavailable before trace bounds"
        else:
            try:
                write_trace_from_captures(
                    resolved_trace,
                    captures,
                    board_id=effective_board_id,
                    bus_id=effective_bus_id,
                )
                trace_bytes = resolved_trace.read_bytes()
                trace_sha256 = hashlib.sha256(trace_bytes).hexdigest()
                rows = power_trace_issue137.read_issue137_power_trace(
                    resolved_trace,
                    run_start_monotonic_ns=record["workload_start_monotonic_ns"],
                    run_end_monotonic_ns=record["workload_end_monotonic_ns"],
                    source_established=source_provenance_verified,
                )
                timestamps = [row["timestamp_monotonic_ns"] for row in rows["samples"]]
                start_ns = record["workload_start_monotonic_ns"]
                target_end_ns = record["workload_end_monotonic_ns"]
                if (
                    rows.get("valid") is not True
                    or not timestamps
                    or min(timestamps) > start_ns
                    or max(timestamps) < target_end_ns
                ):
                    raise Issue137Error("Issue #137 trace lacks terminal workload coverage")
                record["trace"] = {
                    "path": resolved_trace.name,
                    "schema": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA,
                    "sample_count": len(timestamps),
                    "run_start_monotonic_ns": start_ns,
                    "run_end_monotonic_ns": target_end_ns,
                    "actual_end_monotonic_ns": record["workload_actual_end_monotonic_ns"],
                    "terminal_coverage": True,
                    "sha256": trace_sha256,
                    "byte_count": len(trace_bytes),
                    "unit_provenance": (
                        power_trace_issue137.ISSUE137_UNIT_PROVENANCE
                        if source_provenance_verified
                        else None
                    ),
                }
            except (Issue137Error, OSError, ValueError) as exc:
                record["trace_error"] = type(exc).__name__
    failure_message = result_error or cleanup_failure
    if returncode != 0:
        failure_message = failure_message or "device runner did not return status zero"
    if timed_out:
        failure_message = failure_message or "device runner reached the fixed guard"
    if failure_message:
        record = _failure_record(
            record,
            kind=("runner_result_unavailable" if result is None else "runner_execution_failure"),
            message=failure_message,
            returncode=returncode,
        )
    elif record.get("trace_error"):
        record = _failure_record(
            record, kind="trace_capture_failure", message=str(record["trace_error"])
        )
    else:
        record["status"] = "ok"
        record["verdict_evidence"] = mode == "long"
    _persist_record(resolved_output, safe_run_id, record)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--mode", choices=("plan", "smoke", "long"), default="plan")
    parser.add_argument("--configuration", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-number", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("out/issue137"))
    parser.add_argument("--preceding-pre-snapshot", type=Path, default=None)
    parser.add_argument("--board-id", default=None)
    parser.add_argument("--bus-id", default=None)
    parser.add_argument("--approve-long-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.mode == "plan":
            print(strict_json_dumps(build_planning_record(plan), indent=2))
            return 0
        if args.configuration is None or args.run_id is None:
            raise Issue137Error("measurement modes require --configuration and --run-id")
        selected = configuration(plan, args.configuration)
        preceding = None
        if args.preceding_pre_snapshot is not None:
            preceding_path = run_resident._safe_output_path(
                args.preceding_pre_snapshot,
                description="Issue #137 preceding pre-run snapshot",
            )
            payload = json.loads(preceding_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise Issue137Error("preceding pre-run snapshot must be an object")
            preceding = payload
        output_dir = _safe_output_dir(args.out_dir)
        record = run_one(
            plan=plan,
            configuration=selected,
            mode=args.mode,
            output_dir=output_dir,
            run_id=args.run_id,
            run_number=args.run_number,
            approve_long_run=args.approve_long_run,
            command_runner=(_run_issue137_workload if args.execute else None),
            capture_fn=(capture_readonly_snapshot if args.execute else None),
            preceding_pre_snapshot=preceding,
            board_id=args.board_id,
            bus_id=args.bus_id,
            trace_path=(
                output_dir / f"issue137-power-trace-{args.run_id}.csv"
                if args.execute
                else None
            ),
        )
        print(strict_json_dumps(_safe_json_value(record), indent=2))
        return 0 if record.get("status") != "failed" else 2
    except (Issue137Error, OSError, json.JSONDecodeError) as exc:
        print(f"issue137: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2


__all__ = [
    "CONTAINER_VISIBLE_DEVICE_ID",
    "COOLDOWN_CAP_SECONDS",
    "DEVICE_NODE",
    "ISSUE137_DEVICE_RESULT_SCHEMA",
    "ISSUE137_LOAD_MODE",
    "ISSUE137_RECORD_SCHEMA",
    "ISSUE137_SNAPSHOT_SCHEMA",
    "WRAPPER_CLEANUP_MARGIN_SECONDS",
    "Issue137Configuration",
    "Issue137Error",
    "build_environment_record",
    "build_planning_record",
    "build_run_command",
    "capture_readonly_snapshot",
    "configuration",
    "cooldown_gate",
    "evaluate_cooldown",
    "load_plan",
    "long_run_order",
    "main",
    "plan_configurations",
    "resolve_result_path",
    "run_one",
    "run_timing",
    "sample_issue137_trace",
    "trace_rows_from_captures",
    "validate_environment_for_run",
    "validate_runner_result",
    "wait_for_cooldown",
    "write_trace_from_captures",
]


if __name__ == "__main__":
    raise SystemExit(main())
