"""Run the Issue #100 correctness and performance catalogue in one device session."""

from __future__ import annotations

import argparse
import csv
import datetime
import inspect
import json
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.newton_schulz_kernel import run_newton_schulz_kernel
from enodia.tt.bench.newton_schulz_reference import (
    NEWTON_SCHULZ_ITERATIONS,
    bf16_round_complex,
    initial_value,
    newton_schulz_reference,
    random_hpd_batch,
)
from tools.newton_schulz_issue100_same_run import (
    ISSUE100_COMPARISON_CONFIGS,
    ISSUE100_DEVICE_ID,
    ISSUE100_LAUNCHES,
    ISSUE100_SHAPES,
    run_issue100_comparison,
)

ISSUE101_CORRECTNESS_CASES = (
    (4, 16),
    (8192, 16),
    (4, 32),
    (8192, 32),
    (1, 16),
    (3, 16),
    (5, 32),
    (31, 32),
    (63, 32),
)
ISSUE101_CORRECTNESS_THRESHOLD = 0.01
ISSUE101_CONTAINER_TIMEOUT_S = 600
ISSUE101_RUNNER = "tools/newton_schulz_issue101_combined.py"
ISSUE101_OUTPUT_NAME = "issue101-combined.json"
ISSUE101_RAW_OUTPUT_NAME = "issue101-combined-raw.json"
ISSUE101_RAW_SCHEMA = "adr-0005-issue101-combined-raw-v1"
ISSUE101_POWER_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")
ISSUE101_FAILURE_STAGES = (
    "open",
    "correctness",
    "performance",
    "telemetry",
    "close",
    "record_construction",
)
_METADATA_PRIVATE_KEYS = frozenset(
    {
        "hostname",
        "host_name",
        "host_hostname",
        "username",
        "user_name",
        "user",
        "home",
        "home_directory",
        "cwd",
        "working_directory",
        "password",
        "secret",
        "token",
        "access_token",
        "api_key",
        "private_key",
    }
)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:])/(?:home|Users|tmp|var/tmp|workspace|workspaces|mnt|opt|root|run/user)/[^\s,;\"']+"
)
ISSUE101_NEW_DEFAULT = {
    "variant": "bf16",
    "state_format": "BF16",
    "math_fidelity": "HiFi3",
    "fuse_s": True,
    "fp32_dest_acc_en": True,
    "matrix_block": 8,
    "double_buffer": True,
    "dst_full_sync_en": True,
    "input_memory": "l1",
    "r_memory": "l1",
    "x0_memory": "l1",
    "output_memory": "dram",
    "initial_value": "I/||R||inf",
    "iterations": NEWTON_SCHULZ_ITERATIONS,
}
ISSUE101_EXECUTION_CONFIG = {
    key: ISSUE101_NEW_DEFAULT[key]
    for key in (
        "variant",
        "math_fidelity",
        "fuse_s",
        "fp32_dest_acc_en",
        "matrix_block",
        "double_buffer",
        "input_memory",
        "r_memory",
        "x0_memory",
        "output_memory",
        "dst_full_sync_en",
    )
}
ISSUE101_PREVIOUS_DEFAULT = {
    "variant": "bf16-fp32state",
    "state_format": "FP32",
    "math_fidelity": "HiFi3",
    "fuse_s": True,
    "fp32_dest_acc_en": True,
    "matrix_block": 4,
    "double_buffer": True,
    "dst_full_sync_en": True,
    "input_memory": "l1",
    "r_memory": "l1",
    "x0_memory": "l1",
    "output_memory": "dram",
    "initial_value": "I/||R||inf",
    "iterations": NEWTON_SCHULZ_ITERATIONS,
}
ISSUE101_SUPERSEDED_RECORDS = (
    "docs/measurements/2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000.json",
    "docs/measurements/2026-10-05-p150a-newton-schulz-issue101-default-correctness.json",
)


def _sanitize_metadata(value: Any, *, key: str | None = None) -> Any:
    """Keep raw diagnostics useful without copying host or credential metadata."""
    if key is not None and key.lower() in _METADATA_PRIVATE_KEYS:
        return None
    if isinstance(value, dict):
        sanitized = {}
        for child_key, child_value in value.items():
            if str(child_key).lower() in _METADATA_PRIVATE_KEYS:
                continue
            sanitized[child_key] = _sanitize_metadata(
                child_value, key=str(child_key)
            )
        return sanitized
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, str):
        return _ABSOLUTE_PATH_RE.sub("<redacted-path>", value)
    return value


def _error_text(error: BaseException | str) -> str:
    """Format an exception without exposing an absolute user-specific path."""
    if isinstance(error, BaseException):
        detail = f"{type(error).__name__}: {error}"
    else:
        detail = str(error)
    return _sanitize_metadata(detail)


def _failure_details(stage: str, error: BaseException | str) -> dict[str, str]:
    """Return stable, machine-readable failure metadata for a raw artifact."""
    return {"stage": stage, "error": _error_text(error)}


def _mark_run_failed(
    run: dict,
    stage: str,
    error: BaseException | str,
    *,
    replace_existing: bool = False,
) -> dict:
    """Add a failure without discarding rows already collected by the run."""
    updated = dict(run)
    details = _failure_details(stage, error)
    existing = updated.get("failure")
    if existing and not replace_existing:
        secondary = list(updated.get("secondary_failures", []))
        secondary.append(details)
        updated["secondary_failures"] = secondary
        return updated
    if existing:
        details["prior_failure"] = existing
    updated["status"] = "failed"
    updated["failure_stage"] = stage
    updated["error"] = details["error"]
    updated["failure"] = details
    return updated


def _supports_keyword(function: Any, keyword: str) -> bool:
    """Check optional capture seams without constraining injected test runners."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _correctness_case(ttnn: Any, device: Any, batch: int, size: int) -> dict:
    """Execute one new-default correctness case without retrying failures."""
    case = f"batch{batch}-L{size}"
    matrices = random_hpd_batch(batch, size, seed=100 + batch + size)
    expected = newton_schulz_reference(
        bf16_round_complex(matrices), x0=initial_value(matrices)
    )
    try:
        actual = run_newton_schulz_kernel(
            ttnn,
            device,
            matrices,
            **ISSUE101_EXECUTION_CONFIG,
        )
        relative_error = float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        )
    except Exception as exc:  # noqa: BLE001 - the failed row is the terminal result
        row = {
            "case": case,
            "batch": batch,
            "size": size,
            "reference": "BF16-rounded-R fixed-N=12 reference",
            "threshold": ISSUE101_CORRECTNESS_THRESHOLD,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"issue101_combined_correctness {case} status=failed", flush=True)
        return row

    status = "pass" if relative_error <= ISSUE101_CORRECTNESS_THRESHOLD else "failed"
    row = {
        "case": case,
        "batch": batch,
        "size": size,
        "reference": "BF16-rounded-R fixed-N=12 reference",
        "threshold": ISSUE101_CORRECTNESS_THRESHOLD,
        "relative_error": relative_error,
        "status": status,
    }
    print(
        "issue101_combined_correctness "
        f"case={case} relative_error={relative_error:.10e} status={status}",
        flush=True,
    )
    return row


def run_issue101_correctness(
    ttnn: Any, device: Any, *, rows: list[dict] | None = None
) -> list[dict]:
    """Run the nine correctness cases on the already-open device.

    ``rows`` is an execution sink rather than a second source of truth.  It lets
    the combined driver retain completed rows if an injected or unexpected
    exception escapes between cases.
    """
    collected = rows if rows is not None else []
    for batch, size in ISSUE101_CORRECTNESS_CASES:
        row = _correctness_case(ttnn, device, batch, size)
        collected.append(row)
        if row["status"] != "pass":
            break
    return collected


def _run_correctness_with_capture(
    ttnn: Any, device: Any, rows: list[dict]
) -> list[dict]:
    runner = run_issue101_correctness
    if _supports_keyword(runner, "rows"):
        return runner(ttnn, device, rows=rows)
    return runner(ttnn, device)


def _run_performance_with_capture(
    ttnn: Any, device: Any, *, repeats: int, rows: list[dict]
) -> list[dict]:
    runner = run_issue100_comparison
    if _supports_keyword(runner, "results_sink"):
        return runner(
            ttnn,
            device,
            repeats=repeats,
            stop_on_failure=True,
            results_sink=rows,
        )
    return runner(ttnn, device, repeats=repeats, stop_on_failure=True)


def run_issue101_combined(ttnn: Any, device: Any, *, repeats: int = ISSUE100_LAUNCHES) -> dict:
    """Run correctness first, then the four performance rows on one device."""
    correctness: list[dict] = []
    try:
        correctness_result = _run_correctness_with_capture(ttnn, device, correctness)
        if correctness_result is not correctness:
            correctness = list(correctness_result)
    except BaseException as exc:  # noqa: BLE001 - preserve completed rows
        return _mark_run_failed(
            {
                "status": "failed",
                "correctness_cases": correctness,
                "performance_rows": [],
            },
            "correctness",
            exc,
        )

    if len(correctness) != len(ISSUE101_CORRECTNESS_CASES) or any(
        row["status"] != "pass" for row in correctness
    ):
        failed_row = next(
            (row for row in correctness if row.get("status") != "pass"),
            None,
        )
        return _mark_run_failed(
            {
                "status": "failed",
                "correctness_cases": correctness,
                "performance_rows": [],
            },
            "correctness",
            (failed_row or {}).get("error", "correctness did not pass"),
        )

    performance: list[dict] = []
    try:
        performance_result = _run_performance_with_capture(
            ttnn, device, repeats=repeats, rows=performance
        )
        if performance_result is not performance:
            performance = list(performance_result)
    except BaseException as exc:  # noqa: BLE001 - preserve completed rows
        partial = getattr(exc, "partial_results", None)
        if isinstance(partial, list) and partial is not performance:
            performance = list(partial)
        return _mark_run_failed(
            {
                "status": "failed",
                "correctness_cases": correctness,
                "performance_rows": performance,
            },
            "performance",
            exc,
        )

    expected_performance = len(ISSUE100_SHAPES) * len(ISSUE100_COMPARISON_CONFIGS)
    if len(performance) != expected_performance or any(
        row.get("status") != "ok" for row in performance
    ):
        failed_row = next(
            (row for row in performance if row.get("status") != "ok"),
            None,
        )
        return _mark_run_failed(
            {
                "status": "failed",
                "correctness_cases": correctness,
                "performance_rows": performance,
            },
            "performance",
            (failed_row or {}).get("error", "performance did not pass"),
        )
    return {
        "status": "pass",
        "correctness_cases": correctness,
        "performance_rows": performance,
    }


def _single_artifact(output_dir: Path, pattern: str) -> Path:
    """Find one wrapper artifact in the fresh output directory."""
    matches = sorted(output_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern} artifact in {output_dir}, found {len(matches)}"
        )
    return matches[0]


def _identity_value(value: Any) -> Any | None:
    """Treat missing and empty telemetry identity values as absent."""
    return None if value is None or value == "" else value


def resolve_board_serial_identity(board: dict[str, Any]) -> dict[str, Any]:
    """Resolve the board serial, accepting ``board_id`` as a telemetry alias.

    Some ``tt-smi`` snapshots expose only ``board_id``.  For ADR-0005 records,
    that value is the serial identity when no explicit serial is present.  An
    explicit ``serial`` (or legacy ``board_serial``) remains authoritative, but
    a simultaneous board ID must agree so a telemetry inconsistency fails fast.
    """
    explicit_serial = _identity_value(board.get("serial"))
    legacy_serial = _identity_value(board.get("board_serial"))
    board_id = _identity_value(board.get("board_id"))
    if (
        explicit_serial is not None
        and legacy_serial is not None
        and str(explicit_serial) != str(legacy_serial)
    ):
        raise ValueError("board serial and board_serial telemetry values disagree")
    serial = explicit_serial if explicit_serial is not None else legacy_serial
    if serial is not None and board_id is not None and str(serial) != str(board_id):
        raise ValueError("board serial and board_id telemetry values disagree")
    if serial is None and board_id is not None:
        # This is the only permitted board_id -> serial alias.  Keep its source
        # in the normalized environment so the record explains the identity.
        serial = board_id
        source = "board_id_alias"
    elif explicit_serial is not None:
        source = "explicit_serial"
    elif legacy_serial is not None:
        source = "board_serial_alias"
    else:
        source = "missing"
    return {
        "serial": serial,
        "board_id": board_id if board_id is not None else serial,
        "source": source,
        "alias_applied": source == "board_id_alias",
        "rule": (
            "board_id is accepted as serial identity only when telemetry has no "
            "explicit serial; an explicit serial takes precedence and must match board_id."
        ),
    }


def normalize_environment(raw: dict) -> dict:
    """Add README-authoritative aliases and validate board identity telemetry."""
    environment = dict(raw)
    image = environment.get("image")
    if isinstance(image, str) and "@" in image:
        environment["image_digest"] = image.rsplit("@", 1)[1]

    if "kernel" in environment:
        environment["host_kernel"] = environment["kernel"]
    if "kmd_version" in environment:
        environment["kernel_driver_version"] = environment["kmd_version"]
    environment.setdefault("python", platform.python_version())
    if environment.get("tt_env_active_release"):
        environment.setdefault("toolchain_release", environment["tt_env_active_release"])

    board = environment.get("board") or environment.get("board_info")
    if board is None and "board_id" in environment:
        board = {"board_id": environment["board_id"]}
    if isinstance(board, dict):
        board = dict(board)
        board.setdefault("device_id", ISSUE100_DEVICE_ID)
        identity = resolve_board_serial_identity(board)
        if identity["serial"] is not None:
            board["serial"] = identity["serial"]
            board["board_id"] = identity["board_id"]
        board["serial_identity_source"] = identity["source"]
        environment["board"] = board
        environment["board_serial_identity"] = identity

    firmware = environment.get("firmware") or environment.get("firmwares")
    if isinstance(firmware, dict):
        environment["firmware"] = dict(firmware)

    environment.setdefault("compiler", None)
    environment.setdefault("tt_metal_source_commit", None)
    environment.setdefault(
        "kernel_compiler_note",
        "The pinned release image did not expose a source compiler commit.",
    )
    environment.setdefault(
        "harness_capture_note",
        "The synchronized source tree was clean at the recorded revision; "
        "git metadata was excluded from the temporary device staging copy.",
    )
    return environment


def _environment_artifact(output_dir: Path) -> tuple[Path, dict, dict]:
    environment_path = _single_artifact(output_dir, "env-*.json")
    raw = json.loads(environment_path.read_text())
    return environment_path, raw, normalize_environment(raw)


def _environment_from_output(output_dir: Path) -> dict:
    return _environment_artifact(output_dir)[2]


def _power_trace_artifact(path: Path) -> dict:
    """Read the complete wrapper CSV without reducing its telemetry samples."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ISSUE101_POWER_COLUMNS:
            raise ValueError(
                f"power trace {path.name} has unexpected columns {reader.fieldnames!r}"
            )
        samples = [dict(row) for row in reader]
    if not samples:
        raise ValueError(f"power trace {path.name} contains no samples")
    return {
        "file": path.name,
        "columns": list(ISSUE101_POWER_COLUMNS),
        "samples": samples,
        "sampling_source": "tt-smi snapshot",
    }


def _capture_telemetry(output_dir: Path) -> dict:
    """Collect available wrapper artifacts without hiding a collection failure."""
    telemetry: dict[str, Any] = {
        "status": "failed",
        "environment_file": None,
        "environment": None,
        "normalized_environment": None,
        "power_trace": None,
        "failures": [],
    }
    try:
        environment_path, raw_environment, normalized_environment = _environment_artifact(
            output_dir
        )
    except Exception as exc:  # noqa: BLE001 - raw output must still be published
        telemetry["failures"].append(_failure_details("telemetry.environment", exc))
        # Normalization can fail after the wrapper JSON has been read.  Retain
        # that actual snapshot without inventing aliases or record fields.
        try:
            environment_path = _single_artifact(output_dir, "env-*.json")
            raw_environment = json.loads(environment_path.read_text())
        except Exception as fallback_exc:  # noqa: BLE001 - retain first failure
            telemetry["failures"].append(
                _failure_details("telemetry.environment_raw", fallback_exc)
            )
        else:
            telemetry["environment_file"] = environment_path.name
            telemetry["environment"] = raw_environment
    else:
        telemetry["environment_file"] = environment_path.name
        telemetry["environment"] = raw_environment
        telemetry["normalized_environment"] = normalized_environment

    try:
        power_trace = _power_trace_artifact(
            _single_artifact(output_dir, "power-*.csv")
        )
    except Exception as exc:  # noqa: BLE001 - preserve environment when available
        telemetry["failures"].append(_failure_details("telemetry.power", exc))
    else:
        telemetry["power_trace"] = power_trace

    if not telemetry["failures"]:
        telemetry["status"] = "complete"
    return telemetry


def _validate_complete_run(run: dict, *, repeats: int) -> None:
    """Reject a record that would silently omit a requested raw result."""
    if run.get("status") != "pass":
        raise ValueError("a combined record requires all correctness and performance rows to pass")
    correctness = run.get("correctness_cases")
    if not isinstance(correctness, list) or len(correctness) != len(ISSUE101_CORRECTNESS_CASES):
        raise ValueError("the combined record requires all nine correctness rows")
    for row, (batch, size) in zip(correctness, ISSUE101_CORRECTNESS_CASES, strict=True):
        if row.get("batch") != batch or row.get("size") != size or row.get("status") != "pass":
            raise ValueError(f"invalid correctness row for batch{batch}-L{size}")
    performance = run.get("performance_rows")
    expected_performance = len(ISSUE100_SHAPES) * len(ISSUE100_COMPARISON_CONFIGS)
    if not isinstance(performance, list) or len(performance) != expected_performance:
        raise ValueError("the combined record requires all four performance rows")
    for index, row in enumerate(performance):
        if row.get("status") != "ok":
            raise ValueError(f"performance row {index} did not pass")
        samples = row.get("seconds_per_launch_samples")
        if not isinstance(samples, list) or len(samples) != repeats:
            raise ValueError(f"performance row {index} lacks all {repeats} launch samples")
        for field in (
            "seconds_per_launch_p50",
            "seconds_per_launch_p99",
            "seconds_per_launch_p99_9",
        ):
            if field not in row:
                raise ValueError(f"performance row {index} lacks {field}")


def _record_from_parts(
    run: dict,
    *,
    environment: dict,
    power_trace: dict,
    repeats: int,
    raw_artifact_name: str | None = None,
) -> dict:
    """Build the immutable record from either live artifacts or raw recovery data."""
    _validate_complete_run(run, repeats=repeats)
    harness_commit = environment.get("harness_commit")
    if not harness_commit or environment.get("harness_dirty") is not False:
        raise ValueError("the combined record requires a clean harness commit")
    if environment.get("image_pinned") is not True:
        raise ValueError("the combined record requires a digest-pinned image")
    required_environment = (
        "image",
        "image_digest",
        "toolchain_release",
        "tt_env_active_release",
        "host_kernel",
        "kernel_driver_version",
        "kmd_version",
        "python",
        "board",
        "firmware",
    )
    missing_environment = [
        name for name in required_environment if not environment.get(name)
    ]
    if missing_environment:
        raise ValueError(
            "the combined record is missing environment fields: "
            + ", ".join(missing_environment)
        )
    board = environment["board"]
    if any(not board.get(name) for name in ("board_type", "board_id", "serial")):
        raise ValueError("the combined record requires board type, id, and serial")
    if not environment["firmware"].get("fw_bundle_version"):
        raise ValueError("the combined record requires firmware bundle version")

    record = {
        "record_schema": "adr-0005-issue101-combined-catalog-1000-v1",
        "status": "pass",
        "issue": "#100",
        "adr": "ADR-0005",
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "harness_commit": harness_commit,
        "environment": environment,
        "supersedes": {
            "records": list(ISSUE101_SUPERSEDED_RECORDS),
            "scope": (
                "This combined correctness and performance record supersedes the earlier "
                "PR #101 correctness-only and performance records for Issue #100; both "
                "predecessors remain immutable and visible."
            ),
            "reason": "All nine correctness cases and four performance rows were retaken in one device session.",
        },
        "measurement": {
            "description": (
                "Issue #100 new-default versus previous-default comparison with all nine "
                "correctness cases and four 1,000-launch performance rows in one device session."
            ),
            "device_id": ISSUE100_DEVICE_ID,
            "same_device_session": True,
            "same_python_process": True,
            "watcher": False,
            "container_timeout_s": ISSUE101_CONTAINER_TIMEOUT_S,
            "correctness_reference": "BF16-rounded-R fixed-N=12 reference",
            "correctness_threshold_relative_error": ISSUE101_CORRECTNESS_THRESHOLD,
            "new_default": dict(ISSUE101_NEW_DEFAULT),
            "previous_default": dict(ISSUE101_PREVIOUS_DEFAULT),
            "performance_launches_per_row": repeats,
            "performance_watcher": False,
            "correctness_cases": run["correctness_cases"],
            "performance_rows": run["performance_rows"],
            "power_trace": power_trace["file"],
            "power_clock_provenance": {
                "trace": power_trace["file"],
                "columns": list(ISSUE101_POWER_COLUMNS),
                "samples": len(power_trace["samples"]),
                "power_column": "power_w",
                "clock_column": "aiclk_mhz",
                "temperature_column": "asic_temp_c",
                "sampling_source": "tt-smi snapshot",
            },
            "commands": {
                "wrapper": "enodia/tt/bench/run_in_container.sh",
                "runner": ISSUE101_RUNNER,
                "device_node": "/dev/tenstorrent/0",
                "selection": {
                    "correctness_cases": [
                        f"batch{batch}-L{size}" for batch, size in ISSUE101_CORRECTNESS_CASES
                    ],
                    "performance_shapes": list(ISSUE100_SHAPES),
                    "performance_configurations": [
                        config["name"] for config in ISSUE100_COMPARISON_CONFIGS
                    ],
                },
            },
        },
        "recovery": {
            "timeout": False,
            "abnormal_exit": False,
            "reset_performed": False,
            "stage1_health_probe": "not needed after normal closure",
            "docker_ps_before_each_run": "recorded by the outer device protocol",
            "docker_ps_after_each_run": "recorded by the outer device protocol",
        },
        "power_trace": power_trace["file"],
        "notes": [
            "All nine correctness cases and four performance rows passed in one device session.",
            "Raw 1,000-launch samples and p50/p99/p99.9 values are retained per performance row.",
            "TFLOPS fields are explicitly p50-derived and fastest-launch-derived.",
            "The raw artifact retains the complete pre-builder environment and power samples.",
            "No host name or user-specific absolute path is included in this record.",
        ],
    }
    if raw_artifact_name is not None:
        record["raw_artifact"] = {
            "schema": ISSUE101_RAW_SCHEMA,
            "file": raw_artifact_name,
            "external_temporary": True,
            "not_committed": True,
        }
    return record


def build_combined_record(
    run: dict,
    *,
    output_dir: Path,
    repeats: int = ISSUE100_LAUNCHES,
    raw_artifact_path: Path | None = None,
) -> dict:
    """Build one ADR-0005 record after all rows have passed."""
    _, _, environment = _environment_artifact(output_dir)
    power_trace = _power_trace_artifact(_single_artifact(output_dir, "power-*.csv"))
    return _record_from_parts(
        run,
        environment=environment,
        power_trace=power_trace,
        repeats=repeats,
        raw_artifact_name=raw_artifact_path.name if raw_artifact_path else None,
    )


def _atomic_json_write(path: Path, payload: dict) -> None:
    """Write JSON beside the destination and publish it with one atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            handle.write(strict_json_dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _raw_artifact_payload(
    run: dict,
    *,
    output_dir: Path,
    repeats: int,
    telemetry: dict | None = None,
    artifact_failure: dict | None = None,
    recovery_run: dict | None = None,
) -> dict:
    """Capture every builder input before any record construction is attempted."""
    captured_telemetry = telemetry if telemetry is not None else _capture_telemetry(output_dir)
    safe_run = _sanitize_metadata(run)
    payload = {
        "raw_schema": ISSUE101_RAW_SCHEMA,
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "artifact_status": "failed" if artifact_failure else run.get("status"),
        "run": safe_run,
        "correctness_results": safe_run.get("correctness_cases", []),
        "performance_results": safe_run.get("performance_rows", []),
        "configs": {
            "correctness_cases": [
                {"batch": batch, "size": size}
                for batch, size in ISSUE101_CORRECTNESS_CASES
            ],
            "new_default": dict(ISSUE101_NEW_DEFAULT),
            "previous_default": dict(ISSUE101_PREVIOUS_DEFAULT),
            "execution": dict(ISSUE101_EXECUTION_CONFIG),
            "comparison": [dict(config) for config in ISSUE100_COMPARISON_CONFIGS],
            "shapes": list(ISSUE100_SHAPES),
            "launches_per_row": repeats,
        },
        "telemetry": _sanitize_metadata(captured_telemetry),
    }
    if artifact_failure is not None:
        payload["failure"] = _sanitize_metadata(artifact_failure)
    elif run.get("failure") is not None:
        payload["failure"] = _sanitize_metadata(run["failure"])
    if run.get("secondary_failures"):
        payload["secondary_failures"] = _sanitize_metadata(run["secondary_failures"])
    if recovery_run is not None:
        payload["recovery_run"] = _sanitize_metadata(recovery_run)
    return payload


def write_raw_artifact(
    run: dict,
    *,
    output_dir: Path,
    repeats: int = ISSUE100_LAUNCHES,
    telemetry: dict | None = None,
    artifact_failure: dict | None = None,
    recovery_run: dict | None = None,
) -> Path:
    """Persist the complete device-session result before invoking the builder."""
    artifact_path = output_dir / ISSUE101_RAW_OUTPUT_NAME
    _atomic_json_write(
        artifact_path,
        _raw_artifact_payload(
            run,
            output_dir=output_dir,
            repeats=repeats,
            telemetry=telemetry,
            artifact_failure=artifact_failure,
            recovery_run=recovery_run,
        ),
    )
    return artifact_path


def persist_raw_and_build(
    run: dict, *, output_dir: Path, repeats: int = ISSUE100_LAUNCHES
) -> tuple[Path, dict | None]:
    """Write raw data first, then build; preserve the raw path on builder failure."""
    telemetry = _capture_telemetry(output_dir)
    prepared_run = run
    if prepared_run.get("status") == "failed" and not prepared_run.get("failure"):
        prepared_run = dict(prepared_run)
        stage = prepared_run.get("failure_stage", "unknown")
        prepared_run["failure"] = _failure_details(
            stage, prepared_run.get("error", "run failed")
        )
    if telemetry["status"] != "complete":
        telemetry_error = telemetry["failures"][0]["error"]
        prepared_run = _mark_run_failed(prepared_run, "telemetry", telemetry_error)
    raw_path = write_raw_artifact(
        prepared_run,
        output_dir=output_dir,
        repeats=repeats,
        telemetry=telemetry,
    )
    if prepared_run.get("status") != "pass":
        return raw_path, None
    try:
        record = build_combined_record(
            prepared_run,
            output_dir=output_dir,
            repeats=repeats,
            raw_artifact_path=raw_path,
        )
    except Exception as exc:
        failure = _failure_details("record_construction", exc)
        failed_run = _mark_run_failed(
            prepared_run, "record_construction", exc
        )
        try:
            # Keep a clean copy for host recovery while marking the run itself
            # failed so the artifact explains why no record was published.
            write_raw_artifact(
                failed_run,
                output_dir=output_dir,
                repeats=repeats,
                telemetry=telemetry,
                artifact_failure=failure,
                recovery_run=prepared_run,
            )
        except Exception as persist_exc:  # noqa: BLE001 - preserve builder error
            raise RuntimeError(
                "combined record builder failed after raw artifact "
                f"{raw_path}: {_error_text(exc)}; failed to update failure metadata: "
                f"{_error_text(persist_exc)}"
            ) from exc
        raise RuntimeError(
            f"combined record builder failed after raw artifact {raw_path}: "
            f"{_error_text(exc)}"
        ) from exc
    return raw_path, record


def recover_combined_record(raw_path: Path) -> dict:
    """Rebuild a complete record from a pre-builder raw JSON artifact on the host."""
    payload = json.loads(raw_path.read_text())
    if payload.get("raw_schema") != ISSUE101_RAW_SCHEMA:
        raise ValueError(f"unsupported raw artifact schema in {raw_path.name}")
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, dict):
        raise TypeError("raw artifact is missing telemetry")
    raw_environment = telemetry.get("environment")
    power_trace = telemetry.get("power_trace")
    if not isinstance(raw_environment, dict) or not isinstance(power_trace, dict):
        raise TypeError("raw artifact is missing environment or power data")
    environment = normalize_environment(raw_environment)
    recovery_run = payload.get("recovery_run", payload["run"])
    if recovery_run.get("status") != "pass":
        stage = recovery_run.get("failure_stage", "unknown")
        raise ValueError(
            f"raw artifact failed at stage {stage}; no complete record is recoverable"
        )
    return _record_from_parts(
        recovery_run,
        environment=environment,
        power_trace=power_trace,
        repeats=int(payload["configs"]["launches_per_row"]),
        raw_artifact_name=raw_path.name,
    )


def _initial_failed_run(stage: str, error: BaseException | str) -> dict:
    return _mark_run_failed(
        {
            "status": "failed",
            "correctness_cases": [],
            "performance_rows": [],
        },
        stage,
        error,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the device session, or recover its record without importing ttnn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.recover_raw is not None:
        if args.output is None:
            parser.error("--output is required with --recover-raw")
        record = recover_combined_record(args.recover_raw)
        _atomic_json_write(args.output, record)
        print(f"recovered combined record -> {args.output}", flush=True)
        return 0

    output_dir = Path(os.environ.get("HEKATUS_TT_OUTPUT_DIR", "/out"))
    try:
        import ttnn
    except Exception as exc:  # noqa: BLE001 - publish an open-stage artifact
        run = _initial_failed_run("open", exc)
        run["cleanup"] = {
            "device_opened": False,
            "close_attempted": False,
            "close_succeeded": False,
        }
        raw_path, _ = persist_raw_and_build(
            run, output_dir=output_dir, repeats=ISSUE100_LAUNCHES
        )
        print(
            "issue101_combined status=failed "
            f"stage=open raw_artifact={raw_path}",
            flush=True,
        )
        return 1

    run: dict
    close_error: BaseException | None = None
    device_opened = False
    try:
        device = ttnn.open_device(device_id=ISSUE100_DEVICE_ID)
        device_opened = True
    except BaseException as exc:  # noqa: BLE001 - persist open failures
        run = _initial_failed_run("open", exc)
    else:
        try:
            run = run_issue101_combined(
                ttnn, device, repeats=ISSUE100_LAUNCHES
            )
        except BaseException as exc:  # noqa: BLE001 - preserve the session result
            run = _initial_failed_run("device_session", exc)
        finally:
            cleanup = dict(run.get("cleanup", {})) if "run" in locals() else {}
            cleanup.update(
                {
                    "device_opened": True,
                    "close_attempted": True,
                }
            )
            try:
                ttnn.close_device(device)
            except BaseException as exc:  # noqa: BLE001 - persist before re-raise
                close_error = exc
                run = _mark_run_failed(
                    run,
                    "close",
                    exc,
                    replace_existing=True,
                )
                cleanup.update(
                    {
                        "close_succeeded": False,
                        "close_error": _error_text(exc),
                    }
                )
            else:
                cleanup["close_succeeded"] = True
            run["cleanup"] = cleanup

    if not device_opened:
        run["cleanup"] = {
            "device_opened": False,
            "close_attempted": False,
            "close_succeeded": False,
        }

    raw_path, record = persist_raw_and_build(
        run, output_dir=output_dir, repeats=ISSUE100_LAUNCHES
    )
    if close_error is not None:
        raise close_error
    if record is None:
        print(
            "issue101_combined status=failed "
            f"stage={run.get('failure_stage', 'unknown')} raw_artifact={raw_path}",
            flush=True,
        )
        return 1

    output_path = output_dir / ISSUE101_OUTPUT_NAME
    _atomic_json_write(output_path, record)
    print(f"combined record -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
