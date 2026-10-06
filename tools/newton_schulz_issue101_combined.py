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
import uuid
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
# Keep the unsuffixed name for the first direct, one-run invocation.  Device
# wrapper runs always provide HEKATUS_TT_RUN_ID and therefore use the suffixed
# form so a reused output directory cannot overwrite an earlier artifact.
ISSUE101_RAW_OUTPUT_NAME = "issue101-combined-raw.json"
ISSUE101_RAW_OUTPUT_PREFIX = "issue101-combined-raw-"
ISSUE101_RAW_OUTPUT_GLOB = "issue101-combined-raw*.json"
ISSUE101_RUN_ID_ENV = "HEKATUS_TT_RUN_ID"
ISSUE101_RAW_SCHEMA = "adr-0005-issue101-combined-raw-v1"
ISSUE101_POWER_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
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
_CAPTURED_ENVIRONMENT_REQUIRED_FIELDS = (
    "captured_at",
    "image",
    "image_digest",
    "image_pinned",
    "kernel",
    "host_kernel",
    "kmd_version",
    "kernel_driver_version",
    "tt_env_active_release",
    "toolchain_release",
    "python",
    "harness_commit",
    "harness_dirty",
    "board",
    "firmware",
    "board_serial_identity",
    "run_id",
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


def _new_run_id() -> str:
    """Create the identity shared by one device session and its artifacts."""
    return uuid.uuid4().hex


def _validate_run_id(run_id: Any, *, source: str = "run_id") -> str:
    """Reject an absent or path-like run identity before it reaches a filename."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"{source} must be a non-empty safe artifact identifier")
    return run_id


def _filename_run_id(path: Path, artifact_kind: str) -> str | None:
    """Read a run identity from an artifact basename, if its form carries one."""
    prefixes = {
        "environment": ("env-", ".json"),
        "power": ("power-", ".csv"),
        "raw": (ISSUE101_RAW_OUTPUT_PREFIX, ".json"),
    }
    try:
        prefix, suffix = prefixes[artifact_kind]
    except KeyError as exc:  # pragma: no cover - an internal caller bug
        raise ValueError(f"unknown artifact kind {artifact_kind!r}") from exc
    name = path.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    value = name[len(prefix) : -len(suffix)]
    if not value:
        return None
    return _validate_run_id(value, source=f"{artifact_kind} artifact filename")


def _json_artifact_run_id(path: Path, artifact_kind: str) -> str | None:
    """Read an embedded identity without turning a malformed artifact into a match."""
    if artifact_kind == "power":
        return None
    try:
        value = json.loads(path.read_text()).get("run_id")
    except (OSError, TypeError, ValueError, AttributeError):
        return None
    if value is None:
        return None
    return _validate_run_id(value, source=f"{artifact_kind} artifact run_id")


def _artifact_run_ids(path: Path, artifact_kind: str) -> set[str]:
    """Return filename and embedded identities, rejecting disagreement."""
    identities = {
        identity
        for identity in (
            _filename_run_id(path, artifact_kind),
            _json_artifact_run_id(path, artifact_kind),
        )
        if identity is not None
    }
    if len(identities) > 1:
        raise ValueError(
            f"{artifact_kind} artifact {path.name} has conflicting run_id values"
        )
    return identities


def _single_artifact(
    output_dir: Path,
    pattern: str,
    *,
    run_id: str | None = None,
    artifact_kind: str | None = None,
) -> Path:
    """Select one artifact, never guessing among runs in a reused directory."""
    matches = sorted(output_dir.glob(pattern))
    if artifact_kind is None:
        artifact_kind = {
            "env-*.json": "environment",
            "power-*.csv": "power",
            ISSUE101_RAW_OUTPUT_GLOB: "raw",
        }.get(pattern)
    if run_id is not None:
        run_id = _validate_run_id(run_id)
    if len(matches) == 0:
        if run_id is None:
            raise RuntimeError(
                f"no {pattern} artifact in {output_dir}; an explicit run_id cannot be inferred"
            )
        raise RuntimeError(
            f"no {pattern} artifact in {output_dir} matches explicit run_id {run_id!r}"
        )
    if run_id is None:
        if len(matches) != 1:
            raise RuntimeError(
                f"ambiguous {pattern} artifacts in {output_dir}: found {len(matches)}; "
                "explicit run_id is required"
            )
        return matches[0]

    selected = []
    for path in matches:
        if artifact_kind is None:
            # A caller that does not describe the artifact can only use the
            # filename token.  All production callers provide a kind.
            identities = {
                identity
                for identity in (_filename_run_id(path, "raw"),)
                if identity is not None
            }
        else:
            identities = _artifact_run_ids(path, artifact_kind)
        if run_id in identities:
            selected.append(path)
    if len(selected) != 1:
        detail = "none" if not selected else str(len(selected))
        raise RuntimeError(
            f"expected exactly one {pattern} artifact for explicit run_id {run_id!r} "
            f"in {output_dir}, found {detail}; refusing ambiguous association"
        )
    return selected[0]


def _infer_output_run_id(output_dir: Path) -> str | None:
    """Infer the sole legacy identity, or fail before mixed artifacts are used."""
    identities: set[str] = set()
    for pattern, kind in (
        ("env-*.json", "environment"),
        ("power-*.csv", "power"),
    ):
        for path in sorted(output_dir.glob(pattern)):
            identities.update(_artifact_run_ids(path, kind))
    if len(identities) > 1:
        raise RuntimeError(
            f"artifacts in {output_dir} belong to multiple run_id values "
            f"({', '.join(sorted(identities))}); explicit run_id is required"
        )
    return next(iter(identities), None)


def _session_run_id(run: dict, output_dir: Path) -> str:
    """Resolve the session identity before selecting environment or power data."""
    supplied = []
    if run.get("run_id") is not None:
        supplied.append(_validate_run_id(run["run_id"], source="run.run_id"))
    environment_run_id = os.environ.get(ISSUE101_RUN_ID_ENV)
    if environment_run_id:
        supplied.append(
            _validate_run_id(environment_run_id, source=ISSUE101_RUN_ID_ENV)
        )
    if len(set(supplied)) > 1:
        raise ValueError("run identity sources disagree")
    if supplied:
        return supplied[0]
    inferred = _infer_output_run_id(output_dir)
    return inferred or _new_run_id()


def _set_run_id(run: dict, run_id: str) -> dict:
    """Return a run copy carrying the identity without changing injected input."""
    current = run.get("run_id")
    if current is not None and _validate_run_id(current, source="run.run_id") != run_id:
        raise ValueError("run.run_id does not match the session run_id")
    if current == run_id:
        return run
    updated = dict(run)
    updated["run_id"] = run_id
    return updated


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


def _captured_environment_snapshot(
    raw: dict, normalized: dict, *, run_id: str | None
) -> dict:
    """Complete a live snapshot once; recovery never calls this helper."""
    environment = _sanitize_metadata(normalized)
    captured_at = environment.get("captured_at") or raw.get("captured_at")
    if captured_at is None:
        # Legacy direct callers did not put the capture timestamp in their
        # fixture.  A real wrapper capture always does, and this fallback is
        # deliberately confined to the live collection path.
        captured_at = datetime.datetime.now(datetime.UTC).isoformat()
    environment["captured_at"] = captured_at
    captured_run_id = raw.get("run_id") or environment.get("run_id")
    if captured_run_id is not None:
        captured_run_id = _validate_run_id(
            captured_run_id, source="captured environment run_id"
        )
    if run_id is not None and captured_run_id is not None and captured_run_id != run_id:
        raise ValueError(
            f"environment artifact run_id {captured_run_id!r} does not match "
            f"requested run_id {run_id!r}"
        )
    if run_id is not None:
        environment["run_id"] = run_id
    elif captured_run_id is not None:
        environment["run_id"] = captured_run_id
    return environment


def _environment_artifact(
    output_dir: Path, *, run_id: str | None = None
) -> tuple[Path, dict, dict]:
    """Read and normalize a live environment artifact for one session."""
    environment_path = _single_artifact(
        output_dir,
        "env-*.json",
        run_id=run_id,
        artifact_kind="environment",
    )
    raw = json.loads(environment_path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"environment artifact {environment_path.name} is not an object")
    identities = _artifact_run_ids(environment_path, "environment")
    artifact_run_id = next(iter(identities), None)
    if run_id is not None and artifact_run_id not in (None, run_id):
        raise ValueError(
            f"environment artifact {environment_path.name} is not for run_id {run_id!r}"
        )
    normalized = normalize_environment(raw)
    normalized = _captured_environment_snapshot(
        raw, normalized, run_id=run_id or artifact_run_id
    )
    return environment_path, raw, normalized


def _environment_from_output(output_dir: Path, *, run_id: str | None = None) -> dict:
    return _environment_artifact(output_dir, run_id=run_id)[2]


def _power_trace_artifact(path: Path, *, run_id: str | None = None) -> dict:
    """Read a complete wrapper CSV and bind it to the selected session."""
    identities = _artifact_run_ids(path, "power")
    artifact_run_id = next(iter(identities), None)
    if run_id is not None:
        run_id = _validate_run_id(run_id)
        if artifact_run_id is not None and artifact_run_id != run_id:
            raise ValueError(
                f"power trace {path.name} is for run_id {artifact_run_id!r}, "
                f"not {run_id!r}"
            )
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
        "run_id": run_id or artifact_run_id,
        "columns": list(ISSUE101_POWER_COLUMNS),
        "samples": samples,
        "sampling_source": "tt-smi snapshot",
    }


def _capture_telemetry(output_dir: Path, *, run_id: str | None = None) -> dict:
    """Collect only artifacts associated with the selected session."""
    telemetry: dict[str, Any] = {
        "status": "failed",
        "run_id": run_id,
        "environment_file": None,
        "environment": None,
        "normalized_environment": None,
        "power_trace": None,
        "failures": [],
    }
    try:
        environment_loader = _environment_artifact
        if run_id is not None and _supports_keyword(environment_loader, "run_id"):
            environment_path, raw_environment, normalized_environment = environment_loader(
                output_dir, run_id=run_id
            )
        else:
            # Keep injected one-argument seams working while making production
            # selection explicit whenever the real helper is used.
            environment_path, raw_environment, normalized_environment = environment_loader(
                output_dir
            )
    except Exception as exc:  # noqa: BLE001 - raw output must still be published
        telemetry["failures"].append(_failure_details("telemetry.environment", exc))
        # Normalization can fail after the wrapper JSON has been read.  Retain
        # that actual snapshot without inventing aliases or record fields.
        try:
            environment_path = _single_artifact(
                output_dir,
                "env-*.json",
                run_id=run_id,
                artifact_kind="environment",
            )
            raw_environment = json.loads(environment_path.read_text())
        except Exception as fallback_exc:  # noqa: BLE001 - retain first failure
            telemetry["failures"].append(
                _failure_details("telemetry.environment_raw", fallback_exc)
            )
        else:
            telemetry["environment_file"] = environment_path.name
            telemetry["environment"] = raw_environment
    else:
        captured_environment = dict(raw_environment)
        captured_environment.setdefault("run_id", normalized_environment.get("run_id"))
        telemetry["environment_file"] = environment_path.name
        telemetry["environment"] = captured_environment
        telemetry["normalized_environment"] = normalized_environment
        telemetry["run_id"] = normalized_environment.get("run_id", run_id)

    try:
        power_path = _single_artifact(
            output_dir,
            "power-*.csv",
            run_id=run_id or telemetry.get("run_id"),
            artifact_kind="power",
        )
        power_trace = _power_trace_artifact(
            power_path, run_id=run_id or telemetry.get("run_id")
        )
    except Exception as exc:  # noqa: BLE001 - preserve environment when available
        telemetry["failures"].append(_failure_details("telemetry.power", exc))
    else:
        if telemetry.get("run_id") is None:
            telemetry["run_id"] = power_trace.get("run_id")
        if power_trace.get("run_id") is None:
            power_trace["run_id"] = telemetry.get("run_id")
        if (
            telemetry.get("run_id") is not None
            and power_trace.get("run_id") is not None
            and telemetry["run_id"] != power_trace["run_id"]
        ):
            telemetry["failures"].append(
                _failure_details(
                    "telemetry.association",
                    "environment and power artifacts have different run_id values",
                )
            )
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


def _validate_captured_environment(
    environment: dict, *, run_id: str | None = None
) -> str:
    """Validate an already-captured environment without deriving any field."""
    if not isinstance(environment, dict):
        raise TypeError("captured normalized_environment must be an object")
    missing = [
        field
        for field in _CAPTURED_ENVIRONMENT_REQUIRED_FIELDS
        if field not in environment
    ]
    if missing:
        raise ValueError(
            "captured normalized_environment is missing fields: " + ", ".join(missing)
        )
    if _ABSOLUTE_PATH_RE.search(json.dumps(environment, sort_keys=True)):
        raise ValueError("captured normalized_environment contains an unsanitized host path")
    for field in (
        "captured_at",
        "image",
        "image_digest",
        "kernel",
        "host_kernel",
        "kmd_version",
        "kernel_driver_version",
        "tt_env_active_release",
        "toolchain_release",
        "python",
        "harness_commit",
    ):
        if not isinstance(environment[field], str) or not environment[field]:
            raise ValueError(
                f"captured normalized_environment field {field!r} must be a non-empty string"
            )
    if environment["image_pinned"] is not True:
        raise ValueError("the combined record requires a digest-pinned captured image")
    if environment["harness_dirty"] is not False:
        raise ValueError("the combined record requires a clean captured harness")
    captured_run_id = _validate_run_id(
        environment["run_id"], source="captured normalized_environment run_id"
    )
    if run_id is not None and captured_run_id != _validate_run_id(run_id):
        raise ValueError(
            f"captured normalized_environment run_id {captured_run_id!r} does not match "
            f"run_id {run_id!r}"
        )
    board = environment["board"]
    if not isinstance(board, dict) or any(
        not board.get(name) for name in ("board_type", "board_id", "serial")
    ):
        raise ValueError("the combined record requires captured board type, id, and serial")
    identity = environment["board_serial_identity"]
    if not isinstance(identity, dict):
        raise TypeError("captured board serial identity is missing")
    if identity.get("serial") != board["serial"] or identity.get("board_id") != board["board_id"]:
        raise ValueError("captured board serial identity does not match captured board")
    firmware = environment["firmware"]
    if not isinstance(firmware, dict) or not firmware.get("fw_bundle_version"):
        raise ValueError("the combined record requires captured firmware bundle version")
    return captured_run_id


def _validate_captured_power_trace(
    power_trace: dict, *, run_id: str
) -> None:
    """Validate power provenance before copying the captured metadata to a record."""
    if not isinstance(power_trace, dict):
        raise TypeError("captured power trace must be an object")
    filename = power_trace.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\\\" in filename
    ):
        raise ValueError("captured power trace file must be a sanitized basename")
    trace_run_id = power_trace.get("run_id")
    if trace_run_id is None:
        raise ValueError("captured power trace is missing run_id")
    if _validate_run_id(trace_run_id, source="power trace run_id") != run_id:
        raise ValueError("captured power trace is associated with a different run_id")
    samples = power_trace.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("captured power trace has no samples")


def _record_from_parts(
    run: dict,
    *,
    environment: dict,
    power_trace: dict,
    repeats: int,
    raw_artifact_name: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Build a record only from captured provenance and completed run rows."""
    _validate_complete_run(run, repeats=repeats)
    captured_run_id = _validate_captured_environment(environment, run_id=run_id)
    run_id = captured_run_id
    if run.get("run_id") is not None and _validate_run_id(run["run_id"], source="run.run_id") != run_id:
        raise ValueError("run data is associated with a different run_id")
    _validate_captured_power_trace(power_trace, run_id=run_id)
    harness_commit = environment["harness_commit"]

    record = {
        "record_schema": "adr-0005-issue101-combined-catalog-1000-v1",
        "status": "pass",
        "issue": "#100",
        "adr": "ADR-0005",
        "captured_at": environment["captured_at"],
        "run_id": run_id,
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
            "run_id": run_id,
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
        if Path(raw_artifact_name).name != raw_artifact_name or "/" in raw_artifact_name or "\\\\" in raw_artifact_name:
            raise ValueError("raw artifact file must be a sanitized basename")
        record["raw_artifact"] = {
            "schema": ISSUE101_RAW_SCHEMA,
            "file": raw_artifact_name,
            "run_id": run_id,
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
    run_id: str | None = None,
) -> dict:
    """Build one record after selecting environment and power for one run_id."""
    selected_run_id = _validate_run_id(run_id, source="run_id") if run_id else _session_run_id(run, output_dir)
    _, _, environment = _environment_artifact(output_dir, run_id=selected_run_id)
    power_path = _single_artifact(
        output_dir,
        "power-*.csv",
        run_id=selected_run_id,
        artifact_kind="power",
    )
    power_trace = _power_trace_artifact(power_path, run_id=selected_run_id)
    return _record_from_parts(
        run,
        environment=environment,
        power_trace=power_trace,
        repeats=repeats,
        raw_artifact_name=raw_artifact_path.name if raw_artifact_path else None,
        run_id=selected_run_id,
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


def _telemetry_for_raw(telemetry: dict, *, run_id: str) -> dict:
    """Bind supplied telemetry to a run without recalculating its contents."""
    captured = dict(telemetry)
    telemetry_run_id = captured.get("run_id")
    if telemetry_run_id is not None and _validate_run_id(
        telemetry_run_id, source="telemetry run_id"
    ) != run_id:
        raise ValueError("telemetry is associated with a different run_id")
    captured["run_id"] = run_id
    for field in ("environment", "normalized_environment"):
        value = captured.get(field)
        if isinstance(value, dict):
            value = dict(value)
            value_run_id = value.get("run_id")
            if value_run_id is not None and _validate_run_id(
                value_run_id, source=f"telemetry {field} run_id"
            ) != run_id:
                raise ValueError(f"telemetry {field} is associated with a different run_id")
            value["run_id"] = run_id
            captured[field] = value
    power_trace = captured.get("power_trace")
    if isinstance(power_trace, dict):
        power_trace = dict(power_trace)
        power_run_id = power_trace.get("run_id")
        if power_run_id is not None and _validate_run_id(
            power_run_id, source="telemetry power trace run_id"
        ) != run_id:
            raise ValueError("telemetry power trace is associated with a different run_id")
        power_trace["run_id"] = run_id
        captured["power_trace"] = power_trace
    return captured


def _raw_artifact_payload(
    run: dict,
    *,
    output_dir: Path,
    repeats: int,
    run_id: str,
    artifact_file: str,
    telemetry: dict | None = None,
    artifact_failure: dict | None = None,
    recovery_run: dict | None = None,
) -> dict:
    """Capture every builder input before any record construction is attempted."""
    captured_telemetry = telemetry if telemetry is not None else _capture_telemetry(
        output_dir, run_id=run_id
    )
    captured_telemetry = _telemetry_for_raw(captured_telemetry, run_id=run_id)
    safe_run = _sanitize_metadata(_set_run_id(run, run_id))
    payload = {
        "raw_schema": ISSUE101_RAW_SCHEMA,
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "run_id": run_id,
        "artifact_file": artifact_file,
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
        payload["recovery_run"] = _sanitize_metadata(
            _set_run_id(recovery_run, run_id)
        )
    return payload


def _raw_path_for_run(
    output_dir: Path,
    run_id: str,
    *,
    legacy_name: bool = False,
) -> Path:
    """Choose a non-overwriting raw basename for the selected run."""
    if legacy_name and not (output_dir / ISSUE101_RAW_OUTPUT_NAME).exists():
        return output_dir / ISSUE101_RAW_OUTPUT_NAME
    return output_dir / f"{ISSUE101_RAW_OUTPUT_PREFIX}{run_id}.json"


def write_raw_artifact(
    run: dict,
    *,
    output_dir: Path,
    repeats: int = ISSUE100_LAUNCHES,
    telemetry: dict | None = None,
    artifact_failure: dict | None = None,
    recovery_run: dict | None = None,
    run_id: str | None = None,
    raw_artifact_path: Path | None = None,
    legacy_name: bool = False,
    overwrite: bool = False,
) -> Path:
    """Persist one uniquely identified session before invoking the builder."""
    selected_run_id = _validate_run_id(run_id, source="run_id") if run_id else _session_run_id(run, output_dir)
    prepared_run = _set_run_id(run, selected_run_id)
    artifact_path = raw_artifact_path or _raw_path_for_run(
        output_dir, selected_run_id, legacy_name=legacy_name
    )
    if artifact_path.parent != output_dir:
        raise ValueError("raw artifact must be written directly in the output directory")
    if not overwrite:
        for existing in sorted(output_dir.glob(ISSUE101_RAW_OUTPUT_GLOB)):
            try:
                existing_payload = json.loads(existing.read_text())
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(existing_payload, dict) and existing_payload.get("run_id") == selected_run_id:
                raise RuntimeError(
                    f"run_id {selected_run_id!r} already has raw artifact {existing.name}; "
                    "a new session requires a unique run_id"
                )
    if artifact_path.exists() and not overwrite:
        raise RuntimeError(
            f"raw artifact {artifact_path.name} already exists; run_id must be unique"
        )
    _atomic_json_write(
        artifact_path,
        _raw_artifact_payload(
            prepared_run,
            output_dir=output_dir,
            repeats=repeats,
            run_id=selected_run_id,
            artifact_file=artifact_path.name,
            telemetry=telemetry,
            artifact_failure=artifact_failure,
            recovery_run=recovery_run,
        ),
    )
    return artifact_path


def persist_raw_and_build(
    run: dict, *, output_dir: Path, repeats: int = ISSUE100_LAUNCHES
) -> tuple[Path, dict | None]:
    """Write one run's raw data first, then build without mixing directory entries."""
    requested_run_id = run.get("run_id") or os.environ.get(ISSUE101_RUN_ID_ENV)
    run_id = _session_run_id(run, output_dir)
    telemetry = _capture_telemetry(output_dir, run_id=run_id)
    prepared_run = _set_run_id(run, run_id)
    if prepared_run.get("status") == "failed" and not prepared_run.get("failure"):
        prepared_run = dict(prepared_run)
        stage = prepared_run.get("failure_stage", "unknown")
        prepared_run["failure"] = _failure_details(
            stage, prepared_run.get("error", "run failed")
        )
    if telemetry["status"] != "complete":
        failures = telemetry.get("failures") or [{"error": "telemetry collection failed"}]
        telemetry_error = failures[0]["error"]
        prepared_run = _mark_run_failed(prepared_run, "telemetry", telemetry_error)
    raw_path = write_raw_artifact(
        prepared_run,
        output_dir=output_dir,
        repeats=repeats,
        telemetry=telemetry,
        run_id=run_id,
        legacy_name=requested_run_id is None,
    )
    if prepared_run.get("status") != "pass":
        return raw_path, None
    try:
        record = build_combined_record(
            prepared_run,
            output_dir=output_dir,
            repeats=repeats,
            raw_artifact_path=raw_path,
            run_id=run_id,
        )
    except Exception as exc:
        failure = _failure_details("record_construction", exc)
        failed_run = _mark_run_failed(
            prepared_run, "record_construction", exc
        )
        try:
            # Keep the same raw basename so host recovery sees one coherent
            # artifact whose run_id and failure metadata agree.
            write_raw_artifact(
                failed_run,
                output_dir=output_dir,
                repeats=repeats,
                telemetry=telemetry,
                artifact_failure=failure,
                recovery_run=prepared_run,
                run_id=run_id,
                raw_artifact_path=raw_path,
                overwrite=True,
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


def _load_selected_raw_artifact(
    source: Path, *, run_id: str | None = None
) -> tuple[Path, dict]:
    """Select a raw artifact by its embedded identity, not directory order."""
    source = Path(source)
    requested = _validate_run_id(run_id, source="run_id") if run_id else None
    if source.is_dir():
        candidates = sorted(source.glob(ISSUE101_RAW_OUTPUT_GLOB))
        if not candidates:
            raise RuntimeError(
                f"no raw artifacts in {source}; an explicit run_id cannot be resolved"
            )
        if requested is None and len(candidates) != 1:
            raise RuntimeError(
                f"ambiguous raw artifacts in {source}: found {len(candidates)}; "
                "explicit run_id is required"
            )
        selected: list[tuple[Path, dict]] = []
        for candidate in candidates:
            try:
                candidate_payload = json.loads(candidate.read_text())
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"cannot validate raw artifact {candidate.name}: {exc}"
                ) from exc
            if not isinstance(candidate_payload, dict):
                raise TypeError(f"raw artifact {candidate.name} is not an object")
            candidate_id = candidate_payload.get("run_id")
            if candidate_id is None:
                raise ValueError(f"raw artifact {candidate.name} has no run_id")
            try:
                candidate_id = _validate_run_id(
                    candidate_id, source=f"raw artifact {candidate.name} run_id"
                )
            except ValueError:
                if requested is None:
                    raise
                continue
            if requested is None or candidate_id == requested:
                selected.append((candidate, candidate_payload))
        if len(selected) != 1:
            match_text = "none" if not selected else str(len(selected))
            identity_text = requested or "<missing>"
            raise RuntimeError(
                f"expected exactly one raw artifact for explicit run_id {identity_text!r} "
                f"in {source}, found {match_text}; refusing ambiguous association"
            )
        return selected[0]
    if not source.is_file():
        raise FileNotFoundError(f"raw artifact does not exist: {source}")
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"raw artifact {source.name} is not an object")
    payload_id = payload.get("run_id")
    if payload_id is None:
        if requested is not None:
            raise ValueError(f"raw artifact {source.name} has no run_id")
    else:
        payload_id = _validate_run_id(payload_id, source="raw artifact run_id")
        if requested is not None and payload_id != requested:
            raise ValueError(
                f"raw artifact {source.name} has run_id {payload_id!r}, "
                f"not requested run_id {requested!r}"
            )
    return source, payload


def _validate_sanitized_basename(value: Any, *, field: str) -> str:
    """Ensure provenance stores a basename rather than a recovery-host path."""
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or "/" in value
        or "\\\\" in value
    ):
        raise ValueError(f"{field} must be a sanitized artifact basename")
    return value


def _validate_raw_payload(
    raw_path: Path, payload: dict, *, run_id: str | None = None
) -> tuple[str, dict, dict, dict, int, dict]:
    """Validate all captured associations before recovery can build a record."""
    if payload.get("raw_schema") != ISSUE101_RAW_SCHEMA:
        raise ValueError(f"unsupported raw artifact schema in {raw_path.name}")
    payload_run_id = _validate_run_id(payload.get("run_id"), source="raw artifact run_id")
    if run_id is not None and payload_run_id != _validate_run_id(run_id):
        raise ValueError("raw artifact run_id does not match requested run_id")
    artifact_file = _validate_sanitized_basename(
        payload.get("artifact_file"), field="raw artifact file"
    )
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, dict):
        raise TypeError("raw artifact is missing telemetry")
    telemetry_run_id = _validate_run_id(
        telemetry.get("run_id"), source="raw telemetry run_id"
    )
    if telemetry_run_id != payload_run_id:
        raise ValueError("raw telemetry and artifact have different run_id values")
    environment_file = _validate_sanitized_basename(
        telemetry.get("environment_file"), field="raw environment file"
    )
    power_trace = telemetry.get("power_trace")
    if not isinstance(power_trace, dict):
        raise TypeError("raw artifact is missing power data")
    _validate_captured_power_trace(power_trace, run_id=payload_run_id)
    power_file = _validate_sanitized_basename(
        power_trace.get("file"), field="raw power trace file"
    )
    for filename, kind in ((environment_file, "environment"), (power_file, "power")):
        token = _filename_run_id(Path(filename), kind)
        if token is not None and token != payload_run_id:
            raise ValueError(
                f"raw {kind} artifact {filename} is associated with run_id {token!r}, "
                f"not {payload_run_id!r}"
            )
    normalized_environment = telemetry.get("normalized_environment")
    if not isinstance(normalized_environment, dict):
        raise TypeError(
            "raw artifact is missing captured telemetry.normalized_environment"
        )
    _validate_captured_environment(normalized_environment, run_id=payload_run_id)
    raw_environment = telemetry.get("environment")
    if not isinstance(raw_environment, dict):
        raise TypeError("raw artifact is missing captured environment")
    raw_environment_run_id = raw_environment.get("run_id")
    if raw_environment_run_id is not None and _validate_run_id(
        raw_environment_run_id, source="raw environment run_id"
    ) != payload_run_id:
        raise ValueError("raw environment and artifact have different run_id values")
    recovery_run = payload.get("recovery_run", payload.get("run"))
    if not isinstance(recovery_run, dict):
        raise TypeError("raw artifact is missing run data")
    recovery_run_id = _validate_run_id(
        recovery_run.get("run_id"), source="raw run data run_id"
    )
    if recovery_run_id != payload_run_id:
        raise ValueError("raw run data and artifact have different run_id values")
    if recovery_run.get("status") != "pass":
        stage = recovery_run.get("failure_stage", "unknown")
        raise ValueError(
            f"raw artifact failed at stage {stage}; no complete record is recoverable"
        )
    configs = payload.get("configs")
    if not isinstance(configs, dict):
        raise TypeError("raw artifact is missing configs")
    try:
        repeats = int(configs["launches_per_row"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw artifact has invalid launches_per_row") from exc
    if repeats <= 0:
        raise ValueError("raw artifact launches_per_row must be positive")
    return (
        payload_run_id,
        normalized_environment,
        power_trace,
        recovery_run,
        repeats,
        {"file": artifact_file},
    )


def recover_combined_record(
    raw_path: Path, *, run_id: str | None = None
) -> dict:
    """Rebuild a record from captured provenance without reading recovery-host state."""
    selected_path, payload = _load_selected_raw_artifact(raw_path, run_id=run_id)
    (
        captured_run_id,
        normalized_environment,
        power_trace,
        recovery_run,
        repeats,
        raw_artifact,
    ) = _validate_raw_payload(selected_path, payload, run_id=run_id)
    return _record_from_parts(
        recovery_run,
        environment=normalized_environment,
        power_trace=power_trace,
        repeats=repeats,
        raw_artifact_name=raw_artifact["file"],
        run_id=captured_run_id,
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
    parser.add_argument(
        "--run-id",
        help="select this run_id when --recover-raw names a reused output directory",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.recover_raw is not None:
        if args.output is None:
            parser.error("--output is required with --recover-raw")
        record = recover_combined_record(args.recover_raw, run_id=args.run_id)
        destination = args.output.resolve()
        source = args.recover_raw.resolve()
        captured_raw_name = record.get("raw_artifact", {}).get("file")
        if destination == source or (
            source.is_dir()
            and destination.parent == source
            and destination.name == captured_raw_name
        ):
            parser.error("--output must not overwrite the captured raw artifact")
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

    result_path = os.environ.get("HEKATUS_TT_RESULT_PATH")
    output_path = Path(result_path) if result_path else output_dir / ISSUE101_OUTPUT_NAME
    _atomic_json_write(output_path, record)
    print(f"combined record -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
