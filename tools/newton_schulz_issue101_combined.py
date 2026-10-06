"""Run the Issue #100 correctness and performance catalogue in one device session."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import platform
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


def run_issue101_correctness(ttnn: Any, device: Any) -> list[dict]:
    """Run the nine correctness cases on the already-open device."""
    rows: list[dict] = []
    for batch, size in ISSUE101_CORRECTNESS_CASES:
        row = _correctness_case(ttnn, device, batch, size)
        rows.append(row)
        if row["status"] != "pass":
            break
    return rows


def run_issue101_combined(ttnn: Any, device: Any, *, repeats: int = ISSUE100_LAUNCHES) -> dict:
    """Run correctness first, then the four performance rows on one device."""
    correctness = run_issue101_correctness(ttnn, device)
    if len(correctness) != len(ISSUE101_CORRECTNESS_CASES) or any(
        row["status"] != "pass" for row in correctness
    ):
        return {
            "status": "failed",
            "failure_stage": "correctness",
            "correctness_cases": correctness,
            "performance_rows": [],
        }

    performance = run_issue100_comparison(
        ttnn,
        device,
        repeats=repeats,
        stop_on_failure=True,
    )
    if len(performance) != len(ISSUE100_SHAPES) * len(ISSUE100_COMPARISON_CONFIGS) or any(
        row.get("status") != "ok" for row in performance
    ):
        return {
            "status": "failed",
            "failure_stage": "performance",
            "correctness_cases": correctness,
            "performance_rows": performance,
        }
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


def _raw_artifact_payload(run: dict, *, output_dir: Path, repeats: int) -> dict:
    """Capture every builder input before any record construction is attempted."""
    environment_path, raw_environment, normalized_environment = _environment_artifact(output_dir)
    power_trace = _power_trace_artifact(_single_artifact(output_dir, "power-*.csv"))
    return {
        "raw_schema": ISSUE101_RAW_SCHEMA,
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "run": run,
        "correctness_results": run.get("correctness_cases", []),
        "performance_results": run.get("performance_rows", []),
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
        "telemetry": {
            "environment_file": environment_path.name,
            "environment": raw_environment,
            "normalized_environment": normalized_environment,
            "power_trace": power_trace,
        },
    }


def write_raw_artifact(run: dict, *, output_dir: Path, repeats: int) -> Path:
    """Persist the complete device-session result before invoking the builder."""
    artifact_path = output_dir / ISSUE101_RAW_OUTPUT_NAME
    _atomic_json_write(
        artifact_path,
        _raw_artifact_payload(run, output_dir=output_dir, repeats=repeats),
    )
    return artifact_path


def persist_raw_and_build(
    run: dict, *, output_dir: Path, repeats: int = ISSUE100_LAUNCHES
) -> tuple[Path, dict | None]:
    """Write raw data first, then build; preserve the raw path on builder failure."""
    raw_path = write_raw_artifact(run, output_dir=output_dir, repeats=repeats)
    if run.get("status") != "pass":
        return raw_path, None
    try:
        record = build_combined_record(
            run,
            output_dir=output_dir,
            repeats=repeats,
            raw_artifact_path=raw_path,
        )
    except Exception as exc:
        raise RuntimeError(
            f"combined record builder failed after raw artifact {raw_path}: "
            f"{type(exc).__name__}: {exc}"
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
    return _record_from_parts(
        payload["run"],
        environment=environment,
        power_trace=power_trace,
        repeats=int(payload["configs"]["launches_per_row"]),
        raw_artifact_name=raw_path.name,
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

    import ttnn

    output_dir = Path(os.environ.get("HEKATUS_TT_OUTPUT_DIR", "/out"))
    device = ttnn.open_device(device_id=ISSUE100_DEVICE_ID)
    try:
        try:
            run = run_issue101_combined(ttnn, device, repeats=ISSUE100_LAUNCHES)
        except Exception as exc:  # noqa: BLE001 - retain partial raw device results
            run = {
                "status": "failed",
                "failure_stage": "device_session",
                "error": f"{type(exc).__name__}: {exc}",
                "correctness_cases": [],
                "performance_rows": [],
            }
    finally:
        ttnn.close_device(device)

    raw_path, record = persist_raw_and_build(
        run, output_dir=output_dir, repeats=ISSUE100_LAUNCHES
    )
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
