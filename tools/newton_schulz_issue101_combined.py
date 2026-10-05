"""Run the Issue #100 correctness and performance catalogue in one device session."""

from __future__ import annotations

import datetime
import json
import os
import platform
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


def normalize_environment(raw: dict) -> dict:
    """Add the README-authoritative environment aliases to telemetry output."""
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
    if isinstance(board, dict):
        board = dict(board)
        board.setdefault("device_id", ISSUE100_DEVICE_ID)
        serial = board.get("serial") or board.get("board_serial")
        if serial is not None:
            board.setdefault("serial", serial)
            board.setdefault("board_id", serial)
        environment["board"] = board

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


def _environment_from_output(output_dir: Path) -> dict:
    environment_path = _single_artifact(output_dir, "env-*.json")
    return normalize_environment(json.loads(environment_path.read_text()))


def build_combined_record(
    run: dict,
    *,
    output_dir: Path,
    repeats: int = ISSUE100_LAUNCHES,
) -> dict:
    """Build one ADR-0005 record after all rows have passed."""
    if run.get("status") != "pass":
        raise ValueError("a combined record requires all correctness and performance rows to pass")
    environment = _environment_from_output(output_dir)
    power_trace = _single_artifact(output_dir, "power-*.csv")
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

    return {
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
            "previous_default": {
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
            },
            "performance_launches_per_row": repeats,
            "performance_watcher": False,
            "correctness_cases": run["correctness_cases"],
            "performance_rows": run["performance_rows"],
            "power_trace": power_trace.name,
            "power_clock_provenance": {
                "trace": power_trace.name,
                "columns": list(ISSUE101_POWER_COLUMNS),
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
        "power_trace": power_trace.name,
        "notes": [
            "All nine correctness cases and four performance rows passed in one device session.",
            "Raw 1,000-launch samples and p50/p99/p99.9 values are retained per performance row.",
            "TFLOPS fields are explicitly p50-derived and fastest-launch-derived.",
            "No host name or user-specific absolute path is included in this record.",
        ],
    }


def main() -> int:
    """Run the combined catalogue and write a record only after full success."""
    import ttnn

    output_dir = Path(os.environ.get("HEKATUS_TT_OUTPUT_DIR", "/out"))
    device = ttnn.open_device(device_id=ISSUE100_DEVICE_ID)
    try:
        run = run_issue101_combined(ttnn, device, repeats=ISSUE100_LAUNCHES)
    finally:
        ttnn.close_device(device)

    if run.get("status") != "pass":
        print(
            f"issue101_combined status=failed stage={run.get('failure_stage', 'unknown')}",
            flush=True,
        )
        return 1

    record = build_combined_record(run, output_dir=output_dir)
    output_path = output_dir / ISSUE101_OUTPUT_NAME
    output_path.write_text(strict_json_dumps(record, indent=2) + "\n")
    print(f"combined record -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
