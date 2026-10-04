"""Capture tag-local official performance counters for one Issue #96 row."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tools.tracy.process_model_log import run_device_profiler

_ROWS = {
    "full": {
        "shape": "newton_schulz_L16_b8192",
        "variant": "bf16",
        "matrix_block": 8,
        "dst_full_sync_en": True,
    },
    "half": {
        "shape": "newton_schulz_L16_b8192",
        "variant": "bf16",
        "matrix_block": 2,
        "dst_full_sync_en": False,
    },
}


def _build_target_command(row: dict, output: str) -> str:
    sync_flag = "" if row["dst_full_sync_en"] else " --no-dst-full-sync-en"
    return (
        "/opt/venv/bin/python3 /work/enodia/tt/bench/run_matmul.py"
        f" --device-id 0 --only {row['shape']} --dtype bfloat16 --memory l1"
        " --kind custom_newton_schulz"
        f" --custom-variant {row['variant']} --custom-math-fidelity HiFi3"
        f" --fuse-s --matrix-block {row['matrix_block']} --double-buffer"
        " --input-memory l1 --r-memory l1 --x0-memory l1"
        f" --custom-output-memory dram{sync_flag}"
        f" --iters 1 --repeats 1 --out {output}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", choices=tuple(_ROWS))
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--target-out", default="/out/perf-target.json")
    args = parser.parse_args(argv)

    row = _ROWS[args.row]
    os.environ.pop("TT_METAL_WATCHER", None)
    os.environ["TT_METAL_LOGS_PATH"] = "/out"
    args.logs.mkdir(parents=True, exist_ok=True)
    run_device_profiler(
        _build_target_command(row, args.target_out),
        str(args.logs),
        check_test_return_code=True,
        python_post_process=True,
        capture_perf_counters_groups=["all"],
        is_command_binary_exe=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
