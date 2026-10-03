"""Temporary board runner for the Issue #63 no-Watcher three-row rerun."""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict
from pathlib import Path

import ttnn

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.run_matmul import (
    POWER_TRACE_COLUMNS,
    STOCK_KIND,
    _complex_catalogue_inputs,
    _complex_catalogue_reference,
    _run_complex_catalogue_correctness,
    _stock_math_fidelity,
    run_custom_newton_schulz,
    run_shape,
)
from enodia.tt.bench.shapes import default_catalogue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-json", type=Path, default=None)
    args = parser.parse_args()

    shape = next(
        item for item in default_catalogue() if item.name == "newton_schulz_L32_b8192"
    )
    matrices = _complex_catalogue_inputs(shape.batch, shape.m, seed=95)
    expected = _complex_catalogue_reference(matrices)
    device = ttnn.open_device(device_id=0)
    results = []
    try:
        stock = {
            "shape": asdict(shape),
            "execution_shape": asdict(shape),
            "representative": shape.representative,
            "dtype": "bfloat16",
            "memory": "l1",
            "input_memory": "l1",
            "memory_placement": {
                name: {"buffer": "l1", "layout": "interleaved"}
                for name in ("input_a", "input_b", "output")
            },
            "program_config": {"name": "default", "kind": "default"},
            "iterations": 1,
            "repeats": 1000,
            "launches_requested_per_row": 1000,
            "kind": STOCK_KIND,
            "row": "stock_best",
            **_stock_math_fidelity("bfloat16", None),
        }
        stock.update(
            run_shape(
                ttnn,
                device,
                shape,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                memory_name="l1",
                iters=1,
                repeats=1000,
            )
        )
        results.append(stock)

        for full_sync, block in ((True, 4), (False, 2)):
            row_name = (
                "one_tile_full_sync_block4"
                if full_sync
                else "one_tile_half_sync_block2"
            )
            correctness = _run_complex_catalogue_correctness(
                ttnn,
                device,
                matrices,
                expected=expected,
                two_tile_complex=False,
                dst_full_sync_en=full_sync,
                matrix_block=block,
            )
            row = {
                "shape": asdict(shape),
                "execution_shape": asdict(shape),
                "representative": shape.representative,
                "dtype": "bfloat16",
                "memory": "l1",
                "input_memory": "l1",
                "r_memory": "l1",
                "x0_memory": "l1",
                "output_memory": "dram",
                "memory_placement": {
                    "input": "l1",
                    "r": "l1",
                    "x0": "l1",
                    "compute": "l1",
                    "output": "dram",
                },
                "program_config": {
                    "name": "custom_newton_schulz",
                    "kind": "custom_newton_schulz",
                    "variant": "bf16-fp32state",
                    "math_fidelity": "HiFi3",
                    "fuse_s": True,
                    "two_tile_complex": False,
                    "batch_reads": False,
                    "matrix_block": block,
                    "fp32_dest_acc_en": True,
                    "dst_full_sync_en": full_sync,
                    "input_memory": "l1",
                    "r_memory": "l1",
                    "x0_memory": "l1",
                },
                "iterations": 1,
                "repeats": 1000,
                "launches_requested_per_row": 1000,
                "kind": "custom_newton_schulz",
                "row": row_name,
                "correctness": correctness,
                "measurement_mode": "correctness-and-throughput",
            }
            if correctness["status"] == "pass":
                row.update(
                    run_custom_newton_schulz(
                        ttnn,
                        device,
                        shape,
                        dtype_name="bfloat16",
                        memory_name="l1",
                        variant="bf16-fp32state",
                        math_fidelity="HiFi3",
                        fuse_s=True,
                        two_tile_complex=False,
                        batch_reads=False,
                        matrix_block=block,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=full_sync,
                        input_memory="l1",
                        r_memory="l1",
                        x0_memory="l1",
                        row_name=row_name,
                        iters=1,
                        repeats=1000,
                    )
                )
            else:
                row.update({"status": correctness["status"], "kind": "custom_newton_schulz"})
                if "error" in correctness:
                    row["error"] = correctness["error"]
            results.append(row)
    finally:
        ttnn.close_device(device)

    env_path = args.env_json
    if env_path is None:
        env_files = sorted(glob.glob("/out/env-*.json"))
        if not env_files:
            raise RuntimeError("wrapper environment metadata was not found")
        env_path = Path(env_files[-1])
    environment = json.loads(env_path.read_text())
    traces = sorted(glob.glob("/out/power-*.csv"))
    power_trace = Path(traces[-1]).name if traces else None
    payload = {
        "environment": environment,
        "configuration_mode": "issue_63_one_tile_throughput",
        "selection": {
            "shape_filters": [shape.name],
            "device_id": 0,
            "batch": shape.batch,
            "iters": 1,
            "launches_per_row": 1000,
            "variant": "bf16-fp32state",
            "math_fidelity": "HiFi3",
            "fuse_s": True,
            "two_tile_complex": False,
            "input_memory": "l1",
            "r_memory": "l1",
            "x0_memory": "l1",
            "output_memory": "dram",
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": [True, False],
            "matrix_blocks": {"full": 4, "half": 2},
        },
        "measurement": {
            "kind": "board_throughput_condition_rerun",
            "same_device_run": True,
            "watcher": False,
            "watcher_environment": "TT_METAL_WATCHER absent; wrapper received no watcher environment entry",
            "wrapper": "enodia/tt/bench/run_in_container.sh",
            "named_container": True,
            "container_timeout_s": 60,
            "device_id": 0,
            "stock_best_row": "stock_best",
            "rows": [
                "stock_best",
                "one_tile_full_sync_block4",
                "one_tile_half_sync_block2",
            ],
            "power_trace": power_trace,
            "power_clock_provenance": {
                "trace": power_trace,
                "columns": list(POWER_TRACE_COLUMNS),
                "power_column": "power_w",
                "clock_column": "aiclk_mhz",
                "temperature_column": "asic_temp_c",
                "sampling_source": "tt-smi snapshot",
            },
            "environment_provenance": {
                "source": "--env-json",
                "record_field": "environment",
                "adr": "ADR-0005",
            },
            "comparison_record": "2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000.json",
            "comparison_row": "block4_all_l1",
        },
        "peak_tflops": 332.0,
        "peak_note": "332 TFLOPS BF16 design peak; cross-row denominator",
        "results": results,
    }
    args.out.write_text(strict_json_dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
