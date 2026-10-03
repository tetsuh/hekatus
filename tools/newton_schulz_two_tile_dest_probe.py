"""Run the known-value one-call DEST-slot probe on device 0.

The wrapper supplies the pinned container, named-container cleanup, Watcher,
and power sampling.  This runner owns only batch four, one matrix product per
input, and one ``matmul_block`` call per matrix; it never resets the device.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.newton_schulz_kernel import (
    _cb_page_size,
    _core_grid,
    _device_tensor,
    _download_float32,
    _runtime_args,
)
from enodia.tt.bench.two_tile_probe import (
    DEST_PROBE_BATCH,
    DEST_PROBE_STAGE,
    classify_dest_probe_outputs,
    dest_probe_contract,
    dest_probe_input_pages,
    dest_probe_source_audit,
    expected_dest_probe_outputs,
    known_dest_probe_xr,
)

TILE = 32
POWER_TRACE_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")


def _cb_descriptors(ttnn, core_ranges):
    formats = {
        15: (ttnn.float32, 1),
        16: (ttnn.float32, 1),
        20: (ttnn.bfloat16, 4),
        22: (ttnn.float32, 1),
    }
    descriptors = []
    for index, (data_format, page_count) in formats.items():
        page_size = _cb_page_size(ttnn, data_format)
        descriptor = ttnn.CBFormatDescriptor(
            buffer_index=index,
            data_format=data_format,
            page_size=page_size,
            tile=ttnn.TileDescriptor(TILE, TILE, False),
        )
        descriptors.append(
            ttnn.CBDescriptor(
                total_size=page_count * page_size,
                core_ranges=core_ranges,
                format_descriptors=[descriptor],
            )
        )
    return descriptors


def _source_paths() -> dict[str, Path]:
    kernel_dir = Path(__file__).resolve().parents[1] / "enodia/tt/bench/kernels"
    return {
        "reader": kernel_dir / "two_tile_dest_probe_reader.cpp",
        "writer": kernel_dir / "two_tile_dest_probe_writer.cpp",
        "compute": kernel_dir / "two_tile_dest_probe_compute.cpp",
    }


def run_dest_probe(ttnn, device) -> dict:
    xr = known_dest_probe_xr()
    pages = dest_probe_input_pages(xr)
    expected = expected_dest_probe_outputs(xr)
    coordinates, core_ranges, work_ranges = _core_grid(ttnn, device, DEST_PROBE_BATCH, 1)

    inputs = []
    outputs = []
    try:
        inputs = [
            _device_tensor(
                ttnn,
                pages["in0"],
                device,
                dtype=ttnn.bfloat16,
                input_memory="dram",
            ),
            _device_tensor(
                ttnn,
                pages["in1"],
                device,
                dtype=ttnn.float32,
                input_memory="dram",
            ),
        ]
        output_shape = ttnn.Shape((DEST_PROBE_BATCH, 1, TILE, TILE))
        outputs = [
            ttnn.allocate_tensor_on_device(
                output_shape,
                ttnn.float32,
                ttnn.TILE_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]

        reader_compile_args = [1]
        for tensor in inputs:
            reader_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        writer_compile_args = [1]
        for tensor in outputs:
            writer_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        reader_args = _runtime_args(
            ttnn,
            coordinates,
            [tensor.buffer_address() for tensor in inputs],
            work_ranges,
        )
        writer_args = _runtime_args(
            ttnn,
            coordinates,
            [tensor.buffer_address() for tensor in outputs],
            work_ranges,
        )
        compute_args = _runtime_args(ttnn, coordinates, [], work_ranges)
        paths = _source_paths()
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(paths["reader"].resolve()),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_ranges,
                    compile_time_args=reader_compile_args,
                    runtime_args=reader_args,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(paths["writer"].resolve()),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_ranges,
                    compile_time_args=writer_compile_args,
                    runtime_args=writer_args,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(paths["compute"].resolve()),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_ranges,
                    compile_time_args=[],
                    runtime_args=compute_args,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi3,
                        dst_full_sync_en=True,
                        fp32_dest_acc_en=True,
                    ),
                ),
            ],
            semaphores=[],
            cbs=_cb_descriptors(ttnn, core_ranges),
        )
        ttnn.generic_op([*inputs, *outputs], program)
        ttnn.synchronize_device(device)
        actual_slot0 = _download_float32(ttnn, outputs[0])[:, 0]
        actual_slot1 = _download_float32(ttnn, outputs[1])[:, 0]
    finally:
        for tensor in [*inputs, *outputs]:
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask probe result
                pass

    classifications = classify_dest_probe_outputs(actual_slot0, actual_slot1, xr)
    expected_errors = [row["relative_error"] for row in classifications]
    return {
        "stage": DEST_PROBE_STAGE,
        "status": "pass" if all(row["pass"] for row in classifications) else "fail",
        "batch": DEST_PROBE_BATCH,
        "iterations": 1,
        "matrix_block": 1,
        "relative_errors": {
            "slot0": [
                float(np.linalg.norm(actual_slot0[index] - expected["slot0"][index]))
                / max(float(np.linalg.norm(expected["slot0"][index])), 1.0)
                for index in range(DEST_PROBE_BATCH)
            ],
            "slot1": [
                float(np.linalg.norm(actual_slot1[index] - expected["slot1"][index]))
                / max(float(np.linalg.norm(expected["slot1"][index])), 1.0)
                for index in range(DEST_PROBE_BATCH)
            ],
        },
        "max_relative_error": max(expected_errors),
        "tolerance": 1e-2,
        "finite": bool(np.isfinite(actual_slot0).all() and np.isfinite(actual_slot1).all()),
        "classifications": classifications,
        "expected": "known real Xr with in0 pages [I, 2I, sentinel0, sentinel1]",
        "host_input": {
            "complex_oracle": "known_dest_probe_complex_x",
            "device_page": "Xr",
        },
        "contract": dest_probe_contract(),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    import ttnn

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-json", type=Path, default=None)
    args = parser.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    try:
        result = run_dest_probe(ttnn, device)
    finally:
        ttnn.close_device(device)

    env_path = args.env_json
    if env_path is None:
        env_files = sorted(glob.glob("/out/env-*.json"))
        if not env_files:
            raise RuntimeError("wrapper environment metadata was not found")
        env_path = Path(env_files[-1])
    environment = json.loads(env_path.read_text())
    power_files = sorted(glob.glob("/out/power-*.csv"))
    power_trace = Path(power_files[-1]).name if power_files else None
    payload = {
        "environment": environment,
        "configuration_mode": "issue_63_two_tile_dest_slot_probe",
        "selection": {
            "stage": DEST_PROBE_STAGE,
            "batch": DEST_PROBE_BATCH,
            "iterations": 1,
            "matrix_block": 1,
            "device_id": 0,
            "input_memory": "dram",
            "output_memory": "dram",
            "math_fidelity": "HiFi3",
            "variant": "bf16-fp32state",
        },
        "measurement": {
            "kind": "device_two_tile_dest_slot_probe",
            "stage": DEST_PROBE_STAGE,
            "watcher": os.environ.get("TT_METAL_WATCHER") == "1",
            "dprint": {
                "environment_variable": "TT_METAL_DPRINT_CORES",
                "requested_cores": os.environ.get("TT_METAL_DPRINT_CORES"),
                "source_instrumented": True,
                "location": "before each pack_tile",
            },
            "container_timeout_s": 60,
            "named_container": True,
            "same_device_run": True,
            "device_id": 0,
            "power_trace": power_trace,
            "power_clock_provenance": {
                "trace": power_trace,
                "columns": list(POWER_TRACE_COLUMNS),
                "clock_column": "aiclk_mhz",
                "power_column": "power_w",
                "temperature_column": "asic_temp_c",
                "sampling_source": "tt-smi snapshot",
            },
            "environment_provenance": {
                "source": "--env-json",
                "record_field": "environment",
                "adr": "ADR-0005",
            },
        },
        "source_host_report": {
            "source_files": [str(path) for path in _source_paths().values()],
            "contract": dest_probe_contract(),
            "source_audit": dest_probe_source_audit(),
            "in0": "four pages [I, 2I, sentinel0, sentinel1]; pages 0/1 are consumed",
            "in1": "one FP32 Xr page; this one-call real probe does not supply Xi",
            "dest_slots": [0, 1],
            "pack": [
                {"dst_index": 0, "output_cb": 15, "pack_tile_count": 1},
                {"dst_index": 1, "output_cb": 16, "pack_tile_count": 1},
            ],
            "tile_regs_sequence": dest_probe_contract()["tile_regs_sequence"],
            "cb_wait_reserve_push_pop": dest_probe_contract()["cb_lifecycle_per_matrix"],
        },
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(strict_json_dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
