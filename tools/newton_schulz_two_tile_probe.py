"""Run one minimal two-tile a/b_prime/c probe on device 0.

The wrapper supplies the pinned container, named-container cleanup, Watcher,
and power sampling.  This runner owns only one stage, batch four, and one
matrix product per input; it never resets the device.
"""

from __future__ import annotations

import argparse
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
    ALL_PROBE_STAGE_CHOICES,
    expected_probe_outputs,
    normalize_probe_stage,
    probe_input_pages,
    probe_stage_contract,
)

TILE = 32
BATCH = 4
STAGE_INDEX = {"a": 0, "a1": 1, "a2": 2, "a3": 3, "b_prime": 4, "c": 5}
POWER_TRACE_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")


def _probe_matrices() -> tuple[np.ndarray, np.ndarray]:
    rng_r = np.random.default_rng(6300)
    rng_x = np.random.default_rng(6301)
    r = (
        rng_r.standard_normal((BATCH, TILE, TILE), dtype=np.float32)
        + 1j * rng_r.standard_normal((BATCH, TILE, TILE), dtype=np.float32)
    ).astype(np.complex64)
    x = (
        rng_x.standard_normal((BATCH, TILE, TILE), dtype=np.float32)
        + 1j * rng_x.standard_normal((BATCH, TILE, TILE), dtype=np.float32)
    ).astype(np.complex64)
    return r, x


def _cb_descriptors(ttnn, core_ranges, state_dtype):
    formats = {
        3: (state_dtype, 1),
        4: (state_dtype, 1),
        5: (ttnn.bfloat16, 1),
        6: (ttnn.bfloat16, 1),
        13: (state_dtype, 1),
        15: (state_dtype, 1),
        16: (state_dtype, 1),
        20: (ttnn.bfloat16, 6),
        21: (state_dtype, 4),
        # Four pages permit one consumed X column and one produced S column
        # to occupy the same CB without changing its descriptor.
        22: (state_dtype, 2),
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


def _relative_errors(actual: np.ndarray, expected: np.ndarray) -> list[float]:
    return [
        float(np.linalg.norm(actual[index] - expected[index]) / np.linalg.norm(expected[index]))
        for index in range(actual.shape[0])
    ]


def _output_tile_errors(actual: np.ndarray, expected: np.ndarray) -> list[dict[str, float | int]]:
    """Report each matrix's packed real and imaginary output tile separately."""
    rows = []
    for index in range(actual.shape[0]):
        real_error = float(
            np.linalg.norm(actual[index].real - expected[index].real)
            / np.linalg.norm(expected[index].real)
        )
        imag_error = float(
            np.linalg.norm(actual[index].imag - expected[index].imag)
            / np.linalg.norm(expected[index].imag)
        )
        rows.append({"matrix": index, "real_tile": real_error, "imag_tile": imag_error})
    return rows


def run_stage(ttnn, device, stage: str) -> dict:
    stage = normalize_probe_stage(stage)
    r, x = _probe_matrices()
    pages = probe_input_pages(r, x)
    expected = expected_probe_outputs(r, x)[stage]
    coordinates, core_ranges, work_ranges = _core_grid(ttnn, device, BATCH, 1)
    state_dtype = ttnn.float32

    input_values = [
        pages["in0_r"],
        x.real[:, None, :, :].astype(np.float32),
        x.imag[:, None, :, :].astype(np.float32),
        (-x.imag)[:, None, :, :].astype(np.float32),
        pages["one_identity"],
        pages["zero"],
    ]
    input_dtypes = [
        ttnn.bfloat16,
        state_dtype,
        state_dtype,
        state_dtype,
        ttnn.bfloat16,
        ttnn.bfloat16,
    ]
    inputs = [
        _device_tensor(
            ttnn,
            values,
            device,
            dtype=dtype,
            input_memory="dram",
        )
        for values, dtype in zip(input_values, input_dtypes, strict=True)
    ]
    output_shape = ttnn.Shape((BATCH, 1, TILE, TILE))
    outputs = [
        ttnn.allocate_tensor_on_device(
            output_shape,
            state_dtype,
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
    kernel_dir = Path(__file__).resolve().parents[1] / "enodia/tt/bench/kernels"
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str((kernel_dir / "two_tile_probe_reader.cpp").resolve()),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=reader_compile_args,
                runtime_args=reader_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str((kernel_dir / "two_tile_probe_writer.cpp").resolve()),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=writer_compile_args,
                runtime_args=writer_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str((kernel_dir / "two_tile_probe_compute.cpp").resolve()),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=[STAGE_INDEX[stage]],
                runtime_args=compute_args,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi3,
                    dst_full_sync_en=True,
                    fp32_dest_acc_en=True,
                ),
            ),
        ],
        semaphores=[],
        cbs=_cb_descriptors(ttnn, core_ranges, state_dtype),
    )
    try:
        ttnn.generic_op([*inputs, *outputs], program)
        ttnn.synchronize_device(device)
        actual_real = _download_float32(ttnn, outputs[0])[:, 0]
        actual_imag = _download_float32(ttnn, outputs[1])[:, 0]
        actual = actual_real + 1j * actual_imag
    finally:
        for tensor in [*inputs, *outputs]:
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask probe result
                pass

    errors = _relative_errors(actual, expected)
    output_tile_errors = _output_tile_errors(actual, expected)
    return {
        "stage": stage,
        "status": "pass" if max(errors) <= 1e-2 else "fail",
        "batch": BATCH,
        "iterations": 1,
        "matrix_block": 1,
        "relative_errors": errors,
        "output_tile_relative_errors": output_tile_errors,
        "max_relative_error": max(errors),
        "tolerance": 1e-2,
        "finite": bool(np.isfinite(actual).all()),
        "expected": "independent NumPy probe oracle with BF16-rounded R",
        "host_input_seed": {"r": 6300, "x": 6301},
        "contract": probe_stage_contract(stage),
    }


def main(argv: list[str] | None = None) -> int:
    import ttnn

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=ALL_PROBE_STAGE_CHOICES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-json", type=Path, default=None)
    args = parser.parse_args(argv)
    canonical_stage = normalize_probe_stage(args.stage)

    device = ttnn.open_device(device_id=0)
    try:
        stage_result = run_stage(ttnn, device, canonical_stage)
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
        "configuration_mode": "issue_63_two_tile_minimal_probe",
        "selection": {
            "stage": canonical_stage,
            "stage_index": STAGE_INDEX[canonical_stage],
            "batch": BATCH,
            "iterations": 1,
            "matrix_block": 1,
            "device_id": 0,
            "input_memory": "dram",
            "output_memory": "dram",
            "math_fidelity": "HiFi3",
            "variant": "bf16-fp32state",
        },
        "measurement": {
            "kind": "device_two_tile_minimal_probe",
            "stage": canonical_stage,
            "watcher": os.environ.get("TT_METAL_WATCHER") == "1",
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
            "source_files": [
                "enodia/tt/bench/kernels/two_tile_probe_reader.cpp",
                "enodia/tt/bench/kernels/two_tile_probe_compute.cpp",
                "enodia/tt/bench/kernels/two_tile_probe_writer.cpp",
                "enodia/tt/bench/two_tile_probe.py",
            ],
            "contract": probe_stage_contract(canonical_stage),
            "in0": "CB_TWO_TILE_R six-page K=3 block for a/b_prime, CB_TWO_TILE_X for c; matmul maps it to SrcB",
            "in1": "CB_TWO_TILE_S [Xr, Xi] plus resident BF16 CB_IDENTITY [I]; matmul maps both to SrcA",
            "dest_slots": [0, 1],
            "tile_regs_sequence": ["acquire", "commit", "wait", "release"],
            "cb_wait_reserve_push_pop": probe_stage_contract(canonical_stage)["cb_counts"],
        },
        "result": stage_result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(strict_json_dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0 if stage_result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
