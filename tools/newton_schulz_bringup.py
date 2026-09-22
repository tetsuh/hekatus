"""Single-stage board bring-up diagnostic for issue #63.

The stages intentionally use separate compute sources.  A later experiment can
therefore fail to compile without changing the source used by an earlier
boundary check.  This script is diagnostic-only; it does not reset hardware.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TILE = 32
NUMERICAL_TOLERANCE = 1e-2
TILE_BYTES_BFLOAT16 = TILE * TILE * 2
TILE_BYTES_FLOAT32 = TILE * TILE * 4
DEFAULT_CB_FORMATS = ("bfloat16",) * 25
DEFAULT_CB_PAGE_SIZES = (TILE_BYTES_BFLOAT16,) * 25
STAGE_50_CB_FORMATS = tuple("float32" if index == 16 else "bfloat16" for index in range(25))
STAGE_51_CB_FORMATS = tuple("float32" if index in (16, 17) else "bfloat16" for index in range(25))
STAGE_50_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index == 16 else TILE_BYTES_BFLOAT16 for index in range(25)
)
STAGE_51_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in (16, 17) else TILE_BYTES_BFLOAT16 for index in range(25)
)
STAGE_55_CB_FORMATS = tuple("float32" if index in (16, 17) else "bfloat16" for index in range(25))
STAGE_55_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in (16, 17) else TILE_BYTES_BFLOAT16 for index in range(25)
)
KERNEL_DIR = (Path(__file__).resolve().parents[1] / "enodia/tt/bench/kernels").resolve()


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    batch: int
    cores: int
    compute_source: str
    reader_source: str
    writer_source: str
    kind: str
    iterations: int
    fp32_dest_acc_en: bool = False
    input_seed: int | None = None
    input_count: int = 6
    input_dtypes: tuple[str, ...] = ("bfloat16",) * 6
    output_dtype: str = "bfloat16"
    cb_formats: tuple[str, ...] = DEFAULT_CB_FORMATS
    cb_page_sizes: tuple[int, ...] = DEFAULT_CB_PAGE_SIZES
    output_count_override: int | None = None


STAGES = {
    1: Stage(
        1,
        "real_one_tile",
        1,
        1,
        "bringup_real_compute.cpp",
        "bringup_real_reader.cpp",
        "bringup_real_writer.cpp",
        "real",
        1,
    ),
    2: Stage(
        2,
        "real_multiple_tiles",
        4,
        1,
        "bringup_real_compute.cpp",
        "bringup_real_reader.cpp",
        "bringup_real_writer.cpp",
        "real",
        1,
    ),
    3: Stage(
        3,
        "real_multiple_cores",
        4,
        2,
        "bringup_real_compute.cpp",
        "bringup_real_reader.cpp",
        "bringup_real_writer.cpp",
        "real",
        1,
    ),
    4: Stage(
        4,
        "complex_one_matmul",
        1,
        1,
        "bringup_complex_compute.cpp",
        "bringup_complex_reader.cpp",
        "bringup_writer.cpp",
        "complex",
        1,
    ),
    5: Stage(
        5,
        "complex_newton_schulz_one_iteration",
        1,
        1,
        "bringup_ns_one_compute.cpp",
        "bringup_ns_one_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        1,
    ),
    6: Stage(
        6,
        "complex_newton_schulz_eight_iterations",
        1,
        1,
        "bringup_ns_eight_compute.cpp",
        "bringup_ns_eight_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        8,
        False,
        6306,
    ),
    41: Stage(
        41,
        "complex_modern_startup",
        1,
        1,
        "bringup_complex_modern_compute.cpp",
        "bringup_complex_modern_reader.cpp",
        "bringup_complex_modern_writer.cpp",
        "complex_modern_startup",
        1,
    ),
    42: Stage(
        42,
        "complex_two_groups",
        1,
        1,
        "bringup_complex_two_groups_compute.cpp",
        "bringup_complex_two_groups_reader.cpp",
        "bringup_complex_two_groups_writer.cpp",
        "complex_two_groups",
        1,
    ),
    43: Stage(
        43,
        "newton_residual_only",
        1,
        1,
        "bringup_newton_residual_compute.cpp",
        "bringup_newton_residual_reader.cpp",
        "bringup_newton_residual_writer.cpp",
        "newton_residual",
        1,
    ),
    44: Stage(
        44,
        "newton_residual_reader_copy",
        1,
        1,
        "bringup_newton_residual_copy_compute.cpp",
        "bringup_newton_residual_copy_reader.cpp",
        "bringup_newton_residual_copy_writer.cpp",
        "newton_residual_reader_copy",
        1,
    ),
    45: Stage(
        45,
        "newton_one_compute_copy",
        1,
        1,
        "bringup_newton_one_compute_copy_compute.cpp",
        "bringup_newton_one_compute_copy_reader.cpp",
        "bringup_newton_one_compute_copy_writer.cpp",
        "newton_one_compute_copy",
        1,
    ),
    46: Stage(
        46,
        "newton_residual_correct",
        1,
        1,
        "bringup_newton_residual_correct_compute.cpp",
        "bringup_newton_residual_correct_reader.cpp",
        "bringup_newton_residual_correct_writer.cpp",
        "newton_residual_correct",
        1,
    ),
    47: Stage(
        47,
        "newton_residual_correct_reader_copy",
        1,
        1,
        "bringup_newton_residual_correct_reader_copy_compute.cpp",
        "bringup_newton_residual_correct_reader_copy_reader.cpp",
        "bringup_newton_residual_correct_reader_copy_writer.cpp",
        "newton_residual_correct_reader_copy",
        1,
    ),
    48: Stage(
        48,
        "newton_one_correct_reader_copy",
        1,
        1,
        "bringup_newton_one_correct_reader_copy_compute.cpp",
        "bringup_newton_one_correct_reader_copy_reader.cpp",
        "bringup_newton_one_correct_reader_copy_writer.cpp",
        "newton_one_correct_reader_copy",
        1,
    ),
    49: Stage(
        49,
        "complex_newton_schulz_eight_iterations_fp32_dest_acc",
        1,
        1,
        "bringup_ns_eight_compute.cpp",
        "bringup_ns_eight_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        8,
        True,
        6306,
    ),
    50: Stage(
        50,
        "real_fp32_dest_acc_one_matmul",
        1,
        1,
        "bringup_precision_real_one_compute.cpp",
        "bringup_precision_real_one_reader.cpp",
        "bringup_precision_real_writer.cpp",
        "precision_real",
        1,
        True,
        6350,
        2,
        ("bfloat16", "bfloat16"),
        "float32",
        STAGE_50_CB_FORMATS,
        STAGE_50_CB_PAGE_SIZES,
        1,
    ),
    51: Stage(
        51,
        "real_fp32_dest_acc_two_matmuls",
        1,
        1,
        "bringup_precision_real_two_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6351,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    52: Stage(
        52,
        "real_fp32_dest_acc_two_matmuls_reconfig",
        1,
        1,
        "bringup_precision_real_two_reconfig_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6352,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    53: Stage(
        53,
        "real_fp32_dest_acc_two_matmuls_unpack_reconfig",
        1,
        1,
        "bringup_precision_real_two_unpack_reconfig_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6353,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    54: Stage(
        54,
        "real_fp32_dest_acc_two_matmuls_pack_reconfig",
        1,
        1,
        "bringup_precision_real_two_pack_reconfig_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6354,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    55: Stage(
        55,
        "real_fp32_dest_acc_two_matmuls_output_reconfig",
        1,
        1,
        "bringup_precision_real_two_output_reconfig_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_output_reconfig_writer.cpp",
        "precision_real_two",
        1,
        True,
        6355,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "bfloat16",
        STAGE_55_CB_FORMATS,
        STAGE_55_CB_PAGE_SIZES,
        1,
    ),
}


def stage_for(number: int) -> Stage:
    try:
        return STAGES[number]
    except KeyError as exc:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {number}") from exc


def _hpd_batch(batch: int, *, condition_number: float, seed: int) -> np.ndarray:
    """Build deterministic complex HPD matrices with a requested spectrum."""
    eigenvalues = np.geomspace(1.0, condition_number, TILE)
    rng = np.random.default_rng(seed)
    matrices = np.empty((batch, TILE, TILE), dtype=np.complex64)
    for index in range(batch):
        random = rng.normal(size=(TILE, TILE)) + 1j * rng.normal(size=(TILE, TILE))
        basis, _ = np.linalg.qr(random)
        matrices[index] = (basis * eigenvalues) @ basis.conj().T
    return matrices


def _inputs(stage: Stage) -> list[np.ndarray]:
    """Build deterministic float32 tile inputs in the fixed descriptor order."""
    input_seed = 6300 + stage.number if stage.input_seed is None else stage.input_seed
    rng = np.random.default_rng(input_seed)
    shape = (stage.batch, TILE, TILE)
    if stage.kind in {
        "newton_schulz",
        "newton_residual",
        "newton_residual_reader_copy",
        "newton_one_compute_copy",
        "newton_residual_correct",
        "newton_residual_correct_reader_copy",
        "newton_one_correct_reader_copy",
    }:
        matrices = _hpd_batch(stage.batch, condition_number=100.0, seed=input_seed)
        a_real = matrices.real.astype(np.float32)
        a_imag = matrices.imag.astype(np.float32)
        b_real = np.empty_like(a_real)
        b_imag = np.empty_like(a_imag)
    else:
        a_real = rng.normal(0.0, 0.05, shape).astype(np.float32)
        b_real = rng.normal(0.0, 0.05, shape).astype(np.float32)
        a_imag = rng.normal(0.0, 0.05, shape).astype(np.float32)
        b_imag = rng.normal(0.0, 0.05, shape).astype(np.float32)
    if stage.kind in {
        "newton_schulz",
        "newton_residual",
        "newton_residual_reader_copy",
        "newton_one_compute_copy",
        "newton_residual_correct",
        "newton_residual_correct_reader_copy",
        "newton_one_correct_reader_copy",
    }:
        x0 = _initial_value(a_real.astype(np.complex64) + 1j * a_imag)
        b_real = x0.real.astype(np.float32)
        b_imag = x0.imag.astype(np.float32)
    identity = np.broadcast_to(
        2.0 * np.eye(TILE, dtype=np.float32), (stage.batch, TILE, TILE)
    ).copy()
    zero = np.zeros(shape, dtype=np.float32)
    return [a_real, b_real, a_imag, b_imag, identity, zero]


def _initial_value(values: np.ndarray) -> np.ndarray:
    norm_1 = np.linalg.norm(values, ord=1, axis=(-2, -1))
    norm_inf = np.linalg.norm(values, ord=np.inf, axis=(-2, -1))
    return np.swapaxes(values.conj(), -1, -2) / (norm_1 * norm_inf)[:, None, None]


def expected_output(stage: Stage, inputs: list[np.ndarray]) -> np.ndarray:
    a_real, b_real, a_imag, b_imag, identity, _ = inputs
    if stage.kind == "real":
        return np.matmul(a_real, b_real).astype(np.complex64)
    if stage.kind == "precision_real":
        return np.matmul(a_real, b_real).astype(np.float32)
    if stage.kind == "precision_real_two":
        first = np.matmul(a_real, b_real)
        return np.matmul(first, a_imag).astype(np.float32)
    if stage.kind in {"complex", "complex_modern_startup", "complex_two_groups"}:
        left = a_real.astype(np.complex64) + 1j * a_imag
        right = b_real.astype(np.complex64) + 1j * b_imag
        return np.matmul(left, right).astype(np.complex64)

    r = a_real.astype(np.complex64) + 1j * a_imag
    if stage.kind in {
        "newton_residual",
        "newton_residual_reader_copy",
        "newton_residual_correct",
        "newton_residual_correct_reader_copy",
    }:
        return (identity.astype(np.complex64) - np.matmul(r, _initial_value(r))).astype(
            np.complex64
        )
    x = _initial_value(r)
    for _ in range(stage.iterations):
        product = np.matmul(r, x)
        s = identity.astype(np.complex64) - product
        x = np.matmul(x, s)
    return x.astype(np.complex64)


def source_paths(stage: Stage) -> tuple[Path, Path, Path]:
    compute = (KERNEL_DIR / stage.compute_source).resolve()
    reader = (KERNEL_DIR / stage.reader_source).resolve()
    writer = (KERNEL_DIR / stage.writer_source).resolve()
    for path in (compute, reader, writer):
        if not path.is_file():
            raise FileNotFoundError(path)
    return compute, reader, writer


def _core_coordinates(ttnn: Any, device: Any, count: int) -> tuple[list[tuple[int, int]], Any]:
    grid = device.compute_with_storage_grid_size()
    if count < 1 or count > grid.x * grid.y:
        raise ValueError(f"requested {count} cores, device grid is {grid.x}x{grid.y}")
    coordinates = [(index % grid.x, index // grid.x) for index in range(count)]
    ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coordinates]
    )
    return coordinates, ranges


def _runtime_args(
    ttnn: Any,
    coordinates: list[tuple[int, int]],
    values: list[int],
    tiles_per_core: int,
) -> Any:
    args = ttnn.RuntimeArgs()
    for index, (x, y) in enumerate(coordinates):
        args[x][y] = [*values, index * tiles_per_core, tiles_per_core]
    return args


def _device_tensor(ttnn: Any, values: np.ndarray, device: Any, dtype_name: str) -> Any:
    if values.ndim == 3:
        values = values[:, None, :, :]
    tensor = ttnn.Tensor(np.ascontiguousarray(values), getattr(ttnn, dtype_name))
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return ttnn.to_device(tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)


def _download(ttnn: Any, tensor: Any) -> np.ndarray:
    host = ttnn.from_device(tensor)
    row_major = ttnn.to_layout(host, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.typecast(row_major, ttnn.float32).to_numpy()


def _descriptor(ttnn: Any, index: int, core_ranges: Any, data_format: str, page_size: int) -> Any:
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=index,
        data_format=getattr(ttnn, data_format),
        page_size=page_size,
        tile=ttnn.TileDescriptor(TILE, TILE, False),
    )
    return ttnn.CBDescriptor(
        total_size=4 * page_size,
        core_ranges=core_ranges,
        format_descriptors=[fmt],
    )


def output_count(stage: Stage) -> int:
    if stage.output_count_override is not None:
        return stage.output_count_override
    return 1 if stage.kind == "real" else 2


def _prepare(ttnn: Any, device: Any, stage: Stage) -> tuple[Any, list[Any], list[Any], np.ndarray]:
    inputs = _inputs(stage)
    expected = expected_output(stage, inputs)
    coordinates, core_ranges = _core_coordinates(ttnn, device, stage.cores)
    if stage.batch % stage.cores:
        raise ValueError("diagnostic stages require an even tile partition")
    tiles_per_core = stage.batch // stage.cores
    input_values = inputs[:2] if stage.kind == "real" else inputs[: stage.input_count]
    input_dtypes = stage.input_dtypes[: len(input_values)]
    device_inputs = [
        _device_tensor(ttnn, value, device, dtype_name)
        for value, dtype_name in zip(input_values, input_dtypes)
    ]
    output_shape = ttnn.Shape((stage.batch, 1, TILE, TILE))
    outputs = [
        ttnn.allocate_tensor_on_device(
            output_shape,
            getattr(ttnn, stage.output_dtype),
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG,
        )
        for _ in range(output_count(stage))
    ]

    reader_compile_args: list[int] = []
    for tensor in device_inputs:
        reader_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_compile_args: list[int] = []
    for tensor in outputs:
        writer_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    input_addresses = [tensor.buffer_address() for tensor in device_inputs]
    output_addresses = [tensor.buffer_address() for tensor in outputs]
    compute, reader, writer = source_paths(stage)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(reader),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=reader_compile_args,
            runtime_args=_runtime_args(ttnn, coordinates, input_addresses, tiles_per_core),
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(writer),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=writer_compile_args,
            runtime_args=_runtime_args(ttnn, coordinates, output_addresses, tiles_per_core),
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(compute),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=[tiles_per_core],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                dst_full_sync_en=True, fp32_dest_acc_en=stage.fp32_dest_acc_en
            ),
        ),
    ]
    program = ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[],
        cbs=[
            _descriptor(
                ttnn, index, core_ranges, stage.cb_formats[index], stage.cb_page_sizes[index]
            )
            for index in range(25)
        ],
    )
    return program, device_inputs, outputs, expected


def run_stage(ttnn: Any, device: Any, stage: Stage) -> dict[str, Any]:
    program, inputs, outputs, expected = _prepare(ttnn, device, stage)
    try:
        started = time.perf_counter()
        ttnn.generic_op([*inputs, *outputs], program)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - started
        real = _download(ttnn, outputs[0])[:, 0]
        imag = (
            np.zeros_like(real) if output_count(stage) == 1 else _download(ttnn, outputs[1])[:, 0]
        )
        actual = real + 1j * imag
        finite = bool(np.isfinite(actual).all())
        error = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))
        tolerance = NUMERICAL_TOLERANCE
        passed = finite and error <= tolerance
        return {
            "stage": stage.number,
            "name": stage.name,
            "status": "pass" if passed else "fail",
            "batch": stage.batch,
            "cores": stage.cores,
            "tile_shape": [TILE, TILE],
            "elapsed_s": elapsed,
            "numerical_error": error,
            "tolerance": tolerance,
            "finite": finite,
        }
    finally:
        for tensor in [*inputs, *outputs]:
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - diagnostics must attempt all cleanup
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    try:
        stage = stage_for(args.stage)
        import ttnn

        device = ttnn.open_device(device_id=args.device_id)
        try:
            result = run_stage(ttnn, device, stage)
        finally:
            ttnn.close_device(device)
    except Exception as exc:  # noqa: BLE001 - emit a machine-readable failure record
        result = {
            "stage": args.stage,
            "name": STAGES[args.stage].name if args.stage in STAGES else None,
            "status": "fail",
            "batch": STAGES[args.stage].batch if args.stage in STAGES else None,
            "cores": STAGES[args.stage].cores if args.stage in STAGES else None,
            "tile_shape": [TILE, TILE],
            "elapsed_s": None,
            "numerical_error": None,
            "finite": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
