"""Hand-written batched complex Newton-Schulz kernel through ``ttnn.generic_op``.

The accelerator module is passed in rather than imported here.  Host-only
accounting and reference tests therefore do not acquire a toolchain dependency.
The throughput variants use 32x32 tiles and a fixed eight-iteration inverse.
The first variant keeps BF16 state; ``bf16-fp32state`` keeps R in BF16 while
using FP32 for X, S, products, state, and outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NEWTON_SCHULZ_ITERATIONS = 8
COMPLEX_MATMULS_PER_INVERSE = 2 * NEWTON_SCHULZ_ITERATIONS
MATH_FIDELITY_CHOICES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
_SUPPORTED_VARIANTS = ("bf16", "bf16-fp32state")
_VARIANTS = {name: name == "bf16-fp32state" for name in _SUPPORTED_VARIANTS}
_TILE = 32
_TILE_BYTES_BFLOAT16 = _TILE * _TILE * 2
_TILE_BYTES_FLOAT32 = _TILE * _TILE * 4
_KERNEL_DIR = Path(__file__).with_name("kernels")

# CB indices are shared by the three kernels.  The first seven are reader
# inputs; the remaining queues are compute-owned intermediates and outputs.
CB_R_REAL = 0
CB_R_NEG_IMAG = 1
CB_R_IMAG = 2
CB_X0_REAL = 3
CB_X0_IMAG = 4
CB_IDENTITY = 5
CB_ZERO = 6
CB_STATE_REAL = 7
CB_STATE_IMAG = 8
CB_S_REAL = 9
CB_S_IMAG = 10
CB_PRODUCT_REAL = 11
CB_PRODUCT_IMAG = 12
CB_NEG_X_IMAG = 13
CB_OUTPUT_REAL = 15
CB_OUTPUT_IMAG = 16


def _initial_value(matrices: np.ndarray) -> np.ndarray:
    """Return the fixed X0 without depending on the NumPy oracle."""
    matrices = np.asarray(matrices, dtype=np.complex64)
    norm_1 = np.linalg.norm(matrices, ord=1, axis=(-2, -1))
    norm_inf = np.linalg.norm(matrices, ord=np.inf, axis=(-2, -1))
    denominator = (norm_1 * norm_inf)[:, None, None]
    return np.swapaxes(matrices.conj(), -1, -2) / denominator


def _balanced_ranges(batch: int, core_count: int) -> list[tuple[int, int]]:
    """Split ``batch`` contiguous tiles across active cores with a remainder."""
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    if core_count < 1:
        raise ValueError(f"core_count must be positive, got {core_count}")
    active_cores = min(batch, core_count)
    base, remainder = divmod(batch, active_cores)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(active_cores):
        count = base + int(index < remainder)
        ranges.append((start, count))
        start += count
    return ranges


def _pack_matrices(matrices: np.ndarray, *, packed: bool, tile_count: int) -> np.ndarray:
    """Pad or (for legacy host helpers) diagonal-pack matrices into tiles."""
    batch, size, _ = matrices.shape
    if packed:
        packed_matrices = np.zeros((tile_count, 1, _TILE, _TILE), dtype=np.float32)
        for source in range(batch):
            corner = 0 if source % 2 == 0 else size
            packed_matrices[source // 2, 0, corner : corner + size, corner : corner + size] = (
                matrices[source]
            )
        return packed_matrices

    padded = np.zeros((tile_count, 1, size, size), dtype=np.float32)
    padded[:batch, 0] = matrices
    return padded


def _unpack_matrices(values: np.ndarray, *, batch: int, size: int, packed: bool) -> np.ndarray:
    if not packed:
        return np.asarray(values[:batch, 0, :size, :size], dtype=np.float32)

    unpacked = np.empty((batch, size, size), dtype=np.float32)
    for destination in range(batch):
        corner = 0 if destination % 2 == 0 else size
        unpacked[destination] = values[
            destination // 2, 0, corner : corner + size, corner : corner + size
        ]
    return unpacked


def _device_tensor(ttnn, values: np.ndarray, device, *, dtype):
    host = ttnn.Tensor(np.ascontiguousarray(values), dtype)
    tiled = ttnn.to_layout(host, ttnn.TILE_LAYOUT)
    return ttnn.to_device(tiled, device, memory_config=ttnn.L1_MEMORY_CONFIG)


def _download_float32(ttnn, tensor) -> np.ndarray:
    host = ttnn.from_device(tensor)
    row_major = ttnn.to_layout(host, ttnn.ROW_MAJOR_LAYOUT)
    converted = ttnn.typecast(row_major, ttnn.float32)
    return converted.to_numpy()


def _core_grid(ttnn, device, batch: int):
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    ranges = _balanced_ranges(batch, total_cores)
    coordinates = [
        (index % grid.x, index // grid.x)
        for index in range(len(ranges))
    ]
    core_ranges = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in coordinates
        ]
    )
    return coordinates, core_ranges, ranges


def _runtime_args(ttnn, coordinates, values: list[int], ranges):
    args = ttnn.RuntimeArgs()
    for (x, y), (start, count) in zip(coordinates, ranges, strict=True):
        args[x][y] = [*values, start, count]
    return args


def _math_fidelity_value(ttnn, math_fidelity: str):
    """Map a public fidelity name to the pinned TTNN enum."""
    if math_fidelity not in MATH_FIDELITY_CHOICES:
        raise ValueError(
            f"unknown math fidelity {math_fidelity!r}; "
            f"choose from {MATH_FIDELITY_CHOICES}"
        )
    return getattr(ttnn.MathFidelity, math_fidelity)


def _state_dtype(ttnn, variant: str):
    """Return the state/output dtype selected by a throughput variant."""
    return ttnn.float32 if variant == "bf16-fp32state" else ttnn.bfloat16


def _output_memory_name(variant: str) -> str:
    """Return the output placement; FP32 fallback outputs live in DRAM."""
    return "dram" if variant == "bf16-fp32state" else "l1"


def _cb_definitions(ttnn, state_dtype) -> dict[int, tuple[Any, int]]:
    """Describe the CB formats shared by both state-precision variants."""
    return {
        CB_R_REAL: (ttnn.bfloat16, 2),
        CB_R_NEG_IMAG: (ttnn.bfloat16, 2),
        CB_R_IMAG: (ttnn.bfloat16, 2),
        CB_X0_REAL: (state_dtype, 2),
        CB_X0_IMAG: (state_dtype, 2),
        CB_IDENTITY: (ttnn.float32, 1),
        CB_ZERO: (ttnn.float32, 1),
        CB_STATE_REAL: (state_dtype, 2),
        CB_STATE_IMAG: (state_dtype, 2),
        CB_S_REAL: (state_dtype, 1),
        CB_S_IMAG: (state_dtype, 1),
        CB_PRODUCT_REAL: (ttnn.float32, 1),
        CB_PRODUCT_IMAG: (ttnn.float32, 1),
        CB_NEG_X_IMAG: (state_dtype, 1),
        # Keep the descriptor vector contiguous; this index is reserved
        # for later variants and is not referenced by this one.
        14: (ttnn.bfloat16, 1),
        CB_OUTPUT_REAL: (state_dtype, 2),
        CB_OUTPUT_IMAG: (state_dtype, 2),
    }


def benchmark_matrices(batch: int, size: int = _TILE, *, seed: int = 6300) -> np.ndarray:
    """Return deterministic non-zero inputs for a throughput run."""
    if batch < 1 or size != _TILE:
        raise ValueError(f"benchmark inputs require a positive batch and size {_TILE}")
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((batch, size, size), dtype=np.float32)
    imag = rng.standard_normal((batch, size, size), dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


@dataclass
class NewtonSchulzKernel:
    """Prepared tensors and one fixed-count ProgramDescriptor."""

    ttnn: Any
    device: Any
    batch: int
    size: int
    variant: str
    math_fidelity: str
    output_memory: str
    tile_count: int
    inputs: list[Any]
    outputs: list[Any]
    program: Any
    core_ranges: Any
    work_ranges: tuple[tuple[int, int], ...]

    @classmethod
    def prepare(
        cls,
        ttnn,
        device,
        matrices: np.ndarray,
        *,
        variant: str = "bf16",
        math_fidelity: str = "HiFi4",
        iterations: int = NEWTON_SCHULZ_ITERATIONS,
    ) -> NewtonSchulzKernel:
        if iterations != NEWTON_SCHULZ_ITERATIONS:
            raise ValueError(
                f"the kernel is fixed at {NEWTON_SCHULZ_ITERATIONS} iterations, got {iterations}"
            )
        if variant not in _SUPPORTED_VARIANTS:
            raise ValueError(f"unknown kernel variant {variant!r}")
        state_fp32 = variant == "bf16-fp32state"

        matrices = np.asarray(matrices, dtype=np.complex64)
        if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
            raise ValueError("matrices must have shape (batch, size, size)")
        batch, size, _ = matrices.shape
        if batch < 1:
            raise ValueError("batch must be positive")
        if size != _TILE:
            raise ValueError(f"the throughput kernel only supports L={_TILE}, got {size}")
        math_fidelity_value = _math_fidelity_value(ttnn, math_fidelity)

        coordinates, core_ranges, work_ranges = _core_grid(ttnn, device, batch)
        tile_count = batch
        x0 = _initial_value(matrices)
        r_real_values = _pack_matrices(matrices.real, packed=False, tile_count=tile_count)
        r_imag_values = _pack_matrices(matrices.imag, packed=False, tile_count=tile_count)
        r_negative_imag_values = -r_imag_values
        x_real_values = _pack_matrices(x0.real, packed=False, tile_count=tile_count)
        x_imag_values = _pack_matrices(x0.imag, packed=False, tile_count=tile_count)
        identity_values = (2.0 * np.eye(_TILE, dtype=np.float32))[None, None]
        zero_values = np.zeros((1, 1, _TILE, _TILE), dtype=np.float32)

        input_values = (
            r_real_values,
            r_negative_imag_values,
            r_imag_values,
            x_real_values,
            x_imag_values,
            identity_values,
            zero_values,
        )
        state_dtype = _state_dtype(ttnn, variant)
        input_dtypes = (
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            state_dtype,
            state_dtype,
            ttnn.float32,
            ttnn.float32,
        )
        inputs = [
            _device_tensor(ttnn, values, device, dtype=dtype)
            for values, dtype in zip(input_values, input_dtypes, strict=True)
        ]
        output_shape = ttnn.Shape(r_real_values.shape)
        output_memory = _output_memory_name(variant)
        output_memory_config = (
            ttnn.DRAM_MEMORY_CONFIG if output_memory == "dram" else ttnn.L1_MEMORY_CONFIG
        )
        outputs = [
            ttnn.allocate_tensor_on_device(
                output_shape,
                state_dtype,
                ttnn.TILE_LAYOUT,
                device,
                output_memory_config,
            )
            for _ in range(2)
        ]

        cb_definitions = _cb_definitions(ttnn, state_dtype)
        cbs = []
        for index, (data_format, page_count) in cb_definitions.items():
            page_size = (
                _TILE_BYTES_FLOAT32 if data_format == ttnn.float32 else _TILE_BYTES_BFLOAT16
            )
            descriptor = ttnn.CBFormatDescriptor(
                buffer_index=index,
                data_format=data_format,
                page_size=page_size,
                tile=ttnn.TileDescriptor(_TILE, _TILE, False),
            )
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=page_count * page_size,
                    core_ranges=core_ranges,
                    format_descriptors=[descriptor],
                )
            )

        reader_compile_args = [iterations]
        for tensor in inputs:
            reader_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        writer_compile_args: list[int] = []
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

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str((_KERNEL_DIR / "newton_schulz_reader.cpp").resolve()),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=reader_compile_args,
                runtime_args=reader_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str((_KERNEL_DIR / "newton_schulz_writer.cpp").resolve()),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=writer_compile_args,
                runtime_args=writer_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str((_KERNEL_DIR / "newton_schulz_compute.cpp").resolve()),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=[iterations, int(state_fp32)],
                runtime_args=compute_args,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=math_fidelity_value,
                    dst_full_sync_en=True,
                    fp32_dest_acc_en=True,
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return cls(
            ttnn=ttnn,
            device=device,
            batch=batch,
            size=size,
            variant=variant,
            math_fidelity=math_fidelity,
            output_memory=output_memory,
            tile_count=tile_count,
            inputs=inputs,
            outputs=outputs,
            program=program,
            core_ranges=core_ranges,
            work_ranges=tuple(work_ranges),
        )

    def launch(self) -> None:
        """Launch once for the complete batch."""
        self.ttnn.generic_op([*self.inputs, *self.outputs], self.program)

    def result(self) -> np.ndarray:
        real = _download_float32(self.ttnn, self.outputs[0])
        imag = _download_float32(self.ttnn, self.outputs[1])
        return real[: self.batch, 0] + 1j * imag[: self.batch, 0]

    def close(self) -> None:
        for tensor in [*self.inputs, *self.outputs]:
            try:
                self.ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - cleanup must not hide the result
                pass


def run_newton_schulz_kernel(
    ttnn,
    device,
    matrices: np.ndarray,
    *,
    variant: str = "bf16",
    math_fidelity: str = "HiFi4",
) -> np.ndarray:
    """Prepare, launch, download, and release one correctness run."""
    kernel = NewtonSchulzKernel.prepare(
        ttnn,
        device,
        matrices,
        variant=variant,
        math_fidelity=math_fidelity,
    )
    try:
        kernel.launch()
        ttnn.synchronize_device(device)
        return kernel.result()
    finally:
        kernel.close()
