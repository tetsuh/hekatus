"""Hand-written batched complex Newton-Schulz kernel through ``ttnn.generic_op``.

The accelerator module is passed in rather than imported here.  Host-only
accounting and reference tests therefore do not acquire a toolchain dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from enodia.tt.bench.newton_schulz_reference import (
    NEWTON_SCHULZ_ITERATIONS,
    initial_value,
)

_TILE = 32
_TILE_BYTES_BFLOAT16 = _TILE * _TILE * 2
_KERNEL_DIR = Path(__file__).with_name("kernels")
_VARIANTS = {
    "fused": (False, False),
    "fused_resident": (True, False),
    "packed_fused": (False, True),
    "packed_fused_resident": (True, True),
}


def _pack_matrices(matrices: np.ndarray, *, packed: bool, tile_count: int) -> np.ndarray:
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


def _device_tensor(ttnn, values: np.ndarray, device):
    host = ttnn.Tensor(np.ascontiguousarray(values), ttnn.bfloat16)
    tiled = ttnn.to_layout(host, ttnn.TILE_LAYOUT)
    return ttnn.to_device(tiled, device, memory_config=ttnn.L1_MEMORY_CONFIG)


def _download_float32(ttnn, tensor) -> np.ndarray:
    host = ttnn.from_device(tensor)
    row_major = ttnn.to_layout(host, ttnn.ROW_MAJOR_LAYOUT)
    converted = ttnn.typecast(row_major, ttnn.float32)
    return converted.to_numpy()


def _core_grid(ttnn, device, useful_tiles: int):
    grid = device.compute_with_storage_grid_size()
    active_count = min(useful_tiles, grid.x * grid.y)
    coordinates = [
        (index % grid.x, index // grid.x)
        for index in range(active_count)
    ]
    core_ranges = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in coordinates
        ]
    )
    return coordinates, core_ranges


def _runtime_args(ttnn, coordinates, values: list[int], *, tiles_per_core: int):
    args = ttnn.RuntimeArgs()
    start = 0
    for x, y in coordinates:
        args[x][y] = [*values, start, tiles_per_core]
        start += tiles_per_core
    return args


@dataclass
class NewtonSchulzKernel:
    """Prepared tensors and one fixed-count ProgramDescriptor."""

    ttnn: Any
    device: Any
    batch: int
    size: int
    variant: str
    packed: bool
    keep_r_resident: bool
    tile_count: int
    inputs: list[Any]
    outputs: list[Any]
    program: Any

    @classmethod
    def prepare(
        cls,
        ttnn,
        device,
        matrices: np.ndarray,
        *,
        variant: str = "fused_resident",
        iterations: int = NEWTON_SCHULZ_ITERATIONS,
    ) -> NewtonSchulzKernel:
        if iterations != NEWTON_SCHULZ_ITERATIONS:
            raise ValueError(
                f"the kernel is fixed at {NEWTON_SCHULZ_ITERATIONS} iterations, got {iterations}"
            )
        try:
            keep_r_resident, packed = _VARIANTS[variant]
        except KeyError as exc:
            raise ValueError(f"unknown kernel variant {variant!r}") from exc

        matrices = np.asarray(matrices, dtype=np.complex64)
        if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
            raise ValueError("matrices must have shape (batch, size, size)")
        batch, size, _ = matrices.shape
        if size not in (16, 32):
            raise ValueError(f"only L=16 and L=32 are supported, got {size}")
        if packed and size != 16:
            raise ValueError("tile packing is only defined for L=16")

        useful_tiles = (batch + 1) // 2 if packed else batch
        coordinates, core_ranges = _core_grid(ttnn, device, useful_tiles)
        core_count = len(coordinates)
        tiles_per_core = (useful_tiles + core_count - 1) // core_count
        tile_count = tiles_per_core * core_count

        x0 = initial_value(matrices)
        r_real_values = _pack_matrices(matrices.real, packed=packed, tile_count=tile_count)
        r_imag_values = _pack_matrices(matrices.imag, packed=packed, tile_count=tile_count)
        x_real_values = _pack_matrices(x0.real, packed=packed, tile_count=tile_count)
        x_imag_values = _pack_matrices(x0.imag, packed=packed, tile_count=tile_count)
        logical_size = _TILE if packed else size
        identity_values = (2.0 * np.eye(logical_size, dtype=np.float32))[None, None]
        zero_values = np.zeros((1, 1, logical_size, logical_size), dtype=np.float32)

        inputs = [
            _device_tensor(ttnn, values, device)
            for values in (
                r_real_values,
                r_imag_values,
                x_real_values,
                x_imag_values,
                identity_values,
                zero_values,
            )
        ]
        output_shape = ttnn.Shape(r_real_values.shape)
        outputs = [
            ttnn.allocate_tensor_on_device(
                output_shape,
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                device,
                ttnn.L1_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]

        formats = [
            ttnn.CBFormatDescriptor(
                buffer_index=index,
                data_format=ttnn.bfloat16,
                page_size=_TILE_BYTES_BFLOAT16,
                tile=ttnn.TileDescriptor(_TILE, _TILE, False),
            )
            for index in (*range(8), *range(16, 25))
        ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=4 * _TILE_BYTES_BFLOAT16,
                core_ranges=core_ranges,
                format_descriptors=[format_descriptor],
            )
            for format_descriptor in formats
        ]

        reader_compile_args = [iterations, int(keep_r_resident)]
        for tensor in inputs:
            reader_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        writer_compile_args: list[int] = []
        for tensor in outputs:
            writer_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

        reader_args = _runtime_args(
            ttnn,
            coordinates,
            [tensor.buffer_address() for tensor in inputs],
            tiles_per_core=tiles_per_core,
        )
        writer_args = _runtime_args(
            ttnn,
            coordinates,
            [tensor.buffer_address() for tensor in outputs],
            tiles_per_core=tiles_per_core,
        )
        compute_args = [iterations, int(keep_r_resident), tiles_per_core]

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
                compile_time_args=compute_args,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(dst_full_sync_en=True),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return cls(
            ttnn=ttnn,
            device=device,
            batch=batch,
            size=size,
            variant=variant,
            packed=packed,
            keep_r_resident=keep_r_resident,
            tile_count=tile_count,
            inputs=inputs,
            outputs=outputs,
            program=program,
        )

    def launch(self) -> None:
        """Launch once for the complete padded batch."""
        self.ttnn.generic_op([*self.inputs, *self.outputs], self.program)

    def result(self) -> np.ndarray:
        real = _unpack_matrices(
            _download_float32(self.ttnn, self.outputs[0]),
            batch=self.batch,
            size=self.size,
            packed=self.packed,
        )
        imag = _unpack_matrices(
            _download_float32(self.ttnn, self.outputs[1]),
            batch=self.batch,
            size=self.size,
            packed=self.packed,
        )
        return real + 1j * imag

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
    variant: str = "fused_resident",
) -> np.ndarray:
    """Prepare, launch, download, and release one correctness run."""
    kernel = NewtonSchulzKernel.prepare(ttnn, device, matrices, variant=variant)
    try:
        kernel.launch()
        ttnn.synchronize_device(device)
        return kernel.result()
    finally:
        kernel.close()
