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
MATRIX_BLOCK_CHOICES = (1, 2, 4)
_TILE = 32
_TILE_BYTES_BFLOAT16 = _TILE * _TILE * 2
_TILE_BYTES_FLOAT32 = _TILE * _TILE * 4
# The board exposes 16 32x32 DEST tiles.  FP32 accumulation halves the tile
# count, and half-sync mode halves it once more.  Keep these limits explicit so
# an invalid matrix block fails before any tensor allocation or launch.
_DEST_TILES = 16
_L1_CB_BUDGET_BYTES = 1_300_000
# TT-Metal reserves a fixed static L1 prefix before interleaved tensor buffers.
# The remaining 1.5 MiB Tensix L1 is the allocator-visible budget.
_L1_STATIC_BASE_BYTES = 111_360
_L1_TOTAL_BUDGET_BYTES = 1_572_864
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
CB_R_NEG_REAL = 14
CB_OUTPUT_REAL = 15
CB_OUTPUT_IMAG = 16
CB_PROFILE_READER = 17
CB_PROFILE_COMPUTE = 18
CB_PROFILE_WRITER = 19
PROFILE_MEASUREMENT_CORE = 0
PROFILE_PAGES_PER_CORE = 3
PROFILE_PAGE_WORDS = _TILE * _TILE
PROFILE_MAGIC = 0x5052464C
PROFILE_UINT32_MASK = 0xFFFFFFFF
PROFILE_READY_OFFSET = 31
PROFILE_WARMUP_READY_OFFSET = 63
PROFILE_SLOT_STRIDE = 64
PROFILE_WARMUP_BASE = 32
PROFILE_TOTAL_OFFSET = 0
PROFILE_R_WAIT_OFFSET = 1
PROFILE_X_WAIT_OFFSET = 2
PROFILE_COMPLEX_REAL_OFFSET = 3
PROFILE_COMPLEX_IMAG_OFFSET = 4
PROFILE_S_BINARY_OFFSET = 5
PROFILE_PACK_PUSH_OFFSET = 6
PROFILE_STATE_HANDOFF_OFFSET = 7
PROFILE_SECTION_SUM_OFFSET = 8
PROFILE_RESIDUAL_OFFSET = 9
PROFILE_EVENT_COUNT_OFFSET = 10
PROFILE_WARMUP_EVENT_COUNT_OFFSET = 11
PROFILE_SAMPLE_COUNT_OFFSET = PROFILE_EVENT_COUNT_OFFSET
PROFILE_READER_CB_WAIT_OFFSET = 1
PROFILE_READER_NOC_READ_OFFSET = 2
PROFILE_READER_COUNT_OFFSET = PROFILE_EVENT_COUNT_OFFSET
PROFILE_WRITER_CB_WAIT_OFFSET = 1
PROFILE_WRITER_NOC_WRITE_OFFSET = 2
PROFILE_WRITER_COUNT_OFFSET = PROFILE_EVENT_COUNT_OFFSET


def _initial_value(matrices: np.ndarray) -> np.ndarray:
    """Return the fixed X0 without depending on the NumPy oracle."""
    matrices = np.asarray(matrices, dtype=np.complex64)
    norm_1 = np.linalg.norm(matrices, ord=1, axis=(-2, -1))
    norm_inf = np.linalg.norm(matrices, ord=np.inf, axis=(-2, -1))
    denominator = (norm_1 * norm_inf)[:, None, None]
    return np.swapaxes(matrices.conj(), -1, -2) / denominator


def _balanced_ranges(
    batch: int, core_count: int, matrix_block: int = 1
) -> list[tuple[int, int]]:
    """Split contiguous tiles into core ranges aligned to matrix blocks."""
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    if core_count < 1:
        raise ValueError(f"core_count must be positive, got {core_count}")
    _validate_matrix_block(matrix_block)

    full_groups, partial = divmod(batch, matrix_block)
    group_count = full_groups + int(partial != 0)
    active_cores = min(group_count, core_count)
    base, remainder = divmod(group_count, active_cores)
    ranges: list[tuple[int, int]] = []
    group_start = 0
    for index in range(active_cores):
        groups = base + int(index < remainder)
        start = group_start * matrix_block
        end = min((group_start + groups) * matrix_block, batch)
        ranges.append((start, end - start))
        group_start += groups
    return ranges


def _matrix_block_ranges(start: int, count: int, matrix_block: int) -> list[tuple[int, int]]:
    """Return ``(start, count)`` groups, including a final partial group."""
    _validate_matrix_block(matrix_block)
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    return [
        (group_start, min(matrix_block, start + count - group_start))
        for group_start in range(start, start + count, matrix_block)
    ]


def _dest_slot_limit(*, fp32_dest_acc_en: bool, dst_full_sync_en: bool) -> int:
    """Return the hardware DEST tile count for a compute configuration."""
    tiles = _DEST_TILES // (2 if fp32_dest_acc_en else 1)
    if not dst_full_sync_en:
        tiles //= 2
    return tiles


def _validate_matrix_block(
    matrix_block: int,
    *,
    fp32_dest_acc_en: bool = True,
    dst_full_sync_en: bool = True,
    variant: str | None = None,
) -> None:
    """Validate block, variant, and the two-output complex DEST footprint."""
    if (
        isinstance(matrix_block, bool)
        or not isinstance(matrix_block, int)
        or matrix_block not in MATRIX_BLOCK_CHOICES
    ):
        raise ValueError(
            f"matrix_block must be one of {MATRIX_BLOCK_CHOICES}, got {matrix_block!r}"
        )
    if variant is not None and variant not in _SUPPORTED_VARIANTS:
        raise ValueError(f"unknown kernel variant {variant!r}")
    if variant == "bf16-fp32state" and not fp32_dest_acc_en:
        raise ValueError("bf16-fp32state requires fp32_dest_acc_en")
    # Complex products, fused S, state handoff, and final output all reserve
    # two independent DEST tiles per matrix (real and imaginary).
    required_slots = 2 * matrix_block
    available_slots = _dest_slot_limit(
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
    )
    if required_slots > available_slots:
        raise ValueError(
            f"matrix_block={matrix_block} requires {required_slots} DEST slots, "
            f"but the selected DEST configuration provides {available_slots}"
        )


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


def _download_uint32(ttnn, tensor) -> np.ndarray:
    host = ttnn.from_device(tensor)
    row_major = ttnn.to_layout(host, ttnn.ROW_MAJOR_LAYOUT)
    return row_major.to_numpy()


def _core_grid(ttnn, device, batch: int, matrix_block: int = 1):
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    ranges = _balanced_ranges(batch, total_cores, matrix_block)
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


def _cb_page_size(ttnn, data_format) -> int:
    """Return one 32x32 page in bytes, including uint32 profile tiles."""
    uint32 = getattr(ttnn, "uint32", None)
    return _TILE_BYTES_FLOAT32 if data_format == ttnn.float32 or data_format == uint32 else _TILE_BYTES_BFLOAT16


def _cb_definitions(
    ttnn,
    state_dtype,
    *,
    profile: bool = False,
    fuse_s: bool = False,
    matrix_block: int = 1,
) -> dict[int, tuple[Any, int]]:
    """Describe the CB formats shared by both state-precision variants."""
    _validate_matrix_block(matrix_block)
    definitions = {
        CB_R_REAL: (ttnn.bfloat16, 2),
        CB_R_NEG_IMAG: (ttnn.bfloat16, 2),
        CB_R_IMAG: (ttnn.bfloat16, 2),
        CB_X0_REAL: (state_dtype, 2),
        CB_X0_IMAG: (state_dtype, 2),
        # Fused S starts each DEST tile from host-prepared BF16 2I.  The
        # baseline keeps its original FP32 binary-operation identity.
        CB_IDENTITY: (ttnn.bfloat16 if fuse_s else ttnn.float32, 1),
        CB_ZERO: (ttnn.float32, 1),
        CB_STATE_REAL: (state_dtype, 2),
        CB_STATE_IMAG: (state_dtype, 2),
        CB_S_REAL: (state_dtype, 1),
        CB_S_IMAG: (state_dtype, 1),
        CB_PRODUCT_REAL: (ttnn.float32, 1),
        CB_PRODUCT_IMAG: (ttnn.float32, 1),
        CB_NEG_X_IMAG: (state_dtype, 1),
        # CB14 is a second resident R input only for fused S.
        CB_R_NEG_REAL: (ttnn.bfloat16, 2 if fuse_s else 1),
        CB_OUTPUT_REAL: (state_dtype, 2),
        CB_OUTPUT_IMAG: (state_dtype, 2),
    }
    # Identity/zero and profile pages are resident singletons.  Every queue
    # carrying a matrix, intermediate, or output is widened for one block.
    resident = {CB_IDENTITY, CB_ZERO, CB_PROFILE_READER, CB_PROFILE_COMPUTE, CB_PROFILE_WRITER}
    definitions = {
        index: (
            data_format,
            page_count
            if index in resident or matrix_block == 1
            else (
                page_count
                if fuse_s and index in {CB_PRODUCT_REAL, CB_PRODUCT_IMAG}
                else max(page_count, matrix_block)
            ),
        )
        for index, (data_format, page_count) in definitions.items()
    }
    if profile:
        definitions.update(
            {
                CB_PROFILE_READER: (ttnn.uint32, 1),
                CB_PROFILE_COMPUTE: (ttnn.uint32, 1),
                CB_PROFILE_WRITER: (ttnn.uint32, 1),
            }
        )
    return definitions


def _cb_l1_bytes(ttnn, definitions: dict[int, tuple[Any, int]]) -> int:
    """Return the per-core circular-buffer footprint in bytes."""
    return sum(
        _cb_page_size(ttnn, data_format) * page_count
        for data_format, page_count in definitions.values()
    )


def _tensor_l1_bytes(
    ttnn,
    *,
    batch: int,
    core_count: int,
    state_dtype,
    fuse_s: bool,
    output_memory: str,
) -> int:
    """Estimate the largest per-core tensor footprint before CB allocation."""
    tiles_per_core = (batch + core_count - 1) // core_count
    r_inputs = 4 if fuse_s else 3
    input_bytes = r_inputs * _cb_page_size(ttnn, ttnn.bfloat16)
    input_bytes += 2 * _cb_page_size(ttnn, state_dtype)
    resident_bytes = _cb_page_size(ttnn, ttnn.bfloat16) + _cb_page_size(ttnn, ttnn.float32)
    output_bytes = 0
    if output_memory == "l1":
        output_bytes = 2 * _cb_page_size(ttnn, state_dtype)
    return tiles_per_core * (input_bytes + output_bytes) + resident_bytes


def _validate_l1_budget(
    ttnn,
    definitions: dict[int, tuple[Any, int]],
    *,
    tensor_bytes: int = 0,
) -> int:
    """Reject CBs plus tensors that cannot coexist in one Tensix L1."""
    cb_bytes = _cb_l1_bytes(ttnn, definitions)
    usage = _L1_STATIC_BASE_BYTES + cb_bytes + tensor_bytes
    if cb_bytes > _L1_CB_BUDGET_BYTES or usage > _L1_TOTAL_BUDGET_BYTES:
        raise ValueError(
            f"matrix_block needs {cb_bytes} CB bytes plus {tensor_bytes} tensor bytes "
            f"and {_L1_STATIC_BASE_BYTES} static bytes per core, above the "
            f"{_L1_TOTAL_BUDGET_BYTES}-byte L1 budget"
        )
    return usage


def _decode_counter_page(
    page: np.ndarray,
    section_specs: tuple[tuple[str, int], ...],
    *,
    profile_page_ready: bool,
) -> dict:
    """Decode all-scope and first-sample counter scopes with exact checks."""
    all_total = int(page[PROFILE_TOTAL_OFFSET])
    warmup_total = int(page[PROFILE_WARMUP_BASE + PROFILE_TOTAL_OFFSET])
    named_sections = [
        {
            "name": name,
            "cycles": int(page[offset]),
            "percent_of_total": 100.0 * int(page[offset]) / all_total if all_total else 0.0,
            "warmup_cycles": int(page[PROFILE_WARMUP_BASE + offset]),
            "warmup_percent_of_total": (
                100.0 * int(page[PROFILE_WARMUP_BASE + offset]) / warmup_total
                if warmup_total
                else 0.0
            ),
        }
        for name, offset in section_specs
    ]
    named_section_sum = sum(entry["cycles"] for entry in named_sections)
    warmup_named_section_sum = sum(entry["warmup_cycles"] for entry in named_sections)
    all_residual = (all_total - named_section_sum) & PROFILE_UINT32_MASK
    warmup_residual = (warmup_total - warmup_named_section_sum) & PROFILE_UINT32_MASK
    all_sections = [
        *named_sections,
        {
            "name": "unclassified_overhead",
            "cycles": all_residual,
            "percent_of_total": 100.0 * all_residual / all_total if all_total else 0.0,
            "warmup_cycles": warmup_residual,
            "warmup_percent_of_total": (
                100.0 * warmup_residual / warmup_total if warmup_total else 0.0
            ),
        },
    ]
    warmup_sections = [
        {
            "name": entry["name"],
            "cycles": entry["warmup_cycles"],
            "percent_of_total": entry["warmup_percent_of_total"],
        }
        for entry in all_sections
    ]
    all_section_sum = sum(entry["cycles"] for entry in all_sections)
    warmup_section_sum = sum(entry["cycles"] for entry in warmup_sections)
    recorded_all_sum = int(page[PROFILE_SECTION_SUM_OFFSET])
    recorded_warmup_sum = int(page[PROFILE_WARMUP_BASE + PROFILE_SECTION_SUM_OFFSET])
    recorded_all_residual = int(page[PROFILE_RESIDUAL_OFFSET])
    recorded_warmup_residual = int(page[PROFILE_WARMUP_BASE + PROFILE_RESIDUAL_OFFSET])
    all_record_valid = (
        recorded_all_sum == named_section_sum & PROFILE_UINT32_MASK
        and recorded_all_residual == all_residual
    )
    warmup_record_valid = (
        recorded_warmup_sum == warmup_named_section_sum & PROFILE_UINT32_MASK
        and recorded_warmup_residual == warmup_residual
    )
    named_sections_cover_total = all_total == named_section_sum
    warmup_named_sections_cover_total = warmup_total == warmup_named_section_sum
    all_exact = all_total == all_section_sum
    warmup_exact = warmup_total == warmup_section_sum
    return {
        "profile_page_ready": profile_page_ready,
        "total_cycles": all_total,
        "warmup_total_cycles": warmup_total,
        "event_count": int(page[PROFILE_EVENT_COUNT_OFFSET]),
        "warmup_event_count": int(page[PROFILE_WARMUP_BASE + PROFILE_WARMUP_EVENT_COUNT_OFFSET]),
        "sample_count": int(page[PROFILE_WARMUP_BASE + PROFILE_WARMUP_EVENT_COUNT_OFFSET]),
        "sections": all_sections,
        "warmup_sections": warmup_sections,
        "section_sum_cycles": named_section_sum,
        "coverage_section_sum_cycles": all_section_sum,
        "warmup_section_sum_cycles": warmup_named_section_sum,
        "warmup_coverage_section_sum_cycles": warmup_section_sum,
        "recorded_section_sum_cycles": recorded_all_sum,
        "recorded_warmup_section_sum_cycles": recorded_warmup_sum,
        "residual_cycles": all_residual,
        "warmup_residual_cycles": warmup_residual,
        "recorded_residual_cycles": recorded_all_residual,
        "recorded_warmup_residual_cycles": recorded_warmup_residual,
        "named_sections_cover_total": named_sections_cover_total,
        "warmup_named_sections_cover_total": warmup_named_sections_cover_total,
        "consistency_exact": all_exact,
        "warmup_consistency_exact": warmup_exact,
        "consistency_record_valid": all_record_valid,
        "warmup_consistency_record_valid": warmup_record_valid,
        "consistency_pass": all_exact and all_record_valid,
        "warmup_consistency_pass": warmup_exact and warmup_record_valid,
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
    profile: bool
    fuse_s: bool
    batch_reads: bool
    matrix_block: int
    profile_output: Any | None
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
        profile: bool = False,
        fuse_s: bool = False,
        batch_reads: bool = False,
        matrix_block: int = 1,
        fp32_dest_acc_en: bool = True,
        dst_full_sync_en: bool = True,
        iterations: int = NEWTON_SCHULZ_ITERATIONS,
    ) -> NewtonSchulzKernel:
        if iterations != NEWTON_SCHULZ_ITERATIONS:
            raise ValueError(
                f"the kernel is fixed at {NEWTON_SCHULZ_ITERATIONS} iterations, got {iterations}"
            )
        if variant not in _SUPPORTED_VARIANTS:
            raise ValueError(f"unknown kernel variant {variant!r}")
        _validate_matrix_block(
            matrix_block,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            variant=variant,
        )
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

        coordinates, core_ranges, work_ranges = _core_grid(
            ttnn, device, batch, matrix_block
        )
        tile_count = batch
        x0 = _initial_value(matrices)
        r_real_values = _pack_matrices(matrices.real, packed=False, tile_count=tile_count)
        r_imag_values = _pack_matrices(matrices.imag, packed=False, tile_count=tile_count)
        r_negative_imag_values = -r_imag_values
        r_negative_real_values = -r_real_values
        x_real_values = _pack_matrices(x0.real, packed=False, tile_count=tile_count)
        x_imag_values = _pack_matrices(x0.imag, packed=False, tile_count=tile_count)
        identity_values = (2.0 * np.eye(_TILE, dtype=np.float32))[None, None]
        zero_values = np.zeros((1, 1, _TILE, _TILE), dtype=np.float32)

        input_values = [
            r_real_values,
            r_negative_imag_values,
            r_imag_values,
        ]
        if fuse_s:
            input_values.append(r_negative_real_values)
        input_values.extend(
            [
                x_real_values,
                x_imag_values,
                identity_values,
                zero_values,
            ]
        )
        state_dtype = _state_dtype(ttnn, variant)
        cb_definitions = _cb_definitions(
            ttnn,
            state_dtype,
            profile=profile,
            fuse_s=fuse_s,
            matrix_block=matrix_block,
        )
        tensor_l1_bytes = _tensor_l1_bytes(
            ttnn,
            batch=batch,
            core_count=len(work_ranges),
            state_dtype=state_dtype,
            fuse_s=fuse_s,
            output_memory=_output_memory_name(variant),
        )
        _validate_l1_budget(ttnn, cb_definitions, tensor_bytes=tensor_l1_bytes)
        input_dtypes = [ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16]
        if fuse_s:
            input_dtypes.append(ttnn.bfloat16)
        input_dtypes.extend(
            [
                state_dtype,
                state_dtype,
                ttnn.bfloat16 if fuse_s else ttnn.float32,
                ttnn.float32,
            ]
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
        profile_output = None
        if profile:
            # The profile CB pages are raw uint32 words, not tile-face data.
            # Use one row-major 4096-byte page per tensor page so the NOC
            # writer and host decoder see the same word offsets.
            profile_shape = ttnn.Shape((PROFILE_PAGES_PER_CORE, 1, 1, PROFILE_PAGE_WORDS))
            profile_output = ttnn.allocate_tensor_on_device(
                profile_shape,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            outputs.append(profile_output)

        cbs = []
        for index, (data_format, page_count) in cb_definitions.items():
            page_size = _cb_page_size(ttnn, data_format)
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
        if profile or fuse_s or batch_reads or matrix_block > 1:
            reader_compile_args.extend([int(fuse_s), int(batch_reads), matrix_block])
        for tensor in inputs:
            reader_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        writer_compile_args: list[int] = [matrix_block]
        for tensor in outputs:
            writer_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

        reader_args = _runtime_args(
            ttnn,
            coordinates,
            [tensor.buffer_address() for tensor in inputs],
            work_ranges,
        )
        if profile:
            writer_args = ttnn.RuntimeArgs()
            output_addresses = [tensor.buffer_address() for tensor in outputs[:2]]
            profile_address = outputs[2].buffer_address()
            for (x, y), (start, count) in zip(coordinates, work_ranges, strict=True):
                writer_args[x][y] = [
                    *output_addresses,
                    profile_address,
                    PROFILE_MEASUREMENT_CORE,
                    start,
                    count,
                ]
        else:
            writer_args = _runtime_args(
                ttnn,
                coordinates,
                [tensor.buffer_address() for tensor in outputs],
                work_ranges,
            )
        compute_args = _runtime_args(ttnn, coordinates, [], work_ranges)
        reader_source = (
            _KERNEL_DIR / "newton_schulz_reader_profile.cpp"
            if profile
            else (
                _KERNEL_DIR / "newton_schulz_reader_optimized.cpp"
                if fuse_s or batch_reads or matrix_block > 1
                else _KERNEL_DIR / "newton_schulz_reader.cpp"
            )
        )

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(reader_source.resolve()),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=reader_compile_args,
                runtime_args=reader_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(
                    (_KERNEL_DIR / ("newton_schulz_writer_profile.cpp" if profile else "newton_schulz_writer.cpp")).resolve()
                ),
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
                compile_time_args=[iterations, int(state_fp32), int(profile), int(fuse_s), matrix_block],
                runtime_args=compute_args,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=math_fidelity_value,
                    dst_full_sync_en=dst_full_sync_en,
                    fp32_dest_acc_en=fp32_dest_acc_en,
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
            profile=profile,
            fuse_s=fuse_s,
            batch_reads=batch_reads,
            matrix_block=matrix_block,
            profile_output=profile_output,
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

    def profile_records(self) -> list[dict]:
        """Decode core-0 row-major uint32 pages and exact scope checks."""
        if self.profile_output is None:
            return []
        pages = _download_uint32(self.ttnn, self.profile_output).reshape(
            PROFILE_PAGES_PER_CORE, -1
        )
        reader, compute, writer = pages
        reader_ready = (
            int(reader[PROFILE_READY_OFFSET]) == PROFILE_MAGIC
            and int(reader[PROFILE_WARMUP_READY_OFFSET]) == PROFILE_MAGIC
        )
        compute_ready = all(
            int(compute[base + ready]) == PROFILE_MAGIC
            for base in (0, PROFILE_SLOT_STRIDE, 2 * PROFILE_SLOT_STRIDE)
            for ready in (PROFILE_READY_OFFSET, PROFILE_WARMUP_READY_OFFSET)
        )
        writer_ready = (
            int(writer[PROFILE_READY_OFFSET]) == PROFILE_MAGIC
            and int(writer[PROFILE_WARMUP_READY_OFFSET]) == PROFILE_MAGIC
        )
        core_index = PROFILE_MEASUREMENT_CORE
        records: list[dict] = []
        common = {"core_index": core_index, "measurement_core": core_index}
        reader_record = _decode_counter_page(
            reader,
            (("reader_cb_empty_wait", PROFILE_READER_CB_WAIT_OFFSET),
             ("reader_noc_read_and_barrier", PROFILE_READER_NOC_READ_OFFSET)),
            profile_page_ready=reader_ready,
        )
        reader_record.update(common)
        reader_record["risc"] = "NCRISC"
        records.append(reader_record)
        compute_sections = (
            ("r_cb_wait", PROFILE_R_WAIT_OFFSET),
            ("x_cb_wait", PROFILE_X_WAIT_OFFSET),
            ("complex_real", PROFILE_COMPLEX_REAL_OFFSET),
            ("complex_imag", PROFILE_COMPLEX_IMAG_OFFSET),
            ("s_binary", PROFILE_S_BINARY_OFFSET),
            ("pack_push", PROFILE_PACK_PUSH_OFFSET),
            ("state_handoff", PROFILE_STATE_HANDOFF_OFFSET),
        )
        for risc_index, risc in enumerate(("TRISC0", "TRISC1", "TRISC2")):
            base = risc_index * PROFILE_SLOT_STRIDE
            record = _decode_counter_page(
                compute[base : base + PROFILE_SLOT_STRIDE],
                compute_sections,
                profile_page_ready=compute_ready,
            )
            record.update(common)
            record["risc"] = risc
            records.append(record)
        writer_record = _decode_counter_page(
            writer,
            (("writer_cb_wait_front", PROFILE_WRITER_CB_WAIT_OFFSET),
             ("writer_noc_write_and_barrier", PROFILE_WRITER_NOC_WRITE_OFFSET)),
            profile_page_ready=writer_ready,
        )
        writer_record.update(common)
        writer_record["risc"] = "BRISC"
        records.append(writer_record)
        return records

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
    profile: bool = False,
    fuse_s: bool = False,
    batch_reads: bool = False,
    matrix_block: int = 1,
) -> np.ndarray:
    """Prepare, launch, download, and release one correctness run."""
    kernel = NewtonSchulzKernel.prepare(
        ttnn,
        device,
        matrices,
        variant=variant,
        math_fidelity=math_fidelity,
        profile=profile,
        fuse_s=fuse_s,
        batch_reads=batch_reads,
        matrix_block=matrix_block,
    )
    try:
        kernel.launch()
        ttnn.synchronize_device(device)
        return kernel.result()
    finally:
        kernel.close()
