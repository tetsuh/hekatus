"""Hand-written batched complex Newton-Schulz kernel through ``ttnn.generic_op``.

The accelerator module is passed in rather than imported here.  Host-only
accounting and reference tests therefore do not acquire a toolchain dependency.
The throughput variants use 32x32 tiles and a fixed twelve-iteration inverse.
Issue #100's block-8 default is fail-fast when DEST or L1 preflight rejects it;
no implicit block or memory fallback is performed.
L=16 inputs are paired on the diagonal of each 32x32 tile.  The first variant
keeps BF16 state; ``bf16-fp32state`` keeps R in BF16 while using FP32 for X, S,
products, state, and outputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NEWTON_SCHULZ_ITERATIONS = 12
COMPLEX_MATMULS_PER_INVERSE = 2 * NEWTON_SCHULZ_ITERATIONS
MATH_FIDELITY_CHOICES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
_SUPPORTED_VARIANTS = ("bf16", "bf16-fp32state")
_VARIANTS = {name: name == "bf16-fp32state" for name in _SUPPORTED_VARIANTS}
MATRIX_BLOCK_CHOICES = (1, 2, 4, 8)
INPUT_MEMORY_CHOICES = ("l1", "dram")

# Issue #100 selects the fastest measured host configuration.  These are
# defaults only: callers can still select every prior variant explicitly.
DEFAULT_VARIANT = "bf16"
DEFAULT_MATH_FIDELITY = "HiFi3"
DEFAULT_FUSE_S = True
DEFAULT_FP32_DEST_ACC_EN = True
DEFAULT_MATRIX_BLOCK = 8
DEFAULT_DOUBLE_BUFFER = True
DEFAULT_DST_FULL_SYNC_EN = True
DEFAULT_OUTPUT_MEMORY = "dram"
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

# CB indices are shared by the three kernels.  CB_R_REAL remains index 0 for
# the non-fused ABI, but fused S leaves that slot unused because it consumes
# only signed R components.  The remaining indices are never renumbered.
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

_CB_DIAGNOSTIC_NAMES = {
    CB_R_REAL: "CB_R_REAL",
    CB_R_NEG_IMAG: "CB_R_NEG_IMAG",
    CB_R_IMAG: "CB_R_IMAG",
    CB_X0_REAL: "CB_X0_REAL",
    CB_X0_IMAG: "CB_X0_IMAG",
    CB_IDENTITY: "CB_IDENTITY",
    CB_ZERO: "CB_ZERO",
    CB_STATE_REAL: "CB_STATE_REAL",
    CB_STATE_IMAG: "CB_STATE_IMAG",
    CB_S_REAL: "CB_S_REAL",
    CB_S_IMAG: "CB_S_IMAG",
    CB_PRODUCT_REAL: "CB_PRODUCT_REAL",
    CB_PRODUCT_IMAG: "CB_PRODUCT_IMAG",
    CB_NEG_X_IMAG: "CB_NEG_X_IMAG",
    CB_R_NEG_REAL: "CB_R_NEG_REAL",
    CB_OUTPUT_REAL: "CB_OUTPUT_REAL",
    CB_OUTPUT_IMAG: "CB_OUTPUT_IMAG",
    CB_PROFILE_READER: "CB_PROFILE_READER",
    CB_PROFILE_COMPUTE: "CB_PROFILE_COMPUTE",
    CB_PROFILE_WRITER: "CB_PROFILE_WRITER",
}

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
PROFILE_BLOCK_INPUT_CB_WAIT_OFFSET = 12
PROFILE_BLOCK_OUTPUT_CB_WAIT_OFFSET = 13
PROFILE_BLOCK_INPUT_CB_RESERVE_OFFSET = 14
PROFILE_BLOCK_OUTPUT_CB_RESERVE_OFFSET = 15
PROFILE_BLOCK_DEST_ACQUIRE_WAIT_OFFSET = 16
PROFILE_BLOCK_DEST_PACK_WAIT_OFFSET = 17


def _canonicalize_matrices(
    matrices: np.ndarray,
    *,
    allow_empty_batch: bool = False,
    check_norm: bool = True,
) -> np.ndarray:
    """Canonicalize and validate batched square matrices without importing spec."""
    values = np.asarray(matrices)
    if (
        values.ndim != 3
        or values.shape[0] < (0 if allow_empty_batch else 1)
        or values.shape[1] < 1
        or values.shape[1] != values.shape[2]
    ):
        raise ValueError("matrices must have shape (batch, size, size)")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            canonical = values.astype(np.complex64, copy=False)
    except (TypeError, ValueError, RuntimeWarning) as exc:
        raise ValueError("matrices must be numeric and finite after conversion") from exc
    if not np.all(np.isfinite(canonical)):
        raise ValueError("matrices must be finite")
    if check_norm and canonical.shape[0]:
        norm_inf = np.linalg.norm(canonical, ord=np.inf, axis=(-2, -1))
        if not np.all(np.isfinite(norm_inf)) or np.any(norm_inf <= 0.0):
            raise ValueError("matrices must have finite, non-zero infinity norms")
    return canonical


def _validate_iterations(iterations: int, *, fixed: bool = False) -> int:
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(
        iterations, (int, np.integer)
    ):
        raise TypeError("iterations must be a non-negative integer")
    if iterations < 0:
        raise ValueError("iterations must not be negative")
    if fixed and iterations != NEWTON_SCHULZ_ITERATIONS:
        raise ValueError(
            f"the kernel is fixed at {NEWTON_SCHULZ_ITERATIONS} iterations, got {iterations}"
        )
    return int(iterations)


def _initial_value(matrices: np.ndarray) -> np.ndarray:
    """Return X0 = I / ||R||_inf without depending on the NumPy oracle."""
    matrices = _canonicalize_matrices(matrices)
    norm_inf = np.linalg.norm(matrices, ord=np.inf, axis=(-2, -1))
    identity = np.eye(matrices.shape[-1], dtype=np.complex64)
    return identity[None, :, :] / norm_inf[:, None, None]


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


def _dest_slots_required(matrix_block: int) -> int:
    """Return the DEST slots retained by one matrix block."""
    if matrix_block not in MATRIX_BLOCK_CHOICES:
        raise ValueError(f"matrix_block must be one of {MATRIX_BLOCK_CHOICES}, got {matrix_block!r}")
    # Blocks 1/2/4 use two DEST tiles per matrix (one for each complex half).
    # Block 8 is executed one half at a time and therefore retains one tile per
    # matrix.
    return (1 if matrix_block == 8 else 2) * matrix_block


def _validate_matrix_block(
    matrix_block: int,
    *,
    fp32_dest_acc_en: bool = True,
    dst_full_sync_en: bool = True,
    variant: str | None = None,
) -> None:
    """Validate block, variant, and the host-modeled DEST footprint."""
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
    required_slots = _dest_slots_required(matrix_block)
    available_slots = _dest_slot_limit(
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
    )
    if required_slots > available_slots:
        raise ValueError(
            f"matrix_block={matrix_block} requires {required_slots} DEST slots, "
            f"but the selected DEST configuration provides {available_slots}"
        )


def _validate_memory(memory: str, *, name: str) -> None:
    """Reject an input placement value before any device work can begin."""
    if memory not in INPUT_MEMORY_CHOICES:
        raise ValueError(
            f"{name} must be one of {INPUT_MEMORY_CHOICES}, got {memory!r}"
        )


def _validate_input_memory(input_memory: str) -> None:
    """Reject the compatibility input placement value."""
    _validate_memory(input_memory, name="input_memory")


def _resolve_input_memories(
    input_memory: str = "l1",
    *,
    r_memory: str | None = None,
    x0_memory: str | None = None,
) -> tuple[str, str, str]:
    """Resolve per-tensor placement with the legacy shorthand as fallback.

    ``input_memory`` also remains the placement for the resident identity and
    zero tensors.  Explicit R/X0 values override it only for their tensor
    groups; this preserves the old all-input behavior for compatibility calls.
    """
    if input_memory is None:
        input_memory = "l1"
    _validate_input_memory(input_memory)
    if r_memory is None:
        r_memory = input_memory
    else:
        _validate_memory(r_memory, name="r_memory")
    if x0_memory is None:
        x0_memory = input_memory
    else:
        _validate_memory(x0_memory, name="x0_memory")
    return input_memory, r_memory, x0_memory


def _input_memory_config(ttnn, input_memory: str, *, name: str = "input_memory"):
    """Return an interleaved device memory config for one input group."""
    _validate_memory(input_memory, name=name)
    return ttnn.L1_MEMORY_CONFIG if input_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG


def _physical_tile_count(batch: int, size: int) -> int:
    """Return the number of 32x32 tiles needed for a logical matrix batch."""
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    if size not in (16, _TILE):
        raise ValueError(f"logical matrix size must be 16 or {_TILE}, got {size}")
    return (batch + 1) // 2 if size == 16 else batch


def _padded_tile_count(tile_count: int, matrix_block: int) -> int:
    """Round a physical tile count up to a complete matrix block."""
    if tile_count < 1:
        raise ValueError(f"tile_count must be positive, got {tile_count}")
    _validate_matrix_block(matrix_block)
    return ((tile_count + matrix_block - 1) // matrix_block) * matrix_block


def _pack_matrices(matrices: np.ndarray, *, packed: bool, tile_count: int) -> np.ndarray:
    """Pad or diagonal-pack logical matrices into the 32x32 kernel tiles."""
    batch, size, _ = matrices.shape
    if packed:
        if size * 2 != _TILE:
            raise ValueError(f"diagonal packing requires logical size 16, got {size}")
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


def _reader_input_values(
    matrices: np.ndarray,
    x0: np.ndarray,
    *,
    fuse_s: bool,
    tile_count: int,
    packed: bool = False,
) -> list[np.ndarray]:
    """Build reader tensors in the exact runtime/accessor argument order."""
    r_imag_values = _pack_matrices(matrices.imag, packed=packed, tile_count=tile_count)
    r_negative_imag_values = -r_imag_values
    if fuse_s:
        r_inputs = [
            r_negative_imag_values,
            r_imag_values,
            -_pack_matrices(matrices.real, packed=packed, tile_count=tile_count),
        ]
    else:
        r_inputs = [
            _pack_matrices(matrices.real, packed=packed, tile_count=tile_count),
            r_negative_imag_values,
            r_imag_values,
        ]
    r_inputs.extend(
        [
            _pack_matrices(x0.real, packed=packed, tile_count=tile_count),
            _pack_matrices(x0.imag, packed=packed, tile_count=tile_count),
        ]
    )
    return r_inputs


def _reader_input_dtypes(ttnn, state_dtype, *, fuse_s: bool) -> list[Any]:
    """Return dtypes matching ``_reader_input_values`` and its constants."""
    return [
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.bfloat16,
        state_dtype,
        state_dtype,
        ttnn.bfloat16 if fuse_s else ttnn.float32,
        ttnn.float32,
    ]


def _reader_compile_args(
    *,
    iterations: int,
    profile: bool,
    fuse_s: bool,
    batch_reads: bool,
    matrix_block: int,
    reload_r: bool,
) -> list[int]:
    """Return reader compile arguments for the selected source ABI."""
    iterations = _validate_iterations(iterations)
    args = [iterations]
    if profile or fuse_s or batch_reads or matrix_block > 1 or reload_r:
        args.extend([int(fuse_s), int(batch_reads), matrix_block, int(reload_r)])
    return args


def _compute_compile_args(
    *,
    iterations: int,
    state_fp32: bool,
    profile: bool,
    fuse_s: bool,
    matrix_block: int,
    reload_r: bool,
) -> list[int]:
    """Return compute compile arguments shared by host and device kernels."""
    iterations = _validate_iterations(iterations)
    return [
        iterations,
        int(state_fp32),
        int(profile),
        int(fuse_s),
        matrix_block,
        int(reload_r),
    ]


def _reader_input_memories(
    *,
    input_memory: str = "l1",
    r_memory: str | None = None,
    x0_memory: str | None = None,
) -> list[str]:
    """Return memory placement in the exact host/reader tensor order.

    The five reader inputs are three R variants followed by the two X0
    halves.  Identity and zero retain the compatibility shorthand placement.
    """
    input_memory, r_memory, x0_memory = _resolve_input_memories(
        input_memory, r_memory=r_memory, x0_memory=x0_memory
    )
    return [r_memory] * 3 + [x0_memory] * 2 + [input_memory] * 2


def _device_tensor(
    ttnn,
    values: np.ndarray,
    device,
    *,
    dtype,
    input_memory: str = "l1",
):
    """Move one reader input to the selected interleaved memory."""
    memory_config = _input_memory_config(ttnn, input_memory)
    host = ttnn.Tensor(np.ascontiguousarray(values), dtype)
    tiled = ttnn.to_layout(host, ttnn.TILE_LAYOUT)
    return ttnn.to_device(tiled, device, memory_config=memory_config)


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


def _validate_core_group_capacities(
    work_ranges: list[tuple[int, int]], matrix_block: int
) -> None:
    """Reject core groups that do not divide every matrix-block CB capacity."""
    _validate_matrix_block(matrix_block)
    capacities = [("matrix", matrix_block)]
    if matrix_block == 8:
        capacities.append(("state", 2 * matrix_block))
    for core_index, (start, count) in enumerate(work_ranges):
        for group_start, group_count in _matrix_block_ranges(start, count, matrix_block):
            for name, capacity in capacities:
                if capacity % group_count != 0:
                    raise ValueError(
                        f"core group {core_index} at tile {group_start} has "
                        f"group_count={group_count}, which does not divide "
                        f"{name} CB capacity={capacity}"
                    )


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
    """Return the default output placement for either supported state variant."""
    return DEFAULT_OUTPUT_MEMORY


def _cb_page_size(ttnn, data_format) -> int:
    """Return one 32x32 page in bytes, including uint32 profile tiles."""
    uint32 = getattr(ttnn, "uint32", None)
    return _TILE_BYTES_FLOAT32 if data_format == ttnn.float32 or data_format == uint32 else _TILE_BYTES_BFLOAT16


def _matrix_queue_pages(matrix_block: int, *, double_buffer: bool) -> int:
    """Return the input/output CB capacity for the selected block window.

    Matrix block 1 already has a two-page queue and keeps that historical
    capacity.  Larger blocks use one block window by default and two windows
    when ``double_buffer`` is enabled; the compute-side state queues retain
    their source-derived capacities separately.
    """
    _validate_matrix_block(matrix_block)
    single_window_pages = 2 if matrix_block == 1 else matrix_block
    if double_buffer and matrix_block > 1:
        return 2 * single_window_pages
    return single_window_pages


def _cb_page_count(
    index: int,
    base_page_count: int,
    *,
    resident: set[int],
    double_buffered: set[int],
    matrix_block: int,
    matrix_queue_pages: int,
    state_queue_pages: int,
    fuse_s: bool,
) -> int:
    """Return the page count for one CB under the selected block window."""
    if fuse_s and index == CB_R_NEG_REAL:
        base_page_count = max(base_page_count, 2)
    if index in resident or matrix_block == 1:
        return base_page_count
    if fuse_s and index in {CB_PRODUCT_REAL, CB_PRODUCT_IMAG}:
        return base_page_count
    if index in {CB_STATE_REAL, CB_STATE_IMAG}:
        return max(base_page_count, state_queue_pages)
    if index in double_buffered:
        return max(base_page_count, matrix_queue_pages)
    return max(base_page_count, matrix_block)


def _cb_definitions(
    ttnn,
    state_dtype,
    *,
    profile: bool = False,
    fuse_s: bool = False,
    matrix_block: int = 1,
    double_buffer: bool = False,
) -> dict[int, tuple[Any, int]]:
    """Describe CB formats, optionally exposing two block input/output windows."""
    _validate_matrix_block(matrix_block)
    definitions = {
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
        CB_R_NEG_REAL: (ttnn.bfloat16, 1),
        CB_OUTPUT_REAL: (state_dtype, 2),
        CB_OUTPUT_IMAG: (state_dtype, 2),
    }
    if not fuse_s:
        # Preserve the baseline descriptor and its ABI index exactly.  Fused S
        # has no positive R-real reader input, so CB index 0 is intentionally
        # absent rather than renumbering any shared CB.
        definitions[CB_R_REAL] = (ttnn.bfloat16, 2)
        definitions = {CB_R_REAL: definitions.pop(CB_R_REAL), **definitions}
    if profile:
        definitions.update(
            {
                CB_PROFILE_READER: (ttnn.uint32, 1),
                CB_PROFILE_COMPUTE: (ttnn.uint32, 1),
                CB_PROFILE_WRITER: (ttnn.uint32, 1),
            }
        )
    # Identity/zero and profile pages are resident singletons.  S, product,
    # and negated-state queues are consumed within one compute block; only the
    # external R/X0 inputs and final outputs need a second producer/consumer
    # window.
    resident = {CB_IDENTITY, CB_ZERO, CB_PROFILE_READER, CB_PROFILE_COMPUTE, CB_PROFILE_WRITER}
    double_buffered = {
        CB_R_REAL,
        CB_R_NEG_IMAG,
        CB_R_IMAG,
        CB_R_NEG_REAL,
        CB_X0_REAL,
        CB_X0_IMAG,
        CB_OUTPUT_REAL,
        CB_OUTPUT_IMAG,
    }
    # The non-one-destination branch pops the current state block before it
    # reserves the reused state output, so blocks 2 and 4 need one window.
    # Block 8 reserves output before popping its current state for the two
    # DEST-half passes and therefore needs two windows.  The matrix_block == 1
    # path retains its original two-page descriptor.
    state_queue_pages = 2 * matrix_block if matrix_block == 8 else matrix_block
    matrix_queue_pages = _matrix_queue_pages(matrix_block, double_buffer=double_buffer)
    definitions = {
        index: (
            data_format,
            _cb_page_count(
                index,
                page_count,
                resident=resident,
                double_buffered=double_buffered,
                matrix_block=matrix_block,
                matrix_queue_pages=matrix_queue_pages,
                state_queue_pages=state_queue_pages,
                fuse_s=fuse_s,
            ),
        )
        for index, (data_format, page_count) in definitions.items()
    }
    return definitions


def _cb_l1_bytes_by_name(
    ttnn, definitions: dict[int, tuple[Any, int]]
) -> dict[str, int]:
    """Return each defined circular buffer's per-core footprint in bytes."""
    return {
        _CB_DIAGNOSTIC_NAMES.get(index, f"CB_{index}"): _cb_page_size(ttnn, data_format)
        * page_count
        for index, (data_format, page_count) in definitions.items()
    }


def _cb_l1_bytes(ttnn, definitions: dict[int, tuple[Any, int]]) -> int:
    """Return the per-core circular-buffer footprint in bytes."""
    return sum(_cb_l1_bytes_by_name(ttnn, definitions).values())


def _tensor_l1_bytes(
    ttnn,
    *,
    batch: int,
    core_count: int,
    state_dtype,
    fuse_s: bool,
    output_memory: str,
    input_memory: str = "l1",
    r_memory: str | None = None,
    x0_memory: str | None = None,
    matrix_block: int = 1,
) -> int:
    """Estimate the largest assigned core's tensor footprint.

    Reader inputs and the identity/zero constants are device tensors.  R and
    X0 use independent placements; the compatibility shorthand controls both
    groups when their explicit values are absent.  The resident constants
    continue to follow that shorthand.  Interleaved DRAM inputs contribute no
    tensor bytes here because their pages are fetched into the CBs.
    """
    input_memory, r_memory, x0_memory = _resolve_input_memories(
        input_memory, r_memory=r_memory, x0_memory=x0_memory
    )
    _validate_matrix_block(matrix_block)
    tiles_per_core = max(
        count for _, count in _balanced_ranges(batch, core_count, matrix_block)
    )
    r_bytes = 0
    if r_memory == "l1":
        # Fused S uses signed (-R_re, +R_im, -R_im) pages; the positive
        # R-real tensor is omitted.  The non-fused ABI also has three R pages.
        r_bytes = 3 * _cb_page_size(ttnn, ttnn.bfloat16)
    x0_bytes = 0
    if x0_memory == "l1":
        x0_bytes = 2 * _cb_page_size(ttnn, state_dtype)
    resident_bytes = 0
    if input_memory == "l1":
        identity_dtype = ttnn.bfloat16 if fuse_s else ttnn.float32
        resident_bytes = _cb_page_size(ttnn, identity_dtype) + _cb_page_size(ttnn, ttnn.float32)
    _validate_memory(output_memory, name="output_memory")
    output_bytes = 0
    if output_memory == "l1":
        output_bytes = 2 * _cb_page_size(ttnn, state_dtype)
    return tiles_per_core * (r_bytes + x0_bytes + output_bytes) + resident_bytes


def _format_cb_bytes(entries: list[tuple[str, int]]) -> str:
    """Format named CB byte counts, including the matching kernel name."""
    return ", ".join(
        f"{name}={byte_count} bytes ({name.lower()})" for name, byte_count in entries
    )


def _l1_budget_breakdown(
    ttnn,
    definitions: dict[int, tuple[Any, int]],
    *,
    tensor_bytes: int,
) -> dict[str, Any]:
    """Return all host-side L1 accounting fields used by the preflight error."""
    cb_bytes_by_name = _cb_l1_bytes_by_name(ttnn, definitions)
    cb_entries = sorted(cb_bytes_by_name.items(), key=lambda item: (-item[1], item[0]))
    cb_bytes = sum(cb_bytes_by_name.values())
    total_bytes = _L1_STATIC_BASE_BYTES + cb_bytes + tensor_bytes
    largest_bytes = cb_entries[0][1] if cb_entries else 0
    largest_cbs = [entry for entry in cb_entries if entry[1] == largest_bytes]
    cb_budget_overage = max(0, cb_bytes - _L1_CB_BUDGET_BYTES)
    total_budget_overage = max(0, total_bytes - _L1_TOTAL_BUDGET_BYTES)
    # No individual-CB limit exists; when the aggregate check fails, retain
    # every CB contribution so the caller can report which queues make up the
    # over-budget total rather than inventing a per-CB threshold.
    over_budget_cbs = (
        cb_entries if cb_budget_overage or total_budget_overage else []
    )
    return {
        "cb_bytes": cb_bytes,
        "cb_bytes_by_name": cb_bytes_by_name,
        "cb_entries": cb_entries,
        "static_prefix_bytes": _L1_STATIC_BASE_BYTES,
        "tensor_bytes": tensor_bytes,
        "total_bytes": total_bytes,
        "budget_bytes": _L1_TOTAL_BUDGET_BYTES,
        "largest_cbs": largest_cbs,
        "over_budget_cbs": over_budget_cbs,
        "cb_budget_overage": cb_budget_overage,
        "total_budget_overage": total_budget_overage,
    }


def _validate_l1_budget(
    ttnn,
    definitions: dict[int, tuple[Any, int]],
    *,
    tensor_bytes: int = 0,
    matrix_block: int | None = None,
) -> int:
    """Reject CBs plus tensors that cannot coexist in one Tensix L1."""
    breakdown = _l1_budget_breakdown(ttnn, definitions, tensor_bytes=tensor_bytes)
    if breakdown["cb_budget_overage"] or breakdown["total_budget_overage"]:
        largest = _format_cb_bytes(breakdown["largest_cbs"])
        over_budget = _format_cb_bytes(breakdown["over_budget_cbs"]) or "none"
        prefix = "" if matrix_block is None else f"matrix_block={matrix_block} "
        overages = []
        if breakdown["cb_budget_overage"]:
            overages.append(f"CB budget over by {breakdown['cb_budget_overage']} bytes")
        if breakdown["total_budget_overage"]:
            overages.append(f"L1 budget over by {breakdown['total_budget_overage']} bytes")
        raise ValueError(
            f"{prefix}L1 preflight failed: "
            f"total CB bytes={breakdown['cb_bytes']}, "
            f"static prefix={breakdown['static_prefix_bytes']} bytes, "
            f"tensor bytes={breakdown['tensor_bytes']}, "
            f"total={breakdown['total_bytes']} bytes, "
            f"budget={breakdown['budget_bytes']} bytes; "
            f"{'; '.join(overages)}; "
            f"largest CBs: {largest}; "
            f"CBs in over-budget total: {over_budget}; "
            f"CB breakdown: {_format_cb_bytes(breakdown['cb_entries'])}"
        )
    return breakdown["total_bytes"]


def _validate_l1_preflight(
    ttnn,
    *,
    batch: int,
    core_count: int,
    state_dtype,
    profile: bool = False,
    fuse_s: bool = DEFAULT_FUSE_S,
    output_memory: str = DEFAULT_OUTPUT_MEMORY,
    input_memory: str = "l1",
    r_memory: str | None = None,
    x0_memory: str | None = None,
    matrix_block: int = DEFAULT_MATRIX_BLOCK,
    double_buffer: bool = DEFAULT_DOUBLE_BUFFER,
    variant: str | None = DEFAULT_VARIANT,
    fp32_dest_acc_en: bool = DEFAULT_FP32_DEST_ACC_EN,
    dst_full_sync_en: bool = DEFAULT_DST_FULL_SYNC_EN,
) -> int:
    """Validate L1 usage without touching a device or allocating tensors.

    The Issue #100 defaults fail fast when this budget is exceeded; callers must
    select an explicit block or memory placement rather than receiving a fallback.
    """
    input_memory, r_memory, x0_memory = _resolve_input_memories(
        input_memory, r_memory=r_memory, x0_memory=x0_memory
    )
    if output_memory is None:
        output_memory = DEFAULT_OUTPUT_MEMORY
    _validate_memory(output_memory, name="output_memory")
    _validate_matrix_block(
        matrix_block,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
        variant=variant,
    )
    definitions = _cb_definitions(
        ttnn,
        state_dtype,
        profile=profile,
        fuse_s=fuse_s,
        matrix_block=matrix_block,
        double_buffer=double_buffer,
    )
    tensor_bytes = _tensor_l1_bytes(
        ttnn,
        batch=batch,
        core_count=core_count,
        state_dtype=state_dtype,
        fuse_s=fuse_s,
        output_memory=output_memory,
        input_memory=input_memory,
        r_memory=r_memory,
        x0_memory=x0_memory,
        matrix_block=matrix_block,
    )
    return _validate_l1_budget(
        ttnn,
        definitions,
        tensor_bytes=tensor_bytes,
        matrix_block=matrix_block,
    )


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
    if batch < 1 or size not in (16, _TILE):
        raise ValueError(f"benchmark inputs require a positive batch and size 16 or {_TILE}")
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((batch, size, size), dtype=np.float32)
    imag = rng.standard_normal((batch, size, size), dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


def _deallocate_tensors(ttnn, tensors: list[Any]) -> None:
    """Release prepared device tensors without masking the original failure."""
    for tensor in tensors:
        try:
            ttnn.deallocate(tensor)
        except Exception:  # noqa: BLE001, S110 - cleanup must not hide the result
            pass


@dataclass
class NewtonSchulzKernel:
    """Prepared tensors and one fixed-count ProgramDescriptor."""

    ttnn: Any
    device: Any
    batch: int
    size: int
    packed: bool
    variant: str
    math_fidelity: str
    input_memory: str
    r_memory: str
    x0_memory: str
    output_memory: str
    profile: bool
    fuse_s: bool
    batch_reads: bool
    reload_r: bool
    matrix_block: int
    double_buffer: bool
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
        variant: str = DEFAULT_VARIANT,
        math_fidelity: str = DEFAULT_MATH_FIDELITY,
        profile: bool = False,
        fuse_s: bool = DEFAULT_FUSE_S,
        batch_reads: bool = False,
        reload_r: bool = False,
        matrix_block: int = DEFAULT_MATRIX_BLOCK,
        double_buffer: bool = DEFAULT_DOUBLE_BUFFER,
        input_memory: str = "l1",
        r_memory: str | None = None,
        x0_memory: str | None = None,
        output_memory: str | None = DEFAULT_OUTPUT_MEMORY,
        fp32_dest_acc_en: bool = DEFAULT_FP32_DEST_ACC_EN,
        dst_full_sync_en: bool = DEFAULT_DST_FULL_SYNC_EN,
        iterations: int = NEWTON_SCHULZ_ITERATIONS,
    ) -> NewtonSchulzKernel:
        input_memory, r_memory, x0_memory = _resolve_input_memories(
            input_memory, r_memory=r_memory, x0_memory=x0_memory
        )
        iterations = _validate_iterations(iterations, fixed=True)
        if variant not in _SUPPORTED_VARIANTS:
            raise ValueError(f"unknown kernel variant {variant!r}")
        if output_memory is None:
            output_memory = _output_memory_name(variant)
        else:
            _validate_memory(output_memory, name="output_memory")
        _validate_matrix_block(
            matrix_block,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            variant=variant,
        )
        state_fp32 = variant == "bf16-fp32state"

        matrices = _canonicalize_matrices(
            matrices, allow_empty_batch=True, check_norm=False
        )
        batch, size, _ = matrices.shape
        if batch < 1:
            raise ValueError("batch must be positive")
        if size not in (16, _TILE):
            raise ValueError(f"the throughput kernel only supports L=16 or L={_TILE}, got {size}")
        packed = size == 16
        physical_tile_count = _physical_tile_count(batch, size)
        tile_count = _padded_tile_count(physical_tile_count, matrix_block)
        math_fidelity_value = _math_fidelity_value(ttnn, math_fidelity)

        coordinates, core_ranges, work_ranges = _core_grid(
            ttnn, device, tile_count, matrix_block
        )
        _validate_core_group_capacities(work_ranges, matrix_block)
        # Normalize each logical matrix before pair packing; a packed norm would
        # couple the two independent 16x16 matrices.
        x0 = _initial_value(matrices)
        input_values = _reader_input_values(
            matrices, x0, fuse_s=fuse_s, tile_count=tile_count, packed=packed
        )
        identity_values = (2.0 * np.eye(_TILE, dtype=np.float32))[None, None]
        zero_values = np.zeros((1, 1, _TILE, _TILE), dtype=np.float32)
        input_values.extend([identity_values, zero_values])
        state_dtype = _state_dtype(ttnn, variant)
        cb_definitions = _cb_definitions(
            ttnn,
            state_dtype,
            profile=profile,
            fuse_s=fuse_s,
            matrix_block=matrix_block,
            double_buffer=double_buffer,
        )
        tensor_l1_bytes = _tensor_l1_bytes(
            ttnn,
            batch=tile_count,
            core_count=len(work_ranges),
            state_dtype=state_dtype,
            fuse_s=fuse_s,
            output_memory=output_memory,
            input_memory=input_memory,
            r_memory=r_memory,
            x0_memory=x0_memory,
            matrix_block=matrix_block,
        )
        _validate_l1_budget(
            ttnn,
            cb_definitions,
            tensor_bytes=tensor_l1_bytes,
            matrix_block=matrix_block,
        )
        input_dtypes = _reader_input_dtypes(ttnn, state_dtype, fuse_s=fuse_s)
        input_memories = _reader_input_memories(
            input_memory=input_memory, r_memory=r_memory, x0_memory=x0_memory
        )
        allocated: list[Any] = []
        try:
            inputs: list[Any] = []
            for values, dtype, memory in zip(
                input_values, input_dtypes, input_memories, strict=True
            ):
                tensor = _device_tensor(
                    ttnn,
                    values,
                    device,
                    dtype=dtype,
                    input_memory=memory,
                )
                allocated.append(tensor)
                inputs.append(tensor)

            output_shape = ttnn.Shape((tile_count, 1, _TILE, _TILE))
            output_memory_config = (
                ttnn.DRAM_MEMORY_CONFIG if output_memory == "dram" else ttnn.L1_MEMORY_CONFIG
            )
            outputs: list[Any] = []
            for _ in range(2):
                tensor = ttnn.allocate_tensor_on_device(
                    output_shape,
                    state_dtype,
                    ttnn.TILE_LAYOUT,
                    device,
                    output_memory_config,
                )
                allocated.append(tensor)
                outputs.append(tensor)

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
                allocated.append(profile_output)
                outputs.append(profile_output)
        except Exception:
            _deallocate_tensors(ttnn, allocated)
            raise

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

        reader_compile_args = _reader_compile_args(
            iterations=iterations,
            profile=profile,
            fuse_s=fuse_s,
            batch_reads=batch_reads,
            matrix_block=matrix_block,
            reload_r=reload_r,
        )
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
                if fuse_s or batch_reads or matrix_block > 1 or reload_r
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
                compile_time_args=_compute_compile_args(
                    iterations=iterations,
                    state_fp32=state_fp32,
                    profile=profile,
                    fuse_s=fuse_s,
                    matrix_block=matrix_block,
                    reload_r=reload_r,
                ),
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
            packed=packed,
            variant=variant,
            math_fidelity=math_fidelity,
            input_memory=input_memory,
            r_memory=r_memory,
            x0_memory=x0_memory,
            output_memory=output_memory,
            profile=profile,
            fuse_s=fuse_s,
            batch_reads=batch_reads,
            reload_r=reload_r,
            matrix_block=matrix_block,
            double_buffer=double_buffer,
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
            ("block_external_input_cb_wait", PROFILE_BLOCK_INPUT_CB_WAIT_OFFSET),
            ("block_external_output_cb_wait", PROFILE_BLOCK_OUTPUT_CB_WAIT_OFFSET),
            ("block_external_input_cb_reserve", PROFILE_BLOCK_INPUT_CB_RESERVE_OFFSET),
            ("block_external_output_cb_reserve", PROFILE_BLOCK_OUTPUT_CB_RESERVE_OFFSET),
            ("block_dest_acquire_wait", PROFILE_BLOCK_DEST_ACQUIRE_WAIT_OFFSET),
            ("block_dest_pack_wait", PROFILE_BLOCK_DEST_PACK_WAIT_OFFSET),
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
        _deallocate_tensors(self.ttnn, [*self.inputs, *self.outputs])


def run_newton_schulz_kernel(
    ttnn,
    device,
    matrices: np.ndarray,
    *,
    variant: str = DEFAULT_VARIANT,
    math_fidelity: str = DEFAULT_MATH_FIDELITY,
    profile: bool = False,
    fuse_s: bool = DEFAULT_FUSE_S,
    batch_reads: bool = False,
    reload_r: bool = False,
    matrix_block: int = DEFAULT_MATRIX_BLOCK,
    double_buffer: bool = DEFAULT_DOUBLE_BUFFER,
    input_memory: str = "l1",
    r_memory: str | None = None,
    x0_memory: str | None = None,
    output_memory: str | None = DEFAULT_OUTPUT_MEMORY,
    fp32_dest_acc_en: bool = DEFAULT_FP32_DEST_ACC_EN,
    dst_full_sync_en: bool = DEFAULT_DST_FULL_SYNC_EN,
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
        reload_r=reload_r,
        matrix_block=matrix_block,
        double_buffer=double_buffer,
        input_memory=input_memory,
        r_memory=r_memory,
        x0_memory=x0_memory,
        output_memory=output_memory,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
    )
    try:
        kernel.launch()
        ttnn.synchronize_device(device)
        return kernel.result()
    finally:
        kernel.close()
