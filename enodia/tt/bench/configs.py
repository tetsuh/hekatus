"""Host-only catalogue of explicit stock ``ttnn.matmul`` configurations.

The catalogue contains plain data rather than ``ttnn`` objects so its shape
rules can be reviewed and tested without an accelerator or toolchain image.
All dimensions below are in 32 x 32 tiles, matching the pinned ttnn 0.70.1
program-config API.

The p150a grid and DRAM-bank count are part of this measurement catalogue.
Changing either changes which configurations are admissible and therefore
requires a new measurement record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

from enodia.tt.bench.shapes import MatmulShape

TILE_SIZE = 32
P150_COMPUTE_GRID = (11, 10)
P150_DRAM_BANKS = 8
# The board has about 1.5 MB of L1 per Tensix (design.md section 2). Keep
# explicit reuse configs below 1.3 MB so runtime and allocator-reserved space
# are not mistaken for circular-buffer capacity.
P150_MATMUL_CB_BUDGET_BYTES = 1_300_000
_TILE_BYTES = {
    "bfloat16": TILE_SIZE * TILE_SIZE * 2,
    "float32": TILE_SIZE * TILE_SIZE * 4,
}
_MAX_SUBBLOCK_TILES = 4  # valid for both BF16 and FP32 destination accumulation

_CONFIG_KINDS = {
    "reuse",
    "mcast_1d",
    "mcast_2d",
    "dram_sharded",
    "batched_dram_sharded",
}


@dataclass(frozen=True)
class ProgramConfigSpec:
    """One explicit program configuration and the tensor arrangement it needs.

    ``grid`` is the compute grid for the reuse and multicast variants. For
    DRAM-sharded variants it records the storage grid: DRAM banks for batched
    sharding, or the L1 width-shard grid for the unbatched variant. The batched
    DRAM implementation chooses its non-rectangular worker cores from the
    device's DRAM-bank mapping at runtime.
    """

    name: str
    kind: str
    grid: tuple[int, int]
    in0_block_w: int
    out_subblock_h: int
    out_subblock_w: int
    per_core_m: int
    per_core_n: int
    out_block_h: int | None = None
    out_block_w: int | None = None
    transpose_mcast: bool = False
    fuse_batch: bool = False
    mcast_in0: bool | None = None
    memory_plan: str = "interleaved"
    batch_multiple: int = 1


def _tiles(value: int) -> int:
    return ceil(value / TILE_SIZE)


def _largest_divisor_at_most(value: int, limit: int) -> int:
    return next(
        candidate for candidate in range(min(value, limit), 0, -1) if value % candidate == 0
    )


def _subblock(block_h: int, block_w: int) -> tuple[int, int]:
    """Largest portable subblock that divides the output block."""
    choices = ((4, 1), (2, 2), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1))
    return next((h, w) for h, w in choices if block_h % h == 0 and block_w % w == 0)


def _block_widths(k_tiles: int) -> tuple[int, ...]:
    return (1,) if k_tiles == 1 else (1, k_tiles)


def _tile_bytes(dtype: str) -> int:
    try:
        return _TILE_BYTES[dtype]
    except KeyError:
        supported = ", ".join(sorted(_TILE_BYTES))
        raise ValueError(f"unsupported matmul dtype {dtype!r}; choose from {supported}") from None


def _reuse_cb_bytes(
    per_core_m: int,
    per_core_n: int,
    in0_block_w: int,
    dtype: str = "bfloat16",
) -> int:
    """Pinned ttnn's double-buffered input plus output/intermediate CB estimate."""
    tiles = 2 * (per_core_m * in0_block_w + per_core_n * in0_block_w + per_core_m * per_core_n)
    return tiles * _tile_bytes(dtype)


def _batched_dram_l1_bytes(shape: MatmulShape, dtype: str = "bfloat16") -> int:
    batch = ceil(shape.batch / P150_DRAM_BANKS) * P150_DRAM_BANKS
    batches_per_core = batch // P150_DRAM_BANKS
    m_tiles, k_tiles, n_tiles = _tiles(shape.m), _tiles(shape.k), _tiles(shape.n)
    resident_tiles = batches_per_core * m_tiles * (k_tiles + n_tiles)
    return resident_tiles * _tile_bytes(dtype) + _reuse_cb_bytes(
        m_tiles, n_tiles, k_tiles, dtype
    )


def _reuse_configs(shape: MatmulShape, dtype: str = "bfloat16") -> list[ProgramConfigSpec]:
    m_tiles, k_tiles, n_tiles = _tiles(shape.m), _tiles(shape.k), _tiles(shape.n)
    output_blocks = _largest_divisor_at_most(m_tiles * n_tiles, 64)
    m_blocks = _largest_divisor_at_most(m_tiles, output_blocks)
    n_blocks = output_blocks // m_blocks
    if n_tiles % n_blocks:
        n_blocks = _largest_divisor_at_most(n_tiles, output_blocks // m_blocks)
    per_m, per_n = m_tiles // m_blocks, n_tiles // n_blocks
    grid = (min(8, output_blocks), ceil(output_blocks / min(8, output_blocks)))

    partitions = [(m_tiles, n_tiles, (1, 1)), (per_m, per_n, grid)]
    configs = []
    for per_core_m, per_core_n, candidate_grid in dict.fromkeys(partitions):
        if per_core_n != n_tiles:
            continue
        sub_h, sub_w = _subblock(per_core_m, per_core_n)
        for block_w in _block_widths(k_tiles):
            if (
                _reuse_cb_bytes(per_core_m, per_core_n, block_w, dtype)
                > P150_MATMUL_CB_BUDGET_BYTES
            ):
                continue
            configs.append(
                ProgramConfigSpec(
                    name=(
                        f"reuse_g{candidate_grid[0]}x{candidate_grid[1]}_k{block_w}"
                        f"_m{per_core_m}_n{per_core_n}_s{sub_h}x{sub_w}"
                    ),
                    kind="reuse",
                    grid=candidate_grid,
                    in0_block_w=block_w,
                    out_subblock_h=sub_h,
                    out_subblock_w=sub_w,
                    per_core_m=per_core_m,
                    per_core_n=per_core_n,
                )
            )
    return configs


def _mcast_1d_configs(shape: MatmulShape) -> list[ProgramConfigSpec]:
    if shape.batch != 1:
        return []
    m_tiles, k_tiles, n_tiles = _tiles(shape.m), _tiles(shape.k), _tiles(shape.n)
    wide = n_tiles >= m_tiles
    split_tiles = n_tiles if wide else m_tiles
    cores = _largest_divisor_at_most(split_tiles, 64)
    grid_x = _largest_divisor_at_most(cores, P150_COMPUTE_GRID[0])
    grid_y = cores // grid_x
    if grid_y > P150_COMPUTE_GRID[1]:
        return []
    grid = (grid_x, grid_y)
    per_m = m_tiles if wide else m_tiles // cores
    per_n = n_tiles // cores if wide else n_tiles
    block_h, block_w = min(per_m, 4), min(per_n, 4)
    block_h = _largest_divisor_at_most(per_m, block_h)
    block_w = _largest_divisor_at_most(per_n, block_w)
    sub_h, sub_w = _subblock(block_h, block_w)

    return [
        ProgramConfigSpec(
            name=(
                f"mcast1d_{'in0' if wide else 'in1'}_g{grid[0]}x{grid[1]}_k{inner}"
                f"_m{per_m}_n{per_n}_b{block_h}x{block_w}_s{sub_h}x{sub_w}"
            ),
            kind="mcast_1d",
            grid=grid,
            in0_block_w=inner,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            out_block_h=block_h,
            out_block_w=block_w,
            per_core_m=per_m,
            per_core_n=per_n,
            fuse_batch=True,
            mcast_in0=wide,
        )
        for inner in _block_widths(k_tiles)
    ]


def _mcast_2d_configs(shape: MatmulShape) -> list[ProgramConfigSpec]:
    if shape.batch != 1:
        return []
    m_tiles, k_tiles, n_tiles = _tiles(shape.m), _tiles(shape.k), _tiles(shape.n)
    grid_x = _largest_divisor_at_most(n_tiles, P150_COMPUTE_GRID[0])
    grid_y = _largest_divisor_at_most(m_tiles, P150_COMPUTE_GRID[1])
    per_m, per_n = m_tiles // grid_y, n_tiles // grid_x
    block_h = _largest_divisor_at_most(per_m, min(per_m, 4))
    block_w = _largest_divisor_at_most(per_n, min(per_n, 4))
    sub_h, sub_w = _subblock(block_h, block_w)

    return [
        ProgramConfigSpec(
            name=(
                f"mcast2d_g{grid_x}x{grid_y}_k{k_tiles}_m{per_m}_n{per_n}"
                f"_b{block_h}x{block_w}_s{sub_h}x{sub_w}"
            ),
            kind="mcast_2d",
            grid=(grid_x, grid_y),
            in0_block_w=k_tiles,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            out_block_h=block_h,
            out_block_w=block_w,
            per_core_m=per_m,
            per_core_n=per_n,
            fuse_batch=True,
        )
    ]


def _dram_sharded_configs(shape: MatmulShape, dtype: str = "bfloat16") -> list[ProgramConfigSpec]:
    m_tiles, k_tiles, n_tiles = _tiles(shape.m), _tiles(shape.k), _tiles(shape.n)
    if shape.batch != 1 or m_tiles != 1:
        return []
    storage_cores = _largest_divisor_at_most(k_tiles, min(k_tiles, P150_DRAM_BANKS))
    per_n = n_tiles // _largest_divisor_at_most(n_tiles, storage_cores)
    if _reuse_cb_bytes(1, per_n, 1, dtype) > P150_MATMUL_CB_BUDGET_BYTES:
        return []
    sub_h, sub_w = _subblock(1, per_n)
    return [
        ProgramConfigSpec(
            name=f"dram_sharded_g{storage_cores}x1_k1_m1_n{per_n}_s{sub_h}x{sub_w}",
            kind="dram_sharded",
            grid=(storage_cores, 1),
            in0_block_w=1,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_m=1,
            per_core_n=per_n,
            memory_plan="width_sharded_dram",
        )
    ]


def _batched_dram_sharded_configs(
    shape: MatmulShape, dtype: str = "bfloat16"
) -> list[ProgramConfigSpec]:
    if shape.batch == 1 or _batched_dram_l1_bytes(shape, dtype) > P150_MATMUL_CB_BUDGET_BYTES:
        return []
    m_tiles, k_tiles, n_tiles = _tiles(shape.m), _tiles(shape.k), _tiles(shape.n)
    sub_h, sub_w = _subblock(m_tiles, n_tiles)
    return [
        ProgramConfigSpec(
            name=(
                f"batched_dram_sharded_g{P150_DRAM_BANKS}x1_k{k_tiles}"
                f"_m{m_tiles}_n{n_tiles}_s{sub_h}x{sub_w}"
            ),
            kind="batched_dram_sharded",
            grid=(P150_DRAM_BANKS, 1),
            in0_block_w=k_tiles,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_m=m_tiles,
            per_core_n=n_tiles,
            memory_plan="batch_sharded_dram",
            batch_multiple=P150_DRAM_BANKS,
        )
    ]


def validate_config(
    shape: MatmulShape, config: ProgramConfigSpec, dtype: str = "bfloat16"
) -> None:
    """Raise ``ValueError`` before a config can reach ttnn's fatal checks."""
    if config.kind not in _CONFIG_KINDS:
        raise ValueError(f"unknown program config kind: {config.kind}")
    grid_x, grid_y = config.grid
    if not (1 <= grid_x <= P150_COMPUTE_GRID[0] and 1 <= grid_y <= P150_COMPUTE_GRID[1]):
        raise ValueError(
            f"{config.name}: compute grid {config.grid} exceeds p150 grid {P150_COMPUTE_GRID}"
        )

    values = {
        "in0_block_w": config.in0_block_w,
        "out_subblock_h": config.out_subblock_h,
        "out_subblock_w": config.out_subblock_w,
        "per_core_m": config.per_core_m,
        "per_core_n": config.per_core_n,
        "batch_multiple": config.batch_multiple,
    }
    if any(value < 1 for value in values.values()):
        raise ValueError(f"{config.name}: config values must be positive: {values}")

    m_tiles, k_tiles, n_tiles = _tiles(shape.m), _tiles(shape.k), _tiles(shape.n)
    if k_tiles % config.in0_block_w:
        raise ValueError(
            f"{config.name}: K tiles {k_tiles} must be divisible by in0 block {config.in0_block_w}"
        )
    if config.per_core_m % config.out_subblock_h:
        raise ValueError(
            f"{config.name}: per-core M {config.per_core_m} must be divisible by subblock height "
            f"{config.out_subblock_h}"
        )
    if config.per_core_n % config.out_subblock_w:
        raise ValueError(
            f"{config.name}: per-core N {config.per_core_n} must be divisible by subblock width "
            f"{config.out_subblock_w}"
        )
    if config.out_subblock_h * config.out_subblock_w > _MAX_SUBBLOCK_TILES:
        raise ValueError(
            f"{config.name}: subblock has more than {_MAX_SUBBLOCK_TILES} destination tiles"
        )

    if config.kind == "reuse":
        if config.per_core_n != n_tiles:
            raise ValueError(
                f"{config.name}: reuse per-core N {config.per_core_n} must cover all {n_tiles} N tiles"
            )
        if m_tiles % config.per_core_m or n_tiles % config.per_core_n:
            raise ValueError(
                f"{config.name}: M tiles {m_tiles} and N tiles {n_tiles} must divide into per-core blocks"
            )
        blocks = (m_tiles // config.per_core_m) * (n_tiles // config.per_core_n)
        if blocks > grid_x * grid_y:
            raise ValueError(
                f"{config.name}: {blocks} output blocks exceed compute grid {config.grid}"
            )
        cb_bytes = _reuse_cb_bytes(
            config.per_core_m, config.per_core_n, config.in0_block_w, dtype
        )
        if cb_bytes > P150_MATMUL_CB_BUDGET_BYTES:
            raise ValueError(
                f"{config.name}: circular buffers need {cb_bytes} bytes, above the p150 budget"
            )
        return

    if config.kind in {"mcast_1d", "mcast_2d"}:
        if shape.batch != 1:
            raise ValueError(
                f"{config.name}: multicast cannot consume independently batched operands"
            )
        block_h = config.out_block_h or config.per_core_m
        block_w = config.out_block_w or config.per_core_n
        if config.per_core_m % block_h or config.per_core_n % block_w:
            raise ValueError(f"{config.name}: output block must divide the per-core output")
        if block_h % config.out_subblock_h or block_w % config.out_subblock_w:
            raise ValueError(f"{config.name}: subblock must divide the output block")
        if config.kind == "mcast_2d":
            if config.per_core_m * grid_y != m_tiles or config.per_core_n * grid_x != n_tiles:
                raise ValueError(f"{config.name}: 2D grid does not cover M and N tiles exactly")
        elif config.mcast_in0:
            if config.per_core_m != m_tiles or config.per_core_n * grid_x * grid_y != n_tiles:
                raise ValueError(
                    f"{config.name}: 1D in0 multicast does not cover output tiles exactly"
                )
        elif config.per_core_n != n_tiles or config.per_core_m * grid_x * grid_y != m_tiles:
            raise ValueError(f"{config.name}: 1D in1 multicast does not cover output tiles exactly")
        return

    if config.kind == "dram_sharded":
        if shape.batch != 1 or m_tiles != 1 or config.per_core_m != 1:
            raise ValueError(
                f"{config.name}: DRAM sharding requires an unbatched one-tile-high output"
            )
        cb_bytes = _reuse_cb_bytes(
            config.per_core_m, config.per_core_n, config.in0_block_w, dtype
        )
        if cb_bytes > P150_MATMUL_CB_BUDGET_BYTES:
            raise ValueError(
                f"{config.name}: circular buffers need {cb_bytes} bytes, above the p150 budget"
            )
        return

    if shape.batch == 1:
        raise ValueError(f"{config.name}: batched DRAM sharding requires batched operands")
    if config.per_core_m != m_tiles or config.per_core_n != n_tiles:
        raise ValueError(f"{config.name}: each DRAM worker must execute complete batched matrices")
    if config.batch_multiple != P150_DRAM_BANKS:
        raise ValueError(f"{config.name}: batch padding must match the p150 DRAM-bank count")
    l1_bytes = _batched_dram_l1_bytes(shape, dtype)
    if l1_bytes > P150_MATMUL_CB_BUDGET_BYTES:
        raise ValueError(
            f"{config.name}: sharded input, output, and circular buffers need {l1_bytes} bytes"
        )


def executed_shape(shape: MatmulShape, config: ProgramConfigSpec) -> MatmulShape:
    """The padded arithmetic a configuration actually sends to the matrix engine."""
    batch = ceil(shape.batch / config.batch_multiple) * config.batch_multiple
    return replace(shape, batch=batch)


def configuration_catalogue(
    shape: MatmulShape, dtype: str = "bfloat16"
) -> list[ProgramConfigSpec]:
    """Explicit stock configurations admissible for ``shape`` and ``dtype`` on p150a."""
    configs = [
        *_reuse_configs(shape, dtype),
        *_mcast_1d_configs(shape),
        *_mcast_2d_configs(shape),
        *_dram_sharded_configs(shape, dtype),
        *_batched_dram_sharded_configs(shape, dtype),
    ]
    for config in configs:
        validate_config(shape, config, dtype)
    return configs
