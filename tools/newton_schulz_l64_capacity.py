"""Board-free L=64 L1-capacity estimate for Issue #63 Scope 5.

This is an accounting aid, not a kernel or a production dispatch path.  It
reuses the current fused-S circular-buffer descriptors and static L1 prefix,
then applies the explicitly stated L=64 tile geometry to a hypothetical
layout.  The estimate keeps each logical matrix's four physical tiles on one
core and treats ``matrix_block`` as a group of logical matrices.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enodia.tt.bench import newton_schulz_kernel as kernel

CORE_COUNT = 110
LOGICAL_BATCH = 8192
L64_TILES_PER_MATRIX = 4
L64_K_TILES = 2
STATE_DTYPE = "fp32"


@dataclass(frozen=True)
class CapacityRow:
    """One placement and matrix-block row in the L=64 estimate."""

    matrix_block: int
    placement: str
    max_logical_matrices_per_core: int
    max_physical_tiles_per_core: int
    cb_bytes: int
    static_prefix_bytes: int
    tensor_bytes: int
    total_bytes: int
    budget_bytes: int
    headroom_bytes: int

    @property
    def fits(self) -> bool:
        """Whether the modeled total is within the current L1 budget."""
        return self.total_bytes <= self.budget_bytes


@dataclass(frozen=True)
class _Placement:
    name: str
    r_l1: bool
    x0_l1: bool
    constants_l1: bool


_PLACEMENTS = (
    _Placement("all-L1", r_l1=True, x0_l1=True, constants_l1=True),
    _Placement(
        "R-L1/X0-DRAM", r_l1=True, x0_l1=False, constants_l1=True
    ),
    _Placement(
        "R-DRAM/X0-L1", r_l1=False, x0_l1=True, constants_l1=True
    ),
    _Placement("all-DRAM", r_l1=False, x0_l1=False, constants_l1=False),
)


def _ttnn_stub() -> SimpleNamespace:
    """Provide the dtype names needed by the current host accounting."""
    return SimpleNamespace(bfloat16="bf16", float32="fp32")


def _max_aligned_assignment(matrix_block: int) -> tuple[int, int]:
    """Return max logical and physical tiles assigned to one core.

    The current range splitter is applied to logical matrices here so that a
    hypothetical L=64 matrix is never split across cores.  Physical tile
    counts are then expanded by its 2x2 32x32 layout.
    """
    ranges = kernel._balanced_ranges(LOGICAL_BATCH, CORE_COUNT, matrix_block)
    max_logical = max(count for _, count in ranges)
    return max_logical, max_logical * L64_TILES_PER_MATRIX


def _l64_cb_bytes(matrix_block: int) -> int:
    """Scale active current CB pages over L=64's four physical tiles.

    Fused S has no positive R-real CB.  Identity, zero, and the two fused
    product descriptors remain one-page compile-time/resident entries.  Every
    other current queue is a matrix-tile queue, so its current page count is
    multiplied by four.  K=2 is streamed through DEST and does not add a
    second resident CB window in this estimate.
    """
    ttnn = _ttnn_stub()
    definitions = kernel._cb_definitions(
        ttnn, STATE_DTYPE, fuse_s=True, matrix_block=matrix_block
    )
    unscaled = {
        kernel.CB_IDENTITY,
        kernel.CB_ZERO,
        kernel.CB_PRODUCT_REAL,
        kernel.CB_PRODUCT_IMAG,
    }
    total = 0
    for index, (data_format, page_count) in definitions.items():
        pages = page_count if index in unscaled else page_count * L64_TILES_PER_MATRIX
        total += kernel._cb_page_size(ttnn, data_format) * pages
    return total


def _tensor_bytes(max_physical_tiles: int, placement: _Placement) -> int:
    """Return per-core device-tensor bytes for one placement row."""
    ttnn = _ttnn_stub()
    r_bytes = 3 * kernel._cb_page_size(ttnn, ttnn.bfloat16)
    x0_bytes = 2 * kernel._cb_page_size(ttnn, ttnn.float32)
    resident_bytes = (
        kernel._cb_page_size(ttnn, ttnn.bfloat16)
        + kernel._cb_page_size(ttnn, ttnn.float32)
        if placement.constants_l1
        else 0
    )
    return (
        max_physical_tiles * (r_bytes if placement.r_l1 else 0)
        + max_physical_tiles * (x0_bytes if placement.x0_l1 else 0)
        + resident_bytes
    )


def capacity_rows() -> tuple[CapacityRow, ...]:
    """Build the reproducible L=64 capacity table."""
    rows: list[CapacityRow] = []
    for matrix_block in (1, 4):
        max_logical, max_physical = _max_aligned_assignment(matrix_block)
        cb_bytes = _l64_cb_bytes(matrix_block)
        for placement in _PLACEMENTS:
            tensor_bytes = _tensor_bytes(max_physical, placement)
            total_bytes = kernel._L1_STATIC_BASE_BYTES + cb_bytes + tensor_bytes
            budget_bytes = kernel._L1_TOTAL_BUDGET_BYTES
            rows.append(
                CapacityRow(
                    matrix_block=matrix_block,
                    placement=placement.name,
                    max_logical_matrices_per_core=max_logical,
                    max_physical_tiles_per_core=max_physical,
                    cb_bytes=cb_bytes,
                    static_prefix_bytes=kernel._L1_STATIC_BASE_BYTES,
                    tensor_bytes=tensor_bytes,
                    total_bytes=total_bytes,
                    budget_bytes=budget_bytes,
                    headroom_bytes=budget_bytes - total_bytes,
                )
            )
    return tuple(rows)


def render_markdown() -> str:
    """Render the table for review and documentation copy/paste."""
    lines = [
        "| matrix_block | placement | max logical/core | max physical tiles/core | CB bytes | tensor bytes | static prefix | total | budget | boundary |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in capacity_rows():
        boundary = "fits" if row.fits else f"does not fit (+{row.total_bytes - row.budget_bytes:,})"
        lines.append(
            f"| {row.matrix_block} | {row.placement} | "
            f"{row.max_logical_matrices_per_core} | {row.max_physical_tiles_per_core} | "
            f"{row.cb_bytes:,} | {row.tensor_bytes:,} | {row.static_prefix_bytes:,} | "
            f"{row.total_bytes:,} | {row.budget_bytes:,} | {boundary} |"
        )
    return "\n".join(lines)


def main() -> int:
    """Print the host-only estimate without opening a device."""
    print(render_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
