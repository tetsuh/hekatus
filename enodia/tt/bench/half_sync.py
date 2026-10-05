"""Host-side manifest and preflight helpers for the Issue #96 sweep."""

from __future__ import annotations

from enodia.tt.bench.newton_schulz_kernel import (
    MATRIX_BLOCK_CHOICES,
    NEWTON_SCHULZ_ITERATIONS,
    _dest_slot_limit,
    _dest_slots_required,
    _padded_tile_count,
    _physical_tile_count,
    _state_dtype,
    _validate_l1_preflight,
)

HALF_SYNC_SIZES = (32, 16)
HALF_SYNC_VARIANTS = ("bf16-fp32state", "bf16")
HALF_SYNC_MODES = (True, False)
HALF_SYNC_BATCH = 8192
HALF_SYNC_CORRECTNESS_BATCHES = (4, 8192)
HALF_SYNC_LAUNCHES = 1000
HALF_SYNC_MATH_FIDELITY = "HiFi3"
HALF_SYNC_CORE_COUNT = 110


def _row_name(size: int, variant: str, dst_full_sync_en: bool, matrix_block: int) -> str:
    sync_name = "full_sync" if dst_full_sync_en else "half_sync"
    return f"L{size}_b{HALF_SYNC_BATCH}_{variant}_{sync_name}_block{matrix_block}"


def _dest_decision(matrix_block: int, dst_full_sync_en: bool) -> dict:
    available_slots = _dest_slot_limit(
        fp32_dest_acc_en=True,
        dst_full_sync_en=dst_full_sync_en,
    )
    required_slots = _dest_slots_required(matrix_block)
    admitted = required_slots <= available_slots
    return {
        "dest_slot_limit": available_slots,
        "dest_slots_required": required_slots,
        "dest_status": "admitted" if admitted else "rejected",
        "dest_rejection_reason": (
            None
            if admitted
            else (
                f"matrix_block={matrix_block} requires {required_slots} DEST slots, "
                f"but the selected DEST configuration provides {available_slots}"
            )
        ),
    }


def _manifest_row(
    size: int,
    variant: str,
    dst_full_sync_en: bool,
    matrix_block: int,
) -> dict:
    row = {
        "row": _row_name(size, variant, dst_full_sync_en, matrix_block),
        "shape": f"newton_schulz_L{size}_b{HALF_SYNC_BATCH}",
        "size": size,
        "batch": HALF_SYNC_BATCH,
        "variant": variant,
        "dst_full_sync_en": dst_full_sync_en,
        "sync_mode": "full" if dst_full_sync_en else "half",
        "fp32_dest_acc_en": True,
        "matrix_block": matrix_block,
        **_dest_decision(matrix_block, dst_full_sync_en),
        "iterations": NEWTON_SCHULZ_ITERATIONS,
        "x0": "I/||R||inf",
        "math_fidelity": HALF_SYNC_MATH_FIDELITY,
        "fuse_s": True,
        "input_memory": "l1",
        "r_memory": "l1",
        "x0_memory": "l1",
        "output_memory": "dram",
        "double_buffer": True,
        "launches_requested": HALF_SYNC_LAUNCHES,
    }
    return row


def half_sync_row_manifest() -> list[dict]:
    """Return every requested row before L1 preflight or device execution."""
    return [
        _manifest_row(size, variant, dst_full_sync_en, matrix_block)
        for size in HALF_SYNC_SIZES
        for variant in HALF_SYNC_VARIANTS
        for dst_full_sync_en in HALF_SYNC_MODES
        for matrix_block in MATRIX_BLOCK_CHOICES
    ]


def _l1_preflight_bytes(
    ttnn,
    row: dict,
    *,
    core_count: int,
    profile: bool,
) -> int:
    physical_batch = _physical_tile_count(row["batch"], row["size"])
    padded_batch = _padded_tile_count(physical_batch, row["matrix_block"])
    return _validate_l1_preflight(
        ttnn,
        batch=padded_batch,
        core_count=core_count,
        state_dtype=_state_dtype(ttnn, row["variant"]),
        profile=profile,
        fuse_s=row["fuse_s"],
        output_memory=row["output_memory"],
        input_memory=row["input_memory"],
        r_memory=row["r_memory"],
        x0_memory=row["x0_memory"],
        matrix_block=row["matrix_block"],
        double_buffer=row["double_buffer"],
        variant=row["variant"],
        fp32_dest_acc_en=row["fp32_dest_acc_en"],
        dst_full_sync_en=row["dst_full_sync_en"],
    )


def _preflight_rejected(row: dict, reason: str) -> dict:
    row["preflight_status"] = "rejected"
    row["preflight_bytes"] = None
    row["preflight_reason"] = reason
    return row


def _preflight_admitted(row: dict, bytes_used: int) -> dict:
    row["preflight_status"] = "admitted"
    row["preflight_bytes"] = bytes_used
    row["preflight_reason"] = None
    return row


def _preflight_row(
    ttnn,
    manifest_row: dict,
    *,
    core_count: int,
    profile: bool,
) -> dict:
    row = dict(manifest_row)
    try:
        bytes_used = _l1_preflight_bytes(
            ttnn,
            row,
            core_count=core_count,
            profile=profile,
        )
    except ValueError as exc:
        return _preflight_rejected(row, str(exc))
    return _preflight_admitted(row, bytes_used)


def preflight_half_sync_rows(
    ttnn,
    *,
    core_count: int = HALF_SYNC_CORE_COUNT,
    profile: bool = False,
) -> list[dict]:
    """Attach an L1 preflight result to every manifest row without device work."""
    return [
        _preflight_row(
            ttnn,
            manifest_row,
            core_count=core_count,
            profile=profile,
        )
        for manifest_row in half_sync_row_manifest()
    ]
