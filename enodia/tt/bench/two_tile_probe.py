"""Board-free contracts for the minimal two-tile matmul probes.

The probe keeps the physical two-tile representation explicit: each complex
32x32 matrix is four real pages in column-major output-row order, while each
page remains row-major.  The accelerator probe consumes the same pages for
stages a, b, and c; this module is the independent NumPy oracle and metadata
surface used by the host tests and board record.

The DEST-slot probe is a separate one-call diagnostic.  Its four-page in0
block starts with the two pages consumed by ``rt=2, ct=1, kt=1`` — ``I`` and
``2I`` — followed by distinct sentinels.  The sentinels make a wrong page
stride observable without changing the pages used by the requested call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PROBE_STAGES = ("a", "b", "c")
DECOMPOSITION_STAGES = ("a1", "a2", "a3")
ALL_PROBE_STAGES = (*DECOMPOSITION_STAGES, *PROBE_STAGES)
TILE = 32


def _as_complex_batch(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"{name} must have shape (batch, size, size)")
    return values


def _bfloat16_round(values: np.ndarray) -> np.ndarray:
    """Round FP32 values to the bits represented by a BF16 input tile."""
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32) & np.uint32(0xFFFF0000)
    return words.view(np.float32)


def probe_input_pages(r: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """Return the exact CB page groups consumed by the three probe stages."""
    r = _as_complex_batch(r, name="r")
    x = _as_complex_batch(x, name="x")
    if r.shape != x.shape:
        raise ValueError(f"r and x must have identical shapes, got {r.shape} and {x.shape}")
    r_real = _bfloat16_round(r.real)
    r_imag = _bfloat16_round(r.imag)
    x_real = np.asarray(x.real, dtype=np.float32)
    x_imag = np.asarray(x.imag, dtype=np.float32)
    return {
        # [A00, A10, A01, A11] for A=[[-Rr, Ri], [-Ri, -Rr]].
        "in0_r": np.stack((-r_real, -r_imag, r_imag, -r_real), axis=1),
        # [B00, B10] for the first K=1 call of R*X.
        "in1_x_column": np.stack((x_real, x_imag), axis=1),
        # [A00, A10, A01, A11] for X=[[Xr, -Xi], [Xi, Xr]].
        "in0_x_block": np.stack((x_real, x_imag, -x_imag, x_real), axis=1),
        # [Sr, Si] is assembled in the compute kernel after stage a/b.
        "in1_s_column": np.empty((r.shape[0], 2, r.shape[1], r.shape[2]), dtype=np.float32),
        "identity": (2.0 * np.eye(r.shape[1], dtype=np.float32))[None, None, ...],
        "zero": np.zeros((1, 1, r.shape[1], r.shape[2]), dtype=np.float32),
    }


def expected_probe_partials(r: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the independent NumPy partial products for a1/a2/a3."""
    r = _as_complex_batch(r, name="r")
    x = _as_complex_batch(x, name="x")
    if r.shape != x.shape:
        raise ValueError(f"r and x must have identical shapes, got {r.shape} and {x.shape}")
    r_real = _bfloat16_round(r.real)
    r_imag = _bfloat16_round(r.imag)
    x_real = np.asarray(x.real, dtype=np.float32)
    x_imag = np.asarray(x.imag, dtype=np.float32)
    a1 = (-r_real @ x_real) + 1j * (-r_imag @ x_real)
    a2 = (r_imag @ x_imag) + 1j * (-r_real @ x_imag)
    return {"a1": a1, "a2": a2, "a3": a1 + a2}


def expected_probe_outputs(r: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """Compute independent expected outputs for stages a, b, and c."""
    r = _as_complex_batch(r, name="r")
    x = _as_complex_batch(x, name="x")
    if r.shape != x.shape:
        raise ValueError(f"r and x must have identical shapes, got {r.shape} and {x.shape}")
    partials = expected_probe_partials(r, x)
    rx = -partials["a3"]
    s = 2.0 * np.eye(r.shape[1], dtype=np.complex64)[None, ...] - rx
    return {"a": partials["a3"], "b": s, "c": x @ s, **partials}


def probe_stage_contract(stage: str) -> dict[str, Any]:
    """Describe one stage's fixed dimensions, queues, and register protocol."""
    if stage not in ALL_PROBE_STAGES:
        raise ValueError(f"stage must be one of {ALL_PROBE_STAGES}, got {stage!r}")
    uses_x_product = stage == "c"
    decomposition_calls = {
        # rt=2 writes DEST slots dst and dst + 1 in one call.
        "a1": [{"in0_offset": 0, "in1_offset": 0, "dst": 0}],
        "a2": [{"in0_offset": 2, "in1_offset": 1, "dst": 0}],
        "a3": [
            {"in0_offset": 0, "in1_offset": 0, "dst": 0},
            {"in0_offset": 2, "in1_offset": 1, "dst": 0},
        ],
    }
    if stage in DECOMPOSITION_STAGES:
        operation = {
            "a1": "[−Rr;−Ri]·Xr",
            "a2": "[Ri;−Rr]·Xi",
            "a3": "a1+a2 = −R·X",
        }[stage]
        matmul_calls = decomposition_calls[stage]
        dest_seed = "none"
    else:
        operation = {"a": "R·X", "b": "2I−R·X", "c": "X·(2I−R·X)"}[stage]
        matmul_calls = decomposition_calls["a3"] if stage in ("a", "b", "c") else []
        dest_seed = "none" if stage == "a" else "[2I, 0]"
    wait_count = 10 if uses_x_product else (5 if stage == "b" else 3)
    reserve_count = 6 if uses_x_product else 3
    push_count = 6 if uses_x_product else 3
    pop_count = 15 if uses_x_product else 9
    cb_counts = {
        "wait": wait_count,
        "reserve": reserve_count,
        "push": push_count,
        "pop": pop_count,
        "wait_front": wait_count,
        "reserve_back": reserve_count,
        "push_back": push_count,
        "pop_front": pop_count,
    }
    return {
        "stage": stage,
        "operation": operation,
        "matmul_dimensions": {"rt": 2, "ct": 1, "kt": 1},
        "matmul_call_count": len(matmul_calls),
        "matmul_calls": matmul_calls,
        "matmul_block_init_positions": ["before_tile_regs_acquire"],
        "in0_register": "SrcB",
        "in1_register": "SrcA",
        "tile_traversal": {
            "r_pages": ["A00", "A10", "A01", "A11"],
            "x_column_pages": ["B00", "B10"],
            "x_block_pages": ["A00", "A10", "A01", "A11"],
        },
        "dest_slots": [0, 1],
        "dest_seed": dest_seed,
        "pack_indices": [0, 1],
        "tile_regs_sequence": ["acquire", "commit", "wait", "release"],
        "cb_order": {
            "in0": "CB_TWO_TILE_R" if not uses_x_product else "CB_TWO_TILE_X",
            "in1": "CB_TWO_TILE_S",
            "outputs": ["CB_OUTPUT_REAL", "CB_OUTPUT_IMAG"],
        },
        "cb_counts": cb_counts,
        "batch": 4,
        "matrix_block": 1,
        "iterations": 1,
    }


DEST_PROBE_STAGE = "dest"
DEST_PROBE_BATCH = 4
DEST_PROBE_SENTINEL_SCALES = (7.0, 11.0)
DEST_PROBE_CLASSIFICATION_TOLERANCE = 1e-2
DEST_PROBE_TT_METAL_REVISION = "901dd9ce93816ffd1fd185b801fc727065e9ae07"


def known_dest_probe_complex_x() -> np.ndarray:
    """Return four distinct, asymmetric complex host-oracle tiles."""
    rows = np.arange(TILE, dtype=np.float32)[:, None]
    columns = np.arange(TILE, dtype=np.float32)[None, :]
    matrices = np.arange(DEST_PROBE_BATCH, dtype=np.float32)[:, None, None]
    real = 1.0 + 17.0 * rows + 3.0 * columns + 0.25 * matrices
    imag = -2.0 + 5.0 * rows - 11.0 * columns + 0.5 * matrices
    return (real + 1j * imag).astype(np.complex64)


def known_dest_probe_xr() -> np.ndarray:
    """Return the real Xr tile supplied to the one real matmul call."""
    return known_dest_probe_complex_x().real.astype(np.float32)


def _diagonal_tiles(scales: np.ndarray) -> np.ndarray:
    """Build one diagonal tile per scale without relying on a board layout."""
    scales = np.asarray(scales, dtype=np.float32)
    return scales[:, None, None] * np.eye(TILE, dtype=np.float32)[None, ...]


def _as_dest_probe_xr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    expected_shape = (DEST_PROBE_BATCH, TILE, TILE)
    if values.shape != expected_shape:
        raise ValueError(f"xr must have shape {expected_shape}, got {values.shape}")
    return values


def dest_probe_input_pages(xr: np.ndarray) -> dict[str, np.ndarray]:
    """Return the four in0 pages and one in1 page for each probe matrix.

    Pages 0 and 1 are the only pages consumed by the one ``rt=2`` call.  Pages
    2 and 3 are matrix-specific sentinels; a mistaken K/row stride can then be
    classified instead of silently producing another plausible product.
    """
    xr = _as_dest_probe_xr(xr)
    identity = _diagonal_tiles(np.ones(DEST_PROBE_BATCH, dtype=np.float32))
    twice_identity = _diagonal_tiles(
        np.full(DEST_PROBE_BATCH, 2.0, dtype=np.float32)
    )
    sentinel0 = _diagonal_tiles(
        np.asarray(DEST_PROBE_SENTINEL_SCALES[0])
        + np.arange(DEST_PROBE_BATCH, dtype=np.float32)
    )
    sentinel1 = _diagonal_tiles(
        np.asarray(DEST_PROBE_SENTINEL_SCALES[1])
        + np.arange(DEST_PROBE_BATCH, dtype=np.float32)
    )
    return {
        "in0": np.stack((identity, twice_identity, sentinel0, sentinel1), axis=1),
        "in1": xr[:, None, :, :].astype(np.float32),
        "identity": identity,
        "twice_identity": twice_identity,
        "sentinel0": sentinel0,
        "sentinel1": sentinel1,
    }


def expected_dest_probe_outputs(xr: np.ndarray) -> dict[str, np.ndarray]:
    """Return DEST slot 0/1 oracle tiles for the one known matmul."""
    xr = _as_dest_probe_xr(xr)
    return {"slot0": xr, "slot1": 2.0 * xr}


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), 1.0)
    return float(np.linalg.norm(actual - expected) / denominator)


def classify_dest_probe_outputs(
    actual_slot0: np.ndarray,
    actual_slot1: np.ndarray,
    xr: np.ndarray,
    *,
    tolerance: float = DEST_PROBE_CLASSIFICATION_TOLERANCE,
) -> list[dict[str, Any]]:
    """Classify every returned matrix/DEST tile against known candidates."""
    xr = _as_dest_probe_xr(xr)
    actual = (
        np.asarray(actual_slot0, dtype=np.float32),
        np.asarray(actual_slot1, dtype=np.float32),
    )
    expected_shape = (DEST_PROBE_BATCH, TILE, TILE)
    if any(values.shape != expected_shape for values in actual):
        raise ValueError(f"actual DEST tiles must have shape {expected_shape}")
    pages = dest_probe_input_pages(xr)
    candidates = {
        "Xr": xr,
        "2Xr": 2.0 * xr,
        "zero": np.zeros_like(xr),
        "other_tile_0": pages["sentinel0"] @ xr,
        "other_tile_1": pages["sentinel1"] @ xr,
    }
    rows: list[dict[str, Any]] = []
    for matrix in range(DEST_PROBE_BATCH):
        for dest_slot, values in enumerate(actual):
            errors = {
                label: _relative_error(values[matrix], target[matrix])
                for label, target in candidates.items()
            }
            best_label = min(errors, key=errors.get)
            classification = (
                "other_tile"
                if best_label.startswith("other_tile_") and errors[best_label] <= tolerance
                else best_label if errors[best_label] <= tolerance else "unclassified"
            )
            expected_label = "Xr" if dest_slot == 0 else "2Xr"
            rows.append(
                {
                    "matrix": matrix,
                    "dest_slot": dest_slot,
                    "output_tile": dest_slot,
                    "expected_label": expected_label,
                    "classification": classification,
                    "pass": classification == expected_label,
                    "relative_error": errors[expected_label],
                    "candidate_relative_errors": errors,
                    "tolerance": tolerance,
                    "finite": bool(np.isfinite(values[matrix]).all()),
                }
            )
    return rows


def dest_probe_source_audit() -> dict[str, Any]:
    """Return the pinned LLK/API evidence applied to the one-call probe."""
    return {
        "revision": DEST_PROBE_TT_METAL_REVISION,
        "question": (
            "Whether rt=2, ct=1, kt=1 consumes consecutive in0 rows and writes "
            "DEST slots 0/1 without interpreting SrcA/SrcB as ct/rt."
        ),
        "sources": [
            {
                "path": "tt_metal/hw/inc/api/compute/matmul.h",
                "lines": "170-182, 221-241",
                "finding": (
                    "matmul_block_init and matmul_block pass ct_dim, rt_dim, and "
                    "kt_dim to unpack/math; in0 is SrcB and in1 is SrcA."
                ),
            },
            {
                "path": "tt_metal/hw/ckernels/blackhole/metal/llk_api/"
                "llk_unpack_AB_matmul_api.h",
                "lines": "70-127",
                "finding": (
                    "The two CB indices and dimensions are forwarded unchanged to "
                    "the Blackhole unpack LLK."
                ),
            },
            {
                "path": "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h",
                "lines": "355-366, 379-425",
                "finding": (
                    "For ct=1, rt=2, the two in0/SrcB rows are consecutive and the "
                    "in1/SrcA tile is reused; kt_dim=1 leaves a one-page K stride."
                ),
            },
            {
                "path": "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h",
                "lines": "719-760",
                "finding": (
                    "Math visits output rows in order and accumulates into dst and "
                    "dst+1; ct does not swap with rt."
                ),
            },
            {
                "path": "tt_metal/hw/ckernels/blackhole/metal/llk_api/"
                "llk_unpack_common_api.h",
                "lines": "37-65",
                "finding": (
                    "Source-register strides use the CB page sizes; this probe uses "
                    "BF16 in0 and FP32 in1 consistently."
                ),
            },
        ],
        "application": {
            "in0_pages_consumed": [0, 1],
            "in0_pages_sentinels": [2, 3],
            "in0_row_stride_pages": 1,
            "in0_k_stride_pages": 1,
            "in1_page_offset": 0,
            "dest_slots": [0, 1],
            "expected": ["Xr", "2Xr"],
        },
    }


def dest_probe_contract() -> dict[str, Any]:
    """Describe the dedicated one-call DEST-slot probe."""
    return {
        "stage": DEST_PROBE_STAGE,
        "operation": "[I; 2I] · Xr",
        "matmul_dimensions": {"rt": 2, "ct": 1, "kt": 1},
        "matmul_block_init": {
            "count": 1,
            "arguments": ["in0", "in1", False, 1, 2, 1],
            "position": "before_tile_regs_acquire",
        },
        "matmul_calls": [
            {
                "count": 1,
                "in0_offset": 0,
                "in1_offset": 0,
                "dest_slot_start": 0,
                "arguments": ["in0", "in1", 0, 0, 0, False, 1, 2, 1],
            }
        ],
        "in0_register": "SrcB",
        "in1_register": "SrcA",
        "in0_pages": ["I", "2I", "sentinel0", "sentinel1"],
        "in0_pages_consumed": [0, 1],
        "in0_pages_sentinels": [2, 3],
        "in1_pages": ["Xr"],
        "cb_page_offsets": {"in0": [0, 1, 2, 3], "in1": [0]},
        "dest_slots": [0, 1],
        "expected_dest_tiles": {"0": "Xr", "1": "2Xr"},
        "pack_operations": [
            {"dst_index": 0, "output_cb": 15, "pack_api": "pack_tile", "count": 1},
            {"dst_index": 1, "output_cb": 16, "pack_api": "pack_tile", "count": 1},
        ],
        "tile_regs_sequence": [
            "acquire",
            "matmul",
            "commit",
            "wait",
            "pack",
            "release",
        ],
        "cb_lifecycle_per_matrix": {
            "wait_front": 2,
            "reserve_back": 4,
            "push_back": 4,
            "pop_front": 2,
        },
        "batch": DEST_PROBE_BATCH,
        "matrix_block": 1,
        "iterations": 1,
        "source_audit": dest_probe_source_audit(),
    }
