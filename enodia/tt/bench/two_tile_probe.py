"""Board-free contracts for the minimal two-tile matmul probes.

The probe keeps the physical two-tile representation explicit: each complex
32x32 matrix is four real pages in column-major output-row order, while each
page remains row-major.  The accelerator probe consumes the same pages for
stages a, b, and c; this module is the independent NumPy oracle and metadata
surface used by the host tests and board record.
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
        "a1": [{"in0_offset": 0, "in1_offset": 0, "dest_real": 0, "dest_imag": 1}],
        "a2": [{"in0_offset": 2, "in1_offset": 1, "dest_real": 0, "dest_imag": 1}],
        "a3": [
            {"in0_offset": 0, "in1_offset": 0, "dest_real": 0, "dest_imag": 1},
            {"in0_offset": 2, "in1_offset": 1, "dest_real": 0, "dest_imag": 1},
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
