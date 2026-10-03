"""Board-free contracts for the minimal two-tile matmul probes.

The probe keeps the physical two-tile representation explicit: each complex
32x32 matrix is a real page block in column-major output-row order, while
each page remains row-major.  The accelerator probe consumes the same pages
for stages a, b_prime, and c; this module is the independent NumPy oracle and
metadata surface used by the host tests and board record.

The DEST-slot probe is a separate one-call diagnostic.  Its four-page in0
block starts with the two pages consumed by ``rt=2, ct=1, kt=1`` — ``I`` and
``2I`` — followed by distinct sentinels.  The sentinels make a wrong page
stride observable without changing the pages used by the requested call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PROBE_STAGES = ("a", "b_prime", "c")
PROBE_STAGE_ALIASES = {"b": "b_prime", "b-prime": "b_prime"}
PROBE_STAGE_CHOICES = (*PROBE_STAGES, *PROBE_STAGE_ALIASES)
DECOMPOSITION_STAGES = ("a1", "a2", "a3")
ALL_PROBE_STAGES = (*DECOMPOSITION_STAGES, *PROBE_STAGES)
ALL_PROBE_STAGE_CHOICES = (*ALL_PROBE_STAGES, *PROBE_STAGE_ALIASES)
TILE = 32


def normalize_probe_stage(stage: str) -> str:
    """Return the canonical diagnostic stage name.

    ``b`` remains the command-line spelling for compatibility with the prior
    probe; records use ``b_prime`` so the seed-free boundary is unambiguous.
    """
    canonical = PROBE_STAGE_ALIASES.get(stage, stage)
    if canonical not in ALL_PROBE_STAGES:
        raise ValueError(
            f"stage must be one of {ALL_PROBE_STAGE_CHOICES}, got {stage!r}"
        )
    return canonical


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
    identity = np.eye(r.shape[1], dtype=np.float32)
    twice_identity = 2.0 * identity
    zero = np.zeros_like(identity)
    return {
        # [A00, A10, A01, A11, A02, A12] for
        # A=[[-Rr, Ri, 2I], [-Ri, -Rr, 0]].
        "in0_r": np.stack(
            (
                -r_real,
                -r_imag,
                r_imag,
                -r_real,
                np.broadcast_to(twice_identity, r_real.shape),
                np.broadcast_to(zero, r_real.shape),
            ),
            axis=1,
        ),
        # [B00, B10, B20] for the seed-free K=3 R*X calls.  The I page is
        # resident BF16 input; it is repeated here only for the NumPy contract.
        "in1_x_column": np.stack(
            (
                x_real,
                x_imag,
                np.broadcast_to(identity, x_real.shape),
            ),
            axis=1,
        ),
        # [A00, A10, A01, A11] for X=[[Xr, -Xi], [Xi, Xr]].
        "in0_x_block": np.stack((x_real, x_imag, -x_imag, x_real), axis=1),
        # [Sr, Si] is assembled in the compute kernel after stage a/b.
        "in1_s_column": np.empty((r.shape[0], 2, r.shape[1], r.shape[2]), dtype=np.float32),
        "identity": (2.0 * np.eye(r.shape[1], dtype=np.float32))[None, None, ...],
        "zero": zero[None, None, ...],
        "one_identity": identity[None, None, ...],
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
    return {
        "a": partials["a3"],
        "b": s,
        "b_prime": s,
        "c": x @ s,
        **partials,
    }


def two_tile_matrix_block_contract(block_count: int) -> dict[str, Any]:
    """Expand the production two-tile offsets for one matrix block.

    Each logical matrix owns two consecutive DEST rows.  Keeping this small
    host contract next to the diagnostic contract lets source tests verify the
    production loop without requiring a board or a compiler.
    """
    if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count < 1:
        raise ValueError(f"block_count must be a positive integer, got {block_count!r}")
    return {
        "matmul_dimensions": {"rt": 2, "ct": 1, "kt": 1},
        "r_times_x": [
            [
                {"in0_offset": 6 * matrix, "in1_offset": 2 * matrix, "dst": 2 * matrix},
                {"in0_offset": 6 * matrix + 2, "in1_offset": 2 * matrix + 1, "dst": 2 * matrix},
                {
                    "in0_offset": 6 * matrix + 4,
                    "in1_offset": 2 * matrix + 2,
                    "in1_physical_offset": 0,
                    "in1_cb": "CB_IDENTITY",
                    "dst": 2 * matrix,
                },
            ]
            for matrix in range(block_count)
        ],
        "x_times_s": [
            [
                {"in0_offset": 4 * matrix, "in1_offset": 2 * matrix, "dst": 2 * matrix},
                {"in0_offset": 4 * matrix + 2, "in1_offset": 2 * matrix + 1, "dst": 2 * matrix},
            ]
            for matrix in range(block_count)
        ],
        "pack": [
            {
                "matrix": matrix,
                "dest_slots": (2 * matrix, 2 * matrix + 1),
                "output_page": matrix,
                "count": block_count,
            }
            for matrix in range(block_count)
        ],
    }


def two_tile_fidelity_audit() -> dict[str, Any]:
    """Return the board-free probe/production mapping audit.

    The table is intentionally literal rather than inferred from source text:
    source tests below assert the names and offsets against this contract.  It
    covers the batch-4, 32x32, ``bf16-fp32state`` staged case and records the
    matrix-block capacity rule used by the production descriptors.
    """
    page_sizes = {"BF16": 2 * TILE * TILE, "FP32": 4 * TILE * TILE}
    probe_cb = {
        "CB_TWO_TILE_R": {
            "id": 20,
            "format": "BF16",
            "page_size_bytes": page_sizes["BF16"],
            "pages_per_matrix": 6,
        },
        "CB_TWO_TILE_X": {
            "id": 21,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": 4,
        },
        "CB_TWO_TILE_S": {
            "id": 22,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": 2,
        },
        "CB_IDENTITY": {
            "id": 5,
            "format": "BF16",
            "page_size_bytes": page_sizes["BF16"],
            "pages_per_matrix": 1,
            "resident": True,
        },
        "CB_ZERO": {
            "id": 6,
            "format": "BF16",
            "page_size_bytes": page_sizes["BF16"],
            "pages_per_matrix": 1,
            "resident": True,
        },
        "CB_OUTPUT_REAL": {
            "id": 15,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": 1,
        },
        "CB_OUTPUT_IMAG": {
            "id": 16,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": 1,
        },
    }
    production_cb = {
        **probe_cb,
        "CB_X0_REAL": {
            "id": 3,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": "matrix_block",
        },
        "CB_X0_IMAG": {
            "id": 4,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": "matrix_block",
        },
        "CB_STATE_REAL": {
            "id": 7,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": "matrix_block",
        },
        "CB_STATE_IMAG": {
            "id": 8,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": "matrix_block",
        },
        "CB_NEG_X_IMAG": {
            "id": 13,
            "format": "FP32(state)",
            "page_size_bytes": page_sizes["FP32"],
            "pages_per_matrix": "matrix_block",
            "producer": "negate_two_tile_state_imag_block",
        },
    }
    production_capacity = {}
    for matrix_block in (1, 2, 4, 8):
        production_capacity[str(matrix_block)] = {
            name: (
                matrix_block
                if name
                in {
                    "CB_X0_REAL",
                    "CB_X0_IMAG",
                    "CB_STATE_REAL",
                    "CB_STATE_IMAG",
                    "CB_NEG_X_IMAG",
                    "CB_OUTPUT_REAL",
                    "CB_OUTPUT_IMAG",
                }
                else value["pages_per_matrix"]
            )
            for name, value in production_cb.items()
            if name not in {"CB_IDENTITY", "CB_ZERO"}
        }
        production_capacity[str(matrix_block)].update(
            {
                "CB_TWO_TILE_R": 6 * matrix_block,
                "CB_TWO_TILE_X": 4 * matrix_block,
                "CB_TWO_TILE_S": 2 * matrix_block,
                "CB_IDENTITY": 1,
                "CB_ZERO": 1,
            }
        )

    return {
        "contract": "issue-63-two-tile-fidelity-split-audit",
        "version": 1,
        "scope": {
            "batch": 4,
            "size": TILE,
            "iterations": 1,
            "variant": "bf16-fp32state",
            "input_memory": "dram",
            "output_memory": "dram",
            "board_execution": "not run",
        },
        "source_files": {
            "probe": [
                "tools/newton_schulz_two_tile_probe.py",
                "enodia/tt/bench/kernels/two_tile_probe_reader.cpp",
                "enodia/tt/bench/kernels/two_tile_probe_compute.cpp",
                "enodia/tt/bench/kernels/two_tile_probe_writer.cpp",
                "enodia/tt/bench/two_tile_probe.py",
            ],
            "production": [
                "tools/newton_schulz_runner.py",
                "enodia/tt/bench/newton_schulz_kernel.py",
                "enodia/tt/bench/kernels/newton_schulz_reader_two_tile.cpp",
                "enodia/tt/bench/kernels/newton_schulz_compute.cpp",
                "enodia/tt/bench/kernels/newton_schulz_writer.cpp",
            ],
        },
        "input_tensors": {
            "probe": {
                "runtime_tensor_order": ["R", "Xr", "Xi", "-Xi", "I", "0"],
                "tensors": {
                    "R": {
                        "dtype": "BF16",
                        "shape": ["batch", 6, TILE, TILE],
                        "page_order": ["-Rr", "-Ri", "+Ri", "-Rr", "+2I", "0"],
                        "layout": "row-major values within each tile; column-major pages by output row",
                    },
                    "Xr": {"dtype": "FP32(state)", "shape": ["batch", 1, TILE, TILE]},
                    "Xi": {"dtype": "FP32(state)", "shape": ["batch", 1, TILE, TILE]},
                    "-Xi": {"dtype": "FP32(state)", "shape": ["batch", 1, TILE, TILE]},
                    "I": {"dtype": "BF16", "shape": [1, 1, TILE, TILE], "resident": True},
                    "0": {"dtype": "BF16", "shape": [1, 1, TILE, TILE], "resident": True},
                },
            },
            "production": {
                "runtime_tensor_order": ["R", "X0r", "X0i", "I", "0"],
                "tensors": {
                    "R": {
                        "dtype": "BF16",
                        "shape": ["tile_count", 6, TILE, TILE],
                        "page_order": ["-Rr", "-Ri", "+Ri", "-Rr", "+2I", "0"],
                        "layout": "row-major values within each tile; column-major pages by output row",
                    },
                    "X0r": {"dtype": "FP32(state)", "shape": ["tile_count", 1, TILE, TILE]},
                    "X0i": {"dtype": "FP32(state)", "shape": ["tile_count", 1, TILE, TILE]},
                    "I": {"dtype": "BF16", "shape": [1, 1, TILE, TILE], "resident": True},
                    "0": {"dtype": "BF16", "shape": [1, 1, TILE, TILE], "resident": True},
                    "-Xi": {
                        "dtype": "FP32(state)",
                        "shape": ["tile_count", 1, TILE, TILE],
                        "producer": "negate_two_tile_state_imag_block",
                    },
                },
            },
            "comparison": {
                "R": "match after BF16 conversion; same six-page order",
                "X": "match logically; production computes -Xi in CB13 instead of a host tensor",
                "constants": "match; I and zero are resident BF16 singleton pages",
            },
        },
        "cb_definitions": {
            "probe": probe_cb,
            "production": production_cb,
            "production_capacity_pages": production_capacity,
            "page_size_rule": {"BF16": page_sizes["BF16"], "FP32(state)": page_sizes["FP32"]},
        },
        "reader": {
            "probe": {
                "compile_time_accessor_start": 1,
                "runtime_args": ["R", "Xr", "Xi", "-Xi", "I", "0", "start_tile", "tile_count"],
                "accessors": [
                    {"tensor": "R", "cb": "CB_TWO_TILE_R", "page": "tile * 6 + face"},
                    {"tensor": "Xr", "cb": "CB_X_REAL", "page": "tile"},
                    {"tensor": "Xi", "cb": "CB_X_IMAG", "page": "tile"},
                    {"tensor": "-Xi", "cb": "CB_NEGATIVE_X_IMAG", "page": "tile"},
                    {"tensor": "I", "cb": "CB_IDENTITY", "page": 0},
                    {"tensor": "0", "cb": "CB_ZERO", "page": 0},
                ],
            },
            "production": {
                "compile_time_accessor_start": 3,
                "compile_time_prefix": ["iterations", "batch_reads", "matrix_block"],
                "runtime_args": ["R", "X0r", "X0i", "I", "0", "start_tile", "tile_count"],
                "accessors": [
                    {"tensor": "R", "cb": "CB_TWO_TILE_R", "page": "tile * 6 + face"},
                    {"tensor": "X0r", "cb": "CB_X0_REAL", "page": "tile"},
                    {"tensor": "X0i", "cb": "CB_X0_IMAG", "page": "tile"},
                    {"tensor": "I", "cb": "CB_IDENTITY", "page": 0},
                    {"tensor": "0", "cb": "CB_ZERO", "page": 0},
                ],
                "block_destination_stride": {
                    "R": "index * 6 * get_tile_size(CB_TWO_TILE_R)",
                    "X0r": "index * get_tile_size(CB_X0_REAL)",
                    "X0i": "index * get_tile_size(CB_X0_IMAG)",
                },
            },
            "mapping": "production runtime/accessor order matches its five-input host order; probe adds host -Xi",
        },
        "compute": {
            "matmul_dimensions": {"ct": 1, "rt": 2, "kt": 1},
            "probe": {
                "R_times_X": [
                    {"in0": "CB_TWO_TILE_R", "in0_offset": 0, "in1": "CB_TWO_TILE_S", "in1_offset": 0, "dst": 0},
                    {"in0": "CB_TWO_TILE_R", "in0_offset": 2, "in1": "CB_TWO_TILE_S", "in1_offset": 1, "dst": 0},
                    {"in0": "CB_TWO_TILE_R", "in0_offset": 4, "in1": "CB_IDENTITY", "in1_offset": 0, "dst": 0},
                ],
                "X_times_S": [
                    {"in0": "CB_TWO_TILE_X", "in0_offset": 0, "in1": "CB_TWO_TILE_S", "in1_offset": 0, "dst": 0},
                    {"in0": "CB_TWO_TILE_X", "in0_offset": 2, "in1": "CB_TWO_TILE_S", "in1_offset": 1, "dst": 0},
                ],
            },
            "production": {
                "R_times_X": "same offsets per matrix; in0=CB_TWO_TILE_R, in1=CB_TWO_TILE_S, dst=2*index",
                "X_times_S": "same offsets per matrix; in0=CB_TWO_TILE_X, in1=CB_TWO_TILE_S, dst=2*index",
                "destination_rows": ["2*index -> real", "2*index+1 -> imaginary"],
                "pack": {
                    "S": "pack_tile<true>(2*index, CB_S_REAL, index) and pack_tile<true>(2*index+1, CB_S_IMAG, index)",
                    "output": "pack_tile<true>(2*index, output_real, index) and pack_tile<true>(2*index+1, output_imag, index)",
                    "out_of_order_index": "index is within the block-reserved output CB; it is not a DEST index",
                },
            },
            "format_transitions": [
                {
                    "boundary": "R*X K0/K1",
                    "SrcA": "CB_TWO_TILE_S FP32(state)",
                    "SrcB": "CB_TWO_TILE_R BF16",
                    "operation_init": "matmul_block_init(CB_TWO_TILE_R, CB_TWO_TILE_S, ct=1, rt=2, kt=1)",
                },
                {
                    "boundary": "R*X K2",
                    "SrcA": "CB_IDENTITY BF16",
                    "SrcB": "CB_TWO_TILE_R BF16",
                    "operation_init": "same short matmul init; independent srca/srcb new-only transitions",
                },
                {
                    "boundary": "X*S",
                    "SrcA": "CB_TWO_TILE_S FP32(state)",
                    "SrcB": "CB_TWO_TILE_X FP32(state)",
                    "operation_init": "matmul_block_init(CB_TWO_TILE_X, CB_TWO_TILE_S, ct=1, rt=2, kt=1)",
                },
                {
                    "boundary": "state -Xi construction",
                    "SrcA": "CB_ZERO BF16",
                    "SrcB": "X_i FP32(state)",
                    "operation_init": "sub_tiles_init; no init_common",
                },
            ],
            "initialization": {"uses_init_common": False, "uses_short_matmul_init": True},
            "state_output_alias": False,
        },
        "writer": {
            "probe": {
                "source": "enodia/tt/bench/kernels/two_tile_probe_writer.cpp",
                "input_cbs": ["CB_OUTPUT_REAL", "CB_OUTPUT_IMAG"],
                "output_tensors": ["real", "imag"],
                "dtype": "FP32(state)",
                "page": "tile",
            },
            "production": {
                "source": "enodia/tt/bench/kernels/newton_schulz_writer.cpp",
                "input_cbs": ["CB_OUTPUT_REAL", "CB_OUTPUT_IMAG"],
                "output_tensors": ["real", "imag"],
                "dtype": "FP32(state)",
                "page": "tile (matrix_block=1) or start_tile+offset+index (blocked)",
            },
            "mapping": "one FP32 page per real/imag output tensor; state CBs 7/8 are not writer inputs",
        },
        "host_readback": {
            "probe": {
                "shape": ["batch", 1, TILE, TILE],
                "download": "_download_float32(outputs[0/1])[:, 0]",
                "layout": "one row-major tile per output tensor page",
            },
            "production": {
                "shape": ["tile_count", 1, TILE, TILE],
                "download": "_download_float32(outputs[0/1])",
                "interpretation": "_unpack_matrices(values, batch, size, packed); for L=32 this selects values[:batch, 0]",
                "layout": "same row-major real/imag pages; complex result is real + 1j*imag",
            },
            "comparison": "identical for the staged L=32 case; no packed L=16 interpretation is involved",
        },
        "findings": [
            {
                "id": "stale-two-tile-negation-format-reference",
                "status": "fixed",
                "file": "enodia/tt/bench/kernels/newton_schulz_compute.cpp",
                "finding": "production passed CB_ZERO/X_IMAG as old operands after startup or X*S, although those boundaries leave different SrcA/SrcB CBs active; a format reconfiguration could therefore be skipped",
                "fix": "two-tile -Xi construction now uses independent new-only SrcA=CB_ZERO and SrcB=X_IMAG transitions; X-block copy names CB_ZERO as its actual old SrcA",
                "probe_difference": "probe supplies -Xi as a host input and does not exercise this production-only boundary",
            },
            {
                "id": "two-dest-row-output-alias",
                "status": "no-mismatch",
                "finding": "production maps each matrix to DEST rows 2*index and 2*index+1, packs each row to separate real/imag CB pages, and writes only final CB_OUTPUT_REAL/IMAG tensors",
            },
        ],
    }


def probe_stage_contract(stage: str) -> dict[str, Any]:
    """Describe one stage's fixed dimensions, queues, and register protocol."""
    stage = normalize_probe_stage(stage)
    uses_x_product = stage == "c"
    r_calls = [
        {"in0_offset": 0, "in1_offset": 0, "dst": 0},
        {"in0_offset": 2, "in1_offset": 1, "dst": 0},
        {"in0_offset": 4, "in1_offset": 2, "dst": 0},
    ]
    x_calls = [
        {"in0_offset": 0, "in1_offset": 0, "dst": 0},
        {"in0_offset": 2, "in1_offset": 1, "dst": 0},
    ]
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
        r_product_calls = matmul_calls
        x_product_calls = []
    else:
        operation = {"a": "−R·X", "b_prime": "2I−R·X", "c": "X·(2I−R·X)"}[stage]
        r_product_calls = r_calls if stage in ("b_prime", "c") else r_calls[:2]
        x_product_calls = x_calls if stage == "c" else []
        matmul_calls = [*r_product_calls, *x_product_calls]
    dest_seed = "none"
    wait_count = 10 if uses_x_product else (7 if stage == "b_prime" else 3)
    reserve_count = 6 if uses_x_product else 3
    push_count = 6 if uses_x_product else 3
    pop_count = 15 if uses_x_product else (10 if stage == "b_prime" else 9)
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
        "r_times_x_calls": r_product_calls,
        "x_times_s_calls": x_product_calls,
        "r_times_x_physical_inputs": [
            {"cb": "CB_TWO_TILE_S", "offset": 0},
            {"cb": "CB_TWO_TILE_S", "offset": 1},
            {"cb": "CB_IDENTITY", "offset": 0},
        ],
        "matmul_block_init_positions": ["before_tile_regs_acquire"],
        "in0_register": "SrcB",
        "in1_register": "SrcA",
        "tile_traversal": {
            "r_pages": ["A00", "A10", "A01", "A11", "A02", "A12"],
            "x_column_pages": ["B00", "B10", "B20"],
            "x_block_pages": ["A00", "A10", "A01", "A11"],
        },
        "dest_slots": [0, 1],
        "dest_seed": dest_seed,
        "pack_indices": [0, 1],
        "tile_regs_sequence": ["acquire", "commit", "wait", "release"],
        "cb_order": {
            "in0": ["CB_TWO_TILE_R", "CB_TWO_TILE_X"] if uses_x_product else "CB_TWO_TILE_R",
            "in1": ["CB_TWO_TILE_S", "CB_IDENTITY"],
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
