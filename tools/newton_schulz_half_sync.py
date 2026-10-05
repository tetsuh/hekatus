"""Device-side correctness helper for the Issue #96 half-sync sweep."""

from __future__ import annotations

import numpy as np

from enodia.tt.bench.newton_schulz_kernel import run_newton_schulz_kernel
from enodia.tt.bench.newton_schulz_reference import (
    bf16_round_complex,
    initial_value,
    newton_schulz_reference,
    random_hpd_batch,
)


def half_sync_correctness(
    ttnn,
    device,
    *,
    size: int,
    batch: int,
    variant: str,
    matrix_block: int,
    fp32_dest_acc_en: bool,
    dst_full_sync_en: bool,
    output_memory: str,
    math_fidelity: str = "HiFi3",
    fuse_s: bool = True,
    double_buffer: bool = True,
) -> dict:
    """Run one device case against the BF16-rounded-R NumPy reference."""
    matrices = random_hpd_batch(batch, size, seed=95 + size + batch)
    expected = newton_schulz_reference(
        bf16_round_complex(matrices), x0=initial_value(matrices)
    )
    actual = run_newton_schulz_kernel(
        ttnn,
        device,
        matrices,
        variant=variant,
        math_fidelity=math_fidelity,
        fuse_s=fuse_s,
        matrix_block=matrix_block,
        double_buffer=double_buffer,
        input_memory="l1",
        r_memory="l1",
        x0_memory="l1",
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
        output_memory=output_memory,
    )
    relative_error = float(
        np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    )
    return {
        "batch": batch,
        "reference": "newton_schulz_reference(bf16_round_complex(R))",
        "gate": 0.01,
        "relative_error": relative_error,
        "passed": relative_error <= 0.01,
    }
