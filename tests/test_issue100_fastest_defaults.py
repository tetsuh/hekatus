"""Board-free regression coverage for Issue #100's fastest defaults."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from enodia.tt.bench import newton_schulz_kernel, run_matmul
from enodia.tt.bench.shapes import MatmulShape

TTNN = SimpleNamespace(bfloat16="bf16", float32="fp32", uint32="u32")


def test_public_kernel_defaults_select_the_issue100_configuration():
    prepare = inspect.signature(newton_schulz_kernel.NewtonSchulzKernel.prepare)
    run = inspect.signature(newton_schulz_kernel.run_newton_schulz_kernel)
    runner = inspect.signature(run_matmul.run_custom_newton_schulz)

    for signature in (prepare, run, runner):
        assert signature.parameters["variant"].default == "bf16"
        assert signature.parameters["matrix_block"].default == 8
        assert signature.parameters["double_buffer"].default is True
        assert signature.parameters["fp32_dest_acc_en"].default is True
        assert signature.parameters["dst_full_sync_en"].default is True


def test_bench_runner_defaults_and_explicit_legacy_overrides_are_visible():
    parser = run_matmul._build_parser()
    defaults = parser.parse_args([])
    assert defaults.custom_variant == "bf16"
    assert defaults.matrix_block == 8
    assert defaults.double_buffer is True
    assert defaults.fp32_dest_acc_en is True
    assert defaults.dst_full_sync_en is True

    legacy = parser.parse_args(
        [
            "--custom-variant",
            "bf16-fp32state",
            "--matrix-block",
            "4",
            "--no-double-buffer",
            "--no-fp32-dest-acc",
            "--no-dst-full-sync",
        ]
    )
    assert legacy.custom_variant == "bf16-fp32state"
    assert legacy.matrix_block == 4
    assert legacy.double_buffer is False
    assert legacy.fp32_dest_acc_en is False
    assert legacy.dst_full_sync_en is False


@pytest.mark.parametrize(
    ("size", "logical_batch"),
    [(16, 1), (16, 3), (16, 4), (16, 8192), (32, 4), (32, 5)],
)
def test_issue100_default_preflight_accepts_batch4_and_partial_tail_cases(size, logical_batch):
    physical_tiles = newton_schulz_kernel._physical_tile_count(logical_batch, size)
    tile_count = newton_schulz_kernel._padded_tile_count(physical_tiles, 8)

    total = newton_schulz_kernel._validate_l1_preflight(
        TTNN,
        batch=tile_count,
        core_count=110,
        state_dtype=TTNN.bfloat16,
        variant="bf16",
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
        fuse_s=False,
        input_memory="l1",
        output_memory="l1",
    )

    assert total <= newton_schulz_kernel._L1_TOTAL_BUDGET_BYTES


def test_existing_partial_device_cases_fit_host_preflight_with_explicit_legacy_placement():
    for logical_batch in range(1, 65):
        for matrix_block in (1, 2, 4, 8):
            for size in (16, 32):
                for fuse_s in (False, True):
                    physical_tiles = newton_schulz_kernel._physical_tile_count(
                        logical_batch, size
                    )
                    tile_count = newton_schulz_kernel._padded_tile_count(
                        physical_tiles, matrix_block
                    )
                    total = newton_schulz_kernel._validate_l1_preflight(
                        TTNN,
                        batch=tile_count,
                        core_count=110,
                        state_dtype="fp32",
                        variant="bf16-fp32state",
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,
                        fuse_s=fuse_s,
                        input_memory="l1",
                        r_memory="l1",
                        x0_memory="l1",
                        output_memory="dram",
                        matrix_block=matrix_block,
                        double_buffer=True,
                    )
                    assert total <= newton_schulz_kernel._L1_TOTAL_BUDGET_BYTES


def test_issue100_default_preflight_fails_fast_for_l32_batch8192_without_fallback():
    with pytest.raises(ValueError, match=r"matrix_block=8 L1 preflight failed") as excinfo:
        newton_schulz_kernel._validate_l1_preflight(
            TTNN,
            batch=8192,
            core_count=110,
            state_dtype=TTNN.bfloat16,
            variant="bf16",
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
            fuse_s=False,
            input_memory="l1",
            output_memory="l1",
        )

    message = str(excinfo.value)
    assert "total=1716992 bytes" in message
    assert "matrix_block=8" in message
    assert "fallback" not in message.lower()


def test_issue100_default_dest_limit_rejects_half_sync_without_fallback():
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_matrix_block(
            8,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
            variant="bf16",
        )


def test_runner_reports_default_l32_l1_failure_before_kernel_prepare():
    shape = MatmulShape(
        name="newton_schulz_L32_b8192",
        batch=8192,
        m=32,
        k=32,
        n=32,
        real_matmuls=4,
        family="newton_schulz",
        note="",
    )

    record = run_matmul.run_custom_newton_schulz(
        TTNN,
        device=object(),
        shape=shape,
        dtype_name="bfloat16",
        memory_name="l1",
        iters=1,
        repeats=1,
    )

    assert record["status"] == "failed"
    assert "matrix_block=8 L1 preflight failed" in record["error"]
    assert "total=1716992 bytes" in record["error"]
