"""Board-free contracts for staged Newton-Schulz iteration counts."""

from pathlib import Path

import numpy as np
import pytest

from enodia.tt.bench import newton_schulz_kernel
from enodia.tt.bench.newton_schulz_reference import newton_schulz_reference, random_hpd_batch
from tools import newton_schulz_runner

KERNEL_DIR = Path(__file__).parents[1] / "enodia/tt/bench/kernels"


def test_iterations_are_default_eight_with_explicit_one_step_opt_in():
    assert newton_schulz_kernel.NEWTON_SCHULZ_ITERATIONS == 8
    assert newton_schulz_kernel.NEWTON_SCHULZ_ITERATION_CHOICES == (1, 8)
    for iterations in (1, 8):
        newton_schulz_kernel._validate_iterations(iterations)
    for invalid in (0, 2, 7, True, "1"):
        with pytest.raises(ValueError, match=r"one of \(1, 8\)"):
            newton_schulz_kernel._validate_iterations(invalid)


def test_prepare_rejects_iteration_counts_other_than_one_or_eight_without_device_work():
    matrices = np.zeros((1, 16, 16), dtype=np.complex64)
    with pytest.raises(ValueError, match=r"one of \(1, 8\)"):
        newton_schulz_kernel.NewtonSchulzKernel.prepare(None, None, matrices, iterations=2)
    with pytest.raises(ValueError, match="default eight iterations"):
        newton_schulz_kernel.NewtonSchulzKernel.prepare(
            None,
            None,
            matrices,
            iterations=1,
            fidelity_split=(4, 4),
            math_fidelity="HiFi3",
        )


def test_reference_uses_the_same_requested_iteration_count():
    matrices = random_hpd_batch(2, 4, seed=63)
    one = newton_schulz_reference(matrices, iterations=1)
    eight = newton_schulz_reference(matrices, iterations=8)
    assert not np.array_equal(one, eight)


def test_compute_sources_accept_only_one_or_eight_iterations():
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    split = (KERNEL_DIR / "newton_schulz_fidelity_split_compute.cpp").read_text()
    assert "iterations == 1 || iterations == 8" in compute
    assert "iterations == 1 || iterations == 8" in split
    assert "iterations == 8, \"the throughput kernel has a fixed eight-iteration count\"" not in compute
    assert "iterations == 8, \"the throughput kernel has a fixed eight-iteration count\"" not in split


def test_stage_runner_defaults_and_explicit_one_step_metadata():
    parser = newton_schulz_runner._build_parser()
    assert parser.parse_args([]).iterations == 8
    assert parser.parse_args(["--iterations", "1", "--matrix-block", "4"]).iterations == 1
    assert parser.parse_args(["--iterations", "1", "--matrix-block", "4"]).matrix_block == 4


def test_stage_runner_validation_does_not_import_or_open_a_device():
    args = newton_schulz_runner._build_parser().parse_args(
        ["--batch", "4", "--iterations", "1", "--matrix-block", "2"]
    )
    newton_schulz_runner._validate(newton_schulz_runner._build_parser(), args)
    assert args.batch == 4
    assert args.iterations == 1


def test_production_contract_expands_logical_offsets_and_physical_identity():
    from enodia.tt.bench.two_tile_probe import two_tile_matrix_block_contract

    contract = two_tile_matrix_block_contract(4)
    assert contract["matmul_dimensions"] == {"rt": 2, "ct": 1, "kt": 1}
    for matrix, calls in enumerate(contract["r_times_x"]):
        assert [call["in0_offset"] for call in calls] == [6 * matrix, 6 * matrix + 2, 6 * matrix + 4]
        assert [call["in1_offset"] for call in calls] == [2 * matrix, 2 * matrix + 1, 2 * matrix + 2]
        assert [call["dst"] for call in calls] == [2 * matrix] * 3
        assert calls[2]["in1_cb"] == "CB_IDENTITY"
        assert calls[2]["in1_physical_offset"] == 0
    for matrix, calls in enumerate(contract["x_times_s"]):
        assert [call["in0_offset"] for call in calls] == [4 * matrix, 4 * matrix + 2]
        assert [call["in1_offset"] for call in calls] == [2 * matrix, 2 * matrix + 1]
        assert [call["dst"] for call in calls] == [2 * matrix] * 2
    assert contract["pack"] == [
        {"matrix": matrix, "dest_slots": (2 * matrix, 2 * matrix + 1), "output_page": matrix, "count": 4}
        for matrix in range(4)
    ]


# Keep this import-level smoke check intentionally tiny: a host test must not
# require ttnn merely to inspect the runner's public API.
def test_runner_uses_the_production_api_name():
    assert "run_newton_schulz_kernel" in Path(newton_schulz_runner.__file__).read_text()
