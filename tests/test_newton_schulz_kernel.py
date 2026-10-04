"""L0 equivalence and accounting for the fixed-count Newton-Schulz kernel."""

from __future__ import annotations

import ast
import importlib.util
import os
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from enodia.tt.bench import newton_schulz_kernel
from enodia.tt.bench.newton_schulz_kernel import run_newton_schulz_kernel
from enodia.tt.bench.newton_schulz_reference import (
    COMPLEX_MATMULS_PER_INVERSE,
    NEWTON_SCHULZ_ITERATIONS,
    bf16_round_complex,
    initial_value,
    inverse_flops,
    newton_schulz_reference,
    random_hpd_batch,
)
from enodia.tt.bench.shapes import newton_schulz_shapes, total_flops

DEVICE_TEST = (
    os.environ.get("HEKATUS_TT_DEVICE_TEST") == "1"
    and os.environ.get("HEKATUS_TT_PINNED_CONTAINER") == "1"
)
HAS_TTNN = importlib.util.find_spec("ttnn") is not None
PARTIAL_BLOCK_CASES = tuple(
    (batch, matrix_block, size, fuse_s)
    for batch in range(1, 65)
    for matrix_block in (1, 2, 4, 8)
    for size in (16, 32)
    for fuse_s in (False, True)
)
PARTIAL_BLOCK_CASE_IDS = [
    f"batch{batch}-block{matrix_block}-L{size}-fuse_s_{str(fuse_s).lower()}"
    for batch, matrix_block, size, fuse_s in PARTIAL_BLOCK_CASES
]


class ReferenceTests(unittest.TestCase):
    def test_accelerator_initial_value_matches_the_independent_oracle(self):
        matrices = random_hpd_batch(3, 16, seed=19)

        actual = newton_schulz_kernel._initial_value(matrices)
        expected = initial_value(matrices)

        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

    def test_initial_value_uses_identity_scaled_by_infinity_norm(self):
        matrices = random_hpd_batch(3, 16, seed=19)
        norm_inf = np.linalg.norm(matrices, ord=np.inf, axis=(-2, -1))
        expected = np.eye(16, dtype=np.complex64)[None, :, :] / norm_inf[:, None, None]

        np.testing.assert_allclose(newton_schulz_kernel._initial_value(matrices), expected)
        np.testing.assert_allclose(initial_value(matrices), expected)

    def test_random_input_is_hermitian_positive_definite_at_the_requested_condition(self):
        matrices = random_hpd_batch(3, 16, condition_number=100.0, seed=7)

        np.testing.assert_allclose(matrices, matrices.conj().swapaxes(-1, -2), atol=2e-6)
        for matrix in matrices:
            eigenvalues = np.linalg.eigvalsh(matrix)
            self.assertGreater(eigenvalues[0], 0.0)
            self.assertAlmostEqual(float(eigenvalues[-1] / eigenvalues[0]), 100.0, delta=0.02)

    def test_reference_uses_the_fixed_initial_value_and_iteration_count(self):
        matrices = random_hpd_batch(2, 4, seed=3)
        expected = initial_value(matrices)
        identity = np.eye(4, dtype=np.complex64)
        for _ in range(NEWTON_SCHULZ_ITERATIONS):
            expected = expected @ (2.0 * identity - matrices @ expected)

        np.testing.assert_allclose(newton_schulz_reference(matrices), expected, rtol=2e-6)

    def test_inverse_accounting_is_the_catalogue_count_times_two_matmuls_per_iteration(self):
        shape = newton_schulz_shapes(batches=(8192,))[0]

        self.assertEqual(
            inverse_flops(shape),
            COMPLEX_MATMULS_PER_INVERSE * total_flops(shape),
        )
        self.assertEqual(
            newton_schulz_kernel.COMPLEX_MATMULS_PER_INVERSE,
            COMPLEX_MATMULS_PER_INVERSE,
        )

    def test_kernel_variants_only_claim_the_two_approved_state_behaviors(self):
        self.assertEqual(set(newton_schulz_kernel._VARIANTS), {"bf16", "bf16-fp32state"})
        self.assertFalse(newton_schulz_kernel._VARIANTS["bf16"])
        self.assertTrue(newton_schulz_kernel._VARIANTS["bf16-fp32state"])

    def test_math_fidelity_names_map_to_pinned_enum_and_default_is_hifi4(self):
        ttnn = SimpleNamespace(
            MathFidelity=SimpleNamespace(
                LoFi="lofi",
                HiFi2="hifi2",
                HiFi3="hifi3",
                HiFi4="hifi4",
            )
        )
        for name in ("LoFi", "HiFi2", "HiFi3", "HiFi4"):
            self.assertEqual(
                newton_schulz_kernel._math_fidelity_value(ttnn, name),
                getattr(ttnn.MathFidelity, name),
            )
        with self.assertRaisesRegex(ValueError, "choose from"):
            newton_schulz_kernel._math_fidelity_value(ttnn, "invalid")

        source = Path(
            Path(__file__).parents[1]
            / "enodia"
            / "tt"
            / "bench"
            / "newton_schulz_kernel.py"
        ).read_text()
        self.assertIn('math_fidelity: str = "HiFi4"', source)
        self.assertIn("math_fidelity=math_fidelity_value", source)

    def test_fp32_state_selects_fp32_state_and_output_descriptors(self):
        ttnn = SimpleNamespace(bfloat16="bf16", float32="fp32")
        bf16_defs = newton_schulz_kernel._cb_definitions(ttnn, ttnn.bfloat16)
        fp32_defs = newton_schulz_kernel._cb_definitions(ttnn, ttnn.float32)

        self.assertEqual(newton_schulz_kernel._state_dtype(ttnn, "bf16"), "bf16")
        self.assertEqual(newton_schulz_kernel._state_dtype(ttnn, "bf16-fp32state"), "fp32")
        self.assertEqual(newton_schulz_kernel._output_memory_name("bf16"), "l1")
        self.assertEqual(newton_schulz_kernel._output_memory_name("bf16-fp32state"), "dram")
        self.assertEqual(bf16_defs[newton_schulz_kernel.CB_OUTPUT_REAL], ("bf16", 2))
        self.assertEqual(bf16_defs[newton_schulz_kernel.CB_OUTPUT_IMAG], ("bf16", 2))
        self.assertEqual(fp32_defs[newton_schulz_kernel.CB_OUTPUT_REAL], ("fp32", 2))
        self.assertEqual(fp32_defs[newton_schulz_kernel.CB_OUTPUT_IMAG], ("fp32", 2))
        for index in (
            newton_schulz_kernel.CB_X0_REAL,
            newton_schulz_kernel.CB_STATE_REAL,
            newton_schulz_kernel.CB_S_REAL,
            newton_schulz_kernel.CB_NEG_X_IMAG,
        ):
            self.assertEqual(fp32_defs[index][0], "fp32")

    def test_balanced_ranges_cover_batch_without_padding_or_gaps(self):
        ranges = newton_schulz_kernel._balanced_ranges(8192, 120)

        self.assertEqual(len(ranges), 120)
        self.assertEqual(ranges[0], (0, 69))
        self.assertEqual(ranges[-1], (8124, 68))
        self.assertEqual(sum(count for _, count in ranges), 8192)
        self.assertEqual({count for _, count in ranges}, {68, 69})
        for previous, current in pairwise(ranges):
            self.assertEqual(previous[0] + previous[1], current[0])

    def test_small_batches_use_only_the_cores_that_have_work(self):
        self.assertEqual(newton_schulz_kernel._balanced_ranges(3, 120), [(0, 1), (1, 1), (2, 1)])

    def test_fixed_count_and_compile_argument_layout_match_the_host_driver(self):
        compute_source = (
            Path(__file__).parents[1]
            / "enodia"
            / "tt"
            / "bench"
            / "kernels"
            / "newton_schulz_compute.cpp"
        ).read_text()
        reader_source = (
            Path(__file__).parents[1]
            / "enodia"
            / "tt"
            / "bench"
            / "kernels"
            / "newton_schulz_reader.cpp"
        ).read_text()

        self.assertEqual(newton_schulz_kernel.NEWTON_SCHULZ_ITERATIONS, 12)
        self.assertEqual(
            newton_schulz_kernel.COMPLEX_MATMULS_PER_INVERSE,
            2 * newton_schulz_kernel.NEWTON_SCHULZ_ITERATIONS,
        )
        complex_start = compute_source.index("void complex_real_impl")
        complex_end = compute_source.index("template <bool profile_sample>", complex_start)
        complex_source = compute_source[complex_start:complex_end]
        self.assertEqual(complex_source.count("tile_regs_acquire();"), 2)
        self.assertEqual(complex_source.count("tile_regs_commit();"), 2)
        self.assertEqual(complex_source.count("tile_regs_release();"), 0)
        self.assertEqual(complex_source.count("cb_reserve_back(output, 1);"), 2)
        self.assertEqual(complex_source.count("pack_one(output)"), 2)
        self.assertIn("matmul_block(left_real, right_real, 0, 0, 0", complex_source)
        self.assertIn("matmul_block(left_real, right_imag, 0, 0, 0", complex_source)
        self.assertNotIn("dst1", complex_source)
        self.assertEqual(compute_source.count("matmul_block(left_real, right_real"), 1)
        self.assertIn("matmul_block(left_imag, right_imag", compute_source)
        self.assertIn("matmul_block(left_real, right_imag", compute_source)
        self.assertIn("matmul_block(left_imag, right_real", compute_source)
        self.assertIn("cb_negative_x_imag", compute_source)
        self.assertIn("reconfig_data_format(current_srca, cb_zero, current_srcb, x_imag)", compute_source)
        self.assertIn("sub_tiles(cb_zero, x_imag", compute_source)
        self.assertNotIn("negative_tile", compute_source)
        self.assertIn("negate_state_imag_impl(x_imag", compute_source)
        self.assertEqual(compute_source.count("cb_wait_front(cb_r_real, 1)"), 1)
        self.assertIn("bool resident_left", compute_source)
        self.assertIn("bool consume_right", compute_source)
        self.assertIn("bool consume_left", compute_source)
        self.assertIn("consume_right", compute_source)
        self.assertNotIn("cb_pop_front(cb_identity", compute_source)
        self.assertNotIn("cb_pop_front(cb_zero", compute_source)
        self.assertIn("complex_matmul<true>", compute_source)
        self.assertIn("complex_matmul<false>", compute_source)
        self.assertNotIn("break;", compute_source)
        self.assertIn("get_compile_time_arg_val(0)", compute_source)
        self.assertIn("get_compile_time_arg_val(1)", compute_source)
        self.assertIn("get_compile_time_arg_val(2)", compute_source)
        kernel_source = Path(
            Path(__file__).parents[1]
            / "enodia"
            / "tt"
            / "bench"
            / "newton_schulz_kernel.py"
        ).read_text()
        self.assertIn("profile: bool = False", kernel_source)
        self.assertIn("fuse_s: bool = False", kernel_source)
        self.assertIn("batch_reads: bool = False", kernel_source)
        self.assertIn("state_fp32", compute_source)
        self.assertIn("get_arg_val<std::uint32_t>(1)", compute_source)
        self.assertIn("TensorAccessorArgs<1>()", reader_source)
        self.assertIn("r_negative_imag_address", reader_source)
        self.assertNotIn("copy_tile", reader_source)
        self.assertNotIn("route_", reader_source)

    def test_reload_r_compile_args_and_default_resident_path_are_explicit(self):
        resident_reader = newton_schulz_kernel._reader_compile_args(
            iterations=NEWTON_SCHULZ_ITERATIONS,
            profile=False,
            fuse_s=False,
            batch_reads=False,
            matrix_block=1,
            reload_r=False,
        )
        reload_reader = newton_schulz_kernel._reader_compile_args(
            iterations=NEWTON_SCHULZ_ITERATIONS,
            profile=False,
            fuse_s=True,
            batch_reads=False,
            matrix_block=4,
            reload_r=True,
        )
        resident_compute = newton_schulz_kernel._compute_compile_args(
            iterations=NEWTON_SCHULZ_ITERATIONS,
            state_fp32=True,
            profile=False,
            fuse_s=True,
            matrix_block=4,
            reload_r=False,
        )
        reload_compute = newton_schulz_kernel._compute_compile_args(
            iterations=NEWTON_SCHULZ_ITERATIONS,
            state_fp32=True,
            profile=False,
            fuse_s=True,
            matrix_block=4,
            reload_r=True,
        )

        assert resident_reader == [NEWTON_SCHULZ_ITERATIONS]
        assert reload_reader == [NEWTON_SCHULZ_ITERATIONS, 1, 0, 4, 1]
        assert resident_compute == [NEWTON_SCHULZ_ITERATIONS, 1, 0, 1, 4, 0]
        assert reload_compute == [NEWTON_SCHULZ_ITERATIONS, 1, 0, 1, 4, 1]

        compute = (
            Path(__file__).parents[1]
            / "enodia/tt/bench/kernels/newton_schulz_compute.cpp"
        ).read_text()
        reader = (
            Path(__file__).parents[1]
            / "enodia/tt/bench/kernels/newton_schulz_reader_optimized.cpp"
        ).read_text()
        assert "constexpr bool reload_r = get_compile_time_arg_val(5) != 0;" in compute
        assert "constexpr bool reload_r = get_compile_time_arg_val(4) != 0;" in reader
        assert compute.index("if constexpr (!reload_r)") < compute.index(
            "for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)"
        )

    def test_reload_r_cb_ledger_replays_and_consumes_r_without_changing_capacity(self):
        compute = (
            Path(__file__).parents[1]
            / "enodia/tt/bench/kernels/newton_schulz_compute.cpp"
        ).read_text()
        reader = (
            Path(__file__).parents[1]
            / "enodia/tt/bench/kernels/newton_schulz_reader_optimized.cpp"
        ).read_text()
        assert "pop_r_inputs<fuse_s>();" in compute
        assert "pop_r_inputs_block<fuse_s>(block_count);" in compute
        assert "read_matrix_reload" in reader
        assert "read_matrix_block_reload" in reader
        assert reader.count("for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)") >= 2
        assert "if (first_iteration)" in reader

        ttnn = SimpleNamespace(bfloat16="bf16", float32="fp32")
        for matrix_block in (1, 2, 4, 8):
            definitions = newton_schulz_kernel._cb_definitions(
                ttnn, "fp32", fuse_s=True, matrix_block=matrix_block
            )
            expected_pages = 2 if matrix_block == 1 else matrix_block
            assert definitions[newton_schulz_kernel.CB_R_NEG_IMAG][1] == expected_pages
            assert definitions[newton_schulz_kernel.CB_R_IMAG][1] == expected_pages
            assert definitions[newton_schulz_kernel.CB_R_NEG_REAL][1] == expected_pages

    def test_reload_reader_accessor_abi_starts_after_all_mode_arguments(self):
        reader_args = newton_schulz_kernel._reader_compile_args(
            iterations=NEWTON_SCHULZ_ITERATIONS,
            profile=False,
            fuse_s=True,
            batch_reads=False,
            matrix_block=4,
            reload_r=True,
        )
        assert reader_args == [NEWTON_SCHULZ_ITERATIONS, 1, 0, 4, 1]
        tensor_compile_args = [
            *reader_args,
            0,
            2 * 32 * 32,
            0,
            2 * 32 * 32,
            0,
            2 * 32 * 32,
            0,
            4 * 32 * 32,
            0,
            4 * 32 * 32,
        ]
        first_accessor = len(reader_args)
        assert tensor_compile_args[first_accessor] == 0
        assert tensor_compile_args[first_accessor + 1] == 2_048

        kernel_dir = Path(__file__).parents[1] / "enodia/tt/bench/kernels"
        for reader_name in (
            "newton_schulz_reader_optimized.cpp",
            "newton_schulz_reader_profile.cpp",
        ):
            source = (kernel_dir / reader_name).read_text()
            assert "first_input_compile_arg = 5" in source
            assert "TensorAccessorArgs<first_input_compile_arg>()" in source
            assert (
                "(fuse_s ? first_input_compile_arg : first_input_args.next_compile_time_args_offset())"
                in source
            )
            assert "TensorAccessorArgs<4>()" not in source

    def test_reload_reader_fused_batch4_block4_l32_model_has_no_zero_reads(self):
        block_count = 4
        r_page_bytes = 32 * 32 * 2
        x0_page_bytes = 32 * 32 * 4
        modeled_iterations = []
        for iteration in range(NEWTON_SCHULZ_ITERATIONS):
            r_pages = 3 * block_count
            x0_pages = 2 * block_count if iteration == 0 else 0
            modeled_iterations.append(
                {
                    "page_count": r_pages + x0_pages,
                    "byte_count": r_pages * r_page_bytes + x0_pages * x0_page_bytes,
                    "r_pages": r_pages,
                    "x0_pages": x0_pages,
                }
            )

        assert [entry["page_count"] for entry in modeled_iterations] == [
            20,
            *([12] * (NEWTON_SCHULZ_ITERATIONS - 1)),
        ]
        assert [entry["byte_count"] for entry in modeled_iterations] == [
            57_344,
            *([24_576] * (NEWTON_SCHULZ_ITERATIONS - 1)),
        ]
        assert all(entry["page_count"] > 0 for entry in modeled_iterations)
        assert all(entry["byte_count"] > 0 for entry in modeled_iterations)
        assert all(entry["r_pages"] == 12 for entry in modeled_iterations)
        assert modeled_iterations[0]["x0_pages"] == 8
        assert all(entry["x0_pages"] == 0 for entry in modeled_iterations[1:])

        # Fused S has three signed BF16 R pages per matrix and no positive
        # R-real page.  The model therefore matches the CB ledger and cannot
        # dispatch a zero-page group for the padded four-tile work range.
        assert newton_schulz_kernel.CB_R_REAL not in newton_schulz_kernel._cb_definitions(
            SimpleNamespace(bfloat16="bf16", float32="fp32"),
            "fp32",
            fuse_s=True,
            matrix_block=4,
        )
        assert newton_schulz_kernel._padded_tile_count(4, 4) == 4
        assert newton_schulz_kernel._matrix_block_ranges(0, 4, 4) == [(0, 4)]

    def test_fused_s_host_descriptors_prepare_signed_r_inputs_and_reader_dispatch(self):
        ttnn = SimpleNamespace(bfloat16="bf16", float32="fp32")
        fused = newton_schulz_kernel._cb_definitions(ttnn, ttnn.float32, fuse_s=True)
        baseline = newton_schulz_kernel._cb_definitions(ttnn, ttnn.float32)
        self.assertNotIn(newton_schulz_kernel.CB_R_REAL, fused)
        self.assertIn(newton_schulz_kernel.CB_R_REAL, baseline)
        self.assertEqual(fused[newton_schulz_kernel.CB_IDENTITY], ("bf16", 1))
        self.assertEqual(fused[newton_schulz_kernel.CB_R_NEG_REAL], ("bf16", 2))
        self.assertEqual(baseline[newton_schulz_kernel.CB_IDENTITY], ("fp32", 1))
        self.assertEqual(baseline[newton_schulz_kernel.CB_R_NEG_REAL], ("bf16", 1))
        optimized_reader = (
            Path(__file__).parents[1]
            / "enodia"
            / "tt"
            / "bench"
            / "kernels"
            / "newton_schulz_reader_optimized.cpp"
        ).read_text()
        self.assertIn("constexpr bool fuse_s", optimized_reader)
        self.assertIn("constexpr bool batch_reads", optimized_reader)
        self.assertIn("cb_reserve_back(cb_x0_imag, 1)", optimized_reader)
        self.assertIn("noc_async_read_barrier();", optimized_reader)
        self.assertIn("r_negative_imag_address = get_arg_val<std::uint32_t>(0)", optimized_reader)
        self.assertIn("r_real_address = get_arg_val<std::uint32_t>(0)", optimized_reader)
        self.assertIn("first_input_compile_arg = 5", optimized_reader)
        self.assertIn(
            "(fuse_s ? first_input_compile_arg : first_input_args.next_compile_time_args_offset())",
            optimized_reader,
        )
        self.assertNotIn("TensorAccessorArgs<4>()", optimized_reader)

        compute_source = (
            Path(__file__).parents[1]
            / "enodia"
            / "tt"
            / "bench"
            / "kernels"
            / "newton_schulz_compute.cpp"
        ).read_text()
        fused_start = compute_source.index("void fused_s_matmul")
        fused_end = compute_source.index("void subtract_one_impl", fused_start)
        fused_source = compute_source[fused_start:fused_end]
        self.assertIn("reconfig_data_format_srca(x_real, cb_identity)", fused_source)
        self.assertIn("copy_tile_init(cb_identity)", fused_source)
        self.assertLess(
            fused_source.index("reconfig_data_format_srca(x_real, cb_identity)"),
            fused_source.index("copy_tile_init(cb_identity)"),
        )
        self.assertIn("cb_wait_front(cb_zero, 1)", fused_source)
        self.assertIn("copy_tile_to_dst_init_short_with_dt(cb_identity, cb_zero)", fused_source)
        self.assertIn("copy_tile(cb_zero, 0, 1)", fused_source)
        self.assertIn("reconfig_data_format(x_real, negative_r_real)", fused_source)
        self.assertNotIn("cb_product_real", fused_source)
        self.assertIn("matmul_block(positive_r_imag, x_imag, 0, 0, 0", fused_source)
        self.assertIn("matmul_block(negative_r_imag, x_real, 0, 0, 1", fused_source)
        self.assertNotIn("init_common", compute_source)
        self.assertIn("cb_pop_front(cb_r_negative_real, 1)", compute_source)

    def test_packed_odd_batch_round_trips_and_isolates_blocks(self):
        matrices = np.arange(3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)
        packed = newton_schulz_kernel._pack_matrices(matrices, packed=True, tile_count=2)

        self.assertEqual(packed.shape, (2, 1, 32, 32))
        np.testing.assert_array_equal(packed[0, 0, :16, :16], matrices[0])
        np.testing.assert_array_equal(packed[0, 0, 16:, 16:], matrices[1])
        np.testing.assert_array_equal(packed[1, 0, :16, :16], matrices[2])
        np.testing.assert_array_equal(packed[1, 0, 16:, 16:], np.zeros((16, 16)))
        np.testing.assert_array_equal(packed[:, 0, :16, 16:], np.zeros((2, 16, 16)))
        np.testing.assert_array_equal(packed[:, 0, 16:, :16], np.zeros((2, 16, 16)))

        unpacked = newton_schulz_kernel._unpack_matrices(
            packed, batch=3, size=16, packed=True
        )
        np.testing.assert_array_equal(unpacked, matrices)

    def test_packed_l16_newton_schulz_matches_independent_reference(self):
        matrices = random_hpd_batch(3, 16, seed=17)
        x0 = newton_schulz_kernel._initial_value(matrices)
        packed_r = newton_schulz_kernel._pack_matrices(
            matrices.real, packed=True, tile_count=2
        )[:, 0]
        packed_r = packed_r + 1j * newton_schulz_kernel._pack_matrices(
            matrices.imag, packed=True, tile_count=2
        )[:, 0]
        packed_x = newton_schulz_kernel._pack_matrices(
            x0.real, packed=True, tile_count=2
        )[:, 0]
        packed_x = packed_x + 1j * newton_schulz_kernel._pack_matrices(
            x0.imag, packed=True, tile_count=2
        )[:, 0]
        identity = 2.0 * np.eye(32, dtype=np.complex64)

        for _ in range(NEWTON_SCHULZ_ITERATIONS):
            packed_x = packed_x @ (identity - packed_r @ packed_x)

        actual = newton_schulz_kernel._unpack_matrices(
            packed_x.real[:, None], batch=3, size=16, packed=True
        ) + 1j * newton_schulz_kernel._unpack_matrices(
            packed_x.imag[:, None], batch=3, size=16, packed=True
        )
        expected = newton_schulz_reference(matrices)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)

    def test_packed_odd_batch_matmul_matches_independent_products(self):
        left = (np.arange(3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16) % 5)
        right = (np.arange(3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16) % 7)
        packed_left = newton_schulz_kernel._pack_matrices(left, packed=True, tile_count=2)
        packed_right = newton_schulz_kernel._pack_matrices(right, packed=True, tile_count=2)

        packed_products = np.matmul(packed_left[:, 0], packed_right[:, 0])
        expected = newton_schulz_kernel._pack_matrices(
            np.matmul(left, right), packed=True, tile_count=2
        )

        np.testing.assert_array_equal(packed_products[:, None], expected)

    def test_nonpacked_l32_preserves_each_tile_and_zero_fills_extra_tiles(self):
        matrices = np.arange(3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)
        padded = newton_schulz_kernel._pack_matrices(matrices, packed=False, tile_count=4)

        self.assertEqual(padded.shape, (4, 1, 32, 32))
        np.testing.assert_array_equal(padded[:3, 0], matrices)
        np.testing.assert_array_equal(padded[3], np.zeros((1, 32, 32)))
        np.testing.assert_array_equal(
            newton_schulz_kernel._unpack_matrices(padded, batch=3, size=32, packed=False),
            matrices,
        )

    def test_l16_uses_paired_32x32_reader_tiles_and_keeps_l32_tile_count(self):
        matrices = newton_schulz_kernel.benchmark_matrices(3, 16)
        x0 = newton_schulz_kernel._initial_value(matrices)

        assert newton_schulz_kernel._physical_tile_count(3, 16) == 2
        assert newton_schulz_kernel._physical_tile_count(3, 32) == 3
        assert matrices.shape == (3, 16, 16)
        values = newton_schulz_kernel._reader_input_values(
            matrices,
            x0,
            fuse_s=True,
            tile_count=2,
            packed=True,
        )
        assert len(values) == 5
        assert all(value.shape == (2, 1, 32, 32) for value in values)
        for value in values:
            np.testing.assert_array_equal(value[0, 0, :16, 16:], 0.0)
            np.testing.assert_array_equal(value[0, 0, 16:, :16], 0.0)

    def test_prepare_rejects_unimplemented_shapes_variants_and_iteration_counts(self):
        matrices = np.zeros((1, 16, 16), dtype=np.complex64)

        with self.assertRaisesRegex(ValueError, "fixed at 12"):
            newton_schulz_kernel.NewtonSchulzKernel.prepare(
                None, None, matrices, iterations=7
            )
        with self.assertRaisesRegex(ValueError, "unknown kernel variant"):
            newton_schulz_kernel.NewtonSchulzKernel.prepare(
                None, None, matrices, variant="resident"
            )
        with self.assertRaisesRegex(ValueError, "matrices must have shape"):
            newton_schulz_kernel.NewtonSchulzKernel.prepare(
                None, None, np.zeros((16, 16), dtype=np.complex64)
            )
        with self.assertRaisesRegex(ValueError, "matrices must have shape"):
            newton_schulz_kernel.NewtonSchulzKernel.prepare(
                None, None, np.zeros((1, 16, 15), dtype=np.complex64)
            )
        with self.assertRaisesRegex(ValueError, "only supports L=16 or L=32"):
            newton_schulz_kernel.NewtonSchulzKernel.prepare(
                None, None, np.zeros((1, 8, 8), dtype=np.complex64)
            )
        with self.assertRaisesRegex(ValueError, "unknown kernel variant"):
            newton_schulz_kernel.NewtonSchulzKernel.prepare(
                None, None, np.zeros((1, 32, 32), dtype=np.complex64), variant="packed_fused"
            )

    def test_prepare_rejects_empty_batch_before_device_access(self):
        with self.assertRaisesRegex(ValueError, "batch must be positive"):
            newton_schulz_kernel.NewtonSchulzKernel.prepare(
                None, None, np.zeros((0, 16, 16), dtype=np.complex64)
            )

    def test_accelerator_modules_do_not_import_reference_or_spec_modules(self):
        accelerator_root = Path(__file__).parents[1] / "enodia" / "tt"
        forbidden_reference = "enodia.tt.bench.newton_schulz_reference"
        forbidden_spec_prefix = "enodia.spec"
        reference_violations = []
        spec_violations = []
        for path in accelerator_root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative_path = str(path.relative_to(accelerator_root))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == forbidden_reference:
                        reference_violations.append(relative_path)
                    if module == forbidden_spec_prefix or module.startswith(f"{forbidden_spec_prefix}."):
                        spec_violations.append(relative_path)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == forbidden_reference:
                            reference_violations.append(relative_path)
                        if alias.name == forbidden_spec_prefix or alias.name.startswith(
                            f"{forbidden_spec_prefix}."
                        ):
                            spec_violations.append(relative_path)

        self.assertEqual(reference_violations, [])
        self.assertEqual(spec_violations, [])


@pytest.mark.parametrize(
    ("batch", "matrix_block", "size", "fuse_s"),
    PARTIAL_BLOCK_CASES,
    ids=PARTIAL_BLOCK_CASE_IDS,
)
def test_partial_block_padding_preserves_capacity_and_logical_unpack(
    batch, matrix_block, size, fuse_s
):
    physical_tile_count = newton_schulz_kernel._physical_tile_count(batch, size)
    tile_count = newton_schulz_kernel._padded_tile_count(
        physical_tile_count, matrix_block
    )
    assert tile_count >= physical_tile_count
    assert tile_count % matrix_block == 0

    work_ranges = newton_schulz_kernel._balanced_ranges(tile_count, 110, matrix_block)
    newton_schulz_kernel._validate_core_group_capacities(work_ranges, matrix_block)
    capacities = [matrix_block]
    if matrix_block == 8:
        capacities.append(2 * matrix_block)
    for start, count in work_ranges:
        for _, group_count in newton_schulz_kernel._matrix_block_ranges(
            start, count, matrix_block
        ):
            assert all(capacity % group_count == 0 for capacity in capacities)

    matrices = (
        np.arange(batch * size * size, dtype=np.float32).reshape(batch, size, size)
        + 1j
    ).astype(np.complex64)
    x0 = np.zeros_like(matrices)
    reader_values = newton_schulz_kernel._reader_input_values(
        matrices,
        x0,
        fuse_s=fuse_s,
        tile_count=tile_count,
        packed=size == 16,
    )
    assert len(reader_values) == 5
    assert all(value.shape[0] == tile_count for value in reader_values)
    if tile_count > physical_tile_count:
        for value in reader_values:
            np.testing.assert_array_equal(value[physical_tile_count:], 0.0)

    real = newton_schulz_kernel._unpack_matrices(
        newton_schulz_kernel._pack_matrices(
            matrices.real, packed=size == 16, tile_count=tile_count
        ),
        batch=batch,
        size=size,
        packed=size == 16,
    )
    imag = newton_schulz_kernel._unpack_matrices(
        newton_schulz_kernel._pack_matrices(
            matrices.imag, packed=size == 16, tile_count=tile_count
        ),
        batch=batch,
        size=size,
        packed=size == 16,
    )
    result = real + 1j * imag
    assert result.shape == (batch, size, size)
    np.testing.assert_array_equal(result, matrices)


@pytest.mark.parametrize(
    ("matrix_block", "group_count"), ((4, 3), (8, 3), (8, 5), (8, 6), (8, 7))
)
def test_prepare_capacity_guard_rejects_partial_matrix_block_groups(
    matrix_block, group_count
):
    with pytest.raises(ValueError, match="does not divide"):
        newton_schulz_kernel._validate_core_group_capacities(
            [(0, group_count)], matrix_block
        )


def test_prepare_deallocates_inputs_when_a_later_input_allocation_fails(monkeypatch):
    class _Ttnn:
        bfloat16 = "bf16"
        float32 = "fp32"
        MathFidelity = SimpleNamespace(HiFi4="hifi4")

        def __init__(self):
            self.deallocated = []

        def deallocate(self, tensor):
            self.deallocated.append(tensor)

    ttnn = _Ttnn()
    monkeypatch.setattr(
        newton_schulz_kernel,
        "_core_grid",
        lambda *_args, **_kwargs: ([(0, 0)], object(), [(0, 1)]),
    )
    monkeypatch.setattr(
        newton_schulz_kernel,
        "_cb_definitions",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        newton_schulz_kernel,
        "_tensor_l1_bytes",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        newton_schulz_kernel,
        "_validate_l1_budget",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        newton_schulz_kernel,
        "_reader_input_dtypes",
        lambda *_args, **_kwargs: ["dtype"] * 7,
    )
    monkeypatch.setattr(
        newton_schulz_kernel,
        "_reader_input_memories",
        lambda **_kwargs: ["l1"] * 7,
    )
    created = []

    def fail_on_second_input(*_args, **_kwargs):
        if created:
            raise RuntimeError("input allocation failed")
        created.append("input-0")
        return created[-1]

    monkeypatch.setattr(newton_schulz_kernel, "_device_tensor", fail_on_second_input)
    matrices = np.ones((1, 32, 32), dtype=np.complex64)
    with pytest.raises(RuntimeError, match="input allocation failed"):
        newton_schulz_kernel.NewtonSchulzKernel.prepare(ttnn, None, matrices)
    assert ttnn.deallocated == ["input-0"]


@pytest.mark.tt_device
@pytest.mark.skipif(
    not (DEVICE_TEST and HAS_TTNN),
    reason="requires run_in_container.sh --pytest in the pinned toolchain with a board",
)
@pytest.mark.parametrize(
    ("batch", "matrix_block", "size", "fuse_s"),
    PARTIAL_BLOCK_CASES,
    ids=PARTIAL_BLOCK_CASE_IDS,
)
def test_device_partial_block_padding_matches_numpy(
    batch, matrix_block, size, fuse_s
):
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        matrices = random_hpd_batch(batch, size, seed=95 + batch + size + matrix_block)
        expected = newton_schulz_reference(
            bf16_round_complex(matrices), x0=initial_value(matrices)
        )
        actual = run_newton_schulz_kernel(
            ttnn,
            device,
            matrices,
            variant="bf16-fp32state",
            math_fidelity="HiFi3",
            fuse_s=fuse_s,
            matrix_block=matrix_block,
            input_memory="l1",
            r_memory="l1",
            x0_memory="l1",
        )
        assert actual.shape == (batch, size, size)
        relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert relative_error <= 1e-2
    finally:
        ttnn.close_device(device)


@pytest.mark.tt_device
@unittest.skipUnless(
    DEVICE_TEST and HAS_TTNN,
    "requires run_in_container.sh --pytest in the pinned toolchain with a board",
)
class DeviceEquivalenceTests(unittest.TestCase):
    def test_batch_8192_matches_numpy_at_l32_block4_fused_hifi3_fp32_state(self):
        import ttnn

        device = ttnn.open_device(device_id=0)
        try:
            matrices = random_hpd_batch(8192, 32, seed=95)
            expected = newton_schulz_reference(
                bf16_round_complex(matrices), x0=initial_value(matrices)
            )
            actual = run_newton_schulz_kernel(
                ttnn,
                device,
                matrices,
                variant="bf16-fp32state",
                math_fidelity="HiFi3",
                fuse_s=True,
                matrix_block=4,
                input_memory="l1",
                r_memory="l1",
                x0_memory="l1",
            )
            relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            self.assertLessEqual(relative_error, 1e-2)
        finally:
            ttnn.close_device(device)

    def test_batch_8192_true_inverse_error_is_recorded_without_a_threshold(self):
        import ttnn

        device = ttnn.open_device(device_id=0)
        try:
            matrices = random_hpd_batch(8192, 32, seed=95)
            actual = run_newton_schulz_kernel(
                ttnn,
                device,
                matrices,
                variant="bf16-fp32state",
                math_fidelity="HiFi3",
                fuse_s=True,
                matrix_block=4,
                input_memory="l1",
                r_memory="l1",
                x0_memory="l1",
            )
            true_inverse = np.linalg.inv(matrices.astype(np.complex128))
            relative_error = np.linalg.norm(
                actual.astype(np.complex128) - true_inverse
            ) / np.linalg.norm(true_inverse)
            print(f"true inverse relative error (L=32, batch=8192): {relative_error:.8e}")
        finally:
            ttnn.close_device(device)

    def test_batch_8192_matches_numpy_at_l16_diagonal_pairs_block4_fused_hifi3_fp32_state(
        self,
    ):
        import ttnn

        device = ttnn.open_device(device_id=0)
        try:
            matrices = random_hpd_batch(8192, 16, seed=95)
            expected = newton_schulz_reference(
                bf16_round_complex(matrices), x0=initial_value(matrices)
            )
            actual = run_newton_schulz_kernel(
                ttnn,
                device,
                matrices,
                variant="bf16-fp32state",
                math_fidelity="HiFi3",
                fuse_s=True,
                matrix_block=4,
                input_memory="l1",
                r_memory="l1",
                x0_memory="l1",
            )
            relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            self.assertLessEqual(relative_error, 1e-2)
        finally:
            ttnn.close_device(device)


@pytest.mark.tt_device
@pytest.mark.skipif(
    not (DEVICE_TEST and HAS_TTNN),
    reason="requires run_in_container.sh --pytest in the pinned toolchain with a board",
)
@pytest.mark.parametrize("batch", (4, 8192), ids=("batch4", "batch8192"))
@pytest.mark.parametrize("reload_r", (False, True), ids=("resident", "reload_r"))
def test_device_reload_r_matches_numpy_for_both_residency_paths(batch, reload_r):
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        matrices = random_hpd_batch(batch, 32, seed=95 + batch)
        expected = newton_schulz_reference(
            bf16_round_complex(matrices), x0=initial_value(matrices)
        )
        actual = run_newton_schulz_kernel(
            ttnn,
            device,
            matrices,
            variant="bf16-fp32state",
            math_fidelity="HiFi3",
            fuse_s=True,
            reload_r=reload_r,
            matrix_block=4,
            input_memory="l1",
            r_memory="l1",
            x0_memory="l1",
        )
        relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert relative_error <= 1e-2
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    unittest.main()
