"""L0 equivalence and accounting for the fixed-count Newton-Schulz kernel."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
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


class ReferenceTests(unittest.TestCase):
    def test_accelerator_initial_value_matches_the_independent_oracle(self):
        matrices = random_hpd_batch(3, 16, seed=19)

        actual = newton_schulz_kernel._initial_value(matrices)
        expected = initial_value(matrices)

        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

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

        self.assertEqual(newton_schulz_kernel.NEWTON_SCHULZ_ITERATIONS, 8)
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

    def test_fidelity_split_accepts_all_hifi3_and_parameterized_hifi2_prefixes(self):
        normalize = newton_schulz_kernel._normalize_fidelity_split

        self.assertIsNone(normalize(None))
        self.assertEqual(normalize("0+8"), (0, 8))
        self.assertEqual(normalize("4+4"), (4, 4))
        self.assertEqual(normalize((6, 2)), (6, 2))
        for value in ("4+3", "8+0", "-1+9", "four+four", (1, 1, 6)):
            with self.assertRaisesRegex(ValueError, "fidelity split"):
                normalize(value)

    def test_fidelity_split_uses_matching_direct_llk_pairs_and_keeps_default_source(self):
        root = Path(__file__).parents[1] / "enodia" / "tt" / "bench" / "kernels"
        default_source = (root / "newton_schulz_compute.cpp").read_text()
        split_source = (root / "newton_schulz_fidelity_split_compute.cpp").read_text()

        self.assertNotIn("llk_math_matmul_init<", default_source)
        self.assertIn('#include "newton_schulz_compute.cpp"', split_source)
        init_fidelities = set(
            re.findall(r"split_matmul_block_init_impl<MathFidelity::(HiFi[23])>", split_source)
        )
        execute_fidelities = set(
            re.findall(r"split_matmul_block_impl<MathFidelity::(HiFi[23])>", split_source)
        )
        self.assertEqual(init_fidelities, {"HiFi2", "HiFi3"})
        self.assertEqual(execute_fidelities, init_fidelities)
        self.assertIn("get_compile_time_arg_val(5)", split_source)
        self.assertNotRegex(split_source, r"matmul_block\([^\n]*MathFidelity")

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
        self.assertIn(
            "(fuse_s ? 4 : first_input_args.next_compile_time_args_offset())",
            optimized_reader,
        )

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

        with self.assertRaisesRegex(ValueError, "fixed at 8"):
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
            expected = newton_schulz_reference(matrices)
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

    def test_batch_8192_matches_numpy_at_l16_diagonal_pairs_block4_fused_hifi3_fp32_state(
        self,
    ):
        import ttnn

        device = ttnn.open_device(device_id=0)
        try:
            matrices = random_hpd_batch(8192, 16, seed=95)
            expected = newton_schulz_reference(matrices)
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

    def test_batch_8192_matches_numpy_for_hifi3_and_fidelity_splits(self):
        import ttnn

        device = ttnn.open_device(device_id=0)
        try:
            matrices = random_hpd_batch(8192, 32, seed=95)
            expected = newton_schulz_reference(matrices)
            cases = (
                ("all-hifi3", None),
                ("0+8", (0, 8)),
                ("4+4", (4, 4)),
                ("6+2", (6, 2)),
            )
            measurements = []
            for name, fidelity_split in cases:
                actual = run_newton_schulz_kernel(
                    ttnn,
                    device,
                    matrices,
                    variant="bf16-fp32state",
                    math_fidelity="HiFi3",
                    fidelity_split=fidelity_split,
                    fuse_s=True,
                    matrix_block=4,
                    input_memory="l1",
                    r_memory="l1",
                    x0_memory="l1",
                )
                relative_error = float(
                    np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                )
                measurements.append(
                    {
                        "name": name,
                        "fidelity_split": name if fidelity_split is not None else None,
                        "relative_error": relative_error,
                        "tolerance": 1e-2,
                        "status": "pass" if relative_error <= 1e-2 else "fail",
                    }
                )
            print(
                "FIDELITY_SPLIT_CORRECTNESS "
                + json.dumps(measurements, sort_keys=True),
                flush=True,
            )
            for measurement in measurements:
                self.assertLessEqual(
                    measurement["relative_error"],
                    1e-2,
                    msg=measurement["name"],
                )
        finally:
            ttnn.close_device(device)


if __name__ == "__main__":
    unittest.main()
