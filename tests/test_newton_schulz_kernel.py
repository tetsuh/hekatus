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

DEVICE_TEST = os.environ.get("HEKATUS_TT_DEVICE_TEST") == "1"
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
        self.assertEqual(compute_source.count("matmul_block(left_real, right_real"), 1)
        self.assertIn("matmul_block(left_imag_for_real, right_imag", compute_source)
        self.assertIn("matmul_block(left_real, right_imag", compute_source)
        self.assertIn("matmul_block(left_imag_for_imag, right_real", compute_source)
        self.assertIn("cb_negative_x_imag", compute_source)
        self.assertIn("reconfig_data_format(cb_zero, cb_zero, cb_product_imag, x_imag)", compute_source)
        self.assertIn("sub_tiles(cb_zero, x_imag", compute_source)
        self.assertNotIn("negative_tile", compute_source)
        self.assertIn("negate_state_imag(x_imag);", compute_source)
        self.assertEqual(compute_source.count("cb_wait_front(cb_r_real, 1)"), 1)
        self.assertIn("bool resident_left", compute_source)
        self.assertIn("bool consume_right", compute_source)
        self.assertIn("bool consume_left", compute_source)
        self.assertIn("consume_right", compute_source)
        self.assertNotIn("cb_pop_front(cb_identity", compute_source)
        self.assertNotIn("cb_pop_front(cb_zero", compute_source)
        self.assertIn("cb_product_imag,\n                true,\n                false,\n                false);", compute_source)
        self.assertIn("cb_s_imag,\n                output_real,\n                output_imag,\n                false,\n                true,\n                true);", compute_source)
        self.assertNotIn("break;", compute_source)
        self.assertIn("get_compile_time_arg_val(0)", compute_source)
        self.assertIn("get_compile_time_arg_val(1)", compute_source)
        self.assertIn("state_fp32", compute_source)
        self.assertIn("get_arg_val<std::uint32_t>(1)", compute_source)
        self.assertIn("TensorAccessorArgs<1>()", reader_source)
        self.assertIn("r_negative_imag_address", reader_source)
        self.assertNotIn("copy_tile", reader_source)
        self.assertNotIn("route_", reader_source)

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
        with self.assertRaisesRegex(ValueError, "only supports L=32"):
            newton_schulz_kernel.NewtonSchulzKernel.prepare(
                None, None, np.zeros((1, 16, 16), dtype=np.complex64)
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

    def test_accelerator_modules_do_not_import_the_numpy_reference(self):
        accelerator_root = Path(__file__).parents[1] / "enodia" / "tt"
        forbidden_module = "enodia.tt.bench.newton_schulz_reference"
        violations = []
        for path in accelerator_root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == forbidden_module:
                    violations.append(str(path.relative_to(accelerator_root)))
                elif isinstance(node, ast.Import):
                    violations.extend(
                        str(path.relative_to(accelerator_root))
                        for alias in node.names
                        if alias.name == forbidden_module
                    )

        self.assertEqual(violations, [])


@unittest.skipUnless(DEVICE_TEST and HAS_TTNN, "requires the pinned TT container and a board")
class DeviceEquivalenceTests(unittest.TestCase):
    def test_batch_8192_matches_numpy_at_l32_with_fp32_state(self):
        import ttnn

        device = ttnn.open_device(device_id=0)
        try:
            matrices = random_hpd_batch(8192, 32, seed=95)
            expected = newton_schulz_reference(matrices)
            actual = run_newton_schulz_kernel(
                ttnn, device, matrices, variant="bf16-fp32state"
            )
            relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            self.assertLessEqual(relative_error, 1e-2)
        finally:
            ttnn.close_device(device)


if __name__ == "__main__":
    unittest.main()
