"""L0 equivalence and accounting for the fixed-count Newton-Schulz kernel."""

from __future__ import annotations

import ast
import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

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
    def test_batch_8192_matches_numpy_at_l16_and_l32(self):
        import ttnn

        device = ttnn.open_device(device_id=0)
        try:
            for size in (16, 32):
                with self.subTest(size=size):
                    matrices = random_hpd_batch(8192, size, seed=63 + size)
                    expected = newton_schulz_reference(matrices)
                    actual = run_newton_schulz_kernel(ttnn, device, matrices)
                    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                    self.assertLessEqual(relative_error, 1e-2)
        finally:
            ttnn.close_device(device)


if __name__ == "__main__":
    unittest.main()
