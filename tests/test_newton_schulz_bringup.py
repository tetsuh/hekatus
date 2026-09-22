import ast
import json
import unittest
from pathlib import Path

import numpy as np

from tools import newton_schulz_bringup as bringup


class BringupHostTests(unittest.TestCase):
    def test_stages_add_one_named_behavior_at_a_time(self):
        self.assertEqual(
            tuple(bringup.STAGES),
            (
                1,
                2,
                3,
                4,
                5,
                6,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
            ),
        )
        production = [bringup.STAGES[number] for number in range(1, 7)]
        self.assertEqual(
            [stage.name for stage in production],
            [
                "real_one_tile",
                "real_multiple_tiles",
                "real_multiple_cores",
                "complex_one_matmul",
                "complex_newton_schulz_one_iteration",
                "complex_newton_schulz_eight_iterations",
            ],
        )
        self.assertEqual([stage.batch for stage in production], [1, 4, 4, 1, 1, 1])
        self.assertEqual([stage.cores for stage in production], [1, 1, 2, 1, 1, 1])
        self.assertEqual([stage.iterations for stage in production], [1, 1, 1, 1, 1, 8])
        self.assertEqual(
            [
                (bringup.STAGES[number].name, bringup.STAGES[number].kind)
                for number in (41, 42, 43, 44, 45, 46, 47, 48)
            ],
            [
                ("complex_modern_startup", "complex_modern_startup"),
                ("complex_two_groups", "complex_two_groups"),
                ("newton_residual_only", "newton_residual"),
                ("newton_residual_reader_copy", "newton_residual_reader_copy"),
                ("newton_one_compute_copy", "newton_one_compute_copy"),
                ("newton_residual_correct", "newton_residual_correct"),
                ("newton_residual_correct_reader_copy", "newton_residual_correct_reader_copy"),
                ("newton_one_correct_reader_copy", "newton_one_correct_reader_copy"),
            ],
        )

    def test_fp32_dest_acc_stage_reuses_stage6_sources_and_config(self):
        stage6 = bringup.STAGES[6]
        stage49 = bringup.STAGES[49]
        self.assertEqual(
            (stage49.compute_source, stage49.reader_source, stage49.writer_source),
            (stage6.compute_source, stage6.reader_source, stage6.writer_source),
        )
        self.assertFalse(stage6.fp32_dest_acc_en)
        self.assertTrue(stage49.fp32_dest_acc_en)
        self.assertEqual(stage49.name, "complex_newton_schulz_eight_iterations_fp32_dest_acc")
        self.assertEqual((stage49.batch, stage49.iterations, stage49.kind), (1, 8, "newton_schulz"))
        self.assertEqual(stage6.input_seed, stage49.input_seed)
        self.assertEqual(stage6.input_seed, 6306)
        for field in (
            "batch",
            "cores",
            "compute_source",
            "reader_source",
            "writer_source",
            "kind",
            "iterations",
            "input_seed",
        ):
            self.assertEqual(getattr(stage6, field), getattr(stage49, field), field)
        for stage6_input, stage49_input in zip(bringup._inputs(stage6), bringup._inputs(stage49)):
            np.testing.assert_array_equal(stage6_input, stage49_input)
        self.assertEqual(bringup.NUMERICAL_TOLERANCE, 1e-2)
        for number, stage in bringup.STAGES.items():
            if number not in (49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59):
                self.assertFalse(stage.fp32_dest_acc_en)
        source = Path("tools/newton_schulz_bringup.py").read_text()
        self.assertIn("fp32_dest_acc_en=stage.fp32_dest_acc_en", source)

    def test_precision_boundary_stages_use_explicit_float32_cb_configuration(self):
        stage50 = bringup.STAGES[50]
        stage51 = bringup.STAGES[51]
        self.assertTrue(stage50.fp32_dest_acc_en)
        self.assertTrue(stage51.fp32_dest_acc_en)
        self.assertEqual(stage50.input_count, 2)
        self.assertEqual(stage51.input_count, 3)
        self.assertEqual(stage50.output_dtype, "float32")
        self.assertEqual(stage51.output_dtype, "float32")
        self.assertEqual(stage50.cb_formats[16], "float32")
        self.assertEqual(stage51.cb_formats[16:18], ("float32", "float32"))
        self.assertEqual(stage50.cb_page_sizes[16], bringup.TILE_BYTES_FLOAT32)
        self.assertEqual(stage51.cb_page_sizes[16:18], (bringup.TILE_BYTES_FLOAT32,) * 2)
        self.assertEqual(stage50.cb_page_sizes[0], bringup.TILE_BYTES_BFLOAT16)
        self.assertEqual(stage51.cb_page_sizes[18], bringup.TILE_BYTES_BFLOAT16)
        self.assertEqual(len(set(bringup.source_paths(stage50))), 3)
        self.assertEqual(len(set(bringup.source_paths(stage51))), 3)

        one_compute = (bringup.KERNEL_DIR / stage50.compute_source).read_text()
        one_reader = (bringup.KERNEL_DIR / stage50.reader_source).read_text()
        two_compute = (bringup.KERNEL_DIR / stage51.compute_source).read_text()
        two_reader = (bringup.KERNEL_DIR / stage51.reader_source).read_text()
        for compute in (one_compute, two_compute):
            self.assertEqual(compute.count("compute_kernel_hw_startup<SrcOrder::Reverse>("), 1)
            self.assertNotIn("copy_tile", compute)
        self.assertEqual(one_compute.count("matmul_one();"), 1)
        self.assertEqual(two_compute.count("matmul_one("), 3)
        self.assertIn("matmul_block_init(cb_second_a, cb_second_b", two_compute)
        self.assertIn("(32 * 32 * 4) / sizeof(std::uint32_t)", two_reader)
        self.assertIn("copy_float32_tile(cb_second_a", two_reader)
        self.assertIn("cb_pop_front(cb_product, 1)", two_reader)
        self.assertEqual(one_reader.count("read_tile(cb_a"), 1)
        self.assertEqual(one_reader.count("read_tile(cb_b"), 1)
        self.assertEqual(two_reader.count("read_tile(cb_first_a"), 1)
        self.assertEqual(two_reader.count("read_tile(cb_first_b"), 1)
        self.assertEqual(two_reader.count("read_tile(cb_second_b"), 1)
        self.assertLess(
            two_reader.index("reuse_first_product()"), two_reader.index("read_tile(cb_second_b")
        )

        stage50_inputs = bringup._inputs(stage50)
        stage51_inputs = bringup._inputs(stage51)
        np.testing.assert_allclose(
            bringup.expected_output(stage50, stage50_inputs),
            np.matmul(stage50_inputs[0], stage50_inputs[1]),
        )
        np.testing.assert_allclose(
            bringup.expected_output(stage51, stage51_inputs),
            np.matmul(np.matmul(stage51_inputs[0], stage51_inputs[1]), stage51_inputs[2]),
        )

    def test_precision_reconfig_stage_explicitly_reconfigures_mixed_formats(self):
        stage = bringup.STAGES[52]
        compute = (bringup.KERNEL_DIR / stage.compute_source).read_text()
        self.assertEqual(
            (stage.batch, stage.cores, stage.input_count, stage.output_dtype), (1, 1, 3, "float32")
        )
        self.assertTrue(stage.fp32_dest_acc_en)
        self.assertEqual(stage.cb_formats[16:18], ("float32", "float32"))
        self.assertEqual(stage.cb_page_sizes[16:18], (bringup.TILE_BYTES_FLOAT32,) * 2)
        self.assertIn('#include "api/compute/reconfig_data_format.h"', compute)
        self.assertIn("reconfig_data_format_srca(cb_first_b, cb_second_a);", compute)
        self.assertIn("pack_reconfig_data_format(cb_product);", compute)
        self.assertLess(
            compute.index("reconfig_data_format_srca"),
            compute.index("matmul_block_init(cb_second_a, cb_second_b"),
        )
        self.assertLess(
            compute.index("matmul_block_init(cb_second_a, cb_second_b"),
            compute.index("pack_reconfig_data_format"),
        )
        inputs = bringup._inputs(stage)
        np.testing.assert_allclose(
            bringup.expected_output(stage, inputs),
            np.matmul(np.matmul(inputs[0], inputs[1]), inputs[2]),
        )

    def test_precision_reconfig_variants_isolate_unpacker_and_packer(self):
        unpack = (bringup.KERNEL_DIR / bringup.STAGES[53].compute_source).read_text()
        pack = (bringup.KERNEL_DIR / bringup.STAGES[54].compute_source).read_text()
        self.assertIn("reconfig_data_format_srca(cb_first_b, cb_second_a);", unpack)
        self.assertNotIn("pack_reconfig_data_format", unpack)
        self.assertIn("pack_reconfig_data_format(cb_product);", pack)
        self.assertNotIn("reconfig_data_format_srca", pack)
        for number in (53, 54):
            stage = bringup.STAGES[number]
            self.assertTrue(stage.fp32_dest_acc_en)
            self.assertEqual(stage.cb_formats[16:18], ("float32", "float32"))
            self.assertEqual(stage.cb_page_sizes[16:18], (bringup.TILE_BYTES_FLOAT32,) * 2)
            inputs = bringup._inputs(stage)
            np.testing.assert_allclose(
                bringup.expected_output(stage, inputs),
                np.matmul(np.matmul(inputs[0], inputs[1]), inputs[2]),
            )

    def test_precision_reconfig_can_switch_to_a_distinct_bfloat16_output_cb(self):
        stage = bringup.STAGES[55]
        compute = (bringup.KERNEL_DIR / stage.compute_source).read_text()
        writer = (bringup.KERNEL_DIR / stage.writer_source).read_text()
        self.assertEqual(
            (stage.batch, stage.cores, stage.input_count, stage.output_dtype), (1, 1, 3, "bfloat16")
        )
        self.assertTrue(stage.fp32_dest_acc_en)
        self.assertEqual(stage.cb_formats[16:20], ("float32", "float32", "bfloat16", "bfloat16"))
        self.assertEqual(
            stage.cb_page_sizes[16:20],
            (
                bringup.TILE_BYTES_FLOAT32,
                bringup.TILE_BYTES_FLOAT32,
                bringup.TILE_BYTES_BFLOAT16,
                bringup.TILE_BYTES_BFLOAT16,
            ),
        )
        self.assertIn("reconfig_data_format_srca(cb_first_b, cb_second_a);", compute)
        self.assertIn("pack_reconfig_data_format(cb_product, cb_second_product);", compute)
        self.assertIn("matmul_one(cb_second_a, cb_second_b, cb_second_product);", compute)
        self.assertIn("cb_wait_front(19, 1);", writer)
        self.assertIn("get_read_ptr(19)", writer)
        self.assertLess(
            compute.index("matmul_block_init(cb_second_a, cb_second_b"),
            compute.index("pack_reconfig_data_format(cb_product, cb_second_product)"),
        )
        inputs = bringup._inputs(stage)
        np.testing.assert_allclose(
            bringup.expected_output(stage, inputs),
            np.matmul(np.matmul(inputs[0], inputs[1]), inputs[2]),
        )

    def test_correct_srcb_reconfig_stages_isolate_input_and_output_transitions(self):
        stage56 = bringup.STAGES[56]
        stage57 = bringup.STAGES[57]
        compute56 = (bringup.KERNEL_DIR / stage56.compute_source).read_text()
        compute57 = (bringup.KERNEL_DIR / stage57.compute_source).read_text()
        writer57 = (bringup.KERNEL_DIR / stage57.writer_source).read_text()
        self.assertIn("reconfig_data_format_srcb(cb_first_a, cb_second_a);", compute56)
        self.assertNotIn("reconfig_data_format_srca", compute56)
        self.assertNotIn("pack_reconfig_data_format", compute56)
        self.assertIn("reconfig_data_format_srcb(cb_first_a, cb_second_a);", compute57)
        self.assertIn("pack_reconfig_data_format(cb_product, cb_second_product);", compute57)
        self.assertIn("matmul_one(cb_second_a, cb_second_b, cb_second_product);", compute57)
        self.assertIn("cb_wait_front(19, 1);", writer57)
        self.assertIn("get_read_ptr(19)", writer57)
        for stage in (stage56, stage57):
            self.assertTrue(stage.fp32_dest_acc_en)
            self.assertEqual(stage.cb_formats[16:18], ("float32", "float32"))
            self.assertEqual(stage.cb_page_sizes[16:18], (bringup.TILE_BYTES_FLOAT32,) * 2)
            inputs = bringup._inputs(stage)
            np.testing.assert_allclose(
                bringup.expected_output(stage, inputs),
                np.matmul(np.matmul(inputs[0], inputs[1]), inputs[2]),
            )
        self.assertEqual(stage56.output_dtype, "float32")
        self.assertEqual(stage57.output_dtype, "bfloat16")
        self.assertEqual(stage57.cb_formats[19], "bfloat16")
        self.assertEqual(stage57.cb_page_sizes[19], bringup.TILE_BYTES_BFLOAT16)

    def test_same_output_packer_reconfig_isolated_after_correct_srcb_transition(self):
        stage = bringup.STAGES[58]
        compute = (bringup.KERNEL_DIR / stage.compute_source).read_text()
        self.assertEqual(
            (stage.batch, stage.cores, stage.input_count, stage.output_dtype), (1, 1, 3, "float32")
        )
        self.assertTrue(stage.fp32_dest_acc_en)
        self.assertEqual(stage.input_seed, 6358)
        self.assertEqual(stage.cb_formats[16:19], ("float32", "float32", "bfloat16"))
        self.assertEqual(
            stage.cb_page_sizes[16:19],
            (bringup.TILE_BYTES_FLOAT32, bringup.TILE_BYTES_FLOAT32, bringup.TILE_BYTES_BFLOAT16),
        )
        self.assertIn("reconfig_data_format_srcb(cb_first_a, cb_second_a);", compute)
        self.assertIn("pack_reconfig_data_format(cb_product);", compute)
        self.assertNotIn("cb_second_product", compute)
        self.assertLess(
            compute.index("matmul_block_init(cb_second_a, cb_second_b"),
            compute.index("pack_reconfig_data_format(cb_product)"),
        )
        inputs = bringup._inputs(stage)
        np.testing.assert_allclose(
            bringup.expected_output(stage, inputs),
            np.matmul(np.matmul(inputs[0], inputs[1]), inputs[2]),
        )

    def test_distinct_float32_output_cb_isolated_after_correct_srcb_transition(self):
        stage = bringup.STAGES[59]
        compute = (bringup.KERNEL_DIR / stage.compute_source).read_text()
        writer = (bringup.KERNEL_DIR / stage.writer_source).read_text()
        self.assertEqual(
            (stage.batch, stage.cores, stage.input_count, stage.output_dtype), (1, 1, 3, "float32")
        )
        self.assertTrue(stage.fp32_dest_acc_en)
        self.assertEqual(stage.input_seed, 6359)
        self.assertEqual(stage.cb_formats[16:20], ("float32", "float32", "bfloat16", "float32"))
        self.assertEqual(
            stage.cb_page_sizes[16:20],
            (
                bringup.TILE_BYTES_FLOAT32,
                bringup.TILE_BYTES_FLOAT32,
                bringup.TILE_BYTES_BFLOAT16,
                bringup.TILE_BYTES_FLOAT32,
            ),
        )
        self.assertIn("reconfig_data_format_srcb(cb_first_a, cb_second_a);", compute)
        self.assertIn("pack_reconfig_data_format(cb_product, cb_second_product);", compute)
        self.assertIn("matmul_one(cb_second_a, cb_second_b, cb_second_product);", compute)
        self.assertIn("cb_wait_front(19, 1);", writer)
        self.assertIn("get_read_ptr(19)", writer)
        self.assertLess(
            compute.index("matmul_block_init(cb_second_a, cb_second_b"),
            compute.index("pack_reconfig_data_format(cb_product, cb_second_product)"),
        )
        inputs = bringup._inputs(stage)
        np.testing.assert_allclose(
            bringup.expected_output(stage, inputs),
            np.matmul(np.matmul(inputs[0], inputs[1]), inputs[2]),
        )

    def test_newton_inputs_are_deterministic_hpd_with_requested_condition(self):
        for number in (5, 6, 46, 47, 48):
            matrices = bringup._inputs(bringup.STAGES[number])
            values = matrices[0] + 1j * matrices[2]
            np.testing.assert_allclose(values, values.conj().swapaxes(-1, -2), atol=2e-5)
            for matrix in values:
                eigenvalues = np.linalg.eigvalsh(matrix)
                self.assertGreater(float(eigenvalues[0]), 0.0)
                self.assertAlmostEqual(float(eigenvalues[-1] / eigenvalues[0]), 100.0, delta=0.02)

    def test_stage_validation_and_defaults_are_host_only(self):
        self.assertEqual(bringup.stage_for(1).number, 1)
        with self.assertRaisesRegex(ValueError, "stage must be one of"):
            bringup.stage_for(0)
        parser = bringup.argparse.ArgumentParser()
        parser.add_argument("--stage", type=int, required=True)
        parser.add_argument("--device-id", type=int, default=0)
        self.assertEqual(parser.parse_args(["--stage", "1"]).device_id, 0)

    def test_deterministic_inputs_and_expectations(self):
        for stage in bringup.STAGES.values():
            first = bringup._inputs(stage)
            second = bringup._inputs(stage)
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left, right)
            expected = bringup.expected_output(stage, first)
            self.assertEqual(expected.shape, (stage.batch, bringup.TILE, bringup.TILE))
            self.assertTrue(np.isfinite(expected).all())

    def test_absolute_sources_are_present_and_later_stages_are_isolated(self):
        paths = [bringup.source_paths(stage) for stage in bringup.STAGES.values()]
        for compute, reader, writer in paths:
            self.assertTrue(compute.is_absolute())
            self.assertTrue(reader.is_absolute())
            self.assertTrue(writer.is_absolute())
            self.assertTrue(compute.is_file())
            self.assertTrue(reader.is_file())
            self.assertTrue(writer.is_file())
        self.assertEqual(paths[0][0], paths[1][0])
        self.assertEqual(paths[1][0], paths[2][0])
        self.assertNotEqual(paths[2][0], paths[3][0])
        self.assertNotEqual(paths[3][0], paths[4][0])
        self.assertNotEqual(paths[4][0], paths[5][0])
        self.assertNotEqual(paths[0][2], paths[3][2])
        self.assertNotEqual(paths[0][1], paths[3][1])

    def test_compute_sources_use_block_matmul_and_stage_specific_additions(self):
        sources = {stage.number: stage.compute_source for stage in bringup.STAGES.values()}
        real = (bringup.KERNEL_DIR / sources[1]).read_text()
        complex_source = (bringup.KERNEL_DIR / sources[4]).read_text()
        one = (bringup.KERNEL_DIR / sources[5]).read_text()
        eight = (bringup.KERNEL_DIR / sources[6]).read_text()
        self.assertIn("mm_block_init", real)
        self.assertIn("matmul_block", real)
        self.assertNotIn("add_tiles", real)
        self.assertNotIn("sub_tiles", real)
        self.assertNotIn("binary_op", real)
        self.assertEqual(complex_source.count("matmul_one();"), 4)
        self.assertNotIn("copy_tile_init", complex_source)
        self.assertNotIn("route_product", complex_source)
        self.assertIn("add_tiles", complex_source)
        self.assertIn("sub_tiles", complex_source)
        self.assertIn("complex_matmul_products", eight)
        self.assertNotIn("stream_operands", one)
        self.assertNotIn("cb_pop_front(cb_x_real", eight)
        self.assertIn("iteration < 8", eight)
        self.assertIn("matmul_block_init", one)
        for stage_number in (4, 5, 6):
            source = (bringup.KERNEL_DIR / sources[stage_number]).read_text()
            operand_names = (
                "cb_operand_a, cb_operand_b" if stage_number == 4 else "cb_matmul_a, cb_matmul_b"
            )
            if stage_number == 4:
                self.assertIn(f"mm_block_init({operand_names}", source)
                self.assertEqual(source.count("mm_block_init("), 1)
                self.assertNotIn("compute_kernel_hw_startup", source)
                self.assertEqual(source.count("matmul_block_init("), 0)
            else:
                self.assertNotIn("mm_block_init", source)
                self.assertIn('#include "api/compute/compute_kernel_hw_startup.h"', source)
                self.assertEqual(
                    source.count(
                        "compute_kernel_hw_startup<SrcOrder::Reverse>("
                        "cb_matmul_a, cb_matmul_b, cb_product)"
                    ),
                    1,
                )
                self.assertIn(
                    "compute_kernel_hw_startup<SrcOrder::Reverse>("
                    "cb_matmul_a, cb_matmul_b, cb_product);\n"
                    "    matmul_block_init(cb_matmul_a, cb_matmul_b, false, 1, 1, 1);",
                    source,
                )
                self.assertEqual(source.count("matmul_block_init("), 2)
            self.assertNotIn("matmul_block(left, right", source)
            self.assertNotIn("matmul_block(cb_a", source)
            self.assertEqual(source.count(f"matmul_block({operand_names}"), 1)
            self.assertNotIn("route_product", source)
            self.assertNotIn("copy_tile_init", source)
            self.assertNotIn("duplicate_tile", source)
            self.assertNotIn("stream_operands", source)
            if stage_number == 6:
                self.assertIn("restore_matmul", source)
            else:
                self.assertNotIn("restore_matmul", source)

    def test_diagnostic_stages_have_isolated_modern_sources_and_expected_outputs(self):
        paths = [bringup.source_paths(bringup.STAGES[number]) for number in (41, 42, 43)]
        self.assertEqual(len({path for paths_for_stage in paths for path in paths_for_stage}), 9)
        for number, expected_kind in ((41, "complex_modern_startup"), (42, "complex_two_groups")):
            stage = bringup.STAGES[number]
            self.assertEqual(stage.kind, expected_kind)
            source = (bringup.KERNEL_DIR / stage.compute_source).read_text()
            self.assertIn('#include "api/compute/compute_kernel_hw_startup.h"', source)
            startup = (
                "compute_kernel_hw_startup<SrcOrder::Reverse>(cb_matmul_a, cb_matmul_b, cb_product)"
            )
            self.assertEqual(source.count(startup), 1)
            self.assertNotIn("mm_block_init", source)
            self.assertEqual(source.count("matmul_block_init("), 1 if number == 41 else 2)
            self.assertEqual(source.count("matmul_block(cb_matmul_a, cb_matmul_b"), 1)
            if number == 41:
                self.assertIn("subtract_one<cb_product_rr, cb_product_ii, cb_out_real>", source)
                self.assertIn("add_one<cb_product_ri, cb_product_ir, cb_out_imag>", source)
            else:
                self.assertIn("cb_private_real = 8", source)
                self.assertIn("cb_private_imag = 9", source)
                self.assertIn("cb_out_real = 23", source)
                self.assertIn("cb_out_imag = 24", source)
                self.assertIn("matmul_group();", source)
                self.assertIn("output_group();", source)
            for forbidden in ("copy_tile_init", "copy_tile", "duplicate_tile", "stream_operands"):
                self.assertNotIn(forbidden, source)
            reader = (bringup.KERNEL_DIR / stage.reader_source).read_text()
            self.assertEqual(reader.count("route_complex_products();"), 1 if number == 41 else 2)
            for destination in ("cb_product_rr", "cb_product_ii", "cb_product_ri", "cb_product_ir"):
                self.assertEqual(reader.count(f"route_product({destination})"), 1)
            inputs = bringup._inputs(stage)
            expected = bringup.expected_output(stage, inputs)
            left = inputs[0] + 1j * inputs[2]
            right = inputs[1] + 1j * inputs[3]
            np.testing.assert_allclose(expected, np.matmul(left, right))

        stage = bringup.STAGES[43]
        source = (bringup.KERNEL_DIR / stage.compute_source).read_text()
        self.assertEqual(source.count("matmul_block_init("), 1)
        self.assertNotIn("mm_block_init", source)
        self.assertNotIn("copy_tile", source)
        reader = (bringup.KERNEL_DIR / stage.reader_source).read_text()
        self.assertIn("read_tile(cb_identity, 0, identity)", reader)
        self.assertIn("read_tile(cb_zero, 0, zero)", reader)
        inputs = bringup._inputs(stage)
        residual = inputs[4].astype(np.complex64) - np.matmul(
            inputs[0] + 1j * inputs[2], inputs[1] + 1j * inputs[3]
        )
        np.testing.assert_allclose(bringup.expected_output(stage, inputs), residual)
        self.assertEqual(
            [bringup.output_count(bringup.STAGES[number]) for number in (41, 42, 43)], [2, 2, 2]
        )

    def test_residual_copy_and_compute_copy_boundaries(self):
        diagnostic_numbers = (44, 45)
        paths = [bringup.source_paths(bringup.STAGES[number]) for number in diagnostic_numbers]
        self.assertEqual(len({path for paths_for_stage in paths for path in paths_for_stage}), 6)

        stage = bringup.STAGES[44]
        compute = (bringup.KERNEL_DIR / stage.compute_source).read_text()
        reader = (bringup.KERNEL_DIR / stage.reader_source).read_text()
        self.assertEqual(compute.count("compute_kernel_hw_startup<SrcOrder::Reverse>("), 1)
        self.assertEqual(compute.count("matmul_block_init("), 1)
        self.assertNotIn("mm_block_init", compute)
        self.assertNotIn("copy_tile", compute)
        self.assertIn("cb_out_real = 19", compute)
        self.assertIn("cb_out_imag = 20", compute)
        for argument in range(8):
            self.assertIn(f"get_arg_val<std::uint32_t>({argument})", reader)
        self.assertIn(
            "zero_args = TensorAccessorArgs<identity_args.next_compile_time_args_offset()>", reader
        )
        self.assertEqual(reader.count("copy_output("), 3)
        self.assertIn("stream_external_complex(cb_matmul_a, tile, r_real, r_imag, true)", reader)
        self.assertIn("stream_external_complex(cb_matmul_b, tile, x_real, x_imag, false)", reader)
        self.assertEqual(reader.count("route_complex_products();"), 1)
        self.assertIn("read_tile(cb_identity, 0, identity)", reader)
        self.assertIn("read_tile(cb_zero, 0, zero)", reader)
        self.assertIn("copy_output(cb_source_real, cb_out_real)", reader)
        self.assertIn("copy_output(cb_source_imag, cb_out_imag)", reader)
        input_route = reader.index("route_complex_products();")
        identity = reader.index("read_tile(cb_identity, 0, identity)")
        output_copy = reader.index("copy_output(cb_source_real, cb_out_real)")
        self.assertLess(input_route, identity)
        self.assertLess(identity, output_copy)
        self.assertIn("cb_pop_front(source, 1)", reader)
        inputs = bringup._inputs(stage)
        np.testing.assert_allclose(
            bringup.expected_output(stage, inputs),
            inputs[4].astype(np.complex64)
            - np.matmul(inputs[0] + 1j * inputs[2], inputs[1] + 1j * inputs[3]),
        )

        stage = bringup.STAGES[45]
        compute = (bringup.KERNEL_DIR / stage.compute_source).read_text()
        reader = (bringup.KERNEL_DIR / stage.reader_source).read_text()
        self.assertEqual(compute.count("compute_kernel_hw_startup<SrcOrder::Reverse>("), 1)
        self.assertEqual(compute.count("matmul_block_init("), 2)
        self.assertNotIn("mm_block_init", compute)
        self.assertIn("cb_x_real = 17", compute)
        self.assertIn("cb_x_imag = 18", compute)
        self.assertEqual(compute.count("copy_source_to_operand(cb_x_real, cb_matmul_a, true)"), 2)
        self.assertEqual(compute.count("copy_source_to_operand(cb_x_imag, cb_matmul_a, true)"), 2)
        self.assertEqual(reader.count("read_tile(cb_x_real, tile, real)"), 2)
        self.assertEqual(reader.count("read_tile(cb_x_imag, tile, imag)"), 2)
        self.assertIn("cb_pop_front(cb_s_real, 1)", compute)
        self.assertIn("cb_pop_front(cb_s_imag, 1)", compute)
        self.assertIn("cb_pop_front(source, 1)", compute)
        copy_start = compute.index("void stage_second_operands()")
        reinit = compute.index("matmul_block_init(", compute.index("stage_second_operands();"))
        second_group = compute.index("complex_matmul();", reinit)
        self.assertLess(compute.index("copy_tile_init"), reinit)
        self.assertIn("copy_source_to_operand", compute[copy_start:reinit])
        self.assertNotIn("copy_tile", compute[reinit:second_group])
        self.assertIn("stream_source_x", reader)
        self.assertIn("cb_x_real", reader)
        self.assertIn("cb_x_imag", reader)
        self.assertEqual(reader.count("route_complex_products();"), 2)
        self.assertEqual(reader.count("route_product(cb_product_rr)"), 1)
        self.assertEqual(reader.count("route_product(cb_product_ii)"), 1)
        self.assertEqual(reader.count("route_product(cb_product_ri)"), 1)
        self.assertEqual(reader.count("route_product(cb_product_ir)"), 1)
        first_route = reader.index("route_complex_products();")
        source_x = reader.index("stream_source_x(tile, x_real, x_imag);")
        second_route = reader.index("route_complex_products();", source_x)
        self.assertLess(first_route, source_x)
        self.assertGreater(second_route, source_x)
        inputs = bringup._inputs(stage)
        expected = bringup.expected_output(stage, inputs)
        r = inputs[0] + 1j * inputs[2]
        x = inputs[1] + 1j * inputs[3]
        s = inputs[4].astype(np.complex64) - np.matmul(r, x)
        np.testing.assert_allclose(expected, np.matmul(x, s))

    def test_corrected_newton_diagnostics_consume_all_products(self):
        diagnostic_numbers = (46, 47, 48)
        paths = [bringup.source_paths(bringup.STAGES[number]) for number in diagnostic_numbers]
        self.assertEqual(len({path for paths_for_stage in paths for path in paths_for_stage}), 9)

        for number in (46, 47):
            stage = bringup.STAGES[number]
            compute = (bringup.KERNEL_DIR / stage.compute_source).read_text()
            reader = (bringup.KERNEL_DIR / stage.reader_source).read_text()
            self.assertEqual(compute.count("compute_kernel_hw_startup<SrcOrder::Reverse>("), 1)
            self.assertEqual(compute.count("matmul_block_init("), 1)
            self.assertNotIn("mm_block_init", compute)
            for forbidden in ("copy_tile_init", "copy_tile", "duplicate_tile", "stream_operands"):
                self.assertNotIn(forbidden, compute)
            self.assertIn("subtract_one<cb_product_rr, cb_product_ii, cb_rx_real>", compute)
            self.assertIn("add_one<cb_product_ri, cb_product_ir, cb_rx_imag>", compute)
            self.assertIn("subtract_one<cb_identity, cb_rx_real", compute)
            self.assertIn("subtract_one<cb_zero, cb_rx_imag", compute)
            self.assertEqual(reader.count("route_complex_products();"), 1)
            for destination in ("cb_product_rr", "cb_product_ii", "cb_product_ri", "cb_product_ir"):
                self.assertEqual(reader.count(f"route_product({destination})"), 1)
            for argument in range(8):
                self.assertIn(f"get_arg_val<std::uint32_t>({argument})", reader)
            if number == 47:
                self.assertIn("cb_rx_real = 8", compute)
                self.assertIn("cb_rx_imag = 9", compute)
                self.assertIn("cb_out_real = 19", compute)
                self.assertIn("cb_out_imag = 20", compute)
                self.assertEqual(reader.count("copy_output("), 3)
                self.assertIn("copy_output(cb_source_real, cb_out_real)", reader)
                self.assertIn("copy_output(cb_source_imag, cb_out_imag)", reader)
            inputs = bringup._inputs(stage)
            residual = inputs[4].astype(np.complex64) - np.matmul(
                inputs[0] + 1j * inputs[2], inputs[1] + 1j * inputs[3]
            )
            np.testing.assert_allclose(bringup.expected_output(stage, inputs), residual)

        stage = bringup.STAGES[48]
        compute = (bringup.KERNEL_DIR / stage.compute_source).read_text()
        reader = (bringup.KERNEL_DIR / stage.reader_source).read_text()
        self.assertEqual(compute.count("compute_kernel_hw_startup<SrcOrder::Reverse>("), 1)
        self.assertEqual(compute.count("matmul_block_init("), 2)
        self.assertNotIn("mm_block_init", compute)
        for forbidden in ("copy_tile_init", "copy_tile", "duplicate_tile", "stream_operands"):
            self.assertNotIn(forbidden, compute)
        self.assertIn("cb_rx_real = 8", compute)
        self.assertIn("cb_rx_imag = 9", compute)
        self.assertIn("cb_s_real = 19", compute)
        self.assertIn("cb_s_imag = 20", compute)
        first_group = (
            "subtract_one<cb_product_rr, cb_product_ii, cb_rx_real>();\n"
            "        add_one<cb_product_ri, cb_product_ir, cb_rx_imag>();\n"
            "        subtract_one<cb_identity, cb_rx_real, cb_s_real>();\n"
            "        subtract_one<cb_zero, cb_rx_imag, cb_s_imag>();"
        )
        self.assertIn(first_group, compute)
        self.assertIn("subtract_one<cb_product_rr, cb_product_ii, cb_out_real>", compute)
        self.assertIn("add_one<cb_product_ri, cb_product_ir, cb_out_imag>", compute)
        for argument in range(8):
            self.assertIn(f"get_arg_val<std::uint32_t>({argument})", reader)
        self.assertEqual(reader.count("route_complex_products();"), 2)
        first_route = reader.index("route_complex_products();")
        second_x = reader.index("stream_external_complex(cb_matmul_a, tile, x_real, x_imag, true)")
        second_route = reader.index("route_complex_products();", second_x)
        self.assertLess(first_route, second_x)
        self.assertGreater(second_route, second_x)
        self.assertIn("stream_cb_complex(cb_matmul_b, cb_s_real, cb_s_imag, false, true)", reader)
        first_s = compute.index("subtract_one<cb_product_rr, cb_product_ii, cb_rx_real>")
        reinit = compute.index("matmul_block_init(", first_s)
        second_group = compute.index("complex_matmul();", reinit)
        self.assertIn("add_one<cb_product_ri, cb_product_ir, cb_rx_imag>", compute[first_s:reinit])
        self.assertIn("subtract_one<cb_identity, cb_rx_real, cb_s_real>", compute[first_s:reinit])
        self.assertIn("subtract_one<cb_zero, cb_rx_imag, cb_s_imag>", compute[first_s:reinit])
        self.assertNotIn("copy_tile", compute[reinit:second_group])
        inputs = bringup._inputs(stage)
        r = inputs[0] + 1j * inputs[2]
        x = inputs[1] + 1j * inputs[3]
        s = inputs[4].astype(np.complex64) - np.matmul(r, x)
        np.testing.assert_allclose(bringup.expected_output(stage, inputs), np.matmul(x, s))

    def test_production_newton_stages_use_complete_residual_construction(self):
        for number in (5, 6):
            compute = (bringup.KERNEL_DIR / bringup.STAGES[number].compute_source).read_text()
            reader = (bringup.KERNEL_DIR / bringup.STAGES[number].reader_source).read_text()
            self.assertIn("cb_rx_real = 8", compute)
            self.assertIn("cb_rx_imag = 9", compute)
            self.assertIn("cb_s_real = 19", compute)
            self.assertIn("cb_s_imag = 20", compute)
            indent = "        " if number == 5 else "            "
            first_group = (
                f"{indent}subtract_one<cb_product_rr, cb_product_ii, cb_rx_real>();\n"
                f"{indent}add_one<cb_product_ri, cb_product_ir, cb_rx_imag>();\n"
                f"{indent}subtract_one<cb_identity, cb_rx_real, cb_s_real>();\n"
                f"{indent}subtract_one<cb_zero, cb_rx_imag, cb_s_imag>();"
            )
            self.assertIn(first_group, compute)
            self.assertEqual(reader.count("route_complex_products();"), 2)
            self.assertEqual(reader.count("read_tile(cb_identity, 0, identity)"), 1)
            self.assertEqual(reader.count("read_tile(cb_zero, 0, zero)"), 1)
            for forbidden in ("copy_tile_init", "copy_tile", "duplicate_tile", "stream_operands"):
                self.assertNotIn(forbidden, compute)

        one_compute = (bringup.KERNEL_DIR / bringup.STAGES[5].compute_source).read_text()
        one_reader = (bringup.KERNEL_DIR / bringup.STAGES[5].reader_source).read_text()
        self.assertIn(
            "stream_cb_complex(cb_matmul_b, cb_s_real, cb_s_imag, false, true)", one_reader
        )
        self.assertIn("subtract_one<cb_product_rr, cb_product_ii, cb_out_real>", one_compute)
        self.assertIn("add_one<cb_product_ri, cb_product_ir, cb_out_imag>", one_compute)

        eight_compute = (bringup.KERNEL_DIR / bringup.STAGES[6].compute_source).read_text()
        eight_reader = (bringup.KERNEL_DIR / bringup.STAGES[6].reader_source).read_text()
        self.assertIn(
            "for (std::uint32_t iteration = 0; iteration < 8; ++iteration)", eight_compute
        )
        self.assertEqual(eight_compute.count("restore_matmul();"), 2)
        self.assertIn("if (iteration + 1 == 8)", eight_compute)
        self.assertEqual(eight_reader.count("route_complex_products();"), 2)
        self.assertIn(
            "stream_cb_complex(cb_matmul_b, cb_state_real, cb_state_imag, false, false)",
            eight_reader,
        )
        self.assertIn(
            "stream_cb_complex(cb_matmul_a, cb_state_real, cb_state_imag, true, true)",
            eight_reader,
        )

    def test_real_stages_have_one_device_output_and_host_zero_imaginary_part(self):
        self.assertEqual(sum(stage.kind == "real" for stage in bringup.STAGES.values()), 3)
        self.assertEqual(
            [bringup.output_count(bringup.STAGES[number]) for number in (1, 2, 3)], [1, 1, 1]
        )
        self.assertEqual(
            [bringup.output_count(bringup.STAGES[number]) for number in (4, 5, 6)], [2, 2, 2]
        )
        complex_reader = (bringup.KERNEL_DIR / "bringup_complex_reader.cpp").read_text()
        ns_reader = (bringup.KERNEL_DIR / "bringup_ns_eight_reader.cpp").read_text()
        for destination in ("cb_product_rr", "cb_product_ii", "cb_product_ri", "cb_product_ir"):
            self.assertEqual(complex_reader.count(f"route_product({destination})"), 1)
            self.assertEqual(ns_reader.count(f"route_product({destination})"), 1)
        self.assertEqual(complex_reader.count("route_complex_products();"), 1)
        self.assertEqual(ns_reader.count("route_complex_products();"), 2)
        ns_one_reader = (bringup.KERNEL_DIR / "bringup_ns_one_reader.cpp").read_text()
        for source in (ns_one_reader, ns_reader):
            self.assertIn("stream_external_complex", source)
            self.assertIn("stream_cb_complex", source)
            self.assertIn("route_complex_products", source)
        self.assertIn(
            "stream_cb_complex(cb_matmul_b, cb_state_real, cb_state_imag, false, false)",
            ns_reader,
        )
        self.assertIn(
            "stream_cb_complex(cb_matmul_a, cb_state_real, cb_state_imag, true, true)",
            ns_reader,
        )
        self.assertIn(
            "stream_cb_complex(cb_matmul_b, cb_s_real, cb_s_imag, false, true)",
            ns_reader,
        )
        real_reader = (bringup.KERNEL_DIR / "bringup_real_reader.cpp").read_text()
        real_writer = (bringup.KERNEL_DIR / "bringup_real_writer.cpp").read_text()
        self.assertIn("get_arg_val<std::uint32_t>(2)", real_reader)
        self.assertEqual(real_writer.count("noc_async_write_page"), 1)
        self.assertIn("np.zeros_like(real)", Path("tools/newton_schulz_bringup.py").read_text())
        self.assertEqual(bringup.NUMERICAL_TOLERANCE, 1e-2)

    def test_no_accelerator_module_imports_the_reference(self):
        accelerator_root = Path(__file__).parents[1] / "enodia" / "tt"
        forbidden = "enodia.tt.bench.newton_schulz_reference"
        violations = []
        for path in accelerator_root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == forbidden:
                    violations.append(str(path))
                if isinstance(node, ast.Import):
                    violations.extend(str(path) for alias in node.names if alias.name == forbidden)
        self.assertEqual(violations, [])

    def test_final_record_is_json_serializable_and_machine_readable(self):
        stage = bringup.STAGES[1]
        record = {
            "stage": stage.number,
            "status": "pass",
            "batch": stage.batch,
            "cores": stage.cores,
            "tile_shape": [bringup.TILE, bringup.TILE],
            "elapsed_s": 0.0,
            "numerical_error": 0.0,
        }
        parsed = json.loads(json.dumps(record))
        self.assertEqual(parsed["stage"], 1)
        self.assertEqual(parsed["status"], "pass")


if __name__ == "__main__":
    unittest.main()
