import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from enodia.tt.bench import newton_schulz_kernel, run_matmul

KERNEL_DIR = Path(__file__).parents[1] / "enodia/tt/bench/kernels"
HISTORICAL_MATRIX_BLOCK_RECORD = (
    Path(__file__).parents[1]
    / "docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000.json"
)


def _ttnn():
    return SimpleNamespace(bfloat16="bf16", float32="fp32")


def test_dest_slot_limit_is_controlled_by_fp32_accumulation_and_sync_mode():
    assert newton_schulz_kernel._dest_slot_limit(
        fp32_dest_acc_en=True, dst_full_sync_en=True
    ) == 8
    assert newton_schulz_kernel._dest_slot_limit(
        fp32_dest_acc_en=False, dst_full_sync_en=True
    ) == 16
    assert newton_schulz_kernel._dest_slot_limit(
        fp32_dest_acc_en=True, dst_full_sync_en=False
    ) == 4


def test_issue94_all_l1_inputs_and_dram_outputs_fit_every_required_row():
    ttnn = _ttnn()
    common = {
        "core_count": 110,
        "fuse_s": True,
        "input_memory": "l1",
        "r_memory": "l1",
        "x0_memory": "l1",
        "output_memory": "dram",
        "matrix_block": 4,
        "double_buffer": False,
    }
    rows = (
        ("bf16", True, "bf16", 1_008_384),
        ("bf16", False, "bf16", 1_008_384),
        ("bf16-fp32state", True, "fp32", 1_393_408),
    )
    for variant, fp32_dest_acc_en, state_dtype, expected in rows:
        total = newton_schulz_kernel._validate_l1_preflight(
            ttnn,
            batch=8192,
            state_dtype=state_dtype,
            variant=variant,
            fp32_dest_acc_en=fp32_dest_acc_en,
            **common,
        )
        assert total == expected
        assert total <= newton_schulz_kernel._L1_TOTAL_BUDGET_BYTES


@pytest.mark.parametrize("matrix_block", [1, 2, 4, 8])
def test_supported_matrix_blocks_validate_and_scale_matrix_queues(matrix_block):
    newton_schulz_kernel._validate_matrix_block(matrix_block)
    definitions = newton_schulz_kernel._cb_definitions(
        _ttnn(), "fp32", fuse_s=True, matrix_block=matrix_block
    )

    expected_queue_pages = 2 if matrix_block == 1 else matrix_block
    matrix_queue_indices = (
        newton_schulz_kernel.CB_R_NEG_IMAG,
        newton_schulz_kernel.CB_R_IMAG,
        newton_schulz_kernel.CB_X0_REAL,
        newton_schulz_kernel.CB_X0_IMAG,
        newton_schulz_kernel.CB_S_REAL,
        newton_schulz_kernel.CB_S_IMAG,
        newton_schulz_kernel.CB_NEG_X_IMAG,
        newton_schulz_kernel.CB_R_NEG_REAL,
        newton_schulz_kernel.CB_OUTPUT_REAL,
        newton_schulz_kernel.CB_OUTPUT_IMAG,
    )
    assert newton_schulz_kernel.CB_R_REAL not in definitions
    if matrix_block == 1:
        assert definitions[newton_schulz_kernel.CB_OUTPUT_REAL][1] == expected_queue_pages
    else:
        assert all(definitions[index][1] == expected_queue_pages for index in matrix_queue_indices)
    expected_state_pages = {1: 2, 2: 2, 4: 4, 8: 16}
    assert definitions[newton_schulz_kernel.CB_STATE_REAL][1] == expected_state_pages[matrix_block]
    assert definitions[newton_schulz_kernel.CB_STATE_IMAG][1] == expected_state_pages[matrix_block]
    # Fused S never routes products; keep their descriptors to one page for
    # compile-time CB identity without reserving unused block pages.
    assert definitions[newton_schulz_kernel.CB_PRODUCT_REAL][1] == 1
    assert definitions[newton_schulz_kernel.CB_PRODUCT_IMAG][1] == 1
    # Constants remain resident singletons rather than consuming block slots.
    assert definitions[newton_schulz_kernel.CB_IDENTITY][1] == 1
    assert definitions[newton_schulz_kernel.CB_ZERO][1] == 1


def test_fused_reader_input_values_omit_positive_r_real_and_keep_signed_order():
    matrices = np.zeros((1, 32, 32), dtype=np.complex64)
    matrices[0].real.fill(3.0)
    matrices[0].imag.fill(5.0)
    x0 = np.zeros_like(matrices)
    fused = newton_schulz_kernel._reader_input_values(
        matrices, x0, fuse_s=True, tile_count=1
    )
    baseline = newton_schulz_kernel._reader_input_values(
        matrices, x0, fuse_s=False, tile_count=1
    )

    assert len(fused) == len(baseline) == 5
    assert newton_schulz_kernel._reader_input_dtypes(
        _ttnn(), "fp32", fuse_s=True
    ) == ["bf16", "bf16", "bf16", "fp32", "fp32", "bf16", "fp32"]
    assert newton_schulz_kernel._reader_input_dtypes(
        _ttnn(), "fp32", fuse_s=False
    ) == ["bf16", "bf16", "bf16", "fp32", "fp32", "fp32", "fp32"]
    np.testing.assert_array_equal(fused[0], -baseline[2])
    np.testing.assert_array_equal(fused[1], baseline[2])
    np.testing.assert_array_equal(fused[2], baseline[0] * -1)
    np.testing.assert_array_equal(baseline[0], matrices.real[None, ...])
    np.testing.assert_array_equal(fused[3:], baseline[3:])


def test_nonfused_cb_and_tensor_ledgers_keep_positive_r_real():
    ttnn = _ttnn()
    fused = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=8
    )
    baseline = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=False, matrix_block=8
    )
    assert newton_schulz_kernel.CB_R_REAL not in fused
    assert baseline[newton_schulz_kernel.CB_R_REAL] == ("bf16", 8)
    assert baseline[newton_schulz_kernel.CB_R_NEG_REAL] == ("bf16", 8)
    fused_tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
    )
    baseline_tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=False,
        output_memory="dram",
    )
    assert baseline_tensor_bytes == fused_tensor_bytes + 2_048


def test_matrix_block_default_is_baseline_and_invalid_values_fail_host_side():
    assert newton_schulz_kernel.MATRIX_BLOCK_CHOICES == (1, 2, 4, 8)
    baseline = newton_schulz_kernel._cb_definitions(_ttnn(), "fp32", fuse_s=True)
    explicit_baseline = newton_schulz_kernel._cb_definitions(
        _ttnn(), "fp32", fuse_s=True, matrix_block=1
    )
    assert baseline == explicit_baseline

    for invalid in (0, 3, 5):
        with pytest.raises(ValueError, match="matrix_block"):
            newton_schulz_kernel._validate_matrix_block(invalid)


def test_double_buffer_doubles_only_external_block_windows_and_preserves_defaults():
    ttnn = _ttnn()
    external = (
        newton_schulz_kernel.CB_R_NEG_IMAG,
        newton_schulz_kernel.CB_R_IMAG,
        newton_schulz_kernel.CB_X0_REAL,
        newton_schulz_kernel.CB_X0_IMAG,
        newton_schulz_kernel.CB_R_NEG_REAL,
        newton_schulz_kernel.CB_OUTPUT_REAL,
        newton_schulz_kernel.CB_OUTPUT_IMAG,
    )
    internal = (
        newton_schulz_kernel.CB_STATE_REAL,
        newton_schulz_kernel.CB_STATE_IMAG,
        newton_schulz_kernel.CB_S_REAL,
        newton_schulz_kernel.CB_S_IMAG,
        newton_schulz_kernel.CB_NEG_X_IMAG,
    )
    expected_single = {1: 2, 2: 2, 4: 4, 8: 8}

    for matrix_block in newton_schulz_kernel.MATRIX_BLOCK_CHOICES:
        baseline = newton_schulz_kernel._cb_definitions(
            ttnn, "fp32", fuse_s=True, matrix_block=matrix_block
        )
        explicit_default = newton_schulz_kernel._cb_definitions(
            ttnn,
            "fp32",
            fuse_s=True,
            matrix_block=matrix_block,
            double_buffer=False,
        )
        selected = newton_schulz_kernel._cb_definitions(
            ttnn,
            "fp32",
            fuse_s=True,
            matrix_block=matrix_block,
            double_buffer=True,
        )
        assert explicit_default == baseline
        expected_double = expected_single[matrix_block] * (2 if matrix_block > 1 else 1)
        assert all(selected[index][1] == expected_double for index in external)
        assert all(selected[index][1] == baseline[index][1] for index in internal)

    # The Issue #92 target is an eight-page window for a four-page push.
    selected = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=4, double_buffer=True
    )
    assert all(selected[index][1] == 8 for index in external)
    work_ranges = newton_schulz_kernel._balanced_ranges(8192, 110, 4)
    newton_schulz_kernel._validate_core_group_capacities(work_ranges, 4)
    assert all(8 % group_count == 0 for start, count in work_ranges for _, group_count in
               newton_schulz_kernel._matrix_block_ranges(start, count, 4))


@pytest.mark.parametrize(
    ("fuse_s", "cb_index"),
    [
        pytest.param(True, newton_schulz_kernel.CB_R_NEG_IMAG, id="fused-r-neg-imag"),
        pytest.param(True, newton_schulz_kernel.CB_R_IMAG, id="fused-r-imag"),
        pytest.param(True, newton_schulz_kernel.CB_R_NEG_REAL, id="fused-r-neg-real"),
        pytest.param(True, newton_schulz_kernel.CB_X0_REAL, id="fused-x0-real"),
        pytest.param(True, newton_schulz_kernel.CB_X0_IMAG, id="fused-x0-imag"),
        pytest.param(True, newton_schulz_kernel.CB_OUTPUT_REAL, id="fused-output-real"),
        pytest.param(True, newton_schulz_kernel.CB_OUTPUT_IMAG, id="fused-output-imag"),
        pytest.param(False, newton_schulz_kernel.CB_R_REAL, id="nonfused-r-real"),
        pytest.param(False, newton_schulz_kernel.CB_R_NEG_IMAG, id="nonfused-r-neg-imag"),
        pytest.param(False, newton_schulz_kernel.CB_R_IMAG, id="nonfused-r-imag"),
        pytest.param(False, newton_schulz_kernel.CB_R_NEG_REAL, id="nonfused-r-neg-real"),
        pytest.param(False, newton_schulz_kernel.CB_X0_REAL, id="nonfused-x0-real"),
        pytest.param(False, newton_schulz_kernel.CB_X0_IMAG, id="nonfused-x0-imag"),
        pytest.param(False, newton_schulz_kernel.CB_OUTPUT_REAL, id="nonfused-output-real"),
        pytest.param(False, newton_schulz_kernel.CB_OUTPUT_IMAG, id="nonfused-output-imag"),
    ],
)
def test_matrix_block_four_double_buffer_external_cb_pages(fuse_s, cb_index):
    ttnn = _ttnn()
    single_window = newton_schulz_kernel._cb_definitions(
        ttnn,
        "fp32",
        fuse_s=fuse_s,
        matrix_block=4,
        double_buffer=False,
    )
    double_window = newton_schulz_kernel._cb_definitions(
        ttnn,
        "fp32",
        fuse_s=fuse_s,
        matrix_block=4,
        double_buffer=True,
    )

    assert single_window[cb_index][1] == 4
    assert double_window[cb_index][1] == 8


@pytest.mark.parametrize("fuse_s", [True, False], ids=["fused", "nonfused"])
@pytest.mark.parametrize(
    "cb_index",
    [
        pytest.param(newton_schulz_kernel.CB_STATE_REAL, id="state-real"),
        pytest.param(newton_schulz_kernel.CB_STATE_IMAG, id="state-imag"),
        pytest.param(newton_schulz_kernel.CB_S_REAL, id="s-real"),
        pytest.param(newton_schulz_kernel.CB_S_IMAG, id="s-imag"),
        pytest.param(newton_schulz_kernel.CB_PRODUCT_REAL, id="product-real"),
        pytest.param(newton_schulz_kernel.CB_PRODUCT_IMAG, id="product-imag"),
        pytest.param(newton_schulz_kernel.CB_NEG_X_IMAG, id="neg-x-imag"),
    ],
)
def test_matrix_block_four_double_buffer_preserves_internal_cb_pages(fuse_s, cb_index):
    ttnn = _ttnn()
    single_window = newton_schulz_kernel._cb_definitions(
        ttnn,
        "fp32",
        fuse_s=fuse_s,
        matrix_block=4,
        double_buffer=False,
    )
    double_window = newton_schulz_kernel._cb_definitions(
        ttnn,
        "fp32",
        fuse_s=fuse_s,
        matrix_block=4,
        double_buffer=True,
    )

    assert double_window[cb_index][1] == single_window[cb_index][1]


def test_double_buffer_l1_preflight_fits_l32_and_l16_including_profile_pages():
    ttnn = _ttnn()
    ttnn.uint32 = "u32"
    expected = {
        (8192, False): 1_483_520,
        (8192, True): 1_495_808,
        (4096, False): 967_424,
        (4096, True): 979_712,
    }
    for (batch, profile), total_expected in expected.items():
        total = newton_schulz_kernel._validate_l1_preflight(
            ttnn,
            batch=batch,
            core_count=110,
            state_dtype="fp32",
            profile=profile,
            fuse_s=True,
            output_memory="dram",
            matrix_block=4,
            double_buffer=True,
            variant="bf16-fp32state",
        )
        assert total == total_expected
        assert total <= newton_schulz_kernel._L1_TOTAL_BUDGET_BYTES


def test_dest_limit_uses_fp32_and_sync_mode_not_a_soft_block_cap():
    for matrix_block in (1, 2, 4):
        newton_schulz_kernel._validate_matrix_block(
            matrix_block,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
            variant="bf16-fp32state",
        )
    # Block 8 uses one DEST half at a time and reaches the eight-slot FP32
    # limit, while the existing blocks retain two DEST slots per matrix.
    newton_schulz_kernel._validate_matrix_block(
        8, fp32_dest_acc_en=True, dst_full_sync_en=True, variant="bf16-fp32state"
    )
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_matrix_block(
            4, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_matrix_block(
            8, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
    with pytest.raises(ValueError, match="requires fp32_dest_acc_en"):
        newton_schulz_kernel._validate_matrix_block(
            2, fp32_dest_acc_en=False, dst_full_sync_en=True, variant="bf16-fp32state"
        )


@pytest.mark.parametrize("matrix_block", [1, 2, 4, 8])
def test_cb_l1_accounting_matches_state_ledger_and_dram_inputs_fit(matrix_block):
    ttnn = _ttnn()
    definitions = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=matrix_block
    )
    expected_cb_bytes = {1: 88064, 2: 100352, 4: 186368, 8: 423936}
    expected_total_bytes = {1: 199424, 2: 211712, 4: 297728, 8: 535296}

    expected_state_pages = {1: 2, 2: 2, 4: 4, 8: 16}
    assert definitions[newton_schulz_kernel.CB_STATE_REAL][1] == expected_state_pages[matrix_block]
    assert definitions[newton_schulz_kernel.CB_STATE_IMAG][1] == expected_state_pages[matrix_block]
    assert (
        newton_schulz_kernel._cb_l1_bytes(ttnn, definitions)
        == expected_cb_bytes[matrix_block]
    )

    total = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="dram",
        matrix_block=matrix_block,
        double_buffer=False,
        variant="bf16-fp32state",
    )
    assert total == expected_total_bytes[matrix_block]
    assert total == (
        newton_schulz_kernel._L1_STATIC_BASE_BYTES
        + expected_cb_bytes[matrix_block]
    )


def test_l1_preflight_accepts_fitting_blocks_and_rejects_only_block8_for_l1_inputs():
    ttnn = _ttnn()
    expected_total_bytes = {1: 1_280_768, 2: 1_307_392, 4: 1_393_408}
    for matrix_block, expected_total in expected_total_bytes.items():
        total = newton_schulz_kernel._validate_l1_preflight(
            ttnn,
            batch=8192,
            core_count=110,
            state_dtype="fp32",
            fuse_s=True,
            output_memory="dram",
            matrix_block=matrix_block,
            double_buffer=False,
            variant="bf16-fp32state",
        )
        assert total == expected_total
        assert total <= newton_schulz_kernel._L1_TOTAL_BUDGET_BYTES

    with pytest.raises(ValueError, match="matrix_block=8 L1 preflight failed") as excinfo:
        newton_schulz_kernel._validate_l1_preflight(
            ttnn,
            batch=8192,
            core_count=110,
            state_dtype="fp32",
            fuse_s=True,
            output_memory="dram",
            matrix_block=8,
            double_buffer=False,
            variant="bf16-fp32state",
        )
    assert "L1 budget over by 115456 bytes" in str(excinfo.value)


def test_current_descriptors_match_7472_historical_catalogue_for_blocks_1_2_4():
    historical = json.loads(HISTORICAL_MATRIX_BLOCK_RECORD.read_text())
    implementation = historical["implementation"]
    historical_l1 = implementation["l1_accounting"]["by_matrix_block"]
    assert implementation["commit"] == "d82296220fe58affba6bc436da1761fff1bada7a"
    assert implementation["matrix_block_choices"] == [1, 2, 4]

    ttnn = _ttnn()
    removed_r_real_bytes = {1: 4096, 2: 4096, 4: 8192}
    for matrix_block in (1, 2, 4):
        definitions = newton_schulz_kernel._cb_definitions(
            ttnn, "fp32", fuse_s=True, matrix_block=matrix_block
        )
        assert newton_schulz_kernel._cb_l1_bytes(ttnn, definitions) == (
            historical_l1[str(matrix_block)]["cb_bytes"]
            - removed_r_real_bytes[matrix_block]
        )

    current = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=8
    )
    assert current[newton_schulz_kernel.CB_STATE_REAL][1] == 16
    assert current[newton_schulz_kernel.CB_STATE_IMAG][1] == 16


def test_block8_l1_preflight_rejects_with_full_accounting_and_cb_breakdown():
    ttnn = _ttnn()
    definitions = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=8
    )
    tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        matrix_block=8,
    )

    with pytest.raises(ValueError) as excinfo:
        newton_schulz_kernel._validate_l1_budget(
            ttnn, definitions, tensor_bytes=tensor_bytes, matrix_block=8
        )

    message = str(excinfo.value)
    assert "matrix_block=8 L1 preflight failed" in message
    assert "total CB bytes=423936" in message
    assert "static prefix=111360 bytes" in message
    assert "tensor bytes=1153024" in message
    assert "total=1688320 bytes" in message
    assert "budget=1572864 bytes" in message
    assert "L1 budget over by 115456 bytes" in message
    assert "largest CBs:" in message
    assert "CB_STATE_REAL=65536 bytes (cb_state_real)" in message
    assert "CBs in over-budget total:" in message
    assert "CB_X0_REAL=32768 bytes (cb_x0_real)" in message
    assert "CB_R_REAL" not in message
    assert "CB_PRODUCT_REAL=4096 bytes (cb_product_real)" in message


def test_matrix_block_ranges_keep_a_final_partial_group_and_align_core_ranges():
    assert newton_schulz_kernel._matrix_block_ranges(4, 6, 4) == [(4, 4), (8, 2)]
    assert newton_schulz_kernel._balanced_ranges(9, 2, 8) == [(0, 8), (8, 1)]
    ranges = newton_schulz_kernel._balanced_ranges(5, 2)
    assert ranges == [(0, 3), (3, 2)]
    assert newton_schulz_kernel._balanced_ranges(5, 2, 4) == [(0, 4), (4, 1)]

    aligned = newton_schulz_kernel._balanced_ranges(8192, 110, 4)
    assert {count for _, count in aligned} == {72, 76}
    assert all(start % 4 == 0 for start, _ in aligned)
    assert all(count % 4 == 0 for _, count in aligned)
    assert sum(count for _, count in aligned) == 8192


@pytest.mark.parametrize(
    ("memory_kwargs", "expected"),
    (
        pytest.param(
            {"input_memory": "l1"},
            ["l1"] * 7,
            id="all-inputs-l1",
        ),
        pytest.param(
            {"input_memory": "dram"},
            ["dram"] * 7,
            id="all-inputs-dram",
        ),
        pytest.param(
            {"input_memory": "l1", "r_memory": "l1", "x0_memory": "dram"},
            ["l1"] * 3 + ["dram"] * 2 + ["l1"] * 2,
            id="r-l1-x0-dram",
        ),
    ),
)
def test_reader_compile_dispatch_supports_each_input_placement(memory_kwargs, expected):
    assert newton_schulz_kernel._reader_input_memories(**memory_kwargs) == expected

    reader_functions = {
        "newton_schulz_reader_optimized.cpp": (
            "void read_matrix(",
            "void read_matrix_block(",
        ),
        "newton_schulz_reader_profile.cpp": (
            "void read_matrix(",
            "void read_matrix_profiled(",
            "void read_matrix_block(",
            "void read_matrix_block_profiled(",
        ),
    }
    for reader_name, functions in reader_functions.items():
        source = (KERNEL_DIR / reader_name).read_text()
        for function_name in functions:
            function_start = source.index(function_name)
            template_start = source.rfind("template", 0, function_start)
            opening = source.index("{", function_start)
            declaration = re.sub(
                r"\s+", " ", source[template_start:opening]
            )
            assert re.search(
                r"typename RAccessor, typename X0Accessor", declaration
            )
            signature = source[function_start:opening]
            for parameter in ("r_real", "r_negative_imag", "r_imag", "r_negative_real"):
                assert re.search(
                    rf"const RAccessor\s*&\s*{parameter}", signature
                )
            for parameter in ("x0_real", "x0_imag"):
                assert re.search(
                    rf"const X0Accessor\s*&\s*{parameter}", signature
                )


def test_per_input_memory_keeps_reader_order_and_compatibility_shorthand():
    assert newton_schulz_kernel._reader_input_memories() == ["l1"] * 7
    assert newton_schulz_kernel._reader_input_memories(input_memory="dram") == [
        "dram"
    ] * 7
    assert newton_schulz_kernel._reader_input_memories(
        r_memory="l1", x0_memory="dram"
    ) == ["l1"] * 3 + ["dram"] * 2 + ["l1"] * 2
    assert newton_schulz_kernel._reader_input_memories(
        input_memory="dram", x0_memory="l1"
    ) == ["dram"] * 3 + ["l1"] * 2 + ["dram"] * 2

    optimized_reader = (KERNEL_DIR / "newton_schulz_reader_optimized.cpp").read_text()
    assert "r_negative_imag_address = get_arg_val<std::uint32_t>(0)" in optimized_reader
    assert "r_imag_address = get_arg_val<std::uint32_t>(1)" in optimized_reader
    assert "r_negative_real_address = get_arg_val<std::uint32_t>(2)" in optimized_reader
    assert "x0_real_address = get_arg_val<std::uint32_t>(3)" in optimized_reader
    assert "x0_imag_address = get_arg_val<std::uint32_t>(4)" in optimized_reader

    class _Tensor:
        pass

    class _Ttnn:
        TILE_LAYOUT = "tile"
        L1_MEMORY_CONFIG = "l1"
        DRAM_MEMORY_CONFIG = "dram"

        def Tensor(self, values, dtype):
            tensor = _Tensor()
            tensor.values = values
            tensor.dtype = dtype
            return tensor

        def to_layout(self, tensor, layout):
            tensor.layout = layout
            return tensor

        def to_device(self, tensor, device, *, memory_config):
            tensor.device = device
            tensor.memory_config = memory_config
            return tensor

    ttnn = _Ttnn()
    values = np.zeros((1, 1, 32, 32), dtype=np.float32)
    assert newton_schulz_kernel._device_tensor(
        ttnn, values, object(), dtype="bf16", input_memory="l1"
    ).memory_config == "l1"
    assert newton_schulz_kernel._device_tensor(
        ttnn, values, object(), dtype="bf16", input_memory="dram"
    ).memory_config == "dram"
    with pytest.raises(ValueError, match="input_memory"):
        newton_schulz_kernel._device_tensor(
            ttnn, values, object(), dtype="bf16", input_memory="sram"
        )
    with pytest.raises(ValueError, match="r_memory"):
        newton_schulz_kernel._reader_input_memories(r_memory="sram")
    with pytest.raises(ValueError, match="x0_memory"):
        newton_schulz_kernel._reader_input_memories(x0_memory="sram")


def test_nonfused_l1_tensor_accounting_uses_fp32_identity_page():
    ttnn = _ttnn()
    fused = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="l1",
    )
    nonfused = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=False,
        output_memory="dram",
        input_memory="l1",
    )
    assert fused == 1_081_344
    assert nonfused == fused + 2_048


def test_dram_inputs_remove_tensor_l1_bytes_but_keep_static_cb_accounting():
    ttnn = _ttnn()
    l1_tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="l1",
    )
    dram_tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="dram",
    )
    assert l1_tensor_bytes == 1_081_344
    assert dram_tensor_bytes == 0
    assert newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        r_memory="l1",
        x0_memory="dram",
        matrix_block=8,
    ) == 497_664
    assert newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        r_memory="dram",
        x0_memory="l1",
        matrix_block=8,
    ) == 661_504
    definitions = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=8
    )
    total = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="dram",
        matrix_block=8,
        double_buffer=False,
        variant="bf16-fp32state",
    )
    assert total == (
        newton_schulz_kernel._L1_STATIC_BASE_BYTES
        + newton_schulz_kernel._cb_l1_bytes(ttnn, definitions)
    )
    assert total == 535_296
    with pytest.raises(ValueError, match="input_memory"):
        newton_schulz_kernel.NewtonSchulzKernel.prepare(
            None,
            None,
            np.zeros((1, 32, 32), dtype=np.complex64),
            input_memory="sram",
        )


def _function_source(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for offset in range(opening, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated C++ function: {signature}")


_CB_CALL = re.compile(
    r"\b(cb_(?:reserve_back|push_back|pop_front))\(\s*([A-Za-z_][A-Za-z0-9_]*)"
)
_CB_OPERATION = re.compile(
    r"\b(cb_(?:wait_front|reserve_back|push_back|pop_front))\(\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)


def _normalize_profile_wrappers(source: str) -> str:
    source = re.sub(
        r"\bblock_cb_(wait_front|reserve_back|push_back|pop_front)<profile_sample>\(\s*([^,\s]+)\s*,\s*([^,\s]+)[^)]*\)",
        r"cb_\1(\2, \3)",
        source,
    )
    return re.sub(r"\bblock_tile_regs_(acquire|wait)<profile_sample>\([^)]*\)", r"tile_regs_\1();", source)


def _without_comments(source: str) -> str:
    return _normalize_profile_wrappers(re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL))


def _cb_ledger(source: str) -> Counter:
    return Counter(_CB_CALL.findall(_without_comments(source)))


def _cb_operation_ledger(source: str) -> Counter:
    return Counter(_CB_OPERATION.findall(_without_comments(source)))


def _operation_guards(source: str, pattern: str) -> list[bool]:
    """Report whether each matching operation is inside ``!fuse_s``."""
    source = _without_comments(source)
    guards = []
    conditions = []
    index = 0
    while index < len(source):
        if source.startswith(pattern, index):
            guards.append(any("!fuse_s" in condition for condition in conditions))
        if source[index] == "{":
            prefix = source[max(0, index - 120) : index]
            match = re.search(r"if constexpr \(([^)]*)\)\s*$", prefix)
            conditions.append(match.group(1) if match else "")
        elif source[index] == "}" and conditions:
            conditions.pop()
        index += 1
    return guards


def test_fused_reader_and_compute_ledgers_never_touch_positive_r_real():
    optimized_reader = (KERNEL_DIR / "newton_schulz_reader_optimized.cpp").read_text()
    profile_reader = (KERNEL_DIR / "newton_schulz_reader_profile.cpp").read_text()
    for reader in (optimized_reader, profile_reader):
        for operation in (
            "cb_reserve_back(cb_r_real",
            "cb_push_back(cb_r_real",
            "cb_wait_front(cb_r_real",
            "cb_pop_front(cb_r_real",
            "read_one(cb_r_real",
        ):
            assert all(_operation_guards(reader, operation)), operation
        assert all(_operation_guards(reader, "noc_async_read_page(tile_id, r_real"))
        assert all(_operation_guards(reader, "noc_async_read_page(tile, r_real"))

    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    for function_name in ("void wait_r_inputs(", "void wait_r_inputs_block("):
        function = _function_source(compute, function_name)
        assert all(_operation_guards(function, "cb_wait_front(cb_r_real"))
    for function_name in ("void kernel_main_impl(", "void process_matrix_block("):
        function = _function_source(compute, function_name)
        assert all(_operation_guards(function, "cb_pop_front(cb_r_real"))


def _complex_block_branch_source(source: str, *, one_dest: bool) -> str:
    function = _function_source(source, "void complex_matmul_block(")
    branch_start = function.index("if constexpr (one_dest_half) {")
    else_start = function.index("    } else {", branch_start)
    post_start = function.index("\n    if constexpr (one_dest_half) {", else_start)
    right_start = function.index("\n    if (consume_right)", post_start)
    if one_dest:
        return _normalize_profile_wrappers(function[branch_start:else_start] + function[post_start:])
    return _normalize_profile_wrappers(function[else_start:post_start] + function[right_start:])


def _writer_branch_source(source: str, *, matrix_block: int) -> str:
    function = _function_source(source, "void kernel_main()")
    branch_start = function.index("if constexpr (matrix_block == 1) {")
    else_start = function.index("    } else {", branch_start)
    return function[branch_start:else_start] if matrix_block == 1 else function[else_start:]


@pytest.mark.parametrize(
    ("matrix_block", "one_dest"),
    ((1, False), (2, False), (4, False), (8, True)),
)
def test_matrix_block_branch_ledgers_publish_and_consume_every_page(
    matrix_block, one_dest
):
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    expected = Counter(
        {
            ("cb_reserve_back", "output_real"): 1,
            ("cb_reserve_back", "output_imag"): 1,
            ("cb_push_back", "output_real"): 1,
            ("cb_push_back", "output_imag"): 1,
            ("cb_pop_front", "left_real"): 1,
            ("cb_pop_front", "left_imag_for_real"): 1,
            ("cb_pop_front", "left_imag_for_imag"): 1,
            ("cb_pop_front", "right_real"): 1,
            ("cb_pop_front", "right_imag"): 1,
        }
    )

    if matrix_block == 1:
        real = _function_source(compute, "void complex_real_impl(")
        imag = _function_source(compute, "void complex_imag_impl(")
        pack = _function_source(compute, "void pack_one(")
        single = _function_source(compute, "void complex_matmul(")
        assert _cb_ledger(real + imag) == Counter(
            {
                ("cb_reserve_back", "output"): 2,
            }
        )
        assert _cb_ledger(pack) == Counter({("cb_push_back", "output"): 1})
        assert real.count("pack_one(output)") == 1
        assert imag.count("pack_one(output)") == 1
        assert _cb_ledger(single) == Counter(
            {
                ("cb_pop_front", "left_real"): 1,
                ("cb_pop_front", "left_imag_for_real"): 1,
                ("cb_pop_front", "left_imag_for_imag"): 1,
                ("cb_pop_front", "right_real"): 1,
                ("cb_pop_front", "right_imag"): 1,
            }
        )
        return

    branch = _complex_block_branch_source(compute, one_dest=one_dest)
    assert _cb_ledger(branch) == expected
    pack_offsets = {
        "output_real": branch.index("pack_tile_block(0, output_real, block_count)"),
        "output_imag": branch.index(
            "pack_tile_block(0, output_imag, block_count)"
            if one_dest
            else "pack_tile_block(block_count, output_imag, block_count)"
        ),
    }
    for output, pack_offset in pack_offsets.items():
        push_offset = branch.index(f"cb_push_back({output}, block_count)")
        assert pack_offset < push_offset


@pytest.mark.parametrize("matrix_block", [1, 2, 4, 8])
def test_reader_writer_and_state_helpers_balance_every_cb_page(matrix_block):
    reader = (KERNEL_DIR / "newton_schulz_reader_optimized.cpp").read_text()
    writer = (KERNEL_DIR / "newton_schulz_writer.cpp").read_text()
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    input_names = (
        "cb_r_real",
        "cb_r_negative_imag",
        "cb_r_imag",
        "cb_r_negative_real",
        "cb_x0_real",
        "cb_x0_imag",
    )

    if matrix_block == 1:
        read_one = _function_source(reader, "void read_one(")
        assert _cb_ledger(read_one) == Counter(
            {
                ("cb_reserve_back", "cb"): 1,
                ("cb_push_back", "cb"): 1,
            }
        )
        read_matrix = _function_source(reader, "void read_matrix(")
        assert read_matrix.count("read_one(") == len(input_names)
        compute_owner = _function_source(compute, "void kernel_main_impl()")
        state_source_name = "void fused_s_matmul("
        negate_source_name = "void negate_state_imag_impl("
        wait_source_name = "void wait_r_inputs("
    else:
        read_matrix = _function_source(reader, "void read_matrix_block(")
        read_ledger = _cb_ledger(read_matrix)
        for name in input_names:
            assert read_ledger[("cb_reserve_back", name)] == 1
            assert read_ledger[("cb_push_back", name)] == 1
        compute_owner = _function_source(compute, "void process_matrix_block(")
        state_source_name = "void fused_s_matmul_block("
        negate_source_name = "void negate_state_imag_block("
        wait_source_name = "void wait_r_inputs_block("

    owner_ledger = _cb_ledger(compute_owner)
    for name in input_names[:4]:
        assert owner_ledger[("cb_pop_front", name)] == 1
    wait_ledger = _cb_operation_ledger(_function_source(compute, wait_source_name))
    for name in input_names[:4]:
        assert wait_ledger[("cb_wait_front", name)] == 1

    writer_branch = _writer_branch_source(writer, matrix_block=matrix_block)
    writer_ledger = _cb_operation_ledger(writer_branch)
    for name in ("cb_output_real", "cb_output_imag"):
        assert writer_ledger[("cb_wait_front", name)] == 1
        assert writer_ledger[("cb_pop_front", name)] == 1

    state_ledger = _cb_ledger(_function_source(compute, state_source_name))
    for name in ("cb_s_real", "cb_s_imag"):
        assert state_ledger[("cb_reserve_back", name)] == 1
        assert state_ledger[("cb_push_back", name)] == 1

    negate_source = _function_source(compute, negate_source_name)
    negate_ledger = _cb_ledger(negate_source)
    assert negate_ledger[("cb_reserve_back", "cb_negative_x_imag")] == 1
    if matrix_block == 1:
        assert negate_source.count("pack_one(cb_negative_x_imag)") == 1
        assert _cb_ledger(_function_source(compute, "void pack_one(")) == Counter(
            {("cb_push_back", "output"): 1}
        )
    else:
        assert negate_ledger[("cb_push_back", "cb_negative_x_imag")] == 1


def _assert_capacity_at_least(definitions, index, required_pages):
    actual_pages = definitions[index][1]
    if actual_pages < required_pages:
        raise ValueError(
            f"CB {index} has {actual_pages} pages; ledger requires {required_pages}"
        )


@pytest.mark.parametrize(
    ("matrix_block", "required_state_pages"),
    ((1, 2), (2, 2), (4, 4), (8, 16)),
)
def test_state_capacity_follows_reserve_pop_order_and_rejects_under_capacity(
    matrix_block, required_state_pages
):
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    definitions = newton_schulz_kernel._cb_definitions(
        _ttnn(), "fp32", fuse_s=True, matrix_block=matrix_block
    )
    for index in (
        newton_schulz_kernel.CB_STATE_REAL,
        newton_schulz_kernel.CB_STATE_IMAG,
    ):
        assert definitions[index][1] == required_state_pages
        _assert_capacity_at_least(definitions, index, required_state_pages)

    if matrix_block == 1:
        real = _function_source(compute, "void complex_real_impl(")
        imag = _function_source(compute, "void complex_imag_impl(")
        assert _cb_ledger(real + imag)[("cb_reserve_back", "output")] == 2
    else:
        branch = _complex_block_branch_source(compute, one_dest=matrix_block == 8)
        pop_offset = branch.index("cb_pop_front(left_real, block_count)")
        reserve_offset = branch.index("cb_reserve_back(output_real, block_count)")
        if matrix_block == 8:
            assert reserve_offset < pop_offset
        else:
            assert pop_offset < reserve_offset

    undersized = dict(definitions)
    undersized[newton_schulz_kernel.CB_STATE_REAL] = (
        undersized[newton_schulz_kernel.CB_STATE_REAL][0],
        required_state_pages - 1,
    )
    with pytest.raises(ValueError, match="ledger requires"):
        _assert_capacity_at_least(
            undersized,
            newton_schulz_kernel.CB_STATE_REAL,
            required_state_pages,
        )


def test_block8_compute_uses_one_dest_half_for_products_s_and_output():
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    complex_start = compute.index("template <bool one_dest_half, bool profile_sample>\nvoid complex_matmul_block")
    fused_start = compute.index("template <bool one_dest_half, bool profile_sample>\nvoid fused_s_matmul_block")
    complex_source = compute[complex_start:fused_start]
    assert "if constexpr (one_dest_half)" in complex_source
    assert complex_source.count("block_tile_regs_acquire<profile_sample>") == 3
    assert complex_source.count("block_tile_regs_wait<profile_sample>") == 3
    assert complex_source.count("tile_regs_commit();") == 3
    assert "pack_tile_block(0, output_real, block_count);" in complex_source
    assert "pack_tile_block(0, output_imag, block_count);" in complex_source
    two_dest_source = _normalize_profile_wrappers(complex_source[complex_source.index("    } else {") :])
    assert two_dest_source.index("cb_pop_front(left_real, block_count)") < two_dest_source.index(
        "cb_reserve_back(output_real, block_count)"
    )
    assert "complex_matmul_block<one_dest_half, profile_sample>" in compute
    assert "fused_s_matmul_block<one_dest_half, profile_sample>" in compute
    assert "process_matrix_block<" in compute
    assert "reload_r," in compute


def test_invalid_input_memory_is_rejected_before_custom_prepare():
    shape = SimpleNamespace(family="newton_schulz", m=32, k=32, n=32, batch=8192)
    from enodia.tt.bench import run_matmul

    record = run_matmul.run_custom_newton_schulz(
        object(),
        object(),
        shape,
        dtype_name="bfloat16",
        memory_name="l1",
        variant="bf16-fp32state",
        input_memory="sram",
        iters=1,
        repeats=1,
    )
    assert record["status"] == "failed"
    assert "input_memory" in record["error"]


def test_reader_writer_stream_groups_and_pop_bulk():
    reader = (KERNEL_DIR / "newton_schulz_reader_optimized.cpp").read_text()
    writer = (KERNEL_DIR / "newton_schulz_writer.cpp").read_text()
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()

    assert "read_matrix_block" in reader
    assert '#include "api/compute/pack.h"' in compute
    assert "cb_reserve_back(cb_r_real, block_count)" in reader
    assert "cb_push_back(cb_r_real, block_count)" in reader
    assert "offset += matrix_block" in reader
    assert "cb_wait_front(cb_output_real, block_count)" in writer
    assert "cb_pop_front(cb_output_real, block_count)" in writer
    assert "pack_tile_block(0, output_real, block_count)" in compute
    assert "cb_push_back(output_real, block_count)" in compute
    assert "cb_pop_front(left_real, block_count)" in compute
    assert "fused_s_matmul_block" in compute


def test_cli_exposes_matrix_block_with_issue100_default():
    parser = run_matmul._build_parser()
    assert parser.parse_args([]).matrix_block == 8
    assert parser.parse_args([]).input_memory == "l1"
    assert parser.parse_args(["--input-memory", "dram"]).input_memory == "dram"
    per_tensor = parser.parse_args(["--r-memory", "l1", "--x0-memory", "dram"])
    assert per_tensor.r_memory == "l1"
    assert per_tensor.x0_memory == "dram"
    assert parser.parse_args(["--matrix-block", "2"]).matrix_block == 2
    assert parser.parse_args(["--matrix-block", "4"]).matrix_block == 4
    assert parser.parse_args(["--matrix-block", "8"]).matrix_block == 8
    assert parser.parse_args(["--reload-r"]).reload_r is True
    assert parser.parse_args(["--compare-reload-r"]).compare_reload_r is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--matrix-block", "3"])
