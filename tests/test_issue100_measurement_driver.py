from tools.newton_schulz_issue100_same_run import (
    ISSUE100_COMPARISON_CONFIGS,
    ISSUE100_SHAPES,
)


def test_issue100_same_run_driver_pins_both_default_configurations():
    assert ISSUE100_SHAPES == (
        "newton_schulz_L32_b8192",
        "newton_schulz_L16_b8192",
    )
    assert ISSUE100_COMPARISON_CONFIGS == (
        {
            "name": "new_default",
            "variant": "bf16",
            "matrix_block": 8,
            "double_buffer": True,
            "fuse_s": True,
            "math_fidelity": "HiFi3",
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": True,
        },
        {
            "name": "previous_default",
            "variant": "bf16-fp32state",
            "matrix_block": 4,
            "double_buffer": True,
            "fuse_s": True,
            "math_fidelity": "HiFi3",
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": True,
        },
    )
