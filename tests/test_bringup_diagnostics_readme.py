from pathlib import Path

README_PATH = (
    Path(__file__).resolve().parents[1]
    / "enodia/tt/bench/kernels/bringup_diagnostics_README.md"
)


def test_bringup_readme_rules_contract_is_standalone_and_complete():
    readme = README_PATH.read_text(encoding="utf-8")
    rules_start = readme.index("## Rules")
    record_start = readme.index("## Stage guide", rules_start)
    rules = " ".join(readme[rules_start:record_start].split())

    assert rules_start < record_start
    required_statements = (
        "Output CBs are separated when switching output identity or format",
        "the diagnostics established that rule",
        "Initialization/reconfiguration is performed before every data-format boundary",
        "short operation init plus explicit unpack-side `reconfig_data_format`",
        "pack-side `pack_reconfig_data_format`",
        "Full common initialization is not used mid-kernel",
        "After the first matmul-family call, no `*_init_common` may occur in a normal compute kernel",
        "use short init plus explicit reconfiguration instead",
        "small explicit exclusion list for legacy diagnostic failure reproductions",
        "new production/diagnostic sources must not be added to it casually",
        "The all-Float32-state Stage 61 passed at relative error `0.0040098457`",
        "Stage 70's isolated BF16 first-residual Variant A passed at `0.0014451430179178715`",
        "not full-algorithm evidence",
        "Stage 72's four-BF16/four-Float32 hybrid completed but failed at `0.7506909370422363` against `0.01`",
        "The hybrid is not accepted",
        "remaining numerical cause is unresolved",
        "Every new numerical stage starts with `TT_METAL_WATCHER=1` and an external 60-second cap",
        "unless an explicit opt-out is recorded",
    )
    missing = [statement for statement in required_statements if statement not in rules]
    assert not missing, f"README Rules section is missing: {missing}"

    record = readme[record_start:]
    assert "## Stage 72 board result and recovery" in record
    assert "0.7506909370422363" in record
