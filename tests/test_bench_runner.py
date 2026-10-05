"""The runner's accounting must match what it actually executes.

A benchmark that divides the FLOPs of four matmuls by the time of one
reports a number four times too large, and does it silently. These tests
pin the execution count to the accounting, using a stub in place of the
toolchain so they run anywhere.
"""

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from enodia.tt.bench import run_matmul
from enodia.tt.bench.configs import configuration_catalogue
from enodia.tt.bench.newton_schulz_reference import (
    COMPLEX_MATMULS_PER_INVERSE,
    inverse_flops,
)
from enodia.tt.bench.shapes import MatmulShape, default_catalogue, total_flops


class _StubTensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.deallocated = False


class _StubConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _StubCoreGrid:
    def __init__(self, *, y: int, x: int) -> None:
        self.y = y
        self.x = x


class _StubCoreCoord:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class _StubTtnn:
    """Records what the runner asked the toolchain to do."""

    TILE_LAYOUT = "tile"
    bfloat16 = "bf16"
    float32 = "fp32"
    NOC = SimpleNamespace(NOC_0="noc-0")
    DRAM_MEMORY_CONFIG = "dram"
    L1_MEMORY_CONFIG = "l1"
    CoreGrid = _StubCoreGrid
    CoreCoord = _StubCoreCoord
    MatmulMultiCoreReuseProgramConfig = _StubConfig
    MatmulMultiCoreReuseMultiCastProgramConfig = _StubConfig
    MatmulMultiCoreReuseMultiCast1DProgramConfig = _StubConfig
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig = _StubConfig
    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig = _StubConfig

    def __init__(self, fail_on: str | None = None) -> None:
        self.matmul_calls = 0
        self.sync_calls = 0
        self.memory_config_calls = 0
        self.tensor_factory_calls = 0
        self.deallocated: list[str] = []
        self.fail_on = fail_on

    def MemoryConfig(self, *args, **kwargs):
        self.memory_config_calls += 1
        return _StubConfig(args=args, kwargs=kwargs)

    def rand(self, shape, **kwargs):
        self.tensor_factory_calls += 1
        if self.fail_on == "rand":
            raise RuntimeError("out of memory")
        return _StubTensor(f"rand{shape}")

    def ones(self, shape, **kwargs):
        self.tensor_factory_calls += 1
        if self.fail_on in ("rand", "ones"):
            raise RuntimeError("out of memory")
        return _StubTensor(f"ones{shape}")

    def zeros(self, shape, **kwargs):
        self.tensor_factory_calls += 1
        if self.fail_on in ("rand", "ones", "zeros"):
            raise RuntimeError("out of memory")
        return _StubTensor(f"zeros{shape}")

    def matmul(self, a, b, **kwargs):
        self.matmul_calls += 1
        return _StubTensor("out")

    def synchronize_device(self, device):
        self.sync_calls += 1

    def deallocate(self, tensor):
        tensor.deallocated = True
        self.deallocated.append(tensor.name)


def _shape(real_matmuls: int, *, batch: int = 2) -> MatmulShape:
    return MatmulShape(
        name="probe",
        batch=batch,
        m=4,
        k=4,
        n=4,
        real_matmuls=real_matmuls,
        family="newton_schulz",
        note="",
    )


class _StubDevice:
    def __init__(self, worker_count: int) -> None:
        self.worker_count = worker_count
        self.assignment_calls = 0

    def get_optimal_dram_bank_to_logical_worker_assignment(self, noc):
        self.assignment_calls += 1
        return [object()] * self.worker_count


def test_issue94_modes_and_rows_preserve_fixed_selection_metadata_without_a_device():
    parser = run_matmul._build_parser()
    assert parser.parse_args(["--issue94-catalogue"]).issue94_catalogue is True
    assert parser.parse_args(["--issue94-correctness"]).issue94_correctness is True
    rows = [run_matmul._issue94_row_metadata(config) for config in run_matmul.ISSUE94_CONFIGS]
    assert [row["name"] for row in rows] == [
        "bf16_fp32dest_block4",
        "bf16_bf16dest_block4",
        "bf16_fp32state_block4",
    ]
    assert [row["dest_slot_limit"] for row in rows] == [8, 16, 8]
    assert all(row["double_buffer"] is False for row in rows)
    assert all(row["iterations"] == 12 for row in rows)
    assert all(row["input_memory"] == "l1" and row["output_memory"] == "dram" for row in rows)


def test_issue94_correctness_writer_records_fixed_reference_without_a_device(
    monkeypatch, tmp_path
):
    ttnn = _StubTtnn()
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    config = run_matmul.ISSUE94_CONFIGS[0]
    monkeypatch.setattr(run_matmul, "ISSUE94_CORRECTNESS_CASES", ((2, 1),))
    monkeypatch.setattr(run_matmul, "ISSUE94_CONFIGS", (config,))

    from enodia.tt.bench import newton_schulz_kernel

    def fake_kernel(_ttnn, _device, matrices, **_kwargs):
        return run_matmul._issue94_fixed_reference(
            run_matmul._issue94_bf16_round_complex(matrices),
            run_matmul._issue94_initial_value(matrices),
        )

    monkeypatch.setattr(newton_schulz_kernel, "run_newton_schulz_kernel", fake_kernel)
    output = tmp_path / "issue94-correctness.json"

    assert run_matmul.main(["--issue94-correctness", "--out", str(output)]) == 0

    payload = json.loads(output.read_text())
    assert {
        result["reference"] for result in payload["results"]
    } == {"enodia/tt/bench/run_matmul.py:_issue94_fixed_reference"}


def test_issue94_correctness_keeps_rows_after_an_intermediate_failure(
    monkeypatch, tmp_path
):
    ttnn = _StubTtnn()
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(run_matmul, "ISSUE94_CORRECTNESS_CASES", ((2, 1),))

    from enodia.tt.bench import newton_schulz_kernel

    calls = 0

    def fake_kernel(_ttnn, _device, matrices, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected middle row failure")
        return run_matmul._issue94_fixed_reference(
            run_matmul._issue94_bf16_round_complex(matrices),
            run_matmul._issue94_initial_value(matrices),
        )

    monkeypatch.setattr(newton_schulz_kernel, "run_newton_schulz_kernel", fake_kernel)
    output = tmp_path / "issue94-correctness-failure.json"

    assert run_matmul.main(["--issue94-correctness", "--out", str(output)]) == 0

    payload = json.loads(output.read_text())
    results = payload["results"]
    assert output.exists()
    assert [result["status"] for result in results] == ["pass", "failed", "pass"]
    failed = results[1]
    config = run_matmul.ISSUE94_CONFIGS[1]
    assert failed["size"] == 2
    assert failed["batch"] == 1
    assert failed["name"] == config["name"]
    assert failed["variant"] == config["variant"]
    assert failed["matrix_block"] == config["matrix_block"]
    assert failed["error"] == "RuntimeError: injected middle row failure"
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "injected middle row failure"


def test_repeatable_shape_and_config_filters_parse_without_a_device():
    args = run_matmul._build_parser().parse_args(
        [
            "--only",
            "newton_schulz_L16_b1024",
            "--only",
            "newton_schulz_L32_b1024",
            "--config-kind",
            "batched_dram_sharded",
        ]
    )

    assert args.only == ["newton_schulz_L16_b1024", "newton_schulz_L32_b1024"]
    assert args.config_kind == ["batched_dram_sharded"]
    fidelity_args = run_matmul._build_parser().parse_args(
        ["--custom-math-fidelity", "LoFi", "--custom-math-fidelity", "HiFi3"]
    )
    assert fidelity_args.custom_math_fidelity == ["LoFi", "HiFi3"]
    default_flags = run_matmul._build_parser().parse_args([])
    assert default_flags.fuse_s is False
    assert default_flags.batch_reads is False
    assert default_flags.input_memory == "l1"
    assert default_flags.r_memory is None
    assert default_flags.x0_memory is None
    assert default_flags.double_buffer is True
    assert default_flags.compare_double_buffer is False
    assert default_flags.fp32_dest_acc_en is True
    assert default_flags.dst_full_sync_en is True
    enabled_flags = run_matmul._build_parser().parse_args(
        [
            "--fuse-s",
            "--batch-reads",
            "--input-memory",
            "dram",
            "--r-memory",
            "l1",
            "--x0-memory",
            "dram",
            "--double-buffer",
        ]
    )
    assert enabled_flags.fuse_s is True
    assert enabled_flags.batch_reads is True
    assert enabled_flags.input_memory == "dram"
    assert enabled_flags.r_memory == "l1"
    assert enabled_flags.x0_memory == "dram"
    assert enabled_flags.double_buffer is True
    alias_flags = run_matmul._build_parser().parse_args(["--block-double-buffer"])
    assert alias_flags.double_buffer is True


@pytest.mark.parametrize("reload_flag", ["--reload-r", "--compare-reload-r"])
def test_acceptance_catalogue_rejects_reload_modes_before_ttnn_import(
    monkeypatch, capsys, reload_flag
):
    original_import = builtins.__import__

    def reject_ttnn_import(name, *args, **kwargs):
        if name == "ttnn":
            raise AssertionError("invalid CLI arguments imported ttnn")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ttnn_import)

    with pytest.raises(SystemExit) as excinfo:
        run_matmul.main(["--acceptance-catalogue", reload_flag])

    assert excinfo.value.code == 2
    assert (
        "--reload-r/--compare-reload-r require the normal custom-row runner"
        in capsys.readouterr().err
    )


def test_normal_reload_r_runner_remains_accepted(monkeypatch, tmp_path):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    calls = []

    def fake_custom(*args, **kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "kind": "custom_newton_schulz",
            "variant": kwargs["variant"],
            "math_fidelity": kwargs["math_fidelity"],
            "reload_r": kwargs["reload_r"],
            "output_memory": "l1",
            "achieved_tflops": 1.0,
            "seconds_per_launch_p50": 1.0,
            "seconds_per_launch_p99": 1.0,
            "seconds_per_launch_p99_9": 1.0,
        }

    monkeypatch.setattr(run_matmul, "run_custom_newton_schulz", fake_custom)
    output = tmp_path / "reload-r.json"

    assert (
        run_matmul.main(
            [
                "--only",
                "newton_schulz_L32_b8192",
                "--dtype",
                "bfloat16",
                "--memory",
                "l1",
                "--kind",
                "custom_newton_schulz",
                "--reload-r",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0]["reload_r"] is True
    assert json.loads(output.read_text())["selection"]["reload_r"] is True


def test_repeatable_shape_filters_use_or_substring_semantics():
    shapes = default_catalogue()

    selected = run_matmul._select_shapes(
        shapes,
        ["newton_schulz_L16_b1024", "newton_schulz_L32_b1024"],
    )

    assert [shape.name for shape in selected] == [
        "newton_schulz_L16_b1024",
        "newton_schulz_L32_b1024",
    ]
    assert [shape.name for shape in run_matmul._select_shapes(shapes, ["L16_b1024"])] == [
        "newton_schulz_L16_b1024"
    ]


def test_config_kind_filter_enumerates_exactly_four_default_dtype_rows():
    selected = run_matmul._select_shapes(
        default_catalogue(),
        ["newton_schulz_L16_b1024", "newton_schulz_L32_b1024"],
    )
    rows = [
        (shape.name, dtype_name, program_spec, memory_name, base_memory_name)
        for shape in selected
        for dtype_name in ("bfloat16", "float32")
        for program_spec, memory_name, base_memory_name in run_matmul._row_specs(
            shape, ["dram", "l1"], "all", ["batched_dram_sharded"]
        )
    ]

    assert len(rows) == 4
    assert {(name, dtype) for name, dtype, *_ in rows} == {
        ("newton_schulz_L16_b1024", "bfloat16"),
        ("newton_schulz_L16_b1024", "float32"),
        ("newton_schulz_L32_b1024", "bfloat16"),
        ("newton_schulz_L32_b1024", "float32"),
    }
    assert all(program_spec.kind == "batched_dram_sharded" for _, _, program_spec, *_ in rows)
    assert all(memory_name == "batch_sharded_dram" for _, _, _, memory_name, _ in rows)
    assert all(base_memory_name == "dram" for _, _, _, _, base_memory_name in rows)


def test_stock_fidelity_metadata_matches_source_mapping():
    default_bf16 = run_matmul._stock_math_fidelity("bfloat16", None)
    explicit_bf16 = run_matmul._stock_math_fidelity("bfloat16", object())
    default_fp32 = run_matmul._stock_math_fidelity("float32", None)

    assert default_bf16["math_fidelity"] == "HiFi2"
    assert explicit_bf16["math_fidelity"] == "LoFi"
    assert default_fp32["math_fidelity"] == "HiFi4"
    assert "increase_fidelity" in default_bf16["math_fidelity_source"]
    assert "program_config" in explicit_bf16["math_fidelity_source"]
    assert "source mapping" in default_fp32["math_fidelity_source"]


def test_custom_rows_repeat_for_requested_fidelities(monkeypatch, tmp_path):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    calls = []

    def fake_custom(*args, **kwargs):
        calls.append(kwargs["math_fidelity"])
        return {
            "status": "ok",
            "kind": "custom_newton_schulz",
            "variant": kwargs["variant"],
            "math_fidelity": kwargs["math_fidelity"],
            "output_memory": "l1",
            "achieved_tflops": 1.0,
            "seconds_per_launch_p50": 1.0,
            "seconds_per_launch_p99": 1.0,
            "seconds_per_launch_p99_9": 1.0,
        }

    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(run_matmul, "run_custom_newton_schulz", fake_custom)
    output = tmp_path / "fidelity.json"
    assert run_matmul.main(
        [
            "--only", "newton_schulz_L32_b8192",
            "--dtype", "bfloat16",
            "--memory", "l1",
            "--kind", "custom_newton_schulz",
            "--custom-math-fidelity", "LoFi",
            "--custom-math-fidelity", "HiFi4",
            "--out", str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text())
    assert calls == ["LoFi", "HiFi4"]
    assert [row["math_fidelity"] for row in payload["results"]] == ["LoFi", "HiFi4"]


def test_custom_flags_reach_dispatch_and_result_metadata(monkeypatch, tmp_path):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    calls = []

    def fake_custom(*args, **kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "kind": "custom_newton_schulz",
            "variant": kwargs["variant"],
            "math_fidelity": kwargs["math_fidelity"],
            "fuse_s": kwargs["fuse_s"],
            "batch_reads": kwargs["batch_reads"],
            "output_memory": "l1",
            "achieved_tflops": 1.0,
            "seconds_per_launch_p50": 1.0,
            "seconds_per_launch_p99": 1.0,
            "seconds_per_launch_p99_9": 1.0,
        }

    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(run_matmul, "run_custom_newton_schulz", fake_custom)
    output = tmp_path / "flags.json"
    assert run_matmul.main(
        [
            "--only",
            "newton_schulz_L32_b8192",
            "--dtype",
            "bfloat16",
            "--memory",
            "l1",
            "--kind",
            "custom_newton_schulz",
            "--fuse-s",
            "--batch-reads",
            "--r-memory",
            "l1",
            "--x0-memory",
            "dram",
            "--out",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text())
    assert len(calls) == 1
    assert calls[0]["fuse_s"] is True
    assert calls[0]["batch_reads"] is True
    assert calls[0]["r_memory"] == "l1"
    assert calls[0]["x0_memory"] == "dram"
    assert payload["selection"]["r_memory"] == "l1"
    assert payload["selection"]["x0_memory"] == "dram"
    assert payload["results"][0]["program_config"]["r_memory"] == "l1"
    assert payload["results"][0]["program_config"]["x0_memory"] == "dram"
    assert payload["results"][0]["program_config"]["fuse_s"] is True
    assert payload["results"][0]["program_config"]["batch_reads"] is True


def test_double_buffer_selection_reaches_same_run_comparison(monkeypatch, tmp_path):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    calls = []

    def fake_custom(*args, **kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "kind": "custom_newton_schulz",
            "variant": kwargs["variant"],
            "math_fidelity": kwargs["math_fidelity"],
            "double_buffer": kwargs["double_buffer"],
            "output_memory": "dram",
            "achieved_tflops": 1.0,
            "seconds_per_launch_p50": 1.0,
            "seconds_per_launch_p99": 1.0,
            "seconds_per_launch_p99_9": 1.0,
        }

    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(run_matmul, "run_custom_newton_schulz", fake_custom)
    output = tmp_path / "double-buffer.json"
    assert run_matmul.main(
        [
            "--only",
            "newton_schulz_L32_b8192",
            "--dtype",
            "bfloat16",
            "--memory",
            "l1",
            "--kind",
            "custom_newton_schulz",
            "--fuse-s",
            "--matrix-block",
            "4",
            "--compare-double-buffer",
            "--out",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text())
    assert [call["double_buffer"] for call in calls] == [False, True]
    assert [row["program_config"]["double_buffer"] for row in payload["results"]] == [
        False,
        True,
    ]
    assert [row["row"] for row in payload["results"]] == [
        "custom_block4_double_buffer_false",
        "custom_block4_double_buffer_true",
    ]
    assert payload["selection"]["compare_double_buffer"] is True


def test_row_specs_applies_dtype_specific_catalogue_filtering():
    shape = next(
        shape for shape in default_catalogue() if shape.name == "beamspace_B16_ch128_p4096"
    )

    bf16_rows = list(
        run_matmul._row_specs(
            shape, ["dram", "l1"], "all", ["reuse"], dtype="bfloat16"
        )
    )
    fp32_rows = list(
        run_matmul._row_specs(
            shape, ["dram", "l1"], "all", ["reuse"], dtype="float32"
        )
    )

    assert len(bf16_rows) == 2
    assert fp32_rows == []
    assert {row[1] for row in bf16_rows} == {"dram", "l1"}


def test_batched_dram_worker_mismatch_fails_before_board_work():
    ttnn = _StubTtnn()
    device = _StubDevice(worker_count=7)
    shape = _shape(real_matmuls=1, batch=1024)
    config = next(
        config
        for config in configuration_catalogue(shape)
        if config.kind == "batched_dram_sharded"
    )

    record = run_matmul.run_shape(
        ttnn,
        device=device,
        shape=shape,
        dtype="bf16",
        memory_config="dram",
        program_spec=config,
        iters=1,
        repeats=1,
    )

    assert record["status"] == "failed"
    assert "catalogue expects 8 p150 DRAM workers, device reported 7" in record["error"]
    assert device.assignment_calls == 1
    assert ttnn.memory_config_calls == 0
    assert ttnn.tensor_factory_calls == 0
    assert ttnn.matmul_calls == 0


@pytest.mark.parametrize("real_matmuls", [1, 2, 4])
def test_execution_count_matches_the_declared_real_matmuls(real_matmuls):
    """One logical operation costs `real_matmuls` real ones, so that many run."""
    ttnn = _StubTtnn()
    iters, repeats = 3, 2

    record = run_matmul.run_shape(
        ttnn,
        device=object(),
        shape=_shape(real_matmuls),
        dtype="bf16",
        memory_config="dram",
        iters=iters,
        repeats=repeats,
    )

    assert record["status"] == "ok"
    expected = real_matmuls * (1 + iters * repeats)  # one warm-up round, then the blocks
    assert ttnn.matmul_calls == expected
    assert len(record["seconds_per_iteration_samples"]) == repeats
    assert record["seconds_per_iteration"] == min(record["seconds_per_iteration_samples"])


def test_a_shape_that_cannot_be_allocated_is_recorded_not_raised():
    ttnn = _StubTtnn(fail_on="rand")

    record = run_matmul.run_shape(
        ttnn,
        device=object(),
        shape=_shape(4),
        dtype="bf16",
        memory_config="l1",
        iters=1,
        repeats=1,
    )

    assert record["status"] == "failed"
    assert "out of memory" in record["error"]


def test_inputs_are_released_even_when_a_shape_fails():
    ttnn = _StubTtnn()
    run_matmul.run_shape(
        ttnn,
        device=object(),
        shape=_shape(1),
        dtype="bf16",
        memory_config="dram",
        iters=1,
        repeats=1,
    )

    assert len([n for n in ttnn.deallocated if n.startswith(("rand", "ones", "zeros"))]) == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--iters", "0"],
        ["--iters", "-1"],
        ["--repeats", "0"],
        ["--peak-tflops", "0"],
        ["--peak-tflops", "-5"],
        ["--peak-tflops", "nan"],
        ["--peak-tflops", "inf"],
    ],
)
def test_invalid_controls_are_rejected_before_the_device_is_opened(argv):
    """Rejection happens before the toolchain import, so it needs no board —
    and a zero iteration count must not reach the timing loop and divide."""
    with pytest.raises(SystemExit) as excinfo:
        run_matmul.main(argv)

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "selection",
    [
        ["--config-kind", "not-a-catalogue-kind"],
        ["--config-kind", "reuse", "--config-mode", "default-only"],
        [
            "--only",
            "newton_schulz_L16_b1024",
            "--config-kind",
            "mcast_1d",
        ],
        [
            "--only",
            "reference_square_4096",
            "--config-kind",
            "reuse",
        ],
    ],
)
def test_invalid_catalogue_selections_fail_before_device_or_output(
    monkeypatch, tmp_path, selection
):
    opened = []
    ttnn = SimpleNamespace(open_device=lambda **kwargs: opened.append(kwargs))
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    output = tmp_path / "results.json"

    with pytest.raises(SystemExit) as excinfo:
        run_matmul.main([*selection, "--out", str(output)])

    assert excinfo.value.code == 2
    assert opened == []
    assert not output.exists()


def test_main_serializes_selection_metadata_for_partial_runs(monkeypatch, tmp_path):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.float32 = "fp32"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(
        run_matmul,
        "run_shape",
        lambda *args, **kwargs: {"status": "failed", "error": "host-only stub"},
    )

    output = tmp_path / "results.json"
    assert (
        run_matmul.main(
            [
                "--only",
                "newton_schulz_L16_b1024",
                "--only",
                "newton_schulz_L32_b1024",
                "--config-kind",
                "batched_dram_sharded",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text())
    assert payload["selection"] == {
        "shape_filters": ["newton_schulz_L16_b1024", "newton_schulz_L32_b1024"],
        "program_config_kind_filters": ["batched_dram_sharded"],
        "custom_math_fidelity": ["HiFi4"],
        "input_memory": "l1",
        "r_memory": "l1",
        "x0_memory": "l1",
        "fuse_s": False,
        "batch_reads": False,
        "reload_r": False,
        "compare_reload_r": False,
        "double_buffer": True,
        "compare_double_buffer": False,
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": True,
    }
    assert len(payload["results"]) == 4
    assert all(
        result["program_config"]["kind"] == "batched_dram_sharded"
        for result in payload["results"]
    )


def test_l16_catalogue_rows_name_stock_best_and_custom_blocks(monkeypatch, tmp_path):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(
        run_matmul,
        "run_shape",
        lambda *args, **kwargs: {
            "status": "ok",
            "kind": "ttnn.matmul",
            "achieved_tflops": 1.0,
            "seconds_per_iteration": 1.0,
            "seconds_per_iteration_samples": [1.0],
        },
    )

    stock_output = tmp_path / "l16-stock.json"
    assert run_matmul.main(
        [
            "--only",
            "newton_schulz_L16_b8192",
            "--dtype",
            "bfloat16",
            "--memory",
            "l1",
            "--kind",
            "ttnn.matmul",
            "--config-mode",
            "default-only",
            "--out",
            str(stock_output),
        ]
    ) == 0
    stock_rows = json.loads(stock_output.read_text())["results"]
    assert [row["row"] for row in stock_rows] == ["stock_best"]

    def fake_custom(*args, **kwargs):
        return {
            "status": "ok",
            "kind": "custom_newton_schulz",
            "variant": kwargs["variant"],
            "math_fidelity": kwargs["math_fidelity"],
            "achieved_tflops": 1.0,
            "seconds_per_launch_p50": 1.0,
            "seconds_per_launch_p99": 1.0,
            "seconds_per_launch_p99_9": 1.0,
        }

    monkeypatch.setattr(run_matmul, "run_custom_newton_schulz", fake_custom)
    for matrix_block in (1, 4):
        output = tmp_path / f"l16-custom-{matrix_block}.json"
        assert run_matmul.main(
            [
                "--only",
                "newton_schulz_L16_b8192",
                "--dtype",
                "bfloat16",
                "--memory",
                "l1",
                "--kind",
                "custom_newton_schulz",
                "--matrix-block",
                str(matrix_block),
                "--out",
                str(output),
            ]
        ) == 0
        row = json.loads(output.read_text())["results"][0]
        assert row["row"] == f"custom_block{matrix_block}"
        assert row["program_config"]["matrix_block"] == matrix_block


def test_acceptance_catalogue_dispatches_both_shapes_and_all_required_rows(
    monkeypatch, tmp_path
):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.float32 = "fp32"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    stock_calls = []
    custom_calls = []

    def fake_stock(*args, **kwargs):
        stock_calls.append(kwargs)
        return {
            "status": "ok",
            "kind": "ttnn.matmul",
            "achieved_tflops": 1.0,
            "seconds_per_iteration": 1.0,
            "seconds_per_iteration_samples": [1.0] * kwargs["repeats"],
            "launches_measured": kwargs["iters"] * kwargs["repeats"],
        }

    def fake_custom(*args, **kwargs):
        shape = args[2]
        custom_calls.append(kwargs)
        return {
            "status": "ok",
            "kind": "custom_newton_schulz",
            "variant": kwargs["variant"],
            "math_fidelity": kwargs["math_fidelity"],
            "fuse_s": kwargs["fuse_s"],
            "matrix_block": kwargs["matrix_block"],
            "row": kwargs["row_name"],
            "flops_per_iteration": total_flops(shape) * COMPLEX_MATMULS_PER_INVERSE,
            "achieved_tflops": 1.0,
            "seconds_per_iteration": 1.0,
            "seconds_per_launch_samples": [1.0] * kwargs["repeats"],
            "seconds_per_launch_p50": 1.0,
            "seconds_per_launch_p99": 1.0,
            "seconds_per_launch_p99_9": 1.0,
            "launches_measured": kwargs["iters"] * kwargs["repeats"],
        }

    monkeypatch.setattr(run_matmul, "run_shape", fake_stock)
    monkeypatch.setattr(run_matmul, "run_custom_newton_schulz", fake_custom)
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"image": "sha256:pinned", "board": "p150a"}))
    power_trace = tmp_path / "power.csv"
    output = tmp_path / "acceptance.json"

    assert run_matmul.main(
        [
            "--acceptance-catalogue",
            "--env-json",
            str(environment),
            "--power-trace",
            str(power_trace),
            "--out",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text())
    assert payload["configuration_mode"] == "acceptance-catalogue"
    assert payload["environment"] == {
        "python": payload["environment"]["python"],
        "image": "sha256:pinned",
        "board": "p150a",
    }
    measurement = payload["measurement"]
    assert measurement["shape_names"] == [
        "newton_schulz_L32_b8192",
        "newton_schulz_L16_b8192",
    ]
    assert measurement["launches_per_row"] == 1000
    assert measurement["stock_best_row"] == "stock_best"
    assert measurement["power_trace"] == "power.csv"
    assert measurement["power_clock_provenance"]["columns"] == [
        "timestamp_utc",
        "power_w",
        "aiclk_mhz",
        "asic_temp_c",
    ]
    assert len(stock_calls) == 2
    assert all(call["iters"] == 1 and call["repeats"] == 1000 for call in stock_calls)
    assert len(custom_calls) == 16
    assert all(call["iters"] == 1 and call["repeats"] == 1000 for call in custom_calls)
    assert all(call["variant"] == "bf16-fp32state" for call in custom_calls)
    assert all(call["input_memory"] == call["r_memory"] == call["x0_memory"] == "l1"
               for call in custom_calls)
    custom_rows = [row for row in payload["results"] if row["kind"] == "custom_newton_schulz"]
    assert len(custom_rows) == 16
    assert {
        (row["shape"]["name"], row["matrix_block"], row["fuse_s"], row["math_fidelity"])
        for row in custom_rows
    } == {
        (shape, block, fuse_s, fidelity)
        for shape in ("newton_schulz_L32_b8192", "newton_schulz_L16_b8192")
        for block in (1, 4)
        for fuse_s in (False, True)
        for fidelity in ("HiFi3", "HiFi4")
    }
    assert all(row["launches_requested_per_row"] == 1000 for row in payload["results"])
    assert all(row["launches_measured"] == 1000 for row in payload["results"])
    l16 = next(
        row
        for row in custom_rows
        if row["shape"]["name"] == "newton_schulz_L16_b8192"
        and row["matrix_block"] == 4
        and row["fuse_s"]
        and row["math_fidelity"] == "HiFi3"
    )
    l16_shape = next(shape for shape in default_catalogue() if shape.name == l16["shape"]["name"])
    assert l16["flops_per_iteration"] == total_flops(l16_shape) * COMPLEX_MATMULS_PER_INVERSE


def test_successful_main_serializes_repeat_timing_samples(monkeypatch, tmp_path):
    """The host-only runner seam produces the same JSON shape as a device run."""
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)

    output = tmp_path / "results.json"
    assert (
        run_matmul.main(
            [
                "--only",
                "frontend_fir_taps64_w2",
                "--dtype",
                "bfloat16",
                "--memory",
                "dram",
                "--iters",
                "1",
                "--repeats",
                "2",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text())
    assert "host" not in payload["environment"]
    assert "profiling" not in payload
    assert len(payload["results"]) == 5
    assert [result["program_config"]["kind"] for result in payload["results"]] == [
        "default",
        "reuse",
        "mcast_1d",
        "mcast_1d",
        "mcast_2d",
    ]
    for result in payload["results"]:
        assert result["status"] == "ok"
        assert result["math_fidelity"] in {"HiFi2", "LoFi"}
        assert result["math_fidelity_source"]
        assert result["memory_placement"] == {
            name: {"buffer": "dram", "layout": "interleaved"}
            for name in ("input_a", "input_b", "output")
        }
        assert len(result["seconds_per_iteration_samples"]) == 2
        assert result["seconds_per_iteration"] == min(result["seconds_per_iteration_samples"])


def test_default_only_mode_omits_the_explicit_catalogue(monkeypatch, tmp_path):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)

    output = tmp_path / "results.json"
    assert (
        run_matmul.main(
            [
                "--only",
                "frontend_fir_taps64_w2",
                "--dtype",
                "bfloat16",
                "--memory",
                "dram",
                "--config-mode",
                "default-only",
                "--iters",
                "1",
                "--repeats",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text())
    assert payload["configuration_mode"] == "default-only"
    assert [result["program_config"]["kind"] for result in payload["results"]] == ["default"]


def test_a_program_config_failure_is_recorded_without_aborting_the_sweep(monkeypatch, tmp_path):
    ttnn = _StubTtnn()
    ttnn.bfloat16 = "bf16"
    ttnn.open_device = lambda device_id: object()
    ttnn.close_device = lambda device: None

    def reject_reuse(**kwargs):
        raise RuntimeError("program config rejected")

    ttnn.MatmulMultiCoreReuseProgramConfig = reject_reuse
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)

    output = tmp_path / "results.json"
    assert (
        run_matmul.main(
            [
                "--only",
                "frontend_fir_taps64_w2",
                "--dtype",
                "bfloat16",
                "--memory",
                "dram",
                "--iters",
                "1",
                "--repeats",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    results = json.loads(output.read_text())["results"]
    assert len(results) == 5
    assert results[1]["program_config"]["kind"] == "reuse"
    assert results[1]["status"] == "failed"
    assert "program config rejected" in results[1]["error"]
    assert results[-1]["status"] == "ok"


def test_custom_row_retains_launch_samples_and_percentiles(monkeypatch):
    class _FakeKernel:
        work_ranges = ((0, 2), (2, 1))
        output_memory = "dram"

        @classmethod
        def prepare(
            cls,
            ttnn,
            device,
            matrices,
            *,
            variant,
            math_fidelity,
            profile,
            fuse_s,
            batch_reads,
            **kwargs,
        ):
            assert variant == "bf16-fp32state"
            assert math_fidelity == "HiFi4"
            assert profile is False
            assert fuse_s is False
            assert batch_reads is False
            assert kwargs["matrix_block"] == 4
            assert kwargs["double_buffer"] is False
            assert matrices is not None
            return cls()

        def launch(self):
            return None

        def close(self):
            return None

    from enodia.tt.bench import newton_schulz_kernel

    monkeypatch.setattr(newton_schulz_kernel, "NewtonSchulzKernel", _FakeKernel)
    monkeypatch.setattr(
        newton_schulz_kernel,
        "benchmark_matrices",
        lambda batch, size, seed: object(),
    )
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
    ttnn = _StubTtnn()
    record = run_matmul.run_custom_newton_schulz(
        ttnn,
        device=object(),
        shape=shape,
        dtype_name="bfloat16",
        memory_name="l1",
        variant="bf16-fp32state",
        matrix_block=4,
        double_buffer=False,
        iters=2,
        repeats=2,
    )

    assert record["status"] == "ok"
    assert record["kind"] == "custom_newton_schulz"
    assert record["variant"] == "bf16-fp32state"
    assert record["fuse_s"] is False
    assert record["batch_reads"] is False
    assert record["output_memory"] == "dram"
    assert len(record["seconds_per_launch_samples"]) == 4
    assert record["seconds_per_launch_p50"] <= record["seconds_per_launch_p99"]
    assert record["seconds_per_launch_p99"] <= record["seconds_per_launch_p99_9"]
    assert record["flops_per_iteration"] == total_flops(shape) * COMPLEX_MATMULS_PER_INVERSE
    assert ttnn.sync_calls == 5  # one warm-up plus four timed launches


def test_custom_l16_dispatch_keeps_logical_flop_denominator(monkeypatch):
    class _FakeKernel:
        work_ranges = ((0, 40),)
        output_memory = "dram"
        seen_matrices = None
        seen_kwargs = None

        @classmethod
        def prepare(cls, ttnn, device, matrices, **kwargs):
            cls.seen_matrices = matrices
            cls.seen_kwargs = kwargs
            return cls()

        def launch(self):
            return None

        def close(self):
            return None

    from enodia.tt.bench import newton_schulz_kernel

    monkeypatch.setattr(newton_schulz_kernel, "NewtonSchulzKernel", _FakeKernel)
    monkeypatch.setattr(
        newton_schulz_kernel,
        "benchmark_matrices",
        lambda batch, size, seed: SimpleNamespace(shape=(batch, size, size)),
    )
    shape = MatmulShape(
        name="newton_schulz_L16_b8192",
        batch=8192,
        m=16,
        k=16,
        n=16,
        real_matmuls=4,
        family="newton_schulz",
        note="",
    )
    record = run_matmul.run_custom_newton_schulz(
        _StubTtnn(),
        device=object(),
        shape=shape,
        dtype_name="bfloat16",
        memory_name="l1",
        variant="bf16-fp32state",
        matrix_block=4,
        iters=1,
        repeats=1,
    )

    assert record["status"] == "ok"
    assert _FakeKernel.seen_matrices.shape == (8192, 16, 16)
    assert _FakeKernel.seen_kwargs["matrix_block"] == 4
    assert record["matrix_block"] == 4
    assert record["physical_tile_count"] == 4096
    assert record["packing"] == "diagonal_pairs_32x32"
    assert record["flops_per_iteration"] == total_flops(shape) * COMPLEX_MATMULS_PER_INVERSE


def test_block_double_buffer_record_flops_match_current_inverse_accounting():
    record_path = (
        Path(__file__).parents[1]
        / "docs/measurements/2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32.json"
    )
    record = json.loads(record_path.read_text())
    shapes = {shape.name: shape for shape in default_catalogue()}

    assert COMPLEX_MATMULS_PER_INVERSE == 24
    for size in (16, 32):
        shape = shapes[f"newton_schulz_L{size}_b8192"]
        expected = inverse_flops(shape)
        assert expected == COMPLEX_MATMULS_PER_INVERSE * total_flops(shape)
        assert record["performance"]["flops_per_launch"][f"L{size}"] == expected


def test_custom_row_rejects_non_target_shapes_without_opening_kernel():
    shape = _shape(4)
    record = run_matmul.run_custom_newton_schulz(
        _StubTtnn(),
        device=object(),
        shape=shape,
        dtype_name="bfloat16",
        memory_name="l1",
        variant="bf16",
        iters=1,
        repeats=1,
    )

    assert record["status"] == "failed"
    assert record["kind"] == "custom_newton_schulz"


def test_custom_block8_l1_preflight_rejects_before_kernel_prepare():
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
    ttnn = SimpleNamespace(bfloat16="bf16", float32="fp32")

    record = run_matmul.run_custom_newton_schulz(
        ttnn,
        device=object(),
        shape=shape,
        dtype_name="bfloat16",
        memory_name="l1",
        variant="bf16-fp32state",
        fuse_s=True,
        matrix_block=8,
        double_buffer=False,
        iters=1,
        repeats=1,
    )

    assert record["status"] == "failed"
    assert "L1 preflight failed" in record["error"]
    assert "total CB bytes=423936" in record["error"]
    assert "CB_STATE_REAL=65536 bytes" in record["error"]


def test_custom_block8_per_input_placement_dispatches_with_passing_preflight(monkeypatch):
    class _FakeKernel:
        work_ranges = ((0, 80),)
        output_memory = "dram"
        seen_kwargs = None

        @classmethod
        def prepare(cls, ttnn, device, matrices, **kwargs):
            cls.seen_kwargs = kwargs
            return cls()

        def launch(self):
            return None

        def close(self):
            return None

    from enodia.tt.bench import newton_schulz_kernel

    monkeypatch.setattr(newton_schulz_kernel, "NewtonSchulzKernel", _FakeKernel)
    monkeypatch.setattr(
        newton_schulz_kernel,
        "benchmark_matrices",
        lambda batch, size, seed: object(),
    )
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
        _StubTtnn(),
        device=object(),
        shape=shape,
        dtype_name="bfloat16",
        memory_name="l1",
        variant="bf16-fp32state",
        fuse_s=True,
        matrix_block=8,
        r_memory="l1",
        x0_memory="dram",
        double_buffer=False,
        iters=1,
        repeats=1,
    )

    assert record["status"] == "ok"
    assert record["r_memory"] == "l1"
    assert record["x0_memory"] == "dram"
    assert record["l1_preflight_bytes"] == 1_032_960
    assert _FakeKernel.seen_kwargs["r_memory"] == "l1"
    assert _FakeKernel.seen_kwargs["x0_memory"] == "dram"


def test_custom_block4_l1_preflight_accepts_the_ledger_minimum():
    from enodia.tt.bench import newton_schulz_kernel

    ttnn = SimpleNamespace(bfloat16="bf16", float32="fp32")
    total = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="l1",
        matrix_block=4,
        double_buffer=False,
        variant="bf16-fp32state",
    )

    assert total == 1_393_408
    assert total <= newton_schulz_kernel._L1_TOTAL_BUDGET_BYTES


def test_efficiency_is_omitted_without_a_peak(tmp_path):
    """Quoting an efficiency against an unstated denominator is worse than
    quoting none, so the field simply is not there."""
    record = {"achieved_tflops": 10.0}
    assert run_matmul.with_efficiency(record, peak_tflops=None) == record
    assert run_matmul.with_efficiency(dict(record), peak_tflops=100.0)["efficiency"] == 0.1
    json.dumps(record)  # the record stays serializable
