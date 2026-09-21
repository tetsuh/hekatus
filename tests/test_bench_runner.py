"""The runner's accounting must match what it actually executes.

A benchmark that divides the FLOPs of four matmuls by the time of one
reports a number four times too large, and does it silently. These tests
pin the execution count to the accounting, using a stub in place of the
toolchain so they run anywhere.
"""

import json
import sys
from types import SimpleNamespace

import pytest

from enodia.tt.bench import run_matmul
from enodia.tt.bench.configs import configuration_catalogue
from enodia.tt.bench.shapes import MatmulShape


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


def test_efficiency_is_omitted_without_a_peak(tmp_path):
    """Quoting an efficiency against an unstated denominator is worse than
    quoting none, so the field simply is not there."""
    record = {"achieved_tflops": 10.0}
    assert run_matmul.with_efficiency(record, peak_tflops=None) == record
    assert run_matmul.with_efficiency(dict(record), peak_tflops=100.0)["efficiency"] == 0.1
    json.dumps(record)  # the record stays serializable
