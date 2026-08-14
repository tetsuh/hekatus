"""The runner's accounting must match what it actually executes.

A benchmark that divides the FLOPs of four matmuls by the time of one
reports a number four times too large, and does it silently. These tests
pin the execution count to the accounting, using a stub in place of the
toolchain so they run anywhere.
"""

import json
import sys

import pytest

from enodia.tt.bench import run_matmul
from enodia.tt.bench.shapes import MatmulShape


class _StubTensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.deallocated = False


class _StubTtnn:
    """Records what the runner asked the toolchain to do."""

    TILE_LAYOUT = "tile"
    DRAM_MEMORY_CONFIG = "dram"
    L1_MEMORY_CONFIG = "l1"

    def __init__(self, fail_on: str | None = None) -> None:
        self.matmul_calls = 0
        self.sync_calls = 0
        self.deallocated: list[str] = []
        self.fail_on = fail_on

    def rand(self, shape, **kwargs):
        if self.fail_on == "rand":
            raise RuntimeError("out of memory")
        return _StubTensor(f"rand{shape}")

    def ones(self, shape, **kwargs):
        if self.fail_on in ("rand", "ones"):
            raise RuntimeError("out of memory")
        return _StubTensor(f"ones{shape}")

    def zeros(self, shape, **kwargs):
        if self.fail_on in ("rand", "ones", "zeros"):
            raise RuntimeError("out of memory")
        return _StubTensor(f"zeros{shape}")

    def matmul(self, a, b):
        self.matmul_calls += 1
        return _StubTensor("out")

    def synchronize_device(self, device):
        self.sync_calls += 1

    def deallocate(self, tensor):
        tensor.deallocated = True
        self.deallocated.append(tensor.name)


def _shape(real_matmuls: int) -> MatmulShape:
    return MatmulShape(
        name="probe",
        batch=2,
        m=4,
        k=4,
        n=4,
        real_matmuls=real_matmuls,
        family="newton_schulz",
        note="",
    )


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
    assert run_matmul.main(
        ["--only", "frontend_fir_taps64_w2", "--dtype", "bfloat16", "--memory", "dram",
         "--iters", "1", "--repeats", "2", "--out", str(output)]
    ) == 0

    payload = json.loads(output.read_text())
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["status"] == "ok"
    assert len(result["seconds_per_iteration_samples"]) == 2
    assert result["seconds_per_iteration"] == min(result["seconds_per_iteration_samples"])


def test_efficiency_is_omitted_without_a_peak(tmp_path):
    """Quoting an efficiency against an unstated denominator is worse than
    quoting none, so the field simply is not there."""
    record = {"achieved_tflops": 10.0}
    assert run_matmul.with_efficiency(record, peak_tflops=None) == record
    assert run_matmul.with_efficiency(dict(record), peak_tflops=100.0)["efficiency"] == 0.1
    json.dumps(record)  # the record stays serializable
