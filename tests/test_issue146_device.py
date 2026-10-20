"""Board-free Issue #146 device-runner preflight and grouped wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from enodia.tt.bench import load_generators, newton_schulz_kernel
from tests.test_issue146_runner import PLAN, _environment
from tools import issue146, issue146_device


def test_issue146_grouped_preparation_passes_plan_repetitions(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeKernel:
        @classmethod
        def prepare(cls, _ttnn, _device, matrices, **kwargs):
            captured.append({"batch": matrices.shape[0], **kwargs})
            return object()

    monkeypatch.setattr(newton_schulz_kernel, "NewtonSchulzKernel", FakeKernel)
    config = issue146.configuration(issue146.load_plan(PLAN), "C018")
    load_generators.prepare_newton_schulz_grouped_load(
        object(), object(), core_plan=config.core_plan, profile=False, outer_repetitions=16
    )
    assert captured[0]["outer_repetitions"] == 16
    assert captured[0]["iterations"] == 12
    assert captured[0]["core_coordinates"] == config.core_plan.worker_cores


def test_issue146_device_group_wiring_enqueues_eight_then_syncs_once(
    tmp_path: Path, monkeypatch
) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    environment_path = tmp_path / "env.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    result_path = tmp_path / "runner-result.json"
    group_path = tmp_path / "device-groups.json"
    events: list[str] = []
    clock = iter((0, 40_000_000_000))

    class Prepared:
        def launch(self):
            events.append("launch")

        def close(self):
            events.append("close")

    class FakeTtnn:
        def open_device(self, *, device_id, num_command_queues):
            assert (device_id, num_command_queues) == (0, 2)
            events.append("open")
            return object()

        def synchronize_device(self, _device):
            events.append("sync")

        def close_device(self, _device):
            events.append("device-close")

    monkeypatch.setitem(sys.modules, "ttnn", FakeTtnn())
    monkeypatch.setattr(
        issue146_device,
        "prepare_newton_schulz_grouped_load",
        lambda *args, **kwargs: (events.append(f"prepare-{kwargs['outer_repetitions']}"), Prepared())[1],
    )
    monkeypatch.setattr(issue146_device.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setenv("HEKATUS_TT_RESULT_PATH", str(result_path))
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "wrapper-fresh-run-id")
    args = issue146_device._parser().parse_args(
        [
            "--env-json", str(environment_path), "--groups", str(group_path),
            "--configuration", "C018", "--mode", "smoke", "--device-id", "0",
            "--load-core-count", "18", "--load-total-blocks", "171",
            "--load-matrix-block", "8", "--outer-repetitions", "16",
            "--launches-per-sync", "8", "--target-seconds", "40",
            "--outer-timeout-seconds", "90", "--plan-hash", issue146._plan_hash(plan),
        ]
    )
    assert issue146_device.run(args) == 0
    assert events == ["open", "prepare-16", *(["launch"] * 8), "sync", "close", "device-close"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["launch_count"] == 8
    assert json.loads(group_path.read_text(encoding="utf-8")) == [
        {"start_monotonic_ns": 0, "end_monotonic_ns": 40_000_000_000}
    ]


def test_issue146_device_rejects_bad_identity_before_ttnn_import(tmp_path: Path, monkeypatch) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    environment["device_node"] = "/dev/tenstorrent/0"
    environment_path = tmp_path / "env.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    monkeypatch.setenv("HEKATUS_TT_RESULT_PATH", str(tmp_path / "runner-result.json"))
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "wrapper-fresh-run-id")
    imported = False
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        nonlocal imported
        if name == "ttnn":
            imported = True
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    args = issue146_device._parser().parse_args(
        [
            "--env-json", str(environment_path), "--configuration", "C018", "--mode", "smoke",
            "--load-core-count", "18", "--load-total-blocks", "171", "--load-matrix-block", "8",
            "--outer-repetitions", "16", "--launches-per-sync", "8", "--target-seconds", "40",
            "--outer-timeout-seconds", "90", "--plan-hash", issue146._plan_hash(plan),
        ]
    )
    with pytest.raises(issue146.Issue146Error, match="device_node"):
        issue146_device.run(args)
    assert imported is False
