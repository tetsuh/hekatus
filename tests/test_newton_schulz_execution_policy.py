import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from tools import newton_schulz_bringup as bringup


class _FakeRuntime:
    def __init__(self, expected_watcher=None):
        self.expected_watcher = expected_watcher
        self.generic_calls = []
        self.deallocated = []

    def generic_op(self, tensors, program):
        self.generic_calls.append((tensors, program))
        if self.expected_watcher is not None:
            assert os.environ["TT_METAL_WATCHER"] == self.expected_watcher

    def synchronize_device(self, device):
        return None

    def deallocate(self, tensor):
        self.deallocated.append(tensor)


def test_default_policy_is_watcher_and_external_60_seconds(monkeypatch):
    monkeypatch.delenv("TT_METAL_WATCHER", raising=False)

    args = bringup._argument_parser().parse_args(["--stage", "1", "--device-id", "0"])
    policy = bringup.resolve_numerical_execution_policy(
        watcher=args.watcher,
        timeout_s=args.timeout_s,
        environment={},
    )

    assert policy == bringup.DEFAULT_NUMERICAL_EXECUTION_POLICY
    assert policy.as_record() == {
        "watcher": {"enabled": True, "value": "1"},
        "external_timeout_s": 60,
        "external_timeout_command": ["timeout", "60s"],
        "external_timeout_enforced_by": "caller",
    }


def test_no_watcher_is_explicit_and_restores_the_callers_environment(monkeypatch):
    monkeypatch.setenv("TT_METAL_WATCHER", "caller-value")
    args = bringup._argument_parser().parse_args(["--stage", "1", "--no-watcher"])
    policy = bringup.resolve_numerical_execution_policy(
        watcher=args.watcher,
        timeout_s=args.timeout_s,
    )

    assert policy.watcher.enabled is False
    assert policy.watcher.value is None
    with bringup.watcher_environment(policy):
        assert "TT_METAL_WATCHER" not in os.environ
    assert os.environ["TT_METAL_WATCHER"] == "caller-value"


def test_existing_watcher_environment_is_an_explicit_override(monkeypatch):
    monkeypatch.setenv("TT_METAL_WATCHER", "0")

    policy = bringup.resolve_numerical_execution_policy()

    assert policy.watcher.enabled is False
    assert policy.watcher.value == "0"
    with bringup.watcher_environment(policy):
        assert os.environ["TT_METAL_WATCHER"] == "0"
    assert os.environ["TT_METAL_WATCHER"] == "0"


def test_timeout_opt_out_and_override_are_machine_readable():
    no_timeout_args = bringup._argument_parser().parse_args(["--stage", "1", "--timeout", "0"])
    no_timeout = bringup.resolve_numerical_execution_policy(timeout_s=no_timeout_args.timeout_s)
    assert no_timeout.external_timeout_s is None
    assert no_timeout.external_timeout_command is None
    assert no_timeout.as_record()["external_timeout_enforced_by"] is None

    override_args = bringup._argument_parser().parse_args(["--stage", "1", "--timeout", "17"])
    override = bringup.resolve_numerical_execution_policy(timeout_s=override_args.timeout_s)
    assert override.external_timeout_s == 17
    assert override.external_timeout_command == ("timeout", "17s")
    assert override.as_record()["external_timeout_command"] == ["timeout", "17s"]

    no_timeout_flag = bringup._argument_parser().parse_args(["--stage", "1", "--no-timeout"])
    assert no_timeout_flag.timeout_s == 0


def test_success_and_failure_records_carry_the_same_policy(monkeypatch):
    runtime = _FakeRuntime(expected_watcher="1")
    stage = bringup.STAGES[1]
    expected = np.zeros((stage.batch, bringup.TILE, bringup.TILE), dtype=np.complex64)
    monkeypatch.setattr(
        bringup,
        "_prepare",
        lambda ttnn, device, selected: ("program", ["input"], ["output"], expected),
    )
    monkeypatch.setattr(
        bringup,
        "_download",
        lambda ttnn, tensor: np.zeros((stage.batch, 1, bringup.TILE, bringup.TILE)),
    )
    policy = bringup.resolve_numerical_execution_policy(watcher=True, timeout_s=17)

    success = bringup.run_stage(runtime, object(), stage, execution_policy=policy)
    failure = bringup._stage_failure_record(1, policy=policy, error="dispatch failed")

    for record in (success, failure):
        assert record["watcher"] == {"enabled": True, "value": "1"}
        assert record["external_timeout_s"] == 17
        assert record["external_timeout_command"] == ["timeout", "17s"]
        assert record["execution_policy"]["external_timeout_s"] == 17
    assert success["status"] == "pass"
    assert success["numerical_error"] == 0.0
    assert failure["status"] == "fail"
    assert failure["numerical_error"] is None


def test_main_failure_record_carries_the_effective_policy_without_opening_hardware(
    monkeypatch, capsys
):
    def fail_open(**kwargs):
        assert os.environ["TT_METAL_WATCHER"] == "1"
        raise RuntimeError("fake runtime unavailable")

    child = _FakeChildProcess(record={"stage": 1, "status": "fail"})
    monkeypatch.setattr(bringup.subprocess, "Popen", lambda command, **kwargs: child)
    monkeypatch.setitem(sys.modules, "ttnn", SimpleNamespace(open_device=fail_open))
    monkeypatch.setattr(sys, "argv", ["newton_schulz_bringup.py", "--stage", "1", "--device-id", "0"])
    monkeypatch.delenv("TT_METAL_WATCHER", raising=False)

    assert bringup.main() == 1
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "fail"
    assert record["watcher"] == {"enabled": True, "value": "1"}
    assert record["external_timeout_s"] == 60
    assert record["external_timeout_command"] == ["timeout", "60s"]
    assert "TT_METAL_WATCHER" not in os.environ


class _FakeChildProcess:
    def __init__(self, *, record=None, timeout=False):
        self.pid = 4321
        self.returncode = 0
        self.record = record or {"status": "pass"}
        self.timeout = timeout
        self.communicate_timeouts = []

    def communicate(self, timeout=None):
        self.communicate_timeouts.append(timeout)
        if self.timeout:
            self.timeout = False
            raise subprocess.TimeoutExpired(["newton_schulz_bringup.py"], timeout)
        return json.dumps(self.record), ""

    def wait(self, timeout=None):
        return self.returncode


def test_default_direct_stage_supervises_before_fake_device_open(monkeypatch, capsys):
    events = []
    child = _FakeChildProcess(record={"stage": 1, "status": "pass"})

    def fake_popen(command, **kwargs):
        events.append(("spawn", command, kwargs))
        return child

    def fail_open(**kwargs):
        events.append(("open", kwargs))
        raise AssertionError("the parent must not open a device")

    monkeypatch.setattr(bringup.subprocess, "Popen", fake_popen)
    monkeypatch.setitem(sys.modules, "ttnn", SimpleNamespace(open_device=fail_open))
    monkeypatch.setattr(sys, "argv", ["newton_schulz_bringup.py", "--stage", "1"])
    monkeypatch.delenv("TT_METAL_WATCHER", raising=False)

    assert bringup.main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "pass"
    assert events[0][0] == "spawn"
    assert not any(event[0] == "open" for event in events)
    assert events[0][2]["start_new_session"] is True
    assert events[0][2]["shell"] is False
    assert isinstance(events[0][1], list)
    assert child.communicate_timeouts == [60]
    assert "--_numerical-child" in events[0][1]


def test_timeout_terminates_the_child_process_group_and_records_failure(monkeypatch, capsys):
    child = _FakeChildProcess(timeout=True)
    killed = []

    monkeypatch.setattr(bringup.subprocess, "Popen", lambda command, **kwargs: child)
    monkeypatch.setattr(
        bringup.os,
        "killpg",
        lambda process_group_id, signum: killed.append((process_group_id, signum)),
    )
    monkeypatch.setattr(sys, "argv", ["newton_schulz_bringup.py", "--stage", "1", "--timeout", "1"])

    assert bringup.main() == 1
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "fail"
    assert record["error"]["code"] == "timeout"
    assert record["error"]["details"]["timeout_s"] == 1
    assert record["error"]["details"]["process_group_id"] == child.pid
    assert record["external_timeout_s"] == 1
    assert record["timeout"] is True
    assert record["process_group_terminated"] is True
    assert (child.pid, signal.SIGTERM) in killed
    assert (child.pid, signal.SIGKILL) in killed
    assert child.communicate_timeouts == [1, None]


def test_explicit_no_timeout_uses_fake_device_without_supervisor(monkeypatch, capsys):
    events = []

    class FakeTTNN:
        def open_device(self, **kwargs):
            events.append(("open", kwargs))
            return object()

        def close_device(self, device):
            events.append(("close", device))

    def fake_run_stage(ttnn, device, stage, *, execution_policy):
        events.append(("dispatch", stage.number, execution_policy.external_timeout_s))
        return {"stage": stage.number, "status": "pass"}

    def fail_popen(*args, **kwargs):
        raise AssertionError("explicit no-timeout must not spawn a supervisor")

    monkeypatch.setattr(bringup.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(bringup, "run_stage", fake_run_stage)
    monkeypatch.setitem(sys.modules, "ttnn", FakeTTNN())
    monkeypatch.setattr(
        sys, "argv", ["newton_schulz_bringup.py", "--stage", "1", "--no-timeout"]
    )

    assert bringup.main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "pass"
    assert events[0][0] == "open"
    assert events[1] == ("dispatch", 1, None)
    assert events[2][0] == "close"


def test_construction_and_build_only_records_keep_their_existing_controls(tmp_path, monkeypatch):
    construction = bringup.construction_probe_record("P8")
    assert construction["external_timeout_s"] == 60
    assert "watcher" not in construction
    assert "execution_policy" not in construction

    monkeypatch.setattr(
        bringup,
        "_prepare_stage_program",
        lambda ttnn, device, stage, input_values: ("program", ["input"], ["output"]),
    )
    record = bringup.run_build_only_stage(_FakeRuntime(), object(), 62, tmp_path / "cache")
    assert record["status"] == "pass"
    assert "watcher" not in record
    assert "execution_policy" not in record
    assert "external_timeout_s" not in record
