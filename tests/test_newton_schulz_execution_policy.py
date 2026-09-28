import json
import os
import sys
from queue import Empty
from types import SimpleNamespace

import numpy as np
import pytest

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


class _FakeQueue:
    def __init__(self):
        self.values = []
        self.closed = False
        self.thread_joined = False

    def put(self, value):
        self.values.append(value)

    def get_nowait(self):
        if not self.values:
            raise Empty
        return self.values.pop(0)

    def close(self):
        self.closed = True

    def join_thread(self):
        self.thread_joined = True


class _FakeProcess:
    def __init__(self, *, record=None, alive=False, terminate_stops=True):
        self.pid = 4321
        self.record = record
        self.alive = alive
        self.terminate_stops = terminate_stops
        self.queue = None
        self.join_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.started = False
        self.closed = False

    def start(self):
        self.started = True
        if self.record is not None:
            self.queue.put(self.record)

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        if timeout is None or self.terminate_calls and self.terminate_stops:
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1
        if self.terminate_stops:
            self.alive = False

    def kill(self):
        self.kill_calls += 1
        self.alive = False

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self, *, record=None, alive=False, terminate_stops=True):
        self.queue = _FakeQueue()
        self.process = _FakeProcess(
            record=record, alive=alive, terminate_stops=terminate_stops
        )
        self.target = None
        self.args = None

    def Queue(self):
        return self.queue

    def Process(self, target, args):
        self.target = target
        self.args = args
        self.process.queue = self.queue
        return self.process


def _install_fake_process_context(monkeypatch, context):
    methods = []

    def get_context(method):
        methods.append(method)
        return context

    monkeypatch.setattr(
        bringup,
        "multiprocessing",
        SimpleNamespace(get_context=get_context),
        raising=False,
    )
    return methods


def test_removed_child_flag_is_rejected_as_unknown():
    parser = bringup._argument_parser()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["--stage", "1", "--_numerical-child"])
    assert raised.value.code == 2


def test_direct_numerical_invocation_uses_spawn_supervisor_before_runtime(monkeypatch, capsys):
    context = _FakeContext(record={"stage": 1, "status": "pass"})
    methods = _install_fake_process_context(monkeypatch, context)

    def fail_open(**kwargs):
        raise AssertionError("the parent must not open a device")

    monkeypatch.setitem(sys.modules, "ttnn", SimpleNamespace(open_device=fail_open))
    monkeypatch.setattr(sys, "argv", ["newton_schulz_bringup.py", "--stage", "1"])
    monkeypatch.delenv("TT_METAL_WATCHER", raising=False)

    assert bringup.main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "pass"
    assert methods == ["spawn"]
    assert context.process.started is True
    assert context.process.join_calls == [60]
    assert context.target is bringup._numerical_stage_process_entry
    assert context.args[:3] == (1, 0, bringup.DEFAULT_NUMERICAL_EXECUTION_POLICY)
    assert context.queue.closed is True
    assert context.queue.thread_joined is True
    assert context.process.closed is True
    assert "TT_METAL_WATCHER" not in os.environ


def test_supervisor_passes_effective_safety_policy_to_child_and_record(monkeypatch, capsys):
    context = _FakeContext(record={"stage": 1, "status": "pass"})
    _install_fake_process_context(monkeypatch, context)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "newton_schulz_bringup.py",
            "--stage",
            "1",
            "--no-watcher",
            "--timeout",
            "17",
        ],
    )

    assert bringup.main() == 0
    record = json.loads(capsys.readouterr().out)
    policy = context.args[2]
    assert policy.watcher == bringup.WatcherPolicy(enabled=False, value=None)
    assert policy.external_timeout_s == 17
    assert context.process.join_calls == [17]
    assert record["watcher"] == {"enabled": False, "value": None}
    assert record["external_timeout_s"] == 17
    assert record["execution_policy"] == {
        "watcher": {"enabled": False, "value": None},
        "external_timeout_s": 17,
        "external_timeout_command": ["timeout", "17s"],
        "external_timeout_enforced_by": "caller",
    }


def test_timeout_terminates_then_kills_remaining_process_and_records_failure(monkeypatch, capsys):
    context = _FakeContext(alive=True, terminate_stops=False)
    _install_fake_process_context(monkeypatch, context)
    monkeypatch.setattr(sys, "argv", ["newton_schulz_bringup.py", "--stage", "1"])

    assert bringup.main() == 1
    record = json.loads(capsys.readouterr().out)
    process = context.process
    assert record["status"] == "fail"
    assert record["error"]["code"] == "timeout"
    assert record["error"]["details"]["timeout_s"] == 60
    assert record["error"]["details"]["process_id"] == process.pid
    assert record["external_timeout_s"] == 60
    assert record["timeout"] is True
    assert record["process_terminated"] is True
    assert record["process_killed"] is True
    assert process.join_calls == [60, 1, None]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert context.queue.closed is True
    assert context.queue.thread_joined is True
    assert process.closed is True


def test_explicit_no_timeout_keeps_spawn_supervisor_and_records_no_cap(monkeypatch, capsys):
    context = _FakeContext(record={"stage": 1, "status": "pass"})
    methods = _install_fake_process_context(monkeypatch, context)
    monkeypatch.setattr(
        sys, "argv", ["newton_schulz_bringup.py", "--stage", "1", "--no-timeout"]
    )

    assert bringup.main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "pass"
    assert methods == ["spawn"]
    assert context.process.join_calls == [None]
    assert record["external_timeout_s"] is None
    assert record["external_timeout_command"] is None
    assert record["external_timeout_enforced_by"] is None
    assert record["execution_policy"]["external_timeout_s"] is None
    assert context.process.terminate_calls == 0
    assert context.process.kill_calls == 0
    assert context.process.closed is True


def test_child_exception_is_transferred_as_a_failed_record(monkeypatch):
    queue = _FakeQueue()

    class FakeTTNN:
        def open_device(self, **kwargs):
            raise RuntimeError("dispatch unavailable")

    monkeypatch.setitem(sys.modules, "ttnn", FakeTTNN())
    bringup._numerical_stage_process_entry(
        1,
        0,
        bringup.DEFAULT_NUMERICAL_EXECUTION_POLICY,
        queue,
    )

    record = queue.get_nowait()
    assert record["status"] == "fail"
    assert record["error"]["code"] == "child_exception"
    assert record["watcher"] == {"enabled": True, "value": "1"}


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
