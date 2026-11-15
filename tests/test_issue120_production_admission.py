"""Board-free session, control and execution-admission counterexamples."""
from __future__ import annotations

import copy
import itertools
import json
import subprocess

import pytest

from enodia.tt.bench.issue120_production import plan, protocol, records, runner, runtime, sessions


def test_wrapper_command_has_literal_caps_image_device_and_watcher(monkeypatch):
    monkeypatch.setenv("TT_METAL_WATCHER", "1")
    control = plan.run_spec("production-v1-control-500000")
    command, environment = runner.wrapper_command(control, "unit-session")
    assert environment["HEKATUS_TT_CONTAINER_TIMEOUT_S"] == "600"
    assert environment["HEKATUS_TT_DEVICE_NODE"] == "/dev/tenstorrent/0"
    assert environment["HEKATUS_TT_IMAGE"] == runtime.PINNED_IMAGE
    assert "TT_METAL_WATCHER" not in environment
    assert "--production-approved" in command
    first = plan.run_spec("production-v1-control-watch")
    _, env = runner.wrapper_command(first, "unit-session")
    assert env["TT_METAL_WATCHER"] == "1"
    assert env["HEKATUS_TT_CONTAINER_TIMEOUT_S"] == "60"


def test_missing_stage_receipt_never_admits_production():
    assert len(sessions.admitted_specs("B")) == 2
    with pytest.raises(ValueError, match="admission"):
        sessions.admitted_specs("P")


@pytest.mark.parametrize("failure", ["execution", "warning", "watcher"])
def test_failed_startup_cannot_admit_calibration(monkeypatch, failure):
    rows = [{"plan_identity": s.identity, "configuration": plan.program_configuration(s),
             "resident_integrity_passed": True, "condition": s.condition, "phase": s.phase,
             "execution_eligible": True, "warning_check": {"passed": True},
             "coverage_analysis": {"coverage": "not_applicable" if s.condition == "control" else "covered"},
             "watcher": {"requested": True, "fresh": True, "scan_passed": True}}
            for s in plan.RUNS[:2]]
    if failure == "execution":
        rows[0]["execution_eligible"] = False
    elif failure == "warning":
        rows[1]["warning_check"]["passed"] = False
    else:
        rows[1]["watcher"]["scan_passed"] = False
    monkeypatch.setattr(sessions, "_committed_admission", lambda *a: {"operator_watcher_reviewed": True})
    monkeypatch.setattr(sessions, "_records_from_receipt", lambda _: rows)
    with pytest.raises(ValueError, match="Watcher"):
        sessions.admitted_specs("C")
    for row in rows:
        row["execution_eligible"] = row["warning_check"]["passed"] = row["watcher"]["scan_passed"] = True
    assert sessions.admitted_specs("C") == list(plan.RUNS[2:4])


def test_session_is_exclusive_boot_bound_and_program_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS", tmp_path)
    monkeypatch.setattr(runtime, "program_identity", lambda: {"native_sha256": "a"*64})
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "b"*40))
    data = sessions.create_session(list(plan.RUNS[:2]))
    got, _ = sessions.read_session(data["session_id"])
    assert got == data
    assert got["runs"][plan.RUNS[0].identity] == got["runs"][plan.RUNS[1].identity]
    with pytest.raises(ValueError):
        sessions.session_path("../escape")
    monkeypatch.setattr(sessions, "boot_fingerprint", lambda: "other-boot")
    with pytest.raises(ValueError):
        sessions.read_session(data["session_id"])


def test_max_sum_and_torn_tuple_enumeration():
    n = protocol.UINT32_MAX - 500002
    protocol.admit_domain(frame_count=500000, repeat_count=n)
    assert protocol.encode(n, 500002)[1] == protocol.UINT32_MAX
    with pytest.raises(ValueError):
        protocol.admit_domain(frame_count=500000, repeat_count=n + 1)
    entries = [(1, 0), (2, 1), (3, 2)]
    index = protocol.TransitionIndex(entries, frame_count=3, repeat_count=3)
    actual = {protocol.encode(s, c) for s, c in entries}
    for sequence, summed in itertools.product((1, 2, 3), (1, 3, 5)):
        words = [protocol.word(sequence, summed, i) for i in range(1024)]
        if (sequence, summed) in actual:
            assert protocol.observe(words, index, frame_count=3, repeat_count=3)[0] == sequence
        else:
            with pytest.raises(ValueError):
                protocol.observe(words, index, frame_count=3, repeat_count=3)


def comparable(condition):
    loaded = condition == "loaded"
    offset = 20 if loaded else 0
    host = {"start_ns": 2+offset, "end_ns": 7+offset} if loaded else None
    cpu = dict(host, operations=327680, reads=10, transfers=10) if loaded else {"operations": 0, "reads": 0, "transfers": 0}
    config = {"frame_count": 2, "repeat_count": 10, "command_queues": 2}
    return {"schema": records.SCHEMA, "condition": condition, "session_id": "s",
            "session_manifest_sha256": "a"*64, "host_boot_id_sha256": "b"*64,
            "configuration": config, "configuration_sha256": plan.configuration_digest(config),
            "program_identity": {"native_sha256": "c"*64}, "environment": {"serial": "board0"},
            "attempt_identity": condition, "timing_evidence": True,
            "native": {"frame_count": 2, "repeat_count": 10, "host_interval": host, "host_cpu_work": cpu,
                       "host_scheduling": {k: v+offset for k, v in {
                           "enqueue_begin_ns": 1, "enqueue_return_ns": 3, "load_join_completed_ns": 8,
                           "completion_observer_joined_ns": 9, "finish_begin_ns": 10,
                           "finish_end_ns": 11, "drain_end_ns": 12}.items()},
                       "resident_host_interval": {"start_ns": 1+offset, "end_ns": 11+offset,
                                                  "scope": "enqueue_to_post_join_finish_return_upper_bound"},
                       "program_host_interval": {"start_ns": 1+offset, "end_ns": 9+offset,
                                                 "start_utc_ns": 101+offset, "end_utc_ns": 109+offset,
                                                 "scope": "enqueue_to_l1_terminal_observed_upper_bound"}},
            "metrics": {"frame_intervals": {"max_ticks": 10+offset, "percentiles": {
                k: {"status": "insufficient"} for k in ("p50", "p99", "p99_9", "p99_99")}}}}


def test_unlike_control_cannot_produce_a_difference():
    control, loaded = comparable("control"), comparable("loaded")
    difference = records.compare_records(control, loaded)
    assert difference["loaded_minus_control_ticks"]["max_ticks"] == 20
    assert difference["loaded_minus_control_ticks"]["p50"] is None
    for key in ("session_id", "session_manifest_sha256", "host_boot_id_sha256", "program_identity", "environment"):
        bad = copy.deepcopy(loaded)
        bad[key] = "different"
        with pytest.raises(ValueError):
            records.compare_records(control, bad)
    bad = copy.deepcopy(loaded)
    bad["configuration"]["command_queues"] = 1
    bad["configuration_sha256"] = plan.configuration_digest(bad["configuration"])
    with pytest.raises(ValueError):
        records.compare_records(control, bad)


def calibration(frames, repetitions, elapsed):
    row = comparable("loaded")
    row["phase"] = "calibration"
    row["coverage_analysis"] = {"coverage": "covered"}
    row["configuration"].update(frame_count=frames, repeat_count=repetitions)
    row["configuration_sha256"] = plan.configuration_digest(row["configuration"])
    native = row["native"]
    native.update(frame_count=frames, repeat_count=repetitions)
    end = int(elapsed*1e9)
    native["host_interval"] = {"start_ns": 0, "end_ns": end}
    native["host_cpu_work"] = dict(native["host_interval"], operations=repetitions*32768,
                                   reads=repetitions, transfers=repetitions)
    native["host_scheduling"] = {"enqueue_begin_ns": 0, "enqueue_return_ns": 100,
        "load_join_completed_ns": end+1, "completion_observer_joined_ns": end+2,
        "finish_begin_ns": end+3, "finish_end_ns": end+4, "drain_end_ns": end+5}
    native["resident_host_interval"].update(start_ns=0, end_ns=end+4)
    observed = int((frames/1000+.1)*1e9)
    native["program_host_interval"].update(start_ns=0, end_ns=observed,
                                          start_utc_ns=10**18, end_utc_ns=10**18+observed)
    row["overhead"] = {"setup_seconds": 1, "per_frame_analysis_seconds": .000001,
                       "sampler_wait_seconds": 2}
    return row


def test_fixed_count_proposal_never_becomes_authority():
    rows = [calibration(16384,1048576,24), calibration(32768,1572864,36)]
    proposal = records.calibration_proposal(rows)
    assert proposal["predicted_slow_seconds"] + proposal["overhead_reserve_seconds"] <= 600
    assert proposal["execution_authorized"] is False
    assert proposal["requires_precommit_and_owner_approval"]
    rows[1]["timing_evidence"] = False
    with pytest.raises(ValueError):
        records.calibration_proposal(rows)


def test_no_count_that_covers_window_and_fits_cap_means_stop():
    rows = [calibration(16384,1048576,20), calibration(32768,1572864,50)]
    with pytest.raises(ValueError, match="600-second"):
        records.calibration_proposal(rows)


def test_no_execute_side_effect_without_explicit_authority(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a))
    assert runner.main(["--launch", "production-v1-control-watch", "--session-id", "s"]) == 2
    assert calls == []
    assert "approval" in capsys.readouterr().err
    assert json.loads(json.dumps(plan.proposal()))["execution_authorized"] is False
