"""CPU-only lifecycle tests: Docker and clocks are fakes, no container/device use."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock

import pytest

from enodia.tt.bench import issue30 as m


@pytest.fixture(autouse=True)
def no_real_docker_ps(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_docker_ps", lambda path, timeout: path.write_bytes(b""))
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "ASSETS", tmp_path)


def test_cli_defaults_to_the_receipted_assets(tmp_path, monkeypatch, capsys):
    observed = []
    original = m.request

    def record_request(phase, identity, assets, output, **kwargs):
        observed.append(assets.as_posix())
        return original(phase, identity, assets, output, **kwargs)

    monkeypatch.setattr(m, "request", record_request)
    monkeypatch.setattr("sys.argv", ["issue30", "plan", "--phase", "A", "--attempt", "synthetic"])
    assert m.main() == 0
    assert json.loads(capsys.readouterr().out)["Image"] == m.IMAGE
    assert observed == [m.ASSETS.as_posix()]
    assert m.CURRENT_RECEIPT.name == "receipt-v6.json"
    assert m.CURRENT_RECEIPT.parent == m.ROOT / "docs/measurements/issue30-g0"


def test_capture_preserves_bytes_and_direct_exit(tmp_path, monkeypatch):
    calls = []

    def api(method, path, body=None, timeout=5):
        calls.append((method, path, timeout))
        if path == "containers/json?all=1":
            return []
        if path.startswith("containers/create"):
            assert body["HostConfig"]["Devices"] == []
            return {"Id": "a" * 64}
        if path.endswith("/json"):
            return {"State": {"ExitCode": 7, "Running": False}}
        return {}

    def popen(command, stdout, stderr):
        stdout.write(b"\x00\xffafter-result\n"); stderr.write(b"\xfe\n")
        return Mock(wait=Mock(return_value=0))

    monkeypatch.setattr(m, "_api", api)
    monkeypatch.setattr(m.subprocess, "Popen", popen)
    output = tmp_path / "fresh"
    state = m.capture("A", "synthetic", tmp_path, output)
    assert state["exit"] == 7
    assert state["client_exit"] == 0
    assert state["device_free"]
    assert state["containers_clean"]
    assert (output / "stdout.bin").read_bytes() == b"\x00\xffafter-result\n"
    assert (output / "stderr.bin").read_bytes() == b"\xfe\n"
    assert json.loads((output / "execution.json").read_bytes())["schema"] == "issue30-execution-v1"
    assert any(x[1] == "containers/" + "a" * 64 + "?force=1" for x in calls)
    with pytest.raises(FileExistsError):
        m.capture("A", "synthetic", tmp_path, output)


def test_timeout_kills_named_container_then_client_before_cleanup(tmp_path, monkeypatch):
    actions = []
    now = [100.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: now[0])
    process = Mock()

    def wait(timeout):
        if now[0] == 149.0:
            now[0] = 151.0
            raise subprocess.TimeoutExpired("synthetic", 1)
        now[0] = 153.0
        return 137

    def popen(*args, **kwargs):
        now[0] = 149.0
        return process

    process.wait.side_effect = wait
    monkeypatch.setattr(m.subprocess, "Popen", popen)

    def api(method, path, body=None, timeout=5):
        actions.append((method, path, timeout))
        if path.startswith("containers/create"):
            return {"Id": "a" * 64}
        if path.endswith("/json"):
            return {"State": {"ExitCode": 137, "Running": False}}
        return [] if path == "containers/json?all=1" else {}

    monkeypatch.setattr(m, "_api", api)
    state = m.capture("B", "synthetic", tmp_path, tmp_path / "attempt")
    assert state["timed_out"]
    assert state["terminated"]
    assert process.wait.call_args_list[0].kwargs["timeout"] == 1.0
    assert ("POST", "containers/" + "a" * 64 + "/kill", 4.0) in actions
    process.kill.assert_called_once()
    assert state["elapsed_seconds"] <= 60


def test_collision_does_not_delete_existing_container(tmp_path, monkeypatch):
    calls = []

    def api(method, path, body=None, timeout=5):
        calls.append(method)
        if path.startswith("containers/create"):
            raise subprocess.CalledProcessError(22, "curl")
        return []

    monkeypatch.setattr(m, "_api", api)
    state = m.capture("A", "synthetic", tmp_path, tmp_path / "attempt")
    assert state["exit"] is None
    assert not state["started"]
    assert "DELETE" not in calls


def approval_inputs(tmp_path):
    from tests.test_issue30 import environment

    env = environment()
    for i, card in enumerate(env["cards"]):
        card["node"] = f"/dev/tenstorrent/{i}"
    for name in m.BINARIES.values():
        (tmp_path / name).write_bytes(b"synthetic-not-an-executable")
    receipt = {"schema": "issue30-g0-v1", "all_passed": True,
               "current_wrapper_sources": m.program_sources(),
               "binaries": {name: m.digest(b"synthetic-not-an-executable") for name in m.BINARIES.values()}}
    approval = m.approval_record("A", "synthetic-a", env, receipt, "synthetic-head")
    return env, receipt, approval


def test_failed_g0_receipt_cannot_authorize_hardware(tmp_path):
    env, receipt, approval = approval_inputs(tmp_path)
    positive = m.board_request("A", "synthetic-a", tmp_path, tmp_path / "out", approval=approval,
                               environment=env, receipt=receipt, head="synthetic-head")
    assert len(positive["HostConfig"]["Devices"]) == 2
    assert positive["Env"][0] == "TT_METAL_WATCHER=1"
    assert all(x["CgroupPermissions"] == "rw" for x in positive["HostConfig"]["Devices"])
    receipt["all_passed"] = False
    approval["receipt_sha256"] = m.digest(json.dumps(receipt, sort_keys=True).encode())
    with pytest.raises(ValueError, match="g0_not_ready"):
        m.board_request("A", "synthetic-a", tmp_path, tmp_path / "out", approval=approval,
                        environment=env, receipt=receipt, head="synthetic-head")
    approval["head"] = "changed"
    with pytest.raises(ValueError, match="approval_binding"):
        m.board_request("A", "synthetic-a", tmp_path, tmp_path / "out", approval=approval,
                        environment=env, receipt=receipt, head="synthetic-head")


def test_stale_source_receipt_cannot_authorize_hardware(tmp_path):
    env, receipt, approval = approval_inputs(tmp_path)
    assert m.board_request("A", "synthetic-a", tmp_path, tmp_path / "out", approval=approval,
                           environment=env, receipt=receipt, head="synthetic-head")["Image"] == m.IMAGE
    receipt["current_wrapper_sources"]["enodia/tt/bench/issue30.py"] = "0" * 64
    approval["receipt_sha256"] = m.digest(json.dumps(receipt, sort_keys=True).encode())
    with pytest.raises(ValueError, match="receipt_source_binding"):
        m.board_request("A", "synthetic-a", tmp_path, tmp_path / "out", approval=approval,
                        environment=env, receipt=receipt, head="synthetic-head")


def test_paths_and_identity_rejected_before_io(tmp_path, monkeypatch):
    assert m._owned(tmp_path / "safe") == tmp_path / "safe"
    assert m._identity("synthetic") == "synthetic"
    with pytest.raises(ValueError, match="program-owned"):
        m.capture("A", "synthetic", tmp_path, tmp_path.parent / "escape")
    with pytest.raises(ValueError, match="identity"):
        m.capture("A", "../escape", tmp_path, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_cli_rejects_arbitrary_path_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["issue30", "capture", "--phase", "A", "--attempt", "synthetic",
                                    "--out", "/unapproved"])
    with pytest.raises(SystemExit) as error:
        m.main()
    assert error.value.code == 2


def test_B_authorization_rechecks_A_before_preparing_device_mapping(tmp_path, monkeypatch):
    from tests.test_issue30_topology import certified, env, judge, records

    before = env()
    discovery = [certified(judge(records()))]
    fake_receipt = {"source_commit": m.SOURCE_COMMIT, "runtime_library_sha256": m.LIBRARY_SHA256}
    values = {m.BEFORE_FILE: before, m.CURRENT_RECEIPT.name: fake_receipt, "approval.json": {},
              "accepted-a.json": discovery[0]}
    monkeypatch.setattr(m, "_load", lambda path, *args: discovery[0] if path.name == "discovery.json" else values[path.name])
    pins = {m.SOURCE_ROOT: m.SOURCE_COMMIT, m.SOURCE_ROOT / "tt_metal/third_party/umd": m.UMD_COMMIT}
    monkeypatch.setattr(m, "_git", lambda path, cmd, *args: "" if cmd == "status" else pins.get(path, "synthetic-head"))
    original = m.Path.read_text
    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/sys/kernel/random/boot_id":
            return before["boot"]
        if str(path) == "/sys/module/tenstorrent/version":
            return before["kmd"]
        return original(path, *args, **kwargs)
    monkeypatch.setattr(m.Path, "read_text", read_text)
    monkeypatch.setattr(m.Path, "is_file", lambda path: True)
    prepare = Mock(return_value={})
    monkeypatch.setattr(m, "board_request", prepare)
    m._authorize("B", "synthetic-a", tmp_path, tmp_path / "out")
    prepare.assert_called_once()
    prepare.reset_mock()
    discovery[0] = judge(records(active={4, 6}))  # Only A connectivity changes.
    with pytest.raises(ValueError, match="owner_acceptance_binding"):
        m._authorize("B", "synthetic-a", tmp_path, tmp_path / "out")
    prepare.assert_not_called()
    values[m.BEFORE_FILE] = {k: v for k, v in before.items() if k != "topology"}
    with pytest.raises(ValueError, match="topology_policy"):
        m._authorize("A", "synthetic-a", tmp_path, tmp_path / "out")
    prepare.assert_not_called()


def test_daemon_id_is_not_a_path_or_option():
    assert m._container_id("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="daemon container identity"):
        m._container_id("../../other")
