"""CPU-only lifecycle tests: Docker and clocks are fakes, no container/device use."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock

import pytest

from enodia.tt.bench import issue30 as m


@pytest.fixture(autouse=True)
def no_real_docker_ps(monkeypatch):
    monkeypatch.setattr(m, "_docker_ps", lambda path, timeout: path.write_bytes(b""))


def test_cli_defaults_to_the_receipted_assets(tmp_path, monkeypatch, capsys):
    observed = []
    original = m.request

    def record_request(phase, identity, assets, output, **kwargs):
        observed.append(assets.as_posix())
        return original(phase, identity, assets, output, **kwargs)

    monkeypatch.setattr(m, "request", record_request)
    monkeypatch.setattr("sys.argv", ["issue30", "plan", "--phase", "A", "--attempt", "synthetic",
                                    "--out", str(tmp_path / "attempt")])
    assert m.main() == 0
    assert json.loads(capsys.readouterr().out)["Image"] == m.IMAGE
    assert observed == ["out/issue30/g0/assets"]


def test_capture_preserves_bytes_and_direct_exit(tmp_path, monkeypatch):
    calls = []

    def api(method, path, body=None, timeout=5):
        calls.append((method, path, timeout))
        if path == "containers/json?all=1":
            return []
        if path.startswith("containers/create"):
            assert body["HostConfig"]["Devices"] == []
            return {"Id": "synthetic-container-id"}
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
    assert state["exit"] == 7 and state["client_exit"] == 0
    assert state["device_free"] and state["containers_clean"]
    assert (output / "stdout.bin").read_bytes() == b"\x00\xffafter-result\n"
    assert (output / "stderr.bin").read_bytes() == b"\xfe\n"
    assert json.loads((output / "execution.json").read_bytes())["schema"] == "issue30-execution-v1"
    assert any(x[1] == "containers/synthetic-container-id?force=1" for x in calls)
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
            return {"Id": "synthetic-id"}
        if path.endswith("/json"):
            return {"State": {"ExitCode": 137, "Running": False}}
        return [] if path == "containers/json?all=1" else {}

    monkeypatch.setattr(m, "_api", api)
    state = m.capture("B", "synthetic", tmp_path, tmp_path / "attempt")
    assert state["timed_out"] and state["terminated"]
    assert process.wait.call_args_list[0].kwargs["timeout"] == 1.0
    assert ("POST", "containers/synthetic-id/kill", 4.0) in actions
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
    assert state["exit"] is None and not state["started"]
    assert "DELETE" not in calls


def test_failed_g0_receipt_cannot_authorize_hardware(tmp_path, monkeypatch):
    from tests.test_issue30 import environment

    env = environment()
    for i, card in enumerate(env["cards"]):
        card["node"] = f"/dev/tenstorrent/{i}"
    for name in m.BINARIES.values():
        (tmp_path / name).write_bytes(b"synthetic-not-an-executable")
    receipt = {"schema": "issue30-g0-v1", "all_passed": True,
               "binaries": {name: m.digest(b"synthetic-not-an-executable") for name in m.BINARIES.values()}}
    approval = {"schema": "issue30-owner-approval-v1", "attempt": "synthetic-a", "phase": "A",
                "head": "synthetic-head", "boot": env["boot"],
                "environment_sha256": m.digest(json.dumps(env, sort_keys=True).encode()),
                "receipt_sha256": m.digest(json.dumps(receipt, sort_keys=True).encode())}
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
