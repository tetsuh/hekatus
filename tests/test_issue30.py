"""Synthetic source-shaped fixtures, never output from a real board."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from enodia.tt.bench import issue30 as m


def environment():
    return {"schema": "issue30-environment-v1", "attempt": "synthetic-a", "boot": "synthetic-boot",
            "image": m.IMAGE, "kmd": "2.12.0", "watcher": "1",
            "cards": [{"chip": i, "pcie_index": i, "serial": s, "pci": p, "bundle": "19.6.0.0",
                       "cmfw": "0.28.0.0", "erisc": "1.4.1", "aiclk_mhz": 800,
                       "dram_healthy": True, "eth_heartbeat_healthy": True}
                      for i, (s, p) in enumerate(m.CARDS)],
            "links": [{"chip": 0, "channel": 4, "core": "0-4", "peer": 1, "peer_core": "0-4"},
                      {"chip": 1, "channel": 4, "core": "0-4", "peer": 0, "peer_core": "0-4"}],
            "uncabled": [[0, 5], [1, 5]]}


def transcript():
    return ("info | Established firmware bundle version: 19.6.0\n"
            "info | Found 2 chips in cluster:\n"
            "Chip: 0 PCIe: 0 Unique ID: 0\n"
            " eth channel 4 core 0-4 link UP (QSFP), connected to Chip 1 core 0-4\n"
            "\tRetrain count: 0\n"
            " eth channel 5 core 0-5 link DOWN/unconnected (QSFP)\n\n"
            "Chip: 1 PCIe: 1 Unique ID: 1\n"
            " eth channel 4 core 0-4 link UP (QSFP), connected to Chip 0 core 0-4\n"
            "\tRetrain count: 0\n"
            " eth channel 5 core 0-5 link DOWN/unconnected (QSFP)\n\n"
            "warning | Chip: 0 PCIe: 0 Unique ID: 0 eth channel 5 core 0-5 link DOWN/unconnected (QSFP)\n"
            "warning | Chip: 1 PCIe: 1 Unique ID: 1 eth channel 5 core 0-5 link DOWN/unconnected (QSFP)\n").encode("ascii")


def xml(phase="A"):
    suite, test = m.TESTS[phase].split(".")
    return (f'<testsuites tests="1" failures="0" errors="0"><testsuite name="{suite}">'
            f'<testcase name="{test}" classname="{suite}" status="run" result="completed"/>'
            '</testsuite></testsuites>').encode()


def verdict(phase="A", **kw):
    return m.evaluate(phase, "synthetic-a", environment(), environment(), xml(phase),
                      transcript() if phase == "A" else b"info | fabric initialized\n", b"",
                      m.GOOD_EXECUTION | kw,
                      discovery=verdict("A") if phase == "B" else None)


def test_pinned_umd_core_string_format():
    # U device/types/xy_pair.cpp:11: str() is x-y, not RelativeCoreCoord's format.
    env = environment()
    for link in env["links"]:
        link["core"] = link["core"].replace("(x=0,y=4)", "0-4")
        link["peer_core"] = link["peer_core"].replace("(x=0,y=4)", "0-4")
    raw = transcript().replace(b"(x=0,y=4)", b"0-4").replace(b"(x=0,y=5)", b"0-5")
    assert m.evaluate("A", "synthetic-a", env, env, xml(), raw, b"", m.GOOD_EXECUTION)["reason"] == "accepted"


def test_positive_controls():
    assert verdict()["verdict"] == "accepted_discovery"
    assert verdict("B")["verdict"] == "accepted_transfer"


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("exit", 1, "execution"), ("timed_out", True, "execution"),
    ("terminated", False, "execution"), ("watcher_clean", False, "watcher"),
    ("containers_clean", False, "containers"), ("elapsed_seconds", 60.01, "execution"),
    ("client_exit", 1, "execution"), ("started", False, "execution"),
])
def test_one_execution_mutation(field, value, reason):
    assert verdict()["reason"] == "accepted"
    assert verdict(**{field: value})["reason"] == reason


@pytest.mark.parametrize(("old", "new", "reason"), [
    (b'Found 2', b'Found 0', "chip_count"),
    (b'connected to Chip 0', b'connected to Chip 1', "peer_links"),
    (b'Retrain count: 0', b'Retrain count: 1', "retrain"),
    (b'19.6.0', b'19.5.0', "bundle"),
    (b'PCIe: 1', b'PCIe: 9', "chip_identity"),
])
def test_source_report_single_mutation(old, new, reason):
    assert verdict()["verdict"] == "accepted_discovery"
    result = m.evaluate("A", "synthetic-a", environment(), environment(), xml(),
                        transcript().replace(old, new, 1), b"", m.GOOD_EXECUTION)
    assert result["reason"] == reason


def test_uncabled_warning_must_match_chip_identity():
    assert verdict()["reason"] == "accepted"
    changed = transcript().replace(b"warning | Chip: 0 PCIe: 0", b"warning | Chip: 0 PCIe: 999", 1)
    result = m.evaluate("A", "synthetic-a", environment(), environment(), xml(), changed, b"",
                        m.GOOD_EXECUTION)
    assert result["reason"] == "diagnostic"


@pytest.mark.parametrize("severity", ["warning", "error", "fatal", "critical"])
def test_unknown_warning_rejected_not_exit_zero(severity):
    assert verdict()["reason"] == "accepted"
    r = m.evaluate("A", "synthetic-a", environment(), environment(), xml(),
                   transcript(), f"{severity} | unexpected firmware health problem\n".encode("ascii"), m.GOOD_EXECUTION)
    assert r["reason"] == "diagnostic"


@pytest.mark.parametrize("change", ["skip", "wrong", "empty", "malformed", "entity"])
def test_xml_negative_reason(change):
    assert verdict()["reason"] == "accepted"
    raw = {"skip": xml().replace(b'/></testsuite>', b'><skipped/></testcase></testsuite>'),
           "wrong": xml().replace(b'ReportSystemHealth', b'Other'), "empty": b'<testsuites/>',
           "malformed": b'<bad', "entity": b'<!DOCTYPE a>' + xml()}[change]
    assert m.evaluate("A", "synthetic-a", environment(), environment(), raw, transcript(), b"",
                      m.GOOD_EXECUTION)["reason"] == "test_result"


def test_source_warning_location_suffix_is_not_a_health_error():
    assert verdict()["reason"] == "accepted"
    lines = transcript().decode("ascii").splitlines()
    source = "\n".join(line + " (test_system_health.cpp:361)" if line.startswith("warning |") else line
                       for line in lines).encode("ascii")
    assert m.evaluate("A", "synthetic-a", environment(), environment(), xml(), source, b"",
                      m.GOOD_EXECUTION)["reason"] == "accepted"


def test_no_device_skip_is_not_success():
    assert verdict("B")["reason"] == "accepted"
    raw = xml("B").replace(b'/></testsuite>', b'><skipped/></testcase></testsuite>')
    assert m.evaluate("B", "synthetic-a", environment(), environment(), raw, b"", b"",
                      m.GOOD_EXECUTION)["reason"] == "test_result"


@pytest.mark.parametrize(("field", "value"), [("serial", "wrong"), ("erisc", "0.0.0"),
                                               ("dram_healthy", False), ("aiclk_mhz", 0),
                                               ("eth_heartbeat_healthy", False), ("cmfw", "unknown"), ("cmfw", "0.0.0.0")])
def test_one_environment_mutation(field, value):
    assert verdict()["reason"] == "accepted"
    after = copy.deepcopy(environment()); after["cards"][1][field] = value
    assert m.evaluate("A", "synthetic-a", environment(), after, xml(), transcript(), b"",
                      m.GOOD_EXECUTION)["reason"] == "environment"


def test_deadlines_absolute_not_reset_by_stage():
    assert m.remaining(100.0, 50, 139.0) == 11.0
    assert m.remaining(100.0, 60, 165.0) == 0
    assert m.DEADLINES == {"binary": 40, "container": 50, "cleanup": 60}


def test_payload_sandbox_and_fixed_argv(tmp_path):
    payload = m.request("A", "synthetic-a", tmp_path, tmp_path, uid=1000, gid=1000)
    h = payload["HostConfig"]
    assert payload["Image"] == m.IMAGE
    assert payload["User"] == "1000:1000"
    assert h["NetworkMode"] == "none"
    assert h["ReadonlyRootfs"]
    assert h["Devices"] == []
    assert h["CapDrop"] == ["ALL"]
    assert {"/sys", "/proc/bus", "/proc/driver"} <= set(h["MaskedPaths"])
    assert payload["Env"][:3] == ["TT_METAL_WATCHER=1", "HOME=/scratch", "LD_BIND_NOW=1"]
    assert "mode=0700,uid=1000,gid=1000" in h["Tmpfs"]["/scratch"]
    assert "TT_METAL_RUNTIME_ROOT=/opt/venv/lib/python3.12/site-packages/ttnn" in payload["Env"]
    assert payload["Cmd"][0:3] == ["timeout", "--signal=KILL", "40s"]
    assert '--gtest_filter=Cluster.ReportSystemHealth' in payload["Cmd"]
    assert '--gtest_output=xml:/capture/native/gtest.xml' in payload["Cmd"]
    with pytest.raises(ValueError, match="identity"):
        m.request("A", "../unsafe", tmp_path, tmp_path, uid=1000, gid=1000)
    with pytest.raises(ValueError, match="non-root"):
        m.request("A", "safe", tmp_path, tmp_path, uid=0, gid=0)


def test_atomic_result_no_overwrite(tmp_path):
    target = tmp_path / "result.json"
    m.publish(target, {"schema": "issue30-result-v1", "value": 1})
    assert json.loads(target.read_text())["value"] == 1
    with pytest.raises(FileExistsError):
        m.publish(target, {"value": 2})
    assert json.loads(target.read_text())["value"] == 1
    (tmp_path / "link.json").symlink_to(target)
    with pytest.raises(FileExistsError):
        m.publish(tmp_path / "link.json", {})


def test_public_manifest_has_only_relative_artifact_paths(tmp_path):
    p = tmp_path / "bytes.bin"; p.write_bytes(b"\x00\xff\n")
    assert m.artifact(p, tmp_path) == {"path": "bytes.bin", "bytes": 3,
                                       "sha256": m.digest(b"\x00\xff\n")}
    with pytest.raises(ValueError):
        m.artifact(Path("/etc/hostname"), tmp_path)
