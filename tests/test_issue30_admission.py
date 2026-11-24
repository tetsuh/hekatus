"""CPU-only discovery admission controls; never physical observations."""
from __future__ import annotations

import copy

import pytest

from enodia.tt.bench import issue30 as m
from tests.test_issue30 import xml
from tests.test_issue30_topology import env, records, report

REASON = {"reason": "blackhole_pre_19_9_telemetry_unavailable", "sources": [
    "umd/device/firmware/firmware_info_provider.cpp:217-229",
    "umd/device/firmware/firmware_utils.cpp:80-99",
    "umd/device/topology/topology_discovery.cpp:302-306"]}


def discovery_env():
    value = env()
    for card in value["cards"]:
        card["erisc"] = None
        card["eth_heartbeat_healthy"] = None
        card["unobservable"] = {"erisc": copy.deepcopy(REASON), "eth_heartbeat_healthy": copy.deepcopy(REASON)}
        card["arc_heartbeat"] = {"first": 100, "second": 102}
        card["temperature_c"] = 55.0
    return value


def result(before=None, after=None, raw=None):
    before = discovery_env() if before is None else before
    return m.evaluate("A", "synthetic-a", before, before if after is None else after, xml(),
                      report(records()) if raw is None else raw, b"", m.GOOD_EXECUTION)


def test_A_payload_and_authority_scope_admit_reasoned_unknown_without_execution(tmp_path):
    from tests.test_issue30_capture import approval_inputs

    _, receipt, _ = approval_inputs(tmp_path)
    before = discovery_env()
    for i, card in enumerate(before["cards"]):
        card["node"] = f"/dev/tenstorrent/{i}"
    approval = m.approval_record("A", "synthetic-a", before, receipt, "synthetic-head")
    payload = m.board_request("A", "synthetic-a", tmp_path, tmp_path / "out", approval=approval,
                              environment=before, receipt=receipt, head="synthetic-head")
    assert len(payload["HostConfig"]["Devices"]) == 2
    assert approval["owner_authorization"]["invocations"] == 1
    assert approval["operator_correction"]["decision_by"] == "operator"
    approval["owner_authorization"]["invocations"] = 2
    with pytest.raises(ValueError, match="approval_binding"):
        m.board_request("A", "synthetic-a", tmp_path, tmp_path / "out", approval=approval,
                        environment=before, receipt=receipt, head="synthetic-head")


def test_A_reasoned_unknown_admitted_but_remains_unknown():
    value = result()
    assert value["verdict"] == "accepted_discovery"
    assert value["firmware_observation"]["per_core_versions"] == []
    assert value["firmware_observation"]["eth_heartbeat_health"] == "unknown"


@pytest.mark.parametrize("field", ["erisc", "eth_heartbeat_healthy"])
def test_A_unknown_without_reason_is_refused(field):
    assert result()["verdict"] == "accepted_discovery"
    changed = discovery_env(); del changed["cards"][0]["unobservable"][field]
    assert result(changed)["reason"] == "unobservable_reason"


@pytest.mark.parametrize(("field", "value"), [("erisc", "1.4.0"), ("eth_heartbeat_healthy", False)])
def test_known_failure_is_not_hidden_by_unknown_reason(field, value):
    assert result()["verdict"] == "accepted_discovery"
    changed = discovery_env(); changed["cards"][0][field] = value
    assert result(changed)["reason"] == "environment"


def test_unknown_reason_does_not_apply_to_newer_bundle():
    assert result()["verdict"] == "accepted_discovery"
    changed = discovery_env(); changed["cards"][0]["bundle"] = "19.9.0.0"
    assert result(changed)["reason"] == "unobservable_reason"


@pytest.mark.parametrize(("field", "value"), [("arc_heartbeat", {"first": 100, "second": 100}),
                                               ("aiclk_mhz", 799), ("dram_healthy", False)])
def test_other_discovery_preconditions_are_not_relaxed(field, value):
    assert result()["verdict"] == "accepted_discovery"
    changed = discovery_env(); changed["cards"][0][field] = value
    assert result(changed)["reason"] == "environment"


def test_progressing_arc_and_temperature_are_observations_not_identity_drift():
    before = discovery_env(); after = copy.deepcopy(before)
    after["cards"][0]["arc_heartbeat"] = {"first": 105, "second": 107}
    after["cards"][0]["temperature_c"] = 56.0
    assert result(before, after)["verdict"] == "accepted_discovery"


def test_reported_first_core_version_is_retained_not_per_core_certification():
    assert result()["verdict"] == "accepted_discovery"
    line = b"debug | Established ETH FW version from first discovered ETH core: 1.7.0\n"
    value = result(raw=line + report(records()))
    assert value["verdict"] == "accepted_discovery"
    assert value["firmware_observation"]["reported_versions"][0]["version"] == "1.7.0"
    assert value["firmware_observation"]["per_core_versions"] == []


def test_firmware_warning_and_low_version_are_recorded_and_stop():
    assert result()["verdict"] == "accepted_discovery"
    warning = b"warning | The expected ETH firmware version 1.3.0 is older than the minimum supported version 1.4.1\n"
    value = result(raw=warning + report(records()))
    assert value["verdict"] == "rejected"
    assert value["reason"] == "diagnostic"
    assert warning.decode().strip() in value["firmware_observation"]["diagnostic_lines"]


def test_B_reasoned_unknown_requires_separate_owner_binding():
    known = env()
    m._environment(known, "synthetic-a", "B")
    unknown = discovery_env()
    m._environment(unknown, "synthetic-a", "B")
    with pytest.raises(ValueError, match="discovery_binding"):
        m._transfer_binding(unknown, None, b"")


def test_old_accepted_A_output_or_inventory_cannot_replace_owner_decision():
    good = result()
    assert good["verdict"] == "accepted_discovery"
    good["firmware_observation"]["per_core_versions"] = [
        {"chip": r["chip"], "core": r["core"], "version": "1.4.1"} for r in good["observed_qsfp"]]
    before = discovery_env()
    with pytest.raises(ValueError, match="discovery_binding"):
        m._transfer_binding(before, good, b"")
