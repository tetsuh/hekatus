"""Synthetic source-shaped topology reports, never hardware observations."""
from __future__ import annotations

import copy
import json

import pytest

from enodia.tt.bench import issue30 as m
from tests.test_issue30 import environment, verdict, xml


def policy():
    return json.loads((m.ROOT / "docs/measurements/issue30-a-topology.json").read_bytes())


def env():
    value = environment() | {"topology": policy(), "links": [], "uncabled": []}
    for card in value["cards"]:
        card["arc_heartbeat"] = {"first": 100, "second": 102}
    return value


def certified(discovery):
    from tests.test_issue30_owner import accepted

    # Synthetic owner decision, never a physical approval or per-core inventory.
    return accepted(env()) | {"observed_qsfp": discovery["observed_qsfp"],
                              "topology": discovery["topology"]}


def core(chip, channel):
    # Deliberately synthetic and different on each card; not a SoC descriptor.
    return f"{chip + 1}-{channel + 20}"


def records(hypothesis="H1", *, active=None, crossed=False):
    selected = next(x for x in policy()["hypotheses"] if x["id"] == hypothesis)
    active = set(selected["connected_channels"]) if active is None else set(active)
    partners = {ch: ch for ch in range(4, 12)}
    if crossed:
        for channels in policy()["port_channels"].values():
            a, b = channels
            partners[a], partners[b] = b, a
    result = []
    for chip in (0, 1):
        for channel in range(4, 12):
            item = {"chip": chip, "channel": channel, "core": core(chip, channel),
                    "state": "UP" if channel in active else m.DOWN}
            if channel in active:
                item |= {"peer": 1 - chip, "peer_core": core(1 - chip, partners[channel]),
                         "retrain_count": 0}
            result.append(item)
    return result


def report(rows):
    lines = ["info | Established firmware bundle version: 19.6.0", "info | Found 2 chips in cluster:"]
    warnings = []
    for chip in (0, 1):
        header = f"Chip: {chip} PCIe: {chip} Unique ID: {chip}"
        lines.append(header)
        for item in [x for x in rows if x["chip"] == chip]:
            text = f" eth channel {item['channel']} core {item['core']} link "
            if item["state"] == "UP":
                lines.extend([text + f"UP (QSFP), connected to Chip {item['peer']} core {item['peer_core']}",
                              f"Retrain count: {item['retrain_count']}"])
            else:
                text += "DOWN/unconnected (QSFP)"
                lines.append(text)
                warnings.append("warning | " + header + text)
    return ("\n".join(lines + warnings) + "\n").encode("ascii")


def judge(rows, **kwargs):
    before = kwargs.pop("before", env())
    return m.evaluate("A", "synthetic-a", before, before, xml(), report(rows), b"",
                      m.GOOD_EXECUTION | kwargs)


def test_hypothesis_input_is_not_an_exact_lane_table():
    assert verdict()["reason"] == "accepted"  # Existing path remains a control.
    result = judge(records(crossed=True))
    assert result["verdict"] == "accepted_discovery"
    assert result["topology"]["hypothesis"] == "H1"
    assert result["observed_qsfp"] == records(crossed=True)


@pytest.mark.parametrize("hypothesis", ["H1", "H2"])
@pytest.mark.parametrize("crossed", [False, True])
def test_complete_hypotheses_and_within_port_lane_crossing(hypothesis, crossed):
    result = judge(records(hypothesis, crossed=crossed))
    assert result["verdict"] == "accepted_discovery"
    assert result["topology"]["hypothesis"] == hypothesis
    assert result["schema"] == "issue30-verdict-v3"


@pytest.mark.parametrize(("active", "reason"), [(set(), "no_peers"), ({4, 6}, "partial_peers"),
                                                 ({4, 5}, "partial_peers")])
def test_single_connectivity_input_change_stops_without_training_causality(active, reason):
    assert judge(records())["verdict"] == "accepted_discovery"
    # Only the connection-set input changes; both projections of each physical
    # link change together. All identity/health/XML/execution inputs stay fixed.
    rows = records(active=active)
    result = judge(rows)
    assert result["verdict"] == "link_not_established"
    assert result["reason"] == reason
    assert result["observed_qsfp"] == rows


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("peer", 0, "loopback"), ("peer", 2, "foreign_peer"),
    ("peer_core", core(1, 6), "nonreciprocal"),
    ("peer_core", core(1, 5), "port_pair"),
    ("peer_core", "99-99", "unknown_peer_core"),
])
def test_single_peer_field_mutation_is_unexpected(field, value, reason):
    assert judge(records())["verdict"] == "accepted_discovery"
    rows = records(); rows[0][field] = value
    result = judge(rows)
    assert result["verdict"] == "unexpected_topology"
    assert result["reason"] == reason
    assert result["observed_qsfp"] == rows


def test_source_remote_unique_id_is_retained_as_unresolved_topology():
    assert judge(records())["verdict"] == "accepted_discovery"
    raw = report(records()).replace(b"connected to Chip 1 core 2-24",
                                   b"connected to Unique ID: abcd core 2-24", 1)
    result = m.evaluate("A", "synthetic-a", env(), env(), xml(), raw, b"", m.GOOD_EXECUTION)
    assert result["verdict"] == "unexpected_topology"
    assert result["reason"] == "unresolved_peer_identity"
    assert result["observed_qsfp"][0]["peer_unique_id"] == "abcd"


def test_one_way_link_is_unexpected_not_partial():
    assert judge(records())["verdict"] == "accepted_discovery"
    rows = records(); rows[8] = {k: v for k, v in rows[8].items() if k not in ("peer", "peer_core", "retrain_count")}
    rows[8]["state"] = m.DOWN  # One endpoint state changes, not the whole physical link.
    result = judge(rows)
    assert result["verdict"] == "unexpected_topology"
    assert result["reason"] == "nonreciprocal"


def test_mixed_orientation_connections_are_not_either_hypothesis():
    assert judge(records())["verdict"] == "accepted_discovery"
    result = judge(records(active={4, 6, 8, 10}))
    assert result["verdict"] == "unexpected_topology"
    assert result["reason"] == "hypothesis_mismatch"


@pytest.mark.parametrize(("mutation", "reason"), [("missing", "incomplete_topology_report"),
                                                     ("duplicate", "duplicate_observation")])
def test_report_shape_not_misreported_as_physical_link_absence(mutation, reason):
    assert judge(records())["verdict"] == "accepted_discovery"
    rows = records()
    if mutation == "missing":
        rows.pop()
    else:
        rows.append(copy.deepcopy(rows[0]))
    result = judge(rows)
    assert result["verdict"] == "rejected"
    assert result["reason"] == reason


def test_policy_cannot_be_redefined_by_input():
    assert judge(records())["verdict"] == "accepted_discovery"
    changed = env(); changed["topology"]["hypotheses"][0]["connected_channels"] = [4]
    assert judge(records(), before=changed)["reason"] == "topology_policy"


def test_new_policy_does_not_relax_heartbeat_or_execution():
    assert judge(records())["verdict"] == "accepted_discovery"
    changed = env(); changed["cards"][0]["eth_heartbeat_healthy"] = None
    assert judge(records(), before=changed)["reason"] == "unobservable_reason"
    assert judge(records(), timed_out=True)["reason"] == "execution"


def test_only_corresponding_down_summary_warning_is_allowed():
    assert judge(records())["verdict"] == "accepted_discovery"
    raw = report(records()).replace(b"warning | Chip: 0 PCIe: 0", b"warning | Chip: 0 PCIe: 9", 1)
    result = m.evaluate("A", "synthetic-a", env(), env(), xml(), raw, b"", m.GOOD_EXECUTION)
    assert result["reason"] == "diagnostic"


def test_symmetric_card_relabeling_does_not_choose_a_physical_serial():
    assert judge(records(crossed=True))["verdict"] == "accepted_discovery"
    swapped = records(crossed=True)
    for row in swapped:
        row["chip"] = 1 - row["chip"]
        if row["state"] == "UP":
            row["peer"] = 1 - row["peer"]
    result = judge(swapped)
    assert result["verdict"] == "accepted_discovery"
    assert result["topology"]["hypothesis"] == "H1"


def test_channel_port_implementation_matches_frozen_input():
    from enodia.tt.bench import issue30_topology

    assert issue30_topology.CHANNEL_PORT == {channel: int(port)
                                            for port, channels in policy()["port_channels"].items()
                                            for channel in channels}


def test_B_recomputes_A_topology_instead_of_trusting_verdict_label():
    before = env()
    good = certified(judge(records()))
    def transfer(discovery):
        from tests.test_issue30_owner import smoke_stdout

        return m.evaluate("B", "synthetic-a", before, before, xml("B"), smoke_stdout(), b"",
                          m.GOOD_EXECUTION, discovery)
    assert transfer(good)["verdict"] == "accepted_transfer"
    partial = judge(records(active={4, 6}))
    assert transfer(partial)["reason"] == "discovery_binding"
    partial["verdict"] = "accepted_discovery"  # One shallow-label mutation must not release B.
    assert transfer(partial)["reason"] == "discovery_binding"
