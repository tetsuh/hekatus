"""Pure, precommitted two-hypothesis P150 observation classification; no device IO."""
from __future__ import annotations

import re
from pathlib import Path

from enodia.tt.bench import issue30_formats as formats

POLICY_FILE = "docs/measurements/issue30-a-topology.json"
CHANNEL_PORT = {11: 1, 9: 1, 8: 2, 10: 2, 7: 3, 5: 3, 4: 4, 6: 4}
DOWN = "DOWN/unconnected"


def policy():
    return formats.load((Path(__file__).resolve().parents[3] / POLICY_FILE).read_bytes())


def validate_input(env):
    if (not formats.equal(env.get("topology"), policy())
            or not formats.equal(env.get("links"), []) or not formats.equal(env.get("uncabled"), [])):
        raise ValueError("topology_policy")


def _check_record(record):
    if type(record) is not dict:
        raise ValueError('invalid_observation')
    state = record.get('state')
    peer = ('peer_unique_id',) if 'peer_unique_id' in record else ('peer',)
    extra = (*peer, 'peer_core', 'retrain_count') if state == 'UP' else ()
    formats.fields(record, ('chip', 'channel', 'core', 'state', *extra), reason='invalid_observation')
    if (type(record["chip"]) is not int or type(record["channel"]) is not int
            or not isinstance(record["core"], str) or record["state"] not in ("UP", DOWN)):
        raise ValueError("invalid_observation")
    if not re.fullmatch(r"\d{1,3}-\d{1,3}", record["core"], re.ASCII):
        raise ValueError("invalid_observation")
    if record["state"] == "UP":
        _check_peer(record)
        if type(record.get("retrain_count")) is not int or record["retrain_count"] != 0:
            raise ValueError("retrain")


def _check_peer(record):
    if (not isinstance(record.get("peer_core"), str)
            or not re.fullmatch(r'[0-9]{1,3}-[0-9]{1,3}', record['peer_core'])):
        raise TypeError("invalid_observation")
    if "peer_unique_id" in record:
        uid = record["peer_unique_id"]
        if "peer" in record or not isinstance(uid, str) or not re.fullmatch(r"[a-f0-9]{1,16}", uid):
            raise ValueError("invalid_observation")
    elif type(record.get("peer")) is not int:
        raise ValueError("invalid_observation")


def _index(records):
    if not isinstance(records, list) or len(records) > 32:
        raise ValueError("incomplete_topology_report")
    for record in records:
        _check_record(record)
    channels = [(r["chip"], r["channel"]) for r in records]
    cores = [(r["chip"], r["core"]) for r in records]
    if len(set(channels)) != len(channels) or len(set(cores)) != len(cores):
        raise ValueError("duplicate_observation")
    if set(channels) != {(chip, ch) for chip in (0, 1) for ch in CHANNEL_PORT}:
        raise ValueError("incomplete_topology_report")
    return {(r["chip"], r["core"]): r for r in records}


def _peer_reason(record, indexed):
    if "peer_unique_id" in record:
        return "unresolved_peer_identity"
    if record["peer"] == record["chip"]:
        return "loopback"
    if record["peer"] != 1 - record["chip"]:
        return "foreign_peer"
    remote = indexed.get((record["peer"], record["peer_core"]))
    if remote is None:
        return "unknown_peer_core"
    if CHANNEL_PORT[record["channel"]] != CHANNEL_PORT[remote["channel"]]:
        return "port_pair"
    if (remote["state"] != "UP" or remote.get("peer") != record["chip"]
            or remote.get("peer_core") != record["core"]):
        return "nonreciprocal"
    return None


def _peer_failure(records, indexed):
    for record in records:
        if record["state"] == "UP":
            reason = _peer_reason(record, indexed)
            if reason:
                return reason
    return None


def _decision(outcome, reason, hypothesis=None):
    return {"outcome": outcome, "reason": reason, "hypothesis": hypothesis}


def classify(records):
    try:
        indexed = _index(records)
        reason = _peer_failure(records, indexed)
    except (ValueError, TypeError, KeyError) as exc:
        return _decision("invalid", str(exc) if isinstance(exc, ValueError) else "invalid_observation")
    if reason:
        return _decision("unexpected", reason)
    connected = {(r["chip"], r["channel"]) for r in records if r["state"] == "UP"}
    if not connected:
        return _decision("not_established", "no_peers")
    candidates = {h["id"]: {(chip, ch) for chip in (0, 1) for ch in h["connected_channels"]}
                  for h in policy()["hypotheses"]}
    for name, expected in candidates.items():
        if connected == expected:
            return _decision("accepted", "accepted", name)
    if any(connected < expected for expected in candidates.values()):
        return _decision("not_established", "partial_peers")
    return _decision("unexpected", "hypothesis_mismatch")


def check_warnings(records, warnings):
    down = {(r["chip"], r["channel"], r["core"]) for r in records if r["state"] == DOWN}
    for warning in warnings:
        if (warning["chip"], warning["channel"], warning["core"]) not in down:
            raise ValueError("diagnostic")


def require_accepted_discovery(discovery):
    formats.verdict(discovery)
    decision = classify(discovery.get("observed_qsfp"))
    if (discovery.get("schema") != "issue30-verdict-v3" or decision["outcome"] != "accepted"
            or not formats.equal(discovery.get("topology"), decision)):
        raise ValueError("discovery_binding")
