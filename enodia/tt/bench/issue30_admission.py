"""Reasoned unknown health and owner-accepted link evidence; no device IO."""
from __future__ import annotations

import re

from enodia.tt.bench import issue30_topology as topology

REASON = {"reason": "blackhole_pre_19_9_telemetry_unavailable", "sources": [
    "umd/device/firmware/firmware_info_provider.cpp:217-229",
    "umd/device/firmware/firmware_utils.cpp:80-99",
    "umd/device/topology/topology_discovery.cpp:302-306"]}
CORRECTION = {"decision_by": "operator", "date": "2026-11-24",
              "decision": "admit_reasoned_unobservable_health_for_discovery_only",
              "record": "docs/adr/0021-discovery-observability-gate.md"}


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}", value, re.ASCII):
        raise ValueError("environment")
    return tuple(map(int, value.split(".")))


def _unknown(card, field, phase):
    if phase not in ("A", "B"):
        raise ValueError("environment")
    bundle = tuple(map(int, card["bundle"].split(".")[:3]))
    reasons = card.get("unobservable")
    if bundle >= (19, 9, 0) or not isinstance(reasons, dict) or reasons.get(field) != REASON:
        raise ValueError("unobservable_reason")


def card_health(card, phase):
    if card.get("erisc") is None:
        _unknown(card, "erisc", phase)
    elif version(card.get("erisc")) < (1, 4, 1):
        raise ValueError("environment")
    heartbeat = card.get("eth_heartbeat_healthy")
    if heartbeat is None:
        _unknown(card, "eth_heartbeat_healthy", phase)
    elif heartbeat is not True:
        raise ValueError("environment")


def arc_progress(card):
    sample = card.get("arc_heartbeat", {})
    if not isinstance(sample, dict):
        raise TypeError("environment")
    first, second = sample.get("first"), sample.get("second")
    if type(first) is not int or type(second) is not int or not 0 <= first < second:
        raise ValueError("environment")
    if card.get("aiclk_mhz") != 800:
        raise ValueError("environment")


def stable_environment(env):
    result = {k: v for k, v in env.items() if k != "attempt"}
    result["cards"] = [{k: v for k, v in c.items() if k not in ("arc_heartbeat", "temperature_c")}
                       for c in env.get("cards", [])]
    return result


def observe_firmware(raw):
    text = raw[:8 * 1024 * 1024].decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines()
             if any(word in line.lower() for word in ("firmware", "eth fw", "erisc", "heartbeat"))]
    found = []
    prefix = "Established ETH FW version from first discovered ETH core: "
    for line in lines:
        if prefix in line:
            tokens = line.partition(prefix)[2].split()
            if not tokens:
                continue
            token = tokens[0]
            try:
                version(token)
            except ValueError:
                continue
            found.append({"scope": "first_core_only", "version": token, "line": line})
    return {"reported_versions": found, "per_core_versions": [],
            "erisc_health": "unknown", "eth_heartbeat_health": "unknown",
            "diagnostic_lines": lines, "truncated": len(raw) > 8 * 1024 * 1024}


def check_reported_versions(observation):
    if any(version(item["version"]) < (1, 4, 1) for item in observation["reported_versions"]):
        raise ValueError("erisc_version")


def owner_links(record, baseline_sha, boot_sha, cabling_sha):
    if record.get("schema") != "issue30-owner-accepted-A-v1":
        raise ValueError("discovery_binding")
    if record.get("boot_sha256") != boot_sha:
        raise ValueError("owner_boot")
    required = {"schema": "issue30-owner-accepted-A-v1", "decision_by": "owner",
                "decision_date": "2026-11-24", "logical_unit": "issue30-a-discovery-watcher-v1",
                "not_a_rejudgment": True, "allows_B_consideration": True,
                "baseline_sha256": baseline_sha, "cabling_policy_sha256": cabling_sha,
                "bundle": "19.6.0.0"}
    original = record.get("original_verdict", {})
    decision = topology.classify(record.get("observed_qsfp"))
    if (any(record.get(k) != v for k, v in required.items())
            or original.get("phase") != "A" or original.get("verdict") != "rejected"
            or original.get("reason") != "watcher" or decision["outcome"] != "accepted"
            or decision["hypothesis"] != "H1" or record.get("topology") != decision):
        raise ValueError("discovery_binding")
