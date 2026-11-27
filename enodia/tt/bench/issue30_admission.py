"""Reasoned unknown health and owner-accepted link evidence; no device IO."""
from __future__ import annotations

import re

from enodia.tt.bench import issue30_formats as formats
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
    if bundle >= (19, 9, 0) or not isinstance(reasons, dict) or not formats.equal(reasons.get(field), REASON):
        raise ValueError("unobservable_reason")


def card_health(card, phase):
    if any(not formats.equal(reason, REASON) for reason in card.get('unobservable', {}).values()):
        raise ValueError('unobservable_reason')
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
    if type(card.get("aiclk_mhz")) is not int or card["aiclk_mhz"] != 800:
        raise ValueError("environment")


def stable_environment(env):
    # Rejected JSON still needs a fingerprint; preserve malformed container types
    # instead of dereferencing them or normalizing them into valid card objects.
    if type(env) is not dict:
        return env
    result = {k: v for k, v in env.items() if k not in ("attempt", "dram_admission")}
    cards = env.get('cards', [])
    if type(cards) is list:
        result['cards'] = [_stable_card(c) for c in cards]
    return result


def _stable_card(card):
    if type(card) is not dict:
        return card
    return {k: v for k, v in card.items() if k not in ('arc_heartbeat', 'temperature_c')}


def _dram_card(card):
    formats.fields(card, ('serial', 'dram_status', 'raw_ddr_status', 'uncorrectable_errors',
                          'arc_heartbeat', 'aiclk_mhz'), reason='dram_gate_reading')
    if type(card['serial']) is not str:
        raise ValueError('dram_gate_cards')
    heartbeat = card['arc_heartbeat']
    if (card['dram_status'] is not True or card['raw_ddr_status'] != '0x5555'
            or type(card['uncorrectable_errors']) is not int or card['uncorrectable_errors'] != 0
            or type(card['aiclk_mhz']) is not int or card['aiclk_mhz'] != 800
            or type(heartbeat) is not int or heartbeat < 0):
        raise ValueError('dram_gate_reading')


def _dram_sample(sample, serials):
    formats.fields(sample, ('monotonic_seconds', 'raw_sha256', 'cards'), reason='dram_gate_sample')
    stamp = sample.get("monotonic_seconds")
    if not formats.finite(stamp) or stamp < 0:
        raise ValueError("dram_gate_timestamp")
    raw_hash = sample['raw_sha256']
    if type(raw_hash) is not str or not re.fullmatch(r'[a-f0-9]{64}', raw_hash):
        raise ValueError('dram_gate_raw')
    cards = sample['cards']
    if type(cards) is not list or len(cards) != 2:
        raise ValueError('dram_gate_cards')
    for card in cards:
        _dram_card(card)
    if {card['serial'] for card in cards} != serials:
        raise ValueError('dram_gate_cards')
    return {c['serial']: c['arc_heartbeat'] for c in cards}


def dram_gate(env):
    if "owner_unit" not in env:
        return
    evidence = env.get("dram_admission", {})
    formats.fields(evidence, ('schema', 'samples'), reason='dram_gate_five_required')
    samples = evidence.get("samples", [])
    if type(samples) is not list or evidence.get("schema") != "issue30-dram-five-v1" or len(samples) != 5:
        raise ValueError("dram_gate_five_required")
    serials = {c["serial"] for c in env["cards"]}
    previous, previous_time = None, None
    for sample in samples:
        current = _dram_sample(sample, serials)
        stamp = sample["monotonic_seconds"]
        if previous is not None and (stamp - previous_time < 5 or any(current[s] <= previous[s] for s in serials)):
            raise ValueError("dram_gate_spacing_or_heartbeat")
        previous, previous_time = current, stamp


def observe_firmware(raw):
    text = raw[:8 * 1024 * 1024].decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines()
             if any(word in line.lower() for word in ("firmware", "eth fw", "erisc", "heartbeat"))]
    found = []
    prefix = "Established ETH FW version from first discovered ETH core: "
    for line in lines:
        if prefix in line:
            token = re.sub(r' \([A-Za-z0-9_./-]+:\d+\)$', '', line.partition(prefix)[2], flags=re.ASCII)
            found.append({"scope": "first_core_only", "version": token, "line": line})
    return {"reported_versions": found, "per_core_versions": [],
            "erisc_health": "unknown", "eth_heartbeat_health": "unknown",
            "diagnostic_lines": lines, "truncated": len(raw) > 8 * 1024 * 1024}


def check_reported_versions(observation):
    try:
        if any(version(item['version']) < (1, 4, 1) for item in observation['reported_versions']):
            raise ValueError('erisc_version')
    except ValueError as exc:
        raise ValueError('erisc_version') from exc


def owner_links(record, baseline_sha, boot_sha, cabling_sha):
    formats.fields(record, ('schema', 'decision_by', 'decision_date', 'logical_unit',
        'not_a_rejudgment', 'allows_B_consideration', 'baseline_sha256', 'boot_sha256',
        'cabling_policy_sha256', 'bundle', 'original_verdict', 'observed_qsfp', 'topology'),
        ('acceptance_reasons', 'attempt', 'baseline_match_sha256', 'cabling', 'cards_before_after',
         'does_not_assert', 'erisc_version', 'eth_heartbeat_health', 'original_record',
         'original_record_sha256', 'original_verdict_sha256', 'raw_evidence', 'stable_card_state'),
        'discovery_binding')
    if record.get("schema") != "issue30-owner-accepted-A-v1":
        raise ValueError("discovery_binding")
    if record.get("boot_sha256") != boot_sha:
        raise ValueError("owner_boot")
    required = {"schema": "issue30-owner-accepted-A-v1", "decision_by": "owner",
                "decision_date": "2026-11-24", "logical_unit": "issue30-a-discovery-watcher-v1",
                "not_a_rejudgment": True, "allows_B_consideration": True,
                "baseline_sha256": baseline_sha, "cabling_policy_sha256": cabling_sha,
                "bundle": ".".join(map(str, (19, 6, 0, 0)))}
    original = record.get("original_verdict", {})
    if type(original) is not dict:
        raise ValueError('discovery_binding')
    decision = topology.classify(record.get("observed_qsfp"))
    if (any(not formats.equal(record.get(k), v) for k, v in required.items())
            or original.get("phase") != "A" or original.get("verdict") != "rejected"
            or original.get("reason") != "watcher" or decision["outcome"] != "accepted"
            or decision["hypothesis"] != "H1" or not formats.equal(record.get("topology"), decision)):
        raise ValueError("discovery_binding")
