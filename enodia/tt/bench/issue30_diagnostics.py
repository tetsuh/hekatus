"""Exact #176/#120 baseline comparisons; no device IO or warning suppression."""
from __future__ import annotations

import re
from functools import lru_cache

from enodia.tt.bench import issue30_formats as formats
from enodia.tt.bench.issue120_production import checks

ALARM = re.compile(r"\b(?:warning|error|fatal|critical|traceback)\b|TT_FATAL|assertion failed|terminate called",
                   re.IGNORECASE)
SERIOUS = re.compile(r"\b(?:error|fatal|critical|traceback)\b|TT_FATAL|assertion failed|terminate called",
                     re.IGNORECASE)


def normalized(text):
    # #120's one-card comparison plus the variable chip ID of the second card.
    return re.sub(r"chip_id=\d+", "chip_id=<chip>", checks.normalized(text))


@lru_cache(maxsize=1)
def baseline():
    value = formats.load(checks.BASELINE.read_bytes())
    return {normalized(hit["text"]): row["condition"]
            for row in value["evidence"] for hit in row["matches"]}


def _bus_binding(text, cards):
    pairs = re.findall(r"chip_id=(\d+) \(bus_id=(0x[0-9a-fA-F]+)\)", text)
    if len(pairs) != text.count("bus_id="):
        return False
    buses = {str(c["chip"]): int(c["pci"].split(":")[1], 16) for c in cards}
    return all(chip in buses and int(bus, 16) == buses[chip] for chip, bus in pairs)


def _condition(line, cards):
    # The frozen motherboard body contains an exact U+2014 dash. Only the
    # invocation numerals are ASCII; do not generalize the literal's encoding.
    if len(line) > 16384 or any(c.isdecimal() and not c.isascii() for c in line):
        return None
    envelope = formats.logger(line)
    if envelope is None or envelope['level'] != 'warning  ':
        return None
    text = line.split("| warning  |", 1)[1].strip()
    physical = re.findall(r"physical_device_id: (\d+)", text)
    chips = re.findall(r"chip_id=(\d+)", text)
    if (any(int(x) not in {c["pcie_index"] for c in cards} for x in physical)
            or any(int(x) not in {c["chip"] for c in cards} for x in chips)):
        return None
    if not _bus_binding(text, cards):
        return None
    return baseline().get(normalized(text))


def _physical(line):
    match = re.search(r"physical_device_id: (\d+)", line)
    return match[1] if match else None


def baseline_at(lines, index, cards):
    kind = _condition(lines[index], cards)
    if kind not in ("numa_membind", "numa_fallback_detail"):
        return kind is not None
    other = index + (1 if kind == "numa_membind" else -1)
    expected = "numa_fallback_detail" if kind == "numa_membind" else "numa_membind"
    return (0 <= other < len(lines) and _condition(lines[other], cards) == expected
            and _physical(lines[index]) == _physical(lines[other]))


def _watcher_event(line):
    if line.startswith('At '):
        match = re.fullmatch(r'At (\d+\.\d{3})s (?:starting|(?:attach|detach) device [01])', line, re.ASCII)
        if match is None:
            raise ValueError('watcher_time')
        return 'event', None, float(match[1])
    if line.startswith('Dump #'):
        match = re.fullmatch(r'Dump #(\d+)( completed)? at (\d+\.\d{3})s', line, re.ASCII)
        if match is None:
            raise ValueError('watcher_dump')
        return 'end' if match[2] else 'start', int(match[1]), float(match[3])
    return None


def _dump_step(state, kind, number):
    if kind == 'start':
        if state['active'] is not None or number != state['count'] + 1:
            raise ValueError('watcher_dump')
        state.update(active=number, body=False)
    elif kind == 'end':
        if state['active'] != number or not state['body']:
            raise ValueError('watcher_dump')
        state.update(active=None, count=state['count'] + 1)


def _device_row(line):
    if not line.startswith('Device '):
        return False
    if not line.removeprefix('Device ').strip():
        raise ValueError('watcher_dump')
    return True


def _watcher_dumps(text, bound):
    state, previous = {'active': None, 'count': 0, 'body': False}, 0.0
    for line in text.splitlines():
        event = _watcher_event(line)
        if event is None:
            if state['active'] is not None and _device_row(line):
                state['body'] = True
            continue
        kind, number, stamp = event
        if not previous <= stamp <= bound:
            raise ValueError('watcher_time')
        previous = stamp
        _dump_step(state, kind, number)
    if state['active'] is not None or state['count'] == 0:
        raise ValueError('watcher_dump')
    return state['count']


def watcher_observation(raw, elapsed=40):
    refused = {'fresh': False, 'scan_passed': False, 'dump_count': 0}
    if len(raw) > checks.MAX_WATCHER_BYTES or not formats.finite(elapsed) or not 0 <= elapsed <= 60:
        return refused
    try:
        text = raw.decode('utf-8')
        count = _watcher_dumps(text, min(40, elapsed))
    except ValueError:
        return refused
    result, _ = checks.watcher_check(None, raw, modified_after_start=True)
    extra = re.search(rb"\b(?:fatal|critical|hung|timeout|mismatch)\b", raw, re.IGNORECASE)
    return result | {'dump_count': count, 'scan_passed': result['scan_passed'] and extra is None}
