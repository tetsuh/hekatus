"""Exact #176/#120 baseline comparisons; no device IO or warning suppression."""
from __future__ import annotations

import json
import re
from functools import lru_cache

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
    value = json.loads(checks.BASELINE.read_bytes())
    return {normalized(hit["text"]): row["condition"]
            for row in value["evidence"] for hit in row["matches"]}


def _condition(line, cards):
    if len(line) > 16384 or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \| warning {2}\| +[\w:]+ +\| .+", line):
        return None
    text = line.split("| warning  |", 1)[1].strip()
    physical = re.findall(r"physical_device_id: (\d+)", text)
    chips = re.findall(r"chip_id=(\d+)", text)
    if (any(int(x) not in {c["pcie_index"] for c in cards} for x in physical)
            or any(int(x) not in {c["chip"] for c in cards} for x in chips)):
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


def watcher_observation(raw):
    if len(raw) > checks.MAX_WATCHER_BYTES:
        return {"fresh": False, "scan_passed": False, "dump_count": 0}
    result, _ = checks.watcher_check(None, raw, modified_after_start=True)
    extra = re.search(rb"\b(?:fatal|critical|hung|timeout|mismatch)\b", raw, re.IGNORECASE)
    return result | {"scan_passed": result["scan_passed"] and extra is None}
