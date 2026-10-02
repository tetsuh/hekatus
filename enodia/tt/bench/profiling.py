"""Host-side parsing for the optional TT-Metal device profiler CSV."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def parse_device_profile_csv(path: str | Path, *, zone_prefix: str = "NS-") -> dict:
    """Aggregate named device zones into per-RISC cycle records.

    TT-Metal emits a preamble followed by a CSV header containing ``zone name``
    and ``zone phase``.  This parser pairs begin/end markers by core, RISC,
    run, and zone name, then reports total cycles and each zone's percentage of
    the named zones on that RISC.  Unpaired markers are retained as diagnostics.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "zone name" in line and "zone phase" in line and "time[cycles since reset]" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"device profiler CSV header not found in {path}")

    starts: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    aggregates: dict[tuple[str, str], list[int]] = defaultdict(list)
    unmatched_begins = 0
    unmatched_ends = 0
    reader = csv.DictReader(lines[header_index:])
    for row in reader:
        zone = (row.get("zone name") or "").strip()
        if not zone.startswith(zone_prefix):
            continue
        phase = (row.get("zone phase") or "").strip().lower()
        try:
            timestamp = int((row.get("time[cycles since reset]") or "").strip())
        except ValueError:
            continue
        key = (
            (row.get("core_x") or "").strip(),
            (row.get("core_y") or "").strip(),
            (row.get("RISC processor type") or "").strip(),
            (row.get("Run ID") or "").strip(),
        )
        zone_key = (*key, zone)
        if phase == "begin":
            starts[zone_key].append(timestamp)
        elif phase == "end":
            if not starts[zone_key]:
                unmatched_ends += 1
                continue
            aggregates[(key[2], zone)].append(timestamp - starts[zone_key].pop())

    for pending in starts.values():
        unmatched_begins += len(pending)

    totals_by_risc = {
        risc: sum(cycles for (zone_risc, _), values in aggregates.items() if zone_risc == risc for cycles in values)
        for risc in {risc for risc, _ in aggregates}
    }
    zones = []
    for (risc, zone), values in sorted(aggregates.items()):
        total_cycles = sum(values)
        risc_total = totals_by_risc[risc]
        zones.append(
            {
                "risc": risc,
                "zone": zone,
                "count": len(values),
                "cycles": total_cycles,
                "min_cycles": min(values),
                "max_cycles": max(values),
                "percent_of_named_risc_cycles": (
                    100.0 * total_cycles / risc_total if risc_total else 0.0
                ),
            }
        )
    return {
        "source": str(path),
        "zone_prefix": zone_prefix,
        "zones": zones,
        "unmatched_begins": unmatched_begins,
        "unmatched_ends": unmatched_ends,
    }
