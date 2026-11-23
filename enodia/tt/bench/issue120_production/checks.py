"""Conservative warning/fresh-Watcher scans, not hardware proofs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

MAX_WATCHER_BYTES = 64 * 1024 * 1024
BASELINE = Path(__file__).resolve().parents[4] / "docs/measurements/2026-11-14-p150a-issue120-p6-api-lock-semaphore-v3-baseline-verification.json"


def normalized(text: str) -> str:
    text = re.sub(r"physical_device_id: \d+", "physical_device_id: <device>", text)
    text = re.sub(r"pid: \d+ tid: \d+", "pid: <pid> tid: <tid>", text)
    return re.sub(r"bus_id=0x[0-9a-f]+", "bus_id=<bus>", text)


def _warning_kind(line: str, allowed: set[str]) -> str:
    if len(line) > 16384:
        return "unexpected"
    if "| warning  |" in line and normalized(line.split("| warning  |", 1)[1].strip()) in allowed:
        return "known"
    if re.search(r"\b(?:warning|error|fatal|critical|traceback)\b|TT_FATAL|assertion failed|terminate called", line, re.IGNORECASE):
        return "unexpected"
    return "other"


def warning_check(streams: dict) -> dict:
    baseline = json.loads(BASELINE.read_bytes())
    allowed = {normalized(hit["text"]) for condition in baseline["evidence"] for hit in condition["matches"]}
    known, count, sample = 0, 0, []
    for name, source in streams.items():
        lines = source.decode("utf-8", errors="replace").splitlines() if isinstance(source, bytes) else source
        for number, line in enumerate(lines, 1):
            kind = _warning_kind(line, allowed)
            known += int(kind == "known")
            if kind == "unexpected":
                count += 1
                if len(sample) < 16:
                    sample.append({"stream": name, "line": number,
                                   "sha256": hashlib.sha256(line.encode()).hexdigest()})
    return {"passed": count == 0, "known_baseline_lines": known,
            "unexpected_count": count, "unexpected_sample": sample,
            "sample_truncated": count > len(sample),
            "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest()}


def watcher_check(before: bytes | None, after: bytes, *, modified_after_start: bool) -> tuple[dict, bytes]:
    if not after or not modified_after_start or after == before:
        return {"fresh": False, "scan_passed": False, "dump_count": 0}, b""
    fresh = after[len(before):] if before is not None and after.startswith(before) else after
    text = fresh.decode("utf-8", errors="replace")
    dumps = len(re.findall(r"^Dump #\d+ at ", text, re.MULTILINE))
    abnormal = bool(re.search(r"assert|exception|error|hang|invalid|warning|saniti", text, re.IGNORECASE))
    return {"fresh": True, "scan_passed": dumps > 0 and not abnormal, "dump_count": dumps,
            "scope": "fresh suffix or rewritten file after execution start; owner inspection still required"}, fresh
