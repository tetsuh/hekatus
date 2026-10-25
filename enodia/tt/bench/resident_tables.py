"""Board-free table wire format and frame-local checksum for Issue #119.

The metadata is one DRAM page: a four-word header, four words per profile
(config, generation, ready-at attempted frame, reserved), then three words
per switch (attempted frame, config, generation). Every table has the same
configured byte count. Readiness is a deterministic positive-control model,
not host-triggered table derivation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enodia.tt.bench.resident_harness import ResidentConfig

WORDS = 1024
MASK = 0xffffffff
TABLE_CONFIG_FIELDS = (
    "profile_tables", "switch_schedule", "initial_identity", "table_bytes", "swap_strategy",
)


def enabled(config: ResidentConfig) -> bool:
    return bool(config.profile_tables or config.switch_schedule or config.table_bytes)


def _uint(value) -> bool:
    return type(value) is int and 0 <= value <= MASK


def _validate_profile_rows(config: ResidentConfig) -> None:
    identities = set()
    for row in config.profile_tables:
        if len(row) != 3 or not all(map(_uint, row)):
            raise ValueError("profile_tables rows must be config,generation,ready_frame uint32")
        if tuple(row[:2]) in identities:
            raise ValueError("duplicate profile table identity")
        identities.add(tuple(row[:2]))


def _validate_schedule(config: ResidentConfig) -> None:
    previous = -1
    for row in config.switch_schedule:
        if len(row) != 3 or not all(map(_uint, row)):
            raise ValueError("switch_schedule rows must be frame,config,generation uint32")
        if row[0] <= previous or row[0] >= config.frame_count:
            raise ValueError("switch_schedule frames must increase and be below frame_count")
        previous = row[0]


def validate_tables(config: ResidentConfig) -> None:
    """Validate bounded wire-format inputs; caller translates ValueError to preflight."""
    if config.swap_strategy not in {"copy_on_demand", "prefetch"}:
        raise ValueError("swap_strategy must be copy_on_demand or prefetch")
    if not _uint(config.table_bytes) or config.table_bytes % 4096:
        raise ValueError("table_bytes must be a uint32 multiple of 4096")
    if len(config.initial_identity) != 2 or not all(map(_uint, config.initial_identity)):
        raise ValueError("initial_identity must contain two uint32 words")
    _validate_profile_rows(config)
    _validate_schedule(config)
    if enabled(config) and not config.table_bytes:
        raise ValueError("table_bytes must be positive with tables or switches")
    if 4 + 4 * len(config.profile_tables) + 3 * len(config.switch_schedule) > WORDS:
        raise ValueError("table metadata exceeds one page")
    if 20 + 4 * (len(config.switch_schedule) + 1) > WORDS:
        raise ValueError("switch stats exceed one page")
    if config.table_bytes * len(config.profile_tables) > 1024**3:
        raise ValueError("preloaded profile tables exceed the DRAM allocation bound")


def identity_at(config: ResidentConfig, frame: int) -> tuple[tuple[int, int], int]:
    identity = config.initial_identity
    segment = 0
    for index, (start, config_id, generation) in enumerate(config.switch_schedule):
        if start > frame:
            break
        identity = (config_id, generation)
        segment = index + 1
    return tuple(identity), segment


def ready_index(config: ResidentConfig, frame: int, identity: tuple[int, int]) -> int | None:
    for index, (config_id, generation, ready_frame) in enumerate(config.profile_tables):
        if identity == (config_id, generation) and frame >= ready_frame:
            return index
    return None


def metadata_words(config: ResidentConfig) -> list[int]:
    validate_tables(config)
    words = [len(config.profile_tables), len(config.switch_schedule), *config.initial_identity]
    for row in config.profile_tables:
        words.extend((*row, 0))
    for row in config.switch_schedule:
        words.extend(row)
    return words + [0] * (WORDS - len(words))


def table_word(identity: tuple[int, int], word: int) -> int:
    """Synthetic derivative input, generated once on host before launch."""
    config_id, generation = identity
    return ((config_id + 1) * 0x9e3779b1 ^ (generation + 1) * 0x85ebca77 ^ word) & MASK


def frame_checksum(config: ResidentConfig, frame: int, identity: tuple[int, int]) -> int:
    accumulator = 0
    for work in range(config.work_per_frame):
        word = work % WORDS
        payload = ((frame + word) & 0x7ff) | (((3 * frame + 5 * word) & 0x7ff) << 16)
        accumulator = (
            accumulator * 33 ^ payload ^ ((work + frame) & MASK)
            ^ table_word(identity, work % (config.table_bytes // 4))
        ) & MASK
    return accumulator


def _verify_frame_rows(
    config: ResidentConfig, rows: list[list[int]], timestamps: list[int],
) -> tuple[dict, dict[int, int | None]]:
    """Verify every downloaded frame; rows are checksum,elapsed,frame,id,gen,segment."""
    if len(rows) != len(timestamps):
        raise ValueError("table checksum rows must match timestamp count")
    mismatches = 0
    previous_frame = -1
    digest = hashlib.sha256()
    intervals: dict[int, int | None] = {}
    for index, (checksum, _, frame, config_id, generation, segment) in enumerate(rows):
        identity = (config_id, generation)
        if frame <= previous_frame or frame >= config.frame_count:
            raise ValueError("table frame indices must increase and be below frame_count")
        if identity_at(config, frame) != (identity, segment):
            raise ValueError("stamped identity or switch segment differs from producer schedule")
        if ready_index(config, frame, identity) is None:
            raise ValueError("discarded identity emitted a timestamp")
        if segment not in intervals:
            intervals[segment] = timestamps[index] - timestamps[index - 1] if index else None
        mismatches += checksum != frame_checksum(config, frame, identity)
        digest.update(struct.pack("<4I", frame, config_id, generation, checksum))
        previous_frame = frame
    return {"checked_frames": len(rows), "mismatch_count": mismatches,
            "passed": mismatches == 0, "sha256": digest.hexdigest()}, intervals


def build_table_evidence(
    config: ResidentConfig, rows: list[list[int]], timestamps: list[int], summary: list[int],
) -> dict:
    checksum, intervals = _verify_frame_rows(config, rows, timestamps)
    switches = []
    all_discards = []
    for segment in range(len(config.switch_schedule) + 1):
        base = 20 + 4 * segment
        discards, low, high, valid = summary[base:base + 4]
        if valid not in (0, 1, 2) or (valid == 0) != (segment not in intervals):
            raise ValueError("switch interval validity differs from consumed frames")
        actual_interval = (low | high << 32) if valid == 2 else None
        if actual_interval != intervals.get(segment):
            raise ValueError("card switch interval differs from timestamp stream")
        all_discards.append(discards)
        if segment:
            frame, config_id, generation = config.switch_schedule[segment - 1]
            switches.append({
                "frame_index": frame, "identity": [config_id, generation],
                "discard_count": discards, "switch_frame_interval_ticks": actual_interval,
            })
    result = {
        "schema": "resident-table-swap-v1",
        "discard_count": summary[18], "retired_frame_count": summary[19],
        "initial_discard_count": all_discards[0], "switches": switches,
        "switch_interval_definition": "previous completion to first consumed frame in schedule segment",
        "checksum": checksum,
    }
    validate_table_evidence(config, result, len(rows))
    return result


def validate_table_evidence(config: ResidentConfig, evidence: dict, consumed: int) -> None:
    """Check persisted evidence relationships, without pretending it proves board execution."""
    validate_tables(config)
    if not isinstance(evidence, dict) or evidence.get("schema") != "resident-table-swap-v1":
        raise ValueError("missing table swap evidence")
    discarded = _count(evidence.get("discard_count"))
    initial = _count(evidence.get("initial_discard_count"))
    retired = _count(evidence.get("retired_frame_count"))
    if retired != consumed + discarded or retired > config.frame_count:
        raise ValueError("retired frames must equal consumed plus discarded")
    switches = evidence.get("switches")
    if not isinstance(switches, list) or len(switches) != len(config.switch_schedule):
        raise ValueError("switch stats must match the producer schedule")
    _validate_readiness_counts(config, switches, initial, retired)
    total = initial
    for row, (frame, config_id, generation) in zip(switches, config.switch_schedule, strict=True):
        if row.get("frame_index") != frame or row.get("identity") != [config_id, generation]:
            raise ValueError("switch stats identity differs from schedule")
        total += _count(row.get("discard_count"))
        interval = row.get("switch_frame_interval_ticks")
        if interval is not None and (type(interval) is not int or interval <= 0):
            raise ValueError("switch interval must be positive ticks or null")
    if total != discarded:
        raise ValueError("per-switch discard sum differs from total")
    _validate_checksum_evidence(evidence.get("checksum", {}), consumed)


def _count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("table swap counters must be non-negative integers")
    return value


def _validate_readiness_counts(config, switches, initial, retired) -> None:
    # Even when producer overflow hides attempted frames, a segment cannot
    # discard more identities than its modeled unready interval. A fully
    # retired no-overflow run must match the model exactly.
    starts = [0, *(row[0] for row in config.switch_schedule), config.frame_count]
    counts = [initial, *(_count(row.get("discard_count")) for row in switches)]
    identities = [config.initial_identity, *(row[1:] for row in config.switch_schedule)]
    for segment, (identity, discards) in enumerate(zip(identities, counts, strict=True)):
        start, end = starts[segment:segment + 2]
        ready = next((row[2] for row in config.profile_tables
                      if tuple(row[:2]) == tuple(identity)), end)
        expected = max(0, min(end, ready) - start)
        if discards > expected or (retired == config.frame_count and discards != expected):
            raise ValueError("discard counts disagree with modeled readiness")


def _validate_checksum_evidence(checksum, consumed) -> None:
    mismatches = _count(checksum.get("mismatch_count"))
    if (_count(checksum.get("checked_frames")) != consumed or mismatches > consumed
            or checksum.get("passed") is not (mismatches == 0)):
        raise ValueError("checksum verification counts are inconsistent")
    digest = checksum.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("checksum digest must be SHA-256")


class TableConsumerModel:
    """One-pass consumer beside RingAccounting; discarded slots still retire."""

    def __init__(self, config: ResidentConfig):
        validate_tables(config)
        self.config = config
        self.discard_count = 0
        self.discards_per_switch = [0] * len(config.switch_schedule)
        self.switch_intervals: list[int | None] = [None] * len(config.switch_schedule)
        self.timestamps: list[int] = []
        self._seen_segments: set[int] = set()

    def consume(self, frame: int, identity: tuple[int, int], timestamp: int) -> int | None:
        _, segment = identity_at(self.config, frame)
        if ready_index(self.config, frame, identity) is None:
            self.discard_count += 1
            if segment:
                self.discards_per_switch[segment - 1] += 1
            return None
        if segment not in self._seen_segments:
            if segment and self.timestamps:
                self.switch_intervals[segment - 1] = timestamp - self.timestamps[-1]
            self._seen_segments.add(segment)
        self.timestamps.append(timestamp)
        return frame_checksum(self.config, frame, identity)
