"""Recompute causal intervals and unchanged resident metric algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from enodia.tt.bench.resident_harness import frame_interval_statistics, timestamp_digest
from enodia.tt.bench.resident_record import detect_pairs

from .artifacts import ticks
from .protocol import TransitionIndex, decode, resident_checksum


class MissingProgress(ValueError):
    """Valid observations do not contain the required fresh progress."""


def _pair(left, right, token, index: TransitionIndex) -> None:
    ls, lc = decode(*left)
    rs, rc = decode(*right)
    index.bind(ls, lc)
    index.bind(rs, rc)
    if rs <= ls or rc != token:
        raise MissingProgress("fresh CPU/DMA response not observed")


def _startup(stats, index: TransitionIndex) -> tuple[int, int, int]:
    if stats[55] != 1 or stats[56] != 1:
        raise MissingProgress("startup gate has no validated progress")
    _pair(stats[36:38], stats[38:40], 1, index)
    issued, seen, start_sent = ticks(stats, 40), ticks(stats, 42), ticks(stats, 57)
    if not issued <= ticks(stats, 44) <= seen <= start_sent:
        raise ValueError("startup response/start ordering differs")
    return ticks(stats, 44), stats[38], start_sent


def _frame(row, previous_end: int, previous_sequence: int, start_sent: int,
           index: TransitionIndex) -> None:
    if row[15] != 1:
        raise ValueError("torn/invalid progress page")
    if row[16] != 0:
        raise ValueError("ambiguous consumer clock")
    _pair(row[7:9], row[9:11], row[17], index)
    if row[7] < previous_sequence:
        raise ValueError("progress observations went backwards")
    completion, admission = ticks(row, 0), ticks(row, 5)
    issued, seen = ticks(row, 11), ticks(row, 13)
    work_start = ticks(row, 25)
    if not (previous_end <= ticks(row, 19) <= ticks(row, 21) <= issued <= admission
            <= ticks(row, 23) <= seen <= work_start <= completion):
        raise ValueError("frame causal interval ordering differs")
    if row[4] == 0 and not issued <= start_sent <= admission:
        raise ValueError("first-frame challenge/start boundary missing")
    if row[3] != completion - work_start:
        raise ValueError("resident arithmetic timing differs")


def _tail(stats, previous_end: int, previous_sequence: int, frames: int,
          index: TransitionIndex) -> None:
    if stats[54] != 1:
        raise ValueError("tail progress page invalid")
    _pair(stats[46:48], stats[48:50], frames + 2, index)
    if stats[46] < previous_sequence or not previous_end <= ticks(stats, 50) <= ticks(stats, 52):
        raise ValueError("last-frame causal endpoint missing")


def _accounting(native: dict, frames: int) -> bool:
    ps, cs = native["producer_stats"], native["consumer_stats"]
    return (ps[0], ps[1], ps[2], ps[3], ps[4], ps[11], ps[12], cs[1], cs[2], cs[12], cs[35]) == (
        0, frames, 0, frames, 0, 0, 0, frames, 0, 0, 0)


@dataclass
class Analysis:
    loaded: bool
    expected: int
    state: str = "covered"
    reason: str | None = None
    integrity: bool = True
    verified: int = 0
    inside: int = 0
    tail_ok: bool = False
    timestamps: list[int] = field(default_factory=list)
    work: list[int] = field(default_factory=list)

    def fail(self, reason: str, malformed: bool = True):
        kind = "not_evaluable" if malformed else "not_covered"
        if self.reason is None or self.state == "not_covered" and malformed:
            self.state, self.reason = kind, reason

    def coverage(self) -> dict:
        pages = self.verified + int(self.tail_ok)
        return {"schema": "issue120-production-coverage-v1",
                "coverage": self.state if self.loaded else "not_applicable",
                "frame_count": self.expected, "verified_frames": self.verified,
                "challenge_interval_transfers_lower_bound": pages if self.loaded else 0,
                "challenge_interval_bytes_lower_bound": pages * 4096 if self.loaded else 0,
                "first_through_last_transfers_lower_bound": self.inside if self.loaded else 0,
                "first_through_last_bytes_lower_bound": self.inside * 4096 if self.loaded else 0,
                "first_through_last_cpu_operations_lower_bound": self.inside * 32768 if self.loaded else 0,
                "byte_scope": "fresh_challenge_issue_to_scan_completion_not_scan_end_to_scan_end",
                "reason": self.reason if self.loaded else None}

    def metrics(self, native: dict) -> dict:
        timestamps = self.timestamps
        intervals = [b - a for a, b in pairwise(timestamps)]
        monotonic = all(x > 0 for x in intervals)
        statistics = frame_interval_statistics(intervals) if monotonic else None
        pairs = detect_pairs(timestamps) if monotonic else []
        host = native.get("host_cpu_work", {})
        return {"frame_intervals": statistics,
                "paired_outliers": pairs,
                "paired_outlier_count": len(pairs) if monotonic else None,
                "raw_timestamp_digest": timestamp_digest(timestamps),
                "overflow_count": native["producer_stats"][0],
                "frames_dropped": native["producer_stats"][4],
                "frames_attempted": native["producer_stats"][3],
                "frames_produced": native["producer_stats"][1],
                "frames_consumed": native["consumer_stats"][1],
                "frames_aborted": native["producer_stats"][3] - native["producer_stats"][1] - native["producer_stats"][4],
                "startup_ticks": ticks(native["consumer_stats"], 4),
                "arithmetic_min_ticks": min(self.work, default=None),
                "arithmetic_max_ticks": max(self.work, default=None),
                "checksum_passed": self.integrity, "latency_threshold": None,
                "load_accounting": {"submitted_transfers": host.get("transfers", 0),
                                    "submitted_bytes": host.get("transfers", 0) * 4096,
                                    "cpu_operations": host.get("operations", 0),
                                    "targeted_load_reads": host.get("reads", 0),
                                    "targeted_load_read_bytes": host.get("reads", 0) * 4,
                                    "separate_terminal_observer_reads": native.get("observer_reads", 0)}}


def _inspect_row(row, number, accumulator, analysis: Analysis) -> tuple[int, bool]:
    expected = resident_checksum(accumulator, number, 64)
    good = row[4] == number and row[18] == 64 and row[2] == expected
    good &= row[16] == 0 and ticks(row, 25) <= ticks(row, 0)
    good &= row[3] == ticks(row, 0) - ticks(row, 25)
    if analysis.timestamps and ticks(row, 0) <= analysis.timestamps[-1]:
        good = False
    if not good:
        analysis.integrity = False
        analysis.fail("frame identity/checksum/work/clock differs")
    analysis.timestamps.append(ticks(row, 0))
    analysis.work.append(row[3])
    return expected, good


def _attempt(check, analysis: Analysis, *args):
    try:
        return check(*args)
    except MissingProgress as error:
        analysis.fail(str(error), False)
    except ValueError as error:
        analysis.fail(str(error))
    return None


def _first_baseline(row, stats) -> None:
    if (tuple(row[7:9]), ticks(row, 19), ticks(row, 21)) != (
            tuple(stats[38:40]), ticks(stats, 44), ticks(stats, 42)):
        raise ValueError("first-frame baseline differs from the startup gate observation")


def _check_frame_row(row, number, frames, stats, previous_end, previous_sequence, start_sent, index):
    if number >= frames or row[17] != number + 2:
        raise ValueError("frame challenge/count differs")
    if number == 0:
        _first_baseline(row, stats)
    _frame(row, previous_end, previous_sequence, start_sent, index)
    return True


def _finish_tail(stats, previous_end, previous_sequence, frames, index):
    _tail(stats, previous_end, previous_sequence, frames, index)
    return True


def analyze(rows, native: dict, transitions: TransitionIndex | None) -> dict:
    frames = native["frame_count"]
    loaded = native["condition"] == "loaded"
    analysis = Analysis(loaded, frames)
    if not _accounting(native, frames):
        analysis.integrity = False
        analysis.fail("incomplete resident stream or failure counters")
    stats = native["consumer_stats"]
    position = (0, 0, 0)
    if loaded:
        if transitions is None:
            raise ValueError("loaded analysis requires the committed association map")
        position = _attempt(_startup, analysis, stats, transitions) or position
    previous_end, previous_sequence, start_sent = position
    accumulator, count = 0, 0
    for number, row in enumerate(rows):
        count += 1
        accumulator, good = _inspect_row(row, number, accumulator, analysis)
        checked = _attempt(_check_frame_row, analysis, row, number, frames, stats, previous_end,
                           previous_sequence, start_sent, transitions) if loaded else False
        if checked and good:
            analysis.verified += 1
            analysis.inside += int(number > 0)
        previous_end, previous_sequence = ticks(row, 0), row[9]
    if count != frames or native["checksum_verified_frames"] != frames:
        analysis.integrity = False
        analysis.fail("missing frame or checksum observations")
    if loaded:
        analysis.tail_ok = bool(_attempt(_finish_tail, analysis, stats, previous_end,
                                        previous_sequence, frames, transitions))
    return {"coverage_analysis": analysis.coverage(), "metrics": analysis.metrics(native),
            "resident_integrity_passed": analysis.integrity}
