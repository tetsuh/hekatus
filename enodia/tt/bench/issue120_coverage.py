"""Card-clock proof of fixed host work across an actual resident frame stream.

The proof uses consistent payload generations verified on the consumer core.
Host envelopes and submitted command counts are deliberately not inputs.
"""

from __future__ import annotations

from typing import Any

FRAME_COUNT = 4_096
FRAME_PERIOD_TICKS = 1_350_000
FRAME_WORK_ITERATIONS = 64
PROGRESS_BYTES = 4_096
TRANSFER_REPEATS = 262_144
CPU_ROUNDS_PER_TRANSFER = 32
CPU_WORDS_PER_ROUND = 1_024
MAX_PROGRESS_GENERATION = (1 << 19) - 1
MAX_CHALLENGE = (1 << 13) - 1


def pack_progress(sequence: int, challenge: int) -> int:
    """Injective 32-bit wire token; XOR of the two fields would collide."""
    if not _whole(sequence) or not 1 <= sequence <= MAX_PROGRESS_GENERATION:
        raise ValueError("progress generation exceeds its 19-bit wire field")
    if not _whole(challenge) or challenge > MAX_CHALLENGE:
        raise ValueError("CPU challenge exceeds its 13-bit wire field")
    return (challenge << 19) | sequence


def expected_frame_checksum(frame: int) -> int:
    accumulator = 0
    for work in range(FRAME_WORK_ITERATIONS):
        word = ((frame + work) & 0x7FF) | (((3 * frame + 5 * work) & 0x7FF) << 16)
        if work == 0:
            word = frame
        accumulator = ((accumulator * 33) ^ word ^ (work + frame)) & 0xFFFFFFFF
    return accumulator


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _header_issue(evidence: dict[str, Any]) -> str | None:
    fields = ("frame_count", "attempted", "produced", "consumed", "overflow", "aborted", "load_start_ticks", "load_end_ticks", "startup_response_challenge", "repeat_count")
    if not all(_whole(evidence.get(field)) for field in fields):
        return "missing or invalid frame/clock accounting"
    count = evidence["frame_count"]
    checks = (
        (1 <= count <= FRAME_COUNT, "frame count outside the bounded probe"),
        (1 <= evidence["repeat_count"] <= MAX_PROGRESS_GENERATION, "invalid fixed transfer count"),
        (evidence.get("clock") == "consumer_low32_wrap_tracked", "clock is not the designated consumer clock"),
        (evidence.get("clock_gap_ambiguous") is False, "card clock sampling is ambiguous"),
        (evidence.get("producer_gated_after_load_observation") is True, "producer started without card-side load observation"),
        (evidence.get("startup_response_challenge") == 1, "no causal CPU response before frames"),
        (all(evidence[field] == count for field in ("attempted", "produced", "consumed")), "incomplete real frame stream"),
        (evidence["overflow"] == 0 and evidence["aborted"] == 0, "overflow or aborted frames"),
        (evidence.get("transfer_bytes") == PROGRESS_BYTES, "unexpected transfer payload size"),
        (isinstance(evidence.get("frames"), list), "missing frame stream"),
    )
    return _first_failed(checks)


def _first_failed(checks: tuple[tuple[bool, str], ...]) -> str | None:
    return next((reason for passed, reason in checks if not passed), None)


def _frame_issue(frame: Any, index: int, previous_end: int, previous_sequence: int, limit: int) -> str | None:
    if not isinstance(frame, dict):
        return "frame row is not an object"
    fields = ("frame_index", "start_ticks", "end_ticks", "left_ticks", "right_ticks", "left_sequence", "right_sequence", "payload_frame_id", "checksum", "work_iterations", "issued_challenge", "response_challenge")
    if not all(_whole(frame.get(field)) for field in fields):
        return "missing frame or progress endpoints"
    checks = (
        (frame["frame_index"] == index and frame["payload_frame_id"] == index, "missing, duplicate or reordered frame"),
        (previous_end <= frame["start_ticks"] <= frame["end_ticks"], "reversed frame interval"),
        (frame["right_ticks"] >= frame["start_ticks"], "response observed before frame admission"),
        (previous_end <= frame["left_ticks"] <= frame["right_ticks"] <= frame["end_ticks"], "progress bracket outside frame interval"),
        (previous_sequence <= frame["left_sequence"] < frame["right_sequence"] <= limit, "stalled or reversed transfer progress"),
        (frame.get("progress_valid") is True, "torn or invalid progress payload"),
        (frame.get("issued_challenge") == index + 2 and frame.get("response_challenge") == index + 2, "CPU work was not caused by this frame's card challenge"),
        (frame["checksum"] == expected_frame_checksum(index), "frame checksum mismatch"),
        (frame["work_iterations"] == FRAME_WORK_ITERATIONS, "frame work changed"),
    )
    return _first_failed(checks)


def _tail_issue(tail: Any, last_end: int, last_sequence: int, load_end: int, challenge: int, limit: int) -> str | None:
    if not isinstance(tail, dict):
        return "missing post-frame progress"
    fields = ("left_ticks", "right_ticks", "left_sequence", "right_sequence", "issued_challenge", "response_challenge")
    if not all(_whole(tail.get(field)) for field in fields):
        return "invalid post-frame progress endpoints"
    checks = (
        (last_end <= tail["left_ticks"] <= tail["right_ticks"] <= load_end, "load ended before the last frame"),
        (last_sequence <= tail["left_sequence"] < tail["right_sequence"] <= limit, "no transfer progress after the last frame"),
        (tail.get("progress_valid") is True, "invalid post-frame payload"),
        (tail.get("issued_challenge") == challenge and tail.get("response_challenge") == challenge, "no causal CPU work after the last frame"),
    )
    return _first_failed(checks)


def _result(status: str, count: int, verified: int, completions: int, in_frames: int, reason: str | None) -> dict[str, Any]:
    return {
        "coverage": status, "frame_count": count, "verified_frames": verified,
        "completed_bytes_lower_bound": completions * PROGRESS_BYTES,
        "completed_bytes_during_frames_lower_bound": in_frames * PROGRESS_BYTES,
        "reason": reason,
    }


def evaluate_frame_coverage(evidence: Any) -> dict[str, Any]:
    """Recompute coverage; never trust a declared covered flag or host envelope."""
    if not isinstance(evidence, dict):
        return _result("not_evaluable", 0, 0, 0, 0, "frame evidence must be an object")
    issue = _header_issue(evidence)
    if issue:
        return _result("not_evaluable", 0, 0, 0, 0, issue)
    count = evidence["frame_count"]
    frames = evidence["frames"]
    if len(frames) != count:
        return _result("not_evaluable", count, 0, 0, 0, "frame stream length differs from accounting")
    return _evaluate_frames(evidence, frames)


def _evaluate_frames(evidence: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    previous_end = evidence["load_start_ticks"]
    previous_sequence = 1
    completed = 0
    in_frames = 0
    for index, frame in enumerate(frames):
        issue = _frame_issue(frame, index, previous_end, previous_sequence, evidence["repeat_count"])
        if issue:
            return _result("not_covered", len(frames), index, completed, in_frames, issue)
        # A newer consistent generation witnesses at least one transfer. Do not
        # assume that unobserved intermediate generations also completed here.
        completed += 1
        if index > 0:
            in_frames += 1
        previous_end = frame["end_ticks"]
        previous_sequence = frame["right_sequence"]
    return _evaluate_boundaries(evidence, previous_end, previous_sequence, completed, in_frames)


def _evaluate_boundaries(evidence: dict[str, Any], last_end: int, last_sequence: int, completed: int, in_frames: int) -> dict[str, Any]:
    count = evidence["frame_count"]
    first = evidence["frames"][0]
    if evidence["load_start_ticks"] > first["start_ticks"]:
        return _result("not_covered", count, count, completed, in_frames, "load started after the first frame")
    tail = evidence.get("tail")
    issue = _tail_issue(tail, last_end, last_sequence, evidence["load_end_ticks"], count + 2, evidence["repeat_count"])
    if issue:
        return _result("not_covered", count, count, completed, in_frames, issue)
    completed += 1
    return _result("covered", count, count, completed, in_frames, None)
