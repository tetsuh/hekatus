"""Wide progress words and the bounded, committed host association table."""
from __future__ import annotations

from array import array
from bisect import bisect_right

UINT32_MAX = 2**32 - 1
PATTERN = 0x120C0000
PAGE_WORDS = 1024
SCHEMA = "issue120-wide-progress-v1"


def _integer(value: object, low: int, high: int, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"invalid {field}")
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"invalid {field}")
    return number


def admit_domain(*, frame_count: int, repeat_count: int) -> None:
    _integer(frame_count, 1, 500000, "frame count")
    _integer(repeat_count, 1, UINT32_MAX - frame_count - 2, "sequence domain")


def encode(sequence: int, challenge: int) -> tuple[int, int]:
    _integer(sequence, 0, UINT32_MAX, "sequence")
    _integer(challenge, 0, UINT32_MAX, "challenge")
    if sequence + challenge > UINT32_MAX or (sequence == 0 and challenge != 0):
        raise ValueError("wire sum would wrap or collide with initialization")
    return sequence, sequence + challenge


def decode(sequence: int, summed: int) -> tuple[int, int]:
    _integer(sequence, 0, UINT32_MAX, "sequence")
    _integer(summed, sequence, UINT32_MAX, "summed challenge")
    return sequence, summed - sequence


def word(sequence: int, summed: int, index: int) -> int:
    if index == 0:
        return sequence
    if index == 1:
        return summed
    return (sequence if index % 2 == 0 else summed) ^ index ^ PATTERN


def payload(sequence: int, challenge: int) -> list[int]:
    sequence, summed = encode(sequence, challenge)
    return [word(sequence, summed, i) for i in range(PAGE_WORDS)]


class TransitionIndex:
    """O(F) uint32 storage, never one entry per transfer."""

    def __init__(self, entries, *, frame_count: int, repeat_count: int):
        admit_domain(frame_count=frame_count, repeat_count=repeat_count)
        self.frame_count = frame_count
        self.repeat_count = repeat_count
        self.starts = array("I")
        self.challenges = array("I")
        for entry in entries:
            if len(entry) != 2:
                raise ValueError("invalid transition row")
            first = _integer(entry[0], 1, repeat_count, "transition sequence")
            challenge = _integer(entry[1], 0, frame_count + 2, "transition challenge")
            if not self.starts and first != 1:
                raise ValueError("transition map must start at sequence1")
            if self.starts and (first <= self.starts[-1] or challenge <= self.challenges[-1]):
                raise ValueError("transition rows must increase strictly")
            if len(self.starts) >= frame_count + 3:
                raise ValueError("transition map exceeds fixed frame capacity")
            self.starts.append(first)
            self.challenges.append(challenge)
        if not self.starts:
            raise ValueError("transition map is empty")

    def challenge_for(self, sequence: int) -> int:
        _integer(sequence, 1, self.repeat_count, "observed sequence")
        return self.challenges[bisect_right(self.starts, sequence) - 1]

    def bind(self, sequence: int, challenge: int) -> None:
        if self.challenge_for(sequence) != challenge:
            raise ValueError("observed tuple differs from committed host association")


def validate_transitions(entries, *, frame_count: int, repeat_count: int) -> TransitionIndex:
    return TransitionIndex(entries, frame_count=frame_count, repeat_count=repeat_count)


def observe(words, entries, *, frame_count: int, repeat_count: int) -> tuple[int, int]:
    if len(words) != PAGE_WORDS:
        raise ValueError("incomplete progress page")
    sequence, challenge = decode(words[0], words[1])
    _integer(sequence, 1, repeat_count, "observed sequence")
    _integer(challenge, 0, frame_count + 2, "observed challenge")
    if any(type(value) is not int or value != word(words[0], words[1], i)
           for i, value in enumerate(words)):
        raise ValueError("torn progress page")
    index = entries if isinstance(entries, TransitionIndex) else validate_transitions(
        entries, frame_count=frame_count, repeat_count=repeat_count)
    if index.frame_count != frame_count or index.repeat_count != repeat_count:
        raise ValueError("transition map belongs to another domain")
    index.bind(sequence, challenge)
    return sequence, challenge


def resident_checksum(accumulator: int, frame: int, work_per_frame: int) -> int:
    for work in range(work_per_frame):
        word_index = work % PAGE_WORDS
        i = (frame + word_index) & 0x7FF
        q = (3 * frame + 5 * word_index) & 0x7FF
        accumulator = (((accumulator * 33) & UINT32_MAX) ^ (i | (q << 16))
                       ^ (work + frame))
    return accumulator
