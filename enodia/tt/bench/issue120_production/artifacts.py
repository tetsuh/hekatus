"""Compact fixed-frame evidence; complete size/hash validation before use."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

HEADER = struct.Struct("<8sIIII")
FRAME = struct.Struct("<32I")
TRANSITION = struct.Struct("<2I")


@dataclass(frozen=True)
class BinaryEvidence:
    path: Path
    count: int
    bound: int
    row: struct.Struct
    sha256: str

    def rows(self):
        with self.path.open("rb") as handle:
            if len(handle.read(HEADER.size)) != HEADER.size:
                raise ValueError("binary evidence header disappeared")
            remaining = self.count
            while remaining:
                count = min(4096, remaining)
                raw = handle.read(count * self.row.size)
                if len(raw) != count * self.row.size:
                    raise ValueError("binary evidence truncated")
                yield from self.row.iter_unpack(raw)
                remaining -= count
            if handle.read(1):
                raise ValueError("binary evidence has a trailing suffix")

    def metadata(self) -> dict:
        return {"file": self.path.name, "byte_count": HEADER.size + self.count * self.row.size,
                "sha256": self.sha256, "row_count": self.count,
                "word_count": self.row.size // 4, "bound": self.bound,
                "endianness": "little", "version": 1}


def read_evidence(directory: Path, filename: str, *, frames: int, emitted: int) -> BinaryEvidence:
    formats = {"frames.bin": (b"IP120F1\0", FRAME, frames),
               "transitions.bin": (b"IP120T1\0", TRANSITION, frames + 3)}
    if filename not in formats:
        raise ValueError("unsupported evidence filename")
    if directory.is_symlink() or directory.resolve() != directory.absolute():
        raise ValueError("evidence directory is redirected")
    return _read(directory / filename, formats[filename], frames=frames, emitted=emitted)


def read_public_evidence(directory: Path, run_id: str, kind: str, *, frames: int, emitted: int) -> BinaryEvidence:
    from enodia.tt.bench.run_binding import validate_run_id

    validate_run_id(run_id)
    formats = {"frames": (b"IP120F1\0", FRAME, frames),
               "transitions": (b"IP120T1\0", TRANSITION, frames + 3)}
    if kind not in formats or directory.resolve() != directory.absolute():
        raise ValueError("invalid public evidence kind/root")
    return _read(directory / f"results-{run_id}-{kind}.bin", formats[kind], frames=frames, emitted=emitted)


def _read(path: Path, specification, *, frames: int, emitted: int) -> BinaryEvidence:
    if path.is_symlink() or not path.is_file():
        raise ValueError("binary evidence must be a regular non-symlink file")
    magic, row, limit = specification
    with path.open("rb") as handle:
        raw = handle.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("binary evidence header missing")
        got_magic, version, words, count, bound = HEADER.unpack(raw)
        if (got_magic, version, words) != (magic, 1, row.size // 4) or count > limit:
            raise ValueError("binary evidence schema/count differs")
        expected_bound = frames if row.size == FRAME.size else emitted
        if bound != expected_bound or path.stat().st_size != HEADER.size + count * row.size:
            raise ValueError("binary evidence bound/size differs")
        digest = hashlib.sha256(raw)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return BinaryEvidence(path, count, bound, row, digest.hexdigest())


def ticks(words, offset: int) -> int:
    return words[offset] | (words[offset + 1] << 32)
