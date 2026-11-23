"""Replay the two pinned B4 archives without rewriting original evidence."""
from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

from . import runtime

DIRECTORY = "docs/measurements/issue120-production/2026-11-18-b4"
HARNESS = "98739788c1a07c1e3e3ea1b6f7b41a056d002f1a"
SESSION = "8caa5d94ed7443d9982606ea45072452"
SESSION_SHA = "c8a54a31622e64c3f3f9582c8a037c040185536a6e77e3b6c0988ed4a9f17f25"
PINS = (
    ("B1", "20261118T1445539271829941429764960Z",
     "e4652097838ffd712f7b85a288706cfe56f16aecb59c7e430ea6f8e1f8a3370e",
     "871161088bd3260dd9f3d785c1c31dbdc882f28ac1088adf97bc187bbb07fc26"),
    ("B2", "20261118T14472425729255814391413846Z",
     "e3d1a23a016bc27b3ab7568deb542955c12065f861d3552c73be0f545962f9e9",
     "e1ab9c9b84d4cefaeab958fc2c2bb2906a627776e7ea436db2dd6f43a35c8e04"),
)
MAX_BYTES = 64 * 1024 * 1024


def _path(relative: str) -> Path:
    return runtime.fixed_path(runtime.ROOT / relative, runtime.ROOT / "docs/measurements")


def _session() -> dict:
    path = _path(DIRECTORY + "/runtime-session-manifest.json")
    if runtime.sha(path) != SESSION_SHA:
        raise ValueError("B4 session manifest differs")
    session = json.loads(path.read_bytes())
    if session["session_id"] != SESSION or session["harness_commit"] != HARNESS:
        raise ValueError("B4 evidence source differs")
    return session


def evidence_identity() -> dict:
    return _session()["program_identity"]


def evidence_image() -> str:
    return _session()["image"]


def references() -> list[dict]:
    refs = []
    for label, identity, archive_sha, record_sha in PINS:
        manifest = f"{DIRECTORY}/{label}-manifest.json"
        refs.append({"case": label, "archive": f"{DIRECTORY}/{label}-evidence.tar.gz",
                     "archive_sha256": archive_sha, "manifest": manifest,
                     "manifest_sha256": runtime.sha(_path(manifest)),
                     "record_file": f"results-{identity}.json", "record_sha256": record_sha})
    return refs


def validate_references(refs) -> None:
    if refs != references():
        raise ValueError("admission does not bind the original B4 evidence pins")


def _trusted_identity(case: str) -> str:
    return next(pin[1] for pin in PINS if pin[0] == case)


def _trusted_names(case: str) -> tuple[str, ...]:
    identity = _trusted_identity(case)
    suffixes = (".json", "-frames.bin", "-transitions.bin", "-watcher.log", "-native-result.json",
                "-native-execution.json", "-native-stdout.bin", "-native-stderr.bin")
    return (f"env-{identity}.json", f"power-{identity}.csv", f"power-trace-snapshot-{identity}-1.csv",
            *(f"results-{identity}{suffix}" for suffix in suffixes))


def _copy_members(archive: tarfile.TarFile, directory: Path, facts: dict, names: tuple[str, ...]) -> None:
    members, total = {}, 0
    for member in archive:
        if (not member.isfile() or member.name not in facts or member.name in members or
                Path(member.name).name != member.name or member.name in (".", "..")):
            raise ValueError("B4 archive has an unexpected/duplicate/redirected member")
        members[member.name] = member
        total += member.size
        if not 0 <= member.size <= MAX_BYTES or total > MAX_BYTES or len(members) > 11:
            raise ValueError("B4 archive extraction bound exceeded")
    if set(members) != set(facts) or set(facts) != set(names) or len(members) != 11:
        raise ValueError("original B4 archive is incomplete")
    # Filesystem names come only from fixed independent attempt pins, never tar metadata.
    for name in names:
        path = directory / name
        with archive.extractfile(members[name]) as source, path.open("xb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        if path.stat().st_size != facts[name]["byte_count"] or runtime.sha(path) != facts[name]["sha256"]:
            raise ValueError("original B4 payload fingerprint differs")


def _load_one(ref: dict) -> dict:
    from .recording import load_validated_record

    path = _path(ref["archive"])
    if path.stat().st_size > MAX_BYTES or runtime.sha(path) != ref["archive_sha256"]:
        raise ValueError("original B4 archive fingerprint differs")
    manifest_path = _path(ref["manifest"])
    if runtime.sha(manifest_path) != ref["manifest_sha256"]:
        raise ValueError("B4 artifact manifest fingerprint differs")
    manifest = json.loads(manifest_path.read_bytes())
    facts = {item["file"]: item for item in manifest["artifacts"]}
    if len(facts) != 11 or manifest["record_sha256"] != ref["record_sha256"]:
        raise ValueError("B4 primary artifact binding differs")
    with tempfile.TemporaryDirectory(prefix="issue120-bounded-") as temporary:
        directory = Path(temporary)
        with tarfile.open(path, "r:gz") as archive:
            _copy_members(archive, directory, facts, _trusted_names(ref["case"]))
        record_path = directory / f"results-{_trusted_identity(ref['case'])}.json"
        if runtime.sha(record_path) != ref["record_sha256"]:
            raise ValueError("B4 primary record differs")
        record = load_validated_record(record_path)
    if (record["schema"] != "issue120-production-v2" or record["environment"]["commit"] != HARNESS or
            record["session_id"] != SESSION or record["session_manifest_sha256"] != SESSION_SHA):
        raise ValueError("B4 measured source/session differs")
    return record


def load_records(refs: list[dict]) -> list[dict]:
    validate_references(refs)
    rows = [_load_one(ref) for ref in refs]
    expected = evidence_identity()
    if any(row["program_identity"] != expected for row in rows):
        raise ValueError("B4 records do not have their original program identity")
    return rows
