"""Recompute both immutable B4 archive records without a device."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from enodia.tt.bench.issue120_production import recording

DIRECTORY = Path(__file__).resolve().parents[1] / "docs/measurements/issue120-production/2026-11-18-b4"


@pytest.mark.parametrize("case,condition,coverage,dumps", [
    ("B1", "control", "not_applicable", 7),
    ("B2", "loaded", "covered", 8),
])
def test_public_b4_archive_recomputes_without_mutating_original_bytes(tmp_path, case, condition, coverage, dumps):
    manifest = json.loads((DIRECTORY / f"{case}-manifest.json").read_bytes())
    archive_path = DIRECTORY / manifest["archive"]["file"]
    original = archive_path.read_bytes()
    assert len(original) == manifest["archive"]["byte_count"]
    assert hashlib.sha256(original).hexdigest() == manifest["archive"]["sha256"]
    with tarfile.open(archive_path) as archive:
        assert len(archive.getmembers()) == manifest["original_payload_count"] == 11
        archive.extractall(tmp_path, filter="data")
    for item in manifest["artifacts"]:
        raw = (tmp_path / item["file"]).read_bytes()
        assert len(raw) == item["byte_count"]
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    path = tmp_path / manifest["record_file"]
    before = path.read_bytes()
    record = recording.load_validated_record(path)
    assert record["schema"] == "issue120-production-v2"
    assert record["condition"] == condition
    assert record["coverage_analysis"]["coverage"] == coverage
    assert record["resident_integrity_passed"] is True
    assert record["execution_eligible"] is True
    assert record["timing_evidence"] is False
    assert record["warning_check"]["unexpected_count"] == 0
    assert record["watcher"]["dump_count"] == dumps
    assert record["power_trace_snapshot"]["errors"] == []
    assert record["power_trace_snapshot"]["coverage_complete"] is True
    assert record["environment"]["commit"] == "98739788c1a07c1e3e3ea1b6f7b41a056d002f1a"
    session_bytes = (DIRECTORY / "runtime-session-manifest.json").read_bytes()
    assert record["session_manifest_sha256"] == hashlib.sha256(session_bytes).hexdigest()
    assert path.read_bytes() == before
    assert archive_path.read_bytes() == original
