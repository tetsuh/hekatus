"""Replay actual C2 bytes and reject provenance downgrade on archive copies."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from enodia.tt.bench.issue120_production import recording, runtime

ROOT = Path(__file__).resolve().parents[1] / "docs/measurements/issue120-production"


def unpack(case, destination):
    directory = ROOT / f"2026-11-21-{case.lower()}"
    inventory = json.loads((directory / f"{case}-manifest.json").read_bytes())
    packed = (directory / inventory["archive"]["file"]).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == inventory["archive"]["sha256"]
    assert len(packed) == inventory["archive"]["byte_count"]
    with tarfile.open(directory / inventory["archive"]["file"]) as bundle:
        assert len(bundle.getmembers()) == inventory["original_payload_count"]
        bundle.extractall(destination, filter="data")
    for item in inventory["artifacts"]:
        content = (destination / item["file"]).read_bytes()
        assert len(content) == item["byte_count"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
    return destination / inventory["record_file"]


def test_actual_c2_archive_recomputes_covered_execution_and_manifest(tmp_path):
    path = unpack("C2", tmp_path)
    original = path.read_bytes()
    row = recording.load_validated_record(path)
    assert row["environment"]["commit"] == "f5d46fbfe00234acb92db1d2f9759444ae42bfbf"
    assert row["coverage_analysis"]["coverage"] == "covered"
    assert row["execution_eligible"] is True
    assert row["timing_evidence"] is True
    assert row["resident_integrity_passed"] is True
    assert row["metrics"]["frames_consumed"] == 32768
    assert row["metrics"]["paired_outlier_count"] == 0
    assert row["warning_check"]["unexpected_count"] == 0
    trace = row["native"]["host_phase_trace"]
    assert trace["final_count"] == 1572864
    assert len(trace["samples"]) == 25
    assert trace["main_enqueue_return_observation"]["committed_count"] == 24863
    manifest = row["calibration_provenance"]["session_manifest"]
    assert runtime.sha(tmp_path / manifest["file"]) == row["session_manifest_sha256"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("downgrade", ["coordinated_deletion", "null_session_digest"])
def test_actual_c2_copy_cannot_downgrade_mandatory_provenance(tmp_path, downgrade):
    path = unpack("C2", tmp_path)
    row = json.loads(path.read_bytes())
    if downgrade == "coordinated_deletion":
        del row["program_identity"]["calibration_compatibility_contract"]
        for key in ("plan_identity", "phase", "calibration_provenance"):
            del row[key]
    else:
        row["session_manifest_sha256"] = None
        row["calibration_provenance"]["session_manifest_sha256"] = None
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="calibration"):
        recording.load_validated_record(path)


def test_actual_compatible_pair_preserves_distinct_origins_and_no_authority(tmp_path, monkeypatch):
    paths = [unpack(case, tmp_path / case) for case in ("C1", "C2")]
    original = [path.read_bytes() for path in paths]
    monkeypatch.setattr(runtime, "CALIBRATION_RECORD_ROOT", tmp_path)
    result = recording.calibration_pair_analysis(paths)
    assert result["production_coverage_eligible"] is True
    assert result["execution_authorized"] is False
    assert result["current_550s_floor_unchanged"] is True
    commits = [row["harness_commit"] for row in result["records"]]
    assert commits[0] != commits[1]
    assert [path.read_bytes() for path in paths] == original
