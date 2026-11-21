"""Replay the actual Watcher-off C1 with its immutable host submission trace."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from enodia.tt.bench.issue120_production import recording

ARCHIVE_ROOT = Path(__file__).resolve().parents[1] / "docs/measurements/issue120-production/2026-11-21-c1"


def test_c1_archive_preserves_current_trace_and_recomputes_covered_execution(tmp_path):
    inventory = json.loads((ARCHIVE_ROOT / "C1-manifest.json").read_bytes())
    packed = (ARCHIVE_ROOT / inventory["archive"]["file"]).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == inventory["archive"]["sha256"]
    assert len(packed) == inventory["archive"]["byte_count"]
    with tarfile.open(ARCHIVE_ROOT / inventory["archive"]["file"]) as bundle:
        assert len(bundle.getmembers()) == 10
        bundle.extractall(tmp_path, filter="data")
    expected = {item["file"]: item for item in inventory["artifacts"]}
    for name, item in expected.items():
        content = (tmp_path / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
        assert len(content) == item["byte_count"]
    primary = tmp_path / inventory["record_file"]
    original = primary.read_bytes()
    recovered = recording.load_validated_record(primary)
    assert recovered["plan_identity"] == "production-v1-calibration-small"
    assert recovered["environment"]["commit"] == "b38095571dd62742a161662e5d166fafa75f5b41"
    assert recovered["program_identity"]["native_sha256"] == "0a36ed6011a8cb71773545751f98767961e1392ffb16244f2e07e4fe709234e6"
    assert recovered["coverage_analysis"]["coverage"] == "covered"
    assert recovered["execution_eligible"] is True
    assert recovered["timing_evidence"] is True
    assert recovered["resident_integrity_passed"] is True
    assert recovered["watcher"]["requested"] is False
    assert recovered["warning_check"]["unexpected_count"] == 0
    assert recovered["metrics"]["frames_consumed"] == 16384
    trace = recovered["native"]["host_phase_trace"]
    assert trace["final_count"] == 1048576
    assert [sample["committed_count"] for sample in trace["samples"]] == [1, *range(65536, 1048577, 65536)]
    assert trace["main_enqueue_return_observation"]["committed_count"] == 24974
    assert primary.read_bytes() == original
    assert (ARCHIVE_ROOT / inventory["archive"]["file"]).read_bytes() == packed
