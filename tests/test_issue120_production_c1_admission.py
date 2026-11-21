"""C1-only, explicit old-B4/new-host-native admission; never identity rewriting."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from enodia.tt.bench.issue120_production import plan, runtime, sessions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/measurements/issue120-production/2026-11-18-b4"


@pytest.fixture(autouse=True)
def frozen_c1_inventory(tmp_path_factory, monkeypatch):
    """Characterize C1's original source universe, not the new P package."""
    original = runtime.SOURCE
    saved = json.loads((ROOT / sessions.C1_ADMISSION_PATH).read_bytes())["program_identity"]["python_sources_sha256"]
    source = tmp_path_factory.mktemp("frozen-c1-source")
    for name in saved:
        (source / name).write_text("# Historical inventory fixture, not executed.\n")
    (source / "native.cpp").write_bytes((original / "native.cpp").read_bytes())
    real_sha = runtime.sha
    monkeypatch.setattr(runtime, "SOURCE", source)
    monkeypatch.setattr(runtime, "sha", lambda path: saved[path.name] if path.parent == source and path.name in saved else real_sha(path))


def _receipt():
    from enodia.tt.bench.issue120_production import admission_bridge, bounded_evidence

    old = json.loads((DATA / "runtime-session-manifest.json").read_bytes())["program_identity"]
    current = copy.deepcopy(old)
    current["native_sha256"] = "a" * 64
    current["host_trace_contract"] = "issue120-host-submission-trace-v1"
    current["python_sources_sha256"] = {p.name: runtime.sha(p) for p in runtime.SOURCE.glob("*.py")}
    receipt = admission_bridge.make_receipt(current)
    assert receipt["records"] == bounded_evidence.references()
    return receipt, current, old


def test_original_b4_archives_supply_exact_old_identity_not_new_native():
    from enodia.tt.bench.issue120_production import admission_bridge

    receipt, current, old = _receipt()
    admission_bridge.validate_bridge(receipt, current)
    rows = sessions._records_from_receipt(receipt)
    assert [r["condition"] for r in rows] == ["control", "loaded"]
    assert all(r["program_identity"] == old for r in rows)
    assert all(r["program_identity"] != current for r in rows)


@pytest.mark.parametrize("field", ["native", "library", "device", "python", "scope", "archive", "image"])
def test_bridge_rejects_unapproved_identity_or_evidence_changes(field):
    from enodia.tt.bench.issue120_production import admission_bridge

    receipt, current, _ = _receipt()
    if field == "native":
        current["native_sha256"] = "b" * 64
    elif field == "library":
        receipt["evidence_program_identity"]["library_sha256"] = "b" * 64
    elif field == "device":
        receipt["evidence_program_identity"]["compile_manifest_sha256"] = "b" * 64
    elif field == "python":
        current["python_sources_sha256"]["coverage.py"] = "b" * 64
        receipt["program_identity"] = copy.deepcopy(current)
    elif field == "image":
        receipt["image"] = "different image"
    elif field == "scope":
        receipt["allowed_plan_identities"].append(plan.RUNS[3].identity)
    else:
        receipt["records"][0]["archive_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        admission_bridge.validate_bridge(receipt, current)


def test_bridge_admits_c1_only_and_rejects_production(monkeypatch):
    receipt, _, _ = _receipt()
    monkeypatch.setattr(sessions, "_committed_admission", lambda *args: receipt)
    assert sessions.admitted_specs("C") == [plan.RUNS[2]]
    with pytest.raises(ValueError, match="C1"):
        sessions.admitted_specs("P")


def test_handcrafted_c2_session_is_rejected_before_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(sessions, "SESSIONS", tmp_path)
    monkeypatch.setattr(sessions, "admitted_specs", lambda unit: [plan.RUNS[2]])
    with pytest.raises(ValueError, match="C1"):
        sessions.create_session([plan.RUNS[3]])
    assert list(tmp_path.iterdir()) == []
