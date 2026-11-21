"""Version-dispatched P sessions must not use the C2-only launch admission."""
from __future__ import annotations

import copy

import pytest

from enodia.tt.bench.issue120_production import production_admission as production
from enodia.tt.bench.issue120_production import runtime, sessions


@pytest.fixture
def receipt(monkeypatch):
    report = {"schema": "issue120-production-host-postprocess-v1", "fixture_only": True,
              "frames": 500000, "passed": True, "peak_rss_bytes": 200000000,
              "charged_host_seconds": 8.0, "launcher_reread_seconds": 6.0,
              "device_retrieval_measured": False}
    monkeypatch.setattr(production, "postprocess_proof", lambda: {"file": "unit-fixture", "sha256": "a" * 64, "report": report})
    current = production.fixture_identity()
    monkeypatch.setattr(runtime, "program_identity", lambda: current)
    return production.make_receipt(current)


def test_explicit_versioned_production_dispatch_does_not_reinterpret_bounded(receipt, monkeypatch):
    kinds = []

    def admission(kind, bindings=None):
        kinds.append(kind)
        if kind != "production":
            pytest.fail("new P must not reinterpret C2-only admission as live current")
        return receipt

    monkeypatch.setattr(sessions, "_committed_admission", admission)
    specs = sessions.admitted_specs("P")
    assert [s.repeat_count for s in specs] == [production.selection()["fixed_repetitions"]] * 2
    assert kinds == ["production"]


def test_compatible_session_rejects_head_drift_without_opening_devices(receipt, tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS", tmp_path)
    monkeypatch.setattr(sessions, "_committed_admission", lambda *a: receipt)
    monkeypatch.setattr(sessions, "admission_hashes", lambda phase: {"production": "a" * 64})
    monkeypatch.setattr(sessions, "current_head", lambda: "a" * 40, raising=False)
    manifest = sessions.create_session(production.specs(receipt))
    assert manifest["harness_commit"] == "a" * 40
    monkeypatch.setattr(sessions, "current_head", lambda: "b" * 40)
    with pytest.raises(ValueError, match="head|revision"):
        sessions.read_session(manifest["session_id"])


def test_compatible_session_rejects_boot_drift(receipt, tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS", tmp_path)
    monkeypatch.setattr(sessions, "_committed_admission", lambda *a: receipt)
    monkeypatch.setattr(sessions, "admission_hashes", lambda phase: {"production": "a" * 64})
    manifest = sessions.create_session(production.specs(receipt))
    monkeypatch.setattr(sessions, "boot_fingerprint", lambda: "b" * 64)
    with pytest.raises(ValueError):
        sessions.read_session(manifest["session_id"])


def test_changed_selection_invalidates_existing_receipt(receipt, monkeypatch):
    selected = copy.deepcopy(production.selection())
    selected["target_worker_seconds"] += 1
    monkeypatch.setattr(production, "selection", lambda: selected)
    with pytest.raises(ValueError):
        production.validate_receipt(receipt, receipt["program_identity"])


def test_production_session_must_match_entire_admitted_pair(receipt, tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS", tmp_path)
    monkeypatch.setattr(sessions, "_committed_admission", lambda *a: receipt)
    monkeypatch.setattr(sessions, "admission_hashes", lambda phase: {"production": "a" * 64})
    incomplete = production.specs(receipt)[:1]
    with pytest.raises(ValueError, match="production"):
        sessions.create_session(incomplete)
    assert not list(tmp_path.iterdir())
