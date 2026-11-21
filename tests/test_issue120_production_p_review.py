"""Regression boundaries from the pre-hardware production provenance review."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from enodia.tt.bench.issue120_production import plan, records, runtime, sessions
from enodia.tt.bench.issue120_production import production_admission as production


def valid_origin():
    config = plan.program_configuration(replace(plan.RUNS[4], repeat_count=production.selection()["fixed_repetitions"]))
    program = {"production_admission_contract": runtime.PRODUCTION_CONTRACT}
    manifest = {"session_id": "a" * 32, "program_identity": program, "harness_commit": "a" * 40,
                "host_boot_id_sha256": "c" * 64, "production_contract": runtime.PRODUCTION_CONTRACT,
                "runs": {plan.RUNS[4].identity: {"configuration": config, "configuration_sha256": plan.configuration_digest(config)}}}
    row = {"schema": records.SCHEMA, "phase": "production", "condition": "control", "plan_identity": plan.RUNS[4].identity,
           "session_id": manifest["session_id"], "session_manifest_sha256": "a" * 64,
           "host_boot_id_sha256": manifest["host_boot_id_sha256"], "program_identity": copy.deepcopy(program),
           "environment": {"commit": manifest["harness_commit"]}, "configuration": config,
           "configuration_sha256": plan.configuration_digest(config), "native": {"frame_count": 500000},
           "resident_integrity_passed": True, "execution_eligible": True, "timing_evidence": True,
           "coverage_analysis": {"coverage": "not_applicable"}, "metrics": {"paired_outlier_count": 0}}
    return manifest, row


@pytest.mark.parametrize("field", ["condition", "phase", "plan_identity", "configuration", "configuration_sha256"])
def test_new_p_session_cannot_release_nonproduction_p1(field):
    manifest, row = valid_origin()
    row[field] = "historical"
    with (patch.object(sessions, "read_session", return_value=(manifest, "a" * 64)), pytest.raises(ValueError)):
        production.control_release(row, manifest["session_id"])


def test_authentic_control_role_still_passes():
    manifest, row = valid_origin()
    with patch.object(sessions, "read_session", return_value=(manifest, "a" * 64)):
        production.control_release(row, manifest["session_id"])


@pytest.mark.parametrize("field", ["serial", "pci", "firmware", "kmd", "board_model", "image_digest", "cpu_model"])
def test_authenticated_environment_reduction_does_not_follow_primary_claims(field):
    raw = {"board": {"board_type": "p150a", "board_id": "000004033191411c", "bus_id": "0000:06:00.0"},
           "firmware": {"fw_bundle_version": "19.6.0.0"}, "image": runtime.PINNED_IMAGE,
           "kmd_version": "2.12.0", "harness_commit": "a" * 40}
    expected = production.environment_identity(raw, {"host_cpu_model": "synthetic CPU"})
    claimed = expected.copy()
    claimed[field] = "tampered"
    assert claimed != production.environment_identity(raw, {"host_cpu_model": "synthetic CPU"})


def test_production_program_cannot_delete_session_contract_to_skip_container_gates(tmp_path, monkeypatch):
    sid = "a" * 32
    directory = tmp_path / sid
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"schema": "issue120-production-session-v1", "session_id": sid,
        "program_identity": {"production_admission_contract": runtime.PRODUCTION_CONTRACT},
        "runs": {plan.RUNS[4].identity: {}}, "host_boot_id_sha256": "old"}))
    monkeypatch.setattr(sessions, "SESSIONS", tmp_path)
    monkeypatch.setattr(runtime, "ROOT", Path("/work"))
    monkeypatch.setattr(sessions, "boot_fingerprint", lambda: "new")
    with pytest.raises(ValueError, match="contract|boot"):
        sessions.read_session(sid)


def test_native_fixture_flag_cannot_be_deleted_only_from_primary(monkeypatch):
    from enodia.tt.bench.issue120_production import recording

    row = {"native": {"fixture_only": True}, "program_identity": {"production_admission_contract": runtime.PRODUCTION_CONTRACT},
           "production_provenance": {"receipt": {}}}
    output = runtime.ROOT / "out/issue120-production/sessions" / ("a" * 32) / "cases" / plan.RUNS[4].identity
    with pytest.raises(ValueError, match="fixture"):
        recording.recheck_calibration_provenance(output, {**row, "native": {**row["native"], "schema": "issue120-production-native-v2", "frame_count": 500000}})


def test_new_p_checks_boot_inside_work_mount(tmp_path, monkeypatch):
    sid = "a" * 32
    directory = tmp_path / sid
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"schema": "issue120-production-session-v1", "session_id": sid,
        "production_contract": runtime.PRODUCTION_CONTRACT, "host_boot_id_sha256": "old"}))
    monkeypatch.setattr(sessions, "SESSIONS", tmp_path)
    monkeypatch.setattr(runtime, "ROOT", Path("/work"))
    monkeypatch.setattr(sessions, "boot_fingerprint", lambda: "new")
    with pytest.raises(ValueError, match="boot"):
        sessions.read_session(sid)
