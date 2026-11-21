"""C2-only admission and artifact-preserving cross-precommit calibration."""
from __future__ import annotations

import copy
import hashlib
import json
import tarfile
from unittest.mock import patch

import pytest

from enodia.tt.bench.issue120_production import plan, recording, runtime, sessions


def target_identity():
    # These tests characterize the frozen C2 contract, not new P admission.
    return copy.deepcopy(json.loads((runtime.SOURCE / "bounded_admission.json").read_bytes())["program_identity"])


@pytest.fixture(autouse=True)
def historical_c2_source_context(monkeypatch):
    original = sessions._python_transition

    def transition(old, current):
        # Emulate the original source inventory only during C2 version validation;
        # direct inventory/shadowing tests still execute the real current guard.
        with patch.object(runtime, "python_source_hashes", return_value=target_identity()["python_sources_sha256"]):
            return original(old, current)

    monkeypatch.setattr(sessions, "_python_transition", transition)


def test_c2_receipt_binds_original_c1_and_changes_metadata_only(monkeypatch):
    current = target_identity()
    receipt = sessions.make_c2_admission(current)
    sessions.validate_c2_admission(receipt, current)
    assert receipt["allowed_plan_identities"] == [plan.RUNS[3].identity]
    assert receipt["c1_program_identity"]["native_sha256"] == current["native_sha256"]
    assert receipt["c1_program_identity"] != current
    assert set(receipt["python_transition"]["changed"]) <= {"runtime.py", "sessions.py", "recording.py"}
    monkeypatch.setattr(sessions, "_committed_admission", lambda *a: receipt)
    assert sessions.admitted_specs("C") == [plan.RUNS[3]]
    with pytest.raises(ValueError, match="C2"):
        sessions.admitted_specs("P")


@pytest.mark.parametrize("change", ["native", "kernel", "image", "python", "scope", "work", "c1", "baseline"])
def test_c2_cannot_admit_drift_by_relabeling_or_ignoring_digests(change):
    current = target_identity()
    receipt = sessions.make_c2_admission(current)
    if change == "native":
        current["native_sha256"] = "b" * 64
        receipt["program_identity"] = copy.deepcopy(current)
    elif change == "kernel":
        current["compile_manifest_sha256"] = "b" * 64
        receipt["program_identity"] = copy.deepcopy(current)
    elif change == "image":
        receipt["image"] = "different"
    elif change == "python":
        receipt["python_transition"]["changed"]["coverage.py"] = {"before": "a" * 64, "after": "b" * 64}
    elif change == "scope":
        receipt["allowed_plan_identities"].append(plan.RUNS[2].identity)
    elif change == "work":
        receipt["work_units"]["bytes_per_write"] = 8192
    elif change == "c1":
        receipt["c1_evidence"]["record_sha256"] = "b" * 64
    else:
        receipt["stable_environment"]["kernel"] = "Linux changed"
    with pytest.raises(ValueError):
        sessions.validate_c2_admission(receipt, current)


@pytest.fixture
def paired(tmp_path, monkeypatch):
    c1, raw = sessions.c1_calibration_evidence()
    current = target_identity()
    receipt = sessions.make_c2_admission(current)
    monkeypatch.setattr(sessions, "_committed_admission", lambda *a: receipt)
    monkeypatch.setattr(runtime, "program_identity", lambda: current)
    c2 = copy.deepcopy(c1)
    spec = plan.RUNS[3]
    c2.update(plan_identity=spec.identity, attempt_identity="c2-compat-fixture", session_id="2" * 32,
              program_identity=current, configuration=plan.program_configuration(spec))
    c2["configuration_sha256"] = plan.configuration_digest(c2["configuration"])
    c2["environment"]["commit"] = "c" * 40
    c2["session_manifest_sha256"] = "2" * 64
    c2["native"]["frame_count"] = spec.frame_count
    c2["native"]["repeat_count"] = spec.repeat_count
    c2["native"]["host_cpu_work"].update(operations=spec.repeat_count * 32768, reads=spec.repeat_count, transfers=spec.repeat_count)
    trace = c2["native"]["host_phase_trace"]
    trace["final_count"] = spec.repeat_count
    # Compatibility policy tests use an artifact-loader seam; actual trace replay is covered separately.
    for key in ("frames_consumed", "frames_produced", "frames_attempted"):
        c2["metrics"][key] = spec.frame_count
    monkeypatch.setattr(runtime, "CALIBRATION_RECORD_ROOT", tmp_path, raising=False)
    directory = tmp_path / "C2"
    directory.mkdir()
    raw = copy.deepcopy(raw)
    raw.update(harness_commit=c2["environment"]["commit"], run_id=c2["attempt_identity"])
    (directory / f"env-{c2['attempt_identity']}.json").write_text(json.dumps(raw))
    receipt_sha = hashlib.sha256((runtime.SOURCE / "bounded_admission.json").read_bytes()).hexdigest()
    manifest = {"schema": "issue120-production-session-v1", "session_id": c2["session_id"],
                "harness_commit": c2["environment"]["commit"], "program_identity": current,
                "image": runtime.PINNED_IMAGE, "host_boot_id_sha256": c2["host_boot_id_sha256"],
                "admission_hashes": {"bounded": receipt_sha}, "created_monotonic_ns": 1,
                "runs": {spec.identity: {"configuration": c2["configuration"],
                                         "configuration_sha256": c2["configuration_sha256"]}}}
    manifest_bytes = json.dumps(manifest).encode()
    (directory / f"session-{c2['session_id']}.json").write_bytes(manifest_bytes)
    c2["session_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    c2["calibration_provenance"] = recording.calibration_provenance(directory, c2, receipt_sha)
    paths = [tmp_path / "C1.json", directory / f"results-{c2['attempt_identity']}.json"]
    rows = [c1, c2]
    source = runtime.ROOT / sessions.C1_DIRECTORY / "C1-evidence.tar.gz"
    with tarfile.open(source) as archive:
        paths[0].write_bytes(archive.extractfile(f"results-{sessions.C1_ATTEMPT}.json").read())
    paths[1].write_text(json.dumps(c2))
    monkeypatch.setattr(recording, "load_validated_record", lambda path: copy.deepcopy(rows[paths.index(path)]))
    return paths, rows, receipt


def test_pair_preserves_both_original_identities_and_harness_commits(paired):
    paths, rows, _ = paired
    original = copy.deepcopy(rows)
    facts = recording.calibration_pair_analysis(paths)
    assert facts["diagnostic_calibration_usable"] is True
    assert facts["production_coverage_eligible"] is True
    assert facts["execution_authorized"] is False
    assert [x["harness_commit"] for x in facts["records"]] == [rows[0]["environment"]["commit"], "c" * 40]
    assert rows == original


def test_healthy_not_covered_keeps_rate_cost_facts_without_coverage_promotion(paired):
    paths, rows, _ = paired
    rows[1]["coverage_analysis"].update(coverage="not_covered", reason="fixed work ended first")
    rows[1].update(execution_eligible=False, timing_evidence=False)
    facts = recording.calibration_pair_analysis(paths)
    assert facts["diagnostic_calibration_usable"] is True
    assert facts["production_coverage_eligible"] is False
    assert facts["records"][1]["coverage"] == "not_covered"
    assert facts["records"][1]["worker_writes_per_second"] > 0


@pytest.mark.parametrize("change", ["environment", "native", "not_evaluable", "missing_provenance", "env_capture"])
def test_pair_refuses_unapproved_provenance_or_invalid_evidence(paired, change):
    paths, rows, _ = paired
    if change == "environment":
        rows[1]["environment"]["kmd"] = "changed"
    elif change == "native":
        rows[1]["program_identity"]["native_sha256"] = "b" * 64
    elif change == "not_evaluable":
        rows[1]["coverage_analysis"]["coverage"] = "not_evaluable"
    elif change == "missing_provenance":
        del rows[1]["calibration_provenance"]
    else:
        path = paths[1].parent / f"env-{rows[1]['attempt_identity']}.json"
        path.write_text(path.read_text().replace('6.8.0-149', '6.8.0-999'))
    with pytest.raises(ValueError):
        recording.calibration_pair_analysis(paths)


@pytest.mark.parametrize("missing", ["contract", "provenance"])
def test_traced_c2_cannot_drop_its_required_environment_provenance(paired, missing):
    paths, rows, _ = paired
    if missing == "contract":
        del rows[1]["program_identity"]["calibration_compatibility_contract"]
    else:
        del rows[1]["calibration_provenance"]
    with pytest.raises(ValueError):
        recording.recheck_calibration_provenance(paths[1].parent, rows[1])


@pytest.mark.parametrize("name", ["coverage/__init__.py", "coverage/child.py", "coverage.abi3.so", "coverage.pyc", "coverage.pyw"])
def test_nested_or_uninventoried_importable_sources_cannot_shadow_validation(tmp_path, monkeypatch, name):
    (tmp_path / "safe.py").write_text("pass\n")
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("pass\n")
    monkeypatch.setattr(runtime, "SOURCE", tmp_path)
    with pytest.raises(ValueError):
        runtime.python_source_hashes()


@pytest.mark.parametrize("claim", ["null_digest", "reassigned_session", "phase"])
def test_session_or_phase_claims_cannot_bypass_independent_provenance(paired, claim):
    paths, rows, _ = paired
    row = rows[1]
    if claim == "null_digest":
        row["session_manifest_sha256"] = None
        row["calibration_provenance"]["session_manifest_sha256"] = None
    elif claim == "reassigned_session":
        row["session_id"] = "4" * 32
        row["calibration_provenance"]["session_id"] = "4" * 32
    else:
        del row["phase"]
        del row["calibration_provenance"]
    with pytest.raises(ValueError):
        recording.recheck_calibration_provenance(paths[1].parent, row)


@pytest.mark.parametrize("field", ["session_id", "harness_commit", "program_identity", "host_boot_id_sha256",
                                   "image", "admission_hashes", "runs", "created_monotonic_ns"])
def test_session_artifact_bindings_survive_duplicated_claim_rehash(paired, field):
    paths, rows, _ = paired
    row = rows[1]
    path = paths[1].parent / row["calibration_provenance"]["session_manifest"]["file"]
    manifest = json.loads(path.read_bytes())
    manifest[field] = 0 if field == "created_monotonic_ns" else "different"
    content = json.dumps(manifest).encode()
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    row["session_manifest_sha256"] = digest
    row["calibration_provenance"]["session_manifest_sha256"] = digest
    row["calibration_provenance"]["session_manifest"].update(sha256=digest, byte_count=len(content))
    with pytest.raises(ValueError):
        recording.recheck_calibration_provenance(paths[1].parent, row)


@pytest.mark.parametrize("remove_phase", [False, True])
def test_coordinated_discriminator_deletion_cannot_hide_native_c2_work(paired, remove_phase):
    paths, rows, _ = paired
    row = rows[1]
    del row["program_identity"]["calibration_compatibility_contract"]
    del row["plan_identity"]
    del row["calibration_provenance"]
    if remove_phase:
        del row["phase"]
    with pytest.raises(ValueError):
        recording.recheck_calibration_provenance(paths[1].parent, row)


@pytest.mark.parametrize("redirected", [False, True])
def test_pair_cli_paths_must_stay_in_program_owned_record_root(tmp_path, monkeypatch, redirected):
    root = tmp_path / "approved"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    path = outside
    if redirected:
        path = root / "redirected.json"
        path.symlink_to(outside)
    monkeypatch.setattr(runtime, "CALIBRATION_RECORD_ROOT", root, raising=False)
    monkeypatch.setattr(sessions, "_committed_admission", lambda *a: pytest.fail("unvalidated path reached admission"))
    with pytest.raises(ValueError):
        recording.calibration_pair_analysis([path, path])


def test_legacy_exact_provenance_proposal_was_not_weakened(paired):
    from enodia.tt.bench.issue120_production import records

    _, rows, _ = paired
    with pytest.raises(ValueError, match="provenance"):
        records.calibration_proposal(rows)
