"""A separate compatible-production contract must not relax historical admission."""
from __future__ import annotations

import copy
import math

import pytest

from enodia.tt.bench.issue120_production import plan, runtime
from enodia.tt.bench.issue120_production import production_admission as production


@pytest.fixture
def prepared(monkeypatch):
    report = {"schema": "issue120-production-host-postprocess-v1", "fixture_only": True,
              "frames": 500000, "passed": True, "peak_rss_bytes": 200000000,
              "charged_host_seconds": 8.0, "launcher_reread_seconds": 6.0,
              "device_retrieval_measured": False}
    monkeypatch.setattr(production, "postprocess_proof", lambda: {"file": "unit-fixture", "sha256": "a" * 64, "report": report})
    current = production.fixture_identity()
    monkeypatch.setattr(runtime, "program_identity", lambda: current)
    return production.make_receipt(current), current


def test_preparation_is_single_selection_and_artifact_bound_not_authority(prepared):
    receipt, current = prepared
    production.validate_receipt(receipt, current)
    selection = production.selection()
    assert receipt["selection"] == selection
    assert receipt["fixed_repetitions"] == selection["fixed_repetitions"]
    assert receipt["execution_authorized"] is False
    assert receipt["owner_adopted"] is False
    assert set(receipt["calibrations"]) == {"C1", "C2"}
    assert len(receipt["model"]["slabs"]) == 40
    assert receipt["native_device_unchanged"] is True
    assert [s.frame_count for s in production.specs(receipt)] == [500000, 500000]
    assert [s.repeat_count for s in production.specs(receipt)] == [selection["fixed_repetitions"]] * 2


@pytest.mark.parametrize("field", ["count", "native", "baseline", "archive", "reserve", "source", "selection"])
def test_receipt_cannot_relabel_evidence_or_override_selected_model(prepared, field):
    receipt, current = prepared
    receipt = copy.deepcopy(receipt)
    if field == "count":
        receipt["fixed_repetitions"] += 1
    elif field == "native":
        receipt["program_identity"]["native_sha256"] = "b" * 64
    elif field == "baseline":
        receipt["baseline"]["kernel"] = "changed"
    elif field == "archive":
        receipt["calibrations"]["C2"]["archive_sha256"] = "b" * 64
    elif field == "reserve":
        receipt["model"]["reserve"]["optimistic_seconds"] = 0
    elif field == "source":
        receipt["python_transition"]["after"]["coverage.py"] = "b" * 64
    else:
        receipt["selection"]["target_worker_seconds"] += 1
    with pytest.raises(ValueError):
        production.validate_receipt(receipt, current)


def test_an_owner_alternative_changes_only_selection_pair(prepared, monkeypatch):
    receipt, current = prepared
    selected = copy.deepcopy(receipt["selection"])
    selected["target_worker_seconds"] += 10
    model = receipt["model"]
    selected["fixed_repetitions"] = 1 + math.ceil(model["r_fast"] * (selected["target_worker_seconds"] - model["L_fast_seconds"]))
    monkeypatch.setattr(production, "selection", lambda: selected)
    alternative = production.make_receipt(current)
    assert alternative["fixed_repetitions"] == selected["fixed_repetitions"]
    assert alternative["selection"] == selected
    assert plan.RUNS[2].repeat_count == 1048576
    assert plan.RUNS[3].repeat_count == 1572864


def test_frozen_evidence_and_fixture_identity_need_no_local_native_build(monkeypatch):
    monkeypatch.setattr(runtime, "_native_receipt", lambda: pytest.fail("CPU evidence must not require a native build"))
    assert production.calibration_summary()["production_coverage_eligible"] is True
    assert production.fixture_identity()["production_admission_contract"] == runtime.PRODUCTION_CONTRACT


def test_pending_host_proof_cannot_unlock_operational_p(prepared):
    receipt, _ = prepared
    receipt["postprocess_proof"]["report"]["measurement_pending"] = True
    with pytest.raises(ValueError, match="measured"):
        production.specs(receipt)


def test_native_or_nonmetadata_python_change_requires_stopping(prepared):
    _, current = prepared
    current = copy.deepcopy(current)
    current["python_sources_sha256"]["coverage.py"] = "b" * 64
    with pytest.raises(ValueError):
        production.make_receipt(current)
