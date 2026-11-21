"""Dated authority metadata is separate from readiness and durable consumption."""
from __future__ import annotations

import copy
import json

import pytest

from enodia.tt.bench.issue120_production import plan, runtime, sessions
from enodia.tt.bench.issue120_production import production_admission as production


@pytest.fixture
def adoption_input(tmp_path, monkeypatch):
    path = tmp_path / 'adoption.json'
    original = json.loads((runtime.SOURCE / 'production_adoption.json').read_bytes())
    path.write_text(json.dumps(original, indent=2) + '\n')
    monkeypatch.setattr(production, 'ADOPTION', path, raising=False)
    return path, original


@pytest.mark.parametrize('change', ['missing', 'count', 'target', 'hash', 'date', 'future', 'scope', 'conditional', 'extra'])
def test_adoption_input_rejects_missing_mismatch_or_malformed_scope(adoption_input, change):
    path, data = adoption_input
    if change == 'missing':
        path.unlink()
    else:
        if change == 'count':
            data['fixed_repetitions'] += 1
        elif change == 'target':
            data['target_worker_seconds'] += 1
        elif change == 'hash':
            data['selection_file_sha256'] = 'f' * 64
        elif change == 'date':
            data['decision_date'] = '2026-02-30'
        elif change == 'future':
            data['decision_date'] = '9999-01-01'
        elif change == 'scope':
            data['scope'][plan.RUNS[4].identity] = 2
        elif change == 'conditional':
            data['conditional_loaded_requires_healthy_control'] = False
        else:
            data['ignore_checks'] = True
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        production.owner_adoption()


def test_adopted_receipt_is_explicit_and_legacy_generation_unchanged(adoption_input, monkeypatch):
    report = {'schema': 'issue120-production-host-postprocess-v1', 'fixture_only': True,
              'frames': 500000, 'passed': True, 'peak_rss_bytes': 200000000,
              'charged_host_seconds': 8., 'launcher_reread_seconds': 6., 'device_retrieval_measured': False}
    monkeypatch.setattr(production, 'postprocess_proof', lambda: {'file': 'fixture', 'sha256': 'a' * 64, 'report': report})
    current = production.fixture_identity()
    legacy = production.make_receipt(current)
    adopted = production.make_receipt(current, adoption=production.owner_adoption())
    assert legacy['schema'] == production.SCHEMA
    assert legacy['owner_adopted'] is False
    assert legacy['execution_authorized'] is False
    assert adopted['schema'] == production.ADOPTED_SCHEMA
    assert adopted['owner_adopted'] is True
    assert adopted['execution_authorized'] is True
    production.validate_receipt(adopted, current)
    adoption_input[0].unlink()
    production.validate_receipt(legacy, current)
    production.validate_receipt(adopted, current, live=False)
    with pytest.raises((ValueError, OSError)):
        production.validate_receipt(adopted, current)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, 'SESSIONS', tmp_path / 'sessions')
    sid = 'a' * 32
    manifest = {'session_id': sid, 'harness_commit': 'b' * 40, 'host_boot_id_sha256': 'c' * 64,
                'admission_hashes': {'production': 'd' * 64}, 'owner_adoption_sha256': 'e' * 64}
    receipt = {'schema': 'issue120-compatible-production-admission-v2', 'owner_adopted': True,
               'execution_authorized': True, 'owner_decision': {'sha256': 'e' * 64}}
    case = sessions.session_path(sid) / 'cases' / plan.RUNS[4].identity
    return manifest, receipt, case


def test_exclusive_claim_burns_role_even_without_native_or_result(ledger):
    manifest, receipt, case = ledger
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    assert (case / 'launch-claim.json').is_file()
    with pytest.raises((ValueError, OSError)):
        sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)


def test_adoption_cannot_be_replayed_in_another_real_session(ledger):
    manifest, receipt, _ = ledger
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    second = copy.deepcopy(manifest)
    second['session_id'] = '1' * 32
    with pytest.raises(ValueError, match='session'):
        sessions.reserve_production_launch(plan.RUNS[4], second, '2' * 64, receipt)


def test_container_entry_needs_host_claim_and_cannot_repeat(ledger):
    manifest, receipt, case = ledger
    with pytest.raises((ValueError, OSError)):
        sessions.enter_production_native(plan.RUNS[4], manifest, 'f' * 64, receipt, case)
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    sessions.enter_production_native(plan.RUNS[4], manifest, 'f' * 64, receipt, case)
    assert (case / 'native-entry.json').is_file()
    with pytest.raises((ValueError, OSError)):
        sessions.enter_production_native(plan.RUNS[4], manifest, 'f' * 64, receipt, case)


def test_container_rejects_claim_from_another_manifest(ledger):
    manifest, receipt, case = ledger
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    with pytest.raises(ValueError, match='claim'):
        sessions.enter_production_native(plan.RUNS[4], manifest, '0' * 64, receipt, case)


def test_unadopted_operational_launch_is_not_readiness(ledger):
    manifest, receipt, _ = ledger
    receipt.update(schema=production.SCHEMA, owner_adopted=False, execution_authorized=False)
    with pytest.raises(ValueError, match='adopt'):
        sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
