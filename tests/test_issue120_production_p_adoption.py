"""Dated authority metadata is separate from readiness and durable consumption."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_issue120_production_p_review import valid_origin

from enodia.tt.bench.issue120_production import plan, recording, runner, runtime, sessions
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
    assert json.loads(production.owner_adoption()['text']) == data
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
    if change == 'missing':
        with pytest.raises(FileNotFoundError) as refused:
            production.owner_adoption()
        assert refused.value.filename == str(path)
    else:
        messages = {
            'count': 'owner adoption selection/date/scope differs',
            'target': 'owner adoption selection/date/scope differs',
            'hash': 'owner adoption selection/date/scope differs',
            'date': 'day is out of range for month',
            'future': 'owner adoption decision date is invalid or future',
            'scope': 'owner adoption selection/date/scope differs',
            'conditional': 'owner adoption selection/date/scope differs',
            'extra': 'owner adoption selection/date/scope differs',
        }
        with pytest.raises(ValueError, match=messages[change]):
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
    with pytest.raises(FileNotFoundError) as refused:
        production.validate_receipt(adopted, current)
    assert refused.value.filename == str(adoption_input[0])


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
    original = (case / 'launch-claim.json').read_bytes()
    with pytest.raises(FileExistsError) as refused:
        sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    assert refused.value.filename == str(case)
    assert (case / 'launch-claim.json').read_bytes() == original
    assert not (case / 'native-entry.json').exists()


def test_adoption_cannot_be_replayed_in_another_real_session(ledger):
    manifest, receipt, _ = ledger
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    second = copy.deepcopy(manifest)
    second['session_id'] = '1' * 32
    with pytest.raises(ValueError, match='^owner adoption is already assigned to another session$'):
        sessions.reserve_production_launch(plan.RUNS[4], second, 'f' * 64, receipt)
    assert not sessions.session_path(second['session_id']).exists()


def _path_refusals(monkeypatch):
    original = runtime.fixed_path
    refused = []

    def observed(path, root, **kwargs):
        try:
            return original(path, root, **kwargs)
        except ValueError:
            refused.append(path)
            raise

    monkeypatch.setattr(runtime, 'fixed_path', observed)
    return refused


def test_container_entry_needs_host_claim(ledger, monkeypatch):
    manifest, receipt, case = ledger
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    sessions.enter_production_native(plan.RUNS[4], manifest, 'f' * 64, receipt, case)
    # Restore the pre-entry output state; only the required host claim is removed.
    (case / 'native-entry.json').unlink()
    (case / 'launch-claim.json').unlink()
    refused_paths = _path_refusals(monkeypatch)
    with pytest.raises(ValueError, match='^required program-owned path missing$'):
        sessions.enter_production_native(plan.RUNS[4], manifest, 'f' * 64, receipt, case)
    assert refused_paths == [case / 'launch-claim.json']
    assert not (case / 'native-entry.json').exists()


def test_container_entry_cannot_repeat(ledger):
    manifest, receipt, case = ledger
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    sessions.enter_production_native(plan.RUNS[4], manifest, 'f' * 64, receipt, case)
    stamp = (case / 'native-entry.json').read_bytes()
    with pytest.raises(FileExistsError) as refused:
        sessions.enter_production_native(plan.RUNS[4], manifest, 'f' * 64, receipt, case)
    assert refused.value.filename == str(case / 'native-entry.json')
    assert (case / 'native-entry.json').read_bytes() == stamp


def test_container_rejects_claim_from_another_manifest(ledger):
    manifest, receipt, case = ledger
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    sessions.enter_production_native(plan.RUNS[4], manifest, 'f' * 64, receipt, case)
    (case / 'native-entry.json').unlink()  # Reset the positive control's output.
    with pytest.raises(ValueError, match='^container launch claim differs from its session origin$'):
        sessions.enter_production_native(plan.RUNS[4], manifest, '0' * 64, receipt, case)
    assert not (case / 'native-entry.json').exists()


@pytest.fixture
def completed_control(ledger, monkeypatch):
    manifest, receipt, case = ledger
    origin, row = valid_origin()
    origin.update(manifest)
    manifest.update(origin)
    row.update(session_id=manifest['session_id'], session_manifest_sha256='f' * 64,
               host_boot_id_sha256=manifest['host_boot_id_sha256'])
    row['environment']['commit'] = manifest['harness_commit']
    captured = case / 'receipt.json'
    receipt_bytes = json.dumps(receipt).encode()
    manifest['admission_hashes']['production'] = hashlib.sha256(receipt_bytes).hexdigest()
    row['production_provenance'] = {'receipt': {'file': captured.name}}
    spec = sessions.spec_for_session(plan.RUNS[4].identity, manifest)
    sessions.reserve_production_launch(spec, manifest, 'f' * 64, receipt)
    captured.write_bytes(receipt_bytes)
    primary = case / 'results-fixture.json'
    primary.write_text(json.dumps(row))
    sessions.enter_production_native(spec, manifest, 'f' * 64, receipt, case)

    def read_session(identity):
        assert identity == manifest['session_id']
        return manifest, 'f' * 64

    def load_record(path):
        # Only artifact loading is a seam; role/claim/entry/completion checks are real.
        assert path == primary
        return json.loads(path.read_bytes())

    monkeypatch.setattr(sessions, 'read_session', read_session)
    monkeypatch.setattr(recording, 'load_validated_record', load_record)
    sessions.finish_production_launch(spec, manifest, 'f' * 64, receipt, primary,
                                     launcher_exit=0, wrapper_exit=0, container_exit=0)
    assert runner._control(manifest['session_id'], plan.RUNS[5]) == row
    return manifest, receipt, case, spec, primary


def test_published_p1_without_completed_launcher_cannot_release_p2(completed_control, monkeypatch):
    manifest, _, case, _, _ = completed_control
    acknowledgement = case / 'external-completion.json'
    acknowledgement.unlink()  # The sole changed prerequisite after a successful release.
    refused_paths = _path_refusals(monkeypatch)
    with pytest.raises(ValueError, match='^required program-owned path missing$'):
        runner._control(manifest['session_id'], plan.RUNS[5])
    assert refused_paths == [acknowledgement]
    assert not acknowledgement.exists()
    assert not (case.parent / plan.RUNS[5].identity).exists()


def test_registration_fsync_failure_rejects_before_second_role_creation(ledger, monkeypatch):
    manifest, receipt, _ = ledger
    sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    monkeypatch.setattr(sessions.os, 'fsync', lambda _: (_ for _ in ()).throw(OSError('injected registration barrier failure')))
    with pytest.raises(OSError, match='^injected registration barrier failure$'):
        sessions.reserve_production_launch(plan.RUNS[5], manifest, 'f' * 64, receipt)
    assert not (sessions.session_path(manifest['session_id']) / 'cases' / plan.RUNS[5].identity).exists()


@pytest.mark.parametrize('field', ['launcher_exit', 'wrapper_exit', 'container_exit'])
def test_failed_external_outcome_cannot_acknowledge_completion(completed_control, field):
    manifest, receipt, case, spec, primary = completed_control
    acknowledgement = case / 'external-completion.json'
    acknowledgement.unlink()  # Restore the pre-publication output state.
    exits = {'launcher_exit': 0, 'wrapper_exit': 0, 'container_exit': 0}
    exits[field] = 2
    with pytest.raises(ValueError, match='^production completion needs all observed exits zero$'):
        sessions.finish_production_launch(spec, manifest, 'f' * 64, receipt, primary, **exits)
    assert not acknowledgement.exists()
    assert not (case.parent / plan.RUNS[5].identity).exists()


def test_unadopted_operational_launch_is_not_readiness(ledger):
    manifest, receipt, _ = ledger
    assert sessions._production_claim(plan.RUNS[4], manifest, 'f' * 64, receipt) is not None
    receipt['schema'] = production.SCHEMA
    with pytest.raises(ValueError, match='^operational production launch requires owner adoption$'):
        sessions.reserve_production_launch(plan.RUNS[4], manifest, 'f' * 64, receipt)
    assert not sessions.session_path(manifest['session_id']).exists()
