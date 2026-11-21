"""Replay exact actual production evidence without native SDK or board access."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from enodia.tt.bench.issue120_production import production_admission as production
from enodia.tt.bench.issue120_production import recording, records, runtime

DIRECTORY = Path(__file__).resolve().parents[1] / 'docs/measurements/issue120-production/2026-11-21-p'
PINS = {
    'P1': ('064a6ac9f5b65eeccc81b4c34b4f64cb84754ae057caf001f50e9e54711dbb28',
           '8404a4548031f0857edfeecaca9b774dcc8f74132b4f9674b8a2f7811973eb5b'),
    'P2': ('0f7d1d2cd0228236bbd648385b9e00a65ca414b846e90dd22878167abe0899ed',
           '5547e05c34b0c5f416d5eb197ec539758c4c51ef68554a1979757d36ea91c8b8'),
}
MEASURED_HEAD = '4fd4900b8813e813263ed3d5148e3b5feb3a59b8'
ATTEMPTS = {'P1': '20261121T16080360639616673963217100Z', 'P2': '20261121T16190313060076874183029578Z'}
CAPTURED_RECEIPT = 'production-receipt-c478ec91fa4421234adae76244bd9e11e2d865eda393b0f03028cbf0e3a39b7f.json'


@pytest.fixture(scope='module')
def actual_pair(tmp_path_factory):
    base = tmp_path_factory.mktemp('actual-production-p')
    values = {}
    with patch.object(runtime, '_native_receipt', side_effect=AssertionError('archive replay must not require live SDK')):
        for role, (archive_sha, manifest_sha) in PINS.items():
            manifest = DIRECTORY / f'{role}-manifest.json'
            archive = DIRECTORY / f'{role}-evidence.tar.gz'
            assert runtime.sha(manifest) == manifest_sha
            assert runtime.sha(archive) == archive_sha
            facts = json.loads(manifest.read_bytes())
            root = base / role
            root.mkdir()
            with tarfile.open(archive) as bundle:
                members = bundle.getmembers()
                names = {a['file'] for a in facts['artifacts']}
                assert len(members) == len(names) == 16
                assert all(m.isfile() and Path(m.name).name == m.name for m in members)
                assert sum(m.size for m in members) < 80 * 2**20
                assert {m.name for m in members} == names
                bundle.extractall(root, filter='data')
            expected = {item['file']: item for item in facts['artifacts']}
            for path in root.iterdir():
                item = expected[path.name]
                assert path.stat().st_size == item['byte_count']
                assert runtime.sha(path) == item['sha256']
            primary = root / f'results-{ATTEMPTS[role]}.json'
            assert primary.name == facts['record_file']
            original = primary.read_bytes()
            row = recording.load_validated_record(primary)
            assert primary.read_bytes() == original
            values[role] = (primary, row)
        yield values


@pytest.mark.parametrize('role', ['P1', 'P2'])
def test_actual_production_replays_all_frames_and_consumption_evidence(actual_pair, role):
    primary, row = actual_pair[role]
    assert row['environment']['commit'] == MEASURED_HEAD
    assert row['session_id'] == 'a729ec3dec50456998fc668b2100648f'
    assert row['execution_eligible'] is True
    assert row['timing_evidence'] is True
    assert row['resident_integrity_passed'] is True
    assert row['metrics']['frames_consumed'] == 500000
    assert row['native']['checksum_verified_frames'] == 500000
    assert row['native']['startup_diagnostics']['mailbox_post_finish'] == 500002
    assert row['metrics']['paired_outlier_count'] == 0
    assert row['coverage_analysis']['coverage'] == ('not_applicable' if role == 'P1' else 'covered')
    stamp = json.loads((primary.parent / 'external-completion.json').read_bytes())
    assert stamp['record_sha256'] == runtime.sha(primary)
    assert stamp['native_entry_sha256'] == runtime.sha(primary.parent / 'native-entry.json')
    assert stamp['launcher_exit'] == stamp['wrapper_exit'] == stamp['container_exit'] == 0


def test_actual_same_session_difference_and_causal_bounds(actual_pair):
    control, loaded = actual_pair['P1'][1], actual_pair['P2'][1]
    difference = records.compare_records(control, loaded)
    assert difference == loaded['control_comparison']
    assert difference['loaded_minus_control_ticks'] == {'p50': -10., 'p99': 18., 'p99_9': 5., 'p99_99': -22., 'max_ticks': 19}
    assert loaded['metrics']['load_accounting']['submitted_transfers'] == 22155167
    assert loaded['metrics']['load_accounting']['cpu_operations'] == 725980512256
    assert loaded['coverage_analysis']['first_through_last_transfers_lower_bound'] == 499999
    assert loaded['coverage_analysis']['first_through_last_cpu_operations_lower_bound'] == 16383967232
    assert loaded['coverage_analysis']['first_through_last_bytes_lower_bound'] == 2047995904
    assert difference['latency_threshold'] is None


def test_captured_adoption_keeps_dated_meaning_without_live_input(actual_pair, tmp_path, monkeypatch):
    primary, row = actual_pair['P2']
    reference = row['production_provenance']['receipt']
    assert reference['file'] == CAPTURED_RECEIPT
    receipt = json.loads((primary.parent / CAPTURED_RECEIPT).read_bytes())
    monkeypatch.setattr(production, 'ADOPTION', tmp_path / 'missing-future-adoption.json')
    production.validate_receipt(receipt, row['program_identity'], live=False)
    assert receipt['schema'] == production.ADOPTED_SCHEMA
    assert json.loads(receipt['owner_decision']['text'])['decision_date'] == '2026-11-21'
    assert receipt['owner_adopted'] is True


@pytest.mark.parametrize('change', ['environment', 'provenance'])
def test_actual_production_rejects_primary_only_tampering(actual_pair, change):
    primary, _ = actual_pair['P2']
    original = primary.read_bytes()
    changed = json.loads(original)
    if change == 'environment':
        changed['environment']['serial'] = 'tampered'
    else:
        changed.pop('production_provenance')
    try:
        primary.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            recording.load_validated_record(primary)
    finally:
        primary.write_bytes(original)
