"""Review-driven loop1 controls: positive, one logical mutation, named refusal."""
import copy
import json
import math
from datetime import UTC, datetime

import pytest

from enodia.tt.bench import issue30 as m
from enodia.tt.bench import issue30_admission as a
from enodia.tt.bench import issue30_diagnostics as d
from enodia.tt.bench import issue30_evidence as e
from enodia.tt.bench import issue30_formats as f
from tests.test_issue30 import environment, transcript, xml
from tests.test_issue30_owner import accepted
from tests.test_issue30_raw_evidence import fixture
from tests.test_issue30_topology import records, report
from tools import replay_issue30_evidence as replay


@pytest.mark.parametrize('payload', [b'', b' ', b'\t', b'\xc2\xa0'])
def test_watcher_device_prefix_needs_nonwhitespace_payload(payload):
    raw = b'Dump #1 at 0.100s\nDevice 0 waypoint\nDump #1 completed at 0.200s\n'
    assert d.watcher_observation(raw)['scan_passed']
    changed = raw.replace(b'Device 0 waypoint', b'Device ' + payload, 1)
    assert d.watcher_observation(changed) == {'fresh': False, 'scan_passed': False, 'dump_count': 0}


@pytest.mark.parametrize('phase', ['A', 'B'])
@pytest.mark.parametrize('space', ['', ' ', '\t', '\r\n', ' \t\r\n'])
@pytest.mark.parametrize('quote', ['"', "'"])
def test_xml_declaration_encoding_whitespace(phase, space, quote):
    declaration = f'<?xml version="1.0" encoding{space}={space}{quote}UTF-8{quote}?>\n'.encode()
    original = declaration + xml(phase)
    m._xml(phase, original)
    changed = original.replace(b'UTF-8', b'UTF-16', 1)
    with pytest.raises(ValueError, match='^test_result$'):
        m._xml(phase, changed)


@pytest.mark.parametrize('prefix', [b'<!-- <?xml encoding="UTF-16"?> -->',
                                  b'<?xmlnote encoding="UTF-16"?>'])
def test_xml_encoding_guard_retains_prior_xml_like_directive_refusals(prefix):
    m._xml('A', xml())
    changed = prefix + b'\n' + xml()
    with pytest.raises(ValueError, match='^test_result$'):
        m._xml('A', changed)


def _clock_case():
    before, first = fixture('before')
    after, last = fixture('after', 40)
    left = e.series(before, 'A', 'before', first.__getitem__)
    right = e.series(after, 'A', 'after', last.__getitem__)
    epoch = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    stamp = datetime.fromtimestamp(epoch + 30, UTC).replace(tzinfo=None).isoformat(timespec='milliseconds')
    raw = f'<testsuites timestamp="{stamp}"/>'.encode()
    clock = {'schema': 'issue30-capture-clock-v1', 'attempt': 'synthetic',
             'start_wall': epoch + 25, 'end_wall': epoch + 38,
             'start_monotonic': 25.0, 'end_monotonic': 38.0,
             'create_wall': epoch + 29, 'create_monotonic': 29.0}
    execution = {'attempt': 'synthetic', 'elapsed_seconds': 13.0}
    return left, right, raw, execution, clock, epoch


@pytest.mark.parametrize('event', ['create', 'launch'])
@pytest.mark.parametrize('boundary', ['next_float', 'five_seconds'])
def test_request_boundary_cannot_follow_native_beyond_one_millisecond(event, boundary):
    left, right, raw, execution, clock, epoch = _clock_case()
    # Native XML has three fractional digits: allow one representation quantum,
    # not another second of host clock skew or DRAM freshness slack.
    maximum = epoch + 30 + 0.001
    clock |= {event + '_wall': maximum, event + '_monotonic': maximum - epoch}
    e.bracket(left, right, raw, execution, clock)
    changed = copy.deepcopy(clock)
    wall = math.nextafter(maximum, math.inf) if boundary == 'next_float' else epoch + 35
    changed |= {event + '_wall': wall, event + '_monotonic': wall - epoch}
    with pytest.raises(ValueError, match='^dram_capture_clock$'):
        e.bracket(left, right, raw, execution, changed)


def _renewed_result(before):
    return m.evaluate('A', before['attempt'], before, before, xml(), report(records()), b'', m.GOOD_EXECUTION)


@pytest.mark.parametrize('value', [None, False, 1, 'card', [], ['serial']])
def test_nested_dram_card_object_checked_before_dereference(value):
    before, _ = fixture('before')
    assert _renewed_result(before)['reason'] == 'accepted'
    before['dram_admission']['samples'][0]['cards'][0] = value
    assert _renewed_result(before)['reason'] == 'dram_gate_reading'


@pytest.mark.parametrize('field,value,reason', [
    ('serial', [], 'dram_gate_cards'), ('serial', {}, 'dram_gate_cards'),
    ('serial', None, 'dram_gate_cards'), ('raw_sha256', [], 'dram_gate_raw'),
    ('raw_sha256', None, 'dram_gate_raw')])
def test_dram_identity_and_hash_types_have_named_refusals(field, value, reason):
    before, _ = fixture('before')
    assert _renewed_result(before)['reason'] == 'accepted'
    sample = before['dram_admission']['samples'][0]
    target = sample['cards'][0] if field == 'serial' else sample
    target[field] = value
    assert _renewed_result(before)['reason'] == reason


@pytest.mark.parametrize('value', [None, [], 'verdict', False])
def test_owner_original_verdict_is_object_before_lookup(value):
    from tests.test_issue30_admission import discovery_env
    before = discovery_env()
    original = accepted(before)
    args = (m.baseline_digest(before), m.digest(before['boot'].encode()),
            m.digest(m.json.dumps(before['topology'], sort_keys=True).encode()))
    a.owner_links(original, *args)
    original['original_verdict'] = value
    with pytest.raises(ValueError, match='^discovery_binding$'):
        a.owner_links(original, *args)


@pytest.mark.parametrize('value', [None, {}, False, 'versions'])
def test_discovery_reported_versions_is_list(value):
    from tests.test_issue30_morning import env as renewed_env
    from tests.test_issue30_morning import unit
    from tests.test_issue30_topology import judge
    before = renewed_env() | {'owner_unit': unit()}
    original = judge(records(), before=before) | {'attempt': unit()['A']}
    f.verdict(original)
    original['firmware_observation']['reported_versions'] = value
    with pytest.raises(ValueError, match='^discovery_binding$'):
        f.verdict(original)


def _baseline():
    return {'results': [{'attempt': identity, 'current_evaluate': conclusion}
                        for (_, identity), conclusion in zip(replay.CASES, replay.CONCLUSIONS, strict=True)]}


@pytest.mark.parametrize('value', [None, False, [], 'row'])
def test_replay_baseline_row_type_precedes_lookup(value):
    before = _baseline()
    replay._validate_baseline(before)
    before['results'][0] = value
    with pytest.raises(ValueError, match='^replay_baseline_schema$'):
        replay._validate_baseline(before)


@pytest.mark.parametrize('value', [[], {}, None])
def test_replay_baseline_attempt_type_precedes_set(value):
    before = _baseline()
    replay._validate_baseline(before)
    before['results'][0]['attempt'] = value
    with pytest.raises(ValueError, match='^replay_baseline_schema$'):
        replay._validate_baseline(before)


@pytest.mark.parametrize('value', [None, False, [], 'receipt'])
def test_receipt_object_before_approval_lookup(value):
    before = environment()
    receipt = m._load(m.CURRENT_RECEIPT, m.ROOT) | {'current_wrapper_sources': m.program_sources()}
    m._approval('A', 'synthetic-a', m.approval_record('A', 'synthetic-a', before, receipt, 'head'),
                before, receipt, 'head')
    approval = m.approval_record('A', 'synthetic-a', before, value, 'head')
    with pytest.raises(ValueError, match='^g0_not_ready$'):
        m._approval('A', 'synthetic-a', approval, before, value, 'head')


@pytest.mark.parametrize('value', [None, [], 'binaries'])
def test_binary_receipt_mapping_before_lookup(value, tmp_path):
    receipt = {'binaries': {}}
    for name in m.BINARIES.values():
        raw = ('synthetic binary fixture: ' + name).encode('ascii')
        (tmp_path / name).write_bytes(raw)
        receipt['binaries'][name] = m.digest(raw)
    m._binary_binding(tmp_path, receipt)
    receipt['binaries'] = value
    with pytest.raises(ValueError, match='^binary_binding$'):
        m._binary_binding(tmp_path, receipt)


@pytest.mark.parametrize('value', [None, False, [], 'execution'])
def test_direct_evaluate_execution_root_is_named_rejection(value):
    before = environment()
    args = ('A', 'synthetic-a', before, before, xml(), transcript(), b'')
    assert m.evaluate(*args, m.GOOD_EXECUTION)['reason'] == 'accepted'
    assert m.evaluate(*args, value)['reason'] == 'execution'


@pytest.mark.parametrize('where,value', [('root', None), ('root', []), ('root', 'before'),
    ('cards', None), ('cards', {}), ('card', None), ('card', 'card'), ('card', [])])
def test_malformed_environment_has_rejection_not_fingerprint_traceback(where, value):
    before = environment()
    assert m.evaluate('A', 'synthetic-a', before, before, xml(), transcript(), b'', m.GOOD_EXECUTION)['reason'] == 'accepted'
    if where == 'root':
        before = value
    elif where == 'cards':
        before['cards'] = value
    else:
        before['cards'][0] = value
    result = m.evaluate('A', 'synthetic-a', before, before, xml(), transcript(), b'', m.GOOD_EXECUTION)
    assert result['verdict'] == 'rejected'
    assert result['reason'] == 'environment'


@pytest.mark.parametrize('field', ['start_wall', 'end_wall', 'start_monotonic', 'end_monotonic'])
def test_capture_clock_overflow_has_named_refusal(field):
    left, right, raw, execution, clock, _ = _clock_case()
    e.bracket(left, right, raw, execution, clock)
    clock[field] = 10 ** 400
    with pytest.raises(ValueError, match='^dram_capture_clock$'):
        e.bracket(left, right, raw, execution, clock)


def test_capture_elapsed_overflow_has_named_refusal():
    left, right, raw, execution, clock, _ = _clock_case()
    e.bracket(left, right, raw, execution, clock)
    execution['elapsed_seconds'] = 10 ** 400
    with pytest.raises(ValueError, match='^dram_capture_window$'):
        e.bracket(left, right, raw, execution, clock)


def test_reported_sample_clock_overflow_has_named_refusal():
    left, right, raw, execution, clock, _ = _clock_case()
    e.bracket(left, right, raw, execution, clock)
    left[0]['reported_monotonic_seconds'] = 10 ** 400
    with pytest.raises(ValueError, match='^dram_summary_clock$'):
        e.bracket(left, right, raw, execution, clock)


def test_hash_bound_watcher_review_uses_the_same_row_validator(monkeypatch, tmp_path):
    data = {'raw': b'Dump #1 at 0.100s\nDevice 0 waypoint\nDump #1 completed at 0.200s\n'}
    monkeypatch.setattr(m, '_captured', lambda *args: data['raw'])
    review = {'watcher_clean': True, 'watcher_sha256': m.digest(data['raw'])}
    assert m._watcher_review('B', tmp_path, m.GOOD_EXECUTION, review)
    data['raw'] = data['raw'].replace(b'Device 0 waypoint', b'Device ', 1)
    review['watcher_sha256'] = m.digest(data['raw'])
    assert m._watcher_review('B', tmp_path, m.GOOD_EXECUTION, review) is False
    with pytest.raises(ValueError, match='^watcher$'):
        m._execution(m.GOOD_EXECUTION | {'watcher_clean': False}, 'B')


def test_judge_validates_xml_before_clock_consumption(monkeypatch, tmp_path):
    left, right, _, _, _, _ = _clock_case()
    monkeypatch.setattr(m, '_raw_series', lambda env, phase, role, inputs: left if role == 'before' else right)
    native = xml().replace(b'<testsuites ', b'<testsuites timestamp="2026-01-01T00:00:30.000" ')
    data = {'raw': b'<?xml version="1.0" encoding = "UTF-8"?>\n' + native}
    monkeypatch.setattr(m, '_captured', lambda *args: data['raw'])
    execution = {'elapsed_seconds': 1.0, 'artifacts': []}
    before = {'owner_unit': {}}
    m._series_proof('A', tmp_path, tmp_path, execution, before, {})
    data['raw'] = data['raw'].replace(b'UTF-8', b'UTF-16', 1)
    with pytest.raises(ValueError, match='^test_result$'):
        m._series_proof('A', tmp_path, tmp_path, execution, before, {})


def _capture_header():
    from tests.test_issue30_owner import captured_environment
    receipt = m._load(m.CURRENT_RECEIPT, m.ROOT)
    execution = m.GOOD_EXECUTION | {'schema': 'issue30-execution-v1', 'attempt': 'synthetic-a',
        'device_free': False, 'image': m.IMAGE, 'binary_sha256': receipt['binaries'][m.BINARIES['A']],
        'argv': m.request('A', 'synthetic-a', m.ASSETS, m.WORK, uid=1000, gid=1000)['Cmd'],
        'environment': captured_environment(), 'artifacts': []}
    raw = json.dumps(execution).encode()
    review = {'schema': 'issue30-review-v1', 'attempt': 'synthetic-a', 'watcher_clean': None,
              'execution_sha256': m.digest(raw), 'before_sha256': 'a' * 64, 'after_sha256': 'b' * 64}
    return receipt, raw, review


@pytest.mark.parametrize('value', [None, False, [], 'review'])
def test_review_object_precedes_hash_lookup(value, monkeypatch, tmp_path):
    _, raw, review = _capture_header()
    monkeypatch.setattr(m, '_verified', lambda *args: raw)
    assert m._review_execution('A', 'synthetic-a', tmp_path, review)['attempt'] == 'synthetic-a'
    with pytest.raises(ValueError, match='^review_schema$'):
        m._review_execution('A', 'synthetic-a', tmp_path, value)


@pytest.mark.parametrize('value', [None, [], {'binaries': None}])
def test_judge_receipt_object_precedes_binary_lookup(value, monkeypatch, tmp_path):
    receipt, raw, review = _capture_header()
    data = {'receipt': receipt}
    monkeypatch.setattr(m, '_verified', lambda *args: raw)
    monkeypatch.setattr(m, '_load', lambda *args: data['receipt'])
    assert m._review_execution('A', 'synthetic-a', tmp_path, review)['attempt'] == 'synthetic-a'
    data['receipt'] = value
    with pytest.raises(ValueError, match='^execution_provenance$'):
        m._review_execution('A', 'synthetic-a', tmp_path, review)


@pytest.mark.parametrize('role', ['before', 'after'])
@pytest.mark.parametrize('value', [None, False, [], 'owner_unit'])
def test_capture_observation_objects_precede_proof_use(role, value, monkeypatch, tmp_path):
    data = {'before': environment(), 'after': environment()}
    monkeypatch.setattr(m, '_load', lambda *args: {})
    monkeypatch.setattr(m, '_review_execution', lambda *args: m.GOOD_EXECUTION | {'artifacts': []})
    monkeypatch.setattr(m, '_container_observations', lambda *args: None)
    monkeypatch.setattr(m, '_observations', lambda *args: (data['before'], data['after']))
    monkeypatch.setattr(m, '_captured', lambda *args: json.dumps(data['before']).encode())
    monkeypatch.setattr(m, '_watcher_review', lambda *args: True)
    assert m._review_capture('A', 'synthetic-a', tmp_path, tmp_path)[0]['watcher_clean'] is True
    data[role] = value
    with pytest.raises(ValueError, match='^environment$'):
        m._review_capture('A', 'synthetic-a', tmp_path, tmp_path)
