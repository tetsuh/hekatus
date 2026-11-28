"""Additional same-class controls; all observations are synthetic."""
import copy

import pytest

from enodia.tt.bench import issue30 as m
from enodia.tt.bench import issue30_diagnostics as d
from enodia.tt.bench import issue30_formats as f
from enodia.tt.bench import issue30_topology as t
from tests.test_issue30 import WATCHER_DUMP, environment, xml
from tests.test_issue30_admission import discovery_env
from tests.test_issue30_owner import smoke_stdout, transfer
from tests.test_issue30_topology import env, judge, records


@pytest.mark.parametrize('left,right', [(False, 0), (800, 800.0), ({'v': [False]}, {'v': [0]}),
                                       ({'v': 800}, {'v': 800.0}), (float('nan'), float('nan'))])
def test_recursive_comparison_preserves_scalar_types(left, right):
    assert f.equal({'v': [800, False, None]}, {'v': [800, False, None]})
    assert not f.equal(left, right)


@pytest.mark.parametrize('mutation', ['extra', 'bool_chip', 'float_channel', 'bad_peer_core'])
def test_topology_records_have_closed_typed_fields(mutation):
    original = records()
    assert t.classify(original)['outcome'] == 'accepted'
    changed = copy.deepcopy(original)
    if mutation == 'extra':
        changed[0]['unknown'] = 1
    elif mutation == 'bool_chip':
        changed[0]['chip'] = False
    elif mutation == 'float_channel':
        changed[0]['channel'] = 4.0
    else:
        changed[0]['peer_core'] = 'bad core'
    assert t.classify(changed)['outcome'] == 'invalid'


def test_frozen_policy_type_is_not_numeric_equivalence():
    before = env()
    t.validate_input(before)
    before['topology']['hypotheses'][0]['connected_channels'][0] = 4.0
    with pytest.raises(ValueError, match='topology_policy'):
        t.validate_input(before)


@pytest.mark.parametrize('target', ['links', 'uncabled'])
def test_legacy_topology_also_checks_integer_types(target):
    before = environment()
    m._environment(before, 'synthetic-a', 'A')
    if target == 'links':
        before['links'][0]['chip'] = False
    else:
        before['uncabled'][0][0] = False
    with pytest.raises(ValueError, match='environment'):
        m._environment(before, 'synthetic-a', 'A')


def test_baseline_numeric_variables_are_ascii_only():
    from tests.test_issue30_owner import baseline_lines
    raw = ('\n'.join(baseline_lines()) + '\n').encode() + smoke_stdout()
    assert '\u2014' in raw.decode()  # The frozen motherboard literal is not ASCII-only.
    assert transfer(stdout=raw)['reason'] == 'accepted'
    changed = raw.replace(b'pid: 1', 'pid: ١'.encode(), 1)
    assert transfer(stdout=changed)['reason'] == 'diagnostic'


def test_unknown_logger_severity_is_not_silent_info():
    from tests.test_issue30 import transcript
    before = environment()
    assert m.evaluate('A', 'synthetic-a', before, before, xml(), transcript(), b'', m.GOOD_EXECUTION)['reason'] == 'accepted'
    changed = transcript().replace(b'| warning  |', b'| notice   |', 1)
    assert m.evaluate('A', 'synthetic-a', before, before, xml(), changed, b'', m.GOOD_EXECUTION)['reason'] == 'diagnostic'


def test_legacy_idle_clock_also_requires_exact_800_mhz():
    before = environment()
    m._environment(before, 'synthetic-a', 'A')
    before['cards'][0]['aiclk_mhz'] = 801
    with pytest.raises(ValueError, match='environment'):
        m._environment(before, 'synthetic-a', 'A')


def test_unused_health_reason_is_still_a_closed_field():
    before = discovery_env()
    m._environment(before, 'synthetic-a', 'A')
    before['cards'][0]['erisc'] = '1.4.1'
    before['cards'][0]['unobservable']['erisc']['unknown'] = True
    with pytest.raises(ValueError, match='unobservable_reason'):
        m._environment(before, 'synthetic-a', 'A')


@pytest.mark.parametrize('raw', [b'[ RUN      ] Other.Test\n', b'[  FAILED  ] Other.Test\n',
                                b'[       OK ] Other.Test (1 ms)\n'])
def test_extra_native_test_markers_refuse(raw):
    assert transfer()['reason'] == 'accepted'
    assert transfer(stdout=smoke_stdout() + raw)['reason'] == 'packet_evidence'


def test_xml_time_must_fit_the_process_cap():
    m._xml('A', xml())
    changed = xml().replace(b'<testsuites ', b'<testsuites time="86400" ')
    with pytest.raises(ValueError, match='test_result'):
        m._xml('A', changed)


def test_fresh_result_rejects_type_equal_canonical_copy():
    from tests.test_issue30_morning import env as renewed_env
    from tests.test_issue30_morning import unit
    before = renewed_env() | {'owner_unit': unit()}
    original = judge(records(), before=before) | {'attempt': unit()['A'], 'xml_present': True}
    m._fresh_result(before, original, original, original)
    changed = copy.deepcopy(original)
    changed['xml_present'] = 1
    with pytest.raises(ValueError, match='fresh_A_canonical_result'):
        m._fresh_result(before, changed, original, original)


@pytest.mark.parametrize('line', [b'At -1.000s starting\n', b'At 90000.000s attach device 0\n'])
def test_other_watcher_timestamps_are_not_unchecked(line):
    raw = WATCHER_DUMP
    assert d.watcher_observation(raw)['scan_passed']
    assert not d.watcher_observation(line + raw)['scan_passed']


@pytest.mark.parametrize('raw', [b'{"v": 1e400}', b'{"v": Infinity}', b'{"v": 1, "v": 2}',
                                b'\xff', b'\xef\xbb\xbf{}'])
def test_other_invalid_json_forms_are_not_alternate_encodings(raw):
    assert f.load(b'{"v": 1.0}') == {'v': 1.0}
    with pytest.raises(ValueError, match='evidence_json'):
        f.load(raw)


@pytest.mark.parametrize('stamp', ['2026-02-30T00:00:00.000', '2026-01-01T24:00:00.000',
                                  '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.٠٠٠'])
def test_timestamp_calendar_and_ascii_grammar(stamp):
    assert f.stamp('2026-01-01T00:00:00.000') > 0
    with pytest.raises(ValueError, match='evidence_timestamp'):
        f.stamp(stamp)


@pytest.mark.parametrize('change,reason', [('environment', 'execution_provenance'),
    ('unknown', 'execution_schema'), ('argv', 'execution_provenance'),
    ('create', 'execution'), ('bytes', 'capture_artifact')])
def test_capture_metadata_is_closed_and_phase_bound(change, reason):
    from tests.test_issue30_owner import captured_environment
    value = m.GOOD_EXECUTION | {'schema': 'issue30-execution-v1', 'attempt': 'synthetic-a',
        'device_free': False, 'image': m.IMAGE, 'binary_sha256': 'a' * 64,
        'argv': m.request('B', 'synthetic-a', m.ASSETS, m.WORK, uid=1000, gid=1000)['Cmd'],
        'environment': captured_environment(),
        'artifacts': [{'path': 'stdout.bin', 'bytes': 0, 'sha256': m.digest(b'')}]}
    f.execution(value)
    m._captured_command('B', value)
    if change == 'environment':
        value['environment'][0] = 'TT_METAL_WATCHER=0'
    elif change == 'unknown':
        value['ignored'] = True
    elif change == 'argv':
        value['argv'].append('--extra')
    elif change == 'create':
        value['create_status'] = 'unknown'
    else:
        value['artifacts'][0]['bytes'] = False
    if change == 'argv':
        f.execution(value)
        with pytest.raises(ValueError, match=reason):
            m._captured_command('B', value)
    else:
        with pytest.raises(ValueError, match=reason):
            f.execution(value)


def test_legacy_capture_enclosure_cannot_hide_a_stale_launch():
    from datetime import UTC, datetime, timedelta

    from enodia.tt.bench import issue30_evidence as e
    from tests.test_issue30_raw_evidence import fixture
    before, first = fixture('before')
    after, last = fixture('after', 140)
    a = e.series(before, 'A', 'before', first.__getitem__)
    b = e.series(after, 'A', 'after', last.__getitem__)
    native = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=85)
    raw = ('<testsuites timestamp="' + native.replace(tzinfo=None).isoformat(timespec='milliseconds') + '"/>').encode()
    # Observed request boundaries prove create age40/start age45; the enclosing
    # capture entry alone at second55 does not prove when launch occurred.
    clock = {'schema': 'issue30-capture-clock-v1', 'attempt': 'synthetic',
             'start_wall': native.timestamp() - 30, 'end_wall': native.timestamp(),
             'start_monotonic': 55.0, 'end_monotonic': 85.0,
             'create_wall': native.timestamp() - 25, 'create_monotonic': 60.0,
             'launch_wall': native.timestamp() - 20, 'launch_monotonic': 65.0}
    execution = {'attempt': 'synthetic', 'elapsed_seconds': 30.0}
    e.bracket(a, b, raw, execution, clock)
    with pytest.raises(ValueError, match='dram_launch_freshness'):
        e.bracket(a, b, raw, execution)


def test_incomplete_audit_is_never_a_fully_passed_result(tmp_path, monkeypatch):
    from tools import replay_issue30_evidence as r
    monkeypatch.setattr(m, 'WORK', tmp_path)
    expected = {identity: {'verdict': m.ACCEPTED[phase], 'reason': 'accepted'} for phase, identity in r.CASES}
    monkeypatch.setattr(r, '_judge', lambda phase, identity: expected[identity])
    results = r._positives(expected)
    assert all(row['matches'] for row in results)
    for row in results[2:]:
        assert row['validation_complete'] is False
        assert row['all_checks_passed'] is False
        assert row['eligible_for_admission'] is False
        assert row['unavailable_checks']
