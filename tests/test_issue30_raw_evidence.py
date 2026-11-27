"""Raw series controls; all identities and times are synthetic."""
import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from enodia.tt.bench import issue30_evidence as e
from tests.test_issue30_morning import env, unit


def fixture(role, offset=0):
    value = env() | {'owner_unit': unit(), 'attempt': unit()['A']}
    samples, files = [], {}
    for index in range(5):
        devices, cards = [], []
        for card in value['cards']:
            heartbeat = offset + index + 1
            devices.append({'board_info': {'board_id': card['serial'], 'bus_id': card['pci'],
                                            'dram_status': True},
                            'firmwares': {'fw_bundle_version': card['bundle'], 'cm_fw': card['cmfw']},
                            'telemetry': {'heartbeat': str(heartbeat), 'aiclk': ' 800',
                                          'asic_temperature': str(card.get('temperature_c', 50.0))},
                            'smbus_telem': {'DDR_STATUS': '0x5555', 'GDDR_UNCORR_ERRS': '0x0'}})
            cards.append({'serial': card['serial'], 'arc_heartbeat': heartbeat,
                          'dram_status': True, 'raw_ddr_status': '0x5555',
                          'uncorrectable_errors': 0, 'aiclk_mhz': 800})
        timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=offset + index * 5)
        local = timestamp.astimezone().replace(tzinfo=None)
        raw = json.dumps({'time': local.isoformat(timespec='microseconds'), 'device_info': devices}).encode()
        files[f'telemetry/{role}-A/snapshot{index + 1}.stdout'] = raw
        samples.append({'monotonic_seconds': offset + index * 5,
                        'raw_sha256': hashlib.sha256(raw).hexdigest(), 'cards': cards})
    value['dram_admission']['samples'] = samples
    for card in value['cards']:
        card['arc_heartbeat'] = {'first': offset + 1, 'second': offset + 5}
    return value, files


@pytest.mark.parametrize('value,expected', [(' 800', 800), ('800 ', 800), (' 800 ', 800),
    ('7717', 7717), ('0x0', 0), ('0x5555', 21845), (' 0x0 ', 0)])
def test_recorded_unsigned_numeric_forms(value, expected):
    assert e._integer(value) == expected


@pytest.mark.parametrize('value', [True, False, -1, 800.0, 800.1, '800.0', '8e2', '800MHz',
    '', ' ', '  ', '+800', '-800', '--800', '0x', '80 0', '\t800', '800\n', '\u00a0800', '٨٠٠'])
def test_numeric_padding_does_not_allow_lossy_or_non_ascii_values(value):
    with pytest.raises(ValueError, match='dram_raw_integer'):
        e._integer(value)


@pytest.mark.parametrize('value', ['53.2', '53.4', '53.5', '53.6', '53.9', '54.4', '54.5',
    '54.6', '56.2', '56.4', '57.4', '57.5', '57.6', '57.8', ' 53.2 ', '50'])
def test_temperature_display_is_separate_from_unsigned_counters(value):
    assert e._temperature(value) == float(value)


@pytest.mark.parametrize('value', [True, 53.2, 'NaN', 'inf', '53.2C', '\t53.2', ' ', '+53.2', '-53.2'])
def test_invalid_temperature_display_refuses(value):
    with pytest.raises(ValueError, match='raw_temperature'):
        e._temperature(value)


def test_raw_values_hash_missing_and_role_swap():
    before, files = fixture('before')
    assert len(e.series(before, 'A', 'before', files.__getitem__)) == 5
    for mutation, reason in [('byte', 'dram_raw_binding'), ('value', 'dram_raw_values'),
                             ('role', 'dram_raw_binding')]:
        changed, raw = copy.deepcopy(before), dict(files)
        if mutation == 'byte':
            raw['telemetry/before-A/snapshot1.stdout'] += b' '
        elif mutation == 'value':
            changed['dram_admission']['samples'][0]['cards'][0]['arc_heartbeat'] = 999
        else:
            _, other = fixture('after', 40)
            raw['telemetry/before-A/snapshot1.stdout'] = other['telemetry/after-A/snapshot1.stdout']
        with pytest.raises(ValueError, match=reason):
            e.series(changed, 'A', 'before', raw.__getitem__)
    with pytest.raises(KeyError):
        e.series(before, 'A', 'before', {}.__getitem__)


def test_truncated_clock_value_fails_strict_value_comparison():
    before, files = fixture('before')
    name = 'telemetry/before-A/snapshot1.stdout'
    value = json.loads(files[name]); value['device_info'][0]['telemetry']['aiclk'] = '80'
    files[name] = json.dumps(value).encode()
    before['dram_admission']['samples'][0]['raw_sha256'] = hashlib.sha256(files[name]).hexdigest()
    with pytest.raises(ValueError, match='dram_raw_values'):
        e.series(before, 'A', 'before', files.__getitem__)


def test_cross_series_reuse_order_heartbeat_and_capture_window():
    before, first = fixture('before')
    after, last = fixture('after', 40)
    a = e.series(before, 'A', 'before', first.__getitem__)
    b = e.series(after, 'A', 'after', last.__getitem__)
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=30)
    xml = f'<testsuites timestamp="{stamp.replace(tzinfo=None).isoformat(timespec="milliseconds")}"/>'.encode()
    execution = {'elapsed_seconds': 1.0}
    e.bracket(a, b, xml, execution)
    native = stamp.timestamp()
    clock = {'schema': 'issue30-capture-clock-v1', 'attempt': 'synthetic',
             'start_wall': native - 1, 'end_wall': native + 1,
             'start_monotonic': 29.0, 'end_monotonic': 31.0}
    bound_execution = {'attempt': 'synthetic', 'elapsed_seconds': 2.0}
    e.bracket(a, b, xml, bound_execution, clock)
    with pytest.raises(ValueError, match='dram_capture_clock'):
        e.bracket(a, b, xml, bound_execution, clock | {'attempt': 'another'})
    changed = copy.deepcopy(b)
    changed[0]['reported_monotonic_seconds'] = 0
    with pytest.raises(ValueError, match='dram_summary_clock'):
        e.bracket(a, changed, xml, bound_execution, clock)
    with pytest.raises(ValueError, match='dram_snapshot_reuse'):
        e.ordered(a, a)
    with pytest.raises(ValueError, match='dram_capture_window'):
        e.bracket(b, a, xml, execution)
    changed = copy.deepcopy(b)
    changed[0]['cards'][0]['arc_heartbeat'] = 0
    with pytest.raises(ValueError, match='dram_series_order'):
        e.ordered(a, changed)


def test_full_judge_fresh_A_and_B_evidence_chain(tmp_path, monkeypatch):
    from enodia.tt.bench import issue30 as m
    from tests.test_issue30 import xml
    from tests.test_issue30_owner import captured_environment, smoke_stdout
    from tests.test_issue30_topology import records, report

    monkeypatch.setattr(m, 'WORK', tmp_path)
    receipt = m._load(m.CURRENT_RECEIPT, m.ROOT)
    a_result = None
    for phase, start in (('A', 0), ('B', 80)):
        before, first = fixture('before', start)
        after, last = fixture('after', start + 40)
        identity = unit()[phase]
        for value in (before, after):
            value['attempt'] = identity
        output, inputs = m._slots(identity, True)
        output.mkdir(parents=True); inputs.mkdir(parents=True)
        (output / 'native').mkdir()
        for name, raw in (first | last).items():
            path = inputs / name.replace('-A/', '-' + phase + '/')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        for role, value in (('before', before), ('after', after)):
            m.publish(inputs / (role + '.json'), value)
            m.publish(output / ('docker-' + role + '.json'), {'containers': []})
            (output / ('docker-' + role + '.txt')).write_bytes(b'')
        m.publish(output / 'observations-before.json', before)
        stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=start + 30)
        native = xml(phase).replace(b'<testsuites ',
            ('<testsuites timestamp="' + stamp.replace(tzinfo=None).isoformat(timespec='milliseconds') + '" ').encode())
        (output / 'native/gtest.xml').write_bytes(native)
        (output / 'stdout.bin').write_bytes(report(records()) if phase == 'A' else smoke_stdout())
        (output / 'stderr.bin').write_bytes(b'')
        review = {'schema': 'issue30-review-v1', 'attempt': identity, 'watcher_clean': None}
        if phase == 'B':
            watcher = output / 'logs/generated/watcher/watcher.log'
            watcher.parent.mkdir(parents=True)
            watcher.write_bytes(b'Dump #1 at 0.100s\nDevice 0 synthetic waypoint\nDump #1 completed at 0.200s\n')
            review |= {'watcher_clean': True, 'watcher_sha256': m.digest(watcher.read_bytes())}
            m.publish(inputs / 'discovery.json', a_result)
        execution = m.GOOD_EXECUTION | {'schema': 'issue30-execution-v1', 'attempt': identity,
            'device_free': False, 'image': m.IMAGE, 'binary_sha256': receipt['binaries'][m.BINARIES[phase]],
            'argv': m.request(phase, identity, output, output, uid=1000, gid=1000)['Cmd'],
            'environment': captured_environment(),
            'artifacts': m._captured_files(output)}
        m.publish(output / 'execution.json', execution)
        for role in ('before', 'after'):
            review[role + '_sha256'] = m.digest((inputs / (role + '.json')).read_bytes())
        review['execution_sha256'] = m.digest((output / 'execution.json').read_bytes())
        m.publish(inputs / 'review.json', review)
        result = m._judgment(phase, identity, output, inputs, a_result)
        assert result['verdict'] == m.ACCEPTED[phase]
        if phase == 'A':
            a_result = result
            m.publish(output / 'result.json', result)
            m.publish(inputs / 'approval.json', {'synthetic': True})
    monkeypatch.setattr(m, '_approval', lambda *args: None)
    monkeypatch.setattr(m, '_git', lambda *args: 'synthetic-head')
    assert m._bound_discovery(inputs) == a_result
    aout, ain = m._slots(unit()['A'], True)
    raw_path = ain / 'telemetry/before-A/snapshot1.stdout'
    saved = raw_path.read_bytes()
    raw_path.unlink()
    a_identity = unit()['A']
    with pytest.raises(ValueError, match='missing, redirected or oversized evidence'):
        m._judgment('A', a_identity, aout, ain)
    raw_path.write_bytes(saved)
    assert m._bound_discovery(inputs) == a_result
    (aout / 'stdout.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='capture_artifact'):
        m._bound_discovery(inputs)
