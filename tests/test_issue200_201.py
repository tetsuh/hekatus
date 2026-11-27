"""Host-only strictness controls: positive, one mutation, named refusal."""
from unittest.mock import Mock

import pytest

from enodia.tt.bench import issue30 as m
from enodia.tt.bench import issue30_admission as a
from enodia.tt.bench import issue30_diagnostics as d
from tests.test_issue30 import environment, transcript, xml
from tests.test_issue30_owner import smoke_stdout, transfer
from tests.test_issue30_raw_evidence import fixture


@pytest.mark.parametrize('tag', ['failure', 'error', 'skipped', 'testcase', 'testsuite'])
def test_xml_namespaced_extra_node_is_not_ignored(tag):
    m._xml('A', xml())
    changed = xml().replace(b'/></testsuite>',
        f'><x:{tag} xmlns:x="urn:extra"/></testcase></testsuite>'.encode())
    with pytest.raises(ValueError, match='test_result'):
        m._xml('A', changed)


@pytest.mark.parametrize('change', ['component', 'location', 'body'])
def test_dispatch_logger_envelope_is_exact(change):
    # Full source-shaped real format is the positive, not a body-only surrogate.
    raw = smoke_stdout()
    assert transfer(stdout=raw)['reason'] == 'accepted'
    old, new = {'component': (b'           Test |', b'            UMD |'),
                'location': (b'fixture.hpp:87', b'fixture.hpp:88'),
                'body': (b'fast dispatch (', b'fast dispatch extra (')}[change]
    assert old in raw
    assert transfer(stdout=raw.replace(old, new, 1))['reason'] == 'packet_evidence'


def test_stderr_cannot_add_a_second_pass_marker():
    before = environment()
    # The topology-aware fixture enables native packet evidence.
    from tests.test_issue30_admission import discovery_env
    from tests.test_issue30_owner import accepted
    before = discovery_env()
    args = ('B', 'synthetic-a', before, before, xml('B'), smoke_stdout())
    assert m.evaluate(*args, b'', m.GOOD_EXECUTION, accepted(before))['reason'] == 'accepted'
    assert m.evaluate(*args, b'[  PASSED  ] 1 test.\n', m.GOOD_EXECUTION,
                      accepted(before))['reason'] == 'packet_evidence'


@pytest.mark.parametrize('change', ['uid', 'extra_header', 'extra_up'])
def test_health_report_tokens_and_uid_are_bound(change):
    env = environment()
    assert m.evaluate('A', 'synthetic-a', env, env, xml(), transcript(), b'', m.GOOD_EXECUTION)['reason'] == 'accepted'
    old, new, reason = {
        'uid': (b'Test | Chip: 0 PCIe: 0 Unique ID: 0',
                b'Test | Chip: 0 PCIe: 0 Unique ID: f', 'diagnostic'),
        'extra_header': (b'Chip: 0 PCIe: 0 Unique ID: 0\n',
                         b'Chip: 0 PCIe: 0 Unique ID: 0 extra\n', 'chip_identity'),
        'extra_up': (b'connected to Chip 1 core', b'connected to Chip 1 extra core', 'peer_links'),
    }[change]
    result = m.evaluate('A', 'synthetic-a', env, env, xml(), transcript().replace(old, new, 1), b'', m.GOOD_EXECUTION)
    assert result['reason'] == reason


@pytest.mark.parametrize('value', [False, 0.0])
def test_chip_identity_is_an_integer_not_a_numeric_equal(value):
    env = environment()
    m._environment(env, 'synthetic-a', 'A')
    env['cards'][0]['chip'] = value
    with pytest.raises(ValueError, match='environment'):
        m._environment(env, 'synthetic-a', 'A')


def test_dram_summary_clock_is_not_float_equal_to_integer():
    from tests.test_issue30_dram import before
    env = before()
    a.dram_gate(env)
    env['dram_admission']['samples'][0]['cards'][0]['aiclk_mhz'] = 800.0
    with pytest.raises(ValueError, match='dram_gate_reading'):
        a.dram_gate(env)


@pytest.mark.parametrize('where', ['environment', 'card', 'arc'])
def test_unknown_acceptance_fields_are_not_ignored(where):
    from tests.test_issue30_admission import discovery_env
    env = discovery_env()
    m._environment(env, 'synthetic-a', 'A')
    target = {'environment': env, 'card': env['cards'][0],
              'arc': env['cards'][0]['arc_heartbeat']}[where]
    target['unrecognized'] = True
    with pytest.raises(ValueError, match='environment'):
        m._environment(env, 'synthetic-a', 'A')


def test_json_encoding_and_nonfinite_values_refuse(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'WORK', tmp_path)
    path = tmp_path / 'input.json'
    path.write_bytes(b'{"value": 0}')
    assert m._load(path) == {'value': 0}
    for raw in ('{"value": 0}'.encode('utf-16'), b'{"value": NaN}'):
        path.write_bytes(raw)
        with pytest.raises(ValueError, match='evidence_json'):
            m._load(path)


def test_publish_failure_removes_only_owned_staging(tmp_path, monkeypatch):
    target = tmp_path / 'result.json'
    m.publish(target, {'value': 1})
    target.unlink()
    monkeypatch.setattr(m.os, 'fsync', Mock(side_effect=OSError('synthetic sync failure')))
    with pytest.raises(OSError, match='sync failure'):
        m.publish(target, {'value': 2})
    assert not target.exists()
    assert not target.with_name('result.json.tmp').exists()


@pytest.mark.parametrize('phase', ['A', 'B'])
def test_stale_dram_admission_has_a_positive_control(tmp_path, monkeypatch, phase):
    from enodia.tt.bench import issue30_evidence as e
    before, files = fixture('before', 80)
    samples = e.series(before, 'A', 'before', files.__getitem__)
    after_a, earlier_files = fixture('after', 40)
    earlier = e.series(after_a, 'A', 'after', earlier_files.__getitem__)
    monkeypatch.setattr(m, '_raw_series', lambda env, ph, role, inputs: samples if role == 'before' else earlier)
    monkeypatch.setattr(m, '_load', lambda *args: after_a)
    last = samples[-1]
    monkeypatch.setattr(m.time, 'monotonic', lambda: last['reported_monotonic_seconds'] + 10)
    monkeypatch.setattr(m.time, 'time', lambda: last['monotonic_seconds'] + 10)
    m._admit_series(phase, tmp_path, before)
    monkeypatch.setattr(m.time, 'time', lambda: last['monotonic_seconds'] + 86400)
    with pytest.raises(ValueError, match='dram_launch_freshness'):
        m._admit_series(phase, tmp_path, before)


def test_watcher_requires_completed_dump_utf8_and_sane_time():
    raw = (b'Dump #1 at 0.100s\nDevice 0 synthetic waypoint\n'
           b'Dump #1 completed at 0.200s\n')
    assert d.watcher_observation(raw)['scan_passed']
    for changed in (b'Dump #1 at 0.100s\n', raw + b'\xff',
                    raw.replace(b'0.100s', b'999999.100s')):
        assert not d.watcher_observation(changed)['scan_passed']
