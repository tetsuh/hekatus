"""Final owner scope: mandatory live DRAM evidence and prior XML markup refusal."""
import copy
import json
from unittest.mock import Mock

import pytest

from enodia.tt.bench import issue30 as m
from tests.test_issue30 import xml
from tests.test_issue30_morning import unit
from tests.test_issue30_raw_evidence import capture_chain
from tests.test_issue200_201_loop2 import RequestClockHarness, rebind
from tools import replay_issue30_evidence as replay


def legacy(value):
    changed = copy.deepcopy(value)
    changed.pop('owner_unit')
    changed.pop('dram_admission')
    return changed


@pytest.mark.parametrize('phase', ['A', 'B'])
def test_live_admission_refuses_legacy_before_platform_checks(tmp_path, monkeypatch, phase):
    capture_chain(tmp_path, monkeypatch)
    identity = unit()[phase]
    output, inputs = m._slots(identity, True)
    before = m._load(inputs / 'before.json')
    h = RequestClockHarness(phase, '')
    monkeypatch.setattr(m.time, 'time', lambda: h.last['monotonic_seconds'] + 10)
    monkeypatch.setattr(m.time, 'monotonic', lambda: h.last['reported_monotonic_seconds'] + 10)
    assert len(m._admit_series(phase, inputs, before)) == 5
    monkeypatch.setattr(m, '_renewed_gate', lambda *args: None)
    monkeypatch.setattr(m.topology, 'validate_input', lambda *args: None)
    platform = Mock(side_effect=RuntimeError('platform_boundary'))
    monkeypatch.setattr(m, '_environment', platform)
    with pytest.raises(RuntimeError, match='^platform_boundary$'):
        m._authorize(phase, identity, inputs, output)
    platform.reset_mock()
    (inputs / 'before.json').write_text(json.dumps(legacy(before)))
    with pytest.raises(ValueError, match='^dram_live_series_required$'):
        m._authorize(phase, identity, inputs, output)
    platform.assert_not_called()


@pytest.mark.parametrize('phase', ['A', 'B'])
@pytest.mark.parametrize('event', ['create', 'launch'])
def test_live_request_guard_requires_series(tmp_path, monkeypatch, phase, event):
    h = RequestClockHarness(phase, '')
    h.install(monkeypatch, tmp_path)
    clocks = {}
    m._request_clock(h.samples, event, clocks)
    assert clocks[event + '_wall'] == h.last['monotonic_seconds'] + 10
    clocks.clear()
    with pytest.raises(ValueError, match='^dram_live_series_required$'):
        m._request_clock(None, event, clocks)
    assert clocks == {}


@pytest.mark.parametrize('phase', ['A', 'B'])
def test_capture_legacy_cannot_create(tmp_path, monkeypatch, phase):
    h = RequestClockHarness(phase, '')
    h.install(monkeypatch, tmp_path)
    identity = h.before['attempt']
    positive = m.capture(phase, identity, tmp_path, tmp_path / 'positive', observations=h.before)
    assert positive['started']
    h.requests.clear()
    refused = m.capture(phase, identity, tmp_path, tmp_path / 'legacy', observations=legacy(h.before))
    assert not refused['started']
    assert h.requests == []
    assert refused['error_type'] == 'ValueError'


@pytest.mark.parametrize('phase', ['A', 'B'])
def test_device_bound_payload_requires_observations(tmp_path, monkeypatch, phase):
    h = RequestClockHarness(phase, '')
    h.install(monkeypatch, tmp_path)
    identity = h.before['attempt']
    payload = m.request(phase, identity, tmp_path, tmp_path / 'unused', uid=1000, gid=1000)
    positive = m.capture(phase, identity, tmp_path, tmp_path / 'g0', approved_payload=payload)
    assert positive['started']
    assert positive['device_free']
    h.requests.clear()
    payload['HostConfig']['Devices'] = [{'PathOnHost': '/dev/tenstorrent/0',
                                        'PathInContainer': '/dev/tenstorrent/0', 'CgroupPermissions': 'rw'}]
    refused = m.capture(phase, identity, tmp_path, tmp_path / 'bound', approved_payload=payload)
    assert not refused['started']
    assert h.requests == []
    assert refused['error_type'] == 'ValueError'


@pytest.mark.parametrize('phase', ['A', 'B'])
def test_live_judge_refuses_coherently_bound_legacy_shape(tmp_path, monkeypatch, phase):
    capture_chain(tmp_path, monkeypatch)
    identity = unit()[phase]
    output, inputs = m._slots(identity, True)
    discovery = m._load(inputs / 'discovery.json') if phase == 'B' else None
    assert m._judgment(phase, identity, output, inputs, discovery)['reason'] == 'accepted'
    before = legacy(m._load(inputs / 'before.json'))
    (inputs / 'before.json').write_text(json.dumps(before))
    rebind(output, inputs, 'observations-before.json', json.dumps(before).encode())
    review = m._load(inputs / 'review.json')
    review['before_sha256'] = m.digest((inputs / 'before.json').read_bytes())
    (inputs / 'review.json').write_text(json.dumps(review))
    with pytest.raises(ValueError, match='^dram_live_series_required$'):
        m._judgment(phase, identity, output, inputs, discovery)


@pytest.mark.parametrize('phase', ['A', 'B'])
def test_offline_legacy_uses_audit_only_and_is_not_eligible(monkeypatch, phase):
    canonical = {'retained': True}
    monkeypatch.setattr(m, '_load', lambda path, *args: canonical if path.name != 'before.json' else {})
    audit = Mock(return_value={'verdict': m.ACCEPTED[phase], 'reason': 'accepted'})
    live = Mock(side_effect=AssertionError('historical input reached live judgment'))
    monkeypatch.setattr(replay, '_audit_capture', audit)
    monkeypatch.setattr(m, '_judgment', live)
    result = replay._judge(phase, 'synthetic-history')
    assert result['eligible_for_admission'] is False
    audit.assert_called_once_with(phase, 'synthetic-history')
    live.assert_not_called()


@pytest.mark.parametrize('phase', ['A', 'B'])
def test_whitespace_cdata_restores_prior_lexical_refusal(phase):
    original = xml(phase)
    m._xml(phase, original)
    changed = original.replace(b'</testsuites>', b'<![CDATA[ \t\n]]></testsuites>', 1)
    assert changed != original
    with pytest.raises(ValueError, match='^test_result$'):
        m._xml(phase, changed)


@pytest.mark.parametrize('phase', ['A', 'B'])
def test_cdata_refuses_through_live_judge(tmp_path, monkeypatch, phase):
    capture_chain(tmp_path, monkeypatch)
    identity = unit()[phase]
    output, inputs = m._slots(identity, True)
    discovery = m._load(inputs / 'discovery.json') if phase == 'B' else None
    assert m._judgment(phase, identity, output, inputs, discovery)['reason'] == 'accepted'
    raw = (output / 'native/gtest.xml').read_bytes()
    changed = raw.replace(b'</testsuites>', b'<![CDATA[ \t\n]]></testsuites>', 1)
    rebind(output, inputs, 'native/gtest.xml', changed)
    with pytest.raises(ValueError, match='^test_result$'):
        m._judgment(phase, identity, output, inputs, discovery)
