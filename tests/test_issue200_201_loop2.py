"""Last review loop: committed or synthetic controls, one mutation, named refusal."""
import copy
import json
import math
import subprocess
from unittest.mock import Mock

import pytest

from enodia.tt.bench import issue30 as m
from enodia.tt.bench import issue30_diagnostics as d
from enodia.tt.bench import issue30_evidence as e
from tests.test_issue30 import WATCHER_ROW, xml
from tests.test_issue30_owner import discovery_env
from tests.test_issue30_raw_evidence import fixture

ROW = WATCHER_ROW
WATCHER = b'At 0.000s starting\nDump #1 at 0.100s\n' + ROW + b'\nDump #1 completed at 0.200s\nAt 0.300s detach device 0\n'


@pytest.mark.parametrize('phase', ['A', 'B'])
@pytest.mark.parametrize('encoding', ['utf-16le', 'utf-16be', 'utf-32le', 'utf-32be'])
def test_native_encoding_is_not_redetected_after_utf8_decode(phase, encoding):
    original = xml(phase)
    m._xml(phase, original)
    changed = original.decode().encode(encoding)
    with pytest.raises(ValueError, match='^test_result$'):
        m._xml(phase, changed)


@pytest.mark.parametrize('phase', ['A', 'B'])
@pytest.mark.parametrize('markup', [b'<!-- hidden -->', b'<?hidden data?>'])
def test_native_ignored_markup_cannot_become_success(phase, markup):
    original = xml(phase)
    m._xml(phase, original)
    changed = original.replace(b'/></testsuite>', b'>' + markup + b'</testcase></testsuite>', 1)
    assert changed != original
    with pytest.raises(ValueError, match='^test_result$'):
        m._xml(phase, changed)


@pytest.mark.parametrize('declaration', [b' xmlns=""', b' xmlns:unused="urn:unused"'])
def test_unused_namespace_declarations_are_not_dropped(declaration):
    original = xml()
    m._xml('A', original)
    changed = original.replace(b'<testsuites ', b'<testsuites' + declaration + b' ', 1)
    with pytest.raises(ValueError, match='^test_result$'):
        m._xml('A', changed)


def test_native_duration_exact_decimal_cap():
    original = xml().replace(b'<testsuites ', b'<testsuites time="40" ', 1)
    m._xml('A', original)
    changed = original.replace(b'time="40"', b'time="40.0000000000000001"', 1)
    with pytest.raises(ValueError, match='^test_result$'):
        m._xml('A', changed)


@pytest.mark.parametrize('row', [b'Device banana', b'Device 99 waypoint', b'Device 0',
    b'Device 0 waypoint', ROW.replace(b'Device 0', b'Device 99'), ROW.replace(b'rmsg:', b'rmessage:')])
def test_watcher_needs_source_shaped_admitted_device_row(row):
    assert d.watcher_observation(WATCHER, 1)['scan_passed']
    changed = WATCHER.replace(ROW, row, 1)
    assert d.watcher_observation(changed, 1) == {'fresh': False, 'scan_passed': False, 'dump_count': 0}


@pytest.mark.parametrize('replacement', [b' At 99999.000s starting', b'At\t99999.000s starting',
    b'Dump\t#1 at 0.100s'])
def test_malformed_control_records_cannot_be_silently_ignored(replacement):
    assert d.watcher_observation(WATCHER, 1)['scan_passed']
    target = b'Dump #1 at 0.100s' if replacement.startswith(b'Dump') else b'At 0.000s starting'
    assert not d.watcher_observation(WATCHER.replace(target, replacement, 1), 1)['scan_passed']


def test_watcher_missing_footer_is_the_only_mutation():
    assert d.watcher_observation(WATCHER, 1)['scan_passed']
    assert not d.watcher_observation(WATCHER.replace(b'Dump #1 completed at 0.200s\n', b'', 1), 1)['scan_passed']


def test_watcher_overcap_event_keeps_increasing_order():
    assert d.watcher_observation(WATCHER, 1)['scan_passed']
    assert not d.watcher_observation(WATCHER.replace(b'At 0.300s', b'At 1.001s', 1), 1)['scan_passed']


@pytest.mark.parametrize('clock', [None, [], False, 'clock'])
def test_present_clock_is_never_legacy_absence(clock):
    from tests.test_issue200_201_loop1 import _clock_case
    left, right, raw, execution, valid, _ = _clock_case()
    e.bracket(left, right, raw, execution, valid)
    with pytest.raises(ValueError, match='^dram_capture_clock$'):
        e.bracket(left, right, raw, execution, clock)


@pytest.mark.parametrize('field', ['monotonic_seconds', 'reported_monotonic_seconds'])
@pytest.mark.parametrize('bad', ['clock', None, False, [], 10 ** 400])
def test_launch_operands_are_typed_before_subtraction(field, bad):
    value, files = fixture('before')
    samples = e.series(value, 'A', 'before', files.__getitem__)
    last = samples[-1]
    wall, mono = last['monotonic_seconds'] + 10, last['reported_monotonic_seconds'] + 10
    e.launch_freshness(samples, wall, mono)
    changed = copy.deepcopy(samples)
    changed[-1][field] = bad
    with pytest.raises(ValueError, match='^dram_launch_freshness$'):
        e.launch_freshness(changed, wall, mono)


def test_api_and_ps_use_the_same_socket_under_hostile_environment(tmp_path, monkeypatch):
    calls = []
    for name in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH'):
        monkeypatch.setenv(name, 'hostile-synthetic-value')
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b'[]' if argv[0] == 'curl' else b'')
    monkeypatch.setattr(m.subprocess, 'run', run)
    assert m._api('GET', 'containers/json?all=1') == []
    m._docker_ps(tmp_path / 'ps.txt', 5)
    curl, ps = calls
    assert curl[0][curl[0].index('--unix-socket') + 1] == m.DOCKER_SOCKET
    assert curl[0][-1] == 'http://localhost/v1.52/containers/json?all=1'
    assert ps[0][:3] == list(m.DOCKER_CLI)
    assert not any(k.startswith('DOCKER_') for k in ps[1]['env'])
    assert (tmp_path / 'ps.txt').read_bytes() == b''


def rebind(output, inputs, name, raw, **updates):
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    execution = json.loads((output / 'execution.json').read_bytes()) | updates
    entry = m.artifact(path, output)
    execution['artifacts'] = [e for e in execution['artifacts'] if e['path'] != name] + [entry]
    encoded = json.dumps(execution).encode()
    (output / 'execution.json').write_bytes(encoded)
    review = json.loads((inputs / 'review.json').read_bytes())
    review['execution_sha256'] = m.digest(encoded)
    if name == 'logs/generated/watcher/watcher.log':
        review['watcher_sha256'] = m.digest(raw)
    (inputs / 'review.json').write_text(json.dumps(review))


def test_A_optional_watcher_is_hash_checked_not_semantically_required(tmp_path, monkeypatch):
    from tests.test_issue30_morning import unit
    from tests.test_issue30_raw_evidence import capture_chain
    capture_chain(tmp_path, monkeypatch)
    identity = unit()['A']
    output, inputs = m._slots(identity, True)
    assert m._judgment('A', identity, output, inputs)['reason'] == 'accepted'
    name = 'logs/generated/watcher/watcher.log'
    rebind(output, inputs, name, b'Not an application Watcher requirement for A')
    assert m._judgment('A', identity, output, inputs)['reason'] == 'accepted'
    (output / name).unlink()
    with pytest.raises(ValueError, match='^capture_artifact$'):
        m._judgment('A', identity, output, inputs)


@pytest.mark.parametrize('phase', ['A', 'B'])
@pytest.mark.parametrize('bad', [None, [], {'start_wall': -1}, {'launch_wall': 1.0}])
def test_available_clocks_are_checked_by_incomplete_audit(tmp_path, monkeypatch, phase, bad):
    from datetime import UTC, datetime

    from tests.test_issue30_morning import unit
    from tests.test_issue30_raw_evidence import capture_chain
    from tools import replay_issue30_evidence as replay
    capture_chain(tmp_path, monkeypatch)
    identity = unit()[phase]
    output, inputs = m._slots(identity, True)
    epoch = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    offset = 30 if phase == 'A' else 110
    clock = {'schema': 'issue30-capture-clock-v1', 'attempt': identity,
             'start_wall': epoch + offset - 1, 'end_wall': epoch + offset + 1,
             'start_monotonic': offset - 1, 'end_monotonic': offset + 1,
             'create_wall': epoch + offset - .5, 'create_monotonic': offset - .5,
             'launch_wall': epoch + offset - .25, 'launch_monotonic': offset - .25}
    rebind(output, inputs, 'capture-clock.json', json.dumps(clock).encode(), elapsed_seconds=2.0)
    # Adding valid clock evidence changes no conclusion or persisted result.
    assert replay._audit_capture(phase, identity)['reason'] == 'accepted'
    changed = clock | bad if type(bad) is dict else bad
    rebind(output, inputs, 'capture-clock.json', json.dumps(changed).encode())
    with pytest.raises(ValueError, match='^dram_capture_clock$'):
        replay._audit_capture(phase, identity)


@pytest.mark.parametrize('severity', ['info', 'debug', 'trace', None])
def test_DOWN_body_cannot_hide_behind_nonwarning_severity(severity):
    from tests.test_issue30 import log
    from tests.test_issue30_owner import smoke_stdout, transfer
    before = discovery_env()
    body = 'Chip: 0 PCIe: 0 Unique ID: 0 eth channel 8 core 1-28 link DOWN/unconnected (QSFP)'
    warning = log('warning', body, location='test_system_health.cpp:361').encode()
    assert transfer(before=before, stdout=smoke_stdout() + b'\n' + warning)['reason'] == 'accepted'
    changed = body.encode() if severity is None else log(
        severity, body, location='test_system_health.cpp:361').encode()
    assert transfer(before=before, stdout=smoke_stdout() + b'\n' + changed)['reason'] == 'diagnostic'


@pytest.mark.parametrize('path', [(), ('device_info',), ('device_info', 0),
    ('device_info', 0, 'board_info'), ('device_info', 0, 'firmwares'),
    ('device_info', 0, 'telemetry'), ('device_info', 0, 'smbus_telem')])
@pytest.mark.parametrize('bad', [None, False, 'shape', 1])
def test_raw_nested_shapes_are_checked_before_projection(path, bad):
    before, files = fixture('before')
    assert len(e.series(before, 'A', 'before', files.__getitem__)) == 5
    name = 'telemetry/before-A/snapshot1.stdout'
    value = json.loads(files[name])
    if not path:
        value = bad
    else:
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
    files[name] = json.dumps(value).encode()
    before['dram_admission']['samples'][0]['raw_sha256'] = m.digest(files[name])
    with pytest.raises(ValueError, match='^dram_raw_shape$'):
        e.series(before, 'A', 'before', files.__getitem__)


def test_renewed_UID_projection_is_bound_to_canonical_A_bytes(tmp_path, monkeypatch):
    from tests.test_issue30_morning import unit
    from tests.test_issue30_raw_evidence import capture_chain
    record = capture_chain(tmp_path, monkeypatch)
    output, inputs = m._slots(unit()['A'], True)
    bout, binp = m._slots(unit()['B'], True)
    before = json.loads((binp / 'before.json').read_bytes())
    assert m._discovery_uids(before, record) == {0: '0', 1: '1'}
    raw = (output / 'stdout.bin').read_bytes()
    changed = raw.replace(b'Unique ID: 0', b'Unique ID: abc', 1)
    assert changed != raw
    rebind(output, inputs, 'stdout.bin', changed)
    with pytest.raises(ValueError, match='^A_native_binding$'):
        m._discovery_uids(before, record)
    assert bout != output


@pytest.mark.parametrize('placement', ['leading_space', 'stderr_block'])
def test_UID_projection_uses_the_same_stream_rules_as_A(tmp_path, monkeypatch, placement):
    from tests.test_issue30_morning import unit
    from tests.test_issue30_raw_evidence import capture_chain
    capture_chain(tmp_path, monkeypatch)
    identity = unit()['A']
    output, inputs = m._slots(identity, True)
    raw = (output / 'stdout.bin').read_bytes()
    if placement == 'leading_space':
        changed = raw.replace(b'Chip: 0', b' Chip: 0', 1)
    else:
        start, end = raw.index(b'Chip: 0'), raw.index(b'Chip: 1')
        changed = raw[:start] + raw[end:]
        rebind(output, inputs, 'stderr.bin', raw[start:end])
    assert changed != raw
    rebind(output, inputs, 'stdout.bin', changed)
    checked = m._judgment('A', identity, output, inputs)
    assert checked['reason'] == 'accepted'
    (output / 'result.json').write_text(json.dumps(checked))
    before = json.loads((inputs / 'before.json').read_bytes())
    assert m._discovery_uids(before, checked) == {0: '0', 1: '1'}


def test_historical_UID_projection_uses_immutable_owner_native_hashes(tmp_path, monkeypatch):
    import tarfile
    owner = m._load(m.ROOT / m.OWNER_ACCEPTANCE, m.ROOT)
    archive = m.ROOT / 'docs/measurements/2026-11-24-issue30-a-discovery/A-native-evidence.tar.gz'
    with tarfile.open(archive) as source:
        raw = {name: source.extractfile(next(p for p in source.getmembers()
               if p.name.endswith(name))).read() for name in ('stdout.bin', 'stderr.bin', 'gtest.xml')}
    data = {'stdout.bin': raw['stdout.bin'], 'stderr.bin': raw['stderr.bin'], 'native/gtest.xml': raw['gtest.xml']}
    monkeypatch.setattr(m, 'WORK', tmp_path)
    # Model a locally coherent capture reader, leaving canonical binding real.
    monkeypatch.setattr(m, '_review_capture', lambda *args: ({}, {}))
    monkeypatch.setattr(m, '_captured', lambda out, execution, name: data[name])
    before = discovery_env()
    headers = [m._chip(line) for line in m._diagnostic_lines(data['stdout.bin']) if line.startswith('Chip: ')]
    expected = {chip: uid for chip, _, uid in headers}
    assert m._discovery_uids(before, owner) == expected
    old = ('Unique ID: ' + expected[0]).encode()
    data['stdout.bin'] = data['stdout.bin'].replace(old, b'Unique ID: abc', 1)
    with pytest.raises(ValueError, match='^A_native_binding$'):
        m._discovery_uids(before, owner)


class RequestClockHarness:
    def __init__(self, phase, boundary):
        self.before, files = fixture('before', 0 if phase == 'A' else 80)
        self.before['attempt'] = self.before['owner_unit'][phase]
        files = {k.replace('-A/', '-' + phase + '/'): v for k, v in files.items()}
        self.samples = e.series(self.before, phase, 'before', files.__getitem__)
        self.old_env, old_files = fixture('after', 40)
        self.old = e.series(self.old_env, 'A', 'after', old_files.__getitem__)
        self.last = self.samples[-1]
        self.age, self.attack, self.boundary = 10.0, False, boundary
        self.requests, self.errors = [], []
        self.original_freshness, self.original_open = e.launch_freshness, m.Path.open

    def install(self, monkeypatch, root):
        monkeypatch.setattr(m, 'ASSETS', root)
        monkeypatch.setattr(m, 'WORK', root)
        monkeypatch.setattr(m.os, 'getuid', lambda: 1000)
        monkeypatch.setattr(m.os, 'getgid', lambda: 1000)
        monkeypatch.setattr(m.time, 'time', lambda: self.last['monotonic_seconds'] + self.age)
        monkeypatch.setattr(m.time, 'monotonic', lambda: self.last['reported_monotonic_seconds'] + self.age)
        monkeypatch.setattr(m, '_raw_series', lambda env, p, role, root:
                            self.old if role == 'after' else self.samples)
        # The B predecessor is synthetic; the freshness checks remain real.
        monkeypatch.setattr(m, '_load', lambda *args: self.old_env)
        monkeypatch.setattr(m, '_docker_ps', lambda path, timeout: path.write_bytes(b''))
        monkeypatch.setattr(e, 'launch_freshness', self.freshness)
        monkeypatch.setattr(m.Path, 'open', lambda path, *args, **kw: self.opened(path, *args, **kw))
        monkeypatch.setattr(m.subprocess, 'run', self.run)
        monkeypatch.setattr(m.subprocess, 'Popen', self.popen)

    def freshness(self, values, wall, mono):
        try:
            self.original_freshness(values, wall, mono)
        except ValueError as exc:
            self.errors.append(str(exc))
            raise
        if self.attack and self.boundary == 'create_after_check':
            self.age = 60.001

    def opened(self, path, *args, **kwargs):
        stream = self.original_open(path, *args, **kwargs)
        if self.attack and self.boundary == 'start_after_open' and path.name == m.STDERR_FILE:
            self.age = 60.001
        return stream

    def run(self, argv, **kwargs):
        method = argv[argv.index('-X') + 1]
        path = argv[-1].split('/v1.52/')[1]
        if method == 'POST' and path.startswith('containers/create'):
            self.requests.append(('create', self.age))
            value = {'Id': 'a' * 64}
        elif path.endswith('/json'):
            value = {'State': {'ExitCode': 0, 'Running': False}}
        else:
            value = []
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(value).encode())

    def popen(self, *args, **kwargs):
        self.requests.append(('start', self.age))
        return Mock(wait=Mock(return_value=0))


@pytest.mark.parametrize('phase', ['A', 'B'])
@pytest.mark.parametrize('boundary', ['create_after_check', 'start_after_open'])
def test_request_clock_pair_is_checked_after_all_preparation(tmp_path, monkeypatch, phase, boundary):
    h = RequestClockHarness(phase, boundary)
    h.install(monkeypatch, tmp_path)
    positive_out = tmp_path / 'positive'
    positive = m.capture(phase, h.before['attempt'], tmp_path, positive_out, observations=h.before)
    assert positive['started']
    assert h.requests == [('create', 10.0), ('start', 10.0)]
    clock = json.loads((positive_out / 'capture-clock.json').read_bytes())
    for event in ('create', 'launch'):
        assert clock[event + '_wall'] == h.last['monotonic_seconds'] + 10
        assert clock[event + '_monotonic'] == h.last['reported_monotonic_seconds'] + 10
    h.age, h.attack = 59.999, True
    h.requests.clear()
    negative = m.capture(phase, h.before['attempt'], tmp_path, tmp_path / 'negative', observations=h.before)
    assert not negative['started']
    assert h.errors == ['dram_launch_freshness']
    target = 'create' if boundary == 'create_after_check' else 'start'
    assert not any(kind == target for kind, _ in h.requests)


def test_native_teardown_banner_is_not_a_DOWN_summary():
    from tests.test_issue30_owner import smoke_stdout, transfer
    raw = smoke_stdout() + b'\n[----------] Global test environment tear-down\n'
    assert transfer(stdout=raw)['reason'] == 'accepted'


def test_expired_container_start_budget_refuses_before_Popen(tmp_path, monkeypatch):
    now, calls = [0.0], []
    monkeypatch.setattr(m.time, 'monotonic', lambda: now[0])
    def popen(*args, **kwargs):
        calls.append('start')
        return Mock(wait=Mock(return_value=0))
    monkeypatch.setattr(m.subprocess, 'Popen', popen)
    monkeypatch.setattr(m, '_api', lambda *args, **kwargs: {'State': {'ExitCode': 0, 'Running': False}})
    first, last = tmp_path / 'first', tmp_path / 'last'
    first.mkdir()
    last.mkdir()
    m._run_client('synthetic', 'a' * 64, first, 0, {'timed_out': False})
    assert calls == ['start']
    now[0] = 50.001
    with pytest.raises(subprocess.TimeoutExpired, match='docker start budget'):
        m._run_client('synthetic', 'a' * 64, last, 0, {'timed_out': False})
    assert calls == ['start']


@pytest.mark.parametrize('clock', ['wall', 'monotonic'])
def test_owner_approved_sixty_seconds_is_inclusive_without_extra_slack(clock):
    before, files = fixture('before')
    samples = e.series(before, 'A', 'before', files.__getitem__)
    last = samples[-1]
    wall, mono = last['monotonic_seconds'] + 60, last['reported_monotonic_seconds'] + 60
    e.launch_freshness(samples, wall, mono)
    wall = math.nextafter(wall, math.inf) if clock == 'wall' else wall
    mono = math.nextafter(mono, math.inf) if clock == 'monotonic' else mono
    with pytest.raises(ValueError, match='^dram_launch_freshness$'):
        e.launch_freshness(samples, wall, mono)
