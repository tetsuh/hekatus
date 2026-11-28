"""Mock-only lifecycle, sysfs association and atomic publication controls."""
import subprocess
from unittest.mock import Mock

import pytest

from enodia.tt.bench import issue30 as m


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'WORK', tmp_path)
    monkeypatch.setattr(m, 'ASSETS', tmp_path)
    monkeypatch.setattr(m.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(m.os, 'getgid', lambda: 1000)
    monkeypatch.setattr(m, '_docker_ps', lambda path, timeout: path.write_bytes(b''))


def test_lost_create_response_is_explicitly_unknown_not_clean(tmp_path, monkeypatch):
    calls, lost = [], [False]
    def api(method, path, *args, **kwargs):
        calls.append((method, path))
        if path.startswith('containers/create'):
            if lost[0]:
                raise subprocess.TimeoutExpired('synthetic lost response', 1)
            return {'Id': 'a' * 64}
        return []
    monkeypatch.setattr(m, '_api', api)
    monkeypatch.setattr(m, '_run_client', lambda name, cid, out, start, state: state.update(
        started=True, terminated=True, exit=0, client_exit=0))
    positive = m.capture('A', 'positive', tmp_path, tmp_path / 'positive')
    assert positive['create_status'] == 'identified'
    assert positive['containers_clean']
    lost[0] = True
    calls.clear()
    state = m.capture('A', 'synthetic', tmp_path, tmp_path / 'attempt')
    assert not state['started']
    assert not state['terminated']
    assert state['exit'] is None
    assert state['create_status'] == 'unknown'
    assert state['containers_clean'] is False
    assert not any(method in ('DELETE', 'POST') and 'create' not in path for method, path in calls)


def test_start_and_ps_pin_the_api_socket_and_start_id(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv('DOCKER_HOST', 'tcp://wrong.invalid:2375')
    monkeypatch.setenv('DOCKER_CONTEXT', 'wrong-context')
    def popen(argv, **kwargs):
        assert not any(k.startswith('DOCKER_') for k in kwargs['env'])
        calls.append(argv)
        return Mock(wait=Mock(return_value=0))
    monkeypatch.setattr(m.subprocess, 'Popen', popen)
    monkeypatch.setattr(m, '_api', lambda *args, **kwargs: {'State': {'ExitCode': 0, 'Running': False}})
    m._run_client('wrong-name', 'a' * 64, tmp_path, m.time.monotonic(), {'timed_out': False})
    assert calls == [[*m.DOCKER_CLI, 'start', '-a', '--', 'a' * 64]]
    assert m.DOCKER_CLI == ('docker', '--host', 'unix://' + m.DOCKER_SOCKET)


@pytest.mark.parametrize('first_node', ['0', '000000'])
def test_sysfs_resolution_binds_node_to_approved_pci(monkeypatch, first_node):
    from tests.test_issue30 import environment
    env = environment()
    for card, component in zip(env['cards'], (first_node, '1'), strict=True):
        card['node'] = f'/dev/tenstorrent/{component}'
    actual = {first_node: '0000:06:00.0', '1': '0000:09:00.0'}
    original = m.Path.resolve
    def resolve(path, *args, **kwargs):
        if str(path).startswith('/sys/class/tenstorrent/'):
            node = path.parent.name.split('!')[1]
            return m.Path('/sys/devices/pci0000:00') / actual[node]
        if str(path).startswith('/sys/bus/pci/devices/'):
            return m.Path('/sys/devices/pci0000:00') / path.name
        return original(path, *args, **kwargs)
    monkeypatch.setattr(m.Path, 'resolve', resolve)
    assert m._approved_nodes(env) == [f'/dev/tenstorrent/{first_node}', '/dev/tenstorrent/1']
    actual['1'] = '0000:ff:00.0'
    with pytest.raises(ValueError, match='device_pci_binding'):
        m._approved_nodes(env)


def test_preexisting_staging_is_not_owned_or_removed(tmp_path):
    target = tmp_path / 'result.json'
    staging = tmp_path / 'result.json.tmp'
    staging.write_bytes(b'preexisting')
    with pytest.raises(FileExistsError):
        m.publish(target, {})
    assert staging.read_bytes() == b'preexisting'
    assert not target.exists()


def test_failed_rename_preserves_destination_and_cleans_staging(tmp_path, monkeypatch):
    target = tmp_path / 'result.json'
    m.publish(target, {'positive': True})
    target.unlink()
    def race(temporary, destination):
        destination.write_bytes(b'other publisher')
        raise FileExistsError('synthetic no-replace race')
    monkeypatch.setattr(m, '_rename', race)
    with pytest.raises(FileExistsError):
        m.publish(target, {})
    assert target.read_bytes() == b'other publisher'
    assert not (tmp_path / 'result.json.tmp').exists()


def test_successful_rename_does_not_remove_a_later_publishers_staging(tmp_path, monkeypatch):
    target = tmp_path / 'result.json'
    m.publish(target, {'positive': True})
    assert not (tmp_path / 'result.json.tmp').exists()
    target.unlink()
    original = m._rename
    def rename(temporary, destination):
        original(temporary, destination)
        temporary.write_bytes(b'later publisher')
    monkeypatch.setattr(m, '_rename', rename)
    m.publish(target, {'positive': True})
    assert (tmp_path / 'result.json.tmp').read_bytes() == b'later publisher'


def test_production_still_refuses_uid_zero(tmp_path):
    m.request('A', 'synthetic', tmp_path, tmp_path, uid=1000, gid=1000)
    with pytest.raises(ValueError, match='non-root'):
        m.request('A', 'synthetic', tmp_path, tmp_path, uid=0, gid=0)


@pytest.mark.parametrize('when', ['before_create', 'after_create'])
def test_dram_is_rechecked_after_prelaunch_work_before_create(tmp_path, monkeypatch, when):
    from enodia.tt.bench import issue30_evidence as e
    from tests.test_issue30_raw_evidence import fixture
    before, files = fixture('before')
    samples = e.series(before, 'A', 'before', files.__getitem__)
    last = samples[-1]
    age, delay, calls, errors = [10.0], [False], [], []
    monkeypatch.setattr(m.time, 'monotonic', lambda: last['reported_monotonic_seconds'] + age[0])
    monkeypatch.setattr(m.time, 'time', lambda: last['monotonic_seconds'] + age[0])
    monkeypatch.setattr(m, '_raw_series', lambda *args: samples)
    original = m._admit_series
    def admit(*args):
        try:
            return original(*args)
        except ValueError as exc:
            errors.append(str(exc))
            raise
    monkeypatch.setattr(m, '_admit_series', admit)
    def ps(path, timeout):
        if delay[0] and when == 'before_create' and path.name == 'docker-before.txt':
            age[0] = 61.0
        path.write_bytes(b'')
    def api(method, path, *args, **kwargs):
        boundary = kwargs.get('before_request')
        if boundary is not None:
            boundary()
        calls.append(path)
        if delay[0] and when == 'after_create' and path.startswith('containers/create'):
            age[0] = 61.0
        return {'Id': 'a' * 64} if path.startswith('containers/create') else []
    def client(name, cid, out, start, state, *, before_start=None):
        if before_start is not None:
            before_start()
        state.update(started=True, terminated=True, exit=0, client_exit=0)
    monkeypatch.setattr(m, '_docker_ps', ps)
    monkeypatch.setattr(m, '_api', api)
    monkeypatch.setattr(m, '_run_client', client)
    identity = before['attempt']
    positive = m.capture('A', identity, tmp_path, tmp_path / 'positive', observations=before)
    assert positive['create_status'] == 'identified'
    assert not errors
    delay[0] = True
    calls.clear()
    negative = m.capture('A', identity, tmp_path, tmp_path / 'negative', observations=before)
    assert not negative['started']
    assert errors == ['dram_launch_freshness']
    if when == 'before_create':
        assert negative['create_status'] == 'not_requested'
        assert not any(path.startswith('containers/create') for path in calls)
    else:
        assert negative['create_status'] == 'identified'
        assert 'containers/' + 'a' * 64 + '?force=1' in calls
