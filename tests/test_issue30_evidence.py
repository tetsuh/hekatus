"""Single-mutation evidence controls, with no device or daemon access."""
import copy
import json

import pytest

from enodia.tt.bench import issue30 as m
from enodia.tt.bench import issue196_compile as compile_check
from tests.test_issue30 import xml
from tests.test_issue30_admission import discovery_env
from tests.test_issue30_owner import accepted, baseline_lines, smoke_stdout, transfer


def test_captured_reader_requires_exact_unique_manifest(tmp_path):
    path = tmp_path / 'stdout.bin'
    path.write_bytes(b'captured')
    entry = m.artifact(path, tmp_path)
    execution = {'artifacts': [entry]}
    assert m._captured(tmp_path, execution, 'stdout.bin') == b'captured'
    for entries in ([], [entry, entry], [entry | {'sha256': '0' * 64}]):
        with pytest.raises(ValueError, match='capture_artifact'):
            m._captured(tmp_path, {'artifacts': entries}, 'stdout.bin')
    path.write_bytes(b'CaptureD')
    with pytest.raises(ValueError, match='capture_artifact'):
        m._captured(tmp_path, execution, 'stdout.bin')


def test_xml_requires_direct_suite_and_case():
    original = xml()
    m._xml('A', original)
    for old, new in ((b'<testsuite ', b'<wrapper><testsuite '),):
        changed = original.replace(old, new).replace(b'</testsuite>', b'</testsuite></wrapper>')
        with pytest.raises(ValueError, match='test_result'):
            m._xml('A', changed)


def test_utf16_dtd_is_forbidden():
    m._xml('A', xml())
    changed = '<!DOCTYPE testsuites>' + xml().decode()
    encoded = changed.encode('utf-16')
    with pytest.raises(ValueError, match='test_result'):
        m._xml('A', encoded)


def test_client_exit_boolean_is_not_integer_zero():
    m._execution(m.GOOD_EXECUTION, 'A')
    with pytest.raises(ValueError, match='execution'):
        m._execution(m.GOOD_EXECUTION | {'client_exit': False}, 'A')


@pytest.mark.parametrize('change', ['prefix', 'duplicate', 'order', 'mode'])
def test_native_markers_are_unique_anchored_ordered(change):
    assert transfer()['reason'] == 'accepted'
    raw = smoke_stdout()
    if change == 'prefix':
        raw = raw.replace(b'[ RUN      ]', b'quoted [ RUN      ]')
    elif change == 'duplicate':
        raw += b'[  PASSED  ] 1 test.\n'
    elif change == 'order':
        raw = raw.replace(b'[  PASSED  ] 1 test.\n', b'')
        raw = b'[  PASSED  ] 1 test.\n' + raw
    else:
        raw = raw.replace(b'STRICT_SYSTEM_HEALTH_SETUP_MODE', b'NON_STRICT')
    assert transfer(stdout=raw)['reason'] == 'packet_evidence'


def test_baseline_prefix_and_cross_stream_pair_cannot_hide_alarm():
    pair = baseline_lines()[:2]
    raw = ('\n'.join(pair) + '\n').encode() + smoke_stdout()
    assert transfer(stdout=raw)['reason'] == 'accepted'
    changed = b'critical | ' + raw
    assert transfer(stdout=changed)['reason'] == 'diagnostic'
    before = discovery_env()
    owner = accepted(before)
    value = m.evaluate('B', 'synthetic-a', before, before, xml('B'),
                       pair[0].encode() + b'\n' + smoke_stdout(), pair[1].encode(),
                       m.GOOD_EXECUTION, owner)
    assert value['reason'] == 'diagnostic'


def test_cleanup_other_container_is_not_empty(tmp_path, monkeypatch):
    def invoke(response):
        monkeypatch.setattr(m, '_api', lambda *a, **k: response)
        monkeypatch.setattr(m, '_docker_ps', lambda path, *a: path.write_bytes(b''))
        state = {}
        m._cleanup('own', None, tmp_path, m.time.monotonic(), state)
        (tmp_path / 'docker-after.json').unlink()
        return state['containers_clean']
    assert invoke([]) is True
    assert invoke([{'Names': ['/unrelated']}]) is False
    assert invoke({}) is False


def test_offline_cases_bind_associations_and_source_hash_coverage():
    original = json.loads(compile_check.MANIFEST.read_bytes())
    compile_check.validate_manifest(original)
    for mutation in ('processor', 'source_hash'):
        changed = copy.deepcopy(original)
        if mutation == 'processor':
            next(c for c in changed['cases'] if c['id'] == 'captured-rx')['processor'] = 'ncrisc'
        else:
            changed['kernel_source_sha256'].pop(next(iter(changed['kernel_source_sha256'])))
        with pytest.raises(ValueError, match='provenance'):
            compile_check.validate_manifest(changed)
