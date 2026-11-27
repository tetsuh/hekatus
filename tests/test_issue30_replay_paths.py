"""Replay CLI paths stay within fixed evidence and private task roots."""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from enodia.tt.bench import issue30 as m
from tools import replay_issue30_evidence as r


@pytest.fixture
def root():
    path = Path(tempfile.mkdtemp(prefix='tidd-195-path-test-', dir=tempfile.gettempdir()))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_canonical_private_paths_and_baseline(root):
    value = {'results': []}
    (root / 'replay-baseline.json').write_text(json.dumps(value))
    assert r._private_root(root) == root
    assert r._baseline(root / 'replay-baseline.json') == value
    assert r._output_path(root / 'new-copy') == root / 'new-copy'
    assert r._source_root(m.ROOT / 'out/issue30') == m.ROOT / 'out/issue30'


@pytest.mark.parametrize('value', ['/etc/passwd', '/tmp/replay-baseline.json',
    '/tmp/tidd-195-valid/../secret/replay-baseline.json'])
def test_baseline_cannot_read_arbitrary_paths(value):
    path = Path(value)
    with pytest.raises(ValueError, match='replay_'):
        r._baseline(path)


def test_fixed_baseline_filename_and_root_mode(root):
    path = root / 'secret.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='replay_baseline_scope'):
        r._baseline(path)
    root.chmod(0o755)
    with pytest.raises(ValueError, match='replay_private_root'):
        r._private_root(root)
    root.chmod(0o700)


def test_redirected_baseline_root_and_output_refuse(root):
    target = root / 'actual.json'
    target.write_text('{}')
    (root / 'replay-baseline.json').symlink_to(target)
    with pytest.raises(ValueError, match='canonical'):
        r._baseline(root / 'replay-baseline.json')
    link = root / 'redirected'
    link.symlink_to(m.ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match='canonical'):
        r._output_path(link)
    with pytest.raises(ValueError, match='replay_source_scope'):
        r._source_root(root)


def test_root_prefix_and_output_components_refuse(root):
    bad_root = root / 'nested'
    bad_root.mkdir()
    with pytest.raises(ValueError, match='replay_private_root'):
        r._private_root(bad_root)
    bad_output = root / 'bad.output'
    with pytest.raises(ValueError, match='replay_output_scope'):
        r._output_path(bad_output)
