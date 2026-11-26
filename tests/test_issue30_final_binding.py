"""Final correction controls: actual identity, not independent set membership."""
import copy

import pytest

from enodia.tt.bench import issue30 as m
from tests.test_issue30_morning import env, unit
from tests.test_issue30_owner import baseline_lines, smoke_stdout, transfer
from tests.test_issue30_topology import judge, records
from tools import replay_issue30_evidence as replay


def test_warning_bus_belongs_to_its_chip():
    line = next(x for x in baseline_lines() if 'bus_id=' in x)
    good = line.replace('chip_id=0', 'chip_id=1')
    assert transfer(stdout=good.encode() + b'\n' + smoke_stdout())['reason'] == 'accepted'
    for old, new in (('bus_id=0x9', 'bus_id=0xff'), ('chip_id=1', 'chip_id=0')):
        changed = good.replace(old, new)
        assert m.diagnostics._condition(changed, env()['cards']) is None


def test_fresh_links_requires_the_actual_unit_attempt():
    before = env() | {'owner_unit': unit()}
    result = judge(records(), before=before) | {'attempt': unit()['A']}
    m._fresh_links(before, result)
    changed = copy.deepcopy(result); changed['attempt'] = 'other-attempt'
    with pytest.raises(ValueError, match='fresh_A_binding'):
        m._fresh_links(before, changed)


def test_os_temp_root_and_checkout_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.tempfile, 'gettempdir', lambda: str(tmp_path))
    selected = tmp_path / 'tidd-195-test'
    selected.mkdir(mode=0o700)
    assert replay._private_root(selected) == selected
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='replay_private_root'):
        replay._private_root(selected)
