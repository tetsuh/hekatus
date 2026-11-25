"""Synthetic owner-scoped five-reading DRAM admission, never telemetry."""
import copy

import pytest

from enodia.tt.bench import issue30_admission as a


def series():
    return {'schema': 'issue30-dram-five-v1', 'samples': [
        {'monotonic_seconds': 100.0 + i * 5, 'raw_sha256': 'a' * 64,
         'cards': [{'serial': str(c), 'dram_status': True, 'raw_ddr_status': '0x5555',
                    'uncorrectable_errors': 0, 'arc_heartbeat': 100 + i, 'aiclk_mhz': 800}
                   for c in (0, 1)]} for i in range(5)]}


def before():
    return {'owner_unit': {'date': '2026-11-25'}, 'cards': [{'serial':str(c)} for c in (0,1)],
            'dram_admission': series()}


def test_five_good_admits():
    a.dram_gate(before())


@pytest.mark.parametrize('index',range(5))
@pytest.mark.parametrize('field,value',[('dram_status',False),('raw_ddr_status','0x0'),
                                       ('uncorrectable_errors',1),('arc_heartbeat',-1),('aiclk_mhz',700)])
def test_one_abnormal_refuses(index,field,value):
    value_before=before();value_before['dram_admission']['samples'][index]['cards'][1][field]=value
    with pytest.raises(ValueError,match='dram_gate'):
        a.dram_gate(value_before)


@pytest.mark.parametrize('kind',('series','sample','card','field','short_interval','missing_raw','frozen_arc'))
def test_missing_or_incomplete_refuses(kind):
    value=before();s=value['dram_admission']['samples']
    if kind=='series':del value['dram_admission']
    elif kind=='sample':s.pop()
    elif kind=='card':s[2]['cards'].pop()
    elif kind=='field':del s[2]['cards'][1]['raw_ddr_status']
    elif kind=='short_interval':s[1]['monotonic_seconds']=104.9
    elif kind=='missing_raw':del s[2]['raw_sha256']
    else:s[2]['cards'][1]['arc_heartbeat']=s[1]['cards'][1]['arc_heartbeat']
    with pytest.raises(ValueError,match='dram_gate'):
        a.dram_gate(value)


def test_series_is_dynamic_evidence_not_stable_baseline_or_old_scope():
    original=before();changed=copy.deepcopy(original)
    changed['dram_admission']['samples'][0]['raw_sha256']='b'*64
    assert a.stable_environment(original)==a.stable_environment(changed)
    a.dram_gate({'cards': []})  # Original historical scope is not silently revised.


def test_board_request_refuses_missing_series_before_approval_or_mapping(tmp_path):
    from enodia.tt.bench import issue30 as m
    from tests.test_issue30_morning import env, unit

    value=env() | {'owner_unit':unit()}
    value.pop('dram_admission')
    with pytest.raises(ValueError,match='dram_gate'):
        m.board_request('A', value['attempt'], tmp_path, tmp_path/'out',
                        approval={}, environment=value, receipt={}, head='synthetic')
