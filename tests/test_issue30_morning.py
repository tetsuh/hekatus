"""Synthetic renewed-unit controls; never physical owner approval."""
import copy

import pytest

from enodia.tt.bench import issue30 as m
from tests.test_issue30_dram import series
from tests.test_issue30_topology import env as original_env
from tests.test_issue30_topology import judge, records


def env():
    value = original_env()
    readings = series()
    for sample in readings['samples']:
        for index, card in enumerate(sample['cards']):
            card['serial'] = value['cards'][index]['serial']
    return value | {'dram_admission': readings}


def unit():
    return {"date": "2026-11-25", "A": "issue30-a-discovery-watcher-v2-synthetic",
            "B": "issue30-b-fabric1d-smoke-watcher-v2-synthetic"}


def test_fresh_h1_binds_new_unit_and_rejects_single_mutations():
    before = env() | {"owner_unit": unit()}
    result = judge(records(), before=before)
    m._fresh_links(before, result)
    for field, value in (("baseline_sha256", "0" * 64), ("verdict", "rejected"), ("phase", "B")):
        changed = copy.deepcopy(result); changed[field] = value
        with pytest.raises(ValueError, match="fresh_A"):
            m._fresh_links(before, changed)
    with pytest.raises(ValueError, match="fresh_A"):
        m._fresh_links(before | {"boot": "new-boot"}, result)
    changed = copy.deepcopy(result); changed["observed_qsfp"][0]["retrain_count"] = 1
    with pytest.raises(ValueError, match="fresh_A"):
        m._fresh_links(before, changed)


def test_new_A_only_accepts_H1_and_prior_down_channels():
    before = env() | {"owner_unit": unit()}
    assert judge(records(), before=before)["verdict"] == "accepted_discovery"
    assert judge(records("H2"), before=before)["verdict"] != "accepted_discovery"


def test_renewed_approval_is_not_old_authority():
    before = env() | {"owner_unit": unit()}
    for phase in ("A", "B"):
        value = m.approval_record(phase, unit()[phase], before, {}, "synthetic-head")
        assert value["owner_authorization"]["date"] == "2026-11-25"
        assert value["owner_authorization"]["invocations"] == 1
        assert value["owner_unit"] == unit()
    with pytest.raises(ValueError, match="owner_unit"):
        m.approval_record("B", "wrong-id", before, {}, "synthetic-head")


def test_fresh_native_revalidates_canonical_capture_before_B(tmp_path, monkeypatch):
    from tests.test_issue30 import xml
    from tests.test_issue30_topology import report

    before = env() | {"owner_unit": unit(), "attempt": unit()["A"]}
    raw = {"gtest.xml": xml(), "stdout.bin": report(records()), "stderr.bin": b"",
           "before.json": b"before", "after.json": b"after"}
    value = m.evaluate("A", unit()["A"], before, before, xml(), raw["stdout.bin"], b"", m.GOOD_EXECUTION)
    value["xml_present"] = True
    values = {"result.json": value, "before.json": before, "after.json": before,
              "approval.json": {"synthetic": True}, m.CURRENT_RECEIPT.name: {}}
    review = {k + "_sha256": m.digest(raw[k + ".json"]) for k in ("before", "after")}
    monkeypatch.setattr(m, "_load", lambda path, *args: values[path.name])
    monkeypatch.setattr(m, "_read", lambda path, *args, **kw: raw[path.name])
    execution = m.GOOD_EXECUTION | {'artifacts': [
        {'path': name, 'bytes': len(raw[name.split('/')[-1]]),
         'sha256': m.digest(raw[name.split('/')[-1]])}
        for name in ('native/gtest.xml', 'stdout.bin', 'stderr.bin')]}
    monkeypatch.setattr(m, "_review", lambda *args: (execution, review))
    monkeypatch.setattr(m, "_observations", lambda *args: (before, before))
    checked = []
    monkeypatch.setattr(m, "_approval", lambda *args: checked.append(args))
    monkeypatch.setattr(m, "_git", lambda *args: "synthetic-head")
    m._fresh_native(before, value)
    assert checked[0][:2] == ("A", unit()["A"])
    for field, replacement in (("attempt", "old-a"), ("verdict", "rejected")):
        changed = copy.deepcopy(value); changed[field] = replacement
        with pytest.raises(ValueError, match="fresh_A_canonical"):
            m._fresh_native(before, changed)
    raw["stdout.bin"] += b"warning | unexpected diagnostic\n"
    with pytest.raises(ValueError, match="capture_artifact"):
        m._fresh_native(before, value)
