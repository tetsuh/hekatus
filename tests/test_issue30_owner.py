"""Owner-gated CPU controls; all execution/boot/packet fixtures are synthetic."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from enodia.tt.bench import issue30 as m
from enodia.tt.bench.issue120_production import checks
from tests.test_issue30 import xml
from tests.test_issue30_admission import discovery_env
from tests.test_issue30_topology import records, report


def accepted(before=None):
    before = discovery_env() if before is None else before
    return {"schema": "issue30-owner-accepted-A-v1", "decision_by": "owner",
            "decision_date": "2026-11-24", "logical_unit": "issue30-a-discovery-watcher-v1",
            "not_a_rejudgment": True, "allows_B_consideration": True,
            "original_verdict": {"phase": "A", "verdict": "rejected", "reason": "watcher"},
            "baseline_sha256": m.baseline_digest(before),
            "boot_sha256": m.digest(before["boot"].encode()),
            "cabling_policy_sha256": m.digest(json.dumps(before["topology"], sort_keys=True).encode()),
            "bundle": "19.6.0.0", "observed_qsfp": records(),
            "topology": {"outcome": "accepted", "reason": "accepted", "hypothesis": "H1"}}


def baseline_lines():
    baseline = json.loads(checks.BASELINE.read_bytes())
    return ["2026-01-01 00:00:00.000 | warning  | " + condition["matches"][0]["text"]
            for condition in baseline["evidence"]]


def smoke_stdout():
    return (b"Running main() from gmock_main.cc\n"
            b"info | Running fabric api tests with fast dispatch\n"
            b"info | Fabric Reliability Mode: STRICT_SYSTEM_HEALTH_SETUP_MODE\n"
            b"[ RUN      ] Fabric1DFixture.TestUnicastConnAPI1DSmoke\n"
            b"[       OK ] Fabric1DFixture.TestUnicastConnAPI1DSmoke (125 ms)\n"
            b"[  PASSED  ] 1 test.\n")


def transfer(*, before=None, owner=None, stdout=None, execution=None, raw_xml=None):
    before = discovery_env() if before is None else before
    return m.evaluate("B", "synthetic-a", before, before, xml("B") if raw_xml is None else raw_xml,
                      smoke_stdout() if stdout is None else stdout, b"",
                      m.GOOD_EXECUTION if execution is None else execution,
                      accepted(before) if owner is None else owner)


def test_owner_accepted_A_releases_only_same_boot_B_without_firmware_invention():
    value = transfer()
    assert value["verdict"] == "accepted_transfer"
    assert value["firmware_observation"]["eth_heartbeat_health"] == "unknown"
    assert value["packet_observation"]["expected_packets_from_pinned_source"] == 5
    assert value["packet_observation"]["basis"] == "source_bound_official_receiver_assertions"


def test_owner_A_observation_does_not_rewrite_original_verdict():
    owner = accepted()
    snapshot = copy.deepcopy(owner)
    assert transfer(owner=owner)["verdict"] == "accepted_transfer"
    assert owner == snapshot
    assert owner["original_verdict"]["verdict"] == "rejected"


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("boot_sha256", "0" * 64, "owner_boot"),
    ("baseline_sha256", "0" * 64, "discovery_binding"),
    ("cabling_policy_sha256", "0" * 64, "discovery_binding"),
    ("decision_by", "operator", "discovery_binding"),
    ("allows_B_consideration", False, "discovery_binding"),
])
def test_single_owner_binding_mutation(field, value, reason):
    assert transfer()["verdict"] == "accepted_transfer"
    owner = accepted(); owner[field] = value
    assert transfer(owner=owner)["reason"] == reason


def test_changed_boot_refuses_even_when_both_new_samples_match():
    assert transfer()["verdict"] == "accepted_transfer"
    before = discovery_env(); owner = accepted(before); before["boot"] = "changed-boot"
    assert transfer(before=before, owner=owner)["reason"] == "owner_boot"


def test_owner_label_cannot_hide_partial_or_retrained_A():
    assert transfer()["verdict"] == "accepted_transfer"
    owner = accepted(); owner["observed_qsfp"] = records(active={4, 6})
    assert transfer(owner=owner)["reason"] == "discovery_binding"
    owner = accepted(); owner["observed_qsfp"][0]["retrain_count"] = 1
    assert transfer(owner=owner)["reason"] == "discovery_binding"


def test_A_watcher_is_not_applicable_but_B_missing_is_refused():
    before = discovery_env()
    a = m.evaluate("A", "synthetic-a", before, before, xml(), report(records()), b"",
                   m.GOOD_EXECUTION | {"watcher_clean": None})
    assert a["verdict"] == "accepted_discovery"
    assert transfer()["verdict"] == "accepted_transfer"
    assert transfer(execution=m.GOOD_EXECUTION | {"watcher_clean": None})["reason"] == "watcher"


def test_exact_three_baselines_and_observed_down_are_allowed():
    assert transfer()["verdict"] == "accepted_transfer"
    lines = baseline_lines()
    # Only invocation variables change, including the second enumerated chip.
    lines = [line.replace("physical_device_id: 1", "physical_device_id: 0")
             .replace("chip_id=0", "chip_id=1").replace("bus_id=0x9", "bus_id=0x6")
             .replace("pid: 1 tid: 136557857326912", "pid: 7 tid: 99") for line in lines]
    down = [line for line in report(records()).decode().splitlines() if line.startswith("warning |")]
    raw = ("\n".join(lines + down) + "\n").encode() + smoke_stdout()
    assert transfer(stdout=raw)["verdict"] == "accepted_transfer"


@pytest.mark.parametrize(("old", "new"), [
    ("Operation not permitted", "Cannot allocate memory"), ("NodeSet: [0]", "NodeSet: [1]"),
    ("ch: 0", "ch: 1"), ("(cpuset_lib.cpp:440)", "(cpuset_lib.cpp:441)"),
    ("physical_device_id: 1", "physical_device_id: 999"),
    ("TUF GAMING Z790-PLUS WIFI D4", "different board"),
    ("SMC buffer is unavailable", "SMC buffer is corrupt"),
])
def test_baseline_single_literal_mutation_is_not_whitelisted(old, new):
    raw = ("\n".join(baseline_lines()) + "\n").encode() + smoke_stdout()
    assert transfer(stdout=raw)["verdict"] == "accepted_transfer"
    changed = raw.replace(old.encode(), new.encode(), 1)
    assert changed != raw
    assert transfer(stdout=changed)["reason"] == "diagnostic"


def test_numa_fallback_must_be_adjacent_and_same_device():
    pair = baseline_lines()[:2]
    assert transfer(stdout=("\n".join(pair) + "\n").encode() + smoke_stdout())["reason"] == "accepted"
    raw = (pair[0] + "\ninfo | intervening message\n" + pair[1] + "\n").encode() + smoke_stdout()
    assert transfer(stdout=raw)["reason"] == "diagnostic"


def test_B_down_warning_requires_exact_owner_A_channel_core_and_identity():
    good = b"warning | Chip: 0 PCIe: 0 Unique ID: 0 eth channel 8 core 1-28 link DOWN/unconnected (QSFP)\n"
    assert transfer(stdout=good + smoke_stdout())["reason"] == "accepted"
    for old, new in [(b"channel 8 core 1-28", b"channel 4 core 1-24"),
                     (b"core 1-28", b"core 99-99"), (b"PCIe: 0", b"PCIe: 9")]:
        assert transfer(stdout=good.replace(old, new) + smoke_stdout())["reason"] == "diagnostic"


@pytest.mark.parametrize("line", [b"warning | other warning\n", b"error | bad\n",
                                   b"TT_FATAL: bad\n", b"Traceback: bad\n"])
def test_new_diagnostics_always_stop_B(line):
    assert transfer()["reason"] == "accepted"
    assert transfer(stdout=line + smoke_stdout())["reason"] == "diagnostic"


def test_known_bad_firmware_and_reasonless_unknown_still_refuse_B():
    assert transfer()["reason"] == "accepted"
    for field, value, reason in [("erisc", "1.4.0", "environment"),
                                 ("eth_heartbeat_healthy", False, "environment")]:
        before = discovery_env(); before["cards"][0][field] = value
        assert transfer(before=before)["reason"] == reason
    before = discovery_env(); del before["cards"][0]["unobservable"]["erisc"]
    assert transfer(before=before)["reason"] == "unobservable_reason"


def test_missing_packet_completion_or_xml_failure_is_not_success():
    assert transfer()["reason"] == "accepted"
    assert transfer(stdout=b"info | fabric initialized\n")["reason"] == "packet_evidence"
    bad = xml("B").replace(b"/></testsuite>", b"><failure>mismatch</failure></testcase></testsuite>")
    assert transfer(raw_xml=bad)["reason"] == "test_result"


def test_native_and_all_previous_A_files_remain_immutable():
    root = m.ROOT / "docs/measurements/2026-11-24-issue30-a-discovery"
    value = json.loads((root / "A-discovery.json").read_bytes())
    assert value["formal_judgment"]["verdict"] == "rejected"
    assert value["formal_judgment"]["reason"] == "watcher"
    assert hashlib.sha256((root / "A-native-evidence.tar.gz").read_bytes()).hexdigest() == value["raw_archive"]["sha256"]


@pytest.mark.parametrize("change", ["missing", "no_dump", "abnormal", "not_captured", "changed_after_capture", "unreviewed"])
def test_B_Watcher_requires_captured_fresh_dump_and_clean_hash_bound_review(tmp_path, monkeypatch, change):
    monkeypatch.setattr(m, "WORK", tmp_path)
    output, inputs = tmp_path / "run", tmp_path / "inputs"
    output.mkdir(); inputs.mkdir()
    before = discovery_env()
    m.publish(inputs / "before.json", before)
    m.publish(inputs / "after.json", before)
    m.publish(output / "observations-before.json", before)
    for role in ("before", "after"):
        m.publish(output / ("docker-" + role + ".json"), {"containers": []})
        (output / ("docker-" + role + ".txt")).write_bytes(b"")
    other_entries = [m.artifact(path, output) for path in output.iterdir() if path.is_file()]
    watcher = output / "logs/generated/watcher/watcher.log"
    watcher.parent.mkdir(parents=True)
    raw = b"Dump #1 at synthetic time\nsynthetic healthy waypoint\n"
    watcher.write_bytes(raw)
    entry = m.artifact(watcher, output)
    execution = m.GOOD_EXECUTION | {"schema": "issue30-execution-v1", "attempt": "synthetic-a",
                                     "device_free": False, "artifacts": other_entries + [entry],
                                     "image": m.IMAGE,
                                     "binary_sha256": m._load(m.CURRENT_RECEIPT, m.ROOT)['binaries'][m.BINARIES['B']]}
    m.publish(output / "execution.json", execution)
    review = {"schema": "issue30-review-v1", "attempt": "synthetic-a", "watcher_clean": True,
              "execution_sha256": m.digest((output / "execution.json").read_bytes()),
              "watcher_sha256": entry["sha256"],
              "before_sha256": m.digest((inputs / "before.json").read_bytes()),
              "after_sha256": m.digest((inputs / "after.json").read_bytes())}
    m.publish(inputs / "review.json", review)
    assert m._review("B", "synthetic-a", output, inputs)[0]["watcher_clean"] is True
    # Each case mutates only one underlying fact and updates fingerprints as needed;
    # a matching review digest cannot create missing dumps or captured provenance.
    if change == "missing":
        watcher.unlink()
    elif change in ("no_dump", "abnormal", "changed_after_capture"):
        new = {"no_dump": b"synthetic header only\n", "abnormal": raw + b"hung kernel\n",
               "changed_after_capture": raw + b"second synthetic line\n"}[change]
        watcher.write_bytes(new)
        review["watcher_sha256"] = m.digest(new)
        if change != "changed_after_capture":
            execution["artifacts"] = other_entries + [m.artifact(watcher, output)]
    elif change == "not_captured":
        execution["artifacts"] = other_entries
    else:
        review["watcher_clean"] = None
    # These are synthetic fixture rewrites, never modifications of real evidence.
    (output / "execution.json").write_text(json.dumps(execution))
    review["execution_sha256"] = m.digest((output / "execution.json").read_bytes())
    (inputs / "review.json").write_text(json.dumps(review))
    assert m._review("B", "synthetic-a", output, inputs)[0]["watcher_clean"] is False


def test_B_authority_does_not_inherit_A_or_a_second_invocation(tmp_path):
    from tests.test_issue30_capture import approval_inputs

    before, receipt, _ = approval_inputs(tmp_path)
    approval = m.approval_record("B", "synthetic-a", before, receipt, "synthetic-head")
    assert approval["schema"] == "issue30-owner-approval-v3"
    assert approval["owner_decision"]["logical_unit"] == "issue30-b-fabric1d-smoke-watcher-v1"
    m._approval("B", "synthetic-a", approval, before, receipt, "synthetic-head")
    approval["owner_decision"]["invocations"] = True
    with pytest.raises(ValueError, match="approval_binding"):
        m._approval("B", "synthetic-a", approval, before, receipt, "synthetic-head")


def test_raw_warning_header_cannot_hide_a_serious_alarm():
    assert transfer()["reason"] == "accepted"
    line = b"warning | Chip: 0 PCIe: 0 Unique ID: 0 critical eth channel 8 core 1-28 link DOWN/unconnected (QSFP)\n"
    assert transfer(stdout=line + smoke_stdout())["reason"] == "diagnostic"


def test_low_reported_firmware_is_refused_for_B_even_if_unknown_input_was_admitted():
    assert transfer()["reason"] == "accepted"
    line = b"debug | Established ETH FW version from first discovered ETH core: 1.3.0\n"
    assert transfer(stdout=line + smoke_stdout())["reason"] == "erisc_version"
