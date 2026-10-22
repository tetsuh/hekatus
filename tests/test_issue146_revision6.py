"""Focused Issue #146 revision-6 control-pair and replay checks."""

from __future__ import annotations

import pytest

from tests.test_issue146_analyzer import _plan, _record, _rehash, _trace_row
from tools import analyze_issue146 as analyzer
from tools.issue146_analysis import (
    CONTROL_STRICT_MAJORITY_OPERATOR,
    control_pair_evaluation,
)

REVISION5_PLAN_SHA256 = "5fc8d7a19533e10117e1b862603e5cb32ca37549b03194367b3396637dbff3ef"
REVISION5_HARNESS_COMMIT = "78b1b5480c785d492dd82eac99c07f088d734522"


def _control_row(word: int, aiclk: float, offset: int = 60) -> dict[str, object]:
    row = _trace_row(offset, configuration_id="C018")
    row["aiclk_arb_max_raw"] = hex(word)
    row["aiclk_mhz"] = aiclk
    return row


def _k1(rows: list[dict[str, object]]) -> dict[str, object]:
    spec = analyzer._plan_spec(_plan())
    decoded = {id(row): {"arb": analyzer.decode_arbiter_row(row)} for row in rows}
    return analyzer._control_k1(rows, decoded, spec)


@pytest.mark.parametrize(
    ("dip_count", "sample_count", "expected"),
    [
        (1, 4, True),
        (2, 4, False),
        (3, 4, False),
        (2, 5, True),
        (3, 5, False),
    ],
)
def test_issue146_revision6_k1_uses_strict_clamped_majority(
    dip_count: int, sample_count: int, expected: bool
) -> None:
    rows = [
        _control_row(
            0x70545 if index < dip_count else 0x80546,
            1343.0 if index < dip_count else 1350.0,
            60 + index,
        )
        for index in range(sample_count)
    ]
    result = _k1(rows)
    assert result["pass"] is expected
    assert result["clamped_count"] == sample_count - dip_count
    assert result["dip_count"] == dip_count


def test_issue146_revision6_k1_rejects_empty_and_mismatched_control_pairs() -> None:
    assert _k1([])["pass"] is False
    for word, aiclk in ((0x70544, 1343.0), (0x70545, 1337.0), (0x80546, 1349.0)):
        assert _k1([_control_row(word, aiclk)])["pass"] is False


def test_issue146_revision6_pair_evaluation_is_shared_and_exact() -> None:
    result = control_pair_evaluation(
        [_control_row(0x80546, 1350.0), _control_row(0x70545, 1343.0)]
    )
    assert result["clamped_count"] == 1
    assert result["dip_count"] == 1
    assert result["invalid_count"] == 0
    assert result["strict_majority_pass"] is False


def test_issue146_revision6_smoke_c018_r7_uses_the_same_strict_pair_rule() -> None:
    rows = [_control_row(0x80546, 1350.0, 10 + index) for index in range(4)]
    rows[0].update(aiclk_arb_max_raw="0x70545", aiclk_mhz=1343.0)
    criterion = analyzer._smoke_index_criterion(rows, "C018")
    assert criterion["pass"] is True
    rows[1].update(aiclk_arb_max_raw="0x70545", aiclk_mhz=1343.0)
    assert analyzer._smoke_index_criterion(rows, "C018")["pass"] is False
    rows[1].update(aiclk_arb_max_raw="0x70544", aiclk_mhz=1343.0)
    assert analyzer._smoke_index_criterion(rows, "C018")["pass"] is False


def test_issue146_revision6_known_dip_stays_nominal_for_glitch_detection() -> None:
    spec = analyzer._plan_spec(_plan())
    row = _control_row(0x70545, 1343.0)
    row["enabled_max_arb_raw"] = "0x0"
    decoded_rows, errors = analyzer._decoded_glitch_rows([row])
    assert errors == []
    reasons, whole, _r6 = analyzer._glitch_reasons(
        decoded_rows[0], mode="long-control", spec=spec
    )
    assert whole is True
    assert "enabled_max_arb_glitch" in reasons


def test_issue146_revision6_control_dips_have_truthful_window_intervals() -> None:
    rows = [
        _control_row(0x80546, 1350.0, 10),
        _control_row(0x70545, 1343.0, 20),
        _control_row(0x70545, 1343.0, 40),
    ]
    report = analyzer._control_dip_facts(
        rows,
        first_launch_start_monotonic_ns=0,
        window_start_seconds=10.0,
        window_end_seconds=40.0,
    )
    assert report["sample_count"] == 2
    assert report["dip_count"] == 1
    assert report["dip_fraction"] == 0.5
    assert len(report["dips"]) == 1
    assert report["intervals_between_captured_dips_seconds"] == []
    zero = analyzer._control_dip_facts(
        [_control_row(0x80546, 1350.0, 10)],
        first_launch_start_monotonic_ns=0,
        window_start_seconds=10.0,
        window_end_seconds=40.0,
    )
    assert zero["dip_count"] == 0
    assert zero["dip_fraction"] == 0.0
    assert zero["intervals_between_captured_dips_seconds"] == []


def _rev5_replay_record() -> dict[str, object]:
    record = _record("C018", 1)
    run = record["run"]
    assert isinstance(run, dict)
    run["mode"] = "smoke"
    run["observed_duration_seconds"] = 40.0
    record["plan_sha256"] = REVISION5_PLAN_SHA256
    environment = record["environment"]
    assert isinstance(environment, dict)
    environment["harness_commit"] = REVISION5_HARNESS_COMMIT
    environment["harness_dirty"] = False
    environment["harness_pushed"] = True
    record["provenance"] = {
        "plan_sha256": REVISION5_PLAN_SHA256,
        "unit_provenance": record["provenance"]["unit_provenance"],
    }
    record["runner_result"] = {
        "plan_sha256": REVISION5_PLAN_SHA256,
        "core_plan": record["configuration"]["core_plan"],
    }
    groups = record["groups"]
    assert isinstance(groups, dict)
    groups["groups"] = [{"start_monotonic_ns": 0, "end_monotonic_ns": 40_000_000_000}]
    trace = record["trace"]
    assert isinstance(trace, dict)
    trace["run_end_monotonic_ns"] = 40_000_000_000
    record["run"]["workload_end_monotonic_ns"] = 40_000_000_000
    _rehash(record)
    return record


def test_issue146_revision6_report_metadata_matches_control_policy() -> None:
    reports = []
    for configuration_id in ("C018", "C081", "C108"):
        record = _record(configuration_id, 1)
        run = record["run"]
        assert isinstance(run, dict)
        run["mode"] = "smoke"
        run["observed_duration_seconds"] = 40.0
        reports.append(record)
    result = analyzer.analyze_smoke(reports, plan=_plan())
    metadata = {
        report["run"]["configuration_id"]: report["run"]["evaluation"]
        for report in result["runs"]
    }
    assert metadata["C018"]["control_majority_operator"] == CONTROL_STRICT_MAJORITY_OPERATOR
    assert metadata["C018"]["control_pair_validity_requirement"] == "invalid_count == 0"
    for configuration_id in ("C081", "C108"):
        assert metadata[configuration_id]["control_majority_operator"] is None
        assert metadata[configuration_id]["control_pair_validity_requirement"] is None


def test_issue146_revision6_replays_only_known_revision5_smoke_lineage() -> None:
    result = analyzer.analyze_smoke([_rev5_replay_record()], plan=_plan())
    report = result["runs"][0]["run"]
    assert report["missing_fields"] == ["sample_count", "sample_count.R6", "sample_count.raw"]
    gates = [item for item in result["runs"][0]["acceptance"] if item["gate"] is True]
    assert len(gates) == 4
    assert all(item["pass"] is True for item in gates)
    assert report["acquisition"]["plan_revision"] == 5
    assert report["acquisition"]["plan_sha256"] == REVISION5_PLAN_SHA256
    assert report["evaluation"]["plan_revision"] == 6
    assert report["evaluation"]["plan_sha256"] == analyzer.plan_sha256(_plan())


def test_issue146_revision6_replay_rejects_unknown_harness_and_cadence() -> None:
    record = _rev5_replay_record()
    environment = record["environment"]
    assert isinstance(environment, dict)
    environment["harness_commit"] = "c" * 40
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert any(field == "plan_sha256" for field in result["runs"][0]["run"]["missing_fields"])

    record = _rev5_replay_record()
    cadence = record["trace_cadence"]
    assert isinstance(cadence, dict)
    cadence["host_sampler_interval_seconds"] = 2.0
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert any(
        "trace_cadence.host_sampler_interval_seconds" in field
        for field in result["runs"][0]["run"]["missing_fields"]
    )


def test_issue146_revision6_replay_rejects_long_and_unknown_plan_lineages() -> None:
    record = _rev5_replay_record()
    run = record["run"]
    assert isinstance(run, dict)
    run["mode"] = "long"
    result = analyzer.analyze_runs([record], plan=_plan())
    assert any(field == "plan_sha256" for field in result["runs"][0]["missing_fields"])

    record = _rev5_replay_record()
    record["plan_sha256"] = "a" * 64
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert any(field == "plan_sha256" for field in result["runs"][0]["run"]["missing_fields"])


def test_issue146_revision6_replay_preserves_artifact_hash_failures() -> None:
    record = _rev5_replay_record()
    trace = record["trace"]
    assert isinstance(trace, dict)
    trace["sha256"] = "0" * 64
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert any("trace_sha256" in field for field in result["runs"][0]["run"]["missing_fields"])


def test_issue146_revision6_replay_preserves_trace_clock_binding_failures() -> None:
    record = _rev5_replay_record()
    trace = record["trace"]
    assert isinstance(trace, dict)
    trace["run_end_monotonic_ns"] = 39_000_000_000
    _rehash(record)
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert "telemetry.trace.run_end_monotonic_ns.binding" in result["runs"][0]["run"]["missing_fields"]
