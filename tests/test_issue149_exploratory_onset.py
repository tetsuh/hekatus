"""Board-free full-workload exploratory clock-onset regressions."""

from __future__ import annotations

import pytest

from tests.test_issue146_analyzer import (
    _plan,
    _record,
    _rehash,
    _trace_row,
)
from tools import analyze_issue146 as analyzer


@pytest.mark.parametrize(("dip_seconds", "expected"), [((10, 60), 10.0), ((60,), 60.0), ((), None)])
def test_clock_onset_uses_full_loaded_trace(dip_seconds, expected) -> None:
    record = _record("C018", 1)
    for row in record["trace"]["samples"]:
        if row["timestamp_monotonic_ns"] // 1_000_000_000 in dip_seconds:
            row.update(aiclk_arb_max_raw="0x70545", aiclk_mhz=1343.0)
    _rehash(record)
    report = analyzer.analyze_runs([record], plan=_plan())["runs"][0]
    assert report["status"] == "ok"
    onset = report["exploratory"]["onset"]
    assert onset["first_aiclk_arb_max_below_fmax_seconds"] == expected
    assert onset["first_board_power_within_5_w_of_limit_seconds"] is None


@pytest.mark.parametrize("interval_seconds", [2.0, 1.0])
def test_clock_cadence_distinguishes_observed_smoke_interval_from_plan(interval_seconds) -> None:
    record = _record("C018", 1)
    record["run"].update(
        mode="smoke", observed_duration_seconds=40.0, workload_end_monotonic_ns=40_000_000_000
    )
    record["groups"]["groups"] = [{"start_monotonic_ns": 0, "end_monotonic_ns": 40_000_000_000}]
    record["trace"]["run_end_monotonic_ns"] = 40_000_000_000
    record["trace"]["samples"] = [
        _trace_row(offset, configuration_id="C018")
        for offset in range(0, 41, int(interval_seconds))
    ]
    record["trace"]["sampling_interval_seconds"] = interval_seconds
    record["trace_cadence"].update(
        host_sampler_interval_seconds=interval_seconds,
        test_override=interval_seconds != 1.0,
    )
    _rehash(record)
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert all(gate["pass"] is True for gate in result["gates"])
    report = result["runs"][0]["run"]
    assert report["status"] == "ok"
    selection = report["exploratory"]["onset"]["clock_selection"]
    assert selection["observed_capture_interval_seconds"] == interval_seconds
    assert selection["planned_capture_interval_seconds"] == 1.0
    assert "sample_capture_interval_seconds" not in selection


def test_clock_cadence_uses_loaded_timestamp_median_before_telemetry_exclusions() -> None:
    rows = [
        _trace_row(-100, configuration_id="C018"),
        _trace_row(0, configuration_id="C018"),
        _trace_row(2, configuration_id="C018", aiclk_arb_max_raw="malformed"),
        _trace_row(6, configuration_id="C018", board_power_w=0.0),
        _trace_row(14, configuration_id="C018"),
        _trace_row(190, configuration_id="C018"),
        _trace_row(1000, configuration_id="C018"),
    ]
    selection = _exploratory_onset(rows)["clock_selection"]
    assert selection["observed_capture_interval_seconds"] == 4.0
    assert selection["planned_capture_interval_seconds"] == 1.0


@pytest.mark.parametrize("offsets", [[], [-1, 190], [10]])
def test_clock_cadence_is_unavailable_with_fewer_than_two_loaded_rows(offsets) -> None:
    selection = _exploratory_onset(
        [_trace_row(offset, configuration_id="C018") for offset in offsets]
    )["clock_selection"]
    assert selection["observed_capture_interval_seconds"] is None
    assert selection["observed_capture_interval_reason"] == "fewer than two loaded rows"
    assert selection["planned_capture_interval_seconds"] == 1.0


def _exploratory_onset(rows, *, last_group_end=190_000_000_000, configuration_id="C018"):
    # Keep the registered plateau reference separate from the raw onset candidates.
    plateau = [_trace_row(60, configuration_id=configuration_id)]
    decoded = analyzer._decode_plateau_values(plateau, [])
    return analyzer._exploratory(
        rows,
        plateau,
        decoded,
        first=0,
        mode="long",
        configuration_id=configuration_id,
        post_device=None,
        last_group_end=last_group_end,
        spec=analyzer._plan_spec(_plan()),
        duration_weighted={},
        group_timings=[],
        post_capture_timestamp=None,
    )["onset"]


@pytest.mark.parametrize("outside_seconds", [-1, 190, 191])
def test_clock_ignores_preload_and_postload_but_power_is_unchanged(outside_seconds) -> None:
    rows = sorted(
        [
            _trace_row(
                outside_seconds,
                configuration_id="C018",
                aiclk_arb_max_raw="0x70545",
                board_power_w=300.0,
            ),
            _trace_row(60, configuration_id="C018"),
        ],
        key=lambda row: row["timestamp_monotonic_ns"],
    )
    onset = _exploratory_onset(rows)
    assert onset["first_aiclk_arb_max_below_fmax_seconds"] is None
    assert onset["first_board_power_within_5_w_of_limit_seconds"] == outside_seconds
    skipped = onset["clock_selection"]["skipped_rows"]
    assert len(skipped) == 1
    assert skipped[0]["reasons"] == ["pre_load" if outside_seconds < 0 else "post_load"]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"aiclk_arb_max_raw": "malformed"}, "telemetry_decode_error"),
        ({field: "0x0" for field in analyzer.ARBITER_FIELDS}, "all_arbiter_words_zero"),
        ({"board_power_w": 0.0}, "board_power_zero"),
        ({"enabled_max_arb_raw": "0x0"}, "enabled_max_arb_glitch"),
    ],
)
def test_clock_skips_malformed_and_whole_sample_glitches(changes, reason) -> None:
    rows = [
        _trace_row(10, configuration_id="C081", **changes),
        _trace_row(60, configuration_id="C081"),
    ]
    onset = _exploratory_onset(rows, configuration_id="C081")
    assert onset["first_aiclk_arb_max_below_fmax_seconds"] == 60.0
    skipped = onset["clock_selection"]["skipped_rows"]
    assert skipped[0]["row_index"] == 0
    assert reason in skipped[0]["reasons"]
    if reason == "telemetry_decode_error":
        assert skipped[0]["error"]


def test_clock_keeps_r6_only_glitch_and_first_launch_boundary() -> None:
    rows = [
        _trace_row(0, configuration_id="C081", gddr_0_1_temp_c="0x0"),
        _trace_row(60, configuration_id="C081"),
    ]
    onset = _exploratory_onset(rows, configuration_id="C081")
    assert onset["first_aiclk_arb_max_below_fmax_seconds"] == 0.0
    assert onset["clock_selection"]["skipped_rows"] == []


def test_clock_uses_overrun_until_final_group_end() -> None:
    rows = [_trace_row(189, configuration_id="C081")]
    onset = _exploratory_onset(rows, configuration_id="C081")
    assert onset["first_aiclk_arb_max_below_fmax_seconds"] == 189.0


@pytest.mark.parametrize("word", ["0x0", "0x90545"])
def test_clock_does_not_add_index_or_realised_clock_filters(word) -> None:
    # A lone zero word is decodable, unlike the registered all-four-zero glitch.
    onset = _exploratory_onset(
        [
            _trace_row(10, configuration_id="C018", aiclk_arb_max_raw=word),
        ]
    )
    assert onset["first_aiclk_arb_max_below_fmax_seconds"] == 10.0
    assert onset["clock_selection"]["skipped_rows"] == []


def test_clock_nominal_glitch_check_uses_registered_plateau_power_reference() -> None:
    onset = _exploratory_onset(
        [
            _trace_row(10, configuration_id="C081", enabled_max_arb_raw="0x0", board_power_w=400.0),
            _trace_row(60, configuration_id="C081"),
        ],
        configuration_id="C081",
    )
    # Not nominal against the all-plateau reference: do not silently exclude it.
    assert onset["first_aiclk_arb_max_below_fmax_seconds"] == 10.0
    assert onset["clock_selection"]["skipped_rows"] == []


def test_clock_missing_load_end_is_explicit_and_does_not_change_power() -> None:
    onset = _exploratory_onset(
        [_trace_row(10, configuration_id="C081")],
        last_group_end=None,
        configuration_id="C081",
    )
    assert onset["first_aiclk_arb_max_below_fmax_seconds"] is None
    assert onset["first_board_power_within_5_w_of_limit_seconds"] == 10.0
    assert onset["clock_selection"]["status"] == "not_evaluable"
    assert onset["clock_selection"]["reason"] == "final group end is unavailable"
    assert onset["clock_selection"]["observed_capture_interval_seconds"] is None
    assert (
        onset["clock_selection"]["observed_capture_interval_reason"]
        == "final group end is unavailable"
    )
