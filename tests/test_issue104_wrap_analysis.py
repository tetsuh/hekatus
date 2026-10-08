from __future__ import annotations

import copy
import hashlib
import itertools
import json
import struct
from pathlib import Path

import pytest

from tools.analyze_issue104_wrap import (
    CLOCK_MODULUS_TICKS,
    PAIR_TARGET_TICKS,
    _detect_pairs,
    analyze_file,
)


def _record_for(timestamps: list[int], starts: list[int]) -> dict:
    raw = b"".join(struct.pack("<Q", value) for value in timestamps)
    pair_records = []
    pair_sums = []
    for start in starts:
        first = timestamps[start] - timestamps[start - 1]
        second = timestamps[start + 1] - timestamps[start]
        pair_sum = first + second
        pair_sums.append(pair_sum)
        pair_records.append(
            {
                "first_interval_end_elapsed_seconds": (timestamps[start] - timestamps[0])
                / 1_350_000_000,
                "first_interval_end_elapsed_ticks": timestamps[start] - timestamps[0],
                "interval_end_frame_indices": [start, start + 1],
                "long_ticks": max(first, second),
                "pair_sum_ticks": pair_sum,
                "short_ticks": min(first, second),
                "sum_delta_ticks": pair_sum - PAIR_TARGET_TICKS,
            }
        )

    outlier_analysis = {
        "method": (
            "short < 1250000 ticks, long > 1450000 ticks, "
            "adjacent pair sum within 1000 ticks of 2700000"
        ),
        "status": "observation_only_no_causal_claim",
        "pair_count": len(starts),
        "pair_sum_target_ticks": PAIR_TARGET_TICKS,
        "pair_sums": pair_sums,
        "frame_gaps": [second - first for first, second in itertools.pairwise(starts)],
        "period_frames": 6363,
        "periodicity": (
            "No paired short/long outliers were observed; "
            "no phase distribution is applicable."
        ),
        "comparison_to_prior": {
            "prior_pair_count": 27,
            "prior_common_gap_frames": 6363,
            "prior_gap_distribution": "previous 27-pair record had 6363 as 11 of 26 gaps, gcd 1",
        },
        "aggregate_by_quotient_remainder": {},
    }
    if starts:
        outlier_analysis.update(
            {
                "pair_sum_min_ticks": min(pair_sums),
                "pair_sum_max_ticks": max(pair_sums),
                "pair_sum_mean_ticks": sum(pair_sums) / len(pair_sums),
                "sum_delta_min_ticks": min(pair - PAIR_TARGET_TICKS for pair in pair_sums),
                "sum_delta_max_ticks": max(pair - PAIR_TARGET_TICKS for pair in pair_sums),
                "frame_start_indices": starts,
                "event_frame_positions": [
                    frame_index for start in starts for frame_index in (start, start + 1)
                ],
                "event_elapsed_seconds": [
                    (timestamps[frame_index] - timestamps[0]) / 1_350_000_000
                    for start in starts
                    for frame_index in (start, start + 1)
                ],
                "event_elapsed_ticks": [
                    timestamps[frame_index] - timestamps[0]
                    for start in starts
                    for frame_index in (start, start + 1)
                ],
                "event_elapsed_ticks_mod_period": [
                    (timestamps[frame_index] - timestamps[0]) % CLOCK_MODULUS_TICKS
                    for start in starts
                    for frame_index in (start, start + 1)
                ],
                "pairs": pair_records,
            }
        )
    return {
        "raw_timestamps": {
            "count": len(timestamps),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "retained_outside_repository": True,
        },
        "outlier_analysis": outlier_analysis,
    }


def _analyze_record(tmp_path, timestamps: list[int], record: dict) -> dict:
    raw_path = tmp_path / "raw-timestamps.bin"
    raw_path.write_bytes(b"".join(struct.pack("<Q", value) for value in timestamps))
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record))
    return analyze_file(raw_path, record_path)


def _analyze_case(tmp_path, timestamps: list[int], starts: list[int]) -> dict:
    return _analyze_record(tmp_path, timestamps, _record_for(timestamps, starts))


def test_wrap_phase_and_record_endpoints_use_little_endian_absolute_values(tmp_path):
    period = CLOCK_MODULUS_TICKS
    timestamps = [
        period - 1_000,
        period + 100,
        period + 2_699_000,
        2 * period - 1_000,
        2 * period + 100,
        2 * period + 2_699_000,
        3 * period - 1_350_000,
        3 * period + 100,
        3 * period + 1_350_100,
    ]
    raw_path = tmp_path / "raw-timestamps.bin"
    raw_path.write_bytes(b"".join(struct.pack("<Q", value) for value in timestamps))
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(_record_for(timestamps, [1, 4])))

    result = analyze_file(raw_path, record_path)

    assert result["raw_timestamps"]["encoding"] == "unsigned 64-bit absolute timestamp"
    assert result["raw_timestamps"]["endianness"] == "little"
    assert result["raw_timestamps"]["packing"] == "<Q"
    assert result["pair_order_counts"] == {
        "short-first": 2,
        "long-first": 0,
        "other": 0,
    }
    assert result["record_correspondence"]["exact_frame_start_match"] is True
    assert result["record_correspondence"]["pair_endpoint_mismatches"] == []
    assert result["phase_analysis"]["raw_event_phase_ticks_mod_period"] == [100, 100]
    assert result["phase_analysis"]["first_interval_end_elapsed_ticks_mod_period"] == [1_100, 1_100]
    assert result["phase_analysis"]["rayleigh_R"] == 1.0
    assert result["phase_analysis"]["circular_arc_width_ticks"] == 0
    assert result["phase_analysis"]["all_event_phases_in_window"] is True
    assert result["phase_window"]["wrap_candidate_count"] == 3
    assert result["phase_window"]["non_event_wrap_candidate_count"] == 1
    assert result["phase_window"]["wrap_candidate_order_counts"] == {
        "short-first": 2,
        "long-first": 0,
        "other": 1,
    }
    assert result["phase_window"]["pair_to_wrap_candidate_ratio"] == {
        "numerator_event_pair_starts_in_window": 2,
        "denominator_wrap_candidates_in_window": 3,
        "fraction": 2 / 3,
    }


def test_pair_order_detects_long_first_without_calling_it_short_first():
    pairs = _detect_pairs([0, 1_600_000, 2_700_000])

    assert len(pairs) == 1
    assert pairs[0]["order"] == "long-first"
    assert pairs[0]["first_interval_ticks"] == 1_600_000
    assert pairs[0]["second_interval_ticks"] == 1_100_000


@pytest.mark.parametrize(
    "record_name",
    [
        "2026-10-06-p150a-issue104-sampler-default.json",
        "2026-10-06-p150a-issue104-sampler-5s.json",
    ],
    ids=["historical-default", "historical-5s"],
)
def test_historical_sampled_records_use_only_record_clock_as_unverified_aiclk(
    tmp_path: Path, record_name: str
):
    timestamps = [0, 1_000_000, 2_700_000]
    raw = b"".join(struct.pack("<Q", value) for value in timestamps)
    record = copy.deepcopy(
        json.loads((Path("docs/measurements") / record_name).read_text())
    )
    record["environment"]["aiclk_mhz_observed"] = [800]
    record["raw_timestamps"].update(
        count=len(timestamps), sha256=hashlib.sha256(raw).hexdigest()
    )
    record["histogram"]["N"] = len(timestamps) - 1
    record["outlier_analysis"] = _record_for(timestamps, [1])["outlier_analysis"]

    raw_path = tmp_path / "raw-timestamps.bin"
    raw_path.write_bytes(raw)
    record_path = tmp_path / record_name
    record_path.write_text(json.dumps(record))

    result = analyze_file(raw_path, record_path)

    assert result["aiclk_mhz"] == 1_350
    assert result["aiclk_source"] == "legacy_unverified"
    assert result["outlier_analysis"]["aiclk_source"] == "legacy_unverified"
    assert result["invariant_validation"]["out_of_scope"] is True
    assert result["invariant_validation"]["timing_evidence"] is False
    assert record["timing_evidence"] is True


def test_zero_pair_analysis_returns_empty_counts_and_null_phase_statistics(tmp_path):
    timestamps = [0, 1_350_000, 2_700_000, 4_050_000]

    result = _analyze_case(tmp_path, timestamps, [])

    assert result["pair_count"] == 0
    assert result["pairs"] == []
    assert result["pair_order_counts"] == {
        "short-first": 0,
        "long-first": 0,
        "other": 0,
    }
    assert result["pair_sum_min_ticks"] is None
    assert result["pair_sum_max_ticks"] is None
    assert result["pair_sum_delta_min_ticks"] is None
    assert result["pair_sum_delta_max_ticks"] is None
    assert result["record_correspondence"]["record_pair_count"] == 0
    assert result["record_correspondence"]["raw_derived_pair_count"] == 0
    assert result["record_correspondence"]["exact_frame_start_match"] is True

    phase_window = result["phase_window"]
    assert phase_window["wrap_candidate_count"] == 0
    assert phase_window["wrap_candidate_order_counts"] == {
        "short-first": 0,
        "long-first": 0,
        "other": 0,
    }
    assert phase_window["pair_to_wrap_candidate_ratio"] == {
        "numerator_event_pair_starts_in_window": 0,
        "denominator_wrap_candidates_in_window": 0,
        "fraction": None,
    }

    phase = result["phase_analysis"]
    assert phase["event_count"] == 0
    assert phase["event_frame_indices"] == []
    assert phase["raw_event_phase_ticks_mod_period"] == []
    assert phase["first_interval_end_elapsed_ticks"] == []
    assert phase["first_interval_end_elapsed_ticks_mod_period"] == []
    assert phase["raw_phase_min_ticks"] is None
    assert phase["raw_phase_max_ticks"] is None
    assert phase["elapsed_phase_min_ticks"] is None
    assert phase["elapsed_phase_max_ticks"] is None
    assert phase["rayleigh_R"] is None
    assert phase["circular_mean_raw_phase_ticks"] is None
    assert phase["circular_arc_width_ticks"] is None
    assert phase["circular_arc_width_ms"] is None
    assert phase["all_event_phases_in_window"] is None


def test_one_pair_analysis_preserves_phase_and_zero_circular_gap(tmp_path):
    period = CLOCK_MODULUS_TICKS
    timestamps = [period - 1_000, period + 100, period + 2_699_000]

    result = _analyze_case(tmp_path, timestamps, [1])

    assert result["pair_count"] == 1
    assert result["pairs"][0]["pair_sum_ticks"] == PAIR_TARGET_TICKS
    assert result["pair_order_counts"] == {
        "short-first": 1,
        "long-first": 0,
        "other": 0,
    }
    assert result["record_correspondence"]["exact_frame_start_match"] is True
    assert result["record_correspondence"]["pair_endpoint_mismatches"] == []

    phase = result["phase_analysis"]
    assert phase["event_count"] == 1
    assert phase["raw_event_phase_ticks_mod_period"] == [100]
    assert phase["first_interval_end_elapsed_ticks_mod_period"] == [1_100]
    assert phase["raw_phase_min_ticks"] == 100
    assert phase["raw_phase_max_ticks"] == 100
    assert phase["elapsed_phase_min_ticks"] == 1_100
    assert phase["elapsed_phase_max_ticks"] == 1_100
    assert phase["rayleigh_R"] == 1.0
    assert phase["circular_mean_raw_phase_ticks"] == pytest.approx(100)
    assert phase["circular_arc_width_ticks"] == 0
    assert phase["circular_arc_width_ms"] == 0
    assert phase["all_event_phases_in_window"] is True

    phase_window = result["phase_window"]
    assert phase_window["wrap_candidate_frame_indices"] == [1]
    assert phase_window["event_pair_starts_in_window"] == 1
    assert phase_window["event_pair_starts_outside_window"] == []
    assert phase_window["pair_to_wrap_candidate_ratio"]["fraction"] == 1.0


def test_two_pair_analysis_preserves_circular_gap_and_phase_extrema(tmp_path):
    period = CLOCK_MODULUS_TICKS
    timestamps = [
        period - 1_000,
        period + 100,
        period + 2_699_000,
        2 * period - 1_200,
        2 * period - 100,
        2 * period + 2_698_800,
    ]

    result = _analyze_case(tmp_path, timestamps, [1, 4])

    assert result["pair_count"] == 2
    assert [pair["interval_end_frame_indices"] for pair in result["pairs"]] == [
        [1, 2],
        [4, 5],
    ]
    assert result["pair_order_counts"] == {
        "short-first": 2,
        "long-first": 0,
        "other": 0,
    }
    assert result["record_correspondence"]["exact_frame_start_match"] is True
    assert result["record_correspondence"]["pair_endpoint_mismatches"] == []

    phase = result["phase_analysis"]
    assert phase["event_count"] == 2
    assert phase["raw_event_phase_ticks_mod_period"] == [100, period - 100]
    assert phase["first_interval_end_elapsed_ticks_mod_period"] == [1_100, 900]
    assert phase["raw_phase_min_ticks"] == 100
    assert phase["raw_phase_max_ticks"] == period - 100
    assert phase["elapsed_phase_min_ticks"] == 900
    assert phase["elapsed_phase_max_ticks"] == 1_100
    assert phase["rayleigh_R"] == pytest.approx(1.0)
    circular_mean = phase["circular_mean_raw_phase_ticks"]
    assert min(circular_mean, period - circular_mean) == pytest.approx(0, abs=1e-6)
    assert phase["circular_arc_width_ticks"] == 200
    assert phase["circular_arc_width_ms"] == pytest.approx(200 / 1_350_000)
    assert phase["all_event_phases_in_window"] is False

    phase_window = result["phase_window"]
    assert phase_window["wrap_candidate_frame_indices"] == [1]
    assert phase_window["event_pair_starts_in_window"] == 1
    assert phase_window["event_pair_starts_outside_window"] == [4]
    assert phase_window["non_event_wrap_candidate_count"] == 0
    assert phase_window["pair_to_wrap_candidate_ratio"]["fraction"] == 1.0


def test_duplicate_record_frame_starts_are_a_correspondence_mismatch(tmp_path):
    period = CLOCK_MODULUS_TICKS
    timestamps = [period - 1_000, period + 100, period + 2_699_000]

    result = _analyze_case(tmp_path, timestamps, [1, 1])

    correspondence = result["record_correspondence"]
    assert correspondence["exact_frame_start_match"] is False
    assert correspondence["frame_start_length_mismatch"] is True
    assert correspondence["record_duplicate_frame_start_indices"] == [1]
    assert correspondence["frame_start_multiset_match"] is False
    assert correspondence["missing_raw_pair_starts"] == [1]
    assert correspondence["record_pair_list_length_mismatch"] is True
    assert correspondence["pair_endpoint_list_length_mismatch"] is True


def test_record_pair_count_mismatch_reports_tail_and_common_value_order_errors(tmp_path):
    period = CLOCK_MODULUS_TICKS
    timestamps = [
        period - 1_000,
        period + 100,
        period + 2_699_000,
        2 * period - 1_200,
        2 * period - 100,
        2 * period + 2_698_800,
    ]
    record = _record_for(timestamps, [1])
    record["outlier_analysis"]["pairs"][0]["short_ticks"] += 1
    record["outlier_analysis"]["pairs"][0]["order"] = "long-first"

    result = _analyze_record(tmp_path, timestamps, record)

    correspondence = result["record_correspondence"]
    assert correspondence["record_pair_count"] == 1
    assert correspondence["raw_derived_pair_count"] == 2
    assert correspondence["frame_start_length_mismatch"] is True
    assert correspondence["frame_start_multiset_match"] is False
    assert correspondence["unexpected_raw_pair_starts"] == [4]
    assert correspondence["pair_endpoint_list_length_mismatch"] is True
    assert correspondence["pair_endpoint_missing_record_indices"] == [1]
    assert correspondence["pair_value_missing_record_indices"] == [1]
    assert correspondence["pair_value_mismatches"] == [0]
    assert correspondence["pair_order_mismatches"] == [0]


def test_pair_endpoint_order_mismatch_is_reported_without_zip_truncation(tmp_path):
    period = CLOCK_MODULUS_TICKS
    timestamps = [
        period - 1_000,
        period + 100,
        period + 2_699_000,
        2 * period - 1_200,
        2 * period - 100,
        2 * period + 2_698_800,
    ]
    record = _record_for(timestamps, [1, 4])
    record["outlier_analysis"]["pairs"][0]["interval_end_frame_indices"] = [2, 1]

    result = _analyze_record(tmp_path, timestamps, record)

    correspondence = result["record_correspondence"]
    assert correspondence["pair_endpoint_list_length_match"] is True
    assert correspondence["pair_endpoint_order_mismatches"] == [0]
    assert correspondence["pair_endpoint_mismatches"] == [0]
    assert correspondence["pair_value_mismatches"] == []
