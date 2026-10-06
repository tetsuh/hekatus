from __future__ import annotations

import hashlib
import struct

from tools.analyze_issue104_wrap import (
    CLOCK_MODULUS_TICKS,
    _detect_pairs,
    analyze_file,
)


def _record_for(timestamps: list[int], starts: list[int]) -> dict:
    raw = b"".join(struct.pack("<Q", value) for value in timestamps)
    pairs = []
    for start in starts:
        first = timestamps[start] - timestamps[start - 1]
        second = timestamps[start + 1] - timestamps[start]
        pairs.append(
            {
                "interval_end_frame_indices": [start, start + 1],
                "short_ticks": min(first, second),
                "long_ticks": max(first, second),
            }
        )
    return {
        "raw_timestamps": {"count": len(timestamps), "sha256": hashlib.sha256(raw).hexdigest()},
        "outlier_analysis": {"frame_start_indices": starts, "pairs": pairs},
    }


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
    import json

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
