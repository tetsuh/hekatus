"""Independent board-free verification for the final Issue #88 companion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import newton_schulz_issue88 as runner

ROOT = Path(__file__).parents[1]
MEASUREMENTS = ROOT / "docs/measurements"
PRIMARY = MEASUREMENTS / (
    "2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json"
)
SUMMARY = MEASUREMENTS / (
    "2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json"
)
PRIMARY_SHA256 = "04743a45c41405b8123db0f9ce6022557b83e22e7c8e2ffeb345cf84a9cb08f6"
SUMMARY_SHA256 = "26583bedd319f894526b6c9e3ec6986a74f2523672073cea395b026cd9adfb97"
SUPERSEDED_HASHES = {
    "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json": (
        "50ceddd6bff5afff972d393451fec86931bbb1ba1c9916f481c12ab2eee44567"
    ),
    "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-summary.json": (
        "27186c518b7382a0d05edc15d87876c1300be18af2de3edd498e4261b810d851"
    ),
    "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-l1-accounting.json": (
        "0f22bc6f0592410c4dceaf4b7159e82241083bcce86e2693a63c40c8de59d28e"
    ),
    "2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv": (
        "1086480d4cec82e0efaf309221e0a254f9390c5f5e85c9a192dac2c43a484df1"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _beam_metrics(source_row: dict) -> dict:
    beam = source_row["correctness"]["quality_vs_true_inverse"]["beam_pattern"]
    db = beam["db_pattern"]
    return {
        "metric_reference": source_row["correctness"]["quality_vs_true_inverse"][
            "metric_reference"
        ],
        "phase_sensitive_complex_response_relative_frobenius_error": beam[
            "phase_sensitive_complex_response_relative_frobenius_error"
        ],
        "best_complex_scalar_phase_aligned_complex_response_relative_frobenius_error": beam[
            "phase_aligned_complex_response_relative_frobenius_error"
        ],
        "magnitude_response_relative_frobenius_error": beam[
            "magnitude_response_relative_frobenius_error"
        ],
        "normalized_db_floored_pattern": {
            "rms_absolute_error_db": db["rms_absolute_error_db"],
            "max_absolute_error_db": db["max_absolute_error_db"],
            "floor_db": db["floor_db"],
            "amplitude_floor": db["amplitude_floor"],
            "definition": db["definition"],
        },
    }


def _row_view(source_row: dict, measurement: dict) -> dict:
    preflight = source_row["preflight"]
    correctness = source_row.get("correctness", {})
    quality = correctness.get("quality_vs_true_inverse")
    view = {
        "row": source_row["row"],
        "variant": "BF16-R" if source_row["variant"] == "bf16" else "FP32-R",
        "L": source_row["size"],
        "R_placement": source_row["r_memory"].upper(),
        "status": source_row["status"],
        "selected_controls": {
            key: source_row[key]
            for key in (
                "batch",
                "iterations",
                "initial_value",
                "state_format",
                "destination_format",
                "fp32_dest_acc_en",
                "math_fidelity",
                "fuse_s",
                "matrix_block",
                "double_buffer",
                "dst_full_sync_en",
                "x0_memory",
                "output_memory",
            )
        },
        "allocation_attempted": preflight.get(
            "allocation_attempted", source_row["status"] != "preflight_rejected"
        ),
        "launched_work": source_row.get("launches_measured", 0),
        "inverse": None,
        "mv_weight_direction": None,
        "beam_metrics": None,
        "timing": {
            "launches_measured": source_row.get("launches_measured", 0),
            "row_timeout_s": source_row.get(
                "row_timeout_s", measurement["row_timeout_s"]
            ),
            "seconds_per_launch_p50": source_row.get("seconds_per_launch_p50"),
            "tflops_p50_derived": source_row.get("tflops_p50_derived"),
        },
        "l1": {
            "status": preflight["status"],
            "accepted": preflight["accepted"],
            "total_bytes": preflight["total_bytes"],
            "budget_bytes": preflight["budget_bytes"],
            "overage_bytes": preflight["overage_bytes"],
            "headroom_bytes": preflight["headroom_bytes"],
            "cb_bytes": preflight["cb_bytes"],
            "tensor_bytes": preflight["tensor_bytes"],
            "static_prefix_bytes": preflight["static_prefix_bytes"],
            "placement": preflight["placement"],
        },
    }
    if quality is not None:
        view["inverse"] = {
            "metric_reference": correctness["metric_references"][
                "relative_frobenius_error_vs_true_inverse"
            ],
            "relative_frobenius_error": correctness[
                "relative_frobenius_error_vs_true_inverse"
            ],
        }
        view["mv_weight_direction"] = {
            "metric_reference": quality["metric_reference"],
            "definition": quality["mv_weight_direction"]["definition"],
            "max_cosine_deficit": quality["mv_weight_direction"][
                "max_cosine_deficit"
            ],
        }
        view["beam_metrics"] = _beam_metrics(source_row)
    return view


def _differences(left: dict, right: dict) -> dict:
    left_beam = left["beam_metrics"]
    right_beam = right["beam_metrics"]
    left_db = left_beam["normalized_db_floored_pattern"]
    right_db = right_beam["normalized_db_floored_pattern"]
    return {
        "inverse_error": {
            "left_minus_right": left["inverse"]["relative_frobenius_error"]
            - right["inverse"]["relative_frobenius_error"]
        },
        "mv_direction_max_cosine_deficit": {
            "left_minus_right": left["mv_weight_direction"]["max_cosine_deficit"]
            - right["mv_weight_direction"]["max_cosine_deficit"]
        },
        "beam_metrics": {
            "phase_sensitive_error": {
                "left_minus_right": left_beam[
                    "phase_sensitive_complex_response_relative_frobenius_error"
                ]
                - right_beam[
                    "phase_sensitive_complex_response_relative_frobenius_error"
                ]
            },
            "phase_aligned_error": {
                "left_minus_right": left_beam[
                    "best_complex_scalar_phase_aligned_complex_response_relative_frobenius_error"
                ]
                - right_beam[
                    "best_complex_scalar_phase_aligned_complex_response_relative_frobenius_error"
                ]
            },
            "magnitude_error": {
                "left_minus_right": left_beam[
                    "magnitude_response_relative_frobenius_error"
                ]
                - right_beam["magnitude_response_relative_frobenius_error"]
            },
            "normalized_db_floored_pattern": {
                "db_rms_error": {
                    "left_minus_right": left_db["rms_absolute_error_db"]
                    - right_db["rms_absolute_error_db"]
                },
                "db_max_error": {
                    "left_minus_right": left_db["max_absolute_error_db"]
                    - right_db["max_absolute_error_db"]
                },
            },
        },
        "performance": {
            "p50_tflops": {
                "right_minus_left": right["timing"]["tflops_p50_derived"]
                - left["timing"]["tflops_p50_derived"]
            },
            "p50_latency_seconds": {
                "right_minus_left": right["timing"]["seconds_per_launch_p50"]
                - left["timing"]["seconds_per_launch_p50"]
            },
        },
        "l1": {
            "l1_total_bytes_delta": {
                "right_minus_left": right["l1"]["total_bytes"]
                - left["l1"]["total_bytes"]
            },
            "l1_headroom_bytes_delta": {
                "right_minus_left": right["l1"]["headroom_bytes"]
                - left["l1"]["headroom_bytes"]
            },
            "l1_cb_bytes_delta": {
                "right_minus_left": right["l1"]["cb_bytes"]
                - left["l1"]["cb_bytes"]
            },
            "l1_tensor_bytes_delta": {
                "right_minus_left": right["l1"]["tensor_bytes"]
                - left["l1"]["tensor_bytes"]
            },
        },
    }


def test_final_companion_recomputes_every_value_from_primary_record():
    primary_bytes = PRIMARY.read_bytes()
    primary = json.loads(primary_bytes)
    summary = json.loads(
        SUMMARY.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {token}")
        )
    )

    assert _sha256(PRIMARY) == PRIMARY_SHA256
    assert _sha256(SUMMARY) == SUMMARY_SHA256
    assert json.loads(runner._serialize_record(summary)) == summary
    assert summary["source_record"] == {
        "file": PRIMARY.name,
        "sha256": hashlib.sha256(primary_bytes).hexdigest(),
    }
    assert summary["provenance"]["inherits_primary_record"] is True
    assert summary["provenance"]["primary_record"] == summary["source_record"]
    assert summary["status"] == primary["status"] == "pass"
    assert summary["authoritative"] is True

    source_rows = primary["measurement"]["comparison_rows"]
    expected_rows = {
        row["row"]: _row_view(row, primary["measurement"])
        for row in source_rows
    }
    assert summary["rows"] == [
        expected_rows[row["row"]]
        for row in source_rows
        if row["status"] == "ok"
    ]
    assert summary["l1_preflight_accounting"]["source_record"] == summary[
        "source_record"
    ]
    assert summary["l1_preflight_accounting"]["rows"] == [
        {"row": row_name, **expected["l1"]}
        for row_name, expected in expected_rows.items()
    ]

    expected_superceded = list(SUPERSEDED_HASHES)
    assert summary["supersedes"]["records"] == expected_superceded
    assert summary["supersedes"]["immutable"] is True
    assert summary["supersedes"]["reason"] == (
        "The final runner primary record embeds the complete L1 ledger and final evidence."
    )
    assert summary["supersedes"]["artifact_sha256"] == [
        {"file": name, "sha256": SUPERSEDED_HASHES[name]}
        for name in expected_superceded
    ]
    assert {
        name: _sha256(MEASUREMENTS / name) for name in expected_superceded
    } == SUPERSEDED_HASHES

    comparison_specs = [
        ("bf16-r-L16", "fp32-r-L16", True, "variant"),
        ("bf16-r-L32-r-dram", "fp32-r-L32-r-dram", True, "variant"),
        ("bf16-r-L32", "bf16-r-L32-r-dram", True, "placement"),
        ("bf16-r-L32", "fp32-r-L32-r-dram", False, "context"),
    ]
    assert [item["rows"] for item in summary["comparisons"]] == [
        [left, right] for left, right, _matched, _kind in comparison_specs
    ]
    for item, (left_name, right_name, matched, kind) in zip(
        summary["comparisons"], comparison_specs, strict=True
    ):
        left = expected_rows[left_name]
        right = expected_rows[right_name]
        assert item["matched_placement"] is matched
        assert item["kind"] == kind
        assert item["values"] == {"left": left, "right": right}
        assert item["differences"] == _differences(left, right)

    matched = summary["matched_placement_differences"]
    assert [item["rows"] for item in matched] == [
        ["bf16-r-L16", "fp32-r-L16"],
        ["bf16-r-L32-r-dram", "fp32-r-L32-r-dram"],
    ]
    for item in matched:
        left = expected_rows[item["rows"][0]]
        right = expected_rows[item["rows"][1]]
        differences = _differences(left, right)
        quality = item["quality_error_reduction"]
        expected_quality = {
            "inverse_error_reduction": differences["inverse_error"]["left_minus_right"],
            "mv_direction_deficit_reduction": differences[
                "mv_direction_max_cosine_deficit"
            ]["left_minus_right"],
            "phase_sensitive_reduction": differences["beam_metrics"][
                "phase_sensitive_error"
            ]["left_minus_right"],
            "phase_aligned_reduction": differences["beam_metrics"][
                "phase_aligned_error"
            ]["left_minus_right"],
            "magnitude_reduction": differences["beam_metrics"][
                "magnitude_error"
            ]["left_minus_right"],
            "normalized_db_floored_pattern_rms_db": differences["beam_metrics"][
                "normalized_db_floored_pattern"
            ]["db_rms_error"]["left_minus_right"],
            "normalized_db_floored_pattern_max_db": differences["beam_metrics"][
                "normalized_db_floored_pattern"
            ]["db_max_error"]["left_minus_right"],
        }
        assert quality == expected_quality
        assert item["all_quality_indicators_improve"] is True
        assert item["costs"]["l1_total_increase_bytes"] == differences["l1"][
            "l1_total_bytes_delta"
        ]["right_minus_left"]
        assert item["costs"]["p50_tflops_delta"] == differences[
            "performance"
        ]["p50_tflops"]["right_minus_left"]
        assert item["costs"]["p50_tflops_decrease"] == -item["costs"][
            "p50_tflops_delta"
        ]

    outcome = summary["fp32_r_outcome_under_common_true_inverse_reference"]
    assert outcome["inverse_error_improves"] is True
    assert outcome["mv_direction_deficit_improves"] is True
    assert outcome["all_four_beam_indicators_improve"] is True
    assert summary["power_snapshot"]["file"] == primary["measurement"][
        "power_clock_provenance"
    ]["trace"]
    assert summary["power_snapshot"]["sha256"] == primary["measurement"][
        "power_clock_provenance"
    ]["trace_sha256"]
