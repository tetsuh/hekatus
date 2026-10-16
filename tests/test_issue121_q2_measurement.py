"""Board-free tests for the Issue #121 Q2 plan, harness, and analyzer."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import analyze_issue121_q2 as analysis
from tools import issue121_q2 as runner
from tools import issue121_q2_wrapper_probe as wrapper_probe

ROOT = Path(__file__).parents[1]
PLAN = ROOT / runner.ISSUE121_Q2_PLAN
SUMMARY = ROOT / "docs/measurements/2026-10-16-host-newton-schulz-issue121-q2-summary.json"
GDDR_FAN_SUPPLEMENT = ROOT / (
    "docs/measurements/issue121-q2/2026-10-16-post-hoc-gddr-fan-observation.json"
)
IMMUTABLE_RUN_ROOT = ROOT / "docs/measurements/issue121-q2/runs"
Q2_RUN_ROOT = IMMUTABLE_RUN_ROOT / "run-1-device0"
Q2_RUN_RECORD = Q2_RUN_ROOT / "run-q2-run-1-device0.json"
Q2_TRACE = Q2_RUN_ROOT / "extended-trace-q2-run-1-device0.csv"
Q2_COMMITTED_ANALYSIS = Q2_RUN_ROOT / "analysis-q2-run-1-device0.json"


def _snapshot() -> str:
    return json.dumps(
        {
            "host_info": {"hostname": "not-retained", "user": "not-retained"},
            "processes": [{"pid": 1, "command": "not-retained"}],
            "host_sw_vers": {"tt_smi": "fixture"},
            "device_info": [
                {
                    "board_info": {
                        "board_id": "fixture-board",
                        "serial": "fixture-board",
                        "board_type": "p150a",
                        "bus_id": "0000:01:00.0",
                    },
                    "firmwares": {"fw_bundle_version": "19.6.0.0"},
                    "limits": {
                        "tdp_limit": "150",
                        "board_power_limit": "300",
                        "tdc_limit": "200",
                        "therm_trip_l1_limit": "90",
                        "thm_limit": "110",
                        "asic_fmax": "1350",
                    },
                    "smbus_telem": {
                        "BOARD_POWER_LIMIT": "0x12c",
                        "TDC_LIMIT_MAX": "0xc8",
                        "THERM_TRIP_COUNT": "0x0",
                    },
                    "telemetry": {
                        "power": "50",
                        "aiclk": "1350",
                        "asic_temperature": "60",
                        "current": "4",
                        "voltage": "12",
                        "board_power": "50",
                        "fan_speed": "1000",
                        "heartbeat": "1",
                    },
                }
            ],
        }
    )


def _row(
    seconds: int,
    *,
    aiclk: float = 1350,
    power: float = 50,
    board_power: float = 50,
    current: float = 50,
    temperature: float = 60,
) -> dict[str, object]:
    return {
        "timestamp_utc": f"2026-01-01T00:00:{seconds:02d}+00:00",
        "power_w": power,
        "aiclk_mhz": aiclk,
        "asic_temp_c": temperature,
        "timestamp_monotonic_ns": seconds * 1_000_000_000,
        "snapshot_elapsed_ns": 1,
        "current_a": current,
        "voltage_v": 12,
        "board_power_w": board_power,
        "fan_rpm": 1000,
        "heartbeat": seconds,
        "therm_trip_count": 0,
        "throttle_status": None,
        "throttle_reason": None,
    }


def _trace(
    *,
    aiclk: float = 1350,
    power: float = 50,
    board_power: float = 50,
    current: float = 50,
    temperature: float = 60,
) -> dict[str, object]:
    return {
        "schema": runner.ISSUE121_Q2_TRACE_SCHEMA,
        "schema_version": 1,
        "columns": list(runner.EXTENDED_TRACE_COLUMNS),
        "samples": [
            _row(0, aiclk=1350, power=50, board_power=50, current=50, temperature=60),
            _row(
                100,
                aiclk=1350,
                power=power,
                board_power=board_power,
                current=current,
                temperature=temperature,
            ),
            _row(
                200,
                aiclk=1200,
                power=power,
                board_power=board_power,
                current=current,
                temperature=temperature,
            ),
            _row(
                300,
                aiclk=1200,
                power=power,
                board_power=board_power,
                current=current,
                temperature=temperature,
            ),
        ],
    }


def test_q2_plan_is_frozen_and_names_the_extended_contract() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["status"] == "planned"
    assert plan["board_run"] is False
    assert plan["measurement_scope"]["device_id"] == 1
    assert plan["measurement_scope"]["runs"] == 3
    assert plan["measurement_scope"]["run_duration_seconds"] == 285
    assert plan["measurement_scope"]["run_duration_definition"] == (
        "The fixed workload interval targets 285 seconds of sustained device work; "
        "setup and teardown are outside the interval."
    )
    assert plan["measurement_scope"]["run_guard_seconds"] == 330
    assert plan["trace"]["mode"] == "extended-v1"
    assert plan["trace"]["legacy_columns"] == list(runner.LEGACY_POWER_TRACE_COLUMNS)
    assert plan["trace"]["columns"] == list(runner.EXTENDED_TRACE_COLUMNS)
    assert plan["source_snapshot"]["source_type"] == "prior_host_snapshot"
    assert plan["source_snapshot"]["reported_limit_key_values"]["tdc_limit"]["unit"] == "A"
    assert plan["source_snapshot"]["raw_smbus_fields"]["TDC_LIMIT_MAX"]["semantic_unit"] == "A"
    assert plan["source_snapshot"]["raw_smbus_fields"]["THM_LIMIT_THROTTLE"]["value"] is None
    assert plan["source_snapshot"]["raw_smbus_fields"]["THM_LIMIT_THROTTLE"]["unit"] == (
        "raw snapshot value"
    )
    assert plan["source_snapshot"]["raw_smbus_fields"]["THM_LIMIT_THROTTLE"]["semantic_unit"] == (
        "degC"
    )
    assert plan["revision_history"][0]["revision"] == 1
    assert plan["revision_history"][0]["date"] == "2026-10-16"
    assert plan["revision_history"][0]["change"] == (
        "Created the pre-measurement Q2 plan from the existing ADR-0013 and Issue #118 "
        "108-core surrogate, fixing device 1, three 300-second runs, the 330-second "
        "guard, extended-v1 fields, hypothesis thresholds, cooldown gate, and "
        "source-snapshot provenance."
    )
    assert plan["revision_history"][1] == {
        "revision": 2,
        "date": "2026-10-16",
        "change": (
            "pre-measurement change 300 -> 285 load and 150-300 -> 150-285 plateau "
            "to fit existing 40-second margin and 330-second guard; criteria and "
            "prediction unchanged; triggered by 8ad7ea9 preflight rejection with 0 "
            "device frames."
        ),
    }
    assert plan["revision_history"][2] == {
        "revision": 3,
        "date": "2026-10-16",
        "change": (
            "Clarified that device_id: 1 is the host device-node number while the "
            "single-device container opens visible device_id 0; hypothesis criteria "
            "and prediction unchanged."
        ),
    }
    assert plan["revision_history"][3] == {
        "revision": 4,
        "date": "2026-10-16",
        "change": (
            "post-measurement analysis change introduced by e09d4f9 after all 3 runs; "
            "before: AICLK onset used the first left-held segment clipped at run_start "
            "whose AICLK was <= asic_fmax*0.95, so the pre-run idle 800 MHz row could "
            "make onset t=0; after: AICLK onset uses the first trace sample segment "
            "whose timestamp is at or after run_start (post-start sample only), with "
            "the same 1350*0.95=1282.5 threshold; reason: exclude pre-run idle from "
            "load-onset timing. The pre-change definition classified all three "
            "completed runs H-OTHER, and the post-change definition also classified "
            "all three H-OTHER; criteria and prediction did not change."
        ),
    }
    assert "criteria and prediction unchanged" in plan["revision_history"][1]["change"]
    assert "330-second guard" in plan["revision_history"][1]["change"]
    assert "device_id: 1" in plan["revision_history"][2]["change"]
    assert "visible device_id 0" in plan["revision_history"][2]["change"]
    assert plan["revision_history"][4] == {
        "revision": 5,
        "date": "2026-10-16",
        "change": (
            "post-measurement unit-disposition remediation after all 3 runs: inspected the pinned host tt-smi 6.2.1 and tt-umd 0.9.8 source provenance without hardware access; TDC_LIMIT_MAX/limits.tdc_limit and current are established as comparable A values, and THM_LIMIT_THROTTLE is established as a degC thermal limit mapped directly to limits.therm_trip_l1_limit. Raw TDC_LIMIT_MAX and THM_LIMIT_THROTTLE values remain retained with source provenance; thresholds, criteria, and prediction are unchanged. Regenerated only analysis JSONs and this summary; no run record, trace, or snapshot changed."
        ),
    }
    assert plan["unit_provenance"]["status"] == "established"
    assert plan["unit_provenance"]["comparisons"]["current_a_vs_tdc_limit"]["comparable"] is True
    assert plan["unit_provenance"]["comparisons"]["asic_temp_c_vs_thm_limit_throttle"]["comparable"] is True
    assert plan["hypotheses"]["H-TDP"]["reference_only"].startswith("Issue #118 plateau 132-135 W")
    assert plan["hypotheses"]["H-TDP"]["plateau_proximity"]["lower_bound_at_150_w"] == 142.5
    assert "numeric_nonproximity_threshold_fraction" not in plan["hypotheses"]["H-OTHER"]
    assert plan["classification_policy"]["multiple_supported_known_hypotheses"] == "multiple"
    assert plan["prediction"]["expected_classification"] == ["H-TDC", "H-OTHER"]
    assert plan["prediction"]["expected_classification_note"] == (
        "H-TDC or H-OTHER; H-TDP and H-THM are unsupported by the prior evidence."
    )
    assert plan["prediction"]["hypotheses"]["H-TDP"].startswith("unsupported")
    assert plan["plateau"] == {
        "start_seconds": 150,
        "end_seconds": 285,
        "interval_definition": plan["plateau"]["interval_definition"],
        "requires_positive_duration_segments": True,
        "terminal_coverage_required": True,
    }


def test_q2_post_measurement_summary_is_separate_and_artifact_bound() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    observations = summary["post_hoc_observations"]

    assert summary["plan"]["revision"] == 5
    assert summary["unit_provenance"]["status"] == "established"
    assert observations["status"] == "post_measurement"
    assert observations["scope"] == (
        "Post-hoc observations only; these are not preregistered hypothesis results."
    )
    board = observations["board_power_plateau"]
    assert board["limit_w"] == 300.0
    assert [
        (
            row["time_weighted_mean_w"],
            row["min_w"],
            row["max_w"],
            row["delta_from_300_w"],
        )
        for row in board["runs"]
    ] == [
        (299.44635551097036, 241.0, 310.0, -0.5536444890296366),
        (300.88188737094816, 288.0, 313.0, 0.8818873709481636),
        (300.06724754488147, 247.0, 309.0, 0.067247544881468),
    ]

    transient = observations["transient_at_approximately_4_5_seconds"]["runs"]
    assert [
        (
            row["aiclk_mhz"],
            row["power_w"],
            row["current_a"],
            row["voltage_v"],
            row["board_power_w"],
        )
        for row in transient
    ] == [
        (1343.0, 210.0, 254.0, 0.83, 407.0),
        (1343.0, 212.0, 257.0, 0.83, 405.0),
        (1343.0, 210.0, 255.0, 0.83, 394.0),
    ]
    next_sample = observations["next_sample_at_approximately_6_7_seconds"]["runs"]
    assert [(row["aiclk_mhz"], row["board_power_w"]) for row in next_sample] == [
        (1037.0, 295.0),
        (1037.0, 292.0),
        (1031.0, 297.0),
    ]
    assert observations["voltage_and_power_interpretation"] == {
        "transient_voltage_v": 0.83,
        "post_transient_voltage_v_range": [0.73, 0.75],
        "relationship": "Power is approximately voltage*current in these telemetry samples.",
        "limitation": "This run cannot separate TDC from TDP.",
    }

    evolution = observations["load_plateau_evolution"]
    assert evolution["transition_values"] == {
        "aiclk_mhz_range": [1031.0, 1037.0],
        "asic_temp_c_range": [73.0, 73.3],
    }
    assert evolution["later_loaded_plateau_values"]["aiclk_mhz_range"] == [950.0, 962.0]
    assert evolution["later_loaded_plateau_values"]["board_power_w_remains_near"] == 300.0

    raw_throttle = observations["raw_snapshot_limits"]["THM_LIMIT_THROTTLE"]
    assert raw_throttle["pre"] == ["0x5a", "0x5a", "0x5a"]
    assert raw_throttle["post"] == ["0x5a", "0x5a", "0x5a"]
    assert raw_throttle["unit"] == "degC"
    assert raw_throttle["raw_unit"] is None
    assert raw_throttle["semantic_unit"] == "degC"
    assert "Raw snapshot value remains 0x5a" in raw_throttle["interpretation"]

    assert observations["unit_disposition"]["status"] == "established"
    assert observations["unit_disposition"]["comparisons"]["current_a_vs_tdc_limit"]["limit_unit"] == "A"
    assert observations["unit_disposition"]["comparisons"]["asic_temp_c_vs_thm_limit_throttle"]["limit_unit"] == "degC"
    assert observations["device_comparison"]["direct_comparison"] is False
    assert observations["device_comparison"]["issue_118"]["host_device_id"] == 0
    assert observations["device_comparison"]["q2_measurement"]["host_device_id"] == 1
    assert observations["device_comparison"]["q2_measurement"]["container_visible_device_id"] == 0
    assert observations["causality_and_next_intervention"]["causality"] == (
        "Correlation is not causation."
    )
    assert "new preregistered intervention" in observations[
        "causality_and_next_intervention"
    ]["requirement"]
    assert "below 300 W" in observations["causality_and_next_intervention"]["example"]

    onset = observations["aiclk_onset_definitions"]
    assert onset["threshold"] == {
        "baseline_asic_fmax_mhz": 1350.0,
        "threshold_expression": "1350*0.95=1282.5",
        "threshold_mhz": 1282.5,
    }
    assert onset["before"]["onset_times_seconds"] == [0.0, 0.0, 0.0]
    assert onset["before"]["classifications"] == ["H-OTHER"] * 3
    assert onset["after"]["onset_times_seconds"] == [
        6.727259794,
        6.719855854,
        6.731211923,
    ]
    assert onset["after"]["classifications"] == ["H-OTHER"] * 3
    assert "post-start sample only" in onset["after"]["definition"]
    assert "pre-run idle" in onset["reason"]

    assert [run["classification"] for run in summary["runs"]] == ["H-OTHER"] * 3
    assert all(run["signals"]["current_a"]["unit"] == "A" for run in summary["runs"])
    assert all(
        run["signals"]["current_a"]["limit_proximity"]["limit_unit"] == "A"
        for run in summary["runs"]
    )
    assert summary["prediction"] == {
        "expected_classification": ["H-TDC", "H-OTHER"],
        "excluded_by_prior_evidence": ["H-TDP", "H-THM"],
        "actual_classifications": ["H-OTHER", "H-OTHER", "H-OTHER"],
        "outcome": "matched",
        "comparison": (
            "All three runs classified H-OTHER; H-TDP and H-THM were not supported, "
            "while current evidence made H-TDC not_supported under the fixed criteria."
        ),
    }

    tracked = [
        path
        for path in subprocess.run(
            ["git", "ls-files", "--", "docs/measurements/issue121-q2/runs"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if "/analysis-" not in path
    ]
    assert tracked
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in IMMUTABLE_RUN_ROOT.rglob("*")
        if path.is_file() and "/analysis-" not in path.relative_to(ROOT).as_posix()
    }
    assert actual == set(tracked)
    for relative in tracked:
        expected = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert (ROOT / relative).read_bytes() == expected


def test_gddr_fan_supplement_is_complete_and_sanitized() -> None:
    record = json.loads(GDDR_FAN_SUPPLEMENT.read_text(encoding="utf-8"))
    acquisition = record["acquisition"]

    assert record["record_schema"] == "issue121-q2-post-hoc-gddr-fan-observation-v1"
    assert acquisition["acquired_at_utc"] == "2026-10-16T13:13:52Z"
    assert acquisition["raw_reported_time"] == "2026-10-16T22:13:53.012165"
    assert acquisition["raw_snapshot_basename"] == (
        "raw-tt-smi-snapshot-20261016T131352Z.json"
    )
    assert "outside the Q2 harness" in acquisition["provenance"]
    assert acquisition["boards_captured"] == 2
    assert record["environment"] == {
        "tt_smi_version": "6.2.1",
        "tt_umd_version": "0.9.8",
        "firmware_bundle": "19.6.0.0",
        "kernel": "6.8.0-138-generic",
        "kernel_driver": "TT-KMD 2.11.0",
    }

    boards = {board["pci_bus_id"]: board for board in record["boards"]}
    assert set(boards) == {"0000:06:00.0", "0000:09:00.0"}
    board_zero_fields = boards["0000:06:00.0"]["snapshot_fields"]
    board_one_fields = boards["0000:09:00.0"]["snapshot_fields"]
    assert board_zero_fields["asic_temperature"] == {
        "raw": "0x34b67a",
        "value_c": 52.7,
        "unit": "degC",
    }
    assert board_one_fields["asic_temperature"] == {
        "raw": "0x2f43f2",
        "value_c": 47.3,
        "unit": "degC",
    }
    assert board_zero_fields["gddr_words"]["GDDR_6_7_TEMP"] == {
        "raw": "0x34342e30",
        "channels_c": {
            "GDDR_6": {"bottom_c": 48, "top_c": 46},
            "GDDR_7": {"bottom_c": 52, "top_c": 52},
        },
    }
    assert board_one_fields["gddr_words"]["GDDR_2_3_TEMP"] == {
        "raw": "0x48482828",
        "channels_c": {
            "GDDR_2": {"bottom_c": 40, "top_c": 40},
            "GDDR_3": {"bottom_c": 72, "top_c": 72},
        },
    }
    assert board_zero_fields["max_gddr_temp"] == {
        "raw": "0x38",
        "value": None,
        "unit": "unknown",
    }
    assert board_one_fields["max_gddr_temp"] == {
        "raw": "0x48",
        "value": None,
        "unit": "unknown",
    }
    assert board_zero_fields["fan_speed"] == {
        "raw": "0x2a",
        "value_percent": 42,
        "unit": "percent",
        "meaning": "firmware target fan speed",
    }
    assert board_one_fields["fan_speed"] == {
        "raw": "0x46",
        "value_percent": 70,
        "unit": "percent",
        "meaning": "firmware target fan speed",
    }
    assert board_zero_fields["fan_rpm"] == {
        "raw": "0x80c",
        "value_rpm": 2060,
        "unit": "rpm",
        "meaning": "reported actual fan speed",
    }
    assert board_one_fields["fan_rpm"] == {
        "raw": "0xd96",
        "value_rpm": 3478,
        "unit": "rpm",
        "meaning": "reported actual fan speed",
    }

    provenance = record["source_provenance"]
    assert provenance["packages"] == [
        {
            "name": "tt-smi",
            "version": "6.2.1",
            "repository": "tenstorrent/tt-smi",
            "source_files": ["tt_smi/backend.py", "tt_smi/constants.py", "tt_smi/utils.py"],
        },
        {
            "name": "tt-umd",
            "version": "0.9.8",
            "repository": "tenstorrent/tt-umd",
            "source_files": [
                "device/api/umd/device/types/telemetry.hpp",
                "device/api/umd/device/types/gddr_telemetry.hpp",
                "device/api/umd/device/firmware/firmware_info_provider.hpp",
                "device/api/umd/device/firmware/firmware_telemetry_mapping.hpp",
                "device/firmware/firmware_info_provider.cpp",
            ],
            "source_kind": "equivalent repository source for the installed wheel",
        },
    ]
    assert provenance["verification_scope"].startswith("Local source inspection only")
    assert provenance["fields"]["GDDR_x_y_TEMP"]["unit"] == "degC"
    assert provenance["fields"]["GDDR_x_y_TEMP"]["packing"].startswith(
        "Two modules share each 32-bit word"
    )
    assert "bit positions" in provenance["fields"]["GDDR_x_y_TEMP"]["byte_order"]
    assert provenance["fields"]["MAX_GDDR_TEMP"]["unit"] == "unknown"
    assert provenance["fields"]["MAX_GDDR_TEMP"]["packing"] == "unknown"
    assert provenance["fields"]["MAX_GDDR_TEMP"]["byte_order"] == "unknown"
    assert provenance["fields"]["FAN_SPEED"]["unit"] == "percent"
    assert provenance["fields"]["FAN_RPM"]["unit"] == "rpm"

    comparison = record["derived_comparison"]
    assert comparison["q2_run_snapshots"]["source_count"] == 14
    assert comparison["cooldown_snapshots"]["source_count"] == 2
    all_snapshots = (
        comparison["q2_run_snapshots"]["records"]
        + comparison["cooldown_snapshots"]["records"]
    )
    assert all(
        snapshot["board"]["pci_bus_id"] == "0000:09:00.0"
        for snapshot in all_snapshots
    )
    assert all(
        set(snapshot["raw"]) == {
            "ASIC_TEMPERATURE",
            "GDDR_0_1_TEMP",
            "GDDR_2_3_TEMP",
            "GDDR_4_5_TEMP",
            "GDDR_6_7_TEMP",
            "MAX_GDDR_TEMP",
            "FAN_SPEED",
            "FAN_RPM",
        }
        for snapshot in all_snapshots
    )
    assert comparison["issue_118_gddr_temperature"]["status"] == "absent"
    assert "no GDDR temperature fields" in comparison[
        "issue_118_gddr_temperature"
    ]["finding"]

    encoded = json.dumps(record)
    for forbidden in (
        "host_info",
        "processes",
        "cmdline",
        "hostname",
        "Memory",
        "Platform",
        "Python",
        "docker_ps",
        "snapshot_command",
        "/home/",
        "/root/",
    ):
        assert forbidden not in encoded
    assert all(snapshot["source"].startswith("docs/") for snapshot in all_snapshots)

    summary_observation = json.loads(SUMMARY.read_text(encoding="utf-8"))[
        "post_hoc_observations"
    ]["gddr_fan_observation"]
    idle_snapshot = summary_observation["idle_snapshot"]
    assert idle_snapshot["acquired_at_utc"] == "2026-10-16T13:13:52Z"
    assert idle_snapshot["raw_reported_time"] == "2026-10-16T22:13:53.012165"
    assert "outside the Q2 harness" in idle_snapshot["provenance"]
    assert len(idle_snapshot["boards"]) == 2
    summary_boards = {board["pci_bus_id"]: board for board in idle_snapshot["boards"]}
    for pci_bus_id, board in boards.items():
        fields = board["snapshot_fields"]
        assert summary_boards[pci_bus_id]["asic_temperature_c"] == fields[
            "asic_temperature"
        ]["value_c"]
        assert summary_boards[pci_bus_id]["gddr_words"] == {
            name: value["raw"] for name, value in fields["gddr_words"].items()
        }
        assert summary_boards[pci_bus_id]["fan_target_percent"] == fields[
            "fan_speed"
        ]["value_percent"]
        assert summary_boards[pci_bus_id]["fan_rpm"] == fields["fan_rpm"]["value_rpm"]
    assert len(summary_observation["q2_device_1_snapshot_comparison"]["run_pre_post"]) == 6
    assert len(summary_observation["q2_device_1_snapshot_comparison"]["cooldown"]) == 2
    issue_118 = summary_observation["issue_118_comparison"]
    assert issue_118["gddr_temperature_status"] == "absent"
    assert "no GDDR temperature fields" in issue_118["finding"]
    assert "cannot be attributed to run conditions alone" in issue_118[
        "thermal_limitation"
    ]
    assert "confounded" in issue_118["board_thermal_confound"]["statement"]
    assert "speculation" in summary_observation["interpretation"][0]


def test_regenerated_analysis_retains_unit_provenance_and_effects() -> None:
    for run in json.loads(SUMMARY.read_text(encoding="utf-8"))["runs"]:
        result = json.loads((ROOT / run["analysis"]).read_text(encoding="utf-8"))
        provenance = result["unit_provenance"]
        assert provenance["status"] == "established"
        assert provenance["packages"] == [
            {
                "name": "tt-smi",
                "version": "6.2.1",
                "repository": "tenstorrent/tt-smi",
            },
            {
                "name": "tt-umd",
                "version": "0.9.8",
                "repository": "tenstorrent/tt-umd",
            },
        ]
        assert result["statistics"]["signals"]["current_a"]["limit_proximity"]["limit_unit"] == "A"
        assert result["hypotheses"]["H-TDC"]["classification"] == "not_supported"
        throttle = result["hypotheses"]["H-THM"]["details"]["thm_limit_throttle"]
        assert throttle["status"] == "pass"
        assert throttle["limit_value"] == 90.0
        assert throttle["limit_unit"] == "degC"


def test_record_cli_retains_established_units_and_classifications(tmp_path: Path) -> None:
    output = tmp_path / "analysis.json"
    assert analysis.main(
        [
            "--trace",
            str(Q2_TRACE),
            "--record",
            str(Q2_RUN_RECORD),
            "--output",
            str(output),
        ]
    ) == 0

    rerun = json.loads(output.read_text(encoding="utf-8"))
    committed = json.loads(Q2_COMMITTED_ANALYSIS.read_text(encoding="utf-8"))

    assert rerun["unit_provenance"] == committed["unit_provenance"]
    assert rerun["statistics"]["signals"]["current_a"]["limit_proximity"] == committed[
        "statistics"
    ]["signals"]["current_a"]["limit_proximity"]
    assert rerun["hypotheses"]["H-TDC"] == committed["hypotheses"]["H-TDC"]
    assert rerun["hypotheses"]["H-THM"]["classification"] == committed["hypotheses"][
        "H-THM"
    ]["classification"]
    assert rerun["hypotheses"]["H-THM"]["details"]["thm_limit_throttle"] == committed[
        "hypotheses"
    ]["H-THM"]["details"]["thm_limit_throttle"]


def test_snapshot_command_is_bounded_and_host_fields_are_not_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=_snapshot(), returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    capture = runner.capture_readonly_snapshot(stage="pre_run")

    assert calls[0][0] == ("tt-smi", "-s", "--snapshot_no_tty")
    assert calls[0][1]["timeout"] == runner.SNAPSHOT_TIMEOUT_SECONDS
    assert capture["status"] == "ok"
    assert "host_info" not in json.dumps(capture)
    assert "processes" not in json.dumps(capture)
    assert capture["snapshot"]["device_info"][0]["smbus_telem"]["TDC_LIMIT_MAX"] == "0xc8"


def test_workload_command_pins_host_node_and_visible_device_zero() -> None:
    output_dir = Path("out/q2-run-1")
    command = runner.build_workload_command(output_dir=output_dir, run_id="q2-run-1")

    assert runner.DEVICE_ID == 1
    assert runner.DEVICE_NODE == "/dev/tenstorrent/1"
    assert runner.CONTAINER_VISIBLE_DEVICE_ID == 0
    assert command[1] == str(output_dir)
    assert "--device-id" in command
    assert command[command.index("--device-id") + 1] == "0"
    assert command[command.index("--outer-timeout-seconds") + 1] == "330"
    assert runner.RUN_DURATION_SECONDS == 285.0
    assert runner.RUN_GUARD_SECONDS == 330.0
    calibration = runner.build_calibration_payload()
    assert calibration["calibration"]["target_seconds"] == 285.0
    assert command[command.index("--out") + 1] == "/out/runner-result-q2-run-1.json"
    assert command[command.index("--calibration-json") + 1] == "/out/issue121-q2-calibration.json"
    assert "/out/" in " ".join(command)
    assert "/home/" not in " ".join(command)


@pytest.mark.parametrize("invalid_device_id", [1, -1, 2])
def test_workload_command_rejects_nonzero_container_visible_device(
    invalid_device_id: int,
) -> None:
    with pytest.raises(runner.Issue121Q2Error, match="container-visible device_id must be 0"):
        runner.build_workload_command(
            output_dir=Path("out/q2-run-1"),
            run_id="q2-run-1",
            container_device_id=invalid_device_id,
        )
    with pytest.raises(runner.Issue121Q2Error, match="container-visible device_id must be 0"):
        runner.build_workload_environment(
            run_id="q2-run-1",
            image_digest="sha256:" + "a" * 64,
            container_device_id=invalid_device_id,
        )


def test_run_one_rejects_nonzero_container_device_before_capture_or_workload(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.Issue121Q2Error, match="container-visible device_id must be 0"):
        runner.run_one(
            tmp_path,
            run_id="q2-run-1",
            board_id="fixture-board",
            bus_id="0000:01:00.0",
            image_digest="sha256:" + "a" * 64,
            harness_commit="b" * 40,
            kernel="Linux fixture",
            kernel_driver_version="2.11.0",
            container_device_id=1,
            workload_runner=lambda *_args: pytest.fail("workload must not run"),
            capture_fn=lambda **_kwargs: pytest.fail("snapshot must not run"),
        )


@pytest.mark.parametrize(
    "invalid_name",
    [
        ".",
        "..",
        "../runner-result.json",
        "nested/runner-result.json",
        "nested\\runner-result.json",
    ],
)
def test_workload_command_rejects_artifact_path_traversal(invalid_name: str) -> None:
    with pytest.raises(runner.Issue121Q2Error, match="basename"):
        runner.build_workload_command(
            output_dir=Path("out/q2-run-1"),
            run_id="q2-run-1",
            result_name=invalid_name,
        )
    with pytest.raises(runner.Issue121Q2Error, match="basename"):
        runner.build_workload_command(
            output_dir=Path("out/q2-run-1"),
            run_id="q2-run-1",
            calibration_name=invalid_name,
        )


@pytest.mark.parametrize("invalid_output_dir", [Path(".."), Path("out/../q2-run-1")])
def test_workload_command_rejects_output_dir_path_traversal(
    invalid_output_dir: Path,
) -> None:
    with pytest.raises(runner.Issue121Q2Error, match="path traversal"):
        runner.build_workload_command(output_dir=invalid_output_dir, run_id="q2-run-1")


def test_workload_environment_uses_fixed_ghcr_image_name() -> None:
    digest = "sha256:" + "a" * 64
    environment = runner.build_workload_environment(
        run_id="q2-run-1",
        image_digest=digest,
    )

    assert environment["HEKATUS_TT_IMAGE"] == (
        "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@" + digest
    )
    assert environment["HEKATUS_TT_DEVICE_NODE"] == "/dev/tenstorrent/1"
    assert environment["HEKATUS_TT_CONTAINER_DEVICE_ID"] == "0"


def test_workload_output_retains_only_a_bounded_clean_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "".join(f"stdout-{index}\n" for index in range(100))
    stderr = "clean diagnostic\n"

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner._run_workload(("fixture-runner",), {}, repo_root=Path("."))

    assert result["stdout_tail"] == "".join(
        f"stdout-{index}\n"
        for index in range(100 - runner.WORKLOAD_OUTPUT_TAIL_LINES, 100)
    )
    assert result["stderr_tail"] == stderr
    assert "output_omissions" not in result
    assert len(result["stdout_tail"]) <= runner.WORKLOAD_OUTPUT_TAIL_MAX_CHARS


def test_workload_output_omits_a_tail_containing_a_user_path() -> None:
    result = runner._normalize_workload_result(
        {
            "returncode": 1,
            "status": "completed",
            "timed_out": False,
            "stdout": "failure in /home/alice/private/repository\n" + "safe\n" * 100,
            "stderr": "clean diagnostic\n",
        }
    )

    assert "stdout_tail" not in result
    assert result["output_omissions"] == {"stdout": "not_saved: path_detected"}
    assert result["stderr_tail"] == "clean diagnostic\n"
    assert "/home/" not in json.dumps(result)


def test_run_workload_persists_raw_bytes_and_stream_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout = b"header" + bytes((0, 255)) + b"\nlast line\n"
    stderr = b"failed diagnostic\r\n"

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        stdout_handle = kwargs["stdout"]
        stderr_handle = kwargs["stderr"]
        assert hasattr(stdout_handle, "write")
        assert hasattr(stderr_handle, "write")
        stdout_handle.write(stdout)
        stderr_handle.write(stderr)
        return SimpleNamespace(returncode=7, stdout=None, stderr=None)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner._run_workload(
        ("fixture-runner",), {}, repo_root=Path("."), output_dir=tmp_path
    )

    assert (tmp_path / runner.WORKLOAD_STDOUT_LOG_NAME).read_bytes() == stdout
    assert (tmp_path / runner.WORKLOAD_STDERR_LOG_NAME).read_bytes() == stderr
    assert result["returncode"] == 7
    assert result["local_diagnostics"] == {
        "stdout": {
            "path": runner.WORKLOAD_STDOUT_LOG_NAME,
            "byte_count": len(stdout),
            "sha256": hashlib.sha256(stdout).hexdigest(),
            "status": "saved",
        },
        "stderr": {
            "path": runner.WORKLOAD_STDERR_LOG_NAME,
            "byte_count": len(stderr),
            "sha256": hashlib.sha256(stderr).hexdigest(),
            "status": "saved",
        },
    }
    assert result["stdout_tail"] == "header\x00�\nlast line\n"
    assert result["stderr_tail"] == "failed diagnostic\r\n"


def test_run_workload_timeout_keeps_available_output_in_raw_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout = b"timeout stdout\n"
    stderr = b"timeout stderr\n"

    def fake_run(*_args: object, **kwargs: object) -> None:
        kwargs["stdout"].write(stdout)
        kwargs["stderr"].write(stderr)
        raise subprocess.TimeoutExpired(("fixture-runner",), 1)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner._run_workload(
        ("fixture-runner",), {}, repo_root=Path("."), output_dir=tmp_path
    )

    assert result["status"] == "timeout"
    assert result["timed_out"] is True
    assert (tmp_path / runner.WORKLOAD_STDOUT_LOG_NAME).read_bytes() == stdout
    assert (tmp_path / runner.WORKLOAD_STDERR_LOG_NAME).read_bytes() == stderr
    assert result["local_diagnostics"]["stdout"]["byte_count"] == len(stdout)
    assert result["local_diagnostics"]["stderr"]["sha256"] == hashlib.sha256(stderr).hexdigest()


def test_run_one_failed_path_output_is_local_only_and_record_survives(tmp_path: Path) -> None:
    counter = iter((0, 1_000_000, 2_000_000, 3_000_000))

    def fake_capture(*, stage: str, **_kwargs: object) -> dict[str, object]:
        timestamp = next(counter)
        return {
            "schema": runner.ISSUE121_Q2_SNAPSHOT_SCHEMA,
            "stage": stage,
            "command": list(runner.SNAPSHOT_COMMAND_TUPLE),
            "timeout_seconds": 5.0,
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
            "timestamp_monotonic_start_ns": timestamp,
            "timestamp_monotonic_ns": timestamp + 1,
            "snapshot_elapsed_ns": 1,
            "status": "ok",
            "returncode": 0,
            "error_type": None,
            "stdout_sha256": "a" * 64,
            "snapshot": json.loads(_snapshot()),
        }

    stdout = b"failure in /home/alice/private/repository\n"
    stderr = b"temporary output in /tmp/private\n"

    def failed_workload(*_args: object) -> dict[str, object]:
        return {
            "returncode": 3,
            "status": "completed",
            "timed_out": False,
            "stdout": stdout,
            "stderr": stderr,
        }

    record = runner.run_one(
        tmp_path,
        run_id="q2-run-failed-output",
        board_id="fixture-board",
        bus_id="0000:01:00.0",
        image_digest="sha256:" + "a" * 64,
        harness_commit="b" * 40,
        kernel="Linux fixture",
        kernel_driver_version="2.11.0",
        workload_runner=failed_workload,
        capture_fn=fake_capture,
    )

    assert record["status"] == "failed"
    assert (tmp_path / runner.WORKLOAD_STDOUT_LOG_NAME).read_bytes() == stdout
    assert (tmp_path / runner.WORKLOAD_STDERR_LOG_NAME).read_bytes() == stderr
    workload_result = record["workload_result"]
    assert "stdout_tail" not in workload_result
    assert "stderr_tail" not in workload_result
    assert workload_result["output_omissions"] == {
        "stdout": "not_saved: path_detected",
        "stderr": "not_saved: path_detected",
    }
    assert workload_result["local_diagnostics"]["stdout"]["path"] == (
        runner.WORKLOAD_STDOUT_LOG_NAME
    )
    assert workload_result["local_diagnostics"]["stderr"]["byte_count"] == len(stderr)
    assert workload_result["local_diagnostics"]["stdout"]["sha256"] == hashlib.sha256(
        stdout
    ).hexdigest()
    persisted = json.dumps(record)
    assert "/home/" not in persisted
    assert "/tmp/" not in persisted
    assert str(tmp_path) not in persisted
    assert (tmp_path / record["output_paths"]["record"]).exists()


def test_workload_log_write_failure_is_recorded_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "_write_workload_log", lambda *_args: "write_failed")
    fields = runner._persist_workload_streams(
        tmp_path, {"stdout": b"output", "stderr": b"diagnostic"}
    )

    assert fields["local_diagnostics"]["stdout"]["status"] == "write_failed"
    assert fields["local_diagnostics"]["stderr"]["status"] == "write_failed"
    assert fields["local_diagnostics"]["stdout"]["path"] == runner.WORKLOAD_STDOUT_LOG_NAME


def test_resident_wrapper_probe_reconstructs_added_args_without_device() -> None:
    vector = wrapper_probe.build_resident_docker_argv(
        repo_root=Path("/repo"),
        output_dir=Path("/out/q2"),
        image="registry.example/tt@sha256:" + "a" * 64,
        run_id="static-probe",
        container_name="hekatus-bench-static",
        container_timeout_seconds=330,
        runner_args=("--condition", "newton_schulz"),
        device_node="/dev/tenstorrent/1",
    )

    assert "--device" not in vector
    assert "-v" in vector
    mount_indices = [index for index, argument in enumerate(vector) if argument == "-v"]
    assert [vector[index + 1] for index in mount_indices] == [
        "/dev/hugepages-1G:/dev/hugepages-1G",
        "/repo:/work",
        "/out/q2:/out",
    ]
    runner_index = vector.index(wrapper_probe.RESIDENT_RUNNER)
    assert vector[runner_index:] == (
        wrapper_probe.RESIDENT_RUNNER,
        "--out",
        "/out/runner-result.json",
        "--condition",
        "newton_schulz",
        "--env-json",
        "/out/env-static-probe.json",
        "--power-trace",
        "power-static-probe.csv",
    )

    wrapper_source = (ROOT / "enodia/tt/bench/run_in_container.sh").read_text(encoding="utf-8")
    assert '    -v "${OUT_DIR}:/out" \\' in wrapper_source
    assert '  RESIDENT_ARGS+=(--env-json "${RESIDENT_ENV_JSON}")' in wrapper_source
    assert '    --device "${DEVICE_NODE}" \\' in wrapper_source

    q2_vector = wrapper_probe.build_resident_docker_argv(
        repo_root=Path("/repo"),
        output_dir=Path("/out/q2"),
        image="registry.example/tt@sha256:" + "a" * 64,
        run_id="q2-wrapper-id",
        container_name="hekatus-bench-static",
        container_timeout_seconds=330,
        runner_args=(
            "--out",
            "/out/runner-result-q2-run-1.json",
            "--device-id",
            "0",
            "--condition",
            "newton_schulz",
            "--calibration-json",
            "/out/issue121-q2-calibration.json",
            "--outer-timeout-seconds",
            "330",
        ),
    )
    q2_runner_index = q2_vector.index(wrapper_probe.RESIDENT_RUNNER)
    assert q2_vector[q2_runner_index:] == (
        wrapper_probe.RESIDENT_RUNNER,
        "--out",
        "/out/runner-result-q2-run-1.json",
        "--device-id",
        "0",
        "--condition",
        "newton_schulz",
        "--calibration-json",
        "/out/issue121-q2-calibration.json",
        "--outer-timeout-seconds",
        "330",
        "--env-json",
        "/out/env-q2-wrapper-id.json",
        "--power-trace",
        "power-q2-wrapper-id.csv",
    )


def test_run_one_persists_pre_post_snapshots_and_provenance(tmp_path: Path) -> None:
    counter = iter((0, 1_000_000, 2_000_000, 3_000_000))
    def fake_capture(*, stage: str, **_kwargs: object) -> dict[str, object]:
        timestamp = next(counter)
        return {
            "schema": runner.ISSUE121_Q2_SNAPSHOT_SCHEMA,
            "stage": stage,
            "command": list(runner.SNAPSHOT_COMMAND_TUPLE),
            "timeout_seconds": 5.0,
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
            "timestamp_monotonic_start_ns": timestamp,
            "timestamp_monotonic_ns": timestamp + 1,
            "snapshot_elapsed_ns": 1,
            "status": "ok",
            "returncode": 0,
            "error_type": None,
            "stdout_sha256": "a" * 64,
            "snapshot": json.loads(_snapshot()),
        }

    workload_calls: list[tuple[tuple[str, ...], dict[str, str], float]] = []

    def fake_workload(
        command: tuple[str, ...], environment: dict[str, str], timeout: float
    ) -> dict[str, object]:
        workload_calls.append((command, environment, timeout))
        return {
            "returncode": 0,
            "status": "completed",
            "timed_out": False,
            "stdout": "resident record -> /out/runner-result-q2-run-1.json\n",
            "stderr": "",
        }

    record = runner.run_one(
        tmp_path,
        run_id="q2-run-1",
        board_id="fixture-board",
        bus_id="0000:01:00.0",
        image_digest="sha256:" + "a" * 64,
        harness_commit="b" * 40,
        kernel="Linux fixture",
        kernel_driver_version="2.11.0",
        workload_runner=fake_workload,
        capture_fn=fake_capture,
    )

    assert record["status"] == "ok"
    assert record["workload"]["device_id"] == 1
    assert record["workload"]["container_device_id"] == 0
    assert record["workload"]["device_node"] == "/dev/tenstorrent/1"
    assert record["workload_result"]["stdout_tail"] == (
        "resident record -> /out/runner-result-q2-run-1.json\n"
    )
    assert record["workload_result"]["stderr_tail"] == ""
    assert record["workload_result"]["local_diagnostics"]["stdout"]["path"] == (
        runner.WORKLOAD_STDOUT_LOG_NAME
    )
    assert record["output_paths"]["workload_stdout"] == runner.WORKLOAD_STDOUT_LOG_NAME
    assert record["output_paths"]["workload_stderr"] == runner.WORKLOAD_STDERR_LOG_NAME
    assert (tmp_path / runner.WORKLOAD_STDOUT_LOG_NAME).read_bytes() == (
        b"resident record -> /out/runner-result-q2-run-1.json\n"
    )
    assert (tmp_path / runner.WORKLOAD_STDERR_LOG_NAME).read_bytes() == b""
    assert record["workload"]["run_duration_target_seconds"] == 285.0
    assert record["workload"]["run_guard_seconds"] == 330.0
    assert record["run_timing"]["target_duration_seconds"] == 285.0
    assert record["run_timing"]["guard_seconds"] == 330.0
    assert record["environment"]["board"]["serial"] == "fixture-board"
    assert record["environment"]["board"]["identity"]["bus_id"] == "0000:01:00.0"
    assert record["environment"]["device_id"] == 1
    assert record["environment"]["device_node"] == "/dev/tenstorrent/1"
    assert record["environment"]["container_device_id"] == 0
    assert record["environment"]["firmware"]["bundle"] == "19.6.0.0"
    assert record["environment"]["kernel_driver_version"] == "2.11.0"
    assert record["environment"]["image_digest"] == "sha256:" + "a" * 64
    assert record["environment"]["harness_commit"] == "b" * 40
    assert record["trace"]["board_id"] == "fixture-board"
    assert record["trace"]["bus_id"] == "0000:01:00.0"
    assert record["snapshot_limits"]["pre"]["recognized"]["asic_fmax_mhz"]["value"] == "1350"
    assert record["snapshot_limits"]["pre"]["THM_LIMIT_THROTTLE"] is None
    assert record["raw_window"]["columns"] == list(runner.RAW_WINDOW_COLUMNS)
    assert record["snapshots"]["pre"]["path"].startswith("pre-snapshot-")
    assert record["snapshots"]["pre"]["identity"]["serial"] == "fixture-board"
    assert record["snapshots"]["pre"]["identity"]["bus_id"] == "0000:01:00.0"
    assert record["snapshots"]["post"]["path"].startswith("post-snapshot-")
    assert record["snapshots"]["post"]["identity"]["serial"] == "fixture-board"
    assert record["snapshots"]["post"]["identity"]["bus_id"] == "0000:01:00.0"
    assert record["snapshot_identities"]["post"]["bus_id"] == "0000:01:00.0"
    assert len(workload_calls) == 1
    assert workload_calls[0][0][workload_calls[0][0].index("--device-id") + 1] == "0"
    assert workload_calls[0][1]["HEKATUS_TT_DEVICE_NODE"] == "/dev/tenstorrent/1"
    assert workload_calls[0][1]["HEKATUS_TT_CONTAINER_DEVICE_ID"] == "0"
    assert workload_calls[0][0][1] == str(tmp_path)
    assert workload_calls[0][0][1] != "/out"
    assert workload_calls[0][2] == runner.RUN_GUARD_SECONDS
    calibration = json.loads(
        (tmp_path / record["output_paths"]["calibration"]).read_text(encoding="utf-8")
    )
    assert calibration["calibration"]["target_seconds"] == 285.0
    assert (tmp_path / record["output_paths"]["calibration"]).exists()
    persisted = json.dumps(record)
    assert "hostname" not in persisted
    assert "username" not in persisted
    assert "/home/" not in persisted
    assert str(tmp_path) not in persisted
    assert "--calibration-json" not in persisted
    assert (tmp_path / record["output_paths"]["record"]).exists()


@pytest.mark.parametrize(
    ("post_status", "post_returncode", "post_error"),
    [
        ("timeout", None, "TimeoutExpired"),
        ("command_failed", 7, "nonzero_returncode"),
        ("invalid_json", 0, "invalid_snapshot"),
    ],
    ids=["timeout", "nonzero", "invalid"],
)
def test_run_one_fails_closed_when_post_snapshot_capture_is_unusable(
    tmp_path: Path,
    post_status: str,
    post_returncode: int | None,
    post_error: str,
) -> None:
    capture_number = 0

    def fake_capture(*, stage: str, **_kwargs: object) -> dict[str, object]:
        nonlocal capture_number
        capture_number += 1
        is_post = stage == "post_run"
        return {
            "schema": runner.ISSUE121_Q2_SNAPSHOT_SCHEMA,
            "stage": stage,
            "command": list(runner.SNAPSHOT_COMMAND_TUPLE),
            "timeout_seconds": 5.0,
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
            "timestamp_monotonic_start_ns": capture_number * 1_000_000,
            "timestamp_monotonic_ns": capture_number * 1_000_000 + 1,
            "snapshot_elapsed_ns": 1,
            "status": post_status if is_post else "ok",
            "returncode": post_returncode if is_post else 0,
            "error_type": post_error if is_post else None,
            "stdout_sha256": "a" * 64,
            "snapshot": None if is_post else json.loads(_snapshot()),
        }

    record = runner.run_one(
        tmp_path,
        run_id="q2-run-post-failure",
        board_id="fixture-board",
        bus_id="0000:01:00.0",
        image_digest="sha256:" + "a" * 64,
        harness_commit="b" * 40,
        kernel="Linux fixture",
        kernel_driver_version="2.11.0",
        workload_runner=lambda *_args: {
            "returncode": 0,
            "status": "completed",
            "timed_out": False,
        },
        capture_fn=fake_capture,
        trace_interval_seconds=100.0,
    )

    assert record["status"] == "failed"
    assert record["snapshots"]["post"]["status"] == post_status
    assert record["snapshots"]["post"]["identity"]["status"] == "unavailable"
    assert post_status in record["snapshots"]["post"]["identity"]["reason"]
    assert record["snapshot_identities"]["post"]["status"] == "unavailable"
    assert record["snapshot_limits"]["post"]["status"] == "unavailable"
    assert record["trace"]["post_row_written"] is False


def test_run_one_fails_closed_when_post_snapshot_identity_mismatches(
    tmp_path: Path,
) -> None:
    capture_number = 0

    def fake_capture(*, stage: str, **_kwargs: object) -> dict[str, object]:
        nonlocal capture_number
        capture_number += 1
        snapshot = json.loads(_snapshot())
        if stage == "post_run":
            snapshot["device_info"][0]["board_info"]["bus_id"] = "0000:02:00.0"
        return {
            "schema": runner.ISSUE121_Q2_SNAPSHOT_SCHEMA,
            "stage": stage,
            "command": list(runner.SNAPSHOT_COMMAND_TUPLE),
            "timeout_seconds": 5.0,
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
            "timestamp_monotonic_start_ns": capture_number * 1_000_000,
            "timestamp_monotonic_ns": capture_number * 1_000_000 + 1,
            "snapshot_elapsed_ns": 1,
            "status": "ok",
            "returncode": 0,
            "error_type": None,
            "stdout_sha256": "a" * 64,
            "snapshot": snapshot,
        }

    record = runner.run_one(
        tmp_path,
        run_id="q2-run-post-identity-mismatch",
        board_id="fixture-board",
        bus_id="0000:01:00.0",
        image_digest="sha256:" + "a" * 64,
        harness_commit="b" * 40,
        kernel="Linux fixture",
        kernel_driver_version="2.11.0",
        workload_runner=lambda *_args: {
            "returncode": 0,
            "status": "completed",
            "timed_out": False,
        },
        capture_fn=fake_capture,
        trace_interval_seconds=100.0,
    )

    assert record["status"] == "failed"
    assert record["snapshots"]["post"]["status"] == "ok"
    assert record["snapshots"]["post"]["identity"]["status"] == "unavailable"
    assert "bus_id" in record["snapshots"]["post"]["identity"]["reason"]
    assert record["snapshot_identities"]["post"]["status"] == "unavailable"
    assert record["trace"]["post_row_written"] is False


def test_snapshot_limits_preserve_raw_throttle_and_recognized_values() -> None:
    snapshot = json.loads(_snapshot())
    snapshot["device_info"][0]["smbus_telem"]["THM_LIMIT_THROTTLE"] = "0x5a"
    captured = {"stage": "post_run", "snapshot": snapshot}

    limits = runner._extract_snapshot_limits(
        captured, board_id="fixture-board", bus_id="0000:01:00.0"
    )

    assert limits["THM_LIMIT_THROTTLE"] == "0x5a"
    assert limits["raw"]["THM_LIMIT_THROTTLE"] == "0x5a"
    assert limits["recognized"]["thm_limit_throttle"]["value"] == "0x5a"
    assert limits["recognized"]["thm_limit_throttle"]["source_key"] == (
        "smbus_telem.THM_LIMIT_THROTTLE"
    )


def test_cooldown_gate_matches_issue127_power_temperature_rule() -> None:
    passing = runner.evaluate_cooldown(52.0, 61.5, observed_idle_aiclk_mhz=800.0)
    failing = runner.evaluate_cooldown(52.01, 61.5, observed_idle_aiclk_mhz=800.0)

    assert passing["verified"] is True
    assert passing["aiclk_gate"] is False
    assert passing["power_limit_w"] == 52.0
    assert passing["temperature_limit_c"] == 61.5
    assert failing["verified"] is False


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        (_trace(current=190), "H-TDC"),
        (_trace(temperature=88), "H-THM"),
        (_trace(power=145), "H-TDP"),
        (_trace(), "H-OTHER"),
    ],
    ids=["tdc", "thm", "tdp", "other"],
)
def test_analysis_classifies_each_synthetic_hypothesis(
    trace: dict[str, object], expected: str
) -> None:
    limits = {
        "tdc_limit": 200,
        "therm_trip_l1_limit_c": 90,
        "thm_shutdown_limit_c": 110,
        "tdp_limit_w": 150,
        "board_power_limit_w": 300,
        "asic_fmax_mhz": 1350,
        "unit_provenance": analysis.UNIT_PROVENANCE,
    }
    result = analysis.analyze_trace(
        trace,
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits=limits,
    )

    assert result["classification"] == expected
    assert result["causality"] == "correlation_is_not_causation"
    assert result["time_axis"]["column"] == "timestamp_monotonic_ns"
    assert result["plateau"]["coverage_complete"] is True


def test_analysis_marks_tdc_inconclusive_when_units_are_unknown() -> None:
    result = analysis.analyze_trace(
        _trace(current=190),
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits={
            "tdc_limit": 200,
            "therm_trip_l1_limit_c": 90,
            "thm_shutdown_limit_c": 110,
            "asic_fmax_mhz": 1350,
        },
    )

    proximity = result["statistics"]["signals"]["current_a"]["limit_proximity"]
    assert proximity["status"] == "not_evaluable"
    assert proximity["limit_value"] == 200.0
    assert result["onsets"]["current_tdc_threshold"]["status"] == "not_evaluable"
    assert result["hypotheses"]["H-TDC"]["classification"] == "inconclusive"
    assert result["hypotheses"]["H-TDC"]["criteria"] == {
        "plateau_proximity": None,
        "onset_order": None,
    }
    assert result["classification"] == "H-OTHER"

    raw_throttle = analysis.analyze_trace(
        _trace(temperature=88),
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits={"asic_fmax_mhz": 1350, "thm_limit_throttle": "0x5a"},
    )["hypotheses"]["H-THM"]["details"]["thm_limit_throttle"]
    assert raw_throttle["status"] == "not_evaluable"
    assert raw_throttle["limit_value"] == 90.0
    assert raw_throttle["limit_unit"] is None


def test_analysis_missing_tdp_and_thermal_limits_is_inconclusive() -> None:
    result = analysis.analyze_trace(
        _trace(power=145, temperature=88),
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits={"asic_fmax_mhz": 1350},
    )

    assert result["limits"] == {
        "tdc_limit": None,
        "therm_trip_l1_limit_c": None,
        "thm_shutdown_limit_c": None,
        "tdp_limit_w": None,
        "board_power_limit_w": None,
        "thm_limit_throttle": None,
        "asic_fmax_mhz": 1350.0,
    }
    assert result["statistics"]["signals"]["power_w"]["limit_proximity"] == {
        "status": "not_evaluable",
        "limit_key": "TDP_LIMIT_MAX",
        "limit_value": None,
        "limit_unit": "W",
        "reason": "signal or source-proven limit is unavailable",
    }
    assert result["statistics"]["signals"]["asic_temp_c"]["limit_proximity"][
        "status"
    ] == "not_evaluable"
    assert result["statistics"]["signals"]["asic_temp_c"]["limit_proximity"][
        "limit_value"
    ] is None
    assert result["onsets"]["power_tdp_threshold"]["status"] == "not_evaluable"
    assert result["onsets"]["temperature_thm_threshold"]["status"] == "not_evaluable"
    assert result["hypotheses"]["H-TDP"]["classification"] == "inconclusive"
    assert result["hypotheses"]["H-TDP"]["criteria"] == {
        "plateau_proximity": None,
        "onset_order": None,
    }
    assert result["hypotheses"]["H-THM"]["classification"] == "inconclusive"
    assert result["hypotheses"]["H-THM"]["criteria"]["plateau_proximity"] is None
    assert result["hypotheses"]["H-THM"]["criteria"]["onset_order"] is None
    assert result["hypotheses"]["H-THM"]["criteria"]["shutdown_bound"] is None
    assert result["classification"] == "H-OTHER"


def test_analysis_reports_time_weighted_statistics_and_percentiles() -> None:
    result = analysis.analyze_trace(_trace())
    aiclk = result["statistics"]["aiclk_mhz"]

    assert analysis.DEFAULT_PLATEAU_END_SECONDS == 285.0
    cli_args = analysis._parser().parse_args(["--trace", "trace.csv"])
    assert cli_args.plateau_end_seconds == 285.0
    assert result["plateau"]["start_seconds"] == 150.0
    assert result["plateau"]["end_seconds"] == 285.0
    assert aiclk["min"] == 1200.0
    assert aiclk["max"] == 1350.0
    assert aiclk["time_weighted_mean"] == pytest.approx(1255.5555555555557)
    assert aiclk["percentiles"]["p50"] == 1200.0
    assert aiclk["percentiles"]["p95"] == 1350.0


def _issue118_trace() -> dict[str, object]:
    samples = [
        _row(
            0,
            aiclk=800,
            power=133,
            board_power=285,
            current=50,
            temperature=81,
        )
    ]
    samples.extend(
        [
            _row(
                2,
                aiclk=1350,
                power=133,
                board_power=285,
                current=50,
                temperature=81,
            ),
            _row(
                3,
                aiclk=1200,
                power=133,
                board_power=285,
                current=50,
                temperature=81,
            ),
        ]
        + [
            _row(
                seconds,
                aiclk=1200,
                power=133,
                board_power=285,
                current=50,
                temperature=81,
            )
            for seconds in range(5, 303, 2)
        ]
    )
    return {
        "schema": runner.ISSUE121_Q2_TRACE_SCHEMA,
        "schema_version": 1,
        "columns": list(runner.EXTENDED_TRACE_COLUMNS),
        "run_start_monotonic_ns": 1_000_000_000,
        "samples": samples,
    }


def _source_limits() -> dict[str, object]:
    return {
        "tdc_limit": 200,
        "therm_trip_l1_limit_c": 90,
        "thm_shutdown_limit_c": 110,
        "tdp_limit_w": 150,
        "board_power_limit_w": 300,
        "asic_fmax_mhz": 1350,
        "unit_provenance": analysis.UNIT_PROVENANCE,
    }


def test_issue118_prediction_ignores_pre_start_idle_and_keeps_two_second_onset_resolution() -> None:
    result = analysis.analyze_trace(
        _issue118_trace(),
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits=_source_limits(),
    )

    assert result["classification"] == "H-OTHER"
    assert result["hypotheses"]["H-TDP"]["classification"] == "not_supported"
    assert result["hypotheses"]["H-THM"]["classification"] == "not_supported"
    assert result["onsets"]["aiclk_drop"]["timestamp_monotonic_ns"] == 3_000_000_000
    assert result["onsets"]["aiclk_drop"]["seconds_from_run_start"] == pytest.approx(2.0)
    assert result["onsets"]["aiclk_drop"]["seconds_from_run_start"] != 0.0
    assert result["raw_window"]["rows"][0]["seconds_from_run_start"] == 1.0
    assert len(result["raw_window"]["rows"]) == 6
    assert any("approximately 2 second" in item for item in result["limitations"])
    board = result["statistics"]["signals"]["board_power_w"]["limit_proximity"]
    assert board["limit_key"] == "BOARD_POWER_LIMIT"
    assert board["fraction"] == pytest.approx(0.05)


def test_multiple_classification_is_explicit_for_multiple_supported_known_hypotheses() -> None:
    result = analysis.analyze_trace(
        _trace(current=190, temperature=88, power=145),
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits=_source_limits(),
    )

    assert result["classification"] == "multiple"
    assert result["hypotheses"]["H-TDC"]["classification"] == "supported"
    assert result["hypotheses"]["H-THM"]["classification"] == "supported"
    assert result["hypotheses"]["H-TDP"]["classification"] == "supported"


def test_analysis_does_not_guess_aiclk_baseline_when_fmax_is_missing() -> None:
    result = analysis.analyze_trace(
        _issue118_trace(),
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits={"tdp_limit_w": 150, "therm_trip_l1_limit_c": 90},
    )

    assert result["aiclk_baseline"]["asic_fmax_mhz"] is None
    assert result["onsets"]["aiclk_drop"]["status"] == "not_evaluable"
    assert result["onsets"]["aiclk_drop"]["timestamp_monotonic_ns"] is None


def test_analysis_does_not_invent_missing_throttle_limit() -> None:
    result = analysis.analyze_trace(
        _trace(temperature=88),
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits={"asic_fmax_mhz": 1350},
    )

    comparison = result["hypotheses"]["H-THM"]["details"]["thm_limit_throttle"]
    assert comparison["status"] == "not_evaluable"
    assert comparison["knowledge_status"] == "unknown"
    assert comparison["limit_value"] is None
    assert result["hypotheses"]["H-THM"]["classification"] == "inconclusive"

    known = analysis.analyze_trace(
        _trace(temperature=88),
        plateau_start_seconds=150,
        plateau_end_seconds=285,
        limits={
            "asic_fmax_mhz": 1350,
            "thm_limit_throttle": 100,
            "unit_provenance": analysis.UNIT_PROVENANCE,
        },
    )
    assert known["hypotheses"]["H-THM"]["details"]["thm_limit_throttle"]["status"] == "pass"
