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
IMMUTABLE_RUN_ROOT = ROOT / "docs/measurements/issue121-q2/runs"


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
    assert result["hypotheses"]["H-THM"]["classification"] == "supported"

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
