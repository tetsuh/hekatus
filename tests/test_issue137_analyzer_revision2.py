"""Revision-2 Issue #137 analyzer acceptance coverage with offline runner seams."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import analyze_issue137 as analyzer
from tools import issue137 as runner

PLAN_PATH = Path("docs/measurements/2026-10-18-host-newton-schulz-issue137-plan.json")


def _plan() -> dict[str, object]:
    return runner.load_plan(PLAN_PATH)


def _environment(plan: dict[str, object]) -> dict[str, object]:
    scope = plan["measurement_scope"]
    assert isinstance(scope, dict)
    return {
        "board": {
            "serial": scope["board_serial"],
            "board_id": scope["board_serial"],
            "bus_id": scope["pci_bus_id"],
        },
        "firmware": {"bundle": "19.6.0.0"},
        "kernel": "fixture-kernel",
        "kmd_version": "fixture-driver",
        "image": "fixture-image@sha256:" + "a" * 64,
        "harness_commit": "b" * 40,
        "device_node": scope["device_node"],
        "harness_dirty": False,
        "harness_pushed": True,
        "telemetry_sampler": {"mode": "default", "interval_seconds": 2.0},
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
    }


def _snapshot(
    plan: dict[str, object], *, values: dict[str, float]
) -> dict[str, object]:
    scope = plan["measurement_scope"]
    assert isinstance(scope, dict)
    return {
        "timestamp_utc": "2026-10-20T00:00:00+00:00",
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
        "device_info": [
            {
                "board_info": {
                    "board_id": scope["board_serial"],
                    "bus_id": scope["pci_bus_id"],
                },
                "firmwares": {"fw_bundle_version": "19.6.0.0"},
                "limits": {
                    "board_power_limit": "300",
                    "tdc_limit": "200",
                    "tdp_limit": "150",
                    "therm_trip_l1_limit": "90",
                    "asic_fmax": "1350",
                },
                "smbus_telem": {
                    "THERM_TRIP_COUNT": "0x0",
                    "GDDR_0_1_TEMP": "0x36363234",
                    "GDDR_2_3_TEMP": "0x32343432",
                    "GDDR_4_5_TEMP": "0x30323638",
                    "GDDR_6_7_TEMP": "0x34342e30",
                },
                "telemetry": {
                    "power": str(values["power_w"]),
                    "aiclk": str(values["aiclk_mhz"]),
                    "asic_temperature": str(values["asic_temp_c"]),
                    "current": str(values["current_a"]),
                    "voltage": "0.8",
                    "board_power": str(values["board_power_w"]),
                    "fan_speed": "1000",
                    "heartbeat": "1",
                },
            }
        ],
    }


def _runner_records(tmp_path: Path) -> list[dict[str, object]]:
    """Produce canonical persisted records through the mocked runner seam."""
    plan = _plan()
    plan_hash = runner._plan_hash(plan)
    environment = _environment(plan)
    order = runner.long_run_order(plan)
    records: list[dict[str, object]] = []
    for run_number, configuration_id in enumerate(order, start=1):
        configuration = runner.configuration(plan, configuration_id)
        if configuration_id == "C018":
            values = {
                "aiclk_mhz": 1350.0,
                "board_power_w": 250.0,
                "power_w": 100.0,
                "current_a": 100.0,
                "asic_temp_c": 70.0,
            }
        elif configuration_id == "C081":
            values = {
                "aiclk_mhz": 1100.0 if run_number == 4 else 1150.0,
                "board_power_w": 300.0,
                "power_w": 100.0,
                "current_a": 100.0,
                "asic_temp_c": 70.0,
            }
        elif configuration_id == "C108":
            values = {
                "aiclk_mhz": 900.0,
                "board_power_w": 300.0,
                "power_w": 100.0,
                "current_a": 100.0,
                "asic_temp_c": 70.0,
            }
        else:
            values = {
                "aiclk_mhz": 1000.0,
                "board_power_w": 250.0,
                "power_w": 100.0,
                "current_a": 100.0,
                "asic_temp_c": 70.0,
            }
        pre = _snapshot(plan, values=values)
        result_path = tmp_path / f"runner-result-{run_number}.json"
        first = 1_000
        target_end = first + 120_000_000_000
        result = {
            "record_schema": runner.ISSUE137_DEVICE_RESULT_SCHEMA,
            "status": "ok",
            "run_id": f"wrapper-run-{run_number}",
            "mode": "long",
            "load_mode": runner.ISSUE137_LOAD_MODE,
            "target_seconds": 120.0,
            "guard_seconds": 170.0,
            "overrun_limit_seconds": 10.0,
            "launch_count": 3,
            "first_launch_start_monotonic_ns": first,
            "workload_end_monotonic_ns": target_end,
            "last_launch_end_monotonic_ns": target_end,
            "overrun_seconds": 0.0,
            "core_plan": configuration.core_plan.as_record(),
            "plan_sha256": plan_hash,
            "environment": environment,
        }

        def capture(
            *,
            stage: str,
            _first: int = first,
            _target_end: int = target_end,
            _pre: dict[str, object] = pre,
        ) -> dict[str, object]:
            timestamp = {"pre_run": _first, "in_run": 60_000_000_001, "post_run": _target_end}[stage]
            capture = {
                "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
                "stage": stage,
                "status": "ok",
                "returncode": 0,
                "captured_at_utc": "2026-10-20T00:00:00+00:00",
                "timestamp_monotonic_ns": timestamp,
                "snapshot_elapsed_ns": 1,
                "snapshot": _pre,
            }
            return capture

        def command_runner(
            *_args: object,
            _result_path: Path = result_path,
            _result: dict[str, object] = result,
            **_kwargs: object,
        ) -> SimpleNamespace:
            _result_path.write_text(json.dumps(_result), encoding="utf-8")
            time.sleep(0.01)
            return SimpleNamespace(
                returncode=0,
                stdout=f"results -> {_result_path.name}\n",
            )

        record = runner.run_one(
            plan=plan,
            configuration=configuration,
            mode="long",
            output_dir=tmp_path,
            run_id=f"host-run-{run_number}",
            run_number=run_number,
            approve_long_run=True,
            command_runner=command_runner,
            environment=environment,
            preceding_pre_snapshot=(
                {
                    "stage": "pre_run",
                    "status": "ok",
                    "returncode": 0,
                    "snapshot": pre,
                }
                if run_number > 1
                else None
            ),
            cooldown_readings=([{"power_w": 50.0, "asic_temp_c": 50.0}] if run_number > 1 else []),
            cooldown_elapsed_seconds=1.0,
            capture_fn=capture,
            trace_path=tmp_path / f"power-{run_number}.csv",
            trace_interval_seconds=0.001,
        )
        assert record["status"] == "ok"
        persisted_path = tmp_path / f"run-host-run-{run_number}.json"
        persisted_path.write_text(json.dumps(record), encoding="utf-8")
        records.append(json.loads(persisted_path.read_text()))
    return records


def _analyze_persisted(records: list[dict[str, object]], tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "analysis.json"
    argv = ["--plan", str(PLAN_PATH), "--output", str(output)]
    for index, record in enumerate(records):
        path = tmp_path / f"analysis-record-{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        argv.extend(("--record", str(path)))
    assert analyzer.main(argv) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_revision2_persisted_runner_records_are_accepted_by_cli(tmp_path: Path) -> None:
    records = _runner_records(tmp_path)
    output = tmp_path / "analysis.json"
    argv = ["--output", str(output)]
    for record in records:
        path = tmp_path / f"run-{record['run']['run_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        argv.extend(("--record", str(path)))
    assert analyzer.main(argv) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["classification"] == "H-BPL"
    assert len(result["verdict_runs"]) == 6
    assert result["plan"]["long_run_order"] == runner.long_run_order(_plan())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record["runner_result"].update(launch_count=0),
        lambda record: record["runner_result"].update(target_seconds=119.0),
        lambda record: record["runner_result"].update(workload_end_monotonic_ns=1),
        lambda record: record["runner_result"].update(overrun_seconds=11.0),
        lambda record: record["runner_result"].update(guard_seconds=70.0),
    ],
)
def test_revision2_bad_duration_binding_is_not_evaluable(tmp_path: Path, mutation) -> None:
    records = _runner_records(tmp_path)
    mutation(records[0])
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"
    assert result["verdict_evidence_complete"] is False


def test_revision2_configuration_environment_and_identity_binding_fail_closed(tmp_path: Path) -> None:
    records = _runner_records(tmp_path)
    records[0]["configuration"]["batch"] += 1
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"

    identity_path = tmp_path / "identity"
    records = _runner_records(identity_path)
    records[0]["environment"]["device_node"] = "/dev/other"
    result = _analyze_persisted(records, identity_path)
    assert result["classification"] == "not_evaluable"


def test_revision2_pre_snapshot_limits_are_source_unit_checked(tmp_path: Path) -> None:
    records = _runner_records(tmp_path)
    pre_path = tmp_path / records[0]["pre_run_snapshot"]["path"]
    capture = json.loads(pre_path.read_text(encoding="utf-8"))
    capture["snapshot"]["device_info"][0]["limits"]["board_power_limit"] = {
        "value": 300,
        "unit": "mW",
    }
    pre_path.write_text(json.dumps(capture), encoding="utf-8")
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"
    assert any("board_power_limit_w" in field for field in result["missing_fields"])


def test_revision2_gddr_without_bound_provenance_is_raw_exploratory(tmp_path: Path) -> None:
    records = _runner_records(tmp_path)
    for record in records:
        record["provenance"].pop("unit_provenance")
        record["trace"].pop("unit_provenance")
    record_paths: list[Path] = []
    for record in records:
        path = tmp_path / f"persisted-{record['run']['run_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        record_paths.append(path)
    result = analyzer.analyze_runs(record_paths, plan=_plan())
    assert result["classification"] == "H-BPL"
    run = result["runs"][0]
    assert run["statistics"]["gddr_channels"] == {}
    assert run["exploratory"]["gddr"]["status"] == "not_evaluable"
    assert len(run["exploratory"]["gddr"]["raw_fields"]) == 4


def test_revision2_mismatched_gddr_versions_do_not_block_non_gddr_verdict(
    tmp_path: Path,
) -> None:
    records = _runner_records(tmp_path)
    for record in records:
        record["runner_result"]["environment"]["host_sw_vers"] = {
            "tt_smi": "unknown",
            "tt_umd": "unknown",
        }
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "H-BPL"
    assert result["runs"][0]["statistics"]["gddr_channels"] == {}
    assert result["runs"][0]["exploratory"]["gddr"]["status"] == "not_evaluable"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("image_digest", "sha256:" + "c" * 64),
        ("image_digest", "not-a-digest"),
        ("image", "fixture-image@sha256:" + "c" * 64),
        ("kernel", "different-kernel"),
        ("harness_commit", "d" * 40),
    ),
)
def test_revision2_runner_environment_must_match_canonical_record(
    tmp_path: Path, field: str, value: str
) -> None:
    records = _runner_records(tmp_path)
    records[0]["runner_result"]["environment"][field] = value
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"
    assert any("environment" in missing for missing in result["missing_fields"])


def test_revision2_trace_byte_count_mismatch_is_not_evaluable(tmp_path: Path) -> None:
    records = _runner_records(tmp_path)
    records[0]["trace"]["byte_count"] += 1
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"
    assert any("trace_byte_count" in field for field in result["missing_fields"])


def test_revision2_trace_byte_tampering_is_not_evaluable(tmp_path: Path) -> None:
    records = _runner_records(tmp_path)
    trace_path = tmp_path / records[0]["trace"]["path"]
    raw = trace_path.read_bytes()
    assert b"250.0" in raw
    trace_path.write_bytes(raw.replace(b"250.0", b"251.0", 1))
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"
    assert any("trace_sha256" in field for field in result["missing_fields"])


@pytest.mark.parametrize("field", ["sha256", "byte_count"])
def test_revision2_trace_hash_metadata_is_required(tmp_path: Path, field: str) -> None:
    records = _runner_records(tmp_path)
    records[0]["trace"].pop(field)
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"
    assert any("trace_" in missing for missing in result["missing_fields"])


def test_revision2_does_not_accept_legacy_calibration_or_repeat_inputs(tmp_path: Path) -> None:
    records = _runner_records(tmp_path)
    records[0]["run"]["repeat_count"] = 4
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.pop("record_revision"),
        lambda record: record.__setitem__("record_revision", 1),
        lambda record: record["run"].__setitem__("mode", "smoke"),
        lambda record: record.__setitem__("status", "failed"),
        lambda record: record.pop("environment"),
    ],
)
def test_revision2_public_verdict_inputs_require_canonical_identity_and_provenance(
    tmp_path: Path, mutation
) -> None:
    records = _runner_records(tmp_path)
    mutation(records[0])
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "not_evaluable"
    assert result["verdict_evidence_complete"] is False


def test_revision2_missing_gddr_provenance_never_decodes_numeric_channels(tmp_path: Path) -> None:
    records = _runner_records(tmp_path)
    records[0]["provenance"].pop("unit_provenance")
    records[0]["trace"].pop("unit_provenance")
    result = _analyze_persisted(records, tmp_path)
    assert result["classification"] == "H-BPL"
    run = result["runs"][0]
    assert run["statistics"]["gddr_channels"] == {}
    assert run["exploratory"]["gddr"]["status"] == "not_evaluable"
    assert "time_weighted_mean" not in json.dumps(run["exploratory"]["gddr"])
