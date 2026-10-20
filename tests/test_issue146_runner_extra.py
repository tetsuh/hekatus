"""Focused fail-closed Issue #146 runner checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enodia.strict_json import dumps as strict_json_dumps
from tests.test_issue146_runner import PLAN, _capture_factory, _device_result, _environment
from tools import issue146


def test_issue146_plan_pin_rejects_mutated_owner_bytes(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["measurement_scope"]["order"] = ["C018", "C081", "C108", "C108", "C081", "C018"]
    mutated = tmp_path / "mutated-plan.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(issue146.Issue146Error, match="immutable owner"):
        issue146.load_plan(mutated)


def test_issue146_bad_pre_snapshot_words_fail_before_command(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    called = False

    def capture(*, stage: str):
        value = _capture_factory("C018", "smoke")(stage=stage)
        if stage == "pre_run":
            value["snapshot"]["device_info"][0]["smbus_telem"]["BOARD_POWER_LIMIT"] = "0x12b"
        return value

    def command_runner(*_args: object, **_kwargs: object):
        nonlocal called
        called = True

    record = issue146.run_one(
        plan=plan, configuration=config, mode="smoke", output_dir=tmp_path,
        run_id="issue146-bad-pre", options=issue146.Issue146RunOptions(
            command_runner=command_runner, environment=environment, capture_fn=capture,
        ),
    )
    assert record["status"] == "not_evaluable"
    assert record["classification"] == "not_evaluable"
    assert record["reason"] == "configuration"
    assert record["failure"]["kind"] == "configuration_mismatch"
    assert record["pre_run_snapshot"]["byte_count"] > 0
    assert called is False


def test_issue146_missing_pre_snapshot_telemetry_is_not_configuration_mismatch(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    called = False

    def capture(*, stage: str):
        value = _capture_factory("C018", "smoke")(stage=stage)
        if stage == "pre_run":
            value["snapshot"]["device_info"][0].pop("smbus_telem")
        return value

    def command_runner(*_args: object, **_kwargs: object):
        nonlocal called
        called = True

    record = issue146.run_one(
        plan=plan, configuration=config, mode="smoke", output_dir=tmp_path,
        run_id="issue146-missing-pre-telemetry", options=issue146.Issue146RunOptions(
            command_runner=command_runner, environment=environment, capture_fn=capture,
        ),
    )
    assert record["status"] == "failed"
    assert record["failure"]["kind"] == "pre_run_snapshot_failure"
    assert record["reason"] == "telemetry"
    assert record["pre_run_snapshot"]["byte_count"] > 0
    assert called is False


def test_issue146_cooldown_fails_closed_at_plan_cap_without_command(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C108")
    environment = _environment(config)
    preceding = _capture_factory("C108", "long")(stage="pre_run")
    called = False

    def command_runner(*_args: object, **_kwargs: object):
        nonlocal called
        called = True

    record = issue146.run_one(
        plan=plan, configuration=config, mode="long", output_dir=tmp_path,
        run_id="issue146-cooldown-fail", options=issue146.Issue146RunOptions(
            run_number=2, approve_long_run=True, command_runner=command_runner,
            environment=environment, preceding_pre_snapshot=preceding,
            cooldown_readings=({"power_w": 1000, "asic_temp_c": 1000},),
            cooldown_elapsed_seconds=600,
        ),
    )
    assert record["status"] == "failed"
    assert record["failure"]["kind"] == "cooldown_gate_failure"
    assert called is False


@pytest.mark.parametrize("tamper", ("missing-group", "tampered-group", "stale-group", "missing-csv", "semantic-id"))
def test_issue146_runner_rejects_unbound_group_or_companion_artifacts(
    tmp_path: Path, tamper: str
) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    run_id = f"issue146-smoke-artifact-{tamper}"
    wrapper_id = f"wrapper-{tamper}"
    result = _device_result(plan, config, environment, "smoke", wrapper_id, run_id)
    case_dir = tmp_path / tamper
    case_dir.mkdir()
    if tamper == "stale-group":
        (case_dir / result["groups_path"]).write_bytes(
            (strict_json_dumps(result["group_timings"], indent=2) + "\n").encode("utf-8")
        )

    def command_runner(*_args: object, **_kwargs: object):
        name = f"runner-result-{wrapper_id}.json"
        (case_dir / name).write_text(json.dumps(result), encoding="utf-8")
        if tamper not in {"missing-group", "stale-group"}:
            groups = result["group_timings"]
            if tamper == "tampered-group":
                groups = [{"start_monotonic_ns": 0, "end_monotonic_ns": 1}]
            (case_dir / result["groups_path"]).write_bytes(
                (strict_json_dumps(groups, indent=2) + "\n").encode("utf-8")
            )
        if tamper != "missing-csv":
            (case_dir / result["power_trace_path"]).write_text(
                "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
                "2026-10-20T00:00:00+00:00,250,1350,70\n",
                encoding="utf-8",
            )
        return type("Completed", (), {"returncode": 0, "stdout": f"issue146 result -> /out/{name}\n"})()

    if tamper == "semantic-id":
        result["requested_run_id"] = "other-requested-run"
    record = issue146.run_one(
        plan=plan, configuration=config, mode="smoke", output_dir=case_dir,
        run_id=run_id,
        options=issue146.Issue146RunOptions(
            command_runner=command_runner, environment=environment,
            capture_fn=_capture_factory("C018", "smoke"),
        ),
    )
    assert record["status"] == "failed"
    assert record["verdict_evidence"] is False


def test_issue146_snapshot_serial_binding_allows_alias_but_rejects_conflict() -> None:
    capture = _capture_factory("C018", "smoke")(stage="pre_run")
    issue146.validate_capture(capture, stage="pre_run", board_id=issue146.BOARD_ID, bus_id=issue146.BUS_ID)
    del capture["snapshot"]["device_info"][0]["board_info"]["serial"]
    issue146.validate_capture(capture, stage="pre_run", board_id=issue146.BOARD_ID, bus_id=issue146.BUS_ID)
    conflicting = _capture_factory("C018", "smoke")(stage="pre_run")
    conflicting["snapshot"]["device_info"][0]["board_info"]["serial"] = "other-board"
    with pytest.raises(issue146.Issue146Error, match="serial"):
        issue146.validate_capture(
            conflicting, stage="pre_run", board_id=issue146.BOARD_ID, bus_id=issue146.BUS_ID
        )


def test_issue146_output_collision_is_rejected_before_command(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    (tmp_path / "run-issue146-collision.json").write_text("old", encoding="utf-8")
    with pytest.raises(issue146.Issue146Error, match="collision"):
        issue146.run_one(
            plan=plan, configuration=config, mode="smoke", output_dir=tmp_path,
            run_id="issue146-collision",
            options=issue146.Issue146RunOptions(command_runner=lambda *_a, **_k: None),
        )


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    (("failed", 2), ("not_evaluable", 2), ("ok", 0)),
)
def test_issue146_main_returns_nonzero_for_nonaccepted_run_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, expected_exit: int,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_capture(*, stage: str) -> dict[str, object]:
        return {"stage": stage}

    def fake_run_one(**kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return {"status": status}

    monkeypatch.setattr(issue146, "capture_readonly_snapshot", fake_capture)
    monkeypatch.setattr(issue146, "run_one", fake_run_one)

    exit_code = issue146.main(
        [
            "--plan", str(PLAN), "--mode", "smoke", "--configuration", "C018",
            "--run-id", "issue146-cli-status", "--out-dir", str(tmp_path), "--execute",
        ]
    )

    assert exit_code == expected_exit
    assert len(captured) == 1
    assert captured[0]["options"].capture_fn is fake_capture


def test_issue146_main_plan_mode_remains_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(issue146, "build_planning_record", lambda _plan: {"status": "planned"})
    assert issue146.main(["--plan", str(PLAN), "--mode", "plan"]) == 0
