"""Focused fail-closed Issue #146 runner checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_issue146_runner import PLAN, _capture_factory, _environment
from tools import issue146


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
    assert record["status"] == "failed"
    assert record["failure"]["kind"] == "pre_run_snapshot_failure"
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
