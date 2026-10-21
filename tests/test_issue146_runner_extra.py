"""Focused fail-closed Issue #146 runner checks."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

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


def test_issue146_plan_pin_rejects_revision_three_history(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["revision_history"][-1]["revision"] = 3
    revision_three = tmp_path / "revision-three-plan.json"
    revision_three.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(issue146.Issue146Error, match="immutable owner"):
        issue146.load_plan(revision_three)


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


def _diagnostic_run(
    tmp_path: Path,
    *,
    run_id: str,
    command_runner,
) -> dict[str, object]:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    return issue146.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id=run_id,
        options=issue146.Issue146RunOptions(
            command_runner=command_runner,
            environment=environment,
            capture_fn=_capture_factory("C018", "smoke"),
            trace_path=tmp_path / f"trace-{run_id}.csv",
            trace_interval_seconds=0.001,
        ),
    )


def _write_valid_wrapper_artifacts(
    tmp_path: Path,
    *,
    result: dict[str, object],
) -> None:
    time.sleep(0.005)
    wrapper_id = str(result["wrapper_run_id"])
    (tmp_path / f"runner-result-{wrapper_id}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    (tmp_path / str(result["groups_path"])).write_bytes(
        (strict_json_dumps(result["group_timings"], indent=2) + "\n").encode("utf-8")
    )
    (tmp_path / str(result["power_trace_path"])).write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-10-20T00:00:00+00:00,250,1350,70\n",
        encoding="utf-8",
    )


def test_issue146_wrapper_diagnostics_persist_on_success_and_redact_paths(
    tmp_path: Path,
) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    run_id = "issue146-diagnostics-success"
    wrapper_id = "wrapper-diagnostics-success"
    result = _device_result(plan, config, environment, "smoke", wrapper_id, run_id)
    local_path = str(tmp_path / "private-output" / "result.json")
    runtime_hostname = socket.gethostname()
    runtime_fqdn = socket.getfqdn()
    short_host = runtime_hostname.split(".", 1)[0]
    stdout = (
        f"issue146 result -> /out/runner-result-{wrapper_id}.json\n"
        f"host={runtime_hostname} fqdn={runtime_fqdn} short={short_host}\n"
        f"relative-host-path=relative/{short_host}.py\n"
        f"local={local_path} var=/var/lib/issue146-private/data "
        "link=/var/lib/issue146-link/trace opt=/opt/issue146-private/log\n"
        "safe-work=/work/tools/issue146_device.py safe-out=/out/result.json\n"
        "url=https://example.test/var/lib/remote\n"
        "ModuleNotFoundError: No module named 'scipy'\n"
        + "diagnostic line\n" * 25
    )
    stderr = (
        f"Traceback (most recent call): File \"{local_path}\"\n"
        f"host={runtime_hostname} /var/lib/issue146-private/error\n"
    )

    def command_runner(*_args: object, **_kwargs: object):
        _write_valid_wrapper_artifacts(tmp_path, result=result)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)

    record = _diagnostic_run(tmp_path, run_id=run_id, command_runner=command_runner)
    assert record["status"] == "ok"
    diagnostics = record["wrapper_diagnostics"]
    assert diagnostics["stdout_path"] == f"wrapper-stdout-{run_id}.txt"
    assert diagnostics["stderr_path"] == f"wrapper-stderr-{run_id}.txt"
    saved_stdout = (tmp_path / diagnostics["stdout_path"]).read_text(encoding="utf-8")
    saved_stderr = (tmp_path / diagnostics["stderr_path"]).read_text(encoding="utf-8")
    assert local_path not in saved_stdout
    assert local_path not in saved_stderr
    for token in {runtime_hostname, runtime_fqdn, short_host}:
        if len(token) >= 2:
            assert token not in saved_stdout
            assert token not in saved_stderr
    for path in (
        "/var/lib/issue146-private/data",
        "/var/lib/issue146-link/trace",
        "/opt/issue146-private/log",
    ):
        assert path not in saved_stdout
    assert "/work/tools/issue146_device.py" in saved_stdout
    assert "/out/result.json" in saved_stdout
    assert "https://example.test/var/lib/remote" in saved_stdout
    assert "/out/runner-result-wrapper-diagnostics-success.json" in saved_stdout
    assert "ModuleNotFoundError: No module named 'scipy'" in saved_stdout
    assert len(diagnostics["stdout_tail"]) <= 20
    assert str(tmp_path) not in json.dumps(record)


def test_issue146_diagnostic_redaction_rejects_safe_prefix_lookalikes() -> None:
    text = "/outbox/private/file /worktree/private/file /dev/tenstorrent-extra/private/file"
    redacted = issue146._redact_diagnostic_text(text, output_dir=Path("/"))
    for path in (
        "/outbox/private/file",
        "/worktree/private/file",
        "/dev/tenstorrent-extra/private/file",
    ):
        assert path not in redacted


def test_issue146_diagnostic_prefix_lookalikes_redact_whole_path() -> None:
    repository_root = issue146._REPOSITORY_ROOT.as_posix()
    text = (
        "/homeowner/private/file "
        f"{repository_root}owner/private/file"
    )
    redacted = issue146._redact_diagnostic_text(text, output_dir=Path("/"))
    assert "/homeowner/private/file" not in redacted
    assert "owner/private/file" not in redacted


def test_issue146_record_sanitizer_preserves_structured_host_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"hostname": 0, "fqdn": 0}

    def fake_hostname() -> str:
        calls["hostname"] += 1
        return "smoke"

    def fake_fqdn() -> str:
        calls["fqdn"] += 1
        return "tenstorrent"

    monkeypatch.setattr(issue146.socket, "gethostname", fake_hostname)
    monkeypatch.setattr(issue146.socket, "getfqdn", fake_fqdn)
    record = {
        "status": "ok",
        "run": {"mode": "smoke"},
        "environment": {"device_node": "/dev/tenstorrent/1", "mode": "smoke"},
        "runner_result": {
            "status": "ok", "mode": "smoke",
            "environment": {"device_node": "/dev/tenstorrent/1"},
        },
        "failure": {"message": "host smoke tenstorrent /var/lib/diagnostic"},
        "wrapper_diagnostics": {
            "stdout_path": "wrapper-stdout-structured.txt",
            "stderr_path": "wrapper-stderr-structured.txt",
            "stdout_tail": ["host smoke tenstorrent /var/lib/diagnostic"],
            "stderr_tail": [],
        },
    }
    issue146._persist_record(tmp_path, "structured-host", record)
    persisted = json.loads(
        (tmp_path / "run-structured-host.json").read_text(encoding="utf-8")
    )
    for value in (record, persisted):
        assert value["run"]["mode"] == "smoke"
        assert value["environment"]["device_node"] == "/dev/tenstorrent/1"
        assert value["runner_result"]["mode"] == "smoke"
        assert value["runner_result"]["environment"]["device_node"] == "/dev/tenstorrent/1"
        assert "smoke" not in value["failure"]["message"]
        assert "tenstorrent" not in value["failure"]["message"]
        assert "/var/lib/diagnostic" not in value["failure"]["message"]
        assert "smoke" not in json.dumps(value["wrapper_diagnostics"])
        assert "tenstorrent" not in json.dumps(value["wrapper_diagnostics"])
    assert calls == {"hostname": 1, "fqdn": 1}


def test_issue146_diagnostic_redacts_local_file_urls() -> None:
    text = (
        "remote=https://example.test/var/lib/remote "
        "file:///homeowner/private.db "
        "file://localhost/var/lib/private.db "
        "FILE://localhost/var/lib/private-upper.db "
        "sqlite:////opt/private.db"
    )
    redacted = issue146._redact_diagnostic_text(text, output_dir=Path("/"))
    assert "https://example.test/var/lib/remote" in redacted
    assert "file:///homeowner/private.db" not in redacted
    assert "file://localhost/var/lib/private.db" not in redacted
    assert "FILE://localhost/var/lib/private-upper.db" not in redacted
    assert "sqlite:////opt/private.db" not in redacted
    assert "/homeowner/private.db" not in redacted
    assert "/var/lib/private.db" not in redacted
    assert "/var/lib/private-upper.db" not in redacted
    assert "/opt/private.db" not in redacted


def test_issue146_diagnostic_redaction_root_prefix_keeps_safe_text() -> None:
    prefixes = issue146._diagnostic_path_prefixes(Path("/"))
    assert "" not in prefixes
    text = (
        "keep /work/tools/issue146.py /out/result.json /dev/tenstorrent/1 "
        "https://example.test/remote relative/tools/issue146.py "
        "relative/tmp/source.py relative/home/source.py"
    )
    redacted = issue146._redact_diagnostic_text(text, output_dir=Path("/"))
    assert redacted == text


def test_issue146_wrapper_diagnostics_keep_nonzero_failure_and_tails(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    run_id = "issue146-diagnostics-nonzero"
    wrapper_id = "wrapper-diagnostics-nonzero"
    result = _device_result(plan, config, environment, "smoke", wrapper_id, run_id)

    def command_runner(*_args: object, **_kwargs: object):
        _write_valid_wrapper_artifacts(tmp_path, result=result)
        return SimpleNamespace(
            returncode=17,
            stdout=f"issue146 result -> /out/runner-result-{wrapper_id}.json\n",
            stderr="wrapper failed after writing its result\n",
        )

    record = _diagnostic_run(tmp_path, run_id=run_id, command_runner=command_runner)
    assert record["status"] == "failed"
    assert record["failure"]["returncode"] == 17
    assert record["failure"]["wrapper_diagnostics"]["stderr_tail"] == [
        "wrapper failed after writing its result"
    ]
    assert (tmp_path / f"wrapper-stdout-{run_id}.txt").is_file()
    assert (tmp_path / f"wrapper-stderr-{run_id}.txt").is_file()


def test_issue146_wrapper_diagnostics_keep_missing_result_failure(tmp_path: Path) -> None:
    run_id = "issue146-diagnostics-missing"

    def command_runner(*_args: object, **_kwargs: object):
        return SimpleNamespace(returncode=0, stdout="wrapper completed\n", stderr="no result\n")

    record = _diagnostic_run(tmp_path, run_id=run_id, command_runner=command_runner)
    assert record["status"] == "failed"
    assert "runner result artifact" in record["failure"]["message"]
    assert record["failure"]["wrapper_diagnostics"]["stdout_path"] == (
        f"wrapper-stdout-{run_id}.txt"
    )


def test_issue146_wrapper_diagnostics_keep_timeout_bytes_and_exception(tmp_path: Path) -> None:
    run_id = "issue146-diagnostics-timeout"
    private_path = str(tmp_path / "timeout-output" / "traceback.py")

    def command_runner(*_args: object, **_kwargs: object):
        raise subprocess.TimeoutExpired(
            ["wrapper"], 90, output=f"partial {private_path}\n".encode(),
            stderr=b"Traceback: ModuleNotFoundError: No module named 'scipy'\n",
        )

    record = _diagnostic_run(tmp_path, run_id=run_id, command_runner=command_runner)
    assert record["status"] == "failed"
    assert record["failure"]["exception_type"] == "TimeoutExpired"
    assert "TimeoutExpired" in record["failure"]["message"]
    stdout = (tmp_path / f"wrapper-stdout-{run_id}.txt").read_text(encoding="utf-8")
    stderr = (tmp_path / f"wrapper-stderr-{run_id}.txt").read_text(encoding="utf-8")
    assert private_path not in stdout
    assert "ModuleNotFoundError: No module named 'scipy'" in stderr
    assert record["failure"]["wrapper_diagnostics"]["stderr_tail"]


def test_issue146_wrapper_diagnostics_keep_oserror_exception(tmp_path: Path) -> None:
    run_id = "issue146-diagnostics-oserror"
    runtime_hostname = socket.gethostname()
    private_paths = "/var/lib/issue146-private/error /opt/issue146-private/error"

    def command_runner(*_args: object, **_kwargs: object):
        raise OSError(
            f"wrapper executable is unavailable on {runtime_hostname}: {private_paths}"
        )

    record = _diagnostic_run(tmp_path, run_id=run_id, command_runner=command_runner)
    assert record["status"] == "failed"
    assert record["failure"]["exception_type"] == "OSError"
    assert "wrapper executable is unavailable" in record["failure"]["message"]
    assert runtime_hostname not in record["failure"]["message"]
    for path in private_paths.split():
        assert path not in record["failure"]["message"]
    persisted = json.loads(
        (tmp_path / f"run-{run_id}.json").read_text(encoding="utf-8")
    )
    assert runtime_hostname not in json.dumps(persisted)
    for path in private_paths.split():
        assert path not in json.dumps(persisted)
    assert (tmp_path / f"wrapper-stdout-{run_id}.txt").read_bytes() == b""
    assert (tmp_path / f"wrapper-stderr-{run_id}.txt").read_bytes() == b""


def test_issue146_diagnostics_collision_is_rejected_before_command(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    run_id = "issue146-diagnostics-collision"
    (tmp_path / f"wrapper-stdout-{run_id}.txt").write_text("old", encoding="utf-8")
    options = issue146.Issue146RunOptions(
        command_runner=lambda *_a, **_k: None,
        capture_fn=_capture_factory("C018", "smoke"),
    )
    with pytest.raises(issue146.Issue146Error, match="collision"):
        issue146.run_one(
            plan=plan,
            configuration=config,
            mode="smoke",
            output_dir=tmp_path,
            run_id=run_id,
            options=options,
        )


def test_issue146_output_collision_is_rejected_before_command(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    (tmp_path / "run-issue146-collision.json").write_text("old", encoding="utf-8")
    options = issue146.Issue146RunOptions(command_runner=lambda *_a, **_k: None)
    with pytest.raises(issue146.Issue146Error, match="collision"):
        issue146.run_one(
            plan=plan, configuration=config, mode="smoke", output_dir=tmp_path,
            run_id="issue146-collision", options=options,
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


def test_issue146_main_loads_preceding_snapshot_for_later_long_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preceding = _capture_factory("C108", "long")(stage="pre_run")
    snapshot_path = tmp_path / "preceding-pre-run.json"
    snapshot_path.write_text(json.dumps(preceding), encoding="utf-8")
    captured: list[dict[str, object]] = []
    command_calls: list[object] = []

    def fake_command(*args: object, **kwargs: object) -> None:
        command_calls.append((args, kwargs))

    def fake_run_one(**kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(issue146, "run_one", fake_run_one)
    monkeypatch.setattr(issue146, "capture_readonly_snapshot", lambda **_kwargs: preceding)
    monkeypatch.setattr(issue146.issue137, "_run_issue137_workload", fake_command)

    exit_code = issue146.main(
        [
            "--plan", str(PLAN), "--mode", "long", "--configuration", "C108",
            "--run-id", "issue146-cli-long-2", "--run-number", "2",
            "--approve-long-run", "--out-dir", str(tmp_path), "--execute",
            "--preceding-pre-snapshot", str(snapshot_path),
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    options = captured[0]["options"]
    assert isinstance(options, issue146.Issue146RunOptions)
    assert options.preceding_pre_snapshot == preceding
    assert options.capture_fn is issue146.capture_readonly_snapshot
    assert options.command_runner is fake_command
    assert command_calls == []


@pytest.mark.parametrize("case", ("missing", "wrong-stage", "wrong-board", "malformed", "unsafe"))
def test_issue146_main_rejects_invalid_preceding_snapshot_before_run_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    preceding = _capture_factory("C108", "long")(stage="pre_run")
    snapshot_path = tmp_path / "preceding-pre-run.json"
    if case == "wrong-stage":
        preceding["stage"] = "post_run"
        snapshot_path.write_text(json.dumps(preceding), encoding="utf-8")
    elif case == "wrong-board":
        preceding["snapshot"]["device_info"][0]["board_info"]["bus_id"] = "0000:0a:00.0"
        snapshot_path.write_text(json.dumps(preceding), encoding="utf-8")
    elif case == "malformed":
        snapshot_path.write_text("{not-json", encoding="utf-8")
    elif case == "unsafe":
        snapshot_path = Path("/etc/issue146-preceding-pre-run.json")
    captured: list[dict[str, object]] = []
    capture_calls: list[object] = []
    command_calls: list[object] = []

    def fake_run_one(**kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return {"status": "ok"}

    def fake_capture(**kwargs: object) -> dict[str, object]:
        capture_calls.append(kwargs)
        return preceding

    def fake_command(*args: object, **kwargs: object) -> None:
        command_calls.append((args, kwargs))

    monkeypatch.setattr(issue146, "run_one", fake_run_one)
    monkeypatch.setattr(issue146, "capture_readonly_snapshot", fake_capture)
    monkeypatch.setattr(issue146.issue137, "_run_issue137_workload", fake_command)

    argv = [
        "--plan", str(PLAN), "--mode", "long", "--configuration", "C108",
        "--run-id", "issue146-cli-long-invalid", "--run-number", "2",
        "--approve-long-run", "--out-dir", str(tmp_path), "--execute",
    ]
    if case != "missing":
        argv.extend(("--preceding-pre-snapshot", str(snapshot_path)))

    assert issue146.main(argv) == 2
    assert captured == []
    assert capture_calls == []
    assert command_calls == []


def test_issue146_main_plan_mode_remains_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(issue146, "build_planning_record", lambda _plan: {"status": "planned"})
    assert issue146.main(["--plan", str(PLAN), "--mode", "plan"]) == 0


def test_issue146_plan_cli_accepts_revision_six_without_board(capsys: pytest.CaptureFixture[str]) -> None:
    assert issue146.main(["--plan", str(PLAN), "--mode", "plan"]) == 0
    planning = json.loads(capsys.readouterr().out)
    assert planning["plan_revision"] == 6
    assert planning["plan_sha256"] == issue146.AUTHORITATIVE_PLAN_CANONICAL_SHA256
