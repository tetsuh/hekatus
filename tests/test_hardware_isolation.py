"""Regression tests for the board-free pytest hardware boundary."""

from __future__ import annotations

import builtins
import io
import os
import pathlib
import subprocess
import sys

import pytest
from hardware_isolation import HardwareAccessViolation

_BLOCKED_COMMANDS = (
    ("tt-smi", "-s"),
    ("/usr/bin/tt-smi", "-s"),
    ("docker", "run", "image"),
    ("/usr/bin/docker", "run", "image"),
    ("run_in_container.sh", "--", "--iters", "1"),
    ("/work/enodia/tt/bench/run_in_container.sh", "--", "--iters", "1"),
    ("tt-topology", "--json"),
    ("/usr/bin/tt-flash", "--help"),
    ("nohup", "tt-smi", "-s"),
)
_PERMISSION_MATRIX = tuple(
    (scope, route, scope != "hardware")
    for scope in ("default", "wrapper", "hardware")
    for route in ("direct-process", "mixed-shell", "absolute-open", "relative-fd-open")
)


@pytest.fixture
def inert_process_downstream(monkeypatch: pytest.MonkeyPatch):
    calls: list[object] = []

    def downstream_spy(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("dangerous process reached the downstream spy")

    monkeypatch.setattr(subprocess, "_USE_POSIX_SPAWN", False)
    monkeypatch.setattr(subprocess, "_fork_exec", downstream_spy)
    return calls


def _invoke_without_exception_swallow(action):
    try:
        return action()
    except Exception as exc:
        raise AssertionError("the guard must not be an Exception") from exc


@pytest.mark.parametrize(
    ("scope", "route", "blocked"),
    _PERMISSION_MATRIX,
    ids=lambda value: str(value).replace(" ", "-"),
)
def test_permission_scope_matrix_uses_only_inert_downstreams(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    route: str,
    blocked: bool,
) -> None:
    import hardware_isolation as isolation
    from hardware_isolation import permit_hardware, permit_wrapper, restore_access

    calls: list[object] = []

    def downstream_process(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("process reached the inert downstream spy")

    def downstream_open(*args: object, **kwargs: object) -> int:
        path = args[0] if args else kwargs.get("file")
        flags = args[1] if len(args) > 1 else 0
        sys.audit("open", path, None, flags)
        calls.append((args, kwargs))
        raise AssertionError("open reached the inert downstream spy")

    monkeypatch.setattr(subprocess, "_USE_POSIX_SPAWN", False)
    monkeypatch.setattr(subprocess, "_fork_exec", downstream_process)
    token = None
    if scope == "wrapper":
        token = permit_wrapper()
    elif scope == "hardware":
        token = permit_hardware()
    try:
        if route == "direct-process":
            action = subprocess.run
            arguments = (("tt-smi", "-s"),)
            keyword_arguments = {"check": False}
        elif route == "mixed-shell":
            action = subprocess.run
            arguments = (("/bin/sh", "-c", "run_in_container.sh; tt-smi -s"),)
            keyword_arguments = {"check": False}
        elif route == "absolute-open":
            monkeypatch.setattr(io, "open", downstream_open)
            action = pathlib.Path("/dev/tenstorrent/0").open
            arguments = ("rb",)
            keyword_arguments = {}
        else:
            monkeypatch.setattr(isolation.os, "readlink", lambda _path: "/dev")
            monkeypatch.setattr(isolation, "_original_os_open", downstream_open)
            action = isolation.os.open
            arguments = ("tenstorrent/0", isolation.os.O_RDONLY)
            keyword_arguments = {"dir_fd": 99}

        if blocked:
            with pytest.raises(HardwareAccessViolation):
                action(*arguments, **keyword_arguments)
            assert calls == []
        else:
            with pytest.raises(AssertionError):
                action(*arguments, **keyword_arguments)
            assert calls
    finally:
        if token is not None:
            restore_access(token)


@pytest.mark.parametrize("command", _BLOCKED_COMMANDS)
def test_subprocess_entrypoints_are_blocked_before_downstream_process(
    command: tuple[str, ...], inert_process_downstream: list[object]
) -> None:
    with pytest.raises(HardwareAccessViolation):
        subprocess.run(command, check=False)

    assert inert_process_downstream == []


@pytest.mark.parametrize(
    "entrypoint",
    (subprocess.run, subprocess.check_output, subprocess.check_call, subprocess.call),
    ids=("run", "check-output", "check-call", "call"),
)
def test_cached_subprocess_entrypoints_are_blocked(
    entrypoint, inert_process_downstream: list[object]
) -> None:
    if entrypoint is subprocess.run:
        action = entrypoint
        command = ("tt-smi", "-s")
        kwargs = {"check": False}
    elif entrypoint is subprocess.call:
        action = entrypoint
        command = ("/usr/bin/docker", "run")
        kwargs = {}
    else:
        action = entrypoint
        command = ("tt-topology", "--json")
        kwargs = {}

    with pytest.raises(HardwareAccessViolation):
        action(command, **kwargs)

    assert inert_process_downstream == []


@pytest.mark.parametrize(
    "entrypoint",
    (subprocess.getoutput, subprocess.getstatusoutput),
    ids=("get-output", "get-status-output"),
)
def test_shell_subprocess_helpers_are_blocked(
    entrypoint, inert_process_downstream: list[object]
) -> None:
    with pytest.raises(HardwareAccessViolation):
        entrypoint("tt-smi -s")

    assert inert_process_downstream == []


def test_cached_popen_alias_is_blocked_before_process_creation(
    inert_process_downstream: list[object],
) -> None:
    cached_popen = subprocess.Popen

    with pytest.raises(HardwareAccessViolation):
        cached_popen(("/usr/bin/tt-flash", "--help"))

    assert inert_process_downstream == []


@pytest.mark.parametrize(
    "command",
    (
        "tt-smi -s",
        "/usr/bin/docker run image",
        ("/bin/sh", "-c", "exec /usr/bin/tt-topology --json"),
        ("env", "--", "/usr/bin/tt-flash", "--help"),
        ("/bin/sh", "-c", "true;tt-smi -s"),
        ("/bin/sh", "-c", "echo x|tt-smi -s"),
        ("bash", "-lc", "tt-smi -s"),
        ("bash", "-e", "run_in_container.sh", "--"),
    ),
    ids=(
        "shell-string",
        "absolute-shell-string",
        "shell-vector",
        "env-vector",
        "semicolon-without-space",
        "pipe-without-space",
        "combined-c-option",
        "shell-script-option",
    ),
)
def test_shell_and_nested_command_forms_are_blocked(
    command: str | tuple[str, ...], inert_process_downstream: list[object]
) -> None:
    kwargs = {"shell": True} if isinstance(command, str) else {}
    with pytest.raises(HardwareAccessViolation):
        subprocess.run(command, check=False, **kwargs)

    assert inert_process_downstream == []


def test_os_system_shell_form_is_blocked_by_the_audit_boundary() -> None:
    with pytest.raises(HardwareAccessViolation):
        sys.audit("os.system", "exec /usr/bin/tt-smi -s")


def test_exception_cannot_be_hidden_by_an_exception_handler(
    inert_process_downstream: list[object],
) -> None:
    action = lambda: subprocess.run(("docker", "run", "image"), check=False)

    with pytest.raises(HardwareAccessViolation):
        _invoke_without_exception_swallow(action)

    assert inert_process_downstream == []


@pytest.mark.parametrize("api", ("builtins", "io", "os", "pathlib"))
def test_device_open_apis_are_blocked_before_downstream_open(
    monkeypatch: pytest.MonkeyPatch, api: str
) -> None:
    calls: list[object] = []

    def downstream_open(*args: object, **kwargs: object):
        path = args[0] if args else kwargs.get("file")
        flags = args[1] if len(args) > 1 else 0
        # The C-level APIs emit this event before touching the device.  The
        # inert shim preserves that ordering while keeping RED safe.
        sys.audit("open", path, None, flags)
        calls.append((args, kwargs))
        raise AssertionError("device open reached the downstream spy")

    device_path = "/dev/tenstorrent/child"
    if api == "builtins":
        monkeypatch.setattr(builtins, "open", downstream_open)
        operation = lambda: builtins.open(device_path, "rb")  # noqa: SIM115
    elif api == "io":
        monkeypatch.setattr(io, "open", downstream_open)
        operation = lambda: io.open(device_path, "rb")  # noqa: SIM115, UP020
    elif api == "os":
        monkeypatch.setattr(os, "open", downstream_open)
        operation = lambda: os.open(device_path, os.O_RDONLY)
    else:
        monkeypatch.setattr(io, "open", downstream_open)
        operation = lambda: pathlib.Path(device_path).open("rb")  # noqa: SIM115

    with pytest.raises(HardwareAccessViolation):
        operation()

    assert calls == []


def test_device_root_and_normalized_descendants_are_blocked() -> None:
    from hardware_isolation import _is_device_path

    assert _is_device_path("/dev/tenstorrent") is True
    assert _is_device_path("//dev/tenstorrent/./1") is True
    assert _is_device_path("/dev/tenstorrent-neighbor") is False


def test_device_symlink_alias_is_blocked_before_downstream_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    alias = tmp_path / "device-alias"
    alias.symlink_to("/dev/tenstorrent/0")
    calls: list[object] = []

    def downstream_open(*args: object, **kwargs: object):
        path = args[0] if args else kwargs.get("file")
        flags = args[1] if len(args) > 1 else 0
        sys.audit("open", path, None, flags)
        calls.append((args, kwargs))
        raise AssertionError("device symlink open reached the downstream spy")

    monkeypatch.setattr(io, "open", downstream_open)
    operation = lambda: pathlib.Path(alias).open("rb")  # noqa: SIM115

    with pytest.raises(HardwareAccessViolation):
        operation()

    assert calls == []


def test_relative_device_path_uses_mocked_cwd_before_downstream_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def downstream_open(*args: object, **kwargs: object):
        path = args[0] if args else kwargs.get("path")
        flags = args[1] if len(args) > 1 else 0
        sys.audit("open", path, None, flags)
        calls.append((args, kwargs))
        raise AssertionError("relative device open reached the downstream spy")

    monkeypatch.setattr(os, "getcwd", lambda: "/dev/tenstorrent")
    monkeypatch.setattr(os, "open", downstream_open)
    operation = lambda: os.open("0", os.O_RDONLY)

    with pytest.raises(HardwareAccessViolation):
        operation()

    assert calls == []


def test_relative_dir_fd_device_path_is_blocked_before_downstream_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hardware_isolation as isolation

    calls: list[object] = []

    def downstream_open(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("dir_fd device open reached the downstream spy")

    cached_open = isolation.os.open
    monkeypatch.setattr(isolation.os, "readlink", lambda _path: "/dev")
    monkeypatch.setattr(isolation, "_original_os_open", downstream_open)
    operation = lambda: cached_open("tenstorrent/0", isolation.os.O_RDONLY, dir_fd=99)

    with pytest.raises(HardwareAccessViolation):
        operation()

    assert calls == []


def test_relative_dir_fd_metadata_failure_fails_closed_before_downstream_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hardware_isolation as isolation

    calls: list[object] = []

    def downstream_open(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("unclassifiable dir_fd open reached the downstream spy")

    def missing_metadata(_path: str) -> str:
        raise OSError("metadata unavailable")

    cached_open = isolation.os.open
    monkeypatch.setattr(isolation.os, "readlink", missing_metadata)
    monkeypatch.setattr(isolation, "_original_os_open", downstream_open)
    operation = lambda: cached_open("relative-child", isolation.os.O_RDONLY, dir_fd=99)

    with pytest.raises(HardwareAccessViolation):
        operation()

    assert calls == []


def test_hardware_permission_forwards_relative_dir_fd_to_inert_original_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hardware_isolation as isolation
    from hardware_isolation import permit_hardware, restore_access

    calls: list[object] = []

    def downstream_open(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("hardware permission did not reach the inert spy")

    cached_open = isolation.os.open
    monkeypatch.setattr(isolation, "_original_os_open", downstream_open)
    token = permit_hardware()
    try:
        operation = lambda: cached_open("tenstorrent/0", isolation.os.O_RDONLY, dir_fd=99)
        with pytest.raises(AssertionError):
            operation()
    finally:
        restore_access(token)

    assert calls


def test_ordinary_relative_dir_fd_open_remains_available(tmp_path: pathlib.Path) -> None:
    child = tmp_path / "child"
    child.write_text("safe\n", encoding="utf-8")
    directory_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        child_fd = os.open("child", os.O_RDONLY, dir_fd=directory_fd)
        try:
            assert os.read(child_fd, 5) == b"safe\n"
        finally:
            os.close(child_fd)
    finally:
        os.close(directory_fd)


def test_at_fdcwd_relative_open_remains_available(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child"
    child.write_text("safe\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    at_fdcwd = getattr(os, "AT_FDCWD", -100)
    file_fd = os.open("child", os.O_RDONLY, dir_fd=at_fdcwd)
    try:
        assert os.read(file_fd, 1)
    finally:
        os.close(file_fd)


def test_at_fdcwd_mocked_device_cwd_is_blocked_without_chdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hardware_isolation as isolation

    calls: list[object] = []

    def downstream_open(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("mocked device cwd reached the downstream spy")

    cached_open = isolation.os.open
    monkeypatch.setattr(os, "getcwd", lambda: "/dev/tenstorrent")
    monkeypatch.setattr(isolation, "_original_os_open", downstream_open)
    operation = lambda: cached_open(
        "0", isolation.os.O_RDONLY, dir_fd=getattr(os, "AT_FDCWD", -100)
    )

    with pytest.raises(HardwareAccessViolation):
        operation()

    assert calls == []


def test_wrapper_permission_does_not_allow_hardware_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hardware_isolation import permit_wrapper, restore_access

    calls: list[object] = []

    def downstream_spy(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("hardware process reached the downstream spy")

    monkeypatch.setattr(subprocess, "_USE_POSIX_SPAWN", False)
    monkeypatch.setattr(subprocess, "_fork_exec", downstream_spy)
    token = permit_wrapper()
    try:
        action = subprocess.run
        command = ("tt-smi", "-s")
        with pytest.raises(HardwareAccessViolation):
            action(command, check=False)
    finally:
        restore_access(token)

    assert calls == []


def test_wrapper_permission_does_not_allow_a_hardware_command_with_wrapper_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hardware_isolation import permit_wrapper, restore_access

    calls: list[object] = []

    def downstream_spy(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("hardware process reached the downstream spy")

    monkeypatch.setattr(subprocess, "_USE_POSIX_SPAWN", False)
    monkeypatch.setattr(subprocess, "_fork_exec", downstream_spy)
    token = permit_wrapper()
    try:
        action = subprocess.run
        command = ("docker", "run", "run_in_container.sh")
        with pytest.raises(HardwareAccessViolation):
            action(command, check=False)
    finally:
        restore_access(token)

    assert calls == []


def test_wrapper_permission_rejects_mixed_shell_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hardware_isolation import permit_wrapper, restore_access

    calls: list[object] = []

    def downstream_spy(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("mixed shell command reached the downstream spy")

    monkeypatch.setattr(subprocess, "_USE_POSIX_SPAWN", False)
    monkeypatch.setattr(subprocess, "_fork_exec", downstream_spy)
    token = permit_wrapper()
    try:
        action = subprocess.run
        command = ("/bin/sh", "-c", "run_in_container.sh; tt-smi -s")
        with pytest.raises(HardwareAccessViolation):
            action(command, check=False)
    finally:
        restore_access(token)

    assert calls == []


def test_wrapper_permission_reaches_only_inert_wrapper_downstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hardware_isolation import permit_wrapper, restore_access

    calls: list[object] = []

    def downstream_spy(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        raise AssertionError("wrapper reached the inert downstream spy")

    monkeypatch.setattr(subprocess, "_USE_POSIX_SPAWN", False)
    monkeypatch.setattr(subprocess, "_fork_exec", downstream_spy)
    token = permit_wrapper()
    try:
        action = subprocess.run
        command = ("run_in_container.sh", "--", "--iters", "1")
        with pytest.raises(AssertionError):
            action(command, check=False)
    finally:
        restore_access(token)

    assert calls


def test_issue137_missing_capture_callback_is_blocked_before_workload(
    tmp_path: pathlib.Path,
    inert_process_downstream: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import issue137 as runner

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    plan = runner.load_plan(
        repo_root / "docs/measurements/2026-10-18-host-newton-schulz-issue137-plan.json"
    )
    configuration = runner.configuration(plan, "C018")
    workload_calls: list[object] = []

    def command_runner(*args: object, **kwargs: object):
        workload_calls.append((args, kwargs))
        raise AssertionError("workload callback was reached before snapshot capture")

    options = runner.Issue137RunOptions(command_runner=command_runner)
    with pytest.raises(HardwareAccessViolation):
        runner.run_one(
            plan=plan,
            configuration=configuration,
            mode="smoke",
            output_dir=tmp_path,
            run_id="issue137-missing-capture-callback",
            options=options,
        )

    assert workload_calls == []
    assert inert_process_downstream == []


def test_ordinary_subprocess_and_file_operations_remain_available(tmp_path: pathlib.Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "print('safe')"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == "safe\n"
    data_argument = subprocess.run(
        [sys.executable, "-c", "print('safe')", "docker"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert data_argument.stdout == "safe\n"

    path = tmp_path / "safe.txt"
    path.write_text("safe\n", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "safe\n"


@pytest.mark.usefixtures("hardware_access")
def test_hardware_permission_fixture_is_disabled_by_default(request) -> None:
    if request.config.getoption("--allow-hardware-tests"):
        pytest.skip("hardware permission opt-in is explicitly enabled")
    pytest.fail("hardware permission fixture unexpectedly enabled")


@pytest.mark.usefixtures("wrapper_execution")
def test_wrapper_execution_permission_fixture_is_disabled_by_default(request) -> None:
    if request.config.getoption("--allow-wrapper-execution"):
        pytest.skip("wrapper execution opt-in is explicitly enabled")
    pytest.fail("wrapper execution permission fixture unexpectedly enabled")
