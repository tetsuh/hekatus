"""Host-only acceptance coverage for the benchmark container boundary."""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "enodia/tt/bench/run_in_container.sh"


def _fake_tools(tmp_path: Path, *, sampler_exit: int | None = None) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sample = (
        f"    exit {sampler_exit}"
        if sampler_exit is not None
        else "    trap 'exit 0' TERM INT\n    while :; do sleep 1; done"
    )
    (bindir / "python3").write_text(
        f"""#!/bin/sh
set -eu
case "$2" in
  capture-env)
    out=""
    while [ "$#" -gt 0 ]; do
      [ "$1" = --out ] && {{ out="$2"; shift 2; continue; }}
      shift
    done
    printf '%s\\n' '{{"fake": true}}' > "$out"
    ;;
  sample)
{sample}
    ;;
esac
"""
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    (bindir / "docker").write_text(
        """#!/bin/sh
set -eu
printf '%s\\n' "$@" > "$DOCKER_ARGS"
sleep "${DOCKER_DELAY:-0}"
exit "${DOCKER_EXIT:-0}"
"""
    )
    (bindir / "docker").chmod(stat.S_IRWXU)
    return bindir


def test_wrapper_forwards_hostile_runner_arguments_literally(tmp_path):
    """Both documented output forms reach Docker without shell evaluation."""
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    injected = tmp_path / "injected"
    hostile = f"; touch {injected}"

    # Run a copy rooted in tmp_path so the documented default output directory
    # is isolated from the repository's tracked and untracked state.
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]

    for output_args in ([], [str(tmp_path / "explicit")]):
        completed = subprocess.run(
            [str(copied_wrapper), *output_args, "--", "--iters", hostile],
            cwd=copied_root,
            env={
                **os.environ,
                "PATH": f"{bindir}:{os.environ['PATH']}",
                "DOCKER_ARGS": str(args_log),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        docker_args = args_log.read_text().splitlines()
        assert hostile in docker_args
        assert not injected.exists()

        output_dir = Path(output_args[0]) if output_args else copied_root / "out/bench"
        env_files = list(output_dir.glob("env-*.json"))
        assert env_files
        assert json.loads(env_files[-1].read_text()) == {"fake": True}


def test_wrapper_fails_when_sampler_exits_before_docker(tmp_path):
    """A prematurely dead sampler cannot produce a successful benchmark."""
    bindir = _fake_tools(tmp_path, sampler_exit=23)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]

    completed = subprocess.run(
        [str(copied_wrapper), "--", "--iters", "1"],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "DOCKER_DELAY": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode != 0
    assert "sampler" in completed.stderr


def test_wrapper_reaps_docker_when_interrupted(tmp_path):
    """Interrupting the wrapper must not leave its benchmark child running."""
    bindir = _fake_tools(tmp_path)
    child_pid_file = tmp_path / "docker-child-pid"
    args_log = tmp_path / "docker-args"
    docker = bindir / "docker"
    docker.write_text(
        f"""#!{sys.executable}
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

Path(os.environ["DOCKER_ARGS"]).write_text("\\n".join(sys.argv[1:]) + "\\n")
Path(os.environ["DOCKER_CHILD_PID"]).write_text(str(os.getpid()))
time.sleep(float(os.environ.get("DOCKER_DELAY", "0")))
raise SystemExit(int(os.environ.get("DOCKER_EXIT", "0")))
"""
    )
    docker.chmod(stat.S_IRWXU)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]

    process = subprocess.Popen(
        [str(copied_wrapper), "--", "--iters", "1"],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "DOCKER_CHILD_PID": str(child_pid_file),
            "DOCKER_DELAY": "30",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    docker_pid = None
    try:
        deadline = time.monotonic() + 5
        while not child_pid_file.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("wrapper exited before starting Docker")
            time.sleep(0.01)
        assert child_pid_file.exists()
        docker_pid = int(child_pid_file.read_text())

        process.terminate()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode != 0, stdout + stderr
        with pytest.raises(ProcessLookupError):
            os.kill(docker_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if docker_pid is not None:
            try:
                os.kill(docker_pid, 9)
            except ProcessLookupError:
                pass


def test_a_failed_benchmark_reports_its_own_status_even_if_the_sampler_died(tmp_path):
    """When both fail, the benchmark's status is the one worth surfacing:
    a caller has to be able to tell a broken run from a broken observer."""
    bindir = _fake_tools(tmp_path, sampler_exit=23)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]

    completed = subprocess.run(
        [str(copied_wrapper), "--", "--iters", "1"],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "DOCKER_DELAY": "1",
            "DOCKER_EXIT": "17",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 17, completed.stderr
    assert "sampler exited before benchmark completion" in completed.stderr


def test_the_sampler_is_reaped_when_the_wrapper_is_signalled_before_docker_reports(tmp_path):
    """The traps must own the sampler from before it is launched.

    The existing interruption test waits for the fake Docker to record its PID
    first, so it only exercises the phase where both children are known. This
    one signals while Docker has produced nothing, which is the phase the
    ordering fix is about.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sampler_pid_file = tmp_path / "sampler.pid"
    (bindir / "python3").write_text(
        f"""#!{sys.executable}
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

if sys.argv[2] == "capture-env":
    out = sys.argv[sys.argv.index("--out") + 1]
    Path(out).write_text('{{"fake": true}}')
    raise SystemExit(0)
Path({str(sampler_pid_file)!r}).write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
"""
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    # Docker blocks without announcing anything, so the wrapper is signalled
    # while the sampler is the only child that has made itself known.
    (bindir / "docker").write_text(
        f"""#!{sys.executable}
import time

while True:
    time.sleep(0.05)
"""
    )
    (bindir / "docker").chmod(stat.S_IRWXU)

    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]

    process = subprocess.Popen(
        [str(copied_wrapper), "--", "--iters", "1"],
        cwd=copied_root,
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sampler_pid = None
    try:
        deadline = time.monotonic() + 5
        while not sampler_pid_file.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("wrapper exited before starting the sampler")
            time.sleep(0.01)
        assert sampler_pid_file.exists()
        sampler_pid = int(sampler_pid_file.read_text())

        process.terminate()
        process.communicate(timeout=10)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(sampler_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        with pytest.raises(ProcessLookupError):
            os.kill(sampler_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if sampler_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(sampler_pid, 9)


@pytest.mark.parametrize("assignment", ["sampler", "docker"])
def test_cleanup_owns_the_child_before_its_pid_is_published(tmp_path, assignment):
    """Exercise each `${!:-}` fallback before its PID assignment executes."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sampler_pid_file = tmp_path / "sampler.pid"
    docker_pid_file = tmp_path / "docker.pid"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    (bindir / "python3").write_text(
        f"""#!{sys.executable}
import os
import sys
import time
from pathlib import Path

if sys.argv[2] == "capture-env":
    Path(sys.argv[sys.argv.index("--out") + 1]).write_text('{{"fake": true}}')
    raise SystemExit(0)
Path({str(sampler_pid_file)!r}).write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
"""
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    (bindir / "docker").write_text(
        f"""#!{sys.executable}
import os
import time
from pathlib import Path

Path({str(docker_pid_file)!r}).write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
"""
    )
    (bindir / "docker").chmod(stat.S_IRWXU)

    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    assignment_line = "SAMPLER_PID=$!" if assignment == "sampler" else "DOCKER_PID=$!"
    barrier = (
        ': > "${PID_ASSIGNMENT_READY:?}"\n'
        'while [[ ! -e "${PID_ASSIGNMENT_RELEASE:?}" ]]; do :; done\n'
        f"{assignment_line}"
    )
    wrapper_text = copied_wrapper.read_text()
    assert wrapper_text.count(assignment_line) == 1
    copied_wrapper.write_text(wrapper_text.replace(assignment_line, barrier))
    copied_wrapper.chmod(stat.S_IRWXU)
    copied_root = copied_wrapper.parents[3]

    process = subprocess.Popen(
        [str(copied_wrapper), "--", "--iters", "1"],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "PID_ASSIGNMENT_READY": str(ready),
            "PID_ASSIGNMENT_RELEASE": str(release),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pids = []
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("wrapper exited before the PID-assignment barrier")
            time.sleep(0.01)
        assert ready.exists()

        deadline = time.monotonic() + 5
        required = [sampler_pid_file]
        if assignment == "docker":
            required.append(docker_pid_file)
        while not all(path.exists() for path in required) and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("wrapper exited before starting required child")
            time.sleep(0.01)
        assert all(path.exists() for path in required)
        pids = [int(path.read_text()) for path in required]
        if assignment == "sampler":
            assert not docker_pid_file.exists()

        process.terminate()
        process.communicate(timeout=10)
        assert process.returncode != 0
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for pid in pids:
            with suppress(ProcessLookupError):
                os.kill(pid, 9)


def test_the_default_toolchain_image_is_digest_pinned_and_recorded(tmp_path):
    """The default invocation records the exact immutable image provenance."""
    expected_image = (
        "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@"
        "sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"
    )
    wrapper = (
        Path(__file__).resolve().parents[1] / "enodia" / "tt" / "bench" / "run_in_container.sh"
    ).read_text()
    match = re.search(r'^IMAGE="\$\{HEKATUS_TT_IMAGE:-([^}]+)\}"', wrapper, re.MULTILINE)
    assert match is not None, "the wrapper no longer defines IMAGE with a default"
    default = match.group(1)
    assert default == expected_image
    # Against the wrapper's own default, not against the constant above: the
    # equality already pins the value, and this keeps the property being
    # guarded — a digest rather than a tag — checked where it can still fail.
    assert re.search(r"@sha256:[0-9a-f]{64}$", default)

    bindir = _fake_tools(tmp_path)
    telemetry = tmp_path / "repo/enodia/tt/bench/telemetry.py"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", telemetry)
    copied_root = copied_wrapper.parents[3]
    (bindir / "python3").write_text(
        f"""#!{sys.executable}
import json
import signal
import sys
import time
from pathlib import Path

def raise_system_exit(*_):
    raise SystemExit

if sys.argv[2] == "capture-env":
    arguments = sys.argv[1:]
    output = arguments[arguments.index("--out") + 1]
    image = arguments[arguments.index("--image") + 1]
    Path(output).write_text(json.dumps({{
        "capture_env_argv": arguments,
        "image": image,
        "image_pinned": "--image-pinned" in arguments,
    }}))
else:
    signal.signal(signal.SIGTERM, raise_system_exit)
    while True:
        time.sleep(1)
"""
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    args_log = tmp_path / "docker-args"
    output_dir = tmp_path / "output"
    completed = subprocess.run(
        [str(copied_wrapper), str(output_dir), "--", "--iters", "1"],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr

    env_files = list(output_dir.glob("env-*.json"))
    assert len(env_files) == 1
    environment = json.loads(env_files[0].read_text())
    assert environment == {
        "capture_env_argv": [
            str(telemetry),
            "capture-env",
            "--out",
            str(env_files[0]),
            "--image",
            expected_image,
            "--image-pinned",
        ],
        "image": expected_image,
        "image_pinned": True,
    }
