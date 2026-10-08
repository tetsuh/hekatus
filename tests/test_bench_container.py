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
RESIDENT_RUNNER_ALIASES = (
    "./enodia/tt/bench/run_resident.py",
    "enodia/tt/bench/../bench/run_resident.py",
    "/work/enodia/tt/bench/run_resident.py",
    "enodia//tt//bench//run_resident.py",
)
MATMUL_RUNNER_ALIASES = (
    "./enodia/tt/bench/run_matmul.py",
    "enodia/tt/bench/../bench/run_matmul.py",
    "/work/enodia/tt/bench/run_matmul.py",
    "enodia//tt//bench//run_matmul.py",
)


def _fake_tools(
    tmp_path: Path,
    *,
    sampler_exit: int | None = None,
    timeout_log: Path | None = None,
    sampler_log: Path | None = None,
) -> Path:
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
if [ "${{2:-}}" = sample ] && [ -n "${{SAMPLER_LOG:-}}" ]; then
  printf '%s\\n' "$@" > "$SAMPLER_LOG"
fi
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
if [ "${1:-}" = kill ]; then
  printf '%s\\n' "$@" > "${DOCKER_CONTROL_ARGS:-/dev/null}"
  exit 0
fi
printf '%s\\n' "$@" > "$DOCKER_ARGS"
result_path=""
for argument in "$@"; do
  case "$argument" in
    HEKATUS_TT_RESULT_PATH=*) result_path="${argument#*=}" ;;
  esac
done
if [ -n "${RUNNER_RESULT_HOST:-}" ] && [ -n "$result_path" ]; then
  printf '%s\\n' '{"fake_runner_result": true}' > "$RUNNER_RESULT_HOST"
fi
sleep "${DOCKER_DELAY:-0}"
exit "${DOCKER_EXIT:-0}"
"""
    )
    (bindir / "docker").chmod(stat.S_IRWXU)
    if timeout_log is not None:
        (bindir / "timeout").write_text(
            """#!/bin/sh
set -eu
printf '%s\\n' "$@" > "${TIMEOUT_LOG:?}"
shift 3
exec "$@"
"""
        )
        (bindir / "timeout").chmod(stat.S_IRWXU)
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


@pytest.mark.parametrize(
    ("output_argument", "case"),
    [
        ("relative-output", "relative"),
        ("absolute-output", "absolute"),
        ("missing-parent/child-output", "nonexistent-parent"),
    ],
    ids=["relative-source", "absolute-source", "source-under-nonexistent-parent"],
)
def test_wrapper_normalizes_every_docker_bind_source(
    tmp_path, output_argument, case
):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]
    if case == "absolute":
        output_argument = str(tmp_path / output_argument)
    expected_output = Path(output_argument)
    if not expected_output.is_absolute():
        expected_output = copied_root / expected_output
    expected_output = expected_output.resolve()

    completed = subprocess.run(
        [str(copied_wrapper), output_argument, "--", "--iters", "1"],
        cwd=copied_root,
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "DOCKER_ARGS": str(args_log)},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    docker_args = args_log.read_text().splitlines()
    mounts = [
        docker_args[index + 1]
        for index, argument in enumerate(docker_args[:-1])
        if argument == "-v"
    ]
    assert mounts == [
        "/dev/hugepages-1G:/dev/hugepages-1G",
        f"{copied_root}:/work",
        f"{expected_output}:/out",
    ]
    assert all(Path(mount.rsplit(":", 1)[0]).is_absolute() for mount in mounts)
    if case == "nonexistent-parent":
        assert expected_output.is_dir()


def test_wrapper_rejects_an_unnormalizable_mount_source_before_docker(tmp_path):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]

    completed = subprocess.run(
        [str(copied_wrapper), "", "--", "--iters", "1"],
        cwd=copied_root,
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "DOCKER_ARGS": str(args_log)},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "cannot normalize" in completed.stderr
    assert not args_log.exists()


def test_wrapper_rejects_runner_outside_repository_before_docker(tmp_path):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]
    outside_runner = copied_root.parent / "outside" / "runner.py"

    completed = subprocess.run(
        [str(copied_wrapper), "--", "--iters", "1"],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "HEKATUS_TT_RUNNER": str(outside_runner),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "outside repository" in completed.stderr
    assert not args_log.exists()
    assert not (copied_root / "out").exists()


def test_wrapper_accepts_equivalent_absolute_host_runner_path(tmp_path):
    timeout_log = tmp_path / "timeout-args"
    bindir = _fake_tools(tmp_path, timeout_log=timeout_log)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]
    host_runner = copied_root / "enodia/tt/bench/run_resident.py"

    completed = subprocess.run(
        [str(copied_wrapper), "--", "--watcher"],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": str(host_runner),
            "TT_METAL_WATCHER": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert timeout_log.read_text().splitlines()[2] == "60s"
    assert "enodia/tt/bench/run_resident.py" in args_log.read_text().splitlines()


def _run_resident_wrapper(
    tmp_path,
    *,
    container_timeout,
    runner_args,
    watcher_env=None,
    image_override=None,
    timeout_log=None,
    sampler_log=None,
    runner="enodia/tt/bench/run_resident.py",
):
    bindir = _fake_tools(tmp_path, timeout_log=timeout_log, sampler_log=sampler_log)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "DOCKER_ARGS": str(args_log),
    }
    env.pop("HEKATUS_TT_RUNNER", None)
    env.pop("HEKATUS_TT_CONTAINER_TIMEOUT_S", None)
    env.pop("HEKATUS_TT_IMAGE", None)
    env.pop("SAMPLER_LOG", None)
    env.pop("TIMEOUT_LOG", None)
    if runner is not None:
        env["HEKATUS_TT_RUNNER"] = runner
    if container_timeout is not None:
        env["HEKATUS_TT_CONTAINER_TIMEOUT_S"] = container_timeout
    if timeout_log is not None:
        env["TIMEOUT_LOG"] = str(timeout_log)
    if sampler_log is not None:
        env["SAMPLER_LOG"] = str(sampler_log)
    env.pop("TT_METAL_WATCHER", None)
    if watcher_env is not None:
        env["TT_METAL_WATCHER"] = watcher_env
    if image_override is not None:
        env["HEKATUS_TT_IMAGE"] = image_override
    return subprocess.run(
        [str(copied_wrapper), "--", *runner_args],
        cwd=copied_wrapper.parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


class TestBenchDefaults:
    @pytest.mark.parametrize(
        "runner",
        RESIDENT_RUNNER_ALIASES,
        ids=("dot", "parent", "work-absolute", "duplicate-slashes"),
    )
    @pytest.mark.parametrize(
        ("runner_args", "watcher_env", "expected_timeout"),
        [([], None, "600s"), (["--watcher"], "1", "60s")],
        ids=("timing", "watcher"),
    )
    def test_resident_runner_alias_uses_mode_timeout_when_unset(
        self, tmp_path, runner, runner_args, watcher_env, expected_timeout
    ):
        timeout_log = tmp_path / "timeout-args"
        completed = _run_resident_wrapper(
            tmp_path,
            container_timeout=None,
            runner_args=runner_args,
            watcher_env=watcher_env,
            timeout_log=timeout_log,
            runner=runner,
        )
        assert completed.returncode == 0, completed.stderr
        assert timeout_log.read_text().splitlines()[2] == expected_timeout

    @pytest.mark.parametrize(
        "runner",
        RESIDENT_RUNNER_ALIASES,
        ids=("dot", "parent", "work-absolute", "duplicate-slashes"),
    )
    @pytest.mark.parametrize(
        ("runner_args", "watcher_env", "container_timeout"),
        [(["--watcher"], "1", "61"), ([], None, "601")],
        ids=("watcher", "timing"),
    )
    def test_resident_runner_alias_rejects_explicit_timeout_over_cap(
        self, tmp_path, runner, runner_args, watcher_env, container_timeout
    ):
        completed = _run_resident_wrapper(
            tmp_path,
            container_timeout=container_timeout,
            runner_args=runner_args,
            watcher_env=watcher_env,
            runner=runner,
        )
        assert completed.returncode == 2
        assert "approved cap" in completed.stderr
        assert not (tmp_path / "docker-args").exists()

    @pytest.mark.parametrize(
        "runner",
        MATMUL_RUNNER_ALIASES,
        ids=("dot", "parent", "work-absolute", "duplicate-slashes"),
    )
    def test_matmul_runner_alias_uses_power_trace_and_matmul_docker_branch(
        self, tmp_path, runner
    ):
        timeout_log = tmp_path / "timeout-args"
        completed = _run_resident_wrapper(
            tmp_path,
            container_timeout=None,
            runner_args=["--iters", "1"],
            timeout_log=timeout_log,
            runner=runner,
        )
        assert completed.returncode == 0, completed.stderr
        assert timeout_log.read_text().splitlines()[2] == "900s"
        docker_args = (tmp_path / "docker-args").read_text().splitlines()
        assert docker_args[docker_args.index("--entrypoint") + 1] == "/bin/bash"
        assert docker_args.count("enodia/tt/bench/run_matmul.py") == 1
        assert "--power-trace" in docker_args

    @pytest.mark.parametrize(
        ("runner_args", "watcher_env", "expected_timeout"),
        [([], None, "600s"), (["--watcher"], "1", "60s")],
        ids=("timing", "watcher"),
    )
    def test_resident_wrapper_uses_mode_timeout_when_unset(
        self, tmp_path, runner_args, watcher_env, expected_timeout
    ):
        timeout_log = tmp_path / "timeout-args"
        completed = _run_resident_wrapper(
            tmp_path,
            container_timeout=None,
            runner_args=runner_args,
            watcher_env=watcher_env,
            timeout_log=timeout_log,
        )
        assert completed.returncode == 0, completed.stderr
        assert timeout_log.read_text().splitlines()[2] == expected_timeout

    def test_nonresident_wrapper_keeps_900_second_default(self, tmp_path):
        timeout_log = tmp_path / "timeout-args"
        sampler_log = tmp_path / "sampler-args"
        completed = _run_resident_wrapper(
            tmp_path,
            container_timeout=None,
            runner_args=["--iters", "1"],
            timeout_log=timeout_log,
            sampler_log=sampler_log,
            runner=None,
        )
        assert completed.returncode == 0, completed.stderr
        assert timeout_log.read_text().splitlines()[2] == "900s"
        assert sampler_log.read_text().splitlines()[-1] == "2"
        docker_args = (tmp_path / "docker-args").read_text().splitlines()
        assert "enodia/tt/bench/run_matmul.py" in docker_args
        assert docker_args[docker_args.index("--device") + 1] == "/dev/tenstorrent/0"
        output_mount = next(argument for argument in docker_args if argument.endswith(":/out"))
        output_host, output_container = output_mount.rsplit(":", 1)
        assert Path(output_host).name == "bench"
        assert output_container == "/out"


def test_resident_wrapper_rejects_unpinned_image_before_telemetry(tmp_path):
    completed = _run_resident_wrapper(
        tmp_path,
        container_timeout="600",
        runner_args=[],
        image_override="registry.example/tt:latest",
    )
    assert completed.returncode == 2
    assert "requires" in completed.stderr
    assert not list((tmp_path / "repo/out").glob("env-*.json"))


def test_resident_wrapper_accepts_verified_digest_image(tmp_path):
    completed = _run_resident_wrapper(
        tmp_path,
        container_timeout="600",
        runner_args=[],
        image_override="registry.example/tt@sha256:" + "a" * 64,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("container_timeout", "runner_args", "watcher_env", "error"),
    [
        ("601", [], None, "approved cap"),
        ("61", ["--watcher"], "1", "approved cap"),
        ("-1", [], None, "decimal digits"),
        ("", [], None, "decimal digits"),
        ("abc", [], None, "decimal digits"),
        ("+600", [], None, "decimal digits"),
        ("0", [], None, "decimal digits"),
        ("999999999999999999999", [], None, "too many digits"),
        ("0600", [], None, "decimal digits"),
    ],
)
def test_resident_wrapper_rejects_unsafe_timeout_strings(
    tmp_path, container_timeout, runner_args, watcher_env, error
):
    completed = _run_resident_wrapper(
        tmp_path,
        container_timeout=container_timeout,
        runner_args=runner_args,
        watcher_env=watcher_env,
    )
    assert completed.returncode == 2
    assert error in completed.stderr
    assert not list((tmp_path / "repo/out").glob("env-*.json"))


@pytest.mark.parametrize(
    ("container_timeout", "runner_args", "watcher_env"),
    [("600", [], None), ("60", ["--watcher"], "1")],
)
def test_resident_wrapper_accepts_exact_timeout_boundaries(
    tmp_path, container_timeout, runner_args, watcher_env
):
    completed = _run_resident_wrapper(
        tmp_path,
        container_timeout=container_timeout,
        runner_args=runner_args,
        watcher_env=watcher_env,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("runner_args", "watcher_env"),
    [(["--watcher"], None), ([], "1")],
)
def test_resident_wrapper_rejects_watcher_mode_mismatch(tmp_path, runner_args, watcher_env):
    completed = _run_resident_wrapper(
        tmp_path,
        container_timeout="60",
        runner_args=runner_args,
        watcher_env=watcher_env,
    )
    assert completed.returncode == 2
    assert "same mode" in completed.stderr


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
    control_log = tmp_path / "docker-control-args"
    docker = bindir / "docker"
    docker.write_text(
        f"""#!{sys.executable}
import os
import sys
import time
from pathlib import Path

if sys.argv[1:2] == ["kill"]:
    Path(os.environ["DOCKER_CONTROL_ARGS"]).write_text("\\n".join(sys.argv[1:]) + "\\n")
    raise SystemExit(0)
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
            "DOCKER_CONTROL_ARGS": str(control_log),
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
        control_args = control_log.read_text().splitlines()
        assert control_args[0] == "kill"
        assert "--signal" in control_args
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
import sys
import time
from pathlib import Path

if sys.argv[1:2] == ["kill"]:
    Path({str(tmp_path / "docker-control-args")!r}).write_text("\\n".join(sys.argv[1:]) + "\\n")
    raise SystemExit(0)
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
import sys
import time
from pathlib import Path

if sys.argv[1:2] == ["kill"]:
    raise SystemExit(0)
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


def test_wrapper_runs_device_pytest_only_in_the_pinned_container(tmp_path):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")

    completed = subprocess.run(
        [
            str(copied_wrapper),
            "--pytest",
            "-m",
            "tt_device",
            "tests/test_newton_schulz_kernel.py",
        ],
        cwd=copied_wrapper.parents[3],
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "DOCKER_ARGS": str(args_log)},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    docker_args = args_log.read_text().splitlines()
    assert docker_args[docker_args.index("--entrypoint") + 1] == "/usr/local/bin/uv"
    assert "HEKATUS_TT_DEVICE_TEST=1" in docker_args
    assert "HEKATUS_TT_PINNED_CONTAINER=1" in docker_args
    assert docker_args[-12:] == [
        "run",
        "--no-project",
        "--with",
        "pytest==8.3.5",
        "--with",
        "scipy==1.13.1",
        "python",
        "-m",
        "pytest",
        "-m",
        "tt_device",
        "tests/test_newton_schulz_kernel.py",
    ]


def test_wrapper_rejects_an_unpinned_image_for_device_pytest(tmp_path):
    bindir = _fake_tools(tmp_path)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")

    completed = subprocess.run(
        [str(copied_wrapper), "--pytest", "-m", "tt_device"],
        cwd=copied_wrapper.parents[3],
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "HEKATUS_TT_IMAGE": "tt-metal:latest",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "digest-pinned" in completed.stderr


def test_wrapper_uses_a_named_container_and_inner_timeout(tmp_path):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    completed = subprocess.run(
        [str(copied_wrapper), "--", "--iters", "1"],
        cwd=copied_wrapper.parents[3],
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "DOCKER_ARGS": str(args_log)},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    docker_args = args_log.read_text().splitlines()
    name = docker_args[docker_args.index("--name") + 1]
    assert re.fullmatch(r"hekatus-bench-[0-9]+-[0-9]+", name)
    assert docker_args[docker_args.index("--device") + 1] == "/dev/tenstorrent/0"
    assert docker_args[docker_args.index("--entrypoint") + 1] == "/bin/bash"
    assert "--power-trace" in docker_args
    power_trace = docker_args[docker_args.index("--power-trace") + 1]
    assert power_trace.startswith("/out/power-")
    assert power_trace.endswith(".csv")
    assert "HEKATUS_TT_RESULT_PATH=/out/runner-result.json" not in docker_args
    assert re.search(r"results     -> .*/results-[0-9TZ]+\.json", completed.stdout)


def test_wrapper_runs_sampler_off_without_power_trace(tmp_path):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    output_dir = tmp_path / "off"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")

    completed = subprocess.run(
        [str(copied_wrapper), str(output_dir), "--"],
        cwd=copied_wrapper.parents[3],
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "HEKATUS_TT_TELEMETRY_MODE": "off",
            "HEKATUS_TT_RUNNER": "enodia/tt/bench/run_resident.py",
            "HEKATUS_TT_CONTAINER_TIMEOUT_S": "600",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    docker_args = args_log.read_text().splitlines()
    assert "HEKATUS_TT_TELEMETRY_MODE=off" in docker_args
    assert "--out" in docker_args
    assert docker_args[docker_args.index("--out") + 1] == "/out/runner-result.json"
    assert "--power-trace" not in docker_args
    assert not list(output_dir.glob("power-*.csv"))
    assert "sampler -> off (diagnostic-only; no power trace by design)" in completed.stdout


def test_wrapper_sampled_then_sampler_off_reuses_output_without_old_trace(tmp_path):
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    output_dir = tmp_path / "shared-output"
    sampled_tools = tmp_path / "sampled-tools"
    sampled_tools.mkdir()
    sampled_bindir = _fake_tools(sampled_tools)
    sampled_args = sampled_tools / "docker-args"
    common_environment = {
        **os.environ,
        "HEKATUS_TT_RUNNER": "enodia/tt/bench/run_resident.py",
        "HEKATUS_TT_CONTAINER_TIMEOUT_S": "600",
    }

    sampled = subprocess.run(
        [str(copied_wrapper), str(output_dir), "--"],
        cwd=copied_wrapper.parents[3],
        env={
            **common_environment,
            "PATH": f"{sampled_bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(sampled_args),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert sampled.returncode == 0, sampled.stderr
    sampled_docker_args = sampled_args.read_text().splitlines()
    trace_index = sampled_docker_args.index("--power-trace")
    sampled_trace = sampled_docker_args[trace_index + 1]
    assert sampled_trace.startswith("power-")
    assert sampled_trace.endswith(".csv")
    environment_files = sorted(output_dir.glob("env-*.json"))
    assert len(environment_files) == 1
    run_id = environment_files[0].stem.removeprefix("env-")
    assert sampled_trace == f"power-{run_id}.csv"
    old_trace = output_dir / sampled_trace
    old_trace.write_text("old trace\n")

    off_tools = tmp_path / "off-tools"
    off_tools.mkdir()
    off_bindir = _fake_tools(off_tools)
    off_args = off_tools / "docker-args"
    off = subprocess.run(
        [str(copied_wrapper), str(output_dir), "--"],
        cwd=copied_wrapper.parents[3],
        env={
            **common_environment,
            "PATH": f"{off_bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(off_args),
            "HEKATUS_TT_TELEMETRY_MODE": "off",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert off.returncode == 0, off.stderr
    off_docker_args = off_args.read_text().splitlines()
    assert "--power-trace" not in off_docker_args
    assert [path.name for path in output_dir.glob("power-*.csv")] == [old_trace.name]
    assert "sampler -> off (diagnostic-only; no power trace by design)" in off.stdout


def test_wrapper_forwards_explicit_sampler_interval_and_keeps_default_trace_contract(tmp_path):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    output_dir = tmp_path / "explicit"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")

    completed = subprocess.run(
        [str(copied_wrapper), str(output_dir), "--", "--iters", "1"],
        cwd=copied_wrapper.parents[3],
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "HEKATUS_TT_TELEMETRY_MODE": "explicit",
            "HEKATUS_TT_TELEMETRY_INTERVAL_S": "5",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    docker_args = args_log.read_text().splitlines()
    assert "HEKATUS_TT_TELEMETRY_MODE=explicit" in docker_args
    assert "HEKATUS_TT_TELEMETRY_INTERVAL_S=5" in docker_args
    assert "--power-trace" in docker_args
    assert "sampler -> explicit (5s)" in completed.stdout


def test_wrapper_kills_the_named_container_when_inner_timeout_expires(tmp_path):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    control_log = tmp_path / "docker-control-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    completed = subprocess.run(
        [str(copied_wrapper), "--", "--iters", "1"],
        cwd=copied_wrapper.parents[3],
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "DOCKER_CONTROL_ARGS": str(control_log),
            "DOCKER_DELAY": "30",
            "HEKATUS_TT_CONTAINER_TIMEOUT_S": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 124, completed.stderr
    control_args = control_log.read_text().splitlines()
    assert control_args[0] == "kill"
    assert "--signal" in control_args


def test_wrapper_can_run_a_probe_with_the_same_container_lifecycle(tmp_path):
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    completed = subprocess.run(
        [str(copied_wrapper), "--", "--stage", "1", "--no-watcher"],
        cwd=copied_wrapper.parents[3],
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_bringup.py",
            "TT_METAL_WATCHER": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    docker_args = args_log.read_text().splitlines()
    assert "--entrypoint" in docker_args
    assert docker_args[docker_args.index("--entrypoint") + 1] == "python3"
    watcher_index = docker_args.index("TT_METAL_WATCHER=1")
    assert docker_args[watcher_index - 1] == "-e"
    assert docker_args[-4:] == [
        "tools/newton_schulz_bringup.py",
        "--stage",
        "1",
        "--no-watcher",
    ]


CUSTOM_RUNNERS = (
    "tools/newton_schulz_bringup.py",
    "tools/newton_schulz_perf_counters.py",
    "tools/newton_schulz_issue100_same_run.py",
    "tools/newton_schulz_issue101_combined.py",
)


@pytest.mark.parametrize("runner", CUSTOM_RUNNERS)
def test_wrapper_uses_one_result_contract_for_every_custom_runner(tmp_path, runner):
    """Every discovered custom runner gets and writes the stable result path."""
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    output_dir = tmp_path / "comparison"
    result_path = output_dir / "runner-result.json"
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")

    completed = subprocess.run(
        [str(copied_wrapper), str(output_dir), "--"],
        cwd=copied_wrapper.parents[3],
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(args_log),
            "RUNNER_RESULT_HOST": str(result_path),
            "HEKATUS_TT_RUNNER": runner,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    docker_args = args_log.read_text().splitlines()
    assert runner in docker_args
    assert "HEKATUS_TT_RESULT_PATH=/out/runner-result.json" in docker_args
    assert f"results     -> {result_path}" in completed.stdout
    assert "results-" not in completed.stdout
    assert json.loads(result_path.read_text()) == {"fake_runner_result": True}


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
    child_env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "DOCKER_ARGS": str(args_log),
    }
    # This test is about the default the wrapper falls back to, and the wrapper
    # honours HEKATUS_TT_IMAGE over it. Anyone aiming a run at another image
    # exports that variable, so inheriting it from the shell would silently
    # measure the override instead and fail reporting the wrong digest.
    child_env.pop("HEKATUS_TT_IMAGE", None)
    completed = subprocess.run(
        [str(copied_wrapper), str(output_dir), "--", "--iters", "1"],
        cwd=copied_root,
        env=child_env,
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
            "--sampler-mode",
            "default",
            "--sampler-interval",
            "2",
        ],
        "image": expected_image,
        "image_pinned": True,
    }
