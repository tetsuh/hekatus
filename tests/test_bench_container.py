"""Host-only acceptance coverage for the benchmark container boundary."""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

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
