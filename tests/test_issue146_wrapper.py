"""Board-free Issue #146 wrapper routing through inert local stubs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_bench_container import _fake_tools

ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "enodia/tt/bench/run_in_container.sh"


def _copy_wrapper(tmp_path: Path) -> tuple[Path, Path]:
    copied = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, copied)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied.parent / "telemetry.py")
    return copied, copied.parents[3]


@pytest.mark.usefixtures("wrapper_execution")
def test_issue146_wrapper_selects_fixed_smoke_timeout_and_owned_paths(tmp_path: Path) -> None:
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    copied, root = _copy_wrapper(tmp_path)
    environment = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "DOCKER_ARGS": str(args_log),
        "HEKATUS_TT_RUNNER": "tools/issue146_device.py",
        "HEKATUS_TT_CONTAINER_TIMEOUT_S": "90",
    }
    completed = subprocess.run(
        [
            str(copied), str(tmp_path / "out"), "--",
            "--configuration", "C018", "--mode", "smoke",
            "--target-seconds", "40", "--outer-timeout-seconds", "90",
            "--outer-repetitions", "16", "--launches-per-sync", "8",
            "--load-core-count", "18", "--load-total-blocks", "171",
            "--load-matrix-block", "8", "--plan-hash", "a" * 64,
        ],
        cwd=root, env=environment, capture_output=True, text=True, check=False, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    docker_args = args_log.read_text(encoding="utf-8").splitlines()
    assert "tools/issue146_device.py" in docker_args
    assert docker_args[docker_args.index("--entrypoint") + 1] == "python3"
    assert docker_args[docker_args.index("--outer-timeout-seconds") + 1] == "90"
    assert "--out" not in docker_args
    assert "--groups" in docker_args
    assert any(value.startswith("/out/power-") for value in docker_args)


@pytest.mark.usefixtures("wrapper_execution")
def test_issue146_wrapper_rejects_generic_or_conflicting_timeout_before_docker(tmp_path: Path) -> None:
    bindir = _fake_tools(tmp_path)
    args_log = tmp_path / "docker-args"
    copied, root = _copy_wrapper(tmp_path)
    environment = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "DOCKER_ARGS": str(args_log),
        "HEKATUS_TT_RUNNER": "tools/issue146_device.py",
    }
    completed = subprocess.run(
        [
            str(copied), str(tmp_path / "out"), "--",
            "--configuration", "C018", "--mode", "long",
            "--target-seconds", "180", "--outer-timeout-seconds", "170",
            "--outer-repetitions", "16", "--launches-per-sync", "8",
            "--plan-hash", "a" * 64,
        ],
        cwd=root, env=environment, capture_output=True, text=True, check=False, timeout=10,
    )
    assert completed.returncode == 2
    assert "requires" in completed.stderr
    assert not args_log.exists()
