"""Board-free Issue #146 wrapper routing through inert local stubs."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_bench_container import _fake_tools
from tests.test_issue146_runner import PLAN, _capture_factory, _environment
from tools import analyze_issue146, issue146

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
            "--configuration", "C018", "--requested-run-id", "issue146-smoke-c018-1",
            "--mode", "smoke",
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
def test_issue146_wrapper_device_bytes_replay_through_host_and_analyzer(tmp_path: Path) -> None:
    copied, root = _copy_wrapper(tmp_path)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    output = tmp_path / "out"
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment_path = tmp_path / "environment-source.json"
    environment_path.write_text(json.dumps(_environment(config)), encoding="utf-8")
    real_root = ROOT

    (bindir / "python3").write_text(
        """#!/usr/bin/env python3
import os
import signal
import sys
import time
from pathlib import Path

args = sys.argv[1:]
def value(name):
    index = args.index(name)
    return args[index + 1]
if "capture-env" in args:
    Path(value("--out")).write_bytes(Path(os.environ["FAKE_ENV_SOURCE"]).read_bytes())
elif "sample" in args:
    path = Path(value("--out"))
    path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\\n"
        "2026-10-20T00:00:00+00:00,250,1350,70\\n",
        encoding="utf-8",
    )
    signal.signal(signal.SIGUSR1, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.05)
else:
    raise SystemExit(0)
""".replace("#!/usr/bin/env python3", f"#!{sys.executable}"),
        encoding="utf-8",
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    (bindir / "tt-smi").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bindir / "tt-smi").chmod(stat.S_IRWXU)
    (bindir / "docker").write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args and args[0] == "kill":
    raise SystemExit(0)
for index, value in enumerate(args):
    if index > 0 and args[index - 1] == "-e" and "=" in value:
        key, setting = value.split("=", 1)
        os.environ[key] = setting
mount = next(value for value in args if value.endswith(":/out"))
output = Path(mount.rsplit(":", 1)[0])
runner_index = args.index("tools/issue146_device.py")
runner_args = args[runner_index + 1:]
for index, value in enumerate(runner_args):
    if value in {"--env-json", "--groups", "--out"}:
        runner_args[index + 1] = str(output / Path(runner_args[index + 1]).name)
    elif value == "--plan":
        runner_args[index + 1] = str(Path(os.environ["REAL_ROOT"]) / runner_args[index + 1])
sys.path.insert(0, os.environ["REAL_ROOT"])
from tools import issue146_device

class Prepared:
    def launch(self):
        pass
    def close(self):
        pass
class FakeTtnn:
    def open_device(self, **_kwargs):
        return object()
    def synchronize_device(self, _device):
        pass
    def close_device(self, _device):
        pass
issue146_device.prepare_newton_schulz_grouped_load = lambda *args, **kwargs: Prepared()
issue146_device.time.monotonic_ns = iter((0, 40_000_000_000)).__next__
os.chdir(os.environ["REAL_ROOT"])
sys.modules["ttnn"] = FakeTtnn()
os.environ["HEKATUS_TT_RESULT_PATH"] = str(output / Path(os.environ["HEKATUS_TT_RESULT_PATH"]).name)
raise SystemExit(issue146_device.main(runner_args))
""".replace("#!/usr/bin/env python3", f"#!{sys.executable}"),
        encoding="utf-8",
    )
    (bindir / "docker").chmod(stat.S_IRWXU)

    def command_runner(command, **kwargs):
        runner_environment = dict(kwargs["env"])
        runner_environment.update(
            {
                "PATH": f"{bindir}:{os.environ['PATH']}",
                "FAKE_ENV_SOURCE": str(environment_path),
                "REAL_ROOT": str(real_root),
                "HEKATUS_TT_RUNNER": "tools/issue146_device.py",
            }
        )
        completed = subprocess.run(
            [str(copied), str(output), "--", *command[3:]],
            cwd=root,
            env=runner_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if completed.returncode != 0:
            raise AssertionError(f"wrapper failed: {completed.returncode}: {completed.stderr}")
        return completed

    run_id = "issue146-smoke-c018-integration"
    record = issue146.run_one(
        plan=plan, configuration=config, mode="smoke", output_dir=output, run_id=run_id,
        options=issue146.Issue146RunOptions(
            command_runner=command_runner,
            environment=_environment(config),
            capture_fn=_capture_factory("C018", "smoke"),
            trace_path=output / "arbiter-trace.csv",
            trace_interval_seconds=0.001,
        ),
    )
    assert record["status"] == "ok"
    assert record["provenance"]["wrapper_run_id"]
    assert record["groups"]["device_artifact"]["source"] == "device-produced raw group array"
    assert record["provenance"]["legacy_power_trace"]["status"] == "ok"
    replay = analyze_issue146.analyze_smoke([record], plan=plan)
    assert replay["mode"] == "smoke-diagnostic"
    assert (output / record["groups"]["path"]).is_file()
    assert (output / record["provenance"]["legacy_power_trace"]["path"]).is_file()


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
            "--configuration", "C018", "--requested-run-id", "issue146-long-c018-1",
            "--mode", "long",
            "--target-seconds", "180", "--outer-timeout-seconds", "170",
            "--outer-repetitions", "16", "--launches-per-sync", "8",
            "--plan-hash", "a" * 64,
        ],
        cwd=root, env=environment, capture_output=True, text=True, check=False, timeout=10,
    )
    assert completed.returncode == 2
    assert "requires" in completed.stderr
    assert not args_log.exists()
