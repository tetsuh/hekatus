"""Device-free root publisher -> real unprivileged launcher readback check.

This is separate from pytest's deliberately Docker-denying hardware audit guard.
It never launches an accelerator executable. Outputs are test fixtures only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path
from unittest.mock import patch

from enodia.tt.bench.issue120_production import plan, runner, sessions

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "out/issue120-publication-check"
REAL_RUN = subprocess.run


def _command(args: list[str]) -> subprocess.CompletedProcess:
    return REAL_RUN(args, check=True, capture_output=True, text=True, timeout=60)


def _create(name: str, case: Path, config: Path) -> None:
    payload = {"Image": runner.runtime.PINNED_IMAGE, "User": "0:0", "WorkingDir": "/work",
               "Entrypoint": ["python3", "/work/tests/issue120_production_replay.py"],
               "Env": ["PYTHONPATH=/work", "PYTHONDONTWRITEBYTECODE=1"],
               "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True,
                              "CapDrop": ["ALL"], "CapAdd": ["DAC_OVERRIDE"],
                              "SecurityOpt": ["no-new-privileges"],
                              "Binds": [str(ROOT) + ":/work:ro", str(case) + ":/out:rw"],
                              "MaskedPaths": ["/sys", "/proc/bus", "/proc/driver"]}}
    config.write_text(json.dumps(payload))
    _command(["curl", "-fsS", "--unix-socket", "/var/run/docker.sock", "-H", "Content-Type: application/json",
              "-X", "POST", "http://localhost/containers/create?name=" + name,
              "--data-binary", "@" + str(config)])


def _inspect(name: str, directory: Path) -> None:
    result = _command(["docker", "inspect", name])
    (directory / "inspect.json").write_text(result.stdout)
    actual = json.loads(result.stdout)[0]
    host = actual["HostConfig"]
    assert not host.get("Devices")
    assert not host.get("DeviceRequests")
    assert host["NetworkMode"] == "none"
    assert host["ReadonlyRootfs"]
    assert not host["Privileged"]
    mounts = {entry["Destination"]: entry for entry in actual["Mounts"]}
    assert mounts["/work"]["RW"] is False
    assert mounts["/out"]["RW"] is True
    assert host["CapDrop"] == ["ALL"]
    assert host["CapAdd"] == ["DAC_OVERRIDE"]
    assert {"/sys", "/proc/bus", "/proc/driver"}.issubset(host["MaskedPaths"])
    assert actual["Config"]["User"] == "0:0"


def _run_check(directory: Path) -> dict:
    original = ROOT / "docs/measurements/issue120-production/2026-11-18-b/runtime-session-manifest.json"
    manifest = json.loads(original.read_bytes())
    session = directory / "session"
    case = session / "cases" / plan.RUNS[0].identity
    name = "hekatus-power-fixture-" + uuid.uuid4().hex

    def child(command, **kwargs):
        assert command == ["root-fixture-child"]
        _create(name, case, directory / "config.json")
        try:
            _inspect(name, directory)
            result = REAL_RUN(["docker", "start", "-a", name], **kwargs)
            assert result.returncode == 0
            snapshot = next(case.glob("power-trace-snapshot-*.csv"))
            assert snapshot.stat().st_uid == 0
            return result
        finally:
            _command(["docker", "rm", "-f", name])

    with (patch.object(sessions, "session_path", return_value=session),
          patch.object(runner, "_session_spec", return_value=(plan.RUNS[0], manifest, "fixture-hash")),
          patch.object(runner, "_clean_pushed", return_value=manifest["harness_commit"]),
          patch.object(runner, "wrapper_command", return_value=(["root-fixture-child"], {})),
          patch.object(subprocess, "run", side_effect=child)):
        args = argparse.Namespace(owner_approved=True, session_id=manifest["session_id"],
                                  launch=plan.RUNS[0].identity, production_approved=False)
        code = runner._launch(args)
    assert code == 0
    assert '"publisher_euid": 0' in (case / "wrapper.log").read_text()
    return {"fixture_only": True, "publisher_euid": 0, "reader_euid": os.geteuid(),
            "launcher_exit": code, "snapshot_mode": oct(next(case.glob("power-trace-snapshot-*.csv")).stat().st_mode & 0o777)}


def main() -> int:
    if os.geteuid() == 0:
        raise RuntimeError("the host reader must be unprivileged")
    directory = OUTPUT / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    try:
        result = _run_check(directory)
    except (OSError, ValueError, AssertionError, subprocess.SubprocessError) as error:
        result = {"fixture_only": True, "passed": False, "error": type(error).__name__}
        (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"report": str(directory.relative_to(ROOT)), **result}))
        raise
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(directory.relative_to(ROOT)), **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
