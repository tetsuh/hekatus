"""CPU-only fixture publisher; replay is not a new hardware measurement."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from enodia.tt.bench import issue120_results
from enodia.tt.bench.issue120_production import plan, recording, runtime

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/measurements/issue120-production/2026-11-18-b/B1-evidence.tar.gz"
IDENTITY = "20261118T124104054310116761824398Z"
SESSION_ID = "13ba906388a94f7ba744f367481ae818"
HARNESS_COMMIT = "b474216164c92bdda026fdf35e3a1d768635c7fa"


def publish_fixture(output: Path, *, watcher: bool = True, creation_umask: int = 0o022) -> Path:
    """Publish through the real record/frozen-snapshot path, without a device."""
    previous = os.umask(creation_umask)
    try:
        return _publish_fixture(output, watcher=watcher)
    finally:
        os.umask(previous)


def _publish_fixture(output: Path, *, watcher: bool) -> Path:
    source = output / "fixture-input"
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(source, filter="data")
    old = json.loads((source / f"results-{IDENTITY}.json").read_bytes())
    attempt = issue120_results.NativeAttempt(IDENTITY, output / f"native-{IDENTITY}")
    attempt.result_directory.mkdir(parents=True)
    (attempt.directory / "evidence").mkdir()
    for name in ("stdout", "stderr", "execution"):
        meta = old["native_capture"][name]
        shutil.copyfile(source / meta["file"], attempt.directory / (name + (".json" if name == "execution" else ".bin")))
    shutil.copyfile(source / old["native_capture"]["result"]["file"], attempt.result_file)
    os.chmod(attempt.result_file, 0o600)  # Keep private ADR-0018 native bytes private.
    for kind in ("frames", "transitions"):
        target = attempt.directory / "evidence" / (kind + ".bin")
        shutil.copyfile(source / f"results-{IDENTITY}-{kind}.bin", target)
        os.chmod(target, 0o600)
    shutil.copyfile(source / f"env-{IDENTITY}.json", output / f"env-{IDENTITY}.json")
    shutil.copyfile(source / f"power-{IDENTITY}.csv", output / f"power-{IDENTITY}.csv")
    fake_root = output / "fixture-runtime"
    log = fake_root / "generated/watcher/watcher.log"
    log.parent.mkdir(parents=True)
    log.write_bytes((source / old["watcher"]["artifact"]["file"]).read_bytes())
    spec = replace(plan.RUNS[0], watcher=watcher)
    session = {"harness_commit": old["environment"]["commit"], "session_id": old["session_id"],
               "host_boot_id_sha256": old["host_boot_id_sha256"]}
    bounds = old["execution_interval"]
    with patch.object(runtime, "ROOT", fake_root), patch.object(recording.issue120_probe, "_environment_record", return_value=old["environment"]):
        record = recording.make_record(spec, attempt, old["native"], old["program_identity"], session,
            old["session_manifest_sha256"], execute_start_ns=bounds["start_ns"],
            execute_end_ns=bounds["end_ns"], start_utc_ns=bounds["start_utc_ns"],
            end_utc_ns=bounds["end_utc_ns"], watcher_before=None)
    path = output / f"results-{IDENTITY}.json"
    recording.write_record(path, record)
    return path


if __name__ == "__main__":
    assert not Path("/dev/tenstorrent").exists()
    assert all(not p.exists() or not any(p.iterdir()) for p in map(Path, ("/sys", "/proc/bus", "/proc/driver")))
    result = publish_fixture(Path("/out"))
    print(json.dumps({"fixture_only": True, "publisher_euid": os.geteuid(), "record": result.name}))
