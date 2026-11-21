"""Full P host path using fixture-only CPU publishers, never the device wrapper."""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from enodia.tt.bench.issue120_production import plan, recording, records, runner, runtime, sessions
from enodia.tt.bench.issue120_production import production_admission as production
from tools import issue120_production_fixture as fixture

ROOT = runtime.ROOT / "out/issue120-production-p-prep"


def _settings(directory, session):
    with tarfile.open(runtime.ROOT / sessions.C1_DIRECTORY / "C1-evidence.tar.gz") as bundle:
        raw = json.loads(bundle.extractfile(f"env-{sessions.C1_ATTEMPT}.json").read())
    fixture.save(directory / "fixture-settings.json", {"session_id": session["session_id"],
                 "monotonic_origin_ns": session["created_monotonic_ns"] + 10**8,
                 "UTC_origin_ns": time.time_ns(), "environment_template": raw})


def _args(identity, session):
    return argparse.Namespace(launch=identity, session_id=session["session_id"], owner_approved=True, production_approved=True)


def run() -> dict:
    directory = ROOT / uuid.uuid4().hex
    directory.mkdir(parents=True)
    observed, child_calls, rereads = {}, [], []
    real_loader = recording.load_validated_record
    failed_control = False

    def loader(path):
        start = time.perf_counter()
        row = real_loader(path)
        if path.is_relative_to(directory):
            rereads.append({"identity": row["plan_identity"], "seconds": time.perf_counter() - start})
            observed[row["plan_identity"]] = row
        return row

    def wrapper(spec, identity):
        role = "control" if spec.condition == "control" else "loaded"
        child_calls.append(role)
        if failed_control and role == "control":
            command = [sys.executable, str(Path(__file__).resolve()), "--failed-control", "--directory", str(directory)]
        else:
            command = [sys.executable, str(runtime.ROOT / "tools/issue120_production_fixture.py"),
                       "--mode", "publish", "--role", role, "--directory", str(directory)]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(runtime.ROOT)
        return command, environment

    def rejected_call(args, label, expected=(OSError, ValueError, subprocess.SubprocessError)):
        before = len(child_calls)
        try:
            runner._launch(args)
        except expected:
            if len(child_calls) != before:
                raise AssertionError("preflight rejection invoked a fixture child") from None
            return True
        raise AssertionError("expected rejection missing: " + label)

    with (patch.object(sessions, "SESSIONS", directory / "sessions"),
          patch.object(runtime, "program_identity", production.fixture_identity),
          patch.object(runner, "_clean_pushed", sessions.current_head),
          patch.object(runner, "wrapper_command", wrapper),
          patch.object(recording, "load_validated_record", loader)):
        specs = sessions.admitted_specs("P")
        session = sessions.create_session(specs, fixture_only=True)
        _settings(directory, session)
        control, loaded = (_args(s.identity, session) for s in specs)
        rejected = {}
        with patch.object(sessions, "boot_fingerprint", return_value="f" * 64):
            rejected["boot"] = rejected_call(control, "boot")
        with patch.object(sessions, "current_head", return_value="b" * 40):
            rejected["head"] = rejected_call(control, "head")
        empty = directory / "empty-source"
        empty.mkdir()
        with patch.object(runtime, "SOURCE", empty):
            rejected["missing_receipt"] = rejected_call(control, "missing receipt")
        modified = directory / "modified-source"
        modified.mkdir()
        altered = json.loads((runtime.SOURCE / "production_admission.json").read_bytes())
        altered["fixed_repetitions"] += 1
        fixture.save(modified / "production_admission.json", altered)
        with patch.object(runtime, "SOURCE", modified):
            rejected["modified_receipt"] = rejected_call(control, "modified receipt")
        rejected["missing_P1"] = rejected_call(loaded, "P1 missing")
        control.production_approved = False
        rejected["missing_approval"] = rejected_call(control, "approval")
        control.production_approved = True
        with patch.object(runner, "_clean_pushed", side_effect=ValueError("simulated dirty/unpushed Git state")):
            rejected["dirty_source"] = rejected_call(control, "dirty source")
        assert not child_calls
        p1, p2 = runner._launch(control), runner._launch(loaded)
        comparison = records.compare_records(observed[specs[0].identity], observed[specs[1].identity])
        assert p1 == p2 == 0 and len(child_calls) == 2
        positive_launches = len(child_calls)
        # Separate failed-P1 session: corrupt an actual published frame companion,
        # not an eligibility flag, and require real loader rejection before P2.
        failure_session = sessions.create_session(specs, fixture_only=True)
        _settings(directory, failure_session)
        failed_control = True
        try:
            runner._launch(_args(specs[0].identity, failure_session))
        except ValueError as error:
            if "fingerprint" not in str(error):
                raise
        else:
            raise AssertionError("corrupt P1 was accepted")
        rejected["failed_P1"] = rejected_call(_args(specs[1].identity, failure_session), "failed P1")
    result = {"schema": "issue120-production-fixture-dry-run-v1", "fixture_only": True, "P_executed": False,
              "native_invoked": False, "positive_launches": positive_launches, "P1_exit": p1, "P2_exit": p2,
              "comparison_recomputed": True, "comparison": comparison, "rejected": rejected,
              "actual_launcher_rereads": rereads, "launcher_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "external_state_simulated": ["clean/pushed Git state", "fixture native result/clock/telemetry claims"],
              "real_paths": ["committed receipt validation", "session creation/read/head/boot checks", "admission", "launcher/control gate", "record construction/publication", "artifact-based launcher reread/comparison"],
              "fixture_root": directory.relative_to(runtime.ROOT).as_posix(), "source_sha256": runtime.sha(Path(__file__))}
    fixture.save(directory / "dry-run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-control", action="store_true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.failed_control:
        directory = runtime.issue120_probe._bounded_path(args.directory, ROOT, "fixture directory", "directory")
        fixture.publish(directory, "control")
        settings = json.loads((directory / "fixture-settings.json").read_bytes())
        path = directory / "sessions" / settings["session_id"] / "cases" / plan.RUNS[4].identity / "results-synthetic-p1-frames.bin"
        with path.open("r+b") as output:
            output.seek(24)
            value = output.read(1)
            output.seek(24)
            output.write(bytes([value[0] ^ 1]))
    else:
        print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
