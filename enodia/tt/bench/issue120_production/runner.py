"""Board-free by default; execution requires explicit stage/session admission."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from enodia.tt.bench import issue120_results
from enodia.tt.bench.run_binding import current_run_id

from . import plan, recording, records, runtime, sessions


def _unit(identity: str) -> str:
    position = [s.identity for s in plan.RUNS].index(identity)
    return ("B", "B", "C", "C", "P", "P")[position]


def _session_spec(identity: str, session_id: str):
    manifest, digest = sessions.read_session(session_id)
    spec = sessions.spec_for_session(identity, manifest)
    approved = {s.identity: s for s in sessions.admitted_specs(_unit(identity), manifest["admission_hashes"])}
    if spec != approved.get(identity) or manifest["program_identity"] != runtime.program_identity():
        raise ValueError("session differs from the admitted precommit")
    return spec, manifest, digest


def _clean_pushed() -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=runtime.ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain") or git("rev-parse", "@{upstream}") != head:
        raise ValueError("hardware execution requires a clean pushed source revision")
    return head


def wrapper_command(spec: plan.RunSpec, session_id: str) -> tuple[list[str], dict]:
    session_id = sessions.canonical_session_id(session_id)
    case = sessions.session_path(session_id) / "cases" / spec.identity
    environment = os.environ.copy()
    environment.update(HEKATUS_TT_RUNNER="enodia/tt/bench/issue120_production/run_production.py",
                       HEKATUS_TT_CONTAINER_TIMEOUT_S=str(spec.cap_seconds),
                       HEKATUS_TT_DEVICE_NODE="/dev/tenstorrent/0", HEKATUS_TT_IMAGE=runtime.PINNED_IMAGE,
                       HEKATUS_TT_TELEMETRY_MODE="default")
    environment.pop("HEKATUS_TT_TELEMETRY_INTERVAL_S", None)
    if spec.watcher:
        environment["TT_METAL_WATCHER"] = "1"
    else:
        environment.pop("TT_METAL_WATCHER", None)
    args = ["bash", str(runtime.ROOT / "enodia/tt/bench/run_in_container.sh"), str(case),
            "--", "--run", spec.identity, "--owner-approved", "--session-id", session_id]
    if spec.phase == "production":
        args.append("--production-approved")
    return args, environment


def _single_result(directory: Path) -> Path:
    paths = list(directory.glob("results-*.json"))
    # Companions have additional suffixes, never count as a primary candidate.
    primaries = []
    for p in paths:
        try:
            data = issue120_results._read_json(p)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("schema") == records.SCHEMA:
            primaries.append(p)
    if len(primaries) != 1:
        raise ValueError("one fresh production record is required")
    return primaries[0]


def _control(session_id: str, spec: plan.RunSpec) -> dict | None:
    if spec.condition != "loaded" or spec.phase == "calibration":
        return None
    control_id = plan.RUNS[0].identity if spec.phase == "startup" else plan.RUNS[4].identity
    path = sessions.session_path(session_id) / "cases" / control_id
    primary = _single_result(path)
    control = recording.load_validated_record(primary)
    if not control.get("execution_eligible") or control["session_id"] != session_id:
        raise ValueError("healthy same-session control required before loaded")
    if spec.phase == "production" and not control.get("timing_evidence"):
        raise ValueError("production control lacks timing/power eligibility")
    if spec.phase == "production":
        manifest, digest = sessions.read_session(session_id)
        if manifest.get("production_contract") == runtime.PRODUCTION_CONTRACT:
            from . import production_admission
            production_admission.control_release(control, session_id)
            reference = control["production_provenance"]["receipt"]
            captured = runtime.fixed_path(primary.parent / reference["file"], primary.parent)
            if runtime.sha(captured) != manifest["admission_hashes"]["production"]:
                raise ValueError("P1 captured receipt differs from the admitted session")
            receipt = json.loads(captured.read_bytes())
            control_spec = sessions.spec_for_session(control_id, manifest)
            sessions.verify_production_completion(control_spec, manifest, digest, receipt, primary)
    return control


def _launch(args) -> int:
    if not args.owner_approved or not args.session_id:
        raise ValueError("explicit owner approval and session are required")
    spec, session, digest = _session_spec(args.launch, args.session_id)
    if spec.phase == "production" and not args.production_approved:
        raise ValueError("production needs separate explicit owner approval")
    if _clean_pushed() != session["harness_commit"]:
        raise ValueError("session source revision changed")
    _control(args.session_id, spec)
    case = sessions.session_path(args.session_id) / "cases" / spec.identity
    if spec.phase == "production" and session.get("production_contract") == runtime.PRODUCTION_CONTRACT:
        receipt = sessions.compatible_production_receipt(session["admission_hashes"])
        sessions.reserve_production_launch(spec, session, digest, receipt)
    else:
        case.mkdir(parents=True, exist_ok=False)
    command, environment = wrapper_command(spec, args.session_id)
    # This is a launcher watchdog, not a larger native/container measurement cap.
    with (case / "wrapper.log").open("xb") as output:
        subprocess.run(command, env=environment, cwd=runtime.ROOT, stdout=output,
                       stderr=subprocess.STDOUT, check=True, timeout=spec.cap_seconds + 120)
    result = recording.load_validated_record(_single_result(case))
    print(json.dumps({"record": result["attempt_identity"], "execution_eligible": result["execution_eligible"]}))
    return 0 if result["execution_eligible"] else 2


def _enter_native_claim(spec: plan.RunSpec, session: dict, digest: str) -> None:
    if spec.phase == "production" and session.get("production_contract") == runtime.PRODUCTION_CONTRACT:
        receipt = sessions.compatible_production_receipt(session["admission_hashes"])
        sessions.enter_production_native(spec, session, digest, receipt, Path("/out"))


def _run(args) -> int:
    if not args.owner_approved or not args.session_id:
        raise ValueError("no native execution without explicit approval/session")
    if runtime.ROOT != Path("/work"):
        raise ValueError("native execution requires the pinned wrapper mount")
    spec, session, digest = _session_spec(args.run, args.session_id)
    if spec.phase == "production" and not args.production_approved:
        raise ValueError("production needs separate explicit owner approval")
    identity = current_run_id()
    if not identity:
        raise ValueError("wrapper-owned attempt identity required")
    environment = runtime.issue120_probe._environment_record(f"/out/env-{identity}.json")
    if environment["commit"] != session["harness_commit"]:
        raise ValueError("wrapper provenance differs from the prepared session")
    control = _control(args.session_id, spec)
    attempt = issue120_results.create_attempt(identity)
    watcher_path = runtime.ROOT / "generated/watcher/watcher.log"
    watcher_before = recording.read_watcher_bytes(watcher_path) if spec.watcher and watcher_path.is_file() else None
    _enter_native_claim(spec, session, digest)
    started, utc_start = time.monotonic_ns(), time.time_ns()
    native, program_identity = runtime.execute_native(spec, attempt)
    ended, utc_end = time.monotonic_ns(), time.time_ns()
    record = recording.make_record(spec, attempt, native, program_identity, session, digest,
                                   execute_start_ns=started, execute_end_ns=ended,
                                   start_utc_ns=utc_start, end_utc_ns=utc_end, watcher_before=watcher_before)
    if control is not None and spec.phase == "production" and record["timing_evidence"]:
        record["control_comparison"] = records.compare_records(control, record)
    recording.write_record(Path("/out") / f"results-{identity}.json", record)
    return 0 if record["execution_eligible"] and (spec.watcher or record["timing_evidence"]) else 2


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build-only", action="store_true")
    group.add_argument("--prepare-unit", choices=("B", "C", "P"))
    group.add_argument("--launch", choices=[s.identity for s in plan.RUNS])
    group.add_argument("--run", choices=[s.identity for s in plan.RUNS])
    group.add_argument("--compare", type=Path, nargs=2)
    group.add_argument("--calibration-proposal", type=Path, nargs=2)
    parser.add_argument("--owner-approved", action="store_true")
    parser.add_argument("--production-approved", action="store_true")
    parser.add_argument("--session-id")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.build_only:
            runtime.build()
        elif args.run:
            return _run(args)
        elif args.launch:
            return _launch(args)
        elif args.prepare_unit:
            _clean_pushed()
            manifest = sessions.create_session(sessions.admitted_specs(args.prepare_unit))
            print(json.dumps(manifest, indent=2))
        elif args.compare:
            rows = [recording.load_validated_record(p) for p in args.compare]
            print(json.dumps(records.compare_records(*rows), indent=2))
        elif args.calibration_proposal:
            rows = [recording.load_validated_record(p) for p in args.calibration_proposal]
            print(json.dumps(records.calibration_proposal(rows), indent=2))
        else:
            print(json.dumps(plan.proposal(), indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        print(f"Issue120 production rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
