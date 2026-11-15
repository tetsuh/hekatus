"""Exclusive session manifests and committed stage admissions."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

from . import plan, runtime

SESSIONS = runtime.ROOT / "out/issue120-production/sessions"


def boot_fingerprint() -> str:
    return hashlib.sha256(Path("/proc/sys/kernel/random/boot_id").read_bytes().strip()).hexdigest()


def canonical_session_id(value: str) -> str:
    if not isinstance(value, str) or len(value) != 32:
        raise ValueError("session identity must be canonical UUID hex")
    number = int(value, 16)
    if not 0 <= number < 2**128 or value != format(number, "032x"):
        raise ValueError("session identity must be canonical UUID hex")
    return format(number, "032x")


def session_path(identity: str) -> Path:
    return runtime.fixed_path(SESSIONS / canonical_session_id(identity), SESSIONS, exists=False)


def read_session(identity: str) -> tuple[dict, str]:
    path = session_path(identity) / "manifest.json"
    raw = runtime.fixed_path(path, SESSIONS).read_bytes()
    data = json.loads(raw)
    if (data.get("schema") != "issue120-production-session-v1" or data.get("session_id") != identity
            or runtime.ROOT != Path("/work") and data.get("host_boot_id_sha256") != boot_fingerprint()):
        raise ValueError("session manifest identity/schema differs")
    return data, hashlib.sha256(raw).hexdigest()


def create_session(specs: list[plan.RunSpec]) -> dict:
    identity = uuid.uuid4().hex
    directory = session_path(identity)
    directory.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=runtime.ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    data = {"schema": "issue120-production-session-v1", "session_id": identity,
            "created_monotonic_ns": time.monotonic_ns(), "harness_commit": commit,
            "host_boot_id_sha256": boot_fingerprint(),
            "program_identity": runtime.program_identity(), "image": runtime.PINNED_IMAGE,
            "admission_hashes": admission_hashes(specs[0].phase),
            "runs": {s.identity: {"configuration": plan.program_configuration(s),
                                   "configuration_sha256": plan.configuration_digest(plan.program_configuration(s))}
                     for s in specs}}
    with (directory / "manifest.json").open("x") as output:
        json.dump(data, output, sort_keys=True, indent=2)
        output.write("\n")
    return data


def admission_hashes(phase: str) -> dict:
    names = {"startup": (), "calibration": ("bounded",), "production": ("bounded", "production")}[phase]
    hashes = {}
    for kind in names:
        _committed_admission(kind)
        hashes[kind] = runtime.sha(runtime.SOURCE / f"{kind}_admission.json")
    return hashes


def _committed_admission(kind: str, bindings: dict | None = None) -> dict:
    path = runtime.SOURCE / f"{kind}_admission.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{kind} stage admission is absent; no execution authorized")
    raw = path.read_bytes()
    if bindings is not None:
        if bindings.get(kind) != runtime.sha(path):
            raise ValueError("stage admission differs from the host-verified session")
    else:
        relative = path.relative_to(runtime.ROOT).as_posix()
        result = subprocess.run(["git", "show", "HEAD:" + relative], cwd=runtime.ROOT,
                                check=True, capture_output=True)
        if result.stdout != raw:
            raise ValueError("stage admission must be precommitted unchanged")
    receipt = json.loads(raw)
    if receipt.get("schema") != f"issue120-{kind}-admission-v1":
        raise ValueError("stage admission schema differs")
    if receipt.get("program_identity") != runtime.program_identity():
        raise ValueError("stage admission is for a different program")
    return receipt


def _records_from_receipt(receipt: dict) -> list[dict]:
    from .recording import load_validated_record

    refs = receipt.get("records")
    if not isinstance(refs, list) or len(refs) != 2:
        raise ValueError("stage admission requires both artifact-bound records")
    records = []
    for ref in refs:
        relative = Path(ref["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("stage record reference is not repository-relative")
        path = runtime.fixed_path(runtime.ROOT / relative, runtime.ROOT / "docs/measurements")
        if runtime.sha(path) != ref["sha256"]:
            raise ValueError("stage record digest differs")
        record = load_validated_record(path)
        if record["program_identity"] != receipt["program_identity"]:
            raise ValueError("stage record program differs")
        records.append(record)
    return records


def _healthy_bounded(record: dict, spec: plan.RunSpec) -> bool:
    watch = record.get("watcher", {})
    flags = (record.get("resident_integrity_passed"), record.get("execution_eligible"),
             record.get("warning_check", {}).get("passed"), watch.get("requested"),
             watch.get("fresh"), watch.get("scan_passed"))
    expected_coverage = "not_applicable" if spec.condition == "control" else "covered"
    return (all(flag is True for flag in flags) and record.get("plan_identity") == spec.identity
            and record.get("condition") == spec.condition and record.get("phase") == spec.phase
            and record.get("coverage_analysis", {}).get("coverage") == expected_coverage
            and record.get("configuration") == plan.program_configuration(spec))


def admitted_specs(unit: str, bindings: dict | None = None) -> list[plan.RunSpec]:
    if unit == "B":
        return list(plan.RUNS[:2])
    bounded = _committed_admission("bounded", bindings)
    rows = _records_from_receipt(bounded)
    if (bounded.get("operator_watcher_reviewed") is not True or
            not all(_healthy_bounded(r, s) for r, s in zip(rows, plan.RUNS[:2], strict=True)) or
            rows[1]["coverage_analysis"]["coverage"] != "covered"):
        raise ValueError("healthy first-kernel Watcher admission missing")
    if unit == "C":
        return list(plan.RUNS[2:4])
    if unit != "P":
        raise ValueError("unsupported approval unit")
    from .records import calibration_proposal

    receipt = _committed_admission("production", bindings)
    proposal = calibration_proposal(_records_from_receipt(receipt))
    if receipt.get("fixed_repetitions") != proposal["fixed_repetitions"]:
        raise ValueError("fixed production count differs from the calibrated precommit")
    return [replace(s, repeat_count=proposal["fixed_repetitions"]) for s in plan.RUNS[4:]]


def spec_for_session(identity: str, session: dict) -> plan.RunSpec:
    spec = plan.run_spec(identity)
    entry = session["runs"].get(identity)
    if not isinstance(entry, dict):
        raise TypeError("run is not assigned to this session")
    spec = replace(spec, repeat_count=entry["configuration"]["repeat_count"])
    if plan.program_configuration(spec) != entry["configuration"] or plan.configuration_digest(
            entry["configuration"]) != entry["configuration_sha256"]:
        raise ValueError("session program configuration differs")
    return spec
