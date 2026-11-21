"""Exclusive session manifests and committed stage admissions."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tarfile
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

from . import admission_bridge, bounded_evidence, plan, runtime

SESSIONS = runtime.ROOT / "out/issue120-production/sessions"
MEASUREMENTS_ROOT = runtime.ROOT / "docs/measurements"


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
    if data.get("fixture_only") and not SESSIONS.is_relative_to(runtime.ROOT / "out/issue120-production-p-prep"):
        raise ValueError("fixture session cannot enter the operational session root")
    if (data.get("production_contract") == runtime.PRODUCTION_CONTRACT and runtime.ROOT != Path("/work")
            and data.get("harness_commit") != current_head()):
        raise ValueError("production session source head/revision changed")
    return data, hashlib.sha256(raw).hexdigest()


def current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=runtime.ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def create_session(specs: list[plan.RunSpec], *, fixture_only: bool = False) -> dict:
    if specs[0].phase == "calibration" and specs != admitted_specs("C"):
        raise ValueError("calibration session must match the admitted C1/current scope")
    if specs[0].phase == "production" and specs != admitted_specs("P"):
        raise ValueError("production session must match the entire admitted pair")
    if fixture_only and not SESSIONS.is_relative_to(runtime.ROOT / "out/issue120-production-p-prep"):
        raise ValueError("fixture sessions require their isolated program-owned root")
    identity = uuid.uuid4().hex
    directory = session_path(identity)
    directory.mkdir(parents=True, exist_ok=False)
    commit = current_head()
    data = {"schema": "issue120-production-session-v1", "session_id": identity,
            "created_monotonic_ns": time.monotonic_ns(), "harness_commit": commit,
            "host_boot_id_sha256": boot_fingerprint(),
            "program_identity": runtime.program_identity(), "image": runtime.PINNED_IMAGE,
            "admission_hashes": admission_hashes(specs[0].phase),
            "runs": {s.identity: {"configuration": plan.program_configuration(s),
                                   "configuration_sha256": plan.configuration_digest(plan.program_configuration(s))}
                     for s in specs}}
    if specs[0].phase == "production" and set(data["admission_hashes"]) == {"production"}:
        data["production_contract"] = runtime.PRODUCTION_CONTRACT
    if fixture_only:
        data["fixture_only"] = True
    with (directory / "manifest.json").open("x") as output:
        json.dump(data, output, sort_keys=True, indent=2)
        output.write("\n")
    return data


def admission_hashes(phase: str) -> dict:
    if phase == "production" and compatible_production_receipt() is not None:
        return {"production": runtime.sha(runtime.SOURCE / "production_admission.json")}
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
    _validate_receipt_identity(kind, receipt)
    return receipt


def _validate_receipt_identity(kind: str, receipt: dict) -> None:
    current = runtime.program_identity()
    if kind == "production":
        from . import production_admission
        if receipt.get("schema") == production_admission.SCHEMA:
            production_admission.validate_receipt(receipt, current)
            return
    if kind == "bounded" and receipt.get("schema") == C2_SCHEMA:
        validate_c2_admission(receipt, current)
        return
    if kind == "bounded" and receipt.get("schema") == admission_bridge.SCHEMA:
        admission_bridge.validate_bridge(receipt, current)
        return
    if receipt.get("schema") != f"issue120-{kind}-admission-v1":
        raise ValueError("stage admission schema differs")
    if receipt.get("program_identity") != current:
        raise ValueError("stage admission is for a different program")


def _records_from_receipt(receipt: dict) -> list[dict]:
    from .recording import load_validated_record
    from .records import SCHEMA

    refs = receipt.get("records")
    if receipt.get("schema") in (admission_bridge.SCHEMA, C2_SCHEMA):
        rows = bounded_evidence.load_records(refs)
        if any(row["program_identity"] != receipt.get("evidence_program_identity") for row in rows):
            raise ValueError("original B4 identity was relabeled")
        return rows
    if not isinstance(refs, list) or len(refs) != 2:
        raise ValueError("stage admission requires both artifact-bound records")
    records = []
    for ref in refs:
        relative = Path(ref["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("stage record reference is not repository-relative")
        path = runtime.fixed_path(runtime.ROOT / relative, MEASUREMENTS_ROOT)
        if runtime.sha(path) != ref["sha256"]:
            raise ValueError("stage record digest differs")
        record = load_validated_record(path)
        if record.get("schema") != SCHEMA:
            raise ValueError("new stage admission requires the current record version")
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


def compatible_production_receipt(bindings: dict | None = None) -> dict | None:
    from . import production_admission

    try:
        receipt = _committed_admission("production", bindings)
    except ValueError:
        if not (runtime.SOURCE / "production_admission.json").is_file():
            return None
        raise
    return receipt if receipt.get("schema") == production_admission.SCHEMA else None


def admitted_specs(unit: str, bindings: dict | None = None) -> list[plan.RunSpec]:
    if unit == "P":
        from . import production_admission
        receipt = compatible_production_receipt(bindings)
        if receipt is not None:
            return production_admission.specs(receipt)
    return _legacy_admitted_specs(unit, bindings)


def _legacy_admitted_specs(unit: str, bindings: dict | None = None) -> list[plan.RunSpec]:
    if unit == "B":
        return list(plan.RUNS[:2])
    bounded = _committed_admission("bounded", bindings)
    rows = _records_from_receipt(bounded)
    if (bounded.get("operator_watcher_reviewed") is not True or
            not all(_healthy_bounded(r, s) for r, s in zip(rows, plan.RUNS[:2], strict=True)) or
            rows[1]["coverage_analysis"]["coverage"] != "covered"):
        raise ValueError("healthy first-kernel Watcher admission missing")
    if bounded.get("schema") == C2_SCHEMA:
        if unit != "C":
            raise ValueError("C2-only admission does not permit P")
        return [plan.RUNS[3]]
    if bounded.get("schema") == admission_bridge.SCHEMA:
        if unit != "C" or bounded.get("allowed_plan_identities") != [admission_bridge.C1]:
            raise ValueError("host-only B4 admission permits C1, not C2 or P")
        return [plan.run_spec(admission_bridge.C1)]
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


# These pins name original C1 evidence, not a current-identity replacement.
C2_SCHEMA = "issue120-bounded-admission-v3"
C1_DIRECTORY = "docs/measurements/issue120-production/2026-11-21-c1"
C1_ATTEMPT = "20261121T0107019969389561539527928Z"
C1_HARNESS = "b38095571dd62742a161662e5d166fafa75f5b41"
C1_RECORD_SHA = "7ec089657d949849465b67cdb2df38dea17b71c16e433866462822bd35571609"
C1_ARCHIVE_SHA = "e76ea42e1b7712f530e25e96eaf45ac5b3eaeae704768f9f80c76133537f6865"
C1_MANIFEST_SHA = "849e126f36c6982dbfb39632d7cde094ba98b9cc72f5297406f0f6b2e776955c"
C1_SESSION_SHA = "b6d06c027ec2854f6d2350492e8121989c33fe02c766e3b61aed68162deeb342"
C1_ADMISSION_PATH = "docs/measurements/issue120-production-c2-prep/c1-admission.json"
C1_ADMISSION_SHA = "9d18d931b03bc4371933cc24a2894002430a4b720aade24a05f4864f785dd9b7"
C2_CHANGED_PYTHON = {"runtime.py", "sessions.py", "recording.py"}
STABLE_ENVIRONMENT = ("kernel", "kmd_version", "tt_env_active_release", "image", "image_pinned",
                      "telemetry_sampler", "board_info", "firmwares", "limits", "host_sw_vers",
                      "smbus_telem_limits", "device_node")


def stable_calibration_environment(raw: dict) -> dict:
    if (raw.get("kernel") != "Linux 6.8.0-149-generic" or raw.get("kmd_version") != "2.12.0"
            or raw.get("image") != runtime.PINNED_IMAGE or raw.get("device_node") != "/dev/tenstorrent/0"):
        raise ValueError("calibration baseline/image/device differs")
    if any(key not in raw for key in STABLE_ENVIRONMENT):
        raise ValueError("calibration static environment is incomplete")
    return {key: raw[key] for key in STABLE_ENVIRONMENT}


def c1_evidence_reference() -> dict:
    return {"archive": C1_DIRECTORY + "/C1-evidence.tar.gz", "archive_sha256": C1_ARCHIVE_SHA,
            "manifest_sha256": C1_MANIFEST_SHA, "record_sha256": C1_RECORD_SHA,
            "session_sha256": C1_SESSION_SHA, "harness_commit": C1_HARNESS,
            "original_admission_sha256": C1_ADMISSION_SHA}


def _c1_member_names() -> tuple[str, ...]:
    return (f"env-{C1_ATTEMPT}.json", f"power-{C1_ATTEMPT}.csv",
            f"power-trace-snapshot-{C1_ATTEMPT}-1.csv", *(f"results-{C1_ATTEMPT}{suffix}" for suffix in
            (".json", "-frames.bin", "-transitions.bin", "-native-result.json", "-native-execution.json",
             "-native-stdout.bin", "-native-stderr.bin")))


def _copy_c1_archive(archive, directory: Path, facts: dict) -> None:
    members, total = {}, 0
    for member in archive:
        if not member.isfile() or member.name not in facts or member.name in members:
            raise ValueError("C1 archive member differs")
        total += member.size
        if not 0 <= member.size <= bounded_evidence.MAX_BYTES or total > bounded_evidence.MAX_BYTES:
            raise ValueError("C1 archive exceeds its extraction bound")
        members[member.name] = member
    if set(members) != set(_c1_member_names()) or set(facts) != set(members):
        raise ValueError("C1 archive must contain its ten pinned payloads")
    # Only independently pinned names reach the filesystem, never archive metadata.
    for name in _c1_member_names():
        path = directory / name
        with archive.extractfile(members[name]) as source, path.open("xb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        if path.stat().st_size != facts[name]["byte_count"] or runtime.sha(path) != facts[name]["sha256"]:
            raise ValueError("C1 payload fingerprint differs")


def c1_calibration_evidence() -> tuple[dict, dict]:
    from .recording import load_validated_record

    directory = runtime.fixed_path(runtime.ROOT / C1_DIRECTORY, MEASUREMENTS_ROOT)
    checks = (("C1-evidence.tar.gz", C1_ARCHIVE_SHA), ("C1-manifest.json", C1_MANIFEST_SHA),
              ("runtime-session-manifest.json", C1_SESSION_SHA))
    for name, digest in checks:
        if runtime.sha(runtime.fixed_path(directory / name, directory)) != digest:
            raise ValueError("original C1 evidence pin differs")
    session = json.loads((directory / "runtime-session-manifest.json").read_bytes())
    manifest = json.loads((directory / "C1-manifest.json").read_bytes())
    facts = {item["file"]: item for item in manifest["artifacts"]}
    with tempfile.TemporaryDirectory(prefix="issue120-c1-compat-") as tmp:
        output = Path(tmp)
        with tarfile.open(directory / "C1-evidence.tar.gz", "r:gz") as archive:
            _copy_c1_archive(archive, output, facts)
        path = output / f"results-{C1_ATTEMPT}.json"
        if runtime.sha(path) != C1_RECORD_SHA:
            raise ValueError("original C1 primary differs")
        row = load_validated_record(path)
        raw = json.loads((output / f"env-{C1_ATTEMPT}.json").read_bytes())
    if (row["environment"]["commit"] != C1_HARNESS or row["program_identity"] != session["program_identity"]
            or row["session_manifest_sha256"] != C1_SESSION_SHA or raw.get("harness_commit") != C1_HARNESS):
        raise ValueError("original C1 provenance was relabeled")
    _healthy_calibration(row, plan.RUNS[2], allow_not_covered=False)
    stable_calibration_environment(raw)
    return row, raw


def _healthy_calibration(row: dict, spec: plan.RunSpec, *, allow_not_covered: bool) -> None:
    from . import recording

    accepted = ("covered", "not_covered") if allow_not_covered else ("covered",)
    flags = (row.get("resident_integrity_passed"), row.get("warning_check", {}).get("passed"),
             row.get("watcher", {}).get("scan_passed"), row.get("power_trace_snapshot", {}).get("coverage_complete"))
    if (row.get("schema") != "issue120-production-v2" or row.get("phase") != "calibration"
            or row.get("condition") != "loaded" or row.get("plan_identity") != spec.identity
            or row.get("configuration") != plan.program_configuration(spec)
            or row.get("coverage_analysis", {}).get("coverage") not in accepted
            or row.get("watcher", {}).get("requested") is not False or not all(flag is True for flag in flags)
            or not recording._valid_v2_weighting(row.get("power_trace_snapshot", {}))):
        raise ValueError("calibration is not healthy artifact-validated diagnostic evidence")
    metrics = row["metrics"]
    if any(metrics.get(key) != spec.frame_count for key in ("frames_consumed", "frames_produced", "frames_attempted")):
        raise ValueError("calibration frames are incomplete")
    if any(metrics.get(key) != 0 for key in ("frames_aborted", "frames_dropped", "overflow_count")):
        raise ValueError("calibration frame health differs")


def _python_transition(old: dict, current: dict) -> dict:
    before, after = old["python_sources_sha256"], current["python_sources_sha256"]
    actual = runtime.python_source_hashes()
    if set(before) != set(after) or after != actual:
        raise ValueError("C2 Python source set/target differs")
    changed = {name: {"before": before[name], "after": after[name]} for name in before if before[name] != after[name]}
    if set(changed) - C2_CHANGED_PYTHON:
        raise ValueError("C2 changed non-metadata source")
    expected = copy.deepcopy(old)
    expected.update(python_sources_sha256=after, calibration_compatibility_contract=runtime.CALIBRATION_CONTRACT)
    if expected != current:
        raise ValueError("C2 native/device/trace identity changed")
    return {"before": before, "after": after, "changed": changed,
            "scope": "host metadata admission/provenance/offline analysis only; native/device unchanged"}


def _c2_fields(current: dict, c1: dict, raw: dict) -> dict:
    path = runtime.fixed_path(runtime.ROOT / C1_ADMISSION_PATH, MEASUREMENTS_ROOT)
    if runtime.sha(path) != C1_ADMISSION_SHA:
        raise ValueError("C1 original admission was modified")
    original = json.loads(path.read_bytes())
    if original["program_identity"] != c1["program_identity"] or original["schema"] != admission_bridge.SCHEMA:
        raise ValueError("C1 original admission identity differs")
    return {"schema": C2_SCHEMA, "program_identity": copy.deepcopy(current), "image": runtime.PINNED_IMAGE,
            "evidence_program_identity": bounded_evidence.evidence_identity(),
            "records": bounded_evidence.references(), "operator_watcher_reviewed": True,
            "allowed_plan_identities": [plan.RUNS[3].identity], "execution_authorized": False,
            "c1_evidence": c1_evidence_reference(), "c1_program_identity": c1["program_identity"],
            "python_transition": _python_transition(c1["program_identity"], current),
            "stable_environment": stable_calibration_environment(raw),
            "host_only_source_proof": admission_bridge._source_proof(),
            "work_units": {"cpu_operations_per_write": 32768, "bytes_per_write": 4096},
            "configuration_sha256": {s.identity: plan.configuration_digest(plan.program_configuration(s)) for s in plan.RUNS[2:4]},
            "harness_binding": "retain original C1 commit; target binds fresh session commit and exact source maps",
            "not_covered_policy": "diagnostic rates/costs only; no coverage or production promotion",
            "offline_tools_sha256": {name: runtime.sha(runtime.fixed_path(runtime.ROOT / name, runtime.ROOT / "tools"))
                                     for name in ("tools/prepare_issue120_c2_admission.py", "tools/analyze_issue120_calibration_pair.py")}}


def make_c2_admission(current: dict) -> dict:
    c1, raw = c1_calibration_evidence()
    receipt = _c2_fields(current, c1, raw)
    rows = bounded_evidence.load_records(receipt["records"])
    if not all(_healthy_bounded(r, s) for r, s in zip(rows, plan.RUNS[:2], strict=True)):
        raise ValueError("C2 needs original healthy B4 Watcher evidence")
    return receipt


def validate_c2_admission(receipt: dict, current: dict) -> None:
    c1, raw = c1_calibration_evidence()
    if receipt != _c2_fields(current, c1, raw):
        raise ValueError("C2-only receipt/source/provenance differs")
