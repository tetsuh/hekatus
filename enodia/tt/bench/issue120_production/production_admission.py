"""Versioned production eligibility from original compatible calibration evidence.

No device access or current readiness is granted here; v2 records dated owner authority.
Legacy admissions remain in sessions with their original exact-identity rules.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import tarfile
import tempfile
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from . import artifacts, bounded_evidence, plan, records, runtime, sessions

SCHEMA = "issue120-compatible-production-admission-v1"
ADOPTED_SCHEMA = "issue120-compatible-production-admission-v2"
ADOPTION = runtime.SOURCE / "production_adoption.json"
SELECTION = runtime.SOURCE / "production_selection.json"
POSTPROCESS = runtime.ROOT / "docs/measurements/issue120-production-p-prep/postprocess.json"
C2_DIRECTORY = sessions.MEASUREMENTS_ROOT / "issue120-production/2026-11-21-c2"
C2_ATTEMPT = "20261121T06165291396130455596811034Z"
C2_SESSION = "1d8690df2f1d40249d1e8f2aaa6b19fb"
C2_ARCHIVE_SHA = "4277603473c2fcf50eb27483d36c725a64d1f3f5c5cd96751079b1a903a6b904"
C2_MANIFEST_SHA = "308685312483379b70f53e9d3cc164fd396855d9186e407cfd3e86e5d343a058"
C2_SESSION_SHA = "8a235154c995a5d36f6e500014c681d2de6ffefb8a088d7c71afb9833cbffb4f"
C2_RECORD_SHA = "d26e5668b95fa8d21d924bd339e6c76b45caaeef6e38e8c7e2a047b69c5462fb"
C2_RECEIPT_SHA = "c60f0fbafe3f003b30e9974e19ab7ad51d9f84908995ecb64b452bea41cad185"
CHANGED_PYTHON = {"runtime.py", "sessions.py", "recording.py", "runner.py", "production_admission.py"}


def selection() -> dict:
    value = json.loads(SELECTION.read_bytes())
    if (value.get("schema") != "issue120-production-selection-v1" or value.get("execution_authorized") is not False
            or type(value.get("target_worker_seconds")) not in (float, int)
            or not math.isfinite(value["target_worker_seconds"]) or value["target_worker_seconds"] <= 0
            or type(value.get("fixed_repetitions")) is not int or value["fixed_repetitions"] <= 0):
        raise ValueError("production selection must be a finite non-authorizing recommendation")
    return value


def _adoption_record(reference: dict, selected: dict) -> dict:
    keys = {"file", "text", "sha256", "byte_count", "selection_text"}
    if set(reference) != keys or reference["file"] != "enodia/tt/bench/issue120_production/production_adoption.json":
        raise ValueError("owner adoption reference differs")
    raw = reference["text"].encode("utf-8")
    if len(raw) != reference["byte_count"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("owner adoption bytes differ")
    data = json.loads(raw)
    selection_raw = reference["selection_text"].encode("utf-8")
    if json.loads(selection_raw) != selected:
        raise ValueError("owner adoption selection differs")
    expected = {"schema": "issue120-owner-production-adoption-v1", "decision_date": data.get("decision_date"),
                "target_worker_seconds": selected["target_worker_seconds"], "fixed_repetitions": selected["fixed_repetitions"],
                "selection_file_sha256": hashlib.sha256(selection_raw).hexdigest(),
                "scope": {s.identity: 1 for s in plan.RUNS[4:]},
                "conditional_loaded_requires_healthy_control": True, "same_session_required": True,
                "automatic_retry_allowed": False}
    if data != expected or type(data.get("fixed_repetitions")) is not int:
        raise ValueError("owner adoption selection/date/scope differs")
    _adoption_scope_date(data)
    return data


def _adoption_scope_date(data: dict) -> None:
    day = date.fromisoformat(data["decision_date"])
    if day.isoformat() != data["decision_date"] or day > datetime.now(UTC).date():
        raise ValueError("owner adoption decision date is invalid or future")
    if (any(type(n) is not int for n in data["scope"].values())
            or data["conditional_loaded_requires_healthy_control"] is not True
            or data["same_session_required"] is not True or data["automatic_retry_allowed"] is not False):
        raise ValueError("owner adoption must authorize exactly one conditional pair")


def owner_adoption() -> dict:
    raw = ADOPTION.read_bytes()
    reference = {"file": "enodia/tt/bench/issue120_production/production_adoption.json", "text": raw.decode("utf-8"),
                 "sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw), "selection_text": SELECTION.read_text()}
    _adoption_record(reference, selection())
    return reference


def _names(case: str) -> tuple[str, ...]:
    attempt = sessions.C1_ATTEMPT if case == "C1" else C2_ATTEMPT
    common = (f"env-{attempt}.json", f"power-{attempt}.csv", f"power-trace-snapshot-{attempt}-1.csv",
              *(f"results-{attempt}{s}" for s in (".json", "-frames.bin", "-transitions.bin",
                                                 "-native-result.json", "-native-execution.json", "-native-stdout.bin", "-native-stderr.bin")))
    return common if case == "C1" else (*common, f"session-{C2_SESSION}.json")


def _extract_members(case: str, directory: Path, output: Path, expected: dict) -> None:
    names = _names(case)
    with tarfile.open(directory / f"{case}-evidence.tar.gz") as bundle:
        members = bundle.getmembers()
        if (len(members) != len(names) or {m.name for m in members} != set(names)
                or not all(m.isfile() for m in members) or sum(m.size for m in members) > bounded_evidence.MAX_BYTES):
            raise ValueError("original calibration archive layout differs")
        for name in names:
            raw = bundle.extractfile(name).read()
            path = output / name
            path.write_bytes(raw)
            if len(raw) != expected[name]["byte_count"] or runtime.sha(path) != expected[name]["sha256"]:
                raise ValueError("original calibration payload fingerprint differs")


def _extract(case: str, output: Path) -> tuple[dict, dict]:
    from .recording import load_validated_record

    directory = runtime.ROOT / sessions.C1_DIRECTORY if case == "C1" else C2_DIRECTORY
    pins = ((sessions.C1_ARCHIVE_SHA, sessions.C1_MANIFEST_SHA, sessions.C1_SESSION_SHA, sessions.C1_RECORD_SHA)
            if case == "C1" else (C2_ARCHIVE_SHA, C2_MANIFEST_SHA, C2_SESSION_SHA, C2_RECORD_SHA))
    for name, digest in zip((f"{case}-evidence.tar.gz", f"{case}-manifest.json", "runtime-session-manifest.json"), pins[:3], strict=True):
        if runtime.sha(runtime.fixed_path(directory / name, directory)) != digest:
            raise ValueError("original calibration archive/session fingerprint differs")
    manifest = json.loads((directory / f"{case}-manifest.json").read_bytes())
    expected = {item["file"]: item for item in manifest["artifacts"]}
    _extract_members(case, directory, output, expected)
    attempt = sessions.C1_ATTEMPT if case == "C1" else C2_ATTEMPT
    path = output / f"results-{attempt}.json"
    if runtime.sha(path) != pins[3]:
        raise ValueError("original calibration primary differs")
    row = load_validated_record(path)
    spec = plan.RUNS[2] if case == "C1" else plan.RUNS[3]
    sessions._healthy_calibration(row, spec, allow_not_covered=False)
    if not row["execution_eligible"] or not row["timing_evidence"] or row["metrics"]["paired_outlier_count"]:
        raise ValueError("production needs healthy covered calibration, not diagnostic-only evidence")
    return row, {"archive_sha256": pins[0], "manifest_sha256": pins[1], "session_sha256": pins[2],
                 "primary_sha256": pins[3], "environment_sha256": expected[f"env-{attempt}.json"]["sha256"],
                 "harness_commit": row["environment"]["commit"], "program_identity": row["program_identity"]}


def _row_facts(row: dict, output: Path, case: str) -> dict:
    attempt = sessions.C1_ATTEMPT if case == "C1" else C2_ATTEMPT
    native = row["native"]
    trace = native["host_phase_trace"]
    slabs = []
    for before, after in zip(trace["samples"], trace["samples"][1:]):
        writes = after["committed_count"] - before["committed_count"]
        seconds = (after["monotonic_ns"] - before["monotonic_ns"]) / 1e9
        slabs.append({"writes": writes, "seconds": seconds, "writes_per_second": writes / seconds})
    frames = artifacts.read_public_evidence(output, attempt, "frames", frames=native["frame_count"], emitted=native["repeat_count"])
    first, last = None, None
    for frame in frames.rows():
        stamp = artifacts.ticks(frame, 0)
        if first is None:
            first = stamp
        last = stamp
    span = (last - first) / (row["clock"]["observed_host_envelope_weighted_aiclk_mhz"] * 1e6)
    host = (native["program_host_interval"]["end_ns"] - native["program_host_interval"]["start_ns"]) / 1e9
    raw = json.loads((output / f"env-{attempt}.json").read_bytes())
    return {"baseline": sessions.stable_calibration_environment(raw), "slabs": slabs, "lead_seconds": (trace["samples"][0]["monotonic_ns"] - trace["worker_start_ns"]) / 1e9,
            "projected_frame_envelope_seconds": span / (native["frame_count"] - 1) * 499999 + host - span,
            "overhead": row["overhead"]}


def calibration_evidence() -> tuple[dict, dict, list[dict]]:
    refs, rows, facts = {}, [], []
    with tempfile.TemporaryDirectory(prefix="issue120-production-proof-") as tmp:
        for case in ("C1", "C2"):
            output = Path(tmp) / case
            output.mkdir()
            row, refs[case] = _extract(case, output)
            rows.append(row)
            facts.append(_row_facts(row, output, case))
    before = copy.deepcopy(rows[0]["program_identity"])
    after = rows[1]["program_identity"]
    before.update(python_sources_sha256=after["python_sources_sha256"], calibration_compatibility_contract=runtime.CALIBRATION_CONTRACT)
    changed = {k for k, v in rows[0]["program_identity"]["python_sources_sha256"].items() if after["python_sources_sha256"].get(k) != v}
    if (before != after or changed - sessions.C2_CHANGED_PYTHON or facts[0]["baseline"] != facts[1]["baseline"]
            or {k: v for k, v in rows[0]["environment"].items() if k != "commit"}
            != {k: v for k, v in rows[1]["environment"].items() if k != "commit"}):
        raise ValueError("original C1/C2 compatibility changed")
    if runtime.sha(runtime.SOURCE / "bounded_admission.json") != C2_RECEIPT_SHA:
        raise ValueError("original C2-only receipt changed")
    bounded = bounded_evidence.load_records(bounded_evidence.references())
    if not all(sessions._healthy_bounded(row, spec) for row, spec in zip(bounded, plan.RUNS[:2], strict=True)):
        raise ValueError("production needs the original healthy B4 Watcher evidence")
    return refs, rows[1]["calibration_provenance"]["stable_environment"], facts


def calibration_summary() -> dict:
    """Explicit frozen-evidence path, never current C2 execution admission."""
    refs, _, _ = calibration_evidence()
    return {"records": [{"harness_commit": refs[case]["harness_commit"]} for case in ("C1", "C2")],
            "production_coverage_eligible": True, "execution_authorized": False,
            "current_550s_floor_unchanged": True}


def python_transition(old: dict, current: dict, *, live: bool) -> dict:
    before, after = old["python_sources_sha256"], current["python_sources_sha256"]
    changed = {k: {"before": before.get(k), "after": after.get(k)} for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    expected = copy.deepcopy(old)
    expected.update(python_sources_sha256=after, production_admission_contract=runtime.PRODUCTION_CONTRACT)
    if (expected != current or set(changed) - CHANGED_PYTHON or set(before) - set(after)
            or live and after != runtime.python_source_hashes()):
        raise ValueError("production changed native/device or unreviewed metadata source")
    return {"before": before, "after": after, "changed": changed,
            "scope": "versioned production admission and record metadata only; native/device unchanged"}


def fixture_identity() -> dict:
    """Assume archived native only for CPU fixtures, never live native admission."""
    value = copy.deepcopy(json.loads((C2_DIRECTORY / "runtime-session-manifest.json").read_bytes())["program_identity"])
    value.update(python_sources_sha256=runtime.python_source_hashes(), production_admission_contract=runtime.PRODUCTION_CONTRACT)
    return value


def postprocess_proof() -> dict:
    path = runtime.fixed_path(POSTPROCESS, sessions.MEASUREMENTS_ROOT)
    report = json.loads(path.read_bytes())
    if (report.get("schema") != "issue120-production-host-postprocess-v1" or report.get("fixture_only") is not True
            or report.get("frames") != 500000 or report.get("passed") is not True or report.get("device_retrieval_measured") is not False):
        raise ValueError("500000-frame device-free host postprocessing proof missing")
    if (report.get("measurement_pending") is not True
            and (report.get("fixture_source_sha256") != runtime.sha(runtime.ROOT / "tools/issue120_production_fixture.py")
                 or report.get("measured_python_sources_sha256") != runtime.python_source_hashes())):
        raise ValueError("postprocessing source differs from its measurement")
    return {"file": path.relative_to(runtime.ROOT).as_posix(), "sha256": runtime.sha(path), "report": report}


def _model(facts: list[dict], selected: dict, proof: dict) -> dict:
    slabs = [s for row in facts for s in row["slabs"]]
    fast = max(s["writes_per_second"] for s in slabs) * 1.02
    slow = min(s["writes_per_second"] for s in slabs) * .98
    lf, ls = min(f["lead_seconds"] for f in facts), max(f["lead_seconds"] for f in facts)
    envelope = max(f["projected_frame_envelope_seconds"] for f in facts)
    count = 1 + math.ceil(fast * (selected["target_worker_seconds"] - lf))
    if count != selected["fixed_repetitions"] or selected["target_worker_seconds"] <= envelope:
        raise ValueError("selected N/target does not match explicit model rounding/coverage")
    report = proof["report"]
    host = report["charged_host_seconds"]
    if type(host) not in (int, float) or not math.isfinite(host) or host < 0 or type(report["peak_rss_bytes"]) is not int:
        raise ValueError("postprocessing measurement is not finite")
    components = {"measured_charged_host_seconds": host, "setup_scenario_seconds": max(f["overhead"]["setup_seconds"] for f in facts),
                  "snapshot_wait_scenario_seconds": max(f["overhead"]["sampler_wait_seconds"] for f in facts),
                  "unmeasured_native_2GB_retrieval_budget_seconds": 10.0, "CLI_residual_and_guard_seconds": 5.0}
    optimistic = max(20.0, sum(components.values()))
    pessimistic = max(36.873251197606606, optimistic + 10.0)
    lo, hi = lf + (count - 1) / fast, ls + (count - 1) / slow
    return {"slabs": slabs, "r_fast": fast, "r_slow": slow, "fraction": .02, "L_fast_seconds": lf, "L_slow_seconds": ls,
            "frame_envelope_seconds": envelope, "formula": "N=1+ceil(r_fast*(target-L_fast))", "rounding": "ceil;first observed write counted once",
            "reserve": {"components": components, "optimistic_seconds": optimistic, "pessimistic_seconds": pessimistic,
                        "launcher_reread_outside_cap_seconds": report["launcher_reread_seconds"], "native_retrieval_measured": False},
            "fast_worker_seconds": lo, "slow_worker_seconds": hi, "frame_margin_seconds": lo - envelope,
            "cap_margin_optimistic_seconds": 600 - hi - optimistic, "cap_margin_pessimistic_seconds": 600 - hi - pessimistic,
            "rejected_models": ["negative native per-frame fit", "whole enqueue API as dead-worker lead", "inclusive means as slab envelope", "two-point physical fixed lead"],
            "bounds_established": False}


def make_receipt(current: dict, *, selected: dict | None = None, proof: dict | None = None,
                 live: bool = True, adoption: dict | None = None) -> dict:
    refs, baseline, facts = calibration_evidence()
    selected = selection() if selected is None else selected
    proof = postprocess_proof() if proof is None else proof
    transition = python_transition(refs["C2"]["program_identity"], current, live=live)
    model = _model(facts, selected, proof)
    build = runtime.ROOT / "docs/measurements/issue120-production-c1-prep-checks.json"
    build_proof = json.loads(build.read_bytes())
    receipt = {"schema": SCHEMA, "program_identity": copy.deepcopy(current), "selection": copy.deepcopy(selected),
            "selection_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
            "fixed_repetitions": selected["fixed_repetitions"], "model": model, "postprocess_proof": proof,
            "calibrations": refs, "baseline": baseline, "python_transition": transition,
            "build_proof_sha256": runtime.sha(build), "build_receipt": build_proof["build_receipt"], "image": runtime.PINNED_IMAGE,
            "bounded_evidence": bounded_evidence.references(), "native_device_unchanged": True,
            "allowed_plan_identities": [s.identity for s in plan.RUNS[4:]], "frames": 500000, "cap_seconds": 600,
            "watcher": False, "device_node": "/dev/tenstorrent/0", "owner_adopted": False, "execution_authorized": False}
    if adoption is not None:
        _adoption_record(adoption, selected)
        receipt.update(schema=ADOPTED_SCHEMA, owner_adopted=True, execution_authorized=True,
                       owner_decision=copy.deepcopy(adoption))
    return receipt


def validate_receipt(receipt: dict, current: dict, *, live: bool = True) -> None:
    selected = selection() if live else receipt["selection"]
    proof = postprocess_proof() if live else receipt["postprocess_proof"]
    adoption = None
    if receipt.get("schema") == ADOPTED_SCHEMA:
        adoption = owner_adoption() if live else receipt["owner_decision"]
    if receipt != make_receipt(current, selected=selected, proof=proof, live=live, adoption=adoption):
        raise ValueError("compatible production receipt/provenance/selection differs")


def specs(receipt: dict) -> list[plan.RunSpec]:
    if (receipt["model"]["cap_margin_pessimistic_seconds"] <= 0
            or receipt["postprocess_proof"]["report"].get("measurement_pending") is True):
        raise ValueError("production requires measured host costs and a feasible explicit cap model")
    return [replace(s, repeat_count=receipt["fixed_repetitions"]) for s in plan.RUNS[4:]]


def _hash_field(value) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("production fingerprint is missing/malformed")
    return value


def _captured_json(output: Path, name: str, digest: str, source: Path | None) -> tuple[dict, dict]:
    from .recording import copy_exclusive

    path = runtime.fixed_path(output / name, output, exists=source is None)
    if source is not None:
        copy_exclusive(source, path)
    if path.stat().st_size > 4 * 1024 * 1024 or runtime.sha(path) != _hash_field(digest):
        raise ValueError("production captured JSON fingerprint differs")
    return json.loads(path.read_bytes()), {"file": name, "sha256": digest, "byte_count": path.stat().st_size}


def _session_bindings(manifest: dict, record: dict, receipt: dict, receipt_sha: str) -> None:
    configurations = {s.identity: {"configuration": plan.program_configuration(replace(s, repeat_count=receipt["fixed_repetitions"])),
                                  "configuration_sha256": plan.configuration_digest(plan.program_configuration(replace(s, repeat_count=receipt["fixed_repetitions"])))}
                      for s in plan.RUNS[4:]}
    expected = {"schema": "issue120-production-session-v1", "session_id": record["session_id"],
                "harness_commit": record["environment"]["commit"], "program_identity": record["program_identity"],
                "host_boot_id_sha256": record["host_boot_id_sha256"], "image": runtime.PINNED_IMAGE,
                "production_contract": runtime.PRODUCTION_CONTRACT, "admission_hashes": {"production": receipt_sha}, "runs": configurations}
    if receipt["schema"] == ADOPTED_SCHEMA:
        expected["owner_adoption_sha256"] = receipt["owner_decision"]["sha256"]
    if (any(manifest.get(k) != v for k, v in expected.items())
            or bool(manifest.get("fixture_only")) != bool(record.get("fixture_only"))):
        raise ValueError("production captured session binding differs")
    created = manifest.get("created_monotonic_ns")
    if type(created) is not int or not 0 < created <= record["execution_interval"]["start_ns"]:
        raise ValueError("production session creation is outside its run origin")
    spec = plan.run_spec(record.get("plan_identity"))
    if (spec.phase != "production" or record.get("phase") != spec.phase or record.get("condition") != spec.condition
            or record.get("configuration") != configurations[spec.identity]["configuration"]):
        raise ValueError("production plan/configuration declaration differs")


def environment_identity(raw: dict, manifest: dict) -> dict:
    board, firmware = raw["board"], raw["firmware"]
    cpu = manifest.get("host_cpu_model")
    if not isinstance(cpu, str) or not cpu.strip():
        raise ValueError("production session CPU identity is missing")
    return {"board_model": str(board.get("board_type")),
            "serial": runtime.issue120_probe.normalize_board_id(board.get("serial") or board.get("board_id")),
            "pci": runtime.issue120_probe.normalize_pci_bus_id(board.get("bus_id")),
            "firmware": str(firmware.get("fw_bundle_version")), "kmd": str(raw.get("kmd_version")),
            "image_digest": str(raw["image"]).rsplit("@", 1)[-1], "cpu_model": cpu, "commit": raw["harness_commit"]}


def record_provenance(output: Path, record: dict, receipt_sha: str, *, publish: bool = False) -> dict:
    """Authenticate actual immutable session/receipt/host-proof/env companions."""
    identity = sessions.canonical_session_id(record["session_id"])
    receipt_sha = _hash_field(receipt_sha)
    source = runtime.SOURCE / "production_admission.json" if publish else None
    receipt, receipt_ref = _captured_json(output, f"production-receipt-{receipt_sha}.json", receipt_sha, source)
    validate_receipt(receipt, record["program_identity"], live=False)
    if receipt["postprocess_proof"]["report"].get("measurement_pending") and not record.get("fixture_only"):
        raise ValueError("pending production proof is fixture-only")
    digest = _hash_field(record.get("session_manifest_sha256"))
    source = sessions.session_path(identity) / "manifest.json" if publish else None
    manifest, session_ref = _captured_json(output, f"session-{identity}.json", digest, source)
    _session_bindings(manifest, record, receipt, receipt_sha)
    proof = receipt["postprocess_proof"]
    source = POSTPROCESS if publish else None
    host, host_ref = _captured_json(output, f"production-host-proof-{_hash_field(proof['sha256'])}.json", proof["sha256"], source)
    if host != proof["report"]:
        raise ValueError("production host-cost proof differs")
    # Reuse only the raw-environment checks, not C2's sole-run session contract.
    env_path = runtime.fixed_path(output / f"env-{record['attempt_identity']}.json", output)
    if env_path.stat().st_size > 1024 * 1024:
        raise ValueError("production environment exceeds its capture bound")
    raw = json.loads(env_path.read_bytes())
    fixture_flags = (record.get("fixture_only"), record["native"].get("fixture_only"),
                     manifest.get("fixture_only"), raw.get("fixture_only"))
    if any(flag is True for flag in fixture_flags) and not all(flag is True for flag in fixture_flags):
        raise ValueError("production fixture declarations disagree with the captured native/session/environment")
    if (raw.get("harness_commit") != record["environment"]["commit"] or raw.get("run_id") != record["attempt_identity"]
            or raw.get("harness_dirty") is not False or raw.get("harness_pushed") is not True
            or raw.get("board") != raw.get("board_info") or raw.get("firmware") != raw.get("firmwares")
            or sessions.stable_calibration_environment(raw) != receipt["baseline"]
            or record["environment"] != environment_identity(raw, manifest)):
        raise ValueError("production environment/harness/baseline differs")
    return {"schema": runtime.PRODUCTION_CONTRACT, "receipt": receipt_ref, "session": session_ref,
            "postprocess": host_ref, "environment": {"file": env_path.name, "sha256": runtime.sha(env_path), "byte_count": env_path.stat().st_size}}


def control_release(control: dict, identity: str) -> None:
    manifest, digest = sessions.read_session(identity)
    expected = manifest.get("runs", {}).get(plan.RUNS[4].identity)
    if expected is None:
        raise ValueError("production control session has no admitted role")
    expected = records.mapping(expected, "production control role")
    roles = {"schema": "issue120-production-v2", "phase": "production", "condition": "control",
             "plan_identity": plan.RUNS[4].identity, "configuration": expected["configuration"],
             "configuration_sha256": expected["configuration_sha256"], "host_boot_id_sha256": manifest["host_boot_id_sha256"],
             "resident_integrity_passed": True, "execution_eligible": True, "timing_evidence": True}
    if (any(control.get(key) != value for key, value in roles.items())
            or control.get("session_id") != identity or control.get("session_manifest_sha256") != digest
            or control.get("program_identity") != manifest["program_identity"]
            or control.get("environment", {}).get("commit") != manifest["harness_commit"]
            or control.get("metrics", {}).get("paired_outlier_count") != 0
            or control.get("native", {}).get("frame_count") != 500000
            or control.get("coverage_analysis", {}).get("coverage") != "not_applicable"):
        raise ValueError("production control role/origin or paired-clock anomalies require owner review")


def recheck_record_provenance(output: Path, record: dict) -> None:
    value = record.get("production_provenance")
    if (record.get("program_identity", {}).get("production_admission_contract") != runtime.PRODUCTION_CONTRACT
            or not isinstance(value, dict)):
        raise ValueError("production declared provenance is missing")
    if ((record.get("fixture_only") or record.get("native", {}).get("fixture_only"))
            and output.is_relative_to(runtime.ROOT / "out/issue120-production/sessions")):
        raise ValueError("production fixture evidence cannot enter operational sessions")
    if record_provenance(output, record, value.get("receipt", {}).get("sha256")) != value:
        raise ValueError("production persisted provenance differs")
