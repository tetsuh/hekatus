"""Exact B4-to-host-trace compatibility for C1 only, not general native reuse."""
from __future__ import annotations

import copy

from . import bounded_evidence, host_trace, host_trace_contract, plan, runtime

SCHEMA = "issue120-bounded-admission-v2"
C1 = "production-v1-calibration-small"
UNCHANGED_IDENTITIES = ("library_sha256", "kernel_sources_sha256", "device_compile_proof_sha256", "compile_manifest_sha256")
CHANGED_PYTHON = {"recording.py", "runtime.py", "sessions.py"}
ADDED_PYTHON = {"host_trace.py", "host_trace_contract.py", "host_build_inputs.py", "bounded_evidence.py", "admission_bridge.py"}


def _source_proof() -> dict:
    base = (runtime.SOURCE / "native.cpp").read_text()
    return host_trace_contract.verify_native_source(host_trace_contract.derive_native(base))


def _python_compatibility(old: dict, current: dict) -> None:
    before = old["python_sources_sha256"]
    after = current["python_sources_sha256"]
    actual = {p.name: runtime.sha(p) for p in runtime.SOURCE.glob("*.py")}
    if after != actual or not set(before) <= set(after) or set(after) - set(before) != ADDED_PYTHON:
        raise ValueError("successor Python source set differs")
    for name, digest in before.items():
        if name not in CHANGED_PYTHON and after[name] != digest:
            raise ValueError("non-host source changed across B4 admission")


def validate_bridge(receipt: dict, current: dict) -> None:
    if (receipt.get("schema") != SCHEMA or receipt.get("allowed_plan_identities") != [C1] or
            receipt.get("execution_authorized") is not False or receipt.get("operator_watcher_reviewed") is not True):
        raise ValueError("bounded host-only admission must be C1-only evidence, not execution authority")
    if receipt.get("image") != runtime.PINNED_IMAGE or runtime.PINNED_IMAGE != bounded_evidence.evidence_image():
        raise ValueError("B4/current image identity differs")
    if receipt.get("program_identity") != current:
        raise ValueError("C1 admission is not bound to the actual successor")
    old = bounded_evidence.evidence_identity()
    if receipt.get("evidence_program_identity") != old or old["native_sha256"] != host_trace_contract.BASE_BINARY_SHA256:
        raise ValueError("B4 evidence identity was relabeled")
    if current["native_sha256"] == old["native_sha256"] or current.get("host_trace_contract") != host_trace.TRACE_SCHEMA:
        raise ValueError("new host trace requires its actual new native contract")
    if any(old[key] != current[key] for key in UNCHANGED_IDENTITIES):
        raise ValueError("device/library/CTA/compiler identity differs")
    _python_compatibility(old, current)
    proof = _source_proof()
    if receipt.get("host_only_source_proof") != proof:
        raise ValueError("host-only native source-pair proof differs")
    bounded_evidence.validate_references(receipt.get("records"))
    spec = plan.run_spec(C1)
    if (spec.frame_count, spec.repeat_count, spec.cap_seconds, spec.watcher, spec.condition) != (16384, 1048576, 60, False, "loaded"):
        raise ValueError("C1 work or Watcher policy changed")


def make_receipt(current: dict) -> dict:
    from .sessions import _healthy_bounded

    refs = bounded_evidence.references()
    rows = bounded_evidence.load_records(refs)
    if not all(_healthy_bounded(row, spec) for row, spec in zip(rows, plan.RUNS[:2], strict=True)):
        raise ValueError("B4 health does not permit C1 preparation")
    receipt = {"schema": SCHEMA, "program_identity": copy.deepcopy(current), "image": runtime.PINNED_IMAGE,
               "evidence_program_identity": bounded_evidence.evidence_identity(),
               "records": refs, "allowed_plan_identities": [C1], "operator_watcher_reviewed": True,
               "execution_authorized": False,
               "host_only_source_proof": _source_proof()}
    validate_bridge(receipt, current)
    return receipt
