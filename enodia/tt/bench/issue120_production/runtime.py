"""Pinned build/launch boundary; no hardware access during import or planning."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from enodia.tt.bench import issue120_probe, issue120_results

from . import kernel_adapter, plan

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
BUILD = ROOT / "out/issue120-production-build"
DEPS = ROOT / "out/issue120-deps"
PROOF = ROOT / "docs/measurements/issue120-production-device-compile-check.json"
PROTOCOL_HEADER = "protocol.hpp"
COMPILE_MANIFEST = "compile_args.json"
LIBRARY_SHA256 = "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141"
PINNED_IMAGE = "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_path(path: Path, root: Path, *, exists: bool = True) -> Path:
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_relative_to(root):
        raise ValueError("program-owned path is redirected")
    if exists and not path.exists():
        raise ValueError("required program-owned path missing")
    return path


def source_hashes() -> dict[str, str]:
    paths = [SOURCE / name for name in ("native.cpp", PROTOCOL_HEADER, "host_protocol.hpp",
                                       "host_sequence.hpp", "CMakeLists.txt")]
    paths += [SOURCE.parent / "issue120" / name for name in ("CMakeLists.txt", "native_result_io.hpp",
                                                            "q4_host_sequence.hpp", "q4_runtime_args.hpp")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def build() -> None:
    fixed_path(BUILD, ROOT / "out", exists=False)
    kernel_adapter.generate(BUILD / "kernels")
    command = ["cmake", "-S", str(SOURCE), "-B", str(BUILD),
               f"-DTT_METAL_SOURCE_ROOT={DEPS / 'source'}",
               "-DTT_METAL_SOURCE_REVISION=d9a68815f5fcf08a5bfbffb6f1f811823fba8edd",
               f"-DTT_METAL_LIBRARY={DEPS / 'lib/libtt_metal.so'}",
               f"-DTT_METAL_EXTRA_INCLUDE_DIRS={DEPS / 'include'}",
               f"-DTT_METAL_EXTRA_LIBRARY_DIRS={DEPS / 'lib'};{DEPS / 'wheel-libs'}"]
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(BUILD), "--target", "issue120_production",
                    "--parallel", "2"], check=True)
    receipt = {"schema": "issue120-production-build-v1", "source_sha256": source_hashes(),
               "native_sha256": sha(BUILD / "issue120_production"),
               "library_sha256": sha(BUILD / "libtt_metal.so")}
    (BUILD / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def _native_receipt() -> dict:
    fixed_path(BUILD, ROOT / "out")
    executable = fixed_path(BUILD / "issue120_production", BUILD)
    receipt = json.loads(fixed_path(BUILD / "build-receipt.json", BUILD).read_bytes())
    if receipt.get("schema") != "issue120-production-build-v1" or receipt.get("source_sha256") != source_hashes():
        raise ValueError("native receipt is stale for these sources")
    if receipt.get("native_sha256") != sha(executable) or receipt.get("library_sha256") != LIBRARY_SHA256:
        raise ValueError("native artifact pin differs")
    if sha(fixed_path(BUILD / "libtt_metal.so", BUILD)) != LIBRARY_SHA256:
        raise ValueError("native runtime library differs")
    return receipt


def _device_proof() -> dict:
    proof = json.loads(PROOF.read_bytes())
    manifest = json.loads((SOURCE / COMPILE_MANIFEST).read_bytes())
    if not proof.get("all_passed") or proof.get("manifest_sha256") != sha(SOURCE / COMPILE_MANIFEST):
        raise ValueError("offline device compile proof is missing or stale")
    cases = {case["case"]: case for case in proof["cases"]}
    expected = {f"{role}-watcher-{watcher}" for role in ("producer", "consumer") for watcher in (0, 1)}
    if set(cases) != expected or any(case["returncode"] != 0 for case in cases.values()):
        raise ValueError("device compiler cases incomplete")
    for name, case in cases.items():
        if case["compile_time_args"] != manifest[name.split("-")[0]]:
            raise ValueError("device compiler arguments differ")
    return proof


def _derived_hashes() -> dict:
    generated = BUILD / "kernels"
    hashes = {}
    for role in ("producer", "consumer"):
        p = fixed_path(generated / f"production_{role}.cpp", generated)
        if p.read_text() != kernel_adapter.derive(role):
            raise ValueError("derived kernel differs from pinned adapter")
        hashes[p.name] = sha(p)
    for name, original in (("device.hpp", SOURCE / "device.hpp"),
                           (PROTOCOL_HEADER, SOURCE / PROTOCOL_HEADER),
                           ("resident_tables.h", kernel_adapter.BASE / "resident_tables.h")):
        p = fixed_path(generated / name, generated)
        if p.read_bytes() != original.read_bytes():
            raise ValueError("derived support header is stale")
        hashes[name] = sha(p)
    return hashes


def program_identity() -> dict:
    receipt, proof, hashes = _native_receipt(), _device_proof(), _derived_hashes()
    if proof.get("derived_sources") != hashes:
        raise ValueError("actual derived kernels differ from offline proof")
    return {"native_sha256": receipt["native_sha256"], "library_sha256": LIBRARY_SHA256,
            "kernel_sources_sha256": hashes, "device_compile_proof_sha256": sha(PROOF),
            "compile_manifest_sha256": sha(SOURCE / COMPILE_MANIFEST),
            "python_sources_sha256": {p.name: sha(p) for p in sorted(SOURCE.glob("*.py"))}}


def execute_native(spec: plan.RunSpec, attempt: issue120_results.NativeAttempt):
    if (os.environ.get("TT_METAL_WATCHER") == "1") != spec.watcher:
        raise ValueError("Watcher mode differs from planned first-kernel policy")
    identity = program_identity()
    command = [str(BUILD / "issue120_production"), "--condition", spec.condition,
               "--frames", str(spec.frame_count), "--repeats", str(spec.repeat_count),
               "--cap", str(spec.cap_seconds), "--budget", str(spec.run_budget_ticks),
               "--attempt-identity", attempt.identity, "--result-file", str(attempt.result_file)]
    completed = issue120_results.execute(command, attempt, spec.cap_seconds,
                                        issue120_probe._runtime_environment, ROOT)
    if completed.returncode != 0:
        raise ValueError("native failed; direct status and raw streams retained")
    return issue120_results.read_result(attempt), identity
