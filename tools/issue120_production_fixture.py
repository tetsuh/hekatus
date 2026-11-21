"""CPU-only production companions and real host record pipeline; never native code.

All synthetic captures/clock/telemetry are explicitly fixture-only. No device,
board discovery, SDK, tt-smi, wrapper or hardware runner is invoked.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from enodia.tt.bench.issue120_production import (
    artifacts,
    host_trace,
    plan,
    protocol,
    recording,
    runner,
    runtime,
    sessions,
)
from enodia.tt.bench.issue120_production import production_admission as production
from enodia.tt.bench.issue120_results import NativeAttempt

ROOT = runtime.ROOT / "out/issue120-production-p-prep"


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def put64(words, offset, value):
    words[offset:offset + 2] = [value & 0xffffffff, value >> 32]


def completion_jitter(number, loaded):
    return ((number * 2654435761) ^ (number >> 7)) % 2001 if loaded and number else 0


@contextmanager
def measured_calls():
    """Time real calls without replacing their computation or result."""
    costs = {"runtime_hash_seconds": 0.0, "binary_size_hash_seconds": 0.0, "copy_fsync_seconds": 0.0}

    def timed(function, label):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                costs[label] += time.perf_counter() - start
        return call

    with ExitStack() as stack:
        for module, name, label in ((runtime, "sha", "runtime_hash_seconds"),
                                    (artifacts, "_read", "binary_size_hash_seconds"),
                                    (recording, "copy_exclusive", "copy_fsync_seconds")):
            stack.enter_context(patch.object(module, name, timed(getattr(module, name), label)))
        yield costs


def write_companions(directory, count, repeats, loaded):
    accumulator = 0
    with (directory / "frames.bin").open("xb", buffering=1024 * 1024) as output:
        output.write(artifacts.HEADER.pack(b"IP120F1\0", 1, 32, count, count))
        for number in range(count):
            tick = number * 1350000 + completion_jitter(number, loaded)
            row = [0] * 32
            accumulator = protocol.resident_checksum(accumulator, number, 64)
            put64(row, 0, tick + 16)
            row[2:5] = [accumulator, 2, number]
            row[18] = 64
            put64(row, 25, tick + 14)
            if loaded:
                put64(row, 5, tick + 11)
                row[7:11] = [2 * number + 2, 3 * number + 3, 2 * number + 3, 3 * number + 5]
                put64(row, 11, tick + 9)
                put64(row, 13, tick + 13)
                row[15], row[17] = 1, number + 2
                for offset, delta in ((19, 7), (21, 8), (23, 12)):
                    put64(row, offset, tick + delta)
            output.write(artifacts.FRAME.pack(*row))
    with (directory / "transitions.bin").open("xb", buffering=1024 * 1024) as output:
        output.write(artifacts.HEADER.pack(b"IP120T1\0", 1, 2, count + 2 if loaded else 0, repeats if loaded else 0))
        if loaded:
            for number in range(count + 2):
                output.write(artifacts.TRANSITION.pack(2 * number + 1, number + 1))


def native_fixture(spec, base, utc):
    count, repeats = spec.frame_count, spec.repeat_count
    loaded = spec.condition == "loaded"
    ps, cs = [0] * 1024, [0] * 1024
    ps[1] = ps[3] = cs[1] = count
    cs[32:36] = [count + 2, count + 2, count + 2, 0]
    if loaded:
        cs[36:40] = [1, 2, 2, 3]
        for offset, value in ((40, 5), (42, 8), (44, 7), (57, 10)):
            put64(cs, offset, value)
        cs[54:57] = [1, 1, 1]
        cs[46:50] = [2 * count + 2, 3 * count + 3, 2 * count + 3, 3 * count + 5]
        last = (count - 1) * 1350000 + 16 + completion_jitter(count - 1, loaded)
        put64(cs, 50, last + 3)
        put64(cs, 52, last + 5)
    ns = lambda second: base + second * 10**9
    scheduling = dict(zip(("enqueue_begin_ns", "enqueue_return_ns", "load_join_completed_ns", "completion_observer_joined_ns", "finish_begin_ns", "finish_end_ns", "drain_end_ns"), map(ns, (1, 3, 507, 508, 509, 510, 511)), strict=True))
    cpu = {"start_ns": ns(2) if loaded else 0, "end_ns": ns(506) if loaded else 0,
           "operations": repeats * 32768 if loaded else 0, "reads": repeats if loaded else 0, "transfers": repeats if loaded else 0}
    counts = [1, *range(65536, repeats + 1, 65536)] if loaded else []
    samples = [{"committed_count": value, "monotonic_ns": ns(3) + (ns(506) - ns(3)) * (i + 1) // (len(counts) + 1)} for i, value in enumerate(counts)]
    trace = {"schema": host_trace.TRACE_SCHEMA, "clock": "CLOCK_MONOTONIC", "applicable": loaded,
             "period_writes": 65536, "capacity": 65536, "sample_storage_bytes": 1048576,
             "worker_start_ns": cpu["start_ns"], "worker_end_ns": cpu["end_ns"], "join_ns": ns(507),
             "final_count": repeats if loaded else 0, "first_committed_enqueue_observed_ns": samples[0]["monotonic_ns"] if loaded else 0,
             "main_enqueue_return_observation": {"before_ns": ns(3), "after_ns": ns(3) + 100, "committed_count": 0}, "samples": samples}
    return {"schema": host_trace.NATIVE_SCHEMA, "fixture_only": True, "condition": spec.condition,
            "frame_count": count, "repeat_count": repeats, "producer_stats": ps, "consumer_stats": cs,
            "checksum_verified_frames": count, "host_interval": {"start_ns": ns(2), "end_ns": ns(506)} if loaded else None,
            "host_cpu_work": cpu, "host_scheduling": scheduling, "host_phase_trace": trace,
            "resident_host_interval": {"start_ns": ns(1), "end_ns": ns(510), "scope": "enqueue_to_post_join_finish_return_upper_bound"},
            "program_host_interval": {"start_ns": ns(1), "end_ns": ns(501), "start_utc_ns": utc + 10**9, "end_utc_ns": utc + 501 * 10**9,
                                      "scope": "enqueue_to_l1_terminal_observed_upper_bound"},
            "startup_diagnostics": {"expected_after_consumed_frames": count + 2, "expected_after_full_run": count + 2, "consumer_reported_expectation": count + 2}}


def publish(directory: Path, role: str):
    settings = json.loads((directory / "fixture-settings.json").read_bytes())
    sessions.SESSIONS = directory / "sessions"
    session, digest = sessions.read_session(settings["session_id"])
    spec = sessions.spec_for_session(plan.RUNS[4 if role == "control" else 5].identity, session)
    case = sessions.session_path(session["session_id"]) / "cases" / spec.identity
    case.mkdir(parents=True, exist_ok=True)
    # Only the external /out mount root changes; the real capture parser stays.
    runtime.issue120_probe.OUTPUT_ROOT = case
    identity = "synthetic-p1" if role == "control" else "synthetic-p2"
    capture = case / f"native-{identity}"
    (capture / "result").mkdir(parents=True)
    (capture / "evidence").mkdir()
    offset = 0 if role == "control" else 600 * 10**9
    base, utc = settings["monotonic_origin_ns"] + offset, settings["UTC_origin_ns"] + offset
    started = time.perf_counter()
    write_companions(capture / "evidence", spec.frame_count, spec.repeat_count, role == "loaded")
    synthesis_seconds = time.perf_counter() - started
    # _run validates P1 again inside the wrapper before native execution.
    # Keep that host cost charged; do not hide it in the outer launcher reread.
    began = time.perf_counter()
    if role == "loaded":
        runner._control(session["session_id"], spec)
    in_container_control_seconds = time.perf_counter() - began
    native = native_fixture(spec, base, utc)
    save(capture / "result/result.json", {"schema": "issue120-native-result-v1", "attempt_identity": identity, "fixture_only": True, "result": native})
    execution = {"schema": "issue120-native-execution-v1", "fixture_only": True, "attempt_identity": identity,
                 "native_started": True, "native_returncode": 0, "timed_out": False, "failure_type": None}
    for stream in ("stdout", "stderr"):
        (capture / f"{stream}.bin").write_bytes(b"")
        execution[stream] = {"sha256": hashlib.sha256(b"").hexdigest(), "byte_count": 0}
    save(capture / "execution.json", execution)
    raw = settings["environment_template"].copy()
    raw.update(fixture_only=True, harness_commit=session["harness_commit"], harness_dirty=False, harness_pushed=True, run_id=identity)
    save(case / f"env-{identity}.json", raw)
    with (case / f"power-{identity}.csv").open("x") as output:
        output.write("timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n")
        for second in range(0, 515, 2):
            output.write(f"{dt.datetime.fromtimestamp(utc / 1e9 + second, dt.UTC).isoformat()},70,1350,55\n")
    started = time.perf_counter()
    gc_before = gc.get_stats()
    with measured_calls() as costs:
        row = recording.make_record(spec, NativeAttempt(identity, capture), native, session["program_identity"], session, digest,
                                    execute_start_ns=base, execute_end_ns=base + 512 * 10**9,
                                    start_utc_ns=utc, end_utc_ns=utc + 512 * 10**9, watcher_before=None)
    construct_seconds = time.perf_counter() - started
    path = case / f"results-{identity}.json"
    started = time.perf_counter()
    recording.write_record(path, row)
    publication_seconds = time.perf_counter() - started
    assert row["resident_integrity_passed"] and row["execution_eligible"] and row["timing_evidence"]
    result = {"fixture_only": True, "role": role, "frames": spec.frame_count, "synthesis_seconds": synthesis_seconds,
              "record_construction_seconds": construct_seconds, "publication_seconds": publication_seconds,
              "charged_host_seconds": construct_seconds + publication_seconds + in_container_control_seconds,
              "in_container_control_validation_seconds": in_container_control_seconds,
              "analysis_seconds": row["overhead"]["python_frame_analysis_seconds"], "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "frames_bytes": row["artifacts"]["frames"]["byte_count"], "transitions_bytes": row["artifacts"]["transitions"]["byte_count"],
              "trace_samples": len(native["host_phase_trace"]["samples"]), "device_retrieval_measured": False,
              "io_costs": costs, "GC_collections": [b["collections"] - a["collections"] for a, b in zip(gc_before, gc.get_stats(), strict=True)],
              "primary_JSON_bytes": path.stat().st_size, "synthetic_completion_jitter_ticks": 2000 if role == "loaded" else 0}
    save(directory / f"{role}-publisher.json", result)
    print(json.dumps(result))


def readback(directory: Path, role: str):
    settings = json.loads((directory / "fixture-settings.json").read_bytes())
    identity = "synthetic-p1" if role == "control" else "synthetic-p2"
    spec_id = plan.RUNS[4 if role == "control" else 5].identity
    path = directory / "sessions" / settings["session_id"] / "cases" / spec_id / f"results-{identity}.json"
    started = time.perf_counter()
    gc_before = gc.get_stats()
    with measured_calls() as costs:
        row = recording.load_validated_record(path)
    seconds = time.perf_counter() - started
    assert row["fixture_only"] and row["metrics"]["frames_consumed"] == 500000 and row["execution_eligible"]
    result = {"fixture_only": True, "role": role, "launcher_reread_seconds": seconds,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "io_costs": costs,
              "GC_collections": [b["collections"] - a["collections"] for a, b in zip(gc_before, gc.get_stats(), strict=True)]}
    save(directory / f"{role}-reader.json", result)
    print(json.dumps(result))


def bootstrap():
    directory = ROOT / uuid.uuid4().hex
    directory.mkdir(parents=True)
    target = production.POSTPROCESS
    target.parent.mkdir(parents=True, exist_ok=True)
    provisional = {"schema": "issue120-production-host-postprocess-v1", "fixture_only": True, "measurement_pending": True,
                   "frames": 500000, "passed": True, "peak_rss_bytes": 0, "charged_host_seconds": 0.0, "launcher_reread_seconds": 0.0, "device_retrieval_measured": False}
    save(target, provisional)
    receipt = production.make_receipt(production.fixture_identity())
    save(runtime.SOURCE / "production_admission.json", receipt)
    sid = uuid.uuid4().hex
    now = time.monotonic_ns()
    specs = [plan.RUNS[4], plan.RUNS[5]]
    from dataclasses import replace
    specs = [replace(s, repeat_count=production.selection()["fixed_repetitions"]) for s in specs]
    manifest = {"schema": "issue120-production-session-v1", "session_id": sid, "created_monotonic_ns": now,
                "harness_commit": sessions.current_head(), "host_boot_id_sha256": sessions.boot_fingerprint(),
                "program_identity": production.fixture_identity(), "image": runtime.PINNED_IMAGE,
                "production_contract": runtime.PRODUCTION_CONTRACT, "fixture_only": True,
                "admission_hashes": {"production": runtime.sha(runtime.SOURCE / "production_admission.json")},
                "runs": {s.identity: {"configuration": plan.program_configuration(s), "configuration_sha256": plan.configuration_digest(plan.program_configuration(s))} for s in specs}}
    session_dir = directory / "sessions" / sid
    session_dir.mkdir(parents=True)
    save(session_dir / "manifest.json", manifest)
    import tarfile
    with tarfile.open(runtime.ROOT / sessions.C1_DIRECTORY / "C1-evidence.tar.gz") as bundle:
        raw = json.loads(bundle.extractfile(f"env-{sessions.C1_ATTEMPT}.json").read())
    save(directory / "fixture-settings.json", {"session_id": sid, "monotonic_origin_ns": now + 10**8, "UTC_origin_ns": time.time_ns(), "environment_template": raw})
    for role in ("control", "loaded"):
        for mode in ("publish", "readback"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--role", role, "--directory", str(directory)], check=True, cwd=runtime.ROOT)
    publishers = [json.loads((directory / f"{r}-publisher.json").read_bytes()) for r in ("control", "loaded")]
    readers = [json.loads((directory / f"{r}-reader.json").read_bytes()) for r in ("control", "loaded")]
    report = {**provisional, "measurement_pending": False, "passed": True,
              "charged_host_seconds": max(r["charged_host_seconds"] for r in publishers),
              "launcher_reread_seconds": max(r["launcher_reread_seconds"] for r in readers),
              "peak_rss_bytes": max(r["peak_rss_bytes"] for r in [*publishers, *readers]),
              "publishers": publishers, "readers": readers, "fixture_source_sha256": runtime.sha(Path(__file__)),
              "memory_scope": "fresh child lifetime ru_maxrss; separate publisher and reader, not summed",
              "excluded": ["synthetic companion generation", "real native execution and 2048000000-byte device retrieval", "real hardware telemetry"],
              "artifact_root": directory.relative_to(runtime.ROOT).as_posix()}
    save(target, report)
    save(runtime.SOURCE / "production_admission.json", production.make_receipt(production.fixture_identity()))
    print(json.dumps({"report": target.relative_to(runtime.ROOT).as_posix(), "charged_host_seconds": report["charged_host_seconds"], "peak_rss_bytes": report["peak_rss_bytes"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bootstrap", "publish", "readback"), required=True)
    parser.add_argument("--role", choices=("control", "loaded"))
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.mode == "bootstrap":
        bootstrap()
    else:
        directory = runtime.issue120_probe._bounded_path(args.directory, ROOT, "fixture directory", "directory")
        (publish if args.mode == "publish" else readback)(directory, args.role)


if __name__ == "__main__":
    main()
