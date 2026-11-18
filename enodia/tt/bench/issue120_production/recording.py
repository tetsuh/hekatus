"""Attempt-bound binary decoding, frozen power and production record assembly."""
from __future__ import annotations

import contextlib
import csv
import datetime as dt
import json
import math
import os
import shutil
import time
from pathlib import Path

from enodia.tt.bench import issue120_probe, issue120_results
from enodia.tt.bench.power_trace_snapshot import (
    ISSUE120_V2_LOAD_INTERVAL_CONTRACT,
    ISSUE120_V2_LOAD_INTERVAL_SOURCE,
    LEGACY_LOAD_INTERVAL_CONTRACT,
    PowerTraceSnapshotOptions,
    atomic_power_trace_snapshot,
    public_power_trace_snapshot,
    read_power_trace_snapshot,
)
from enodia.tt.bench.resident_clock import AICLK_SOURCE_LOAD_INTERVAL_TRACE
from enodia.tt.bench.run_binding import validate_run_id

from . import artifacts, checks, coverage, diagnostics, plan, protocol, records, runtime


def copy_exclusive(source: Path, target: Path) -> None:
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fchmod(dst.fileno(), 0o644)
        os.fsync(dst.fileno())


def write_record(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fchmod(output.fileno(), 0o644)
        os.fsync(output.fileno())
    os.link(temporary, path)
    temporary.unlink()


def utc(ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ns / 1e9, dt.UTC)


def _last_sample(path: Path) -> dt.datetime | None:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return dt.datetime.fromisoformat(rows[-1]["timestamp_utc"]) if rows else None
    except (OSError, ValueError, KeyError):
        return None


def freeze_power(output: Path, identity: str, run_start: int, run_end: int, native: dict) -> tuple[dict, float]:
    source = output / f"power-{identity}.csv"
    began = time.monotonic()
    deadline = began + 5
    while time.monotonic() < deadline:
        last = _last_sample(source)
        if last is not None and last >= utc(run_end):
            break
        time.sleep(.05)
    snapshot = atomic_power_trace_snapshot(source, output, run_id=identity, snapshot_number=1,
                                           publish_mode=0o644)
    program = native["program_host_interval"]
    metadata = read_power_trace_snapshot(snapshot, options=PowerTraceSnapshotOptions(
        run_id=identity, run_start=utc(run_start), run_end=utc(run_end), source_path=source,
        sampler_interval_seconds=2, coverage_definition="explicit_final_sample",
        load_interval_start=utc(program["start_utc_ns"]), load_interval_end=utc(program["end_utc_ns"]),
        load_interval_source=ISSUE120_V2_LOAD_INTERVAL_SOURCE,
        load_interval_contract=ISSUE120_V2_LOAD_INTERVAL_CONTRACT,
        include_issue118_metadata=True, require_explicit_load_interval=True, require_all_rows_valid=True))
    return public_power_trace_snapshot(metadata), time.monotonic() - began


def _read_captures(attempt: issue120_results.NativeAttempt, output: Path) -> tuple[dict, dict]:
    execution = json.loads(attempt.execution_path.read_bytes())
    paths, refs = {}, {}
    for name in ("stdout", "stderr"):
        path = attempt.directory / (name + ".bin")
        if path.stat().st_size != execution[name]["byte_count"] or runtime.sha(path) != execution[name]["sha256"]:
            raise ValueError("raw stream fingerprint differs")
        target = output / f"results-{attempt.identity}-native-{name}.bin"
        copy_exclusive(path, target)
        paths[name] = target
        refs[name] = {"file": target.name, "sha256": runtime.sha(target), "byte_count": path.stat().st_size}
    target = output / f"results-{attempt.identity}-native-execution.json"
    copy_exclusive(attempt.execution_path, target)
    refs["execution"] = {"file": target.name, "sha256": runtime.sha(target), "byte_count": target.stat().st_size}
    target = output / f"results-{attempt.identity}-native-result.json"
    copy_exclusive(attempt.result_file, target)
    refs["result"] = {"file": target.name, "sha256": runtime.sha(target), "byte_count": target.stat().st_size}
    return refs, _check_stream_files(paths)


def _check_stream_files(paths: dict) -> dict:
    with contextlib.ExitStack() as stack:
        streams = {}
        for name, path in paths.items():
            handle = stack.enter_context(path.open(encoding="utf-8", errors="replace"))
            streams[name] = iter(lambda h=handle: h.readline(16385), "")
        return checks.warning_check(streams)


def _copy_frames(native: dict, attempt, output: Path):
    loaded = native["condition"] == "loaded"
    emitted = native["host_cpu_work"]["transfers"] if loaded else 0
    values = {}
    for kind in ("frames", "transitions"):
        item = artifacts.read_evidence(attempt.directory / "evidence", kind + ".bin",
                                       frames=native["frame_count"], emitted=emitted)
        target = output / f"results-{attempt.identity}-{kind}.bin"
        copy_exclusive(item.path, target)
        values[kind] = artifacts.read_public_evidence(output, attempt.identity, kind,
                                                      frames=native["frame_count"], emitted=emitted)
    if values["frames"].count != native["consumer_stats"][1]:
        raise ValueError("binary frame count differs from consumer accounting")
    if not loaded and values["transitions"].count:
        raise ValueError("control has host transitions")
    index = protocol.TransitionIndex(values["transitions"].rows(), frame_count=native["frame_count"],
                                     repeat_count=emitted) if loaded else None
    return values, index


def _diagnostic_check(native: dict) -> None:
    expected = diagnostics.expectations(native["consumer_stats"][1], native["frame_count"])
    for key, value in expected.items():
        if native["startup_diagnostics"].get(key) != value:
            raise ValueError("phase-specific diagnostic expectation differs")
    if native["startup_diagnostics"].get("consumer_reported_expectation") != expected["expected_after_consumed_frames"]:
        raise ValueError("consumer expectation echo differs")


def make_record(spec: plan.RunSpec, attempt, native: dict, program_identity: dict,
                session: dict, session_sha: str, *, execute_start_ns: int, execute_end_ns: int,
                start_utc_ns: int, end_utc_ns: int, watcher_before: bytes | None) -> dict:
    output = attempt.directory.parent
    identity = validate_run_id(attempt.identity)
    native_config = plan.program_configuration(spec)
    environment = issue120_probe._environment_record(str(output / f"env-{identity}.json"))
    if environment["commit"] != session["harness_commit"]:
        raise ValueError("session harness changed")
    record = {"schema": records.SCHEMA, "phase": spec.phase, "condition": spec.condition,
              "plan_identity": spec.identity, "attempt_identity": identity,
              "session_id": session["session_id"], "session_manifest_sha256": session_sha,
              "configuration": native_config, "configuration_sha256": plan.configuration_digest(native_config),
              "program_identity": program_identity, "environment": environment, "native": native,
              "host_boot_id_sha256": session["host_boot_id_sha256"],
              "execution_interval": {"start_ns": execute_start_ns, "end_ns": execute_end_ns,
                                     "start_utc_ns": start_utc_ns, "end_utc_ns": end_utc_ns}}
    records.validate_record(record)
    _diagnostic_check(native)
    values, index = _copy_frames(native, attempt, output)
    began = time.monotonic()
    analyzed = coverage.analyze(values["frames"].rows(), native, index)
    analysis_seconds = time.monotonic() - began
    record.update(analyzed)
    record["artifacts"] = {name: item.metadata() for name, item in values.items()}
    record["native_capture"], record["warning_check"] = _read_captures(attempt, output)
    watcher = {"requested": spec.watcher, "fresh": False, "scan_passed": not spec.watcher}
    if spec.watcher:
        path = runtime.ROOT / "generated/watcher/watcher.log"
        after = read_watcher_bytes(path) if path.is_file() else b""
        facts, fresh = checks.watcher_check(watcher_before, after,
            modified_after_start=path.is_file() and path.stat().st_mtime_ns >= start_utc_ns)
        watcher.update(facts)
        target = output / f"results-{identity}-watcher.log"
        with target.open("xb") as handle:
            handle.write(fresh)
            os.fchmod(handle.fileno(), 0o644)
        watcher["artifact"] = {"file": target.name, "sha256": runtime.sha(target), "byte_count": len(fresh)}
    record["watcher"] = watcher
    power, waited = freeze_power(output, identity, start_utc_ns, end_utc_ns, native)
    record["power_trace_snapshot"] = power
    record["clock"] = clock_metadata(power, record["metrics"])
    record["execution_eligible"], record["timing_evidence"] = eligibility(record, power)
    native_start = native["program_host_interval"]["start_ns"]
    retrieval = max(0, execute_end_ns - native["host_scheduling"]["drain_end_ns"]) / 1e9
    record["overhead"] = {"setup_seconds": max(0, native_start - execute_start_ns) / 1e9,
                          "native_retrieval_verify_close_publish_seconds": retrieval,
                          "python_frame_analysis_seconds": analysis_seconds,
                          "analysis_frame_count": spec.frame_count,
                          "per_frame_analysis_seconds": (retrieval + analysis_seconds) / spec.frame_count,
                          "per_frame_term_is_amortized_not_linear_slope": True,
                          "sampler_wait_seconds": waited}
    return record


def load_validated_record(path: Path) -> dict:
    record = issue120_results._read_json(path)
    records.validate_record(record)
    output = path.parent.resolve()
    identity = record["attempt_identity"]
    if path.name != f"results-{identity}.json":
        raise ValueError("record filename is not attempt-bound")
    native = record["native"]
    emitted = native["host_cpu_work"].get("transfers", 0)
    values = {}
    for kind in ("frames", "transitions"):
        item = artifacts.read_public_evidence(output, identity, kind, frames=native["frame_count"], emitted=emitted)
        if item.metadata() != record["artifacts"][kind]:
            raise ValueError("persisted binary fingerprint differs")
        values[kind] = item
    index = protocol.TransitionIndex(values["transitions"].rows(), frame_count=native["frame_count"],
                                     repeat_count=emitted) if record["condition"] == "loaded" else None
    analysis = coverage.analyze(values["frames"].rows(), native, index)
    for key, value in analysis.items():
        if record.get(key) != value:
            raise ValueError("persisted coverage/metrics differ from complete artifact recomputation")
    snapshot = record["power_trace_snapshot"]
    source = output / snapshot["file"]
    if source.name != snapshot["file"] or source.is_symlink() or source.resolve() != source.absolute() or runtime.sha(source) != snapshot["sha256"]:
        raise ValueError("frozen power fingerprint differs")
    _diagnostic_check(native)
    _recheck_captures(output, record)
    _recheck_watcher(output, record)
    _recheck_power(output, record)
    return record


def _recheck_captures(output: Path, record: dict) -> None:
    identity = record["attempt_identity"]
    refs = record["native_capture"]
    if set(refs) != {"stdout", "stderr", "execution", "result"}:
        raise ValueError("native capture set differs")
    paths = {}
    for name, meta in refs.items():
        suffix = ".bin" if name in {"stdout", "stderr"} else ".json"
        expected = f"results-{identity}-native-{name}{suffix}"
        path = output / expected
        if meta["file"] != expected or path.is_symlink() or not path.is_file():
            raise ValueError("native capture identity differs")
        if runtime.sha(path) != meta["sha256"] or path.stat().st_size != meta["byte_count"]:
            raise ValueError("native capture fingerprint differs")
        paths[name] = path
    envelope = issue120_results._read_json(paths["result"])
    if envelope.get("schema") != issue120_results.RESULT_SCHEMA or envelope.get("attempt_identity") != identity or envelope.get("result") != record["native"]:
        raise ValueError("record differs from its atomic native envelope")
    execution = issue120_results._read_json(paths["execution"])
    expected = {"schema": issue120_results.EXECUTION_SCHEMA, "attempt_identity": identity,
                "native_started": True, "native_returncode": 0, "timed_out": False, "failure_type": None}
    if any(execution.get(k) != v for k, v in expected.items()) or type(execution.get("native_returncode")) is not int:
        raise ValueError("direct native execution does not permit success")
    if _check_stream_files({k: paths[k] for k in ("stdout", "stderr")}) != record["warning_check"]:
        raise ValueError("runtime warning review differs")


def _recheck_power(output: Path, record: dict) -> None:
    old = record["power_trace_snapshot"]
    execution = record["execution_interval"]
    program = record["native"]["program_host_interval"]
    source, contract = _power_contract(record["schema"])
    metadata = read_power_trace_snapshot(output / old["file"], options=PowerTraceSnapshotOptions(
        run_id=record["attempt_identity"], run_start=utc(execution["start_utc_ns"]),
        run_end=utc(execution["end_utc_ns"]), sampler_interval_seconds=2,
        coverage_definition="explicit_final_sample", include_issue118_metadata=True,
        load_interval_start=utc(program["start_utc_ns"]), load_interval_end=utc(program["end_utc_ns"]),
        load_interval_source=source, load_interval_contract=contract,
        require_explicit_load_interval=True, require_all_rows_valid=True))
    keys = ["sha256", "byte_count", "coverage_complete", "aiclk_mhz_load_interval_weighted"]
    if record["schema"] == records.SCHEMA:
        keys += ["errors", "load_interval_source", "aiclk_mhz_load_interval_weighted_seconds",
                 "aiclk_mhz_load_interval_weighted_source"]
    for key in keys:
        if metadata.get(key) != old.get(key):
            raise ValueError("frozen power interpretation differs")
    execution, timing = eligibility(record, metadata)
    if (record["execution_eligible"], record["timing_evidence"]) != (execution, timing):
        raise ValueError("execution eligibility differs from recomputation")
    if record["clock"] != clock_metadata(metadata, record["metrics"], schema=record["schema"]):
        raise ValueError("timing/clock interpretation differs from recomputation")


def read_watcher_bytes(path: Path) -> bytes:
    if path.is_symlink() or path.stat().st_size > checks.MAX_WATCHER_BYTES:
        raise ValueError("Watcher capture is redirected or exceeds fixed diagnostic bound")
    return path.read_bytes()


def _recheck_watcher(output: Path, record: dict) -> None:
    watcher = record["watcher"]
    if watcher["requested"] != record["configuration"]["watcher"]:
        raise ValueError("Watcher policy differs from configuration")
    if not watcher["requested"]:
        if "artifact" in watcher or watcher.get("fresh") or watcher.get("scan_passed") is not True:
            raise ValueError("disabled Watcher cannot attribute a stale artifact")
        return
    meta = watcher["artifact"]
    name = f"results-{record['attempt_identity']}-watcher.log"
    path = output / name
    if meta["file"] != name or not path.is_file():
        raise ValueError("Watcher evidence is missing or misbound")
    raw = read_watcher_bytes(path)
    if len(raw) != meta["byte_count"] or runtime.sha(path) != meta["sha256"]:
        raise ValueError("Watcher evidence fingerprint differs")
    facts, _ = checks.watcher_check(None, raw, modified_after_start=True)
    if any(watcher.get(k) != facts[k] for k in ("fresh", "scan_passed", "dump_count")):
        raise ValueError("Watcher evidence scan differs")


def _power_contract(schema: str) -> tuple[str, str]:
    if schema == records.LEGACY_SCHEMA:
        return "issue120_enqueue_to_l1_terminal_observed_upper_bound", LEGACY_LOAD_INTERVAL_CONTRACT
    return ISSUE120_V2_LOAD_INTERVAL_SOURCE, ISSUE120_V2_LOAD_INTERVAL_CONTRACT


def _positive_finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _valid_v2_weighting(power: dict) -> bool:
    return (power.get("errors") == [] and power.get("load_interval_source") == ISSUE120_V2_LOAD_INTERVAL_SOURCE
            and power.get("aiclk_mhz_load_interval_weighted_source") == AICLK_SOURCE_LOAD_INTERVAL_TRACE
            and _positive_finite(power.get("aiclk_mhz_load_interval_weighted"))
            and _positive_finite(power.get("aiclk_mhz_load_interval_weighted_seconds")))


def eligibility(record: dict, power: dict) -> tuple[bool, bool]:
    causal = record["condition"] == "control" or record["coverage_analysis"]["coverage"] == "covered"
    watch = record["watcher"]
    watch_ok = watch["scan_passed"] and (not watch["requested"] or watch["fresh"])
    execution = bool(record["resident_integrity_passed"] and causal and record["warning_check"]["passed"] and watch_ok)
    timing = bool(execution and power.get("coverage_complete") and not watch["requested"])
    if record.get("schema") != records.LEGACY_SCHEMA:
        timing = timing and _valid_v2_weighting(power)
    return execution, timing


def clock_metadata(power: dict, metrics: dict, *, schema: str = records.SCHEMA) -> dict:
    weighted = power.get("aiclk_mhz_load_interval_weighted")
    if schema != records.LEGACY_SCHEMA and not _valid_v2_weighting(power):
        weighted = None
    estimates = None
    intervals = metrics.get("frame_intervals")
    if type(weighted) in (int, float) and weighted > 0 and intervals is not None:
        estimates = {name: row["value_ticks"] / weighted if row["status"] == "ok" else None
                     for name, row in intervals["percentiles"].items()}
        estimates["max"] = intervals["max_ticks"] / weighted if intervals["max_ticks"] is not None else None
    return {"basis": "consumer_low32_wrap_tracked_cycle_domain", "aiclk_mhz": power.get("aiclk_mhz_in_run_max"),
            "observed_host_envelope_weighted_aiclk_mhz": weighted,
            "observed_envelope_conversion_us": estimates,
            "conversion_scope": "host_observed_program_upper_bound_not_exact_frame_window",
            "product_wall_time_guarantee": False}
