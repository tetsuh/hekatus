"""Narrow new-schema invariants; not a complete validator for arbitrary records."""
from __future__ import annotations

import math

from .plan import configuration_digest

SCHEMA = "issue120-production-v1"


class SchemaOutOfScope(ValueError):
    """An existing record belongs to its unchanged legacy validator."""


def integer(value, field: str, low: int = 0, high: int = 2**64 - 1) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {field}")
    return value


def mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"missing {field}")
    return value


def validate_host_intervals(record: dict) -> None:
    if not isinstance(record, dict):
        raise TypeError("record must be a mapping")
    if record.get("schema") != SCHEMA:
        raise SchemaOutOfScope("legacy/unknown schema uses a different validator")
    native = mapping(record.get("native"), "native result")
    scheduling = mapping(native.get("host_scheduling"), "host scheduling")
    names = ("enqueue_begin_ns", "enqueue_return_ns", "load_join_completed_ns",
             "completion_observer_joined_ns", "finish_begin_ns", "finish_end_ns", "drain_end_ns")
    values = [integer(scheduling.get(name), name) for name in names]
    if values != sorted(values):
        raise ValueError("enqueue/join/finish ordering differs")
    interval = mapping(native.get("resident_host_interval"), "resident envelope")
    if interval != {"start_ns": values[0], "end_ns": values[5],
                    "scope": "enqueue_to_post_join_finish_return_upper_bound"}:
        raise ValueError("resident envelope must retain its upper-bound scope")
    cpu = mapping(native.get("host_cpu_work"), "host CPU work")
    if record.get("condition") == "control":
        if native.get("host_interval") is not None or any(cpu.get(k) != 0 for k in (
            "operations", "reads", "transfers")):
            raise ValueError("control must not fabricate host work")
        return
    if record.get("condition") != "loaded":
        raise ValueError("unsupported condition")
    host = mapping(native.get("host_interval"), "host load interval")
    start = integer(host.get("start_ns"), "host start")
    end = integer(host.get("end_ns"), "host end")
    if start > end or end > values[2]:
        raise ValueError("host work ends after its join or before its start")
    if (integer(cpu.get("start_ns"), "CPU start"), integer(cpu.get("end_ns"), "CPU end")) != (start, end):
        raise ValueError("CPU and host interval endpoints differ")


def validate_record(record: dict) -> None:
    validate_host_intervals(record)
    config = mapping(record.get("configuration"), "configuration")
    if record.get("configuration_sha256") != configuration_digest(config):
        raise ValueError("configuration digest differs")
    native = record["native"]
    for key in ("frame_count", "repeat_count"):
        if integer(native.get(key), key, 1) != config.get(key):
            raise ValueError("native work differs from the plan")
    if record["condition"] == "loaded":
        cpu = native["host_cpu_work"]
        count = config["repeat_count"]
        if (cpu.get("operations"), cpu.get("reads"), cpu.get("transfers")) != (count * 32768, count, count):
            raise ValueError("fixed host work is incomplete")
    program = mapping(native.get("program_host_interval"), "program observation interval")
    start = integer(program.get("start_ns"), "program start")
    end = integer(program.get("end_ns"), "program end")
    if start != native["host_scheduling"]["enqueue_begin_ns"] or not start <= end <= native["host_scheduling"]["completion_observer_joined_ns"]:
        raise ValueError("program completion observer binding differs")
    if program.get("scope") != "enqueue_to_l1_terminal_observed_upper_bound":
        raise ValueError("program interval is not an exact card completion")
    utc_start = integer(program.get("start_utc_ns"), "UTC start")
    utc_end = integer(program.get("end_utc_ns"), "UTC end")
    if utc_end < utc_start or abs((utc_end - utc_start) - (end - start)) > 2002000:
        raise ValueError("UTC/monotonic duration binding exceeds the existing capture bound")
    if not isinstance(record.get("session_id"), str) or not record["session_id"]:
        raise ValueError("session identity missing")


def _validate_comparison(control: dict, loaded: dict) -> None:
    for record in (control, loaded):
        validate_record(record)
        if not record.get("timing_evidence"):
            raise ValueError("control comparison requires eligible timing records")
    if control["condition"] != "control" or loaded["condition"] != "loaded":
        raise ValueError("control and loaded roles differ")
    for key in ("session_id", "session_manifest_sha256", "host_boot_id_sha256", "configuration_sha256", "program_identity", "environment"):
        if control.get(key) is None or control.get(key) != loaded.get(key):
            raise ValueError("unlike session/device/program configuration")
    if control.get("attempt_identity") == loaded.get("attempt_identity"):
        raise ValueError("control and loaded attempts must be distinct")
    if loaded["native"]["program_host_interval"]["start_ns"] < control["native"]["program_host_interval"]["end_ns"]:
        raise ValueError("control must precede loaded in the session")


def compare_records(control: dict, loaded: dict) -> dict:
    _validate_comparison(control, loaded)
    differences = {}
    for name in ("p50", "p99", "p99_9", "p99_99"):
        a = control["metrics"]["frame_intervals"]["percentiles"][name]
        b = loaded["metrics"]["frame_intervals"]["percentiles"][name]
        differences[name] = b["value_ticks"] - a["value_ticks"] if a["status"] == b["status"] == "ok" else None
    differences["max_ticks"] = loaded["metrics"]["frame_intervals"]["max_ticks"] - control["metrics"]["frame_intervals"]["max_ticks"]
    counters = {}
    for name in ("overflow_count", "paired_outlier_count", "frames_dropped", "frames_aborted"):
        a, b = control["metrics"].get(name), loaded["metrics"].get(name)
        counters[name] = b - a if type(a) is int and type(b) is int else None
    return {"schema": "issue120-production-comparison-v1", "loaded_minus_control_ticks": differences,
            "loaded_minus_control_counts": counters,
            "control_attempt": control["attempt_identity"], "loaded_attempt": loaded["attempt_identity"],
            "latency_threshold": None}


def _calibration_observation(record: dict, count: int) -> tuple[float, float, float]:
    validate_record(record)
    if (record.get("phase") != "calibration" or not record.get("timing_evidence") or
            record.get("coverage_analysis", {}).get("coverage") != "covered"):
        raise ValueError("calibration is not causally covered")
    native = record["native"]
    if native["frame_count"] != count:
        raise ValueError("wrong calibration identity")
    elapsed = (native["host_interval"]["end_ns"] - native["host_interval"]["start_ns"]) / 1e9
    program = native["program_host_interval"]
    if elapsed <= 0:
        raise ValueError("zero calibration work interval")
    projected = (program["end_ns"] - program["start_ns"]) / 1e9 * 500000 / count
    overhead = record.get("overhead", {})
    values = [overhead.get(k) for k in ("setup_seconds", "per_frame_analysis_seconds", "sampler_wait_seconds")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("calibration overhead accounting missing")
    return native["repeat_count"] / elapsed, projected, values[0] + values[1] * 500000 + values[2] + 5


def calibration_proposal(calibrations: list[dict]) -> dict:
    if len(calibrations) != 2:
        raise ValueError("both fixed-size calibration records required")
    observations = [_calibration_observation(record, count)
                    for record, count in zip(calibrations, (16384, 32768), strict=True)]
    rates, projected, overheads = zip(*observations, strict=True)
    for key in ("program_identity", "environment"):
        if calibrations[0].get(key) != calibrations[1].get(key):
            raise ValueError("calibration provenance differs")
    fast, slow = max(rates) * 1.02, min(rates) * .98
    target = max(550, max(projected) * 1.02 + 10)
    repeats = math.ceil(target * fast)
    reserve = max(20, max(overheads))
    if repeats + 500002 > 2**32 - 1 or repeats / slow + reserve > 600:
        raise ValueError("no conservative fixed count fits the 600-second proposal")
    return {"schema": "issue120-production-count-proposal-v1", "fixed_repetitions": repeats,
            "target_seconds": target, "predicted_slow_seconds": repeats / slow,
            "overhead_reserve_seconds": reserve, "planning_fraction": .02,
            "execution_authorized": False, "requires_precommit_and_owner_approval": True,
            "model_is_not_a_guarantee": True}
