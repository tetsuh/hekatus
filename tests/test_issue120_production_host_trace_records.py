"""New host diagnostics are versioned and never upgrade B4 evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "docs/measurements/issue120-production/2026-11-18-b4"


@pytest.fixture
def traced_native():
    manifest = json.loads((DIRECTORY / "B2-manifest.json").read_bytes())
    with tarfile.open(DIRECTORY / manifest["archive"]["file"]) as archive:
        old = json.load(archive.extractfile(manifest["record_file"]))["native"]
    native = copy.deepcopy(old)
    native["schema"] = "issue120-production-native-v2"
    cpu = native["host_cpu_work"]
    counts = [1, *range(65536, native["repeat_count"] + 1, 65536)]
    start = native["host_scheduling"]["enqueue_return_ns"]
    samples = [{"committed_count": count, "monotonic_ns": start + (cpu["end_ns"] - start) * (i + 1) // (len(counts) + 1)}
               for i, count in enumerate(counts)]
    native["host_phase_trace"] = {
        "schema": "issue120-host-submission-trace-v1", "clock": "CLOCK_MONOTONIC", "applicable": True,
        "period_writes": 65536, "capacity": 65536, "sample_storage_bytes": 1048576,
        "worker_start_ns": cpu["start_ns"], "worker_end_ns": cpu["end_ns"],
        "join_ns": native["host_scheduling"]["load_join_completed_ns"], "final_count": native["repeat_count"],
        "first_committed_enqueue_observed_ns": samples[0]["monotonic_ns"],
        "main_enqueue_return_observation": {"before_ns": start, "after_ns": start + 100, "committed_count": 0},
        "samples": samples}
    return native, old


def test_trace_requires_new_native_and_leaves_old_records_unchanged(traced_native):
    from enodia.tt.bench.issue120_production import host_trace

    native, old = traced_native
    before = copy.deepcopy(old)
    host_trace.validate_native(native, require_new=True)
    host_trace.validate_native(old)
    assert old == before
    with pytest.raises(ValueError):
        host_trace.validate_native(old, require_new=True)


@pytest.mark.parametrize("mutation", ["missing", "boolean_count", "cadence", "clock", "join", "snapshot"])
def test_malformed_new_trace_cannot_downgrade_to_legacy(traced_native, mutation):
    from enodia.tt.bench.issue120_production import host_trace

    native, _ = traced_native
    trace = native["host_phase_trace"]
    if mutation == "missing":
        del native["host_phase_trace"]
    elif mutation == "boolean_count":
        trace["main_enqueue_return_observation"]["committed_count"] = True
    elif mutation == "cadence":
        trace["samples"][1]["committed_count"] += 1
    elif mutation == "clock":
        trace["samples"][0]["monotonic_ns"] = trace["worker_start_ns"] - 1
    elif mutation == "join":
        trace["join_ns"] += 1
    else:
        trace["main_enqueue_return_observation"]["after_ns"] = trace["join_ns"] + 1
    with pytest.raises((ValueError, TypeError)):
        host_trace.validate_native(native)


def test_synthetic_trace_survives_full_native_envelope_readback(tmp_path, traced_native):
    from enodia.tt.bench.issue120_production import recording

    native, _ = traced_native
    manifest = json.loads((DIRECTORY / "B2-manifest.json").read_bytes())
    with tarfile.open(DIRECTORY / manifest["archive"]["file"]) as archive:
        archive.extractall(tmp_path, filter="data")
    primary = tmp_path / manifest["record_file"]
    record = json.loads(primary.read_bytes())
    record["native"] = native  # Synthetic fixture only; no archived bytes are edited.
    record["program_identity"]["host_trace_contract"] = "issue120-host-submission-trace-v1"
    capture = record["native_capture"]["result"]
    envelope_path = tmp_path / capture["file"]
    envelope = json.loads(envelope_path.read_bytes())
    envelope["result"] = native
    raw = json.dumps(envelope).encode()
    envelope_path.write_bytes(raw)
    capture.update(sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw))
    primary.write_text(json.dumps(record))
    reread = recording.load_validated_record(primary)
    assert reread["native"]["host_phase_trace"] == native["host_phase_trace"]
    assert reread["coverage_analysis"] == record["coverage_analysis"]
    # Matching edited envelope/fingerprints cannot downgrade a declared new contract.
    native["schema"] = "issue120-production-native-v1"
    del native["host_phase_trace"]
    raw = json.dumps(envelope).encode()
    envelope_path.write_bytes(raw)
    capture.update(sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw))
    primary.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="native version"):
        recording.load_validated_record(primary)
