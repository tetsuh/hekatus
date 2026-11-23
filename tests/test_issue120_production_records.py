"""New-schema-only host bindings and compatibility with accepted evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from enodia.tt.bench.issue120_probe import validate_probe_record
from enodia.tt.bench.issue120_production import diagnostics, records


def example():
    return {"schema": records.SCHEMA, "condition": "loaded", "native": {
        "host_interval": {"start_ns": 2, "end_ns": 7},
        "host_cpu_work": {"start_ns": 2, "end_ns": 7, "operations": 32768},
        "host_scheduling": {"enqueue_begin_ns": 1, "enqueue_return_ns": 3,
                            "load_join_completed_ns": 8, "completion_observer_joined_ns": 9,
                            "finish_begin_ns": 10, "finish_end_ns": 11, "drain_end_ns": 12},
        "resident_host_interval": {"start_ns": 1, "end_ns": 11,
                                   "scope": "enqueue_to_post_join_finish_return_upper_bound"},
    }}


def test_post_join_host_work_is_rejected_by_new_schema_only():
    record = example()
    records.validate_host_intervals(record)
    record["native"]["host_interval"] = {"start_ns": 9, "end_ns": 99}
    record["native"]["host_cpu_work"].update(start_ns=9, end_ns=99)
    with pytest.raises(ValueError, match="join"):
        records.validate_host_intervals(record)


@pytest.mark.parametrize("mutation", ["start", "end", "finish", "scope", "bool"])
def test_interval_binding_counterexamples(mutation):
    record = example()
    native = record["native"]
    if mutation == "start":
        native["host_cpu_work"]["start_ns"] = 3
    elif mutation == "end":
        native["host_cpu_work"]["end_ns"] = 6
    elif mutation == "finish":
        native["host_scheduling"]["finish_begin_ns"] = 4
    elif mutation == "scope":
        native["resident_host_interval"]["scope"] = "exact_resident_completion"
    else:
        native["host_interval"]["start_ns"] = True
    with pytest.raises(ValueError):
        records.validate_host_intervals(record)


def test_accepted_v3_records_and_all_bytes_are_unchanged():
    base = Path("docs/measurements")
    for role in ("p6", "p7"):
        path = base / (f"2026-11-14-p150a-issue120-{role}-api-lock-semaphore-v3-wrapper-result.json")
        raw = path.read_bytes()
        record = json.loads(raw)
        old = copy.deepcopy(record)
        assert validate_probe_record(record) == []
        with pytest.raises(records.SchemaOutOfScope):
            records.validate_host_intervals(record)
        assert old == record
        assert path.read_bytes() == raw


@pytest.mark.parametrize(("consumed", "configured"), [(0, 4096), (17, 4096), (500000, 500000)])
def test_phase_correct_diagnostic_expectations(consumed, configured):
    result = diagnostics.expectations(consumed, configured)
    assert result["expected_after_consumed_frames"] == consumed + 2
    assert result["expected_after_full_run"] == configured + 2


def test_control_does_not_fabricate_host_work_interval():
    record = example()
    record["condition"] = "control"
    record["native"]["host_interval"] = None
    record["native"]["host_cpu_work"] = {"operations": 0, "reads": 0, "transfers": 0}
    records.validate_host_intervals(record)
    record["native"]["host_cpu_work"]["operations"] = 1
    with pytest.raises(ValueError):
        records.validate_host_intervals(record)
