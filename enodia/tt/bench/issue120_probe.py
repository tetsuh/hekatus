"""Board-free contracts and CLI for the Issue #120 feasibility probes.

The native Metalium runner is intentionally isolated under ``issue120/``.
Importing this module never imports TTNN or opens a device.  Execution is
opt-in and is only supported from the digest-pinned benchmark container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from enodia.tt.bench import issue120_results, issue120_runtime
from enodia.tt.bench.board_identity import normalize_board_id, normalize_pci_bus_id
from enodia.tt.bench.issue120_coverage import (
    CPU_ROUNDS_PER_TRANSFER,
    CPU_WORDS_PER_ROUND,
    FRAME_COUNT,
    FRAME_PERIOD_TICKS,
    MAX_PROGRESS_GENERATION,
    PROGRESS_BYTES,
    TRANSFER_REPEATS,
    evaluate_frame_coverage,
)
from enodia.tt.bench.load_harness import (
    HOST_TRANSFER_BUFFER_POOL_SIZE,
    HOST_TRANSFER_BYTES,
    HOST_TRANSFER_CORE,
    RESIDENT_CORES,
)

PROBE_SCHEMA = "issue120-probe-v1"
DEFAULT_PROBE_CAP_SECONDS = 30
MAX_PROBE_CAP_SECONDS = 60
PINNED_IMAGE_DIGEST = "sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"
Q1_TRANSFER_COUNT = 1
Q2_TRANSFER_COUNT = 256
Q3_PAGE_BYTES = 4_096
Q3_PAGE_COUNT = 256
Q4_TRANSFER_COUNT = 256
Q4_CPU_ROUNDS = 1_024
Q4_CPU_WORDS = 1_024


class ProbeRecordError(ValueError):
    """A probe plan or persisted observation violates its contract."""


@dataclass(frozen=True)
class ProbeSpec:
    """Fixed-work and allocation contract for one probe family."""

    probe_id: str
    purpose: str
    repeat_count: int
    transfer_bytes: int
    buffer_pool_size: int
    host_buffer_slots: int
    device_buffer_slots: int
    load_cores: tuple[tuple[int, int], ...]
    host_cpu_rounds: int = 0
    host_cpu_words_per_round: int = 0
    resident_cores: tuple[tuple[int, int], ...] = RESIDENT_CORES
    outer_cap_seconds: int = DEFAULT_PROBE_CAP_SECONDS
    watcher: bool = False
    max_outstanding_transfers: int | None = None
    auxiliary_bytes: int = 3 * 4_096
    allocation_policy: str = "fixed_pool_with_bounded_runtime_queue_backpressure"
    queue_bound_evidence: str = "fixed_metalium_command_ring;occupancy_not_sampled"

    @property
    def allocated_bytes(self) -> int:
        return (self.host_buffer_slots + self.device_buffer_slots) * self.transfer_bytes + self.auxiliary_bytes

    def as_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["load_cores"] = [list(core) for core in self.load_cores]
        result["resident_cores"] = [list(core) for core in self.resident_cores]
        result["allocated_bytes"] = self.allocated_bytes
        result["device_open"] = {"device_id": 0, "command_queues": 2}
        return result


_LOAD_CORE = (HOST_TRANSFER_CORE,)
ISSUE120_PROBES = (
    ProbeSpec(
        probe_id="q1-cq1-completion",
        purpose=(
            "Observe a complete CQ1 payload on the resident core; bracket its "
            "visibility with that core's card-tick window and fixed tail."
        ),
        repeat_count=Q1_TRANSFER_COUNT,
        transfer_bytes=HOST_TRANSFER_BYTES,
        buffer_pool_size=HOST_TRANSFER_BUFFER_POOL_SIZE,
        host_buffer_slots=1,
        device_buffer_slots=HOST_TRANSFER_BUFFER_POOL_SIZE,
        load_cores=_LOAD_CORE,
        max_outstanding_transfers=1,
    ),
    ProbeSpec(
        probe_id="q2-h2d-write-only",
        purpose=(
            "Test bounded repeated H2D writes from one immutable, ordinary unpinned "
            "host buffer without per-repeat completion reads or readback."
        ),
        repeat_count=Q2_TRANSFER_COUNT,
        transfer_bytes=HOST_TRANSFER_BYTES,
        buffer_pool_size=HOST_TRANSFER_BUFFER_POOL_SIZE,
        host_buffer_slots=1,
        device_buffer_slots=HOST_TRANSFER_BUFFER_POOL_SIZE,
        load_cores=(),
    ),
    ProbeSpec(
        probe_id="q3-pinned-noc",
        purpose=(
            "Run a fixed read/write stream from one disjoint worker core to a "
            "pinned host allocation mapped for NoC access."
        ),
        repeat_count=Q3_PAGE_COUNT,
        transfer_bytes=Q3_PAGE_BYTES,
        buffer_pool_size=1,
        host_buffer_slots=3,
        device_buffer_slots=0,
        load_cores=_LOAD_CORE,
        host_cpu_rounds=Q4_CPU_ROUNDS,
        host_cpu_words_per_round=Q4_CPU_WORDS,
        max_outstanding_transfers=1,
        auxiliary_bytes=21 * Q3_PAGE_BYTES,
    ),
    ProbeSpec(
        probe_id="q4-fixed-window",
        purpose=(
            "Run fixed host CPU arithmetic and H2D writes during a short synthetic "
            "resident observer window; report coverage without claiming frames."
        ),
        repeat_count=Q4_TRANSFER_COUNT,
        transfer_bytes=HOST_TRANSFER_BYTES,
        buffer_pool_size=HOST_TRANSFER_BUFFER_POOL_SIZE,
        host_buffer_slots=1,
        device_buffer_slots=HOST_TRANSFER_BUFFER_POOL_SIZE,
        load_cores=(),
        host_cpu_rounds=Q4_CPU_ROUNDS,
        host_cpu_words_per_round=Q4_CPU_WORDS,
    ),
)
COVERAGE_PROBES = (
    ProbeSpec(
        probe_id="q4-frame-coverage",
        purpose="Verify card-clock progress across every actual resident ring frame.",
        repeat_count=TRANSFER_REPEATS, transfer_bytes=PROGRESS_BYTES,
        buffer_pool_size=1, host_buffer_slots=1, device_buffer_slots=1,
        load_cores=(), outer_cap_seconds=60, host_cpu_rounds=CPU_ROUNDS_PER_TRANSFER,
        host_cpu_words_per_round=CPU_WORDS_PER_ROUND,
        auxiliary_bytes=(FRAME_COUNT + 11) * PROGRESS_BYTES + 32,
    ),
)
_PROBE_BY_ID = {probe.probe_id: probe for probe in (*ISSUE120_PROBES, *COVERAGE_PROBES)}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BUILD_ROOT = REPOSITORY_ROOT / "out/issue120-native-build"
DEPENDENCY_ROOT = REPOSITORY_ROOT / "out/issue120-deps"
SOURCE_ROOT = DEPENDENCY_ROOT / "source"
INCLUDE_ROOT = DEPENDENCY_ROOT / "include"
LIBRARY_ROOT = DEPENDENCY_ROOT / "lib"
WHEEL_LIBRARY_ROOT = DEPENDENCY_ROOT / "wheel-libs"
OUTPUT_ROOT = Path("/out")
PINNED_SOURCE_REVISION = "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd"
PINNED_LIBRARY_HASH = "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141"
_ENVIRONMENT_FIELDS = (
    "board_model",
    "serial",
    "pci",
    "firmware",
    "kmd",
    "image_digest",
    "cpu_model",
    "commit",
)
_SECRET_OR_MACHINE_FIELDS = {"hostname", "host_name", "machine_name", "username", "user"}
_ABSOLUTE_PATH = re.compile(r"(?:^|[\s=:])/(?:home|tmp|mnt|work)/[^\s]+")


class LowWordClock:
    """Software-extend sampled low 32-bit card timestamps with a gap guard."""

    def __init__(self, *, maximum_unobserved_ticks: int = 1 << 31) -> None:
        self.maximum_unobserved_ticks = maximum_unobserved_ticks
        self.last_low: int | None = None
        self.extended_ticks: int | None = None

    def observe(self, low_word: int) -> int:
        if isinstance(low_word, bool) or not isinstance(low_word, int) or not 0 <= low_word < 1 << 32:
            raise ProbeRecordError("card timestamp must be a uint32 low word")
        if self.last_low is None:
            self.last_low = low_word
            self.extended_ticks = low_word
            return low_word
        delta = (low_word - self.last_low) & 0xFFFFFFFF
        if delta >= self.maximum_unobserved_ticks:
            raise ProbeRecordError("card timestamp sampling gap is ambiguous")
        assert self.extended_ticks is not None
        self.extended_ticks += delta
        self.last_low = low_word
        return self.extended_ticks


def allocate_low_word_clock() -> LowWordClock:
    """Return a wrap-tracking clock with a half-range sampling-gap guard."""
    return LowWordClock()


def planned_probe(
    probe_id: str,
    *,
    repeat_count: int | None = None,
    outer_cap_seconds: int | None = None,
    first_kernel_run: bool = False,
) -> ProbeSpec:
    """Resolve a safe execution plan without importing or opening device APIs."""
    try:
        base = _PROBE_BY_ID[probe_id]
    except KeyError as exc:
        raise ProbeRecordError(f"unknown Issue #120 probe {probe_id!r}") from exc
    repeats = _probe_repeats(base, repeat_count)
    cap = _probe_cap(base, outer_cap_seconds, first_kernel_run)
    fields = asdict(base)
    fields.update(repeat_count=repeats, outer_cap_seconds=cap, watcher=first_kernel_run)
    return ProbeSpec(**fields)


def _probe_repeats(base: ProbeSpec, requested: int | None) -> int:
    repeats = base.repeat_count if requested is None else requested
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ProbeRecordError("repeat_count must be a positive integer")
    if repeats > 0xFFFFFFFF:
        raise ProbeRecordError("repeat_count must fit the native uint32 work counter")
    if base.probe_id == "q1-cq1-completion" and repeats != 1:
        raise ProbeRecordError("Q1 uses one payload whose every page is card-side witnessed")
    if base.probe_id == "q4-frame-coverage" and repeats > MAX_PROGRESS_GENERATION:
        raise ProbeRecordError("Q4 repeat_count exceeds the unique 19-bit progress field")
    return repeats


def _probe_cap(base: ProbeSpec, requested: int | None, first: bool) -> int:
    cap = requested
    if cap is None:
        cap = base.outer_cap_seconds
        if first:
            cap = MAX_PROBE_CAP_SECONDS
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_PROBE_CAP_SECONDS:
        raise ProbeRecordError("Issue #120 probes require a 60-second cap or less")
    if first and cap != MAX_PROBE_CAP_SECONDS:
        raise ProbeRecordError("first kernel run requires the exact 60-second cap")
    return cap


def card_tick_window_relation(
    *,
    resident_start_ticks: int,
    resident_end_ticks: int,
    transfer_completion_ticks: int | None,
) -> str:
    """Classify an exact card-side transfer marker timestamp."""
    if (
        isinstance(resident_start_ticks, bool)
        or isinstance(resident_end_ticks, bool)
        or resident_start_ticks < 0
        or resident_end_ticks < resident_start_ticks
    ):
        raise ProbeRecordError("resident card-tick window is invalid")
    if transfer_completion_ticks is None:
        return "not_evaluable"
    if (
        isinstance(transfer_completion_ticks, bool)
        or transfer_completion_ticks < 0
    ):
        raise ProbeRecordError("transfer completion card ticks are invalid")
    if resident_start_ticks <= transfer_completion_ticks <= resident_end_ticks:
        return "inside_resident_window"
    if transfer_completion_ticks > resident_end_ticks:
        return "after_resident_window"
    return "before_resident_window"


def card_tick_interval_relation(
    *,
    resident_start_ticks: int,
    resident_end_ticks: int,
    completion_lower_ticks: int | None,
    completion_upper_ticks: int | None,
) -> str:
    """Classify a marker completion bracket without guessing across a boundary."""
    if completion_lower_ticks is None or completion_upper_ticks is None:
        return "not_evaluable"
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (resident_start_ticks, resident_end_ticks, completion_lower_ticks, completion_upper_ticks)
    ):
        raise ProbeRecordError("card tick interval endpoints must be non-negative integers")
    if resident_end_ticks < resident_start_ticks:
        raise ProbeRecordError("resident card-tick window is reversed")
    if completion_lower_ticks > completion_upper_ticks:
        raise ProbeRecordError("completion timestamp bracket is reversed")
    if completion_upper_ticks <= resident_start_ticks:
        return "before_resident_window"
    if completion_lower_ticks >= resident_end_ticks:
        return "after_resident_window"
    if completion_lower_ticks >= resident_start_ticks and completion_upper_ticks <= resident_end_ticks:
        return "inside_resident_window"
    return "not_evaluable"


def fixed_window_coverage(
    *, cpu_start_ns: int, cpu_end_ns: int, resident_start_ns: int, resident_end_ns: int,
) -> str:
    """Falsify insufficient CPU coverage; one payload cannot prove PCIe coverage."""
    if cpu_start_ns > resident_start_ns or cpu_end_ns < resident_end_ns:
        return "not_covered"
    return "not_evaluable_pcie_progress_not_observed"


def _reject_sensitive_fields(value: Any, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _SECRET_OR_MACHINE_FIELDS:
                raise ProbeRecordError(f"{path} contains a prohibited machine-specific field")
            _reject_sensitive_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_fields(child, f"{path}[{index}]")
    elif isinstance(value, str) and _ABSOLUTE_PATH.search(value):
        raise ProbeRecordError(f"{path} contains an absolute host path")


def _plan_record(spec: ProbeSpec, *, first_kernel_run: bool) -> dict[str, Any]:
    """Stable record binding for one selected fixed-work probe plan."""
    return {
        "probe_id": spec.probe_id,
        "repeat_count": spec.repeat_count,
        "transfer_bytes": spec.transfer_bytes,
        "allocated_bytes": spec.allocated_bytes,
        "buffer_pool_size": spec.buffer_pool_size,
        "host_buffer_slots": spec.host_buffer_slots,
        "device_buffer_slots": spec.device_buffer_slots,
        "auxiliary_bytes": spec.auxiliary_bytes,
        "host_cpu_rounds": spec.host_cpu_rounds,
        "host_cpu_words_per_round": spec.host_cpu_words_per_round,
        "max_outstanding_transfers": spec.max_outstanding_transfers,
        "resident_cores": [list(core) for core in spec.resident_cores],
        "load_cores": [list(core) for core in spec.load_cores],
        "device_open": {"device_id": 0, "command_queues": 2},
        "outer_cap_seconds": spec.outer_cap_seconds,
        "watcher": spec.watcher,
        "first_kernel_run": first_kernel_run,
        "allocation_policy": spec.allocation_policy,
        "queue_bound_evidence": spec.queue_bound_evidence,
    }


def _validate_record_header(record: dict[str, Any]) -> str:
    """Validate schema, identity, and pinned source provenance."""
    if not isinstance(record, dict):
        raise ProbeRecordError("probe record must be an object")
    _reject_sensitive_fields(record)
    if record.get("schema") != PROBE_SCHEMA:
        raise ProbeRecordError("probe record schema is missing or unsupported")
    probe_id = record.get("probe_id")
    if probe_id not in _PROBE_BY_ID:
        raise ProbeRecordError("probe_id is unknown")
    environment = record.get("environment")
    if not isinstance(environment, dict):
        raise ProbeRecordError("environment must be an object")
    missing = [field for field in _ENVIRONMENT_FIELDS if not environment.get(field)]
    if missing:
        raise ProbeRecordError("environment is missing " + ", ".join(missing))
    if environment["image_digest"] != PINNED_IMAGE_DIGEST:
        raise ProbeRecordError("environment.image_digest differs from the Issue #120 pin")
    if not re.fullmatch(r"[0-9a-f]{40}", environment["commit"]):
        raise ProbeRecordError("environment.commit must be a full commit id")
    source = record.get("source")
    if not isinstance(source, dict) or source.get("tt_metal_tag") != "v0.75.0":
        raise ProbeRecordError("native source tag provenance is required")
    if source.get("tt_metal_commit") != "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd":
        raise ProbeRecordError("native source commit differs from the pinned tag")
    if source.get("libtt_metal_sha256") != "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141":
        raise ProbeRecordError("native library digest differs from the pinned image")
    return _PROBE_BY_ID[probe_id].probe_id


def _validate_record_sections(record: dict[str, Any]) -> None:
    resident = record.get("resident_interval")
    host = record.get("host_interval")
    work = record.get("work")
    memory = record.get("memory")
    if not all(isinstance(item, dict) for item in (resident, host, work, memory)):
        raise ProbeRecordError("resident, host, work, and memory evidence are required")
    for obj, start, end, label in (
        (resident, "start_ticks", "end_ticks", "resident card-tick interval"),
        (host, "start_ns", "end_ns", "host monotonic interval"),
    ):
        if any(isinstance(obj.get(key), bool) or not isinstance(obj.get(key), int) for key in (start, end)):
            raise ProbeRecordError(f"{label} endpoints must be integers")
        if obj[end] < obj[start]:
            raise ProbeRecordError(f"{label} end precedes start")


def _validate_work(record: dict[str, Any]) -> None:
    work = record["work"]
    probe_id = record["probe_id"]
    count_fields = ("repeat_count", "transfer_count", "bytes_per_transfer", "bytes_total")
    if any(isinstance(work.get(key), bool) or not isinstance(work.get(key), int) or work[key] < 0 for key in count_fields):
        raise ProbeRecordError("work counters must be non-negative integers")
    if work["bytes_total"] != work["transfer_count"] * work["bytes_per_transfer"]:
        raise ProbeRecordError("work byte total does not match transfer count")
    expected_transfers = work["repeat_count"] * (2 if probe_id == "q3-pinned-noc" else 1)
    if work["repeat_count"] < 1 or work["transfer_count"] != expected_transfers:
        raise ProbeRecordError("transfer counters disagree with fixed work")


def _validate_memory(record: dict[str, Any]) -> None:
    memory = record["memory"]
    for field in ("buffer_pool_size", "allocated_bytes"):
        if isinstance(memory.get(field), bool) or not isinstance(memory.get(field), int) or memory[field] < 0:
            raise ProbeRecordError(f"memory.{field} must be a non-negative integer")
    outstanding = memory.get("max_outstanding_transfers")
    if outstanding is not None and (
        isinstance(outstanding, bool) or not isinstance(outstanding, int) or outstanding < 0
    ):
        raise ProbeRecordError("memory.max_outstanding_transfers must be a count or null")
    if outstanding is None and not memory.get("outstanding_bound"):
        raise ProbeRecordError("unknown in-flight count requires its fixed queue-bound evidence")


def _validate_cap(record: dict[str, Any]) -> None:
    cap = record.get("outer_cap_seconds")
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_PROBE_CAP_SECONDS:
        raise ProbeRecordError("outer cap exceeds the 60-second limit")
    if not isinstance(record.get("watcher"), bool):
        raise ProbeRecordError("watcher must be boolean")
    first_kernel_run = record.get("first_kernel_run")
    if not isinstance(first_kernel_run, bool):
        raise ProbeRecordError("first_kernel_run must be boolean")
    if first_kernel_run and (cap != 60 or record["watcher"] is not True):
        raise ProbeRecordError("first kernel runs require Watcher and the exact 60-second cap")


def _validate_plan_identity(record: dict[str, Any]) -> None:
    probe_id = record["probe_id"]
    cap = record["outer_cap_seconds"]
    first_kernel_run = record["first_kernel_run"]
    work = record["work"]
    plan = record.get("plan")
    if not isinstance(plan, dict) or plan.get("probe_id") != probe_id:
        raise ProbeRecordError("record must bind to its probe plan")
    if plan.get("outer_cap_seconds") != cap or plan.get("watcher") != record["watcher"]:
        raise ProbeRecordError("record cap or Watcher differs from its plan")
    if plan.get("first_kernel_run") != first_kernel_run:
        raise ProbeRecordError("record first-kernel state differs from its plan")
    if plan.get("repeat_count") != work.get("repeat_count"):
        raise ProbeRecordError("record fixed repeat count differs from its plan")
    if plan.get("transfer_bytes") != work.get("bytes_per_transfer"):
        raise ProbeRecordError("record transfer size differs from its plan")


def _validate_plan_memory(record: dict[str, Any]) -> None:
    plan = record["plan"]
    memory = record["memory"]
    if plan.get("allocated_bytes") != memory.get("allocated_bytes"):
        raise ProbeRecordError("record memory allocation differs from its plan")
    if plan.get("allocated_bytes") != (
        memory.get("host_payload_bytes", 0)
        + memory.get("device_buffer_bytes", 0)
        + memory.get("auxiliary_bytes", 0)
    ):
        raise ProbeRecordError("record allocation breakdown does not sum to its fixed total")
    if plan.get("max_outstanding_transfers") != memory.get("max_outstanding_transfers"):
        raise ProbeRecordError("record in-flight claim differs from its plan")
    if plan.get("buffer_pool_size") != memory.get("buffer_pool_size"):
        raise ProbeRecordError("record fixed buffer pool differs from its plan")
    if memory.get("max_outstanding_transfers") is None and not memory.get("outstanding_bound"):
        raise ProbeRecordError("unknown queue occupancy requires bounded-ring evidence")


def _validate_partition(record: dict[str, Any]) -> None:
    plan = record["plan"]
    spec = _PROBE_BY_ID[record["probe_id"]]
    if plan.get("resident_cores") != [list(core) for core in spec.resident_cores]:
        raise ProbeRecordError("record resident core partition differs from the plan")
    if plan.get("load_cores") != [list(core) for core in spec.load_cores]:
        raise ProbeRecordError("record load core partition differs from the plan")


def _validate_status(record: dict[str, Any]) -> None:
    errors = record.get("errors")
    if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
        raise ProbeRecordError("errors must be a list of strings")
    status = record.get("status")
    if status not in {"complete", "not_evaluable", "error"}:
        raise ProbeRecordError("status must be complete, not_evaluable, or error")
    if status == "complete" and errors:
        raise ProbeRecordError("complete probe records cannot contain errors")
    if status in {"not_evaluable", "error"} and not errors and not record.get("not_evaluable_reason"):
        raise ProbeRecordError("non-evaluable probe records require an explicit reason")


def _validate_q1_completion(record: dict[str, Any]) -> None:
    if record["status"] != "complete":
        return
    tick = record.get("transfer_completion_ticks")
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise ProbeRecordError("Q1 completion evidence must be card-side timestamps")
    resident = record["resident_interval"]
    relation = card_tick_interval_relation(
        resident_start_ticks=resident["start_ticks"], resident_end_ticks=resident["end_ticks"],
        completion_lower_ticks=record.get("transfer_completion_lower_ticks", tick),
        completion_upper_ticks=record.get("transfer_completion_upper_ticks", tick),
    )
    if record.get("transfer_completion_relation") != relation:
        raise ProbeRecordError("Q1 card-side completion classification disagrees with ticks")
    _validate_q1_active_interval(record)


def _validate_q1_active_interval(record: dict[str, Any]) -> None:
    interval = record.get("resident_program_interval")
    if not isinstance(interval, dict):
        raise ProbeRecordError("Q1 must record the actual observer active interval")
    if not isinstance(interval.get("observer_end_ticks"), int):
        raise ProbeRecordError("Q1 observer program endpoint must be card-side")
    relation = card_tick_interval_relation(
        resident_start_ticks=interval.get("start_ticks"),
        resident_end_ticks=interval.get("observer_end_ticks"),
        completion_lower_ticks=record.get("transfer_completion_lower_ticks"),
        completion_upper_ticks=record.get("transfer_completion_upper_ticks"),
    )
    names = {"before_resident_window": "before_resident_program", "inside_resident_window": "resident_program_active"}
    expected = names.get(relation, "not_evaluable")
    if record.get("transfer_program_relation") != expected:
        raise ProbeRecordError("Q1 must distinguish primary-window timing from program activity")
    if record.get("transfer_program_relation") == "after_resident_program":
        raise ProbeRecordError("a primary-window miss is not evidence of post-program completion")


def _validate_q2(record: dict[str, Any]) -> None:
    if record.get("host_buffer_mode") != "ordinary_unpinned_immutable_raw_host_pointer":
        raise ProbeRecordError("Q2 must use an ordinary unpinned immutable raw host pointer")
    if record.get("explicit_completion_queue_reads_during_submission") is not False:
        raise ProbeRecordError("Q2 must not explicitly read CQ completion during submission")
    if record.get("completion_reads_during_final_drain") is not True:
        raise ProbeRecordError("Q2 must record its one final CQ drain accurately")
    if record.get("host_readback_during_submission") is not False:
        raise ProbeRecordError("Q2 must not read transfer data back to the host during submission")
    if record.get("host_enqueue_may_backpressure") is not True:
        raise ProbeRecordError("Q2 must disclose possible fixed-ring enqueue backpressure")
    if record.get("device_wait_stream_may_stall") is not True:
        raise ProbeRecordError("Q2 must distinguish no host wait from device progress")
    drain = record.get("final_drain_interval")
    if not isinstance(drain, dict) or drain.get("end_ns", 0) < drain.get("start_ns", 0):
        raise ProbeRecordError("Q2 final completion drain interval is required")


def _validate_q3_address(record: dict[str, Any]) -> None:
    noc = record.get("noc_preflight")
    payload = record.get("load_payload_validation")
    if not isinstance(noc, dict) or noc.get("noc_address_available") is not True:
        raise ProbeRecordError("Q3 requires a pinned NOC address and preflight record")
    if record["status"] == "complete" and (not isinstance(payload, dict) or payload.get("valid") is not True):
        raise ProbeRecordError("Q3 cannot be complete without read/write payload verification")
    if noc.get("requested_pin_bytes", 0) > noc.get("max_total_pin_size", -1):
        raise ProbeRecordError("Q3 pinned-memory request exceeds the reported capacity")
    if noc.get("device_id") != noc.get("noc_address_device_id"):
        raise ProbeRecordError("Q3 NOC address is not for the selected MMIO device")
    if noc.get("can_map_to_noc") is not True:
        raise ProbeRecordError("Q3 runtime preflight did not advertise NoC-accessible pinned memory")


def _validate_q3_work(record: dict[str, Any]) -> None:
    if record.get("cross_core_tick_comparison") != "not_assumed":
        raise ProbeRecordError("Q3 must not compare unrelated worker and observer timestamps")
    if record.get("coverage") != "not_evaluable_cross_core_ticks_not_compared":
        raise ProbeRecordError("Q3 coverage remains explicitly unclassified without a shared clock")
    work = record["work"]
    if work.get("load_worker_reads") != work.get("repeat_count"):
        raise ProbeRecordError("Q3 read count differs from fixed work")
    if work.get("load_worker_writes") != work.get("repeat_count"):
        raise ProbeRecordError("Q3 write count differs from fixed work")
    noc = record["noc_preflight"]
    if noc.get("pinned_buffer_bytes", 0) < noc.get("requested_pin_bytes", 0):
        raise ProbeRecordError("Q3 pinned allocation is smaller than the requested host range")
    if noc.get("max_pins", 0) < 1 or noc.get("required_aligned_pin_bytes", 0) > noc.get("max_total_pin_size", -1):
        raise ProbeRecordError("Q3 pinned allocation exceeds the preflight count/byte capacity")


def _validate_cpu_work(record: dict[str, Any]) -> None:
    cpu = record.get("host_cpu_work")
    if not isinstance(cpu, dict) or cpu.get("operations") != cpu.get("rounds", 0) * cpu.get("words_per_round", 0):
        raise ProbeRecordError("fixed host CPU work count is absent or inconsistent")
    plan = record["plan"]
    if cpu.get("rounds") != plan.get("host_cpu_rounds") or cpu.get("words_per_round") != plan.get("host_cpu_words_per_round"):
        raise ProbeRecordError("fixed host CPU work differs from its committed plan")
    if cpu.get("end_ns", 0) < cpu.get("start_ns", 0):
        raise ProbeRecordError("fixed host CPU work interval is reversed")


def _validate_legacy_q4(record: dict[str, Any]) -> None:
    if record.get("coverage_scope") != "one_payload_witness_in_short_synthetic_observer_window_not_frame_stream":
        raise ProbeRecordError("Q4 coverage scope must disclaim unexecuted frame coverage")
    program = record.get("resident_host_interval", {})
    cpu = record["host_cpu_work"]
    expected = fixed_window_coverage(
        cpu_start_ns=cpu["start_ns"], cpu_end_ns=cpu["end_ns"],
        resident_start_ns=program["start_ns"], resident_end_ns=program["end_ns"],
    )
    if record.get("coverage") != expected:
        raise ProbeRecordError("Q4 cannot claim full-window coverage from one payload witness")


def _validate_frame_host_scope(record: dict[str, Any]) -> None:
    program = record.get("resident_host_interval", {})
    scheduling = record.get("host_scheduling")
    if not isinstance(program, dict):
        raise ProbeRecordError("Q4 host scope metadata must be an object")
    if scheduling is None and "scope" not in program:
        return  # Historical records retain their original host metadata.
    if not isinstance(scheduling, dict) or scheduling.get("policy") != "join_before_any_finish":
        raise ProbeRecordError("Q4 post-join host scope requires explicit scheduling semantics")
    if program.get("scope") != "enqueue_to_post_join_finish_return_upper_bound":
        raise ProbeRecordError("Q4 post-join host scope is an upper bound, not resident completion")
    times = [program.get("start_ns"), scheduling.get("load_join_completed_ns"), program.get("end_ns")]
    if not all(type(value) is int and value >= 0 for value in times) or times != sorted(times):
        raise ProbeRecordError("Q4 post-join host scope timestamps are invalid")
    if scheduling.get("resident_finish_returned_ns") != program["end_ns"]:
        raise ProbeRecordError("Q4 post-join host scope endpoint differs from finish return")


def _validate_frame_probe(record: dict[str, Any]) -> None:
    _validate_frame_host_scope(record)
    evidence = record.get("frame_coverage", {})
    if not isinstance(evidence, dict):
        raise ProbeRecordError("Q4 frame evidence must be an object")
    result = evaluate_frame_coverage(evidence)
    if evidence.get("frame_count") != FRAME_COUNT:
        raise ProbeRecordError("Q4 must bind to its fixed real frame count")
    if evidence.get("repeat_count") != record["work"]["repeat_count"]:
        raise ProbeRecordError("Q4 progress counter limit differs from fixed work")
    if evidence.get("frame_period_ticks") != FRAME_PERIOD_TICKS:
        raise ProbeRecordError("Q4 must bind to its fixed nominal frame period")
    if record.get("cpu_challenge_protocol") != "direct_l1_span_read_then_fixed_cpu_work_then_cq1_write":
        raise ProbeRecordError("Q4 must retain causal CPU challenge observation")
    if record.get("coverage_scope") != "all_real_frame_completion_intervals_consumer_clock":
        raise ProbeRecordError("Q4 frame coverage scope is missing")
    if record.get("coverage") != result["coverage"]:
        raise ProbeRecordError("Q4 declared coverage differs from card-side frame evidence")
    if record["status"] == "complete" and result["coverage"] != "covered":
        raise ProbeRecordError("Q4 cannot be complete without every frame covered")


def _validate_frame_cpu(record: dict[str, Any]) -> None:
    cpu = record.get("host_cpu_work", {})
    if not isinstance(cpu, dict):
        raise ProbeRecordError("Q4 CPU work must be an object")
    repeats = record["work"]["repeat_count"]
    expected = {
        "rounds": CPU_ROUNDS_PER_TRANSFER, "words_per_round": CPU_WORDS_PER_ROUND,
        "outer_repetitions": repeats,
        "operations": repeats * CPU_ROUNDS_PER_TRANSFER * CPU_WORDS_PER_ROUND,
    }
    for field, value in expected.items():
        if cpu.get(field) != value:
            raise ProbeRecordError("Q4 fixed CPU work differs from the committed transfer plan")
    work = record["work"]
    if work.get("host_mmio_reads") != repeats or work.get("host_mmio_payload_bytes") != 4 * repeats:
        raise ProbeRecordError("Q4 challenge read accounting differs from fixed work")


def validate_probe_record(record: dict[str, Any]) -> list[str]:
    """Apply the same legacy invariants separately, without weakening any gate."""
    probe_id = _validate_record_header(record)
    common = (
        _validate_record_sections, _validate_work, _validate_memory, _validate_cap,
        _validate_plan_identity, _validate_plan_memory, _validate_partition, _validate_status,
    )
    for check in common:
        check(record)
    checks = {
        "q1-cq1-completion": (_validate_q1_completion,),
        "q2-h2d-write-only": (_validate_q2,),
        "q3-pinned-noc": (_validate_q3_address, _validate_q3_work, _validate_cpu_work),
        "q4-fixed-window": (_validate_cpu_work, _validate_legacy_q4),
        "q4-frame-coverage": (_validate_frame_probe, _validate_frame_cpu),
    }
    for check in checks[probe_id]:
        check(record)
    return []


def _plan_json() -> dict[str, Any]:
    return {
        "schema": PROBE_SCHEMA,
        "board_access": False,
        "probes": [planned_probe(probe.probe_id).as_record() | {"purpose": probe.purpose} for probe in ISSUE120_PROBES],
        "coverage_probes": [probe.as_record() for probe in COVERAGE_PROBES],
        "q1_stall_group_variants": ["resident-and-transfer", "transfer-only"],
        "first_kernel_run": {"watcher": True, "outer_cap_seconds": MAX_PROBE_CAP_SECONDS},
        "normal_probe_cap_seconds": DEFAULT_PROBE_CAP_SECONDS,
        "q4_fixed_cpu_work": {"rounds": Q4_CPU_ROUNDS, "words_per_round": Q4_CPU_WORDS},
    }


def _environment_record(path: str) -> dict[str, str]:
    """Reduce the wrapper snapshot to sanitized ADR-0005 identity fields."""
    try:
        snapshot = _bounded_path(path, OUTPUT_ROOT, "environment", "file")
        raw = json.loads(snapshot.read_text(encoding="utf-8"))
        board = raw["board"]
        firmware = raw["firmware"]
        image = raw["image"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ProbeRecordError("wrapper environment snapshot is incomplete") from exc
    image_digest = str(image).rsplit("@", 1)[-1]
    if raw.get("image_pinned") is not True or image_digest != PINNED_IMAGE_DIGEST:
        raise ProbeRecordError("native probes require the Issue #120 pinned image digest")
    if raw.get("harness_dirty") is not False or raw.get("harness_pushed") is not True:
        raise ProbeRecordError("probe harness commit must be clean and pushed before execution")
    commit = raw.get("harness_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProbeRecordError("wrapper harness_commit is not a full commit id")
    cpu_model = ""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ProbeRecordError("CPU model is unavailable") from exc
    for line in cpuinfo.splitlines():
        if line.lower().startswith(("model name", "hardware")):
            cpu_model = line.split(":", 1)[-1].strip()
            break
    serial = normalize_board_id(board.get("serial") or board.get("board_id"))
    pci = normalize_pci_bus_id(board.get("bus_id"))
    firmware_version = firmware.get("fw_bundle_version")
    kmd = raw.get("kmd_version")
    board_model = board.get("board_type")
    if not all((serial, pci, firmware_version, kmd, board_model, cpu_model)):
        raise ProbeRecordError("board, PCI, firmware, KMD, or CPU identity is missing")
    if board_model != "p150a":
        raise ProbeRecordError("Issue #120 native probes require the recorded device-0 p150a")
    return {
        "board_model": str(board_model),
        "serial": serial,
        "pci": pci,
        "firmware": str(firmware_version),
        "kmd": str(kmd),
        "image_digest": image_digest,
        "cpu_model": cpu_model,
        "commit": commit,
    }


def _bounded_path(value: str | Path, root: Path, label: str, kind: str) -> Path:
    """Break CLI path taint before file access or command construction."""
    if not value or any(character in str(value) for character in (";", "$", "\n", "\r", "\x00")):
        raise ProbeRecordError(f"{label} path contains forbidden command syntax")
    expected_root = root.resolve()
    if root.absolute() != expected_root:
        raise ProbeRecordError(f"{label} expected root is redirected by a symlink")
    resolved = Path(value).resolve()
    if not resolved.is_relative_to(expected_root):
        raise ProbeRecordError(f"{label} path escapes its expected root")
    if kind == "file" and not resolved.is_file():
        raise ProbeRecordError(f"{label} must be an existing file")
    if kind == "directory" and not resolved.is_dir():
        raise ProbeRecordError(f"{label} must be an existing directory")
    return resolved


@dataclass(frozen=True)
class BuildPaths:
    build: Path
    source: Path
    library: Path
    includes: tuple[Path, ...]
    library_dirs: tuple[Path, ...]


def _dependency_library_dir(value: str | Path) -> Path:
    resolved = Path(value).resolve()
    for root in (LIBRARY_ROOT, WHEEL_LIBRARY_ROOT):
        if resolved.is_relative_to(root.resolve()):
            return _bounded_path(value, root, "dependency library", "directory")
    raise ProbeRecordError("dependency library path escapes its expected root")


def _validated_build_paths(args: argparse.Namespace) -> BuildPaths:
    if args.tt_metal_source_revision != PINNED_SOURCE_REVISION:
        raise ProbeRecordError("source revision must be the pinned v0.75.0 revision")
    if not args.tt_metal_extra_include or not args.tt_metal_extra_lib_dir:
        raise ProbeRecordError("third-party include and dependency library roots are required")
    return BuildPaths(
        build=_bounded_path(args.build_dir, BUILD_ROOT, "build", "build"),
        source=_bounded_path(args.tt_metal_source_root, SOURCE_ROOT, "source", "directory"),
        library=_bounded_path(args.tt_metal_library, LIBRARY_ROOT, "library", "file"),
        includes=tuple(_bounded_path(path, INCLUDE_ROOT, "include", "directory") for path in args.tt_metal_extra_include),
        library_dirs=tuple(_dependency_library_dir(path) for path in args.tt_metal_extra_lib_dir),
    )


def _build_native_probe(args: argparse.Namespace) -> int:
    """Only validated rooted paths and a literal revision reach CMake."""
    paths = _validated_build_paths(args)
    command = [
        "cmake", "-S", str(Path(__file__).with_name("issue120").resolve()),
        "-B", str(paths.build),
        f"-DTT_METAL_SOURCE_ROOT={paths.source}",
        f"-DTT_METAL_SOURCE_REVISION={PINNED_SOURCE_REVISION}",
        f"-DTT_METAL_LIBRARY={paths.library}",
        f"-DTT_METAL_EXTRA_INCLUDE_DIRS={';'.join(map(str, paths.includes))}",
        f"-DTT_METAL_EXTRA_LIBRARY_DIRS={';'.join(map(str, paths.library_dirs))}",
    ]
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(paths.build), "--target", "issue120_probe", "--parallel", "2"], check=True)
    return 0


def _native_stall_group(value: str | None) -> str | None:
    if value is None:
        return None
    allowed = {"resident-and-transfer": "resident-and-transfer", "transfer-only": "transfer-only"}
    if not isinstance(value, str) or value not in allowed:
        raise ProbeRecordError("stall-group must be a fixed allowed enumeration")
    return allowed[value]


def _selected_stall_group(probe_id: str, requested: str | None) -> str | None:
    stall = _native_stall_group(requested)
    if probe_id == "q1-cq1-completion" and stall is None:
        raise ProbeRecordError("Q1 requires an explicit --stall-group variant")
    if probe_id == "q4-frame-coverage":
        if stall not in {None, "transfer-only"}:
            raise ProbeRecordError("Q4 frame coverage requires the fixed transfer-only stall group")
        return "transfer-only"
    return stall


def _native_paths(value: str | Path) -> tuple[Path, Path]:
    build = _bounded_path(value, BUILD_ROOT, "build", "directory")
    executable = _bounded_path(build / "issue120_probe", BUILD_ROOT, "executable", "file")
    library = _bounded_path(build / "libtt_metal.so", BUILD_ROOT, "library", "file")
    return executable, library


def _result_destination(value: str | Path | None) -> Path:
    if not value:
        raise ProbeRecordError("a new wrapper-owned result path is required before execution")
    destination = _bounded_path(value, OUTPUT_ROOT, "result", "output")
    if destination.exists():
        raise ProbeRecordError("a new wrapper-owned result path is required before execution")
    return destination


def _native_arguments(executable: Path, spec: ProbeSpec, stall: str | None, attempt: issue120_results.NativeAttempt) -> list[str]:
    canonical = _PROBE_BY_ID[spec.probe_id].probe_id
    command = [
        str(executable), "--execute", "--probe", canonical,
        "--repeat-count", str(int(spec.repeat_count)),
        "--cap-seconds", str(int(spec.outer_cap_seconds)),
        "--watcher", "1" if spec.watcher else "0",
        "--result-file", str(attempt.result_file), "--attempt-identity", attempt.identity,
    ]
    if stall is not None:
        command.extend(("--stall-group", stall))
    return command


def _runtime_environment() -> dict[str, str]:
    try:
        runtime_root = issue120_runtime.validate_runtime_assets()
    except issue120_runtime.RuntimeAssetError as exc:
        raise ProbeRecordError(str(exc)) from exc
    runtime_env = os.environ.copy()
    runtime_env["TT_METAL_RUNTIME_ROOT"] = str(runtime_root)
    runtime_env.pop("TT_METAL_HOME", None)
    paths = [
        "/opt/venv/lib/python3.12/site-packages/ttnn/build/lib",
        "/opt/venv/lib/python3.12/site-packages/ttnn.libs",
    ]
    if runtime_env.get("LD_LIBRARY_PATH"):
        paths.append(runtime_env["LD_LIBRARY_PATH"])
    runtime_env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    return runtime_env


def _execute_native(executable: Path, spec: ProbeSpec, stall: str | None, attempt: issue120_results.NativeAttempt | None = None) -> subprocess.CompletedProcess:
    if (os.environ.get("TT_METAL_WATCHER") == "1") != spec.watcher:
        raise ProbeRecordError("first-kernel Watcher mode does not match the environment")
    try:
        if attempt is None:
            attempt = issue120_results.create_attempt(os.environ.get("HEKATUS_TT_RUN_ID"))
        command = _native_arguments(executable, spec, stall, attempt)
        return issue120_results.execute(command, attempt, spec.outer_cap_seconds, _runtime_environment, REPOSITORY_ROOT)
    except issue120_results.NativeResultError as exc:
        raise ProbeRecordError(str(exc)) from exc


def _record_native_result(completed: subprocess.CompletedProcess, spec: ProbeSpec, environment: dict[str, str], library_hash: str, first: bool, attempt: issue120_results.NativeAttempt) -> dict[str, Any]:
    try:
        if completed.returncode != 0:
            raise issue120_results.NativeResultError("native exit does not permit a result")
        result = issue120_results.read_result(attempt)
    except issue120_results.NativeResultError as exc:
        raise ProbeRecordError(str(exc)) from exc
    result["native_capture"] = {
        "attempt_identity": attempt.identity,
        "directory": attempt.directory.relative_to(issue120_results.OUTPUT_ROOT).as_posix(),
        "execution_file": "execution.json", "stdout_file": "stdout.bin", "stderr_file": "stderr.bin",
        "result_file": "result/result.json", "native_returncode": completed.returncode,
    }
    result.update(schema=PROBE_SCHEMA, environment=environment)
    result["source"] = {
        "tt_metal_tag": "v0.75.0", "tt_metal_commit": PINNED_SOURCE_REVISION,
        "libtt_metal_sha256": library_hash,
        "source_tree_parity_with_image": "not_asserted; image library hash is pinned",
    }
    result["plan"] = _plan_record(spec, first_kernel_run=first)
    result["first_kernel_run"] = first
    result["timing_evidence"] = False
    if spec.probe_id == "q4-frame-coverage":
        result["coverage_analysis"] = evaluate_frame_coverage(result.get("frame_coverage", {}))
        result["coverage"] = result["coverage_analysis"]["coverage"]
        result["status"] = "complete" if result["coverage"] == "covered" else "not_evaluable"
        result["not_evaluable_reason"] = result["coverage_analysis"]["reason"]
    validate_probe_record(result)
    return result


def _run_native_probe(args: argparse.Namespace) -> int:
    if not args.execute:
        raise ProbeRecordError("board execution requires the explicit --execute gate")
    if os.environ.get("HEKATUS_TT_PINNED_CONTAINER") != "1":
        raise ProbeRecordError("native execution is supported only in the pinned container")
    stall = _selected_stall_group(args.probe, getattr(args, "stall_group", None))
    spec = planned_probe(args.probe, repeat_count=args.repeat_count,
                         outer_cap_seconds=args.cap_seconds, first_kernel_run=args.first_kernel_run)
    environment = _environment_record(os.environ.get("HEKATUS_TT_ENV_JSON", ""))
    destination = _result_destination(os.environ.get("HEKATUS_TT_RESULT_PATH"))
    executable, library = _native_paths(args.build_dir)
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    if library_hash != PINNED_LIBRARY_HASH:
        raise ProbeRecordError("staged libtt_metal.so does not match the audited pinned image library")
    try:
        attempt = issue120_results.create_attempt(os.environ.get("HEKATUS_TT_RUN_ID"))
    except issue120_results.NativeResultError as exc:
        raise ProbeRecordError(str(exc)) from exc
    completed = _execute_native(executable, spec, stall, attempt)
    if completed.returncode != 0:
        print(f"native exit {completed.returncode}; raw diagnostics retained", file=sys.stderr)
        return completed.returncode
    result = _record_native_result(completed, spec, environment, library_hash, args.first_kernel_run, attempt)
    with destination.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="print board-free probe inventory")
    parser.add_argument("--probe", choices=tuple(_PROBE_BY_ID))
    parser.add_argument("--execute", action="store_true", help="explicitly permit board access")
    parser.add_argument("--build-only", action="store_true", help="board-free staged native build")
    parser.add_argument("--tt-metal-source-root")
    parser.add_argument("--tt-metal-source-revision")
    parser.add_argument("--tt-metal-library")
    parser.add_argument("--tt-metal-extra-include", action="append", default=[])
    parser.add_argument("--tt-metal-extra-lib-dir", action="append", default=[])
    parser.add_argument("--first-kernel-run", action="store_true")
    parser.add_argument("--stall-group", choices=("resident-and-transfer", "transfer-only"))
    parser.add_argument("--repeat-count", type=int)
    parser.add_argument("--cap-seconds", type=int, default=DEFAULT_PROBE_CAP_SECONDS)
    parser.add_argument("--build-dir", default="out/issue120-native-build")
    args = parser.parse_args(argv)
    try:
        if args.plan:
            print(json.dumps(_plan_json(), indent=2, sort_keys=True))
            return 0
        if args.build_only:
            return _build_native_probe(args)
        if args.probe is None:
            parser.error("select --plan, --build-only, or --probe")
        return _run_native_probe(args)
    except (ProbeRecordError, OSError, subprocess.CalledProcessError) as exc:
        print(f"issue120 probe rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
