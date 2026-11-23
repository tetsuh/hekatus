"""Fixed production proposals, not authority to execute them."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

FRAME_COUNT = 500_000
PERIOD = 1_350_000
PAGE_BYTES = 4096
CPU_OPERATIONS_PER_WRITE = 32 * 1024
CANDIDATE_REPETITIONS = 24_439_669
SCHEMA = "issue120-production-plan-v1"


@dataclass(frozen=True)
class RunSpec:
    identity: str
    phase: str
    condition: str
    frame_count: int
    repeat_count: int
    cap_seconds: int
    watcher: bool

    @property
    def run_budget_ticks(self) -> int:
        # Same resident run-wide planning components, with a 10% safety margin.
        return (self.frame_count * (PERIOD + 100_000) * 110 + 99) // 100 + 135_000_000


RUNS = (
    RunSpec("production-v2-control-watch-b4", "startup", "control", 4096, 262144, 60, True),
    RunSpec("production-v2-load-watch-b4", "startup", "loaded", 4096, 262144, 60, True),
    RunSpec("production-v1-calibration-small", "calibration", "loaded", 16384, 1048576, 60, False),
    RunSpec("production-v1-calibration-large", "calibration", "loaded", 32768, 1572864, 60, False),
    RunSpec("production-v1-control-500000", "production", "control", FRAME_COUNT,
            CANDIDATE_REPETITIONS, 600, False),
    RunSpec("production-v1-loaded-500000", "production", "loaded", FRAME_COUNT,
            CANDIDATE_REPETITIONS, 600, False),
)


def run_spec(identity: str) -> RunSpec:
    for spec in RUNS:
        if spec.identity == identity:
            return spec
    raise ValueError("unsupported precommitted run identity")


def memory_bounds(*, frame_count: int, repeat_count: int) -> dict[str, int]:
    if type(frame_count) is not int or not 0 < frame_count <= FRAME_COUNT:
        raise ValueError("unsupported frame capacity")
    if type(repeat_count) is not int or not 0 < repeat_count <= 2**32 - frame_count - 3:
        raise ValueError("unsupported repeat count")
    return {"resident_timestamp_bytes": frame_count * PAGE_BYTES,
            "host_payload_bytes": PAGE_BYTES, "retrieval_chunk_bytes": 256 * PAGE_BYTES,
            "transition_capacity": frame_count + 3,
            "transition_bytes": (frame_count + 3) * 8,
            "frame_evidence_bytes": frame_count * 128,
            "ring_bytes": 4 * PAGE_BYTES,
            "fixed_auxiliary_buffer_bytes": 6 * PAGE_BYTES,
            "device_cb_scratch_bytes": 3 * PAGE_BYTES}


def program_configuration(spec: RunSpec) -> dict:
    return {"schema": "issue120-production-program-v1", "protocol": "issue120-wide-progress-v1",
            "frame_count": spec.frame_count, "repeat_count": spec.repeat_count,
            "period_ticks": PERIOD, "work_per_frame": 64, "ring_pages": 4,
            "cycle_budget_ticks": 100000, "run_budget_ticks": spec.run_budget_ticks,
            "device": 0, "command_queues": 2, "resident_subdevice": 0,
            "transfer_subdevice": 1, "resident_cores": [[0, 0], [1, 0]],
            "transfer_core": [2, 0], "stall_group": [1], "watcher": spec.watcher,
            "outer_cap_seconds": spec.cap_seconds, "sampler_interval_seconds": 2,
            "tables_enabled": False, "memory": memory_bounds(
                frame_count=spec.frame_count, repeat_count=spec.repeat_count)}


def configuration_digest(configuration: dict) -> str:
    return hashlib.sha256(json.dumps(configuration, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def work_estimate() -> dict:
    fast = 262144 / 6.017380392
    slow = 262144 / 6.040565943
    n = math.ceil(550 * fast * 1.02)
    return {"source": "issue120-run8-fixed-work-host-intervals", "observed_fast_writes_per_s": fast,
            "observed_slow_writes_per_s": slow, "planning_fraction": 0.02,
            "fast_target_seconds": 550, "candidate_repetitions": n,
            "operations": n * CPU_OPERATIONS_PER_WRITE, "submitted_bytes": n * PAGE_BYTES,
            "slow_planning_seconds": n / (slow * 0.98),
            "cap_headroom_seconds": 600 - n / (slow * 0.98),
            "overhead_reserve_seconds": 20, "calibrated": False,
            "production_admitted": False}


def proposal() -> dict:
    return {"schema": SCHEMA, "board_access": False, "execution_authorized": False,
            "work_estimate": work_estimate(), "runs": [asdict(s) for s in RUNS],
            "approval_units": [[s.identity for s in RUNS[:2]],
                               [s.identity for s in RUNS[2:4]],
                               [s.identity for s in RUNS[4:]]],
            "remaining_gates": ["bounded new-kernel validation", "exact-work calibration",
                                "fixed-count production precommit", "separate owner approval"]}
