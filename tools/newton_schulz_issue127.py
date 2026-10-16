"""Run the Issue #127 two-run ABBA/BADC R-format comparison.

This runner is deliberately separate from the Issue #88 catalogue runner.  It
keeps the four approved placements, fixed occurrence order, long timed launch
counts, and the run-2 cooldown contract in one board-side entry point.  The
Newton-Schulz kernel is imported from the existing implementation and is not
changed here.  Board-free callers can use the planning, provenance, and
interval-fact helpers without importing ``ttnn``.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.newton_schulz_reference import (
    NEWTON_SCHULZ_ITERATIONS,
    random_hpd_batch,
)
from enodia.tt.bench.power_trace_snapshot import (
    EXPLICIT_FINAL_SAMPLE,
    PowerTraceSnapshotOptions,
    atomic_power_trace_snapshot,
    left_held_aiclk_interval_facts,
    read_power_trace_snapshot,
)
from enodia.tt.bench.run_binding import (
    current_run_id,
    require_run_artifact,
    run_artifact_name,
    validate_run_id,
)
from enodia.tt.bench.sampler_contract import (
    DEFAULT_SAMPLER_INTERVAL_SECONDS,
    normalize_sampler_metadata,
)
from enodia.tt.bench.telemetry import _parse_trace_timestamp
from tools import newton_schulz_issue88 as issue88

ISSUE127_RUNNER = "tools/newton_schulz_issue127.py"
ISSUE127_RECORD_SCHEMA = "issue127-abba-v2"
ISSUE127_RAW_SCHEMA = "issue127-abba-raw-v2"
INITIAL_VALUE = "I/||R||inf"
JSON_SUFFIX = ".json"
REDACTED_PATH = "<redacted-path>"
DEVICE_NODE_KEY = "device_node"
DEVICE_ID = 0
BATCH = 8192
CONDITION_NUMBER = 100.0
INPUT_SEED = 6300
FIXED_ITERATIONS = NEWTON_SCHULZ_ITERATIONS
NOMINAL_AICLK_MHZ = 1350.0
STARTUP_HOLD_AICLK_MHZ = 800.0
WARMUP_LAUNCHES = 4500
DEFAULT_RUN_TIMEOUT_SECONDS = 60.0
RUN_TIMEOUT_SECONDS = DEFAULT_RUN_TIMEOUT_SECONDS
APPROVED_RUN_TIMEOUT_SECONDS = 90.0
APPROVED_RUN_TIMEOUT_DATE = "2026-10-16"
APPROVED_RUN_TARGETS = {
    "confirmation": 1,
    "measurement-1": 1,
    "measurement-2": 2,
    "measurement-2-rerun": 2,
}

# The run-2 cooldown is a thermal/power readiness gate.  These values come
# from the first measurement-1 trace sample at or before its run start; the
# idle clock is retained as provenance and is deliberately not part of the
# gate.
COOLDOWN_BASELINE_SOURCE = "measurement-1 pre-run power trace"
COOLDOWN_BASELINE_POWER_W = 47.0
COOLDOWN_BASELINE_ASIC_TEMPERATURE_C = 58.5
COOLDOWN_POWER_TOLERANCE_W = 5.0
COOLDOWN_TEMPERATURE_TOLERANCE_C = 3.0
COOLDOWN_POWER_LIMIT_W = COOLDOWN_BASELINE_POWER_W + COOLDOWN_POWER_TOLERANCE_W
COOLDOWN_TEMPERATURE_LIMIT_C = (
    COOLDOWN_BASELINE_ASIC_TEMPERATURE_C + COOLDOWN_TEMPERATURE_TOLERANCE_C
)
COOLDOWN_OBSERVED_IDLE_AICLK_MHZ = 800.0
COOLDOWN_WAIT_CAP_SECONDS = 600.0
POWER_TRACE_POLL_TIMEOUT_SECONDS = 6.0
POWER_TRACE_POLL_INTERVAL_SECONDS = 0.25
POWER_TRACE_SNAPSHOT_PREFIX = "issue127-abba-power-snapshot"
SAMPLER_INTERVAL_SECONDS = DEFAULT_SAMPLER_INTERVAL_SECONDS
SAMPLER_OBSERVED_GAP_EXPECTED_SECONDS = 2.24
ROW_DURATION_MIN_SECONDS = SAMPLER_OBSERVED_GAP_EXPECTED_SECONDS
POWER_SAMPLES_PER_ROW_TARGET = 2
RUN_BUDGET_INITIAL_OCCURRENCE_ESTIMATE_SECONDS = 6.0
RUN_BUDGET_SETUP_MARGIN_SECONDS = 1.25

# These names are intentionally the only comparison configurations in this
# runner.  In particular, the rejected FP32-R/L32/R-in-L1 Issue #88 row is not
# a hidden sixth configuration here.
def comparison_rows() -> tuple[dict[str, Any], ...]:
    """Return the four approved Issue #127 configurations in canonical order."""
    common = {
        "batch": BATCH,
        "iterations": FIXED_ITERATIONS,
        "initial_value": INITIAL_VALUE,
        "state_format": "BF16",
        "destination_format": "FP32",
        "fp32_dest_acc_en": True,
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "input_memory": "l1",
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "output_memory": "dram",
        "x0_memory": "l1",
    }
    return tuple(
        {
            **common,
            "name": name,
            "variant": variant,
            "r_format": r_format,
            "size": size,
            "packing": packing,
            "r_memory": r_memory,
        }
        for name, variant, r_format, size, packing, r_memory in (
            ("bf16-r-L16", "bf16", "BF16", 16, "diagonal_pairs_32x32", "l1"),
            ("fp32-r-L16", "fp32-r", "FP32", 16, "diagonal_pairs_32x32", "l1"),
            (
                "bf16-r-L32-r-dram",
                "bf16",
                "BF16",
                32,
                "native_32x32",
                "dram",
            ),
            (
                "fp32-r-L32-r-dram",
                "fp32-r",
                "FP32",
                32,
                "native_32x32",
                "dram",
            ),
        )
    )


ISSUE127_COMPARISON_ROWS = comparison_rows()
ISSUE127_ROW_NAMES = tuple(row["name"] for row in ISSUE127_COMPARISON_ROWS)
ROW_BY_NAME = {row["name"]: row for row in ISSUE127_COMPARISON_ROWS}
LAUNCHES_BY_SIZE = {16: 11000, 32: 5800}
RUN_1_ORDER = (
    "bf16-r-L16",
    "fp32-r-L16",
    "fp32-r-L16",
    "bf16-r-L16",
    "bf16-r-L32-r-dram",
    "fp32-r-L32-r-dram",
    "fp32-r-L32-r-dram",
    "bf16-r-L32-r-dram",
)
RUN_2_ORDER = (
    "fp32-r-L16",
    "bf16-r-L16",
    "bf16-r-L16",
    "fp32-r-L16",
    "fp32-r-L32-r-dram",
    "bf16-r-L32-r-dram",
    "bf16-r-L32-r-dram",
    "fp32-r-L32-r-dram",
)
RUN_ORDERS = {1: RUN_1_ORDER, 2: RUN_2_ORDER}

# These are pre-board AICLK predictions, not measurements.  They are assigned
# by occurrence position so the order-swap run changes row placement without
# changing the predicted clock bands.  The bands interpolate the #121 evidence
# points of 2 seconds / 1243 MHz and 35 seconds / 1068 MHz.
PREDICTED_AICLK_RANGES_MHZ = {
    1: (
        (1200.0, 1243.0),
        (1180.0, 1230.0),
        (1160.0, 1210.0),
        (1140.0, 1190.0),
        (1120.0, 1170.0),
        (1100.0, 1150.0),
        (1080.0, 1130.0),
        (1068.0, 1115.0),
    ),
    2: (
        (1200.0, 1243.0),
        (1180.0, 1230.0),
        (1160.0, 1210.0),
        (1140.0, 1190.0),
        (1120.0, 1170.0),
        (1100.0, 1150.0),
        (1080.0, 1130.0),
        (1068.0, 1115.0),
    ),
}

# The prediction document and the runner share one named sampler decision.
SAMPLER_DECISION = {
    "mode": "default",
    "interval_seconds": SAMPLER_INTERVAL_SECONDS,
    "requested_interval_seconds": SAMPLER_INTERVAL_SECONDS,
    "default_source": "enodia/tt/bench/run_in_container.sh",
    "sampler_contract_source": "enodia/tt/bench/sampler_contract.py",
    "snapshot_command_source": "enodia/tt/bench/telemetry.py",
    "snapshot_command": "tt-smi -s --snapshot_no_tty",
    "sample_before_sleep": True,
    "shortened_interval": False,
    "observed_gap_expected_seconds": SAMPLER_OBSERVED_GAP_EXPECTED_SECONDS,
    "selection_reason": (
        "Issue #104 committed evidence covers only default 2.0 seconds and explicit "
        "5.0 seconds; no evidence shows that 0.5 seconds or less does not disturb "
        "the measurement, so the current default is retained."
    ),
}

# The deltas are percentages: 100 * (FP32-R P50 / BF16-R P50 - 1).  The values
# are fixed before the board run in the prediction plan and are not inferred
# from adjusted P50 values.
R_FORMAT_DELTA_THRESHOLDS_PERCENT = {"L16": -0.847, "L32_DRAM": -2.341}
CLOCK_DELTA_HALF_THRESHOLDS_PERCENT = {"L16": 0.4235, "L32_DRAM": 1.1705}
POSITION_EFFECT_THRESHOLDS_PERCENT = dict(CLOCK_DELTA_HALF_THRESHOLDS_PERCENT)

_SAFE_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)
_BOARD_ID_RE = re.compile(r"^(?:[0-9]{8,24}|board-[A-Za-z0-9_-]{1,48})$")
_PCI_BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", re.IGNORECASE)
_DEVICE_NODE_RE = re.compile(r"^/dev/tenstorrent/(?:[0-9]+|by-id/[A-Za-z0-9_.:-]+)$")
# Path candidates deliberately allow filename punctuation and spaces.  The
# first component must not begin with whitespace or the double-pipe marker used
# by quotient prose, while a single ``|`` remains a valid filename character.
# All consumers use this one candidate grammar; formula and device-node
# exceptions are applied at the traversal sites below.
_PATH_COMPONENT = r'''[^/\\\x00-\x1f"']+'''
_PATH_FIRST_COMPONENT = r'''(?!(?:\s|\|\|))(?=[^/\\\x00-\x1f"'])''' + _PATH_COMPONENT
_PATH_SEPARATOR = r'''[\\/]'''
_PATH_BODY = rf"{_PATH_FIRST_COMPONENT}(?:{_PATH_SEPARATOR}+{_PATH_COMPONENT})*{_PATH_SEPARATOR}*"
_PATH_WRAPPER = rf"(?:file|path):{_PATH_SEPARATOR}+{_PATH_BODY}"
_PATH_POSIX = rf"/{_PATH_BODY}"
_PATH_HOME = rf"~{_PATH_SEPARATOR}+{_PATH_BODY}"
_PATH_DRIVE = rf"[A-Za-z]:{_PATH_SEPARATOR}+{_PATH_BODY}"
_PATH_UNC = (
    rf"{_PATH_SEPARATOR}{_PATH_SEPARATOR}{_PATH_FIRST_COMPONENT}"
    rf"(?:{_PATH_SEPARATOR}+{_PATH_COMPONENT})+{_PATH_SEPARATOR}*"
)
_PATH_CANDIDATE = (
    rf"(?:{_PATH_WRAPPER}|{_PATH_HOME}|{_PATH_DRIVE}|{_PATH_UNC}|{_PATH_POSIX})"
)
_PATH_CANDIDATE_RE = re.compile(
    rf"(?:(?<![A-Za-z0-9_]){_PATH_WRAPPER}|"
    rf"(?<![A-Za-z0-9_/:])(?:{_PATH_HOME}|{_PATH_DRIVE}|{_PATH_UNC}|{_PATH_POSIX}))",
    re.IGNORECASE,
)
# Keep these private compatibility regexes for callers that used the old
# names; path decisions themselves go through ``_PATH_CANDIDATE_RE``.
_ABSOLUTE_PATH_RE = re.compile(rf"^{_PATH_CANDIDATE}$", re.IGNORECASE)
_EMBEDDED_HOST_PATH_RE = _PATH_CANDIDATE_RE
_FORMULA_SLASH_RE = re.compile(
    r"/[A-Za-z_][A-Za-z0-9_]*\([^/\\]*\)(?=$|[\s,;:}\]])"
)
_FORMULA_SLASH_FRAGMENT_RE = re.compile(
    r"/[A-Za-z_][A-Za-z0-9_]*\((?=$|[\s,;:}\]])"
)
_RECORD_PATH_FIELD_NAMES = frozenset(
    {
        "artifact_path",
        "environment_path",
        "file",
        "path",
        "power_trace",
        "raw_path",
        "result_file",
        "result_path",
        "source_file",
        "source_path",
        "trace",
    }
)
_FORBIDDEN_RECORD_KEYS = frozenset(
    {"hostname", "host_name", "worktree_path", "repository_path"}
)
_ENVIRONMENT_PATH_KEYS = frozenset(
    {"path", "pythonpath", "ld_library_path", "library_path", "library_paths"}
)
_FORBIDDEN_ENV_KEY_PARTS = (
    "hostname",
    "host_name",
    "worktree",
    "cwd",
    "absolute_path",
    "repo_path",
    "repository_path",
    "working_directory",
)


def validate_comparison_rows(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Require exactly the four approved controls and no placement fallback."""
    selected = ISSUE127_COMPARISON_ROWS if rows is None else tuple(rows)
    if tuple(row.get("name") for row in selected) != ISSUE127_ROW_NAMES:
        raise ValueError("Issue #127 comparison rows must be the exact four canonical rows")
    common = {
        "batch": BATCH,
        "iterations": FIXED_ITERATIONS,
        "initial_value": INITIAL_VALUE,
        "state_format": "BF16",
        "destination_format": "FP32",
        "fp32_dest_acc_en": True,
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "input_memory": "l1",
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "output_memory": "dram",
        "x0_memory": "l1",
    }
    for row, canonical in zip(selected, ISSUE127_COMPARISON_ROWS):
        if dict(row) != dict(canonical):
            raise ValueError(f"{canonical['name']} changes an approved control or placement")
        if any(row.get(key) != value for key, value in common.items()):
            raise ValueError(f"{row['name']} changes a required execution control")


def launches_for_row(row: Mapping[str, Any] | str) -> int:
    """Return the fixed timed launch count for one configuration."""
    selected = ROW_BY_NAME[row] if isinstance(row, str) else row
    try:
        launches = LAUNCHES_BY_SIZE[int(selected["size"])]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Issue #127 row has an unsupported matrix size") from exc
    return launches


launch_count_for_row = launches_for_row


def order_for_run(run_number: int) -> tuple[str, ...]:
    """Return the immutable occurrence order for run 1 or run 2."""
    try:
        return RUN_ORDERS[int(run_number)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run_number must be 1 or 2") from exc


def build_occurrence_plan(run_number: int) -> tuple[dict[str, Any], ...]:
    """Expand one fixed order into occurrence- and index-addressable rows."""
    order = order_for_run(run_number)
    counts: collections.Counter[str] = collections.Counter()
    planned: list[dict[str, Any]] = []
    for order_index, name in enumerate(order):
        if name not in ROW_BY_NAME:
            raise ValueError(f"unknown Issue #127 row in order: {name!r}")
        counts[name] += 1
        row = dict(ROW_BY_NAME[name])
        occurrence = counts[name]
        row.update(
            {
                "row": name,
                "occurrence": occurrence,
                "order_index": order_index,
                "occurrence_id": f"{name}#{occurrence}",
                "launches_requested": launches_for_row(name),
                "predicted_aiclk_range_mhz": list(
                    PREDICTED_AICLK_RANGES_MHZ[int(run_number)][order_index]
                ),
            }
        )
        planned.append(row)
    validate_occurrence_plan(planned, run_number=run_number)
    return tuple(planned)


def _validate_occurrence_prefix(
    planned: Sequence[Mapping[str, Any]], *, run_number: int
) -> None:
    """Validate a failed run's completed prefix without accepting reordering."""
    expected = order_for_run(run_number)
    if len(planned) > len(expected):
        raise ValueError("Issue #127 occurrence prefix is longer than the fixed order")
    names = tuple(row.get("row", row.get("name")) for row in planned)
    if names != expected[: len(names)]:
        raise ValueError("Issue #127 occurrence prefix is not the approved fixed order")
    counts: collections.Counter[str] = collections.Counter()
    for index, row in enumerate(planned):
        name = names[index]
        counts[name] += 1
        if row.get("order_index") != index:
            raise ValueError("Issue #127 order_index does not identify the occurrence")
        if row.get("occurrence") != counts[name]:
            raise ValueError("Issue #127 occurrence numbering is not monotonic")
        if row.get("occurrence_id") != f"{name}#{counts[name]}":
            raise ValueError("Issue #127 occurrence_id is not deterministic")
        if row.get("launches_requested") != launches_for_row(name):
            raise ValueError("Issue #127 occurrence launch count is not fixed")
        expected_range = list(PREDICTED_AICLK_RANGES_MHZ[int(run_number)][index])
        if row.get("predicted_aiclk_range_mhz") != expected_range:
            raise ValueError("Issue #127 occurrence AICLK prediction is not fixed")


def validate_occurrence_plan(
    planned: Sequence[Mapping[str, Any]], *, run_number: int
) -> None:
    """Validate fixed order, two occurrences per configuration, and indices."""
    expected = order_for_run(run_number)
    if len(planned) != len(expected):
        raise ValueError("Issue #127 requires exactly eight measured occurrences per run")
    _validate_occurrence_prefix(planned, run_number=run_number)
    counts = collections.Counter(row.get("row", row.get("name")) for row in planned)
    if counts != collections.Counter({name: 2 for name in ISSUE127_ROW_NAMES}):
        raise ValueError("Issue #127 requires two occurrences of every configuration")


def resolve_run_timeout(
    run_number: int,
    requested_seconds: float | None = None,
    run_scope: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve the default budget or one explicit owner-approved exception."""
    if requested_seconds is None and run_scope is None:
        return DEFAULT_RUN_TIMEOUT_SECONDS, {
            "default_run_timeout_seconds": DEFAULT_RUN_TIMEOUT_SECONDS,
            "effective_run_timeout_seconds": DEFAULT_RUN_TIMEOUT_SECONDS,
            "approved_override": False,
            "approval_date": None,
            "scope": None,
            "run_number": run_number,
        }
    if requested_seconds is None:
        raise ValueError(
            "Issue #127 approved timeout requires --run-scope and "
            f"--run-timeout-seconds {int(APPROVED_RUN_TIMEOUT_SECONDS)}"
        )
    try:
        requested = float(requested_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Issue #127 run timeout must be numeric") from exc
    if not math.isfinite(requested):
        raise ValueError("Issue #127 run timeout must be finite")
    if _same_float(requested, DEFAULT_RUN_TIMEOUT_SECONDS) and run_scope is None:
        return DEFAULT_RUN_TIMEOUT_SECONDS, {
            "default_run_timeout_seconds": DEFAULT_RUN_TIMEOUT_SECONDS,
            "effective_run_timeout_seconds": DEFAULT_RUN_TIMEOUT_SECONDS,
            "approved_override": False,
            "approval_date": None,
            "scope": None,
            "run_number": run_number,
        }
    if run_scope is None:
        raise ValueError(
            "Issue #127 approved timeout requires --run-scope and "
            f"--run-timeout-seconds {int(APPROVED_RUN_TIMEOUT_SECONDS)}"
        )
    if not _same_float(requested, APPROVED_RUN_TIMEOUT_SECONDS):
        raise ValueError(
            "Issue #127 permits only the default 60-second timeout or the "
            "owner-approved 90-second timeout"
        )
    if run_scope not in APPROVED_RUN_TARGETS:
        raise ValueError("Issue #127 run scope is not an approved target")
    expected_run_number = APPROVED_RUN_TARGETS[run_scope]
    if int(run_number) != expected_run_number:
        raise ValueError(
            f"Issue #127 run scope {run_scope!r} requires run_number="
            f"{expected_run_number}"
        )
    return APPROVED_RUN_TIMEOUT_SECONDS, {
        "default_run_timeout_seconds": DEFAULT_RUN_TIMEOUT_SECONDS,
        "effective_run_timeout_seconds": APPROVED_RUN_TIMEOUT_SECONDS,
        "approved_override": True,
        "approval_date": APPROVED_RUN_TIMEOUT_DATE,
        "scope": run_scope,
        "run_number": run_number,
        "approved_targets": dict(APPROVED_RUN_TARGETS),
    }


def row_duration_guard(
    duration_seconds: float,
    row: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Require a row to span more than one expected sampler interval.

    Row duration is deliberately not capped: lower AICLK can make a valid row
    longer. The run-level budget is the only upper bound.
    """
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("timed row duration must be finite and positive")
    size = int((ROW_BY_NAME[row] if isinstance(row, str) else row)["size"])
    above_sampler_gap = duration > ROW_DURATION_MIN_SECONDS
    return {
        "target_seconds_min": ROW_DURATION_MIN_SECONDS,
        "target_seconds_max": None,
        "sampler_gap_seconds": SAMPLER_OBSERVED_GAP_EXPECTED_SECONDS,
        "measured_seconds": duration,
        "launches_requested": launches_for_row(row),
        "size": size,
        "above_sampler_gap": above_sampler_gap,
        "within_target": above_sampler_gap,
        "passes": above_sampler_gap,
    }


def project_run_budget(
    elapsed_seconds: float,
    completed_rows: Sequence[Mapping[str, Any]],
    remaining_occurrences: int,
    *,
    run_timeout_seconds: float = RUN_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Project completion before starting the next fixed-work occurrence."""
    elapsed = float(elapsed_seconds)
    remaining = int(remaining_occurrences)
    timeout = float(run_timeout_seconds)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("run budget elapsed time must be finite and non-negative")
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("run budget timeout must be finite and positive")
    if remaining < 0:
        raise ValueError("run budget remaining occurrence count must be non-negative")
    durations = []
    for row in completed_rows:
        value = row.get("timed_duration_seconds")
        if value is None:
            continue
        duration = float(value)
        if math.isfinite(duration) and duration > 0.0:
            durations.append(duration)
    observed_max = max(durations, default=0.0)
    estimated_occurrence = max(
        RUN_BUDGET_INITIAL_OCCURRENCE_ESTIMATE_SECONDS,
        observed_max + RUN_BUDGET_SETUP_MARGIN_SECONDS,
    )
    projected_total = elapsed + remaining * estimated_occurrence
    return {
        "elapsed_seconds": elapsed,
        "completed_occurrences": len(completed_rows),
        "remaining_occurrences": remaining,
        "observed_timed_duration_max_seconds": observed_max or None,
        "estimated_seconds_per_remaining_occurrence": estimated_occurrence,
        "projected_total_seconds": projected_total,
        "run_timeout_seconds": timeout,
        "within_budget": projected_total <= timeout,
        "estimation_policy": (
            "max(initial fixed estimate, observed maximum timed row plus setup margin); "
            "launch counts remain fixed per occurrence"
        ),
    }


def _same_float(left: Any, right: Any) -> bool:
    """Compare numeric values without relying on exact floating-point equality."""
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError, OverflowError):
        return False


def adjusted_p50_tflops(raw_p50_tflops: float, row_time_weighted_aiclk_mhz: float | None) -> float | None:
    """Compute the supplementary 1350-MHz-linear adjusted P50 value."""
    try:
        raw = float(raw_p50_tflops)
        aiclk = float(row_time_weighted_aiclk_mhz)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in (raw, aiclk)):
        return None
    return raw * NOMINAL_AICLK_MHZ / aiclk


def _timestamp(value: datetime.datetime | str, *, field: str) -> str:
    return _parse_trace_timestamp(value, field=field).isoformat()


def validate_row_bounds(row: Mapping[str, Any]) -> None:
    """Require UTC and monotonic timed bounds with a positive interval."""
    start = _parse_trace_timestamp(row.get("timed_start_utc"), field="timed_start_utc")
    end = _parse_trace_timestamp(row.get("timed_end_utc"), field="timed_end_utc")
    if end <= start:
        raise ValueError("timed UTC bounds must be strictly increasing")
    try:
        mono_start = float(row["timed_start_monotonic"])
        mono_end = float(row["timed_end_monotonic"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timed monotonic bounds are required") from exc
    if not all(math.isfinite(value) for value in (mono_start, mono_end)) or mono_end <= mono_start:
        raise ValueError("timed monotonic bounds must be finite and increasing")
    duration = float(row.get("timed_duration_seconds", mono_end - mono_start))
    if not math.isclose(duration, mono_end - mono_start, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("timed duration must match monotonic row bounds")
    row_duration_guard(duration, row)


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("environment contains a non-finite value")
        return value
    if isinstance(value, np.generic):
        return _json_safe_scalar(value.item())
    return None


def _environment_key_allowed(key: str) -> bool:
    lowered = key.lower()
    return not any(part in lowered for part in _FORBIDDEN_ENV_KEY_PARTS) and not _contains_embedded_host_path(
        key
    )


def _sanitize_environment_device_node(value: Any) -> str:
    if isinstance(value, str) and _DEVICE_NODE_RE.fullmatch(value):
        return value
    return REDACTED_PATH


def _sanitize_environment_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for child_key, child_value in value.items():
        if not isinstance(child_key, str) or not _environment_key_allowed(child_key):
            continue
        if child_key == DEVICE_NODE_KEY:
            result[child_key] = _sanitize_environment_device_node(child_value)
            continue
        sanitized = _sanitize_environment_value(child_value, key=child_key)
        if sanitized is not None:
            result[child_key] = sanitized
    return result


def _sanitize_environment_sequence(
    value: list[Any] | tuple[Any, ...], *, key: str | None
) -> list[Any]:
    sanitized_values = (_sanitize_environment_value(item, key=key) for item in value)
    return [item for item in sanitized_values if item is not None]


def _sanitize_environment_scalar(value: Any, *, key: str | None) -> Any:
    scalar = _json_safe_scalar(value)
    if scalar is None:
        return None
    if not isinstance(scalar, str):
        return scalar
    if key == DEVICE_NODE_KEY and _DEVICE_NODE_RE.fullmatch(scalar):
        return scalar
    if _environment_value_contains_path(scalar, key=key):
        return REDACTED_PATH
    if _is_whole_absolute_path(scalar):
        return None
    return scalar


def _sanitize_environment_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_environment_mapping(value)
    if isinstance(value, (list, tuple)):
        return _sanitize_environment_sequence(value, key=key)
    return _sanitize_environment_scalar(value, key=key)


def sanitize_environment(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project wrapper metadata onto complete, path-safe harness provenance."""
    if not isinstance(raw, Mapping):
        raise TypeError("environment artifact must be an object")
    environment = _sanitize_environment_value(dict(raw))
    if not isinstance(environment, dict):
        raise TypeError("environment artifact did not normalize to an object")

    board = environment.get("board") or environment.get("board_info")
    if isinstance(board, dict):
        board = dict(board)
        if not board.get("serial") and board.get("board_id"):
            board["serial"] = board["board_id"]
        environment["board"] = board
    firmware = environment.get("firmware") or environment.get("firmwares")
    if isinstance(firmware, dict):
        environment["firmware"] = dict(firmware)
    if "kmd_version" in environment and "kernel_driver_version" not in environment:
        environment["kernel_driver_version"] = environment["kmd_version"]
    image = environment.get("image")
    if isinstance(image, str) and "@" in image and "image_digest" not in environment:
        environment["image_digest"] = image.rsplit("@", 1)[1]
    sampler = environment.get("telemetry_sampler")
    if sampler is not None:
        environment["telemetry_sampler"] = normalize_sampler_metadata(
            sampler, complete_default=False
        )
    return environment


def _validate_environment_run_id(
    environment: Mapping[str, Any], run_id: str | None
) -> None:
    if run_id is not None and environment.get("run_id") != run_id:
        raise ValueError("environment run_id does not match the current run")


def _validate_environment_board(environment: Mapping[str, Any]) -> None:
    board = environment.get("board")
    if not isinstance(board, Mapping):
        raise TypeError("board serial environment is missing")
    if str(board.get("board_type", "")).lower() != "p150a":
        raise ValueError("board_type must be p150a")
    serial = board.get("serial") or board.get("board_id")
    board_id = board.get("board_id") or serial
    if not isinstance(serial, str) or _BOARD_ID_RE.fullmatch(serial) is None:
        raise ValueError("board serial is missing or unsafe")
    if not isinstance(board_id, str) or _BOARD_ID_RE.fullmatch(board_id) is None:
        raise ValueError("board_id is missing or unsafe")


def _validate_environment_toolchain(environment: Mapping[str, Any]) -> None:
    firmware = environment.get("firmware")
    if not isinstance(firmware, Mapping) or not str(firmware.get("fw_bundle_version", "")):
        raise ValueError("firmware bundle version is missing")
    driver = environment.get("kernel_driver_version") or environment.get("kmd_version")
    if not isinstance(driver, str) or not driver.strip():
        raise ValueError("kernel driver version is missing")
    image_digest = environment.get("image_digest")
    if not isinstance(image_digest, str) or _IMAGE_DIGEST_RE.fullmatch(image_digest) is None:
        raise ValueError("image digest is missing or unsafe")
    if environment.get("image_pinned") is not True:
        raise ValueError("image must be digest-pinned")
    harness_commit = environment.get("harness_commit")
    if not isinstance(harness_commit, str) or _COMMIT_RE.fullmatch(harness_commit) is None:
        raise ValueError("harness commit is missing or not a full commit")
    if environment.get("harness_dirty") is not False:
        raise ValueError("harness must be clean")


def _validate_environment_sampler(environment: Mapping[str, Any]) -> None:
    sampler = normalize_sampler_metadata(
        environment.get("telemetry_sampler"), complete_default=False
    )
    if sampler["mode"] != "default" or not _same_float(
        sampler["interval_seconds"], SAMPLER_INTERVAL_SECONDS
    ):
        raise ValueError("Issue #127 requires the current default 2.0-second sampler")


def validate_harness_environment(
    environment: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> None:
    """Fail closed unless the wrapper supplied the complete harness identity."""
    if not isinstance(environment, Mapping):
        raise TypeError("normalized harness environment is missing")
    _validate_environment_run_id(environment, run_id)
    _validate_environment_board(environment)
    _validate_environment_toolchain(environment)
    _validate_environment_sampler(environment)


def _finite_cooldown_value(value: Any) -> float | None:
    """Return a finite telemetry value, or None when the sample is unusable."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def evaluate_cooldown_gate(
    power_w: Any,
    asic_temperature_c: Any,
    *,
    baseline_power_w: float = COOLDOWN_BASELINE_POWER_W,
    baseline_temperature_c: float = COOLDOWN_BASELINE_ASIC_TEMPERATURE_C,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
    observed_idle_aiclk_mhz: Any = None,
) -> dict[str, Any]:
    """Evaluate the run-2 readiness gate without making AICLK a condition."""
    baseline_power = _finite_cooldown_value(baseline_power_w)
    baseline_temperature = _finite_cooldown_value(baseline_temperature_c)
    power_tolerance = _finite_cooldown_value(power_tolerance_w)
    temperature_tolerance = _finite_cooldown_value(temperature_tolerance_c)
    observed_power = _finite_cooldown_value(power_w)
    observed_temperature = _finite_cooldown_value(asic_temperature_c)
    observed_aiclk = _finite_cooldown_value(observed_idle_aiclk_mhz)
    valid_baseline = all(
        value is not None
        for value in (
            baseline_power,
            baseline_temperature,
            power_tolerance,
            temperature_tolerance,
        )
    ) and power_tolerance >= 0.0 and temperature_tolerance >= 0.0
    power_limit = (
        baseline_power + power_tolerance if valid_baseline else None
    )
    temperature_limit = (
        baseline_temperature + temperature_tolerance if valid_baseline else None
    )
    power_passes = (
        observed_power is not None
        and power_limit is not None
        and observed_power <= power_limit
    )
    temperature_passes = (
        observed_temperature is not None
        and temperature_limit is not None
        and observed_temperature <= temperature_limit
    )
    passes = bool(power_passes and temperature_passes)
    reason = None
    if not valid_baseline or observed_power is None or observed_temperature is None:
        reason = "cooldown telemetry sample is missing or non-finite"
    elif not passes:
        reason = "cooldown power/temperature gate did not pass"
    return {
        "baseline_source": COOLDOWN_BASELINE_SOURCE,
        "baseline_power_w": baseline_power,
        "baseline_temperature_c": baseline_temperature,
        "power_tolerance_w": power_tolerance,
        "temperature_tolerance_c": temperature_tolerance,
        "power_limit_w": power_limit,
        "temperature_limit_c": temperature_limit,
        "observed_power_w": observed_power,
        "observed_temperature_c": observed_temperature,
        "observed_idle_aiclk_mhz": observed_aiclk,
        "aiclk_gate": False,
        "aiclk_role": "provenance_only",
        "power_passes": power_passes,
        "temperature_passes": temperature_passes,
        "passes": passes,
        "result": "pass" if passes else "fail",
        "verified": passes,
        "failure_type": None if passes else "cooldown_gate_failure",
        "failure_reason": reason,
    }


def build_cooldown_metadata(
    *,
    run_number: int,
    wait_seconds: float | None,
    expected_aiclk_mhz: float | None,
    environment: Mapping[str, Any],
    start_utc: str | None = None,
    end_utc: str | None = None,
    start_monotonic: float | None = None,
    end_monotonic: float | None = None,
    wait_performed: bool = False,
    baseline_source: str = COOLDOWN_BASELINE_SOURCE,
    baseline_power_w: float = COOLDOWN_BASELINE_POWER_W,
    baseline_temperature_c: float = COOLDOWN_BASELINE_ASIC_TEMPERATURE_C,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
    observed_power_w: float | None = None,
    observed_temperature_c: float | None = None,
    observed_idle_aiclk_mhz: float | None = None,
    gate_result: str | None = None,
    gate_verified: bool | None = None,
) -> dict[str, Any]:
    """Build cooldown metadata supplied by the wrapper before measurement."""
    observed = environment.get("aiclk_mhz_observed")
    observed_values = list(observed) if isinstance(observed, list) else None
    if observed_idle_aiclk_mhz is None and observed_values:
        observed_idle_aiclk_mhz = observed_values[-1]
    actual_wait_seconds = None
    if start_utc is not None and end_utc is not None:
        start = _parse_trace_timestamp(start_utc, field="cooldown start_utc")
        end = _parse_trace_timestamp(end_utc, field="cooldown end_utc")
        actual_wait_seconds = (end - start).total_seconds()
    gate = evaluate_cooldown_gate(
        observed_power_w,
        observed_temperature_c,
        baseline_power_w=baseline_power_w,
        baseline_temperature_c=baseline_temperature_c,
        power_tolerance_w=power_tolerance_w,
        temperature_tolerance_c=temperature_tolerance_c,
        observed_idle_aiclk_mhz=observed_idle_aiclk_mhz,
    )
    gate["baseline_source"] = baseline_source
    gate_result = gate_result or (gate["result"] if observed_power_w is not None else "pending")
    if gate_verified is None:
        gate_verified = gate_result == "pass" and gate["passes"]
    cooldown_wait_seconds = wait_seconds
    if run_number == 2 and cooldown_wait_seconds is None:
        cooldown_wait_seconds = COOLDOWN_WAIT_CAP_SECONDS
    metadata = {
        "required": run_number == 2,
        "run_number": run_number,
        "cooldown_wait_seconds": cooldown_wait_seconds,
        "wait_cap_seconds": COOLDOWN_WAIT_CAP_SECONDS if run_number == 2 else None,
        "actual_wait_seconds": actual_wait_seconds,
        "wait_performed": wait_performed,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "wait_bounds": {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "source": "enodia/tt/bench/run_in_container.sh wrapper UTC bounds",
        },
        "start_monotonic": start_monotonic,
        "end_monotonic": end_monotonic,
        # Retain this historical field as a non-gating observation.  The
        # corrected contract never compares it to a required clock value.
        "expected_aiclk_mhz": expected_aiclk_mhz,
        "environment_observed_aiclk_mhz": observed_values,
        "observed_idle_aiclk_mhz": _finite_cooldown_value(observed_idle_aiclk_mhz),
        "environment_observation_source": "wrapper capture-env before this run",
        "bounds_source": "enodia/tt/bench/run_in_container.sh wrapper UTC bounds",
        "baseline_source": baseline_source,
        "baseline_power_w": baseline_power_w,
        "baseline_temperature_c": baseline_temperature_c,
        "power_tolerance_w": power_tolerance_w,
        "temperature_tolerance_c": temperature_tolerance_c,
        "power_limit_w": COOLDOWN_POWER_LIMIT_W,
        "temperature_limit_c": COOLDOWN_TEMPERATURE_LIMIT_C,
        "observed_power_w": observed_power_w,
        "observed_temperature_c": observed_temperature_c,
        "gate": gate,
        "gate_result": gate_result,
        "gate_verified": bool(gate_verified),
        "aiclk_gate": False,
        "aiclk_role": "provenance_only",
        "environment_verified": False,
        "power_trace_facts": None,
        "power_trace_verified": False,
        "verified": False,
    }
    validate_cooldown_metadata(metadata, environment=environment)
    metadata["environment_verified"] = True
    return metadata


def _validate_cooldown_wait(metadata: Mapping[str, Any]) -> float:
    try:
        wait_seconds = float(metadata.get("cooldown_wait_seconds"))
        wait_cap = float(metadata.get("wait_cap_seconds"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("run 2 cooldown wait bounds are missing") from exc
    if not math.isfinite(wait_seconds) or wait_seconds <= 0.0:
        raise ValueError("run 2 cooldown_wait_seconds must be positive")
    if not math.isfinite(wait_cap) or not _same_float(
        wait_cap, COOLDOWN_WAIT_CAP_SECONDS
    ):
        raise ValueError("run 2 cooldown wait cap must be 600 seconds")
    if wait_seconds > wait_cap:
        raise ValueError("run 2 cooldown_wait_seconds exceeds the 600-second cap")
    if metadata.get("wait_performed") is not True:
        raise ValueError("run 2 cooldown wait was not performed")
    return wait_cap


def _validate_cooldown_provenance(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata.get("environment_observed_aiclk_mhz"), list):
        raise TypeError("run 2 environment AICLK provenance is missing")
    if metadata.get("aiclk_gate") is not False:
        raise ValueError("run 2 AICLK must remain provenance only")
    if metadata.get("aiclk_role") != "provenance_only":
        raise ValueError("run 2 AICLK role is not provenance-only")
    baseline_source = metadata.get("baseline_source")
    if not isinstance(baseline_source, str) or not baseline_source.strip():
        raise ValueError("run 2 cooldown baseline source is missing")


def _cooldown_numeric_values(metadata: Mapping[str, Any]) -> dict[str, float]:
    fields = (
        "baseline_power_w",
        "baseline_temperature_c",
        "power_tolerance_w",
        "temperature_tolerance_c",
        "power_limit_w",
        "temperature_limit_c",
    )
    values: dict[str, float] = {}
    for field in fields:
        value = _finite_cooldown_value(metadata.get(field))
        if value is None:
            raise ValueError(f"run 2 cooldown {field} is missing or non-finite")
        values[field] = value
    return values


def _validate_cooldown_numeric_values(values: Mapping[str, float]) -> None:
    if values["power_tolerance_w"] < 0.0 or values["temperature_tolerance_c"] < 0.0:
        raise ValueError("run 2 cooldown tolerances must not be negative")
    expected = {
        "baseline_power_w": COOLDOWN_BASELINE_POWER_W,
        "baseline_temperature_c": COOLDOWN_BASELINE_ASIC_TEMPERATURE_C,
        "power_tolerance_w": COOLDOWN_POWER_TOLERANCE_W,
        "temperature_tolerance_c": COOLDOWN_TEMPERATURE_TOLERANCE_C,
        "power_limit_w": COOLDOWN_POWER_LIMIT_W,
        "temperature_limit_c": COOLDOWN_TEMPERATURE_LIMIT_C,
    }
    if any(
        not math.isclose(values[field], expected_value, rel_tol=0.0, abs_tol=1e-9)
        for field, expected_value in expected.items()
    ):
        raise ValueError("run 2 cooldown baseline and tolerances are fixed")
    limits_match = (
        math.isclose(
            values["power_limit_w"],
            values["baseline_power_w"] + values["power_tolerance_w"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            values["temperature_limit_c"],
            values["baseline_temperature_c"] + values["temperature_tolerance_c"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    if not limits_match:
        raise ValueError("run 2 cooldown gate limits do not match baseline tolerances")


def _validate_cooldown_time_bounds(
    metadata: Mapping[str, Any], *, wait_cap: float
) -> None:
    for field in ("start_utc", "end_utc"):
        _parse_trace_timestamp(metadata.get(field), field=f"cooldown {field}")
    try:
        actual_wait = float(metadata.get("actual_wait_seconds"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("run 2 cooldown actual_wait_seconds is missing") from exc
    if not math.isfinite(actual_wait) or actual_wait <= 0.0 or actual_wait > wait_cap:
        raise ValueError("run 2 cooldown UTC bounds exceed the 600-second cap")
    monotonic_values = [metadata.get("start_monotonic"), metadata.get("end_monotonic")]
    if all(value is None for value in monotonic_values):
        return
    if any(value is None for value in monotonic_values):
        raise ValueError("cooldown monotonic bounds are invalid")
    valid_values = all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in monotonic_values
    )
    if not valid_values or float(monotonic_values[1]) <= float(monotonic_values[0]):
        raise ValueError("cooldown monotonic bounds are invalid")


def _validate_cooldown_gate_proof(
    metadata: Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None,
    require_power_trace: bool,
) -> None:
    gate_result = metadata.get("gate_result")
    if gate_result not in {"pending", "pass", "fail"}:
        raise ValueError("run 2 cooldown gate result is invalid")
    if not isinstance(metadata.get("gate_verified"), bool):
        raise TypeError("run 2 cooldown gate verification is missing")
    if environment is not None:
        validate_harness_environment(environment)
    if require_power_trace and (
        metadata.get("power_trace_verified") is not True
        or metadata.get("gate_result") != "pass"
        or metadata.get("gate_verified") is not True
    ):
        raise ValueError("run 2 cooldown gate proof is missing")


def validate_cooldown_metadata(
    metadata: Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
    require_power_trace: bool = False,
) -> None:
    """Reject missing bounds, baseline evidence, or required gate proof."""
    if not isinstance(metadata, Mapping):
        raise TypeError("cooldown metadata is missing")
    run_number = metadata.get("run_number")
    if run_number not in (1, 2):
        raise ValueError("cooldown metadata run_number must be 1 or 2")
    required = run_number == 2
    if bool(metadata.get("required")) != required:
        raise ValueError("cooldown required flag disagrees with run number")
    if run_number == 1:
        if metadata.get("wait_performed") not in (False, None):
            raise ValueError("run 1 must not claim a cooldown wait")
        return
    wait_cap = _validate_cooldown_wait(metadata)
    _validate_cooldown_provenance(metadata)
    _validate_cooldown_numeric_values(_cooldown_numeric_values(metadata))
    _validate_cooldown_time_bounds(metadata, wait_cap=wait_cap)
    _validate_cooldown_gate_proof(
        metadata,
        environment=environment,
        require_power_trace=require_power_trace,
    )


def _cooldown_trace_sample(
    samples: Sequence[Mapping[str, Any]], *, end_utc: Any
) -> dict[str, Any] | None:
    """Return the newest usable cooldown sample at or before the gate end."""
    try:
        end = _parse_trace_timestamp(end_utc, field="cooldown end_utc")
    except (TypeError, ValueError):
        return None
    candidates: list[tuple[datetime.datetime, Mapping[str, Any]]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        try:
            timestamp = _parse_trace_timestamp(
                sample.get("timestamp_utc"), field="cooldown sample timestamp_utc"
            )
        except (TypeError, ValueError):
            continue
        if timestamp <= end:
            power = _finite_cooldown_value(sample.get("power_w"))
            temperature = _finite_cooldown_value(sample.get("asic_temp_c"))
            if power is not None and temperature is not None:
                candidates.append((timestamp, sample))
    if not candidates:
        return None
    return dict(max(candidates, key=lambda item: item[0])[1])


def verify_cooldown_power_trace(
    metadata: dict[str, Any], *, power_trace: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify power/temperature readiness from the final immutable trace."""
    if metadata.get("run_number") != 2:
        metadata["power_trace_verified"] = False
        metadata["verified"] = metadata.get("environment_verified") is True
        return metadata
    samples = power_trace.get("samples")
    if not isinstance(samples, list):
        facts: dict[str, Any] = {
            "available": False,
            "coverage_complete": False,
            "reason": "immutable_trace_samples_unavailable",
            "segments": [],
        }
        trace_sample = None
    else:
        facts = left_held_aiclk_interval_facts(
            samples,
            start=metadata.get("start_utc"),
            end=metadata.get("end_utc"),
            row_label="run 2 cooldown AICLK interval",
        )
        trace_sample = _cooldown_trace_sample(samples, end_utc=metadata.get("end_utc"))
    observed_power = metadata.get("observed_power_w")
    observed_temperature = metadata.get("observed_temperature_c")
    observed_aiclk = metadata.get("observed_idle_aiclk_mhz")
    if observed_power is None and trace_sample is not None:
        observed_power = trace_sample.get("power_w")
    if observed_temperature is None and trace_sample is not None:
        observed_temperature = trace_sample.get("asic_temp_c")
    if observed_aiclk is None and trace_sample is not None:
        observed_aiclk = trace_sample.get("aiclk_mhz")
    gate = evaluate_cooldown_gate(
        observed_power,
        observed_temperature,
        baseline_power_w=metadata.get("baseline_power_w"),
        baseline_temperature_c=metadata.get("baseline_temperature_c"),
        power_tolerance_w=metadata.get("power_tolerance_w"),
        temperature_tolerance_c=metadata.get("temperature_tolerance_c"),
        observed_idle_aiclk_mhz=observed_aiclk,
    )
    gate["baseline_source"] = metadata.get("baseline_source")
    facts.update(
        {
            "trace": power_trace.get("file"),
            "trace_source": power_trace.get("source_file"),
            "trace_sha256": power_trace.get("sha256"),
            "trace_sha256_definition": power_trace.get("sha256_definition"),
            "requested_sampler_interval_seconds": SAMPLER_INTERVAL_SECONDS,
            "coverage_definition": power_trace.get("coverage_definition"),
            "trace_coverage_complete": power_trace.get("coverage_complete") is True,
            "immutable_snapshot": power_trace.get("immutable_snapshot") is True,
            "terminal_held_aiclk_mhz": facts.get("last_held_aiclk_mhz"),
            "gate_sample": trace_sample,
            "gate": gate,
            "aiclk_gate": False,
            "aiclk_role": "provenance_only",
        }
    )
    # AICLK interval facts are retained as provenance.  They are deliberately
    # not part of the cooldown gate: an idle 800-MHz observation is valid.
    verified = (
        gate["passes"]
        and facts.get("trace_coverage_complete") is True
        and facts.get("immutable_snapshot") is True
    )
    metadata["observed_power_w"] = gate["observed_power_w"]
    metadata["observed_temperature_c"] = gate["observed_temperature_c"]
    metadata["observed_idle_aiclk_mhz"] = gate["observed_idle_aiclk_mhz"]
    metadata["gate"] = gate
    metadata["gate_result"] = gate["result"]
    metadata["gate_verified"] = bool(verified)
    metadata["power_trace_facts"] = facts
    metadata["power_trace_verified"] = bool(verified)
    metadata["verified"] = metadata.get("environment_verified") is True and bool(verified)
    return metadata


def validate_sampler_decision(
    sampler: Mapping[str, Any], *, trace_timestamp_gaps: Sequence[float] | None = None
) -> None:
    """Validate the measured-run sampler choice and its non-shortening rationale."""
    normalized = normalize_sampler_metadata(sampler, complete_default=False)
    if (
        normalized["mode"] != "default"
        or not _same_float(normalized["interval_seconds"], 2.0)
        or not _same_float(sampler.get("requested_interval_seconds"), 2.0)
    ):
        raise ValueError("Issue #127 sampler must be default at 2.0 seconds")
    if sampler.get("shortened_interval") is not False:
        raise ValueError("Issue #127 sampler shortening must remain disabled")
    if sampler.get("sample_before_sleep") is not True:
        raise ValueError("sampler must snapshot before sleeping")
    if trace_timestamp_gaps is not None:
        gaps = [float(value) for value in trace_timestamp_gaps]
        if not gaps or any(not math.isfinite(value) or value <= 0.0 for value in gaps):
            raise ValueError("power trace timestamp gaps are unavailable")


def _flops_per_launch(row: Mapping[str, Any]) -> int:
    size = int(row["size"])
    return BATCH * 4 * 2 * size**3 * (2 * FIXED_ITERATIONS)


def _timing_summary(samples: Sequence[float], flops_per_launch: int) -> dict[str, Any]:
    if not samples:
        raise ValueError("timed launch samples are missing")
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timed launch samples must be finite and positive")
    ordered = sorted(values)

    def percentile(quantile: float) -> float:
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    p50_seconds = percentile(0.5)
    return {
        "launches_measured": len(values),
        "seconds_per_launch_samples": values,
        "seconds_per_launch_min": min(values),
        "seconds_per_launch_p50": p50_seconds,
        "seconds_per_launch_p99": percentile(0.99),
        "seconds_per_launch_p99_9": percentile(0.999),
        "raw_p50_tflops": flops_per_launch / p50_seconds / 1e12,
        "raw_p50_definition": (
            "flops_per_launch / P50 of synchronized timed launches / 1e12; "
            "correctness and occurrence warm-up launches are excluded"
        ),
    }


def _kernel_prepare_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant": config["variant"],
        "math_fidelity": config["math_fidelity"],
        "fuse_s": config["fuse_s"],
        "batch_reads": False,
        "matrix_block": config["matrix_block"],
        "double_buffer": config["double_buffer"],
        "input_memory": config["input_memory"],
        "r_memory": config["r_memory"],
        "x0_memory": config["x0_memory"],
        "output_memory": config["output_memory"],
        "fp32_dest_acc_en": config["fp32_dest_acc_en"],
        "dst_full_sync_en": config["dst_full_sync_en"],
        "iterations": FIXED_ITERATIONS,
    }


def _run_sustained_warmup(
    ttnn: Any,
    device: Any,
    matrices: np.ndarray,
    config: Mapping[str, Any],
    *,
    kernel_class: Any,
    run_deadline: float,
    run_timeout_seconds: float,
    time_fn: Callable[[], float],
    utc_now_fn: Callable[[], datetime.datetime],
) -> dict[str, Any]:
    """Run the separate BF16-R/L32/R-in-DRAM startup warm-up."""
    start_monotonic = time_fn()
    start_utc = _timestamp(utc_now_fn(), field="warmup_start_utc")
    kernel = None
    launches_completed = 0
    try:
        kernel = kernel_class.prepare(
            ttnn, device, matrices, **_kernel_prepare_kwargs(config)
        )
        for _ in range(WARMUP_LAUNCHES):
            if time_fn() > run_deadline:
                raise TimeoutError(
                    f"Issue #127 run exceeded {run_timeout_seconds:g} seconds during warm-up"
                )
            kernel.launch()
            ttnn.synchronize_device(device)
            launches_completed += 1
    finally:
        if kernel is not None:
            kernel.close()
    end_monotonic = time_fn()
    end_utc = _timestamp(utc_now_fn(), field="warmup_end_utc")
    return {
        "configuration": config["name"],
        "variant": config["variant"],
        "size": config["size"],
        "r_memory": config["r_memory"],
        "launches_requested": WARMUP_LAUNCHES,
        "launches_completed": launches_completed,
        "synchronized": True,
        "measured": False,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "start_monotonic": start_monotonic,
        "end_monotonic": end_monotonic,
        "duration_seconds": end_monotonic - start_monotonic,
        "startup_hold_aiclk_mhz": STARTUP_HOLD_AICLK_MHZ,
        "gate": "first measured row must begin after the startup 800-MHz hold",
        "hold_gate_verified": False,
    }


def _initial_occurrence_result(
    config: Mapping[str, Any], occurrence: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(occurrence)
    result.update(
        {
            "shape_name": f"newton_schulz_L{config['size']}_b{BATCH}",
            "flops_per_launch": _flops_per_launch(config),
            "status": "failed",
            "correctness_launch": None,
            "occurrence_warmup": None,
            "timing_protocol": (
                "one synchronized correctness launch and one synchronized occurrence "
                "warm-up excluded from exactly the fixed timed launch count"
            ),
        }
    )
    return result


def _check_occurrence_deadline(
    run_deadline: float, *, time_fn: Callable[[], float], message: str
) -> None:
    if time_fn() > run_deadline:
        raise TimeoutError(message)


def _run_correctness_launch(
    ttnn: Any,
    device: Any,
    kernel: Any,
    reference: Mapping[str, Any],
    result: dict[str, Any],
    *,
    run_deadline: float,
    run_timeout_seconds: float,
    time_fn: Callable[[], float],
) -> dict[str, Any]:
    _check_occurrence_deadline(
        run_deadline,
        time_fn=time_fn,
        message=(
            f"Issue #127 run exceeded {run_timeout_seconds:g} seconds before correctness"
        ),
    )
    kernel.launch()
    ttnn.synchronize_device(device)
    correctness = issue88.correctness_metrics(kernel.result(), reference)
    result["correctness"] = correctness
    if not correctness["device_test_gate"]["pass"]:
        raise ValueError("correctness device-test gate failed")
    return correctness


def _run_occurrence_warmup(
    ttnn: Any,
    device: Any,
    kernel: Any,
    *,
    run_deadline: float,
    run_timeout_seconds: float,
    time_fn: Callable[[], float],
) -> None:
    _check_occurrence_deadline(
        run_deadline,
        time_fn=time_fn,
        message=(
            f"Issue #127 run exceeded {run_timeout_seconds:g} seconds before occurrence warm-up"
        ),
    )
    kernel.launch()
    ttnn.synchronize_device(device)


def _run_timed_occurrence_launches(
    ttnn: Any,
    device: Any,
    kernel: Any,
    config: Mapping[str, Any],
    *,
    run_deadline: float,
    run_timeout_seconds: float,
    time_fn: Callable[[], float],
    utc_now_fn: Callable[[], datetime.datetime],
    timing: dict[str, Any],
) -> None:
    timing["timed_start_utc"] = _timestamp(utc_now_fn(), field="timed_start_utc")
    timing["timed_start_monotonic"] = time_fn()
    for _ in range(launches_for_row(config)):
        _check_occurrence_deadline(
            run_deadline,
            time_fn=time_fn,
            message=(
                f"Issue #127 run exceeded {run_timeout_seconds:g} seconds during timed launches"
            ),
        )
        launch_started = time_fn()
        kernel.launch()
        ttnn.synchronize_device(device)
        timing["samples"].append(time_fn() - launch_started)
    timing["timed_end_monotonic"] = time_fn()
    timing["timed_end_utc"] = _timestamp(utc_now_fn(), field="timed_end_utc")


def _complete_occurrence_result(
    result: dict[str, Any], config: Mapping[str, Any], timing: Mapping[str, Any]
) -> None:
    samples = timing["samples"]
    launch_count = launches_for_row(config)
    result.update(_timing_summary(samples, _flops_per_launch(config)))
    result.update(
        {
            "timed_start_utc": timing["timed_start_utc"],
            "timed_end_utc": timing["timed_end_utc"],
            "timed_start_monotonic": timing["timed_start_monotonic"],
            "timed_end_monotonic": timing["timed_end_monotonic"],
            "timed_duration_seconds": (
                timing["timed_end_monotonic"] - timing["timed_start_monotonic"]
            ),
            "timed_launches_requested": launch_count,
            "launches_requested": launch_count,
            "timed_launches_excluded": 2,
        }
    )
    validate_row_bounds(result)
    result["duration_guard"] = row_duration_guard(
        result["timed_duration_seconds"], config
    )
    result["status"] = "ok" if result["duration_guard"]["passes"] else "failed"
    if result["status"] != "ok":
        result["failure"] = "timed row duration did not exceed the sampler gap"


def _record_occurrence_failure(
    result: dict[str, Any], config: Mapping[str, Any], timing: Mapping[str, Any], exc: Exception
) -> None:
    result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    start_utc = timing.get("timed_start_utc")
    end_utc = timing.get("timed_end_utc")
    start_monotonic = timing.get("timed_start_monotonic")
    end_monotonic = timing.get("timed_end_monotonic")
    result["failure_stage"] = "timed_row" if start_utc else "row_setup"
    if start_utc is not None:
        result["timed_start_utc"] = start_utc
        result["timed_start_monotonic"] = start_monotonic
    if end_utc is not None:
        result["timed_end_utc"] = end_utc
        result["timed_end_monotonic"] = end_monotonic
    if start_monotonic is not None and end_monotonic is not None:
        result["timed_duration_seconds"] = end_monotonic - start_monotonic
    if timing["samples"]:
        result.update(_timing_summary(timing["samples"], _flops_per_launch(config)))
    result["timed_launches_requested"] = launches_for_row(config)
    result["launches_requested"] = launches_for_row(config)


def _close_occurrence_kernel(result: dict[str, Any], kernel: Any | None) -> None:
    if kernel is None:
        return
    try:
        kernel.close()
    except Exception as exc:  # noqa: BLE001 - preserve cleanup failure
        result.setdefault("cleanup_failures", []).append(type(exc).__name__)


def _run_occurrence(
    ttnn: Any,
    device: Any,
    matrices: np.ndarray,
    config: Mapping[str, Any],
    occurrence: Mapping[str, Any],
    *,
    kernel_class: Any,
    run_deadline: float,
    run_timeout_seconds: float,
    time_fn: Callable[[], float],
    utc_now_fn: Callable[[], datetime.datetime],
) -> dict[str, Any]:
    """Run one correctness launch, one excluded warm-up, and fixed timed work."""
    result = _initial_occurrence_result(config, occurrence)
    kernel = None
    timing: dict[str, Any] = {"samples": []}
    try:
        preflight = issue88._preflight_accounting(config, ttnn)
        result["preflight"] = preflight
        if not preflight["accepted"]:
            raise ValueError("approved Issue #127 row failed its preflight")
        reference = issue88.reference_context(matrices, config["variant"])
        kernel = kernel_class.prepare(
            ttnn, device, matrices, **_kernel_prepare_kwargs(config)
        )
        correctness = _run_correctness_launch(
            ttnn,
            device,
            kernel,
            reference,
            result,
            run_deadline=run_deadline,
            run_timeout_seconds=run_timeout_seconds,
            time_fn=time_fn,
        )
        result["correctness_launch"] = {
            "launches": 1,
            "synchronized": True,
            "downloaded": True,
            "finite": correctness["finite"],
            "excluded_from_timing": True,
        }
        _run_occurrence_warmup(
            ttnn,
            device,
            kernel,
            run_deadline=run_deadline,
            run_timeout_seconds=run_timeout_seconds,
            time_fn=time_fn,
        )
        result["occurrence_warmup"] = {
            "launches": 1,
            "synchronized": True,
            "excluded_from_timing": True,
        }
        _run_timed_occurrence_launches(
            ttnn,
            device,
            kernel,
            config,
            run_deadline=run_deadline,
            run_timeout_seconds=run_timeout_seconds,
            time_fn=time_fn,
            utc_now_fn=utc_now_fn,
            timing=timing,
        )
        _complete_occurrence_result(result, config, timing)
    except Exception as exc:  # noqa: BLE001 - preserve a row-local board failure
        _record_occurrence_failure(result, config, timing, exc)
    finally:
        _close_occurrence_kernel(result, kernel)
        result["timed_launches_measured"] = len(timing["samples"])
    return result


def _prepare_inputs(
    config: Mapping[str, Any],
    *,
    matrices_by_size: dict[int, np.ndarray],
    references_by_key: dict[tuple[int, str], dict[str, Any]],
    fingerprints: dict[str, dict[str, Any]],
    matrices_factory: Callable[..., np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    size = int(config["size"])
    if size not in matrices_by_size:
        matrices_by_size[size] = matrices_factory(
            BATCH,
            size,
            condition_number=CONDITION_NUMBER,
            seed=INPUT_SEED,
        )
        fingerprints[str(size)] = {
            "size": size,
            "batch": BATCH,
            "condition_number": CONDITION_NUMBER,
            "seed": INPUT_SEED,
            "matrix_sha256": issue88._array_sha256(matrices_by_size[size]),
        }
    key = (size, config["variant"])
    if key not in references_by_key:
        references_by_key[key] = issue88.reference_context(
            matrices_by_size[size], config["variant"]
        )
    return matrices_by_size[size], references_by_key[key]


def _resolve_comparison_timeout(
    run_number: int,
    requested_seconds: float,
    policy: Mapping[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    try:
        effective_timeout = float(requested_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Issue #127 run timeout must be numeric") from exc
    if not math.isfinite(effective_timeout) or effective_timeout <= 0.0:
        raise ValueError("Issue #127 run timeout must be finite and positive")
    if policy is None:
        if not _same_float(effective_timeout, DEFAULT_RUN_TIMEOUT_SECONDS):
            raise ValueError(
                "Issue #127 non-default timeout requires the owner-approved "
                "run timeout policy"
            )
        return resolve_run_timeout(run_number)
    policy_timeout = policy.get("effective_run_timeout_seconds", effective_timeout)
    resolved_timeout, _ = resolve_run_timeout(
        run_number, policy_timeout, policy.get("scope")
    )
    if not _same_float(resolved_timeout, effective_timeout):
        raise ValueError("Issue #127 run timeout policy does not match its budget")
    return effective_timeout, dict(policy)


def _run_comparison_occurrences(
    ttnn: Any,
    device: Any,
    plan: Sequence[Mapping[str, Any]],
    *,
    matrices_by_size: dict[int, np.ndarray],
    references_by_key: dict[tuple[int, str], dict[str, Any]],
    fingerprints: dict[str, dict[str, Any]],
    matrices_factory: Callable[..., np.ndarray],
    kernel_class: Any,
    run_deadline: float,
    run_timeout_seconds: float,
    time_fn: Callable[[], float],
    utc_now_fn: Callable[[], datetime.datetime],
    start_monotonic: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any] | None]:
    occurrences: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    stopped_on_failure = False
    budget_failure: dict[str, Any] | None = None
    for occurrence in plan:
        projection = project_run_budget(
            time_fn() - start_monotonic,
            occurrences,
            len(plan) - len(occurrences),
            run_timeout_seconds=run_timeout_seconds,
        )
        projection.update(
            {
                "next_order_index": occurrence["order_index"],
                "next_occurrence_id": occurrence["occurrence_id"],
            }
        )
        projections.append(projection)
        if not projection["within_budget"]:
            budget_failure = {
                "type": "TimeoutError",
                "message": (
                    f"Issue #127 projected run completion exceeds the "
                    f"{run_timeout_seconds:g}-second budget before occurrence "
                    f"{occurrence['occurrence_id']}"
                ),
                "order_index": occurrence["order_index"],
                "projection": projection,
            }
            stopped_on_failure = True
            break
        config = ROW_BY_NAME[occurrence["row"]]
        matrices, _ = _prepare_inputs(
            config,
            matrices_by_size=matrices_by_size,
            references_by_key=references_by_key,
            fingerprints=fingerprints,
            matrices_factory=matrices_factory,
        )
        result = _run_occurrence(
            ttnn,
            device,
            matrices,
            config,
            occurrence,
            kernel_class=kernel_class,
            run_deadline=run_deadline,
            run_timeout_seconds=run_timeout_seconds,
            time_fn=time_fn,
            utc_now_fn=utc_now_fn,
        )
        occurrences.append(result)
        if result.get("status") != "ok":
            stopped_on_failure = True
            break
    return occurrences, projections, stopped_on_failure, budget_failure


def run_comparison(
    ttnn: Any,
    device: Any,
    *,
    run_number: int,
    run_id: str,
    run_timeout_seconds: float = RUN_TIMEOUT_SECONDS,
    run_timeout_policy: Mapping[str, Any] | None = None,
    time_fn: Callable[[], float] = time.monotonic,
    utc_now_fn: Callable[[], datetime.datetime] | None = None,
    kernel_class: Any | None = None,
    matrices_factory: Callable[..., np.ndarray] = random_hpd_batch,
) -> dict[str, Any]:
    """Run one fixed Issue #127 order on an already-open device."""
    validate_run_id(run_id)
    validate_comparison_rows()
    effective_timeout, timeout_policy = _resolve_comparison_timeout(
        run_number, run_timeout_seconds, run_timeout_policy
    )
    plan = build_occurrence_plan(run_number)
    now_utc = (
        (lambda: datetime.datetime.now(datetime.UTC))
        if utc_now_fn is None
        else utc_now_fn
    )
    if kernel_class is None:
        from enodia.tt.bench.newton_schulz_kernel import NewtonSchulzKernel

        kernel_class = NewtonSchulzKernel
    start_monotonic = time_fn()
    run_deadline = start_monotonic + effective_timeout
    matrices_by_size: dict[int, np.ndarray] = {}
    references_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    warmup_config = ROW_BY_NAME["bf16-r-L32-r-dram"]
    warmup_matrices, _ = _prepare_inputs(
        warmup_config,
        matrices_by_size=matrices_by_size,
        references_by_key=references_by_key,
        fingerprints=fingerprints,
        matrices_factory=matrices_factory,
    )
    warmup = _run_sustained_warmup(
        ttnn,
        device,
        warmup_matrices,
        warmup_config,
        kernel_class=kernel_class,
        run_deadline=run_deadline,
        run_timeout_seconds=effective_timeout,
        time_fn=time_fn,
        utc_now_fn=now_utc,
    )
    occurrences, projections, stopped_on_failure, budget_failure = (
        _run_comparison_occurrences(
            ttnn,
            device,
            plan,
            matrices_by_size=matrices_by_size,
            references_by_key=references_by_key,
            fingerprints=fingerprints,
            matrices_factory=matrices_factory,
            kernel_class=kernel_class,
            run_deadline=run_deadline,
            run_timeout_seconds=effective_timeout,
            time_fn=time_fn,
            utc_now_fn=now_utc,
            start_monotonic=start_monotonic,
        )
    )
    end_monotonic = time_fn()
    run_budget = {
        "policy": (
            "Project the fixed eight-occurrence run before each occurrence and stop "
            "before launch when the projection exceeds the effective run timeout."
        ),
        "projections": projections,
        "stopped_before_occurrence": (
            budget_failure["order_index"] if budget_failure is not None else None
        ),
        "failure": budget_failure,
    }
    run_duration = end_monotonic - start_monotonic
    run_under_timeout = run_duration <= effective_timeout
    return {
        "status": (
            "pass"
            if len(occurrences) == len(plan) and not stopped_on_failure and run_under_timeout
            else "failed"
        ),
        "run_number": run_number,
        "run_id": run_id,
        "order": list(order_for_run(run_number)),
        "occurrences_requested": len(plan),
        "occurrences_completed": len(occurrences),
        "stopped_on_failure": stopped_on_failure,
        "failure": budget_failure,
        "run_budget": run_budget,
        "run_start_monotonic": start_monotonic,
        "run_end_monotonic": end_monotonic,
        "run_duration_seconds": run_duration,
        "run_timeout_seconds": effective_timeout,
        "run_under_timeout": run_under_timeout,
        "run_timeout_policy": timeout_policy,
        "warmup": warmup,
        "comparison_rows": occurrences,
        "input_generation": {
            "generator": "enodia.tt.bench.newton_schulz_reference.random_hpd_batch",
            "condition_number": CONDITION_NUMBER,
            "seed": INPUT_SEED,
            "same_matrix_for_matching_size_variants": True,
            "fingerprints": fingerprints,
        },
    }


def _trace_timestamp_gaps(samples: Sequence[Mapping[str, Any]]) -> list[float]:
    timestamps = [
        _parse_trace_timestamp(sample.get("timestamp_utc"), field="timestamp_utc")
        for sample in samples
    ]
    return [
        (right - left).total_seconds()
        for left, right in itertools.pairwise(timestamps)
        if (right - left).total_seconds() > 0.0
    ]


def _trace_gap_metadata(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gaps = _trace_timestamp_gaps(samples)
    if not gaps:
        return {
            "sample_count": len(samples),
            "observed_timestamp_gap_seconds": [],
            "observed_timestamp_gap_seconds_median": None,
            "observed_timestamp_gap_seconds_min": None,
            "observed_timestamp_gap_seconds_max": None,
            "expected_timestamp_gap_seconds": SAMPLER_OBSERVED_GAP_EXPECTED_SECONDS,
        }
    return {
        "sample_count": len(samples),
        "observed_timestamp_gap_seconds": gaps,
        "observed_timestamp_gap_seconds_median": float(np.median(gaps)),
        "observed_timestamp_gap_seconds_min": min(gaps),
        "observed_timestamp_gap_seconds_max": max(gaps),
        "expected_timestamp_gap_seconds": SAMPLER_OBSERVED_GAP_EXPECTED_SECONDS,
    }


def apply_final_trace_facts(
    comparison_rows: Sequence[dict[str, Any]],
    *,
    power_trace: Mapping[str, Any],
) -> None:
    """Re-evaluate every occurrence against the immutable final trace."""
    samples = power_trace.get("samples")
    if not isinstance(samples, list):
        for row in comparison_rows:
            row["aiclk_interval"] = {
                "available": False,
                "coverage_complete": False,
                "reason": "immutable_trace_samples_unavailable",
                "trace": power_trace.get("file"),
                "trace_sha256": power_trace.get("sha256"),
                "requested_sampler_interval_seconds": SAMPLER_INTERVAL_SECONDS,
                "coverage_definition": EXPLICIT_FINAL_SAMPLE,
                "power_sample_count_in_interval": 0,
                "power_sample_target": POWER_SAMPLES_PER_ROW_TARGET,
                "power_sample_target_met": False,
            }
            row["adjusted_p50_tflops"] = None
        return
    for row in comparison_rows:
        facts = left_held_aiclk_interval_facts(
            samples,
            start=row.get("timed_start_utc"),
            end=row.get("timed_end_utc"),
            row_label=f"{row.get('occurrence_id', 'row')} AICLK interval",
        )
        facts.update(
            {
                "trace": power_trace.get("file"),
                "trace_source": power_trace.get("source_file"),
                "trace_sha256": power_trace.get("sha256"),
                "trace_sha256_definition": power_trace.get("sha256_definition"),
                "requested_sampler_interval_seconds": SAMPLER_INTERVAL_SECONDS,
                "coverage_definition": power_trace.get("coverage_definition"),
                "trace_coverage_complete": power_trace.get("coverage_complete") is True,
                "immutable_snapshot": power_trace.get("immutable_snapshot") is True,
                "power_sample_count_in_interval": len(facts.get("segments", [])),
                "power_sample_target": POWER_SAMPLES_PER_ROW_TARGET,
                "power_sample_target_met": len(facts.get("segments", []))
                >= POWER_SAMPLES_PER_ROW_TARGET,
            }
        )
        row["aiclk_interval"] = facts
        row["row_time_weighted_aiclk_mhz"] = facts.get("time_weighted_aiclk_mhz")
        raw_p50 = row.get("raw_p50_tflops")
        row["adjusted_p50_tflops"] = adjusted_p50_tflops(
            raw_p50, facts.get("time_weighted_aiclk_mhz")
        ) if raw_p50 is not None else None


def verify_warmup_hold_gate(
    warmup: dict[str, Any], first_row: Mapping[str, Any], *, power_trace: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the first timed row is after the 800-MHz startup hold."""
    facts = first_row.get("aiclk_interval")
    verified = False
    reason = "row AICLK interval is unavailable"
    if isinstance(facts, Mapping):
        held = facts.get("first_held_aiclk_mhz")
        try:
            warmup_end = _parse_trace_timestamp(warmup.get("end_utc"), field="warmup end_utc")
            first_start = _parse_trace_timestamp(
                first_row.get("timed_start_utc"), field="first timed_start_utc"
            )
            held_value = float(held)
        except (TypeError, ValueError, OverflowError):
            reason = "warm-up or first-row timestamp/AICLK evidence is invalid"
        else:
            verified = (
                warmup_end <= first_start
                and held_value > STARTUP_HOLD_AICLK_MHZ
                and facts.get("coverage_complete") is True
                and facts.get("trace_coverage_complete") is True
            )
            reason = (
                "first row starts after warm-up and its left-held trace sample is above "
                "the startup 800-MHz hold"
                if verified
                else "first measured row did not pass the post-800-MHz hold gate"
            )
    gate = {
        "required": True,
        "startup_hold_aiclk_mhz": STARTUP_HOLD_AICLK_MHZ,
        "warmup_end_utc": warmup.get("end_utc"),
        "first_measured_row_start_utc": first_row.get("timed_start_utc"),
        "first_measured_row_left_held_aiclk_mhz": (
            facts.get("first_held_aiclk_mhz") if isinstance(facts, Mapping) else None
        ),
        "trace": power_trace.get("file"),
        "trace_sha256": power_trace.get("sha256"),
        "verified": verified,
        "reason": reason,
    }
    warmup["hold_gate"] = gate
    warmup["hold_gate_verified"] = verified
    return gate


def _aggregate_configuration_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce two occurrences to four configuration summaries without new rows."""
    summaries: list[dict[str, Any]] = []
    by_name: dict[str, list[Mapping[str, Any]]] = {name: [] for name in ISSUE127_ROW_NAMES}
    for row in rows:
        name = row.get("row")
        if name in by_name:
            by_name[name].append(row)
    for name in ISSUE127_ROW_NAMES:
        occurrences = by_name[name]
        raw_values = [float(row["raw_p50_tflops"]) for row in occurrences if row.get("raw_p50_tflops") is not None]
        adjusted_values = [
            float(row["adjusted_p50_tflops"])
            for row in occurrences
            if row.get("adjusted_p50_tflops") is not None
        ]
        aiclk_values = [
            float(row["row_time_weighted_aiclk_mhz"])
            for row in occurrences
            if row.get("row_time_weighted_aiclk_mhz") is not None
        ]
        summaries.append(
            {
                "row": name,
                "occurrence_indices": [row.get("order_index") for row in occurrences],
                "occurrence_count": len(occurrences),
                "raw_p50_tflops": float(np.mean(raw_values)) if raw_values else None,
                "adjusted_p50_tflops": float(np.mean(adjusted_values)) if adjusted_values else None,
                "row_time_weighted_aiclk_mhz": float(np.mean(aiclk_values)) if aiclk_values else None,
                "aggregation": "arithmetic mean of the two occurrence facts; adjusted P50 is supplementary",
            }
        )
    return summaries


def _resolve_result_path(args: argparse.Namespace) -> Path:
    value = os.environ.get("HEKATUS_TT_RESULT_PATH") or args.out
    if value is None:
        raise ValueError("Issue #127 requires HEKATUS_TT_RESULT_PATH or --out")
    path = Path(value)
    if not _SAFE_BASENAME_RE.fullmatch(path.name):
        raise ValueError("result path must end in a safe basename")
    return path


def _validated_run_artifact(
    path: Path,
    output_dir: Path,
    *,
    run_id: str,
    prefix: str,
    suffix: str,
    kind: str,
) -> Path:
    root = Path(output_dir).resolve()
    selected = require_run_artifact(
        root,
        prefix=prefix,
        suffix=suffix,
        run_id=run_id,
        explicit=Path(path),
        kind=kind,
    )
    try:
        resolved = selected.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{kind} path could not be resolved safely") from exc
    if selected.is_symlink() or resolved.parent != root:
        raise ValueError(f"{kind} path is outside the current run output directory")
    return resolved


def _read_environment(
    path: Path,
    *,
    run_id: str,
    output_dir: Path | None = None,
    read_text_fn: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    artifact_dir = (
        Path(path).parent if output_dir is None else Path(output_dir)
    ).resolve()
    _validated_run_artifact(
        path,
        artifact_dir,
        run_id=run_id,
        prefix="env-",
        suffix=JSON_SUFFIX,
        kind="environment",
    )
    safe_path = artifact_dir / run_artifact_name(
        prefix="env-", suffix=JSON_SUFFIX, run_id=run_id
    )
    text = (
        safe_path.read_text(encoding="utf-8")
        if read_text_fn is None
        else read_text_fn(safe_path)
    )
    raw = json.loads(text)
    normalized = sanitize_environment(raw)
    validate_harness_environment(normalized, run_id=run_id)
    return normalized


def _read_final_power_snapshot(
    source: Path,
    output_dir: Path,
    *,
    run_id: str,
    run_start: str,
    run_end: str,
    snapshot_number: int,
) -> dict[str, Any]:
    snapshot = atomic_power_trace_snapshot(
        source,
        output_dir,
        run_id=run_id,
        snapshot_number=snapshot_number,
        snapshot_prefix=POWER_TRACE_SNAPSHOT_PREFIX,
    )
    return read_power_trace_snapshot(
        snapshot,
        options=PowerTraceSnapshotOptions(
            run_id=run_id,
            run_start=run_start,
            run_end=run_end,
            snapshot_prefix=POWER_TRACE_SNAPSHOT_PREFIX,
            source_path=source,
            sampler_interval_seconds=SAMPLER_INTERVAL_SECONDS,
            coverage_definition=EXPLICIT_FINAL_SAMPLE,
            require_all_rows_valid=True,
            include_issue118_metadata=True,
        ),
    )


def _complete_final_power_trace(
    trace: dict[str, Any], *, samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    trace["poll_complete"] = True
    trace["requested_sampler_interval_seconds"] = SAMPLER_INTERVAL_SECONDS
    trace["coverage_contract"] = EXPLICIT_FINAL_SAMPLE
    trace["timestamp_gap_facts"] = _trace_gap_metadata(samples)
    return trace


def _capture_final_power_trace(
    output_dir: Path,
    *,
    run_id: str,
    run_start: str,
    run_end: str,
    power_path: Path,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    """Snapshot and read the wrapper CSV until the shared coverage gate passes."""
    source = output_dir / run_artifact_name(prefix="power-", suffix=".csv", run_id=run_id)
    if power_path.name != source.name or power_path.parent.resolve() != output_dir.resolve():
        raise ValueError("power trace path is not the current run artifact")
    deadline = monotonic_fn() + POWER_TRACE_POLL_TIMEOUT_SECONDS
    snapshot_number = 1
    last_trace: dict[str, Any] | None = None
    while True:
        if source.exists():
            try:
                trace = _read_final_power_snapshot(
                    source,
                    output_dir,
                    run_id=run_id,
                    run_start=run_start,
                    run_end=run_end,
                    snapshot_number=snapshot_number,
                )
            except (OSError, ValueError):
                pass
            else:
                snapshot_number += 1
                last_trace = trace
                if trace.get("coverage_complete") is True:
                    return _complete_final_power_trace(
                        trace, samples=trace.get("samples", [])
                    )
        if monotonic_fn() >= deadline:
            if last_trace is None:
                raise TimeoutError("no usable immutable Issue #127 power trace was captured")
            last_trace["poll_complete"] = False
            return last_trace
        sleep_fn(POWER_TRACE_POLL_INTERVAL_SECONDS)


def _atomic_json_write(
    path: Path, payload: Mapping[str, Any], *, output_dir: Path
) -> None:
    serialized = _serialize_record(payload)
    root = Path(output_dir).resolve()
    candidate = Path(path)
    if not _SAFE_BASENAME_RE.fullmatch(candidate.name) or not candidate.name.endswith(
        JSON_SUFFIX
    ):
        raise ValueError("JSON destination must have a safe basename")
    try:
        candidate_parent = candidate.parent.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("JSON destination parent could not be resolved safely") from exc
    if candidate.is_symlink() or candidate_parent != root:
        raise ValueError("JSON destination must stay in the current run output directory")
    destination = root / candidate.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_field_path(path: tuple[str, ...]) -> str:
    """Render a stable field path without exposing any value."""
    return ".".join(path) if path else "<record>"


_PATH_KIND_NONE = "none"
_PATH_KIND_WHOLE = "whole"
_PATH_KIND_EMBEDDED = "embedded"


def _formula_path_match(value: str, match: re.Match[str]) -> re.Match[str] | None:
    """Return a slash-prefixed function expression at a candidate start."""
    if value[match.start()] != "/":
        return None
    return _FORMULA_SLASH_RE.match(value, match.start()) or _FORMULA_SLASH_FRAGMENT_RE.match(
        value, match.start()
    )


def _path_matches(value: str) -> list[re.Match[str]]:
    """Return host-path candidates after the shared formula exception."""
    matches = []
    search_start = 0
    while candidate := _PATH_CANDIDATE_RE.search(value, search_start):
        formula = _formula_path_match(value, candidate)
        if formula is not None:
            search_start = formula.end()
            continue
        matches.append(candidate)
        search_start = candidate.end()
    return matches


def _classify_path(value: str) -> str:
    """Classify one string using the shared path grammar and exceptions."""
    matches = _path_matches(value)
    if not matches:
        return _PATH_KIND_NONE
    if any(match.start() == 0 and match.end() == len(value) for match in matches):
        return _PATH_KIND_WHOLE
    if (
        len(value) >= 2
        and value[0] in "'\""
        and value[-1] == value[0]
        and any(match.start() == 1 and match.end() == len(value) - 1 for match in matches)
    ):
        return _PATH_KIND_WHOLE
    return _PATH_KIND_EMBEDDED


def _is_whole_absolute_path(value: str) -> bool:
    """Recognize a complete host path without scanning formulas or prose."""
    return _classify_path(value) == _PATH_KIND_WHOLE


def _contains_embedded_host_path(value: str) -> bool:
    """Find host-path candidates embedded in diagnostics or mapping keys."""
    return bool(_path_matches(value))


def _looks_like_absolute_path(value: str) -> bool:
    """Recognize a path value for a field whose name explicitly carries a path."""
    return _classify_path(value) != _PATH_KIND_NONE


def _is_known_path_field(key: str | None) -> bool:
    return isinstance(key, str) and key.lower() in _RECORD_PATH_FIELD_NAMES


def _path_basename(value: str) -> str:
    """Extract a candidate basename without retaining its host-path prefix."""
    matches = _path_matches(value)
    if not matches:
        return ""
    token = matches[0].group()
    token = re.sub(r"^(?:file|path):[\\/]+", "", token, flags=re.IGNORECASE)
    return token.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


def _replace_known_path(value: str, *, field_path: str) -> tuple[str, dict[str, str]]:
    """Return a safe basename replacement and an audit fact for a path field."""
    key = field_path.rsplit(".", 1)[-1]
    if key == "device_node" and _DEVICE_NODE_RE.fullmatch(value):
        return value, {}
    if not _looks_like_absolute_path(value):
        return value, {}
    basename = _path_basename(value)
    replacement = basename if _SAFE_BASENAME_RE.fullmatch(basename) else REDACTED_PATH
    return replacement, {
        "field": field_path,
        "replacement": replacement,
        "reason": "absolute path replaced by harness-relative basename",
    }


def _environment_value_contains_path(value: str, *, key: str | None) -> bool:
    """Recognize embedded paths in environment values without matching formulas."""
    if key == DEVICE_NODE_KEY and _DEVICE_NODE_RE.fullmatch(value):
        return False
    if _contains_embedded_host_path(value):
        return True
    if isinstance(key, str) and key.lower() in _ENVIRONMENT_PATH_KEYS:
        if _DEVICE_NODE_RE.fullmatch(value):
            return False
        for marker in re.finditer(r"(?:^|[:;=])(?:/|~/|~\\|[A-Za-z]:[\\/])", value):
            path_start = marker.end() - 1
            if value[path_start] == "/" and (
                _FORMULA_SLASH_RE.match(value, path_start) is not None
                or _FORMULA_SLASH_FRAGMENT_RE.match(value, path_start) is not None
            ):
                continue
            return True
    return False


def _normalize_record_mapping(
    value: Mapping[Any, Any],
    *,
    path: tuple[str, ...],
    replacements: list[dict[str, str]],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, child_value in value.items():
        if not isinstance(key, str):
            raise TypeError("Issue #127 record has a non-string key")
        if _contains_embedded_host_path(key):
            raise ValueError(
                "Issue #127 record contains an unsafe path-shaped mapping key "
                f"in field {_record_field_path((*path, '<path-key>'))}"
            )
        child_path = (*path, key)
        if key.lower() in _FORBIDDEN_RECORD_KEYS:
            raise ValueError(
                f"Issue #127 record must not retain host identity field "
                f"{_record_field_path(child_path)}"
            )
        normalized[key] = _normalize_record_paths(
            child_value, path=child_path, replacements=replacements
        )
    return normalized


def _normalize_record_sequence(
    value: list[Any] | tuple[Any, ...],
    *,
    path: tuple[str, ...],
    replacements: list[dict[str, str]],
) -> list[Any]:
    return [
        _normalize_record_paths(
            item, path=(*path, f"[{index}]"), replacements=replacements
        )
        for index, item in enumerate(value)
    ]


def _normalize_record_string(
    value: str, *, path: tuple[str, ...], replacements: list[dict[str, str]]
) -> str:
    field_path = _record_field_path(path)
    key = path[-1] if path else None
    if key == DEVICE_NODE_KEY and _DEVICE_NODE_RE.fullmatch(value):
        return value
    if _is_known_path_field(key):
        replacement, audit = _replace_known_path(value, field_path=field_path)
        if audit:
            replacements.append(audit)
        return replacement
    if _contains_embedded_host_path(value):
        raise ValueError(
            f"Issue #127 record contains an embedded host path in field {field_path}"
        )
    if _is_whole_absolute_path(value):
        raise ValueError(
            f"Issue #127 record contains an unsafe absolute path in field {field_path}"
        )
    return value


def _normalize_record_paths(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    replacements: list[dict[str, str]],
) -> Any:
    """Copy records while changing only explicitly known artifact path fields."""
    if isinstance(value, Mapping):
        return _normalize_record_mapping(value, path=path, replacements=replacements)
    if isinstance(value, (list, tuple)):
        return _normalize_record_sequence(value, path=path, replacements=replacements)
    if isinstance(value, str):
        return _normalize_record_string(value, path=path, replacements=replacements)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Issue #127 record contains a non-finite number")
    return value


def _validate_record_mapping(value: Mapping[Any, Any], *, path: tuple[str, ...]) -> None:
    for child_key, child_value in value.items():
        if not isinstance(child_key, str):
            raise TypeError("Issue #127 record has a non-string key")
        if _contains_embedded_host_path(child_key):
            raise ValueError(
                "Issue #127 record contains an unsafe path-shaped mapping key "
                f"in field {_record_field_path((*path, '<path-key>'))}"
            )
        child_path = (*path, child_key)
        if child_key.lower() in _FORBIDDEN_RECORD_KEYS:
            raise ValueError(
                f"Issue #127 record must not retain host identity field "
                f"{_record_field_path(child_path)}"
            )
        _validate_record_safety(child_value, path=child_path)


def _validate_record_sequence(
    value: list[Any] | tuple[Any, ...], *, path: tuple[str, ...]
) -> None:
    for index, item in enumerate(value):
        _validate_record_safety(item, path=(*path, f"[{index}]"))


def _validate_record_string(value: str, *, path: tuple[str, ...]) -> None:
    key = path[-1] if path else None
    if key == DEVICE_NODE_KEY and _DEVICE_NODE_RE.fullmatch(value):
        return
    if _contains_embedded_host_path(value):
        raise ValueError(
            f"Issue #127 record contains an embedded host path in field "
            f"{_record_field_path(path)}"
        )
    if _is_whole_absolute_path(value):
        raise ValueError(
            f"Issue #127 record contains an unsafe absolute path in field "
            f"{_record_field_path(path)}"
        )


def _validate_record_safety(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        _validate_record_mapping(value, path=path)
    elif isinstance(value, (list, tuple)):
        _validate_record_sequence(value, path=path)
    elif isinstance(value, str):
        _validate_record_string(value, path=path)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Issue #127 record contains a non-finite number")


def _serialize_record(payload: Mapping[str, Any]) -> str:
    replacements: list[dict[str, str]] = []
    normalized = _normalize_record_paths(payload, replacements=replacements)
    if not isinstance(normalized, dict):
        raise TypeError("Issue #127 record must be an object")
    if replacements:
        normalized["record_safety"] = {
            "path_policy": (
                "shared host-path classifier; known artifact fields retain safe basenames"
            ),
            "path_replacements": replacements,
        }
    _validate_record_safety(normalized)
    serialized = strict_json_dumps(normalized, indent=2)
    decoded = json.loads(serialized)
    _validate_record_safety(decoded)
    return serialized


def _public_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    public = dict(trace)
    public.pop("_last_timestamp_datetime", None)
    return public


def build_record(
    *,
    run: Mapping[str, Any],
    run_number: int,
    run_id: str,
    environment: Mapping[str, Any],
    cooldown: Mapping[str, Any],
    run_start_utc: str,
    run_end_utc: str,
    power_trace: Mapping[str, Any],
    result_path: Path | None = None,
) -> dict[str, Any]:
    """Build the dedicated record after all row facts and provenance exist."""
    rows = [
        row for row in run.get("comparison_rows", []) if isinstance(row, Mapping)
    ]
    if len(rows) == len(order_for_run(run_number)):
        validate_occurrence_plan(rows, run_number=run_number)
    elif run.get("status") == "failed":
        _validate_occurrence_prefix(rows, run_number=run_number)
    else:
        raise ValueError("a passing Issue #127 record must contain all eight occurrences")
    sampler = dict(SAMPLER_DECISION)
    gaps = power_trace.get("timestamp_gap_facts", {}).get("observed_timestamp_gap_seconds", [])
    validate_sampler_decision(sampler, trace_timestamp_gaps=gaps)
    validate_harness_environment(environment, run_id=run_id)
    validate_cooldown_metadata(cooldown, environment=environment)
    rows = [dict(row) for row in run["comparison_rows"]]
    first_row = rows[0] if rows else {}
    gate = verify_warmup_hold_gate(run["warmup"], first_row, power_trace=power_trace)
    close_failed = run.get("device_close_failed") is True or (
        isinstance(run.get("cleanup"), Mapping)
        and run["cleanup"].get("close_succeeded") is False
    )
    effective_timeout = float(run.get("run_timeout_seconds", RUN_TIMEOUT_SECONDS))
    timeout_policy = dict(
        run.get("run_timeout_policy") or resolve_run_timeout(run_number)[1]
    )
    under_timeout = run.get("run_under_timeout")
    if under_timeout is None and _same_float(
        effective_timeout, DEFAULT_RUN_TIMEOUT_SECONDS
    ):
        under_timeout = run.get("run_under_60_seconds")
    run_status = (
        run.get("status") == "pass"
        and all(row.get("status") == "ok" for row in rows)
        and all(row.get("adjusted_p50_tflops") is not None for row in rows)
        and gate["verified"]
        and power_trace.get("coverage_complete") is True
        and under_timeout is True
        and not close_failed
        and (run_number != 2 or cooldown.get("gate_verified") is True)
    )
    trace = _public_trace(power_trace)
    failure = run.get("failure")
    if failure is None and run_number == 2 and cooldown.get("gate_verified") is not True:
        failure = {
            "type": "cooldown_gate_failure",
            "message": "run 2 power/temperature cooldown gate did not pass",
            "measurement_started": True,
        }
    record = {
        "record_schema": ISSUE127_RECORD_SCHEMA,
        "status": "pass" if run_status else "failed",
        "issue": "#127",
        "runner": ISSUE127_RUNNER,
        "run_id": run_id,
        "run_number": run_number,
        "failure": failure,
        "device_close_failed": close_failed,
        "captured_at": run_start_utc,
        "run_start": run_start_utc,
        "run_end": run_end_utc,
        "environment": dict(environment),
        "algorithm": {
            "batch": BATCH,
            "iterations": FIXED_ITERATIONS,
            "initial_value": INITIAL_VALUE,
            "condition_number": CONDITION_NUMBER,
            "state_format": "BF16",
            "destination_format": "FP32",
            "math_fidelity": "HiFi3",
            "fuse_s": True,
            "matrix_block": 8,
            "double_buffer": True,
            "dst_full_sync_en": True,
            "input_memory": "l1",
            "x0_memory": "l1",
            "output_memory": "dram",
        },
        "input_generation": run.get("input_generation"),
        "order": {
            "run_number": run_number,
            "names": list(order_for_run(run_number)),
            "occurrence_count_per_configuration": 2,
            "occurrence_identity": "row + occurrence + order_index",
        },
        "cooldown": dict(cooldown),
        "warmup": run.get("warmup"),
        "warmup_hold_gate": gate,
        "comparison_rows": rows,
        "comparison_summary": _aggregate_configuration_rows(rows),
        "run_timing": {
            "scope": "device measurement interval only; wrapper cooldown is outside this interval",
            "run_start_monotonic": run.get("run_start_monotonic"),
            "run_end_monotonic": run.get("run_end_monotonic"),
            "run_duration_seconds": run.get("run_duration_seconds"),
            "run_timeout_seconds": effective_timeout,
            "under_timeout": under_timeout,
        },
        "run_timeout_policy": timeout_policy,
        "run_budget": run.get("run_budget"),
        "sampler_decision": sampler,
        "power_trace": trace,
        "power_trace_provenance": {
            "source": trace.get("file"),
            "source_file": trace.get("source_file"),
            "sha256": trace.get("sha256"),
            "sha256_definition": trace.get("sha256_definition"),
            "requested_sampler_interval_seconds": SAMPLER_INTERVAL_SECONDS,
            "coverage_definition": trace.get("coverage_definition"),
            "coverage": trace.get("coverage"),
            "immutable_snapshot": trace.get("immutable_snapshot"),
            "timestamp_gap_facts": trace.get("timestamp_gap_facts"),
        },
        "adjusted_p50_note": (
            "adjusted_p50_tflops = raw_p50_tflops * 1350 / row_time_weighted_aiclk_mhz. "
            "It is supplementary and assumes processing capability scales linearly with "
            "AICLK; the primary decision uses raw P50 order-swap comparisons."
        ),
        "decision_policy": {
            "raw_delta_definition": "100 * (FP32-R raw P50 / BF16-R raw P50 - 1)",
            "r_format_thresholds_percent": R_FORMAT_DELTA_THRESHOLDS_PERCENT,
            "clock_half_thresholds_percent": CLOCK_DELTA_HALF_THRESHOLDS_PERCENT,
            "position_effect_thresholds_percent": POSITION_EFFECT_THRESHOLDS_PERCENT,
            "adjusted_p50_is_primary": False,
            "conclusion_statuses": ["r_format", "clock", "inconclusive"],
        },
        "raw_artifact": None,
    }
    if result_path is not None:
        record["result_file"] = result_path.name
    return record


def _raw_path(result_path: Path, run_id: str) -> Path:
    return result_path.parent / f"issue127-abba-raw-{validate_run_id(run_id)}{JSON_SUFFIX}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-number", type=int, choices=(1, 2), required=True)
    parser.add_argument("--run-timeout-seconds", type=float, default=None)
    parser.add_argument("--run-scope", choices=tuple(APPROVED_RUN_TARGETS), default=None)
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--cooldown-wait-seconds", type=float, default=None)
    parser.add_argument("--cooldown-start-utc", default=None)
    parser.add_argument("--cooldown-end-utc", default=None)
    parser.add_argument("--expected-aiclk-mhz", type=float, default=None)
    parser.add_argument("--cooldown-baseline-source", default=COOLDOWN_BASELINE_SOURCE)
    parser.add_argument("--cooldown-baseline-power-w", type=float, default=COOLDOWN_BASELINE_POWER_W)
    parser.add_argument(
        "--cooldown-baseline-temperature-c",
        type=float,
        default=COOLDOWN_BASELINE_ASIC_TEMPERATURE_C,
    )
    parser.add_argument(
        "--cooldown-power-tolerance-w",
        type=float,
        default=COOLDOWN_POWER_TOLERANCE_W,
    )
    parser.add_argument(
        "--cooldown-temperature-tolerance-c",
        type=float,
        default=COOLDOWN_TEMPERATURE_TOLERANCE_C,
    )
    parser.add_argument("--cooldown-observed-power-w", type=float, default=None)
    parser.add_argument("--cooldown-observed-temperature-c", type=float, default=None)
    parser.add_argument("--cooldown-observed-aiclk-mhz", type=float, default=None)
    parser.add_argument("--cooldown-gate-result", choices=("pass", "fail"), default=None)
    parser.add_argument(
        "--cooldown-gate-verified",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--env-json", type=Path, default=None)
    parser.add_argument("--power-trace", type=Path, default=None)
    parser.add_argument("--kind", action="append", default=None)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--input-memory", choices=("l1", "dram"), default="l1")
    parser.add_argument("--output-memory", choices=("l1", "dram"), default="dram")
    parser.add_argument("--matrix-block", type=int, default=8)
    parser.add_argument("--custom-math-fidelity", default="HiFi3")
    parser.add_argument("--fuse-s", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--double-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dst-full-sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp32-dest-acc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x0-memory", choices=("l1", "dram"), default="l1")
    return parser


def _validate_fixed_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.device_id != DEVICE_ID:
        parser.error("Issue #127 must run on visible device_id=0 inside the container")
    if args.kind and any(kind != "custom_newton_schulz" for kind in args.kind):
        parser.error("Issue #127 accepts only custom_newton_schulz")
    if args.iters != 1:
        parser.error("Issue #127 counts one fixed-N=12 kernel launch per sample")
    if args.input_memory != "l1" or args.x0_memory != "l1":
        parser.error("Issue #127 requires input and X0 in L1")
    if args.output_memory != "dram":
        parser.error("Issue #127 requires output in DRAM")
    if args.matrix_block != 8:
        parser.error("Issue #127 requires matrix_block=8")
    if args.custom_math_fidelity != "HiFi3":
        parser.error("Issue #127 requires HiFi3")
    if not all(
        (args.fuse_s, args.double_buffer, args.dst_full_sync, args.fp32_dest_acc)
    ):
        parser.error(
            "Issue #127 requires fused S, double buffering, full DEST sync, and FP32 accumulation"
        )


def _validate_run2_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.cooldown_wait_seconds is not None and (
        not math.isfinite(args.cooldown_wait_seconds)
        or args.cooldown_wait_seconds <= 0.0
        or args.cooldown_wait_seconds > COOLDOWN_WAIT_CAP_SECONDS
    ):
        parser.error("run 2 cooldown wait must be positive and at most 600 seconds")
    if args.cooldown_start_utc is None or args.cooldown_end_utc is None:
        parser.error("run 2 requires wrapper-provided cooldown UTC bounds")
    if args.cooldown_gate_verified is False:
        parser.error("run 2 cannot start with an unverified cooldown gate")


def _validate_no_run2_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    values = (
        args.cooldown_wait_seconds,
        args.cooldown_start_utc,
        args.cooldown_end_utc,
        args.expected_aiclk_mhz,
        args.cooldown_observed_power_w,
        args.cooldown_observed_temperature_c,
        args.cooldown_observed_aiclk_mhz,
        args.cooldown_gate_result,
        args.cooldown_gate_verified,
    )
    if any(value is not None for value in values):
        parser.error("cooldown and observed-AICLK confirmation belong to run 2")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _validate_fixed_args(parser, args)
    if args.run_number == 2:
        _validate_run2_args(parser, args)
    else:
        _validate_no_run2_args(parser, args)


def main(
    argv: list[str] | None = None,
    *,
    utc_now_fn: Callable[[], datetime.datetime] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    read_text_fn: Callable[[Path], str] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        run_timeout_seconds, run_timeout_policy = resolve_run_timeout(
            args.run_number,
            args.run_timeout_seconds,
            args.run_scope,
        )
    except ValueError as exc:
        parser.error(str(exc))
    now_utc = (lambda: datetime.datetime.now(datetime.UTC)) if utc_now_fn is None else utc_now_fn
    monotonic = time.monotonic if monotonic_fn is None else monotonic_fn
    sleep = time.sleep if sleep_fn is None else sleep_fn
    try:
        run_id = current_run_id()
        result_path = _resolve_result_path(args)
        output_dir = result_path.parent
        environment_path = args.env_json or (
            output_dir / run_artifact_name(prefix="env-", suffix=JSON_SUFFIX, run_id=run_id)
        )
        environment = _read_environment(
            environment_path,
            run_id=run_id,
            output_dir=output_dir,
            read_text_fn=read_text_fn,
        )
        validate_harness_environment(environment, run_id=run_id)
        cooldown_start_utc = args.cooldown_start_utc if args.run_number == 2 else None
        cooldown_end_utc = args.cooldown_end_utc if args.run_number == 2 else None
        cooldown = build_cooldown_metadata(
            run_number=args.run_number,
            wait_seconds=args.cooldown_wait_seconds,
            expected_aiclk_mhz=args.expected_aiclk_mhz,
            environment=environment,
            start_utc=cooldown_start_utc,
            end_utc=cooldown_end_utc,
            wait_performed=args.run_number == 2,
            baseline_source=args.cooldown_baseline_source,
            baseline_power_w=args.cooldown_baseline_power_w,
            baseline_temperature_c=args.cooldown_baseline_temperature_c,
            power_tolerance_w=args.cooldown_power_tolerance_w,
            temperature_tolerance_c=args.cooldown_temperature_tolerance_c,
            observed_power_w=args.cooldown_observed_power_w,
            observed_temperature_c=args.cooldown_observed_temperature_c,
            observed_idle_aiclk_mhz=args.cooldown_observed_aiclk_mhz,
            gate_result=args.cooldown_gate_result,
            gate_verified=args.cooldown_gate_verified,
        )
        measurement_start_utc = _timestamp(now_utc(), field="run_start")
        measurement_start_monotonic = monotonic()
        device = None
        run: dict[str, Any]
        try:
            import ttnn

            device = ttnn.open_device(device_id=DEVICE_ID)
            run = run_comparison(
                ttnn,
                device,
                run_number=args.run_number,
                run_id=run_id,
                run_timeout_seconds=run_timeout_seconds,
                run_timeout_policy=run_timeout_policy,
                time_fn=monotonic,
                utc_now_fn=now_utc,
            )
        except Exception as exc:  # noqa: BLE001 - write a failure record when possible
            run = {
                "status": "failed",
                "run_number": args.run_number,
                "run_id": run_id,
                "order": list(order_for_run(args.run_number)),
                "occurrences_requested": 8,
                "occurrences_completed": 0,
                "stopped_on_failure": True,
                "run_start_monotonic": monotonic(),
                "run_end_monotonic": monotonic(),
                "run_duration_seconds": 0.0,
                "run_timeout_seconds": run_timeout_seconds,
                "run_under_timeout": False,
                "run_timeout_policy": run_timeout_policy,
                "warmup": {
                    "launches_requested": WARMUP_LAUNCHES,
                    "launches_completed": 0,
                    "hold_gate_verified": False,
                    "start_utc": measurement_start_utc,
                    "end_utc": measurement_start_utc,
                },
                "comparison_rows": [],
                "input_generation": None,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        finally:
            if device is not None:
                try:
                    import ttnn

                    ttnn.close_device(device)
                except Exception:  # noqa: BLE001 - preserve the completed run record
                    run["device_close_failed"] = True
        measurement_end_monotonic = monotonic()
        run["run_start_monotonic"] = measurement_start_monotonic
        run["run_end_monotonic"] = measurement_end_monotonic
        run["run_duration_seconds"] = measurement_end_monotonic - measurement_start_monotonic
        run["run_under_timeout"] = (
            run["run_duration_seconds"] <= run_timeout_seconds
        )
        run["run_timeout_seconds"] = run_timeout_seconds
        run["run_timeout_policy"] = run_timeout_policy
        measurement_end_utc = _timestamp(now_utc(), field="run_end")
        power_path = args.power_trace or (
            output_dir / run_artifact_name(prefix="power-", suffix=".csv", run_id=run_id)
        )
        power_trace = _capture_final_power_trace(
            output_dir,
            run_id=run_id,
            run_start=measurement_start_utc,
            run_end=measurement_end_utc,
            power_path=power_path,
            monotonic_fn=monotonic,
            sleep_fn=sleep,
        )
        cooldown = verify_cooldown_power_trace(cooldown, power_trace=power_trace)
        if run.get("comparison_rows"):
            apply_final_trace_facts(run["comparison_rows"], power_trace=power_trace)
            verify_warmup_hold_gate(
                run["warmup"], run["comparison_rows"][0], power_trace=power_trace
            )
        record = build_record(
            run=run,
            run_number=args.run_number,
            run_id=run_id,
            environment=environment,
            cooldown=cooldown,
            run_start_utc=measurement_start_utc,
            run_end_utc=measurement_end_utc,
            power_trace=power_trace,
            result_path=result_path,
        )
        raw_path = _raw_path(result_path, run_id)
        raw_payload = {
            "record_schema": ISSUE127_RAW_SCHEMA,
            "issue": "#127",
            "run_id": run_id,
            "runner": ISSUE127_RUNNER,
            "environment": dict(environment),
            "cooldown": dict(cooldown),
            "run": run,
            "power_trace": _public_trace(power_trace),
        }
        _atomic_json_write(raw_path, raw_payload, output_dir=output_dir)
        record["raw_artifact"] = {
            "file": raw_path.name,
            "sha256": _file_sha256(raw_path),
            "sha256_definition": "SHA-256 of the complete Issue #127 raw JSON bytes",
        }
        _atomic_json_write(result_path, record, output_dir=output_dir)
        print(f"issue127 result -> {result_path}", flush=True)
        print(f"issue127 raw -> {raw_path.name}", flush=True)
        return 0 if record["status"] == "pass" else 1
    except Exception as exc:  # noqa: BLE001 - command-line contract reports failure
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
