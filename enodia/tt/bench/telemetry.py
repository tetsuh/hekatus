"""Board telemetry and environment capture.

Kept in Python rather than embedded in the wrapper script, because the first
run produced a power trace containing nothing but its header: the sampler
had been written inline and its quoting was wrong, and nothing tested it.
Parsing lives here so it can fail in a test instead of in a measurement.

The power trace is not incidental. Whether the board enforces the limit its
firmware reports is an open question (#28), and the answer decides whether a
throughput figure measured on one board is a lower bound for another. Under
sustained load, this trace is the answer.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench import power_trace_extended as _power_trace_extended
from enodia.tt.bench.board_identity import (
    normalize_board_id,
    normalize_pci_bus_id,
    resolve_board_identity,
)
from enodia.tt.bench.sampler_contract import (
    COVERAGE_DEFINITIONS,
    DEFAULT_SAMPLER_INTERVAL_SECONDS,
    EXPLICIT_FINAL_SAMPLE,
    ONE_INTERVAL_BOUND,
    POWER_TRACE_COMPLETION_MARKER_PREFIX,
    SAMPLER_MODES,
    normalize_sampler_metadata,
    power_trace_completion_marker_path,
)

EXTENDED_POWER_TRACE_COLUMNS = _power_trace_extended.EXTENDED_POWER_TRACE_COLUMNS
EXTENDED_POWER_TRACE_HEADER = _power_trace_extended.EXTENDED_POWER_TRACE_HEADER
EXTENDED_POWER_TRACE_SCHEMA = _power_trace_extended.EXTENDED_POWER_TRACE_SCHEMA
EXTENDED_POWER_TRACE_SCHEMA_VERSION = _power_trace_extended.EXTENDED_POWER_TRACE_SCHEMA_VERSION
ExtendedPowerTraceError = _power_trace_extended.ExtendedPowerTraceError
extended_csv_row = _power_trace_extended.extended_csv_row
extended_power_trace_header = _power_trace_extended.extended_power_trace_header
extended_telemetry_csv_row = _power_trace_extended.extended_telemetry_csv_row
format_extended_power_trace_row = _power_trace_extended.format_extended_power_trace_row
parse_extended_power_trace = _power_trace_extended.parse_extended_power_trace
parse_extended_snapshot = _power_trace_extended.parse_extended_snapshot
parse_extended_telemetry = _power_trace_extended.parse_extended_telemetry
parse_power_trace_extended = _power_trace_extended.parse_power_trace_extended
read_extended_power_trace = _power_trace_extended.read_extended_power_trace
read_extended_power_trace_csv = _power_trace_extended.read_extended_power_trace_csv
read_extended_power_trace_snapshot = _power_trace_extended.read_extended_power_trace_snapshot
read_power_trace_extended_csv = _power_trace_extended.read_power_trace_extended_csv
select_extended_board = _power_trace_extended.select_extended_board
is_extended_snapshot_well_formed = _power_trace_extended.is_extended_snapshot_well_formed
validate_extended_power_trace = _power_trace_extended.validate_extended_power_trace

SNAPSHOT_COMMAND = ("tt-smi", "-s", "--snapshot_no_tty")
CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c"
_DEFAULT_DEVICE_NODE = "/dev/tenstorrent/0"
_EXTENDED_NODE_RESOLUTION_RETRIES = 2
_PCI_BUS_ID_RE = re.compile(
    r"(?<![0-9a-f])([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])(?![0-9a-f])",
    re.IGNORECASE,
)
_BOARD_BUS_ID_KEYS = ("bus_id", "pci_bus_id", "pci_address", "pci_bdf")
_SMBUS_LIMIT_KEYS = (
    "VDD_LIMITS",
    "BOARD_POWER_LIMIT",
    "TDC_LIMIT_MAX",
    "THM_LIMIT_THROTTLE",
    "THM_LIMIT_SHUTDOWN",
    "AICLK_LIMIT_MAX",
    "TDP_LIMIT_MAX",
    "AICLK_ARB_MIN",
    "AICLK_ARB_MAX",
)
_BLACKHOLE_BY_ID_RE = re.compile(r"^blackhole-[0-9A-Za-z][0-9A-Za-z._-]*$")

POWER_TRACE_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")
_FULL_COMMIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_VALID_BOOLEAN_OVERRIDE_VALUES = frozenset({"0", "1", "false", "true"})


def _parse_trace_timestamp(value: Any, *, field: str) -> datetime.datetime:
    """Parse one timezone-bearing ISO-8601 trace timestamp."""
    if isinstance(value, datetime.datetime):
        try:
            has_timezone = value.tzinfo is not None and value.utcoffset() is not None
            parsed = value.astimezone(datetime.UTC) if has_timezone else None
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
        if parsed is None:
            raise ValueError(f"{field} must include a timezone")
        return parsed
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    try:
        has_timezone = parsed.tzinfo is not None and parsed.utcoffset() is not None
        normalized = parsed.astimezone(datetime.UTC) if has_timezone else None
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if normalized is None:
        raise ValueError(f"{field} must include a timezone")
    return normalized


def _finite_integer(
    value: Any, *, field: str, positive: bool = False, maximum: int = (1 << 63) - 1
) -> int | None:
    """Return a bounded integer conversion, or ``None`` for unusable input."""
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(converted) or converted > maximum or (
        positive and converted <= 0.0
    ):
        return None
    try:
        result = int(converted)
    except (OverflowError, TypeError, ValueError):
        return None
    if result < 0 or (positive and result <= 0):
        return None
    return result


def _trace_number(value: Any, *, field: str, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{field} must be {requirement}")
    return parsed


def _legacy_power_trace_error(kind: str, error: Exception | None) -> str:
    if kind == "power_trace_unreadable" and error is not None:
        return f"power trace is not readable: {type(error).__name__}"
    return str(error) if error is not None else kind


def _legacy_power_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "file",
        "columns",
        "sha256",
        "readable",
        "nonempty",
        "csv_row_count",
        "sample_count",
        "valid_row_count",
        "in_run_valid_row_count",
        "timestamps_parse",
        "timestamps_ordered",
        "first_timestamp",
        "last_timestamp",
        "run_start",
        "run_end",
        "covers_run_start",
        "covers_run_end",
        "coverage_complete",
        "coverage_definition",
        "sampler_interval_seconds",
        "aiclk_source",
        "aiclk_mhz",
        "aiclk_mhz_in_run",
    )
    public = {field: trace.get(field) for field in fields}
    coverage = trace.get("coverage")
    public["coverage"] = {
        field: coverage.get(field, False) if isinstance(coverage, Mapping) else False
        for field in (
            "definition",
            "nonempty",
            "timestamps_parse",
            "timestamps_ordered",
            "first_at_or_before_run_start",
            "last_at_or_after_run_end",
            "last_within_interval_bound",
            "readable",
            "complete",
        )
    }
    errors = list(trace.get("errors", []))
    if not trace.get("readable") and trace.get("error") is not None:
        errors = [trace["error"]]
    public["errors"] = errors
    return public


def parse_power_trace(
    path: Path,
    *,
    run_start: datetime.datetime | str | None = None,
    run_end: datetime.datetime | str | None = None,
    coverage_definition: str = EXPLICIT_FINAL_SAMPLE,
    sampler_interval_seconds: float | None = None,
    load_interval_start: datetime.datetime | str | None = None,
    load_interval_end: datetime.datetime | str | None = None,
    load_interval_source: str | None = None,
) -> dict[str, Any]:
    """Parse a power CSV through the shared snapshot/read lifecycle."""
    if not isinstance(path, Path):
        path = Path(path)
    if coverage_definition not in COVERAGE_DEFINITIONS:
        raise ValueError("coverage_definition is not supported")
    if coverage_definition == ONE_INTERVAL_BOUND:
        try:
            sampler_interval_seconds = float(sampler_interval_seconds)
        except (TypeError, ValueError, OverflowError):
            sampler_interval_seconds = float("nan")
        if not math.isfinite(sampler_interval_seconds) or sampler_interval_seconds <= 0.0:
            raise ValueError(
                "one_interval_bound requires a finite positive sampler interval"
            )
    from enodia.tt.bench.power_trace_snapshot import (
        PowerTraceReadOptions,
        read_power_trace_csv,
    )

    trace = read_power_trace_csv(
        path,
        run_start=run_start,
        run_end=run_end,
        sampler_interval_seconds=sampler_interval_seconds,
        coverage_definition=coverage_definition,
        load_interval_start=load_interval_start,
        load_interval_end=load_interval_end,
        load_interval_source=load_interval_source,
        require_all_rows_valid=True,
        error_factory=_legacy_power_trace_error,
        reject_incomplete_rows=False,
        read_options=PowerTraceReadOptions(
            row_label="power row",
            tolerate_invalid_run_bounds=True,
        ),
    )
    return _legacy_power_trace(trace)


def validate_power_trace_coverage(
    path: Path,
    *,
    run_start: datetime.datetime | str | None = None,
    run_end: datetime.datetime | str | None = None,
    coverage_definition: str = EXPLICIT_FINAL_SAMPLE,
    sampler_interval_seconds: float | None = None,
    load_interval_start: datetime.datetime | str | None = None,
    load_interval_end: datetime.datetime | str | None = None,
    load_interval_source: str | None = None,
) -> dict[str, Any]:
    """Board-free alias for :func:`parse_power_trace` used by record builders."""
    return parse_power_trace(
        path,
        run_start=run_start,
        run_end=run_end,
        coverage_definition=coverage_definition,
        sampler_interval_seconds=sampler_interval_seconds,
        load_interval_start=load_interval_start,
        load_interval_end=load_interval_end,
        load_interval_source=load_interval_source,
    )


# Keep both names discoverable for callers that describe the seam as parsing
# or validation; they intentionally share one implementation and one catalog.
read_power_trace = parse_power_trace
validate_power_trace = validate_power_trace_coverage


def _requested_device_node() -> str:
    return os.environ.get("HEKATUS_TT_DEVICE_NODE") or _DEFAULT_DEVICE_NODE


def _normalize_pci_bus_id(value: object) -> str | None:
    return normalize_pci_bus_id(value)


def _normalize_pci_bus_ids(values: object) -> set[str]:
    """Normalize one injected PCI identity and reject malformed collections."""
    if isinstance(values, str):
        values = (values,)
    elif isinstance(values, (list, tuple, set, frozenset)):
        values = tuple(values)
    else:
        return set()
    normalized: set[str] = set()
    for value in values:
        bus_id = _normalize_pci_bus_id(value)
        if bus_id is None:
            return set()
        normalized.add(bus_id)
    return normalized


def _nested_text_values(device: dict, keys: tuple[str, ...]) -> set[str]:
    """Return scalar identity values from a device and its board metadata."""
    values: set[str] = set()
    sources = [device]
    board_info = device.get("board_info")
    if isinstance(board_info, dict):
        sources.append(board_info)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                values.add(str(value).strip())
    return {value for value in values if value}


def _device_bus_identity(device: dict) -> set[str] | None:
    """Return all valid PCI ids, or None when an identity field is malformed."""
    values = _nested_text_values(device, _BOARD_BUS_ID_KEYS)
    identity = {_normalize_pci_bus_id(value) for value in values}
    if None in identity:
        return None
    return {value for value in identity if value is not None}


def _udevadm_info(path: Path) -> str | None:
    """Return udev's identity text for a device node, or None on failure."""
    try:
        completed = subprocess.run(
            ("udevadm", "info", "--query=all", f"--name={path}"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _udev_devpaths(info: str) -> tuple[str, ...]:
    """Extract the sysfs paths carrying the device identity from udev text."""
    paths: list[str] = []
    for line in info.splitlines():
        line = line.strip()
        if line.startswith("E: DEVPATH="):
            paths.append(line.removeprefix("E: DEVPATH="))
        elif line.startswith("P: "):
            paths.append(line.removeprefix("P: "))
    return tuple(paths)


def _udev_pci_bus_ids(info: str | None) -> set[str]:
    """Derive the nearest PCI BDF from udev's sysfs identity.

    A PCI path can include parent bridges.  The last BDF in each DEVPATH is
    the PCI ancestor of the accelerator, while independent udev identities
    must agree.  No numeric device-node name is considered here.
    """
    if not isinstance(info, str) or not info.strip():
        return set()

    fields: dict[str, list[str]] = {}
    for line in info.splitlines():
        line = line.strip()
        if line.startswith("E: ") and "=" in line[3:]:
            key, value = line[3:].split("=", 1)
            fields.setdefault(key, []).append(value)

    devpaths = _udev_devpaths(info)
    path_bus_ids: list[str] = []
    for devpath in devpaths:
        if not devpath.startswith("/devices/"):
            return set()
        matches = tuple(_normalize_pci_bus_id(value) for value in _PCI_BUS_ID_RE.findall(devpath))
        if not matches or any(value is None for value in matches):
            return set()
        path_bus_ids.append(matches[-1])
    if path_bus_ids and len(set(path_bus_ids)) != 1:
        return set()

    id_path_bus_ids: set[str] = set()
    for id_path in fields.get("ID_PATH", []):
        matches = _PCI_BUS_ID_RE.findall(id_path)
        if matches:
            bus_id = _normalize_pci_bus_id(matches[-1])
            if bus_id is None:
                return set()
            id_path_bus_ids.add(bus_id)
    if len(id_path_bus_ids) > 1:
        return set()
    id_path_bus_id = next(iter(id_path_bus_ids), None)
    if id_path_bus_id is not None and path_bus_ids and id_path_bus_id != path_bus_ids[0]:
        return set()

    slot_values = fields.get("PCI_SLOT_NAME", [])
    slot_bus_ids = {_normalize_pci_bus_id(value) for value in slot_values}
    if None in slot_bus_ids or len(slot_bus_ids) > 1:
        return set()
    slot_bus_id = next(iter(slot_bus_ids), None)
    identity_bus_ids = {
        bus_id
        for bus_id in (path_bus_ids[0] if path_bus_ids else None, id_path_bus_id, slot_bus_id)
        if bus_id is not None
    }
    return identity_bus_ids if len(identity_bus_ids) == 1 else set()


def _device_pci_bus_ids(path: Path) -> set[str]:
    """Resolve a node's PCI identity through udev's sysfs-backed metadata."""
    return _udev_pci_bus_ids(_udevadm_info(path))


def _by_id_pci_bus_id(path: Path) -> str | None:
    """Read an optional PCI identity encoded in a ``by-id/pci-*`` name."""
    if not path.name.startswith("pci-"):
        return None
    return _normalize_pci_bus_id(path.name.removeprefix("pci-"))


def _by_id_name_is_valid(path: Path) -> bool:
    """Accept the driver names while requiring a real by-id naming shape."""
    return path.name.startswith("pci-") or _BLACKHOLE_BY_ID_RE.fullmatch(path.name) is not None


def _resolve_device_node_to_pci(
    node: str,
    *,
    identity_resolver: Callable[[Path], object] | None = None,
) -> set[str]:
    """Resolve a real node or by-id link through verified PCI identity.

    Numeric names classify a direct node but never identify its board.  Both a
    direct numeric node and a by-id symlink are accepted only when their
    resolved target has exactly one PCI BDF in udev's sysfs identity.  A
    ``pci-*`` label is an additional check, never the sole source of identity.
    """
    path = Path(node)
    if not path.is_absolute():
        return set()

    if path.parent.name == "by-id":
        if path.parent.parent.name != "tenstorrent":
            return set()
        if not path.is_symlink() or not _by_id_name_is_valid(path):
            return set()
        try:
            device_dir = path.parent.parent.resolve(strict=True)
            resolved_node = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return set()
        if resolved_node.parent != device_dir:
            return set()
        named_bus_id = _by_id_pci_bus_id(path)
        if path.name.startswith("pci-") and named_bus_id is None:
            return set()
    else:
        if path.parent.name != "tenstorrent" or not path.name.isdecimal():
            return set()
        try:
            device_dir = path.parent.resolve(strict=True)
            resolved_node = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return set()
        if resolved_node.parent != device_dir:
            return set()
        named_bus_id = None

    resolver = _device_pci_bus_ids if identity_resolver is None else identity_resolver
    try:
        resolved_bus_ids = _normalize_pci_bus_ids(resolver(resolved_node))
    except Exception:  # noqa: BLE001 - unverifiable identity must be rejected
        return set()
    if len(resolved_bus_ids) != 1:
        return set()
    resolved_bus_id = next(iter(resolved_bus_ids))
    if named_bus_id is not None and named_bus_id != resolved_bus_id:
        return set()
    return {resolved_bus_id}


def _resolved_pci_bus_ids(node: str, resolver: Callable[[str], object]) -> set[str]:
    """Normalize an injected resolver result, rejecting malformed identities."""
    try:
        return _normalize_pci_bus_ids(resolver(node))
    except Exception:  # noqa: BLE001 - an unverifiable node must be rejected
        return set()


def _snapshot_devices(snapshot: str) -> list[dict] | None:
    """Decode and validate the device list shared by both selection contracts."""
    try:
        root = json.loads(snapshot)
        devices = root["device_info"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(devices, list) or not devices:
        return None
    if not all(isinstance(device, dict) for device in devices):
        return None
    return devices


def _requested_q2_identity(
    board_id: str | None, bus_id: str | None
) -> tuple[str, str] | None:
    """Normalize an explicitly supplied Q2 identity, or retain legacy selection."""
    if board_id is None and bus_id is None:
        return None
    normalized_board_id = normalize_board_id(board_id)
    if normalized_board_id is None:
        raise ValueError("board_id must be a non-empty string")
    normalized_bus_id = _normalize_pci_bus_id(bus_id)
    if normalized_bus_id is None:
        raise ValueError("bus_id must be a valid PCI BDF")
    return normalized_board_id, normalized_bus_id


def _q2_node_matches(
    requested_node: str, requested_bus_id: str, resolver: Callable[[str], object]
) -> bool:
    requested_bus_ids = _resolved_pci_bus_ids(requested_node, resolver)
    return len(requested_bus_ids) == 1 and next(iter(requested_bus_ids)) == requested_bus_id


def _q2_board_matches(
    devices: list[dict], requested_board_id: str, requested_bus_id: str
) -> tuple[list[dict], list[dict], list[dict]]:
    board_matches: list[dict] = []
    bus_matches: list[dict] = []
    pair_matches: list[dict] = []
    for device in devices:
        board_info = device.get("board_info")
        if not isinstance(board_info, dict):
            continue
        candidate_board_id = normalize_board_id(board_info.get("board_id"))
        candidate_bus_id = _normalize_pci_bus_id(board_info.get("bus_id"))
        board_matches_identity = candidate_board_id == requested_board_id
        bus_matches_identity = candidate_bus_id == requested_bus_id
        if board_matches_identity:
            board_matches.append(device)
        if bus_matches_identity:
            bus_matches.append(device)
        if board_matches_identity and bus_matches_identity:
            pair_matches.append(device)
    return board_matches, bus_matches, pair_matches


def _select_q2_device(
    devices: list[dict],
    *,
    requested_node: str,
    resolver: Callable[[str], object],
    requested_board_id: str,
    requested_bus_id: str,
) -> dict | None:
    """Select one board after validating the explicit Q2 node identity."""
    if not _q2_node_matches(requested_node, requested_bus_id, resolver):
        return None
    board_matches, bus_matches, pair_matches = _q2_board_matches(
        devices, requested_board_id, requested_bus_id
    )
    if len(board_matches) != 1 or len(bus_matches) != 1 or len(pair_matches) != 1:
        return None
    if pair_matches[0] is not board_matches[0] or pair_matches[0] is not bus_matches[0]:
        return None
    return pair_matches[0]


def _select_legacy_device(
    devices: list[dict], *, requested_node: str, resolver: Callable[[str], object]
) -> dict | None:
    """Preserve legacy default-node and verified PCI selection behavior."""
    if requested_node == _DEFAULT_DEVICE_NODE:
        return devices[0]
    requested_bus_ids = _resolved_pci_bus_ids(requested_node, resolver)
    if len(requested_bus_ids) != 1:
        return None
    requested_bus_id = next(iter(requested_bus_ids))
    matches: list[dict] = []
    for device in devices:
        identity = _device_bus_identity(device)
        if identity is None or requested_bus_id not in identity:
            continue
        if len(identity) != 1:
            return None
        matches.append(device)
    return matches[0] if len(matches) == 1 else None


def _device_info(
    snapshot: str,
    *,
    node_resolver: Callable[[str], object] | None = None,
    board_id: str | None = None,
    bus_id: str | None = None,
) -> dict | None:
    """Select a board while retaining the device-node/PCI contract.

    Legacy callers that do not supply a Q2 identity retain the historical
    default-node behavior.  Once either Q2 identity is requested, both
    ``board_info.board_id`` and ``board_info.bus_id`` are required and must
    identify exactly one entry.  The selected device node, including the
    default node, is resolved through its verified PCI identity and must agree
    with the requested bus.  The ``device_info`` array position is never used
    in the Q2 branch.
    """
    devices = _snapshot_devices(snapshot)
    if devices is None:
        return None

    # Board identity is opt-in here so the established four-column sampler
    # remains unchanged.  Identity-aware callers resolve environment values
    # before passing the complete pair; no-argument callers stay on legacy
    # default-node selection.
    requested_node = _requested_device_node()
    resolver = _resolve_device_node_to_pci if node_resolver is None else node_resolver
    try:
        identity = _requested_q2_identity(board_id, bus_id)
    except ValueError:
        return None
    if identity is not None:
        return _select_q2_device(
            devices,
            requested_node=requested_node,
            resolver=resolver,
            requested_board_id=identity[0],
            requested_bus_id=identity[1],
        )
    return _select_legacy_device(
        devices, requested_node=requested_node, resolver=resolver
    )


def parse_telemetry(
    snapshot: str,
    *,
    node_resolver: Callable[[str], object] | None = None,
    board_id: str | None = None,
    bus_id: str | None = None,
) -> dict[str, str] | None:
    """Extract one finite sampled reading from the requested board."""
    device = _device_info(
        snapshot,
        node_resolver=node_resolver,
        board_id=board_id,
        bus_id=bus_id,
    )
    if device is None:
        return None
    try:
        telemetry = device["telemetry"]
        reading = {
            "power_w": str(telemetry["power"]).strip(),
            "aiclk_mhz": str(telemetry["aiclk"]).strip(),
            "asic_temp_c": str(telemetry["asic_temperature"]).strip(),
        }
        _trace_number(reading["power_w"], field="telemetry power_w")
        _trace_number(reading["aiclk_mhz"], field="telemetry aiclk_mhz", positive=True)
        _trace_number(reading["asic_temp_c"], field="telemetry asic_temp_c")
    except (KeyError, TypeError, ValueError):
        return None
    return reading


def telemetry_csv_row(
    snapshot: str,
    *,
    timestamp: str,
    node_resolver: Callable[[str], object] | None = None,
    board_id: str | None = None,
    bus_id: str | None = None,
) -> str | None:
    """One finite CSV row from the requested board, or None if unusable."""
    reading = parse_telemetry(
        snapshot,
        node_resolver=node_resolver,
        board_id=board_id,
        bus_id=bus_id,
    )
    if reading is None:
        return None
    return f"{timestamp},{reading['power_w']},{reading['aiclk_mhz']},{reading['asic_temp_c']}"


def parse_environment(
    snapshot: str,
    *,
    node_resolver: Callable[[str], object] | None = None,
    board_id: str | None = None,
    bus_id: str | None = None,
) -> dict:
    """Return board provenance without retaining host snapshot sections.

    ``host_info`` and ``processes`` are deliberately not copied.  The
    ``host_sw_vers`` mapping is a tool-provided provenance value and is retained
    when present; SMBus telemetry and the sibling limits object are retained as
    raw objects so this layer does not assert a unit conversion.
    """
    try:
        snapshot_object = json.loads(snapshot)
    except (TypeError, ValueError):
        snapshot_object = None
    device = _device_info(
        snapshot,
        node_resolver=node_resolver,
        board_id=board_id,
        bus_id=bus_id,
    )
    if device is None:
        return {
            "board_snapshot_error": "no verifiable device information in the snapshot",
            "device_node": _requested_device_node(),
        }
    environment = {
        key: device[key] for key in ("board_info", "firmwares", "limits") if key in device
    } | {
        "board": device.get("board_info"),
        "firmware": device.get("firmwares"),
        "limits": device.get("limits"),
        "device_node": _requested_device_node(),
    }
    host_sw_vers = (
        snapshot_object.get("host_sw_vers") if isinstance(snapshot_object, dict) else None
    )
    if isinstance(host_sw_vers, Mapping):
        # This is a tool-provided mapping.  Retain that mapping as-is; host_info
        # and processes are intentionally not part of the environment record.
        environment["host_sw_vers"] = host_sw_vers

    smbus_telem = device.get("smbus_telem")
    if isinstance(smbus_telem, Mapping):
        # SMBus telemetry and limits are sibling objects in the snapshot. Keep
        # the raw SMBus object and an explicit raw limit subset without deriving
        # values or asserting units.
        environment["smbus_telem"] = smbus_telem
        smbus_limits = {
            key: smbus_telem[key]
            for key in _SMBUS_LIMIT_KEYS
            if key in smbus_telem
        }
        if smbus_limits:
            environment["smbus_telem_limits"] = smbus_limits

    reading = parse_telemetry(
        snapshot,
        node_resolver=node_resolver,
        board_id=board_id,
        bus_id=bus_id,
    )
    if reading is not None:
        observed = _finite_integer(
            reading["aiclk_mhz"], field="aiclk_mhz", positive=True
        )
        if observed is not None:
            environment["aiclk_mhz_observed"] = [observed]
    return environment


def sampler_metadata(mode: str, interval: float | None = None) -> dict[str, object]:
    """Validate and describe the wrapper's telemetry-sampler mode."""
    return normalize_sampler_metadata({"mode": mode, "interval_seconds": interval})


def _run(command: tuple[str, ...] | str) -> str:
    try:
        completed = subprocess.run(
            command,
            shell=isinstance(command, str),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return completed.stdout
    except Exception as exc:  # noqa: BLE001 - a missing tool must not stop a measurement
        return f"<unavailable: {exc}>"


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _git(repo: Path, *args: str) -> tuple[str, bool]:
    """Output of one git query, and whether it answered at all.

    `_run` returns whatever landed on standard output and discards the exit
    status, which is right for the environment block: a tool that is missing
    costs one field. It is wrong for provenance. `git rev-parse` outside a
    repository — an ownership check refusing the bind-mounted tree is the
    realistic case here — exits non-zero with nothing on stdout, and empty
    stdout is exactly what a clean `git status` produces. Read through `_run`,
    a harness nobody could identify would be recorded as an unnamed commit in
    a clean tree, which is the one thing this must never claim.
    """
    try:
        completed = subprocess.run(
            ("git", "-C", str(repo), *args),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - a missing git must not stop a measurement
        return f"git {' '.join(args)} could not run: {type(exc).__name__}: {exc}", False
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        return f"git {' '.join(args)} failed: {detail}", False
    return completed.stdout.strip(), True


def _remote_tracking_refs_containing(
    repo: Path, commit: str
) -> tuple[str, bool]:
    """Return remote-tracking refs that contain one exact harness revision."""
    return _git(repo, "branch", "--remotes", "--contains", commit)


def _canonical_commit_oid(value: str) -> str | None:
    """Return one canonical full OID, or ``None`` for a non-commit value."""
    if not isinstance(value, str) or not _FULL_COMMIT_OID_RE.fullmatch(value):
        return None
    return value


def _unknown_override_identity(identity: dict[str, Any], reason: str) -> dict:
    """Fail closed when an override cannot be tied to one immutable commit."""
    identity.update(
        harness_commit=None,
        harness_dirty=None,
        harness_pushed=None,
        harness_identity_error=reason,
    )
    return identity


def _override_identity_base(override_dirty: str) -> tuple[dict[str, Any], bool]:
    if override_dirty not in _VALID_BOOLEAN_OVERRIDE_VALUES:
        return (
            {
                "harness_commit": None,
                "harness_dirty": None,
                "harness_identity_error": (
                    "HEKATUS_TT_HARNESS_DIRTY must be one of 0, 1, false, true"
                ),
            },
            False,
        )
    return (
        {
            "harness_commit": None,
            "harness_dirty": override_dirty in {"1", "true"},
            "harness_identity_source": "HEKATUS_TT_HARNESS_COMMIT",
        },
        True,
    )


def _override_publication_error(publication_override: str | None) -> str | None:
    if publication_override is not None and publication_override not in _VALID_BOOLEAN_OVERRIDE_VALUES:
        return "HEKATUS_TT_HARNESS_PUSHED must be one of 0, 1, false, true"
    return None


def _resolve_override_commit(
    repo: Path, override_commit: str
) -> tuple[str | None, str, bool]:
    canonical_output, canonical_ok = _git(
        repo,
        "rev-parse",
        "--verify",
        f"{override_commit}^{{commit}}",
    )
    canonical_commit = _canonical_commit_oid(canonical_output) if canonical_ok else None
    return canonical_commit, canonical_output, canonical_ok


def _identity_without_git(
    identity: dict[str, Any],
    override_commit: str,
    canonical_output: str,
    publication_override: str | None,
) -> dict:
    publication_error = _override_publication_error(publication_override)
    if publication_error is not None:
        return _unknown_override_identity(identity, publication_error)
    if _canonical_commit_oid(override_commit) is None:
        return _unknown_override_identity(
            identity,
            "override commit cannot be host-verified as a full commit OID; "
            f"{canonical_output}",
        )
    if publication_override is None:
        return _unknown_override_identity(
            identity,
            "Git metadata unavailable; a canonical full commit OID requires "
            "HEKATUS_TT_HARNESS_PUSHED",
        )
    identity.update(
        harness_commit=override_commit,
        harness_pushed=publication_override in {"1", "true"},
        harness_publication_source="HEKATUS_TT_HARNESS_PUSHED",
    )
    return identity


def _publish_override_identity(
    repo: Path,
    identity: dict[str, Any],
    canonical_commit: str,
    publication_override: str | None,
) -> dict:
    publication_error = _override_publication_error(publication_override)
    if publication_error is not None:
        identity.update(harness_pushed=None, harness_identity_error=publication_error)
        return identity
    remote_refs, remote_refs_ok = _remote_tracking_refs_containing(
        repo, canonical_commit
    )
    if remote_refs_ok:
        identity.update(
            harness_pushed=bool(remote_refs),
            harness_publication_source="remote_tracking_ref",
        )
        return identity
    if publication_override is not None:
        identity.update(
            harness_pushed=publication_override in {"1", "true"},
            harness_publication_source="HEKATUS_TT_HARNESS_PUSHED",
        )
        return identity
    identity.update(harness_pushed=None, harness_identity_error=remote_refs)
    return identity


def _override_identity(repo: Path, override_commit: str) -> dict:
    override_dirty = os.environ.get("HEKATUS_TT_HARNESS_DIRTY", "")
    identity, valid = _override_identity_base(override_dirty)
    if not valid:
        return identity
    publication_override = os.environ.get("HEKATUS_TT_HARNESS_PUSHED")
    canonical_commit, canonical_output, canonical_ok = _resolve_override_commit(
        repo, override_commit
    )
    if not canonical_ok:
        return _identity_without_git(
            identity, override_commit, canonical_output, publication_override
        )
    if canonical_commit is None:
        return _unknown_override_identity(
            identity,
            "HEKATUS_TT_HARNESS_COMMIT did not resolve to a full commit OID: "
            f"{canonical_output}",
        )
    identity["harness_commit"] = canonical_commit
    return _publish_override_identity(
        repo, identity, canonical_commit, publication_override
    )


def _regular_identity(repo: Path) -> dict:
    commit, commit_ok = _git(repo, "rev-parse", "HEAD")
    # Tracked changes only. The toolchain writes its build output into the
    # tree on every run, so counting untracked files would make the flag true
    # always, which says nothing about the code that computed the result.
    status, status_ok = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if not (commit_ok and status_ok):
        return {
            "harness_commit": None,
            "harness_dirty": None,
            "harness_pushed": None,
            "harness_identity_error": commit if not commit_ok else status,
        }
    remote_refs, remote_refs_ok = _remote_tracking_refs_containing(repo, commit)
    if not remote_refs_ok:
        return {
            "harness_commit": commit,
            "harness_dirty": bool(status),
            "harness_pushed": None,
            "harness_identity_error": remote_refs,
        }
    return {
        "harness_commit": commit,
        "harness_dirty": bool(status),
        "harness_pushed": bool(remote_refs),
    }


def harness_identity() -> dict:
    """Which revision of this tree computed the result.

    The FLOP accounting behind every efficiency figure lives here, so a
    measurement that cannot name the harness that produced it cannot be
    reproduced or compared against a later one. A modified tree is recorded
    as modified rather than silently attributed to its last commit, and a
    tree git could not read at all is recorded as unknown rather than clean.
    A transferred source tree may deliberately omit ``.git``; the wrapper can
    then provide the source revision explicitly, but publication still needs
    a host-side remote-ref query or an explicit host-verified boolean. Override
    revisions are resolved to full object IDs before publication is checked.
    """
    repo = Path(__file__).resolve().parents[3]
    override_commit = os.environ.get("HEKATUS_TT_HARNESS_COMMIT")
    if override_commit:
        return _override_identity(repo, override_commit)
    return _regular_identity(repo)


def capture_environment(
    image: str,
    image_pinned: bool,
    *,
    sampler_mode: str = "default",
    sampler_interval: float | None = None,
    board_id: str | None = None,
    bus_id: str | None = None,
) -> dict:
    """Everything needed to name the environment a measurement came from."""
    effective_board_id, effective_bus_id = resolve_board_identity(board_id, bus_id)
    environment = {
        "captured_at": _now(),
        "image": image,
        "image_pinned": image_pinned,
        "telemetry_sampler": sampler_metadata(sampler_mode, sampler_interval),
        "kernel": _run("uname -sr").strip(),
        "kmd_version": _run(
            "modinfo tenstorrent 2>/dev/null | awk '/^version:/{print $2}'"
        ).strip(),
        "tt_env_active_release": _run(
            "tt-env status 2>/dev/null | awk '/Active release:/{print $3}'"
        ).strip(),
    }
    environment.update(harness_identity())
    environment.update(
        parse_environment(
            _run(SNAPSHOT_COMMAND),
            board_id=effective_board_id,
            bus_id=effective_bus_id,
        )
    )
    run_id = os.environ.get("HEKATUS_TT_RUN_ID")
    if run_id:
        # The outer wrapper creates this identity once and exports it to both
        # environment capture and the benchmark process.  Do not invent one
        # here: an unset value is useful to callers that only exercise this
        # parser module.
        environment["run_id"] = run_id
    return environment


def _sample_loop(handle, interval: float) -> None:
    """Sample until terminated, surviving a bad reading but not a bad writer.

    The sampler observes a run it must not disturb, so a reading that fails —
    an unresponsive tool, a snapshot that will not parse — costs one row and
    the loop continues.

    Failing to *record* is a different thing and is not survived. Retrying a
    write that cannot succeed would turn a broken run into a quietly short
    trace, which is precisely the failure this module was extracted to stop
    happening silently. The wrapper certifies the closed file only after each
    complete row has been written and the handle has been closed.
    """
    handle.write(CSV_HEADER + "\n")
    while True:
        try:
            row = telemetry_csv_row(_run(SNAPSHOT_COMMAND), timestamp=_now())
        except Exception as exc:  # noqa: BLE001 - one lost row beats a lost trace
            print(f"telemetry sample failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            row = None
        if row is not None:
            handle.write(row + "\n")
        time.sleep(interval)


def _required_extended_identity(
    board_id: str | None, bus_id: str | None
) -> tuple[str, str]:
    """Resolve the complete identity required by the extended sampler."""
    effective_board_id, effective_bus_id = resolve_board_identity(board_id, bus_id)
    if effective_board_id is None or effective_bus_id is None:
        raise ValueError("extended-v1 requires both board_id and bus_id")
    return effective_board_id, effective_bus_id


def _extended_device_info(
    snapshot: str,
    *,
    board_id: str,
    bus_id: str,
) -> dict | None:
    """Resolve the extended sampler's node identity with bounded transient retries."""
    resolved_bus_ids: set[str] | None = None

    def resolve_node(node: str) -> set[str]:
        nonlocal resolved_bus_ids
        resolved_bus_ids = _resolved_pci_bus_ids(node, _resolve_device_node_to_pci)
        return resolved_bus_ids

    for attempt in range(_EXTENDED_NODE_RESOLUTION_RETRIES + 1):
        resolved_bus_ids = None
        device = _device_info(
            snapshot,
            node_resolver=resolve_node,
            board_id=board_id,
            bus_id=bus_id,
        )
        if device is not None:
            return device
        # An empty result is an unverifiable udev identity and may be transient.
        # A non-empty result that does not match the requested bus is a mismatch
        # and must remain fail-closed without retrying it.
        if resolved_bus_ids:
            return None
        if attempt == _EXTENDED_NODE_RESOLUTION_RETRIES:
            return None
    return None


def _sample_extended_loop(
    handle,
    interval: float,
    *,
    board_id: str | None = None,
    bus_id: str | None = None,
    monotonic_ns: Callable[[], int] | None = None,
    utc_now: Callable[[], str] | None = None,
) -> None:
    """Sample the opt-in sidecar without changing the legacy sampler.

    Each row has one snapshot window.  The monotonic values bracket exactly
    one ``tt-smi`` call, while the UTC value is captured after that call just
    as the legacy sampler captures its row timestamp.
    """
    effective_board_id, effective_bus_id = _required_extended_identity(board_id, bus_id)
    clock = time.monotonic_ns if monotonic_ns is None else monotonic_ns
    now = _now if utc_now is None else utc_now
    handle.write(EXTENDED_POWER_TRACE_HEADER + "\n")
    while True:
        snapshot_start_ns = clock()
        snapshot = _run(SNAPSHOT_COMMAND)
        snapshot_end_ns = clock()
        snapshot_finished_utc = now()
        if not is_extended_snapshot_well_formed(snapshot):
            print(
                "extended telemetry sample dropped: malformed snapshot",
                file=sys.stderr,
            )
            time.sleep(interval)
            continue
        # The extended selector validates snapshot identities, while this
        # verified resolver also binds the requested board to the benchmark
        # process's actual device node.  A mismatch is a run-level failure,
        # not a dropped telemetry sample.
        if _extended_device_info(
            snapshot,
            board_id=effective_board_id,
            bus_id=effective_bus_id,
        ) is None:
            raise ExtendedPowerTraceError(
                "benchmark device node does not match requested board_id/bus_id"
            )
        _power_trace_extended.select_extended_board(
            snapshot,
            board_id=effective_board_id,
            bus_id=effective_bus_id,
        )
        try:
            row = extended_telemetry_csv_row(
                snapshot,
                board_id=effective_board_id,
                bus_id=effective_bus_id,
                timestamp_utc=snapshot_finished_utc,
                timestamp_monotonic_ns=snapshot_end_ns,
                snapshot_elapsed_ns=snapshot_end_ns - snapshot_start_ns,
            )
        except Exception as exc:  # noqa: BLE001 - one lost row beats a lost trace
            print(
                f"extended telemetry sample failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            row = None
        if row is not None:
            handle.write(row + "\n")
        time.sleep(interval)


def _trusted_output_roots() -> tuple[Path, ...]:
    """Return output roots established by the host harness or test runner."""
    roots = (
        Path.cwd(),
        Path(tempfile.gettempdir()),
        Path("/out"),
        Path(__file__).resolve().parents[3],
    )
    return tuple(root.resolve(strict=False) for root in roots)


def _trusted_output_path(path: Path) -> Path:
    """Resolve a CLI output path only when it stays in a trusted root."""
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=False)
        if candidate.is_symlink():
            raise ValueError("output path must not be a symbolic link")
        trusted = _trusted_output_roots()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("output path cannot be resolved safely") from exc
    if not any(resolved.is_relative_to(root) for root in trusted):
        raise ValueError("output path must be beneath a trusted output directory")
    return resolved


def _sample_extended_forever(
    out_path: Path,
    interval: float,
    *,
    board_id: str | None = None,
    bus_id: str | None = None,
) -> None:
    effective_board_id, effective_bus_id = _required_extended_identity(board_id, bus_id)
    safe_output_path = _trusted_output_path(out_path)
    with safe_output_path.open(
        "w", encoding="utf-8", newline="", buffering=1
    ) as handle:
        _sample_extended_loop(
            handle,
            interval,
            board_id=effective_board_id,
            bus_id=effective_bus_id,
        )


class _SamplerStopped(BaseException):
    """Signal that the wrapper intentionally stopped the live sampler."""


def _stop_sampler(_signum: int, _frame: object) -> None:
    raise _SamplerStopped


def _write_power_trace_completion_marker(path: Path) -> None:
    """Publish a hash marker only after the complete CSV has been closed."""
    marker = power_trace_completion_marker_path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            f"{POWER_TRACE_COMPLETION_MARKER_PREFIX}{digest}\n", encoding="ascii"
        )
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_forever(out_path: Path, interval: float) -> None:
    """Write rows until the wrapper stops us, then certify the closed CSV."""
    marker = power_trace_completion_marker_path(out_path)
    marker.unlink(missing_ok=True)
    # SIGUSR1 is reserved for the wrapper's successful benchmark completion.
    # SIGTERM keeps its abort behavior and must never certify the trace.
    previous_handler = signal.signal(signal.SIGUSR1, _stop_sampler)
    stopped_intentionally = False
    try:
        with out_path.open("w", buffering=1) as handle:
            try:
                _sample_loop(handle, interval)
            except _SamplerStopped:
                stopped_intentionally = True
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
    if stopped_intentionally:
        _write_power_trace_completion_marker(out_path)


def _telemetry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture board environment or sample telemetry")
    sub = parser.add_subparsers(dest="mode", required=True)

    capture = sub.add_parser("capture-env", help="write the environment as JSON and exit")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--image", required=True)
    capture.add_argument("--image-pinned", action="store_true")
    capture.add_argument("--sampler-mode", choices=SAMPLER_MODES, default="default")
    capture.add_argument("--sampler-interval", type=float, default=None)
    capture.add_argument("--board-id", default=None)
    capture.add_argument("--bus-id", default=None)

    sample = sub.add_parser("sample", help="append telemetry rows until terminated")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--interval", type=float, default=DEFAULT_SAMPLER_INTERVAL_SECONDS)
    sample.add_argument(
        "--trace-schema",
        choices=("legacy-v1", "extended-v1"),
        default="legacy-v1",
        help="select the opt-in extended sidecar schema without changing legacy-v1",
    )
    sample.add_argument("--board-id", default=None)
    sample.add_argument("--bus-id", default=None)
    return parser


def _validate_telemetry_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.mode == "sample" and not (math.isfinite(args.interval) and args.interval > 0):
        # Zero turns the loop into a busy wait on the snapshot tool; negative
        # and non-finite values reach sleep and end the sampler outright.
        parser.error(f"--interval must be positive and finite, got {args.interval}")
    if args.mode == "sample" and args.trace_schema == "extended-v1":
        try:
            args.board_id, args.bus_id = _required_extended_identity(
                args.board_id, args.bus_id
            )
            args.out = _trusted_output_path(args.out)
        except ValueError as exc:
            parser.error(str(exc))
    if (
        args.mode == "sample"
        and args.trace_schema == "legacy-v1"
        and (args.board_id is not None or args.bus_id is not None)
    ):
        parser.error("legacy-v1 does not accept --board-id or --bus-id")


def _capture_environment_cli(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    try:
        if (
            args.sampler_mode == "default"
            and args.sampler_interval is None
            and args.board_id is None
            and args.bus_id is None
        ):
            # Keep the two-argument call compatible with board-free callers
            # that replace capture_environment in tests.
            environment = capture_environment(args.image, args.image_pinned)
        else:
            environment = capture_environment(
                args.image,
                args.image_pinned,
                sampler_mode=args.sampler_mode,
                sampler_interval=args.sampler_interval,
                board_id=args.board_id,
                bus_id=args.bus_id,
            )
    except ValueError as exc:
        parser.error(str(exc))
    args.out.write_text(strict_json_dumps(environment, indent=2) + "\n")
    print(f"environment -> {args.out}")


def _sample_telemetry_cli(args: argparse.Namespace) -> None:
    try:
        if args.trace_schema == "extended-v1":
            _sample_extended_forever(
                args.out,
                args.interval,
                board_id=args.board_id,
                bus_id=args.bus_id,
            )
        else:
            _sample_forever(args.out, args.interval)
    except KeyboardInterrupt:
        # The wrapper terminates the sampler when the run finishes; that is
        # the normal way this ends, not a failure.
        pass


def main(argv: list[str] | None = None) -> None:
    parser = _telemetry_parser()
    args = parser.parse_args(argv)
    _validate_telemetry_arguments(parser, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "capture-env":
        _capture_environment_cli(parser, args)
        return
    _sample_telemetry_cli(args)


if __name__ == "__main__":
    main()
