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
import csv
import datetime
import hashlib
import io
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.sampler_contract import (
    COVERAGE_DEFINITIONS,
    DEFAULT_SAMPLER_INTERVAL_SECONDS,
    EXPLICIT_FINAL_SAMPLE,
    ONE_INTERVAL_BOUND,
    SAMPLER_MODES,
    normalize_sampler_metadata,
)

SNAPSHOT_COMMAND = ("tt-smi", "-s", "--snapshot_no_tty")
CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c"
_DEFAULT_DEVICE_NODE = "/dev/tenstorrent/0"
_PCI_BUS_ID_RE = re.compile(
    r"(?<![0-9a-f])([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])(?![0-9a-f])",
    re.IGNORECASE,
)
_BOARD_BUS_ID_KEYS = ("bus_id", "pci_bus_id", "pci_address", "pci_bdf")
_BLACKHOLE_BY_ID_RE = re.compile(r"^blackhole-[0-9A-Za-z][0-9A-Za-z._-]*$")

POWER_TRACE_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")


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


def _trace_timestamp_text(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _parse_trace_timestamp(value, field=field).isoformat()


def _trace_number(value: Any, *, field: str, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{field} must be {requirement}")
    return parsed


def _power_trace_base(
    path: Path,
    *,
    run_start: Any,
    run_end: Any,
    coverage_definition: str = EXPLICIT_FINAL_SAMPLE,
    sampler_interval_seconds: float | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        start_text = _trace_timestamp_text(run_start, field="run_start") if run_start is not None else None
    except ValueError as exc:
        start_text = None
        errors.append(str(exc))
    try:
        end_text = _trace_timestamp_text(run_end, field="run_end") if run_end is not None else None
    except ValueError as exc:
        end_text = None
        errors.append(str(exc))
    return {
        "file": path.name,
        "columns": list(POWER_TRACE_COLUMNS),
        "sha256": None,
        "readable": False,
        "nonempty": False,
        "csv_row_count": 0,
        "sample_count": 0,
        "valid_row_count": 0,
        "in_run_valid_row_count": 0,
        "timestamps_parse": False,
        "timestamps_ordered": False,
        "first_timestamp": None,
        "last_timestamp": None,
        "run_start": start_text,
        "run_end": end_text,
        "covers_run_start": False,
        "covers_run_end": False,
        "coverage_complete": False,
        "coverage_definition": coverage_definition,
        "sampler_interval_seconds": sampler_interval_seconds,
        "coverage": {
            "definition": coverage_definition,
            "nonempty": False,
            "timestamps_parse": False,
            "timestamps_ordered": False,
            "first_at_or_before_run_start": False,
            "last_at_or_after_run_end": False,
            "last_within_interval_bound": False,
            "readable": False,
            "complete": False,
        },
        "aiclk_source": "no_valid_in_run_samples",
        "aiclk_mhz": None,
        "aiclk_mhz_in_run": [],
        "errors": errors,
    }


def parse_power_trace(
    path: Path,
    *,
    run_start: datetime.datetime | str | None = None,
    run_end: datetime.datetime | str | None = None,
    coverage_definition: str = EXPLICIT_FINAL_SAMPLE,
    sampler_interval_seconds: float | None = None,
) -> dict[str, Any]:
    """Parse a power CSV and validate the selected run-coverage definition.

    ``explicit_final_sample`` requires a usable sample at or after
    ``run_end``.  ``one_interval_bound`` accepts a final usable sample only in
    the inclusive interval ``run_end - sampler_interval_seconds <=
    sample_timestamp <= run_end``.  The interval is never inferred from CSV
    timestamps.  The returned metadata is board-free and deliberately
    separates physical file facts, timestamp facts, run coverage, and AICLK
    provenance.
    """
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
    trace = _power_trace_base(
        path,
        run_start=run_start,
        run_end=run_end,
        coverage_definition=coverage_definition,
        sampler_interval_seconds=sampler_interval_seconds,
    )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        trace["errors"] = [
            f"power trace is not readable: {type(exc).__name__}"
        ]
        return trace
    trace["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != POWER_TRACE_COLUMNS:
            raise ValueError(f"power trace has unexpected columns {fieldnames!r}")
        rows = [dict(row) for row in reader]
    except (UnicodeDecodeError, csv.Error, TypeError, ValueError) as exc:
        trace["errors"] = [str(exc)]
        return trace

    trace["readable"] = True
    trace["csv_row_count"] = len(rows)
    trace["sample_count"] = len(rows)
    trace["nonempty"] = bool(rows)
    parsed_rows: list[tuple[datetime.datetime, float]] = []
    errors: list[str] = list(trace.get("errors", []))
    for index, row in enumerate(rows):
        if None in row or any(column not in row for column in POWER_TRACE_COLUMNS):
            errors.append(f"power row {index} does not have exactly the declared columns")
            continue
        try:
            timestamp = _parse_trace_timestamp(
                row["timestamp_utc"], field=f"power row {index} timestamp_utc"
            )
            _trace_number(row["power_w"], field=f"power row {index} power_w")
            aiclk = _trace_number(
                row["aiclk_mhz"], field=f"power row {index} aiclk_mhz", positive=True
            )
            _trace_number(row["asic_temp_c"], field=f"power row {index} asic_temp_c")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            parsed_rows.append((timestamp, aiclk))

    trace["valid_row_count"] = len(parsed_rows)
    trace["timestamps_parse"] = bool(rows) and len(parsed_rows) == len(rows)
    if trace["timestamps_parse"]:
        timestamps = [timestamp for timestamp, _aiclk in parsed_rows]
        trace["first_timestamp"] = timestamps[0].isoformat()
        trace["last_timestamp"] = timestamps[-1].isoformat()
        trace["timestamps_ordered"] = all(
            left <= right for left, right in itertools.pairwise(timestamps)
        )
        try:
            start = _parse_trace_timestamp(trace["run_start"], field="run_start")
            end = _parse_trace_timestamp(trace["run_end"], field="run_end")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if start > end:
                errors.append("run_start must be at or before run_end")
            trace["covers_run_start"] = timestamps[0] <= start <= end
            trace["covers_run_end"] = timestamps[-1] >= end
            interval_bound = False
            if coverage_definition == ONE_INTERVAL_BOUND:
                interval_start = end - datetime.timedelta(
                    seconds=float(sampler_interval_seconds)
                )
                interval_bound = interval_start <= timestamps[-1] <= end
            trace["coverage"]["last_within_interval_bound"] = interval_bound
            in_run = [
                aiclk
                for timestamp, aiclk in parsed_rows
                if start <= timestamp <= end
            ]
            trace["in_run_valid_row_count"] = len(in_run)
            trace["aiclk_mhz_in_run"] = in_run
            if in_run:
                trace["aiclk_source"] = "run_trace_samples"
                trace["aiclk_mhz"] = max(in_run)
            trace["coverage_complete"] = all(
                (
                    trace["readable"],
                    trace["nonempty"],
                    trace["valid_row_count"] == trace["csv_row_count"],
                    trace["timestamps_parse"],
                    trace["timestamps_ordered"],
                    timestamps[0] <= start,
                    start <= end,
                    (
                        end <= timestamps[-1]
                        if coverage_definition == EXPLICIT_FINAL_SAMPLE
                        else interval_bound
                    ),
                )
            )
    if not trace["timestamps_parse"] and rows:
        trace["aiclk_source"] = "no_valid_in_run_samples"
    trace["errors"] = errors
    trace["coverage"] = {
        "definition": coverage_definition,
        "nonempty": trace["nonempty"],
        "timestamps_parse": trace["timestamps_parse"],
        "timestamps_ordered": trace["timestamps_ordered"],
        "first_at_or_before_run_start": trace["covers_run_start"],
        "last_at_or_after_run_end": trace["covers_run_end"],
        "last_within_interval_bound": trace["coverage"].get(
            "last_within_interval_bound", False
        ),
        "readable": trace["readable"],
        "complete": trace["coverage_complete"],
    }
    return trace


def validate_power_trace_coverage(
    path: Path,
    *,
    run_start: datetime.datetime | str | None = None,
    run_end: datetime.datetime | str | None = None,
    coverage_definition: str = EXPLICIT_FINAL_SAMPLE,
    sampler_interval_seconds: float | None = None,
) -> dict[str, Any]:
    """Board-free alias for :func:`parse_power_trace` used by record builders."""
    return parse_power_trace(
        path,
        run_start=run_start,
        run_end=run_end,
        coverage_definition=coverage_definition,
        sampler_interval_seconds=sampler_interval_seconds,
    )


# Keep both names discoverable for callers that describe the seam as parsing
# or validation; they intentionally share one implementation and one catalog.
read_power_trace = parse_power_trace
validate_power_trace = validate_power_trace_coverage


def _requested_device_node() -> str:
    return os.environ.get("HEKATUS_TT_DEVICE_NODE") or _DEFAULT_DEVICE_NODE


def _normalize_pci_bus_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = _PCI_BUS_ID_RE.fullmatch(value.strip())
    return match.group(1).lower() if match else None


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


def _device_info(
    snapshot: str,
    *,
    node_resolver: Callable[[str], object] | None = None,
) -> dict | None:
    """Select the board addressed by ``HEKATUS_TT_DEVICE_NODE``.

    With no explicit node, the historical first-device behavior remains the
    default for ``/dev/tenstorrent/0``.  A non-default node is accepted only
    when a verified by-id link resolves it to exactly one PCI bus id and that
    id identifies exactly one snapshot entry.  Numeric node indexes and
    snapshot list positions are never used as physical-board identity.
    """
    try:
        devices = json.loads(snapshot)["device_info"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(devices, list) or not devices:
        return None
    if not all(isinstance(device, dict) for device in devices):
        return None

    requested_node = _requested_device_node()
    if requested_node == _DEFAULT_DEVICE_NODE:
        return devices[0]

    resolver = _resolve_device_node_to_pci if node_resolver is None else node_resolver
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


def parse_telemetry(
    snapshot: str,
    *,
    node_resolver: Callable[[str], object] | None = None,
) -> dict[str, str] | None:
    """Extract one finite sampled reading from the requested board."""
    device = _device_info(snapshot, node_resolver=node_resolver)
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
) -> str | None:
    """One finite CSV row from the requested board, or None if unusable."""
    reading = parse_telemetry(snapshot, node_resolver=node_resolver)
    if reading is None:
        return None
    return f"{timestamp},{reading['power_w']},{reading['aiclk_mhz']},{reading['asic_temp_c']}"


def parse_environment(
    snapshot: str,
    *,
    node_resolver: Callable[[str], object] | None = None,
) -> dict:
    """The board identity that every result has to carry with it."""
    device = _device_info(snapshot, node_resolver=node_resolver)
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
    reading = parse_telemetry(snapshot, node_resolver=node_resolver)
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


def harness_identity() -> dict:
    """Which revision of this tree computed the result.

    The FLOP accounting behind every efficiency figure lives here, so a
    measurement that cannot name the harness that produced it cannot be
    reproduced or compared against a later one. A modified tree is recorded
    as modified rather than silently attributed to its last commit, and a
    tree git could not read at all is recorded as unknown rather than clean.
    A transferred source tree may deliberately omit ``.git``; the wrapper can
    then provide the already-verified source revision explicitly.
    """
    override_commit = os.environ.get("HEKATUS_TT_HARNESS_COMMIT")
    if override_commit:
        override_dirty = os.environ.get("HEKATUS_TT_HARNESS_DIRTY", "")
        if override_dirty not in {"0", "1", "false", "true"}:
            return {
                "harness_commit": None,
                "harness_dirty": None,
                "harness_identity_error": (
                    "HEKATUS_TT_HARNESS_DIRTY must be one of 0, 1, false, true"
                ),
            }
        return {
            "harness_commit": override_commit,
            "harness_dirty": override_dirty in {"1", "true"},
            "harness_identity_source": "HEKATUS_TT_HARNESS_COMMIT",
        }
    repo = Path(__file__).resolve().parents[3]
    commit, commit_ok = _git(repo, "rev-parse", "HEAD")
    # Tracked changes only. The toolchain writes its build output into the
    # tree on every run, so counting untracked files would make the flag true
    # always, which says nothing about the code that computed the result.
    status, status_ok = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if not (commit_ok and status_ok):
        return {
            "harness_commit": None,
            "harness_dirty": None,
            "harness_identity_error": commit if not commit_ok else status,
        }
    return {"harness_commit": commit, "harness_dirty": bool(status)}


def capture_environment(
    image: str,
    image_pinned: bool,
    *,
    sampler_mode: str = "default",
    sampler_interval: float | None = None,
) -> dict:
    """Everything needed to name the environment a measurement came from."""
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
    environment.update(parse_environment(_run(SNAPSHOT_COMMAND)))
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
    happening silently.
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


def _sample_forever(out_path: Path, interval: float) -> None:
    with out_path.open("w", buffering=1) as handle:
        _sample_loop(handle, interval)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture board environment or sample telemetry")
    sub = parser.add_subparsers(dest="mode", required=True)

    capture = sub.add_parser("capture-env", help="write the environment as JSON and exit")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--image", required=True)
    capture.add_argument("--image-pinned", action="store_true")
    capture.add_argument("--sampler-mode", choices=SAMPLER_MODES, default="default")
    capture.add_argument("--sampler-interval", type=float, default=None)

    sample = sub.add_parser("sample", help="append telemetry rows until terminated")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--interval", type=float, default=DEFAULT_SAMPLER_INTERVAL_SECONDS)

    args = parser.parse_args(argv)
    if args.mode == "sample" and not (math.isfinite(args.interval) and args.interval > 0):
        # Zero turns the loop into a busy wait on the snapshot tool; negative
        # and non-finite values reach sleep and end the sampler outright.
        parser.error(f"--interval must be positive and finite, got {args.interval}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "capture-env":
        try:
            if args.sampler_mode == "default" and args.sampler_interval is None:
                # Keep the two-argument call compatible with board-free callers
                # that replace capture_environment in tests.
                environment = capture_environment(args.image, args.image_pinned)
            else:
                environment = capture_environment(
                    args.image,
                    args.image_pinned,
                    sampler_mode=args.sampler_mode,
                    sampler_interval=args.sampler_interval,
                )
        except ValueError as exc:
            parser.error(str(exc))
        args.out.write_text(strict_json_dumps(environment, indent=2) + "\n")
        print(f"environment -> {args.out}")
        return

    try:
        _sample_forever(args.out, args.interval)
    except KeyboardInterrupt:
        # The wrapper terminates the sampler when the run finishes; that is
        # the normal way this ends, not a failure.
        pass


if __name__ == "__main__":
    main()
