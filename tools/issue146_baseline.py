"""Board-free resolver for the preceding Issue #146 cooldown snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from enodia.strict_json import dumps as strict_json_dumps
from tools.issue146_contract import (
    AUTHORITATIVE_PLAN_CANONICAL_SHA256,
    AUTHORITATIVE_PLAN_RAW_SHA256,
    DEFAULT_PLAN_PATH,
)

RECORD_SCHEMA = "issue146-measurement-v1"
SNAPSHOT_SCHEMA = "issue146-tt-smi-snapshot-v1"


class BaselineResolutionError(ValueError):
    """Raised when a preceding record cannot identify its pre-run snapshot."""


def _read_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineResolutionError(f"{label} cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise BaselineResolutionError(f"{label} must be an object")
    return payload, raw


def _inside(root: Path, path: Path, *, label: str) -> Path:
    if path.is_absolute():
        candidate = path
    else:
        candidate = root / path
    try:
        if candidate.is_symlink():
            raise BaselineResolutionError(f"{label} is a symlink")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve())
        metadata = resolved.stat()
    except BaselineResolutionError:
        raise
    except (OSError, ValueError) as exc:
        raise BaselineResolutionError(f"{label} cannot be validated") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise BaselineResolutionError(f"{label} is not an exclusive regular file")
    return resolved


def _plan_identity(plan_path: Path) -> tuple[str, str, str]:
    payload, raw = _read_json(plan_path, label="Issue #146 plan")
    if hashlib.sha256(raw).hexdigest() != AUTHORITATIVE_PLAN_RAW_SHA256:
        raise BaselineResolutionError("Issue #146 plan bytes are not the pinned revision-6 plan")
    try:
        canonical = hashlib.sha256(
            strict_json_dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise BaselineResolutionError("Issue #146 plan cannot be canonicalized") from exc
    if canonical != AUTHORITATIVE_PLAN_CANONICAL_SHA256:
        raise BaselineResolutionError("Issue #146 plan canonical hash is not pinned")
    scope = payload.get("measurement_scope")
    if not isinstance(scope, Mapping):
        raise BaselineResolutionError("Issue #146 measurement scope is missing")
    board_serial = scope.get("board_serial")
    bus_id = scope.get("pci_bus_id")
    if not isinstance(board_serial, str) or not board_serial:
        raise BaselineResolutionError("Issue #146 board serial is missing from the plan")
    if not isinstance(bus_id, str) or not bus_id:
        raise BaselineResolutionError("Issue #146 PCI identity is missing from the plan")
    return board_serial, bus_id, canonical


def _require_run_identity(
    record: Mapping[str, Any],
    *,
    expected_run_number: int,
    expected_run_id: str,
    expected_configuration: str,
    expected_plan_hash: str,
) -> Mapping[str, Any]:
    if record.get("record_schema") != RECORD_SCHEMA:
        raise BaselineResolutionError("preceding record schema is unsupported")
    if record.get("status") != "ok":
        raise BaselineResolutionError("preceding record is not completed successfully")
    if record.get("plan_sha256") != expected_plan_hash:
        raise BaselineResolutionError("preceding record plan hash is not the pinned evaluation plan")
    run = record.get("run")
    if not isinstance(run, Mapping):
        raise BaselineResolutionError("preceding record run block is missing")
    if run.get("mode") != "long":
        raise BaselineResolutionError("preceding record is not a long run")
    run_number = run.get("run_number")
    if isinstance(run_number, bool) or not isinstance(run_number, int) or run_number != expected_run_number:
        raise BaselineResolutionError("preceding record run number is not the expected predecessor")
    if run.get("run_id") != expected_run_id:
        raise BaselineResolutionError("preceding record run ID is not the expected predecessor")
    configuration = record.get("configuration")
    if not isinstance(configuration, Mapping) or configuration.get("id") != expected_configuration:
        raise BaselineResolutionError("preceding record configuration is not the expected predecessor")
    return run


def _require_descriptor(
    record: Mapping[str, Any], *, expected_run_id: str, board_serial: str, bus_id: str
) -> Mapping[str, Any]:
    descriptor = record.get("pre_run_snapshot")
    if not isinstance(descriptor, Mapping):
        raise BaselineResolutionError("preceding record pre_run_snapshot descriptor is missing")
    expected_name = f"pre-run-snapshot-{expected_run_id}.json"
    path = descriptor.get("path")
    if (
        not isinstance(path, str)
        or not path
        or Path(path).name != path
        or "\\" in path
        or path != expected_name
    ):
        raise BaselineResolutionError("pre_run_snapshot path is not the expected basename")
    if descriptor.get("schema") != SNAPSHOT_SCHEMA:
        raise BaselineResolutionError("pre_run_snapshot descriptor schema is unsupported")
    selected_device = descriptor.get("selected_device")
    if not isinstance(selected_device, Mapping):
        raise BaselineResolutionError("pre_run_snapshot selected_device identity is missing")
    if selected_device.get("board_id") != board_serial or selected_device.get("bus_id") != bus_id:
        raise BaselineResolutionError("pre_run_snapshot selected_device identity is wrong")
    digest = descriptor.get("sha256")
    byte_count = descriptor.get("byte_count")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise BaselineResolutionError("pre_run_snapshot SHA-256 is malformed")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise BaselineResolutionError("pre_run_snapshot byte_count is malformed")
    return descriptor


def _snapshot_device(
    payload: Mapping[str, Any], *, board_serial: str, bus_id: str
) -> Mapping[str, Any]:
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise BaselineResolutionError("pre_run snapshot payload is missing snapshot")
    devices = snapshot.get("device_info")
    if not isinstance(devices, list):
        raise BaselineResolutionError("pre_run snapshot device_info is missing")
    matches: list[Mapping[str, Any]] = []
    for device in devices:
        if not isinstance(device, Mapping):
            continue
        board = device.get("board_info")
        if not isinstance(board, Mapping):
            continue
        if board.get("board_id") != board_serial or board.get("bus_id") != bus_id:
            continue
        serial = board.get("serial")
        if serial is not None and str(serial) != board_serial:
            raise BaselineResolutionError("pre_run snapshot serial identity is wrong")
        matches.append(device)
    if len(matches) != 1:
        raise BaselineResolutionError("pre_run snapshot selected-device identity is ambiguous")
    return matches[0]


def _snapshot_run_identity(
    payload: Mapping[str, Any], *, expected_run_id: str
) -> None:
    sources: list[Mapping[str, Any]] = [payload]
    for key in ("snapshot", "provenance"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    for source in sources:
        for key in ("run_id", "requested_run_id"):
            value = source.get(key)
            if value is not None and value != expected_run_id:
                raise BaselineResolutionError(
                    f"pre_run snapshot {key} does not match the preceding run"
                )


def resolve_pre_snapshot(
    record_path: Path,
    *,
    expected_run_number: int,
    expected_run_id: str,
    expected_configuration: str,
    plan_path: Path = DEFAULT_PLAN_PATH,
    repository_root: Path | None = None,
) -> str:
    """Return the repo-relative preceding pre-run snapshot after full validation."""
    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    record = _inside(root, Path(record_path), label="preceding record")
    plan = plan_path if plan_path.is_absolute() else root / plan_path
    try:
        plan_source = plan.resolve(strict=True)
    except OSError as exc:
        raise BaselineResolutionError("Issue #146 plan cannot be located") from exc
    board_serial, bus_id, plan_hash = _plan_identity(plan_source)
    record_payload, _record_raw = _read_json(record, label="preceding record")
    _require_run_identity(
        record_payload,
        expected_run_number=expected_run_number,
        expected_run_id=expected_run_id,
        expected_configuration=expected_configuration,
        expected_plan_hash=plan_hash,
    )
    descriptor = _require_descriptor(
        record_payload,
        expected_run_id=expected_run_id,
        board_serial=board_serial,
        bus_id=bus_id,
    )
    snapshot_path = _inside(root, record.parent / str(descriptor["path"]), label="pre_run snapshot")
    try:
        snapshot_raw = snapshot_path.read_bytes()
    except OSError as exc:
        raise BaselineResolutionError("pre_run snapshot cannot be read") from exc
    if hashlib.sha256(snapshot_raw).hexdigest() != descriptor["sha256"]:
        raise BaselineResolutionError("pre_run snapshot SHA-256 does not match its descriptor")
    if len(snapshot_raw) != descriptor["byte_count"]:
        raise BaselineResolutionError("pre_run snapshot byte_count does not match its descriptor")
    payload, _raw = _read_json(snapshot_path, label="pre_run snapshot")
    if payload.get("schema") != SNAPSHOT_SCHEMA or payload.get("stage") != "pre_run":
        raise BaselineResolutionError("pre_run snapshot schema or stage is invalid")
    if payload.get("status") != "ok" or payload.get("returncode") != 0:
        raise BaselineResolutionError("pre_run snapshot capture did not complete successfully")
    _snapshot_device(payload, board_serial=board_serial, bus_id=bus_id)
    _snapshot_run_identity(payload, expected_run_id=expected_run_id)
    return snapshot_path.relative_to(root).as_posix()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--expected-run-number", type=int, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-configuration", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = resolve_pre_snapshot(
            args.record,
            expected_run_number=args.expected_run_number,
            expected_run_id=args.expected_run_id,
            expected_configuration=args.expected_configuration,
            plan_path=args.plan,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Issue #146 baseline resolver rejected input: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
