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
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.strict_json import dumps as strict_json_dumps

SNAPSHOT_COMMAND = ("tt-smi", "-s", "--snapshot_no_tty")
CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c"
_DEFAULT_DEVICE_NODE = "/dev/tenstorrent/0"
_PCI_BUS_ID_RE = re.compile(
    r"(?<![0-9a-f])([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])(?![0-9a-f])",
    re.IGNORECASE,
)
_BOARD_BUS_ID_KEYS = ("bus_id", "pci_bus_id", "pci_address", "pci_bdf")


def _requested_device_node() -> str:
    return os.environ.get("HEKATUS_TT_DEVICE_NODE") or _DEFAULT_DEVICE_NODE


def _normalize_pci_bus_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = _PCI_BUS_ID_RE.fullmatch(value.strip())
    return match.group(1).lower() if match else None


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


def _by_id_pci_bus_id(path: Path) -> str | None:
    """Read a PCI identity from a verified ``by-id/pci-*`` symlink."""
    if path.parent.name != "by-id" or not path.is_symlink():
        return None
    if not path.name.startswith("pci-"):
        return None
    return _normalize_pci_bus_id(path.name.removeprefix("pci-"))


def _resolve_device_node_to_pci(node: str) -> set[str]:
    """Resolve a device node to the PCI ids of verified by-id links.

    Numeric node names are deliberately not interpreted as board indexes.  A
    direct node is usable only when a sibling ``by-id/pci-*`` symlink resolves
    to that exact node.  A by-id node is usable only when it is itself a real
    symlink into the same device directory.
    """
    path = Path(node)
    if not path.is_absolute():
        return set()
    try:
        resolved_node = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return set()

    if path.parent.name == "by-id":
        bus_id = _by_id_pci_bus_id(path)
        if bus_id is None:
            return set()
        try:
            device_dir = path.parent.parent.resolve(strict=True)
        except (OSError, RuntimeError):
            return set()
        return {bus_id} if resolved_node.parent == device_dir else set()

    matches: set[str] = set()
    try:
        by_id_entries = tuple((path.parent / "by-id").iterdir())
    except OSError:
        return set()
    for entry in by_id_entries:
        bus_id = _by_id_pci_bus_id(entry)
        if bus_id is None:
            continue
        try:
            if entry.resolve(strict=True) == resolved_node:
                matches.add(bus_id)
        except (OSError, RuntimeError):
            continue
    return matches


def _resolved_pci_bus_ids(
    node: str, resolver: Callable[[str], object]
) -> set[str]:
    """Normalize an injected resolver result, rejecting malformed identities."""
    try:
        resolved = resolver(node)
    except Exception:  # noqa: BLE001 - an unverifiable node must be rejected
        return set()
    if isinstance(resolved, str):
        values = (resolved,)
    elif isinstance(resolved, (list, tuple, set, frozenset)):
        values = tuple(resolved)
    else:
        return set()
    normalized: set[str] = set()
    for value in values:
        bus_id = _normalize_pci_bus_id(value)
        if bus_id is None:
            return set()
        normalized.add(bus_id)
    return normalized


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
    """Extract the sampled quantities, or None if the snapshot is unusable."""
    device = _device_info(snapshot, node_resolver=node_resolver)
    if device is None:
        return None
    try:
        telemetry = device["telemetry"]
        return {
            "power_w": str(telemetry["power"]).strip(),
            "aiclk_mhz": str(telemetry["aiclk"]).strip(),
            "asic_temp_c": str(telemetry["asic_temperature"]).strip(),
        }
    except (KeyError, TypeError):
        return None


def telemetry_csv_row(
    snapshot: str,
    *,
    timestamp: str,
    node_resolver: Callable[[str], object] | None = None,
) -> str | None:
    """One CSV row in the order of CSV_HEADER, or None if nothing was read."""
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
    return {
        key: device[key] for key in ("board_info", "firmwares", "limits") if key in device
    } | {
        "board": device.get("board_info"),
        "firmware": device.get("firmwares"),
        "limits": device.get("limits"),
        "device_node": _requested_device_node(),
    }


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


def capture_environment(image: str, image_pinned: bool) -> dict:
    """Everything needed to name the environment a measurement came from."""
    environment = {
        "captured_at": _now(),
        "image": image,
        "image_pinned": image_pinned,
        "kernel": _run("uname -sr").strip(),
        "kmd_version": _run("modinfo tenstorrent 2>/dev/null | awk '/^version:/{print $2}'").strip(),
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

    sample = sub.add_parser("sample", help="append telemetry rows until terminated")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--interval", type=float, default=2.0)

    args = parser.parse_args(argv)
    if args.mode == "sample" and not (math.isfinite(args.interval) and args.interval > 0):
        # Zero turns the loop into a busy wait on the snapshot tool; negative
        # and non-finite values reach sleep and end the sampler outright.
        parser.error(f"--interval must be positive and finite, got {args.interval}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "capture-env":
        args.out.write_text(
            strict_json_dumps(capture_environment(args.image, args.image_pinned), indent=2) + "\n"
        )
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
