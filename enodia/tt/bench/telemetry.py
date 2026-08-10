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
import subprocess
import sys
import time
from pathlib import Path

SNAPSHOT_COMMAND = ("tt-smi", "-s", "--snapshot_no_tty")
CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c"


def _device_info(snapshot: str) -> dict | None:
    """The first device, or None for anything this cannot read.

    Every shape the snapshot might arrive in is checked rather than assumed.
    An exception raised here would propagate out of the sampling loop and end
    it, and a dead sampler says nothing at all — which is how the first run
    produced a trace containing only its header.
    """
    try:
        devices = json.loads(snapshot)["device_info"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(devices, list) or not devices:
        return None
    return devices[0] if isinstance(devices[0], dict) else None


def parse_telemetry(snapshot: str) -> dict[str, str] | None:
    """Extract the sampled quantities, or None if the snapshot is unusable."""
    device = _device_info(snapshot)
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


def telemetry_csv_row(snapshot: str, *, timestamp: str) -> str | None:
    """One CSV row in the order of CSV_HEADER, or None if nothing was read."""
    reading = parse_telemetry(snapshot)
    if reading is None:
        return None
    return f"{timestamp},{reading['power_w']},{reading['aiclk_mhz']},{reading['asic_temp_c']}"


def parse_environment(snapshot: str) -> dict:
    """The board identity that every result has to carry with it."""
    device = _device_info(snapshot)
    if device is None:
        return {"board_snapshot_error": "no device information in the snapshot"}
    return {
        key: device[key] for key in ("board_info", "firmwares", "limits") if key in device
    } | {
        "board": device.get("board_info"),
        "firmware": device.get("firmwares"),
        "limits": device.get("limits"),
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
    environment.update(parse_environment(_run(SNAPSHOT_COMMAND)))
    return environment


def _sample_forever(out_path: Path, interval: float) -> None:
    """Sample until terminated, surviving anything one sample can throw.

    The sampler is a passive observer of a run it must not disturb, and its
    death is silent: whoever reads the trace afterwards sees a short file,
    not an error. So a failed sample costs one row, never the trace.
    """
    with out_path.open("w", buffering=1) as handle:
        handle.write(CSV_HEADER + "\n")
        while True:
            try:
                row = telemetry_csv_row(_run(SNAPSHOT_COMMAND), timestamp=_now())
                if row is not None:
                    handle.write(row + "\n")
            except Exception as exc:  # noqa: BLE001 - one lost row beats a lost trace
                print(f"telemetry sample failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
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
            json.dumps(capture_environment(args.image, args.image_pinned), indent=2) + "\n"
        )
        print(f"environment -> {args.out}")
        return 0

    try:
        _sample_forever(args.out, args.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
