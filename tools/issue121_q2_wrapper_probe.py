"""Construct the resident wrapper's Docker argv without touching a device.

This is a board-free static model of the resident branch in
``enodia/tt/bench/run_in_container.sh``.  It deliberately omits the wrapper's
``--device`` pair from the returned argv: the result is for inspection only and
must never be passed to Docker.  The omitted pair is reported separately by the
CLI so a manual probe can compare the hardware binding without starting one.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

RESIDENT_RUNNER = "enodia/tt/bench/run_resident.py"
DEFAULT_HUGEPAGES_SOURCE = "/dev/hugepages-1G"
DEFAULT_DEVICE_NODE = "/dev/tenstorrent/1"


def _resident_runner_args(
    runner_args: Sequence[str],
    *,
    telemetry_mode: str,
    environment_name: str,
    power_trace_name: str,
    runner_result_path: str,
) -> tuple[str, ...]:
    """Apply only the resident-branch argument rewrites from the shell wrapper."""
    args = list(runner_args)
    if "--out" not in args:
        args = ["--out", runner_result_path, *args]

    resident_args: list[str] = []
    has_power_trace = False
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--env-json":
            if index + 1 >= len(args):
                raise ValueError("--env-json requires a value")
            index += 2
            continue
        if argument.startswith("--env-json="):
            index += 1
            continue
        if argument == "--power-trace":
            if index + 1 >= len(args):
                raise ValueError("--power-trace requires a value")
            has_power_trace = True
            if telemetry_mode != "off":
                resident_args.extend(("--power-trace", power_trace_name))
            index += 2
            continue
        if argument.startswith("--power-trace="):
            has_power_trace = True
            if telemetry_mode != "off":
                resident_args.append(f"--power-trace={power_trace_name}")
            index += 1
            continue
        resident_args.append(argument)
        index += 1

    resident_args.extend(("--env-json", environment_name))
    if telemetry_mode != "off" and not has_power_trace:
        resident_args.extend(("--power-trace", power_trace_name))
    return tuple(resident_args)


def build_resident_docker_argv(
    *,
    repo_root: Path,
    output_dir: Path,
    image: str,
    run_id: str,
    container_name: str,
    container_timeout_seconds: int,
    runner_args: Sequence[str] = (),
    telemetry_mode: str = "default",
    telemetry_interval_seconds: str = "2",
    device_node: str = DEFAULT_DEVICE_NODE,
    hugepages_source: str = DEFAULT_HUGEPAGES_SOURCE,
    runner_result_path: str = "/out/runner-result.json",
) -> tuple[str, ...]:
    """Return the resident ``docker run`` argv, excluding ``--device``.

    ``run_id`` is the wrapper-generated id, not the caller's workload id.  The
    wrapper uses it for the sibling environment and power-trace basenames.
    ``output_dir`` is resolved because the shell wrapper normalizes the exact
    host bind source before invoking Docker.  ``device_node`` is accepted only
    so callers can report the omitted hardware argument alongside this vector.
    """
    if not run_id or any(character in run_id for character in "\r\n"):
        raise ValueError("run_id must be non-empty and contain no newline")
    if not container_name or any(character in container_name for character in "\r\n"):
        raise ValueError("container_name must be non-empty and contain no newline")
    if telemetry_mode not in {"off", "default", "explicit"}:
        raise ValueError("telemetry_mode must be off, default, or explicit")
    if not device_node or any(character in device_node for character in "\r\n"):
        raise ValueError("device_node must be non-empty and contain no newline")
    if container_timeout_seconds <= 0:
        raise ValueError("container_timeout_seconds must be positive")
    if telemetry_mode == "explicit" and not telemetry_interval_seconds:
        raise ValueError("explicit telemetry mode requires an interval")

    resolved_repo = repo_root.resolve()
    resolved_output = output_dir.resolve()
    environment_name = f"/out/env-{run_id}.json"
    power_trace_name = f"power-{run_id}.csv"
    resident_args = _resident_runner_args(
        runner_args,
        telemetry_mode=telemetry_mode,
        environment_name=environment_name,
        power_trace_name=power_trace_name,
        runner_result_path=runner_result_path,
    )

    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "-v",
        f"{hugepages_source}:/dev/hugepages-1G",
        "-v",
        f"{resolved_repo}:/work",
        "-v",
        f"{resolved_output}:/out",
        "-w",
        "/work",
        "-e",
        "PYTHONPATH=/work",
        "-e",
        f"HEKATUS_TT_RUN_ID={run_id}",
    ]
    if telemetry_mode != "off":
        argv.extend(("-e", f"HEKATUS_TT_TELEMETRY_MODE={telemetry_mode}"))
        argv.extend(("-e", f"HEKATUS_TT_TELEMETRY_INTERVAL_S={telemetry_interval_seconds}"))
    argv.extend(("-e", f"HEKATUS_TT_RESULT_PATH={runner_result_path}"))
    argv.extend(
        (
            "--entrypoint",
            "python3",
            image,
            RESIDENT_RUNNER,
            *resident_args,
        )
    )
    return tuple(argv)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--container-name", default="hekatus-bench-static-probe")
    parser.add_argument("--container-timeout-seconds", type=int, default=330)
    parser.add_argument("--telemetry-mode", choices=("off", "default", "explicit"), default="default")
    parser.add_argument("--telemetry-interval-seconds", default="2")
    parser.add_argument("--device-node", default=DEFAULT_DEVICE_NODE)
    parser.add_argument("--runner-arg", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    vector = build_resident_docker_argv(
        repo_root=args.repo_root,
        output_dir=args.out_dir,
        image=args.image,
        run_id=args.run_id,
        container_name=args.container_name,
        container_timeout_seconds=args.container_timeout_seconds,
        runner_args=args.runner_arg,
        telemetry_mode=args.telemetry_mode,
        telemetry_interval_seconds=args.telemetry_interval_seconds,
        device_node=args.device_node,
    )
    print(
        json.dumps(
            {
                "docker_argv_without_device": list(vector),
                "omitted_hardware_argv": ["--device", args.device_node],
                "started_container": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
