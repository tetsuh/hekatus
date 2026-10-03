"""Run one production Newton-Schulz correctness stage and emit JSON.

This runner is intentionally small: it calls ``run_newton_schulz_kernel``
directly, uses the host NumPy oracle with the same iteration count, and writes a
machine-readable result for each staged board run.  The default remains the
production eight-iteration configuration; ``--iterations 1`` is an explicit
bring-up opt-in.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.newton_schulz_kernel import (
    INPUT_MEMORY_CHOICES,
    MATRIX_BLOCK_CHOICES,
    NEWTON_SCHULZ_ITERATION_CHOICES,
    NEWTON_SCHULZ_ITERATIONS,
    _validate_iterations,
    run_newton_schulz_kernel,
)
from enodia.tt.bench.newton_schulz_reference import (
    newton_schulz_reference,
    random_hpd_batch,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, choices=(16, 32), default=32)
    parser.add_argument(
        "--iterations",
        type=int,
        choices=NEWTON_SCHULZ_ITERATION_CHOICES,
        default=NEWTON_SCHULZ_ITERATIONS,
        help="Newton-Schulz steps (default: 8; use 1 only for staged bring-up)",
    )
    parser.add_argument("--matrix-block", type=int, choices=MATRIX_BLOCK_CHOICES, default=1)
    parser.add_argument("--variant", choices=("bf16", "bf16-fp32state"), default="bf16-fp32state")
    parser.add_argument("--math-fidelity", choices=("LoFi", "HiFi2", "HiFi3", "HiFi4"), default="HiFi3")
    parser.add_argument("--two-tile-complex", action="store_true")
    parser.add_argument("--fuse-s", action="store_true")
    parser.add_argument("--batch-reads", action="store_true")
    parser.add_argument("--input-memory", choices=INPUT_MEMORY_CHOICES, default="dram")
    parser.add_argument("--r-memory", choices=INPUT_MEMORY_CHOICES, default=None)
    parser.add_argument("--x0-memory", choices=INPUT_MEMORY_CHOICES, default=None)
    parser.add_argument(
        "--dst-full-sync-en",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use full DEST synchronization (default: enabled)",
    )
    parser.add_argument("--fp32-dest-acc-en", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=6300)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("newton-schulz-result.json"))
    parser.add_argument("--env-json", type=Path, default=None)
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch < 1:
        parser.error(f"--batch must be positive, got {args.batch}")
    if args.tolerance <= 0 or not np.isfinite(args.tolerance):
        parser.error(f"--tolerance must be positive and finite, got {args.tolerance}")
    try:
        _validate_iterations(args.iterations)
    except ValueError as exc:
        parser.error(str(exc))
    if args.two_tile_complex and args.fuse_s:
        parser.error("--two-tile-complex and --fuse-s are mutually exclusive")


def _environment(args: argparse.Namespace) -> dict:
    environment = {"python": platform.python_version()}
    if args.env_json is not None and args.env_json.exists():
        environment.update(json.loads(args.env_json.read_text()))
    return environment


def run_stage(args: argparse.Namespace) -> dict:
    import ttnn

    matrices = random_hpd_batch(args.batch, args.size, seed=args.seed)
    expected = newton_schulz_reference(matrices, iterations=args.iterations)
    device = ttnn.open_device(device_id=args.device_id)
    try:
        actual = run_newton_schulz_kernel(
            ttnn,
            device,
            matrices,
            variant=args.variant,
            math_fidelity=args.math_fidelity,
            fuse_s=args.fuse_s,
            two_tile_complex=args.two_tile_complex,
            batch_reads=args.batch_reads,
            matrix_block=args.matrix_block,
            fp32_dest_acc_en=args.fp32_dest_acc_en,
            dst_full_sync_en=args.dst_full_sync_en,
            input_memory=args.input_memory,
            r_memory=args.r_memory,
            x0_memory=args.x0_memory,
            iterations=args.iterations,
        )
    finally:
        ttnn.close_device(device)

    per_matrix_errors = np.linalg.norm(actual - expected, axis=(1, 2)) / np.maximum(
        np.linalg.norm(expected, axis=(1, 2)), 1.0
    )
    relative_error = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))
    return {
        "status": "pass" if np.isfinite(actual).all() and relative_error <= args.tolerance else "fail",
        "batch": args.batch,
        "size": args.size,
        "iterations": args.iterations,
        "matrix_block": args.matrix_block,
        "relative_error": relative_error,
        "max_matrix_relative_error": float(np.max(per_matrix_errors)),
        "per_matrix_relative_errors": [float(value) for value in per_matrix_errors],
        "tolerance": args.tolerance,
        "finite": bool(np.isfinite(actual).all()),
        "reference": "newton_schulz_reference",
        "production_api": "run_newton_schulz_kernel",
        "production_compute": "enodia/tt/bench/kernels/newton_schulz_compute.cpp",
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    try:
        result = run_stage(args)
    except Exception as exc:  # noqa: BLE001 - preserve device failures in JSON
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "batch": args.batch,
            "size": args.size,
            "iterations": args.iterations,
            "matrix_block": args.matrix_block,
            "production_api": "run_newton_schulz_kernel",
        }
    payload = {
        "environment": _environment(args),
        "configuration_mode": "production_newton_schulz_stage",
        "selection": {
            "batch": args.batch,
            "size": args.size,
            "iterations": args.iterations,
            "matrix_block": args.matrix_block,
            "variant": args.variant,
            "math_fidelity": args.math_fidelity,
            "fuse_s": args.fuse_s,
            "two_tile_complex": args.two_tile_complex,
            "batch_reads": args.batch_reads,
            "input_memory": args.input_memory,
            "r_memory": args.r_memory,
            "x0_memory": args.x0_memory,
            "fp32_dest_acc_en": args.fp32_dest_acc_en,
            "dst_full_sync_en": args.dst_full_sync_en,
            "device_id": args.device_id,
            "seed": args.seed,
        },
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(strict_json_dumps(payload, indent=2) + "\n")
    print(strict_json_dumps(payload, indent=2), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
