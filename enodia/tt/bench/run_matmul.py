"""Run the shape catalogue on the accelerator and record what it achieved.

Runs inside the toolchain container, so it imports nothing from the
reference implementation — only the standard library and the modules beside
it, which are standard-library-only by design.

**Accounting matches execution.** A complex operation costs four real
matmuls, and this runs four. Counting four and timing one would report four
times the achieved throughput, silently, in the direction that flatters.

**What it measures.** A block of iterations is timed with a single
synchronization at the end, so per-iteration cost is not swamped by
synchronization on the small shapes; the block is repeated and the best
block is reported, which keeps scheduler noise out of a throughput figure.
Each result is released as it is produced, both to keep the larger shapes
inside memory and because reusing buffers is what a real implementation
does.

**Failures are results.** A shape that will not fit in L1 fails here, and
that failure is recorded rather than aborting the run. Where the boundary
falls is the answer to the question design.md §2 calls paramount — whether
the data fits on-chip — so it is data, not an error.

**Efficiency is optional.** Without an explicit peak, only achieved FLOPS
are reported. An efficiency quoted against the wrong peak is worse than no
efficiency, so the peak and the note describing it are recorded next to
anything derived from them.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):  # invoked as a plain script inside the container
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.tt.bench.shapes import MatmulShape, default_catalogue, total_flops


def _make_tensor(ttnn, shape: tuple[int, ...], dtype, layout, device, memory_config):
    """Allocate a device tensor, tolerating differences in the creation API.

    Values do not affect matmul timing on this architecture — there is no
    sparsity shortcut to hit — so any of these is acceptable, and the one
    that worked is recorded with the result.
    """
    attempts = []
    for name in ("rand", "ones", "zeros"):
        factory = getattr(ttnn, name, None)
        if factory is None:
            continue
        try:
            tensor = factory(
                shape, dtype=dtype, layout=layout, device=device, memory_config=memory_config
            )
        except Exception as exc:  # noqa: BLE001 - the API surface is what is under test
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        return tensor, name
    raise RuntimeError("no usable tensor factory; tried " + " | ".join(attempts))


def _execute_once(ttnn, a, b, real_matmuls: int) -> None:
    """One logical operation: every real matmul the accounting charges for."""
    for _ in range(real_matmuls):
        out = ttnn.matmul(a, b)
        ttnn.deallocate(out)


def with_efficiency(record: dict, peak_tflops: float | None) -> dict:
    """Add an efficiency only when there is a stated peak to divide by."""
    if peak_tflops and record.get("achieved_tflops"):
        record["efficiency"] = record["achieved_tflops"] / peak_tflops
    return record


def run_shape(
    ttnn,
    device,
    shape: MatmulShape,
    *,
    dtype,
    memory_config,
    iters: int,
    repeats: int,
) -> dict:
    """Execute one shape and return its record, including any failure."""
    tensors = []
    try:
        a, factory = _make_tensor(
            ttnn, (shape.batch, 1, shape.m, shape.k), dtype, ttnn.TILE_LAYOUT, device, memory_config
        )
        tensors.append(a)
        b, _ = _make_tensor(
            ttnn, (shape.batch, 1, shape.k, shape.n), dtype, ttnn.TILE_LAYOUT, device, memory_config
        )
        tensors.append(b)

        # Warm up: the first execution pays for program compilation and cache
        # population, which is real but is not what a steady-state frame costs.
        _execute_once(ttnn, a, b, shape.real_matmuls)
        ttnn.synchronize_device(device)

        best = None
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iters):
                _execute_once(ttnn, a, b, shape.real_matmuls)
            ttnn.synchronize_device(device)
            elapsed = (time.perf_counter() - start) / iters
            best = elapsed if best is None else min(best, elapsed)

        flops = total_flops(shape)
        return {
            "status": "ok",
            "seconds_per_iteration": best,
            "achieved_tflops": flops / best / 1e12,
            "flops_per_iteration": flops,
            "real_matmuls_per_iteration": shape.real_matmuls,
            "tensor_factory": factory,
        }
    except Exception as exc:  # noqa: BLE001 - a shape that cannot run is a result
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        for tensor in tensors:
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask the result
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", action="append", default=None, help="repeatable")
    parser.add_argument("--memory", action="append", default=None, choices=["dram", "l1"])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", default=None, help="substring filter on the shape name")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("bench-results.json"))
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--peak-note", default=None, help="what that peak refers to")
    parser.add_argument("--env-json", type=Path, default=None, help="environment to embed")
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject controls that would produce nonsense, before any device is opened."""
    if args.iters < 1:
        parser.error(f"--iters must be at least 1, got {args.iters}")
    if args.repeats < 1:
        parser.error(f"--repeats must be at least 1, got {args.repeats}")
    if args.peak_tflops is not None and not (
        math.isfinite(args.peak_tflops) and args.peak_tflops > 0
    ):
        parser.error(f"--peak-tflops must be positive and finite, got {args.peak_tflops}")


def _format_line(shape: MatmulShape, dtype_name: str, memory_name: str, record: dict) -> str:
    line = f"{shape.name:38s} {dtype_name:9s} {memory_name:4s} "
    if record["status"] != "ok":
        return line + f"failed: {record['error'][:60]}"
    line += f"{record['achieved_tflops']:8.2f} TFLOPS"
    if "efficiency" in record:
        line += f"  {record['efficiency'] * 100:5.1f}%"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)

    import ttnn  # imported after validation, so bad arguments need no accelerator

    dtypes = args.dtype or ["bfloat16", "float32"]
    memories = args.memory or ["dram", "l1"]
    dtype_map = {name: getattr(ttnn, name) for name in dtypes if hasattr(ttnn, name)}
    missing = sorted(set(dtypes) - set(dtype_map))
    if missing:
        print(f"unknown dtype(s) for this toolchain: {missing}", file=sys.stderr)
        return 2
    memory_map = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}

    catalogue = [s for s in default_catalogue() if not args.only or args.only in s.name]
    if not catalogue:
        print(f"no shape matches {args.only!r}", file=sys.stderr)
        return 2

    environment = {"python": platform.python_version(), "host": platform.node()}
    if args.env_json and args.env_json.exists():
        environment.update(json.loads(args.env_json.read_text()))

    device = ttnn.open_device(device_id=args.device_id)
    results = []
    try:
        for shape in catalogue:
            for dtype_name, dtype in dtype_map.items():
                for memory_name in memories:
                    record = {
                        "shape": asdict(shape),
                        "representative": shape.representative,
                        "dtype": dtype_name,
                        "memory": memory_name,
                        "iterations": args.iters,
                        "repeats": args.repeats,
                    }
                    record.update(
                        run_shape(
                            ttnn,
                            device,
                            shape,
                            dtype=dtype,
                            memory_config=memory_map[memory_name],
                            iters=args.iters,
                            repeats=args.repeats,
                        )
                    )
                    with_efficiency(record, args.peak_tflops)
                    print(_format_line(shape, dtype_name, memory_name, record), flush=True)
                    results.append(record)
    finally:
        ttnn.close_device(device)

    payload = {
        "environment": environment,
        "peak_tflops": args.peak_tflops,
        "peak_note": args.peak_note,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
