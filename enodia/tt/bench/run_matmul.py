"""Run the shape catalogue on the accelerator and record what it achieved.

Runs inside the toolchain container, so it imports nothing from the
reference implementation — only the standard library and the shape
catalogue beside it, which is standard-library-only by design.

**What it measures.** A block of iterations is timed with a single
synchronization at the end, so the per-iteration cost is not swamped by
synchronization overhead on the small shapes; the block is repeated and the
best block is reported, which is the usual way to keep scheduler noise out
of a throughput figure. Achieved FLOPS come from the catalogue's accounting,
never from a vendor counter.

**Failures are results.** A shape that will not fit in L1 fails here, and
that failure is recorded rather than aborting the run. Where the boundary
falls is the answer to the question design.md §2 calls paramount — whether
the data fits on-chip — so it is data, not an error.

**Efficiency is optional.** Without an explicit peak on the command line,
only achieved FLOPS are reported. An efficiency figure quoted against the
wrong peak is worse than no efficiency figure, so the peak and the note
describing it are recorded next to every number derived from them.
"""

from __future__ import annotations

import argparse
import json
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
    that works is recorded with the result.
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
        out = ttnn.matmul(a, b)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

        best = None
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iters):
                out = ttnn.matmul(a, b)
            ttnn.synchronize_device(device)
            elapsed = (time.perf_counter() - start) / iters
            ttnn.deallocate(out)
            best = elapsed if best is None else min(best, elapsed)

        flops = total_flops(shape)
        return {
            "status": "ok",
            "seconds_per_iteration": best,
            "achieved_tflops": flops / best / 1e12,
            "flops_per_iteration": flops,
            "tensor_factory": factory,
        }
    except Exception as exc:  # noqa: BLE001 - a shape that cannot run is a result
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        for t in tensors:
            try:
                ttnn.deallocate(t)
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask the result
                pass


def main() -> int:
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
    args = parser.parse_args()

    import ttnn  # imported here so --help works without an accelerator present

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
                    if args.peak_tflops and record.get("achieved_tflops"):
                        record["efficiency"] = record["achieved_tflops"] / args.peak_tflops
                    line = f"{shape.name:38s} {dtype_name:9s} {memory_name:4s} "
                    if record["status"] == "ok":
                        line += f"{record['achieved_tflops']:8.2f} TFLOPS"
                        if "efficiency" in record:
                            line += f"  {record['efficiency'] * 100:5.1f}%"
                    else:
                        line += f"failed: {record['error'][:60]}"
                    print(line, flush=True)
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
