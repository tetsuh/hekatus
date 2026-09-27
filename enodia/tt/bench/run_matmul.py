"""Run stock and hand-written shape-catalogue operations on the accelerator.

The stock path imports only the shape catalogue.  The custom Newton-Schulz
path is selected for the first throughput payload and is kept beside the
stock row in the same result file.
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

CUSTOM_KIND = "custom_newton_schulz"
STOCK_KIND = "ttnn.matmul"
_CUSTOM_TARGET = ("newton_schulz", 32, 32, 32, 8192)


def _make_tensor(ttnn, shape: tuple[int, ...], dtype, layout, device, memory_config):
    """Allocate a device tensor, tolerating differences in the creation API."""
    attempts = []
    for name in ("rand", "ones", "zeros"):
        factory = getattr(ttnn, name, None)
        if factory is None:
            continue
        try:
            tensor = factory(
                shape, dtype=dtype, layout=layout, device=device, memory_config=memory_config
            )
        except Exception as exc:  # noqa: BLE001 - the API surface is under test
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        return tensor, name
    raise RuntimeError("no usable tensor factory; tried " + " | ".join(attempts))


def _execute_once(ttnn, a, b, real_matmuls: int) -> None:
    """One logical stock operation: execute every charged real matmul."""
    for _ in range(real_matmuls):
        out = ttnn.matmul(a, b)
        ttnn.deallocate(out)


def _percentile(samples: list[float], quantile: float) -> float:
    """Linear-interpolated percentile without adding a numerical dependency."""
    if not samples:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _timing_fields(samples: list[float]) -> dict:
    return {
        "seconds_per_launch_samples": samples,
        "seconds_per_launch_p50": _percentile(samples, 0.50),
        "seconds_per_launch_p99": _percentile(samples, 0.99),
        "seconds_per_launch_p99_9": _percentile(samples, 0.999),
        "launches_measured": len(samples),
    }


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
    """Execute one stock shape and return its record, including failures."""
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

        # The first execution pays for program compilation and cache
        # population, which is not the steady-state row being quoted.
        _execute_once(ttnn, a, b, shape.real_matmuls)
        ttnn.synchronize_device(device)

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iters):
                _execute_once(ttnn, a, b, shape.real_matmuls)
            ttnn.synchronize_device(device)
            samples.append((time.perf_counter() - start) / iters)

        flops = total_flops(shape)
        record = {
            "status": "ok",
            "kind": STOCK_KIND,
            "seconds_per_iteration": min(samples),
            "seconds_per_iteration_samples": samples,
            "achieved_tflops": flops / min(samples) / 1e12,
            "flops_per_iteration": flops,
            "real_matmuls_per_iteration": shape.real_matmuls,
            "tensor_factory": factory,
        }
        # A stock block is synchronized once after its `iters` launches; the
        # retained sample is therefore the measured per-launch block average.
        record.update(_timing_fields(samples))
        return record
    except Exception as exc:  # noqa: BLE001 - a shape that cannot run is a result
        return {"status": "failed", "kind": STOCK_KIND, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        for tensor in tensors:
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask the result
                pass


def _is_custom_target(shape: MatmulShape) -> bool:
    return (
        shape.family,
        shape.m,
        shape.k,
        shape.n,
        shape.batch,
    ) == _CUSTOM_TARGET


def run_custom_newton_schulz(
    ttnn,
    device,
    shape: MatmulShape,
    *,
    dtype_name: str,
    memory_name: str,
    variant: str,
    iters: int,
    repeats: int,
) -> dict:
    """Run one prepared fixed-count custom inverse and retain launch samples."""
    if not _is_custom_target(shape):
        return {
            "status": "failed",
            "kind": CUSTOM_KIND,
            "error": "the first custom row is only defined for L=32 batch=8192",
        }
    if dtype_name != "bfloat16":
        return {
            "status": "failed",
            "kind": CUSTOM_KIND,
            "error": "custom rows require bfloat16 R inputs",
        }
    if variant not in {"bf16", "bf16-fp32state"}:
        return {
            "status": "failed",
            "kind": CUSTOM_KIND,
            "error": f"unknown custom variant {variant!r}",
        }
    if memory_name != "l1":
        return {
            "status": "failed",
            "kind": CUSTOM_KIND,
            "error": "custom input/compute memory must be l1",
        }

    from enodia.tt.bench.newton_schulz_kernel import (
        COMPLEX_MATMULS_PER_INVERSE,
        NewtonSchulzKernel,
        benchmark_matrices,
    )

    kernel = None
    try:
        matrices = benchmark_matrices(shape.batch, shape.m, seed=6300)
        kernel = NewtonSchulzKernel.prepare(ttnn, device, matrices, variant=variant)
        kernel.launch()
        ttnn.synchronize_device(device)

        # A timed block is one program launch plus one synchronization.  This
        # keeps each retained sample a true per-launch duration and avoids
        # adding hidden synchronizations inside a multi-launch block.
        launch_samples: list[float] = []
        for _ in range(repeats):
            for _ in range(iters):
                launch_start = time.perf_counter()
                kernel.launch()
                ttnn.synchronize_device(device)
                launch_samples.append(time.perf_counter() - launch_start)

        block_samples = launch_samples
        best = min(block_samples)
        flops = total_flops(shape) * COMPLEX_MATMULS_PER_INVERSE
        record = {
            "status": "ok",
            "kind": CUSTOM_KIND,
            "variant": variant,
            "output_memory": kernel.output_memory,
            "seconds_per_iteration": best,
            "seconds_per_iteration_samples": block_samples,
            "achieved_tflops": flops / best / 1e12,
            "flops_per_iteration": flops,
            "complex_matmuls_per_iteration": COMPLEX_MATMULS_PER_INVERSE,
            "real_matmuls_per_iteration": shape.real_matmuls * COMPLEX_MATMULS_PER_INVERSE,
            "core_work_ranges": [list(pair) for pair in kernel.work_ranges],
        }
        record.update(_timing_fields(launch_samples))
        return record
    except Exception as exc:  # noqa: BLE001 - a device failure is a result
        return {"status": "failed", "kind": CUSTOM_KIND, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if kernel is not None:
            kernel.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", action="append", default=None, help="repeatable")
    parser.add_argument("--memory", action="append", default=None, choices=["dram", "l1"])
    parser.add_argument("--kind", action="append", choices=[STOCK_KIND, CUSTOM_KIND], default=None)
    parser.add_argument(
        "--custom-variant",
        choices=["bf16", "bf16-fp32state"],
        default="bf16",
        help="state precision for custom_newton_schulz rows",
    )
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
    """Reject controls that would produce nonsense before opening a device."""
    if args.iters < 1:
        parser.error(f"--iters must be at least 1, got {args.iters}")
    if args.repeats < 1:
        parser.error(f"--repeats must be at least 1, got {args.repeats}")
    if args.peak_tflops is not None and not (
        math.isfinite(args.peak_tflops) and args.peak_tflops > 0
    ):
        parser.error(f"--peak-tflops must be positive and finite, got {args.peak_tflops}")


def _format_line(shape: MatmulShape, dtype_name: str, memory_name: str, record: dict) -> str:
    kind = record.get("kind", STOCK_KIND)
    line = f"{shape.name:38s} {kind:24s} {dtype_name:9s} {memory_name:4s} "
    if record["status"] != "ok":
        return line + f"failed: {record['error'][:60]}"
    line += f"{record['achieved_tflops']:8.2f} TFLOPS"
    if "efficiency" in record:
        line += f"  {record['efficiency'] * 100:5.1f}%"
    if "seconds_per_launch_p99_9" in record:
        line += (
            f"  P50={record['seconds_per_launch_p50']:.6g}s"
            f" P99={record['seconds_per_launch_p99']:.6g}s"
            f" P99.9={record['seconds_per_launch_p99_9']:.6g}s"
        )
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

    environment = {"python": platform.python_version()}
    if args.env_json and args.env_json.exists():
        environment.update(json.loads(args.env_json.read_text()))

    device = ttnn.open_device(device_id=args.device_id)
    results = []
    try:
        for shape in catalogue:
            for dtype_name, dtype in dtype_map.items():
                for memory_name in memories:
                    kinds = args.kind or [STOCK_KIND]
                    if (
                        args.kind is None
                        and _is_custom_target(shape)
                        and dtype_name == "bfloat16"
                        and memory_name == "l1"
                    ):
                        kinds = [STOCK_KIND, CUSTOM_KIND]
                    for kind in kinds:
                        record = {
                            "shape": asdict(shape),
                            "representative": shape.representative,
                            "dtype": dtype_name,
                            "memory": memory_name,
                            "iterations": args.iters,
                            "repeats": args.repeats,
                            "kind": kind,
                        }
                        if kind == STOCK_KIND:
                            measured = run_shape(
                                ttnn,
                                device,
                                shape,
                                dtype=dtype,
                                memory_config=memory_map[memory_name],
                                iters=args.iters,
                                repeats=args.repeats,
                            )
                        else:
                            measured = run_custom_newton_schulz(
                                ttnn,
                                device,
                                shape,
                                dtype_name=dtype_name,
                                memory_name=memory_name,
                                variant=args.custom_variant,
                                iters=args.iters,
                                repeats=args.repeats,
                            )
                        record.update(measured)
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
