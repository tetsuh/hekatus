"""Run the shape catalogue on the accelerator and record what it achieved.

Runs inside the toolchain container, so it imports nothing from the
reference implementation — only the standard library and the modules beside
it, which are standard-library-only by design.

**Accounting matches execution.** A complex operation costs four real
matmuls, and this runs four. Counting four and timing one would report four
times the achieved throughput, silently, in the direction that flatters.

**What it measures.** A block of iterations is timed with a single
synchronization at the end, so per-iteration cost is not swamped by
synchronization on the small shapes. The block is repeated, the best is
reported, and every repeat is kept beside it: the best keeps scheduler noise
out of the throughput figure, and the spread of the rest is what says whether
that figure is stable enough to quote.
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
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):  # invoked as a plain script inside the container
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.tt.bench.configs import (
    P150_DRAM_BANKS,
    ProgramConfigSpec,
    configuration_catalogue,
    executed_shape,
)
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


@dataclass(frozen=True)
class _RuntimePlan:
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    a_memory_config: object
    b_memory_config: object
    output_memory_config: object
    program_config: object | None
    memory_placement: dict


def _build_program_config(ttnn, config: ProgramConfigSpec):
    common = {
        "in0_block_w": config.in0_block_w,
        "out_subblock_h": config.out_subblock_h,
        "out_subblock_w": config.out_subblock_w,
        "per_core_M": config.per_core_m,
        "per_core_N": config.per_core_n,
    }
    if config.kind == "reuse":
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(config.grid[0], config.grid[1]),
            **common,
        )
    if config.kind == "mcast_1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(config.grid[0], config.grid[1]),
            out_block_h=config.out_block_h,
            out_block_w=config.out_block_w,
            fuse_batch=config.fuse_batch,
            mcast_in0=config.mcast_in0,
            **common,
        )
    if config.kind == "mcast_2d":
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(config.grid[0], config.grid[1]),
            out_block_h=config.out_block_h,
            out_block_w=config.out_block_w,
            transpose_mcast=config.transpose_mcast,
            fuse_batch=config.fuse_batch,
            **common,
        )

    dram_common = {
        "in0_block_w": config.in0_block_w,
        "per_core_M": config.per_core_m,
        "per_core_N": config.per_core_n,
    }
    if config.kind == "dram_sharded":
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(**dram_common)
    if config.kind == "batched_dram_sharded":
        return ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(**dram_common)
    raise ValueError(f"unknown program config kind: {config.kind}")


def _dram_shard_grid(ttnn, banks: int):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})


def _worker_shard_grid(ttnn, workers):
    # Order is part of the batched-DRAM contract: the factory pairs this list
    # with the device's DRAM-bank-to-worker assignment in the same order.
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(core.x, core.y), ttnn.CoreCoord(core.x, core.y))
            for core in workers
        ]
    )


def _memory_config(ttnn, layout, buffer_type, grid, shard_shape):
    return ttnn.MemoryConfig(
        layout,
        buffer_type,
        ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _batched_dram_runtime_plan(ttnn, device, shape, config, program_config):
    workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    if len(workers) != P150_DRAM_BANKS:
        raise RuntimeError(
            f"catalogue expects {P150_DRAM_BANKS} p150 DRAM workers, device reported {len(workers)}"
        )
    batch_per_bank = shape.batch // P150_DRAM_BANKS
    m_padded = math.ceil(shape.m / 32) * 32
    k_padded = math.ceil(shape.k / 32) * 32
    n_padded = math.ceil(shape.n / 32) * 32
    worker_grid = _worker_shard_grid(ttnn, workers)
    dram_grid = _dram_shard_grid(ttnn, P150_DRAM_BANKS)
    a_memory = _memory_config(
        ttnn,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        worker_grid,
        [batch_per_bank * m_padded, k_padded],
    )
    b_memory = _memory_config(
        ttnn,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        dram_grid,
        [batch_per_bank * k_padded, n_padded],
    )
    output_memory = _memory_config(
        ttnn,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        worker_grid,
        [batch_per_bank * m_padded, n_padded],
    )
    worker_coords = [[core.x, core.y] for core in workers]
    return _RuntimePlan(
        a_shape=(1, shape.batch, m_padded, k_padded),
        b_shape=(1, shape.batch, k_padded, n_padded),
        a_memory_config=a_memory,
        b_memory_config=b_memory,
        output_memory_config=output_memory,
        program_config=program_config,
        memory_placement={
            "input_a": {
                "buffer": "l1",
                "layout": "height_sharded",
                "worker_cores": worker_coords,
                "shard_shape": [batch_per_bank * m_padded, k_padded],
            },
            "input_b": {
                "buffer": "dram",
                "layout": "height_sharded",
                "dram_banks": P150_DRAM_BANKS,
                "shard_shape": [batch_per_bank * k_padded, n_padded],
            },
            "output": {
                "buffer": "l1",
                "layout": "height_sharded",
                "worker_cores": worker_coords,
                "shard_shape": [batch_per_bank * m_padded, n_padded],
            },
        },
    )


def _dram_sharded_runtime_plan(ttnn, shape, config, program_config):
    grid = ttnn.CoreGrid(y=config.grid[1], x=config.grid[0])
    m_padded = math.ceil(shape.m / 32) * 32
    k_padded = math.ceil(shape.k / 32) * 32
    a_shape = (1, 1, m_padded, k_padded)
    b_shape = (1, 1, k_padded, shape.n)
    a_memory = ttnn.create_sharded_memory_config(
        a_shape,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    n_padded = math.ceil(shape.n / (32 * P150_DRAM_BANKS)) * 32 * P150_DRAM_BANKS
    b_memory = _memory_config(
        ttnn,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        _dram_shard_grid(ttnn, P150_DRAM_BANKS),
        [math.ceil(shape.k / 32) * 32, n_padded // P150_DRAM_BANKS],
    )
    return _RuntimePlan(
        a_shape=a_shape,
        b_shape=b_shape,
        a_memory_config=a_memory,
        b_memory_config=b_memory,
        output_memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=program_config,
        memory_placement={
            "input_a": {
                "buffer": "l1",
                "layout": "width_sharded",
                "grid": list(config.grid),
            },
            "input_b": {
                "buffer": "dram",
                "layout": "width_sharded",
                "dram_banks": P150_DRAM_BANKS,
                "shard_shape": [math.ceil(shape.k / 32) * 32, n_padded // P150_DRAM_BANKS],
            },
            "output": {"buffer": "l1", "layout": "width_sharded"},
        },
    )


def _runtime_plan(ttnn, device, shape, config, memory_config, memory_name):
    execution = shape if config is None else executed_shape(shape, config)
    program_config = None if config is None else _build_program_config(ttnn, config)
    if config and config.kind == "batched_dram_sharded":
        return execution, _batched_dram_runtime_plan(
            ttnn, device, execution, config, program_config
        )
    if config and config.kind == "dram_sharded":
        return execution, _dram_sharded_runtime_plan(ttnn, execution, config, program_config)
    return execution, _RuntimePlan(
        a_shape=(execution.batch, 1, execution.m, execution.k),
        b_shape=(execution.batch, 1, execution.k, execution.n),
        a_memory_config=memory_config,
        b_memory_config=memory_config,
        output_memory_config=memory_config,
        program_config=program_config,
        memory_placement={
            name: {"buffer": memory_name, "layout": "interleaved"}
            for name in ("input_a", "input_b", "output")
        },
    )


def _execute_once(ttnn, a, b, real_matmuls: int, plan: _RuntimePlan) -> None:
    """One logical operation: every real matmul the accounting charges for."""
    kwargs = {"memory_config": plan.output_memory_config}
    if plan.program_config is not None:
        kwargs["program_config"] = plan.program_config
    for _ in range(real_matmuls):
        out = ttnn.matmul(a, b, **kwargs)
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
    memory_name: str = "selected",
    program_spec: ProgramConfigSpec | None = None,
    iters: int,
    repeats: int,
) -> dict:
    """Execute one shape/config pair and return its record, including failure."""
    tensors = []
    try:
        execution, plan = _runtime_plan(
            ttnn, device, shape, program_spec, memory_config, memory_name
        )
        a, factory = _make_tensor(
            ttnn,
            plan.a_shape,
            dtype,
            ttnn.TILE_LAYOUT,
            device,
            plan.a_memory_config,
        )
        tensors.append(a)
        b, _ = _make_tensor(
            ttnn,
            plan.b_shape,
            dtype,
            ttnn.TILE_LAYOUT,
            device,
            plan.b_memory_config,
        )
        tensors.append(b)

        # Warm up: the first execution pays for program compilation and cache
        # population, which is real but is not what a steady-state frame costs.
        _execute_once(ttnn, a, b, shape.real_matmuls, plan)
        ttnn.synchronize_device(device)

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iters):
                _execute_once(ttnn, a, b, shape.real_matmuls, plan)
            ttnn.synchronize_device(device)
            samples.append((time.perf_counter() - start) / iters)

        flops = total_flops(execution)
        return {
            "status": "ok",
            "execution_shape": asdict(execution),
            "memory_placement": plan.memory_placement,
            "seconds_per_iteration": min(samples),
            "seconds_per_iteration_samples": samples,
            "achieved_tflops": flops / min(samples) / 1e12,
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
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="repeatable exact-or-substring shape-name filter (OR semantics)",
    )
    parser.add_argument(
        "--config-kind",
        action="append",
        default=None,
        help="repeatable program-config kind filter; excludes default rows (OR semantics)",
    )
    parser.add_argument(
        "--config-mode",
        choices=("all", "default-only"),
        default="all",
        help="run the explicit stock catalogue or default ttnn.matmul rows only",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("bench-results.json"))
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--peak-note", default=None, help="what that peak refers to")
    parser.add_argument("--env-json", type=Path, default=None, help="environment to embed")
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject controls and selections before importing or opening the device."""
    if args.iters < 1:
        parser.error(f"--iters must be at least 1, got {args.iters}")
    if args.repeats < 1:
        parser.error(f"--repeats must be at least 1, got {args.repeats}")
    if args.peak_tflops is not None and not (
        math.isfinite(args.peak_tflops) and args.peak_tflops > 0
    ):
        parser.error(f"--peak-tflops must be positive and finite, got {args.peak_tflops}")

    shapes = _select_shapes(default_catalogue(), args.only)
    if not shapes:
        parser.error(f"no shape matches {args.only!r}")

    if args.config_kind is None:
        return

    known_kinds = sorted(
        {
            config.kind
            for shape in default_catalogue()
            for config in configuration_catalogue(shape)
        }
    )
    unknown = sorted(set(args.config_kind) - set(known_kinds))
    if unknown:
        parser.error(
            f"unknown --config-kind value(s): {unknown}; choose from {known_kinds}"
        )
    if args.config_mode == "default-only":
        parser.error("--config-kind cannot be combined with --config-mode default-only")

    rows = [
        config
        for shape in shapes
        if shape.representative
        for config in configuration_catalogue(shape)
        if config.kind in args.config_kind
    ]
    if not rows:
        parser.error(
            "no explicit catalogue rows match "
            f"shape filter(s) {args.only!r} and --config-kind {args.config_kind!r}"
        )


def _select_shapes(shapes: list[MatmulShape], selectors: list[str] | None) -> list[MatmulShape]:
    """Select shapes by exact name or substring, preserving catalogue order."""
    if not selectors:
        return shapes
    return [shape for shape in shapes if any(selector in shape.name for selector in selectors)]


def _row_specs(
    shape: MatmulShape,
    memories: list[str],
    config_mode: str,
    config_kinds: list[str] | None = None,
):
    if config_kinds is None:
        for memory_name in memories:
            yield None, memory_name, memory_name
    if config_mode == "default-only" or not shape.representative:
        return
    for config in configuration_catalogue(shape):
        if config_kinds is not None and config.kind not in config_kinds:
            continue
        if config.memory_plan == "interleaved":
            for memory_name in memories:
                yield config, memory_name, memory_name
        else:
            yield config, config.memory_plan, "dram"


def _format_line(
    shape: MatmulShape,
    dtype_name: str,
    memory_name: str,
    config_name: str,
    record: dict,
) -> str:
    line = f"{shape.name:38s} {dtype_name:9s} {memory_name:20s} {config_name:42s} "
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

    catalogue = _select_shapes(default_catalogue(), args.only)
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
                for program_spec, memory_name, base_memory_name in _row_specs(
                    shape, memories, args.config_mode, args.config_kind
                ):
                    config_record = (
                        {"name": "default", "kind": "default"}
                        if program_spec is None
                        else asdict(program_spec)
                    )
                    execution = (
                        shape if program_spec is None else executed_shape(shape, program_spec)
                    )
                    record = {
                        "shape": asdict(shape),
                        "execution_shape": asdict(execution),
                        "representative": shape.representative,
                        "dtype": dtype_name,
                        "memory": memory_name,
                        "memory_placement": {"plan": memory_name},
                        "program_config": config_record,
                        "iterations": args.iters,
                        "repeats": args.repeats,
                    }
                    record.update(
                        run_shape(
                            ttnn,
                            device,
                            shape,
                            dtype=dtype,
                            memory_config=memory_map[base_memory_name],
                            memory_name=memory_name,
                            program_spec=program_spec,
                            iters=args.iters,
                            repeats=args.repeats,
                        )
                    )
                    with_efficiency(record, args.peak_tflops)
                    print(
                        _format_line(
                            shape,
                            dtype_name,
                            memory_name,
                            config_record["name"],
                            record,
                        ),
                        flush=True,
                    )
                    results.append(record)
    finally:
        ttnn.close_device(device)

    payload = {
        "environment": environment,
        "configuration_mode": args.config_mode,
        "selection": {
            "shape_filters": args.only or [],
            "program_config_kind_filters": args.config_kind or [],
        },
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
