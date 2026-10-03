"""Run the shape catalogue on the accelerator and record what it achieved.

Runs inside the toolchain container, so it imports nothing from the
reference implementation. The explicit stock catalogue remains host-reviewable;
the custom correctness catalogue uses NumPy only for its independent gate.

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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # invoked as a plain script inside the container
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.configs import (
    P150_COMPUTE_GRID,
    P150_DRAM_BANKS,
    ProgramConfigSpec,
    configuration_catalogue,
    executed_shape,
)
from enodia.tt.bench.newton_schulz_kernel import (
    INPUT_MEMORY_CHOICES,
    MATRIX_BLOCK_CHOICES,
    _normalize_fidelity_split,
    _resolve_input_memories,
)
from enodia.tt.bench.profiling import parse_device_profile_csv
from enodia.tt.bench.shapes import MatmulShape, default_catalogue, total_flops

CUSTOM_KIND = "custom_newton_schulz"
STOCK_KIND = "ttnn.matmul"
_CUSTOM_TARGETS = frozenset(
    {
        ("newton_schulz", 16, 16, 16, 8192),
        ("newton_schulz", 32, 32, 32, 8192),
    }
)
CUSTOM_MATH_FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
STOCK_FIDELITY_SOURCE = (
    "tt-metal ttnn/operations/matmul/device/matmul_device_operation.cpp "
    "create_matmul_attributes source mapping; no board observation"
)
ACCEPTANCE_CATALOGUE_SHAPES = (
    "newton_schulz_L32_b8192",
    "newton_schulz_L16_b8192",
)
ACCEPTANCE_CATALOGUE_MATRIX_BLOCKS = (1, 4)
ACCEPTANCE_CATALOGUE_FIDELITIES = ("HiFi3", "HiFi4")
ACCEPTANCE_CATALOGUE_LAUNCHES = 1000
POWER_TRACE_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")


def _stock_math_fidelity(dtype_name: str, program_spec: ProgramConfigSpec | None) -> dict:
    """Return source-derived stock fidelity metadata without claiming hardware evidence."""
    if dtype_name == "bfloat16":
        if program_spec is None:
            return {
                "math_fidelity": "HiFi2",
                "math_fidelity_source": (
                    f"{STOCK_FIDELITY_SOURCE}; BF16 default has no program_config/user_grid, "
                    "so increase_fidelity selects HiFi2"
                ),
            }
        return {
            "math_fidelity": "LoFi",
            "math_fidelity_source": (
                f"{STOCK_FIDELITY_SOURCE}; BF16 explicit program_config disables "
                "increase_fidelity and selects LoFi"
            ),
        }
    if dtype_name == "float32":
        return {
            "math_fidelity": "HiFi4",
            "math_fidelity_source": (
                f"{STOCK_FIDELITY_SOURCE}; FP32 non-Wormhole override selects HiFi4 "
                "(architecture-specific source mapping)"
            ),
        }
    return {
        "math_fidelity": "unknown",
        "math_fidelity_source": f"{STOCK_FIDELITY_SOURCE}; dtype mapping not modeled",
    }


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
        record = {
            "status": "ok",
            "kind": STOCK_KIND,
            "execution_shape": asdict(execution),
            "memory_placement": plan.memory_placement,
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
    ) in _CUSTOM_TARGETS


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
    math_fidelity: str = "HiFi4",
    fidelity_split: str | tuple[int, int] | list[int] | None = None,
    profile: bool = False,
    fuse_s: bool = False,
    two_tile_complex: bool = False,
    batch_reads: bool = False,
    matrix_block: int = 1,
    fp32_dest_acc_en: bool = True,
    dst_full_sync_en: bool = True,
    input_memory: str = "l1",
    r_memory: str | None = None,
    x0_memory: str | None = None,
    row_name: str | None = None,
) -> dict:
    """Run one prepared fixed-count custom inverse and retain launch samples."""
    explicit_r_memory = r_memory is not None
    explicit_x0_memory = x0_memory is not None
    try:
        input_memory, r_memory, x0_memory = _resolve_input_memories(
            input_memory, r_memory=r_memory, x0_memory=x0_memory
        )
    except ValueError as exc:
        return {"status": "failed", "kind": CUSTOM_KIND, "error": str(exc)}
    if not _is_custom_target(shape):
        return {
            "status": "failed",
            "kind": CUSTOM_KIND,
            "error": "custom rows are only defined for L=16 or L=32 batch=8192",
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
    from enodia.tt.bench.newton_schulz_kernel import (
        _physical_tile_count,
        _state_dtype,
        _validate_l1_preflight,
        _validate_matrix_block,
    )

    try:
        _validate_matrix_block(
            matrix_block,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            variant=variant,
        )
    except ValueError as exc:
        return {"status": "failed", "kind": CUSTOM_KIND, "error": str(exc)}
    try:
        fidelity_split = _normalize_fidelity_split(fidelity_split)
    except ValueError as exc:
        return {"status": "failed", "kind": CUSTOM_KIND, "error": str(exc)}
    if fidelity_split is not None and math_fidelity != "HiFi3":
        return {
            "status": "failed",
            "kind": CUSTOM_KIND,
            "error": "fidelity split requires math_fidelity=HiFi3",
        }
    if math_fidelity not in CUSTOM_MATH_FIDELITIES:
        return {
            "status": "failed",
            "kind": CUSTOM_KIND,
            "error": f"unknown math fidelity {math_fidelity!r}",
        }
    if memory_name != "l1":
        return {
            "status": "failed",
            "kind": CUSTOM_KIND,
            "error": "custom input/compute memory must be l1",
        }
    try:
        l1_preflight_bytes = _validate_l1_preflight(
            ttnn,
            batch=_physical_tile_count(shape.batch, shape.m),
            core_count=P150_COMPUTE_GRID[0] * P150_COMPUTE_GRID[1],
            state_dtype=_state_dtype(ttnn, variant),
            profile=profile,
            fuse_s=fuse_s,
            output_memory="dram" if variant == "bf16-fp32state" else "l1",
            input_memory=input_memory,
            r_memory=r_memory,
            x0_memory=x0_memory,
            matrix_block=matrix_block,
            variant=variant,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            two_tile_complex=two_tile_complex,
        )
    except ValueError as exc:
        return {"status": "failed", "kind": CUSTOM_KIND, "error": str(exc)}

    from enodia.tt.bench.newton_schulz_kernel import (
        COMPLEX_MATMULS_PER_INVERSE,
        NewtonSchulzKernel,
        benchmark_matrices,
    )

    kernel = None
    try:
        matrices = benchmark_matrices(shape.batch, shape.m, seed=6300)
        prepare_kwargs = {
            "variant": variant,
            "math_fidelity": math_fidelity,
            "profile": profile,
            "fuse_s": fuse_s,
            "batch_reads": batch_reads,
        }
        if input_memory != "l1":
            prepare_kwargs["input_memory"] = input_memory
        if explicit_r_memory:
            prepare_kwargs["r_memory"] = r_memory
        if explicit_x0_memory:
            prepare_kwargs["x0_memory"] = x0_memory
        if fidelity_split is not None:
            prepare_kwargs["fidelity_split"] = fidelity_split
        # Keep the baseline dispatch signature intact for callers that provide
        # a legacy host stub; non-default blocks must be explicit.
        if matrix_block != 1:
            prepare_kwargs["matrix_block"] = matrix_block
        if two_tile_complex:
            prepare_kwargs["two_tile_complex"] = True
        if not fp32_dest_acc_en:
            prepare_kwargs["fp32_dest_acc_en"] = False
        if not dst_full_sync_en:
            prepare_kwargs["dst_full_sync_en"] = False
        kernel = NewtonSchulzKernel.prepare(ttnn, device, matrices, **prepare_kwargs)
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

        best = min(launch_samples)
        flops = total_flops(shape) * COMPLEX_MATMULS_PER_INVERSE
        record = {
            "status": "ok",
            "kind": CUSTOM_KIND,
            "variant": variant,
            "math_fidelity": math_fidelity,
            "fidelity_split": (
                None if fidelity_split is None else f"{fidelity_split[0]}+{fidelity_split[1]}"
            ),
            "fuse_s": fuse_s,
            "two_tile_complex": two_tile_complex,
            "batch_reads": batch_reads,
            "matrix_block": matrix_block,
            "fp32_dest_acc_en": fp32_dest_acc_en,
            "dst_full_sync_en": dst_full_sync_en,
            "row": row_name or f"custom_block{matrix_block}",
            "physical_tile_count": getattr(
                kernel, "tile_count", _physical_tile_count(shape.batch, shape.m)
            ),
            "packing": "diagonal_pairs_32x32" if shape.m == 16 else "native_32x32",
            "input_memory": input_memory,
            "r_memory": r_memory,
            "x0_memory": x0_memory,
            "l1_preflight_bytes": l1_preflight_bytes,
            "output_memory": kernel.output_memory,
            "seconds_per_iteration": best,
            "seconds_per_iteration_samples": launch_samples,
            "achieved_tflops": flops / best / 1e12,
            "flops_per_iteration": flops,
            "complex_matmuls_per_iteration": COMPLEX_MATMULS_PER_INVERSE,
            "complex_product_tiles": 2 if two_tile_complex else 1,
            "complex_product_matmul_block_calls_per_product": 2 if two_tile_complex else 4,
            "real_matmuls_per_iteration": shape.real_matmuls * COMPLEX_MATMULS_PER_INVERSE,
            "core_work_ranges": [list(pair) for pair in kernel.work_ranges],
        }
        record.update(_timing_fields(launch_samples))
        if profile:
            record["profile_mode"] = "l1_cycle_counters"
            record["measurement_core"] = 0
            record["profile_clock"] = "get_timestamp_32b_lower_32_wall_clock"
            record["profile_sampling"] = "all_tiles_all_iterations_with_core_0_first_tile_iteration_warmup"
            record["profile_aggregation"] = "core_0_per_risc_reader_compute_writer_pages"
            profile_records = kernel.profile_records()
            record["profile_records"] = profile_records
            record["profile_transport_complete"] = all(
                entry["profile_page_ready"] for entry in profile_records
            )
            record["profile_consistency"] = {
                "all_scopes_pass": all(entry["consistency_pass"] for entry in profile_records),
                "warmup_scopes_pass": all(
                    entry["warmup_consistency_pass"] for entry in profile_records
                ),
                "named_sections_cover_total": all(
                    entry["named_sections_cover_total"] for entry in profile_records
                ),
                "warmup_named_sections_cover_total": all(
                    entry["warmup_named_sections_cover_total"] for entry in profile_records
                ),
            }
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
    parser.add_argument(
        "--input-memory",
        choices=INPUT_MEMORY_CHOICES,
        default="l1",
        help="compatibility placement for custom-kernel inputs",
    )
    parser.add_argument(
        "--r-memory",
        choices=INPUT_MEMORY_CHOICES,
        default=None,
        help="interleaved placement for all R reader tensors",
    )
    parser.add_argument(
        "--x0-memory",
        choices=INPUT_MEMORY_CHOICES,
        default=None,
        help="interleaved placement for both X0 reader tensors",
    )
    parser.add_argument("--kind", action="append", choices=[STOCK_KIND, CUSTOM_KIND], default=None)
    parser.add_argument(
        "--custom-variant",
        choices=["bf16", "bf16-fp32state"],
        default="bf16",
        help="state precision for custom_newton_schulz rows",
    )
    parser.add_argument(
        "--custom-math-fidelity",
        action="append",
        choices=list(CUSTOM_MATH_FIDELITIES),
        default=None,
        help="repeatable custom math fidelity; default is HiFi4",
    )
    parser.add_argument(
        "--fidelity-split",
        action="append",
        type=_normalize_fidelity_split,
        default=None,
        help=(
            "repeatable explicit HiFi2+HiFi3 iteration split (for example 4+4 or 6+2); "
            "0+8 selects the opt-in all-HiFi3 direct-LLK path"
        ),
    )
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
    parser.add_argument(
        "--power-trace",
        type=Path,
        default=None,
        help="companion tt-smi CSV; its schema is recorded in the measurement metadata",
    )
    parser.add_argument(
        "--acceptance-catalogue",
        action="store_true",
        help=(
            "run the single-run Issue #63 L16/L32 catalogue: stock_best plus "
            "block 1/4, fuse_s false/true, and HiFi3/HiFi4 custom rows"
        ),
    )
    parser.add_argument(
        "--complex-product-catalogue",
        action="store_true",
        help=(
            "gate and compare stock, current one-tile, and opt-in two-tile "
            "products for full/half DEST sync"
        ),
    )
    parser.add_argument(
        "--complex-product-batch",
        type=int,
        default=8192,
        help="batch size for --complex-product-catalogue (default: 8192)",
    )
    parser.add_argument(
        "--complex-product-correctness-only",
        action="store_true",
        help="run only correctness rows for --complex-product-catalogue",
    )
    parser.add_argument(
        "--launches-per-row",
        type=int,
        default=ACCEPTANCE_CATALOGUE_LAUNCHES,
        help="timed launches per row for --acceptance-catalogue (default: 1000)",
    )
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--peak-note", default=None, help="what that peak refers to")
    parser.add_argument("--env-json", type=Path, default=None, help="environment to embed")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable L1 cycle-counter profiling",
    )
    parser.add_argument(
        "--fuse-s",
        action="store_true",
        help="fuse S = 2I - R @ X into BF16/FP32 DEST accumulation",
    )
    parser.add_argument(
        "--two-tile-complex",
        action="store_true",
        help="use the opt-in two-tile complex products",
    )
    parser.add_argument(
        "--fp32-dest-acc-en",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable FP32 DEST accumulation (default: enabled)",
    )
    parser.add_argument(
        "--dst-full-sync-en",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable full DEST synchronization (default: enabled)",
    )
    parser.add_argument(
        "--batch-reads",
        action="store_true",
        help="group each matrix's reader NoC reads behind one barrier",
    )
    parser.add_argument(
        "--matrix-block",
        type=int,
        choices=MATRIX_BLOCK_CHOICES,
        default=1,
        help="number of independent matrices processed per compute block",
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=None,
        help="parse a TT-Metal profile_log_device.csv into the result payload",
    )
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
    if args.launches_per_row < 1:
        parser.error(
            f"--launches-per-row must be at least 1, got {args.launches_per_row}"
        )
    if args.fidelity_split and args.custom_math_fidelity and any(
        math_fidelity != "HiFi3" for math_fidelity in args.custom_math_fidelity
    ):
        parser.error(
            "--fidelity-split can only be combined with --custom-math-fidelity HiFi3"
        )
    if args.complex_product_batch < 1:
        parser.error(
            f"--complex-product-batch must be positive, got {args.complex_product_batch}"
        )
    if args.complex_product_correctness_only and not args.complex_product_catalogue:
        parser.error("--complex-product-correctness-only requires --complex-product-catalogue")
    if args.acceptance_catalogue or args.complex_product_catalogue:
        if args.acceptance_catalogue and args.complex_product_catalogue:
            parser.error("catalogue modes are mutually exclusive")
        if args.fidelity_split:
            parser.error("--fidelity-split cannot be combined with a catalogue mode")
        if args.complex_product_catalogue and args.two_tile_complex:
            parser.error("--complex-product-catalogue selects both product forms")
        return

    shapes = _select_shapes(default_catalogue(), args.only)
    if not shapes:
        parser.error(f"no shape matches {args.only!r}")

    if args.config_kind is None:
        return

    requested_dtypes = args.dtype or ["bfloat16", "float32"]
    known_kinds = sorted(
        {
            config.kind
            for shape in default_catalogue()
            for dtype in requested_dtypes
            for config in configuration_catalogue(shape, dtype=dtype)
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
        for dtype in requested_dtypes
        for config in configuration_catalogue(shape, dtype=dtype)
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
    dtype: str = "bfloat16",
):
    if config_kinds is None:
        for memory_name in memories:
            yield None, memory_name, memory_name
    if config_mode == "default-only" or not shape.representative:
        return
    for config in configuration_catalogue(shape, dtype=dtype):
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
    kind = record.get("kind", STOCK_KIND)
    line = (
        f"{shape.name:38s} {kind:24s} {dtype_name:9s} "
        f"{memory_name:20s} {config_name:42s} "
    )
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


def _record_path(path: Path | None) -> str | None:
    """Store companion paths as sibling filenames, not container mount paths."""
    return None if path is None else path.name


def _acceptance_row_name(matrix_block: int, fuse_s: bool, math_fidelity: str) -> str:
    return (
        f"custom_block{matrix_block}_fuse_s_{str(fuse_s).lower()}_"
        f"{math_fidelity}"
    )


def _acceptance_measurement_metadata(args: argparse.Namespace) -> dict:
    """Describe the fixed Issue #63 acceptance catalogue in the result record."""
    power_trace = _record_path(args.power_trace)
    return {
        "catalogue": "issue_63_newton_schulz_l16_l32_b8192",
        "shape_names": list(ACCEPTANCE_CATALOGUE_SHAPES),
        "same_device_run": True,
        "launches_per_row": args.launches_per_row,
        "stock_best_row": "stock_best",
        "custom_rows": {
            "matrix_blocks": list(ACCEPTANCE_CATALOGUE_MATRIX_BLOCKS),
            "fuse_s": [False, True],
            "math_fidelities": list(ACCEPTANCE_CATALOGUE_FIDELITIES),
            "variant": "bf16-fp32state",
            "dtype": "bfloat16",
            "input_memory": "l1",
            "r_memory": "l1",
            "x0_memory": "l1",
            "output_memory": "dram",
            "two_tile_complex": False,
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": True,
        },
        "power_trace": power_trace,
        "power_clock_provenance": {
            "trace": power_trace,
            "columns": list(POWER_TRACE_COLUMNS),
            "power_column": "power_w",
            "clock_column": "aiclk_mhz",
            "temperature_column": "asic_temp_c",
            "sampling_source": "tt-smi snapshot",
        },
        "environment_provenance": {
            "source": "--env-json",
            "record_field": "environment",
            "adr": "ADR-0005",
        },
    }


def _complex_catalogue_inputs(batch: int, size: int, *, seed: int) -> np.ndarray:
    """Create deterministic HPD inputs without importing the reference module."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((batch, size, size), dtype=np.float32)
    imag = rng.standard_normal((batch, size, size), dtype=np.float32)
    factor = (real + 1j * imag).astype(np.complex64)
    matrices = factor @ factor.conj().swapaxes(-1, -2)
    matrices += np.eye(size, dtype=np.complex64)[None, ...]
    return matrices.astype(np.complex64)


def _complex_catalogue_reference(matrices: np.ndarray) -> np.ndarray:
    """Independent fixed-eight-step NumPy oracle for the catalogue gate."""
    norm_1 = np.linalg.norm(matrices, ord=1, axis=(-2, -1))
    norm_inf = np.linalg.norm(matrices, ord=np.inf, axis=(-2, -1))
    x = matrices.conj().swapaxes(-1, -2) / (norm_1 * norm_inf)[:, None, None]
    identity = np.eye(matrices.shape[-1], dtype=np.complex64)
    for _ in range(8):
        x = x @ (2.0 * identity - matrices @ x)
    return x


def _run_complex_catalogue_correctness(
    ttnn,
    device,
    matrices: np.ndarray,
    *,
    expected: np.ndarray,
    two_tile_complex: bool,
    dst_full_sync_en: bool,
    matrix_block: int,
) -> dict:
    from enodia.tt.bench.newton_schulz_kernel import run_newton_schulz_kernel

    try:
        actual = run_newton_schulz_kernel(
            ttnn,
            device,
            matrices,
            variant="bf16-fp32state",
            math_fidelity="HiFi3",
            fuse_s=not two_tile_complex,
            two_tile_complex=two_tile_complex,
            matrix_block=matrix_block,
            input_memory="dram",
            r_memory="dram",
            x0_memory="dram",
            dst_full_sync_en=dst_full_sync_en,
        )
        relative_error = float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        )
        return {
            "relative_error": relative_error,
            "tolerance": 1e-2,
            "status": "pass" if relative_error <= 1e-2 else "fail",
        }
    except Exception as exc:  # noqa: BLE001 - preserve compile/timeout failures as rows
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _run_complex_product_catalogue(
    ttnn,
    device,
    *,
    args: argparse.Namespace,
    memory_map: dict[str, object],
) -> list[dict]:
    """Gate and time current/two-tile products for both DEST sync modes."""
    shape = next(
        shape for shape in default_catalogue() if shape.name == "newton_schulz_L32_b8192"
    )
    shape = replace(
        shape,
        name=f"newton_schulz_L32_b{args.complex_product_batch}",
        batch=args.complex_product_batch,
    )
    launches = args.launches_per_row
    matrices = _complex_catalogue_inputs(shape.batch, shape.m, seed=95)
    expected = _complex_catalogue_reference(matrices)
    results: list[dict] = []

    if not args.complex_product_correctness_only:
        stock_record = {
            "shape": asdict(shape),
            "execution_shape": asdict(shape),
            "representative": shape.representative,
            "dtype": "bfloat16",
            "memory": "l1",
            "input_memory": "l1",
            "memory_placement": {name: {"buffer": "l1", "layout": "interleaved"}
                                 for name in ("input_a", "input_b", "output")},
            "program_config": {"name": "default", "kind": "default"},
            "iterations": 1,
            "repeats": launches,
            "launches_requested_per_row": launches,
            "kind": STOCK_KIND,
            "row": "stock_best",
            **_stock_math_fidelity("bfloat16", None),
        }
        stock_record.update(
            run_shape(
                ttnn,
                device,
                shape,
                dtype=ttnn.bfloat16,
                memory_config=memory_map["l1"],
                memory_name="l1",
                iters=1,
                repeats=launches,
            )
        )
        with_efficiency(stock_record, args.peak_tflops)
        results.append(stock_record)

    for dst_full_sync_en, matrix_block in ((True, 4), (False, 2)):
        for two_tile_complex in (False, True):
            label = "two_tile" if two_tile_complex else "one_tile"
            row_name = f"{label}_{'full' if dst_full_sync_en else 'half'}_sync_block{matrix_block}"
            correctness = _run_complex_catalogue_correctness(
                ttnn,
                device,
                matrices,
                expected=expected,
                two_tile_complex=two_tile_complex,
                dst_full_sync_en=dst_full_sync_en,
                matrix_block=matrix_block,
            )
            row = {
                "shape": asdict(shape),
                "execution_shape": asdict(shape),
                "representative": shape.representative,
                "dtype": "bfloat16",
                "memory": "l1",
                "input_memory": "dram",
                "r_memory": "dram",
                "x0_memory": "dram",
                "memory_placement": {
                    "input": "dram",
                    "r": "dram",
                    "x0": "dram",
                    "compute": "l1",
                },
                "program_config": {
                    "name": CUSTOM_KIND,
                    "kind": CUSTOM_KIND,
                    "variant": "bf16-fp32state",
                    "math_fidelity": "HiFi3",
                    "fuse_s": not two_tile_complex,
                    "two_tile_complex": two_tile_complex,
                    "batch_reads": False,
                    "matrix_block": matrix_block,
                    "fp32_dest_acc_en": True,
                    "dst_full_sync_en": dst_full_sync_en,
                    "input_memory": "dram",
                    "r_memory": "dram",
                    "x0_memory": "dram",
                },
                "iterations": 1,
                "repeats": launches,
                "launches_requested_per_row": launches,
                "kind": CUSTOM_KIND,
                "row": row_name,
                "correctness": correctness,
                "measurement_mode": (
                    "correctness-only"
                    if args.complex_product_correctness_only
                    else "correctness-and-throughput"
                ),
            }
            if correctness["status"] == "pass" and not args.complex_product_correctness_only:
                row.update(
                    run_custom_newton_schulz(
                        ttnn,
                        device,
                        shape,
                        dtype_name="bfloat16",
                        memory_name="l1",
                        variant="bf16-fp32state",
                        math_fidelity="HiFi3",
                        fuse_s=not two_tile_complex,
                        two_tile_complex=two_tile_complex,
                        batch_reads=False,
                        matrix_block=matrix_block,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=dst_full_sync_en,
                        input_memory="dram",
                        r_memory="dram",
                        x0_memory="dram",
                        row_name=row_name,
                        iters=1,
                        repeats=launches,
                    )
                )
            else:
                row.update({"status": correctness["status"], "kind": CUSTOM_KIND})
                if "error" in correctness:
                    row["error"] = correctness["error"]
            with_efficiency(row, args.peak_tflops)
            results.append(row)
    return results


def _run_acceptance_catalogue(
    ttnn,
    device,
    *,
    args: argparse.Namespace,
    memory_map: dict[str, object],
) -> list[dict]:
    """Run the board-free-dispatchable, single-device acceptance row plan."""
    shapes_by_name = {shape.name: shape for shape in default_catalogue()}
    shapes = [shapes_by_name[name] for name in ACCEPTANCE_CATALOGUE_SHAPES]
    results: list[dict] = []
    launches = args.launches_per_row

    for shape in shapes:
        stock_record = {
            "shape": asdict(shape),
            "execution_shape": asdict(shape),
            "representative": shape.representative,
            "dtype": "bfloat16",
            "memory": "l1",
            "input_memory": "l1",
            "memory_placement": {name: {"buffer": "l1", "layout": "interleaved"}
                                 for name in ("input_a", "input_b", "output")},
            "program_config": {"name": "default", "kind": "default"},
            "iterations": 1,
            "repeats": launches,
            "launches_requested_per_row": launches,
            "kind": STOCK_KIND,
            "row": "stock_best",
            **_stock_math_fidelity("bfloat16", None),
        }
        stock_record.update(
            run_shape(
                ttnn,
                device,
                shape,
                dtype=ttnn.bfloat16,
                memory_config=memory_map["l1"],
                memory_name="l1",
                iters=1,
                repeats=launches,
            )
        )
        with_efficiency(stock_record, args.peak_tflops)
        print(
            _format_line(shape, "bfloat16", "l1", "stock_best", stock_record),
            flush=True,
        )
        results.append(stock_record)

        for matrix_block in ACCEPTANCE_CATALOGUE_MATRIX_BLOCKS:
            for fuse_s in (False, True):
                for math_fidelity in ACCEPTANCE_CATALOGUE_FIDELITIES:
                    row_name = _acceptance_row_name(matrix_block, fuse_s, math_fidelity)
                    custom_record = {
                        "shape": asdict(shape),
                        "execution_shape": asdict(shape),
                        "representative": shape.representative,
                        "dtype": "bfloat16",
                        "memory": "l1",
                        "input_memory": "l1",
                        "r_memory": "l1",
                        "x0_memory": "l1",
                        "memory_placement": {
                            "input": "l1",
                            "r": "l1",
                            "x0": "l1",
                            "compute": "l1",
                        },
                        "program_config": {
                            "name": CUSTOM_KIND,
                            "kind": CUSTOM_KIND,
                            "variant": "bf16-fp32state",
                            "math_fidelity": math_fidelity,
                            "fuse_s": fuse_s,
                            "two_tile_complex": False,
                            "batch_reads": False,
                            "matrix_block": matrix_block,
                            "fp32_dest_acc_en": True,
                            "dst_full_sync_en": True,
                            "input_memory": "l1",
                            "r_memory": "l1",
                            "x0_memory": "l1",
                        },
                        "iterations": 1,
                        "repeats": launches,
                        "launches_requested_per_row": launches,
                        "kind": CUSTOM_KIND,
                        "row": row_name,
                    }
                    custom_record.update(
                        run_custom_newton_schulz(
                            ttnn,
                            device,
                            shape,
                            dtype_name="bfloat16",
                            memory_name="l1",
                            variant="bf16-fp32state",
                            math_fidelity=math_fidelity,
                            fuse_s=fuse_s,
                            two_tile_complex=False,
                            batch_reads=False,
                            matrix_block=matrix_block,
                            fp32_dest_acc_en=True,
                            dst_full_sync_en=True,
                            input_memory="l1",
                            r_memory="l1",
                            x0_memory="l1",
                            row_name=row_name,
                            iters=1,
                            repeats=launches,
                        )
                    )
                    with_efficiency(custom_record, args.peak_tflops)
                    print(
                        _format_line(shape, "bfloat16", "l1", row_name, custom_record),
                        flush=True,
                    )
                    results.append(custom_record)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    input_memory, r_memory, x0_memory = _resolve_input_memories(
        args.input_memory, r_memory=args.r_memory, x0_memory=args.x0_memory
    )

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
    if args.complex_product_catalogue:
        try:
            results = _run_complex_product_catalogue(
                ttnn,
                device,
                args=args,
                memory_map=memory_map,
            )
        finally:
            ttnn.close_device(device)
        power_trace = _record_path(args.power_trace)
        payload = {
            "environment": environment,
            "configuration_mode": "complex-product-catalogue",
            "selection": {
                "shape_filters": [
                    f"newton_schulz_L32_b{args.complex_product_batch}"
                ],
                "program_config_kind_filters": [],
                "custom_math_fidelity": ["HiFi3"],
                "input_memory": "dram",
                "r_memory": "dram",
                "x0_memory": "dram",
                "two_tile_complex": [False, True],
                "fp32_dest_acc_en": True,
                "dst_full_sync_en": [True, False],
                "matrix_blocks": {"full": 4, "half": 2},
                "batch": args.complex_product_batch,
                "correctness_only": args.complex_product_correctness_only,
                "launches_per_row": args.launches_per_row,
            },
            "peak_tflops": args.peak_tflops,
            "peak_note": args.peak_note,
            "measurement": {
                "catalogue": "two_tile_complex",
                "batch": args.complex_product_batch,
                "correctness_only": args.complex_product_correctness_only,
                "same_device_run": True,
                "correctness_reference": "independent NumPy fixed-eight-step oracle in this runner",
                "correctness_tolerance": 1e-2,
                "stock_best_row": "stock_best",
                "power_trace": power_trace,
                "power_clock_provenance": {
                    "trace": power_trace,
                    "columns": list(POWER_TRACE_COLUMNS),
                    "power_column": "power_w",
                    "clock_column": "aiclk_mhz",
                    "temperature_column": "asic_temp_c",
                    "sampling_source": "tt-smi snapshot",
                },
                "environment_provenance": {
                    "source": "--env-json",
                    "record_field": "environment",
                    "adr": "ADR-0005",
                },
            },
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(strict_json_dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
        return 0
    if args.acceptance_catalogue:
        try:
            results = _run_acceptance_catalogue(
                ttnn,
                device,
                args=args,
                memory_map=memory_map,
            )
        finally:
            ttnn.close_device(device)
        payload = {
            "environment": environment,
            "configuration_mode": "acceptance-catalogue",
            "selection": {
                "shape_filters": list(ACCEPTANCE_CATALOGUE_SHAPES),
                "program_config_kind_filters": [],
                "custom_math_fidelity": list(ACCEPTANCE_CATALOGUE_FIDELITIES),
                "input_memory": "l1",
                "r_memory": "l1",
                "x0_memory": "l1",
                "fuse_s": [False, True],
                "two_tile_complex": False,
                "fp32_dest_acc_en": True,
                "dst_full_sync_en": True,
                "batch_reads": False,
                "matrix_blocks": list(ACCEPTANCE_CATALOGUE_MATRIX_BLOCKS),
                "launches_per_row": args.launches_per_row,
            },
            "peak_tflops": args.peak_tflops,
            "peak_note": args.peak_note,
            "measurement": _acceptance_measurement_metadata(args),
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(strict_json_dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
        return 0

    results = []
    run_stock = args.kind is None or STOCK_KIND in args.kind
    run_custom = args.kind is None or CUSTOM_KIND in args.kind
    if args.fidelity_split and args.custom_math_fidelity:
        # Keep the legacy all-HiFi3 row beside each opt-in split in one device
        # run when the caller requests both forms explicitly.
        custom_splits = [None, *args.fidelity_split]
        custom_fidelities = args.custom_math_fidelity
    elif args.fidelity_split:
        custom_splits = args.fidelity_split
        custom_fidelities = ["HiFi3"]
    else:
        custom_splits = [None]
        custom_fidelities = args.custom_math_fidelity or ["HiFi4"]
    try:
        for shape in catalogue:
            for dtype_name, dtype in dtype_map.items():
                for program_spec, memory_name, base_memory_name in _row_specs(
                    shape, memories, args.config_mode, args.config_kind, dtype_name
                ):
                    config_record = (
                        {"name": "default", "kind": "default"}
                        if program_spec is None
                        else asdict(program_spec)
                    )
                    execution = (
                        shape if program_spec is None else executed_shape(shape, program_spec)
                    )
                    if run_stock:
                        record = {
                            "shape": asdict(shape),
                            "execution_shape": asdict(execution),
                            "representative": shape.representative,
                            "dtype": dtype_name,
                            "memory": memory_name,
                            "input_memory": base_memory_name,
                            "memory_placement": {"plan": memory_name},
                            "program_config": config_record,
                            "iterations": args.iters,
                            "repeats": args.repeats,
                            "kind": STOCK_KIND,
                            "row": (
                                "stock_best"
                                if program_spec is None and _is_custom_target(shape)
                                else config_record["name"]
                            ),
                            **_stock_math_fidelity(dtype_name, program_spec),
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

                    if (
                        run_custom
                        and program_spec is None
                        and _is_custom_target(shape)
                        and dtype_name == "bfloat16"
                        and memory_name == "l1"
                    ):
                        for fidelity_split in custom_splits:
                            for math_fidelity in custom_fidelities:
                                split_name = (
                                    None
                                    if fidelity_split is None
                                    else f"{fidelity_split[0]}+{fidelity_split[1]}"
                                )
                                row_name = (
                                    f"custom_block{args.matrix_block}"
                                    if split_name is None
                                    else f"custom_block{args.matrix_block}_split_{split_name}"
                                )
                                custom_record = {
                                    "shape": asdict(shape),
                                    "execution_shape": asdict(shape),
                                    "representative": shape.representative,
                                    "dtype": dtype_name,
                                    "memory": memory_name,
                                    "input_memory": input_memory,
                                    "r_memory": r_memory,
                                    "x0_memory": x0_memory,
                                    "memory_placement": {
                                        "input": input_memory,
                                        "r": r_memory,
                                        "x0": x0_memory,
                                        "compute": "l1",
                                    },
                                    "program_config": {
                                        "name": CUSTOM_KIND,
                                        "kind": CUSTOM_KIND,
                                        "variant": args.custom_variant,
                                        "math_fidelity": math_fidelity,
                                        "fidelity_split": split_name,
                                        "fuse_s": args.fuse_s,
                                        "two_tile_complex": args.two_tile_complex,
                                        "batch_reads": args.batch_reads,
                                        "matrix_block": args.matrix_block,
                                        "fp32_dest_acc_en": args.fp32_dest_acc_en,
                                        "dst_full_sync_en": args.dst_full_sync_en,
                                        "input_memory": input_memory,
                                        "r_memory": r_memory,
                                        "x0_memory": x0_memory,
                                    },
                                    "iterations": args.iters,
                                    "repeats": args.repeats,
                                    "kind": CUSTOM_KIND,
                                    "row": row_name,
                                    "fidelity_split": split_name,
                                }
                                custom_kwargs = {
                                    "ttnn": ttnn,
                                    "device": device,
                                    "shape": shape,
                                    "dtype_name": dtype_name,
                                    "memory_name": memory_name,
                                    "variant": args.custom_variant,
                                    "math_fidelity": math_fidelity,
                                    "profile": args.profile,
                                    "fuse_s": args.fuse_s,
                                    "two_tile_complex": args.two_tile_complex,
                                    "batch_reads": args.batch_reads,
                                    "matrix_block": args.matrix_block,
                                    "fp32_dest_acc_en": args.fp32_dest_acc_en,
                                    "dst_full_sync_en": args.dst_full_sync_en,
                                    "input_memory": input_memory,
                                    "r_memory": r_memory,
                                    "x0_memory": x0_memory,
                                    "iters": args.iters,
                                    "repeats": args.repeats,
                                    "row_name": row_name,
                                }
                                if fidelity_split is not None:
                                    custom_kwargs["fidelity_split"] = fidelity_split
                                custom_record.update(run_custom_newton_schulz(**custom_kwargs))
                                with_efficiency(custom_record, args.peak_tflops)
                                print(
                                    _format_line(
                                        shape,
                                        dtype_name,
                                        memory_name,
                                        f"{CUSTOM_KIND}:{split_name or math_fidelity}",
                                        custom_record,
                                    ),
                                    flush=True,
                                )
                                results.append(custom_record)
    finally:
        ttnn.close_device(device)

    selection = {
        "shape_filters": args.only or [],
        "program_config_kind_filters": args.config_kind or [],
        "custom_math_fidelity": custom_fidelities,
        "input_memory": input_memory,
        "r_memory": r_memory,
        "x0_memory": x0_memory,
        "fuse_s": args.fuse_s,
        "two_tile_complex": args.two_tile_complex,
        "batch_reads": args.batch_reads,
        "fp32_dest_acc_en": args.fp32_dest_acc_en,
        "dst_full_sync_en": args.dst_full_sync_en,
    }
    if args.fidelity_split:
        selection["fidelity_split"] = [f"{prefix}+{suffix}" for prefix, suffix in args.fidelity_split]

    payload = {
        "environment": environment,
        "configuration_mode": args.config_mode,
        "selection": selection,
        "peak_tflops": args.peak_tflops,
        "peak_note": args.peak_note,
        "results": results,
    }
    if args.power_trace is not None:
        power_trace = _record_path(args.power_trace)
        payload["measurement"] = {
            "power_trace": power_trace,
            "power_clock_provenance": {
                "trace": power_trace,
                "columns": list(POWER_TRACE_COLUMNS),
                "power_column": "power_w",
                "clock_column": "aiclk_mhz",
                "temperature_column": "asic_temp_c",
                "sampling_source": "tt-smi snapshot",
            },
        }
    if args.profile:
        payload["profiling"] = {
            "mode": "l1_cycle_counters",
            "measurement_core": 0,
            "clock": "get_timestamp_32b_lower_32_wall_clock",
            "sampling": "all_tiles_all_iterations_with_core_0_first_tile_iteration_warmup",
            "aggregation": "core_0_per_risc_reader_compute_writer_pages",
        }
    if args.profile_csv is not None:
        try:
            payload["device_profile"] = parse_device_profile_csv(args.profile_csv)
        except Exception as exc:  # noqa: BLE001 - profile parsing is diagnostic data
            payload["device_profile_error"] = f"{type(exc).__name__}: {exc}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(strict_json_dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
