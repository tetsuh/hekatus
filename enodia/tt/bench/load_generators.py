"""Fixed work generators and merged-program planning for Issue #118.

The planning layer is board-free.  The optional TTNN helpers only construct a
single ``ProgramDescriptor`` from already prepared descriptors; they do not
open a device or execute a command on import.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enodia.tt.bench.load_harness import (
    DEVICE_OPEN_SETTINGS,
    DRAM_BANK_COUNT,
    DRAM_PAGE_BYTES,
    DRAM_PAGES_PER_BANK,
    DRAM_TOTAL_PAGES,
    HOST_ARITHMETIC_ROUNDS,
    HOST_ARITHMETIC_WORDS,
    HOST_RESIDENT_SUB_DEVICE_ID,
    HOST_TRANSFER_BUFFER_POOL_SIZE,
    HOST_TRANSFER_BYTES,
    HOST_TRANSFER_CORE,
    HOST_TRANSFER_COUNT,
    HOST_TRANSFER_SUB_DEVICE_ID,
    LOAD_CONDITIONS,
    NS_LOAD_BATCH,
    NS_LOAD_DOUBLE_BUFFER,
    NS_LOAD_DST_FULL_SYNC_EN,
    NS_LOAD_FP32_DEST_ACC_EN,
    NS_LOAD_FUSE_S,
    NS_LOAD_ITERATIONS,
    NS_LOAD_L,
    NS_LOAD_MATH_FIDELITY,
    NS_LOAD_MATRIX_BLOCK,
    NS_LOAD_OUTER_REPETITIONS,
    NS_LOAD_OUTPUT_MEMORY,
    NS_LOAD_R_MEMORY,
    NS_LOAD_SHAPE,
    NS_LOAD_VARIANT,
    NS_LOAD_X0_MEMORY,
    P150_RESIDUAL_WORKER_COORDINATES,
    RESIDENT_COMMAND_QUEUE,
    RESIDENT_CORES,
    TRANSFER_COMMAND_QUEUE,
    LoadConfigurationError,
    normalize_condition,
    p150_core_partition,
    validate_device_open_settings,
)

_KERNEL_DIR = Path(__file__).with_name("kernels")
PAGE_WORDS = 32 * 32
PAGE_BYTES = PAGE_WORDS * 4
LOAD_CB_INDEX = 22
LOAD_INTERVAL_CB_INDEX = 23
DRAM_WORKER_COORDINATES = P150_RESIDUAL_WORKER_COORDINATES[:DRAM_BANK_COUNT]


@dataclass(frozen=True)
class LoadProgramPlan:
    """A fixed, auditable load plan that is merged with resident kernels."""

    condition: str
    worker_cores: tuple[tuple[int, int], ...]
    resident_cores: tuple[tuple[int, int], ...] = RESIDENT_CORES
    same_program: bool = True
    resident_command_queue: int = RESIDENT_COMMAND_QUEUE
    transfer_command_queue: int = TRANSFER_COMMAND_QUEUE
    fixed_work: bool = True
    data_dependent_exit: bool = False
    kernel_source: str | None = None

    def __post_init__(self) -> None:
        canonical = normalize_condition(self.condition)
        object.__setattr__(self, "condition", canonical)
        partition = p150_core_partition()
        if (
            tuple(self.worker_cores) != partition.worker_cores
            and canonical not in {"unloaded", "host_cpu_transfer"}
        ):
            raise LoadConfigurationError("load workers must be the pinned 108 residual cores")
        if set(self.worker_cores) & set(self.resident_cores):
            raise LoadConfigurationError("resident and load core ranges overlap")
        if self.same_program is not True:
            raise LoadConfigurationError("resident and device load must share one ProgramDescriptor")
        if self.resident_command_queue != RESIDENT_COMMAND_QUEUE:
            raise LoadConfigurationError("resident work must use CQ0")
        if self.transfer_command_queue != TRANSFER_COMMAND_QUEUE:
            raise LoadConfigurationError("transfer-only work must use CQ1")
        if self.fixed_work is not True or self.data_dependent_exit is not False:
            raise LoadConfigurationError("load generators must be fixed-count and data-independent")

    def as_record(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "worker_cores": [list(core) for core in self.worker_cores],
            "resident_cores": [list(core) for core in self.resident_cores],
            "same_program": self.same_program,
            "resident_command_queue": self.resident_command_queue,
            "transfer_command_queue": self.transfer_command_queue,
            "fixed_work": self.fixed_work,
            "data_dependent_exit": self.data_dependent_exit,
            "kernel_source": self.kernel_source,
        }


def build_load_plan(condition: str) -> LoadProgramPlan:
    """Return the one-program plan for one of the four conditions."""
    canonical = normalize_condition(condition)
    source = {
        # The NS generator is the existing ADR-0013 kernel; the plan names
        # its source logically rather than inventing a second arithmetic path.
        "newton_schulz": "newton_schulz_kernel.py",
        "dram": "issue118_dram_load.cpp",
    }.get(canonical)
    return LoadProgramPlan(
        condition=canonical,
        worker_cores=() if canonical in {"unloaded", "host_cpu_transfer"} else P150_RESIDUAL_WORKER_COORDINATES,
        kernel_source=source,
        fixed_work=True,
        data_dependent_exit=False,
    )


def newton_schulz_load_work() -> dict[str, Any]:
    """Describe the exact fixed NS workload without touching an accelerator."""
    return {
        "shape": NS_LOAD_SHAPE,
        "L": NS_LOAD_L,
        "batch": NS_LOAD_BATCH,
        "iterations": NS_LOAD_ITERATIONS,
        "outer_repetitions": NS_LOAD_OUTER_REPETITIONS,
        "variant": NS_LOAD_VARIANT,
        "math_fidelity": NS_LOAD_MATH_FIDELITY,
        "fuse_s": NS_LOAD_FUSE_S,
        "fp32_dest_acc_en": NS_LOAD_FP32_DEST_ACC_EN,
        "matrix_block": NS_LOAD_MATRIX_BLOCK,
        "double_buffer": NS_LOAD_DOUBLE_BUFFER,
        "dst_full_sync_en": NS_LOAD_DST_FULL_SYNC_EN,
        "r_memory": NS_LOAD_R_MEMORY,
        "x0_memory": NS_LOAD_X0_MEMORY,
        "output_memory": NS_LOAD_OUTPUT_MEMORY,
        "worker_count": len(P150_RESIDUAL_WORKER_COORDINATES),
        "control_flow": "fixed_outer_repetition_and_fixed_iteration_loops",
        "convergence_check": False,
        "data_dependent_exit": False,
    }


def prepare_newton_schulz_load(
    ttnn: Any,
    device: Any,
    *,
    repeat_count: int = 1,
    profile: bool = True,
) -> Any:
    """Prepare ADR-0013 work on residual workers for a fixed repeat count."""
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
        raise LoadConfigurationError("repeat_count must be a positive integer")
    from enodia.tt.bench.newton_schulz_kernel import (
        NewtonSchulzKernel,
        benchmark_matrices,
    )

    return NewtonSchulzKernel.prepare(
        ttnn,
        device,
        benchmark_matrices(NS_LOAD_BATCH, NS_LOAD_L),
        profile=profile,
        variant="bf16",
        math_fidelity="HiFi3",
        fuse_s=True,
        matrix_block=NS_LOAD_MATRIX_BLOCK,
        double_buffer=True,
        input_memory="l1",
        r_memory="l1",
        x0_memory="l1",
        output_memory="dram",
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
        iterations=NS_LOAD_ITERATIONS,
        outer_repetitions=NS_LOAD_OUTER_REPETITIONS * repeat_count,
        core_coordinates=P150_RESIDUAL_WORKER_COORDINATES,
    )


@dataclass
class PreparedDramLoad:
    """Prepared fixed DRAM stream resources for one merged program."""

    ttnn: Any
    device: Any
    read_buffer: Any
    write_buffer: Any
    stats_buffer: Any
    program: Any
    repeat_count: int
    worker_coordinates: tuple[tuple[int, int], ...]

    def program_parts(self) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        return tuple(self.program.kernels), tuple(self.program.semaphores), tuple(self.program.cbs)

    def interval_words(self) -> list[int]:
        host = self.ttnn.from_device(self.stats_buffer, blocking=True, queue_id=0)
        row_major = self.ttnn.to_layout(host, self.ttnn.ROW_MAJOR_LAYOUT)
        values = row_major.to_numpy().reshape(-1)
        # One four-word interval and two counters per bank; retain all rows so
        # the host can audit that every planned bank worker completed.
        return [int(value) for value in values]

    def close(self) -> None:
        for tensor in (self.read_buffer, self.write_buffer, self.stats_buffer):
            try:
                self.ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask run errors
                pass


def prepare_dram_load(ttnn: Any, device: Any, *, repeat_count: int = 1) -> PreparedDramLoad:
    """Prepare the fixed eight-bank stream and host-downloadable stats buffer."""
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
        raise LoadConfigurationError("repeat_count must be a positive integer")
    coordinates = DRAM_WORKER_COORDINATES
    core_ranges = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in coordinates
        ]
    )
    shape = ttnn.Shape((1, DRAM_TOTAL_PAGES, PAGE_WORDS))
    stats_shape = ttnn.Shape((1, DRAM_BANK_COUNT, PAGE_WORDS))
    read_buffer = ttnn.zeros(
        shape,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    write_buffer = ttnn.zeros(
        shape,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    stats_buffer = ttnn.zeros(
        stats_shape,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    read_compile = ttnn.TensorAccessorArgs(read_buffer).get_compile_time_args()
    write_compile = ttnn.TensorAccessorArgs(write_buffer).get_compile_time_args()
    stats_compile = ttnn.TensorAccessorArgs(stats_buffer).get_compile_time_args()
    runtime_args = ttnn.RuntimeArgs()
    for bank, (x, y) in enumerate(coordinates):
        runtime_args[x][y] = [
            read_buffer.buffer_address(),
            write_buffer.buffer_address(),
            DRAM_PAGES_PER_BANK,
            stats_buffer.buffer_address(),
            repeat_count,
            bank,
            bank,
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str((_KERNEL_DIR / "issue118_dram_load.cpp").resolve()),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[*read_compile, *write_compile, *stats_compile],
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb_format = ttnn.CBFormatDescriptor(
        buffer_index=LOAD_CB_INDEX,
        data_format=ttnn.uint32,
        page_size=PAGE_BYTES,
        tile=ttnn.TileDescriptor(32, 32, False),
    )
    interval_format = ttnn.CBFormatDescriptor(
        buffer_index=LOAD_INTERVAL_CB_INDEX,
        data_format=ttnn.uint32,
        page_size=PAGE_BYTES,
        tile=ttnn.TileDescriptor(32, 32, False),
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=PAGE_BYTES,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        ),
        ttnn.CBDescriptor(
            total_size=PAGE_BYTES,
            core_ranges=core_ranges,
            format_descriptors=[interval_format],
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
    return PreparedDramLoad(
        ttnn=ttnn,
        device=device,
        read_buffer=read_buffer,
        write_buffer=write_buffer,
        stats_buffer=stats_buffer,
        program=program,
        repeat_count=repeat_count,
        worker_coordinates=coordinates,
    )


def merge_resident_and_load_program(ttnn: Any, resident_program: Any, load_kernel: Any) -> Any:
    """Merge resident and one prepared load kernel into one descriptor."""
    return merge_program_parts(ttnn, (resident_program.kernels, resident_program.semaphores, resident_program.cbs), load_kernel.program_parts())


def card_interval_from_words(words: list[int] | tuple[int, ...]) -> Any:
    """Decode four little-endian words emitted by a device load interval CB."""
    if len(words) != 4:
        raise LoadConfigurationError("card interval output must contain four words")
    values = tuple(int(value) for value in words)
    if any(value < 0 or value >= 1 << 32 for value in values):
        raise LoadConfigurationError("card interval output words must be uint32")
    from enodia.tt.bench.load_harness import CardLoadInterval

    return CardLoadInterval(values[0] | (values[1] << 32), values[2] | (values[3] << 32))


def card_interval_from_profile_records(records: list[Mapping[str, Any]]) -> Any:
    """Decode the wrap-tracked TRISC0 profile endpoints as a card interval."""
    from enodia.tt.bench.load_harness import CardLoadInterval

    selected = next(
        (
            record
            for record in records
            if record.get("risc") == "TRISC0"
            and record.get("load_end_ticks", 0) != record.get("load_start_ticks", 0)
        ),
        None,
    )
    if selected is None:
        raise LoadConfigurationError("NS profile has no wrap-tracked card load interval")
    return CardLoadInterval(selected["load_start_ticks"], selected["load_end_ticks"])


def dram_load_work() -> dict[str, Any]:
    """Describe the exact fixed read/write stream without a board."""
    return {
        "bank_count": DRAM_BANK_COUNT,
        "pages_per_bank": DRAM_PAGES_PER_BANK,
        "page_bytes": DRAM_PAGE_BYTES,
        "read_pages": DRAM_BANK_COUNT * DRAM_PAGES_PER_BANK,
        "write_pages": DRAM_BANK_COUNT * DRAM_PAGES_PER_BANK,
        "counters": ("read_pages", "write_pages", "elapsed_card_ticks", "achieved_gbps"),
        "load_cb_index": LOAD_CB_INDEX,
        "interval_cb_index": LOAD_INTERVAL_CB_INDEX,
        "control_flow": "fixed_bank_page_loops",
        "data_dependent_exit": False,
    }


def host_cpu_transfer_work() -> dict[str, Any]:
    """Return the condition (c) fixed-work contract."""
    return {
        "arithmetic": "fixed_host_integer_arithmetic",
        "arithmetic_rounds": HOST_ARITHMETIC_ROUNDS,
        "arithmetic_words": HOST_ARITHMETIC_WORDS,
        "transfers": "fixed_size_host_device_transfers",
        "transfer_bytes": HOST_TRANSFER_BYTES,
        "transfer_count": HOST_TRANSFER_COUNT,
        "command_queue": TRANSFER_COMMAND_QUEUE,
        "resident_sub_device_id": HOST_RESIDENT_SUB_DEVICE_ID,
        "transfer_sub_device_id": HOST_TRANSFER_SUB_DEVICE_ID,
        "transfer_core": list(HOST_TRANSFER_CORE),
        "completion_scope": "transfer_sub_device_only",
        "buffer_pool_size": HOST_TRANSFER_BUFFER_POOL_SIZE,
        "buffer_pool": "reusable_host_and_device_buffers",
    }


def run_host_cpu_transfer_work(
    *,
    repetitions: int = 1,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    transfer: Callable[[bytes, int], None] | None = None,
) -> dict[str, Any]:
    """Run the fixed host model with injectable clock/transfer seams.

    The default transfer seam is an in-process copy for board-free tests.  A
    device runner supplies a CQ1 transfer callback; this function never opens
    a device by itself.
    """
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise LoadConfigurationError("host repetitions must be a positive integer")
    payload = bytes(HOST_TRANSFER_BYTES)
    started = clock_ns()
    accumulator = 0
    for _ in range(repetitions):
        for _ in range(HOST_ARITHMETIC_ROUNDS):
            for word in range(HOST_ARITHMETIC_WORDS):
                accumulator = ((accumulator * 33) ^ word) & 0xFFFFFFFF
        if transfer is None:
            for _ in range(HOST_TRANSFER_COUNT):
                bytes(payload)
        else:
            for _ in range(HOST_TRANSFER_COUNT):
                transfer(payload, TRANSFER_COMMAND_QUEUE)
    ended = clock_ns()
    if not isinstance(started, int) or not isinstance(ended, int) or ended < started:
        raise LoadConfigurationError("host clock interval is invalid")
    return {
        **host_cpu_transfer_work(),
        "outer_repetitions": repetitions,
        "transfer_count": HOST_TRANSFER_COUNT * repetitions,
        "start_ns": started,
        "end_ns": ended,
        "elapsed_ns": ended - started,
        "arithmetic_checksum": accumulator,
    }


def _descriptor_parts(program: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """Read parts from board-free stubs and TTNN descriptor objects."""
    missing = [name for name in ("kernels", "semaphores", "cbs") if not hasattr(program, name)]
    if missing:
        raise LoadConfigurationError(
            "ProgramDescriptor does not expose mergeable parts: " + ", ".join(missing)
        )
    return list(program.kernels), list(program.semaphores), list(program.cbs)


def _cb_index(descriptor: Any) -> int:
    formats = getattr(descriptor, "format_descriptors", None)
    if not formats:
        raise LoadConfigurationError("CB descriptor has no format descriptor")
    index = getattr(formats[0], "buffer_index", None)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise LoadConfigurationError("CB descriptor has an invalid buffer index")
    return index


def merge_program_descriptors(ttnn: Any, *programs: Any) -> Any:
    """Construct one ProgramDescriptor and reject circular-buffer collisions."""
    if len(programs) < 1:
        raise LoadConfigurationError("at least one program is required")
    kernels: list[Any] = []
    semaphores: list[Any] = []
    cbs: list[Any] = []
    seen_cb_indices: set[int] = set()
    for program in programs:
        program_kernels, program_semaphores, program_cbs = _descriptor_parts(program)
        kernels.extend(program_kernels)
        semaphores.extend(program_semaphores)
        for descriptor in program_cbs:
            index = _cb_index(descriptor)
            if index in seen_cb_indices:
                raise LoadConfigurationError(f"circular-buffer index collision at CB{index}")
            seen_cb_indices.add(index)
            cbs.append(descriptor)
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def merge_program_parts(ttnn: Any, *parts: tuple[list[Any], list[Any], list[Any]]) -> Any:
    """Merge raw ``(kernels, semaphores, cbs)`` triples into one descriptor."""
    class _Parts:
        pass

    programs = []
    for kernels, semaphores, cbs in parts:
        value = _Parts()
        value.kernels = kernels
        value.semaphores = semaphores
        value.cbs = cbs
        programs.append(value)
    return merge_program_descriptors(ttnn, *programs)


def open_device_settings() -> dict[str, Any]:
    """Return a copy of the required settings for every eventual mode."""
    return validate_device_open_settings(DEVICE_OPEN_SETTINGS)


def open_device(ttnn: Any, *, device_id: int) -> Any:
    """Open with the shared two-CQ setting; this helper is not called by tests."""
    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise LoadConfigurationError("device_id must be a non-negative integer")
    return ttnn.open_device(device_id=device_id, num_command_queues=2)


def validate_plan_catalogue() -> tuple[LoadProgramPlan, ...]:
    """Return all four plans as a board-free construction check."""
    plans = tuple(build_load_plan(condition) for condition in LOAD_CONDITIONS)
    if any(plan.same_program is not True for plan in plans):
        raise LoadConfigurationError("a plan would enqueue resident/load programs separately")
    return plans


__all__ = [
    "DRAM_WORKER_COORDINATES",
    "LOAD_CB_INDEX",
    "LOAD_INTERVAL_CB_INDEX",
    "LoadProgramPlan",
    "PreparedDramLoad",
    "build_load_plan",
    "card_interval_from_profile_records",
    "card_interval_from_words",
    "dram_load_work",
    "host_cpu_transfer_work",
    "merge_program_descriptors",
    "merge_program_parts",
    "merge_resident_and_load_program",
    "newton_schulz_load_work",
    "open_device",
    "open_device_settings",
    "prepare_dram_load",
    "prepare_newton_schulz_load",
    "validate_plan_catalogue",
]
