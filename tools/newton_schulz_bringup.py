"""Single-stage board bring-up diagnostic for issue #63.

The stages intentionally use separate compute sources.  A later experiment can
therefore fail to compile without changing the source used by an earlier
boundary check.  This script is diagnostic-only; it does not reset hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TILE = 32
NUMERICAL_TOLERANCE = 1e-2
TILE_BYTES_BFLOAT16 = TILE * TILE * 2
TILE_BYTES_FLOAT32 = TILE * TILE * 4
DEFAULT_CB_FORMATS = ("bfloat16",) * 25
DEFAULT_CB_PAGE_SIZES = (TILE_BYTES_BFLOAT16,) * 25
STAGE_50_CB_FORMATS = tuple("float32" if index == 16 else "bfloat16" for index in range(25))
STAGE_51_CB_FORMATS = tuple("float32" if index in (16, 17) else "bfloat16" for index in range(25))
STAGE_50_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index == 16 else TILE_BYTES_BFLOAT16 for index in range(25)
)
STAGE_51_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in (16, 17) else TILE_BYTES_BFLOAT16 for index in range(25)
)
STAGE_55_CB_FORMATS = tuple("float32" if index in (16, 17) else "bfloat16" for index in range(25))
STAGE_55_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in (16, 17) else TILE_BYTES_BFLOAT16 for index in range(25)
)
STAGE_57_CB_FORMATS = tuple("float32" if index in (16, 17) else "bfloat16" for index in range(25))
STAGE_57_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in (16, 17) else TILE_BYTES_BFLOAT16 for index in range(25)
)
STAGE_59_CB_FORMATS = tuple(
    "float32" if index in (16, 17, 19) else "bfloat16" for index in range(25)
)
STAGE_59_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in (16, 17, 19) else TILE_BYTES_BFLOAT16 for index in range(25)
)
STAGE_61_FLOAT32_CBS = (2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24)
STAGE_61_CB_FORMATS = tuple(
    "float32" if index in STAGE_61_FLOAT32_CBS else "bfloat16" for index in range(25)
)
STAGE_61_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in STAGE_61_FLOAT32_CBS else TILE_BYTES_BFLOAT16
    for index in range(25)
)
STAGE_62_FLOAT32_CBS = (2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 22, 23, 24)
STAGE_62_CB_FORMATS = tuple(
    "float32" if index in STAGE_62_FLOAT32_CBS else "bfloat16" for index in range(25)
)
STAGE_62_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in STAGE_62_FLOAT32_CBS else TILE_BYTES_BFLOAT16
    for index in range(25)
)
STAGE_63_FLOAT32_CBS = (23,)
STAGE_63_CB_FORMATS = tuple(
    "float32" if index in STAGE_63_FLOAT32_CBS else "bfloat16" for index in range(25)
)
STAGE_63_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in STAGE_63_FLOAT32_CBS else TILE_BYTES_BFLOAT16
    for index in range(25)
)
STAGE_64_FLOAT32_CBS = (23,)
STAGE_64_CB_FORMATS = tuple(
    "float32" if index in STAGE_64_FLOAT32_CBS else "bfloat16" for index in range(25)
)
STAGE_64_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in STAGE_64_FLOAT32_CBS else TILE_BYTES_BFLOAT16
    for index in range(25)
)
STAGE_65_FLOAT32_CBS = (14, 16, 17)
STAGE_65_CB_FORMATS = tuple(
    "float32" if index in STAGE_65_FLOAT32_CBS else "bfloat16" for index in range(25)
)
STAGE_65_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in STAGE_65_FLOAT32_CBS else TILE_BYTES_BFLOAT16
    for index in range(25)
)
STAGE_66_FLOAT32_CBS = (14, 16, 17, 19)
STAGE_66_CB_FORMATS = tuple(
    "float32" if index in STAGE_66_FLOAT32_CBS else "bfloat16" for index in range(25)
)
STAGE_66_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in STAGE_66_FLOAT32_CBS else TILE_BYTES_BFLOAT16
    for index in range(25)
)
STAGE_67_FLOAT32_CBS = (2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24)
STAGE_67_CB_FORMATS = tuple(
    "float32" if index in STAGE_67_FLOAT32_CBS else "bfloat16" for index in range(25)
)
STAGE_67_CB_PAGE_SIZES = tuple(
    TILE_BYTES_FLOAT32 if index in STAGE_67_FLOAT32_CBS else TILE_BYTES_BFLOAT16
    for index in range(25)
)
# Stage 68 retains the audited Stage-62 descriptors and reuses its BF16 CBs 12/13
# as the diagnostic output pair after S is produced in CBs 10/11.
STAGE_68_CB_FORMATS = STAGE_62_CB_FORMATS
STAGE_68_CB_PAGE_SIZES = STAGE_62_CB_PAGE_SIZES
STAGE_68_DIAGNOSTIC_OUTPUT_CBS = (12, 13)
# Stage 69 is a host-identical waypoint-instrumented variant of Stage 68.
STAGE_69_CB_FORMATS = STAGE_68_CB_FORMATS
STAGE_69_CB_PAGE_SIZES = STAGE_68_CB_PAGE_SIZES
STAGE_69_DIAGNOSTIC_OUTPUT_CBS = STAGE_68_DIAGNOSTIC_OUTPUT_CBS
STAGE_61_ACTIVE_CB_INDICES = tuple(
    index for index in range(25) if index in STAGE_61_FLOAT32_CBS or index in (0, 17, 18)
)
STAGE_62_ACTIVE_CB_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
)
STAGE_67_ACTIVE_CB_INDICES = tuple(range(25))
CONSTRUCTION_CB_COUNT = 25
CONSTRUCTION_CB_PAGE_COUNT = 4
CONSTRUCTION_CORE_COORDINATES = ((0, 0),)
CONSTRUCTION_TIMEOUT_SECONDS = 60
# The host shape retains one compile tile for comparison with the recorded stages.
# The authorized dispatch overrides that argument to zero so real compute sources
# enter no tile loop and cannot wait on CB data.
CONSTRUCTION_DISPATCH_COMPILE_TILE_COUNT = 0
KERNEL_DIR = (Path(__file__).resolve().parents[1] / "enodia/tt/bench/kernels").resolve()

# Build-only probing is intentionally a separate path from both ordinary stage
# execution and the construction-time P0-P8 zero-work probes.  The full stage
# sources and their normal one-tile compile shape are required here so a cache
# record says something about the actual failing stages.
BUILD_ONLY_JIT_STAGES = (62, 67)
BUILD_ONLY_REQUIRED_ENV = (
    "TT_METAL_CACHE",
    "TT_METAL_FORCE_JIT_COMPILE",
    "TT_METAL_LOG_KERNELS_COMPILE_COMMANDS",
    "TT_METAL_KERNELS_EARLY_RETURN",
)
BUILD_ONLY_REQUIRED_VALUE = "1"


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    batch: int
    cores: int
    compute_source: str
    reader_source: str
    writer_source: str
    kind: str
    iterations: int
    fp32_dest_acc_en: bool = False
    input_seed: int | None = None
    input_count: int = 6
    input_dtypes: tuple[str, ...] = ("bfloat16",) * 6
    output_dtype: str = "bfloat16"
    cb_formats: tuple[str, ...] = DEFAULT_CB_FORMATS
    cb_page_sizes: tuple[int, ...] = DEFAULT_CB_PAGE_SIZES
    output_count_override: int | None = None


@dataclass(frozen=True)
class ConstructionProbe:
    """Host description of one zero-work construction bisection probe.

    This is deliberately independent of ttnn.  It records the descriptor and
    source choices for the host record and the explicitly gated dispatch, but
    it contains no device handles and no numerical acceptance fields.
    """

    name: str
    purpose: str
    compute_source: str
    reader_source: str
    writer_source: str
    input_dtypes: tuple[str, ...]
    cb_formats: tuple[str, ...]
    cb_page_sizes: tuple[int, ...]
    active_cb_indices: tuple[int, ...]
    batch: int = 1
    core_count: int = 1
    core_coordinates: tuple[tuple[int, int], ...] = CONSTRUCTION_CORE_COORDINATES
    core_ranges: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (((0, 0), (0, 0)),)
    tiles_per_core: int = 1
    tile_count: int = 0
    iterations: int = 8
    fp32_dest_acc_en: bool = True
    output_dtypes: tuple[str, ...] = ("float32", "float32")
    input_count: int = 6
    reader_accessor_count: int = 6
    writer_accessor_count: int = 2
    reader_runtime_arg_count: int = 8
    writer_runtime_arg_count: int = 4
    compute_compile_args: tuple[int, ...] = (1,)
    compute_runtime_args: tuple[object, ...] = ()
    kernel_order: tuple[str, ...] = ("reader", "writer", "compute")
    semaphores: tuple[object, ...] = ()
    cb_page_count: int = CONSTRUCTION_CB_PAGE_COUNT
    external_timeout_seconds: int = CONSTRUCTION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.batch != 1 or self.core_count != 1:
            raise ValueError("construction probes require batch=1 and one core")
        if self.core_coordinates != CONSTRUCTION_CORE_COORDINATES:
            raise ValueError("construction probes use core (0, 0) only")
        if self.core_ranges != (((0, 0), (0, 0)),):
            raise ValueError("construction probes use one CoreRange for core (0, 0)")
        if self.tiles_per_core != 1 or self.tile_count != 0:
            raise ValueError("construction probes keep one compile tile and zero runtime tiles")
        if self.iterations != 8 or not self.fp32_dest_acc_en:
            raise ValueError("construction probes keep the shared iteration configuration")
        if len(self.input_dtypes) != self.input_count or self.input_count != 6:
            raise ValueError("construction probes require six input dtypes")
        if len(self.output_dtypes) != 2 or self.output_dtypes != ("float32", "float32"):
            raise ValueError("construction probes require two Float32 outputs")
        if len(self.cb_formats) != CONSTRUCTION_CB_COUNT:
            raise ValueError("construction probes require 25 CB formats")
        if len(self.cb_page_sizes) != CONSTRUCTION_CB_COUNT:
            raise ValueError("construction probes require 25 CB page sizes")
        if len(self.active_cb_indices) not in (18, 24, 25):
            raise ValueError("construction probes use the Stage-61, Stage-62, or Stage-67 CB set")
        if tuple(sorted(set(self.active_cb_indices))) != self.active_cb_indices:
            raise ValueError("active CB indices must be sorted and unique")
        if any(index < 0 or index >= CONSTRUCTION_CB_COUNT for index in self.active_cb_indices):
            raise ValueError("active CB indices must be in the 25-descriptor range")
        expected_page_sizes = tuple(
            TILE_BYTES_FLOAT32 if fmt == "float32" else TILE_BYTES_BFLOAT16
            for fmt in self.cb_formats
        )
        if self.cb_page_sizes != expected_page_sizes:
            raise ValueError("CB page sizes must match their data formats")
        if self.reader_accessor_count != 6 or self.writer_accessor_count != 2:
            raise ValueError("construction probes keep six reader and two writer accessors")
        if self.reader_runtime_arg_count != 8 or self.writer_runtime_arg_count != 4:
            raise ValueError("construction probes keep the existing runtime argument shapes")
        if self.compute_compile_args != (self.tiles_per_core,) or self.compute_runtime_args != ():
            raise ValueError("construction probes keep the existing compute argument shape")
        if self.kernel_order != ("reader", "writer", "compute") or self.semaphores != ():
            raise ValueError("construction probes keep kernel order and empty semaphores")
        if self.external_timeout_seconds != CONSTRUCTION_TIMEOUT_SECONDS:
            raise ValueError("construction probes require the external 60-second timeout")

    @property
    def descriptor_total_sizes(self) -> tuple[int, ...]:
        """Return the four-page allocation size for each CB descriptor."""
        return tuple(page_size * self.cb_page_count for page_size in self.cb_page_sizes)

    @property
    def descriptor_total_bytes(self) -> int:
        return sum(self.descriptor_total_sizes)


@dataclass(frozen=True)
class ConstructionDispatchConfiguration:
    """Zero-work arguments layered over a recorded construction probe.

    ``ConstructionProbe`` intentionally keeps the host-comparison shape, whose
    compile-time tile argument is one.  This configuration is only for the
    explicitly gated device dispatch and changes the compute loop argument to
    zero while keeping the recorded descriptor and kernel shape unchanged.
    """

    probe: ConstructionProbe
    reader_tile_count: int = 0
    writer_tile_count: int = 0
    compute_compile_args: tuple[int, ...] = (CONSTRUCTION_DISPATCH_COMPILE_TILE_COUNT,)

    def __post_init__(self) -> None:
        if self.reader_tile_count != 0 or self.writer_tile_count != 0:
            raise ValueError("construction dispatch reader and writer tile counts must be zero")
        if self.compute_compile_args != (CONSTRUCTION_DISPATCH_COMPILE_TILE_COUNT,):
            raise ValueError("construction dispatch compute loop count must be zero")


def construction_dispatch_configuration(
    probe_or_name: ConstructionProbe | str,
) -> ConstructionDispatchConfiguration:
    """Return the explicit zero-work override without importing ttnn."""
    probe = (
        construction_probe_for(probe_or_name) if isinstance(probe_or_name, str) else probe_or_name
    )
    return ConstructionDispatchConfiguration(probe=probe)


STAGES = {
    1: Stage(
        1,
        "real_one_tile",
        1,
        1,
        "bringup_real_compute.cpp",
        "bringup_real_reader.cpp",
        "bringup_real_writer.cpp",
        "real",
        1,
    ),
    2: Stage(
        2,
        "real_multiple_tiles",
        4,
        1,
        "bringup_real_compute.cpp",
        "bringup_real_reader.cpp",
        "bringup_real_writer.cpp",
        "real",
        1,
    ),
    3: Stage(
        3,
        "real_multiple_cores",
        4,
        2,
        "bringup_real_compute.cpp",
        "bringup_real_reader.cpp",
        "bringup_real_writer.cpp",
        "real",
        1,
    ),
    4: Stage(
        4,
        "complex_one_matmul",
        1,
        1,
        "bringup_complex_compute.cpp",
        "bringup_complex_reader.cpp",
        "bringup_writer.cpp",
        "complex",
        1,
    ),
    5: Stage(
        5,
        "complex_newton_schulz_one_iteration",
        1,
        1,
        "bringup_ns_one_compute.cpp",
        "bringup_ns_one_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        1,
    ),
    6: Stage(
        6,
        "complex_newton_schulz_eight_iterations",
        1,
        1,
        "bringup_ns_eight_compute.cpp",
        "bringup_ns_eight_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        8,
        False,
        6306,
    ),
    41: Stage(
        41,
        "complex_modern_startup",
        1,
        1,
        "bringup_complex_modern_compute.cpp",
        "bringup_complex_modern_reader.cpp",
        "bringup_complex_modern_writer.cpp",
        "complex_modern_startup",
        1,
    ),
    42: Stage(
        42,
        "complex_two_groups",
        1,
        1,
        "bringup_complex_two_groups_compute.cpp",
        "bringup_complex_two_groups_reader.cpp",
        "bringup_complex_two_groups_writer.cpp",
        "complex_two_groups",
        1,
    ),
    43: Stage(
        43,
        "newton_residual_only",
        1,
        1,
        "bringup_newton_residual_compute.cpp",
        "bringup_newton_residual_reader.cpp",
        "bringup_newton_residual_writer.cpp",
        "newton_residual",
        1,
    ),
    44: Stage(
        44,
        "newton_residual_reader_copy",
        1,
        1,
        "bringup_newton_residual_copy_compute.cpp",
        "bringup_newton_residual_copy_reader.cpp",
        "bringup_newton_residual_copy_writer.cpp",
        "newton_residual_reader_copy",
        1,
    ),
    45: Stage(
        45,
        "newton_one_compute_copy",
        1,
        1,
        "bringup_newton_one_compute_copy_compute.cpp",
        "bringup_newton_one_compute_copy_reader.cpp",
        "bringup_newton_one_compute_copy_writer.cpp",
        "newton_one_compute_copy",
        1,
    ),
    46: Stage(
        46,
        "newton_residual_correct",
        1,
        1,
        "bringup_newton_residual_correct_compute.cpp",
        "bringup_newton_residual_correct_reader.cpp",
        "bringup_newton_residual_correct_writer.cpp",
        "newton_residual_correct",
        1,
    ),
    47: Stage(
        47,
        "newton_residual_correct_reader_copy",
        1,
        1,
        "bringup_newton_residual_correct_reader_copy_compute.cpp",
        "bringup_newton_residual_correct_reader_copy_reader.cpp",
        "bringup_newton_residual_correct_reader_copy_writer.cpp",
        "newton_residual_correct_reader_copy",
        1,
    ),
    48: Stage(
        48,
        "newton_one_correct_reader_copy",
        1,
        1,
        "bringup_newton_one_correct_reader_copy_compute.cpp",
        "bringup_newton_one_correct_reader_copy_reader.cpp",
        "bringup_newton_one_correct_reader_copy_writer.cpp",
        "newton_one_correct_reader_copy",
        1,
    ),
    49: Stage(
        49,
        "complex_newton_schulz_eight_iterations_fp32_dest_acc",
        1,
        1,
        "bringup_ns_eight_compute.cpp",
        "bringup_ns_eight_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        8,
        True,
        6306,
    ),
    50: Stage(
        50,
        "real_fp32_dest_acc_one_matmul",
        1,
        1,
        "bringup_precision_real_one_compute.cpp",
        "bringup_precision_real_one_reader.cpp",
        "bringup_precision_real_writer.cpp",
        "precision_real",
        1,
        True,
        6350,
        2,
        ("bfloat16", "bfloat16"),
        "float32",
        STAGE_50_CB_FORMATS,
        STAGE_50_CB_PAGE_SIZES,
        1,
    ),
    51: Stage(
        51,
        "real_fp32_dest_acc_two_matmuls",
        1,
        1,
        "bringup_precision_real_two_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6351,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    52: Stage(
        52,
        "real_fp32_dest_acc_two_matmuls_reconfig",
        1,
        1,
        "bringup_precision_real_two_reconfig_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6352,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    53: Stage(
        53,
        "real_fp32_dest_acc_two_matmuls_unpack_reconfig",
        1,
        1,
        "bringup_precision_real_two_unpack_reconfig_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6353,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    54: Stage(
        54,
        "real_fp32_dest_acc_two_matmuls_pack_reconfig",
        1,
        1,
        "bringup_precision_real_two_pack_reconfig_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6354,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    55: Stage(
        55,
        "real_fp32_dest_acc_two_matmuls_output_reconfig",
        1,
        1,
        "bringup_precision_real_two_output_reconfig_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_output_reconfig_writer.cpp",
        "precision_real_two",
        1,
        True,
        6355,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "bfloat16",
        STAGE_55_CB_FORMATS,
        STAGE_55_CB_PAGE_SIZES,
        1,
    ),
    56: Stage(
        56,
        "real_fp32_dest_acc_two_matmuls_correct_srcb",
        1,
        1,
        "bringup_precision_real_two_correct_srcb_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6356,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    57: Stage(
        57,
        "real_fp32_dest_acc_two_matmuls_correct_srcb_output_reconfig",
        1,
        1,
        "bringup_precision_real_two_correct_srcb_output_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_correct_srcb_output_writer.cpp",
        "precision_real_two",
        1,
        True,
        6357,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "bfloat16",
        STAGE_57_CB_FORMATS,
        STAGE_57_CB_PAGE_SIZES,
        1,
    ),
    58: Stage(
        58,
        "real_fp32_dest_acc_two_matmuls_correct_srcb_same_output_pack",
        1,
        1,
        "bringup_precision_real_two_correct_srcb_same_output_pack_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_writer.cpp",
        "precision_real_two",
        1,
        True,
        6358,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_51_CB_FORMATS,
        STAGE_51_CB_PAGE_SIZES,
        1,
    ),
    59: Stage(
        59,
        "real_fp32_dest_acc_two_matmuls_correct_srcb_distinct_float32_output",
        1,
        1,
        "bringup_precision_real_two_correct_srcb_distinct_float32_output_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_correct_srcb_output_writer.cpp",
        "precision_real_two",
        1,
        True,
        6359,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_59_CB_FORMATS,
        STAGE_59_CB_PAGE_SIZES,
        1,
    ),
    60: Stage(
        60,
        "real_fp32_dest_acc_two_matmuls_correct_srcb_distinct_float32_output_no_pack",
        1,
        1,
        "bringup_precision_real_two_correct_srcb_distinct_float32_output_no_pack_compute.cpp",
        "bringup_precision_real_two_reader.cpp",
        "bringup_precision_real_two_correct_srcb_output_writer.cpp",
        "precision_real_two",
        1,
        True,
        6360,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_59_CB_FORMATS,
        STAGE_59_CB_PAGE_SIZES,
        1,
    ),
    61: Stage(
        61,
        "complex_newton_schulz_eight_iterations_float32_state",
        1,
        1,
        "bringup_ns_float32_state_compute.cpp",
        "bringup_ns_float32_state_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        8,
        True,
        6306,
        6,
        ("bfloat16", "float32", "bfloat16", "float32", "float32", "float32"),
        "float32",
        STAGE_61_CB_FORMATS,
        STAGE_61_CB_PAGE_SIZES,
    ),
    62: Stage(
        62,
        "complex_newton_schulz_four_bfloat16_four_float32_state",
        1,
        1,
        "bringup_ns_four_plus_four_compute.cpp",
        "bringup_ns_four_plus_four_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        8,
        True,
        6306,
        6,
        ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "float32"),
        "float32",
        STAGE_62_CB_FORMATS,
        STAGE_62_CB_PAGE_SIZES,
    ),
    63: Stage(
        63,
        "bfloat16_to_float32_tile_conversion",
        1,
        1,
        "bringup_bfloat16_to_float32_compute.cpp",
        "bringup_bfloat16_to_float32_reader.cpp",
        "bringup_bfloat16_to_float32_writer.cpp",
        "precision_convert",
        1,
        False,
        6363,
        1,
        ("bfloat16",),
        "float32",
        STAGE_63_CB_FORMATS,
        STAGE_63_CB_PAGE_SIZES,
        1,
    ),
    64: Stage(
        64,
        "bfloat16_to_float32_tile_conversion_with_startup",
        1,
        1,
        "bringup_bfloat16_to_float32_startup_compute.cpp",
        "bringup_bfloat16_to_float32_reader.cpp",
        "bringup_bfloat16_to_float32_writer.cpp",
        "precision_convert",
        1,
        False,
        6363,
        1,
        ("bfloat16",),
        "float32",
        STAGE_64_CB_FORMATS,
        STAGE_64_CB_PAGE_SIZES,
        1,
    ),
    65: Stage(
        65,
        "bfloat16_state_to_float32_matmul_same_output_cb",
        1,
        1,
        "bringup_bfloat16_state_matmul_same_output_compute.cpp",
        "bringup_bfloat16_state_matmul_reader.cpp",
        "bringup_bfloat16_state_matmul_same_output_writer.cpp",
        "precision_convert_matmul",
        1,
        True,
        6365,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_65_CB_FORMATS,
        STAGE_65_CB_PAGE_SIZES,
        1,
    ),
    66: Stage(
        66,
        "bfloat16_state_to_float32_matmul_distinct_output_cb",
        1,
        1,
        "bringup_bfloat16_state_matmul_distinct_output_compute.cpp",
        "bringup_bfloat16_state_matmul_reader.cpp",
        "bringup_bfloat16_state_matmul_distinct_output_writer.cpp",
        "precision_convert_matmul",
        1,
        True,
        6365,
        3,
        ("bfloat16", "bfloat16", "bfloat16"),
        "float32",
        STAGE_66_CB_FORMATS,
        STAGE_66_CB_PAGE_SIZES,
        1,
    ),
    67: Stage(
        67,
        "complex_newton_schulz_four_bfloat16_four_float32_state_distinct_second_output_cb",
        1,
        1,
        "bringup_ns_four_plus_four_distinct_output_compute.cpp",
        "bringup_ns_four_plus_four_distinct_output_reader.cpp",
        "bringup_writer.cpp",
        "newton_schulz",
        8,
        True,
        6306,
        6,
        ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "float32"),
        "float32",
        STAGE_67_CB_FORMATS,
        STAGE_67_CB_PAGE_SIZES,
    ),
    68: Stage(
        68,
        "complex_newton_schulz_first_residual_bfloat16_output",
        1,
        1,
        "bringup_ns_first_residual_compute.cpp",
        "bringup_ns_first_residual_reader.cpp",
        "bringup_ns_first_residual_writer.cpp",
        "newton_first_residual",
        1,
        True,
        6306,
        6,
        ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "float32"),
        "bfloat16",
        STAGE_68_CB_FORMATS,
        STAGE_68_CB_PAGE_SIZES,
    ),
    69: Stage(
        69,
        "complex_newton_schulz_first_residual_bfloat16_output_waypoint",
        1,
        1,
        "bringup_ns_first_residual_waypoint_compute.cpp",
        "bringup_ns_first_residual_reader.cpp",
        "bringup_ns_first_residual_writer.cpp",
        "newton_first_residual",
        1,
        True,
        6306,
        6,
        ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "float32"),
        "bfloat16",
        STAGE_69_CB_FORMATS,
        STAGE_69_CB_PAGE_SIZES,
    ),
}


_CONSTRUCTION_NOOP_COMPUTE_SOURCE = "bringup_construction_noop_compute.cpp"
_CONSTRUCTION_NOOP_READER_SOURCE = "bringup_construction_noop_reader.cpp"
_CONSTRUCTION_NOOP_WRITER_SOURCE = "bringup_construction_noop_writer.cpp"


def _construction_probe(
    name: str,
    purpose: str,
    *,
    compute_source: str,
    reader_source: str,
    writer_source: str,
    input_dtypes: tuple[str, ...],
    cb_formats: tuple[str, ...],
    cb_page_sizes: tuple[int, ...],
    active_cb_indices: tuple[int, ...],
) -> ConstructionProbe:
    return ConstructionProbe(
        name=name,
        purpose=purpose,
        compute_source=compute_source,
        reader_source=reader_source,
        writer_source=writer_source,
        input_dtypes=input_dtypes,
        cb_formats=cb_formats,
        cb_page_sizes=cb_page_sizes,
        active_cb_indices=active_cb_indices,
    )


CONSTRUCTION_PROBES = {
    "P0": _construction_probe(
        "P0",
        "Stage-61 host and source control",
        compute_source=STAGES[61].compute_source,
        reader_source=STAGES[61].reader_source,
        writer_source=STAGES[61].writer_source,
        input_dtypes=STAGES[61].input_dtypes,
        cb_formats=STAGES[61].cb_formats,
        cb_page_sizes=STAGES[61].cb_page_sizes,
        active_cb_indices=STAGE_61_ACTIVE_CB_INDICES,
    ),
    "P1": _construction_probe(
        "P1",
        "Initial X real and imaginary inputs in BF16",
        compute_source=STAGES[61].compute_source,
        reader_source=STAGES[61].reader_source,
        writer_source=STAGES[61].writer_source,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[61].cb_formats,
        cb_page_sizes=STAGES[61].cb_page_sizes,
        active_cb_indices=STAGE_61_ACTIVE_CB_INDICES,
    ),
    "P2": _construction_probe(
        "P2",
        "Stage-62 mixed CB formats and coupled page sizes",
        compute_source=STAGES[61].compute_source,
        reader_source=STAGES[61].reader_source,
        writer_source=STAGES[61].writer_source,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[62].cb_formats,
        cb_page_sizes=STAGES[62].cb_page_sizes,
        active_cb_indices=STAGE_61_ACTIVE_CB_INDICES,
    ),
    "P3a": _construction_probe(
        "P3a",
        "No-op source control with the Stage-61 active CB set",
        compute_source=_CONSTRUCTION_NOOP_COMPUTE_SOURCE,
        reader_source=_CONSTRUCTION_NOOP_READER_SOURCE,
        writer_source=_CONSTRUCTION_NOOP_WRITER_SOURCE,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[62].cb_formats,
        cb_page_sizes=STAGES[62].cb_page_sizes,
        active_cb_indices=STAGE_61_ACTIVE_CB_INDICES,
    ),
    "P3b": _construction_probe(
        "P3b",
        "The same no-op source with the Stage-62 active CB set",
        compute_source=_CONSTRUCTION_NOOP_COMPUTE_SOURCE,
        reader_source=_CONSTRUCTION_NOOP_READER_SOURCE,
        writer_source=_CONSTRUCTION_NOOP_WRITER_SOURCE,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[62].cb_formats,
        cb_page_sizes=STAGES[62].cb_page_sizes,
        active_cb_indices=STAGE_62_ACTIVE_CB_INDICES,
    ),
    "P4": _construction_probe(
        "P4",
        "Real Stage-62 reader with the no-op compute and writer",
        compute_source=_CONSTRUCTION_NOOP_COMPUTE_SOURCE,
        reader_source=STAGES[62].reader_source,
        writer_source=_CONSTRUCTION_NOOP_WRITER_SOURCE,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[62].cb_formats,
        cb_page_sizes=STAGES[62].cb_page_sizes,
        active_cb_indices=STAGE_62_ACTIVE_CB_INDICES,
    ),
    "P5": _construction_probe(
        "P5",
        "Real Stage-62 reader and compute with no-op writer",
        compute_source=STAGES[62].compute_source,
        reader_source=STAGES[62].reader_source,
        writer_source=_CONSTRUCTION_NOOP_WRITER_SOURCE,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[62].cb_formats,
        cb_page_sizes=STAGES[62].cb_page_sizes,
        active_cb_indices=STAGE_62_ACTIVE_CB_INDICES,
    ),
    "P6": _construction_probe(
        "P6",
        "Stage-62 sources with the Stage-67 Float32 CB 13 descriptor",
        compute_source=STAGES[62].compute_source,
        reader_source=STAGES[62].reader_source,
        writer_source=_CONSTRUCTION_NOOP_WRITER_SOURCE,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[67].cb_formats,
        cb_page_sizes=STAGES[67].cb_page_sizes,
        active_cb_indices=STAGE_62_ACTIVE_CB_INDICES,
    ),
    "P7": _construction_probe(
        "P7",
        "Real Stage-67 reader with the Stage-62 compute",
        compute_source=STAGES[62].compute_source,
        reader_source=STAGES[67].reader_source,
        writer_source=_CONSTRUCTION_NOOP_WRITER_SOURCE,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[67].cb_formats,
        cb_page_sizes=STAGES[67].cb_page_sizes,
        active_cb_indices=STAGE_67_ACTIVE_CB_INDICES,
    ),
    "P8": _construction_probe(
        "P8",
        "Real Stage-67 reader and compute with the Stage-67 CB set",
        compute_source=STAGES[67].compute_source,
        reader_source=STAGES[67].reader_source,
        writer_source=_CONSTRUCTION_NOOP_WRITER_SOURCE,
        input_dtypes=STAGES[62].input_dtypes,
        cb_formats=STAGES[67].cb_formats,
        cb_page_sizes=STAGES[67].cb_page_sizes,
        active_cb_indices=STAGE_67_ACTIVE_CB_INDICES,
    ),
}


def construction_probe_for(name: str) -> ConstructionProbe:
    try:
        return CONSTRUCTION_PROBES[name]
    except KeyError as exc:
        raise ValueError(
            f"construction probe must be one of {tuple(CONSTRUCTION_PROBES)}, got {name!r}"
        ) from exc


def stage_for(number: int) -> Stage:
    try:
        return STAGES[number]
    except KeyError as exc:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {number}") from exc


class BuildOnlyProbeConfigurationError(ValueError):
    """A machine-readable preflight failure for the build-only path."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, object] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def build_only_stage_for(number: int) -> Stage:
    """Return a stage allowed by the explicit build-only JIT probe."""
    if number not in BUILD_ONLY_JIT_STAGES:
        raise BuildOnlyProbeConfigurationError(
            "stage_not_allowlisted",
            f"build-only JIT probe stage must be one of {BUILD_ONLY_JIT_STAGES}, got {number}",
            details={"allowed_stages": list(BUILD_ONLY_JIT_STAGES), "stage": number},
        )
    return stage_for(number)


def _resolved_cache_directory(value: str | os.PathLike[str]) -> Path:
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BuildOnlyProbeConfigurationError(
            "cache_directory_invalid",
            "TT_METAL_CACHE must name a usable directory path",
            details={"reason": "path could not be resolved"},
        ) from exc
    if not str(path):
        raise BuildOnlyProbeConfigurationError(
            "cache_directory_missing",
            "TT_METAL_CACHE must name a caller-provided persistent directory",
        )
    return path


def validate_build_only_environment(
    environment: Mapping[str, str] | None = None,
    *,
    cache_directory: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Validate the controls before a build-only path can import ``ttnn``.

    The optional CLI argument is another explicit way for the caller to supply
    ``TT_METAL_CACHE``.  If both sources are supplied they must resolve to the
    same directory; no existing runtime setting is silently replaced.
    """
    supplied = dict(os.environ if environment is None else environment)
    invalid: dict[str, str] = {}
    configured_cache = supplied.get("TT_METAL_CACHE")
    if cache_directory is None:
        if configured_cache is None or not configured_cache:
            invalid["TT_METAL_CACHE"] = "missing"
            resolved_cache = None
        else:
            resolved_cache = _resolved_cache_directory(configured_cache)
    else:
        resolved_cache = _resolved_cache_directory(cache_directory)
        if configured_cache is not None:
            if not configured_cache:
                invalid["TT_METAL_CACHE"] = "missing"
            else:
                try:
                    configured_path = _resolved_cache_directory(configured_cache)
                except BuildOnlyProbeConfigurationError:
                    invalid["TT_METAL_CACHE"] = "invalid"
                else:
                    if configured_path != resolved_cache:
                        invalid["TT_METAL_CACHE"] = "does not match cache_directory"

    for name in BUILD_ONLY_REQUIRED_ENV[1:]:
        if supplied.get(name) != BUILD_ONLY_REQUIRED_VALUE:
            invalid[name] = "missing" if name not in supplied else "expected '1'"

    if invalid:
        raise BuildOnlyProbeConfigurationError(
            "required_environment_invalid",
            "build-only JIT probe requires TT_METAL_CACHE and three explicit TT-Metal controls",
            details={"variables": invalid},
        )
    assert resolved_cache is not None
    effective = dict(supplied)
    effective["TT_METAL_CACHE"] = str(resolved_cache)
    return effective


def _ensure_cache_directory(cache_directory: Path) -> None:
    try:
        cache_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BuildOnlyProbeConfigurationError(
            "cache_directory_unusable",
            "TT_METAL_CACHE could not be created or opened",
            details={"reason": type(exc).__name__},
        ) from exc
    if not cache_directory.is_dir():
        raise BuildOnlyProbeConfigurationError(
            "cache_directory_unusable",
            "TT_METAL_CACHE must name a directory",
        )


@contextmanager
def _temporary_required_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Expose validated controls while importing and running the runtime."""
    previous: dict[str, str | None] = {
        name: os.environ.get(name) for name in BUILD_ONLY_REQUIRED_ENV
    }
    for name in BUILD_ONLY_REQUIRED_ENV:
        os.environ[name] = values[name]
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _hpd_batch(batch: int, *, condition_number: float, seed: int) -> np.ndarray:
    """Build deterministic complex HPD matrices with a requested spectrum."""
    eigenvalues = np.geomspace(1.0, condition_number, TILE)
    rng = np.random.default_rng(seed)
    matrices = np.empty((batch, TILE, TILE), dtype=np.complex64)
    for index in range(batch):
        random = rng.normal(size=(TILE, TILE)) + 1j * rng.normal(size=(TILE, TILE))
        basis, _ = np.linalg.qr(random)
        matrices[index] = (basis * eigenvalues) @ basis.conj().T
    return matrices


def _inputs(stage: Stage) -> list[np.ndarray]:
    """Build deterministic float32 tile inputs in the fixed descriptor order."""
    input_seed = 6300 + stage.number if stage.input_seed is None else stage.input_seed
    rng = np.random.default_rng(input_seed)
    shape = (stage.batch, TILE, TILE)
    if stage.kind in {
        "newton_schulz",
        "newton_first_residual",
        "newton_residual",
        "newton_residual_reader_copy",
        "newton_one_compute_copy",
        "newton_residual_correct",
        "newton_residual_correct_reader_copy",
        "newton_one_correct_reader_copy",
    }:
        matrices = _hpd_batch(stage.batch, condition_number=100.0, seed=input_seed)
        a_real = matrices.real.astype(np.float32)
        a_imag = matrices.imag.astype(np.float32)
        b_real = np.empty_like(a_real)
        b_imag = np.empty_like(a_imag)
    else:
        a_real = rng.normal(0.0, 0.05, shape).astype(np.float32)
        b_real = rng.normal(0.0, 0.05, shape).astype(np.float32)
        a_imag = rng.normal(0.0, 0.05, shape).astype(np.float32)
        b_imag = rng.normal(0.0, 0.05, shape).astype(np.float32)
    if stage.kind in {
        "newton_schulz",
        "newton_first_residual",
        "newton_residual",
        "newton_residual_reader_copy",
        "newton_one_compute_copy",
        "newton_residual_correct",
        "newton_residual_correct_reader_copy",
        "newton_one_correct_reader_copy",
    }:
        x0 = _initial_value(a_real.astype(np.complex64) + 1j * a_imag)
        b_real = x0.real.astype(np.float32)
        b_imag = x0.imag.astype(np.float32)
    identity = np.broadcast_to(
        2.0 * np.eye(TILE, dtype=np.float32), (stage.batch, TILE, TILE)
    ).copy()
    zero = np.zeros(shape, dtype=np.float32)
    return [a_real, b_real, a_imag, b_imag, identity, zero]


def _initial_value(values: np.ndarray) -> np.ndarray:
    norm_1 = np.linalg.norm(values, ord=1, axis=(-2, -1))
    norm_inf = np.linalg.norm(values, ord=np.inf, axis=(-2, -1))
    return np.swapaxes(values.conj(), -1, -2) / (norm_1 * norm_inf)[:, None, None]


def _bfloat16_roundtrip(values: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 precision and widen them back to float32."""
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)


def expected_output(stage: Stage, inputs: list[np.ndarray]) -> np.ndarray:
    a_real, b_real, a_imag, b_imag, identity, _ = inputs
    if stage.kind == "precision_convert":
        return _bfloat16_roundtrip(a_real)
    if stage.kind == "precision_convert_matmul":
        r_bfloat16 = _bfloat16_roundtrip(a_real)
        state_bfloat16 = _bfloat16_roundtrip(a_imag)
        return np.matmul(r_bfloat16, state_bfloat16).astype(np.float32)
    if stage.kind == "real":
        return np.matmul(a_real, b_real).astype(np.complex64)
    if stage.kind == "precision_real":
        return np.matmul(a_real, b_real).astype(np.float32)
    if stage.kind == "precision_real_two":
        first = np.matmul(a_real, b_real)
        return np.matmul(first, a_imag).astype(np.float32)
    if stage.kind in {"complex", "complex_modern_startup", "complex_two_groups"}:
        left = a_real.astype(np.complex64) + 1j * a_imag
        right = b_real.astype(np.complex64) + 1j * b_imag
        return np.matmul(left, right).astype(np.complex64)

    r = a_real.astype(np.complex64) + 1j * a_imag
    if stage.kind in {
        "newton_first_residual",
        "newton_residual",
        "newton_residual_reader_copy",
        "newton_residual_correct",
        "newton_residual_correct_reader_copy",
    }:
        return (identity.astype(np.complex64) - np.matmul(r, _initial_value(r))).astype(
            np.complex64
        )
    x = _initial_value(r)
    for _ in range(stage.iterations):
        product = np.matmul(r, x)
        s = identity.astype(np.complex64) - product
        x = np.matmul(x, s)
    return x.astype(np.complex64)


def source_paths(stage_or_probe: Stage | ConstructionProbe) -> tuple[Path, Path, Path]:
    compute = (KERNEL_DIR / stage_or_probe.compute_source).resolve()
    reader = (KERNEL_DIR / stage_or_probe.reader_source).resolve()
    writer = (KERNEL_DIR / stage_or_probe.writer_source).resolve()
    for path in (compute, reader, writer):
        if not path.is_file():
            raise FileNotFoundError(path)
    return compute, reader, writer


def construction_source_paths(probe: ConstructionProbe) -> tuple[Path, Path, Path]:
    """Resolve a probe's source triplet without importing or requiring ttnn."""
    return source_paths(probe)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_artifacts_for_source(cache_directory: Path, source_name: str) -> list[dict[str, object]]:
    token = Path(source_name).stem.lower()
    artifacts: list[dict[str, object]] = []
    for path in cache_directory.rglob("*"):
        if not path.is_file() or token not in path.relative_to(cache_directory).as_posix().lower():
            continue
        relative = path.relative_to(cache_directory).as_posix()
        artifacts.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    artifacts.sort(key=lambda artifact: str(artifact["relative_path"]))
    return artifacts


def cache_artifact_manifest(
    cache_directory: str | os.PathLike[str], stage_or_number: Stage | int
) -> dict[str, object]:
    """Return a deterministic manifest for one stage's reader and compute cache entries."""
    stage = stage_for(stage_or_number) if isinstance(stage_or_number, int) else stage_or_number
    cache_path = _resolved_cache_directory(cache_directory)
    _ensure_cache_directory(cache_path)
    manifest: dict[str, object] = {}
    for kind, source_name in (
        ("reader", stage.reader_source),
        ("compute", stage.compute_source),
    ):
        artifacts = _cache_artifacts_for_source(cache_path, source_name)
        manifest[kind] = {
            "source": source_name,
            "status": "matched" if artifacts else "no_matching_artifacts",
            "artifacts": artifacts,
        }
    return manifest


def _core_coordinates(ttnn: Any, device: Any, count: int) -> tuple[list[tuple[int, int]], Any]:
    grid = device.compute_with_storage_grid_size()
    if count < 1 or count > grid.x * grid.y:
        raise ValueError(f"requested {count} cores, device grid is {grid.x}x{grid.y}")
    coordinates = [(index % grid.x, index // grid.x) for index in range(count)]
    ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coordinates]
    )
    return coordinates, ranges


def _runtime_args(
    ttnn: Any,
    coordinates: list[tuple[int, int]],
    values: list[int],
    tiles_per_core: int,
) -> Any:
    args = ttnn.RuntimeArgs()
    for index, (x, y) in enumerate(coordinates):
        args[x][y] = [*values, index * tiles_per_core, tiles_per_core]
    return args


def _device_tensor(ttnn: Any, values: np.ndarray, device: Any, dtype_name: str) -> Any:
    if values.ndim == 3:
        values = values[:, None, :, :]
    tensor = ttnn.Tensor(np.ascontiguousarray(values), getattr(ttnn, dtype_name))
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return ttnn.to_device(tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)


def _download(ttnn: Any, tensor: Any) -> np.ndarray:
    host = ttnn.from_device(tensor)
    row_major = ttnn.to_layout(host, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.typecast(row_major, ttnn.float32).to_numpy()


def _descriptor(ttnn: Any, index: int, core_ranges: Any, data_format: str, page_size: int) -> Any:
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=index,
        data_format=getattr(ttnn, data_format),
        page_size=page_size,
        tile=ttnn.TileDescriptor(TILE, TILE, False),
    )
    return ttnn.CBDescriptor(
        total_size=4 * page_size,
        core_ranges=core_ranges,
        format_descriptors=[fmt],
    )


def output_count(stage: Stage) -> int:
    if stage.output_count_override is not None:
        return stage.output_count_override
    return 1 if stage.kind in {"real", "precision_convert", "precision_convert_matmul"} else 2


def construction_probe_record(probe_or_name: ConstructionProbe | str) -> dict[str, Any]:
    """Serialize a probe's host configuration without making a device call."""
    probe = (
        construction_probe_for(probe_or_name) if isinstance(probe_or_name, str) else probe_or_name
    )
    dispatch = construction_dispatch_configuration(probe)
    compute, reader, writer = construction_source_paths(probe)
    return {
        "probe": probe.name,
        "status": "host-configured",
        "board_run": False,
        "numerical_acceptance": False,
        "external_timeout_s": probe.external_timeout_seconds,
        "batch": probe.batch,
        "core_count": probe.core_count,
        "core_range_count": len(probe.core_ranges),
        "core_coordinates": [list(coordinate) for coordinate in probe.core_coordinates],
        "core_ranges": [[list(start), list(end)] for start, end in probe.core_ranges],
        "tiles_per_core": probe.tiles_per_core,
        "tile_count": probe.tile_count,
        "iterations": probe.iterations,
        "fp32_dest_acc_en": probe.fp32_dest_acc_en,
        "input_dtypes": list(probe.input_dtypes),
        "output_dtypes": list(probe.output_dtypes),
        "cb_descriptor_count": len(probe.cb_formats),
        "cb_page_count": probe.cb_page_count,
        "cb_formats": list(probe.cb_formats),
        "cb_page_sizes": list(probe.cb_page_sizes),
        "cb_descriptor_total_bytes": probe.descriptor_total_bytes,
        "active_cb_indices": list(probe.active_cb_indices),
        "kernel_order": list(probe.kernel_order),
        "semaphores": [],
        "reader_accessor_count": probe.reader_accessor_count,
        "writer_accessor_count": probe.writer_accessor_count,
        "reader_runtime_arg_count": probe.reader_runtime_arg_count,
        "writer_runtime_arg_count": probe.writer_runtime_arg_count,
        "compute_compile_args": list(probe.compute_compile_args),
        "compute_runtime_args": [],
        "dispatch_zero_work": {
            "reader_tile_count": dispatch.reader_tile_count,
            "writer_tile_count": dispatch.writer_tile_count,
            "compute_compile_args": list(dispatch.compute_compile_args),
        },
        "sources": {
            "compute": compute.name,
            "reader": reader.name,
            "writer": writer.name,
        },
        "purpose": probe.purpose,
    }


def is_successful_construction_dispatch_record(record: object) -> bool:
    """Return whether a JSON-decoded record proves a successful zero-work dispatch.

    This pure predicate deliberately validates only machine-readable dispatch
    evidence, so a shell wrapper can parse JSON and call it without fragile
    text matching.
    """
    if not isinstance(record, dict):
        return False
    if record.get("status") != "dispatched" or record.get("board_run") is not True:
        return False

    device_id = record.get("device_id")
    if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id < 0:
        return False

    elapsed = record.get("dispatch_elapsed_s")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        return False
    if isinstance(elapsed, float) and not math.isfinite(elapsed):
        return False
    if elapsed < 0:
        return False

    zero_work = record.get("dispatch_zero_work")
    if not isinstance(zero_work, dict) or set(zero_work) != {
        "reader_tile_count",
        "writer_tile_count",
        "compute_compile_args",
    }:
        return False
    reader_tile_count = zero_work.get("reader_tile_count")
    writer_tile_count = zero_work.get("writer_tile_count")
    compute_compile_args = zero_work.get("compute_compile_args")
    return (
        type(reader_tile_count) is int
        and reader_tile_count == 0
        and type(writer_tile_count) is int
        and writer_tile_count == 0
        and type(compute_compile_args) is list
        and len(compute_compile_args) == 1
        and type(compute_compile_args[0]) is int
        and compute_compile_args[0] == 0
    )


def _deallocate_tensors(ttnn: Any, tensors: list[Any]) -> None:
    for tensor in tensors:
        try:
            ttnn.deallocate(tensor)
        except Exception:  # noqa: BLE001, S110 - diagnostics must attempt all cleanup
            pass


def build_construction_probe_program(
    ttnn: Any,
    device: Any,
    probe_or_name: ConstructionProbe | str,
) -> tuple[Any, list[Any], list[Any]]:
    """Allocate tensors and build one probe's zero-work ``ProgramDescriptor``.

    The returned input and output tensors belong to the caller.  This function
    does not launch or download anything; the authorized dispatch path owns the
    launch and cleanup.  If construction fails after a partial allocation, the
    tensors allocated here are released before the exception is propagated.
    """
    probe = (
        construction_probe_for(probe_or_name) if isinstance(probe_or_name, str) else probe_or_name
    )
    dispatch = construction_dispatch_configuration(probe)
    inputs: list[Any] = []
    outputs: list[Any] = []
    handed_off = False
    try:
        coordinates, core_ranges = _core_coordinates(ttnn, device, probe.core_count)
        if tuple(coordinates) != probe.core_coordinates:
            raise ValueError(
                f"device core coordinates {tuple(coordinates)!r} do not match "
                f"probe coordinates {probe.core_coordinates!r}"
            )
        input_values = _construction_input_values(probe)
        for values, dtype_name in zip(input_values, probe.input_dtypes):
            inputs.append(_device_tensor(ttnn, values, device, dtype_name))
        output_shape = ttnn.Shape((probe.batch, 1, TILE, TILE))
        for dtype_name in probe.output_dtypes:
            outputs.append(
                ttnn.allocate_tensor_on_device(
                    output_shape,
                    getattr(ttnn, dtype_name),
                    ttnn.TILE_LAYOUT,
                    device,
                    ttnn.L1_MEMORY_CONFIG,
                )
            )

        reader_compile_args: list[int] = []
        for tensor in inputs:
            reader_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        writer_compile_args: list[int] = []
        for tensor in outputs:
            writer_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        input_addresses = [tensor.buffer_address() for tensor in inputs]
        output_addresses = [tensor.buffer_address() for tensor in outputs]
        compute, reader, writer = construction_source_paths(probe)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(reader),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=reader_compile_args,
                runtime_args=_runtime_args(
                    ttnn, coordinates, input_addresses, dispatch.reader_tile_count
                ),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(writer),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=writer_compile_args,
                runtime_args=_runtime_args(
                    ttnn, coordinates, output_addresses, dispatch.writer_tile_count
                ),
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(compute),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                # The recorded host shape remains [1].  This explicit [0]
                # override makes every real compute source skip its outer loop.
                compile_time_args=list(dispatch.compute_compile_args),
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    dst_full_sync_en=True, fp32_dest_acc_en=probe.fp32_dest_acc_en
                ),
            ),
        ]
        # ProgramDescriptor has no separate active-index list in this API.
        # Keep all 25 indexed descriptors so each probe's source metadata and
        # format/page mapping are present in the constructed program.
        program = ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=list(probe.semaphores),
            cbs=[
                _descriptor(
                    ttnn, index, core_ranges, probe.cb_formats[index], probe.cb_page_sizes[index]
                )
                for index in range(CONSTRUCTION_CB_COUNT)
            ],
        )
        handed_off = True
        return program, inputs, outputs
    finally:
        if not handed_off:
            _deallocate_tensors(ttnn, [*inputs, *outputs])


def _construction_dispatch_record(
    probe: ConstructionProbe,
    *,
    device_id: int,
    elapsed: float,
) -> dict[str, Any]:
    record = construction_probe_record(probe)
    record.update(
        {
            "status": "dispatched",
            "board_run": True,
            "device_id": device_id,
            "dispatch_elapsed_s": elapsed,
        }
    )
    return record


def dispatch_construction_probe(
    probe_or_name: ConstructionProbe | str,
    *,
    allow_device_dispatch: bool = False,
    device_id: int = 0,
    dispatcher: Callable[[ConstructionProbe], Any] | None = None,
    ttnn_module: Any | None = None,
) -> Any:
    """Dispatch one explicitly authorized zero-work construction probe.

    Normal callers must pass ``allow_device_dispatch=True``.  The optional
    ``dispatcher`` remains a host-only injection seam for tests and does not
    open a device; the normal authorized path imports ttnn, opens ``device_id``,
    builds the real source descriptors, invokes ``generic_op``, synchronizes,
    and releases every tensor and the device.  No output is downloaded.
    """
    probe = (
        construction_probe_for(probe_or_name) if isinstance(probe_or_name, str) else probe_or_name
    )
    if not allow_device_dispatch:
        raise PermissionError(
            "construction probe dispatch is disabled; use the host configuration only"
        )
    if dispatcher is not None:
        return dispatcher(probe)
    if ttnn_module is None:
        import ttnn
    else:
        ttnn = ttnn_module

    device = ttnn.open_device(device_id=device_id)
    try:
        program, inputs, outputs = build_construction_probe_program(ttnn, device, probe)
        try:
            started = time.perf_counter()
            ttnn.generic_op([*inputs, *outputs], program)
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - started
            return _construction_dispatch_record(probe, device_id=device_id, elapsed=elapsed)
        finally:
            _deallocate_tensors(ttnn, [*inputs, *outputs])
    finally:
        ttnn.close_device(device)


def _construction_input_values(probe: ConstructionProbe) -> list[np.ndarray]:
    """Return allocation-only tiles; construction dispatch has no oracle."""
    return [
        np.zeros((probe.batch, TILE, TILE), dtype=np.float32)
        for _ in range(len(probe.input_dtypes))
    ]


def _prepare_stage_program(
    ttnn: Any,
    device: Any,
    stage: Stage,
    input_values: list[np.ndarray],
) -> tuple[Any, list[Any], list[Any]]:
    """Build a normal full-work program from caller-provided allocation values."""
    coordinates, core_ranges = _core_coordinates(ttnn, device, stage.cores)
    if stage.batch % stage.cores:
        raise ValueError("diagnostic stages require an even tile partition")
    tiles_per_core = stage.batch // stage.cores
    input_dtypes = stage.input_dtypes[: len(input_values)]
    device_inputs: list[Any] = []
    outputs: list[Any] = []
    try:
        device_inputs = [
            _device_tensor(ttnn, value, device, dtype_name)
            for value, dtype_name in zip(input_values, input_dtypes)
        ]
        output_shape = ttnn.Shape((stage.batch, 1, TILE, TILE))
        outputs = [
            ttnn.allocate_tensor_on_device(
                output_shape,
                getattr(ttnn, stage.output_dtype),
                ttnn.TILE_LAYOUT,
                device,
                ttnn.L1_MEMORY_CONFIG,
            )
            for _ in range(output_count(stage))
        ]

        reader_compile_args: list[int] = []
        for tensor in device_inputs:
            reader_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        writer_compile_args: list[int] = []
        for tensor in outputs:
            writer_compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        input_addresses = [tensor.buffer_address() for tensor in device_inputs]
        output_addresses = [tensor.buffer_address() for tensor in outputs]
        compute, reader, writer = source_paths(stage)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(reader),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=reader_compile_args,
                runtime_args=_runtime_args(ttnn, coordinates, input_addresses, tiles_per_core),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(writer),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                compile_time_args=writer_compile_args,
                runtime_args=_runtime_args(ttnn, coordinates, output_addresses, tiles_per_core),
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(compute),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_ranges,
                # This is the ordinary stage shape.  Build-only probing never
                # uses the construction probe's zero-work compile override.
                compile_time_args=[tiles_per_core],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    dst_full_sync_en=True, fp32_dest_acc_en=stage.fp32_dest_acc_en
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=[],
            cbs=[
                _descriptor(
                    ttnn, index, core_ranges, stage.cb_formats[index], stage.cb_page_sizes[index]
                )
                for index in range(25)
            ],
        )
        return program, device_inputs, outputs
    except Exception:
        _deallocate_tensors(ttnn, [*device_inputs, *outputs])
        raise


def _prepare(ttnn: Any, device: Any, stage: Stage) -> tuple[Any, list[Any], list[Any], np.ndarray]:
    inputs = _inputs(stage)
    expected = expected_output(stage, inputs)
    input_values = inputs[:2] if stage.kind == "real" else inputs[: stage.input_count]
    program, device_inputs, outputs = _prepare_stage_program(ttnn, device, stage, input_values)
    return program, device_inputs, outputs, expected


def _build_only_input_values(stage: Stage) -> list[np.ndarray]:
    """Return allocation-only values without constructing a numerical oracle."""
    input_count = 2 if stage.kind == "real" else stage.input_count
    return [np.zeros((stage.batch, TILE, TILE), dtype=np.float32) for _ in range(input_count)]


def run_stage(ttnn: Any, device: Any, stage: Stage) -> dict[str, Any]:
    program, inputs, outputs, expected = _prepare(ttnn, device, stage)
    try:
        started = time.perf_counter()
        ttnn.generic_op([*inputs, *outputs], program)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - started
        real = _download(ttnn, outputs[0])[:, 0]
        imag = (
            np.zeros_like(real) if output_count(stage) == 1 else _download(ttnn, outputs[1])[:, 0]
        )
        actual = real + 1j * imag
        finite = bool(np.isfinite(actual).all())
        error = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))
        tolerance = NUMERICAL_TOLERANCE
        passed = finite and error <= tolerance
        return {
            "stage": stage.number,
            "name": stage.name,
            "status": "pass" if passed else "fail",
            "batch": stage.batch,
            "cores": stage.cores,
            "tile_shape": [TILE, TILE],
            "elapsed_s": elapsed,
            "numerical_error": error,
            "tolerance": tolerance,
            "finite": finite,
        }
    finally:
        for tensor in [*inputs, *outputs]:
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - diagnostics must attempt all cleanup
                pass


def _build_only_record(
    stage: Stage,
    *,
    cache_directory: Path,
    pre_manifest: dict[str, object] | None,
    post_manifest: dict[str, object] | None,
    elapsed: float,
    success: bool,
    exit_code: int,
    status: str,
    error: object | None = None,
    device_id: int | None = None,
) -> dict[str, object]:
    compute, reader, writer = source_paths(stage)
    record: dict[str, object] = {
        "stage": stage.number,
        "name": stage.name,
        "status": status,
        "success": success,
        "exit_code": exit_code,
        "exit_state": "success" if success else "failure",
        "build_only": True,
        "numerical_acceptance": False,
        "output_download": False,
        "batch": stage.batch,
        "cores": stage.cores,
        "tile_shape": [TILE, TILE],
        "compile_time_tile_count": [stage.batch // stage.cores],
        "reader_tile_count": stage.batch // stage.cores,
        "writer_tile_count": stage.batch // stage.cores,
        "elapsed_s": elapsed,
        "cache_directory": str(cache_directory),
        "cache_artifacts": {"pre": pre_manifest, "post": post_manifest},
        "sources": {
            "compute": compute.name,
            "reader": reader.name,
            "writer": writer.name,
        },
    }
    if device_id is not None:
        record["device_id"] = device_id
    if error is not None:
        record["error"] = error
    return record


def _build_only_error_record(
    stage_number: int | None,
    *,
    cache_directory: str | os.PathLike[str] | None,
    elapsed: float,
    error: BuildOnlyProbeConfigurationError,
) -> dict[str, object]:
    stage = STAGES.get(stage_number) if stage_number is not None else None
    try:
        cache_path = (
            _resolved_cache_directory(cache_directory)
            if cache_directory is not None
            else Path.cwd()
        )
    except BuildOnlyProbeConfigurationError:
        cache_path = Path.cwd()
    record: dict[str, object] = {
        "stage": stage_number,
        "name": stage.name if stage is not None else None,
        "status": "configuration-error",
        "success": False,
        "exit_code": 2,
        "exit_state": "configuration-failure",
        "build_only": True,
        "numerical_acceptance": False,
        "output_download": False,
        "elapsed_s": elapsed,
        "cache_directory": str(cache_path) if cache_directory is not None else None,
        "cache_artifacts": {"pre": None, "post": None},
        "error": {
            "code": error.code,
            "message": str(error),
            "details": error.details,
        },
    }
    if stage is not None:
        record["sources"] = {
            "compute": stage.compute_source,
            "reader": stage.reader_source,
            "writer": stage.writer_source,
        }
    return record


def run_build_only_stage(
    ttnn: Any,
    device: Any,
    stage_or_number: Stage | int,
    cache_directory: str | os.PathLike[str],
    *,
    device_id: int | None = None,
) -> dict[str, object]:
    """Build and dispatch one allow-listed stage without downloading output.

    This lower-level seam accepts an already-open device so host tests can
    exercise the full-work program shape with a fake runtime and never import
    or open a real device.
    """
    stage = (
        build_only_stage_for(stage_or_number)
        if isinstance(stage_or_number, int)
        else stage_or_number
    )
    if stage.number not in BUILD_ONLY_JIT_STAGES:
        raise BuildOnlyProbeConfigurationError(
            "stage_not_allowlisted",
            f"build-only JIT probe stage must be one of {BUILD_ONLY_JIT_STAGES}, got {stage.number}",
            details={"allowed_stages": list(BUILD_ONLY_JIT_STAGES), "stage": stage.number},
        )
    if stage != build_only_stage_for(stage.number):
        raise BuildOnlyProbeConfigurationError(
            "stage_definition_mismatch",
            "build-only JIT probing requires the registered full stage definition",
            details={"stage": stage.number},
        )
    cache_path = _resolved_cache_directory(cache_directory)
    _ensure_cache_directory(cache_path)
    pre_manifest = cache_artifact_manifest(cache_path, stage)
    started = time.perf_counter()
    device_inputs: list[Any] = []
    outputs: list[Any] = []
    success = False
    error: object | None = None
    try:
        tiles_per_core = stage.batch // stage.cores
        if tiles_per_core != 1:
            raise BuildOnlyProbeConfigurationError(
                "full_tile_shape_unavailable",
                "allow-listed build-only stages must compile one tile per core",
                details={"tiles_per_core": tiles_per_core},
            )
        program, device_inputs, outputs = _prepare_stage_program(
            ttnn, device, stage, _build_only_input_values(stage)
        )
        ttnn.generic_op([*device_inputs, *outputs], program)
        ttnn.synchronize_device(device)
        success = True
    except BuildOnlyProbeConfigurationError as exc:
        error = {"code": exc.code, "message": str(exc), "details": exc.details}
    except Exception as exc:  # noqa: BLE001 - diagnostics must return machine-readable failure
        error = {"code": "dispatch_failed", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        _deallocate_tensors(ttnn, [*device_inputs, *outputs])
    elapsed = time.perf_counter() - started
    post_manifest = cache_artifact_manifest(cache_path, stage)
    return _build_only_record(
        stage,
        cache_directory=cache_path,
        pre_manifest=pre_manifest,
        post_manifest=post_manifest,
        elapsed=elapsed,
        success=success,
        exit_code=0 if success else 1,
        status="pass" if success else "fail",
        error=error,
        device_id=device_id,
    )


def run_build_only_jit_probe(
    stage_number: int,
    *,
    cache_directory: str | os.PathLike[str] | None = None,
    device_id: int = 0,
    environment: Mapping[str, str] | None = None,
    ttnn_module: Any | None = None,
) -> dict[str, object]:
    """Run the explicit full-work build-only probe after a closed preflight."""
    started = time.perf_counter()
    try:
        build_only_stage_for(stage_number)
        effective_environment = validate_build_only_environment(
            environment, cache_directory=cache_directory
        )
        cache_path = _resolved_cache_directory(effective_environment["TT_METAL_CACHE"])
        _ensure_cache_directory(cache_path)
    except BuildOnlyProbeConfigurationError as exc:
        return _build_only_error_record(
            stage_number,
            cache_directory=cache_directory,
            elapsed=time.perf_counter() - started,
            error=exc,
        )

    with _temporary_required_environment(effective_environment):
        try:
            if ttnn_module is None:
                import ttnn
            else:
                ttnn = ttnn_module
            device = ttnn.open_device(device_id=device_id)
            try:
                return run_build_only_stage(
                    ttnn,
                    device,
                    stage_number,
                    cache_path,
                    device_id=device_id,
                )
            finally:
                ttnn.close_device(device)
        except Exception as exc:  # noqa: BLE001 - preserve a flushed JSON failure record
            pre_manifest = cache_artifact_manifest(cache_path, stage_number)
            post_manifest = cache_artifact_manifest(cache_path, stage_number)
            stage = stage_for(stage_number)
            return _build_only_record(
                stage,
                cache_directory=cache_path,
                pre_manifest=pre_manifest,
                post_manifest=post_manifest,
                elapsed=time.perf_counter() - started,
                success=False,
                exit_code=1,
                status="fail",
                error={"code": "runtime_unavailable", "message": f"{type(exc).__name__}: {exc}"},
                device_id=device_id,
            )


def _emit_json(record: object, *, stream: Any = None) -> None:
    target = sys.stdout if stream is None else stream
    target.write(json.dumps(record, sort_keys=True) + "\n")
    target.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage", type=int)
    selection.add_argument("--construction-probe", choices=tuple(CONSTRUCTION_PROBES))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--dispatch-construction-probe",
        action="store_true",
        help="explicitly open the selected device for a zero-work construction dispatch",
    )
    parser.add_argument(
        "--build-only-jit-probe",
        action="store_true",
        help="build and dispatch stage 62 or 67 with early-return, without output download",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="caller-provided persistent per-run directory for TT_METAL_CACHE",
    )
    args = parser.parse_args()

    if args.build_only_jit_probe:
        if args.construction_probe is not None or args.dispatch_construction_probe:
            result = _build_only_error_record(
                None,
                cache_directory=args.cache_dir,
                elapsed=0.0,
                error=BuildOnlyProbeConfigurationError(
                    "path_conflict",
                    "build-only JIT probing cannot be combined with construction-probe options",
                ),
            )
        else:
            result = run_build_only_jit_probe(
                args.stage,
                cache_directory=args.cache_dir,
                device_id=args.device_id,
            )
        _emit_json(result)
        return int(result["exit_code"])

    if args.dispatch_construction_probe and args.construction_probe is None:
        parser.error("--dispatch-construction-probe requires --construction-probe")
    if args.construction_probe is not None:
        probe = construction_probe_for(args.construction_probe)
        if args.dispatch_construction_probe:
            try:
                result = dispatch_construction_probe(
                    probe,
                    allow_device_dispatch=True,
                    device_id=args.device_id,
                )
            except Exception as exc:  # noqa: BLE001 - keep the gate machine-readable
                result = construction_probe_record(probe)
                result.update(
                    {
                        "status": "fail",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        else:
            result = construction_probe_record(probe)
        _emit_json(result)
        return 0 if result["status"] in {"host-configured", "dispatched"} else 1

    try:
        stage = stage_for(args.stage)
        import ttnn

        device = ttnn.open_device(device_id=args.device_id)
        try:
            result = run_stage(ttnn, device, stage)
        finally:
            ttnn.close_device(device)
    except Exception as exc:  # noqa: BLE001 - emit a machine-readable failure record
        result = {
            "stage": args.stage,
            "name": STAGES[args.stage].name if args.stage in STAGES else None,
            "status": "fail",
            "batch": STAGES[args.stage].batch if args.stage in STAGES else None,
            "cores": STAGES[args.stage].cores if args.stage in STAGES else None,
            "tile_shape": [TILE, TILE],
            "elapsed_s": None,
            "numerical_error": None,
            "finite": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _emit_json(result)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
