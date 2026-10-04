"""Board-free source audit for the L=16 tile decision in Issue #63."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _tt_metal_root() -> Path:
    configured = os.environ.get("HEKATUS_TT_METAL_ROOT")
    if not configured:
        pytest.skip("set HEKATUS_TT_METAL_ROOT to a tt-metal source checkout for this audit")
    candidate = Path(configured)
    if not ((candidate / ".git").exists() and (candidate / "tt_metal").is_dir()):
        pytest.fail(f"HEKATUS_TT_METAL_ROOT is not a tt-metal checkout: {candidate}")
    return candidate


def _source(root: Path, relative: str) -> str:
    path = root / relative
    assert path.is_file(), f"audit source is missing: {path}"
    return path.read_text()


def test_host_apis_admit_16x16_descriptor_and_cb_tile_metadata():
    """16x16 is representable in the host descriptor and CB metadata APIs."""
    root = _tt_metal_root()
    descriptor = _source(root, "tt_metal/api/tt-metalium/program_descriptors.hpp")
    tile = _source(root, "tt_metal/impl/data_format/tile.cpp")
    nanobind = _source(root, "ttnn/cpp/ttnn-nanobind/program_descriptors.cpp")
    tensor = _source(root, "ttnn/core/tensor/tensor.cpp")

    assert "TileDescriptor(uint32_t height, uint32_t width, bool transpose)" in descriptor
    assert "std::optional<TileDescriptor> tile;" in descriptor
    assert "{{{16, 16}, {16, 16}}}" in tile
    assert 'TT_THROW("Tile size is not valid for our hardware")' in tile
    assert "nb::init<uint32_t, uint32_t, bool>()" in nanobind
    assert "nb::init<uint8_t, tt::DataFormat, uint32_t, std::optional<tt::tt_metal::TileDescriptor>>()" in nanobind
    assert "only matmul op and ccl all-gather currently supports the customized tile shape" in tensor


def test_blackhole_standard_matmul_apis_do_not_accept_native_16x16():
    """The standard matmul_tiles/matmul_block path rejects two 16x16 operands."""
    root = _tt_metal_root()
    api = _source(root, "tt_metal/hw/inc/api/compute/matmul.h")
    blackhole_matmul = _source(root, "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h")

    assert "ALWI void matmul_tiles(" in api
    assert "ALWI void matmul_block(" in api
    assert "MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)))" in api
    assert "MATH((matmul_block_math_dynamic_throttle(" in api
    assert "16x16 inputs not supported" in blackhole_matmul
    assert '"16x16 by 16x16 matmul is not supported"' in blackhole_matmul


def test_audit_has_no_machine_local_checkout_dependency():
    design = (Path(__file__).parents[1] / "docs" / "design.md").read_text()
    assert "901dd9ce93816ffd1fd185b801fc727065e9ae07" in design
    assert "image contains no source-revision metadata" in design
    assert "same" in design and "16x16 by 16x16 matmul is not supported" in design


def test_audit_conclusion_requires_the_32x32_diagonal_fallback():
    """Host API support alone must not be mistaken for native matmul support."""
    root = _tt_metal_root()
    blackhole_matmul = _source(root, "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h")
    assert "no dedicated math path" in blackhole_matmul
    assert "falls to 32x32 default which is incorrect for < 4 faces" in blackhole_matmul
