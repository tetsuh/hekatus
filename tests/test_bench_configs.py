"""Host-only tests for the explicit stock matmul configuration catalogue."""

from dataclasses import replace

import pytest

from enodia.tt.bench.configs import (
    P150_COMPUTE_GRID,
    ProgramConfigSpec,
    configuration_catalogue,
    executed_shape,
    validate_config,
)
from enodia.tt.bench.shapes import MatmulShape, default_catalogue, total_flops


def _shape(*, batch: int = 1, m: int = 64, k: int = 64, n: int = 64) -> MatmulShape:
    return MatmulShape(
        name="probe",
        batch=batch,
        m=m,
        k=k,
        n=n,
        real_matmuls=1,
        family="newton_schulz",
        note="",
    )


def _reuse(**changes) -> ProgramConfigSpec:
    config = ProgramConfigSpec(
        name="reuse_g1x1_k1_m2_n2_s2x2",
        kind="reuse",
        grid=(1, 1),
        in0_block_w=1,
        out_subblock_h=2,
        out_subblock_w=2,
        per_core_m=2,
        per_core_n=2,
    )
    return replace(config, **changes)


def test_program_config_rejects_a_grid_the_board_would_reject():
    invalid = _reuse(grid=(P150_COMPUTE_GRID[0] + 1, P150_COMPUTE_GRID[1]))

    with pytest.raises(ValueError, match="compute grid"):
        validate_config(_shape(), invalid)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_reuse(in0_block_w=3), "K tiles"),
        (_reuse(out_subblock_h=3), "per-core M"),
        (_reuse(out_subblock_h=2, out_subblock_w=4), "subblock"),
        (_reuse(per_core_m=3, out_subblock_h=1), "M tiles"),
    ],
)
def test_program_config_rejects_dimension_combinations_ttnn_rejects(config, message):
    with pytest.raises(ValueError, match=message):
        validate_config(_shape(), config)


@pytest.mark.parametrize(
    ("m", "expected_grid"),
    [(320, (10, 1)), (384, (6, 2))],
)
def test_rectangular_mcast_1d_grids_cover_exactly(m, expected_grid):
    shape = _shape(m=m, k=96, n=160)

    configs = configuration_catalogue(shape)
    multicast = [config for config in configs if config.kind == "mcast_1d"]

    assert multicast
    assert {config.grid for config in multicast} == {expected_grid}
    for config in multicast:
        validate_config(shape, config)
        assert config.per_core_n == (shape.n + 31) // 32
        assert config.per_core_m * config.grid[0] * config.grid[1] == (shape.m + 31) // 32


def test_every_catalogued_configuration_is_valid_for_its_shape():
    representative = [shape for shape in default_catalogue() if shape.representative]

    assert representative
    for shape in representative:
        configs = configuration_catalogue(shape)
        assert configs, shape.name
        assert len({config.name for config in configs}) == len(configs), shape.name
        for config in configs:
            validate_config(shape, config)


def test_catalogue_covers_reuse_multicast_and_dram_sharded_programs():
    kinds = {
        config.kind
        for shape in default_catalogue()
        if shape.representative
        for config in configuration_catalogue(shape)
    }

    assert {
        "reuse",
        "mcast_1d",
        "mcast_2d",
        "dram_sharded",
        "batched_dram_sharded",
    } <= kinds


def test_multicast_is_not_offered_for_independently_batched_operands():
    shape = _shape(batch=1024)

    assert all(
        config.kind not in {"mcast_1d", "mcast_2d"} for config in configuration_catalogue(shape)
    )


def test_unbatched_dram_sharding_is_only_offered_for_one_tile_high_outputs():
    one_tile = _shape(m=16, k=128, n=4096)
    two_tiles = _shape(m=64, k=64, n=64)

    assert any(config.kind == "dram_sharded" for config in configuration_catalogue(one_tile))
    assert all(config.kind != "dram_sharded" for config in configuration_catalogue(two_tiles))


def test_batch_sharding_counts_the_padding_it_executes():
    shape = _shape(batch=1024, m=32, k=32, n=32)
    config = next(
        config for config in configuration_catalogue(shape) if config.kind == "batched_dram_sharded"
    )

    execution = executed_shape(shape, config)

    assert execution.batch == 1029
    assert total_flops(execution) == 1029 * 2 * 32**3
