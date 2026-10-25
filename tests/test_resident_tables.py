"""Board-free Issue #119 invariants; no TTNN or device access.

I3 includes fixed-work min/max/valid and per-frame budget exclusion for discarded
frames: an all-discard run has no work samples; delay D=3 removes three samples.
Only the run budget applies to a discarded frame.
"""

from dataclasses import replace

import pytest

from enodia.tt.bench.resident_harness import (
    ResidentConfig,
    ResidentPreflightError,
    RingAccounting,
    resident_l1_allocation_table,
    validate_configuration,
)
from enodia.tt.bench.resident_tables import (
    TableConsumerModel,
    frame_checksum,
    identity_at,
    metadata_words,
    table_word,
)


def config(**changes):
    base = ResidentConfig(
        frame_count=20, frame_interval_ticks=1000, producer_core=(0, 0),
        consumer_core=(1, 0), ring_pages=4, work_per_frame=64,
        designated_timestamp_core=(1, 0), cycle_budget=10000,
        outer_timeout_seconds=60, fixed_work_ticks_per_frame=100,
    )
    return replace(base, **changes)


def tables(**changes):
    return config(
        profile_tables=((0, 0, 0), (1, 0, 7), (1, 1, 0)),
        switch_schedule=((4, 1, 0), (10, 1, 1)),
        table_bytes=4096, **changes,
    )


@pytest.mark.parametrize("identity,accepted", [
    ((0, 0), True), ((1, 1), True), ((9, 0), False),
    ((0, 1), False), ((1, 9), False), ((9, 9), False),
])
def test_i1_full_identity_only(identity, accepted):
    model = TableConsumerModel(tables())
    result = model.consume(2, identity, 100)
    assert (result is not None) == accepted
    assert model.discard_count == int(not accepted)


@pytest.mark.parametrize("strategy", ["copy_on_demand", "prefetch"])
def test_i2_i3_readiness_delay_discard_resume_without_catchup(strategy):
    cfg = tables(swap_strategy=strategy)
    model = TableConsumerModel(cfg)
    ring = RingAccounting(2)
    consumed = []
    for frame in range(12):
        ring.reserve_producer()
        ring.publish()
        identity, _ = identity_at(cfg, frame)
        result = model.consume(frame, identity, (frame + 1) * 1000)
        ring.consume()
        if result is not None:
            consumed.append(frame)
    assert consumed == [0, 1, 2, 3, 7, 8, 9, 10, 11]
    assert model.discard_count == 3
    assert model.discards_per_switch == [3, 0]
    assert ring.producer_full_count == 0
    assert len(model.timestamps) == 9
    assert model.timestamps[4] == 8000
    assert model.switch_intervals == [4000, 1000]


def test_i4_checksum_is_frame_local_and_reads_selected_table():
    cfg = tables()
    model = TableConsumerModel(cfg)
    for frame, identity in [(0, (0, 0)), (10, (1, 1)), (11, (0, 0))]:
        expected = frame_checksum(cfg, frame, identity)
        assert model.consume(frame, identity, (frame + 1) * 1000) == expected
        accumulator = 0
        for work in range(cfg.work_per_frame):
            word = work % 1024
            payload = ((frame + word) & 0x7ff) | (((3 * frame + 5 * word) & 0x7ff) << 16)
            accumulator = ((accumulator * 33) ^ payload ^ (work + frame)
                           ^ table_word(identity, work % (cfg.table_bytes // 4))) & 0xffffffff
        assert expected == accumulator
    assert frame_checksum(cfg, 10, (0, 0)) != frame_checksum(cfg, 10, (1, 1))


def test_i5_schedule_and_tables_are_prelaunch_inputs():
    cfg = tables()
    words = metadata_words(cfg)
    assert len(words) == 1024
    assert words[:4] == [3, 2, 0, 0]
    assert words[4:8] == [0, 0, 0, 0]
    assert words[16:22] == [4, 1, 0, 10, 1, 1]
    assert identity_at(cfg, 3) == ((0, 0), 0)
    assert identity_at(cfg, 4) == ((1, 0), 1)


@pytest.mark.parametrize("strategy,copies", [("copy_on_demand", 1), ("prefetch", 2)])
def test_i6_i7_i8_l1_ledger_and_oversize_rejection(strategy, copies):
    cfg = tables(swap_strategy=strategy)
    validate_configuration(cfg)
    rows = resident_l1_allocation_table(
        ring_pages=4, table_bytes=cfg.table_bytes, swap_strategy=strategy,
    )
    assert next(row for row in rows if row["component"] == "profile_table_copies")["bytes"] == (
        cfg.table_bytes * copies
    )
    oversized = replace(cfg, table_bytes=1024 * 1024)
    with pytest.raises(ResidentPreflightError, match="L1"):
        validate_configuration(oversized)


@pytest.mark.parametrize("delay", [0, 1, 5, 10])
def test_readiness_discards_exactly_d_frames(delay):
    cfg = config(profile_tables=((0, 0, 0), (1, 0, 4 + delay)),
                 switch_schedule=((4, 1, 0),), table_bytes=4096)
    model = TableConsumerModel(cfg)
    for frame in range(cfg.frame_count):
        identity, _ = identity_at(cfg, frame)
        model.consume(frame, identity, (frame + 1) * 1000)
    assert model.discard_count == delay
    assert len(model.timestamps) == cfg.frame_count - delay
    assert model.discards_per_switch == [delay]


def test_unknown_identity_then_full_match_resumes_with_no_fallback():
    cfg = config(profile_tables=((0, 3, 0), (7, 3, 0)), initial_identity=(0, 3),
                 switch_schedule=((2, 9, 3), (5, 7, 3)), table_bytes=4096)
    model = TableConsumerModel(cfg)
    for frame in range(8):
        identity, _ = identity_at(cfg, frame)
        model.consume(frame, identity, (frame + 1) * 1000)
    assert model.timestamps == [1000, 2000, 6000, 7000, 8000]
    assert model.discards_per_switch == [3, 0]
    assert model.switch_intervals == [None, 4000]


def test_producer_overflow_and_identity_discard_are_distinct():
    cfg = tables()
    model = TableConsumerModel(cfg)
    ring = RingAccounting(2)
    for _ in range(3):
        slot = ring.reserve_producer()
        if slot is not None:
            ring.publish()
    assert ring.producer_full_count == 1
    assert model.discard_count == 0
    assert ring.consume() == 0
    assert model.consume(4, (1, 0), 1000) is None
    assert model.discard_count == 1
    assert ring.producer_full_count == 1
    assert ring.reserve_producer() == 0


def test_all_discard_has_no_work_samples_or_per_frame_budget_failure():
    """I3: even a large discard elapsed is not fixed consumer work."""
    cfg = config(table_bytes=4096)
    model = TableConsumerModel(cfg)
    for frame in range(cfg.frame_count):
        assert model.consume(frame, (0, 0), (frame + 1) * 1000,
                             elapsed_ticks=999, cycle_budget=100) is None
    assert model.work_count == 0
    assert model.work_valid is False
    assert model.work_min_ticks is None
    assert model.work_max_ticks is None
    assert model.work_budget_exceeded is False
    assert model.timestamps == []


def test_readiness_delay_three_excludes_discard_work_statistics():
    """I3: D=3 samples neither zero elapsed nor the oversized discard elapsed."""
    cfg = tables()
    model = TableConsumerModel(cfg)
    expected = []
    for frame in range(cfg.frame_count):
        identity, _ = identity_at(cfg, frame)
        elapsed = 9999 if 4 <= frame < 7 else 1062 + frame
        model.consume(frame, identity, (frame + 1) * 2000,
                      elapsed_ticks=elapsed, cycle_budget=1087)
        if not 4 <= frame < 7:
            expected.append(elapsed)
    assert model.discard_count == 3
    assert model.work_count == len(expected)
    assert model.work_count == cfg.frame_count - 3
    assert model.work_valid is True
    assert model.work_min_ticks == min(expected)
    assert model.work_max_ticks == max(expected)
    assert model.work_budget_exceeded is False


@pytest.mark.parametrize("ready", [False, True])
def test_work_budget_applies_only_to_processed_frames_and_keeps_inclusive_boundary(ready):
    cfg = config(profile_tables=((0, 0, 0),) if ready else (), table_bytes=4096)
    model = TableConsumerModel(cfg)
    model.consume(0, (0, 0), 1000, elapsed_ticks=100, cycle_budget=100)
    assert model.work_budget_exceeded is ready


def test_discard_still_receives_run_budget_check():
    model = TableConsumerModel(config(table_bytes=4096))
    model.consume(0, (0, 0), 1000, elapsed_ticks=999, cycle_budget=100,
                  run_elapsed_ticks=500, run_budget_ticks=500)
    assert model.work_budget_exceeded is False
    assert model.run_budget_exceeded is True
    assert model.work_valid is False


def test_i9_empty_schedule_preserves_legacy_config_and_ring():
    cfg = config()
    assert "profile_tables" not in cfg.as_record()
    assert "switch_schedule" not in cfg.as_record()
    assert "table_bytes" not in cfg.as_record()
    assert resident_l1_allocation_table(ring_pages=4) == resident_l1_allocation_table(
        ring_pages=4, table_bytes=0, swap_strategy="copy_on_demand",
    )
    ring = RingAccounting(2)
    ring.reserve_producer()
    ring.publish()
    assert ring.consume() == 0
    assert ring.drain_complete(True)


@pytest.mark.parametrize("changes", [
    {"table_bytes": 1}, {"table_bytes": True}, {"swap_strategy": "unknown"},
    {"profile_tables": ((0, 0, 0), (0, 0, 4))},
    {"switch_schedule": ((5, 0, 0), (4, 1, 0))},
    {"switch_schedule": ((20, 0, 0),)},
    {"switch_schedule": ((True, 0, 0),)},
    {"initial_identity": (-1, 0)},
])
def test_invalid_parameters_reject_before_execution(changes):
    invalid = replace(tables(), **changes)
    with pytest.raises(ResidentPreflightError):
        validate_configuration(invalid)
