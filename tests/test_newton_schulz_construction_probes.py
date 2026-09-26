import json
from dataclasses import fields

import pytest

from tools import newton_schulz_bringup as bringup

PROBE_NAMES = ("P0", "P1", "P2", "P3a", "P3b", "P4", "P5", "P6", "P7", "P8")


def _probe_fields(probe):
    ignored = {"name", "purpose"}
    return {
        field.name: getattr(probe, field.name)
        for field in fields(probe)
        if field.name not in ignored
    }


def _changed_fields(left, right):
    return {
        name for name, value in _probe_fields(left).items() if value != _probe_fields(right)[name]
    }


def test_probe_sequence_is_host_only_and_has_the_fixed_dispatch_shape():
    assert tuple(bringup.CONSTRUCTION_PROBES) == PROBE_NAMES
    for name in PROBE_NAMES:
        probe = bringup.construction_probe_for(name)
        assert probe.batch == 1
        assert probe.core_count == 1
        assert probe.core_coordinates == ((0, 0),)
        assert probe.core_ranges == (((0, 0), (0, 0)),)
        assert probe.tiles_per_core == 1
        assert probe.tile_count == 0
        assert probe.iterations == 8
        assert probe.fp32_dest_acc_en is True
        assert probe.output_dtypes == ("float32", "float32")
        assert probe.reader_accessor_count == 6
        assert probe.writer_accessor_count == 2
        assert probe.reader_runtime_arg_count == 8
        assert probe.writer_runtime_arg_count == 4
        assert probe.compute_compile_args == (1,)
        assert probe.compute_runtime_args == ()
        assert probe.kernel_order == ("reader", "writer", "compute")
        assert probe.semaphores == ()
        assert probe.external_timeout_seconds == 60
        assert len(probe.cb_formats) == 25
        assert len(probe.cb_page_sizes) == 25
        assert probe.cb_page_sizes == tuple(
            bringup.TILE_BYTES_FLOAT32 if fmt == "float32" else bringup.TILE_BYTES_BFLOAT16
            for fmt in probe.cb_formats
        )
        assert len(probe.active_cb_indices) in (18, 24, 25)
        assert all(path.is_file() for path in bringup.construction_source_paths(probe))


def test_p0_to_p2_isolates_input_and_descriptor_construction_changes():
    p0, p1, p2 = (bringup.construction_probe_for(name) for name in ("P0", "P1", "P2"))
    assert _changed_fields(p0, p1) == {"input_dtypes"}
    assert [
        index
        for index, (before, after) in enumerate(zip(p0.input_dtypes, p1.input_dtypes))
        if before != after
    ] == [1, 3]
    assert p0.input_dtypes[0] == p1.input_dtypes[0] == "bfloat16"
    assert [p0.input_dtypes[index] for index in (2, 4, 5)] == [
        p1.input_dtypes[index] for index in (2, 4, 5)
    ]
    assert _changed_fields(p1, p2) == {"cb_formats", "cb_page_sizes"}
    assert [
        index
        for index, (before, after) in enumerate(zip(p1.cb_formats, p2.cb_formats))
        if before != after
    ] == [14, 15, 20, 21]
    assert [
        index
        for index, (before, after) in enumerate(zip(p1.cb_page_sizes, p2.cb_page_sizes))
        if before != after
    ] == [14, 15, 20, 21]
    assert p2.cb_formats[14:16] == ("float32", "float32")
    assert p2.cb_formats[20:22] == ("bfloat16", "bfloat16")
    assert p2.cb_page_sizes[14:16] == (bringup.TILE_BYTES_FLOAT32,) * 2
    assert p2.cb_page_sizes[20:22] == (bringup.TILE_BYTES_BFLOAT16,) * 2


def test_p3_pair_keeps_the_same_noop_source_and_changes_only_active_cb_indices():
    p3a, p3b = (bringup.construction_probe_for(name) for name in ("P3a", "P3b"))
    assert _changed_fields(p3a, p3b) == {"active_cb_indices"}
    assert p3a.active_cb_indices == bringup.STAGE_61_ACTIVE_CB_INDICES
    assert p3b.active_cb_indices == bringup.STAGE_62_ACTIVE_CB_INDICES
    assert len(p3a.active_cb_indices) == 18
    assert len(p3b.active_cb_indices) == 24
    assert bringup.construction_source_paths(p3a) == bringup.construction_source_paths(p3b)
    for source in bringup.construction_source_paths(p3a):
        text = source.read_text()
        assert "matmul" not in text
        assert "copy_tile" not in text
        assert "binary_op" not in text


def test_p4_to_p8_changes_only_the_next_approved_boundary():
    p3b, p4, p5, p6, p7, p8 = (
        bringup.construction_probe_for(name) for name in ("P3b", "P4", "P5", "P6", "P7", "P8")
    )
    assert _changed_fields(p3b, p4) == {"reader_source"}
    assert _changed_fields(p4, p5) == {"compute_source"}
    assert _changed_fields(p5, p6) == {"cb_formats", "cb_page_sizes"}
    assert _changed_fields(p6, p7) == {"reader_source", "active_cb_indices"}
    assert _changed_fields(p7, p8) == {"compute_source"}
    assert [
        index
        for index, (before, after) in enumerate(zip(p5.cb_formats, p6.cb_formats))
        if before != after
    ] == [13]
    assert [
        index
        for index, (before, after) in enumerate(zip(p5.cb_page_sizes, p6.cb_page_sizes))
        if before != after
    ] == [13]
    assert p5.cb_formats[13] == "bfloat16"
    assert p6.cb_formats[13] == "float32"
    assert p5.cb_page_sizes[13] == bringup.TILE_BYTES_BFLOAT16
    assert p6.cb_page_sizes[13] == bringup.TILE_BYTES_FLOAT32
    assert p6.active_cb_indices == bringup.STAGE_62_ACTIVE_CB_INDICES
    assert p7.active_cb_indices == p8.active_cb_indices == bringup.STAGE_67_ACTIVE_CB_INDICES


def test_probe_record_is_serializable_and_makes_no_numerical_or_board_claim():
    record = bringup.construction_probe_record("P8")
    assert record["status"] == "host-configured"
    assert record["board_run"] is False
    assert record["numerical_acceptance"] is False
    assert record["tile_count"] == 0
    assert record["compute_compile_args"] == [1]
    assert record["compute_runtime_args"] == []
    assert record["semaphores"] == []
    assert record["core_range_count"] == 1
    assert record["core_ranges"] == [[[0, 0], [0, 0]]]
    assert record["cb_descriptor_count"] == 25
    assert record["cb_page_count"] == 4
    assert record["cb_descriptor_total_bytes"] == 352256
    json.loads(json.dumps(record))


def test_future_dispatch_is_explicitly_gated_and_accepts_only_an_injected_runner():
    with pytest.raises(PermissionError, match="disabled"):
        bringup.dispatch_construction_probe("P0")
    with pytest.raises(RuntimeError, match="dispatcher"):
        bringup.dispatch_construction_probe("P0", allow_device_dispatch=True)
    seen = []
    result = bringup.dispatch_construction_probe(
        "P0",
        allow_device_dispatch=True,
        dispatcher=lambda probe: seen.append(probe.name) or "future-result",
    )
    assert result == "future-result"
    assert seen == ["P0"]
