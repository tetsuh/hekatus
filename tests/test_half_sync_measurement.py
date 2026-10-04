from types import SimpleNamespace

import pytest

from enodia.tt.bench import newton_schulz_kernel
from enodia.tt.bench.half_sync import half_sync_row_manifest, preflight_half_sync_rows


@pytest.fixture
def ttnn():
    return SimpleNamespace(bfloat16="bf16", float32="fp32", uint32="u32")


def test_cb_page_count_helper_preserves_all_descriptor_page_rules():
    resident = {newton_schulz_kernel.CB_IDENTITY}
    double_buffered = {newton_schulz_kernel.CB_OUTPUT_REAL}
    assert newton_schulz_kernel._cb_page_count(
        newton_schulz_kernel.CB_IDENTITY,
        1,
        resident=resident,
        double_buffered=double_buffered,
        matrix_block=4,
        matrix_queue_pages=8,
        state_queue_pages=4,
        fuse_s=True,
    ) == 1
    assert newton_schulz_kernel._cb_page_count(
        newton_schulz_kernel.CB_OUTPUT_REAL,
        2,
        resident=resident,
        double_buffered=double_buffered,
        matrix_block=4,
        matrix_queue_pages=8,
        state_queue_pages=4,
        fuse_s=True,
    ) == 8
    assert newton_schulz_kernel._cb_page_count(
        newton_schulz_kernel.CB_STATE_REAL,
        2,
        resident=resident,
        double_buffered=double_buffered,
        matrix_block=4,
        matrix_queue_pages=8,
        state_queue_pages=4,
        fuse_s=True,
    ) == 4


def test_half_sync_manifest_records_dest_admission_for_both_variants_and_sizes():
    rows = half_sync_row_manifest()
    assert len(rows) == 32
    assert {(row["size"], row["variant"]) for row in rows} == {
        (16, "bf16"),
        (16, "bf16-fp32state"),
        (32, "bf16"),
        (32, "bf16-fp32state"),
    }
    admitted = [row for row in rows if row["dest_status"] == "admitted"]
    assert {
        (row["dst_full_sync_en"], row["matrix_block"])
        for row in admitted
    } == {(True, 1), (True, 2), (True, 4), (True, 8), (False, 1), (False, 2)}
    assert all(row["fp32_dest_acc_en"] is True for row in rows)
    assert all(row["double_buffer"] is True for row in rows)
    assert all(row["fuse_s"] is True for row in rows)
    assert all(row["math_fidelity"] == "HiFi3" for row in rows)
    assert all(row["input_memory"] == "l1" for row in rows)
    assert all(row["output_memory"] == "dram" for row in rows)


def test_half_sync_preflight_marks_rejected_rows_without_device_execution(ttnn):
    rows = preflight_half_sync_rows(ttnn)
    assert len(rows) == 32
    assert all("preflight_status" in row for row in rows)
    assert all("preflight_reason" in row for row in rows)
    assert all(
        row["preflight_status"] == "rejected"
        for row in rows
        if row["dest_status"] == "rejected"
    )
    assert all(
        row["preflight_bytes"] is not None
        for row in rows
        if row["dest_status"] == "admitted" and row["preflight_status"] == "admitted"
    )


def test_dest_limits_are_explicit_for_full_and_half_sync():
    assert newton_schulz_kernel._dest_slot_limit(
        fp32_dest_acc_en=True, dst_full_sync_en=True
    ) == 8
    assert newton_schulz_kernel._dest_slot_limit(
        fp32_dest_acc_en=True, dst_full_sync_en=False
    ) == 4
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_matrix_block(
            4, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
