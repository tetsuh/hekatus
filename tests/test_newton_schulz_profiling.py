from pathlib import Path
from types import SimpleNamespace

import numpy as np

from enodia.tt.bench.newton_schulz_kernel import (
    PROFILE_COMPLEX_IMAG_OFFSET,
    PROFILE_COMPLEX_REAL_OFFSET,
    PROFILE_MAGIC,
    PROFILE_PACK_PUSH_OFFSET,
    PROFILE_PAGES_PER_CORE,
    PROFILE_READY_OFFSET,
    PROFILE_S_BINARY_OFFSET,
    PROFILE_SECTION_SUM_OFFSET,
    PROFILE_SLOT_STRIDE,
    PROFILE_STATE_HANDOFF_OFFSET,
    PROFILE_WARMUP_BASE,
    PROFILE_WARMUP_EVENT_COUNT_OFFSET,
    PROFILE_WARMUP_READY_OFFSET,
    NewtonSchulzKernel,
)
from enodia.tt.bench.profiling import parse_device_profile_csv

ROOT = Path(__file__).parents[1]
KERNEL_DIR = ROOT / "enodia/tt/bench/kernels"


def test_profiler_markers_cover_all_required_kernel_owners():
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    reader = (KERNEL_DIR / "newton_schulz_reader.cpp").read_text()
    writer = (KERNEL_DIR / "newton_schulz_writer.cpp").read_text()

    assert '#include "tools/profiler/kernel_profiler.hpp"' in compute
    for marker in (
        "NS-COMPUTE-TOTAL",
        "NS-COMPUTE-R-CB-WAIT",
        "NS-COMPUTE-X-CB-WAIT",
        "NS-COMPUTE-COMPLEX-REAL",
        "NS-COMPUTE-COMPLEX-IMAG",
        "NS-COMPUTE-S-BINARY",
        "NS-COMPUTE-PACK-PUSH",
        "NS-COMPUTE-STATE-HANDOFF",
    ):
        assert marker in compute
    assert "NS-READER-CONSTANT-READ" in reader
    assert "NS-READER-READ-AND-WAIT" in reader
    assert "NS-WRITER-WRITES" in writer


def test_profile_flag_defaults_off_in_host_parser():
    from enodia.tt.bench import run_matmul

    assert run_matmul._build_parser().parse_args([]).profile is False
    assert run_matmul._build_parser().parse_args(["--profile"]).profile is True


def test_cycle_counter_profile_uses_l1_transport_and_timestamps():
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    reader = (KERNEL_DIR / "newton_schulz_reader_profile.cpp").read_text()
    writer = (KERNEL_DIR / "newton_schulz_writer_profile.cpp").read_text()

    assert "get_timestamp_32b()" in compute
    assert "get_timestamp_32b()" in reader
    assert "get_timestamp_32b()" in writer
    assert "cb_profile_compute" in compute
    assert "tt_l1_ptr" in compute
    assert "cb_profile_reader" in reader
    assert "cb_profile_writer" in writer
    assert "TT_METAL_DEVICE_PROFILER" not in compute + reader + writer
    assert "start_tile == 0" in compute
    assert "start_tile == 0" in reader
    assert "start_tile == 0" in writer
    assert "if (!measure_core)" in writer
    assert "const bool measure_core = start_tile == 0" in reader
    assert "read_tile_profiled" in reader
    assert "profile_core" in compute
    assert "event_count" in compute
    assert "warmup_event_count" in compute
    assert "if (start_tile == 0)" in compute
    assert "TensorAccessorArgs<0>()" in writer
    assert "get_arg_val<std::uint32_t>(2)" in writer
    assert "get_arg_val<std::uint32_t>(3)" in writer
    assert "get_arg_val<std::uint32_t>(4)" in writer
    assert "get_arg_val<std::uint32_t>(5)" in writer
    assert "profile_cb_wait_offset" in reader
    assert "profile_noc_read_offset" in reader
    assert "profile_cb_wait_offset" in writer
    assert "profile_noc_write_offset" in writer
    assert "compute_profile[profile_ready_offset] = 0" in writer
    assert "void clear_profile_ready" in compute
    assert "clear_profile_ready(start_tile)" in compute
    slot_source = compute[compute.index("void write_profile_slot"):]
    assert "COMPILE_FOR_TRISC == 0" in slot_source
    assert "COMPILE_FOR_TRISC == 1" in slot_source
    assert "COMPILE_FOR_TRISC == 2" in slot_source
    assert "profile[base + profile_ready_offset] = profile_magic" in slot_source
    assert "profile[warmup_base + profile_ready_offset] = profile_magic" in slot_source
    assert "profile_warmup_ready_offset" in slot_source
    assert "profile_warmup_event_count_offset" in slot_source
    assert "while (profile[profile_ready_offset] != profile_magic" in slot_source
    assert "cb_push_back(cb_profile_compute, 1)" in slot_source
    slot0 = slot_source.split("COMPILE_FOR_TRISC == 0", 1)[1].split("COMPILE_FOR_TRISC == 1", 1)[0]
    slot1 = slot_source.split("COMPILE_FOR_TRISC == 1", 1)[1].split("COMPILE_FOR_TRISC == 2", 1)[0]
    slot2 = slot_source.split("COMPILE_FOR_TRISC == 2", 1)[1]
    assert "profile_slot_stride +" not in slot0
    assert "profile,\n        profile_slot_stride" in slot1
    assert "profile[profile_total_offset]" not in slot2
    assert "cb_push_back(cb_profile_compute, 1)" not in slot0 + slot1



def test_profile_false_has_no_profile_cbs_or_tracy_environment():
    from enodia.tt.bench import newton_schulz_kernel, run_matmul

    ttnn = SimpleNamespace(bfloat16="bf16", float32="fp32", uint32="u32")
    normal = newton_schulz_kernel._cb_definitions(ttnn, ttnn.bfloat16)
    profiled = newton_schulz_kernel._cb_definitions(ttnn, ttnn.bfloat16, profile=True)
    assert not set(normal).intersection(
        {
            newton_schulz_kernel.CB_PROFILE_READER,
            newton_schulz_kernel.CB_PROFILE_COMPUTE,
            newton_schulz_kernel.CB_PROFILE_WRITER,
        }
    )
    assert set(profiled).issuperset(
        {
            newton_schulz_kernel.CB_PROFILE_READER,
            newton_schulz_kernel.CB_PROFILE_COMPUTE,
            newton_schulz_kernel.CB_PROFILE_WRITER,
        }
    )
    assert newton_schulz_kernel._cb_page_size(ttnn, ttnn.uint32) == 32 * 32 * 4
    runner_source = Path(run_matmul.__file__).read_text()
    assert "TT_METAL_DEVICE_PROFILER" not in runner_source
    assert "all_tiles_all_iterations_with_core_0_first_tile_iteration_warmup" in runner_source
    assert "profile_shape = ttnn.Shape((PROFILE_PAGES_PER_CORE, 1, 1, PROFILE_PAGE_WORDS))" in Path(
        newton_schulz_kernel.__file__
    ).read_text()
    assert "ttnn.ROW_MAJOR_LAYOUT" in Path(newton_schulz_kernel.__file__).read_text()


def test_row_major_profile_pages_restore_risc_slots_and_scopes(monkeypatch):
    raw = np.zeros((PROFILE_PAGES_PER_CORE, 1, 1, 32 * 32), dtype=np.uint32)
    pages = raw.reshape(PROFILE_PAGES_PER_CORE, -1)

    # Reader page: all scope and first-tile warmup scope.
    pages[0, 0:3] = [100, 70, 30]
    pages[0, 8:12] = [100, 0, 4, 4]
    pages[0, 32:35] = [20, 12, 8]
    pages[0, 32 + 8 : 32 + 12] = [20, 0, 1, 1]
    pages[0, PROFILE_READY_OFFSET] = PROFILE_MAGIC
    pages[0, PROFILE_WARMUP_READY_OFFSET] = PROFILE_MAGIC

    # Compute slots: each slot owns one TRISC's sections and has both scopes.
    for base, total in ((0, 10), (PROFILE_SLOT_STRIDE, 20), (2 * PROFILE_SLOT_STRIDE, 30)):
        pages[1, base + 0] = total
        pages[1, base + 8] = total
        pages[1, base + 9] = 0
        pages[1, base + 10] = 16
        pages[1, base + PROFILE_READY_OFFSET] = PROFILE_MAGIC
        pages[1, base + PROFILE_WARMUP_BASE + 0] = 4
        pages[1, base + PROFILE_WARMUP_BASE + 8] = 4
        pages[1, base + PROFILE_WARMUP_BASE + 9] = 0
        pages[1, base + PROFILE_WARMUP_BASE + PROFILE_WARMUP_EVENT_COUNT_OFFSET] = 1
        pages[1, base + PROFILE_WARMUP_READY_OFFSET] = PROFILE_MAGIC
    pages[1, PROFILE_SLOT_STRIDE + PROFILE_COMPLEX_REAL_OFFSET] = 7
    pages[1, PROFILE_SLOT_STRIDE + PROFILE_COMPLEX_IMAG_OFFSET] = 13
    pages[1, PROFILE_SLOT_STRIDE + PROFILE_WARMUP_BASE + PROFILE_COMPLEX_REAL_OFFSET] = 4
    pages[1, 2 * PROFILE_SLOT_STRIDE + PROFILE_PACK_PUSH_OFFSET] = 11

    # Writer page: waits and writes are disjoint sections.
    pages[2, 0:3] = [40, 15, 25]
    pages[2, 8:12] = [40, 0, 4, 4]
    pages[2, 32:35] = [8, 3, 5]
    pages[2, 32 + 8 : 32 + 12] = [8, 0, 1, 1]
    pages[2, PROFILE_READY_OFFSET] = PROFILE_MAGIC
    pages[2, PROFILE_WARMUP_READY_OFFSET] = PROFILE_MAGIC

    kernel = object.__new__(NewtonSchulzKernel)
    kernel.ttnn = None
    kernel.profile_output = object()
    kernel.work_ranges = [(0, 1)]
    monkeypatch.setattr(
        "enodia.tt.bench.newton_schulz_kernel._download_uint32",
        lambda _ttnn, _tensor: raw,
    )

    records = kernel.profile_records()
    by_risc = {record["risc"]: record for record in records}

    assert {risc: by_risc[risc]["total_cycles"] for risc in ("TRISC0", "TRISC1", "TRISC2")} == {
        "TRISC0": 10,
        "TRISC1": 20,
        "TRISC2": 30,
    }
    assert by_risc["TRISC1"]["sections"][2]["cycles"] == 7
    assert by_risc["TRISC2"]["sections"][5]["cycles"] == 11
    assert by_risc["TRISC1"]["warmup_total_cycles"] == 4
    assert by_risc["TRISC1"]["consistency_pass"]
    assert by_risc["TRISC1"]["warmup_consistency_pass"]
    assert by_risc["NCRISC"]["consistency_pass"]
    assert by_risc["BRISC"]["consistency_pass"]
    assert by_risc["NCRISC"]["sections"][0]["name"] == "reader_cb_empty_wait"
    assert by_risc["BRISC"]["sections"][1]["name"] == "writer_noc_write_and_barrier"
    assert all(record["profile_page_ready"] for record in records)


def test_cycle_counter_profile_records_decode_l1_pages(monkeypatch):
    raw = np.zeros((PROFILE_PAGES_PER_CORE, 32 * 32), dtype=np.uint32)
    pages = raw.reshape(PROFILE_PAGES_PER_CORE, -1)
    pages[0, 0:3] = [120, 80, 40]
    pages[0, 8:12] = [120, 0, 2, 1]
    pages[0, 32:35] = [40, 25, 15]
    pages[0, 32 + 8 : 32 + 12] = [40, 0, 1, 1]
    pages[0, PROFILE_READY_OFFSET] = PROFILE_MAGIC
    pages[0, PROFILE_WARMUP_READY_OFFSET] = PROFILE_MAGIC
    for base in (0, PROFILE_SLOT_STRIDE, 2 * PROFILE_SLOT_STRIDE):
        pages[1, base + 0] = 300
        pages[1, base + 8] = 300
        pages[1, base + 10] = 16
        pages[1, base + PROFILE_WARMUP_BASE + 0] = 30
        pages[1, base + PROFILE_WARMUP_BASE + 8] = 30
        pages[1, base + PROFILE_WARMUP_BASE + PROFILE_WARMUP_EVENT_COUNT_OFFSET] = 1
        pages[1, base + PROFILE_READY_OFFSET] = PROFILE_MAGIC
        pages[1, base + PROFILE_WARMUP_READY_OFFSET] = PROFILE_MAGIC
    pages[1, PROFILE_SLOT_STRIDE + PROFILE_COMPLEX_REAL_OFFSET] = 100
    pages[1, PROFILE_SLOT_STRIDE + PROFILE_COMPLEX_IMAG_OFFSET] = 120
    pages[1, PROFILE_SLOT_STRIDE + PROFILE_S_BINARY_OFFSET] = 40
    pages[1, PROFILE_SLOT_STRIDE + PROFILE_STATE_HANDOFF_OFFSET] = 40
    pages[1, PROFILE_SLOT_STRIDE + PROFILE_SECTION_SUM_OFFSET] = 300
    pages[1, 2 * PROFILE_SLOT_STRIDE + PROFILE_PACK_PUSH_OFFSET] = 300
    pages[2, 0:3] = [75, 35, 40]
    pages[2, 8:12] = [75, 0, 2, 1]
    pages[2, 32:35] = [20, 8, 12]
    pages[2, 32 + 8 : 32 + 12] = [20, 0, 1, 1]
    pages[2, PROFILE_READY_OFFSET] = PROFILE_MAGIC
    pages[2, PROFILE_WARMUP_READY_OFFSET] = PROFILE_MAGIC

    kernel = object.__new__(NewtonSchulzKernel)
    kernel.ttnn = None
    kernel.profile_output = object()
    kernel.work_ranges = [(0, 1)]
    monkeypatch.setattr(
        "enodia.tt.bench.newton_schulz_kernel._download_uint32",
        lambda _ttnn, _tensor: raw,
    )

    records = kernel.profile_records()

    assert {record["risc"] for record in records} == {"NCRISC", "TRISC0", "TRISC1", "TRISC2", "BRISC"}
    assert {record["measurement_core"] for record in records} == {0}
    assert len(records) == 5
    compute = next(record for record in records if record["risc"] == "TRISC1")
    assert compute["total_cycles"] == 300
    assert {section["name"] for section in compute["sections"]} >= {
        "complex_real",
        "complex_imag",
        "s_binary",
        "state_handoff",
    }
    writer = next(record for record in records if record["risc"] == "BRISC")
    assert writer["sections"][0]["name"] == "writer_cb_wait_front"
    assert writer["consistency_exact"] is True
    assert all(record["profile_page_ready"] for record in records)


def test_profile_records_keep_counter_values_when_ready_markers_are_missing(monkeypatch):
    raw = np.zeros((PROFILE_PAGES_PER_CORE, 32 * 32), dtype=np.uint32)
    raw[0, 0:4] = [120, 80, 40, 2]
    raw[1, PROFILE_SLOT_STRIDE : PROFILE_SLOT_STRIDE + 9] = [300, 20, 30, 100, 120, 40, 0, 40, 1]
    raw[2, 0:3] = [75, 75, 2]

    kernel = object.__new__(NewtonSchulzKernel)
    kernel.ttnn = None
    kernel.profile_output = object()
    kernel.work_ranges = [(0, 1)]
    monkeypatch.setattr(
        "enodia.tt.bench.newton_schulz_kernel._download_uint32",
        lambda _ttnn, _tensor: raw,
    )

    records = kernel.profile_records()

    assert not any(record["profile_page_ready"] for record in records)
    trisc1 = next(record for record in records if record["risc"] == "TRISC1")
    assert trisc1["total_cycles"] == 300
    assert "residual_cycles" in trisc1
    assert trisc1["consistency_record_valid"] is False


def test_profile_csv_parser_returns_per_risc_cycles_and_percentages(tmp_path: Path):
    path = tmp_path / "profile_log_device.csv"
    path.write_text(
        """ARCH: blackhole, CHIP_FREQ[MHz]: 1200
PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],stat value,Run ID,zone name,zone phase,source line,source file
0,1,2,TRISC0,10,100,0,0,NS-COMPUTE-COMPLEX-REAL,begin,1,kernel.cpp
0,1,2,TRISC0,10,160,0,0,NS-COMPUTE-COMPLEX-REAL,end,1,kernel.cpp
0,1,2,TRISC0,11,200,0,0,NS-COMPUTE-PACK-PUSH,begin,2,kernel.cpp
0,1,2,TRISC0,11,240,0,0,NS-COMPUTE-PACK-PUSH,end,2,kernel.cpp
0,1,2,TRISC1,12,300,0,0,NS-COMPUTE-S-BINARY,begin,3,kernel.cpp
0,1,2,TRISC1,12,360,0,0,NS-COMPUTE-S-BINARY,end,3,kernel.cpp
0,1,2,NCRISC,13,400,0,0,NS-READER-READ-AND-WAIT,begin,4,reader.cpp
0,1,2,NCRISC,13,500,0,0,NS-READER-READ-AND-WAIT,end,4,reader.cpp
0,1,2,BRISC,14,600,0,0,NS-WRITER-WRITES,begin,5,writer.cpp
0,1,2,BRISC,14,750,0,0,NS-WRITER-WRITES,end,5,writer.cpp
""",
        encoding="utf-8",
    )

    record = parse_device_profile_csv(path)

    assert {zone["risc"] for zone in record["zones"]} == {"TRISC0", "TRISC1", "NCRISC", "BRISC"}
    real = next(zone for zone in record["zones"] if zone["zone"] == "NS-COMPUTE-COMPLEX-REAL")
    assert real["cycles"] == 60
    assert real["percent_of_named_risc_cycles"] == 60.0
    assert record["unmatched_begins"] == 0
    assert record["unmatched_ends"] == 0


def test_profile_csv_parser_reports_missing_header(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("not a profiler file\n", encoding="utf-8")

    try:
        parse_device_profile_csv(path)
    except ValueError as error:
        assert "header not found" in str(error)
    else:
        raise AssertionError("expected missing-header validation")
