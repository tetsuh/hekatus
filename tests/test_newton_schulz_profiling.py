from pathlib import Path

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
