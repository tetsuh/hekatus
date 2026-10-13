"""Board-free regression tests for left-held AICLK interval facts."""

from __future__ import annotations

from pathlib import Path

import pytest

from enodia.tt.bench.power_trace_snapshot import (
    PowerTraceCsvOptions,
    read_power_trace_csv,
)

CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
RUN_START = "2026-01-01T00:00:02+00:00"
RUN_END = "2026-01-01T00:00:04+00:00"
LOAD_START = "2026-01-01T00:00:01+00:00"
LOAD_END = "2026-01-01T00:00:03+00:00"


def test_snapshot_extrema_use_positive_left_held_segments(tmp_path: Path):
    trace_path = tmp_path / "power.csv"
    trace_path.write_text(
        CSV_HEADER
        + "2026-01-01T00:00:00+00:00,40,800,50\n"
        + "2026-01-01T00:00:02+00:00,70,1350,51\n"
        + "2026-01-01T00:00:03+00:00,40,700,50\n"
        + "2026-01-01T00:00:04+00:00,60,1200,50\n",
        encoding="utf-8",
    )

    trace = read_power_trace_csv(
        trace_path,
        options=PowerTraceCsvOptions(
            run_start=RUN_START,
            run_end=RUN_END,
            load_interval_start=LOAD_START,
            load_interval_end=LOAD_END,
            immutable_snapshot=True,
        ),
    )

    assert trace["aiclk_mhz_in_run_min"] == 800.0
    assert trace["aiclk_mhz_in_run_max"] == 1350.0
    assert trace["aiclk_mhz"] == 1350.0
    assert trace["aiclk_mhz_load_interval_weighted"] == pytest.approx(1075.0)
    assert trace["in_run_valid_row_count"] == 3
    assert [row["aiclk_mhz"] for row in trace["in_run_samples"]] == [
        "1350",
        "700",
        "1200",
    ]
