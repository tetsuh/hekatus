"""Board-free regression tests for Issue #124 power-trace option shims."""

from __future__ import annotations

from pathlib import Path

import pytest

from enodia.tt.bench.power_trace_snapshot import (
    read_power_trace_csv,
    read_power_trace_snapshot,
)

CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
RUN_START = "2026-01-01T00:00:00+00:00"
RUN_END = "2026-01-01T00:00:01+00:00"


def _empty_trace(_path: Path) -> str:
    return CSV_HEADER


@pytest.mark.parametrize("missing", ("run_start", "run_end"))
def test_csv_reader_rejects_omitted_required_run_bound(
    tmp_path: Path, missing: str
) -> None:
    options = {"run_start": RUN_START, "run_end": RUN_END}
    options.pop(missing)

    with pytest.raises(
        TypeError,
        match=rf"read_power_trace_csv\(\) missing required keyword argument '{missing}'",
    ):
        read_power_trace_csv(tmp_path / "power.csv", **options)


def test_csv_reader_accepts_explicit_none_run_bounds(tmp_path: Path) -> None:
    trace = read_power_trace_csv(
        tmp_path / "power.csv",
        run_start=None,
        run_end=None,
        read_text_fn=_empty_trace,
    )

    assert trace["readable"] is True
    assert trace["run_start"] is None
    assert trace["run_end"] is None
    assert trace["error"] == {"code": "power_trace_no_usable_rows"}


@pytest.mark.parametrize("missing", ("run_id", "run_start", "run_end"))
def test_snapshot_reader_rejects_omitted_required_run_field(
    tmp_path: Path, missing: str
) -> None:
    options = {
        "run_id": "run-124",
        "run_start": RUN_START,
        "run_end": RUN_END,
    }
    options.pop(missing)

    with pytest.raises(
        TypeError,
        match=rf"read_power_trace_snapshot\(\) missing required keyword argument '{missing}'",
    ):
        read_power_trace_snapshot(tmp_path / "snapshot.csv", **options)


def test_snapshot_reader_accepts_explicit_none_run_bounds(tmp_path: Path) -> None:
    trace = read_power_trace_snapshot(
        tmp_path / "snapshot-run-124-1.csv",
        run_id="run-124",
        run_start=None,
        run_end=None,
        snapshot_prefix="snapshot",
        read_text_fn=_empty_trace,
    )

    assert trace["readable"] is True
    assert trace["run_start"] is None
    assert trace["run_end"] is None
    assert trace["error"] == {"code": "power_trace_no_usable_rows"}
