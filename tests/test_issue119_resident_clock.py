"""Board-free checks for the table runner's host-program AICLK window."""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from enodia.tt.bench import run_resident
from enodia.tt.bench.resident_harness import AUDITED_CLOCK_SOURCE_IMAGE
from enodia.tt.bench.resident_record import validate_resident_record
from enodia.tt.bench.resident_tables import build_table_evidence, frame_checksum
from enodia.tt.bench.run_binding import RunBindingError


@pytest.mark.parametrize("strategy", ("copy_on_demand", "prefetch"))
@pytest.mark.parametrize("watcher", (False, True))
@pytest.mark.parametrize("coverage", ("complete", "missing_final", "missing_start"))
def test_table_runner_weights_aiclk_over_host_program_span(
    tmp_path, monkeypatch, strategy, watcher, coverage,
):
    """The program window, not device setup/teardown, defines weighted AICLK."""
    launch = "2026-01-01T00:00:02+00:00"
    completion = "2026-01-01T00:00:03+00:00"
    run_id = "issue119-board-free"
    trace = tmp_path / f"power-{run_id}.csv"
    initial_rows = (
        "2026-01-01T00:00:00+00:00,75,800,60\n"
        "2026-01-01T00:00:02+00:00,75,1000,60\n"
    ) if coverage != "missing_start" else ""
    trace.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        + initial_rows + "2026-01-01T00:00:02.250000+00:00,75,1200,60\n"
    )
    environment = {
        "board": {"serial": "board-free-fixture", "board_type": "synthetic"},
        "firmware": {"fw_bundle_version": "fixture"},
        "kmd_version": "fixture",
        "image": AUDITED_CLOCK_SOURCE_IMAGE,
        "image_pinned": True,
        "harness_commit": "0123456789abcdef",
        "harness_dirty": False,
        "harness_pushed": True,
        "tt_env_active_release": "0.75.0",
        "run_id": run_id,
        "telemetry_sampler": {
            "mode": "default", "interval_seconds": 2.0,
            "power_trace": "required", "timing_evidence": "available",
        },
    }
    env_path = tmp_path / "environment.json"
    env_path.write_text(json.dumps(environment))
    output = tmp_path / "result.json"
    monkeypatch.setattr(run_resident, "_runner_output_dir", lambda: tmp_path)
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", run_id)
    monkeypatch.delenv("HEKATUS_TT_RESULT_PATH", raising=False)
    if watcher:
        monkeypatch.setenv("TT_METAL_WATCHER", "1")
    else:
        monkeypatch.delenv("TT_METAL_WATCHER", raising=False)

    # Never import TTNN or open a real device: every device entry point is fake.
    monkeypatch.setitem(sys.modules, "ttnn", SimpleNamespace(
        open_device=lambda **_: object(), close_device=lambda _: None,
    ))

    def fake_run(_ttnn, _device, config, *, watcher):
        timestamps = [1000 + i * config.frame_interval_ticks for i in range(3)]
        rows = [[frame_checksum(config, i, (0, 1)), 100, i, 0, 1, 0] for i in range(3)]
        summary = [0] * 1024
        summary[19] = 3
        summary[23] = 1
        return {
            "timestamps": timestamps, "producer_full_count": 0, "consumer_empty_count": 0,
            "kernel_error_flag": 0, "frames_attempted": 3, "frames_produced": 3,
            "frames_dropped": 0, "frames_aborted": 0, "frames_consumed": 3,
            "startup_ticks": 0, "startup_ticks_valid": False,
            "work_min_ticks": None, "work_max_ticks": None, "work_ticks_valid": False,
            "failure_check": {"code": 0, "name": "none", "source": "none"},
            "program_launch_ns": 2_000_000_000, "program_completion_ns": 3_000_000_000,
            "program_launch_utc": launch, "program_completion_utc": completion,
            "table_swap": build_table_evidence(config, rows, timestamps, summary),
        }

    monkeypatch.setattr(run_resident, "_run_device", fake_run)
    waits = []

    def final_sample(path, edge, *, sampler_interval_seconds):
        waits.append((path, edge, sampler_interval_seconds))
        assert path == trace
        assert edge == completion
        if coverage == "missing_final":
            raise RunBindingError("power trace has no usable sample at or after program completion UTC")
        # Stand in for the existing sampler publishing the bracketing row.
        with path.open("a") as handle:
            handle.write("2026-01-01T00:00:03+00:00,75,1350,60\n")

    monkeypatch.setattr(run_resident, "_wait_for_power_trace_sample", final_sample)
    real_datetime = datetime.datetime
    times = iter((0, 4, 4))

    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 0, 0, next(times), tzinfo=datetime.UTC)

    monkeypatch.setattr(run_resident.datetime, "datetime", FixedDatetime)
    args = [
        "--out", str(output), "--env-json", str(env_path), "--power-trace", trace.name,
        "--frame-count", "3", "--profile-table", "0:1:0", "--initial-identity", "0:1",
        "--table-bytes", "4096", "--swap-strategy", strategy,
    ]
    if watcher:
        args.append("--watcher")
    assert run_resident.main(args) == (0 if coverage == "complete" else 2)
    record = json.loads(output.read_text())
    assert waits == [(trace, completion, 2.0)]
    if coverage != "complete":
        assert record["status"] == "rejected"
        assert record["timing_evidence"] is False
        if coverage == "missing_final":
            assert "no usable sample" in record["rejection_reason"]
            assert not list(tmp_path.glob("resident-power-snapshot-*.csv"))
        else:
            assert "not bounded by the trace" in record["rejection_reason"]
        return
    assert record["aiclk_mhz_load_interval_weighted"] == pytest.approx(1150.0)
    assert record["clock"]["aiclk_mhz_load_interval_weighted"] == pytest.approx(1150.0)
    assert record["clock"]["aiclk_mhz"] == 1350  # Preserve the public maximum.
    provenance = record["power_clock_provenance"]
    assert provenance["load_interval_start"] == launch
    assert provenance["load_interval_end"] == completion
    assert provenance["load_interval_source"] == "issue118_program_host_interval"
    assert provenance["aiclk_mhz_load_interval_weighted_source"] == "load_interval_trace_samples"
    assert provenance["aiclk_mhz_load_interval_weighted_seconds"] == 1.0
    assert record["program_host_interval"]["elapsed_ns"] == 1_000_000_000
    assert record["timing_evidence"] is (not watcher)
    snapshot = tmp_path / record["power_trace"]
    frozen_bytes = snapshot.read_bytes()
    assert hashlib.sha256(frozen_bytes).hexdigest() == record["power_trace_sha256"]
    assert validate_resident_record(record, power_trace_path=snapshot)["valid"]
    with trace.open("a") as handle:
        handle.write("2026-01-01T00:00:05+00:00,75,800,60\n")
    assert snapshot.read_bytes() == frozen_bytes
    assert json.loads(output.read_text()) == record
