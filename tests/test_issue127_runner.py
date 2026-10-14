"""Board-free contract tests for the Issue #127 ABBA/BADC runner."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from enodia.tt.bench.power_trace_snapshot import EXPLICIT_FINAL_SAMPLE
from tools import newton_schulz_issue127 as runner

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "docs/measurements/2026-10-12-host-newton-schulz-issue127-abba-plan.json"
CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"


def _environment(*, observed: list[int] | None = None) -> dict:
    return {
        "run_id": "run127-test",
        "image": "ghcr.io/tenstorrent/tt-metal@sha256:" + "a" * 64,
        "image_pinned": True,
        "image_digest": "sha256:" + "a" * 64,
        "kernel": "Linux 6.8.0-1-generic",
        "kmd_version": "2.11.0",
        "tt_env_active_release": "0.75.0",
        "harness_commit": "b" * 40,
        "harness_dirty": False,
        "board_info": {
            "board_type": "p150a",
            "board_id": "0000040331914124",
            "bus_id": "0000:09:00.0",
        },
        "firmwares": {"fw_bundle_version": "19.6.0.0"},
        "limits": {"asic_fmax": "1350"},
        "device_node": "/dev/tenstorrent/0",
        "telemetry_sampler": {
            "mode": "default",
            "interval_seconds": 2.0,
            "power_trace": "required",
            "timing_evidence": "available",
        },
        "aiclk_mhz_observed": [1350] if observed is None else observed,
        "hostname": "must-not-be-recorded",
        "worktree_path": "/home/user/private-worktree",
    }


def test_issue127_canonical_controls_and_placements_are_exactly_four_rows():
    rows = runner.ISSUE127_COMPARISON_ROWS
    runner.validate_comparison_rows()

    assert [row["name"] for row in rows] == [
        "bf16-r-L16",
        "fp32-r-L16",
        "bf16-r-L32-r-dram",
        "fp32-r-L32-r-dram",
    ]
    assert [(row["variant"], row["size"], row["r_memory"]) for row in rows] == [
        ("bf16", 16, "l1"),
        ("fp32-r", 16, "l1"),
        ("bf16", 32, "dram"),
        ("fp32-r", 32, "dram"),
    ]
    assert {
        key: rows[0][key]
        for key in (
            "batch",
            "iterations",
            "initial_value",
            "state_format",
            "destination_format",
            "fp32_dest_acc_en",
            "math_fidelity",
            "fuse_s",
            "input_memory",
            "matrix_block",
            "double_buffer",
            "dst_full_sync_en",
            "output_memory",
            "x0_memory",
        )
    } == {
        "batch": 8192,
        "iterations": 12,
        "initial_value": "I/||R||inf",
        "state_format": "BF16",
        "destination_format": "FP32",
        "fp32_dest_acc_en": True,
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "input_memory": "l1",
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "output_memory": "dram",
        "x0_memory": "l1",
    }


def test_issue127_orders_and_occurrence_balance_are_fixed():
    run_1 = runner.build_occurrence_plan(1)
    run_2 = runner.build_occurrence_plan(2)

    assert [row["row"] for row in run_1] == list(runner.RUN_1_ORDER)
    assert [row["row"] for row in run_2] == list(runner.RUN_2_ORDER)
    for plan in (run_1, run_2):
        assert [row["order_index"] for row in plan] == list(range(8))
        assert {row["row"] for row in plan} == set(runner.ISSUE127_ROW_NAMES)
        assert {name: sum(row["row"] == name for row in plan) for name in runner.ISSUE127_ROW_NAMES} == {
            name: 2 for name in runner.ISSUE127_ROW_NAMES
        }
        assert all(row["occurrence_id"] == f"{row['row']}#{row['occurrence']}" for row in plan)


def test_issue127_launch_counts_and_duration_guard_are_explicit():
    assert runner.launches_for_row("bf16-r-L16") == 6500
    assert runner.launches_for_row("fp32-r-L16") == 6500
    assert runner.launches_for_row("bf16-r-L32-r-dram") == 4000
    assert runner.launches_for_row("fp32-r-L32-r-dram") == 4000

    passing = runner.row_duration_guard(2.7, "bf16-r-L16")
    assert passing["passes"] is True
    assert passing["within_target"] is True
    assert runner.row_duration_guard(3.1, "bf16-r-L16")["hard_max_pass"] is False
    assert runner.row_duration_guard(2.5, "bf16-r-L16")["passes"] is True
    assert runner.row_duration_guard(2.3, "bf16-r-L16")["passes"] is False


def test_issue127_timed_bounds_and_adjusted_p50_formula():
    row = {
        "timed_start_utc": "2026-10-12T00:00:00+00:00",
        "timed_end_utc": "2026-10-12T00:00:02.700000+00:00",
        "timed_start_monotonic": 10.0,
        "timed_end_monotonic": 12.7,
        "timed_duration_seconds": 2.7,
        "size": 16,
    }
    runner.validate_row_bounds(row)
    assert runner.adjusted_p50_tflops(16.0, 1200.0) == pytest.approx(18.0)
    assert runner.adjusted_p50_tflops(16.0, None) is None

    with pytest.raises(ValueError, match="strictly increasing"):
        runner.validate_row_bounds({**row, "timed_end_utc": row["timed_start_utc"]})


def test_issue127_shared_left_held_row_facts_include_start_and_reject_unbounded_end():
    samples = [
        {"timestamp_utc": "2026-10-12T00:00:00+00:00", "power_w": "40", "aiclk_mhz": "800", "asic_temp_c": "50"},
        {"timestamp_utc": "2026-10-12T00:00:02+00:00", "power_w": "70", "aiclk_mhz": "1350", "asic_temp_c": "51"},
        {"timestamp_utc": "2026-10-12T00:00:03+00:00", "power_w": "40", "aiclk_mhz": "700", "asic_temp_c": "50"},
        {"timestamp_utc": "2026-10-12T00:00:04+00:00", "power_w": "60", "aiclk_mhz": "1200", "asic_temp_c": "50"},
    ]
    facts = runner.left_held_aiclk_interval_facts(
        samples,
        start="2026-10-12T00:00:01+00:00",
        end="2026-10-12T00:00:04+00:00",
    )
    assert facts["available"] is True
    assert facts["first_held_aiclk_mhz"] == 800.0
    assert facts["aiclk_mhz_min"] == 700.0
    assert facts["aiclk_mhz_max"] == 1350.0
    assert facts["time_weighted_aiclk_mhz"] == pytest.approx((800 + 1350 + 700) / 3)
    assert facts["semantics"] == "left_held_positive_segments"

    unavailable = runner.left_held_aiclk_interval_facts(
        samples,
        start="2026-10-12T00:00:03+00:00",
        end="2026-10-12T00:00:05+00:00",
    )
    assert unavailable["available"] is False
    assert unavailable["time_weighted_aiclk_mhz"] is None
    assert unavailable["aiclk_mhz_min"] is None
    assert unavailable["aiclk_mhz_max"] is None
    assert unavailable["first_held_aiclk_mhz"] is None
    assert unavailable["last_held_aiclk_mhz"] is None
    assert unavailable["segments"] == []


def test_issue127_final_trace_facts_record_source_and_adjusted_p50():
    row = {
        **runner.build_occurrence_plan(1)[0],
        "timed_start_utc": "2026-10-12T00:00:01+00:00",
        "timed_end_utc": "2026-10-12T00:00:03+00:00",
        "timed_start_monotonic": 1.0,
        "timed_end_monotonic": 3.0,
        "timed_duration_seconds": 2.0,
        "raw_p50_tflops": 16.0,
    }
    trace = {
        "file": "issue127-abba-power-snapshot-run127-test-1.csv",
        "source_file": "power-run127-test.csv",
        "sha256": "c" * 64,
        "sha256_definition": "SHA-256 of the complete immutable per-run power CSV snapshot bytes",
        "coverage_definition": EXPLICIT_FINAL_SAMPLE,
        "coverage_complete": True,
        "immutable_snapshot": True,
        "samples": [
            {"timestamp_utc": "2026-10-12T00:00:00+00:00", "power_w": "40", "aiclk_mhz": "1350", "asic_temp_c": "50"},
            {"timestamp_utc": "2026-10-12T00:00:02+00:00", "power_w": "70", "aiclk_mhz": "1200", "asic_temp_c": "51"},
            {"timestamp_utc": "2026-10-12T00:00:04+00:00", "power_w": "60", "aiclk_mhz": "1200", "asic_temp_c": "50"},
        ],
    }
    runner.apply_final_trace_facts([row], power_trace=trace)

    assert row["aiclk_interval"]["trace_sha256"] == "c" * 64
    assert row["row_time_weighted_aiclk_mhz"] == pytest.approx(1275.0)
    assert row["adjusted_p50_tflops"] == pytest.approx(16.0 * 1350.0 / 1275.0)
    assert row["aiclk_interval"]["requested_sampler_interval_seconds"] == 2.0


def test_issue127_final_trace_waits_for_sample_after_run_end(tmp_path: Path):
    run_id = "run127-rollover"
    source = tmp_path / f"power-{run_id}.csv"
    header = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
    before_end = (
        header
        + "2026-10-12T00:00:00+00:00,40,1200,50\n"
        + "2026-10-12T00:00:03+00:00,50,1200,51\n"
    )
    after_end = before_end + "2026-10-12T00:00:05+00:00,60,1350,52\n"
    source.write_text(before_end, encoding="utf-8")
    def update_sampler_source(_interval: float) -> None:
        source.write_text(after_end, encoding="utf-8")

    trace = runner._capture_final_power_trace(
        tmp_path,
        run_id=run_id,
        run_start="2026-10-12T00:00:00+00:00",
        run_end="2026-10-12T00:00:04+00:00",
        power_path=source,
        monotonic_fn=lambda: 0.0,
        sleep_fn=update_sampler_source,
    )

    assert trace["coverage_definition"] == EXPLICIT_FINAL_SAMPLE
    assert trace["coverage_complete"] is True
    assert trace["last_timestamp"] == "2026-10-12T00:00:05+00:00"
    assert trace["file"].endswith("-2.csv")


def test_issue127_warmup_800mhz_hold_gate_is_verified_from_row_facts():
    warmup = {"end_utc": "2026-10-12T00:00:01+00:00"}
    first_row = {
        "timed_start_utc": "2026-10-12T00:00:02+00:00",
        "aiclk_interval": {
            "first_held_aiclk_mhz": 1350.0,
            "coverage_complete": True,
            "trace_coverage_complete": True,
        },
    }
    gate = runner.verify_warmup_hold_gate(
        warmup,
        first_row,
        power_trace={"file": "snapshot.csv", "sha256": "d" * 64},
    )
    assert gate["verified"] is True
    assert warmup["hold_gate_verified"] is True


def test_issue127_cooldown_metadata_requires_wait_and_keeps_pre_cooldown_aiclk_provenance():
    environment = runner.sanitize_environment(_environment(observed=[800]))
    runner.validate_harness_environment(environment, run_id="run127-test")
    metadata = runner.build_cooldown_metadata(
        run_number=2,
        wait_seconds=30.0,
        expected_aiclk_mhz=1350.0,
        environment=environment,
        start_utc="2026-10-12T00:00:00+00:00",
        end_utc="2026-10-12T00:00:30+00:00",
        start_monotonic=10.0,
        end_monotonic=40.0,
        wait_performed=True,
    )
    assert metadata["environment_verified"] is True
    assert metadata["environment_observed_aiclk_mhz"] == [800]
    assert metadata["verified"] is False
    with pytest.raises(ValueError, match="power-trace proof"):
        runner.validate_cooldown_metadata(metadata, environment=environment, require_power_trace=True)
    with pytest.raises(ValueError, match="cooldown_wait_seconds"):
        runner.validate_cooldown_metadata(
            {**metadata, "cooldown_wait_seconds": None}, environment=environment
        )
    with pytest.raises(ValueError, match="expected_aiclk_mhz must be 1350"):
        runner.validate_cooldown_metadata(
            {**metadata, "expected_aiclk_mhz": 1200}, environment=environment
        )


def _cooldown_trace(*, terminal_timestamp: str = "2026-10-12T00:00:04+00:00") -> dict:
    return {
        "file": "issue127-abba-power-snapshot-run127-test-1.csv",
        "source_file": "power-run127-test.csv",
        "sha256": "e" * 64,
        "sha256_definition": "SHA-256 of the complete immutable per-run power CSV snapshot bytes",
        "coverage_definition": EXPLICIT_FINAL_SAMPLE,
        "coverage_complete": True,
        "immutable_snapshot": True,
        "samples": [
            {"timestamp_utc": "2026-10-12T00:00:00+00:00", "power_w": "40", "aiclk_mhz": "800", "asic_temp_c": "50"},
            {"timestamp_utc": "2026-10-12T00:00:01+00:00", "power_w": "50", "aiclk_mhz": "1000", "asic_temp_c": "51"},
            {"timestamp_utc": "2026-10-12T00:00:02+00:00", "power_w": "60", "aiclk_mhz": "1350", "asic_temp_c": "52"},
            {"timestamp_utc": terminal_timestamp, "power_w": "60", "aiclk_mhz": "1350", "asic_temp_c": "52"},
        ],
    }


def test_issue127_cooldown_proof_uses_final_trace_terminal_held_segment():
    environment = runner.sanitize_environment(_environment(observed=[800]))
    runner.validate_harness_environment(environment, run_id="run127-test")
    metadata = runner.build_cooldown_metadata(
        run_number=2,
        wait_seconds=3.0,
        expected_aiclk_mhz=1350.0,
        environment=environment,
        start_utc="2026-10-12T00:00:00.500000+00:00",
        end_utc="2026-10-12T00:00:03.500000+00:00",
        wait_performed=True,
    )
    verified = runner.verify_cooldown_power_trace(
        metadata, power_trace=_cooldown_trace()
    )
    assert verified["power_trace_verified"] is True
    assert verified["verified"] is True
    assert verified["power_trace_facts"]["first_held_aiclk_mhz"] == 800.0
    assert verified["power_trace_facts"]["terminal_held_aiclk_mhz"] == 1350.0

    unbounded = runner.verify_cooldown_power_trace(
        metadata,
        power_trace=_cooldown_trace(terminal_timestamp="2026-10-12T00:00:03+00:00"),
    )
    assert unbounded["power_trace_verified"] is False
    assert unbounded["power_trace_facts"]["coverage_complete"] is False
    assert unbounded["verified"] is False


def test_issue127_close_failure_cannot_produce_pass_record():
    environment = runner.sanitize_environment(_environment())
    rows = []
    for row in runner.build_occurrence_plan(1):
        measured = dict(row)
        measured.update(
            {
                "status": "ok",
                "raw_p50_tflops": 16.0,
                "adjusted_p50_tflops": 16.0,
                "row_time_weighted_aiclk_mhz": 1350.0,
                "timed_start_utc": "2026-10-12T00:00:02+00:00",
                "aiclk_interval": {
                    "coverage_complete": True,
                    "trace_coverage_complete": True,
                    "first_held_aiclk_mhz": 1350.0,
                },
            }
        )
        rows.append(measured)
    record = runner.build_record(
        run={
            "status": "pass",
            "comparison_rows": rows,
            "warmup": {"end_utc": "2026-10-12T00:00:01+00:00"},
            "run_under_60_seconds": True,
            "run_duration_seconds": 20.0,
            "run_start_monotonic": 0.0,
            "run_end_monotonic": 20.0,
            "input_generation": {},
            "device_close_failed": True,
        },
        run_number=1,
        run_id="run127-test",
        environment=environment,
        cooldown=runner.build_cooldown_metadata(
            run_number=1,
            wait_seconds=None,
            expected_aiclk_mhz=None,
            environment=environment,
        ),
        run_start_utc="2026-10-12T00:00:00+00:00",
        run_end_utc="2026-10-12T00:01:00+00:00",
        power_trace={
            "file": "snapshot.csv",
            "source_file": "power-run127-test.csv",
            "sha256": "f" * 64,
            "sha256_definition": "SHA-256 of the complete immutable per-run power CSV snapshot bytes",
            "coverage_definition": EXPLICIT_FINAL_SAMPLE,
            "coverage_complete": True,
            "immutable_snapshot": True,
            "timestamp_gap_facts": {"observed_timestamp_gap_seconds": [2.24]},
        },
    )
    assert record["status"] == "failed"
    assert record["device_close_failed"] is True


def test_issue127_environment_is_complete_but_does_not_retain_host_identity_or_worktree():
    environment = runner.sanitize_environment(_environment())
    runner.validate_harness_environment(environment, run_id="run127-test")

    assert environment["board"]["serial"] == "0000040331914124"
    assert environment["firmware"]["fw_bundle_version"] == "19.6.0.0"
    assert environment["kernel_driver_version"] == "2.11.0"
    assert environment["image_digest"].startswith("sha256:")
    assert "hostname" not in environment
    assert "worktree_path" not in environment
    assert environment["device_node"] == "/dev/tenstorrent/0"
    with pytest.raises(ValueError, match="absolute path"):
        runner._serialize_record({"bad": "/home/user/private-worktree"})


def test_issue127_sampler_decision_is_default_and_not_shortened():
    runner.validate_sampler_decision(runner.SAMPLER_DECISION, trace_timestamp_gaps=[2.24])
    assert runner.SAMPLER_DECISION["requested_interval_seconds"] == 2.0
    assert runner.SAMPLER_DECISION["shortened_interval"] is False
    with pytest.raises(ValueError, match="2.0"):
        runner.validate_sampler_decision(
            {**runner.SAMPLER_DECISION, "requested_interval_seconds": 0.5},
            trace_timestamp_gaps=[0.5],
        )


def test_issue127_prediction_plan_pins_orders_sampler_and_thresholds():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["runner"] == runner.ISSUE127_RUNNER
    assert tuple(plan["run_orders"]["run_1"]) == runner.RUN_1_ORDER
    assert tuple(plan["run_orders"]["run_2"]) == runner.RUN_2_ORDER
    assert plan["sampler_decision"]["requested_interval_seconds"] == 2.0
    assert plan["sampler_decision"]["shortened_interval"] is False
    assert plan["timing_protocol"]["startup_warmup"]["launches"] == 4500
    assert plan["timing_protocol"]["row_duration_target_seconds"] == [2.4, 2.8]
    assert plan["predicted_aiclk_ranges_mhz"]["basis"].endswith("predictions, not measured results.")
    for run_number, key in ((1, "run_1"), (2, "run_2")):
        planned = runner.build_occurrence_plan(run_number)
        expected = runner.PREDICTED_AICLK_RANGES_MHZ[run_number]
        assert [item["range_mhz"] for item in plan["predicted_aiclk_ranges_mhz"][key]] == [
            list(value) for value in expected
        ]
        assert [row["predicted_aiclk_range_mhz"] for row in planned] == [
            list(value) for value in expected
        ]
    assert plan["decision_criteria"]["r_format_hypothesis"]["thresholds_percent"] == {
        "L16_mean_at_most": -0.847,
        "L32_DRAM_mean_at_most": -2.341,
    }
    assert plan["decision_criteria"]["adjusted_p50_is_primary"] is False


def _fake_wrapper_tools(tmp_path: Path) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text(
        """#!/bin/sh
set -eu
case "${2:-}" in
  capture-env)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\n' '{"fake": true}' > "$out"
    ;;
  sample)
    trap 'exit 0' TERM INT
    while :; do sleep 1; done
    ;;
esac
""",
        encoding="utf-8",
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    (bindir / "docker").write_text(
        """#!/bin/sh
set -eu
printf '%s\n' "$@" > "${DOCKER_ARGS:?}"
exit 0
""",
        encoding="utf-8",
    )
    (bindir / "docker").chmod(stat.S_IRWXU)
    (bindir / "timeout").write_text(
        """#!/bin/sh
set -eu
printf '%s\n' "$@" > "${TIMEOUT_LOG:?}"
shift 3
exec "$@"
""",
        encoding="utf-8",
    )
    (bindir / "timeout").chmod(stat.S_IRWXU)
    return bindir


def test_issue127_wrapper_runs_cooldown_before_docker_timeout_window(tmp_path: Path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    events = tmp_path / "events"
    (bindir / "python3").write_text(
        """#!/bin/sh
set -eu
case "${2:-}" in
  capture-env)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\\n' '{"fake": true}' > "$out"
    ;;
  sample)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\\n' 'timestamp_utc,power_w,aiclk_mhz,asic_temp_c' > "$out"
    printf '%s\\n' '2026-10-12T00:00:00+00:00,40,1350,50' >> "$out"
    trap 'exit 0' TERM INT
    while :; do :; done
    ;;
esac
""",
        encoding="utf-8",
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    (bindir / "sleep").write_text(
        f"""#!/bin/sh
printf 'sleep:%s\\n' \"$*\" >> \"{events}\"
exit 0
""",
        encoding="utf-8",
    )
    (bindir / "sleep").chmod(stat.S_IRWXU)
    (bindir / "docker").write_text(
        f"""#!/bin/sh
set -eu
if [ "${{1:-}}" = kill ]; then exit 0; fi
printf 'docker\\n' >> "{events}"
printf '%s\\n' "$@" > "${{DOCKER_ARGS:?}}"
exit 0
""",
        encoding="utf-8",
    )
    (bindir / "docker").chmod(stat.S_IRWXU)
    (bindir / "timeout").write_text(
        """#!/bin/sh
set -eu
printf '%s\\n' "$@" > "${TIMEOUT_LOG:?}"
shift 3
exec "$@"
""",
        encoding="utf-8",
    )
    (bindir / "timeout").chmod(stat.S_IRWXU)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "enodia/tt/bench/run_in_container.sh", copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]
    docker_args = tmp_path / "docker-args"
    timeout_log = tmp_path / "timeout-args"
    completed = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "output"),
            "--",
            "--run-number",
            "2",
            "--cooldown-wait-seconds",
            "0.1",
            "--expected-aiclk-mhz",
            "1350",
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    events_seen = events.read_text(encoding="utf-8").splitlines()
    cooldown_indices = [
        index for index, event in enumerate(events_seen) if event == "sleep:-- 0.1"
    ]
    docker_index = events_seen.index("docker")
    assert len(cooldown_indices) == 1
    assert cooldown_indices[0] < docker_index
    docker = docker_args.read_text(encoding="utf-8").splitlines()
    assert "--cooldown-start-utc" in docker
    assert "--cooldown-end-utc" in docker
    assert timeout_log.read_text(encoding="utf-8").splitlines()[2] == "60s"
    assert "sleep(args.cooldown_wait_seconds)" not in Path(runner.__file__).read_text(
        encoding="utf-8"
    )

    rejected = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "output-large"),
            "--",
            "--run-number",
            "2",
            "--cooldown-wait-seconds",
            "1000.0",
            "--expected-aiclk-mhz",
            "1350",
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert rejected.returncode == 2
    assert "less than 1000" in rejected.stderr
    assert events.read_text(encoding="utf-8").splitlines() == events_seen


def test_issue127_wrapper_selects_generic_custom_runner_with_bound_artifacts(tmp_path: Path):
    bindir = _fake_wrapper_tools(tmp_path)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "enodia/tt/bench/run_in_container.sh", copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]
    docker_args = tmp_path / "docker-args"
    timeout_log = tmp_path / "timeout-args"
    completed = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "output"),
            "--",
            "--run-number",
            "1",
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    docker = docker_args.read_text(encoding="utf-8").splitlines()
    assert "tools/newton_schulz_issue127.py" in docker
    assert docker[docker.index("--entrypoint") + 1] == "python3"
    assert "--env-json" in docker
    assert "--power-trace" in docker
    assert timeout_log.read_text(encoding="utf-8").splitlines()[2] == "60s"

    docker_before_sampler_off = docker_args.read_text(encoding="utf-8")
    sampler_off = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "output-sampler-off"),
            "--",
            "--run-number",
            "1",
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
            "HEKATUS_TT_TELEMETRY_MODE": "off",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert sampler_off.returncode == 2
    assert "requires telemetry sampler" in sampler_off.stderr
    assert docker_args.read_text(encoding="utf-8") == docker_before_sampler_off
