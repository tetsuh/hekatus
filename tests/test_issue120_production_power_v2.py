"""Post-child launcher readback, versioned power provenance and B1 replay."""
from __future__ import annotations

import argparse
import errno
import hashlib
import multiprocessing
import os
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest
from issue120_production_replay import (
    ARCHIVE,
    HARNESS_COMMIT,
    IDENTITY,
    SESSION_ID,
    publish_fixture,
)

from enodia.tt.bench import power_trace_snapshot as power
from enodia.tt.bench.issue120_production import plan, recording, runner, sessions

SCOPE = "issue120_v2_enqueue_to_l1_terminal_observed_upper_bound"
CONTRACT = "issue120-production-v2"


def test_real_launcher_reopens_child_artifacts_across_permission_seam(tmp_path, monkeypatch):
    """CI seam: publisher owns files, reader has a different UID, not root."""
    session = tmp_path / "session"
    case = session / "cases" / plan.RUNS[0].identity
    monkeypatch.setattr(sessions, "session_path", lambda _: session)
    monkeypatch.setattr(runner, "_session_spec", lambda *a: (plan.RUNS[0], {"harness_commit": HARNESS_COMMIT}, "hash"))
    monkeypatch.setattr(runner, "_clean_pushed", lambda: HARNESS_COMMIT)
    monkeypatch.setattr(runner, "wrapper_command", lambda *a: (["cpu-fixture-child"], {}))
    original_open = Path.open
    reopens = []

    def host_open(path, mode="r", *args, **kwargs):
        # Model another UID's POSIX read access, then perform actual file I/O.
        if str(mode).startswith("r") and path.is_file() and path.is_relative_to(case):
            if not path.stat().st_mode & stat.S_IROTH:
                raise PermissionError(errno.EACCES, "cross-UID artifact unreadable", str(path))
            reopens.append(path.name)
        return original_open(path, mode, *args, **kwargs)

    def child(command, **kwargs):
        assert command == ["cpu-fixture-child"]
        process = multiprocessing.get_context("spawn").Process(target=publish_fixture, args=(case,),
                                                            kwargs={"creation_umask": 0o077})
        process.start()
        process.join(30)
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0
        monkeypatch.setattr(Path, "open", host_open)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", child)
    args = argparse.Namespace(owner_approved=True, session_id=SESSION_ID,
                              launch=plan.RUNS[0].identity, production_approved=False)
    assert runner._launch(args) == 0
    assert any(name.startswith("power-trace-snapshot-") for name in reopens)
    assert f"results-{IDENTITY}.json" in reopens


@pytest.mark.parametrize("settings", [
    {"mode": "wb"}, {"publish_mode": 0o666}, {"publish_mode": 0o600}, {"publish_mode": True},
])
def test_public_output_policy_rejects_unsafe_settings_before_open(tmp_path, settings):
    path = tmp_path / "existing"
    path.write_bytes(b"retain original")
    with pytest.raises(ValueError), recording._public_output(path, **settings):
        pytest.fail("invalid publication settings must not open an artifact")
    assert path.read_bytes() == b"retain original"


def test_replay_publisher_restores_parent_umask(tmp_path):
    prior = os.umask(0o027)
    try:
        publish_fixture(tmp_path, creation_umask=0o077)
        observed = os.umask(0o027)
        assert observed == 0o027
    finally:
        os.umask(prior)


def test_b1_archive_is_still_accepted_without_reinterpretation(tmp_path):
    before = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(tmp_path, filter="data")
    path = tmp_path / f"results-{IDENTITY}.json"
    original = path.read_bytes()
    record = recording.load_validated_record(path)
    assert record["schema"] == "issue120-production-v1"
    assert record["power_trace_snapshot"]["aiclk_mhz_load_interval_weighted"] is None
    assert record["power_trace_snapshot"]["errors"] == [{"code": "load_interval_provenance_invalid"}]
    assert record["timing_evidence"] is False
    assert path.read_bytes() == original
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == before


def test_new_record_has_explicit_v2_power_scope_and_valid_conversion(tmp_path):
    path = publish_fixture(tmp_path, watcher=False)
    record = recording.load_validated_record(path)
    assert record["schema"] == CONTRACT
    assert record["power_trace_snapshot"]["load_interval_source"] == SCOPE
    assert record["power_trace_snapshot"]["errors"] == []
    assert record["power_trace_snapshot"]["aiclk_mhz_load_interval_weighted"] == pytest.approx(1350)
    assert record["timing_evidence"] is True
    overhead = record["overhead"]
    total = overhead["native_retrieval_verify_close_publish_seconds"] + overhead["python_frame_analysis_seconds"]
    assert total == pytest.approx(overhead["per_frame_analysis_seconds"] * overhead["analysis_frame_count"])
    assert overhead["per_frame_term_is_amortized_not_linear_slope"] is True


@pytest.mark.parametrize("bad", [{"errors": [{"code": "load_interval_provenance_invalid"}]},
                                 {"aiclk_mhz_load_interval_weighted": None},
                                 {"aiclk_mhz_load_interval_weighted": float("nan")},
                                 {"aiclk_mhz_load_interval_weighted_seconds": 0},
                                 {"aiclk_mhz_load_interval_weighted_seconds": float("nan")},
                                 {"aiclk_mhz_load_interval_weighted": float("inf")},
                                 {"aiclk_mhz_load_interval_weighted": -1},
                                 {"coverage_complete": False},
                                 {"aiclk_mhz_load_interval_weighted_source": "unknown"},
                                 {"load_interval_source": "unknown"}])
def test_new_timing_cannot_ignore_power_provenance_errors(bad):
    record = {"schema": CONTRACT, "condition": "control", "resident_integrity_passed": True,
              "warning_check": {"passed": True}, "watcher": {"requested": False, "scan_passed": True}}
    trace = {"coverage_complete": True, "errors": [], "load_interval_source": SCOPE,
             "aiclk_mhz_load_interval_weighted": 1350,
             "aiclk_mhz_load_interval_weighted_seconds": 5,
             "aiclk_mhz_load_interval_weighted_source": "load_interval_trace_samples"}
    assert recording.eligibility(record, trace) == (True, True)
    trace.update(bad)
    assert recording.eligibility(record, trace) == (True, False)


def test_reader_only_accepts_new_label_under_new_contract(tmp_path):
    source = tmp_path / "power-unit.csv"
    source.write_text("timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
                      "2026-01-01T00:00:00+00:00,70,800,40\n"
                      "2026-01-01T00:00:02+00:00,70,1350,40\n"
                      "2026-01-01T00:00:04+00:00,70,1350,40\n")
    snap = power.atomic_power_trace_snapshot(source, tmp_path, run_id="unit", snapshot_number=1)
    options = {"run_id": "unit", "run_start": "2026-01-01T00:00:00+00:00",
               "run_end": "2026-01-01T00:00:04+00:00", "include_issue118_metadata": True,
               "load_interval_start": "2026-01-01T00:00:01+00:00",
               "load_interval_end": "2026-01-01T00:00:03+00:00",
               "load_interval_source": SCOPE, "require_explicit_load_interval": True}
    legacy = power.read_power_trace_snapshot(snap, options=power.PowerTraceSnapshotOptions(**options))
    assert legacy["aiclk_mhz_load_interval_weighted"] is None
    fresh = power.read_power_trace_snapshot(snap, options=power.PowerTraceSnapshotOptions(
        **options, load_interval_contract=CONTRACT))
    assert fresh["aiclk_mhz_load_interval_weighted"] == 1075
    assert fresh["errors"] == []
    options["load_interval_source"] = "unknown"
    rejected = power.read_power_trace_snapshot(snap, options=power.PowerTraceSnapshotOptions(
        **options, load_interval_contract=CONTRACT))
    assert rejected["aiclk_mhz_load_interval_weighted"] is None
    assert rejected["errors"] == [{"code": "load_interval_provenance_invalid"}]


@pytest.mark.parametrize("start,end", [("00:00:01", "00:00:01"), ("00:00:03", "00:00:01"),
                                        ("00:00:01", "00:00:05"), (None, "00:00:03")])
def test_v2_weighting_needs_a_positive_bounded_explicit_interval(tmp_path, start, end):
    source = tmp_path / "power-unit.csv"
    source.write_text("timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
                      "2026-01-01T00:00:00+00:00,70,1350,40\n"
                      "2026-01-01T00:00:04+00:00,70,1350,40\n")
    snap = power.atomic_power_trace_snapshot(source, tmp_path, run_id="unit", snapshot_number=1)
    def stamp(value):
        return "2026-01-01T" + value + "+00:00" if value else None
    trace = power.read_power_trace_snapshot(snap, options=power.PowerTraceSnapshotOptions(
        run_id="unit", run_start=stamp("00:00:00"), run_end=stamp("00:00:04"),
        include_issue118_metadata=True, load_interval_contract=CONTRACT,
        load_interval_source=SCOPE, require_explicit_load_interval=True,
        load_interval_start=stamp(start), load_interval_end=stamp(end)))
    assert trace["aiclk_mhz_load_interval_weighted"] is None


@pytest.mark.parametrize("contract,explicit", [("unknown", True), (CONTRACT, False)])
def test_unknown_or_implicit_versioned_contract_is_rejected(tmp_path, contract, explicit):
    with pytest.raises(ValueError):
        power.read_power_trace_csv(tmp_path / "absent.csv", options=power.PowerTraceCsvOptions(
            run_start=None, run_end=None, load_interval_contract=contract,
            require_explicit_load_interval=explicit))


def test_snapshot_mode_is_explicit_atomic_and_cannot_rewrite_legacy(tmp_path):
    source = tmp_path / "power-unit.csv"
    source.write_text("fixture bytes\n")
    old = power.atomic_power_trace_snapshot(source, tmp_path, run_id="unit", snapshot_number=1)
    assert stat.S_IMODE(old.stat().st_mode) == 0o600
    fresh = power.atomic_power_trace_snapshot(source, tmp_path, run_id="unit", snapshot_number=2,
                                              publish_mode=0o644)
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o644
    assert fresh.read_bytes() == old.read_bytes()
    with pytest.raises(FileExistsError):
        power.atomic_power_trace_snapshot(source, tmp_path, run_id="unit", snapshot_number=1,
                                          publish_mode=0o644)
    assert stat.S_IMODE(old.stat().st_mode) == 0o600
    for mode in (0o666, 0o777, True):
        with pytest.raises(ValueError):
            power.atomic_power_trace_snapshot(source, tmp_path, run_id="unit", snapshot_number=3,
                                              publish_mode=mode)


def test_v2_fixture_rejects_modified_evidence_and_watcher_timing(tmp_path):
    path = publish_fixture(tmp_path)
    record = recording.load_validated_record(path)
    assert record["timing_evidence"] is False
    frames = path.parent / record["artifacts"]["frames"]["file"]
    with frames.open("ab") as out:
        out.write(b"tamper")
    with pytest.raises(ValueError):
        recording.load_validated_record(path)
