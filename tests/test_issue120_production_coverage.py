"""Co-developed integration cases for widened causal binary evidence."""
from __future__ import annotations

import copy

import pytest

from enodia.tt.bench.issue120_production import artifacts, coverage, protocol


def fixture():
    ps, cs = [0] * 1024, [0] * 1024
    ps[1], ps[3], cs[1] = 2, 2, 2
    cs[32:36] = [4, 4, 4, 0]
    cs[36:40] = [1, 2, 2, 3]
    cs[40], cs[42], cs[44] = 5, 8, 7
    cs[46:50] = [6, 9, 7, 11]
    cs[50], cs[52] = 29, 31
    cs[54:57] = [1, 1, 1]
    cs[57] = 10
    rows, accumulator = [], 0
    for number in range(2):
        row = [0] * 32
        end, admission, issued, seen, work_start = (16 + 10 * number, 11 + 10 * number,
                                                   9 + 10 * number, 13 + 10 * number,
                                                   14 + 10 * number)
        accumulator = protocol.resident_checksum(accumulator, number, 64)
        row[0], row[2], row[3], row[4], row[5] = end, accumulator, 2, number, admission
        left, right = 2 + 2 * number, 3 + 2 * number
        row[7:11] = [left, left + 1 + number, right, right + 2 + number]
        row[11], row[13], row[15], row[17], row[18] = issued, seen, 1, number + 2, 64
        row[19], row[21] = (7, 8) if number == 0 else (17, 18)
        row[23], row[25] = 12 + 10 * number, work_start
        rows.append(row)
    native = {"condition": "loaded", "frame_count": 2, "repeat_count": 10,
              "producer_stats": ps, "consumer_stats": cs, "checksum_verified_frames": 2}
    index = protocol.TransitionIndex([(1, 1), (3, 2), (5, 3), (7, 4)],
                                     frame_count=2, repeat_count=10)
    return rows, native, index


def test_every_frame_and_both_endpoints_are_recomputed():
    rows, native, index = fixture()
    result = coverage.analyze(rows, native, index)
    assert result["coverage_analysis"]["coverage"] == "covered"
    assert result["coverage_analysis"]["verified_frames"] == 2
    assert result["coverage_analysis"]["challenge_interval_bytes_lower_bound"] == 12288
    # Frame0's challenge preceded admission; never count its bytes in-window.
    assert result["coverage_analysis"]["first_through_last_bytes_lower_bound"] == 4096
    assert result["metrics"]["frame_intervals"]["N"] == 1
    assert result["resident_integrity_passed"]


@pytest.mark.parametrize("mutation", ["startup", "checksum", "gap", "torn", "clock",
                                     "tail", "timestamp", "duplicate", "map"])
def test_false_coverage_is_not_promoted(mutation):
    rows, native, index = fixture()
    assert coverage.analyze(rows, native, index)["coverage_analysis"]["coverage"] == "covered"
    if mutation == "startup":
        native["consumer_stats"][56] = 0
    elif mutation == "checksum":
        rows[0][2] ^= 1
    elif mutation == "gap":
        rows[1][9:11] = rows[1][7:9]
    elif mutation == "torn":
        rows[0][15] = 0
    elif mutation == "clock":
        rows[1][16] = 1
    elif mutation == "tail":
        native["consumer_stats"][48:50] = [6, 9]
    elif mutation == "timestamp":
        rows[1][11] = 15  # before previous completion16
    elif mutation == "duplicate":
        rows[1][4] = 0
    else:
        rows[0][10] += 1  # individually plausible but never emitted tuple
    # Check the intended seam before any downstream cascade can hide its cause.
    if mutation in {"checksum", "duplicate"}:
        result = coverage.analyze(rows, native, index)["coverage_analysis"]
        assert result["coverage"] == "not_evaluable"
        assert result["reason"] == "frame identity/checksum/work/clock differs"
    else:
        checks = {
            "startup": (coverage._startup, (native["consumer_stats"], index),
                        coverage.MissingProgress, "startup gate has no validated progress"),
            "gap": (coverage._frame, (rows[1], 16, 3, 10, index),
                    coverage.MissingProgress, "fresh CPU/DMA response not observed"),
            "torn": (coverage._frame, (rows[0], 7, 2, 10, index),
                     ValueError, "torn/invalid progress page"),
            "clock": (coverage._frame, (rows[1], 16, 3, 10, index),
                      ValueError, "ambiguous consumer clock"),
            "tail": (coverage._tail, (native["consumer_stats"], 26, 5, 2, index),
                     coverage.MissingProgress, "fresh CPU/DMA response not observed"),
            "timestamp": (coverage._frame, (rows[1], 16, 3, 10, index),
                          ValueError, "frame causal interval ordering differs"),
            "map": (coverage._frame, (rows[0], 7, 2, 10, index),
                    ValueError, "observed tuple differs from committed host association"),
        }
        check, args, exception, message = checks[mutation]
        with pytest.raises(exception, match=f"^{message}$"):
            check(*args)
        assert coverage.analyze(rows, native, index)["coverage_analysis"]["coverage"] != "covered"


def test_exhausted_fixed_work_is_honest_not_covered():
    rows, native, _ = fixture()
    index = protocol.TransitionIndex([(1, 1), (3, 2)], frame_count=2, repeat_count=10)
    rows[1][9:11] = [5, 7]  # fully valid, but still an old challenge
    native["consumer_stats"][46:50] = [6, 8, 7, 9]
    result = coverage.analyze(rows, native, index)
    assert result["coverage_analysis"]["coverage"] == "not_covered"
    assert result["resident_integrity_passed"]


def test_control_never_invents_progress():
    rows, native, _ = fixture()
    native["condition"] = "control"
    for row in rows:
        row[7:11] = [0, 0, 0, 0]
        row[15] = 0
    result = coverage.analyze(rows, native, None)
    assert result["coverage_analysis"]["coverage"] == "not_applicable"
    assert result["coverage_analysis"]["challenge_interval_bytes_lower_bound"] == 0
    assert result["resident_integrity_passed"]


def test_encoded_seed_has_no_positive_sequence_word_collisions():
    seed = protocol.payload(0, 0)
    for sequence, challenge in [(1, 0), (2, 1), (24439669, 500002)]:
        assert all(a != b for a, b in zip(seed, protocol.payload(sequence, challenge), strict=True))


def test_binary_evidence_is_complete_bound_and_nonredirected(tmp_path):
    rows, _, _ = fixture()
    path = tmp_path / "frames.bin"
    data = artifacts.HEADER.pack(b"IP120F1\0", 1, 32, 2, 2)
    data += b"".join(artifacts.FRAME.pack(*row) for row in rows)
    path.write_bytes(data)
    evidence = artifacts.read_evidence(tmp_path, "frames.bin", frames=2, emitted=10)
    assert list(evidence.rows()) == list(map(tuple, rows))
    assert evidence.metadata()["byte_count"] == 24 + 2 * 128
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError):
        artifacts.read_evidence(tmp_path, "frames.bin", frames=2, emitted=10)
    path.unlink()
    other = tmp_path / "other.bin"
    other.write_bytes(data)
    path.symlink_to(other)
    with pytest.raises(ValueError):
        artifacts.read_evidence(tmp_path, "frames.bin", frames=2, emitted=10)
    with pytest.raises(ValueError):
        artifacts.read_evidence(tmp_path, "../frames.bin", frames=2, emitted=10)


@pytest.mark.parametrize("watcher", [False, True])
def test_record_factory_and_persisted_artifact_recomputation(tmp_path, monkeypatch, watcher):
    import datetime as dt
    import json

    from enodia.tt.bench.issue120_production import plan, recording, runtime
    from enodia.tt.bench.issue120_results import NativeAttempt

    rows, native, index = fixture()
    identity = "production-unit-attempt"
    capture = tmp_path / ("native-" + identity)
    (capture / "result").mkdir(parents=True)
    (capture / "evidence").mkdir()
    base = int(dt.datetime(2026, 11, 14, tzinfo=dt.UTC).timestamp()) * 10**9
    native.update(schema="issue120-production-native-v1",
                  host_interval={"start_ns": 2*10**9, "end_ns": 6*10**9},
                  host_cpu_work={"start_ns": 2*10**9, "end_ns": 6*10**9,
                                 "operations": 10*32768, "reads": 10, "transfers": 10},
                  host_scheduling={"enqueue_begin_ns": 10**9, "enqueue_return_ns": 3*10**9,
                                   "load_join_completed_ns": 7*10**9,
                                   "completion_observer_joined_ns": 8*10**9,
                                   "finish_begin_ns": 9*10**9, "finish_end_ns": 9100000000,
                                   "drain_end_ns": 9200000000},
                  resident_host_interval={"start_ns": 10**9, "end_ns": 9100000000,
                                         "scope": "enqueue_to_post_join_finish_return_upper_bound"},
                  program_host_interval={"start_ns": 10**9, "end_ns": 6500000000,
                                         "start_utc_ns": base+10**9, "end_utc_ns": base+6500000000,
                                         "scope": "enqueue_to_l1_terminal_observed_upper_bound"},
                  startup_diagnostics={"expected_after_consumed_frames": 4,
                                       "expected_after_full_run": 4,
                                       "consumer_reported_expectation": 4})
    for name, magic, words, data, bound in (
        ("frames", b"IP120F1\0", 32, rows, 2),
        ("transitions", b"IP120T1\0", 2, list(zip(index.starts, index.challenges, strict=True)), 10),
    ):
        row = artifacts.FRAME if words == 32 else artifacts.TRANSITION
        (capture / "evidence" / (name+".bin")).write_bytes(
            artifacts.HEADER.pack(magic, 1, words, len(data), bound)
            + b"".join(row.pack(*entry) for entry in data))
    import hashlib
    execution = {"schema": "issue120-native-execution-v1", "attempt_identity": identity,
                 "native_returncode": 0, "native_started": True, "timed_out": False, "failure_type": None}
    for name in ("stdout", "stderr"):
        (capture / (name+".bin")).write_bytes(b"")
        execution[name] = {"sha256": hashlib.sha256(b"").hexdigest(), "byte_count": 0}
    (capture / "execution.json").write_text(json.dumps(execution))
    (capture / "result/result.json").write_text(json.dumps({"schema": "issue120-native-result-v1",
                                                         "attempt_identity": identity, "result": native}))
    power = ["timestamp_utc,power_w,aiclk_mhz,asic_temp_c"]
    for second in range(0, 13, 2):
        text = dt.datetime.fromtimestamp(base/1e9+second, dt.UTC).isoformat()
        power.append(f"{text},70,1350,55")
    (tmp_path / f"power-{identity}.csv").write_text("\n".join(power)+"\n")
    monkeypatch.setattr(runtime.issue120_probe, "_environment_record", lambda _: {"commit": "a"*40})
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    if watcher:
        log = tmp_path / "generated/watcher/watcher.log"
        log.parent.mkdir(parents=True)
        log.write_bytes(b"Dump #1 at unit\n")
    record = recording.make_record(plan.RunSpec("fixture", "calibration", "loaded", 2, 10, 60, watcher),
        NativeAttempt(identity, capture), native, {"native_sha256": "b"*64},
        {"session_id": "unit", "harness_commit": "a"*40, "host_boot_id_sha256": "c"*64}, "d"*64,
        execute_start_ns=0, execute_end_ns=10*10**9, start_utc_ns=base,
        end_utc_ns=base+10*10**9, watcher_before=None)
    assert record["timing_evidence"] is not watcher
    path = tmp_path / f"results-{identity}.json"
    recording.write_record(path, record)
    assert recording.load_validated_record(path) == record
    if watcher:
        log = tmp_path / record["watcher"]["artifact"]["file"]
        raw = log.read_bytes()
        log.unlink()
        with pytest.raises(ValueError):
            recording.load_validated_record(path)
        log.write_bytes(raw)
    with (tmp_path / record["artifacts"]["frames"]["file"]).open("ab") as handle:
        handle.write(b"bad")
    with pytest.raises(ValueError):
        recording.load_validated_record(path)


def test_validator_does_not_mutate_binary_input():
    rows, native, index = fixture()
    original = copy.deepcopy(rows)
    coverage.analyze(rows, native, index)
    assert rows == original
