"""Board-free, non-verdict-bearing sampling diagnostics."""

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tools import explore_issue137_sampling as exploration

ARCHIVE = Path(__file__).resolve().parents[1] / "docs/measurements/issue137-board-power/2026-10-18"


def _row(seconds, board=300, current=200, power=150, clock=1100):
    return {
        "timestamp_monotonic_ns": str(1_000_000_000 + int(seconds * 1e9)),
        "board_power_w": str(board),
        "current_a": str(current),
        "power_w": str(power),
        "aiclk_mhz": str(clock),
    }


def test_sample_selection_and_busy_threshold_are_exact():
    rows = [
        _row(59.999999999, board=999),
        _row(60, board=294, current=149, power=90, clock=1095),
        _row(90, board=300, current=150, power=110, clock=1100),
        _row(119.999999999, board=306, current=200, power=120, clock=1105),
        _row(120, board=999),
    ]
    result = exploration.summarize_samples(rows, workload_start_ns=1_000_000_000)
    assert result["sample_count"] == 3
    assert result["selected_trace_row_indices"] == [1, 2, 3]
    assert result["board_power_w"]["mean"] == 300
    assert result["board_power_w"]["sample_sd"] == 6
    assert result["board_power_w"]["naive_se"] == pytest.approx(6 / math.sqrt(3))
    assert result["board_power_w"]["minimum"] == 294
    assert result["board_power_w"]["maximum"] == 306
    assert result["busy"]["sample_count"] == 2
    assert result["busy"]["fraction"] == pytest.approx(2 / 3)
    assert result["busy"]["means"] == {"board_power_w": 303, "current_a": 175, "power_w": 115}
    assert result["aiclk_mhz"] == {"mean": 1100, "sample_sd": 5}


def test_one_sample_and_no_busy_samples_have_explicit_undefined_statistics():
    result = exploration.summarize_samples([_row(60, current=120)], workload_start_ns=1_000_000_000)
    assert result["board_power_w"]["sample_sd"] is None
    assert result["board_power_w"]["naive_se"] is None
    assert result["busy"]["fraction"] == 0
    assert all(value is None for value in result["busy"]["means"].values())


@pytest.mark.parametrize("field", ["board_power_w", "current_a", "power_w", "aiclk_mhz"])
@pytest.mark.parametrize("bad", ["nan", "inf", "", "not-a-number"])
def test_invalid_selected_measurement_is_not_silently_discarded(field, bad):
    row = _row(60)
    row[field] = bad
    with pytest.raises(ValueError):
        exploration.summarize_samples([row], workload_start_ns=1_000_000_000)


def test_empty_plateau_is_an_error():
    rows = [_row(59), _row(120)]
    with pytest.raises(ValueError, match="no samples"):
        exploration.summarize_samples(rows, workload_start_ns=1_000_000_000)


def test_archive_file_cannot_escape_through_parent_or_symlink(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "alias.json").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        exploration.archive_file(root, "../outside.json")
    with pytest.raises(ValueError, match="outside"):
        exploration.archive_file(root, "alias.json")


def test_real_archive_reproduction_preserves_all_existing_bytes():
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARCHIVE.rglob("*")
        if p.suffix in (".json", ".csv")
    }
    report = exploration.explore_archive(ARCHIVE)
    assert report["verdict_bearing"] is False
    assert [r["sample_count"] for r in report["runs"]] == [26, 27, 27, 27, 27, 26, 27, 26]
    runs = {r["run_number"]: r for r in report["runs"]}
    assert runs[5]["board_power_w"]["mean"] == pytest.approx(292.44444444444446)
    assert runs[5]["duration_weighted_reference"]["board_power_w"] == pytest.approx(
        290.8248480083167
    )
    assert runs[2]["busy"]["means"]["current_a"] == pytest.approx(199.6315789473684)
    assert runs[7]["busy"]["means"]["current_a"] == pytest.approx(196.6315789473684)
    assert report["summary"]["high_run_duration_weighted_board_power_mean_w"] == pytest.approx(
        297.5557102906875
    )
    assert report == json.loads((ARCHIVE / "exploratory-sampling.json").read_text())
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_cli_emits_json_to_stdout_without_creating_files(tmp_path):
    script = Path(exploration.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--archive", str(ARCHIVE)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report == exploration.explore_archive(ARCHIVE)
    assert list(tmp_path.iterdir()) == []


def test_summarize_does_not_mutate_rows():
    rows = [_row(60), _row(90)]
    original = copy.deepcopy(rows)
    exploration.summarize_samples(rows, workload_start_ns=1_000_000_000)
    assert rows == original
