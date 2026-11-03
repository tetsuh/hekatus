"""Preserve Issue #165 measurement semantics while constraining output writes."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from enodia.spec.sim import pulse_support_sweep as sweep


@pytest.mark.parametrize("destination", ["outside", "traversal", "root", "symlink"])
def test_sweep_rejects_paths_outside_its_measurement_tree_before_measuring(
    tmp_path, monkeypatch, destination
):
    root = tmp_path / "measurements"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    paths = {
        "outside": outside / "result",
        "traversal": root / ".." / "outside" / "result",
        "root": root,
        "symlink": link / "result",
    }
    monkeypatch.setattr(sweep, "MEASUREMENT_ROOT", root, raising=False)
    calls = []
    monkeypatch.setattr(sweep, "_git", lambda *args: calls.append(args) or "dirty")
    output_dir = paths[destination]
    with pytest.raises(ValueError, match="measurement directory"):
        sweep.run_sweep(output_dir)
    assert calls == []
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("filename", [sweep.OUTPUT_JSON, sweep.OUTPUT_README])
def test_sweep_refuses_dangling_output_symlinks_before_measuring(tmp_path, monkeypatch, filename):
    root = tmp_path / "measurements"
    output_dir = root / "new-run"
    output_dir.mkdir(parents=True)
    target = tmp_path / "missing-record"
    (output_dir / filename).symlink_to(target)
    monkeypatch.setattr(sweep, "MEASUREMENT_ROOT", root, raising=False)
    calls = []
    monkeypatch.setattr(sweep, "_git", lambda *args: calls.append(args) or "dirty")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sweep.run_sweep(output_dir)
    assert calls == []
    assert not target.exists()


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        ([0.0], [0.0], 0.0),
        ([0.0], [1e-300], math.inf),
        ([1e-300], [2e-300], 1.0),
        ([-2.0, 1.0], [-1.5, 0.75], 0.25),
    ],
)
def test_relative_error_preserves_exact_zero_and_tiny_nonzero_semantics(
    reference, candidate, expected
):
    assert sweep._relative_error(np.array(candidate), np.array(reference)) == expected


@pytest.mark.parametrize("label", ["load average:", "load averages:", "LOAD AVERAGE:"])
def test_load_average_parser_preserves_captured_text(monkeypatch, label):
    raw = f"12:34 up 1 day, 1 user, {label}  0.10, 0.20, 0.30"
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=raw + "\n"),
    )
    observation = sweep._uptime_load_average()
    assert observation["load_average"] == "0.10, 0.20, 0.30"
    assert observation["uptime_output"] == raw
    assert observation["captured_at_utc"]


@pytest.mark.parametrize("raw", ["no load information", "up 1 day, load average:   "])
def test_load_average_parser_refuses_missing_values(monkeypatch, raw):
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=raw),
    )
    with pytest.raises(RuntimeError, match="load average"):
        sweep._uptime_load_average()


def test_existing_measurement_readme_remains_exact_script_output():
    record_dir = (
        Path(__file__).resolve().parents[1]
        / "docs/measurements/issue165-pulse-support/2026-11-02"
    )
    record = json.loads((record_dir / sweep.OUTPUT_JSON).read_text(encoding="utf-8"))
    assert sweep._render_readme(record) == (record_dir / sweep.OUTPUT_README).read_text(
        encoding="utf-8"
    )
