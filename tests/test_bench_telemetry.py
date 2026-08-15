"""Telemetry parsing, kept out of shell quoting where it cannot be tested.

The first run of the harness produced a power trace containing only its
header: the sampler had been embedded in the wrapper script, and its
quoting was wrong. The parsing lives here instead, where it is exercised.
"""

import json
from pathlib import Path

import pytest

from enodia.tt.bench import telemetry
from enodia.tt.bench.telemetry import parse_environment, parse_telemetry, telemetry_csv_row

SNAPSHOT = json.dumps(
    {
        "device_info": [
            {
                "board_info": {"board_type": "p100a", "bus_id": "0000:01:00.0"},
                "firmwares": {"fw_bundle_version": "19.4.1.0"},
                "limits": {"tdp_limit": "150"},
                "telemetry": {
                    "power": " 37.0",
                    "aiclk": " 800",
                    "asic_temperature": "56.8",
                },
            }
        ]
    }
)


def test_telemetry_is_parsed_and_stripped():
    reading = parse_telemetry(SNAPSHOT)

    assert reading == {"power_w": "37.0", "aiclk_mhz": "800", "asic_temp_c": "56.8"}


def test_a_row_carries_a_timestamp_and_the_reading_in_column_order():
    row = telemetry_csv_row(SNAPSHOT, timestamp="2026-08-10T16:08:28+00:00")

    assert row == "2026-08-10T16:08:28+00:00,37.0,800,56.8"


def test_unparseable_telemetry_yields_no_row_rather_than_a_broken_one():
    assert parse_telemetry("not json at all") is None
    assert parse_telemetry(json.dumps({"device_info": []})) is None
    assert telemetry_csv_row("not json at all", timestamp="t") is None


def test_environment_keeps_the_identity_of_the_board_and_its_firmware():
    env = parse_environment(SNAPSHOT)

    assert env["board"]["board_type"] == "p100a"
    assert env["firmware"]["fw_bundle_version"] == "19.4.1.0"
    assert env["limits"]["tdp_limit"] == "150"


def test_environment_survives_a_snapshot_it_cannot_read():
    env = parse_environment("")

    assert "board_snapshot_error" in env
    assert "board" not in env


@pytest.mark.parametrize(
    "snapshot",
    [
        json.dumps({"device_info": {}}),
        json.dumps({"device_info": {"board_info": {}}}),
        json.dumps({"device_info": 5}),
        json.dumps({"device_info": [5]}),
        json.dumps({"device_info": ["not a device"]}),
        json.dumps([]),
    ],
)
def test_a_malformed_snapshot_yields_nothing_rather_than_escaping(snapshot):
    """An exception here would kill the sampler, and a dead sampler is silent
    — which is exactly how the first run produced a trace with only a header."""
    assert parse_telemetry(snapshot) is None
    assert telemetry_csv_row(snapshot, timestamp="t") is None
    assert "board_snapshot_error" in parse_environment(snapshot)


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_sampling_intervals_are_rejected(interval, tmp_path):
    """Zero hammers the tool; negative and non-finite values reach sleep and
    end the sampler. Every one of them yields a trace nobody can use."""
    argv = ["sample", "--out", str(tmp_path / "p.csv"), "--interval", interval]

    with pytest.raises(SystemExit) as excinfo:
        telemetry.main(argv)

    assert excinfo.value.code == 2


class _WriterThatFillsUp:
    """Accepts the header, then fails the way a full disk does."""

    def __init__(self) -> None:
        self.writes = 0

    def write(self, text: str) -> int:
        self.writes += 1
        if self.writes == 1:
            return len(text)
        if self.writes > 20:
            raise AssertionError("the sampler kept retrying after a write failure")
        raise OSError("No space left on device")


def test_a_failing_writer_ends_the_sampler_rather_than_retrying(monkeypatch):
    """Surviving a bad sample is the point; surviving the inability to record
    anything is not — that turns a broken run into a quietly short trace."""
    monkeypatch.setattr(telemetry, "_run", lambda command: SNAPSHOT)

    writer = _WriterThatFillsUp()

    with pytest.raises(OSError, match="No space left"):
        telemetry._sample_loop(writer, interval=0.0)


def _fake_git(status_output="", *, seen=None):
    def fake_git(repo, *args):
        if seen is not None:
            seen.append(args)
        return ("abc1234def", True) if args[0] == "rev-parse" else (status_output, True)

    return fake_git


def test_the_environment_names_the_harness_that_produced_it(monkeypatch):
    """A result is only reproducible if the code that computed it can be named:
    the FLOP accounting behind every efficiency figure lives in this tree."""
    monkeypatch.setattr(telemetry, "_git", _fake_git())

    identity = telemetry.harness_identity()

    assert identity["harness_commit"] == "abc1234def"
    assert identity["harness_dirty"] is False


def test_a_modified_tree_is_recorded_as_such(monkeypatch):
    monkeypatch.setattr(telemetry, "_git", _fake_git(" M enodia/tt/bench/run_matmul.py"))

    assert telemetry.harness_identity()["harness_dirty"] is True


def test_untracked_files_do_not_mark_the_harness_modified(monkeypatch):
    """The toolchain writes build output into the tree on every run, so a flag
    that counts untracked files is true always and says nothing about the code
    that computed the result."""
    seen = []
    monkeypatch.setattr(telemetry, "_git", _fake_git(seen=seen))

    identity = telemetry.harness_identity()

    status = next(args for args in seen if args[0] == "status")
    assert "--untracked-files=no" in status
    assert identity["harness_dirty"] is False


@pytest.mark.parametrize("failing", ["rev-parse", "status"])
def test_a_git_query_that_fails_leaves_the_harness_unknown_rather_than_clean(failing, monkeypatch):
    """A failed query and a clean tree both put nothing on standard output.
    Told apart by the exit status only — and if they are not told apart, a
    harness nobody can name is recorded as an identified, unmodified one."""

    def fake_git(repo, *args):
        if args[0] == failing:
            return "git failed: fatal: detected dubious ownership", False
        return ("abc1234def", True) if args[0] == "rev-parse" else ("", True)

    monkeypatch.setattr(telemetry, "_git", fake_git)

    identity = telemetry.harness_identity()

    assert identity["harness_commit"] is None
    assert identity["harness_dirty"] is None
    assert "dubious ownership" in identity["harness_identity_error"]


def test_a_nonzero_exit_is_a_failure_even_with_empty_output(monkeypatch):
    """The exit status is the whole signal, so it is read from the process
    rather than inferred from what the process printed."""

    class _Failed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository\n"

    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: _Failed())

    output, ok = telemetry._git(Path("/nowhere"), "rev-parse", "HEAD")

    assert ok is False
    assert "not a git repository" in output


def test_a_missing_git_is_a_failure_rather_than_an_exception(monkeypatch):
    """The sampler and the environment capture must survive a missing tool;
    what they must not do is report an identity they did not obtain."""

    def explode(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(telemetry.subprocess, "run", explode)

    output, ok = telemetry._git(Path("/nowhere"), "rev-parse", "HEAD")

    assert ok is False
    assert "FileNotFoundError" in output
