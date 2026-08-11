"""Telemetry parsing, kept out of shell quoting where it cannot be tested.

The first run of the harness produced a power trace containing only its
header: the sampler had been embedded in the wrapper script, and its
quoting was wrong. The parsing lives here instead, where it is exercised.
"""

import json

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
    with pytest.raises(SystemExit) as excinfo:
        telemetry.main(["sample", "--out", str(tmp_path / "p.csv"), "--interval", interval])

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

    with pytest.raises(OSError, match="No space left"):
        telemetry._sample_loop(_WriterThatFillsUp(), interval=0.0)
