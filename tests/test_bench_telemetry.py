"""Telemetry parsing, kept out of shell quoting where it cannot be tested.

The first run of the harness produced a power trace containing only its
header: the sampler had been embedded in the wrapper script, and its
quoting was wrong. The parsing lives here instead, where it is exercised.
"""

import json

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
