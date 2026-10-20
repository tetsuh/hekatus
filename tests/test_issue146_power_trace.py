"""Board-free Issue #146 trace schema and telemetry coverage."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from enodia.tt.bench import power_trace_issue146 as trace
from enodia.tt.bench import telemetry

_BOARD_ID = "fixture-board"
_BUS_ID = "0000:09:00.0"
_GDDR_WORDS = {
    "GDDR_0_1_TEMP": "0x36363234",
    "GDDR_2_3_TEMP": "0x32343432",
    "GDDR_4_5_TEMP": "0x30323638",
    "GDDR_6_7_TEMP": "0x34342e30",
}
_ARB_WORDS = {
    "AICLK_ARB_MIN": "0x10546",
    "AICLK_ARB_MAX": "0x7048f",
    "ENABLED_MIN_ARB": "0x3",
    "ENABLED_MAX_ARB": "0x1d1",
}


def _snapshot(*, arb_words: dict[str, str] | None = None) -> dict[str, object]:
    smbus = {
        "THERM_TRIP_COUNT": "0x0",
        **_GDDR_WORDS,
        **_ARB_WORDS,
    }
    if arb_words is not None:
        smbus.update(arb_words)
    return {
        "timestamp_utc": "2026-10-20T00:00:00+00:00",
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
        "device_info": [
            {
                "board_info": {
                    "board_id": _BOARD_ID,
                    "serial": _BOARD_ID,
                    "bus_id": _BUS_ID,
                },
                "firmwares": {"fw_bundle_version": "19.6.0.0"},
                "limits": {"asic_fmax": "1350"},
                "smbus_telem": smbus,
                "telemetry": {
                    "power": "50",
                    "aiclk": "1350",
                    "asic_temperature": "60",
                    "current": "50",
                    "voltage": "0.8",
                    "board_power": "250",
                    "fan_speed": "1000",
                    "heartbeat": "1",
                },
            }
        ],
    }


def _row(timestamp: int = 0, **raw_words: str) -> dict[str, object]:
    return {
        "timestamp_utc": "2026-10-20T00:00:00+00:00",
        "power_w": 50,
        "aiclk_mhz": 1350,
        "asic_temp_c": 60,
        "timestamp_monotonic_ns": timestamp,
        "snapshot_elapsed_ns": 10,
        "current_a": 50,
        "voltage_v": 0.8,
        "board_power_w": 250,
        "fan_rpm": 1000,
        "heartbeat": 1,
        "therm_trip_count": 0,
        "throttle_status": None,
        "throttle_reason": None,
        "gddr_0_1_temp_c": _GDDR_WORDS["GDDR_0_1_TEMP"],
        "gddr_2_3_temp_c": _GDDR_WORDS["GDDR_2_3_TEMP"],
        "gddr_4_5_temp_c": _GDDR_WORDS["GDDR_4_5_TEMP"],
        "gddr_6_7_temp_c": _GDDR_WORDS["GDDR_6_7_TEMP"],
        "aiclk_arb_min_raw": _ARB_WORDS["AICLK_ARB_MIN"],
        "aiclk_arb_max_raw": _ARB_WORDS["AICLK_ARB_MAX"],
        "enabled_min_arb_raw": _ARB_WORDS["ENABLED_MIN_ARB"],
        "enabled_max_arb_raw": _ARB_WORDS["ENABLED_MAX_ARB"],
        **raw_words,
    }


def _payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": trace.ISSUE146_POWER_TRACE_SCHEMA,
        "schema_version": trace.ISSUE146_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(trace.ISSUE146_POWER_TRACE_COLUMNS),
        "samples": rows,
    }


def test_issue146_schema_appends_raw_arbiter_words_to_issue137_prefix() -> None:
    assert trace.ISSUE146_POWER_TRACE_SCHEMA == "issue-146-power-trace-v1"
    assert trace.ISSUE146_POWER_TRACE_COLUMNS[: len(trace.ISSUE137_POWER_TRACE_COLUMNS)] == (
        trace.ISSUE137_POWER_TRACE_COLUMNS
    )
    assert trace.ISSUE146_POWER_TRACE_COLUMNS[-4:] == (
        "aiclk_arb_min_raw",
        "aiclk_arb_max_raw",
        "enabled_min_arb_raw",
        "enabled_max_arb_raw",
    )
    assert trace.issue146_power_trace_header() == ",".join(trace.ISSUE146_POWER_TRACE_COLUMNS)


def test_issue146_snapshot_and_telemetry_environment_retain_raw_words() -> None:
    snapshot = _snapshot()
    row = trace.parse_issue146_snapshot(
        snapshot,
        board_id=_BOARD_ID,
        bus_id=_BUS_ID,
        timestamp_monotonic_ns=0,
        snapshot_elapsed_ns=10,
    )

    assert row["aiclk_arb_min_raw"] == "0x10546"
    assert row["aiclk_arb_max_raw"] == "0x7048f"
    assert row["enabled_min_arb_raw"] == "0x3"
    assert row["enabled_max_arb_raw"] == "0x1d1"
    assert row["gddr_0_1_temp_c"] == "0x36363234"
    assert "aiclk_arb_max_mhz" not in row
    assert "aiclk_arb_max_index" not in row

    environment = telemetry.parse_environment(json.dumps(snapshot))
    assert environment["smbus_telem_limits"] == {
        "AICLK_ARB_MIN": "0x10546",
        "AICLK_ARB_MAX": "0x7048f",
        "ENABLED_MIN_ARB": "0x3",
        "ENABLED_MAX_ARB": "0x1d1",
    }


def test_issue146_csv_and_json_round_trip_preserves_words_and_gddr_provenance(
    tmp_path: Path,
) -> None:
    rows = [_row(), _row(10, aiclk_arb_max_raw="0x0")]
    csv_path = tmp_path / "issue146.csv"
    trace.write_issue146_power_trace(csv_path, rows)
    parsed = trace.read_issue146_power_trace(csv_path, source_established=True)

    assert parsed["valid"] is True
    assert parsed["schema"] == trace.ISSUE146_POWER_TRACE_SCHEMA
    assert parsed["samples"][1]["aiclk_arb_max_raw"] == "0x0"
    assert parsed["unit_provenance"] == trace.ISSUE137_UNIT_PROVENANCE
    assert parsed["gddr_decoded"][0]["gddr_0_1_temp_c"]["status"] == "decoded"

    json_path = tmp_path / "issue146.json"
    json_path.write_text(json.dumps(_payload(rows)), encoding="utf-8")
    from_json = trace.read_issue146_power_trace_snapshot(json_path)
    assert from_json["valid"] is True
    assert from_json["samples"][0]["enabled_max_arb_raw"] == "0x1d1"

    mapped = trace.read_issue146_power_trace_mapping(_payload(rows))
    assert mapped["valid"] is True
    assert mapped["samples"][0]["aiclk_arb_min_raw"] == "0x10546"
    assert "aiclk_arb_max_mhz" not in mapped["samples"][0]


def test_issue146_zero_words_are_valid_raw_data() -> None:
    row = _row(
        aiclk_arb_min_raw="0x0",
        aiclk_arb_max_raw="0x00000000",
        enabled_min_arb_raw="0x0",
        enabled_max_arb_raw="0x0",
    )
    formatted = trace.format_issue146_power_trace_row(row)
    assert "0x00000000" in formatted
    assert trace.issue146_trace_row(
        _snapshot(arb_words={key: "0x0" for key in _ARB_WORDS}),
        board_id=_BOARD_ID,
        bus_id=_BUS_ID,
        timestamp_monotonic_ns=0,
        snapshot_elapsed_ns=10,
    )


def test_issue146_raw_word_hex_digit_case_survives_csv_round_trip(tmp_path: Path) -> None:
    row = _row(
        aiclk_arb_min_raw="0xA1b2C3d4",
        aiclk_arb_max_raw="0x0000EfF0",
        enabled_min_arb_raw="0xAbC",
        enabled_max_arb_raw="0x1D1",
    )
    path = tmp_path / "case.csv"
    trace.write_issue146_power_trace(path, [row, _row(10)])
    result = trace.read_issue146_power_trace(path)
    assert result["valid"] is True
    assert result["samples"][0]["aiclk_arb_min_raw"] == "0xA1b2C3d4"
    assert result["samples"][0]["aiclk_arb_max_raw"] == "0x0000EfF0"
    assert result["samples"][0]["enabled_min_arb_raw"] == "0xAbC"
    assert result["samples"][0]["enabled_max_arb_raw"] == "0x1D1"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("aiclk_arb_min_raw"),
        lambda row: row.__setitem__("aiclk_arb_max_raw", "not-a-word"),
        lambda row: row.__setitem__("enabled_max_arb_raw", "0X1d1"),
        lambda row: row.__setitem__("enabled_min_arb_raw", "0x100000000"),
    ],
)
def test_issue146_format_rejects_missing_or_malformed_raw_words(mutation) -> None:
    row = _row()
    mutation(row)
    with pytest.raises(trace.Issue146PowerTraceError, match="raw hexadecimal word|missing"):
        trace.format_issue146_power_trace_row(row)


def test_issue146_reader_does_not_drop_invalid_rows(tmp_path: Path) -> None:
    row = _row()
    row.pop("enabled_max_arb_raw")
    fields = list(trace.ISSUE146_POWER_TRACE_COLUMNS)
    fields.remove("enabled_max_arb_raw")
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="").writerow([row[field] for field in fields])
    path = tmp_path / "missing.csv"
    path.write_text(",".join(fields) + "\n" + buffer.getvalue() + "\n", encoding="utf-8")

    result = trace.read_issue146_power_trace(path)
    assert result["valid"] is False
    assert result["valid_row_count"] == 0
    assert result["invalid_row_count"] == 1
    assert any("enabled_max_arb_raw" in error for error in result["errors"])

    payload = _payload([_row()])
    payload["samples"][0].pop("enabled_max_arb_raw")
    mapped = trace.read_issue146_power_trace_mapping(payload)
    assert mapped["valid"] is False
    assert any("enabled_max_arb_raw" in error for error in mapped["errors"])


def test_issue146_validation_api_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "valid.csv"
    trace.write_issue146_power_trace(path, [_row(), _row(10)])
    report = trace.validate_issue146_power_trace(path)
    assert report["valid"] is True
    assert report["schema"] == trace.ISSUE146_POWER_TRACE_SCHEMA
    assert report["columns"] == list(trace.ISSUE146_POWER_TRACE_COLUMNS)

    bad = _payload([_row()])
    bad["schema"] = "issue-137-power-trace-v1"
    bad_report = trace.validate_issue146_power_trace(bad)
    assert bad_report["valid"] is False
    assert bad_report["failures"]
