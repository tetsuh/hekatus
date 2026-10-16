"""Board-free Issue #121 Q2 extended power-trace coverage."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import pytest

from enodia.tt.bench import telemetry
from enodia.tt.bench.power_trace_extended import (
    EXTENDED_POWER_TRACE_COLUMNS,
    EXTENDED_POWER_TRACE_HEADER,
    EXTENDED_POWER_TRACE_SCHEMA,
    ExtendedPowerTraceError,
    _normalise_source_number,
    extended_telemetry_csv_row,
    read_extended_power_trace_csv,
    read_extended_power_trace_snapshot,
    select_extended_board,
    validate_extended_power_trace,
)

LEGACY_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")


def _snapshot(*, throttle: bool = False) -> str:
    devices = [
        {
            "board_info": {"board_id": "fixture-board-b", "bus_id": "0000:02:00.0"},
            "telemetry": {
                "voltage": "12.5",
                "current": "4.0",
                "power": "999",
                "board_power": "50.0",
                "aiclk": "1350",
                "asic_temperature": "62",
                "fan_speed": "1200",
                "heartbeat": "7",
            },
            "smbus_telem": {
                "THERM_TRIP_COUNT": "0x2",
                "BOARD_POWER_LIMIT": "0x12c",
                "TDC_LIMIT_MAX": "0xc8",
                **(
                    {"THROTTLE_STATUS": "active", "THROTTLE_REASON": "thermal"}
                    if throttle
                    else {}
                ),
            },
            "limits": {"board_power_limit": "fixture-limit"},
        },
        {
            "board_info": {"board_id": "fixture-board-a", "bus_id": "0000:01:00.0"},
            "telemetry": {
                "voltage": "11.0",
                "current": "1.0",
                "power": "11",
                "board_power": "11.0",
                "aiclk": "800",
                "asic_temperature": "40",
                "fan_speed": "900",
                "heartbeat": "3",
            },
            "smbus_telem": {
                "THERM_TRIP_COUNT": "0x1",
                "BOARD_POWER_LIMIT": "0x12c",
                "TDC_LIMIT_MAX": "0xc8",
            },
            "limits": {"board_power_limit": "fixture-limit-a"},
        },
    ]
    return json.dumps(
        {
            "host_sw_vers": {
                "tt_smi": "fixture-version",
                "tt_metal": "fixture-version",
            },
            "host_info": {"hostname": "not-retained", "user": "not-retained"},
            "processes": [{"pid": 123, "command": "not-retained"}],
            "device_info": devices,
        }
    )


def _row(
    *,
    timestamp: int,
    elapsed: int,
    current: str,
    voltage: str,
    power: str,
    fan: str,
    heartbeat: str,
    therm: str,
    aiclk: str = "1000",
    temperature: str = "40",
) -> str:
    return ",".join(
        (
            f"2026-01-01T00:00:{timestamp // 1_000_000_000:02d}+00:00",
            power,
            aiclk,
            temperature,
            str(timestamp),
            str(elapsed),
            current,
            voltage,
            power,
            fan,
            heartbeat,
            therm,
            "",
            "",
        )
    )


def test_extended_sampler_measures_one_snapshot_without_changing_legacy(
    monkeypatch: pytest.MonkeyPatch,
):
    output = io.StringIO()
    clock = iter((100, 250))
    utc = iter(("2026-01-01T00:00:01+00:00",))
    monkeypatch.setattr(telemetry, "_run", lambda _command: _snapshot())
    monkeypatch.setattr(
        telemetry,
        "_resolve_device_node_to_pci",
        lambda _node: {"0000:01:00.0"},
    )
    monkeypatch.setattr(
        telemetry.time,
        "sleep",
        lambda _interval: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        telemetry._sample_extended_loop(
            output,
            interval=1.0,
            board_id="fixture-board-a",
            bus_id="0000:01:00.0",
            monotonic_ns=lambda: next(clock),
            utc_now=lambda: next(utc),
        )

    lines = output.getvalue().splitlines()
    assert lines[0] == EXTENDED_POWER_TRACE_HEADER
    row = next(csv.reader([lines[1]]))
    assert row[0] == "2026-01-01T00:00:01+00:00"
    assert row[4:6] == ["250", "150"]
    assert telemetry.CSV_HEADER == ",".join(LEGACY_COLUMNS)


@pytest.mark.parametrize(
    "invalid_snapshot",
    [
        "<unavailable: tt-smi failed>",
        "not json at all",
        json.dumps({}),
        json.dumps({"device_info": []}),
        json.dumps({"device_info": [5]}),
    ],
)
def test_extended_sampler_drops_transient_malformed_snapshot_before_valid_row(
    monkeypatch: pytest.MonkeyPatch, invalid_snapshot: str
):
    output = io.StringIO()
    snapshots = iter((invalid_snapshot, _snapshot()))
    clock = iter((100, 200, 300, 450))
    utc = iter(("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00"))
    sleep_calls = 0

    def sleep(_interval: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(telemetry, "_run", lambda _command: next(snapshots))
    monkeypatch.setattr(
        telemetry,
        "_resolve_device_node_to_pci",
        lambda _node: {"0000:01:00.0"},
    )
    monkeypatch.setattr(telemetry.time, "sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        telemetry._sample_extended_loop(
            output,
            interval=1.0,
            board_id="fixture-board-a",
            bus_id="0000:01:00.0",
            monotonic_ns=lambda: next(clock),
            utc_now=lambda: next(utc),
        )

    lines = output.getvalue().splitlines()
    assert lines[0] == EXTENDED_POWER_TRACE_HEADER
    row = next(csv.reader([lines[1]]))
    assert row[0] == "2026-01-01T00:00:01+00:00"
    assert row[4:6] == ["450", "150"]
    assert len(lines) == 2


def test_extended_sampler_fails_when_target_board_is_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    output = io.StringIO()
    clock = iter((100, 200))
    monkeypatch.setattr(telemetry, "_run", lambda _command: _snapshot())
    monkeypatch.setattr(
        telemetry,
        "_resolve_device_node_to_pci",
        lambda _node: {"0000:01:00.0"},
    )

    with pytest.raises(ExtendedPowerTraceError, match="board_id"):
        telemetry._sample_extended_loop(
            output,
            interval=1.0,
            board_id="missing-board",
            bus_id="0000:01:00.0",
            monotonic_ns=lambda: next(clock),
            utc_now=lambda: "2026-01-01T00:00:00+00:00",
        )


def test_extended_sampler_rejects_default_node_pci_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    output = io.StringIO()
    resolved_nodes: list[str] = []
    monkeypatch.delenv("HEKATUS_TT_DEVICE_NODE", raising=False)
    monkeypatch.setattr(telemetry, "_run", lambda _command: _snapshot())

    def resolve_node(node: str) -> set[str]:
        resolved_nodes.append(node)
        return {"0000:02:00.0"}

    monkeypatch.setattr(telemetry, "_resolve_device_node_to_pci", resolve_node)

    with pytest.raises(ExtendedPowerTraceError, match="device node"):
        telemetry._sample_extended_loop(
            output,
            interval=1.0,
            board_id="fixture-board-a",
            bus_id="0000:01:00.0",
            monotonic_ns=iter((100, 200)).__next__,
            utc_now=lambda: "2026-01-01T00:00:00+00:00",
        )

    assert resolved_nodes == [telemetry._DEFAULT_DEVICE_NODE]
    assert output.getvalue().splitlines() == [EXTENDED_POWER_TRACE_HEADER]


def test_extended_sampler_selects_board_identity_not_device_info_position():
    snapshot = _snapshot()

    selected = select_extended_board(
        snapshot, board_id="fixture-board-a", bus_id="0000:01:00.0"
    )
    assert selected["board_info"]["board_id"] == "fixture-board-a"
    assert telemetry.parse_telemetry(
        snapshot,
        node_resolver=lambda _node: {"0000:01:00.0"},
        board_id="fixture-board-a",
        bus_id="0000:01:00.0",
    ) == {"power_w": "11", "aiclk_mhz": "800", "asic_temp_c": "40"}

    assert (
        extended_telemetry_csv_row(
            snapshot,
            board_id="fixture-board-a",
            bus_id="0000:01:00.0",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            timestamp_monotonic_ns=100,
            snapshot_elapsed_ns=0,
        )
        == "2026-01-01T00:00:00+00:00,11,800,40,100,0,1.0,11.0,11.0,900,3,1,,"
    )


def test_extended_selector_normalizes_pci_bus_ids_before_matching():
    payload = json.loads(_snapshot())
    payload["device_info"][1]["board_info"]["bus_id"] = " 0000:0A:0B.1 "

    selected = select_extended_board(
        payload, board_id="fixture-board-a", bus_id="0000:0a:0b.1"
    )

    assert selected["board_info"]["board_id"] == "fixture-board-a"


def test_padded_board_id_canonicalization_matches_extended_and_q2_selectors():
    payload = json.loads(_snapshot())
    payload["device_info"][1]["board_info"]["board_id"] = "  fixture-board-a  "

    selected = select_extended_board(
        payload, board_id=" fixture-board-a ", bus_id="0000:01:00.0"
    )
    reading = telemetry.parse_telemetry(
        json.dumps(payload),
        node_resolver=lambda _node: {"0000:01:00.0"},
        board_id=" fixture-board-a ",
        bus_id="0000:01:00.0",
    )

    assert selected["board_info"]["board_id"] == "  fixture-board-a  "
    assert reading == {"power_w": "11", "aiclk_mhz": "800", "asic_temp_c": "40"}


@pytest.mark.parametrize("bus_id", ["not-a-pci-bdf", "0000:01:00.8"])
def test_extended_selector_rejects_malformed_requested_bus_id(bus_id: str):
    with pytest.raises(ExtendedPowerTraceError, match="valid PCI BDF"):
        select_extended_board(
            _snapshot(), board_id="fixture-board-a", bus_id=bus_id
        )


def test_extended_selector_rejects_malformed_snapshot_bus_id():
    payload = json.loads(_snapshot())
    payload["device_info"][1]["board_info"]["bus_id"] = "not-a-pci-bdf"

    with pytest.raises(ExtendedPowerTraceError, match="bus_id"):
        select_extended_board(
            payload, board_id="fixture-board-a", bus_id="0000:01:00.0"
        )


@pytest.mark.parametrize(
    ("board_id", "bus_id"),
    [
        ("missing-board", "0000:01:00.0"),
        ("fixture-board-a", "0000:09:00.0"),
    ],
)
def test_extended_sampler_rejects_missing_or_mismatched_board_identity(
    board_id: str, bus_id: str
):
    with pytest.raises(ExtendedPowerTraceError):
        select_extended_board(_snapshot(), board_id=board_id, bus_id=bus_id)

    row = extended_telemetry_csv_row(
        _snapshot(),
        board_id=board_id,
        bus_id=bus_id,
        timestamp_utc="2026-01-01T00:00:00+00:00",
        timestamp_monotonic_ns=100,
        snapshot_elapsed_ns=0,
    )
    assert row is None


def test_extended_board_identity_must_be_unique():
    payload = json.loads(_snapshot())
    payload["device_info"].append(payload["device_info"][1])
    with pytest.raises(ExtendedPowerTraceError, match="exactly one"):
        select_extended_board(
            payload, board_id="fixture-board-a", bus_id="0000:01:00.0"
        )


def test_extended_trace_requires_all_versioned_columns(tmp_path: Path):
    missing = list(EXTENDED_POWER_TRACE_COLUMNS)
    missing.remove("board_power_w")
    path = tmp_path / "extended.csv"
    path.write_text(",".join(missing) + "\n")

    trace = read_extended_power_trace_csv(path)
    assert trace["schema"] == EXTENDED_POWER_TRACE_SCHEMA
    assert trace["valid"] is False
    assert trace["readable"] is False
    assert "unexpected columns" in trace["error"]
    assert validate_extended_power_trace(trace)["valid"] is False


def test_extended_trace_allows_missing_throttle_keys_as_null():
    row = extended_telemetry_csv_row(
        _snapshot(),
        board_id="fixture-board-a",
        bus_id="0000:01:00.0",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        timestamp_monotonic_ns=100,
        snapshot_elapsed_ns=0,
    )
    assert row is not None
    parsed = next(csv.DictReader(io.StringIO(EXTENDED_POWER_TRACE_HEADER + "\n" + row)))
    assert parsed["throttle_status"] == ""
    assert parsed["throttle_reason"] == ""

    with_throttle = extended_telemetry_csv_row(
        _snapshot(throttle=True),
        board_id="fixture-board-b",
        bus_id="0000:02:00.0",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        timestamp_monotonic_ns=100,
        snapshot_elapsed_ns=0,
    )
    assert with_throttle is not None
    assert with_throttle.endswith(",active,thermal")


def test_extended_trace_aggregates_legacy_measurements_without_inference(tmp_path: Path):
    row = _row(
        timestamp=100,
        elapsed=0,
        current="2.0",
        voltage="12.0",
        power="24.0",
        fan="1000",
        heartbeat="1",
        therm="0",
        aiclk="1200",
        temperature="55",
    )
    path = tmp_path / "extended.csv"
    path.write_text(
        EXTENDED_POWER_TRACE_HEADER
        + "\n"
        + row
        + "\n"
        + row.replace("100,0", "200,100")
        + "\n"
    )
    trace = read_extended_power_trace_csv(
        path, run_start_monotonic_ns=100, run_end_monotonic_ns=200
    )
    assert trace["valid"] is True
    assert trace["aggregates"]["power_w"]["time_weighted"] == pytest.approx(24.0)
    assert trace["aggregates"]["aiclk_mhz"]["time_weighted"] == pytest.approx(1200.0)
    assert trace["aggregates"]["asic_temp_c"]["time_weighted"] == pytest.approx(55.0)
    assert trace["aggregates"]["board_power_w"]["time_weighted"] == pytest.approx(24.0)
    assert trace["samples"][0]["throttle_status"] is None


@pytest.mark.parametrize(
    ("aiclk", "timestamps"),
    [("800", (0, 1, 3)), ("1000", (0, 1, 9))],
    ids=["constant-800", "constant-1000"],
)
def test_extended_trace_clamps_constant_clock_roundoff_for_multiple_durations(
    tmp_path: Path, aiclk: str, timestamps: tuple[int, ...]
):
    rows = [
        _row(
            timestamp=timestamp,
            elapsed=10,
            current="1.0",
            voltage="10.0",
            power="10.0",
            fan="100",
            heartbeat="1",
            therm="0",
            aiclk=aiclk,
            temperature="40",
        )
        for timestamp in timestamps
    ]
    path = tmp_path / f"constant-{aiclk}.csv"
    path.write_text(EXTENDED_POWER_TRACE_HEADER + "\n" + "\n".join(rows) + "\n")

    trace = read_extended_power_trace_csv(
        path,
        run_start_monotonic_ns=timestamps[0],
        run_end_monotonic_ns=timestamps[-1],
    )

    assert trace["valid"] is True
    assert trace["aggregates"]["aiclk_mhz"]["time_weighted"] == float(aiclk)


def test_extended_trace_allows_one_ulp_weighted_bound_noise_but_rejects_real_excursion(
    tmp_path: Path,
):
    rows = [
        _row(
            timestamp=timestamp,
            elapsed=10,
            current="1.0",
            voltage="10.0",
            power="10.0",
            fan="100",
            heartbeat="1",
            therm="0",
            aiclk="800",
            temperature="40",
        )
        for timestamp in (0, 1, 3)
    ]
    path = tmp_path / "weighted-bound.csv"
    path.write_text(EXTENDED_POWER_TRACE_HEADER + "\n" + "\n".join(rows) + "\n")
    trace = read_extended_power_trace_csv(
        path, run_start_monotonic_ns=0, run_end_monotonic_ns=3
    )

    near_bound = math.nextafter(800.0, math.inf)
    trace["aggregates"]["aiclk_mhz"]["time_weighted"] = near_bound
    assert validate_extended_power_trace(trace)["valid"] is True

    trace["aggregates"]["aiclk_mhz"]["time_weighted"] = math.nextafter(
        near_bound, math.inf
    )
    report = validate_extended_power_trace(trace)
    assert report["valid"] is False
    assert any("outside its bounds" in failure for failure in report["failures"])


@pytest.mark.parametrize(
    "value",
    [
        "1.0",
        "1e3",
        "1.5",
        "1e-3",
        "1e-400",
        "9007199254740993.0",
        "0o10",
        "0b10",
    ],
)
def test_extended_integer_columns_reject_non_exact_integer_spellings(value: str):
    for field in (
        "timestamp_monotonic_ns",
        "snapshot_elapsed_ns",
        "fan_rpm",
        "heartbeat",
        "therm_trip_count",
    ):
        with pytest.raises(ExtendedPowerTraceError, match="non-negative integer"):
            _normalise_source_number(value, field=field)


def test_extended_integer_columns_keep_hex_and_reject_imprecise_float_inputs():
    assert _normalise_source_number("0x2", field="therm_trip_count") == 2
    assert _normalise_source_number("0002", field="heartbeat") == 2
    assert _normalise_source_number(float((1 << 53) - 1), field="heartbeat") == (
        1 << 53
    ) - 1
    with pytest.raises(ExtendedPowerTraceError, match="non-negative integer"):
        _normalise_source_number(float(1 << 53), field="heartbeat")


@pytest.mark.parametrize(
    "bound",
    ["1.0", "1e3", "1.5", "1e-400", "9007199254740993.0", float(1 << 53)],
    ids=["decimal-fraction", "exponent", "fraction", "underflow", "precision-loss", "float"],
)
def test_extended_trace_rejects_non_exact_timestamp_bounds(
    tmp_path: Path, bound: object
):
    rows = [
        _row(
            timestamp=timestamp,
            elapsed=10,
            current="1.0",
            voltage="10.0",
            power="10.0",
            fan="100",
            heartbeat="1",
            therm="0",
        )
        for timestamp in (0, 1)
    ]
    path = tmp_path / "timestamp-bound.csv"
    path.write_text(EXTENDED_POWER_TRACE_HEADER + "\n" + "\n".join(rows) + "\n")

    trace = read_extended_power_trace_csv(
        path,
        run_start_monotonic_ns=bound,
        run_end_monotonic_ns=1,
    )

    assert trace["readable"] is False
    assert trace["valid"] is False
    assert "non-negative integer" in trace["error"]
    assert trace["errors"] == [trace["error"]]


def test_extended_trace_converts_oversized_csv_field_to_invalid_result(tmp_path: Path):
    row_text = _row(
        timestamp=0,
        elapsed=10,
        current="1.0",
        voltage="10.0",
        power="10.0",
        fan="100",
        heartbeat="1",
        therm="0",
    )
    row = next(csv.reader([row_text]))
    row[-2] = "throttle-label-" + "x" * csv.field_size_limit()
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="").writerow(row)
    path = tmp_path / "oversized-label.csv"
    path.write_text(EXTENDED_POWER_TRACE_HEADER + "\n" + buffer.getvalue() + "\n")

    trace = read_extended_power_trace_csv(path)

    assert trace["readable"] is False
    assert trace["valid"] is False
    assert "field larger than field limit" in trace["error"]
    assert trace["errors"] == [trace["error"]]
    assert validate_extended_power_trace(trace)["valid"] is False


def test_extended_json_sidecar_rejects_non_mapping_sample(tmp_path: Path):
    path = tmp_path / "invalid-sample.json"
    path.write_text(
        json.dumps(
            {
                "schema": EXTENDED_POWER_TRACE_SCHEMA,
                "schema_version": 1,
                "columns": list(EXTENDED_POWER_TRACE_COLUMNS),
                "samples": [None],
            }
        )
    )

    trace = read_extended_power_trace_snapshot(path)

    assert trace["readable"] is False
    assert trace["valid"] is False
    assert "samples[0]" in trace["error"]
    assert "AttributeError" not in trace["error"]
    assert trace["errors"] == [trace["error"]]
    assert validate_extended_power_trace(trace)["valid"] is False


def test_extended_trace_uses_left_held_min_max_and_time_weighted_values(tmp_path: Path):
    rows = [
        _row(
            timestamp=0,
            elapsed=10,
            current="1.0",
            voltage="10.0",
            power="10.0",
            fan="100",
            heartbeat="1",
            therm="0",
            aiclk="1000",
            temperature="40",
        ),
        _row(
            timestamp=2_000_000_000,
            elapsed=20,
            current="3.0",
            voltage="12.0",
            power="36.0",
            fan="300",
            heartbeat="2",
            therm="1",
            aiclk="1200",
            temperature="50",
        ),
        _row(
            timestamp=5_000_000_000,
            elapsed=30,
            current="2.0",
            voltage="11.0",
            power="22.0",
            fan="200",
            heartbeat="3",
            therm="2",
            aiclk="1100",
            temperature="45",
        ),
        _row(
            timestamp=6_000_000_000,
            elapsed=40,
            current="2.0",
            voltage="11.0",
            power="22.0",
            fan="200",
            heartbeat="3",
            therm="2",
            aiclk="1100",
            temperature="45",
        ),
    ]
    path = tmp_path / "extended.csv"
    path.write_text(EXTENDED_POWER_TRACE_HEADER + "\n" + "\n".join(rows) + "\n")

    trace = read_extended_power_trace_csv(
        path, run_start_monotonic_ns=1_000_000_000, run_end_monotonic_ns=6_000_000_000
    )
    assert trace["valid"] is True
    power = trace["aggregates"]["board_power_w"]
    assert power["min"] == 10.0
    assert power["max"] == 36.0
    assert power["time_weighted"] == pytest.approx((10.0 + 36.0 * 3.0 + 22.0) / 5.0)
    assert power["time_weighted_ns"] == 5_000_000_000
    assert trace["aggregates"]["aiclk_mhz"]["min"] == 1000.0
    assert trace["aggregates"]["aiclk_mhz"]["max"] == 1200.0
    assert trace["aggregates"]["asic_temp_c"]["min"] == 40.0
    assert trace["aggregates"]["asic_temp_c"]["max"] == 50.0
    assert trace["aggregates"]["snapshot_elapsed_ns"]["min"] == 10
    assert trace["aggregates"]["snapshot_elapsed_ns"]["max"] == 30
    assert trace["aggregates"]["fan_rpm"]["min"] == 100
    assert trace["aggregates"]["fan_rpm"]["max"] == 300


def test_extended_trace_rejects_null_measurements(tmp_path: Path):
    row = _row(
        timestamp=100,
        elapsed=10,
        current="",
        voltage="12.0",
        power="24.0",
        fan="1000",
        heartbeat="1",
        therm="0",
    )
    path = tmp_path / "extended-null.csv"
    path.write_text(EXTENDED_POWER_TRACE_HEADER + "\n" + row + "\n")
    trace = read_extended_power_trace_csv(path)
    assert trace["valid"] is False
    assert trace["invalid_row_count"] == 1
    assert "current_a" in trace["error"]


def test_extended_trace_does_not_fabricate_board_power_from_power():
    payload = json.loads(_snapshot())
    del payload["device_info"][1]["telemetry"]["board_power"]
    assert (
        extended_telemetry_csv_row(
            payload,
            board_id="fixture-board-a",
            bus_id="0000:01:00.0",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            timestamp_monotonic_ns=100,
            snapshot_elapsed_ns=10,
        )
        is None
    )


def test_legacy_power_trace_columns_and_row_are_unchanged():
    assert telemetry.POWER_TRACE_COLUMNS == LEGACY_COLUMNS
    assert telemetry.CSV_HEADER == ",".join(LEGACY_COLUMNS)
    assert telemetry.telemetry_csv_row(
        _snapshot(),
        timestamp="2026-01-01T00:00:00+00:00",
        node_resolver=lambda _node: {"0000:01:00.0"},
        board_id="fixture-board-a",
        bus_id="0000:01:00.0",
    ) == "2026-01-01T00:00:00+00:00,11,800,40"


def test_environment_keeps_raw_hierarchy_without_host_snapshot_sections():
    environment = telemetry.parse_environment(
        _snapshot(),
        node_resolver=lambda _node: {"0000:02:00.0"},
        board_id="fixture-board-b",
        bus_id="0000:02:00.0",
    )
    assert environment["host_sw_vers"] == {
        "tt_smi": "fixture-version",
        "tt_metal": "fixture-version",
    }
    assert environment["smbus_telem"] == {
        "THERM_TRIP_COUNT": "0x2",
        "BOARD_POWER_LIMIT": "0x12c",
        "TDC_LIMIT_MAX": "0xc8",
    }
    assert environment["limits"] == {"board_power_limit": "fixture-limit"}
    assert environment["smbus_telem_limits"] == {
        "BOARD_POWER_LIMIT": "0x12c",
        "TDC_LIMIT_MAX": "0xc8",
    }
    serialized = json.dumps(environment)
    assert "host_info" not in serialized
    assert "processes" not in serialized
