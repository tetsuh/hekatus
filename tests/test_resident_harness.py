"""Board-free acceptance tests for the resident producer/consumer harness."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from enodia.tt.bench import run_resident
from enodia.tt.bench.resident_harness import (
    CURRENT_WRAP_OBSERVED_WORK_MAX_TICKS,
    CURRENT_WRAP_WORK_PER_FRAME,
    PROVENANCE_PHASE_POST_RUN,
    PROVENANCE_PHASE_PREFLIGHT,
    REQUIRED_PROVENANCE_FIELDS,
    RESIDENT_SEMAPHORE_COUNT,
    SEMAPHORE_BYTES,
    TIMESTAMP_GAP_LIMIT_TICKS,
    UINT32_MAX,
    WORK_TICKS_PER_UNIT_UPPER_BOUND,
    ResidentConfig,
    ResidentPreflightError,
    RingAccounting,
    build_measurement_record,
    cycle_budget_exceeded,
    failure_name,
    frame_interval_statistics,
    interval_ticks_for_microseconds,
    periodic_gap_decomposition,
    required_samples_for_percentile,
    resident_l1_allocation_bytes,
    resident_l1_allocation_table,
    run_budget_breakdown,
    run_budget_exceeded,
    select_failure_check,
    split_u64,
    startup_allowance_ticks,
    termination_reason,
    ticks_to_seconds,
    timestamp_digest,
    validate_configuration,
    validate_pinned_environment,
    validate_post_run_provenance,
    validate_preflight_provenance,
    validate_record_inputs,
    validate_run_budget_fits_outer_cap,
    wrap_delta,
)
from enodia.tt.bench.run_resident import (
    _config_from_args,
    _is_consumer_fixed_work_budget,
    _parser,
    _resolve_watcher_mode,
    _runtime_u32,
)

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "docs/measurements/README.md"


def _continued_line(line: str) -> bool:
    trimmed = line.rstrip()
    trailing_backslashes = len(trimmed) - len(trimmed.rstrip("\\"))
    return trailing_backslashes % 2 == 1


def _shell_tokens(command: str) -> list[str]:
    command = re.sub(r"\\[ \t]*\r?\n", " ", command).strip()
    command = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", command)
    command = re.sub(r"\s*```$", "", command).strip()
    command = re.sub(r"^\s*[-+*]\s+", "", command)
    if command.startswith("`") and command.endswith("`"):
        command = command[1:-1]
    return shlex.split(command, comments=True, posix=True)


_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _shell_environment(tokens: list[str]) -> tuple[dict[str, str], set[str]]:
    """Return environment assignments and explicit unsets before a shell command."""
    assignments: dict[str, str] = {}
    unset: set[str] = set()
    index = 0
    if tokens and tokens[0] == "env":
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in {"-u", "--unset"}:
                if index + 1 >= len(tokens):
                    raise AssertionError(f"env unset option has no variable: {tokens!r}")
                unset.add(tokens[index + 1])
                index += 2
                continue
            if token.startswith("--unset="):
                unset.add(token.split("=", 1)[1])
                index += 1
                continue
            if token == "--":
                index += 1
                break
            match = _ENV_ASSIGNMENT.fullmatch(token)
            if match is None:
                break
            name, value = token.split("=", 1)
            assignments[name] = value
            index += 1
    while index < len(tokens):
        match = _ENV_ASSIGNMENT.fullmatch(tokens[index])
        if match is None:
            break
        name, value = tokens[index].split("=", 1)
        assignments[name] = value
        index += 1
    return assignments, unset


def _extract_resident_command_records(text: str) -> list[dict[str, object]]:
    """Extract resident commands with shell environment metadata from README text."""
    lines = text.splitlines()
    records: list[dict[str, object]] = []
    seen_spans: set[tuple[int, int]] = set()
    for index, line in enumerate(lines):
        if "run_resident.py" not in line:
            continue
        start = index
        while start > 0 and _continued_line(lines[start - 1]):
            start -= 1
        end = index
        while end + 1 < len(lines) and _continued_line(lines[end]):
            end += 1
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        command = "\n".join(lines[start : end + 1])
        tokens = _shell_tokens(command)
        runner_indexes = [
            token_index
            for token_index, token in enumerate(tokens)
            if Path(token.strip("`")).name == "run_resident.py"
        ]
        wrapper_indexes = [
            token_index
            for token_index, token in enumerate(tokens)
            if Path(token.strip("`")).name == "run_in_container.sh"
        ]
        if wrapper_indexes:
            wrapper_index = wrapper_indexes[0]
            try:
                argument_separator = tokens.index("--", wrapper_index + 1)
            except ValueError as exc:
                raise AssertionError(
                    f"resident wrapper invocation has no runner argument separator: {tokens!r}"
                ) from exc
            runner_start = argument_separator + 1
        elif runner_indexes:
            runner_start = runner_indexes[0] + 1
        else:
            raise AssertionError(f"could not locate resident runner in command: {tokens!r}")
        command_end = next(
            (
                token_index
                for token_index in range(runner_start, len(tokens))
                if tokens[token_index] in {";", "&&", "||", "|", "&"}
            ),
            len(tokens),
        )
        records.append(
            {
                "command": command,
                "tokens": tokens,
                "argv": tokens[runner_start:command_end],
                "environment": _shell_environment(tokens),
            }
        )
    return records


def _extract_resident_invocations(text: str) -> list[list[str]]:
    """Extract runner argv vectors from shell commands containing run_resident.py."""
    return [record["argv"] for record in _extract_resident_command_records(text)]


def _parse_documented_resident_args(argv: list[str]):
    parser = _parser()
    args = parser.parse_args(argv)
    required_options = {
        option
        for action in parser._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    for option in required_options:
        assert any(argument == option or argument.startswith(f"{option}=") for argument in argv)
    out_options = [
        argument for argument in argv if argument == "--out" or argument.startswith("--out=")
    ]
    assert len(out_options) == 1
    out = args.out
    assert out.is_absolute()
    assert out.parts[:2] == ("/", "out")
    relative_out = out.relative_to(Path("/out"))
    assert relative_out.parts and ".." not in relative_out.parts
    assert out.suffix == ".json"
    if args.raw_timestamps_out is not None:
        raw_timestamps = args.raw_timestamps_out
        assert raw_timestamps.is_absolute()
        assert raw_timestamps.parts[:2] == ("/", "out")
        assert raw_timestamps.relative_to(Path("/out")).parts
        assert raw_timestamps.suffix == ".bin"
    return args


def _readme_subsections(text: str) -> dict[str, str]:
    headings = list(re.finditer(r"^### (?P<title>.+?)\s*$", text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections[heading.group("title").strip()] = text[start:end]
    return sections


def _readme_shell_blocks(section: str) -> list[str]:
    return re.findall(r"```(?:bash|sh|shell)\s*\r?\n(.*?)```", section, flags=re.DOTALL)


def _environment() -> dict:
    return {
        "board": {"serial": "redacted-board-serial", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kmd_version": "2.11.0",
        "image": "registry.example/tt@sha256:" + "a" * 64,
        "image_pinned": True,
        "harness_commit": "0123456789abcdef",
        "tt_env_active_release": "0.75.0",
    }


def _config(**overrides) -> ResidentConfig:
    values = {
        "frame_count": 100,
        "frame_interval_ticks": 1_000,
        "producer_core": (0, 0),
        "consumer_core": (1, 0),
        "ring_pages": 4,
        "work_per_frame": 64,
        "designated_timestamp_core": (1, 0),
        "cycle_budget": 10_000,
        "outer_timeout_seconds": 60,
        "fixed_work_ticks_per_frame": 100,
    }
    values.update(overrides)
    return ResidentConfig(**values)


def test_ring_accounting_counts_full_and_empty_transitions():
    ring = RingAccounting(capacity=2)

    assert ring.reserve_producer() == 0
    ring.publish()
    assert ring.reserve_producer() == 1
    ring.publish()
    assert ring.reserve_producer() is None
    assert ring.producer_full_count == 1
    assert ring.attempted == 3
    assert ring.produced == 2
    assert ring.dropped == 1

    assert ring.consume() == 0
    assert ring.consume() == 1
    assert ring.consume() is None
    assert ring.consumer_empty_count == 1

    assert ring.reserve_producer() == 0
    ring.publish()
    assert ring.consume() == 0


def test_wrap_delta_is_unsigned_and_wrap_safe():
    assert wrap_delta(0x00000005, 0xFFFFFFFE, bits=32) == 7
    assert wrap_delta(0x0000000000000005, 0xFFFFFFFFFFFFFFFE, bits=64) == 7
    with pytest.raises(ValueError):
        wrap_delta(1, 0, bits=31)


def test_failure_codes_and_precedence_are_explicit():
    assert failure_name(0) == "none"
    assert failure_name(2) == "producer_pacing_wait"
    producer = {
        "code": 2,
        "name": "producer_pacing_wait",
        "source": "producer",
        "elapsed_ticks": 12,
        "limit_ticks": 10,
    }
    consumer = {
        "code": 4,
        "name": "consumer_fixed_work_budget",
        "source": "consumer",
        "elapsed_ticks": 100,
        "limit_ticks": 90,
    }
    assert select_failure_check(producer, consumer)["name"] == "consumer_fixed_work_budget"
    assert select_failure_check()["name"] == "none"
    with pytest.raises(ValueError):
        failure_name(99)


def test_cycle_budget_hit_only_tracks_consumer_fixed_work_failure():
    assert _is_consumer_fixed_work_budget(
        {"code": 4, "name": "consumer_fixed_work_budget"}
    ) is True
    assert _is_consumer_fixed_work_budget({"code": 4, "name": "other_check"}) is False
    assert _is_consumer_fixed_work_budget({"code": 1, "name": "run_wide_budget"}) is False


def test_cycle_budget_and_termination_are_explicit():
    assert not cycle_budget_exceeded(elapsed_ticks=99, cycle_budget=100)
    assert cycle_budget_exceeded(elapsed_ticks=100, cycle_budget=100)
    assert termination_reason(frame_count_reached=True, cycle_budget_hit=False, outer_timeout=False) == (
        "frame_count"
    )
    assert termination_reason(frame_count_reached=False, cycle_budget_hit=True, outer_timeout=False) == (
        "cycle_budget"
    )
    assert termination_reason(frame_count_reached=False, cycle_budget_hit=False, outer_timeout=True) == (
        "outer_timeout"
    )


def test_run_budget_covers_n_frames_interval_work_and_margin():
    config = _config(
        frame_count=2_001,
        frame_interval_ticks=1_350_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
    )
    breakdown = run_budget_breakdown(config)
    assert breakdown == {
        "frame_count": 2_001,
        "frame_interval_ticks": 1_350_000,
        "pacing_ticks": 2_701_350_000,
        "per_frame_work_budget_ticks": 100_000,
        "cycle_budget_ticks_per_frame": 10_000_000,
        "fixed_work_ticks": 200_100_000,
        "critical_path_ticks": 2_701_350_000,
        "schedule_ticks": 2_836_350_000,
        "overlap_model": "producer pacing and consumer fixed work overlap; critical path is max",
        "startup_allowance_ms": 100,
        "budget_aiclk_mhz": 1_350,
        "startup_allowance_ticks": 135_000_000,
        "safety_margin_percent": 10,
        "watcher_overhead_margin_percent": 0,
        "total_margin_percent": 10,
        "safety_margin_ticks": 283_635_000,
        "run_budget_ticks": 3_119_985_000,
    }
    assert not run_budget_exceeded(
        elapsed_ticks=4 * config.frame_interval_ticks + 4 * config.cycle_budget,
        run_budget_ticks=breakdown["run_budget_ticks"],
    )
    assert interval_ticks_for_microseconds(microseconds=1_000, aiclk_mhz=800) == 800_000
    assert interval_ticks_for_microseconds(microseconds=1_000, aiclk_mhz=1_350) == 1_350_000
    assert split_u64(breakdown["run_budget_ticks"]) == (3_119_985_000, 0)
    watcher_breakdown = run_budget_breakdown(config, watcher=True)
    assert watcher_breakdown["watcher_overhead_margin_percent"] == 100
    assert watcher_breakdown["total_margin_percent"] == 110
    assert watcher_breakdown["run_budget_ticks"] == 5_956_335_000


def test_wrap_tracked_low_word_extension_is_monotonic_across_wrap():
    extended = 0xFFFFFFFE
    values = []
    for low in (0xFFFFFFFF, 0x00000000, 0x00000001, 0x00000010):
        extended += (low - (extended & 0xFFFFFFFF)) & 0xFFFFFFFF
        values.append(extended)
    assert values == [0xFFFFFFFF, 0x1_0000_0000, 0x1_0000_0001, 0x1_0000_0010]
    assert values == sorted(values)


def test_ticks_to_seconds_uses_aiclk_hz_conversion():
    assert ticks_to_seconds(ticks=1_350_000, aiclk_mhz=1_350) == pytest.approx(0.001)
    assert ticks_to_seconds(ticks=80_000_000, aiclk_mhz=800) == pytest.approx(0.1)


def test_periodic_gap_decomposition_preserves_quotients_and_remainders():
    result = periodic_gap_decomposition([6_363, 3_182, 6_363], period_frames=6_363)
    assert result["entries"] == [
        {"gap_frames": 6_363, "quotient": 1, "remainder": 0},
        {"gap_frames": 3_182, "quotient": 0, "remainder": 3_182},
        {"gap_frames": 6_363, "quotient": 1, "remainder": 0},
    ]
    assert result["aggregate_by_quotient_remainder"] == {"1,0": 2, "0,3182": 1}


def test_startup_allowance_is_100_ms_at_both_supported_clocks():
    assert startup_allowance_ticks(aiclk_mhz=800) == 80_000_000
    assert startup_allowance_ticks(aiclk_mhz=1_350) == 135_000_000


def test_run_budget_uses_64_bit_overflow_checks_and_scopes_errors():
    config = _config(
        frame_count=2_001,
        frame_interval_ticks=800_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
        budget_aiclk_mhz=800,
    )
    breakdown = run_budget_breakdown(config)
    assert breakdown["pacing_ticks"] == 1_600_800_000
    assert breakdown["run_budget_ticks"] == 1_848_880_000
    assert run_budget_exceeded(
        elapsed_ticks=breakdown["run_budget_ticks"],
        run_budget_ticks=breakdown["run_budget_ticks"],
    )
    with pytest.raises(ValueError):
        split_u64(1 << 64)
    with pytest.raises(ValueError):
        interval_ticks_for_microseconds(microseconds=0, aiclk_mhz=1_350)
    huge = _config(frame_count=UINT32_MAX, frame_interval_ticks=TIMESTAMP_GAP_LIMIT_TICKS - 1)
    assert run_budget_breakdown(huge)["run_budget_ticks"] <= (1 << 64) - 1


def test_outer_cap_rejects_60000_frames_and_reports_safe_alternative():
    config = _config(
        frame_count=60_000,
        frame_interval_ticks=1_350_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
        outer_timeout_seconds=60,
        budget_aiclk_mhz=1_350,
    )
    with pytest.raises(ResidentPreflightError, match="outer cap"):
        validate_run_budget_fits_outer_cap(config, watcher=False)
    safe = _config(
        frame_count=54_445,
        frame_interval_ticks=1_350_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
        outer_timeout_seconds=60,
        budget_aiclk_mhz=1_350,
    )
    breakdown = validate_run_budget_fits_outer_cap(safe, watcher=False)
    assert breakdown["run_budget_ticks"] == 80_999_325_000
    longest = _config(
        frame_count=600_000,
        frame_interval_ticks=1_350_000,
        cycle_budget=10_000_000,
        fixed_work_ticks_per_frame=100_000,
        outer_timeout_seconds=600,
        budget_aiclk_mhz=1_350,
    )
    with pytest.raises(ResidentPreflightError, match="outer cap"):
        validate_run_budget_fits_outer_cap(longest, watcher=False)


def test_resident_l1_allocation_ledger_matches_formula_and_cores():
    table = resident_l1_allocation_table(ring_pages=220)
    assert sum(row["bytes"] for row in table) == 917_520
    assert resident_l1_allocation_bytes(ring_pages=220) == 917_520
    assert sum(row["bytes"] for row in table if row["scope"] == "producer_core") == 8_200
    assert sum(row["bytes"] for row in table if row["scope"] == "consumer_core") == 909_320
    assert sum(row["bytes"] for row in table if row["scope"] == "shared") == 0
    assert {row["component"] for row in table} == {
        "ring_storage",
        "control_page",
        "consumer_cb_page",
        "ready_done_semaphores",
        "producer_anchor_page",
        "producer_cb_page",
        "free_error_semaphores",
        "timestamps_DRAM",
        "producer_consumer_stats_DRAM",
        "watcher_extra_resident_allocation",
    }
    assert resident_l1_allocation_bytes(ring_pages=220, watcher=True) == 917_520
    assert validate_configuration(_config(ring_pages=220))
    with pytest.raises(ResidentPreflightError, match="L1 preflight budget"):
        validate_configuration(_config(ring_pages=221))


def test_resident_semaphore_l1_bytes_are_in_preflight_accounting():
    source = Path("enodia/tt/bench/resident_harness.py").read_text()
    assert RESIDENT_SEMAPHORE_COUNT == 4
    assert SEMAPHORE_BYTES == 4
    assert "ready_done_semaphores" in source
    assert "free_error_semaphores" in source


def test_wrap_tracked_single_gap_bounds_and_work_formula():
    assert CURRENT_WRAP_WORK_PER_FRAME == 64
    assert CURRENT_WRAP_OBSERVED_WORK_MAX_TICKS == 1_087
    assert WORK_TICKS_PER_UNIT_UPPER_BOUND == 19
    assert validate_configuration(
        _config(
            frame_interval_ticks=TIMESTAMP_GAP_LIMIT_TICKS - 1,
            cycle_budget=TIMESTAMP_GAP_LIMIT_TICKS - 1,
            work_per_frame=(TIMESTAMP_GAP_LIMIT_TICKS - 1) // WORK_TICKS_PER_UNIT_UPPER_BOUND,
            fixed_work_ticks_per_frame=100_000,
        )
    )
    with pytest.raises(ResidentPreflightError, match="frame_interval_ticks"):
        validate_configuration(_config(frame_interval_ticks=TIMESTAMP_GAP_LIMIT_TICKS))
    with pytest.raises(ResidentPreflightError, match="cycle_budget"):
        validate_configuration(_config(cycle_budget=TIMESTAMP_GAP_LIMIT_TICKS))
    with pytest.raises(ResidentPreflightError, match="work_per_frame"):
        validate_configuration(
            _config(
                frame_interval_ticks=TIMESTAMP_GAP_LIMIT_TICKS - 1,
                work_per_frame=(TIMESTAMP_GAP_LIMIT_TICKS - 1) // WORK_TICKS_PER_UNIT_UPPER_BOUND + 1,
                fixed_work_ticks_per_frame=100_000,
            )
        )


@pytest.mark.parametrize("aiclk_mhz", [800, 1_350])
def test_600_second_timing_cap_maximum_safe_frame_boundary(aiclk_mhz):
    interval_ticks = aiclk_mhz * 1_000
    safe = _config(
        frame_count=545_354,
        frame_interval_ticks=interval_ticks,
        outer_timeout_seconds=600,
        budget_aiclk_mhz=aiclk_mhz,
    )
    breakdown = validate_run_budget_fits_outer_cap(safe)
    cap_ticks = 600 * aiclk_mhz * 1_000_000
    assert breakdown["run_budget_ticks"] <= cap_ticks
    expected = 479_999_520_000 if aiclk_mhz == 800 else 809_999_190_000
    assert breakdown["run_budget_ticks"] == expected

    rejected = _config(
        frame_count=545_355,
        frame_interval_ticks=interval_ticks,
        outer_timeout_seconds=600,
        budget_aiclk_mhz=aiclk_mhz,
    )
    with pytest.raises(ResidentPreflightError, match="outer cap"):
        validate_run_budget_fits_outer_cap(rejected)


def test_runtime_uint32_bounds_cover_kernel_arguments():
    assert validate_configuration(
        _config(frame_interval_ticks=TIMESTAMP_GAP_LIMIT_TICKS - 1)
    ).frame_interval_ticks == TIMESTAMP_GAP_LIMIT_TICKS - 1
    invalid_interval = _config(frame_interval_ticks=TIMESTAMP_GAP_LIMIT_TICKS)
    with pytest.raises(ResidentPreflightError, match="frame_interval_ticks.*wrap-tracked"):
        validate_configuration(invalid_interval)

    invalid_configs = {
        field: _config(**{field: UINT32_MAX + 1})
        for field in ("frame_count", "ring_pages", "work_per_frame", "cycle_budget")
    }
    for field, invalid_config in invalid_configs.items():
        with pytest.raises(ResidentPreflightError, match=f"{field}.*uint32"):
            validate_configuration(invalid_config)


def test_watcher_mode_requires_matching_cli_and_environment():
    assert _resolve_watcher_mode(False, None) is False
    assert _resolve_watcher_mode(True, "1") is True
    with pytest.raises(ResidentPreflightError, match="same mode"):
        _resolve_watcher_mode(True, None)
    with pytest.raises(ResidentPreflightError, match="same mode"):
        _resolve_watcher_mode(False, "1")
    with pytest.raises(ResidentPreflightError, match="unset or exactly 1"):
        _resolve_watcher_mode(False, "0")
    with pytest.raises(ResidentPreflightError, match="unset or exactly 1"):
        _resolve_watcher_mode(False, "")


def test_resident_parser_rejects_abbreviated_watcher_flag():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--out", "record.json", "--wat"])


class TestResidentDefaults:
    DEFAULTS: ClassVar[dict[str, object]] = {
        "frame_count": 100,
        "frame_interval_ticks": 1_350_000,
        "producer_core": (0, 0),
        "consumer_core": (1, 0),
        "ring_pages": 4,
        "work_per_frame": 64,
        "cycle_budget": 10_000_000,
        "fixed_work_ticks_per_frame": 100_000,
        "budget_aiclk_mhz": 1_350,
        "outer_timeout_seconds": 60,
        "histogram_bin_ticks": 1,
        "device_id": 0,
        "watcher": False,
    }

    def test_parser_defaults_are_explicit_and_host_independent(self):
        args = _parser().parse_args(["--out", "resident.json"])
        for name, expected in self.DEFAULTS.items():
            assert getattr(args, name) == expected
        assert args.env_json is None
        assert args.power_trace is None
        assert args.raw_timestamps_out is None
        config = _config_from_args(args)
        assert config.designated_timestamp_core == (1, 0)

    @pytest.mark.parametrize(
        ("runner_args", "watcher", "valid", "reason"),
        [
            pytest.param([], False, True, None, id="default-timing"),
            pytest.param(["--watcher"], True, True, None, id="default-watcher"),
            pytest.param(
                ["--outer-timeout-seconds", "600"], False, True, None, id="timing-cap"
            ),
            pytest.param(
                ["--watcher", "--outer-timeout-seconds", "600"],
                True,
                False,
                "60s Stage 1 Watcher cap",
                id="watcher-cap",
            ),
            pytest.param(
                [
                    "--frame-count",
                    "545355",
                    "--frame-interval-ticks",
                    "1350000",
                    "--outer-timeout-seconds",
                    "600",
                ],
                False,
                False,
                "run budget exceeds outer cap",
                id="timing-budget",
            ),
        ],
    )
    def test_default_combinations_are_preflighted_without_device(
        self, runner_args, watcher, valid, reason
    ):
        args = _parser().parse_args(["--out", "resident.json", *runner_args])
        config = _config_from_args(args)
        if valid:
            assert validate_run_budget_fits_outer_cap(
                validate_configuration(config, watcher=watcher), watcher=watcher
            )
        else:
            with pytest.raises(ResidentPreflightError, match=reason):
                validate_run_budget_fits_outer_cap(config, watcher=watcher)


class TestIssue12Runbook:
    RUNBOOK_PROCEDURES: ClassVar[tuple[dict[str, object], ...]] = (
        {
            "name": "cleanup-gate",
            "heading": "Cleanup gate",
            "values": (("Before and after each invocation", "docker ps --format '{{.Names}}'"),),
        },
        {
            "name": "watcher-validation",
            "heading": "Watcher validation",
            "values": (
                ("Watcher-enabled", "TT_METAL_WATCHER=1"),
                ("one-frame", "--frame-count 1"),
                ("device 0", "--device-id 0"),
                ("1 ms", "--frame-interval-ticks 1350000"),
                ("60-second", "--outer-timeout-seconds 60"),
                ("run_resident.py", "run_resident.py"),
                (
                    "out/bench/issue12-watcher",
                    "run_in_container.sh out/bench/issue12-watcher --",
                ),
                ("/out/runner-result.json", "--out /out/runner-result.json"),
            ),
        },
        {
            "name": "no-watcher-timing",
            "heading": "No-Watcher timing",
            "values": (
                ("no-Watcher", "env -u TT_METAL_WATCHER"),
                ("500,000 frames", "--frame-count 500000"),
                ("device 0", "--device-id 0"),
                ("1 ms", "--frame-interval-ticks 1350000"),
                ("1,350 MHz", "--budget-aiclk-mhz 1350"),
                ("600-second", "--outer-timeout-seconds 600"),
                ("run_resident.py", "run_resident.py"),
                (
                    "out/bench/issue12-timing",
                    "run_in_container.sh out/bench/issue12-timing --",
                ),
                ("/out/runner-result.json", "--out /out/runner-result.json"),
                (
                    "/out/issue12-timing-500000.bin",
                    "--raw-timestamps-out /out/issue12-timing-500000.bin",
                ),
            ),
        },
        {
            "name": "record-retention",
            "heading": "Record creation and output retention (host-side shell commands)",
            "values": (
                (
                    "out/bench/issue12-watcher/runner-result.json",
                    "out/bench/issue12-watcher",
                ),
                (
                    "out/bench/issue12-timing/runner-result.json",
                    "out/bench/issue12-timing",
                ),
                (
                    "out/bench/issue12-timing/issue12-timing-500000.bin",
                    "issue12-timing-500000.bin",
                ),
                ("issue12-retained/", "issue12-retained/"),
                (
                    "Retain the",
                    "cp -a out/bench/issue12-watcher out/bench/issue12-timing",
                ),
                ("environment and power provenance", "'env-*.json'"),
                ("before any analysis", "sha256sum"),
            ),
        },
        {
            "name": "abnormal-recovery",
            "heading": "Abnormal exit or timeout: one-reset recovery",
            "values": (
                ("at most one reset", "tt-smi -r /dev/tenstorrent/0"),
                ("device 0", "--device-id 0"),
                ("fixed-image", "HEKATUS_TT_IMAGE="),
                ("Stage-1", "--stage 1"),
                ("Watcher mode", "TT_METAL_WATCHER=1"),
                ("60-second", "HEKATUS_TT_CONTAINER_TIMEOUT_S=60"),
                ("health probe", "tools/newton_schulz_bringup.py"),
            ),
        },
    )

    def test_expected_runbook_procedures_have_commands_and_explicit_values(self):
        text = README_PATH.read_text()
        sections = _readme_subsections(text)
        assert self.RUNBOOK_PROCEDURES
        for procedure in self.RUNBOOK_PROCEDURES:
            name = str(procedure["name"])
            heading = str(procedure["heading"])
            section = sections.get(heading)
            assert section is not None, f"runbook procedure {name} is missing"
            blocks = _readme_shell_blocks(section)
            assert blocks, f"runbook procedure {name} has no shell command"
            command_text = "\n".join(blocks)
            assert any(
                line.strip() and not line.lstrip().startswith("#")
                for block in blocks
                for line in block.splitlines()
            ), f"runbook procedure {name} has no executable command"
            for prose_value, command_value in procedure["values"]:
                assert prose_value in section, (
                    f"runbook procedure {name} no longer states {prose_value!r}"
                )
                assert command_value in command_text, (
                    f"runbook procedure {name} does not pass {command_value!r} explicitly"
                )

    def test_every_readme_resident_invocation_reaches_preflight_without_device(self):
        records = _extract_resident_command_records(README_PATH.read_text())
        assert records, "the measurement runbook must retain a resident invocation"
        for record in records:
            argv = record["argv"]
            args = _parse_documented_resident_args(argv)
            config = _config_from_args(args)
            assignments, unset = record["environment"]
            environment_value = assignments.get("TT_METAL_WATCHER")
            assert _resolve_watcher_mode(args.watcher, environment_value) == args.watcher
            if args.watcher:
                assert environment_value == "1"
                assert "TT_METAL_WATCHER" not in unset
            else:
                assert environment_value is None
                assert "TT_METAL_WATCHER" in unset
            assert validate_run_budget_fits_outer_cap(
                validate_configuration(config, watcher=args.watcher), watcher=args.watcher
            )

    def test_resident_invocations_use_separate_wrapper_output_directories(self):
        records = _extract_resident_command_records(README_PATH.read_text())
        assert len(records) == 2
        output_directories = []
        for record in records:
            tokens = record["tokens"]
            wrapper_index = next(
                index
                for index, token in enumerate(tokens)
                if Path(token.strip("`")).name == "run_in_container.sh"
            )
            separator = tokens.index("--", wrapper_index + 1)
            assert separator == wrapper_index + 2
            output_directory = Path(tokens[wrapper_index + 1])
            args = _parse_documented_resident_args(record["argv"])
            expected_directory = Path(
                "out/bench/issue12-watcher"
                if args.watcher
                else "out/bench/issue12-timing"
            )
            assert output_directory == expected_directory
            assert args.out == Path("/out/runner-result.json")
            output_directories.append(output_directory)
        assert len(set(output_directories)) == len(output_directories)

    def test_timing_invocation_is_explicitly_no_watcher(self):
        records = _extract_resident_command_records(README_PATH.read_text())
        timing_records = [
            record
            for record in records
            if "--frame-count" in record["argv"] and "500000" in record["argv"]
        ]
        assert len(timing_records) == 1
        timing = timing_records[0]
        assignments, unset = timing["environment"]
        assert "TT_METAL_WATCHER" in unset
        assert "TT_METAL_WATCHER" not in assignments
        assert "--watcher" not in timing["argv"]


def test_resident_invocation_extractor_handles_continuations_and_quoted_paths():
    command = """TT_METAL_WATCHER=1 HEKATUS_TT_RUNNER=enodia/tt/bench/run_resident.py \\
  ./enodia/tt/bench/run_in_container.sh -- \\
  --out '/out/resident result.json' --frame-count 1 --watcher"""
    invocations = _extract_resident_invocations(command)
    assert invocations == [
        ["--out", "/out/resident result.json", "--frame-count", "1", "--watcher"]
    ]
    _parse_documented_resident_args(invocations[0])


def test_runtime_addresses_are_checked_at_uint32_boundary():
    assert _runtime_u32(UINT32_MAX, "address") == UINT32_MAX
    with pytest.raises(ResidentPreflightError, match="address.*uint32"):
        _runtime_u32(UINT32_MAX + 1, "address")


def test_ring_drop_policy_drains_without_producer_wait():
    ring = RingAccounting(capacity=2)
    assert ring.reserve_producer() == 0
    ring.publish()
    assert ring.reserve_producer() == 1
    ring.publish()
    assert ring.reserve_producer() is None
    assert ring.drain_complete(producer_done=False) is False
    assert ring.consume() == 0
    assert ring.consume() == 1
    assert ring.drain_complete(producer_done=True) is True


def test_fixed_work_preflight_uses_half_interval_boundary():
    assert validate_configuration(
        _config(frame_interval_ticks=800_000, fixed_work_ticks_per_frame=400_000)
    )
    assert validate_configuration(
        _config(frame_interval_ticks=1_350_000, fixed_work_ticks_per_frame=675_000)
    )
    too_much_work_at_800 = _config(frame_interval_ticks=800_000, fixed_work_ticks_per_frame=400_001)
    with pytest.raises(ValueError, match="half the frame interval"):
        validate_configuration(too_much_work_at_800)
    too_much_work_at_1350 = _config(frame_interval_ticks=1_350_000, fixed_work_ticks_per_frame=675_001)
    with pytest.raises(ValueError, match="half the frame interval"):
        validate_configuration(too_much_work_at_1350)


def test_configuration_rejects_outer_cap_and_core_clock_mismatch():
    overlong_timing = _config(outer_timeout_seconds=601)
    with pytest.raises(ValueError, match="600"):
        validate_configuration(overlong_timing)
    assert validate_configuration(_config(outer_timeout_seconds=600))
    overlong_watcher = _config(outer_timeout_seconds=61)
    with pytest.raises(ValueError, match="60"):
        validate_configuration(overlong_watcher, watcher=True)
    assert validate_configuration(_config(outer_timeout_seconds=60), watcher=True)
    mismatched_timestamp_core = _config(designated_timestamp_core=(0, 0))
    with pytest.raises(ValueError, match="designated_timestamp_core"):
        validate_configuration(mismatched_timestamp_core)
    same_cores = _config(consumer_core=(0, 0))
    with pytest.raises(ValueError, match="different cores"):
        validate_configuration(same_cores)


def test_histogram_percentiles_have_exact_sample_thresholds():
    assert required_samples_for_percentile(0.50) == 40
    assert required_samples_for_percentile(0.99) == 2_000
    assert required_samples_for_percentile(0.999) == 20_000
    assert required_samples_for_percentile(0.9999) == 200_000

    short = frame_interval_statistics([10, 20, 30], bin_width_ticks=1)
    assert short["N"] == 3
    assert short["min_ticks"] == 10
    assert short["max_ticks"] == 30
    for name in ("p50", "p99", "p99_9", "p99_99"):
        assert short["percentiles"][name]["status"] == "insufficient"
        assert "value_ticks" not in short["percentiles"][name]

    enough = frame_interval_statistics(list(range(40)), bin_width_ticks=5)
    assert enough["percentiles"]["p50"]["status"] == "ok"
    assert enough["percentiles"]["p99"]["status"] == "insufficient"
    assert enough["histogram"]["bin_width_ticks"] == 5
    assert sum(item["count"] for item in enough["histogram"]["bins"]) == 40


def test_timestamp_digest_redacts_raw_values():
    timestamps = [0x100000000, 0x100000007, 0x10000000F]
    redacted = timestamp_digest(timestamps)
    expected = b"".join(value.to_bytes(8, "little") for value in timestamps)
    assert redacted == {"count": 3, "sha256": hashlib.sha256(expected).hexdigest()}
    assert "timestamps" not in redacted


def _failure(code: int, name: str) -> dict:
    return {
        "code": code,
        "name": name,
        "source": "producer" if name.startswith("producer") or name == "other_check" else "consumer",
        "elapsed_ticks": 123,
        "limit_ticks": 100,
        "unit": "device_clock_ticks",
    }


def test_pacing_failure_accepts_one_aborted_attempt():
    record = build_measurement_record(
        config=_config(frame_count=100),
        aiclk_mhz=1_350,
        timestamps=[1_000],
        producer_full_count=0,
        consumer_empty_count=1,
        cycle_budget_hit=True,
        kernel_error_flag=1,
        attempted_frame_count=2,
        produced_frame_count=1,
        dropped_frame_count=0,
        aborted_attempts=1,
        failure_check=_failure(2, "producer_pacing_wait"),
        harness_commit="0123456789abcdef",
        environment=_environment(),
        power_trace="resident-power.csv",
    )
    assert record["status"] == "error"
    assert record["parameters"]["aborted_attempts"] == 1
    assert record["ring"]["aborted_attempts"] == 1


@pytest.mark.parametrize(
    ("code", "name"),
    [
        (1, "run_wide_budget"),
        (3, "consumer_empty_wait"),
        (4, "consumer_fixed_work_budget"),
        (5, "other_check"),
    ],
)
def test_error_paths_accept_zero_aborted_attempts(code, name):
    record = build_measurement_record(
        config=_config(frame_count=3),
        aiclk_mhz=1_350,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=0,
        consumer_empty_count=1,
        cycle_budget_hit=True,
        kernel_error_flag=1,
        attempted_frame_count=3,
        produced_frame_count=3,
        dropped_frame_count=0,
        aborted_attempts=0,
        failure_check=_failure(code, name),
        harness_commit="0123456789abcdef",
        environment=_environment(),
        power_trace="resident-power.csv",
    )
    assert record["status"] == "error"
    assert record["parameters"]["aborted_attempts"] == 0


def test_aborted_attempts_reject_invalid_normal_and_error_relations():
    base = {
        "config": _config(frame_count=3),
        "aiclk_mhz": 1_350,
        "timestamps": [1_000, 2_000, 3_000],
        "producer_full_count": 0,
        "consumer_empty_count": 0,
        "cycle_budget_hit": False,
        "kernel_error_flag": 0,
        "harness_commit": "0123456789abcdef",
        "environment": _environment(),
        "power_trace": "resident-power.csv",
    }
    with pytest.raises(ValueError, match="normal counter relation"):
        build_measurement_record(**base, aborted_attempts=1)
    with pytest.raises(ValueError, match="producer_full_count must equal"):
        build_measurement_record(**{**base, "producer_full_count": 1}, aborted_attempts=0)
    error_base = {
        **base,
        "cycle_budget_hit": True,
        "kernel_error_flag": 1,
        "failure_check": _failure(2, "producer_pacing_wait"),
        "timestamps": [1_000],
        "attempted_frame_count": 2,
        "produced_frame_count": 1,
        "dropped_frame_count": 0,
    }
    for invalid in (-1, 2, 1.5, "1"):
        with pytest.raises(ValueError, match="aborted_attempts"):
            build_measurement_record(**error_base, aborted_attempts=invalid)
    with pytest.raises(ValueError, match="error counter relation"):
        build_measurement_record(**error_base, aborted_attempts=0)


def test_watcher_timing_evidence_is_always_false():
    kwargs = {
        "config": _config(frame_count=3),
        "aiclk_mhz": 1_350,
        "timestamps": [1_000, 2_000, 3_000],
        "producer_full_count": 0,
        "consumer_empty_count": 0,
        "cycle_budget_hit": False,
        "kernel_error_flag": 0,
        "harness_commit": "0123456789abcdef",
        "environment": _environment(),
        "power_trace": "resident-power.csv",
        "watcher": True,
    }
    assert build_measurement_record(**kwargs)["timing_evidence"] is False
    assert build_measurement_record(**kwargs, timing_evidence=True)["timing_evidence"] is False


def test_existing_issue12_measurement_counters_obey_protocol():
    checked = 0
    for path in sorted(Path("docs/measurements").glob("*issue12-stage1*.json")):
        record = json.loads(path.read_text())
        ring = record.get("ring")
        counts = record.get("counts")
        if ring and ring.get("attempted_frame_count") is not None:
            source = ring
            attempted_key = "attempted_frame_count"
            produced_key = "produced_frame_count"
            consumed_key = "consumed_frame_count"
            dropped_key = "dropped_frame_count"
            full_key = "producer_full_count"
            overflow_key = "overflow_count"
        elif counts and counts.get("attempted") is not None:
            source = counts
            attempted_key = "attempted"
            produced_key = "produced"
            consumed_key = "consumed"
            dropped_key = "dropped"
            full_key = "producer_full"
            overflow_key = "overflow"
        else:
            assert record.get("status") in {"blocked", "blocked_timing", "rejected", "error", "abnormal_exit"}
            assert record.get("timing_evidence") is not True
            continue
        checked += 1
        attempted = source[attempted_key]
        produced = source[produced_key]
        consumed = source[consumed_key]
        dropped = source[dropped_key]
        aborted = source.get("aborted_attempts", record.get("parameters", {}).get("aborted_attempts", 0))
        assert source[full_key] == dropped
        assert source[overflow_key] == source[full_key]
        assert consumed <= produced
        ring_pages = record.get("parameters", {}).get("ring_pages", 4)
        assert produced - consumed <= ring_pages
        assert attempted == produced + dropped + aborted
        if "histogram" in record and "N" in record["histogram"]:
            assert record["histogram"]["N"] == max(0, consumed - 1)
    assert checked > 0


def test_record_schema_is_strict_and_excludes_raw_timestamps():
    config = _config()
    record = build_measurement_record(
        config=config,
        aiclk_mhz=1_350,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=97,
        consumer_empty_count=1,
        startup_ticks=123,
        startup_ticks_valid=True,
        work_min_ticks=10,
        work_max_ticks=20,
        failure_check={
            "code": 4,
            "name": "consumer_fixed_work_budget",
            "source": "consumer",
            "elapsed_ticks": 123,
            "limit_ticks": 100,
            "unit": "device_clock_ticks",
        },
        cycle_budget_hit=False,
        kernel_error_flag=0,
        harness_commit="0123456789abcdef",
        environment=_environment(),
        power_trace="resident-power.csv",
    )
    encoded = json.dumps(record, allow_nan=False)
    parsed = json.loads(encoded)
    assert parsed["schema"] == "issue-12-stage-1-resident-v1"
    assert parsed["parameters"]["frame_interval_is_not_acquisition_rate_claim"] is True
    assert parsed["clock"]["timestamp_api"] == "get_timestamp_32b"
    assert parsed["clock"]["timestamp_semantics"] == "32-bit low word, software-extended (wrap-tracked)"
    assert parsed["raw_timestamps"]["count"] == 3
    assert '"timestamps":' not in encoded
    assert parsed["histogram"]["N"] == 2
    assert parsed["ring"]["producer_full_count"] == 97
    assert parsed["ring"]["consumer_empty_count"] == 1
    assert parsed["ring"]["dropped_frame_count"] == 97
    assert parsed["startup"]["observed_ticks"] == 123
    assert parsed["startup"]["observed_valid"] is True
    assert parsed["work_ticks"] == {
        "minimum": 10,
        "maximum": 20,
        "valid": True,
        "unit": "device_clock_ticks",
    }
    assert parsed["failure_check"]["name"] == "consumer_fixed_work_budget"
    assert parsed["failure_check"]["elapsed_ticks"] == 123
    assert parsed["failure_check"]["limit_ticks"] == 100
    assert parsed["timing_evidence"] is False
    assert parsed["ring"]["full_ring_policy"] == "drop_new_frame_without_waiting_or_overwriting"
    assert parsed["parameters"]["full_ring_policy"] == "drop_new_frame_without_waiting_or_overwriting"
    assert parsed["parameters"]["attempted_frame_count"] == 100
    assert parsed["parameters"]["aborted_attempts"] == 0
    assert parsed["ring"]["aborted_attempts"] == 0
    assert parsed["environment"]["image"] == "registry.example/tt@sha256:" + "a" * 64


class TestResidentSemaphoreDecisionAudit:
    """Board-free inventory and interleaving checks for shared kernel decisions."""

    DECISION_INVENTORY: ClassVar[tuple[tuple[str, str, str, int], ...]] = (
        (
            "consumer",
            "initial-ready-requirement",
            "if (ready_count < required) {",
            0,
        ),
        (
            "consumer",
            "initial-done-drain",
            "if (*done_sem != 0) {",
            0,
        ),
        (
            "consumer",
            "wait-ready-requirement",
            "while (*ready_sem < required) {",
            0,
        ),
        (
            "consumer",
            "wait-done-drain",
            "if (*done_sem != 0) {",
            1,
        ),
        (
            "consumer",
            "after-wait-ready-requirement",
            "if (*ready_sem < required) {",
            0,
        ),
        (
            "consumer",
            "after-wait-done-drain",
            "if (*done_sem != 0) {",
            2,
        ),
        (
            "producer",
            "cancel-before-attempt",
            "if (*error_sem != 0) {",
            0,
        ),
        (
            "producer",
            "cancel-during-pacing",
            "if (*error_sem != 0) {",
            1,
        ),
        (
            "producer",
            "cancel-before-slot-check",
            "if (*error_sem != 0) {",
            2,
        ),
        (
            "producer",
            "cancel-before-free-observation",
            "if (*error_sem != 0) {",
            3,
        ),
        (
            "producer",
            "free-slot-requirement",
            "slot_full = *free_sem < consumed_required;",
            0,
        ),
    )
    PRODUCER_CANCEL_CHECKPOINTS: ClassVar[tuple[str, ...]] = (
        "cancel-before-attempt",
        "cancel-during-pacing",
        "cancel-before-slot-check",
        "cancel-before-free-observation",
    )

    @staticmethod
    def _source(kernel: str) -> str:
        return (ROOT / "enodia/tt/bench/kernels" / f"resident_{kernel}.cpp").read_text()

    @staticmethod
    def _positions(source: str, marker: str) -> list[int]:
        return [match.start() for match in re.finditer(re.escape(marker), source)]

    @classmethod
    def _block_at(cls, source: str, marker: str, occurrence: int) -> str:
        position = cls._positions(source, marker)[occurrence]
        opening = source.index("{", position)
        depth = 0
        for index in range(opening, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[position : index + 1]
        raise AssertionError(f"unclosed block for {marker!r}")

    def test_decision_inventory_is_complete(self):
        assert len(self.DECISION_INVENTORY) == 11
        labels = [entry[1] for entry in self.DECISION_INVENTORY]
        assert len(labels) == len(set(labels))

        expected = {
            (kernel, marker, occurrence)
            for kernel, _label, marker, occurrence in self.DECISION_INVENTORY
        }
        observed = set()
        for kernel in ("consumer", "producer"):
            source = self._source(kernel)
            markers = {entry[2] for entry in self.DECISION_INVENTORY if entry[0] == kernel}
            for marker in markers:
                positions = self._positions(source, marker)
                for occurrence in range(len(positions)):
                    observed.add((kernel, marker, occurrence))

        assert observed == expected
        for kernel in ("consumer", "producer"):
            source = self._source(kernel)
            expected_conditions = sorted(
                marker
                for entry_kernel, _label, marker, _occurrence in self.DECISION_INVENTORY
                if entry_kernel == kernel and marker.startswith(("if (", "while ("))
            )
            observed_conditions = sorted(
                line.strip()
                for line in source.splitlines()
                if line.strip().startswith(("if (", "while ("))
                and (
                    "ready_count <" in line
                    or any(
                        semaphore in line
                        for semaphore in ("*ready_sem", "*done_sem", "*free_sem", "*error_sem")
                    )
                )
            )
            assert observed_conditions == expected_conditions

        consumer = self._source("consumer")
        producer = self._source("producer")
        assert consumer.count("if (*done_sem != 0) {") == 3
        assert consumer.count("if (ready_count < required) {") == 1
        assert consumer.count("while (*ready_sem < required) {") == 1
        assert consumer.count("if (*ready_sem < required) {") == 1
        assert producer.count("if (*error_sem != 0) {") == 4
        assert producer.count("slot_full = *free_sem < consumed_required;") == 1

    def test_shared_observations_have_required_freshness_ordering(self):
        consumer = self._source("consumer")
        producer = self._source("producer")

        done_blocks = [
            self._block_at(consumer, "if (*done_sem != 0) {", occurrence)
            for occurrence in range(3)
        ]
        for block in done_blocks:
            done = block.index("if (*done_sem != 0) {")
            invalidate = block.index("invalidate_l1_cache();", done)
            fresh_ready = block.index(
                "const std::uint32_t fresh_ready_count = *ready_sem;", invalidate
            )
            fresh_decision = block.index(
                "if (fresh_ready_count == frames_consumed) {", fresh_ready
            )
            assert done < invalidate < fresh_ready < fresh_decision
            assert block.count("fresh_ready_count == frames_consumed") == 1
            assert "if (ready_count == frames_consumed)" not in block
            assert "*ready_sem == frames_consumed" not in block

        assert re.search(
            r"invalidate_l1_cache\(\);\s*"
            r"const std::uint32_t required = frames_consumed \+ 1;\s*"
            r"const std::uint32_t ready_count = \*ready_sem;",
            consumer,
        )
        wait_block = self._block_at(consumer, "while (*ready_sem < required) {", 0)
        assert wait_block.lstrip().startswith("while (*ready_sem < required) {")
        assert re.search(r"\{\s*invalidate_l1_cache\(\);", wait_block)
        assert re.search(
            r"invalidate_l1_cache\(\);\s*if \(\*ready_sem < required\) \{", consumer
        )

        error_checks = [
            self._block_at(producer, "if (*error_sem != 0) {", occurrence)
            for occurrence in range(4)
        ]
        assert len(
            re.findall(r"invalidate_l1_cache\(\);\s*if \(\*error_sem != 0\) \{", producer)
        ) == len(error_checks)
        for block in error_checks:
            assert "error_flag = 1;" in block
            assert "break;" in block

        free_position = producer.index("slot_full = *free_sem < consumed_required;")
        slot_gate = producer.index("if (consumed_required != 0) {")
        free_invalidation = producer.rfind("invalidate_l1_cache();", slot_gate, free_position)
        assert slot_gate < free_invalidation < free_position

    @staticmethod
    def _stale_consumer_step(done: bool, consumed: int, original_ready: int) -> str:
        if done and original_ready == consumed:
            return "terminate"
        return "consume" if original_ready > consumed else "wait"

    @staticmethod
    def _corrected_consumer_step(
        done: bool, consumed: int, ready: int
    ) -> tuple[str, int]:
        if done:
            fresh_ready = ready
            if fresh_ready == consumed:
                return "terminate", consumed
        if ready > consumed:
            return "consume", consumed + 1
        return "wait", consumed

    def test_final_ready_publication_between_ready_and_done_reads_is_not_dropped(self):
        ready = 0
        done = False
        consumed = 0
        events = []

        original_ready = ready
        events.append("consumer-original-ready-read")
        events.append("producer-final-frame-write")
        ready += 1
        events.append("producer-ready-publication")
        done = True
        events.append("producer-done-publication")
        done_observed = done
        events.append("consumer-done-observation")

        stale_decision = self._stale_consumer_step(done_observed, consumed, original_ready)
        corrected_decision, corrected_consumed = self._corrected_consumer_step(
            done_observed, consumed, ready
        )
        original_read = events.index("consumer-original-ready-read")
        ready_publication = events.index("producer-ready-publication")
        done_publication = events.index("producer-done-publication")
        done_observation = events.index("consumer-done-observation")
        assert original_read < ready_publication < done_publication < done_observation
        assert stale_decision == "terminate"
        assert corrected_decision == "consume"
        assert corrected_consumed == 1
        assert corrected_consumed == ready

    @pytest.mark.parametrize("checkpoint", PRODUCER_CANCEL_CHECKPOINTS)
    def test_producer_cancel_publication_is_seen_at_each_checkpoint(self, checkpoint):
        error_sem = 0
        ready_count = 0
        checked: list[str] = []
        canceled = False
        for stage in self.PRODUCER_CANCEL_CHECKPOINTS:
            if stage == checkpoint:
                error_sem = 1
            checked.append(stage)
            if error_sem != 0:
                canceled = True
                break
        if not canceled:
            ready_count += 1

        assert canceled
        assert checked[-1] == checkpoint
        assert ready_count == 0


def test_kernel_protocol_uses_accessor_ring_metadata_and_budgeted_waits():
    producer = Path("enodia/tt/bench/kernels/resident_producer.cpp").read_text()
    consumer = Path("enodia/tt/bench/kernels/resident_consumer.cpp").read_text()

    assert "get_timestamp()" not in producer
    assert "WALL_CLOCK_H" not in producer
    assert "get_timestamp_32b()" in producer
    assert "extended += static_cast<std::uint32_t>(low - static_cast<std::uint32_t>(extended))" in producer
    assert "noc_async_write_page" in producer
    assert "noc_async_read_page" not in producer
    assert "control.get_noc_addr(0)" in producer
    assert "consumer_x" not in producer
    assert "noc_inline_dw_write" not in producer
    assert "noc_semaphore_inc" in producer
    assert "get_semaphore(free_semaphore_id)" in producer
    assert "while (consumed_required" not in producer
    assert "Drop-new policy" in producer
    assert "run_budget_ticks" in producer
    assert "failure_producer_pacing_wait" in producer
    assert "failure_run_wide_budget" in producer
    assert "failure_elapsed_ticks" in producer
    assert "control_probe" not in producer
    assert "noc_async_read_page(0, control" not in producer
    assert "error_semaphore_id" in producer
    assert "*error_sem" in producer
    assert "noc_semaphore_inc(done_noc, 1)" in producer
    assert "clock.read() - run_start >= run_budget_ticks" in producer

    assert "get_timestamp()" not in consumer
    assert "WALL_CLOCK_H" not in consumer
    assert "get_timestamp_32b()" in consumer
    assert "extended += static_cast<std::uint32_t>(low - static_cast<std::uint32_t>(extended))" in consumer
    assert "noc_semaphore_inc" in consumer
    assert "get_semaphore(ready_semaphore_id)" in consumer
    assert "get_semaphore(done_semaphore_id)" in consumer
    assert "error_semaphore_id" in consumer
    assert "noc_semaphore_inc(error_noc, 1)" in consumer
    assert "*done_sem" in consumer
    assert "control_local" not in consumer
    assert "control_done_word" not in consumer
    assert "control_produced_word" not in consumer
    assert "clock.read() - run_start >= run_budget_ticks" in consumer
    assert "failure_consumer_empty_wait" in consumer
    assert "failure_consumer_fixed_work_budget" in consumer
    assert "failure_run_wide_budget" in consumer
    assert "failure_elapsed_ticks" in consumer
    assert "per_frame_work_budget_ticks" in consumer
    assert "noc_inline_dw_write" not in consumer
    assert "invalidate_l1_cache" in consumer


def test_control_page_is_consumer_l1_and_passed_to_both_accessors():
    runner = Path("enodia/tt/bench/run_resident.py").read_text()
    assert "control = ttnn.zeros" in runner
    assert "memory_config=_sharded_pages_config(ttnn, config, 1)" in runner
    assert "control.buffer_address()" in runner
    assert "control_compile" in runner
    assert "producer_anchor" in runner
    assert "SemaphoreDescriptor(0" in runner
    assert "SemaphoreDescriptor(1" in runner
    assert "SemaphoreDescriptor(2" in runner
    assert "SemaphoreDescriptor(3" in runner
    assert "consumer_compile = [\n        *ring_compile,\n        *producer_anchor_compile" in runner
    assert "allow-budget-margin-over-cap" not in runner
    assert "_resolve_watcher_mode(args.watcher" in runner
    assert "validate_configuration(config, watcher=watcher)" in runner


def test_consumer_budget_failure_cancels_producer_without_control_sync():
    consumer = Path("enodia/tt/bench/kernels/resident_consumer.cpp").read_text()
    producer = Path("enodia/tt/bench/kernels/resident_producer.cpp").read_text()
    assert "error_sent" in consumer
    assert "if (error_flag != 0)" in consumer
    assert "*error_sem != 0" in producer
    assert "attempts_started = attempted + 1" in producer
    assert "ready_count += 1" in producer
    assert "frames_dropped += 1" in producer


def test_every_resident_noc_signal_and_write_has_a_matching_barrier():
    producer = Path("enodia/tt/bench/kernels/resident_producer.cpp").read_text()
    consumer = Path("enodia/tt/bench/kernels/resident_consumer.cpp").read_text()

    def assert_signal_barriers(source, signal_count):
        assert source.count("noc_semaphore_inc(") == signal_count
        positions = []
        cursor = 0
        while (position := source.find("noc_semaphore_inc(", cursor)) >= 0:
            positions.append(position)
            cursor = position + 1
        for position in positions:
            barrier = source.find("noc_async_atomic_barrier()", position)
            assert barrier >= 0
            next_signal = source.find("noc_semaphore_inc(", position + 1)
            assert next_signal < 0 or barrier < next_signal

    def assert_write_barriers(source, write_count):
        assert source.count("noc_async_write_page(") == write_count
        positions = []
        cursor = 0
        while (position := source.find("noc_async_write_page(", cursor)) >= 0:
            positions.append(position)
            cursor = position + 1
        for position in positions:
            barrier = source.find("noc_async_write_barrier()", position)
            assert barrier >= 0
            next_write = source.find("noc_async_write_page(", position + 1)
            assert next_write < 0 or barrier < next_write

    assert "noc_async_read_page(" not in producer + consumer
    assert_signal_barriers(producer, 2)
    assert_signal_barriers(consumer, 4)
    assert_write_barriers(producer, 3)
    assert_write_barriers(consumer, 2)
    assert producer.index("noc_async_write_page(ready_count") < producer.index(
        "noc_semaphore_inc(ready_noc"
    )
    assert producer.index("noc_async_write_page(0, stats") < producer.index(
        "noc_semaphore_inc(done_noc"
    )
    assert "failure_consumer_empty_wait" in consumer
    assert "failure_consumer_fixed_work_budget" in consumer
    assert "failure_run_wide_budget" in consumer
    assert "Drop-new policy" in producer


def test_resident_environment_requires_verified_digest_image():
    valid = {
        "image": "registry.example/tt@sha256:" + "a" * 64,
        "image_pinned": True,
    }
    validate_pinned_environment(valid)
    with pytest.raises(ResidentPreflightError, match="image_pinned"):
        validate_pinned_environment({"image": "registry.example/tt:latest", "image_pinned": False})
    with pytest.raises(ResidentPreflightError, match="image_pinned"):
        validate_pinned_environment({"image": "registry.example/tt@sha256:" + "A" * 64, "image_pinned": True})


class TestResidentProvenance:
    """Board-free tests generated from the production provenance table."""

    PREFLIGHT_FIELDS = tuple(
        field
        for field in REQUIRED_PROVENANCE_FIELDS
        if field.phase == PROVENANCE_PHASE_PREFLIGHT
    )
    POST_RUN_FIELDS = tuple(
        field
        for field in REQUIRED_PROVENANCE_FIELDS
        if field.phase == PROVENANCE_PHASE_POST_RUN
    )

    @staticmethod
    def _set_field(environment: dict, field_path: str, value, *, missing: bool = False) -> None:
        components = field_path.split(".")
        assert components[0] == "environment"
        target = environment
        for component in components[1:-1]:
            target = target[component]
        if missing:
            target.pop(components[-1])
        else:
            target[components[-1]] = value

    @staticmethod
    def _record_value(record: dict, field_path: str):
        value = record
        for component in field_path.split("."):
            assert isinstance(value, dict), field_path
            assert component in value, field_path
            value = value[component]
        return value

    @staticmethod
    def _post_run_record(field_path: str, value, *, missing: bool = False) -> dict:
        record: dict = {}
        target = record
        components = field_path.split(".")
        for component in components[:-1]:
            target[component] = {}
            target = target[component]
        if not missing:
            target[components[-1]] = value
        return record

    def test_valid_complete_environment_passes_preflight(self):
        environment = _environment()
        validate_record_inputs(
            harness_commit=environment["harness_commit"],
            environment=environment,
            power_trace="resident-power.csv",
        )

    @pytest.mark.parametrize(
        ("board", "canonical_serial"),
        [
            ({"board_type": "p150a", "board_id": "board-id-only"}, "board-id-only"),
            ({"board_type": "p150a", "serial": "serial-only"}, "serial-only"),
            (
                {"board_type": "p150a", "serial": "matching", "board_id": "matching"},
                "matching",
            ),
        ],
        ids=["board-id-only", "serial-only", "matching-identities"],
    )
    def test_preflight_accepts_telemetry_board_identity_shapes(self, board, canonical_serial):
        environment = _environment()
        environment["board"] = board
        original = copy.deepcopy(environment)

        normalized = validate_preflight_provenance(
            harness_commit=environment["harness_commit"], environment=environment
        )

        assert normalized["board"]["serial"] == canonical_serial
        assert environment == original

    def test_preflight_rejects_conflicting_telemetry_board_identity(self):
        environment = _environment()
        environment["board"] = {
            "board_type": "p150a",
            "serial": "serial",
            "board_id": "different",
        }
        original = copy.deepcopy(environment)

        with pytest.raises(ValueError, match="serial and environment.board.board_id"):
            validate_preflight_provenance(
                harness_commit=environment["harness_commit"], environment=environment
            )

        assert environment == original

    def test_preflight_preserves_required_serial_rejection_when_identity_is_absent(self):
        environment = _environment()
        environment["board"] = {"board_type": "p150a"}

        with pytest.raises(ValueError, match="environment.board.serial"):
            validate_preflight_provenance(
                harness_commit=environment["harness_commit"], environment=environment
            )

    def test_record_builder_stores_a_normalized_environment_copy(self):
        environment = _environment()
        environment["board"] = {"board_type": "p150a", "board_id": "board-id-only"}
        original = copy.deepcopy(environment)

        record = build_measurement_record(
            config=_config(frame_count=2),
            aiclk_mhz=1_350,
            timestamps=[1_000, 2_000],
            producer_full_count=0,
            consumer_empty_count=0,
            cycle_budget_hit=False,
            kernel_error_flag=0,
            harness_commit=environment["harness_commit"],
            environment=environment,
            power_trace="resident-power.csv",
        )

        assert record["environment"]["board"]["serial"] == "board-id-only"
        assert environment == original
        assert "serial" not in environment["board"]

    @pytest.mark.parametrize("field", PREFLIGHT_FIELDS, ids=lambda field: field.path)
    @pytest.mark.parametrize("bad_kind", ["missing", "blank", "wrong_type"])
    def test_preflight_rejects_every_table_field_before_device_opening(self, field, bad_kind):
        environment = copy.deepcopy(_environment())
        value = " " if bad_kind == "blank" else 123
        if field.path == "harness_commit":
            harness_commit = None if bad_kind == "missing" else value
        else:
            self._set_field(environment, field.path, value, missing=bad_kind == "missing")
            harness_commit = "0123456789abcdef"
        with pytest.raises(ValueError, match=re.escape(field.path)):
            validate_record_inputs(
                harness_commit=harness_commit,
                environment=environment,
                power_trace="resident-power.csv",
            )

    def test_preflight_rejection_is_written_without_opening_device(self, tmp_path, monkeypatch):
        environment = _environment()
        del environment["board"]["serial"]
        environment_path = tmp_path / "environment.json"
        output_path = tmp_path / "rejection.json"
        environment_path.write_text(json.dumps(environment))
        opened = []
        monkeypatch.setitem(
            sys.modules,
            "ttnn",
            SimpleNamespace(open_device=lambda **kwargs: opened.append(kwargs)),
        )

        result = run_resident.main(
            [
                "--out",
                str(output_path),
                "--env-json",
                str(environment_path),
                "--power-trace",
                "resident-power.csv",
            ]
        )

        assert result == 2
        assert opened == []
        rejection = json.loads(output_path.read_text())
        assert rejection["status"] == "rejected"
        assert "environment.board.serial" in rejection["rejection_reason"]

    def test_accepted_current_wrap_record_satisfies_every_table_entry(self):
        record = json.loads(
            (
                ROOT
                / "docs/measurements/2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json"
            ).read_text()
        )
        for field in REQUIRED_PROVENANCE_FIELDS:
            value = self._record_value(record, field.path)
            assert field.validator(value), field.path

    @pytest.mark.parametrize("field", POST_RUN_FIELDS, ids=lambda field: field.path)
    @pytest.mark.parametrize("bad_kind", ["missing", "blank", "wrong_type"])
    def test_post_run_rejects_missing_blank_and_wrong_type_aiclk(self, field, bad_kind):
        value = " " if bad_kind == "blank" else "1350"
        record = self._post_run_record(
            field.path, value, missing=bad_kind == "missing"
        )
        with pytest.raises(ValueError, match=re.escape(field.path)):
            validate_post_run_provenance(record)

    @pytest.mark.parametrize("field", POST_RUN_FIELDS, ids=lambda field: field.path)
    @pytest.mark.parametrize("value", [0, -1])
    def test_post_run_rejects_non_positive_aiclk(self, field, value):
        with pytest.raises(ValueError, match=re.escape(field.path)):
            validate_post_run_provenance(self._post_run_record(field.path, value))


def test_record_preflight_rejects_empty_commit_and_non_filename_trace():
    environment = _environment()
    with pytest.raises(ValueError, match="harness_commit"):
        validate_record_inputs(
            harness_commit=" ",
            environment=environment,
            power_trace="resident-power.csv",
        )
    with pytest.raises(ValueError, match="power_trace"):
        validate_record_inputs(
            harness_commit="0123456789abcdef",
            environment=environment,
            power_trace="nested/resident-power.csv",
        )


def test_record_rejects_missing_environment_provenance():
    config = _config()
    with pytest.raises(ValueError, match="environment"):
        build_measurement_record(
            config=config,
            aiclk_mhz=1_350,
            timestamps=[1, 2],
            producer_full_count=0,
            consumer_empty_count=0,
            cycle_budget_hit=False,
            kernel_error_flag=0,
            harness_commit="0123456789abcdef",
            environment={},
            power_trace="resident-power.csv",
        )
