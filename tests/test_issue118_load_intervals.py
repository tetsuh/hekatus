"""Board-free contracts for Issue #118 load intervals and generators."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from enodia.tt.bench import load_generators, newton_schulz_kernel
from enodia.tt.bench.load_harness import (
    DEVICE_OPEN_SETTINGS,
    LOAD_CONDITIONS,
    NS_LOAD_BATCH,
    NS_LOAD_L1_TOTAL_BYTES,
    NS_LOAD_MAX_TILES_PER_CORE,
    NUM_COMMAND_QUEUES,
    P150_RESIDUAL_WORKER_COORDINATES,
    P150_RESIDUAL_WORKER_COUNT,
    RESIDENT_CORES,
    CalibrationMetadata,
    CardLoadInterval,
    HostLoadInterval,
    LoadEvidenceError,
    build_calibration_load_record,
    build_load_record,
    choose_repeat_count,
    extend_card_clock_timestamps,
    load_evidence_valid,
    newton_schulz_l1_ledger,
    ns_work_distribution,
    validate_load_record,
    validate_mode_records,
    validate_worker_model,
)
from enodia.tt.bench.resident_harness import ResidentConfig, run_budget_breakdown
from enodia.tt.bench.run_load import _parser as load_parser
from enodia.tt.bench.run_load import _read_calibration as read_load_calibration
from enodia.tt.bench.run_load import build_calibration, build_run_configuration
from enodia.tt.bench.run_resident import (
    _bound_calibration_config,
    _host_join_timeout_seconds,
    _host_load_clock,
    _load_achieved_throughput,
    _load_execution_selection,
    _read_load_calibration,
)
from enodia.tt.bench.run_resident import _parser as resident_parser

ROOT = Path(__file__).parents[1]
KERNEL_DIR = ROOT / "enodia/tt/bench/kernels"


def test_p150_worker_model_reserves_resident_cores_and_fails_closed():
    assert P150_RESIDUAL_WORKER_COUNT == 108
    assert P150_RESIDUAL_WORKER_COORDINATES[0] == (2, 0)
    assert P150_RESIDUAL_WORKER_COORDINATES[-1] == (10, 9)
    assert not set(P150_RESIDUAL_WORKER_COORDINATES) & set(RESIDENT_CORES)
    assert validate_worker_model() == P150_RESIDUAL_WORKER_COORDINATES
    with pytest.raises(ValueError, match="unsupported worker grid"):
        validate_worker_model(grid=(12, 10))
    with pytest.raises(ValueError, match="pinned residual set"):
        validate_worker_model(coordinates=P150_RESIDUAL_WORKER_COORDINATES[:-1])


def test_newton_schulz_distribution_and_l1_ledger_are_fixed():
    rows = ns_work_distribution()
    assert len(rows) == 108
    assert sum(row["tile_count"] for row in rows) == NS_LOAD_BATCH
    assert max(row["tile_count"] for row in rows) == NS_LOAD_MAX_TILES_PER_CORE == 80
    assert rows[0]["start_tile"] == 0
    assert rows[-1]["start_tile"] + rows[-1]["tile_count"] == NS_LOAD_BATCH
    ns = load_generators.newton_schulz_load_work()
    assert ns["shape"] == "newton_schulz_L32_b8192"
    assert ns["iterations"] == 12
    assert ns["outer_repetitions"] == 3
    assert ns["variant"] == "bf16"
    assert ns["math_fidelity"] == "HiFi3"
    assert ns["fuse_s"] is True
    assert ns["matrix_block"] == 8
    ledger = newton_schulz_l1_ledger()
    assert ledger["total_bytes"] == NS_LOAD_L1_TOTAL_BYTES == 1_295_104
    assert ledger["total_bytes"] < ledger["budget_bytes"] == 1_572_864
    ttnn = SimpleNamespace(bfloat16="bf16", float32="fp32")
    assert newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=108,
        state_dtype="bf16",
        fuse_s=True,
        output_memory="dram",
        matrix_block=8,
        double_buffer=True,
        variant="bf16",
    ) == 1_295_104


def test_fixed_load_sources_have_no_convergence_or_data_dependent_exit():
    ns_source = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    dram_source = (KERNEL_DIR / "issue118_dram_load.cpp").read_text()
    assert "static_assert(iterations == 12" in ns_source
    assert "for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)" in ns_source
    assert "const std::uint32_t outer_repetitions = get_arg_val<std::uint32_t>(2)" in ns_source
    assert "for (std::uint32_t outer = 0; outer < outer_repetitions; ++outer)" in ns_source
    writer_source = (KERNEL_DIR / "newton_schulz_writer.cpp").read_text()
    profile_writer_source = (KERNEL_DIR / "newton_schulz_writer_profile.cpp").read_text()
    reader_profile_source = (KERNEL_DIR / "newton_schulz_reader_profile.cpp").read_text()
    assert "get_arg_val<std::uint32_t>(4)" in writer_source
    assert "get_arg_val<std::uint32_t>(6)" in profile_writer_source
    assert writer_source.count("outer_repetitions") >= 2
    assert profile_writer_source.count("outer_repetitions") >= 2
    assert reader_profile_source.count("outer == 0 && offset == 0") == 4
    assert profile_writer_source.count("outer == 0 && offset == 0") == 1
    assert "if (conver" not in ns_source.lower()
    assert "static_assert(dram_banks == 8" in dram_source
    assert "const std::uint32_t bank_index" in dram_source
    assert "for (std::uint32_t page = 0; page < pages_per_bank; ++page)" in dram_source
    assert "const std::uint32_t local_addr = get_write_ptr(cb_stream);" in dram_source
    assert "noc_async_read_page(page_index, read, local_addr);" in dram_source
    assert "noc_async_write_page(page_index, write, local_addr);" in dram_source
    assert "clock.read();" in dram_source
    assert "break;" not in ns_source
    assert "break;" not in dram_source


def test_resident_cb_indices_are_disjoint_from_newton_schulz_abi():
    source = (ROOT / "enodia/tt/bench/run_resident.py").read_text()
    assert "RESIDENT_PRODUCER_CB_INDEX = 20" in source
    assert "RESIDENT_CONSUMER_CB_INDEX = 21" in source
    assert "return_parts" in source


def test_calibration_duration_bounds_actual_resident_frame_count():
    config = ResidentConfig(
        frame_count=100,
        frame_interval_ticks=1_350_000,
        producer_core=(0, 0),
        consumer_core=(1, 0),
        ring_pages=4,
        work_per_frame=64,
        designated_timestamp_core=(1, 0),
        cycle_budget=10_000_000,
        outer_timeout_seconds=60,
        fixed_work_ticks_per_frame=100_000,
        budget_aiclk_mhz=1_350,
    )
    bounded = _bound_calibration_config(config, 60.0)
    assert bounded.frame_count > config.frame_count
    assert bounded.outer_timeout_seconds == 60
    assert run_budget_breakdown(bounded)["run_budget_ticks"] <= 54 * 1_350 * 1_000_000


def test_runner_parsers_expose_loaded_conditions_and_calibration_controls():
    resident = resident_parser().parse_args(
        [
            "--out",
            "/out/result.json",
            "--condition",
            "newton_schulz",
            "--calibrate",
            "--repeat-count",
            "1",
            "--calibration-duration-seconds",
            "60",
        ]
    )
    assert resident.condition == "newton_schulz"
    assert resident.calibrate is True
    assert resident.repeat_count == 1
    loaded = load_parser().parse_args(
        ["--out", "/out/config.json", "--load-condition", "dram", "--run-config"]
    )
    assert loaded.condition == "dram"
    assert loaded.run_config is True


def test_fixed_repeat_reaches_reader_and_compute_runtime_args():
    class RuntimeArgs:
        def __init__(self):
            self.values = {}

        def __getitem__(self, x):
            return _GridRow(self.values, x)

    class _GridRow:
        def __init__(self, values, x):
            self.values = values
            self.x = x

        def __setitem__(self, y, value):
            self.values[(self.x, y)] = value

    ttnn = SimpleNamespace(RuntimeArgs=RuntimeArgs)
    args = newton_schulz_kernel._runtime_args(
        ttnn, [(2, 0)], [123], [(0, 8)], [3]
    )
    assert args.values[(2, 0)] == [123, 0, 8, 3]


def test_resident_calibration_reader_filters_extra_fields_and_rejects_model_source(tmp_path):
    payload = build_calibration(
        condition="dram",
        duration_seconds=5,
        achieved_throughput=2,
        target_seconds=500,
    )
    payload["calibration"].update(
        {
            "requested_duration_seconds": 60,
            "actual_frame_count": 123,
            "actual_duration_seconds": 52.0,
            "measurement_repeat_count": 1,
        }
    )
    path = tmp_path / "resident-calibration.json"
    path.write_text(json.dumps(payload))
    parsed = _read_load_calibration(path)
    parsed_board_free = read_load_calibration(path)
    assert parsed.condition == "dram"
    assert parsed_board_free.condition == "dram"
    model_payload = dict(payload)
    model_payload["calibration"] = {**payload["calibration"], "source": "board_free_model"}
    path.write_text(json.dumps(model_payload))
    with pytest.raises(ValueError, match="watcher_off_calibration"):
        _read_load_calibration(path)


def test_host_cleanup_stop_and_join_are_in_outer_finally():
    source = (ROOT / "enodia/tt/bench/run_resident.py").read_text()
    assert "stop_event.is_set()" in source
    assert "host_stop.set()" in source
    assert source.rfind("host_stop.set()") < source.rfind("host_thread.join")
    assert "HostCleanupUnsafe" in source


def test_loaded_run_rejects_a_short_outer_timeout_for_calibrated_target(tmp_path):
    calibration = CalibrationMetadata(
        condition="dram",
        duration_seconds=60,
        achieved_throughput=2,
        throughput_unit="repeats_per_card_second",
        repeat_count=1100,
        target_seconds=500,
        safety_factor=1.10,
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"calibration": calibration.as_record()}))
    args = SimpleNamespace(
        condition="dram",
        calibrate=False,
        calibration_json=path,
        repeat_count=None,
        outer_timeout_seconds=60,
    )
    with pytest.raises(ValueError, match="outer-timeout-seconds"):
        _load_execution_selection(args)
    args.outer_timeout_seconds = 600
    assert _load_execution_selection(args)[1] == 1100


def test_host_load_clock_uses_launch_once_and_fresh_endpoint():
    import threading

    launched = threading.Event()
    launched.set()
    start_holder = {"launch_ns": 100}
    clock = _host_load_clock(
        resident_launched=launched,
        stop_event=threading.Event(),
        start_holder=start_holder,
        monotonic_ns=iter((250, 400)).__next__,
    )
    assert clock() == 100
    assert start_holder["start_ns"] == 100
    assert clock() == 250
    assert start_holder["start_ns"] == 100


def test_host_join_timeout_never_exceeds_resident_outer_cap():
    assert _host_join_timeout_seconds(repeat_count=550, outer_timeout_seconds=60) == 60.0
    assert _host_join_timeout_seconds(repeat_count=550, outer_timeout_seconds=600) == 600.0



def test_actual_program_plumbing_names_repeat_and_interval_outputs():
    resident_source = (ROOT / "enodia/tt/bench/run_resident.py").read_text()
    generators_source = (ROOT / "enodia/tt/bench/load_generators.py").read_text()
    assert "_run_loaded_device" in resident_source
    assert "prepare_newton_schulz_load" in resident_source
    assert "prepare_dram_load" in resident_source
    assert "merge_program_parts" in resident_source
    assert "_merged_program_io_tensors(resident_tensors)" in resident_source
    assert "return [consumer_stats, consumer_stats]" in resident_source
    assert "load_tensors.append(load_kernel.profile_output)" in resident_source
    assert "io_tensors.append(self.profile_output)" in (ROOT / "enodia/tt/bench/newton_schulz_kernel.py").read_text()
    assert "queue_id=queue_id" in resident_source
    assert 'record["timing_evidence"] = False' in resident_source
    assert "load_record=resident_load_record" in resident_source
    assert "if calibration_output is None:" in resident_source
    assert resident_source.index("record = build_measurement_record") < resident_source.index(
        "if calibration_output is not None:"
    )
    assert 'record["status"] = "calibration"' in resident_source
    assert resident_source.rindex('host_start_holder["launch_ns"] = launch_ns') < resident_source.rindex("resident_launched.set()")
    assert resident_source.rindex('host_start_holder["launch_ns"] = launch_ns') < resident_source.rindex("ttnn.synchronize_device(device, cq_id=0)")
    assert resident_source.index("measurement_end_ns = time.monotonic_ns()") < resident_source.index("host_thread.join")
    assert resident_source.index('"launch_ns" not in start_holder') < resident_source.rindex("run_host_cpu_transfer_work")
    assert "outer_repetitions=NS_LOAD_OUTER_REPETITIONS * repeat_count" in generators_source
    assert "LOAD_INTERVAL_CB_INDEX" in generators_source


def test_profile_cleanup_uses_identity_not_tensor_equality():
    class Tensor:
        def __eq__(self, _other):
            raise AssertionError("tensor equality must not be used for cleanup")

    class Ttnn:
        def deallocate(self, _tensor):
            return None

    kernel = object.__new__(newton_schulz_kernel.NewtonSchulzKernel)
    kernel.ttnn = Ttnn()
    kernel.inputs = [Tensor()]
    kernel.outputs = [Tensor()]
    kernel.profile_output = Tensor()
    kernel.close()


def test_merged_generic_io_pair_is_homogeneous():
    from enodia.tt.bench.run_resident import _merged_program_io_tensors

    tensor = object()
    pair = _merged_program_io_tensors({"consumer_stats": tensor})
    assert pair == [tensor, tensor]


def test_program_plans_use_one_program_and_disjoint_core_ranges():
    plans = load_generators.validate_plan_catalogue()
    assert {plan.condition for plan in plans} == set(LOAD_CONDITIONS)
    for plan in plans:
        assert plan.same_program is True
        assert plan.resident_command_queue == 0
        assert plan.transfer_command_queue == 1
        assert not set(plan.worker_cores) & set(plan.resident_cores)


def test_merged_program_rejects_cb_collisions_and_keeps_all_groups():
    class Descriptor:
        def __init__(self, index):
            self.format_descriptors = [SimpleNamespace(buffer_index=index)]

    class Program:
        def __init__(self, index, kernel):
            self.kernels = [kernel]
            self.semaphores = []
            self.cbs = [Descriptor(index)]

    class Ttnn:
        def ProgramDescriptor(self, **kwargs):
            return SimpleNamespace(**kwargs)

    merged = load_generators.merge_program_descriptors(
        Ttnn(), Program(20, "resident"), Program(0, "load")
    )
    assert merged.kernels == ["resident", "load"]
    assert [descriptor.format_descriptors[0].buffer_index for descriptor in merged.cbs] == [20, 0]
    with pytest.raises(ValueError, match="collision"):
        load_generators.merge_program_descriptors(Ttnn(), Program(20, "a"), Program(20, "b"))


def test_fixed_dram_and_host_work_metadata():
    transfers = []
    host_run = load_generators.run_host_cpu_transfer_work(
        clock_ns=iter((100, 250)).__next__,
        transfer=lambda payload, queue: transfers.append((len(payload), queue)),
    )
    assert host_run["start_ns"] == 100
    assert host_run["end_ns"] == 250
    assert transfers == [(64 * 1024, 1)]
    dram = load_generators.dram_load_work()
    assert dram["bank_count"] == 8
    assert dram["read_pages"] == dram["write_pages"] == 8 * 256
    assert dram["data_dependent_exit"] is False
    host = load_generators.host_cpu_transfer_work()
    assert host["command_queue"] == 1
    assert host["arithmetic_rounds"] == 16
    assert host["arithmetic_words"] == 4_096
    assert host["transfer_bytes"] == 64 * 1024
    assert host["transfer_count"] == 1
    assert host["transfers"] == "fixed_size_host_device_transfers"


def test_card_profile_interval_requires_wrap_tracked_endpoints():
    interval_from_words = load_generators.card_interval_from_words((0xFFFFFFFE, 0, 0xFFFFFFFF, 0))
    assert interval_from_words.start_ticks == 0xFFFFFFFE
    assert interval_from_words.end_ticks == 0xFFFFFFFF
    interval = load_generators.card_interval_from_profile_records(
        [
            {"risc": "TRISC1", "load_start_ticks": 10, "load_end_ticks": 20},
            {"risc": "TRISC0", "load_start_ticks": 0x1_0000_0000, "load_end_ticks": 0x1_0000_0100},
        ]
    )
    assert interval.start_ticks == 0x1_0000_0000
    with pytest.raises(ValueError, match="no wrap-tracked"):
        load_generators.card_interval_from_profile_records([{"risc": "TRISC0"}])


def test_uncovered_calibration_is_not_main_timing_evidence():
    calibration = CalibrationMetadata(
        condition="dram",
        duration_seconds=60,
        achieved_throughput=10,
        throughput_unit="repeats_per_card_second",
        repeat_count=5500,
        target_seconds=500,
    )
    record = build_calibration_load_record(
        condition="dram",
        repeat_count=1,
        resident_timestamps=[100, 200, 300],
        card_interval=CardLoadInterval(150, 250),
        calibration=calibration,
        achieved_throughput=10,
        throughput_unit="repeats_per_card_second",
    )
    assert record["status"] == "calibration"
    assert record["valid_evidence"] is False
    assert record["coverage"]["verdict"] == "uncovered"
    assert record["calibration"]["repeat_count"] == 5500


def test_card_interval_wrap_extension_and_coverage_are_card_only():
    timestamps = extend_card_clock_timestamps((0xFFFFFFFE, 0xFFFFFFFF, 0, 2))
    assert timestamps == (0xFFFFFFFE, 0xFFFFFFFF, 0x1_0000_0000, 0x1_0000_0002)
    record = build_load_record(
        condition="newton_schulz",
        repeat_count=12,
        resident_timestamps=[100, 200, 300],
        card_interval=CardLoadInterval(90, 310),
    )
    assert record["coverage"]["verdict"] == "covered"
    assert record["card_load_interval"]["clock"] == "card_riscv_wall_clock"
    assert record["host_load_interval"] is None
    assert load_evidence_valid(record)
    uncovered = dict(record)
    uncovered["coverage"] = {**record["coverage"], "covered": False, "verdict": "uncovered"}
    uncovered["valid_evidence"] = False
    assert not load_evidence_valid(uncovered)
    with pytest.raises(LoadEvidenceError):
        build_load_record(
            condition="dram",
            repeat_count=1,
            resident_timestamps=[100, 200, 300],
            card_interval=CardLoadInterval(101, 310),
        )


def test_host_condition_uses_host_interval_for_measurement_coverage():
    record = build_load_record(
        condition="host_cpu_transfer",
        repeat_count=7,
        resident_timestamps=None,
        host_interval=HostLoadInterval(90, 310),
        measurement_start_ns=100,
        measurement_end_ns=300,
    )
    assert record["coverage"]["clock"] == "host_monotonic_ns"
    assert record["coverage"]["verdict"] == "covered"
    validate_load_record(record)
    with pytest.raises(LoadEvidenceError):
        build_load_record(
            condition="host_cpu_transfer",
            repeat_count=7,
            resident_timestamps=None,
            host_interval=HostLoadInterval(101, 310),
            measurement_start_ns=100,
            measurement_end_ns=300,
        )


def test_device_coverage_rejects_host_timestamp_substitution():
    record = build_load_record(
        condition="newton_schulz",
        repeat_count=1,
        resident_timestamps=[100, 200],
        card_interval=CardLoadInterval(90, 210),
    )
    record["card_load_interval"] = None
    record["host_load_interval"] = HostLoadInterval(0, 1)
    record["valid_evidence"] = False
    assert not load_evidence_valid(record)


def test_two_cq_setting_is_present_on_all_four_mode_records():
    records = {
        "unloaded": build_load_record(condition="unloaded", repeat_count=0, resident_timestamps=None),
        "newton_schulz": build_load_record(
            condition="newton_schulz",
            repeat_count=1,
            resident_timestamps=[100, 200],
            card_interval=CardLoadInterval(90, 210),
        ),
        "dram": build_load_record(
            condition="dram",
            repeat_count=1,
            resident_timestamps=[100, 200],
            card_interval=CardLoadInterval(90, 210),
        ),
        "host_cpu_transfer": build_load_record(
            condition="host_cpu_transfer",
            repeat_count=1,
            resident_timestamps=None,
            host_interval=HostLoadInterval(90, 210),
            measurement_start_ns=100,
            measurement_end_ns=200,
        ),
    }
    assert DEVICE_OPEN_SETTINGS["num_command_queues"] == NUM_COMMAND_QUEUES == 2
    validate_mode_records(records)
    assert all(record["num_command_queues"] == 2 for record in records.values())


def test_actual_runner_reports_repeat_rate_not_work_rate():
    ns_result = {"card_load_interval": CardLoadInterval(0, 1_000_000)}
    dram_result = {"card_load_interval": CardLoadInterval(0, 1_000_000)}
    host_result = {
        "host_load_interval": HostLoadInterval(0, 1_000_000),
        "host_work": {"transfer_count": 1},
    }
    assert _load_achieved_throughput(
        ns_result, condition="newton_schulz", repeat_count=1, aiclk_mhz=1000
    ) == (1000.0, "repeats_per_card_second")
    assert _load_achieved_throughput(
        dram_result, condition="dram", repeat_count=1, aiclk_mhz=1000
    ) == (1000.0, "repeats_per_card_second")
    assert _load_achieved_throughput(
        host_result, condition="host_cpu_transfer", repeat_count=1, aiclk_mhz=1000
    ) == (1000.0, "repeats_per_host_second")


def test_calibration_repeat_rate_is_not_work_rate():
    expected_units = {
        "newton_schulz": "repeats_per_card_second",
        "dram": "repeats_per_card_second",
        "host_cpu_transfer": "repeats_per_host_second",
    }
    for condition, expected_unit in expected_units.items():
        calibration = build_calibration(
            condition=condition,
            duration_seconds=5,
            achieved_throughput=2,
            target_seconds=10,
        )
        assert calibration["calibration"]["repeat_count"] == 22
        assert calibration["calibration"]["throughput_unit"] == expected_unit
    with pytest.raises(ValueError, match="throughput-unit"):
        build_calibration(
            condition="dram",
            duration_seconds=5,
            achieved_throughput=2,
            throughput_unit="repeats_per_host_second",
            target_seconds=10,
        )
    bounded = build_calibration(
        condition="dram",
        duration_seconds=5,
        achieved_throughput=2,
        target_seconds=500,
    )
    assert bounded["calibration"]["repeat_count"] / 2 == 550
    assert bounded["calibration"]["repeat_count"] / 2 < 600


def test_calibration_records_persist_watcher_off_throughput_and_repeat_count(tmp_path):
    payload = build_calibration(
        condition="dram",
        duration_seconds=5,
        achieved_throughput=10,
        target_seconds=500,
    )
    assert payload["watcher_enabled"] is False
    assert "/home/" not in json.dumps(payload)
    assert payload["calibration"]["duration_seconds"] == 5.0
    assert payload["calibration"]["repeat_count"] == choose_repeat_count(
        condition="dram", calibration_throughput=10, target_seconds=500
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    calibration = CalibrationMetadata(**payload["calibration"])
    config = build_run_configuration(condition="dram", calibration=calibration)
    assert config["repeat_count"] == payload["selected_repeat_count"]
    with pytest.raises(ValueError, match="calibration metadata"):
        build_run_configuration(condition="dram", repeat_count=1)
    assert config["valid_evidence"] is False
    assert config["num_command_queues"] == 2


def test_calibration_is_capped_at_sixty_seconds():
    with pytest.raises(ValueError, match="<= 60"):
        build_calibration(condition="dram", duration_seconds=60.001, achieved_throughput=1)
    with pytest.raises(ValueError, match="<= 500"):
        build_calibration(
            condition="dram",
            duration_seconds=5,
            achieved_throughput=1,
            target_seconds=500.001,
        )
