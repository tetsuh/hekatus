"""Board-free contracts for Issue #118 load intervals and generators."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from enodia.tt.bench import load_generators, newton_schulz_kernel
from enodia.tt.bench import run_resident as resident_runner
from enodia.tt.bench.load_harness import (
    CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE,
    CALIBRATION_LOAD_TIMEOUT_MARGIN_SECONDS,
    CARD_THROUGHPUT_UNIT_DEFINITION,
    DEVICE_OPEN_SETTINGS,
    HOST_RESIDENT_SUB_DEVICE_ID,
    HOST_TRANSFER_CORE,
    HOST_TRANSFER_SUB_DEVICE_ID,
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
    CardWorkerInterval,
    HostLoadInterval,
    LoadEvidenceError,
    build_calibration_load_record,
    build_load_record,
    choose_repeat_count,
    extend_card_clock_timestamps,
    load_evidence_valid,
    newton_schulz_l1_ledger,
    ns_work_distribution,
    predicted_load_time_seconds,
    validate_calibration_load_record,
    validate_load_record,
    validate_mode_records,
    validate_worker_model,
)
from enodia.tt.bench.power_trace_snapshot import read_power_trace_csv
from enodia.tt.bench.resident_harness import (
    AUDITED_CLOCK_SOURCE_IMAGE,
    ResidentConfig,
    build_measurement_record,
    run_budget_breakdown,
    validate_resident_record,
)
from enodia.tt.bench.resident_record import build_outlier_analysis, validate_pair_analysis
from enodia.tt.bench.run_load import _parser as load_parser
from enodia.tt.bench.run_load import _read_calibration as read_load_calibration
from enodia.tt.bench.run_load import build_calibration, build_run_configuration
from enodia.tt.bench.run_resident import (
    _bound_calibration_config,
    _complete_host_transfers,
    _config_from_args,
    _host_join_timeout_seconds,
    _host_load_clock,
    _load_achieved_throughput,
    _load_execution_selection,
    _read_load_calibration,
    _run_host_transfers,
    _wait_for_power_trace_sample,
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


def test_host_transfer_subdevice_partition_is_disjoint_and_source_backed():
    assert HOST_RESIDENT_SUB_DEVICE_ID == 0
    assert HOST_TRANSFER_SUB_DEVICE_ID == 1
    assert HOST_TRANSFER_CORE in P150_RESIDUAL_WORKER_COORDINATES
    assert HOST_TRANSFER_CORE not in RESIDENT_CORES
    source = (ROOT / "enodia/tt/bench/run_resident.py").read_text()
    assert "create_sub_device_manager" in source
    assert "set_sub_device_stall_group" in source
    assert "sub_device_ids=transfer_sub_device_ids" in source
    assert "sub_device_ids=[ttnn.SubDeviceId(HOST_RESIDENT_SUB_DEVICE_ID)]" in source


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


class _FakeHostTransferTensor:
    def __init__(self, kind: str, slot: int | None = None):
        self.kind = kind
        self.slot = slot


class _FakeHostTransferTtnn:
    DRAM_MEMORY_CONFIG = object()
    ROW_MAJOR_LAYOUT = object()
    uint32 = object()

    def __init__(self, *, fail_on_copy: int | None = None):
        self.events = []
        self.host_allocations = []
        self.device_allocations = []
        self.deallocated = []
        self.copy_count = 0
        self.fail_on_copy = fail_on_copy
        self.synchronize_sub_device_ids = []

    def Tensor(self, _array, _dtype):
        tensor = _FakeHostTransferTensor("host", len(self.host_allocations))
        self.host_allocations.append(tensor)
        return tensor

    def to_layout(self, tensor, _layout):
        return tensor

    def Shape(self, shape):
        return shape

    def allocate_tensor_on_device(self, _shape, _dtype, _layout, _device, _memory_config):
        tensor = _FakeHostTransferTensor("device", len(self.device_allocations))
        self.device_allocations.append(tensor)
        return tensor

    def copy_host_to_device_tensor(self, host, device, *, cq_id):
        self.copy_count += 1
        if self.fail_on_copy == self.copy_count:
            raise RuntimeError("injected CQ1 copy failure")
        self.events.append(("host_to_device", host.slot, device.slot, cq_id))

    def copy_device_to_host_tensor(self, device, host, *, blocking, cq_id):
        self.events.append(("device_to_host", device.slot, host.slot, blocking, cq_id))

    def synchronize_device(self, _device, *, cq_id, sub_device_ids=None):
        self.synchronize_sub_device_ids.append(sub_device_ids)
        self.events.append(("synchronize", cq_id))

    def deallocate(self, tensor):
        self.deallocated.append(tensor)


def _run_fake_host_transfer(
    monkeypatch,
    repetitions,
    *,
    transfer_count=5,
    fail_on_copy=None,
    transfer_sub_device_ids=None,
):
    import threading

    monkeypatch.setattr(resident_runner, "HOST_TRANSFER_COUNT", transfer_count)
    monkeypatch.setattr(resident_runner, "HOST_ARITHMETIC_ROUNDS", 2)
    monkeypatch.setattr(resident_runner, "HOST_ARITHMETIC_WORDS", 3)
    ttnn = _FakeHostTransferTtnn(fail_on_copy=fail_on_copy)
    device_tensors = []
    host_tensors = []
    started = threading.Event()
    resident_launched = threading.Event()
    resident_launched.set()
    result = _run_host_transfers(
        ttnn,
        object(),
        repetitions=repetitions,
        started_event=started,
        resident_launched=resident_launched,
        start_holder={},
        device_tensors_holder=device_tensors,
        host_tensors_holder=host_tensors,
        transfer_sub_device_ids=transfer_sub_device_ids,
    )
    return ttnn, device_tensors, host_tensors, started, result


def test_host_transfer_ring_keeps_allocations_fixed_and_synchronizes_repeats(monkeypatch):
    allocation_counts = []
    for repetitions in (1, 3):
        ttnn, device_tensors, host_tensors, started, result = _run_fake_host_transfer(
            monkeypatch, repetitions
        )
        expected_pairs = 5 * repetitions
        expected_events = []
        for _ in range(repetitions):
            for transfer_index in range(5):
                slot = transfer_index % resident_runner.HOST_TRANSFER_BUFFER_POOL_SIZE
                expected_events.extend(
                    [
                        ("host_to_device", slot, slot, 1),
                        ("device_to_host", slot, slot, False, 1),
                    ]
                )
            expected_events.append(("synchronize", 1))
        assert ttnn.events == expected_events
        assert len([event for event in ttnn.events if event[0] == "host_to_device"]) == expected_pairs
        assert len([event for event in ttnn.events if event[0] == "device_to_host"]) == expected_pairs
        assert len([event for event in ttnn.events if event[0] == "synchronize"]) == repetitions
        assert started.is_set()
        assert result["transfer_count"] == expected_pairs
        assert result["transfer_completion"] == "per_repeat_cq1"
        expected_checksum = 0
        for _ in range(repetitions):
            for _ in range(2):
                for word in range(3):
                    expected_checksum = ((expected_checksum * 33) ^ word) & 0xFFFFFFFF
        assert result["arithmetic_checksum"] == expected_checksum
        assert result["buffer_pool_size"] == resident_runner.HOST_TRANSFER_BUFFER_POOL_SIZE
        allocation_counts.append((len(ttnn.device_allocations), len(ttnn.host_allocations)))
        assert allocation_counts[-1] == (
            resident_runner.HOST_TRANSFER_BUFFER_POOL_SIZE,
            resident_runner.HOST_TRANSFER_BUFFER_POOL_SIZE,
        )
        _complete_host_transfers(ttnn, object(), device_tensors, host_tensors)
        assert not device_tensors
        assert not host_tensors
        assert len(ttnn.deallocated) == 2 * resident_runner.HOST_TRANSFER_BUFFER_POOL_SIZE
    assert allocation_counts[0] == allocation_counts[1]


def test_host_transfer_sync_targets_only_the_transfer_subdevice(monkeypatch):
    transfer_sub_device_ids = [object()]
    ttnn, _device_tensors, _host_tensors, _started, _result = _run_fake_host_transfer(
        monkeypatch,
        repetitions=3,
        transfer_sub_device_ids=transfer_sub_device_ids,
    )
    assert ttnn.synchronize_sub_device_ids == [transfer_sub_device_ids] * 3


def test_host_transfer_waits_for_resident_submission_before_first_cq1_batch(monkeypatch):
    import threading

    monkeypatch.setattr(resident_runner, "HOST_TRANSFER_COUNT", 1)
    monkeypatch.setattr(resident_runner, "HOST_ARITHMETIC_ROUNDS", 1)
    monkeypatch.setattr(resident_runner, "HOST_ARITHMETIC_WORDS", 1)
    ttnn = _FakeHostTransferTtnn()
    device_tensors = []
    host_tensors = []
    started = threading.Event()
    resident_launched = threading.Event()
    result_holder = {}

    def run():
        result_holder.update(
            _run_host_transfers(
                ttnn,
                object(),
                started_event=started,
                resident_launched=resident_launched,
                start_holder={},
                device_tensors_holder=device_tensors,
                host_tensors_holder=host_tensors,
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(timeout=1.0)
    assert ttnn.events == []
    resident_launched.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert result_holder["transfer_count"] == 1
    assert [event[0] for event in ttnn.events] == [
        "host_to_device",
        "device_to_host",
        "synchronize",
    ]
    _complete_host_transfers(ttnn, object(), device_tensors, host_tensors)


def test_host_transfer_runner_queues_256_pairs_per_repeat(monkeypatch):
    ttnn, device_tensors, host_tensors, _started, result = _run_fake_host_transfer(
        monkeypatch,
        repetitions=2,
        transfer_count=256,
    )
    copy_events = [event for event in ttnn.events if event[0] in {"host_to_device", "device_to_host"}]
    assert len([event for event in copy_events if event[0] == "host_to_device"]) == 512
    assert len([event for event in copy_events if event[0] == "device_to_host"]) == 512
    assert result["transfer_count"] == 512
    assert ttnn.events.count(("synchronize", 1)) == 2
    _complete_host_transfers(ttnn, object(), device_tensors, host_tensors)


def test_host_transfer_cleanup_releases_a_partially_used_fixed_pool(monkeypatch):
    ttnn = _FakeHostTransferTtnn(fail_on_copy=2)
    device_tensors = []
    host_tensors = []
    monkeypatch.setattr(resident_runner, "HOST_TRANSFER_COUNT", 5)
    with pytest.raises(RuntimeError, match="injected CQ1 copy failure"):
        _run_host_transfers(
            ttnn,
            object(),
            repetitions=10,
            device_tensors_holder=device_tensors,
            host_tensors_holder=host_tensors,
        )
    _complete_host_transfers(ttnn, object(), device_tensors, host_tensors)
    assert not device_tensors
    assert not host_tensors
    assert len(ttnn.deallocated) == 2 * resident_runner.HOST_TRANSFER_BUFFER_POOL_SIZE


def test_loaded_run_rejects_a_short_outer_timeout_for_calibrated_target(tmp_path):
    calibration = CalibrationMetadata(
        condition="dram",
        duration_seconds=60,
        achieved_throughput=2,
        throughput_unit="repeats_per_card_second",
        repeat_count=1100,
        target_seconds=500,
        safety_factor=1.10,
        load_card_ticks=1_000_000_000,
        load_measurement_repeat_count=2,
        load_interval_aiclk_mhz=1_000,
        load_interval_aiclk_source=CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE,
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


def test_newton_schulz_cap_guard_uses_calibrated_ticks_and_sampler_aiclk(tmp_path):
    calibration = CalibrationMetadata(
        condition="newton_schulz",
        duration_seconds=60,
        achieved_throughput=606.704754,
        throughput_unit="repeats_per_card_second",
        repeat_count=333_688,
        target_seconds=500,
        load_card_ticks=40_052_430_488,
        load_measurement_repeat_count=18_000,
        load_interval_aiclk_mhz=1_191.2587677857841,
        load_interval_aiclk_source=CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE,
    )
    path = tmp_path / "ns-calibration.json"
    path.write_text(json.dumps({"calibration": calibration.as_record()}))
    predicted = predicted_load_time_seconds(calibration)
    assert predicted == pytest.approx(623.2909901739931)
    assert predicted + CALIBRATION_LOAD_TIMEOUT_MARGIN_SECONDS > 600
    assert predicted + CALIBRATION_LOAD_TIMEOUT_MARGIN_SECONDS < 800
    args = SimpleNamespace(
        condition="newton_schulz",
        calibrate=False,
        calibration_json=path,
        repeat_count=None,
        outer_timeout_seconds=600,
    )
    with pytest.raises(ValueError, match="calibrated real load time"):
        _load_execution_selection(args)
    assert args.outer_timeout_seconds == 600
    args.outer_timeout_seconds = 800
    selected_condition, selected_repeat, selected_calibration = _load_execution_selection(args)
    assert selected_condition == "newton_schulz"
    assert selected_repeat == 333_688
    assert selected_calibration is not None
    assert selected_calibration.load_interval_aiclk_source == (
        CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE
    )
    assert selected_calibration.load_card_ticks == 40_052_430_488


def test_host_measurement_is_rejected_for_issue_120_but_calibration_remains_available(
    tmp_path,
):
    calibration = CalibrationMetadata(
        condition="host_cpu_transfer",
        duration_seconds=60,
        achieved_throughput=1,
        throughput_unit="repeats_per_host_second",
        repeat_count=2,
        target_seconds=500,
    )
    path = tmp_path / "host-calibration.json"
    path.write_text(json.dumps({"calibration": calibration.as_record()}))
    args = SimpleNamespace(
        condition="host_cpu_transfer",
        calibrate=False,
        calibration_json=path,
        repeat_count=None,
        outer_timeout_seconds=600,
    )
    with pytest.raises(ValueError, match="#120"):
        _load_execution_selection(args)
    calibration_args = SimpleNamespace(
        condition="host_cpu_transfer",
        calibrate=True,
        calibration_json=None,
        repeat_count=1,
        calibration_duration_seconds=60,
        outer_timeout_seconds=60,
    )
    assert _load_execution_selection(calibration_args) == (
        "host_cpu_transfer",
        1,
        None,
    )


def test_host_load_clock_uses_launch_once_and_fresh_endpoint():
    import threading

    start_holder = {}
    clock = _host_load_clock(
        stop_event=threading.Event(),
        start_holder=start_holder,
        monotonic_ns=iter((250, 400)).__next__,
    )
    assert clock() == 250
    assert start_holder["start_ns"] == 250
    assert clock() == 400
    assert start_holder["start_ns"] == 250


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
    assert "copy_host_to_device_tensor" in resident_source
    assert "copy_device_to_host_tensor" in resident_source
    assert "blocking=False" in resident_source
    assert "cq_id=TRANSFER_COMMAND_QUEUE" in resident_source
    assert resident_source.rindex("host_thread.join") < resident_source.rindex("_complete_host_transfers")
    assert 'record["timing_evidence"] = False' in resident_source
    assert "load_record=resident_load_record" in resident_source
    assert "if calibration_output is None:" in resident_source
    assert resident_source.index("record = build_measurement_record") < resident_source.index(
        "if calibration_output is not None:"
    )
    assert 'record["status"] = "calibration"' in resident_source
    host_runner_start = resident_source.index("def _run_host_transfers(")
    host_runner_end = resident_source.index("def _host_join_timeout_seconds(")
    host_runner_source = resident_source[host_runner_start:host_runner_end]
    assert host_runner_source.index("started_event.set()") < host_runner_source.index(
        "_run_host_transfer_repetitions("
    )
    transfer_source = resident_source[
        resident_source.index("def _run_host_transfer_repetitions(") : host_runner_end
    ]
    assert transfer_source.index("_run_host_arithmetic_repeat(") < transfer_source.index(
        "_wait_for_resident_launch"
    )
    assert transfer_source.index("_wait_for_resident_launch") < transfer_source.index(
        "_queue_host_transfer_batch("
    )
    assert transfer_source.index("_queue_host_transfer_batch(") < transfer_source.index(
        "_synchronize_transfer_queue("
    )
    assert "if transfer_index and slot == 0" not in resident_source
    assert resident_source.index("started_event.set()") < resident_source.index("ttnn.generic_op(_merged_program_io_tensors")
    assert resident_source.index("measurement_start_ns = time.monotonic_ns()") < resident_source.index("ttnn.generic_op(_merged_program_io_tensors")
    assert resident_source.index("ttnn.generic_op(_merged_program_io_tensors") < resident_source.rindex("resident_launched.set()")
    assert resident_source.index("measurement_end_ns = time.monotonic_ns()") < resident_source.index("host_thread.join")
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
    duplicate_ttnn = Ttnn()
    duplicate_first = Program(20, "a")
    duplicate_second = Program(20, "b")
    with pytest.raises(ValueError, match="collision"):
        load_generators.merge_program_descriptors(
            duplicate_ttnn, duplicate_first, duplicate_second
        )


def test_fixed_dram_and_host_work_metadata():
    transfers = []
    host_run = load_generators.run_host_cpu_transfer_work(
        clock_ns=iter((100, 250)).__next__,
        transfer=lambda payload, queue: transfers.append((len(payload), queue)),
    )
    assert host_run["start_ns"] == 100
    assert host_run["end_ns"] == 250
    assert transfers == [(64 * 1024, 1)] * 256
    dram = load_generators.dram_load_work()
    assert dram["bank_count"] == 8
    assert dram["read_pages"] == dram["write_pages"] == 8 * 256
    assert dram["data_dependent_exit"] is False
    host = load_generators.host_cpu_transfer_work()
    assert host["command_queue"] == 1
    assert host["arithmetic_rounds"] == 4_096
    assert host["arithmetic_words"] == 4_096
    assert host["transfer_bytes"] == 64 * 1024
    assert host["transfer_count"] == 256
    assert host["transfers"] == "fixed_size_host_device_transfers"
    assert host["buffer_pool_size"] == 4
    assert host["buffer_pool"] == "reusable_host_and_device_buffers"
    assert host["resident_sub_device_id"] == HOST_RESIDENT_SUB_DEVICE_ID
    assert host["transfer_sub_device_id"] == HOST_TRANSFER_SUB_DEVICE_ID
    assert host["transfer_core"] == list(HOST_TRANSFER_CORE)
    assert host["completion_scope"] == "transfer_sub_device_only"


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


def test_host_calibration_may_be_uncovered_without_becoming_timing_evidence():
    calibration = CalibrationMetadata(
        condition="host_cpu_transfer",
        duration_seconds=60,
        achieved_throughput=1,
        throughput_unit="repeats_per_host_second",
        repeat_count=2,
        target_seconds=500,
    )
    record = build_calibration_load_record(
        condition="host_cpu_transfer",
        repeat_count=1,
        resident_timestamps=None,
        host_interval=HostLoadInterval(90, 110),
        measurement_start_ns=100,
        measurement_end_ns=300,
        calibration=calibration,
        achieved_throughput=1,
        throughput_unit="repeats_per_host_second",
    )
    assert record["status"] == "calibration"
    assert record["valid_evidence"] is False
    assert record["coverage"]["valid"] is True
    assert record["coverage"]["covered"] is False
    assert record["coverage"]["verdict"] == "uncovered"


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
    invalid_dram_interval = CardLoadInterval(101, 310)
    with pytest.raises(LoadEvidenceError):
        build_load_record(
            condition="dram",
            repeat_count=1,
            resident_timestamps=[100, 200, 300],
            card_interval=invalid_dram_interval,
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
    invalid_host_interval = HostLoadInterval(101, 310)
    with pytest.raises(LoadEvidenceError):
        build_load_record(
            condition="host_cpu_transfer",
            repeat_count=7,
            resident_timestamps=None,
            host_interval=invalid_host_interval,
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
        "host_work": {"transfer_count": 256},
    }
    assert _load_achieved_throughput(
        ns_result, condition="newton_schulz", repeat_count=1, aiclk_mhz=1000
    ) == (1350.0, "repeats_per_card_second")
    assert _load_achieved_throughput(
        dram_result, condition="dram", repeat_count=1, aiclk_mhz=1000
    ) == (1350.0, "repeats_per_card_second")
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


def test_power_trace_keeps_varying_in_run_clock_and_uses_time_weighted_interval(tmp_path):
    trace = tmp_path / "power.csv"
    trace.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00Z,75,1000,60\n"
        "2026-01-01T00:00:01Z,75,800,60\n"
        "2026-01-01T00:00:03Z,75,1200,60\n"
    )
    parsed = read_power_trace_csv(
        trace,
        run_start="2026-01-01T00:00:00Z",
        run_end="2026-01-01T00:00:03Z",
    )
    assert parsed["aiclk_mhz"] == 1200
    assert parsed["aiclk_mhz_in_run_min"] == 800
    assert parsed["aiclk_mhz_in_run_max"] == 1200
    assert parsed["aiclk_mhz_load_interval_weighted"] == pytest.approx(866.6666667)
    assert parsed["aiclk_mhz_load_interval_weighted_source"] == "load_interval_trace_samples"
    assert len(parsed["in_run_samples"]) == 3


def test_loaded_pair_validation_reuses_weighted_elapsed_aiclk():
    intervals = [1_000_000, 1_700_000] + [1_350_000] * 2_698 + [1_000_000, 1_700_000]
    timestamps = [0]
    for interval in intervals:
        timestamps.append(timestamps[-1] + interval)
    outlier = build_outlier_analysis(
        timestamps,
        aiclk_mhz=1_350,
        elapsed_aiclk_mhz=1_100.5,
        elapsed_aiclk_source="load_interval_trace_samples",
        sampler_metadata={"mode": "default", "interval_seconds": 2.0},
    )
    assert validate_pair_analysis(
        outlier,
        timestamps=timestamps,
        aiclk_mhz=1_350,
        sampler_mode="default",
        sampler_interval_seconds=2.0,
    ) == []


def test_wait_for_power_trace_sample_accepts_completion_edge(tmp_path):
    trace = tmp_path / "power.csv"
    trace.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00Z,75,1350,60\n"
        "2026-01-01T00:00:02Z,75,1100,70\n"
    )
    _wait_for_power_trace_sample(
        trace,
        "2026-01-01T00:00:02+00:00",
        sampler_interval_seconds=2.0,
    )


def test_delayed_worker_uses_slowest_worker_interval_not_aggregate_span():
    result = {
        "card_load_interval": CardLoadInterval(0, 10_000),
        "card_load_worker_intervals": [
            {"worker_index": 0, "start_ticks": 0, "end_ticks": 1_000},
            {"worker_index": 1, "start_ticks": 9_000, "end_ticks": 11_000},
        ],
    }
    throughput, unit = _load_achieved_throughput(
        result, condition="dram", repeat_count=1, aiclk_mhz=1_000
    )
    assert throughput == pytest.approx(675_000.0)
    assert unit == "repeats_per_card_second"
    assert result["throughput_interval"]["source"] == (
        "slowest_worker_own_elapsed_interval"
    )


def test_dram_record_retains_all_worker_rows_and_aggregate_coverage_span():
    workers = [
        CardWorkerInterval(index, index * 10, 300 + index * 10)
        for index in range(8)
    ]
    record = build_load_record(
        condition="dram",
        repeat_count=1,
        resident_timestamps=[100, 200],
        card_interval=CardLoadInterval(0, 370),
        worker_intervals=workers,
    )
    assert len(record["card_load_worker_intervals"]) == 8
    assert record["card_load_interval_statistics"]["aggregate_span_role"] == "coverage_only"
    assert record["card_load_interval_statistics"]["slowest_worker_elapsed_ticks"] == 300
    validate_load_record(record)


def test_calibration_record_is_schema_valid_and_annotates_card_tick_unit():
    calibration = CalibrationMetadata(
        condition="dram",
        duration_seconds=5,
        achieved_throughput=2,
        throughput_unit="repeats_per_card_second",
        repeat_count=12,
        target_seconds=500,
    )
    record = build_calibration_load_record(
        condition="dram",
        repeat_count=1,
        resident_timestamps=[100, 200],
        card_interval=CardLoadInterval(0, 300),
        calibration=calibration,
        achieved_throughput=2,
        throughput_unit="repeats_per_card_second",
    )
    validate_calibration_load_record(record)
    assert record["calibration"]["throughput_unit_definition"] == CARD_THROUGHPUT_UNIT_DEFINITION


def test_resident_record_persists_ordered_host_monotonic_program_edges():
    config = ResidentConfig(
        frame_count=3,
        frame_interval_ticks=1_000,
        producer_core=(0, 0),
        consumer_core=(1, 0),
        ring_pages=4,
        work_per_frame=64,
        designated_timestamp_core=(1, 0),
        cycle_budget=10_000,
        outer_timeout_seconds=60,
        fixed_work_ticks_per_frame=100,
        budget_aiclk_mhz=1_350,
    )
    sampler = {
        "mode": "off",
        "interval_seconds": None,
        "power_trace": "absent_by_design",
        "timing_evidence": "diagnostic_only",
    }
    environment = {
        "board": {"serial": "board", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kmd_version": "2.11.0",
        "image": AUDITED_CLOCK_SOURCE_IMAGE,
        "image_pinned": True,
        "harness_commit": "a" * 40,
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": sampler,
    }
    record = build_measurement_record(
        config=config,
        aiclk_mhz=1_350,
        timestamps=[100, 1_100, 2_100],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=environment,
        power_trace=None,
        program_launch_ns=10_000,
        program_completion_ns=20_000,
    )
    assert record["program_host_interval"] == {
        "clock": "host_monotonic_ns",
        "start_ns": 10_000,
        "end_ns": 20_000,
        "launch_ns": 10_000,
        "completion_ns": 20_000,
        "elapsed_ns": 10_000,
    }
    assert record["resident_tick_interval"]["elapsed_ticks"] == 2_000
    with pytest.raises(ValueError, match="ordered non-negative monotonic"):
        build_measurement_record(
            config=config,
            aiclk_mhz=1_350,
            timestamps=[100, 1_100, 2_100],
            producer_full_count=0,
            consumer_empty_count=0,
            kernel_error_flag=0,
            harness_commit="a" * 40,
            environment=environment,
            power_trace=None,
            program_launch_ns=20_000,
            program_completion_ns=10_000,
        )


def test_compute_timeout_cap_is_selectable_at_600_or_800_and_watcher_stays_60():
    args = resident_parser().parse_args(
        [
            "--out",
            "/out/result.json",
            "--condition",
            "newton_schulz",
            "--outer-timeout-seconds",
            "800",
        ]
    )
    config = _config_from_args(args)
    assert config.outer_timeout_cap_seconds == 800
    assert config.outer_timeout_cap_source == "issue118_compute_load_800s"
    assert resident_runner.validate_configuration(config).outer_timeout_cap_seconds == 800
    with pytest.raises(ValueError, match="60"):
        resident_runner.validate_configuration(config, watcher=True)
    args.outer_timeout_seconds = 600
    config = _config_from_args(args)
    assert config.outer_timeout_cap_seconds == 600
    assert config.outer_timeout_cap_source == "issue12_timing_600s"


def test_old_committed_record_remains_out_of_scope_valid():
    path = ROOT / "docs/measurements/2026-10-06-p150a-issue104-sampler-default.json"
    report = validate_resident_record(json.loads(path.read_text()))
    assert report["out_of_scope"] is True
    assert report["valid"] is None
