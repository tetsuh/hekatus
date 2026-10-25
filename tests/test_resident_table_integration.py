"""Wire protocol and runner tests independent of TTNN and physical devices."""

import copy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from enodia.tt.bench import run_resident
from enodia.tt.bench.resident_harness import (
    build_measurement_record,
    build_rejection_record,
    validate_resident_record,
)
from enodia.tt.bench.resident_tables import (
    build_table_evidence,
    frame_checksum,
    identity_at,
    ready_index,
    validate_table_evidence,
)
from tests.test_resident_tables import config, tables


def outputs(cfg):
    rows, timestamps = [], []
    summary = [0] * 1024
    summary[3] = cfg.frame_count
    summary[4] = 1
    summary[6] = 1
    summary[13] = 1
    summary[15] = 2
    summary[17] = 1
    for frame in range(cfg.frame_count):
        identity, segment = identity_at(cfg, frame)
        base = 20 + 4 * segment
        if ready_index(cfg, frame, identity) is None:
            summary[18] += 1
            summary[base] += 1
        else:
            timestamp = (frame + 1) * 1000
            if not summary[base + 3]:
                summary[base + 1] = timestamp - timestamps[-1] if timestamps else 0
                summary[base + 3] = 2 if timestamps else 1
            rows.append([frame_checksum(cfg, frame, identity), 1, frame, *identity, segment])
            timestamps.append(timestamp)
    summary[1] = len(rows)
    summary[19] = cfg.frame_count
    return rows, timestamps, summary


def environment():
    return {
        "board": {"serial": "redacted", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"}, "kmd_version": "2.11.0",
        "image": "registry.example/tt@sha256:" + "a" * 64,
        "image_pinned": True, "harness_commit": "0123456789abcdef",
        "harness_dirty": False, "harness_pushed": True,
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": {"mode": "off"},
    }


def record(cfg, evidence, timestamps):
    return build_measurement_record(
        config=cfg, aiclk_mhz=1350, timestamps=timestamps, producer_full_count=0,
        consumer_empty_count=0, kernel_error_flag=0, harness_commit="0123456789abcdef",
        environment=environment(), power_trace=None, table_swap=evidence,
    )


@pytest.mark.parametrize("strategy", ["copy_on_demand", "prefetch"])
def test_decode_and_record_keep_discards_separate_and_verify_every_frame(monkeypatch, strategy):
    cfg = tables(swap_strategy=strategy)
    rows, timestamps, summary = outputs(cfg)
    tensor = np.zeros((cfg.frame_count, 1, 1, 1024), dtype=np.uint32)
    for index, (row, timestamp) in enumerate(zip(rows, timestamps, strict=True)):
        tensor[index, 0, 0, :8] = [timestamp, 0, *row]
    producer = np.zeros(1024, dtype=np.uint32)
    producer[1] = producer[3] = cfg.frame_count
    tensors = {"timestamps": tensor, "producer_stats": producer,
               "consumer_stats": np.array(summary, dtype=np.uint32)}
    monkeypatch.setattr(run_resident, "_download", lambda _, tensor: tensor)
    decoded = run_resident._decode_resident_result(None, cfg, tensors)
    assert decoded["frames_consumed"] == 17
    assert decoded["frames_dropped"] == decoded["producer_full_count"] == 0
    evidence = decoded["table_swap"]
    assert evidence["discard_count"] == 3
    assert evidence["checksum"]["checked_frames"] == 17
    assert evidence["checksum"]["passed"] is True
    assert [row["switch_frame_interval_ticks"] for row in evidence["switches"]] == [4000, 1000]
    built = record(cfg, evidence, timestamps)
    assert built["status"] == "ok_with_drops"
    assert built["termination_reason"] == "frame_count"
    assert built["ring"]["retired_frame_count"] == 20
    assert built["ring"]["consumed_frame_count"] == 17
    assert built["raw_timestamps"]["count"] == 17
    assert validate_resident_record(built, timestamps=timestamps)["valid"]


def test_all_unknown_frames_discard_without_timestamp_or_fake_interval():
    cfg = config(switch_schedule=((0, 9, 9),), table_bytes=4096)
    rows, timestamps, summary = outputs(cfg)
    evidence = build_table_evidence(cfg, rows, timestamps, summary)
    assert timestamps == []
    assert evidence["discard_count"] == cfg.frame_count
    assert evidence["switches"][0]["switch_frame_interval_ticks"] is None
    assert record(cfg, evidence, timestamps)["status"] == "ok_with_drops"


def test_checksum_failure_is_recorded_not_treated_as_success():
    cfg = tables()
    rows, timestamps, summary = outputs(cfg)
    rows[5][0] ^= 1
    evidence = build_table_evidence(cfg, rows, timestamps, summary)
    assert evidence["checksum"]["mismatch_count"] == 1
    assert evidence["checksum"]["passed"] is False
    assert record(cfg, evidence, timestamps)["status"] == "error"


@pytest.mark.parametrize("field,value", [("discard_count", 1), ("retired_frame_count", 19),
                                        ("initial_discard_count", True)])
def test_corrupt_card_counters_rejected(field, value):
    cfg = tables()
    rows, timestamps, summary = outputs(cfg)
    evidence = build_table_evidence(cfg, rows, timestamps, summary)
    evidence[field] = value
    with pytest.raises(ValueError):
        validate_table_evidence(cfg, evidence, len(rows))


@pytest.mark.parametrize("word", [2, 3, 4, 5])
def test_corrupt_stamped_frame_or_identity_rejected(word):
    cfg = tables()
    rows, timestamps, summary = outputs(cfg)
    rows[5][word] += 1
    with pytest.raises(ValueError):
        build_table_evidence(cfg, rows, timestamps, summary)


def test_persisted_record_rejects_tampered_discard_and_schedule():
    cfg = tables()
    rows, timestamps, summary = outputs(cfg)
    built = record(cfg, build_table_evidence(cfg, rows, timestamps, summary), timestamps)
    for mutate in [
        lambda item: item["ring"].update(discard_count=0),
        lambda item: item["table_swap"]["switches"][0].update(identity=[9, 0]),
        lambda item: item["parameters"].update(switch_schedule=[]),
        lambda item: item.pop("table_swap"),
    ]:
        corrupt = copy.deepcopy(built)
        mutate(corrupt)
        assert not validate_resident_record(corrupt, timestamps=timestamps)["valid"]


def test_cli_round_trips_schedule_and_profiles():
    args = run_resident._parser().parse_args([
        "--table-bytes", "8192", "--profile-table", "7:3:0",
        "--profile-table", "8:3:15", "--initial-identity", "7:3",
        "--switch", "10:8:3", "--swap-strategy", "prefetch",
    ])
    cfg = run_resident._config_from_args(args)
    assert cfg.table_bytes == 8192
    assert cfg.profile_tables == ((7, 3, 0), (8, 3, 15))
    assert cfg.switch_schedule == ((10, 8, 3),)
    assert cfg.initial_identity == (7, 3)
    assert cfg.swap_strategy == "prefetch"
    parser = run_resident._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--switch", "1:2"])


def test_oversize_rejection_record_retains_reason_and_configuration():
    cfg = tables()
    built = build_rejection_record(config=cfg, reason="L1 capacity exceeded", environment=environment())
    assert built["status"] == "rejected"
    assert built["parameters"]["switch_schedule"] == cfg.switch_schedule
    assert "L1" in built["rejection_reason"]


class FakeTT:
    uint32 = "uint32"
    ROW_MAJOR_LAYOUT = "row-major"
    DRAM_MEMORY_CONFIG = "dram"
    CoreType = SimpleNamespace(WORKER="worker")
    TileDescriptor = staticmethod(lambda *args: args)
    CBFormatDescriptor = staticmethod(lambda **kwargs: kwargs)
    CBDescriptor = staticmethod(lambda **kwargs: kwargs)
    ReaderConfigDescriptor = staticmethod(dict)
    SemaphoreDescriptor = staticmethod(lambda *args: args)
    ProgramDescriptor = staticmethod(lambda **kwargs: kwargs)
    KernelDescriptor = SimpleNamespace(SourceType=SimpleNamespace(FILE_PATH="file"))

    class RuntimeArgs(defaultdict):
        def __init__(self):
            super().__init__(dict)

    def __init__(self):
        self.uploads = []
        self.events = []
        self.KernelDescriptor = lambda **kwargs: kwargs
        self.KernelDescriptor.SourceType = SimpleNamespace(FILE_PATH="file")

    def TensorAccessorArgs(self, tensor):
        return SimpleNamespace(get_compile_time_args=lambda: [tensor.buffer_address()])

    def Tensor(self, values, dtype):
        assert dtype == self.uint32
        self.uploads.append(values.copy())
        return values

    def to_device(self, host, device, *, memory_config):
        assert memory_config == "dram"
        self.events.append("upload")
        address = 1000 + len(self.uploads)
        return SimpleNamespace(buffer_address=lambda: address)

    def synchronize_device(self, device, *, cq_id):
        assert cq_id == 0
        self.events.append("synchronize")


def test_preload_and_single_program_descriptor_use_dram_accessors(monkeypatch):
    fake = FakeTT()
    monkeypatch.setattr(run_resident, "_allocate", lambda *args: SimpleNamespace(buffer_address=lambda: 100))
    fake.zeros = lambda *args, **kwargs: SimpleNamespace(buffer_address=lambda: 100)
    fake.Shape = lambda shape: shape
    monkeypatch.setattr(run_resident, "_sharded_pages_config", lambda *args, **kwargs: "sharded")
    monkeypatch.setattr(run_resident, "_core_range", lambda *args: "core-range")
    cfg = tables(swap_strategy="prefetch")
    tensors = run_resident._allocate_resident_tensors(fake, None, cfg)
    assert fake.events == ["upload", "upload", "synchronize"]
    assert fake.uploads[0].shape == (1, 1, 1024)
    assert fake.uploads[1].shape == (1, 3, 1024)
    assert fake.uploads[1][0, 0, 0] != fake.uploads[1][0, 1, 0]
    program = run_resident._program(fake, cfg, tensors, watcher=False)
    assert len(program["kernels"]) == 2
    assert len(program["semaphores"]) == 4
    assert [cb["format_descriptors"][0]["buffer_index"] for cb in program["cbs"]] == list(range(20, 26))
    assert program["cbs"][4]["total_size"] == 2 * cfg.table_bytes
    producer, consumer = program["kernels"]
    assert producer["runtime_args"][0][0][8:] == [1, 1001]
    assert consumer["runtime_args"][1][0][10:] == [1, 1001, 1002, 1, 1]
    assert producer["compile_time_args"][-1] == 1001
    assert consumer["compile_time_args"][-2:] == [1001, 1002]


def test_kernel_contract_copies_are_budgeted_and_discards_retire_without_timestamp():
    root = Path("enodia/tt/bench/kernels")
    producer = (root / "resident_producer.cpp").read_text()
    consumer = (root / "resident_consumer.cpp").read_text()
    common = (root / "resident_tables.h").read_text()
    assert "a.config == b.config && a.generation == b.generation" in common
    assert "page < pages" in common
    assert "noc_async_read_barrier();" in common
    assert producer.index("attempted >= metadata") < producer.index("if (slot_full)")
    for word in range(4):
        assert f"source[{word}] =" in producer
    start = consumer.index("const std::uint64_t start = clock.read();")
    end = consumer.index("const std::uint64_t end = clock.read();")
    assert consumer.count("resident_tables::copy_table(") == 2
    assert all(start < position < end for position in [
        consumer.index("resident_tables::copy_table("),
        consumer.rindex("resident_tables::copy_table("),
    ])
    assert "selected_table[work % (table_pages * page_words)]" in consumer
    discard = consumer[consumer.index("if (discard) {"):consumer.index("previous_completion = end;")]
    assert "discard_count += 1" in discard
    assert "frames_retired += 1" in discard
    assert "noc_semaphore_inc(free_noc, 1)" in discard
    assert "continue;" in discard
    assert "noc_async_write_page" not in discard
    assert "elapsed >= static_cast<std::uint64_t>(per_frame_work_budget_ticks)" in consumer
    assert "failure_code = failure_consumer_fixed_work_budget;" in consumer


def test_default_program_has_only_legacy_buffers_and_cbs(monkeypatch):
    fake = FakeTT()
    monkeypatch.setattr(run_resident, "_core_range", lambda *args: "core-range")
    tensors = {key: SimpleNamespace(buffer_address=lambda: 100) for key in (
        "ring", "control", "producer_anchor", "producer_stats", "consumer_stats", "timestamps",
    )}
    program = run_resident._program(fake, config(), tensors, watcher=False)
    assert len(program["cbs"]) == 2
    assert program["kernels"][0]["runtime_args"][0][0][8] == 0
    assert program["kernels"][1]["runtime_args"][1][0][10] == 0
