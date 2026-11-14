"""Board-free contracts for Issue #120 feasibility probes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from enodia.tt.bench.issue120_probe import (
    ISSUE120_PROBES,
    PINNED_IMAGE_DIGEST,
    ProbeRecordError,
    allocate_low_word_clock,
    card_tick_interval_relation,
    card_tick_window_relation,
    planned_probe,
    validate_probe_record,
)
from enodia.tt.bench.load_harness import (
    HOST_TRANSFER_BUFFER_POOL_SIZE,
    HOST_TRANSFER_BYTES,
    P150_RESIDUAL_WORKER_COORDINATES,
    RESIDENT_CORES,
)


def test_probe_inventory_is_fixed_capped_and_core_disjoint() -> None:
    assert {probe.probe_id for probe in ISSUE120_PROBES} == {
        "q1-cq1-completion",
        "q2-h2d-write-only",
        "q3-pinned-noc",
        "q4-fixed-window",
    }
    for probe in ISSUE120_PROBES:
        assert probe.outer_cap_seconds <= 60
        assert probe.repeat_count > 0
        assert probe.transfer_bytes > 0
        assert probe.resident_cores == RESIDENT_CORES
        assert set(probe.load_cores).isdisjoint(probe.resident_cores)
        assert set(probe.load_cores).issubset(P150_RESIDUAL_WORKER_COORDINATES)


def test_first_kernel_run_requires_watcher_and_sixty_second_cap() -> None:
    first_run = planned_probe("q3-pinned-noc", first_kernel_run=True)
    assert first_run.watcher is True
    assert first_run.outer_cap_seconds == 60

    repeat = planned_probe("q3-pinned-noc", first_kernel_run=False)
    assert repeat.watcher is False
    assert repeat.outer_cap_seconds <= 60


def test_repeat_count_does_not_change_buffer_or_inflight_bounds() -> None:
    short = planned_probe("q2-h2d-write-only", repeat_count=1)
    long = planned_probe("q2-h2d-write-only", repeat_count=100_000)
    assert short.buffer_pool_size == HOST_TRANSFER_BUFFER_POOL_SIZE
    assert long.buffer_pool_size == short.buffer_pool_size
    assert short.allocated_bytes == long.allocated_bytes
    assert long.max_outstanding_transfers is None
    assert "runtime_queue_backpressure" in long.allocation_policy
    assert planned_probe("q1-cq1-completion").repeat_count == 1
    q4 = planned_probe("q4-fixed-window")
    assert q4.repeat_count == 256
    assert q4.host_cpu_rounds * q4.host_cpu_words_per_round == 1_048_576
    assert long.allocated_bytes == (
        HOST_TRANSFER_BYTES * (1 + HOST_TRANSFER_BUFFER_POOL_SIZE) + long.auxiliary_bytes
    )


def test_low_word_clock_tracks_wrap_without_timestamp_api() -> None:
    clock = allocate_low_word_clock()
    assert clock.observe(0xFFFFFFFE) == 0xFFFFFFFE
    assert clock.observe(0xFFFFFFFF) == 0xFFFFFFFF
    assert clock.observe(0) == 1 << 32
    assert clock.observe(3) == (1 << 32) + 3
    with pytest.raises(ProbeRecordError, match="gap"):
        clock.observe((1 << 31) + 4)


@pytest.mark.parametrize(
    ("completion_ticks", "expected"),
    [
        (99, "inside_resident_window"),
        (100, "inside_resident_window"),
        (101, "after_resident_window"),
        (None, "not_evaluable"),
    ],
)
def test_q1_relation_uses_card_completion_tick(completion_ticks, expected) -> None:
    assert card_tick_window_relation(
        resident_start_ticks=50,
        resident_end_ticks=100,
        transfer_completion_ticks=completion_ticks,
    ) == expected


def test_completion_bracket_must_not_guess_across_resident_boundary() -> None:
    assert card_tick_interval_relation(
        resident_start_ticks=10,
        resident_end_ticks=20,
        completion_lower_ticks=9,
        completion_upper_ticks=11,
    ) == "not_evaluable"
    assert card_tick_interval_relation(
        resident_start_ticks=10,
        resident_end_ticks=20,
        completion_lower_ticks=12,
        completion_upper_ticks=14,
    ) == "inside_resident_window"
    assert card_tick_interval_relation(
        resident_start_ticks=10,
        resident_end_ticks=20,
        completion_lower_ticks=22,
        completion_upper_ticks=24,
    ) == "after_resident_window"


def test_record_requires_machine_environment_and_load_evidence() -> None:
    record = {
        "schema": "issue120-probe-v1",
        "source": {
            "tt_metal_tag": "v0.75.0",
            "tt_metal_commit": "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd",
            "libtt_metal_sha256": "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141",
        },
        "probe_id": "q2-h2d-write-only",
        "status": "complete",
        "environment": {
            "board_model": "p150a",
            "serial": "serial-placeholder",
            "pci": "0000:00:00.0",
            "firmware": "19.6.0.0",
            "kmd": "2.11.0",
            "image_digest": PINNED_IMAGE_DIGEST,
            "cpu_model": "CPU model placeholder",
            "commit": "b" * 40,
        },
        "resident_interval": {"start_ticks": 10, "end_ticks": 20},
        "host_interval": {"start_ns": 100, "end_ns": 200},
        "work": {
            "repeat_count": 3,
            "transfer_count": 3,
            "bytes_per_transfer": 65536,
            "bytes_total": 196608,
        },
        "memory": {
            "buffer_pool_size": 4,
            "allocated_bytes": 339968,
            "host_payload_bytes": 65536,
            "device_buffer_bytes": 262144,
            "auxiliary_bytes": 12288,
            "max_outstanding_transfers": None,
            "outstanding_bound": "fixed CQ1 ring with enqueue backpressure",
        },
        "outer_cap_seconds": 60,
        "watcher": False,
        "first_kernel_run": False,
        "plan": {
            "probe_id": "q2-h2d-write-only",
            "repeat_count": 3,
            "transfer_bytes": 65536,
            "allocated_bytes": 339968,
            "buffer_pool_size": 4,
            "max_outstanding_transfers": None,
            "resident_cores": [[0, 0], [1, 0]],
            "load_cores": [],
            "outer_cap_seconds": 60,
            "watcher": False,
            "first_kernel_run": False,
        },
        "final_drain_interval": {"start_ns": 200, "end_ns": 210},
        "errors": [],
        "host_buffer_mode": "ordinary_unpinned_immutable_raw_host_pointer",
        "explicit_completion_queue_reads_during_submission": False,
        "completion_reads_during_final_drain": True,
        "host_readback_during_submission": False,
        "host_enqueue_may_backpressure": True,
        "device_wait_stream_may_stall": True,
    }
    assert validate_probe_record(record) == []
    record["work"]["transfer_count"] = 12
    record["work"]["bytes_total"] = 786432
    with pytest.raises(ProbeRecordError, match="fixed work"):
        validate_probe_record(record)
    record["work"]["transfer_count"] = 3
    record["work"]["bytes_total"] = 196608

    del record["environment"]["serial"]
    with pytest.raises(ProbeRecordError, match="serial"):
        validate_probe_record(record)


def test_record_rejects_host_only_q1_completion_claim() -> None:
    record = {
        "schema": "issue120-probe-v1",
        "source": {
            "tt_metal_tag": "v0.75.0",
            "tt_metal_commit": "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd",
            "libtt_metal_sha256": "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141",
        },
        "probe_id": "q1-cq1-completion",
        "status": "complete",
        "environment": {
            "board_model": "p150a",
            "serial": "serial-placeholder",
            "pci": "0000:00:00.0",
            "firmware": "19.6.0.0",
            "kmd": "2.11.0",
            "image_digest": PINNED_IMAGE_DIGEST,
            "cpu_model": "CPU model placeholder",
            "commit": "b" * 40,
        },
        "resident_interval": {"start_ticks": 10, "end_ticks": 20},
        "host_interval": {"start_ns": 100, "end_ns": 200},
        "work": {
            "repeat_count": 1,
            "transfer_count": 1,
            "bytes_per_transfer": 65536,
            "bytes_total": 65536,
        },
        "memory": {
            "buffer_pool_size": 4,
            "allocated_bytes": 339968,
            "host_payload_bytes": 65536,
            "device_buffer_bytes": 262144,
            "auxiliary_bytes": 12288,
            "max_outstanding_transfers": 1,
        },
        "outer_cap_seconds": 60,
        "watcher": True,
        "first_kernel_run": True,
        "plan": {
            "probe_id": "q1-cq1-completion",
            "repeat_count": 1,
            "transfer_bytes": 65536,
            "allocated_bytes": 339968,
            "buffer_pool_size": 4,
            "max_outstanding_transfers": 1,
            "resident_cores": [[0, 0], [1, 0]],
            "load_cores": [[2, 0]],
            "outer_cap_seconds": 60,
            "watcher": True,
            "first_kernel_run": True,
        },
        "errors": [],
        "transfer_completion_ticks": None,
        "transfer_completion_lower_ticks": None,
        "transfer_completion_upper_ticks": None,
        "transfer_completion_relation": "not_evaluable",
    }
    with pytest.raises(ProbeRecordError, match="card-side"):
        validate_probe_record(record)


def test_one_payload_witness_cannot_prove_full_window_coverage() -> None:
    from enodia.tt.bench.issue120_probe import fixed_window_coverage

    assert fixed_window_coverage(
        cpu_start_ns=11, cpu_end_ns=19, resident_start_ns=10, resident_end_ns=20,
    ) == "not_covered"
    assert fixed_window_coverage(
        cpu_start_ns=9, cpu_end_ns=21, resident_start_ns=10, resident_end_ns=20,
    ) == "not_evaluable_pcie_progress_not_observed"


def test_environment_is_validated_before_any_native_execution(monkeypatch) -> None:
    import argparse

    from enodia.tt.bench import issue120_probe as probe

    def reject_environment(_path):
        raise ProbeRecordError("environment rejected before board access")

    def forbidden_execution(*_args, **_kwargs):
        pytest.fail("a subprocess ran before provenance validation")

    monkeypatch.setenv("HEKATUS_TT_PINNED_CONTAINER", "1")
    monkeypatch.setattr(probe, "_environment_record", reject_environment)
    monkeypatch.setattr(probe.subprocess, "run", forbidden_execution)
    args = argparse.Namespace(
        execute=True, probe="q2-h2d-write-only", repeat_count=1,
        cap_seconds=30, first_kernel_run=False,
    )
    with pytest.raises(ProbeRecordError, match="before board access"):
        probe._run_native_probe(args)


def test_caps_reject_any_probe_that_exceeds_sixty_seconds() -> None:
    with pytest.raises(ProbeRecordError, match="60-second"):
        planned_probe("q1-cq1-completion", outer_cap_seconds=61)


def test_native_probe_source_keeps_q2_unpinned_and_q3_noc_pinned() -> None:
    root = Path(__file__).parents[1]
    host = (root / "enodia/tt/bench/issue120/issue120_probe.cpp").read_text()
    noc = (root / "enodia/tt/bench/issue120/pinned_noc_load.cpp").read_text()
    observer = (root / "enodia/tt/bench/issue120/resident_observer.cpp").read_text()
    q2_body = host.split("Json run_host_transfer_probe", 1)[1].split(
        "Json run_pinned_noc_probe", 1
    )[0]
    assert "source.data(), false" in q2_body
    assert "cq1.finish()" in q2_body
    assert "PinnedMemory::Create" not in q2_body
    assert "PinnedMemory::Create" in host
    assert "std::aligned_alloc" in host
    assert "read_payload_matches &= final_host[word] == (*host_words)[word]" not in host
    assert "get_noc_addr" in host
    assert "noc_read_with_state" in noc
    assert "noc_wwrite_with_state" in noc
    assert "get_timestamp_32b" in noc
    assert "get_timestamp_32b" in observer
    assert not re.search(r"\bget_timestamp\s*\(", noc + observer)
    assert "ncrisc_noc_reads_flushed" in noc
    assert "ncrisc_noc_reads_flushed" in observer
    assert "last_negative_read_start" in observer
    assert "words[word] == expected" in observer
    assert "page_count" in observer
    assert "observer_deadline" in observer
    assert "CQNocFlags" in noc
    assert "max_total_pin_size" in host
    assert "has_device" in host
    assert "get_noc_addr" in host
    assert "noc_addr->device_id" in host
    assert "load_words[4] == args.repeat_count" in host
    assert "load_words[5] == args.repeat_count" in host


def test_observer_uses_concrete_accessor_templates_and_supported_write_barrier() -> None:
    source = (Path(__file__).parents[1] / "enodia/tt/bench/issue120/resident_observer.cpp").read_text()
    assert "const TensorAccessor&" not in source
    assert "ncrisc_noc_writes_flushed" not in source
    assert "ncrisc_noc_nonposted_writes_flushed" in source
    assert "const uint64_t program_start = clock.extended;" in source


def test_native_repeat_loop_reuses_allocated_device_pool() -> None:
    source = (Path(__file__).parents[1] / "enodia/tt/bench/issue120/issue120_probe.cpp").read_text()
    transfer_source = source.split("Json run_host_transfer_probe", 1)[1]
    _setup, loop = transfer_source.split(
        "for (uint32_t repetition = 0; repetition < transfer_count; ++repetition)", 1
    )
    assert "make_data_buffer_pool(mesh, pool_size)" in transfer_source
    submission_loop = loop.split("const uint64_t host_load_submission_end_ns", 1)[0]
    assert "data_pool[repetition % pool_size]" in submission_loop
    assert "make_dram_buffer" not in submission_loop


def test_native_build_is_pinned_toolchain_only_and_has_no_board_free_fallback() -> None:
    root = Path(__file__).parents[1]
    cmake = (root / "enodia/tt/bench/issue120/CMakeLists.txt").read_text()
    runner = (root / "enodia/tt/bench/issue120/run_issue120_probe.py").read_text()
    contract = (root / "enodia/tt/bench/issue120_probe.py").read_text()
    assert "TT_METAL_SOURCE_REVISION" in cmake
    assert "bc77ac2f7382ffeb9fa5430055f0854d53979489" in cmake
    assert '"${UMD_API}"' in cmake
    assert "libtt_stl.so" in cmake
    assert "TT_METAL_LIBRARY_SHA256" in cmake
    source = (root / "enodia/tt/bench/issue120/issue120_probe.cpp").read_text()
    assert "run_fixed_cpu_work" in source
    assert "1'048'576" in source
    assert "20'000" not in source
    assert "coverage_scope" in source
    assert "not_frame_stream" in source
    assert "find_package(TT-Metalium REQUIRED)" not in cmake
    assert "HEKATUS_TT_PINNED_CONTAINER" in contract
    assert "native execution is supported only in the pinned container" in contract
    assert "raise SystemExit(main())" in runner
