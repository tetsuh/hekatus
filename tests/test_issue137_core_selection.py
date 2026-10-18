"""Board-free Issue #137 worker selection and L1 seam tests."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from enodia.tt.bench import load_generators, load_harness, newton_schulz_kernel, run_resident
from enodia.tt.bench.load_harness import (
    NS_LOAD_BATCH,
    NS_LOAD_L1_TOTAL_BYTES,
    NS_LOAD_MATRIX_BLOCK,
    P150_ALL_CORE_COORDINATES,
    P150_RESIDUAL_WORKER_COORDINATES,
    RESIDENT_CORES,
    build_newton_schulz_core_plan,
    newton_schulz_l1_ledger,
    newton_schulz_total_blocks,
    ns_work_distribution,
    select_worker_cores,
)
from enodia.tt.bench.run_resident import _LoadedDeviceResources, _prepare_loaded_device


@pytest.mark.parametrize(
    ("core_count", "total_blocks", "batch"),
    ((18, 171, 1368), (54, 512, 4096), (81, 768, 6144), (108, 1024, 8192)),
)
def test_issue137_planned_core_distributions_and_l1_ledger(
    core_count: int, total_blocks: int, batch: int
):
    plan = build_newton_schulz_core_plan(core_count=core_count)

    assert plan.total_blocks == total_blocks
    assert plan.batch == batch
    assert len(plan.worker_cores) == core_count
    assert not set(plan.worker_cores) & set(RESIDENT_CORES)
    assert sum(row["block_count"] for row in plan.block_distribution) == total_blocks
    assert {row["block_count"] for row in plan.block_distribution} == {9, 10}
    assert sum(row["tile_count"] for row in plan.work_distribution) == batch
    assert plan.l1_ledger["total_bytes"] == NS_LOAD_L1_TOTAL_BYTES
    assert plan.l1_ledger["profile_total_bytes"] == 1_307_392
    assert plan.l1_ledger["reference_l1_equal"] is True
    allocation = plan.l1_ledger["interleaved_input_allocation"]
    assert allocation["global_payload_bytes"] == {
        18: 14_014_464,
        54: 41_949_184,
        81: 62_920_704,
        108: 83_892_224,
    }[core_count]
    assert allocation["tensor_shapes"][:5] == [[batch, 1, 32, 32]] * 5
    assert allocation["tensor_shapes"][5:] == [[1, 1, 32, 32]] * 2
    assert allocation["tensor_dtypes"][:5] == ["bfloat16"] * 5
    assert allocation["tensor_dtypes"][5:] == ["bfloat16", "float32"]
    assert allocation["input_payload_bytes"] == batch * 5 * 2 * 32 * 32
    assert allocation["constant_payload_bytes"] == 2 * 32 * 32 + 4 * 32 * 32
    assert allocation["bank_distribution_estimate"]["bank_count"] == 110
    assert allocation["bank_distribution_estimate"]["measured"] is False
    bank_estimate = allocation["bank_distribution_estimate"]
    assert bank_estimate["within_budget"] is True
    assert bank_estimate["mapping_exact"] is False
    records = bank_estimate["per_tensor"]
    assert [row["pages_per_bank_upper_bound"] for row in records[:5]] == [
        (batch + 109) // 110
    ] * 5
    assert bank_estimate["busiest_bank_reservation_bytes_upper_bound"] == {
        18: 139_264,
        54: 395_264,
        81: 579_584,
        108: 774_144,
    }[core_count]
    assert newton_schulz_l1_ledger(
        core_count=core_count,
        total_blocks=total_blocks,
    )["total_bytes"] == NS_LOAD_L1_TOTAL_BYTES
    assert len(
        ns_work_distribution(core_count=core_count, total_blocks=total_blocks)
    ) == core_count
    assert plan.as_record()["workload"]["worker_core_coordinates"] == [
        list(core) for core in plan.worker_cores
    ]


def test_issue137_worker_subset_is_row_major_deterministic_and_excludes_resident_cores():
    expected = P150_RESIDUAL_WORKER_COORDINATES[:18]

    assert select_worker_cores(pool=P150_ALL_CORE_COORDINATES, core_count=18) == expected
    assert select_worker_cores(pool=tuple(reversed(P150_ALL_CORE_COORDINATES)), core_count=18) == expected
    assert newton_schulz_total_blocks(18) == 171
    assert not set(expected) & set(RESIDENT_CORES)


def test_issue137_explicit_plan_reaches_kernel_prepare_with_batch_and_coordinates(monkeypatch):
    plan = build_newton_schulz_core_plan(core_count=18)
    captured: dict[str, object] = {}

    class FakeKernel:
        @classmethod
        def prepare(cls, ttnn, device, matrices, **kwargs):
            captured["batch"] = matrices.shape[0]
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(newton_schulz_kernel, "NewtonSchulzKernel", FakeKernel)
    load_generators.prepare_newton_schulz_load(
        object(),
        object(),
        core_plan=plan,
        profile=False,
    )

    assert captured["batch"] == 1368
    assert captured["matrix_block"] == 8
    assert captured["core_coordinates"] == plan.worker_cores
    assert captured["outer_repetitions"] == 3


def test_issue137_explicit_plan_reaches_resident_prepare(monkeypatch):
    plan = build_newton_schulz_core_plan(core_count=18)
    captured: dict[str, object] = {}

    class FakeLoad:
        def __init__(self):
            self.inputs = []
            self.profile_output = None
            self.outputs = []

        @staticmethod
        def program_parts():
            return ([], [], [])

    monkeypatch.setattr(run_resident, "_allocate_resident_tensors", lambda *_args: {})
    monkeypatch.setattr(
        run_resident,
        "_program",
        lambda *_args, **_kwargs: ([], [], []),
    )

    def fake_prepare(*_args, **kwargs):
        captured.update(kwargs)
        return FakeLoad()

    monkeypatch.setattr(run_resident, "prepare_newton_schulz_load", fake_prepare)
    monkeypatch.setattr(run_resident, "merge_program_parts", lambda *_args: object())
    resources = _LoadedDeviceResources("newton_schulz")

    _prepare_loaded_device(
        object(),
        object(),
        object(),
        resources,
        repeat_count=1,
        watcher=False,
        newton_schulz_core_plan=plan,
    )

    assert captured["core_plan"] is plan
    assert resources.newton_schulz_core_plan is plan


def test_issue137_n108_is_the_q2_workload_and_default_api_is_unchanged():
    plan = build_newton_schulz_core_plan(core_count=108)
    rows = ns_work_distribution()
    workload = load_generators.newton_schulz_load_work()

    assert plan.worker_cores == P150_RESIDUAL_WORKER_COORDINATES
    assert plan.total_blocks == 1024
    assert plan.batch == NS_LOAD_BATCH == 8192
    assert plan.matrix_block == NS_LOAD_MATRIX_BLOCK == 8
    assert [
        {
            "core": row["core"],
            "start_tile": row["start_block"] * 8,
            "tile_count": row["block_count"] * 8,
        }
        for row in plan.block_distribution
    ] == list(rows)
    assert workload["batch"] == 8192
    assert workload["worker_count"] == 108


def test_issue137_l1_reference_difference_is_reported_not_rejected(monkeypatch):
    original = load_harness._newton_schulz_l1_breakdown

    def different_reference(*, batch, core_count, matrix_block, profile):
        result = original(
            batch=batch,
            core_count=core_count,
            matrix_block=matrix_block,
            profile=profile,
        )
        if core_count == 108:
            result = dict(result)
            result["total_bytes"] += 256
        return result

    monkeypatch.setattr(load_harness, "_newton_schulz_l1_breakdown", different_reference)
    plan = load_harness.build_newton_schulz_core_plan(core_count=18)

    assert plan.l1_ledger["reference_l1_equal"] is False
    assert plan.l1_ledger["within_budget"] is True


def test_issue137_invalid_work_is_rejected_before_device_open():
    with pytest.raises(ValueError, match="L1/work invariant|9 or 10"):
        build_newton_schulz_core_plan(core_count=18, total_blocks=160)


def test_issue137_l1_mismatch_is_rejected_before_device_open(tmp_path, monkeypatch):
    environment = {
        "board": {"serial": "redacted-board-serial", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kmd_version": "2.11.0",
        "image": "registry.example/tt@sha256:" + "a" * 64,
        "image_pinned": True,
        "harness_commit": "0123456789abcdef",
        "harness_dirty": False,
        "harness_pushed": True,
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": {
            "mode": "off",
            "interval_seconds": None,
            "power_trace": "absent_by_design",
            "timing_evidence": "diagnostic_only",
        },
    }
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(environment))
    output_path = tmp_path / "rejection.json"
    open_calls: list[tuple[object, ...]] = []

    def device_open_must_not_be_called(*args, **kwargs):
        open_calls.append((args, kwargs))
        raise AssertionError("device open must not be called")

    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(open_device=device_open_must_not_be_called),
    )

    result = run_resident.main(
        [
            "--out",
            str(output_path),
            "--env-json",
            str(environment_path),
            "--condition",
            "newton_schulz",
            "--load-core-count",
            "18",
            "--load-total-blocks",
            "160",
            "--calibrate",
        ]
    )

    assert result == 2
    assert open_calls == []
    assert "L1" in json.loads(output_path.read_text())["rejection_reason"]
