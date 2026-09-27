import hashlib
import inspect
import io
from pathlib import Path

import pytest

from tools import newton_schulz_bringup as bringup


class _FakeRuntime:
    def __init__(self):
        self.generic_calls = []
        self.synchronized = 0
        self.deallocated = []

    def generic_op(self, tensors, program):
        self.generic_calls.append((tensors, program))

    def synchronize_device(self, device):
        self.synchronized += 1

    def deallocate(self, tensor):
        self.deallocated.append(tensor)


def _valid_environment(cache_directory):
    return {
        "TT_METAL_CACHE": str(cache_directory),
        "TT_METAL_FORCE_JIT_COMPILE": "1",
        "TT_METAL_LOG_KERNELS_COMPILE_COMMANDS": "1",
        "TT_METAL_KERNELS_EARLY_RETURN": "1",
    }


def test_build_only_probe_allow_lists_only_stages_62_and_67():
    assert bringup.BUILD_ONLY_JIT_STAGES == (62, 67)
    assert [bringup.build_only_stage_for(number).number for number in (62, 67)] == [62, 67]
    for number in (1, 61, 68):
        with pytest.raises(bringup.BuildOnlyProbeConfigurationError) as caught:
            bringup.build_only_stage_for(number)
        assert caught.value.code == "stage_not_allowlisted"


def test_build_only_probe_requires_all_runtime_controls(tmp_path):
    cache_directory = tmp_path / "jit-cache"
    environment = _valid_environment(cache_directory)
    effective = bringup.validate_build_only_environment(environment)
    assert Path(effective["TT_METAL_CACHE"]) == cache_directory.resolve()
    for name in bringup.BUILD_ONLY_REQUIRED_ENV:
        missing = dict(environment)
        missing.pop(name)
        with pytest.raises(bringup.BuildOnlyProbeConfigurationError) as caught:
            bringup.validate_build_only_environment(missing)
        assert caught.value.code == "required_environment_invalid"
        assert name in caught.value.details["variables"]

    disabled = dict(environment, TT_METAL_KERNELS_EARLY_RETURN="0")
    with pytest.raises(bringup.BuildOnlyProbeConfigurationError) as caught:
        bringup.validate_build_only_environment(disabled)
    assert caught.value.details["variables"]["TT_METAL_KERNELS_EARLY_RETURN"] == "expected '1'"


def test_build_only_path_uses_real_sources_and_full_one_tile_shape():
    for number in bringup.BUILD_ONLY_JIT_STAGES:
        stage = bringup.build_only_stage_for(number)
        assert stage.batch // stage.cores == 1
        assert not any(
            "construction" in source
            for source in (stage.compute_source, stage.reader_source, stage.writer_source)
        )
        compute, reader, writer = bringup.source_paths(stage)
        assert compute.is_file() and reader.is_file() and writer.is_file()

    prepare_source = inspect.getsource(bringup._prepare_stage_program)
    dispatch_source = inspect.getsource(bringup.run_build_only_stage)
    assert "compile_time_args=[tiles_per_core]" in prepare_source
    assert "CONSTRUCTION_DISPATCH_COMPILE_TILE_COUNT" not in prepare_source
    assert "construction_dispatch_configuration" not in dispatch_source
    assert "_prepare_stage_program" in dispatch_source


def test_build_only_dispatch_does_not_accept_numerical_or_download_path(monkeypatch, tmp_path):
    runtime = _FakeRuntime()
    seen = {}

    def fake_prepare(ttnn, device, stage, input_values):
        seen["stage"] = stage.number
        seen["input_values"] = input_values
        return "full-work-program", ["input"], ["output"]

    def forbidden(*args, **kwargs):
        raise AssertionError("build-only path must not use numerical/download helpers")

    monkeypatch.setattr(bringup, "_prepare_stage_program", fake_prepare)
    monkeypatch.setattr(bringup, "expected_output", forbidden)
    monkeypatch.setattr(bringup, "_download", forbidden)

    result = bringup.run_build_only_stage(runtime, object(), 62, tmp_path / "cache")

    assert result["success"] is True
    assert result["status"] == "pass"
    assert result["numerical_acceptance"] is False
    assert result["output_download"] is False
    assert result["compile_time_tile_count"] == [1]
    assert result["reader_tile_count"] == 1
    assert result["writer_tile_count"] == 1
    assert seen["stage"] == 62
    assert len(seen["input_values"]) == bringup.STAGES[62].input_count
    assert all(value.shape == (1, bringup.TILE, bringup.TILE) for value in seen["input_values"])
    assert runtime.generic_calls == [(["input", "output"], "full-work-program")]
    assert runtime.synchronized == 1


def test_cache_manifest_is_sorted_relative_and_hashes_deterministically(tmp_path):
    stage = bringup.STAGES[62]
    reader = tmp_path / "reader" / f"compiled-{stage.reader_source}.bin"
    compute = tmp_path / "compute" / f"compiled-{stage.compute_source}.bin"
    reader.parent.mkdir()
    compute.parent.mkdir()
    reader.write_bytes(b"reader-artifact")
    compute.write_bytes(b"compute-artifact")
    (tmp_path / "unrelated.bin").write_bytes(b"ignored")

    first = bringup.cache_artifact_manifest(tmp_path, stage)
    second = bringup.cache_artifact_manifest(tmp_path, stage)
    assert first == second
    assert first["reader"]["status"] == "matched"
    assert first["compute"]["status"] == "matched"
    for kind, path, payload in (
        ("reader", reader, b"reader-artifact"),
        ("compute", compute, b"compute-artifact"),
    ):
        artifact = first[kind]["artifacts"][0]
        assert not Path(artifact["relative_path"]).is_absolute()
        assert artifact["relative_path"] == path.relative_to(tmp_path).as_posix()
        assert artifact["size_bytes"] == len(payload)
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()

    reader.write_bytes(b"reader-artifact-updated")
    changed = bringup.cache_artifact_manifest(tmp_path, stage)
    assert changed["reader"] != first["reader"]
    assert changed["compute"] == first["compute"]


def test_cache_manifest_explicitly_records_missing_stage_artifacts(tmp_path):
    manifest = bringup.cache_artifact_manifest(tmp_path, bringup.STAGES[67])
    assert manifest["reader"] == {
        "source": bringup.STAGES[67].reader_source,
        "status": "no_matching_artifacts",
        "artifacts": [],
    }
    assert manifest["compute"] == {
        "source": bringup.STAGES[67].compute_source,
        "status": "no_matching_artifacts",
        "artifacts": [],
    }


def test_build_only_failure_keeps_pre_manifest_before_dispatch_when_close_fails(
    monkeypatch, tmp_path
):
    stage = bringup.STAGES[62]
    cache_directory = tmp_path / "jit-cache"
    artifact = cache_directory / "compiled" / f"compiled-{stage.compute_source}.bin"
    payload = b"dispatch-created-artifact"

    class _CloseFailingRuntime(_FakeRuntime):
        def __init__(self):
            super().__init__()
            self.opened_device_ids = []
            self.closed_devices = []

        def open_device(self, device_id):
            self.opened_device_ids.append(device_id)
            return object()

        def generic_op(self, tensors, program):
            super().generic_op(tensors, program)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(payload)

        def close_device(self, device):
            self.closed_devices.append(device)
            raise RuntimeError("close failed")

    monkeypatch.setattr(
        bringup,
        "_prepare_stage_program",
        lambda ttnn, device, selected, input_values: ("full-work-program", ["input"], ["output"]),
    )
    runtime = _CloseFailingRuntime()
    result = bringup.run_build_only_jit_probe(
        stage.number,
        cache_directory=cache_directory,
        device_id=3,
        environment=_valid_environment(cache_directory),
        ttnn_module=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "fail"
    assert result["error"]["code"] == "runtime_unavailable"
    assert runtime.opened_device_ids == [3]
    assert len(runtime.generic_calls) == 1
    assert len(runtime.closed_devices) == 1

    pre = result["cache_artifacts"]["pre"]
    post = result["cache_artifacts"]["post"]
    assert pre["compute"] == {
        "source": stage.compute_source,
        "status": "no_matching_artifacts",
        "artifacts": [],
    }
    assert post["compute"] == {
        "source": stage.compute_source,
        "status": "matched",
        "artifacts": [
            {
                "relative_path": artifact.relative_to(cache_directory).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    assert post["reader"]["artifacts"] == []


def test_build_only_preflight_failure_is_machine_readable_without_runtime_import(tmp_path):
    result = bringup.run_build_only_jit_probe(
        62,
        cache_directory=tmp_path / "cache",
        environment={},
    )
    assert result["exit_code"] == 2
    assert result["success"] is False
    assert result["error"]["code"] == "required_environment_invalid"
    assert result["error"]["details"]["variables"]["TT_METAL_FORCE_JIT_COMPILE"] == "missing"


def test_json_emitter_flushes_a_single_machine_readable_record():
    stream = io.StringIO()
    bringup._emit_json({"stage": 62, "success": False}, stream=stream)
    assert stream.getvalue() == '{"stage": 62, "success": false}\n'
