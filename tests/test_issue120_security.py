"""Board-free rejection tests for the Issue #120 command boundary."""

from __future__ import annotations

import argparse

import pytest

from enodia.tt.bench import issue120_probe as probe


@pytest.fixture
def roots(tmp_path, monkeypatch):
    for name in ("BUILD_ROOT", "SOURCE_ROOT", "INCLUDE_ROOT", "LIBRARY_ROOT", "WHEEL_LIBRARY_ROOT", "OUTPUT_ROOT"):
        root = tmp_path / name.lower()
        root.mkdir()
        monkeypatch.setattr(probe, name, root, raising=False)
    return tmp_path


@pytest.mark.parametrize("attack", ["../escape", "folder;--install", "folder\n--help", "$(command)"])
def test_path_rejects_escape_and_command_syntax(roots, attack):
    with pytest.raises(probe.ProbeRecordError):
        probe._bounded_path(probe.BUILD_ROOT / attack, probe.BUILD_ROOT, "build", "directory")


def test_path_rejects_absolute_escape_and_symlink(roots):
    outside = roots / "outside"
    outside.mkdir()
    (probe.BUILD_ROOT / "link").symlink_to(outside, target_is_directory=True)
    for value in (outside, probe.BUILD_ROOT / "link"):
        with pytest.raises(probe.ProbeRecordError, match="root"):
            probe._bounded_path(value, probe.BUILD_ROOT, "build", "directory")


def test_trusted_root_itself_cannot_be_redirected(roots):
    outside = roots / "outside"
    outside.mkdir()
    probe.BUILD_ROOT.rmdir()
    probe.BUILD_ROOT.symlink_to(outside, target_is_directory=True)
    with pytest.raises(probe.ProbeRecordError, match="root"):
        probe._bounded_path(probe.BUILD_ROOT / "new", probe.BUILD_ROOT, "build", "build")


def test_path_accepts_resolved_descendant_and_rejects_wrong_kind(roots):
    child = probe.BUILD_ROOT / "child"
    child.mkdir()
    assert probe._bounded_path(child / ".", probe.BUILD_ROOT, "build", "directory") == child.resolve()
    with pytest.raises(probe.ProbeRecordError, match="file"):
        probe._bounded_path(child, probe.BUILD_ROOT, "library", "file")


@pytest.mark.parametrize("value", ["transfer-only;command", "--help", "unknown"])
def test_stall_group_is_canonical_allowlist_only(value):
    with pytest.raises(probe.ProbeRecordError):
        probe._native_stall_group(value)


def test_frame_probe_rejects_mismatched_stall_group():
    with pytest.raises(probe.ProbeRecordError, match="transfer-only"):
        probe._selected_stall_group("q4-frame-coverage", "resident-and-transfer")
    assert probe._selected_stall_group("q4-frame-coverage", None) == "transfer-only"


def test_stall_group_uses_constant_value():
    assert probe._native_stall_group("transfer-only") == "transfer-only"
    assert probe._native_stall_group(None) is None


@pytest.mark.parametrize("value", ["--help", "q2-h2d-write-only;command", "unknown"])
def test_argparse_rejects_unknown_probe_before_execution(value):
    with pytest.raises(SystemExit):
        probe.main(["--probe", value])


@pytest.mark.parametrize("value", [True, 0, -1, 1 << 32, "3;command"])
def test_invalid_work_integer_is_rejected(value):
    with pytest.raises(probe.ProbeRecordError):
        probe.planned_probe("q2-h2d-write-only", repeat_count=value)


def build_args():
    return argparse.Namespace(
        build_dir=probe.BUILD_ROOT,
        tt_metal_source_root=probe.SOURCE_ROOT,
        tt_metal_source_revision="d9a68815f5fcf08a5bfbffb6f1f811823fba8edd",
        tt_metal_library=probe.LIBRARY_ROOT / "libtt_metal.so",
        tt_metal_extra_include=[probe.INCLUDE_ROOT],
        tt_metal_extra_lib_dir=[probe.LIBRARY_ROOT, probe.WHEEL_LIBRARY_ROOT],
    )


@pytest.mark.parametrize("field", ["build_dir", "tt_metal_source_root", "tt_metal_library", "tt_metal_extra_include", "tt_metal_extra_lib_dir"])
def test_build_rejects_every_untrusted_path_before_subprocess(roots, monkeypatch, field):
    (probe.LIBRARY_ROOT / "libtt_metal.so").touch()
    args = build_args()
    value = roots / "outside"
    value.mkdir()
    if field.startswith("tt_metal_extra"):
        setattr(args, field, [value])
    else:
        setattr(args, field, value)
    monkeypatch.setattr(probe.subprocess, "run", lambda *_a, **_k: pytest.fail("subprocess executed"))
    with pytest.raises(probe.ProbeRecordError):
        probe._build_native_probe(args)


def test_build_rejects_revision_argument_injection(roots, monkeypatch):
    args = build_args()
    args.tt_metal_source_revision = "-DCMAKE_PROJECT_INCLUDE=attack"
    monkeypatch.setattr(probe.subprocess, "run", lambda *_a, **_k: pytest.fail("subprocess executed"))
    with pytest.raises(probe.ProbeRecordError, match="revision"):
        probe._build_native_probe(args)


def test_build_subprocess_receives_validated_paths_and_literal_revision(roots, monkeypatch):
    library = probe.LIBRARY_ROOT / "libtt_metal.so"
    library.touch()
    commands = []
    monkeypatch.setattr(probe.subprocess, "run", lambda command, **_k: commands.append(command))
    assert probe._build_native_probe(build_args()) == 0
    assert commands[0][0] == "cmake"
    assert f"-DTT_METAL_SOURCE_ROOT={probe.SOURCE_ROOT.resolve()}" in commands[0]
    assert commands[1] == ["cmake", "--build", str(probe.BUILD_ROOT.resolve()), "--target", "issue120_probe", "--parallel", "2"]


def test_native_paths_reject_build_and_executable_symlink_escape(roots):
    outside = roots / "outside"
    outside.mkdir()
    (outside / "issue120_probe").touch()
    (probe.BUILD_ROOT / "issue120_probe").symlink_to(outside / "issue120_probe")
    (probe.BUILD_ROOT / "libtt_metal.so").touch()
    with pytest.raises(probe.ProbeRecordError, match="root"):
        probe._native_paths(probe.BUILD_ROOT)
    with pytest.raises(probe.ProbeRecordError, match="root"):
        probe._native_paths(outside)


def test_native_paths_reject_library_symlink_escape(roots):
    outside = roots / "library.so"
    outside.touch()
    (probe.BUILD_ROOT / "issue120_probe").touch()
    (probe.BUILD_ROOT / "libtt_metal.so").symlink_to(outside)
    with pytest.raises(probe.ProbeRecordError, match="root"):
        probe._native_paths(probe.BUILD_ROOT)


def test_result_destination_cannot_escape_output_root_or_overwrite(roots):
    with pytest.raises(probe.ProbeRecordError, match="root"):
        probe._result_destination(roots / "outside.json")
    existing = probe.OUTPUT_ROOT / "record.json"
    existing.touch()
    with pytest.raises(probe.ProbeRecordError, match="new"):
        probe._result_destination(existing)
    assert probe._result_destination(probe.OUTPUT_ROOT / "new.json") == probe.OUTPUT_ROOT / "new.json"


def test_environment_path_cannot_escape_output_root(roots):
    with pytest.raises(probe.ProbeRecordError):
        probe._environment_record(str(roots / "outside.json"))
