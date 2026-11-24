"""Board-free layout/checker controls; no physical observations or native execution."""
from __future__ import annotations

import json

import pytest

from enodia.tt.bench import issue30 as m


def test_layout_root_not_just_kernel_search_path(tmp_path):
    value = m.request("B", "synthetic-layout", tmp_path, tmp_path, uid=1000, gid=1000)
    assert "TT_METAL_RUNTIME_ROOT=/capture/runtime" in value["Env"]
    assert value["Cmd"][:3] == ["timeout", "--signal=KILL", "40s"]
    assert value["Cmd"][3:] == ["python3", "/harness/enodia/tt/bench/issue30_runtime.py", "B"]
    assert value["HostConfig"]["ReadonlyRootfs"]
    assert value["HostConfig"]["NetworkMode"] == "none"
    assert value["HostConfig"]["Devices"] == []


def test_runtime_composes_image_assets_tests_and_owned_generated(tmp_path, monkeypatch):
    from enodia.tt.bench import issue30_runtime as runtime

    image = tmp_path / "image"; image.mkdir()
    for name in ("tt_metal", "runtime", "build", "ttnn"):
        (image / name).mkdir()
    (image / "build/lib").mkdir()
    (image / "build/lib/libtt_metal.so").write_bytes(b"synthetic-library")
    source = tmp_path / "source"; (source / "tests").mkdir(parents=True)
    (source / "tests/header.hpp").write_bytes(b"synthetic-header")
    capture = tmp_path / "capture"; capture.mkdir()
    monkeypatch.setattr(runtime, "IMAGE_ROOT", image)
    monkeypatch.setattr(runtime, "SOURCE_ROOT", source)
    monkeypatch.setattr(runtime, "CAPTURE", capture)
    monkeypatch.setattr(runtime, "validate_image", lambda: image)
    root = runtime.prepare()
    assert (root / "tests/header.hpp").read_bytes() == b"synthetic-header"
    assert (root / "tt_metal").resolve() == image / "tt_metal"
    assert (root / "build/lib/libtt_metal.so").read_bytes() == b"synthetic-library"
    assert (root / "generated/fabric").is_dir()
    assert not (root / "generated").is_symlink()
    assert (root / "generated").resolve().is_relative_to(capture)
    with pytest.raises(FileExistsError):
        runtime.prepare()


def test_inventory_matches_fixture_and_all_saved_registered_kernel_names():
    from enodia.tt.bench import issue196_compile as check

    value = json.loads(check.MANIFEST.read_bytes())
    assert check.kernel_inventory() == set(value["kernel_sources"])
    assert len(check.kernel_inventory()) == 7
    assert {x["kernel"] for x in value["cases"]} == check.kernel_inventory()
    assert len(value["cases"]) == 24  #22 observed dispatch/router TUs plus RX/TX.
    assert all(x["defines"] for x in value["cases"])


def test_manifest_rejects_missing_specialization_and_injected_compiler_option():
    from enodia.tt.bench import issue196_compile as check

    value = json.loads(check.MANIFEST.read_bytes())
    check.validate_manifest(value)
    value["cases"].pop()
    with pytest.raises(ValueError, match="inventory"):
        check.validate_manifest(value)
    value = json.loads(check.MANIFEST.read_bytes())
    value["captured_common_flags"].append("-fplugin=unapproved")
    with pytest.raises(ValueError, match="compiler options"):
        check.validate_manifest(value)


def test_B_v2_power_cycle_does_not_reuse_old_A_or_B_scope(tmp_path, monkeypatch):
    calls = []
    def only_gate(path, *args):
        calls.append(path.name)
        return {"owner_approved": False}
    monkeypatch.setattr(m, "_load", only_gate)
    with pytest.raises(ValueError, match="B_v2_requires_new_owner_and_fresh_A_decision"):
        m._authorize("B", "synthetic", tmp_path, tmp_path / "run")
    assert calls == ["morning-admission.json"]


def test_offline_options_do_not_add_source_root_include_workaround():
    from enodia.tt.bench import issue196_compile as check

    value = json.loads(check.MANIFEST.read_bytes())
    case = next(x for x in value["cases"] if x["id"] == "captured-rx")
    from pathlib import Path

    args = check.options(value, case, Path("/capture/runtime"))
    assert "-I/capture/runtime/" in args
    assert "-I/source" not in args
    assert len([x for x in args if x.startswith("-I/source/")]) == 1


def test_runtime_root_redirect_and_unknown_native_phase_are_refused(tmp_path, monkeypatch):
    from enodia.tt.bench import issue30_runtime as runtime

    image = tmp_path / "image"; image.mkdir()
    source = tmp_path / "source"; (source / "tests").mkdir(parents=True)
    capture = tmp_path / "redirect"; capture.symlink_to(image, target_is_directory=True)
    monkeypatch.setattr(runtime, "IMAGE_ROOT", image)
    monkeypatch.setattr(runtime, "SOURCE_ROOT", source)
    monkeypatch.setattr(runtime, "CAPTURE", capture)
    monkeypatch.setattr(runtime, "validate_image", lambda: image)
    with pytest.raises(ValueError, match="redirected"):
        runtime.prepare()
    monkeypatch.setattr("sys.argv", ["runtime", "../wrong"])
    with pytest.raises(SystemExit):
        runtime.main()
