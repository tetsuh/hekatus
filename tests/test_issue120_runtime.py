"""Board-free contracts for the pinned native runtime-root adapter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from enodia.tt.bench import issue120_probe as probe
from enodia.tt.bench import issue120_runtime as runtime

ROOT = "/opt/venv/lib/python3.12/site-packages/ttnn"


@pytest.fixture
def assets(tmp_path, monkeypatch):
    root = tmp_path / "wheel"
    root.mkdir()
    monkeypatch.setattr(runtime, "PINNED_RUNTIME_ROOT", root)
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(probe.issue120_results, "OUTPUT_ROOT", output)
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "runtime-unit")
    row = [{"path": "asset", "kind": "file", "size": 7, "executable": False}]
    digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    trees = []
    for name, path in (
        ("metal", root / "tt_metal"),
        ("runtime", root / "runtime"),
        ("libraries", root / "build" / "lib"),
        ("sfpi", tmp_path / "sfpi"),
    ):
        path.mkdir(parents=True)
        (path / "asset").write_bytes(b"fixture")
        trees.append(runtime.RuntimeAssetTree(name, path, digest))
    monkeypatch.setattr(runtime, "ASSET_TREES", tuple(trees))
    return root, trees


def execute(monkeypatch):
    monkeypatch.setenv("TT_METAL_WATCHER", "1")
    spec = probe.planned_probe("q1-cq1-completion", first_kernel_run=True)
    return probe._execute_native(Path("/fixture/native"), spec, "resident-and-transfer")


def forbid_native(monkeypatch):
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *_a, **_k: pytest.fail("native subprocess executed"))


def test_program_owned_paths_and_audited_fingerprints():
    assert str(runtime.PINNED_RUNTIME_ROOT) == ROOT
    assert [str(t.path) for t in runtime.ASSET_TREES] == [
        ROOT + "/tt_metal", ROOT + "/runtime", ROOT + "/build/lib", "/opt/tenstorrent/sfpi",
    ]
    assert [t.inventory_sha256 for t in runtime.ASSET_TREES] == [
        "8d85e48cca4c004fa41962846378fd06bd28bf57321ce24d7a9cafaf27a16314",
        "56fe4e441edf3f309dca55bd97db56a03c3f060e8bafa6bd1db2d3f05b41ac8b",
        "06f873632238323b3891239edaca92e54eebc7e674755d0749e14c533df9c8f2",
        "fb5e85084358f9deda5081a6ee953c5ee3360f7ce084da84b63a42a51b18ca8b",
    ]


def test_valid_assets_pass_fixed_root_to_native_and_override_ambient_root(assets, monkeypatch):
    root, _ = assets
    monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", "/external/untrusted")
    monkeypatch.setenv("TT_METAL_HOME", "/external/legacy")
    captured = {}

    class Process:
        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)

        def wait(self, timeout):
            captured["timeout"] = timeout
            return 0

    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    assert execute(monkeypatch).returncode == 0
    assert captured["env"]["TT_METAL_RUNTIME_ROOT"] == str(root)
    assert "TT_METAL_HOME" not in captured["env"]
    assert captured["timeout"] == 60
    assert captured["command"][-1] == "resident-and-transfer"
    assert captured["env"]["TT_METAL_WATCHER"] == "1"


def replace_asset(path, kind):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(b"not a directory")


def link_asset(path, tmp_path, mutation):
    item = path / "asset"
    item.unlink()
    target = path / "other" if mutation == "link_inside" else tmp_path / "outside"
    if mutation != "broken_link":
        target.write_bytes(b"fixture")
    item.symlink_to(target)


@pytest.mark.parametrize("tree_index", range(4))
@pytest.mark.parametrize("mutation", ["missing_tree", "tree_file", "missing_file", "wrong_kind", "empty_file", "executable", "extra_file", "fifo", "link_outside", "link_inside", "broken_link"])
def test_every_asset_tree_rejects_damage_before_native(assets, monkeypatch, tmp_path, tree_index, mutation):
    _, trees = assets
    path = trees[tree_index].path
    item = path / "asset"
    actions = {
        "missing_tree": lambda: shutil.rmtree(path),
        "tree_file": lambda: replace_asset(path, "file"),
        "missing_file": item.unlink,
        "wrong_kind": lambda: replace_asset(item, "directory"),
        "empty_file": lambda: item.write_bytes(b""),
        "executable": lambda: item.chmod(0o755),
        "extra_file": lambda: (path / "unexpected").write_bytes(b"extra"),
        "fifo": lambda: replace_asset(item, "fifo"),
    }
    if mutation in actions:
        actions[mutation]()
    else:
        link_asset(path, tmp_path, mutation)
    forbid_native(monkeypatch)
    with pytest.raises(probe.ProbeRecordError, match="runtime"):
        execute(monkeypatch)


@pytest.mark.parametrize("tree_index", range(4))
def test_redirected_asset_tree_is_rejected(assets, monkeypatch, tmp_path, tree_index):
    _, trees = assets
    path = trees[tree_index].path
    outside = tmp_path / "redirected"
    shutil.move(path, outside)
    path.symlink_to(outside, target_is_directory=True)
    forbid_native(monkeypatch)
    with pytest.raises(probe.ProbeRecordError, match="runtime"):
        execute(monkeypatch)


@pytest.mark.parametrize("mutation", ["missing", "file", "redirected"])
def test_wheel_root_is_validated_before_native(assets, monkeypatch, tmp_path, mutation):
    root, _ = assets
    if mutation == "redirected":
        target = tmp_path / "redirected"
        shutil.move(root, target)
        root.symlink_to(target, target_is_directory=True)
    else:
        shutil.rmtree(root)
        if mutation == "file":
            root.write_bytes(b"wrong kind")
    forbid_native(monkeypatch)
    with pytest.raises(probe.ProbeRecordError, match="runtime"):
        execute(monkeypatch)


def test_asset_io_failure_is_reported_before_native(assets, monkeypatch):
    _, trees = assets
    original = Path.rglob

    def denied(path, pattern):
        if path == trees[0].path:
            raise PermissionError("denied fixture")
        return original(path, pattern)

    monkeypatch.setattr(Path, "rglob", denied)
    forbid_native(monkeypatch)
    with pytest.raises(probe.ProbeRecordError, match="runtime"):
        execute(monkeypatch)


def test_runtime_resolution_failure_is_reported_before_native(assets, monkeypatch):
    root, _ = assets
    original = Path.resolve

    def failed(path, *args, **kwargs):
        if path == root:
            raise RuntimeError("resolution failed fixture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", failed)
    forbid_native(monkeypatch)
    with pytest.raises(probe.ProbeRecordError, match="runtime"):
        execute(monkeypatch)


def test_runtime_root_has_no_cli_override(monkeypatch):
    forbid_native(monkeypatch)
    with pytest.raises(SystemExit):
        probe.main(["--runtime-root", "/external"])


def test_plan_does_not_access_runtime_assets(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "validate_runtime_assets", lambda: pytest.fail("runtime assets accessed"))
    assert probe.main(["--plan"]) == 0
    assert json.loads(capsys.readouterr().out)["board_access"] is False
