"""Generated-input isolation and ancestor preservation, without a real build."""
from __future__ import annotations

import os

import pytest

from enodia.tt.bench.issue120_production import runtime


def fake_build(monkeypatch, tmp_path):
    build = tmp_path / "out/issue120-production-build"
    build.mkdir(parents=True)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "BUILD", build)
    monkeypatch.setattr(runtime.kernel_adapter, "generate", lambda *a: None)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "source_hashes", dict)
    monkeypatch.setattr(runtime.host_build_inputs, "verify_flags", lambda *a: str(tmp_path))
    monkeypatch.setattr(runtime.host_build_inputs, "identity", lambda *a: {})
    monkeypatch.setattr(runtime, "sha", lambda *a: "a" * 64)
    return build


def test_old_generated_symlink_cannot_overwrite_preserved_source(monkeypatch, tmp_path):
    build = fake_build(monkeypatch, tmp_path)
    ancestor = tmp_path / "preserved-ancestor.cpp"
    ancestor.write_bytes(b"preserve original ancestor")
    (build / "native_trace.cpp").symlink_to(ancestor)
    runtime.build()
    assert ancestor.read_bytes() == b"preserve original ancestor"


def test_atomic_generated_replacement_preserves_a_hardlinked_ancestor(monkeypatch, tmp_path):
    build = fake_build(monkeypatch, tmp_path)
    generated = build / "host-trace-input"
    generated.mkdir(mode=0o700)
    ancestor = tmp_path / "ancestor.cpp"
    ancestor.write_bytes(b"preserve ancestor")
    os.link(ancestor, generated / "native_trace.cpp")
    runtime.build()
    assert ancestor.read_bytes() == b"preserve ancestor"
    assert (generated / "native_trace.cpp").stat().st_nlink == 1


@pytest.mark.parametrize("poison", ["protocol.hpp", "native_trace.cpp"])
def test_dedicated_generated_input_rejects_shadow_headers_and_output_links(monkeypatch, tmp_path, poison):
    build = fake_build(monkeypatch, tmp_path)
    generated = build / "host-trace-input"
    generated.mkdir(mode=0o700)
    if poison == "protocol.hpp":
        (generated / poison).write_text("#error shadow header")
    else:
        ancestor = tmp_path / "ancestor.cpp"
        ancestor.write_text("original")
        (generated / poison).symlink_to(ancestor)
    with pytest.raises(ValueError):
        runtime.build()
