"""Compiler input proof rejects cached/ambient quote-search bypasses."""
from __future__ import annotations

import shutil

import pytest


@pytest.fixture
def inputs(tmp_path):
    from enodia.tt.bench.issue120_production import host_build_inputs as h

    root = tmp_path / "project"
    build = root / "out/issue120-production-build"
    rules = build / "CMakeFiles/issue120_production.dir"
    rules.mkdir(parents=True)
    (rules / "host-trace-input").mkdir()
    cache = "CMAKE_BUILD_TYPE:STRING=\nCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\nCMAKE_CXX_FLAGS:STRING=\n"
    (build / "CMakeCache.txt").write_text(cache)
    includes = " ".join('-I' + str(root / relative) for relative in h.INCLUDE_PATHS)
    (rules / "flags.make").write_text('CXX_DEFINES = ' + h.DEFINES + '\nCXX_INCLUDES = ' + includes + '\nCXX_FLAGS = -std=gnu++20\n')
    for name in h.PROTECTED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("protected header\n")
    dep = rules / "host-trace-input/native_trace.cpp.o.d"
    dep.write_text("target.o: " + " ".join(str(root / name) for name in h.PROTECTED) + "\n")
    return h, root, build, rules


@pytest.mark.parametrize("kind", ["cache", "flags", "launcher"])
def test_quote_search_and_launcher_overrides_cannot_receive_a_receipt(inputs, kind):
    h, _, build, rules = inputs
    if kind == "flags":
        path = rules / "flags.make"
        path.write_text(path.read_text().replace("-std=gnu++20", "-iquote /untrusted -std=gnu++20"))
    else:
        with (build / "CMakeCache.txt").open("a") as output:
            output.write("CMAKE_CXX_FLAGS:STRING=-iquote /untrusted\n" if kind == "cache" else "CMAKE_CXX_COMPILER_LAUNCHER:STRING=untrusted\n")
    with pytest.raises(ValueError):
        h.verify_flags(build)


def test_actual_dependency_shadow_is_rejected(inputs):
    h, root, build, rules = inputs
    dep = rules / "host-trace-input/native_trace.cpp.o.d"
    dep.write_text(dep.read_text().replace(str(root / "enodia/tt/bench/issue120_production/host_sequence.hpp"),
                                         str(build / "host_sequence.hpp")))
    with pytest.raises(ValueError, match="header"):
        h.identity(build, root)


def test_compile_input_identity_is_relocatable_and_has_no_private_paths(inputs, tmp_path):
    h, root, build, _ = inputs
    identity = h.identity(build, root)
    assert set(identity["protected_headers_sha256"]) == set(h.PROTECTED)
    assert str(tmp_path) not in str(identity)
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert h.identity(relocated / "out/issue120-production-build", relocated) == identity


def test_ambient_compiler_flags_are_rejected(monkeypatch):
    from enodia.tt.bench.issue120_production import host_build_inputs

    monkeypatch.setenv("CXXFLAGS", "-iquote /untrusted")
    with pytest.raises(ValueError, match="override"):
        host_build_inputs.environment()
