"""Bind effective compiler flags and resolved protected headers, not intentions."""
from __future__ import annotations

import hashlib
import os
import posixpath
import shlex
from pathlib import Path

PROD = "enodia/tt/bench/issue120_production/"
LEGACY = "enodia/tt/bench/issue120/"
PROTECTED = tuple(PROD + name for name in ("protocol.hpp", "host_protocol.hpp", "host_protocol_trace.hpp", "host_trace.hpp", "host_sequence.hpp")) + tuple(
    LEGACY + name for name in ("native_result_io.hpp", "q4_host_sequence.hpp", "q4_runtime_args.hpp"))
INCLUDE_PATHS = ("enodia/tt/bench/issue120_production", "out/issue120-production-build/legacy/generated",
                 "out/issue120-deps/source", "out/issue120-deps/source/tt_metal", "out/issue120-deps/source/tt_metal/impl",
                 "out/issue120-deps/source/tt_metal/api/tt-metalium", "out/issue120-deps/source/tt_metal/api",
                 "out/issue120-deps/source/tt_metal/hostdevcommon/api", "out/issue120-deps/source/tt_metal/hostdevcommon",
                 "out/issue120-deps/source/tt_metal/hw/inc", "out/issue120-deps/source/tt_stl",
                 "out/issue120-deps/source/tt_metal/third_party/umd/device/api", "out/issue120-deps/include")
DEFINES = r'-DFMT_HEADER_ONLY -DISSUE120_SOURCE_DIR=\"enodia/tt/bench/issue120\" -DSPDLOG_FMT_EXTERNAL'
RULES = "CMakeFiles/issue120_production.dir"
DEPENDENCY = RULES + "/host-trace-input/native_trace.cpp.o.d"
OVERRIDES = {"CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "CPLUS_INCLUDE_PATH",
             "C_INCLUDE_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH", "LD_PRELOAD", "LD_LIBRARY_PATH"}


def environment() -> dict:
    env = os.environ.copy()
    if any(value and (key in OVERRIDES or key.startswith("CMAKE_")) for key, value in env.items()):
        raise ValueError("ambient compiler/build override is not admitted")
    return env


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: str, prefix: str) -> str:
    path = posixpath.normpath(path)
    if not path.startswith(prefix + "/"):
        raise ValueError("compiler path is outside its verified project prefix")
    return path[len(prefix) + 1:]


def _cache(build: Path) -> None:
    values = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    expected = {"CMAKE_BUILD_TYPE": "", "CMAKE_CXX_FLAGS": "", "CMAKE_CXX_COMPILER": "/usr/bin/c++"}
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("cached compiler/build flags differ from the B4 configuration")
    forbidden = ("CMAKE_TOOLCHAIN_FILE", "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_ARG1",
                 "CMAKE_PROJECT_INCLUDE", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES", "CMAKE_USER_MAKE_RULES_OVERRIDE")
    if any(values.get(key) for key in forbidden):
        raise ValueError("cached compiler/toolchain override is not admitted")


def verify_flags(build: Path) -> str:
    _cache(build)
    flags = {}
    for line in (build / RULES / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = value.strip()
    if flags.get("CXX_FLAGS") != "-std=gnu++20" or flags.get("CXX_DEFINES") != DEFINES:
        raise ValueError("effective compiler flags/definitions changed")
    tokens = shlex.split(flags.get("CXX_INCLUDES", ""))
    suffix = "/" + INCLUDE_PATHS[0]
    if not tokens or not tokens[0].startswith("-I") or not tokens[0][2:].endswith(suffix):
        raise ValueError("protected host include priority differs")
    prefix = tokens[0][2:][:-len(suffix)]
    if any(not token.startswith("-I") for token in tokens):
        raise ValueError("effective compiler include mode changed")
    if tuple(_relative(token[2:], prefix) for token in tokens) != INCLUDE_PATHS:
        raise ValueError("effective compiler include paths differ")
    return prefix


def identity(build: Path, root: Path) -> dict:
    prefix = verify_flags(build)
    dependency = build / DEPENDENCY
    tokens = shlex.split(dependency.read_text().replace("\\\n", " "))[1:]
    expected = {Path(name).name: name for name in PROTECTED}
    seen = set()
    for token in tokens:
        name = posixpath.basename(token)
        if name in expected:
            relative = _relative(token, prefix)
            if relative != expected[name]:
                raise ValueError("compiler resolved a shadow protected header")
            seen.add(relative)
    if seen != set(PROTECTED):
        raise ValueError("compiler dependency proof is missing a protected header")
    return {"schema": "issue120-host-compiler-inputs-v1", "compiler": "/usr/bin/c++",
            "configuration": "empty-build-type/no-user-flags", "include_paths": list(INCLUDE_PATHS),
            "flags_sha256": _sha(build / RULES / "flags.make"), "cache_sha256": _sha(build / "CMakeCache.txt"),
            "dependency_sha256": _sha(dependency),
            "protected_headers_sha256": {name: _sha(root / name) for name in PROTECTED}}
