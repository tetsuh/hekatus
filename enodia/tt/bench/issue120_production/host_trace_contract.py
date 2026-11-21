"""Mechanical host-only successor proof; no arbitrary native mismatch exemption."""
from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
BASE_NATIVE_SHA256 = "f88288968958bdc7ba1cae385ae9b2de3ba2125f83e8356e2ca279828ff8d9e1"
BASE_BINARY_SHA256 = "bd0ac1384aeebcce715a9cb7527211453ef3378f28e868bbc85080fe7bc4c3c0"
BASE_CMAKE_SHA256 = "83a2e6b23752bc9d4ccbd3f9a0d34eb63befd0f6c7404ff675906d44eede657f"
BASE_SUPPORT = {
    "host_protocol.hpp": "679fce752a119d864965336f73b8e3008166813f34164d691b2d91724e7b1525",
    "host_sequence.hpp": "ed3a7d6c36de7c63f268a4abc075d732a985c370dae75ad1e63a94835ca79e8f",
    "protocol.hpp": "12b44c53730d4a80e9d0fd530126e3d14ef14ebbdaa6ad98122904728b11b2ed",
    "../issue120/CMakeLists.txt": "368b338ceacad300a19dbef4ba249d9f0d90a3e6682fe69de4837a1c67e7362d",
    "../issue120/native_result_io.hpp": "2b3c235adcea60590635d66ea0570cd209e18567dadea32f86e9ad94d730cea1",
    "../issue120/q4_host_sequence.hpp": "cd796ffbecb9f1fabab39efd159ab103b21d108a84ce59e012b310c71e60a344",
    "../issue120/q4_runtime_args.hpp": "017eb0ac1fce8154e34cd242923a89046073587f8994985b5cc3fcc89de96111",
}
APPROVED_HOST_HEADERS = {
    "host_trace.hpp": "e806ffa7752172aaf65b5013b330086c10a1513ac3ad408ff83584942ab911f4",
    "host_protocol_trace.hpp": "388b713e105e1dd0c24ed7e1ee251065da6dcec328a948cb0182891213000d63",
}
# Fixed text, not user-provided hooks. Reversal must recover the full measured source.
REPLACEMENTS = (
    ('#include "../issue120/native_result_io.hpp"', '#include "../../../enodia/tt/bench/issue120/native_result_io.hpp"'),
    ('#include "../issue120/q4_host_sequence.hpp"', '#include "../../../enodia/tt/bench/issue120/q4_host_sequence.hpp"'),
    ('#include "host_protocol.hpp"', '#include "host_protocol.hpp"\n#include "host_protocol_trace.hpp"'),
    ('    p::Traffic traffic(a.frames);', '    p::HostTrace trace;\n    p::Traffic traffic(a.frames);'),
    ('            load_start=now(); volatile uint32_t checksum=0;\n            try {',
     '            volatile uint32_t checksum=0;\n            try {\n                load_start=now();'),
    ('p::submit_fixed(a.repeats,a.frames,traffic,', 'p::submit_fixed_traced(a.repeats,a.frames,traffic,'),
    ('},[&](const auto& payload) { q1.enqueue_write_mesh_buffer(b.progress,payload.data(),false); });',
     '},[&](const auto& payload) { q1.enqueue_write_mesh_buffer(b.progress,payload.data(),false); },trace,now);\n                cpu_checksum=checksum; load_end=now();'),
    ('            } catch(...) { load_error=std::current_exception(); cancel=true; }\n            cpu_checksum=checksum; load_end=now();\n        });',
     '            } catch(...) { load_error=std::current_exception(); cancel=true; }\n        });'),
    ('EnqueueMeshWorkload(q0,work,false); enqueue_return=now();',
     'EnqueueMeshWorkload(q0,work,false); enqueue_return=now(); trace.capture_enqueue_return(enqueue_return,now);'),
    ('{{"schema","issue120-production-native-v1"}', '{{"schema","issue120-production-native-v2"}'),
    ('        {"observer_reads",observer_reads},{"routing",routing},',
     '        {"host_phase_trace",trace.describe<Json>(a.loaded,load_start,load_end,joined)},\n        {"observer_reads",observer_reads},{"routing",routing},'),
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def derive_native(base: str) -> str:
    if digest(base.encode()) != BASE_NATIVE_SHA256:
        raise ValueError("measured native base differs")
    for old, new in REPLACEMENTS:
        if base.count(old) != 1:
            raise ValueError("host trace hook is not unique")
        base = base.replace(old, new, 1)
    return base


def verify_native_source(text: str) -> dict:
    original = text
    for old, new in reversed(REPLACEMENTS):
        if text.count(new) != 1:
            raise ValueError("host trace successor differs from the fixed insertions")
        text = text.replace(new, old, 1)
    if digest(text.encode()) != BASE_NATIVE_SHA256:
        raise ValueError("device/dispatch or base native source changed")
    if digest((SOURCE / "native.cpp").read_bytes()) != BASE_NATIVE_SHA256:
        raise ValueError("measured native source was modified")
    for name, expected in BASE_SUPPORT.items():
        if digest((SOURCE / name).read_bytes()) != expected:
            raise ValueError("measured protocol or lifecycle changed")
    headers = _host_build_proof()
    return {"schema": "issue120-host-only-source-proof-v1", "device_dispatch_unchanged": True,
            "base_native_source_sha256": BASE_NATIVE_SHA256, "base_native_sha256": BASE_BINARY_SHA256,
            "successor_source_sha256": digest(original.encode()), "host_headers_sha256": headers,
            "preserved_support_sha256": BASE_SUPPORT, "base_cmake_sha256": BASE_CMAKE_SHA256,
            "cmake_host_source_only_change": True, "insertions": len(REPLACEMENTS)}


def _host_build_proof() -> dict:
    cmake = (SOURCE / "CMakeLists.txt").read_text()
    target = 'add_executable(issue120_production "${CMAKE_CURRENT_BINARY_DIR}/host-trace-input/native_trace.cpp")'
    include = 'target_include_directories(issue120_production BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")\n'
    if cmake.count(target) != 1 or cmake.count(include) != 1:
        raise ValueError("host successor build target differs")
    base = cmake.replace(target, "add_executable(issue120_production native.cpp)", 1).replace(include, "", 1)
    if digest(base.encode()) != BASE_CMAKE_SHA256:
        raise ValueError("host build flags or dependencies changed")
    actual = {name: digest((SOURCE / name).read_bytes()) for name in APPROVED_HOST_HEADERS}
    if actual != APPROVED_HOST_HEADERS:
        raise ValueError("reviewed host-only header changed")
    return actual
