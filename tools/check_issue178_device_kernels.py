#!/usr/bin/env python3
"""Syntax/constexpr-check Q4 with pinned SFPI inside a hardware-isolated image.

Run in the pinned image with network none, no devices, masked discovery paths,
read-only source/root, and a new writable /out. This never imports TT runtime.
The native host rejects production CTA vectors differing from this manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "enodia/tt/bench/issue120"
RUNTIME = Path("/opt/venv/lib/python3.12/site-packages/ttnn")
COMPILER = Path("/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++")
OUTPUT = Path("/out/issue178-device-check")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flags() -> list[str]:
    # From the retained run7 BRISC invocation; no shell or arbitrary options.
    options = [
        "-O2", "-std=c++17", "-ftt-nttp", "-ftt-constinit", "-ftt-consteval",
        "-ftt-no-dyninit", "-flto=auto", "-ffast-math", "-fno-exceptions", "-fno-rtti",
        "-fno-use-cxa-atexit", "-g", "-Wall", "-Werror",
        "-Wno-error=deprecated-declarations", "-Wno-error=multistatement-macros",
        "-Wno-error=parentheses", "-Wno-error=unused-but-set-variable",
        "-Wno-unused-variable", "-Wno-unused-function", "-mcpu=tt-bh",
        "-fno-tree-loop-distribute-patterns",
    ]
    includes = [
        "", "ttnn", "ttnn/cpp", "tt_metal", "tt_metal/hw/inc", "tt_metal/tt-llk/common",
        "tt_metal/hostdevcommon/api", "tt_metal/api",
        "tt_metal/hw/ckernels/blackhole/metal/common",
        "tt_metal/hw/ckernels/blackhole/metal/llk_io",
        "tt_metal/hw/inc/internal/tt-1xx", "tt_metal/hw/inc/internal/tt-1xx/blackhole",
        "tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines",
        "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc",
        "tt_metal/tt-llk/tt_llk_blackhole/common/inc",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib", "tt_metal/hw/firmware/src/tt-1xx",
    ]
    options += ["-I" + str(RUNTIME / path) for path in includes]
    options += ["-I" + str(SOURCE)]
    defines = [
        "IS_NOT_POW2_NUM_L1_BANKS=1", "LOG_BASE_2_OF_NUM_DRAM_BANKS=3",
        "NUM_DRAM_BANKS=8", "NUM_L1_BANKS=110", "PCIE_NOC_X=19", "PCIE_NOC_Y=24",
        "TENSIX_FIRMWARE", "LOCAL_MEM_EN=0", "PROCESSOR_INDEX=0", "COMPILE_FOR_BRISC",
        "ARCH_BLACKHOLE", "DISPATCH_MESSAGE_ADDR=4290184248", "KERNEL_BUILD",
        'FULL_KERNEL_NAME="issue178-offline/"', "NOC_INDEX=0", "NOC_MODE=0",
    ]
    return options + ["-D" + define for define in defines]


def _compile_role(layout: str, role: str, args: list[int]) -> list[dict]:
    if not args or not all(type(x) is int and 0 <= x < 2**32 for x in args):
        raise ValueError("invalid compile-time argument words")
    unit = OUTPUT / f"{layout}-{role}.cpp"
    unit.write_text('#include "internal/firmware_common.h"\n'
                    f'#include "{SOURCE}/q4_frame_{role}.cpp"\n')
    results = []
    for watcher in (False, True):
        name = f"{layout}-{role}-watcher-{int(watcher)}"
        command = [str(COMPILER), *_flags(),
                   "-DKERNEL_COMPILE_TIME_ARGS=" + ",".join(map(str, args)),
                   *(["-DWATCHER_ENABLED"] if watcher else []), "-fsyntax-only", str(unit)]
        completed = subprocess.run(command, capture_output=True, check=False, timeout=60)
        (OUTPUT / (name + ".stdout")).write_bytes(completed.stdout)
        (OUTPUT / (name + ".stderr")).write_bytes(completed.stderr)
        results.append({"case": name, "compile_time_args": args,
                        "returncode": completed.returncode,
                        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest()})
        print(name, completed.returncode, flush=True)
    return results


def main() -> int:
    if Path("/dev/tenstorrent").exists():
        raise RuntimeError("device nodes must not be exposed")
    for path in (Path("/sys"), Path("/proc/bus"), Path("/proc/driver")):
        if path.exists() and any(path.iterdir()):
            raise RuntimeError("hardware discovery paths must be masked")
    manifest_path = SOURCE / "q4_compile_args.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "issue178-device-compile-args-v3":
        raise ValueError("unsupported fixed compile manifest")
    OUTPUT.mkdir(exist_ok=False)
    results = []
    for layout in ("v3", "legacy_non_sharded"):
        for role in ("producer", "consumer"):
            results.extend(_compile_role(layout, role, manifest[layout][role]))
    report = {"schema": "issue178-offline-device-check-v3", "compiler_sha256": _hash(COMPILER),
              "manifest_sha256": _hash(manifest_path), "cases": results,
              "source_sha256": {name: _hash(SOURCE / name) for name in (
                  "q4_frame_producer.cpp", "q4_frame_consumer.cpp", "q4_clock.hpp",
                  "q4_runtime_args.hpp", "q4_accessor_diagnostics.hpp", "q4_protocol.hpp")},
              "scope": "syntax/constexpr only, not device link, dispatch or execution",
              "all_passed": all(case["returncode"] == 0 for case in results)}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
