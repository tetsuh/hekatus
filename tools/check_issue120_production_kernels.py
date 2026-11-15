"""Check the actual versioned resident adapters with pinned, device-free SFPI."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "enodia/tt/bench/issue120_production"
OUTPUT = Path("/out/issue120-production-device-check")
COMPILER = Path("/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++")


def _require_isolation() -> None:
    if Path("/dev/tenstorrent").exists():
        raise RuntimeError("device nodes must not be exposed")
    for path in (Path("/sys"), Path("/proc/bus"), Path("/proc/driver")):
        if path.exists() and any(path.iterdir()):
            raise RuntimeError("hardware discovery must be masked")


def _check_case(name: str, unit: Path, args: list[int], flags: list[str], watcher: bool) -> dict:
    command = [str(COMPILER), *flags,
               "-DKERNEL_COMPILE_TIME_ARGS=" + ",".join(map(str, args)),
               *(["-DWATCHER_ENABLED"] if watcher else []), "-fsyntax-only", str(unit)]
    result = subprocess.run(command, capture_output=True, check=False, timeout=60)
    for stream in ("stdout", "stderr"):
        (OUTPUT / (name + "." + stream)).write_bytes(getattr(result, stream))
    print(name, result.returncode, flush=True)
    return {"case": name, "compile_time_args": args, "returncode": result.returncode,
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()}


def _check_role(role: str, kernel: Path, args: list[int], flags: list[str]) -> list[dict]:
    if not args or any(type(x) is not int or not 0 <= x < 2**32 for x in args):
        raise ValueError("invalid compile words")
    unit = OUTPUT / (role + ".cpp")
    unit.write_text('#include "internal/firmware_common.h"\n' + f'#include "{kernel}"\n')
    return [_check_case(f"{role}-watcher-{int(watcher)}", unit, args, flags, watcher)
            for watcher in (False, True)]


def main() -> int:
    _require_isolation()
    from enodia.tt.bench.issue120_production.kernel_adapter import generate

    helper = ROOT / "tools/check_issue178_device_kernels.py"
    spec = importlib.util.spec_from_file_location("pinned_sfpi_flags", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    OUTPUT.mkdir(exist_ok=False)
    kernels = generate(OUTPUT / "kernels")
    manifest = json.loads((SOURCE / "compile_args.json").read_bytes())
    if manifest.get("schema") != "issue120-production-compile-args-v1":
        raise ValueError("unsupported production compile manifest")
    cases = []
    for role, kernel in kernels.items():
        cases.extend(_check_role(role, kernel, manifest[role], module._flags()))
    report = {"schema": "issue120-production-device-check-v1", "cases": cases,
              "all_passed": all(x["returncode"] == 0 for x in cases),
              "scope": "syntax/constexpr only; not device link, dispatch or execution",
              "compiler_sha256": hashlib.sha256(COMPILER.read_bytes()).hexdigest(),
              "flags_helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256((SOURCE / "compile_args.json").read_bytes()).hexdigest(),
              "derived_sources": json.loads((OUTPUT / "kernels/sources.json").read_bytes())}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
