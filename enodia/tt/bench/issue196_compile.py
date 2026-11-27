"""Fixed offline compile inventory from immutable B output; no TT runtime imports."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path

from enodia.tt.bench import issue30_runtime as runtime

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "enodia/tt/bench/issue196_compile_manifest.json"
INPUTS = ROOT / "docs/measurements/issue196/offline-inputs.tar.gz"
FIXTURE = ROOT / "docs/measurements/issue196/official-smoke-source.txt"
B_ARCHIVE = ROOT / "docs/measurements/2026-11-24-issue30-b-smoke/B-native-evidence.tar.gz"
COMPILER = "/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++"
OLD_ROOT = "/opt/venv/lib/python3.12/site-packages/ttnn"
DEPENDENCY_FILE = "kernel.d"
PROCESSORS = {"brisc", "ncrisc", "trisc0", "trisc1", "trisc2", "active_erisc", "subordinate_active_erisc"}
EXPECTED_IDS = {f"observed-{i:02d}-{p}" for i, p in enumerate(
    ["brisc", "ncrisc", "trisc0", "trisc1", "trisc2", "brisc"] +
    ["subordinate_active_erisc", "subordinate_active_erisc", "active_erisc",
     "subordinate_active_erisc", "active_erisc", "subordinate_active_erisc", "active_erisc",
     "active_erisc", "subordinate_active_erisc", "subordinate_active_erisc", "active_erisc",
     "active_erisc", "active_erisc", "subordinate_active_erisc", "subordinate_active_erisc", "active_erisc"])} | {
    "captured-rx", "derived-tx"}
FLAGS = ("-O2", "-std=c++17", "-ftt-nttp", "-ftt-constinit", "-ftt-consteval", "-ftt-no-dyninit",
         "-flto=auto", "-ffast-math", "-fno-exceptions", "-fno-rtti", "-fno-use-cxa-atexit", "-g",
         "-MMD", "-Wall", "-Werror", "-Wno-error=deprecated-declarations",
         "-Wno-error=multistatement-macros", "-Wno-error=parentheses", "-Wno-error=unused-but-set-variable",
         "-Wno-unused-variable", "-Wno-unused-function", "-mcpu=tt-bh", "-fno-tree-loop-distribute-patterns")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def kernel_inventory():
    with tarfile.open(B_ARCHIVE, "r:gz") as archive:
        raw = archive.extractfile("logs/generated/watcher/kernel_names.txt").read()
    observed = set(re.findall(r"^\d+: (.+\.cpp)$", raw.decode(), re.MULTILINE))
    fixture = set(re.findall(r'"(tests/[^"\n]+\.cpp)"', FIXTURE.read_text()))
    if not fixture <= observed:
        raise ValueError("fixture kernel inventory differs from retained names")
    return observed | fixture


def validate_manifest(value):
    cases = value.get("cases", [])
    if (value.get("schema") != "issue196-offline-kernel-manifest-v1"
            or {x["id"] for x in cases} != EXPECTED_IDS or len(cases) != len(EXPECTED_IDS)
            or {x["kernel"] for x in cases} != kernel_inventory()
            or set(value.get("kernel_sources", [])) != kernel_inventory()):
        raise ValueError("kernel inventory differs")
    original = value["failed_argv"]
    prefix = original[1:original.index("-c")]
    original_includes = [x for x in prefix if x.startswith("-I")]
    if value["captured_includes"] != original_includes or sha(FIXTURE.read_bytes()) != value["origin_sha256"]["official_smoke"]:
        raise ValueError("fixture or captured include roots differ")
    if tuple(value["captured_common_flags"]) != FLAGS or value["failed_argv"][0] != COMPILER:
        raise ValueError("compiler options differ from captured control")
    for case in cases:
        if case.get("processor") not in PROCESSORS:
            raise ValueError("unsupported processor option")
        if case.get("optimization") not in ("O2", "O3", "Os"):
            raise ValueError("unsafe compiler optimization option")
        if not case["defines"] or any("BENCHMARK_MODE" in x or x.startswith("-") for x in case["defines"]):
            raise ValueError("unsafe or benchmark compiler option")


    _validate_provenance(value)


def _validate_provenance(value):
    # Freeze associations as well as set projections: case ID, processor, flags,
    # source/support hashes and original argv are one provenance object.
    if (set(value.get("kernel_source_sha256", {})) != kernel_inventory()
            or any(not re.fullmatch(r"[a-f0-9]{64}", x)
                   for x in value["kernel_source_sha256"].values())):
        raise ValueError("compile source hash provenance differs")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if sha(canonical) != "a407b8e54cd79bf8cb7c6956c7786ccae91292043b58af16fbe670feb4a1f714":
        raise ValueError("compile case provenance differs")


def isolated():
    if Path("/dev/tenstorrent").exists():
        raise ValueError("device mappings are forbidden")
    for name in ("/sys", "/proc/bus", "/proc/driver"):
        if any(Path(name).iterdir()):
            raise ValueError("device discovery must be masked")


def support_files(value, output):
    if sha(INPUTS.read_bytes()) != value["input_archive_sha256"]:
        raise ValueError("compile inputs differ")
    with tarfile.open(INPUTS, "r:gz") as archive:
        members = archive.getmembers()
        if {x.name for x in members} != set(value["support_sha256"]):
            raise ValueError("support inventory differs")
        for member in members:
            if not member.isfile() or member.type != tarfile.REGTYPE or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("unsafe generated input")
            raw = archive.extractfile(member).read()
            if sha(raw) != value["support_sha256"][member.name]:
                raise ValueError("generated input hash differs")
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)


def options(value, case, root):
    flags = ["-" + case["optimization"] if x == "-O2" else x for x in value["captured_common_flags"]]
    includes = [x.replace(OLD_ROOT, str(root)) for x in value["captured_includes"] if not x.startswith("-I/source/")]
    processor = case["processor"]
    extra = []
    if processor.startswith("trisc"):
        flags = [x.replace("-O2", "-O3").replace("-mcpu=tt-bh", "-mcpu=tt-bh-tensix")
                 for x in flags if x != "-fno-tree-loop-distribute-patterns"]
        index = next(i for i, x in enumerate(includes) if "/hw/ckernels/" in x)
        includes.insert(index, "-I/opt/tenstorrent/sfpi/include")
        extra = ["tt_metal/hw/ckernels/blackhole/metal/llk_api",
                 "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"]
    elif "active_erisc" in processor:
        extra = ["tt_metal/hw/inc/ethernet"]
        if processor == "active_erisc":
            flags.append("-Werror=stack-usage=1912")
    # This is the normal pinned HAL per-core include list, not an added source-root workaround.
    firmware = includes.pop()
    includes += ["-I" + str(root / path) for path in extra] + [firmware]
    includes.append("-I" + str(runtime.SOURCE_ROOT / Path(case["kernel"]).parent))
    return flags + includes + ["-D" + x for x in case["defines"]]


def driver(case):
    p = case["processor"]
    name = {"brisc": "brisck.cc", "ncrisc": "ncrisck.cc"}.get(p)
    if p.startswith("trisc"):
        name = "trisck.cc"
    if "active_erisc" in p:
        name = "active_erisck.cc"
    if name is None:
        raise ValueError("unsupported processor")
    return "tt_metal/hw/firmware/src/tt-1xx/" + name


def dependency_hashes(path):
    text = path.read_text().replace("\\\n", " ").partition(":")[2]
    names = sorted(set(text.split()))
    hashes = []
    allowed = (runtime.CAPTURE, runtime.IMAGE_ROOT, runtime.SOURCE_ROOT, Path("/opt/tenstorrent/sfpi"))
    for name in names:
        source = (Path(name) if Path(name).is_absolute() else path.parent / name).resolve()
        if not any(source.is_relative_to(base) for base in allowed) or not source.is_file():
            raise ValueError("dependency escaped fixed input/output roles")
        hashes.append({"path": name, "sha256": sha(source.read_bytes())})
    return hashes


def compile_case(value, case, root, inputs):
    working = inputs / case["id"] / case["processor"]
    working.mkdir()
    output = working / "kernel.o"
    command = [COMPILER, *options(value, case, root), "-c", "-o", str(output),
               str(root / driver(case)), "-MF", str(working / DEPENDENCY_FILE)]
    start = time.monotonic()
    done = subprocess.run(command, cwd=working, capture_output=True, check=False, timeout=40)
    elapsed = time.monotonic() - start
    for role, data in (("stdout", done.stdout), ("stderr", done.stderr)):
        with (working / role).open("xb") as stream:
            stream.write(data)
    return {"id": case["id"], "kernel": case["kernel"], "processor": case["processor"],
            "origin": case["origin"], "argv": command, "exit": done.returncode,
            "elapsed_seconds": elapsed, "stdout_sha256": sha(done.stdout), "stderr_sha256": sha(done.stderr),
            "object_sha256": sha(output.read_bytes()) if output.is_file() else None,
            "dependency_sha256": sha((working / DEPENDENCY_FILE).read_bytes()) if (working / DEPENDENCY_FILE).is_file() else None,
            "missing_include_control": b"kernel_utils.hpp: No such file or directory" in done.stderr,
            "input_dependencies": dependency_hashes(working / DEPENDENCY_FILE) if done.returncode == 0 else []}


def run(mode):
    isolated()
    value = json.loads(MANIFEST.read_bytes())
    validate_manifest(value)
    if any(sha((runtime.SOURCE_ROOT / name).read_bytes()) != digest for name, digest in value["kernel_source_sha256"].items()):
        raise ValueError("official kernel source bytes changed")
    root = runtime.IMAGE_ROOT if mode == "red" else runtime.prepare()
    export = None
    if mode == "green":
        target = root / "generated/fabric/layout-probe.yaml"
        with target.open("x") as stream:
            stream.write("scope: device-free-writeability-probe\n")
        export = {"path": str(target), "sha256": sha(target.read_bytes()),
                  "owned": target.resolve().is_relative_to(runtime.CAPTURE)}
    inputs = runtime.CAPTURE / "offline-inputs"
    inputs.mkdir()
    support_files(value, inputs)
    selected = [x for x in value["cases"] if x["id"] == "captured-rx"] if mode == "red" else value["cases"]
    start = time.monotonic()
    results = []
    for case in selected:
        item = compile_case(value, case, root, inputs)
        results.append(item)
        print(case["id"], item["exit"], round(item["elapsed_seconds"], 3), flush=True)
    passed = (results[0]["exit"] != 0 and results[0]["missing_include_control"]) if mode == "red" else all(x["exit"] == 0 for x in results)
    report = {"schema": "issue196-device-free-compile-v1", "mode": mode, "cases": results,
              "all_passed": passed, "elapsed_seconds": time.monotonic() - start,
              "compiler_sha256": sha(Path(COMPILER).read_bytes()), "manifest_sha256": sha(MANIFEST.read_bytes()),
              "export_write_probe": export,
              "layout_scope": "same root and compiler roles as runtime; no per-kernel include-root patch",
              "limits": value["limits"]}
    with (runtime.CAPTURE / "compile-report.json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    return 0 if passed else 2
