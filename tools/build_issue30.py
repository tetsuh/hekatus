"""Build upstream test units inside the pinned, isolated image, without devices."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

IMAGE = "sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"
ROOT = Path("/build")
PINS = {"source": ("/source", "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd"),
        "umd": ("/source/tt_metal/third_party/umd", "bc77ac2f7382ffeb9fa5430055f0854d53979489"),
        "descriptors": ("/source/tt_metal/third_party/tt-cluster-descriptors", "7b2176e2fe913089f8cd2be9dfb738ead6e7aa27"),
        "tracy": ("/source/tt_metal/third_party/tracy", "27221a69789f24d492135093e096b978b4ca3a68")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolation():
    if os.getuid() == 0 or Path("/dev/tenstorrent").exists():
        raise RuntimeError("requires non-root device-free sandbox")
    for name in ("/sys", "/proc/bus", "/proc/driver"):
        if any(Path(name).iterdir()):
            raise RuntimeError("discovery not masked")


def _git(directory, *args):
    return subprocess.check_output(["git", "-c", "safe.directory=" + directory, "-C", directory, *args],
                                   text=True, env=os.environ | {"GIT_OPTIONAL_LOCKS": "0"}).strip()


def _revisions():
    revisions = {}
    for role, (directory, pin) in PINS.items():
        observed = _git(directory, "rev-parse", "HEAD")
        if observed != pin or _git(directory, "status", "--porcelain"):
            raise RuntimeError("pinned source state mismatch: " + role)
        revisions[role] = observed
    for dependency in Path("/deps").iterdir():
        revisions[dependency.name] = _git(str(dependency), "rev-parse", "HEAD")
    return revisions


def _files(role, base):
    if role == "source":
        return [base / s for s in ("tests/tt_metal/tt_fabric/system_health/test_system_health.cpp",
                                   "tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp",
                                   "third_party/CMakeLists.txt", "tests/CMakeLists.txt")]
    return [p for p in base.rglob("*") if p.is_file() and ".git" not in p.parts]


def _inputs():
    inputs = {"build_driver": sha(Path(__file__))}
    for name in ("tt_metal", "tt_stl", "tt-umd"):
        path = Path(f"/opt/venv/lib/python3.12/site-packages/ttnn/build/lib/lib{name}.so")
        inputs["runtime/lib" + name + ".so"] = sha(path)
    for role, base in {"source": Path("/source"), "dependencies": Path("/deps"), "recipe": Path("/recipe")}.items():
        for path in sorted(_files(role, base)):
            inputs[role + "/" + path.relative_to(base).as_posix()] = sha(path)
    inputs["compiler"] = sha(Path("/usr/bin/c++"))
    inputs["linker"] = sha(Path("/usr/bin/ld"))
    return inputs


def _steps():
    steps = []
    commands = (("compiler", ["/usr/bin/c++", "--version"]),
                ("configure", ["cmake", "-S", "/recipe", "-B", "/build/native", "-G", "Ninja",
                               "-DCMAKE_CXX_COMPILER=/usr/bin/c++", "-DCMAKE_BUILD_TYPE=Release"]),
                ("compile", ["cmake", "--build", "/build/native", "--parallel", "2"]))
    for role, command in commands:
        with (ROOT / (role + ".stdout.bin")).open("xb") as stdout, (ROOT / (role + ".stderr.bin")).open("xb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False, timeout=240)
        steps.append({"role": role, "argv": command, "exit": result.returncode})
        if result.returncode:
            break
    return steps


def main():
    _isolation()
    receipt = {"schema": "issue30-build-v1", "image_digest": IMAGE,
               "scope": "unmodified upstream test translation units, pinned runtime libraries",
               "revisions": _revisions(), "inputs": _inputs(), "steps": _steps()}
    steps = receipt["steps"]
    receipt["all_passed"] = len(steps) == 3 and steps[-1]["exit"] == 0
    receipt["outputs"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*"))
                          if p.is_file() and p.name != "build-receipt.json"}
    with (ROOT / "build-receipt.json").open("x") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")
    return 0 if receipt["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
