"""Build exact upstream test translation units inside the pinned, isolated image.

Run only with /source, /deps and /recipe read-only and /build writable. No
runtime/device calls. Failed builds retain receipts and byte-exact diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

IMAGE = "sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"
ROOT = Path("/build")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if os.getuid() == 0 or Path("/dev/tenstorrent").exists():
        raise RuntimeError("requires non-root device-free sandbox")
    for name in ("/sys", "/proc/bus", "/proc/driver"):
        if any(Path(name).iterdir()):
            raise RuntimeError("discovery not masked")
    receipt = {"schema": "issue30-build-v1", "image_digest": IMAGE, "all_passed": False,
               "scope": "unmodified upstream test translation units, pinned runtime libraries",
               "inputs": {"build_driver": sha(Path(__file__))}, "outputs": {}, "steps": []}
    expected = {"source": ("/source", "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd"),
                "umd": ("/source/tt_metal/third_party/umd", "bc77ac2f7382ffeb9fa5430055f0854d53979489"),
                "descriptors": ("/source/tt_metal/third_party/tt-cluster-descriptors", "7b2176e2fe913089f8cd2be9dfb738ead6e7aa27"),
                "tracy": ("/source/tt_metal/third_party/tracy", "27221a69789f24d492135093e096b978b4ca3a68")}
    receipt["revisions"] = {}
    for role, (directory, pin) in expected.items():
        git = ["git", "-c", "safe.directory=" + directory, "-C", directory]
        observed = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(git + ["status", "--porcelain"], env=os.environ | {"GIT_OPTIONAL_LOCKS": "0"})
        if observed != pin or dirty:
            raise RuntimeError("pinned source state mismatch: " + role)
        receipt["revisions"][role] = observed
    for dependency in Path("/deps").iterdir():
        receipt["revisions"][dependency.name] = subprocess.check_output(
            ["git", "-c", "safe.directory=" + str(dependency), "-C", str(dependency), "rev-parse", "HEAD"], text=True).strip()
    bases = {"source": Path("/source"), "dependencies": Path("/deps"), "recipe": Path("/recipe")}
    for name in ("tt_metal", "tt_stl", "tt-umd"):
        p = Path(f"/opt/venv/lib/python3.12/site-packages/ttnn/build/lib/lib{name}.so")
        receipt["inputs"]["runtime/lib" + name + ".so"] = sha(p)
    for role, base in bases.items():
        if role == "dependencies" or role == "recipe":
            files = [p for p in base.rglob("*") if p.is_file() and ".git" not in p.parts]
        else:
            files = [base / s for s in (
                "tests/tt_metal/tt_fabric/system_health/test_system_health.cpp",
                "tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp",
                "third_party/CMakeLists.txt", "tests/CMakeLists.txt")]
        for path in sorted(files):
            receipt["inputs"][role + "/" + path.relative_to(base).as_posix()] = sha(path)
    for role, command in (
        ("compiler", ["/usr/bin/c++", "--version"]),
        ("configure", ["cmake", "-S", "/recipe", "-B", "/build/native", "-G", "Ninja",
                       "-DCMAKE_CXX_COMPILER=/usr/bin/c++", "-DCMAKE_BUILD_TYPE=Release"]),
        ("compile", ["cmake", "--build", "/build/native", "--parallel", "2"]),
    ):
        with (ROOT / (role + ".stdout.bin")).open("xb") as stdout, (ROOT / (role + ".stderr.bin")).open("xb") as stderr:
            r = subprocess.run(command, stdout=stdout, stderr=stderr, check=False, timeout=240)
        receipt["steps"].append({"role": role, "argv": command, "exit": r.returncode})
        if r.returncode:
            break
    receipt["all_passed"] = len(receipt["steps"]) == 3 and receipt["steps"][-1]["exit"] == 0
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "build-receipt.json":
            receipt["outputs"][path.relative_to(ROOT).as_posix()] = sha(path)
    for label, name in (("compiler", "/usr/bin/c++"), ("linker", "/usr/bin/ld")):
        receipt["inputs"][label] = sha(Path(name))
    with (ROOT / "build-receipt.json").open("x") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")
    return 0 if receipt["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
