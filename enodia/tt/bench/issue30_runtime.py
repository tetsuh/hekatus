"""Compose a fresh attempt-owned runtime from immutable pinned image assets.

No accelerator imports. Only generated output below CAPTURE is writable.
The supported TT_METAL_RUNTIME_ROOT controls both normal JIT includes and exports;
TT_METAL_KERNEL_PATH remains the pinned official source-file search root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

IMAGE_ROOT = Path("/opt/venv/lib/python3.12/site-packages/ttnn")
SOURCE_ROOT = Path("/source")
CAPTURE = Path("/capture")
SCRIPT_ROOT = Path(__file__).resolve().parents[3]


def validate_image():
    sys.path.insert(0, str(SCRIPT_ROOT))
    from enodia.tt.bench.issue120_runtime import validate_runtime_assets

    return validate_runtime_assets()


def prepare():
    if validate_image() != IMAGE_ROOT:
        raise ValueError("pinned image runtime root differs")
    if CAPTURE.resolve() != CAPTURE.absolute() or not (SOURCE_ROOT / "tests").is_dir():
        raise ValueError("runtime layout roots are missing or redirected")
    root = CAPTURE / "runtime"
    root.mkdir(mode=0o700, exist_ok=False)
    for name in ("tt_metal", "runtime", "build"):
        (root / name).symlink_to(IMAGE_ROOT / name, target_is_directory=True)
    if (IMAGE_ROOT / "ttnn").is_dir():
        (root / "ttnn").symlink_to(IMAGE_ROOT / "ttnn", target_is_directory=True)
    (root / "tests").symlink_to(SOURCE_ROOT / "tests", target_is_directory=True)
    (root / "generated/fabric").mkdir(mode=0o700, parents=True)
    record = {"schema": "issue196-attempt-runtime-v1", "image_asset_root": str(IMAGE_ROOT),
              "runtime_root": str(root), "test_source_root": str(SOURCE_ROOT / "tests"),
              "generated_output": str(root / "generated/fabric"),
              "scope": "image/source asset bytes unchanged; only new attempt output is writable"}
    with (CAPTURE / "runtime-layout.json").open("x") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("A", "B"))
    parser.add_argument("--list-tests", action="store_true")
    args = parser.parse_args()
    root = prepare()
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(root)
    os.environ["TT_METAL_KERNEL_PATH"] = str(SOURCE_ROOT)
    os.environ["TT_METAL_CACHE"] = str(CAPTURE / "cache")
    os.environ["TT_METAL_LOGS_PATH"] = str(CAPTURE / "logs")
    binary, selector = {"A": ("test_system_health", "Cluster.ReportSystemHealth"),
                        "B": ("fabric_smoke_tests", "Fabric1DFixture.TestUnicastConnAPI1DSmoke")}[args.phase]
    command = ["/assets/" + binary]
    command += (["--gtest_list_tests"] if args.list_tests else
                ["--gtest_filter=" + selector, "--gtest_output=xml:/capture/native/gtest.xml"])
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
