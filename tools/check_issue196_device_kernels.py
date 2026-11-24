#!/usr/bin/env python3
"""Run only inside the fixed masked/no-device compiler container; never a TT test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enodia.tt.bench.issue196_compile import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("red", "green"))
    return run(parser.parse_args().mode)


if __name__ == "__main__":
    raise SystemExit(main())
