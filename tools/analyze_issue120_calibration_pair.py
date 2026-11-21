"""Read C1/C2 artifacts and print explicit compatible, non-authorizing scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from enodia.tt.bench.issue120_production import recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs=2, metavar="RECORD")
    args = parser.parse_args()
    print(json.dumps(recording.calibration_pair_analysis(args.records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
