"""Regenerate the non-authorizing compatible-production receipt from one selection.

Native/build verification reads local files only; this tool never runs native.
"""
from __future__ import annotations

import argparse
import json

from enodia.tt.bench.issue120_production import production_admission, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace only the program-owned production receipt")
    args = parser.parse_args()
    receipt = production_admission.make_receipt(runtime.program_identity())
    production_admission.validate_receipt(receipt, runtime.program_identity())
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.write:
        path = runtime.fixed_path(runtime.SOURCE / "production_admission.json", runtime.SOURCE, exists=False)
        path.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
