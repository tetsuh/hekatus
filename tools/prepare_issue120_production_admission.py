"""Regenerate a compatible-production receipt; explicit adoption records owner authority.

Native/build verification reads local files only; this tool never runs native.
"""
from __future__ import annotations

import argparse
import json

from enodia.tt.bench.issue120_production import production_admission, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace only the program-owned production receipt")
    parser.add_argument("--adopt-owner-decision", action="store_true", help="record the dated committed owner adoption as explicit v2")
    args = parser.parse_args()
    adoption = production_admission.owner_adoption() if args.adopt_owner_decision else None
    receipt = production_admission.make_receipt(runtime.program_identity(), adoption=adoption)
    production_admission.validate_receipt(receipt, runtime.program_identity())
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.write:
        path = runtime.fixed_path(runtime.SOURCE / "production_admission.json", runtime.SOURCE, exists=False)
        path.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
