"""Print a board-free C1-only admission from pinned B4 evidence and current build.

No session, device query or execution. Redirect the JSON into the fixed
production bounded_admission.json path, review it, and commit before future use.
"""
from __future__ import annotations

import json

from enodia.tt.bench.issue120_production import admission_bridge, runtime


def main() -> None:
    print(json.dumps(admission_bridge.make_receipt(runtime.program_identity()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
