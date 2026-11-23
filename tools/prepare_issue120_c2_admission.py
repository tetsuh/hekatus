"""Print a non-authorizing C2 receipt from unchanged native and original C1/B4."""
from __future__ import annotations

import json

from enodia.tt.bench.issue120_production import runtime, sessions


def main() -> None:
    receipt = sessions.make_c2_admission(runtime.program_identity())
    sessions.validate_c2_admission(receipt, runtime.program_identity())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
