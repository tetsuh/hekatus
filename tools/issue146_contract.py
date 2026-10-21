"""Lightweight Issue #146 plan identity shared by host and device entrypoints.

This module deliberately has no host analysis or accelerator import.  The
revision and exact plan digests are one authority so the device preflight can
verify the owner plan in an environment that contains no SciPy.
"""

from pathlib import Path
from typing import Any

DEFAULT_PLAN_PATH = Path(
    "docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json"
)
CURRENT_PLAN_REVISION = 5
AUTHORITATIVE_PLAN_RAW_SHA256 = "19a7c45de299b8e15b2d2e69ffa8a100091701ff2b8e32ca41980b465aade41b"
AUTHORITATIVE_PLAN_CANONICAL_SHA256 = "5fc8d7a19533e10117e1b862603e5cb32ca37549b03194367b3396637dbff3ef"
SMOKE_SAMPLE_WINDOW_MARKER = (
    "10 s <= (timestamp_monotonic_ns - first_launch_start_monotonic_ns) < 40 s"
)


def parse_smoke_sample_window(value: Any) -> tuple[float, float]:
    """Parse the registered smoke window without an unbounded regex."""
    if (
        not isinstance(value, str)
        or SMOKE_SAMPLE_WINDOW_MARKER not in value
        or "outside every item" not in value
    ):
        raise ValueError("Issue #146 smoke sample window is malformed")
    return 10.0, 40.0
