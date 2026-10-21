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
CURRENT_PLAN_REVISION = 6
AUTHORITATIVE_PLAN_RAW_SHA256 = "d9e1b4b1b73097e7873b9e5fe0a2d3af5c0665990688cb8ecaafbbc5355a1197"
AUTHORITATIVE_PLAN_CANONICAL_SHA256 = "1c46dae48245ee7e12f00f8c3717d6f5101bbae6b978db7c0ddb9b4d7addbf2e"
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
