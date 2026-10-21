"""Lightweight Issue #146 plan identity shared by host and device entrypoints.

This module deliberately has no host analysis or accelerator import.  The
revision and exact plan digests are one authority so the device preflight can
verify the owner plan in an environment that contains no SciPy.
"""

from pathlib import Path

DEFAULT_PLAN_PATH = Path(
    "docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json"
)
CURRENT_PLAN_REVISION = 5
AUTHORITATIVE_PLAN_RAW_SHA256 = "19a7c45de299b8e15b2d2e69ffa8a100091701ff2b8e32ca41980b465aade41b"
AUTHORITATIVE_PLAN_CANONICAL_SHA256 = "5fc8d7a19533e10117e1b862603e5cb32ca37549b03194367b3396637dbff3ef"
