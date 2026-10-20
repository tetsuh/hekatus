"""Lightweight Issue #146 plan identity shared by host and device entrypoints.

This module deliberately has no host analysis or accelerator import.  The
revision and exact plan digests are one authority so the device preflight can
verify the owner plan in an environment that contains no SciPy.
"""

from pathlib import Path

DEFAULT_PLAN_PATH = Path(
    "docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json"
)
CURRENT_PLAN_REVISION = 4
AUTHORITATIVE_PLAN_RAW_SHA256 = "420b35d4c76dc7cd264b52e6f4b8fd4c1784b919364b7feef6a0e07b0a766b9b"
AUTHORITATIVE_PLAN_CANONICAL_SHA256 = "5c1afdb57eb128c5056b2b42d9eac672fda32da88f5e5b0fb7574b72c6e87a5a"
