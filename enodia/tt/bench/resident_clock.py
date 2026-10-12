"""Shared AICLK provenance and device-tick conversion helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

AICLK_SOURCE_CONFIGURED = "configured"
AICLK_SOURCE_RUN_TRACE_SAMPLES = "run_trace_samples"
AICLK_SOURCE_LEGACY_UNVERIFIED = "legacy_unverified"
_MAX_AICLK_MHZ = (1 << 63) - 1


def safe_aiclk_value(value: Any) -> float | None:
    """Return a finite positive AICLK value, retaining weighted fractions."""
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > _MAX_AICLK_MHZ:
        return None
    return parsed


def select_load_interval_aiclk(
    *,
    weighted_aiclk_mhz: Any,
    provenance: Any,
    required_provenance: str = "load_interval_trace_samples",
) -> dict[str, Any]:
    """Select the weighted AICLK for Issue #118 card-tick conversion.

    The ordinary ``clock.aiclk_mhz`` remains the existing integer maximum.  A
    loaded run gets a separate elapsed-time selection and is rejected unless
    its weighted value names the expected load-interval trace provenance.
    """
    weighted = safe_aiclk_value(weighted_aiclk_mhz)
    if weighted is None:
        raise ValueError("load-interval weighted AICLK is unavailable")
    if provenance != required_provenance:
        raise ValueError(
            "load-interval weighted AICLK lacks required trace provenance"
        )
    return {
        "aiclk_mhz": weighted,
        "aiclk_mhz_exact": weighted,
        "aiclk_source": provenance,
    }


def safe_aiclk_integer(value: Any) -> int | None:
    """Return a finite positive integer AICLK value, or ``None``."""
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > _MAX_AICLK_MHZ:
        return None
    try:
        converted = int(parsed)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if converted > 0 else None


def select_elapsed_aiclk(
    *,
    budget_aiclk_mhz: Any,
    sampler_mode: str | None,
    trace_aiclk_mhz: Any = None,
    trace_aiclk_source: str | None = None,
    legacy_aiclk_mhz: Any = None,
    legacy_aiclk_source: str = AICLK_SOURCE_LEGACY_UNVERIFIED,
    allow_configured_fallback: bool = False,
) -> dict[str, Any]:
    """Select the AICLK used to interpret device ticks as elapsed seconds.

    Sampler-off has no in-run trace, so it always uses the configured run
    budget clock.  Sampled runs require the exact in-run trace provenance
    sentinel; an environment snapshot is never a valid fallback.  The
    ``legacy_aiclk_mhz`` branch is limited to old board-free callers that lack
    trace bytes and is explicitly marked unverified.
    """
    configured = safe_aiclk_integer(budget_aiclk_mhz)
    if configured is None:
        raise ValueError("budget_aiclk_mhz must be a finite positive AICLK")
    mode = "off" if sampler_mode is None else sampler_mode
    if mode == "off":
        return {
            "aiclk_mhz": configured,
            "aiclk_source": AICLK_SOURCE_CONFIGURED,
        }
    if mode not in {"default", "explicit"}:
        raise ValueError(f"unsupported sampler mode for elapsed-time AICLK: {mode!r}")

    trace_aiclk = safe_aiclk_integer(trace_aiclk_mhz)
    if trace_aiclk_source == AICLK_SOURCE_RUN_TRACE_SAMPLES and trace_aiclk is not None:
        return {
            "aiclk_mhz": trace_aiclk,
            "aiclk_source": AICLK_SOURCE_RUN_TRACE_SAMPLES,
        }
    legacy = safe_aiclk_integer(legacy_aiclk_mhz)
    if legacy is not None:
        return {
            "aiclk_mhz": legacy,
            "aiclk_source": legacy_aiclk_source,
        }
    if allow_configured_fallback:
        return {
            "aiclk_mhz": configured,
            "aiclk_source": AICLK_SOURCE_CONFIGURED,
        }
    raise ValueError(
        "sampled elapsed-time conversion requires a valid in-run power-trace AICLK"
    )


def ticks_to_seconds(
    *,
    ticks: int,
    selection: Mapping[str, Any] | None = None,
    aiclk_mhz: int | None = None,
) -> float:
    """Convert device-clock ticks using one selected AICLK provenance."""
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
        raise ValueError("ticks must be a non-negative integer")
    if selection is None:
        if aiclk_mhz is None:
            raise ValueError("an AICLK selection is required")
        selected = safe_aiclk_value(aiclk_mhz)
    else:
        selected = safe_aiclk_value(
            selection.get("aiclk_mhz_exact", selection.get("aiclk_mhz"))
        )
    if selected is None:
        raise ValueError("selected AICLK must be a finite positive number")
    return ticks / (selected * 1_000_000)


__all__ = [
    "AICLK_SOURCE_CONFIGURED",
    "AICLK_SOURCE_LEGACY_UNVERIFIED",
    "AICLK_SOURCE_RUN_TRACE_SAMPLES",
    "safe_aiclk_integer",
    "safe_aiclk_value",
    "select_elapsed_aiclk",
    "select_load_interval_aiclk",
    "ticks_to_seconds",
]
