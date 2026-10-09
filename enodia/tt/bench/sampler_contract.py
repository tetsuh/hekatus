"""Shared telemetry-sampler mode and interval contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

SAMPLER_CONTRACT: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "off": MappingProxyType(
            {
                "interval_rule": "absent",
                "interval_seconds": None,
                "power_trace": "absent_by_design",
                "timing_evidence": "diagnostic_only",
            }
        ),
        "default": MappingProxyType(
            {
                "interval_rule": "exact",
                "interval_seconds": 2.0,
                "power_trace": "required",
                "timing_evidence": "available",
            }
        ),
        "explicit": MappingProxyType(
            {
                "interval_rule": "positive_finite",
                "interval_seconds": None,
                "power_trace": "required",
                "timing_evidence": "available",
            }
        ),
    }
)
SAMPLER_MODES = tuple(SAMPLER_CONTRACT)
DEFAULT_SAMPLER_INTERVAL_SECONDS = float(SAMPLER_CONTRACT["default"]["interval_seconds"])
MIN_EXPLICIT_SAMPLER_INTERVAL_SECONDS = 0.1
MAX_EXPLICIT_SAMPLER_INTERVAL_SECONDS = 3_600.0

_MISSING = object()


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def normalize_sampler_metadata(
    raw_sampler: Mapping[str, Any] | None = None, *, complete_default: bool = True
) -> dict[str, Any]:
    """Return canonical sampler metadata after enforcing the shared contract.

    A missing sampler object is the legacy default and is completed to the
    existing two-second sampler.  A default object may likewise omit its
    interval only when ``complete_default`` is true; record validation passes
    false so a persisted record must carry the explicit contract fields.
    """
    if raw_sampler is None:
        raw: dict[str, Any] = {"mode": "default"}
    elif not isinstance(raw_sampler, Mapping):
        raise TypeError("telemetry sampler metadata must be an object")
    else:
        raw = dict(raw_sampler)

    mode = raw.get("mode", _MISSING)
    if not isinstance(mode, str) or mode not in SAMPLER_CONTRACT:
        raise ValueError("telemetry sampler mode must be off, default, or explicit")

    interval = raw.get("interval_seconds", _MISSING)
    contract = SAMPLER_CONTRACT[mode]
    interval_rule = contract["interval_rule"]
    if interval_rule == "absent":
        if interval is not _MISSING and interval is not None:
            raise ValueError("sampler-off metadata must not carry an interval")
        normalized_interval = None
    elif interval_rule == "exact":
        if interval is _MISSING or interval is None:
            if not complete_default:
                raise ValueError("default sampler mode must carry a 2.0-second interval")
            normalized_interval = DEFAULT_SAMPLER_INTERVAL_SECONDS
        elif not _is_finite_number(interval):
            raise ValueError("default sampler interval must be a finite number")
        elif float(interval) != DEFAULT_SAMPLER_INTERVAL_SECONDS:
            raise ValueError("default sampler mode must use a 2.0-second interval")
        else:
            normalized_interval = DEFAULT_SAMPLER_INTERVAL_SECONDS
    else:
        if interval is _MISSING or interval is None:
            raise ValueError("explicit sampler mode must carry a positive finite interval")
        if not _is_finite_number(interval):
            raise ValueError("explicit sampler interval must be finite")
        normalized_interval = float(interval)
        if not (
            MIN_EXPLICIT_SAMPLER_INTERVAL_SECONDS
            <= normalized_interval
            <= MAX_EXPLICIT_SAMPLER_INTERVAL_SECONDS
        ):
            raise ValueError(
                "explicit sampler interval must be between "
                f"{MIN_EXPLICIT_SAMPLER_INTERVAL_SECONDS} and "
                f"{MAX_EXPLICIT_SAMPLER_INTERVAL_SECONDS} seconds"
            )

    normalized = dict(raw)
    normalized["mode"] = mode
    normalized["interval_seconds"] = normalized_interval
    normalized.setdefault("power_trace", contract["power_trace"])
    normalized.setdefault("timing_evidence", contract["timing_evidence"])
    return normalized


__all__ = [
    "DEFAULT_SAMPLER_INTERVAL_SECONDS",
    "MAX_EXPLICIT_SAMPLER_INTERVAL_SECONDS",
    "MIN_EXPLICIT_SAMPLER_INTERVAL_SECONDS",
    "SAMPLER_CONTRACT",
    "SAMPLER_MODES",
    "normalize_sampler_metadata",
]
