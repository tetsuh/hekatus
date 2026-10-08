"""Immutable clock-source audit data shared by resident record producers."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

AUDITED_CLOCK_SOURCE_IMAGE = (
    "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@"
    "sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"
)
CLOCK_SOURCE_UNAUDITED_DIAGNOSTIC = "clock source unaudited for this image"

# The complete digest is the lookup key.  A release tag or repository basename
# is deliberately insufficient to select an audit entry.
CLOCK_SOURCE_AUDIT_TABLE: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        AUDITED_CLOCK_SOURCE_IMAGE: MappingProxyType(
            {
                "toolchain": "tt-metal v0.75.0",
                "tt_metal_revision": "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd",
                "clock_api": "tt_metal/hw/inc/internal/tt-1xx/risc_common.h:254-255",
                "blackhole_read_api": "tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h:503-510",
                "dataflow_ring_api": "tt_metal/hw/inc/api/dataflow/dataflow_api.h:404-485",
                "semaphore_api": "tt_metal/hw/inc/api/dataflow/dataflow_api.h:1514-1525,1934-1992",
                "frequency_api": "tt_metal/api/tt-metalium/device.hpp:86-89",
                "profiler_conversion": "tt_metal/impl/profiler/profiler_analysis.cpp:300-303",
                "cross_core_note": (
                    "intervals use only the designated consumer core; cross-core "
                    "correlation is out of scope"
                ),
            }
        )
    }
)


def clock_source_audit_for_image(image: Any) -> Mapping[str, str] | None:
    """Return the audit entry for one exact digest-pinned image reference."""
    if not isinstance(image, str):
        return None
    return CLOCK_SOURCE_AUDIT_TABLE.get(image)


def source_evidence_for_image(image: Any) -> dict[str, Any]:
    """Build evidence whose audited fields derive only from the digest table."""
    audited = clock_source_audit_for_image(image)
    if audited is None:
        return {
            **{field: None for field in next(iter(CLOCK_SOURCE_AUDIT_TABLE.values()), {})},
            "image": image,
            "audit_status": "unaudited",
            "diagnostic": CLOCK_SOURCE_UNAUDITED_DIAGNOSTIC,
        }
    return {
        **audited,
        "image": image,
        "audit_status": "audited",
    }


__all__ = [
    "AUDITED_CLOCK_SOURCE_IMAGE",
    "CLOCK_SOURCE_AUDIT_TABLE",
    "CLOCK_SOURCE_UNAUDITED_DIAGNOSTIC",
    "clock_source_audit_for_image",
    "source_evidence_for_image",
]
