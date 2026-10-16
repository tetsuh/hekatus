"""Canonical board identity helpers shared by telemetry selectors."""

from __future__ import annotations

import os
import re

_PCI_BUS_ID_RE = re.compile(
    r"(?<![0-9a-f])([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])(?![0-9a-f])",
    re.IGNORECASE,
)
_BOARD_ID_ENV_VARS = ("HEKATUS_TT_BOARD_ID", "HEKATUS_TT_TARGET_BOARD_ID")
_BUS_ID_ENV_VARS = (
    "HEKATUS_TT_BUS_ID",
    "HEKATUS_TT_BOARD_BUS_ID",
    "HEKATUS_TT_TARGET_BUS_ID",
)


def normalize_board_id(value: object) -> str | None:
    """Return one comparable board identity, or ``None`` when it is unusable."""
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def normalize_pci_bus_id(value: object) -> str | None:
    """Return a lower-case, trimmed PCI BDF, or ``None`` when malformed."""
    if not isinstance(value, str):
        return None
    match = _PCI_BUS_ID_RE.fullmatch(value.strip())
    return match.group(1).lower() if match else None


def _environment_value(names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in os.environ:
            return os.environ[name]
    return None


def resolve_board_identity(
    board_id: object | None = None, bus_id: object | None = None
) -> tuple[str | None, str | None]:
    """Resolve and normalize one explicit-or-environment identity pair.

    An entirely absent pair keeps the legacy caller's default-node behavior.
    Once either side is supplied, both sides must be valid; callers must not
    silently fall back to a different board after a malformed request.
    """
    requested_board_id = (
        board_id if board_id is not None else _environment_value(_BOARD_ID_ENV_VARS)
    )
    requested_bus_id = (
        bus_id if bus_id is not None else _environment_value(_BUS_ID_ENV_VARS)
    )
    if requested_board_id is None and requested_bus_id is None:
        return None, None

    normalized_board_id = normalize_board_id(requested_board_id)
    if normalized_board_id is None:
        raise ValueError("board_id must be a non-empty string")
    normalized_bus_id = normalize_pci_bus_id(requested_bus_id)
    if normalized_bus_id is None:
        raise ValueError("bus_id must be a valid PCI BDF")
    return normalized_board_id, normalized_bus_id
