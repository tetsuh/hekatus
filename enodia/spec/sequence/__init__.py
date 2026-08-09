"""Transmit-sequence description, in its smallest useful form.

The full physical-quantity schema of design.md §19 — element coordinates,
firing delays, apodization, and the contribution map that generalizes MLA and
transmit compounding — arrives with #7. What is already true here is the part
that is expensive to retrofit: transmit descriptions are stated as physical
quantities, and the transmit-type tag is an open set of strings rather than
an enum, because shear-wave push and tracking transmits join it later
(design.md §11.5).

This module covers the conventional case: one transmit event forms one
scanline, so the contribution map is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from enodia.spec.probe import ProbeProfile


@dataclass(frozen=True)
class TxEvent:
    """One transmit event, described in physical quantities [m]."""

    event_index: int
    line_index: int  # identity contribution map: event k forms line k
    tx_type: str
    line_x_m: float  # lateral position of the scanline (the beam axis)
    virtual_source_m: tuple[float, float]  # virtual source (x, z)


def make_bmode_sequence(profile: ProbeProfile) -> list[TxEvent]:
    """Conventional focused-beam B-mode: one scanline above each element."""
    return [
        TxEvent(
            event_index=k,
            line_index=k,
            tx_type="bmode_focused",
            line_x_m=float(x),
            virtual_source_m=(float(x), profile.tx_focus_m),
        )
        for k, x in enumerate(profile.element_x())
    ]
