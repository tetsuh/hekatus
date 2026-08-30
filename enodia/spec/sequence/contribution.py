"""The contribution map: which transmit events form which lines (§7, #53).

design.md §7 names one structure under both MLA and transmit compounding —
a weighted sparse mapping from transmit event to output line — because the
two are two uses of the same fact: the transmit beam has finite width. §19
fixes its provenance: maps are **derivatives**, computed by enodia from the
transmit description, never received. The external description accordingly
says nothing about lines (`enodia.spec.sequence`), and everything here is
derived from an accepted `TransmitConfig`.

**Work per line is constant, by construction.** Every line carries exactly
`cap` slots. A frame-edge line that has fewer real contributions than the
cap does not get a shorter row: it gets **inert slots** — weight zero,
pointing at a real event — so the beamformer runs the same reads and
multiply-adds on every line. The absolute rules forbid data-dependent work
on the real-time path, and the natural implementation (iterate only the
transmits that contribute) passes every image check while breaking
worst-case latency. The map's shape is where that rule is enforceable.

**Renormalization is a property of derivation.** Each line's weights sum to
one, frame edges included, with a floor below which a line is refused
rather than amplified. The beamformer stays a plain weighted sum, and edge
lines — which receive fewer contributing transmits — are not darker for it.
The alternative, dividing by the count of contributing transmits, is wrong
as soon as the weights stop being uniform (#53's second recorded decision).

**MLA line placement** (#53's first recorded decision): the receive lines
of one transmit subdivide the transmit line pitch evenly and sit
symmetrically about the transmit axis, so MLA 1 degenerates to the
conventional geometry exactly — no epsilon — and the placement translates
with the transmit, which is what keeps delay-table translation invariance
on convex and sector probes (polar coordinates, absolute rules). The
alternative, an output line pitch decoupled from the transmit pitch, is
more general and gives that invariance up.

What deliberately is not here: the transmit-compounding weight function —
how the transmit-beam amplitude profile enters the weights needs the beam
model, which is #9 — and the compounding parameters themselves (window
width, apodization, truncation count), which §17 keeps as measured
quantities. `synthetic_uniform_map` exercises the multi-contribution
structure with uniform weights and a small fixed cap; it selects no
production value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enodia.spec.probe import ProbeProfile
from enodia.spec.sequence import TransmitConfig

# The MLA counts a map can be derived for: {2, 4} are the fixed
# specification and 8 is the experiment slot for color flow (§7); 1 is the
# conventional case the others degenerate from.
MLA_COUNTS: tuple[int, ...] = (1, 2, 4, 8)

# Below this per-line weight sum, a line is refused rather than normalized:
# dividing by a vanishing sum multiplies noise into a plausible-looking
# line, and wrong output is worse than no output (absolute rules).
WEIGHT_SUM_FLOOR: float = 1e-6


@dataclass(frozen=True)
class ContributionMap:
    """Event → line, with weights, at fixed work per line.

    `event_indices[l, s]` names the transmit event slot `s` of line `l`
    reads; `weights[l, s]` is its weight, 0.0 for an inert slot. Every
    line's weights sum to one unless the map was explicitly constructed
    `normalized=False` — which exists so a test can show the artifact
    renormalization prevents, and for no other reason.

    Both arrays are frozen at construction: a map is a derivative of one
    configuration generation, and a consumer mutating it would desynchronize
    every other consumer of the same generation (§19).
    """

    line_x_m: tuple[float, ...]
    event_indices: np.ndarray  # (n_lines, cap) int
    weights: np.ndarray  # (n_lines, cap) float64
    normalized: bool = True

    def __post_init__(self) -> None:
        indices = np.ascontiguousarray(self.event_indices)
        weights = np.ascontiguousarray(self.weights, dtype=np.float64)
        n_lines = len(self.line_x_m)
        if indices.shape != weights.shape or indices.ndim != 2 or indices.shape[0] != n_lines:
            raise ValueError(
                f"map shape mismatch: {len(self.line_x_m)} lines,"
                f" indices {indices.shape}, weights {weights.shape}"
            )
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("contribution weights must be finite and non-negative")
        sums = weights.sum(axis=1)
        if np.any(sums < WEIGHT_SUM_FLOOR):
            worst = int(np.argmin(sums))
            raise ValueError(
                f"line {worst} has weight sum {sums[worst]:.3e}, below the"
                f" {WEIGHT_SUM_FLOOR:g} floor; refused rather than amplified"
            )
        if self.normalized:
            weights = weights / sums[:, None]
        indices.flags.writeable = False
        weights.flags.writeable = False
        object.__setattr__(self, "event_indices", indices)
        object.__setattr__(self, "weights", weights)

    @property
    def n_lines(self) -> int:
        return len(self.line_x_m)

    @property
    def cap(self) -> int:
        """Contributing-transmit slots per line — the fixed work."""
        return int(self.event_indices.shape[1])


def identity_map(config: TransmitConfig) -> ContributionMap:
    """Event k forms line k with unit weight — the conventional case.

    This is the map every pre-#53 image was formed under, and the general
    beamformer under this map must reproduce those images to the last bit:
    the identity is a case of the structure, not a preserved second path.
    """
    n = len(config.events)
    return ContributionMap(
        line_x_m=tuple(ev.line_x_m for ev in config.events),
        event_indices=np.arange(n, dtype=np.intp)[:, None],
        weights=np.ones((n, 1), dtype=np.float64),
    )


def mla_map(config: TransmitConfig, profile: ProbeProfile, *, mla: int) -> ContributionMap:
    """MLA as a property of the map: `mla` receive lines per transmit.

    The lines of transmit k sit at

        x_k + pitch · (2j − (mla − 1)) / (2 · mla),   j = 0 .. mla−1

    — the transmit pitch subdivided evenly, symmetric about the transmit
    axis. At mla=1 the offset is exactly zero, so the conventional geometry
    is recovered with no epsilon. Pure MLA keeps one contributing transmit
    per line (cap 1); several transmits per line is transmit compounding,
    whose weights are measured quantities (§17) and are not chosen here.
    """
    if mla not in MLA_COUNTS:
        raise ValueError(f"MLA count must be one of {MLA_COUNTS}, got {mla}")
    line_x: list[float] = []
    indices = np.empty((mla * len(config.events), 1), dtype=np.intp)
    for k, ev in enumerate(config.events):
        for j in range(mla):
            offset = profile.pitch_m * (2 * j - (mla - 1)) / (2 * mla)
            line_x.append(ev.line_x_m + offset)
            indices[k * mla + j, 0] = k
    return ContributionMap(
        line_x_m=tuple(line_x),
        event_indices=indices,
        weights=np.ones((mla * len(config.events), 1), dtype=np.float64),
    )


def synthetic_uniform_map(config: TransmitConfig, *, cap: int) -> ContributionMap:
    """A multi-contribution map with uniform weights — a fixture, not a spec.

    Line k draws from the `cap` events centred on k, clipped at the frame
    edges; live slots carry equal weight and the row is renormalized, so
    edge lines are not darker for having fewer real contributions. Edge
    rows are padded to the cap with inert slots (weight zero, pointing at
    the line's own event), keeping work per line constant.

    It exists to exercise the summation structure and the edge
    renormalization — pure MLA leaves one contribution per line, which
    would leave both untested. It selects no production compounding window,
    weight function, or truncation count; those are §17's measured
    quantities, gated on the beam model (#9).
    """
    n = len(config.events)
    if not (1 <= cap <= n):
        raise ValueError(f"cap must be within [1, {n}], got {cap}")
    half = (cap - 1) // 2
    indices = np.empty((n, cap), dtype=np.intp)
    weights = np.zeros((n, cap), dtype=np.float64)
    for k in range(n):
        for s in range(cap):
            e = k - half + s
            if 0 <= e < n:
                indices[k, s] = e
                weights[k, s] = 1.0 / cap
            else:
                indices[k, s] = k  # inert slot: same read, zero weight
                weights[k, s] = 0.0
    return ContributionMap(
        line_x_m=tuple(ev.line_x_m for ev in config.events),
        event_indices=indices,
        weights=weights,
    )
