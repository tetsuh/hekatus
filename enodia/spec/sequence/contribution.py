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

import math
from dataclasses import dataclass, field

import numpy as np

from enodia.spec.sequence import TransmitConfig

# The MLA counts a map can be derived for: {2, 4} are the fixed
# specification and 8 is the experiment slot for color flow (§7); 1 is the
# conventional case the others degenerate from.
MLA_COUNTS: tuple[int, ...] = (1, 2, 4, 8)

# Below this per-line weight sum, a line is refused rather than normalized:
# dividing by a vanishing sum multiplies noise into a plausible-looking
# line, and wrong output is worse than no output (absolute rules).
WEIGHT_SUM_FLOOR: float = 1e-6


def _validated_line_x(line_x_m) -> tuple[float, ...]:
    """The line abscissae, coerced and finite.

    A non-finite abscissa does not produce a non-finite image: the read
    position casts to a garbage integer index and the line comes out
    silently black, which the absolute rules single out as worse than no
    output at all. Written as "not all finite" rather than a comparison,
    since every comparison against NaN is false.
    """
    line_x = tuple(float(x) for x in line_x_m)
    if not line_x:
        raise ValueError("a contribution map must form at least one line")
    if not all(math.isfinite(x) for x in line_x):
        worst = next(i for i, x in enumerate(line_x) if not math.isfinite(x))
        raise ValueError(f"line {worst} has a non-finite abscissa {line_x[worst]}")
    return line_x


def _check_provenance(cmap: ContributionMap) -> None:
    """The identity a consumer compares against the frame, and its counters.

    Required, not optional: while these could be empty, `check_frame` had
    nothing to compare and every unbound map — including one supplied
    explicitly — was consumed unchecked.
    """
    if isinstance(cmap.param_generation, bool) or not isinstance(cmap.param_generation, int):
        raise TypeError(f"parameter generation must be an integer, got {cmap.param_generation!r}")
    if cmap.param_generation < 0:
        raise ValueError(f"parameter generation must not be negative, got {cmap.param_generation}")
    if isinstance(cmap.n_events, bool) or not isinstance(cmap.n_events, int):
        raise TypeError(f"event count must be an integer, got {cmap.n_events!r}")
    if not cmap.config_id or not cmap.probe_profile_id or cmap.n_events <= 0:
        raise ValueError(
            "a contribution map must name the configuration it was derived"
            f" from; got config_id={cmap.config_id!r},"
            f" probe_profile_id={cmap.probe_profile_id!r}, n_events={cmap.n_events}"
        )


def _check_type_conditions(line_tx_type, n_lines: int) -> None:
    """§7's transmit-type matching condition, one per line.

    Coercing with `str()` would turn 123 into "123": a tag that looks valid,
    matches no record, and fails at consumption instead of here. An empty
    condition matches everything, which is the state this field exists to
    remove.
    """
    if any(not isinstance(t, str) for t in line_tx_type):
        raise TypeError("transmit-type conditions must be strings")
    if len(line_tx_type) != n_lines:
        raise ValueError(
            f"map has {n_lines} lines but {len(line_tx_type)} transmit-type conditions"
        )
    if any(not t for t in line_tx_type):
        raise ValueError("every line must name the transmit type it is formed from")


def _validated_routes(
    event_indices, weights, n_lines: int, n_events: int, config_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """The routes and their weights, checked and normalized to unit row sums.

    ADR-0010 decision 3 puts the normalization here, at derivation, so every
    consumer of a map sees the same one.
    """
    indices = np.ascontiguousarray(event_indices)
    weight_array = np.ascontiguousarray(weights, dtype=np.float64)
    if indices.shape != weight_array.shape or indices.ndim != 2 or indices.shape[0] != n_lines:
        raise ValueError(
            f"map shape mismatch: {n_lines} lines,"
            f" indices {indices.shape}, weights {weight_array.shape}"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        # A float index silently truncates at the read, so line k would draw
        # event floor(k) and no error would ever be raised.
        raise ValueError(f"event indices must be an integer dtype, got {indices.dtype}")
    if indices.size and int(indices.min()) < 0:
        raise ValueError(f"event index {int(indices.min())} is negative")
    if n_events and indices.size and int(indices.max()) >= n_events:
        raise ValueError(
            f"event index {int(indices.max())} is past the"
            f" {n_events} events of configuration {config_id!r}"
        )
    if np.any(weight_array < 0.0) or not np.all(np.isfinite(weight_array)):
        raise ValueError("contribution weights must be finite and non-negative")
    with np.errstate(over="ignore"):
        sums = weight_array.sum(axis=1)
    if not np.all(np.isfinite(sums)):
        worst = int(np.flatnonzero(~np.isfinite(sums))[0])
        raise ValueError(
            f"line {worst} has a non-finite weight sum; refused rather than normalized"
        )
    if np.any(sums < WEIGHT_SUM_FLOOR):
        worst = int(np.argmin(sums))
        raise ValueError(
            f"line {worst} has weight sum {sums[worst]:.3e}, below the"
            f" {WEIGHT_SUM_FLOOR:g} floor; refused rather than amplified"
        )
    return indices, weight_array / sums[:, None]


@dataclass(frozen=True)
class ContributionMap:
    """Event → line, with weights, at fixed work per line.

    `event_indices[l, s]` names the transmit event slot `s` of line `l`
    reads; `weights[l, s]` is its weight, 0.0 for an inert slot. **Every
    line's weights sum to one**, with no way to construct one that does
    not — and, since the validation publishes through an overridable
    method, **no way to subclass one that does not** either: ADR-0010
    decision 3 puts the normalization at derivation precisely so every
    consumer of a map sees the same normalization, and an escape hatch on
    the accepted type — a constructor flag or an inherited seam — would be a
    second normalization the decision says does not exist. A test that needs the un-normalized behaviour as a
    negative control builds its own object rather than asking this one to
    be less than it is.

    **A map names the configuration it was derived from — always.**
    `config_id`, `n_events` and `param_generation` are required fields, not
    optional ones, and a consumer checks them against the frame it is about
    to form
    (`enodia.spec.beamform`). Without that check, a map derived for one
    configuration renders another configuration's records — the event
    indices line up, the record identity checks pass, and the line geometry
    is silently stale. That is the accident the generation tag exists to
    prevent: processing with the wrong tables is worse than dropping frames
    (absolute rules, §19).

    The generation is carried separately from the id because a depth or
    focus change is an **in-config** change (§19): the configuration id
    holds still while every derivative behind it — delay tables, and this
    map — is invalidated and re-derived. A map checked on the id alone would
    survive exactly the change that invalidates it.

    **`line_tx_type` is the transmit-type matching condition §7 requires**:
    the one kind of transmit each line is formed from. A line is refused at
    derivation if its live slots span kinds, so a B-mode and a colour-flow
    transmit of an interleaved sequence can never be summed into one pixel.
    The tag is an open set of strings (§11.5), so the condition is equality
    between whatever strings the sequence uses, not membership of an enum
    this module would have to be edited to extend. It is a **required**
    field with no empty entries: a condition that may be absent is a
    condition a hand-built map can omit, and a consumer that skips absent
    conditions is back where this field started. Both consumers check every
    live slot against it, so the condition is enforced where the sum happens
    and not only where the map is derived.

    Line abscissae are coerced and checked for finiteness before anything
    else reads them, for the same reason the schema checks its own
    quantities at ingress (`enodia.spec.sequence`): a non-finite coordinate
    reaches the delay and aperture arithmetic and comes back as a silently
    black line, not as an error.

    **The map owns its data at construction**, the same way `IQEventRecord`
    does at its publication boundary (#6): the arrays are copied into
    immutable `bytes` and exposed as `np.frombuffer` views over them, and
    `line_x_m` is coerced to a tuple of floats. Clearing the writeable flag
    on a caller's array is not enough — `np.ascontiguousarray` returns the
    caller's own array when it is already contiguous, and NumPy lets an
    array that owns its memory have WRITEABLE set back to True. A map is a
    derivative of one configuration generation; a caller that could still
    reach its routes or its normalized weights afterwards could make two
    consumers of the same map form different images from the same frame
    (§19).
    """

    line_x_m: tuple[float, ...]
    event_indices: np.ndarray  # (n_lines, cap) int
    weights: np.ndarray  # (n_lines, cap) float64
    config_id: str
    probe_profile_id: str
    n_events: int
    param_generation: int
    line_tx_type: tuple[str, ...]
    _index_owner: bytes = field(init=False, repr=False, compare=False)
    _weight_owner: bytes = field(init=False, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs) -> None:
        """The type is final, because the type is what carries the guarantee.

        `__post_init__` dispatches through `self._publish`, so a subclass
        that overrides it runs the whole validation and then publishes
        something else: the rows a consumer reads need never be the rows
        `_validated_routes` normalized. An `isinstance` check at a consumer
        then confirms the class and not the derivation (`SOL-57-001`).

        Sealing the class is the correction at the seam rather than at the
        one call site that happened to notice it — an exact-type check
        wherever a map is consumed leaves the seam in place for the next
        consumer written without one. The ingress checks the exact type as
        well, because a spec-spoofing test double satisfies `isinstance`
        without being a subclass at all.
        """
        raise TypeError(
            "ContributionMap is final: a subclass can override _publish and "
            "expose weights that _validated_routes never normalized"
        )

    def __post_init__(self) -> None:
        # One validator per group, so the field-against-invariant matrix is
        # visible rather than implied by the order of forty lines.
        line_x = _validated_line_x(self.line_x_m)
        object.__setattr__(self, "line_x_m", line_x)
        _check_provenance(self)
        _check_type_conditions(self.line_tx_type, len(line_x))
        indices, weights = _validated_routes(
            self.event_indices, self.weights, len(line_x), self.n_events, self.config_id
        )
        self._publish(indices, weights)

    def _publish(self, indices: np.ndarray, weights: np.ndarray) -> None:
        """Copy into immutable bytes and expose read-only views over them.

        The base chain of each exposed array ends at a `bytes` object, which
        no flag can make writable.
        """
        shape = indices.shape
        index_owner = np.ascontiguousarray(indices, dtype=np.intp).tobytes()
        weight_owner = np.ascontiguousarray(weights, dtype=np.float64).tobytes()
        object.__setattr__(self, "_index_owner", index_owner)
        object.__setattr__(self, "_weight_owner", weight_owner)
        object.__setattr__(
            self, "event_indices", np.frombuffer(index_owner, dtype=np.intp).reshape(shape)
        )
        object.__setattr__(
            self, "weights", np.frombuffer(weight_owner, dtype=np.float64).reshape(shape)
        )
        object.__setattr__(self, "line_tx_type", tuple(self.line_tx_type))

    @property
    def n_lines(self) -> int:
        """Output lines this map forms — not the transmit-event count."""
        return len(self.line_x_m)

    @property
    def cap(self) -> int:
        """Contributing-transmit slots per line — the fixed work."""
        return int(self.event_indices.shape[1])

    def check_profile(self, profile_name: str) -> None:
        """Refuse a probe this map's configuration was not accepted against.

        A configuration names exactly one probe profile (§19, #46), and the
        producing side already refuses a mismatch (`simulate_frame`). The
        consumers take the profile as a separate argument and read the whole
        of the geometry from it — element positions, sound speed, sampling
        rate, F-number — so a mismatched profile beamforms a real
        acquisition on another probe's delays and returns a plausible image.
        """
        if self.probe_profile_id != profile_name:
            raise ValueError(
                f"contribution map was derived for probe profile"
                f" {self.probe_profile_id!r}, beamforming on {profile_name!r}"
            )

    def check_frame(self, config_id: str, n_events: int, param_generation: int) -> None:
        """Refuse a frame this map was not derived for.

        The map carries line geometry; the records carry the configuration
        id and its parameter generation (`EventHeader`). When they disagree,
        the events still resolve — indices are small integers and every
        configuration has them — so nothing downstream would notice, and the
        frame would be formed on stale scanlines.
        """
        if self.config_id != config_id:
            raise ValueError(
                f"contribution map was derived for configuration"
                f" {self.config_id!r}, frame carries {config_id!r}"
            )
        if self.n_events != n_events:
            raise ValueError(
                f"contribution map was derived for {self.n_events} transmit"
                f" events, frame carries {n_events}"
            )
        if self.param_generation != param_generation:
            raise ValueError(
                f"contribution map was derived at parameter generation"
                f" {self.param_generation}, frame carries {param_generation}"
            )


def _line_tx_types(
    config: TransmitConfig, indices: np.ndarray, weights: np.ndarray
) -> tuple[str, ...]:
    """The one transmit kind each line is formed from — §7's matching condition.

    Refuses a line whose live slots span kinds. §7 requires it so that a
    B-mode and a colour-flow transmit of an interleaved sequence are never
    summed into one pixel: they carry different pulses and different slow-time
    meaning, and the sum is a plausible-looking number with no physical
    reading. Inert slots (weight zero) are ignored — they contribute nothing,
    and a padded frame-edge row must not be refused for the kind of a
    transmit it does not use.
    """
    types: list[str] = []
    for line in range(indices.shape[0]):
        live = {
            config.events[int(indices[line, slot])].tx_type
            for slot in range(indices.shape[1])
            if weights[line, slot] > 0.0
        }
        if len(live) > 1:
            raise ValueError(
                f"line {line} would sum transmit kinds {sorted(live)};"
                " the contribution map matches transmit type (design.md §7)"
            )
        types.append(live.pop() if live else "")
    return tuple(types)


def identity_map(config: TransmitConfig) -> ContributionMap:
    """Event k forms line k with unit weight — the conventional case.

    This is the map every pre-#53 image was formed under, and the general
    beamformer under this map must reproduce those images to the last bit:
    the identity is a case of the structure, not a preserved second path.
    """
    n = len(config.events)
    indices = np.arange(n, dtype=np.intp)[:, None]
    weights = np.ones((n, 1), dtype=np.float64)
    return ContributionMap(
        line_x_m=tuple(ev.line_x_m for ev in config.events),
        event_indices=indices,
        weights=weights,
        config_id=config.config_id,
        probe_profile_id=config.probe_profile_id,
        n_events=n,
        param_generation=config.param_generation,
        line_tx_type=_line_tx_types(config, indices, weights),
    )


def _uniform_pitch_m(values: np.ndarray, what: str, config_id: str) -> float:
    """The single spacing of a uniform 1D grid [m], or a refusal.

    From the span over the interval count rather than one neighbour
    difference: consecutive differences of a centred float64 grid vary in
    their last bits, and the span recovers the pitch the grid was built
    from exactly.
    """
    if values.size < 2:
        raise ValueError(f"configuration {config_id!r} has too few {what} for a pitch")
    pitch = float((values[-1] - values[0]) / (values.size - 1))
    spacing = np.diff(values)
    if not np.allclose(spacing, pitch, rtol=0.0, atol=abs(pitch) * 1e-9):
        raise ValueError(
            f"configuration {config_id!r} has a non-uniform {what} grid;"
            " MLA line geometry is defined for a uniform pitch"
        )
    return pitch


def element_pitch_m(config: TransmitConfig) -> float:
    """The element pitch of an accepted configuration's own geometry [m].

    Taken from `config.element_x_m` rather than from a `ProbeProfile`
    handed in beside it. A caller can hold a valid configuration and a
    different valid profile, and a map built from the second one's pitch
    puts every line in the wrong place without raising: the shapes match,
    the indices resolve, and only the image is wrong. Deriving from the
    configuration removes the mismatch instead of detecting it.

    This is the **element** pitch. It is not what MLA subdivides — see
    `transmit_line_pitch_m` — and the two coincide only for a sequence that
    puts one transmit above each element.
    """
    return _uniform_pitch_m(
        np.asarray(config.element_x_m, dtype=np.float64), "element", config.config_id
    )


def transmit_line_pitch_m(config: TransmitConfig) -> float:
    """The spacing between the transmit beam axes of a configuration [m].

    This is what §7's MLA subdivision is defined against: the receive lines
    of one transmit fill the gap to the next *transmit*, not the gap to the
    next element. The two coincide for the conventional sequence — one
    scanline above each element — and part as soon as a sequence walks the
    aperture at some other stride or fires fewer events than there are
    elements. Deriving from `element_x_m` made MLA groups too narrow or too
    wide in exactly those cases, and MLA 1 could not show it because its
    offset is zero either way.

    A non-uniform line grid is refused rather than averaged: a sequence with
    varying line spacing has no single pitch to subdivide, and what MLA
    should do there is not defined (a convex or sector sequence is angular,
    which arrives with those probes).
    """
    return _uniform_pitch_m(
        np.array([ev.line_x_m for ev in config.events], dtype=np.float64),
        "transmit line",
        config.config_id,
    )


def mla_map(config: TransmitConfig, *, mla: int) -> ContributionMap:
    """MLA as a property of the map: `mla` receive lines per transmit.

    The lines of transmit k sit at

        x_k + pitch · (2j − (mla − 1)) / (2 · mla),   j = 0 .. mla−1

    — the **transmit line** pitch subdivided evenly, symmetric about the
    transmit axis, read from the configuration's own beam axes rather than
    from its elements: the receive lines of one transmit fill the gap to the
    next transmit. At mla=1 the offset is exactly zero, so the conventional
    geometry is recovered with no epsilon. Pure MLA keeps one contributing
    transmit per line (cap 1); several transmits per line is transmit
    compounding, whose weights are measured quantities (§17) and are not
    chosen here.
    """
    if mla not in MLA_COUNTS:
        raise ValueError(f"MLA count must be one of {MLA_COUNTS}, got {mla}")
    pitch = transmit_line_pitch_m(config) if mla > 1 else 0.0
    line_x: list[float] = []
    indices = np.empty((mla * len(config.events), 1), dtype=np.intp)
    for k, ev in enumerate(config.events):
        for j in range(mla):
            offset = pitch * (2 * j - (mla - 1)) / (2 * mla)
            line_x.append(ev.line_x_m + offset)
            indices[k * mla + j, 0] = k
    weights = np.ones((mla * len(config.events), 1), dtype=np.float64)
    return ContributionMap(
        line_x_m=tuple(line_x),
        event_indices=indices,
        weights=weights,
        config_id=config.config_id,
        probe_profile_id=config.probe_profile_id,
        n_events=len(config.events),
        param_generation=config.param_generation,
        line_tx_type=_line_tx_types(config, indices, weights),
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
    if cap % 2 == 0:
        # This map is centred on line k. An even cap has no centre slot, so
        # the contributions straddle the line and non-uniform data shifts
        # the result laterally by half a line — a fixture that biases what
        # it is meant to isolate. A centred even cap needs a half-line
        # output geometry, which is not defined here.
        raise ValueError(f"cap must be odd for a line-centred map, got {cap}")
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
        config_id=config.config_id,
        probe_profile_id=config.probe_profile_id,
        n_events=n,
        param_generation=config.param_generation,
        line_tx_type=_line_tx_types(config, indices, weights),
    )
