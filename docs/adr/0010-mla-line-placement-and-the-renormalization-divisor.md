# ADR-0010: MLA line placement, and what frame-edge renormalization divides by

- Status: **Accepted** — 2026-08-31
- Date: 2026-08-30
- Owner decision: yes (`DEC-57-001`)

## Context

`docs/design.md` §7 adopts transmit compounding and fixes the MLA
specification at {2, 4}, with 8 as the colour experiment slot. It names the
structure both run through — a weighted sparse mapping from transmit event
to output line — and says what it costs and why it is worth it. It does not
say **where the MLA receive lines sit**, and it does not say **what the
frame-edge renormalization divides by**.

Both were open when #53 began, and neither is a parameter a sweep settles:
they are geometry and arithmetic that every later measurement is stated
against. §17 owns the compounding *parameters* — window width, apodization,
truncation count — and deliberately does not own these two.

They are recorded here rather than left in the implementation because they
alter a contract. Line placement decides the output grid an L0 comparison
runs on: two implementations that place MLA lines differently disagree
everywhere while both being internally consistent, and no tolerance
distinguishes that from a numerical fault. The renormalization divisor
decides what a pixel value *means* under a non-uniform weighting, which is
the value every gCNR and PSF figure is computed from.

## Options considered

**Where the MLA receive lines sit**

- **A. Subdivide the transmit line pitch evenly, symmetric about the
  transmit axis (chosen).** The lines of transmit k sit at
  `x_k + pitch·(2j − (mla − 1)) / (2·mla)`, `j = 0 … mla−1`, where `pitch`
  is the spacing of the transmit beam axes. MLA 1 gives offset exactly
  zero, so the conventional geometry is recovered with no epsilon and no
  separate code path. The group translates with its transmit, which is what
  keeps delay-table translation invariance when the sequence is angular
  rather than linear (convex and sector probes, polar coordinates — an
  absolute rule).
- **B. An output line pitch decoupled from the transmit pitch.** More
  general: the output grid becomes a free parameter, and MLA is whatever
  falls out of resampling it. Rejected: the output line positions stop
  translating with the transmit, so the per-scanline delay table is no
  longer one table shifted, which is the property §8 and the TT placement
  both lean on. It also makes MLA 1 a resampling special case rather than
  an identity.
- **C. Uniform across the whole aperture, ignoring transmit grouping.**
  Simplest to state, but it detaches the receive lines from the beam that
  illuminated them, which is the one fact MLA exists to exploit (§7:
  "the transmit beam has finite width"). Rejected.
- **D. Sensitivity-weighted placement, pulled toward the transmit axis.**
  The two-way physics invites it: for Gaussian transmit (width a) and
  receive (width b) beams, the two-way response of a line at nominal offset
  d peaks at `d·a²/(a²+b²)` — between the line and the transmit axis — with
  amplitude falling as `exp(−d²/2(a²+b²))`. An even nominal grid is
  therefore even in neither sensitivity nor effective position, and trading
  sampling uniformity for outer-line sensitivity is a defensible response.
  Rejected here, in two halves. With no beam model in the repository there
  is nothing to derive the pull-in from, so it could only enter as a
  hand-tuned constant with no provenance — which this project rejects
  everywhere else (ADR-0008 records what an unsourced number costs). And as
  compensation it addresses MLA run *without* transmit compounding: the
  illumination non-uniformity it patches is the artifact §7 adopts stage-2
  compounding to remove, and under compounding an output line is formed
  from several transmits, so "the" axis to pull toward stops being well
  defined — the same physics returns instead as the compounding weight
  function, which is data (§17, gated on the beam model of #9). **Re-heard
  when both hold**: the beam model of #9 has landed, so a placement could
  be *derived* — equalized two-way effective spacing, or an SNR floor —
  rather than tuned; and measurement shows compounding does not recover the
  outer-line sensitivity on its own. The map-as-data structure keeps the
  door open: a derived placement is one more derivation function, uniform
  across transmits, so translation invariance survives it.

**What frame-edge renormalization divides by**

- **E. The sum of the weights actually applied to that line (chosen).**
  With a floor below which the line is refused rather than normalized.
- **F. The count of contributing transmits.** Equivalent to E while weights
  are uniform, and wrong the moment they are not — which is the moment
  compounding weights arrive (§17, gated on the beam model, #9). Adopting it
  would mean rewriting every figure measured under it. Rejected.
- **G. No renormalization; correct the gain downstream.** Rejected: the
  artifact is a smooth lateral shading that reads as anatomy, and a
  downstream correction would need the map anyway.

## Decision

1. **MLA line placement.** Option A. The receive lines of one transmit
   subdivide the **transmit line** pitch evenly and sit symmetric about the
   transmit axis. The pitch is the spacing of the beam axes
   (`TxEvent.line_x_m`), not of the elements: the two coincide only for a
   sequence that fires one transmit above each element, and deriving from
   the elements gives groups of the wrong width for any other stride.
2. **A non-uniform transmit-line grid is refused**, not averaged. It has no
   single pitch to subdivide, and what MLA should do on one is not defined
   here; an angular sequence arrives with convex and sector probes.
3. **Renormalization divides by the applied weight sum.** Option E, at
   derivation rather than in the beamformer, so the beamformer stays a plain
   weighted sum and every consumer of a map sees the same normalization.
4. **A line whose weight sum falls below `WEIGHT_SUM_FLOOR` is refused.**
   Dividing by a vanishing sum multiplies noise into a plausible-looking
   line; wrong output is worse than no output (absolute rules).
5. **These fix geometry and arithmetic, not compounding parameters.**
   Window width, apodization and truncation count stay measurement-owned
   (§17) and gated on the beam model (#9). The synthetic uniform-weight map
   that exercises the multi-contribution structure is a fixture and selects
   no production value.

## Consequences

- The output grid is now specified, so an L0 comparison of two
  implementations at MLA 2 or 4 compares like with like. Before this record
  a port could satisfy every stated requirement and still place its lines
  half a pitch away.
- MLA 1 is not a special case: it is Option A at `mla = 1`, and a test pins
  that its line positions equal the identity map's exactly, with no
  tolerance.
- Every image-quality figure measured under compounding is stated against
  the applied-weight-sum normalization. If option F were ever adopted, those
  figures would have to be retaken, which is the reason to record the choice
  now rather than when the weights stop being uniform.
- Frame-edge lines have the same *work* as interior lines and differ only in
  their weights: the map pads short rows with inert slots. That is a
  separate property from this record's, and it is what makes the
  no-variable-loops absolute rule assertable on data.
- A sequence whose beam axes are unevenly spaced cannot use `mla_map` at
  all. That is deliberate for now; defining angular MLA is work that belongs
  with the probes that need it.
- Sensitivity-weighted placement is recorded in option D with its two
  re-hearing conditions, because this record's content is immutable once it
  lands (workflow §6). Until a re-hearing, an outer MLA line that measures
  dark is evidence for the compounding weights to absorb, not a reason to
  move the line.

## Status history

- 2026-08-30: Proposed in pull request #57, which carries the contribution
  map, both placements, the renormalization, and the tests that pin them.
  Held at `Proposed` while that pull request is open; the transition to
  `Accepted` is written in the same pull request at the owner-authorized
  pre-merge boundary, so `Proposed` never reaches `main` (workflow §6,
  ADR-0004).
- 2026-08-31: Option D added to the placement alternatives —
  sensitivity-weighted placement pulled toward the transmit axis, argued
  from the two-way response peaking between the line and the transmit axis
  — rejected with two named re-hearing conditions. The decision itself was
  unchanged; the record's rejected alternatives were completed while it was
  still `Proposed`, which is the only time this record's content may change
  (workflow §6). The divisor options were relabelled E, F, G so no letter is
  used twice.
- 2026-08-31: Accepted at PR #57's owner-authorized pre-merge boundary
  (`DEC-57-001`), after the review's implementation findings were corrected
  — transmit-type matching in the map, and the frame's parameter generation
  taken from the configuration — and validation was green at that head.
