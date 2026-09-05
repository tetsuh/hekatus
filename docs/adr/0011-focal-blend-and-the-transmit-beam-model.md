# ADR-0011: The focal blend, and what the transmit beam model is measured against

- Status: **Proposed**
- Date: 2026-09-05
- Owner decision: pending

## Context

`docs/design.md` §18 lists the transmit beam model among the details settled
at implementation time, and it is unusually specific about one of them:

> **the virtual-source focal singularity (mandatory)**: the transmit delay
> `±|z − z_f|/c` flips sign at the focal depth, producing a discontinuity and
> hourglass artifacts near focus — a classic retrospective-transmit-focusing
> pitfall. Build the smooth blend across the focus from the start.

It was not built from the start. The simulator has carried
`sign(z_s − z_v)·|r_s − r_v|` since MVP-1, and off the beam axis the two
one-sided limits at the focal depth differ by `2|x_s − x_v| / c` — several
periods of `f0` for a scatterer a few wavelengths off axis.

**The existing suite could not have caught it.** Every simulator test places
its scatterer at exactly `z = 20 mm`, which is the focal depth, and that is
the one depth where the unblended model is well behaved: `sign(0)` is zero,
so the blended and unblended models agree there exactly. The artifact lives
immediately either side of the focus, not on it.

Three things had to be decided to close this, and none of them is a
parameter a later sweep can quietly revise: they fix what the reference
simulator's transmit field *is*, and every image metric computed from
simulated data is stated against it.

## Options considered

**How to carry the sign through the focus.**

- **A global smooth sign, e.g. `tanh(u/Δ)`.** One expression, no seams. But
  `tanh` is asymptotic: it perturbs the arrival time at every depth to remove
  a defect that exists at one, and the perturbation only becomes negligible
  several Δ out. A model that is wrong everywhere by a little, to fix
  somewhere by a lot, is the wrong trade for a reference implementation whose
  job is to be the thing ports are compared against.
- **A compactly supported blend**: `sign(u)` outside `|u| < h`, something
  smooth inside. The delay field outside the transition zone is then the
  unblended one *bit for bit*, so the change is a local repair and every
  existing golden that sits outside the zone is provably untouched rather
  than tolerably close.
- **Leave the delay singular and taper the amplitude to zero near the
  focus.** Removes the artifact by removing the signal, and puts a dark band
  at the depth the operator focused on. The defect is in the delay.

**What smooth function inside the zone.**

- **A linear ramp** `u/h`. Continuous at the seams, with a corner there: the
  first derivative jumps from `1/h` to `0`. Trading a discontinuity in the
  delay for a discontinuity in its slope moves the artifact rather than
  removing it.
- **`sin(πu / 2h)`.** Meets ±1 at ±h with zero slope, which is `sign`'s own
  slope outside, so the result is C¹ across the focus *and* across both
  seams. Odd, so it introduces no asymmetry the geometry does not have.

**What sets the width.**

- **A length in millimetres.** Does not travel between profiles; a 13 MHz
  probe (#10) would need its own number for the same physics.
- **Wavelengths.** Travels, but is the wrong scale: the region where a
  converging wavefront stops resembling a spherical wave from a point is set
  by how much curvature the aperture imposes, not by the period.
- **`λ·F#²`.** The depth-of-field scale: the axial extent over which the
  wavefront curvature across the aperture falls below a wavelength. Dimensionally
  it is the scale the physics names, and it carries the F-number, which is
  what actually decides how tight the focus is.

**What the width is measured against.** A free parameter chosen by eye is
what the implementation style forbids — design parameters are decided by
measurement. Measuring the blend against the unblended model is circular; it
needs a model that does not share the virtual-source assumption. Aperture
superposition is that model, and design.md §18 already names it as the
switchable higher-fidelity alternative. It also has no free parameter of its
own and no singularity to blend, because it never introduces a virtual
source: it sums the pulse over the transmitting elements using the
per-element firing delays and apodization the transmit schema (#52) already
carries and validates.

## Decision

1. **The blend is compactly supported and C¹.** `blended_sign(u, h)` is
   `sign(u)` for `|u| ≥ h` and `sin(πu / 2h)` inside. Outside the transition
   zone the delay field is the unblended one exactly — the blend is a local
   repair, not a reshaping of the beam.

2. **The half-width is `BLEND_HALF_WIDTH_FACTOR · λ · F#²`**, the
   depth-of-field scale, so it travels between profiles rather than being a
   length one probe happens to want.

3. **The factor is selected by measurement, and the measurement is
   `enodia.spec.sim.blend_sweep`.** It minimises the worst arrival-time
   *shape* error against the aperture-superposition model over the focal
   region. The two models do not share a time origin — the firing delays are
   referenced so the earliest firing element sits at zero, which the schema
   states — so both fields are referred to their own value on the beam axis
   at the focal depth and the per-event constant cancels. The selected value
   is **2.0**, with a worst-case error of 1.33 periods against 4.02 for the
   unblended model. The minimum is interior to the swept range, not at an
   end, which is what makes it a selection.

4. **Aperture superposition is the yardstick, not the default.** design.md
   §18 adopts the virtual-source approximation and keeps superposition
   switchable; `enodia.spec.sim.transmit` follows that, exactly as the
   RF-domain ideal-delay DAS is kept as the golden path for the IQ
   approximation rather than becoming the processing path.

5. **The unblended model stays, as a named function.** `virtual_source_unblended`
   is not a model a caller should select for imaging; it exists so the defect
   the blend removes can be exhibited. A fix whose defect cannot be shown is
   not evidence of anything.

## Consequences

- Simulated frames change for scatterers within `2·λ·F#²` of the focal depth
  and nowhere else. On `linear-5mhz` that is 17.54–22.46 mm. Frames outside
  the zone are bit-for-bit what they were, which a test asserts rather than
  assumes.
- Every existing simulator test sits either outside the zone or exactly on
  the focal depth, so none of them moved — and that fact is now recorded as a
  gap in coverage that the new tests close, not as evidence of correctness.
- The reference implementation now has two transmit models, and L0
  equivalence is stated against the default one. A port is not required to
  implement superposition; it is required to reproduce the blended
  virtual-source field, kernel and all, in the same way ADR-0007 requires the
  interpolation kernel.
- The residual is bounded below and known: a one-delay model cannot beat the
  spread of the arrivals it stands for, which the sweep reports as 0.44
  periods. The blend reaches 1.33. Closing that gap means abandoning the
  one-delay model, which is what selecting superposition does.
- **The contribution-map weight function is now unblocked but not decided.**
  design.md §7 and §17 keep it open, ADR-0010 records what stops holding once
  the weights are not uniform, and this ADR supplies the beam model that
  question needs. It is its own decision and its own issue.
- `λ·F#²` assumes a focused transmit with a real F-number. Plane-wave and
  diverging-wave transmits, and the shear-wave push and tracking transmits
  the open transmit-type set will bring, have no focal singularity to blend
  and no `F#` to scale by. They do not use this path; when one of them needs
  a beam model it will need its own decision.

## Status history

- 2026-09-05: Proposed with #9.
