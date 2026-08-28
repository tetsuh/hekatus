# ADR-0009: The transmit description is accepted, not trusted

- Status: **Proposed** — 2026-08-28
- Date: 2026-08-28
- Owner decision: pending

## Context

design.md §19 makes enodia a subordinate system. Control software owns the
transmit master data; enodia receives a description of it through an API and
configures receive-side computation from it. §19 fixes the vocabulary of
that description — element coordinates, virtual-source positions [mm],
firing delays [ns], apodization values, transmit-type tags — states that
hekatus owns the schema, and rejects the alternative by name: interpreting
FPGA-facing data would add a reverse converter, break on FPGA revisions, and
give L0 a second specification to disagree with.

None of it was implemented. `enodia/spec/sequence/` carried a `TxEvent` with
a line index, a transmit-type tag, a scanline abscissa and a virtual source,
and a `TransmitConfig` naming one `probe_profile_id` (#46, ADR-0008). The
boundary §19 calls the edge of the whole system existed only as prose, so
every question below was open and none of them had a recorded answer.

What made the questions concrete rather than academic is that the profile
already holds the same geometry the description transports. §4 gives
`ProbeProfile` its element count and pitch, and `element_x()` computes the
coordinates from them in float64. So a description arriving with element
coordinates in it presents two numbers for one quantity, and something has
to say which one the delay tables are derived from and what happens when
they differ.

They do differ. Converting `linear-5mhz`'s element coordinates to
millimetres and back moves 6 of the 128, by at most 8.673617379884035e-19 m.
Bit equality is therefore not available as an acceptance rule for a schema
whose units are millimetres, and a tolerance has to be chosen and justified
rather than assumed away.

§19 also sets out three lines of defence against a transmit description that
disagrees with the machine — the failure mode it describes as "image looks
fine, subtly degraded, near-undiscoverable" — and names the first as "the
physical schema is the specification". That is a claim about what the schema
refuses, and a schema that refuses nothing does not make it.

## Options considered

- **A. Accept and canonicalize (chosen).** The external description is a
  distinct type in §19's units. Ingress checks it against the named probe
  profile, refuses what disagrees beyond a stated tolerance, and then
  computes from the profile's canonical geometry rather than the transported
  numbers. The description carries no derived quantity, so nothing in it can
  contradict a derivative. Costs one conversion layer and one type the
  earlier code did not have.
- **B. Trust the description.** Take the transported coordinates as the
  geometry and let the profile hold only the pulse and the sampling. One
  representation, no tolerance to choose. Rejected: a port is accepted by
  numerical equivalence against this reference implementation (L0,
  ADR-0007's principle), and two implementations handed slightly different
  coordinates compute different delay tables and fail L0 for a reason that
  has nothing to do with the port. It also puts a converter's floating-point
  behaviour inside the L0 contract.
- **C. Trust the profile and drop the coordinates from the schema.** No
  duplication, no tolerance, no check. Rejected: it deletes a term §19 names
  in the vocabulary, and with it the only place where a converter's geometry
  and enodia's can be seen to disagree. §19's second line of defence — the
  test-only reverse converter, diffed against the enodia description — has
  nothing to diff if the description does not carry the geometry.
- **D. Accept in SI units.** Keep one type, in metres and seconds, and let
  the host convert. Rejected: §19 states the units, and the conversion has to
  happen somewhere. Putting it outside the boundary means the contract does
  not say what arrives, which is the thing this record exists to fix.

## Decision

1. **Two types, one boundary.** `TransmitDescription` / `TxEventDescription`
   are the external form, in §19's units — millimetres, nanoseconds,
   dimensionless apodization, an open set of transmit-type strings.
   `accept(description, profile)` returns `TransmitConfig` / `TxEvent` in SI
   units. Nothing downstream of `accept` reads the external form.
2. **The description carries no derivative.** Contribution maps, delay
   tables and phase-rotation coefficients are enodia's (§19, "enodia derives
   its own derivatives"). In particular the external description does not say
   which output line an event forms; `TxEvent.line_index` is derived, and is
   the identity while the map is the identity (#53 generalizes it).
3. **Canonical geometry wins.** Transported element coordinates are compared
   against `ProbeProfile.element_x()` and then dropped. Every derivative is
   computed from the profile's geometry, so two implementations of the same
   profile compute from the same numbers and L0 compares like with like.
4. **The coordinate tolerance is stated in units in the last place of the
   aperture scale**, `COORDINATE_TOLERANCE_ULP = 4` times `np.spacing` of the
   aperture half-width — 4 ulp because a description may pass through more
   than one binary64 unit conversion, each able to move a value by half an
   ulp. At the aperture scale rather than per coordinate, because the spacing
   of binary64 varies with magnitude and an element near the array centre
   would otherwise be held to a tolerance hundreds of times tighter for no
   physical reason. On `linear-5mhz` this is 1.4e-17 m: thirteen orders below
   one element pitch (3e-4 m), and above the 8.7e-19 m the millimetre round
   trip actually costs.
5. **The description must be self-consistent, and inconsistency is refused.**
   For every element that fires — apodization strictly positive — the firing
   delay plus the geometric time of flight to the declared virtual source is
   one instant, within `DELAY_TOLERANCE_NS = 1.0`, which is §19's "ns-class
   delay tolerance" and 1.5 µm of path at the reference sound speed. Silent
   elements are not checked: a zero-weighted element's delay makes no
   physical claim, and checking it would refuse every sparse aperture.
6. **Refuse, never repair.** A profile-id mismatch, a wrong element count, a
   non-finite quantity, a negative apodization weight, an all-silent
   aperture, an empty transmit-type tag, a duplicate event index, or a
   coordinate or delay past tolerance raises. Processing with a wrong
   description is worse than dropping frames (absolute rules), and a repaired
   description is a description nobody wrote.
7. **The transmit-type tag stays an open set of strings.** A configuration
   carrying a tag this implementation has never seen is accepted and carried
   to the frame header; only the empty tag is refused. Shear-wave push and
   tracking transmits join the set later (§11.5).
8. **The fields are validated, not consumed.** The per-element firing delays
   and apodization are checked here; no transmit field is synthesized from
   them. The simulator keeps the virtual-source approximation §18 records.
   The transmit beam model — the focal blend and the switch to aperture
   superposition — is #9, and this record does not pre-empt it.

## Consequences

- The simulator becomes the schema's first consumer, which §18 item 2 and
  §19's closing paragraph both name as how the schema is stress-tested before
  anything is built on it. `simulate_frame(profile, config, scatterers)`
  takes the accepted configuration, so a frame cannot be stamped with the
  identity of a configuration it did not come from.
- No image moves. The raw RF and the golden image of the demo are
  bit-identical to what they were before this record (`rf_sha256`
  `b6401ee8…`, `img_sha256` `236a276f…`), which is the sense in which the
  schema re-describes MVP-1 rather than changing it.
- §19's second line of defence — the test-only reverse converter, diffed
  against the enodia description at ns-class tolerance — now has something to
  diff against. Building it belongs to whatever first carries FPGA-facing
  data; it never enters a data path.
- A probe profile whose element coordinates are not derived from a uniform
  pitch — a measured or per-unit calibrated array — will need the tolerance
  reread, because the round-trip residue is a property of the coordinates,
  not of the rule. The test that measures it fails loudly if it ever reaches
  zero differing coordinates, which is the signal to reread.
- `TransmitConfig` gains `element_x_m`. It is the profile's geometry, carried
  so a consumer holding only a configuration still reads the coordinates the
  derivatives came from.
- Nothing in the data plane changes: no frame-header field is added or
  altered, and `docs/dataplane.md` is untouched.

## Status history

- 2026-08-28: Proposed in pull request for #52, which carries the schema, the
  ingress rule, the simulator's consumption of it, the §19 update and the
  tests that pin them. Held at `Proposed` while that pull request is open;
  the transition to `Accepted` is written in the same pull request at the
  owner-authorized pre-merge boundary, so `Proposed` never reaches `main`
  (workflow §6, ADR-0004).
