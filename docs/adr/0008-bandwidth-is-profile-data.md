# ADR-0008: The effective two-way bandwidth is probe-profile data

- Status: **Accepted** — 2026-08-23
- Date: 2026-08-22
- Owner decision: yes (`DEC-46-BANDWIDTH-AUTHORITY-001`)

## Context

design.md §5 derived its band-edge phase-error figures — 47° at 13 MHz,
54° at 5 MHz — from "band edges" of 5.2 MHz and 1.5 MHz, without saying at
what spectral level those edges sat or who owned the number. It went
unnoticed while one frequency was all that was derived from it. The kernel
sweep behind ADR-0007 made it load-bearing (#22, #46): the ranking of
four-tap kernels changes with the assumed pulse width, and the same number
decides how much of the four-tap error matters at D=8 against D=4, which
§17 keeps open. #6, the first IQ-path MVP, would have had to interpret its
decimation sweep against an undefined quantity.

The repository already had a different source. §4 lists bandwidth among
the settings a probe profile bundles; `ProbeProfile.bandwidth_frac` exists,
is 0.7 for the implemented `linear-5mhz`, and drives the simulator's
effective two-way Gaussian pulse. Under that pulse the 0.7 is the full
fractional width between the half-amplitude points, so the one-sided edge
is 1.75 MHz, not 1.5 MHz, and the 5 MHz phase-rotation-only error is 63°,
not 54°. §5's arithmetic and the implemented profile disagreed. Separately,
§15 had frozen an RF-oracle benchmark at the same 0.7 (#25), the external
transmit contract §19 carried no pulse bandwidth, no 13 MHz profile existed,
and no manufacturer or measured provenance existed for any value.

The question was therefore not only *what level* — it was *who owns the
number*, what it physically means, whether it belongs in the external
transmit schema, and how it relates to the frozen oracle.

## Options considered

- **A. Profile data.** The effective two-way pulse bandwidth is `ProbeProfile`
  data under §4, with an explicit level convention and per-probe provenance
  or provisional status. §5 derives its edges from it. §19 setup maps each
  transmit configuration to one `probe_profile_id`, so the runtime config ID
  selects the profile transitively without carrying a bandwidth. §15's
  benchmark stays a separately named, frozen synthetic record. Matches where
  the code already keeps the number; adds the smallest possible thing to the
  external contract (an id reference, not a physical quantity).
- **B. Transmit-schema data.** Add a pulse bandwidth to §19's physical-quantity
  transmit description and define how it combines with the probe and AFE
  response. Puts a receive-side processing assumption into a contract owned
  for transmit description, duplicates a number the profile already holds,
  and would need its own contract synchronization and a rule for what wins
  when the two disagree. Rejected.
- **C. Leave it synthetic.** Keep §5's assumptions as separate synthetic
  figures and defer any physical profile bandwidth. Leaves #6's
  interpretation and the profile-specific rerun of #25's measurement
  unresolved, and leaves §5 and the implemented profile in contradiction.
  Rejected.

A is chosen — owner decision `DEC-46-BANDWIDTH-AUTHORITY-001`, recorded
on issue #46.

## Decision

1. **Ownership.** The effective two-way pulse bandwidth is probe-profile data
   (§4, `ProbeProfile`). It lives there and nowhere else.
2. **Semantics.** `bandwidth_frac` is the full fractional bandwidth of the
   effective two-way pulse relative to `f0`; its endpoints are the two points
   half amplitude (20·log10 2 ≈ 6.0206 dB) below the spectral peak —
   equivalently −6.0206 dB in power. The one-sided analysis edge of the
   symmetric baseband model is `bandwidth_frac · f0 / 2`. It describes the
   effective two-way pulse the processing profile assumes, not a transmit-
   only spectrum and not an AFE anti-alias guarantee.
3. **Provenance.** `bandwidth_source` names the manufacturer data or
   measurement behind the value, or is `None`, which means *provisional* —
   never measured, manufacturer-backed or validated. A consumer of a
   provisional value says so beside its result and reruns when the value or
   its provenance changes. Provisional is a statement of evidence, not
   permission to choose another number. Missing provenance is stated, not
   invented: `linear-5mhz` keeps 0.7 with no source.
4. **Derivation.** §5 derives its profile-specific band edges and every figure
   built on them from the profile. Where no profile exists — 13 MHz until
   #10 — the sweep runs on a named synthetic design envelope and every such
   figure carries that label (`synthetic-80pct-design-envelope`), claiming
   no profile or physical authority.
5. **External contract boundary.** §19 setup maps each transmit configuration
   to exactly one `probe_profile_id`; the runtime config ID selects the
   profile transitively; `docs/dataplane.md` counts the profile as part of
   the table set a (config ID, generation) names. No bandwidth value enters
   §19, and no frame-header or data-plane field is added or changed.
6. **Separation from the frozen oracle.** §15's benchmark is
   `rf-oracle-frozen-0p7`: a historical synthetic record whose 0.7 is its own
   constant. It is not relabelled as a profile record even when a profile
   carries the same number; its residuals and its 1.082 % / 0.791 %
   acceptance limits stay as frozen. What a profile implies is a separate
   profile-reconciliation output reported beside it.

## Consequences

- §5's 5 MHz figures become the profile's: 63° / 31.5° without interpolation
  at D=8 / D=4, and for the Lagrange cubic 14.00 % / 2.39 % pulse-weighted
  error against the 10.82 % / 1.40 % the former 1.5 MHz edge gave. The kernel
  choice of ADR-0007 is unchanged — this record fixes what the sweep assumes,
  not which kernel it picks — and the narrative in §5 is restated to what the
  regenerated tables show.
- The frozen acceptance limits of §15 are no longer "one tenth of §5's
  current figure": they are one tenth of the figure as it stood at the
  freeze, recorded as such. The 5 MHz limit is now stricter than a tenth of
  what is measured; the 13 MHz one is 0.003 points looser than a tenth of the
  exact-level envelope figure; both sit far above the production residuals.
- #6 consumes `linear-5mhz` as a named provisional profile, labels its
  artifacts accordingly, and reruns them if the value or provenance changes.
  #10 creates the 13 MHz profile under the same definition and triggers the
  13 MHz reconciliation and the rerun of the 13 MHz sweep.
- A sourced bandwidth arrives through a reviewed profile update, never by
  editing a synthetic figure in place; the sweep and the reconciliation
  rerun, and the frozen record does not.
- If implementation of any of the above needs a broader external-API or
  record change than the `probe_profile_id` reference, that is a new owner
  decision, not an extension of this one.

## Status history

- 2026-08-22: Proposed in pull request #50, which carries the profile
  definition (§4), the profile-derived §5 sweep, the §19 / dataplane
  mapping, the profile reconciliation beside the frozen oracle, and the
  tests that pin them. Held at `Proposed` while that pull request is open;
  the transition to `Accepted` is written in the same pull request at the
  owner-authorized pre-merge boundary, so `Proposed` never reaches `main`
  (workflow §6, ADR-0004).
- 2026-08-23: Accepted at the pre-merge boundary of that pull request, on
  owner instruction, after four review-driven commits and with every check
  green at the head that carries this line. The effective date is the day of
  the transition and the `Date` line above stays the day the record was
  written, as ADR-0003 and ADR-0007 also separate them.
