# ADR-0007: The interpolation kernel is part of the L0 contract

- Status: **Accepted** — 2026-08-17
- Date: 2026-08-16
- Owner decision: yes

## Context

Acceptance of any port of enodia is numerical equivalence with the reference
implementation — L0 (design.md §0, §15). design.md §5 required 4-tap
interpolation for fractional delays and wrote it as `interp4(z_dec, n − d)`
without saying which interpolation. Review of #21 pointed out what that
leaves open (#22): two implementations that pick different 4-tap kernels —
Lagrange, Catmull-Rom, a windowed sinc — both look right in isolation and
differ in exactly the way L0 exists to catch, so an unnamed kernel makes the
reference implementation the only authority by accident rather than by
decision.

While #22 was open, a proposal arrived to go further: make the kernel a
replaceable parameter of the reference drawn from a closed enumeration,
carry the kernel ID in a processing profile, and define L0 per
(implementation, kernel ID) pair. Its stated ground was that the 4-tap
choice is an adaptation to the accelerator's memory, and that a
higher-order kernel in the golden path (#25) would otherwise leave L0
undefined against a 4-tap port.

Checked against the repository, neither ground holds. What L1 residency
forces is *where* delays are applied — after IQ demodulation, never on raw
RF (CLAUDE.md, absolute rules); the tap count comes from §5's band-edge
error, which is a signal-processing requirement at a given oversampling
ratio and hardware-neutral. And the golden RF path is not L0's counterpart:
it is the yardstick that measures the IQ path's approximation error, and is
designed to differ from it (`enodia/spec/beamform/__init__.py`, #25). The
IQ reference implementation is L0's counterpart, and #25 does not touch it.

Three principles in the proposal are right, and this record adopts them.
Its mechanism is deferred, for a reason recorded below.

## Options considered

- **A. Name the kernel in §5 and leave it at that.** Fixes #22's defect.
  But it leaves the *relationship* between kernel and L0 unstated: a later
  reader could still argue that a port with a different kernel is "close
  enough" if the numbers land within threshold, which is the tolerance
  question this record closes.
- **B. Kernel enumeration in the data contract now** — a kernel ID in the
  record metadata or a profile, and L0 defined per (implementation, ID).
  Correct in shape, and it is where this would go if a second kernel and a
  second consumer existed. Neither does: nothing implements the IQ path
  yet (#6 is unstarted), and `enodia/tt/` holds a benchmark harness. A
  mechanism designed against one real case and one imagined one is a
  contract nobody can yet review against use. Also, `docs/dataplane.md`
  already carries a config ID and parameter-generation counter per record;
  if a kernel identity ever has to travel with data it belongs there, not
  in a new "profile" concept that would collide with probe profiles.
- **C. Absorb kernel differences in the L0 tolerance.** Rejected outright.
  The threshold exists to bound implementation error against a fixed
  specification; spending it on a specification difference makes L0 unable
  to say what it was built to say.
- **D. Name the kernel, state its place in L0, and give the reference a
  seam with one value defined.** A is done; the seam is not a contract but
  costs nothing to widen; B is revisited when it has two of everything.

D is chosen.

## Decision

1. **The interpolation kernel is named in the specification** — family,
   closed-form coefficients, tap order, coordinate convention, and boundary
   rule — such that two implementations cannot differ (design.md §5).
2. **A port runs the kernel the specification names, and L0 compares like
   with like.** Checkpoint 2 of §15 (post-delay channel vectors) is a
   comparison between two implementations of the same kernel.
3. **A kernel difference is never absorbed by a tolerance.** The L0
   threshold bounds implementation error; it is not widened to accommodate
   a different specification.
4. **The reference implementation takes the kernel as a named argument
   from a closed set with exactly one member.** The argument exists so that
   an L0 run names what it compares; widening the set is a change to this
   contract, made by a new record, not by adding a string.
5. **A kernel identity in the data contract is deferred** until a second
   kernel and a second consumer both exist. If it comes, it rides the
   config ID / generation mechanism `docs/dataplane.md` already has.

Which kernel is not decided here: it is a sweep-backed parameter and lives
in design.md §5 with its sweep, per AGENTS.md rule 7. That the specified
kernel is currently Lagrange cubic, chosen for robustness across two open
parameters rather than for dominance, and provisional on an axial-PSF
measurement, does not change anything decided above — which is exactly the
point of separating the two. The contract is that *a* kernel is named and
that L0 runs it; replacing the named kernel is a §5 change with its sweep,
not a change to this record.

## Consequences

- #22 closes with §5 carrying a definition two people implement the same
  way, including at the record ends, and #6 can proceed against it.
- #25 is unaffected in scope and gains a clarification: it changes the
  yardstick's own interpolation, which is not L0's counterpart, so it does
  not need — and must not be framed as — a change of the L0 reference.
- The one-member set is a real constraint on the code: a test asserts it,
  and a second member fails that test until a record widens the set.
- The reasoning for the tap count itself — §5's band-edge argument, and the
  sweep now behind the kernel — stays in the specification, hardware-neutral,
  and not in a placement-derived rule.

## Status history

- 2026-08-16: Proposed in pull request #45, which carries the kernel
  definition in design.md §5, the reference implementation of it, and the
  sweep behind the choice. Held at `Proposed` while that pull request was
  open, per owner decision CLPR45-D1.
- 2026-08-17: Accepted at the pre-merge boundary of that pull request, on
  owner instruction. The effective date is the merge day and the `Date`
  line above stays the day the record was written, as ADR-0003 also
  separates them.
