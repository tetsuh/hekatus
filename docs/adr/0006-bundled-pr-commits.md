# ADR-0006: How many commits a bundled pull request may have

- Status: **Accepted** — 2026-08-16
- Date: 2026-08-16
- Owner decision: yes

## Context

The workflow permits one pull request to close several issues when they form
one small change, and requires **one commit per issue** when it does. PR #36
met the precondition and could not meet the count. It was filed with two
commits and merged with ten, the rest added by three rounds of review. Two
separate owner exceptions were granted, one at nine commits and one at ten,
because the count kept moving while the reason for it did not.

Three things are wrong, and they are independent.

**The count contradicts ADR-0001 for exactly the pull requests it governs.**
ADR-0001 §Decision 1 makes normal merge the default because "one commit per
PR is too coarse for retrospection; `main` history should preserve
meaningful step-level commits". One commit per issue is that same squash
applied per issue. So two small issues filed separately land with their
steps intact, and the same two issues bundled — for the sole purpose of
keeping overhead proportionate to a small diff — are required to land
coarser. The rule penalizes the thing it exists to make cheap.

**It was never weighed against that decision, because it has no record.**
The rule entered the contract document in `f9e6d70`, a commit *inside the
pull request it was written to authorize*, naming #18 — an issue about the
capacity basis of the budget tables. Its own body says as much: the workflow
"said the 1:1:1 unit was 'normal' without saying what the exception is; this
PR is one, so make the rule explicit". The exemption in the same paragraph —
recording an already-settled rule is not a decision needing an ADR — does
not reach it, because the rule was not already settled. It was new, and it
was a process decision.

**It collides with ADR-0005.** A measurement record names the harness
revision that computed it, and that revision is always an **ancestor** of
the commit carrying the record: the measurement runs against a committed
tree, so the commit holding the result can never be the commit the result
names. #36's record names `112ff585`, a commit inside the same pull request.
Squashing #11 into one commit leaves the field pointing at a SHA that no
longer exists, and editing the field to match is inventing a provenance
rather than recording one.

Underneath all three is a general property: **review makes pull requests
grow.** A rule that can be satisfied when a branch is filed, and not after
it has been reviewed, is broken hardest by the changes that received the
most scrutiny — which are the changes whose history is worth keeping.

## Options considered

- **A. Keep the count; split instead.** Every commit that would exceed the
  allowance gets an issue of its own, so #36 becomes harness (new issue) →
  measurement run → record (#11) → machine (#28). Fully consistent, and it
  keeps the guarantee that a bundled diff can be separated by taking one
  commit per issue. Rejected: it files issues for mechanical reasons, which
  is the governance surface ADR-0001 declined to buy; and it does not
  address the growth problem, since a branch that grows under review would
  need a new issue per review round to stay compliant.
- **B. Exempt measurement pull requests**, naming ADR-0005 as the reason.
  Narrowest change, and it settles the case actually encountered. Rejected:
  it leaves the contradiction with ADR-0001 standing everywhere else, and
  the exemption would have to be widened at each new class of change that
  meets the same wall — which is a rule that never finishes being written.
- **C. State the invariant as traceability; leave the count unconstrained.**
- Also weighed and not viable: **enforce the count by squashing at merge.**
  That reverses ADR-0001 §Decision 1 outright, and for a measurement branch
  it destroys the provenance in the same stroke.

C is chosen. It is what the rule's own sentence already said the
requirement was — traceability — with the count recognized as one means of
reaching it rather than the thing being required.

## Decision

1. **Traceability is the requirement.** A bundled pull request satisfies it
   when all three hold: every commit header names exactly one issue; every
   issue a commit names appears in `Closes #NN` in the pull-request body; and
   every issue in `Closes` is named by at least one commit. All three are
   mechanically checkable from the branch and the body alone.
2. **The commit count is not constrained.** §7 governs a bundled pull
   request exactly as it governs any other: history keeps the meaningful
   steps. A bundled branch is not squashed per issue.
3. **The precondition for bundling is unchanged**: issues from one review
   pass or one small change, and only then; never unrelated work; never a
   decision that establishes or alters a contract, which is reviewed on its
   own.
4. **A branch carrying a landed measurement record cannot be rebuilt
   without re-running the measurement.** This follows from ADR-0005 rather
   than from anything decided here, and it is written into §7 because it is
   a merge-policy fact: it constrains what may be done to a branch, not what
   the record must contain.

## Consequences

- Attribution survives without the count. `git log --grep '#NN'` recovers
  every commit belonging to an issue, which is what "one commit per issue"
  was there to make possible by inspection.
- What is given up: the guarantee that a bundled diff can be separated into
  per-issue pieces by lifting one commit each. In exchange, a bundled branch
  is no longer required to discard the steps that a single-issue branch is
  required to keep.
- The three conditions in Decision 1 are checkable by CI, filed as #39.
  Until that check exists they are checked in review, as the count was.
- Neither exception granted on #36 would have been needed under this rule,
  and the second one — granted because a review round added one commit —
  would not have been requested at all.
- Bundling is not made more attractive. The precondition is untouched, and
  the reason to bundle is still a small diff rather than a convenient
  history.
- ADR-0001 is confirmed rather than amended: nothing in it changes, and its
  §Decision 1 now reaches the case it always described.

## Status history

- 2026-08-16: Accepted in pull request #38, which carries the change to
  `docs/development_workflow.md` §2 and §7 that this record decides.
