# ADR-0006: How many commits a bundled pull request may have

- Status: **Accepted** — 2026-08-16
- Amends: ADR-0001 (development process)
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

**The rule has never been met.** Two pull requests in this repository have
closed more than one issue. #24 closed four with seven commits — three
for #18, two for #19, one each for #20 and #23 — and `f9e6d70`, the commit
that wrote the rule, is one of the three it contributed to #18. The rule was
broken by the pull request that introduced it, in the act of introducing it,
and nothing flagged it. #36 is the second bundled pull request and the first
occasion the rule was enforced.

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
- **D. Remove the permission to bundle at all**, holding `1 Issue = 1 PR`
  without exception. The count question then does not arise rather than
  being answered: with nothing bundled, §7 governs every branch uniformly
  and no exception can be needed. This is also the narrower thing the owner
  decided — `1 Issue = 1 branch = 1 PR` carries an owner decision, while the
  count never did. Rejected on the evidence of use. Bundling has been used
  twice and both times for what it was written for: #24 gathered four small
  findings from two documentation reviews, and #36 carried a measurement
  together with the record of the machine it ran on, whose acceptance
  criteria required data that same run produced. What has failed is the
  count, not the permission. Splitting a review pass into one pull request
  per finding also multiplies review cycles across a diff smaller than one
  ordinary change, which is the overhead the permission exists to avoid.
- Also weighed and not viable: **enforce the count by squashing at merge.**
  That reverses ADR-0001 §Decision 1 outright, and for a measurement branch
  it destroys the provenance in the same stroke.

C is chosen. It is what the rule's own sentence already said the
requirement was — traceability — with the count recognized as one means of
reaching it rather than the thing being required.

D was raised by the owner while reviewing the pull request that carries this
record, on the grounds that no owner decision on commit count had ever been
made. That is correct, and it is why D belongs here: the first draft of this
record weighed only how to set the count, never whether the rule it hangs
from should exist. The owner chose C over D on the use evidence above. The
omission is recorded rather than quietly repaired, because a record that
lists no option the decision could have gone to has not shown its work.

## Decision

1. **Traceability is the requirement.** A bundled pull request satisfies it
   when all three hold: every authored commit header names exactly one
   issue; every issue a commit names appears in `Closes #NN` in the
   pull-request body; and every issue in `Closes` is named by at least one
   commit. Platform-generated integration commits stay exempt, as ADR-0003
   made them. All three conditions are mechanically checkable from the
   branch and the body alone.
2. **The commit count is not constrained.** §7 governs a bundled pull
   request exactly as it governs any other: history keeps the meaningful
   steps. A bundled branch is not squashed per issue.
3. **Bundling is narrowed to what the pull request itself produces.** A
   pull request is opened for exactly one issue. Issues raised while it is
   open — by its review, or by the work itself — may be closed by it as
   well, when the change they call for belongs in the same diff. Issues that
   already exist when a pull request is opened are never grouped into it,
   and a decision that establishes or alters a contract is still reviewed
   on its own.

   This is an owner amendment to option C, made while this record was under
   review. It draws the line where the two cases actually differ. Bundling
   decided at filing time is the grouping the rule was always meant to
   prevent. Bundling that arrives afterwards is not a grouping at all: the
   workflow already requires a problem found during work to become an
   issue, and closing it where it was found keeps the fix with the change
   that caused it.

   The amendment was first worded as "raised by review", and widened to
   "raised while open" before landing, when the first issue found under it
   (#40) was raised by the author while writing this record rather than by
   a reviewer. The narrower wording excluded a case its own rationale
   covers, and excluded it for no reason the rationale gives: an issue the
   work surfaces is no more a filing-time grouping than one a reviewer
   surfaces, and pre-existing issues are barred by a separate sentence
   either way.
4. **A rewrite must preserve the harness revision a measurement record
   names.** Commits after that revision may be rewritten as any branch may
   — #36 itself had three rewritten under review with the record intact.
   Rewriting or removing the named revision invalidates the record, and
   re-running the measurement is the only way to rebuild such a branch.
   What the rerun does with the old record is ADR-0005's rule, not a new
   one: a record that has landed on `main` is never rewritten and the rerun
   supersedes it with a later record; a record still on the unmerged branch
   has not landed, and the branch replaces it — as #36 did when it retook
   its measurement before merge. This follows from ADR-0005 rather than
   from anything decided here, and
   it is written into §7 because it is a merge-policy fact: it constrains
   what may be done to a branch, not what the record must contain.

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
- Bundling becomes rarer, not more attractive. Both historical cases were
  bundled at filing time and would be filed differently today. Three of
  #24's four issues were raised in review of #16 and would be closed by #16
  itself; the fourth, #23, was raised in review of #21 and would be closed
  there. #36 would be opened for #11, with #28 following it. Neither loses
  the work — the fixes land closer to what caused them.
- A pull request that grows a second issue while open does not become
  irregular for it, which is the case that produced two exceptions on #36.
- ADR-0001 is amended. Its `1 issue = 1 branch = 1 PR` gains the one
  exception Decision 3 states, and its status history records that. Its
  §Decision 1 — normal merge, step-level history — is unchanged and now
  reaches the bundled case it always described.

## Status history

- 2026-08-16: Accepted in pull request #38, which carries the change to
  `docs/development_workflow.md` §2 and §7 that this record decides.
