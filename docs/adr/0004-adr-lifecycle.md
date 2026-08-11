# ADR-0004: The ADR lifecycle — states, and when each is written

- Status: **Accepted** — 2026-08-12
- Date: 2026-08-12
- Owner decision: yes

## Context

Three records once read `Status: Proposed`, each promising in its own words
to become accepted when the change carrying it landed. All three landed — two
in the authorized bootstrap commit, one by a merged pull request — and all
three still said `Proposed`. The cause was not carelessness but design:
advancing the status was an action separable from the landing, and nobody
owned it.

Repairing that (#33) required deciding how the lifecycle works, and review
found the first attempt incoherent in three ways. It said an ADR reaching
`main` is accepted **by that fact**, and in the next paragraph allowed a
`Proposed` record to sit on `main` — a rule contradicting itself within a
section. It named `Rejected`, `Superseded`, and `Deprecated` without saying
when any of them is written, leaving three states as decoration. And it
declared a five-section format while every existing record carried a sixth
section the format did not mention.

Changing how decisions are recorded is itself a process decision, so it needs
a record. That this is an ADR about ADRs is recursive but not circular: the
rule requiring it predates it.

## Options considered

On whether a `Proposed` record may reach `main`:

- **A. Never.** `Proposed` is the state of an ADR inside an open pull
  request only. A decision that needs wider discussion uses an issue, which
  is what issues are for. Nothing on `main` can be stale, because nothing on
  `main` is unfinished.
- **B. Allowed when explicitly marked**, naming the issue that will decide
  it; the deciding pull request then advances it. This keeps the option of
  publishing a proposal for comment, and bounds the drift risk by giving the
  pending flip an owner — the deciding issue cannot close without it.

B is what the reviewer recommended. A is chosen here, because the failure
being repaired was precisely a status waiting for someone to come back to it,
and B reintroduces that shape in a smaller form. The cost of A is the ability
to publish an undecided record, which this repository has not yet needed and
which an issue serves. If that need appears, a later ADR can adopt B; the
reverse — noticing drift after it has happened again — is the outcome this
one exists to avoid.

## Decision

1. **`Proposed` never reaches `main`.** It is the state of a record under
   review in an open pull request. Everything on `main` is settled:
   `Accepted`, `Rejected`, `Superseded by ADR-NNNN`, or `Deprecated`.
2. **Every transition is written in the pull request that carries the
   decision**, never afterwards. The status line records the state and its
   effective date (`Accepted — 2026-08-12`), and the record's
   `Status history` gains one line naming what caused it.
3. **`Status history` is part of the format**: a record is a header status
   line followed by five sections, of which it is the last. It is where a
   record says how it came to say what it says.
4. A record that is written and then declined lands as `Rejected` rather
   than disappearing: the status is flipped in the record's own pull
   request, which then merges — declining the decision, never discarding
   the record. Numbering therefore stays gapless, since a number is spent
   only by landing; a draft abandoned before it is worth keeping is closed
   unmerged and never consumed one.

## Consequences

- No status can drift, because no status is advanced by an action separable
  from a merge. This is the property the previous wording lacked.
- A proposal that genuinely needs discussion before a decision is held in an
  issue or in an open pull request, not on `main`. If that proves too
  restrictive, a later ADR adopts option B above.
- The declared format now matches the records as they are — a header status
  line and five sections ending in `Status history` — instead of describing
  a shape none of them had.
- `Rejected` and `Deprecated` become reachable states rather than names, so
  the record of a declined option has somewhere to live.
- **The three records repaired alongside this one are a one-time exception**:
  their acceptance is written after the changes that caused it — the
  bootstrap commit for two, a merged pull request for one — because the rule
  requiring it to be written at the time did not exist then. Under this ADR
  no further backfill can arise, since a record cannot reach `main`
  undecided.

## Status history

- 2026-08-12: Accepted in pull request #34, which repaired three stale
  statuses; review of that repair found the lifecycle rule contradicting
  itself, which is what this record settles.
