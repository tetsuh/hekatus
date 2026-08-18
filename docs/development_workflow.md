# Development workflow

Branching, ticket-driven development (TiDD), testing, ADR, and merge rules for
this repository. Explicit rules take precedence over implicit convention,
because implementation is largely performed by AI coding agents.
[`AGENTS.md`](../AGENTS.md) is the short mandatory entry point; this document
is the complete authority.

## 1. Branching: trunk-based

- `main` is always green and releasable. Direct pushes to `main` are
  prohibited. The sole exception is the repository bootstrap commit, recorded
  in the bootstrap Issue.
- Every change uses one short-lived branch:
  `feat/<issue-number>-<short-kebab-description>` (prefix is always `feat/`;
  the change type lives in the Issue label and commit type).
- No develop / release / long-lived feature branches. Releases are tags on `main`.

## 2. Ticket-driven development (TiDD)

**No Ticket, No Commit.** Every change starts from a GitHub Issue.

- The normal unit is `1 Issue = 1 feat branch = 1 PR`, with `Closes #<n>` in
  the PR body.
- An Issue states, at minimum: purpose, verifiable acceptance criteria, and
  out-of-scope / related issues. (Requirement-ID and contract-registry
  bookkeeping are deliberately not adopted at this stage; see §9.)
- Problems discovered during work get a **new Issue**. Drive-by fixes are
  prohibited (typo-level corrections at reviewer discretion).

**A pull request is opened for exactly one issue.** It may also close
issues raised while it was open — by its review, or by the work itself —
when the change they call for belongs in the same diff. That is the normal
way a change grows under scrutiny: a problem found during work becomes an
issue rather than a drive-by fix, and closing it here keeps the fix with
what caused it.

Issues that already exist when a pull request is opened are **never grouped
into it**, and a decision that establishes or alters a contract is reviewed
on its own. Recording an already-settled rule in a contract document is not
such a decision.

The requirement is that **traceability survives**. It is met when all three
hold:

- every authored commit header names exactly one issue (platform-generated
  integration commits stay exempt, per §3 and ADR-0003);
- every issue a commit names is listed with `Closes #NN` in the PR body;
- every issue in `Closes` is named by at least one commit.

**The commit count is not constrained.** §7 governs a bundled pull request
exactly as it governs any other: history keeps the meaningful steps, and a
bundled branch is not squashed per issue to satisfy this section. ADR-0006
records why, including what a measurement record needs from it.

### Planning model

- **Milestones hold the coarse plan.** Issues are added to milestones
  incrementally; large issue batches are not pre-filed.
- The working unit is an **MVP**: a small increment that runs end-to-end,
  demonstrated by one command, guarded by an acceptance test. Normally
  1 MVP = 1 Issue; split into child issues when large.
- Labels: `mvp-candidate` (next-MVP candidates), `icebox` (inventory), plus
  type labels.

## 3. Commit messages

Conventional Commits with an Issue reference:

```text
<type>(<scope>): <imperative summary> (#<issue>)

- <what changed, and why when non-obvious>
- <verification or constraint>

Co-Authored-By: <name> <email>
```

- type: `feat` / `fix` / `docs` / `test` / `refactor` / `build` / `ci` / `chore`.
- **Scope is required** on authored commits made after ADR-0003 is accepted,
  naming the component the change belongs to (`spec`, `bench`, `workflows`,
  `readme`, …). It makes history scannable by area, and the header is wide
  enough to afford it.
- Header: English, lowercase imperative, no trailing period, **≤ 100
  characters** — the same width this repository sets for code
  (`[tool.ruff] line-length`), so there is one number rather than two.
  Aim shorter regardless: a header that will not fit in a line is usually a
  commit that holds more than one idea. Note that a subject beyond roughly
  72 characters is shown truncated in some web views, with the full text an
  expander away.
- Body: bullet items (`-`) only; may be omitted for trivial commits.
- **Trailers are footers, not body.** Lines of the form `Key: value` after
  the bullets, separated from them by one blank line — `Co-Authored-By:`,
  which is how this repository attributes agent-authored commits, and
  `BREAKING CHANGE: <description>` when applicable — are footers in the
  Conventional Commits sense and in git's own (`git interpret-trailers`).
  The bullet rule does not apply to them. Stated because a reviewer reading
  the bullet rule alone once concluded that most of `main` violated it.
- **Commit granularity matters.** Normal merge is the default (§7), so
  feature-branch commits are part of `main` history and should be meaningful,
  self-contained steps.
- **Platform-generated integration commits are exempt.** The merge commit
  GitHub writes (`Merge pull request #NN from …`) is not an authored change;
  it is the record of an integration, and its subject already names the pull
  request that carries the traceability. The owner may replace it with a
  Conventional subject when the default is unhelpful, but is never required
  to. See ADR-0003.

## 4. Testing (TDD-lite)

- Tests are mandatory for spec/production code. Each MVP has an acceptance
  test derived from its acceptance criteria.
- **RED one-liner**: an implementation PR records one line in its body — the
  command that was run and the test that failed before the implementation
  existed. This proves the test can fail (defense against tautological tests).
- Measurement, sweep-experiment, and docs-only PRs record `RED: N/A (<reason>)`.
- No structured RED-evidence format, and no mandatory RED commit, are required.
  Intentional RED commits are permitted in feature-branch history when clearly
  labeled and followed by GREEN before merge.

## 5. Contract-document synchronization

A change that touches a **contract** must update the contract document in the
same PR. Contracts are:

- `docs/dataplane.md` (inter-process records, stream/tap policies),
- the absolute rules in the project instruction files,
- the external API contract (transmit-description schema).

Learnings from measurement or implementation (parameter decisions, overturned
estimates) are reflected into `design.md` / `docs/open-issues.md` as part of
the PR that produced them.

## 6. ADR: recording decisions and their evolution

- Location: `docs/adr/NNNN-<slug>.md`, numbered sequentially, **without
  gaps**. A number is spent by landing on `main`: a declined record lands as
  `Rejected` and keeps its number, while a draft abandoned before it is worth
  keeping is closed unmerged and never consumed one.
- Format: a header status line (`- Status: **Accepted** — 2026-08-12`)
  followed by five sections — **Context / Options considered / Decision /
  Consequences / Status history**. The last is where a record says how it
  came to say what it says.

### Status

```text
Proposed ──► Accepted ──► Superseded by ADR-NNNN
    │            └──────► Deprecated
    └──────► Rejected
```

**`Proposed` never reaches `main`.** It is the state of a record under review
in an open pull request. A decision that needs wider discussion first is held
in an issue. Everything on `main` is settled, so nothing there can go stale.

**Every transition is written in the pull request that carries the decision**,
never afterwards. The status line names the state and its effective date, and
`Status history` gains one line saying what caused it:

```text
- Status: **Accepted** — 2026-08-12
...
## Status history

- 2026-08-12: Accepted in pull request #34.
```

This is the rule because three ADRs once promised, each in its own words, to
become accepted when the change carrying them landed; all three landed, and
all three still read `Proposed`. A status advanced by an action separable
from the landing is a status that drifts. See ADR-0004.

| State | Written when |
|---|---|
| **Accepted** | the pull request carrying the decision merges — in force from its effective date |
| **Rejected** | the decision goes against the record: the status is flipped **in the record's own pull request, which then merges** — declining the decision, never discarding the record. The reasoning lands with it, so the same option is not proposed again with that reasoning unrecorded |
| **Superseded by ADR-NNNN** | the replacing ADR merges; the same pull request updates the old record's status line and history, and the new record carries `Supersedes: NNNN` |
| **Deprecated** | the pull request that removes what the decision governed updates the record alongside the removal |

**Invariant**: once a record has landed on `main`, in any state, its Context,
Options considered, Decision, and Consequences are not rewritten. To change a
decision, write a new ADR. Corrections of spelling, formatting, and broken
links are permitted; anything that alters what the record says is not.

### When to write one

- **Write an ADR for**: architecture, contract, or process decisions, and any
  reversal of a past decision.
- **Do not write an ADR for**: numeric parameters determined by sweeps or
  measurement (update `design.md` / `docs/open-issues.md` instead).
- A small decision may be bundled with its implementation PR; include `[ADR]`
  in the PR title.

## 7. Merge policy: owner-only authority

- The repository owner makes every merge decision. Agents and automation stop
  at merge-ready: they report evidence (tests, RED line, demo output) and
  residual risks, then wait.
- An owner instruction to merge applies to that PR at its current head, once.
  A new commit or a new blocking finding voids it.
- **Normal merge is the default.** Rationale: one squashed commit per PR is
  too coarse for retrospection; `main` history should preserve meaningful
  step-level commits. Use `git log --first-parent main` for the PR-level view.
- **Squash merge only on explicit owner instruction** for that PR.
  Rebase merge and auto-merge are prohibited.
- **A rewrite must preserve the harness revision a measurement record
  names.** The record names the revision that produced it (ADR-0005), and
  that revision is an ancestor inside the same branch — the measurement runs
  against a committed tree, so the commit holding a result can never be the
  commit the result names. Commits after that revision may be rewritten as
  on any branch. Rewriting or removing the named revision leaves the record
  pointing at a SHA that no longer exists; re-running the measurement is the
  only way to rebuild such a branch.
- Delete the `feat/` branch after merge.

## 8. CI

Required checks (introduced by the bootstrap Issue, extended by later Issues
when a need appears — no staged plan):

- lint (`ruff check`) and tests (`pytest`) on Linux;
- **keyword guard**: scans the tree for prohibited internal keywords. The word
  list is supplied via a CI repository secret and is never committed to this
  repository. Reporting never discloses the prohibited text: contents hits
  report the file and an occurrence count, and a violating path is printed
  with the word masked.

  The guard covers same-repository pull requests and pushes to `main`. A
  pull request from a fork receives no secret and therefore **fails
  explicitly** rather than passing unscanned; such a branch is re-run from a
  branch in this repository. Fork coverage is revisited if external
  contributors appear.

## 9. Deliberately not adopted (reserved for the productization gate)

Contract registry, requirement-ID traceability, gate issues, frozen issue
checklists, and structured RED evidence are **not** adopted in this phase.
They are reserved: when the specification freezes and the project crosses a
productization gate, traceability governance of that grade is introduced.
See ADR-0001 for the reasoning and the comparison that led here.

## 10. Language

English only: code, comments, commit messages, issues, PRs, and documents.
