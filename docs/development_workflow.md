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

**One pull request may close several issues** when they came out of one
review pass or otherwise form a single small change, and only then. The
requirement is that traceability survives: **one commit per issue**, each
naming its issue, and every issue listed with `Closes #NN` in the PR body.
Bundling is for keeping review and merge overhead proportionate to a small
diff — never for grouping unrelated work, and never for a decision that
establishes or alters a contract, which is reviewed on its own. Recording
an already-settled rule in a contract document is not such a decision.

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
```

- type: `feat` / `fix` / `docs` / `test` / `refactor` / `build` / `ci` / `chore`.
- Header: English, lowercase imperative, no trailing period, ≤ 72 characters.
- Body: `- ` bullet items only; may be omitted for trivial commits.
- `BREAKING CHANGE: <description>` footer when applicable.
- **Commit granularity matters.** Normal merge is the default (§7), so
  feature-branch commits are part of `main` history and should be meaningful,
  self-contained steps.

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

- Location: `docs/adr/NNNN-<slug>.md`, numbered sequentially.
- Format, five sections: **Context / Options considered / Decision /
  Consequences / Status**.
- Status: `Proposed` → `Accepted`. A reversal creates a **new** ADR with
  `Supersedes: NNNN`; the old one gains `Superseded-by: NNNN`. The decision
  history lives in this chain — do not rewrite old ADRs.
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
