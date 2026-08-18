# AGENTS.md — Agent Entry Point

This file is the mandatory entry point for AI coding agents (any model) working on
this repository. The complete authority is
[docs/development_workflow.md](docs/development_workflow.md) — read it before
changing anything. **Read `docs/design.md` before writing code**; design decisions and
their rationale live there.

## Non-negotiable rules (summary)

1. **No Ticket, No Commit.** Every change starts from a GitHub Issue.
2. **1 Issue = 1 branch = 1 PR.** Branch `feat/<issue-number>-<short-kebab-description>`,
   PR body contains `Closes #<n>`. A PR may also close issues raised while it
   was open, when the fix belongs in the same diff (workflow §2, ADR-0006).
3. **Conventional Commits**: header `<type>(<scope>): <summary> (#<issue>)`,
   body is a `- ` bullet list only, followed by trailers (`Co-Authored-By:`
   and the like) which are footers, not body. Commit in meaningful,
   reviewable steps —
   normal merge is the default, so feature-branch commits become part of
   `main` history.
4. **main is always green.** Never push directly to `main`.
5. **Tests are mandatory for spec/production code.** Record one RED line in the
   PR body: the command and the test that failed before implementation.
   Measurement/experiment/docs-only PRs write `RED: N/A (<reason>)`.
6. **Contract-touching changes update the contract document in the same PR**
   (`docs/dataplane.md`, the absolute rules, the external API contract).
7. **Architecture, contract, or process decisions — and reversals of past
   decisions — require an ADR** under `docs/adr/`. Sweep-determined numeric
   parameters do not (update `docs/design.md` / `docs/open-issues.md` instead).
8. **English only** in code, comments, commits, issues, PRs, and docs.
9. **Never commit internal keywords.** The guard list is injected via a CI
   repository secret and is not stored in this repository. When unsure
   whether knowledge is public-textbook or internal, ask the owner before
   writing it down.
10. **Merge authority is owner-only.** Agents stop at merge-ready and report
    evidence and residual risks. Normal merge is the default; squash only on
    explicit owner instruction; auto-merge is prohibited. An instruction to
    merge applies to one PR at its current head, once.

## Planning model

- GitHub **Milestones hold the coarse plan**; Issues are added to them
  incrementally as work proceeds. Do not pre-file large issue batches.
- The working unit is an **MVP**: a small increment that runs end-to-end with a
  one-command demo and an acceptance test. Labels: `mvp-candidate`, `icebox`.
- Retrospective learning goes back into `docs/design.md` / `docs/open-issues.md`
  as part of the PR that produced it.

## Starting a task

1. Read the full target Issue, its Milestone, referenced ADRs, and the relevant
   `docs/design.md` sections.
2. Confirm the acceptance criteria are verifiable; ask the owner if not.
3. Follow [docs/development_workflow.md](docs/development_workflow.md).
