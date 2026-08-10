# ADR-0001: Development process — ticket-driven MVPs with lightweight governance

- Status: **Proposed** (accepted when the bootstrap PR merges)
- Date: 2026-08-09
- Owner decision: yes

## Context

This repository is developed by one human owner with AI coding agents (multiple
models) doing most of the implementation. The domain is measurement-driven: a
large share of design parameters is explicitly deferred to sweeps and
measurements, so any long-range plan is expected to be invalidated repeatedly.

The owner's prior OSS projects, [sitos](https://github.com/tetsuh/sitos) and
[sitometron](https://github.com/tetsuh/sitometron), adopted a heavier
governance stack (requirement-ID traceability, contract registry, gate issues,
frozen issue checklists, structured RED evidence, large pre-filed issue sets
mapped 1:1 to design documents). Experience there showed two failure modes:
plans required constant correction, and the governance surface itself became a
maintenance burden that competed with product work.

At the same time, agent-driven implementation has a characteristic failure
mode that governance must address: **tautological tests** — the same hand
writes implementation and test, embedding the same misunderstanding in both,
so the test passes while verifying nothing.

## Options considered

- **A. TiDD core only**: No Ticket No Commit; 1 issue = 1 branch = 1 PR;
  Conventional Commits; green main; owner-only merge; tests mandatory;
  contract-document synchronization. Cheapest (≈2–5 min/ticket overhead) but
  leaves the tautological-test risk to review alone.
- **B. Full governance** (as in the prior projects): strongest agent control
  and built-in traceability, but ≈30–60 min/ticket overhead plus days of
  upfront governance documents; poor fit for sweep/measurement tickets
  (constant N/A bookkeeping); the complexity failure mode is already
  demonstrated in the prior projects.
- **C. A plus a RED one-liner**: A, plus one line per implementation PR
  recording the command and the failing test observed before implementation.
  Directly targets the tautological-test risk at ≈1 line of cost.

The recommendation evolved during the decision: A was recommended first; the
threat analysis (tautological tests as the most frequent real failure of
agent-written code) shifted the recommendation to C. This evolution is
recorded deliberately — see Consequences.

## Decision

Adopt **C**, with these owner amendments:

1. **Normal merge is the default**; squash only on explicit owner instruction.
   One commit per PR is too coarse for retrospection; `main` history should
   preserve meaningful step-level commits (`git log --first-parent` gives the
   PR-level view).
2. **Milestones hold the coarse plan**; issues are added incrementally. Large
   issue batches are not pre-filed.
3. **`AGENTS.md` is the model-neutral agent entry point** (multiple AI models
   participate).
4. **English only** across the repository, matching the sibling projects.
5. ADRs (this format) record architecture/contract/process decisions and
   their reversals; sweep-determined numeric parameters are recorded in the
   design documents instead.

Full rules: `docs/development_workflow.md`.

## Consequences

- Per-ticket overhead stays in the minutes range; exploration and measurement
  tickets are not taxed by traceability bookkeeping.
- The tautological-test risk is addressed by the RED one-liner; scope drift is
  addressed by TiDD discipline; contract/document drift by same-PR
  synchronization.
- **Reserved for the productization gate**: when the specification freezes,
  traceability governance of grade B (requirement IDs, contract registry,
  gates) is introduced. Introducing it earlier would freeze requirements that
  are explicitly measurement-pending.
- Decision evolution is preserved in ADRs by design: reversals create a new
  ADR with `Supersedes:` links rather than editing history. This ADR itself
  records one such evolution (A → C) as a worked example.

## Status history

- 2026-08-09: Proposed (process already agreed with the owner in session).
- 2026-08-10: Amended by ADR-0003, which sets the commit header width, makes
  the scope requirement explicit, and exempts platform-generated integration
  commits. The rest of this decision stands.
