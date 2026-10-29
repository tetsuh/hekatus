# ADR-0017: Table-enabled resident controls use the Issue #118 program-interval trust bound

- Status: **Accepted** — 2026-10-29
- Date: 2026-10-29
- Owner decision: accepted by the owner on 2026-10-29

## Context

[ADR-0014](0014-issue118-utc-monotonic-trust-bound.md) limits the UTC/monotonic
program-interval trust bound to loaded Issue #118 records and explicitly
excludes the unloaded `run_bounds` control. It requires a new decision record
for a scope change. Its capture order, serialized precision, scheduling
headroom, and numeric bound remain appropriate; the question here is scope,
not a new tolerance fitted to observations.

Issue #119's table-enabled no-switch control runs the same resident kernel
and table mechanism as the switching rows. It has a captured host program
span even though its condition is `unloaded`: there is resident work but no
separate load kernel. Commit `5a9c4b1` applied the existing
`issue118_program_host_interval` provenance and completion-bracketing sample
wait to table-enabled unloaded runs. The Stage 2 control record collected at
harness `11fa346e2de9ddb4e310bc751a33719542eea21d` obtains its weighted AICLK
through that path, as do the table-enabled switching rows.

This decision records the scope extension explicitly. It does not rewrite
ADR-0014 or any existing measurement record. Under
[ADR-0015](0015-issue121-card-timing-basis.md), these card-tick observations
remain cycle-domain evidence, not a product wall-time guarantee.

## Options considered

- **Do not extend the rule.** Not selected: without the captured program-window
  weighting path, the table control's weighted AICLK is unavailable (`null`),
  leaving the control and switching rows without a common program-interval
  comparison basis. Reverting or reinterpreting placed records is not an
  acceptable way to address the missing scope decision.
- **Extend it only to table-enabled unloaded runs.** Selected: these controls
  and switching rows already capture the same resident-program endpoints and
  can use the existing binding and bounded completion-bracketing wait without
  changing the bound.
- **Extend it to every unloaded run.** Not selected: legacy no-table
  `run_bounds` controls do not record a program span for their power-trace
  interval. Their existing evidence cannot supply the required program-window
  provenance, and fabricating endpoints or reinterpreting it is not allowed.

## Decision

1. A resident run with `condition == "unloaded"` and `tables_enabled(config)`
   uses the same captured host program launch-to-completion UTC interval and
   `issue118_program_host_interval` provenance as a loaded run. Its UTC
   interval is bound to the corresponding host `CLOCK_MONOTONIC` program
   interval, including the existing exact-alias checks.
2. The trust bound remains **`2 × (1,000 + 1,000,000) = 2,002,000 ns`**:
   monotonic is captured before UTC at each endpoint, UTC is serialized at
   1,000 ns resolution, and scheduling headroom is 1,000,000 ns per endpoint.
   The absolute elapsed-time difference must be at most that bound. No
   capture order, precision, headroom, or numeric threshold changes.
3. Before freezing the snapshot, the host uses the existing loaded-run
   bounded wait for a usable sample at or after program completion. Weighted
   AICLK uses the existing duration-weighted, left-held samples over the
   captured program span, with bracketing evidence required rather than
   extrapolated. Missing program-span weighting evidence is rejected.
   `clock.aiclk_mhz` retains its existing public meaning as the in-run maximum;
   it is not substituted for weighted AICLK.
4. Legacy unloaded controls without tables retain their `run_bounds`
   semantics and remain outside this UTC/monotonic comparison as specified
   by ADR-0014. This extension covers table-enabled no-switch controls and
   switching runs, not every unloaded record.
5. Existing record, trace, and frozen-snapshot bytes remain immutable. This
   ADR documents the behavior already used at the measured harness revision;
   it neither changes their provenance nor assigns measurements to a later
   commit. Program-span binding does not replace whole-run coverage or any
   other timing-evidence predicate. The original prefetch attempt and
   readiness-delay row remain non-timing records with their recorded coverage
   failures; they are neither deleted nor promoted by this decision.

## Consequences

- The table-enabled control and switching rows have an explicitly approved
  program-interval comparison basis. Legacy no-table controls keep their
  existing semantics and compatibility.
- The numeric trust bound remains capture-derived and independent of the
  Stage 2 observations. A later change to capture assumptions or scope still
  requires a new decision record.
- Completion bracketing establishes the program-window sample boundary; it
  does not alone prove the separate whole-run power-trace coverage predicate.
  In particular, the 2,002,000 ns clock-binding bound is not the two-second
  sampler coverage bound. This ADR does not relax either check or repair the
  Issue #157 capture-order race.
- Kernels, wrapper, validator, and measurement artifacts need no changes for
  this documentation decision. The measured harness ancestry is retained;
  no rebase, merge, or retrospective record edit is required.
- ADR-0015 remains binding: AICLK provenance supports cycle-domain
  interpretation, not a product wall-clock timing claim or card-timer pacing.

## Status history

- 2026-10-29: Drafted for Issue #119 / PR #156 in response to review finding
  `CONV-156-ADR0014-SCOPE`.
- 2026-10-29: Accepted by the owner for PR #156; the table-enabled unloaded
  scope extension is binding without changing existing measurement records.
