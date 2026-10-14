# ADR-0014: Issue #118 UTC/monotonic program-interval trust bound

- Status: **Accepted** — 2026-10-14
- Date: 2026-10-14
- Owner decision: accepted by the owner on 2026-10-14

## Context

Issue #118 loaded records bind the UTC program interval used by the power trace
to the host `CLOCK_MONOTONIC` program interval. `run_resident` captures each
interval endpoint from monotonic time first and UTC second. The monotonic
values remain in nanoseconds, while the UTC endpoint is serialized with
1,000 ns resolution. A scheduling interruption between the two reads can add
another difference, so the validity rule needs a declared capture bound rather
than a bound fitted to the committed observations.

The rule applies to loaded Issue #118 validation, where
`load_interval_source` is `issue118_program_host_interval`. It does not apply
to the unloaded `run_bounds` control. The committed loaded records differ by
647 ns for Newton-Schulz and 1,861 ns for DRAM; both differences have ample
margin below a capture-derived bound.

## Options considered

- **Do not bind UTC and monotonic intervals.** Rejected: an unloaded or
  mismatched interval could be accepted as timing evidence without a check
  that the two clocks described the same program span.
- **Fit the bound to the committed observations.** Rejected: 647 ns and
  1,861 ns are evidence that the records pass, not a normative validity rule.
- **Use a bound derived from endpoint capture and serialization.** Chosen:
  this records the known capture order, serialized precision, and scheduling
  headroom independently of the observations.

## Decision

1. `run_resident` captures each endpoint in the order monotonic first, then
   UTC: it reads `CLOCK_MONOTONIC` and then captures the UTC timestamp.
2. The serialized UTC endpoint resolution is **1,000 ns**. The validator
   reserves **1,000,000 ns** of scheduling headroom per endpoint between the
   monotonic and UTC reads.
3. The UTC/monotonic trust bound is
   **`2 × (1,000 + 1,000,000) = 2,002,000 ns`**. A loaded record passes this
   binding when the absolute elapsed-time difference is at most that bound;
   a larger difference is rejected.
4. The bound applies only to loaded Issue #118 validation identified by
   `load_interval_source == issue118_program_host_interval`. It does not
   apply to the unloaded `run_bounds` control.
5. The committed loaded records pass with substantial margin: the
   Newton-Schulz difference is **647 ns**, and the DRAM difference is
   **1,861 ns**. These observations confirm margin; they do not set or tune
   the bound.

## Consequences

- Loaded Issue #118 records have a deterministic, capture-derived validity
  rule for the UTC/monotonic interval binding.
- The unloaded `run_bounds` control retains its existing interval semantics
  and is not judged against the UTC/monotonic trust bound.
- The committed measurement artifacts and power traces remain unchanged; both
  committed loaded records remain valid under the rule.
- Changing the capture order, serialized precision, scheduling headroom, or
  scope of this validity rule requires a new decision record rather than
  silently fitting the bound to later observations.

## Status history

- 2026-10-14: Proposed in pull request #130 as the Issue #110 AC3
  capture-derived tolerance rationale; it records the existing rule and does
  not introduce observation-based tuning.
- 2026-10-14: Accepted by the owner in pull request #130.
