# ADR-0015: Issue #121 card timing basis

- Status: **Accepted** — 2026-10-17
- Date: 2026-10-17
- Owner decision: accepted by the owner on 2026-10-17

## Context

Issue #121 required a timing basis that remains valid when sustained compute
load changes the card clock. Q1 provided the contrast: the unloaded control's
card interval converted to 499.998999987 s while its host
`CLOCK_MONOTONIC` program interval was 500.320116957 s; the loaded synthetic
Newton-Schulz interval converted to 706.520574308 s while the host interval was
707.446161353 s. A card-tick interval therefore carries cycle-domain evidence,
but its wall-time interpretation depends on the loaded AICLK.

Q4's bounded audit of the fixed tt-metal v0.75.0 source found no documented
AICLK-independent real-time counter. The audit examined the wall-clock,
watchdog, timestamp, and performance-counter paths; it did not make a claim
about undocumented hardware.

Q2 then showed that the ADR-0013 synthetic Newton-Schulz load lowers sustained
AICLK by about 28% from the 1,350 MHz unloaded value. That result is a
synthetic-load reference, not a full-enodia measurement and not a limiter
identification. The full-enodia clock and limiter questions remain separate
follow-ups.

## Options considered

- **Use card ticks as the product time basis.** Rejected: sustained load
  distorts their wall-time interpretation, and Q4 found no documented
  AICLK-independent counter to justify treating them as real time.
- **Pace product frames with a host clock.** Rejected: product frame timing is
  driven by data arrival and completion, not by an independent host pacing
  loop inserted into the card data path.
- **Use data arrival and completion for frame timing, and host monotonic time
  for supervision.** Chosen: it separates the product data path from
  load-distorted cycle evidence while giving supervisory deadlines a stated
  host time basis.

## Decision

1. The **card tick is not the product time basis**. Tick records are retained
   as cycle-domain evidence only and must not be treated as wall-clock time
   without a separately documented basis.
2. **Frame timing follows data arrival and completion**, not a card timer or a
   card-tick-derived pacing period.
3. **Supervisory timeouts use host `CLOCK_MONOTONIC`**, not card ticks.
4. If a documented AICLK-independent card real-time counter is found, these
   rules may be revisited in a new decision record; undocumented assumptions
   do not change this ADR.

## Consequences

- Sustained load can distort tick-based timing in real time, so a card-tick
  span cannot by itself establish a product wall-clock deadline or latency
  claim.
- Card-tick records remain valid cycle-domain evidence for comparisons when
  their AICLK and interval provenance are retained; they are not product-time
  records.
- The product frame path follows data arrival and completion rather than
  using a load-distorted card timer for pacing.
- Host `CLOCK_MONOTONIC` supplies the basis for supervisory timeout decisions.
- A documented AICLK-independent counter would be a reason to revisit the
  timing basis. The Q2 synthetic result does not settle full-enodia sustained
  AICLK (#136) or limiter causality (#137).

## Status history

- 2026-10-17: Accepted by the owner in this pull request for Issue #121 after
  the Q1/Q4 evidence and Q2 synthetic-load record were documented; the timing
  basis is binding without settling full-enodia AICLK or limiter causality.
