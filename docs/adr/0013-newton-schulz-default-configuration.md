# ADR-0013: Newton-Schulz kernel default configuration and contract

- Status: **Accepted** — 2026-10-05
- Date: 2026-10-05
- Owner decision: yes (Issue #100)

## Context

Issue #100 selects the measured Newton-Schulz configuration for the host API
and benchmark runner. The configuration is recorded in `docs/design.md` and is
implemented by the Newton-Schulz preparation and runner code under
`enodia/tt/bench/`. The default must name both the selected resources and the
boundary behavior, so a caller can distinguish an explicit legacy comparison
from the production default.

The Issue #94 record,
`docs/measurements/2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck.json`,
records that BF16 quantization of R already dominates the solver error against
the true inverse. Carrying the solver state in FP32 would not remove that
input-representation bound. The representation of R itself remains unresolved
in Issue #88.

## Options considered

- **Keep the previous default configuration.** Rejected: Issue #100's measured
  configuration is the selected host and benchmark default.
- **Select the measured configuration and silently reduce it when resources are
  insufficient.** Rejected: silent changes to block size, placement, or
  synchronization make the selected configuration unknowable and hide a
  contract violation.
- **Select the measured configuration, fail fast at the resource boundary, and
  preserve previous choices as explicit arguments.** Chosen: it makes the
  default reproducible while retaining explicit legacy comparisons.

## Decision

1. The Newton-Schulz default is **BF16 state**, **FP32 DEST**, **block8**,
   **double-buffer**, **full-sync**, **fused S**, and **HiFi3**. The R and X0
   inputs are in **L1**, and the output is in **DRAM** (`r_memory=l1`,
   `x0_memory=l1`, and `output_memory=dram`). In code these choices are
   `variant=bf16`, `fp32_dest_acc_en=true`, `matrix_block=8`,
   `double_buffer=true`, `dst_full_sync_en=true`, `fuse_s=true`, and
   `math_fidelity=HiFi3`.
2. The default is **fail-fast with no silent fallback**. `prepare` checks the
   selected DEST capacity and then the L1 preflight before device tensor
   allocation. A DEST or L1 capacity failure raises `ValueError` and does not
   change the selected block, placement, or synchronization mode.
3. Previous configurations remain selectable through explicit arguments. The
   default does not remove the legacy state, block, fidelity, fusion,
   synchronization, or placement choices used by comparisons and sweeps.
4. This decision does not settle the R format. That question remains open in
   Issue #88.

## Consequences

- New host-API and benchmark-runner calls receive one named configuration
  whose resource requirements and failure behavior are defined together.
- A caller that needs another configuration must select it explicitly; a
  resource failure is visible to the caller instead of being converted into a
  different measurement.
- BF16 state is consistent with the Issue #94 input-precision finding, while
  the unresolved R-format question remains available for the explicit follow-up
  in Issue #88.

## Status history

- 2026-10-05: Accepted in pull request #101 for Issue #100, recording the
  owner decision for the Newton-Schulz default configuration and fail-fast
  contract.
