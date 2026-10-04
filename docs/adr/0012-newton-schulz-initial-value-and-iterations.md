# ADR-0012: Newton-Schulz initial value and iteration count

- Status: **Proposed** — 2026-10-03
- Date: 2026-10-03
- Owner decision: yes

## Context

The Newton-Schulz inverse is part of the fixed-work MV path. Its initial value
and iteration count therefore determine both numerical behavior and the cost
of the real-time path. Issue #85 Stage 1 measured the two initial-value choices
and fixed counts from 8 through 16 on condition numbers 10, 30, 100, and 300,
with aperture sizes 16, 32, and 64, in float32 and float64. The record is
`docs/measurements/2026-10-03-host-newton-schulz-reference-sweep.json`.

The Stage 2 decision is provisional until the M5 image-quality confirmation.
The execution count remains fixed: the real-time path must not use a
convergence-gated loop.

## Options considered

- **`X0 = R^H / (||R||_1 ||R||_infinity)`.** For a Hermitian positive-definite
  `R`, the eigenvalues of `R X0` are `lambda^2 / c`. This squares the condition
  number and delays the quadratic-convergence regime, so it is rejected.
- **`X0 = I / ||R||_infinity`.** The eigenvalues of `R X0` are
  `lambda / ||R||_infinity`, so the condition number is not squared. This is
  the selected initial value.
- **A smaller fixed count, including `N = 10`.** The Stage 1 record shows
  that `N = 10` covers only through `kappa <= 100`; it does not cover the
  `kappa = 300` range used for the decision.
- **A convergence-gated variable loop.** The real-time execution rule requires
  a fixed amount of work, so a data-dependent exit is rejected.

## Decision

Use `X0 = I / ||R||_infinity` and a fixed `N = 12` for the Newton-Schulz
reference and TT kernel. The Stage 1 record's `kappa = 300`, `L = 32`,
float32, `N = 12` point has inverse relative error at or below `5e-4` and MV
weight direction cosine deficit `8.33e-8`, below `1e-7`. The count is selected
from inverse and MV-direction evidence; beam-pattern evidence remains recorded
but is not the Stage 1 gate.

This is a provisional decision. M5 image-quality confirmation is a condition
of the final decision. Until that confirmation, the implementation carries
out exactly twelve iterations and does not add a convergence test.

## Consequences

- The independent host reference and the TT host preparation use the selected
  initial value and fixed count.
- The compute kernel's compile-time iteration assertion follows `N = 12`.
- The Newton-Schulz iteration-dependent cost is `12 / 8 = 1.5` times the
  previous eight-iteration planning figures; the corresponding design and
  budget estimates are updated on this branch.
- Passing `R` in BF16 introduces a condition-number-dependent perturbation:
  the 2026-10-04 diagnostics report, at κ=100, approximately `2e-2` inverse
  error, `2.5e-4` MV-direction deficit, and `1.4e-2` beam-pattern error, and
  at κ=300 approximately `5.9e-2`, `3.2e-3`, and `5.9e-2`; the device kernel
  is approximately `3.7e-3` from the BF16-R reference.
- The representation of `R` remains open: a follow-up Issue will compare an
  FP32-R variant with the BF16 variant in the same run before deciding the
  production format.
- Image-quality confirmation remains open. A later decision record or the
  decision-carrying pull request must record the M5 outcome before this
  proposal can become accepted.

## Status history

- 2026-10-03: Proposed for Issue #85 Stage 2; the owner decision and Stage 1
  evidence select `X0 = I / ||R||_infinity` and `N = 12`, subject to M5
  confirmation.
