# open-issues.md

Extract of design.md §17 plus a working tracker.

**Lifecycle**: when an item settles, its outcome is recorded in
`design.md` — that document is the authority — and the item moves to the
`Settled` section here, where it stays only as a short-term reminder of
what recently changed. **Entries are cleared from that section when the
milestone during which they settled closes**, by the owner, as part of
closing it; the record in `design.md` is what persists.

---

## Blocker candidates (kill early)

| # | Item | Who | State |
|---|---|---|---|
| B1 | ERISC custom-firmware development procedure; whether the deprecated or the fabric-based EDM is the current recommendation | Track B | blocked until a chip-to-chip transfer runs (#30) |
| B2 | Effective efficiency is measured for both stock and hand-written Newton-Schulz rows: stock reaches 3.024% on its best BF16 row, while Issue #63 reaches 15.6% at L=32 and 3.24% at packed L=16. The result is between 3.2% and 30%; the residual planning gap remains open | Track B | measured; residual gap open |
| B3 | `run_routing()` firing conditions and their jitter impact | Track B | blocked until a link carries traffic; it is an idle-loop property of the Ethernet core |
| B4 | Card-to-card latency/jitter measurement | Track B | blocked until the two boards' link trains (#30); the boards and cabling are in place |
| B5 | TT→host DMA write-ordering guarantee (payload → completion-flag visibility) | Track B | open |
| B6 | Clock calibration between the TT cycle counter and host CLOCK_MONOTONIC | Track B | open |

The Ethernet items above were blocked on a board without ports; that is no
longer the constraint. Two target boards are cabled together, but no link
has trained, and this generation trains from the runtime rather than from a
flashing step (design.md §2) — so each of them now waits on the same thing:
a transfer that actually crosses the wire.

**B2 still matters most**, but the question has changed shape. The #65
0.75.0 full sweep found 190 successful rows and 94 failed rows out of 284.
Explicit stock configs help broad shapes: front-end FIR width 32 in L1 rises
from 4.3389 TFLOPS (1.307%) by default to 15.2871 (4.605%), and beamspace
B=16, 128 channels, 65536 pixels rises from 3.5508 (1.070%) to 7.1949
(2.167%). Newton-Schulz has a stricter boundary: no explicit config beats the
best default where the default L1 row succeeds, while DRAM-only large-batch
reuse gains 7.7%, 10.0%, and about 35% for L=16, L=32, and L=64 respectively;
the largest is only 0.4251 TFLOPS (0.128%). The stock inverse denominator is
still 3.024% at L=64, batch 1024, default L1, but the Issue #63 hand-written
rows now measure 51.92 TFLOPS / 15.6% at L=32 and 10.77 TFLOPS / 3.24% at
packed L=16. The measured result is between 3.2% and 30%, so the 30%
efficiency target is not established. Applying the measured L=32 workload
efficiency (15.6%, about 52 TFLOPS per card) to the roughly 100 TFLOPS 1D
all-mode estimate gives about 2 cards. This is an extrapolation from the
Newton-Schulz workload, not a full-system or all-mode benchmark.

The three Scope 5 conclusions are: (1) L=32 is 5.7x the same-run stock best
but remains below the 30% target; (2) packed L=16 is 14.5x stock and faster in
wall-clock than L=32 only as a cost/operation-volume comparison for the
diagonal fallback, because it uses fewer logical dimensions and less work;
this is not a beamspace-dimension reduction versus an MV image-quality
comparison. The L=16 record is
`docs/measurements/2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json`;
(3) fixed handoff/queue overhead is the leading measured explanation, while
math, unpack, and reader were not established as causal bottlenecks. The
steady record explicitly declines a unique bottleneck; its RISC windows
overlap and compute retains unclassified cycles. The optimization, unpack,
and matrix-block records correlate the handoff/queue interpretation but do
not prove causality.

The 0.70.1 default-only comparison is separate, with 59 successes and 9
failures out of 68. Its 4096-square reference is 58.687% versus 58.410% in
0.75.0, and NS L=32 batch 8192 L1 is 3.026% versus 2.992%; small
dispatch-bound beamspace p4096 rows differ by up to 0.872 percentage points,
so this is bounded evidence rather than a universal toolchain claim. The
records also show that the August 3.227% L=64 batch 8192 L1 row does not
reproduce under the current all-L1 output placement: both September records
fail it with allocator OOM. The old harness placed only inputs in L1 and left
output placement at the operation default; the current harness explicitly
places output in L1. The identical failure in both current images is not
evidence of a toolchain regression. The exact stock records are
`docs/measurements/2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0.json`,
its targeted four-row superseder
`docs/measurements/2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0.json`,
its targeted two-row unbatched superseder
`docs/measurements/2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0.json`,
`docs/measurements/2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1.json`,
and the historical source is
`docs/measurements/2026-08-14-p150a-effective-efficiency.json`.
The hand-written records are
`docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000.json`,
`docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000.json`,
and `docs/measurements/2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json`.

---

## Parameters decided by measurement

- Newton-Schulz: precision split (BF16/TF32/FP32), iteration count, choice of
  the initial value X0
- Newton-Schulz state precision (BF16 or FP32). The Issue #94 / PR #95 record
  (`docs/measurements/2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck.json`)
  shows BF16 state with FP32 DEST at 0.0041–0.0044 against the BF16-rounded-R
  reference generated by `enodia/tt/bench/run_matmul.py:_issue94_fixed_reference`
  and comparable to FP32 state against the true inverse, consistent with BF16
  quantization of R dominating the error. The record's fixed-N=12 reference
  uses BF16-rounded R with X0 from original R; it does not call
  `enodia/tt/bench/newton_schulz_reference.py`. Throughput is +6.0% at L=32
  and +4.0% at L=16. BF16 DEST exceeds the 0.01 correctness threshold. A
  default precision change is pending the owner's decision; the combination
  with #92 double buffering was not measured.
- Beamspace: basis design and dimension. **The dimension is no longer a free
  choice on compute grounds alone, and the planning claim has changed.** The
  stock catalogue made 32x32 faster in wall-clock than 16x16 because a 16x16
  matrix paid for empty tile area. The packed Issue #63 kernel reverses that
  ordering: L=16 `custom_block4` reaches 10.77 TFLOPS (3.24%) and has a
  0.4019 ms per-iteration median, versus L=32's 0.6655 ms in the same-shape
  batch records. Use L=16 for beamspace cost planning when sample support
  permits; this wall-clock result is only the diagonal fallback's
  cost/operation-volume comparison, not a beamspace-dimension or MV
  image-quality comparison. Do not append the old stock ordering as if it
  still governed the kernel. The L=16 record is
  `docs/measurements/2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json`;
  the L=32 records are
  `docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000.json`
  and
  `docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000.json`.
  The catalogue names that dimension L after the subaperture, because the
  shape is the same either way — a beamspace covariance of dimension B and a
  subaperture covariance of dimension L give the Newton-Schulz step the same
  matrix to invert
- Issue #63 bottleneck evidence. The steady
  cycle-counter record
  `docs/measurements/2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady.json`
  separates queue waits but establishes no unique causal bottleneck because
  RISC windows overlap and compute has unclassified residuals. The optimization
  record
  `docs/measurements/2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000.json`
  shows only small changes from `fuse_s` and `batch_reads`; the unpack record
  `docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000.json`
  does not establish unpack as causal; and the matrix-block record
  `docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000.json`
  correlates block-4's gain with fewer fixed queue/handoff turns. Thus fixed
  handoff/queue overhead is the leading measured explanation, not proof; math,
  unpack, and reader remain unestablished as causal bottlenecks.
- Transmit compounding: window width, apodization, contributing-transmit
  truncation
- Decimation ratio and interpolation tap count (are 4 taps enough?). The
  kernel is fixed — Lagrange cubic, design.md §5 — and the axial-PSF
  consequence is now **measured** (#6, design.md §5 / §17; record
  `docs/measurements/2026-08-23-host-iq-path-vs-golden.json`) on the point-
  scatterer phantom with the named `linear-5mhz` profile, whose bandwidth is
  **provisional** (§4): D=8 broadens the axial width by +33–42 % at −6 dB
  (0.257–0.276 vs 0.194 mm) and +23 / +25 / +61 % at −40 dB (0.614 /
  0.626 / 0.804 vs 0.500–0.501 mm, per scatterer at 15 / 25 / 40 mm) and
  costs 21 % at L0 checkpoint 2; D=4 +4 % at −6 dB (0.201–0.202 mm), −2 %
  at −40 dB (0.490–0.492 mm) and 2.4 %. The artifacts say
  "provisional" and are rerun if the profile's value or provenance
  changes. Still open: whether D=8's width is acceptable for the product
  (an image-quality judgement), the same on a sourced bandwidth, and the
  13 MHz case. **#10 owns the 13 MHz profile** — until it lands, 13 MHz
  figures are the synthetic 80% envelope — and triggers the 13 MHz rerun of
  the §5 sweep, the §15 profile reconciliation, and this measurement
- Diagonal loading (2D may need more than 1D because of μBF grating lobes)
- Core allocation (front-end / beamforming / inference)
- Group-batch size and its boundary artifacts
- Aberration-estimation update rate and spatial smoothing extent
- TF32 availability on the target hardware

---

## Investigation items

- Tensix dest-register accumulation precision and read-out behavior
- AFE anti-aliasing characteristics (does the 13 MHz configuration suppress
  everything above 20 MHz?)
- Actual TGC behavior of the target front end (discontinuities, gain-step
  granularity). **With MLA, the depth-to-TGC correspondence shifts per
  scanline, which can create inconsistencies under transmit compounding**
- Which power limit the board enforces — the firmware reports 150 W as
  `tdp_limit` and 300 W as the board limit. The historical 102 W comparison
  is from `docs/measurements/2026-08-14-p150a-effective-efficiency-power.csv`
  (the 2026-08-14 effective-efficiency trace); the 110 W result is from the
  2026-09-20 ttnn 0.75.0 full-sweep trace. Neither bound was reached, so the
  question waits for a workload that does
- **The latency budget does not close across the full range of its own
  stage estimates** (docs/budget.md): the critical path sums to ~27 ms at
  the optimistic end and ~39 ms at the pessimistic end, against a ≤ 30 ms
  target. Decide which stages must hit their optimistic values, or correct
  the estimate that is wrong, once the TT spike gives real numbers

---

## Gated (held in an issue; not active)

- **Local sound-speed map — §8 model (b) re-hearing (#44).** Rejected in
  design.md §8; #43 records which rejection reasons are bound to the
  placement and which to the role. The question re-activates only when both
  of #44's conditions hold: the hardware-neutral estimator (#42) has passed
  L0, and a placement exists in this repository on which per-pixel path
  integrals are cheap. First product on activation is an ADR, and its
  evidence must include the implicit-table hypothesis §8 marks as
  unverified. Until then this line exists so the question is not re-derived
  from scratch

## Needs external input (open)

- Concrete clinical use for organ recognition (determines model scale and
  latency requirements)
- Exact element counts and pitches of the target probes (a placeholder set —
  five 1D probes + one 2D — is in use; kinds and frequency bands are agreed,
  and exact geometry swaps in later as profile data)
- Effective two-way pulse bandwidth of the target probes, with provenance
  (manufacturer data or a measured pulse response). `linear-5mhz` runs on a
  provisional 0.7 with no source (design.md §4, ADR-0008); a sourced value
  replaces it through a reviewed profile update

## Settled (recorded; reflected in design.md)

- MLA {2, 4} fixed; 8 is a color-flow experiment slot. Velocity-bias
  verification uses the flow phantom
- Depth/focus changes assume continuous knob operation; fast re-derivation
  with coalescing control
- Elastography: strain first; transmit-type tags form an open set
- The enodia API is a physical-quantity schema + config IDs; interpreting
  FPGA-facing data was rejected (the round-trip converter is test-only)
- Effective efficiency is measured, and the measurement environment is
  recorded with it (design.md §2, docs/measurements/)
- License: Apache-2.0
- An IP-landscape review is a productization-gate item
