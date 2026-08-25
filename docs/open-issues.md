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
| B2 | Effective efficiency **measured**: 3.2% from the stock toolchain against 58.6% on a large square matmul, both on one p150a development board (docs/budget.md). What remains open is not the number but the gap — how much of the twelvefold a hand-written kernel recovers | Track B | measured; gap open |
| B3 | `run_routing()` firing conditions and their jitter impact | Track B | blocked until a link carries traffic; it is an idle-loop property of the Ethernet core |
| B4 | Card-to-card latency/jitter measurement | Track B | blocked until the two boards' link trains (#30); the boards and cabling are in place |
| B5 | TT→host DMA write-ordering guarantee (payload → completion-flag visibility) | Track B | open |
| B6 | Clock calibration between the TT cycle counter and host CLOCK_MONOTONIC | Track B | open |

The Ethernet items above were blocked on a board without ports; that is no
longer the constraint. Two target boards are cabled together, but no link
has trained, and this generation trains from the runtime rather than from a
flashing step (design.md §2) — so each of them now waits on the same thing:
a transfer that actually crosses the wire.

**B2 still matters most**, but the question has changed shape. It was "is
40% real?"; the board answers 58.6% on a shape it likes, so the hardware is
not the doubt. It is now "how much of the eighteenfold penalty this
workload's shapes carry can a kernel take back?" — and that is answered by
writing one, not by measuring again.

---

## Parameters decided by measurement

- Newton-Schulz: precision split (BF16/TF32/FP32), iteration count, choice of
  the initial value X0
- Beamspace: basis design and dimension. **The dimension is no longer a free
  choice on compute grounds alone**: measured on one p150a development board,
  one complex matmul of the Newton-Schulz step — the iteration issues two —
  costs 250 µs on a 32x32 matrix against 627 µs on a 16x16 one, so the larger
  dimension is 2.5x faster in wall-clock while paying eight times the
  arithmetic. A 16x16 matrix fills half of a 32x32 tile and pays for the empty
  half. That reverses under a kernel that packs several small matrices into
  one tile, so the choice is now coupled to how far Track B goes into
  hand-written kernels, and to the sample-support limit that made 16
  attractive in the first place (§9, §11). The two figures are records
  `newton_schulz_L32_b8192` and `newton_schulz_L16_b8192`, bfloat16 in L1, of
  the 2026-08-14 measurement; the catalogue names that dimension L after the
  subaperture, because the shape is the same either way — a beamspace
  covariance of dimension B and a subaperture covariance of dimension L give
  the Newton-Schulz step the same matrix to invert
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
  `tdp_limit` and 300 W as the board limit. The compute measurement did not
  approach either (102 W peak at full clock), so the question waits for a
  workload that does
- **The latency budget does not close across the full range of its own
  stage estimates** (docs/budget.md): the critical path sums to ~27 ms at
  the optimistic end and ~39 ms at the pessimistic end, against a ≤ 30 ms
  target. Decide which stages must hit their optimistic values, or correct
  the estimate that is wrong, once the TT spike gives real numbers

---

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
