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
| B2 | Effective efficiency **measured**: 3.2% from the stock toolchain against 58.5% on a large square matmul, both on one p150a development board (docs/budget.md). What remains open is not the number but the gap — how much of the twelvefold a hand-written kernel recovers | Track B | measured; gap open |
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
40% real?"; the board answers 58.5% on a shape it likes, so the hardware is
not the doubt. It is now "how much of the eighteenfold penalty this
workload's shapes carry can a kernel take back?" — and that is answered by
writing one, not by measuring again.

---

## Parameters decided by measurement

- Newton-Schulz: precision split (BF16/TF32/FP32), iteration count, choice of
  the initial value X0
- Beamspace: basis design and dimension. **The dimension is no longer a free
  choice on compute grounds alone**: measured on one p150a development
  board,
  one complex matmul of the Newton-Schulz step — the iteration issues two —
  costs 247 µs at B=32 against 624 µs at B=16, so B=32 is
  2.5x faster in wall-clock while paying eight times the arithmetic — B=16
  fills half of a 32x32 tile and pays for the empty half. That reverses under
  a kernel that packs several small matrices into one tile, so the choice is
  now coupled to how far Track B goes into hand-written kernels, and to the
  sample-support limit that made 16 attractive in the first place (§9, §11)
- Transmit compounding: window width, apodization, contributing-transmit
  truncation
- Decimation ratio and interpolation tap count (are 4 taps enough?)
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
  approach either (93 W peak at full clock), so the question waits for a
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
