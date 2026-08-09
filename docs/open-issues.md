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
| B1 | ERISC custom-firmware development procedure; whether the deprecated or the fabric-based EDM is the current recommendation | Track B | open |
| B2 | Effective-efficiency measurement (estimates assume 40%; the foundation of every card-count estimate) | Track B | open |
| B3 | `run_routing()` firing conditions and their jitter impact | Track B | open |
| B4 | Card-to-card latency/jitter measurement | Track B | open |
| B5 | TT→host DMA write-ordering guarantee (payload → completion-flag visibility) | Track B | open |
| B6 | Clock calibration between the TT cycle counter and host CLOCK_MONOTONIC | Track B | open |

**B2 matters most.** The entire "fits on one card" claim rests on it.

---

## Parameters decided by measurement

- Newton-Schulz: precision split (BF16/TF32/FP32), iteration count, choice of
  the initial value X0
- Beamspace: basis design and dimension
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
- License: Apache-2.0
- An IP-landscape review is a productization-gate item
