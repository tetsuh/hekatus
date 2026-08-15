# budget.md

Working extract of the estimate tables in design.md §10. When a number
changes here, fix design.md too.

**Assumptions for every table**: 30 fps and 2048 depth points unless stated.

**Two capacity bases appear in this document; each table names the one it
uses.**

| Basis | Value | Used by |
|---|---|---|
| Theoretical peak (BF16) | 332 TFLOPS | the "1-card %" columns |
| Usable per card (peak × 40% effective) | 133 TFLOPS | the "cards" columns |

A percentage from one table cannot be combined with a card count from
another without converting: 1-card % × 2.5 gives the share of usable
capacity.

> **The 40% is now read as a target for hand-written kernels rather than an
> expectation of the toolchain.** This document previously called it an
> unverified assumption and said the "fits on one card" claim would collapse
> if it halved. Effective efficiency has since been measured on a p150a: the
> stock toolchain delivers **3.2%** on this workload's shapes — see below.
> Reading the 40% as a target is therefore a change of claim, not a
> restatement of what was meant before, and it is justified only because
> design.md assumes hand-written kernels throughout. Every card count here
> describes what the design aims at, with a factor of twelve still to close.

---

## Measured efficiency (p150a, 2026-08-14)

Measured with `enodia/tt/bench`, one development board, bfloat16, against
the 332 TFLOPS peak above. Raw results and the environment that produced
them — board, firmware, driver, toolchain digest, and the harness revision:
`docs/measurements/2026-08-14-p150a-effective-efficiency.json`.

| Shape | DRAM | L1 | best, as % of peak |
|---|---|---|---|
| Newton-Schulz L=64, batch 8192 | 0.31 | **10.71** | **3.2%** |
| Newton-Schulz L=32, batch 8192 | 0.07 | 8.60 | 2.6% |
| Beamspace B=16, 256 ch, 65536 px | 4.98 | 4.38 | 1.5% |
| Front-end FIR, output width 32 | 7.16 | 4.39 | 2.2% |
| *Reference: 4096³ square matmul* | *194.69* | *10.95* | *58.6%* |

Each figure is the best of three timed blocks, and the record keeps all
three. Behind the rows above the three agree to within 1.4% — worst on the
L=32 L1 line at 1.37%, and closest on the two figures the argument rests on,
0.14% for Newton-Schulz L=64 and 0.24% for the reference — so the
differences between shapes are the shapes and not the noise.

Not every record is that steady: eight of the sixty-one successful ones
spread by more than 1.4%, and all eight are the catalogue's smallest shapes
(batch 1024, or 4096 pixels), where one block costs tens of microseconds and
per-dispatch overhead dominates what it measures. The largest, a factor of
3.6, falls from the first block to the last and is a warm-up cost; the rest
scatter in both directions. No figure quoted here comes from those shapes,
and a reported figure is the best of three rather than their mean.

Three things follow, and the third is the one that matters.

**The silicon reaches 58.6% on a shape it likes.** The 40% assumption was
never unreasonable *for the hardware*; a large square matmul beats it. So
the deficit is not silicon, and not the measurement.

**On-chip residency is worth 34x.** The same Newton-Schulz shape moves from
0.31 to 10.71 TFLOPS between DRAM and L1 — a factor of 34.0 — which is design.md §2's "fitting
on-chip is the paramount design concern" as a number. The L1 configurations
that fail — batch 65536 at every L, and L=64/float32 at batch 8192 — map the
on-chip budget by where they stop.

**What the stock toolchain gives for free is 3.2%, and the gap to 40% is the
work.** The representative shapes reach one eighteenth of what the flattering
one does, because they are small, thin, or both (design.md §10). Closing that
is what a hand-written kernel is for: packing small matrices into full tiles,
fusing the four real matmuls of a complex one, keeping R resident across the
iteration, and a resident kernel that pays no per-operation dispatch. How
much of the twelvefold is recoverable is now the central open question of
Track B, and it is a question about kernels rather than about the board.

**Power and clock did not bind.** Across the run the board peaked at 102 W at
its full 1350 MHz and 77.7 °C, so neither the 150 W nor the 300 W reading of
the firmware limit constrains this workload — a figure that itself says the
engines are idle much of the time.

---

## Governing law

MV cost is dominated by the inverse, `L^3` (L = subaperture size). Growing
the element count scales `L ∝ N` and the scanline count `∝ N`, so the
**total goes as N^4**. 64 → 128 receive channels is a 16× increase.

---

## By method (64 receive channels, 30 fps) — basis: theoretical peak

| Method | TFLOPS | 1-card % |
|---|---|---|
| DAS | 0.004 | ~0% |
| CF / PCF / F-DMAS | 0.015 | ~0% |
| SLSC | 1 | 0.3% |
| MV: R formation only (sliding update) | 2 | 0.6% |
| MV: with Newton-Schulz inverse | 33 | ~10% |
| ESBMV (eigendecomposition) | 100–170 | 30–50% |

## By configuration — basis: usable per card (133 TFLOPS)

| Configuration | Recv ch | L | TFLOPS | Cards |
|---|---|---|---|---|
| 128 elements / 64 ch receive | 64 | 32 | 35 | 1 (26% used) |
| 256 elements / 128 ch receive | 128 | 64 | 560 | 5 (4.2 rounded up) |
| 256 elements + beamspace (B=16) | 128 | 16 | 19 | 1 (14% used) |
| post-μBF 256 ch, volume | 256 | 128 | 1,100 | 9 |
| post-μBF 256 ch + beamspace | 256 | 16 | 37 | 1 (28% used) |
| 2D fully digital 4096 ch full MV | 4096 | 2048 | ~7.2e7 | impossible |

The last row follows the N⁴ law from the 256-channel volume row
(1,100 × 16⁴ ≈ 7.2e7). An earlier revision carried 1.85e8 here, which did
not reconcile with the law stated above; the conclusion is unchanged.

## Target configuration (1D 256 elements / 128 ch receive + post-μBF 2D) — basis: theoretical peak

| Mode | Beamformer | TFLOPS | 1-card % |
|---|---|---|---|
| 1D B-mode | DAS + phase-screen correction | ~5 | 2% |
| 1D B-mode | + SLSC / CF / DMAS | ~40 | 12% |
| 1D B-mode | + beamspace MV | ~25 | 8% |
| 1D color flow | per-channel wall filter + MV | ~30 | 9% |
| 2D volume | beamspace MV | ~37 | 11% |

**Everything for 1D running at once is ~100 TFLOPS: about 30% of theoretical
peak, or about 75% of one card's usable capacity.** The headroom claim is
"70% of peak remains" — on the usable basis it is roughly 25%, which is what
lampas has to fit into. State which basis is meant whenever the claim is
quoted.

---

## Transmit-compounding multiplier

Receive-beamforming work is "formed scanlines × contributing transmits per
scanline" and **does not depend on the MLA count**. For a 13 MHz linear
probe with 434 scanlines:

| Configuration | line-formations / frame | vs no compounding |
|---|---|---|
| No compounding | 434 | 1.0 |
| Compounding ±4 | 3,906 | 9.0 |

**The 9× applies only to the delay-and-sum part.** In the compound-then-MV
arrangement, R formation and Newton-Schulz run once, after compounding.
DAS is ~0% to begin with, so the total barely moves.

---

## Latency budget

| Stage | Estimate |
|---|---|
| Acoustic round trip (13 MHz / 3 cm) | 39 µs (physics) |
| One frame acquisition (4 MLA, 109 transmits) | 4.3 ms (physics) |
| Ethernet transfer + buffering | 1–2 ms |
| Transmit-compounding dependency (±4 transmits) | 0.35 ms (structural) |
| Beamforming compute | design target |
| Scan conversion + display | 5–16 ms |
| **Total target** | **≤ 30 ms** |

**Throughput and latency follow different rules here.** Pipelining runs the
stages concurrently on different frames, which is what sustains the frame
rate; it does not shorten the journey of any single frame. The ≤ 30 ms
target is per-frame latency, and along a frame's critical dependency path
the stage times **do** add:

```text
acquisition 4.3 → transfer 1–2 → compounding dependency 0.35
            → compute (≤ 16.7) → scan conversion + display 5–16
```

At the optimistic end that path sums to about 27 ms and the target holds;
at the pessimistic end it reaches about 39 ms and the target does not.
**The budget therefore does not yet close across the full range of its own
stage estimates** — which stages must land at their optimistic values, or
which estimate is wrong, is an open item (docs/open-issues.md).

Two rates appear and mean different things: **30 fps is the processing-rate
assumption** behind the compute tables above, while **60 Hz is the display
deadline**. Acquisition takes 4.3 ms but display ticks every 16.7 ms, so the
delivered frame rate is display-bound and compute has 16.7 ms per frame.
End-to-end (probe → display) beyond 100 ms feels wrong to the operator.

---

## Data rates

| Point | Rate |
|---|---|
| Input (256 ch × 40 MHz × 2 B) | 20.5 GB/s |
| One QSFP-DD port | 100 GB/s (20% utilization) |
| T5 tap (inference primary input, 60 fps) | 160 MB/s |
| GDDR6 | 512 GB/s |
