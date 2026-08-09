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

> **The 40% effective efficiency is an unverified assumption.** Measure it
> first on Track B (docs/open-issues.md B2). If it halves, the "fits on one
> card" claim collapses.

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

**The stages are pipelined, not sequential — do not add the column.** Each
stage runs concurrently on a different frame; the column gives each stage's
own budget, and the ≤ 30 ms target is the end-to-end latency of one frame
travelling through the pipeline, which overlaps stages wherever a stage does
not depend on the previous one completing.

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
