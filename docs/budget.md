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

> **The 40% is a target for hand-written kernels, not an expectation of the
> stock toolchain.** The current stock Newton-Schulz denominator is 3.024%
> of peak, so about 13.2x remains to the target. The earlier 3.2% figure is
> retained as historical evidence, with its non-reproduction explained below.
> Every card count here describes what the design aims at.

---

## Measured efficiency (p150a, 2026-09-20; targeted correction 2026-09-21)

Issue #65 measured stock `ttnn.matmul` with the catalogue on one p150a,
against the 332 TFLOPS peak above. The 0.75.0 full sweep has 284 rows,
including the default and every catalogue candidate, with 190 successes and
94 failures. A targeted record supersedes its four batch-1024 L16/L32
`batched_dram_sharded` failures after correcting the catalogue's DRAM-worker
count: two BF16 replacements succeed and two FP32 replacements fail later at
program compilation, making the effective totals 192 successes and 92
failures. The separate 0.70.1 record is default-only: 68 rows, with 59
successes and 9 failures. All records report firmware 19.6.0.0 and KMD 2.11.0;
their image digests and companion traces remain separate. The full record and
its 183-sample trace are
`docs/measurements/2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0.json`
and
`docs/measurements/2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0-power.csv`.
The four-row superseding record and its 2-sample trace are
`docs/measurements/2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0.json`
and
`docs/measurements/2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0-power.csv`.

The tables below keep all 16 representative shapes visible while grouping them by
workload family. Each result cell is `efficiency / TFLOPS (memory)`. **Default**
is the fastest successful BF16 default row among the recorded memory placements;
**Best** is the fastest successful BF16 row across the full stock catalogue for
that shape. Values are taken from the 0.75.0 full-sweep record cited above; the
targeted superseding record replaces four failed rows but does not change any
best row below.

#### Newton-Schulz

| Shape | Default BF16 | Best BF16 | Best configuration |
|---|---:|---:|---|
| Newton-Schulz L=16, batch 1024 | 0.194% / 0.6424 TFLOPS (L1) | 0.194% / 0.6424 TFLOPS (L1) | `default` |
| Newton-Schulz L=16, batch 8192 | 0.239% / 0.7919 TFLOPS (L1) | 0.239% / 0.7919 TFLOPS (L1) | `default` |
| Newton-Schulz L=16, batch 65536 | 0.003% / 0.0089 TFLOPS (DRAM) | 0.003% / 0.0095 TFLOPS (DRAM) | `reuse_g1x1_k1_m1_n1_s1x1` |
| Newton-Schulz L=32, batch 1024 | 1.426% / 4.7359 TFLOPS (L1) | 1.426% / 4.7359 TFLOPS (L1) | `default` |
| Newton-Schulz L=32, batch 8192 | 2.992% / 9.9326 TFLOPS (L1) | 2.992% / 9.9326 TFLOPS (L1) | `default` |
| Newton-Schulz L=32, batch 65536 | 0.021% / 0.0708 TFLOPS (DRAM) | 0.023% / 0.0779 TFLOPS (DRAM) | `reuse_g1x1_k1_m1_n1_s1x1` |
| Newton-Schulz L=64, batch 1024 | 3.024% / 10.0391 TFLOPS (L1) | 3.024% / 10.0391 TFLOPS (L1) | `default` |
| Newton-Schulz L=64, batch 8192 | 0.095% / 0.3148 TFLOPS (DRAM) | 0.128% / 0.4250 TFLOPS (DRAM) | `reuse_g1x1_k2_m2_n2_s2x2` |
| Newton-Schulz L=64, batch 65536 | 0.095% / 0.3149 TFLOPS (DRAM) | 0.128% / 0.4251 TFLOPS (DRAM) | `reuse_g1x1_k2_m2_n2_s2x2` |

#### Beamspace

| Shape | Default BF16 | Best BF16 | Best configuration |
|---|---:|---:|---|
| Beamspace B=16, 128 ch, 4096 px | 0.384% / 1.2740 TFLOPS (L1) | 0.492% / 1.6339 TFLOPS (DRAM) | `mcast1d_in0_g8x8_k4_m1_n2_b1x2_s1x2` |
| Beamspace B=16, 128 ch, 65536 px | 1.368% / 4.5424 TFLOPS (DRAM) | 2.167% / 7.1949 TFLOPS (L1) | `mcast1d_in0_g8x8_k4_m1_n32_b1x4_s1x4` |
| Beamspace B=16, 256 ch, 4096 px | 0.775% / 2.5745 TFLOPS (L1) | 0.918% / 3.0470 TFLOPS (L1) | `mcast1d_in0_g8x8_k8_m1_n2_b1x2_s1x2` |
| Beamspace B=16, 256 ch, 65536 px | 1.494% / 4.9616 TFLOPS (DRAM) | 2.208% / 7.3298 TFLOPS (L1) | `mcast1d_in0_g8x8_k1_m1_n32_b1x4_s1x4` |

#### Front-end FIR

| Shape | Default BF16 | Best BF16 | Best configuration |
|---|---:|---:|---|
| Front-end FIR, output width 2 | 0.134% / 0.4461 TFLOPS (DRAM) | 0.287% / 0.9534 TFLOPS (L1) | `mcast1d_in1_g8x8_k2_m128_n1_b4x1_s4x1` |
| Front-end FIR, output width 8 | 0.534% / 1.7725 TFLOPS (DRAM) | 1.162% / 3.8570 TFLOPS (L1) | `mcast1d_in1_g8x8_k2_m128_n1_b4x1_s4x1` |
| Front-end FIR, output width 32 | 2.139% / 7.1005 TFLOPS (DRAM) | 4.605% / 15.2871 TFLOPS (L1) | `mcast1d_in1_g8x8_k2_m128_n1_b4x1_s4x1` |

For contrast only, the non-representative 4096³ square matmul reaches
58.410% / 193.9204 TFLOPS (DRAM) with the default configuration; it is not
included in the representative-shape tables.

Explicit stock configurations help the broad shapes. For front-end FIR width
32 in L1, the default is 4.3389 TFLOPS (1.307%) and the best explicit row is
15.2871 TFLOPS (4.605%), a 3.52x gain. For beamspace B=16, 128 channels and
65536 pixels, L1 improves from 3.5508 TFLOPS (1.070%) to 7.1949 TFLOPS
(2.167%), or 2.03x; for 256 channels it improves from 4.3593 TFLOPS (1.313%)
to 7.3298 TFLOPS (2.208%), or 1.68x. These rows and their configuration
names are in the 0.75.0 full-sweep record cited above.

Newton-Schulz is different. No explicit configuration beats the best default
on a shape where the default L1 row succeeds. On DRAM-only large-batch rows,
explicit reuse does beat the DRAM default: by 7.7% for L=16 batch 65536, 10.0%
for L=32 batch 65536, and about 35% for L=64 batch 8192 and 65536. The largest
of those gains reaches only 0.4251 TFLOPS (0.128%), so it does not change the
3.024% best inverse denominator at L=64 batch 1024. Configuration selection
therefore does not close the roughly 13.2x gap to 40%; hand-written kernel
recovery remains the lever for the MV inverse.

The two toolchains agree on the decision-driving default rows without implying
that every row is identical. The 4096-square BF16 reference is 58.687% in the
0.70.1 default-only record versus 58.410% in the 0.75.0 full sweep, and
Newton-Schulz L=32 batch 8192 in L1 is 3.026% versus 2.992%. Small,
dispatch-bound beamspace p4096 rows differ more, by up to 0.872 percentage
points in the overlapping successful defaults. The bounded comparison supports
the conclusion that the repin does not explain the roughly 3% inverse
denominator; it is not a claim that every row is toolchain-invariant. The
0.70.1 record is
`docs/measurements/2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1.json`,
with its 88-sample trace at
`docs/measurements/2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1-power.csv`.

### The August L1 row does not reproduce

The 2026-08-14 record contains `newton_schulz_L64_b8192`, BF16, L1 at
10.7136 TFLOPS (3.227%). It used image digest
`ead7b800bdb6bebb9425c377222314447c5b2052f6e8b1e3c9caa1818cb7d8c4`, KMD
2.8.0, and harness `112ff585f4b52f90525d23650a90b14ec6d7a55d`. Both September
records fail that BF16 default L1 row with allocator OOM. The 0.75.0 record
reports an attempted 67,108,864-byte output allocation, 610,304 bytes needed
per bank, 1,220,608 bytes already allocated, and only 240,896 of 1,461,504
bytes free per bank. The 0.70.1 record reports the same allocation and
per-bank allocation, with 241,152 of 1,461,760 bytes free.

This is not an equivalent reproduction. In the August harness,
`_execute_once` called `ttnn.matmul(a, b)` after creating A and B with the
selected memory config, leaving output placement at the operation default. In
the current harness, `_RuntimePlan` assigns A, B, and output to the selected
memory and `_execute_once` calls `ttnn.matmul(...,
memory_config=plan.output_memory_config)`. Thus the August L1 label means L1
inputs with default output placement, while the September L1 label means all
three tensors explicitly in L1. The old 3.2% headline did not reproduce under
the stricter all-L1 placement; the difference is the harness's output-placement
semantics. Both current images fail identically, so this failure is not
evidence of a toolchain regression. The board is powered off and no rerun is
available.

**The silicon reaches 58.410% on a shape it likes.** The 40% target remains
plausible for the hardware: the large square matmul beats it, while the
workload-shaped Newton-Schulz cases do not. The stock baseline leaves the
remaining inverse gap to a hand-written kernel: packing small matrices into
full tiles, fusing the four real matmuls of a complex one, keeping R resident
across the iteration, and avoiding per-operation dispatch remain kernel work.

**Power and clock did not bind in the full sweep.** The 0.75.0 trace peaked
at 110 W, 1350 MHz, and 73.9 °C; neither the 150 W firmware-reported
limit nor the 300 W board limit was reached.

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
