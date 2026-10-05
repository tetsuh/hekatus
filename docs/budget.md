# budget.md

Working extract of the estimate tables in design.md §10. When a number
changes here, fix design.md too.

**Assumptions for every table**: 30 fps and 2048 depth points unless stated.

## Issue #100 measured-default configuration (host-only update; device validation pending)

The Issue #100 defaults now exactly match the PR98/#96 measured row:
`variant=bf16`, `math_fidelity=HiFi3`, `fuse_s=true`,
`fp32_dest_acc_en=true`, `matrix_block=8`, `double_buffer=true`, and
`dst_full_sync_en=true`. R and X0 are in L1 and output tensors are in DRAM;
the compatibility input shorthand and its resident identity/zero buffers remain
in L1. Every previous state, fusion, fidelity, block, synchronization, and
placement choice remains selectable explicitly.

The read-only source is the PR98/#96 measurement record row
`L32_b8192_bf16_full_sync_block8` (`performance.rows`): L=32, batch=8,192,
BF16 state, HiFi3, fused S, FP32 DEST, block 8, full-sync, double buffering,
R/X0 in L1, DRAM output, and `preflight_bytes=1,295,104`. Its stored
78.78349 TFLOPS from p50 (about 78.8) remains historical user evidence only;
this change makes no new performance claim. Device status for this branch is
**UNMEASURED**: no board, container, SSH, or measurement rerun was performed,
and no measurement JSON/CSV is changed.

The policy at both DEST and L1 boundaries is **fail-fast**. `prepare` validates
DEST usage and then L1 usage before tensor allocation; `run_newton_schulz_kernel`
propagates the `ValueError`, while the bench runner records the same error in a
failed row. Neither path silently changes `matrix_block`, memory placement, or
synchronization mode.

Board-free L1 preflight for the measured defaults (R/X0 L1, DRAM output, BF16
state, fused S, HiFi3, block 8, double buffer, full-sync DEST, and the 110-core
host model) is:

| Case | Padded 32x32 tile count | Host preflight result |
|---|---:|---|
| L=16, batch=4 | 8 | fits, 557,824 bytes |
| L=16, batch=8,192 | 4,096 | fits, 885,504 bytes |
| L=32, batch=4 | 8 | fits, 557,824 bytes |
| L=32, batch=8,192 | 8,192 | fits, 1,295,104 bytes |
| L=16 tails batch=1/3/5/31/63 | 8/8/8/16/32 | fits, 557,824 bytes in each case |
| L=32 tails batch=1/3/5/31/63 | 8/8/8/32/64 | fits, 557,824 bytes in each case |

The explicit legacy `fuse_s=false`, HiFi4, `output_memory=l1` L=32
batch-8,192 row remains fail-fast at 1,716,992 bytes (144,128 over the
1,572,864-byte budget); there is no implicit fallback. Under the measured
defaults, all 128 existing batch-1..64 partial cases (L=16/32, block 8) fit;
the existing 1,024-case explicit block/fusion inventory remains covered by
board-free preflight tests.

Tomorrow's device-only commands intentionally omit fuse, fidelity, and
placement overrides so the defaults exercise the measured configuration:

```bash
# Correctness for the default batch and tail cases.
enodia/tt/bench/run_in_container.sh --pytest \
  -m tt_device \
  tests/test_newton_schulz_kernel.py::test_device_issue100_defaults_match_bf16_rounded_reference

# Same-run 1,000-launch record for both default shapes.
enodia/tt/bench/run_in_container.sh out/issue100-defaults -- \
  --device-id 0 \
  --only newton_schulz_L32_b8192 \
  --only newton_schulz_L16_b8192 \
  --dtype bfloat16 --memory l1 --kind custom_newton_schulz \
  --iters 1 --repeats 1000
```

The Issue #100 card denominator is **UNMEASURED** on this branch:
`p50_tflops_per_card = <tomorrow's ADR-0005 record>`. The resulting card range
must be filled from that same-run p50, without inventing a value:
`100 / p50_tflops_per_card` through `127.5 / p50_tflops_per_card` cards.
Power, clock, duration, and new-default device performance remain unmeasured
until those commands are run.

**Two capacity bases appear in this document; each table names the one it
uses.**

| Basis | Value | Used by |
|---|---|---|
| Theoretical peak (BF16) | 332 TFLOPS | the "% of theoretical peak" columns |
| Usable per card (peak × 40% effective) | 133 TFLOPS | the "cards" columns |

A percentage from one table cannot be combined with a card count from
another without converting: peak % × 2.5 gives the share of usable capacity.

> **The 40% is a target for hand-written kernels, not an expectation of the
> stock toolchain.** The current stock Newton-Schulz denominator is 3.024%
> of peak (a stored `achieved_tflops` fastest-launch stock-catalogue value,
> not p50), and Issue #63's measured hand-written rows below are stored
> `achieved_tflops` fastest-launch values, not p50; they land between the
> historical 3.2% floor and the 30% planning target. The earlier 3.2% figure is
> retained as historical evidence, with its non-reproduction explained below.
> ADV-99-1 shows that the #90 aggregate 12/8 scaling is an upper bound, not an
> inverse-only correction. The documented N=12 range for the current 1D all-mode
> estimate remains 100–127.5 TFLOPS, but Issue #100 leaves the per-card
> denominator and card count **UNMEASURED** until tomorrow's same-run p50 record;
> the equation and missing mode-shape assumptions are explicit below. This is an
> extrapolation, not a full-system
> benchmark or an all-mode simultaneous benchmark.

---

## Measured efficiency (p150a, 2026-09-20; targeted corrections 2026-09-21 and 2026-09-23)

Issue #65 measured stock `ttnn.matmul` with the catalogue on one p150a,
against the 332 TFLOPS peak above. The 0.75.0 full sweep has 284 rows,
including the default and every catalogue candidate, with 190 successes and
94 failures. The final effective count is derived from the `results` and
`supersedes.rows` arrays in the landed records, rather than copied from a
previous summary:

| Record and role | Rows counted from `results` |
| --- | --- |
| `docs/measurements/2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0.json` — original catalogue | 190 `ok`, 94 `failed` |
| `docs/measurements/2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0.json` — replaces four failed batch-1024 L16/L32 `batched_dram_sharded` rows named in its `supersedes.rows` | 2 `ok`, 2 `failed` |
| `docs/measurements/2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0.json` — replaces two original `ok` beamspace B=16, 256-channel, 4096-pixel `dram_sharded` rows named in its `supersedes.rows` | 2 `ok` |

The first superseder maps the four original failed g7x1 rows to two successful
and two compilation-failed g8x1 replacements. The second maps the two original
`ok` g4x1 rows to two successful g8x1 replacements. Apply each
`supersedes.rows` entry once: remove those six named predecessor rows from the
original count and add the six replacement rows. The effective count is
`190 - 2 + 2 + 2 = 192` successes and `94 - 4 + 2 = 92` failures.
The other 278 original rows remain authoritative, so the effective catalogue
still has 284 rows. The separate 0.70.1 record is default-only: 68 rows, with
59 successes and 9 failures. All records report firmware 19.6.0.0 and KMD
2.11.0; their image digests and companion traces remain separate. The full
record and its 183-sample trace are the first JSON path above and
`docs/measurements/2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0-power.csv`.
The four-row superseding record and its 2-sample trace are the second JSON
path above and
`docs/measurements/2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0-power.csv`.
The two-row superseding record and its 3-sample trace are the third JSON path
above and
`docs/measurements/2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0-power.csv`.

The tables below keep all 16 representative shapes visible while grouping them by
workload family. Each result cell is `efficiency / TFLOPS (memory)`. **Default**
is the fastest successful BF16 default row among the recorded memory placements;
**Best** is the fastest successful BF16 row across the full stock catalogue for
that shape. Values are taken from the 0.75.0 full-sweep record cited above; the
two targeted superseding records replace six rows in total and do not change any
best row below. The table's stored `achieved_tflops` values are fastest-launch
values selected by the runner, not p50-derived values; they are retained as
stock-catalogue comparison evidence.

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

For contrast only, the non-representative 4096³ square matmul's stored
`achieved_tflops`/efficiency values are 58.410% / 193.9204 TFLOPS (DRAM) at
fastest launch, not p50, with the default configuration; it is not included in
the representative-shape tables.

The following stock-catalogue comparisons use stored
`achieved_tflops`/efficiency fields: fastest-launch values, not p50. Explicit
stock configurations help the broad shapes. For front-end FIR width 32 in L1,
the default is 4.3389 TFLOPS (1.307%) and the best explicit row is 15.2871
TFLOPS (4.605%), a 3.52x gain. For beamspace B=16, 128 channels and 65536
pixels, L1 improves from 3.5508 TFLOPS (1.070%) to 7.1949 TFLOPS (2.167%), or
2.03x; for 256 channels it improves from 4.3593 TFLOPS (1.313%) to 7.3298
TFLOPS (2.208%), or 1.68x. These rows and their configuration names are in the
0.75.0 full-sweep record cited above.

Newton-Schulz is different. No explicit configuration beats the best default
on a shape where the default L1 row succeeds. On DRAM-only large-batch rows,
explicit reuse does beat the DRAM default: by 7.7% for L=16 batch 65536, 10.0%
for L=32 batch 65536, and about 35% for L=64 batch 8192 and 65536 (each
comparison uses stored `achieved_tflops` fastest-launch values, not p50). The
largest stored `achieved_tflops` fastest-launch value in those rows, not p50,
reaches only 0.4251 TFLOPS (0.128%), so it does not change the 3.024% best stock
inverse denominator at L=64 batch 1024 (also a stored `achieved_tflops`
fastest-launch value, not p50).
Configuration selection alone does not close the stock-to-40% gap; the historical
Issue #63 `achieved_tflops` headlines recover to 15.6% at L=32 and 3.24% at packed L=16
(fastest-launch values, not p50), leaving the 30% planning target open rather
than a 13.2x stock-only statement.

The two toolchains agree on the decision-driving default rows without implying
that every row is identical. The stored stock-catalogue values below are
fastest-launch values, not p50: the 4096-square BF16 reference is 58.687% in the
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
10.7136 TFLOPS (3.227%), a stored `achieved_tflops` fastest-launch
stock-catalogue value, not p50. It used image digest
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

**The silicon's stored stock-catalogue result is 58.410% at fastest launch,
not p50, on a shape it likes.** The 40% target remains plausible for the
hardware: the large square matmul beats it, while the
workload-shaped Newton-Schulz cases do not. The stock baseline leaves the
remaining inverse gap to a hand-written kernel: packing small matrices into
full tiles, fusing the four real matmuls of a complex one, keeping R resident
across the iteration, and avoiding per-operation dispatch remain kernel work.

**Power and clock did not bind in the full sweep.** The 0.75.0 trace peaked
at 110 W, 1350 MHz, and 73.9 °C; neither the 150 W firmware-reported
limit nor the 300 W board limit was reached.

## Issue #63 Scope 5 — measured kernel boundary

The same-shape batch-8192 records change the planning claim. Against the
332 TFLOPS peak, the L=32 `block4_all_l1` row's stored `achieved_tflops` is
51.9164 TFLOPS (15.6375%, reported here as **51.92 TFLOPS and 15.6%**), a
fastest-launch value rather than p50, and 5.7x the same-run stock best's
stored fastest-launch value, not p50, of 9.1096 TFLOPS. The 51.92 headline is
the row in
`docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000.json`.
The requested current/history check
`docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000.json`
records 50.0263 TFLOPS for its current reproduction and 50.6196 TFLOPS for
its historical row; those stored `achieved_tflops` values are fastest-launch,
and the rows are cited as repeatability context, not as the source of the
51.92 value.

**R-residency attribution disposition.** The complete-provenance superseding
record
`docs/measurements/2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-superseding.json`
supersedes the immutable predecessor
`docs/measurements/2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control.json`.
It measures resident and per-iteration R reload in one watcher-free run under the
same L=32, batch-8192, block-4, fused-S, HiFi3, FP32-state, all-inputs-L1
conditions, with 1,000 launches per row. Both variants pass the batch-4 and
batch-8192 correctness gates. The stored `achieved_tflops` values below are
fastest-launch values, not p50: resident reaches **51.82917713218822 TFLOPS**
and reload-R reaches **31.33414466591468 TFLOPS**, so reload-R is **39.5434%
lower** in TFLOPS and 65.1843% slower at median latency. This retained Issue
#63 control is historical eight-iteration evidence, not the current Stage 2
N=12 specification. The resident result remains within the existing roughly
+/-2-4% repeatability context of the 51.92 TFLOPS headline; the headline and
historical roughly two-card extrapolation therefore remain unchanged. R
residency is a measured material attribution, while the control does not claim
it is the sole cause of the remaining combined-kernel gap. The
companion power trace is
`docs/measurements/2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-superseding-power.csv`.

The packed L=16 `custom_block4` row's stored `achieved_tflops` is 10.7733
TFLOPS (3.2450%, reported as **10.77 TFLOPS and 3.24%**), a fastest-launch
value rather than p50, and 14.5x the same-run stock best of 0.7420 TFLOPS, in
`docs/measurements/2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json`.
Its 0.4019 ms per-iteration median is below the L=32 row's 0.6655 ms in the
L=32 per-input record: the packed L=16 path is faster in wall-clock despite
its lower useful-work TFLOPS. This is only a cost/operation-volume comparison
for the diagonal fallback, because L=16 uses fewer logical dimensions and less
work. It is not a beamspace-dimension reduction versus an MV image-quality
comparison. The L=16 record is
`docs/measurements/2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json`;
beamspace dimension and image quality remain separate decisions.

The three Issue #63 conclusions are direct:

1. The historical L=32 `achieved_tflops` headline is 15.6% of peak at
   fastest launch, below the 30% efficiency target. Applying that historical
   workload efficiency to the roughly 100 TFLOPS eight-iteration 1D all-mode
   estimate gave about 2 cards in that historical N=8 scenario. Issue #100's
   current N=12 denominator remains unmeasured until tomorrow's same-run p50
   record; both the historical count and the pending count are extrapolations,
   not full-system or all-mode benchmarks.
2. Packed L=16 is 3.24% and 14.5x stock, yet faster in wall-clock than L=32
   only as a cost/operation-volume comparison for the diagonal fallback,
   because it uses fewer logical dimensions and less work. It is not a
   beamspace-dimension reduction versus an MV image-quality comparison.
3. The steady cycle-counter and optimization evidence points to fixed
   handoff/queue overhead as the leading measured explanation. Math, unpack,
   and reader were not established as causal bottlenecks: the steady record
   explicitly declines a unique bottleneck, and its RISC windows overlap.

The evidence for that third conclusion is bounded. The steady counter record
`docs/measurements/2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady.json`
separates reader and writer waits but retains unclassified compute cycles and
warns that RISC totals are overlapping. The optimization record
`docs/measurements/2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000.json`
shows the stored `achieved_tflops` fastest-launch throughput changing by only
-0.09% and +0.13% versus baseline; these are not p50/median values. The unpack
diagnostic
`docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000.json`
reports a variant difference but makes no causal unpack claim. The
matrix-block record
`docs/measurements/2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000.json`
shows block 4 improving over block 1, which correlates with fewer fixed
handoffs/queue turns but does not prove causality.

### L=64 board-free capacity estimate (no kernel implementation)

This is a host-only extrapolation from the current fused-S ledger, not L=64
dispatch support. It models 8,192 logical matrices, 110 active cores, and
four 32x32 physical tiles per logical matrix. Each logical matrix remains on
one core: the maximum aligned assignment is 75 matrices (300 physical tiles)
for `matrix_block=1` and 76 matrices (304 physical tiles) for
`matrix_block=4`. The current static prefix is 111,360 bytes and the total
budget is 1,572,864 bytes.

`fuse_s` omits positive `R_REAL`; three BF16 signed R tensors remain. X0 is
two FP32 tensors. Identity is one resident BF16 page and zero is one resident
FP32 page when the compatibility `input_memory=l1` placement is used. The
`bf16-fp32state` variant keeps state and output CB pages in FP32 and allocates
its output tensors in DRAM, as the current source does. The CB estimate scales
each active current queue by L64's four physical tiles while retaining one-page
identity/zero and fused-product descriptors: 309,248 bytes for block 1 and
702,464 bytes for block 4. The 2x2 L64 matmul has K=2; this estimate streams the
two K tiles through DEST and does not add a second resident CB window. It says
nothing about a future kernel's DEST schedule or correctness.

The per-tensor rows make the placement shorthand explicit. `all-L1` places
R, X0, identity, and zero in L1; `all-DRAM` places all four groups in DRAM;
the two middle rows keep the constants in L1 and split R/X0 as named. Output
tensors remain DRAM in every row.

| matrix_block | placement | max logical/core | max physical tiles/core | CB bytes | tensor bytes | static prefix | total | budget | boundary |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | all-L1 | 75 | 300 | 309,248 | 4,306,944 | 111,360 | 4,727,552 | 1,572,864 | does not fit (+3,154,688) |
| 1 | R-L1/X0-DRAM | 75 | 300 | 309,248 | 1,849,344 | 111,360 | 2,269,952 | 1,572,864 | does not fit (+697,088) |
| 1 | R-DRAM/X0-L1 | 75 | 300 | 309,248 | 2,463,744 | 111,360 | 2,884,352 | 1,572,864 | does not fit (+1,311,488) |
| 1 | all-DRAM | 75 | 300 | 309,248 | 0 | 111,360 | 420,608 | 1,572,864 | fits |
| 4 | all-L1 | 76 | 304 | 702,464 | 4,364,288 | 111,360 | 5,178,112 | 1,572,864 | does not fit (+3,605,248) |
| 4 | R-L1/X0-DRAM | 76 | 304 | 702,464 | 1,873,920 | 111,360 | 2,687,744 | 1,572,864 | does not fit (+1,114,880) |
| 4 | R-DRAM/X0-L1 | 76 | 304 | 702,464 | 2,496,512 | 111,360 | 3,310,336 | 1,572,864 | does not fit (+1,737,472) |
| 4 | all-DRAM | 76 | 304 | 702,464 | 0 | 111,360 | 813,824 | 1,572,864 | fits |

The arithmetic is reproducible without a board with
`python3 tools/newton_schulz_l64_capacity.py`; the regression is
`tests/test_newton_schulz_l64_capacity.py`. No L=64 production dispatch is
added.

---

## Governing law

MV cost is dominated by the inverse, `L^3` (L = subaperture size). Growing
the element count scales `L ∝ N` and the scanline count `∝ N`, so the
**total goes as N^4**. 64 → 128 receive channels is a 16× increase.

## ADV-99-1: inverse-only iteration correction

### History of the #90 values

The pre-#90 table at `f3503c9^1` contains the historical N=8 values. Commit
`39e2d1a` (the feature commit merged by PR #90 as `f3503c9`) changed the
inverse-containing aggregate rows as follows:

| Aggregate row | N=8 before #90 | Value in `39e2d1a` / #90 | History reading |
|---|---:|---:|---|
| MV with Newton-Schulz inverse | 33 | 49.5 | full 12/8 multiplication |
| 1D B-mode beamspace MV | 25 | 37.5 | full 12/8 multiplication |
| 1D color flow wall filter + MV | 30 | 45 | full 12/8 multiplication |
| 2D beamspace MV | 37 | 55.5 | full 12/8 multiplication |

The fixed rows `~5` (DAS + phase-screen correction) and `~40` (SLSC / CF /
DMAS) were unchanged. The same commit also multiplied the generic inverse
configuration rows (`35→52.5`, `560→840`, `19→28.5`, `1,100→1,650`, and
`37→55.5`). The commit contains no per-mode fixed/inverse decomposition.
Therefore those #90 N=12 values are retained below only as **full-scaling
upper bounds**, not as corrected totals. The current `shapes.py` history was
updated to account for 12 iterations by `bb85c02`; that code change does not
supply the missing mode-level decomposition.

### Source-derived inverse equation

Let `P` be the number of pixels in one frame, `L` the MV matrix dimension, `r`
the frame rate, and `n` the Newton-Schulz iteration count. The source creates a
square `MatmulShape(batch=P, m=L, k=L, n=L, real_matmuls=4)` for one complex
matrix multiplication. `shapes.total_flops` therefore gives

```text
M(P, L) = P × 4 × (2 × L × L × L)
I(n, P, L, r) = r × (2 complex matmuls/iteration) × n × M(P, L) / 10^12
              = r × 2n × P × 4 × 2L^3 / 10^12 TFLOPS.
```

The `4` is `MatmulShape.real_matmuls` in `enodia/tt/bench/shapes.py`; the
`2 × L × L × L` is `total_flops`' multiply-plus-add count. The two complex
matmuls per iteration are stated by `newton_schulz_shapes` and implemented as
`COMPLEX_MATMULS_PER_INVERSE = 2 × NEWTON_SCHULZ_ITERATIONS` in
`enodia/tt/bench/newton_schulz_reference.py`. No R formation, beamspace
projection, wall filter, or other iteration-independent work belongs in `I`.

For a mode with N=8 total `T_8` and inverse component `I_8`, the corrected
N=12 total is

```text
T_12 = (T_8 − I_8) + (12/8)I_8 = T_8 + 0.5I_8,
U_12 = T_8 + 0.5T_8 = 1.5T_8.
```

`U_12` is the old full-row scaling and is an upper bound because `0 ≤ I_8 ≤
T_8`; it is not a claim that fixed work scales. The equation is numerically
checkable for documented shape points. For the exact catalogue shape
`newton_schulz_L16_b65536` at `r=30`, `M=2,147,483,648` FLOPs and
`I(8,65536,16,30)=1.03079215104` TFLOPS,
`I(12,65536,16,30)=1.54618822656` TFLOPS. If the 13 MHz geometry note is used,
`P=434×2048=888,832` and `L=16` instead give
`I(8,888832,16,30)=13.98011854848` and
`I(12,888832,16,30)=20.97017782272` TFLOPS. These are shape calculations,
not measurements and are not substituted into an aggregate mode unless its
`P` and `L` are explicitly bound to that shape.

The target table binds B-mode to beamspace dimension `B=16`, but it does not
bind one exact pixel count to that mode. It does not state `P` or `L` for the
color-flow MV after the 64-channel wall filter, and it does not state the 2D
volume pixel count. The 64 wall-filter channels do not define the MV matrix
size. Those missing bindings prevent a trustworthy single corrected point for
every MV row; no value is invented below.

### Tables: N=8 baseline and N=12 upper bound

The method and configuration tables retain their N=8 source values and show
#90's N=12 full-scaling result explicitly as an upper bound. A mode-specific
corrected value is `T_8 + 0.5I_8` from the equation above.

## By method (64 receive channels, 30 fps) — budget estimates, basis:
theoretical peak

| Method | N=8 baseline TFLOPS | N=12 upper-bound TFLOPS |
|---|---:|---:|
| DAS | 0.004 | 0.004 |
| CF / PCF / F-DMAS | 0.015 | 0.015 |
| SLSC | 1 | 1 |
| MV: R formation only (sliding update) | 2 | 2 |
| MV: with Newton-Schulz inverse | 33 | 49.5 |
| ESBMV (eigendecomposition) | 100–170 | 100–170 |

## By configuration — budget estimates, basis: usable per card (133 TFLOPS)

| Configuration | Recv ch | L | N=8 baseline TFLOPS | N=12 upper-bound TFLOPS |
|---|---:|---:|---:|---:|
| 128 elements / 64 ch receive | 64 | 32 | 35 | 52.5 |
| 256 elements / 128 ch receive | 128 | 64 | 560 | 840 |
| 256 elements + beamspace (B=16) | 128 | 16 | 19 | 28.5 |
| post-μBF 256 ch, volume | 256 | 128 | 1,100 | 1,650 |
| post-μBF 256 ch + beamspace | 256 | 16 | 37 | 55.5 |
| 2D fully digital 4096 ch full MV | 4096 | 2048 | ~7.2e7 | ~1.08e8 |

The last upper-bound row follows the N⁴ law from the N=8 256-channel volume
row (`1,100 × 16^4 ≈ 7.2e7`; the upper bound is `1,650 × 16^4 ≈ 1.08e8`).
An earlier revision carried 1.85e8, which did not reconcile with the law.
These generic cards are still the separate 40% target basis; they are not the
the historical denominator context above; Issue #100's denominator is pending.

## Target configuration (1D 256 elements / 128 ch receive + post-μBF 2D) —
budget target, basis: theoretical peak

| Mode | Beamformer | N=8 baseline TFLOPS | N=12 corrected expression / range |
|---|---|---:|---|
| 1D B-mode | DAS + phase-screen correction | ~5 | ~5 (no inverse) |
| 1D B-mode | + SLSC / CF / DMAS | ~40 | ~40 (no inverse) |
| 1D B-mode | + beamspace MV | ~25 | `25 + 0.5I_B,8`, bounded by 25–37.5 |
| 1D color flow | per-channel wall filter + MV | ~30 | `30 + 0.5I_C,8`, bounded by 30–45 |
| 2D volume | beamspace MV | ~37 | `37 + 0.5I_2D,8`, bounded by 37–55.5 |

Here `I_B,8`, `I_C,8`, and `I_2D,8` are the source-derived inverse terms;
R formation, projection, wall filtering, and all other fixed work remain in
the N=8 baseline. The old #90 entries 37.5, 45, and 55.5 are the right-hand
endpoints only.

### Scope 5 total and cards

The historical N=8 1D all-mode total is
`T_8 = 5 + 40 + 25 + 30 = 100 TFLOPS`. The inverse-only correction is

```text
T_12,1D = 45 + (25 + 30) + 0.5(I_B,8 + I_C,8)
         = 100 + 0.5(I_B,8 + I_C,8) TFLOPS.
```

The documented aggregate rows do not provide `P_C`, `L_C`, or the fixed/inverse
split needed to evaluate `I_C,8`. However, the mode rows bound the inverse terms:
`0 ≤ I_B,8 ≤ 25` and `0 ≤ I_C,8 ≤ 30`. The fixed 45 TFLOPS is not scaled by
12/8, so

```text
T_12,1D = 100 + 0.5 × (I_B,8 + I_C,8) TFLOPS
100 ≤ T_12,1D ≤ 100 + 0.5 × (25 + 30) = 127.5 TFLOPS.
```

The geometric B-mode shape calculation above can be used once the color-flow
bindings are supplied; it cannot close the aggregate calculation by itself.
This is a bounded range, not an invented aggregate point.

The Issue #100 card denominator is reserved for tomorrow's same-run device
record and is **UNMEASURED** on this branch. The record must provide the
L=32/batch=8,192 new-default p50 and retain p99/p99.9 beside the L=16 and
batch-4/tail rows. Until then, write
`p50_tflops_per_card = <tomorrow's ADR-0005 record>` and
`100 / p50_tflops_per_card` through `127.5 / p50_tflops_per_card` cards; no
numeric card count is asserted here. The older PR93-derived denominator remains
historical context only and is not reused for Issue #100.

**Stage 2 measured timing (separate from the planning estimate):** the
2026-10-04 record
`docs/measurements/2026-10-04-p150a-newton-schulz-stage2-pr90-current-head.json`
was measured with harness commit `ce8bb30`, and device tests were run at
commit `0d786b0`, as recorded in the measurement record. It supersedes the two
earlier BF16 diagnostics. It measures the twelve-iteration
BF16-`R` path at 58.1 TFLOPS / 0.888 ms p50 for L=32 and 12.5 TFLOPS /
0.516 ms p50 for packed L=16, from 1,000 launches per case. The L=32 launch
time is about 1.33× the comparable eight-iteration record. The board-gated
test selection passed 1,031/1,031 at commit `0d786b0`. The diff from
`ce8bb30` to `0d786b0` contains only the measurement-record update and no
kernel or valid numerical-path changes. The measured TT kernel and benchmark
numerical paths are unchanged. Commit `0debbe2` aligned the spec defaults (X0
and iteration count) with ADR-0012. These are board timings, not a revised
theoretical peak or an M5 image-quality result; the BF16 input-`R`
perturbation remains a separate representation question. A separate
source-evidence supplement corroborates the physical-board provenance for the
two predecessor diagnostics:
`docs/measurements/2026-10-04-p150a-newton-schulz-stage2-bf16-provenance-supplement.json`.

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
