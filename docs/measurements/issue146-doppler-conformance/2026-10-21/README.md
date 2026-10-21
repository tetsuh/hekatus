# Issue #146 Doppler-slow conformance — 2026-10-21

## Scope and provenance

This archive contains the owner-confirmed six-run H-DS measurement and its
board-free diagnostic lineage. The preregistered order was
**C018 → C108 → C081 → C081 → C108 → C018**. Each run used a 180-second
host-duration-bounded workload with a 230-second guard. The strict plateau was
`[60, 180)` seconds from the first launch start; it used the captured rows
without a held boundary point, padding, or interpolation.

- Plan: [revision 6](../../2026-10-20-host-newton-schulz-issue146-plan.json),
  unchanged from raw SHA-256
  `b024dfc33dc9ab132130ccaa429d25b37805d2d472fcb38b549fdd67070cfd5c`.
  Its canonical SHA-256 is
  `28a9e96b026f74243ee769ce61436e2c10d1b883c5a84e9bf6306252a85bf324`.
- Harness: `9caa399026e5aa3e64f1c0f24911a3d3c30baea3`, clean and pushed when
  the runs were acquired.
- Board identity: serial `0000040331914124`, PCI
  `0000:09:00.0`, host node `/dev/tenstorrent/1`, and container-visible
  device 0. No other device was used.
- Firmware bundle: `19.6.0.0`; KMD `2.11.0`; kernel
  `Linux 6.8.0-138-generic`.
- Image digest:
  `sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621`.
- Host readers: tt-smi `6.2.1`, tt-umd `0.9.8`, and pyluwen `0.9.0`.
- The plan requested a 1-second trace interval. Captured telemetry cadence
  was approximately 1.24 seconds because each tt-smi capture also takes time.
- Each launch used 16 outer repetitions; groups contained 8 launches. The
  long-run order, 180-second load, 230-second guard, and all thresholds were
  unchanged.

The six explicit run directories contain all 143 source files, preserving
source basenames and bytes: records, environments, pre/post snapshots, Issue
#146 traces, legacy companion traces and completion markers, device group
arrays and envelopes, runner results, wrapper output, execution records,
Docker listings, and ownership-helper records. The official analyzer output
is [analysis.json](analysis.json). The onset supplement is a separate
exploratory artifact at
[raw-trace-onset-supplement.json](raw-trace-onset-supplement.json).

## Preregistered outcome

**Classification: H-DS supported.** All six runs were `ok` and evaluable; all
required fields were present; all R1–R7 and K1–K4 criteria passed; and P1–P3
passed in both order halves. H-OTHER was not selected, and the result is not a
missing-data classification. There were no retries, resets, added workloads,
or cooldown-cap stops.

The regulated-run power criterion was applied exactly as registered: each
regulated run's two-sided 99 percent interval had to intersect the fixed
predicted interval `[300, 301]` W, with half-width at most 10 W. The prediction
was not moved and the threshold was not widened. The four regulated intervals
were `[297.328661, 307.413607]`, `[292.098132, 305.819393]`,
`[292.849102, 307.150898]`, and `[299.279030, 308.575137]` W.

| Registered matrix | Result |
|---|---|
| R1 power regulated | PASS in runs 2, 3, 4, and 5; each 99% interval intersects `[300, 301]` W |
| R2 arbiter unclamped | PASS in all four regulated runs; every plateau AICLK and arbiter frequency is below 1350 MHz and every mean is below 1282.5 MHz |
| R3 masks unchanged | PASS: `ENABLED_MAX_ARB=0x1d1` and `ENABLED_MIN_ARB=0x3` |
| R4 busy minimum arbiter | PASS: every regulated plateau sample has `AICLK_ARB_MIN=0x10546` |
| R5 AICLK tracks arbiter | PASS: every regulated plateau residual is 0–7 MHz |
| R6 thermal not binding | PASS: ASIC and all decoded GDDR die temperatures remain below their limits |
| R7 reported index | PASS: every regulated plateau index is 7 or 8 |
| K1 control clamped | PASS in runs 1 and 6; only the two registered control pairs occur and the clamp pair is a strict majority |
| K2 control unregulated | PASS in runs 1 and 6; each 99% interval is entirely below 300 W |
| K3 control mask and busy | PASS in runs 1 and 6 |
| K4 control thermal | PASS in runs 1 and 6 |
| P1 control unthrottled | PASS in both halves |
| P2 regulated depressed | PASS in both halves |
| P3 C081 above C108 | PASS: 1136.938144 > 1017.536082 MHz in the first half and 1134.711340 > 1013.375000 MHz in the second |

## Plateau statistics

The verdict estimator is the arithmetic mean and sample SD (`n-1`) of the
strictly selected plateau rows. Positive lag-1 correlation uses the registered
AR(1) effective sample-size correction; non-positive `r1` leaves `n_eff=n`.
Degrees of freedom are `max(2, floor(n_eff)-1)`. Intervals are two-sided 99%
t intervals.

| Run | Configuration | n | Mean W | SD W | r1 | n_eff | df | 99% CI W | Half-width W |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | C018 | 97 | 259.164948 | 6.422098 | 0.059476 | 86.109329 | 85 | [257.341396, 260.988501] | 1.823552 |
| 2 | C108 | 97 | 302.371134 | 18.897376 | -0.106728 | 97.000000 | 96 | [297.328661, 307.413607] | 5.042473 |
| 3 | C081 | 97 | 298.958763 | 20.018698 | 0.233134 | 60.322730 | 59 | [292.098132, 305.819393] | 6.860631 |
| 4 | C081 | 97 | 300.000000 | 20.299323 | 0.257318 | 57.296640 | 56 | [292.849102, 307.150898] | 7.150898 |
| 5 | C108 | 96 | 303.927083 | 17.325518 | -0.001549 | 96.000000 | 95 | [299.279030, 308.575137] | 4.648053 |
| 6 | C018 | 97 | 259.814433 | 6.031739 | -0.103157 | 97.000000 | 96 | [258.204956, 261.423909] | 1.609476 |

There were 581 retained plateau rows in total and zero classified glitches.
No row was removed from the verdict statistics.

## Arbiter words

All 895 raw trace rows carry all four arbiter words. The raw and plateau scopes
must not be conflated:

- Each regulated raw trace contains four pre-plateau clamp rows with
  `AICLK_ARB_MAX=0x80546`: three occur before the first launch and one is an
  early-loaded sample. The early-loaded relative times are 0.286576525 s
  (run 2), 0.668380628 s (run 3), 0.783952403 s (run 4), and 0.505580697 s
  (run 5). All four rows are outside the strict `[60, 180)` plateau; they are
  not all pre-load rows.
- The 387 regulated plateau rows (97 + 97 + 97 + 96) all have reported index 7
  and arbiter frequency below 1350 MHz. This is a plateau statement, not a
  claim that every regulated raw row has index 7 or a depressed frequency.
- Across the entire raw control trace, run 1 contains one
  `0x70545`/1343-MHz control dip and otherwise `0x80546`/1350 MHz. Run 6 is
  entirely `0x80546`/1350 MHz. K1 uses exactly these two pairs and requires
  the clamp pair to be a strict majority.
- The post-load minimum arbiter returns to idle (`0x10320`), not busy. It is
  therefore not substituted for the loaded plateau R4 word `0x10546`.

Each plateau source property is tested separately: power CI, frequency,
mask, busy minimum arbiter, residual, thermal channels, and reported index
are not inferred from one another. The disabled TDC current is recorded only
as exploratory data.

## Execution and cooldown

The grouped-launch load contract remained active throughout the six runs. The
observed load, group, and cleanup facts were:

| Run | Configuration | Observed load s | Groups / launches | Overrun s | Plateau idle fraction | Trace rows |
|---:|---|---:|---:|---:|---:|---:|
| 1 | C018 | 180.014822304 | 2536 / 20288 | 0.014822304 | 0.0000509519 | 149 |
| 2 | C108 | 180.018029828 | 1931 / 15448 | 0.018029828 | 0.0000372531 | 149 |
| 3 | C081 | 180.044842367 | 2153 / 17224 | 0.044842367 | 0.0000400675 | 149 |
| 4 | C081 | 180.039497076 | 2143 / 17144 | 0.039497076 | 0.0000397649 | 150 |
| 5 | C108 | 180.045607887 | 1915 / 15320 | 0.045607887 | 0.0000421283 | 148 |
| 6 | C018 | 180.047621608 | 2537 / 20296 | 0.047621608 | 0.0000452294 | 150 |

The 10-second overrun limit and 0.02 idle-fraction limit passed in every run.
All six CLI statuses and exit codes were successful. Docker listings before,
after, and after the ownership helper were empty for every run. Ownership
helpers returned zero; they changed ownership only so the complete artifacts
could be retained. No reset or rerun occurred.

Cooldown used the preceding run's pre-run snapshot, with
`power_w <= preceding pre power_w + 5 W` and ASIC temperature
`<= preceding pre temperature + 3 degC`, capped at 600 seconds. AICLK was not
a cooldown criterion.

| Run | Pre ASIC power W | Pre ASIC temperature degC | Gate limit W / degC | Gate observation W / degC |
|---:|---:|---:|---|---|
| 1 | 45 | 57.9 | N/A | N/A |
| 2 | 45 | 55.9 | 50 / 60.9 | 45 / 55.6 |
| 3 | 45 | 57.3 | 50 / 58.9 | 47 / 56.9 |
| 4 | 47 | 60.5 | 50 / 60.3 | 47 / 60.3 |
| 5 | 47 | 59.4 | 52 / 63.5 | 47 / 59.6 |
| 6 | 45 | 57.4 | 52 / 62.4 | 47 / 57.5 |

Run 4's 60.3 degC gate observation and its later 60.5 degC pre snapshot are
different captures and are reported without substitution. This relative gate
does not establish a common initial thermal state.

## Firmware model

The semantics used to preregister the hypotheses and decode the words are
pinned to repository `tenstorrent/tt-zephyr-platforms`, tag `v19.6.0`. The
six core files and their hashes are:

| Pinned source file | SHA-256 |
|---|---|
| `lib/tenstorrent/bh_arc/aiclk_ppm.c` | `641584562ad3da67e2a3c14f049debd5d2f9ef18a631538d44e6211c88a8beac` |
| `lib/tenstorrent/bh_arc/aiclk_ppm.h` | `7a82600e5d6bc8e1f909f3034daa7d6f08f6893ce3018c161327c6902fcb3fe4` |
| `lib/tenstorrent/bh_arc/throttler.c` | `22b9360ba05faaf0595faad151c99309049cea0a948363caa0355e2e58f7330e` |
| `lib/tenstorrent/bh_arc/throttler.h` | `37ddcc68a23bdcf6dacdf2f1aa447e203088e51f64fccad4dc7a2a96fcf081e9` |
| `lib/tenstorrent/bh_arc/telemetry.c` | `7504ffdc8b5883bba80cdedd4e8b725a98f5d8aa396603a6b71f0d8053d3912f` |
| `lib/tenstorrent/bh_arc/telemetry.h` | `04833aed46893cb9abedf8490fa38dd96dd179749e80477df6c53b5d5a2a9e6d` |

The five clock-control source hashes are:

| Pinned source file | SHA-256 |
|---|---|
| `drivers/clock_control/clock_control_tt_bh.c` | `17fc86d3f36389d695ccc833b52a7aaebdad6905eb192af1ddbc1812b6ecb012` |
| `include/zephyr/drivers/clock_control/clock_control_tt_bh.h` | `6bd38a2c589ee0caefc73f46dac495f1581c96b7d5ce9dcf5c29d7c1ed3c5b9a` |
| `lib/tenstorrent/bh_arc/pll.c` | `7fe42813df13e922be98a5cd3ef602906f7e1a44aeeeba8ec94144d320adc1cd` |
| `lib/tenstorrent/bh_arc/pll.h` | `1f584cab38938207f00c24b4ee1e4c4c027f02d228e0cf225d9fca2671b05d17` |
| `dts/vendor/tenstorrent/tt_blackhole_smc.dtsi` | `a3e2e2571f66d81546c8affd5eda2804999345111a7f4e5c915b3ad51c75d906` |

The pinned model is a slow Doppler board-power regulator, not a board-power
hard cap. `INPUT_POWER` is DMC-reported board power plus the 20 W additive
term. Every 1-ms DVFS tick uses the integer 1000-sample moving average,
`error = (300 - average) / 300`, and
`output = 0.0025 * error + 0.3 * (error - previous_error)`. The
`doppler_slow` maximum arbiter adds `output * 500` and is clamped to
[800, 1350] MHz. Its integral action predicts a settled integer-floor average
in [300, 301) W when the load can reach the budget. Thermal arbiters remain
below their limits here; TDP, TDC, and board-power arbiters are disabled under
`doppler_en`.

The boot masks are `ENABLED_MAX_ARB=0x1d1` and `ENABLED_MIN_ARB=0x3`.
`AICLK_ARB_MAX` reports a truncated effective maximum and an index whose
update compares a uint32 value with float arbiter values. Consequently the
reported index can be ambiguous (for example, index 7 can be the Doppler
arbiter at a non-integral value or the GDDR thermal arbiter at an integral
value). The verdict therefore combines the independent mask, frequency, busy,
and thermal conditions rather than treating the index alone as causal.

The PLL/control model uses a 6.25-MHz feedback-divider grid. The static
signed residual criterion is 0–6 MHz for the integer words, with one extra MHz
allowed for a telemetry read straddling an update, hence R5's 0–7 MHz bound.
A one-watt moving-average step contributes approximately -0.5 MHz through the
derivative term and the model predicts recovery in about 2 ms. This **about
2-ms value is a source-model recovery estimate, not a measured pulse width**.
Likewise, the control dip word `0x70545` and reported AICLK 1343 are model
pairs, not an instrumented firmware timing measurement.

## Secondary

These summaries are descriptive and never replace the per-run registered
criteria. The pooled rows concatenate samples by configuration, omit the
cross-run pair from the lag-1 numerator, and use a 95% interval.

| Configuration | n | Mean W | SD W | r1 | n_eff / df | 95% CI W |
|---|---:|---:|---:|---:|---|---|
| C018 | 194 | 259.489691 | 6.222340 | -0.012762 | 194.000000 / 193 | [258.608575, 260.370806] |
| C081 | 194 | 299.479381 | 20.113978 | 0.245850 | 117.434064 / 116 | [295.803145, 303.155618] |
| C108 | 193 | 303.145078 | 18.102132 | -0.056275 | 193.000000 / 192 | [300.575006, 305.715150] |

The duration-weighted left-held board-power means from the verified reporting
helper are a cross-check only:

| Run | Arithmetic mean W | Duration-weighted mean W |
|---:|---:|---:|
| 1 | 259.165 | 259.151 |
| 2 | 302.371 | 302.293 |
| 3 | 298.959 | 298.908 |
| 4 | 300.000 | 299.980 |
| 5 | 303.927 | 304.330 |
| 6 | 259.814 | 259.855 |

The official verdict remains the arithmetic, per-run estimator in
[analysis.json](analysis.json); no weighted value changes H-DS.

## Exploratory

### Current dips

For regulated runs only, a current dip is strictly below 0.8 times that run's
plateau current median. These rows are diagnostics, not TDC evidence or new
gates.

| Run | Configuration | Dips / n | Dip board power W | Dip ASIC power W | Nearest group-boundary distance ms |
|---:|---|---:|---|---|---|
| 2 | C108 | 13 / 97 | 247–323 | 65–102 | 5.780–46.148 |
| 3 | C081 | 11 / 97 | 224–301 | 94–107 | 3.039–32.159 |
| 4 | C081 | 14 / 97 | 248–323 | 90–108 | 2.529–41.290 |
| 5 | C108 | 9 / 96 | 243–324 | 86–101 | 14.599–37.655 |

All 47 dip captures fell inside synchronized group intervals; none fell in a
host-visible group gap. Capture and firmware-update timing overlap the dense
group schedule, so boundary distance is descriptive and does not establish a
dispatch cause or causality.

### Control dips and drift

Run 1 had 1/97 control-dip samples, at 266 W board power; run 6 had 0/97.
The model-allowed pair and strict-majority rule remain the only control
criteria. The 1/97 and 0/97 counts are not a measured dip duration or period.

| Run | AICLK first → last MHz | AICLK range MHz | ASIC temperature first → last degC | GDDR temperature first → last degC | Fan range rpm |
|---:|---|---|---|---|---:|
| 1 | 1350 → 1350 | 1343–1350 | 77.0 → 78.8 | 60 → 64 | 3137–3356 |
| 2 | 1031 → 1012 | 1012–1031 | 80.6 → 81.7 | 60 → 64 | 3664–3870 |
| 3 | 1143 → 1131 | 1131–1143 | 81.1 → 81.9 | 60 → 64 | 3664–3934 |
| 4 | 1137 → 1131 | 1131–1143 | 81.4 → 82.1 | 64 → 66 | 3779–3902 |
| 5 | 1018 → 1012 | 1006–1018 | 81.2 → 82.3 | 64 → 66 | 3750–3902 |
| 6 | 1350 → 1350 | 1350 | 77.1 → 78.7 | 60 → 64 | 3116–3333 |

### Onset and post-load words

The official analyzer's exploratory clock-onset display uses a plateau-only
decoded map when inspecting the full trace. That source bug can miss a
pre-plateau depression in runs 2–5. It is tracked as [Issue
#149](https://github.com/tetsuh/hekatus/issues/149) and has no verdict impact.
The official JSON is unchanged. The separate raw-trace supplement applies the
registered predicates to the raw workload rows and reports first observed
onsets of **1.522849722, 1.901850963, 2.020702915, and 1.776471635 seconds**
for runs 2–5, respectively, versus the formal display's approximately
60-second observations. These are capture-grid first observations, not exact
transition times.

| Run | Post-load AICLK_ARB_MAX word | Index / frequency MHz | Capture delay s | Plateau mean AICLK MHz |
|---:|---|---|---:|---:|
| 1 | `0x80546` | 8 / 1350 | 0.558017572 | 1349.928 |
| 2 | `0x70443` | 7 / 1091 | 0.544689685 | 1017.536 |
| 3 | `0x704b2` | 7 / 1202 | 0.509923698 | 1136.938 |
| 4 | `0x704d2` | 7 / 1234 | 0.539176001 | 1134.711 |
| 5 | `0x70455` | 7 / 1109 | 0.528947507 | 1013.375 |
| 6 | `0x80546` | 8 / 1350 | 0.621389673 | 1350.000 |

The post-load snapshots are idle observations: AICLK itself is 800 MHz and the
minimum arbiter is idle, not the loaded busy value. These release words and
frequencies are exploratory only.

## Interpretation boundaries

This is one physical board and one approved six-run series. The per-run
intervals use an AR(1)-like serial-correlation approximation, and the telemetry
is discretely sampled at approximately 1.24-second cadence rather than
continuously instrumented. The result is conformance of the registered
observables to the pinned model; it is not firmware-code-path instrumentation
or proof that the source executed on every tick.

The relative cooldown gate controls preceding ASIC power and temperature but
does not establish a common initial thermal state. In particular, the run 4
gate observation was 60.3 degC while the separate pre snapshot was 60.5 degC;
both values remain as recorded. Current dips, onset times, post-load words,
fan/temperature drift, and the reported index ambiguity do not establish
causality. The disabled 200-A TDC limit is not a verdict mechanism in this
firmware configuration.

## Smoke diagnostics

Smoke artifacts are retained for execution-history and acceptance-history
reproducibility. None enters the six-run H-DS verdict.

### Revision 4, 2-second cadence

The `revision4/` namespace preserves the three completed C018, C081, and C108
run directories plus the original `successful3-analysis.json` and
`successful3-summary.json`. The C018 smoke acceptance passed its then-current
gates. C081 and C108 failed the then-current/board-power-SD and R5 smoke items;
the failure analysis led to revision 5. These are historical diagnostic
results, not long-run evidence.

### Revision 5, 1-second cadence

The `revision5/` namespace preserves all three complete run directories and
the original `smoke-analysis.json` and `summary.json`. Revision 5 passed the
load-contract, arbiter-word, and revised R5 items for C081 and C108, while C018
failed the old R7 all-clamped rule because two of 24 window samples were the
model-predicted `0x70545`/1343 pair. That observation led to revision 6; no
long-run verdict was changed by the smoke result.

### Revision 6 replay lineage

The historical [smoke-analysis-final.json](smoke/revision6/smoke-analysis-final.json)
is byte-preserved from the revision-6 replay of the same revision-5 records.
Its `evaluation.plan_sha256` is
`1c46dae48245ee7e12f00f8c3717d6f5101bbae6b978db7c0ddb9b4d7addbf2e`, the
revision-6 plan before the final wording/pin refresh. It must not be confused
with the final plan canonical SHA above.

Because no pin-refresh replay artifact existed, this archive also contains
[smoke-analysis-final-current-plan.json](smoke/revision6/smoke-analysis-final-current-plan.json),
a new board-free derivative replay of the three archived revision-5 records
using the current immutable revision-6 plan. It is not a new acquisition. Its
plan SHA is `28a9e96b026f74243ee769ce61436e2c10d1b883c5a84e9bf6306252a85bf324`;
all 12 smoke gates pass, the strict control majority is 22/24, and
`affects_long_verdict` is false.

### Initial import failure

The `initial-import-failure/c018-attempt1-import-failure/` namespace preserves
the complete failed first-import directory, including diagnosis, snapshots,
raw one-row sampler evidence, Docker event evidence, and the failed analyzer
output. The persisted host record stopped with no ingested runner result. The
owner's later board-free diagnosis reproduced a missing SciPy import in the
pinned image; the persisted original evidence alone classifies whether a
workload opened the device as unknown. The preserved diagnosis retains its original generic repository-relative
command string unchanged; it is historical evidence, not a current replay
command. This failure is diagnostic only and is not included in the six-run
verdict.

## Reproduction and integrity

The archive is self-contained for board-free analysis. The official result and
raw onset supplement are immutable copies; all run records resolve their
companion, group, snapshot, trace, and wrapper paths within their archive run
directory. `manifest.json` lists every archived artifact's deterministic
archive-relative path, SHA-256, and byte count, including this README and all
smoke artifacts, while excluding itself to avoid a recursive hash.

The following command reproduces the six-run analysis without a board or
telemetry utility and writes only an ephemeral file outside the archive:

```bash
export PYTHONDONTWRITEBYTECODE=1
uv run python tools/analyze_issue146.py \
  --plan docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json \
  --record docs/measurements/issue146-doppler-conformance/2026-10-21/runs/run1-c018/run-issue146-long-run1-c018.json \
  --record docs/measurements/issue146-doppler-conformance/2026-10-21/runs/run2-c108/run-issue146-long-run2-c108.json \
  --record docs/measurements/issue146-doppler-conformance/2026-10-21/runs/run3-c081/run-issue146-long-run3-c081.json \
  --record docs/measurements/issue146-doppler-conformance/2026-10-21/runs/run4-c081/run-issue146-long-run4-c081.json \
  --record docs/measurements/issue146-doppler-conformance/2026-10-21/runs/run5-c108/run-issue146-long-run5-c108.json \
  --record docs/measurements/issue146-doppler-conformance/2026-10-21/runs/run6-c018/run-issue146-long-run6-c018.json \
  --output /tmp/issue146-long-reanalysis.json
```

The replay must report `status=ok`, classification `H-DS`, no missing fields,
and no failed criteria. The performed replay reported those values and wrote a
14,824,165-byte result whose SHA-256 is
`2d1537fea1d40d8964b0e0fa145808274b733c89c6cd382f5964103314656445`; it was
byte-for-byte equal to the archived `analysis.json`. Thus there were no
remaining path-only differences in this analyzer output; archive-relative
source paths, if emitted by a future implementation, are not measurement
inputs.

The smoke replay can likewise be reproduced from the three archived revision-5
records with `--smoke`; it is diagnostic and must report 12/12 gates passed
under the final plan. No command in this reproduction section opens a device,
uses Docker, invokes tt-smi, changes a source artifact, or changes a threshold.
