# Measurements

Raw results kept as data, each with the reproducibility information that
produced it.

A throughput figure is only evidence if the environment behind it can be
reproduced later. design.md §2 records that a firmware update once changed
the core count, and the first measurements on this project ran on a different
board from the target — so every accelerator-backed result file here carries,
in its own `environment` block, the board type and serial, firmware bundle,
kernel driver version, the toolchain image by digest, and the revision of the
harness that computed the numbers. For accelerator-backed records,
`environment.tt_env_active_release` is host-side context captured before the
pinned container starts; it is not the release identity of the container.
The immutable image digest in `environment.image` is authoritative for the
container and toolchain identity. A host-side measurement — one the
reference implementation takes with no board involved — carries the
platform, machine architecture, CPU, Python / NumPy / SciPy versions and the
harness revision instead, and says `"board": null`; it does not require a
host identity. A companion trace is plain data with no such block: it
inherits its provenance from the result file sharing its filename stem, and
is meaningless apart from it.

Naming: `YYYY-MM-DD-<board>-<what-was-measured>.json`, with any companion
trace beside it under the same stem.

## Issue #102 raw-record rebuild

The Issue #101 runner writes a self-contained raw run artifact before record
construction. Its exact supported schema is
`adr-0005-issue101-combined-raw-v1`; the artifact carries the run ID, captured
normalized environment, run rows, and captured power samples. Rebuild is
board-free and reads no `env-*.json`, `power-*.csv`, current environment, or
clock at rebuild time:

```bash
uv run python tools/newton_schulz_issue101_combined.py \
  --recover-raw /path/to/issue101-combined-raw.json \
  --output /path/to/rebuilt-record.json
```

When a retained output directory contains more than one raw run, select the
embedded identity explicitly. The command validates the raw filename and
embedded identities, schema, captured timestamps, harness commit, digest-pinned
image, board identity, result rows, and power-sample shape before writing:

```bash
uv run python tools/newton_schulz_issue101_combined.py \
  --recover-raw /path/to/reused-output-directory \
  --run-id <captured-run-id> \
  --output /path/to/rebuilt-record.json
```

Missing or conflicting identities, mixed or ambiguous raw artifacts, unknown
schema values, incomplete captured fields, and an existing destination are
rejected. The raw input is never overwritten; the power filename in the rebuilt
ADR-0005 record is provenance from the saved raw value, not a request to reread
a mutable capture file.

## Issue #118 board-free calibration configuration

The board-free CLI emits one calibration record or one run configuration and
never opens a device. A calibration is watcher-off and capped at 60 seconds;
its `selected_repeat_count` is the only repeat count to carry into the later
run. The target duration may describe the later long measurement, while the
calibration observation itself remains capped:

```bash
uv run python -m enodia.tt.bench.run_load \
  --condition newton_schulz --calibrate --duration-seconds 60 \
  --target-seconds 500 --achieved-throughput 100 \
  --out calibration-newton-schulz.json
uv run python -m enodia.tt.bench.run_load \
  --condition newton_schulz --run-config \
  --calibration-json calibration-newton-schulz.json \
  --out run-config-newton-schulz.json
```

The throughput argument is the Watcher-off measurement reported by the
board-free configuration path; it does not claim board evidence. The parent
runs the bounded board calibration once per device load with no Watcher and a
60-second cap using the resident runner. The emitted calibration record
retains the measured throughput and the calibration-selected repeat count for
the later run. A later loaded resident invocation requires an explicit
outer-timeout cap: 600 seconds for control/DRAM, or 800 seconds for the
compute-loaded Newton-Schulz condition when its calibrated load-time estimate
needs the amended allowance. The runner never selects the cap automatically.
Watcher remains capped at 60 seconds. The selected cap and its provenance are
persisted in the resident record; the
wrapper rejects a value above the selected cap. A later device record is not
evidence until it adds a wrap-tracked card-clock load interval for
`newton_schulz`/`dram`, or a host monotonic interval for `host_cpu_transfer`,
and the corresponding coverage verdict is `covered`.

The immutable power snapshot retains every usable in-run sample, the in-run
minimum and maximum, and a left-held time-weighted AICLK over the explicit
load interval. `clock.aiclk_mhz` remains the in-run maximum for public
compatibility. For loaded DRAM/Newton-Schulz records, the separate
`clock.aiclk_mhz_for_elapsed_seconds` value is the observed weighted
load-interval AICLK; conservative prediction may instead select the recorded
in-run minimum. Sampled or unloaded resident records may intentionally retain
`in_run_maximum` or `configured_budget` as their elapsed-time selection, with
that provenance persisted in `clock.aiclk_source_for_elapsed_seconds` and its
semantic selection. Missing load-interval bounds or provenance fail closed.
The resident record's AICLK observation is a post-run check: preflight never
substitutes a snapshot or configured value for the required post-run
power-trace observation.
Card-load calibration throughput keeps the historical `repeats_per_card_second`
spelling but is a separate fixed-reference field: repeats per 1.35e9 card-clock
ticks at 1350 MHz, not repeats per weighted real-time second. Later conversion
of those card ticks to loaded elapsed wall time uses the observed weighted
load-interval AICLK, or the recorded conservative in-run minimum when prediction
policy selects it. The record also retains host `CLOCK_MONOTONIC` program
launch/completion beside the card tick spans. DRAM records retain all eight
worker rows and use the slowest worker's own elapsed interval for calibration;
the aggregate worker span is coverage metadata only.
Condition (c)'s fixed host repeat is 4,096 rounds of 4,096-word arithmetic
plus 256 fixed 64 KiB host/device transfer pairs on CQ1. The retained host-load
implementation is a #120 starting point only: non-calibration host runs are
rejected and cannot produce Issue #118 evidence. A fixed four-slot reusable
host/device pool and per-repeat CQ1 completion keep transfer buffering bounded
while targeting approximately one-second calibration granularity; the work
remains data-independent. A calibration record may therefore be
`uncovered` when its single measurement repeat finishes before the full resident
interval; it remains `status=calibration` and `valid_evidence=false`. The
calibration interval is not promoted to timing evidence, while the later
calibrated run must still provide covered interval metadata.

## Issue #118 pre-measurement prediction

This prediction is committed before the same-commit device measurements. The
board-free regression command for this contract is
`uv run pytest -q -m "not slow"`; it performs no device, board, Docker-device,
Watcher, or reset operation. The three in-scope runs use device 0, 500,000
frames, a 1 ms period, no Watcher, two command queues, and no sub-device
manager: CQ0 carries the resident program and CQ1 is reserved for transfer-only
work. The control and DRAM runs use a 600-second wrapper cap; the
AICLK-reducing Newton-Schulz run uses the approved 800-second cap. The host
CPU plus PCIe condition (c) is moved to a separate follow-up Issue.

The prediction revision is harness commit
`2734d88a4aefd4ab56c6447d3f7f360408abbd42`. Both calibrations were rerun on
that commit with sampler on, Watcher off, and a 60-second cap. The historical
`repeats_per_card_second` value is a fixed-reference throughput: repeats per
1.35e9 card ticks at 1350 MHz, not repeats per weighted real-time second.
Therefore the reference prediction is 550 seconds (500-second target × 1.10),
and loaded elapsed time is obtained by converting the card ticks with the
observed load-interval weighted AICLK, or the recorded conservative in-run
minimum when prediction policy selects it. NS is conservatively projected at
sustained 1025 MHz, based on the observed AICLK decline; DRAM uses its
calibration weighted AICLK. The measured NS gate used a manual 1025 MHz
projection; this PR's code guard now uses the conservative minimum of weighted
and in-run-minimum AICLK. The resident schedule is 500.1 seconds and the planning allowance for device open,
JIT, cleanup, and close is 4 seconds. Loaded work overlaps the resident
program, so load and resident times are not added.

| condition | calibration repeat | weighted load AICLK (MHz) | reference throughput (repeats / 1.35e9 ticks) | predicted load time (s) | cap (s) | predicted whole-run time (s) | margin (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| unloaded control | — | — | — | — | 600 | 504.1 | 95.9 |
| Newton-Schulz | 333,688 | 1,191.259 | 606.704754 | 724.391 (1025 MHz projection) | 800 | 728.391 | 71.609 |
| DRAM | 2,552,005 | 1,346.531 | 4,640.008824 | 551.417 | 600 | 555.417 | 44.583 |

Calibration provenance: the NS run used 18,000 repeats and 40,052,430,488
card ticks; the DRAM run used 138,076 repeats and its slowest worker interval
was 40,172,897,738 card ticks (aggregate span 42,081,820,472 ticks). Both
records are `status=calibration`, `valid_evidence=false`, and uncovered by
design. These are pre-measurement predictions, not evidence. The control and
DRAM predictions exceed the required 40-second margin and are below the
<=560-second prediction gate for their 600-second caps. Newton-Schulz uses the amended
800-second cap and a <=760-second prediction gate; its predicted whole-run time
is 728.391 seconds. The three runs may proceed only on this pushed commit.

## Issue #118 measured results (2026-10-11)

The accepted three-run sequence used the same harness commit
`2e143f872e5439eef223d62d43acfc0b2aa68dec` for the unloaded control, DRAM,
and Newton-Schulz runs. All runs used device 0, 500,000 frames, a
1,350,000-tick period, two command queues (CQ0 resident and CQ1 transfer), no
sub-device manager, sampler-on at the default 2-second interval, and Watcher
off. The control and DRAM runs used a 600-second cap; Newton-Schulz used the
approved 800-second cap. All three runs terminated by frame count with
500,000 produced and consumed frames, zero overflow/dropped/aborted attempts,
and zero paired outliers.

The calibration records retain the parent executable commit
`2734d88a4aefd4ab56c6447d3f7f360408abbd42`; the later prediction commit changed
documentation only. The NS calibration used 18,000 repeats and 40,052,430,488
card ticks, weighted AICLK 1,191.259 MHz, throughput 606.704754 repeats per
1.35e9 ticks, and selected repeat 333,688. The DRAM calibration used 138,076
repeats, a slowest-worker interval of 40,172,897,738 ticks, weighted AICLK
1,346.531 MHz, throughput 4,640.008824 repeats per 1.35e9 ticks, and selected
repeat 2,552,005. Both calibration records were `status=calibration`,
`valid_evidence=false`, and intentionally uncovered.

| condition | frame percentiles (ticks: P50 / P99 / P99.9 / P99.99 / max) | same percentiles converted to ms | control delta (ticks; P50 / P99 / P99.9 / P99.99 / max) | load coverage | load evidence | AICLK min / max / weighted (MHz) |
|---|---|---|---|---|---|---|
| unloaded control | 1,349,988 / 1,350,061 / 1,350,065 / 1,350,065 / 1,350,091 | 0.999991 / 1.000045 / 1.000048 / 1.000048 / 1.000067 | — | not applicable | no load | 1,350 / 1,350 / 1,350 reference |
| DRAM | 1,349,996 / 1,350,102 / 1,350,139 / 1,350,177.0002 / 1,350,261 | 1.000114 / 1.000193 / 1.000220 / 1.000248 / 1.000311 | +8 / +41 / +74 / +112.0002 / +170 | covered | 8 workers; each read/write 653,313,280 pages; repeat 2,552,005 | 800 / 1,350 / 1,349.842 |
| Newton-Schulz | 1,350,002 / 1,350,076 / 1,350,120 / 1,350,166 / 1,350,259 | 1.284598 / 1.284669 / 1.284710 / 1.284754 / 1.284843 | +14 / +15 / +55 / +101 / +168 | covered | fixed 108-core non-overlapping partition; selected TRISC0 profile interval; 1,001,064 effective outer repetitions; fixed 12-iteration profile | 1,018 / 1,350 / 1,050.914 |

The Newton-Schulz record's selected TRISC0 profile reports 1,001,064
warmup events, the same count as `3 × 333,688` executed outer repetitions.
The profile therefore accumulates its warmup counters across the repeated fixed
work; it is not evidence of a single one-time warmup. The fixed-work evidence
remains valid because the runtime loop still executes exactly three outer
repetitions per selected repeat and twelve iterations per outer repetition. A
kernel-side correction to warmup counter scope is a follow-up candidate and is
not applied to this measured record.

The converted values use each record's `aiclk_mhz_for_elapsed_seconds`, with
the paired `aiclk_source_for_elapsed_seconds` naming whether the selection is
configured, in-run maximum, time-weighted load interval, or the conservative
in-run minimum. Loaded DRAM/Newton-Schulz records use the time-weighted load
interval for elapsed conversion; sampled or unloaded resident records may
retain `in_run_maximum` or `configured_budget`. The P50 converted difference
from the control is +0.000123 ms for DRAM and +0.284607 ms for Newton-Schulz.

### Prediction comparison

| condition | predicted load / whole-run (s) | observed host program / wrapper (s) | whole-run difference (s) |
|---|---:|---:|---:|
| unloaded control | — / 504.100 | 500.320 / 503.753 | -0.347 |
| DRAM | 551.417 / 555.417 | 550.333 / 553.924 | -1.493 |
| Newton-Schulz | 724.391 / 728.391 (1025 MHz projection) | 707.446 / 711.979 | -16.412 |

### Tick-to-host-time cross-check

The host interval is the recorded `CLOCK_MONOTONIC` program interval. The card
interval is the resident interval for control and the aggregate load interval
for loaded runs; the DRAM throughput calibration separately uses its slowest
worker interval.

| condition | card interval (ticks) | weighted AICLK (MHz) | ticks converted to seconds | host program interval (s) |
|---|---:|---:|---:|---:|
| unloaded control resident | 674,998,649,982 | 1,350.000 | 499.999 | 500.320 |
| DRAM load aggregate | 742,787,678,116 | 1,349.842 | 550.278 | 550.333 |
| Newton-Schulz load | 742,492,282,048 | 1,050.914 | 706.521 | 707.446 |

This conversion is an approximation: one time-weighted AICLK is used to
convert an entire interval and every frame percentile, although the clock is
not constant. In particular, Newton-Schulz moved between 1,018 and 1,350 MHz
and spent most of its loaded interval near 1,025 MHz. The interval and host
cross-check supports the changing-clock interpretation but does not establish
an instantaneous per-frame tick-to-wall-time mapping.

The frozen ADR-0005 records and their #111 power-trace snapshots are:

- `2026-10-11-p150a-issue118-control-500000-adr0005.json` and its companion
  `2026-10-11-p150a-issue118-control-500000-adr0005-power.csv`
- `2026-10-11-p150a-issue118-dram-500000-adr0005.json` and its companion
  `2026-10-11-p150a-issue118-dram-500000-adr0005-power.csv`
- `2026-10-11-p150a-issue118-newton-schulz-500000-adr0005.json` and its
  companion `2026-10-11-p150a-issue118-newton-schulz-500000-adr0005-power.csv`

Each companion is the immutable snapshot referenced by its record; the record
contains its SHA-256, sampler coverage, host monotonic interval, and in-run
AICLK trace. The host-load condition (c) implementation is retained as a #120
starting point, but it cannot produce #118 evidence; non-calibration host runs
are rejected. See [Issue #120](https://github.com/tetsuh/hekatus/issues/120).
The Issue #121 Q1/Q2 timing and budget conclusions are recorded in
[`docs/budget.md`](../budget.md) and [`docs/design.md`](../design.md); the
binding timing basis is [ADR-0015](../adr/0015-issue121-card-timing-basis.md).
The separate conversion audit remains in
[Issue #122](https://github.com/tetsuh/hekatus/issues/122). Full-enodia
sustained AICLK and limiter confirmation remain the #136 and #137 follow-ups.

Two earlier artifacts are excluded from this evidence set: the Newton-Schulz
run at `ece5d09a34304b1182ab3a74cdc7a8055bf36631` was killed by the 600-second
outer timeout (exit 137), and the unloaded control at that commit is excluded
because its harness commit differs from the accepted three-run sequence.
No pass/fail threshold is imposed on these measurements; the results are
observations for the latency and budget discussion.

### Watcher-enabled first confirmations

No board rerun is part of this correction. The existing first confirmations
are:

- Newton-Schulz — commit `f1a4eb8f803112559bf0030eb14d2f8a69c4a693`,
  2026-10-11, `status=ok`, 100/100 frames, Watcher enabled, 60-second cap.
- DRAM — commit `be11135b51f6b34402b30652a6bead5be2e6d688`, 2026-10-11,
  `status=ok`, 100/100 frames, Watcher enabled, 60-second cap.

For kernel-source provenance, `enodia/tt/bench/kernels/` has no diff from
`be11135b51f6b34402b30652a6bead5be2e6d688` to `2e143f8`. The Newton-Schulz
run record names `f1a4eb8f803112559bf0030eb14d2f8a69c4a693`; the Newton-Schulz
load-kernel files themselves are unchanged from that run to `2e143f8`. The
only later kernel-directory change before `be11135b` was the DRAM address fix,
unrelated to the Newton-Schulz confirmation. This distinction keeps the run commit and
the final kernel baseline traceable without claiming a false whole-directory
comparison.

## Issue #104 paired-outlier records

The wrapper supports the existing telemetry sampler as `default` (2 seconds),
`explicit` with a bounded `HEKATUS_TT_TELEMETRY_INTERVAL_S` in the supported
0.1–3600.0 second range, and `off` with no sampler or power trace. Sampler-off
output is diagnostic-only; sampled output retains its same-stem power trace.
New canonical outlier records contain numeric measurements; human conclusions
remain in this wrap-analysis document. The three Issue #104 runs below were produced
by kernel/harness commit `487bc36fa30e870b6cd9b2a62199275397cde004`, exactly as
recorded in each JSON record. That commit predates PR #103's synchronization-
protocol fix, so these runs must not be described as having used that fix.

| File | What it is |
|---|---|
| `2026-10-06-p150a-issue104-sampler-off.json` | 400,000-frame sampler-off diagnostic run: 22 paired outliers, no power trace by design, and `timing_evidence=false`. |
| `2026-10-06-p150a-issue104-sampler-default.json` | 400,000-frame run with the unchanged 2-second sampler: 18 paired outliers, N=399,999, P50/P99/P99.9/P99.99 = 1,349,988/1,350,048/1,350,050/1,350,050 ticks, min/max 26,829/2,673,133 ticks, and 181 power samples. |
| `2026-10-06-p150a-issue104-sampler-default-power.csv` | Companion power/AICLK/temperature trace for the default-sampler run. |
| `2026-10-06-p150a-issue104-sampler-5s.json` | 400,000-frame run with an explicit 5-second sampler: 9 paired outliers, N=399,999, the same percentile body, min/max 27,285/2,672,693 ticks, and 77 power samples. |
| `2026-10-06-p150a-issue104-sampler-5s-power.csv` | Companion power/AICLK/temperature trace for the explicit-interval run. |
| `2026-10-06-p150a-issue104-wrap-analysis.md` | Board-free supplement that verifies the raw format and pair endpoints, folds event phases by the exact `2^32`-tick wall-clock period, and analyzes the available Issue #12 500,000-frame artifact without changing any JSON or CSV leaf. |

### Issue #104 invariant revalidation

The shared board-free `validate_resident_record` field matrix and invariant
catalog in `enodia/tt/bench/resident_record.py` are scoped to records written
by the current runner, which carry top-level `resident_record_schema: 1`.
The final Issue #12 evidence record and all three committed Issue #104 records
predate that marker, so validation reports them as `out_of_scope` historical
records rather than invalid records. No JSON or CSV leaf is rewritten. A new
runner-shaped record still receives the ordinary required-field, type/range,
and relationship checks; full record-kind consistency and agreement between
declared coverage flags and parsed trace facts remain deferred to Issue #110.

The board-free analyzer still analyzes the two historical sampled records even
though they have no explicit `run_start`/`run_end` bounds or persisted
`aiclk_source`: it uses each record's persisted `clock.aiclk_mhz` as the
historical maximum-based, `legacy_unverified` elapsed-time provenance. It
never treats the public clock field as a new-record conversion policy and
never selects an environment
snapshot AICLK, and this historical path is not timing evidence. The historical
sampler-off record uses its configured budget AICLK for elapsed-time display and
remains diagnostic-only. The external raw-timestamp-byte limitation applies to
all three records because the referenced `raw-timestamps.bin` files are not
committed, although the sampled records retain their committed count and
histogram relationships.

The wrap analysis gives `2^32 / 1,350,000,000 = 3.1814572563` seconds,
or 3,181.4572563 configured 1-ms frames, so a frame-gap gcd of 1 does not
reject a strict wall-clock period. Pair starts in all three runs have Rayleigh
R `0.9999999999999996`–`0.9999999999999998` and circular widths
`0.0000296296`–`0.0000422222 ms` immediately after the wrap; the short interval
comes first and the long catch-up interval follows. The same phase lock in the
off run means a host sampler is not necessary. Together with PR #103's
controlled low-word-only clock experiment, the paired events are phase-locked to
the wall-clock wrap and disappear after switching to low-word-only reads with
software wrap tracking. This supports the clock-read path as the cause. Direct
producer-versus-consumer attribution remains open because there is no producer
stamp on the same clock. The short-first order is consistent with an early
producer send, but does not establish producer-side attribution. A shared
tile-latch overwrite remains a hypothesis only.

The `2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json` record
is the **intermediate controlled comparison**: it has `pair_count=0` and
min/max `1,349,924..1,350,068` ticks after the low-word-only clock change, but
it is superseded for accepted Stage 1 evidence because the later drain decision
fix was not yet present. The
`2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005.json` record is
the **final authority**: it includes the corrected drain/provenance path, has
N=499,999, P50/P99/P99.9/P99.99 = 1,349,988/1,350,061/1,350,065/1,350,065
ticks, min/max `1,349,889..1,350,065` ticks, and zero paired outliers. Both
records remain immutable; no Issue #104 measurement JSON or CSV is rewritten.

## Issue #12 recovery runbook (owner approval required)

The resident harness must not return to hardware without a new owner approval.
When approved, the bounded procedure below uses device 0 through the
named-container wrapper. Each resident invocation uses a separate explicit host
output directory. The wrapper exports a run-specific `HEKATUS_TT_RESULT_PATH`
inside the container; the resident runner uses that path when `--out` is absent.
The mounted output directories are retained on the development machine.

### Cleanup gate

Before and after each invocation, run the following shell inspection command and
require zero running containers. Never stop another container:

```bash
docker ps --format '{{.Names}}'
```

Prior cycle-budget records collapsed producer/consumer causes and did not
serialize elapsed/limit ticks, so their startup attribution remains uncertain;
new runs serialize named failure checks and diagnostics.

### Watcher validation

After the owner approval and a clean cleanup gate, run one Watcher-enabled,
one-frame validation on device 0 with a 1 ms harness interval and a separate
60-second outer cap. It is not timing evidence. The explicit invocation is:

```bash
TT_METAL_WATCHER=1 HEKATUS_TT_RUNNER=enodia/tt/bench/run_resident.py \
  enodia/tt/bench/run_in_container.sh out/bench/issue12-watcher -- \
  --device-id 0 \
  --frame-count 1 \
  --frame-interval-ticks 1350000 \
  --budget-aiclk-mhz 1350 \
  --outer-timeout-seconds 60 \
  --watcher
```

Leave `HEKATUS_TT_CONTAINER_TIMEOUT_S` unset so the wrapper selects the
mode-specific 60-second cap. The wrapper owns the run-specific result path through
`HEKATUS_TT_RESULT_PATH`; the command names the runner, device, frame count,
interval, clock used for the conversion, outer cap, and Watcher mode instead of
relying on runner defaults. Supplying `--out` is optional, and if supplied it
must identify the same path as the environment contract.

### No-Watcher timing

Only if the Watcher validation passes, and still under the owner approval, run
one no-Watcher timing attempt on device 0. It uses 500,000 frames, a 1 ms
harness interval at the documented 1,350 MHz clock, and a 600-second outer cap.
It writes the result and raw timestamps to container-internal `/out` paths:

```bash
env -u TT_METAL_WATCHER HEKATUS_TT_RUNNER=enodia/tt/bench/run_resident.py \
  enodia/tt/bench/run_in_container.sh out/bench/issue12-timing -- \
  --device-id 0 \
  --frame-count 500000 \
  --frame-interval-ticks 1350000 \
  --budget-aiclk-mhz 1350 \
  --outer-timeout-seconds 600 \
  --raw-timestamps-out /out/issue12-timing-500000.bin
```

The margin-inclusive run budget must fit the 600-second cap; there is no
override for a schedule that exceeds it. `env -u TT_METAL_WATCHER` and the
absence of `--watcher` are both intentional: this is the no-Watcher timing
run, not a second Watcher validation.

### Record creation and output retention (host-side shell commands)

The wrapper bind-mounts each explicit output directory at container `/out`.
On the development machine, verify the resident JSON records, raw timestamp
bytes, and the wrapper's environment and power provenance before analysis.
Retain the run-specific `runner-result*.json` files in
`out/bench/issue12-watcher` and `out/bench/issue12-timing`, as well as
`out/bench/issue12-timing/issue12-timing-500000.bin` files:

```bash
find out/bench/issue12-watcher out/bench/issue12-timing -maxdepth 1 -type f \
  \( -name 'runner-result*.json' \
     -o -name 'issue12-timing-500000.bin' \
     -o -name 'env-*.json' \
     -o -name 'power-*.csv' \) -print
mkdir -p issue12-retained
cp -a out/bench/issue12-watcher out/bench/issue12-timing issue12-retained/
sha256sum issue12-retained/issue12-timing/runner-result*.json \
  issue12-retained/issue12-timing/issue12-timing-500000.bin
```

The raw timestamp file is a companion to the timing result and is copied to
the development machine before any analysis. Keep each result, its matching
`env-*.json`, matching `power-*.csv`, and any raw timestamp companion together;
these outputs are evidence only when their provenance remains together.

### Abnormal exit or timeout: one-reset recovery

On an abnormal exit or timeout, stop the planned run and do not start another
hardware run. After the owner approves recovery, perform at most one reset of
device 0, then run exactly one fixed-image Stage-1 health probe. Inspect the
container state before and after recovery as well:

```bash
docker ps --format '{{.Names}}'
tt-smi -r /dev/tenstorrent/0
env \
  TT_METAL_WATCHER=1 \
  HEKATUS_TT_RUNNER=tools/newton_schulz_bringup.py \
  HEKATUS_TT_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621 \
  HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  enodia/tt/bench/run_in_container.sh -- \
  --stage 1 \
  --device-id 0 \
  --watcher \
  --timeout 60
docker ps --format '{{.Names}}'
```

The health command explicitly selects the fixed image, Stage 1, device 0,
Watcher mode, and its 60-second runner and container caps. If the health probe
fails, stop hardware work; after this one recovery attempt, stop hardware work
even if it passes. Do not perform a second reset. A cycle-budget error or outer
timeout may terminate a resident run earlier than its frame count, and those
runs are not timing evidence unless the recorded counters and raw timestamps
support the claim.

Successful non-error runs are frame-count terminated. The frame interval is a
harness parameter, not an acquisition-rate claim. For no-Watcher timing under
the 600-second cap, the explicit startup, overlap, and 10% margin formula gives
a maximum safe count of 545,354 frames at both 800 and 1350 MHz for the
corresponding 1 ms tick conversion; 545,355 is rejected. The separate first
Watcher validation remains capped at 60 seconds. The next approved timing rerun
must stay within 545,354 frames at 1 ms and must copy raw timestamps back to the
development machine for retention before analysis. The owner-requested
60,000-frame, 1 ms, 60 s configuration is preserved as a historical rejection
record with its 54,445-frame boundary. If a future 60,000-frame run is approved
with a suitable cap, P99.9 has sufficient N (>=20,000) while P99.99 remains
insufficient (N < 200,000). The run budget includes an explicit 100 ms startup
allowance converted at the configured AICLK; Watcher validation also has its
separate explicit overhead margin. If the ring is full, the producer uses
drop-new: the attempted frame is counted as overflow/dropped, never waits and
never overwrites an occupied slot. Consumer-completion histograms exclude
dropped attempts, so their N and interval samples must be interpreted beside
the attempted/produced/consumed/dropped counts. This is a procedure only: it
does not authorize a future device run by itself.

## Newton-Schulz reference correction

Six landed Issue #63 JSON records retain the historical correctness wording
`enodia/spec Newton-Schulz`:

- `2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000.json`
- `2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000.json`
- `2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge.json`
- `2026-09-27-p150a-newton-schulz-l32-b8192.json`
- `2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000.json`
- `2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000.json`

The actual oracle for those comparisons is the independent NumPy
fixed-iteration reference in
`enodia/tt/bench/newton_schulz_reference.py`, not `enodia/spec`. The
numerical comparisons remain valid; this note corrects the reference name
without rewriting any landed JSON record. Future records and documentation
must use the independent NumPy reference name.

**Issue #94 record correction (unmerged record).** The correctness rows in
`2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck.json` were produced
by `enodia/tt/bench/run_matmul.py:_issue94_fixed_reference`, not by
`enodia/tt/bench/newton_schulz_reference.py`. The runner first builds
`original_R` with `run_matmul.py:_issue94_random_hpd_batch`, rounds its real
and imaginary float32 planes independently to BF16 with RNE through
`_issue94_bf16_round_complex`, and builds `X0 = I / ||original_R||_infinity`
through `_issue94_initial_value`. `_issue94_fixed_reference` then applies the
fixed twelve-step recurrence to the BF16-rounded R and that original-R X0;
the companion true-inverse comparison is
`np.linalg.inv(original_R.astype(np.complex128))`. All twelve row labels now
name the actual function. No measurement value, sample array, power trace, or
harness identity was changed.

The board-side acceptance catalogue is one named-container invocation:

```text
./enodia/tt/bench/run_in_container.sh -- --acceptance-catalogue
```

It records both L=32 and L=16 batch-8192 rows, 1,000 launches per row by
default, and passes the sibling `tt-smi` power/clock trace into the JSON
provenance. Device correctness tests use the same pinned context through
`run_in_container.sh --pytest`; host pytest deliberately skips the `tt_device`
marker unless the wrapper sets both `HEKATUS_TT_DEVICE_TEST=1` and
`HEKATUS_TT_PINNED_CONTAINER=1`.

| File | What it is |
|---|---|
| `2026-10-06-p150a-issue12-stage1-60000-preflight-rejected.json` | Board-free rejection of the owner-requested 60,000-frame/1 ms/60 s configuration; explicit calculations show a safe 54,445-frame maximum under the current cap. |
| `2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005.json` | ADR-0005 record for the corrected semaphore 60,000-frame run: complete zero-overflow timing statistics, provenance, hashes, and the non-acceptance work-extrema limitation. |
| `2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005-power.csv` | Companion power, AICLK, and temperature trace for the ADR-0005 60,000-frame record. |
| `2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005.json` | ADR-0005 record for the corrected semaphore 600,000-frame run: complete zero-overflow timing statistics, provenance, hashes, and the non-acceptance work-extrema limitation. Owner decision 2026-10-07: not timing evidence (ran beyond the approved 600 s cap); retained as a historical record. The configured timeout was 660s; cap correction commit `7fda4a1` is recorded separately. This README disposition takes precedence over the JSON `timing_evidence=true` field. |
| `2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005-power.csv` | Companion power, AICLK, and temperature trace for the ADR-0005 600,000-frame record. |
| `2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005.json` | Final 100-frame Watcher validation: 100/100 produced/consumed, zero overflow/cycle error, work ticks 1,073–1,084, and explicitly non-timing evidence. |
| `2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005-power.csv` | Companion power, AICLK, and temperature trace for the final Watcher validation. |
| `2026-10-06-p150a-issue12-stage1-final-500000-adr0005.json` | Historical final in-cap 500,000-frame no-Watcher timing record: N=499,999, P50/P99/P99.9/P99.99 = 1,349,988/1,350,048/1,350,050/1,350,050 ticks, min/max 111,119/2,588,915, work ticks 1,074–1,098, and zero overflow; superseded by `2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json` because synchronization and clock representation changed. |
| `2026-10-06-p150a-issue12-stage1-final-500000-adr0005-power.csv` | Companion power, AICLK, and temperature trace for the superseded historical 500,000-frame timing record. |
| `2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005.json` | Superseded current-wrap Watcher validation: N=99, timing evidence false, work ticks 1,063–1,073, zero overflow/cycle error. Superseded by the 2026-10-07 board-id-alias record because eaf6632 corrected the resident drain completion decision; this record remains unchanged. |
| `2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005-power.csv` | Companion power trace for the superseded current-wrap Watcher validation; retained unchanged. |
| `2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json` | Superseded current-wrap in-cap 500,000-frame timing record: N=499,999, P50/P99/P99.9/P99.99 = 1,349,988/1,350,033/1,350,059/1,350,061 ticks, min/max 1,349,924/1,350,068, work ticks 1,063–1,087, zero overflow, and zero paired short/long outliers. Superseded by the 2026-10-07 board-id-alias record because eaf6632 corrected the resident drain completion decision; this record remains unchanged. |
| `2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005-power.csv` | Companion power, AICLK, and temperature trace for the superseded current-wrap 500,000-frame timing record; retained unchanged. |
| `2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005.json` | Current corrected-drain 100-frame Watcher validation under the board-id serial-alias fix: 100/100 produced/consumed, N=99, P50=1,350,008 ticks, min/max 1,349,787/1,350,096, work ticks 1,063–1,073, zero overflow/cycle error, and timing evidence false. |
| `2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005-power.csv` | Companion power, AICLK, and temperature trace for the current board-id-alias Watcher validation (2 samples). |
| `2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005.json` | Current corrected-drain in-cap 500,000-frame no-Watcher timing record under the board-id serial-alias fix: N=499,999, P50/P99/P99.9/P99.99 = 1,349,988/1,350,061/1,350,065/1,350,065 ticks, min/max 1,349,889/1,350,065, work ticks 1,062–1,087, zero overflow, and zero paired short/long outliers. |
| `2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005-power.csv` | Companion power, AICLK, and temperature trace for the current board-id-alias 500,000-frame timing record (226 samples). |
| `2026-08-14-p150a-effective-efficiency.json` | The B2 measurement: 17 shapes x 2 dtypes x DRAM/L1 on one p150a, against the 332 TFLOPS BF16 peak. Summarized in docs/budget.md |
| `2026-08-14-p150a-effective-efficiency-power.csv` | Board power, clock, and temperature sampled through that run |
| `2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0.json` | Issue #65 full stock matmul catalogue sweep: 284 rows (190 successful, 94 failed), image digest `sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621` in its environment block. The four failed `batched_dram_sharded` rows for the batch-1024 L16/L32 shapes are superseded by the 2026-09-21 record below, and the two unbatched `dram_sharded` rows for beamspace B=16, 256 channels, 4096 pixels are superseded by the 2026-09-23 record; the other 278 rows remain authoritative. |
| `2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0-power.csv` | Power, clock, and temperature trace for the 0.75.0 full sweep (183 samples); its provenance is the matching result record above |
| `2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0.json` | Issue #65 targeted four-row rerun with the corrected eight-worker mapping. It supersedes only the batch-1024 L16/L32 BF16/FP32 `batched_dram_sharded_g7x1_k1_m1_n1_s1x1` failures in the 2026-09-20 full sweep: the BF16 replacements succeeded, while both FP32 replacements reached program compilation and failed because static circular buffers clashed with L1 tensor buffers. The file's `supersedes` block maps every prior row to its replacement. |
| `2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0-power.csv` | Power, clock, and temperature trace for the targeted superseding rerun (2 samples); its provenance is the matching result record above |
| `2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0.json` | Issue #65 targeted two-row rerun for the beamspace B=16, 256-channel, 4096-pixel unbatched `dram_sharded` rows after the corrected eight-worker mapping. It supersedes the BF16 and FP32 `dram_sharded_g4x1_k1_m1_n32_s1x4` rows in the 2026-09-20 full sweep with successful `dram_sharded_g8x1_k1_m1_n16_s1x4` rows; the file's `supersedes` block maps both prior rows to their replacements. |
| `2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0-power.csv` | Power, clock, and temperature trace for the targeted two-row superseding rerun (3 samples); its provenance is the matching result record above |
| `2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1.json` | Issue #65 toolchain-separated default-only comparison: 68 rows (59 successful, 9 failed), image digest `sha256:ead7b800bdb6bebb9425c377222314447c5b2052f6e8b1e3c9caa1818cb7d8c4` in its environment block |
| `2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1-power.csv` | Power, clock, and temperature trace for the 0.70.1 default-only comparison (88 samples); its provenance is the matching result record above |
| `2026-08-23-host-iq-path-vs-golden.json` | The #6 measurement: the IQ path (front end + IQ DAS) against the RF golden on the development host — per-stage errors at L0 checkpoints 1 and 2, image difference, and the axial PSF at −6 / −20 / −40 dB at D=8 and D=4, on the provisional `linear-5mhz` profile. Written by `python -m enodia.spec.beamform.decimation_sweep --record`; summarized in design.md §5 and §15 |
| `2026-10-02-host-newton-schulz-fidelity-switch-source-audit.json` | Board-free source audit of the pinned tt-metal math-fidelity path: descriptor and public compute APIs are kernel-wide, while matching direct LLK template specializations can be selected at source-level operation boundaries. |
| `2026-10-03-host-newton-schulz-two-tile-source-audit.json` | Board-free audit of the pinned Blackhole `matmul_block`/unpack/math traversal for `ct=1, rt=2, kt=1`, including exact CB page order, DEST accumulation slots, and a NumPy R*X/X*S simulation. |
| `2026-10-03-host-throughput-condition-audit.json` | Board-free audit of previous throughput artifacts: the 2.1945/25.0023 TFLOPS catalogue had Watcher attached while loaded power samples reached 1350 MHz. |
| `2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000.json` | Same-run no-Watcher rerun of stock best plus one-tile full-sync block 4 and half-sync block 2, all L1 inputs, L=32/batch-8192, 1,000 launches per row; loaded `aiclk_mhz` values were 1343 and 1350, and the wrapper stdout has no Watcher line. |
| `2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000-power.csv` | Power, clock, and temperature trace for the no-Watcher three-row throughput rerun above. |
| `2026-10-03-p150a-newton-schulz-two-tile-batch4-blocked.json` | Batch-4 Watcher correctness probe after the board-free audit: one-tile full/half pass at 0.00527355, two-tile full/half fail with exact-zero output at relative error 1.0; the later temporary diagnostic exit 137 triggered the required one reset and stopped further experiments. |
| `2026-10-03-p150a-newton-schulz-two-tile-batch4-blocked-power.csv` | Power, clock, and temperature trace for the blocked batch-4 two-tile probe above. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-a-blocked.json` | Minimal stage-a probe: batch 4, one iteration, matrix block 1, no DEST seed; all four `-R·X` products failed the 1e-2 NumPy gate, so stages b/c were not run. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-a-blocked-power.csv` | Power, clock, and temperature trace for the first-failure stage-a probe above. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-a1-blocked.json` | Decomposed stage-a1 probe (k=0 only): real output tiles pass while imaginary output tiles fail, so a2/a3 and later stages were not run. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-a1-blocked-power.csv` | Power, clock, and temperature trace for the first-failure a1 probe above. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-a3-pass.json` | After the DEST-row fix, the batch-4 Watcher a3 probe passes all four matrices and both output tiles below the 1e-2 NumPy gate. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-a3-pass-power.csv` | Power, clock, and temperature trace for the passing a3 probe. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-b-prime-pass.json` | Seed-free K=3 stage b′: the BF16 `[2I; 0]` term is a third matmul contribution, and all four batch-4 matrices pass below the 1e-2 NumPy gate without DEST seeding. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-b-prime-pass-power.csv` | Power, clock, and temperature trace for the passing b′ probe. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-c-blocked.json` | Stage c then produced non-finite output for all four matrices; this numerical failure stopped the experiment without reset, batch-8192 validation, or throughput. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-c-blocked-power.csv` | Power, clock, and temperature trace for the blocked c probe. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-c-pass.json` | After explicit SrcA/SrcB format transitions, seed-free b′ followed by X·S passes batch 4 with maximum relative error 0.0020847. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-c-pass-power.csv` | Power, clock, and temperature trace for the passing c probe. |
| `2026-10-03-p150a-newton-schulz-two-tile-b8192-correctness-blocked.json` | Production batch-8192 two-tile correctness: both full-sync block 4 and half-sync block 2 fail the NumPy gate (3.2010 and 3.3318); no throughput was measured. |
| `2026-10-03-p150a-newton-schulz-two-tile-b8192-correctness-blocked-power.csv` | Power, clock, and temperature trace for the blocked batch-8192 correctness run. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-blocked.json` | First production staged run (batch 4, matrix block 1, one iteration) fails before later stages; the runner's norm metric overflowed while the downloaded output remained finite. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-blocked-power.csv` | Power, clock, and temperature trace for the first production staged run. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-pass.json` | After the board-free `-Xi` format fix, production stage A (batch 4, block 1, one iteration) passes with relative error 0.00006057. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-pass-power.csv` | Power, clock, and temperature trace for the passing production stage A. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-b-pass.json` | Production stage B (batch 4, block 1, eight iterations) passes with relative error 0.00432563. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-b-pass-power.csv` | Power, clock, and temperature trace for the passing production stage B. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-c-pass.json` | Production stage C (batch 4, block 2, eight iterations) passes with relative error 0.00432563. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-c-pass-power.csv` | Power, clock, and temperature trace for the passing production stage C. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-d-pass.json` | Production stage D (batch 4, block 4, eight iterations) passes with relative error 0.00432563. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-stage-d-pass-power.csv` | Power, clock, and temperature trace for the passing production stage D. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-b8192-full-pass.json` | Production batch-8192 full-sync block-4 correctness passes with relative error 0.00652210. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-b8192-full-pass-power.csv` | Power, clock, and temperature trace for the passing batch-8192 full-sync run. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-b8192-half-pass.json` | Production batch-8192 half-sync block-2 correctness passes with relative error 0.00652210. |
| `2026-10-03-p150a-newton-schulz-production-two-tile-b8192-half-pass-power.csv` | Power, clock, and temperature trace for the passing batch-8192 half-sync run. |
| `2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000.json` | Same no-Watcher device run with 1,000 launches per row: current one-tile all-L1 and executable two-tile DRAM-placement full/half rows, plus stock context. |
| `2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000-power.csv` | Power, clock, and temperature trace for the mixed-placement throughput catalogue. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-b-blocked.json` | The prior batch-4 Watcher b probe entered exit 137 before producing a result; the required single reset and Stage-1 recovery followed, so c and batch 8192 were not attempted. |
| `2026-10-03-p150a-newton-schulz-two-tile-probe-b-blocked-power.csv` | Power, clock, and temperature trace for the blocked b probe. |
| `2026-10-03-p150a-stage1-health-after-two-tile-b-reset.json` | Required post-reset Stage-1 health probe after the b abnormal exit: pass at relative error 0.004564372822642326, with cleanup clear. |
| `2026-10-03-p150a-stage1-health-after-two-tile-b-reset-power.csv` | Power, clock, and temperature trace for the post-reset health probe. |
| `2026-10-03-p150a-stage1-health-after-two-tile-debug-reset.json` | Required post-reset Stage-1 health probe: pass at relative error 0.004564372822642326, with cleanup checks clear. |
| `2026-10-03-p150a-stage1-health-after-two-tile-debug-reset-power.csv` | Power, clock, and temperature trace for the post-reset health probe above. |
| `2026-10-02-p150a-newton-schulz-l32-b8192-fidelity-split-catalog-1000.json` | First device fidelity-split measurement: legacy all-HiFi3 and direct-LLK `0+8` pass the batch-8192 NumPy gate; `4+4` and `6+2` fail, so only the passing forms receive same-run 1,000-launch throughput rows. |
| `2026-10-02-p150a-newton-schulz-l32-b8192-fidelity-split-catalog-1000-power.csv` | Power and clock trace for the fidelity-split comparison above. |
| `2026-09-27-p150a-newton-schulz-l32-b8192.json` | Issue #63: L=32, batch 8192 fixed-eight-iteration Newton-Schulz throughput; same-run stock `ttnn.matmul` and `custom_newton_schulz` rows, with correctness evidence and per-launch P50/P99/P99.9. The companion power trace shares the stem. |
| `2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge.json` | Supersedes the preceding Issue #63 record after the stock catalogue merge; repeats the same device-0 stock/custom comparison with the merged runner's default-only mode. |
| `2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000.json` | Supersedes the preceding Issue #63 throughput record; measures the complete stock configuration catalogue and custom row in one run with 1,000 launches per row for P99.9. |
| `2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000.json` | Order of work 2: supersedes the preceding catalogue record and measures the two-DEST complex-product kernel under the same 1,000-launch stock/custom comparison. |
| `2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000.json` | Fidelity sweep: supersedes the Order 2 record, records failed LoFi/HiFi2 gates without throughput rows, and measures valid HiFi3/HiFi4 plus the full stock catalogue at 1,000 launches. |
| `2026-09-28-p150a-newton-schulz-profile-breakdown-blocked.json` | Blocked profiling attempt: the pinned release lacks a Tracy-enabled device-profiler build, so no cycle table or bottleneck claim is made. |
| `2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-blocked.json` | Supersedes the Tracy-blocked profiling attempt: the cycle-counter run completed, but the first host transport used tiled storage for raw uint32 pages; the corrected row-major path is committed without a third hardware run, so no cycle table or bottleneck claim is made. |
| `2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter.json` | Supersedes the cycle-counter transport failure after the known-value row-major host test: records the valid HiFi3 core-0 RISC cycle table, does not determine the bottleneck, and motivates the next measurement separating steady all-work, warmup, reader, writer, and compute scopes. |
| `2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady.json` | Supersedes the prior cycle-counter table with all-tile/all-iteration compute and separated reader/writer intervals; residuals are explicit and consistency passes, but no unique bottleneck is claimed because RISC windows overlap and compute still has unclassified overhead. |
| `2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000.json` | Supersedes the fidelity catalogue; same-device HiFi3 comparison of stock, baseline, `fuse_s`, and `fuse_s` plus `batch_reads`, with 1,000 launches per row and batch-4/batch-8192 correctness evidence. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000.json` | Supersedes the optimization catalogue; diagnostic same-device HiFi3 `fuse_s` comparison of BF16-state and FP32-state X, records the BF16 correctness-gate failure, L1/DRAM output placement, and does not claim a causal unpack bottleneck. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000.json` | Supersedes the unpack diagnostic; same-device HiFi3 FP32-state `fuse_s` comparison of matrix blocks 1, 2, and 4 against the best stock row, with batch-4 and batch-8192 correctness and explicit L1/DEST accounting. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block8-preflight-blocked.json` | Block-8 host preflight: DEST-slot accounting passes, but the FP32-state/fused-S L1 footprint is 1,721,088 bytes versus the 1,572,864-byte budget, so no device run was attempted. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block8-blocked.json` | Block-8 DRAM-input runtime diagnostic: batch-4 correctness passed, but batch-8192 timed out with no output after the allowed recovery reset; the requested comparison was not measured. |
| `2026-10-01-p150a-stage1-health-container-cleanup.json` | One post-reset Stage 1 recovery probe through the named-container wrapper; it passed with relative error 0.00456437 and records the owner-performed reset increment. |
| `2026-10-01-p150a-stage1-health-container-cleanup-power.csv` | Power trace for the one Stage 1 recovery probe above. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000.json` | Supersedes the blocked block-8 runtime attempt; records block-8 DRAM correctness at batch 4 and 8192 plus the same-run stock, block-1 L1, block-4 L1 rejection, block-4 DRAM, and block-8 DRAM catalogue with 1,000 launch samples where applicable. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000-power.csv` | Power trace for the same-run input-memory/block catalogue above. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000.json` | Supersedes the input-memory catalogue; records block-4 L1 correctness, the same-run stock/block-1/block-4 1,000-launch comparison, and successful reproduction of the `7472bb1` kernel configuration. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000-power.csv` | Power trace for the block-4 L1 and historical-kernel comparison above. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-x0-bf16-blocked.json` | Blocked x0-BF16 experiment: block-8 batch-4 passed, but batch-8192 produced non-finite output, so block-4 correctness and throughput comparison were not run. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-x0-bf16-blocked-power.csv` | Power trace for the blocked x0-BF16 batch-8192 probe. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-block8-compile-blocked.json` | Blocked per-input-memory probe: host preflight passes, but the reader JIT cannot compile mixed L1/DRAM TensorAccessor types before execution. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-block8-compile-blocked-power.csv` | Power trace for the blocked per-input-memory probe. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000.json` | Supersedes the mixed-accessor compile-blocked record; records R-L1/X0-DRAM correctness for batches 4 and 8192 plus the same-run 1,000-launch stock/block comparison. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000-power.csv` | Power trace for the successful per-input-memory comparison above. |
| `2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json` | L=16 diagonal-pair fallback: records batch-4/batch-8192 correctness and the same-run stock catalogue plus custom block-1/block-4 1,000-launch comparison. |
| `2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000-power.csv` | Power trace for the L=16 diagonal-pair comparison above. |
| `2026-10-02-p150a-newton-schulz-l16-l32-acceptance-catalog-1000.json` | Issue #63 acceptance catalogue: both L=16 and L=32 stock/custom rows, fuse_s variants, HiFi3/HiFi4, and 1,000 launches per row, with device correctness-test evidence. |
| `2026-10-02-p150a-newton-schulz-l16-l32-acceptance-catalog-1000-power.csv` | Power and clock trace for the combined L=16/L=32 acceptance catalogue. |
| `2026-10-03-host-newton-schulz-reference-sweep.json` | The #85 stage-1 host sweep: both fixed Newton-Schulz X0 choices, N=8..16, condition numbers 10/30/100/300, L=16/32/64, and float64/float32 inverse, MV-direction, and same-array beam-pattern metrics. Written by `python -m enodia.spec.beamform.newton_schulz_sweep --record docs/measurements/2026-10-03-host-newton-schulz-reference-sweep.json`; the beam-pattern metric is measured evidence, not the stage-1 gate |
| `2026-10-04-p150a-newton-schulz-stage2-bf16-diagnostic.json` | Issue #85 Stage 2 board-free and board diagnostic: BF16 R quantization versus the original float32 R, plus HiFi3/HiFi4 kernel comparisons on the fixed L=32 batch-4 configuration; no device threshold is changed |
| `2026-10-04-p150a-newton-schulz-stage2-bf16-full-diagnostic.json` | Issue #85 Stage 2 full diagnostic: 24-point host BF16 input-R sweep plus L=32/L=16 batch-8192 HiFi3 device validation and 1,000-launch throughput; thresholds and existing code are unchanged |
| `2026-10-04-p150a-newton-schulz-stage2-bf16-provenance-supplement.json` | Source evidence/corroboration for the two preceding Stage 2 BF16 diagnostics: records their verified run identifiers, harness revisions, and physical board serials; each source record carries the same serial locally. |
| `2026-10-04-p150a-newton-schulz-stage2-pr90-current-head.json` | Supersedes the two Stage 2 BF16 diagnostic records above. Throughput and error measurements use harness commit `ce8bb30`; device tests use commit `0d786b0`, as recorded in the measurement fields; L=32/L=16 batch-4/batch-8192 validation and 1,000-launch throughput. |
| `2026-10-04-p150a-newton-schulz-partial-block-padding-fuse-s-true.json` | ADV-81-001 device verification: the seven named fuse_s=true partial-block L=16/L=32 cases passed against the independent NumPy reference under the pinned 60-second Watcher run. |
| `2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control.json` | Immutable predecessor for the PR #81 resident-versus-reload-R control; it remains readable but is superseded by the complete-provenance retake below. |
| `2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-power.csv` | Immutable predecessor power, clock, and temperature trace; its provenance is the matching predecessor record above. |
| `2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-superseding.json` | Supersedes the predecessor record above with a clean harness capture: L=32, batch 8192, block 4, fused S, HiFi3, FP32 state, all inputs L1; both variants passed batch-4 and batch-8192 correctness, then each ran 1,000 launches in one watcher-free run. Resident reached 51.82917713218822 TFLOPS and reload-R 31.33414466591468 TFLOPS; the companion power trace shares the stem. |
| `2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-superseding-power.csv` | Power, clock, and temperature trace for the complete-provenance superseding resident-versus-reload-R control above. |
| `2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck.json` | Issue #94 same-device L=16/L=32, batch-8192, block-4, fused-S, HiFi3 comparison of BF16 state with FP32 DEST accumulation, BF16 state without FP32 DEST accumulation, and FP32 state; all inputs L1, outputs DRAM, N=12, with batch-4/batch-8192 correctness against BF16-rounded-R and true-inverse references. The record includes the 0.01 gate, DEST-capacity controls, power trace, and bounded pack/unpack-versus-synchronization inference. |
| `2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck-power.csv` | Power, clock, and temperature trace for the Issue #94 BF16-state/DEST comparison above. |
| `2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32.json` | Issue #92 same-device L=16/L=32, batch-8192, block-4, fused-S, HiFi3, FP32-state, all-inputs-L1 comparison of one and two external CB windows; includes batch-4/batch-8192 BF16-rounded-reference correctness, 1,000-launch p50/p99/p99.9 rows, L1 preflight, and cycle-counter profiling. The corrected N=12/24-complex-matmul FLOPs fields are L16=6442450944 and L32=51539607552; p50-derived values are L16 12.4716->13.9849 and L32 58.3906->66.5621 TFLOPS, while stored `achieved_tflops` values are the fastest/best-launch TFLOPS selected by the runner; the p50-derived values differ by definition. |
| `2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32-power.csv` | Power, clock, and temperature trace for the Issue #92 block double-buffer comparison above. |
| `2026-10-04-p150a-newton-schulz-block-double-buffer-risc-profiler-diagnostic.json` | Issue #92/PR #93 diagnostic-only Tracy profile of L=32, batch 8192, block 4, fused-S, HiFi3, FP32-state one-window versus two-window runs. Reader/writer waits are primarily waits on compute; two-buffering shortens the full interval by about 15%, and compute-side instruction supply plus unpack/math/pack handoff remain cautious, non-exclusive candidates. Counter definitions do not establish pack-side causality. The recorded NoC bytes and approximately 0.774 ms launch interval derive approximately 240 GB/s aggregate and 2 GB/s/core (not a new measurement); two-window counters remain unavailable. Existing HiFi3/HiFi4, full/half-sync, and BF16/FP32-state comparisons are cited with their confounds. Not a throughput headline or replacement for prior records. |
| `2026-10-04-p150a-newton-schulz-block-double-buffer-counter-decomposition.json` | Issue #92/PR #93 diagnostic-only official Tracy all-counter multipass decomposition for both one-window and two-window L=32/batch-8192 runs. Tag-local definitions distinguish semaphore waits, instruction availability versus distinct issue-rate fields, packer efficiency/handoff, and destination-read backpressure; no distinct THREAD_INSTRUCTIONS_N-derived issue-rate field was present, and both modes retain raw replay values plus unavailable formula-branch inputs. Not a throughput headline or replacement for prior records. |
| `2026-10-04-p150a-newton-schulz-block-double-buffer-profile-compute-decomposition.json` | Issue #92/PR #93 diagnostic-only fixed-release-image profile-only compute decomposition: eight Watcher-enabled batch-4/batch-8192 L16/L32 one/two-window captures with per-TRISC external-CB polling intervals, RISC-V-side enqueue/dispatch brackets around DEST semaphore-wait instructions, and explicitly unclassified residual work. Exact v0.75.0 source audit and official semaphore waits are recorded; no Tracy profiler or new throughput claim. |
| `2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000.json` | Issue #96 same-device L16/L32 full-sync versus half-sync DEST catalogue: both state variants, admitted DEST blocks, L1 preflight rejections, BF16-rounded-R batch-4/batch-8192 correctness, 1,000-launch p50/p99/p99.9, p50/fastest TFLOPS, and official semaphore/pack counters. |
| `2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000-power.csv` | Power, clock, and temperature trace for the Issue #96 half-sync catalogue. |
| `2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000.json` | Historical Issue #100 same-device new-default versus previous-default comparison; superseded for the combined acceptance claim by the 2026-10-06 record below and retained unchanged. |
| `2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000-power.csv` | Historical power, clock, and temperature trace for the superseded Issue #100 comparison above. |
| `2026-10-05-p150a-newton-schulz-issue101-default-correctness.json` | Historical PR #101 correctness-only rerun; superseded for the combined acceptance claim by the 2026-10-06 record below and retained unchanged. |
| `2026-10-05-p150a-newton-schulz-issue101-default-correctness-power.csv` | Historical power, clock, and temperature trace for the superseded correctness-only rerun above. |
| `2026-10-06-p150a-newton-schulz-issue101-combined-catalog-1000.json` | Authoritative combined Issue #100/PR #101 record: one device-0 session with all nine correctness rows, four 1,000-launch performance rows, p50/p99/p99.9 and both TFLOPS derivations, board-id serial alias provenance, complete environment, cleanup evidence, and external raw-artifact provenance. The raw artifact is persisted for audit; rebuilding is deferred to Issue #102. This record supersedes both 2026-10-05 records above; those predecessors remain immutable. |
| `2026-10-06-p150a-newton-schulz-issue101-combined-catalog-1000-power.csv` | Power, clock, and temperature trace for the authoritative combined record above. |
| `2026-10-07-host-newton-schulz-issue88-fp32-r-plan.json` | Issue #88 host-side ADR-0005-compatible plan: explicit FP32-R accounting, the default L=32 rejection, fitting R-in-DRAM/block-2 alternatives, first-launch Watcher/60-second procedure, and five pending 1,000-launch comparison rows. No board run was made yet. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r.json` | Immutable historical Issue #88 device-1 comparison record, superseded by the 2026-10-09 device-1 retake because its historical beam metrics used the defective batched `_weight` einsum denominator and it did not retain device inverse arrays. The JSON and trace remain unchanged. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r-power.csv` | Historical power, clock, and temperature trace for the superseded four-row comparison record above; immutable. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup.json` | Immutable historical Issue #88 L=32 placement supplement, superseded by the 2026-10-09 device-1 retake for the same defective-beam-metric and missing-inverse-array reasons. The JSON and trace remain unchanged. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup-power.csv` | Historical power, clock, and temperature trace for the superseded placement supplement above; immutable. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json` | Immutable historical Issue #88 convergence retake, superseded by the 2026-10-09 device-1 record below. Its five measured rows and expected FP32-R/L32/R-in-L1 preflight rejection remain readable as predecessor evidence only. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-summary.json` | Immutable historical derived summary for the superseded convergence retake. Its original-R and variant-matched tables remain historical context; its device beam-pattern fields are withdrawn because the source used the defective `_weight` denominator. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r-beam-metric-summary.json` | Immutable host-only diagnostic of the withdrawn historical beam metrics. Its corrected fixed-N host values are not device results; the retained source had no downloaded device inverse arrays. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv` | Historical power, clock, and temperature trace for the superseded convergence retake; immutable. |
| `2026-10-08-p150a-newton-schulz-issue88-fp32-r-first-launch-power.csv` | Historical power, clock, and temperature trace for the first FP32-R Watcher-capped launch preceding the superseded comparison run; immutable. |
| `2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json` | Immutable superseded Issue #88 device-1 retake. The final runner retake below supersedes it because the primary record now carries each row's detailed L1 breakdown and total, rather than relying on a derived accounting artifact. |
| `2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-summary.json` | Immutable superseded companion metadata for the 2026-10-09 retake; the final runner retake below is authoritative because its primary record carries the L1 accounting and matched-row evidence directly. |
| `2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv` | Immutable power companion for the superseded 2026-10-09 retake; its bytes remain unchanged. |
| `2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-l1-accounting.json` | Immutable superseded host-only L1 derivation. The final runner retake below supersedes it because the primary record itself carries every row's L1 breakdown and total, plus mechanically matched BF16-R/FP32-R deltas. |
| `2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json` | Final ADR-0005-compatible Issue #88 device-1 runner retake at harness commit `0242b31238c6642131e3a806c805f97c22126e6d`: five measured rows plus the expected FP32-R/L32/R-in-L1 rejection, detailed per-row L1 breakdowns/totals, corrected device beam diagnostics, retained inverse metadata, run-bound power provenance, and complete cleanup. |
| `2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json` | **Authoritative derived companion** for the final runner retake: it names and hashes the primary record, inherits its ADR-0005 provenance, mechanically recomputes all matched-placement quality/performance/L1 differences, and lists the immutable 2026-10-09 predecessor JSON/summary/L1-accounting/CSV artifacts. |
| `2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv` | Same-stem power companion for the final runner retake; byte-identical to its immutable run snapshot, SHA-256 `ef12ca4972a7467c75ea8d35b735f1bb8bee78ad2cbd73f19bbcdac9d1c30cd5`, 523 bytes, and 10 usable samples. |

**Issue #88 device-1 retake (2026-10-09, superseded).** The immutable predecessor set is `2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json`, its `-summary.json`, its `-l1-accounting.json`, and its `-power.csv` companion. The final runner retake supersedes all four because its primary record itself carries the complete L1 ledger and final evidence, together with the paired BF16-R/FP32-R values from which matched deltas are mechanically derived. Every 2026-10-09 artifact remains immutable historical evidence.

**Issue #88 final runner retake (2026-10-10).** The authoritative record is `2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json`, its authoritative derived companion is `2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json`, and its power companion is `2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv`. It is the one device-1 session under the unchanged six-row catalogue, with board type p150a, serial `0000040331914124`, PCI `0000:09:00.0`, watcher disabled, no reset, five successful rows at exactly 1,000 launches each, and the expected FP32-R/L32/R-in-L1 rejection before allocation. The primary record embeds the complete L1 ledger for all six rows: successful totals are 885,504 (BF16-R/L16/L1), 1,229,568 (FP32-R/L16/L1), 1,295,104 (BF16-R/L32/L1), 803,584 (BF16-R/L32/DRAM), and 901,888 bytes (FP32-R/L32/DRAM); the rejected row is 1,884,928 bytes, 312,064 over budget. The companion mechanically recomputes matched L1 total deltas as FP32-R minus BF16-R: +344,064 bytes at L16/L1 and +98,304 bytes at L32/DRAM, plus inverse, MV, all four beam, P50 TFLOPS, and latency differences. The record also carries the corrected four beam indicators, common original-R true-inverse basis, retained inverse metadata, run-bound power snapshot, and allowlisted serialization.

**Issue #88 companion audit (2026-10-10).** The final runner retake's frozen raw snapshot remains in external raw storage, while the tracked same-stem power companion above contains identical bytes; the JSON summary is derived metadata and does not duplicate that CSV. The convergence-rerun direct record already has its same-stem companion; its tracked trace matches the raw trace byte-for-byte (SHA-256 `b0d9f48dad2d5527cac393f4098f6fb9d888e3f5ddcf66b15638c86f61dc159d`, 476 bytes, 9 sample rows, 10 physical lines). The convergence-rerun-summary is a derived summary and intentionally reuses that existing trace rather than duplicating it. First-launch is trace-only and has no published JSON record; its tracked trace also matches the raw trace byte-for-byte (SHA-256 `9caaedfbdf894135d5c510fe3627fe85d2050f14c7089c94d46109a021bcddfa`, 139 bytes, 2 sample rows, 3 physical lines). No other missing direct-record companion was found.

**Issue #88 beam-metric correction.** The prior source JSON and reference-labeled summaries remain immutable. The runner's historical `_weight` denominator used `einsum("j,bi->b", ...)`, which independently summed steering and weights instead of computing `aᴴPa`; for the half-wavelength 16/32-element array, the steering-vector sum approaches zero at ±30°, producing singular normalization and the reported enormous complex-response errors. The corrected contraction is `einsum("j,bj->b", ...)`. Host regression checks prove `wᴴa=1` and zero self-comparison; the source audit found no complex multiply sign, conjugation, transpose, or output-reconstruction defect. The historical device beam ranking is withdrawn, and the corrected device metrics now live in the 2026-10-10 final runner retake with retained inverse arrays; the 2026-10-09 retake remains immutable predecessor evidence.

Issue 12 debugging trials from 2026-10-05 through 2026-10-06 (failed runs used to chase harness defects) were removed before merge because they support no claim; they remain in this PR branch history, in the commits that removed them.

Outlier note: the minimum and maximum are determined by paired short/long intervals (3 pairs for 60,000 frames and 15 pairs for 600,000 frames), each pair summing approximately 2 × 1,350,000 ticks; the cause is out of scope and is a follow-up candidate.

The corrected-drain board-id-alias runs observed zero paired short/long outliers in both the Watcher validation and 500,000-frame timing run, versus 27 pairs in the prior 2616f96 record. The timing run has no phase entries because there were no pairs; the Watcher command did not request a raw timestamp file, so its outlier audit uses the retained fine-bin histogram. This is an observation only with no causal claim.

Final 500,000-frame outlier observation: 27 adjacent short/long pairs were found with pair sums 2,699,967–2,700,034 ticks (target 2,700,000); full frame endpoints, corrected elapsed seconds, and every adjacent-event quotient/remainder for period 6,363 are in the ADR-0005 record. The aggregate keys are `1,0`=11, `3,6362`=4, `0,3182`=2, `1,3182`=2, `2,0`=1, `5,0`=1, `11,3180`=1, `7,3181`=2, `5,6362`=1, and `2,3181`=1; 352,863→359,226 and 457,851→464,214→470,577 appear as q=1,r=0. This is an observation only with no causal claim; no single exact period was detected (6363 was the most common gap, 11/26, gcd 1).

### Issue #12 Stage 1 numeric provenance map

Every resident timing number in the README, open-issues table, and PR #103 maps to exactly one record here:

- **Current corrected-drain timing record:** `2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005.json` — N=499,999; P50/P99/P99.9/P99.99=`1,349,988/1,350,061/1,350,065/1,350,065` ticks; min/max=`1,349,889/1,350,065`; work min/max=`1,062/1,087`; paired outliers=`0`; companion power trace is the same-stem CSV. The no-Watcher timing evidence claim is true.
- **Current corrected-drain Watcher validation:** `2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005.json` — N=99, timing evidence false; work min/max=`1,063/1,073`; P50=`1,350,008` ticks; P99/P99.9/P99.99 insufficient; companion power trace is the same-stem CSV. The old current-wrap records above are superseded by these corrected-drain records.
- **Historical superseded final record:** `2026-10-06-p150a-issue12-stage1-final-500000-adr0005.json` — N=499,999; P50/P99/P99.9/P99.99=`1,349,988/1,350,048/1,350,050/1,350,050` ticks; min/max=`111,119/2,588,915`; work min/max=`1,074/1,098`; paired outliers=`27`; retained unchanged for provenance only.
- **Historical 660-second exclusion:** `2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005.json` — N=599,999; it is not timing evidence because its configured timeout exceeded the approved 600-second cap.

The preceding numeric sets are historical or current exactly as labeled; no number is silently transferred between records.

**Correction (peak fidelity, 2026-10-03).** `2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000.json` associates the 332 TFLOPS BF16 peak with LoFi and derives HiFi2/HiFi3/HiFi4 reference peaks of 166.0/110.7/83.0 TFLOPS from it. The record states that association as an inference; it does not hold. tt-metal's `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md` gives the ideal cycles per tile product as 16 (LoFi), 32 (HiFi2), 48 (HiFi3) and 64 (HiFi4), or about 5.4 TFLOPS per matrix engine at LoFi and 1.35 GHz, and `docs/design.md` §2 lists Block FP8 at 664 TFLOPS beside BF16 at 332. The 332 figure is therefore the HiFi2-rate BF16 peak, and the LoFi rate is about twice it. The record is not rewritten; its efficiency figures remain correct against the 332 denominator, while its derived per-fidelity reference peaks should be read as half their true values.

All imported JSON records retain their source `harness_commit` values unchanged. Full commit resolution for every recorded value was verified on the retained `feat/63-fidelity-split-diagnostics` branch, which is the provenance authority for these diagnostics; some of those commits are also reachable from `main`. Companion CSV traces inherit provenance from the matching JSON stem.
Results are not rewritten. A measurement that turns out to be wrong, or is
retaken on a corrected harness, is superseded by a later record that says
so — the same invariant ADR-0004 sets for decisions. The reasoning is in
ADR-0005, which also fixes what a result must carry and how a figure quoted
elsewhere refers back to it.

The host name is outside ADR-0005's contract. Removing it from a landed
record does not violate ADR-0005's prohibition on rewriting because the
measurement values and reproducibility provenance are unchanged. The string
remains in git history; removing it from history would rewrite `main`, which
is not done.

## Issue #122 AICLK audit

The reproducible board-free corpus audit classifies endpoint-only and
intermediate AICLK traces, and separates exact covered intervals from CSV-span
diagnostics. See [the generated Issue #122 audit](issue122-aiclk-audit.md).
Issue #127's counterbalanced records retain the precommitted raw P50 decision
while qualifying the clock sensitivity: under (a) `s` in `[0, 1]` and (b) the
#122 left-held time-weighted AICLK representing effective clock, all eight
matched-occurrence sensitivity intervals remain negative. The persisted min/max
intersections are descriptive range overlap only, not an effect bound or proof
of AICLK alignment; without (b), range-only information does not fully exclude
a zero/clock explanation. The audit itself remains unchanged.

## Issue #127 confirmation run

- `2026-10-15-p150a-newton-schulz-issue127-confirmation-run1.json` is the owner-reviewed confirmation run record.
- `2026-10-15-p150a-newton-schulz-issue127-confirmation-run1-power.csv` is its immutable power-trace companion.
- The confirmation record validates execution and provenance only. It is explicitly excluded from the Issue #127 plan prediction and decision criteria; its reference deltas must not change the prediction, thresholds, or fixed run order.

## Issue #127 failed measurement-2 attempt

- `2026-10-15-p150a-newton-schulz-issue127-measurement-2-failed-cooldown.json` is the committed failed record for the original `measurement-2` attempt.
- `2026-10-15-p150a-newton-schulz-issue127-measurement-2-failed-cooldown-power.csv` is its immutable power-trace companion.
- The record is excluded from the Issue #127 prediction and decision criteria. Its cooldown gate incorrectly required 1350 MHz while the card was idle at 800 MHz; all eight measured rows completed, but the run is not accepted as measurement evidence.

## Issue #127 accepted measurement records

- `2026-10-15-p150a-newton-schulz-issue127-measurement-1.json` is the first accepted measurement record.
- `2026-10-15-p150a-newton-schulz-issue127-measurement-2-rerun.json` is the accepted replacement for the failed `measurement-2` attempt, using the separately approved `measurement-2-rerun` scope.
- `2026-10-15-p150a-newton-schulz-issue127-abba-summary.json` mechanically applies the committed plan criteria and the qualified clock-sensitivity calculation to those two records. The confirmation and failed-attempt records are excluded; the summary's range overlaps are descriptive only, not effect bounds.
- The precommitted raw P50 decision remains **R-format effect**; under assumptions (a) and (b), all eight sensitivity intervals remain negative. Without (b), range-only information does not fully exclude a zero/clock explanation. The default remains BF16-R and FP32-R remains an explicit variant.

### Issue #127 power-sample count definitions

Future Issue #127 records generated by the runner use two deliberately separate interval counts:

- `power_sample_count_in_interval` is the number of samples contributing to the timed interval `[start, end)`, including the immediately preceding sample held across the interval start.
- `power_samples_started_in_interval` is the number of samples whose timestamp starts inside `[start, end)`; it excludes that left-held sample whose timestamp precedes the interval start.

The committed confirmation and measurement records are immutable and do not gain the new `power_samples_started_in_interval` field. The derived summary's `sampling_audit` carries that field while applying the second definition mechanically to the four immutable Issue #127 power traces and all 32 comparison rows. All 32 rows contain at least two samples that started inside their timed interval (minimum 2, maximum 3), so the two-samples-per-row sampling target remains met. The confirmation run, measurement-1 record, failed measurement-2 record, rerun record, their power CSVs, and the plan remain byte-for-byte immutable; only this derived summary carries the audit for those existing records.
