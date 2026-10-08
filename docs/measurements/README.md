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
catalog in `enodia/tt/bench/resident_record.py` were run against all three
committed Issue #104 JSON records and their committed companion CSVs without
rewriting either data format. The validator guarantees required field
presence/type/range and major relationships; undeclared telemetry or
production fields are retained and reported as warnings rather than rejected.
Full record-kind consistency and agreement between declared coverage flags and
parsed trace facts are deferred to Issue #110. Revalidation therefore reports
added fields without rewriting the historical JSON or CSV.
The sampler-off record passes the diagnostic structural checks:
its trace is absent by design and `timing_evidence=false`; its elapsed-seconds
fields are interpreted with the configured `parameters.budget_aiclk_mhz` of
1,350 MHz, not the 800 MHz pre-run environment snapshot. The record's elapsed
seconds and `aiclk_mhz_for_elapsed_seconds` are therefore revalidated as
configured-clock values without selecting that snapshot. Its raw timestamp
hash cannot be rechecked or recomputed from retained bytes because the
referenced external `raw-timestamps.bin` is not committed. The immutable JSON
and CSV leaves are unchanged. It is not timing evidence. The sampled records
remain invalid under the strict catalog even though their CSV bytes are
readable and their declared sample counts and SHA-256 values match: 181 rows
and `0c3b8f1893bc329004c4c0a5a59a76d572ff479d04f686c03effb6b110f41b7b` for
`2026-10-06-p150a-issue104-sampler-default.json`, and 77 rows and
`f728f53d7f64bf3a599d9abfc85161e7d66d17794c723d8f19a792f2839b6f50` for
`2026-10-06-p150a-issue104-sampler-5s.json`. Both sampled JSON records lack
explicit `run_start`/`run_end` bounds, valid-row counts, and in-run AICLK
provenance, so their first/last coverage and timing AICLK are unverifiable
under the PR #109 rule; their committed `timing_evidence=true` therefore does
not pass the gate. The same external raw-timestamp-byte limitation applies to
both sampled records, although their committed `raw_timestamps.count=400000`
and `histogram.N=399999` satisfy the count relationship.

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
output directory and the wrapper's standard custom-runner result path inside
the container; the mounted output directories are retained on the development
machine.

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
  --out /out/runner-result.json \
  --device-id 0 \
  --frame-count 1 \
  --frame-interval-ticks 1350000 \
  --budget-aiclk-mhz 1350 \
  --outer-timeout-seconds 60 \
  --watcher
```

Leave `HEKATUS_TT_CONTAINER_TIMEOUT_S` unset so the wrapper selects the
mode-specific 60-second cap. The command names the runner, result path, device,
frame count, interval, clock used for the conversion, outer cap, and Watcher
mode instead of relying on runner defaults.

### No-Watcher timing

Only if the Watcher validation passes, and still under the owner approval, run
one no-Watcher timing attempt on device 0. It uses 500,000 frames, a 1 ms
harness interval at the documented 1,350 MHz clock, and a 600-second outer cap.
It writes the result and raw timestamps to container-internal `/out` paths:

```bash
env -u TT_METAL_WATCHER HEKATUS_TT_RUNNER=enodia/tt/bench/run_resident.py \
  enodia/tt/bench/run_in_container.sh out/bench/issue12-timing -- \
  --out /out/runner-result.json \
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
Retain the `out/bench/issue12-watcher/runner-result.json`,
`out/bench/issue12-timing/runner-result.json`, and
`out/bench/issue12-timing/issue12-timing-500000.bin` files:

```bash
find out/bench/issue12-watcher out/bench/issue12-timing -maxdepth 1 -type f \
  \( -name 'runner-result.json' \
     -o -name 'issue12-timing-500000.bin' \
     -o -name 'env-*.json' \
     -o -name 'power-*.csv' \) -print
mkdir -p issue12-retained
cp -a out/bench/issue12-watcher out/bench/issue12-timing issue12-retained/
sha256sum issue12-retained/issue12-timing/runner-result.json \
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
