# Issue #120 B attempt — stopped after B1 host reread

Harness: `b474216164c92bdda026fdf35e3a1d768635c7fa`.
Session: `13ba906388a94f7ba744f367481ae818`.
Owner approved Linux6.8.0-149-generic/TT-KMD2.12.0 as the new baseline on
**2026-11-18**. Run8 used Linux6.8.0-138-generic/TT-KMD2.11.0; this is not an
identical-environment comparison. Firmware19.6.0.0, image/native/kernel proofs,
device0 serial411c/PCI0000:06:00.0 and other required identities matched.

## Outcome and stop branch

B1 `production-v1-control-watch` executed4096 frames with Watcher and60s native/
outer caps. Native, named container and wrapper exited0. The outer host launcher
then exited2 while rereading the immutable root-owned mode0600 power snapshot:

```text
Issue120 production rejected: [Errno 13] Permission denied: '<case>/power-trace-snapshot-20261118T124104054310116761824398Z-1.csv'
```

**B2 was not executed.** No C/P, automatic retry or reset occurred. Both cards
returned800MHz with healthy DRAM; no container survived. The original failed
invocation is not relabeled successful by its stored eligibility flag or by the
subsequent forensic copy. Snapshot readability is tracked in **#182**.

The runtime session manifest is preserved unchanged. `baseline-session.json`
binds its exact SHA-256 to the dated owner decision and fresh full environment
baseline. This session record supplements, not rewrites, the runtime manifest.

## Evidence and independent reread

`B1-evidence.tar.gz` contains the original primary record and all12 required
files, including exact frame/map/native/power/Watcher bytes. `manifest.json`
gives every member size/SHA-256 and the archive fingerprint. Extract into an
empty host-owned directory, then use the unchanged loader, for example:

```bash
uv run python -c 'import tarfile; tarfile.open("B1-evidence.tar.gz").extractall("B1", filter="data")'
uv run python -c 'from pathlib import Path; from enodia.tt.bench.issue120_production.recording import load_validated_record; print(load_validated_record(Path("B1/results-20261118T124104054310116761824398Z.json"))["resident_integrity_passed"])'
```

Extraction and complete artifact recomputation passed. A named **never-started**
no-device/read-only inspection container copied protected evidence; it was then
removed. Original artifacts and their permissions were not modified. Public
Watcher text strips trailing whitespace; public wrapper/launcher text replaces
the user-specific case root with `<case>`. Original hashes are retained, and the
byte-exact Watcher/native data remain inside the archive. No new whitespace
exception or evidence-validator change was made.

## B1 reference values only

All4096 attempted/produced/consumed frames passed identity, cumulative checksum
and work checks. Overflow/drop/abort/pair counts were all0. Native stderr was
empty. All7 warning lines matched the same three #176 conditions, including the
adjacent NUMA fallback detail; no changed wording/new warning was accepted.
Watcher had7 fresh completed dumps with no detected anomaly.

| Metric | B1 control | B2 loaded |
|---|---:|---|
| Interval samples |4095 | not run |
| P50 ticks |1350005 | unavailable |
| P99 ticks |1350847.06 | unavailable |
| P99.9 | insufficient:4095 <20000 required samples | unavailable |
| Max ticks |1351227 | unavailable |
| Overflow / drop / abort / paired outlier |0 /0 /0 /0 | unavailable |

Raw diagnostics: consumed/full expected tokens4098/4098, mailbox4098, consumer
expectation echo4098; producer peer runtime xy[2,2], consumer peer[1,2], consumer
self NoC0[2,2]. Full raw stats/routing are in the original native envelope.
Control coverage is not_applicable, never manufactured from these diagnostics.

Watcher means timing_evidence=false. Frozen whole-run power coverage was
complete, but `load_interval_provenance_invalid` leaves the weighted observed-
host-envelope AICLK unavailable. The new source label is not recognized by the
shared explicit-load reader; timing eligibility also needs review before C/P
(**#183**). No old Issue #118 label was substituted to make it pass.

## Planning update, not calibration

B1 native execution took7.984162394 host seconds; the enqueue-to-terminal-observed
upper-bound span was4.897998353s. Setup was2.884949817s, recorded amortized
retrieval/analysis59.724475us/frame, sampler wait0.856223480s. The amortized term
contains fixed post-processing/teardown too; it is **not a measured linear slope**.

Card pre-arithmetic observation cost: median60721.5 ticks, min60366, max61672;
this median is4.4979% of the nominal1350000-tick period. Arithmetic alone was
1403..1431 ticks (median1407). These are separate instrumented phases, not a
product wall-time conversion or proof of total observation overhead.

No loaded host worker ran, so no new per-transfer time/rate or calibrated N
exists. Retain the provisional N24439669 and old run8 slow-rate model574.655s
only as uncalibrated inputs. Applying the **B1 amortized overhead literally** to
500000 frames gives29.862s retrieval/analysis and38.603s total reserve under the
existing proposal formula; combined with the old rate, that is613.258s and
**-13.258s** cap headroom. This conditional short/Watcher/control extrapolation
is not a bound or a new-baseline throughput measurement; it shows why the old
20s reserve cannot be assumed. Naively scaling the whole4.898s span to597.900s
also incorrectly scales cold/fixed costs and is not a production estimate.

Before C approval: resolve #182/#183, obtain a fully successful authorized B
unit, then separately authorize the fixed16384/1048576 and32768/1572864
frame/write calibrations. Those two sizes must determine new worker rates,
fixed versus frame-scaled retrieval/analysis/setup costs, observed frame-span
scaling and whether one fixed count plus margins fits600s. No time-adaptive
extension, latency threshold, loaded-minus-control claim or production result
is supplied here. `design.md` section12 is unchanged.
