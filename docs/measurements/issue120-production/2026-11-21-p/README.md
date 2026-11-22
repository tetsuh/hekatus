# Issue #120 production host CPU / PCIe pair — 2026-11-21

## Scope and provenance

The owner adopted **520 s / 22155167 fixed writes** on 2026-11-21 and authorized
one P1 control followed by one P2 only after all P1 release gates. Both completed
in session `a729ec3dec50456998fc668b2100648f`, manifest SHA256
`b9f60b19edb3d62e693683ab00e8d22eaeb54273bea02af4a60d704ec62c50b8`, at precommitted,
clean/pushed/tested harness `4fd4900b8813e813263ed3d5148e3b5feb3a59b8`.
Predictions preceded execution in PR #181 and the
[adoption plan](../../issue120-production-p-adoption-plan.md). No advance latency
pass/fail threshold was set. Receipt, adoption, source and build/device proof
hashes are in each captured record and pre-run-artifact-gate.json.

Each row used 500000 frames,1350000-tick period,64 fixed checksum/work iterations,
a four-page resident ring,the same two command queues/sub-device/device-open
configuration,Watcher off,and native/outer600 s caps. Only device0 was opened:
p150a serial411c,PCI0000:06:00.0,firmware19.6.0.0,Linux6.8.0-149/KMD2.12.0,
pinned tt-metal0.75.0 image. P1 submitted zero load;P2 ran the exact adopted count.
The selected mechanism is ordinary unpinned CQ1 writes with direct-L1 challenge
reads and a bounded FIFO issue path;no per-write finish or readback/event chain.
Measured native0a36ed60... and device/library/CTA/compile inputs remained unchanged.

This is one sequential pair of a synthetic resident timing/causal instrument,
not the complete ultrasound processing path or Ethernet ingest,and not a
population-level determinism or product wall-time guarantee.

## Intervals and integrity

Each distribution contains **499999** consumer-clock completion intervals.
Percentiles use the unchanged harness interpolation/sample sufficiency rules.
Full one-tick histograms are in the archived primaries under
`metrics.frame_intervals.histogram`:P1 has2408 bins,P2 has2411;each totals499999.

| Device ticks | P1 control | P2 loaded | Loaded minus control |
|---|---:|---:|---:|
| P50 |1350001 |1349991 |−10 |
| P99 |1350816 |1350834 |+18 |
| P99.9 |1351065 |1351070 |+5 |
| P99.99 |1351227 |1351205 |−22 |
| Maximum |1351658 |1351677 |+19 |
| Minimum |1348509 |1348282 |−227 |
| Attempted / produced / consumed |500000 each |500000 each |0 |
| Overflow / drop / abort |0 /0 /0 |0 /0 /0 |0 /0 /0 |
| #104 paired outliers |0 |0 |0 |
| Startup ticks |135069260 |135068725 |−535 |
| Per-frame arithmetic min / max ticks |1078 /1125 |1078 /1128 |0 /+3 |

All500000 cumulative checksums/native verified-frame counts passed in each run;
fixed work,identity,startup,clock monotonicity,routing and terminal expectations
passed ordinary artifact-based validation. Terminal mailbox and expectation
were500002,with producer/consumer runtime peers(2,2)/(1,2) and matched CTA routes.
All native/container/wrapper/launcher exits were0;both new-schema frozen-power
checks passed and timing_evidence=true. P1's completion was externally
acknowledged only after observed exits and healthy return;P2 could not be
released from publication alone. Durable claims/native entries consumed both
roles once and bound the adopted scope to this single session.

## Clock / conversion assumptions

Weighted AICLK over the host-observed enqueue-to-L1-terminal **upper-bound**
program envelope was1349.9999999999998 MHz(control) and1349.9687235718334 MHz(loaded).
These are duration-weighted,left-held frozen sampler observations,not maximum
AICLK and not exact per-frame frequency measurements. Maximum AICLK was1350 MHz;
whole-run minimum800 MHz includes setup/idle and is not used as the frame clock.
The snapshots had228/246 valid ordered rows,complete explicit-final-sample
coverage and no errors. The legacy `last_within_interval_bound=false` diagnostic
is unchanged from C2;the applicable explicit-final-sample coverage is complete.

| Qualified estimate, microseconds | P1 | P2 |
|---|---:|---:|
| P50 |1000.000741 |1000.016501 |
| P99 |1000.604444 |1000.640960 |
| P99.9 |1000.788889 |1000.815779 |
| P99.99 |1000.908889 |1000.915782 |
| Maximum |1001.228148 |1001.265419 |

The median tick difference is negative while this qualified converted difference
is positive because the two observed weighted clocks differ. Do not substitute
the maximum1350 MHz,interpret these as exact frame wall times,or infer a
statistically established loaded effect from one pair. See ADR-0015/0019/0020.

## Fixed work and causal coverage

| Accounted work | P1 | P2 |
|---|---:|---:|
| Successful host submissions |0 |22155167 |
| Submitted H2D bytes |0 |90747564032 |
| Targeted challenge reads |0 |22155167 |
| Targeted challenge-read bytes(4 each) |0 |88620668 |
| CPU integer operations |0 |725980512256 |
| Separate terminal-observer reads |98135 |98260 |

Memory does not grow with N:fixed runtime rings,reused bounded payload storage,
fixed-capacity host trace and at most frame_count+3 coalesced challenge transitions.
The fixed-frame device allocation is2048000000 bytes and the emitted compact
frame companion64000024 bytes;neither is a per-write allocation.

P2 artifact evaluation verifies **covered** for all500000 frames,including
startup/first/last/tail observations and emitted-transition binding. A conservative
first-through-last lower bound is499999 challenge-linked transfers,
2047995904 H2D bytes and16383967232 CPU operations. The wider fresh-challenge-issue
to scan-completion scope reports500001 transfers/2048004096 bytes;it is not the
scan-end-to-scan-end scope. These lower bounds are not the full submission totals.
The prior feasibility sources/short Watcher evidence remain the mechanism basis;
host submission timestamps alone do not establish device completion or coverage.

## Host trace and predictions versus observations

The loaded worker lasted**531.795383779 s**,inclusive41661.074307 writes/s.
Its339 fixed-cadence samples yield338 slabs with observed41253.159..41900.121
writes/s and population CV0.202100%;first/last-half mean rates41649.590/41672.985.
The first committed enqueue was observed77.768 us after worker start.
The post-main-enqueue-return bracket57999230214125..57999230214232 ns observed
25218 committed submissions:neither an exact API-return count nor device retirement.
Worker end followed observed terminal completion by31.070218 s. One long run and
correlated slabs still do not establish future or worst-case sustained bounds.

| Pre-run assumption | Observation |
|---|---|
| Both centers near1350000 ticks;differences near0,unknown sign/max |Centers+1/−9 ticks;quantile/max differences −22..+19 ticks;no threshold verdict |
| Conditional frame envelope500.708738 s |Host upper bounds500.720928 /500.725240 s,about12/17 ms longer |
| Worker fast..slow520..549.060 s |531.795384 s,inside assumed scenario,not validation of a rate bound |
| Conditional frame margin19.291 s |P1-envelope feasibility margin19.279 s before P2;actual causal coverage verified |
| Host charged13.282386 s includes P1 reread;outer reader excluded |Loaded pre-native container work8.008287 s and post-native7.385413 s;not isolated P1 validation timers |
| Native2GB budget10 s,unmeasured before run |Observed retrieval/verify/close/publication aggregate0.573805 /0.468980 s;not isolated DMA |
| Ropt31.525745 /Rpess41.525745 s,conditional cap margin19.414 /9.414 s |Actual container-window margins88.413 /50.368 s;different observed scope,not new bounds |

No count,target,cap,latency limit or validator was changed after observing these
results. No retry/reset/extension or extra hardware run occurred.

## Actual elapsed time / retrieval scope

| Seconds | P1 | P2 |
|---|---:|---:|
| Host program upper bound |500.720928 |500.725240 |
| Native execute interval |503.375653 |534.238306 |
| Margin against native600 s |96.624347 |65.761694 |
| Container start-to-die |511.587126 |549.632006 |
| Margin against600 s using that container window |88.412874 |50.367994 |
| Container create-to-destroy |511.683584 |549.732045 |
| Full outside launcher wall |520.14 |565.41 |
| Container before native |1.920015 |8.008287 |
| Container after native |6.291458 |7.385413 |
| Native retrieval/verify/close/publish aggregate |0.573805 |0.468980 |
| Python frame analysis |5.265384 |6.198835 |
| Snapshot wait |0.001455 |0.001513 |

Container start-to-die excludes Docker creation and surrounding wrapper work;
it is not the exact timeout command's start-to-return duration. Outside launcher
includes checks/control reread and final reread,not all charged to the container
cap. The actual600 s watchdogs were unchanged and did not fire. Native phase
boundaries do not isolate DMA from checksum,compact-file writing,device close
and atomic result publication. Both runs actually retrieved all2048000000 device
bytes in chunks;the measured inclusive aggregates are available,not a fabricated
standalone2GB DMA timer. Unseparated startup,wrapper and postprocessing residuals
remain explicit in reference-analysis.json.

## Gates, warnings, health and originals

The exact pre-run head passed6350 full tests/1057 skipped/8subtests,ruff/C901/diff/
ADR guard,CPU-only adopted-v2 launcher dry-run with14 refusal categories,strict
no-device startup and root-to-UID1000 publication. All9 CI checks succeeded and
public Sonar was0issues/0hotspots/gateOK without acceptance/suppression. Fresh
baseline/device/proof/boot/disk/memory/quiescence checks preceded each role and
immediate launch checks repeated zero containers/quiet host. Source preservation
verified70historical files and112 native/device/header/build-input files.

Each run retained only7 recognized baseline warning lines across wrapper/native
streams,with no unexpected warnings/errors. Both cards returned800 MHz/healthy
DRAM and zero GDDR uncorrected errors;containers remaining0,resets0,retries0.
Raw lifecycle evidence and originals remain private/local;public derivatives
contain only normalized host/project paths. No hostname or private user path
appears in public evidence.

Each archive holds16 exact published payloads,including the64000024-byte frame
companion,transition companion,native envelope/streams,raw environment,frozen
snapshot,v2 receipt/session/host proof and durable role/entry/completion metadata.
P1/P2 gzip sizes are20566491/27411975 bytes,below the conservative50 MiB/file
publication bound. Every payload size/hash,source mode and ordinary extracted
loader replay passed. Originals and their permissions remain untouched under
the ignored local session tree;the private0600 native transport is not chmodded.
Preflight disk headroom was about115.7/115.5 GiB and memory available51.7/51.6 GiB.
These are availability observations,not actual peak RSS. The separate110.60 MiB
CPU-fixture peak is not relabeled as native/runtime memory consumption.
Raw padded2GB pages are not a retained artifact:the unchanged native streams
and emits only compact first32-word rows. No uncompressed64 MB Git blob or
unapproved external storage was used. Wrapper originals stay local,with hashes
and normalized public derivatives. Manifests pin all payload/archive hashes.

## AC disposition and limits

| Issue #120 AC | Evidence / disposition |
|---|---|
| Feasibility Q1–Q4/source/short probes |Existing feasibility plan,run8 Watcher/no-Watcher and B4/C1/C2 records;ordinary CQ1/direct-L1 selected;no claim of pinned-NoC capability beyond its recorded preflight |
| Fixed work,bounded repeat memory,first-to-last coverage |22155167 writes/725980512256 CPU operations;source/build identity and strict500000-frame causal proof |
|500000frames/1ms/noWatcher/device0/600s,same-session control,frozen power |Both exact configurations/session/head/receipt,healthy explicit-final snapshots |
| Intervals,overflow,paired outliers,control difference,load evidence |Tables,full archived histograms,comparison and causal lower bounds |
| Predictions before run,no advance threshold,design/index results |PR181 and precommit;new result section12/index;no timing pass/fail threshold |
| New board-free tests,first-kernel Watcher at60s |Existing bounded first-kernel B4 proof unchanged;new adoption/consumption/completion tests and actual archive replay regressions;no device-facing change |

This fills the narrowly scoped measurement objective,not a general timing
certification. Remaining concerns include single-pair order/drift,other machines
and runtime versions,different CPU/PCIe loading,instrumentation costs,clock-sample
resolution,unidentified native cost terms and the existing independent open
issues. The PR was draft at result publication and is now open and non-draft;
review and merge remain separate owner-controlled steps.

### Known error-path class limits — follow-up #193

The measured production native's error paths were not hardened as a whole.
The production plan's Invariant 3 (join before finish, drain before release,
retain the initiating failure) is verified only for cases covered by the
host-sequence tests, not across the entire error-path class. Three known
counterexamples illustrate the class:

1. In host_sequence.hpp, both drains share one try block. If the first finish
   callback throws, the second is not executed.
2. An observer failure can cause a secondary loader cancellation; the join
   callback can report that cancellation instead of the original observer fault.
3. Cleanup timestamp collection can throw between joins, after joins or around
   drains. It can interrupt a join/drain, replace a saved worker failure, or
   leave an unjoined thread that causes abnormal termination even when queue
   calls would otherwise succeed.

Healthy runs do not take these paths. B4 control/loaded, C1, C2, P1 and P2
completed without these errors and completed both drains; their records are
unaffected. This does not establish error-path safety or absence of other defects.

The owner's 2026-11-22 decision keeps the measured native source and digest
unchanged and defers defects confined to this class to
[Issue #193](https://github.com/tetsuh/hekatus/issues/193). The operator recorded
the examples and the class under that principle. Further cases are expected and
belong to #193; this is not an exhaustive list. Correction requires a new
instrument version with its own digest, build receipt and device-free startup
proof before another board measurement. Existing records are not relabeled.

Source comments, including "Drain queued work before releasing its
buffers/program", remain broader than the exceptional-path behavior. They are
intentionally not edited because they are part of the preserved measured source.
No further board run is authorized here.
