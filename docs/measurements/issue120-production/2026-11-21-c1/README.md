# Issue #120 C1 — one bounded Watcher-off calibration

## Authority, identity and outcome

The owner powered on the target after reboot and authorized **only**
`production-v1-calibration-small`:16384frames/1048576writes,32768 CPU operations
and4096bytes/write,device0,Watcher off,native/Docker caps60s,one attempt.
C2/P were not authorized or executed. Harness precommit:
`b38095571dd62742a161662e5d166fafa75f5b41`; the predictions and stop branches are
in the [C1 preparation plan](../../issue120-production-c1-prep-plan.md).

Native SHA256:
`0a36ed6011a8cb71773545751f98767961e1392ffb16244f2e07e4fe709234e6`.
Its prepared build receipt/effective flags/eight protected dependencies,
host-only source proof,unchanged device/kernel/CTA/library/compiler proof,
pinned image and committed C1-only admission all matched. No rebuild was needed.
Fresh matching-head CI had nine successful checks; public Sonar was0/0/OK.
The exact-head device-free publication checker again passed root publisher to
unprivileged reader with snapshot0644 and launcher0.

Session `57c99372bc2a42a7a33845dde446591f` binds the new boot; manifest SHA256
`b6d06c027ec2854f6d2350492e8121989c33fe02c766e3b61aed68162deeb342`.
Attempt: `20261121T0107019969389561539527928Z`.

- Native/container/wrapper/launcher: **0/0/0/0**. No timeout/reset/retry.
- All16384 attempted/produced/consumed frames passed identity/checksum/work;
  overflow/drop/abort/paired outliers: **0/0/0/0**.
- Causal frame/map coverage: **covered**. First-through-last lower bounds:
  16383transfers,67104768bytes,536838144 CPU operations. Whole-worker submitted
  totals1048576transfers/4294967296bytes/34359738368 CPU operations are not all
  attributed to that frame interval.
- Only the three known warning classes:seven matching lines,no new warnings or
  errors. Mailbox/expectation16386,expected peer coordinates and CTA vectors.
- Frozen power coverage complete,no errors. Watcher off; persisted
  `timing_evidence=true` is the instrument's bounded eligibility,not production
  timing acceptance or a product wall-time guarantee.
- Named container `hekatus-bench-15395-31501` exited0 and was destroyed. Both
  cards returned800MHz/healthy DRAM/zero uncorrected GDDR errors; survivors0.

Fresh Linux6.8.0-149-generic/KMD2.12.0,board/serial/PCI/firmware components and
limits matched the owner-approved2026-11-18 baseline. Only serial411c at
PCI0000:06:00.0 was opened for execution. Both detected boards were healthy.
Host quiescence was checked again immediately before launch; no tests/builds/
scans ran concurrently. Earlier preflight stops did not launch hardware. An
unused earlier session was preserved,not reused. A local process-table parser
was corrected before the successful gates; measured source/native bytes did
not change. See `fresh-preflight.json`,`immediate-host-quiet.json` and
`execution-summary.json`.

## Reference frame intervals, no control difference

| Metric | Consumer ticks | Qualified observed-envelope microseconds |
|---|---:|---:|
| P50 |1349998 |999.998519 |
| P99 |1350851 |1000.630370 |
| P99.9 |insufficient |insufficient |
| Maximum |1351279 |1000.947407 |

There are16383 intervals; P99.9 requires20000 under the existing policy.
Clock conversion uses the host-observed envelope's weighted approximately
1350MHz,not a per-frame clock calibration. There is **no same-session control**,
so no interval difference is reported. This is a small calibration,not a500000
frame production result; design.md section12's production result slot stays empty.

## What the17 host samples resolve

The actual committed-prefix trace is inside the original native/primary records.
All samples are host successful submissions,not CQ retirement or device completion.
`reference-analysis.json` retains the16 adjacent slab calculations.

| Observation | Value |
|---|---:|
| Worker elapsed |25.305179491s |
| Inclusive rate |41437.208551writes/s |
| First committed-write observation after worker start |69.164us |
| Main enqueue-return count bracket |0.602707020..0.602707204s |
| Published committed prefix in that184ns bracket |24974writes |
| Prefix mean through the bracket |41436.384796writes/s |
| Adjacent slab minimum/maximum |41174.575019 /41720.916452writes/s |
| Slab mean/population standard deviation |41437.941061 /165.132273writes/s |
| Slab coefficient of variation |0.398505% |
| First8/last8 slab mean |41575.029346 /41300.852775writes/s |

The first slab has65535writes,the remaining slabs65536; rates use actual count
differences. Samples count1 and every65536,17rows total. Roughly2.382% of the
work was already committed at main enqueue return. Therefore C1's entire0.603s
main API duration was **not** a dead worker lead. Removing that whole interval
before computing a steady slope would misclassify real work. This does not
retrospectively supply B4's missing enqueue-return count.

A descriptive fit `sample_elapsed=a+b*count` gives intercept−19.473ms,
slope24.125288us/write and41450.282658writes/s; max absolute residual27.454ms,
RMS14.036ms. The negative intercept is not a physical negative startup. Later
slabs are mildly slower; adjacent slabs are correlated,not independent trial
replicates. Use the measured first-commit delay and explicit segment rates,
not a fitted intercept as a guaranteed fixed lead. There is no measured
500-second rate bound or isolated CPU-vs-PCIe cost from these mixed iterations.
Host-only observation has nonzero cost; B4 used another native and Watcher.

## Card/host timeline and postprocessing

First-through-last completion span:22117050102 consumer ticks,conditionally
16.383000076s at the observed-envelope weighting. Native enqueue-begin through
terminal observation upper bound:17.091292107s. Their0.708292031s difference
mixes clocks/startup/host observation; it is not a measured fixed device delay.
The worker continued8.214048312s after that terminal observation. Coverage is
proved by frame/map causality,not merely these enclosing host spans.

| Non-overlapping or explicitly concurrent component | C1 seconds |
|---|---:|
| Execute start to program enqueue begin |2.134215262 |
| Worker start after enqueue begin |0.000160928 |
| Worker duration (concurrent with frames) |25.305179491 |
| Program terminal-observed envelope (concurrent) |17.091292107 |
| Worker end to drain end |0.000382307 |
| Of that,queue Finish calls |0.000198523 |
| Post-drain native retrieval/verification/close/publication |0.155181941 |
| Python frame analysis |0.200742569 |
| Frozen snapshot wait |1.109143950 |
| Native execute envelope |27.595119929 |
| Container start to die |29.409079368 |
| Container create to destroy |29.500588020 |

Do not add worker and card durations: use setup+max(worker,terminal)+drain+
postprocessing. Container start to native execute was0.364506982s. Native
execute end to container exit was1.449452198s; after Python analysis/snapshot
wait,the unseparated remainder is0.139565679s for record/publication/exit work.
Lifecycle time is a lower bound on the Docker client's charged interval,not
an exact wrapper stopwatch. Launcher reread is outside the measurement timeout;
it is not moved out of the cap by this analysis. Retrieval,first analysis,
snapshot waiting and in-container publication stay within the original cap.

B4's loaded4096-frame post-drain native/Python components were
0.173632572/0.050221829s. A two-point `a+b*F` fit gives:

- Python: b=12.249409us/frame,a=48.249us; conditional500k cost6.124753s.
  This descriptive trend is close to linear at these two sizes,but is not
  a variance/nonlinearity or500k validation.
- Native: b=−1.501516us/frame,a=0.179783s;500k extrapolation would be
  **−0.570975s**,which is physically invalid and must not be used. Fixed work,
  caching/noise and differing contention are unresolved; do not infer negative
  retrieval cost or a precise fixed-fee decomposition.

Both observations have N=64F,so frame and write terms are collinear;
Watcher,native,boot/cache/thermal conditions also change. Two totals cannot identify `a+bF+cN`. Retrieval volume,256-page chunks,
companion hashing/copying,trace JSON rows and publication remain distinct
potential costs. Native output pages would total2.048GB at500k frames. Native
trace rows also grow with N even though sample storage is fixed. No optimization,
cap charging,count or validator change was made.

## Conditional550/530/520s count table — no count adopted

Use same-origin enqueue-begin envelopes. Scaling only499999 frame intervals and
retaining each observed mixed-clock residual yields C1 500.707294337s and
B4 500.782808186s. The old B4 worker-start bound4.878675135s begins102.014us
later than its enqueue-begin bound4.878777149s; this table uses enqueue-begin
consistently,not a changed record. Take D=500.782808186s as an illustrative
maximum,not a bound on future startup/tail/frequency.

A current-trace scenario uses L=69.164us,
`r_fast=max_slab*1.02=42555.334781`,
`r_slow=min_slab*0.98=40351.083518`writes/s,
`N=1+ceil(r_fast*(T-L))`,`slow=L+(N-1)/r_slow`.
L anchors the first observed completed write,including its work; it does not
identify an independent fixed startup delay. The2% margin is a planning
convention,not a confidence interval or proven bound.

| Total fast-side target T | N | Fast-side margin over D | Slow worker |600s margin,R20 |600s margin,R36.873251 |
|---:|---:|---:|---:|---:|---:|
|550s |23405433 |+49.217s |580.045s |−0.045s |−16.918s |
|530s |22554326 |+29.217s |558.952s |+21.048s |+4.175s |
|520s |22128773 |+19.217s |548.406s |+31.594s |+14.721s |

R20 is the existing optimistic20s planning floor. R36.873251 is B4's illustrative
all-linear native+Python500k reserve plus setup/snapshot and5s contingency.
C1's own all-linear reserve is19.105313s,still below that20s floor. Neither
reserve bounds the unseparated costs; the negative native fit is never credited
as savings. These columns show conditional coverage/cap trades,not safe limits.
The550s row fails even R20 by about45ms;530s has only4.175s under the pessimistic
scenario. No automatic shorter-target adoption follows.

As a separate hypothesis stress test,if the old unmeasured B4 lead-removed
proxy46071.287writes/s plus2% governed,the same counts would miss D by
2.038/20.149/29.205s respectively. C1's prefix counts undermine applying the
full-dead-lead interpretation to C1,but cannot prove B4's historical progress
or exclude a faster future run. Keep this unobserved scenario separate from
actual current-trace measurements. Thermal/AICLK changes over500s are also
unbounded. No robust fixed production count is selected.

## C2 proposal and remaining P decisions

The existing candidate `production-v1-calibration-large`,32768frames/1572864
writes,is a reasonable **discussion candidate**,not an approved or admitted run.
Under the current-trace assumptions it gives36.960..38.980s worker time,
conditional frame envelope33.550779s and3.409709s fast-side coverage margin.
A phase reserve scenario11.338110s leaves9.682372s to the60s cap; substituting
R20 leaves only1.020482s. Neither is guaranteed. No count increase is justified
just by the earlier pessimistic proxy; re-evaluate before a separate precommit.

C2 doubles F but increases N by50%,changing N/F from64 to48. It should test
frame/retrieval/chunk volume,Python scaling,25 sparse
samples,repeatability of prefix rates/lead and postprocessing under a longer
Watcher-off envelope. It would still be only about39s,not a500s stress test.
It cannot by itself establish long-run tail/frequency/variance bounds.

**Do not skip C2 and silently select P from C1.** Existing proposal code requires
both fixed calibration sizes and exact program/environment provenance. B4 is
Watcher-on and is not the second eligible calibration. Current C1-only admission
rejects C2/P and C2 session construction. A later admission/precommit changes
harness provenance,and potentially Python identity: owner-approved compatibility
rules are needed before combining such records. Do not relabel C1,ignore
identity/environment differences or weaken validation in this measurement task.
Changing that provenance/admission policy requires the appropriate ADR; no
such decision is made here. Skipping C2 would additionally require explicit
owner replacement of the current two-calibration policy,not a statistical claim
that these17 samples suffice.

Before P1control/P2loaded approval,the owner still needs to settle:

1. C2 scope/count and compatible precommitted admission; remaining warm/cold,
   rate,thermal and postprocessing unknowns to accept or investigate.
2.550s floor versus530/520s alternatives,rate/reserve assumptions and whether
   a fixed N can cover all frames inside unchanged600s. Current code still
   retains the550s policy; this table changes no parameter.
3. The explicit treatment of unresolved500k retrieval/serialization and500s
   clock/host variability,without moving charged work outside the cap.
4. A production receipt/precommit binding actual calibration evidence and N,
   with fresh environment and a new same-session P1control/P2loaded pair using
   identical device-open configuration,only device0,Watcher off and separate
   owner approval. Never infer a control difference from C1 alone.

Hardware stopped after this single C1. No C2/P/admission expansion/reset/retry.

## Evidence inventory

`C1-evidence.tar.gz` contains10 original payloads:primary record,environment,
active/frozen power CSVs,native result/execution/stdout/stderr,and compact frame/
transition companions. `C1-manifest.json` records original size/hash and
independent normalized-log hashes. Tar owner/timestamp metadata is normalized;
payload bytes and original permissions are unchanged. Watcher has no new capture
because it was off. The ordinary persisted loader replays the archive.
Original wrapper output remains private because it contains a local path; only
the normalized derivative is public. Runtime session,lifecycle,fresh/immediate
preflight,post-health and offline arithmetic are separately published. No
machine name or user-specific absolute path is present in public evidence.
