# Issue #120 B3 — cap and post-processing study (board-free)

No new board data or calibrated rate is produced. B1 evidence remains unchanged.
The numerical scenarios below are proposals, not approved counts or bounds.

## What the600s cap includes

[Issue #120 AC](https://github.com/tetsuh/hekatus/issues/120) specifies500000
frames, the #118 conditions and600s outer cap. [Issue #118 AC](https://github.com/tetsuh/hekatus/issues/118)
originally required600s and later explicitly allowed800s for the AICLK-depressed
on-chip loaded condition. That exception is not authority for host-load #120.
1350000 ticks are retained; a fixed1ms product wall-time period is not guaranteed.

Actual boundaries:

| Boundary | Included | Outside |
|---|---|---|
| Wrapper `timeout ... docker run` | container start, Python/device setup, work, drain, retrieval, producer analysis, frozen-power wait, record write, container exit | earlier wrapper environment/sampler setup; later sampler shutdown/cleanup |
| Native process timeout | device setup, threads/enqueue, join/drain, page retrieval, native checksum/compaction/fsync, device close/result publication/process completion | Python asset precheck and record analysis |
| Native `program_host_interval` | enqueue-begin to sampled L1 terminal-observed upper bound, including cold enqueue/JIT lead | later worker completion/join/drain/retrieval/close |
| Host launcher | wrapper plus post-container reread/recomputation; separate cap+120s watchdog | not a600s end-to-end success-time definition |

Sources: `run_in_container.sh:171-198,896-945,1432-1504`;
`issue120_production/runner.py:43-60,100-108`;
`issue120_results.py:83-104`; `issue120_production/native.cpp:216-328`.
The production launcher explicitly sets the generic wrapper's timeout to60/600s;
its generic default900s is not used. TERM's5s kill-after grace is not an extra
success budget. The native observer's own cap is not the whole-process cap.

#118 `run_resident.py:2451-2497,2503-2598,2663-2900` closes the device, freezes the
trace, computes metrics/builds and validates/writes the record within that same
container invocation. Moving those steps outside Docker therefore changes which
work is charged to the #118-compatible cap, even if frame arithmetic is unchanged.
The current production host already repeats validation outside the container;
this does not mean the producer's first analysis was outside its cap.

**No boundary is changed here.** A split execution-versus-publication cap needs
owner agreement on AC/comparability, bounded deadlines for both phases, exact
raw-evidence freeze time and distinct execution/record success statuses.

## What the59.724us/frame actually contains

The original B1 record gives:

| Recorded/derived phase | B1 seconds | Amortized us/frame |
|---|---:|---:|
| Native drain-end to Python execute-return: download, native validation/compaction, close/publication/process wait |0.200900834 |49.048055 |
| Python `coverage.analyze` |0.043730617 |10.676420 |
| Sum |0.244631451 |59.724475 |
| Setup before enqueue stamp |2.884949817 | not a frame slope |
| Snapshot terminal-sample wait |0.856223480 | not a frame slope |

This split follows `recording.make_record`'s recorded timestamps/formula, not a
new performance experiment. The source now retains the two component times in
fresh v2 records instead of only their amortized sum. The old B1 record is not
edited. Native close/log timestamps suggest an appreciable fixed component, but
there is no separate authoritative close timer to subtract as a known constant.
The59.7us term also omits some public copying/hashing/serialization work, so it
is not a complete post-processing budget.

At500000 frames, linear Python analysis alone would be5.338210s. The other
24.524028s from scaling the native component is **not a measured linear cost**;
it scales teardown/fixed work too. A model must retain unknown byte/chunk costs,
not assume that the native component becomes a constant either. B1's whole4.898s
program upper bound similarly includes cold/fixed lead; scaling it to597.900s
is not a production frame-span measurement.

## Optimization options, not silently implemented changes

Current native retrieval (`native.cpp:270-293`) reads4096 bytes/frame in256-page
chunks and emits128-byte frame rows. Production means2.048GB read,1954 chunks
and64MB compact row output, plus headers/map. The consumer's original timestamp
page layout remains the #118 layout.

| Option | Potential benefit | Required caution |
|---|---|---|
| Larger chunks, e.g.1024 pages |1954 to489 read calls,1MiB to4MiB temporary buffer | no measured speedup; native digest and memory config change; recalibrate |
| Batch compact file writes | fewer write syscalls, same bytes/checks | native digest changes; exclusive/fsync/short-write behavior must be preserved |
| Read only128 useful bytes/page | much less PCIe data | requires proven API/alignment/strided-read behavior; first64MB is not equivalent to all page prefixes |
| Compact device output | less device storage/transfer | kernel/data layout and observation load change; new instrument/approval |
| Avoid duplicate Python object work | possible same-algorithm speedup | preserve every frame/checksum/coverage/quantile; verify exact results |
| Move first analysis to host | removes that pass from Docker cap | changes cap charging; keep independent deadlines/evidence status; owner decision |
| Overlap retrieval/analysis with frames | may hide cost | adds CPU/PCIe activity inside measurement; not condition-equivalent |

Only publication/provenance repairs and extra descriptive phase fields are
implemented now. No retrieval amount/chunk, kernel/native code,550s target,
frame count, timeout or latency threshold changes.

## N covers frames, not post-processing

Let D be load-start through last frame plus the fresh tail response, r+ the
planning-fast worker rate, r- the planning-slow rate, and R non-overlapping cap
costs. The sufficient count/time model requires

`ceil(r+ * D) <= N <= floor(r- * (600-R))`.

The current proposal already uses overhead only for the cap feasibility check,
not as work that N must cover (`records.calibration_proposal`). Its target is
`max(550, projected_program_span*1.02+10)`;550 is a conservative planning floor,
not an AC or a required post-processing interval. A lower floor would be a new
offline count-rule/precommit decision, not an arbitrary CLI override.

For1350MHz reference pacing, D is roughly500s plus100ms startup, enqueue/cold
lead relative to load start and tail/observation uncertainty.500.1s itself is
not a measured lower bound for every run. Rate/clock variation and first/last
causal observations still matter; enough total work alone cannot prove coverage.

### Sensitivity using the actual old run8 rate model

Use the two distinct old observations, not one common nominal rate:
`r+ = (262144/6.017380392)*1.02`,
`r- = (262144/6.040565943)*0.98`.
Their ratio is about1.044827, **not merely1.02/0.98**. No rate of the new loaded
baseline has yet been measured. R20 is an optimistic planning scenario; R38.603
is the pessimistic B1 all-linear-amortization scenario, neither a guaranteed bound.

| Fast target T | N=ceil(r+T) | Margin above500.1s reference | Slow worker s | Cap headroom R20 | Cap headroom R38.603 |
|---:|---:|---:|---:|---:|---:|
|550 |24439669 |49.9s |574.655 |+5.345s |−13.258s |
|540 |23995312 |39.9s |564.206 |+15.794s |−2.810s |
|530 |23550954 |29.9s |553.758 |+26.242s |+7.638s |
|520 |23106596 |19.9s |543.310 |+36.690s |+18.087s |
|510 |22662239 |9.9s |532.862 |+47.138s |+28.535s |
|505 |22440060 |4.9s |527.637 |+52.363s |+33.759s |
|500 |22217881 |−0.1s |522.413 |+57.587s |+38.983s |

Startup/lead/tail and any longer actual frame span consume the displayed margin.
The500s row is not sufficient even for the simple500.1s reference.530s might be
feasible under these assumptions; it is not adopted. Small-sample±2% is a model
cushion, not a thermal/OS-scheduling guarantee for a500-second run.

## What C must determine and owner choices

After a new successful B1 then B2, C still needs separate approval. Proposed
points remain16384 frames/1048576 writes and32768 frames/1572864 writes,60s each,
Watcher off. Record separately:

- Load elapsed/rate, pre-frame lead, first-to-last span/tail proof, join/drain.
- Native drain-to-process-return, Python analysis, source-copy/hash/write and
  snapshot wait costs; bytes/chunk count, cold/warm/JIT state and AICLK.
- Fixed versus frame/byte-scaled slopes and cap time with no double-counting
  overlapped work. Two points can fit a phase's intercept/slope, not establish
  variance/nonlinearity or jointly identify a+bF+cN from two total times.

If no conservative count fits: choose same-semantics optimization plus new
calibration, a coverage-supported shorter target, a separately approved larger
cap, fewer frames with changed AC/sample sufficiency, split cap semantics with
bounded publication, or no run. Extending the #118800s exception, moving cap
boundaries, or changing500000 frames requires an owner decision; none follows
from this study. No B/C/P execution is authorized here.

## B4 observed-baseline update

B4's [original v2 control/loaded archives](issue120-production/2026-11-18-b4/README.md)
at9873978 were healthy; hardware stopped after B2, no C/P. Both Watcher rows are
reference evidence only. This update preserves the old run8 table above rather
than presenting its rates as measurements of the new Linux149/KMD2.12 baseline.

B2 measured262144 writes in6.369850819s: **41153.868writes/s inclusive**. The
load-start-to-terminal-observed upper bound is4.878675135s. Raw frame first/last
span5528250346 ticks converts to about4.095000256s with the observed-envelope
weighted1350MHz, not an exact frame wall-time clock. A conditional500k estimate
scales only499999 frame intervals, then keeps the mixed-clock residual fixed:
499.999031294+0.783674879=500.782706172s. No residual clock/tail constant is proven.

B2 post-drain native retrieval/verify/close/publication0.173632572s and Python
analysis0.050221829s total0.223854401s,54.652us/frame **amortized**. Linear500k
components would be21.195382s and6.130594s; the native term includes fixed close
and is not a measured slope. B1 components0.177773968/0.041842053s support only
a bounded phase comparison, not a fitted scaling law.

Using the same existing formula, Rlinear=setup2.432002305+snapshot2.115272208+
27.325976685 linear post-processing+5 reserve=**36.873251198s**. R20 remains the
optimistic planning floor. Both are scenarios, not upper/lower guarantees; extra
public copy/hash/serialization costs and long-run behavior remain unknown.
Use a common time origin: setup plus max(worker end, terminal observation),
then non-overlapping drain/retrieval/analysis/snapshot phases. Do not add the
concurrent worker and frame spans, or count a phase again in the reserve.
A worker that ends early can fit a cap while failing causal coverage.

### A fixed-worker-start ambiguity matters more than small rounding

Main enqueue returns0.679886107s after worker start. If that entire interval
were fixed worker stall, the remaining-duration proxy is
`262144/(6.369850819-0.679886107)=46071.287writes/s`.
This is **not a measured steady rate or a bound**: no cumulative submission
counter was sampled at enqueue-return, and contention may vary. The raw inclusive
average contains that possible fixed lead. Extrapolating it as pure per-write
cost could make work finish much earlier than a500-second frame window.

| Total fast-side worker target T | N from inclusive average+2% | N from lead-removed proxy+2% | Fast-side frame-envelope margin | Inclusive-model600s margin R20 | Inclusive-model margin Rlinear |
|---:|---:|---:|---:|---:|---:|
|550s |23087320 |25814043 |+49.217s |+7.551s |−9.322s |
|530s |22247782 |24874189 |+29.217s |+28.367s |+11.494s |
|520s |21828012 |24404262 |+19.217s |+38.776s |+21.902s |

Both target definitions use total worker time. With L=0.679886107s, proxy N is
`ceil(proxy*1.02*(T-L))`, and proxy slow elapsed is `L+N/(proxy*0.98)`.
Its Rlinear cap margins are−9.294/+11.522/+21.930s. Do not omit L or confuse
a post-lead processing target with a total-worker target.

The two cap columns shown are specifically the inclusive model, whose slow
host time is approximately T*1.02/0.98. Similar paired-model margins are
arithmetic, **not empirical agreement between the models**. The displayed
coverage margin is at the planning-fast rate, not the observed average rate
(which predicts about1.02T). It applies only if rate/residual assumptions
extrapolate. B4 is not a new calibrated count.

Stress the assumptions separately:

| Target | Inclusive N coverage margin if fixed lead + fast proxy governs | Proxy-fast N cap margin if inclusive-slow rate governs, Rlinear |
|---:|---:|---:|
|550s |−8.807s |−76.931s |
|530s |−26.672s |−53.628s |
|520s |−35.605s |−41.976s |

These are unresolved hypothesis checks, not demonstrated bounds or a claim that
the extremes must occur together. They show why a positive530/520s cap margin
cannot establish a count that covers frames. No robust fixed count has been
established across this ambiguity; do not silently adopt a shorter target.

### C preparation, still not approval

| Existing proposed identity | Frames/writes | Inclusive worker forecast | Fixed-lead proxy forecast | Fixed-lead fast-proxy forecast | Conditional terminal envelope | Fast-proxy frame margin |
|---|---|---:|---:|---:|---:|---:|
|`production-v1-calibration-small` |16384/1048576 |25.479s |23.440s |22.993s |17.167s |+5.827s |
|`production-v1-calibration-large` |32768/1572864 |38.219s |34.820s |34.150s |33.551s |+0.600s |

Both are still60s/Watcher off and have not run. No-Watcher may have a faster
worker than B4; C2's0.600s illustrative margin is not a safe guaranteed cushion.
An owner can approve C1 first and make C2 conditional on its observed rate/span,
or request a new C2 count/precommit. An unadopted1835008-write C2 example gives
39.729s under the fast proxy and6.178s frame margin, but inclusive-slow worker
45.499s plus the same R20 planning scenario is **65.499s**, beyond60s. A smaller
phase-only cost estimate gives56.837s but omits unknown costs; it does not establish
cap feasibility. Count increases are therefore not a ready solution.
Changing counts is not an arbitrary CLI override: it needs fresh committed plan
and admission/program-identity compatibility review. Do not rewrite B4 identities.

C must separately observe worker enqueue lead and elapsed time, frame span/tail,
join/drain, native post-drain, Python analysis, publication/snapshot costs and
cold/warm/AICLK conditions. A two-point worker intercept+slope in N is conditional
on like contention; F also changes, so two total times do not identify a+bF+cN,
variance or nonlinear500-second behavior. A missed C2 endpoint is not repaired
by extrapolation or adaptive work. The current550s floor and automatic proposal
remain unchanged; only eligible C plus a later precommit/owner decision can
select production work. No bounded C admission or execution is granted here.
