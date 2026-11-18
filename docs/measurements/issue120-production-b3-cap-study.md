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
