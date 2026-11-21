# Issue #120 production preparation (board-free stage A)

Status: design/implementation preparation only. No board access, calibration,
production result, timing acceptance, or execution approval is provided here.
Base: PR #172 merged as `8c96cdc`; preserve all of its measured artifacts and
sources. Production completes the remaining Issue #120 scope in a later
owner-authorized phase of this PR.

## Completed production follow-up — 2026-11-21

The stage-A status above is historical. The owner later adopted520s/22155167writes
and approved the single conditional pair under the
[dated adoption precommit](issue120-production-p-adoption-plan.md). Both actual
500000-frame control/loaded runs completed at4fd4900 in one session,full integrity
and loaded causal coverage,all exits0,zero abort/drop/overflow/paired outliers,
baseline warnings only and healthy return without reset/retry. See the
[immutable production archive](issue120-production/2026-11-21-p/README.md) for
full distributions,prediction contrasts,actual cap/retrieval costs and limits.
No new advance latency threshold,population bound or product timing guarantee
is inferred;original stage/calibration decisions and records remain unchanged.

## Owner-approved execution baseline — 2026-11-18

The owner approved **Linux6.8.0-149-generic and TT-KMD2.12.0 on2026-11-18**
as the new production-measurement baseline and authorized B1, followed by B2
only after a healthy B1. C and P remain unauthorized. The earlier run8 evidence
used Linux6.8.0-138-generic/TT-KMD2.11.0; do not claim an identical-environment
comparison with it. Its rates remain uncalibrated planning inputs only.

Keep all other identity requirements: device0, p150a serial000004033191411c,
PCI0000:06:00.0, firmware bundle19.6.0.0, the pinned image and existing native/
kernel proof digests, idle800MHz and healthy DRAM on both detected cards. Take
fresh environment snapshots rather than reuse the previous preflight. No old
kernel/KMD value is hardcoded in the production admission code or tests; no
algorithm/schema change or rebuild is needed for this baseline decision.

Use one new runtime session manifest for both B rows, retaining its boot,
program/image/configuration bindings. The session archive additionally records
this dated owner decision, the fresh full environment baseline and the exact
runtime-manifest SHA-256. Before each launch compare the current environment
against that baseline and boot binding; retain each wrapper environment snapshot
with its record. B2 must use the unchanged manifest and matching environment,
not an older control. The runtime manifest is not edited between runs.

The previous B1 exercised KMD2.12.0 with tt-metal v0.75.0 successfully at the
native/container layer but failed host reread; it is not a reusable successful
invocation. Any device-open anomaly or warning/error outside the three exact
#176 baseline classes stops the renewed B1;
changed wording is not automatically accepted. Both B runs retain4096 frames,
Watcher and60s native/outer caps, named containers and pre/post lifecycle/idle
checks. No automatic retry; the owner permits at most one reset in this B unit,
only after timeout/abnormal exit. A healthy B2 still grants no C/P authority.

## Invariants (before tests and implementation)

1. Preserve the Issue #118 resident synthetic payload, cadence, drop-new ring,
   persistent 64-iteration consumer checksum, completion timestamps and metrics.
   Add a separately versioned observation adapter, not a relabeled Q4 toy run.
   Old producer/consumer, runner, load generators, native probes and validators
   remain byte-identical. Generated derived kernels must verify their base hashes
   and exact hook sites before compilation; unexpected source drift fails.
2. Control and loaded use identical device program, complete CTA/runtime argument
   layout, CQ count2, resident/transfer subdevice partition, stall group, buffers,
   pacing/work and observer reads. Only the fixed host CPU/CQ1 worker is absent
   in control. Pair execution has one session manifest; unlike-session/config,
   revision/image/device or source hashes cannot supply a control difference.
3. Start the host worker before workload enqueue; join it before **any** CQ0/CQ1
   finish/Synchronize. Do not hold the MeshDevice API lock waiting for CQ0 while
   another thread must enqueue CQ1. Test the actual host order with callables,
   shared-lock fakes and exception paths, not source grep. Cancel and join after
   partial start/enqueue/worker failures, then drain before releasing resources;
   preserve the original failure instead of publishing a successful result.
4. L1 ring/control/anchor/mailbox allocations carry explicit
   BufferDistributionSpec. Validate core/pages/bank pack and complete compiled
   CTA vectors before dispatch. Every cross-core data/semaphore destination uses
   host virtual WORKER xy passed as runtime arguments, not accessor-derived xy.
5. Work is a fixed integer repetition count; each repetition reads the card
   challenge, executes32x1024 integer operations and enqueues one4KiB ordinary
   write. No per-repeat finish/readback/event, time-adaptive work or success retry.
   Storage is bounded independently of transfer count.
6. Keep causal startup, each real-frame interval and the post-last-frame tail in
   one consumer clock. Three fixed page reads per observation, no wait-until-load
   success inside frame work. Missing/torn/stale progress, alias, checksum/frame
   failure, overflow/abort, ambiguous clock or a missed endpoint cannot be covered.
   Diagnostics, host envelopes and submitted totals cannot produce coverage.
7. Version long-run wire/record/evidence schemas separately. New host interval
   rules do not retroactively tighten accepted records. Preserve legacy bytes and
   classify old schemas explicitly as out of scope of the new validator.
8. New diagnostics distinguish actual consumed-frame expectation (consumed+2)
   from planned full-stream expectation (configured frames+2). Zero/incomplete/
   full streams have tests. Neither diagnostic decides coverage.
9. First new-kernel board execution requires Watcher and60s outer/native caps.
   All proposed bounded runs are <=60s. Production control/loaded each need a
   separate owner approval unit, Watcher off, device0 and600s host-monotonic cap.
   No auto retry/reset, warning suppression or CPU-only fallback.
10. Keep raw ticks and AICLK provenance separate (ADR-0015). Nominal1350000-tick
    pacing is not a product wall-time guarantee. Freeze power bytes at retrieval;
    incomplete trace/end coverage cannot be timing evidence. No advance latency
    percentile/max pass threshold.

## Source decision and integration

`run_resident.py:294-423` constructs the #118 resident producer/consumer program;
`:460-620` allocates and decodes its timestamp/stat pages. The no-table producer
writes1024 packed int16-complex words per frame, advances cadence without waiting
on a full ring, and reports attempted/produced/dropped counts. The consumer keeps
a persistent checksum over64 words and records designated-core completion ticks
and work bounds (`kernels/resident_{producer,consumer}.cpp`). Q4 v3 differs: its
payload/checksum and four-semaphore protocol are not the #118 workload. It must
not simply become a500000-frame substitute.

Use an additive native production host plus versioned, reproducible adapters of
those #118 kernel sources. Retain their work and accounting; add runtime-peer
routing, a bounded startup start semaphore and v3-meaning observation hooks.
Record frame admission, observe progress, then sample the original work-start;
keep completion immediately after the unchanged arithmetic. Work duration stays
separate from observation duration. The observation cost remains visible in
completion intervals and is present in **both** conditions. The instrument is not
claimed binary-identical to the historical #118 measurement, nor are its absolute
latencies a retrospective replacement. The controlled difference measures the
presence of host CPU/PCIe work with the same new instrument.

Reuse the existing Python `resident_harness.frame_interval_statistics` and
`resident_record.detect_pairs` algorithms for #118 metrics, and #111 snapshot
utilities for immutable power/AICLK facts. Keep the old non-calibration host
condition rejection intact. A new runner/record schema owns the integration;
legacy `run_host_cpu_transfer_work`/calibration does not become accepted evidence.
Reuse ADR-0018 atomic native envelope/raw-stream transport without stdout parsing.
Large compact binary frame/transition sidecars are exclusive, attempt-bound,
size/hash-bound and read completely before a successful record is accepted.

The lock audit in `docs/design.md` section12 and the #178 diagnostic plan already
establishes shared per-MeshDevice API locking. The existing threaded #118 path
starts the worker at `run_resident.py:999-1011`, synchronizes CQ0 at:1043, and
joins afterward at:1071. The new native callable seam instead orders start,
enqueue, join, CQ0 finish, CQ1 finish, retrieval. No claimed historical timing
cause is inferred beyond the source-backed candidate recorded in PR #172.

Use two queues and the same two-subdevice layout as successful Q4: resident pair
logical(0,0)/(1,0), transfer partition(2,0), transfer-only stall group. Same image,
native/kernel digests and configuration digest bind both session rows. The
future pair orchestrator stops after an invalid control instead of silently
substituting an older record. Matching IDs alone is not sufficient: immutable
pair-manifest/run bindings and actual metadata must agree.

## Long-run protocol (proposed proof and board-free checks)

Let F be configured frames and N fixed write repetitions. Card challenges range
from0 through F+2: startup1, per-frame2..F+1, tailF+2. Sequence is1..N.
The short19/13-bit layout is not reused:

- F=500000 => F+2=500002 needs19 bits, not13.
- The provisional N=24439669 needs25 bits, not19.
- Use two32-bit limbs (64-bit logical domain), no64-bit DMA atomicity assumption.
  Proposed wire limbs are s and t=s+c, decoded as c=t-s without wrapping.
  Require N+F+2 <= UINT32_MAX before dispatch; all arithmetic bounds are checked
  in64 bits. Thus sequence, challenge and the sum do not alias in an attempt.
- Each page word is an XOR-by-index transform of either s or t. Host challenges
  must never decrease; each emitted s and t therefore strictly increases. A
  bounded host transition table records(first_sequence,challenge) on each change,
  at most F+3 entries (including0), rather than logging every transfer.
- Whole-page checks alone are **not sufficient** for multiword coherence. The
  evaluator additionally binds every observed tuple (startup, both frame bounds,
  tail) to that exact host transition table. An un-emitted splice is rejected.
  The mapping is source/attempt/hash-bound, validated and part of the evidence.
  This is not a claim about arbitrary/malicious writers or validator completeness.
- An advancing fully validated observation witnesses all4096 bytes arriving by
  scan completion, **not CQ retirement or an atomic page snapshot**. Attribute
  those bytes only to fresh-challenge-issue through response-scan-completion,
  never between the two scan-end timestamps. Require challenge issue after the
  preceding completion and response scan after current admission/before current
  completion. Frame-zero bytes are not automatically inside the frame window;
  later frame challenges are. Tail bytes are separately after the last frame.
  Do not multiply unseen sequence gaps or double-count sequences.
- This sufficient proof assumes coherent32-bit observations, monotonic host
  challenges, no unrelated writer, encoded(0,0) initialization, source-buffer
  copy before reuse, and scan timestamps after completed reads. These are not
  proofs of arbitrary byte-tearing hardware, continuous CPU/PCIe occupancy, or
  product wall time. Final map lookup is bounded by successful submissions;
  unsuccessful/pending enqueue suffixes never become evidence. Bind the exact
  startup observation used for start, not a later replacement.

The protocol proof is subject to explicit board-free counterexample tests before
implementation: old-mask boundaries, max/new values, wrap/overflow admission,
monotonic mapping/partition coverage, torn tuples and pages, forged/changing
challenge associations, first/last gaps, and valid intervals. A new ADR records
this architecture/schema decision; no change to the landed ADR-0018 decision.

Frame storage remains the fixed #118 F-page timestamp allocation (500000x4096 =
2048000000 bytes), not N queued transfers. Use one reusable4KiB host payload,
fixed scratch/CB/ring pages, bounded chunked retrieval, compact frame sidecars
and a preallocated(F+3)x8-byte transition table (4000024 bytes at production F).
These grow with the separately fixed frame capacity, **not transfer repetitions**.
Runtime queue backpressure/rings remain the source-bound storage limit; queue
occupancy is not measured or described as an exact in-flight count.

## Fixed-work estimate and calibration admission

Run8 P6/P7 completed262144 writes in6.040565943/6.017380392 host seconds. Baseline
rates are43397.258..43564.472 writes/s. They are observations, not throughput
bounds for a new kernel/protocol/500-second run.

Provisional planning model (not a statistical confidence interval):

- Fast planning rate =43564.472x1.02; require550s of fixed host work at that rate.
- N=ceil(550x43564.472x1.02)=**24439669**.
- CPU operations N x32768 =**800839073792**.
- Submitted H2D bytes N x4096 =**100104884224** (not all claimed in-window).
- At unadjusted run8 rates:561.000..563.162s. At the slow planning rate
  43397.258x0.98:574.655s, leaving25.345s under600s. Reserving20s for setup,
  retrieval, analysis and snapshot leaves5.345s; this is provisional, not a promise.
- The550s fast-rate target is10% over the500s reference frame span. It also has
  to cover pre-launch host work and any observed AICLK-related span extension.
  Calibration must recompute that coverage/cap feasibility from explicit source,
  margins and overhead; if no conservative count satisfies both, **stop** and
  seek a new owner decision/precommit, not an adaptive run.

Propose two no-Watcher fixed-work calibration sizes after first-kernel validation:
1048576 writes with16384 frames (~24s host/~16.4s reference frames), then1572864
writes with32768 frames (~36s host/~32.8s frames). Each is capped60s. Record fixed
work elapsed, source hashes, causal coverage, clock ambiguity, in-window page
lower bounds, observed power/AICLK and setup/retrieval overhead. Repeat-count
selection is an offline decision followed by a new immutable precommit; a
production-admission artifact must bind validated calibration records, exact
native/configuration hashes and fixed candidate counts. Stage A cannot produce
that artifact from nonexistent observations.

If host work truly ends before the last frame, record not_covered; do not infer
coverage from later drain. A missed response can also be a genuine scheduling/
progress gap, not insufficient total work. Cap/abnormal exit is failure with raw
streams/status retained, no automatic retry, no production continuation. Slower
AICLK may make the required fixed tick-domain frame run exceed600s; no wall-time
clock assumption or period/count substitution is allowed.

## New record validation and diagnostics (#180 / #179)

Add a new production schema only. Require host_interval endpoints to match
host_cpu_work endpoints exactly; for loaded rows, ordered load endpoints end no
later than load_join_completed_ns, and join precedes either finish. Preserve
post-join resident host intervals as upper-bound envelopes, never exact resident
completion. Bind observed card completion separately if recorded. Validate types,
counts, protocol/map/sidecar bounds and attempt/configuration association.

Legacy `issue120-probe-v1` records continue through the unchanged validator, or
are explicitly out of scope of this new schema. Tests include the actual accepted
P6/P7 records without changing bytes, and a synthetic post-join host-work interval
that must fail under the new schema. Validator completeness is not claimed.
New diagnostics report consumed+2 and configured+2 separately, with tests for
zero/incomplete/full streams; the legacy expected2 artifacts remain unchanged.

## Metrics, predictions and later result slots

Retain raw card ticks; #118 P50/P99/P99.9/P99.99/max, histogram, overflow/drop/
abort counts, #104 paired short/long outliers, startup/work bounds, checksum
counts, host/load/device interval provenance and frozen #111 power/AICLK.
Report loaded-minus-same-session-control metrics only after both rows pass their
configuration/evidence checks. Include submitted counts/bytes and conservative
in-window completed page/byte lower bounds, not a queued-count substitute.

Before any run, publish its identity, exact work, hashes, qualitative prediction
and branches in the PR. Expected healthy branch: intact frames, covered loaded
window, no new warnings/Watcher anomalies; CPU/DMA contention may widen intervals
but no latency threshold is set. Alternative branches: genuine not_covered,
not_evaluable/torn/clock/frame failure, or timeout/abnormal exit. All stop before
production except a specifically authorized calibration sequence's healthy branch.

Later results belong in `docs/design.md` section12 and the measurement README.
Stage A adds only a pointer/reserved result slot: **no result is present**.

## Proposed board runs (not executed)

All require fresh device0 identity/image/head/native/config hashes, quiescent
host, no surviving container, named lifecycle capture, frozen power, only the
previously accepted #176 warning classes, and approved cleanup/health rules.
Any new warning/error, route/CTA mismatch, malformed evidence, failed checksum,
clock ambiguity, timeout or survivor stops. No retry/reset is authorized here.

| Approval unit / identity | Purpose / fixed work | Cap | Watcher | Prediction and next branch |
|---|---|---|---|---|
| B1 `production-v2-control-watch-b4` |4096-frame new resident adapter/control, no host worker |60s | on | intact frames/healthy instrument permits only approved B2; otherwise stop |
| B2 `production-v2-load-watch-b4` |4096 frames,262144 writes, new wide/map protocol |60s | on | covered/intact/healthy permits separately approved calibration; missing startup/interval/tail stops |
| C1 `production-v1-calibration-small` |16384 frames,1048576 writes |60s | off | covered and fixed-work/overhead measurements; insufficient/evidence/cap branch stops |
| C2 `production-v1-calibration-large` |32768 frames,1572864 writes |60s | off | covered plus rate/overhead agreement allows offline count proposal, not production |
| P1 `production-v1-control-500000` |500000 frames, same program/open config, no host worker |600s | off | valid control and complete frozen trace required before loaded counterpart |
| P2 `production-v1-loaded-500000` |500000 frames, candidate24439669 writes only after calibration/precommit |600s | off | covered => report metrics; not_covered/not_evaluable/cap => preserve and stop |

B1/B2, C1/C2 and P1/P2 are **separate owner approval units**, not authorization
from a successful previous stage. Both production rows require explicit separate
owner approval and one same-session/configuration manifest. Host launch verifies
the boot identity; that host-verified manifest binds container execution without
assuming a worktree's external Git directory is visible inside the container. These are planned
identities; paths must be new, never resumed/overwritten. Native internal caps
and outer wrapper caps must match; an omitted generic wrapper timeout is not an
approved production command. Stage A will not run any of them.

## Post-B1 publication/power refinement (board-free, next run only)

[B3 invariants](issue120-production-b3-plan.md) precede RED and implementation of
#182/#183. New outer records use `issue120-production-v2`; native/kernel/wire,
work and open configuration do not change. The v2 explicit power contract has
its own enqueue-to-terminal-observed upper-bound label. Legacy v1/B1 records
still reproduce their original frozen interpretation and cannot be reused for
new stage admissions. Watcher remains timing-ineligible; non-Watcher v2 timing
also requires complete coverage and error-free positive finite weighted facts.

The public snapshot is created0644 before its exclusive atomic link, along
with explicit readable modes on new public record/capture/Watcher companions.
Existing files are never chmodded. The shared legacy snapshot default and
ADR-0018 private native result mode/transport remain unchanged. This follows the
#172 root-container/public-companion model rather than changing UID/device/JIT
cache behavior. No privileged flags, broad ownership repair or writable mounts
are added. #172 did not perform the new host frozen-snapshot reread; production
must test that post-child boundary with a mandatory permission seam and an
actual root-publisher/unprivileged-reader no-device fixture check.

A rerun starts from B1 with a new session and separately authorized B scope;
no repaired reader changes the old invocation's exit2 into success. C/P remain
separate approvals. No cap, frame-count or550s work target is changed by these
reader/publication fixes; the B3 report separates cap options from authorization.

## Renewed B4 owner authorization

The owner authorized a new B1 then healthy-conditional B2 after the B3 repair;
[the B4 precommit](issue120-production-b4-plan.md) records fresh canonical
identities, predictions, exact stop branches and baseline/preflight gates. These
replace only the current B launch labels, not old evidence or the native work.
Final clean/pushed-head CI/Sonar0/0/OK and publication checker precede hardware.
The old exit2 invocation/session is not resumed; C/P still require approval.

## C1 host-trace preparation, hardware paused

The owner retained approval for only `production-v1-calibration-small`,16384
frames/1048576writes,60s/Watcher off/device0, then powered down the board and
restricted the present task to [board-free C1 preparation](issue120-production-c1-prep-plan.md).
No C1/device/tt-smi/preflight/session activation occurs now. C2/P remain excluded.

[ADR-0020](../adr/0020-issue120-host-trace-c1-admission.md) carries the additive
host-trace successor and explicit B4-old/current-new C1-only admission bridge.
The original measured native/protocol files and all B4 bytes remain unchanged;
new samples are host committed-prefix diagnostics, never card coverage. Actual
native digest, exact source-pair/device-proof checks and offline validation are
recorded in the preparation proof. A new native is not hardware-validated by
this preparation. The owner's Watcher-off exception applies only while device
content/dispatch stay unchanged. Next power-on requires fresh baseline/boot
preflight and a new C1-only session at a clean/pushed green head; do not reuse
B4's session or treat the admission receipt as present execution permission.

## C1 bounded outcome — 2026-11-21

After the owner powered on the target and reauthorized fresh preflight,the
[single C1 archive](issue120-production/2026-11-21-c1/README.md) atb380955 records
all direct exits0,16384healthy frames,covered causal evidence and the17sample
trace. Main enqueue-return already had24974committed writes; its full API lead
cannot be subtracted as dead worker time. The postprocessing two-point/native
negative slope is not a physical scaling model. These learnings and conditional
550/530/520s cap tables select no production count and change no catalog/policy.

Earlier host-preflight stops were not hardware attempts. The successful run used
fresh baseline/boot/session and an immediate no-heavy-process check; no retry,
reset,C2 or P followed. Only the actual C1 consumed the one-run authorization.
No source/native/measurement-validator changes were made for execution.

At that stage C2 needed separate approval and compatible precommitted admission;
the C1-only bridge rejected C2/P. Legacy count proposals still require exact
provenance; do not relabel C1/B4 or silently bypass their identity checks.

## C2 bounded outcome and P proposal — 2026-11-21

The owner then approved exactly one unchanged catalog large calibration.
[C2](issue120-production/2026-11-21-c2/README.md) atf5d46fb completed all exits0,
32768healthy frames,covered,25host samples and zero overflow/drop/abort/paired
outliers/new warnings. Native/device/kernel/CTA/library/proof/image/work content
remained identical to C1. ADR-0020's explicit metadata-only compatibility and
C2-only v3 receipt bind both original C1 and new C2 records without relabeling.
Fresh baseline/session,host quietness twice and post-run800MHz/zero containers
passed; no retry/reset/P followed.

The artifact-compatible two-point/slab analysis is diagnostic,not production
admission. It proposes520s/22155167writes for owner consideration:conditional
frame margin19.291s and600s margins30.940s/14.067s with20s/historical36.873s
reserves.550s fails the slab/slow-rate model even atR20;530s has only3.508s at
the stress reserve. None is a measured long-run bound. The current550s floor
and legacy exact-provenance count path remain unchanged. A chosen520s target
requires a separate board-free successor production receipt/policy/test precommit
and matching native/device proofs,then owner approval for same-session500000-frame
control/load600s Watcher-off runs. The C2 report details predictions,stop/progression
rules,expected duration and unbounded thermal/AICLK/rate/postprocessing risks.
At that C2 stage admission rejected P; no production count was adopted.

## P preparation — board-free, not execution authority

The owner subsequently requested [separate production preparation](issue120-production-p-prep/README.md).
The explicit compatible-production receipt is version-dispatched,with a single
recommended target/count selection and unchanged legacy admissions. Actual
CPU-only500000-frame construction/hash/copy/analysis/publication/reread measured
13.231s loaded charged host cost,including the in-wrapper P1 validation. Native2GB
device retrieval remains unmeasured. Current conditional520s recommendation has
19.291s frame/9.465s pessimistic cap margin after measured-host reserve updates,
not a long-run bound or adopted count. Native/device content is unchanged.

Real CPU fixture E2E/negative gates,source-matched measurement,device-free startup/
publication and preservation evidence are indexed in that report. P1/P2 remain
unexecuted and require separate exact-head owner approval plus actual fresh
preflight;no real runtime P session was created here. The PR contains the
threshold-free predictions before any future run.

## Operational admission (board-free implementation)

The new runner defaults to a JSON proposal and never imports an accelerator
module. `uv run python -m enodia.tt.bench.issue120_production.runner --build-only`
only compiles/links and writes a source-bound build receipt. Actual launch needs
an explicit owner-approved flag, a fresh session, clean/pushed sources, fixed
wrapper image/device/caps, and the exact compiled/proved sources.

`--prepare-unit B` prepares only a new session manifest. Each run is launched
separately, not automatically continued. Non-Watcher calibration additionally
requires a committed `bounded_admission.json` binding the healthy B records and
operator Watcher inspection. Production additionally requires a committed
`production_admission.json` with both validated calibration record digests and
the recomputed fixed count. Neither file can be honestly produced in stage A;
the production entry point rejects their absence. The offline count proposal
is not authorization. New count data do not require changing the host algorithm
or widening the admitted wire domain.

The same immutable container wrapper supplies environment/run IDs and the power
sampler. The native and Docker measurement caps are60/600 seconds. A separate
launcher watchdog permits120 seconds of wrapper preflight/cleanup outside that
measurement cap, not a longer native run. Unknown runtime warnings stop success;
only the exact normalized #176 baseline is allowed. Fresh Watcher bytes are
separated from stale append prefixes, size/hash-bound on revalidation, and still
require operator inspection. Generator CLI output is fixed to its program-owned
build directory; library generation admits only build/device-check roots and
rejects redirected targets. Session arguments are reconstructed from canonical
128-bit UUID hex before subprocess use. Bounded admission requires successful warning,
fresh Watcher and execution predicates for both rows, not merely consistent
failure flags. Watcher diagnostic captures have a fixed64MiB limit.

Compact native sidecars are copied byte-exact into primary-stem companions before
record publication. Revalidation recomputes complete frame/map coverage and
resident metrics, checks the original atomic native envelope/direct status/raw
stream fingerprints and frozen power. This is scoped validation, not a claim of
completeness. Power freezes after a bounded wait for a terminal sample and uses
an explicit observed-host-envelope scope; it does not invent an exact card UTC
end or frame-window average. The result slots remain empty.
