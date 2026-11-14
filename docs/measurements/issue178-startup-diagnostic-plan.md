# Issue #178: source diagnosis and proposed bounded startup diagnostics

Status: **confirmed API-lock correction plus implemented bounded semaphore diagnostics;
no hardware authorization**. The shared-MeshDevice mutex explains the startup
response starvation. Producer start observation is an independent open item.
The earlier diagnosis/proposal is retained below with its conclusions superseded
where noted. No board/telemetry/reset run is permitted; only a strictly isolated
device-free startup check is used after host compile/link.

## Evidence and limits

The immutable P6 record is
[the run 5 attempt](2026-11-07-p150a-issue120-p6.json), measured at `2eefe9a`.
All 21 companion hashes/lengths were verified before this analysis. Native
stdout is 7071 bytes, stderr is empty, execution directly records code 0 and
no timeout. The atomic native envelope contains the original unvalidated
result; the wrapper records the stricter validator verdict. Twenty Watcher
dump headers show the producer/consumer kernels on logical `(0,0)`/`(1,0)`
(virtual `(1,2)`/`(2,2)`), but no mailbox values or startup-stage history.
Watcher `K`/`NARW` waypoints are not such a history.

Historical source references below are relative to harness `88f778e`, whose
startup code was unchanged from measured `2eefe9a`. The new correction and
diagnostics described at the end supersede that code, not the saved bytes.

### What `aborted=2` means

`q4_frame_probe.cpp:251` adds producer stats word 3 and consumer stats word 14.
It is a **sum of two binary flags**, not an enum with abort reason 2.

- `q4_frame_producer.cpp:28-40` sets its flag from 0 to 1 on the overall
  `13500000000`-tick budget path while waiting for start or a paced release.
  With attempted=produced=0, this can be the initial start wait or the first
  release wait before attempted is incremented. Those stages are not separated
  in the saved stats. It is not proof the start semaphore was never received.
- `q4_frame_consumer.cpp:112-120` sets its flag on the frame-ready wait budget
  path. With consumed=0, it timed out before the first ready frame. The done
  path alone would not set this abort flag.
- Since each flag is 0/1 and the aggregate is 2, both are 1. Their individual
  opened/exit timestamps and wait stages were not retained; no cross-core clock
  equivalence or exact cause is inferred.

Produced/consumed/attempted=0, empty frame rows, startup response 0, false gate
and tail sequence 12475/response 0 establish **no causal frame coverage**.
The validator's wording does not prove an actual producer start. In particular,
response fields are decoded **observed progress payloads**, not a record of all
262144 raw host mailbox reads. The last host-read value and count of zero reads
are unknown; it is not valid to claim that every host read returned zero.

## Initialization, publication and observation order

The ordering below is source order, not an invented hardware trace.

1. `q4_frame_probe.cpp:72-135`: partition resident/transfer sub-devices; allocate
   separate consumer L1 ring/mailbox, producer L1 anchor, DRAM progress/stats;
   create four zero semaphores and the two kernels. Mailbox address is runtime
   arg 11 for the consumer, and the same buffer address is passed to host reads.
2. Lines 223-228: blocking CQ1 progress zero-write and blocking CQ0 mailbox
   zero-write precede workload submission. These are initialization commands,
   not evidence of later device-store visibility.
3. Lines 232-237: host load `jthread` starts **before** `EnqueueMeshWorkload`.
   Enqueue can compile the program. Thus a host read/CPU work/response 0 before
   card publication is a legal schedule; initial zero alone is not a defect.
4. `traffic` at lines 149-187 reads a four-byte mailbox through `read_challenge`
   before each fixed CPU repetition and immutable CQ1 payload copy. It does not
   wait for a nonzero challenge or wait for each payload's completion. The
   source translates the consumer logical core then calls targeted cluster
   `read_core`, without a CQ finish in that read path.
5. `q4_frame_consumer.cpp:89-105`: initialize the consumer clock, store token 1
   in the local mailbox, fence, observe progress, then seek a valid advancing
   sequence responding to 1 under the existing startup deadline.
6. Lines 106-111 issue token 2, then send the producer start semaphore and wait
   for atomics completion. **This start send is unconditional even if `gated`
   is false.** That is a confirmed control-flow fact, not a demonstrated cause
   of the zero response or both aborts. This work does not change that flow.
7. Producer waits for start, then paces its first frame. Consumer waits for
   ready. Both have the unchanged overall budget. On exit the producer writes
   stats and sends done; the consumer observes tail and writes its final stats.
8. The host finishes CQ0, joins the fixed-count host traffic thread, drains CQ1
   and downloads stats. Later host work/completion does not establish progress
   inside a frame interval that never existed.

Saved artifacts do **not** contain resolved mailbox runtime addresses, local
readback after stores, all raw host reads or start-semaphore values/stages.
The earlier unresolved-root conclusion is superseded: the shared API mutex
blocks CQ1 writes during CQ0 finish, explaining delayed response delivery and
startup gate failure. Missing semaphore evidence still prevents identifying the
independent producer-start observation defect.

## CPU-only characterization, not a device reproduction

`tests/test_issue178_startup_replay.py` compiles the **actual existing `traffic`
function body** with in-memory mailbox/queue seams and the actual wire encoder.
It does not link Metalium or load device code. A deterministic delayed publisher
produces responses `[0,1,2]`; publication before load produces `[1,1,1]`.
Never-published and wrong-target models both produce `[0,0,0]` despite different
published state. Invalid values still reject. These cases prove that the host
encoding preserves a supplied challenge and that zero responses alone do not
identify a cause. They do not simulate firmware, NoC ordering, core mapping or
hardware cache visibility, nor reproduce the full device failure.

These earlier tests were baseline characterization, not a pre-fix RED claim.
The subsequent executable scheduling-seam and host-scope regressions did fail
before implementation, then passed with the API-lock correction. They do not
claim to reproduce physical semaphore visibility.

## Frozen invariants for a future diagnostic implementation

- Keep validator, frame/transfer counts, CPU arithmetic, budgets, core/sub-device
  setup, host-thread/enqueue ordering, and cap semantics unchanged. In particular
  do not poll until nonzero, add warmup transfers, release producers differently,
  or increase work to make a trace pass.
- Do not promote diagnostic stages, host times, submitted counts or clean Watcher
  state to coverage. Existing fail-closed validation remains authoritative.
- Retain original raw streams/result/execution and historical record bytes.
- Use bounded diagnostic storage independent of repeat count: fixed counters,
  first/last snapshots and at most 16 transition records, with explicit dropped
  count/overflow. Do not print per iteration or add host MMIO reads to collect
  the mailbox history: sample the existing reads.
- Distinguish intended stores, device-local readbacks, host-read values, payload
  enqueue/return and card-observed response values. Do not treat them as synonyms.
- Host monotonic and each core's wrap-tracked clocks remain separate domains.

## Earlier broad trace proposal (retained history)

The following broader stage-history proposal was not implemented in the earlier
commit. The fixed post-finish snapshots actually implemented now are specified
in the correction section below; no per-iteration history is added.

| Location | Fixed snapshots/stages to retain |
|---|---|
| Host setup | zero-write completion, buffer/mailbox card address, logical/translated target, actual semaphore IDs returned by creation; workload enqueue entry/return and first traffic-read stage |
| Host existing reads | read count, zero/nonzero/invalid counts, first/last value, first generation reading 1/2, bounded value transitions with generation and host monotonic time |
| Host response path | first generation/value before and after enqueue; first successful enqueue for 1/2, last returned generation, bounded call-stage samples distinguishing blocked read versus blocked enqueue; no payload/completion policy change |
| Consumer | kernel entry; mailbox address; after token 1 store/fence and local readback; startup left/right sequence/challenge/valid/deadline exit; gated flag; token 2 store/readback; before/after start increment and atomics completion; first-ready or budget-abort; tail/stats publication |
| Producer | entry/opened clock; local start semaphore address/initial value; start wait exit/current value; epoch/first-release entry; distinct start/release budget-abort stage; first attempted/produced frame; done/stats publication |
| Watcher | bounded startup waypoints corresponding to those stages, so an incomplete result is not replaced with an invented last stage |

Use currently unused words in the existing fixed stats pages where possible;
explicitly account for any proposed host trace storage. Preserve the existing
aggregate `aborted` and add split flags/stage labels, not replace its semantics.
A trace overflow/incomplete page is marked incomplete, never silently complete.
The actual implementation needs its own invariants/RED tests and review before
an owner-approved diagnostic run; the CPU characterization is not its acceptance.

## Earlier proposal identities (superseded, never executed)

| Attempt | Output | Cap | Watcher | Prediction and prerequisite |
|---|---|---|---|---|
| `p6-startup-trace-v1` | `out/issue120/p6-startup-trace-v1` | native/outer **60 s** | **on** | Same fixed 4096 frames, 262144 transfers, 32x1024 CPU operations per transfer. A negative startup result may recur; bounded stages/read/response history should discriminate paths without weakening validation. Diagnostic implementation, committed revision/fingerprint and renewed owner approval are mandatory first. |
| `p7-startup-trace-v1` | `out/issue120/p7-startup-trace-v1` | native/outer **60 s** | off | Same fixed work; only after a healthy P6 startup and full-frame causal coverage with clean Watcher. Otherwise do not run. Approval must cover the implemented head, not this proposal alone. |

These are precommitted **proposal identities**, not claimed ready-to-run probes.
No directories or native attempts are created. No extra invocation/reset is
automatic. A future implemented revision/native fingerprint must be recorded
and separately approved. This document does not authorize P6/P7 re-execution.

## Confirmed defect 1 and source/evidence correction

The supervising operator identified the device-wide API-lock chain. In v0.75.0
(paths below relative to `tt_metal/`):

- `distributed/fd_mesh_command_queue.cpp:731-734` holds `lock_api_function_()`
  while `finish_nolock()` waits on completion/outstanding reads (`:691-710`).
- `distributed/mesh_command_queue_base.cpp:175-209` takes the same lock for
  buffer writes, and `enqueue_mesh_workload` takes it at
  `distributed/fd_mesh_command_queue.cpp:385`.
- `distributed/mesh_device_impl.hpp:128-130,192` defines one `api_mutex_` per
  MeshDevice, not per CQ.

P6 finished CQ0 while the load thread still enqueued CQ1, preventing subsequent
responses from reaching DRAM until finish returned. Identical valid tail
sequence 12475, host tail 8.205153151 seconds beyond resident finish, and final
drain 104528 ns agree. Remaining 249669 repetitions at the post-release reference
rate about 33 microseconds/repetition explain the tail. This is the confirmed
startup gate starvation cause, not a conclusion drawn merely from response0.
P1-P3 enqueue their writes first in one thread and avoid this conflict.

**Production invariant:** on one MeshDevice, do not call CQ0 finish/Synchronize
while another thread is still enqueueing CQ1: the shared API mutex serializes
those operations. Releasing the Python GIL or selecting another CQ is not enough.

The implemented callable sequence is enqueue -> load join -> CQ0 finish -> CQ1
finish. Thread start remains before enqueue. Tests execute fake callbacks and a
shared-mutex writer, not grep a cpp file. Counts, CPU work, budgets, start-send
control flow and coverage predicates are unchanged.

`resident_host_interval.scope` is now
`enqueue_to_post_join_finish_return_upper_bound`; its endpoint is the post-join
CQ0 finish return, not a card/resident completion timestamp. Join time is
separate in `host_scheduling`. Keeping the finish-return endpoint also remains
an upper bound if host work finishes before the resident. New metadata claiming
an exact completion endpoint is rejected; coverage still uses card evidence
only. Historical records retain their original bytes/interpretation.

## Independent defect 2: implemented fixed diagnostics, cause still open

The shared lock does not explain the producer's missing attempted frames.
Consumer sends start unconditionally after startup; Watcher NSID indicates the
inc-complete path while producer remains K. L1 data cache defaults disabled on
BH (`llrt/hal/tt-1xx/hal_1xx_common.cpp:14-18`,
`hw/inc/internal/firmware_common.h:297-306`); the saved env has no enabling
variable. Shift36 agrees with `NOC_XY_ADDR`/coordinate mask and no sanitizer
error is present. Cache/mask are not assigned as causes.

Both defects are addressed in one correction precommit: scheduling repair plus
these optional `startup_diagnostics` fields, without weakening the validator:

- Consumer stats retain the exact emitted start target (two words), inc tick,
  local semaphore base and final mailbox address/readback. Start send stays
  unconditional, including on a failed gate.
- Producer stats retain exit raw start, polled address/base, opened and exit
  wrap-tracked ticks, while preserving aggregate/split abort values.
- After join and both finishes, host reads each core's four semaphore words via
  targeted `Cluster::read_core`, then the consumer mailbox, and retrieves the
  final progress page. At most 10 four-byte MMIO reads (normally 9), plus one 4096-
  byte progress retrieval: fixed size, no per-iteration log or extra load read.
- Actual CreateSemaphore IDs and kernel-assumed 0-3 are recorded. Expected target
  uses translated producer coordinates, resolved semaphore base and 2*L1
  alignment; CPU-only tests compare with pinned BH `NOC_XY_ADDR` and masked bank
  coordinates. Emitted and producer-polled addresses are compared separately.
- v0.75 has no static HAL SEMAPHORE enum. Firmware uses launch kernel-config
  base plus sem_offset. The host resolves the workload's actual CQ slot using
  `MeshWorkloadImpl::get_sem_base_addr` and separately records the HAL
  KERNEL_CONFIG reference, preventing a false static-base mismatch diagnosis.
- A post-finish start 1 means present at observation time; compare exit raw/ticks
  to distinguish unobserved/late arrival, not automatically a cache defect.
  Start 0 prompts the emitted/expected/polled target comparison. Diagnostics
  cannot promote coverage and load MMIO/work accounting remains unchanged.

## New implemented precommit identities (not authorized to run)

Pinned host compile/link succeeds with native SHA-256
`600d02aca3f1fbd50454357392810007019d189d8a64dcb2f47af0b0c0b252b8`.
CMake requires the pinned workload-implementation header and impl include root;
no substitute library/toolchain is used. The measured old binary remains named
in its historical records, not replaced there. A device-free, network-disabled,
non-root/read-only/capability-dropped sandbox with hardware discovery masked
passed asset preflight and exited directly with code 2 at `No chips detected`.
Raw stdout 768 bytes/stderr 2259 bytes and metadata hashes survived; an absent
result was refused and no container remained. This checks loader startup only,
not device kernel JIT, semaphore behavior or frame coverage.

| Attempt | Output | Native / outer cap | Watcher |
|---|---|---|---|
| `p6-api-lock-semaphore-v1` | `out/issue120/p6-api-lock-semaphore-v1` | 60 s / 60 s | on |
| `p7-api-lock-semaphore-v1` | `out/issue120/p7-api-lock-semaphore-v1` | 60 s / 60 s | off |

Same 4096 frames, 262144 transfers, 32x1024 CPU operations/transfer. At the observed
post-lock reference rate about 33 microseconds/repetition, fixed host work is
predicted to span about 8.7 seconds, versus dispatch 0.4 + startup 0.1 + 4096x1 ms
(about 4.6 seconds at the reference frequency). This is a pre-run estimate, not
an independent real-time clock guarantee. If fixed host work genuinely ends
before the last frame, report actual not_covered; do not increase counts or
mislabel that outcome a scheduling bug.

Three prediction branches:

1. Gate passes and frames flow: require every existing causal coverage predicate,
   then healthy Watcher before P7's identical-work no-Watcher comparison.
2. Gate passes but producer aborts with post-finish start 1: lock correction can
   work while independent observation remains broken/late. Stop, retain raw
   exit/start/address/tick snapshots and do not run P7.
3. Post-finish start 0: compare emitted/expected/polled target and actual IDs to
   discriminate destination/delivery; stop without P7 or speculative bypass.

A failed gate or any not-evaluable/abnormal/new-warning state also stops. No
new identity directory or hardware run is created here. Native fingerprint and
committed implemented revision must be published before renewed owner approval.

## #118 source audit

`load_generators.py::run_host_cpu_transfer_work` itself is synchronous and does
not create threads or synchronize; it calls an injected transfer seam. The real
runner is separate: `run_resident.py:976-1043,1058-1071,1250-1252` starts its host
thread, submits resident work, synchronizes CQ0, and only later joins the host.
The same ordering is present in harness revision 2e143f8, lines 819-880 (used for accepted non-host measurements). Its host
thread enqueues CQ1 batches and synchronizes CQ1 per repeat. Nanobind at
v0.75 `ttnn/cpp/ttnn-nanobind/device.cpp:590-600` releases the GIL but calls
`distributed::Synchronize`; `tt_metal/distributed/distributed.cpp:161-174`
reaches the same queue finish/device API mutex. Thus the concurrent runner
pattern exists and can serialize. It is a source-backed candidate for historical
roughly 49-second waits, not a retroactive timing proof of each old calibration.
No #118 runner change is made in this PR step.

Production remains blocked on resolving/validating the independent semaphore
item, healthy bounded real-frame Watcher/no-Watcher causal coverage, fixed-work
calibration if needed, non-aliasing long-run protocol, production control/load
integration and 500000-frame/600-second/same-session-control precommit/approval.

## V2: explicit runtime peer coordinates and one-shot provenance diagnostics

The operator conveyed owner approval for this bounded correction/precommit,
then `p6-api-lock-semaphore-v2` and conditional `p7-api-lock-semaphore-v2`.
This does not authorize production, retries, changed work or bypassed gates.
Run6 at `af5db15` was branch3: emitted `(1,5)` versus intended producer `(1,2)`,
raw start0 and frames0. The operator identified virtual `(1,5)` as logical
`(0,3)`, an inactive core. Both directions had used sharded accessor coordinates;
return ready/done and frame data needed the same correction, not only start.

### Frozen invariants, RED and implementation

1. Consumer start/free and producer ready/done/ring writes use only peer virtual
   coordinates supplied by host runtime args and `get_noc_addr(x,y,local)`.
   Accessor-derived coordinates are diagnostic-only; no masked bank is a route.
2. Shared `q4_runtime_args.hpp`: producer peer x/y at7/8 after its old0-6;
   consumer peer x/y at12/13 after old0-11, mailbox stays11. No old index moves.
3. Both setup and post-finish diagnostics call the same `peer_coordinates` helper
   with the same IDevice, invoking `virtual_core_from_logical_core(...,WORKER)`.
   Record passed and independently recalculated coordinates, plus device echoes.
4. Diagnostic fields occupy fixed unused words of each final stats page, once
   only. No per-frame log, progress/CPU loop growth, or coverage inference.
5. Preserve counts, CPU work, ring size, budgets, unconditional start, caps,
   join-before-any-finish and strict card validator. New fields remain optional;
   all previous evidence bytes stay unchanged.

RED: `uv run pytest -q tests/test_issue178_runtime_coordinates.py` failed before
implementation (missing runtime-coordinate seam/shared constants;1 failed).
Afterward the actual host seam executes under CPU-only fake translation,
asserting both peer coordinates at shared indices, unchanged prefixes/mailbox,
returned-versus-diagnostic coordinates and refusal to append twice. It also
executes A/B/C classification. This is not a cpp grep or a device reproduction.

`BankCoordsCTAOffset` is public static constexpr in v0.75's
`hw/inc/api/tensor/tensor_accessor_args.h:102`, before its private section.
It is also present in the header extracted from the exact pinned image using a
never-started container. The six relevant staged tensor headers/host packer
match tag `d9a68815f5fcf08a5bfbffb6f1f811823fba8edd`. Runtime host-library
packing remains something to observe, not source/image equivalence assumed.

### Fixed diagnostic layout and comparisons

Consumer final stats append: actual anchor BankCoordsCTAOffset, raw CTA word
at that slot, full unmasked `anchor.get_noc_addr(0)` (lo/hi), `my_x[0]/my_y[0]`,
runtime producer x/y and actual start target. Producer appends the same ring
fields, runtime consumer x/y and ready target. Ready target has a separate
used flag; if no frame was produced it was computed but not sent. Original
start raw/address/ticks, sem IDs/base, both four-word readouts, mailbox and
final progress reads remain. No additional per-frame diagnostic work is added.

Host retains the per-accessor actual CTA vectors supplied to CreateKernel,
their global bank-slot positions, expected `(x&255)<<8 | (y&255)` from virtual
coordinates, and compares kernel raw word to both expected and actual host
words. It also compares accessor xy, self xy, runtime echoes, passed coordinates,
post-finish recalculation and actual target xy. These are nested optional
coordinate diagnostics under the existing startup diagnostic object, never
coverage input. Producer's new frame data path uses consumer runtime xy and
`ring_address + page_index*4096` directly; no accessor supplies that destination.

### Coordinate prediction A/B/C (independent of run branch)

Evaluate separately for consumer anchor/producer peer `(1,2)` (expected word
`0x102`) and producer ring/consumer peer `(2,2)` (expected word `0x202`):

- **A**: kernel raw word equals expected host virtual-coordinate pack and
  accessor coordinate matches the intended peer. Run6's mismatching accessor
  observation is not reproduced; record that instead of forcing an old cause.
- **B**: raw word matches but accessor coordinate differs: device-side
  decode/address construction becomes the focus.
- **C**: raw word differs from the expected host pack: investigate host packing/
  handoff. Actual host CTA vectors and kernel slot/word additionally distinguish
  a host-generated wrong word from a discrepancy after a correct host vector;
  do not infer that sub-cause from the C label alone.

In every case actual cross-core routes use runtime xy, not the accessor result.
If runtime echoes/targets unexpectedly differ, stop rather than trust diagnostics
or invent coverage. Old run5/run6 native and inspector bytes were checked: they
retain no raw CTA word or actual host CTA vectors. Run6's accessor-derived
`(1,5)` target is a similar symptom, but cannot retrospectively select B or C;
run5 has no target readout at all. Their bytes and conclusions are not rewritten.

### Run identities, work, caps and branch gates

| Attempt | Output | Native / outer cap | Watcher |
|---|---|---|---|
| `p6-api-lock-semaphore-v2` | `out/issue120/p6-api-lock-semaphore-v2` | 60 s / 60 s | on |
| `p7-api-lock-semaphore-v2` | `out/issue120/p7-api-lock-semaphore-v2` | 60 s / 60 s | off |

Same4096 frames,1350000 ticks/period,262144 transfers,32x1024 CPU operations per
transfer (8589934592 total),1GiB H2D/1MiB load MMIO and fixed bounded buffers.
The runtime-route correction is predicted to enable real frames; sustained
causal coverage is still a measurement question. Run6 fixed work took6.110s
versus the earlier8.7s estimate; neither is a stopping/adaptation target. If
actual host work ends before the last frame, record genuine not_covered and do
not increase counts/caps. Sparse power samples are not an independent clock.

Run branch definitions follow this V2 brief:

1. Producer observes start, the full intact fixed frame stream flows and the
   unchanged coverage evaluator can evaluate it. Record covered/not_covered
   honestly; neither labels diagnostics as coverage. This branch, healthy
   Watcher and no unexpected warning/error permit identical-work P7 without
   Watcher. Invalid frame identity/checksum/work or ambiguous clock is not a
   healthy branch1 prerequisite.
2. Start was observed (producer exit start>=1 or attempted>=1) but frame stream
   is incomplete. Retain all diagnostics and stop before P7.
3. Start stays0 and no attempt. Compare runtime/actual targets, IDs/base and
   coordinate diagnostics; stop before P7 without a retry.

Failed gate/not_evaluable, unexpected errors/warnings or unplanned diagnostics
also stop. Only #176's three accepted baseline conditions are allowed. Require
clean pushed precommit head, published new native SHA-256, full tests/ruff,
pinned compile/link, isolated device-free startup and matching-head CI/Sonar
before board preflight. Then freshly verify zero containers, both cards800MHz,
serial411c/device0 mapping, environment and no heavy host work. Named containers,
60-second outer caps and post-run zero-container/800MHz checks are unchanged.
Reset only on timeout/abnormal exit, at most once across V2; no automatic retry.
Production remains forbidden. Native fingerprint and verification are recorded
in the V2 precommit publication before either attempt.

V2 pinned native host compile/link passed. Native SHA-256:
`7b9baa12d3adfd6d4f54496f5d0e50463a16af81285827849e74c6cad619fb5a`.
This fingerprint replaces neither V1's measured binary identity nor its records.
Device kernel JIT/behavior was still unverified before the approved P6 V2.

### V2 execution outcome: pre-branch JIT failure, no retry

The [authorized run7](2026-11-07-p150a-issue120-run7.md) at `49743ef` stopped
before resident dispatch: producer compiler CTA was `[0,4096,2,4096]`, while the
new diagnostic unconditionally requested BankCoordsCTAOffset4. Static assertion
`4 < 4` failed. The checked source interprets that layout as unsharded L1 and
DRAM config/page pairs, not the assumed sharded bank-coordinate input. Public
availability of the constexpr offset was insufficient to guarantee a valid slot.
No final stats/native result exists, so run branches1/2/3 and coordinate A/B/C
are unavailable, not inferred zeros/C. Actual host vector copies were not
persisted before the failure. Follow-up must establish actual buffer/sharding/
CTA generation and safe diagnostic applicability/retention tests board-free.
Runtime routing remains hardware-unvalidated. New JIT error triggered stop;
P7/retry/production were not run, no timeout/container survivor, both cards
800MHz/DRAM healthy, reset0. New correction/identity requires renewed approval.

## V3: explicit distribution, guarded snapshots and board-free device compile

After reboot, the operator conveyed owner approval for this board-free change,
precommit and `p6-api-lock-semaphore-v3` plus branch1-only P7 V3. No production,
automatic retry or relaxed validator is authorized. V2 records remain immutable.

### Source conclusion and allocation choice

The operator established the source chain (v0.75.0, paths relative to tt_metal/):
`api/tt-metalium/buffer.hpp:150-153` legacy ShardSpecBuffer constructor populates
only shard_spec; `impl/buffers/buffer.cpp:328-329` copies both optionals unchanged;
`impl/buffers/tensor_accessor_args.cpp:146-157` sets Sharded only if distribution
is present. This explains source-side why legacy L1 allocations produce
interleaved accessor config0 and resolve bank0, consistent with run6 `(1,5)`.
It is a source inference to be checked against V3 values, not fabricated V2 stats.

**Choice:** retain ring/anchor/mailbox allocations but construct them with
`BufferDistributionSpec(Shape{pages,1},Shape{pages,1},single_core,ROW_MAJOR)` and
`BufferShardingArgs(distribution)`. Ring data writes already use runtime xy plus
local ring_address/page offset, so no accessor participates in that route.
Anchor stays diagnostic-only: retaining its original one-page allocation allows
checking the old coordinate explanation without altering pool size/bytes.

The address is valid on the designated core, not a claimed exclusive per-core
allocation: `impl/allocator/allocator.cpp:128-186` uses the normal lockstep L1
manager (per_core_allocation remains false). `bank_manager.cpp:410-447` reserves
a per-bank extent based on num_shards; the common allocator address is reserved
across banks, including the selected worker. Distribution cores select the
single logical worker and its page mapping, not an interleaved bank0 route.
`buffer_distribution_spec.cpp:114-149,330-380` squeezes matching shape dimensions
to rank1. Host checks actual distribution/core count/selected core/per-core pages
and aligned per-bank extent before use. Four local4096-byte ring pages then fit
inside the reserved extent on consumer core; anchor/mailbox each have one page.
This does not assert per-core-only reservation or reduced allocator overhead.

### Invariants -> RED -> implementation

1. Retain V2 runtime peer xy and fixed shared indices for every cross-core
   start/free/ready/done/ring-write destination. Accessor xy never routes data.
2. All L1 allocations carry explicit distribution. Before CreateKernel, host
   admits each ring/anchor accessor only if config is static Sharded L1, one
   bank, shapes match fixed pages, and bank word matches virtual xy pack.
   Actual complete producer/consumer CTA arrays must also equal the offline-
   checked manifest; any mismatch throws explicitly before dispatch, not a retry.
3. Templated `if constexpr(Args::is_sharded)` guards the bank read. Config is
   always recorded. For non-sharded input the slot has an invalid sentinel and
   JSON bank slot/word are null with bank-present=false; no out-of-range read.
4. Final stats snapshot only, with config/Sharded/presence and prior raw address/
   runtime/target fields. Diagnostics remain optional and never coverage input.
5. Counts, CPU work, buffers, ring extent, budgets/caps, unconditional start,
   join-before-finish, strict validator and prior record bytes stay unchanged.

RED: `uv run pytest -q tests/test_issue178_distribution.py` failed2 before
implementation (missing admission helper/guard header). CPU tests subsequently
execute rank1/rank2 admission, bad config/bank/shape/truncation/CRTA rejection,
and prove the non-sharded template never instantiates missing CTA index4.

An additional **real pinned RISC-V RED** ran the old producer with actual run7
CTA `[0,4096,2,4096]` in a strictly device-free container, reproducing
`Index out of range; 4 < 4` before kernel changes. This exposes the prior host-
only validation gap without opening a device.

### Mandatory device-free syntax/constexpr gate

`tools/check_issue178_device_kernels.py` runs inside the exact pinned image,
using its SFPI RISC-V compiler and runtime headers, no TT Python/native runtime.
It passes explicit `-DKERNEL_COMPILE_TIME_ARGS=` arrays from shared
`q4_compile_args.json`, BRISC/NOC0/Blackhole flags matching the saved invocation,
and `-fsyntax-only` over each actual kernel plus firmware_common.h.

Eight cases passed: both kernels, V3 sharded and legacy non-sharded vectors,
Watcher off/on. [Proof with hashes](issue178-v3-device-compile-check.json) binds
compiler, manifest and kernel/header source bytes; raw output stays in the
private work directory. Container has no devices/network, masked sys/proc
hardware discovery, non-root/read-only source/root and all capabilities dropped.

V3 exact producer args are `[1,4096,1,1,4,4,514,2,4096]`; consumer concatenates
ring `[1,4096,1,1,4,4,514]`, anchor `[1,4096,1,1,1,1,258]`, then three DRAM
`[2,4096]` pairs. Host rejects any complete array differing from this manifest,
so submitted arrays cannot silently differ from the ones syntax-checked.
This gate catches constexpr/static-assert errors, **not device linking,
allocation execution, firmware dispatch or physical NoC behavior**; the bounded
Watcher run remains mandatory. Precommit also requires full tests/ruff, pinned
host compile/link, isolated no-chip startup and matching-head CI/public Sonar
via curl (no acceptance).

### V3 identities, predictions and stop rules

| Attempt | Output | Native / outer cap | Watcher |
|---|---|---|---|
| `p6-api-lock-semaphore-v3` | `out/issue120/p6-api-lock-semaphore-v3` | 60 s / 60 s | on |
| `p7-api-lock-semaphore-v3` | `out/issue120/p7-api-lock-semaphore-v3` | 60 s / 60 s | off |

Run branches are unchanged from the V2 brief:1=start observed plus full intact
frame stream with evaluable covered/not_covered (only then P7),2=start observed
but incomplete stream,3=start remains0/no attempts. Unexpected error, warning
outside the three accepted #176 conditions, timeout or failed gate stops;
no diagnostic may manufacture coverage. Fixed4096 frames/262144 writes/32x1024
CPU work,1GiB H2D/1MiB load MMIO remain; no count/cap adaptation for a genuine
early host finish. Production remains forbidden.

Coordinate prediction, separately from the run branch: kernel config1/Sharded,
bank present and packed word258 `(1,2)` for consumer anchor,514 `(2,2)` for
producer ring; each agrees with host CTA/virtual xy and runtime peer echo.
Expected global bank slots are13 and6 respectively. Record every deviation
separately, including absent bank word; actual routes remain runtime-based and
a diagnostic mismatch alone does not turn evaluated coverage into a pass/fail.
These values would support the non-sharded source explanation and repaired
accessor representation, not retroactively create missing run5/run6/V2 words.

Reboot invalidates old snapshots/private temporary evidence. Before P6, freshly
capture both cards' detection/serial/PCI/FW/KMD/800MHz, pinned image digest,
environment and no-heavy-process/container0 preflight. Compare stable values
with archived run7; stop/report any changed value instead of assuming parity.
Use only device0 serial411c, named containers, saved before/after ps, idle800MHz
postchecks; no heavy host tasks during runs. Reset only for timeout/abnormal
exit, at most once across V3, no automatic retry. P7 requires branch1/normal
Watcher/errors checks; no production. New native fingerprint and final
verification will be published before either physical invocation.

V3 pinned native host compile/link passed; native SHA-256:
`c35af4b4506dfd2534fa71debb6bdd0206825bd428fa102a04124a00cf3a9e43`.
The offline device proof binds the unchanged kernel/header/manifest hashes.
Full board-free suite:6120 passed/1057 skipped/8 subtests, ruff/diff/ADR guard
passed. Strict masked no-device/no-network startup passed asset preflight and
stopped at No chips detected, direct native2, byte-exact stdout768/stderr2259
retained, absent result refused, sandbox removed. These are pre-board checks,
not evidence of physical allocation/NoC/coverage success.
