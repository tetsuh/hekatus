# Issue #178: source diagnosis and proposed bounded startup diagnostics

Status: **diagnostic proposal, not an implemented fix or hardware authorization**.
The causal defect cannot yet be identified from the retained bytes. No native,
device, container-start, telemetry or reset command is part of this work.

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

Source references below are relative to harness `88f778e`; its production
startup code is unchanged from the measured `2eefe9a`.

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
readback after stores, all raw host reads, first matching-response enqueue
completion, or start-semaphore values/stages. They therefore cannot distinguish
publication/address/visibility trouble from delayed queued responses or other
startup/order failures. No one of these is selected as root cause.

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

No production fix was made, so there is no pre-fix RED claim. These are baseline
characterization tests, not tests passed off as a failing device regression.

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

## Proposed outputs and stage discrimination

No diagnostic fields listed here are implemented by this proposal.

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

## Proposed identities and predictions (not executable yet)

| Attempt | Output | Cap | Watcher | Prediction and prerequisite |
|---|---|---|---|---|
| `p6-startup-trace-v1` | `out/issue120/p6-startup-trace-v1` | native/outer **60 s** | **on** | Same fixed 4096 frames, 262144 transfers, 32x1024 CPU operations per transfer. A negative startup result may recur; bounded stages/read/response history should discriminate paths without weakening validation. Diagnostic implementation, committed revision/fingerprint and renewed owner approval are mandatory first. |
| `p7-startup-trace-v1` | `out/issue120/p7-startup-trace-v1` | native/outer **60 s** | off | Same fixed work; only after a healthy P6 startup and full-frame causal coverage with clean Watcher. Otherwise do not run. Approval must cover the implemented head, not this proposal alone. |

These are precommitted **proposal identities**, not claimed ready-to-run probes.
No directories or native attempts are created. No extra invocation/reset is
automatic. A future implemented revision/native fingerprint must be recorded
and separately approved. This document does not authorize P6/P7 re-execution.

Production remains blocked on causal root diagnosis/correction, healthy bounded
real-frame Watcher and no-Watcher coverage, bounded work calibration if needed,
non-aliasing long-run protocol, production control/load integration and the
500000-frame/600-second/same-session-control precommit and approval gates.
