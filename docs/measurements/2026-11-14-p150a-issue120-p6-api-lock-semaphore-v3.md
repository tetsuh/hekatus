# Issue #120 P6 V3: branch1 and covered with Watcher

[Primary/23 hashed companions](2026-11-14-p150a-issue120-p6-api-lock-semaphore-v3.json)
/ [run8 interpretation](2026-11-14-p150a-issue120-run8.md).

Clean/pushed harness `e1b37547f0eb7909ae23fdfbb4c176ac7c047cbc`, native SHA-256
`c35af4b4506dfd2534fa71debb6bdd0206825bd428fa102a04124a00cf3a9e43`,
original fixed work, 60-second native/outer caps, Watcher on, only device0
serial411c/PCI0000:06:00.0. All post-reboot stable environment values matched
archived run7; both cards idle/healthy, containers0 and no heavy host work before
launch. Precommit passed full6120 tests/CI/public Sonar0/0/OK, host link/no-chip
startup and8 pinned device-free RISC-V syntax/constexpr cases. The latter were
not dispatch/allocation guarantees; this run supplies bounded physical evidence.

## Result

**Run branch1, coverage=covered.** All4096 attempted/produced/consumed frames,
zero abort/overflow, intact IDs/checksums/work, unambiguous consumer clock,
startup response1/gate=true. The unchanged evaluator verified4096 real-frame
intervals plus first/last causal CPU-response/progress endpoints. It reports
16781312 completed bytes as a conservative lower bound,16773120 during
inter-frame intervals. Never multiply unseen generation deltas into bytes or
claim every submitted1GiB completed within the frame window.

Fixed host work:8589934592 operations,262144 writes/1073741824 bytes and262144
load reads/1048576 bytes. Host work lasted6.040565943 seconds. This is descriptive,
not an elapsed-time stopping loop or independent card-clock projection. The
post-join host endpoint stays an upper-bound finish-return envelope; card proof,
not that envelope or diagnostics, establishes coverage.

## Coordinate/semaphore snapshot

| Field | Consumer -> producer | Producer -> consumer |
|---|---|---|
| Config / Sharded / bank present | 1 / true / true | 1 / true / true |
| Kernel bank word / host expected and actual | 258 / 258 / 258 | 514 / 514 / 514 |
| Global bank slot kernel/host | 13 / 13 | 6 / 6 |
| Peer runtime / accessor / expected virtual | (1,2) / (1,2) / (1,2) | (2,2) / (2,2) / (2,2) |
| Self NoC0 | (2,2) | (1,2) |
| Unmasked accessor address | 0x8100017b000 | 0x8200017c000 |
| Actual start/ready target | 0x81000009f60 | 0x82000009f40 |

All recorded config/word/slot/self/runtime/accessor/target equality flags are
true. Host CTA matched the offline-compiled manifest. Explicit distribution
core/page/extent checks passed. Ring/anchor/mailbox local addresses are
1556480/1552384/1548288; actual routes use runtime coordinates, not accessor xy.

Sem IDs0-3, relocated base0x9f40, alignment16; producer start poll0x9f60.
Producer post-finish words `[0,4096,1,0]`, consumer `[4096,0,0,1]`. Producer
exit/post-finish start1/1, mailbox exit/post-finish4098/4098; final DRAM progress
valid sequence262144/challenge4098. Diagnostic reads remain9/36 bytes plus one
4096-byte progress page, separate from load work and never coverage inputs.

The raw legacy field `mailbox_expected=2` describes the old zero-frame startup
state, not this complete stream: the correct final token is4096+2=4098. Preserve
the raw field and distinguish its phase in interpretation; it is not a validator
input. [Issue #179](https://github.com/tetsuh/hekatus/issues/179) tracks that
non-blocking metadata-scope correction, not a live change/retry here.

## Warnings, health and interpretation

Native/container/wrapper exit0, no timeout. Only seven accepted #176 warning
lines. Fresh Watcher8 dumps, no anomaly. Zero containers after run; both cards
800MHz/DRAM healthy; reset0/retry0. P6 branch1, complete strict card coverage,
normal Watcher and health permitted the sole authorized same-work P7 without
Watcher. No production followed.

Run7's compiler config0 lacked bank coordinates; explicit distribution now
produces measured config1 and correct bank words/accessor xy. This supports the
source explanation and validates the combined runtime-routing/distribution
repair. It does not invent absent run6 bank words or isolate the two changes
by a separate ablation. Frozen5 power samples do not establish production wall-
time/endpoint coverage. Timing evidence remains false; this is bounded causal
feasibility, not the500000-frame host-load timing measurement.
