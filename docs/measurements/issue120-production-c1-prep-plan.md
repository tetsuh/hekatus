# Issue #120 C1 host-trace preparation — invariants before RED

## Scope and authority

C1 approval remains valid, but the owner is powering off p150a: **no board,
tt-smi, device, reset or hardware run in this task**. Finish board-free work and
stop. Device-free startup/publication checks must have no device mappings or
requests, network none, masked hardware discovery and read-only source/root.
Do not prepare a runtime session bound to this pre-reboot boot. C2/P are excluded.

Next-boot proposed identity stays `production-v1-calibration-small`:16384 frames,
1048576 writes,32768 CPU operations and4096 bytes/write,60s native/outer cap,
Watcher off, device0 serial411c/PCI0000:06:00.0. Do not change catalog counts,
frame period, queues/subdevices, fixed-work payload/challenges or dispatch.

## Host-only trace invariants

1. Preserve every device-facing kernel/support byte, CTA, runtime-argument
   construction, buffer layout, queue/open configuration, write count/size/order,
   CPU arithmetic and challenge/transition algorithm. Do not add device reads,
   writes, events, synchronization or a runtime fallback.
2. Trace only existing host lifecycle timestamps and successful enqueue returns:
   worker start, first successful write return, main enqueue-return observation,
   every65536 successful writes, worker end and join. Main observation records
   count plus before/after monotonic bounds, not an impossible exact simultaneous
   clock/counter snapshot. These are host submissions, never CQ retirement.
3. A cross-thread read of ordinary `Traffic.emitted` would race. Use one separate
   lock-free uint32 publication store per successful enqueue, with one main-side
   atomic observation. No atomic read-modify-write, mutex or hot-loop allocation.
   Sparse clock reads have a deterministic sequence cadence; reuse existing
   start/end/join timestamps. No stdout/file/JSON/network work in the loop.
4. Use fixed storage for the entire admitted uint32 repeat domain: at most65536
   samples (first write plus multiples of65536), about1MiB sample array plus
   fixed scalars, independent of selected N. C1 emits17 sparse samples. Serialize
   only after workers join/drain and existing frame retrieval. No adaptive work.
5. Host observation cannot have literally zero execution overhead. State the
   extra atomic stores and sparse clocks, exercise fixed-count seams and calibrate
   this exact new binary later; do not call B4 and C1 identical host throughput.
   The non-interference claim concerns device content/counts and fixed work,
   not guaranteed unchanged wall time. Host samples cannot manufacture coverage.
6. Mechanically reverse only the known host-trace insertions in native.cpp and
   require the exact measured B4 base-source SHA. Hash the trace header and new
   compiled image; fail if protected/device-facing source changes. Preserve the
   old ancestor/build receipt and both B4 archives, never relabel their digests.
7. A host-only native change satisfies the owner's stated Watcher exception
   only with this source/dispatch proof and unchanged actual derived kernels,
   CTA and pinned compiler proof. Then the next approved C1 may remain Watcher
   off. Any device-content/dispatch difference stops preparation for a decision,
   rather than silently treating the new native as already hardware-validated.
8. Validate new trace fields only when present in persisted records, retaining
   accepted v1/v2/B4 behavior. New native success requires the new host trace;
   old records cannot be edited or upgraded to supply it. A failed enqueue must
   not create a successful-return trace point. Coverage stays frame/map-only.

## Admission invariants

Existing exact-native bounded admission cannot honestly reuse B4 after a host
trace changes the native digest. A versioned C1-only receipt must bind:

- Original B4 control/loaded archive, manifest and primary-record digests, their
  measured old program identity, same session/configuration, full healthy frame/
  warning/Watcher predicates and actual loaded causal coverage.
- The actual new current program identity/build receipt and a narrow, verified
  host-only source delta; unchanged library/device-kernel/CTA/compiler identities.
- Only `production-v1-calibration-small`, never C2 or P. No generic native mismatch
  bypass, implicit legacy promotion or equality assertion between old/new natives.

Prefer verified direct archive replay instead of untracked manual extraction
paths. All original bytes remain immutable. Admission data is not runtime/owner
execution authority; no boot/session is pre-created. Document this refinement
in an ADR carried by the same PR. Old exact-identity receipt behavior remains.

## Predictions, not thresholds

From B4 (Watcher on), inclusive worker model predicts25.479s. A hypothesis that
separates its0.679886107s enqueue lead predicts23.440s, or22.993s at+2%fast proxy.
The conditional frame/terminal envelope is17.167s, giving about5.827s fast-proxy
margin. No-Watcher/new host observation may change the rate; none is a bound or
advance latency pass/fail threshold. Expect intact16384 frames and full causal
coverage, with genuine `not_covered` if work ends early or an endpoint is missed.
Do not extend counts, infer coverage from host trace or retry. C2/P remain blocked.

## Board-free completion and next-boot handoff

Before stopping: RED/GREEN for actual host-loop sparse sampling, snapshot race
safety, failure paths, protected-source rejection and current-vs-B4 admission;
compile/link exact pinned libraries, preserve old native digest and record new
native; actual no-device startup/raw-stream/absent-result check; publication
UID-boundary checker; unchanged4case device compile proof; full tests/ruff/diff/
ADR guard; commit/push and matching-head CI plus public Sonar0/0/OK without accept.

At next boot, after the owner powers p150a on, repeat fresh environment/host-load/
no-container preflight against owner-approved Linux6.8.0-149/KMD2.12.0 and all
other exact B4 board/serial/PCI/firmware/image/native/kernel/800MHz/DRAM values
(except the explicitly precommitted new host native). Create a new boot-bound
C1-only session at the then clean/pushed green head; never reuse B4's session.
Use the approved device0 named wrapper, Watcher off and60s native/outer caps.
Only one C1 attempt; no C2/P. Stop on new/changed warnings, malformed evidence,
clock/coverage/frame/health/timeout/exit failures. At most one reset only for
abnormal device execution/timeout, no automatic retry. Record and publish raw
statuses and scope-qualified metrics afterward, with no control difference.
This paragraph is future procedure, not a command executed during preparation.
