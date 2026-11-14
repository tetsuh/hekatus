# Issue #120 P6 API-lock/semaphore validation: branch 3

## Scope and provenance

Owner authorized only `p6-api-lock-semaphore-v1` and conditional P7 at clean,
pushed harness `af5db15ad92a6b6d432cf82837dbc74a5272fd64`. Native SHA-256:
`600d02aca3f1fbd50454357392810007019d189d8a64dcb2f47af0b0c0b252b8`.
The [precommitted plan](issue178-startup-diagnostic-plan.md) fixes work and
native/outer caps at 60 seconds, P6 Watcher on. Only device 0, serial
`000004033191411c`, PCI `0000:06:00.0`, was exposed to the named compute container.
Fresh provenance is in the [primary record](2026-11-07-p150a-issue120-p6-api-lock-semaphore-v1.json).

An initial preflight deferred launch for unrelated host CPU work. After the
operator ended it, fresh preflight confirmed no heavy process, zero containers,
both cards at 800 MHz/DRAM healthy, matching head/hash/device mapping and new
environment values. No measurement occurred during that deferral. No heavy
agent command overlapped the single P6 invocation.

## Result and prediction

**Branch 3: start remains 0 and emitted/expected targets differ. Stop before P7.**

- Native, container and wrapper all exited 0; no timeout. This is a semantic
  instrument failure, not accepted timing evidence.
- Startup response challenge **1**, `producer_gated_after_load_observation=true`:
  the repaired scheduling no longer starved all response writes until finish.
- Produced/consumed/attempted **0/0/0**, aggregate abort **2** (producer and
  consumer binary flags each 1), `coverage=not_evaluable`, reason
  `incomplete real frame stream`. There are no frame rows.
- Tail generations advanced **18845 -> 262144**, response challenge **2**, valid
  payload. Post-drain progress is valid generation **262144**, challenge **2**.
  These establish response progress, not first-through-last-frame coverage.
- All original fixed work completed: **8589934592 CPU operations**, **262144 H2D
  transfers / 1073741824 bytes**, **262144 direct load reads / 1048576 bytes**.
- The only seven warning lines matched the three accepted #176 conditions
  (including the adjacent NUMA fallback detail). No new warning/error was found.
  Watcher retained **12 actual dumps**, no anomaly matches, producer K and
  consumer NSID during the wait. A clean Watcher is not a healthy frame stream.

P7's branch-1 prerequisite is unmet. P7, retries and production were not run.
Post-run containers were zero and both cards returned to 800 MHz/DRAM healthy.
**Reset count 0**: normal process exits and healthy postchecks did not justify one.

## Fixed semaphore/address readouts

IDs are ready/free/start/done in that order. All addresses below are local L1
unless a full NoC target is shown.

| Diagnostic | Value |
|---|---|
| Actual CreateSemaphore IDs / kernel assumptions | `[0,1,2,3]` / `[0,1,2,3]` |
| Resolved / producer / consumer semaphore base | `40768 = 0x9f40` / same / same |
| HAL KERNEL_CONFIG reference | `40704 = 0x9f00` (not the semaphore base) |
| L1 alignment / producer polled start | `16` / `40800 = 0x9f60` |
| Producer words after finish | `[0,0,0,0]` |
| Consumer words after finish | `[0,0,0,0]` |
| Producer exit raw start / post-finish raw start | `0` / `0` |
| Intended producer virtual coordinate | `(1,2)` |
| Expected start target | `0x81000009f60` |
| Consumer emitted start target | `0x141000009f60` |
| Emitted coordinate under pinned BH encoding | `(1,5)` |
| Target / coordinate / polled-local-address matches | `false` / `false` / `true` |
| Consumer mailbox address / exit / post-finish / expected | `1548288` / `2` / `2` / `2` |
| Consumer increment tick (consumer clock) | `3745701013` |
| Producer opened / exit tick (producer clock) | `3745534190` / `17245534289` |
| Diagnostic MMIO / final progress read | `9 reads,36 bytes` / `4096 bytes` |

The expected and emitted encodings have the same local word `0x9f60` but a
different Y field (bits 42-47); X (bits 36-41) is 1 in both. IDs, relocated
semaphore base and producer local poll address agree. The producer's retained
exit raw value and post-finish read are both 0, not merely a post-finish1 whose
arrival time is unknown. This narrows the independent item to coordinate
provenance/addressing or routing/coordinate interpretation, rather than proving
an ID/base mismatch or a cache-only defect.

In harness `af5db15`, the consumer derives the target coordinate from
`anchor.get_noc_addr(0) & coordinate_mask` and then combines it with
`get_semaphore(start_id)`. Host expectation derives from the intended producer's
translated coordinate and workload-resolved semaphore base. **No physical
foreign-core destination, alias interpretation, or complete causal code fix is
proven by this comparison alone.** Trace the sharded anchor/ring TensorAccessor
coordinate generation and virtual/NoC addressing board-free before choosing a
correction. Other cross-core ring/ready/free/done paths need the same audit.
No foreign-core reads or speculative rerun were added.

Diagnostics are fixed post-join/post-finish snapshots and remain separate from
load accounting. They are **never coverage evidence**. No absolute cross-core
tick alignment is assumed. The producer's own elapsed ticks are 13500000099,
consistent with its unchanged wait budget; no complete-frame duration follows.

## Host interval and estimate comparison

Host fixed work lasted **6.110385410 seconds**, shorter than the approximate
8.7-second pre-run estimate. The post-join CQ0 finish returned about4.327 seconds
after join. This endpoint is the explicitly scoped upper-bound host envelope,
not a card completion timestamp. Final CQ1 drain was129840 ns. With zero frames,
there is no last-frame timestamp against which to declare genuine early host
completion; neither the host envelope nor diagnostics manufacture coverage or
justify increasing counts/caps. The branch-3 stop is the precommitted outcome.

## Evidence policy and next gate

The primary manifest hashes **21 companions**, including byte-exact native
result/streams/direct execution metadata and wrapper result/environment/power.
Public generated and wrapper text companions are explicitly redacted and
trailing-whitespace-normalized derivatives; original private hashes remain in
the manifest. All prior records and their bytes are unchanged. The six frozen
power samples (800/1350 MHz) do not establish frame/end coverage or timing.

Q4 remains not evaluable. Ordinary unpinned CQ1 plus direct-L1 CPU challenge
observation remains the first candidate, **not production-adopted**; no CPU-only
fallback is selected. Next is board-free #178 coordinate provenance diagnosis,
tests-first correction if established, and a fresh bounded precommit/owner
approval. Healthy real-frame Watcher coverage and conditional no-Watcher
confirmation, long-run non-aliasing protocol, fixed-work calibration if needed,
and same-session control/loaded production acceptance remain outstanding.
