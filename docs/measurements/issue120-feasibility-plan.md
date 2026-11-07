# Issue #120: board-free feasibility plan

- Status: **phase-1b preparation; no board runs or container starts**.
- Issue: #120; milestone: M3, TT spike (Track B).
- Basis: #118 / PR #123, ADR-0005, ADR-0015, #111 frozen snapshots,
  and #104 paired-outlier analysis.
- Prediction publication: commit and push this plan and the probe implementation,
  and publish the draft PR **before** any board run. The PR remains incomplete
  until feasibility and subsequent measurement evidence exist.

## Closed invariants

1. Phase 1 never opens a device, invokes board telemetry, resets a board, or
   starts a device-enabled container. Stop before the first board run.
2. Source evidence is tagged, with relative file paths and line numbers;
   source guarantees, hypotheses, and unmeasured behavior remain distinct.
3. Q1 uses card-side payload-completion evidence, not host enqueue/wait duration.
   Use the designated observer's clock for both the resident window and its
   completion bracket. Do not assume cross-core clock equality.
4. Read only `get_timestamp_32b()`, continuously extending unsigned low-word
   differences across wraps. Do not use `get_timestamp()`. Missing/ambiguous
   clock observations are not repaired with host timing.
5. Resident cores `(0,0),(1,0)` and load core `(2,0)` are disjoint. Record actual
   CQ/sub-device configuration and use an identically configured control later.
6. CPU and transfer load have fixed counts and sizes. No convergence criterion,
   duration-driven load loop, or repeat-until-covered loop is permitted.
7. Reuse bounded host/device allocations. Runtime command rings may backpressure;
   a four-buffer pool does **not** imply only four outstanding writes. Keep
   payloads, pins and destinations alive through completion before release.
8. Every probe's outer cap is at most 60 seconds. The first run of each new
   kernel uses Watcher and the exact 60-second cap. No automatic reset/retry or
   cap enlargement follows a failure.
9. Tests precede implementation. Board-free tests cover partitions, fixed work,
   repeat-independent memory, record fields, clock bracketing and cap selection.
10. Records carry complete sanitized provenance, counters, actual intervals,
    cap/Watcher state and explicit missing-evidence reasons. A payload witness
    is not full-window load coverage. Probe records are not production timing
    evidence.
11. Historical records remain immutable. No production architecture, external
    API, or timing contract is changed by this experimental instrument.
12. Production remains device 0, 500,000 frames, nominal 1 ms, Watcher off,
    600-second host-monotonic cap, and a same-session control. Card ticks are
    cycle-domain observations, not an AICLK-independent wall-time guarantee.
13. Retain ADR-0005 provenance, #111 frozen trace bytes, #104 paired outliers,
    control differences and first-through-last-frame load evidence.
14. Publish no hostnames, usernames or personal artifact paths. Repository
    prose/code are English. Commit with a real-newline bullet body; create a
    draft PR, without merge, auto-merge or the review driver.

## Q1–Q3 source audit

Source is tt-metal **`v0.75.0`**, annotated tag object
`e45135c5516c6ab18cf855c585d31da684fe64db`, peeled commit
`d9a68815f5fcf08a5bfbffb6f1f811823fba8edd`. The tag was fetched and inspected
through an archive, without changing the source checkout or opening a board.
All line references in this section refer to this tag.

### Q1: execution is not host completion

- `tt_metal/distributed/fd_mesh_command_queue.cpp:383-466` determines a program's
  single sub-device, takes CQ ownership and advances expected worker counts.
  A different CQ ID alone does not guarantee concurrent progress.
- `tt_metal/impl/buffers/dispatch.cpp:78-80,970-979` inserts worker-stream waits
  at the beginning of writes. Data itself can be gated, not just host completion.
- `tt_metal/impl/buffers/dispatch.cpp:1841-1850` substitutes the configured
  sub-device stall group for an empty explicit list. Selecting only the transfer
  sub-device removes the resident from that wait list, but is not a blanket
  independence guarantee for every dispatch path.
- `ttnn/cpp/ttnn-nanobind/device.cpp:590-600` forwards the optional CQ and
  sub-device list. `tt_metal/distributed/distributed.cpp:161-171` finishes only
  the named CQ when supplied, otherwise all CQs. A supplied CQ1 ID is not an
  unconditional CQ0 finish in this source.
- `tt_metal/distributed/fd_mesh_command_queue.cpp:691-729` implements finish by
  recording a host event and waiting for outstanding reads.
  `tt_metal/impl/event/dispatch.cpp:71-116,146-155` places worker waits before
  publishing the event; the final wait also stalls prefetch and applies a NoC
  barrier. `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:1018-1060` polls the
  selected memory/worker stream until the expected count is reached. A blocked
  device wait prevents the host-completion event from arriving.

**Conclusion:** source identifies possible gates, but does not establish whether
CQ1 transfers overlap CQ0 on this setup or uniquely explain #118's approximately
49-second wait. **Source does not settle it; probe it.** Do not turn a delayed
host wait into evidence that no card transfer occurred.

### Q2: ordinary write-only can omit host completion

- `tt_metal/distributed/mesh_command_queue_base.cpp:175-204` dispatches ordinary
  raw-pointer writes and finishes only when `blocking=true`.
- `tt_metal/impl/buffers/dispatch.cpp:539-605` copies ordinary unpinned payload
  bytes inline into the issue command; a new retained host/device tensor per
  transfer is unnecessary.
- Crucial exception: `tt_metal/distributed/mesh_command_queue_base.cpp:231-285`
  automatically records a **host event** for nonblocking writes that actually
  use attached PinnedMemory. `tt_metal/distributed/pinned_memory.cpp:273-310`
  retains unfinished barrier events and drains them on lock. Nonblocking alone
  therefore does not imply completion-free submission or bounded event storage.
- `tt_metal/impl/dispatch/system_memory_manager.cpp:499-542,673-710` uses fixed
  issue/fetch rings with fetch-queue backpressure. No trace/bypass capture is
  used by these probes. This bounds storage, not enqueue latency or progress.

**Conclusion:** one unchanged ordinary host payload and a persistent destination
pool, FIFO CQ1 writes, no per-repeat readback/event/finish, and one final drain
are a source-supported bounded candidate. Reusing an unchanged input needs no
wait for host overwrite; no competing device consumer may assume an unsupported
ordering. The write's worker wait remains. **Sustained progress and coverage are
not settled by source; probe them.**

### Q3: disjoint-core pinned-host NoC traffic is expressible

- `tt_metal/api/tt-metalium/experimental/pinned_memory.hpp:54-62,164-179`
  provides `PinnedMemory::Create(mesh, range_set, HostBuffer&, map_to_noc)` and
  `NocAddr {pcie_xy_enc, uint64 addr, device_id}`. The documented 64-bit address
  requires `noc_wwrite_with_state` and the separate-coordinate
  `noc_read_with_state` overload, not a truncated packed host address.
- `tt_metal/distributed/pinned_memory.cpp:72-85,110-144` maps existing nonempty
  host memory after aligning its base to the system page. Associated MMIO
  devices own the mapping. Blackhole's 64-bit PCIe path does not need iATU.
- `tt_metal/distributed/pinned_memory.cpp:206-245` returns translated PCIe
  coordinates and the full device I/O address. Require matching MMIO device ID.
- `tt_metal/distributed/pinned_memory.cpp:355-385,388-404` creates/attaches the
  pin and advertises pin count/byte capacity and NoC support. Its capability
  query returns zero capacity without IOMMU. Actual host configuration and
  allocation success are unmeasured, not inferred from an old environment.
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:1882-1905,2028-2053`
  implements separate coordinates and full 64-bit read/write addresses.

Allocate bounded source/output/statistics host regions, their pin, L1 scratch
and device statistics once. Account for page-alignment footprint and maximum
pin count/bytes. Initialize before DMA, retain ownership through completion,
use read/nonposted-write barriers before scratch reuse, and validate host output
only after CQ0 completion. No CPU writes DMA-owned memory during execution.

**Conclusion:** the source API represents the mechanism. Mapping support,
Watcher behavior, data integrity, useful traffic and full-window coverage remain
**probe questions, not source-established hardware facts**.

### Clock and pinned-image checks

`tt_metal/hw/inc/internal/tt-1xx/risc_common.h:247-256` distinguishes full-clock
low/high reads from low-word-only reads. The #104 controlled result supports the
low-word path; it does not establish the hypothetical shared-latch mechanism.
Poll NoC readiness while sampling low words. A modulo delta cannot detect an
arbitrary whole-wrap pause; continuous sampling is a requirement, and ambiguous
or stalled observations cannot become timing evidence.

The pinned image digest is
`sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621`.
Board-free inspection used named, stopped containers, no devices, no privilege,
container lists before/after, export/copy only, and removal afterwards. None was
started. The image lacks native development headers/CMake package, but its
`libtt_metal.so` exports the required mesh/pinned entry points. Its SHA-256 is
`daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141`.
Entry-point presence is not proof of full source/image byte equality or DMA.

## Build before board availability

The isolated native instrument supplements Python board-free planning/validation;
TTNN has no binding for the pinned API used here. Build against exact-tag headers,
explicit third-party headers, and the audited pinned-image library/dependencies.
Populate the tag's UMD submodule at
`bc77ac2f7382ffeb9fa5430055f0854d53979489`; an archive of the parent tag alone
omits submodules. Newer UMD coordinate headers change the ABI namespace and must
not shadow this dependency. Stage the tag's `tt_metal/llrt` and
`tt_metal/impl` headers too: Q4 uses a pinned internal targeted-read interface.
CMake checks the pinned coordinate-header hash and links the companion
pinned-image `libtt_stl.so` explicitly. Its pinned Python generator creates the
HAL interface headers during this host-only build; it does not run firmware or
the probe executable.
The build command does **not** run the native executable or open a device:

```bash
uv run python -m enodia.tt.bench.issue120_probe --build-only \
  --tt-metal-source-root out/issue120-deps/source \
  --tt-metal-source-revision d9a68815f5fcf08a5bfbffb6f1f811823fba8edd \
  --tt-metal-library out/issue120-deps/lib/libtt_metal.so \
  --tt-metal-extra-include out/issue120-deps/include \
  --tt-metal-extra-lib-dir out/issue120-deps/lib \
  --tt-metal-extra-lib-dir out/issue120-deps/wheel-libs \
  --build-dir out/issue120-native-build
```

Stage the audited headers/libraries in those repository-local dependency roots
before building. Build output is confined to `out/issue120-native-build`; source,
include and library inputs are confined to their respective `out/issue120-deps`
roots. Paths are resolved before use; root/child symlink escapes and CMake argument
syntax in paths are rejected. The source revision and native probe/stall-group
arguments come from fixed allowlists, not unvalidated command fragments.
Execution additionally confines environment/result paths to the wrapper output
mount. Do not substitute a newer local Metalium library. Host compilation/linking
is not device-kernel JIT validation; the latter remains the first Watcher run.
Phase 1 verified host compilation and linking against these exact tag/UMD
headers and pinned-image libraries, using GCC 13.3.0 with explicit third-party
headers. The executable was never run, including for help/version output.
The staged executable and audited library are ignored build output, not evidence
of a board run. Device-kernel JIT and actual loader/device execution remain
unverified. A board-free inventory is available with:

```bash
uv run python -m enodia.tt.bench.issue120_probe --plan
```

## Planned probe runs and predictions

**Do not execute these in phase 1.** After owner-confirmed board availability,
run from the repository root against the clean/pushed, PR-published head.
Every invocation uses the existing named-container wrapper and device 0.
Artifacts are wrapper-owned `runner-result-<run-id>.json` plus environment/logs
under the indicated relative directory. Failures retain their raw wrapper logs;
unavailable card fields must never be invented to make a complete record.

| Order | Purpose | Native cap | Outer cap | Watcher | Record directory | Prediction, not threshold |
|---|---|---:|---:|---|---|---|
| P1 | Q1, both sub-devices in stall group; first observer/companion kernels | 60 s | 60 s | on | `out/issue120/p1` | Possible worker-stream gating; card witness or explicit missing/ambiguous observation. |
| P2 | Q1, transfer-only stall group | 30 s | 60 s | off | `out/issue120/p2` | Removing resident from this wait list may permit overlap; not guaranteed. |
| P3 | Q2, 256 write-only 64 KiB transfers (16 MiB) | 30 s | 60 s | off | `out/issue120/p3` | No per-transfer host completion, but possible backpressure/device stall. |
| P4 | Q3, 256 fixed 4 KiB reads and writes; first pinned kernel | 60 s | 60 s | on | `out/issue120/p4` | Mapping may support verified DMA; unsupported capacity/mapping blocks selection. |
| P5 | Legacy Q4 fixed CPU arithmetic and 256 H2D writes | 30 s | 60 s | off | `out/issue120/p5` | Initial small fixed work likely fails full-window coverage; calibration only. |
| P6 | Q4 real-frame coverage, first producer/consumer kernels, causal CPU challenge observation | 60 s | 60 s | on | `out/issue120/p6` | With concurrent CQ1/direct-L1 progress and sufficient fixed work, every frame should receive a fresh challenge response. Serialization, torn observations, or exhausted work must fail coverage. |
| P7 | Q4 real-frame coverage, identical fixed work without Watcher | 60 s | 60 s | off | `out/issue120/p7` | Expect the same functional coverage if P6 established a healthy instrument; no latency threshold or production timing result is implied. |

Seven invocations: **420 seconds total outer caps**, with native caps totaling
330 seconds. Build and host metadata preparation are separate from these caps.
No extra run/reset is automatic. The synthetic observer has a primary
2,700,000,000-tick window and the same fixed tail, approximately two plus two
seconds at 1350 MHz, not a wall-time guarantee.

```bash
TT_METAL_WATCHER=1 HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p1 -- \
  --execute --probe q1-cq1-completion --stall-group resident-and-transfer \
  --first-kernel-run --cap-seconds 60

env -u TT_METAL_WATCHER HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p2 -- \
  --execute --probe q1-cq1-completion --stall-group transfer-only --cap-seconds 30

env -u TT_METAL_WATCHER HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p3 -- \
  --execute --probe q2-h2d-write-only --cap-seconds 30

TT_METAL_WATCHER=1 HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p4 -- \
  --execute --probe q3-pinned-noc --first-kernel-run --cap-seconds 60

env -u TT_METAL_WATCHER HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p5 -- \
  --execute --probe q4-fixed-window --cap-seconds 30

TT_METAL_WATCHER=1 HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p6 -- \
  --execute --probe q4-frame-coverage --stall-group transfer-only \
  --first-kernel-run --cap-seconds 60

env -u TT_METAL_WATCHER HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p7 -- \
  --execute --probe q4-frame-coverage --stall-group transfer-only --cap-seconds 60
```

### Evidence interpretation and memory

Q1 zeroes one 64 KiB payload outside measurement. The designated resident observer
polls a witness page, then checks every word of all sixteen pages before recording
completion. The conservative bracket runs from the last negative read's start
through the successful scan's end. The fixed injection delay schedules work; it
is not timing evidence. Boundary-crossing brackets are not_evaluable. The actual
observer-active interval is separate from the primary window: a tail observation
still occurs in an active CQ0 program. Missing observations do not prove absence
of transfer execution, or prove execution only after CQ0 termination.

Q2 uses one ordinary immutable host payload, four reusable device buffers and
fixed scratch/statistics. Submission and final drain intervals are separate.
No host readback/per-repeat completion is requested in the load loop; the
observer's device-local NoC reads are instrumentation. Runtime rings/allocator
metadata are outside the explicit buffer ledger; report process maximum RSS too,
compare short and larger fixed repeat counts before claiming empirical bounded
memory, and never call queued bytes bytes-completed-during-window.

Q3 pins once, checks aligned footprint/capacity/MMIO identity, and validates
read/write payloads and fixed counters after CQ0 completion. Worker and observer
clocks are not assumed equal, so this first Q3 run does **not** claim resident
coverage. No pin or underlying host allocation is released during DMA.

P5 retains the original 1,024 by 1,024 integer-work coverage falsifier unchanged.
It never establishes successful frame coverage from a payload witness or host
envelope. P6/P7 add a separate real-frame instrument, described below.

## Q4 real-frame coverage proof (P6/P7)

The new probe processes **4,096 actual 4 KiB ring frames**, at a nominal
1,350,000-tick period, on the resident pair. The producer fills/publishes each
frame using NoC writes and semaphore credits; drop-new overflow is counted.
The consumer performs 64 fixed work steps per frame and exports its actual
consumer-admission/completion ticks, frame identity, checksum and progress evidence. The
4,096-frame schedule crosses a low-word wrap even at the 1350 MHz reference.
There is no assumption of equal producer/consumer clocks: all proof endpoints
are stamped by the consumer's continuously sampled low-word clock.

Fixed load: **262,144 repetitions**, each with a four-byte direct L1 challenge
read, 32 by 1,024 volatile integer operations, fixed payload generation, and one
ordinary unpinned 4 KiB CQ1 write. This is 8,589,934,592 arithmetic operations,
1 GiB H2D payload, and 1 MiB direct-read payload in total, not all necessarily
inside the frame window. The one staging page is reused only after the ordinary
inline-copy enqueue returns. No per-repeat CQ completion/event/readback is used.
The direct L1 observation is explicit instrumentation and part of the workload;
do not present this as PCIe write traffic alone.

The targeted read uses v0.75.0's pinned internal `MetalContext`/`Cluster`
interface and a fixed host span, with no new device open. In
`tt_metal/llrt/tt_cluster.cpp:867-888`, `read_core` sanitizes the target under
Watcher and delegates a read to the driver, without CQ finish or a global
memory barrier. This is experimental, exact-tag-bound instrumentation, not a
portable production API. The driver may select its configured read transport;
the four-byte payload count excludes PCIe protocol overhead.

The public span `ReadFromDeviceL1` is deliberately **not** used: its implementation
at `tt_metal/impl/host_api/tt_metal.cpp:302-320` adds an L1 barrier. The pinned UMD
revision's `device/chip/local_chip.cpp:445-483` shows the default barrier covering
all L1 cores and Blackhole DRAM. That extra device-wide host-write flush is
unnecessary for this device-owned mailbox: initial host writes are completed
before launch, and the consumer fences each token store before a host can read
it. Skipping the public helper does not skip a CQ completion that this protocol
needs. The targeted interface still needs first-run Watcher and hardware
validation; a stalled observer is not proof that uninstrumented writes fail.

### Required evidence

- Before releasing the producer, the consumer issues startup challenge 1 and
  verifies progressing, consistent responses. Each host response can be created
  only after reading the card-issued token and performing that repetition's
  fixed CPU work. Thus prequeued DMA alone cannot impersonate current CPU work.
- Generation and challenge occupy disjoint 19-bit/13-bit fields of one 32-bit
  wire marker. Every payload word derives from that packed marker; generations
  cannot cancel challenge bits as a simple XOR encoding would. Repetitions are
  restricted to 524,287 and the fixed frame count fits the challenge field;
  neither field wraps or aliases during the probe.
- For each actual frame completion interval, the consumer issues a new token after the
  preceding completion, then samples the fixed-size response before the current
  completion. It requires a newer complete payload generation with the matching
  challenge. The bracket's lower endpoint is the token-issue tick; its upper
  endpoint is the validated response-read tick, on that same consumer clock.
- After the last frame, another newly issued token must receive a fresh CPU/DMA
  response within the bounded tail. Load start precedes the first frame and the
  final response follows the last; all intermediate frame intervals are checked.
- Normal frame processing never waits for a matching response or a higher
  counter. Each observation makes exactly three full-page reads and selects a
  consistent one; it does not retry until coverage succeeds. Only startup and
  the post-frame tail have bounded observation waits, outside frame processing.
- Missing/reordered frames, overflow, aborted work, checksum mismatch, torn or
  stale responses, missing challenge causality, or ambiguous clock sampling
  prevent `covered`. Host timestamps, queued totals and declared coverage flags
  cannot repair them. Python recomputes the verdict from every frame row.

Only **one witnessed H2D payload per valid bracket** is counted toward the byte
lower bound. A generation jump is not converted into unobserved transfer bytes.
The separate during-frame lower bound excludes the first pre-frame bracket and
post-frame tail; queued totals are retained separately. The preallocated frame/output
capacity is fixed by 4,096 frames, not by the transfer repetition count. Runtime
rings, retrieval metadata and allocator alignment overhead are outside the
explicit payload ledger and remain repeat-independent. There is no per-repeat
retained buffer or record list.

P6 and P7 use two CQs with resident/transfer sub-devices and a transfer-only
stall group. Their direct-read challenge instrumentation and device-open setup
must be retained or independently revalidated before extrapolation to production.
P7 follows only after a healthy P6 Watcher run. A timeout, bad Watcher report,
missing progress or insufficient fixed work stops the sequence for an owner
assessment and a newly committed count/configuration if needed, not an automatic
repeat-until-covered loop. These probes establish bounded diagnostic feasibility,
not the 500,000-frame production measurement or an AICLK-independent deadline.

## Mechanism-selection branches (Q4 remains a gate)

1. A wholly inside Q1 bracket establishes overlap for that exact setup. A
   before-window, boundary, missing, or tail-only result retains its narrower
   meaning; host wait duration is never the deciding evidence.
2. **First candidate: ordinary unpinned CQ1 bulk writes with explicit direct-L1
   challenge observation.** It eliminates per-repeat CQ completion/readback
   waits without allocating per repeat. Select this instrumented mechanism only
   after Q1 overlap evidence, fixed-memory accounting, and successful P6/P7
   card-side CPU/DMA progress across every real frame, with valid first/last
   endpoints, zero overflow/aborts and matching frame checksums. No host-envelope
   or queued-count success path exists.
3. If CQ1 remains serialized, evaluate disjoint-core pinned-host NoC traffic.
   Select only after mapping/API checks, Watcher validation, payload/counter
   validation and same-clock coverage evidence succeed. Keep fixed host CPU work
   concurrent and outside the data path.
4. If both work, prefer the simpler verified write-only path. If neither works,
   report the evidence and await the owner's CPU-only decision; never downgrade
   silently.
5. P6/P7 implement the previously missing real-frame confirmation, but have not
   been run. No mechanism is selected and no feasibility AC is marked complete.
   Insufficient counts do not prove a mechanism impossible; a direct-read stall
   may be an instrumentation limitation. The pinned-host alternative still
   needs equivalent causal frame-coverage evidence before it could be selected.
   Production counts require separately committed, conservative calibration.
   The 19-bit/13-bit marker is for this short probe only; production's larger
   frame/repetition domain requires a non-aliasing widened protocol and another
   bounded validation before any 500,000-frame run.

## Production follow-up (not executed or enabled here)

After feasibility and owner approval, integrate the selected mechanism with the
real resident ring runner. Its existing non-calibration host condition remains
rejected; this probe is not a replacement for that guard or production instrument.

1. Commit/push the selected mechanism, fixed CPU/transfer counts, coverage margins
   and predictions, and publish them in the PR before running. No advance latency
   pass/fail threshold. Insufficient predicted work/cap means stop, not fallback.
2. Device 0, 500,000 frames, nominal 1 ms (1,350,000 ticks at the 1350 MHz
   reference), Watcher off, 600-second host-monotonic cap. Do not substitute the
   #118 800-second compute cap. Keep raw ticks and weighted in-run AICLK separate
   under ADR-0015; do not call nominal cycle pacing a product wall-time guarantee.
3. Take an unloaded control in the **same session**, with identical CQ count,
   sub-device setup and selected mechanism's device-open configuration. Bind
   records to the session/configuration; reject an unlike control.
4. Start fixed CPU/PCIe work before the first frame, end after the last, and
   retain actual CPU/load intervals, worker/progress counters and bytes during
   the frame window. Queued totals or an envelope alone are insufficient.
5. Use ADR-0005 records and #111 run-bound frozen power snapshots. Freeze before
   hashing/judging; preserve exactly the referenced bytes, hash, byte/sample
   counts and stated run-end coverage definition. Do not append live rows to
   repair an incomplete snapshot. The #157 capture race is not changed here.
6. Report P50/P99/P99.9/P99.99/max intervals, overflow/drop/abort counts, #104
   paired outliers and positions, load coverage, actual bytes/counts, memory
   evidence and same-session control differences. Retain external raw timestamp
   bytes with count/SHA-256, not personal paths.
7. Keep results immutable and add the outcome to design.md section 12 and the
   measurement index in the result PR. No full-run result is fabricated here.

## Record provenance and privacy

Each probe/production record must carry **board model, serial, PCI BDF, firmware,
KMD, exact image digest, CPU model and clean/pushed harness commit**. Obtain these
from the future run-bound environment, not historical defaults or the powered-off
board in phase 1. Record source tag/commit/library digest, core/CQ/sub-device
configuration, fixed work, allocation ledger/queue-bound caveat, host monotonic
intervals, card completion/window semantics, cap, Watcher, status and errors.
Sanitize to an explicit field allowlist; no hostname, username or personal path.
Probe records set `timing_evidence=false`. Results with absent evidence remain
blocked/diagnostic, never relabeled as production timing evidence.

## Runtime-root correction and renewed P1 precommit (Issue #173)

This section supplements, rather than rewrites, the original P1-P7 predictions.
The original P1 at `b1865fe` stopped during runtime initialization; its
[immutable failure record](2026-11-07-p150a-issue120-p1-launch-failure.json)
is unchanged. P2-P7 remain unexecuted. This correction is board-free and does
not authorize another hardware invocation.

### Closed startup invariants and exact-image audit

1. The image digest remains unchanged. The launcher supplies only the fixed
   installed-wheel root `/opt/venv/lib/python3.12/site-packages/ttnn` through
   `TT_METAL_RUNTIME_ROOT`. There is no runtime-root argument. An inherited
   root is replaced, and the legacy `TT_METAL_HOME` value is removed.
2. Before native subprocess creation, inspect the program-owned wheel root and
   all four fixed asset trees below. Missing/wrong-kind roots, redirected roots,
   symlinks, unsupported file types, access errors and inventory mismatches
   reject the launch. Planning/build-only paths do not need installed assets.
3. Inventories bind sorted relative names, directory/file kinds, file sizes and
   executable attributes. The exact pinned trees contain no symlinks. This is
   a complete layout/metadata check, not a content hash for every file; the
   immutable image digest and existing Metal library SHA-256 remain the content
   provenance boundary. Do not claim source/image byte parity.
4. Asset inspection does not import TTNN or call accelerator APIs. Tests of
   every rejection branch precede implementation, and native calls are mocked.
5. Any device-free native startup diagnostic additionally hides the host's
   hardware-discovery filesystems, runs without devices/network/capabilities,
   and is not a successful Watcher/kernel validation or P1 attempt.

The exact image was inspected using named device-free, network-disabled,
read-only containers and stopped-container file copies. Its installed
`ttnn/__init__.py:599-609` selects its own package directory and calls
`SetRootDir` for an installed wheel. The `ttnn-0.75.0.dist-info/RECORD` and bundled
files were inspected without importing the package. The wheel contains
Blackhole core/SoC descriptors, firmware/kernel sources, LLK headers,
precompiled firmware and runtime objects/linker scripts. SFPI is installed at
its separate fixed image path, not in the wheel's `runtime` directory.
The v0.75.0 source `tt_metal/jit_build/build.cpp:123-140` identifies the local
and system SFPI search locations; source support is not JIT validation.

| Fixed asset tree | Directories + files | Files | Inventory SHA-256 |
|---|---:|---:|---|
| wheel `tt_metal` | 3594 | 2795 | `8d85e48cca4c004fa41962846378fd06bd28bf57321ce24d7a9cafaf27a16314` |
| wheel `runtime` | 78 | 69 | `56fe4e441edf3f309dca55bd97db56a03c3f060e8bafa6bd1db2d3f05b41ac8b` |
| wheel `build/lib` | 6 | 6 | `06f873632238323b3891239edaca92e54eebc7e674755d0749e14c533df9c8f2` |
| `/opt/tenstorrent/sfpi` | 1599 | 1432 | `fb5e85084358f9deda5081a6ee953c5ee3360f7ce084da84b63a42a51b18ca8b` |

### Device-free startup diagnostic (not a P1 result)

The original native executable was launched through the corrected Python native
entry path in the exact image, with no device mappings, network disabled,
read-only root/source mounts, an unprivileged UID and all capabilities dropped.
In addition to Docker's default masks, host `/sys`, `/proc/bus` and `/proc/driver`
were masked, and absence of TT device nodes/visible discovery trees was asserted
before launching native code. Configuration and cleanup were checked.

All four asset inventories passed. The process passed the former empty-root
failure and stopped with exit 2 at `No chips detected in the cluster`, reporting
empty chip/PCIe ID sets. No real board was available to discovery. This verifies
only the library/root/startup path, not hardware initialization, JIT, DMA,
Watcher health or any Q1-Q4 acceptance gate. No source/image parity is inferred.

### New attempt identity and prediction (not executed)

- Logical identity: **`p1-runtime-root-v1`**. Never reuse the first attempt's
  output directory or run ID. The wrapper will assign a fresh collision-safe
  run ID under `out/issue120/p1-runtime-root-v1` after owner authorization.
- Use the clean, pushed correction revision containing this precommit, the
  same native C++ source/build fingerprint, device 0 only, the original
  resident-and-transfer stall group, Watcher on, native cap exactly 60 seconds
  and outer cap exactly 60 seconds. New kernels have not yet passed Watcher.
- Prediction: the pinned assets should pass preflight and the fixed root should
  avoid the observed empty-root failure. Hardware discovery, initialization,
  JIT and observer operation remain unproven. If they succeed, P1 may show
  worker-stream gating, an inside-window witness, or missing/ambiguous card
  evidence as originally predicted. Do not invent a transfer verdict from a
  launcher exit code. No latency threshold is added.
- Missing assets stop before the native process. Any abnormal exit, timeout,
  bad Watcher report or unexpected result stops without continuation/retry.
  Retain named-container cleanup and post-run health checks. The prior phase
  used zero resets; this plan grants no new reset or hardware authority.
- The renewed P1 adds one conditional 60-second outer/native cap. Together
  with the original seven scheduled invocations, the historical-plus-planned
  cap ledger is 480 seconds outer and 390 seconds native. Remaining P1 retry
  plus P2-P7 total 420 seconds outer and 330 seconds native. No batch is run
  automatically; preparation/CI/device-free diagnostics are separate.

Only after renewed owner approval:

```bash
TT_METAL_WATCHER=1 HEKATUS_TT_DEVICE_NODE=/dev/tenstorrent/0 \
  HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p1-runtime-root-v1 -- \
  --execute --probe q1-cq1-completion --stall-group resident-and-transfer \
  --first-kernel-run --cap-seconds 60
```

This is not a production-count or mechanism-selection decision. Keep the
original failure and all future attempts separate, and preserve their named
harness revisions without rebase/amend. Production remains gated on the
remaining bounded evidence, widened protocol, calibration and same-session
control in the original plan.

## Result-file correction and new P1 precommit (Issue #175)

The authorized `p1-runtime-root-v1` stopped at wrapper stdout parsing. Its
[immutable attempt](2026-11-07-p150a-issue120-p1-runtime-root-v1-failure.json)
and the original startup-failure record remain unchanged. No native stream can
be reconstructed retrospectively. This prerequisite is corrected in PR #172,
not deferred outside that PR, under [ADR-0018](../adr/0018-issue120-native-result-file.md).

### Native/wrapper result contract

- The wrapper's fresh `HEKATUS_TT_RUN_ID` is the transport attempt identity.
  It exclusively creates `/out/native-<run-id>/result`. The only accepted native
  result is its `result.json`; stdout/stderr are never result inputs.
- Native argv receives only that program-owned path and validated identity.
  Before mesh creation, native checks the exact derived path, canonical roots
  and empty slot. It writes `result.json.tmp` exclusively, fsyncs/closes it,
  atomically renames without replacement, and syncs the directory.
- The envelope has exactly `schema: issue120-native-result-v1`, matching
  `attempt_identity`, and object-valued `result`. The wrapper then applies the
  existing probe record/provenance/coverage validator. Atomic publication is
  not a substitute for those gates.
- `stdout.bin` and `stderr.bin` in the capture directory receive raw bytes
  directly from spawn. `execution.json` records schema
  `issue120-native-execution-v1`, identity, the direct wait return code,
  started/timeout state, error, stream counts and SHA-256. Preflight/spawn
  failure has no native exit code. No parse outcome deletes these artifacts.
- Exactly one entry must occupy the result slot. Missing/multiple entries,
  leftover temporary files, symlinks, malformed/oversized or duplicate-key JSON,
  schema/identity/type mismatch and nonzero/timeout execution all reject success.
  Cleanup logs can follow arbitrarily without changing result selection.
- An externally killed observer cannot guarantee its own final execution
  metadata; raw bytes already written survive in the output mount. Missing
  final metadata is incomplete evidence, never assumed success. Raw stream
  publication requires a separate identified redacted derivative if sensitive
  text exists; the raw bytes are not edited in place.

### Board-free verification of the new file channel

Host compile/link against the pinned dependencies succeeded. The rebuilt native
binary SHA-256 is
`1ca91e17b09db83b70b1cec1e09d69e61430d00d5bab9168a283d3503f9c2b4d`.
CPU-only publisher tests verify successful schema/identity publication and
no-overwrite behavior without linking Metalium.

A named, device-free, network-disabled, non-root/capability-dropped container
with read-only source/root and masked host discovery paths ran the corrected
native entry path. `No chips detected` ended with directly waited native code 2.
Both raw files survived: stdout 768 bytes and stderr 2282 bytes, with SHA-256
matching execution metadata. The result slot was empty and failure was refused.
No real device was visible, and this is not hardware/kernel/Watcher acceptance.
The two prior attempt archives remain byte-identical.

### New identity and prediction (not executed)

- Logical attempt: **`p1-result-file-v1`**, under the never-reused output
  directory `out/issue120/p1-result-file-v1`. The wrapper separately generates a
  new collision-safe transport run ID. Prior attempt directories are not reused.
- Use the clean/pushed correction revision, newly rebuilt native fingerprint,
  original resident-and-transfer setup, device 0 only, Watcher on, native cap
  exactly 60 seconds and outer cap exactly 60 seconds. Kernels have not yet
  passed an accepted P1 Watcher/result gate.
- Prediction: a clean native completion should publish exactly one matching
  envelope regardless of later diagnostics. Failure should retain both raw
  streams and direct execution status, not fabricate a record. If valid P1
  evidence is returned, interpret the original possible gating/inside-window/
  missing-observation hypotheses from card timestamps only. No new latency
  threshold or mechanism-selection claim is introduced.
- Any abnormal exit, invalid/missing result, timeout, Watcher anomaly or
  unexpected observation stops without retry or P2 continuation. Owner must
  separately authorize the real run; this fix and device-free diagnostic do not.
- One more conditional 60-second P1 adds to the historical cap ledger:
  original seven plus two separate retries total 540 seconds outer and
  450 seconds native. Remaining new P1 plus P2-P7 still total 420 seconds
  outer and 330 seconds native. No automatic batch or reset is authorized.

Only after renewed owner approval:

```bash
TT_METAL_WATCHER=1 HEKATUS_TT_DEVICE_NODE=/dev/tenstorrent/0 \
  HEKATUS_TT_CONTAINER_TIMEOUT_S=60 \
  HEKATUS_TT_RUNNER=enodia/tt/bench/issue120/run_issue120_probe.py \
  enodia/tt/bench/run_in_container.sh out/issue120/p1-result-file-v1 -- \
  --execute --probe q1-cq1-completion --stall-group resident-and-transfer \
  --first-kernel-run --cap-seconds 60
```

### Authorized result-file attempt outcome

The separately authorized `p1-result-file-v1` at clean/pushed `75be8ec`
completed with native/wrapper/container code 0 and a matching atomic envelope.
Its [immutable result](2026-11-07-p150a-issue120-p1-result-file-v1.json) and
[interpretation](2026-11-07-p150a-issue120-p1-result-file-v1.md) show all sixteen
payload pages completing inside the resident window, the predicted witness
branch. Host final drain finishes CQ0 before CQ1; it does not explain the
historical CQ1-wait observation.

Execution stopped conservatively before P2 for NUMA-binding/SMC-buffer runtime
warnings not pre-cleared in this plan. Watcher had no anomaly matches; both
cards returned healthy at 800 MHz with no containers. No reset/retry or
production run occurred. P2-P7 identities remain unconsumed, with their original
predictions unchanged. Further execution needs supervisor/owner assessment;
this outcome does not authorize continuation or adopt a mechanism.

### Owner-accepted baseline warning verification and P2-P4 continuation (#176)

For the separately authorized P2-P7 scope at clean/pushed `62f388c`, all three
known baseline warning conditions were verified **before board access** in
`issue146-doppler-conformance/2026-10-21/runs/run1-c018/wrapper-stdout-issue146-long-run1-c018.txt`:
NUMA membind/fallback at lines 8-9, motherboard bus-ID fallback at line 18,
and unavailable dispatch SMC telemetry buffers at lines 22-25. Warning text
matches apart from invocation-specific timestamp/PID/TID/device/bus values.
The adjacent hugepage NUMA consequence belongs to condition 1, not a new class.
Owner acceptance covers only these known baseline warnings (#176), never
additional warnings/errors. The three attempt records retain the pre-board
proof, source SHA-256/lines and explicit baseline decision.

The [run 4 record](2026-11-07-p150a-issue120-run4.json) and
[interpretation](2026-11-07-p150a-issue120-run4.md) retain P2 transfer-only
inside-window completion, P3's 256 fixed write-only submissions/final drain,
and P4's direct code-2 pinned-capacity preflight rejection. No mapping/pinned
kernel/DMA feasibility is inferred from that rejection; actual limits were not
logged. P5-P7 were not run. All named containers were removed and both cards
returned healthy at 800 MHz; no timeout/reset/retry or production run occurred.

Next steps are conditional proposals only: board-free #177 assessment and an
explicit branch/remaining-feasibility approval, real-frame causal CPU/PCIe
coverage, fixed-count calibration if needed, and widened non-aliasing protocol
validation before any 500,000-frame same-session control/load precommit. See
the run 4 interpretation for caps, Watcher and evidence gates; no additional
identity or workload is authorized here.

### Ordinary-CQ1 P5/P6 continuation and unmet startup gate

The owner separately authorized unchanged P5-P7 at clean/pushed `2eefe9a`,
selecting ordinary unpinned CQ1 for feasibility and explicitly excluding a P4
retry. The [run 5 result](2026-11-07-p150a-issue120-run5.json) and
[interpretation](2026-11-07-p150a-issue120-run5.md) preserve the exact work and
predictions. P5 returned the predicted `not_covered` insufficient-work falsifier;
under the explicit exception, this did not stop P6.

P6's native/wrapper/container exits were 0, but startup response challenge was
0, the producer gate was false, `aborted=2`, and no frames were produced,
consumed or attempted. Coverage is not_evaluable, not successful coverage or
evidence that merely increasing fixed work will help. P7 requires a healthy
P6 instrument and therefore was not run. Watcher anomaly scans and new-warning
checks were clean, both cards returned healthy at 800 MHz and no containers
remained. No reset, retry, P4 rerun or production run occurred.

Q4 diagnosis is tracked in #178. Any corrected startup validation needs closed
invariants/RED tests and a newly precommitted bounded attempt after diagnosis;
this plan does not create one or authorize hardware access. Production still
requires real-frame causal coverage, a functional no-Watcher comparison,
non-aliasing long-run protocol validation and same-session control/load gates.
The run 5 interpretation specifies the conditional 500000-frame/600-second
proposal and missing gates.

After all physical work ended, #177 was evaluated from exact tag/gitlinked UMD
source: effective identity/passthrough or unreadable/non-DMA group type causes
false IOMMU detection and early `{0,0,false}`; global `iommu=pt` alone is not an
unconditional per-device result. No host setting or kernel command line was
changed, no capability query was executed, and source/image parity remains
unasserted. See the linked source audit for conditions and limitations.

### Issue #178 board-free startup diagnosis: root unresolved

The [startup diagnosis and bounded diagnostic proposal](issue178-startup-diagnostic-plan.md)
verifies the saved P6 artifacts and traces source order without claiming a
physical cause from response 0. Abort total 2 combines two binary flags; raw
mailbox read history and individual wait stages were not retained. The actual
host traffic body was exercised in a CPU-only mock: supplied challenges are
encoded correctly, while never-published and wrong-target models both yield
zero responses. This is characterization, not a reproduced hardware defect.

No production startup/validator/count/cap change is made. The proposal defines
bounded stage, mailbox and response-path observations, no extra host MMIO reads,
and no per-iteration log growth. Proposed `p6-startup-trace-v1` (Watcher on) and
`p7-startup-trace-v1` (Watcher off) both retain native/outer 60-second caps and
all fixed work. P7 remains gated on healthy P6 and causal frame coverage.
These proposal identities are not runnable diagnostic builds: implementation,
committed revision/native fingerprint, tests and renewed owner approval are
required first. No new board invocation, directory, workload or automatic retry
is created by this plan. Full production gates remain unmet.
