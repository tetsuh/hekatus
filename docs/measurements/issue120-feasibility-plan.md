# Issue #120: board-free feasibility plan

- Status: **phase-1 preparation; no board runs**.
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
not shadow this dependency. CMake checks the pinned coordinate-header hash and
links the companion pinned-image `libtt_stl.so` explicitly.
The build command does **not** run the native executable or open a device:

```bash
uv run python -m enodia.tt.bench.issue120_probe --build-only \
  --tt-metal-source-root "$TT_METAL_SOURCE" \
  --tt-metal-source-revision d9a68815f5fcf08a5bfbffb6f1f811823fba8edd \
  --tt-metal-library "$PINNED_METAL_LIB" \
  --tt-metal-extra-include "$THIRD_PARTY_INCLUDE" \
  --tt-metal-extra-lib-dir "$PINNED_LIB_DIR" \
  --tt-metal-extra-lib-dir "$PINNED_WHEEL_LIBS" \
  --build-dir out/issue120-native-build
```

Variables name externally provisioned development inputs, not published personal
paths. Do not substitute a newer local Metalium library. Host compilation/linking
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
| P5 | Q4, fixed CPU arithmetic and 256 H2D writes | 30 s | 60 s | off | `out/issue120/p5` | Initial small fixed work likely fails full-window coverage; calibration only. |

Five invocations: **300 seconds total outer caps**, with native caps totaling
210 seconds. Build and host metadata preparation are separate from these caps.
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

Q4 executes 1,024 by 1,024 fixed integer operations and fixed-size submissions.
It is a short synthetic-pair **coverage falsifier**, not the production frame
stream. A CPU span shorter than the conservative program envelope is not_covered;
otherwise PCIe coverage remains not_evaluable without progress evidence. One
payload witness cannot establish continuous PCIe traffic or first-through-last
frame coverage. Do not select production counts from a positive witness alone.

## Mechanism-selection branches (Q4 remains a gate)

1. A wholly inside Q1 bracket establishes overlap for that exact setup. A
   before-window, boundary, missing, or tail-only result retains its narrower
   meaning; host wait duration is never the deciding evidence.
2. **First candidate: ordinary unpinned write-only CQ1.** It is the smallest
   extension of the retained #118 host path, eliminating its per-repeat
   completion/readback dependency without allocating per repeat. Select it only
   after card-side progress, bounded memory and full-window coverage are shown.
3. If CQ1 remains serialized, evaluate disjoint-core pinned-host NoC traffic.
   Select only after mapping/API checks, Watcher validation, payload/counter
   validation and same-clock coverage evidence succeed. Keep fixed host CPU work
   concurrent and outside the data path.
4. If both work, prefer the simpler verified write-only path. If neither works,
   report the evidence and await the owner's CPU-only decision; never downgrade
   silently.
5. These five initial probes do not yet prove Q4's full-window premise. A
   **subsequent precommitted bounded progress/coverage confirmation**, including
   real frame endpoints, is required before selecting production work counts.
   Its configuration/run count must be added before execution. Fixed small
   counts failing coverage do not prove that the mechanism itself is impossible.
   This limitation is explicit phase-1 residual work, not a completed AC.

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
