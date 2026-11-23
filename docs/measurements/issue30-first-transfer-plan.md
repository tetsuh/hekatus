# Issue #30: first chip-to-chip transfer proposal

**Status: board-free plan only; no board action is authorized or executed.**

The G0 follow-up's [receipt, fixed judgment rules and final one-screen A/B
approval units](issue30-g0/README.md) supersede this document's original
preparation gaps. Upstream commands below identify the workload only; future
execution uses the attempt-bound wrapper, not an improvised bare invocation.
A failed/incomplete G0 receipt blocks both approvals.
Refs #30; related #13 and #105 stage 3. Source authority and all T/U reference
pins are in [the source study](issue30-source-study.md). This is not a runtime
training success claim, latency measurement, or selection of the FPGA protocol.

## Invariants before procedures

1. Use the digest-pinned runtime image and the exact source/submodule pins in
   the study. Never use the unrelated tt-metal development checkout or change
   the image tag to make a test run. Match source, headers, library, kernel
   tree and runtime root; no unnoticed wheel/source library mixture.
2. No fake physical descriptor, forced link-up bit, firmware/port write,
   directed retrain test, `tt-topology`, cable restart, automatic retry, cap
   extension or reset. A failed first run is an outcome, not permission to
   experiment. No PCIe/firmware/driver setting changes within a run.
3. Two-card authority is new and explicit. A device0-only #120 approval does
   not carry forward. Each hardware unit below requires its own owner approval
   at the tested/pushed head, current boot and approved environment/artifact
   fingerprints. Approval of A does not approve B; B does not approve timing.
4. Each hardware process/container attempt, including initialization, JIT,
   transfer and close, has a hard **60-second maximum**. Default UMD training
   waits can be 900 seconds (U timeouts header:32), so internal UMD timeouts
   are not the watchdog. First use of every kernel, including official smoke
   kernels here, has Watcher enabled. No Watcher-off run in this plan.
5. No changes to enodia nonblocking/data-plane contracts or old evidence.
   Verification reads the receiver's checked content/status, not an enqueue
   success, host log containing PASS, or a zero exit from a skipped test.

## Devices, scope and risks

| Physical card | Historical / owner-supplied identity | Intended use |
|---|---|---|
| A | serial/board-id alias `000004033191411c`, PCI `0000:06:00.0` (historical device0) | UMD discovery in A; source worker and fabric routers in B, subject to fresh enumeration |
| B | serial `0000040331914124`, PCI `0000:09:00.0` (second card) | UMD discovery in A; receiver worker/L1 and fabric routers in B, subject to fresh enumeration |

These are assertions to **verify**, not current observations. Device numbers
can change across boot/filtering. Record the mapping of runtime chip IDs,
PCI BDF, serial/board ID and Ethernet peer/channel IDs on both cards before B.
Do not assume receiver runtime index 1 means the specified physical B card.
UMD discovery itself opens devices; even A is hardware use, not a board-free
preflight. The official smoke fixture enumerates and opens **all available
chips**, not a CLI-selected single device
(`T:tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:105-127`).

Proposed UMD visibility filter is the two **BDFs**, not an inherited device0
filter: `TT_VISIBLE_DEVICES=0000:06:00.0,0000:09:00.0` (BDF support:
`U:device/pcie/pci_device.cpp:236-314`). Clear inherited
`TT_METAL_VISIBLE_DEVICES` and record both variables. Expose only the two
owner-verified device nodes to the container; do not assume node numbers.
If more/fewer chips appear or either identity differs, stop. No unapproved
third card may be opened. Filtering is not a hardware sandbox; restrict device
mappings and verify enumeration as well.

B introduces real writes/allocations on card B: worker L1 programs, fabric
ERISC application firmware, buffer/credit/routing structures, NoC traffic,
semaphores and JIT/runtime state. The fixture can reserve several Ethernet
cores/routing planes; choosing one forwarding link for five packets does not
mean only one core is initialized. A hang can leave either/both cards in an
unhealthy state. Watcher is diagnostic, not guaranteed recovery. Power/thermal
and driver failure risks extend to both cards. Any unrelated workload must be
quiescent and owners of both cards informed before A or B.

**Reset policy:** zero resets are included in A/B authority. On timeout or
abnormal exit: preserve evidence, kill only the named attempt container by
the cap, stop, report both-card state. If reset is necessary, request a
separate owner decision specifying card(s), reset scope (including whether a
shared reset affects the peer), method and one attempt maximum. A reset or
reboot invalidates run approval/environment/boot evidence and never authorizes
an automatic retry. Firmware flashing is a different intervention again.

## Morning owner checklist (before approving A)

- Power on **both** cards; allow firmware startup to settle. Verify both
  serial/BDF pairs and healthy DRAM/temperature/clock state against a fresh
  owner-approved baseline, not a copied #120 device0 health result.
- Physically trace the cable end-to-end. Record both card serials and the
  printed connector labels/orientation in a cable table; retain a labelled
  photograph/diagram privately if needed. Confirm seating, supported cable
  type/part and intended port pairs against the board/cable documentation.
  Do not unplug/replug during a run. No cable type/length is established by
  the source study.
- Use the source P150 port/channel table found in G0: port 1={11,9},
  2={8,10}, 3={7,5}, 4={4,6}, with port 1 furthest from the motherboard.
  Record both physical cable endpoints and translate ASIC channels through
  the SoC descriptor; compare real discovered peers in both directions.
  If no link exists, do not infer cabling from empty discovery or swap ports
  until a separate physical intervention is approved.
- Confirm actual bundle, CMFW and **per-core base ERISC** versions on both
  cards. Historical bundle 19.6.0.0 / CMFW 0.28.0.0 and tt-smi eth_fw=0 are
  insufficient: before bundle 19.9 that telemetry is unavailable/zero. UMD's
  base ERISC minimum is 1.4.1; bundle compatibility minimum is 18.5.0, its
  latest-supported constant is 19.7.1, while T installation guidance lists
  19.8.1. Resolve/approve the discrepancy and environment baseline explicitly;
  no version change or warning bypass is authorized here. An exact CMFW
  minimum for successful first training is **open**, not invented.
- Verify current host kernel/KMD, hugepage availability for **two** MMIO
  devices, device permissions and container mappings. T's installation table
  lists KMD >=2.8.0; prior #120 KMD/kernel observations are historical only.
  Do not modify IOMMU/ACS/hugepages/driver settings to pass a run. If allocation
  or access fails, record and stop. Virtual-machine requirements, if applicable,
  are in `T:INSTALLING.md:209-229`; do not assume a VM or change host isolation.
- Ensure no competing process on either card and no heavy host work; capture
  Docker state immediately before each attempt. Keep the same approved pins,
  boot, source/binary/library hashes and environment through a run.
- Complete device-free artifact gate G0 below. Green CI and public Sonar
  **0 new issues / 0 security hotspots requiring review / quality gate OK**
  at the PR head; no acceptance/muting action. Those checks do not authorize
  hardware or establish firmware compatibility.

Any future read-only tt-smi health capture is owner-operated/approved hardware
observation, **not executed during this brief**. Never use its write/reset
operations as a preflight shortcut.

## G0: device-free artifact preparation gate (not a board run)

**Identity:** `issue30-g0-official-smoke-build-v1`. Purpose: retain upstream
programs rather than invent a kernel. The inspected runtime image contains
ttnn 0.75.0 and matching router/handshake sources, **not the official test
binaries**. The [G0 follow-up](issue30-g0/README.md) has now built them and
verified isolated loading/listing/zero-chip refusal, with receipts. Fresh
hardware observations and the separate A/B approvals are still required.

In a separate, device-unmapped build environment, initialize the pinned source
submodules and build upstream `test_system_health` and `fabric_smoke_tests`.
The upstream entry for Metal tests is `./build_metal.sh --build-metal-tests`
(`T:build_metal.sh:24-39,184-190`); CMake target definitions/output directories
are `T:tests/tt_metal/tt_fabric/CMakeLists.txt:63-123`, and
`sources.cmake:43` includes the exact smoke source. Preserve the full build
command, compiler/dependency provenance, source state and all binary/library
hashes. Check that the resulting binaries load **inside the pinned runtime
image without devices** and that listing tests includes exactly the requested
names. No test-suite setup/device probe may be run in G0. Do not assume a
runtime-only image supplies the source-build compiler/dependencies; failure
here stops preparation and needs a separately pinned build environment, not
a different runtime image or an unrecorded ABI substitution.

Use the same source-matching runtime/library/kernel tree in A/B. The official
smoke references test kernels outside the wheel, so include the exact pinned
`tests` tree. Confirm no `BENCHMARK_MODE` define: it compiles out receiver
content checks (`T:tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp:52-69`).
No custom MGD, mock descriptor, reliability override, profiler or two-ERISC
mode override. Retain default strict health setup and automatic P150_X2 MGD.
Record `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`, library resolution and working
directory using **logical roles/relative paths in public records**, not machine
names or local absolute paths.

The follow-up implements `enodia.tt.bench.issue30` with board-free tests for
arguments, absolute deadlines, schemas, file publication and source-shaped
positive/single-mutation negative judgment fixtures. Its default capture is
device-free; hardware additionally requires exact owner approval/environment/
head/boot/receipt bindings. See the G0 receipt for actual build/loader status;
source-level preparation is not a successful G0 receipt. Standard GoogleTest
XML is a result file, not the last-stdout-line transport rejected by ADR-0018.
Existing Issue #120 JSON schemas/runners are unchanged and not repurposed.

## Separately approved hardware units

### A — discover / report, no MeshDevice workload

- **Identity:** `issue30-a-discovery-watcher-v1`; fresh collision-safe attempt
  ID/container name also required, never reuse a result directory.
- **Purpose:** find where initialization/training observation stops, establish
  both chip identities and real bidirectional peer links before a fabric test.
- **Cap / Watcher:** 60 s total, inner binary 40 s, named-container watchdog
  50 s with cleanup deadline 60 s; `TT_METAL_WATCHER=1`. No new user kernel.
  Do not interpret Watcher absence of dumps as proof of ERISC health: A need
  not initialize application firmware.
- **Procedure:** after the checklist and fresh Docker/environment capture,
  start one named container with the two approved device mappings and BDF
  visibility above. In the pinned source-matching runtime working directory,
  execute only:

  ```bash
  timeout --signal=KILL 40s build/test/tt_metal/tt_fabric/test_system_health \
    --gtest_filter=Cluster.ReportSystemHealth \
    --gtest_output=xml:out/issue30-runs/ATTEMPT/native/gtest.xml
  ```

  `ATTEMPT` is replaced by the freshly claimed safe attempt ID in G0's frozen
  recipe; output is mounted persistently. Record stdout/stderr separately,
  direct process/container exits, timeout flag and actual argv/environment.
  Inspect reported chip count, peer/channel pairs and warnings, not just XML.
- **Prediction:** if base firmware has trained the cabled ports, discovery
  reports real peers. If PORT_UNKNOWN persists, expect UMD training waits and
  our watchdog to stop it before UMD's much longer wait. If status is settled
  but down, expect missing peer links even if the reporting test exits zero.
  This run is not asserted to initiate physical training.
- **Stop/release:** timeout, any new warning/error, unhealthy/missing/mismatched
  chip, unknown firmware version or no bidirectional connection => stop; B
  withheld. Even a healthy A requires owner review and separate B approval.
  ReportSystemHealth is explicitly non-failing on health errors
  (`T:tests/tt_metal/tt_fabric/system_health/test_system_health.cpp:257-259`).
- **Meaning:** establishes or localizes the discovery part of B1. It neither
  verifies a transfer nor measures B3/B4. No link means the planned #30 stop
  point was reached; #13/#105 remain blocked. Unexpected topology is evidence
  for a revised plan, not permission to force an MGD.

### B — one official 1D fabric smoke invocation

- **Identity:** `issue30-b-fabric1d-smoke-watcher-v1`; a fresh attempt ID and
  separate owner approval after A's evidence. Same fresh verified two cards.
- **Purpose:** one verified chip-to-chip operation using the persistent fabric
  path, **five packets** (the upstream fixed smoke, not a one-packet rewrite).
- **Cap / Watcher:** 60 s total, binary 40 s, container watchdog 50 s, cleanup
  deadline 60 s; `TT_METAL_WATCHER=1` throughout first kernel use. No profiler.
  This cap includes cold JIT and firmware setup; completing inside it is an
  unverified prediction. If compile dominates, stop without extending it.
- **Procedure:** immediately recheck identity/boot/head/quiescence and capture
  before-Docker/environment files. In one named container with G0's approved
  mounts/runtime and the same visibility, execute **only this exact filter**:

  ```bash
  timeout --signal=KILL 40s build/test/tt_metal/tt_fabric/fabric_smoke_tests \
    --gtest_filter=Fabric1DFixture.TestUnicastConnAPI1DSmoke \
    --gtest_output=xml:out/issue30-runs/ATTEMPT/native/gtest.xml
  ```

  Use default fast dispatch (unset `TT_METAL_SLOW_DISPATCH_MODE`), no mock
  cluster or custom MGD environment. The fixture calls FABRIC_1D then opens
  both unit meshes; the test selects an actual forwarding channel, launches
  receiver then sender, waits for both and reads their L1 result regions
  (`T:tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp:36-177`).
  Keep its first available link selection unchanged; record the resolved link
  from available runtime diagnostics, and mark it open if not exposed. Do not
  pretend this tests every cable/lane or both payload directions.
- **Content evidence:** the upstream receiver checks all five PRNG-seeded
  payloads and records mismatch or PASS; host checks sender/receiver PASS and
  byte-count equality. Receiver implementation:
  `T:tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp:42-75,103-114`.
  Accept only one exact, **executed, non-skipped** testcase, no failure/error,
  direct exits zero, no timeout, expected Watcher output without anomalies,
  no new warnings, clean teardown and healthy both-card after-state. Retain
  generated kernel/compile evidence proving checks were not compiled out.
  A skip for no forwarding connection or wrong device count is **not success**.
- **Prediction:** with a discovered connection and successful MGD mapping,
  fabric router handshake should complete, then all five packets should pass
  at receiver and both meshes close. Possible stop phases: mapping/health
  admission, firmware readiness, remote EDM handshake, receiver polling,
  validation or teardown. A timeout does not identify which phase by itself;
  retain last diagnostic/postcode/Watcher evidence and report only what it
  locates. No automatic alternative 2D/legacy run.
- **Stop:** any failed/incomplete/extra/skipped test, missing result, timeout,
  mismatch, new Watcher/link/clock/thermal warning, identity drift, container
  left alive or unhealthy peer => terminate named attempt and stop. Preserve
  all files. No resets/retries are approved.
- **Meaning:** verified receiver content establishes one TT-to-TT fabric
  transfer for B1 and the hardware prerequisite for #13/B4 investigation. It
  does not prove FPGA compatibility, latency tails, a continuous stream,
  absence of context switches, or B3 cost. B3 source conditions are already
  known; B4 remains unmeasured. B5/B6 and #105 stage-3 AC remain separate gates.

### C — not included: timing / retry / recovery

There is **no C execution approval request yet**. After B, owner may commission
B3 timestamp instrumentation/finite-stream experiments (source-study section
4), a B4 round-trip latency test, or investigation of A/B's stop point. Each
needs a new identity, preregistered prediction, <=60 s cap, stop rules and
Watcher on the first use of a new kernel. No custom kernel is justified just
to get the first transfer when the official smoke is available. #105's stage
3 has no implementation-ready AC and cannot start here.

## File and lifecycle capture requirements (G0 must freeze these)

Follow #120's evidence discipline and ADR-0018's no-stdout-result principle;
this paragraph does not change that ADR or its Issue #120 envelope schemas.

- Claim a fresh logical run directory plus collision-safe attempt identity
  before launch; name the container with the attempt ID. No `--rm` loss of
  needed exit/inspect evidence before capture. Never overwrite an old attempt.
- Store `docker-before.txt` / `docker-after.txt` from `docker ps -a` (names,
  states and image, not sensitive full command text), approved environment
  values, image digest, harness commit/clean state, boot identity, both-card
  baseline and source/build/library/kernel hashes. Public records use relative
  companion paths and identity aliases; no machine names/local absolute paths.
- Native stdout/stderr remain separate byte-exact files and are never parsed
  for a JSON/PASS line. GoogleTest writes the fresh `native/gtest.xml` slot.
  Record direct wait return codes and which layer hit its deadline, with no
  inferred successful exit if an outer kill prevented observation.
- XML is accepted only after normal process/container completion and exact
  testcase validation; partial/malformed/missing, skipped or wrong-test XML
  fails closed. It is not an atomic publisher itself. A final attempt-bound
  record is published only after validation using exclusive temporary write,
  flush/sync/close, no-replace atomic rename and directory sync, with hashes
  of XML, raw streams, environment/lifecycle/Watcher companions. Missing final
  publication after a kill is incomplete evidence, never reconstructed success.
  If that publication needs new code, satisfy G0's board-free-test gate first.
- Record start/end/deadline, `started`, `timed_out`, direct exits (nullable when
  unobserved), expected/observed testcase, skip/failure counts, verdict and
  reason, health/Watcher review and B1/B3/B4 scope. XML/exit alone must not
  produce an accepted verdict for A's health or B's post-run health.
- The independent container watchdog must kill **the named container** if the
  Docker client or inner process stalls; killing the client alone can leave
  hardware work running. Budget cleanup before 60 s and verify absence of that
  container, with before/after lists. Failure to establish termination is a
  stop condition for owner recovery, not permission for another workload.
- Capture health of **both** cards afterward under separately approved
  read-only observation. Do not call an attempt complete/healthy if this
  evidence cannot be obtained. Preserve raw diagnostics privately if paths or
  machine names appear; any published redacted derivative is separate and
  labelled, never an edit to original bytes.

Expected morning sequence is physical/firmware review -> G0 receipt -> approval
A (<=60 s plus evidence review) -> approval B (<=60 s plus evidence review), or
stop after A. Preparation and review durations are not board timing estimates.
No board operation, health command, reset or device-open container was run in
preparing this plan.
