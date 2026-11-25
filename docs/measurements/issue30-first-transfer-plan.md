# Issue #30: first chip-to-chip transfer proposal

**Current status (2026-11-24 late night): A observations owner-accepted; one B
is separately authorized, pending the corrected gates, verification and fresh
same-boot preflight. Original proposals remain below; the final owner-decision
section supersedes the earlier admission/Watcher/baseline rules only as stated.**

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

## Discovery observability correction (operator decision, 2026-11-24)

[ADR-0021](../adr/0021-discovery-observability-gate.md) corrects the circular
pre-open ERISC/ETH requirement. The 66caf96 preflight-stop record stays immutable.
The owner renews the unused A-only once scope; the admission correction is the
operator's judgment, not an expansion of owner scope.

Before changing code, the invariants are: A alone permits null ERISC version
and ETH heartbeat with an exact `unobservable` reason citing pinned UMD
firmware_info_provider.cpp:217-229, firmware_utils.cpp:80-99 and
topology_discovery.cpp:302-306, only for bundle <19.9. Bare null/wrong reason,
observed ERISC <1.4.1 or observed heartbeat failure refuse. Identity/bundle,
progressing ARC watchdog, 800 MHz, DRAM, loaded KMD, pages, image/binary/source
hashes, boot, quiet host/zero containers and owner binding remain mandatory.

A retains reported firmware/version diagnostics; missing or first-core-only
output does not certify all cores or heartbeat health. Warnings/refusals stop.
The initial B rule required known health inputs AND per-core inventory/reciprocal
links from A; the late-night owner decision below replaces that circular rule.
Historically it demanded inventory confirmed by accepted A output. The unchanged official report may expose no
per-core inventory: B then remains closed for an owner decision, even when H1/H2
connectivity is accepted. No lack-of-warning inference fills unknown fields.

Before/after ARC sample pairs and temperature are observations, not stable
identity fields; both pairs must show progress, while stable identity/version/
health/policy fields still match. A approvals use an explicitly bound v2 file
recording owner date/scope separately from operator correction. Existing bytes
and native binaries remain unchanged; receipt-v5 requalifies only host changes.

## Frozen A topology hypotheses (owner report, not a hardware observation)

The machine-readable input is [issue30-a-topology.json](issue30-a-topology.json).
It supersedes the earlier exact lane/core-table preparation requirement for A.
These were frozen before A approval/execution. A later reported H1 and the owner
accepted its observations separately; see the final section. No transfer follows
from that observation alone.

**Owner statement:** two cables join physical card a port 1 to card b port 1,
and a port 2 to b port 2, numbering 1..4 in increasing distance from the
motherboard. The owner cannot identify physical a/b with serial ...411c/...4124;
these symmetric same-number connections make that swap immaterial. Cable type
has not been supplied.

**Conversion, not owner testimony:** supplemental vendor documentation calls
port 1 furthest from the motherboard and port 4 nearest. H1 therefore maps
owner 1/2 to vendor 4/3. H2 instead assumes owner and vendor numbering agree.
Pinned `tools/scaleout/board/board.cpp:330-350` supplies channel sets, not the
physical observation or a lane-crossing rule.

| Hypothesis | Same-vendor-port cables | Connected channels on each card | Uncabled channels |
|---|---|---|---|
| H1 (primary) | 4-to-4, 3-to-3 | {4,5,6,7} | {8,9,10,11} |
| H2 (alternative) | 1-to-1, 2-to-2 | {8,9,10,11} | {4,5,6,7} |

Uncertainties are frozen with the input: near/far guidance is outside the pinned
release; riser/extension-board geometry may reverse the intended bracket order;
exact channel/lane pairings are not source-established; two links per cable
depends on the unconfirmed cable type. Fewer links therefore do not by themselves
prove a firmware training defect. Neither hypothesis is called observed fact.

**Invariants:** no lane identity such as 4-to-4 is assigned. Resolve the reported
peer core against that peer card's observed channel/core table. Every UP endpoint
must target the other card, have an exact reciprocal UP endpoint, and preserve
the declared port pair (e.g. either 4-to-4 or 4-to-6 within vendor port 4 is
allowed). All eight QSFP channel records per card must be present once, with
unique local cores, to distinguish explicit DOWN from missing evidence.

After all phase-appropriate execution/XML/identity/firmware/Watcher/diagnostic
gates pass, A's three connectivity outcomes are:

1. **accepted_discovery:** exactly H1 or H2, four undirected links / eight UP
   endpoints, reciprocal and port-consistent. Record which hypothesis and all
   sixteen observed UP/DOWN records. This permits owner consideration of B,
   never automatic B authority or a receiver-content/timing claim.
2. **link_not_established:** zero peers, or a proper reciprocal subset of either
   hypothesis. Preserve the observations and stop; B is withheld. This reaches
   the planned discovery stop point corresponding to the open first-training
   trigger question. It does not establish the cause or prove a host trigger
   cannot exist, and cable type/orientation are still possible explanations.
3. **unexpected_topology:** loopback, one-way/foreign/unknown peer, wrong port
   pair, or connections outside both hypotheses. The source's `connected to
   Unique ID: ...` form is retained as an unresolved peer identity, not guessed
   to be a local chip ID or rejected as an unknown text format. Preserve observations and
   stop; no expectation rewrite, retry, MGD override or B release follows.

Truncated/duplicate/malformed reports and unknown execution/otherwise-required
health are **rejected evidence**, not a fourth physical connectivity inference.
Reasoned unobservable ERISC/ETH values are the A-only exception above; missing
A output remains unknown, not healthy. Under the later owner decision, B binds
accepted A link observations rather than a missing firmware inventory. Raw streams
are retained on every path. Only the source-shaped DOWN warning for a matching
reported chip/PCI/channel/core is eligible for topology classification; unrelated
warnings and nonzero retrains still refuse. B recomputes A's observed topology
before device mappings and again at judgment, rather than trusting a label.

The new input is `before.topology` equal to the full frozen JSON, with
`links=[]` and `uncabled=[]` (hypothesis-dependent, not asserted absent). No exact
core table is in that input. Hardware CLI admission requires this policy; old
exact-table fixtures remain historical CPU compatibility tests only. The current
experimental verdict is `issue30-verdict-v3`; `topology`, `observed_qsfp` and
`firmware_observation` accompany immutable raw files and hashes. Missing native
XML is explicitly flagged, not replaced by a fabricated successful file. Production contracts,
transport atomicity, separate owner approvals and the 40/50/60 s caps do not change.

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
- **Stop/release:** use the frozen H1/H2 three-way connectivity rules above.
  Timeout, invalid or otherwise-required missing health/evidence, new non-topology diagnostics,
  partial/no links or unexpected topology => stop; B withheld. Only matching
  H1/H2 with all other gates can be accepted, still requiring owner review and
  separate B approval.
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


## Owner decision and single B procedure (2026-11-24 late night)

This is the current precommitted authority/rule revision. The owner accepts only
[A observations](2026-11-24-issue30-a-owner-acceptance/README.md), preserving the
original rejected/watcher result, and separately authorizes one unchanged B.
[A NUMA lines13-16](2026-11-24-issue30-a-owner-acceptance/baseline-match.json)
match the #176/#120 known source text/locations; no unmatched line is accepted.
Cluster-only A has no application Watcher output requirement. B does.

### Frozen invariants and admission

- No second A or fresh link probe. Bind the canonical accepted-a.json (and
  original evidence fingerprints), same boot/stable environment, unchanged
  symmetric cabling, full reciprocal H1 channels4-7 and retrain0. Channels8-11
  remain explicitly DOWN. Recompute that frozen graph, not a verdict label;
  do not represent those historical links as newly observed before B. Boot
  drift refuses before device mapping. B approval-v3 binds this accepted-A hash
  and the one-use physical scope separately from the initial operator fix.
- Reasoned pre-19.9 ERISC/ETH unknowns stay unknown, not healthy. No per-core
  inventory is invented or required from this diagnostic. Known low versions,
  known failed heartbeat, bare null or wrong reason still refuse. All identity,
  ARC progression,800MHz,DRAM,KMD,pages,source/image/binary pins/quiet/container
  gates remain. Fresh telemetry is allowed, not native discovery repetition.
- Exact #176 conditions only: NUMA membind and adjacent same-device fallback,
  known-motherboard bus fallback, dispatch telemetry SMC unavailable. Same #120
  literal/logger/source-location matching; only enumerated device/chip,PID/TID,
  bus and timestamp values vary. NodeSet/channel/board/error/location do not.
  The eight A DOWN summaries may appear only for A's exact DOWN chip/PCI/channel/
  core. Any other warning,error,fatal,critical,assert or traceback stops.
- B must produce nonempty fresh Watcher dumps in its new capture directory,
  included in execution's artifact manifest, exact review hash and clean manual/
  bounded scan. Missing/uncaptured/changed-after-capture/unreviewed/abnormal output
  refuses. Capture has no permission to create a clean-log surrogate.
- Keep current pinned source/image, official binary and runtime kernel sources
  byte-exact; no BENCHMARK_MODE, custom MGD, reliability/dispatch override or
  legacy/2D fallback. One binary40s, named-container50s, cleanup60s, no extension.

### Prediction and evidence checkpoints (before B)

Prediction: strict1D fabric setup creates the logical mapping/routing planes,
credit/buffer structures and EDM handshake, worker kernels are JIT-built, receiver
starts before sender, five PRNG packets pass receiver content checks, sender and
receiver status PASS/byte counts agree, all programs finish and devices close.
Cold JIT can consume the entire40s binary budget before transfer: do not extend
or retry, record the furthest observed checkpoint and missing evidence.

| Checkpoint | Available observation / limit |
|---|---|
| Driver/topology/fixture entry | UMD constructor/discovery/starting-devices log lines; `Running fabric api tests with fast dispatch`; `Fabric Reliability Mode: STRICT_SYSTEM_HEALTH_SETUP_MODE` before mesh creation. These alone do not prove routers/handshake finished. |
| Physical/logical fabric configuration | Topology/mesh/route informational lines actually emitted; retain original warnings. No added debug workload or descriptor is used. Absence of detailed output leaves substage unknown. |
| JIT / kernel attachment | BuildKernels firmware/cache/compile telemetry when emitted; Watcher attached-kernel listings and dump waypoints. JIT cache summary at close is retrospective, not a start timestamp. No compulsory fabricated JIT milestone. |
| RX/TX / packet content | Exact `[ RUN ]` / `[ OK ]` / one-test PASS plus non-skipped failure-free XML at the bound official selector; source `test_basic_fabric_smoke.cpp:74-168` fixes five packets, RX-before-TX, reads both status arrays and checks status PASS/equal byte counts; `tt_fabric_1d_rx.cpp:49-71,101-114` polls PRNG data, checks each packet, then writes status. Watcher FPW/FPD or other actual waypoints are retained if present. |
| Close / process/container | Fixture close happens in suite teardown; actual close/destructor log lines, direct process/client exit, termination and zero remaining containers within60s. XML test-body duration is not whole-capture time. |

The unmodified official host test does **not** print per-packet bytes, PRNG seed,
status words or byte counters on success. Do not invent those values. Conditional
content acceptance is source-bound official receiver/status assertions plus exact
completed XML, frozen nonbenchmark kernel/build provenance, fresh clean Watcher,
warning classification, normal lifecycle and unchanged post-state—not a bare
PASS string or exit0. A failure/timeout may never reach packet assertions; record
content as unverified, even if preceding fabric logs look normal.

### Exactly one B — approved identity and operational order

1. Complete board-free changes, RED/GREEN, pushed clean head, full tests/ruff,
   exact-head all CI and public Sonar0/0/OK. No finding acceptance/muting.
2. Fresh same-boot telemetry for both serial/BDF pairs, bundle19.6.0.0, advancing
   ARC,800MHz,DRAM/temperature,loaded KMD2.12.0,1GiB pool/free>=2 and mount,
   exact image/binary/source hashes, zero containers and quiet host. Inspect
   existing hugetlbfs file metadata/allocation read-only to explain A's8->6 free
   pages; never map/read file payload, delete files or configure/reclaim pages.
   Stop without B if any prerequisite fails, especially boot drift or space.
3. Bind approval-v3 to one attempt derived from
   `issue30-b-fabric1d-smoke-watcher-v1`, current head/boot/environment/receipt,
   owner2026-11-24 one B, accepted-A digest and unchanged cabling. No new A.
   Recheck quiet host/containers/counters/clock immediately before launch.
4. Execute only Fabric1DFixture.TestUnicastConnAPI1DSmoke in the named container,
   both approved mappings/BDF visibility, Watcher1 and40/50/60s. Do no heavy host
   test/build while it runs. No fallback/retry/reset/firmware/cable action.
5. Verify container termination/absence, then both-card telemetry. If a card
   fails to respond, telemetry is unavailable, or cleanup is uncertain: preserve
   received evidence, perform no reset/additional intervention, stop and report.
   Never continue trying alternative health/device commands after that failure.
6. Use committed rules and immutable stdout/stderr/XML/Watcher/artifact hashes.
   Stop on skip/failure/mismatch/nonbaseline diagnostics/timeout/uncertainty or
   abnormal card state. Commit/push the outcome, exact-head CI/Sonar; update
   design/open-issues only for actual B facts. No timing/loopback authority.

B1 may gain a content-checked official transfer or a precise bounded stop point.
B3 routing-call jitter, B4 physical latency, #13 loopback and #105 stage3 remain
separate experiments; none is licensed by B success. #105 still requires B5/B6
resolution and prewritten stage3 acceptance criteria. FPGA peer compatibility is
not established by TT-to-TT traffic.

## After the single B: source-only failure analysis and morning proposals

The [B record](2026-11-24-issue30-b-smoke/README.md) preserves the completed,
rejected RX-JIT stop. The owner has powered off the cards; no board/tt-smi/device
access is permitted. The [failure-analysis plan](issue30-b-failure-analysis-plan.md)
separates observed stages from source-control-flow inference, compares fixture
requirements with actual1x2/two-unit meshes/two PCIe-local cards/four links,
explains why this failed rather than skipped, and provides bounded separate
morning approval units. It proposes include/export-layout qualification, not a
firmware/reset remedy for a host compilation/filesystem error. Original A/B
records and current one-use authority are not rewritten or renewed.

## B v2 layout prediction and morning combined-unit proposal (#196)

[Device-free layout/checker evidence](issue196/README.md) prepares
`issue30-b-fabric1d-smoke-watcher-v2`, not a new hardware grant. The runtime root
is a fresh attempt-owned view of audited pinned image assets plus unchanged
pinned tests; normal JIT root-relative includes and generated/fabric output now
share that supported root. No per-kernel include patch, edited official source,
ELF/library/image replacement, extra capability or writable image root.

### Prediction and precommitted judgments

- A fresh approved A observes the two named cards and full reciprocal H1
  channels4-7/retrain0,8-11 explicitly DOWN, while stable state matches. A is
  cluster-only, application Watcher output N/A. No invented ERISC/ETH health.
- Given accepted fresh-A handling and **new** owner approval, B v2 runs exactly
  Fabric1DFixture.TestUnicastConnAPI1DSmoke once, strictFABRIC_1D/fast dispatch,
  Watcher1, source-pinned runtime, named container and40/50/60s. Expected:
  normal routing/credit/router handshake setup, all JIT compile, RX-before-TX
  dispatch,5 PRNG-content-checked packets and orderly device/container close.
- Milestones remain actual UMD/fixture/fabric-init/JIT logs, attached kernel maps/
  Watcher dumps/waypoints, exact non-skipped failure-free XML, source-bound RX
  content/status assertions, original warning/error text and both-card post-state.
  The official test prints no per-packet payload/status/byte counters on success;
  no value is manufactured. PASS/exit alone is not acceptance.
- Cold JIT may still hit40s: BuildKernels/compiler progress or incomplete XML/
  absent completion identifies the last observed stage. No extension/retry.
  Router credit/handshake problems may stop after initialization logs with
  Watcher ERISC/worker wait waypoints or sync-timeout diagnostics; no absent
  message is treated as proof of success. Payload mismatch should appear as
  failed receiver/status/byte assertions/XML failure, not silently accepted.
- Stop on skip/failure/mismatch, any non-#176 warning/error (including any repeat
  of readonly export warnings), missing/abnormal/unreviewed B Watcher, timeout,
  incomplete cleanup/card anomaly. No fallback/legacy/2D/reset/firmware/cable/
  host setting action. Bv1 evidence/verdict is immutable regardless of v2.

### Cold-JIT estimate / cache proposal, not a cap change

The device-free checker measures all24 object-compilation cases serially, with
normal HAL flags, root layout and saved argument specializations. Initial pass
13.17s, successor with dependency/export checks13.48s total on this host. RX/TX
were0.28/0.34s in the initial pass;16 router compilations dominate. This is not
an ELF-link/driver/handshake/traffic measurement or a guaranteed runtime bound.
Bv1's complete fixture setup was4.355s including router setup and the failed RX.
Base firmware has a separate build/precompiled path; its possible recompilation
and full RISC-V ELF linking are **OPEN**, outside the24 registered kernel object
cases. The saved B cache contains10 firmware objects, not proof of a future cache
hit. A conservative rough sum of observed Bv1 setup and serial compile is about18s
(double-counting some router compile), leaving about22s of40 for unmeasured
link/configuration/dispatch/close; **40s appears feasible, not guaranteed**.
No cap increase is currently proposed. If future full cold-JIT qualification
shows it cannot fit, request an owner decision instead of changing the limit.

TT_METAL_CACHE is the normal JIT cache root (rtoptions.cpp:421-428), now always
attempt-owned. Source build.cpp:115-123 also supports compiler ccache via
TT_METAL_CCACHE_KERNEL_SUPPORT, but neither sharing/warming nor an installed
ccache executable is assumed. Offline objects are not drop-in future JIT cache
entries: new boot/configuration/root/key/dependency differences can invalidate
them. A separately qualified, owner-reviewed prewarm within the *same* attempt
and exact native cache protocol is only a future proposal. No copied/warm cache
is used to claim cold success or silently buy more time for B v2.

### One-screen morning approval request (proposal only)

- **Unit:** one new A identity, then `issue30-b-fabric1d-smoke-watcher-v2` once
  **only if** its fresh observations pass the owner-approved conditional rule.
  Two cards: serial...411c/PCI0000:06:00.0 and...4124/PCI0000:09:00.0; same
  symmetric cabling, default strict1D, no third device, no physical intervention.
- **Owner decisions needed first:** approve this combined conditional scope and
  how new A becomes B's accepted evidence. Old A is invalid after card power-off
  even if the host UUID remains unchanged; Bv1 scope is consumed. The pending
  morning-admission registry is false and must not be treated as authority or
  bypassed. Qualify the owner-selected conditional fresh-A binding before either
  run; do not improvise authorization between A and B. Bind renewed scope/fresh-A
  evidence through the normal receipt path.
- **Prerequisites:** verified cold offline compile/export/list/absence proof,
  clean/pushed/tested head, all CI/Sonar0/0/OK, fresh identities/boot-power epoch/
  bundle/ARC800/DRAM/KMD/pages/quiet host/zero containers and unchanged hashes.
- **Duration:** each A/B process<=40s, container<=50s, cleanup<=60s; combined
  process/cleanup windows<=120s, plus bounded telemetry/preparation (roughly2-3min
  total allocation, not a new larger single-run cap). A historical<1s; B cold
  compile estimate above. No heavy host tests/builds while either device run.
- **Risk/stop:** both-card runtime/router/worker/credit/NoC/JIT hangs, unknown base
  ERISC/ETH health, allocation/temperature and cleanup uncertainty. A deviation
  withholds B; any B stop withholds retry. After-state/container proof first,
  unresponsive telemetry means stop without more operations/reset. New authority
  is required for any retry, fallback, reset, cable or firmware maintenance.

## Executable renewed scope — owner 2026-11-25

Owner grants A once then Bv2 once only if fresh A matches H1. No run is implied
by preparation. A requires both named cards, channel4-7 reciprocal UP/retrain0,
8-11 DOWN/unconnected only. Exact #176 NUMA/motherboard/SMC baseline diagnostics
are permitted, no other warning/error; A's cluster-only Watcher output is N/A.
Any H2/partial/new peer/unknown identity/retrain/skip/failure stops before B.
B admission revalidates fresh canonical A native evidence, approval-v4 and stable
baseline (same boot, same cabling policy, same unit/head/receipt), never old A.
Each approval-v4 records owner2026-11-25, one phase invocation and the two named
attempt IDs. Existing pending registry and consumed old scopes grant nothing.
Immediately before each phase: zero containers/quiet host; before B also enough
free1GiB pages. Each phase retains40s process/50s container/60s cleanup; total
process-cleanup windows<=120s plus telemetry/preparation. No heavy tests/builds
while either hardware run. After each: both-card telemetry and container0 first;
unreadable telemetry/unresponsive card/uncertain cleanup means stop, no extra
operation/reset. Source-derived roughly18s cold B expectation is not a bound.
B requires original five PRNG/status/byte checks, complete XML, packet-evidence
markers and fresh reviewed Watcher, not exit/PASS alone. Previous precommitted
stage diagnostics/timeout/content failure rules remain; no retry/fallback/cap
increase/firmware/cable/host change. Fresh A and Bv2 evidence are new records;
11/24 A and11/25 Bv1 evidence/verdicts remain byte-exact.

### Owner DRAM admission correction — 2026-11-25 continuation

The13:43:22Z card...4124 false/raw0x0 then true/0x5555 observation and stop
remain immutable, cause unknown. The operator reports three subsequent read-only
samples around23:05JST (~2s spacing), both cards DRAMtrue/800MHz/ARC advancing;
no raw snapshots survive, so these are operator-reported, NOT native evidence
and NOT any of the required five admission samples. Owner explicitly continues
the SAME unused one-A/conditional-one-Bv2 scope, not a reset/retry grant.

For this renewed owner_unit only, immediately before EACH phase require five
consecutive raw telemetry snapshots at least5s apart. Every card in every sample
must report DRAMtrue/rawDDR_STATUS0x5555, zero GDDR_UNCORR_ERRS,800MHz and advancing
ARC. Keep all five raw snapshots plus monotonic times/content hashes. A missing/
failed/abnormal read, short interval or stopped ARC withholds that phase; stop
without restarting the series, resetting or consuming a native invocation.
After each executed phase also collect multiple raw observations with the same
five-reading protocol; any anomaly/change is retained and withholds the next
phase. No telemetry after an unreadable/unresponsive card or uncertain cleanup.
Snapshots/ARC/temperature are dynamic evidence, not physical identity/cabling;
exclude only the series itself from stable A/B baseline, retain it in exact
approval environment/capture/review fingerprints. No past health is rewritten.
All original unit limits, cleanup, warning, content and Watcher rules remain.
Source-bound successor receipt/tests/CI/Sonar must pass BEFORE fresh sampling
and native execution. Operator report alone cannot satisfy the five-read gate.
