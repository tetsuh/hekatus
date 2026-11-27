# Issue #30 B failure analysis and morning proposals — source/output only

**No hardware authority. Boards are off.** This analysis reads only pinned source
and the completed B artifacts. No device, tt-smi, board probe, compiler/container
experiment, reset or firmware operation was performed for this analysis.

Evidence: [B record](2026-11-24-issue30-b-smoke/README.md), whose raw streams,
XML, Watcher and inspector YAML remain immutable. `T:` below means tt-metal
v0.75.0 commit `d9a68815f5fcf08a5bfbffb6f1f811823fba8edd`; `U:` means its UMD
commit `bc77ac2f7382ffeb9fa5430055f0854d53979489`. Runtime image/digests, harness
`ebf5a82`, device identities and boot fingerprints are in that record.

## 1. Stage localization: observation is not inference

| Stage | Confirmed in saved output | Source/control-flow inference and limits |
|---|---|---|
| Device open | Both local chips/PCIe IDs0,1, no remote chip IDs, bundle19.6.0, KMD2.12.0, cluster start completed; inspector initialized two unit meshes | Not a device-open failure. Two independently PCIe-local cards are permitted; no source requirement for one remote/non-MMIO chip exists in this fixture. |
| Logical/physical mapping | Degree1 on both2-chip logical/physical graphs; inspector system mesh shape[1,2], children[1,1] on0 and1 | Actual auto-discovery path, not proof the static P150_X2 descriptor was selected. `T:tt_metal/impl/context/metal_env.cpp:391-423,458-479` selects no-descriptor auto-discovery for one host; native output explicitly says so. |
| Routing table generation/write | Control-plane constructor returns and later FABRIC_1D initialized on2 devices | `T:tt_metal/fabric/control_plane.cpp:597-622` catches export failures then creates routing-table generator. `T:tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:295-313` writes routing tables before compiling/configuring fabric; `T:tt_metal/fabric/control_plane.cpp:2068-2087` iterates user chips and writes route/connection/telemetry information. Thus the write call returned, inferred from source plus the later log. No raw routing-table readback/content verification was captured. |
| Fabric router launch / sync | FABRIC_1D initialized, both devices, Watcher active ERISC router IDs/K or NWID samples; test body later starts | Log alone precedes configure/sync. Stronger source inference: `T:tt_metal/impl/device/device_manager.cpp:468-487` subsequently invokes fabric.configure; `T:tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:324-335,447-531` polls initialized routers per MMIO device (including independently local devices), default10s timeout; builder expects LOCAL_HANDSHAKE_COMPLETE then writes READY_FOR_TRAFFIC (`T:tt_metal/fabric/fabric_builder_context.cpp:211-217`). Reaching the body without that exception supports sync having returned on this path. It is not a separate CSR trace or independent proof for every lane/credit. |
| EDM handshake / credits | No separate printed per-router handshake status/credit dump | `T:tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:3519-3603` orders local-ERISC setup barrier, sender/receiver remote handshake, local coordination, LOCAL_HANDSHAKE_COMPLETE and host ready signal. The sync return is indirect evidence of these gates, not an observed application-worker-to-EDM connection or sustained credit behavior. Preserve this as inference. |
| Application RX/TX | Exact test starts; RX BRISC compiler error; XML1 failure, body47ms | `T:tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp:129-143` constructs RX and submits it before TX; fixture fast path enqueues mesh workload (`T:tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:135-168`). Compiler stack names RX. No completed RX enqueue/TX application dispatch/content assertions. Registration in kernel_names.txt does not establish dispatch. |
| Packet verification | No PRNG result/status/byte counters printed, one failed testcase | Five expected packets at source:74; receiver checks at `T:tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp:49-71,101-114`; host status/equal bytes at smoke:146-173. Those assertions were not reached; checked/received counts remain unverified, not invented zero or five. |
| Close | Watcher detaches both, cluster close/destructor completes, native/client1, terminated, zero containers, both after telemetry responses | Exception was caught by GoogleTest; fixture teardown closes devices (`T:tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:120-128`). No timeout/reset/retry needed. No claim that all unobserved internal firmware state is healthy. |

## 2. Direct failure and cause candidates

### C1 — runtime/kernel-source root split: primary confirmed blocker

**Confirmed:** RX source line8 requests
`tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp`; the emitted
BRISC command fails `fatal error: ... No such file or directory`. The saved
command searches generated build dirs, runtime-wheel roots and the RX kernel's
directory, but not the pinned source root. Header exists in the pinned checkout
(279 bytes, SHA256
`e784e6dcb9f496745c9c732e5ae036edaa18b7dbb4a08054cebb207b02db4313`). No board was
needed to check its tracked bytes.

Relevant conditions, not guessed compiler semantics:

- `T:tt_metal/impl/kernels/kernel.cpp:43-91` searches CWD/kernel path/system/root
  for the **kernel file**; `T:tt_metal/llrt/rtoptions.cpp:430-437` sets that source
  search path. It is not automatically the compiler include root.
- `T:tt_metal/impl/kernels/kernel.cpp:340-357` adds only the resolved kernel
  parent plus explicit compiler_include_paths. The official RX config at
  smoke:127-134 supplies no extra compiler include path; the API field exists
  at `T:tt_metal/api/tt-metalium/kernel_types.hpp:80-85`.
- `T:tt_metal/jit_build/build.cpp:102-110,300-318,570-577` bases normal `-I`
  roots on runtime root and appends those per-kernel paths. This exactly agrees
  with the captured command. `T:tt_metal/jit_build/build.cpp:59-68,589-598`
  logs compiler failure and throws; `T:tt_metal/impl/program/program.cpp:227-234`
  wraps it as failed-to-generate-binaries, explaining the observed exception.

**Inference / corrective candidate:** source-file discovery succeeded, but
placing the tests tree only under the kernel-search role is insufficient for
its root-relative includes. Stage the exact pinned tests subtree into a
compiler-visible runtime-root role, or a qualified root overlay, without
modifying the official fixture/kernel. Simply setting a new CWD does not make
it a GCC include root once compile_one changes into its build directory.
Changing TT_METAL_KERNEL_PATH alone repeats the same deficiency. Arbitrarily
changing the entire runtime root risks mismatching wheel firmware/LLK assets.

**Board-free checks next (#196, not executed here):** copy the saved failed build
inputs/generated headers to a new role-mapped workspace, replay the exact emitted
compiler/dependency command with device nodes absent and discovery masked; RED
must reproduce the missing include. Qualify proposed layout by compiling both
RX and TX with their unchanged source/flags and complete dependency manifests;
assert no BENCHMARK_MODE and no library/image/kernel-byte substitution. A first
header success alone does not prove all nested includes or the whole JIT succeed.

**Smallest later hardware check:** the same official five-packet smoke once,
after board-free include/export qualification and fresh owner-approved link
baseline on the new boot (unit M2 below),40/50/60s. It jointly tests C1/C2; do not
spend extra discovery/diagnostic runs for an already explicit filesystem error.

### C2 — generated exports target readonly runtime root: confirmed separate stop

**Confirmed messages:** two `Failed to export ... Read-only file system` warnings
for physical-chip/ASIC-to-fabric mappings at control_plane.cpp:605/614. Source
`T:tt_metal/fabric/control_plane.cpp:597-614` constructs both destinations under
runtime root's generated/fabric, independent of TT_METAL_LOGS_PATH. Serializer
`T:tt_metal/fabric/fabric_host_utils.cpp:181-185,221-225` creates parent directories.
The immutable request uses a readonly rootfs, readonly source/assets, writable
capture/log/cache roles; it has no writable role at that destination.

**Inference:** missing writable export staging explains these exact warnings.
They are metadata/visualization export exceptions caught before routing-table
setup, not the direct RX compilation exception. They cannot be dismissed as
#176 baseline, nor blamed on bad routing just because export failed.

**Board-free checks next:** read the effective saved mount/environment layout;
qualify a narrow generated/fabric writable staging role while keeping rootfs
readonly/capabilities dropped/network-none/non-root. Verify parent creation and
YAML export in an isolated no-device environment, separate from any device test.
No blanket writable root, extra privilege, warning waiver or host configuration.
Exact bind/overlay assembly still needs tests; this document does not claim it
was implemented or validated.

**Smallest later hardware check:** unit M2, same once-only official smoke with
qualified include/export layout. Expected new warnings absent; any new warning
still stops, even if packet assertions otherwise pass.

### C3 — wrong device/mesh/fabric/link assumptions: checked, not current cause

**Confirmed in source/output:** fixture requirements and actual configuration
are compared below; actual system[1,2]/unit[1,1] meshes,2 chips, strictFABRIC_1D,
nonempty forwarding path and reported initialization fit the fixture. No mesh
count/no-link skip occurred. These facts refute treating two PCIe-local cards or
1x2 coordinates as inherently unsupported by the chosen smoke fixture.

**Unsettled:** route contents, per-channel service/credit health and application
transfer after corrected JIT. Neither adjacency nor router sync is five-packet
payload proof. A later link loss/new boot could invalidate today's observations.

**Board-free checks:** inspect frozen graph/inspector/kernel map, unchanged
fixture, descriptor and emitted mode definitions. Do not replace discovery with
a fabricated MGD or infer that static descriptor selection occurred. **Hardware:**
M1 on the new boot only with separate approval, then M2; no legacy/2D fallback.

### C4 — insufficient/incompatible base firmware: unknown, unsupported as cause

**Confirmed:** bundle19.6.0.0, per-core ERISC and independent ETH heartbeat
unknown; router setup reported/Watcher active, no firmware-version warning.
Source UMD minimum is1.4.1 (`U:device/api/umd/device/firmware/erisc_firmware.hpp:19-22`),
version reads are per core (`U:device/firmware/firmware_utils.cpp:80-99`) and
verification/mismatch handling is `U:device/topology/topology_discovery.cpp:627-663`.
Absence of warnings is not an inventory or health proof. Bundle version and ERISC
version are different version domains. Pre19.9 zero telemetry means unavailable,
not failed links; source-study prerequisites remain necessary, not sufficient.

**Inference:** firmware cannot cause the observed host preprocessor to lack its
include root or create a directory on a readonly rootfs. It is therefore not the
supported explanation for this failure. Installed guidance19.8.1 versus observed
19.6 remains a compatibility caution, not a demonstrated need to flash. Directed
retraining firmware requirements concern an operation not used/authorized here.

**Board-free checks:** compare already pinned UMD constants/install guidance,
fixture router-sync requirements and observed source-defined status progress.
**Hardware if later specifically needed:** optional M3 version-read proposal
below, not a reset/flash or implicit extra A. Unknown health remains unknown.

## 3. Fixture assumptions vs this two-card configuration

| Requirement / source | Current evidence / difference |
|---|---|
| Base SetUp needs>=2 available devices; body requires exactly2 (`T:tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:69-76`; smoke:43-46) | Exactly2 approved local cards visible, all available IDs opened; no third card. Both have their own MMIO association; fixture does not require a remote chip. |
| `create_unit_meshes` of all IDs,1 hardware CQ and default dispatch config (fixture:103-117) | Inspector actual root mesh[1,2] devices[0,1], child meshes[1,1] on[0] and[1], both initialized. Physical shape1x2 is not Fabric2D mode. |
| Explicit FABRIC_1D (fixture:190-194), default strict reliability (fixture:90-100), no required slow-dispatch for this fixture | Output fast dispatch, STRICT_SYSTEM_HEALTH_SETUP_MODE, FabricConfig::FABRIC_1D. ControlPlaneFixture's separate slow-dispatch precondition is not this fixture. No override/fallback. |
| At least one forwarding ETH channel before packet kernels (smoke:65-70); one selected forwarding port,5 packets at:74 | A has4 reciprocal links/channels4-7, zero retrains; B passes the nonempty guard and reaches RX JIT. No exact four-link requirement in the smoke body. It does not use every link for these5 packets. |
| P150_X2 supported classifier (`T:tt_metal/llrt/tt_cluster.cpp:182-187`) and reference descriptor (`T:tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto:3-12`) | Reference device shape[1,2], host[1,1], channels4 RELAXED matches this hardware count/connectivity. Actual one-host path logged auto-discovery/no MGD (`T:tt_metal/impl/context/metal_env.cpp:404-423,458-479`), so do not claim the reference file was loaded. |
| Runtime router/worker RISC-V compilation, ready/sync status, nonbenchmark PRNG RX correctness | Router IDs/dumps and init returned; RX worker JIT fails filesystem include lookup before it can execute. Per-core base firmware version still unknown, not supplied by bundle number. |

**Why failed rather than skipped:**>=2/exact2 and nonempty-forwarding guards
were passed; code reached the receiver RunProgramNonblocking, compiler failed,
TT_THROW became a GoogleTest exception/failure. XML says executed/completed,
skipped0/failures1. It is not a topology skip converted to PASS, nor a payload
mismatch exception. Both readonly export warnings would independently stop.

## 4. Hugepage result (already collected, no new host/device inspection)

Saved before-B stat lists two regular UMD backing files, device_1_tenstorrent and
tenstorrent, each1GiB logical **and allocated**. Their before-B mtimes are in A's
native interval14:17:41.259/.318UTC. UMD naming is
`U:device/hugepage.cpp:162-196`. Total allocated2GiB exactly accounts for A's
8->6 free1GiB pages; B starts/ends6 free of8 total. Directory libvirt allocates0.
No payload was read/mapped, file removed or page/host setting changed. This
explains retained allocations; it is not evidence of a page leak or permission
to clean them up. Fresh availability must be observed again after future boot.

## 5. Morning proposals — each is a separate, one-screen approval unit

**All below are proposals, not authority. Current B once-only scope is consumed;
owner has powered off the cards. No test is run while off.** Board-free #196
qualification and clean/pushed/tested/CI/Sonar0/0/OK must precede new hardware.

### G1 — device-free include/export layout qualification (no board approval)

- Purpose: discriminate C1/C2 without hardware; exact failed dependency replay
  in a new workspace and new narrow staged layout, official sources unchanged.
- Operations: source/dependency/mount manifest; failing RX preprocessor/compiler
  control; corrected RX/TX compile/dependency controls; isolated export-role
  creation/serialization check. No native device fixture or device nodes.
- Per command cap40s/container50s/cleanup60s; separate immutable artifact IDs
  per control, no timing extension if the host/toolchain needs longer.
- Prediction: old layout repeats include/readonly failure; proposed layout finds
  all pinned test headers and writes only generated output, with no benchmark/
  privilege/rootfs/library substitution. Failure stays board-free, blocks M2.
- Deliverable: #196 RED/GREEN, hashes/command provenance/tests, owner review of
  exact future runtime-layout diff. This analysis does not implement G1.

### M0 — power-on and fresh read-only host/card preflight (owner-operated)

- Scope: owner powers both cards, then explicit telemetry-only approval for the
  two named identities, ARC pairs/800MHz/DRAM/thermal/KMD/boot/pages, image/hash/
  no-container/quiet checks; no test/device workload or configuration writes.
- Budget: collection attempt<=40s with<=60s total bound; timeout/unavailable
  telemetry means stop without retry/reset/alternative probes.
- Prediction: identities match and observed ARC/clock/DRAM conditions hold, enough
  1GiB pages; new boot invalidates old accepted-A binding. No claim of link state
  from pre19.9 zero telemetry. Owner decides whether M1 is next.

### M1 — fresh official A on the new boot (new separate approval)

- Purpose: refresh physical peer/link observation, not packet traffic or C1/C2
  diagnosis. New identity; unchanged Cluster.ReportSystemHealth once, both cards.
- Scope:40/50/60s named container, no application kernel; Watcher output N/A.
  Frozen expected H1 table and exact #176 diagnostics, no forced link/firmware/
  cable/reset/host operations. Full before/after and cleanup required.
- Prediction: same reciprocal4 links/retrain0; deviations/partial/one-way/
  unexpected warnings or state failure stop. Preserve actual output, not reuse
  today's boot. Owner accepts new observations in a new record if appropriate.
- Risk: device initialization/UMD allocation; no inference of unknown ERISC
  health. Not automatically licensed by power-on or current A acceptance.

### M2 — one corrected-layout official B-like5-packet smoke (new approval)

- Prerequisites: G1 qualified, M0 and new accepted link evidence at same boot/
  unchanged cabling, new source-bound receipt/runtime layout, fresh identity/
  owner approval/40/50/60s, default strictFABRIC_1D and Watcher1.
- Operation: unchanged exact1D selector once on both cards, no fallback. Retain
  actual mapping exports, all JIT logs/Watcher/XML/status evidence and after-state.
- Prediction for C1/C2: no missing test header or readonly-export warning; RX/TX
  compile and5 PRNG packets validate under original assertions, normal close.
  If it fails later, the new stage is useful evidence, never permission to retry.
- Stop: any skip/failure/content mismatch/nonbaseline warning/error/timeout/
  unclear cleanup/card anomaly. Cold JIT can still hit40s; no extension.
- Risk: router/worker/credit/NoC hangs, memory placement/performance baseline,
  unknown base firmware health. No measurements of B3/B4 yet; no reset authorized.

### M3 — conditional per-core version read (only if separately justified)

- Trigger: explicit future version mismatch/sync diagnostic or an owner request
  for inventory, not today's missing include. Not on the critical C1/C2 path.
- Scope proposal: bounded source-pinned UMD read-only get_eth_fw_version on
  channels4-11 of each approved card, once, no kernel/packet/retraining. Precommit
  exact getter/core mapping/output and mock tests before seeking approval.
- Budget: read phase<=20s, named-container/cleanup<=50/60s; no retry/reset.
- Prediction: per-core version inventory or explicit unknown/error. It does not
  independently establish ETH heartbeat health; any intervention needs new scope.
- Risk: device open/CSR read can block; no raw guessed addresses/custom writes.

### Later B3 / B4 / #13 (not ready for hardware authorization)

After a verified M2 payload, separately precommit a fixed-size continuous-traffic
B3 instrument (routing counts/interval tails, baseline/control, Watcher), a B4
paired card-clock/clock-calibration and packet-latency experiment, or #13 physical
same-card loopback wiring/test. Each needs its own bounded workload/sample budget,
RED/tests/owner approval and40/50/60s stop/cleanup plan. These are not automatic
continuations. #105 stage3 still requires B5/B6 and its own written AC.

### Reset / firmware update: not proposed for this failure

- Rationale: neither can fix a host compiler search root or readonly export path.
  Normal close/after responsiveness gave no recovery need. No evidence identifies
  an incompatible per-core version, so no flash or directed-retrain candidate is
  justified. TT-Topology does not support these Blackhole PCIe cards.
- If a future card is unresponsive, stop without probing further and ask the
  owner for a **separate** recovery plan specifying affected card(s), shared-PCIe/
  link impact, supported reset mechanism, one action, post-reset boot/link
  invalidation and bounded health verification. Do not invent a reset command.
- A future firmware-maintenance proposal must separately pin supported bundle/
  tool/rollback, confirm both-card compatibility, stable power and operator method,
  explain irreversible/brick/peer-link risks and expected duration. Flash time is
  not known to fit60s; it is not slipped into any<=60s diagnostic run above.
