# Issue #30 G0: device-free preparation

No real board/device node, tt-smi, reset, firmware update or host setting was
accessed/changed. Python imports, CPU tests and compilation do not initialize
TT runtime. Explicit isolated absence probes enter initialization with no
visible chips and fail there; they never open an actual device.

**G0 build/loader/list/absence passed:** both official binaries linked, both
required selectors listed with eager symbol binding, and both rooted absence
probes exited 1 at `No chips detected in the cluster` (T tt_cluster.cpp:122),
with empty local chip/PCIe enumeration. The wrapper returned 2, not acceptance.
See [build receipt](build-receipt.json) / [G0 receipt](receipt.json). No hardware
correctness, dispatch, link training or jitter result is claimed. All 42 local
G0 unit cases pass; full-suite/CI/Sonar evidence is recorded in the PR/report.
CodeRabbit CLI could not authenticate at review time; no review was completed.
The [source study](../issue30-source-study.md) pins T/U and the runtime image.
The final A/B rules below supersede the preparation gaps in the original plan;
neither they nor a passing G0 receipt grant hardware authority.

## Invariants and RED

- Do not replace runtime libraries, stub ABI symbols, alter upstream tests,
  disable content checking or claim a build failed for a hardware reason.
- Default wrapper capture has no devices; hardware requires a separate
  attempt/phase/head/boot/environment/receipt-bound owner approval. An incomplete
  G0 receipt prevents device mappings even with an approval file.
- Exit zero is not acceptance. Exact executed non-skipped GoogleTest XML,
  topology/identity/firmware observations, diagnostic review, lifecycle and
  Watcher gates are independent. Raw streams are not a result transport.
- Fresh directories, exclusive/no-replace atomic publication; no overwritten
  attempts. Binary/container/cleanup absolute budgets are 40/50/60 seconds.
- Initial RED: `uv run pytest tests/test_issue30.py -q` failed import before
  implementation (exit 2). Behavioral RED:
  `uv run pytest tests/test_issue30.py::test_uncabled_warning_must_match_chip_identity -q`
  failed with accepted instead of diagnostic (exit 1): positive control passed,
  then a single PCIe-index mutation in an uncabled warning exposed a missing
  identity association. That gate is now implemented. Negative cases start
  with a positive control and assert the specific rejection reason.

## Build and isolated execution receipt

The checked-in `receipt.json` describes the actual attempts; raw receipts,
compiler/link streams and intermediate files are retained under the relative
`out/issue30/g0` role. `enodia/tt/bench/issue30_build/CMakeLists.txt` compiles
**unmodified upstream test translation units** with the upstream pinned
GoogleTest/gmock_main and headers, linking the image's audited libtt_metal and
libtt_stl. It does not build replacement Metal/UMD libraries or new kernels.
`tools/build_issue30.py` hashes inputs, toolchain, commands and outputs. Runtime
library pin is the same one audited for #120. See the receipt for actual success
or stop phase; no hardware-readiness claim follows from compile alone.

Isolation is non-root, network none, read-only root/source/dependencies, no
capabilities, no-new-privileges, no mapped devices and private writable build/
output/tmp only. Docker 29 rejects user mounts under proc here. Engine
`HostConfig.MaskedPaths` successfully masks sys, proc-bus and proc-driver,
verified empty by the child before work. No systempaths-unconfined or privileged
fallback was used. Earlier mount failures are retained as failed attempts,
not counted as builds. Both direct container exit and build receipt are checked;
a successful Docker cleanup command does not turn a failed build into success.

Device-free checks use `LD_BIND_NOW=1`, Watcher=1 and the same masked sandbox.
Listing must include the two intended selectors. The first negative attempts reached a missing runtime-root error and are
**not qualified device-absence evidence**. A separate runtime-root RED test
fixed the default sandbox environment before fresh v2 attempts. A non-list
invocation without devices must not yield eligible A/B evidence, regardless of GoogleTest exit;
the wrapper's non-list device-free CLI returns 2 and `judge` explicitly rejects
`device_free=true`. Loader/list/absent outcomes are recorded separately in the
receipt, not inferred from build success.

## Wrapper, schemas and acceptance

Entry: `uv run python -m enodia.tt.bench.issue30` with `plan`, `capture`, `judge`.
`plan` emits role-normalized mounts, never local paths. `capture` defaults to
device-free; `--approval`, `--before` and `--receipt` are all required for the
separate hardware branch. No such branch was invoked in G0. The future receipt
layout places binaries in `out/issue30/g0/assets` and pinned source in the
sibling source role; neither source nor device nodes come from inherited TT
variables. Only the two approved nodes and BDF visibility are allowed.

- `issue30-environment-v1`: attempt/boot/image/KMD/Watcher plus exactly the two
  serial/PCI identities, runtime chip-to-PCIe-index map, bundle, CMFW, base ERISC,
  independently observed ERISC heartbeat health, AICLK, DRAM status, expected reciprocal link/core/channel table and explicitly
  uncabled chip/channel pairs. Before/after stable values must match. Node paths
  appear only in private execution inputs. Minimum source gates: KMD 2.8,
  bundle 18.5, ERISC 1.4.1; they are necessary, not sufficient for first training.
- `issue30-owner-approval-v1`: exact attempt/phase/head/boot and SHA-256 of
  canonical environment and G0 receipt. Fresh owner observations are required;
  this file format does not let the agent approve its own hardware run.
- `issue30-execution-v1`: immutable direct container/client exits, started/
  timed-out/terminated/device-free state, elapsed time, binary hash, image,
  actual argv/environment and raw/ps/lifecycle companion hashes. No exit is
  inferred if observation was prevented. Unknown Watcher state stays null.
- `issue30-review-v1`: separate owner-reviewed attempt/execution/before/after
  hashes and Watcher-clean observation; never edit the execution file to add a
  health verdict. This is supplied observation, not automatically obtained
  telemetry. B additionally needs the accepted A verdict at the identical
  baseline fingerprint/boot; B's smoke stdout is not a second health report.
- `issue30-verdict-v1`: accepted_discovery / accepted_transfer or rejected with
  a gate-specific reason, baseline and XML/raw-stream hashes. XML uses
  defusedxml, is bounded, rejects declarations/entities and must contain exactly
  the expected executed, completed, non-skipped testcase and no errors/failures.

A's source-shaped report parser requires exactly two chip headers, expected
PCIe indices, one matching firmware-bundle establishment, every declared UP
link with its reverse peer/core and zero retrain counts. Unexpected DOWN or
UP links fail. It rejects unknown warnings/errors/fatal/critical diagnostics. The sole
warning exception is the source's exact DOWN/unconnected QSFP warning for a
**predeclared uncabled** chip/channel. This is necessary because ReportSystemHealth
warns for intentionally uncabled ports too (T system-health source:344-363).
It never exempts a cabled/expected link or a firmware warning. Warning PCIe
identity must match the chip mapping too. The parser uses source SemVer's
three-component stable log form (U semver.hpp:57-61) and the `x-y` core strings
from U `device/types/xy_pair.cpp:11`, preserving raw bundle observations
separately. These source-format gates have explicit RED/GREEN tests.

Tests' health/XML/firmware examples are explicitly **synthetic, source-derived,
not board output**. They do not establish that real diagnostics match this
parser. Any unexpected real format fails closed; do not edit parser mid-run or
retry. B content verification is upstream kernel/host assertions, not a PASS
substring; a skip, missing/extra testcase, timeout, bad environment/Watcher or
lifecycle evidence rejects it even on exit 0. Missing before/after/review files
cannot produce an accepted record.

## Firmware: what 19.6.0.0 predicts for A

| Evidence in pinned source | Prediction, not a board observation |
|---|---|
| U `device/topology/topology_discovery.cpp:554-602`, `device/firmware/firmware_info_provider.cpp:324-337` | 19.6 is between 18.5 minimum and 19.7.1 latest-supported. This bundle alone is not rejected and does not cause the older-than-minimum warning. Both-card mismatch is a default error. |
| T `INSTALLING.md:26-42` | The install guide lists 19.8.1; it is not the runtime minimum check. No automatic update is necessary merely to satisfy that text. It does not prove 19.6 trains links. |
| U `device/topology/topology_discovery.cpp:302-306,627-664`; `device/api/umd/device/firmware/erisc_firmware.hpp:20-25`; `device/firmware/firmware_utils.cpp:80-99` | Per-core base ERISC <1.4.1 can warn; inconsistent versions can stop discovery. Version is read from core major/minor/patch addresses, not the unavailable telemetry field. **BH discovery skips its heartbeat check**; the absence of warnings cannot prove heartbeat health. A requires an independent before/after heartbeat observation in its owner-provided snapshot; unknown is not accepted. |
| U `device/firmware/firmware_info_provider.cpp:212-230` | On BH 19.6 telemetry ETH_LIVE_STATUS remains zero and ETH_FW_VERSION is unavailable; neither zero proves link-down or ERISC version. Use per-core status/heartbeat and discovered peers. |
| U `device/tt_device/blackhole_tt_device.cpp:309-336`, `device/topology/topology_discovery_blackhole.cpp:207-209` | PORT_UNKNOWN can enter the long firmware-training wait; host retrain is skipped. Our watchdog stops it, not a larger training timeout. Settled-down can return with no peers. |

The pinned wheel supplies runtime firmware ABI headers, not an installed
firmware bundle or a CMFW implementation revision. No contrary requirement for
19.6 was found in its packaged runtime source. An update remains an owner-only
option if firmware evidence warrants it: retain both-card pre-state; choose a
vendor-supported exact bundle/checksum and matching toolchain; separately
approve flashing/reset/power-cycle scope; then recapture versions/health/new
boot and invalidate old approvals. No update command is included in this
executable wrapper, and no flash tool was invoked.

## Cable/port result and three-line physical checklist

T `tools/scaleout/board/board.cpp:330-350` maps P150 ASIC 0 port 1={11,9},
2={8,10}, 3={7,5}, 4={4,6}. This is not the sorted logical-core index: use the
SoC descriptor to translate ASIC channels/harvesting to the reported cores.
Public [Blackhole multi-card docs](https://docs.tenstorrent.com/aibs/blackhole/multi-card-topologies.html#port-and-card-numbering)
state port 1 is furthest from the motherboard, port 4 closest, one passive
QSFP-DD cable gives two links, and any port pair is valid for two cards. That
current public guide is supplemental, not part of the runtime source pin.
Its four-channel description does not override T's RELAXED MGD policy.

1. Trace both cable ends and record serial + physical port (1 top/furthest, 4 bottom/nearest motherboard).
2. Confirm passive QSFP-DD cable type/part, seating and no tension; do not recable while powered for this run.
3. Translate the table to expected channels/cores and mark all other QSFP channels uncabled before approval.

## Host prerequisites (read-only snapshot)

`host-readonly.json` is a safe reduced snapshot, not a board environment record.
On the execution host: 1-GiB pages total 8/free 7/reserved 0; default-size 2-MiB
pages total 0; both hugetlbfs mounts exist. Meminfo's HugePages_Total=0 describes
the default 2-MiB pool, not absence of the separate 1-GiB pool. Installed KMD
module file version is 2.12.0; the tenstorrent loaded-module sysfs directory
was absent. Do not call that a loaded driver version or board-ready host.
No hostname, device-node open or setting change was used.

T `tt_metal/llrt/tt_cluster.cpp:395-414` and U `device/cluster.cpp:430-437`
auto-size host channels by chips per closest MMIO. For two PCIe-local chips,
this is one channel each. U `device/hugepage.cpp:56-98` requires at least one
1-GiB page per MMIO device: **two pages**, conditionally covered by 7 free.
This is not proof of NUMA/permissions/DMA/device access. Owner verifies current
loaded KMD, page mount/permissions and free capacity on the actual board host.
If fewer than two pages are available, separately approved host administration
can reserve at least two 1-GiB pages at boot and mount a pagesize=1G hugetlbfs
at the UMD expected mount role; do not write counts/mount/change permissions
in this task. No increase is justified by the observed pool alone.

## Final approval units (one screen each)

### A: `issue30-a-discovery-watcher-v1`

- Gate: completed G0 receipt + clean/pushed approved head + current owner
  two-card identity/firmware/host observation and cable-derived expected links.
- Execute only Cluster.ReportSystemHealth, Watcher=1, fresh named container,
  binary/container/cleanup deadlines 40/50/60 s. Both cards are opened by UMD;
  second card serial 0000040331914124 / PCI 0000:09:00.0 is explicitly in scope.
- Prediction on 19.6: no bundle-minimum rejection solely for that version;
  per-core firmware/heartbeat and physical training remain conditional.
  PORT_UNKNOWN may reach our cap; settled down may return no peers/exit 0.
- Accept only the fixed A rules above, matching pre/post observations,
  no unexpected diagnostics, zero retrains, known lifecycle/after-state.
  Missing links, format ambiguity, identity/firmware/health drift or timeout
  is a recorded stop, not permission to alter descriptors/cables/firmware.
- Risk: both-device initialization and firmware waits; no payload or timing
  result. Reset 0/retry 0. Recovery needs separate owner scope/approval.
- B1 discovery evidence only; B3/B4 unmeasured. Owner reviews A before B.

### B: `issue30-b-fabric1d-smoke-watcher-v1`

- Separate approval after accepted A, identical boot/baseline/cabling and
  source/build hashes. Only Fabric1DFixture.TestUnicastConnAPI1DSmoke;
  explicit two-card mappings, FABRIC_1D/default dispatch/default two-ERISC,
  Watcher=1, binary/container/cleanup deadlines 40/50/60 s, fresh attempt.
- Prediction: after mapping/firmware/EDM handshake, upstream five checked
  packets reach receiver and close completes. Cold JIT may hit cap; no extension.
- Accept only exact executed non-skipped XML/assertions, accepted A fingerprint,
  unchanged healthy pre/post environment, clean Watcher/lifecycle and no new
  diagnostic. No health result is inferred from exit 0 or the smoke's stdout.
- Stop on skip/missing/extra/failed test, content mismatch, watchdog, unknown
  close state, new warning/error or either-card health drift. No fallback to
  legacy/2D, reset, retry, firmware update or cable action.
- Risk: programs/L1/ERISC/routing/credit/NoC state on both cards; possible hung
  peer. One verified transfer advances B1 and the #13 hardware prerequisite;
  it proves neither FPGA compatibility nor B3 jitter/B4 latency/B5/B6.

Morning owner list: both-card identity/base firmware/health; passive cable
endpoints/expected link table; loaded KMD/hugepage access; completed G0 receipt;
then **separate A and B approvals**. No timing/new-kernel run is authorized.
