# Issue #30 B — bounded stop in RX JIT, 2026-11-24 UTC

## Outcome / authority

**Rejected, reason `execution`. No receiver-content transfer was verified.**
One owner-authorized, unchanged official
Fabric1DFixture.TestUnicastConnAPI1DSmoke ran on both specified cards, attempt
`issue30-b-fabric1d-smoke-watcher-v1-82c9b462d5ba`, clean/pushed/tested harness
`ebf5a82aefca654d044eed517d7e10d9c263ca88`. Owner-accepted A observations were
bound at unchanged boot/cabling; A was not rerun. Source-bound receipt-v7 and
both native ELF hashes remained unchanged. Watcher enabled, strict FABRIC_1D,
fast dispatch, no custom descriptor, fallback, extension or retry.

Container-create admission was **2026-11-25 01:21:04.333835 JST**, before the
owner's01:22 cutoff. An external, deny-only wall-clock gate checked immediately
before the unchanged create API; its code/hash/boundary checks are recorded.
No new push preceded B after the deadline amendment. Capture elapsed4.784315771s,
native/client exits1, no timeout, terminated and container cleanup confirmed.

After-state telemetry and zero containers were prioritized and completed; the
requested power-off permission was then issued immediately. No board, device,
or tt-smi access follows that point. Subsequent work uses saved files only.

## Furthest observed stage / packets

-16:21:04.429 UTC: fixture fast-dispatch setup / cluster construction.
-16:21:04.665: STRICT_SYSTEM_HEALTH_SETUP_MODE.
-16:21:04.670: logical and physical mesh adjacency both report degree1 on2 chips.
-16:21:05.269: Watcher attached both devices; initialized with no disabled features.
-16:21:05.279: Initializing Fabric.
-16:21:07.213-.214: Fabric initialized on Device0,Device1,2 devices and
 FabricConfig::FABRIC_1D explicitly reported.
-16:21:08.709: exact test body begins (XML timestamp).
-16:21:08.753: RX BRISC compile failure; header lookup fails. XML body0.047s,
 one executed/non-skipped testcase, one failure, no error testcase; suite4.355s.
-16:21:08.785: JIT cache stats0/32 hits, Watcher stops/detaches both devices.
-16:21:09.014: cluster close/destructor completed.

These are actual logger/XML observations, not separately timed fabric/handshake
measurements. Routing/credits/EDM substages not independently logged remain
unproven. The fixture reports initialization, but the application RX kernel JIT
fails before receiver enqueue completes; source control flow does not reach TX
dispatch or packet assertions. **Five packets were expected from source, not
observed checked.** No received-byte/PRNG/status words or packet counters were
printed; those values remain null, not invented zero or PASS. A first verified
chip-to-chip payload remains pending.

## Diagnostics, original text retained

Fourteen warning lines exactly match #176/#120 baseline: four NUMA lines,
four known-motherboard fallback lines for the two chips, six SMC-unavailable
lines. Two warnings are **not baseline**, retained verbatim:

```text
2026-11-24 16:21:04.671 | warning  |          Fabric | Failed to export physical chip mesh coordinate mapping: filesystem error: cannot create directories: Read-only file system [/opt/venv/lib/python3.12/site-packages/ttnn/generated/fabric] (control_plane.cpp:605)
2026-11-24 16:21:04.671 | warning  |          Fabric | Failed to export ASIC to Fabric node ID mapping: filesystem error: cannot create directories: Read-only file system [/opt/venv/lib/python3.12/site-packages/ttnn/generated/fabric] (control_plane.cpp:614)
/source/tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp:8:10: fatal error: tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp: No such file or directory
```

The raw stdout also retains full emitted compiler argv, BuildKernels error,
TT_THROW critical diagnostics and exception backtraces. Native stderr is empty.
No warning was newly allowlisted and no result was reinterpreted as success.
The readonly export warnings independently require stop; the RX source resolves
but its source-root-relative test header does not resolve under emitted include
search roots. This is runtime include/export layout evidence, not link failure,
content mismatch or firmware-health diagnosis. [Follow-up #196](https://github.com/tetsuh/hekatus/issues/196)
requests device-free reproduction/layout qualification, not a retry or fix in
this completed attempt.

## Watcher

A captured146731-byte log has4 completed dumps starting at0.840,1.862,2.878,
3.893s, attached devices0/1 and detached at4.356s. Scan/manual sampled review
found no assert/exception/error/hang/invalid/warning/sanitizer/fatal/critical/
timeout/mismatch indication. Last samples show worker GW/W and active ERISC
K/NWID with fabric-router kernel IDs, not application packet completion. TX/RX
kernel names were registered, which is not dispatch evidence. All log text,
kernel name/ELF-path maps and inspector YAML are archived byte-exact. No base
per-core ERISC version or independent ETH heartbeat health is inferred.

## Fresh preflight / after-state / hugepage accounting

Matching A boot, both serial/BDF pairs, bundle19.6.0.0/CMFW0.28.0.0,
loaded KMD2.12.0/kernel6.8.0-149-generic, image/binary/source hashes,
zero containers and quiet host passed. ARC progresses in both fresh pairs:

| Field | ...411c at PCI0000:06:00.0 | ...4124 at PCI0000:09:00.0 |
|---|---|---|
| Before ARC pair | 172095 -> 172110 | 172105 -> 172121 |
| After ARC pair | 172677 -> 172692 | 172687 -> 172703 |
| Clock / DRAM / GDDR uncorrected | 800MHz / true /0 before and after | same |
| ASIC temperature |56.0 ->57.5 C|59.6 ->60.5 C|
| Bundle / CMFW / identity |unchanged|unchanged|
| Per-core version / independent ETH health |unknown|unknown|

Before collection began16:20:24 UTC; after began16:21:22 UTC. Counter pairs and
source timestamps are saved, not continuous health monitoring. No manual reset,
flash, cable or host-setting operation occurred.

Eight1-GiB hugepages total, **six free before and after B**. Read-only stat/list
found regular files `device_1_tenstorrent` and `tenstorrent`, each1073741824
bytes allocated (2GiB total). Their before-B mtimes14:17:41.259/.318 UTC are in
A's native interval; pinned UMD hugepage.cpp:162-196 names these device0/device1
channel0 backing files. Thus retained allocated backing files account exactly
for A's free8->6. The libvirt directory has zero allocated bytes. No file payload
was opened/mapped/read, no file was removed and no pool/settings were changed.

## Evidence / remaining gates

[B-result.json](B-result.json) is a public role-normalized derivative with
unmodified formal verdict, fingerprints, milestones, warning originals, card
values and hugepage stat accounting. `B-native-evidence.tar.gz` is deterministic
gzip/POSIX-ustar with12 regular members: byte-exact stdout/stderr/XML/verdict,
Watcher/maps and inspector logs. XML/backtraces/ELF maps include only fixed
container/runtime/compile-time paths, not a machine name or host-specific path.
Private approval/observations/execution/review/telemetry and partial JIT cache
remain unchanged outside Git.

B1 now has observed strict1D initialization plus a precise RX-JIT stop point;
no first receiver-content transfer or FPGA compatibility proof. B3 cost/jitter
and B4 latency remain unmeasured, #13 loopback remains separate, and #105 stage3
still needs verified transfer plus B5/B6 and prewritten AC. Next: #196 board-free
layout reproduction/qualification, then a **new** explicit owner-approved B-like
identity after power-on/boot/link-evidence decisions. B3/B4/#13 each need their
own precommitted instrumentation/wiring/run scope. Current B approval is consumed;
no extra hardware action, retry or fallback is authorized.
