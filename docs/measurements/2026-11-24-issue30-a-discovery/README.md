# Issue #30 A discovery — 2026-11-24

## Result and scope

One owner-authorized Cluster.ReportSystemHealth invocation ran at precommitted,
clean/pushed/tested head `822aa21f97bf3d899b8378bdb7cbb7439018a14b`, attempt
`issue30-a-discovery-watcher-v1-789de6a8faf8`, both specified p150a cards, Watcher=1,
fixed 40/50/60 s budgets. Operator ADR-0021 admitted reasoned unobservable ERISC/
ETH health only for discovery, without enlarging the owner's A-only scope.

**Connection observations match full H1; A is NOT accepted.** The committed
judge returned `rejected`, reason `watcher`: no Watcher output was generated,
and cleanliness was deliberately not asserted. Independently, four unexpected
UMD NUMA/membind warnings activate the stop rule. Even certifying Watcher would
not waive those diagnostics. No B/reset/retry/firmware/cable/host-setting action
followed. Exit zero and GoogleTest PASS are not acceptance.

The judge stops before topology extraction on the unknown Watcher gate. The
separate connection table is an explicitly observation-only transcription using
the precommitted source-report parser, NOT an edited verdict or B release.

## Reported connections

Each row has an explicit reciprocal UP endpoint and retrain count zero:

| Chip | Channels / logical cores | Peer chip | Peer channels / cores |
|---|---|---|---|
| 0 | 4 / 0-4 | 1 | 4 / 0-4 |
| 0 | 5 / 0-5 | 1 | 5 / 0-5 |
| 0 | 6 / 0-6 | 1 | 6 / 0-6 |
| 0 | 7 / 0-7 | 1 | 7 / 0-7 |
| 1 | 4 / 0-4 | 0 | 4 / 0-4 |
| 1 | 5 / 0-5 | 0 | 5 / 0-5 |
| 1 | 6 / 0-6 | 0 | 6 / 0-6 |
| 1 | 7 / 0-7 | 0 | 7 / 0-7 |

Both chips report channels 8,9,10,11 DOWN/unconnected, with matching summary
warnings. Thus four undirected links / eight UP endpoints fit H1's vendor ports
3/4, not H2's ports 1/2. Exact lane identity is observed here, not a preassigned
condition. These flags/peer records are not receiver-payload verification.

## New diagnostic originals (no allowlist change)

The raw archive retains full timestamped lines. Message bodies, verbatim:

```text
hwloc_set_area_membind(): failed for physical_device_id: 1 on NodeSet: [0] with errno: Operation not permitted (pid: 7 tid: 136331659612416)
---- ttSiliconDevice::init_hugepage: bind_area_to_memory_nodeset() failed (physical_device_id: 1 ch: 0). Hugepage allocation is not on NumaNode matching TT Device. Side-Effect is decreased Device->Host perf (Issue #893).
hwloc_set_area_membind(): failed for physical_device_id: 0 on NodeSet: [0] with errno: Operation not permitted (pid: 7 tid: 136331659612416)
---- ttSiliconDevice::init_hugepage: bind_area_to_memory_nodeset() failed (physical_device_id: 0 ch: 0). Hugepage allocation is not on NumaNode matching TT Device. Side-Effect is decreased Device->Host perf (Issue #893).
```

These are non-whitelisted for A, not necessarily never-before-seen on the host.
No retrospective exemption, added capability, seccomp relaxation or retry was
applied. The message identifies denied memory binding, not its complete policy
cause. Native stderr is byte-empty; no error/fatal line was observed. Eight
DOWN/unconnected summaries correspond to H1's four uncabled channels per card.

## Lifecycle, telemetry and remaining unknowns

Native/container/client exits were 0, one exact non-skipped GoogleTest passed in
0.396 s. Capture CLI returned 2 by design (capture alone never accepts). Whole
capture through cleanup was 0.885326279 s, no timeout, container terminated and
before/after lists empty. No Watcher log/dump was generated; requested Watcher=1
was captured, not converted into a clean-log claim. Per-core ERISC version and
independent ETH heartbeat remain unknown; output reports only bundle 19.6.0.

Both cards responded to after-state telemetry, with ARC counters advancing,
800 MHz, DRAM status true and no uncorrected GDDR errors. ASIC temperatures were
55.9 -> 55.7 C (serial ...411c) and 59.3 -> 59.3 C (...4124). Free 1-GiB pages
were 8 before / 6 after; total remained 8. This is an observed allocation effect,
not an administrative pool change, and no page cleanup/configuration was attempted.

## Evidence, interpretation and next gate

[A-discovery.json](A-discovery.json) is a machine-neutral public derivative with
raw hashes, bound scope/correction facts, before/after values and the unmodified
formal verdict. `A-native-evidence.tar.gz` holds byte-exact stdout.bin, stderr.bin,
gtest.xml and result.json as deterministic gzip/POSIX-ustar regular files. XML's
compile-time container source role is unchanged, never replaced with a local
host path. No machine name or host-specific absolute path is published. Original
private approval, observations, execution/review and telemetry remain unchanged.

This establishes that the pinned runtime reports four reciprocal UP links on
bundle19.6 in this configuration. There is no pre-A observable link baseline,
and boot/telemetry initialization occurred first: it does not prove A caused
first training or that a host trigger is guaranteed. No content transfer or
B3/B4 timing was measured. B remains unapproved and inadmissible: A is rejected,
NUMA warnings/Watcher evidence need owner-reviewed resolution, and A output
still lacks per-core version inventory/ETH health required by B. No automatic
follow-up probe, reset, firmware/cable intervention or retry is authorized.
