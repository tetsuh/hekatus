# Issue #30 renewed A and corrected-layout Bv2

Executed at **f8703fb5bebaaab5a3ae79488cfe3bb51f4259ef**, after clean/pushed
head,7026 passed/1057 skipped/8subtests, ruff, all10 matching CI checks and
public Sonar0/0/OK. Owner2026-11-25 granted one A then one Bv2 only on matching
fresh H1; both invocations are consumed. No retry/fallback/reset/firmware/cable/
host setting change. The earlier DRAM false/0x0 ->true/0x5555
[stop](../2026-11-25-issue30-ab-v2-preflight-stop/README.md) remains unchanged,
cause unknown. Three later healthy observations without raw snapshots are
**operator report only**, not admission data. No persistent-health claim.

## Result

- **A accepted_discovery:** both cards channels4-7 reciprocalUP/retrain0,
  8-11 DOWN/unconnected (same H1 as11/24). Native/client0,1.182740260s through
  cleanup, XML0.350s.12 warning lines:4 exact known NUMA/fallback,8 expected
  uncabled summaries. Watcher requested, output N/A for cluster-only diagnostic.
- **B accepted_transfer:** unmodified official
  Fabric1DFixture.TestUnicastConnAPI1DSmoke, strictFABRIC_1D/fast dispatch,
  initialized2 devices, RX-before-TX/checks/close completed. Native/client0,
  6.031428763s through cleanup, fixture XML5.286s/body0.915s, no skip/failure.
  All14 warning lines match frozen baseline (4NUMA,4motherboard,6SMC), no new
  export warning. Both generated/fabric mapping files exist in owned runtime.
- Content verification is **source-bound**, not just PASS: pinned
  test_basic_fabric_smoke.cpp74-175 sends5 with clock-derived seed, waits for both
  programs, requires sender/receiver PASS and equal byte counts. Unmodified
  RX.cpp52-109 has nonbenchmark PRNG-data validation. XML completion/no failure,
  original native output/content markers, pinned source/binary and reviewed
  Watcher meet the precommitted judge. No raw status values, byte counts or
  per-packet counters were printed, so none is invented. No FPGA or timing proof.
- Watcher184746bytes,5 complete dumps0.919-5.059s, both devices attached/detached;
  full anomaly scan/waypoint aggregation, kernel maps/lifecycle/sample rows and
  explicit hash-bound review passed. RX appears in kernel maps. Wait waypoints
  alone are not hangs or timing. Not a per-lane credit/firmware-health inventory.
- JIT log: `0/34 hits (0.0%) [0 cached, 5 build-once dedup, 0 merged artifacts,
  0 merged genfiles]`. Fresh owned runtime/cache, no prewarm/copied artifacts.
  JIT-only elapsed time not separately emitted. Body0.915s and fabric-init1.930s
  include other work and must NOT be labeled pure JIT or packet latency.

## Five readings before/after each phase

All20 raw snapshot invocations retained, each per-card DRAMtrue/raw0x5555,
GDDR_UNCORR_ERRS0,800MHz and advancing ARC, spacing5.243-5.253s. No restart.

|Series|ARC...411c|ARC...4124|Final temperatureC|1GiB pagesfree/total|
|---|---|---|---|---|
|beforeA|7717->7752|7718->7753|53.6/56.2|8/8|
|afterA|7834->7869|7835->7870|53.4/56.4|6/8|
|beforeB|7896->7931|7896->7931|53.6/56.2|6/8|
|afterB|8024->8059|8025->8060|54.4/57.6|6/8|

Identity000004033191411c/PCI0000:06:00.0 and0000040331914124/PCI0000:09:00.0,
bundle19.6.0.0/CMFW0.28.0.0, KMD2.12.0/kernel6.8.0-149-generic unchanged;
new boot differs from historical A, fresh A alone admitted B. Sysfs identity and
pinned image/official ELF hashes were checked; no firmware-byte replacement.
Both native containers terminated/removed; listings0 before/after, no post-state
anomaly. No further board operation after completed postflight. ETH per-core
firmware/heartbeat remain unknown, not inferred from ARC/link/PASS.

## Evidence / privacy

[AB-result.json](AB-result.json) binds execution/verdict/review and all series,
raw snapshot hashes, owner/operator provenance and archive hashes.
AB-native-and-telemetry-evidence.tar.gz has48 canonical regular members. Native
stdout/stderr/XML/Watcher/execution/verdict/review/layout/coordinate mapping
files are **byte-exact**. All20 public snapshot copies omit host_info and are
explicitly privacy-reduced, NOT raw full snapshots; actual raw stdout/stderr
remain private with hashes. Container-hostname ASIC export copy is explicitly
normalized; raw export hash retained. Selected DDR/uncorrectable/ARC/etc values
are unchanged. Original11/24 A/owner acceptance, Bv1, DRAM-stop and G0 records
and binary bytes remain unchanged. Successful snapshots do not rewrite the
prior anomaly or prove that intermittent faults cannot recur.

B1 now has one content-verified TT-to-TT official smoke; first-training cause,
FPGA protocol completeness and sustained traffic remain open. B3 source answer
stands, jitter cost unmeasured; B4 latency/jitter and #13 loopback need separate
plans/authority. #105 stage3 is not implemented or accepted by this run.
