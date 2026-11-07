# Issue #120 P6 V2: pre-branch device JIT failure

[Primary/hash manifest](2026-11-07-p150a-issue120-p6-api-lock-semaphore-v2-failure.json)
/ [session](2026-11-07-p150a-issue120-run7.json).

## Authorized conditions and precommit

The operator conveyed owner approval for V2 board-free routing correction,
precommit, P6 and conditional P7 only. P6 used clean/pushed
`49743efb848d33fa9ef856dd9429db9bfc8ba400`, native SHA-256
`7b9baa12d3adfd6d4f54496f5d0e50463a16af81285827849e74c6cad619fb5a`,
original fixed work, 60-second native/outer caps and Watcher on. Only device0,
serial `000004033191411c`, PCI `0000:06:00.0`, was exposed to computation.

Before hardware, full local tests and matching-head CI passed6118 tests with
1057 skipped/8 subtests; ruff/diff/ADR guard, pinned native host compile/link and
strict isolated no-chip startup passed. Public Sonar unresolved0/hotspots0/gateOK.
Those checks did **not** compile the device kernels. Fresh environment, empty
container list, both cards800MHz/DRAM healthy, no heavy host process, head/hash
and mapping were reverified immediately before P6. No heavy agent task overlapped
measurement. The named container and lifecycle/health records are retained.

## Result: stop before all planned run/coordinate branches

P6 native/container/wrapper exited **2**, without timeout. Producer device JIT
failed before the resident workload was dispatched:

```text
-DKERNEL_COMPILE_TIME_ARGS=0,4096,2,4096
get_ct_arg<4>()
static assertion failed: Index out of range
comparison reduces to '(4 < 4)'
```

The diagnostic at `q4_frame_producer.cpp:82` unconditionally reads
`get_compile_time_arg_val(ring_args.BankCoordsCTAOffset)`. The instantiated
index is4, but the actual compiler vector has only4 words. A public constexpr
offset does **not** guarantee an applicable bank slot exists.

Under the checked v0.75 ArgsConfig contract, config0 is non-sharded L1 and
config2 is DRAM. This four-word vector consists of config/page pairs for the
ring and stats accessors, with no explicit bank-coordinate word. The static
assumption of a sharded coordinate-bearing ring does not hold for this actual
compiler invocation. Why the runtime supplies that layout despite the intended
sharded allocation remains a separate board-free investigation; a host-library,
allocation/serialization or handoff cause is not selected by this log alone.

**Run branch1/2/3: unavailable. Coordinate A/B/C: unavailable.** No resident
kernel final stats, semaphore/coordinate readout or frame coverage result was
returned. Do not call this branch3/start0 or C/mismatched bank word: no bank word
was read, and absence of a result is not a sampled zero. The logged compiler
vector is the new raw evidence. Actual host CTA copies in RoutingRecord were
not persisted because final result construction was never reached.

The runtime-peer routing change is consequently **not hardware-validated** by
this attempt. P7's branch1 prerequisite could not be evaluated, so P7 and
production were not run. No automatic correction or retry followed.

## Retention, warnings and health

- Native stdout **12398 bytes**, stderr **4394 bytes**, direct execution metadata
  and return2 retained byte-exact with hashes. Native result directory is empty
  and wrapper result is absent; neither is reconstructed.
- **19 hashed companions** include env/power/baseline proof, native streams/status,
  reduced lifecycle/health, and explicit generated/wrapper text derivatives.
  Original private text hashes remain. All previous evidence bytes are unchanged.
- Seven warning lines match the three accepted #176 baseline conditions. The
  new BuildKernels error/TT_THROW condition triggered the hard stop; it is not
  an accepted baseline. Three error/critical logger lines are retained.
- Watcher attached/detached device0, **8 dumps**, no watcher anomaly matches.
  Clean Watcher and registered kernel names do not prove resident JIT success
  or execution.
- No container survivor. Both cards returned800MHz/DRAM healthy. **Reset0**:
  this was an identified software compile exception without timeout, surviving
  workload or unhealthy hardware requiring recovery. **Retry0**.
- Four frozen power samples (800/1350MHz) cover startup/cleanup observations,
  not resident frames or full-window timing. Fixed host work completion/counts
  cannot be asserted without a native result, even if a load thread was started.

## Q4 and remaining gates

Q4 gains no new real-frame coverage evidence and remains not evaluable. Ordinary
unpinned CQ1 with direct-L1 CPU challenge observation remains the first candidate,
not production-adopted; no CPU-only fallback is selected. Q1-Q3's earlier
bounded evidence and unresolved pinning capacity are unchanged; no P4 rerun.

Next is board-free diagnosis of actual buffer/sharding/CTA layout plus safe,
fail-closed diagnostic applicability/retention, then invariants/RED/correction
and a fresh committed identity/fingerprint with renewed owner approval. Existing
runtime-index tests and host/no-chip builds missed device CTA applicability;
record that validation limit rather than treating precommit green as device JIT
proof. Healthy bounded real-frame Watcher and conditional no-Watcher validation,
fixed-work coverage calibration if needed, non-aliasing long-run protocol and
same-session500000-frame/600-second production/control acceptance remain gates.
