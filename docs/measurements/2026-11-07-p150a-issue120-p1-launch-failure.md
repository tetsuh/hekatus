# Issue #120 phase 2: stopped at the first native launch

## Record and boundaries

The authority is the immutable
[attempt record](2026-11-07-p150a-issue120-p1-launch-failure.json), with its
hashed environment, sanitized log, health, Docker lifecycle/listings and frozen
startup power trace. The original prediction and seven-run order remain in
[the plan at the measured harness revision](issue120-feasibility-plan.md).
This is diagnostic startup evidence, **not** a completed probe or timing result.
No result was fabricated in the missing wrapper-owned runner-result slot.

The clean, pushed harness was `b1865feced7b53c1116a0431a115198f6636c7e3`.
The existing native executable matched its previously recorded successful-build
SHA-256 `63e7bb4dbf5c21dc151bbcf7e75fa3a9abf4bc0f744dac19e73c50d90ba686d5`;
no rebuild or toolchain substitution was needed. The staged library matched
its audited pinned-image hash. Identity, firmware, KMD and CPU provenance were
captured afresh after reboot, not copied from historical measurements.

Device 0 alone was exposed to the named probe container: p150a serial
`000004033191411c`, PCI `0000:06:00.0`. Device 1 was checked only by the
requested host telemetry, never exposed to the compute container. Both boards
were at 800 MHz before and after P1. Fresh post-run telemetry reported healthy
DRAM. Container listings were empty before and after; lifecycle events bind the
named container's create/start/die/destroy and exit code 2. No forced cleanup,
reset or retry was needed or performed.

## Outcomes against precommitted predictions

| Probe | Native / outer cap | Watcher requested | Outcome and prediction comparison |
|---|---|---|---|
| P1: Q1, resident-and-transfer stall group | 60 / 60 s | on | Exit 2 during native runtime initialization: `Root Directory is not set.` The predicted observer witness/missing-observation branch was not reached. No conclusion about worker gating or overlap. |
| P2: Q1, transfer-only | 30 / 60 s | off | Not run: stop rule after P1 abnormal exit. Prediction untested. |
| P3: Q2 write-only | 30 / 60 s | off | Not run: stop rule. Prediction untested. |
| P4: Q3 pinned NoC | 60 / 60 s | on | Not run: stop rule. Mapping, payload and Watcher predictions untested. |
| P5: legacy Q4 falsifier | 30 / 60 s | off | Not run: stop rule. Fixed-work insufficiency prediction untested. |
| P6: Q4 actual frames | 60 / 60 s | on | Not run: stop rule. Every-frame challenge/progress prediction untested. |
| P7: Q4 without Watcher | 60 / 60 s | off | Not run: no healthy P6 or earlier sequence. Prediction untested. |

P1's wrapper invocation spanned `2026-11-07T00:04:17Z` to
`2026-11-07T00:04:18Z` at one-second UTC logging resolution; it did not time out.
These host observations are not card completion timestamps. The requested
Watcher setting does not establish Watcher validation: no new kernel reached
that gate. The single startup power sample is preserved, but cannot establish
resident-window coverage without a card program window.

## Interpretation: Q1-Q4 and selection

All four questions remain **unresolved**:

- **Q1:** no completion bracket or resident window was emitted. A launcher
  failure cannot establish serialization, overlap, or post-resident execution.
- **Q2:** the write-only loop was never reached; no sustained progress or
  empirical repeat-independent memory evidence was collected.
- **Q3:** no pinning or disjoint-core NoC kernel was exercised.
- **Q4:** no real frames, CPU challenge responses or full-window DMA progress
  evidence was collected.

The first candidate remains ordinary unpinned CQ1 bulk writes with explicit
same-consumer-clock direct-L1 challenge observation, **not adopted**. Its
selection gates have not been met. This failure rules out neither that candidate
nor pinned-host NoC traffic; no CPU-only fallback is made.

## Startup diagnosis and next precommit proposal

The pinned library's stack shows `RunTimeOptions` construction during default
`MetalEnv` initialization failing before `create_unit_mesh` returned. The native
runner creates its mesh before any probe-specific buffer or kernel setup.
The wrapper/native launcher supplies the runtime library search path but no
explicit runtime-root adapter. The inspected image configuration also has no
runtime-root environment value. The repository root used as container working
directory is not the vendor runtime installation root.

The v0.75.0 source at `tt_metal/llrt/rtoptions.cpp:306-347` checks an installation
root, `TT_METAL_RUNTIME_ROOT`, an API root, and a working-directory fallback
containing `tt_metal/`, then rejects an empty root. This corroborates a missing
startup prerequisite; it does not prove full source/image equivalence or that
setting one variable alone will make the probe work. Runtime assets, JIT and
hardware remain unvalidated.

Before another board invocation:

1. Track the discovered startup prerequisite, inspect exact pinned-wheel runtime
   assets without opening a device, and add a tested, program-owned runtime-root
   adapter plus required-asset preflight. Do not choose a replacement image,
   newer library, arbitrary path, or merely suppress the error.
2. Commit/push the correction, tests, renewed first-kernel Watcher prediction and
   separate attempt identity. Preserve this attempt unchanged. Obtain renewed
   owner authorization before retrying P1 with the same 60-second cap.
3. Continue P2-P7 only after their preceding health/Watcher/prediction gates.
   Reassess runtime prerequisites if another initialization stage fails; never
   retry until covered or enlarge caps automatically.
4. Only after bounded feasibility succeeds, separately precommit widened
   non-aliasing production protocol, fixed CPU/transfer work calibration,
   same-session identically configured control and frozen trace coverage.
   The 500,000-frame production run remains unexecuted and unauthorized here.

No implementation change or further device invocation was made to investigate
around this stop. This commit adds records and interpretation only.
