# ADR-0020: Bind host-only submission observations to a C1-only admission

- Status: **Accepted** — 2026-11-18
- Date: 2026-11-18
- Acceptance: carried by this PR; effective only at the owner's merge under workflow section6

## Context

B4 produced healthy bounded control/loaded v2 records, but its inclusive worker
rate includes an unresolved enqueue/JIT lead. C1 needs first committed enqueue,
main enqueue-return count and sparse host samples to distinguish startup from
proportional work. A concurrent read of plain Traffic.emitted would race.

The owner approved only the catalog small calibration,16384frames/1048576writes,
Watcher off/60s/device0. The owner then powered down p150a and limited this task
to board-free preparation. C1 authority remains for a later power-on/fresh
preflight; no current board/device/tt-smi access, session activation or C2/P.

ADR-0019's exact program-identity admission cannot equate a changed host native
with original B4. The original records must not be rewritten to claim a new
binary. An explicit evidence-compatibility relationship is needed, not an
exception that ignores arbitrary native or Python mismatches.

## Options considered

- Read ordinary Traffic.emitted from main: rejected; data race.
- Lock/rendezvous worker progress or log every iteration: rejected; avoid extra
  scheduling, I/O and repeat-dependent allocation in the measurement loop.
- Modify measured protocol/kernel sources or label B4 as the successor: rejected.
- Reuse the unchanged submit loop through a thin host-only adapter, with an
  additive native successor and an explicit C1-only bridge: selected.
- Generic host_only flag or unconstrained digest exemption: rejected.

## Decision

1. Preserve native.cpp, host_protocol.hpp, host_sequence.hpp and all device/
   protocol sources byte-exactly. Generate native_trace.cpp in the fixed build
   directory by fixed unique substitutions from the measured native base, and
   add a thin adapter around the
   original submit_fixed loop. Reverse those substitutions and hash the entire
   recovered base; reject all other source/dispatch changes. A canonical private
   generated-input directory contains only that translation unit, never headers
   or links. Rebase legacy relative includes to the pinned source tree and reject
   shadow inputs before compilation/identity checks. Use fresh temporary output
   and atomic replacement, not direct truncation or symlink-following writes.
   Fresh configure rejects ambient compiler/include overrides and preserves the
   original empty-build-type/no-user-CXX-flags configuration. Effective generated
   flags and the compiler dependency file's resolved protected headers are checked
   and receipt-bound; an unchecked cached quote-search option is not evidence.
   These boundaries address the board-free findings tracked in Issue #184.
2. Observe a preceding fully committed write before the next original challenge
   read, and the last commit after submit_fixed returns. The definition is a
   **published fully committed successful host enqueue prefix**. It can lag
   API return/bookkeeping slightly. It is not queue retirement/device delivery.
   Payload encoding, CPU arithmetic, read/write count/order and transition logic
   remain the actual original functions, not reconstructed equivalents.
3. Publish only that scalar through a separate lock-free uint32 atomic store.
   Main performs one scalar load between before/after host monotonic timestamps.
   Before join it never reads Traffic, sample rows or worker-owned counters.
   No mutex, wait-for-progress or per-write clock/allocation/I/O is introduced.
4. Sample count1 and each multiple of65536. The fixed array65536x16bytes covers
   the complete positive uint32 domain with worker-end metadata separate. C1
   emits17 rows and one main-side extra clock read; existing start/end/join
   timestamps are reused. A trace clock failure rejects successor execution;
   retain raw failure status and normal existing cancellation/join semantics.
   Existing unrelated queue-drain exception behavior is not changed here.
5. Host observation has nonzero cost: one scalar publication/store plus cadence
   predicate per committed write and sparse clocks. It is not a claim of zero
   host wall-time effect or identical B4/C1 rates. Runtime challenges remain
   observations; identical packet algorithms do not mean identical sampled
   challenge values across executions. Trace never supplies causal coverage.
6. Successor native schema is issue120-production-native-v2 with nested
   issue120-host-submission-trace-v1. Outer record remains production-v2. Current
   native success requires the trace; its program identity declares that contract,
   and persisted records cannot downgrade it by deleting trace/schema fields.
   Old B4/native-v1 records retain their exact existing validator/interpretation.
7. New bounded-admission-v2 roots its old identity in the exact two B4 archives,
   primary-record hashes and session-manifest hash. Replay originals in bounded
   fresh temporary storage, reject unexpected/duplicate/redirected members and
   verify every payload before ordinary record-loader health/coverage checks.
   Never relabel source records as current or depend on untracked extraction.
8. Bind the actual target identity and source-pair proof in committed JSON, not
   self-referential Python constants. Keep library, derived device/support,
   CTA and device-compile-proof identities equal. Only named host-side Python
   changes/new trace/admission modules may differ from the original B4 identity.
   New target/native/Python mismatches and all device/dispatch changes reject;
   the pinned image must also equal the original B4 session image.
   Existing exact-identity admission remains version-dispatched and unchanged.
9. New bridge scope is only production-v1-calibration-small. Reject C2 sessions,
   alternate counts and P. Preserve current identity checks at host/in-container
   launch and bind admission bytes into the later fresh-boot session. A receipt
   is evidence eligibility, not current execution authority.
10. Under the owner's explicit host-only exception, the next approved C1 may
    remain Watcher off if the mechanical device/dispatch proof, actual generated
    hashes and unchanged four-case offline device proof all hold. A changed
    native is not being declared hardware validated; any device-facing delta
    invalidates this exception and requires stopping for owner disposition.

## C2 refinement in the same unmerged PR

After the single healthy C1,the owner approved only catalog C2 large,
32768frames/1572864writes,Watcher off/device0/60s. The
[C2 precommit](../measurements/issue120-production-c2-plan.md) refines admission
without altering native/device content or rewriting C1/B4 evidence.

- Version3 bounded admission is C2-only and separately binds original B4
  Watcher health,C1 archive/record/session/environment and C1's preserved v2
  receipt. v1/v2 semantics remain version-dispatched,not silently expanded.
- Compatibility requires the exact same native/trace,device/kernel/CTA/library/
  compiler/image and work units. Complete old/current Python maps and every
  metadata-only runtime/recording/sessions digest change are explicit. Changed
  harness commits remain separate original facts,not false record equality.
- Compare outer environment except the explicitly retained commit plus selected
  stable raw capture fields,including Linux149/KMD2.12 and board/firmware/tool/
  limits/image/sampler/device facts. Different boot/session/timestamps and thermal
  observations are not claimed equal. Each run retains its fresh session binding.
- New metadata declares its own provenance contract and binds raw environment
  size/hash and session-verified receipt. Historical native/C1/B4 interpretation
  is not retrofitted. Removing mandatory provenance from a declared new record
  must reject on reread.
- Compatible-pair offline analysis is a new explicit path,not a relaxation of
  legacy exact-program/environment calibration or production comparison. Both
  original records are artifact-validated and displayed with their distinct
  provenance. Healthy valid not_covered retains worker/cost diagnostic value,
  but never gains coverage/timing/P authority. Invalid/not_evaluable evidence
  remains unusable. Proposed counts are non-authorizing;550s floor,500000frames,
 600s and all P gates remain unchanged pending a separate precommit/decision.

This refines this still-unmerged ADR under the owner's explicit instruction;
no settled main decision is rewritten and no hardware/P authority is inferred.

## Consequences

- New native digest/build receipt and full offline startup/publication/tests/CI/
  Sonar proof are required; old B4 native/receipt remain reproducible/preserved.
- Sparse rows permit qualified interval rates and startup analysis but cannot
  prove a500-second model, continuous traffic or a production count.
- C2/P remain blocked. Do not create a runtime session or take new environment/
  boot/device samples while the board is powered down.
- Future execution begins with fresh Linux149/KMD2.12 and all other exact baseline
  checks, then a new C1-only boot-bound session at a clean/pushed green head.
- The source proof is a bounded reviewed transformation and build-identity check,
  not a formal proof for arbitrary malicious compilers or source modifications.

## Status history

- C2 follow-up: owner-approved same-PR refinement adds explicit C1/C2 metadata
  compatibility and C2-only evidence admission without changing the native or
  granting production execution. Earlier C1 evidence retains its identity.

- 2026-11-18: Recorded host-only trace and C1 evidence-compatibility refinement
  in the same carrying PR. Effective through owner merge; no separate pre-merge
  architecture acceptance is claimed. This stage performs board-free preparation
  only, preserving all B4 records and measured ancestor sources.
