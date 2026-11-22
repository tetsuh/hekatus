# ADR-0019: Version the native production causal instrument

- Status: **Accepted** — 2026-11-14
- Date: 2026-11-14
- Acceptance: the transition is carried by this PR and takes effect at the owner's merge under development_workflow.md section 6

## Context

PR #172 established bounded4096-frame causal CPU/PCIe coverage, but its Q4 payload,
per-frame reset checksum and semaphore roles are not the Issue #118 resident
program. Its19-bit sequence/13-bit challenge cannot represent the production
counts. The old threaded Python runner also synchronizes CQ0 before joining its
CQ1 writer, which shares the MeshDevice API mutex. No production result follows
from those feasibility records.

Issue #120 needs the same resident computation/metrics, a matching same-session
control, a long-run non-aliasing causal protocol, and stronger new-record host
interval bindings (#180), without rewriting accepted evidence or measured code.

## Options considered

- Increase Q4's frame count and relabel it as Issue #118: rejected; payload,
  arithmetic state, pacing and failure semantics differ.
- Modify the measured resident/probe sources and legacy validator in place:
  rejected; it obscures reproduction and could invalidate accepted records.
- Add a native host with hash-pinned versioned adapters of the resident program,
  a new wire/record schema, and the existing atomic transport: selected.
- Use two independent raw sequence/challenge words without an association proof:
  rejected; a torn read can manufacture a tuple never sent. A checksum alone is
  not an injectivity proof.
- Add invalid/payload/commit enqueues: not selected; device-visible ordering still
  needs proof and extra commands materially change the load being calibrated.

## Decision

1. Preserve legacy sources, records, schemas and guards. A deterministic adapter
   verifies base hashes and exact hook sites, then emits new kernels. Keep the
   original synthetic I/Q payload, cumulative64-step arithmetic, immediate first
   attempt after startup, cadence/drop-new behavior, error cancellation, drain,
   completion timestamp and output words0-3. Add admission/progress observations
   before the original work-start sample, so arithmetic duration remains distinct.
   Observation cost still affects completion intervals in both conditions; this
   is not a binary-identical rerun or retrospective replacement of #118.
2. Use one native MeshDevice per attempt, two CQs and identical resident/transfer
   subdevices and transfer-only stall group for control/loaded. L1 buffers use
   explicit distributions; host WORKER virtual coordinates are runtime arguments
   for all cross-core destinations. Compare complete CTA arrays to the manifest
   checked by the pinned device-free compiler before dispatch.
3. The callable native lifecycle starts workers, enqueues, joins all possible
   CQ1 submitters, then finishes CQ0 and CQ1 before retrieval. Error cleanup also
   stops/joins submitters before any finish, drains queued work before releasing
   resources, and retains the original failure. A low-frequency
   targeted-L1 terminal observer is present in both conditions; its host endpoint
   is an observed upper bound, not exact resident completion. Post-join finish
   return stays a separately named, wider upper-bound envelope.
4. Wire `issue120-wide-progress-v1` has two little-endian32-bit limbs s and s+c,
   with fixed-position reversible XOR transforms over the whole4KiB page. For F
   frames and N fixed repetitions require N+F+2 <= UINT32_MAX in wider arithmetic.
   Initialize using encoded(0,0), never treat sequence0 as load evidence. Challenges
   are read before the fixed CPU work and cannot decrease. Both emitted limbs
   therefore increase strictly without wrap.
5. Persist the exact committed(first_sequence,challenge) transition table, bounded
   by F+3 entries rather than N. Commit an entry only after successful submission.
   Final evaluation binds every used raw pair (including the exact startup pair)
   to that map and successful-submission range. Whole-page validity alone is not
   sufficient. The proof assumes coherent32-bit observations, no unrelated writer,
   immutable source bytes until inline enqueue copy returns, and timestamps after
   completed reads; it does not establish arbitrary hardware byte-coherence.
6. Both conditions run the same bounded100ms startup policy; its final one-period
   slice is reserved for the distinct frame-zero challenge before start release.
   Neither changes resident pacing after release. Startup1, frame i challenge i+2
   and tailF+2 remain separate. Frame observations are fixed-count, never extended
   until progress succeeds. Tail/startup waits are bounded outside frame work.
   Control has no CPU/CQ1 submitter and no advancing-load coverage requirement.
7. An accepted page proves4096 bytes arrived by scan end in its fresh-challenge
   interval, not atomic snapshot or CQ retirement. Do not assign bytes between
   scan-end times, multiply skipped generations, or claim continuous occupation.
   Only later-frame challenges issued after a previous completion count toward
   the first-through-last window byte lower bound; frame-zero and tail scopes are
   separate. Missing first/last or any required causal interval prevents coverage.
8. `issue120-production-v1` is a new outer record with fixed-count configuration,
   compact attempt-bound binary sidecars, existing resident metric algorithms,
   program/source/native identities and frozen power provenance. Bind host/CPU
   endpoints to each other and to the load join only in this schema. Legacy records
   retain their validator or are explicitly out of scope; no completeness claim.
   New diagnostics distinguish consumed+2 from configured+2 (#179).
9. A session manifest binds the new control and loaded records, program/open
   configuration, source/image/native digests and execution order. Historical or
   unlike-session controls cannot produce the required difference. ADR-0015 still
   governs ticks; power-weighted conversion over a named observed host envelope
   is not an exact frame-window AICLK or product wall-time guarantee.
10. Fixed production work is chosen offline from bounded calibration of this exact
    host algorithm, then precommitted. A count proposal is not execution authority.
    No adaptive extension, fallback, automatic retry or board authorization follows
    from a successful plan/build/test. First-kernel, calibration and production
    control/loaded approval units remain separate.

11. Following the B1 integration failure, fresh publication uses outer record
    `issue120-production-v2`. Its power reader explicitly recognizes the v2
    enqueue-to-terminal-observed upper-bound source without borrowing the #118
    label. New non-Watcher timing requires error-free, complete, positive finite
    weighted provenance. Existing v1 records retain their original interpretation,
    including B1's null weighted clock/error, and cannot supply new stage admission.
    Public v2 companions are created0644 before exclusive publication so the host
    user can read root-published bytes; the shared snapshot default remains0600.
    Hash/manifest checks, not mode, establish immutable evidence. No extra
    container privilege or ADR-0018 native transport change is introduced.

## Consequences

- Measured-version implementation limit (2026-11-22): Decision 3 is unchanged,
  but the measured implementation does not satisfy its cleanup/error-retention
  requirements across the error paths as a whole. Production-plan Invariant 3
  is verified only for the cases covered by the host-sequence tests. The class,
  not just the currently listed drain, observer-cause and cleanup-clock examples,
  is deferred to [Issue #193](https://github.com/tetsuh/hekatus/issues/193) under
  the owner's 2026-11-22 measured-source preservation decision. Further cases
  belong there; the corrected instrument must precede another board measurement
  with new provenance. Healthy B4/C1/C2/P1/P2 records are unaffected, and their
  measured source, digest and records are not changed or relabeled.
- At500000 frames the original timestamp pages occupy2048000000 device bytes;
  compact retrieval and the(F+3)x8-byte association table are bounded by frame
  capacity, not transfer repetitions. Runtime queue occupancy is not measured.
- The wider protocol preserves one ordinary write per fixed CPU/read iteration,
  but its overhead still requires new calibration. Run8 extrapolation is only a
  provisional estimate, not proof of550-second coverage or600-second feasibility.
- Derived kernels and support hashes become reproducible build evidence. Syntax/
  constexpr checks do not establish linking, physical routing or runtime health.
- Failures preserve raw transport/artifact evidence without promoting partial
  records to coverage/timing success. New-schema validation is deliberately not
  described as complete for every possible malformed input.
- First/last causal progress is a sampled interval property. It is a sufficient
  condition for this instrument's coverage definition, not a necessary condition
  for all forms of host activity or a claim about every instant of CPU/PCIe load.

## Status history

- 2026-11-22: Recorded the measured implementation's error-path class limitation
  and #193 follow-up under the existing owner decision. This corrects the scope
  of implementation claims, not Decision 3 or this ADR's acceptance convention.
- 2026-11-14: Recorded the production instrument/schema/protocol transition in
  the carrying PR. The owner accepts it by merging that PR; no separate pre-merge
  acceptance or later header transition is implied. No board execution occurred
  in this preparation stage.
- 2026-11-18: Added the versioned power/readback refinement for #182/#183 in the
  same still-unmerged carrying PR. B1 native/wrapper succeeded but host reread
  failed; old evidence and measured native/kernel bytes remain unchanged. This
  refinement is board-free and grants no B rerun, calibration or production.
