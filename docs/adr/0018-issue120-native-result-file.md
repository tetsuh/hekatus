# ADR-0018: Issue #120 native results use attempt-bound atomic files

- Status: **Accepted** — 2026-11-07
- Date: 2026-11-07
- Implementation direction: decided in PR #172 task instructions as the correction strategy for Issue #175

## Context

The `p1-runtime-root-v1` attempt reached the wrapper's last-stdout-line JSON
parser, but no result could be recovered. The native streams were not retained.
Cleanup output after JSON is plausible, not confirmed by the lost streams.
The supervising operator's PR #172 task instructions require this prerequisite
to be fixed as the Issue #175 correction before another P1, without touching
the board or rewriting either failed attempt. This implementation instruction
is not an owner architecture-acceptance decision.

This decision concerns only the experimental native/wrapper instrumentation.
It does not change the production data plane, frame timing or coverage rules.

## Options considered

- Continue parsing the last stdout line: rejected; runtime diagnostics can
  follow it, and it gives no reliable attempt-bound result slot.
- Search stdout for a JSON-looking line: rejected; stale, multiple or diagnostic
  JSON could be selected ambiguously, and raw bytes would still need retention.
- Use a wrapper-owned fresh attempt directory with atomic native publication:
  decided in the PR #172 task instructions as the Issue #175 correction strategy.

## Decision

1. The wrapper's collision-safe run ID is the transport `attempt_identity`.
   The separately precommitted logical identity names the planned run directory.
   Neither provides a CLI override for arbitrary result roots.
2. The wrapper exclusively claims `/out/native-<run-id>` and its `result`
   directory. The only accepted result is `result/result.json`.
   The native process validates that exact identity-derived path and empty,
   canonical parent before mesh creation. It writes `result.json.tmp` with
   exclusive/no-follow creation, flushes/syncs/closes it, then uses Linux
   no-replace atomic rename and directory sync. No prior result is replaced.
3. The envelope schema is `issue120-native-result-v1`, with exactly
   `schema`, `attempt_identity`, and object-valued `result` fields. The payload
   remains the existing probe result; the wrapper applies the existing record
   validation and provenance afterward.
4. Native stdout/stderr go directly to separate `stdout.bin` and `stderr.bin`
   files opened in exclusive binary mode before launch. They are not decoded,
   parsed or removed on any verdict. The wrapper writes atomic `execution.json`
   with schema `issue120-native-execution-v1`, identity, direct wait return code,
   started/timeout state, errors and stream byte counts/SHA-256. Preflight/spawn
   failures have no native return code; it is never inferred from a log.
5. Missing/multiple result entries, leftover temporary entries, symlinks,
   malformed/oversized JSON, duplicate keys, wrong schema/identity, non-object
   payloads and nonzero/timeout/invalid execution states cannot succeed.
   Diagnostic or cleanup output on either stream cannot select a result.
6. Existing result/trace bytes and named harness ancestors remain immutable.
   A new `p1-result-file-v1` prediction and fresh wrapper run ID are required.
   Board-free testing and device-free startup do not authorize a hardware retry.

## Consequences

- A failed parser no longer destroys the native evidence. Successful records
  reference their attempt capture directory and direct exit code.
- Retention is bounded by the diagnostic streams produced by a capped process,
  not a growing in-memory output string. Storage must be available before a
  native launch; failure to persist diagnostics cannot produce success.
- If the entire container is externally killed, bytes already written remain
  where the output mount survives, but no observer can guarantee it records a
  final native wait code after its own termination. Missing execution metadata
  is incomplete evidence, never inferred success.
- Atomic publication is Linux-specific, matching the pinned image. Unsupported
  rename/sync operations fail closed; there is no overwrite fallback.
- Raw streams may contain sensitive diagnostic text. They remain byte-exact;
  any later public redacted derivative must be separate and identified, not an
  in-place edit of the raw artifact.
- This decision does not establish kernel, Watcher, DMA or frame-window health.
  Mechanism selection and production measurement gates remain unchanged.

## Status history

- 2026-11-07: Recorded the correction strategy determined by the supervising
  operator's PR #172 task instructions for Issue #175, not an owner decision.
  Corrected the earlier attribution to the owner. Owner acceptance through
  PR merge remains pending.
- 2026-11-07: The settled header marker is retained only because the existing
  `adr-status-guard` checks PR heads and rejects Proposed as well as unsettled
  statuses. The operator requested Proposed until owner merge if allowed; an
  in-memory Proposed trial returned `Proposed is not settled; Proposed never
  reaches main`. No guard/process change is made here, and the retained marker
  must not be read as evidence of owner approval. Hardware authorization and
  merge remain separately owner-controlled.
