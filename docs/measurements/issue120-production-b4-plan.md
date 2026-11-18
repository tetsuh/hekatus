# Issue #120 B4 — renewed v2 B unit precommit

The owner reauthorized B1 followed by B2 only after a healthy B1, using the new
harness, v2 record and a fresh session. C/P remain unauthorized. This plan does
not reinterpret/reuse the previous B1 invocation: its launcher exit2 and original
bytes remain in the 2026-11-18-b archive.

## Invariants before the new identity tests

Use new canonical launch identities, not new names for existing output files:
`production-v2-control-watch-b4` and `production-v2-load-watch-b4`. This changes
Python plan/catalog identity only. Frame work, budget, CQ/subdevice configuration,
wire/native/kernel/CTA/digests, caps, output transport and v1 record interpretation
remain unchanged. Archived old labels remain readable but are not current launch
choices. Control/loaded configuration digests must still be identical.

## Preconditions, in order

1. Finish B3's Sonar code corrections without acceptance/suppression. Require
   clean/pushed final head, its green CI and public Sonar0issues/0hotspots/OK.
   The B4 identity/plan commit must itself satisfy the same gates before hardware.
2. At the exact board-bound head run
   `uv run python tools/check_issue120_publication_e2e.py`. Require actual
   root-publisher/unprivileged-host launcher readback, readable snapshot,
   retained private native transport and no surviving fixture container.
3. Fresh environment/provenance and host-load preflight; no surviving container
   or heavy host process. Never run tests/builds/scans while B is active.
4. New immutable runtime session for both cases, binding final harness head,
   source/program/image/configuration and current boot. Do not reuse the old
   session, control or case directory. A reboot creates a new boot/session,
   not permission to resume an existing one. Recheck baseline before B2.

## Baseline and fixed execution

The owner accepted Linux6.8.0-149-generic/KMD2.12.0 on2026-11-18. Compare against
fresh values, not run8's different Linux/KMD environment. Everything else must
match the already published baseline: device0 p150a, serial000004033191411c,
PCI0000:06:00.0, firmware19.6.0.0 and all component/limit values, pinned image
sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621,
native bd0ac1384aeebcce715a9cb7527211453ef3378f28e868bbc85080fe7bc4c3c0,
existing kernel/CTA/compiler-proof and library digests, idle800MHz and healthy
DRAM. Both detected cards must be healthy; only device0 is opened for execution.

| Row / new plan identity | Fixed work | Native / Docker cap | Watcher |
|---|---|---|---|
| B1 `production-v2-control-watch-b4` |4096 frames, no host submitter, same configured262144 repeat capacity |60s /60s | on |
| B2 `production-v2-load-watch-b4` |4096 frames,262144 writes of4096 bytes,32768 CPU operations per write |60s /60s | on |

B2 totals8589934592 CPU operations and1073741824 submitted bytes. Submission is
not an in-window completion count. No count/period/chunk/cap changes or retries.
Each wrapper generates a new collision-safe attempt ID after the precommitted
logical identity, under its new session/case; do not override that transport ID.
Named container events, direct statuses and `docker ps` before/after are retained.
Take fresh post-run800MHz/DRAM/zero-error and no-container evidence after each row.

## Predictions and branches fixed before measurement

B1 should verify all4096 attempted/produced/consumed identities/checksums/work,
zero overflow/drop/abort, correct runtime peer/CTA distributions, terminal token
4098, healthy fresh Watcher and successful native/container/wrapper/**launcher
post-child reread**. The v2 snapshot should recognize its own explicit source,
without the old provenance error, and supply finite positive weighted facts.
Watcher still means timing_evidence=false; that alone is expected, not failure.
There is no advance interval percentile/max threshold.

Only after all statuses are directly zero, the full v2 record and artifact
recomputation pass, no new warning/error appears, diagnostics match, and manual
Watcher/post-health review passes, consider B2. Require complete frozen trace and
error-free v2 power interpretation as part of checking the repaired integration.
Otherwise preserve evidence and stop; never substitute the old control.

For B2, the old different-environment run8 rate predicts approximately6s of fixed
worker work versus roughly4.1s of reference frames plus startup/lead. This is
not a new-baseline bound. Predict advancing startup1, per-frame i+2 and tail4098,
with the same distribution/runtime routing as B1 and no corruption/overflow/abort.
Coverage is recomputed from the frame/map causal evaluator, not from those
expected diagnostics, host endpoints or submitted totals:

- `covered` plus intact healthy execution: bounded B evidence only; stop before C.
- Valid early exhaustion or a missing required fresh response: `not_covered`;
  retain the exact reason and stop, even if the total load duration seems long.
- Missing/malformed/ambiguous/torn evidence: `not_evaluable`; retain and stop.
- Any abnormal exit, timeout, new/changed warning/error, route/checksum/Watcher
  anomaly or health/container failure: stop, no B2 or later stage.

Allow only the exact three normalized #176 warning classes from the hashed
baseline proof; timestamps/PIDs/bus substitutions follow the existing scanner.
Changed wording is not automatically accepted. No automatic retry. A reset is
allowed at most once for the whole B unit, only on timeout/abnormal device
execution; a healthy device after a host-only publication error needs no reset.
Retain original text/direct statuses and report any reset explicitly.

## After hardware, not during it

Archive immutable raw records/sidecars and normalized public derivatives with
hashes. Do not publish machine names or user-specific absolute paths. Commit
with `git commit -F`, push without rebase/amend, then CI/public Sonar0/0/OK.
Do not write these Watcher reference observations into design.md section12 or
call them production timing.

Report B1/B2 interval P50/P99/max, overflow/drop/abort/paired-outlier counts and
loaded-minus-control ticks. Distinguish insufficient high-percentile sample size.
From B2 derive actual worker writes/s, load-start-to-last-frame observed span
with clock/scope qualifications, native drain-to-return and Python analysis
components. Update the B3 conditional cap table for550/530/520s with N, frame-span
margin and600s margin. Separate fixed from proportional unknown costs; no positive
headroom is a calibrated guarantee or automatic new target selection.

C preparation only: retain proposed identities `production-v1-calibration-small`
and `production-v1-calibration-large`, respectively16384frames/1048576writes and
32768frames/1572864writes, each60s/Watcher off. Use B4's observations to update
predicted durations and what the two points can identify: worker rates and causal
coverage, cold/warm lead/tail, separate native/Python/post-publication/snapshot
costs and frame/byte slopes. Two points cannot establish long-run variance or
a+bF+cN from total times alone. Precommitting C predictions/admission evidence
is not approval to execute it; ask the owner if coverage/cap feasibility remains
unresolved. No C/P launch follows this B unit.
