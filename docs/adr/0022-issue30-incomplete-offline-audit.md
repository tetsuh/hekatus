# ADR-0022: distinguish incomplete offline audits from live evidence admission

- Status: **Accepted** — 2026-11-27
- Date: 2026-11-27

## Context

Issues 200/201 harden the host-only Ethernet wrapper. The original capture and
observation trees were restored from the owner's November 27 archive. The twenty
raw telemetry snapshots of the renewed November 25 unit were kept only in private
temporary task storage, omitted from that archive and lost on reboot. Public
snapshots explicitly remove host metadata and are not byte-exact substitutes.
Earlier replay proofs remain historical evidence; they cannot prove a new replay
of bytes no longer retained.

The operator explicitly directed an incomplete offline audit without relaxing
live raw binding, substituting public derivatives or modifying original records.

## Options considered

- Reconstruct raw files or replace their fingerprints with public-derivative
  hashes: rejected; that would manufacture evidence and weaken live admission.
- Report all verification blocked: rejected by the operator because original
  native/capture/review artifacts and non-raw checks remain independently useful.
- Separate a read-only, explicitly incomplete audit from live judgment: chosen
  by the operator, retaining fail-stop live checks.

## Decision

1. Live judge, fresh-A reevaluation and B admission retain the raw series proof.
   Missing original raw refuses. There is no live CLI or approval flag that
   disables that proof.
2. The offline replay tool may report non-raw conclusions using the shared
   verified capture reader and evaluator. Its own audit envelope reports each
   unavailable check and `validation_complete=false` for the renewed unit,
   never a fully passed or admission-eligible result. It does not publish live
   verdicts or change original result slots. A v1/B v1 retain their full
   historical-scope checks because they precede the five-reading rule.
3. The partial audit still verifies manifest fingerprints, observations/review,
   native XML/markers/Watcher/diagnostics/topology, strict types and canonical
   fresh-A recomputation for B. Shared pure comparison checks remain common;
   only raw-dependent proof is unavailable. Public selected timestamps may
   inform a clearly labeled timing analysis, not satisfy raw binding.
4. DRAM freshness counterexamples use synthetic raw fixtures. The other four
   findings use coherent mutations of temporary copies of original evidence.
   Restoration must recover each positive conclusion. Original files are
   fingerprinted and checked unchanged even on refusal.
5. The owner approved the **60-second** maximum pre-launch series age on
   **2026-11-28** in [Issue201's owner decision](https://github.com/tetsuh/hekatus/issues/201#issuecomment-7233987817).
   It applies to both phases at admission and immediately before container
   create/start. Approval itself changed no value or implementation; subsequent
   validator corrections enforce that same60-second rule at the prepared host
   request boundaries. This fail-stop
   parameter is not permission for another board run or an extension of workload
   caps, resampling or retry. Exact historical create/start times are not invented
   from a boolean `started`, XML or elapsed duration.

## Consequences

The renewed conclusions can be compared only with the explicit raw-proof
exclusion. Neither the archive nor the earlier accepted transfers is rewritten.
Loss of the raw snapshots is a disclosed retention gap, not a validator pass.
Future measurements need durable retention of all fingerprint-bound raw inputs
and actual capture/create clocks; temporary task storage is not an archive.
The hardware scopes remain consumed, the 40/50/60-second caps remain unchanged,
and ADR-0018's attempt-bound atomic publication is not weakened.

## Status history

- 2026-11-27: Operator directs the separate incomplete audit in the Issues
  200/201 task. Accepted records the carrying PR decision, effective at owner
  merge, not an independent approval of the then-proposed freshness value or hardware.
- 2026-11-28: The owner separately approves the existing60-second parameter in
  [Issue201 comment7233987817](https://github.com/tetsuh/hekatus/issues/201#issuecomment-7233987817).
  This is parameter approval, not new hardware authority; the value and enforcement
  at admission and pre-create/pre-start in both phases remain unchanged.
