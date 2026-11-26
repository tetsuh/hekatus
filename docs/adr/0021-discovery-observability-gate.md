# ADR-0021: reasoned unobservable Ethernet health and owner-bound link evidence

- Status: **Accepted** — 2026-11-24
- Date: 2026-11-24

## Context

Issue #30 unit A is the unchanged official Cluster.ReportSystemHealth discovery
run. Its earlier G0 admission gate required per-core ERISC version and independent
ETH heartbeat health before device opening. Blackhole bundle 19.6 does not expose
these via ordinary telemetry; pinned UMD reads per-core versions during discovery
(device/firmware/firmware_utils.cpp:80-99). The gate was circular and stopped A
before launch at d810847. That immutable stop record remains accurate.

On 2026-11-24 the operator explicitly corrected this admission design. The owner
had authorized the official A diagnostic once, both named cards, Watcher and a
60-second total bound. The correction is the operator's decision, not an
expanded owner authorization. B/reset/retry/firmware/cable actions remain outside
that owner scope.

## Options considered

- Keep demanding unobservable pre-open data: rejected as circular for discovery.
- Fill unknowns from ARC progress, bundle version or absent warnings: rejected
  as invented ETH health and loss of observability provenance.
- Permit explicitly justified unknowns for A only; obtain observations from
  discovery and retain stronger B gates: chosen by the operator.

## Decision

1. A may admit null ERISC version/ETH heartbeat health only with an exact
   source-cited reason in before.json: Blackhole bundle <19.9 telemetry is
   unavailable. Bare null, wrong reason, or applying that reason at >=19.9 is
   refused. Observed ERISC <1.4.1 and observed heartbeat failure still refuse.
2. Identity, consistent bundle, progressing ARC watchdog, idle 800 MHz, healthy
   DRAM, loaded KMD, hugepage capacity, image/binary hashes, current boot,
   zero containers, host quiet and owner/head/environment/receipt binding remain
   mandatory. Counter/temperature observations may change without implying
   stable identity/firmware/health fields may drift.
3. A retains every reported firmware version/diagnostic; absent output remains
   unknown, never healthy. Firmware warnings/errors/refusals stop the attempt.
   H1/H2 connectivity decisions do not turn unknown firmware into a B admission.
4. **Initial operator rule, replaced by the late-night owner amendment below:**
   B does not inherit this exception. B requires known acceptable health inputs
   and accepted A output confirming per-core ERISC versions and reciprocal links.
   First-core/global version text is partial evidence, not a per-core inventory.
   Unknown A fields keep B closed and require an owner decision.
5. The A approval file separately records the 2026-11-24 owner physical scope
   and this operator admission correction. It grants neither B nor a retry.
   Existing records, receipts and binaries remain byte-exact; a new source-bound
   receipt requalifies only host admission/judgment changes with CPU tests and
   unchanged native launch/watchdog proof. A is still executed only after clean,
   pushed/tested head, all CI and public Sonar 0/0/OK.

### Late-night owner amendment — 2026-11-24 (same unmerged PR)

The owner accepts A's **observations**, separately from its original rejected /
watcher verdict. The four NUMA warning lines match the #176/#120 baseline
exactly modulo invocation variables, with source line/hash proof in the
[new owner record](../measurements/2026-11-24-issue30-a-owner-acceptance/README.md).
Cluster-only A dispatches no application kernel, so Watcher output is not
applicable. Its original verdict, null cleanliness and native evidence are not
rewritten or reissued. The owner, not the agent, approves the following revision:

6. B admission binds the canonical owner-accepted A file, original evidence
   fingerprints, identical boot/stable environment, unchanged owner cabling,
   full reciprocal H1 channels4-7 and zero retrains. A is not rerun to refresh
   links. Boot drift refuses before device mapping. Missing per-core inventory
   is not a prerequisite: reasoned pre-19.9 ERISC/ETH unknowns remain unknown,
   never healthy. Bare null/incorrect reasons, observed low versions/heartbeat
   failure and every other environment gate still refuse. This replaces rule4,
   not a general removal of health/identity checks.
7. Permit only exact #176 baseline logger/body/source-location comparisons,
   as in #120: NUMA membind plus adjacent same-device fallback, the known board's
   bus-ID fallback, and dispatch telemetry SMC unavailable. Invocation device/
   chip IDs, PID/TID and bus IDs may vary within the two-card mapping. The eight
   DOWN summaries may match only A's accepted DOWN chip/PCI/channel/core rows.
   Every other warning/error/fatal/critical condition stops. No generic warning
   suppression or source-location/board-name wildcard is allowed.
8. A's future rule is Watcher-not-applicable for this cluster-only diagnostic;
   B requires captured, fresh nonempty Watcher dumps, clean content, capture
   manifest fingerprint and explicit hash-bound manual review. Missing B output
   is rejected, including timeout before Watcher attachment, not a fabricated
   empty clean log. The original A result remains rejected under its old rule.
9. Owner authority is exactly one official B smoke, both named cards, Watcher,
   named container,40/50/60s; no second A, retry/fallback, extension, reset,
   firmware/cable/host-setting operation. B approval-v3 binds accepted-A digest,
   no A recheck and unchanged cabling separately from the initial operator
   correction. Clean pushed/tested head, all CI/Sonar0/0/OK and fresh same-boot
   preflight precede it. Hugepage files are inspected read-only, never removed.
10. A new receipt-v6 binds host-only rule/owner evidence changes; unchanged
    binaries/native watchdogs reuse immutable G0 qualifications. Original
    records/receipts/native evidence remain byte-exact. No production data-plane
    contract or application architecture is changed.


### Renewed conditional unit — owner 2026-11-25 (same unmerged PR)

After card power-off and host reboot, the owner authorizes exactly one new A,
then one Bv2 only on fresh accepted H1. Old A and consumed Bv1 are not authority.
The per-unit environment names both new attempt identities and owner date;
approval-v4 binds each one-use phase to that unit, head, boot, environment and
receipt. A accepts H1 only (channels4-7 reciprocal UP/retrain0,8-11 DOWN), not H2;
exact baseline diagnostics apply, cluster-only Watcher output not required.
B uses the canonical fresh A result at its named output slot, revalidates original
approval/head/receipt, captured before/review/after/XML/stdout/stderr, and matches
B's stable baseline including boot and unit. Non-device or stale evidence cannot
qualify. No new per-core health claim. This supersedes the old no-A-recheck/fixed
2026-11-24 acceptance path for this new unit only; historical records are unchanged.
The pending #196 registry does not itself grant permission: the new branch needs
explicit phase approval-v4 and verified fresh A before preparing a device mapping.
Named containers/Watcher/40-50-60s and all diagnostic/cleanup/health stops remain.
No retry, reset, fallback, firmware/cable/host setting operation is authorized.
A source-bound CPU/AST successor receipt reuses unchanged native compile/layout
and transport evidence; all tests/CI/Sonar must pass before fresh preflight/run.


### Owner DRAM admission correction — 2026-11-25 continuation

The13:43:22Z card...4124 false/raw0x0 then true/0x5555 observation and stop
remain immutable, cause unknown. The operator reports three subsequent read-only
samples around23:05JST (~2s spacing), both cards DRAMtrue/800MHz/ARC advancing;
no raw snapshots survive, so these are operator-reported, NOT native evidence
and NOT any of the required five admission samples. Owner explicitly continues
the SAME unused one-A/conditional-one-Bv2 scope, not a reset/retry grant.

For this renewed owner_unit only, immediately before EACH phase require five
consecutive raw telemetry snapshots at least5s apart. Every card in every sample
must report DRAMtrue/rawDDR_STATUS0x5555, zero GDDR_UNCORR_ERRS,800MHz and advancing
ARC. Keep all five raw snapshots plus monotonic times/content hashes. A missing/
failed/abnormal read, short interval or stopped ARC withholds that phase; stop
without restarting the series, resetting or consuming a native invocation.
After each executed phase also collect multiple raw observations with the same
five-reading protocol; any anomaly/change is retained and withholds the next
phase. No telemetry after an unreadable/unresponsive card or uncertain cleanup.
Snapshots/ARC/temperature are dynamic evidence, not physical identity/cabling;
exclude only the series itself from stable A/B baseline, retain it in exact
approval environment/capture/review fingerprints. No past health is rewritten.
All original unit limits, cleanup, warning, content and Watcher rules remain.
Source-bound successor receipt/tests/CI/Sonar must pass BEFORE fresh sampling
and native execution. Operator report alone cannot satisfy the five-read gate.

## Consequences

Discovery can observe what was unavailable before opening without fabricating
health. The official source may emit only an unscoped first-core version, or no
version at the selected log level; B can remain blocked even if A discovers
four reciprocal links under the initial rule. The owner amendment replaces that
circular B inventory prerequisite with scoped accepted-link evidence, without
claiming previously unknown firmware health. The diagnostic itself, not unknown telemetry, identifies
any minimum-version mismatch. Known failures and missing evidence never become
success. The production data plane and ADR-0018 atomic result transport are
unchanged. Owner authority remains once-only, bounded and non-transitive.

## Evidence-consumption clarification (2026-11-26 correction)

Existing fail-stop rules apply to captured bytes, not unverified summaries.
Execution bytes are anchored by the separate review fingerprint; a common
reader verifies consumed native/Watcher/captured-before/container artifacts
against unique capture-manifest entries before parsing. After-observation bytes
are separately anchored by the review fingerprint. Missing, duplicate
or changed consumed entries refuse. The same read-only judgment boundary serves
judge, fresh A reevaluation and offline replay. Offline replay grants no new
physical authority; live B additionally retains canonical approval/fresh-A gates.

For the five-reading unit, canonical observation storage must include
`telemetry/before-A/snapshot1.stdout` through `snapshot5.stdout`, and similarly
`after-A`, `before-B`, `after-B` in the corresponding attempt's input directory.
Those bytes must match the role-specific summary fingerprints already anchored
by captured before observations and the separate review's before/after hashes.
They are not found by searching an arbitrary digest pool. Parse exact board/BDF/
firmware identity and DRAM/DDR/error/clock/ARC values, bind aggregate ARC endpoints,
reject reused hashes across the entire A/B unit, and verify raw timestamps and
ARC increase across roles. Summary monotonic figures are retained metadata, not
an independent proof of when the raw bytes were collected. Actual raw dates
establish spacing/order; tt-smi naive dates require the recording host timezone.

The pinned UTC root gtest timestamp is inside native execution and therefore
capture. The total recorded capture elapsed conservatively encloses that event
on both sides; require before samples earlier than that enclosing interval and
after samples later. New captures supply a manifest-bound wall/monotonic clock
that narrows the enclosure and verifies claimed sample completion times. This uses native manifest-bound bytes, never filesystem
mtimes or fabricated historical capture timestamps. No deadline is extended.
Complete empty daemon and CLI listings are mandatory before launch and after
cleanup, without removing unrelated containers. Strict XML hierarchy/counts/DTD
prohibition, anchored unique ordered markers, typed integer exits and complete
within-stream baseline diagnostic matches are evidence checks, not new health
baselines. Exact frozen compiler row/source/support provenance remains required.

The operator selected the unchanged current evaluator as the replay baseline.
Old A's original rejected/watcher record is preserved: its current acceptance
is the prospective Watcher correction at lines 57-63 and 80-84 above, not a new
historical-policy profile. Jitter/latency and physical DRAM-cause investigation
remain separate follow-ups; no new observation is authorized by this correction.

## Status history

- 2026-11-24: Operator-directed correction for Issue #30, recorded in PR #195.
  Accepted records the carrying PR decision and takes effect at owner merge;
  it does not claim the owner independently designed this gate correction.
- 2026-11-24 late night: Owner accepts A observations and authorizes one B;
  amends the still-unmerged record's B admission, exact baseline warning and
  phase-appropriate Watcher rules. Original A verdict and observations are immutable.

- 2026-11-25: Owner renews one conditional fresh A then H1-only Bv2 unit; fresh
  native evidence replaces stale fixed acceptance for the renewed scope only.

- 2026-11-25 continuation: Owner preserves transient DRAM stop, accepts operator
  report as report-only, and requires fresh five readings before each phase.
