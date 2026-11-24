# ADR-0021: admit unobservable Ethernet health for discovery only

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
4. B does not inherit this exception. B requires known acceptable health inputs
   and accepted A output confirming per-core ERISC versions and reciprocal links.
   First-core/global version text is partial evidence, not a per-core inventory.
   Unknown A fields keep B closed and require an owner decision.
5. The A approval file separately records the 2026-11-24 owner physical scope
   and this operator admission correction. It grants neither B nor a retry.
   Existing records, receipts and binaries remain byte-exact; a new source-bound
   receipt requalifies only host admission/judgment changes with CPU tests and
   unchanged native launch/watchdog proof. A is still executed only after clean,
   pushed/tested head, all CI and public Sonar 0/0/OK.

## Consequences

Discovery can observe what was unavailable before opening without fabricating
health. The official source may emit only an unscoped first-core version, or no
version at the selected log level; B can remain blocked even if A discovers
four reciprocal links. The diagnostic itself, not unknown telemetry, identifies
any minimum-version mismatch. Known failures and missing evidence never become
success. The production data plane and ADR-0018 atomic result transport are
unchanged. Owner authority remains once-only, bounded and non-transitive.

## Status history

- 2026-11-24: Operator-directed correction for Issue #30, recorded in PR #195.
  Accepted records the carrying PR decision and takes effect at owner merge;
  it does not claim the owner independently designed this gate correction.
