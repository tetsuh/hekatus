# Owner acceptance of A observations — 2026-11-24 late night

This is a **new owner decision**, not a replacement measurement or rejudgment.
The [original A record](../2026-11-24-issue30-a-discovery/README.md), its formal
`rejected / watcher` verdict, NUMA warning lines and byte-exact native archive
remain unchanged. [accepted-a.json](accepted-a.json) accepts only the observations
and binds their original fingerprints, boot, stable environment and cabling.

The owner accepts both cards reporting channels4-7 mutually UP, retrain count0;
channels8-11 explicitly unconnected; bundle19.6.0.0; no abnormal change of card
identity, firmware, 800-MHz clock, DRAM or advancing ARC responsiveness across A.
Counter/temperature variation and the separately recorded host hugepage free
count8->6 are not claimed constant. Per-core ERISC version and independent ETH
heartbeat health remain **unknown**, never certified healthy.

## Baseline warning comparison

[baseline-match.json](baseline-match.json) compares A native stdout lines13-16
with the frozen #176/#120 source proof. All four match, unmatched=0:

| A stdout line | Condition | Frozen source line |
|---|---|---|
| 13 | NUMA membind | #146 accepted run1-c018 stdout:8 |
| 14 | Adjacent hugepage fallback consequence | same:9 |
| 15 | NUMA membind (other enumerated device) | same:8 |
| 16 | Adjacent hugepage fallback consequence | same:9 |

The frozen #120 verification is
`2026-11-14-p150a-issue120-p6-api-lock-semaphore-v3-baseline-verification.json`:
its source SHA25640865499d6e466c5108f8119a7848cce04c2de17651c339905029fd27c1f01d2
and quoted source lines were checked against actual tracked bytes. #120 production
C2 stdout lines7-8 retain the same pair. Motherboard fallback is source line18;
SMC-unavailable lines22-25. These are the three owner-approved conditions (NUMA
and its adjacent fallback are one condition), not a broad warning allowlist.

The [#176 owner comment, 2026-11-07](https://github.com/tetsuh/hekatus/issues/176#issuecomment-6702081648)
explicitly accepts those conditions and variable timestamp/PID/TID/device/bus
values. As in #120, logger, literal body, board name, source file/line, NodeSet,
channel and error text must match; normalize only invocation variables. For two
cards, chip_id is also a variable enumerated-device ID; allowed IDs must be in
the bound two-card mapping. NUMA pairs must be adjacent and same-device. Any
nonmatching line remains a stop, not included by this decision. No warning was
hidden or removed; raw originals remain available.

## Watcher applicability and authority

Cluster-only A dispatches no application kernel, so application Watcher output
is not applicable. Its absence was not evidence of an unhealthy card. This is
a correction of A's future rule, not turning its original null into a clean log.
B does dispatch kernels: it must produce fresh, nonempty, captured/reviewed
Watcher dumps without anomalies. Missing output refuses B.

The same owner decision authorizes **one**
`issue30-b-fabric1d-smoke-watcher-v1`, unchanged official
Fabric1DFixture.TestUnicastConnAPI1DSmoke, both cards, Watcher, named container,
40/50/60s. It does not permit another A, a link probe, retry/fallback, reset,
firmware/cable intervention or host configuration. B binds this accepted A's
same boot, unchanged cabling, complete reciprocal graph and zero retrains; boot
drift refuses before device mapping. There is no new discovery to refresh A.
See [ADR-0021](../../adr/0021-discovery-observability-gate.md) and the precommitted
[B procedure](../issue30-first-transfer-plan.md#owner-decision-and-single-b-procedure-2026-11-24-late-night).
