# Issue #120 P1 result-file attempt: bounded overlap, then conservative stop

## Provenance and scope

[Immutable record](2026-11-07-p150a-issue120-p1-result-file-v1.json), measured
with clean/pushed `75be8ec573a0af9cbed02b139b683c6d366a09a9` and native SHA-256
`1ca91e17b09db83b70b1cec1e09d69e61430d00d5bab9168a283d3503f9c2b4d`.
The precommitted identity is `p1-result-file-v1`, with device 0 only, Watcher on
and 60-second native/outer caps. All environment values were freshly captured;
both cards were idle at 800 MHz before and after. No reset or retry occurred.

## Result and prediction comparison

Native, wrapper and compute container all exited 0. The atomic result envelope
matched schema and transport identity. Exactly one result file survived; both
native streams and the direct wait metadata were retained independently.
Public native stream/result/execution companions are byte-identical copies.

All 16 payload pages were verified on the resident observer clock. The full
completion bracket `[1911527229, 2046823242]` ticks lies strictly inside the
resident window `[1641510863, 4342099876]`. The observer program remained active
until tick `7042425355`; `clock_gap_ambiguous=false`.
This is the inside-window branch explicitly allowed by the P1 prediction, not
a prediction failure. It supports bounded CQ1/CQ0 overlap even with both
sub-devices in the stall group, not sustained throughput or production timing.

The host final drain explicitly finishes CQ0 before CQ1. Its duration must not
be assigned to an unexplained CQ1 completion-wait stall. It does not resolve
the historical wait mechanism or replace P2's transfer-only comparison.
Watcher attached/detached device 0; retained observer/producer diagnostics have
no error/warning/assert/hang scan matches. No P3/P4/P6 kernel health is implied.

## Why execution stopped before P2

Raw stdout retained runtime warnings not pre-cleared in the plan:

- NUMA binding failed with `Operation not permitted`; the runtime warns of
  possible device-to-host performance impact.
- The unknown motherboard fell back to bus ID discovery. The requested target
  serial and PCI remained verified.
- Dispatch telemetry SMC buffers were unavailable. This is distinct from the
  retained external power CSV and does not establish dispatch-telemetry coverage.

The agent conservatively applied the unplanned-condition stop rule. These
warnings are not reclassified as fatal exits, Watcher corruption or failed DMA.
P2-P7 were not run; their original precommitted identities remain unused.
Both cards were healthy at 800 MHz and no container remained, so no reset was
needed or performed. No production run followed.

## Judgments and evidence limitations

Q1 is partially answered by one bounded full-payload overlap witness. Q2-Q4
remain unresolved. Ordinary unpinned CQ1 writes remain the first candidate,
now with this short-window support, but full-frame CPU/PCIe causal coverage and
mechanism-adoption gates remain unmet. No CPU-only fallback was selected.

The frozen four-sample power CSV contains 800/1350 MHz; sparse samples do not
prove program-end coverage. There is no real-time conversion or timing claim.
Native streams remain raw; generated/wrapper public text companions are labeled
redacted/whitespace-normalized derivatives with original hashes in the record.
Private originals and both prior failed attempt archives remain unchanged.
Further probes require supervisor/owner assessment of the warning conditions;
there is no automatic continuation or retry.
