# Issue #30 acceptance evidence

Board-free completion assessment for PR #195. All six stated criteria are
satisfied within their explicit scope; this is not new board authority, an
assertion of general health/FPGA compatibility or a merge decision. Historical
source-study open ledgers describe that study's scope/date, not a replacement
for the later native result. Measurement records/archives/binaries/image are
unchanged by this assessment.

## Acceptance matrix

Paths are repository-relative. Line ranges refer to the preserved/current files;
commits identify the original evidence and subsequent completion clarification.

| AC | Disposition and file:line evidence | Record / commit |
|---|---|---|
| 1. Pinned source/current datamover, runtime training, receiver obligations; unsettled claims open | Satisfied. `docs/measurements/issue30-source-study.md:16-39,60-83,85-203,205-288` pins T/UMD/image and traces retained deprecated legacy CCL vs persistent TT-Fabric, runtime discovery/firmware cooperation, handshake/shared fields/ack/flow control/header. The source does not establish a guaranteed host first-training trigger or sufficient FPGA peer contract; those are explicitly open, not asserted. | Study introduced `8649825`, final source-study revision `90b5656`. Actual links/transfer supplement but do not settle first cause/onset: `docs/measurements/2026-11-25-issue30-ab-v2/README.md:14-38,74-77`, `2d2dbb9`. |
| 2. Precommitted first-transfer steps/predictions/stops/meaning | Satisfied. `docs/measurements/issue30-first-transfer-plan.md:25-46,279-385` defines bounded official phases, predictions and interpretation/stops. Later scoped revisions `:540-673` were committed before the renewed run, including fresh H1 and five-read DRAM gates. No retroactive rule change is used to accept the run. | Initial plan `8649825` precedes board execution; renewed binding `0922bb1`, DRAM owner correction `f8703fb`. Native record `2d2dbb9` names executed head `f8703fb`. |
| 3. Content-verified first transfer or recorded bounded stop; <=60s/Watcher | Satisfied by transfer, not only the earlier stop. `docs/measurements/2026-11-25-issue30-ab-v2/README.md:14-38,40-72`; `AB-result.json:937-1366` in the same directory binds accepted A/B, native exit/cleanup, XML/native fingerprints and source-bound five-packet checks. A1.183s/B6.032s through cleanup, Watcher requested; B5 complete reviewed dumps. No printed raw packet/status/byte counters are invented. | `2d2dbb9`; `AB-native-and-telemetry-evidence.tar.gz` in that directory retains byte-exact native stdout/stderr/XML/Watcher/verdict/review. Pinned `test_basic_fabric_smoke.cpp:74-175` and RX PRNG checks are documented there. |
| 4. B3 source trigger answer plus jitter measurement OR reason still unmeasured | Satisfied via explicit unmeasured branch. `docs/measurements/issue30-source-study.md:290-317` answers yes: local progress/idle and periodic paths, BH master/subordinate coordination, full vs no-sync/debounced service. `docs/design.md:384-411` explains missing interval/helper timestamps, overhead controls and sustained samples, and proposed B3-J0 measurements/first-run40-50-60s/Watcher/new approval. No jitter value or causal attribution inferred from smoke times. | Source revision `90b5656`; this completion clarification (file history identifies its commit). Related open status `docs/open-issues.md:20`. |
| 5. design3/open-issuesB1/B3/B4 updated to established facts/open work | Satisfied. `docs/design.md:362-428` distinguishes content verification from unknown firmware/FPGA/timing and specifies B3/B4 prerequisites. `docs/open-issues.md:18-21,26-45` records verified bounded TT-to-TT smoke, preserved DRAM stop and remaining timing/loopback/stage3 scope. | Native factual update `2d2dbb9`; B3/B4 completion clarification in this document's carrying commit. B4 smoke durations are explicitly not inter-card latency. |
| 6. New code board-free tests | Satisfied. `tests/test_issue30_dram.py:22-64` covers five normal, every-position abnormal, missing/short/stalled evidence and refusal before mapping; `tests/test_issue30_morning.py:26-89` covers renewed H1, authority and canonical native evidence. Wrapper/kernel/layout controls include `tests/test_issue30_capture.py` and `tests/test_issue196.py`. | `f8703fb`:173 focused controls; full7026 passed/1057 skipped/8subtests before execution. `docs/measurements/issue30-g0/host-input-qualification-v12.json` binds RED/CPU proof/source qualification; `2026-11-25-issue30-ab-v2/README.md:3-10` records matching CI/Sonar. Current completion validation is recorded in PR/report. |

## Evidence-consumption hardening

The follow-up host-only qualification is [receipt v13](measurements/issue30-g0/receipt-v13.json).
Shared verified readers cover judge/fresh A/B admission; raw bytes bind all four
telemetry roles, identity, values, time order and ARC progression. Strict result,
diagnostic and empty-container checks preserve fail-stop semantics. Exact offline
compiler provenance and unambiguous historical authority are retained.
[Replay procedure and actual-format inventory](measurements/issue30-evidence-replay.md)
and [offline proof](measurements/issue30-g0/evidence-replay-v13.json) record four
unchanged current-baseline conclusions and two refused copied-data mutations.
`tests/test_issue30_evidence.py` and `tests/test_issue30_raw_evidence.py` provide
positive/single-mutation controls. No old receipt or measurement is replaced.

T is tt-metal v0.75.0 commit
`d9a68815f5fcf08a5bfbffb6f1f811823fba8edd`; UMD gitlink is
`bc77ac2f7382ffeb9fa5430055f0854d53979489` (not the host Python UMD package).
The native record distinguishes literal observations from source-bound test
semantics. Jitter/latency follow-ups do not extend the completed smoke scope.

## Residual limits and follow-up boundary

- Actual per-core ERISC versions and ETH heartbeat health remain unknown;
  bundle/ARC progression/link/content checks do not establish them.
- The11/25 single DRAM false/raw0x0 then true/0x5555 observation has unknown
  cause; preserved stop and later scoped20 normal raw readings are both true
  records. Three operator-reported later readings have no raw snapshots and
  are report-only. No general intermittent-fault immunity claim.
- External FPGA peer compatibility, sufficient receiver contract and sustained
  real-rate ingest are unverified. TT-to-TT smoke is not those guarantees.
- B3 routing/context-switch jitter and B4 defined-boundary latency/jitter are
  unmeasured. Proposed B3-J0/B4-L0 controls/limits are in design.md §3; they
  require board-free instrumentation/calibration and new owner scope.
- #13 is same-card port loopback, not the completed inter-card unicast; any
  topology/recabling/workload needs its own approval. #105 stage3 still needs
  its own AC, real-rate frame input and B5/B6 resolution. No new issues or board
  runs are created by this document. The owner subsequently filed #197 for B3,
  #198 for physical DRAM-cause investigation and #199 for inter-card RTT/latency;
  none grants further board authority.
