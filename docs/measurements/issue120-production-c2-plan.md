# Issue #120 C2 — precommitted compatibility and one-run plan

## Scope and immutable work

The owner approved **one** `production-v1-calibration-large`:32768frames,
1572864writes,32768CPU operations and4096bytes/write,Watcher off,device0,
60s native/Docker caps. No count/catalog/native/device-facing change and no P.
If any native or device/dispatch content must change,stop before execution.

Retain native SHA256
`0a36ed6011a8cb71773545751f98767961e1392ffb16244f2e07e4fe709234e6`,
actual device/kernel/CTA/library/compiler proof and pinned image. Reuse no old
boot/session/case. Preserve C1 and B/B4 payloads and measured sources byte-exact.

## Compatibility invariants before RED

1. Version-dispatch a C2-only bounded receipt(v3),retaining v1/v2 semantics.
   Keep a byte-exact copy of C1's original v2 admission. Root v3 in original
   healthy B4 Watcher evidence and the pinned healthy covered C1 archive/session/
   record/environment,not re-labeled copies or current identities substituted
   into old records. Explicit owner approval is still required.
2. C1 and C2 must have the same actual native,host-trace contract,device/kernel/
   CTA/library/compiler identities,image and immutable work units. Only metadata
   code in runtime.py,recording.py,sessions.py may change; pin complete before/
   after Python maps and list every changed digest. No file-set additions,
   coverage/protocol/plan/native/runner changes or generic ignored differences.
   Explicitly declare the new metadata contract separately from unchanged native
   identity. Later harness commits are retained in a provenance ledger rather
   than overwritten to create false equality.
3. Require Linux6.8.0-149-generic/KMD2.12.0 and all C1 static board/firmware/
   limits/image/tool/sampler/device facts. Compare original outer environment
   fields exactly except harness commit,and separately compare the complete
   selected stable raw environment capture,including kernel. Timestamp/thermal/
   boot/session changes are not asserted equal. Each fresh boot gets a fresh
   session; C1's session is not reused.
4. New C2 records carry a versioned environment/admission provenance descriptor,
   bound to captured raw environment size/hash and the session-verified receipt.
   New declared metadata contracts cannot lose that field on reread. Ordinary
   loading of historical C1/B4 records stays unchanged. Artifact replay always
   precedes offline pair analysis; never rewrite record program/environment to
   make the legacy exact-provenance proposal accept it.
5. Add a distinct compatible-pair analysis/proposal path,not a weakening of
   records.calibration_proposal or production comparison. It accepts validated
   C1 small/C2 large at their exact catalog counts and emits explicit original
   identities/commits,source digest transition and non-authorizing scenarios.
   Retain550s floor and current600s/500000frame rules;530/520s are unadopted
   owner-discussion scenarios. No production receipt or P admission is created.
6. A valid healthy `not_covered` C2 is retained as genuine evidence. Its worker
   and postprocessing observations remain usable as **diagnostic calibration**,
   not_covered never becomes covered/timing-eligible by compatibility or host
   trace. No malformed/not_evaluable/incomplete/warning/health result qualifies.
   Current wrapper/launcher may return2 for noneligible coverage after native0;
   distinguish this semantic stop from timeout/abnormal native failure. No reset
   just for a healthy coverage miss,no retry/count extension. P eligibility and
   any later receipt remain separately gated.

## Predictions and fixed branches

Worker forecast36.96..38.98s;conditional program/frame envelope about33.55s;
fast-side coverage margin about3.4s. A phase-reserve scenario11.338s leaves
about9.68s to60s at the slow forecast; using the illustrative20s reserve leaves
about1.02s. These are not bounds,latency thresholds or count-change authority.
Expect healthy32768frames,25 sparse samples,terminal32770 and advancing causal
coverage. If work ends first or an endpoint is missed,retain true not_covered.

- Covered and healthy:one C2 result,then stop before P.
- Valid not_covered:preserve record and diagnostic rate/cost facts,then stop.
- New/changed warning,error,malformed evidence,frame/clock/health/timeout/exit
  anomaly:stop,do not relabel/retry/extend. At most one reset only for timeout
  or abnormal native/device exit; record all direct statuses.

## Gates and execution

Before hardware require exact clean/pushed precommit,green matching-head CI,
public Sonar0issues/0hotspots/OK without acceptance/suppression,and an immediate
exact-head device-free publication check. Verify unchanged native/build receipt/
host source/device proof and v3 admission. Do not rebuild to hide drift.

Retake complete environment/provenance at the owner-approved baseline. Both
cards800MHz/healthy DRAM/zero uncorrected errors,container0,host quiescent. Set
HEKATUS_TT_DEVICE_NODE=/dev/tenstorrent/0. Check host processes again immediately
before launch. If busy,stop and report only the command name,not its path or
machine name. Create a new C2-only boot-bound session. Use only serial411c,
PCI0000:06:00.0,named container,Watcher off,60s native/outer,one invocation.
No tests/builds/scans during it. Retain lifecycle/direct statuses and verify
no containers,800MHz/DRAM after it. No C1 replay on hardware,C2 retry or P.

## After hardware

Archive immutable originals and privacy-safe derivatives;record no host name or
user-specific absolute paths. Commit/push evidence and verify full tests/CI/
public Sonar. Report C2 health/coverage/exits/warnings/reset and Watcher-off
P50/P99/P99.9/max (32767intervals meet the existing20000 P99.9 minimum),no control
difference. Analyze C1/C2 sparse rates/time drift,setup,native/Python/snapshot/
unseparated phases with confounding and identifiability limits stated.

Propose fixed production N by550/530/520s targets,conditional frame/cap margins
and optimistic/pessimistic costs;recommend a target with risks and any required
board-free550s-floor policy change,but do not execute P. Specify future receipt
contents,predictions for control/loaded distributions and differences without
thresholds,P1-to-P2 gates,expected duration and500s thermal/AICLK/rate/postprocess
extrapolation risks. Deliver a new-workdir issue120-production-c2-report.md,
its path and three-line summary,and stop.
