# Issue #120 C2 — one unchanged-count large calibration

## Authority, provenance and result

Owner approval covered only `production-v1-calibration-large`:32768frames,
1572864writes,32768 CPU operations and4096bytes/write,device0(serial411c,
PCI0000:06:00.0),Watcher off,native/Docker60s caps,one invocation. **P was not
approved or executed.** The [C2 plan](../../issue120-production-c2-plan.md)
and ADR-0020 compatibility refinement were precommitted in1358694,then the
C2-only v3 receipt/metadata implementation in58ae0d2 and final repairs in
**f5d46fbfe00234acb92db1d2f9759444ae42bfbf** before hardware.

Native SHA256 remains
`0a36ed6011a8cb71773545751f98767961e1392ffb16244f2e07e4fe709234e6`.
Device kernels/CTA/library/compile proof,effective compiler inputs,pinned image,
plan/work units,command constructor,protocol,coverage evaluator and runner are
unchanged from C1. Only runtime.py/recording.py/sessions.py changed among the
19 production Python files. Complete before/after maps and inverse transforms
restoring all three C1 source hashes are in the
[preparation proof](../../issue120-production-c2-prep/native-preservation.json).
C1 was not relabeled:its native/record/session/environment/commit retain their
original bytes. `compatible-pair.json` artifact-validates both identities and
captures,declares the reviewed transition,and grants no execution authority.

Issue #185 tracks pre-hardware review repairs:actual session-companion binding,
recursive rejection of nested/uninventoried importable sources,and mandatory
provenance despite phase/contract/plan deletion. Native work counts select C2
independently of removable metadata. An additional escaped-path RED restricted
analysis CLI input to program-owned records. Two Sonar findings at58ae0d2 were
fixed in code,not accepted/suppressed. Final f5d46fb had6286passed/1057skipped/
8subtests,166focused tests,ruff/C901/diff/ADR guard,nine successful CI checks,
and public Sonar0issues/0hotspots/gateOK. Immediate same-head device-free
publication passed root publisher to readerUID1000,snapshot0644,launcher0.

Fresh Linux6.8.0-149-generic/KMD2.12.0,firmware/limits and all identities matched.
Host quiescence and zero containers were checked twice,including immediately
before launch. No tests/builds/scans ran during measurement. New boot-bound
C2-only session:`1d8690df2f1d40249d1e8f2aaa6b19fb`,manifest SHA256
`8a235154c995a5d36f6e500014c681d2de6ffefb8a088d7c71afb9833cbffb4f`.
Attempt:`20261121T06165291396130455596811034Z`.

- Native/container/wrapper/launcher:**0/0/0/0**; no timeout/reset/retry.
- All32768 attempted/produced/consumed frames passed identity/checksum/work.
  Overflow/drop/abort/paired outliers:**0/0/0/0**.
- Coverage:**covered**. First-through-last causal lower bounds:
  32767transfers,134213632bytes,1073709056 CPU operations. Whole-worker totals
  1572864writes/6442450944bytes/51539607552 CPU operations are not all assigned
  to that frame interval.
- Only seven lines of the three accepted warning classes; unexpected0.
  Mailbox/expectation32770 and expected peer routing matched.
- Power snapshot complete,error-free; bounded timing eligibility true,not
  production acceptance or a per-frame wall-time guarantee.
- One named container exited0 and was destroyed. Survivors0; both cards returned
  800MHz,healthy DRAM,zero uncorrected GDDR errors. No P/reset/retry followed.

## Watcher-off interval references, not a control comparison

| Metric | Consumer ticks | Qualified envelope-weighted microseconds |
|---|---:|---:|
| P50 |1349991 |999.993333 |
| P99 |1350851 |1000.630370 |
| P99.9 |1351087.234 |1000.805359 |
| Maximum |1351287 |1000.953333 |

There are32767 intervals,so P99.9 meets the existing20000-sample minimum.
P99.99 remains insufficient. Conversion uses approximately1350MHz weighted
over the observed host program envelope,not exact frame wall-clock calibration.
C1/C2 are both loaded calibrations; there is no same-session control difference.
At the C2 publication, the production result slot in design.md section12 was
empty. The later separately approved [P1/P2 pair](../2026-11-21-p/README.md)
now supplies that result; it does not change this calibration's scope.

## Worker rates, drift and host/card overlap

`reference-analysis.json` retains each adjacent-sample count/time calculation,
OLS diagnostics and both original records' distinct origins.

| Observation | C1 small | C2 large |
|---|---:|---:|
| Trace samples / adjacent slabs |17/16 |25/24 |
| Worker seconds |25.305179491 |37.738006537 |
| Inclusive writes/s |41437.208551 |41678.513105 |
| First committed-write observation |69.164us |64.033us |
| Main-return bracket after worker start |0.602707020..204s |0.604787483..646s |
| Committed prefix in bracket |24974 |24863 |
| Slab minimum/maximum writes/s |41174.575019/41720.916452 |41402.335901/41770.679207 |
| Slab population standard deviation |165.132273 |95.817168 |
| Slab coefficient of variation |0.398505% |0.229894% |
| First-half / last-half slab means |41575.029346/41300.852775 |41653.669799/41703.887187 |

The first slab has65535writes; subsequent full slabs65536. Brackets are184ns
and163ns wide,not exact API-return counts. They measure committed host submits,
not CQ retirement/device completions. Work was already progressing during the
approximately0.603/0.605s main enqueue APIs; subtracting the whole API duration
as dead-worker lead is invalid. C1 trends modestly slower,C2 modestly faster;
there is no identified common long-time trend or500s rate bound.

Within-run OLS intercepts are−19.473ms/+17.619ms,slopes24.125288/23.983243us/write,
max absolute residual27.454/17.579ms. A two-total-size fit gives0.439525s plus
23.713736us/write. These inconsistent intercepts are descriptive,not measured
fixed startup costs. Frame/write counts change together and runs are not
replicated; no separate CPU/PCIe/frame cost or warm-cache/thermal effect is
identified. Use the observed first-commit delays and explicitly assumed rate
envelopes rather than treating any fitted intercept as physical lead.

| Host/card observation | C1 seconds | C2 seconds |
|---|---:|---:|
| Qualified first-to-last card completion span |16.383000076 |32.766999981 |
| Enqueue-begin to terminal-observed host upper bound |17.091292107 |33.476738576 |
| Mixed-clock/startup/observation residual |0.708292031 |0.709738595 |
| Worker continues after terminal observation |8.214048312 |4.261336851 |
| Conditional500k program envelope projection |500.707294337 |500.708738312 |

The residual is not an independently measured fixed device delay. Frame coverage
comes from causal map/frame validation,not enclosing host spans. C2's worker
37.738s is within the precommitted36.96..38.98s scenario; its33.477s host envelope
is near the conditional33.55s forecast. Neither forecast was a pass threshold.

## Cap charging and postprocessing identification limits

Worker and frame execution overlap. Use setup+max(worker,terminal)+drain+
postprocessing,not worker+frame durations. Retrieval,first Python analysis,
snapshot waiting and in-container publication remain inside the Docker cap;
launcher reread remains outside under its existing watchdog. No work was moved
outside the cap and no timeout/count changed.

| Component | C1 seconds | C2 seconds |
|---|---:|---:|
| Execute start to program enqueue begin/setup |2.134215262 |1.932292359 |
| Worker end to drain end |0.000382307 |0.000204620 |
| Of that,queue Finish calls |0.000198523 |0.000081198 |
| Post-drain native retrieval/verify/close/publication |0.155181941 |0.136651710 |
| Python frame analysis |0.200742569 |0.395928687 |
| Frozen snapshot wait |1.109143950 |0.000690693 |
| Native execute envelope |27.595119929 |39.807224116 |
| Container start to native execute |0.364506982 |0.594646021 |
| Unseparated post-execute residual |0.139565679 |0.118470716 |
| Container start to die |29.409079368 |40.916960177 |
| Container create to destroy |29.500588020 |41.010660169 |

Lifecycle time is a lower bound on the charged Docker-client interval,not an
exact timeout stopwatch. C2 leaves18.989s relative to60s on that lower-bound
timeline,not a certified cap margin. Startup import/admission,CLI startup/cleanup,
record encoding/copy/hash/session capture and exit have incompletely separated
costs. C2 provenance differs explicitly; extra admission replay can affect the
pre-native component. Snapshot waiting is polling-phase-dependent,not per-frame.

A two-point Python fit is5.556451ms fixed+11.913215us/frame,predicting5.962164s
at500000frames; all-linear C1/C2 scenarios are6.126177/6.041392s. This is a
conditional extrapolation,not a proven slope. Native's fitted slope is negative
(−1.130996us/frame),predicting−0.391786s at500000:reject it. Observed native
0.137..0.155s cannot identify a scalable fixed/per-frame split; all-linear
scenarios2.085140..4.735777s are not bounds. Setup,snapshot and residual fits
also yield invalid negative long-run predictions and are not used. Two points
cannot distinguish frame/write/chunk costs,caching,scheduler variation or
500000-frame Python memory/GC/I/O scaling. Per-frame fields remain explicitly
amortized,not measured proportional coefficients.

## Proposed production counts — not adopted or authorized

Primary planning uses all40 adjacent slabs,not merely the two inclusive means:

- `r_fast=max(slab)*1.02=42606.092791writes/s`;
  `r_slow=min(slab)*0.98=40351.083518writes/s`.
- `L_fast=64.033us`, `L_slow=69.164us`;
  `N=1+ceil(r_fast*(T-L_fast))`,
  `t_slow=L_slow+(N-1)/r_slow`.
- Conditional frame envelope `D=max(C1,C2 projections)=500.708738s`.
- Optimistic post/setup reserve20s; pessimistic stress reserve36.873251s carried
  from the historical B4 all-linear model. The latter used different native/
  Watcher conditions and is not a same-instrument estimate or measured bound.
  The former includes setup/retrieval/analysis/snapshot/residual/guard assumptions,
  not a validated500000-frame cost. C1/C2-only all-linear scenarios fit inside20s.

| Fast total-worker target T | Proposed N | Fast frame margin | Slow worker |600s margin,R20 |600s margin,R36.873 |
|---|---:|---:|---:|---:|---:|
|550s |23433350 |49.291s |580.737s |−0.737s |−17.610s |
|530s |22581228 |29.291s |559.619s |20.381s |3.508s |
|520s |22155167 |19.291s |549.060s |30.940s |14.067s |

Both cap-margin columns use the slow-rate worker; only the reserve changes.
With the fast worker and20s reserve,the optimistic rate+reserve margins would
be30/50/60s. Every rate/reserve/lead/frame projection is an assumption,not a
confidence interval or guarantee. The separate CLI's inclusive-rate-only
scenarios give different counts; they are retained in compatible-pair.json as
diagnostics,not adopted or mixed with this more conservative slab model.

Under the slab/R36.873 model,coverage requires at least21333242writes and the
cap permits at most22722772; the proposed520s count lies between. This is a
time-only feasible interval,not a substitute for actual causal frame coverage.
At that N,
conditional coverage would fail above44247.618writes/s; the cap would fail below
39343.130writes/s,all else assumed. Neither long-run rate limit is established.

**Recommend owner consideration of520s /22155167writes**,not550s.550s fails even
R20;530s leaves only3.508s under the stress reserve.520s trades10s of projected
frame headroom for10.559s more slow-side cap headroom,leaving19.291s/14.067s.
This is a risk-balanced candidate,not a selected safe count. A500s thermal/
AICLK/rate excursion or nonlinear postprocessing could still invalidate it.

The current550s floor in records.calibration_proposal and plan's550s candidate
remain unchanged. A520s choice requires **separate board-free implementation,
tests and precommit before any P approval**:a versioned compatible-calibration
production path carrying an explicit520s target,exact model/count and budgets,
not silently relaxing legacy exact-provenance admission. Keep old receipts and
records' semantics; extend ADR-0020 if the successor provenance/admission contract
changes. Actual native/device changes would require stopping for new approval.
Current v3 admission permits C2 only; there is no production receipt.

## What separate P1/P2 approval must contain

This is a checklist/proposal,**not approval or an executable receipt**.

1. Precommit a versioned production receipt with both original primary/archive/
   session/environment hashes; C1/C2 native/device/CTA/library/compile/build/image
   proofs and raw baseline; explicit original harness/Python maps and any reviewed
   metadata-only successor transition; healthy artifact-recomputed execution,
   coverage and timing predicates. No historical record rewriting or unreviewed
   source-identity exceptions.
2. Include exact chosen N,target,work units32768ops/4096bytes per write,formula,
   rounding,all40 slab extrema/2% assumptions,lead values,frame envelope,reserve
   components and both margins/rejected models. Bind500000frames,Watcher off,
   device0/serial411c/PCI0000:06:00.0,600s native/outer caps,and identities
   `production-v1-control-500000` then `production-v1-loaded-500000` in **one new
   boot-bound session**. Control declares the selected configuration but executes
   zero load work; loaded executes exactly the selected fixed count. No extension
   to make coverage pass and no automatic retries.
3. Precommit threshold-free predictions:both interval distributions center near
   the1350000-tick period; P99 around the observed1350851-tick neighborhood and
   P99.9 around1351087ticks from C2 are hypotheses,not bounds. Loaded-minus-control
   central/quantile differences are hypothesized near0 with no established sign
   or uncertainty. No meaningful maximum bound follows from two short loaded
   calibrations. P1 supplies the missing contemporaneous control. Publish actual
   P50/P99/P99.9/max and paired diagnostics without inventing latency thresholds.
4. Require clean/pushed tested head,nine green CI checks,Sonar0/0/OK without
   acceptance,immediate device-free publication,fresh baseline/all digests,
   explicit device0,zero containers and host quiescence before each invocation.
   No heavy concurrent work. A changed boot needs a new session,not reuse.
5. P1 may unlock P2 only after native/container/wrapper/launcher0,ordinary persisted
   reread,all500000frames complete/checksum/work-valid,zero abort/drop/overflow,
   healthy clock/routing/mailbox/power diagnostics,only the three baseline warning
   classes,no unexpected warning/error,and zero containers/healthy800MHz return.
   Control coverage is not_applicable,not a loaded-coverage requirement. New
   paired-clock anomalies or unexplained evidence require owner review,not a
   manufactured timing pass threshold. Recheck host quietness and session/boot/
   source/configuration equality immediately before P2. A P1 failure stops P2.
6. Stop on any new warning/error,identity/boot/device-open anomaly,timeout,
   execution/publication/validation failure,incomplete/corrupt frames,or P2
   not_covered. Record original evidence; no automatic acceptance/count change/
   retry. At most one reset only after timeout/abnormal exit,then stop. Return to
   zero containers/800MHz/healthy DRAM. Approval applies once to the exact head.
7. Expected conditional P1 runtime approximately501s frames plus setup/tail/
   postprocessing,roughly521..538s with20..36.873s reserve. Proposed P2 roughly
   520..549s worker plus reserve,about540..586s. Two runs roughly18..19minutes,
   plus fresh gates/analysis. Applying the loaded-derived frame projection to
   control is itself an unverified assumption. Two600s caps allow20minutes,with existing
   separate launcher watchdogs potentially longer on failures. No cap movement.
8. Explicit residual risks:only25s/38s workers observed;500s thermal/AICLK effects,
   host scheduling/rate changes and causal full-frame coverage are unbounded.
   Power samples are envelope-weighted,not exact per-frame clocks. No no-Watcher
   control exists yet.500000-frame retrieval/Python memory/GC/hash/copy/publication
   and launch/exit costs can be nonlinear; two correlated size points do not
   identify them. Review the target/head/receipt risks with the owner before P.

## Archive and verification

C2-evidence.tar.gz contains11 byte-exact original payloads:primary,frames,
transitions,native result/execution/stdout/stderr,raw env,original/frozen power,
and the actual session-manifest companion. C2-manifest.json pins sizes/hashes.
Original private native output permissions were not changed. Wrapper/native log
normalizations are separate derivatives; private raw wrapper/events retain their
hashes,not machine/user-specific paths. Ordinary loader replay succeeded without
changing original/archive bytes. C1/B/B4 archives and measured package sources
remain unchanged by this publication. Tests replay actual C2,compatible C1/C2,
and reject coordinated provenance deletion/null session digests on archive copies.

P remains unexecuted. No count/floor/production authority was adopted here.
