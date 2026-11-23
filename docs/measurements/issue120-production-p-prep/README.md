# Issue #120 production preparation — historical v1 stage

The following describes the original non-authorizing preparation at74917fe.
The owner subsequently adopted520s/N22155167 on2026-11-21 and authorized one P1
then conditional one P2. The [adoption precommit](../issue120-production-p-adoption-plan.md)
now describes explicit v2 metadata,durable consumption/external completion and
the renewed source-matched host proof. Its pre-run host13.282386s and reserves
31.525745s/41.525745s supersede the historical table below;conditional pessimistic
cap margin was positive9.414s in that scenario. This preparation update did not
claim a production run; the later [actual pair](../issue120-production/2026-11-21-p/README.md)
is published separately.

The original stage was board-free only. No board/device/tt-smi access or real
runtime P session occurred. Its separate startup sandbox invoked the unchanged
native with no device nodes/discovery,stopping at No chips detected before board
access. At that stage,520s/22155167writes was an **owner recommendation only**,
not count adoption or execution authority. The
[precommitted invariants](../issue120-production-p-prep-plan.md) govern this work.

## One selection and a distinct receipt

Only enodia/tt/bench/issue120_production/production_selection.json supplies the
runtime target/count pair. To consider an owner alternative,edit that pair and
run `uv run python tools/prepare_issue120_production_admission.py --write`.
The generator reads current build/native proofs but never executes native.
It recomputes rounding/model/margins and rejects inconsistent N rather than
silently adjusting it. A changed selection invalidates the old receipt/session;
regeneration and precommit/validation are required. Numerical fields repeated
in generated receipts/reports are derived,not independent configuration knobs.

The new compatible-production-admission-v1 path is separate from unchanged
legacy exact-provenance,C1-only and C2-only launch admissions. It binds:

- Original C1/C2 archive,manifest,primary,session and raw-environment hashes,
  ordinary artifact validation,healthy covered calibration and B4 Watcher proof.
- Unchanged native/device/kernel/CTA/library/compile/build/image/work proofs and
  original Linux6.8.0-149/KMD2.12/device0 baseline. No history is relabeled.
- Complete original-C2/current metadata Python inventories and explicit deltas,
  including the new production-admission module. No native/device CLI change.
-40slabs,max/min +/-2%,observed offsets,conditional frame projection,ceil rounding,
  chosen count/reserve decomposition/both margins and rejected physical fits.
- The hashed device-free500000-frame postprocessing report and its fixture source.
  New records capture actual receipt/session/environment/host-proof companions.
- Distinct owner_adopted=false/execution_authorized=false declarations. Technical
  eligibility and synthetic fixtures do not grant owner permission.

New P sessions bind both roles to the same boot/head/source/receipt. Changed
head/boot/receipt/selection or missing receipt rejects. Traced500000-frame native
shape independently requires new provenance even if metadata discriminators
are deleted. Primary identity must equal the canonical raw capture plus session
CPU identity. P2 checks the admitted session's contract,not the supplied record
marker,and requires an authentic P1control role/configuration/500000frames plus
exact session-manifest/program/harness/boot origin. New P container reads also
check current boot at /work; deleting a session contract cannot evade the gate.
New paired-clock anomalies require owner review. Captured native/session/raw-env
fixture flags must agree; primary-only deletion cannot enter operational roots.
These pre-hardware review repairs are tracked in Issue #187.

## Measured CPU host pipeline, not device retrieval

`postprocess.json` reports fresh Python publisher and reader processes,each
with500000compact frame rows. Control is constant-period; loaded has bounded
synthetic completion jitter to exercise thousands of histogram entries,not just
one bin. Generation is streaming; loaded transition rows500002,trace samples339.
Frames occupy64000024bytes; loaded transitions4000040bytes. Native's real
2048000000-byte device-page retrieval is **unmeasured and not reproduced**.
All clock,power,native-capture claims are explicitly synthetic fixture inputs.

| Component | Control seconds | Loaded seconds |
|---|---:|---:|
| Companion generation,excluded from cap model |4.877857 |5.384707 |
| In-wrapper P1 reread,charged before native |0 |5.950489 |
| Actual record construction including analysis/hash/copy |6.010007 |7.268395 |
| Of that,frame analysis |5.083466 |6.205263 |
| Binary header/size/hash IO |0.067730 |0.072856 |
| Runtime hash calls |0.007942 |0.008408 |
| Copy and fsync |0.034000 |0.038361 |
| Atomic record publication |0.002731 |0.012239 |
| Total measured charged host work |6.012738 |13.231123 |
| Separate fresh reader outside cap |5.952598 |7.282743 |

Hash/copy timings are subsets of construction,not added twice. Calls execute
real implementations with timing wrappers,not substitute results. The P1 reread
inside P2's wrapper is distinct from the outer launcher's precheck/final reread;
it remains charged,not moved outside the cap.

Publisher peaks:control107593728bytes,loaded114204672bytes. Fresh reader peaks:
107790336/115748864bytes. The maximum is110.39MiB. These are per-child lifetime
Linux ru_maxrss values,not summed concurrent RSS or native/device memory.
Loaded GC collections were23/2/0 for generations0/1/2 during construction and
30/3/0 during reread. Primary JSON625454bytes,well below64MiB reader limit.
Constant loaded exploratory measurement was approximately7.07s construction+
publication; jittered approximately7.29s. No severe superlinear behavior was
observed in these fixtures,but this is not a bound for arbitrary jitter,
filesystem pressure,GC,memory contention or production inputs. No measurement-
meaning-changing optimization was made. These results do not simulate the2GB
native readout or establish accelerator timing/coverage.

## Reserve update and conditional count table

The measured loaded charged host work13.231123s includes5.950489s P1 validation.
Add setup scenario2.134215s,snapshot-wait scenario1.109144s,**unmeasured native
retrieval budget10s**,and CLI/residual/guard5s. Optimistic reserve becomes
31.474482s; pessimistic reserve is max(historical36.873251s,optimistic+10s),
thus41.474482s. These are explicit budget assumptions,not measured upper bounds.
The outer reader7.282743s stays outside the existing measurement cap/watchdog
scope. Admission replay/CLI overhead and native retrieval retain uncertainty.

Using the original40slab envelope r_fast42606.092791/r_slow40351.083518writes/s,
observed offset64.033..69.164us and conditional frame envelope500.708738s:

| Target | Derived N | Fast frame margin | Slow worker |600s optimistic margin |600s pessimistic margin |
|---|---:|---:|---:|---:|---:|
|550s |23433350 |49.291s |580.737s |−12.211s |−22.211s |
|530s |22581228 |29.291s |559.619s |8.907s |−1.093s |
|520s |22155167 |19.291s |549.060s |19.465s |9.465s |

Time-only feasibility does not prove causal full-frame coverage. ±2%,offsets,
frame/reserve projections and native budget remain unbounded. Recommend owner
consideration of520s/N22155167;530s no longer fits the pessimistic budget after
including in-wrapper control validation. No target/count is adopted. Legacy550s
floor remains unchanged; only the explicit versioned path uses the selection.

## Dry-run and validation status

The fixture dry-run retains committed receipt validation,real admission/session/
launcher/control/publication/ordinary-loader/comparison paths. Only external
clean/pushed Git state,CPU wrapper boundary and synthetic native/clock/telemetry
inputs are substituted. No SDK/native executable is used. Negative cases cover
boot/head/missing or modified receipt/missing approval/dirty source/missing P1
and actual corrupt P1 binary evidence,with no P2 fixture child after rejection.
The repaired242a83d dry-run passed both CPU fixture roles,ordinary reread and
comparison. Ten negative categories rejected before a P2 child:boot,head,missing/
modified receipt,missing/failed P1,missing approval,dirty source,all seven primary
environment-summary edits and historical C1 with a rebound UUID. It used three
CPU publishers:two positive roles and one deliberately corrupt P1;zero hardware
P/native invocations. The synthetic jitter difference is test data,not prediction
or production measurement. See dry-run.json.

The device-free startup verified no devices/requests,/sys-/proc-discovery masks,
readonly root/source,nonroot,ALL capabilities dropped,and network none. The
unchanged native0a36ed60... exited2 at No chips detected,absent result refused;
container Python exited0. No card was accessed. Separate root-publisher to
UID1000 real launcher reread passed,snapshot0644. Proofs include their242a83d
code head. Full slow-inclusive regression passed6330tests/1057skipped/8subtests
in228.07s;ruff/diff/ADR guard also passed. Bounded repair follow-up review had
no included findings;the review saw a pending experimental snapshot,subsequently
replaced by the completed source-matched measurement above. Final publication-
head CI/Sonar are checked separately in the final report,not inferred from an
older head or local success. Source preservation verifies70historical files and
112native/device/header/build-input sources plus unchanged command constructors.

Issue #186 records the old C2 publication CI portability defect:an archive test
incorrectly required a local native build. Frozen-evidence/fixture identity is
now explicit and SDK-independent; live C2 admission is not weakened. C1/C2 unit
contracts characterize their frozen source inventories,not new P source. No
historical/native bytes are changed to make CI pass. All fixes remain Refs.

## Owner approval checklist (one screen; not approval)

- [ ] Choose target/N in the single selection,or accept the recommendation only
      after reviewing conditional19.291s frame/9.465s pessimistic cap headroom.
- [ ] Review13.231s measured charged host/110.39MiB peak;2GB native readout is
      unmeasured,10s budget plus10s stress slack is not a bound.
- [ ] Confirm exact final precommit,receipt/source/native/device/build/image/
      calibration/baseline hashes,green full tests/CI and Sonar0/0/OK.
- [ ] Confirm device-free startup/publication and real fixture E2E/negative results.
- [ ] Authorize once only P1 then conditional P2,500000frames,Watcher off,device0,
      600s native/outer caps,in one new boot-bound operational session.
- [ ] Fresh actual baseline/digests/zero containers/host quiet and immediate recheck
      are still required after approval; this stage collected none.
- [ ] Review threshold-free interval/difference predictions in the plan and PR;
      no latency threshold or reliable maximum/500s rate/thermal bound exists.
- [ ] P1 all exits0/full integrity/diagnostics/power/reread/healthy return releases
      P2; source/boot/receipt mismatch or failed P1 blocks it.
- [ ] Stop on unexpected warnings/errors,timeout/integrity/provenance failure or
      loaded not_covered;no retries/count extension,at most one abnormal-only reset.
- [ ] Budget approximately533..543s P1 and552..591s P2 plus gates/outer rereads;
      accept remaining thermal/AICLK/rate/memory/GC/IO/native-extrapolation risks.
