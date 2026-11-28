# Issues 200/201: host-only validation corrections

No board operation is authorized. Historical records, archives, binaries, image
and receipts are immutable. The invariant inventory was written before implementation; the dispositions and
evidence below record the subsequent RED/GREEN and scoped replay.

## Invariants and inspection inventory

| Class invariant | Inspected seams / correction targets |
|---|---|
| Recursive JSON identity preserves types and rejects nonfinite values | Approval/source binding; captured-before/review; raw/summary cards and ARC; frozen policy/reasons; owner decisions; fresh A canonical result and independent reevaluation; replay baseline comparisons. `false`, `0` and `0.0` are distinct. |
| Closed wrapper schemas reject unknown fields | Environment/card/ARC/unobservable objects; DRAM series/sample/card; topology record/decision; capture clock; execution/review/manifest; native XML hierarchy/attributes. Raw telemetry has intentionally unconsumed diagnostic fields, not a closed wrapper schema. |
| Accepted text has complete grammar and declared encoding | UTF-8 JSON/native/Watcher; exact direct XML hierarchy and no namespaces/DTD; full chip/UP/DOWN rows; Test component and source location for DOWN and packet logger lines; unique native markers across both streams. |
| Time has valid syntax, calendar and bounded ordered range | Raw local timestamps; native UTC timestamp; wall/monotonic capture clocks; complete ordered Watcher dump pairs within capture; nonnegative sample completion times; DRAM age before both phases at admission and immediately before create. |
| All live judgments use verified original bytes | Capture manifests, independent reviews, judge, fresh A reevaluation and B admission. Missing DRAM raw remains a live refusal. |
| Offline incompleteness cannot become admission | Separate audit result reports unavailable raw proof, never an accepted live verdict; no raw replacement by public derivatives and no bypass option in live CLI. |
| Container actions remain attempt-bound | Fixed API/CLI daemon; returned ID start; explicit unknown-create state; no deletion of unidentified/unrelated containers. |
| Publication remains exclusive and atomic | Owned staging cleanup on pre-rename failure; no-replace rename, file/directory sync, no overwrite fallback. |
| Device mapping binds the approved PCI function | Sysfs path resolution only; mocked CPU controls, no node open. |
| Tests do not depend on runner privilege | Mock positive non-root identities; retain explicit production UID-0 refusal. |

The inventory is bounded to the Issue30 wrapper/admission/diagnostics/evidence/
topology/runtime, offline replay and compiler-provenance seams. It is not a claim
that every possible malformed evidence class has been found.

## Concrete dispositions

The listed seams were corrected; this is a bounded inspection, not a claim that
all possible malformed evidence has been found.

- DRAM: typed closed summary objects; exact raw/summary comparison; valid calendar
  timestamps; bounded ordered clocks; owner-approved60s freshness at admission, before
  create and after create before start. With no pre-start clock, native root is
  the conservative latest launch, rather than the earlier capture enclosure.
- Watcher: strict UTF-8, complete numbered start/end pairs, nonempty device rows,
  ordered dump/event times within process/capture caps; captured Watcher=1,
  fingerprint binding and independent manual review remain required.
- XML/native: closed direct unnamespaced hierarchy/known attributes, no extra
  children/DTD/alternate encoding, typed exits and bounded time observations;
  exact Test logger/body/source for fast dispatch and strict mode; unique ordered
  RUN/OK/PASSED markers, no extra test markers or stderr duplicates.
- DOWN/topology: complete ASCII row grammar, exact Test/source envelope and UID
  association with the chip header; B warning UIDs come from verified canonical
  A native bytes. Topology records/peer cores have closed typed fields. No new
  baseline warning is allowed; motherboard chip/bus binding is retained.
- Additional comparisons: recursive exact-type identity for approvals/sources,
  policy/reasons, captured-before/review, raw-summary ARC/cards/temperature,
  owner decisions, fresh-A canonical result/rejudgment and replay conclusions.
  Legacy fixture paths also validate typed links/uncabled pairs and exact800MHz.
- Additional metadata: UTF-8 JSON only, duplicate keys/nonfinite constants and
  exponents refused; closed environment/card/ARC/owner-unit/DRAM/review/execution/
  clock/manifest schemas. Optional review annotations are identified as historical
  report-only fields; they do not replace native or raw proof. Every manifest
  entry is validated and its artifact fingerprint checked; captured argv and
  unique environment assignments are phase/provenance-bound.
- Additional diagnostic/time seams: unknown logger severities cannot silently
  become informational text; normalized invocation numerals remain ASCII. ISO
  timestamps round-trip exactly, including refusal of Python3.14's normalized
  `24:00` form. Listing capture uses the same typed execution checks as judgment.

### Real-format correction caught by controls

The frozen motherboard baseline contains a literal U+2014 em dash. An initial
uncommitted ASCII-only whole-line guard incorrectly rejected this valid format;
actual replay and existing positive controls caught it. The correction keeps
that exact frozen body and instead refuses non-ASCII decimal invocation numerals
before normalization. It adds no generic Unicode wildcard or baseline waiver.
The padded raw `" 800"` grammar from the preceding qualification remains unchanged.

### Issue 200 item correspondence

| Item | Disposition / CPU evidence |
|---|---|
| Container ID start | Returned64-hex ID is used for start, inspect/kill/delete; mocked argv assertion. |
| CLI/API endpoint | Fixed shared Unix socket plus explicit CLI host and removed inherited Docker routing variables; fake hostile environment control. |
| Lost create response | Explicit unknown create status, no native start/claimed termination/clean poststate, no name fallback or deletion of unidentified containers; positive and one lost-response mutation. |
| Publication staging | Remove only owned staging on pre-rename failure; retain exclusive create, no-replace rename and directory sync. Existing staging, raced destination and a later publisher's staging after successful rename have controls. |
| Node/PCI identity | Resolve class-device and approved PCI sysfs paths only; no node open/register read; correct/mismatched association mocks. |
| Runtime record docs | Issue196 README identifies attempt-root runtime-layout.json, not a file inside runtime. |
| Current receipt docs | G0 and layout documents identify the selected successor; earlier evidence is explicitly historical. |
| RX argv addendum | Normal-root/driver relocation, fresh object/dependency slots and option ordering are distinguished from original Bv1 argv, with unchanged flags/defines. Immutable proofs are untouched. |
| Tests under root | Focused wrapper/related CPU tests run with actual UID0 inside an unprivileged user namespace; positive identity calls are mocked, explicit production UID0 refusal remains. |

All items are applicable; none is dismissed. Historical records, official assets
and previous receipts remain unchanged. Successor qualification reuses only
explicitly fingerprinted unchanged native evidence; it does not assert the
intentionally changed lifecycle is AST-identical.

## Evidence collected

- Initial RED: `uv run pytest -q tests/test_issue200_201.py` —21 failed/1 passed
  before implementation, including header-only Watcher, stale admission, XML
  namespace, typed identity, unknown fields and staging cleanup counterexamples.
- Additional RED: `uv run pytest -q tests/test_issue201_sweep.py` —9 failed/11 passed
  at that stage. Later targeted logger/ASCII controls also failed before their
  correction. The lost-create control failed before explicit uncertainty state;
  an unrelated initial lifecycle test fixture lacked a timed_out key and was
  corrected, not counted as a demonstrated production defect.
- Focused GREEN: `uv run pytest -q tests/test_issue30*.py tests/test_issue200*.py
  tests/test_issue201_sweep.py tests/test_issue196.py` —322 passed after analyzer corrections.
- The same selection passed322 with UID0 using `unshare --user --map-root-user`;
  this is a user-namespace test, not host-root hardware authority.
- Scoped original-data audit: four conclusion matches, eight coherent copy
  refusals,78 selected original files unchanged. DRAM freshness is shown only
  by synthetic raw fixtures for both phases; the lost20 raw files are not passed.
  The CLI intentionally exits3 for incomplete evidence. Public proof:
  [evidence-replay-v16.json](issue30-g0/evidence-replay-v16.json).
- Full suite, lint and exact-head CI/public Sonar outcomes are recorded in the
  carrying PR/report. This records the pre-review checkpoint; no finding was
  accepted or muted. The later convergence result and loop1 are described below.

## Analyzer corrections before merge

The operator permits regeneration of this PR's new, unmerged v16 artifacts;
PR195-and-earlier evidence remains immutable. Analyzer findings are corrected,
not accepted/muted/excluded: regex decimal classes retain `re.ASCII`; sysfs
components are built from bounded numeric node data and the fixed approved PCI
inventory, preserving node spelling (including padding); warning comparison is
factored without changing refusal conditions; test assertions and exception
boundaries are explicit. AST fingerprints use function/SHA256 records rather
than a function-name-to-hash map, avoiding a credential false positive without
changing digest values. The same controls and scoped original-data audit are
rerun. At this2026-11-27 checkpoint the60s proposal was still pending; see the
2026-11-28 owner decision below for its later approval.

## Loop1: corrections while60s was pending (2026-11-27 checkpoint)

The preliminary convergence gate at46a2754 returned NEEDS DECISION, including the
unapproved60s parameter. The operator separately authorized these independent
format/type/time corrections, not approval of that value or a review restart.

| Finding | Correction and evidence |
| --- | --- |
| WATCHER-EMPTY-DEVICE-ROW | Require non-whitespace `Device ` payload. Positive complete dump; one empty/whitespace payload mutation. Hash-bound B review shares the reader. Other logger/header/packet/firmware prefixes already route to exact parsing/comparison rather than proving payload by presence. |
| XML-DECLARATION-ENCODING | Recognize XML whitespace around equals and both quote forms, reject non-UTF-8 declarations. A/B source-shaped positives and one encoding mutation. Validate native XML before full-judge clock consumption too. |
| CLOCK-NATIVE-ORDER | Each recorded create/pre-start request is no later than native root plus1ms. Native timestamps have three fractional digits; this is one representation quantum, not drift/freshness slack. Inclusive boundary, next-float and coherent five-second controls; outer bounds and60s are unchanged. |
| DRAM-SAMPLE-CARD-TYPE | Validate each nested card object before lookup/set construction; malformed members return `dram_gate_reading`. Serial/hash types have their specific named refusals. |

One same-class path sweep covered capture admission, environment/summary/raw
validation, judge, fresh-A reevaluation, B admission, full replay and incomplete
audit. Additional uncovered sites corrected: owner `original_verdict` mapping;
discovery reported-version list; replay row/attempt types; receipt root/binary
mapping; review objects before hash lookup and observations before proof use;
direct evaluate execution root; rejection fingerprinting of malformed
environment/card containers; overflowing mandatory capture/elapsed/reported
clock integers. Existing guarded schemas were retained, not replaced by broad
exception catches. Valid stable baselines are unchanged; invalid shapes remain
rejected and cannot become discovery evidence.

- Review-driven behavioral RED: `uv run pytest -q tests/test_issue200_201_loop1.py`
  initially65 failed/8 passed before correction. Additional targeted pre-fix RED:
  `-k overflow`6 failed, `-k validates_xml_before_clock`1 failed, and
  `-k 'review_object or judge_receipt or capture_observation'`15 failed. These are
  separate chronology checkpoints, not a claim all98 tests existed initially.
- Two co-developed XML guard-preservation controls failed on an interim anchored
  implementation. The final rule extends the prior lexical refusal with XML
  whitespace; it does not admit previously refused XML-like directives in PIs
  or comments. This is implementation-regression provenance, not original RED.
- Latest focused selection (same command above, whose issue200 glob includes the
  new loop1 file):420 passed normally and at verified UID0 in a user namespace.
- Co-developed integration coverage separately binds Watcher bytes/review hashes
  and exercises the early judge XML check. It is not fabricated original RED.
- Scoped audit is rerun against copies: four unchanged conclusions and eight
  unchanged specific refusal reasons, exit3, unavailable raw still not passed.
- New unmerged v16 qualification is regenerated after full tests; prior evidence
  remains immutable. No post-correction review is authorized by this loop.

## Loop1 fixture finish and owner decision (2026-11-28)

CI found that the new binary-map positive fixture read local `out/issue30`
assets absent from a clean checkout (the same isolation class as Issue186).
The repaired test creates its two synthetic files and matching SHA256 map in
pytest `tmp_path`; its one malformed-map mutation is unchanged. No test is
skipped and production binary binding is not mocked or relaxed. Retained-original
checks belong to the existing offline audit, not ordinary tests. A fresh external
checkout with no retained `out/` is used for full-suite validation; original
`out/` contents are not moved, renamed or deleted. Logs and source identity are
recorded in the unmerged v16 qualification and the operator report.

The owner approved the existing **60-second maximum series age** on **2026-11-28**
in [Issue201 comment7233987817](https://github.com/tetsuh/hekatus/issues/201#issuecomment-7233987817).
It applies to both phases at admission and immediately before container
create/start. That approval-only update changed no value or implementation;
loop2 subsequently strengthens enforcement of the same rule. This parameter decision
neither extends workload caps nor authorizes resampling, retry or another board
run. Earlier pending-approval statements above describe their historical time.

## Loop2: final bounded corrections after fresh review (2026-11-28)

At7379b68 the validated convergence/adversarial results returned FIX BEFORE MERGE
with six Major and two Minor findings. The operator had authorized one remaining
correction loop. No owner decision was pending; the60s source above remains the
authority. The following are implementations of existing Issues200/201 clauses,
not new hardware, policy or native-workload decisions.

| Finding | Existing AC / invariant | Correction and controls |
| --- | --- | --- |
| CONV-202-ENDPOINT-TEST-COVERAGE (Minor) | Issue200 board-free endpoint tests | Exercise actual curl API and docker ps builders under hostile DOCKER variables using mocked subprocess.run; no real daemon. |
| ADV-202-DRAM-REQUEST-AGE (Major) | Issue201 both-phase freshness; approved60s | Load/check operands before subtraction; after serialization/start preparation, check and persist one pair. Both phases reject59.999-to60.001s delays before create/start;60s remains inclusive, next-float rejects. |
| ADV-202-XML-EXACT-FORMAT (Major) | Issue201 native UTF-8/exact grammar/time | Reject NUL redetection, comments/PI/namespace declarations; compare duration as Decimal. Positive declaration whitespace retained; one-mutation encodings/markup/bounds controls. |
| ADV-202-WATCHER-DUMP-GRAMMAR (Major) | Issue201 complete bounded Watcher | Require source-shaped admitted-device rows and mandatory launch fields; malformed control lines refuse. Missing-footer-only and monotonically ordered over-cap mutations isolate both checks. |
| ADV-202-DOWN-SEVERITY-CLASSIFICATION (Major) | Issue201 DOWN grammar; ADR0021.7 | Recognize summary-shaped bodies before level dispatch; exact warning envelope/source required even when changed to info/debug/trace or unwrapped. Gtest tear-down is not a DOWN summary. |
| ADV-202-DOWN-CANONICAL-A-BYTES (Major) | ADR0021 original/canonical A bytes | Verify all native digests against immutable owner or canonical fresh result; recover headers with the same normalization and both streams as A. Locally rebound changed UID refuses; legitimate leading-space/stderr layouts agree. |
| ADV-202-CAPTURE-CLOCK-VALIDATION (Major) | Issue201 typed bounded clocks; ADR0022.3 | Explicit absence sentinel; bound null refuses. Common reader checks available schema/enclosure/request order/reported completion bounds in live and incomplete audit; raw proof is neither disabled live nor fabricated offline. |
| ADV-202-PHASE-A-MANIFEST-BINDING (Minor) | Every listed capture artifact fingerprinted | Optional A Watcher bytes are verified when listed; no A presence/scan requirement. Missing optional bytes refuse; B retains its specific Watcher result. |

The same-class pass covered capture/admission, common reader, judge, fresh-A
revalidation/UID projection, B and complete/incomplete replay. Additional nested
raw snapshot objects/required fields are checked before projection; unconsumed
native diagnostics remain unconsumed, not closed arbitrary wrapper fields. An
exhausted50s start budget now refuses before Popen, using the same pre-start
monotonic observation. Deadline values and post-start watchdog/cleanup paths are
unchanged. This is a bounded pass, not a claim of exhaustive malformed-input coverage.

Phased behavioral RED before each affected correction: the loop2 file first
reported28 failed/13 passed; shared-reader selection13 failed; raw-shape selection
28 failed; canonical-UID selection4 failed; request-boundary selection4 failed;
expired-start-budget selection1 failed. Other independent loop2 corrections were
already present during later targeted RED runs; these are not all claimed to be
clean7379b68 executions. Endpoint-only coverage, existing refusals and unchanged
60s boundaries are review-driven regression controls, not fabricated RED.

A draft DOWN classifier overmatched the legitimate gtest tear-down banner and
caused audit16 to disagree on Bv2. A one-case interim-regression test failed;
classification was corrected to summary shapes, not a broader warning exemption.
Audit17 restored all four scoped conclusions/eight identical refusal records.
Original bytes and the missing-raw limitation were not changed. Old toy Watcher
positive fixtures were replaced by shared source-shaped synthetic rows; no test
was skipped or weakened. Ordinary tests use only committed or synthesized data;
retained originals remain in the separate audit. Qualification records carry final
artifact-free full-suite/source-identity results after the source is committed.

## Owner cutoff: two final fixes only (2026-11-28)

The owner directs only mandatory live DRAM series/freshness and restoration of
the base `<!` XML refusal; no new formal review follows. Remaining final-review
findings are tracked in Issue203. Numeric references and XML1.1 policy are not
changed or added to that follow-up by this patch. Safety has not run.

Live `_admit_series`, prepared request guards and judge/fresh-A proof no longer
skip owner-unit-absent input: `dram_live_series_required` is the named refusal.
Device-free G0 clock recording is separate; a device-bound payload cannot omit
observations. Existing B owner-scope gates can still reject earlier by their
existing named reason. New raw series within the unchanged inclusive60s bound
remain positive. Historical legacy records use only the offline audit reader;
all offline routes explicitly return `eligible_for_admission=false`.

The XML lexical gate again rejects all literal `<!`, including whitespace CDATA,
without removing current encoding/tree/namespace/decimal checks. A/B positives
and single-addition CDATA negatives cover direct parsing and full live judgment.

Invariant inventory preceded tests and implementation. Initial RED had14 real
behavioral failures plus4 synthetic-fixture overwrite errors: atomic `publish`
correctly refused rewriting existing files. Only the private synthetic fixture
rewrite was corrected. The corrected18 controls all fail against unmodified
ed33532 in a separate external checkout and all18 pass with the implementation.
That later baseline run is not falsely described as preceding implementation.
Existing Watcher-only tests now isolate `_review_capture`, not a legacy live raw
proof; the canonical-A/device-map unit explicitly mocks prior DRAM admission.
The mandatory unmocked paths are covered by the new end-to-end controls.

Audit20 retains four scoped conclusions and the same eight refusal records,
adds legacy-live A/B and coherent CDATA A/B negatives, and verifies78 originals
unchanged. The existing Watcher-negative stale review digest remains deferred to
Issue203: unchanged refusal status is not independent semantic-completion proof.
The renewed raw loss, incomplete/non-admissible status and exit3 remain explicit.
Successor v16 qualification records the committed-source clean-checkout/full-suite
results; no original measurement, archive, native binary or old receipt changes.

## Operator decision: unavailable raw telemetry

The owner retained `out/issue30` before cleanup on November 27. The 20 original
raw telemetry snapshots for renewed A/B existed only in private temporary task
storage, which was lost on the subsequent reboot. They were not included in the
retained local archive. Public `snapshotN.public.json` files remove `host_info`
and explicitly are not byte-exact originals. Their selected values/timestamps
are preserved, but their hashes cannot satisfy the original raw binding.

The operator authorizes an explicitly partial offline audit: A v1/B v1 retain
all checks; renewed A/B report DRAM raw binding unavailable, not passed, while
checking restored native/capture/review evidence and all other available checks.
DRAM freshness counterexamples use synthetic raw fixtures; the other four Issue
201 findings use mutations of temporary copies of restored original evidence.
Live gates are not relaxed. See the successor ADR for the separate audit boundary.
