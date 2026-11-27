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
  timestamps; bounded ordered clocks; proposed60s freshness at admission, before
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
  carrying PR/report. No acceptance/muting or formal review is performed here.

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
rerun. The60s proposal remains pending owner approval.

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
