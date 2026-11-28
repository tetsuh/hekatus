# Issue30 evidence-consumption replay

This is a board-free procedure/format inventory, not a new measurement. Original
attempts, approvals, verdicts, archives and receipts are immutable. Replaying
bytes grants no hardware authority; both A/Bv2 one-use scopes are consumed.
Physical DRAM-cause investigation, jitter and latency remain #198/#197/#199.

## Current retention limit (2026-11-27)

The owner restored all four capture/observation trees from the local archive
saved before worktree cleanup on November 27. The twenty original raw telemetry
snapshots for renewed A/B were **not archived**: they existed only in private
temporary task storage and were lost when the recording host rebooted. Public
`host_info`-removed snapshots explicitly are derivatives, not original bytes.
They must never replace raw files or receive replacement raw fingerprints.

Per the operator's decision, the current replay is an **incomplete offline audit**
([ADR-0022](../adr/0022-issue30-incomplete-offline-audit.md)). Renewed A/B return
`validation_complete=false`, `eligible_for_admission=false` and a named unavailable
DRAM proof. Raw binding, raw values/order/clock association and raw freshness are
unverifiable, not passed. Live judge/fresh-A/B admission still refuse missing raw;
there is no live bypass flag. A v1/B v1 retain all checks of their historical scope.
The CLI exits3 for this completed-but-incomplete audit (0 requires complete
validation availability).
The [new audit proof](issue30-g0/evidence-replay-v16.json) must not be confused with
the earlier complete v13-v15 proofs made while the raw snapshots still existed.

## Baseline and procedure

The operator selected the pre-correction **current evaluator** baseline, not
historical verdict reproduction. A v1 originally rejected/watcher, whereas the
current baseline accepts discovery after the prospective Watcher correction.
See ADR-0021:57-63,80-84; no original record or historical-policy profile changes.

Expected conclusions: A v1 accepted_discovery/accepted; B v1 rejected/execution;
A v2 accepted_discovery/accepted; B v2 accepted_transfer/accepted.

```sh
uv run --no-sync python -m tools.replay_issue30_evidence \
  --source out/issue30 \
  --baseline "$PRE_CORRECTION_BASELINE" --output "$NEW_PRIVATE_REPLAY_DIR"
```

The source is fixed to this checkout's canonical `out/issue30`. Baseline must
be named `replay-baseline.json` in an existing owner-only0700 directory directly
under the OS temporary root obtained with `tempfile.gettempdir()`, named
`tidd-195-*` or `pi30-*`. The configured temporary root and every selected task
root must be outside the checkout. Snapshot root, when supplied with `--snapshots`, must be such a private root too. Output is a new alphanumeric/hyphen/underscore child of a private root,
not an arbitrary path. Redirected paths, broader file selection and unsafe root
permissions refuse before staging. This also bounds LLM-supplied CLI arguments.

The fresh output directory must not exist. The tool stages copied original
capture/review/input files at canonical attempt slots. Supplying original raw
with `--snapshots` additionally stages twenty raw files; none is supplied for the
current incomplete audit. Historical owner-unit-absent A v1/B v1 now use only
the separate audit boundary; the live `_judgment` refuses those shapes with
`dram_live_series_required`. Renewed A/B without raw also use that boundary over
`_review_capture`, `_capture_result`
and the shared strict fresh-A comparison. Both routes verify original manifest
entries, native bytes and independent review/observation bindings. The separate
audit is not a successful live judgment and never authorizes capture or publishes
to original result slots. The original
recording process timezone was JST; replay restores that Python process timezone
only to decode naive snapshot dates. It does not change host settings or policy.
Every original source byte is rehashed in a finally guard, including on refusal.

All four cases are scanned: reader errors and unavailable proof are recorded
separately from non-raw semantic conclusions. For v16 the original-head baseline
was reproduced at `fd4d4ec` in a disposable source copy with only unavailable raw
proof omitted; no live source or original input was edited. A conclusion different from baseline, a need to weaken existing
checks/baselines, or a need to edit records/archives stops the correction.
Uncommitted implementation format bugs can be corrected without another approval.
After four scoped conclusion matches, exactly one stdout byte is changed in a
copy. The raw-role-swap negative is available only when original raw snapshots
are supplied; it cannot be rerun after this loss. Additional copy
mutations change a motherboard bus (with derived capture/review fingerprints
updated coherently), discovery attempt, and discovery capture fingerprint. All must refuse
through the same shared boundary; restoring the copied bytes must restore the
positive result. Discovery must equal the canonical, independently rejudged A capture result in
both offline and live paths; only live additionally checks physical authority.
A motherboard chip/bus pair must identify the same admitted card, not independent
members of allowed sets. Unit controls additionally cover XML/DTD/marker/severity/pairing,
container and compiler provenance mutations. The current audit records copied
Watcher completion, XML hierarchy, DOWN logger component and captured-before
integer-type refusals and restored positives. The Watcher negative's stale
review digest remains a known Issue203 limitation: its refusal status is not
independent semantic-completion proof. This final correction does not repair it.
Additional `final_fix_negatives` demonstrate historical A/B refusing through
live judgment while their offline controls remain non-admissible, and coherent
A/B whitespace-CDATA copies refusing `test_result`. The prior `<!` exclusion is
restored alongside tree checks; numeric references/XML1.1 policy is unchanged.
DRAM freshness uses synthetic raw fixtures for both phases, not a substitute for
the lost real raw. No daemon/device is accessed.

## Historical twenty-snapshot format inventory (forty card rows)

Only consumed fields and representation examples are listed, never host metadata.
Tests use these representations as positive controls. The earlier complete
replays checked original bytes; the current audit cannot repeat that proof. The
remaining public derivatives preserve selected values/timestamps only.

| Field | Retained representation | Check |
|---|---|---|
| `time` | String `2026-11-25T23:28:01.359897`, six fractional digits, naive recording-host local time;20 distinct timestamps | Strict ISO shape, raw spacing/order and capture bracket |
| `board_info.board_id` | Strings `000004033191411c`, `0000040331914124` | Exact identity, never numeric conversion |
| `board_info.bus_id` | Strings `0000:06:00.0`, `0000:09:00.0` | Exact serial/BDF association |
| `board_info.dram_status` | JSON boolean `true` | Exactly true, not integer1 |
| `firmwares.fw_bundle_version` | String `19.6.0.0` | Exact environment value |
| `firmwares.cm_fw` | String `0.28.0.0` | Exact environment value |
| `telemetry.aiclk` | String **`" 800"`**, one leading ASCII space | Unsigned integer800; ASCII padding only, no truncation |
| `telemetry.heartbeat` | Unpadded unsigned decimal strings, e.g. `7717`, `7718`;30 distinct values | Exact summary value, progressing per serial, aggregate first/last binding |
| `smbus_telem.DDR_STATUS` | String `0x5555` | Exact mask after ASCII padding normalization |
| `smbus_telem.GDDR_UNCORR_ERRS` | String `0x0` | Unsigned hex zero, never truthiness |
| `telemetry.asic_temperature` | Decimal strings: `53.2,53.4,53.5,53.6,53.9,54.4,54.5,54.6,56.2,56.4,57.4,57.5,57.6,57.8` | Separate finite decimal observation parser; last sample binds supplied temperature summary, no new health threshold |

Integer parsing admits only native nonnegative integers or unsigned ASCII
base10/`0x` strings with leading/trailing U+0020 spaces. It rejects bool, float,
fractional/exponent strings, tabs/newlines/NBSP, non-ASCII digits, malformed signs,
empty/space-only input and extra/truncated tokens. A valid shorter numeral still
fails the exact required value comparison, e.g.80 is not800. Temperature is
explicitly a decimal observation, not an integer counter: never truncate it.

An early uncommitted implementation refused padded aiclk with
`dram_raw_integer`; the positive real replay caught this. The operator authorized
ASCII display-padding support. The original snapshots and summary800 were not
changed to fit the parser. All same-class numeric representations were inspected
in one pass and covered, rather than special-casing that individual sample.

## Loop1 format/type/order corrections

The shared Watcher reader requires non-whitespace payload after `Device ` inside
a dump, not merely the prefix. Native XML accepts only UTF-8 bytes and any XML
declaration must name UTF-8, including when XML whitespace surrounds `=`. Full
judgment validates XML before consuming its capture timestamp, so an alternate
encoding fails as `test_result` rather than escaping via a parser exception.

Every supplied create/pre-start wall boundary must be at most native root time
plus **0.001 seconds**. `issue30_formats.STAMP` fixes the native timestamp to three
fractional digits; one millisecond is therefore one representation quantum. The
comparison is inclusive at `native + 0.001`, with a next-representable-float
negative and a coherent five-second negative (both wall/monotonic fields of the
same event are moved). This is not a clock-drift allowance. Capture enclosure,
wall/monotonic consistency and the owner-approved60s freshness limit are not widened.
Legacy evidence without those request boundaries retains its existing enclosure
and conservative freshness checks; no timestamps are invented.

Nested JSON objects are checked before field access or set construction in DRAM
summaries, discovery records, receipts and replay baselines. Rejected evaluate
inputs preserve malformed container shapes for their rejection fingerprint;
this is not normalization into acceptable evidence. Overflowing clock integers
produce named clock refusals. These checks are shared by the full judge/fresh-A/B
paths and, where the evidence is available, the separate incomplete audit.

## Pre-launch time analysis and owner-approved parameter (2026-11-28)

Raw snapshot `time` is recording-host local naive ISO (JST); public derivatives
explicitly preserve selected values, including that time. Native gtest root
`timestamp` is UTC, as established by the source/recording convention used in
ADR-0021's existing bracketing rule. `monotonic_seconds` in the retained series
is a host monotonic sample-completion observation, not epoch time. The original
capture source at `f8703fb` starts a monotonic budget but persists only elapsed
duration and **boolean** `started`; it does not retain the absolute monotonic
start, create/start wall timestamps or a daemon-created timestamp. A boolean
cannot be subtracted from a timestamp. Filesystem mtimes are not a time proof.

| Phase | Last public before timestamp (JST) | Native root timestamp (UTC) | Difference to native root | Capture elapsed |
|---|---|---|---:|---:|
| A v2 | 2026-11-25T23:28:22.343414 | 2026-11-25T14:28:52.133 | 29.789586 s | 1.182740260 s |
| B v2 | 2026-11-25T23:30:09.551633 | 2026-11-25T14:30:46.725 | 37.173367 s | 6.031428763 s |

**Exact last-reading-to-container-create/start intervals cannot be recovered.**
Since create/start precede native execution and follow capture entry, the
existing total-elapsed enclosure gives only conservative ranges from that last
raw timestamp: A approximately28.606846..29.789586 s; B approximately
31.141938..37.173367 s. These are bounds derived from retained observations,
not separately measured create/start times or sample-completion delays.

The owner approved **60 seconds maximum age** on **2026-11-28** in
[Issue201 comment7233987817](https://github.com/tetsuh/hekatus/issues/201#issuecomment-7233987817).
The implementation checks both last raw wall time and
last reported monotonic completion at admission, immediately before the create
request and again after its response before the start request, for both phases. It covers the retained native-root gaps up to37.174 s
with about22.8 s margin for host validation/launch, while rejecting a one-day
clock advance and stale evidence. It is not a health guarantee or a new workload
budget. Failure stops without automatic resampling, retry or extension. The
approved numeric parameter does not grant another hardware use. Existing
40/50/60-second native/container/cleanup caps are unchanged, and no resampling or
retry is authorized by that decision.
New captures persist pre-create and pre-start request wall/monotonic boundaries
as well as capture clocks; live and complete replay check their freshness and
ordering. These host decision boundaries are not claims about an independently
observed daemon create/start timestamp. Without a pre-start boundary, complete
replay conservatively uses the native root as the latest possible launch, not
the earlier capture-entry enclosure. No new clock is
fabricated for old captures.

Loop2 moves the host request checks after curl payload serialization and start
stream/argument preparation, checking and recording the same wall/monotonic pair.
It also refuses an exhausted50-second start budget before Popen. This strengthens
existing checks without extending60-second freshness or40/50/60-second caps.
Present clock content, including null, is never interpreted as legacy absence.
The shared capture reader checks available clock schema/enclosure/request order
and reported summary completions even in the incomplete audit; raw binding, raw
dates and their dependent proofs remain unavailable, not replaced by summaries.

## Four-capture formats and binding

Every listed Phase A artifact is fingerprint-checked, including an optional
Watcher log. A still neither requires that log nor applies B's semantic scan.
B's DOWN summaries are classified before severity dispatch and require the exact
warning envelope/source. UID projection reads both A streams with A's parser and
checks all three native digests against the immutable owner record or canonical
fresh A result; a locally rebound capture cannot replace accepted A bytes.

Watcher device rows require admitted device0/1, source-shaped core/virtual (and
optional physical) coordinates and mandatory launch fields, not arbitrary text
after Device. The grammar follows pinned
[watcher_device_reader.cpp](https://github.com/tenstorrent/tt-metal/blob/d9a68815f5fcf08a5bfbffb6f1f811823fba8edd/tt_metal/impl/debug/watcher_device_reader.cpp)
Core::Create/DumpLaunchMessage: worker/acteth/idleth/dram, printable report-only
waypoints, rmsg mode/NOC/run state and enable letters, h_id and optional smsg.
Malformed At/Dump/Device control records refuse instead of being ignored. Existing
manual review and clean-content gates remain required; rows do not establish
new per-core firmware/ETH health.

Execution/review/observation/container metadata is UTF-8 JSON with unique keys.
Execution is anchored by the independent review SHA256; consumed manifest entries
must uniquely bind path/byte count/SHA256. Image and official ELF pins match.
Exit/client_exit are integer0 for accepted attempts, integer1 for B v1; booleans
remain distinct. Elapsed values are finite JSON numbers. Lifecycle booleans are
exact; empty daemon arrays and zero-byte CLI listings are mandatory observations.

Native XML bytes must decode as UTF-8 without illegal control characters; Expat
must not redetect NUL-interleaved UTF-16. Comments, processing instructions and
namespace declarations are rejected rather than silently discarded. Duration
strings are compared as exact decimal values against40 seconds, not rounded
binary floats. Legitimate declaration whitespace remains accepted.

All four native XML files have a `testsuites` root, one direct named `testsuite`,
one direct expected `testcase`, decimal-string tests counts, run/completed status,
and UTC `YYYY-MM-DDTHH:MM:SS.mmm` root timestamp. Failure remains in B v1;
execution rejection precedes successful-test interpretation. Parser-level DTD
prohibition applies to UTF-8/UTF-16 alike, with no namespace/wrapper/count bypass.

JSON comparisons are recursive and type-preserving. Wrapper metadata has closed
field sets, validated consumed scalar types and no duplicate keys, nonfinite
extensions/exponents or alternate JSON encodings. Full manifest entries are
unique, relative and typed; every captured artifact is fingerprint-checked.
Raw telemetry's additional diagnostic fields remain explicitly unconsumed,
not permission to add unknown wrapper acceptance fields.

Native XML accepts only the direct unnamespaced root/suite/case hierarchy and
known gtest attributes, exact selected name/run/completed/count values and no
extra children. Optional durations must fit the process cap; supplied timestamps
must be valid ASCII calendar values. Native logger envelopes use millisecond dates, severity and component fields;
only complete frozen #176/#120 diagnostic records match after allowed invocation
normalization. Pair adjacency is within each stream. B markers are exact, unique
and ordered: fast dispatch, strict mode, raw gtest RUN, raw timed OK, raw PASSED.
Watcher is hash-bound, strictly UTF-8 and contains ordered, matching numbered
start/completion pairs with device rows and bounded relative times for B. All
observed Watcher event times must fit native/capture caps; captured Watcher=1
and explicit manual review remain mandatory. It is not applicable to A;
it is not firmware-health proof. No printed packet/byte/status measurements are
invented from source-bound checks. Missing/changed/unlisted consumed files refuse.

New captures include a manifest-bound `capture-clock.json` with attempt, wall and
monotonic endpoints; it narrows the capture enclosure and checks claimed sample
completion times. Older native root timestamps plus total elapsed provide a
conservative enclosure without inventing absent clock observations or using
filesystem mtimes. The same before-start/after-end rule applies to both data
forms. Conservative inability to prove a bracket is a stop, not a deadline change.
