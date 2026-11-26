# Issue30 evidence-consumption replay

This is a board-free procedure/format inventory, not a new measurement. Original
attempts, approvals, verdicts, archives and receipts are immutable. Replaying
bytes grants no hardware authority; both A/Bv2 one-use scopes are consumed.
Physical DRAM-cause investigation, jitter and latency remain #198/#197/#199.

## Baseline and procedure

The operator selected the pre-correction **current evaluator** baseline, not
historical verdict reproduction. A v1 originally rejected/watcher, whereas the
current baseline accepts discovery after the prospective Watcher correction.
See ADR-0021:57-63,80-84; no original record or historical-policy profile changes.

Expected conclusions: A v1 accepted_discovery/accepted; B v1 rejected/execution;
A v2 accepted_discovery/accepted; B v2 accepted_transfer/accepted.

```sh
uv run --no-sync python -m tools.replay_issue30_evidence \
  --source out/issue30 --snapshots "$RETAINED_SNAPSHOTS" \
  --baseline "$PRE_CORRECTION_BASELINE" --output "$NEW_PRIVATE_REPLAY_DIR"
```

The fresh private directory must not exist. The tool stages copied capture,
review/input and20 raw files at canonical attempt/role slots. It invokes the
same `_judgment`/verified readers as live judge and fresh A reevaluation, but
never authorizes capture or republishes to original result paths. The original
recording process timezone was JST; replay restores that Python process timezone
only to decode naive snapshot dates. It does not change host settings or policy.
Every original source byte is rehashed in a finally guard, including on refusal.

All4 cases are scanned: reader errors are recorded separately from semantic
conclusions. A conclusion different from baseline, a need to weaken existing
checks/baselines, or a need to edit records/archives stops the correction.
Uncommitted implementation format bugs can be corrected without another approval.
After four positive matches, exactly one stdout byte is changed in a copy and a
before-B snapshot is replaced with after-B bytes in a copy. Both must refuse
through the same shared boundary; restoring the copied bytes must restore the
positive result. Unit controls additionally cover XML/DTD/marker/severity/pairing,
container and compiler provenance mutations. No daemon/device is accessed.

## Actual20-snapshot format inventory (40 card rows)

Only consumed fields and representation examples are listed, never host metadata.
Tests use these representations as positive controls; the replay checks every
actual value and the original complete bytes rather than these examples alone.

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

## Four-capture formats and binding

Execution/review/observation/container metadata is UTF-8 JSON with unique keys.
Execution is anchored by the independent review SHA256; consumed manifest entries
must uniquely bind path/byte count/SHA256. Image and official ELF pins match.
Exit/client_exit are integer0 for accepted attempts, integer1 for B v1; booleans
remain distinct. Elapsed values are finite JSON numbers. Lifecycle booleans are
exact; empty daemon arrays and zero-byte CLI listings are mandatory observations.

All four native XML files have a `testsuites` root, one direct named `testsuite`,
one direct expected `testcase`, decimal-string tests counts, run/completed status,
and UTC `YYYY-MM-DDTHH:MM:SS.mmm` root timestamp. Failure remains in B v1;
execution rejection precedes successful-test interpretation. Parser-level DTD
prohibition applies to UTF-8/UTF-16 alike, with no namespace/wrapper/count bypass.

Native logger envelopes use millisecond dates, severity and component fields;
only complete frozen #176/#120 diagnostic records match after allowed invocation
normalization. Pair adjacency is within each stream. B markers are exact, unique
and ordered: fast dispatch, strict mode, raw gtest RUN, raw timed OK, raw PASSED.
Watcher is hash-bound, fresh/nonempty/scanned for B and not applicable to A;
it is not firmware-health proof. No printed packet/byte/status measurements are
invented from source-bound checks. Missing/changed/unlisted consumed files refuse.

New captures include a manifest-bound `capture-clock.json` with attempt, wall and
monotonic endpoints; it narrows the capture enclosure and checks claimed sample
completion times. Older native root timestamps plus total elapsed provide a
conservative enclosure without inventing absent clock observations or using
filesystem mtimes. The same before-start/after-end rule applies to both data
forms. Conservative inability to prove a bracket is a stop, not a deadline change.
