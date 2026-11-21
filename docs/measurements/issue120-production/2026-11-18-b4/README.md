# Issue #120 B4 — renewed v2 bounded control/load

Harness: `98739788c1a07c1e3e3ea1b6f7b41a056d002f1a`.
Session: `8caa5d94ed7443d9982606ea45072452`, immutable runtime-manifest SHA256
`c8a54a31622e64c3f3f9582c8a037c040185536a6e77e3b6c0988ed4a9f17f25`.

The owner approved Linux6.8.0-149-generic/KMD2.12.0 as the new baseline on
2026-11-18. Fresh values matched board/serial/PCI/firmware/image/native/kernel/
CTA/800MHz/DRAM requirements. Do not call this the same environment as run8.
The [B4 identities, predictions and stop rules](../../issue120-production-b4-plan.md)
were committed before execution. Preboard full regression6211/1057/8subtests,
CI9checks, public Sonar0/0/OK and actual no-device root-to-host publication
readback passed at that same clean/pushed head.

## Execution and branch

| Row | Plan identity | Attempt | Native/container/wrapper/launcher | Coverage | Fresh Watcher dumps |
|---|---|---|---|---|---:|
| B1 |`production-v2-control-watch-b4` |`20261118T1445539271829941429764960Z` |0/0/0/0 | not_applicable |7 |
| B2 |`production-v2-load-watch-b4` |`20261118T14472425729255814391413846Z` |0/0/0/0 | covered |8 |

Both used only device0,4096 frames, Watcher and60s native/Docker caps. B1 passed
actual launcher reread, full frame/diagnostic/Watcher/power checks and post-health
before B2; baseline/boot were rechecked before B2 in the unchanged session.
Only the three exact #176 warning classes occurred. Both returned800MHz/healthy
DRAM/zero uncorrected errors, their named containers were destroyed, and none
survived. Reset0, retry0. Hardware stopped after B2; C/P were not executed.

A preflight setup assertion first rejected an absent ambient device-node variable
before any B launch. The launcher source already fixes device0; fresh preflight
was repeated with that same approved explicit value. No baseline difference or
new runtime warning was accepted. See `execution-summary.json`.

B1/B2 v2 snapshots are0644 and readable to the real host user, with no reader
errors, complete frozen coverage and valid dedicated v2 scope. Private native
results remain0600. The original v1 B1 exit2/archive/session is unchanged and
not promoted or reused. These runs confirm the previously board-free repairs,
not a retrospective correction of old evidence.

## Reference observations only

| Condition | P50 ticks | P99 ticks | Max ticks | Overflow/drop/abort | Paired outliers |
|---|---:|---:|---:|---|---:|
| Control |1350002 |1350798.18 |1351261 |0/0/0 |0 |
| Loaded |1350005 |1350984.00 |1351306 |0/0/0 |0 |
| Loaded minus control |+3 |+185.82 |+45 |0/0/0 |0 |

All4096 attempted/produced/consumed checksums/work/identities passed. Only4095
intervals exist; P99.9/P99.99 remain insufficient, not fabricated estimates.
Watcher means timing_evidence=false for both. There is no advance latency
threshold, production result or design.md section12 update.

B2 submitted262144x4096=1073741824 bytes with8589934592 fixed CPU operations.
The causal frame/map evaluator verified4096 frames and a first-through-last
lower bound4095 transfers/16773120 bytes/134184960 CPU operations. The separate
fresh-challenge total is4097 transfers/16781312 bytes including endpoint scopes.
Do not credit unseen sequence gaps, submitted totals or terminal diagnostics as
continuous in-window traffic. Mailbox4098/runtime peers matched; those are
health diagnostics, not inputs that manufacture coverage.

## Rates, post-processing and cap uncertainty

`reference-analysis.json` contains exact recorded arithmetic and explicitly
conditional models. B2 worker elapsed6.369850819s, inclusive41153.868writes/s.
Load-start to terminal-observed upper bound is4.878675135s; it is not an exact
host timestamp of the last frame. Native poll sleeps5ms and scheduling/tail
uncertainty is not a fixed subtractable bound. Raw first-to-last span is
5528250346 ticks, about4.095000256s with envelope-weighted1350MHz; that conversion
is not exact frame wall time.

B2 native drain-to-return retrieval/verify/close/publication is0.173632572s,
Python analysis0.050221829s. Their sum0.223854401s is54.652us/frame amortized,
not a measured slope. B1 components are0.177773968s/0.041842053s. Native close
contains fixed work; scaling it all to500k overstates some costs and still omits
other copy/hash/serialization costs.

Critically, main enqueue returns0.679886107s after worker start. If that interval
were entirely fixed worker stall, removing it gives a46071.287writes/s proxy,
not41153.868. No timestamped submission counter proves that attribution. Using
the inclusive rate as a long-run per-write slope can undercount N. The
[updated B3 cap study](../../issue120-production-b3-cap-study.md#b4-observed-baseline-update)
therefore shows both rate models,550/530/520s targets and the unresolved
cross-model risk. Positive cap margin in one model is not coverage admission;
no target/count/cap was changed.

## Evidence and replay

Each manifest lists11 **original byte-exact** payloads in its tar.gz: primary
record, frame/map, native envelope/direct-status/stdout/stderr companions,
Watcher, frozen/active power and environment. Archive headers have deterministic
non-identifying metadata; payload bytes/permissions at the measurement location
are untouched. Re-extraction passes the ordinary full loader and every hash.

Wrapper logs contain local output paths, so their originals stay private with
SHA256/byte counts in the manifest; only separately identified normalized
public derivatives are published. Native/Watcher readable derivatives remove
path prefixes/trailing whitespace; their originals remain inside the archives.
No raw record or diagnostic bytes were rewritten to satisfy formatting checks.
No extra whitespace exception or validator relaxation was added.

Board-free replay (from the repository root):

```sh
uv run pytest -q tests/test_issue120_production_b4_archive.py
```

For future C admission preparation, extract each archive to its own fresh
`out/B1` or `out/B2` directory under this archive directory and call
`recording.load_validated_record` on its primary. `out/` is already ignored;
original payload hashes and the tracked tar manifests bind the expanded bytes.
Do not overwrite a preexisting extraction or reinterpret v1 data. A committed
bounded admission receipt and explicit owner C approval are still absent/required;
this archive alone cannot authorize C. C predictions and the narrow C2 margin
are recorded in the B4 plan; neither C nor a new count is silently executed.
