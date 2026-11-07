# Issue #120 renewed P1: result capture failed, sequence stopped

The immutable [attempt record](2026-11-07-p150a-issue120-p1-runtime-root-v1-failure.json)
and hashed companions preserve this separate `p1-runtime-root-v1` attempt.
The measured harness was clean/pushed `9d5025a3a9e1a9001c6c8a47656121c4b15c45c1`.
The [precommitted plan](issue120-feasibility-plan.md#new-attempt-identity-and-prediction-not-executed)
was used unchanged. The original empty-root failure record remains unchanged.

## Outcomes and predictions

| Probe | Native / outer cap | Watcher | Outcome against prediction |
|---|---|---|---|
| P1, `p1-runtime-root-v1` | 60 / 60 s | on | Wrapper exit 2: `native runner did not return one JSON result`. Fixed-root startup progressed beyond the original failure, but no usable result/window/completion bracket was retained. Transfer-gating prediction untestable. |
| P2, transfer-only Q1 | 30 / 60 s | off | Not run: stop rule after P1 abnormal wrapper exit. |
| P3, write-only Q2 | 30 / 60 s | off | Not run; prediction untested. |
| P4, pinned NoC Q3 | 60 / 60 s | on | Not run; prediction untested. |
| P5, legacy Q4 | 30 / 60 s | off | Not run; prediction untested. |
| P6, real-frame Q4 | 60 / 60 s | on | Not run; prediction untested. |
| P7, real-frame Q4 without Watcher | 60 / 60 s | off | Not run; prediction untested. |

Device 0 alone was exposed to the named container: p150a serial
`000004033191411c`, PCI `0000:06:00.0`. Fresh pre/post telemetry showed both
boards at 800 MHz with healthy DRAM. Device 1 had telemetry checks only, not
compute exposure. Container listings were empty before/after and lifecycle
events record create/start/die/destroy. No forced kill, reset, timeout, retry
or production run followed. Host wrapper bounds were
`2026-11-07T00:57:33Z` to `2026-11-07T00:57:41Z` at one-second UTC resolution;
separate host monotonic bounds are retained, not treated as card timestamps.

## Evidence limits

The wrapper reaches JSON parsing only after a native zero return code. That
is a control-flow inference from the committed launcher, not a directly
persisted native exit-status artifact. It parses the last stdout line; native
stdout/stderr were not preserved on parse failure. Runtime cleanup output
after JSON is a plausible explanation, **not an observed diagnosis**. Issue
#175 tracks result framing and raw-stream retention; no framing fix or hardware
retry was made in this task.

The retained Watcher log attaches/detaches device 0, includes six dumps and
names the resident observer/producer kernels. Its retained text has no
ERROR/WARNING/ASSERT/hang scan matches. Inspector logs retain compile/workload
events. These diagnostics cannot reconstruct the missing card interval,
completion bracket, payload verification or overflow/counter verdict. They
are not an accepted Watcher/kernel or overlap result on their own.

**Q1-Q4 remain unresolved.** In particular, this failure proves neither that
CQ1 transfers overlap nor that they are serialized. P3/P4 and actual frame
coverage were not exercised. The first candidate (ordinary unpinned CQ1 bulk
writes with direct-L1 challenge observation) is not adopted or ruled out.
There is no CPU-only fallback.

Sanitized log/Watcher/inspector derivatives replace personal/container-cache
prefixes and normalize terminal whitespace once before archival; original
private bytes and their hashes are retained. The environment and frozen
power CSV are byte-preserving copies. The trace is diagnostic-only because
no usable card program window survived. No missing card evidence is fabricated.

## Next work

All hardware work is finished at this stop. Per owner instruction, commit/push
these records first, then fix the PR-introduced #174 audit regression with
tests before implementation and without changing any record bytes. The
record commit `8239d15` introduced array-valued companions on this branch;
calling it a baseline did not mean the failures existed on main. Keep
`Refs #174`; its closure remains owner-controlled.

A later #175 correction needs tested result framing, retained native streams,
a separate prediction/identity and renewed owner approval before another P1.
No additional hardware authority is inferred from the current stop.
