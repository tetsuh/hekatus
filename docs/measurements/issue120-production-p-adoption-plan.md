# Issue #120 dated adoption and single-use production precommit

## Owner decision — 2026-11-21

The owner adopts 520 seconds / 22155167 fixed writes and authorizes exactly one
production-v1-control-500000 (P1), followed only after every health/release gate
by exactly one production-v1-loaded-500000 (P2). No renewed permission is needed
after board-free repairs and successful complete gates. This is not permission
to bypass a failed gate. No hardware execution has consumed this authority.

## Invariants before implementation

1. Preserve the selection bytes, C1/C2/B original records/admissions, native,
   kernels, CTA, libraries, compile/build proofs and device-facing constructors.
   Stop rather than change native/device-facing content.
2. A small versioned production_adoption.json records the actual dated owner
   decision, target/count, exact selection-file SHA and one conditional pair.
   Missing, extra/invalid fields, invalid/future dates, wrong scope/count/selection
   hashes reject. Generator --adopt-owner-decision selects an explicit v2 receipt;
   the default remains the exact historical non-authorizing v1 generation path.
3. Adopted receipt declares owner_adopted and execution_authorized true as a
   record of the decision, not current readiness. It embeds the input and its
   byte hash. Historical v1 validation never acquires v2 meanings. Live v2 needs
   the current matching input; persisted v2 uses its captured adoption facts.
4. Fresh preflight, clean pushed tested head, receipt/source/image/boot/session
   binding, explicit runtime approvals and full P1 health gates remain mandatory.
   Operational new P session/launch requires v2; isolated historical v1 fixture
   behavior remains unchanged. All captured fixture flags still agree.
5. Before wrapper launch, reserve the role exclusively in its session case,
   bind a durable claim to manifest hash, receipt, boot/head and role. A consumed
   role can never relaunch after success, failure or partial publication. Bind
   the actual non-fixture adoption to one session through an exclusive durable
   registration; another session cannot replay the single-pair authority.
6. Container entry requires the matching host claim and an exclusive durable
   native-entry stamp before execute_native. Direct/repeated container entry
   cannot reuse authorization. No new native/wrapper arguments or work changes.
   Runtime use never rewrites committed selection/adoption/receipt inputs.
7. RED tests precede implementation, including missing/mismatched/malformed
   adoption and repeated host/native entry; preserve legacy fixture tests.
   Actual CPU launcher dry-run adds missing/mismatch/used-role negatives.
8. Changed Python/tool hashes require fresh source-matched 500000-row CPU host
   remeasurement. Charge P2's in-wrapper P1 reread, separate outer reread, and
   leave actual 2GB native retrieval unmeasured until hardware. Regenerate the
   adopted receipt canonically. Stop if pessimistic cap margin is nonpositive;
   never change target/count or extend caps automatically.
9. Full tests/ruff/C901/diff/ADR, isolated no-device startup/publication, actual
   fixture dry-run, clean push, matching-head CI9 and Sonar0/0/OK without
   acceptance precede actual fresh physical preflight and the one new session.
10. Preserve existing pre-run threshold-free predictions: centers near1350000
    ticks, descriptive C2 quantiles, differences near0 without known sign or
    bound; no advance latency acceptance threshold. Unknown long-run rate,
    AICLK/thermal/frame/memory/IO behavior and native2GB cost remain risks.

## Source-matched adopted-v2 host-cost measurement

The first adopted-v2 CPU measurement completed without native/device access:
control construction6.005598s/publication0.002785s;loaded construction7.303426s/
publication0.011691s plus5.951577s in-wrapper P1 reread/entry,total13.266694s.
Outer reader7.245716s is separate. Peak115773440bytes(110.41MiB);real native2GB
retrieval remains unmeasured. The new receipt is generated through
`uv run python tools/prepare_issue120_production_admission.py --adopt-owner-decision --write`.
It records the dated decision but does not substitute for fresh runtime gates.

| Target | Count | Frame margin | Optimistic cap margin | Pessimistic cap margin |
|---|---:|---:|---:|---:|
|550s scenario,not selected |23433350 |49.291s |−12.246s |−22.246s |
|530s scenario,not selected |22581228 |29.291s |8.871s |−1.129s |
|520s adopted |22155167 |19.291s |19.430s |9.430s |

Reserve31.510054s/41.510054s includes measured host13.266694s,setup2.134215s,
snapshot1.109144s,unmeasured-native budget10s and guard5s,plus10s pessimistic
stress. These are conditional assumptions,not bounds. The selected premise
remains feasible under this model;no count/target/cap change is made. If later
source repairs require remeasurement,these figures must be replaced before entry.

## Hardware gates and stop rules

Exactly the owner brief-120-prod-p: device0 serial411c PCI0000:06:00.0 only,
Watcher off,500000frames each,native/outer600s. Fresh Linux6.8.0-149/KMD2.12,
firmware/image/native/device/CTA/library/compile/C1/C2 hashes,800MHz/DRAM match;
zero containers,quiet host,explicit device0,disk/memory headroom before each run
and again immediately before launch. No heavy work during hardware.

P2 needs every P1 exit0,complete identity/checksum/work,zero abort/drop/overflow,
healthy routing/mailbox/clock/power,baseline3warning classes only,successful
ordinary reread,no unexplained paired-clock evidence,zero containers and both
cards800MHz/DRAM healthy. Stop new warnings/errors,identity/boot/open changes,
timeout,execution/publication/validation/integrity failure or P2 not_covered.
No count changes,extension,retry,automatic acceptance; at most one reset only
for timeout/abnormal exit,then stop. Preserve originals and failed claims.

## Publication

Use the established bounded byte-exact archive mechanism and repository limits;
never upload evidence externally without permission. Report all AC metrics,
causal lower bounds versus submitted totals,weighted-AICLK scope,predictions,
actual cap/retrieval costs and remaining limitations. Only two healthy runs
populate design section12 and the measurements result slot. Commit -F,push,
verify CI/Sonar;leave PR draft/unmerged. Final report is issue120-production-p-report.md
in the new execution work directory. No hostname or private absolute paths in
public evidence.
