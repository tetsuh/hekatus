# Issue #120 B follow-up — board-free invariants before RED

Scope: repair #182/#183 and add post-publication launcher regressions. No board,
telemetry/device discovery, hardware container, B/C/P execution, reset or retry.
B1 original record/evidence bytes and its measured revision remain immutable;
native/kernel source bytes and digests are to remain unchanged.

## Invariants

1. Every artifact consumed by the host launcher must be readable by its actual
   unprivileged execution user after the container exits. Test the publisher/
   reader UID boundary, not only a same-user in-process assembly.
2. Immutable means exclusive atomic publication plus size/hash/manifest binding,
   not an assertion that mode0600 prevents mutation. Existing snapshots are never
   chmodded or rewritten. New production snapshots may explicitly publish0644
   (owner read/write, others read; no group/world write) before the atomic link.
   The shared legacy default remains0600. One exclusive-open policy restricts
   fresh public outputs to0644 and rejects replacement/write modes or other
   permissions before opening a destination.
3. Do not add privileged containers, capabilities, broad chown/chmod, new writable
   mounts or a result-transport fallback. ADR-0018 native result publication is
   unchanged. Mode-scoped publication keeps the existing #172 pinned runtime/
   device/cache ownership path; changing runtime UID would additionally require
   device and old root-owned JIT/Watcher cache permission changes.
4. #172 probes returned their atomic native envelope and raw diagnostics without
   freezing/reopening this container-created power snapshot in the host launcher.
   The production path added that cross-user read. Protected native result bytes
   were copied into readable public companions (or inspected through stopped
   containers), not silently accepted from stdout. Document these distinct paths.
5. Add production record v2 and an explicit v2 power interval contract. Its label
   names the enqueue-to-terminal-observed host upper bound, not exact card/frame
   UTC and not the legacy Issue #118 program-host source. Unknown labels/contracts
   cannot acquire weighted conversion by fallback or relabeling.
6. Weighted AICLK stays null for unknown provenance, missing/invalid/reversed/
   empty interval, absent/invalid samples or an interval not bounded by usable
   trace samples. Whole-run coverage alone does not certify load-envelope timing.
7. New v2 non-Watcher timing requires healthy execution, complete frozen whole-
   run coverage, no reader errors, correct explicit v2 source and positive finite
   weighted AICLK/weighted duration. Watcher always prevents timing eligibility.
   Invalid conversion never becomes a product wall-time guarantee.
8. Preserve v1 interpretation of accepted B1 and legacy #118/#172 records. Loading
   the unchanged B1 archive must still reproduce its original null weighted AICLK,
   stored provenance error and timing_evidence=false. Only fresh v2 publication
   opts into the corrected contract; C/P admission must not borrow legacy data.
9. Exercise the real launcher start->child publication->host reread/validation
   path with a CPU-only publisher and B1 artifact input. Mandatory permission
   seams cover CI without a root publisher; an opt-in actual root-UID no-device,
   network-none, masked/read-only-source container verifies the real ownership
   boundary locally. Tests produce fixtures, never hardware measurements, and
   restore process-global permission state when invoked in the pytest process.
10. Hash/readback/tamper rejection, exclusive paths, boot/session/approval gates,
    stop rules and all old record bytes remain intact. No accepted flag is edited
    merely to make an archive or test eligible.

## Cap and performance investigation

Read #118/#120 AC, PR #123 and actual wrapper/native lifecycles. Do not silently
move retrieval/analysis outside the600s boundary or change frame count. Separate
measured B1 timers into native retrieval/teardown and Python analysis, fixed costs
and uncertain proportional slopes. CPU-only replay may characterize analysis
without estimating PCIe throughput. Fixed N must cover the frame interval plus
startup, not post-processing; lowering550s is an offline proposal requiring actual
C rates/coverage margins and owner approval. Report optimistic/pessimistic models
and owner cap/frame-count choices if no conservative fixed count fits. Implement
only equivalent, clearly safe changes; no native/kernel/performance change is
required merely to report an option.

## Sequence and verification

These invariants precede new tests. Record actual permission/timing RED failures,
then implement narrow versioned changes. Run B1 archive regression, same-user and
cross-user launcher tests, all tests/ruff/diff/ADR checks, device-free startup,
matching-head CI and public curl Sonar0/0/OK without accept. Native compile/link or
device compile is needed only if corresponding source/digests change; otherwise
verify existing receipts/proofs and the unchanged native hash. #182/#183 remain
Refs, all authored headers(#120), no rebase/amend. Stop at a report and proposals,
not a new B execution.
