# Issue #120 P7 V3: branch1 and covered without Watcher

[Primary/15 hashed companions](2026-11-14-p150a-issue120-p7-api-lock-semaphore-v3.json)
/ [run8](2026-11-14-p150a-issue120-run8.md) /
[P6 Watcher counterpart and common readout interpretation](2026-11-14-p150a-issue120-p6-api-lock-semaphore-v3.md).

P7 ran only after healthy P6 branch1/covered, normal Watcher/baseline warnings,
zero containers and fresh idle/environment/head/hash/process rechecks. Same
clean/pushed `e1b37547f0eb7909ae23fdfbb4c176ac7c047cbc`, native
`c35af4b4506dfd2534fa71debb6bdd0206825bd428fa102a04124a00cf3a9e43`,
fixed work, device0 serial411c, native/outer60s; Watcher disabled. Shared stale
Watcher artifacts are not attributed to P7.

**Branch1, covered**,4096 attempted/produced/consumed, zero abort/overflow,
matching frame identity/checksum/work and unambiguous consumer clock. Strict
coverage recomputation verified every frame plus both endpoints, conservative
completed-byte lower bound16781312 (during frames16773120). This does not count
the full submitted1GiB as in-window completion.

Both directions again report config1/Sharded/bank present; consumer-to-producer
bank258 at slot13, producer-to-consumer514 at slot6, matching actual/expected
host packs. Runtime/accessor/virtual peers `(1,2)` and `(2,2)`, self NoC0 `(2,2)`
and `(1,2)`, all equality flags true. Accessor raw addresses0x8100017b000 and
0x8200017c000, actual start/ready0x81000009f60 and0x82000009f40; host CTA manifest
and allocation guards passed. This is coordinate diagnostics, not coverage.

Producer/consumer sem words `[0,4096,1,0]` / `[4096,0,0,1]`, start exit/post1/1,
mailbox4098 (derived final expectation4096+2), final progress262144/challenge4098
valid. The literal legacy expected2 field remains byte-exact and is phase-scoped
in interpretation, with metadata follow-up #179. Nine diagnostic reads/36 bytes and
4096-byte progress retrieval remain separate from load accounting.

All8589934592 CPU operations,262144 writes/1073741824 bytes and262144 load reads/
1048576 bytes completed. Host work6.017380392 seconds; host endpoint is still
an upper-bound envelope, not exact card completion. Native/container/wrapper0,
no timeout, seven baseline warning lines only, no new error. Post-run containers0,
both cards800MHz/DRAM healthy, reset0/retry0. No further hardware or production.

Four frozen power samples do not establish full program-end coverage or a
production timing conversion. This confirms no-Watcher bounded functional and
causal-coverage behavior, not a latency comparison/production result. Ordinary
CQ1/direct-L1 is selected for feasibility per the precommitted branch; widened
long-run protocol, production integration/calibration/control and owner approval
remain outstanding.
