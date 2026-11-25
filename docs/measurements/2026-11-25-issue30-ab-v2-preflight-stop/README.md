# Issue #30 renewed A/Bv2 unit: fresh preflight stop

At prepared head6856565f92539249a61c38af2ebc235753ff22e9, the owner2026-11-25
authority was one new A then one Bv2 only on matching fresh H1. Full regression
6991 passed/1057 skipped/8subtests, ruff, all10 then-reported matching CI checks
and public Sonar0/0/OK had passed before fresh read-only preflight.

**Stopped before A.** Serial0000040331914124 /PCI0000:09:00.0 reported literal
`board_info.dram_status: false` /`smbus_telem.DDR_STATUS: "0x0"` in the first
snapshot, then `true` /`"0x5555"` in the second. The other card remained
true/0x5555. Identity and firmware fields did not change. Neither a lasting
hardware defect nor an intermittent telemetry-read cause is established.
The second true does not cancel the first abnormal observation. No third read,
retest, reset or workload followed; there is no inferred healthy final state.

[preflight-stop.json](preflight-stop.json) retains selected literal observations,
original raw snapshot/output hashes and the fresh stop time. Private byte-exact
snapshots contain host identity and remain outside this public record; normalized
selected fields are explicitly not represented as byte-exact full snapshots.
New boot differs from the historical A boot; that accepted A is invalid for new
admission.8total/8free1GiB pages and container0 were observed in this preflight.
Both ARC pairs advanced3250->3254,800MHz; temperatures53.5->53.6C and56.1->56.2C.
ARC does not establish ETH/ERISC health; per-core firmware/ETH health remain unknown.

A and Bv2 native invocation counts are both0. No phase approval.json, output
slot, Watcher, XML, packet result, JIT or device-close evidence was created.
The initial/renewed preparation and native G0 evidence do not count as executing
this authorized unit. Do not invent a consumed invocation, transfer, firmware
fault, fix or successful recovery. Old A/Bv1 records remain unchanged. Owner
review/disposition is required before any continuation; no automatic reuse of
unused authority after this anomaly. Further telemetry/intervention requires
owner direction, with reset/firmware/cable/host changes still unapproved.

B1 gains no new link/content result. B3 source answer stands, jitter remains
unmeasured; B4 inter-card latency and #13 loopback remain unmeasured. #105 stage3
has no new hardware evidence. This is a readiness stop, not a new transfer/JIT
failure. First-transfer AC permits recorded bounded stops, not a claim of
content verification; remaining measurement work is still open.
