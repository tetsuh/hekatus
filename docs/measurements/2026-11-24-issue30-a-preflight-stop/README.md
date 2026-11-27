# Issue #30 A: stopped before workload launch (2026-11-24)

## Disposition

**A was authorized once, but not executed.** Preparation at
`d8108479e18dba02a79aba05a9a322cc15e7357d` was clean/pushed with full tests,
all matching-head CI checks and public Sonar **0/0/OK**. The fresh read-only
preflight then reached an unsatisfied existing environment gate: actual
per-core ERISC firmware version and independent ETH heartbeat health were
unavailable. The firmware bundle and ARC watchdog cannot substitute for them.
No admission/acceptance rule was changed to force a run.

Official workload launch count: **0**. No bound approval.json, container, native
stdout/stderr/XML, Watcher log or workload duration was created. Missing native
artifacts are not empty success files. This is not any of A's three physical
connectivity outcomes, and neither H1 nor H2 was selected.

Owner scope was A-only once, both specified cards, Watcher and 40/50/60 s. B,
reset, retry, firmware/cable operations and host-setting writes were not approved
and were not performed. Read-only observation collection was permitted.

## Observations and refusal

See [reduced public record](preflight-stop.json). Both identities/BDFs, bundle
19.6.0.0, CMFW 0.28.0.0, idle 800 MHz, DRAM telemetry and increasing ARC watchdog
were observed. Loaded KMD was 2.12.0, kernel 6.8.0-149-generic; 1-GiB pages were
8 total/8 free. Image/binary hashes matched the selected G0 material, containers
were zero and the short idle observation found no heavy local work.

Both ETH firmware/status telemetry values were zero/unavailable, not actual
ERISC version or link-down evidence. Allowed sysfs telemetry did not expose
independent per-core ETH version/heartbeat. Consequently before.erisc and
before.eth_heartbeat_healthy remained null. The committed `_card` gate in
`enodia/tt/bench/issue30.py` requires observed ERISC >=1.4.1 and healthy ETH
heartbeat, so the incomplete draft was refused before device workload launch.

No connection table, native diagnostic, Watcher output or post-workload state
exists. No claim about link training, peer absence, runtime first-training
trigger, firmware defect, receiver content, FPGA compatibility or B3/B4 cost is
made. Design/open-issues were not changed: A established no new link fact.

## Provenance and next decision

Original read-only telemetry remains private and byte-exact; the public record
is a reduced derivative with hashes and no machine names/local absolute paths.
No native streams/XML exist to archive. A copied private collector note retained
its old generic no-authorization label; the owner scope recorded here is current,
and that labeling correction does not alter observations or grant extra scope.

Proceeding requires an owner decision on a bounded read-only method to obtain
the missing per-core observations, or an explicitly revised A-only admission
rule with tests/precommit/CI. Neither was inferred from the A execution approval.
No automatic continuation or use of this record as accepted discovery is allowed.
A launch count zero does not establish continuing authority at a changed head;
any future action needs fresh owner/head/boot/environment binding. B still needs
accepted A evidence and separate owner approval.

RED: N/A (read-only preflight observation/refusal record, no software change).
