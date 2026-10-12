# Issue #122 AICLK audit (generated)

- Input root: `docs/measurements`
- Generation command: `python3 tools/audit_issue122_aiclk.py`
- JSON records: 95
- CSV files: 72
- Interval policy: valid timestamped rows are left-held until the next row; an explicit interval is exact only when ordered trace rows cover both bounds. Missing bounds produce a `CSV capture span diagnostic`; they do not become in-run evidence. Intermediate means any valid AICLK outside `{800,1350}`; its position uses the beginning/middle/end thirds of the CSV.

## JSON record inventory

| record | issue | status | trace reference/provenance | resolved CSV | pairing | rows valid/CSV | unique AICLK (MHz) | classification | min/max (MHz) | min intermediate (MHz) | position | interval evidence | weighted AICLK (MHz) |
|---|---|---|---|---|---|---:|---|---|---:|---:|---|---|---:|
| 2026-08-14-p150a-effective-efficiency.json | — | — | — | 2026-08-14-p150a-effective-efficiency-power.csv | record-stem fallback | 89/89 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1343.638 (diagnostic) |
| 2026-08-23-host-iq-path-vs-golden.json | — | — | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0.json | — | — | — | 2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0-power.csv | record-stem fallback | 183/183 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1346.967 (diagnostic) |
| 2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1.json | — | — | — | 2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1-power.csv | record-stem fallback | 88/88 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1343.670 (diagnostic) |
| 2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0.json | — | — | — | 2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0-power.csv | record-stem fallback | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0.json | — | — | — | 2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0-power.csv | record-stem fallback | 3/3 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1073.938 (diagnostic) |
| 2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000.json | — | — | 2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000-power.csv | 2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000-power.csv | explicit reference | 13/13 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1303.238 (diagnostic) |
| 2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000.json | — | — | 2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000-power.csv | 2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000-power.csv | explicit reference | 13/13 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1303.473 (diagnostic) |
| 2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge.json | — | — | 2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge-power.csv | 2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-09-27-p150a-newton-schulz-l32-b8192.json | — | — | 2026-09-27-p150a-newton-schulz-l32-b8192-power.csv | 2026-09-27-p150a-newton-schulz-l32-b8192-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000.json | — | — | 2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000-power.csv | 2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000-power.csv | explicit reference | 14/14 | 800, 1318, 1350 | contains intermediate values | 800 / 1350 | 1318 | beginning | CSV capture span diagnostic | 1304.587 (diagnostic) |
| 2026-09-28-p150a-newton-schulz-profile-breakdown-blocked.json | — | blocked | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-blocked.json | — | blocked | — | 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-blocked-power.csv | record-stem fallback | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter.json | — | completed | — | 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-power.csv | record-stem fallback | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000.json | — | completed | 2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000-power.csv | 2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000-power.csv | explicit reference | 15/15 | 800, 1343, 1350 | contains intermediate values | 800 / 1350 | 1343 | end | CSV capture span diagnostic | 1309.072 (diagnostic) |
| 2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady.json | — | completed | — | 2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady-power.csv | record-stem fallback | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000.json | — | pass | 2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000-power.csv | explicit reference | 4/4 | 800, 1343, 1350 | contains intermediate values | 800 / 1350 | 1343 | end | CSV capture span diagnostic | 1163.520 (diagnostic) |
| 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000.json | — | — | 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000-power.csv | explicit reference | 5/5 | 800, 1343, 1350 | contains intermediate values | 800 / 1350 | 1343 | middle | CSV capture span diagnostic | 1210.390 (diagnostic) |
| 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block8-blocked.json | — | blocked | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000.json | — | completed | 2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000-power.csv | explicit reference | 15/15 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1310.768 (diagnostic) |
| 2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block8-preflight-blocked.json | — | blocked | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-block8-compile-blocked.json | — | blocked | 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-block8-compile-blocked-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-block8-compile-blocked-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000.json | — | — | 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000-power.csv | explicit reference | 4/4 | 800, 1337, 1350 | contains intermediate values | 800 / 1350 | 1337 | end | CSV capture span diagnostic | 1166.543 (diagnostic) |
| 2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000.json | — | completed | 2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000-power.csv | explicit reference | 14/14 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1307.720 (diagnostic) |
| 2026-10-01-p150a-newton-schulz-l32-b8192-x0-bf16-blocked.json | — | blocked | 2026-10-01-p150a-newton-schulz-l32-b8192-x0-bf16-blocked-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-x0-bf16-blocked-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-01-p150a-stage1-health-container-cleanup.json | — | — | 2026-10-01-p150a-stage1-health-container-cleanup-power.csv | 2026-10-01-p150a-stage1-health-container-cleanup-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-02-host-newton-schulz-fidelity-switch-source-audit.json | — | completed | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json | — | — | 2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000-power.csv | 2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000-power.csv | explicit reference | 14/14 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1307.490 (diagnostic) |
| 2026-10-02-p150a-newton-schulz-l16-l32-acceptance-catalog-1000.json | — | — | power-20261002T000308Z.csv | 2026-10-02-p150a-newton-schulz-l16-l32-acceptance-catalog-1000-power.csv | record-stem fallback | 10/10 | 800, 1325, 1337, 1343, 1350 | contains intermediate values | 800 / 1350 | 1325 | middle | CSV capture span diagnostic | 1284.367 (diagnostic) |
| 2026-10-02-p150a-newton-schulz-l32-b8192-fidelity-split-catalog-1000.json | — | — | 2026-10-02-p150a-newton-schulz-l32-b8192-fidelity-split-catalog-1000-power.csv | 2026-10-02-p150a-newton-schulz-l32-b8192-fidelity-split-catalog-1000-power.csv | explicit reference | 14/14 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1307.425 (diagnostic) |
| 2026-10-03-host-newton-schulz-reference-sweep.json | — | — | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-03-host-newton-schulz-two-tile-source-audit.json | — | — | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-03-host-throughput-condition-audit.json | — | incomplete_pending_same_run_no_watcher_rerun | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000.json | — | — | 2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000-power.csv | 2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000-power.csv | explicit reference | 5/5 | 800, 1343, 1350 | contains intermediate values | 800 / 1350 | 1343 | middle | CSV capture span diagnostic | 1210.407 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-full-pass.json | — | pass | 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-full-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-full-pass-power.csv | explicit reference | 3/3 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1074.923 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-half-pass.json | — | pass | 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-half-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-half-pass-power.csv | explicit reference | 3/3 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1074.641 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-blocked.json | — | blocked_numerical_failure | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-blocked-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-blocked-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-pass.json | — | pass | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-pass-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-b-pass.json | — | pass | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-b-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-b-pass-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-c-pass.json | — | pass | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-c-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-c-pass-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-d-pass.json | — | pass | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-d-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-d-pass-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-b8192-correctness-blocked.json | — | blocked_numerical_failure | 2026-10-03-p150a-newton-schulz-two-tile-b8192-correctness-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-b8192-correctness-blocked-power.csv | explicit reference | 8/8 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1192.534 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-batch4-blocked.json | — | blocked_after_later_abnormal_debug_exit | 2026-10-03-p150a-stage1-health-after-two-tile-debug-reset-power.csv | 2026-10-03-p150a-stage1-health-after-two-tile-debug-reset-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000.json | — | — | 2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000-power.csv | explicit reference | 7/7 | 800, 1337, 1343, 1350 | contains intermediate values | 800 / 1350 | 1337 | middle | CSV capture span diagnostic | 1254.257 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a-blocked.json | — | fail | 2026-10-03-p150a-newton-schulz-two-tile-probe-a-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-a-blocked-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a1-blocked.json | — | fail | 2026-10-03-p150a-newton-schulz-two-tile-probe-a1-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-a1-blocked-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a3-pass.json | — | pass | 2026-10-03-p150a-newton-schulz-two-tile-probe-a3-pass-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-a3-pass-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-b-blocked.json | — | blocked_timeout_or_abnormal_exit | 2026-10-03-p150a-newton-schulz-two-tile-probe-b-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-b-blocked-power.csv | explicit reference | 29/29 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1330.296 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-b-prime-pass.json | — | pass | power-20261003T033716Z.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-b-prime-pass-power.csv | SHA-256 provenance | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-c-blocked.json | — | fail | power-20261003T033733Z.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-c-blocked-power.csv | SHA-256 provenance | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-c-pass.json | — | pass | 2026-10-03-p150a-newton-schulz-two-tile-probe-c-pass-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-c-pass-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-stage1-health-after-two-tile-b-reset.json | — | pass | 2026-10-03-p150a-stage1-health-after-two-tile-b-reset-power.csv | 2026-10-03-p150a-stage1-health-after-two-tile-b-reset-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-03-p150a-stage1-health-after-two-tile-debug-reset.json | — | pass | 2026-10-03-p150a-stage1-health-after-two-tile-debug-reset-power.csv | 2026-10-03-p150a-stage1-health-after-two-tile-debug-reset-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-counter-decomposition.json | #92 | completed | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32.json | #92 | completed | 2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32-power.csv | 2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32-power.csv | explicit reference | 3/3 | 800, 1343, 1350 | contains intermediate values | 800 / 1350 | 1343 | middle | CSV capture span diagnostic | 1070.517 (diagnostic) |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-profile-compute-decomposition.json | #92 | completed | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-risc-profiler-diagnostic.json | #92 | completed | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck.json | #94 | completed | 2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck-power.csv | 2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck-power.csv | explicit reference | 5/5 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1211.776 (diagnostic) |
| 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-superseding.json | #63 | pass | 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-power.csv | 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-power.csv | explicit reference | 3/3 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1074.019 (diagnostic) |
| 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control.json | #63 | pass | 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-power.csv | 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-power.csv | explicit reference | 3/3 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1074.019 (diagnostic) |
| 2026-10-04-p150a-newton-schulz-partial-block-padding-fuse-s-true.json | #63 | pass | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-04-p150a-newton-schulz-stage2-bf16-diagnostic.json | — | — | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-04-p150a-newton-schulz-stage2-bf16-full-diagnostic.json | — | — | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-04-p150a-newton-schulz-stage2-bf16-provenance-supplement.json | — | — | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-04-p150a-newton-schulz-stage2-pr90-current-head.json | — | — | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000.json | #96 | completed | 2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000-power.csv | 2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000-power.csv | explicit reference | 27/27 | 800, 1325, 1337, 1343, 1350 | contains intermediate values | 800 / 1350 | 1325 | beginning | CSV capture span diagnostic | 1326.296 (diagnostic) |
| 2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000.json | #100 | completed | 2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000-power.csv | 2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000-power.csv | explicit reference | 4/4 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1167.063 (diagnostic) |
| 2026-10-05-p150a-newton-schulz-issue101-default-correctness.json | #101 | pass | 2026-10-05-p150a-newton-schulz-issue101-default-correctness-power.csv | 2026-10-05-p150a-newton-schulz-issue101-default-correctness-power.csv | explicit reference | 4/4 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 980.293 (diagnostic) |
| 2026-10-06-p150a-issue104-sampler-5s.json | #104 | ok | 2026-10-06-p150a-issue104-sampler-5s-power.csv | 2026-10-06-p150a-issue104-sampler-5s-power.csv | explicit reference | 77/77 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1342.788 (diagnostic) |
| 2026-10-06-p150a-issue104-sampler-default.json | #104 | ok | 2026-10-06-p150a-issue104-sampler-default-power.csv | 2026-10-06-p150a-issue104-sampler-default-power.csv | explicit reference | 181/181 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1346.949 (diagnostic) |
| 2026-10-06-p150a-issue104-sampler-off.json | #104 | ok | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-06-p150a-issue12-stage1-60000-preflight-rejected.json | #12 | rejected | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json | #12 | ok | 2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005-power.csv | explicit reference | 226/226 | 800, 1343, 1350 | contains intermediate values | 800 / 1350 | 1343 | beginning | CSV capture span diagnostic | 1347.527 (diagnostic) |
| 2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005.json | #12 | ok | 2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-06-p150a-issue12-stage1-final-500000-adr0005.json | #12 | ok | 2026-10-06-p150a-issue12-stage1-final-500000-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-final-500000-adr0005-power.csv | explicit reference | 226/226 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1347.550 (diagnostic) |
| 2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005.json | #12 | ok | 2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005.json | #12 | ok | 2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005-power.csv | explicit reference | 29/29 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1330.380 (diagnostic) |
| 2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005.json | #12 | ok | 2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005-power.csv | explicit reference | 270/270 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1347.959 (diagnostic) |
| 2026-10-06-p150a-newton-schulz-issue101-combined-catalog-1000.json | #101 | pass | 2026-10-06-p150a-newton-schulz-issue101-combined-catalog-1000-power.csv | 2026-10-06-p150a-newton-schulz-issue101-combined-catalog-1000-power.csv | explicit reference | 5/5 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1211.749 (diagnostic) |
| 2026-10-07-host-newton-schulz-issue88-fp32-r-plan.json | #88 | planned | — | — | no reference | — | — | — | — | — | — | no CSV trace | — |
| 2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005.json | #12 | ok | 2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005-power.csv | 2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005-power.csv | explicit reference | 226/226 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1347.561 (diagnostic) |
| 2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005.json | #12 | ok | 2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005-power.csv | 2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005-power.csv | explicit reference | 2/2 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 800 (diagnostic) |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-beam-metric-summary.json | #88 | — | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | explicit reference | 9/9 | 800, 1262, 1350 | contains intermediate values | 800 / 1350 | 1262 | end | CSV capture span diagnostic | 1269.329 (diagnostic) |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-summary.json | #88 | — | — | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | record-stem fallback | 9/9 | 800, 1262, 1350 | contains intermediate values | 800 / 1350 | 1262 | end | CSV capture span diagnostic | 1269.329 (diagnostic) |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json | #88 | pass | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | explicit reference | 9/9 | 800, 1262, 1350 | contains intermediate values | 1262 / 1350 | 1262 | end | exact run interval | 1266.831 (exact) |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup.json | #88 | pass | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup-power.csv | explicit reference | 6/6 | 800, 1312, 1350 | contains intermediate values | 800 / 1350 | 1312 | end | CSV capture span diagnostic | 1232.249 (diagnostic) |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r.json | #88 | pass | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-power.csv | explicit reference | 6/6 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1240.185 (diagnostic) |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-l1-accounting.json | #88 | — | — | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv | record-stem fallback | 10/10 | 800, 1331, 1350 | contains intermediate values | 800 / 1350 | 1331 | middle | CSV capture span diagnostic | 1288.401 (diagnostic) |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-summary.json | #88 | pass | issue88-fp32-r-power-snapshot-20261009T161013132476761435322545Z-4.csv | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv | SHA-256 provenance | 10/10 | 800, 1331, 1350 | contains intermediate values | 1331 / 1350 | 1331 | middle | exact run interval | 1290.615 (exact) |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json | #88 | pass | issue88-fp32-r-power-snapshot-20261009T161013132476761435322545Z-4.csv | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv | SHA-256 provenance | 10/10 | 800, 1331, 1350 | contains intermediate values | 1331 / 1350 | 1331 | middle | exact run interval | 1290.615 (exact) |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json | #88 | pass | issue88-fp32-r-power-snapshot-20261010T0356528751606238822231592Z-4.csv | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv | SHA-256 provenance | 10/10 | 800, 1343, 1350 | contains intermediate values | 1343 / 1350 | 1343 | beginning | exact run interval | 1289.196 (exact) |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json | #88 | pass | issue88-fp32-r-power-snapshot-20261010T0356528751606238822231592Z-4.csv | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv | SHA-256 provenance | 10/10 | 800, 1343, 1350 | contains intermediate values | 1343 / 1350 | 1343 | beginning | exact run interval | 1289.196 (exact) |
| 2026-10-11-p150a-issue118-control-500000-adr0005.json | #118 | ok | 2026-10-11-p150a-issue118-control-500000-adr0005-power.csv | 2026-10-11-p150a-issue118-control-500000-adr0005-power.csv | explicit reference | 226/226 | 800, 1350 | two-valued {800,1350} | 1350 / 1350 | — | — | bounds present; trace coverage incomplete (load) | — |
| 2026-10-11-p150a-issue118-dram-500000-adr0005.json | #118 | ok | 2026-10-11-p150a-issue118-dram-500000-adr0005-power.csv | 2026-10-11-p150a-issue118-dram-500000-adr0005-power.csv | explicit reference | 248/248 | 800, 1350 | two-valued {800,1350} | 1350 / 1350 | — | — | exact load interval | 1349.842 (exact) |
| 2026-10-11-p150a-issue118-newton-schulz-500000-adr0005.json | #118 | ok | 2026-10-11-p150a-issue118-newton-schulz-500000-adr0005-power.csv | 2026-10-11-p150a-issue118-newton-schulz-500000-adr0005-power.csv | explicit reference | 319/319 | 800, 1018, 1025, 1031, 1037, 1043, 1050, 1056, 1062, 1068, 1075, 1081, 1087, 1168, 1268, 1300, 1306, 1350 | contains intermediate values | 1018 / 1350 | 1018 | middle | exact load interval | 1050.914 (exact) |

## Committed CSV inventory

| CSV | JSON records using it | rows valid/CSV | unique AICLK (MHz) | classification | min intermediate (MHz) | position | invalid rows |
|---|---|---:|---|---|---:|---|---:|
| 2026-08-14-p150a-effective-efficiency-power.csv | 2026-08-14-p150a-effective-efficiency.json | 89/89 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0-power.csv | 2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0.json | 183/183 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1-power.csv | 2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1.json | 88/88 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0-power.csv | 2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0-power.csv | 2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0.json | 3/3 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000-power.csv | 2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000.json | 13/13 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000-power.csv | 2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000.json | 13/13 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-27-p150a-newton-schulz-l32-b8192-power.csv | 2026-09-27-p150a-newton-schulz-l32-b8192.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge-power.csv | 2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000-power.csv | 2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000.json | 14/14 | 800, 1318, 1350 | contains intermediate values | 1318 | beginning | 0 |
| 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-blocked-power.csv | 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-blocked.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-power.csv | 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000-power.csv | 2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000.json | 15/15 | 800, 1343, 1350 | contains intermediate values | 1343 | end | 0 |
| 2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady-power.csv | 2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000.json | 4/4 | 800, 1343, 1350 | contains intermediate values | 1343 | end | 0 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000.json | 5/5 | 800, 1343, 1350 | contains intermediate values | 1343 | middle | 0 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000.json | 15/15 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-block8-compile-blocked-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-block8-compile-blocked.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000.json | 4/4 | 800, 1337, 1350 | contains intermediate values | 1337 | end | 0 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000.json | 14/14 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-x0-bf16-blocked-power.csv | 2026-10-01-p150a-newton-schulz-l32-b8192-x0-bf16-blocked.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-01-p150a-stage1-health-container-cleanup-power.csv | 2026-10-01-p150a-stage1-health-container-cleanup.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000-power.csv | 2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json | 14/14 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-02-p150a-newton-schulz-l16-l32-acceptance-catalog-1000-power.csv | 2026-10-02-p150a-newton-schulz-l16-l32-acceptance-catalog-1000.json | 10/10 | 800, 1325, 1337, 1343, 1350 | contains intermediate values | 1325 | middle | 0 |
| 2026-10-02-p150a-newton-schulz-l32-b8192-fidelity-split-catalog-1000-power.csv | 2026-10-02-p150a-newton-schulz-l32-b8192-fidelity-split-catalog-1000.json | 14/14 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000-power.csv | 2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000.json | 5/5 | 800, 1343, 1350 | contains intermediate values | 1343 | middle | 0 |
| 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-full-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-full-pass.json | 3/3 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-half-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-half-pass.json | 3/3 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-blocked-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-blocked.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-pass.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-b-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-b-pass.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-c-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-c-pass.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-d-pass-power.csv | 2026-10-03-p150a-newton-schulz-production-two-tile-stage-d-pass.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-b8192-correctness-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-b8192-correctness-blocked.json | 8/8 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-batch4-blocked-power.csv | — | 3/3 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000.json | 7/7 | 800, 1337, 1343, 1350 | contains intermediate values | 1337 | middle | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-a-blocked.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a1-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-a1-blocked.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a3-pass-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-a3-pass.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-b-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-b-blocked.json | 29/29 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-b-prime-pass-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-b-prime-pass.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-c-blocked-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-c-blocked.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-c-pass-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-probe-c-pass.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-stage1-health-after-two-tile-b-reset-power.csv | 2026-10-03-p150a-stage1-health-after-two-tile-b-reset.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-03-p150a-stage1-health-after-two-tile-debug-reset-power.csv | 2026-10-03-p150a-newton-schulz-two-tile-batch4-blocked.json; 2026-10-03-p150a-stage1-health-after-two-tile-debug-reset.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32-power.csv | 2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32.json | 3/3 | 800, 1343, 1350 | contains intermediate values | 1343 | middle | 0 |
| 2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck-power.csv | 2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck.json | 5/5 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-power.csv | 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-superseding.json; 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control.json | 3/3 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-superseding-power.csv | — | 3/3 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000-power.csv | 2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000.json | 27/27 | 800, 1325, 1337, 1343, 1350 | contains intermediate values | 1325 | beginning | 0 |
| 2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000-power.csv | 2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000.json | 4/4 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-05-p150a-newton-schulz-issue101-default-correctness-power.csv | 2026-10-05-p150a-newton-schulz-issue101-default-correctness.json | 4/4 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-06-p150a-issue104-sampler-5s-power.csv | 2026-10-06-p150a-issue104-sampler-5s.json | 77/77 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-06-p150a-issue104-sampler-default-power.csv | 2026-10-06-p150a-issue104-sampler-default.json | 181/181 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json | 226/226 | 800, 1343, 1350 | contains intermediate values | 1343 | beginning | 0 |
| 2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-06-p150a-issue12-stage1-final-500000-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-final-500000-adr0005.json | 226/226 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005.json | 29/29 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005-power.csv | 2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005.json | 270/270 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-06-p150a-newton-schulz-issue101-combined-catalog-1000-power.csv | 2026-10-06-p150a-newton-schulz-issue101-combined-catalog-1000.json | 5/5 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005-power.csv | 2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005.json | 226/226 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005-power.csv | 2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005.json | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-beam-metric-summary.json; 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-summary.json; 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json | 9/9 | 800, 1262, 1350 | contains intermediate values | 1262 | end | 0 |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-first-launch-power.csv | — | 2/2 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup.json | 6/6 | 800, 1312, 1350 | contains intermediate values | 1312 | end | 0 |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r.json | 6/6 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-l1-accounting.json; 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-summary.json; 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json | 10/10 | 800, 1331, 1350 | contains intermediate values | 1331 | middle | 0 |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json; 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json | 10/10 | 800, 1343, 1350 | contains intermediate values | 1343 | beginning | 0 |
| 2026-10-11-p150a-issue118-control-500000-adr0005-power.csv | 2026-10-11-p150a-issue118-control-500000-adr0005.json | 226/226 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-11-p150a-issue118-dram-500000-adr0005-power.csv | 2026-10-11-p150a-issue118-dram-500000-adr0005.json | 248/248 | 800, 1350 | two-valued {800,1350} | — | — | 0 |
| 2026-10-11-p150a-issue118-newton-schulz-500000-adr0005-power.csv | 2026-10-11-p150a-issue118-newton-schulz-500000-adr0005.json | 319/319 | 800, 1018, 1025, 1031, 1037, 1043, 1050, 1056, 1062, 1068, 1075, 1081, 1087, 1168, 1268, 1300, 1306, 1350 | contains intermediate values | 1018 | middle | 0 |

## Explicit interval records

| record | issue | interval kind | evidence | min/max (MHz) | classification | weighted AICLK (MHz) | resolved CSV |
|---|---|---|---|---:|---|---:|---|
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json | #88 | run | exact run interval | 1262 / 1350 | contains intermediate values | 1266.831 (exact) | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-summary.json | #88 | run | exact run interval | 1331 / 1350 | contains intermediate values | 1290.615 (exact) | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json | #88 | run | exact run interval | 1331 / 1350 | contains intermediate values | 1290.615 (exact) | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json | #88 | run | exact run interval | 1343 / 1350 | contains intermediate values | 1289.196 (exact) | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json | #88 | run | exact run interval | 1343 / 1350 | contains intermediate values | 1289.196 (exact) | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv |
| 2026-10-11-p150a-issue118-control-500000-adr0005.json | #118 | load | bounds present; trace coverage incomplete (load) | 1350 / 1350 | two-valued {800,1350} | — | 2026-10-11-p150a-issue118-control-500000-adr0005-power.csv |
| 2026-10-11-p150a-issue118-dram-500000-adr0005.json | #118 | load | exact load interval | 1350 / 1350 | two-valued {800,1350} | 1349.842 (exact) | 2026-10-11-p150a-issue118-dram-500000-adr0005-power.csv |
| 2026-10-11-p150a-issue118-newton-schulz-500000-adr0005.json | #118 | load | exact load interval | 1018 / 1350 | contains intermediate values | 1050.914 (exact) | 2026-10-11-p150a-issue118-newton-schulz-500000-adr0005-power.csv |

## Issue #12 and #104 resident records

| record | issue | classification | min/max (MHz) | interval evidence | weighted AICLK (MHz) | resolved CSV |
|---|---|---|---:|---|---:|---|
| 2026-10-06-p150a-issue104-sampler-5s.json | #104 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 1342.788 (diagnostic) | 2026-10-06-p150a-issue104-sampler-5s-power.csv |
| 2026-10-06-p150a-issue104-sampler-default.json | #104 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 1346.949 (diagnostic) | 2026-10-06-p150a-issue104-sampler-default-power.csv |
| 2026-10-06-p150a-issue104-sampler-off.json | #104 | no trace | — | no CSV trace | — | — |
| 2026-10-06-p150a-issue12-stage1-60000-preflight-rejected.json | #12 | no trace | — | no CSV trace | — | — |
| 2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json | #12 | contains intermediate values | 800 / 1350 | CSV capture span diagnostic | 1347.527 (diagnostic) | 2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005-power.csv |
| 2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005.json | #12 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 800 (diagnostic) | 2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005-power.csv |
| 2026-10-06-p150a-issue12-stage1-final-500000-adr0005.json | #12 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 1347.550 (diagnostic) | 2026-10-06-p150a-issue12-stage1-final-500000-adr0005-power.csv |
| 2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005.json | #12 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 800 (diagnostic) | 2026-10-06-p150a-issue12-stage1-final-watcher-100-adr0005-power.csv |
| 2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005.json | #12 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 1330.380 (diagnostic) | 2026-10-06-p150a-issue12-stage1-semaphore-timing-60000-adr0005-power.csv |
| 2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005.json | #12 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 1347.959 (diagnostic) | 2026-10-06-p150a-issue12-stage1-semaphore-timing-600000-adr0005-power.csv |
| 2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005.json | #12 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 1347.561 (diagnostic) | 2026-10-07-p150a-issue12-stage1-board-id-alias-500000-adr0005-power.csv |
| 2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005.json | #12 | two-valued {800,1350} | 800 / 1350 | CSV capture span diagnostic | 800 (diagnostic) | 2026-10-07-p150a-issue12-stage1-board-id-alias-watcher-100-adr0005-power.csv |

## Intermediate-value CSV diagnostics

| CSV | minimum intermediate (MHz) | position | unique AICLK (MHz) |
|---|---:|---|---|
| 2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000-power.csv | 1318 | beginning | 800, 1318, 1350 |
| 2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000-power.csv | 1343 | end | 800, 1343, 1350 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000-power.csv | 1343 | end | 800, 1343, 1350 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000-power.csv | 1343 | middle | 800, 1343, 1350 |
| 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000-power.csv | 1337 | end | 800, 1337, 1350 |
| 2026-10-02-p150a-newton-schulz-l16-l32-acceptance-catalog-1000-power.csv | 1325 | middle | 800, 1325, 1337, 1343, 1350 |
| 2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000-power.csv | 1343 | middle | 800, 1343, 1350 |
| 2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000-power.csv | 1337 | middle | 800, 1337, 1343, 1350 |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32-power.csv | 1343 | middle | 800, 1343, 1350 |
| 2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000-power.csv | 1325 | beginning | 800, 1325, 1337, 1343, 1350 |
| 2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005-power.csv | 1343 | beginning | 800, 1343, 1350 |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | 1262 | end | 800, 1262, 1350 |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup-power.csv | 1312 | end | 800, 1312, 1350 |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv | 1331 | middle | 800, 1331, 1350 |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv | 1343 | beginning | 800, 1343, 1350 |
| 2026-10-11-p150a-issue118-newton-schulz-500000-adr0005-power.csv | 1018 | middle | 800, 1018, 1025, 1031, 1037, 1043, 1050, 1056, 1062, 1068, 1075, 1081, 1087, 1168, 1268, 1300, 1306, 1350 |
## #63-family Newton-Schulz timing/source audit

The committed #63-family slice contains 59 JSON records dated 2026-09-27 through 2026-10-06. 30 records contain published timing/TFLOPS fields; 0 contain an explicit AICLK-to-elapsed conversion field.

| record | source/runner named by record | published timing fields | AICLK conversion fields |
|---|---|---|---|
| 2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge.json | other Newton-Schulz source | yes | no |
| 2026-09-27-p150a-newton-schulz-l32-b8192.json | other Newton-Schulz source | yes | no |
| 2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-09-28-p150a-newton-schulz-profile-breakdown-blocked.json | run_matmul.py | no | no |
| 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-blocked.json | other Newton-Schulz source | yes | no |
| 2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter.json | other Newton-Schulz source | yes | no |
| 2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady.json | other Newton-Schulz source | yes | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-block4-l1-history-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-input-memory-block8-blocked.json | other Newton-Schulz source | no | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-matrix-block8-preflight-blocked.json | other Newton-Schulz source | no | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-block8-compile-blocked.json | other Newton-Schulz source | no | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-per-input-memory-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-01-p150a-newton-schulz-l32-b8192-x0-bf16-blocked.json | not stated | no | no |
| 2026-10-02-host-newton-schulz-fidelity-switch-source-audit.json | not stated | no | no |
| 2026-10-02-p150a-newton-schulz-l16-b8192-diagonal-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-02-p150a-newton-schulz-l16-l32-acceptance-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-02-p150a-newton-schulz-l32-b8192-fidelity-split-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-03-host-newton-schulz-reference-sweep.json | not stated | no | no |
| 2026-10-03-host-newton-schulz-two-tile-source-audit.json | other Newton-Schulz source | no | no |
| 2026-10-03-p150a-newton-schulz-l32-b8192-no-watcher-one-tile-catalog-1000.json | run_matmul.py | yes | no |
| 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-full-pass.json | newton_schulz_runner.py (record reference) | no | no |
| 2026-10-03-p150a-newton-schulz-production-two-tile-b8192-half-pass.json | newton_schulz_runner.py (record reference) | no | no |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-blocked.json | newton_schulz_runner.py (record reference) | no | no |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-a-pass.json | newton_schulz_runner.py (record reference) | no | no |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-b-pass.json | newton_schulz_runner.py (record reference) | no | no |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-c-pass.json | newton_schulz_runner.py (record reference) | no | no |
| 2026-10-03-p150a-newton-schulz-production-two-tile-stage-d-pass.json | newton_schulz_runner.py (record reference) | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-b8192-correctness-blocked.json | other Newton-Schulz source | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-batch4-blocked.json | run_matmul.py | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-mixed-memory-catalogue-1000.json | run_matmul.py | yes | no |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a-blocked.json | other Newton-Schulz source | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a1-blocked.json | other Newton-Schulz source | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-a3-pass.json | not stated | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-b-blocked.json | other Newton-Schulz source | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-b-prime-pass.json | not stated | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-c-blocked.json | not stated | no | no |
| 2026-10-03-p150a-newton-schulz-two-tile-probe-c-pass.json | not stated | no | no |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-counter-decomposition.json | run_matmul.py | no | no |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-l16-l32.json | other Newton-Schulz source | yes | no |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-profile-compute-decomposition.json | other Newton-Schulz source | no | no |
| 2026-10-04-p150a-newton-schulz-block-double-buffer-risc-profiler-diagnostic.json | run_matmul.py | yes | no |
| 2026-10-04-p150a-newton-schulz-issue94-bf16-state-recheck.json | run_matmul.py | yes | no |
| 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control-superseding.json | other Newton-Schulz source | yes | no |
| 2026-10-04-p150a-newton-schulz-l32-b8192-r-residency-control.json | other Newton-Schulz source | yes | no |
| 2026-10-04-p150a-newton-schulz-partial-block-padding-fuse-s-true.json | other Newton-Schulz source | yes | no |
| 2026-10-04-p150a-newton-schulz-stage2-bf16-diagnostic.json | other Newton-Schulz source | no | no |
| 2026-10-04-p150a-newton-schulz-stage2-bf16-full-diagnostic.json | other Newton-Schulz source | yes | no |
| 2026-10-04-p150a-newton-schulz-stage2-bf16-provenance-supplement.json | not stated | no | no |
| 2026-10-04-p150a-newton-schulz-stage2-pr90-current-head.json | other Newton-Schulz source | yes | no |
| 2026-10-05-p150a-newton-schulz-half-sync-l16-l32-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-05-p150a-newton-schulz-issue100-defaults-catalog-1000.json | other Newton-Schulz source | yes | no |
| 2026-10-05-p150a-newton-schulz-issue101-default-correctness.json | other Newton-Schulz source | no | no |
| 2026-10-06-p150a-newton-schulz-issue101-combined-catalog-1000.json | newton_schulz_issue101_combined.py | yes | no |

Source inspection result: `enodia/tt/bench/run_matmul.py` functions `run_stock` and `run_custom_newton_schulz` bracket device launches plus `ttnn.synchronize_device` with host `time.perf_counter()`. They publish `seconds_per_iteration`, `seconds_per_launch_*`, `achieved_tflops`, and `tflops_*_derived`; the latter are FLOPs divided by those host elapsed seconds. `tools/newton_schulz_issue100_same_run.py:run_issue100_comparison` delegates to `run_custom_newton_schulz`, and `tools/newton_schulz_issue101_combined.py:_run_performance_with_capture` uses the same path.

The records dated 2026-09-27 through 2026-10-06 contain no persisted `aiclk_mhz_load_interval_weighted`, `aiclk_mhz_for_elapsed_seconds`, or equivalent cycle-to-time conversion for their published TFLOPS/seconds. Their AICLK columns are telemetry context, not the denominator used by the published host timings. The 2026-10-03 correctness records reference `tools/newton_schulz_runner.py`, which is not present in this checkout and does not publish timing fields in the committed records; that gap does not support an AICLK conversion claim. The later `enodia/tt/bench/power_trace_snapshot.py:_time_weighted_aiclk` helper is the explicit-interval path used by #118, not by these older #63 records.

## Follow-up findings

### #88 R-format ordering and confounding

The `comparison_rows` arrays in the convergence rerun, device-1 retake, and final runner retake are block ordered, not alternating: `bf16-r-L16`, `fp32-r-L16`, `bf16-r-L32`, `bf16-r-L32-r-dram`, `fp32-r-L32-r-dram`, and the rejected `fp32-r-L32` row. Each row retains host timing samples and derived seconds/TFLOPS fields, but no row start or cumulative host timestamp that can be aligned to a power-CSV timestamp. Therefore per-condition AICLK min/max/weighted values cannot be calculated from the committed evidence. The defensible bound for any row is the complete run-trace AICLK range, not a condition-specific estimate:

| record | complete run-trace AICLK bound (MHz) | AICLK-only range (`max/min - 1`) | per-row alignment |
|---|---:|---:|---|
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json | 1262–1350 | 6.973% | unavailable |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json | 1331–1350 | 1.427% | unavailable |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json | 1343–1350 | 0.521% | unavailable |

Each AICLK-only range is a model-dependent upper bound under linear AICLK-to-throughput normalization: it assumes the condition could fall at either complete-run endpoint. It is not a replacement for the synchronized host timings retained by each row; row-to-power alignment remains unavailable.

For the authoritative final trace, the complete-run bound gives a 0.521% maximum clock-rate spread. The authoritative summary's retained P50 TFLOPS values give observed P50 throughput changes of L16/L1: -1.694%; L32/DRAM: -4.681%; even under the linear normalization model, the 0.521% clock-rate spread cannot by itself explain those observed throughput changes.

Relative to `docs/design.md`, the current published #88 conclusion is unchanged: FP32-R improves the corrected quality metrics but costs throughput/L1; only the withdrawn defective `_weight` conclusion changed. The withdrawn report's claim that FP32-R worsened beam-pattern error used the defective `_weight` contraction. The corrected final evidence says FP32-R lowers inverse error, MV cosine deficit, and all four beam indicators at both matched placements; BF16-R remains the default cost choice.

### No-bound old records

A trace classified as `two-valued {800,1350}` contains only the two endpoint values. A trace classified as `contains intermediate values` reports the minimum value outside that set and the thirds-based CSV position of its occurrence. Without explicit covered bounds, all such min/max/weighted values are capture-span diagnostics, never in-run timing evidence.

Operator-requested assumption (not proof): for a trace whose only values are 800 and 1350, treat 800 as idle outside the measured interval and 1350 as the measured clock. Under that assumption, using maximum 1350 then has zero AICLK conversion error. The trace alone does not prove that assumption or the interval coverage.

Issue #12 resident records include both endpoint-only traces and historical intermediate-value traces: the current-wrap capture has minimum intermediate 1343 MHz at the beginning of its CSV, while the final corrected-drain and board-id-alias timing captures are endpoint-only; watcher validations and the preflight rejection have no intermediate evidence or no trace. Issue #104's sampler-default and sampler-5s resident traces are endpoint-only; its sampler-off record has no CSV by design. See the focused tables above for every committed #12 and #104 JSON record.

### #118 weighted comparisons

| condition | prior published weighted AICLK (MHz) | audit result | evidence |
|---|---:|---:|---|
| unloaded control | 1,350 reference | unavailable | bounds are not covered through the final CSV sample; no weighted evidence |
| DRAM | 1,349.842 | 1,349.842 | exact left-held load interval |
| Newton-Schulz | 1,050.914 | 1,050.914 | exact left-held load interval |

These values reproduce the prior audit's exact #118 comparisons to three decimal MHz without rewriting any measurement leaf. The control remains a 1,350 MHz reference rather than a computed weighted load interval.
