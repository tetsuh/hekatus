# Issue #122 AICLK audit (generated)

- Input root: `docs/measurements`
- Generation command: `python3 tools/audit_issue122_aiclk.py`
- JSON records: 95
- CSV files: 72
- Interval policy: valid timestamped rows are left-held until the next row; an explicit interval is exact only when ordered trace rows cover both bounds. Missing bounds produce a `CSV capture span diagnostic`; they do not become in-run evidence. Intermediate means any valid AICLK outside `{800,1350}`; its position uses the beginning/middle/end thirds of the CSV.

## Post-#124 cycle-ticks-to-seconds conversion inventory

Line numbers are generated from the current production source. The source column retains the machine-readable record tag where one exists and the semantic selection beside it.

| path | line | conversion path | selected value | chosen source | interpretation |
|---|---:|---|---|---|---|
| enodia/tt/bench/resident_clock.py | 156 | shared ticks → seconds helper | selected AICLK (exact when weighted) | selection['aiclk_source'] / semantic selection | all cycle→time callers must pass a selected value |
| enodia/tt/bench/resident_record.py | 2223 | _sampler_comparison ticks → seconds | elapsed_aiclk_selection | resident elapsed-time source (configured, in-run maximum, or load weighted) | sampler gap seconds |
| enodia/tt/bench/resident_record.py | 2353 | build_outlier_analysis ticks → seconds | elapsed_selection['aiclk_mhz'] | aiclk_source_for_elapsed_seconds | resident event/pair/run elapsed seconds |
| enodia/tt/bench/resident_record.py | 2357 | build_outlier_analysis ticks → seconds | elapsed_selection['aiclk_mhz'] | aiclk_source_for_elapsed_seconds | resident event/pair/run elapsed seconds |
| enodia/tt/bench/resident_record.py | 2365 | build_outlier_analysis ticks → seconds | elapsed_selection['aiclk_mhz'] | aiclk_source_for_elapsed_seconds | resident event/pair/run elapsed seconds |
| enodia/tt/bench/resident_record.py | 2417 | build_outlier_analysis ticks → seconds | elapsed_selection['aiclk_mhz'] | aiclk_source_for_elapsed_seconds | resident event/pair/run elapsed seconds |
| enodia/tt/bench/resident_record.py | 3042 | validate_pair_analysis ticks → seconds | record elapsed selection | persisted aiclk_source_for_elapsed_seconds | validator recomputation of resident elapsed seconds |
| enodia/tt/bench/resident_record.py | 3051 | validate_pair_analysis ticks → seconds | record elapsed selection | persisted aiclk_source_for_elapsed_seconds | validator recomputation of resident elapsed seconds |
| enodia/tt/bench/resident_record.py | 3078 | validate_pair_analysis ticks → seconds | record elapsed selection | persisted aiclk_source_for_elapsed_seconds | validator recomputation of resident elapsed seconds |
| tools/analyze_issue104_wrap.py | 293 | analyze_run ticks → seconds | aiclk_selection | legacy_unverified for historical sampled records; trace selection otherwise | historical wrap analysis preserves numeric results |
| tools/analyze_issue104_wrap.py | 375 | analyze_run ticks → seconds | aiclk_selection | legacy_unverified for historical sampled records; trace selection otherwise | historical wrap analysis preserves numeric results |
| tools/analyze_issue104_wrap.py | 474 | analyze_issue104 ticks → seconds | configured 1350 MHz default selection | configured budget | historical wrap analysis preserves numeric results |
| enodia/tt/bench/run_resident.py | 1451 | throughput reference ticks → seconds | 1350 MHz fixed reference | fixed_reference_1350_mhz | normalization denominator, not observed AICLK |
| enodia/tt/bench/run_resident.py | 1465 | throughput reference ticks → seconds | 1350 MHz fixed reference | fixed_reference_1350_mhz | normalization denominator, not observed AICLK |
| enodia/tt/bench/run_resident.py | 1473 | throughput reference ticks → seconds | 1350 MHz fixed reference | fixed_reference_1350_mhz | normalization denominator, not observed AICLK |
| enodia/tt/bench/run_resident.py | 1511 | calibration actual duration ticks → seconds | throughput_aiclk_mhz | actual_duration_aiclk_source | weighted for loaded records; selected run clock otherwise |
| enodia/tt/bench/load_harness.py | 617 | predicted load card ticks → seconds | min(weighted load interval, in-run minimum) when minimum is present | predicted_load_time_aiclk_source | time_weighted_load_interval or in_run_minimum |
| enodia/tt/bench/load_harness.py | 1308 | persisted throughput reference ticks → seconds | 1350 MHz fixed reference | fixed_reference_1350_mhz | historical repeats_per_card_second normalization |
| enodia/tt/bench/load_harness.py | 1455 | persisted throughput reference ticks → seconds | 1350 MHz fixed reference | fixed_reference_1350_mhz | historical repeats_per_card_second normalization |
| enodia/tt/bench/resident_harness.py | 519 | inverse wall time → ticks | configured budget AICLK | configured_budget | startup/budget policy, not measured elapsed time |
| enodia/tt/bench/resident_harness.py | 599 | inverse outer timeout seconds → ticks | configured budget AICLK | configured_budget | outer-cap policy, not measured elapsed time |
| enodia/tt/bench/run_resident.py | 1403 | inverse calibration target seconds → ticks | configured budget AICLK | configured_budget | calibration budget policy |
| enodia/tt/bench/run_resident.py | 1404 | inverse startup allowance → ticks | configured budget AICLK | configured_budget | calibration budget policy |
| tools/analyze_issue104_wrap.py | 72 | inverse phase-window milliseconds → ticks | selected analyzer AICLK | legacy_unverified for historical sampled records; configured off-mode default otherwise | analysis window policy, not elapsed-time evidence |

`clock.aiclk_mhz` remains the in-run maximum and is not an elapsed-time denominator for new records. Resident elapsed fields use their persisted AICLK/source pair; loaded records use the time-weighted load interval, while conservative prediction may select the in-run minimum. The 1,350 MHz throughput reference is fixed normalization. Configured budget and outer-cap conversions are inverse policy calculations.

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
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json | #88 | pass | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv | explicit reference | 9/9 | 800, 1262, 1350 | contains intermediate values | 800 / 1350 | 1262 | end | exact run interval | 1266.831 (exact) |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup.json | #88 | pass | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-placement-followup-power.csv | explicit reference | 6/6 | 800, 1312, 1350 | contains intermediate values | 800 / 1350 | 1312 | end | CSV capture span diagnostic | 1232.249 (diagnostic) |
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r.json | #88 | pass | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-power.csv | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-power.csv | explicit reference | 6/6 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | CSV capture span diagnostic | 1240.185 (diagnostic) |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-l1-accounting.json | #88 | — | — | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv | record-stem fallback | 10/10 | 800, 1331, 1350 | contains intermediate values | 800 / 1350 | 1331 | middle | CSV capture span diagnostic | 1288.401 (diagnostic) |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-summary.json | #88 | pass | issue88-fp32-r-power-snapshot-20261009T161013132476761435322545Z-4.csv | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv | SHA-256 provenance | 10/10 | 800, 1331, 1350 | contains intermediate values | 800 / 1350 | 1331 | middle | exact run interval | 1290.615 (exact) |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json | #88 | pass | issue88-fp32-r-power-snapshot-20261009T161013132476761435322545Z-4.csv | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv | SHA-256 provenance | 10/10 | 800, 1331, 1350 | contains intermediate values | 800 / 1350 | 1331 | middle | exact run interval | 1290.615 (exact) |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json | #88 | pass | issue88-fp32-r-power-snapshot-20261010T0356528751606238822231592Z-4.csv | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv | SHA-256 provenance | 10/10 | 800, 1343, 1350 | contains intermediate values | 800 / 1350 | 1343 | beginning | exact run interval | 1289.196 (exact) |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json | #88 | pass | issue88-fp32-r-power-snapshot-20261010T0356528751606238822231592Z-4.csv | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv | SHA-256 provenance | 10/10 | 800, 1343, 1350 | contains intermediate values | 800 / 1350 | 1343 | beginning | exact run interval | 1289.196 (exact) |
| 2026-10-11-p150a-issue118-control-500000-adr0005.json | #118 | ok | 2026-10-11-p150a-issue118-control-500000-adr0005-power.csv | 2026-10-11-p150a-issue118-control-500000-adr0005-power.csv | explicit reference | 226/226 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | bounds present; trace coverage incomplete (load) | — |
| 2026-10-11-p150a-issue118-dram-500000-adr0005.json | #118 | ok | 2026-10-11-p150a-issue118-dram-500000-adr0005-power.csv | 2026-10-11-p150a-issue118-dram-500000-adr0005-power.csv | explicit reference | 248/248 | 800, 1350 | two-valued {800,1350} | 800 / 1350 | — | — | exact load interval | 1349.842 (exact) |
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

The inventory contains 8 explicit-interval JSON rows. The six logical target runs are the three #88 retakes and the three #118 conditions; the count is eight because the device-1 and final-runner retakes each have both a source and summary JSON row using the same trace. All eight rows are retained below.

| record | issue | interval kind | evidence | min/max (MHz) | classification | weighted AICLK (MHz) | resolved CSV |
|---|---|---|---|---:|---|---:|---|
| 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json | #88 | run | exact run interval | 800 / 1350 | contains intermediate values | 1266.831 (exact) | 2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun-power.csv |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-summary.json | #88 | run | exact run interval | 800 / 1350 | contains intermediate values | 1290.615 (exact) | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv |
| 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json | #88 | run | exact run interval | 800 / 1350 | contains intermediate values | 1290.615 (exact) | 2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake-power.csv |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-summary.json | #88 | run | exact run interval | 800 / 1350 | contains intermediate values | 1289.196 (exact) | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv |
| 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json | #88 | run | exact run interval | 800 / 1350 | contains intermediate values | 1289.196 (exact) | 2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake-power.csv |
| 2026-10-11-p150a-issue118-control-500000-adr0005.json | #118 | load | bounds present; trace coverage incomplete (load) | 800 / 1350 | two-valued {800,1350} | — | 2026-10-11-p150a-issue118-control-500000-adr0005-power.csv |
| 2026-10-11-p150a-issue118-dram-500000-adr0005.json | #118 | load | exact load interval | 800 / 1350 | two-valued {800,1350} | 1349.842 (exact) | 2026-10-11-p150a-issue118-dram-500000-adr0005-power.csv |
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

The old `0.521%` complete-run-width claim is withdrawn. It treated the whole run as available to every condition and did not use the host timing row order. The corrected lower-bound audit below uses each target record's persisted `rows` in block order and sums the measured `seconds_per_launch_samples` for each row. It is an ordering lower bound, not a reconstructed row-to-power timestamp alignment.

The explicit-interval inventory above reports extrema and weighted values from the same positive-duration left-held trace segments. The three #88 records below additionally report every measured row's sample-length sum, cumulative start lower bound, and the 800 MHz / below-1343 MHz hold ends.

#### `2026-10-08-p150a-newton-schulz-issue88-fp32-r-convergence-rerun.json`

`run_start` = `2026-10-07T23:57:22.731169+00:00`, `run_end` = `2026-10-07T23:57:39.433229+00:00`. Rows are consumed in persisted block order. For row *i*, the start lower bound is `run_start` plus the sum of measured `seconds_per_launch_samples` from earlier rows; no p50 or synthetic duration is used.

| row order/name | status | sample count | sample length sum (s) | cumulative start lower bound (s) | lower-bound timestamp |
|---:|---|---:|---:|---:|---|
| 0: bf16-r-L16 | ok | 1000 | 0.399626350 s | 0.000000000 s | 2026-10-07T23:57:22.731169+00:00 |
| 1: fp32-r-L16 | ok | 1000 | 0.410713220 s | 0.399626350 s | 2026-10-07T23:57:23.130795+00:00 |
| 2: bf16-r-L32 | ok | 1000 | 0.662542210 s | 0.810339570 s | 2026-10-07T23:57:23.541509+00:00 |
| 3: bf16-r-L32-r-dram | ok | 1000 | 0.664225863 s | 1.472881780 s | 2026-10-07T23:57:24.204051+00:00 |
| 4: fp32-r-L32-r-dram | ok | 1000 | 0.698566721 s | 2.137107643 s | 2026-10-07T23:57:24.868277+00:00 |
| 5: fp32-r-L32 | preflight_rejected | — | — | — | — |

Positive-duration trace holds below 1343 MHz are listed below; this includes the 800 MHz hold and excludes zero-duration rows at the interval end. The end offset is measured from `run_start`.

| AICLK (MHz) | hold start offset | hold end offset | hold end timestamp |
|---:|---:|---:|---|
| 800 | 0.000000000 s | 2.137098000 s | 2026-10-07T23:57:24.868267+00:00 |
| 1262 | 13.393038000 s | 15.821391000 s | 2026-10-07T23:57:38.552560+00:00 |

For L16, both BF16-R and FP32-R can overlap the initial low-clock hold under these lower bounds, so the direction of the clock confounding cannot be determined.
The BF16-R and FP32-R L32/DRAM rows are not both after the low-clock hold under their start lower bounds; this audit cannot claim a condition-specific AICLK upper bound.

#### `2026-10-09-p150a-newton-schulz-issue88-fp32-r-device1-retake.json`

`run_start` = `2026-10-09T16:10:14.048367+00:00`, `run_end` = `2026-10-09T16:10:33.458213+00:00`. Rows are consumed in persisted block order. For row *i*, the start lower bound is `run_start` plus the sum of measured `seconds_per_launch_samples` from earlier rows; no p50 or synthetic duration is used.

| row order/name | status | sample count | sample length sum (s) | cumulative start lower bound (s) | lower-bound timestamp |
|---:|---|---:|---:|---:|---|
| 0: bf16-r-L16 | ok | 1000 | 0.396095106 s | 0.000000000 s | 2026-10-09T16:10:14.048367+00:00 |
| 1: fp32-r-L16 | ok | 1000 | 0.403749030 s | 0.396095106 s | 2026-10-09T16:10:14.444462+00:00 |
| 2: bf16-r-L32 | ok | 1000 | 0.656246635 s | 0.799844136 s | 2026-10-09T16:10:14.848211+00:00 |
| 3: bf16-r-L32-r-dram | ok | 1000 | 0.663582893 s | 1.456090771 s | 2026-10-09T16:10:15.504458+00:00 |
| 4: fp32-r-L32-r-dram | ok | 1000 | 0.697182351 s | 2.119673664 s | 2026-10-09T16:10:16.168041+00:00 |
| 5: fp32-r-L32 | preflight_rejected | — | — | — | — |

Positive-duration trace holds below 1343 MHz are listed below; this includes the 800 MHz hold and excludes zero-duration rows at the interval end. The end offset is measured from `run_start`.

| AICLK (MHz) | hold start offset | hold end offset | hold end timestamp |
|---:|---:|---:|---|
| 800 | 0.000000000 s | 2.014112000 s | 2026-10-09T16:10:16.062479+00:00 |
| 1331 | 11.306589000 s | 13.669579000 s | 2026-10-09T16:10:27.717946+00:00 |

For L16, both BF16-R and FP32-R can overlap the initial low-clock hold under these lower bounds, so the direction of the clock confounding cannot be determined.
The BF16-R and FP32-R L32/DRAM rows are not both after the low-clock hold under their start lower bounds; this audit cannot claim a condition-specific AICLK upper bound.

#### `2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json`

`run_start` = `2026-10-10T03:56:53.785058+00:00`, `run_end` = `2026-10-10T03:57:13.057222+00:00`. Rows are consumed in persisted block order. For row *i*, the start lower bound is `run_start` plus the sum of measured `seconds_per_launch_samples` from earlier rows; no p50 or synthetic duration is used.

| row order/name | status | sample count | sample length sum (s) | cumulative start lower bound (s) | lower-bound timestamp |
|---:|---|---:|---:|---:|---|
| 0: bf16-r-L16 | ok | 1000 | 0.395370417 s | 0.000000000 s | 2026-10-10T03:56:53.785058+00:00 |
| 1: fp32-r-L16 | ok | 1000 | 0.402115224 s | 0.395370417 s | 2026-10-10T03:56:54.180428+00:00 |
| 2: bf16-r-L32 | ok | 1000 | 0.661143562 s | 0.797485641 s | 2026-10-10T03:56:54.582544+00:00 |
| 3: bf16-r-L32-r-dram | ok | 1000 | 0.662264669 s | 1.458629203 s | 2026-10-10T03:56:55.243687+00:00 |
| 4: fp32-r-L32-r-dram | ok | 1000 | 0.695446432 s | 2.120893872 s | 2026-10-10T03:56:55.905952+00:00 |
| 5: fp32-r-L32 | preflight_rejected | — | — | — | — |

Positive-duration trace holds below 1343 MHz are listed below; this includes the 800 MHz hold and excludes zero-duration rows at the interval end. The end offset is measured from `run_start`.

| AICLK (MHz) | hold start offset | hold end offset | hold end timestamp |
|---:|---:|---:|---|
| 800 | 0.000000000 s | 2.101907000 s | 2026-10-10T03:56:55.886965+00:00 |

For L16, both BF16-R and FP32-R can overlap the initial low-clock hold under these lower bounds, so the direction of the clock confounding cannot be determined.
The BF16-R and FP32-R L32/DRAM rows are not both after the low-clock hold under their start lower bounds; this audit cannot claim a condition-specific AICLK upper bound.

### Corrected conclusion boundary

The three records retain corrected quality metrics and measured throughput/L1 cost context. Their host timings remain valid observations, but the prior conclusion that a complete-run AICLK width could not explain the throughput differences is withdrawn: the row-order lower bounds above do not establish synchronized condition-specific clock values.

The final summary's retained P50 TFLOPS comparisons are descriptive only and were not used to construct the timing bounds. Recorded comparisons: L16/L1: -1.694%; L32/DRAM: -4.681%.

Accordingly, this audit supports reporting the corrected quality metrics and the observed throughput/L1 trade-off only within the evidence retained by the records. It does not make a stronger FP32-R/BF16-R causal or clock-independent conclusion.

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

These values reproduce the prior audit's exact #118 comparisons to three decimal MHz without rewriting any measurement leaf. The control remains a 1,350 MHz reference rather than a computed weighted load interval. The corrected extrema are not weighted averages: for example, the DRAM interval now reports the positive-duration 800 MHz pre-start hold in its min/max while its weighted value remains 1,349.842 MHz.
