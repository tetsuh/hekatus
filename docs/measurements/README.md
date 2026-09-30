# Measurements

Raw results kept as data, each with the reproducibility information that
produced it.

A throughput figure is only evidence if the environment behind it can be
reproduced later. design.md §2 records that a firmware update once changed
the core count, and the first measurements on this project ran on a different
board from the target — so every result file here carries, in its own
`environment` block, the board type and serial, firmware bundle, kernel
driver version, the toolchain image by digest, and the revision of the
harness that computed the numbers. For accelerator-backed records,
`environment.tt_env_active_release` is host-side context captured before the
pinned container starts; it is not the release identity of the container.
The immutable image digest in `environment.image` is authoritative for the
container and toolchain identity. A host-side measurement — one the
reference implementation takes with no board involved — carries the
platform, machine architecture, CPU, Python / NumPy / SciPy versions and the
harness revision instead, and says `"board": null`; it does not require a
host identity. A companion trace is plain data with no such block: it
inherits its provenance from the result file sharing its filename stem, and
is meaningless apart from it.

Naming: `YYYY-MM-DD-<board>-<what-was-measured>.json`, with any companion
trace beside it under the same stem.

| File | What it is |
|---|---|
| `2026-08-14-p150a-effective-efficiency.json` | The B2 measurement: 17 shapes x 2 dtypes x DRAM/L1 on one p150a, against the 332 TFLOPS BF16 peak. Summarized in docs/budget.md |
| `2026-08-14-p150a-effective-efficiency-power.csv` | Board power, clock, and temperature sampled through that run |
| `2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0.json` | Issue #65 full stock matmul catalogue sweep: 284 rows (190 successful, 94 failed), image digest `sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621` in its environment block. The four failed `batched_dram_sharded` rows for the batch-1024 L16/L32 shapes are superseded by the 2026-09-21 record below, and the two unbatched `dram_sharded` rows for beamspace B=16, 256 channels, 4096 pixels are superseded by the 2026-09-23 record; the other 278 rows remain authoritative. |
| `2026-09-20-p150a-stock-matmul-config-sweep-ttnn-0.75.0-power.csv` | Power, clock, and temperature trace for the 0.75.0 full sweep (183 samples); its provenance is the matching result record above |
| `2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0.json` | Issue #65 targeted four-row rerun with the corrected eight-worker mapping. It supersedes only the batch-1024 L16/L32 BF16/FP32 `batched_dram_sharded_g7x1_k1_m1_n1_s1x1` failures in the 2026-09-20 full sweep: the BF16 replacements succeeded, while both FP32 replacements reached program compilation and failed because static circular buffers clashed with L1 tensor buffers. The file's `supersedes` block maps every prior row to its replacement. |
| `2026-09-21-p150a-stock-matmul-batched-dram-superseding-ttnn-0.75.0-power.csv` | Power, clock, and temperature trace for the targeted superseding rerun (2 samples); its provenance is the matching result record above |
| `2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0.json` | Issue #65 targeted two-row rerun for the beamspace B=16, 256-channel, 4096-pixel unbatched `dram_sharded` rows after the corrected eight-worker mapping. It supersedes the BF16 and FP32 `dram_sharded_g4x1_k1_m1_n32_s1x4` rows in the 2026-09-20 full sweep with successful `dram_sharded_g8x1_k1_m1_n16_s1x4` rows; the file's `supersedes` block maps both prior rows to their replacements. |
| `2026-09-23-p150a-stock-matmul-unbatched-dram-superseding-ttnn-0.75.0-power.csv` | Power, clock, and temperature trace for the targeted two-row superseding rerun (3 samples); its provenance is the matching result record above |
| `2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1.json` | Issue #65 toolchain-separated default-only comparison: 68 rows (59 successful, 9 failed), image digest `sha256:ead7b800bdb6bebb9425c377222314447c5b2052f6e8b1e3c9caa1818cb7d8c4` in its environment block |
| `2026-09-20-p150a-stock-matmul-default-ttnn-0.70.1-power.csv` | Power, clock, and temperature trace for the 0.70.1 default-only comparison (88 samples); its provenance is the matching result record above |
| `2026-08-23-host-iq-path-vs-golden.json` | The #6 measurement: the IQ path (front end + IQ DAS) against the RF golden on the development host — per-stage errors at L0 checkpoints 1 and 2, image difference, and the axial PSF at −6 / −20 / −40 dB at D=8 and D=4, on the provisional `linear-5mhz` profile. Written by `python -m enodia.spec.beamform.decimation_sweep --record`; summarized in design.md §5 and §15 |
| `2026-09-27-p150a-newton-schulz-l32-b8192.json` | Issue #63: L=32, batch 8192 fixed-eight-iteration Newton-Schulz throughput; same-run stock `ttnn.matmul` and `custom_newton_schulz` rows, with correctness evidence and per-launch P50/P99/P99.9. The companion power trace shares the stem. |
| `2026-09-27-p150a-newton-schulz-l32-b8192-stock-catalogue-merge.json` | Supersedes the preceding Issue #63 record after the stock catalogue merge; repeats the same device-0 stock/custom comparison with the merged runner's default-only mode. |
| `2026-09-27-p150a-newton-schulz-l32-b8192-catalog-1000.json` | Supersedes the preceding Issue #63 throughput record; measures the complete stock configuration catalogue and custom row in one run with 1,000 launches per row for P99.9. |
| `2026-09-27-p150a-newton-schulz-l32-b8192-order2-catalog-1000.json` | Order of work 2: supersedes the preceding catalogue record and measures the two-DEST complex-product kernel under the same 1,000-launch stock/custom comparison. |
| `2026-09-28-p150a-newton-schulz-l32-b8192-fidelity-catalog-1000.json` | Fidelity sweep: supersedes the Order 2 record, records failed LoFi/HiFi2 gates without throughput rows, and measures valid HiFi3/HiFi4 plus the full stock catalogue at 1,000 launches. |
| `2026-09-28-p150a-newton-schulz-profile-breakdown-blocked.json` | Blocked profiling attempt: the pinned release lacks a Tracy-enabled device-profiler build, so no cycle table or bottleneck claim is made. |
| `2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter-blocked.json` | Supersedes the Tracy-blocked profiling attempt: the cycle-counter run completed, but the first host transport used tiled storage for raw uint32 pages; the corrected row-major path is committed without a third hardware run, so no cycle table or bottleneck claim is made. |
| `2026-09-29-p150a-newton-schulz-profile-breakdown-cycle-counter.json` | Supersedes the cycle-counter transport failure after the known-value row-major host test: records the valid HiFi3 core-0 RISC cycle table, does not determine the bottleneck, and motivates the next measurement separating steady all-work, warmup, reader, writer, and compute scopes. |
| `2026-09-30-p150a-newton-schulz-profile-breakdown-cycle-counter-steady.json` | Supersedes the prior cycle-counter table with all-tile/all-iteration compute and separated reader/writer intervals; residuals are explicit and consistency passes, but no unique bottleneck is claimed because RISC windows overlap and compute still has unclassified overhead. |
| `2026-09-30-p150a-newton-schulz-l32-b8192-optimization-catalog-1000.json` | Supersedes the fidelity catalogue; same-device HiFi3 comparison of stock, baseline, `fuse_s`, and `fuse_s` plus `batch_reads`, with 1,000 launches per row and batch-4/batch-8192 correctness evidence. |
| `2026-10-01-p150a-newton-schulz-l32-b8192-unpack-diagnostic-catalog-1000.json` | Supersedes the optimization catalogue; diagnostic same-device HiFi3 `fuse_s` comparison of BF16-state and FP32-state X, records the BF16 correctness-gate failure, L1/DRAM output placement, and does not claim a causal unpack bottleneck. |

Results are not rewritten. A measurement that turns out to be wrong, or is
retaken on a corrected harness, is superseded by a later record that says
so — the same invariant ADR-0004 sets for decisions. The reasoning is in
ADR-0005, which also fixes what a result must carry and how a figure quoted
elsewhere refers back to it.

The host name is outside ADR-0005's contract. Removing it from a landed
record does not violate ADR-0005's prohibition on rewriting because the
measurement values and reproducibility provenance are unchanged. The string
remains in git history; removing it from history would rewrite `main`, which
is not done.
