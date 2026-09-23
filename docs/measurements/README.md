# Measurements

Raw results kept as data, each with the environment that produced it.

A throughput figure is only evidence if the machine behind it can be named
again later. design.md §2 records that a firmware update once changed the
core count, and the first measurements on this project ran on a different
board from the target — so every result file here carries, in its own
`environment` block, the board type and serial, firmware bundle, kernel
driver version, the toolchain image by digest, and the revision of the
harness that computed the numbers. For accelerator-backed records,
`environment.tt_env_active_release` is host-side context captured before the
pinned container starts; it is not the release identity of the container.
The immutable image digest in `environment.image` is authoritative for the
container and toolchain identity. A host-side measurement — one the
reference implementation takes with no board involved — carries the host,
its platform and CPU, the Python / NumPy / SciPy versions and the harness
revision instead, and says `"board": null`. A companion trace is
plain data with no such block: it inherits its provenance from the result
file sharing its filename stem, and is meaningless apart from it.

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

Results are not rewritten. A measurement that turns out to be wrong, or is
retaken on a corrected harness, is superseded by a later record that says
so — the same invariant ADR-0004 sets for decisions. The reasoning is in
ADR-0005, which also fixes what a result must carry and how a figure quoted
elsewhere refers back to it.
