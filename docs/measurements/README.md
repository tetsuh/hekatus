# Measurements

Raw results kept as data, each with the environment that produced it.

A throughput figure is only evidence if the machine behind it can be named
again later. design.md §2 records that a firmware update once changed the
core count, and the first measurements on this project ran on a different
board from the target — so every result file here carries, in its own
`environment` block, the board type and serial, firmware bundle, kernel
driver version, the toolchain image by digest, and the revision of the
harness that computed the numbers. A host-side measurement — one the
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
| `2026-08-23-host-iq-path-vs-golden.json` | The #6 measurement: the IQ path (front end + IQ DAS) against the RF golden on the development host — per-stage errors at L0 checkpoints 1 and 2, image difference, and the axial PSF at −6 / −20 / −40 dB at D=8 and D=4, on the provisional `linear-5mhz` profile. Written by `python -m enodia.spec.beamform.decimation_sweep --record`; summarized in design.md §5 and §15 |

Results are not rewritten. A measurement that turns out to be wrong, or is
retaken on a corrected harness, is superseded by a later record that says
so — the same invariant ADR-0004 sets for decisions. The reasoning is in
ADR-0005, which also fixes what a result must carry and how a figure quoted
elsewhere refers back to it.
