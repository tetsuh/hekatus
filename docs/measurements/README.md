# Measurements

Raw results kept as data, each with the environment that produced it.

A throughput figure is only evidence if the machine behind it can be named
again later. design.md §2 records that a firmware update once changed the
core count, and the first measurements on this project ran on a different
board from the target — so every result file here carries, in its own
`environment` block, the board type and serial, firmware bundle, kernel
driver version, and the toolchain image by digest. A companion trace is
plain data with no such block: it inherits its provenance from the result
file sharing its filename stem, and is meaningless apart from it.

Naming: `YYYY-MM-DD-<board>-<what-was-measured>.json`, with any companion
trace beside it under the same stem.

| File | What it is |
|---|---|
| `2026-08-11-p150a-effective-efficiency.json` | The B2 measurement: 17 shapes x 2 dtypes x DRAM/L1 on one p150a, against the 332 TFLOPS BF16 peak. Summarized in docs/budget.md |
| `2026-08-11-p150a-effective-efficiency-power.csv` | Board power, clock, and temperature sampled through that run |

Results are not rewritten. A measurement that turns out to be wrong, or is
retaken on a corrected harness, is superseded by a later record that says
so — the same invariant ADR-0004 sets for decisions. The reasoning is in
ADR-0005, which also fixes what a result must carry and how a figure quoted
elsewhere refers back to it.
