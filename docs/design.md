# design.md — hekatus

Design decisions and their rationale for an ultrasound imaging stack PoC on
the Tenstorrent Blackhole p150a.

This document records **why things are the way they are**. Much of it cannot
be recovered from the code alone. Read it before implementing.

---

## 0. Module structure (the three-way crossing) and role/placement separation

**hekatus is the umbrella name for the compute system of a medical ultrasound
machine** — everything after received channel data reaches Ethernet. The
transmit/receive sequence is owned by control software on the host PC, and
hekatus receives its description through an API (§19). The UI, transmit
control, and infrastructure (sitos / sitometron) are outside.

### Separating role from placement

hekatus / enodia / diaplous / lampas are **names of roles** — hardware-neutral
concepts. Placement onto hardware is a decision in a different layer and may
change (enodia-tt → enodia-???, and so on).

| Role | Content | Required execution properties | Current placement |
|---|---|---|---|
| **enodia** | receive, front end, delays, transmit compounding, R formation, beamformer family, envelope | absorb 20 GB/s, µs-class determinism, fixed execution time, resident | enodia-tt (p150a) |
| **diaplous** | mode-specific processing before scan conversion + display system | stable processing within the frame period, CPU-portable functionality | Holoscan graph |
| **lampas** | companion inference; non-destructive tap observation | never blocks the processing path | lampas-tt (PoC) |

**The Track A reference implementation is the hardware-neutral specification
of enodia.** The acceptance criterion for any future port is numerical
equivalence with the reference implementation (L0), not reuse of Metalium
code.

### What hardware independence means for diaplous

diaplous compute cores are implemented as Array-API-conformant pure functions
(NumPy-compatible in/out); Holoscan operators are thin wrappers. Compute
cores import no specific backend. CI includes a test that runs every compute
core on NumPy (CPU). **The claim is functional portability, not performance
portability** (elastography lateral tracking, for instance, will very likely
not reach real-time performance on a CPU; that is solved by mixed per-process
placement).

Implication for the bill of materials: how much of diaplous a CPU can handle
determines where a "TT + general-purpose CPU (GPU-less)" configuration is
viable, which maps directly onto the price tiers of the machine.

### Internal structure of diaplous (two systems)

```text
enodia ──(stream)──→ [B-mode preprocessing] ──→ result buffer ─┐
       ──(stream)──→ [color flow]           ──→ result buffer ─┼─(latest read)→ [display]
       ──(stream)──→ [PW/CW Doppler]        ──→ result buffer ─┤   60/120 fps
       ──(stream)──→ [elastography]         ──→ result buffer ─┘
```

- **Data-driven processing group**: processes all data (lossless). Stateful
  (inter-frame filters — persistence, spatial compounding — live here).
  Needs slow-time series completeness (PW spectra, color ensembles,
  elastography correlation). B-mode pre-scan-conversion processing is also
  data-driven
- **Display system**: a self-running loop on the display clock (60/120 fps).
  Stateless. Reads the latest from each result buffer; redraws the same data
  if nothing new arrived. Passes acquisition timestamps through, and marks
  the image stale on screen when nothing updates for a set time (a safety
  requirement). At 120 fps the display-latency quantization error shrinks to
  an 8.3 ms bound

### Process separation is the standard form

The mode-specific processing systems (B-mode preprocessing / color / PW /
elastography), the display system, and lampas are **each an independent
process**. sitos exists precisely for this (an infrastructure that removes
the pain of sharing parameters across process boundaries).

- A process is the unit of failure, scheduling, and lifecycle. sitometron
  supervises
- Mode switching = pre-launch the next mode's process and hold it ready,
  then flip the stream routing
- Probe-profile switchover happens as a sitos session snapshot (every
  process sees the same generation)
- Initial granularity: about seven processes — enodia-tt / dp-bmode /
  dp-color / dp-spectral / dp-elasto / dp-display / lampas-tt. Merge or
  split after measuring

**Missing piece (TBD)**: memory usage differs per mode state, and it is
unclear whether static residency across all conditions is feasible. The
candidate solution is "static layouts per mode set + generation-switched
memory pool" (§20). hekatus itself prepares only two things now: each
process declares its memory needs (mode set × process → size expressions),
and buffers arrive as externally assigned partitions.

### One unified data plane (inter-process contract)

One mechanism: **a ring buffer in shared memory + sequence numbers**. Two
policies on top:

- **stream (processing path, lossless)**: enodia → each diaplous processing
  system. Losslessness is guaranteed not by blocking but by **sufficient
  depth + violation detection** (enodia never waits; overwrite violations
  are counted and warned — a violation already is a fault state).
  Doppler/elastography streams carry phase-preserving complex IQ,
  ROI-restrictable
- **tap (observation path, drop-tolerant)**: for lampas. Read-only views;
  what a late reader misses is lost. The adapter (conversion to model input)
  is the consumer's responsibility and budget

enodia → diaplous is a **push** (DMA) from TT to host memory. Completion is
detected by polling (end-of-frame sequence number). The DMA write-ordering
guarantee needs investigation (open-issues). Inside diaplous, result buffer →
display uses the same mechanism's latest-read. Details: `docs/dataplane.md`.

Naming origins: `docs/naming.md`.

---

## 1. Background and purpose

### What is being built

Implement the beamforming processing of a medical ultrasound machine on a
Tenstorrent p150a. Target conventional focused-beam B-mode, and add, in
stages:

- transmit compounding (retrospective transmit focusing)
- phase-screen aberration correction
- adaptive beamforming (CF / DMAS / SLSC / beamspace MV)
- later: color Doppler, real-time organ recognition running alongside

### Why Tenstorrent

**The industry has already moved to software-defined beamforming.**
GE's cSound is a software beamformer that stores channel data in large
memory before image formation. Philips moved from FPGAs to NVIDIA GPUs in
its third generation and reported a 70% improvement in 3D color volume
rates.

So the architecture is already proven, and **the comparison target is the
GPU, not the FPGA**. "We did on TT what a GPU can do" does not carry a PoC.
The differentiation lies in:

1. **Latency determinism** — a resident kernel runs autonomously on the
   card; the host OS/driver never touches the data path. GPUs have deep
   stacks and produce tail outliers. Win on P99.9 frame intervals.
2. **Spatial partitioning** — the 120 cores can be statically partitioned,
   so inference and beamforming coexist by spatial division rather than
   time slicing. Interference is structurally absent.
3. **Power and cost** — 300 W / $1,399.

### What the PoC claims (in the absence of real data)

Real machine data becomes available only after the PoC succeeds. Therefore:

- **Main axis: throughput, latency, compute headroom.** These do not depend
  on whether the input is real or simulated — same format, same volume,
  same performance characteristics.
- **Secondary: image quality.** If simulation shows gCNR improving, anyone
  who owns real data will reply "MV beating DAS has been known for thirty
  years," and that is the end of it.

**The most effective demonstration:**
feed the transmit sequence of a current machine unchanged, run everything
that existing machines gave up for lack of compute, and show that the frame
rate does not drop. **The headroom is the essence** — what to do with the
spare capacity can be decided once real data arrives.

---

## 2. Hardware assumptions

### p150a

| Item | Value |
|---|---|
| Tensix cores | 120 (FW v19.5.0 and later; 140 before) |
| L1 (per Tensix) | ~1.5 MB |
| GDDR6 | 32 GB / 512 GB/s |
| BF16 | 332 TFLOPS |
| Block FP8 | 664 TFLOPS |
| PCIe | 5.0 x16 |
| QSFP-DD | 4 ports × 800 Gbps |
| TDP | 300 W |
| Grid | 12 × 17 |
| RISC-V per Tensix | 5 (BRISC, NCRISC, TRISC0-2) |

**Caution: a firmware update has already changed the core count once
(140 → 120).** Code that depends on the grid layout must pin the firmware
version and keep a regression test.

### The machine measurements are taken on

The table above is the target specification. It is not automatically the
machine a given number came from, and the two have already differed: the
first measurements on this project ran on a p100a, which has no Ethernet at
all and one fewer memory channel. Every measured figure therefore carries
the board, firmware, driver, and toolchain image that produced it
(`docs/measurements/`), and a figure without that provenance is not evidence.

The development machine now carries **two p150a boards**, matching the
target: 120 Tensix and 32 GB each, Ethernet present, firmware 19.6.0.0 and
driver 2.8.0 pinned by the environment manifest. Two differences from the
table are worth recording:

- **PCIe negotiates at gen4**, not 5.0. The ingest path is Ethernet, so this
  does not touch the 20 GB/s of §3, but enodia → diaplous is a DMA push
  across this link, and gen4 x16 is roughly 32 GB/s.
- **The firmware reports two different power limits** — 150 W as `tdp_limit`
  and 300 W as the board limit. Under the compute measurement neither bound:
  the board peaked at 102 W at full clock. Which one is enforced remains
  unresolved, and will stay so until a workload approaches it.

Ethernet is present but no link was up as delivered, and the topology tool
does not support this generation — Blackhole trains its links from the
runtime rather than from a flashing step. Establishing an actual chip-to-chip
transfer is the first step of the receive-path work, not an assumption it can
start from.

### What limits this workload

Arithmetic is a few percent of theoretical peak. **The limits are GDDR6
bandwidth and L1 capacity.**

- If channel data fits in on-chip SRAM, processing completes with almost no
  DRAM traffic
- If it does not, everything goes through DRAM and even DAS collapses into
  bandwidth-bound behavior

Therefore **fitting on-chip is the paramount design concern**.

**The canonical dataflow is the group batch (§7)**: channel data for a group
of scanlines resides in chip SRAM (the union of the cores' L1s), and each
core reads the transmit data it needs over the NoC. The model "replicate one
transmit to every core, zero inter-core traffic" is the **degenerate form
without transmit compounding**, not the canonical one (once compounding is
on, one pixel references several transmits, and the replication model
exceeds L1).

---

## 3. Input specification

| Item | Value |
|---|---|
| ADC | 12 bit |
| TGC | applied on the AFE side |
| Sampling | 40 MHz |
| Channels | 128–256 |
| Data rate | 20.5 GB/s (256 ch × 40 MHz × 2 B) |
| Transport | FPGA → QSFP-DD → ERISC → Tensix L1 |

### Dynamic range after TGC

With TGC having compensated depth attenuation, what remains is tissue
reflectivity differences (40–50 dB) plus speckle variation (±10 dB) —
**50–60 dB total**. 12 bits = 72 dB suffices. Without TGC, 12 bits would
not remotely suffice.

### No bit compression

- 20 GB/s is 20% of one QSFP-DD port (100 GB/s). Bandwidth is not the
  problem
- Lossy compression conflicts with adaptive beamforming: MV exploits minute
  inter-channel phase differences, and compression noise contaminates the
  covariance estimate and degrades sidelobe suppression
- If ever needed, lossless only

### The Ethernet receive path

`erisc_datamover.cpp` in tt-metal has been reviewed. The ERISC delivers
received payloads over the NoC directly into worker Tensix cores' L1 and
signals with semaphores.

- Receive flow: `RECEIVER_WAITING_FOR_ETH` → `RECEIVER_SIGNALING_WORKER`
  → `RECEIVER_WAITING_FOR_WORKER`
- Semaphore-based credit control
- `num_buffers_per_channel` provides built-in multi-buffering; no
  hand-rolled double buffering needed to overlap receive and compute
- Each channel can have independent buffers and worker groups, so 256
  channels can be spread across several receive channels into different
  Tensix core groups

**FPGA side:** the EDM is designed for chip-to-chip links and assumes an EDM
peer. The FPGA must mimic the initial handshake, the `erisc_info` field
operations, and the ack handling. There is no spec document, but everything
derives from the source.

**Caution:** the reviewed code uses the handshake in the `deprecated`
namespace. Check at implementation time whether a newer fabric-based EDM is
the current recommendation.

**Concern:** the main loop calls `run_routing()` after `SWITCH_INTERVAL`
(4M idle iterations). With data always flowing it should never fire, but it
is a potential jitter source across frame gaps and mode switches. Measure.

---

## 4. Probes

Several probes share a fixed 40 MHz sampling clock; the ratios differ.

| Probe | f0 | Depth | Raw samples | Decimation | IQ/ch | 256 ch |
|---|---|---|---|---|---|---|
| Linear | 13 MHz | 3 cm | 1,560 | 2 | 3.1 KB | 0.80 MB |
| Linear | 7.5 MHz | 4 cm | 2,080 | 4 | 2.1 KB | 0.53 MB |
| Linear | 5 MHz | 6 cm | 3,120 | 8 | 1.6 KB | 0.40 MB |
| Convex | 3.5 MHz | 20 cm | 10,400 | 8 | 5.2 KB | ~0.5 MB @ 64–96 recv ch |
| Sector | 2.5 MHz | 20 cm | 10,400 | 16 | 2.6 KB | 0.67 MB |

**Depth × bandwidth (the time-bandwidth product) is nearly constant across
probes, so L1 usage falls in line naturally. Every configuration fits in
1.5 MB.**

### The 13 MHz caveat

40 / 13 is only 3.08×. At 80% fractional bandwidth the signal spans
7.8–18.2 MHz against a 20 MHz Nyquist — 1.8 MHz of margin.

That 80% is a **synthetic design envelope, not a profile**: no 13 MHz
`ProbeProfile` exists, and #10 creates it under the bandwidth definition
below. Until it does, every 13 MHz figure §5's sweep derives from the
envelope — the 5.2 MHz one-sided edge, the 47°, the kernel tables — is
labelled `synthetic-80pct-design-envelope` and claims no profile or physical
authority. The label belongs to that envelope alone: it does not apply to
`rf-oracle-frozen-0p7`, the separately frozen 13 MHz record of §15, nor to
the profile-specific reconciliation #10's profile will produce.

- Verify that the AFE anti-aliasing filter truly kills everything above
  20 MHz
- Aliasing would appear as phantom echoes at depth
- **The first suspect whenever an unexplained artifact appears**
- Decimation ratio is 2: **IQ conversion does not shrink the data**
  (reduction factor = D/2)

### Why 13 MHz may be the main battleground

- Breast, thyroid, superficial vessels. Breast has the largest sound-speed
  contrast — fat (1450 m/s) vs glandular tissue (1550 m/s) — so
  **aberration correction pays off most**
- Phase error is delay divided by wavelength, so higher frequency is more
  severe: a 40 ns delay error from 1 mm of fat is 72° at 5 MHz and 187° at
  13 MHz
- Linear probes deliver all-element data without μBF, so **element-level
  aberration correction is possible**
- Static organs avoid transmit compounding's weakness (motion)

### Probe profiles

Hold the following as one settings bundle; switching is a table swap only
(kernels are shared).

- geometry (element count, pitch, element width, curvature radius, lens focus)
- f0, bandwidth — the effective two-way bandwidth defined below, with its
  provenance
- decimation ratio, complex BPF coefficients
- expected depth, receive aperture size
- MLA count
- beamformer settings (subaperture, diagonal loading, beamspace dimension)
- compute budget (scanline count × frame rate)

A few hundred KB, so L1 transfer takes tens of µs. Keeping several profiles
resident in DRAM makes switching instantaneous.

**In the product, `sitos` (the distributed parameter store) distributes
these profiles.** The PoC phase does not depend on sitos (sitos itself is in
pre-development; keep it off the critical path). Parameters live in plain
structs/files, with **interfaces designed compatible with the sitos type
system (bool / int64 / double / string / bytes + LUT)**, to be swapped in at
productization.

The sitos session snapshot connects directly to the pipeline flush of §12.
The inconsistency where only some tables are new during a probe switch (the
accident of mixing transmits with different settings inside transmit
compounding) is prevented structurally by snapshot-unit switching.

### Bandwidth (normative — ADR-0008)

The profile owns the pulse bandwidth, and it means one thing:

- `bandwidth_frac` is the **full fractional bandwidth of the effective
  two-way pulse** the processing profile assumes, relative to `f0`. Its
  endpoints are the two points **half amplitude** below the spectral peak —
  `20·log10|A(f)/A(f0)| = −20·log10 2 ≈ −6.0206 dB`, equivalently
  −6.0206 dB in power since P ∝ |A|². Not a 6 dB rounding, and not the
  −20 dB point
- full width `= bandwidth_frac · f0`; the **one-sided analysis edge** of the
  symmetric baseband model, which is what §5's band-edge figures take, is
  `bandwidth_edge_hz = bandwidth_frac · f0 / 2`
- it describes the effective two-way pulse, not a transmit-only spectrum and
  not an AFE anti-alias guarantee (the 13 MHz caveat above is a separate
  question)
- `bandwidth_source` is provenance: a non-empty string naming the
  manufacturer data or measurement the value comes from, or `None`, which
  means **provisional** — a working value with no physical backing. `None`
  never means measured, manufacturer-backed or validated. A consumer of a
  provisional value says so beside its result and reruns when the value or
  its provenance changes; "provisional" is a statement of evidence, not
  permission to pick another number

The bandwidth lives in the profile and nowhere else. The transmit
description of §19 references a profile by id and carries no bandwidth; the
frame header carries none (docs/dataplane.md). The simulator's pulse is
built to this definition and checked on its spectrum
(`tests/test_sim_format.py`). Reference: `enodia/spec/probe/__init__.py`.

**Implemented profile** — the repository authority for current PoC work:

| profile | f0 | full fraction | full width | one-sided edge | status |
|---|---:|---:|---:|---:|---|
| `linear-5mhz` | 5 MHz | 0.7 | 3.5 MHz | 1.75 MHz | **provisional** (`bandwidth_source=None`) |

0.7 is the value the simulator and the sweeps have used since MVP-1; no
manufacturer or measured pulse response stands behind it, and #46 made that
explicit rather than inventing one. A sourced value replaces it through a
reviewed profile update and reruns whatever consumed it — §5's 5 MHz tables,
the profile reconciliation of §15, and #6's comparison, which may consume
this named profile for reproducible provisional 5 MHz evidence and labels
its artifacts accordingly. The 13 MHz envelope above is not in this table
because it is not a profile.

---

## 5. Front end (IQ demodulation)

### Why it is mandatory

Raw RF does not fit in L1. At 13 MHz / 256 ch, one line is 4.0 MB. Going
through DRAM means reading all channels per pixel: 33 GB per 512×512 frame,
1 TB/s at 30 fps — twice GDDR6. It does not spin.

### Processing

```text
s(t) → [complex band-pass FIR] → [decimation ↓D] → IQ (int16 complex)
```

Mixing `z(t) = s(t)·e^(-j2πf0·t)` and the LPF **fuse into one complex BPF**.
Fewer stages, fewer intermediate buffers. Coefficients: 64 complex taps =
256 bytes, held per probe.

At `fs = 8·f0` (5 MHz) the reference collapses to an 8-period `cos(πn/4)`
pattern and the multiplies fold into constants — but **other probes are not
integer ratios**. Even 3.5 MHz cycles in 80 samples, a 320-byte table. Not a
problem in practice.

**The reference front end** (`enodia/spec/frontend/`, #6) is defined to the
sample, because L0 checkpoint 1 compares against it:

- prototype: windowed sinc, 64 real taps, Hann window, cutoff `cutoff_frac`
  × the decimated Nyquist frequency fs/(2D) with `cutoff_frac = 1`, DC gain
  **2** — the complex envelope at the real signal's own amplitude (unit gain
  keeps only the positive-frequency half and halves it; checkpoint 2 reads
  that as a 50 % error, which is how it was caught)
- fused taps `h_bp[k] = h_lp[k]·e^(+j2πf0·(k − (L−1)/2)/fs)`, the
  modulation **centred on the filter**; the output sample is rotated by
  `e^(−j2πf0·t)` at the RF time it stands for. Referenced to tap 0 instead,
  every IQ sample carries a constant 2πf0·(L−1)/2/fs — 22.5° at fs = 8f0 —
  that no image shows (checkpoint 2 found that too)
- alignment: IQ sample m stands for RF position **m·D + δ**, δ = L//2 −
  (L−1)/2 = ½ sample for L = 64; δ and D travel with the record
  (`IQEventRecord.rf_offset`, `.decimation`) and the delay stage reads the
  IQ at (τ_i·fs − δ)/D. Half an RF sample is 12.5 ns, 23° at 5 MHz
- output: int16 complex as two int16 planes (§14, dataplane T1), scaled by
  `iq_scale = 1`; a value that would not fit raises rather than clips

Every one of those is a sweepable parameter; what they cost is what the
golden comparison of §15 says, not what this paragraph argues.

### Cost

A 64-tap FIR × 2 (I/Q) at 30 fps ≈ 4 TFLOPS = 1.2% of one card. FIR lowers
to matmul — the shape Tensix likes.

### What it does to beamforming

The real value of IQ is that delay application changes form. What
beamforming needs from channel i is the complex envelope of the signal
delayed by τ. With the demodulation convention fixed above
(`z(t) = s(t)·e^(-j2πf0·t)`), the analytic signal gives

```text
env{s(t − τ)}(t) = z(t − τ) · e^(−j2πf0·τ)
```

and in decimated, sampled form, writing `d = τ·fs'` for the delay in
samples at the decimated rate `fs'`:

```text
x_i[n] ≈ interp4(z_dec, n − d) · e^(−j2πf0·τ)
```

**The sign convention is normative**: the phase factor is `e^(−j2πf0·τ)`
for the demodulation convention above. Flipping either convention without
flipping the other rotates every channel the wrong way and destroys
coherent summation — a failure that still produces a plausible-looking
image, so implementations state which convention they follow and the L0
checkpoints compare phase, not just magnitude (§15).

The expression decomposes into an **integer shift (a read-address offset)
+ a fractional part (the 4-tap interpolation defined below) + a
phase rotation (one complex multiply)**, making random access small and
local. With fixed geometry, the phase term is precomputable.

`enodia/spec/beamform/iq_das.py` is that expression as the reference (#6):
for a pixel at round-trip time t_p = 2z/c on a line, channel i's echo
arrives at τ_i, the delay is τ = t_p − τ_i, d = τ·fs', and x_i = interp4(z_i,
n − d)·e^(−j2πf0·τ) read at n = t_p·fs' — so the read position is τ_i·fs',
the golden's, at the decimated rate. The sign is **asserted at L0
checkpoint 2** (`tests/test_iq_das.py`): energy-weighted over the aperture,
the channel vectors agree in phase with the golden's analytic channel
samples to under 1.2° at D=8 and 0.22° at D=4; with the sign flipped they
are ~97° off — and the image made with the flipped sign still puts its
peaks in plausible places, which is the whole reason the test looks at the
vectors.

### Fractional-delay accuracy (important)

**Phase rotation alone is not enough.** As long as decimation goes toward
Nyquist, the band-edge phase error is tens of degrees whatever the carrier.
The band edge is the profile's one-sided edge of §4 — `bandwidth_frac · f0 /
2` — and the error is `360° · edge · (max fraction)`:

- 13 MHz (D=2, 50 ns spacing, max fraction 25 ns; band edge 5.2 MHz from
  the **synthetic 80% envelope** of §4, no profile exists) → 46.8°, quoted
  as 47°
- 5 MHz (D=8, 200 ns spacing, max fraction 100 ns; band edge 1.75 MHz from
  `linear-5mhz`, 0.7 × 5 MHz / 2, **provisional**) → **63°**; at D=4
  (100 ns spacing, max fraction 50 ns) → 31.5°

Before #46 this section said 54° for 5 MHz, from a 1.5 MHz edge that the
implemented profile never had; the profile's own pulse puts it at 63°.

Left alone, the axial PSF collapses and sidelobes rise.
**4-tap interpolation on IQ + phase rotation** is required; 2-tap linear is
not enough.

Keep the ability to measure how the point-scatterer axial PSF changes
between decimation ratios 8 and 4. **Measured** (#6,
`enodia/spec/beamform/decimation_sweep.py`, pinned by
`tests/test_golden_compare.py`; the record is
`docs/measurements/2026-08-23-host-iq-path-vs-golden.json`, ADR-0005), on the
provisional `linear-5mhz` profile (§4) and the demo's three point-scatterers, full widths of the axial profile through each scatterer:

| | golden | IQ, D=8 | IQ, D=4 |
|---|---|---|---|
| −6 dB width | 0.194 mm | 0.257–0.276 mm (+33–42 %) | 0.201–0.202 mm (+4 %) |
| −20 dB width | 0.355–0.356 mm | 0.420–0.426 mm (+18–20 %) | 0.360–0.361 mm (+1.4 %) |
| −40 dB width (§15: never from −6 dB alone) | 0.500–0.501 mm | 0.614 / 0.626 / 0.804 mm (+23 / +25 / +61 %) | 0.490–0.492 mm (−2 %) |
| peak level, Δ vs golden, per scatterer | — | −0.365 / −0.268 / 0.000 dB; the second peak moves one RF sample (19 µm) | +0.016 / +0.015 / −0.008 dB |
| post-delay channel vectors, checkpoint 2 | — | 19.9–21.1 % / ≤ 1.2° | 2.43 % / ≤ 0.22° |

The −40 dB column is where the heaviest decimation shows most: +61 % on
the deepest scatterer at D=8, whose line also carries the transmit-beam
arc of the MVP-1 simulator (#9), against −2 % at D=4 — the IQ envelope is
marginally narrower than the golden's Hilbert envelope there, which is a
difference between two envelope estimators, not a gain.

D=8 broadens the axial PSF by about 40 % at −6 dB on this pulse — the
width's 3σ reaches past the decimated Nyquist frequency (0.89 fs', above),
so the front end's low-pass takes part of the band (checkpoint 1: 7.9 %
against the ideal baseband, against 0.2 % at D=4) and the four taps then
carry the 14 % §5's sweep predicts; at D=4 the whole chain costs 2.4 % and
4 % of width. The figures are on a provisional bandwidth and are rerun
when it changes (§4); what they settle and what stays open is in §17.

### The interpolation kernel (normative, L0-relevant)

`interp4` is **Lagrange cubic**: the four-point Lagrange basis on the nodes
{−1, 0, +1, +2} around the target. Writing the position in samples as
`t = n − d`, with the record's first sample at `t = 0`, `m = ⌊t⌋` and
`μ = t − m ∈ [0, 1)`, the taps multiply `z[m−1], z[m], z[m+1], z[m+2]` with

```text
h₋₁ = −μ(μ−1)(μ−2)/6
h₀  = (μ+1)(μ−1)(μ−2)/2
h₊₁ = −(μ+1)μ(μ−2)/2
h₊₂ = (μ+1)μ(μ−1)/6
```

`μ = 0` reads `z[m]` exactly; the taps sum to one for every μ. **The record
is zero outside `[0, N)`**: a tap that falls before the first sample or past
the last contributes zero, so a target near an end is a partial sum and one
wholly outside is zero — not clamped, mirrored, or extrapolated, each of
which is defensible and each of which disagrees with the others. The same
real taps apply to I and Q. Reference: `enodia/spec/beamform/interp.py`.

**The sweep** (`enodia/spec/beamform/interp_sweep.py`, every figure below
pinned by `tests/test_interp_kernel.py`) measures two things, because one of
them is not enough to choose by. **Every case names the pulse it assumes**:
the 5 MHz cases are the `linear-5mhz` profile — edge 1.75 MHz, provisional,
no source — and the 13 MHz case is the synthetic 80% envelope of §4, edge
5.2 MHz; the sweep prints identity, status, source, spectral level and width
convention beside its tables, so a figure cannot be quoted without what it
assumed (#46).

*Worst-case error over the fraction, at the band edge* — the metric this
section already used for the no-interpolation case — phase / magnitude:

| kernel | 5 MHz D=8 (0.35 fs') | 13 MHz D=2 (0.26 fs') | 5 MHz D=4 (0.175 fs') |
|---|---|---|---|
| none (phase rotation only) | 63° | 47° | 31.5° |
| 2-tap linear | 13.10° / 54.6% | 4.64° / 31.5% | 1.30° / 14.7% |
| **Lagrange cubic (4)** | **8.25° / 36.6%** | **1.95° / 13.4%** | **0.29° / 3.1%** |
| Keys a=−1/2, Catmull-Rom | 13.10° / 36.6% | 4.64° / 13.4% | 1.30° / 3.1% |
| Keys a=−3/4 | 9.04° / 27.6% | 1.09° / 4.3% | 1.39° / 2.7% |
| Keys a=−1 | 5.40° / 18.6% | 2.15° / 4.8% | 3.93° / 8.5% |
| Hann-windowed sinc (4) | 16.55° / 45.9% | 6.45° / 22.7% | 1.95° / 9.1% |
| *least-squares 4-tap, bound* | *4.09° / 18.4%* | *0.73° / 4.7%* | *0.09° / 0.9%* |

**Read alone, this table does not choose Lagrange.** Keys at a = −1 beats it
at the 5 MHz D=8 edge on both axes — 5.40° against 8.25°, 18.6% against
36.6%. It does so by pre-emphasizing high frequencies, which buys the edge
and pays for it everywhere else — and a metric evaluated at one frequency
cannot see the bill. So the second measurement is the error over the whole
pulse:

*RMS error over a Gaussian pulse whose amplitude is N dB down at that band
edge, averaged over the fraction* — the error energy the delayed channel
signal actually carries. The first column of each case is **the case's own
pulse model**: the edge is its half-amplitude point by definition (§4), so
N = 20·log10 2 = 6.02 dB exactly; −20 and −40 dB are narrower pulses, kept
as a sensitivity sweep because the 5 MHz width is provisional and the
13 MHz one synthetic:

| kernel | 5 MHz D=8: model / −20 / −40 dB | 13 MHz D=2 | 5 MHz D=4 |
|---|---|---|---|
| 2-tap linear | 19.98 / 7.82 / 4.02 % | 13.38 / 4.42 / 2.25 % | 6.55 / 2.04 / 1.03 % |
| **Lagrange cubic** | **14.00 / 3.27 / 0.97 %** | **7.88 / 1.16 / 0.32 %** | **2.39 / 0.27 / 0.07 %** |
| Keys a=−1/2 | 14.03 / 3.38 / 1.06 % | 7.97 / 1.26 / 0.38 % | 2.50 / 0.32 / 0.10 % |
| Keys a=−3/4 | 12.02 / 2.59 / 1.54 % | 6.35 / 1.60 / 1.26 % | 2.06 / 1.21 / 0.89 % |
| Keys a=−1 | 11.44 / 4.55 / 3.58 % | 6.85 / 3.71 / 2.77 % | 4.25 / 2.64 / 1.85 % |
| Hann-windowed sinc (4) | 16.36 / 5.20 / 2.29 % | 10.13 / 2.57 / 1.16 % | 4.17 / 1.04 / 0.49 % |
| *least-squares, bound* | *10.87 / 3.15 / 2.80 %* | *6.22 / 0.92 / 0.83 %* | *1.81 / 0.18 / 0.17 %* |

Before #46 the 5 MHz columns were computed at a 1.5 MHz edge (0.30 and
0.15 fs') and the model column at a 6 dB rounding; Lagrange read 10.82% at
D=8 and 1.40% at D=4, and 7.91% at 13 MHz. The 13 MHz edge is unchanged,
so only its model column moved, from the exact level.

The integration stops at the decimated Nyquist frequency: the record cannot
represent anything above 0.5 cycles/sample, so scoring the kernel there
measures a signal that is not in it. Worth noting that the profile's own
pulse reaches past that limit at D=8 — its 3σ is 0.89 — which says that
width and the heaviest decimation are in tension: what D=8 costs is the
axial-PSF question §17 keeps open.

**Why Lagrange, stated as what the evidence supports.** Among the six closed
forms it is **first in five of these nine cells, second in two, and third in
two**. Where it is not first, the leader is a Keys kernel with a < −1/2, by
16% to 27%: on each case's own pulse model (Keys a=−1 at 5 MHz D=8 by 22%;
a=−3/4 at 13 MHz by 24% and at 5 MHz D=4 by 16%) and in the −20 dB column at
5 MHz D=8 (a=−3/4 by 27%). It is not best everywhere, and the choice is a
trade rather than a ranking.

*The trade.* Those same two kernels are **1.6 to 27 times worse** than
Lagrange once the pulse narrows: at −40 dB, Keys a=−1 is 3.7×, 8.7× and 27×
worse in the three cases, and a=−3/4 is 1.6×, 3.9× and 13×. So the choice
buys a factor of several in the narrow-pulse corner at the cost of about a
fifth to a quarter on the widest pulse.

*It is a trade the document cannot yet settle by measurement, and that is why
robustness wins.* The level the edge sits at is no longer in question —
issue #46 fixed it at half amplitude, so the first column is the model — but
the 5 MHz width behind it is provisional and the 13 MHz one synthetic (§4),
so the narrower columns still describe probes that may turn out to be the
real ones; and the decimation ratio is open (§17). Lagrange's error decays
faster as the pulse narrows than the other evaluated closed forms — being
exact for polynomials to degree 3 it is maximally flat at DC. At −40 dB, a
factor of 1.5 separates the two third-order accurate kernels from every other
candidate; the narrowest margin is Keys a=−3/4 at 5 MHz D=8, 1.59×, and
elsewhere it is at least 3.6×. The kernels that beat it on a wide pulse have
a larger measured narrow-pulse error in this sweep, so the choice remains a
robustness trade rather than a universal ranking.

**So the choice is robustness, not dominance, and it is provisional.** What
settles it is the axial-PSF measurement this section already asks for,
against the point-scatterer phantom, at D=8 and D=4. Until then a port runs
Lagrange, because L0 needs one kernel rather than the best one.

**What the sweep also says**: at D=8, four taps still leave 37% band-edge
magnitude error for the worst fraction, against 3% at D=4 — and the pulse-
weighted error improves by a rolloff-dependent factor: about 5.9× on the
model, 12.3× at −20 dB, and 14.0× at −40 dB. Whether that matters
is the same PSF question, which is why "decimation ratio and
interpolation tap count" stays in §17 as a parameter decided by measurement.
The kernel is fixed; how far to decimate under it is not.

**L0.** The kernel is part of the equivalence contract, beside the
phase-sign convention above: a port runs this kernel, and L0 compares like
with like at checkpoint 2 (§15). A difference between kernels is never
absorbed by a wider tolerance — that would spend the threshold on the thing
it exists to catch. The reference takes the kernel as a named argument with
exactly this one value defined; widening that set is a change to the
contract (ADR-0007), not a new string.

---

## 6. Geometry and the data model

### Coordinate system

**Polar coordinates (scanline × depth) are fundamental; scan conversion is
the final stage, host side.**

For a linear array on a rectilinear grid, the delay is a two-variable
function `τ(x_p − x_c, z)` and one table is shared by all channels
(translation invariance). Convex/sector probes break this — elements on an
arc, fan-shaped scanlines — but **in polar coordinates the translation
invariance returns**.

Scan conversion stays host-side because display resolution and zoom change
under operator control; on the card it would multiply reconfigurations.

### Data model

```text
input:  [channel (≤256) × depth × transmit event × slow time]
output: [depth × scanline]
bridge: the contribution map (transmit event → scanline, weighted, sparse)
```

**The transmit-event axis and the slow-time axis are different things.**
Do not conflate them.
- transmit-event axis: MLA and transmit compounding
- slow-time axis: color-Doppler ensembles. Collapses to length 1 in B-mode

### The contribution map

MLA, transmit compounding, and the conventional case are **the same
structure with different map contents**:

| Configuration | Contribution map |
|---|---|
| Conventional (1 transmit, 1 line) | scanline k ← transmit k only, weight 1 |
| 4 MLA | scanlines 4k..4k+3 ← transmit k, weight 1 |
| 4 MLA + compounding | scanline j ← transmits j/4 ± m, weights from geometry and apodization |

The map also carries **transmit-type matching conditions** (so B-mode and
color-Doppler interleaves never mix transmit kinds).

**Note: channel dropout is not handled in the contribution map.** The map's
axis is the transmit-event dimension; dropout lives in the receive-channel
dimension. Dropout is held as a **per-channel receive apodization mask**,
and in MV additionally **the affected rows/columns are excised from R**
(zeros from dead channels distort the covariance estimate).

---

## 7. Transmit compounding (Stage 2)

### The problem

Focused-beam B-mode focuses at one depth per transmit. At 13 MHz (3 cm
depth, 1.5 cm focus, F=2), the transmit beam is 0.12 mm wide at focus,
0.47 mm at 0.5 cm, 0.35 mm at 3.0 cm.

**Receive-side adaptive processing can only work inside what the transmit
beam illuminated.** If the transmit beam is 0.47 mm, no amount of receive
MV makes the total PSF narrower. That is the truth behind "we added MV and
nothing improved away from focus."

### Why Stage 2 (retrospective transmit focusing)

| Stage | Content | Verdict |
|---|---|---|
| 0 | multi-zone transmit focus | standard on existing machines; frame rate drops to 1/3–1/4 |
| 1 | receive dynamic focus | every machine already does it |
| **2** | **transmit compounding** | **adopted** |
| 2.5 | sparse synthetic aperture / coded transmits | needs transmit-sequence changes |
| 3 | synthetic transmit aperture (STA) | transmit SNR of a single element; deep field collapses |

Reasons for Stage 2:

1. **No transmit-sequence change.** Existing machines, probes, AFE stay as
   they are. What changes is receive-side processing — exactly the part
   Tenstorrent owns
2. **No frame-rate sacrifice.** It reuses scanline data already acquired
3. **Synergy with receive MV.** More independent samples per pixel, so
   subaperture averaging can shrink = effective aperture grows
4. **Compute pays off directly.** The most direct demonstration that
   "abundant compute buys image quality"

### The relationship with MLA (important)

MLA (one transmit → several receive lines) and transmit compounding
(several transmits → one pixel) are **two uses of the same fact: the
transmit beam has finite width**.

And **MLA's block artifacts are solved in principle by transmit
compounding.** MLA receive lines farther from the transmit-beam center are
illuminated more weakly, producing striped artifacts and line warping.
Compounding averages the per-transmit illumination non-uniformity away.

| | MLA alone | MLA + compounding |
|---|---|---|
| Frame rate | 2–4× | 2–4× (kept) |
| Block artifacts | present | resolved |
| Transmit focus | one depth | all depths |
| Compute | 1× | 4–17× |

**One technique, three gains — without touching the transmit sequence.**

### The contribution map as implemented (#53)

Both uses run through one structure,
`enodia/spec/sequence/contribution.py`: per output line, a **fixed number
of slots** (`cap`), each naming a transmit event and a weight. A frame-edge
line with fewer real contributions is padded with inert slots — weight
zero, pointing at a real event — so work per line is constant, which is
where the no-variable-loops absolute rule becomes an assertable property
of data. Weights are **renormalized at derivation** — each line divided by the sum
of the weights actually applied to it (ADR-0010), with a floor below which
the line is refused rather than amplified —
so the beamformer stays a plain weighted sum and edge lines are not darker
for their position. Both DAS paths — RF golden and IQ — read the map; the
identity map reproduces the pre-map images bit for bit.

**MLA line placement (ADR-0010):** the receive lines of one transmit
subdivide the **transmit line** pitch evenly, symmetric about the transmit
axis — `x_k + pitch·(2j−(mla−1))/(2·mla)` — where the pitch is the spacing
of the beam axes, not of the elements. The two coincide only for the
conventional sequence that fires one transmit above each element, and a
sequence at another stride would otherwise get groups of the wrong width.
MLA 1 recovers the conventional geometry exactly and the placement translates with the
transmit (delay-table translation invariance on convex/sector probes). An
output line pitch decoupled from the transmit pitch was considered and
set aside for exactly that invariance.

**A map names the configuration it was derived from** (`config_id`, event
count, **parameter generation**) and both consumers compare that against
what the records' headers name before forming a frame. The generation is
carried separately because depth and focus are in-config parameters (§19):
the id holds still across a knob turn while every derivative behind it,
this map included, is invalidated and re-derived — so a map checked on the
id alone would survive exactly the change that invalidates it. Event indices are small
integers every configuration has, so a map derived elsewhere resolves
cleanly and would put the frame on another configuration's scanlines with
nothing raised — the accident the generation tag exists to prevent (§19).
A frame whose records name more than one configuration, or more than one
generation, is refused for the same reason.

**Every slot runs, inert ones included.** The reference implementation does
not skip zero-weight slots: doing so would give a frame-edge line fewer
delay-and-aperture evaluations than an interior one, which is the
variable-work shape the absolute rules forbid — and this implementation is
the specification a port is written against, so a shortcut taken here reads
as sanctioned.

What stays measured, not chosen here (§17): the compounding weight
function (needs the beam model, #9), window width, and truncation count.
The multi-contribution structure is exercised by a uniform-weight
synthetic map, which selects no production value.

### Compute cost

**Receive-beamforming work is "formed scanlines × contributing transmits
per scanline" and does not depend on the MLA count** (MLA reduces the
transmit count, not the formed-line count). For 13 MHz linear, 434
scanlines:

| Configuration | line-formations / frame | vs none |
|---|---|---|
| No compounding (any MLA) | 434 × 1 = 434 | 1.0 |
| Compounding ±4 (any MLA) | 434 × 9 = 3,906 | **9.0** |

**The 9× applies only to the delay-and-sum (DAS-like) part.** In the
compound-then-MV arrangement (§9), R formation and Newton-Schulz run once
after compounding, so the total cost barely moves (DAS was ~0% to begin
with).

(Note: an older revision of this document carried a table claiming an
"effective multiplier of 2.25 because transmit count drops." That was
wrong — fewer transmits do not reduce receive-side work.)

### Contribution extent

Half-width `w(z) = |z − z_focus| / (2F)`. At 13 MHz (F=2, 15 mm focus,
0.1 mm pitch): 3.75 mm at 3 cm depth = 37 scanlines.

**The contributing-transmit count is fixed at a cap; out-of-range weights
are 0.** Variable-length loops sit poorly with Tensix and break
execution-time determinism. Parameterize the truncation count and measure
where the PSF saturates.

### Memory structure (on TT)

Processing transmit k needs data from transmits k±m, so "load one transmit
into L1 and process" breaks down: 17 transmits are 13.6 MB and do not fit.

| Scheme | Content | Verdict |
|---|---|---|
| (a) scatter | on receiving a transmit, add partial sums into every scanline it contributes to | output buffer keeps the channel axis: 51 MB. Scanlines split across cores; each transmit broadcast to all |
| (b) gather | park channel data in DRAM and re-read | 17× bandwidth = 340 GB/s; collides with MV. **Rejected** |
| (c) group batch | complete scanline groups (e.g. 64 lines) in chip SRAM | boundary artifacts; keeps the pixel-parallel structure |

**The reference implementation is written as (c).** On the host it is a
plain loop, mathematically equal to (a) except at boundaries. If TT work
shows (a) is needed, migration is possible. Group size is a parameter;
boundary artifacts get evaluated.

### The weakness

**Motion.** Compounding sums transmits from different times; hearts and
blood flow decorrelate the phase. At 13 MHz on breast/thyroid (essentially
static) this is not a problem — if the main battleground is superficial,
the weakness is avoided.

### MLA count specification (fixed)

**The fixed specification is MLA {2, 4}; 8 is an experimental slot for
color flow.**

In practice, 8-MLA color flow is known to be unusable due to blocking
noise. The cause is two superimposed effects: (1) illumination
non-uniformity + warping (shared with B-mode), and (2) **a color-specific
velocity bias** — the phase gradient from transmit/receive misalignment is
translated by the Kasai autocorrelator into a false velocity offset,
appearing as line-shaped artifacts in the color image. It is a phase bias,
not a brightness issue, so gain correction cannot fix it.

Transmit compounding solves this in principle: apply the inter-multiline
delays that prevent phase cancellation before summation (= correct the
phase bias per transmit position, then add). Furthermore, freezing the
compounding weights and delays across the ensemble (the same principle as
§11) preserves slow-time phase. With 8 MLA the transmit beam is wide, so
more transmits contribute and the averaging works harder. Compute is set by
line-formation count, which does not depend on MLA count — 8 costs nothing
extra.

**Mind the prior art**: premium-tier products already cover high-MLA color
with transmit compounding. The PoC claim is therefore not "world's first"
but "technology so far reserved for premium machines, delivered on the
compute budget of one general-purpose accelerator, simultaneously with
MV and aberration correction, with the existing transmit chain untouched."
An IP-landscape review is a productization-gate item (open-issues).

Verification plan: quantify velocity bias for "8-MLA color with vs without
compounding" (scanline dependence of the estimated velocity against a known
constant-velocity field). An image in which this known failure mode
disappears makes the most convincing demonstration. For this, the Phase 1
phantom set includes a **flow phantom** (point clouds moving in slow time,
constant-velocity region).

---

## 8. Layer 1: the delay model (aberration correction)

### Why it matters

CF/DMAS/SLSC/MV all sit on top of delays that assume constant sound speed.
When the assumption breaks, they are **cleverly summing signals that are
not in focus**.

Adaptive beamforming with local sound-speed map estimation has been
reported to improve resolution by 29%+ over the best global speed — on par
with or better than what MV gains over DAS.

Moreover, **MV is fragile under aberration.** When the covariance estimate
is corrupted, the adaptive weights optimize in the wrong direction and
cancel signal. CF/SLSC look at coherence and are comparatively robust.

### The adopted model: (a) phase screen

| Model | Content | Verdict |
|---|---|---|
| **(a) near-field phase screen** | one time/phase correction per element | **adopted** |
| (b) local sound-speed map | spatially varying delays | rejected |

**Why (a):**
- geometric delays stay; only a per-element scalar offset is added.
  **Delay-table translation invariance survives**
- estimation at 1–5 Hz refresh suffices (speed/aberration are set by
  patient and probe placement). Amortized cost under 1%

**Why not (b)** — two rejection reasons and one expectation-setting note,
and the reasons do not bind at the same level (§0 separates role from
placement, and placement may change):
- **[role]** translation invariance collapses; the delay table stops being
  one table shifted per scanline and becomes per-pixel data, and the
  contribution map has to be re-derived from the sound-speed map. This cost
  is paid on any hardware. Whether it can be *hidden* — a table made
  implicit by, say, hardware interpolation — is an **unverified
  hypothesis**, and it is the first thing a re-hearing would have to
  measure; nothing here asserts it in either direction
- **[placement]** per pixel × channel path integrals are gather/sequential
  work, the shape the current placement's compute (Tensix) is worst at. A
  placement on which per-pixel gather is cheap removes this objection — the
  property is what matters, and no such placement exists in this repository
  today
- **[expectation-setting]** the 29% figure belongs to (b), so **keep
  expectations honest** — this is not a reason against (b); it is a reason
  not to promise (b)'s results while running (a)

Delay computation is nevertheless abstracted as "path integral over a
sound-speed map," degenerating to conventional DAS under a constant map —
leaving room to grow into (b).

**What would walk through that door.** The rejection is re-heard — not
reversed here — when both conditions of #44 hold: the hardware-neutral
estimator of #42 has passed L0, so (b) would be measured against a working
(a) rather than against a fixed 1540 m/s; and a placement exists in this
repository on which the placement-bound objection does not hold. The
re-hearing's first product is an ADR (ADR-0004 lifecycle), and its evidence
must include the implicit-table hypothesis above, measured. Either way, (b)
would not touch the fixed-iteration rule of the real-time path: the
estimator producing a map is an asynchronous producer of parameter
generations, outside the determinism boundary, exactly as (a)'s estimator
is.

### Estimation algorithm

**Coherence maximization.** R is already being built, so the extra cost is
near zero. (Neighbor cross-correlation accumulates error; global speed
search is cheap but coarse.)

Update rate and spatial smoothing extent are design items.

### Applicability by probe type

- **1D linear**: all-element data without μBF → **element-level phase
  screen possible**
- **2D (post-μBF)**: intra-subarray phase errors are already frozen →
  **restricted to a phase screen across post-μBF channels**

---

## 9. Layer 2: summation weights

### Everything derives from R

CF, DMAS, SLSC, MV are **functions of one and the same quantity**. From the
per-pixel channel vector `x`, form `R = x xᴴ`:

| Method | Expression in R |
|---|---|
| DAS | `1ᵀx` (does not even need R) |
| CF | `(1ᵀR1) / (N · tr R)` |
| F-DMAS | `(1ᵀR̂1 − tr R̂) / 2`, where `R̂` is the R of signed-square-root preprocessed signals |
| SLSC | sum of band-diagonal entries of normalized R |
| MV | `R⁻¹a / (aᴴR⁻¹a)` |
| ESBMV | eigendecomposition of R |

**CF and F-DMAS are the identical computation** (contractions `1ᵀR1` and
`tr R`); the only difference is the elementwise signed square root.
Implementations share nearly 100%.

For F-DMAS, substituting `ŝᵢ = sign(sᵢ)√|sᵢ|` gives
`Σᵢ<ⱼ ŝᵢŝⱼ = ((Σŝ)² − Σŝ²)/2`, and **O(N²) drops to O(N)**. Do not write
the naive double loop.

### Smoothing R

**Keep the raw outer products and let each method do its own contraction.**
Reason: MV wants subaperture averaging + axial time averaging, SLSC wants
full-aperture correlations without subaperture averaging, CF/DMAS want
unsmoothed instantaneous values. Sharing one R across methods requires the
raw outer products.

Smoothing is specified **independently along three axes: subaperture ×
depth × slow time**. B-mode (slow-time length 1), color Doppler, and SLSC
are then the same code with no per-method branches.

Axial time averaging as a **sliding-window incremental update** (subtract
the old term, add the new) cuts R-formation cost to 1/5.

### Solving MV: Newton-Schulz

**No Cholesky.** Its sequential dependencies idle the Tensix matrix engine.

```text
X_{k+1} = X_k (2I − R X_k)
```

- **All matmul.** Quadratic convergence, ~8 iterations
- With diagonal loading ε = 1/100, condition number ~100, convergence is
  stable
- Complex arithmetic decomposes into 4 real matmuls (3 with Karatsuba)

CG solving `Rw = a` directly needs 1/25 the FLOPs — rejected. **The reason
is matrix-engine utilization, not DRAM bandwidth** (R lives in L1/SRAM, so
512 GB/s is not the binding constraint). CG's sequential dot products and
vector updates leave the matrix engine mostly idle. Newton-Schulz, written
as batched small matmuls, is faster in wall-clock terms even paying 25× the
FLOPs. This is the single biggest reason not to port a GPU implementation
as-is.

**The Newton-Schulz initial value is fixed as part of the specification**
(e.g. `X₀ = Rᴴ/(‖R‖₁‖R‖∞)`). Convergence and the required iteration count
depend on it, and determinism demands a fixed default. Whether 8 iterations
suffice at κ≈100 is settled offline, initial value included.

### Precision split (to be measured)

Each iteration passes 16 matmuls, so pure BF16 accumulates error and can
break quadratic convergence. **The working hypothesis is a hybrid:**
- first 4 iterations in BF16 (rough convergence)
- last 4 iterations in FP32 (polish)

FP32 matmul runs at ~1/4 of BF16, so total cost is 1.75×. The reference
implementation sweeps "N iterations in BF16, then M in FP32."

### Beamspace MV

Project channel space onto a low-dimensional orthogonal beam basis, then
solve MV. `L³` falls dramatically (64 → 16 is 64×).

- The signal subspace is inherently low-rank; 16 dimensions span the
  meaningful degrees of freedom
- Sidelobe-suppression capability depends on the basis choice. **Measure**

### Nonlinear-processing cautions

CF, DMAS, MV are nonlinear: they break speckle statistics, put dark blotches
in anechoic regions, and can invalidate downstream quantitative measurements
(Doppler, elastography, attenuation estimation). **Clinical review will
attack exactly here.** The countermeasure is the multi-stream configuration
of §11.

Also, **MV cancels signal near strong reflectors**. Papers underplay it, but
clinically it surfaces as **needles disappearing** — immediately
disqualifying for biopsy guidance. Phantoms must include high-echo
structures.

---

## 10. Compute estimates

### Governing law

MV cost is dominated by the inverse, `L³` (L = subaperture size). Growing
elements scales `L ∝ N` and scanlines `∝ N`, so the **total goes as N⁴**.
64 → 128 receive channels is 16×.

### 30 fps, 2048 depth points, Newton-Schulz ×8, complex→real ×4

| Configuration | Recv ch | L | TFLOPS | Cards @40% |
|---|---|---|---|---|
| 128 elem / 64 ch recv | 64 | 32 | 35 | 1 (26% used) |
| 256 elem / 128 ch recv | 128 | 64 | 560 | 5 (4.2 rounded up) |
| 256 elem + beamspace (B=16) | 128 | 64→16 | 19 | 1 (14% used) |
| post-μBF 256 ch, MV, volume | 256 | 128 | 1,100 | 9 |
| post-μBF 256 ch + beamspace | 256→16 | – | 37 | **1 (28% used)** |
| 2D fully digital 4096 ch full MV | 4096 | 2048 | ~72,000,000 | ~540k (impossible) |

### By method (64 recv ch, 30 fps)

| Method | TFLOPS | of one card |
|---|---|---|
| DAS | 0.004 | ~0% |
| CF / PCF / F-DMAS | 0.015 | ~0% |
| SLSC | 1 | 0.3% |
| MV: R formation only (sliding) | 2 | 0.6% |
| MV: with Newton-Schulz inverse | 33 | ~10% |
| ESBMV (eigendecomposition) | 100–170 | 30–50% |

### Target configuration (1D 256 elem / 128 ch recv + post-μBF 2D)

| Mode | Beamformer | TFLOPS | of one card |
|---|---|---|---|
| 1D B-mode | DAS + phase-screen correction | ~5 | 2% |
| 1D B-mode | + SLSC / CF / DMAS | ~40 | 12% |
| 1D B-mode | + beamspace MV | ~25 | 8% |
| 1D color flow | per-channel wall filter + MV | ~30 | 9% |
| 2D volume | beamspace MV | ~37 | 11% |

**Everything for 1D at once is ~100 TFLOPS: ~30% of theoretical peak, or
~75% of one card's usable capacity at the 40% assumption. Quote the claim
with its basis attached.**

The four-card story: 2D volume with plain MV (beamspace approximation
removed), or 3D volume-rate/resolution upgrades.
**"Card count = how lavish the 3D" is the scaling story.**

### The 13 MHz note

λ/2 = 59 µm, so a 25 mm width needs 434 scanlines (1.7× the 5 MHz case).
Round trips are short (39 µs), so 60–100 fps comes out.

On pixel rate: scanlines ×1.7, fps ×2–3.3, but **depth points less than
half** (3 cm vs 6 cm), so the net is **2–3×** the 5 MHz / 30 fps case (an
older revision said "5–7×," which ignored the depth-point reduction).
Beamspace MV fits easily; plain MV (L=64) may fit one card — recompute.

**Table assumptions**: unless stated, 30 fps, 2048 depth points.
**Two capacity bases appear**: "of one card" percentages are against the
332 TFLOPS theoretical peak, while "cards" counts assume 40% effective
efficiency (133 TFLOPS usable per card). Never combine a percentage from
one basis with a count from the other. **The 40% is a target for
hand-written kernels, not a measured figure**: measured on one p150a
development board, the stock toolchain delivers 3.2% on this workload's
shapes, against 58.6%
on a large square matmul in the same run (docs/budget.md). The card counts
here therefore state what the design aims at, with a factor of twelve still
to close. The 4096-channel row follows
the N⁴ law from the 256-channel volume row; an earlier revision carried
1.85e8 there, which did not reconcile.

On the §12 latency table, **throughput and latency obey different rules**:
pipelining lets stages run concurrently on different frames, which raises
sustained throughput, but a single frame still traverses its critical
dependency path, and along that path the stage times add. 30 fps is the
processing-rate assumption; 60 Hz is the display deadline.

---

## 11. Multi-stream configurations

### A linear stream running alongside

Adaptive methods are nonlinear and break quantitative measurement — one of
the real reasons adaptive beamforming never shipped. With compute headroom:

**Display uses the adaptive stream; quantitative measurement uses a DAS
stream from the same acquisition.** Same transmit events, same R, so no
spatial or temporal registration error. The objection "adaptive processing
breaks measurements" is structurally dissolved.

This works only because DAS costs ~0% of the card.

### Color Doppler split

Velocity is a quantitative output with accuracy requirements; regulatory
weight differs from changing what B-mode looks like.

| Output | Beamformer | Reason |
|---|---|---|
| Color mask (where flow exists) | adaptive | blooming and clutter are display problems |
| Velocity / power values | DAS | leave the quantitative chain untouched |
| B-mode background | adaptive or DAS | selectable |

### A color-Doppler-specific constraint

**Weights are computed once per pixel and applied to the whole ensemble.**

Recomputing MV weights per slow-time sample varies the apodization shot to
shot, imprinting artificial amplitude/phase modulation on slow time →
false Doppler shifts, spectral broadening, velocity bias.

Fixed weights make each pixel a fixed linear combination — slow-time phase
is perfectly preserved. **Also computationally cheaper** (weight cost
amortizes over the ensemble length).

### What color Doppler gives back

The ensemble provides 8–16 snapshots for free, so R can average along slow
time. The same condition number is reached with less subaperture averaging:
**MV may run with a larger effective aperture than in B-mode**.

But static tissue is strongly correlated across slow time, so the effective
independent-sample count is smaller than nominal. The benefit accrues mostly
to the decorrelating blood signal. Measure.

### What headroom unlocks: per-channel wall filters

The usual order is beamform → wall filter → autocorrelation. With headroom
it **can be reversed** — wall-filter each of 64 channels first, then compute
MV weights. R adapts to the blood subspace instead of being
clutter-dominated, and clutter leaking through sidelobes is targeted
directly.

Cost is ×channels (a 12-tap polynomial regression lands at tens of GFLOPS).
Unrealistic on current machines, noise-level on the p150a. This is
**"abundant compute lets you reorder the pipeline itself"** in its purest
form.

### The demo that reads instantly

Color **blooming** past vessel walls is caused directly by lateral-PSF
sidelobes and mainlobe width. MV narrows the mainlobe, so it hits blooming
head-on. Radiologists recognize it at a glance, and it is clinically
meaningful (vessel diameter, stenosis grading, small-vessel separation).

---

## 11.5 Elastography

**Strain elastography first; shear wave is a future slot.**

Strain tracks tissue displacement between frames under manual compression.
Transmits stay ordinary B-mode sequences (no push pulses; no change to the
contribution map or MI/TI management). Processing is just an added dp-elasto
process; the only demand on enodia is "a phase-preserving complex IQ stream"
(the same contract as the color stream; the only difference is inter-frame
rather than slow-time correlation).

- The frame-rate requirement dominates (inter-frame displacement ≤ λ/4 →
  effective ≥ 50 fps). The MLA + compounding design that preserves frame
  rate is also what makes strain viable
- The initial implementation is axial, phase-based (Kasai-family) — light.
  The CPU-portability performance limit arrives when lateral speckle
  tracking is added
- Output the quality metric (correlation-coefficient map) from day one; it
  is a free by-product of displacement estimation

**Only one provision for shear wave**: do not close the transmit-type enum.
Shear wave adds push (non-imaging) and tracking transmits (ultrafast plane
waves — the plane waves rejected in §7 return for tracking only). The
thousands-of-fps tracking acquisition affects stream bandwidth and buffer
sizing, so §12 budgets are recomputed at adoption time (open-issues).

---

## 12. Latency and determinism

### Budget

| Stage | Estimate |
|---|---|
| Acoustic round trip (13 MHz / 3 cm) | 39 µs (physics) |
| One frame acquisition (4 MLA, 109 transmits) | 4.3 ms (physics) |
| Ethernet transfer + buffering | 1–2 ms |
| Compounding dependency (±4 transmits) | 0.35 ms (structural) |
| Beamforming compute | design target |
| Scan conversion + display | 5–16 ms |
| **Total target** | **≤ 30 ms** |

Acquisition takes 4.3 ms but display ticks at 60 Hz = 16.7 ms.
**The real frame rate is display-bound; compute gets 16.7 ms per frame.**

End-to-end (probe → display) beyond 100 ms feels wrong to the operator.

### What breaks determinism, and the counters

**(1) Data-dependent execution time → fix everything**
- Newton-Schulz iterations: fixed count, no convergence test
- Contributing transmits: fixed cap, zero weights outside
- Beamspace dimension, aberration-estimation iterations: fixed

**Principle: nothing in the pipeline may vary its execution time with image
content.** Adaptive processing changes weights, never work. MV has this
property natively — one of its virtues.

**(2) Host involvement → resident kernel + ring buffers**
- Kernels stay resident on the card, self-running off Ethernet receive
- The host does control only; it never enters the data path
- Output is DMA'd from the card side; the host only reads

**(3) Probe/mode switching → preloading**
- kernels shared; only tables swap
- several profiles resident in DRAM

**(4) Pipeline flush**
- the first/last scanlines of a frame have one-sided contributions →
  **renormalize the weights** or the frame edges show banding
- B-mode/color interleaves carry transmit-type matching in the contribution
  map

### Pipeline structure

```text
[Ethernet receive] → [ring buffer]
                        ↓
[front end: BPF + demod + decimation]   ← a few dedicated cores, per transmit
                        ↓
                  [IQ ring buffer]      ← holds the compounding window
                        ↓
[beamforming]                           ← most cores, per scanline
                        ↓
              [output ring buffer] → host / inference
```

Stages run independently, decoupled by ring buffers; inter-stage sync is
pointers only. One stage's variation does not propagate. Buffer depth is
2–3 ms.

### Measurement plan

- **frame-interval histograms** (P50 / P99 / P99.9 / P99.99 / max)
- end-to-end latency (card-side cycle-counter timestamps)
- W/frame
- dropped frames across mode switches

**P99.9 is the metric GPUs are bad at.** CUDA's driver and scheduler stack
is deep, dynamic clocking adds more, and outliers gather in the tail. A
resident, self-running TT kernel is structurally favored. **This is the
main battleground.**

Measurement needs continuous minutes-to-tens-of-minutes runs (1-in-1000
events). Also measure under deliberate load. Timestamps are taken on the
card (host-side timestamps would mix in host jitter).

**Build the timestamp mechanism in from the start. Retrofitting it is
painful.**

---

## 13. Core partitioning and organ recognition

### Spatial partitioning

Physically partition the 120-core grid, statically:

```text
cores 0–14    : front end (BPF/demod)
cores 15–89   : beamforming
cores 90–119  : inference
```

Each region has its own L1 and program, loosely coupled over the NoC.
**Spatial rather than temporal division: whatever inference does,
beamforming is unaffected.** Finer-grained and more deterministic than GPU
SM partitioning.

Shared resources (GDDR6 bandwidth) can still interfere, but a light U-Net
(10M parameters, 20 MB in BF16) can keep **weights resident in L1** and
barely touch DRAM.

### Inference compute

| Model | at 30 fps |
|---|---|
| U-Net (light, 256×256) | 1–3 TFLOPS |
| nnU-Net class (512×512) | 10–20 TFLOPS |
| SAM class (ViT-B) | 50–80 TFLOPS |

At 30% effective, 100 TFLOPS is available — everything but SAM-class fits
easily. **This is Tenstorrent's home game** (CNNs/Transformers map
directly).

### Input choice

| Option | Verdict |
|---|---|
| (a) B-mode image (polar) | **recommended.** Existing pretrained models apply as-is |
| (b) IQ data or R | richer, but no training data exists. Premature |
| (c) feedback of inference into the beamformer | per-region beamformer switching. **Future roadmap** |

### PoC treatment (lampas-tt)

**lampas is fixed as a concept only; its implementation architecture is
deliberately TBD.** The tap specification (the producing side) is enodia's
output interface and can be defined and implemented without settling lampas.

The PoC lampas-tt is the minimal "**open-source recognition model +
adapter**":

- model selection prioritizes "**runs on TT-NN / tt-forge**" over
  "meaningful for ultrasound." Picking from the UNet/YOLO family in the
  Tenstorrent model zoo is the shortest path. Meaningless labels on
  simulated images are still a real load, and the claim (inference
  alongside, P99.9 undisturbed) still holds. Meaningful recognition waits
  for real data
- **the adapter is lampas's responsibility and budget**: the tap contract is
  one form (polar, physical units, metadata); model-specific needs
  (resolution, resizing, normalization, reproducing training-time
  preprocessing) are absorbed by the adapter. A lampas-side scan conversion
  is a separate artifact from the display one (different objective:
  matching the model's input distribution vs looking right). On TT it
  lowers to a matmul against a precomputed sparse interpolation matrix
- run a dummy load (equivalent FLOPs, no model) in parallel for P99.9
  verification, so model porting never sits on the critical path

**Abstract core allocation so regions can be any number.** Adding a third
region later must not mean rewriting resource management.

**Tap specifications live in `docs/dataplane.md`.** The primary inference
input is T5 (post-envelope, polar). Do not make post-scan-conversion (T7)
the primary input — interpolation degrades information and ties input to
display resolution. Implement taps from the start.

### Two-card configuration

Inference on a separate card, linked by card-to-card Ethernet, is also
anticipated.

- Transfer: 1.3 MB/frame of B-mode × 60 fps = 80 MB/s → 0.08% of one
  QSFP-DD 800G port. Bandwidth is a non-issue
- Benefits: **failure isolation** (a hung inference cannot stop imaging),
  development isolation (Metalium vs TT-NN), verification/regulatory
  isolation, inference-only scaling

Card-to-card Ethernet maturity is established (Galaxy: 32 chips in
commercial operation; QuietBox: 4 cards). Two-card discovery is confirmed on
real hardware.

**The PoC starts with one card.** "Everything fits on one card with 70%
spare" argues better than "we need two." The product recommendation will be
two cards for failure isolation.

Abstract the output ring buffer so intra-card, card-to-card Ethernet, and
via-host transports are interchangeable.

---

## 14. Numerical precision

### Dataflow and required bits

```text
input           int16 (12 significant bits, TGC applied)
  ↓ front-end BPF (64 taps)     int32 accumulate → int16
IQ              int16 complex   ← L1-resident
  ↓ 4-tap interpolation + phase rotation   FP32 intermediate
  ↓ transmit compounding (9 transmits)     +3.2 bits
channel vector  FP32
  ↓ R = xxᴴ                     squaring doubles DR; FP32 mandatory
R               FP32
  ↓ Newton-Schulz (8 iter)      hybrid (measure)
weights         FP32
  ↓ apply                       BF16 acceptable
output
```

### L1-resident format: int16 complex

| Format | Size | Effective DR | Verdict |
|---|---|---|---|
| **int16 complex** | 4 B | 90 dB | **adopted.** Linear, so summation is exact |
| BF16 complex | 4 B | 8-bit mantissa = 48 dB | **insufficient against a 12-bit ADC** |
| FP16 complex | 4 B | 10-bit mantissa = 60 dB | barely viable |
| Block FP8 | 2 B | ~8 bits | halves capacity, but precision is risky |

**BF16 loses 4 bits against 12-bit ADC input.** When quantization crushes
the fine intensity variation of speckle, the texture turns unnatural — and
radiologists notice immediately.

Block FP8 would halve capacity, but **L1 already suffices; there is no
reason to gamble.**

### The golden is float32

The gap to BF16 is nearly four orders of magnitude — ample separation.
Three places keep float64:

- **geometry / delay tables**: one-time, so free. Removes "reference phase
  error" from the suspect list
- **spot-checks of MV inverses**: a few hundred pixels compared exactly.
  When a BF16 implementation misbehaves, "the reference is not the cause"
  is then immediate
- **ESBMV eigendecomposition**: float32 is marginal where eigenvalues
  cluster

**dtype is parameterized from the start.** The precision-vs-quality
degradation curve *is* PoC evidence. This is the part that is painful to
retrofit.

**Include TF32 (10-bit mantissa) in sweeps.** If Blackhole supports it,
"all iterations in TF32" may beat the BF16→FP32 hybrid on both speed and
accuracy for Newton-Schulz. Verify, measure.

---

## 15. Verification strategy

### Four layers and what each can prove

| Layer | Object | Proves | Cannot prove |
|---|---|---|---|
| L0 numerical equivalence | TT impl vs float32 reference | implementation correctness | image quality |
| L1 simulation | the algorithm | theoretical performance, truth comparison | real-machine behavior |
| L2 phantom | with real hardware | quantitative metrics, reproducibility | clinical usefulness |
| L3 in vivo | clinic | real aberration, operator acceptance | statistical superiority (case counts) |

**Do not conflate L0 and L1.** "MV runs on TT" is L0; "MV beats DAS" is L1;
"this machine improves" is L2+. Always state which layer a claim stands on.

### L0: numerical equivalence

Check **intermediate quantities, not images**. Eyeballing "looks similar"
makes later subtle mismatches untraceable.

Checkpoints:
1. front-end output (IQ) — relative error
2. post-delay channel vectors — relative error, phase error. **Both
   sides run the interpolation kernel §5 names**; a kernel difference is
   a contract difference, not a tolerance question (ADR-0007)
3. after transmit compounding
4. R — relative Frobenius-norm error
5. **MV weight vector — direction cosine, norm (mandatory)**
6. beamformer output (complex)
7. after envelope/log compression — dB difference

If 5 agrees, Newton-Schulz converged correctly, and any image difference is
downstream. Threshold: from the BF16 theoretical floor (relative 4e-3),
**within 1e-2 per stage**. Beyond that, suspect a bug.

**L0 is automated and lives in CI.** Checkpoints 1 and 2 exist as a
prototype (`enodia/spec/beamform/golden_compare.py`, #6): not yet TT against
reference — nothing runs on TT — but the IQ reference path against the RF
golden, stage by stage, with the reference quantity of each checkpoint
defined: for 1, the analytic signal of the RF times e^(−j2πf0·t),
band-limited by a brick-wall low-pass at fs/(2D) and read at the RF
positions the front end's samples stand for; for 2, the golden's own
delayed channel samples made complex (analytic signal delayed by the
golden's band-limited ideal delay at τ_i·fs, times e^(−j2πf0·t_p)), on the
lines through the scatterers, inside the aperture, with the phase error
energy-weighted. The report quotes the yardstick's own residual beside
every figure and flags a difference below it as not attributable (next
subsection). Measured on the provisional `linear-5mhz` profile; the record
is `docs/measurements/2026-08-23-host-iq-path-vs-golden.json` (ADR-0005):

| checkpoint | D=8 | D=4 |
|---|---|---|
| 1 front-end output vs ideal baseband — unquantized FIR / int16 record | 7.946 % / 7.946 %, 8.2° | 0.204 % / 0.204 %, 0.2° |
| 2 post-delay channel vectors, lines through the three scatterers | 21.1 / 21.1 / 19.9 %, 1.0 / 1.2 / 0.8° | 2.43 / 2.43 / 2.43 %, 0.20 / 0.22 / 0.20° |
| image, log envelope over pixels the golden puts above −40 dB | RMS 5.10 dB, max 20.8 dB | RMS 0.85 dB, max 5.6 dB |
| yardstick floor (golden's residual on this profile's record) | 0.0003 % | 0.0003 % |

The int16 record adds under 0.002 points to the FIR's own error at either
ratio; the 7.9 % at D=8 is the low-pass taking part of a band whose 3σ
reaches past fs'/2, and the 21 % at checkpoint 2 is that plus the four-tap
error §5's sweep puts at 14.0 %. Every figure is pinned by
`tests/test_golden_compare.py`; the axial-PSF consequence is in §5.

### The yardstick's own floor

The RF-domain ideal-delay DAS is not an L0 party — it is the yardstick that
measures how much the IQ + 4-tap path costs (CLAUDE.md, absolute rules) —
and a yardstick's own error has to sit well below what it measures. Its
fractional delays are taken by **band-limited upsampling by 8, then the
Lagrange cubic of §5** (`enodia/spec/beamform/rf_delay.py`): the record is
zero-padded by 256 samples at each end, upsampled through the real FFT
(spectrum zero-stuffed to 8×, Nyquist bin halved, inverse-transformed and
scaled by 8), and read at 8·t by the same kernel, coordinate convention and
zero-extension the IQ side uses. MVP-1 used 2-tap linear interpolation on
the RF, which is what #25 replaced.

**How its error is measured** — a frozen, discrete-time oracle
(`enodia/spec/beamform/rf_delay_sweep.py`), so the figure cannot drift with
the machine or the session. It has a name, **`rf-oracle-frozen-0p7`**: for
each carrier in {5, 13} MHz, the record is 256 samples at 40 MHz of the
simulator's own pulse at 0.7 fractional bandwidth (full width at half
amplitude, §4's convention — the 0.7 is the record's own constant, not read
from a profile), and the ideal delay of that finite sampled record is its
zero-extended sinc reconstruction, evaluated at t = n − μ over every sample
and 201 fractions. The record is historical synthetic evidence: it is not a
probe profile, its 13 MHz half in particular is not the 13 MHz profile #10
will create, and it is never relabelled as one — even though `linear-5mhz`
currently carries the same 0.7 and its 5 MHz record is therefore
numerically identical to this one. What a *profile* implies is a separate
output, below. The residual is

```text
100 × sqrt( Σ (candidate − ideal)² / Σ ideal² )   [%]
```

with both sums over all 256 samples and all 201 fractions. It measures
interpolation of the same sampled record #6 consumes; it says nothing about
pre-ADC fidelity or AFE aliasing (§4).

**The acceptance limit was set at one tenth of the IQ error being
measured, as §5 stated it when the benchmark was frozen** (#25) — 1.082 % at
5 MHz and 0.791 % at 13 MHz, from 10.82 % at the former 1.5 MHz edge and
7.91 % at a 6 dB rounding of the level. The limits are frozen with the
record and were not re-derived when #46 moved §5's figures to the
profile-derived 14.00 % and the exact-level 7.88 %: the 5 MHz limit is now
stricter than a tenth of what is measured, and the 13 MHz one is 0.003
points looser than a tenth of the envelope figure — both far above the
production residuals. Every *residual* below is pinned by
`tests/test_rf_golden_interp.py`, to the digit shown; the runtimes and
memory are measurements of one host and are not.

| method | 5 MHz | 13 MHz | s / frame | peak MiB / event |
|---|---|---|---|---|
| 2-tap linear (MVP-1) | 6.216 % | 38.707 % | 0.4 | 10 |
| Lagrange cubic, direct | 0.961 % | 28.560 % | 3.5 | 43 |
| Lagrange 8-point | 0.047 % | 20.586 % | 15 | 6 † |
| Lagrange 16-point | 0.000 % | 15.094 % | 58 | 6 † |
| Kaiser (β=8) sinc, 16 taps | 0.004 % | 11.463 % | 41 | 8 † |
| Kaiser (β=8) sinc, 32 taps | 0.003 % | 7.421 % | 82 | 13 † |
| rectangular sinc, 256 taps | 0.200 % | 0.242 % | 271 | 30 † |
| polyphase ×4 (Kaiser β=4, 640 taps/phase) + cubic | 0.004 % | 0.361 % | 93 | 78 |
| **FFT ×8, pad 256, + cubic (production)** | **0.000 %** | **0.099 %** | **8** | **86** |
| *least-squares bound, 4 taps on the contiguous support {−1,0,1,2}* | *0.186 %* | *16.472 %* | | |
| *least-squares bound, 4 taps on {−2,0,1,3} — best of the 3060 supports drawn from offsets −8 … +9* | *0.641 %* | *13.041 %* | | |
| acceptance limit | 1.082 % | 0.791 % | | |

Both cost columns are measured on **one transmit event** of the 5 MHz demo
(128 channels × 3373 samples in, 128 × 3012 positions out) on the
development host: the runtime is that event's, multiplied by the frame's
128 events; the memory is that event's peak, not normalized. Reported
rather than pinned, and the timing varies by about half between runs.
† These candidates are evaluated a channel at a time, which is why their
peak is small — vectorized over the stack, a 256-tap gather would hold
about 800 MiB.

**What the table says.** At 5 MHz the Lagrange cubic alone would do. At
13 MHz **no evaluated kernel of 32 taps or fewer reaches the limit**: the
record has −14 dB of energy at Nyquist, and a kernel of modest support has
a transition band below Nyquist that the signal occupies. Lagrange to 16
points and Kaiser-windowed sinc to 32 taps miss by an order of magnitude.
Four taps cannot be rescued by choosing them better: the least-squares fit
of four taps to the oracle itself — the best any kernel *on that support*
could do on this record — misses by twenty times on the contiguous support
every four-tap interpolator here uses, and by sixteen times on the best of
the 3060 four-tap supports drawn from the 18 offsets −8 … +9 around the
target — the same one-sided span the Lagrange nodes use (reviewed and
searched: {−2, 0, 1, 3}, 13.0 %). Nothing is claimed about supports that
reach outside −8 … +9. A finite kernel *does* reach the limit —
the 256-tap rectangular sinc, at 0.242 % — at 64 times the taps of a cubic
per sample. So the alternatives are a long
kernel, or upsampling once per record and reading it with a short one.
Among the costed methods that reach the limit, FFT upsampling is the
**fastest by measured frame time**, by an order of magnitude; it is not the
lightest — at 86 MiB per event it uses the most memory of them — and its
residual is set by the zero padding: with none, 0.972 % at 13 MHz — over
that carrier's 0.791 % limit (and under the separate 5 MHz one).

**Limit and floor are different numbers.** The acceptance limit — 1.082 %
and 0.791 % — is the upper bound the yardstick's own error must stay under
to be a yardstick. The **observed residual** of the production operator,
0.0003 % at 5 MHz and 0.0992 % at 13 MHz, is the *floor* of a comparison —
and the floor is a limit on **attribution, not on detection**. A comparison
against the golden observes any difference numerically, however small; what
it cannot do is tell a difference smaller than the yardstick's own residual
apart from that residual. So such a difference is not, by itself, evidence
of error or of improvement relative to the ideal — the golden itself sits
that far from the ideal, in a direction the comparison does not know.
`golden_compare` prints the floor beside its figures and marks any figure
below it "not attributable", so the rule is applied by the tool rather
than remembered by the reader (#6).

**Profile reconciliation — beside the frozen oracle, never in its place.**
The frozen figures above do not move. What a named profile implies is a
separate output (`enodia/spec/beamform/profile_reconciliation.py`,
`tests/test_profile_reconciliation.py`): for the profile and a decimation
ratio it records the profile name, status and source, carrier, bandwidth
fraction and one-sided edge, spectral level and width convention, the
revision that produced it, the profile-specific IQ-side result — the
Lagrange cubic's band-edge and pulse-weighted error at the profile's edge,
from §5's sweep — and the golden operator's residual on a record built from
the profile, with whether that record is numerically the frozen one. For
`linear-5mhz` (provisional, no source) at 5 MHz it reads: 8.25° / 36.58 %
at the edge and **14.00 %** pulse-weighted at D=8, 0.29° / 3.10 % and
**2.39 %** at D=4; RF golden residual 0.0003 %, the record being the frozen
5 MHz one to the bit. The IQ figure is what changed against what §5 said
before #46 — the former 1.5 MHz edge was inconsistent with that pulse — not
the RF record or its residual. **#10 is the trigger for the 13 MHz
reconciliation**: when it supplies a profile under §4's definition, the same
output is produced for it and reported beside the frozen 13 MHz figure. A
change to a profile's value or provenance reruns its reconciliation and
whatever consumed it; #6 consumes `linear-5mhz` on those terms.

### The simulator's role

Real data comes later, so the simulator owes **formal accuracy, not
acoustic fidelity**.

| | Importance |
|---|---|
| data format (256 ch × 40 MHz × 12 bit, transmit sequencing) | highest |
| data volume and rate (20 GB/s) | highest |
| geometric accuracy (MLA placement, transmit focus) | high |
| TGC, quantization, element dropout | medium |
| acoustic fidelity (finite element size, near field) | **low** |

**Neither Field II nor k-Wave is needed.** A home-built analytic simulator
suffices.

Reasons:
- no MATLAB environment. Field II is MATLAB mex binaries that do not run on
  Octave; personal MATLAB licenses exclude commercial use
- what Field II uniquely provides is the finite-element-size spatial impulse
  response. A 13 MHz linear probe (0.1 mm pitch, 0.12 mm wavelength, 3 cm
  depth) is far from its elements; the sinc approximation error is small
- the requirements that matter here (TGC discontinuities, element dropout,
  MLA, 12-bit quantization) are implementation-side effects Field II does
  not model
- maintaining two languages and environments would reliably slow the sweeps

**When Field II becomes necessary:** the home-built PSF looks suspect, an
external publication is planned, or the battleground moves to low
frequencies / near field. Buy the license then.

As complements: run k-Wave-python on one or two conditions to bound
finite-element-size effects; public datasets (PICMUS, CUBDL) provide
third-party confirmation of the DAS implementation.

### Metrics

- **No CNR; gCNR.** CNR improves spuriously under nonlinear dynamic-range
  manipulation — a good share of "SOTA" claims in the literature step on
  this rake
- **Never argue resolution from FWHM alone.** MV narrows the mainlobe while
  sometimes raising sidelobes; check **width at −40 dB** too
- **Speckle SNR** (mean/std; Rayleigh = 1.91) for texture preservation
- **Contrast resolution** (detectability of small hypoechoic structures) is
  independent of the above

Metrics improving while the image looks wrong is a normal occurrence.
**The final judge is a radiologist's eye.**

### Phantoms

| Phantom | Purpose |
|---|---|
| point-scatterer array (5 mm axial spacing, 3 lateral) | resolution; compounding shows most clearly |
| anechoic cysts (2/4/8 mm, at several depths) | contrast; gCNR |
| uniform speckle region | texture preservation; test against Rayleigh |
| aberrating layer (fat-layer model, variable thickness/speed) | Layer-1 effect |
| **high-echo structures (needles, calcifications)** | **where MV fails hardest. Mandatory** |
| flow phantom (moving point clouds, constant-velocity region) | color velocity-bias verification (§7) |

### Element dropout and variation

Real probes run with a few percent of elements dead. **MV is especially
fragile to dropout** (the covariance estimate breaks).

**Channel-health detection and exclusion are designed in from the start.**
Detect anomalies from no-signal-interval noise floors and neighbor
correlation, then **zero the receive apodization mask + excise the
rows/columns from R** (§6). Compute cost is negligible; retrofitting is
awkward.

---

## 16. Considered and rejected

A record, so the same debates are not repeated.

| Considered | Rejected because |
|---|---|
| Cholesky for MV | sequential dependencies idle the matrix engine |
| CG on `Rw = a` | arithmetic intensity ~1; 1/25 the FLOPs but slower |
| beamforming on raw RF | does not fit L1; DRAM-bandwidth-bound |
| BF16 L1 residency | 8-bit mantissa insufficient for 12-bit ADC |
| Block FP8 L1 residency | L1 already suffices; risk buys nothing |
| phase rotation alone for fractional delay | ~50° band-edge error |
| 2-tap linear interpolation | insufficient at 1.92× oversampling |
| Keys / Catmull-Rom 4-tap (a=−1/2) | magnitude flatness equal to Lagrange, phase error equal to linear (§5) |
| Keys 4-tap with a < −1/2 | wins the band edge by pre-emphasis; its error also tends toward zero as the pulse narrows, but more slowly than Lagrange (§5) |
| least-squares 4-tap fractional delay | a table per pass-band, not a formula two ports can check against each other (§5) |
| local sound-speed map (Layer-1 (b)) | breaks translation invariance (role-bound); gather-bound (placement-bound). Re-hearing gated by #44 (§8) |
| full MV on 2D probes | two orders of magnitude short |
| synthetic transmit aperture (STA, Stage 3) | single-element transmit SNR; deep field collapses |
| gather-style compounding | 340 GB/s DRAM traffic; collides with MV |
| Field II / MUST | no MATLAB; does not cover the actual requirements |
| k-Wave as the main tool | full-wave is overkill for a phase-screen model |
| bit compression | bandwidth is ample; lossy taints covariance |
| CNR as a metric | improves spuriously under nonlinear processing |
| **considering a 2D 16×16 = 256-element array** | not a real configuration; commercial 2D is μBF 4096→256 ch |
| enodia interpreting FPGA-facing data directly | one more reverse-converter; illusory guarantees; couples enodia to FPGA revisions. Demoted to a test-only verifier (§19) |
| control software computing contribution maps | exporting receive-side internals widens L0 verification into the control software. enodia derives its own (§19) |
| discretizing depth/focus (precomputed configs) | combinatorial explosion; conflicts with continuous knobs. Solved by the fast re-derivation path (§19) |
| unifying enodia→diaplous under tap | the processing systems need losslessness (slow-time completeness), contradicting tap. Split into stream/tap (§0) |

---

## 17. Unresolved (settled by measurement or investigation)

### Parameters decided by measurement

- **how much of the measured gap a hand-written kernel recovers.** The
  efficiency itself is measured: 3.2% from the stock toolchain against
  58.6% on a large square matmul, on one p150a (§10, docs/budget.md). The
  40% the card counts assume is the target that gap has to reach
- Newton-Schulz precision split and iteration count (incl. X₀ choice)
- beamspace basis design and dimension
- compounding window width, apodization, truncation count
- decimation ratio and interpolation tap count — **measured at 5 MHz** (#6,
  §5): D=8 broadens the axial PSF by ~40 % at −6 dB and costs 21 % at
  checkpoint 2 on the provisional profile, D=4 by 4 % and 2.4 %. What stays
  open: whether D=8's width is acceptable for the product (a Layer-2 /
  image-quality judgement, L1–L2), the same measurement on a sourced
  bandwidth and on the 13 MHz profile (#10), and the tap count, which the
  sweep bounds but the PSF does not separate from the front-end low-pass
- diagonal loading (2D may need more than 1D, μBF grating lobes)
- core allocation (front end / beamforming / inference)
- group-batch size and boundary artifacts
- aberration-estimation update rate and smoothing extent

### Investigation items

- ERISC custom-firmware development procedure; presence of a newer
  fabric-based EDM
- `run_routing()` firing conditions and jitter impact
- card-to-card latency/jitter measurement
- Tensix dest-register accumulation precision and read-out behavior
- AFE anti-aliasing behavior (is everything above 20 MHz gone at 13 MHz?)
- actual TGC behavior of the target front end (discontinuities, gain-step
  granularity). **With MLA the depth-TGC correspondence shifts per
  scanline, which can create inconsistencies under compounding**

### Unsettled specifications

- concrete clinical use for organ recognition (needs external input; sets
  model scale and latency requirements)
- exact target-probe list (placeholder set in use — linear 13/7.5/5 MHz,
  convex 3.5 MHz, sector 2.5 MHz, post-μBF 2D 256 ch; kinds and bands are
  right, exact element counts/pitches swap in later as profile data)
- concrete design of each diaplous stage (Doppler estimation, persistence…)
- where the measurement package (distance/area/flow) lives (diaplous or UI;
  ties into §11 "quantitative = DAS" — if diaplous, accuracy verification
  is hekatus's burden too)
- memory-pool mechanism details and its ecosystem address (§20)
- stream-bandwidth/budget recomputation at shear-wave adoption (§11.5)
- IP-landscape review at the productization gate

### Settled (record)

- MLA {2, 4} fixed; 8 is the color experiment slot (§7)
- depth/focus changes assume continuous knobs, with coalescing (§19)
- elastography: strain first (§11.5)

---

## 18. First tranche of work

> **Note (2026-08): the order of work is managed in GitHub issues and
> milestones, not in this section.** This section remains the record of the
> work inventory and its rationale. The two-track distinction and the
> warning against deferring Track B remain in force.

### Two tracks

Phase 1 runs two tracks in parallel.

**Track A: host-side reference implementation** (items 1–7 below)

**Track B: the TT spike.** The smallest TT implementation, with no
beamforming:
- a resident kernel fed synthetic data (format and rate identical to spec)
- ring-buffer decoupling between stages
- card-side cycle-counter timestamps and histogram collection
- determining the current EDM recommendation (deprecated vs fabric)
- measured effective efficiency on plain matmul workloads (testing the
  40% assumption)

**Why Track B must not wait**: the PoC's core claims (P99.9, resident
kernels, determinism) are all TT-side properties that no amount of host
polishing verifies. Leaving the riskiest, claim-critical part for last is a
structural mistake. With Track B done, the "claim-proving instrument"
exists from the start.

### Included (Track A)

1. **probe profiles / geometry** — 1D linear 5 MHz and 13 MHz, polar grid
2. **transmit-config description and contribution maps** — configs in the
   physical-quantity schema (§19, owned by hekatus), transmit-event
   sequences, MLA, derived contribution maps. The simulator is the schema's
   first consumer
3. **simulator** — point scatterers, element directivity, attenuation,
   aberration injection, TGC, 12-bit quantization, element dropout. Output:
   40 MHz raw RF int16
4. **front end** — complex BPF + decimation. Output: int16 complex
5. **beamforming (through DAS)** — delay abstraction, phase-screen hook,
   4-tap interpolation + phase rotation, transmit compounding,
   channel-health checking
6. **imaging and evaluation** — envelope, log compression, scan conversion,
   gCNR, FWHM, speckle SNR
7. **phantoms** — six kinds (point scatterers, anechoic cysts, uniform
   speckle, aberrating layer, high-echo structures, **flow phantom** for
   the 8-MLA color velocity-bias verification)

### Excluded (later tranches)

- R formation and adaptive beamformers (CF / DMAS / SLSC / beamspace MV)
- aberration estimation algorithms (injection and correction hooks only)
- color Doppler (the slow-time axis is reserved; processing unimplemented)
- TT implementation, Ethernet receive, core allocation
- inference integration

### Order of work

**Write 1–3 first and confirm the raw RF comes out as intended** before
4–7. If the simulator is wrong, everything after it is meaningless.

### Details settled at implementation time

- transmit-beam model (**virtual-source approximation adopted**, switchable
  to aperture superposition)
- **the virtual-source focal singularity (mandatory)**: the transmit delay
  `±|z − z_f|/c` flips sign at the focal depth, producing a discontinuity
  and hourglass artifacts near focus — a classic retrospective-transmit-
  focusing pitfall. Build the smooth blend across the focus from the start
- the contribution-map weight function (how the transmit-beam amplitude
  profile enters)
- spatial correlation of injected aberration (**give it a correlation
  length**; real fat layers are not white)

---

## 19. enodia's external API contract

enodia does not decide transmit/receive sequences. Control software on the
host PC owns the transmit master data; enodia is a **subordinate system**
that receives the description through an API and configures receive-side
computation from it.

### Input is a physical-quantity schema (owned by hekatus)

The transmit description enodia receives is defined in the **vocabulary of
physical fact**: element coordinates, virtual-source positions [mm], firing
delays [ns], apodization values, transmit-type tags. No FPGA-internal
representations (clock counts, register values). hekatus owns the schema.

**enodia does not interpret FPGA-facing data (considered, rejected).**
Recovering physical quantities from FPGA execution data means writing one
more reverse converter — the converter count does not drop, FPGA revisions
would break enodia, and a second specification would muddy L0 verification.

### The transmit-config scheme

The machine holds a **finite set of transmit configurations**; control
software announces a selected ID. **Depth and focus are in-config
parameters** (continuously variable — knobs are turned continuously).

```text
setup:            description of the config set (physical-quantity schema)
runtime (heavy):  config selection ID    → switch to precomputed derivatives
runtime (light):  in-config parameter change → fast re-derivation, applied within N frames
runtime (cont.):  ROI/gain etc. (independent of derivatives) → frame-tagged application point
readback:         health, statistics, (future) self-diagnosis
```

**Each transmit configuration references exactly one probe profile**, by
`probe_profile_id` (`ProbeProfile.name`, §4), at setup. The runtime config
ID therefore selects the profile transitively — the profile is part of the
table set the config ID names (docs/dataplane.md) — and **no bandwidth
number travels in the transmit description or the frame header**: the
effective two-way bandwidth and its provenance live in the profile and
nowhere else (§4, ADR-0008). Reference: `TransmitDescription` and
`TransmitConfig` in `enodia/spec/sequence/__init__.py`.

### Ingress: what enodia accepts (ADR-0009)

The description and the accepted configuration are **two types**.
`TransmitDescription` is the external form, in the units above —
millimetres, nanoseconds, dimensionless apodization, a transmit-type tag
that is an open set of strings. `accept(description, profile)` returns
`TransmitConfig` in SI units, and nothing downstream reads the external
form.

**The description carries no derivative.** In particular it does not say
which output line an event forms: maps are enodia's (below). The accepted
event's `line_index` is derived, and is the identity while the contribution
map is the identity.

**Canonical geometry wins.** The description transports element coordinates
and the probe profile already holds them (§4). They are compared, and then
the transported numbers are dropped: every **geometry-dependent** derivative
— delay tables, phase-rotation coefficients, aperture weights — is computed
from `ProbeProfile.element_x()`, so a port and this reference implementation
compute from one geometry and L0 compares like with like (ADR-0007's
principle). The comparison cannot be equality — converting `linear-5mhz`'s
coordinates to millimetres and back moves 6 of 128 of them, by at most
8.673617379884035e-19 m — so the tolerance is **4 units in the last place of
the aperture half-width**, 1.4e-17 m on that profile: thirteen orders below
one element pitch, and above what the unit conversion costs.

Conversion **divides by an exact power of ten** (`/ 1e3`, `/ 1e9`) rather
than multiplying by its reciprocal: 1e3 and 1e9 are representable in
binary64 and 1e-3 and 1e-9 are not, so division rounds once where the
multiplication rounds twice. The reciprocal form costs four times the
residue on this profile — 20 coordinates moving by 3.5e-18 m — which the
tolerance would still admit; the point is not to spend it.

**The description must be self-consistent.** For every element that fires —
apodization strictly positive — the firing delay plus the geometric time of
flight to the declared virtual source is one instant, within **1 ns**. This
is the first of the three lines of defence below, made executable: the
physical schema is the specification, so a description that disagrees with
itself is refused rather than beamformed. Silent elements are not checked;
a zero-weighted element's delay makes no physical claim.

**Refuse, never repair.** A profile-id mismatch, a wrong element count, a
non-finite quantity, a negative apodization weight, an all-silent aperture,
an empty transmit-type tag, an event index that is not its position in the
sequence, or a coordinate or delay past tolerance is an error. Processing
with a wrong description is worse than dropping frames, and a repaired
description is a description nobody wrote.

Event indices run 0..n-1 in sequence order rather than merely being
distinct: the index is the event's name in the frame header and, through the
identity map, its scanline.

**Carried, not consumed.** The per-element firing delays and apodization are
validated at ingress; no transmit field is synthesized from them. The
transmit beam model — the virtual-source focal blend and the switch to
aperture superposition — is §18's implementation-time detail and stays with
it.

### enodia derives its own derivatives

Contribution maps, delay tables, phase-rotation coefficients are internal
representations of receive-side computation. They are never received from
control software; enodia derives them deterministically from the sequence
description. L0 verification then closes inside enodia.

- **config switch (heavy)**: all derivatives precomputed and cached;
  zero-wait switching
- **in-config change (light)**: depth/focus changes invalidate delay tables
  and contribution maps, so a **fast re-derivation path** is an official
  operating mode. Derivation is elementwise sqrt/divide chains — ms-class
  on the TT SFPU. Target: "processing under new tables within a few frames
  (~50 ms) of accepting the change"
- **coalescing under continuous operation**: change events are not queued;
  the latest value overwrites. Changes arriving mid-derivation are handled
  next cycle at the latest value only (intermediate values skipped)

### The single source of truth is the data-header tag

The FPGA stamps each Ethernet header with **config ID + parameter-generation
counter**, and enodia processes data with the table set "the data itself
names." Nothing depends on the arrival order of API parameters vs data.

**New-generation data is discarded until the new tables are ready.**
Processing with wrong tables is worse than dropped frames (a plausible but
wrong image is a misdiagnosis risk). Losing a few frames during a depth
change is normal commercial-machine behavior and clinically fine.

### Verifying transmit-description consistency (three lines of defense)

Because FPGA-facing data and the enodia description are generated from the
master data by **separate converters**, mismatches (image looks fine,
subtly degraded, near-undiscoverable) are constrained by verification:

1. **the physical schema is the specification**: vocabulary, units, and
   coordinate conventions of the enodia-facing description are strictly
   defined
2. **the reverse converter is test-only**: convert FPGA-facing data back to
   physical quantities and diff against the enodia description
   (round-trip test, ns-class delay tolerance) in CI. A reverse-converter
   bug shows up as a verification failure (false positive), never in the
   image. It never enters the product data path
3. **self-diagnosis on the machine**: residuals of measured vs theoretical
   delays on strong reflectors (the same computation as §8 aberration
   estimation, near-zero extra cost) is the last line — physics itself,
   regardless of how many converters sit upstream

The Track A simulator's transmit definition *is* this physical schema, and
making the simulator the schema's first consumer stress-tests its
expressiveness before anything else is built. It does: `simulate_frame`
takes an accepted `TransmitConfig`, so a frame cannot be stamped with the
identity of a configuration it did not come from, and the schema had to
express the focused B-mode sequence — aperture, apodization and per-element
firing delays — before that frame could be produced (#52).

---

## 20. Memory management (missing piece, TBD)

Under process separation (§0), per-process memory use varies with machine
mode state. Whether worst-case static residency across all modes fits
(embedded-host RAM limits) is unknown.

**Candidate: static layouts per mode set + a generation-switched memory
pool**

1. One huge physical pool allocated once at startup (hugepages; no kernel
   allocation afterwards). Size = the largest mode-set total + a
   two-generation coexistence margin (not all modes run simultaneously —
   the mode-set collection is finite)
2. Partition layouts within the pool are precomputed static tables per mode
   set
3. Mode switch = layout-generation switch. No malloc runs
4. Old partitions are reclaimed when their generation refcount drains

This combines (a) static-residency determinism (zero runtime allocation,
zero fragmentation) with (b) dynamic-allocation efficiency. Layout
computation is startup/offline work.

**Address**: the mechanism is not hekatus-specific — it is a shared problem
across the compute-systems ecosystem, so it belongs in the ecosystem layer
(sitos distributing layout tables + sitometron admission checks). hekatus
prepares exactly two things now:

- each process **declares** its memory needs (mode set × process → size
  expressions). The reference implementation's probe-profile/mode
  definitions include buffer-size derivation
- processes never malloc; they receive **partitions (shared-memory offset +
  size) from outside**

With these two, either static residency or the pool can be slotted in
later.

---

## Appendix: key numbers

| Item | Value |
|---|---|
| Sound speed (reference) | 1540 m/s |
| Fat / glandular tissue | 1450 / 1550 m/s |
| Attenuation | 0.5 dB/cm/MHz |
| Round trip at 15 cm depth | 195 µs |
| λ/2 at 13 MHz | 59 µm |
| One-core L1 limit (raw RF int16 real, 2048 samples) | ~384 ch (reference) |
| One-core L1 limit (IQ int16 complex, 2048 complex samples) | **~190 ch (the actual resident format)** |
| Rayleigh speckle SNR | 1.91 |
| MV diagonal loading (initial) | ε = 1/100 |
