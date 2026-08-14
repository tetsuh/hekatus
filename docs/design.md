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
  the board peaked at 93 W at full clock. Which one is enforced remains
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
- f0, bandwidth
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
+ a fractional part (the 4-tap interpolation of the next subsection) + a
phase rotation (one complex multiply)**, making random access small and
local. With fixed geometry, the phase term is precomputable.

### Fractional-delay accuracy (important)

**Phase rotation alone is not enough.** As long as decimation goes to
Nyquist, the band-edge phase error is ~50° regardless of frequency:

- 13 MHz (D=2, 50 ns spacing, max fraction 25 ns, band edge 5.2 MHz) → 47°
- 5 MHz (D=8, 200 ns spacing, max fraction 100 ns, band edge 1.5 MHz) → 54°

Left alone, the axial PSF collapses and sidelobes rise.
**4-tap interpolation on IQ + phase rotation** is required; 2-tap linear is
not enough.

Keep the ability to measure how the point-scatterer axial PSF changes
between decimation ratios 8 and 4.

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

**Why not (b):**
- translation invariance collapses; per pixel × channel path integrals
- that is gather/sequential work, **the shape Tensix is worst at**
- the 29% figure belongs to (b), so **keep expectations honest**

Delay computation is nevertheless abstracted as "path integral over a
sound-speed map," degenerating to conventional DAS under a constant map —
leaving room to grow into (b).

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
hand-written kernels, not a measured figure**: measured on the target board
the stock toolchain delivers 3.2% on this workload's shapes, against 58.5%
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
2. post-delay channel vectors — relative error, phase error
3. after transmit compounding
4. R — relative Frobenius-norm error
5. **MV weight vector — direction cosine, norm (mandatory)**
6. beamformer output (complex)
7. after envelope/log compression — dB difference

If 5 agrees, Newton-Schulz converged correctly, and any image difference is
downstream. Threshold: from the BF16 theoretical floor (relative 4e-3),
**within 1e-2 per stage**. Beyond that, suspect a bug.

**L0 is automated and lives in CI.**

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
| local sound-speed map (Layer-1 (b)) | breaks translation invariance, gather-bound. Reserved as an extension |
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

- **effective efficiency (estimates assume 40%; measure first on Track B)**
- Newton-Schulz precision split and iteration count (incl. X₀ choice)
- beamspace basis design and dimension
- compounding window width, apodization, truncation count
- decimation ratio and interpolation tap count
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
expressiveness before anything else is built.

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
