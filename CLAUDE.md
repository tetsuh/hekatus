# CLAUDE.md — hekatus

Project brief for coding agents, loaded at session start.
**Read `docs/design.md` before writing code** — the rationale behind every
design decision lives there. Process rules (TiDD, TDD-lite, ADR, merge policy) live
in [AGENTS.md](AGENTS.md) and [docs/development_workflow.md](docs/development_workflow.md).

---

## What hekatus is

**The compute system of a medical ultrasound machine**: everything after
received channel data reaches Ethernet. Transmit-sequence control, the UI,
and transmit hardware live outside hekatus (a host-side control program owns
them; hekatus receives their description through an API).

Roles and hardware placement are separate decisions. Role concepts are
hardware-neutral; placements can change.

| Role | Content | Current placement |
|---|---|---|
| **enodia** | receive, front-end, delays, transmit compounding, R formation, beamformer family, envelope | enodia-tt (Tenstorrent p150a) |
| **diaplous** | mode-specific processing before scan conversion + display system | Holoscan graph (GPU/CPU) |
| **lampas** | companion inference; observes taps non-destructively | lampas-tt (PoC) |

- enodia → diaplous is the processing path; lampas is a parallel observation path
- **The Track A reference implementation is the hardware-neutral
  specification of enodia.** Any future port (enodia-???) is accepted by
  numerical equivalence with the reference implementation (L0 verification)
- **Never write hardware-specific code in diaplous.** Compute cores are
  Array-API-conformant pure functions (NumPy-compatible in/out); Holoscan
  operators are thin wrappers. CI forces CPU (NumPy) execution of every
  compute core. Functional portability is claimed; performance portability
  is not
- lampas is concept-only for now; its implementation architecture is
  deliberately TBD. The PoC lampas-tt is a minimal "open-source recognition
  model + adapter", and model selection prioritizes "runs on TT" over
  "meaningful for ultrasound"

Ecosystem: sitos (distributed parameter store) / sitometron (admission,
scheduling, supervision, observation of the process fleet). Naming origins:
`docs/naming.md`.

## Current phase

Work is MVP-driven and ticket-driven: milestones hold the coarse plan and
GitHub Issues hold everything actionable. See AGENTS.md. The absolute rules
below and the implementation style are each MVP's Definition of Done.

---

## Absolute rules (permanent — from roles and product requirements)

These survive any change of hardware placement. Never "optimize" them away.

- **[Real-time path only] No convergence-gated variable loops.**
  Newton-Schulz iteration counts, contributing-transmit counts, beamspace
  dimensions: all fixed. The product requirement is worst-case latency;
  data-dependent execution time is not acceptable. Adaptive processing may
  change *weights*, never *work*. Offline convergence sweeps in the
  reference implementation, used to pick the fixed values, are expected and
  encouraged.
- **Never evaluate with CNR; use gCNR.** CNR improves spuriously under
  nonlinear processing.
- **lampas / taps never block the processing path.** Read-only views; late
  readers drop frames. The processing path does not wait. Adapter costs
  (scan conversion, normalization) belong to the consumer's budget.
- **enodia is never made to wait by any downstream.** Stream losslessness is
  guaranteed by sufficient depth + violation detection, not by
  synchronization.
- **Processing with wrong tables is worse than dropping frames.** While
  parameter generations are inconsistent, new-generation data is discarded.
- **No stateful processing in the display system.** Inter-frame filters
  (persistence and the like) belong to the data-driven processing group.
- **Never import Holoscan / CUDA / any specific backend in diaplous compute
  cores.**
- **Never mix the data plane and the parameter plane.** Anything involved in
  frame synchronization is data-plane (shared memory + header tags);
  anything that can switch by generation is parameter-plane
  (sitos-compatible).
- Transmit-type tags form an **open set** (shear-wave push/tracking
  transmits will join later).

## Absolute rules (TT-placement-derived — apply to enodia-tt / lampas-tt)

Consequences of Tensix characteristics. A port to different hardware may
revisit them — but even then, the only contract is numerical equivalence
with the reference implementation.

- **No Cholesky decomposition.** MV inverses are solved by Newton-Schulz
  iteration: sequential dependencies idle the Tensix matrix engine, while
  batched small-matrix matmuls keep it busy and are faster in wall-clock
  terms. Iterative solvers such as CG are rejected for the same reason. The
  initial value X0 is fixed as part of the specification (docs/design.md §9).
- **[TT implementation only] Never apply delays to raw RF.** Beamform after
  IQ demodulation + decimation: raw RF does not fit in L1 and collapses into
  bandwidth-bound behavior. **The host reference implementation, however,
  must always keep an RF-domain ideal-delay DAS (the golden path)** — it is
  the only yardstick that quantifies the "IQ + 4-tap interpolation"
  approximation error.
- **L1-resident format is int16 complex.** BF16 has an 8-bit mantissa and
  loses 4 bits against a 12-bit ADC; speckle texture breaks.
- **Coordinates are polar (scanline × depth).** Scan conversion is outside
  enodia. Polar coordinates preserve delay-table translation invariance for
  convex/sector probes.
- **Fractional delays are 4-tap interpolation + phase rotation.** Phase
  rotation alone leaves ~50° error at the band edge. The kernel is Lagrange
  cubic, defined to the coefficient in design.md §5, and it is part of the
  L0 contract: a port runs that kernel, and no tolerance absorbs a kernel
  difference (ADR-0007).
- **Never attempt full MV on a 2D probe.** The compute is two orders of
  magnitude short; beamspace is the assumption.

## Implementation style

- Every design parameter is sweepable (dtype, decimation ratio,
  transmit-compounding window, beamspace dimension, Newton-Schulz iteration
  split, MLA count, diagonal loading, group size). **These are decided by
  measurement, not debate.** No hardcoding.
- dtype runs float64 / float32 / bfloat16 (+ TF32 where available) through
  the same code.
- Geometry (delay-table generation) alone is float64: one-time, cost-free.
- Each process declares its memory needs (mode set × process → size
  expressions). Buffers are not self-allocated; partitions (shared-memory
  offset + size) are handed in from outside.

---

## Repository layout

```text
CLAUDE.md              this file
AGENTS.md              process entry point for coding agents
docs/
  design.md            design decisions and rationale (read before coding)
  development_workflow.md  complete process rules (TiDD, TDD-lite, ADR, merge)
  adr/                 architecture decision records
  budget.md            compute and latency estimates
  open-issues.md       items decided by measurement; unresolved questions
  dataplane.md         inter-process data-plane contract (stream / tap)
  naming.md            naming origins
enodia/
  spec/                the role's specification = reference implementation (Track A)
    probe/             probe profiles, geometry
    sequence/          transmit-config description (physical-quantity schema), contribution maps
    sim/               simulator
    frontend/          front end (complex BPF + decimation)
    beamform/          delays, transmit compounding, R formation, beamformer family, envelope
    phantom/           phantom definitions (including the flow phantom)
  tt/                  enodia-tt: Metalium kernels, Ethernet receive, ttspike (Track B)
diaplous/
  core/                compute cores (Array-API pure functions, hardware-agnostic)
  graph/               Holoscan wrappers, process definitions
lampas/
  taps.md              requirements as a tap consumer (concept only)
  tt/                  lampas-tt (PoC: open-source model + adapter)
tests/
```

## Terminology

| Term | Meaning |
|---|---|
| Layer 1 | The delay model (sound speed, aberration correction). Sits under DAS/MV |
| Layer 2 | Summation weights (DAS / CF / DMAS / SLSC / MV) |
| Contribution map | Weighted sparse mapping transmit event → scanline. Unifies MLA and transmit compounding |
| Stage 2 | Retrospective transmit focusing (transmit compounding) |
| MLA | Multi-Line Acquisition: several receive lines from one transmit. Fixed spec {2, 4}; 8 is an experiment slot |
| μBF | In-probe micro-beamformer. Aggregates ~4096 elements to ~256 channels on 2D probes |
| Beamspace MV | Project channel space onto a low-dimensional beam basis, then solve MV |
| Transmit config | One of the machine's finite transmit setups, selected by ID. Depth/focus are in-config parameters |
| Derivatives | Delay tables, contribution maps, phase-rotation coefficients — derived by enodia from the config description |
| stream / tap | The two data-plane policies: stream = lossless processing path, tap = drop-tolerant observation path |
| Generation tag | Config ID + parameter-generation counter in the data header. The single source of truth for synchronization |

## When in doubt

- Prefer **deterministic execution time** over compute-saving optimizations
- Prefer **numerical agreement with the golden** over image-quality gains
  (in the current phase)
- Prefer **sweepability** over added abstraction
- Read the relevant `docs/design.md` section; if still unclear, ask a human
