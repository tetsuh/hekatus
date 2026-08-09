# dataplane.md — hekatus inter-process data plane

All inter-process data exchange in hekatus uses **one mechanism**:
**a ring buffer in shared memory + sequence numbers**. Two policies ride on it.

**This contract is frozen to file-format rigor already in the
reference-implementation phase.** If the reference implementation handles
boundaries with plain function calls, the boundaries get reinvented when the
processes split.

---

## Mechanism (common)

- A ring buffer is **owned by its writer**; readers get a read-only view
- Every frame/record carries a monotonically increasing sequence number plus
  metadata: **config ID, parameter-generation counter, transmit-event number,
  acquisition timestamp (card-side cycle counter), transmit-type tag**
- The writer never waits. How a reader catches up differs by policy
- Overwrite collisions are detected by sequence-number comparison
- **Partitions (offset + size) are handed in from outside** (assuming the
  memory-pool design of design.md §20). A process does not allocate its own
- TT → host is DMA **push**. Completion is detected by polling the
  end-of-frame sequence number. The DMA write-ordering guarantee
  (payload → completion-flag visibility) needs investigation (open-issues B5)
- The TT cycle counter is calibrated against host CLOCK_MONOTONIC at startup
  and periodically, and the mapping travels in metadata (open-issues B6)

## Policy 1: stream (processing path, lossless)

Use: enodia → each diaplous data-driven processing system
(B-mode preprocessing / color flow / PW / elastography).

- **Losslessness is guaranteed by "sufficient depth + violation detection",**
  never by blocking. Depth is on the order of 100 ms (sized per mode data
  rate by the §20 layout)
- An overwrite violation (a record written before its reader read it) is
  counted and warned. A violation means the system is already in a fault
  state — a reader stalled for 100 ms — and losing data at that point is
  acceptable
- This is the contract for consumers that need slow-time completeness
  (PW spectra, color ensembles, elastography inter-frame correlation):
  **phase-preserving complex IQ**, ROI-restrictable
- Mode-specific content is distinguished by the transmit-type tag

## Policy 2: tap / latest (observation and display path, drop-tolerant)

Use: lampas observation; diaplous-internal result buffers → display system.

- The reader reads the latest completed record. If it falls behind, it
  **skips old records and catches up**
- A record overwritten mid-read is discarded (sequence-number check) and the
  reader moves on
- Drop rate is measured (high = the consumer is too slow — a diagnostic)
- The display system passes acquisition timestamps through and **marks the
  image stale on screen** when nothing updates for a set time (safety
  requirement: the operator must not mistake a frozen image for live)

---

## Observation points (taps)

| # | Stage | Format | Coordinates | Use |
|---|---|---|---|---|
| T1 | Front-end output | int16 complex | channel × depth | Raw IQ. Training-data generation, aberration analysis |
| T2 | After delay & transmit compounding | FP32 complex | channel × depth × scanline | Channel vectors |
| T3 | After R formation | FP32 complex | subaperture² × depth × scanline | Coherence metrics, aberration detection |
| T4 | Beamformer output | FP32 complex | depth × scanline (polar) | IQ beam sums |
| **T5** | **After envelope (inside enodia)** | **FP32 real** | **depth × scanline (polar)** | **lampas primary input** |
| T6 | After Doppler estimation | FP32 | velocity/power × depth × scanline | Flow-aware inference |
| T7 | After scan conversion | uint8 | Cartesian pixels | Reuse of existing models, display image |

- **Envelope detection lives in enodia** (|z| only, negligible compute).
  T5 existing on the card lets lampas-tt stay entirely on-card, with no
  GPU→TT backflow
- **T5 is the lampas primary input**: scan conversion degrades information by
  interpolation and depends on display resolution, while polar coordinates
  map one-to-one to acquisition geometry
- **The tap contract has a single form** (polar, physical units, with
  metadata). Per-model needs — resolution, resizing, normalization,
  reproducing training-time preprocessing — are absorbed by the **adapter on
  the lampas side**, within the lampas budget. A lampas-side scan conversion
  is a different artifact from the display one (different objective:
  matching the model's input distribution vs looking right)

### Data volumes (13 MHz, 434 scanlines, 1536 depth points, 60 fps)

| tap | per frame | at 60 fps | note |
|---|---|---|---|
| T1 | 87 MB | 5.2 GB/s | ROI / decimation mandatory |
| T2 | 1.4 GB | — | continuous full-frame observation impossible; ROI mandatory |
| T5 | 2.7 MB | 160 MB/s | continuous observation fine |

T1–T3 get an ROI interface ("one full-channel frame per second", etc.) for
training-data generation and debugging.

---

## Future: feedback (lampas → enodia)

In the configuration where inference results switch beamformers per region
(design.md §13 (c)), the tap becomes bidirectional. It works with one frame
of allowed lag, but **must never break the determinism of the processing
path**: if feedback misses its window, the previous frame's settings are
used. Never wait.
