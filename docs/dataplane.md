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

### Generation mismatch: discard, never substitute

The config ID and parameter-generation counter on a record name the table
set that record must be processed with. **A consumer whose matching tables
are not ready discards the record.** It never falls back to the previous
generation's tables: processing with wrong tables produces a plausible but
wrong image, which is worse than a dropped frame (design.md §19).

The table set a `(config ID, parameter-generation counter)` names **includes
the probe profile** the configuration references at setup (design.md §19,
`probe_profile_id`): the profile's bandwidth, and every derivative that
depends on it, is selected by that identity like any other table. No field
is added to the record for it — the identity above is sufficient, and
duplicating the profile or its bandwidth into the header would give the
data plane a second source of truth (design.md §4, ADR-0008).

- Discarding is counted and reported, distinguishably from an overwrite
  violation — the two have different causes
- A record is processable only when **both** identity fields match a ready
  table set — config ID *and* parameter-generation counter. The generation
  counter is not assumed unique across config IDs, so matching it alone
  could select the wrong tables
- Processing resumes on the first record whose full identity matches. No
  catch-up of the discarded interval is attempted
- Losing a few frames while tables are rebuilt (a depth or focus change) is
  expected behaviour, not a fault

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

### Data volumes

Assumptions: 13 MHz linear, 4 MLA (109 transmit events → 434 scanlines),
256 channels, 60 fps. MB = 10⁶ B.

Two different depth counts appear, and mixing them is the easiest way to
get these figures wrong. **T1 is channel data**, so its depth axis is the
per-channel IQ sample count after decimation — 780 samples at 13 MHz with
D = 2 (design.md §4). **T2 and T5 are formed lines**, so their depth axis
is the image grid, 1536 points. Sample sizes also differ: T1 is int16
complex (4 B), T2 is FP32 complex (8 B), T5 is FP32 real (4 B).

| tap | axes | per frame | at 60 fps | note |
|---|---|---|---|---|
| T1 | 109 events × 256 ch × 780 IQ × 4 B | 87 MB | 5.2 GB/s | ROI / decimation mandatory |
| T2 | 434 lines × 256 ch × 1536 × 8 B | 1.4 GB | 82 GB/s if unrestricted | continuous full-frame observation impossible; ROI mandatory |
| T5 | 434 lines × 1536 × 4 B | 2.7 MB | 160 MB/s | continuous observation fine |

T1–T3 get an ROI interface ("one full-channel frame per second", etc.) for
training-data generation and debugging.

---

## Future: feedback (lampas → enodia)

In the configuration where inference results switch beamformers per region
(design.md §13 (c)), inference influences how enodia computes.

**This is a parameter-plane update, not a reverse tap.** Taps stay
one-way and observation-only. What feedback changes — which beamformer
applies where — is generation-switchable state, and putting it on the data
plane would mix the two planes, which the absolute rules forbid. So
feedback is expressed as a parameter update carrying a config ID, a
generation counter, and the frame from which it applies, exactly like any
other parameter change, and it becomes effective through the same
generation mechanism.

It works with one frame of allowed lag, but **must never break the
determinism of the processing path**. Determinism here means output must
not depend on when an update happened to arrive, which fixes the
disposition of a late update:

- An update names the frame from which it applies. The activation boundary
  is the start of that frame's processing — the same point at which any
  other parameter generation becomes effective
- An update that arrives after its own activation boundary is **discarded
  and counted**, not applied late. Applying it at an arbitrary later frame
  would make the image depend on arrival timing
- Every frame is processed with the last generation committed before its
  boundary; a discarded update leaves that generation in force until a
  subsequent update lands in time
- lampas may re-issue a discarded update naming a later frame. enodia never
  waits for one
