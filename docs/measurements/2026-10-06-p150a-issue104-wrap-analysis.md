# Issue #104 wall-clock-wrap analysis supplement

This is a board-free supplement to the three Issue #104 records. It does not
rewrite any existing JSON or CSV record, and no raw timestamp array is stored
in git. The analysis was run against the retained external artifacts with
`tools/analyze_issue104_wrap.py`. The optional Issue #12 artifact was present
and was analyzed with the same method.

## Definitions and method

The resident runner exports one absolute timestamp per record with
`struct.pack("<Q", value)` (`enodia/tt/bench/run_resident.py:402-404`). The
analysis therefore treats each input as an unsigned 64-bit little-endian
value, not as a signed value, delta, or 32-bit value. The consumer captures the
absolute 64-bit `end` timestamp and writes its low and high words before the
DMA (`enodia/tt/bench/kernels/resident_consumer.cpp:106-112,136-147`). The
raw count and SHA-256 are checked against the committed record before any
analysis.

A paired outlier is the same bounded observation used by the records:

- adjacent intervals have one value `< 1,250,000` ticks and the other
  `> 1,450,000` ticks;
- their sum is within 1,000 ticks of 2,700,000 ticks;
- the pair's event position is the endpoint of the first interval.

The wrap definitions are deliberately separate from the pair definition:

1. A **clock crossing** is an adjacent timestamp pair for which
   `floor(t[i] / 2^32)` increases.
2. A **phase-window wrap candidate** is a crossing endpoint whose raw
   `t[i] % 2^32` lies in `[0, 135,000)` ticks, the first 0.1 ms after wrap at
   1,350 MHz. This is the denominator for the candidate ratio; all observed
   crossings are also reported separately.
3. An **event pair** is the first endpoint of a detected short/long pair. The
   event-to-candidate ratio is therefore
   `event-pair starts in the phase window / phase-window wrap candidates`, not
   pair count divided by an unstated observation length. A candidate that is
   not one of those pair starts is a non-event wrap candidate.
4. For phase, each event's absolute raw phase `t[i] % 2^32` is reported, and
   the requested elapsed phase is `(t[i] - t[0]) % 2^32`. The Rayleigh value is
   `R = abs(mean(exp(j * 2*pi*phase / 2^32)))`. Width is the minimum circular
   arc containing all event phases, converted from ticks to milliseconds.

The wall-clock period is exactly `2^32` ticks, or
`2^32 / 1,350,000,000 = 3.1814572563 s` (3.1814573 s rounded). At a 1 ms
frame interval this is `3,181.4572563` frames, so a frame-index gap gcd of 1
does not test or disprove this wall-clock period.

## Results

`R` and width use the event-pair starts, not the catch-up endpoint of the long
interval. Every event-pair start was in the defined post-wrap phase window.
The `elapsed phase` column is the folded `first_interval_end_elapsed_ticks`
range requested for this follow-up; the raw phase column shows the equivalent
near-zero post-wrap phase.

| run | raw count / SHA-256 | event pairs | pair sum ticks | pair order (short/long/other) | raw phase ticks min..max | elapsed phase ticks min..max | Rayleigh R | width (ms) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| sampler off | 400,000 / `b81ecd0b3c8853aabc05116779f477923c0172af532bc78a3a45270cb81d6032` | 22 | 2,699,965..2,700,085 | 22 / 0 / 0 | 11,885..11,942 | 3,112,902,759..3,112,902,816 | 0.9999999999999997 | 0.0000422222 |
| sampler default (2 s) | 400,000 / `56655e1de97c7151443480306347e1c4f413161bd43c2b9a959ef152e5070d0f` | 18 | 2,699,961..2,700,083 | 18 / 0 / 0 | 11,897..11,944 | 2,396,392,253..2,396,392,300 | 0.9999999999999996 | 0.0000348148 |
| sampler explicit (5 s) | 400,000 / `14fbed9b02de60773c9b6c64a14e44cb0073616e8c4ace107a60036609a7a944` | 9 | 2,699,970..2,700,030 | 9 / 0 / 0 | 11,896..11,936 | 3,165,310,313..3,165,310,353 | 0.9999999999999998 | 0.0000296296 |
| Issue #12 final 500k | 500,000 / `6a7baa6660de45ede2b6f752c7623ae6d25787a8c4613c13a81b2faca4480dec` | 27 | 2,699,967..2,700,034 | 27 / 0 / 0 | 11,889..11,941 | 3,918,176,100..3,918,176,152 | 0.9999999999999999 | 0.0000385185 |

The phase widths are all far below 0.1 ms. The displayed `R` values round to
1.000, but the unrounded floating-point values above are retained rather than
claiming an exact mathematical 1.

### Pair endpoints and committed-record correspondence

The analyzer independently redetected the pairs from the raw absolute values
and compared both endpoint indices and interval values with each committed
record. Every comparison was exact: no missing starts, unexpected starts,
endpoint mismatches, order mismatches, or interval-value mismatches occurred.
Each listed start `s` denotes the two event endpoints `[s, s+1]`.

- **Sampler off:** starts `5488, 40484, 43665, 69117, 85024, 88206,
  110476, 126383, 151835, 190012, 199557, 247279, 260004, 263186,
  285456, 301363, 326815, 342722, 345904, 361811, 384081, 387263`.
- **Sampler default:** starts `8139, 27227, 43135, 46316, 59042, 62223,
  74949, 81312, 87675, 129034, 205389, 234022, 240385, 288107, 294470,
  342192, 389913, 393095`.
- **Sampler explicit (5 s):** starts `88245, 107333, 110515, 225047, 228229,
  247317, 256862, 285495, 342761`.
- **Issue #12 final 500k:** starts `6084, 37899, 111072, 117435, 120617,
  126980, 152431, 158794, 168339, 193790, 200153, 212879, 219242, 266964,
  273327, 279690, 305141, 308323, 314686, 352863, 359226, 375133, 422855,
  432400, 457851, 464214, 470577`.

### Wrap-candidate denominator and pair order

The table reports both denominators to avoid conflating all clock crossings with
crossings whose observed completion landed in the 0.1 ms phase window.

| run | all `2^32` crossings | phase-window candidates | event-pair starts in window | event pairs / candidates | event pairs / all crossings | non-event candidates | event starts outside window | candidate order short/long/other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sampler off | 126 | 35 | 22 | 22/35 = 0.628571 | 22/126 = 0.174603 | 13 | 0 | 22 / 0 / 13 |
| sampler default (2 s) | 126 | 31 | 18 | 18/31 = 0.580645 | 18/126 = 0.142857 | 13 | 0 | 18 / 0 / 13 |
| sampler explicit (5 s) | 125 | 23 | 9 | 9/23 = 0.391304 | 9/125 = 0.072000 | 14 | 0 | 9 / 0 / 14 |
| Issue #12 final 500k | 157 | 41 | 27 | 27/41 = 0.658537 | 27/157 = 0.171975 | 14 | 0 | 27 / 0 / 14 |

Thus event pairs are not the same thing as every observed wrap candidate: each
run has non-event candidates. Conversely, every detected event pair starts at a
post-wrap candidate in the defined window. All detected pair orders are
short-first; no long-first or order-mismatched event pair was observed. The
`other` counts in the last column are phase-window candidates that do not form a
qualifying pair, not silently discarded pairs.

## Source inspection and PR #103 correction

The pinned v0.75.0 tt-metal source revision used for the board-free inspection
was `d9a68815f5fcf08a5bfbffb6f1f811823fba8edd`; only relative source paths and
line ranges are recorded here:

- `tt_metal/hw/inc/internal/tt-1xx/risc_common.h:245-250` implements
  `get_timestamp()` as low-register read followed by high-register read. The
  comment states that reading low samples/freezes the upper 32 bits for
  readback.
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h:503-510`
  implements `read_wall_clock()` with the same low-then-high ordering and the
  explicit comment `low` “latches high”.
- The pre-PR #103 resident producer and consumer called `get_timestamp()` locally
  on their respective cores (`enodia/tt/bench/kernels/resident_producer.cpp:60-121`
  and `enodia/tt/bench/kernels/resident_consumer.cpp:64-150`). The committed
  timestamp is the designated consumer completion timestamp; there is no direct
  same-clock producer-write stamp because the producer uses a different core.
  Direct producer-versus-consumer attribution therefore remains open because
  there is no producer stamp on the same designated clock.

The source's low-then-high details and the controlled experiment now have
separate roles. The paired events disappear after switching to low-word-only
reads with software wrap tracking, which supports the old measurement-clock read
path as the cause. The short-first order is consistent with an early producer
send, but this is not direct producer-versus-consumer attribution. Direct
producer-versus-consumer attribution remains open because there is no producer
stamp on the same clock. PR #103 replaced that path with a low-word-only 32-bit
read plus software wrap tracking. Its current 500,000-frame record,
`docs/measurements/2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json`,
records `pair_count=0`, `min_ticks=1,349,924`, and `max_ticks=1,350,068`,
versus 27 pairs in the prior record. This controlled change is indirect
evidence consistent with the measurement-clock read path; it is already the
mitigation in PR #103, not a new fix prescribed by this PR. A shared latch
within a tile being overwritten between reads is a possible mechanism inferred
from the observation and PR #103 evidence only, not an established hardware
fact. No kernel functionality or benchmark default changed.

## Conclusion and device status

The three Issue #104 runs and the available Issue #12 500k run all show the
paired event starts phase-locked to the `2^32`-tick wall-clock wrap, with
measured `R` values above and widths below 0.1 ms. The event order is short
interval first, followed by the long catch-up interval. The same phase lock is
present with the sampler off, so a host telemetry sampler is not a necessary
condition. The paired events disappear after switching to low-word-only reads
with software wrap tracking. This supports the clock-read path as the cause.
Direct producer-versus-consumer attribution remains open because there is no
producer stamp on the same clock. The short-first order is consistent with an
early producer send, without establishing producer-side attribution. The
controlled record
`docs/measurements/2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json`
reports 500,000 frames, `pair_count=0`, and `min_ticks=1,349,924` through
`max_ticks=1,350,068`, versus the prior 27 pairs. A shared tile latch being
overwritten remains a hypothesis, not a proven mechanism. PR #103 already
contains the mitigation, and no additional hardware run or new fix was
performed in this follow-up.

No SSH, Docker, device run, reset, or health probe was performed in this
follow-up because the board was unavailable. The previously authorized runs
were not rerun. Any changed-kernel Watcher validation, timing rerun, or
recovery protocol remains pending explicit board availability and owner
approval.
