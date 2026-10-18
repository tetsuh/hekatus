# Issue #137 board-power intervention — 2026-10-18

## Scope and provenance

These are the eight owner-approved revision-2 long instances, each executed
once in the preregistered order. Each workload interval is 120 seconds under
a 170-second guard. The plateau is [60, 120) seconds from the recorded first
launch start, using positive-duration left-held weighting without extrapolation.
No retry, reset, extra instance, calibration, or warmup was performed.

- Plan: [revision 2](../../2026-10-18-host-newton-schulz-issue137-plan.json),
  unchanged from `f7bdb7a`.
- Plan SHA-256: `19b10e461410fc515c60637ce9ca68a23c2165c00314c795ad3ad4694dfb8da1`.
- Harness: `fd7f659e3e54550802f06b8c0bb4b28d842b619f`.
- Device: serial `0000040331914124`, PCI `0000:09:00.0`, node
  `/dev/tenstorrent/1`, exposed as container-visible device 0.
- Firmware bundle: `19.6.0.0`; KMD: `2.11.0`;
  kernel: `Linux 6.8.0-138-generic`.
- Image digest:
  `sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621`.
- Host snapshot readers: tt-smi `6.2.1`, tt-umd `0.9.8`.

The raw wrapper result, normalized environment, pre/post snapshots, Issue137
trace, and companion wrapper power trace are beside each canonical run record.
The canonical record binds its wrapper result and core plan; trace hashes and
byte counts are retained. JSON and CSV artifacts are immutable after this data
commit. Smoke artifacts and the earlier unauthorized captures are not included.

## Preregistered outcome

**Classification: H-OTHER.** P1, P2, and P3 pass in both order halves, but no
hypothesis satisfies its complete pinned-signal rule. The expected H-BPL label
was not observed; the original prediction and every criterion remain unchanged.

- P1: both C018 means are 1350 MHz, above the 1282.5 MHz boundary.
- P2: every C081 and C108 mean is below 1282.5 MHz.
- P3: C081 exceeds C108 in each half: 1194.044573 > 1099.327633 MHz and
  1190.987574 > 1095.488053 MHz.
- H-BPL: runs 2, 4, and 7 are pinned, but run 5 (C081) is
  **290.8248480083167 W**, below the unchanged 294 W lower boundary for its
  300 W pre-run limit. Support requires all four high-configuration instances,
  not their pooled configuration means.
- H-TDC, H-TDP, and H-THM: none of the four high-configuration instances is
  pinned to the corresponding current, ASIC-power, or temperature limit.
- C054 remains exploratory and does not enter the verdict.

All required verdict inputs are present. The result follows the preregistered
`none_supported` policy, not a missing-data outcome. The P1/P3-failure and
TDC/TDP-support-specific mismatch branches are not activated. No configuration,
window, tolerance, or sample was changed to obtain a different label. This
pattern neither establishes nor excludes a physical firmware mechanism.

## Plateau duration-weighted means

| Run | Configuration | AICLK MHz | Board power W | Current A | ASIC power W | ASIC temperature degC |
|---|---|---:|---:|---:|---:|---:|
| 1 | C018 | 1350.000 | 254.095 | 132.476 | 112.663 | 77.284 |
| 2 | C108 | 1099.328 | 302.380 | 173.494 | 131.618 | 81.553 |
| 3 | C054 | 1309.132 | 299.322 | 169.537 | 140.187 | 81.992 |
| 4 | C081 | 1194.045 | 300.378 | 165.346 | 130.011 | 82.040 |
| 5 | C081 | 1190.988 | 290.825 | 183.354 | 143.887 | 82.363 |
| 6 | C054 | 1305.138 | 297.040 | 165.071 | 136.238 | 82.367 |
| 7 | C108 | 1095.488 | 296.640 | 178.852 | 135.862 | 82.046 |
| 8 | C018 | 1350.000 | 255.414 | 131.635 | 111.530 | 78.763 |

Each run uses its own pre-run limits: board power 300 W, current 200 A,
ASIC power 150 W, ASIC temperature 90 degC, and asic_fmax 1350 MHz. These
values agree across this session; no substitute limits were needed.

## Pinning facts

| Run | H-BPL signal | H-TDC signal | H-TDP signal | H-THM signal | Verdict-bearing |
|---|---|---|---|---|---|
| 1 | no | no | no | no | yes |
| 2 | yes | no | no | no | yes |
| 3 | yes | no | no | no | no |
| 4 | yes | no | no | no | yes |
| 5 | no | no | no | no | yes |
| 6 | yes | no | no | no | no |
| 7 | yes | no | no | no | yes |
| 8 | no | no | no | no | yes |

## Execution and cooldown

All eight canonical statuses are `ok`, each trace has 57 rows and terminal
coverage, and the largest observed overrun is 0.000667412 seconds. L1 fit
checks pass for every core plan; allocation values remain estimates rather
than measured bank mappings.

Runs 2–8 receive the preceding instance's pre-run snapshot through the CLI.
Their cooldown waits are 113.169677, 30.426603, 11.549000, 2.763794,
0.240501, 0.240532, and 0.244290 seconds, all below the 600-second cap.
Every gate meets preceding-baseline ASIC power +5 W and temperature +3 degC;
AICLK is not a gate. These relative gates do not establish a common initial
thermal state: recorded pre-run temperatures drift from 60.2 to 73.7 degC.
Docker is empty before and after every instance and after ownership helpers.

## Exploratory data quality and limitations

Four packed GDDR words and per-die Celsius channels are retained. Run 5 has
raw zero words in `gddr_4_5_temp_c` and `gddr_6_7_temp_c` at sample index 17,
34.684019644 seconds after workload start. They remain unchanged and should
not be interpreted as demonstrated physical zero temperatures. This point is
outside the plateau; GDDR is exploratory and never enters the limiter verdict.
`MAX_GDDR_TEMP` remains raw.

Compilation may occur on the first launch and is not individually timed.
There is no added warmup. This is one board and one approved series, not a
full-enodia capacity result or a causal firmware diagnosis.

## Reproduction and integrity

[analysis.json](analysis.json) is the unmodified analyzer result.
[manifest.json](manifest.json) lists the run order, all plateau means,
cooldown facts, environment, and SHA-256/byte counts of the source artifacts.
An independent CSV integration recomputed every five-signal plateau mean,
P1–P3, pinning, and cooldown baseline linkage. Replaying from this published
layout produced the same decoded analysis as the original local output.

Run this board-free reproduction command from the repository root; it writes
only a new local analysis file, not the immutable published artifacts:

```bash
uv run python - <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path("docs/measurements/issue137-board-power/2026-10-18")
records = sorted(root.glob("runs/*/run-issue137-long-*.json"))
assert len(records) == 8
output = Path("out/issue137-long-reanalysis.json")
output.parent.mkdir(parents=True, exist_ok=True)
command = [sys.executable, "tools/analyze_issue137.py", "--output", str(output)]
for record in records:
    command += ["--record", str(record)]
subprocess.run(command, check=True)
PY
```
