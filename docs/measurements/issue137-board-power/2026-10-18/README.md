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
thermal state: pre-run temperatures range from 60.2 to 73.8 degC, with
73.2 degC in the final instance.
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

## Exploratory (not verdict-bearing)

This post-hoc supplement does not modify the plan, prediction, original analysis,
or **H-OTHER** verdict. It uses only the existing trace bytes; no new run was
performed. [exploratory-sampling.json](exploratory-sampling.json) is a separate
derived artifact, not an addition to the immutable original manifest.

Reproduce from the repository root with
`uv run python tools/explore_issue137_sampling.py`. The script writes JSON only
to stdout, hashes its source records/traces against the original manifest, and
never calls the verdict analyzer or a hardware utility.

### Selection and comparison with the initial inspection

Sample points satisfy `60 <= (timestamp_monotonic_ns - workload_start_monotonic_ns)
/ 1e9 < 120`. There is no held pre-window boundary point, padding, or
interpolation. Statistics below are arithmetic sample statistics, not the
preregistered duration-weighted estimator. SD uses `n-1`; naive SE is `SD/sqrt(n)`.
The JSON records zero-based CSV row indices and offsets for exact comparison.

Strict selection yields 26 points for runs 1, 6, and 8 and 27 for the others,
not 27 in every run. Run 5 has arithmetic mean 292.444444 W versus the unchanged
duration-weighted verdict mean 290.824848 W. Run 7 busy current is 196.631579 A
(about 196.6 A rather than the initial approximately 196.9 A). No boundary was
moved to obtain the initially quoted counts or values.

| Run | Configuration | n | Board mean W | Sample SD W | Naive SE W | Min W | Max W | AICLK mean MHz | AICLK SD MHz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | C018 | 26 | 254.269 | 11.047 | 2.167 | 226 | 264 | 1350.000 | 0.000 |
| 2 | C108 | 27 | 302.037 | 44.835 | 8.629 | 208 | 352 | 1099.333 | 4.731 |
| 3 | C054 | 27 | 300.259 | 29.439 | 5.666 | 227 | 326 | 1309.111 | 3.055 |
| 4 | C081 | 27 | 303.037 | 35.514 | 6.835 | 219 | 337 | 1194.037 | 3.524 |
| 5 | C081 | 27 | 292.444 | 40.792 | 7.850 | 213 | 339 | 1191.000 | 3.328 |
| 6 | C054 | 26 | 296.038 | 29.225 | 5.732 | 228 | 322 | 1305.308 | 2.589 |
| 7 | C108 | 27 | 298.185 | 50.337 | 9.687 | 205 | 344 | 1095.519 | 4.210 |
| 8 | C018 | 26 | 256.154 | 10.067 | 1.974 | 230 | 271 | 1350.000 | 0.000 |

Busy is the post-hoc proxy `current_a >= 150 A`, not a measured device
utilization or a demonstrated launch-alignment state. Low-current points around
100–140 A and a high-current cluster around 190–205 A are consistent with a
gap/busy mixture; intermediate currents also occur, particularly in C054/C081.
Attributing low points to launch gaps remains a hypothesis. C018 has no points
above the proxy threshold despite running its approved workload.

| Run | Busy n / all n | Busy fraction | Busy board mean W | Busy current mean A | Busy ASIC power mean W |
|---|---:|---:|---:|---:|---:|
| 1 | 0 / 26 | 0.000000 | N/A | N/A | N/A |
| 2 | 19 / 27 | 0.703704 | 298.368 | 199.632 | 151.474 |
| 3 | 22 / 27 | 0.814815 | 308.545 | 178.045 | 147.227 |
| 4 | 18 / 27 | 0.666667 | 319.111 | 186.333 | 146.556 |
| 5 | 24 / 27 | 0.888889 | 292.833 | 190.750 | 149.708 |
| 6 | 18 / 26 | 0.692308 | 302.944 | 181.722 | 150.000 |
| 7 | 19 / 27 | 0.703704 | 293.316 | 196.632 | 149.474 |
| 8 | 0 / 26 | 0.000000 | N/A | N/A | N/A |

### Interpretation boundaries

- In the six C054/C081/C108 runs, sampled board power spans 205–352 W,
  sample SD spans 29.225–50.337 W, and naive SE spans 5.666–9.687 W.
  In the four verdict-bearing high runs, SE spans 6.835–9.687 W: the
  preregistered 300 W +/-2% band (half-width 6 W) is narrower than each of
  these naive errors. The mean-pinning rule therefore has an inferential
  **design defect**, not evidence of a defective measurement. Even if a real
  300 W limiter produces a long-run mean of 300 W, a short sampled mean can
  fall below 294 W and fail this rule.
- Under an illustrative IID/approximately normal model with the observed
  high-run SDs, a true mean of 300 W gives approximately 19–27% probability
  of an arithmetic sample mean below 294 W. This is not an empirically
  established failure probability: serial correlation, telemetry update phase,
  and the gap/busy mixture are uncalibrated. `SD/sqrt(n)` is not a validated
  uncertainty estimate for these traces or the duration-weighted estimator.
  Run 5 is about 0.96 naive SE below 300 W using its arithmetic mean; the
  verdict mean is about 1.17 of that same noise scale below 300 W. The latter
  mixes estimators and is only a scale comparison, not a formal z-score.
- The equal-run mean of the four original duration-weighted board-power means
  is **297.555710 W** (about 297.6 W), consistent with a 300 W limiting
  interpretation, not proof of it. The corresponding equal-run arithmetic
  sample-mean average is 298.925926 W. Neither pooling operation repairs or
  replaces the registered requirement that every high run be pinned.
- Non-C018 busy fractions are 0.666667–0.888889. C108 busy current means are
  **199.631579 A** (run 2) and **196.631579 A** (run 7), near the 200 A TDC
  limit. C081 busy means are 186.333333 and 190.750000 A, below that band.
  Current limiting therefore cannot be excluded for C108, while C081 has
  lower conditional current. Conditioning on current raises the conditional
  mean by construction; this does not establish causal TDC throttling or
  support the preregistered H-TDC label.
- AICLK is stable within runs: sample SD is 2.589–4.731 MHz for non-C018
  runs and zero for C018. Stability does not identify the limiting mechanism.
- All of this is **exploratory, not verdict-bearing**. Classification remains
  **H-OTHER**. Any follow-up must be preregistered in a separate Issue before
  new execution; this supplement authorizes no new run.
