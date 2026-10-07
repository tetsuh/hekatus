Merge commit `821c46cd15ecf3b9b764a5a2ce09ae3d75a66115` retains Git's automatic no-conflict merge message and was not amended.

Closes #12

Stage 2 and Stage 3 are tracked in Issue #105.

Prediction: if the torn-read hypothesis is correct, the paired short/long outlier count will be zero in both the Watcher validation and the 500,000-frame timing run.

Current wrap-tracked rerun (harness `b0dec92728b69f87ab78ee1ef991a00a43b528ec`) supersedes the prior `2616f96` timing record because both semaphore barriers and the clock representation changed. ADR-0005 records: `2026-10-06-p150a-issue12-stage1-current-wrap-watcher-100-adr0005.json` and `2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json`. Clock metadata is `get_timestamp_32b`, `32-bit low word, software-extended (wrap-tracked)`; the prior `d429d70` pending-NOC Watcher failure was from the pre-barrier kernel and is not conflated with this result.

- Current Watcher validation: 100/100 produced/consumed, zero overflow/full/cycle error, consumer-empty=101, work min/max=1063/1073 ticks, N=99, timing evidence false. Raw count/SHA256=100/ccb60a5460fae3dee2d53afc6cb317ff4745acbe707f77eba37b5f3a4b836b04; power samples/SHA256=2/0f49331aa9ecbb07f777a53bf534f5ca9127b1859ed8eb7871af790594da4ab4.
- Current 500,000-frame timing run: attempted/produced/consumed/dropped=500000/500000/500000/0, overflow/full=0/0, consumer-empty=500000, work min/max=1063/1087 ticks, N=499999, min/max=1349924/1350068, P50/P99/P99.9/P99.99=1349988/1350033/1350059/1350061 ticks. Raw count/SHA256=500000/19bc5c0f346253ce0855b4ceb36d9760a556b80dc1202de9a9019ac1b8ef5fa6; power samples/SHA256=226/0b4ede33357e5aa52e0795d115de2567f58f157388f34601dca0c8b9bab4bf35.

Prediction outcome: the paired short/long outlier count was zero in both current runs, versus 27 pairs in the prior 2616f96 record; no 6363 phase distribution is present in the current run, and this remains an observation only with no causal claim. Docker was zero before/after both runs; no reset was performed.

Provenance mapping: all current accepted timing numbers belong to `2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json`; the prior `2026-10-06-p150a-issue12-stage1-final-500000-adr0005.json` numbers are historical/superseded, including its 27-pair outlier set.


This PR records the current pinned-v0.75.0 Issue #12 validation and in-cap timing run under ADR-0005. The current executed harness is `b0dec92728b69f87ab78ee1ef991a00a43b528ec`; current records/docs are committed as `d3802c2`. The earlier `2616f96` record remains historical and is superseded by the current-wrap record.

RED: N/A (authorized measurement and board-free post-processing/documentation; no new hardware implementation behavior was introduced).

Validation run:

- Watcher validation, 100 frames, 60-second cap: status `ok`, 100/100 produced/consumed, dropped/overflow/full=`0/0/0`, consumer-empty=`100`, cycle error `none`, work min/max=`1073/1084` ticks, startup=`11954` ticks, N=`99`; timing evidence is explicitly false. Raw timestamps count/SHA256=`100/ecba1f56f0050107d111a25da48a17078c31f50195592f86c99b7037f850d079`; power samples/SHA256=`2/a28fda9b99ea7b7c9642b2e7231711b39fecbf691a28705ef1ac3061d09326f1`.
- In-cap no-Watcher timing run, exactly 500,000 frames, 1 ms interval parameter (`1,350,000` device ticks), 600-second cap: status `ok`, attempted/produced/consumed/dropped=`500000/500000/500000/0`, overflow/full=`0/0`, consumer-empty=`500000`, cycle error `none`, startup=`11949` ticks, work min/max=`1074/1098` ticks, total elapsed=`674998650044` ticks (`499.999000033` seconds). N=`499,999`; min/max=`111,119/2,588,915`; P50=`1,349,988`; P99=`1,350,048`; P99.9=`1,350,050`; P99.99=`1,350,050` ticks. Raw timestamps count/SHA256=`500000/6a7baa6660de45ede2b6f752c7623ae6d25787a8c4613c13a81b2faca4480dec`; power samples/SHA256=`226/6d9b6b7b3c01708732069f1238b001ad4862411f3713a4ee411b144f0488a1c4`.

Both records include board serial/type, firmware versions, KMD, pinned image digest, tt_env release, clean pushed harness identity, observed AICLK, fine-bin histograms, startup/failure diagnostics, work extrema, first-frame exclusion, raw timestamp hashes, and companion power hashes. The frame interval is a parameter, not a real acquisition-rate claim. Docker was zero before and after both runs; no reset or health probe was needed.

Outlier observation only, with no causal claim: 27 adjacent short/long pairs were found. Pair starts by interval-end frame index are `6084,37899,111072,117435,120617,126980,152431,158794,168339,193790,200153,212879,219242,266964,273327,279690,305141,308323,314686,352863,359226,375133,422855,432400,457851,464214,470577`; pair sums range `2699967..2700034` ticks around the `2700000` target. For adjacent-event frame gaps, the complete sanitized quotient/remainder sequence with period `6363` is: `31815=>q5,r0; 73173=>q11,r3180; 6363=>q1,r0; 3182=>q0,r3182; 6363=>q1,r0; 25451=>q3,r6362; 6363=>q1,r0; 9545=>q1,r3182; 25451=>q3,r6362; 6363=>q1,r0; 12726=>q2,r0; 6363=>q1,r0; 47722=>q7,r3181; 6363=>q1,r0; 6363=>q1,r0; 25451=>q3,r6362; 3182=>q0,r3182; 6363=>q1,r0; 38177=>q5,r6362; 6363=>q1,r0; 15907=>q2,r3181; 47722=>q7,r3181; 9545=>q1,r3182; 25451=>q3,r6362; 6363=>q1,r0; 6363=>q1,r0`. Aggregate `(quotient,remainder)` counts are `{(5,0):1,(11,3180):1,(1,0):11,(0,3182):2,(3,6362):4,(1,3182):2,(2,0):1,(7,3181):2,(5,6362):1,(2,3181):1}`. Thus `352863→359226` and `457851→464214→470577` include q=1,r=0 gaps. No single exact period was detected: 6363 is the most common gap (11/26), gcd is 1. Corrected elapsed-time units are seconds in the ADR record; the full per-pair sums and elapsed values are retained there.

The prior 600,000-frame record remains an immutable historical record and is excluded from timing evidence because it used configured 660 seconds, beyond the approved 600-second cap. The README disposition takes precedence over that historical JSON's timing field. Existing JSON/CSV files were not modified.

The root-cause history remains truthful: normal frame-count completion was corrected, semaphore visibility/addressing was corrected with receiving-core semaphores and cache invalidation, and prior Watcher/startup-margin attributions remain incorrect or uncertain rather than timing evidence.

Validation:

- `uv run ruff check .`
- `uv run pytest -q -m 'not slow'` — 1802 passed, 1053 skipped, 45 deselected.
- `git diff --check` and strict JSON/CSV immutability, hash, fine-bin, private-path, seconds-conversion, and periodicity audits passed.
- No hardware, SSH, Docker, reset, or review action occurred after the two authorized runs.
