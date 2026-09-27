# Newton-Schulz bring-up diagnostics

`tools/newton_schulz_bringup.py` is the authoritative stage and configuration mapping. The `bringup_*.cpp` files are isolated diagnostics, not production kernels.

## Stage guide

- **Stages 1–5:** passed on board.
- **Stage 6:** completed on board but failed the `1e-2` numerical tolerance.
- **Stages 41–42:** passed on board.
- **Stages 43–45:** superseded and invalid historical experiments. Stages 43 and 44 completed with numerical failure while using incomplete complex residual arithmetic. Stage 45 timed out and was later found similarly confounded. None is evidence about the corrected algorithm.
- **Stages 46–48:** passed on board with corrected complex residual construction and product consumption.
- **Stage 49:** completed on board but failed numerically despite FP32 destination accumulation.
- **Stage 50:** passed on board at relative error `0.002346657`; isolated BF16×BF16 FP32 destination accumulation into a Float32 CB/output.
- **Stage 51:** completed on board but failed numerically at relative error `3.62880301`; Float32 intermediate reuse as a mixed-format matmul input is unresolved.
- **Stage 52:** completed on board but failed numerically at relative error `3.79775143`; explicit unpacker and packer reconfiguration together did not repair stage 51.
- **Stage 53:** completed on board but failed numerically at relative error `3.75525451`; unpacker reconfiguration alone did not repair stage 51.
- **Stage 54:** completed on board but failed numerically at relative error `4.05933809`; packer reconfiguration alone did not repair stage 51.
- **Stage 55:** completed on board but produced non-finite output (`Infinity` relative error); it switched the second result to a distinct BF16 output CB and used the documented old/new-CB packer reconfiguration overload.
- **Stages 56–57:** stage 56 completed on board with relative error `3.56092668` after correcting SrcB but without packer reconfiguration; stage 57 passed at `0.0032820513` with the corrected SrcB transition plus a distinct BF16 output CB and packer reconfiguration.
- **Stage 58:** completed on board with relative error `3.67779350`; it kept the Float32 output CB and added explicit same-output packer reconfiguration after the corrected SrcB transition.
- **Stage 59:** passed on board at relative error `0.0031062583`; it switches to a distinct Float32 output CB while keeping the output format unchanged, separating output-CB switching from output-format switching.
- **Stage 60:** passed on board at relative error `0.0030487294`; it keeps the distinct Float32 output CB and omits explicit packer reconfiguration.
- **Stage 61:** passed on board at relative error `0.0040098457`; it keeps Float32 X state, intermediates, and output for all eight Newton-Schulz iterations while retaining BF16 R input.
- **Stage 62:** timed out after 180 seconds on board with exit 137 and no numerical result. It keeps BF16 X state for iterations 1–4, converts the fourth update to Float32 state, and runs iterations 5–8 with Float32 state/intermediates/output. FP32 destination accumulation is enabled for the entire stage, so this is a four-plus-four state/storage experiment rather than a state-only comparison. Forced termination required reset #8; the post-reset stage-1 health probe passed at relative error `0.00456437`. The timeout does not isolate which of the new conversion or mixed-format transitions caused the hang. A separate ordinary full-work Watcher run is recorded below.
- **Stage 63 (initial attempt):** board JIT compilation failed before execution with undeclared compute-kernel APIs (`cb_wait_front`, `cb_reserve_back`, tile-register synchronization, `pack_tile`, `cb_push_back`, and `cb_pop_front`), and the host process exited 139. No conversion result was produced, so this is not a runtime conversion failure. The compute source was missing the common compute API include. No reset was required, no container or device-0 user remained, and the post-run stage-1 health probe passed at relative error `0.00456437`; the cumulative reset count remains 8.
- **Stage 63 (corrected source rerun):** JIT compilation succeeded and the kernel closed normally in `0.255 s`, but it produced finite output with relative error `0.9999989867`, failing the `1e-2` threshold. This does not establish a BF16-to-Float32 conversion pass. No reset was required, no container or device-0 user remained, and the cumulative reset count remains 8.
- **Stage 64:** passed on board at relative error `0.0` in `0.302777 s`; it repeats the isolated BF16 tile to Float32 tile copy/repack with `compute_kernel_hw_startup<SrcOrder::Reverse>` before any copy operation. It keeps BF16 input CB 20, distinct Float32 output CB 23, the same deterministic input and widened-output oracle, and no matmul, Newton, or binary arithmetic. Stage 63 used the same conversion path without hardware startup and produced near-zero output (relative error `0.9999989867`), so startup is required for this isolated route. The run closed normally, left no container or device-0 user, and required no reset; cumulative resets remain 8.
- **Stage 65:** completed normally in `0.318909 s` with finite output but failed the `1e-2` threshold at relative error `1.40931547`. It reuses Float32 output CB 16 for the post-conversion matmul, matching the warm-up's output CB.
- **Stage 66:** passed at relative error `0.0003233934` in `0.331232 s`; it writes the post-conversion matmul to distinct Float32 output CB 19.
- **Stages 65–66:** these paired L=32, batch-1, one-core diagnostics isolate conversion followed by using the Float32 state as the next matmul's right/SrcA operand, without Newton or binary arithmetic. Both queue a BF16 `R` tile twice, perform a BF16×BF16 warm-up into Float32 CB 16, drain that warm-up result, signal the writer only after the drain, convert a separate BF16 state from CB 20 to Float32 CB 14, route it through Float32 CB 17, and then perform `R @ X`. The warm-up is a diagnostic control for prior output-CB use, not a Newton step. Inputs and operation order are identical; only the post-conversion output CB differs, with no extra packer reconfiguration. The same-output case failed numerically while the distinct-output case passed, supporting the CB-separation hypothesis for this minimal route. This does not identify the cause of stage 62's timeout, since its full Newton dataflow has additional transitions. Both runs closed normally, left no container or device-0 user, and required no reset; cumulative resets were 8 at that point.

- **Stage 67:** mirrors stage 62's L=32, batch-1, eight-iteration four-BF16/four-Float32 Newton-Schulz path and changes only the Float32-phase `X @ S` product output from CB 16 to a dedicated Float32 CB 13; the first group remains on CB 16. It timed out after 60 seconds with exit 137 and no numerical result. The log ends at device initialization/dispatch telemetry with no stage-specific JIT compilation or result output, matching the stopping point in the original stage-62 timeout log; thus it does not establish whether the changed CB routing executed or explain stage 62's timeout. The residual container was stopped; no device-0 user was present. Reset #9 followed the forced termination without normal device close. The post-reset stage-1 health probe passed at relative error `0.00456437`, with no remaining container or device-0 user; cumulative resets are now 9.
- **Stage 68:** ran once on board on 2026-09-27 as the approved minimum first-residual probe and timed out with status/exit 137 without a numerical JSON result. The Watcher record and required single-reset recovery are documented below. The stop point localizes the approved diagnostic to the first residual path, including BF16 `S` production and drain, but is liveness evidence only; it does not establish a numerical pass or the deferred Stage-62 reconfiguration root cause.
- **Stage 69:** is a waypoint-instrumented copy of Stage 68. It reuses Stage 68's reader, writer, host shape, CB layout, BF16 outputs, and oracle; only the compute source adds custom Watcher markers. Its board result and recovery are documented below; no numerical JSON result was produced.

## Construction-time bisection (zero-work dispatch implemented; board probes completed)

Stages 62 and 67 must not be rerun as full Newton-Schulz programs for this investigation. Their logs stop before stage-specific JIT output, so the board work uses a zero-work construction probe and must not claim numerical or dataflow evidence. The P0-P8 host definitions and the explicitly gated dispatch now live in `tools/newton_schulz_bringup.py` and are covered by host-only tests. Running the runner with `--construction-probe P0` through `P8` remains host-only; adding `--dispatch-construction-probe` is the only path that opens the selected device, allocates the six inputs and two Float32 outputs, builds all descriptors, and invokes `ttnn.generic_op`. That dispatch has been implemented and used for the P0-P8 board results below; those results do not imply numerical or full-dataflow acceptance.

The current host builder constructs the same program shape for all three stages:

- 25 indexed `CBDescriptor` entries, each with four pages, on one `CoreRangeSet` containing one core; the TTNN API has no separate active-index list, so all 25 are retained in the constructed program while `active_cb_indices` remains source/probe metadata;
- no semaphores, the same reader/writer/compute kernel order, and the same `ComputeConfigDescriptor` flags;
- six reader tensor-accessor argument groups and two writer groups; reader runtime arguments contain six buffer addresses plus `start_tile` and `tile_count`, while writer arguments contain two addresses plus those two counters;
- the recorded host configuration keeps compute compile-time arguments `[tiles_per_core]`, which is `[1]` here, and compute runtime arguments are empty; the explicitly gated dispatch overrides only the compute compile-time loop argument to `[0]`;
- two Float32 output tensors and the same batch, core count, tile count, iteration count, and `fp32_dest_acc_en` setting.

The construction differences are narrower than the source-level differences:

- Stage 61 uses input dtypes `(BF16, FP32, BF16, FP32, FP32, FP32)`. Stages 62 and 67 use `(BF16, BF16, BF16, BF16, FP32, FP32)`, so only the initial X real/imaginary tensors change dtype and accessor metadata.
- Stage 61 uses Float32 descriptors at CBs `2–9` and `16–24`; its other descriptors are BF16. Stage 62 moves only CBs `14–15` to Float32 and CBs `20–21` to BF16, giving Float32 descriptors at `2–9`, `14–19`, and `22–24`. The page sizes are coupled to those formats: 4096 bytes for Float32 and 2048 bytes for BF16.
- Stage 67 changes only CB 13 relative to Stage 62, from an unused BF16 descriptor to an active Float32 product CB. The descriptor totals are 344064 bytes for stages 61/62 and 352256 bytes for stage 67; all still declare 25 descriptors. The source-referenced CB index counts are 18, 24, and 25 respectively, with unused indices `{1,10,11,12,13,14,15}`, `{13}`, and `{}`.
- The writer source is shared. Stages 62 and 67 add BF16/Float32 state conversion, extra format reconfiguration, and a larger reader/compute source. Stage 67 additionally routes the Float32-phase products through CB 13. These are kernel-source differences, not host descriptor differences.

The probes are ordered from the least risky construction change to the closest build-only form of the failing stages. Each probe requires a fresh process and an external 60-second wall-clock cap. The host record retains `tiles_per_core=1` and `compute_compile_args=[1]` for comparison with the original stages, while the authorized dispatch uses reader and writer runtime `tile_count=0` plus a documented compute compile override of `[0]`. Thus no reader, writer, or compute iteration dataflow is attempted, including for the real Stage-62/67 sources; the probe records only construction/dispatch completion. It must not fall back to a full hybrid or numerical run.

| Probe | Cumulative addition | Purpose |
|---|---|---|
| P0 | Stage-61 host fields and sources, zero-work dispatch | Establish the minimal construction control. |
| P1 | Change only the two initial X input dtypes to BF16 | Isolate device-buffer allocation, tensor-accessor metadata, and address-vector changes. |
| P2 | Add only the Stage-62 CB format/page changes at 14, 15, 20, and 21 | Isolate the mixed CB descriptor combination while retaining the Stage-61 sources. |
| P3a | Use a delegated no-op source triplet with the Stage-61 active CB index set (18 indices) | Control for source compilation with no matmul, copy, or binary operation. |
| P3b | Keep the P3a no-op source body and change only its active CB index set to Stage 62's 24 indices and placements | Isolate active CB index encoding from the descriptor format change. |
| P4 | Replace only the reader with the real Stage-62 reader; keep the no-op compute source | Determine whether the BF16/Float32 reader source is the first additional boundary. |
| P5 | Replace the compute source with the real Stage-62 compute source | Test the exact Stage-62 source pair without executing its loops. |
| P6 | Change only CB 13 from the Stage-62 descriptor to the Stage-67 Float32 descriptor | Test Stage 67's descriptor-only delta with Stage-62 sources. |
| P7 | Replace only the reader with the real Stage-67 reader | Isolate the CB-13 reader routing and active-index addition. |
| P8 | Replace the compute source with the real Stage-67 compute source | Test the closest build-only construction to Stage 67 without full hybrid dataflow. |

P3a/P3b are a paired index-only branch rather than a numerical stage. Stop at the first failing probe. On a timeout or abnormal exit, stop the residual process/container, perform at most one device reset, rerun the known-good Stage 1 health probe, verify that no device user or container remains, and stop the investigation. Do not continue to another probe after recovery in the same allocation. A clean numerical result is not an acceptance criterion for these probes; only construction/dispatch completion and the exact stop point are evidence.

No separate core-range or semaphore probe is justified by the current host comparison: all three stages use the same one-core range and `semaphores=[]`. Likewise, the compute argument vector shape is unchanged; only input accessor values and addresses can change as a consequence of the two X dtype changes. The host definitions keep the one-core range, empty semaphore list, and compute argument shape explicit, while the dispatch builder records the zero-work runtime/compile override separately. The dispatch flag is explicit and diagnostic-only; it is not a substitute for a hybrid run. It has been implemented and used for the P0-P8 board results below.

Use the stage number with the bring-up runner to select one diagnostic process at a time. Results from superseded stages must not be used as correctness or liveness evidence for the corrected algorithm.

## Construction probe board results

On 2026-09-26, each probe ran in a fresh container process with `--device-id 0`, the explicit zero-work dispatch flag, and an external 60-second timeout. The runner selected device 0 for every dispatch; no kernel was dispatched to device 1. The recorded host shape remained `compute_compile_args=[1]`, while every device dispatch used `reader_tile_count=0`, `writer_tile_count=0`, and the documented compute compile override `[0]`. No numerical output was downloaded or accepted.

| Probe | Result | Dispatch time (s) | Log final position |
|---|---|---:|---|
| P0 | pass (`status=dispatched`) | 0.304614 | `Cluster destructor completed (cluster.cpp:781)` |
| P1 | pass (`status=dispatched`) | 0.300803 | `Cluster destructor completed (cluster.cpp:781)` |
| P2 | pass (`status=dispatched`) | 0.288168 | `Cluster destructor completed (cluster.cpp:781)` |
| P3a | pass (`status=dispatched`) | 0.229854 | `Cluster destructor completed (cluster.cpp:781)` |
| P3b | pass (`status=dispatched`) | 0.234593 | `Cluster destructor completed (cluster.cpp:781)` |
| P4 | pass (`status=dispatched`) | 0.269468 | `Cluster destructor completed (cluster.cpp:781)` |
| P5 | pass (`status=dispatched`) | 0.283236 | `Cluster destructor completed (cluster.cpp:781)` |
| P6 | pass (`status=dispatched`) | 0.273749 | `Cluster destructor completed (cluster.cpp:781)` |
| P7 | pass (`status=dispatched`) | 0.274189 | `Cluster destructor completed (cluster.cpp:781)` |
| P8 | pass (`status=dispatched`) | 0.274219 | `Cluster destructor completed (cluster.cpp:781)` |

P0 itself exited normally and returned the expected dispatched JSON record. The first orchestration wrapper misquoted its JSON `grep` expressions and falsely classified that successful process as a failure. It then performed the required cleanup, an unnecessary reset recorded as reset #10, and a Stage 1 health probe. The reset command exited 0; the health probe passed with relative error `0.0045643728`, and no container or device user remained. This operational wrapper error is not probe or kernel failure evidence. The corrected wrapper ran P1 through P8; none timed out or errored, so no further reset was needed. The final board state was zero running containers and no device-0 user.

## Host-only JIT/cache audit for stages 62 and 67

This audit was performed after the historical timeouts and did not access the board.

- The historical stage-62 and stage-67 commands used `docker run --rm` with the repository and hugepage mounts only. They did not mount a persistent kernel-cache directory or a Docker volume. The runtime reported `cache_path=/root/.cache/ttnn` and `tmp_dir=/tmp/ttnn`; the Metalium JIT cache is a separate path, normally `/root/.cache/tt-metal-cache/` when `HOME=/root`.
- A current host inspection found no retained stage-62 or stage-67 result logs, no `/root/.cache/tt-metal-cache`, `/tmp/tt-metal-cache`, or `/var/cache/tt-metal-cache` tree, no source-specific reader/compute entries, and no Docker volumes. Because the historical containers were removed, this is retention evidence only: missing host artifacts do not prove that the binaries were never generated inside those containers.
- The available historical log tails stop during device initialization/dispatch telemetry. They contain no stage-specific reader or compute compile command, JIT cache statistics, result JSON, or numerical result. `BuildKernels` lines about pre-compiled firmware describe firmware setup, not proof that the user reader and compute kernels were or were not compiled.
- The runner writes its JSON record after dispatch and synchronization. Redirected Python output is buffered, so a timeout can remove the process before the record is flushed. A cache hit can also suppress per-source compile output. Therefore the apparent pre-JIT log boundary is compatible with buffering or a cache hit and is not proof of a JIT-side failure.

The historical timeout records remain inconclusive about their own JIT boundary. The completed probe below is separate evidence from fresh build-only runs: it shows that the stage-specific reader and compute sources can be generated and linked in the pinned image, but it does not retroactively identify what happened during either historical full-hybrid run.

### Approved build-only board probe

The previously proposed build-only path was approved and executed for stages 62 and 67. Each run used device 0 in a fresh process/container, an external 60-second cap, the pinned image, and full compile-time tile count `[1]`. The controls were `TT_METAL_CACHE=/jit-cache`, `TT_METAL_FORCE_JIT_COMPILE=1`, `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`, and `TT_METAL_KERNELS_EARLY_RETURN=1`; runner stdout was unbuffered. Early return compiled the full-size kernels and returned before device-kernel dataflow. The probe intentionally omitted output download and numerical acceptance.

1. Each stage used the real stage-specific reader and compute sources and the shared writer.
2. Pre/post cache manifests and hashes were recorded for the stage-specific reader and compute artifacts, along with the compile-command log.
3. The external cap and fresh-process cleanup protocol were retained; both runs closed normally, left no residual container or device-0 user, and required no reset.

## Build-only JIT/cache probe board results

These results are compile/build and early-return evidence only. A runner `status=pass` means the build-only process exited successfully; it is not a numerical pass. No numerical output was downloaded, and there was no full-hybrid dataflow/liveness acceptance.

### Stage 62 build-only

- Runner JSON: `status=pass`, `success=true`, `exit_code=0`, `exit_state=success`, `runner elapsed_s=0.5411973880000005`, `output_download=false`, `numerical_acceptance=false`.
- Sources: reader `bringup_ns_four_plus_four_reader.cpp`; compute `bringup_ns_four_plus_four_compute.cpp`; shared writer `bringup_writer.cpp`.
- Cache was empty before and had 185 manifest/hash entries after. Reader manifest matched with 10 artifacts; compute manifest matched with 30 artifacts.
- Canonical final ELF artifacts and SHA-256:

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| reader `ncrisc.elf` | 337872 | `bb99defc4dcb725b15ff15b81b55d063065417a698a88945648fffe13872d117` |
| compute `trisc0.elf` | 728628 | `a96a0170d0e3a4e12e5f740a59408a11cf1c0087b8f9b031eea697467d042753` |
| compute `trisc1.elf` | 623684 | `da74c9ef00843ebf05f9260088952a9f007aee32f08d902890313791bed64283` |
| compute `trisc2.elf` | 963964 | `2579c67085cb81bb0d62a841551b6927407bb6e6eb3633ced90d2b0ac53079d6` |

- Compile-command log in unbuffered runner stdout: 21 `g++` compile-command lines and 20 `g++` link-command lines; the reader, compute, and shared writer stages each had one compile and one link line. Final log position: `Cluster destructor completed (cluster.cpp:781)`.
- Cleanup: no residual container or device-0 user remained, and no reset was required.

### Stage 67 build-only

- Runner JSON: `status=pass`, `success=true`, `exit_code=0`, `exit_state=success`, `runner elapsed_s=0.5125056580000091`, `output_download=false`, `numerical_acceptance=false`.
- Sources: reader `bringup_ns_four_plus_four_distinct_output_reader.cpp`; compute `bringup_ns_four_plus_four_distinct_output_compute.cpp`; shared writer `bringup_writer.cpp`.
- Cache was empty before and had 185 manifest/hash entries after. Reader manifest matched with 10 artifacts; compute manifest matched with 30 artifacts.
- Canonical final ELF artifacts and SHA-256:

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| reader `ncrisc.elf` | 343836 | `fe27cd2bdddc74216bb0bf9eee63114a92054747712cd52a579aeb55ea75e976` |
| compute `trisc0.elf` | 729512 | `a319cf5a9cb4d62ae50d880849cdee26288fb62bf8e1e6a97d49336b8f5e2c48` |
| compute `trisc1.elf` | 616284 | `58c60d5ed5bdeefe831321cdf4f242ccc58c6c37fb1641dfe7f5af5777dffeaa` |
| compute `trisc2.elf` | 978096 | `b29b6e3eb8a4358db292b5a80c1a785e522887e4c27e4e9ba9de49bed1b2d38f` |

- Compile-command log in unbuffered runner stdout: 21 `g++` compile-command lines and 20 `g++` link-command lines; the distinct-output stage reader, compute, and shared writer each had one compile and one link line. Final log position: `Cluster destructor completed (cluster.cpp:781)`.
- Cleanup: no residual container or device-0 user remained, and no reset was required.

The fresh build-only runs show that the stage-specific reader/compute ELF artifacts were generated and linked, shifting the historical ambiguity away from inability to compile those sources. This does not claim numerical success, kernel dataflow success, or resolution of the historical Stage-62/67 full-hybrid timeout. The early-return path does not prove normal dataflow liveness; these runs provide compile/build and early-return evidence only, with no numerical output download or full-hybrid dataflow/liveness acceptance.

## Stage 62 ordinary full-work Watcher run

This run is separate from the historical 180-second timeout and the build-only JIT probes above. It was exactly one ordinary full-work Stage 62 execution with `TT_METAL_WATCHER=1` at a 1-second interval, `TT_METAL_WATCHER_DUMP_ALL=1`, and `TT_METAL_WATCHER_NOINLINE=1`. An external 60-second cap timed it out. The pre-stop Watcher log was copied before the process was stopped; the wrapper/container then ended with exit status 137. No numerical JSON result was produced. No DPRINT was enabled because Watcher supplied the requested evidence.

- Dump #1 at `1.097 s` showed an idle device. From Dump #2 at `2.159 s` through Dump #55, the active device-0 worker core `(0,0)` / virtual `(1,2)` remained unchanged.
- The kernel map from `kernel_names.txt-before-stop` is kernel id 6 = `bringup_writer.cpp` (BRISC), id 5 = `bringup_ns_four_plus_four_reader.cpp` (NCRISC), and id 7 = `bringup_ns_four_plus_four_compute.cpp` (TRISC0/TRISC1/TRISC2).
- In the documented BRISC, NCRISC, TRISC0, TRISC1, TRISC2 order, the final status was `CWFW,CWFW,UABD,MWDD,K`, with `rmsg:D1G|BNT`, `smsg:GGGG`, and `k_ids: 6|5|7|7|7`.
- The raw CB diagnostics on that core were exactly `cb[1](rcv 8!=ack 4) cb[6](rcv 1!=ack 0) cb[7](rcv 1!=ack 0) cb[8](rcv 1!=ack 0)`. There were no Watcher assert, NOC-sanitize, CB-sanitize, or hardware-fault messages.

`CWFW` is the Watcher `cb_wait_front` waypoint. BRISC is the writer, so its `CWFW` is the first output wait, CB23 (with CB24 following), as shown by `bringup_writer.cpp`. NCRISC is the reader. Its source has already streamed the second BF16 X operand into CB1 and identity/zero into CB6/CB7 before entering `stream_state_complex` for the residual S source. The first wait in that function is the BF16 S-real source, CB10; CB11 is the following S-imag wait. The CB10 attribution is source-derived inference from the reader control flow, not a CB number printed beside the Watcher waypoint. The counter names are: CB1 is the BF16 X CB, CB6 is identity, CB7 is zero, and CB8 is RX-real.

The compute RISCs do not show `CWFW`: TRISC0 `UABD`, TRISC1 `MWDD`, and TRISC2 `K` are last-waypoint evidence only, not a direct compute CB wait. The compute side did not advance to publish the BF16 residual S outputs (CB10/11) or the next state, while the exact compute-side instruction boundary remains unresolved from this Watcher sample.

Evidence fingerprints for the pre-stop record are the Watcher log SHA-256 `8aa6b728f88138fb0e82be7738e53b4a71432f0b1ec78577108654a6ebc164ac`, 29,813 lines, and 2,012,205 bytes. Its last completed sample was Dump #55 at `58.500 s`. The accompanying `kernel_elf_paths.txt-before-stop` records the BRISC, NCRISC, and TRISC artifacts named by the kernel map above.

## Recovery after the Stage 62 Watcher timeout

The timeout caused cumulative reset #11; the preceding cumulative count was #10. `tt-smi -r all` returned success. The known-good Stage 1 recovery passed with relative error `0.004564372822642326` and elapsed `0.2917247400000633 s`. Post-recovery checks found no running containers and no device-0 users.

This is a runtime/dataflow liveness observation after successful build-only JIT probes, not a numerical acceptance result. The exact compute-side root cause remains unresolved.

## Host-only Stage 62 CB accounting and format-transition audit

This section records the manual source audit performed after the ordinary Stage 62 Watcher run. It used `bringup_ns_four_plus_four_reader.cpp`, `bringup_ns_four_plus_four_compute.cpp`, `bringup_writer.cpp`, and the Stage 62 CB descriptors in `tools/newton_schulz_bringup.py`. No board, container, reset, or device execution was used for this audit.

The notation below describes one tile across the eight iterations:

- `M(A,B) x4`: compute waits for and consumes four tiles from A and B, producing four tiles in CB16.
- `P`: reader waits for and consumes the four CB16 product tiles, pushing one tile to each of CB2--CB5.
- `RX`: CB2/CB3 produce CB8 and CB4/CB5 produce CB9.
- `S`: CB6/CB8 produce CB10 and CB7/CB9 produce CB11.

### Per-iteration ordering

The table records logical kernel order. Reader and compute execute concurrently, so it is not a global wall-clock ordering.

| Iteration | Reader sequence | Compute sequence | State produced |
|---|---|---|---|
| 0 | CB0 x4, CB1 x4, `P`, CB6/CB7 x1, CB1 x4, wait/pop CB10/CB11 and push CB12 x4, `P` | `M(CB0,CB1) x4`, `RX`, `S`, `M(CB1,CB12) x4`, then state construction | CB20/CB21 |
| 1 | CB0 x4, CB20/CB21 to CB1 x4, `P`, CB6/CB7 x1, CB20/CB21 to CB1 x4 and pop, wait/pop S CB10/CB11 and push CB12 x4, `P` | `M(CB0,CB1) x4`, `RX`, `S`, `M(CB1,CB12) x4`, then state construction | CB20/CB21 |
| 2 | Same state-source sequence as iteration 1 | Same compute sequence as iteration 1 | CB20/CB21 |
| 3 | Same state-source sequence as iteration 1 | Same compute sequence as iteration 1 | CB20/CB21 |
| 4 | CB0 x4, CB14/CB15 to CB17 x4, `P`, CB6/CB7 x1, CB14/CB15 to CB17 x4 and pop, wait/pop S CB18/CB19 and push CB22 x4, `P` | Convert CB20/CB21 to CB14/CB15, `M(CB0,CB17) x4`, `RX`, S to CB18/CB19, `M(CB17,CB22) x4`, then state construction | CB14/CB15 |
| 5 | Same Float32 state-source sequence as iteration 4 | Same Float32 compute sequence as iteration 4 | CB14/CB15 |
| 6 | Same Float32 state-source sequence as iteration 4 | Same Float32 compute sequence as iteration 4 | CB14/CB15 |
| 7 | Same Float32 state-source sequence as iteration 4, with final output routing | Same Float32 compute sequence, with final state to CB23/CB24 | CB23/CB24 |

`stream_state_complex()` waits twice for each state source: the first copy does not pop, while the second copy pops. This explains the deliberately larger wait count than push count for CB14/CB15 and the two readers of CB20/CB21.

### Complete push/wait/pop ledger

All Stage 62 descriptors have four pages. Counts below are for one tile and all eight iterations; `R`, `C`, and `W` mean reader, compute, and writer.

| CB | Format | Pushes | `wait_front` | `pop_front` | Capacity observation |
|---:|---|---|---|---|---|
| 0 | BF16 | R: 32 | C: 32 | C: 32 | Balanced |
| 1 | BF16 | R: 32 | C: 32 | C: 32 | Four tiles remain after the second BF16 group is queued |
| 2 | FP32 | R: 16 | C: 16 | C: 16 | Balanced |
| 3 | FP32 | R: 16 | C: 16 | C: 16 | Balanced |
| 4 | FP32 | R: 16 | C: 16 | C: 16 | Balanced |
| 5 | FP32 | R: 16 | C: 16 | C: 16 | Balanced |
| 6 | FP32 | R: 8 | C: 8 | C: 8 | Identity input |
| 7 | FP32 | R: 8 | C: 8 | C: 8 | Zero input |
| 8 | FP32 | C: 8 | C: 8 | C: 8 | RX real |
| 9 | FP32 | C: 8 | C: 8 | C: 8 | RX imaginary |
| 10 | BF16 | C: 4 | R: 4 | R: 4 | BF16 S real |
| 11 | BF16 | C: 4 | R: 4 | R: 4 | BF16 S imaginary |
| 12 | BF16 | R: 16 | C: 16 | C: 16 | BF16 second-group operand |
| 13 | — | 0 | 0 | 0 | Unused by Stage 62 |
| 14 | FP32 | C: 4 | R: 8 | R: 4 | Two reader waits per Float32 iteration |
| 15 | FP32 | C: 4 | R: 8 | R: 4 | Two reader waits per Float32 iteration |
| 16 | FP32 | C: 64 | R: 64 | R: 64 | Product routing drains every tile |
| 17 | FP32 | R: 32 | C: 32 | C: 32 | Float32 X operand |
| 18 | FP32 | C: 4 | R: 4 | R: 4 | Float32 S real |
| 19 | FP32 | C: 4 | R: 4 | R: 4 | Float32 S imaginary |
| 20 | BF16 | C: 4 | R: 6 + C: 1 | R: 3 + C: 1 | Reader state reuse plus iteration-4 conversion |
| 21 | BF16 | C: 4 | R: 6 + C: 1 | R: 3 + C: 1 | Reader state reuse plus iteration-4 conversion |
| 22 | FP32 | R: 16 | C: 16 | C: 16 | Float32 second-group operand |
| 23 | FP32 | C: 1 | W: 1 | W: 1 | Final real output |
| 24 | FP32 | C: 1 | W: 1 | W: 1 | Final imaginary output |

The ledger has no unmatched push/pop count and no producer that must reserve a fifth page. CB1 reaches exactly four resident pages, but the Watcher reported `CWFW`, not a producer `CRBW`, and the reader had already completed that four-tile push. CB1 fullness is therefore downstream backpressure after compute stopped before publishing CB10/CB11, not a source-level circular wait. Before the reader waits for CB10/CB11, it has supplied every reader-side input needed for the first residual; CB12 is intentionally delayed until S exists.

### Unpack and pack reconfiguration audit

The relevant API contracts distinguish full operation initialization from format-only reconfiguration:

- `compute_kernel_hw_startup<SrcOrder::Reverse>` is called once at kernel entry with R, BF16 X, and the FP32 product CB. The Reverse mapping is correct for matmul: `in0` feeds SrcB and `in1` feeds SrcA.
- `binary_op_init_common(left, right, output)` performs full unpacker setup for the two inputs and full packer setup, including `pack_init(output)` and destination initialization. Therefore the first transition from FP32 CB8/CB9 to BF16 CB10/CB11 is not missing a packer call in the source. CB10/CB11 are also distinct from CB8/CB9.
- `matmul_block_init()` configures matmul unpack/math state but does not configure the packer output. After the BF16 S binary operations, `switch_to_bfloat16_second_group()` calls only `matmul_block_init()`; it does not restore unpacker formats from the preceding FP32 binary inputs to BF16 CB1/CB12, and it does not restore the packer from BF16 CB11 to FP32 product CB16.
- The same missing pair occurs after BF16 state construction: `switch_to_bfloat16_first_group()` calls only `matmul_block_init()` before the next BF16 matmul, while the preceding binary operation used FP32 product inputs and left the packer targeting BF16 state output.
- `convert_state_to_float32()` does call `pack_reconfig_data_format(CB20, CB14)` before the first Float32 copy, so the output-side pack transition is explicit. However, `copy_tile_init(CB20)` and `copy_tile_init(CB21)` do not reconfigure unpack data formats by contract. The source does not use the format-aware copy initializer or an equivalent unpack reconfiguration before reading the BF16 state after the preceding FP32 binary operation.
- At the conversion boundary, `reconfig_data_format_srca(CB12, CB17)` correctly reflects that the new Float32 X state is matmul `SrcA`. The helper does not reconfigure `SrcB` from the preceding state/binary configuration back to BF16 R CB0; that transition remains incomplete in the source audit.
- Within the Float32 phase, the source-B transitions between BF16 R and Float32 X are explicitly handled, and product, S, and state output CBs remain Float32. No additional output-format transition is missing inside that phase.

Thus the exact FP32 CB8/9 to BF16 CB10/11 binary operation has full operation initialization, so source inspection does not prove that this particular `pack_tile` is missing a reconfiguration. The broader Stage 62 format-transition contract is nevertheless incomplete immediately after that operation and at the BF16-to-Float32 copy boundary. Those missing transitions are higher-priority candidates for the next implementation review and must not be inferred away from the successful isolated conversion probe.

### Approved minimum hardware probe for the next session

The next probe is intentionally narrower than the full hybrid:

1. Use L=32, batch 1, one core, one tile, the Stage 62 four-page CB layout, and the same FP32 destination-accumulation setting.
2. Preserve the reader's first-iteration sequence through the second four-tile CB1 push, CB6/CB7 supply, and the wait for BF16 S.
3. Run compute only through the first complex matmul group, RX construction, and S production into CB10/CB11. Omit the second matmul, state output, iteration-4 conversion, and all later iterations.
4. Have the reader wait for CB10/CB11 and drain them through a matching-format diagnostic output path so the writer adds no unrelated format transition.
5. Use a fresh process, an external 60-second cap, and Watcher logging.

A timeout would localize the failure to the first residual path, including the FP32-to-BF16 binary output. A pass would justify a one-iteration extension that adds CB12 and the second matmul, with the missing unpack/pack transitions tested explicitly before any full hybrid run. Stage 68 now contains this isolated implementation and its host-only coverage; no board execution was performed in this session.

## Stage 69 host-only waypoint map

Stage 69's compute source is `bringup_ns_first_residual_waypoint_compute.cpp`. The source
sequence is unchanged from Stage 68; the markers below are the only added device-side
operations. Each marker is four characters and is distinct from the built-in Watcher
waypoints. The marker is posted immediately before the named operation.

| Marker | Boundary |
|---|---|
| `M69I` | `matmul_block_init` |
| `M69R`, `M69X` | `matmul_one` waits for `cb_r`, `cb_x_bfloat16` |
| `M69M` | `matmul_block` |
| `M69P`, `M69B` | matmul `pack_tile`, `cb_push_back` |
| `S69L`, `S69R` | `subtract_one` left and right waits |
| `S69I`, `S69T` | subtract binary and tile-operation initialization |
| `S69P`, `S69B` | subtract `pack_tile`, `cb_push_back` |
| `A69L`, `A69R` | `add_one` left and right waits |
| `A69I`, `A69T` | add binary and tile-operation initialization |
| `A69P`, `A69B` | add `pack_tile`, `cb_push_back` |

The map above was recorded before the board probe; Stage 69's board result and recovery are documented below.

## Stage 69 board probe and recovery

On 2026-09-27, Stage 69 ran exactly once in one fresh process/container with the pinned repository image, `--device-id 0`, only `/dev/tenstorrent/0` exposed, and an external 60-second timeout. Watcher was enabled with `TT_METAL_WATCHER=1`, `TT_METAL_WATCHER_DUMP_ALL=1`, and `TT_METAL_WATCHER_NOINLINE=1`. The run timed out with status/exit 137 and produced no Stage-69 numerical JSON result. The residual container was stopped; final cleanup found no running containers and no device users.

The Watcher evidence file contained 21,330 lines and 1,440,366 bytes, with SHA-256 `dcd23d34cf020359cead474f6de5082a2ec2a7901b8d8c9dfed58946abd6ebe6`. The final record was Dump #77, completed at `79.523 s`. The kernel map was id 5 = `bringup_ns_first_residual_reader.cpp` (NCRISC), id 6 = `bringup_ns_first_residual_writer.cpp` (BRISC), and id 7 = `bringup_ns_first_residual_waypoint_compute.cpp` (TRISC0/TRISC1/TRISC2).

On the final worker core `(0,0)`, virtual `(1,2)`, the status in BRISC, NCRISC, TRISC0, TRISC1, TRISC2 order was `CWFW,CWFW,S69R,S69I,S69I`, with `rmsg D1G|BNT`, `smsg GGGG`, and counters `cb[1](rcv 8!=ack 4)`, `cb[6](rcv 1!=ack 0)`, `cb[7](rcv 1!=ack 0)`, and `cb[8](rcv 1!=ack 0)`. No Watcher assert, NoC-sanitize, CB-sanitize, or hardware-fault message was observed.

Using the host-only marker table above, BRISC and NCRISC remained at `CWFW`. TRISC0 last reached `S69R`, immediately before the `subtract_one` right-input `cb_wait_front`, the first residual's CB8/RX-real wait. TRISC1 and TRISC2 last reached `S69I`, immediately before `binary_op_init_common` for that first subtract, whose inputs are CB6 identity and CB8 RX-real and whose output is CB10 BF16. WAYPOINT reports the last marker reached: `S69I` is therefore an init-boundary stop-point, not proof of an instruction inside the initializer. The run directly narrows the compute stop-point to the first residual subtraction boundary; it does not prove the deferred reconfiguration root cause or numerical success.

Recovery used exactly one reset targeted only `/dev/tenstorrent/0`, and the reset exited 0; this is cumulative reset #13 after Stage 68's documented #12. A fresh Stage 1 health probe on device 0 passed with relative error `0.004564372822642326` and elapsed `0.29091544399943814 s`. Final no-container and no-device-user checks passed.

## Stage 68 board probe and recovery

On 2026-09-27, Stage 68 ran once in a fresh container/process with the repository-pinned image, `--device-id 0`, only the device-0 node exposed, and an external 60-second timeout. Watcher was enabled with `TT_METAL_WATCHER=1`, `TT_METAL_WATCHER_DUMP_ALL=1`, and `TT_METAL_WATCHER_NOINLINE=1`. The run timed out with status/exit 137 and produced no Stage-68 numerical JSON result. The residual container was stopped; cleanup left no running containers and no device users.

The Watcher evidence file contained 24,654 lines and 1,664,175 bytes, with SHA-256 `ecd114627b74657683af2923a7eae213f28cf746ccaded2204f4a9b01812e19`. The final record was Dump #89, completed at `91.815 s`. The kernel map was id 5 = `bringup_ns_first_residual_reader.cpp` (NCRISC), id 6 = `bringup_ns_first_residual_writer.cpp` (BRISC), and id 7 = `bringup_ns_first_residual_compute.cpp` (TRISC0/TRISC1/TRISC2).

The final worker core `(0,0)`, virtual `(1,2)`, reported status `CWFW,CWFW,UABD,MWDD,K`, `rmsg D1G|BNT`, and `smsg GGGG`, with `cb[1](rcv 8!=ack 4)`, `cb[6](rcv 1!=ack 0)`, `cb[7](rcv 1!=ack 0)`, and `cb[8](rcv 1!=ack 0)`. No Watcher assert, NoC-sanitize, CB-sanitize, or hardware-fault message was observed. This is liveness/stop-point evidence only, not a numerical pass or a definitive compute root cause. In the approved scope, the timeout localizes the diagnostic to the first residual path, including BF16 `S` production and drain; it does not establish the deferred Stage-62 reconfiguration root cause.

Recovery used exactly one reset, targeted only `/dev/tenstorrent/0`, and exited 0; this is cumulative reset #12 after the preceding recovery's #11. A fresh Stage 1 health probe on device 0 passed with relative error `0.004564372822642326` and elapsed `0.33350358500001676 s`. Final checks again found no running containers and no device users.
