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
- **Stage 62:** timed out after 180 seconds on board with exit 137 and no numerical result. It keeps BF16 X state for iterations 1–4, converts the fourth update to Float32 state, and runs iterations 5–8 with Float32 state/intermediates/output. FP32 destination accumulation is enabled for the entire stage, so this is a four-plus-four state/storage experiment rather than a state-only comparison. Forced termination required reset #8; the post-reset stage-1 health probe passed at relative error `0.00456437`. The timeout does not isolate which of the new conversion or mixed-format transitions caused the hang.
- **Stage 63 (initial attempt):** board JIT compilation failed before execution with undeclared compute-kernel APIs (`cb_wait_front`, `cb_reserve_back`, tile-register synchronization, `pack_tile`, `cb_push_back`, and `cb_pop_front`), and the host process exited 139. No conversion result was produced, so this is not a runtime conversion failure. The compute source was missing the common compute API include. No reset was required, no container or device-0 user remained, and the post-run stage-1 health probe passed at relative error `0.00456437`; the cumulative reset count remains 8.
- **Stage 63 (corrected source rerun):** JIT compilation succeeded and the kernel closed normally in `0.255 s`, but it produced finite output with relative error `0.9999989867`, failing the `1e-2` threshold. This does not establish a BF16-to-Float32 conversion pass. No reset was required, no container or device-0 user remained, and the cumulative reset count remains 8.
- **Stage 64:** passed on board at relative error `0.0` in `0.302777 s`; it repeats the isolated BF16 tile to Float32 tile copy/repack with `compute_kernel_hw_startup<SrcOrder::Reverse>` before any copy operation. It keeps BF16 input CB 20, distinct Float32 output CB 23, the same deterministic input and widened-output oracle, and no matmul, Newton, or binary arithmetic. Stage 63 used the same conversion path without hardware startup and produced near-zero output (relative error `0.9999989867`), so startup is required for this isolated route. The run closed normally, left no container or device-0 user, and required no reset; cumulative resets remain 8.
- **Stage 65:** completed normally in `0.318909 s` with finite output but failed the `1e-2` threshold at relative error `1.40931547`. It reuses Float32 output CB 16 for the post-conversion matmul, matching the warm-up's output CB.
- **Stage 66:** passed at relative error `0.0003233934` in `0.331232 s`; it writes the post-conversion matmul to distinct Float32 output CB 19.
- **Stages 65–66:** these paired L=32, batch-1, one-core diagnostics isolate conversion followed by using the Float32 state as the next matmul's right/SrcA operand, without Newton or binary arithmetic. Both queue a BF16 `R` tile twice, perform a BF16×BF16 warm-up into Float32 CB 16, drain that warm-up result, signal the writer only after the drain, convert a separate BF16 state from CB 20 to Float32 CB 14, route it through Float32 CB 17, and then perform `R @ X`. The warm-up is a diagnostic control for prior output-CB use, not a Newton step. Inputs and operation order are identical; only the post-conversion output CB differs, with no extra packer reconfiguration. The same-output case failed numerically while the distinct-output case passed, supporting the CB-separation hypothesis for this minimal route. This does not identify the cause of stage 62's timeout, since its full Newton dataflow has additional transitions. Both runs closed normally, left no container or device-0 user, and required no reset; cumulative resets were 8 at that point.

- **Stage 67:** mirrors stage 62's L=32, batch-1, eight-iteration four-BF16/four-Float32 Newton-Schulz path and changes only the Float32-phase `X @ S` product output from CB 16 to a dedicated Float32 CB 13; the first group remains on CB 16. It timed out after 60 seconds with exit 137 and no numerical result. The log ends at device initialization/dispatch telemetry with no stage-specific JIT compilation or result output, matching the stopping point in the original stage-62 timeout log; thus it does not establish whether the changed CB routing executed or explain stage 62's timeout. The residual container was stopped; no device-0 user was present. Reset #9 followed the forced termination without normal device close. The post-reset stage-1 health probe passed at relative error `0.00456437`, with no remaining container or device-0 user; cumulative resets are now 9.

## Construction-time bisection (zero-work dispatch implemented; board probes unrun)

Stages 62 and 67 must not be rerun as full Newton-Schulz programs for this investigation. Their logs stop before stage-specific JIT output, so the next board work must use a zero-work construction probe and must not claim numerical or dataflow evidence. The P0-P8 host definitions and the explicitly gated dispatch now live in `tools/newton_schulz_bringup.py` and are covered by host-only tests. Running the runner with `--construction-probe P0` through `P8` remains host-only; adding `--dispatch-construction-probe` is the only path that opens the selected device, allocates the six inputs and two Float32 outputs, builds all descriptors, and invokes `ttnn.generic_op`. That dispatch has been implemented but remains unrun; no board probe or numerical result is implied.

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

No separate core-range or semaphore probe is justified by the current host comparison: all three stages use the same one-core range and `semaphores=[]`. Likewise, the compute argument vector shape is unchanged; only input accessor values and addresses can change as a consequence of the two X dtype changes. The host definitions keep the one-core range, empty semaphore list, and compute argument shape explicit, while the dispatch builder records the zero-work runtime/compile override separately. The dispatch flag is explicit and diagnostic-only; it is not a substitute for a hybrid run. It has been implemented but has not been used on a board.

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
