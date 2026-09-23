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

Use the stage number with the bring-up runner to select one diagnostic process at a time. Results from superseded stages must not be used as correctness or liveness evidence for the corrected algorithm.
