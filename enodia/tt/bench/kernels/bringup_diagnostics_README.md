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

Use the stage number with the bring-up runner to select one diagnostic process at a time. Results from superseded stages must not be used as correctness or liveness evidence for the corrected algorithm.
