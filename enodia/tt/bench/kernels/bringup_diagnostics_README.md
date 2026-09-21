# Newton-Schulz bring-up diagnostics

`tools/newton_schulz_bringup.py` is the authoritative stage and configuration mapping. The `bringup_*.cpp` files are isolated diagnostics, not production kernels.

## Stage guide

- **Stages 1–5:** passed on board.
- **Stage 6:** completed on board but failed the `1e-2` numerical tolerance.
- **Stages 41–42:** passed on board.
- **Stages 43–45:** superseded and invalid historical experiments. Stages 43 and 44 completed with numerical failure while using incomplete complex residual arithmetic. Stage 45 timed out and was later found similarly confounded. None is evidence about the corrected algorithm.
- **Stages 46–48:** passed on board with corrected complex residual construction and product consumption.
- **Stage 49:** completed on board but failed numerically despite FP32 destination accumulation.
- **Stages 50–51:** host-validated precision-boundary diagnostics; not run on board.

Use the stage number with the bring-up runner to select one diagnostic process at a time. Results from superseded stages must not be used as correctness or liveness evidence for the corrected algorithm.
