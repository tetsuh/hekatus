# Issue #63 RED diagnostics

These are board-free, intentional RED tests. They preserve two failures that
must remain visible during acceptance review without making the normal suite
fail. The tests do not alter the production diagonal-pair placement or FLOP
accounting; each wrong variant is local to its diagnostic.

The diagnostic file is `tests/test_issue63_red_diagnostics.py` and is marked
`red_diagnostic`. It is skipped unless `HEKATUS_RUN_INTENTIONAL_RED=1`.

## Intentional failures

- `test_red_wrong_diagonal_tile_placement_fails_l0_equivalence` places the
  second L=16 matrix in an upper-right quadrant while the production unpack
  contract expects the lower-right diagonal quadrant. Its fixed-iteration
  result must fail comparison with
  `enodia/tt/bench/newton_schulz_reference.py`.
- `test_red_padded_l16_kernel_flops_diverge_from_logical_shape_flops` claims
  all padded 32x32 tile work as useful work. Its claimed FLOPs must fail the
  logical `shapes.total_flops` denominator check, including the fixed
  Newton-Schulz iteration multiplier.

## Captured RED command and result

Command:

```text
HEKATUS_RUN_INTENTIONAL_RED=1 .venv/bin/python -m pytest -q -m red_diagnostic tests/test_issue63_red_diagnostics.py
```

Expected result (intentional failure, not a production regression):

```text
FAILED tests/test_issue63_red_diagnostics.py::test_red_wrong_diagonal_tile_placement_fails_l0_equivalence
FAILED tests/test_issue63_red_diagnostics.py::test_red_padded_l16_kernel_flops_diverge_from_logical_shape_flops
2 failed
```

Normal host validation excludes these diagnostics and remains green:

```text
.venv/bin/python -m pytest -q -m 'not red_diagnostic'
```
