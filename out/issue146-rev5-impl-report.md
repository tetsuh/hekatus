# Issue #146 revision 5 board-free implementation report

## Scope

Implemented the owner-approved revision 5 follow-up without changing the
immutable plan, product behavior, Issue #137/Q2 APIs, trace schema, registered
thresholds, or hardware configuration. No board, Docker, telemetry utility, or
network run was performed.

## Immutable plan identity

- Revision: `5`
- Raw plan SHA-256: `19a7c45de299b8e15b2d2e69ffa8a100091701ff2b8e32ca41980b465aade41b`
- Canonical plan SHA-256: `5fc8d7a19533e10117e1b862603e5cb32ca37549b03194367b3396637dbff3ef`
- Plan cadence: `measurement_scope.trace_interval_seconds = 1.0`

## Commits

- `05b02b5` — `fix(bench): implement Issue #146 revision 5 semantics (#146)`
- `9b5e7fa` — `docs(readme): document Issue #146 revision 5 output (#146)`

## Changed files

- `tools/issue146_contract.py`
- `tools/issue146.py`
- `tools/issue146_device.py`
- `tools/issue146_analysis.py`
- `tools/analyze_issue146.py`
- `tests/test_issue146_analyzer.py`
- `tests/test_issue146_analyzer_criteria.py`
- `tests/test_issue146_device.py`
- `tests/test_issue146_runner.py`
- `tests/test_issue146_runner_extra.py`
- `tests/test_issue146_revision5.py`
- `docs/measurements/README.md`
- `out/issue146-rev5-impl-report.md`

## RED and GREEN evidence

RED before the revision 5 implementation was completed:

- `uv run pytest -q tests/test_issue146_revision5.py` — 2 failed, 3 passed; the signed negative R5 smoke case and the non-gating smoke diagnostic seam were not implemented.

GREEN:

- `uv run pytest -q tests/test_issue146*.py` — 136 passed.
- `uv run ruff check .` — passed.
- `git diff --check` — passed before commit.

The existing device import/help regression remains covered by
`tests/test_issue146_device.py`; it imports the device entrypoint with SciPy
blocked and verifies neither analyzer module is imported.

## Analyzer CLI and new report fields

The analyzer CLI remains:

```bash
uv run python tools/analyze_issue146.py --smoke \
  --plan docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json \
  --record out/issue146/run-issue146-smoke-c018.json \
  --output out/issue146/smoke-analysis.json
uv run python tools/analyze_issue146.py \
  --plan docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json \
  --record out/issue146/run-issue146-long-c018-1.json \
  --output out/issue146/analysis.json
```

New fields include:

- `trace_cadence.planned_interval_seconds`, `host_sampler_interval_seconds`,
  `wrapper_companion_interval_seconds`, source/status, and explicit-test-override
  provenance.
- `trace.sampling_interval_seconds` and `trace.sampling_interval_source`, plus
  `run.trace_interval_seconds` and matching provenance fields.
- `R5-aiclk-tracks-arbiter` and smoke R5 signed
  `signed_delta_min_mhz`/`signed_delta_max_mhz` fields, with absolute residual
  retained diagnostically.
- Smoke acceptance with exactly four `gate: true` items and one
  `regulated_load_diagnostics_[10,40)` item with minimum current, below-150-A
  count, board-power mean/sample SD, and `power_r1`.
- Exploratory `current_dips` for C081/C108 over `[60,180)` or `[10,40)` with
  median, strict `< 0.8 * median` threshold, fraction/count, per-dip current,
  board/ASIC power, timestamp/offset, nearest raw group boundaries, containing
  group/gap, and sampler-interval limitation text. C018 is not applicable.

## Residual risks

- No hardware smoke rerun or long run was performed; owner approval and parent
  full-gate review remain required.
- Current-dip boundary correspondence is limited by discrete sampler captures
  and is descriptive only; no clustering cutoff or causal claim is emitted.
