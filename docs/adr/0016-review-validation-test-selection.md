# ADR-0016: Review validation test selection

- Status: **Proposed** — 2026-10-17

## Context

Review validation and CI intentionally use different test selections. Review
validation uses `pytest -q -m "not slow"` from the operator's validation
configuration outside this repository. CI runs the complete suite with
`uv run pytest -q` in `.github/workflows/ci.yml`.

The distinction is more than a runtime preference: it governs which tests may
be omitted during review validation. A broad or accidental use of `slow` could
hide contract, privacy, or boundary coverage. The split is therefore a
process and test-selection contract, and is worthy of an ADR.

## Options considered

- **Run the complete suite for review validation.** Rejected: it gives the
  strongest single invocation but makes the review-time check needlessly
  costly for simulation-backed tests.
- **Keep the filtered review invocation without a written policy.** Rejected:
  the omission boundary would remain implicit, allowing a marker to hide a
  contract, privacy, or boundary regression.
- **Use a cost-based review selection and a complete CI selection, with
  explicit marker boundaries.** Chosen: this preserves a cheap review check
  while making the full-suite build the coverage authority.

## Decision

1. Review validation invokes `pytest -q -m "not slow"` from the operator's
   validation configuration outside this repository.
2. CI invokes `uv run pytest -q` in `.github/workflows/ci.yml`, without a
   marker filter. CI therefore includes every test marked `slow`.
3. `slow` applies only to a costly individual test or to an individual test
   that consumes a heavy fixture. It is never assigned at module scope and is
   never used to omit a whole module.
4. A test primarily asserting contract, privacy, or boundary behavior is
   never marked `slow`. This includes error and invalid-input checks, API and
   ingress checks, type and dtype checks, record-format checks, and
   environment-isolation checks, even when such a test uses an expensive
   fixture.
5. The marker expresses cost-based review selection, not lower importance.
   Tests omitted by review validation remain required in CI and retain their
   normal assertions.
6. Under ADR-0004, the owner's merge of the pull request carrying this ADR is
   its approval and acceptance. The transition to `Accepted` is written in
   that same pull request at the merge boundary.

## Consequences

- Review validation skips only tests explicitly classified as costly, while
  contract, privacy, and boundary guards remain in the review check.
- CI remains the required complete coverage run and exercises every slow test.
- Individual markers can describe expensive numerical simulations, sweeps,
  golden comparisons, PSF checks, and end-to-end image tests without making a
  whole module costly by default.
- A mechanical test guards against reintroducing a module-level `slow` marker;
  it does not attempt to classify test meaning.
- Changing either invocation, the marker scope, or the no-slow boundary is a
  new process decision and requires a later ADR.

## Status history

- 2026-10-17: Proposed in the open pull request carrying Issue #86. Under
  ADR-0004, the owner's merge of that carrying pull request is approval and
  acceptance; the transition to `Accepted` is written in the same pull
  request at the merge boundary.
