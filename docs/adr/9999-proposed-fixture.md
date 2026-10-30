# ADR-9999: Intentional Proposed fixture for Issue #35 CI RED

- Status: **Proposed** — 2026-10-30

## Context

This temporary fixture demonstrates that the ADR status guard rejects a Proposed
record in the pull request head. It is removed by the next commit after the
failing check is recorded; it is not a process or architecture decision.

## Options considered

- Exercise the actual pull-request check with an unsettled record.

## Decision

Use this fixture only as intentional CI RED evidence for Issue #35.

## Consequences

The ADR status check must fail while this file is present.

## Status history

- 2026-10-30: Proposed solely as an intentional CI RED fixture.
