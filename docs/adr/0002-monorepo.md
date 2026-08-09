# ADR-0002: Repository structure — one monorepo

- Status: **Proposed** (accepted when the bootstrap PR merges)
- Date: 2026-08-09
- Owner decision: yes

## Context

hekatus consists of three roles (enodia, diaplous, lampas) that are bound
together by shared contracts: the inter-process data-plane records, the
transmit-description schema, and numerical-equivalence acceptance against the
reference implementation. In the current phase these contracts change
frequently. Development is done by one owner with AI coding agents, and the
working unit is an MVP slice that typically cuts across roles
(simulation → beamforming → display).

## Options considered

- **(a) One monorepo** holding all roles and documents.
- **(b) One repository per role** (plus a meta repository for shared
  documents). Mirrors the role separation, but every contract change becomes
  a cross-repository version-pinning exercise while contracts are still
  fluid, and the shared design documents need a fourth home.
- **(c) A core repository with satellite repositories added as they appear.**
  Inherits (b)'s coordination costs and implies satellites that do not exist.

## Decision

Adopt **(a)**: a single repository `hekatus` with role directories
(`enodia/`, `diaplous/`, `lampas/`) and shared `docs/`.

## Consequences

- Contract changes, cross-role MVP slices, and document synchronization stay
  within one PR.
- Versioning is repository-wide (tags); per-component versioning appears only
  after a split.
- **Split triggers** (any of these reopens the decision, in a new ADR):
  1. a component gains independent consumers or an independent release
     cadence;
  2. a team boundary appears (a contributor working on only one role);
  3. the accelerator measurement harness generalizes beyond ultrasound and
     is worth publishing on its own.
- A split extracts the component with history via `git filter-repo`; the
  monorepo phase does not need to anticipate it structurally.
