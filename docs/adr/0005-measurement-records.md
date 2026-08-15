# ADR-0005: Measurements are records, kept like decisions

- Status: **Accepted** — 2026-08-16
- Date: 2026-08-16
- Owner decision: yes

## Context

This project decides its design parameters by measurement rather than by
argument, and its central claim — that the whole processing chain fits on one
card with room to spare — rests on a single measured number. The first such
measurement has now been taken, and it moved that claim substantially.

A number of that weight needs the same treatment a decision gets. The failure
mode is already visible in this repository's short history: a figure quoted
without the machine behind it, another whose harness version could not be
named, a summary that drifted from the run it summarized. Each is the same
defect the ADR lifecycle exists to prevent, one domain over.

The measurement directory arrived carrying rules — never rewritten, superseded
rather than corrected, provenance in every file — written into its README as
if they were housekeeping. They are not housekeeping; they govern how evidence
is treated, which is a process decision, and process decisions take a record.

## Options considered

- **Leave the rules in the README.** Cheapest, and how they were first
  written. But a rule with no record is one nobody can find a reason for
  later, and this repository has already had to reconstruct the reasoning
  behind conventions that lived only in the file they governed.
- **Treat results as ordinary build output**, regenerated and overwritten.
  Simple until two runs disagree, at which point the earlier one — the
  evidence for whatever was claimed at the time — no longer exists.
- **Keep them as records with the invariant ADRs already use.**

## Decision

1. **A measurement lands as data**, under `docs/measurements/`, with the
   environment that produced it inside the file: board type and serial,
   firmware, kernel driver, toolchain image by digest, and the revision of
   the harness that computed it.
2. **A landed result is not rewritten.** A measurement that proves wrong, or
   is retaken on better hardware or a corrected harness, is **superseded by a
   later record that says so** — the same invariant ADR-0004 sets for
   decisions, for the same reason: the claim made at the time was made on the
   evidence of the time, and deleting that evidence makes the claim
   unreadable.
3. **A companion trace inherits provenance** from the result file sharing its
   filename stem. It carries no environment block of its own, because two
   copies of the same metadata are two things that can disagree.
4. **A figure quoted in a document names the record it came from.** A number
   in prose without that reference is a claim, not evidence.

## Consequences

- The measurements directory grows monotonically. That is the intent: its
  size is the project's evidence, and old records stay readable.
- Retaking a measurement costs a new file rather than an edit, which is
  slightly more work and exactly the friction that keeps the old one.
- A result whose harness cannot be named cannot be landed under this rule.
  The first measurement had to be retaken for that reason, after review found
  it recorded the board but not the code that computed the numbers.
- Nothing here governs how a measurement is *made*; the harness and its
  shape catalogue answer that, and their reasoning lives with them.

## Status history

- 2026-08-16: Accepted in pull request #36, which landed the first
  measurement and the rules it arrived with.
