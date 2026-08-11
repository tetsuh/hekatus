# ADR-0003: Commit header rules — width, scope, and merge commits

- Status: **Proposed** (accepted when this PR merges)
- Amends: ADR-0001 (development process)
- Date: 2026-08-10
- Owner decision: yes

## Context

ADR-0001 adopted Conventional Commits with a header limit of 72 characters,
inherited from the sibling projects, and showed the header template as
`<type>(<scope>): <summary> (#<issue>)` without saying whether the scope
could be omitted.

Two problems surfaced in use.

**The limit was tighter than it looked.** The required scope and issue
reference consume about eighteen characters, so 72 left barely fifty for the
summary. Two commits in one session exceeded it — one by five characters,
one by eight — and one of those cost a history rewrite to correct. Meanwhile
this repository already declares a line width of 100 for code
(`[tool.ruff] line-length`), so the rule asked contributors to hold two
numbers for one concept.

**The scope question was genuinely ambiguous.** A commit written as
`ci: add lint, test, and demo checks` both did and did not comply, depending
on how the template was read — which is exactly how it passed review and
then failed a later one.

**Merge commits were never addressed.** ADR-0001 made normal merge the
default, so GitHub generates integration commits titled
`Merge pull request #NN from owner/branch`. Those carry no type and no
scope. A universal scope requirement would put every merge commit on `main`
in violation of the rule, including the five already there — a rule broken
by its own repository from the day it is written is not a rule.

## Options considered

On width:

- **72**, the inherited convention. Recognizable, and matches where some web
  views truncate a subject; but two numbers for one concept, and squeezed by
  this project's own decorations.
- **100**, matching the code width already set here.
- **120**. Enough that a header stops being a summary and becomes a
  sentence, which loses the discipline that keeps one commit to one idea.
- **72 counting only the summary**, excluding the issue reference. Buys the
  same relief, at the cost of an exception to remember.

On merge commits:

- **Require a scoped Conventional subject on merge commits too.** Uniform,
  but it discards GitHub's generated subject — which carries the PR number
  and source branch — and adds manual work to every merge.
- **Exempt platform-generated integration commits.**
- **Reverse the normal-merge policy.** This would trade away the
  step-level history that ADR-0001 chose normal merge to preserve, to solve
  a naming question.

## Decision

1. The header limit is **100 characters**, the same width this repository
   sets for code. Shorter is still better and the workflow says so.
2. **Scope is required** on authored commits made after this ADR is accepted,
   resolving the ambiguity. At 100 characters there is no pressure to drop it,
   and it is what makes a history scannable by area.
3. **Platform-generated integration commits are exempt** from the header
   convention. They are not authored changes; they are the structural record
   of an integration, and GitHub's generated subject already names the pull
   request, which is their traceability. The owner may replace that subject
   with a Conventional one when the default is unhelpful, but is never
   required to.

## Consequences

- One width applies to code and commit headers alike.
- Subjects beyond roughly 72 characters appear truncated in some web views,
  with the full text an expander away. This is accepted; the body carries
  the detail in any case.
- `git log --first-parent main` remains the pull-request-level view of
  history. Platform-generated integration commits in that view are exempt;
  the PR number governs each integration record.
- The scope rule is prospective. Earlier authored commits are not judged
  retroactively and remain unchanged; rewriting `main` solely to add scopes
  would create more risk than it removes.
- The width rule needs no exception: no subject anywhere on `main` exceeds
  100 characters.
- This ADR amends ADR-0001 rather than superseding it; the rest of that
  decision stands unchanged.

## Status history

- 2026-08-10: Proposed, after a review found the missing ADR and the
  unaddressed merge-commit case.
- 2026-08-11: Clarified that the scope rule is prospective after review found
  multiple unscoped authored commits in the pre-adoption history.
