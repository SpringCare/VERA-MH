# Design decision records

This directory contains short records of architectural decisions that need to
survive beyond the pull request where they were made. A record explains why a
decision was necessary, what was chosen, and what consequences were accepted.
It is not a second copy of the current architecture.

## When to add a record

Add a design record when required by the
[architecture escalation rules](../architecture.md#escalate-stop-and-ask),
including changes to a stable interface, or when a decision has consequences
that future maintainers would not be able to recover from the code alone. Do not
create one for routine implementation choices.

## Canonical documentation

- [Architecture](../architecture.md) defines the current normative structure and
  invariants.
- [CLI use cases](../vera-cli-use-cases.md) define user-facing behavior and
  examples.
- Design records preserve context, rationale, alternatives, and consequences.

Link to canonical documentation instead of copying its rules into a design
record. If the architecture changes later, add a new record and supersede the
old one rather than rewriting history.

## Required format

Use a descriptive, kebab-case filename and one decision per file:

```markdown
# Decision title

Status: Proposed | Accepted | Rejected | Superseded
Date: YYYY-MM-DD

## Context

Why a decision is needed.

## Decision

What was chosen, including links to canonical architecture where appropriate.

## Consequences

Compatibility effects, tradeoffs, migration requirements, and known risks.

## Superseded by

Optional. Link to the replacement record when Status is Superseded.
```

## Lifecycle

- A `Proposed` record may change during review.
- An `Accepted` or `Rejected` record is historical. Fix broken links or factual
  errors, but do not rewrite its decision to match later implementation.
- When a decision changes materially, add a new record, set the old record to
  `Superseded`, and link the two records.
- Keep accepted, rejected, and superseded records in this directory; their value
  is the history they preserve.
- Link the relevant record from any pull request that changes a protected stable
  interface.
