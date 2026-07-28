# Primary V2 Failure Attribution and Setup Viability Plan

Date: 2026-07-28
Status: implemented; both authoritative audits complete

## Purpose

Both primary V2 strategies were correctly rejected. This stage explains whether
the failure comes from the setup population, execution costs, market regime, or
a broad predeclared setup cohort. It does not train, promote, or retain a model.

The audit uses only rows that were validation observations in the authoritative
V2 split. Training rows cannot contribute to viability evidence.

Five-session swing labels overlap on adjacent decision dates. Swing evidence is
therefore split into five fixed calendar phases and every phase is evaluated
separately. A swing cohort must pass all five phases in both validation scopes.
Pooling adjacent five-session labels as independent observations is prohibited.
Intraday paths remain grouped by session because each path ends within its
decision session.

## Authoritative Inputs

- The complete primary V2 run must pass recursive run, candidate, and artifact
  hash verification.
- Its recorded primary V2 implementation hashes must match the current frozen
  V2 contracts, model, and experiment implementation.
- The exact KS3 or KS4 source bundle must pass its existing source authority and
  artifact verification.
- Baseline prediction row IDs are joined one-to-one to exact source rows. A
  missing, duplicated, or extra identity fails the audit.

## Frozen Cohorts

Only one-dimensional cohorts are allowed:

- Both strategies: overall, market regime, sector, fixed volatility bucket.
- Intraday only: market-cap bucket, liquidity bucket, and fixed ET time-of-day
  segment.
- Swing market-cap and liquidity cohorts are not fabricated because those
  fields are not part of the authoritative swing source.

Volatility cutoffs and time boundaries are fixed in
`configs/primary_v2_failure_attribution.toml`. No data-derived quantile boundary,
ticker cohort, multidimensional intersection, or post-result cohort may be
introduced into this audit.

## Evidence

For each validation scope and cohort, publish:

- rows and independent sessions;
- average gross return, stamped round-trip cost, and net return;
- average SPY excess return;
- session-block bootstrap 95% intervals for net and SPY excess;
- win rate, profit factor, and session-level maximum drawdown;
- average maximum favorable and adverse excursion;
- intraday target-first, stop-first, timeout, and resolution statistics.

Swing records additionally identify the non-overlapping phase. Replicated
viability uses the worst phase for return, confidence, profit factor, and sample
counts, and the largest phase drawdown.

Gross minus net is the authoritative stamped cost. The audit fails if the
calculated cost is negative, non-finite, inconsistent by row, or below the
frozen 10 bps minimum on average.

## Replicated Viability

The same `(dimension, value)` must independently pass both walk-forward and
unseen-ticker scopes. Swing requires every one of five non-overlapping phases
inside each scope; intraday has one session-contained phase. Each phase requires:

- at least 200 rows and 60 sessions;
- positive average net return and positive lower 95% confidence bound;
- positive average SPY excess and positive lower 95% confidence bound;
- profit factor at least 1.05;
- maximum drawdown no more than 20%;
- average stamped round-trip cost at least 10 bps.

A cohort that passes only one scope is non-replicated. A replicated pass merely
authorizes a separately frozen V3 hypothesis; it never promotes a V2 model.

## Publication

One immutable output directory contains the request, summary, cohort evidence,
replicated viability table, manifest, and complete authority record. Every
artifact is hash-bound. Reusing a complete directory verifies and returns it;
an existing directory for a different request fails closed.

The process is serialized and must remain below 4 GiB resident memory.

## Completed Evidence

The completed interpretation and immutable artifact identities are recorded in
`docs/model_cards/primary_v2_failure_attribution_20260728.md`.
