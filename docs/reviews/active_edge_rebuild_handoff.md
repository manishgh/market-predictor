# Active Edge Rebuild Handoff

Status: ER1 completed; ER1A in progress; corpus defects diagnosed, fixes not yet built
Last updated: 2026-07-29
Repository: `C:\project\market-predictor`
Remote: `https://github.com/manishgh/market-predictor`
Branch: `r3-lineage`
Last implementation commit: `319dd3e`; last documentation commit: `08a5208`

Read first:

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`
3. This file

Do not infer current state from chat history.

## Objective

Build a correct and complete dataset under a sound technical design, then train.
The goal is predicting the market. Small per-ticker data loss is acceptable and
must not drive the design.

## Scope Constraints Set By The User On 2026-07-29

These supersede the earlier five-year intraday research target.

- No data earlier than 2016 is required for any horizon.
- Intraday: three years of history, with catalysts.
- Swing: seven to ten years from now, with catalysts.
- Some strategies, for example mean reversion, do not require catalysts.
- Exclude a security only when its point-in-time identity cannot be proven.
  Never exclude for thin trading or for delisting.

## Two Conflicts To Resolve Before Building

1. **Swing catalyst coverage is five years, not seven to ten.**
   `data/raw/alpaca_news_20210709_20260708_v1` starts 2021-07-09 and holds
   564,986 rows. Swing price history starts 2019-07-09. A catalyst-bearing swing
   dataset therefore cannot exceed five years today. Options: probe whether the
   provider serves news before 2021-07 and collect back to 2016; or use the
   existing two-profile split, `technical_market` over the full window as the
   baseline and `catalyst_full` over the catalyst window as the only
   promotion-eligible profile; or run a catalyst-free strategy such as mean
   reversion over the full window. Decide before ER2 freezes feature profiles.

   Five years is ample in rows and marginal in independent evidence. ER1
   measured 125 effective ten-session blocks over the current seven-year swing
   corpus; five years yields roughly 90, or about 22 per purged fold, and
   roughly 890 usable sessions against the frozen 1,000-session swing gate. A
   window starting 2021-07 also excludes the 2020 volatility shock entirely, so
   the surviving regime variety is the 2022 decline and the subsequent recovery.

2. **Three years of intraday sits almost exactly on the frozen 750-session floor.**
   2023-07-29 through 2026-07-08 is approximately 738 exchange sessions. The
   frozen gate requires at least 750 causally usable sessions and four purged
   folds retaining at least 60 test sessions each. Either widen the window
   slightly beyond three years or change the gate. Do not silently accept 738.

## What The Three-Year Intraday Window Changes

Most diagnosed intraday defects fall outside it and become irrelevant:

- The `FI` contamination window, 2021-04-27 to 2023-06-06, is entirely outside.
- All three truncated sessions, 2021-10-25, 2022-01-24, and 2022-03-08, are
  entirely outside.
- Of the ER1A collection's 804 sessions, only roughly 236 fall inside.
- The legacy corpus supplies the remaining approximately 501 sessions and
  already carries 04:00-20:00 ET bars.

Consequence for ER1B: extended-session context is needed only for the roughly
236 in-window ER1A sessions, not 804. That is approximately 5,664 request units
rather than 17,688. The ER1B plan and its stopped collection must be regenerated
against the reduced window.

## Verified Defect Inventory

Independently derived at per-(ticker, session) bar-count granularity. Counts
reconcile: 404,194 observed plus 517 missing equals 404,711 planned; 32,033,151
bars matches the collection manifest.

ER1A intraday corpus:

| Defect | Magnitude | Class |
| --- | --- | --- |
| `FI` identity contamination | 111 ticker-sessions, 8,599 bars | wrong data |
| `FI` rename gap | 421 member ticker-sessions, zero bars | missing |
| Truncated sessions | 2022-03-08 and 2021-10-25 median 1 bar; 2022-01-24 median 15 | missing |
| Transport holes | 74 single-session gaps across 13 units | missing |
| Delisting boundary | 22 ticker-sessions, zero of 22 traded | benign |
| Thin-trading absence | 12,101 ticker-sessions below 95%, 10.7x lower volume | benign, never refill |

Interior gap-length distribution is `{1: 74, 421: 1}`. Exactly one price jump
above 3x exists in 32,033,151 bars. Zero zero-volume bars. Zero frozen-price
ticker-sessions.

Legacy 730-day intraday corpus: clean under every detector.

Swing daily corpus: nine ticker-reuse symbols in raw; eight are correctly
filtered by membership interval. `FI` alone escapes into
`swing_technical_decisions` (885,371 rows) as 111 wrong-security rows, 421
fabricated zero-volume rows at a flat 3.15, and 272 correct rows. 421 of the 429
zero-volume rows in that artifact are this one security.

## Root Cause

The point-in-time universe derives historical tickers by taking the current
ticker from `data/universe/sp500_current_20260708.csv` and back-filling it,
breaking the back-fill only where the provider's corporate-actions feed reports
a change. Where that feed is incomplete the back-fill silently becomes a
hindsight assertion. This is a bad derivation, not a bad row. It passed every
hash, authority, and non-overlap check because it is structurally valid and
factually wrong, and it defeats both existing defences: membership-interval
filtering applies a wrong interval faithfully, and security-identity grouping
cannot help because the contaminated rows carry the correct `security_id`.

## The Four Design Fixes To Build

Build these as fail-closed contracts with poison tests. They replace all
per-ticker patching.

1. **Point-in-time symbol derivation.** Never derive a historical ticker from a
   current constituent list. Derive per interval from evidence and fail closed
   where evidence is absent.
2. **Completeness floor.** Segment-aware expected-versus-observed gate per
   (ticker, session): strict for regular session, permissive for extended hours
   where absence is a genuine no-trade observation.
3. **Identity continuity assertion.** Gap-length and price-level continuity
   checks run as a build gate, not as an ad-hoc audit.
4. **Reject provider fabrication.** Zero-volume bars are not observations and
   must never enter a canonical build.

## The Exclusion Rule

Apply universally, never per ticker:

> A security whose point-in-time symbol cannot be proven from evidence is
> excluded from the universe for the affected interval, and the exclusion is
> recorded with its reason and row count.

Never exclude for thin trading; that filters the cross-section by liquidity.
Never exclude for delisting; that is survivorship bias and is why `TWTR`,
`SIVB`, `FRC`, `ATVI`, and `CERN` are deliberately retained.

If the rule excludes materially more than one or two securities, stop and report
before rebuilding anything on top of it.

## Exact Next Steps

1. Resolve the two conflicts above with the user.
2. Build the four gates with poison tests.
3. Re-run the universe build so exclusions fall out by rule; publish a corrected
   universe artifact with provenance.
4. Regenerate the ER1A and ER1B plans. Both fingerprints,
   `96688b25df7abb24d0317649bde87bf29d9497f799d968801b8f5995bb7cb285` and
   `89c91d178ed1095cc33047508c270a817d716e1c594bfe0940064d2de770a250`, are
   invalidated by any universe change.
5. Refill only what the reduced window needs.
6. Materialize regular and extended layers, then ER2.

## Stopped Work

The ER1B extended-session transport was stopped deliberately at 3,712 of 17,688
units. Its output at `data/raw/edge_rebuild_extended_session_context_er1b_20260729`
has no `_authority.json` and is not evidence. It must be discarded rather than
resumed, because the universe correction and the reduced window both change the
plan fingerprint and therefore every `unit_id`. Nothing else is running. The
worktree is clean.

## Deferred, Already Recorded

Section 9A of the plan queues retirement of the deprecated V1 relevance path in
`v3/catalysts.py` for ER4, with the coefficients preserved and the deletion set
verified self-contained.

## Verification Commands

Use the project virtual environment and one process at a time:

```powershell
Set-Location C:\project\market-predictor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check --no-cache .
.\.venv\Scripts\mypy.exe --strict --python-version 3.14 src\market_predictor
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short --branch
```

The local NumPy stubs use syntax newer than the project mypy 3.11 target. The
verified local strict command therefore uses the installed Python 3.14
environment. Do not reinterpret that environment fact as permission to add
Python-3.14-only source syntax.
