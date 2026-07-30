# Active Edge Rebuild Handoff

Status: data acquisition complete for swing; intraday complete for 298 tradable
symbols with ~651 more in collection. ER3 setup admission is the next gate.
Last updated: 2026-07-30
Repository: `C:\project\market-predictor`
Remote: `https://github.com/manishgh/market-predictor`
Branch: `r3-lineage`
Last implementation commit: `9b0c5ce`

Read first:

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`, especially sections 8A, 8B, 8C
3. This file

Do not infer current state from chat history.

## Objective

Build a correct dataset under a sound design, prove the deterministic setups
earn money, and only then train. The goal is a predictive engine, not per-symbol
data completeness. Losing under 5% of symbols is not worth escalating.

## Data Held, Measured From Disk On 2026-07-30

| | Swing | Intraday |
| --- | --- | --- |
| Universe | 654 tickers, 627 securities, verified point-in-time | 572 symbols with five-minute bars |
| Bars | 1,084,622 daily | 31,630,145 regular plus 6,079,043 extended |
| Coverage | 2019-07-09 to 2026-07-08, 7.00 years | 2023-04-10 to 2026-07-08, 3.24 years, 814 sessions |
| News | 714,126 rows over the full 7 years | same corpus, fully covers the window |

Artifacts:

- Verified universe:
  `data/canonical/swing_memberships_verified_20190709_20260708_v2.parquet`
- Intraday per-symbol stores:
  `data/canonical/edge_rebuild_intraday_5m_20260730/{regular,extended}/5m/`
- Swing daily bars:
  `data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3/bars/`
- News: `data/raw/alpaca_news_20190709_20210708_v1` and
  `data/raw/alpaca_news_20210709_20260708_v1`

## The Intraday Universe Gap

The 572 intraday symbols are S&P 500 members because that is what the corpus was
built from. Against the frozen tradability screen:

- 298 of the 572 pass and are usable now.
- 274 fail on volume or price. They stay for swing and are not intraday-tradable.
- ~651 further candidates, mostly non-index, have no bars yet.

A collection for those ~651 is in progress: daily bars, the point-in-time
screen, then news. Estimated about ninety minutes unattended. The intraday
universe is deliberately not index-restricted and the contract refuses an
index-restricted one.

Note that only 3.8% of eligible stock-sessions clear relative volume 2.0, a
median of ten symbols per session. An unconditional population is therefore
roughly 96% symbols that were not moving, which is the most likely reason the V2
intraday setup showed negative average gross return before costs.

## Gates Now Enforced In Code

Each was validated against real data, not only unit tests.

1. **Corpus integrity** (`edge_rebuild/corpus_integrity.py`) — completeness
   judged against each symbol's own history rather than a global floor,
   whole-session truncation, identity continuity, and provider-fabricated bars.
   Isolated single-symbol holes are recorded but do not block; clustering does.
   Run blind against 31,220,235 bars it reproduced every known defect and found
   fifty more.
2. **Membership identity** (`edge_rebuild/universe_identity.py`) — a symbol
   claim must be supported by bar evidence. Applied to 659 intervals it excluded
   three securities, 0.48%, and independently rediscovered two exclusions that
   had previously been hand-coded.
3. **Setup economics** (`edge_rebuild/setup_economics.py`) — the ER3 admission
   gate. Worst-phase aggregation, session-block bootstrap bounds, leave-one-out
   concentration, frozen cost stress.
4. **Strategy contract** (`edge_rebuild/strategy_contract.py`) — refuses random
   cross-validation, raw news counts, a widened experiment budget, an
   index-restricted intraday universe, a sub-1.0 ATR stop, and a daily ATR on a
   thirty-minute hold.

## Frozen Numbers And Where They Came From

Published methods, cited in plan section 8B: triple-barrier labels, purged
k-fold with embargo, event-based sampling. Meta-labeling is deliberately
deferred until one strategy passes admission alone.

- Intraday stop 1.5 ATR, target 2.0 ATR, ATR(14) on five-minute bars. The
  earlier 0.75 stop was below the entire standard range and would have been hit
  by ordinary noise rather than by the thesis failing.
- Swing exit is timeout-only at the tenth session close. Daily bars cannot show
  when an intraday stop was touched.
- Intraday sentiment half-life 90 minutes, evidence-backed. The swing half-life
  of 36 hours is marked provisional: the sentiment literature describes an
  intraday effect, and what persists over ten sessions is post-announcement
  drift, a different mechanism.
- Raw news counts are prohibited as estimator features. Provider coverage grew
  from roughly 75,000 to 113,000 rows per year across the sample, so a raw count
  encodes collection history rather than market behaviour.

## Exact Next Steps

1. ER3 swing setup admission. A build of `swing_setups.py` and its economics
   gate run is in progress; the working tree carries
   `src/market_predictor/edge_rebuild/swing_setups.py` and
   `tests/test_swing_setups.py` uncommitted.
2. ER3 intraday setup admission, once the ~651 collection completes.
3. Only if a setup passes: ER5 training. Deterministic comparator, regularized
   logistic, gradient boosting, grouped learning-to-rank. No deep learning — the
   binding constraint is independent samples, roughly 125 ten-session blocks,
   not model capacity.

**A reproducible rejection is a valid outcome.** V2 died because models were
trained on populations with no edge. Nothing may be tuned to make a gate pass.

## Do Not

- hold any broker credential; this repository emits predictions and does not own
  execution, orders, positions, or sizing;
- use a current screener value in a historical decision;
- exclude symbols for thin trading or delisting, only for unprovable identity;
- merge extended-hours bars into regular-session indicator inputs;
- weaken a frozen gate to make a build or an evaluation pass.

## Verification Commands

Check real exit codes; never pipe these through `tail`, which masks them.

```powershell
Set-Location C:\project\market-predictor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check --no-cache .
.\.venv\Scripts\mypy.exe --strict --python-version 3.14 src\market_predictor
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short --branch
```

The local NumPy stubs use syntax newer than the project mypy 3.11 target, so the
verified strict command uses the installed Python 3.14 environment. That is an
environment fact and not permission to add Python-3.14-only source syntax.
