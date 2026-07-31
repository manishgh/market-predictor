# Active Edge Rebuild Handoff

Status: swing rejected and being rebuilt as a ranking strategy. Labels and
cross-sectional scaling are built. Feature builder and model are not.
Last updated: 2026-07-31
Repository: `C:\project\market-predictor`
Remote: `https://github.com/manishgh/market-predictor`
Branch: `r3-lineage`
Last commit: `7f018f5`

Read in this order:

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`, sections 8A, 8B, 8C
3. This file
4. `configs/edge_rebuild_strategy_contract.toml` — every threshold lives here

Do not infer state from chat history.

## How To Work Here

The user is a strong systems thinker and is not a machine-learning specialist.
Name the problem before the technique. Do not use jargon without unpacking it,
and do not use metaphors like "load-bearing" or "blast radius" — say what would
actually break. This was asked for repeatedly.

Do not investigate individual tickers. Record exclusions with a reason and move
on. Escalate only if the universe loses more than 5% of symbols.

Do not invent methodology. Use published methods, cite them, and name them so an
implementation can be checked against its definition. Sources are in plan
section 8B.

Update this handoff after every successful step, not only at checkpoints.

Never pipe `pytest` through `tail` or `head`. It masks the exit code, and a
failing test has already been committed that way. Redirect to a file and echo
`$?`.

The shell working directory drifts back to `C:\project\trading_flow`. Run
`cd /c/project/market-predictor` at the start of every command.

The heavy-job lease is exclusive. Running the test suite while a collection
holds it produces exit code 75 on CLI tests, which looks like failure and is not.

## Data Held, Measured From Disk

| | Swing | Intraday |
| --- | --- | --- |
| Universe | 627 securities, verified point-in-time, delisted included | 533 operating companies, volume-screened, deliberately not index-restricted |
| Bars | 1,084,622 daily, 2019-07-09 to 2026-07-08, 7.00 years | 37.7M five-minute, 2023-04-10 to 2026-07-08, 814 sessions |
| Selection | n/a | 11,340 in-play stock-sessions, median 13 per session |
| News | 714,126 rows over the full 7 years | 165,142 rows, median 204 articles per company |

Key artifacts:

- `data/canonical/swing_memberships_verified_20190709_20260708_v2.parquet`
- `data/canonical/edge_rebuild_intraday_5m_20260730/{regular,extended}/5m/`
- `data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3/bars/`
- `data/research/intraday_universe_selection_20230410_20260708_v2/`
- `data/raw/edge_rebuild_selected_session_5m_20260731` — 790/790 units,
  875,425 rows, 532 symbols, zero failures, authority present

## What Failed, And Why It Matters

The first swing strategy was rejected. It earned +0.724% gross and +0.524% net
per ten-day trade — a real edge that survives costs — but SPY excess was
-0.180%, losing to simply holding the index in four of seven years.

**The rejection was measured on a flawed test and does not mean the signal is
dead.** The population applied a ten-per-day cap ranked by *dollar volume*, so
the tested portfolio was roughly ten arbitrary qualifying S&P stocks per day. An
unranked basket of index constituents approximately reproduces the index, so
measuring its excess return mostly measured costs. It could not have passed.

Two consequences drive everything below:

1. The strategy must **rank** stocks and hold the top ones, not filter them and
   pick arbitrarily. Standard practice for cross-sectional strategies is
   learning-to-rank; the repo's own earlier V3 work already had a grouped ranker.
2. Features must be expressed **relative to other stocks at the same moment**,
   or a model learns "buy when the indicator is high", which fires on everything
   at once and reproduces the index.

## Built And Verified

`edge_rebuild/labeling.py` — two labels on every row, 14 tests.

- Barrier label: from the entry price, did the position reach its target, hit
  its stop, or survive to expiry. Entry is the next session's open so no part of
  the decision bar prices the fill. A bar whose range spans both barriers
  resolves to the **stop**, because the bar proves both prices traded but not in
  what order. A horizon running past available data is left **unresolved**, not
  labelled a timeout, which would invent an observation.
- Rank label: top fifth, bottom fifth, middle, computed inside one session and
  inside one sector. A cross-section below 50 rows yields no labels.
- The contract refuses either alone. Barrier-only reproduces the failure above.
  Rank-only assumes a position is held to expiry however far it moves against it.

`edge_rebuild/cross_sectional.py` — feature scaling, 11 tests.

- Z-score against the same timestamp's cross-section, tails winsorised first so
  one stock halving cannot dominate the mean and flatten every other score.
- Rank within the cross-section, centred at zero, immune to outliers.
- Sector-relative z-score, without which a sector-wide rally reads as stock
  selection.
- Every group is one timestamp. A test asserts that adding a later session
  leaves earlier scores bit-identical; if it ever fails, future data is leaking.

## Contract, As Frozen

Names are `swing` and `intraday`. No version suffixes — nothing is in
production, so there is no promoted artifact a rename could invalidate.

Swing: enter next session open; exit at target, stop, or the tenth session
close, whichever first; 3.0 / 1.5 daily ATR(14); same-bar ties resolve stop
first; hourly features over daily context; 25 positions; 20% sector cap;
sector-neutral scoring; 20 bps round trip.

Intraday: volume bars built from one-minute input, ~78 per session sized off
trailing median session volume; rolling features reset overnight; 2.0 / 1.5
five-minute ATR(14); 10 bps round trip; opening / midday / late segments.

Intraday universe: not index-restricted; average volume at least 1M shares over
20 sessions; price 8 to 500 dollars; relative volume at least 2.0 measured from
prior sessions only; at most 30 candidates per session; exchange-traded products
excluded because a fund has no issuer and a catalyst setup has nothing to
condition on — measured density is a median of 8 articles for funds against 204
for operating companies.

Validators refuse: random cross-validation, raw news counts as estimator
features, a widened experiment budget, an index-restricted intraday universe, a
stop below one average range, a daily ATR on a thirty-minute hold, clock bars
for intraday decisions, volume bars from coarser than one-minute input, rolling
windows spanning the overnight gap, and either label scheme alone.

## Running Right Now

`materialize-edge-rebuild-intraday-history` is merging the selected-session
five-minute bars into a new corpus at
`data/canonical/edge_rebuild_intraday_5m_20260731`. It holds the workspace
lease. Output directory does not exist yet. If it refused the build it writes
findings to `edge_rebuild_intraday_5m_20260731_rejected.json`; a refusal is
information, and no threshold may be weakened to make it pass.

## Exact Next Steps

1. **Confirm the merge.** Then re-run the full suite; the four CLI tests that
   exited 75 during the last run were lease contention, not failures.
2. **Indicator semantics and relationship features.** The user asked for domain
   knowledge to be built into the engine. The correct scope is narrow: a tree
   finds a threshold like "RSI above 70" by itself, so threshold flags add
   nothing. What a tree cannot derive from one row are **relationships** —
   divergence between price and an indicator across two peaks, volume confirming
   or contradicting a price move, and the same reading meaning opposite things
   in a trend versus a range. Encode those; skip the flags. Note that the
   textbook "RSI above 70 is overbought" is actively wrong in a trend, where RSI
   stays high for weeks.
3. **Swing feature builder.** Assemble one row per security per session:
   momentum, trend, pullback, volume and catalyst features, each passed through
   `add_cross_sectional_features`, plus both labels.
4. **Check the signal orders stocks correctly, before training anything.** Sort
   each session by a simple score and compare the top tenth against the bottom
   tenth over the next ten days. If that spread is near zero there is nothing to
   rank and no model will help; stop and change the signal. This replaces the
   test that misfired.
5. **Deterministic top-25 portfolio, no model.** Build the equity curve against
   SPY. This is the number any model must beat.
6. **Train** LambdaMART or LightGBM ranking, purged splits with embargo, then
   the unseen-stock holdout. Published work reports roughly threefold better
   risk-adjusted return from learning-to-rank on exactly this strategy family.
7. **Intraday setup and its economics gate**, same order: population first,
   model only if the population earns.

## Not Started

- Hourly swing bars. The contract specifies hourly features; only daily exists.
  About 6M bars, one to two hours unattended.
- One-minute intraday bars, required before volume bars can be built. Roughly
  790 units over the in-play sessions. Time bars remain in use until then and
  this is recorded, not silently substituted.
- Any model. Nothing has been trained in this program.

## Two Audits That Died Unfinished

Both were killed by a session limit, neither committed anything.

- A swing failure attribution decomposing the rejected population by
  pre-declared cohorts, concentration, and a benchmark decomposition asking
  whether the problem is stock selection or market timing. Worth restarting only
  if the redesign stalls; the redesign supersedes most of its questions.
- The five-minute merge, which is the job now running.

## Verification Commands

Check real exit codes on each.

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
environment fact, not permission to write Python-3.14-only source.

## Standing Prohibitions

- No broker credential. This repository emits predictions; `trading_flow` owns
  execution, orders, positions, and sizing.
- No current screener value in a historical decision. Finviz numbers choose
  which tickers to consider and nothing more.
- Exclude a security only for unprovable point-in-time identity, never for thin
  trading or delisting.
- Extended-hours bars never reach a regular-session VWAP, moving average, ATR,
  or relative-volume denominator.
- Segment classification uses real exchange session bounds, never clock times.
  An early close at 13:00 ET would otherwise file three hours of post-market
  bars as regular session.
- No gate is weakened to make a build or an evaluation pass. A reproducible
  rejection is a valid outcome.
