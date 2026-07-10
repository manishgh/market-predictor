# Intraday Model Promotion

## Objective

The intraday model ranks long entry setups and estimates stop-first risk. Catalyst/news information is a confirmation and ranking overlay, not a direct technical-entry feature unless a future ablation proves durable incremental value.

## V2 Data Contract

- Labels are computed from complete, consecutive 5-minute bars.
- Signal time is the current completed bar; entry is the next bar open.
- The default opening scope is 09:30-11:30 ET, end-exclusive.
- A 12-bar horizon represents approximately 60 minutes.
- Setup events may be filtered only after path labels exist.
- Non-overlap cooldown uses original session bar positions and is at least horizon plus one bar.
- Target barriers and realized returns include configured round-trip costs.
- Opening-range values are cumulative until the range completes; future bars inside the opening window are never exposed.
- Same-minute relative volume uses only prior sessions for its baseline.
- Benchmark, one-minute, catalyst, and news context is joined by exact ticker/timestamp after labeling and cannot replace labels or OHLCV.

## Model Outputs

Train and audit separate probabilities:

- `target_entry_success_12b`: target is reached before stop.
- `target_exit_risk_12b`: stop is reached before target.
- `target_net_positive_12b`: 60-minute horizon return is positive after costs.

An actionable response should expose entry probability and exit-risk probability. Catalyst confirmation remains a separate overlay so its contribution can be audited independently.

## Promotion Rules

A candidate remains unpromoted unless it passes all configured gates on purged walk-forward OOS predictions and then survives a fresh, untouched shadow interval. Required evidence includes:

- ROC AUC and top-decile lift.
- Cost-adjusted realized return, profit factor, and drawdown.
- Minimum independent selected trades and ticker coverage.
- Stable behavior across market regimes and calendar periods.
- Zero unresolved news/candle alignment errors.
- Known feed type for volume-sensitive features.

Do not lower gates or select a model after repeatedly inspecting the same OOS interval. Failed candidates stay in candidate status with their metrics and reports; they are not copied into the promoted registry.

## 2026-07-10 V2 Result

Lifecycle status: **candidate artifacts; promotion decision rejected**. No V2 model is eligible for the production intraday API while `require_promoted=true`.

The six-month opening-session experiment produced 47,614 non-overlapping setup rows across 196 eligible tickers and 122 sessions. Structural validation found no duplicate ticker/timestamp keys and no cooldown gaps below 13 bars.

The strongest tested exact-path candidate did not pass: ROC AUC 0.5806, top-decile lift 1.1764, selected net realized return -0.184% per trade, profit factor 0.7076, and max drawdown 30.28%. Extra Trees and logistic baselines did not improve ranking. The net-positive direction companion reached ROC AUC 0.4890 and was also rejected.

The selected stream contained 558 capped OOS trades with a 35.13% win rate and 64.44% negative periods. Regime coverage passed across risk-on, neutral, and risk-off observations. Catalyst features were present, but only 21.44% of rows had ticker events and historical Reddit coverage was zero. These facts prohibit claims that Reddit contributed to this trained intraday result.

Monthly entry-success AUC varied materially, from approximately 0.497 in March to 0.645 in June and 0.578 in early July. This is evidence of regime instability, not a production-ready invariant edge. The next valid promotion attempt requires new matured shadow data after 2026-07-08 plus predeclared model and threshold choices.
