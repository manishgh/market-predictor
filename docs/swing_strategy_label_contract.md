# Swing Strategy Label Contract

KS2 replaces the generic five-day direction target with independently
registered setup and outcome contracts. These artifacts are research inputs,
not promoted models and not evidence that a strategy is profitable.

## Registered strategies

| Strategy ID | Setup behavior | Outcome horizon and target |
| --- | --- | --- |
| `SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1` | Positive 20-day return and top-quintile sector-relative rank | Five sessions; positive net return, positive SPY excess, and top-quintile SPY excess |
| `SWING.TIME_SERIES_MOMENTUM.5D.V1` | Positive own trend, SMA50/SMA200 distance, and SMA200 slope | Five sessions; positive net and SPY-excess return |
| `SWING.CATALYST_DRIFT.5D.V1` | Complete catalyst coverage, direct relevant event, positive trend | Five sessions; positive net, SPY-excess, and sector-excess return |
| `SWING.SHORT_TERM_REVERSAL.3D.V1` | Complete catalyst coverage, no event, ATR-scaled decline, and low RSI | Three sessions; positive net and sector-excess return |
| `SWING.BREAKOUT_EXPANSION.5D.V1` | Prior 20-session high, prior compression, consolidated volume, and close location | Five sessions; target reached before stop and positive net return |
| `SWING.SECTOR_RESIDUAL_MOMENTUM.5D.V1` | Positive sector residual and top-quintile sector-relative rank | Five sessions; positive sector and SPY excess |

The frozen numeric policy is
[`configs/swing_strategy_labels.toml`](../configs/swing_strategy_labels.toml).
Changing a setup threshold, horizon, target, execution policy, or barrier rule
creates a different policy hash and requires new artifacts.

## Causality and execution

- Setup features must be available no later than `decision_time_utc`.
- Catalyst completeness is reconstructed from the frozen
  `required_ticker_sources` set, observed/observed-empty status, status
  availability, and coverage freshness. Discovered or caller-named source
  columns cannot redefine that set.
- Generic targets, future columns, prior entry/exit labels, and path columns are
  rejected at the setup boundary.
- Entry is the next exact exchange-session open. Exit is the configured exact
  session close.
- Stock, SPY, QQQ, and sector returns use the same entry and exit sessions.
- The bound execution policy is charged exactly once.
- Breakout target/stop collisions use the conservative stop-first rule.
- Breakout returns, exit timestamps, and SPY/QQQ/sector comparisons use the
  executable target, gap-through stop, or timeout exit rather than a later
  horizon close.
- A non-positive ATR stop is not clamped. The row abstains with
  `invalid_barrier_prices`.
- Missing stock sessions, benchmark intervals, source coverage, or execution
  evidence produce bounded abstention reasons.

Large histories are calculated in identity-complete technical batches and
126-session label chunks. Each chunk includes 20 prior setup sessions and the
maximum future outcome horizon. The test suite proves chunked output is exactly
equal to complete-history output.

## Publication

One immutable canonical artifact is published per strategy. A bundle can resume
after an isolated failure only when its request hash, input hashes, strategy
lineage, artifact hash, and required passed audit evidence all agree. The final
`_manifest.json` is written only after all six artifacts exist.

```powershell
market-predictor-research build-swing-strategy-labels `
  --decisions data/canonical/swing_technical_decisions_20190709_20260708_v1.parquet `
  --benchmark-bars data/artifacts/swing_market_panel_inputs_20190709_20260708_v1/benchmark_bars.parquet `
  --config configs/swing_technical_dataset.toml `
  --strategy-policy configs/swing_strategy_labels.toml `
  --research `
  --out-dir data/features/swing/strategy_labels_20210709_20260708_v4
```

The completed technical replay contains 630,978 decision rows per strategy and
3,785,868 rows in total. It produced 410,936 eligible outcomes at 3.129 GiB
peak working set under the 3.25 GiB safety threshold.

Catalyst Drift and Short-Term Reversal have zero eligible rows in this
technical-only replay. Both require observed catalyst coverage: without it the
system cannot distinguish "no event" from "news was not collected." Their
label paths remain auditable, but they cannot enter KS3 training until joined
to the completed causal catalyst lineage.

## KS3 boundary

KS2 defines setup eligibility and outcomes only. KS3 must join the immutable
feature and label identities, freeze temporal and unseen-ticker folds, compare
deterministic/logistic/HGB/ranker candidates on identical rows, and retain
economic rejection evidence. No KS2 artifact is servable or promotable.
