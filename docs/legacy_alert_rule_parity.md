# Legacy Alert Rule Parity

## Purpose

This document preserves the technical behavior removed from `market-predictor` in ML V3 checkpoint C1. It is migration evidence for TradingFlow, not a requirement to copy every rule into automated trading.

TradingFlow owns alert evaluation, persistence, deduplication, acknowledgement, notifications, strategy decisions, and execution. Market Predictor owns no runtime alert path after C1.

## Removed Predictor Behavior

The removed daily-bar evaluator generated the following events from completed rows:

| Legacy event | Direction | Trigger | Base score |
| --- | --- | --- | --- |
| `macd_bullish_cross` | Up | Prior MACD histogram/diff `<= 0`, current `> 0` | 2.0 |
| `macd_bearish_cross` | Down | Prior MACD histogram/diff `>= 0`, current `< 0` | 2.0 |
| `ema20_bullish_reclaim` | Up | Prior close `<=` prior EMA20 and current close `>` EMA20 | 2.2 |
| `ema20_bearish_loss` | Down | Prior close `>=` prior EMA20 and current close `<` EMA20 | 2.2 |
| `ema20_ema50_bullish_cross` | Up | Prior EMA20 `<=` prior EMA50 and current EMA20 `>` EMA50 | 2.6 |
| `ema20_ema50_bearish_cross` | Down | Prior EMA20 `>=` prior EMA50 and current EMA20 `<` EMA50 | 2.6 |
| `rsi_oversold_rebound` | Up | Prior RSI `< 30` and current RSI `>= 30` | 1.8 |
| `rsi_overbought_rollover` | Down | Prior RSI `> 70` and current RSI `<= 70` | 1.8 |
| `volume_confirmed_breakout` | Up | Close above prior rolling 20-row high and volume z-score `>= 0.75` | 2.5 |
| `volume_confirmed_breakdown` | Down | Close below prior rolling 20-row low and volume z-score `>= 0.75` | 2.5 |

The prior rolling high/low used `shift(1).rolling(20, min_periods=10)` and therefore excluded the current row.

## Removed Scoring

Volume adjustment:

| Volume z-score | Score adjustment |
| --- | --- |
| `>= 1.5` | `+1.0` |
| `>= 0.75` and `< 1.5` | `+0.5` |
| `<= -1.0` | `-0.2` |
| Otherwise or unavailable | `0.0` |

Only events with final score at least `2.0` were emitted.

Severity mapping:

| Final score | Severity |
| --- | --- |
| `>= 3.5` | High |
| `>= 2.5` and `< 3.5` | Medium |
| `< 2.5` | Low |

## TradingFlow Parity Assessment

Current TradingFlow `WishlistBreakoutEvaluator` implements a stronger long-breakout confluence rather than one-to-one event parity:

- Price at or above VWAP and EMA10 support.
- EMA10 at or above EMA20.
- Positive/improving MACD histogram.
- Session relative-volume or increasing-volume participation.
- Recent-high breakout or minimum session gain.
- Maximum VWAP/ATR extension to prevent chasing.
- Optional matched catalyst context.

| Legacy behavior | TradingFlow status | Migration decision |
| --- | --- | --- |
| Bullish MACD transition | Covered as positive/improving MACD confluence | Keep TradingFlow implementation. |
| EMA20 bullish reclaim | Partially covered by price/EMA10/VWAP support and EMA10/EMA20 alignment | Evaluate only if a promoted strategy needs an explicit reclaim event. |
| EMA20/EMA50 bullish cross | Not an alert-level parity rule | Keep as strategy/confluence research, not a notification by default. |
| RSI oversold rebound | Not covered | Research in TradingFlow before adding; do not migrate automatically. |
| Volume-confirmed breakout | Covered with session participation plus recent-high/session-gain confirmation | Keep TradingFlow implementation. |
| Bearish MACD/EMA/RSI/breakdown events | Not covered by the long-only wishlist breakout evaluator | Route only to an explicitly tested short/risk strategy; do not add to long-only automation. |
| Legacy score/severity formula | Not preserved | TradingFlow's confluence score and severity are authoritative. |

## Acceptance Record

- Predictor alert polling and CSV history are removed.
- Predictor alert backtesting is removed.
- No legacy event is silently converted into an automated TradingFlow order rule.
- TradingFlow may use this matrix when designing or testing future strategy-owned signals.
- Model classifications returned by Market Predictor remain prediction evidence, not alerts.
