# Catalyst-Confirmation Swing System: Architecture and State Management

This document specifies how the system sources data, manages the tradable universe, warms up indicator state, and distributes work across machines.

Scope:

- US-listed equities
- Mid/small-cap focus
- Long-only swing trading
- Operational architecture only
- No entry, exit, or sizing rules

This is the operational counterpart to the strategy logic specification.

## 1. Design Principles

### Warm-Up Is Backfill, Not Waiting

Indicator state for a newly added ticker must be reconstructed from historical bars through REST backfill.

The live stream must never be used to slowly accumulate initial state. Waiting for live bars loses the early part of a move and defeats the purpose of catalyst-confirmation scanning.

### One Data Connection

The market-data provider permits one concurrent websocket connection per account. The architecture must fan data out internally from one ingestion point. It must not open parallel provider websocket connections to scale.

### Distribute By Function, Not By Splitting Symbols

Scanning is computationally light. Machines are added for redundancy and separation of concerns, not because symbol scanning needs horizontal partitioning.

Splitting the symbol set across nodes is a last resort reserved for provider connection-capacity exhaustion, and only through strict disjoint shards.

### Separate Daily And Intraday State

Slow trend and regime gates live on the daily timeframe. Entry timing lives on the intraday timeframe.

These require separate state stores and separate warm-up depths because their lookback requirements differ by roughly two orders of magnitude.

## 2. Universe Management

The universe is maintained at two levels: a broad superset and a narrow active set.

### Superset

The superset is the full pool of names that could plausibly qualify for trading.

Target size:

- Approximately 1,500-2,000 symbols
- US-listed mid/small caps
- Basic listing, liquidity, and tradability filters

Refresh cadence:

- Daily or weekly

Purpose:

- Maintain daily-timeframe state in advance
- Ensure any ticker surfaced intraday is already warm on slow gates

### Active Set

The active set contains names eligible to trade today.

Source:

- Finviz Elite screen
- Quality, trend, liquidity, and relative-volume filters

Target size:

- Approximately 100 names

Refresh cadence:

- Pre-open
- Intraday intervals

The active set must always be a subset of the superset. If a screened ticker is not present in the superset, treat it as a superset miss, backfill it, and log the miss.

### Rationale

Maintaining daily state for the full superset is cheap because it requires one bar per symbol per day. That makes every screened ticker daily-warm.

The expensive on-demand work, intraday backfill, is limited to the active set only.

## 3. Data Source And Tier Requirements

### Real-Time Stream

Use a single websocket subscription carrying the active set.

The provider connection limit is hard. The system must respect it absolutely.

### Feed Tier

Full consolidated SIP coverage is required.

IEX-only or single-exchange feeds are insufficient because relative-volume and breakout logic depend on total market volume. A partial feed understates volume and corrupts volume-based filters.

Required tier:

- Algo Trader Plus, or equivalent full-coverage tier
- Equivalent access through an Elite-balance waiver is acceptable if it provides full SIP coverage

### Historical REST Endpoint

REST historical bars are used for all warm-up backfill.

The endpoint should support multi-symbol requests so active-set backfill can be done in a small number of batched calls.

Action item:

- Confirm the selected data tier provides at least the daily warm-up depth required in section 4.

## 4. Warm-Up Depth

Warm-up length equals the longest-lookback indicator on the timeframe, plus stabilization margin for recursive indicators such as EMA, RSI, MACD, and ATR.

Use 3-5x the relevant recursive period as a practical convergence margin.

### Daily Timeframe

Used for:

- Trend gates
- Regime gates
- SMA50
- SMA200
- 200-day slope

Requirements:

- SMA200 needs 200 daily bars before a value exists.
- The 200-day slope check compares current SMA200 to approximately 20 bars prior.
- Required warm-up depth is at least 250 daily bars, approximately one full trading year.

Correction:

- Six months, approximately 126 trading days, is insufficient.
- Any daily-gate warm-up using only six months of history is a defect because SMA200 is undefined or biased.

### Intraday Timeframe

Used for:

- Entry timing
- EMA10
- EMA20
- MACD 12/26/9
- RSI14
- ATR14

Requirements:

- MACD's longest base period is 26 bars.
- Stable MACD signal/histogram state needs roughly 5x that depth.
- Required warm-up depth is approximately 130 intraday bars.

At a five-minute bar size, this is a few trading days.

### Rule

Never apply the daily depth to intraday state or the intraday depth to daily state. They are independent.

## 5. State Stores

### Daily Superset Store

Persistent rolling store of daily bars for every superset symbol.

Retention:

- At least 250 daily bars
- Prefer a margin beyond 250 for safety

Update cadence:

- Once after each session close
- Append the new daily bar
- Drop bars beyond the retention window

Purpose:

- Daily indicators are instantly available for any superset name.
- Intraday discovery does not create a daily warm-up delay.

### Intraday Active Store

In-memory indicator state for the active set only.

Creation:

- Built through REST backfill when a ticker enters the active set
- Maintained by live stream after subscription

Persistence:

- Not persisted for the full universe
- Reconstructed on demand

Reason:

- Storing intraday bars for thousands of names is wasteful.
- REST backfill is fast enough for active-set activation.

### Cache / Blob Layer

Use a cache or blob/database layer to avoid redundant history fetches when a ticker leaves and later re-enters the active set.

Rules:

- Cache is an optimization, not the source of truth.
- Entries have a deployment-defined TTL.
- Re-validate stale entries against REST.
- Scope the cache to recently active names, not the entire superset.

## 6. Ticker Lifecycle

### 1. Superset Membership

The symbol qualifies for the superset.

Daily state is maintained continuously through the post-close daily update. The symbol is daily-warm before it ever appears in the active set.

### 2. Activation

Finviz says `ADD`.

The symbol enters the active set.

### 3. Intraday Backfill

A batched REST call pulls the few days of intraday bars needed.

Intraday indicators are computed immediately. Daily indicators are already present from superset state.

### 4. Subscription

The symbol is added to the single live websocket subscription.

The stream continues from the backfill endpoint without waiting for warm-up.

### 5. Maintenance

Incoming bars or ticks update state incrementally.

Gates and entry conditions are evaluated continuously.

### 6. Deactivation

Finviz says `DROP`.

The symbol is unsubscribed from the stream. Intraday state is released, and a snapshot is written to cache for possible fast reactivation.

Daily state persists as long as the symbol remains in the superset.

### Outcome

Between activation and subscription, the symbol becomes fully warm in milliseconds. No multi-day waiting is allowed.

## 7. Distribution And Redundancy

### Ingestion Node

Exactly one active ingestion node holds the provider websocket and live state.

Responsibilities:

- Maintain provider connection
- Normalize ticks/bars
- Maintain live state
- Re-broadcast internal events to downstream services

### Functional Services

Scanning, catalyst matching, execution, reporting, and logging run as separate services.

They may live on separate machines, but they consume the internal broadcast. They must not open provider websocket connections.

### Redundancy

A hot-standby ingestion node is the highest-value redundancy.

Rules:

- The standby must be able to assume the one permitted connection on failover.
- Primary and standby must never connect simultaneously with the same provider account.

### Symbol Sharding

Symbol sharding is a last resort.

Only implement it if the active universe exceeds what one provider connection can carry.

Rules:

- Use multiple provider accounts/keys.
- Each account owns a disjoint symbol set.
- No symbol overlap is allowed.
- Enforce a global dedup key on `(symbol, signal, time_bucket)`.

For the expected active set of approximately 100 names, sharding is not required and should not be implemented preemptively.

## 8. Operational Hazards

### Daily Under-Warm-Up

Feeding SMA200 or slope checks from fewer than 250 daily bars yields wrong or null gates.

Validation:

- Check daily bar count before allowing a symbol to pass any daily gate.

### Partial-Feed Volume Error

IEX-only feeds understate volume.

Validation:

- Verify full SIP feed at startup.

### Connection Contention

Two processes attempting the provider websocket creates authentication errors and data gaps.

Validation:

- Enforce single-writer connection ownership.

### Duplicate Signals

A symbol observed by more than one node can produce double alerts.

Validation:

- Maintain strict disjoint ownership if sharding is ever used.
- Apply global deduplication on `(symbol, signal, time_bucket)`.

### Cache Staleness

Stale cached backfill can start a symbol with outdated bars.

Validation:

- Enforce TTL.
- Re-validate on reactivation.

### Backfill / Stream Boundary

The first live bar must continue cleanly from the last backfilled bar.

Validation:

- Detect duplicate first live bar.
- Detect gap between last backfilled bar and first streamed bar.

## 9. Parameter Summary

| Parameter | Value / Cadence | Notes |
| --- | --- | --- |
| Superset size | ~1,500-2,000 symbols | US mid/small caps with basic liquidity |
| Superset refresh | Daily or weekly | Slow-changing universe |
| Active set size | ~100 symbols | Output of Finviz Elite screen |
| Active set refresh | Pre-open + intraday | Drives subscriptions |
| Daily warm-up depth | >= 250 daily bars | Satisfies SMA200 + slope |
| Intraday warm-up depth | ~130 intraday bars | Satisfies MACD 12/26/9 stabilization |
| Daily store update | Once post-close | Append and trim rolling window |
| Intraday store | On-demand REST backfill | Active set only, not broadly persisted |
| Provider connections | 1 hard limit | Fan out internally |
| Data tier | Full SIP coverage | IEX-only is insufficient |
| Cache TTL | Deployment-defined | Re-validate on reactivation |
| Active ingestion nodes | 1 primary + hot standby | Single-writer connection ownership |
