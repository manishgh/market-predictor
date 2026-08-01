# Feature Engineering Audit - 2026-08-01

## Scope

Active edge-rebuild swing and intraday paths only. Retired model paths are not
accepted as evidence. The audit covers feature availability, label causality,
missing-data behavior, temporal validation, estimator inputs, and experiment
control.

## Intraday

### Passed controls

- Features are computed from completed volume bars and exact one-minute market,
  SPY, QQQ, and point-in-time sector context.
- Rolling technical state resets each session. Overnight observations cannot
  enter RSI, ATR, EMA, realized-volatility, OBV, or efficiency windows.
- A row is rejected when exact decision-time context is absent. No previous or
  future minute is substituted.
- Entry is the exact next one-minute open. Target, stop, timeout, SPY, and sector
  returns use the same executable interval.
- Feature availability is at or before decision time; label availability is
  strictly after decision time and after the completed outcome path.
- Training partitions are ordered by exchange session, use an overnight
  embargo, and purge any training session whose labels are not available before
  the next partition.
- The locked temporal test is opened once after validation-only candidate
  selection. A deterministic security holdout supplies separate unseen-symbol
  evidence.
- The immutable dataset contains 4,173,230 rows, including 1,410,447 eligible
  rows. Its authority and partition hashes verify.

### Corrected finding

The initial trainer grid contained seven learned candidates while the frozen
experiment budget allowed six. That run was stopped before publication. Commit
`febd2d5` enforces the budget in code and reduces the grid to five learned
candidates: two logistic models, two histogram-gradient-boosting models, and
one ranking model. The deterministic score remains a baseline and is not a
learned candidate.

### Candidate limitations

- Raw `atr_14` is price-scale dependent. A later preregistered feature profile
  should compare it with `atr_14 / close`; the current run remains the frozen
  baseline and must not be relabelled after seeing results.
- Relative volume determines causal universe activation but is not currently an
  estimator feature. A later profile should test activation relative volume,
  minutes since activation, normalized volume overshoot, and volume-bar duration.
- Session behavior is represented continuously by volume-bar progress and is
  audited by opening/midday/late segment. A later ablation may test explicit
  causal time-of-day encoding.
- News/catalyst is deliberately not a direct intraday estimator input. Prior
  evidence found it reduced entry-model validation quality. It remains a
  confirmation, explanation, and ranking overlay.

## Swing

### Passed controls

- Decisions follow completed daily bars and labels enter at the next session
  open.
- Technical features include momentum, trend, pullback, volume, SPY/sector
  relative return, and residual return.
- Daily warm-up is at least 250 sessions. Cross-sectional transforms use only
  same-session eligible securities, are winsorized, and include rank, z-score,
  and sector-relative forms.
- Barrier collisions are resolved stop-first. Return labels include the frozen
  round-trip cost and matching SPY/sector comparisons.

### Blocking findings

- The current edge-rebuild materializer publishes the `technical_market`
  profile. The catalyst feature implementation exists, but ticker news, global
  events, and source-coverage authorities are not yet wired into this
  materializer. A technical swing candidate can be research evidence; it cannot
  satisfy the contract's catalyst-full promotion profile.
- Exact coverage preflight excludes 51 of 658 historical securities because of
  unavailable generations, ticker transitions, delistings, and isolated missing
  sessions. This is 7.75%, above the frozen 5% whole-security limit. No bars are
  imputed. Raising that gate changes the research population and requires an
  explicit approved policy change, recorded in the materialization request.

## Training decision

The corrected intraday baseline training may proceed and remains candidate-only.
Swing training waits for a documented coverage decision, then first trains the
technical profile. Catalyst-full swing training follows only after causal news
and global-event authorities are joined and independently audited.

