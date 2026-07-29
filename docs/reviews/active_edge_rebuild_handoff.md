# Active Edge Rebuild Handoff

Status: ER1 completed; ER1A in progress; ER1B extended-session transport running
Last updated: 2026-07-29
Repository: `C:\project\market-predictor`
Remote: `https://github.com/manishgh/market-predictor`
Branch: `r3-lineage`
Last implementation commit: `319dd3e`

Read first:

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`
3. `docs/model_cards/primary_v2_failure_attribution_20260728.md`

Do not infer current state from chat history.

## Objective

Rebuild setup populations with positive, reproducible economics before training
another directional model. Resume KS5 distributional modelling only after a newly
versioned specialist passes independently.

## Verified Current State

- KS0 through KS4 are completed in the strategy execution ledger.
- KS5 is pending and blocked by the absence of a passed specialist.
- `SWING.CROSS_SECTIONAL_MOMENTUM.5D.V2` is rejected; no model was retained.
- `INTRADAY.VWAP_REVERSION.30M.V2` is rejected; no model was retained.
- ER1 is complete and reports one blocker, `intraday_session_history_below_gate`.
- ER1A regular-session transport is complete: 8,844/8,844 units,
  32,033,151 canonical SIP/`all` five-minute bars, zero failures.
- ER1B is a sub-step of ER1A. Its plan is frozen and its transport is running.
- ER1A remains the only `in_progress` plan step. ER2 is pending and unauthorized.
- No model has been trained and no model artifact exists in this program.

## Authoritative Evidence

ER1 readiness audit:

- Directory: `data/research/edge_rebuild_readiness_er1_20260728`
- Request SHA-256:
  `f80f70ae299bd5e5a6aeae6aeaa503ef4775573696b7d88856d539dbd1355080`
- Status: `blocked_pending_targeted_acquisition`

ER1A frozen plan:

- Directory: `data/research/edge_rebuild_intraday_history_plan_er1a_20260728`
- Plan fingerprint:
  `96688b25df7abb24d0317649bde87bf29d9497f799d968801b8f5995bb7cb285`

ER1A complete transport:

- Directory: `data/raw/edge_rebuild_intraday_history_er1a_20260728`
- Request SHA-256:
  `5cf109183d4128310be7ad8d9353d3d7b568b3c531ad1458f27be939cdd6c377`
- Authority SHA-256:
  `a474c622c10e7dd9ea23b9ac23b5847e248f5cde9305a1ca3566e949f5c69852`
- 32,033,151 rows, 583 observed symbols, 0.293 GiB peak RSS.

ER1B frozen plan:

- Directory:
  `data/research/edge_rebuild_extended_session_context_plan_er1b_20260729`
- Plan fingerprint:
  `89c91d178ed1095cc33047508c270a817d716e1c594bfe0940064d2de770a250`
- Manifest SHA-256:
  `a2a5675bed96aa312c94da4477aab1e5e7b9570935e8f4f2acda5b47be9edea0`
- Policy SHA-256:
  `2fb6118c448438c5ffe59a1cb3319b39f4e80bf47bca5c77df55948e204700d6`

Model card:

- `docs/model_cards/primary_v2_failure_attribution_20260728.md`

## Completed Step: ER1B Contract And Plan

Implementation commit `319dd3e`.

Delivered:

- `configs/edge_rebuild_extended_session_context.toml`
- `src/market_predictor/edge_rebuild/extended_session_context.py`
- command `plan-edge-rebuild-extended-session-context`
- `collect-edge-rebuild-intraday-history --extended-session-context`
- `IntradayTransportConfig` split out of `IntradayHistoryConfig` so one
  verified collector serves both plans
- registered plan/authority schema pairs, so a plan cannot be replayed under
  another layer's identity

Frozen scope: 804 sessions, 2021-04-27 through 2024-07-08, identical to ER1A.
570 tickers, 404,711 point-in-time ticker-sessions. 8,844 pre-market and 8,844
post-market units. Pre-market is 04:00 ET to the session open; post-market is
the session close to 20:00 ET. Peak planning RSS 0.303 GiB.

Live smoke on 2026-07-29: six units, 1,617 canonical bars, 173 observed
symbols, zero failures, 0.345 GiB peak RSS. Every row fell in 04:00-09:25 ET
with zero regular-session rows. `available_at_utc` minus `bar_end_utc` was
exactly 60 seconds on every row.

Verification at `319dd3e`: 601 tests passed with 85 existing warnings; Ruff,
strict mypy across 180 source files, compileall, and `git diff --check` passed.
The frozen ER1A `policy_sha256`
`252886fb7b7fcfca19917a1daa8e1ea43d950e006287adca12796525c911a830` is unchanged
and is now pinned by a regression test.

## Running Work: ER1B Transport

The full ER1B collection is running and is NOT complete. Do not treat it as
evidence until its `_authority.json` exists and replay verifies.

- Output: `data/raw/edge_rebuild_extended_session_context_er1b_20260729`
- 17,688 units total; the collector is resumable at unit granularity.

Resume or complete it with:

```powershell
Set-Location C:\project\market-predictor
.\.venv\Scripts\market-predictor-collect.exe `
  collect-edge-rebuild-intraday-history `
  --plan-dir data\research\edge_rebuild_extended_session_context_plan_er1b_20260729 `
  --out-dir data\raw\edge_rebuild_extended_session_context_er1b_20260729 `
  --policy configs\edge_rebuild_extended_session_context.toml `
  --extended-session-context
```

Verified units resume without another network request. A run that stops on the
five-failure circuit or an operational batch limit is resumed by re-running the
same command.

## Exact Next Implementation Target

`materialize-edge-rebuild-intraday-history`. No command exists for this step yet.

It must:

1. Replay the ER1A plan, ER1A collection, ER1B plan, and ER1B collection
   authorities, and refuse to proceed unless all four verify.
2. Stream unit shards under the 4 GiB budget. The regular-session shuffle is
   8,844 session-chunk shards into per-symbol outputs; use bounded
   month-partitioned staging, not a full in-memory concat.
3. Publish two physically separate per-symbol stores, `regular/` and
   `extended/`, each with stock and benchmark histories. Every row carries
   `session_segment` and `history_era`. Assert that the regular store holds no
   extended row, the extended store holds no regular row, and the two stores
   share no `ticker`/`bar_start_utc` pair.
4. Merge the verified July-2024-forward corpus without conflicting duplicates,
   splitting it by segment rather than discarding its extended bars.
5. Derive `bar_end_utc` and `available_at_utc` for that corpus through the
   shared `canonicalize_bars` path under the frozen `market_interval_close`
   policy and 60-second delay. Those columns do not exist in `ohlcv.v1`, so
   this is derivation and must be recorded as such.
6. Apply point-in-time membership to both eras and publish per-session
   coverage, including the measured 271 `PARA` ticker-session gap.
7. Publish authority only after every output hash verifies.

Then: reconcile each setup's computed bar end against canonical `bar_end_utc`
and its finalization timestamp against canonical `available_at_utc`; rebuild
causal setups; plan and collect selective one-minute paths; rebuild causal
intraday rows; and rerun ER1. Proceed to ER2 only after at least 750 usable
sessions and a `ready_for_ER2` audit.

## Measured Facts That Constrain Materialization

- The two corpora do not overlap. 804 + 501 - 30 warm-up sessions is 1,275
  usable sessions, above the 1,250 target and the 750 gate.
- `data/artifacts/ohlcv/v3_sp500_current_730d_20260708` already carries
  04:00-20:00 ET bars for its 501 sessions. Extended bars are 20.8% of a
  sampled row population. Do not re-collect them and do not discard them.
- Over those 501 sessions, 21,809 ticker-sessions (7.97%) name a non-member
  ticker and are filtered out; 271 ticker-sessions (0.11%, all `PARA`) are
  members with no bars and are published as a coverage gap.
- Pre-market density ranges from 1.2 to 31 bars per session per symbol. Never
  share a denominator between segments.

## Do Not

- train any model;
- use a current static universe for historical acquisition;
- merge extended-hours bars into regular-session VWAP, EMA, ATR, or
  relative-volume inputs;
- collect full-universe one-minute history before setup discovery;
- impute a missing extended-hours or one-minute bar;
- build ER2 context features before ER2 freezes them;
- attribute a post-close move to earnings, or add news-since-prior-close,
  before ER4 supplies first-observed evidence;
- change the 750-session, four-fold, SIP, cost, or benchmark gates;
- begin ER2 while the ER1 audit is blocked.

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

## Step Closure Template

After implementing any ER step:

- implementation commit and remote ref:
- plan step completed:
- evidence paths and SHA-256:
- focused/full tests:
- Ruff/mypy/compileall/diff:
- peak memory:
- rejected, passed, blocked, or environment-pending findings:
- next plan step:
- exact next command:
- dirty files or running processes:

Update this file and `docs/active_edge_rebuild_plan.md`, then commit and push the
documentation closure before starting the next step.
