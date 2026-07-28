# Active Edge Rebuild Handoff

Status: ER1 completed; ER1A targeted intraday history completion is in progress
Last updated: 2026-07-28
Repository: `C:\project\market-predictor`
Remote: `https://github.com/manishgh/market-predictor`
Branch: `r3-lineage`
Last implementation commit: `4257347` (pushed to `origin/r3-lineage`)

Read first:

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`
3. `docs/model_cards/primary_v2_failure_attribution_20260728.md`

Do not infer current state from chat history.

## Objective

Rebuild setup populations with positive, reproducible economics before training another
directional model. Resume KS5 distributional modelling only after a newly versioned
specialist passes independently.

## Verified Current State

- KS0, KS1, KS2, KS3, and KS4 are completed in the strategy execution ledger.
- KS5 is pending and blocked by the absence of a passed specialist.
- `SWING.CROSS_SECTIONAL_MOMENTUM.5D.V2` is rejected; no model was retained.
- `INTRADAY.VWAP_REVERSION.30M.V2` is rejected; no model was retained.
- The final failure-attribution implementation is commit `6a9eb5b`, pushed to
  `origin/r3-lineage`.
- The worktree was clean at `6a9eb5b` before this ER0 documentation change.
- Final verification at `6a9eb5b`: 574 tests passed, 85 existing warnings; Ruff,
  compileall, strict mypy across 172 source files, and `git diff --check` passed.
- Peak audit RSS was 0.569 GiB swing and 0.340 GiB intraday.

## Authoritative Evidence

Swing failure attribution:

- Directory:
  `data/research/primary_v2_failure_attribution_swing_phase_v3_20260728`
- Request SHA-256:
  `ba56fab32f87778fd85307bb58c30a2e2669f42641c62c3e80057ee58547524e`
- 101,918 validation rows, 1,124 sessions, five non-overlapping phases.
- Zero of 180 scope/phase/cohort records passed.

Intraday failure attribution:

- Directory:
  `data/research/primary_v2_failure_attribution_intraday_phase_v3_20260728`
- Request SHA-256:
  `20f1c163d3b5aa6545c7e8d31960208d1aa84b76c0855eb8ac3badab016ab3a2`
- 50,471 validation rows and 355 sessions.
- Average gross return is negative before the exact 10 bps round-trip cost in both
  validation scopes.

Model card:

- `docs/model_cards/primary_v2_failure_attribution_20260728.md`

## Completed Step: ER1

Implementation:

- frozen contract: `5ffa3d3`
- audit engine and CLI: `d9d93c8`
- final lineage, fold, membership, and benchmark evidence: `7b0ce6d`
- command: `audit-edge-rebuild-readiness`

Authoritative local evidence:

- directory: `data/research/edge_rebuild_readiness_er1_20260728`
- request SHA-256:
  `f80f70ae299bd5e5a6aeae6aeaa503ef4775573696b7d88856d539dbd1355080`
- status: `blocked_pending_targeted_acquisition`
- peak RSS: 0.935 GiB
- training performed: false
- download performed: false
- models created: zero

Material findings:

- Swing: 1,254 usable sessions, 583 tickers, 610,818 usable technical
  rows, 125 effective ten-session blocks; all ten phase-capacity checks pass.
- Intraday: 476 usable sessions, 546 tickers, 69,301 usable proxy rows;
  at least 274 additional usable sessions are required.
- Four provisional chronological intraday chunks each have 119 sessions, but
  this does not override the frozen 750-session total-history gate.
- Both horizons verify point-in-time membership identity, SIP feed,
  `adjustment=all`, exact stamped base costs, and exact SPY/sector proxy
  intervals.
- Existing one-minute coverage: 4,469,565 requirements, 88.36% exact. The
  causal sparse-clock policy requires observed trigger/entry/benchmark/exit
  bars and never imputes a trade.
- Catalyst: direct issuer, sector relation, and sentiment are research-ready.
  Business exposure, global context, intraday decision joins, and prospective
  first-observed evidence are not ready.

Verification:

- 596 tests passed; 85 existing warnings.
- Ruff clean.
- Strict mypy clean across 179 source files.
- Compileall and `git diff --check` clean.
- Immutable audit replay verified every published artifact.

## Current Step: ER1A

ER1A is the only `in_progress` step. ER2 remains pending and unauthorized.

Completed ER1A work:

- Inventory verified the five-year PIT universe and found no reusable
  pre-July-2024 intraday bars.
- Full-universe discovery uses SIP/all five-minute bars. Exact one-minute paths
  remain selective and are planned only after causal setup extraction.
- Immutable plan:
  `data/research/edge_rebuild_intraday_history_plan_er1a_20260728`.
- Plan fingerprint:
  `96688b25df7abb24d0317649bde87bf29d9497f799d968801b8f5995bb7cb285`.
- Manifest SHA-256:
  `899a83a33032ab5878779d9feee67adfdabcf0f69f6d5cd0cb39ee768f7b159a`.
- Scope: 804 sessions, 570 historical tickers, 404,711 ticker-sessions,
  8,844 request units, and at most 32,289,798 five-minute rows.
- Peak planning RSS: 0.663 GiB.
- Live smoke: five corrected units, 16,871 canonical rows, zero failures,
  0.294 GiB peak RSS, and verified Alpaca SIP entitlement.
- Provider constraints: 50 symbols/request and 10,000 requests/hour. The
  collector uses two workers and a five-failure scheduling circuit.
- Complete collection:
  `data/raw/edge_rebuild_intraday_history_er1a_20260728`.
- Request SHA-256:
  `5cf109183d4128310be7ad8d9353d3d7b568b3c531ad1458f27be939cdd6c377`.
- Manifest SHA-256:
  `a037802034724bfe54ed1dba0af8320514164c5002bcb35d451061e4daefcefb`.
- Authority SHA-256:
  `a474c622c10e7dd9ea23b9ac23b5847e248f5cde9305a1ca3566e949f5c69852`.
- 8,844/8,844 units, zero failures, 32,033,151 canonical five-minute
  rows, 583 observed symbols, and 0.293 GiB peak collection RSS.

Immediate work:

1. Materialize hash-bound per-symbol five-minute history joined to the PIT
   session snapshots.
2. Reconcile each setup's computed bar end against canonical `bar_end_utc`
   and its finalization timestamp against canonical `available_at_utc`.
3. Rebuild causal setups, then plan and collect selective one-minute paths.
4. Rebuild causal intraday rows and rerun ER1.
5. Proceed to ER2 only after at least 750 usable sessions and a
   `ready_for_ER2` audit.

Exact next implementation target:

Create and test `materialize-edge-rebuild-intraday-history`. It must replay
the plan and collection authorities, stream unit shards under 4 GiB, publish
regular-session per-symbol stock and benchmark histories, merge the verified
July-2024-forward corpus without conflicting duplicates, preserve
`bar_end_utc` and `available_at_utc`, and publish authority only after every
output hash verifies. No command exists for this step yet.

Do not:

- train any model;
- use a current static universe for historical acquisition;
- collect full-universe one-minute history before setup discovery;
- impute missing one-minute trades;
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

The local NumPy stubs use syntax newer than the project mypy 3.11 target. The verified
local strict command therefore uses the installed Python 3.14 environment. Do not
reinterpret that environment fact as permission to add Python-3.14-only source syntax.

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
