# Active Edge Rebuild Handoff

Status: ER0 plan/governance checkpoint in progress
Last updated: 2026-07-28
Repository: `C:\project\market-predictor`
Remote: `https://github.com/manishgh/market-predictor`
Branch: `r3-lineage`
Branch tip before ER0: `6a9eb5b`

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

## Current Step: ER0

In scope:

- establish exactly one active plan and one current handoff;
- make both mandatory in `AGENTS.md`;
- link the active plan from README;
- verify documentation consistency;
- commit and push the planning checkpoint.

Out of scope:

- data collection;
- feature/label implementation;
- estimator training;
- changing promotion gates;
- Azure, deployment, alerts, or TradingFlow execution.

ER0 exit:

- `docs/active_edge_rebuild_plan.md` exists and ER1 is fully specified;
- this handoff contains actual state and exact restart instructions;
- repository guidance requires both files;
- documentation checks pass;
- ER0 is committed and pushed.

## Exact Next Step After ER0

Start ER1 only after the ER0 planning commit is pushed.

ER1 must first implement a read-only, hash-bound effective-sample/data-readiness audit.
It must reuse and verify existing artifacts before requesting any download. It may not
train a model.

Required first ER1 investigation:

1. Inventory the exact daily and one-minute source bundles already present.
2. Report usable sessions by year and why sessions are excluded.
3. Report independent decision groups, ticker breadth, regimes, session segments,
   liquidity, and SIP identity.
4. Test ten-session swing non-overlapping phase capacity.
5. Report causal catalyst readiness separately from technical readiness.
6. Freeze the ER1 audit schema and tests before scanning real data.

Do not:

- rerun V1/V2 training;
- use rejected models as baselines for KS5;
- tune setup thresholds from the existing validation/holdout outcomes;
- bulk-download more data before ER1 proves the gap;
- weaken confidence, cost, drawdown, benchmark, or memory gates.

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
