# Market Predictor Remediation Handoff - R1-R6 Implemented, R7 Next

Date: 2026-07-24
Repository: `C:\project\market-predictor`
Remote: `https://github.com/manishgh/market-predictor` (origin)
Working branch: `r3-lineage` (stacked on `r2-honest-evaluation`, both branched off `main`)
R6.1 checkpoint: `c5a3bc3` (pushed to `origin/r3-lineage`); the current branch tip is authoritative.
Supersedes: `docs/reviews/next_model_handoff_2026-07-22.md` (still valid for R4–R7 scope)

## Objective

Continue the senior-review remediation. **R1 through R6 are implemented in code.**
R6 container execution and security reports still require the new Linux CI run because
Docker is unavailable locally. The next implementation checkpoint is **R7 (Final
Verification and Independent Re-audit)**. This document is self-contained: read it plus
the two review reports below and you can resume without prior chat context.

Context reports (read for the "why"):
- `docs/reviews/ml_statistical_validity_review_2026-07-22.md`
- `docs/reviews/ml_production_architecture_review_2026-07-22.md`
- `docs/reviews/remediation_plan_2026-07-22.md`
- `docs/reviews/next_model_handoff_2026-07-22.md` (the original R2–R7 plan)

The system produces prediction intelligence for swing and intraday workflows. It
does not own alerts, orders, positions, risk, or execution; those remain TradingFlow.

## R4 Completion Update

R4 was implemented, independently reviewed by two read-only senior agents, hardened
against their findings, fully verified, committed, and pushed:

- `69c261a` - immutable hypothesis registry, one-use shadow ledger primitives,
  session-block confidence interval, and promotion attestation primitives.
- `25cd47c` - candidate-only manifests, removal of generic promotion, canonical
  swing/intraday trust workflow, and persisted-evidence substitution rejection.
- `01a0f36` - strict Ed25519 authorization with external trust store, canonical
  retry-safe shadow ledger receipt, prospective shadow timing, OS-released locks,
  content-addressed local releases, atomic activation, immediate-prior rollback,
  path/reparse confinement, durable writes, CLI/docs, and adversarial tests.

Verification at `01a0f36`:

- 263 tests passed.
- Ruff passed across `src` and `tests`.
- strict mypy passed across 106 source files.
- `git diff --check` passed.
- No Python worker remained after verification; model/training guards remain below
  the 4 GiB ceiling.

R4 does **not** claim a real promoted model, real untouched-shadow edge, or Azure
deployment. Those remain `environment_pending`. The remaining reviewed architecture
items are intentionally R5: serving must load one cached model context from the verified
local active pointer, and shadow/outcome inputs must come from the durable deterministic
outcome-maturation repository rather than ad hoc preparation.

## Non-Negotiable Constraints

- **No backward compatibility** before first production deployment. Remove superseded
  paths instead of preserving unsafe legacy behavior.
- **Azure is EXCLUDED** until it is integrated. Skip every Azure item: R4 step 8 (Blob
  ETag/lease CAS), R5 Blob persistence, R7 Azure rehearsal. Implement local equivalents
  where the checkpoint has one (e.g. versioned local release dir + atomic pointer).
- **Reddit alias linkage is DEFERRED** — Reddit is not configured yet. Skip R3 step 3
  (Reddit post-level ticker linkage) until Reddit is integrated.
- **Do not weaken gates to obtain a passing model.** A failed real candidate remains
  failed evidence. Do not simulate `environment_pending` items into a pass.
- **Never inspect, print, or commit `.env` or credentials.**
- **Memory ceiling: keep Python working-set < 4 GiB. One heavy test/training process at
  a time.** The full test suite (`unittest discover`) is one heavy process (~50s).
- **Point-in-time / hash-reproducible** for all training, validation, promotion, replay,
  serving. Serving selection and action semantics must exactly match the policy
  evaluated for promotion.
- **Git policy (current):** commit each verified, atomic unit and push it to
  `origin/<feature-branch>`. Never push `main`. `git push` may be blocked once by the
  auto-mode classifier and succeed on retry.
- **Checkpoint discipline:** commit only after full tests, Ruff, strict mypy, and
  `git diff --check` all pass.

## Environment Gotchas (learned — will bite you)

1. **CWD DRIFT:** the shell's working directory drifts back to `C:\project\trading_flow`
   (a DIFFERENT repo) between calls. ALWAYS prefix git/venv commands with
   `cd /c/project/market-predictor &&`. A bare `git push` from the wrong dir targets the
   wrong repo. Symptom: `./.venv/Scripts/python.exe: No such file or directory`, or a
   `git remote -v` showing `trading_flow`.
2. **pandas 2.x datetime units:** tz-aware `.astype("int64")` can return MICROSECONDS,
   not nanoseconds. Use `pd.DatetimeIndex(x).as_unit("ns").asi8` for ns integers
   (see `canonical/joins.py`, `canonical/reconciliation.py`).
3. **Reserved dataset column prefixes:** `swing/audits.py` and `live_features.py` treat
   columns starting with `future_`, `target_`, `entry_`, `exit_`, `label_` as real
   label/target columns. Do NOT name stamped identity columns with those prefixes — e.g.
   the label-config hash column is `dataset_label_config_sha256`, not `label_config_sha256`.
4. **Permissive test promotion configs:** when you add a promotion gate, the synthetic
   "passes" tests (`tests/test_swing_model.py::_permissive_promotion_config`,
   `tests/test_intraday_model_v1.py`) must disable it, and any new required metric/column
   must be added to the training fixtures (`_training_dataset`) or the trained model
   fails the new gate.
5. **Trainer test fixtures build datasets DIRECTLY** (bypassing the canonical build), so
   they don't get columns stamped by the build pipeline — reads of stamped values use a
   fallback (`stamped_scalar`/`stamped_hash` default to 0/"") for lean fixtures.

## Verification Battery (run before every commit)

```powershell
Set-Location C:\project\market-predictor
.\.venv\Scripts\python.exe -m unittest discover -s tests   # 247 tests, must be OK
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\mypy.exe --strict src\market_predictor
git diff --check
```

## Verified Completed State

Baseline before this work: `845b96d` on `main` (219 tests). Current: `77822e9` on
`r3-lineage` (247 tests). Every commit below is green on the full battery.

### R1 (pre-existing): Frozen time and serving semantics — `9cf7b66`, `2764ea5`

Versioned swing nightly cutoff, canonical `60m` intraday horizon, catalyst is
explanation-only, typed API errors, `SERVING_POLICY_SHA256` in snapshots.

### R2 Causal Core (pre-existing): `56aac9d`

Deterministic stratified ticker holdout, fold-local causal scoring, disjoint
calibration, training-only feature selection, poison tests, final fit after validation.

### R2 Honest Evaluation and Economics — `75b33a7`

- **`src/market_predictor/prediction_policy.py`** (NEW) — single source of truth for
  ranking score, selection, and action labels, shared by serving + evaluation +
  promotion. Immutable `PREDICTION_POLICY_ID`/`PREDICTION_POLICY_SHA256`. Intraday
  ranks by `opportunity * (1 - downside)`; swing by model probability. Also holds
  group-aware ranking metrics (`group_ranking_metrics`) and shared calibration metrics
  (`calibration_summary`, `expected_calibration_error`).
- **`src/market_predictor/execution_policy.py`** (NEW) — versioned, conservative
  execution: gap-through fills (`executable_fill_prices` — stop fills at the worse open),
  bucketed round-trip cost by price/liquidity/volatility, participation caps, cost-stress
  grid, capacity curves. `EXECUTION_POLICY_SHA256`. **Coefficients are conservative
  placeholders; real spread/impact calibration is `environment_pending`.**
- Exact per-bar average uniqueness + weighted effective sample size (`intraday/labels.py`,
  `intraday/evaluation.py`).
- Allocation-aware economics (sequential group compounding, no summed overlaps) + base
  and stress economics; per-regime metrics with `insufficient_evidence`; swing calibration
  gates; promotion gates for independence / effective sample size / folds / stress /
  capacity / worst-regime; and required causal-evidence identity binding.
- 8 required scenario tests in `tests/test_r2_honest_evaluation.py`.

### R3 Event / Universe / Label / Collection Lineage — `77d9cec` … `77822e9`

- **P1-1 event relevance state** (`77d9cec`): dropped `fillna(1.0)` in
  `canonical/joins.py` + both dataset global-event sites. Unknown relevance is excluded
  from sentiment/relevance-mean and counted as low-relevance.
- **P0-3 alignment + reconciliation** (`911b6e4`, `4e133a4`, `27704ea`, `e756950`;
  superseded by R7.3): the original status-only audit has been replaced by immutable
  event-to-decision/window assignments. Canonical builds now reproduce every material
  aggregate from those assignments and bind separate assignment and aggregate hashes
  through training, promotion, and attestation.
- **P1-7 label/execution config hash** (`dea8e44`): `SwingDatasetConfig`/
  `IntradayDatasetConfig.label_config_sha256()`; datasets stamp `dataset_label_config_sha256`
  + `execution_policy_sha256`; `_training_rows` requires exactly one label config (mixed →
  `SchemaMismatchError`); promotion binds it.
- **P1-8 universe identity** (`4929e2e`): swing requires `universe_snapshot_id` +
  membership interval; trainer binds `universe_identity_sha256`; promotion requires it.
- **Collection hardening** (`ae7c037`): new **`src/market_predictor/locking.py`**
  portable `file_lock`; `canonical/store.py` serializes publishers, stages both files,
  renames manifest last; `quota.py::reserve()` atomically checks + records under the lock
  (kills the check-then-act race), plus `record_headers` and temp+rename writes.
- **Symbol master** (`77822e9`): **`src/market_predictor/canonical/symbol_master.py`**
  `SymbolMaster.resolve(symbol, as_of)` across renames/delistings/ticker-reuse; rejects
  overlaps; content-addressed.

### The identity chain now available (important for R4)

Model `metrics` and the canonical manifests now carry a complete, content-addressed
chain that R4's attestation must bind:
`validation_split`, `holdout_assignment_cutoff_utc`, `holdout_ticker_summary_sha256`,
`feature_set_sha256`, `reconciliation_sha256`, `dataset_label_config_sha256`,
`universe_identity_sha256`, `calibration_method`, `folds_causally_ordered`,
`prediction_policy_sha256`, `execution_policy_sha256`, `dataset_sha256`. Promotion
already rejects when any are missing (`swing/promotion.py::_causal_identity_failures`,
same in `intraday/promotion.py`).

## Immediate Next Checkpoint: R4 — Trustworthy Promotion and Release

Implement in `swing/promotion.py`, `intraday/promotion.py`, `registry.py`, and new
modules. **Skip step 8 (Azure).**

1. **Candidate creation cannot set `promoted` status.** Verify `registry.py` /
   `write_model_manifest` cannot emit `promoted`; only the promotion path can. Add a test.
2. **Immutable promotion attestation.** A signed/hashed record binding the full identity
   chain above PLUS: candidate artifact sha, evidence-manifest sha, baseline id,
   gate-config hash, build identity, approver identity, timestamp. Write it atomically
   (reuse `locking.file_lock` + temp+rename from `canonical/store.py`). Promotion writes
   the attestation only after all hashes verify. Mutating any bound input must invalidate
   it (test).
3. **Predeclared hypothesis/baseline registry.** A registry of hypothesis id + frozen
   baseline before evidence is generated. Promotion requires the candidate's hypothesis id
   to be predeclared.
4. **Immutable untouched-shadow bundle + one-time-use shadow ledger.** A shadow interval
   fingerprint that can be consumed exactly once; reusing a consumed fingerprint fails;
   a failed shadow retires that hypothesis family (step 6).
5. **Session-block paired confidence lower bound > 0** for benchmark-relative selected-
   policy improvement. **Reuse the existing session-block bootstrap in
   `src/market_predictor/v3/evaluation.py` (~lines 275–343)** — it already computes
   paired session-block intervals; wire it into canonical promotion gates.
6. **Failed/consumed shadow evidence cannot be reused by the same hypothesis family.**
7. **Versioned local release directories + atomic active pointer.** Publish a release dir
   (model + attestation + evidence), verify all files + attestation, then atomically
   switch ONE active pointer (a symlink or a small pointer file swapped via temp+rename
   under `file_lock`). This is the local equivalent of the excluded Azure activation.
8. ~~Azure Blob ETag/lease CAS activation, rollback, race tests~~ — **EXCLUDED**.

Suggested new modules: `promotion_attestation.py`, `release.py`, `hypothesis_registry.py`,
`shadow_ledger.py`.

Required tests (mirror the R2/R3 style — pure-function where possible, plus a trained-
candidate integration test):
- A candidate manifest that self-declares `promoted` is rejected.
- Attestation binds all identity hashes; mutating any one invalidates it.
- Reusing a consumed shadow fingerprint fails; a fresh one passes once.
- A positive point improvement with a session-block lower bound ≤ 0 fails.
- The active pointer switches only after attestation + all files verify; a partial
  release never becomes active.

## R5–R7 (subsequent checkpoints)

- **R5.1 completed in code:** production routes reference signed local active-release
  repositories; FastAPI preloads one cached context per route; scoring uses cached
  payloads; model release ids enter response evidence; one non-queueing inference
  lease, ticker limits, memory reservations, projected RSS, and serialized route
  replacement are tested. The obsolete Azure serving-release CLI path was removed.
  Real-size combined-model soak/RSS evidence is still `environment_pending`.
- **R5.2 completed in code:** full hash-bound swing/intraday label policies are persisted
  with model manifests; identity-complete live snapshots create immutable semantic
  maturation intents; exact canonical bars produce deterministic attempts, evidence,
  and outcomes; repeated snapshots are deduplicated; online/offline label parity is
  tested.
- **R5.3 completed in code:** canonical outcomes produce hash-validated calibration and
  economics cohorts by release/view/horizon and operational slices. A versioned drift
  policy persists release-specific actionability. Missing, stale, warming, severe,
  tampered, or wrong-release state fails closed in direct and unified serving.
- **R5.4 completed in code:** production startup requires Entra JWT configuration;
  signature/issuer/audience/time claims and `scp`/`roles` are validated. Prediction,
  detailed operations, metrics, and replay have separate scopes; replay defaults off;
  body/ticker/rate state is bounded; structured audit events omit credentials.
- **R5 Bounded Serving / Durable Outcomes / Drift:** cache one active model context (no
  per-request deserialize); admission control + RSS < 4 GiB; deterministic label-horizon
  outcome maturation separate from ad hoc replay; calibration/economic cohorts; versioned
  drift policies that make severe drift non-actionable; auth/scopes/rate-limits/structured
  logs. **Azure Blob persistence EXCLUDED** — use the durable local repository + bounded
  cache; mark Blob as `environment_pending`.
  R5.1-R5.4 are implemented locally. Tenant/app registration, role assignment, JWKS
  refresh, private ingress, and real-size combined-model soak evidence remain deployment
  environment work.
- **R6 Repository / Delivery Hardening implemented:** obsolete Blob release transport
  and its tests were removed; production/collection/research commands have exact golden
  inventories; production imports exclude collection/training frameworks; provider
  credentials use redacted wrappers and Finviz no longer accepts a token in process
  arguments; five universal Python 3.11+ locks use artifact hashes and pinned uv
  0.11.32; the build backend is pinned; the Python 3.11 production and validation locks
  pass clean installs; the production image pins Python 3.11.15 Bookworm by official
  registry digest, installs only the production lock, and contains no compiler/build
  toolchain. CI now runs full tests/Ruff/strict mypy, deterministic lock checks, clean
  production imports, secret and dependency scans, signed-release container startup,
  non-root/read-only liveness, fail-closed readiness, memory capture, license reports,
  Python/image SBOMs, and critical image vulnerability gates. Local verification:
  **305 tests passed**, Ruff clean, strict mypy clean across **125 source/script files**,
  five locks regenerated byte-for-byte, pip-audit found no known production dependency
  vulnerabilities; all 305 tests also pass in a clean Python 3.11
  `validation.lock` environment; and no Python worker remained. The Docker/Trivy portion is still
  `environment_pending` until CI executes it.
- **R7 Final Verification and Evidence Status:** full battery + focused mutation /
  concurrency / memory / release-race / rollback / shadow-ledger / reproducibility tests;
  fresh independent re-audit.

## Do NOT claim complete without real external evidence (`environment_pending`)

- A fresh canonical swing/intraday model passing every promotion gate on real data.
- An untouched real shadow interval with positive confidence lower bounds.
- Real execution/spread/impact calibration for `execution_policy.py` coefficients.
- Azure publish/sync/rollback/DR rehearsal (excluded until Azure is integrated).
- Live source-history quality and matured-outcome performance.

Mark these `environment_pending`, never simulate them into a pass.

## Resume Commands

```powershell
Set-Location C:\project\market-predictor
git checkout r3-lineage
git status --short                # expect empty
.\.venv\Scripts\python.exe -m unittest discover -s tests   # expect 305 OK
# Confirm the R6 Linux CI evidence, then start the R7 independent re-audit.
```

Persistent notes for this effort also live in the assistant memory file
`market-predictor-remediation.md` (constraints, gotchas, per-commit status).
