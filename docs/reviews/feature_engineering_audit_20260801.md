# Current Feature Engineering Audit

Last updated: 2026-09-07

## Current Long-Only Swing Campaign

`configs/swing_research.toml` governs two return regressors crossed with three
profiles and two exit policies: at most six learned specifications and twelve
model/policy comparisons. Historical records below do not authorize their old model
sequence for this campaign. Intraday is paused. The new contract config governs the
statistical procedure; metadata verification cannot authorize training or promotion.

`configs/swing_research_evidence.toml` pins known metadata, not every data directory.
The audit never follows raw/source payload references or parses exposed evaluation
metrics; that evaluation is stream-hashed for identity only. Metadata counts are:
853,417 technical rows / 604 securities / 1,759 sessions; 27,087 matched broker rows
per profile / 11,720 latest announcements; SEC 689,467 events / 853,417 decisions.
Five retained specialist manifests record twelve trials each, 60 total, with possible
duplicates. Unenumerated history is uncovered, not zero. This is neither unique
experiments nor complete lifetime trials/access history nor full source replay.

The canonical technical builder returns 120 ordered inputs from 40 bases: 13
momentum/volatility, 11 trend, 11 pullback and five volume bases, each represented
by cross-sectional z-score/rank and sector z-score. The NUL-delimited ordered-name
SHA-256 is `59125158f03acc0dcfef11a42da14914475c1bdd7c72af1bbb750df40ed3391e`.
Inventory output lists every actual column and its family, with proposed overlaps:

| Proposed relationship | Existing inputs | Remaining evidence gap |
| --- | --- | --- |
| Medium-term strength | 20/60-session returns and SPY/sector-relative context | Six/twelve-month momentum excluding latest month is absent from this estimator order |
| Short reaction | One/five-session return, gap, intraday return, close location | Not availability-anchored post-release reaction; new residual windows require admission |
| Volume/liquidity | Volume z-score/ratio, dollar-volume log, OBV context | New lagged interactions need causal tests; no dollar-capacity claim |
| Regime interactions | Realized volatility and residual return | No distinct direct SPY/QQQ regime levels, breadth or beta in this order |
| Issuer news | No technical_market issuer-news input; separate broker authority | Earnings/guidance content, novelty, precision/recall and observed-time admission unverified |
| Event response | General price-window context only | Both post-release boundaries, availability and batch/live parity unverified |
| SEC content | Form-level event metadata | Original exhibits, guidance, surprise and first disclosure not proven by counts |

Name mapping is not vertical acceptance. New inputs still need source, batch/live,
consumer, clock, missingness and poison-test evidence. Existing panel requests record
warm-up input history from 2018-05-29; decisions begin 2019-07-09. Do not redownload
that history based on the stale May-2019 requirement.

The July-2025 through June-2026 technical test is already exposed. Specialist
unopened-test statements below are historical per-run facts only. Fixed-ten-session
forecasts and managed returns stay distinct; new selection requires funded NAV
versus SPY, not approximate managed-exit-close excess. Checkpoint 2 adds a funded
ledger and paired base/stress SPY accounting, not a new estimator or feature. Its
initial-fit control selects by causal eligibility before outcomes. Separate lots,
cash funding, costs, marks and the maturation tail are explicit; missing selected
outcomes cannot be filtered away. Independent total-return reconciliation remains
unverified, so output is `price_ratio_diagnostics` / `price_basis_pending` and
economically ineligible. This is not a metadata-audit achievement or model alpha.
The retained control additionally fails selected-outcome completeness: 70/30,525
selected stock-days have all eight fixed-horizon return fields nonfinite. Its
immutable blocked receipt is `data/reports/swing_accounting_control/_manifest.json`
(file SHA-256 `f4411da442dcce28c904050045371c3e1ee20e4e1f6e9f041686e7c4f385ee9e`).
It covers 1,221 initial-fit decisions and 1,230 valuation sessions, never validation
or test outcomes. No benchmark payload or funded real-data result was consumed or
produced after that failure; the 70 rows were not filtered away.

Bounded identity/reason replay found 18 affected tickers: ADS (6), AIV (2), BIIB (2),
CPRI (5), CTXS (10), DXC (1), FRC (2), GPS (1), KSS (3), LB (7), NKTR (3), NLSN (4),
PRGO (5), SEDG (1), SLG (9), WU (1), XLNX (5), XRX (3). All 70 have an inexact label
path; 67 also have an unexpected window at recorded membership boundaries. Seventeen
explicitly cross a missing session (CTXS 10, FRC 2, XLNX 5); these categories overlap.
BIIB's two rows and CTXS's first row have expected windows but inexact paths.
The sector-label-unavailable reason is downstream of masked labels, not independent
evidence of missing sector ETF bars (`swing/features/eligibility.py`). The label
builder masks all eight fixed returns when the path is inexact
(`swing/labels/__init__.py`). The following source review supersedes the earlier
reason-code-only diagnosis. Repair requires complete outcomes for already-selected holdings,
not future-aware exclusions or treating index removal as an unexplained disappearance.
[Alpaca's declared adjustment basis](https://docs.alpaca.markets/us/reference/stockbarsingle-1)
includes splits, dividends and spin-offs under `all`; it does not independently
reconcile every retained event or authorize separate dividend credit. Use retained
evidence first without speculative bulk redownloads.

### Holding-Path Source Review

Independent projected review of 36 hash-verified stock artifacts found 40 paths
with complete positive-volume prices (12 tickers), nine paths with zero-volume
records (BIIB 2, LB 7), and 21 paths with absent retained sessions (CTXS 10, FRC 2,
NLSN 4, XLNX 5). The 40 price-complete paths require 91 sessions omitted from the
combined layer. All 129 overlapping projected OHLCV rows matched raw values.
Raw artifacts have no security_id; collection-era and current panel membership IDs
differ for 17/18 tickers. Thus **price-complete does not mean identity-reconciled**.

Source manifests: raw daily `b99a1d13d9075220db30c3870bc0796b3631bf4ec0bf6424469feeeec9115d93`;
combined daily `c5780d2f406531ec2f6d98372577c105a60687ea72c57a7b6c739515934a7e34`.
No old authority was modified and no selected observation was excluded.

Canonical labels now resolve separate security-identified outcome bars using XNYS,
not observed stock or SPY row counts. Membership end no longer suppresses expected
holding outcomes. Fixed/managed paths and live maturation reject unusable daily
observations; precise daily open/close timestamps include early closes. This code
correction does not certify historical identity or corporate-action return units.

The retained SEC inventory has relevant accession metadata but not the filing
bodies needed for accounting. The independent reviewer inspected these official
documents online; they are research pointers, **not retained hash-bound admission**:

- [LB closing 8-K](https://www.sec.gov/Archives/edgar/data/701985/000114036121026658/nt10026698x7_8k.htm):
  parent rename and VSCO distribution, not a simple ticker-only continuation.
- [AMD closing 8-K](https://ir.amd.com/financial-information/sec-filings/content/0000002488-22-000031/amd-20220214.htm):
  XLNX consideration requires successor-share entitlement, not ticker splicing.
- [CTXS closing 8-K](https://www.sec.gov/Archives/edgar/data/877890/000119312522256805/d393433d8k.htm)
  and [Nasdaq cessation notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=eca2022-262):
  consideration and cessation do not establish the spendable cash-credit date.
- [NLSN closing 8-K](https://www.sec.gov/Archives/edgar/data/1492633/000119312522260583/d407513d8k.htm):
  cash consideration requires separately bound settlement treatment.
- [FRC OTC notice](https://infomemo.theocc.com/infomemos?number=52358):
  receivership does not justify assuming common shares were worth zero.
- [BIIB halt notice](https://infomemo.theocc.com/infomemos?number=47809):
  an options-processing reference price is not an executable stock entry.

Next admission requires independent source identity intervals, corporate-action
terms/valuation/settlement policy, total-return reconciliation, and a new immutable
outcome authority with independent replay. Training remains blocked, not failed.

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
- Entry is the exact next one-minute open. Target, stop, timeout, SPY, QQQ, and sector
  returns use the same executable interval. Intraday label schema V2 abstains when any
  exact benchmark interval is unavailable.
- Feature availability is at or before decision time; label availability is
  strictly after decision time and after the completed outcome path.
- Training partitions are ordered by exchange session, use an overnight
  embargo, and purge any training session whose labels are not available before
  the next partition.
- The locked temporal test is opened once after validation-only candidate
  selection. A deterministic security holdout supplies separate unseen-symbol
  evidence.
- The immutable dataset contains 4,173,230 rows, including 1,410,447 eligible
  rows. Its prior authority and partition hashes verify; it predates label schema V2
  and cannot be consumed by the V2-label trainer without causal rematerialization.

### Corrected finding

The initial trainer grid contained seven learned candidates while the frozen
experiment budget allowed six. That run was stopped before publication. Commit
`febd2d5` enforces the budget in code and reduces the grid to five learned
candidates: two logistic models, two histogram-gradient-boosting models, and
one ranking model. The deterministic score remains a baseline and is not a
learned candidate.

### Current feature profile

- Intraday schema V2 replaces price-scale ATR with normalized ATR and adds
  activation relative volume, normalized volume overshoot, volume-bar duration,
  minutes since activation, and session progress.
- The same shared feature builder serves historical batch and live decisions.
  Tests reject future evidence, missing exact benchmark context, stale decisions,
  and any mismatch in the exact 44 estimator features.
- News/catalyst remains outside the intraday entry estimator. It is a separately
  hash-bound confirmation, explanation, and ranking overlay because the earlier
  direct-feature ablation reduced validation quality.
- The prior Intraday V2 model experiment remains economically rejected after costs
  and is not serveable. Its retained bar authority uses obsolete transformation
  `0da898cc...` and is historical/current-ineligible after the canonical volume-bar
  owner migration. Current code requires transformation schema
  `market_predictor.intraday.bar_dataset_transformation.v2` with SHA-256
  `6fdfd0c8f07e4f7445b66d038cbd936e4459db68e087a5ddbcb30eac4795cb51`.
  The current data authority now exists at
  `data/features/intraday_causal_volume_bar_dataset_20260831_v2` with 794 sessions,
  3,095,688 rows, and 1,365,015 eligible rows. Audit v2 binds the exact five-minute
  projection and reports zero raw-source-cutoff or incomplete-prefix eligibility
  violations. Its execution assessment remains incomplete because per-invocation
  telemetry was introduced after the resumed build; no historical memory evidence was
  inferred. Data-authority registration does not authorize training, promotion,
  serving, or locked-test access. The later V3 branch declared five cross-sectional z-score
  columns without a valid contemporaneous decision-cohort implementation; the
  columns were undefined for asynchronous or single-member timestamps. Commit
  `e168482` removes that invalid contract. Any artifact requiring those columns is
  rejected lineage and cannot train, replay, promote, or serve. A future
  cross-sectional feature group must use one verified batch/live decision cohort
  and be backfilled for the complete intraday model horizon before acceptance.
- A4.1 collection now keeps every selected stock-session and records full-session bar
  coverage only as source metadata. Historical feature eligibility uses data through
  the decision cutoff; label eligibility uses only the managed outcome horizon. A
  later missing bar cannot remove an earlier live-equivalent decision.
- Bar-only and microstructure-enhanced profiles require an immutable matched ablation
  cohort with identical decisions, labels, folds, costs, and benchmark intervals.
  Missing microstructure makes only the enhanced row unavailable.
- Raw quote transport preserves finite nonnegative zero-sided quote states. A4.2, not
  transport, must classify valid two-sided duration, observed-zero states, and
  unavailable quote coverage.
- A5.1 now publishes a strict causal event-cohort preflight. The two Alpaca parent
  authorities contain 17,401 broker-action episodes. Correct exact-ticker and
  CIK-compatible namespace reconciliation produces 1,912 unique attached episodes /
  83,636 event-decision pairs; the earlier 19/862 result was an identity-matching
  defect. All historical source timing remains retrospective
  `provider_publication_proxy`, so production eligibility and locked-test reads remain
  zero.

## Swing

### Passed controls

- Decisions follow completed daily bars and labels enter at the next session
  open.
- Technical features include momentum, trend, pullback, volume, SPY/sector
  relative return, and residual return.
- Daily warm-up is at least 250 sessions. Cross-sectional transforms use only
  same-session eligible securities, are winsorized, and include rank, z-score,
  and sector-relative forms.
- Barrier collisions are resolved stop-first after executable overnight-gap handling:
  stop gaps fill at the worse open and target gaps use the conservative resting-limit
  target. Return labels include the frozen round-trip cost.
- Historical gates used managed-exit-session-close benchmark comparisons. Those
  are approximate, not exact intraday comparisons. The new campaign uses separately
  named fixed-ten-session forecasts and funded daily NAV economics.
- The governed split uses explicit dates with XNYS-verified counts, never percentages:
  1,231-session initial fit from `2019-07-09` through `2024-05-28`, 10-session
  validation embargo, 252-session validation, expanding 1,493-session final refit over
  every post-cutoff development session, 10-session final embargo, and the 251-session
  locked test from `2025-07-01` through `2026-06-30`. This is approximately 4.9 years
  initial fit plus one validation year plus one locked-test year; the causal-news cutoff
  is authoritative.
- Temporal generalization uses the full future point-in-time cross-section. Stable 20%
  unseen-security generalization is a held-out-symbol stress test in the same time
  window, fitted separately; both scopes must pass, but they are not independent time
  samples.
- Commit `7b61873` removes the invalid training-time catalyst cohort rewrite. The
  trainer no longer attaches SEC files from a local path, fills unknown SEC coverage
  with zero, recomputes rank labels, or bypasses sector constraints. Commit `cb2aba5`
  then separates the A2 technical baseline from the future A3 event-driven family.
- Swing and intraday evaluations now publish explicit binary diagnostics for the
  estimator target, positive after-cost stock return, and positive SPY/QQQ/sector
  excess return. A deterministic 64-repeat shuffled-label AUC control must remain at
  chance; abnormal discrimination fails evaluation. Single-class scopes are recorded
  as unavailable rather than misreported as an AUC.

### Current implementation and result

- The base materializer now publishes one catalyst-independent `technical_market`
  population. A separate A3.4 authority publishes exact matched technical-only,
  analyst-revision-only, and combined event-conditioned profiles.
- Sparse missing daily sessions now invalidate only affected 250-session warm-up
  windows and 10-session labels. They are never imputed or bridged. The 5%
  whole-security exclusion rule remains unchanged and applies only to genuinely
  unusable full histories.
- The V12 technical authority contains 853,417 rows, 604 modeled securities, and
  1,759 sessions. The first A3.4 join was defective because old event decision hashes
  were compared directly with rebuilt technical-panel hashes. Corrected A3.4 contains
  27,087 matched prediction rows from 11,720 unique latest broker announcements in
  each of three exact comparison datasets. Blocked families are absent and unknown
  source coverage abstains.
- Monthly partitions isolate historical test outcomes. Development training loads
  requested months and columns; per-run controls do not restore the exposed year's
  independence. Replay verifies profile, decision and
  security identities, session bounds/counts, canonical paths, and hashes.
- Candidate v2 trained six governed logistic and histogram-gradient-boosting models.
  Diagnostic AUC reached approximately 0.55-0.57, but every candidate failed at
  least one calendar, portfolio-daily, doubled-cost, or holding-aligned benchmark
  confidence gate across temporal and unseen-security validation. The result is an
  immutable `no_candidate`; the locked test was not read.
- A3.6 then evaluated upgrades and downgrades separately on the same governed
  decision rows and unchanged validation contract. Both directional cohorts passed
  capacity. Upgrade best worst-scope inner AUC was 0.524 (0.533 chronological / 0.524
  unseen); downgrade best was 0.552 (0.552 / 0.560). Neither cohort passed the 0.60
  AUC gate or after-cost economics in both scopes. The result is
  `no_development_candidate`; outer validation and locked test remain unopened.
- The swing training process remained below its 5 GiB hard memory limit.
- The A2 replacement baseline uses four nested technical groups: momentum/volatility,
  trend confirmation, pullback timing, and volume/liquidity. One regularized logistic
  candidate is evaluated per group; the full group also receives one XGBoost ranker
  and regressor. Selection remains economic and validation-only; AUC is diagnostic.
- Current Finviz snapshots do not establish point-in-time quality, profitability,
  investment, valuation, or estimate-revision history. Those feature groups are
  blocked rather than copied backward or represented as zero.
- Every fitted estimator persists its exact ordered feature subset. Serving and the
  research API slice to that subset and reject missing, duplicate, out-of-order, or
  out-of-bundle columns. No real A2 candidate has been trained yet.

## Vertical acceptance matrix

`Verified code` means implementation plus focused tests. It does not mean a real
immutable artifact exists. `Blocked` means training or serving is prohibited.

| Capability | Source and immutable authority | Batch path | Live path | Model contract | Training / promotion / serving | API | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Intraday technical estimator | SIP/all bar and V2 authorities verify | Verified | Verified; parity and staleness rejection | Exact 44-feature schema verified; incomplete V3 z-scores removed | V2 economically rejected; later z-score artifact invalidated | Blocked until promotion | Rejected |
| Intraday ticker-catalyst specialist | Corrected A5.1 historical preflight binds 1,912 proxy-time episodes / 83,636 event-decision pairs; prospective authorities remain the production source | Upgrade and downgrade directional cohorts trained as technical confirmation filters; coverage failed capacity | New causal chain has 451 identity-bound observations and 12 analyst episodes; SIP source horizon is 1/20 sessions | Frozen 24-hour direct-issuer broker-action cohort; unknown/stale coverage abstains; catalyst is not a direct estimator feature | Four historical directional experiments are `no_candidate`; prospective capacity remains blocked and A6 prohibited | Not serveable | Historical research rejected; prospective collection active |
| Intraday global overlay | GDELT collector and global authority verified in code; no production collection published | Not an estimator input | Observed-time collection, immutable query policy, and unknown/zero behavior verified | Separate global overlay contract verified | Ranking use blocked until runtime authority exists | Blocked until promoted bundle and orchestrator exist | Runtime artifact pending |
| Swing technical estimator | Daily SIP/all, point-in-time membership, and V12 panel verify | Verified | Verified; latest closed session required | A2 nested technical schema and per-model subsets verified | Replacement trainer complete; new candidate not run | Blocked until promotion | Implementation ready |
| Swing ticker-catalyst estimator | Two immutable V2 Alpaca issuer-event authorities cover `2019-07-09` through `2026-07-08`; strict replay verified | Corrected A3.4 publishes 27,087 prediction rows / 11,720 unique latest broker announcements per comparison dataset | No event specialist is live | Combined rating/coverage and separate upgrade/downgrade cohorts passed capacity; price-target/generic remains report-only | A3.5 and A3.6 experiments failed inner AUC and canonical portfolio gates; outer validation and locked test unopened | Blocked until a new preregistered strategy version passes | Development rejected |
| Swing global overlay | Global collector and decision authority verified in code | Separate overlay; never attached as ticker news | Verified code | Separate global authority hash and source policy | Cannot rescue or alter a rejected estimator; ranking use requires complete authority | Blocked until promoted bundle and orchestrator exist | Runtime artifact pending |

## Training decision

Technical data readiness does not block the A2 baseline. Candidate v2 was trained
correctly and rejected because its out-of-sample economic edge was not stable, not
because data was missing. The replacement trainer is verified but has not produced a
new statistical result. A3 causal event authorities now exist for the complete
development horizon. Event-family precision review currently admits only broker rating
actions in both eras; every other family remains blocked. Corrected A3.4 provides
27,087 eligible prediction rows from 11,720 unique latest announcements per comparison
dataset. A3.5 separated directional rating changes from coverage initiation and kept
price-target/generic actions report-only. Both specialists had sufficient chronological
capacity, but all 12 development experiments failed the frozen inner-selection and
canonical economic gates. Outer validation and the locked test stayed unopened;
global context remains a separate overlay.

A3.6 separated upgrades from downgrades without changing rows, folds, costs,
estimators, or gates. Capacity passed, but upgrade best worst-scope inner AUC was
0.524 and downgrade best was 0.552; neither produced stable after-cost economics in
both validation scopes. This closes retrospective directional broker-action slicing
with no candidate and no locked-test access.

A5.1 first found an identity defect and then corrected it without weakening CIK checks.
The corrected historical authority attaches 1,912 episodes. Research-only directional
development admitted 805 upgrades and 860 downgrades; 245 coverage initiations failed
the frozen capacity floors. All four eligible subtype/hypothesis experiments failed
discrimination and economic gates and emitted `no_candidate`. Historical publication
timestamps still cannot provide production first-observed evidence. A5 production
evaluation therefore remains blocked until the prospective event and SIP outcome
horizon is causally complete.
Commit `5530246` supplies the prospective observed-time authority needed to start that
horizon, including strict A4.3 namespace binding, current-membership extension checks,
exact HTTP request/response evidence, revision preservation, cutoff uniqueness, and
strict replay. Real August 21 evidence now starts a new causal chain with 451
identity-bound observations and 12 qualifying analyst episodes. The first SIP source
authority covers August 20 and is session 1/20; 22 of 503 stocks are incomplete
(4.374%, below the frozen 5% ceiling), all benchmarks are complete, and no bars were
imputed. Training rows, a candidate model, and serving eligibility remain absent.
