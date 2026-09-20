# Market Predictor

Market Predictor builds causal prediction intelligence for long-only swing and a
separate planned investment cohort. It owns data curation, feature construction, model training, temporal
validation, outcome evaluation, and model governance.

It does **not** own alerts, orders, positions, portfolio risk, or execution. Those
responsibilities belong to `trading_flow`.

The system is not deployed. There are no supported legacy models or compatibility
paths. Serving fails closed until a model passes validation and is published in a
hash-verified promoted bundle.

## Unified Product Boundary

The prediction HTTP surface is swing-only: `POST /v1/predictions/swing` accepts
`mode: "swing"` and `horizon: "auto"` or `"10b"` (ten trading sessions).
Intraday/unified prediction routes are removed, not redirected. Investment replay
is a historical swing-prediction simulation, not a long-term investment forecast;
its public `model_view` must be `swing`. The research workbench lists only swing.

TradingFlow owns screening/UI, watchlists, holdings, final risk, approvals and
execution. Its current client consumes ten-session swing evidence. An unavailable
or research-only model never becomes an order recommendation through integration.
Dedicated day-trading commands are removed: training, promotion, setup, dataset and
specialist collection, including the old cross-sectional command group. Production
commands publish swing features/bundles only and reject retired release activation
or rollback. Shared minute/hourly bars and immutable historical evidence remain.
Internal mixed historical domain implementations are not yet fully removed; they
are not supported day-trading entry points or compatibility promises.

The unused intraday cross-sectional research package and four root helpers are
removed. Retention now distinguishes completed swing models from rejected or
unfinished experiments. Referenced rejection receipts remain historical evidence,
not reusable models. `verify-retained-swing-run` checks a completed return run's
original manifests and payload hashes without loading estimators; source drift is
reported explicitly. This check does not certify replay, profitability or serving.
Fresh verification under the active swing-only strategy contract remains required
before historical models can be reused; original manifests must not be repinned.

The first shared-data building block is a bounded Alpaca news HTTP receipt exchange
with matching Python/C# validation fixtures. It preserves exact provider response
bytes, query identity and original receipt time. File imports require a separately
trusted manifest hash. `market-predictor-collect collect-shared-news` adds one
configured local owner, immutable attempt/receipt publication and verified restart.
Its raw data serves both swing and open-ended investment; it does not enable an
investment model, normalized issuer attribution, full-content coverage or model
admission. TradingFlow automatic import/feed migration is not implemented yet. See
[the architecture](docs/catalyst_confirmation_architecture.md#raw-news-receipt-exchange)
and [implementation guide](docs/implementation_guide.md#news-receipt-exchange).

Remaining internal intraday cleanup and retained-model replay are deferred by user
approval so shared product implementation can proceed. Source-hash mismatches still
block historical model reuse. An investment position may be held without a fixed
exit date, but investment forecast/label horizons need a separate approved contract;
the ten-session swing predictor must never be relabelled as investment advice.

Shared-collection verification (September 20): 4,488 Python tests passed, ten
skipped; Ruff and strict mypy passed (392 sources). Independent scoped code/design
review closed after supported transport findings were fixed. TradingFlow's 27
receipt tests passed; its C# source was unchanged. Saved models remain unchanged,
with no promotion. Live provider collection was not run. Full internal retirement
and historical-model replay remain deferred, not certified by these tests.

## Verified State

### Saved History And Training Dates

September 11 archive inspection confirms the saved source ranges below. Counts
match the inspected Parquet footers for all 670 daily-bar and 6,661 news files.
This is not a claim that every symbol has every session or that all rows are
already joined into an accepted model dataset.

| Source | Saved range | Records |
| --- | --- | --- |
| Alpaca SIP daily bars, adjusted | July 9, 2019-July 8, 2026 | 1,088,146 across 670 symbols |
| Alpaca news | July 9, 2019-July 8, 2026 | 714,126 stored records; repeated stories across queries possible |
| SEC filings | July 9, 2019-July 8, 2026 | 689,467 filings across 624 CIKs; zero failed issuer requests in the archive |
| Corporate-action responses | May 29, 2018-September 11, 2026 process-date scope | 570 queried symbols; terms still require accounting admission |

The initial fitting subset ends May 28, 2024, not the saved archive. Validation
is June 12, 2024-June 13, 2025. The configured final refit extends fitting through
June 13, 2025; the historical test is July 1, 2025-June 30, 2026. Ten-session
embargoes separate the splits, and labels must mature before fitting. The test
has already been examined, so it is retrospective research evidence, not a new
independent proof of performance. Fresh prospective evidence must remain separate.
See [the training protocol](docs/model_training_validation_protocol.md).

The historical archive is distinct from the new source-only catch-up collectors.
The old midnight collector and automatic trainer referenced deleted scripts; they
have been replaced by `MarketPredictorSwingCollectionMidnight`; a real scheduled
offline run passed and its normal action was restored. Alpaca extension through
September 10 added 26,638 daily bars per raw/adjusted format and 23,025 stored news
records, with zero failures and successful offline replay. September 11 observations
are separate partial snapshots. SEC catch-up added 20,723 filing metadata records
across 624 issuers through September 10; offline replay passed with zero failed
issuers. Collection checkpoint `19698d6` passed 3,360 tests (five skips), Ruff and
strict mypy against its isolated commit tree. The historical feature delivery
passed its separate publication-replay gate. Predictor replay matched all 586,305
rows across 551 groups and 59 months; outcome replay also matched every decision
exactly. Independent receipt checks passed for both.

Corrected monthly news publication and both feature joins are complete for the
59-month initial-fit period. News is attributed to 224,709 decisions. Technical
and technical-plus-catalyst profiles each preserve all 586,305 rows, with 479,709
feature-eligible rows. Separate saved-data verification found zero future feature
clocks, exact original outcomes and no rows filtered by their returns. The data
manifest is `data/features/swing_corrected_initial_fit_research/_manifest.json`;
the verified counts and limitations are in the
[current feature audit](docs/reviews/feature_engineering_audit_20260801.md).

Final repository Ruff and strict type checks pass. The complete regression suite
passed 4,329 tests with 10 skips after the return-training implementation. These
are research-only datasets and models, not promoted models. Historical news
reception is not proven, unsupported outcomes remain null, and SEC/Finviz are not
silently included in this Alpaca-only monthly news feature contract.

The fixed-horizon training-input audit now verifies all 1,231 initial-fit sessions
and 545 securities appearing in that period (586 is the campaign-wide retained
list). There are 378,037 usable stock/SPY/QQQ/sector outcome rows; complete-case
input/outcome intersections are 194,679 technical and 193,125 catalyst rows.
These are diagnostic counts, not a rule deleting incomplete rows. The audit
does not itself authorize fitting. The objective-specific return trainer is
implemented, independently reviewed and trained; new relationship/reaction profiles
still require implementation. See the current
feature audit for precise requirements. Run the immutable saved-data diagnostic:

Implementation `b3b2372` is pushed. The fresh hash-bound receipt passed after RAM
recovered, with diagnostics identical to the original audit; all 58 post-format
focused tests passed. The [active handoff](docs/reviews/active_edge_rebuild_handoff.md)
records its exact pin and the newly approved percentage-only memory policy.
The swing audit request now sets `maximum_system_used_percent = 90.0`: it stops
at 90% physical-memory use, without a separate 2 GiB free-RAM floor. Its 5 GiB
process budget and 0.75 GiB process headroom remain. Source collectors keep their
own limits; the shared guards bound into historical replay evidence are unchanged.
The configured audit passed with all diagnostics unchanged; policy implementation
`7bb805b` is pushed and independently reviewed.

```powershell
.venv\Scripts\python.exe -B -m market_predictor.research_cli audit-swing-training-readiness --root . --config configs/swing_training_readiness.json --config-sha256 8d1e2870eac8f7406376eca1af202dbc5a710d344c1e953fce80faa9a58173e4 --output data/reports/swing_initial_fit_training_readiness_percentage_only.json
```

The output already exists locally; choose a new report filename for an explicit
rerun. The command verifies pins and fails rather than overwrite existing evidence.
Cloud deployment is out of
scope until training and evaluation are complete. Research results do not
guarantee outperformance of SPY.

### Trained Swing Return Models

Implementation `07963cc` is pushed. Regularized linear regression and shallow
boosted trees now predict ten-session stock return above SPY after stock costs.
Both use the existing 120 technical inputs, not the unfinished issuer/SEC reaction
profile. Saved initial-fit history spans July 9, 2019-May 28, 2024: 586,305
published decisions across 545 securities, with 314,167 eligible matured rows
used by each final fit. No outer-validation or historical-test data was opened.

All 16 validation fits and two final research fits completed sequentially, with
peak process memory of 2.406 GiB. Four chronological folds use ten-session
embargoes and actual label-maturity purging. Missing-value handling and weighted
linear scaling fit only on training rows; session weights are balanced. Separate
transfer fits exclude all 101 held-out security identities from preprocessing and
training. All 18 artifacts and 266 source/config/code hashes verified afterward.

Results below combine saved out-of-fold predictions, weighting each observed date
equally. Positive error change means worse than predicting zero excess return.

| Model and evaluation | Mean daily rank correlation | Squared-error change versus zero excess |
| --- | ---: | ---: |
| Linear, later dates | 0.0045 | +1.77% |
| Boosted trees, later dates | 0.0147 | +1.99% |
| Linear, unseen securities on later dates | 0.0070 | +1.28% |
| Boosted trees, unseen securities on later dates | 0.0003 | +2.14% |

The predictive signal is weak. Only 187,432 of 289,802 temporal predictions have
admitted comparison outcomes (64.68%); unknown outcomes remain recorded, not zeroed
or silently removed from a portfolio. No funded portfolio, SPY outperformance,
promotion or live readiness is established by these regression diagnostics.

Artifact directory: `data/research/swing_technical_return_models_initial_fit`.
Each learner's `final_refit/model.joblib` is a research model; row-level validation
predictions, preprocessing, runtime versions, source pins and metrics are retained.
The [active handoff](docs/reviews/active_edge_rebuild_handoff.md) records manifest
hashes and exact resume instructions. The completed output is immutable.

```powershell
.venv\Scripts\python.exe -B -u -m market_predictor.research_cli train-swing-returns --root . --config configs/swing_return_training.json --config-sha256 47155e2d6ef43e9efb6bb54621913e3df5ee0f66797153c30f71d8f69e5593cc --output data/research/swing_technical_return_models_initial_fit
```

This command requires a new output directory for a new run. To resume an existing
run, supply `--resume-checkpoint-sha256` with an independently verified checkpoint
hash. Inputs, implementation, configuration and runtime must still match; completed
fits are verified, not overwritten. Do not rerun the completed experiment by default.

### Source Collection

Portable Python entry points (run from the repository root). The provided configs
extend the pinned local archives; they are not a cold-start seven-year downloader:

```powershell
.venv\Scripts\python.exe -B -m market_predictor.swing.datasets.alpaca_incremental --config configs/swing_incremental_collection.toml --through 2026-09-11
.venv\Scripts\python.exe -B -m market_predictor.swing.datasets.sec_incremental --config configs/swing_incremental_sec.toml --through 2026-09-11
```

`--through` is an inclusive UTC date. Omitting it collects completed UTC days
through yesterday. Asking for today also records a separately identified partial
snapshot, never complete-day coverage. Use `--offline` to verify existing
receipts without HTTP. `--max-units` (Alpaca) and `--max-issuers` (SEC) bound new
attempts without counting verified skips. A bounded incomplete run is not success.

Windows adapter and scheduler installation:

```powershell
powershell.exe -NoProfile -File scripts/run_swing_data_collection.ps1
powershell.exe -NoProfile -File scripts/install_swing_collection_task.ps1 -WhatIf
powershell.exe -NoProfile -File scripts/install_swing_collection_task.ps1
```

The adapter requests today's UTC date and runs Alpaca then SEC sequentially. It
loads credentials from the existing environment/`.env`, does not score sentiment
or train models, and preserves nonzero child exit codes. Busy/memory pressure
returns 75. The local task runs at midnight local time while the user is signed
in and AC power is available; missed runs catch up when possible. Old task XML
is exported before migration. The Python collectors do not depend on Windows.

Alpaca retains exact response bytes for SIP raw/adjusted daily bars and news.
News windows use provider update time; publication, revision and observed times
remain separate. An article revised today is not evidence that its revised text
was available when originally published. Three-day overlapping observations
capture recent revisions without replacing earlier receipts. Older revision
completeness is not claimed. Adjusted-bar retrieval vintages must not be spliced
into one supposedly unchanged price series.

SEC reuses the old archive and collects issuer CIKs one at a time through the
existing canonical filing collector. It retains failed attempts and resumes only
after checking hashes and exact request bindings. SEC data here means submissions
and filing metadata, not a downloaded body/exhibit for every filing. Historical
identity relations are query hints, not proof of current ticker ownership.

Progress and results live under `data/raw/swing_incremental_alpaca/` and
`data/raw/swing_incremental_sec/`; wrapper logs are in
`data/runtime/collection_logs/`. Successful acquisition does not establish issuer
attribution, model-ready features, or promotion. Source failures are independent,
and no missing observation becomes a neutral sentiment or zero return.

### Research Status

- Active development branch: `er-intraday-refactoring`.
- Current research priority is long-only, ten-session swing selection against
  buy-and-hold SPY. `configs/swing_research.toml` freezes the return objective,
  six learned specifications, two exit policies, costs, funding and statistical
  procedure. This is a new research contract, not a trained or promoted model.
- The July 2025-June 2026 test period has already been viewed. The retained swing
  trainer refuses to reuse it as a fresh final test before loading data or fitting.
  Historical development research is still permitted; fresh final evidence must
  be recorded prospectively. See the active plan for the implementation sequence.
- **Swing baseline:** estimator inputs are `technical_market` only. Alpaca ticker
  news is a confirmation/explanation overlay and does not alter baseline probability.
- **Swing event-driven:** Alpaca direct ticker news is the only currently permitted
  ticker catalyst source, subject to complete causal authority and specialist ablation.
- **SEC filings:** the current estimator does not consume SEC features. The SEC
  authority distinguishes verified no-filing observations from unknown coverage.
  After causal collection and exact issuer attachment, SEC will be evaluated as a
  separate issuer-specific estimator profile rather than treated as a permanent
  overlay-only source.
- **Other overlay/audit sources:** Finviz and verified global or sector sources.
  They cannot be attributed to a ticker as direct issuer news.
- Reddit and Seeking Alpha are removed and prohibited from collection, features,
  training, and serving.
- Swing model decisions begin on `2019-07-09`. Earlier market bars are indicator
  warm-up only and cannot produce model features, labels, train rows, validation
  rows, or test rows.
- The implemented replacement policy keeps a within-sector ranking target of 50 and
  uses a hard floor of 30. It persists sector peer count, rank eligibility, target
  status, and ranking reliability weight. Groups with 30-49 peers remain eligible
  with weight `decision_time_sector_peer_count / 50`.
- Sector allocation targets 20%; it adapts to 25% when only four sectors are
  represented and 33.3% when only three are represented. Sessions with fewer than
  three represented sectors are skipped.
- Swing accounting now replays cash-funded overlapping lots, daily marked holdings,
  one cost deduction, idle sessions and the complete ten-session exit tail against
  SPY buy-and-hold. Approximate managed-exit-close comparisons cannot qualify or
  rank candidates. Price-basis admission remains blocked: declared adjusted prices
  are not independently verified total-return evidence. No outperformance is claimed.
- The approved event-aware replacement separates tradable shares, spendable cash,
  unpaid proceeds and contingent rights. One lot calculator feeds model targets
  and the same portfolio funding loop. Unknown valuations produce unavailable NAV
  and returns, not zero or a maximum payout. Payments require evidenced availability;
  residual claims are not forcibly sold at the ten-session horizon. These calculations
  are implemented, but real source admission, full-cohort materialization and new training remain
  pending. Historical price-ratio diagnostics cannot substitute for this input contract.
- Exact-unit SIP collection now takes its `raw` or `all` price basis from the
  verified acquisition plan. New responses retain original HTTP bytes and query
  receipts; replay compares those responses with normalized Parquet values.
  Raw and adjusted collections cannot be resumed or consumed interchangeably.
  `plan-swing-initial-fit-raw-prices` reconstructs stock decision/holding sessions
  and complete SPY/QQQ/sector benchmark ranges from pinned identity evidence.
  `collect-swing-initial-fit-raw-prices` requires that plan's independently saved
  authority hash and verifies its reconstruction before collection or offline replay.
  Collection alone does not admit ownership, corporate actions or training labels.
  The first raw archive contains 601,834 daily bars across 564 verified requests;
  offline replay passes. A count audit finds 1,134 fewer rows than requested across
  28 ticker ranges, so it is not yet a complete admitted holding-path dataset.
  `collect-swing-symbol-corrections` supports a separately reviewed, pinned
  historical-symbol correction plan. It reuses existing benchmark evidence and
  downloads only the replacement intervals. Source selection never falls back to
  the replaced ticker; labels, news joins and features must be rebuilt afterward.
  The first corrected source inventory has 602,709 observations after replacing
  601 wrong-issuer rows. It retains 259 missing sessions and 21 zero-volume
  observations as explicit gaps; it is not an admitted training-label dataset.
- `materialize-swing-fixed-holdings` compiles bounded, reviewed JSON requests into
  actual fixed-horizon holding specifications and replay from the existing accounting
  kernel. It reads only pinned raw source segments under the shared workspace lease.
  Missing entry/ownership prevents specification creation; missing later observations
  remain unavailable without shifting sessions. Typed action facts and claim marks
  remain diagnostic interpretations: reportable returns are always null until
  independent source admission. No sale, cash payment or managed exit is inferred.
  `configs/swing_fixed_holding_demonstration.json` describes two real-source diagnostic
  lots, not a training dataset. Use `--request-file`, its independently retained
  `--expected-request-sha256`, and a new immutable `--output`; replay of an existing
  output also requires `--expected-output-sha256`. The current handoff records pins.
- `market-predictor-research audit-swing-accounting-control --root .
  --output-directory data/reports/swing_accounting_control` replays a frozen
  initial-fit momentum control from retained files. It neither trains a model nor
  opens validation/test outcomes. Output is immutable and research-only.
- Event-aware Python target and funded-accounting APIs accept an explicit `simulation`
  context loaded by `load_trade_simulation_context` from the independently file-hashed
  `configs/swing_trade_simulation.toml`. Canonical ordinary-sale executions generate
  modeled dollar receivables and next-XNYS-open reuse, not historical broker receipts.
  The shared simulator records policy/input hashes and generated identities, preserves
  observed availability, and reports retrospective label maturity separately. A final
  sale's next-open cash settlement does not extend the ten-session return. No corporate
  payment/mark or fill is inferred; source admission and live promotion stay separate.
- The first retained-data accounting control is **blocked**: 70 of 30,525 selected
  stock-days lack complete fixed-horizon outcomes. No stock was dropped to produce
  a performance score, and no real-data ledger or SPY result was emitted.
- A new, explicitly retrospective research population can exclude entire security
  identities through `configs/swing_research_cohort.toml`. Run
  `market-predictor-research audit-swing-research-cohort --root . --output
  data/reports/swing_research_cohort/approved_research_population_audit.json` to publish the cumulative
  exclusion and sector/year row audit. The eighteen proposed exclusions plus 27
  inherited exclusions total 45/631 (7.13%), within the user-approved 10% cap for
  this research dataset. The admitted restriction retains 586 securities; no
  original data is deleted. The former 5% blocked audit remains historical evidence.
  `materialize-edge-rebuild-swing-panel --research-cohort <audit.json>` applies an
  accepted restriction before bar batches, labels and peer transforms, in a new
  immutable output directory. It does not certify prices or make retrospective
  results eligible for promotion. The retained classification/ranking trainer
  rejects this restricted population; the planned development-only return trainer
  remains to be implemented for the six new fits.
- `market-predictor-research audit-swing-holding-identity --root . --output
  data/reports/swing_research_cohort/holding_identity_preflight.json` checks the
  retained cohort's exact ten-session ownership windows using metadata only.
  It separates terminal immature decisions from missing post-removal identity,
  keeps excluded competing owners visible, and reports initial-fit coverage
  separately. Exit code 2 means uncovered identity evidence, not failed model
  performance. It does not certify bar availability, prices or benchmark returns.
  The current report covers 842,446 retained decisions: 836,638 covered mature
  windows, 947 requiring additional ownership evidence, and 4,861 terminal immature
  decisions. These counts do not authorize additional stock exclusions or training.
- `market-predictor-research audit-swing-holding-observations --root .
  --output-directory data/reports/swing_holding_observations` reuses the raw SIP
  archive for the frozen initial-fit holding requirements. It reports bar presence,
  price validity and ownership separately; a valid bar never proves the requested
  security owns it. Numeric reads are limited to exact required sessions ending
  within the initial-fit period. Outputs are immutable diagnostics, not an outcome
  source for training. Replay requires `--expected-audit-sha256` with the independently
  retained hash printed by the original run; it verifies the report and source/output hashes.
  Exit code 2 indicates corrupt source observations; ordinary missing observations
  and unresolved ownership are explicit report fields, not silent passes.
  The initial-fit inventory reproduced 566 decisions across 60 securities:
  1,106 distinct required security/ticker sessions, 876 valid observations,
  209 missing and 21 invalid. Ownership is independently unresolved for 574
  sessions; these counts overlap. The 23 securities with missing/invalid bars
  are not automatically excluded. No repaired labels or model result is claimed.
- `market-predictor-collect collect-swing-holding-corporate-actions --root .
  --out-dir data/raw/swing_holding_corporate_actions` collects all Alpaca action
  families for the pinned initial-fit ticker inventory. Raw response bytes and
  pagination receipts are retained; successful tickers resume without refetching.
  Resume requires `--expected-audit-sha256 <retained-report-hash>`; add `--offline`
  to verify without fetching. The pin protects prior successes and failure history.
  The process-date query is not an announcement-time or effective-date filter.
  Empty responses, incomplete fields and failures do not prove that no action
  occurred, and collection does not authorize ownership or return accounting.
- Swing labels now accept separate security-identified outcome bars, retain holding
  windows after index removal, and use exact XNYS sessions and daily timestamps.
  Missing sessions and zero-volume records cannot produce invented fills. This
  corrects the builder, not the retained data: 40 affected paths have price coverage
  pending identity proof, while 30 require additional source evidence. No repaired
  authority or new trained model is claimed.
- Official holding-source documents are collected through
  `market-predictor-collect collect-swing-holding-source-documents --out-dir
  data/raw/swing_holding_source_documents`. Add `--offline` to verify without
  networking. Exact URLs and byte limits live in
  `configs/swing_holding_source_documents.toml`; SEC requests use `SEC_USER_AGENT`.
  Successful responses resume without refetching; failures remain explicit.
  The initial acquisition retained six of ten documents, with two failed and two
  deferred after an OCC HTTP 403. These are unreviewed response bytes, not approved
  corporate-action returns, historical availability or security identities.
- `market-predictor-research replay-swing-transfer-history --root .
  --output-directory data/raw/swing_transfer_history` collects only configured,
  date-anchored daily SIP windows for selected index-transfer holdings. Add
  `--offline` to replay retained bytes without credentials. The frozen inventory
  is `configs/swing_transfer_replay.toml`: eleven tickers, 38 decisions and 140
  required ticker-sessions. Reports show exact OHLCV differences; they never
  overwrite retained bars or authorize identity, fills or total-return accounting.
- `universe/security_class_evidence.py` extracts hash-bound SEC cover-page stock
  class, exchange and issuer facts. Ambiguous registrants/classes fail closed;
  reporting-period dates are not historical ticker-validity intervals.
- Live inference excludes individual missing or cold securities through the governed
  5% ceiling. Cached models are bound to the active contract, trust store, promotion
  policy, and model-size limit.
- The retained pre-repair swing authority was published and strictly replayed:
  853,417 `technical_market` rows, 604 modeled securities, and 1,759 sessions from
  `2019-07-09` through `2026-07-08`.
  It remains historical evidence; the updated holding-path loader requires a new
  schema and implementation-bound authority, rather than reusing these old labels.
- Prior swing candidates are rejection evidence only; no rejected or obsolete model
  is retained as a supported compatibility path.
- A2 replaces that experiment contract with four nested technical ablations
  (momentum/volatility, trend confirmation, pullback timing, volume/liquidity), followed
  by one full-feature XGBoost ranker and regressor. The implementation is verified, but
  no new real candidate has been trained and no performance result is claimed.
- A3 historical issuer-event and precision authorities currently admit only direct-
  issuer broker rating actions. The internal source code calls this family
  `analyst_revision`, but the records are predominantly rating upgrades, rating
  downgrades, and new/resumed coverage; they are not EPS-estimate revisions. Earnings,
  guidance, offerings, M&A, regulatory, product, and SEC families remain unavailable
  because their reviewed precision or source completeness did not pass.
- An identity bug in the first A3.4 comparison joined old and rebuilt security hashes
  directly, reducing 17,401 broker announcements to 50. The corrected immutable
  comparison aligns exact ticker and prediction timestamp, rejects conflicting CIKs,
  and publishes 27,087 prediction rows from 11,720 unique latest broker announcements
  in each of three datasets: technical-only, broker-action-only, and combined.
- A3.5 trained separate rating-change and coverage-initiation specialists. Each used
  technical-only, broker-action-only, and combined profiles with logistic and
  histogram-gradient-boosting estimators. Capacity passed, but all 12 development
  experiments failed the frozen 0.60 AUC/generalization and benchmark-relative
  economic gates in a separate inner selection window. No experiment qualified to
  open outer validation; the locked test also remained unopened and no model was emitted.
- Intraday V2 is published and replayable but economically rejected after costs.
  It is not serveable. The later V3 cross-sectional z-score lineage is invalid and
  prohibited because its declared inputs lacked a valid contemporaneous cohort
  transformation.
- A4.1 now provides bounded, page-resumable Alpaca SIP trade and quote collection with
  immutable request/job/attempt/raw-page lineage, exact session bounds, failed-attempt
  hashing, path isolation, and a 4 GiB hard memory limit. The corrected collection
  plan contains 43,226 selected stock-sessions and 86,452 jobs; end-of-session bar
  coverage is metadata, not an earlier-decision selector. A two-job live Alpaca probe
  completed with zero failures at 0.35 GiB peak RSS. It is intentionally incomplete
  and cannot authorize microstructure features or training.
- A4.3's current bar-only intraday data authority is
  `data/features/intraday_causal_volume_bar_dataset_20260831_v2`: 794 sessions, 501
  tickers, 3,095,688 rows, and 1,365,015 eligible rows from verified SIP/all one- and
  five-minute bars, SPY, QQQ, sector ETFs, and point-in-time membership. Audit schema
  `market_predictor.intraday.bar_dataset_audit.v2` verifies the exact projection lineage,
  raw source cutoffs, five-minute prefix completeness, labels, schemas, duplicates, ATR,
  and prohibited-feature rules. Full per-invocation memory evidence was introduced after
  this resumed build, so its execution assessment remains explicitly incomplete; no
  memory history was reconstructed. A4.4 trained separate continuation and long-reversion
  bar-only baselines on the byte-identical historical authority. Both published immutable
  `no_candidate` evidence: the
  best audited positive-return ROC-AUC was 0.510/0.516 for continuation and 0.513/0.508
  for reversion across seen/unseen securities, with negative after-cost economics in
  the controlling scopes. No candidate model was written and the future holdout stayed
  closed. Trade/quote features remain prohibited until a complete A4.1/A4.2 authority
  exists.
- Intraday label schema V2 requires exact stock, SPY, QQQ, and point-in-time sector
  returns over one executable entry-to-managed-exit interval. Missing benchmark
  evidence abstains. Swing and intraday evaluations report named after-cost binary
  outcomes and a shuffled-label AUC control; AUC remains diagnostic only.
- No swing or intraday model is promoted. The prediction API therefore returns no
  model prediction and must fail closed.

Read these documents in order:

1. [Engineering covenant](AGENTS.md)
2. [Active edge-rebuild plan](docs/active_edge_rebuild_plan.md)
3. [Active handoff](docs/reviews/active_edge_rebuild_handoff.md)
4. [Prediction architecture](docs/catalyst_confirmation_architecture.md)
5. [Implementation guide](docs/implementation_guide.md)
6. [Training protocol](docs/model_training_validation_protocol.md)

## Data Boundaries

- **Alpaca premium:** SIP market bars are estimator data. Direct ticker news is an
  estimator input only for a separately governed event-driven model family.
- **SEC EDGAR:** current issuer authority and causal audit source; planned as a
  separately ablated issuer-specific estimator profile after causal collection and
  attachment are verified.
- **Finviz Elite:** candidate screening and current metadata only; never historical
  membership authority or ticker-news estimator input.
- **Global and sector sources:** separately identified context overlays only.
- **Benchmarks:** SPY, QQQ, and point-in-time sector ETFs.

Unknown coverage is not converted to zero. Historical publication-time backfills are
research evidence and cannot be represented as prospectively observed events.

Issuer history outside index-membership dates can be requested with
`market-predictor-collect collect-alpaca-news-history --query-scope configs/swing_issuer_news_corrections.json --start-date 2019-07-09 --end-date 2024-05-28 --out-dir data/raw/swing_issuer_news_corrections --workers 1 --chunk-days 31`.
Use either `--query-scope` or `--memberships`, never both. Scope files bind historical
symbol intervals and source-document hashes; they do not certify article attribution.
The collector checks laptop RAM as well as process RSS. Defaults stop new requests
at 85% physical-memory use or below 2 GiB available; flags
`--maximum-system-used-percent` and `--minimum-system-free-gib` configure these checks.
Pressure stops leave pages resumable and never publish a completed collection.
These are cooperative checks, not a hard limit on allocation spikes or other apps.

## Setup

Requires Python 3.11 or newer. The verified local environment currently uses Python
3.14.

```powershell
Set-Location C:\project\market-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Credentials belong only in `.env` or a managed secret store.

```powershell
market-predictor-collect --help
market-predictor-research --help
market-predictor-prod --help
```

### Research Workbench

The corrected initial-fit rebuild is still in progress. Its separate stages are:

- `market-predictor-collect materialize-swing-corrected-outcomes`: reconstruct
  hypothetical fixed holdings from verified raw prices and corporate-action scope.
  Unknown payments, rights and managed exits remain unavailable.
- `market-predictor-research materialize-swing-research-predictors`: rebuild
  technical indicators from saved adjusted histories, with independently bound
  raw dollar volume. Completed stock groups have hash-checked resume checkpoints.
- `market-predictor-research materialize-swing-research-dataset`: join the complete
  monthly decision population, corrected outcomes and causal issuer-news evidence.
  Technical/news ablations share a comparable population without deleting rows
  with unknown evidence. These artifacts do not themselves authorize training.

Use each command's `--help` for required independent source hashes. Exact current
artifact paths, pins, pending source issues and verification results are maintained
in `docs/reviews/active_edge_rebuild_handoff.md`. Do not substitute a current file's
hash for an independently reviewed source identity just to make a resume succeed.

The local research workbench inventories two configured swing experiments:
technical and technical-with-catalyst. It reports the real training
state of every experiment. A `no_candidate` result remains unavailable rather than
being replaced by a fallback model.

```powershell
.\.venv\Scripts\python.exe -m uvicorn `
  market_predictor.research_api.server:app `
  --host 127.0.0.1 `
  --port 8123
```

Open `http://127.0.0.1:8123`. The workbench is non-actionable and does not alter the
production API. Scoring reads only integrity-checked snapshots registered through
the canonical live feature store. Missing, stale, non-finite, or schema-incomplete
features cause a readiness error; the workbench does not substitute zero values or
proxy ETF approximations for missing cross-sectional features.

## Verification

Run one heavy process at a time. Swing candidate training has a 5 GiB hard
process limit; intraday and serving workloads retain their 4 GiB limits.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\ruff.exe check --no-cache .
.\.venv\Scripts\mypy.exe --strict --python-version 3.14 src\market_predictor
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short --branch
```

## Core Invariants

- Features are usable only after their recorded availability time.
- Membership, ticker identity, corporate actions, sectors, and benchmarks are
  point-in-time.
- Market bars are Alpaca SIP with `adjustment=all`; bars and timestamps are not
  imputed.
- Sparse gaps invalidate affected windows. Whole-security exclusions cannot exceed
  5% of the filtered universe; benchmark and market-wide failures are never waived.
- Costs are applied once and benchmarks use the same executable holding interval.
- Validation is chronological, purged, and embargoed. Random cross-validation is
  prohibited.
- Passing software tests does not promote a model. Economic, calibration, drawdown,
  stability, future-shadow, and bundle-verification gates must also pass.

This repository is prediction research tooling, not investment advice or an automated
trading system.
