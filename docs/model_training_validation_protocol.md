# Model Training And Validation Protocol

Status: governing research protocol
Aligned: 2026-09-07

The current long-only swing campaign is governed by `configs/swing_research.toml`
and `swing/contracts/research.py`. Those contracts supersede the retained PDF's
general model sequence and historical test assumptions. Intraday development is
paused; its existing contracts are unchanged. No metadata audit authorizes training.

This protocol makes the repository's implementation choices explicit where the
[quantitative trading plan](references/comprehensive_quantitative_trading_model_implementation_plan_intraday_and_swing.pdf)
states the methodology at a higher level. The source PDF is retained unchanged;
its SHA-256 is
`ea8df4ad6f3c1d17666cadff2672887b01ee46211ef11a040715f9890cb2b75b`.

## 1. Questions And Evaluation Scopes

The primary question is whether a model trained on prior market states ranks or
classifies future opportunities across the contemporaneous tradable universe. A
security may occur in training, validation, and test at different times. That is
not leakage: future rows are never available to earlier decisions.

Two distinct scopes are required; they are not independent calendar samples:

1. **Temporal generalization:** future sessions containing the full eligible
   point-in-time cross-section.
2. **Unseen-security generalization:** a deterministic security holdout nested
   inside development data. This measures transfer to names absent from fitting
   and must not replace the temporal test.

Every row from one decision session belongs to one split in the chronological
evaluation. Chronological partitions group whole sessions, never random rows. The
separately fitted stable 20% security holdout intentionally excludes symbols from fitting;
that transfer stress test is not a substitute for future-session evaluation.

## 2. Swing Dataset And Splits

- Universe: point-in-time S&P 500 membership, including historical members and
  delisted securities where identity and bars are provable.
- Decisions begin 2019-07-09. Earlier bars provide 250-session indicator warm-up
  only, never training examples. The retained historical test year is exposed.
- Decision frequency: one cross-section per exchange session.
- Feature normalization: winsorization, z-scores, ranks, sector-relative values,
  imputation, and feature selection are fit or computed without future sessions.
- Forecast target: `future_excess_return_10d_vs_spy`, fixed next-open to tenth-close
  stock net return minus SPY over that interval. Managed target/stop/timeout returns
  remain separately named outcomes. Managed-exit-session-close benchmark excess is
  approximate, not an exact intraday comparison or the new selection authority.
- Data-quality tolerance: exclude the complete security, with an audited reason
  and affected dates, when its data is unavailable or unverifiable. Continue
  through a maximum 5% loss of the filtered point-in-time universe; refuse above
  5%. SPY, sector-benchmark, and market-wide session gaps cannot use this rule.

The existing temporal config records these XNYS-verified historical partitions;
they do not establish a fresh final test for the new campaign:

| Fold | Fit window | Validation window |
| --- | --- | --- |
| 1 | 2019-07-09 through 2024-05-28 (1,231 sessions) | 2024-06-12 through 2025-06-13 (252 sessions) |
| Historical final refit | 2019-07-09 through 2025-06-13 (1,493 sessions) | none |
| Exposed historical test | none | 2025-07-01 through 2026-06-30 (251 sessions) |

The embargoes are 2024-05-29 through 2024-06-11 and 2025-06-16 through 2025-06-30,
ten sessions each. Development folds inside the permitted fit range must use
actual session boundaries and purge by label availability, with at least ten
exchange sessions of embargo. Fitted preprocessing and calibration precede scoring.
Freeze the bounded settings before validation; reviewed validation is development
evidence, not a renewed final test.

The panel request records retained input history beginning 2018-05-29. The old
May-2019 fit requirement and claimed missing warm-up were stale; do not download
that history again based on superseded prose. Metadata inspection is not full
source replay or independent coverage verification.

The August 9 technical evaluation exposed July-2025 through June-2026 outcomes.
Later zero access counts are per-run facts, not project-wide freshness. Reject
exposed intervals before protected outcome loading. Use a frozen prospective
record; renamed artifacts and changed models cannot restore calendar independence.

## 3. Intraday Dataset And Splits

- History target: one to three years of causally complete sessions.
- Inputs: one-minute bars for executable paths; five-minute or volume bars for
  decision features as frozen by the strategy contract.
- Universe: the point-in-time eligible population for each session, not a current
  static ticker list.
- Session isolation: indicators and normalizers reset at session boundaries;
  overnight observations are explicit context and never extend an intraday
  rolling window.
- Split unit: full exchange session. Every ticker and bar belonging to a session
  remains in the same fold.
- Leakage control: purge by the maximum label duration in minutes and embargo
  the overnight boundary. No intraday position survives the frozen close cutoff.

Intraday and swing folds are separate artifacts. A daily split must not be
silently reused for intraday evidence or vice versa.

## 4. Model Sequence

The deterministic technical composite is a comparison baseline. It cannot veto
all estimators. Its failure rejects that exact formula only.

For this swing campaign only, cross two return regressors (regularized linear and
shallow boosted) with three profiles (existing technical, accepted technical
relationships, and qualified issuer reaction added to those relationships).
At most six learned specifications and two frozen exit policies produce twelve
model/policy comparisons. Blocked profiles remain not trained. No inherited
classifier/ranker/Random Forest sequence or post-result threshold grid is authorized.

SPY buy-and-hold, same-universe momentum and the existing technical formula are
deterministic controls. Continuous return forecasts are not probabilities; binary
views need separately named targets and calibration. AUC is diagnostic, not a
universal veto on the new continuous-return objective. Sector constraints remain
frozen, with no silent weakening to increase trade counts.

Feature diagnostics, redundancy removal, and hyperparameter selection occur
inside fitting data only. Catalyst features start as a causal confirmation,
veto, explanation, or ranking overlay. They enter an estimator only after a
preregistered ablation improves both temporal and unseen-security evidence.

## 5. Sector Treatment

Train one global S&P 500 model first. Include point-in-time sector identity,
sector ETF context, stock-versus-sector residual returns, and bounded
sector-feature interactions. Report every metric by sector and enforce portfolio
sector constraints.

Do not begin with separate sector models. A sector specialist is admissible only
when the sector has sufficient independent sessions and opportunities and beats
the frozen global model on untouched, sector-specific evidence after costs.

## 6. Metrics And Statistical Unit

Rows from one date are correlated, and overlapping holdings also correlate dates.
Report at minimum:

- daily Spearman rank information coefficient and its stability;
- NDCG at the frozen selection depth;
- top-minus-bottom quantile spread;
- selected top-k gross, net, SPY-excess, and sector-excess return;
- win rate, profit factor, turnover, capacity, and maximum drawdown;
- probability calibration and barrier-classification metrics;
- performance by year, market regime, sector, capitalization, and catalyst state.

The research config governs the exact statistical procedure: mean daily portfolio
return minus SPY, one-sided lower confidence bounds, moving-block bootstrap,
20-session primary blocks, 40-session conjunctive sensitivity, and familywise
Bonferroni adjustment over twelve model/policy comparisons. Use its frozen resample
count and confidence settings, not twelve ordinary 95% intervals. Stock rows are
not independent observations. Record every comparison; new selection trials are
not covered merely by logging them. Positive estimates whose bounds span zero are
inconclusive.

Headline economics require funded daily NAV against buy-and-hold SPY on the same
calendar, plus net CAGR difference, costs, cash/exposure, drawdown and attribution.
Unit equity does not establish dollar capacity. Costs occur exactly once under the
new contract's timing convention. Checkpoint 2 accounting remains pending.

[Alpaca's bar documentation](https://docs.alpaca.markets/us/reference/stockbarsingle-1)
declares that `all` adjusts splits, cash dividends and spin-offs. This is declared
provider basis, not independent event-by-event reconciliation of retained price
ratios as total returns. Inspect retained evidence first; do not invent dividends,
double-credit distributions or require speculative bulk redownloads.

## 7. Acceptance And Audit Evidence

Every run must bind immutable hashes for source data, point-in-time membership,
feature contract, label policy, split manifest, model configuration, costs, and
selection policy. The audit must prove:

- feature, event and membership evidence is available by cutoff; outcome labels
  mature afterward and are available before their use in fitting;
- all members of a decision date have one fold assignment;
- warm-up rows do not enter training counts or metrics;
- validation and locked-test results are distinguishable and cannot be
  overwritten;
- fixed-horizon benchmarks use the stock's exact interval; managed benchmark
  approximations and unavailable exact timestamps are explicitly represented;
- the reported policy is reproduced from row-level predictions after costs.

The bounded `swing_research_evidence.toml` audit checks known manifest/config hashes,
canonical feature order and disclosed historical access evidence. It never follows
raw payload references or parses the exposed evaluation. Counts are metadata-only,
not full replay. Five retained specialist manifests contain 60 recorded trials;
duplicates are possible and unenumerated history remains uncovered. This is not
complete lifetime trials or access history, and cannot authorize training/promotion.

Freeze model, sources, feature order, selection, exits and costs before prospective
predictions. The research contract requires one assessment after at least 252
decision sessions and the ten-session maturation tail with no new entries. Future
evidence does not delay historical engineering or training but does constrain
promotion. Do not repeatedly assess until significance appears. Aggregate-only
historical rejections cannot reproduce missing prediction rows; learned attribution
requires immutable chronological out-of-fold predictions, never fitted-row replay.
