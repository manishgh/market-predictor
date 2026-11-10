# Market Predictor: research brief for Fable 5.2

Prepared November 10, 2026. This static research brief describes the project's purpose, retained evidence, measured results and unresolved research questions; it is not a replacement execution plan. It requests an independent research assessment, not endorsement of a model. Facts below come from repository documents, source code, saved manifests and independent project audits. No external literature review has been completed for this brief.

## 1. Purpose and desired outcome

Market Predictor is a Python prediction engine for long-only swing decisions and a separate longer-term investment cohort. The swing research horizon is ten exchange sessions. The current investment experiment uses 252 sessions as its primary horizon and 63 sessions as a diagnostic horizon. These are separate prediction tasks; a result for one cannot establish quality for another.

The intended product supplies evidence-backed forecasts through an API: the prediction target and horizon, expected benchmark-relative return or appropriately defined probability, input availability, uncertainty and reasons for abstention. The companion C# application, TradingFlow, owns alerts, broker execution, orders, positions, portfolio risk and final sizing. Market Predictor does not acquire those responsibilities merely because its research evaluates hypothetical investments.

The economic objective is useful stock selection that can ultimately demonstrate net outperformance of an appropriate SPY holding over matched intervals. Evaluation must account for dividends, distributions, splits, mergers and distributed securities where the holding actually owns them. A price-only comparison can misstate investment performance. News is explanatory information available at a decision, not a substitute for total-return accounting. Neither a good AUC nor a low forecast error alone proves that an implementable portfolio beats SPY.

The system is not deployed. The immediate objective is to explain disappointing measured forecasts, complete one justified improvement experiment, and preserve a stable comparison task. The user has also requested a later API arrangement: a verified earlier strong model as V1 and the current experimental model as beta. That is an ordered follow-on after research and evaluation. The earlier model has not yet been identified conclusively, and this request does not turn a historical candidate into an accepted live release.

## 2. What has been built and verified

The project already contains collection, immutable source archives, security identity and membership handling, causal feature construction, outcome accounting, training, saved predictions, research APIs and release controls. The central need is now predictive improvement. Generic proposals to rebuild the platform, recollect everything or change programming languages would miss the problem.

Retained inputs include stock and benchmark candles, issuer news and sentiment, SEC filings and financial facts, corporate-action evidence and benchmark distributions. Alpaca issuer news and SEC evidence have explicit source and availability semantics. Finviz has separate context roles where documented. Reddit and Seeking Alpha are retired dependencies and must not be proposed as existing inputs. Historical evidence has different admission levels: a preserved file or reproducible transformation is not automatically a production-ready source.

The investment dataset reuses the existing nightly 209-column snapshot. At each investment decision it selects exactly the preceding XNYS session's snapshot, preserving the original values, security identity, feature order and availability clocks. It does not select the latest surviving older row if the exact session is missing. Existing peer-relative values retain their original 586-security reference population. Six accounting measures are appended at the exact investment security/decision/close. There is no reconstruction of news windows or technical indicators for this experiment.

The two fitted profiles are 124 technical/relationship columns and 215 total columns: the same 124, plus 85 news columns, plus six financial columns. Technical inputs represent price/volume behavior and relationships to benchmarks and peers. Examples include return_20d_xs_rank, rel_return_5d_vs_spy_xs_z, residual_return_20d_vs_sector_xs_rank, rsi_14_xs_rank and realized_vol_20d_xs_z: cross-sectional ranks or standardized measures of momentum, benchmark-relative behavior and volatility. News features include event-category presence and directional shares, sentiment, source coverage, story age and limited interactions, principally over one- and three-day windows. Concrete examples are news_direct_earnings_present_3d, news_direct_contracts_partnerships_favorable_share_3d, news_direct_finbert_positive_mean_3d, news_direct_latest_age_hours_3d and news_positive_sentiment_times_long_trend_3d. These names identify existing saved inputs; they do not imply that every story's interpretation is correct.

The six financial columns are revenue_growth_ttm, operating_margin_ttm, operating_margin_change, fcf_margin_ttm, capex_to_revenue_ttm and cash_conversion_ttm. They describe trailing revenue growth, operating margin and its change, free-cash-flow margin, capex/revenue and operating-cash-flow/net-income conversion. They are reconstructed accounting measures for nonfinancial issuers, not a comprehensive valuation or earnings-surprise system.

Financial coverage is incomplete and preserved explicitly. Across 22,346 eligible financial decisions, 6,017 have all six measures, 15,485 have at least one and 6,861 have none. These counts describe financial decisions, not training rows or news articles. The audit checked 134,076 stored metric positions, including their availability/missingness semantics.

The completed investment target corpus contains 74,458 horizon rows for 37,229 monthly decisions. Of 44,692 in-scope horizon requests, 27,189 are source-validated and 17,503 retain explicit unresolved source/comparison gaps; another 29,766 rows are outside scope. The validated counts are 14,348 at 63 sessions and 12,841 at 252 sessions. Original checkpoint bytes and producer lineage were preserved. Independent review checked all row identities and origins, plus selected independent holding arithmetic; it did not independently recompute every economic value in the corpus.

Targets use the next session's open and the exact horizon ending session. The stock-versus-SPY target is stock net return minus matched SPY gross return, with the frozen stock cost applied once. The target implementation requires a 20-basis-point prepaid stock cost. Accounting follows canonical holding evidence; unsupported payouts or ownership transitions remain unknown rather than being replaced with convenient prices. The research request does not authorize changing reinvestment, tax or distribution conventions without inspecting their contracts.

The joined dataset retains all 74,458 rows. Per horizon, 22,334 rows have usable features. Verified target plus feature eligibility yields 12,834 annual and 14,341 quarterly training-eligible rows. An independent pre-fit audit checked 5,874,781 feature values and the same number of availability values, all 27,189 verified target bindings, 46 original news-producer source files and 2,430 consumed file hashes. All twelve declared horizon/year/scope split definitions passed.

These are retrospective research admissions. Historical information cutoffs remain distinct from actual November 2026 verification, fitting and scoring clocks. No historical first-seen evidence or prospective deployment is invented by backdating those operations.

## 3. Completed investment experiment

The frozen experiment completed 48 validation fits and eight final research models: two horizons, two feature profiles, two learners, two validation scopes and three score-year folds; final refits comprise the eight horizon/profile/learner combinations. The independent post-fit replay passed November 10 at 20:07:44 UTC. Saved-model prediction replay residual was exactly zero. Split membership, preprocessing, metric calculations, source identities and execution clocks were independently checked.

The learners were fixed Ridge regression and shallow XGBoost regression, reused from the retained settings rather than selected through a new parameter search. Ridge uses alpha 1, an intercept and the LSQR solver. XGBoost uses squared-error loss, 200 trees, maximum depth 3, learning rate 0.05, L2 penalty 10, full row/column sampling and one worker. Their existing numerical settings were not optimized for these investment horizons.

Preprocessing fits separately within each training fold and validation scope: training medians, explicit missingness indicators, and training-only scaling for Ridge. Every input receives a missingness indicator, so 215 input columns produce 430 encoded columns. Dates receive equal total training weight. The models emit uncalibrated return estimates; no additional return-calibration layer has been fitted.

Temporal folds score 2022, 2023 and January–May 2024, with expanding training history beginning July 2019. Training outcomes must mature strictly before the boundary 21 XNYS sessions before the first scored monthly decision. A separate deterministic 20% security holdout excludes held-out security identities from preprocessing and estimator fitting. This remains retrospective development evidence, not an independent future test.

Minimum admission requirements are twelve training months, 100 training rows and twenty securities, plus three scoring months, thirty known scoring rows and ten securities. The first annual temporal fit contains only seventeen training months and 2,678 rows. Monthly annual outcomes overlap substantially; thousands of rows do not represent thousands of independent market histories.

Evaluation uses equal-date MSE and MAE against a zero-SPY-excess forecast, monthly rank correlation and a frozen top-decile cohort. Each date selects ceil(number scored / 10), descending predicted excess with security-ID tie breaking. Selection happens before looking at future outcome availability. Missing selected outcomes cannot be dropped or replaced. There is no positive-score filter, best-model selection, funded portfolio return, CAGR or Sharpe claim in this experiment.

## 4. Results and what they establish

All sixteen aggregate horizon/profile/learner/scope MSEs and MAEs are worse than predicting zero SPY excess. Adding news and financials worsens seven of eight paired MSE comparisons and all eight MAE comparisons.

| Horizon | Temporal XGBoost profile | MSE above zero-excess baseline |
| --- | --- | --- |
| 252 sessions | 124 technical/relationship features | 27.81% |
| 252 sessions | 215 full features | 33.81% |
| 63 sessions | 124 technical/relationship features | 5.22% |
| 63 sessions | 215 full features | 4.85% |

The small quarterly XGBoost improvement is approximately 0.35% relative to technical-profile MSE, while remaining worse than zero excess. These percentages describe prediction-error differences, not realized investment losses.

Each temporal specification scores 11,091 rows over 29 monthly dates. Known outcomes number 7,126 annually and 7,862 quarterly; respectively 3,965 and 3,229 scored outcomes remain unknown. Security-transfer specifications score 2,100 rows each. Every temporal top-decile cohort contains at least one unknown outcome: zero of 29 monthly cohorts is complete for every temporal specification. Complete selected-cohort mean returns therefore remain unavailable. Reported errors and ranking measures describe the observed-outcome subset, with that limitation attached.

Ranking and magnitude need separate investigation. Annual temporal Ridge mean monthly rank correlation rises from 0.08472 with technical features to 0.11958 with full features even though MSE worsens. This suggests a useful diagnostic question, not a proven recoverable edge. Annual XGBoost rank correlation instead falls from 0.07331 to 0.06646. Security-transfer evidence is weaker still. No model was promoted, activated, retuned or selected after observing these results.

## 5. The remembered earlier AUC near 0.67

The user remembers a stronger model some months ago; the reference does not identify an exact month or task. Several retained results fit that description. Do not silently assume the user means July, a four-month forecast or a four-month evaluation window.

| Candidate | Saved result | Important distinction |
| --- | --- | --- |
| July 8 next-day big-up classifier | AUC 0.665665; 47,904 validation rows; 499 tickers; 110 features | January–July 2026 history; model file preserved. Historical label default is a 3% next-open-to-close gain, but run-specific overrides are not pinned in the manifest. |
| August 2 intraday HGB | Temporal AUC 0.667580; unseen-security 0.663132 | Thirty-minute target-before-stop/timeout classification; selected mean net return approximately minus ten basis points. Saved bundle model is selected logistic, not this HGB. |
| August 4 intraday HGB | Temporal AUC 0.667583; unseen-security 0.663375 | Same task, expanded technical features; little AUC change. Saved bundle model is selected logistic, not this HGB. |
| August 20 rating-upgrade specialist | Unseen AUC 0.666184, chronological AUC 0.436840 | Rejected; no fitted model in its manifest. A single favorable scope does not establish robustness. |

A separate July five-day volatile mover reports AUC 0.712604 and retains its model file. It was promoted under earlier gates; that historical status does not establish current readiness or net SPY outperformance.

Git records a material continuity gap. July 21 commit 53c51c2 removed the volatile-model routes and introduced a different five-session net-positive task before a real canonical replacement had been trained/promoted. Later work moved through ten-session swing classification and SPY-excess regression to quarterly/annual investment regression. No later same-target next-day manifest was located. This does not prove monotonically declining quality in one unchanged estimator; it shows the earlier task was not maintained as a continuously measured baseline.

There were genuine methodology repairs. July 21 commit 5ebfa83 removed current-snapshot SEC/rating/surprise fields from historical defaults; July 22 strengthened cutoffs and causal validation. Their effect on the July AUC is unmeasured. Do not declare that score fraudulent, inflated or profitable without a matched replay. August 7 documentation claimed a fix for “Intraday 0.66 AUC” through ranking and z-scoring, without a measured improved fit in that checkpoint. Implementation changes did not establish improved predictions.

## 6. Current diagnosis and research hypotheses

The first diagnostic completed on November 10 at 20:43:44 UTC using the 48 saved validation prediction sets and pinned dataset. Its implementation passed 57 focused tests, affected Ruff and strict mypy checks, and independent quantitative/code review, and is committed on main as 770d0db3dd3099bde109edf404b6fe41b106f5b8. Independent Quant review checked the completed report's hash, counts and numerical identities, with reconciliation residuals below 3.2 × 10⁻¹⁷. No estimator was loaded or refitted, no predictions were modified, and no calibrator was fitted. This adds explanatory evidence, not a newly improved model.

The report separates ordering from magnitude using forecast/target moments and AUC for “net SPY excess greater than zero.” Within-date AUC describes ordering stocks at the same date; pooled AUC also includes between-date comparisons. Neither recreates the earlier classifier's target. The main observed annual-horizon findings are:

1. **Forecasts are systematically too optimistic on the known-outcome temporal subset.** Across 7,126 rows and 29 equally weighted dates, full-feature XGBoost predicts mean excess return of +7.50%, while observed mean excess is −7.11%: a +14.61-percentage-point mean error. Full-feature Ridge predicts +9.89%, giving +17.01 points of error. These are forecast errors, not portfolio returns. Date-mean errors account for 21–27% of annual temporal MSE; the remaining 73–79% is within-date error. Bias correction alone cannot solve the whole problem.
2. **Some temporal ordering survives, but transfer is weak.** Mean within-date AUC moves from 0.55755 to 0.56673 for Ridge and from 0.55543 to 0.54964 for XGBoost when moving from technical to full inputs. Full-feature security-transfer AUCs are only 0.51723 and 0.51869, respectively, on 1,285 known outcomes. The findings do not establish statistically reliable improvement.
3. **Feature-induced forecast changes are too large for their corrective benefit.** Annual temporal full-minus-technical MSE is 0.00497770 for XGBoost: squared forecast changes contribute +0.00557825, offset by only −0.00060055 of interaction with baseline errors. Ridge's corresponding increase is 0.01323757, from +0.01470633 and −0.00146876. These are squared-return units, not return percentages. Forecast dispersion is below target dispersion; the concern is magnitude relative to weak predictive association, not forecasts being more volatile than outcomes. This identity does not isolate a causal news or financial effect.
4. **The annual temporal deterioration occurs in all three scoring years**, with the largest aggregate contribution in 2022 for both learners. It is not confined to one losing year, although the short, overlapping development history limits generalization.
5. **Coverage alone does not explain the deterioration.** For annual temporal XGBoost, the all-six-financials stratum contributes 55.83% of net MSE deterioration at 30.99% of evaluation weight. Harm is not confined to rows missing financial measures, but news changed simultaneously and training behavior can transmit across strata. The partial-news bin, with one to 84 finite columns, carries 97.96% of annual temporal weight. That is not a measure of story quality or a causal explanation. Unknown outcomes and incomplete selected cohorts remain unresolved.

Exact attribution uses MSE(model) minus MSE(zero) = E[p²] minus 2E[p·y]. For full versus technical forecasts, the change equals E[delta²] plus 2E[delta·(p_technical minus y)]. Subgroup contributions retain global date weights. Descriptive slopes and retrospective decompositions are not fitted calibration rules or proof that removing measured bias will improve future predictions. Tail concentration cannot justify deleting losing examples.

Priority hypotheses, not proven causes, are: oversized or biased forecasts; insufficient independent history for model flexibility; one-/three-day news windows mismatched to annual returns; sparse financial coverage or changing missingness patterns; accounting ratios lacking valuation/expectation context; target tails; and unstable relationships across years or securities. The full-versus-technical comparison changes news and financials together, so it cannot identify either group's separate contribution. Software replay success rules out some implementation discrepancies; it does not certify economic signal or eliminate selection bias.

The current Quant recommendation is the bounded 209/130-column factorial comparison before calibration, to separate source-group effects. It requires 48 new validation fits and reuses the old 48; no final refits are proposed. This recommendation is not yet a frozen or implemented experiment. Fable 5.2 should challenge it against the measured evidence and recommend one justified next step.

## 7. Research requested from Fable 5.2

Please answer the following in priority order, distinguishing repository evidence, mathematical implications, literature evidence and conjecture.

1. Define stable, measurable objectives for swing and investment forecasts. Explain which role belongs to classification, cross-sectional ranking and conditional return regression, and how to compare them on the same dates, securities, costs and economic targets. Avoid replacing the task simply because another metric looks better.
2. Interpret the completed decomposition: how much evidence supports useful ordering plus magnitude error, weak transferable signal or feature-induced instability? Specify only the additional evidence needed to distinguish these explanations. Account for missing outcomes and overlapping horizons; do not prescribe repeating the completed diagnostic.
3. Recommend exactly one next experiment based on these findings. A bounded factorial option adds 209-column technical-plus-news and 130-column technical-plus-financial profiles while reusing the existing 48 validation fits for the 124/215 comparisons. Repeating the declared matrix for those two new profiles requires 48 additional validation fits: two profiles × two horizons × two learners × two scopes × three score years. No final refits are included unless separately frozen. Define its hypothesis, unchanged controls and rejection criterion before running it. Do not expand into a parameter sweep.
4. Evaluate causal shrinkage/calibration as an alternative, not an automatic second experiment. Any coefficient must learn from strictly earlier, purged out-of-fold predictions and labels matured before the scoring cutoff. In-sample residual fitting or calibration on the evaluation period cannot demonstrate improvement. Early annual folds may lack sufficient history; justify a separate calibration minimum before seeing its fitted result, or report unavailable evidence. Security-transfer calibration must also exclude held-out identities.
5. Explain how to identify and preserve the earlier strong baseline without depending on the user remembering filenames. Specify the artifact/input/target checks and a matched comparison bridge. Historical HGB metrics without a saved HGB estimator cannot support immediate API wiring.
6. Propose go/no-go rules that separate better forecast error, better ranking and better complete-cohort economics. Include uncertainty appropriate to correlated stocks and overlapping monthly returns, evidence needed beyond repeatedly inspected development periods, and a stopping rule when results remain unconvincing.

An acceptable improvement must be evaluated across dates and securities, with forecast calibration and both temporal and transfer results visible. Its claimed economic advantage must survive declared costs and a predefined cost stress, with complete selected-cohort accounting or an explicit reason why economics remain unavailable. Improvement in observed-subset MSE alone cannot clear that requirement. A stable ranking improvement with poor return calibration should be described accurately rather than advertised as calibrated expected returns. The repeatedly inspected historical periods remain development evidence; a frozen candidate ultimately needs genuinely new, uninspected shadow predictions and matured outcomes before a prospective success claim.

Return a concise diagnosis followed by one fully specified experiment protocol, expected failure modes, resource cost, required attachments and an explicit list of uncertainties. Cite primary research or official technical documentation for external claims and explain applicability to this dataset. Do not promise profitability, recommend broad source reconstruction, introduce unsupported alternative data, or substitute a catalogue of model families for a falsifiable experiment.

## 8. Execution constraints and evidence attachments

Only C:/project/market-predictor is the active checkout, with main retained. Preserve existing sources, models, failed-run receipts and hash-bound evidence. Reuse unaffected partitions; reopen foundations only for a demonstrated numerical, availability or contract defect. Heavy jobs run serially under the shared lease, strictly below 90% system RAM, with the defined five-GiB process budget and 0.75-GiB headroom checks. Unknown measurements refuse execution. Focused tests, affected lint/type checks and independent quantitative/code review precede claims about actual-data results.

Local paths are evidence locators, not attachments accessible to Fable 5.2. Supply the small manifests, model cards, exact requests/settings, audit summaries, completed diagnostic report and selected historical Git excerpts separately. For numerical investigation, additionally provide projected saved out-of-fold predictions with identities, dates, horizons, scope, forecasts, targets, eligibility and missingness reasons; feature-group coverage; and frozen split/weight definitions. Do not upload credentials or entire raw archives by default. Model binaries are unnecessary for initial diagnosis. Distinguish code/design review from any subsequently completed independent numerical replay of this new diagnostic.

The authoritative current documents are AGENTS.md, docs/active_edge_rebuild_plan.md, docs/reviews/active_edge_rebuild_handoff.md and docs/reviews/feature_engineering_audit_20260801.md. Leading current sections override historical appendices. Important artifact locators, relative to the canonical repository unless indicated, are:

Full target corpus. Path: data/research/investment_full_targets_20261109/corpus_0a1ca7d/final/_manifest.json.

SHA-256: 9b70c348ab17738f5874d42c51bff6af386d94061ef383385bdd6ead5bcad4c3

Joined dataset. Path: data/research/investment_dataset_20261110/joined_30b54bf/dataset/_manifest.json.

SHA-256: 9c19fbf3a9f716b3f82b9958c094f9752f477952f5df8c4cb797123e765c1280

Completed training. Path: data/research/investment_training_20261110/models_30b54bf/_manifest.json.

SHA-256: 16eda742f2b6c0b95d25c03796185aac447c6c348f6710f6b7037640675516b3

Independent post-fit audit. Path: Chat outputs/investment-models-independent-postfit-audit-20261110.json.

SHA-256: b139c5e84d98c332aec3598c4751dedb35b5d4d8f2b414630604839cd94404dd

Completed prediction-error diagnostic. Path: data/research/investment_prediction_errors_20261110/diagnostics_770d0db/_manifest.json.

SHA-256: 2e570a4b714e0305d11909fe23ee48eeb116114be42ada16331070a92ce86981

Historical attachments should include models/sp500_6m_next_day_big_up_v2_20260708_candidate.joblib.manifest.json; the corresponding next_week_big_up manifest; models/edge_rebuild_intraday_candidate_20260802_v1/evaluation.json; and data/models/edge_rebuild_intraday_candidate_v3/evaluation.json with their model cards. Relevant Git excerpts are 87b3aad for the contemporaneous baseline record, 53c51c2 for the workflow replacement, 5ebfa83 for snapshot-feature removal, 0332b16 for intraday V3 completion, and 3b2bff5/7bb128c for the later SPY-relative objective and measured swing regressions.
