# Active Edge Rebuild Handoff

Status: active
Last updated: 2026-09-14
Repository: `C:\project\market-predictor`
Branch: `er-intraday-refactoring`
Last completed implementation checkpoint: `b3b2372` (pushed).
Source-collection checkpoint: `19698d6` (pushed).
The combined historical outcome/features delivery is verified and committed.

## Current State

Completed checkpoint: fixed-horizon training-input diagnostics. Implementation
`b3b2372` is pushed and the final receipt refresh passed. It binds the publication and
saved-row receipt, verifies model order, decision clocks, return arithmetic and
exact ten-session maturity, and reports separate input/supervision availability.
It does not flip any training/production flags or fit an estimator. Independent
reviewer Beauvoir (`01a0a03a-327a-7eb1-828f-93b88aaab260`) closed design/diff review
after both supported P2 findings were reproduced and fixed; the reviewer is closed.
Consolidated focused verification passed 289 tests. Ruff and strict mypy passed
on 397 sources. The real audit exited zero and published
`data/reports/swing_initial_fit_training_readiness_verified.json`, SHA256
`84a56de74eb09b5e1126e6a0ba4aee802d82cf3e0707b9aee3460006d916a824`.
It found 378,037 usable fixed-horizon comparison labels and complete-case
intersections of 194,679 technical / 193,125 catalyst rows, with no row removals.
545 securities appear in the initial-fit interval; 586 is the retained full
campaign list. The current feature audit records exact consumer gaps.

Full regression session **42125** completed with exit zero: 4,202 passed,
10 skipped and 132 warnings in 1,775.89 seconds. Log:
`data/runtime/swing_training_readiness_full_tests.log`, JUnit
`.test-tmp/swing-training-readiness-full.xml`. No workers or review agents remain.
The only source edit after the full suite removed a trailing blank line from
`swing/contracts/training_readiness.py`; no Python statements changed. Session
56011 completed the required current-byte refresh after RAM recovered to 66.5%
(5.25 GiB free). All diagnostic values equal the original receipt; only the
expected implementation pin changed. Log:
`data/runtime/swing_training_readiness_recovered_memory.log`.
The original pre-format report remains immutable historical evidence.
Post-format verification passed all 58
readiness/continuity tests in 11.01 seconds; JUnit:
`.test-tmp/training-readiness-post-format.xml`. Repository Ruff and strict mypy
also passed again. No functional code changed after the full suite.

Current checkpoint: the user now explicitly approves a 90% physical-memory ceiling
without an independent 2 GiB free-RAM floor for swing research. Keep the 5 GiB
process budget and unknown-measurement failure. Configure the audit policy without
editing shared measurement/guard files bound into historical replay evidence.
This change requires its own tests and new immutable audit receipt; it is not
implemented yet. Earlier 85% failures are historical, not the requested policy.

Next-consumer design review completed read-only with Bernoulli
(`01a0a07e-fb40-7031-b2d7-a9166dd328c9`), now closed. Its concrete proposed folds,
learner settings, missingness policy, separate temporal/transfer fits, artifact
contract and risk tests are recorded under "Prepared Next Step: Swing Return
Models" in the active plan. They are not implemented or fitted. The existing
security holdout is an unseeded SHA256 security-ID threshold; keep that assignment
distinct from estimator seed 42. Do not reuse the retained classifier/ranker.

The user's three requested stages are complete:
1. News rebuild: complete, including 59 monthly catalyst authorities.
2. Technical and catalyst feature joins: complete, 59 months / 586,305 rows each.
3. Saved-row audit: passed; final regression run completed with one documentation
   status-marker failure, 4,145 passed and 10 skipped. The documentation-only
   correction passed both continuity tests; no source code changed afterward.

No model training, promotion, or SPY outperformance is claimed. Current scope is
the initial-fit period July 9, 2019-May 28, 2024, not the entire raw archive.
Every frozen decision is retained; unknown coverage and outcomes remain explicit.

The original feature-join worker disappeared after 25 completed months through
July 2021. Its termination cause was not retained. Request and implementation
pins were independently rechecked unchanged. Resume session **41926** uses
checkpoint SHA256
`fe70529c1d8372907e01c1f72337a3f54ffbeb0c29a7b64bfac48bed28ac1474`.
The resumed join exited zero and published its final manifest. Its durable process
log is `data/runtime/swing_research_join_resume.log`.
Output: `data/features/swing_corrected_initial_fit_research`.
Full pytest session **41537** finished with exit code 1 after 2,376.03 seconds.
Log: `data/runtime/completed_news_join_full_tests.log`; JUnit:
`.test-tmp/completed-news-join-full.xml`. The sole failure was the continuity
document status marker, not source code or numerical behavior. No data worker
or reviewer remains active.

## Completed Evidence

| Component | Manifest | SHA256 |
| --- | --- | --- |
| Outcome replay | data/reports/swing_outcome_owner_replay_complete/_manifest.json | 0d7628ccffaf6f5d2a057f452ecfb02fc4feb3963aaf189ee3436802128a4727 |
| Predictor replay | data/reports/swing_predictor_owner_replay_storage_verified/_manifest.json | 0871201090732a30b789c80079a19718d67e8176a6a2a0f843880884a77d0a98 |
| Compact news preparation | data/research/swing_initial_fit_monthly_news_inputs/_manifest.json | a97a981ddaf42e1fceb8def06c17d35bc45915417e61d05cade9a90f40719197 |
| Source-proven news identities | data/research/swing_initial_fit_source_proven_news/_manifest.json | e150475190fe0efbafecb115cd5f08bb7acb58576b2661b680fad832ffc00d98 |
| Corrected monthly catalysts | data/research/swing_initial_fit_verified_catalysts/_manifest.json | 8988773e455eb702a7b1494ed0d1e7f000c4b5a9ff64018fb695798b5e117853 |
| Joined feature profiles | data/features/swing_corrected_initial_fit_research/_manifest.json | 63574b3b55cd30adaebf62b4040f92a2030e6dbb17772014afb3e3c168158256 |
| Independent saved-row audit | data/reports/swing_initial_fit_join_verification.json | 3e4a3c9dd241b9776fc368e7856e45d4bff0ed3cd0ef1431750fa94b16ce53b0 |

Both numerical replays cover 59 months and 586,305 decisions exactly. Outcome
replay additionally verified 20 historical code files and 2,588 live evidence pins;
predictor replay verified 27 historical code files and 5,968 live evidence pins.
These complete receipts are reused unchanged by the feature join.

The corrected monthly catalysts cover all 586,305 canonical decisions, with
224,709 decisions having attributed news. Before the identity repair, a
semantically incomplete generation had only 7,521 such decisions. Keep that
generation as failed diagnostic evidence, not an accepted training input.

News identity alignment translated 854,541 of 957,261 relation rows using 445
proven interval mappings; 102,720 relation rows retained original identities.
These are stored relation counts, not unique news articles or extra stock
exclusions. Separate corrected SATS/FISV collections remain unchanged.
Publication config: `data/research/swing_initial_fit_source_proven_news/monthly-publication.json`,
SHA256 `cf6f9c97737cbf6e3ba0da6fa51c9de6d88ce44461c45c999a0d61df2c2588be`.

## Repairs And Review

- Identity mapping requires exact event-time ticker, positive CIK and overlapping
  pinned source/target intervals. No suffix stripping, bare-CIK matching or new
  stock exclusions. Original events and FinBERT scores are reused.
- The compact writer inferred a naive Arrow clock schema from a null-only first
  slice and stripped later timezone metadata. Original UTC records remained
  intact. Source-proven recovery restored 2,120 identity-clock values exactly.
  All 136 compact batches / 957,261 relation rows passed a full clock/mapping
  preflight. The writer now validates every nonnull UTC representation before
  fixing UTC nanosecond schemas. Original compact values and source hashes remain
  explicit provenance. No assumed localization or precision tolerance.
- Distinct published stories sharing model-input text had 26 decision/window
  groups with different sentiment values and four with different relevance.
  Their numeric difference cause is not established. Exact copies of the same
  durable event/source event still must agree. Separate publications use the
  existing earliest-available representative with its original values intact.
  Named request policy: `exact_event_integrity_earliest_available_text_instance`.
  No averaging, rounding, rescore or numeric tolerance. Request v3 / authority v7
  loaders reject old schemas or missing/wrong policy; no compatibility path.
- Boole `01a09f47-b22a-7162-b67d-a004ad53589d` independently reviewed the clock
  and separate-publication fixes, closed all supported findings, and is closed.
  Earlier identity/replay reviewers are also closed. Do not restart broad reviews.

Verification after clock repairs: 96 focused tests passed; Ruff and strict mypy
passed on 393 source files. After the final story-policy change, 134 integration
tests and 27 final authority tests passed. Final repository Ruff and strict mypy
passed after the last source change. The complete run recorded 4,145 passed,
one documentation failure, 10 skipped and 132 warnings; preserve its actual result
rather than describing it as an all-green run. Only documentation changed afterward.
Both continuity tests then passed in 0.09 seconds; receipt
`.test-tmp/completed-news-doc-fix.xml`. Eight skips require Windows symlink
privileges and two require opt-in memory stress execution. No new passing claim
is made for those skipped cases.
Final continuity and package-boundary verification passed 221 tests in 11.17
seconds (`.test-tmp/completed-news-doc-closure.xml`); all eight local Markdown
links in the six updated documents resolved. No Python workers remained at
closeout. System memory was 73.98% used, with 4.08 GiB available.

## Protected Historical Evidence

Never resume failed generations under changed implementation or rewrite their
hashes. No raw/canonical/provider evidence was deleted.

| Nonexecuted implementation snapshot | Manifest SHA256 |
| --- | --- |
| data/evidence/issuer_news_alignment_nullable_clock/implementation/_manifest.json | 85bf3ab87e5744eb82f23832cf0dc68b0bc05d473e3a243bf6e1bdbe3599422c |
| data/evidence/issuer_news_compact_timezone/implementation/_manifest.json | 45b6c20d633319a20ceb9976ca12535243f276a2d99e5ccc31a4b228b95310c9 |
| data/evidence/issuer_news_compaction_schema/implementation/_manifest.json | 8e82b2bfd959312aed0a3b71080dc72afb4a28fd7db4a46a9622805f22a64b9d |
| data/evidence/catalyst_separate_story_scores/implementation/_manifest.json | 45a2284d768a65d2f4e662ca805a43809467048873bdcb6c046ade2ba808b98c |

Earlier snapshots remain protected by the replay/lineage manifests and feature
audit. Superseded run-by-run instructions were removed from this handoff.
The source archives extend through July 8, 2026, with separate Alpaca/SEC catch-up
through September 10 and partial September 11 snapshots; source collection is
closed in `19698d6`. Do not redownload or include held-out dates in initial fit.

### Completed Historical Components

These publications cover the initial-fit research population. They do not expand
the frozen numeric training boundary or authorize promotion.

| Component | Artifact | Manifest SHA256 |
| --- | --- | --- |
| Corrected outcomes | data/labels/swing_corrected_distribution_scoped_initial_fit/_manifest.json | 7106512bde2c3f737a38216ca70c3b7d3593fe28f5aa78928296856291bb300f |
| Verified-prefix predictors | data/features/swing_initial_fit_verified_prefix_predictors/_manifest.json | 079fed1faf45e1a85efc04acdba5087d704cae838520f9b0a0111b5434e60ce6 |
| Early saved news derivation | data/research/issuer_initial_fit_early_saved_v1/_manifest.json | a6c11d649e1d2a38c8b28dba7db9ee092431f2d581f0da9e0b7c64f8e59bc2f0 |
| Later saved news derivation | data/research/issuer_initial_fit_later_saved_v1/_manifest.json | 445c3c53e94f73de37dd44d2488bd2cd2dea603cd6a3719472fa0afd6b23dce3 |
| Corrected issuer attribution | data/research/swing_issuer_news_corrections_attribution_v1/_manifest.json | ccf3a5ea9a8c057672579be230ece1a7aabac7f4067f12beba3c98d411cbf93e |
| Corrected FinBERT scores | data/research/swing_issuer_news_corrections_sentiment_finbert_v1/_manifest.json | be8cc77ca15fb79555d11ffd36f3737a0a1142a81fa987a7c6465821cb1aad6f |

- Outcomes: 586,305 decisions, 450,273 stock-source-admitted outcomes, 378,037
  complete stock/SPY/QQQ/sector comparisons; peak process working set 0.409 GiB.
  Unknown outcomes are nullable. Managed, training and promotion admission are
  false. Full independent outcome replay and receipt verification passed as
  recorded above; replay parity does not turn unavailable outcomes into labels.
- Predictors: 586,305 rows, 563,326 eligible. Recovered 2,048 earlier ATVI/INFO/SBNY
  decisions without using later bad bars; 1,231 WTW decisions remain unavailable
  because historical bars cannot establish the intended issuer. The 547 good
  parent groups remain byte-identical.
- Failure-fact pin: `configs/swing_predictor_failure_facts.json`,
  `1a59f10180bd93082189476006fd5575b5f3fbb1ee48734c92247d17b642d64b`.
  Observation report: `data/reports/swing_predictor_source_failure_observations.json`,
  `70c6f85029e64ba3354ba04b5aa936c2147272f2f527ca9757517b3d31178060`.
- Early derivation: 148,784 scored rows, 364,564 relations, 29 unavailable chunks.
  Later: 319,787 scored rows, 550,421 relations, six unavailable chunks.
  Counts are stored rows, not unique stories across issuer queries.
- Corrected attribution: 643 direct relations from 661 events; FISV 450/456,
  SATS 193/205. No failures/exclusions; 18 events lack a direct match.
  Historical business labels remain unknown where unsupported.
- FinBERT: exact revision `4556d13015211d73dccd3fdd39d39232506f3e43`,
  local `data/cache/huggingface`. Corrected scoring completed 661 rows / 105 chunks,
  zero failures, peak 1.622 GiB. Set `HF_HOME` and explicit revision when needed.
- Corrected all-adjusted FI/SATS histories: 1,510 daily bars each, collected and
  replayed in `data/raw/swing_corrected_feature_history`; authority pin
  `5d380083e7387bdc6f93a399eca977d9706a0e3ea4bbadc479e200807f635ebf`.
- Full action archive: 570/570 query symbols, 14,352 records, process-date range
  May 29, 2018-September 11, 2026, numeric cutoff May 28, 2024.
  `data/raw/swing_full_cohort_corporate_actions`, semantic audit pin
  `04553f69998bd4a6034c04ffef95238aad9b516eaa0cf0ad6849dc034fedf40c`.
  Acquisition/replay is not ownership, payment-date or historical-availability proof.

## Next Actions

Exact next checkpoint: implement the configured percentage-only swing memory ceiling.

1. The earlier audit is complete. Implement the approved 90% policy and verify
   boundaries, no absolute free-RAM floor, unknown measurements and process guards.
2. Publish a new pinned receipt, compare diagnostics, test/review and checkpoint
   the policy change. Preserve the verified receipt above; do not rebuild data.
3. Implement the objective-specific research training consumer, including frozen
   preprocessing, label policy and purged inner-development folds, before fitting
   the existing technical comparison. The retained trainer is not this consumer.
   Distinct technical-relationship and issuer/SEC reaction profiles remain absent;
   do not relabel catalyst_full as a completed reaction model. Validation/test
   publications are needed for their later evaluations, not as a blanket blocker
   on inner-development training. No fitting or economic evaluation has run here.

Decision config: `configs/swing_corrected_outcomes.toml`,
`ded3b30af1185c7ab0759fa5f81c559c37c590419751c942b61ab479e67c2348`.
Strategy config: `configs/edge_rebuild_strategy_contract.toml`,
`02a087be6b9eff4971770026ca75dce3978f8f9e2028c1f21d027daefec9c0e7`.
Use the existing `materialize-swing-research-dataset` command and explicit
predictor/outcome replay pins above. Resume only with an independently checked
current checkpoint SHA256.

## Splits And Research Rules

`configs/edge_rebuild_temporal_manifest.toml` remains the split authority:
- Initial fit: July 9, 2019-May 28, 2024; 1,231 sessions.
- Validation embargo: May 29-June 11, 2024; ten sessions.
- Validation: June 12, 2024-June 13, 2025; 252 sessions.
- Final refit: July 9, 2019-June 13, 2025; 1,493 sessions.
- Final embargo: June 16-June 30, 2025; ten sessions.
- Historical test: July 1, 2025-June 30, 2026; 251 sessions.

250-session warm-up, ten-session horizon. Twenty-percent unseen-security holdout
is a separate transfer test, not a replacement for time splits.
The historical test has already been exposed; it cannot be called untouched.
Newly downloaded old news is not prospective prediction evidence.
Fresh prospective promotion requires the frozen policy and new matured predictions.

The approved population remains 45/631 excluded, 586 retained, under a cumulative
10% ceiling. Do not add exclusions or remove losing outcomes. Event-aware accounting
and explicitly simulated ordinary trades are approved; no broker receipt is needed
for a hypothetical trade. Unknown corporate payouts and contingent rights stay null.
Costs are applied once; stock and SPY/QQQ/sector outcomes use matched intervals.
Old and new adjusted-price vintages cannot be blindly concatenated.

## Retained Source References

Preserve raw archives and independent pins. Historical checkpoint narration and
review details remain in Git; this document records the current continuation.

| Source | Path | Independent pin |
| --- | --- | --- |
| Approved population | data/reports/swing_research_cohort/approved_research_population_audit.json | 41de559dd1c415dab60771e10fd489150853a6c2fff07e387d1024b33961efe0 |
| Holding identity | data/reports/swing_research_cohort/holding_identity_preflight.json | 8f8cdd60ca2f9c1372ceda20d04b7f90eefbfb2336a7f282f30aa97b8c7e723b |
| Raw initial-fit plan | data/reports/swing_initial_fit_raw_share_plan | d912a997af361c820745e8c850f0fd455ee068397c6d034d2b54c00a475dc22e |
| Raw initial-fit archive | data/raw/swing_initial_fit_raw_share_daily | 144cab43741f3c74308b53e9322c84ac7eeaca158d3cbd6a1ddb0d7c0fa8d244 |
| Corrected raw source archive | data/raw/swing_symbol_corrected_daily | fb92efc1df48adc8f03d8bfff47d9c811975428b5a5903fdc03f2567c5b47970 |
| Corrected source selection | data/reports/swing_symbol_corrected_sources.json | 01153e33e8b6a161c04fde2dbea8021eeea369fef6988868c38b49f0e0c065ee |
| Corrected issuer news manifest | data/raw/swing_issuer_news_corrections/_manifest.json | 4da07ce57bf864845b79b0043816e49c104dc047eb81a30372ad952519f855d8 |
| Original adjusted bars manifest | data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3/_manifest.json | b99a1d13d9075220db30c3870bc0796b3631bf4ec0bf6424469feeeec9115d93 |
| SEC original archive manifest | data/external/sec_filings_20190709_20260708_v1/_manifest.json | e75f6c4dd0b4a392e35ba7033c581c5f8b748a7fbbfc5bb847b4e76b6863cd35 |
| SEC query relation | data/canonical/sec_identity/sec_identity_20190709_20260708_v2/sec_identity_relations.parquet | 507043d1d60d9b17db26727a5aeff9bfa23ae2962cc9d80d5637a1ba000a9720 |

Directory pins above are authority-file hashes, not directory hashes.
The 21 SEC completion documents are retained under
`data/raw/swing_cash_merger_completion_documents`,
`swing_share_transition_completion_documents`,
`swing_remaining_merger_completion_documents`.
Their semantic replay pins respectively:
`29ee1cb81230136d7c77dccf81a9a418a58dfe2ab60dd74aec99212cb8160495`,
`680686010346616e0e433681360519f2b69665a062f242c0ac50d83c526d9bc8`,
`194ee145ded436211a3a0a3edbd10629ac7de903edcc31ea2d46e8c28adaab30`.
Reviewed terms and locators are in the feature audit. ABMD CVR valuation and
other unsupported delivery/payment facts remain unknown.

## Working Rules

Read AGENTS.md, the active plan and
`docs/reviews/feature_engineering_audit_20260801.md`.
Use the project `.venv/Scripts/python.exe`; pytest needs a unique
`--basetemp=.test-tmp/<run>` and `-p no:cacheprovider`.
Heavy jobs use one shared lease, system-memory guards and sequential execution.
Track and close owned agents/processes; never kill unrelated applications.
No cloud/security deployment work, intraday development, alerts or broker orders.
No new model has been fitted or proven profitable by this delivery.
