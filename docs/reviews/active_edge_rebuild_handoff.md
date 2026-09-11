# Active Edge Rebuild Handoff

Status: active
Last updated: 2026-09-11
Repository: `C:\project\market-predictor`
Branch: `er-intraday-refactoring`
Last completed implementation commit: `19698d6` (pushed).
Last completed source-collection checkpoint: `19698d6` (pushed).

## Current State

The source-collection checkpoint is committed and pushed. The September 11
historical outcome/feature delivery remains uncommitted. Do not describe component completion as a trained
model, full dataset admission, or SPY outperformance. The only active execution
plan is `docs/active_edge_rebuild_plan.md`.

The user redirected the immediate checkpoint to extending retained sources through
today, repairing broken scheduled collection, and preserving chronological splits.
Historical monthly feature preparation is paused, not restarted or completed.

### Source Collection (Closed)

Implementation `19698d6` is pushed. Coordinators are
`swing/datasets/alpaca_incremental/` and `swing/datasets/sec_incremental.py`;
HTTP transports stay in `sources`, and the canonical SEC parser stays in
`catalysts/sec_filings/collection.py`. No aliases or alternate data format.

- Alpaca: July 9-September 10, 2026, 26,638 raw and 26,638 adjusted daily bars,
  23,025 stored news records; zero failed/pending units. All 2,856 units,
  including revision and partial captures, replayed without HTTP after relocation.
  Status: `data/raw/swing_incremental_alpaca/status.json`.
- September 11 Alpaca capture: 589 bars per format and 112 news records, partial
  only. Publication, update and retrieval clocks stay distinct; revised text is
  not backdated. Original closed-day bytes and three-day revision captures remain
  separate. Query symbols are 670 old codes plus RHT, not current membership.
- SEC: 20,723 new filing metadata records across all 624 CIKs, zero failed issuers;
  closed-day checkpoints through September 10. Today's partial capture is frozen
  at September 11 14:52:22.734419 UTC. This is submissions metadata, not every
  filing body/exhibit. Old identity relations are query hints, not ownership.
- SEC online report: `data/raw/swing_incremental_sec/_runs/df17090af13b459fb1841e7d67356bcc.json`.
  Latest relocated offline replay:
  `data/raw/swing_incremental_sec/_runs/9f950225d2474fc78af483263e8fe19a.json`.
  No original seven-year archive was redownloaded or overwritten.
- Midnight task `MarketPredictorSwingCollectionMidnight` is installed.
  It runs Alpaca then SEC while signed in/on AC power; no automatic training.
  Latest real offline scheduled execution passed at 18:27:24 local, exit zero;
  normal action restored. Evidence:
  `data/runtime/scheduler_migrations/20260911T165348/execution_verification_after_owner_move.json`.
  The two broken tasks were exported before retirement.
- Obsolete, reference-free scripts removed: `scripts/build_intraday_v3.py`,
  `scripts/collect_1m_intraday.ps1`. Raw evidence was not deleted.
- Reviewed portability fix: 13 unchanged readiness-audit files (233,806 bytes)
  are a checked-in test fixture. Both test paths use it; Git preserves exact
  hash-bound bytes, including with autocrlf enabled. No assertion was weakened.
- Verification: 299 focused source/boundary tests; 27 readiness tests; Ruff;
  strict mypy on 360 source files; isolated complete suite **3,360 passed,
  five skipped, 133 warnings**, 1,347.84 seconds. XML:
  `.test-tmp/source-owner-full.xml`. Exact verified Git tree:
  `c57e401ab7c46539caa828e94dbb62bb46753613`.
- Isolated skips: two canonical-data checks (no copied market data), two opt-in
  production-scale memory tests, one Windows symlink-privilege test.
  The real main-workspace canonical hash checks separately passed: **nine tests**.
  Full production-scale RSS and privileged symlink evidence are not claimed.
- The complete suite was verified against the exact source-only commit tree,
  not the mixed worktree. The latter has the historical dependency failure below.
  Source-layer orchestration violations found during verification were fixed;
  historical hash-bound modules were not edited to fake a pass.
- All collection, replay, test sessions and review agents are closed. Source
  collection used about 115-175 MB; observed test working sets stayed below
  350 MB. System-memory guards remain enabled; heavy jobs stay sequential.

### Verified Saved History

Parquet footers were checked for all 7,331 saved bar/news files; row counts matched
their manifests. This is a byte/row inventory, not semantic issuer attribution.

| Source | Directory under data | Saved observations |
| --- | --- | --- |
| SIP adjusted daily bars | raw/swing_daily_sip_sp500_pit_20190709_20260708_v3 | 1,088,146 rows, 670 files/symbols |
| Early Alpaca news | raw/alpaca_news_20190709_20210708_v1 | 149,140 rows, 4,047 files |
| Later Alpaca news | raw/alpaca_news_20210709_20260708_v1 | 564,986 rows, 2,614 files |
| SEC submissions metadata | external/sec_filings_20190709_20260708_v1 | 689,467 filings, 624 CIKs, zero failed issuer requests |

The 714,126 saved news records are not globally unique articles or proof of
every ticker's complete coverage. Actual news publication range is July 9, 2019
through July 8, 2026. Empty provider responses remain distinct from failed requests.
The SEC archive is pinned and checked without loading all filing rows at once.

The old `data/external/market_context` summary is stale. Its Parquet has 31,539
records over June 2024-June 2026, including ETF-tagged news; it is not a verified
seven-year global-news authority and is not an input to this new collector.
GDELT's saved failed/rate-limited query is not completed historical coverage.
Reddit and Seeking Alpha remain retired. Do not add them back.

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
  false. The complete independent outcome `--replay` has not yet run.
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

All prior historical collection, derivation and predictor sessions are finished.
Monthly preparation session 40768 stopped at its memory guard (84.2% used,
2.49 GiB free against this command's 82% / 3 GiB requirement); no monthly authority
was published. Do not repeat completed source downloads.

## Exact next checkpoint:

**Canonical swing archive ownership and publication replay.** The design is
reviewed; implementation has not started. No data recollection is needed.
Fix this before resuming monthly joins or model training.

The mixed-worktree suite stopped after 1,537 passes and two skips on
`test_package_dependency_boundaries.py`'s swing dependency direction assertion.
Four uncommitted files import `edge_rebuild`:
`swing/datasets/corrected_outcomes.py`, `research_features.py`,
`predictor_abstention_derivation.py`, and `swing/features/adjusted_source.py`.
They consume the complete history archive reader and two session helpers.

These four files are directly pinned by completed publications. Even import-only
edits invalidate resume/replay identity. The failure is architectural, not proof
of wrong numerical results. Do not weaken the test, hide imports, duplicate code,
add forwarding aliases, or silently replace artifact pins.

1. Preserve the exact implementation bytes required by existing publications
   before editing. Record their independent hashes and current artifact bindings.
2. Move the archive reader's real dependency closure and session-requirement helpers
   into genuine `swing/datasets` owners. Update all actual consumers. Keep one
   implementation and the existing semantics.
3. Run dependency boundary tests before any expensive artifact build. Verify
   unchanged session sets, source checks, future-poison behavior and numerical
   outputs, then publish an explicit derivation/replay path for existing artifacts.
4. After that gate passes, resume monthly preparation estimate:
   `.venv/Scripts/python.exe -B -m market_predictor.catalysts.issuer_events.monthly_preparation --root C:/project/market-predictor --config configs/swing_initial_fit_monthly_news.json --expected-config-sha256 2a9ac2ae30bcddc75b89ce713d811a8d8fe59e3ed186d5e90f81115df901b2b1 --out-dir data/research/swing_initial_fit_monthly_lineage_inputs --estimate-only`.
   Run actual preparation, publish through `monthly_authority` using its returned
   config/pin, and join the final research dataset.
5. Independently replay outcomes and joined features. Review the feature acceptance
   matrix before six sequential candidate fits. No new model has been trained.

Independent reviewer Planck approved this bounded sequence and is closed.
The original source archives and published rows must remain intact.

Primary integration configs: `configs/swing_corrected_outcomes.toml` (pin
`ded3b30af1185c7ab0759fa5f81c559c37c590419751c942b61ab479e67c2348`),
`configs/swing_corrected_research_features.toml`,
`configs/swing_initial_fit_issuer_derivation.toml`,
`configs/swing_issuer_feature_rebuild_inventory.toml`.
Relevant owners: `catalysts/issuer_events/monthly_preparation.py`,
`monthly_authority.py`, `swing/datasets/research_dataset.py`,
`research_features.py`, `corrected_outcomes.py`,
`swing/features/research_join.py`, `predictors.py`.
Keep hash-bound implementation files unchanged while their job is running.

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
