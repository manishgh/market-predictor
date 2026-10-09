# Current Feature Engineering Audit

### Step 2 execution result — October 9

Software 189242f adds only research/news_feature_enrichment.py and its unit tests.
The component produces lexical category/direction/status cues with authentic source
quotes, source/version clocks, true missing sentiment and a QA bucket adapter.
It does not assert verified world events, stock attribution or stock-return direction.
Consolidated review fixed integer-sentence direction leakage and chunked-negation
loss. Final 64 enrichment + 28 sampler + 2 continuity UNIT/document checks passed
(94 total); targeted Ruff/strict mypy passed. Synthetic fixtures prove unit behavior
only, not actual sentiment inference or data/model acceptance.

Real retained-input pilot completed with natural exit 0, without mocked sources:
data/research/swing_news_feature_pilot_retry/_manifest.json
SHA256 d1499480bd6ecf731ac913c20c686a4519f6d9b222797a2383144d09d2942c01.
192 stored versions selected (16 per Alpaca/SEC source-year, 2019–2024), 190 enriched;
2 retained SEC records lack event_available_at_utc and remain unavailable. Outputs
preserve the existing historical_proxy clock semantics, not live observation proof.
Existing corpus/input files and executed feature code stayed unchanged; the actual
feature calculation reconstructed identically. 309 QA buckets retain small refs.

Counts are overlapping per-version topic cues, not confirmed events or precision:
earnings89, guidance34, contracts27, corporate transactions53, analyst48, products51,
legal43, financing55, management/operations47, sector/market59 and other31. 104
versions have multiple categories; 30 have mixed rule directions. No FinBERT or
embeddings executed; original model sentiment was not supplied in this pilot.
Stock attribution and classification precision remain unmeasured. Source limits
were reached for 21 cue lists, 31 quote windows and one text prefix, explicitly
recorded rather than disguised as complete full-text recognition.

The first pilot stopped at system memory90.2% before enriching a record (empty
partial file). Preserve its logs/stage. A fresh retry used a 2MiB SQLite page cache
and file-backed temp work; source bytes and memory thresholds did not change.
Retry logs/state/exit: workspace review-results/news-feature-pilot-retry.*.
This closes the code/real execution pilot component only. Full corpus processing,
actual sentiment reuse/inference, verified stock association, learned feature
acceptance, two remaining fits and SPY comparison are not completed by these cues.

### Current continuation: full-news feature engineering on the existing foundation

User correction and priority (October 8): the project already has candles, matched
news and technical indicators. Freeze that foundation. Complete useful news feature
engineering, learn conditional future returns and measure results against SPY.
Do not spend the critical path rebuilding schemas/matching or replaying the obsolete
identical-character gate. Current public APIs remain V1; TradingFlow is untouched.

At each decision cutoff, combine available stock/price history, fundamentals and
deduplicated company, sector and market news. Preserve simultaneous positive earnings,
negative guidance, sector disruption and war/macro context; do not collapse them
into one exclusive category or one overall sentiment. Future price paths are labels
and evaluation outcomes only. Fundamentals and consensus expectations must have
been available at the decision, not restated or captured later without evidence.

Recognize earnings/guidance, contracts/partnerships, corporate transactions, analyst
rating/target changes, products/commercial/clinical milestones, legal/regulatory
actions, financing/capital returns, management/operations/security incidents and
other material stories. Record actor, affected company, counterparties, event status,
scope and available time. Rumours, proposed deals and completed events differ.
Fiscal periods are conditional financial fields, never mandatory for every story.

Full-corpus processing and QA sampling have different purposes. Every eligible
retained article/version remains eligible for enrichment and feature construction;
the QA sample is only a subset used to check the extractor's classifications and
misses. The old policy checked just two news categories: 300 labelled-earnings +
600 not-labelled-earnings; 250 labelled-guidance + 600 not-labelled-guidance = 1,750
review packets. Not-labelled-earnings can still be a useful contract or analyst story.
The fixed numbers were the old quality-measurement policy, not article/model quotas.

Actual retained review input: 516,679 versions / 539,399 record rows. Readable,
company-attributable groups: 466,635 (443,915 Alpaca, 22,720 SEC); groups/versions are
not a unique-article count across all providers. The old sample includes 1,745 groups
and 1,977 versions. It emitted 603 candidate EVENT occurrences, 347 earnings and
256 guidance, in 550 packets / 561 documents (426 Alpaca, 135 SEC). This counter
does not count all news. Finviz/Seeking Alpha coverage is not established by these
specific manifests; credentials or a collector class alone do not establish coverage.

The user requests sample buckets while sentiment/category/vector processing runs.
Implement bounded streaming, reproducible buckets by category/source/year and
sentiment/confidence, including mixed and uncertain/other examples. Store references
and small metadata, not the whole text/vector corpus in RAM. The sampling branch
must not discard unsampled records from features or use future returns to choose
examples. Preserve old 1,750 earnings/guidance judgments; targeted new-category
checks are distinct and cannot be declared already qualified by the old labels.

Source interpretation remains publication-first: December 3 contract news can inform
decisions from that availability timestamp; later delivery dates are context.
Preserve exact authentic quotes but compare company/event/period/claims semantically,
not identical reviewer offsets. Resolve relevant explicit/relative temporal context
with its basis and ambiguities recorded; unknown periods limit only dependent
numerical features. Revised text cannot backdate earlier snapshots. No old false/
null answers are silently relabelled as positives, and original hashes are preserved.

Amend only the untrained third news-reaction profile. Freeze its revised feature
columns, missingness and aggregation before its two fits; its width may change.
Retain four baseline fits, two learners, official ten-session return target, splits,
costs and six-specification/twelve-policy comparison. Three-day recovery can be a
future-path diagnostic; predicting T+1/T+3 explicitly would be a separate target/output
amendment. Chronological training/validation/test and overlap purging remain required;
preprocessing fits on training only. Repeatedly inspected historical test is development
evidence, not an untouched final assessment. No new model search or extra experiment.

Supplied Downloads/event_driven_ml_architecture.md is a proposal, not authoritative
instructions. Its point-in-time/mixed-numerical-feature ideas agree with this plan.
Corrections: do not assume exact 5PM intraday VWAP exists in daily inputs; EPS surprise
needs a zero/negative-consensus rule and consistent periods/units/basis; horizon labels
use trading sessions, not calendar days; the stated embargo inequality would remove
earlier training rows instead of only the post-validation interval. CPCV is not a
replacement for the existing past-to-future evaluation and is not being adopted.
Multi-output learning is not already provided by the existing ten-session models.

Actual delivered foundation: matched inputs/technical relationships across 586,305
decision rows / 59 months; four final baseline return models and their temporal fits;
FinBERT sentiment scorer and existing aggregate feature plumbing; serving/admission
code. Broad structured catalyst feature integration and an accepted SPY-beating V1
predictor are NOT delivered. A full-corpus semantic embedding store/training path was
not found in the inspected source; sentiment scores and embeddings are different.

Feature-side streaming QA bucket helper is complete in software commit 44d8c2d:
src/market_predictor/research/news_sample_buckets.py and its unit tests. Final checks:
28 synthetic UNIT cases plus 2 continuity checks passed (30 total); targeted Ruff
and strict source mypy passed. One consolidated review found the unsupported
multi_event label; it is now multi_category, with direct one/two-label regressions.
The helper stores bounded immutable references, records input occurrence counts,
keeps other/unavailable/mixed strata, and samples reproducibly regardless of batch
order. It does not enrich text, create embeddings, discard unsampled feature rows,
qualify sources or fit models. No actual corpus job or integration proof is claimed.
No collection/base/schema changes. Full-news classifier/sentiment/embedding-stream
integration and the revised real feature pilot remain unfinished. Old zero-support
publication 5b8b401105f28785761fc929da9b6aedbdaefb9b37703b2cd9aa9a5af4e98472 remains
unverified historical output after the RAM failure; no fabricated replay/model win.

Historical saved review checkpoint (99% account usage):
- Reviewer A: 173 fully inspected packets, 1 partial packet; 200 saved version assessments (3 partial, 0 metadata-only).
- Reviewer B: 648 fully inspected packets, 2 partial packets; 651 saved version assessments (3 partial, 8 metadata-only).
Actual schema/exact-quote checks passed; complete two-review ingestion/qualification
remain OPEN. Partial packets retain truthful incomplete inspection; no uninspected
negatives or accepted features/models/SPY result are invented. Software5cf8208 and
its focused unit/lint/types/real candidate parser evidence are pushed. Resume own
source-review progress, then full actual ingestion/replay and unchanged metrics.

Latest software checkpoint 5cf8208 implements pinned complete two-review
ingestion/correspondence and independent byte replay. Unit checks 336 passed;
supported pre-lease bulk-read finding fixed, final21 passed; Ruff/types passed.
Actual retained strict candidate parser checked 14,182 versions/16,004 candidates;
decoder/dependencies unchanged after lock fix. Original133/packet24 owners unchanged.
Two fresh independent model-assisted source reviewers are now inspecting real
blinded packets. Genuine partial reviews are not complete-frame qualification:
no new quality/feature/model/SPY acceptance yet. Full1750packet/1977version coverage
per reviewer and actual ingestion publication/replay remain required.

Latest component 75791f4: complete-version annotation validation and exact event
correspondence are installed and pushed. Selected unit checks: 314 passed; Ruff
and strict typing passed. Actual installed reader checked all 1,750 published
packets/1,977 versions, including 15 metadata-only, with source hashes unchanged.
No new review labels, source quality, features, fits or SPY performance follow from
this reader check. Next implement pinned two-review ingestion; fresh independent
source judgments and measured qualification remain required before issuer training.

Last updated: 2026-10-06

Actual complete-document review packet publication and independent byte replay at
d228191 passed (exit 0, October 6 13:44:20 UTC). Manifest 2b4b933565704a06fae86e815f3edab56412c2d2717254f4eb39e5ad5f50b3a9:
516,679 original versions, 539,399 records, 22,722 aliases; 1,750 fresh review samples
and 1,977 packet version/occurrence entries (15 metadata-only). All 24 executed code
pins unchanged. U=466,635; excluded development D=1,748; fresh F=464,887. Original
judgments/data remain unchanged; no new labels, measured source quality, 126 features,
remaining model fits, serving or SPY improvement follows from this packet result.

## Qualified Reaction Profile: Current Acceptance Matrix

Current input binding implementation 852f43f is pushed: shared original-data reader
for reaction publication and verification, with required actual original receipt and
final source checks before completed outputs. All 132 prior replay code pins stay
unchanged; actual new reader exercise passed (347199797d1521967194c85885632040a7065e713b7e9b23efdec6612123af21): 586,305 decisions, 59 months, 551 stock groups and 1,510 SPY rows; 133 implementation pins unchanged, worker/lease closed. Unit scope: 70 passes plus
two setup failures, both corrected cases pass separately (2.32s); receipt exit check
passes (4.47s). Ruff 11 files and strict types five sources pass. No actual qualified
news authority, 126 publication, two remaining fits or SPY improvement follows.

Current proposed order is the unchanged 124-column relationship parent plus the
two existing raw issuer reaction columns (126 total), not aggregate news counts.
No extra model specification or API version is introduced.

| Layer | Current evidence/status |
| --- | --- |
| Source collection | Retained initial-fit Alpaca inventory and corrected SEC archive. Completed export binds 516,679 versions and 1,750 blind samples in manifest 3c59845761740dc1a07a2f994cceaba53354c6f9486f4baa634ebd422a6a84ce. Later SEC bodies stay sealed. Exporting text does not admit an event. |
| Source clocks and aliases | Filing availability and issuer identity clocks remain distinct. Original historical capture gaps remain declared proxies; no past observation timestamps are invented. |
| Candidate precision and recall | Both independent source-only model-assisted reviewers completed all 1,750 samples. Their immutable project copies are hash-bound. Actual qualification session 64007 stopped at system memory use 90.4%, leaving no completed manifest. Its intermediate review metrics reject both families: earnings joint positives 225/300; guidance 216/250. Reviewer False issuer counts include disagreements, not proof every article names another company. Disk-backed history repair 1d0d3be is pushed and unit-verified; exact causal behavior, sources, reviews and gates remain frozen. |
| Historical/live calculation | Shared 124-to-126 projection and complete 120-to-126 candidate adapter are implemented and unit-tested. The adapter requires explicit baseline observation semantics and compatible component authorities. Actual qualified publication and observed live source binding remain pending. |
| Ordered feature contract | Reaction profile fixes 126 columns and three-day selection. Monthly publication and independent replay are implemented and unit-tested. Preserved 124 source/value/clock reuse comparison is implemented, including WTW/ATVI/INFO/SBNY exclusions. Actual reuse report SHA5f480f56...83265 is failed_differences: all 586,305 rows/inherited columns unchanged, but 359 stock-input groups, all 549 SPY-input groups and 547 relationship-feature groups differ. INFO/SBNY fail complete physical-prefix verification. Current-source 124 publication is not authorized. The distinct original-snapshot route passed full replay and independent reproduction (aa8000577dd9628d5edfc5557758238394ebf36aaa8a26f8fae0177fc7d9cec9), zero differences over all 586,305 rows/59 months. Original research-input binding 852f43f is reviewed/unit-verified; its actual retained-data reader exercise passed for every original month/group under receipt 34719979...3af21. Original artifacts and targets remain unchanged. |
| Training/serving consumers | Explicit 126 research dispatch and matching receipt checks are implemented and unit-tested. Original folds, weights, estimators and targets remain frozen; two issuer-profile fits remain pending. V1 return-regressor serving design is frozen but not implemented. Current serving still assumes classifier probabilities. |
| API representation | Public API remains V1. Pure candidate construction grants no training, promotion or serving admission. Live reaction values require observed sources, profile binding and an accepted return model. |
| Poison/tamper/parity | Unit fixtures exercise future revisions, identity clocks, unknown coverage, parent preservation, interrupted checkpoints, source/value replay and resource pressure. The positive synthetic source-chain unit uses actual calculation/replay code and independent reaction-value checks. It is not market-data integration evidence. |
| Frozen funded exits | Raw-source funded provider, saved temporal OOF evaluator and benchmark cost repair are implemented and unit-tested. Stock costs remain 20 bps per round trip; eligible fixed-horizon SPY/QQQ/sector benchmarks have zero costs. Actual funded comparisons and accepted SPY improvement remain pending. Missing payment/action facts must remain explicitly unavailable. |

Implementation 771d053 is pushed. Selected unit/fixture checks passed: core 215,
direct-consumer 258 and final authority/positive-chain 43. Ruff and strict typing
passed affected source modules. Synthetic data and mocks are restricted to unit
tests by the October 5 user instruction recorded in AGENTS.md. Actual qualification,
reuse, feature publication, training and evaluation must use retained/provider data
and actual readers/calculations without patched admission or model output.

Corrective CI implementation 892100c removes the operational synthetic release
builder and confines promotion fixtures to tests/support. Five focused unit checks
passed; Ruff, source-configured strict typing and workflow parsing passed. CI checks
actual unprovisioned startup and 503 refusal only. Docker is unavailable locally;
container execution and successful real-data serving are not claimed.

Latest actual-data attempts are retained failures, not replaced with fixtures:
qualification 4422 exited at 90.0% system memory (1.57 GiB available), with only
request/authority files and the same rejected family metrics. Reuse 37426 exited
at 86.8% system memory (2.07 GiB), above its canonical 85% requirement; no report.
Both processes and leases are gone. No new qualified/current feature authority exists.

Implementation b589725 proves the saved-model strategy/date-config naming changes
explicitly. Actual retained-byte and fold-calendar verification passed for both runs
in data/reports/swing_saved_evaluation_configuration_verification.json. It does not
replay prediction payloads/parent holdouts or prove funded returns. The focused unit
run passed 294 checks but failed two existing package-boundary checks in issuer
publication. Their narrow ownership move is now verified and pushed as c2063a8:
314 affected unit/package checks passed; exact qualification text and all 23
protected producer dependencies are unchanged. No real feature pass follows from
those fixtures. System headroom improved to 4.101 GiB after the unit exited; actual
reuse will be retried after documentation closure. No general gate
exception, source/target rebuild, extra learner or API generation was introduced.

Read-only source review identifies development improvements to assess next: exclude
debt-tender results from earnings candidates; recognize a year directly attached to
earnings guidance; mark equal/contradictory guidance ranges unresolved; bind review
support to the actual exhibit rather than a cover page. Original judgments/metrics
remain unchanged; previously inspected examples cannot be independent new acceptance
evidence. The supported extraction repair is now implemented/pushed in 77ff5f5,
with a shared raw/saved candidate core and exact hash/context/clock-bound saved-text
adapter. Candidate-only sidecar publication/replay and a read-only parent join are
implemented; no original text/SQLite copy, provider rescan or review-label rewrite.
121 initial unit cases passed in 9.64s. Final consumer scope passed 242 cases but
failed its stale CLI inventory; 23 inventory/boundary cases then passed, and the
remaining alphabetical-order error was fixed and its one case passed separately.
Ruff six-file and strict three-source checks passed; one bounded review found no
further issues. Initial strict narrowing errors were repaired with checks retained.
Actual derivative session 64326 completed exit 0 at 00:46:30 UTC October 6: manifest SHA9c5f5580...f82d0, 516,679 versions fully replayed before publication, 14,182 candidate versions, 470,541 unclassified and 31,956 unavailable. All training/qualification/serving flags remain false. Independent 516,679-version verification passed; original candle replay and independent receipt reproduction also passed as recorded in the active handoff; no new qualification or feature admission is claimed.
The actual comparison's separate 131 dependencies remain byte-identical.

These software results do not prove SPY outperformance or satisfy the future 252
decision sessions plus ten maturity sessions required for prospective assessment.
TradingFlow and main remain untouched.

Actual reuse retry 69678 ended exit 2 at 23:08:31 UTC October 5 with no report:
the historical lookup expected an unversioned observation schema, but all four
original WTW/ATVI/INFO/SBNY facts pin the unchanged .v1 report SHA70c6f850...78060.
Repair 46a7669 is pushed. Its sole exact-hash/schema/date-bounds historical reader
is shared by selector/replay; canonical contracts and original evidence remain
unchanged. Selected 294 unit/package/naming/command checks passed in 25.46s,
Ruff/strict two-source typing passed, and one bounded review found no issues.
An actual leased read verified the four metadata references and original report
bytes. It did not replay source values/features. The complete actual retained-data
reuse restarted at 23:17:18 UTC as session 95557/PID 37332, with unchanged config
SHA95dd86ec...209c. It ended exit 2 at 23:56:34 UTC with the real price/feature differences recorded above; no passed reuse report.

Original-input replay implementation c81c70a is pushed after the measured candle
changes. It verifies the original saved OHLCV/ownership/quarantines under current
four-feature calculations and publishes only an independently replayable receipt;
existing data/fits and current-source gates stay unchanged. Its 419 affected unit
cases passed; final 29 units after lease-before-input ordering passed. Ruff and
strict two-source typing passed; one bounded static review found no further issue.
Actual replay and reaction consumer integration remain pending; no fixture pass is
market evidence. The running chain's combined 134 executed source hashes are frozen and unchanged.

## User-Confirmed Astra Plan (October 4)

The user confirmed that the saved plan's Bounded Experiment and Ordered Checkpoints
are the intended Astra plan. Resume that approved sequence; do not replace it with
the recent generic aggregate-news comparison or a regularization search.

The frozen experiment crosses regularized linear and shallow boosted return models
with three profiles: existing technical inputs; distinct price/volume/regime
relationships; and those relationships plus qualified issuer-news/SEC reaction
features. Four specifications are already fitted (`07963cc`, `a8be7cb`); the two
issuer-reaction specifications remain. The measured weaker relationship results
remain evidence, not an accepted improvement. At most six learned specifications
and two frozen exit policies are permitted; any additional trial requires an explicit
scope amendment before inspecting its results. Public APIs remain V1.

Current work is original ordered checkpoint 3: complete qualified issuer reaction
features using existing archives and the implemented `issuer_reaction` measurement
component. Resume existing source/content qualification and define the exact final
feature columns in the acceptance matrix; do not re-run completed aggregate joins
or their source replay merely because names changed. Generic news counts/sentiment
in `catalyst_full` do not satisfy the issuer-reaction feature contract. The component
already measures stock-minus-SPY post-announcement returns and volume relative to
the preceding 20 sessions; its source/event qualification and full batch/inference/
training integration remain unfinished. An observed row/clock error gets a narrow
repair with a regression test; otherwise reuse completed data and evidence.

After the feature profile passes its specified checks, fit its remaining linear
and boosted models on unchanged decisions, targets, folds, weights and costs. Then
complete original checkpoint 5: evaluate the frozen long-only policies using funded
portfolio NAV, costs, cash, turnover and drawdown against SPY. Prospective assessment
and accepted V1 serving follow the original plan; the exposed old test period cannot
be called untouched. TradingFlow remains independently owned and untouched here.

This confirmation supersedes the recent next-step instructions to replay July/all
news aggregates and add a fixed linear generic-news comparison. The completed paired
prediction report, full peer replay and configuration-difference evidence below are
retained; original artifact hashes and numerical values are unchanged.

## Matched-Input Reuse Evidence (October 4)

The existing matched publication was preserved. Current numerical peer transforms
reproduced every saved 120-column technical and 150-column news profile exactly,
including all derived availability clocks and identical common technical values/
eligibility, over all 586,305 decisions / 59 months. This replays saved base values;
it does not recompute candle indicators, raw-news aggregates or future targets.

Peer report: `data/reports/swing_matched_peer_reuse_verification.json`, SHA256
`f51418cae90bb8fce520d4b8b2aff97eb344c0702ad9e9deadaba943b2c323f2`.
It binds 20 calculation/verifier source files captured before replay and rechecked
before reporting, input Parquet/sidecar bytes, strategy bytes and runtime versions.
Its training/promotion flags remain false. The review's missing implementation-pin
finding was fixed before the complete run; a deliberate code-mutation regression
refuses a passed report. The first unbound audit stopped after 17 matching months
without publishing a report. Full guarded rerun completed and released its lease.
Normal canonical readers still reject old sidecar names; this is inspection of
immutable historical evidence, not an old-format production acceptance path.

Configuration report: `data/reports/swing_matched_configuration_differences.json`,
SHA256 `97fe1198f6dc7ed0ebd7e9401fa7ee9215c5d84268fa8990a801761954620b10`.
Twelve of 19 original configurations are byte-identical. The seven changed originals
were recovered at their exact saved hashes from Git `7b5d834`, including original
CRLF bytes. Three change only schema names; monthly-news changes its schema and
corresponding lineage-policy pin. Outcomes and symbol corrections change source
bindings without changing economic settings or correction rules. The technical
feature policy substitutes standalone adjusted authorities for the old combined/
parent panel. These differences do not demonstrate changed numerical values.
Original pins were not rewritten; source-equivalence and fitting remain unasserted.
Exact nested-key/type/date-change regressions passed for this configuration audit.

Locally retained immutable helper bytes (not production entry points):
`C:/Users/manis/Documents/Codex/2026-09-28/c/work/verify_matched_peer_reuse.py`, SHA
`694eb5efbd6799abd9358eab7bc010215382e47c33bbcb7774d53009743061fc`;
`C:/Users/manis/Documents/Codex/2026-09-28/c/work/compare_saved_configuration_semantics.py`,
SHA `1ff45cba71a735d566830b98b2db4ce96f183279dc092807fdb1f2e793df98fd`.
Preserve these as-replayed bytes; a cosmetic edit would change retained evidence.
No project Python/configuration, stored inputs, data format, API version or TF file
changed during this evidence checkpoint. The earlier 301-test implementation checks
remain the code receipt; no new full suite or model fit was performed.

The prior proposal to replay the ten aggregate-news inputs is superseded by the
user-confirmed Astra sequence above. No aggregate row error was demonstrated by
these checks. Retain the completed evidence; resume issuer-reaction feature work.

## Completed Paired Outcome Check (October 4)

Implementation `da67c5c` is pushed. `compare-saved-swing-returns` reuses recorded
predictions without fitting/loading models or changing matched data. All 16 pairs
have identical training/scoring identities, weights, cutoffs/maturity, targets and
settings. Every scored artifact hash and all scoring metadata match; declared
holdout identities and eligibility counts are also checked. Baseline/candidate
roles are explicit. Missing targets remain counted rather than silently dropped.

Report: `data/reports/swing_technical_relationship_paired_comparison.json`, SHA256
`dabf625a79890849c3939d8001f37f4fdaf27c5a166be97bcad5615e5c866393`.
Pooled metrics give each evaluable session equal weight; they are not unweighted
means of stored fold summaries. Each scope has 592 evaluable sessions. Temporal:
289,802 scored / 187,432 known outcomes; held-out securities: 54,964 scored /
35,548 known outcomes. These scopes overlap and are not independent trials.

| Learner / scope | Technical error above zero-excess baseline | Relationship error above zero-excess baseline | Relationship error increase vs technical | Technical / relationship daily rank |
| --- | ---: | ---: | ---: | ---: |
| Linear / later dates | 1.7667% | 2.4877% | 0.7084% | 0.004494 / 0.001255 |
| Linear / held-out stocks | 1.2760% | 1.9799% | 0.6950% | 0.006959 / 0.006091 |
| Boosted / later dates | 1.9946% | 2.2954% | 0.2949% | 0.014672 / 0.007065 |
| Boosted / held-out stocks | 2.1444% | 2.4805% | 0.3291% | 0.000293 / -0.012918 |

Error means mean squared numerical-return prediction error. Zero excess means
predicting no return above SPY; this is not a funded SPY portfolio comparison.
The four added relationship inputs worsen errors and rankings in every pair. No
new accepted model, portfolio performance or untouched-test evidence is asserted.
Historical folds are development evidence for improving features/settings.

Verification tier: component. Final affected tests: 301 passed in 40.94 seconds
(comparison, return validation/artifacts, CLI inventory, continuity and dependency/
architecture boundaries). Changed-file Ruff and strict mypy over both affected
sources pass. Two consolidated P2 findings closed with explicit arm mappings and
both-arm contradictory-holdout tests. Atomic publication exposes no partial
report and refuses replacement; final consumed-file hashes are rechecked. Full
suite, training, provider downloads and C# checks were deliberately not run because
this checkpoint changes only historical comparison/reporting. Real report passed
under the shared lease and unchanged memory limits; no worker remains running.

## Existing Matched Inputs Retained

Keep `data/features/swing_corrected_initial_fit_research`, all 586,305 decisions/
59 months, original target values, availability clocks and both original profiles.
Its 120 technical inputs and 150-input aggregate-news profile remain distinct from
the planned qualified issuer-reaction profile. The full peer replay and exact
configuration comparison are complete. No further routine reconstruction/replay,
generic news-only trial, data-format change or additional learner search is scheduled.
A reproducible affected field or actual consuming contract conflict may justify a
narrow repair with explicit evidence; preserve original provenance throughout.

## Canonical Source Reconstruction

October 3 implementation `86f7d42` reconstructs incremental raw archives using current
page checks: 6,972 units, 7,228 pages, original bytes/clocks preserved, no downloads.
Daily/revision replay has no failed or pending units through October 2. Four empty
capture intents remain unavailable evidence; they do not establish missing daily
history or missing clocks in downloaded pages. The fresh initial-fit raw-price plan
and independently pinned replay also pass, reading only permitted identity columns.
Implementation `2216a2d` also reconstructs all 564 initial-fit raw-price units /
601,834 rows through the canonical collector and current reader. Original provider
bytes/retrieval clocks are preserved; derived ingestion timestamps record actual
materialization. All 44 daily-history tests and lint/types passed; independent review
passed. No new downloads or feature/label publication/training occurred.
Implementation `fdf2b40` closes both corrected source reconstructions: 1,476 raw
and 3,020 adjusted rows, unchanged reviewed mappings/documents, current-reader round
trips. The new raw selection has 602,709 rows; 259 missing sessions and 21 unusable
observations remain explicit, with no imputation/exclusions or label admission.
All 203 affected tests and consolidated review passed. The next independent adjusted
source publication requires complete transport receipts; sampled old warmup/later
source metadata cannot establish those receipts. Alpaca SIP credentials are configured.
Standalone adjusted acquisition/normalization completed in `ce1d07c`: 564 observed
units / 794,279 raw candles, exact SIP receipts and current offline replay pass.
All 13 benchmarks meet required session coverage; stock queries retain 16,170 absent
dates and 582 unusable candles for feature abstentions. These are query-window counts,
not counts of missing eligible decision inputs. No feature/target reconstruction or
training is claimed. The two publication findings are fixed, with private resumption
and full normalization/hash checks before final exposure. 35 plan, 33 adapter and
277 affected component checks passed at the stages recorded in the active handoff;
final Ruff/types passed. Exact paths/hashes and remaining dependencies are in the
active handoff. These facts do not admit current-schema model features/targets or
recertify historical fitted models. The results below remain historical evidence.

Corporate-action source reconstruction completed in `b97ad60`, followed by the
hash-preserving whitespace fix `635ebdb`: 570 queries/pages and 14,352 records in
`data/raw/swing_corporate_action_sources_canonical`. Independent full source/output
hash and body/metadata/clock comparisons passed; normal final-path offline replay
passed. Semantic audit `a713f53f2169be60ddd5cbf81786be8f772127496d65a93862eb72992f643add`.
Original evidence is unchanged. The new receipts prove reconstruction provenance;
announcement availability, absence, ownership and accounting admission remain false.
Selected overlapping source/evidence/CLI/architecture tests, Ruff, strict mypy and
consolidated review passed as recorded in the handoff. No labels or model fit were
produced. Implementation `12baec5` separates target action bindings from the unchanged feature
decision policy and requires exact typed equivalence/target lineage/monthly metadata.
28 helper, 13 CLI, 410 pre-fix integration and final 249 join/direct-consumer checks
passed (overlapping); one string-dtype review finding is fixed with an actual canonical
ID/Parquet roundtrip regression. Ruff/types and review passed. Actual target publication
is now frozen privately, with pilot/full replay and independent audit before exposure.

## Current Canonical Technical Feature Publication

Implementation `f8d266f` completed the source-only technical reconstruction. Exact
query-window bindings and independent memberships replace old combined/special
issuer readers. The price context reads no corporate-action target payloads; its
outcome wrapper still enforces action admission. Final real output preserves all
586,305 decisions in 551 groups/59 months: 564,557 usable technical rows and 21,748
explicit warmup/unavailable rows, no added exclusions. The contract has 40 technical
feature names. All source/artifact hashes, group/month ID equality, parent IDs,
2019-07-09..2024-05-28 bounds and feature clocks were independently checked.
Manifest SHA: `c1601b6683893dcf9d3885b3c7446430dafecd96ef3e06d835c910338d016891`.

| Current canonical profile | Source/batch/order | Fixed-horizon labels | Training/evaluation | Live/API/promotion |
| --- | --- | --- | --- | --- |
| technical predictors | Verified real feature-only publication and 40-name contract; targeted numerical/causality/tamper fixtures pass; full real numerical replay not run | Corporate-action source reconstructed; corrected-label reconstruction pending | Not run on this publication; no SPY outperformance established | Not admitted |

Consolidated review's query-window quarantine and Windows-path findings are fixed.
Selected final test sets (overlapping) passed 59 relationship, 339 predictor/feature/
architecture/CLI, 68 derivation and 292 runtime/source/architecture checks. Ruff and
strict mypy passed affected sources. Resource retries preserved all saved files;
unused-buffer cleanup plus `ARROW_DEFAULT_MEMORY_POOL=system` completed assembly
without changing thresholds or feature formulas. Full suite, full real numerical
replay, training and promotion were not run. Exact pins, resource observations and
the next frozen private target publication slice are in the current handoff.

## Current Long-Only Swing Campaign

Last recorded retention verification: the completed technical return run retains both
final models and 16 evaluation fits with matching original artifact hashes.
`data/reports/swing_retained_run_integrity.json` records 256 matching source pins
and ten implementation mismatches at that audit. Subsequent implementation changes
have not been recounted by that retained report. This preserves historical completion;
it does not reverify causality, numerical replay or current-contract eligibility.
The acceptance evidence below describes the original run. Fresh swing-only
contract verification is required before reuse. No new features or model fits
were produced by the retirement cleanup.

### Fixed-Horizon Training Requirements

This is the current campaign's acceptance matrix; historical A2/A3 rows below
do not certify the new return-regression consumer. Technical fixed-horizon
fitting does not require a managed-exit timestamp or proof of historical news
receipt. It does require independently usable, matured fixed-horizon labels,
causal inputs and an explicit research-only training contract.

| Planned profile | Source, batch and feature order | Fixed-horizon labels | Training consumer and experiment | Live, promotion and API |
| --- | --- | --- | --- | --- |
| existing_technical | Verified corrected publication: technical_market, 120 ordered inputs and availability clocks | Original outcomes replayed; exact costs/comparisons/global maturity audited; inner fitting purges actual maturity independently | Verified implementation 07963cc: 16 real temporal/transfer fits and two final models completed; 4,329 full tests passed, 10 skipped | Not applicable to this offline research experiment; artifact loader explicitly rejects serving/promotion purpose |
| technical_relationships | Fresh UTC-clock publication and independent source replay pass all 586,305 decisions / 59 months; fix 82842e0 passes 94 focused tests and review | Exact original columns/outcomes preserved; additions independently recomputed, baseline numerical evidence explicitly inherited | Frozen 124-column experiment a8be7cb completed 16 validation fits and two final models; independent artifact audit passed; economic edge not established | Not authorized |
| technical_relationships_issuer_reaction | Blocked: catalyst_full has aggregate news features, not qualified issuer/SEC event-reaction inputs; SEC/Finviz coverage is unknown here | Must reuse matched independently admitted labels | Not trained; counts and sentiment cannot substitute for the promised reaction feature contract | Not authorized |

October 5 content-review input component `90a7f16` is pushed: exact source/version,
chosen-field/extracted-text hashes, HTML element span locators, explicit issuer/action
and fiscal-period evidence, independent unavailable/ambiguous reasons and causal
clock replay share one historical/live implementation. Candidates remain unadmitted.
All 548 affected checks, Ruff and strict mypy pass. Thirty-two pinned saved Alpaca
records passed adapter/spans smoke (20 headlines, 12 bodies); this is not event
precision/recall evidence. Corrected initial-fit SEC bodies `b3041a07ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦69318` exist;
older metadata-only wording below is historical. Full-content review authority,
source-linked candidate/rejected annotations, precision/recall, exact model columns,
reaction profile publication and its two fits remain outstanding. Existing title
reviews admit analyst revisions only and cannot admit this new extractor.

September 21 source-qualification component: shared issuer reaction measurements
are implemented in `8e24d45`, separately from final model-profile admission. The new source
contract binds event/version, issuer identity and existing price authorities, with
separate availability semantics for event, identity and bars. It adds no training
columns or fits. Bounded source inspection confirms mixed headline/body Alpaca
text and SEC form-level metadata; sparse corporate-action HTML documents are not a
cohort-wide filing-content archive. Full-content/version qualification, per-ticker/
year coverage, exact final columns and immutable publication/replay remain required.
The component passed 276 focused reaction, relationship-regression and continuity
tests, changed-file Ruff and strict mypy. Independent plan/design/code findings are
closed. This software evidence does not admit event content or any model input.

September 20 component evidence: `return_feature_profiles.py` freezes formulas,
ordering, source semantics and missingness; `return_relationships.py` preserves
the real baseline's 120 values and 160 availability mappings. Fourteen independent
regressions cover the reviewed baseline/source fixes; the worker's 131-test slice,
Ruff and strict mypy pass. Physical Alpaca/SIP daily metadata, exchange-session
alignment, ingestion ordering, future-clock rejection, truncated-prefix parity,
missing sessions, multiple security identities and float32 overflow are tested.
These are local transformation checks, not a full-history verification or fit.

September 21 implementation adds bounded source readers, eight-bucket staging,
immutable monthly publication, pinned restart verification and an independent
source-to-value replay for the additions. The real synthetic pipeline reaches a
354-row, 124-column loader and rejects rehashed altered features. Lease contention,
initializer pinning, corrected FI/SATS histories and WTW/ATVI/INFO/SBNY quarantine
paths are covered. Final changed-file Ruff and strict mypy over 15 modules pass.
Historical baseline numerical correctness is explicitly inherited from accepted
parent evidence; this is not a new baseline numerical replay or benign-drift claim.
No completed-model reuse, real-history admission, profitability or promotion follows
from these software checks. The unrelated SEC collector released the normal lease;
the first real materialization stopped before publication because a new reader
incorrectly required directly listed collection pins instead of their verified
transitive bindings. Reader/fixture correction `ed3c125` passes 69 focused tests,
Ruff, strict mypy and independent review. Real publication completed; independent
verification then rejected an all-null clock dtype mismatch. The builder's NumPy
assignment discarded the UTC extension dtype. A narrow `.array` and explicit UTC
nanosecond construction fix (`82842e0`) passes 94 tests plus lint/types and independent
review; comparisons stay exact. Fresh publication and all-row source replay passed;
real readiness passed. Original data and the prior non-admitted publication remain
unchanged. The independent ML reviewer approved only the two frozen relationship
specifications; actual fitting completed and does not imply profitability/promotion.

### Relationship Return Results

Completed run: `data/research/swing_relationship_return_models_initial_fit`, manifest
`7a699020920024ed29b5d8547724f5355fdc2202ca176f11d38f23418bd4666e`.
Both final models use 314,167 rows; peak working memory was 2.471607 GiB. An
independent reviewer verified all 18 unit/model hashes and 16 prediction hashes,
the frozen settings and 124-column order. No outer validation/test or portfolio ran.

The following are unweighted means of four stored fold diagnostics, not pooled
portfolio returns or significance tests:

| Learner / scope | Mean daily rank correlation | Mean squared error | Zero-excess error |
| --- | ---: | ---: | ---: |
| Linear / temporal | 0.000704 | 0.00262038 | 0.00255703 |
| Linear / unseen security | 0.005503 | 0.00292185 | 0.00286511 |
| Boosted / temporal | 0.006304 | 0.00261576 | 0.00255703 |
| Boosted / unseen security | -0.013526 | 0.00293655 | 0.00286511 |

Fitting succeeded, but ranking remains weak and all four mean errors exceed
predicting zero excess. This does not establish tradable edge. Baseline request
metadata confirms matching input identities, folds and holdout assignment; baseline
result files were OS access-denied at that earlier audit. This limitation is now
resolved by the October 4 exact paired comparison above. Keep the old unweighted
fold summaries distinct from the new pooled equal-date report; neither measures
SPY portfolio outperformance. Do not treat the weaker relationship profile as an
accepted improvement.

Readiness command: `audit-swing-training-readiness`, configuration
`configs/swing_training_readiness.json`. Its scope is bounded initial-fit
diagnostics, not permission to fit or trade. It preserves source publications,
reports every month/sector/year and all missing model values, and keeps these
facts separate:
- feature eligibility, derived without future returns;
- complete model inputs, a diagnostic rather than an imputation decision;
- available fixed-horizon supervision, with all required comparison evidence;
- their intersection, not a historical opportunity-selection mask.

The return consumer binds exact learner settings, missing-feature and missing-label
policy, purged session folds, weights and separate security holdout before fitting.
Ready technical specifications need not wait for the blocked
reaction profile or later outer-validation publication to run bounded inner
development. Missing selected outcomes cannot be removed from economic evaluation.

### Technical Return Experiment Acceptance

The research-only scope was frozen before fitting and verified on the real run:

| Layer | State and evidence |
| --- | --- |
| Source and historical availability | Verified research inputs in the pinned percentage-only readiness report; no prospective first-receipt claim |
| Historical feature construction and order | Verified 120 technical inputs; return input loader checks exact file/sidecar pins, column clocks, population and date limits |
| Labels and costs | Verified by shared fixed-horizon checks; original net excess target is not recosted or replaced by managed outcomes |
| Fitted preprocessing | Verified focused tests: training-only medians, 240 stable encoded columns, weighted scaling, separate fold/scope fits |
| Training and validation | Verified fixed Ridge/XGBoost, four calendar folds, actual-maturity purge, date weights and separate identity-based holdout; all 18 real fits completed; 314,167 rows in each final fit |
| Persistence and inference parity | Verified synthetic and real-estimator serialization/reload, pinned exact-byte loading, request-bound resumable units and prediction parity; all 18 unit manifests and 266 request source pins independently rechecked |
| Live feature construction, API and promotion | Not applicable to initial-fit offline research; existing live model admission remains separate; research artifacts reject serving purpose |

Configuration: `configs/swing_return_training.json`, SHA256
`47155e2d6ef43e9efb6bb54621913e3df5ee0f66797153c30f71d8f69e5593cc`.
No source family, exclusion list or sector threshold changed. Unknown outcome rows
remain in scored-population coverage, not silently removed from a portfolio result.
Regression and per-date rank diagnostics are not evidence of funded SPY outperformance.

Completed real fit: `data/research/swing_technical_return_models_initial_fit/_manifest.json`,
SHA256 `0a30ef2f8e3b95cee252f8c1d1bb5be3c6b98cddda4e47d68e835ad5c1892551`.
Initial-fit inputs remain 586,305 decisions / 545 securities / 1,231 sessions.
There are 101 held-out security identities. Each final model trains on 314,167
eligible rows with matured supervision over 1,022 observed training sessions;
these are not 314,167 independent market episodes. Full calendar folds remain
503/682/861/1,040 training sessions before eligibility, with 179/179/179/181 scored
sessions and ten intervening sessions in each fold. No calendar is shortened
using outcome availability. No outer-validation or historical-test read occurred.

Sequential training peaked at 2.406 GiB. Independent review is closed; 105 focused
tests, 240 CLI/boundary tests, the full 4,329-test suite (10 skips), Ruff and strict
mypy passed. Code and scripts together pass strict typing across 409 files.

Recomputed from concatenated nonduplicate out-of-fold predictions, with equal
total weight for each date having an observed scored outcome:

| Learner / scope | Date-weighted MSE | Zero-excess MSE | Mean daily rank correlation |
| --- | ---: | ---: | ---: |
| Linear / temporal | 0.00260415 | 0.00255894 | 0.00449 |
| Boosted / temporal | 0.00260998 | 0.00255894 | 0.01467 |
| Linear / security transfer | 0.00290470 | 0.00286810 | 0.00696 |
| Boosted / security transfer | 0.00292961 | 0.00286810 | 0.00029 |

Temporal scores: 289,802; known outcomes: 187,432; unknown: 102,370.
Transfer scores: 54,964; known outcomes: 35,548; unknown: 19,416.
Both scopes have 592 dates supporting rank correlation. Coverage is about 64.68%,
so these conditional diagnostics cannot stand in for full-population economics.
Both learners have higher aggregate squared error than predicting zero excess,
and the ranking signal is weak. This is a measured research result, not a runtime
failure or a promotion result. Do not invert scores, tune thresholds or exclude
losers after viewing it. Two relationship specifications subsequently completed;
the two issuer-reaction specifications still require their distinct qualified
inputs. Neither completed profile silently includes news/SEC reaction features.

Real audit completed successfully on September 14:
`data/reports/swing_initial_fit_training_readiness_percentage_only.json`, SHA256
`7a468ed3e0662d8d6390ae575bee700cdd7da4ef8bb7c8ae921aed6b8eead39f`.
The percentage-only policy audit passed in session 40719. All data diagnostics
equal the prior verified report; only expected config/code pins and the explicit
memory-policy record changed. Earlier receipts remain immutable historical
evidence. Successful diagnostics do not authorize fitting.
It covers 59 months, 1,231 sessions and 545 distinct initial-fit securities. The
586-member retained campaign population is not the distinct-security count of
this earlier fitting interval. Every one of the 586,305 published decisions per
profile remains present; no new exclusions.

| Diagnostic | Technical | Aggregate catalyst |
| --- | --- | --- |
| Feature-eligible rows | 479,709 | 479,709 |
| Complete model-input rows | 298,446 | 296,137 |
| Usable fixed-horizon labels with required comparisons | 378,037 | 378,037 |
| Complete-case input/label intersection | 194,679 | 193,125 |

The matched complete-case intersection is 193,125. Many missing values are
sector-normalized inputs: for example return_5d_sector_z is missing on 258,190
published rows, and rsi_bullish_divergence_strength_sector_z on 278,833. The existing
scaler intentionally emits missing sector values for undersized peer groups or
unavailable inputs; the audit does not lower the frozen 30-peer threshold or
silently drop those decisions. The return consumer now fits its missingness
encoding separately within each training partition. The readiness audit itself
did not evaluate outcomes or fit models; real model diagnostics are reported above.

Independent reviewer Beauvoir closed two supported P2 findings after failing
regression tests: changed transformation policies must match publication provenance,
and benchmark-only outcomes must also mature before the numerical fitting boundary.
Consolidated verification passed 289 tests; Ruff and strict mypy passed on 397
source files. The complete suite passed 4,202 tests, with 10 skips and 132 warnings
in 1,775.89 seconds. Durable log: `data/runtime/swing_training_readiness_full_tests.log`;
JUnit: `.test-tmp/swing-training-readiness-full.xml`. Eight skips require Windows
symlink privileges and two are opt-in memory stress tests, not passing evidence.
Reviewer closed; no data, training or test workers remain.
Implementation `b3b2372` is pushed with the final receipt refresh passed.
The 58 post-format readiness/continuity tests passed in
11.01 seconds; Ruff and strict mypy passed again. This does not replace the
objective-specific training contract or authorize fitting.

The subsequent memory-policy implementation `7bb805b` is pushed and independently
reviewed. It applies the approved 90% ceiling without a separate absolute free-RAM
floor, while retaining 5 GiB process budget and 0.75 GiB process headroom.
Shared replay dependencies and source-collection limits were not changed.
Final full verification passed 4,224 tests, 10 skipped, 133 warnings, in 1,755.08
seconds. Log: `data/runtime/swing_percentage_memory_full_tests.log`; JUnit:
`.test-tmp/swing-percentage-memory-full.xml`. Ruff and strict mypy passed.

### Corrected Initial-Fit Data Verified

Implementation checkpoint: `7b5d834`, pushed to `er-intraday-refactoring`.

This section supersedes historical run statuses below. The frozen population
remains 586 identities and 586,305 decisions; no new exclusions.

| Component | Verified result |
| --- | --- |
| Predictor replay | All 551 groups / 59 months / 586,305 rows matched exactly; independent receipt verification passed |
| Outcome replay | All 59 months / 586,305 decisions matched original targets and specifications exactly |
| Monthly news preparation | 59 months / 119 source-month lineages; original events and scores preserved |
| Identity alignment | 445 proven interval mappings; 854,541 translated and 102,720 unchanged relation rows |
| Source-clock recovery | All 136 compact batches checked; 2,120 identity clocks restored exactly from pinned original UTC records |
| Monthly news authority | All 59 months published; 224,709 canonical decisions have attributed news |
| Technical feature join | 586,305 rows; 479,709 feature-eligible; 298,446 complete model-input rows |
| Catalyst feature join | 586,305 rows; 479,709 feature-eligible; 296,137 complete model-input rows |
| Saved-row verification | Zero feature-clock violations; exact original outcome values; matching profile populations; zero outcome-filtered rows |

Completed joined manifest:
`data/features/swing_corrected_initial_fit_research/_manifest.json`,
SHA256 `63574b3b55cd30adaebf62b4040f92a2030e6dbb17772014afb3e3c168158256`.
Separate audit:
`data/reports/swing_initial_fit_join_verification.json`,
SHA256 `3e4a3c9dd241b9776fc368e7856e45d4bff0ed3cd0ef1431750fa94b16ce53b0`.

Each profile has 378,037 complete stock/SPY/QQQ/sector comparisons. The intersection
of feature eligibility, complete model inputs and complete comparisons is 194,679
technical rows and 193,125 catalyst rows. These counts do not constitute a new
training-row selection rule: source, feature, label and training admission remain
separate, and no rows were deleted from the publication.

Alpaca coverage is known for 497,805 one-day and 496,536 three-day windows.
SEC and Finviz coverage flags are unknown throughout this specific monthly news
authority. They are not silently added as model inputs. The SEC metadata archive
exists separately; this publication is not proof of historical causal SEC features.
Historical backfilled news uses publication-proxy availability with fixed inference
latency, not proven historical first receipt. Training/promotion flags remain false.

The rejected 7,521-catalyst-decision generation used mismatched news/candle
identifiers and is superseded. Exact interval/ticker/CIK proof repairs those joins
without guessing historical hash/CUSIP identities or reinstating excluded FI queries.
A separate compaction defect stripped timezone metadata after a null-only first
slice; the writer now validates every nonnull UTC representation before choosing
UTC nanosecond schemas. Original compact values and recovery source hashes remain
explicit evidence.

Distinct publications sharing model-input text no longer require identical
per-event score/relevance values. The frozen named policy selects the existing
earliest-available original instance, then source priority and event ID for ties.
Exact durable-event/source-event conflicts still fail. There is no rounding,
averaging or numeric tolerance. Full inspection found 26 distinct-publication
decision/window groups with different sentiment values and four with different
relevance; the cause of the numeric differences is not established.

Independent review closed all supported repairs; reviewer processes are closed.
Focused verification passed 96 clock/identity tests, 134 integration tests and
27 final authority tests. Final repository Ruff and strict mypy pass on 393 source
files. The complete run finished in 2,376.03 seconds: 4,145 passed, one failed,
10 skipped and 132 warnings. The failure was a required continuity-document
status marker removed during consolidation, not source or numerical behavior.
After correcting only documentation, both continuity tests passed in 0.09 seconds
(`.test-tmp/completed-news-doc-fix.xml`). The original full-run evidence remains
`.test-tmp/completed-news-join-full.xml`; it is not relabeled as all-green.
Skips cover eight unavailable Windows symlink cases and two opt-in memory stress
tests. No source code changed after the complete run.
Final documentation and package-boundary checks passed 221 tests in 11.17 seconds
(`.test-tmp/completed-news-doc-closure.xml`); local Markdown links also resolved.

Population completion does not certify managed-exit readiness, training admission,
historical first receipt, held-out evaluation or profitability. Unavailable
corporate payouts, unsupported issuer bars and other unresolved observations remain
explicit rather than fabricated. No model has been trained by this delivery.

### Dated Issuer-News Correction And System Memory

September 10 collection fills the missing direct SATS and pre-transfer FISV request
intervals without inventing historical index membership. New scope:
`configs/swing_issuer_news_corrections.json`, SHA256
`2e0cd0279dbeb1ac5a3c4224bdd4a8de78c0c5b603d08f14a0da2f86f1c1e3ed`.
Its official-source byte pins were verified before collection and in offline replay.
The separate raw archive is `data/raw/swing_issuer_news_corrections`.

All 105 chunks completed, 92 with events and 13 observed-empty, with no failed
requests. Offline audit passed all 105 pages: **661 unique events**, FISV 456 and
SATS 205, no duplicate event IDs. Publication range: July 15 2019 through May 28
2024. Manifest file SHA256:
`4da07ce57bf864845b79b0043816e49c104dc047eb81a30372ad952519f855d8`.
Audit `data/reports/swing_issuer_news_corrections_audit.json`, SHA256
`b4875e7cbf3334a346fff3047b5202c0c21a0801c473e01fcd9d7110bd202ecb`;
matching CSV SHA256 `03a612d136cfd58d8cb35efb640f91d368a683eee9018113a84ddfbbb6015851`.

Integrity passed is not catalyst admission. The existing auditor's empty-chunk
heuristic marks both issuers as coverage blindspots; its proposed exclusion wording
does not authorize changing the frozen cohort. An empty provider response alone
does not prove missing history or that no real-world articles existed. No new
exclusions were made. CIK/CUSIP reconciliation, article-level relevance and causal
feature attachment remain unadmitted. Existing ECHO aggregates are not reused.

One worker and 31-day chunks were used. Collector peak memory was 0.245560 GiB;
offline audit peak was 0.244118 GiB. The news CLI now checks system-wide available
physical memory at startup and cooperative collection boundaries: stop below 2 GiB
free or at 85% use, preserve saved pages, and cease new scheduling. Existing process
RSS protection remains unchanged. These are sampled guards, not hard allocation caps.

Verification: 32 focused tests, 227 dependency/sentiment tests; repository-wide Ruff;
strict mypy on 349 sources; full suite **3,289 passed, three skipped, 133 warnings**,
1,274.91 seconds, peak 0.331150 GiB. No code changed after the suite started.
The independent review's one supported replay-loop finding was fixed and tested.
Both agents and all test/collection/audit processes are closed. No new model trained.

### Completion Evidence And Approved Research Simulation

September 9 continuation retained 21 missing official SEC HTML documents using the
unchanged original-byte collector. All 21 pass offline receipt/body replay. These
are source collections, not admitted accounting authorities. The active plan's
combined source/outcome/feature delivery is still incomplete. No return, exclusion,
feature row or model changed in this source-only checkpoint.

Inventory files are `configs/swing_cash_merger_completion_documents.toml`,
`configs/swing_share_transition_completion_documents.toml` and
`configs/swing_remaining_merger_completion_documents.toml`. Each has an identically
named archive under `data/raw/`. Independent collection replay hashes, calculated
with `json_sha256(verify_official_document_collection(...))`:

| Inventory/archive stem | Documents | Replay hash |
| --- | ---: | --- |
| swing_cash_merger_completion_documents | 3 | `29ee1cb81230136d7c77dccf81a9a418a58dfe2ab60dd74aec99212cb8160495` |
| swing_share_transition_completion_documents | 7 | `680686010346616e0e433681360519f2b69665a062f242c0ac50d83c526d9bc8` |
| swing_remaining_merger_completion_documents | 11 | `194ee145ded436211a3a0a3edbd10629ac7de903edcc31ea2d46e8c28adaab30` |

Reviewed completion terms below are factual interpretation notes, not executable
event instructions. Document IDs resolve through the pinned receipts above. Legal
conversion, trading transition, source acceptance, claim valuation and cash/share
delivery are separate facts. Ordinary common shares are intended, subject to each
filing's exclusions; employee awards, preferred shares and partnership units must
not supply their exchange ratio. Cash amounts are USD contractual amounts, not marks.

| Affected shares | Completion document ID / section | Reported conversion or distribution |
| --- | --- | --- |
| ABMD | abiomed_merger_completion, Item 2.01 | Dec 22 2022: $380 plus one non-tradeable CVR; $35 is its contingent cap, not value. |
| ATVI | activision_merger_completion, Item 2.01 | Oct 13 2023: right to $95. |
| TIF | tiffany_merger_completion, introduction / Item 3.01 | Jan 7 2021: right to $131.50; suspension before that day's open. |
| SATS | echostar_bss_completion, transaction / Two-Way trading | Sep 10 2019: retains SATS, distribution leads to 0.23523769 DISH Class A; regular-way SATS carries rights through that close. Subsidiary filer CIK 1533758 is not SATS issuer CIK 1415404. |
| APC | anadarko_merger_completion, Item 2.01 | Aug 8 2019 at 10:41 a.m. Eastern: $59 plus 0.2934 OXY; fractional cash separate. |
| RTN | raytheon_merger_completion, Item 2.01; completion release | Apr 3 2020: 2.3348 UTC shares, renamed RTX; RTN stopped before open. Earlier UTC spin-offs do not belong to this RTN conversion. |
| CBS / VIAB | viacom_cbs_merger_completion, Item 2.01 | Dec 4 2019: CBS Class B remains 1:1; VIAB converts at 0.59625 into corresponding Class B. Do not apply VIAB ratio to CBS. |
| DISCA | discovery_warner_completion, Item 3.03 / Item 8.01 | Apr 8 2022: 1:1 WBD; WBD trading begins Apr 11. Spinco's 0.241917 ratio is not DISCA's. |
| CERN | cerner_merger_completion, Item 2.01 / Item 5.01 | Jun 8 2022: $95; tender acceptance is not payment proof. |
| ALXN | alexion_merger_completion, Item 2.01 | Jul 21 2021: $60 plus 2.1243 AstraZeneca ADSs, with holder ordinary-share election; Jul 22 internal merger is not conversion. |
| KSU | kansas_city_merger_completion, Item 2.01 | Dec 14 2021: $90 plus 2.884 CP common shares; voting trust is a separate step. |
| FLIR | flir_merger_completion, introduction / Item 2.01 | May 14 2021, approximately 9 a.m. Eastern: $28 plus 0.0718 TDY; preserve approximate clock. |
| MXIM | maxim_merger_completion, Item 2.01 | Aug 26 2021: 0.63 ADI plus applicable fractional cash. |
| CXO | concho_merger_completion, Item 2.01 | Jan 15 2021: 1.46 COP. Item 3.01's January 19, 2020 year conflicts with completion chronology; do not silently repair source text. |
| VAR | varian_merger_completion, Item 2.01 | Apr 15 2021: right to $177.50. |
| DRE | duke_merger_completion, Item 2.01 | Oct 3 2022: 0.475 PLD plus fractional cash; Prologis is the filer. |
| MYL | mylan_combination_completion, Item 2.01 | Nov 16 2020: one Viatris share per Mylan ordinary share; Pfizer distribution terms and timestamp are separate. |
| INFO | ihs_markit_merger_completion, introduction | Feb 28 2022: 0.2838 SPGI plus fractional cash. |
| PBCT | peoples_united_merger_completion, introduction / Item 2.01 | Apr 1 2022: 0.118 MTB plus fractional cash; Apr 4 suspension/report date is not closing. |

No entry above establishes holder-level delivery, fractional-sale proceeds or cash
spendability. The ABMD completion evidence now resolves the earlier proposed-only
limitation; it does not resolve the CVR valuation. Bank halt/recovery cases, full
cohort/ETF action completeness and post-removal class ownership remain unresolved.

The independent feasibility review identified a real contract conflict: hypothetical
sales cannot have observed broker receipts. The user subsequently approved explicit
simulation of hypothetical fills, costs and funding with real historical inputs.
The implementation now separates ordinary-sale assumptions from observed evidence;
unknown corporate payouts and contingent valuations remain unavailable. Verification
completed September 10: 140 focused tests; repository-wide Ruff; strict mypy on 347
sources; full suite 3,260 passed, three skipped, 133 warnings in 1,536.28 seconds.
Peak test-process memory was 0.331982 GiB. No code changed after full-suite start.
The independent reviewer closed all three supported findings: payment-assumption
provenance through cash release, consumed-only research maturity clocks, and retained
per-lot settlement/replay metadata. Both agents are closed. No source-admitted
full-cohort rebuild or new candidate is asserted.
Next-session reuse is not historical settled cash: the US settlement transition to
T+1 was May 28 2024 ([SEC](https://www.sec.gov/newsroom/press-releases/2024-62)).
Provider pay dates do not guarantee account credits
([Alpaca](https://docs.alpaca.markets/us/docs/daily-processes-and-reconcilations)).

The earlier inventory identified missing SATS/FISV request windows, now collected
as described above. Existing Alpaca/SEC archives remain reusable, but old aggregate
rows cannot certify the corrected issuer bridge. All nine existing technical
relationship outputs already occur in the 120-column comparator; adding them again
would not create an incremental feature profile. These findings are prerequisites
for the feature rebuild, not evidence that it has run.

### Previously Closed Compiler

The current fixed-horizon compiler has produced two actual holding specifications
and canonical kernel replays from the corrected SATS and FI archives. Output:
`data/reports/swing_fixed_holding_demonstration.json`; file SHA256
`92c288af8ba12d0dd0b84fdc28a75bc4df1f924e3a2adc7f50d616858c70cbd6`,
semantic audit `84c032008674fef70ebcef475b37925ea918c8bc1ce9297d2b3195209f791fa8`.
Both lots have ten raw closing observations and no materialization gaps, but
`independent_source_admission_required` and `action_coverage_unproven` remain.
Reportable returns are null and all eligibility flags are false. This verifies
source-to-kernel compilation, not complete distributions, managed exits, benchmark
targets, feature acceptance or model training. Real-run peak memory: 0.398182 GiB.
The original raw/corrected archives are unchanged; no additional downloads/exclusions.

Independent source inventory also identified specific evidence limits: the retained
TWTR closing 8-K states October 27, 2022 effectiveness while the provider record says
October 28; neither is cash availability. The retained ABMD filing describes a
proposed transaction and cannot prove closing or a CVR value. The SBNY-named file
`data/raw/sec_identity_evidence_20260802/SBNY_0001380846-22-000022_targeted_tsc-20211231.htm`
is actually TriState Capital's filing (CIK 1380846), not Signature Bank evidence.
None was admitted by this compiler. Full-cohort/ETF action coverage, SATS's BSS
distribution, successor delivery and the remaining holding-tail facts are unresolved.
Do not treat a provider payable date, options notice or contractual face/cap as
spendable cash, an executable share price or an independent claim mark.

The September 8 user-approved direction permits a new retrospective whole-security
restriction, rather than repairing every old holding before any new dataset.
Its identity-only audit projects 85 monthly partitions: eighteen additional IDs
would remove 10,971 of 853,417 retained-parent rows, leaving 842,446 rows across
586 securities. These are proposed coverage counts, not newly built feature rows.
Cumulative exclusions are 45/631 (7.13%), including 27 inherited failures. The
user approved 10% on September 8; the new immutable audit accepts this research
restriction. No changed stock list, original denominator or performance gate was
authorized. Coverage acceptance is not feature/label or total-return acceptance.
No outcome column was read to choose exclusions or compute this coverage report.
Raw sources and the original 70-path failed control remain unchanged. A frozen
restricted universe must be rebuilt before peer transforms and labels; it cannot
be retrospectively represented as a historical investable universe or a fresh test.
Stock/benchmark price-basis acceptance remains independent and unresolved.

The retained-population holding-identity preflight now covers all 85 months:
836,638 mature windows covered, 947 uncovered across 100 securities, 4,861 terminal
immature decisions. Initial-fit coverage is 580,889 covered and 566 uncovered
across 60 securities. This uses only membership/clock metadata: it is not proof of
missing prices, delisting, losing outcomes or feature acceptance. The uncovered
share is 0.1131% of mature decisions. No additional securities were excluded.
Report: `data/reports/swing_research_cohort/holding_identity_preflight.json`, SHA
`8f8cdd60ca2f9c1372ceda20d04b7f90eefbfb2336a7f282f30aa97b8c7e723b`.
The initial-fit raw-observation inventory now reproduces those 566 decisions / 60
securities using identity-only parent projections and exact required-date Arrow scans.
Report: `data/reports/swing_holding_observations/_manifest.json`; independently retained
audit SHA `b87e18fc2c706600c0063c606c2dd40dbba0422cfad6b6c8c1e4817242714a8a`.
Of 1,106 distinct security/ticker sessions, 876 observations are valid, 209 missing
and 21 invalid. Ownership is unresolved for 574 sessions, independently of validity.
Missing/invalid observations affect 23 IDs; 37 have valid observations for all
requirements. These are not new exclusion decisions. Numeric held-out rows were
not returned. The original collection JSON hash format is verified without rewriting
the source. Immutable replay passed; peak inventory memory was 0.342045 GiB.
Remaining: independently effective-dated security/class ownership, unavailable
trading/corporate actions and verified stock/benchmark total-return accounting.
The diagnostic is not a complete shard outcome authority and cannot train a model.
SEC relations copied from S&P membership intervals do not prove post-removal ownership.
Holding-observation checkpoint verification: full Ruff and strict mypy on 331 sources pass; full suite
passed 2,625 tests, three skipped, 133 warnings in 1,329.16 seconds (22m09s), with
0.326073 GiB peak memory. Consolidated review findings have regression tests for externally pinned
replay and null start-time corruption. No feature admission or promotion changed.

The dependent initial-fit corporate-action collection now acquired all 60 queried
tickers, and independently pinned offline replay reproduces audit SHA
`82dafea3055db20db9ee483800a7a22c508dbb325418b979a1cd23974a244c91` under
`data/raw/swing_holding_corporate_actions/reports/`. Status is
`collected_unreviewed`, not accounting or feature acceptance. Its 649 distinct
provider records comprise 621 cash dividends, 21 mergers, five name changes and
two spin-offs. Query coverage is process-date 2019-07-09 through 2024-05-28, not
announcement or effective-date completeness. Collection peak was 0.237537 GiB;
offline replay peak was 0.233650 GiB. Both workers exited.

Read-only event review found 16 mergers within the affected holding windows;
14 lack payable dates. Five other mergers are outside those windows. The five
name changes lack effective dates. Cross-query participant matching is necessary:
the MBC response includes the FBIN/FBHS spin-off, whereas an acquirer's historical
purchase does not terminate the acquirer. Among 37 all-valid observation cases,
19 have same-CUSIP dividend anchors on both sides of their windows; these do not
prove uninterrupted class ownership. NOV/HFC CUSIP differences, MYL's VTRSV versus
VTRS trading boundary, and a FLIR dividend after its merger need corroboration.
VIAB, RTN, APC, ABMD, SIVB and SBNY lack a resolving in-window merger in this
collection. Missing payment/delivery, currency/unit and unavailable-trading facts
remain unavailable, not invented fills. No exclusion or feature-source set changed.

Collector review fixes are covered by tests: independently pinned online resume,
replayable malformed provider metadata and HTTP error bytes, future-clock rejection,
and propagation of the global typed memory-budget exception. Integrated focused
verification passed 213 tests; full Ruff and strict mypy on 333 sources pass.
The complete suite passed 2,799 tests, three skipped, 133 warnings in 1,088.81
seconds (18m08s), peak 0.326954 GiB. Test PID 35496 exited. No model was trained.

### Raw-Share Collection Dependency

Initial-fit planning implementation now reconstructs exact decision/holding session
unions and complete benchmark ranges with a per-unit count/digest audit. The collection
CLI replays those requirements under the workspace lease using an independently saved
authority-file hash and frozen provider mapping. It rejects scope downgrade and
authority replacement before writes/dispatch. Shared publication remains one owner.
No ownership, cash availability or accounting eligibility is inferred from this plan.
Verification: 358 focused tests; full tracked-Python Ruff; strict mypy 338 sources;
full suite 3,082 passed, three skipped, 134 warnings, 1,266.62s, peak 0.329632 GiB.
PID13376 exited; XML `.test-tmp/raw-plan-full.xml`. Both agents are closed.
Real identity-only reconstruction: 545 in-window IDs, 551 stock plus 13 benchmark
units, 586,305 decision sessions, 586,414 holding sessions, 551 decision-only sessions,
586,965 required stock sessions. Published plan:
`data/reports/swing_initial_fit_raw_share_plan`; independently saved authority-file
SHA `d912a997af361c820745e8c850f0fd455ee068397c6d034d2b54c00a475dc22e`.
Collection and fresh offline replay both completed for all 564 units / 601,834 rows
at `data/raw/swing_initial_fit_raw_share_daily`, with no failed/empty units. Collection
authority-file SHA `144cab43741f3c74308b53e9322c84ac7eeaca158d3cbd6a1ddb0d7c0fa8d244`;
manifest-file SHA `a96c70eee46e0a4b4d0ee3c1a7d47cea40dfc7a4ec2e14903f03e786bd0ced52`.
Plan peak 0.396503 GiB; collection peak 0.395340 GiB. PIDs35088,20404,31288 exited.
The row-count check is separate from admission: 602,968 required stock/ETF sessions
versus 601,834 returned rows, a shortfall of 1,134 across 28 ticker ranges. ECHO has
630 fewer rows, FISV 245; 26 other ranges have nine or ten fewer. This is not yet an
exact missing-session, effective-dated symbol, ownership or corporate-action audit.
No new security was excluded, no return was fabricated and no candidate was fitted.
The following transport evidence remains the preceding checkpoint.

The existing exact-unit collector now derives `raw`/`all` from its verified plan,
retains original transport bytes and query receipts for new acquisitions, and
reconstructs normalized Parquet values from those responses during replay. Consumers
declare their required price basis; adjusted feature-history combination requires
`all`. Historical adjusted archives lacking receipts remain historical evidence,
never raw-share authority. Rejected-query receipt retention is regression-tested.

Focused source/collector tests: 53 passed; combination tests: 14 passed; full
tracked-Python Ruff and strict mypy on 336 sources pass. Retained adjusted warm-up
replay passed for 549 units / 140,383 rows, peak 0.132084 GiB, PID12624 exited.
The full suite passed 2,999 tests, three skipped, 134 warnings in 1,181.30 seconds
(19m41s), peak 0.329193 GiB. PID25540 exited; XML
`.test-tmp/price-basis-full.xml`. Both preceding agents are closed. Initial-fit plan
publication and acquisition are now complete as described above; source/ownership/
action admission and actual target materialization remain pending.

### Event-Aware Accounting

The user approved replacing price-only accounting with event-aware accounting on
September 9. Its calculation/target/funding code is implemented and fully verified;
it is not yet a real-data feature/label authority. It must
separate tradable shares, available cash, unpaid proceeds and contingent rights,
keeping unsupported dates/valuations unavailable and preserving the ten-session
forecast horizon and 10% exclusion cap. The [ABMD completion filing](https://www.sec.gov/Archives/edgar/data/815094/000119312522311074/d353287d8k.htm)
describes $380 cash plus a nontradeable contingent value right, capped at $35.
That cap is not its valuation or cash available to reinvest. The current diagnostic
ledger could not represent this entitlement independently of terminal spendable cash.

The replacement uses strict raw-share specifications with class-owned events,
explicit marks, payment/availability evidence and a research-contract hash. One
lot calculator feeds nullable fixed/managed stock and benchmark target fields and
the existing single funding loop. Component NAV includes known tradable, unpaid
and contingent values; only actual available cash funds purchases. Unknown marks
stop NAV-dependent allocations without deleting the affected decision. Residual
marks can continue through a portfolio endpoint while the model target remains
ten sessions. Benchmarks retain cash distributions without inventing reinvestment.
All source/production eligibility flags remain false pending independent admission.

Consolidated review findings have regression tests: ambiguous entry-time actions,
prior-session sale proceeds available before an opening purchase, unearned post-sale
event metadata poisoning labels, and prematurely referenced successor positions.
Managed exits can resolve on verified bars before a later missing bar; fixed-horizon
coverage remains independent. Unsupported post-horizon managed exits are rejected.
Integrated verification passed 309 tests; full tracked-Python Ruff and strict mypy
on 336 sources pass. The complete suite passed 2,974 tests, three skipped, 133
warnings in 1,096.07 seconds (18m16s), peak 0.328899 GiB. PID 23576 exited; XML
`.test-tmp/event-accounting-full.xml`. All implementation/review/source-advisor agents
are closed. The next raw-source work must not represent this mathematical verification
as completed independent evidence admission.
No real raw/adjusted source was rewritten, no new whole-security exclusion was made,
and none of the six new return regressors has been fitted.

Previous membership-preflight checkpoint: 2,502 tests passed, three skipped, 132 warnings; full Ruff
and strict mypy on 328 sources pass. Focused tests passed 133 cases. Independent
review/test findings were fixed for partial-session competing ownership,
open-ended interval checks, same-owner metadata changes and duplicate decision IDs
across monthly partitions. Test peak was 351,141,888 bytes (0.327 GiB); all owned
workers and agents exited. No new model or economic result was produced.

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

The bounded transfer replay now retains all eleven configured, historical-`asof`
Alpaca windows at `data/raw/swing_transfer_history`. Offline replay reproduces
report file SHA-256
`b8daebd9322da94bd10617d54a1a9cc4c9eb07baa4459f237d76300bc3797198`.
All 140 required ticker-sessions have positive-volume, valid daily observations.
CPRI, DXC, KSS, NKTR, SEDG, SLG, WU and XRX match retained OHLCV exactly (26 selected
decisions, 101 ticker-sessions). ADS, GPS and PRGO have matching volumes but 60,
40 and 56 differing OHLC fields respectively (twelve decisions, 39 ticker-sessions).
No source was overwritten, rescaled or admitted. Price differences remain a source
reconciliation issue, not a reason to exclude previously selected holdings.

The source decoder rejects duplicate JSON keys and colliding normalized symbols;
offline replay uses that same decoder rather than trusting cached parsed values.
SEC stock-class facts now extract for all eleven targets from hash-bound filing
bodies. Nine missing filings and the LB rights announcement were acquired using
`configs/swing_transfer_identity_documents.toml`; ten-of-ten report file SHA-256:
`e09a8ee202c6a195f1c7d2c67b830c96e1efae31f16296b644d7b5f01e70e87a`.
ADS and GPS filings were reused. Facts and matching prices are inputs to the
separate retrospective identity interpretation, not an already published relation.

Canonical labels now resolve separate security-identified outcome bars using XNYS,
not observed stock or SPY row counts. Membership end no longer suppresses expected
holding outcomes. Fixed/managed paths and live maturation reject unusable daily
observations; precise daily open/close timestamps include early closes. This code
correction does not certify historical identity or corporate-action return units.

The original SEC inventory has accession metadata but not the filing bodies needed
for accounting. The new official-document acquisition retained six configured
responses at `data/raw/swing_holding_source_documents`, verified offline. Its
immutable report file SHA-256 is
`3b5dbc4b888034d7c36f2275ec40d49710778b4a39ed5c93503b3ed52654f52c`;
inventory SHA-256 is
`a32b83ac554047fcdc7383af4a5870549d7c21b2b0f4c2a7f0dac53de5484391`.
It contains six `archived_unreviewed` acquisitions, two failed requests (BBWI tax
notice transport failure and OCC HTTP 403), and two unattempted OCC documents
deferred after the 403. No error-response body was claimed as archived.

The retained six are LB/BBWI separation, AMD/XLNX closing, CTXS closing, CTXS Nasdaq
cessation, NLSN closing and FRC FDIC receivership. Bounded local body inspection
confirmed relevant stated terms, not a completed financial interpretation. These
documents still **do not authorize outcome, identity or total-return admission**:

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
  not retained because the OCC host was deferred. The separately retained FDIC
  receivership page does not justify assuming common shares were worth zero.
- [BIIB halt notice](https://infomemo.theocc.com/infomemos?number=47809):
  not retained because the OCC host was deferred. An options-processing reference
  price is not an executable stock entry.

Independent identity review adds an important distinction to the 40 price-complete
paths: 38 across 11 tickers have explicit retained S&P index-transfer announcements,
while two AIV paths involve a documented planned AIRC spin-off. All twelve raw
artifact legacy membership IDs differ from current decision IDs. SEC relation
intervals end at membership removal because the existing constructor copies those
boundaries; they cannot independently authorize the missing tails. The absence of
a transition row, current CIK equality, matching prices or an arbitrary ten-session
extension is not historical security-class continuity evidence.

AIV's retained S&P announcement body is
`data/raw/index_membership/spglobal_official_20180414_20260708_v1/objects/6a/6a14c38afcc16ae39f5d32698ab6e21561d91568746afaed6eb5717b72329a19.html`
(the filename digest is its verified SHA-256). Its December 11 announcement expected
the spin-off after the December 14 close; it does not prove completion. Existing
SEC metadata explicitly records candidate filings
[December 15 8-K](https://www.sec.gov/Archives/edgar/data/922864/000119312520317477/d772672d8k.htm)
and [December 16 8-K](https://www.sec.gov/Archives/edgar/data/922864/000119312520319176/d71118d8k.htm).
Both candidate bodies were subsequently acquired, without changing the ten-document
request, using `configs/swing_aiv_distribution_documents.toml`. Their separate raw
archive is `data/raw/swing_aiv_distribution_documents`, with a two-of-two unreviewed
acquisition report SHA-256
`41e6324d175dfb17cd89d0f78aa371aceca3aa3b6c2b4b53a4ef0ebfa2c73904`.
Acquisition does not establish completion, entitlement or return treatment; semantic
review remains required. The already-retained S&P announcement was not downloaded again.

The independent quant review inspected all eight retained bodies and verified their
receipt/body hashes. The December 15 AIV filing establishes one AIR Class A share
per AIV Class A share; AIV continues. The December 16 filing concerns financing,
not another distribution. Ex-distribution and regular-way rights still require
evidence. LB's similar entitlement issue has a supplemental official
[regular-way trading announcement](https://www.sec.gov/Archives/edgar/data/701985/000114036121023926/nt10022999x10_ex99-1.htm)
subsequently retained in the transfer identity document collection above. Record
dates alone cannot decide these rights; retaining the body does not implement its
entitlements or approve an accounting convention.

Three unresolved policy boundaries are now explicit: cash-merger receivable versus
spendable funds; fractional/distribution reinvestment convention; and mandatory
termination or halt nonexecution/valuation. Normalized total-return units can be
researched without personal brokerage records, but synthetic next-session merger
redemption is not verified settlement. Likewise, adding component daily highs/lows
does not produce a synchronized basket barrier path. An asynchronous user decision
was requested on a separately named synthetic research convention; until approved,
the frozen execution-based rules remain unchanged. Identity/source work may continue.

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
