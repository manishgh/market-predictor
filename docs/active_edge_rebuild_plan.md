# Active Edge Rebuild Plan

## Ordered execution plan — October 9

Goal: learn future stock returns from existing price/fundamental state and all
relevant news available at each decision, then measure the result against SPY.
Collection, raw evidence, matched candles/news, technical indicators and API V1
remain fixed. TradingFlow remains independently owned. Tests alone are not a win.

| Step | Status | Work and concrete completion condition |
| --- | --- | --- |
| 0. Reuse the foundation | Done | Existing matched 586,305 decision rows / 59 months and four fitted baseline models are retained. No new collection/schema/matching rebuild. |
| 1. Streaming quality-check buckets | Software and real pilot integration done; full-corpus run pending | Helper 44d8c2d retained 309 QA buckets during the real 190-version pilot; 28 unit tests pass. Sampling remains separate from the full training-news stream. |
| 2. Broad news feature enrichment | Cue component and real pilot done; quality/attribution pending | Implement one feature-layer recognizer for multiple swing-news categories and mixed signals. Preserve source evidence, availability and uncertain/other stories. Connect buckets to a real retained-news pilot; report actual coverage and examples. |
| 3. Full-corpus enrichment | In progress | Stream every eligible retained article/version through category/sentiment processing in bounded batches. Reuse existing FinBERT outputs/model where available. Persist enriched output and QA references with resumable batch accounting; no unsampled-row filtering. |
| 4. Prediction-time news features | Pending | Aggregate company/sector/market signals and recency at each existing decision cutoff. Combine with existing price/fundamentals. Later revisions/prices cannot enter earlier inputs. Preserve positive earnings and negative guidance simultaneously. |
| 5. Freeze the third feature profile | Pending | Record exact columns, definitions, missingness, source coverage, clocks and train-only preprocessing. Embeddings are optional feature work, not assumed implemented; any encoder must be locally available/pinned and evaluated within the amended profile. |
| 6. Fit remaining two models | Pending | Fit existing Ridge and shallow XGBoost for the amended news profile on the same official ten-session targets, splits, weights and costs. Four fitted baselines remain unchanged. |
| 7. Compare performance | Pending | Evaluate all six specifications and twelve frozen policies with prediction error, ranking, funded NAV, fees, turnover and drawdown against SPY. State measured losses as plainly as gains. No claim of outperformance from a feature/test pass. |
| 8. Independent final assessment | Pending | Preserve genuinely unseen assessment data. Previously repeatedly inspected historical test remains development evidence. Follow the existing prospective assessment requirement without fabricating future observations. |
| 9. Accepted prediction API V1 | Pending | Historical/live features use the same transformation and ordering; signed numerical returns, missing inputs and acceptance are explicit. Coordinate TradingFlow long/swing consumption without modifying its active work unilaterally. |
| 10. Merge and push accepted work | Pending | Publish reviewed implementation checkpoints on the current branch; merge accepted final work to main and push only after its required data/model/consumer conditions are actually satisfied. |

### Step 3 scope and execution — October 9

Reuse the retained population.sqlite version inventory and existing FinBERT outputs.
Do not repeat collection, candle/news matching, base schemas or four baseline fits.
Three original sentiment archives are available (early, later and corrected issuer
queries); their raw row totals overlap across companies and include dates beyond
this experiment. Those totals are not current eligible unique-article coverage.

1. Bind each reused score through its pinned original canonical event artifact,
   original event ID/query identity, raw article-version hash and scored input hash.
   Reconstruct title/summary input with the existing transformation. Missing or
   mismatched scores remain explicit; never join merely by ticker/date/story ID.
2. Stream every retained version into bounded durable shards, including dispositions
   for missing text/clocks. Keep lexical cues and sentiment availability separate:
   a cue can be available before the configured sentiment processing delay expires.
   Preserve historical proxy semantics and unresolved company/event attribution.
3. Resume only verified completed shards under the same source/code/options identity;
   account for every input version and reproduce bounded QA references from outputs.
4. Run focused unit/clock/tamper/resume checks, lint/types and one consolidated
   review, then the real full-corpus job under the shared heavy-job lease and existing
   memory guard. Report actual coverage, missingness, limits and source examples.
5. Close this checkpoint with pushed code/evidence and proceed to decision-time
   aggregation. Cue recognition is not measured event-extraction precision, embeddings
   or model outperformance. Those claims require their own observed evidence.

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

### What is genuinely done

- Existing technical and relationship source/feature work, matched input reuse,
  four fitted baseline models and their temporal evaluation runs.
- Completed source-only A/B reviews: each 1,750 packets / 1,977 versions; no missing
  or partial inspections. They assess earnings/guidance, not all new categories.
- Old annotation publication exists (603 candidate event occurrences, not articles);
  its strict identical-span result is superseded and its independent replay failed
  the RAM guard. It is historical evidence, not source or model acceptance.
- FinBERT scoring implementation and existing generic aggregate feature plumbing.
  A full semantic article-embedding store/training integration is not established.
- QA bucket helper: deterministic bounded reference sampling, multi-category and
  uncertain/mixed cases; final 28 synthetic unit tests, Ruff and strict mypy pass.

### Step 2: bounded implementation now

Problem: the current explicit content extractor only recognizes earnings/guidance.
Contracts, acquisitions, analyst changes, products, legal/regulatory actions,
financing/capital returns, management/operations and sector/market context can be
useful swing signals and must be represented. One article can contain several.

In scope: feature-layer text/cue recognition and category/action evidence; separate
business direction/status from predicted stock return; source/version references;
company versus sector/market/unresolved scope; publication-time availability;
category-specific missingness, no universal fiscal-year demand; bounded QA bucket
integration and a real retained-input pilot. Reuse existing sentiment values when
available; unavailable scoring must stay unavailable, never synthetic or zero.
A heuristic cue is not a verified world event: label its method/uncertainty and do
not declare new-category source quality solely from matching a keyword.

Out of scope: raw collection, canonical base schemas, new alias/compatibility
layers/API versions, rematching candles/news, rewriting A/B judgments/hashes,
additional learners, changed targets/costs/splits, CPCV, forced future-price
correlation labels or manual changes to news sentiment based on observed returns.

Exit checks: source-grounded positive/negative/ambiguous examples for each category;
mixed earnings/guidance and macro stories; contract gain versus loss; rumours,
negation and completion status; analyst actor versus affected company; authentic
evidence references; after-cutoff/later-revision refusal; no fiscal requirement
for nonfinancial stories; deterministic bounded QA samples. Actual pilot must run
on retained data using real code, state its input/output counts and exclusions,
and preserve the existing inputs. Unit fixtures cannot substitute for that pilot.

Do one bounded design/code review, fix supported findings, run affected unit/source
checks and lint/types, commit and update this plan/handoff before the next step.
If a prerequisite is missing, name it exactly and continue independent authorized
work. Do not replace feature work with another foundation project.

Below are retained detailed/historical records. Their dated statuses are not
additional current to-do lists. This ordered table and the one checkpoint below
control execution; original artifact hashes and completed evidence are unchanged.

Current checkpoint: **Full-corpus bounded enrichment and existing sentiment integration** (`in progress`).

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

### Historical continuation: source reviews complete; request comparison repaired; real retry running

Step 1 remains complete: each independent model-assisted source-only set covers
1,750 packets and 1,977 versions with no missing or partial inspections. Their
adopted files, judgments, config and physical hashes remain unchanged.

Step 2's real run exited with code 1 at 2026-10-07T03:35:09.8381625Z. The canonical
packet verifier completed, then annotation ingestion failed at _inputs with
"verified packet request differs" before creating its private output stage.
The saved request has JSON lists for frame_policy.development_sources and
correspondence_policy.roles/verdict_fields; the verifier returns Python tuples.
Those installed policy values have exactly equal canonical JSON bytes but unequal
Python container types. This prevents annotation publication despite successful
packet byte replay. It is not missing credentials, source timestamps or reviews.

Bounded repair: compare the two requests using the existing canonical JSON encoder
at this new annotation consumer boundary. Preserve exact bytes, value types and
array order; do not normalize source content or change the frozen packet owner,
candidate owner, policies, hashes, labels, models, targets or TradingFlow.
Exit checks: unit reproduction using installed tuple-containing policies, rejection
of changed boolean/number values and array order, affected annotation tests,
Ruff/strict mypy, one consolidated bounded review, then actual publication and
independent verification with a fresh run inventory and natural exit code 0.
Preserve all failed-run state, logs, 27-file inventory and exit receipt. No bypass
of the canonical verifier or replacement of actual data with unit fixtures.

Software repair completed: canonical JSON request comparison fixes the observed
list/tuple mismatch without changing stored JSON or policy values. Implementation
commit: bf3bf5a. Before the fix, the unit publisher reproduced the exact
"verified packet request differs" failure using installed tuple-containing policies.
After the fix: 25 affected unit/document checks passed; targeted Ruff passed;
strict mypy passed for the changed module; one bounded independent review found
no supported issue. Unit fixtures are tests only, not actual ingestion evidence.
Frozen packet/candidate/original owners, adopted A/B reviews and config did not
change. The original actual failure and natural exit 1 remain preserved.

The fresh real retry is now active: native session38719, worker84748, started
2026-10-07T03:40:09.077611+00:00. Config hash remains
0d947752ad95397fda707b2f600593bcf9a15cf649fbed8f29dfc86640cfa902.
Its 27-file inventory hash is
6740805d6e13066eb7ef1e5f99623cf19c46a10b95e6b53f5186d8761e3b29bb.
Exactly one executed file differs from the failed run: the repaired annotation
consumer. Packet/candidate/original owners remain unchanged. Current phase:
publishing_actual_complete_source_annotations. No real publication/replay pass is
claimed yet. Do not change code, reviews, config or original data during the run.

Fresh retry runner: work/run_actual_issuer_annotation_ingestion_retry.py.
Fresh state/inventory/log/exit names use actual-issuer-annotation-ingestion-retry.
Retry must still run the installed publisher and independent verifier on real
retained data. A software pass alone does not close requested step 2.

### Historical continuation: source reviews complete; actual ingestion running

User scope: finish the two independent complete source-review sets and actual
installed ingestion/publication plus independent verification. Source review is
complete; the ingestion run is genuinely active, not a passing publication yet.

Actual combined strict DTO/source-quote check and adoption completed October 7 at
01:42 UTC: EACH reviewer has 1,750 packets / 1,977 version assessments, including
1,962 completely inspected readable versions and 15 metadata-only unavailable
versions. Zero missing packets or partial inspections. Complete source ambiguities
remain explicitly unresolved; unavailable text never became a negative. Judgments
are honestly model-assisted source-only inspections, not human-certified gold.
The assembly created no judgments and changed no supplied judgments.

Completed immutable copies: data/research/swing_initial_fit_issuer_complete_source_reviews
- reviewer-a.jsonl SHA256 24ceb555fdd9a4fc13cd9830ed4d259f92239dcda0c447a498c18a1241325ce1
- reviewer-b.jsonl SHA256 910222705531959b8b2e174117ec9023bfaa36c036c64eef4584d96f00190a2d
- _manifest.json SHA256 a740b463d25b071997e9c3c190a64167b958a48052d897b7cf7d147b0308d4d1
Source inspections preserve disjoint context receipts: A prefix[0,1000), A
tail[1000,1750); B main1,733 packets and B helper17 physically pinned assignments.
Both sets remained isolated from extractor/candidates, old development labels,
other independent reviewer judgments, models and outcomes. Own identical-text
read reuse for packet988 is explicitly hash/length/physical-key bound to own fully
read926 text with fresh target identity/clock/family/manual adjudication; no A/B
label sharing or blind label copying. Source receipts are copied with hash checks.

Strict config configs/swing_initial_fit_issuer_annotation_ingestion.json SHA256
0d947752ad95397fda707b2f600593bcf9a15cf649fbed8f29dfc86640cfa902.
Actual run started01:43:53 UTC, worker97320, native session50119. One heavy worker,
27 frozen executed files; workspace review-results/actual-issuer-annotation-ingestion-frozen-implementation.json
SHA256 f92e8885c576458164ea08b5728d3c3e3a4de56dfb4c5dcdeb5359f880abdb2d.
Current phase at this documentation checkpoint: publishing_actual_complete_source_annotations.
Installed publisher invokes the canonical packet verifier before processing, then
installed independent annotation verifier runs sequentially. No patched owner,
mock source, invented model/admission or skipped replay. Do not edit frozen code,
config, review files or original sources while this run is active.

Workspace run evidence: review-results/actual-issuer-annotation-ingestion-state.json,
actual-issuer-annotation-ingestion.stdout.log, .stderr.log, eventual .exit.json.
Prepared runner: work/run_actual_issuer_annotation_ingestion.py. Destination:
data/research/swing_initial_fit_issuer_annotation_correspondence (fresh publication).
Do not rerun the runner or overwrite state/private stage; wait on the owned process
and inspect actual completion/failure. Independent replay and natural exit remain
OPEN. Annotation publication alone grants no source qualification or model approval.

The adopted source-review/config component is complete. Existing software5cf8208
and its unit/lint/types/actual decoder evidence remain closed. No new quality metrics,
features, model fits, SPY improvement, serving or main merge are claimed. Four fits,
matched candles/news, targets, TradingFlow and main remain unchanged. Communicating
APIs remain V1. Later Astra metrics/event/features/model work follows these two steps.
Earlier partial-review/running-context claims below are historical and superseded.

### Historical continuation: finish source reviews and real ingestion only

User scope (October 6): finish the two independent complete source-review artifact
sets, then run actual installed annotation ingestion/publication and its independent
verifier. Do not start metrics, event features, model fits, serving or main merge in
this continuation. Four completed fits, matched candles/news and TradingFlow stay fixed.

Usage was refreshed to 0% consumed at resumption; no reset credit was redeemed.
The previously saved 99% checkpoint below is historical evidence, not current usage.
Implementation 5cf8208 remains pushed on codex/v1-canonical-cleanup; no repository
implementation was changed. Both source-review artifact sets remain incomplete.

Source-only reviewer continuations are running, with no candidate/extractor/outcome
or cross-reviewer judgment access. A is split into disjoint ranges to shorten wall
time while preserving its existing declared model-assisted reviewer identity:

- A main: sorted blind packet indices [0,1000), workspace work/source-review-fresh-a.
- A tail: indices [1000,1750), workspace work/source-review-fresh-a-tail; fresh
  source-only agent context, its own assessments and inspection receipts. It cannot
  read main A judgments. This is one continuation of A, not a third independent set.
- B: all indices [0,1750), workspace work/source-review-fresh-b. Resume its OWN
  noncontiguous completed IDs and remaining read ranges.

All workspace paths are rooted at
C:/Users/manis/Documents/Codex/2026-09-28/c. Each combined reviewer set requires
1,750 packets / 1,977 versions, including 15 metadata-only. Source year or explicit
end date is required for a fully identified fiscal period; never infer year from
publication. Read every readable version fully; keep uncertainty explicit. Never
auto-fill unseen negatives or copy judgments across reviewers.

Prepared workspace operational wrappers (no real ingestion launched yet):

- work/prepare_actual_issuer_annotation_ingestion.py: installed schema/quote checks,
  rejects missing/partial/duplicate/overlapping A ranges, preserves source judgments,
  and --adopt creates immutable complete reviewer JSONL plus hash-bound inspection
  receipts and strict physical config. Run only after all reviewer segments finish.
- work/run_actual_issuer_annotation_ingestion.py --expected-config-sha256 ACTUAL_SHA:
  frozen implementation inventory; installed publisher then independent verifier;
  single exclusive run owner and no retained-evidence overwrite. No mocked owners.

One bounded wrapper review found receipt pins and a competing-launch failure-state
overwrite issue. Both fixes were confirmed by static review; wrapper syntax compile
passed. Existing ingestion unit/lint/type/actual decoder evidence below stays closed.
No new source qualification, features, model acceptance or SPY improvement is claimed.
The current blocker for launching ingestion is incomplete genuine source coverage.

### Historical continuation: actual source-review progress saved at 99% usage

Last implementation 5cf8208 is pushed on codex/v1-canonical-cleanup; its software
evidence was recorded in 193e1a7. Complete-version annotation component 75791f4 is
closed. The real source-review/ingestion exit is still OPEN. No new accepted source
qualification, reaction features, model fits, SPY improvement or main merge exists.

Completed software and real checks:

- Annotation scope: 314 selected synthetic UNIT checks; Ruff/strict types passed;
  installed reader checked all actual 1,750 packets / 1,977 versions, including 15 metadata-only.
- Ingestion scope: 336 selected UNIT checks; supported pre-lock bulk-read finding
  fixed; final 21 affected units passed in 5.20s; Ruff/strict types passed. Bulk evidence
  reads require the processing lease; routing metadata is bounded beforehand.
- Actual strict candidate parser: all 14,182 retained candidate versions / 16,004
  candidates passed, with source manifest/DB unchanged. Final decoder/dependencies
  remain identical after the lock fix; original 133 and packet 24 owners unchanged.
  Reports: issuer-ingestion-actual-candidate-json-shape.json and
  issuer-ingestion-final-decoder-equivalence.json in workspace review-results.

Genuine source-review snapshot, checked against exact real packets/Unicode quotes:

- Reviewer A: 173 fully inspected packets, 1 partial packet; 200 saved version assessments (3 partial, 0 metadata-only).
- Reviewer B: 648 fully inspected packets, 2 partial packets; 651 saved version assessments (3 partial, 8 metadata-only).

These are manually judged, model-assisted real source inspections, not unit fixtures,
human gold labels or complete-frame qualification. Each reviewer still needs all
1,750 packets / 1,977 versions, including 15 metadata-only. No uninspected remainder was
auto-filled. Partial source inspections preserve actual read intervals/all-null
document judgments; do not confuse valid saved records with full text inspection.
Final schema/quote receipt: workspace
review-results/issuer-source-genuine-partial-review-check.json (per-file/source hashes).

Actual progress is saved at absolute workspace directories:
C:/Users/manis/Documents/Codex/2026-09-28/c/work/source-review-fresh-a
C:/Users/manis/Documents/Codex/2026-09-28/c/work/source-review-fresh-b
Use their per-packet strict JSON, sorted assessments.partial.jsonl and progress
receipts. Resume each independently from its OWN completed IDs/partial read ranges.
Do not expose candidate DB/extractor/old labels/outcomes/other review judgments to
either reviewer. Common fiscal-period source criterion requires source year or an
explicit end date; never infer fiscal year from publication. Preserve original text.

After full genuine coverage: assemble each sorted complete JSONL, adopt immutable
actual review files under research data, freeze their physical SourcePins and strict
AnnotationIngestionConfig. Run installed publish_issuer_annotation_correspondence
to a fresh research directory, then independently pin its manifest and run
verify_issuer_annotation_correspondence. Exactly one heavy worker; freeze code,
config, source and output pins across the chain. Current completed packet pin is
data/research/swing_initial_fit_issuer_review_packets_retry/_manifest.json SHA256
2b4b933565704a06fae86e815f3edab56412c2d2717254f4eb39e5ad5f50b3a9.
No packet republication, re-extraction, old-review adaptation or gate waiver.

Later design preparation (not implementation/admission): workspace
work/revised-issuer-packet-design/DESIGN_MULTIVERSION_METRICS.md has one design review
with no supported issues. Reuse existing numerical primitives/POLICY/sampler,
preserve gates/F-only weighting and D worst-case recall bound, retain honest
multi-version truth instead of fabricated aggregate text SHA. Implement only after
the current actual ingestion exit. Event projection, 126-feature publication/replay,
two issuer fits, frozen funded 12-policy comparison and signed-return V1 serving follow.
Four completed fits, targets, candle/news alignment, TradingFlow and main stay fixed.

Both source reviewers were interrupted after durable checkpoints. No heavy worker or
lease is active. Resume A at sorted packet 173 using its saved read intervals; resume
B from its OWN partial/remaining IDs and progress.json. No judgments were changed
during the final sorted-receipt refresh.

Account usage observed 99% consumed; no reset credit consumed.
This is the user-requested continuation checkpoint, not completion of the Astra plan.
Earlier current/running/adoption claims below are historical and superseded here.

### Historical continuation: ingestion software pushed; real source reviews underway

Implementation 5cf8208 is pushed. New canonical research owner
research/issuer_annotation_ingestion.py reads two physically pinned complete JSONL
review files, calls the existing public packet verifier, then acquires a separate
processing lease and checks all source/DB/packet/config/code hashes before and after.
It streams complete assessments and exact-version candidate correspondence to a
private stage, preserves every unresolved judgment, and grants no qualification,
training, serving, economic or promotion permission. Independent verification
repeats the actual calculations and compares all output bytes. No old owner changed.

Selected synthetic UNIT checks: 336 passed in 21.46s. One consolidated code review
found pre-lease bulk output hashing; moved it under the processing lease and bounded
the preliminary routing metadata. Final affected 21 units passed in 5.20s, including
busy-lock rejection without assessment/correspondence reads. Ruff and strict mypy
passed. Real retained candidate parser check separately passed for all 14,182
candidate versions/16,004 candidates (exit 0), with manifest/DB hashes unchanged.
The later lock fix leaves that exact decoder and every dependency unchanged; proved
in workspace review-results/issuer-ingestion-final-decoder-equivalence.json. All
protected original 133 and packet 24 code pins still match. No full-suite release.

Actual complete review ingestion/publication/replay remains OPEN: each reviewer
must inspect 1,750 actual packets/1,977 versions, including 15 metadata-only. Fresh
independent model-assisted source reviewers A and B are inspecting only blinded
source text and identity evidence, with no extractor/old-label/outcome access.
Their genuine partial files are workspace work/source-review-fresh-a and -b;
do not treat them as complete review artifacts or use them for qualification.
Installed schema/quote checks passed on the recorded snapshot A=30 packets/31
versions and B=40 packets/40 versions. Report:
review-results/issuer-source-genuine-partial-review-check.json. Source judgments
are made after actual text inspection; code only checks quotes and serializes them.
Never generate rows for uninspected remainder or infer missing negatives/years.
Common source-only fiscal-period criterion requires source year (or explicit end
date); Q3 alone cannot identify the complete fiscal period. Preserve such quoted
partial evidence with explicit_fiscal_period false, never publication-derived year.
Model assistance is declared honestly; these are not human gold judgments.

Next finish both real independent reviews, validate exact complete coverage/pins,
then run the installed actual publisher and independent verifier with fresh output
and all source/config/code pins frozen. Actual receipt is required before later
metrics integration/event projection, reaction features, two remaining fits and
funded SPY comparisons. Public API V1, targets and four completed fits stay fixed;
TradingFlow/main remain untouched. User requested current handoff at 99% account
usage; latest observed usage is 92%, no reset consumed.

Historical details below are superseded where they describe uninstalled ingestion
code, absent partial reviews or still-running packet/candidate checks.

### Historical bounded slice: pinned complete review ingestion

Implement only research/issuer_annotation_ingestion.py and focused unit tests.
The reviewed design is workspace work/revised-issuer-packet-design/DESIGN_INGESTION_SLICE.md.
Inputs: exact completed packet manifest plus two new physical JSONL reviewer pins;
each file has existing strict PacketAssessment records in sample-ID order with
stable distinct reviewer declarations and complete sample/version coverage.
Canonical public packet verification finishes before the ingestion processing
lease. Recheck every returned source, physical DB and packet output pin at entry
and exit; use only the existing canonical derivative connection for candidate reads.
Outputs: deterministic full assessments and exact candidate correspondence, all
qualification/training/serving/promotion/economic flags false. No label generation,
metrics, event projection, model fits, serving, targets, gates or protected-owner edits.
Exit: focused pin/order/coverage/strict-type/mutation/private-publication units,
Ruff/strict types and one consolidated code review, then actual publication/replay
with TWO new complete real independent reviewer files. Those files do not yet exist;
unit fixtures cannot close the actual-data requirement. Retain any failed private
stage; reject changed/missing/ambiguous evidence without publishing an authority.

### Historical continuation: complete-version annotation component closed

Implementation 75791f4 is pushed. The installed pure annotation reader and exact
event matcher enforce complete physical-version coverage, exact Unicode quotes,
four-role matching and explicit unresolved/metadata-only judgments. They cannot
authorize source qualification, model training, serving or promotion.

Verification: 314 selected synthetic UNIT checks passed in 43.39s; Ruff over four
files and strict mypy over two source modules passed. One consolidated review found
no supported issue. The actual installed reader separately checked all 1,750
published packets: 1,977 versions/occurrences, 15 metadata-only; no annotations were
created. Publication SHA2b4b933565704a06fae86e815f3edab56412c2d2717254f4eb39e5ad5f50b3a9
and all protected original-input 133 and packet 24 implementation hashes matched.
Actual report: workspace review-results/issuer-annotation-actual-published-packet-shape.json;
packet inventory SHA bb995b1cf782e4cf8b83da321146e9168f1587ea24fb92b1de15381c82b7659f.
No full-suite release, new news-quality result, model fit or SPY improvement is claimed.

Next bounded slice: operational pinned two-review ingestion/correspondence. Accept
physical publication/reviewer SourcePins, call the canonical packet verifier, then
read complete independent reviewer artifacts under a processing lease with source
checks before and after. Preserve exact candidate/version evidence and all false
admission flags. Metrics, event projection and training remain later checkpoints.
Two NEW independent source-only review artifacts do not yet exist; software tests
and packet publication cannot supply their judgments. Targets, four trained fits,
TradingFlow, main and public API V1 stay unchanged.

Historical details below are retained evidence; this current closure supersedes
their annotation-adoption and packet-running narration.

### Historical continuation: real packet publication and independent replay complete

Packet component at d228191 is CLOSED. Actual chain 59799/PID 75040 completed
13:44:20 UTC October 6 with exit 0. It independently reproduced all 1,750 packets,
complete source inventory, exclusions, frame, sample and manifest bytes/hashes.
All 24 executed implementation pins remained unchanged throughout both operations.
Worker is gone; no new labels, source quality, model training or serving is claimed.
Durable receipt actual-issuer-review-packets-retry-state.json SHA256
9a3348774be1835abb61b2444d3b6f099cd4bc6279d807de198cd5162b889995.
All source/qualification/training/serving/promotion flags remain false.

Next bounded slice: adopt the reviewed off-repository pure annotation DTO/validator
and same-version/four-role correspondence kernel, replacing only draft test loaders
with normal package imports. No original producer, sampler, 24 packet owners, 133
original input owners, targets, trained fits, metrics or event projection changes.
Exit: focused source/version/quote/duplicate/partial-inspection regression units,
Ruff/strict types and actual complete published-packet shape checks; operational
pinned two-review ingestion and real independent source judgments remain required
before qualification. No aggregate text-hash workaround or API version increment.

Historical run details below describe preparation and intermediate states; the
completed actual receipt above supersedes their running/pending narration.

Fresh actual publication/verification chain started 11:17:05 UTC October 6,
session 59799, worker PID 75040 (venv launcher 6832), implementation d228191.
Frozen 24-file inventory actual-issuer-review-packets-retry-frozen-implementation.json
SHA256 4b0a6a94a5fa4997854782df54c4d26ca0ce3aeb12bd4660954cc727ca8c4768.
Both operations use actual retained sources and installed owners; no mocked data,
admissions or model outputs. Freeze all executed code/config/source pins until the
whole chain ends. Durable phase is publishing_actual_source_review_packets; no
source quality or completed independent verification is claimed.

Actual publication completed 12:24:53 UTC. Manifest
data/research/swing_initial_fit_issuer_review_packets_retry/_manifest.json SHA256
2b4b933565704a06fae86e815f3edab56412c2d2717254f4eb39e5ad5f50b3a9.
Durable phase is now independently_verifying_actual_source_review_packets;
session 59799/PID 75040 still owns the run and all 24 implementation pins stay frozen.
Published inventory: 516,679 versions, 539,399 records (22,720 without a version),
22,722 aliases. U=466,635 clusters; D=1,748 all inside U; F=464,887. New sample:
1,750 entries/1,745 distinct clusters, 300 earnings and 250 guidance candidates,
600 noncandidates each, zero shortages. Packets contain 1,977 version/occurrence
entries, 15 metadata-only. All qualification/training/serving/promotion flags false.
Frame SHAa05bccd0b82a4b4b5331695cec758201446017dc89eec7717864a6b9baaa4e2b;
exclusions SHA91b104ab913d7466c52d48ec8da452761bf9ac5baa954ac138b83ca77b9aad09;
sample SHA6da314ea6de8878fbdbf0aa6986cc92589faf4ce840fa39e58ea97471cf87fca.
Packet slice remains open until actual independent verification exits successfully.

Prepared reader real-data diagnostic also passed over all 884 complete preserved
staged packets: 991 versions/occurrences, seven metadata-only, exact text/inventory
hashes; no annotations created. Report issuer-annotation-actual-staged-packet-shape.json
under workspace review-results. This is a draft shape check, not source admission.

Independent preparation only while that run is frozen: reviewed annotation design
work/revised-issuer-packet-design/DESIGN_ANNOTATION_SLICE.md; off-repository draft
work/annotation-draft/research/issuer_source_annotations.py and
issuer_annotation_correspondence.py with two tiny unit modules. Final 67 synthetic
unit cases pass (0.88s), Ruff and strict typing over workspace package copies pass;
one consolidated draft review found no supported issue. Hash inventory:
review-results/issuer-annotation-draft-inventory.json. No draft is installed,
committed, imported by the actual worker or a source authority. It prepares strict
complete-version validation and pure exact-role matching only; operational pinned
ingestion, new independent labels, metrics and event projection are still absent.
Adoption awaits completed actual packet proof and packet-slice documentation closure.

Actual packet chain started at 10:10:38 UTC October 6: session 12774,
PID 4688, publish-issuer-review-packets run fa253e8c773a475a80cbc5ba787e1726.
Workspace work/run_actual_issuer_review_packets.py sequentially invokes installed
actual publisher and independent verifier. Frozen 24-file implementation inventory:
review-results/actual-issuer-review-packets-frozen-implementation.json SHA256
3c28655bc697c09de95d3b9d0feff1d93fbc5e373d083c52d4f1392a5d2eb4b8.
Stopped only this identified worker at 11:11:17 UTC, exit -1, after more than
1.86 TB logical reads. All 24 frozen implementation files remained unchanged.
Actual EXPLAIN disproved repeated candidate scans: its cluster query uses indexes.
The packet occurrence query scans the original records table for each sample;
there is no cluster index. Evidence: review-results/issuer-packet-actual-query-plan.json.
Private stage and state/log/exit prefix actual-issuer-review-packets remain retained;
no public manifest, packet pass, new labels or qualification is claimed.

Bounded amendment: only the new packet owner reads records once after unchanged
sampling, retaining ordered phase/ordinal/version IDs and exact record JSON hashes
for sampled clusters. Reuse references across both family packets; preserve empty
version IDs, every occurrence, memory guards and source checks. Original DB, producer,
sampler, candidate policy, model inputs and numeric gates stay unchanged. Exit test:
interleaved phase/ordinal/query copies/unavailable versions produce identical
occurrence bytes, counts and digests; packet rendering performs no per-cluster
records query. One narrow design/code review, focused units, lint/types, then fresh
actual publication and independent verification close this measured repair.

Repair d228191 is pushed. Focused 29 units/continuity pass (7.49s); final affected
19 units pass (6.44s), Ruff and strict source typing pass. Narrow review identified
only the sampler tuple/list annotation; repaired with Sequence. All 133 protected
original input implementations are unchanged; only the new packet owner changed
within the 24-file closure. Actual read-only occurrence equivalence passed against
all 884 complete preserved packets; one interrupted JSON remains incomplete.
Original database SHA858c929a1f683745999d9fad085676c4dd6ebfae833b7bc7e5ceea3c8ba56386
matched its authority before and after. Report:
review-results/issuer-packet-actual-occurrence-equivalence.json; this is occurrence
equivalence evidence only, not complete packet publication or source qualification.
Retry work/run_actual_issuer_review_packets_retry.py uses fresh
data/research/swing_initial_fit_issuer_review_packets_retry and durable
actual-issuer-review-packets-retry state/log/exit/inventory files; no old stage reuse.

Implementation a543fe7 is reviewed and pushed: strict packet config/version contract,
frozen frame/document/correspondence identities, complete inventory/exclusion/sample/
blind packet publisher and deterministic verifier, and two thin research commands.
Original sample/review headers and coverage are verified without reusing old judgments.
Full records retain (phase, ordinal) ownership and original JSON hashes; versions,
aliases, query copies, all documents/exhibits and explicit unavailable metadata remain
accounted for. D is removed only from new review eligibility, never from raw sources.
Blind output omits extraction status/spans/rules and old labels. Final source-context
checks and output checks complete before atomic directory publication. Private failed
stages are retained, with no automatic overwrite or resume. All admission flags false.

Focused units/direct CLI/package/schema scope: 338 passed in 35.11s. Ruff found only
import formatting and one long line; strict typing found one missing empty-set dict
annotation. Those were repaired without changing behavior. Final affected packet/
command/inventory scope: 27 passed in 15.20s. Ruff five Python files and strict typing
three source modules pass; one consolidated code/ML review reported no supported
finding. All mocks/fixtures are unit-only. No real packet publication, new labels,
qualification, model fit, serving, SPY result or full-suite release claim is made.
Unit log/XML prefixes: review-results/issuer-packet-focused-units and issuer-packet-final-units.
Protected original-input 133 files and candidate 12 files remain unchanged.

Actual config configs/swing_initial_fit_issuer_review_packets.json binds five real
source pins (population, candidate derivative, sample and both old reviews), each
independently checked. Config byte SHA256:
7edd26c71ad88038ebf0e450f42967aaba13e5791037f160c29b6b1762d103fd.
Next call installed publish_issuer_review_packets to fresh
data/research/swing_initial_fit_issuer_review_packets under one workspace lease,
then independently pin the completed manifest and call verify_issuer_review_packets.
Only those actual outcomes close the packet slice. Freeze its executed code/config/
source files during the sequential run; no fake approvals or extra heavy worker.

Frozen source packet scope:

Frozen first slice of news correspondence (October 6): new strict research packet
contract and publisher/verifier over the already verified candidate sidecar, plus
thin CLI adapters and focused units. Do not edit the original population producer,
candidate core/derivative, numerical sampler or any of the protected 133 original
input files/12 derivative files. No provider collection, outcome/model access,
annotation ingestion, qualification/event projection, model fit or serving change
belongs to this slice. APIs remain V1; no compatibility aliases.

Measured original review metadata: exact original sample hash
f19f46917b1d629bfd55515556b5732b7b4f45cfb7669b1e0901f1a0b59d25c1 has
1,750 entries but 1,748 distinct announcement clusters. Both immutable reviewer
files cover those same 1,750 sample IDs. New exclusion D is the deduplicated union
of all those clusters (both families/roles) and all cluster identities resolving
the five exact development source IDs listed in PACKET_SLICE.md, including copies,
revisions and SEC exhibits. An absent named source rejects publication. Retain D
outside U separately. F=U minus D; every excluded original record/version remains
accounted for and cannot project a newly qualified event. Preserve original text,
database, aliases, clocks and old labels; do not use old judgments as fresh truth.

Use select_content_review_sample(F) with unchanged policy/hash/seed 42, 300 earnings
and 250 guidance candidate draws plus 600 noncandidate draws each. Freeze separate
frame/document/correspondence policy identities before labels. Report full, excluded
and fresh counts and actual F-only N/n weights; shortages stay explicit. No complete-
population quality claim follows from fresh-only samples or packet publication.

Output: hash-bound request, exclusions, complete U frame, F sample, full-version
blind packets and manifest. Bind all original record/version inventory counts and
ordered digests without copying the 7.4 GB database. Every sampled version/exhibit/
revision appears, including future/unknown/unreadable metadata-only entries. Text
display preserves the exact normalized Unicode and separate original clocks.
Candidate IDs/roles/spans/rules/dispositions, old labels and outcomes are absent
from blind packets. Annotation offsets are half-open Unicode code points.

Future correspondence is frozen now: same version/text/security/family; exact
issuer/action/result/fiscal-period offsets and quotes; reviewer anchor within the
candidate statement; exactly one structural match before considering verdicts.
Any unresolved/unsupported candidate fails cluster success. No true exhibit or
unrelated same-document event rescues another candidate. Full document assessments
are required; an omitted/unreadable version cannot become a resolved negative.
This slice records that policy but does not create labels or qualification metrics.

Publisher/verifier each owns one heavy lease and actual canonical derivative
reproduction. Stream into a fresh private stage; no automatic overwrite, deletion
or resume. Final connection/source/code/config/review and artifact checks complete
before atomic directory publication. Independent verifier rebuilds exact exclusions,
frame/sample/packets and hashes, not a claimed JSON pass. All admission flags false.
Exit: focused contract/poison tests, Ruff and strict types, one consolidated review,
then actual retained-data publication and independently pinned full verification.
Review metadata evidence: workspace review-results/revised-packet-original-review-metadata.json.
Frozen designs: PACKET_SLICE.md, CORRESPONDENCE_RULE.md and EXCLUDED_FRAME_AMENDMENT.md
under workspace work/revised-issuer-packet-design. Original input component is closed.

Completed original input reader evidence:

Actual reader check 39766/PID 17544 completed exit 0 at 09:22:38 UTC October 6.
Its installed owner independently reproduced the original receipt, read all 59
months and verified every one of the 586,305 decisions under original ownership,
then read all 551 original stock groups and 1,510 SPY history rows. One stock group
has empty usable history: the preserved original quarantine, not invented candles.
Completed actual report data/reports/swing_original_reaction_input_verification.json:
SHA256 347199797d1521967194c85885632040a7065e713b7e9b23efdec6612123af21,
status passed_actual_retained_input_interface. All 133 executed hashes are unchanged;
its worker and lease are gone. No mock readers, fabricated qualifier or model outputs
were used. Source qualification, reaction publication, training/serving/promotion
flags remain false. This closes the original-input consumer component at 852f43f.
The newer-price comparison stays failed, and main/TradingFlow remain untouched.

Next demonstrated source conflict: existing blind packets contain SEC exhibits, but
old reviews select one document while metrics/projection can admit another candidate
version without exact same-document support. A true exhibit must not rescue a false
candidate elsewhere. Preserve original text/database/reviews and the passed revised
candidate sidecar. Freeze complete document/version annotation correspondence and
the development-inspected exclusion/sampling frame before new labels. Keep numeric
precision, issuer, agreement/kappa and rule gates unchanged. Inspected clusters must
remain explicitly nonqualified, with no projected event; their known history and
unknown coverage remain preserved. Fresh-frame estimates cannot be presented as
full-population estimates. Full-population recall remains unmeasured, with a disclosed
conservative bound that treats excluded clusters as possible missed events.
Reviewed design preparation: work/revised-issuer-packet-design/CHECKPOINT.md and
EXCLUDED_FRAME_AMENDMENT.md. No new packet, review authority or admission is claimed.
The completed actual reader proof is protected evidence; do not change its 133-file
closure to implement the new outer source-review orchestration. No provider rescan,
target/model access, refit of existing models, extra trial or API version change.

Historical running checkpoint (completed above):

Actual retained-data reader check started at 08:25:36 UTC October 6:
session 39766/PID 17544, verify-original-reaction-input-interface,
run f938dfa19cba491ab868f767629ad97a. It is reproducing the actual original receipt
before reading every original month and stock/SPY group. Frozen 133-file inventory:
review-results/actual-original-input-interface-frozen-implementation.json SHA256
0ca65f07e96d09987f17c06265c3d63ecbd57cfde2aa65a52204c035be411573.
Freeze these code files and all input/config bytes until this process exits. State/log/
exit prefix: review-results/actual-original-input-interface. No other heavy job runs.
Actual reader pass, qualified news, 126 publication and new model fits are not claimed.

Implementation 852f43f is reviewed, unit-verified and pushed on
codex/v1-canonical-cleanup. One original_relationship_inputs owner independently
reproduces the original receipt, preserves company/group/month ownership and the
four original quarantines, reads original stock/SPY inputs, and rechecks sources on
exit. Reaction publication and row verification use that same scoped owner, with
one full original replay per operation. The required original_snapshot_replay pin
is explicit; there is no current-price fallback or public skip flag. Original replay
and new reader code are bound in reaction and training implementation inventories.
All 132 files bound by the passed original receipt remain byte-identical.
Numeric kernels, feature order, targets, folds, four fits, main and TradingFlow are
unchanged. Communicating APIs remain V1. Actual source qualification is still pending.

One consolidated static review found two supported issues: outputs were written
before context-exit source checks, and a foreign-source unit patched an unused import.
Both are fixed and the same reviewer confirmed closure. Publisher closes its input
context before completed manifest creation; verifier exits before writing its final
receipt. Both retain the workspace lease and recheck sources before publication.
The new mutation-at-exit UNIT cases verify that failed checks leave no final output.

Verification is component-focused, not a release/full suite. Initial fast scope:
322 passed and two new fixture writers failed on the immutable writer before their
intended poison checks; both fixture writers were corrected. A missing schema-test
filename caused an earlier zero-collection invocation; corrected scope uses
test_schema_naming.py. Superseded native consumer run 41418/PID 92256 was stopped
after final source fixes and redundant unit helper checks superseded its loaded
snapshot; exit -1 at 06:13:39 UTC, no complete pass claimed. Final 72-case native
scope 8736/PID 85264 ended with 70 passed and two UNIT setup failures in 7661.04s:
assigning a frozen parent and holding the busy-test lease in a different runtime
directory. Corrected self-contained boundary/busy cases both passed in 2.32s;
receipt exit-mutation case passed separately in 4.47s. Last source-chain and 126
readiness/loader case passed, preserving inherited rows/settings/fitting weights.
Synthetic inputs/admissions remain tests-only; these are not retained-market proofs.
Final Ruff passed all 11 changed Python files and strict typing all five source files.
No full-suite repeat, actual 126 publication, news fit or funded performance claim.
Unit log prefixes: original-input-fast-units*, original-input-consumer-units*,
original-input-final-units*, original-input-boundaries-final*,
original-input-exit-boundary-fixed*, original-input-receipt-exit-final*.

Next actual operation: workspace work/verify_actual_original_input_interface.py.
It uses three real original candle pins, the installed verified owner and one heavy
lease, independently reproduces the original receipt, reads every original month and
stock/SPY group, and writes evidence only after final context/source checks. No fake
qualification pin, mocked reader, invented candle or model output is supplied.
Fresh report: data/reports/swing_original_reaction_input_verification.json.
Workspace state/log/exit/inventory prefix: review-results/actual-original-input-interface.
An actual input-owner pass is not claimed yet. Qualified real news remains necessary
for the later real reaction publication/pilot and its independent row replay.

Previously completed original source replay:

Actual original-candle chain 35051/PID 72924 completed exit 0 at 01:51:57 UTC
October 6. First full replay passed at 01:32:40 UTC with zero differences, followed
by independent full receipt reproduction. All 586,305 decisions across 59 months
retain their inherited predictors and exact four relationship additions under
current code from the original stock/SPY snapshot. Receipt:
data/reports/swing_original_relationship_replay.json, SHA256
aa8000577dd9628d5edfc5557758238394ebf36aaa8a26f8fae0177fc7d9cec9.
All 132 executed implementation hashes still match the frozen inventory. Its worker
and lease are gone. Durable log/state/exit: review-results/actual-original-candle-checks.*.
No mock reader, invented candle, model refit or overwritten old artifact was used.
The failed newer-price comparison remains failed; original replay is a distinct,
explicit research-input proof. Training, serving and promotion flags remain false.
Real news candidate extraction and independent 516,679-version replay also passed;
document-level source qualification remains pending, not implied by extraction replay.

Next bounded implementation: one original research-input owner, used by reaction
publication and independent verification. Require the exact original receipt and
120/124 ownership; reproduce that receipt under the caller's sole lease; use original
monthly rows and stock/SPY reads with unchanged quarantines and corrections. Bind
current implementation and recheck all input hashes through operation exit. No
modern-price fallback, source waiver, copied feature store or archived code execution.
Public operations own their lease; internal reuse is owner-created, never a public
skip-verification flag. Both publisher and verifier share the same context, avoiding
duplicate full replay in one operation. Current-source equality gates stay strict.
Direct scope: new research/original_relationship_inputs.py, reaction policy,
publisher/verifier, training implementation inventory and affected unit tests.
Numerical kernels, feature order, targets, four fits and TradingFlow stay unchanged.
Exit checks: wrong receipt/parents/ownership/prices/clocks/abstention and mutation
reject; exact monthly inheritance and one-context use; focused units/Ruff/strict
types plus one consolidated review. Actual original-input owner checks use retained
data; actual reaction publication/replay additionally requires real qualified news,
which remains a separate pending dependency. No fixtures substitute for that run.
Reviewed scratch design: work/original-relationship-replay/INPUT_INTERFACE_DESIGN.md.

Historical running checkpoint (superseded by the actual completion above):

Actual chain 60410 ended exit 1 at 01:06:53 UTC October 6 after its news
verification passed. Original candle replay failed with TypeError: datetime64 type
does not support operation all. The WTW quarantine correctly supplies an empty
typed candle table; pandas map retained its datetime dtype and the timezone check
attempted a boolean reduction on that empty datetime array. No candle receipt was
published, no market rows were filled and no fit ran. PID 93872 and its lease ended.
All 134 frozen implementation files were hash-checked unchanged before repair.
The measured one-line repair cc64716 is reviewed and pushed: Python all over each
timestamp preserves nonempty timezone validation and accepts empty stock history;
empty SPY still rejects. Added two UNIT cases for typed-empty/empty-SPY and naive
timestamps. Corrected unit fixture reproduced the actual TypeError before repair
(its first attempt lacked schema_version and failed earlier); after repair all 31
focused units passed in 4.20s, two-file Ruff and strict typing of the source owner
passed. One read-only targeted code review found no further issue. An accidental
.venvenv executable typo launched no test process; no full suite was run.
Logs: review-results/original-empty-clock-units.log/.xml and original actual
chain actual-saved-input-checks.log/.exit.json. Unit passes do not prove market replay.
Actual retry started at 01:12:52 UTC October 6: session 35051, PID 72924,
original-relationship-replay run 30651db259954735986fa073e7ff1032. Workspace
work/run_actual_original_candle_checks.py replays original candles and independently
reproduces a receipt only if it passes. It does not repeat the completed real news
verification; the fix touches none of that job's twelve executed source files.
Freeze all 132 executed source files, configs and inputs throughout the actual retry.
Frozen inventory: review-results/actual-original-candle-checks-frozen-implementation.json,
SHA256 10cffb38bd603be4aae8591c6b179ffaefac3a769afd988476e029e86059de00.
State/log/exit prefix: review-results/actual-original-candle-checks.
No actual rerun pass is claimed; source/model bytes remain unchanged except the
documented one-line repair. No other heavy process or TradingFlow change.

Latest original-input replay implementation c81c70a and source-semantic implementation
77ff5f5 are pushed on
codex/v1-canonical-cleanup. Astra step 3 remains in progress; four
existing fits are retained. Main and TradingFlow are untouched. APIs remain V1,
internal identities unversioned. No compatibility fallback or accepted model claim.

Actual reuse retry 69678/PID 69440, run 2c42ca90bb5542ac8caa5832c136795b,
started at 23:01:29 UTC October 5 and ended exit 2 at 23:08:31 UTC. It reached
the preserved-abstention lookup and reported "preserved abstention lacks one exact
original observation authority". No report exists and its lease is released.
All four original facts pin report 70c6f850...78060, whose bytes are unchanged and
whose original schema ends in .v1. The new historical reuse reader incorrectly
expected the current unversioned schema. Repair 46a7669 is reviewed and pushed:
one exact-hash/schema/bounds historical helper is shared by selector and replay.
Canonical readers, report bytes, facts, boundaries and market values are unchanged.
294 selected unit/package/naming/command checks passed in 25.46s; Ruff and strict
typing over both changed source modules passed. One bounded review found no further
issues. A real read-only, leased metadata check verified all four original references
and unchanged report bytes; it does not prove source/feature value equivalence.
The complete retained-data comparison ended exit 2 at 23:56:34 UTC October 5.
Actual report data/reports/swing_return_relationships_current_reuse_verification.json
SHA256 5f480f56179e054ad4f621f780d72e49e50ac2482af74536c9a3b4aad5e83265
has status failed_differences. All 586,305 rows and inherited columns are unchanged.
Of 549 completed group comparisons, 359 differ in stock inputs, all 549 differ in
SPY open/close values, and 547 differ in relationship additions. For example,
old SPY open 259.51 is current retained-source 258.86; TRV 103.43 is 103.08.
These are actual price/feature differences, not naming differences or fixture output.
Two further groups (INFO and SBNY) fail complete physical-prefix verification.
Original sources/features/models remain unchanged. This report cannot authorize
current 124 publication or claim that existing fits used the current candle inputs.
A bounded source-choice design review is completed as recorded below; no equality
tolerance, source waiver, retraining or original-data rewrite follows from this failed comparison.
Its process and lease have ended. Durable stdout/exit prefix:
review-results/relationship-reuse-real-observation-fixed.

Actual news candidate publication completed exit 0 at 00:46:30 UTC October 6
(session 64326/PID 92408, run 86425b3802554752900eec1d78d24221). Output:
data/research/swing_initial_fit_issuer_candidates_revised/_manifest.json,
SHA256 9c5f55807e9614c2e64d85902130e72c1dafdce73ba5f101f783daa9ac2f82d0.
The candidate-only SQLite sidecar is 368,517,120 bytes. Actual full source population:
516,679 versions; 14,182 review-candidate versions; 470,541 unclassified;
31,956 preserved unavailable; candidate output changed for 14,268 versions.
These are extraction counts, not candidate precision, recall or model improvement.
Publisher rechecked every original input/current code hash and replayed every output
row before atomic publication. Original 7.4 GB population, reviews and raw sources
are unchanged. All qualification/training/serving/promotion flags remain false.
Durable actual log/exit: review-results/issuer-candidate-real-derivative.*.

Historical attempt: sequential actual-data chain 60410 started at 00:47:47 UTC
and ended at 01:06:53 UTC October 6 as described above.
PID 93872 (launcher 93248). News verification lease
verify-saved-issuer-candidates, run 84c2d684857c4e9ca7a194d2f6cf0621, completed
at 01:04:16.738752 UTC. Its subsequent original replay failed. Workspace wrapper
work/run_actual_saved_input_checks.py invokes only installed actual data owners:
(1) independently verify the published news derivative using its actual manifest
SHA, (2) replay original candle relationships to a fresh receipt, (3) independently
reproduce that receipt only if it passes. No mocks, fixtures or model builds.
Current phase/status: review-results/actual-saved-input-checks-state.json.
Durable stdout/exit: review-results/actual-saved-input-checks.log/.exit.json.
Combined executed source inventory has 134 unchanged files, frozen in workspace
review-results/actual-saved-input-checks-frozen-implementation.json SHA256
f731e868299df147d71e7adbd201f1d83bc1f39d173ac65d4a3d06a5456f97cf.
Those sources, inputs and configs were frozen until the chain exited; it imported
both data owners at start. Only one heavy job runs at any time. Independent
post-publication news verification passed at 01:04:16.738752 UTC October 6: all
516,679 versions and published counts were reproduced from retained original inputs.
This establishes extraction replay only; source judgments, training eligibility and
model improvement are not established. Original candle replay started immediately
at 01:04:16.739147 UTC. No original-candle receipt pass is claimed yet.

Bounded checkpoint amendment after measured price differences (October 6):
The user's explicit priority to reuse original matched data supports independently
verifying the original candle snapshot for the frozen experiment. Plan and design
review agree on one research orchestration owner and a narrowly typed receipt over
existing files, without new row copies. First slice: replay the four additions under
current numerical code from exactly the original stock/SPY snapshot, original group
ownership/corrections and WTW/ATVI/INFO/SBNY abstentions. Bind original 120/124
manifests, requests, receipts, physical OHLCV/interval/availability metadata, source
hashes, the failed comparison, current executed code and separate old provenance.
Require all 59 months and 586,305 decisions, exact additions/clocks/null reasons and
all inherited values/targets/eligibility/order. Recheck pins before publishing;
any discrepancy prevents the new receipt. Mutation, substituted SPY, ambiguous
ownership, current-price injection and changed prefix/clock must be rejected.
No new source collection, old-code execution, tolerance, source waiver, retraining,
model specification or production/live approval follows. _verified_authority's
current-source equality gate remains unchanged. Later reaction publisher/verifier
must use one verified research-input interface for original parent, ownership and
both stock/SPY reads; their current adjusted-binding calls would mix price snapshots.
The first slice does not alter those consumers before actual original-input replay.
The reviewed three-file implementation was outside the earlier news publication
job's 12-file closure. It is included in the current chain's frozen 134-file inventory.
This receipt will provide an explicitly
original-input research route, not a passed comparison against newer candles.


Original-input replay implementation c81c70a is reviewed, tested and pushed.
The three changed files were outside the then-running news job's 12 pinned dependencies;
all 12 hashes were checked unchanged before application and after checks. Original
source/data/config/model files remain unchanged. This standalone research owner
adds no current-source admission or consumer fallback. Its receipt verifier reruns
the calculation and acquires the lease before reading inputs. First new unit scope
28 passed (5.06s); affected historical/reuse/kernel/command/schema/package/continuity
scope 419 passed (55.36s). A missing test filename caused one earlier invocation to
collect zero cases; the corrected scope is the 419-pass run. Ruff found import
formatting and a loop-lambda warning, both repaired. After the final lease ordering
fix, all 29 new unit cases passed (4.39s), Ruff passed three files and strict typing
passed both source owners. One bounded static code review found no further issue;
the first actual original-input replay failed as recorded above; no passing receipt exists yet.
Unit logs: original-replay-units-first.log, original-replay-consumers.log,
original-replay-consumers-final.log/.xml/.exit.json and original-replay-final-units.log/.xml.
Actual original replay began in session 60410 at 01:04:16 UTC after independent
news verification passed and released its lease. Use replay_original_relationships from the installed
research.original_relationship_replay owner with the unchanged reuse config
SHA95dd86ec733a70a72c0d0a2f05de78e0111b71e31fef1786d424136141cc209c,
failed_comparison SourcePin(data/reports/swing_return_relationships_current_reuse_verification.json,
5f480f56179e054ad4f621f780d72e49e50ac2482af74536c9a3b4aad5e83265),
and a fresh data/reports/swing_original_relationship_replay.json destination.
Inspect its actual result and SHA; only passed_original_snapshot_replay may be
submitted to verify_original_relationship_receipt. Never invent a receipt hash.

Unit receipts: historical-observation-units.*; actual metadata stdout:
historical-observation-actual-metadata.log. No fixture result is market evidence.

Independent source-semantic repair 77ff5f5 is implemented within step 3: shared candidate
classification rejects tender/consent results as earnings, recognizes explicitly
attached financial-guidance years, and leaves equal/contradictory guidance ranges
unresolved. A candidate-only derivative replays existing saved text with exact
original parent/provenance/identity/clocks and a read-only joined view; no provider
rescan or text/archive copy. Original reviews stay immutable. New blind packet
review and qualification remain required before reaction features or fits.
Named inspected examples are development regressions only. Exit gates: affected
unit/tamper/parity tests, Ruff/strict typing, one bounded consolidated review, and
actual derivative/requalification with unchanged statistical thresholds. All
training/promotion/serving flags remain false until their required evidence passes.
The completed comparison's 131 source dependencies excluded content_review.py and
this new helper. Its exact inventory is saved in workspace review-results/
relationship-reuse-frozen-implementation.json and must remain byte-identical.
The CLI exposes derive-saved-issuer-candidates and verify-saved-issuer-candidates.
First 121 unit cases passed in 9.64s. Final direct-consumer scope passed 242 cases
but failed its command inventory assertion; the inventory was updated for the two
new names. A first inventory rerun passed 23 cases but still failed alphabetical
order; that was fixed, and the exact inventory case passed in 2.92s. No calculation
or source gate was changed to address the fixture failure. Ruff passed six changed
Python files; strict typing passed all three source owners. Initial strict checks
had 19 narrowing errors, repaired without removing runtime checks. One bounded
consolidated review found no further issues. Logs: issuer-semantic-initial.*,
issuer-semantic-final.*, issuer-cli-inventory-final.*, issuer-cli-inventory-corrected.*.
Actual original-population derivative 64326 completed as recorded above. Its
independent verification passed in chain 60410; fresh reviews and qualification
consumer integration remain pending. No training admission.
All 131 comparison source hashes were rechecked unchanged during that completed run.
Only these independent files/tests may change during the job; no concurrent heavy
run or modification of its pinned source/config/data is permitted.

Next source-review design preparation is scratch only. Existing packets already
contain readable SEC exhibits, but each review chooses only one document and the
current metric/projection does not tie each candidate to its exact supporting
version/text/span. Revise that demonstrated correspondence conflict; do not recollect
or rebuild matched news. Old judgments stay immutable. Before new annotation, freeze
the previously inspected-cluster exclusion list and the statistical population:
a fresh frame with excluded clusters has zero inclusion probability for those
clusters, so its N/n estimates cannot be claimed to cover the full population.
Disclose full/excluded/fresh counts and define conservative treatment of excluded
candidates before any admission decision. Numerical precision/agreement thresholds
remain unchanged. Proposal: workspace work/revised-issuer-packet-design/CHECKPOINT.md.
No new packets, labels, sampling population or qualifier have been published.

The user clarified October 5: mocks/synthetic inputs are allowed only in unit tests.
AGENTS.md records this. Fixture tests below are unit/regression evidence, not
real-data integration, even where old test names say "real" or "native". Actual
reuse, qualification, feature publication, training and evaluation use retained
provider data and actual implementations without patched readers or calculations.

The no-mocks requirement exposed a non-unit CI path: it trained a four-row invented
model, fabricated promotion metrics/signatures, and mounted the resulting release.
Corrective implementation 892100c is pushed. That operational builder and CI invocation
are removed; promotion fixtures now live only under tests/support/promotion.py.
CI uses the checked-in config with genuinely absent deployment artifacts, measuring
container liveness and expected 503 refusal only. It does not claim real-data prediction.
One bounded review found the moved helper's config path needed parents[2]; repaired.
Five focused unit tests passed in 7.04s (no-synthetic-ci-2056 JUnit/stdout/exit files).
Ruff passed; strict helper typing passed with MYPYPATH=src; workflow YAML parsed.
An initial isolated mypy invocation could not resolve source annotations and failed;
the correctly configured source invocation passed. Docker is unavailable locally;
actual container execution remains unperformed here. No market/model output is mocked.
Only a lightweight five-case unit run and static checks overlapped qualification;
no other heavy data/model/test job ran. Qualification's pinned sources are unchanged.

The original source export completed exit 0 at 15:11 UTC October 5; its source
session 92732/PID 88592 is closed and lease removed. Do not rescan or resume it.
Output data/research/swing_initial_fit_issuer_content_review_population_source_bound.
Manifest SHA256 3c59845761740dc1a07a2f994cceaba53354c6f9486f4baa634ebd422a6a84ce;
request 3ea48fb4d746d6b77be0d65e96d23261cb578e527d0a3dbbd450a13957088e62;
sample f19f46917b1d629bfd55515556b5732b7b4f45cfb7669b1e0901f1a0b59d25c1.
Actual retained population: 469,668 Alpaca occurrences, 47,011 SEC body records,
22,720 SEC index records, 516,679 source versions, 466,635 sampling clusters,
1,750 blind samples; candidate clusters 12,995 earnings and 739 guidance. Readable
issuer-unknown versions 25,742. Counts include duplicate issuer copies. Capture
clocks remain historical proxies; an export alone admits no event/model/serving.

Both independent source-only model-assisted reviewers completed all 1,750 samples.
Exact model variant unavailable; no human review is claimed. Neutral version choice
was frozen before judgments: minimum SHA256 of compact sorted UTF-8 JSON identity
[sample_id,source_family,source_id,source_version_sha256,text_sha256]. Selection
SHA256 ccde69184e492356f76c1e26b52e09b4c5375c7292785f145ea7e57b956f1b25.
Both judged identical versions without candidate roles, returns or other judgments.
Exact versions/text hashes/quote bounds and all unique sample IDs were checked.
Completed bytes copied unchanged into data/research/swing_initial_fit_issuer_source_reviews:
reviewer A 17132bdcc38ca19738fd00b9e8a0f576f55e972fa20b4fd74a792918d9fc8922;
reviewer B cfa9bea062ca443deed419593c1482a9b634bcb6f47797d770bc6219ade8342d;
export manifest dcd6808a998dd0e0509ddfc0d7463ea5e8c4a292b640d0115e59067f27a9856d.
Workspace helper source_review_tools.py only selects/binds/serializes; no inferred
labels, fabricated negatives, hidden classifier judgment or threshold relaxation.

Actual qualification session 27687/PID 48380 was stopped with verified owned identity,
exit -1, before any completed qualification publication. Durable attempt evidence:
review-results/qualification-real-2012.* under the chat workspace. It exposed a
runtime defect in the uncommitted authority: full Python heap collection per source
version across several 516,679-row passes. At 12 minutes it had consumed 555 CPU
seconds with 0.23 GiB resident memory. The local repair checks memory before every
256 physical records/files and at phase/completion boundaries. Every source hash,
row/disposition, clock and reviewer check still runs; closed guard/producer code
and all data remain unchanged. Bounded review accepted the repair; injected pressure
at projection record 256 prevents a completed manifest. No old attempt is resumed.

Actual-data qualification session 64007 ended exit 2 at 21:01 UTC October 5:
system memory use reached 90.4%, with 1.50 GiB available, above the frozen 90% limit.
Its lease is removed. Output data/research/swing_initial_fit_issuer_content_qualified_batched
contains only _request.json and _authority.json; there are no events, dispositions
or completed manifest. Durable qualification-real-batched-2041 stdout/exit evidence
is retained. Do not resume this failed path or claim a qualified publication.
Intermediate real-review metrics: earnings joint positives 225/300, lower bound
0.7067674358995081; guidance 216/250, lower bound 0.8244206756033808. Both families
fail frozen gates. The wrong_issuer counters (13/4) count any reviewer False,
including disagreements; they do not establish every article names a wrong company.
For example, source 0001193125-21-011422/5 names Broadcom but announces debt tender
results. Do not mutate reviews, lower gates or substitute invented news features.

Actual reuse attempt 39985 ended exit 2 with no report; diagnostic 87347 ended exit 1.
The preserved chained traceback (relationship-reuse-real-trace-2110.log) identifies
relationship_historical_evidence.py:77, not an invalid raw-price plan: the new
inspector accessed baseline child['rows'], but the original 120 publication owns
rows/decision_ids_sha256 at month level and child audit.rows. Original 124 child
has direct rows/hash declarations. Both saved publications contain their metadata.

Repair 1d0d3be has one explicit historical monthly-record projection:
baseline month count/hash plus checked audit.rows; relationship mandatory matching
child count/hash. Sidecar and physical identity checks use that same projection.
The original artifact bytes are unchanged; canonical readers gain no fallback.
The old synthetic baseline unit helper was corrected to the actual document shape,
and 18 unit cases assert both shapes, rehashed contradictions, physical rows/IDs and
post-inspection tamper. This does not claim an actual reuse pass.

Repair 1d0d3be replaces the history dictionaries with one
separate temporary SQLite index (8 MiB cache, temp_store=FILE, mmap_size=0), keeping
all causal clock/owner/proof/fallback/exclusion rules and source connections immutable.
It releases completed review-cluster objects before projection in publisher/replay.
A bounded review found SQLite could coerce integer query IDs into text matches;
query matching now requires a nonempty string and a unit regression preserves the
old ambiguity reasons. Final exclusion-list shape/order and every source disposition
remain unchanged and guarded. No source/review/gate/config/data-format changes.
Repairs are now committed and pushed as 1d0d3be. Focused run 60429 completed:
91 cases outside the new ownership file passed (history 9, authority 29, prior reuse
52, positive source-chain 1); that file initially had 6 failures/9 setup errors
because its deliberate synthetic rewrites used the production immutable writer.
Only that unit file was corrected with an explicit existing-fixture rewrite helper.
Its complete rerun passed all 18 cases in 12.93s, with no production writer change.
Durable receipts: real-failure-repairs-2118.* (725.28s, exit 1 retained) and
historical-ownership-units-2122.* (18 passed, exit 0). These are unit-only results,
not actual market-data integration. Ruff passed all affected Python; strict typing
passed four source modules. Changed tracked Python line endings were normalized
with identical parsed syntax trees; pinned config and source-data bytes untouched.
Both bounded reviews are complete; no additional general review is needed.
Actual qualification rerun 4422/PID 87284 started at 21:35:33 UTC October 5,
run 971522432cab4d81b87defdb61ed3b25. It uses the original population/reviewer pins
and fresh output data/research/swing_initial_fit_issuer_content_qualified_disk_history.
It ended exit 2 at 21:54:44 UTC when system memory reached 90.0% used, 1.57 GiB
available. Its lease is released. Last sampled resident process memory was 0.427 GiB;
final peak was not captured, so the disk repair has no completed actual-data proof.
Only _request.json (58ab1946b0d8ed293a2b6fd0c4ae44a6db09070967171dc3b21b52c9cf0da71a)
and _authority.json (0fce0852ded61c8d3eb2bf43f51c33ef346240296592e5929e72fb4a7fcf2eea)
exist; no events/dispositions/manifest or feature admission. Both family metrics
reproduce the earlier failures. Preserve this failed path. Durable stdout/exit:
review-results/qualification-real-disk-history.* in the chat workspace.
Actual 124 reuse session 37426/PID 87492 started at 21:55:37 UTC, run
3f1a5b1f99ca4393a7879b5c04483072, under its sole canonical source lease.
Original config SHA95dd86ec...209c and data/model bytes are unchanged. Output
data/reports/swing_return_relationships_current_reuse_verification.json;
durable review-results/relationship-reuse-real-repaired.*. Code/config remain frozen
for the job. It ended exit 2 at 21:57:45 UTC: system memory 86.8% used, 2.07 GiB
available, failing the canonical raw-source reader's 85%/2 GiB rule. Sampled peak
process resident memory was 0.285 GiB. No report exists; lease released. No threshold
or guard changes, no repeated source export/matching/reviews, no actual reuse pass.

Saved-config proof implementation b589725 is pushed. Explicit evidence config
configs/swing_saved_evaluation_configuration.json SHA256
4ba8517a0f18fa1d360df7e061aff8cf917d92f2f3eff0bddd49d6ec5413b523
binds exactly the two retained runs. Recovered original bytes under
data/research/swing_saved_evaluation_configuration_bytes match their saved request
hashes; recovery audit SHA106be521a5a9a9650d6dc53460bc5033a80d1ed6a766d2a9f014448714762466.
Actual metadata proof passed exit 0 in 2.47s: report
data/reports/swing_saved_evaluation_configuration_verification.json SHA256
95175eaf55352d5e6fce316d107ebb46480bf60ea759631ae203d8efead13d0c.
Both actual original/current config pairs and saved fold calendars match except
three exact naming changes. Parent holdout assignment/prediction payloads/economics
were not replayed by this small proof. Full evaluator retains those checks; all
admissions stay false. Active readers and original model/request hashes unchanged.
Initial proof helper invocation had a mistyped 63-character hash and exited before
reading inputs; retained saved-configuration-actual.*. Corrected actual proof is
saved-configuration-actual-verified.*. No production verifier change was needed.

Focused unit run 66917: 294 passed, two existing package-boundary checks failed,
434.63s. Durable saved-configuration-consumers JUnit/stdout/exit files retained.
New proof/OOF/CLI/naming cases passed; two failures identify issuer_reaction_publication
importing research/governance from swing. Four source modules pass strict typing;
eight changed Python files pass Ruff and have LF/identical parsed syntax trees.
One consolidated proof review found no further issues. The two demonstrated
ownership violations are repaired in pushed c2063a8: research-only issuer publication
and verification now live in research, with direct consumers/fingerprints updated
and old paths removed without aliases. Qualification is text-identical; all ten
changed-file syntax trees match after only prescribed import/path/root rewrites.
All 23 protected producer/replay dependencies match the actual pre-move request
hash inventory (issuer-producer-bytes-unchanged.json). Git-normalized bytes are not
used as a false baseline for protected CRLF working files. The closed sources and
original data/reviews/configs are unchanged. One bounded move review found no issues.
Final unit scope 59764 passed all 314 cases in 2198.53s; issuer-ownership-final
JUnit/stdout/exit files retained. These include full synthetic source-chain and
training-admission regressions, not a real market-data run. Ruff passed ten files;
strict typing passed five source modules. No owned worker or lease remains.
System memory after the unit exited: 73.87% used, 4.101 GiB available. Close this
documentation checkpoint before retrying actual 124 reuse with unchanged config.
Original action config is byte-identical; no action-reader failure is claimed.
The eight frozen evaluations remain pending actual source access under memory limits.

Pushed software: qualification authority/CLI; explicit old/current 124 reuse proof
with WTW/ATVI/INFO/SBNY abstentions preserved; 124-to-126 publication/replay and
research training consumers; pure 120-to-126 candidate adapter; funded raw-source
policy provider; saved temporal OOF evaluator/CLI; zero-cost benchmark repair.
Stock round trips retain 20 bps once. Zero cost requires benchmark role, canonical
SPY/QQQ/sector ETF identity, fixed horizon and no sale execution. Replay normalizes
only nullable-string storage after Parquet; exact text/null types, numbers, clock
precision, identities, targets, eligibility and ordering remain enforced.

Selected software verification (unit/fixture evidence only): core 215 passed in
1000.40s; remaining direct-consumer regressions 258 passed in 335.03s; final authority,
resource-pressure, string-replay and positive source-chain scope 43 passed in 706.87s.
Durable JUnit/stdout/exit files: review-results/core-resume-1912.*,
integration-tail-2007.*, qualification-and-positive-2031.*. The final positive unit
case computes both reactions independently and follows synthetic reviewed text
through publication and the actual replay implementation; it is not market-data
evidence. Earlier partial runs/failures are retained, not counted as complete suites.
Ruff passed changed/new Python; strict mypy passed 28 changed/new source modules,
and the final batched authority was rechecked. No full-suite/release check claimed.
Pinned config CRLF bytes retained; Git diff checking used core.whitespace=cr-at-eol.

Current reuse config configs/swing_return_relationship_reuse.json SHA256
95dd86ec733a70a72c0d0a2f05de78e0111b71e31fef1786d424136141cc209c.
Actual 124 reuse proof is not run yet. A passed exact source/value/clock/population
comparison authorizes a fresh current 124 derivative and independent receipt while
preserving original 120/124 bytes and targets. No blanket implementation-hash waiver,
current failure invention, matching replay or original indicator/target rebuild.
Qualification retains original 20 producer hashes as history; only two unrelated
HoldingContract/SourcePin dependencies may differ, other 18 and all data pins exact.

No actual 126 publication/readiness, remaining two fits or funded SPY result exists.
Frozen six learned specifications/two exits remain unchanged. OOF evaluation uses
retained temporal scores only: 718 score sessions, 708 entries through May 13, 2024 and
ten maturation sessions to May 28; no refit/search/positive-score/outcome filter.
Missing dividend/payment/residual facts must produce exact unavailable IDs/reasons,
not an invented NAV curve. The pure candidate adapter alone grants no admission.

Return-serving design is frozen for the subsequent software checkpoint: one V1
canonical fitted return-regressor path, no classifier fallback/pseudo-probabilities.
Candidate/bundle binds profile/order, ten-session SPY excess-return target, model,
preprocessing/source authorities and selection-wide information boundary. Signed
promotion and atomic activation remain required. Canonical fitted medians/missingness
indicators/scaler/predict_return serve contractual NaNs; reject infinity/text/bools
and shape/order mismatches. Rank signed finite predicted_excess_return with stable
security ties/sector limits, no positive threshold or outcome selection. Replace
response/persistence/snapshots/intents/monitoring together under /v1/. Observed live
qualified sources required; proxies cannot serve. Verify negative/>1 preservation,
batch/live/single parity, source/future poison, ties/zero selections, maturation
identities and unsigned/research-only rejection. Roll back only to a previously
verified compatible return generation, else abstain. Implementation not started.
Final acceptance requires 252 genuinely new decision sessions plus ten maturity sessions;
historical reruns and software passes do not prove SPY improvement or authorize it.

Latest inspected usage 47% used; update this handoff again at 99% used as requested.
No reset redeemed. Do not claim the whole Astra plan is complete.

### Historical checkpoint record (superseded by the continuation above)

### Latest continuation: pinned historical bridge inspection repaired

Implementation `0471c66` is pushed after the first real source job demonstrated
that `load_identity_bridge` rejects the retained bridge's old canonical sidecar
schema. The job exited before reading news/SEC bodies; the original output
`data/research/swing_initial_fit_issuer_content_review_population` contains only
a 4,687-byte `_request.json`, no checkpoint or SQLite, and no process remains.
Preserve that failed-attempt request. Do not resume it with changed code.

The narrow repair inspects only the parent-pinned historical bridge columns and
sidecar, checks artifact/type/research-only flags, row counts, proof SHA256 values
and typed nonnull UTC clocks, then rechecks all pins. Canonical production readers
still reject the old schema; no compatibility fallback or historical file rewrite
was added. Publisher tests: 19 passed in 5.86 seconds; Ruff and strict mypy passed.
This is a supported reopening of the new publisher's identity-clock consumer.

Next use the same config/hash and command below, replacing output with
`data/research/swing_initial_fit_issuer_content_review_population_source_bound`.
That is a fresh output for changed implementation pins, not a new API/data version.
No real source population, source annotation, feature publication or new fit exists
yet. Astra step 3 stays in progress; the software components below remain verified.


### October 5 current software checkpoint: source review, reaction projection and exits

Implementation `e2abb9f` is committed and pushed on
`codex/v1-canonical-cleanup`. Astra ordered step 3 remains `in_progress`;
this closes software components, not event admission or the whole experiment.
TradingFlow, main, raw archives, matched inputs and the four saved fits are untouched.

Implemented: bounded read-only Alpaca/SEC source iterators; original filing-index
CIK/name proofs; source-only announcement-cluster population and blind review
exports; frozen precision/rule/reviewer and weighted recall metrics; shared
124-to-126 reaction projection; and source-bound ordinary-lot frozen exit compiler.
The new command is `prepare-issuer-content-review` on the research CLI only.
Public APIs stay V1 and internal contracts are unversioned.

Consolidated supported findings were fixed: SEC identity is resolved at the parent
filing availability with its distinct identity clock; future revisions retain
metadata but never enter initial-fit sampling/blind text; full text exports stream
one version at a time; the latest independently pinned resume snapshot survives a
database commit followed by failed checkpoint publication; disk shortage preserves
that snapshot; and reaction diagnostic types are explicit under both pandas string
inference settings. Missing noncandidate reviewer pairs prevent qualification.
Uncertain completed reviews count as possible misses; no new recall cutoff exists.

Narrow reopening of the closed extractor: blind review must retain readable sources
without known issuer aliases, while the existing evidence adapter requires aliases.
Added `extract_source_text`, reused by that adapter, without invented identities,
event availability or admission. All original source bytes and pins were preserved.

Verification: combined 842-test component run had 841 passes and one batch/single
diagnostic dtype failure; the fix retained exact equality and passed all 313 affected
feature tests. Final publisher/CLI checks passed 89 tests after disk/SQL summary
changes; metrics passed 31, reader 15, exit compiler 29. These counts overlap and
must not be summed. RuntimeWarnings were errors in the combined and final root runs.
Ruff over all changed Python files and strict mypy over affected source modules passed.
Combined JUnit: `C:/Users/manis/Documents/Codex/2026-09-28/c/reaction-checkpoint-final.xml`
records the original failure, not an all-pass run. Full suite, real-source population
scan, source reviewer annotation, feature publication, model fit and funded SPY
evaluation have not yet run for this checkpoint. No verification workers remain.

Exact next action, after this documentation closure is pushed: run one leased
initial-fit source population job using
`configs/swing_issuer_content_review_population.json`, SHA256
`d2bb0e06d94acbb4ebd3fd9cf6493bcffe288aa7d69d3b142e42511f02e56659`.
Output: `data/research/swing_initial_fit_issuer_content_review_population`.
It reuses the completed inventories/archive; it does not reconstruct matched
features, targets or old aggregate news. No later sealed SEC content is opened.

```powershell
& C:/project/market-predictor/.venv/Scripts/python.exe -B -m market_predictor.research_cli prepare-issuer-content-review --root C:/project/market-predictor --config configs/swing_issuer_content_review_population.json --config-sha256 d2bb0e06d94acbb4ebd3fd9cf6493bcffe288aa7d69d3b142e42511f02e56659 --output data/research/swing_initial_fit_issuer_content_review_population
```

The job owns the absolute shared workspace heavy lease and a 5 GiB process limit.
Resume only with an independently recorded hash of the current `_checkpoint.json`
and unchanged request/config/implementation; use `--resume-checkpoint-sha256`.
The pointer binds an immutable `.resume/*.sqlite` snapshot; mutable working SQLite
is restored from it. A running job may advance that pointer, so do not treat a
previously observed checkpoint hash as the current one. Completed output is immutable.
Partial blind export requires explicit inspection rather than silent overwrite.

Then complete two independent source-only reviews of the frozen samples, bind each
annotation text hash/span to the original source, and publish measured family
qualification plus rejected/unclassified version history. No family is admitted by
the new software or by a blind export. Qualified event authority, actual 126-column
feature publication, training/inference reader wiring and the remaining two fits
are still missing. The full funded SPY evaluator also remains unfinished; the exit
compiler alone proves neither selected-trade source coverage nor outperformance.
Prospective final assessment requires 252 new decision sessions plus ten maturity
sessions. Do not label the exposed historical period untouched or promote a model
from software test results. User requests a handoff refresh at 99% account usage;
latest observed meter was 98% used. User said they reset usage; do not infer a reset
from chat when the account meter still reports otherwise.


### Active continuation: qualified events, reaction profile and frozen exits

The user authorized completing the whole Astra plan on October 5. Current step 3
remains in progress. Freeze this bounded scope before the next data publication:

- Reuse the completed initial-fit Alpaca inventory and corrected SEC archive.
  Publish source-linked versions, causal aliases, exact clocks, extracted spans and
  candidate/rejected/unclassified/unreadable dispositions. Obtain new causal company
  names from retained filing-index CIK/name blocks at corrected acceptance; a latest
  SEC name propagated backward is not a historical alias. No collector or matched
  aggregate/target reconstruction is required. Keep later SEC content sealed.
- Freeze candidate sampling at existing 300 earnings/250 guidance clusters and
  existing precision/issuer/joint/reviewer gates. Sample 600 noncandidate clusters
  per family by source/year with recorded inclusion probabilities. Report weighted
  recall, false negatives and uncertainty. No new 0.80 hard recall threshold is
  introduced: the profile claims observed qualified events, not complete event
  capture. Unknown coverage and underpowered recall remain explicit; never infer
  zero missed events or complete source coverage from the sample.
- Append only the two existing raw reaction measurements to the unchanged ordered
  124-column parent (126 model columns). Resolve latest available source versions
  before qualification filtering; rejected revisions cannot resurrect old text.
  Use the existing three-day news convention: (decision-72h, decision]. Collapse
  only evidenced duplicate groups by earliest qualified availability, then select
  latest distinct announcement deterministically before checking bars. Preserve
  every original row, label, weight and eligibility value; unavailable events,
  unknown coverage and incomplete reactions remain distinct null diagnostics.
- Accounting may proceed independently on synthetic/source-bound ordinary lots.
  Compile the two frozen target/stop/timeout and stop/timeout policies: next-open
  entry, ten XNYS sessions, +3/-1.5 raw-dollar ATR14 barriers, stop-first collision,
  stop gap at min(open,stop), target at target price and tenth-close timeout.
  Retain source facts; unsupported ownership changes before sale remain unavailable.
  The existing ledger, cost-once and bootstrap owners are reused. This compiler
  does not establish portfolio-source admission or funded SPY improvement.

Exit checks: source/alias/version tamper, sealed-before-body rejection, bounded
source streams, sampling/deduplication and annotation metrics, exact parent parity,
revision/clock/coverage poison, shared batch/live selection, exit collision/gap/
missingness/raw-ATR evidence, focused consumer checks, Ruff/strict mypy and one
consolidated review. No training or serving until the relevant authority passes.
On failure retain explicit reasons and original evidence; no permissive fallback.
Final promotion requires the frozen 252 new sessions plus ten maturity sessions;
historical implementation/selection cannot substitute for future observations.


### October 5 bounded checkpoint: issuer content review inputs (complete; profile remains pending)

Problem: the completed Alpaca content inventory and corrected SEC document archive
have no shared source-bound input for reviewing reported earnings and raised/lowered
guidance. The existing reaction measurement cannot establish what an event says.

Scope: one shared `catalysts/issuer_events/content_review.py` adapter/extractor for
saved Alpaca JSON and SEC HTML. Bind original payload, chosen field, extracted text,
HTML element locators, exact content-version and identity clocks; propose explicit
issuer-action statements with supporting spans and explicit-or-missing fiscal periods.
Every input receives candidates or an unclassified reason. Candidates never establish
training/serving admission. Preserve provider bodies, matched inputs and schemas.
PDFs remain explicitly unavailable to this HTML extractor; no guessed decoding.

Exit: source/field/text tamper rejection, wrong-subject/preview/form-only controls,
HTML/table/entity/hidden-text tests, revision and proxy/observed clock tests,
deterministic duplicate handling and shared historical/live extraction tests; Ruff,
strict mypy, one consolidated senior code/ML review and pushed checkpoint/closure.
Failure: reject mismatched bytes/clocks or unsupported decoding; no fallback event,
neutral value, feature publication, model fit or API activation. Qualification review
must later sample both proposed events and rejected/unclassified records to measure
missed real events as well as false event matches. No aggregate or target replay.

Verified source checkpoint: content inventory `3e4f905e...e9bda` is complete; corrected
SEC initial-fit documents `b3041a07...69318` are complete (45,160 archived documents).
The older metadata-only note below is historical. Existing precision authorities
admit analyst revisions only and do not admit this new full-content extraction.


Implementation `90a7f16` is pushed. The shared Alpaca/SEC adapters preserve exact
payload/field/text hashes and source-element span locators, propose earnings/guidance
annotations, and enforce content-version/identity/observed versus proxy clocks.
They add no training columns or admissions. Consolidated senior ML/code findings
were closed: backdated availability, another issuer's action clause and fiscal
periods borrowed from a separate event. A bounded saved-source probe also exposed
the original Alpaca producer's spaced-JSON hash encoding; the adapter now preserves
that exact encoding, with a compact-hash rejection regression. No saved bytes or
historical pins changed.

Final component verification: 548 tests passed in 74.48 seconds over content-review,
issuer-reaction, content-inventory, continuity, package and architecture checks;
RuntimeWarnings were errors. Changed-file Ruff and strict mypy passed. JUnit:
`C:/Users/manis/Documents/Codex/2026-09-28/c/work/content-review-source-contract.xml`.
Read-only source smoke: 32 corrected initial-fit Alpaca records, 20 headline fields
and 12 provider bodies, all hashes/spans reproduced. None proposed an event; this
small source-adapter check is not a precision/recall sample. SEC HTML behavior is
covered by synthetic tests; no archive-wide replay, sealed later-content access,
provider request, new fit, TradingFlow operation or full suite ran.

Remaining Astra step 3: publish source-linked development review populations and
sample both proposed and rejected/unclassified records; freeze annotation rules,
review agreement and precision/recall gates before examining review results. Then
create a content-qualified event authority, freeze event-to-decision selection and
final feature order, integrate the reaction profile through batch/inference and
training readers, and run its two existing model specifications only after admission.
No accepted model or funded SPY improvement was produced by this component.

Status: active

Last updated: 2026-10-05

Repository: `C:\project\market-predictor`

Branch: `codex/v1-canonical-cleanup`

This is the only active execution plan. Exact artifact state is recorded in
`docs/reviews/active_edge_rebuild_handoff.md`; statistical rules are defined in
`docs/model_training_validation_protocol.md`.

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

## October 4 User Priority And Frozen Checkpoint

Reuse the existing matched candles/news publication; ordinary code or naming edits
must not change dataset format or trigger blanket reconstruction. Preserve all
586,305 decisions, 59 months, original values/clocks, saved model predictions and
historical provenance. API protocols stay V1; internal names stay unversioned.
A naming difference alone does not prove numerical values changed. Verify any
claimed substantive difference and reconstruct only affected derived values.

The owned target reconstruction was intentionally stopped at the user's priority
change: 44 saved months / 434,992 rows; checkpoint
`cf2d1ed86acfa82cb796a8ca9800d0df5f803b1f6db6d052b42bcaa0017dbaaa`.
Private output is preserved, full replay never started, no final target publication
exists, and the shared lease has been acquired/released after stopping the worker.
Earlier instructions below to complete/replay all targets or replace the combined
profile format are superseded pending evidence that the existing values must change.

Completed scope (`da67c5c`): compare immutable saved technical and relationship predictions by
exact decision identity on unchanged folds, labels, training weights and holdouts.
Add one canonical comparator and CLI without changing training, feature/target
kernels, stored data formats or current model-admission rules. Verify independently
pinned requests/unit/artifact bytes, full scoring population and exact metadata.
Calculate pooled equal-date squared/absolute error against the paired model and
zero excess, and daily ranking quality on common evaluable dates. Separate temporal
and security-transfer scopes. This is descriptive development evidence, not SPY
portfolio performance, a new fitted model or an untouched final test.

Exit gates: unequal-fold/session numerical fixtures; exact pairing and population/
null/eligibility/settings rejection; constant/tied rank handling; artifact tamper
and final-pin recheck tests; focused direct-consumer/CLI/architecture tests, Ruff,
strict mypy, one consolidated review, real saved-prediction report, pushed code and
documentation closure. Any mismatch stops reporting without altering source data.
Next work follows original checkpoint 3 and its existing source/feature acceptance
requirements; no new experiment is inferred from the diagnostic result alone.
TradingFlow remains untouched while its owner is changing it.

## Objective And Boundary

User-confirmed V1 workflow (October 3): collect candles/news/company events; clean
duplicates and align security identities, corporate actions and trading sessions;
build features using only information available at each decision; build future-return
targets separately; train and evaluate; iterate using validation; assess the selected
model on untouched test data before connecting an accepted model to the API. Return
regressors use prediction error, ranking and baseline/portfolio comparisons; AUC is
for classification, not a substitute for those return-model measurements. Repeatedly
inspected test periods become development evidence and cannot remain labelled unseen.

V1 is the initial end-to-end system across APIs, data models and features. Work on
new branches; no version increments for implementation iterations. Merge accepted
improvements after measured evidence, and consider V2 only as a deliberate later
system release. A claim to beat SPY requires a reproducible comparison of net portfolio
performance against funded SPY over matching dates/capital/dividend treatment, with
costs and drawdown reported; an AUC increase, mean trade return or software test pass
does not establish that outcome. Freeze evaluation choices before the final test.

Naming rule (updated by user, October 3): the initial complete system remains V1.
The current branch producer declares `market_predictor.prediction.v1` on `/v1/` routes;
TradingFlow adoption remains pending. No iteration-driven V2/V3 or compatibility code. Models are not final, so model, record, class, file and other internal
names carry no version number (`V3`, `.v3`, `_v1`) in either repository.
Old-format records are refused by strict schema/shape validation; there are no
compatibility aliases. Original raw evidence and historical receipts retain their
bytes; those records do not make old derived artifacts valid for the current code.

Current product scope: **long-only swing (roughly one to three weeks) and a
separate open-ended investment cohort**, with verifiable net performance
against buy-and-hold SPY. Dedicated day-trading strategies and training are being
retired by user instruction, not maintained for compatibility. Minute/hourly bars
needed for swing entry timing, protection and causal evidence remain supported.
No profitable or production-admitted candidate is asserted.

The existing research horizons are:

- **Swing:** ten-session stock direction, managed return, and excess return against
  SPY, QQQ, and the point-in-time sector benchmark.
- **Historical intraday:** thirty-minute artifacts are historical research, not
  the unified product. Their raw evidence and hash-bound provenance are retained.
- **Investment:** separate 63- and 252-session forecast/label horizons were approved
  September 25. Holding remains open-ended; allocation/risk budgets remain undecided.
  Do not substitute swing predictions or interpret the label horizon as a sell deadline.

This repository produces predictions, abstentions, explanations, benchmark
comparisons, and matured outcomes. It does not produce alerts, orders, positions,
portfolio risk, or execution instructions. `trading_flow` may consume only a promoted,
versioned prediction API.

Offline portfolio accounting is necessary to evaluate predictions. It is not live
portfolio management: alerts, orders, final position sizing, and execution remain
outside this repository.

## Shared Candle Flow And V1 API Recommendation (October 3)

Status: candle design review recorded; no collector settings, strategy, retention
policy or TradingFlow runtime changed. The branch API reset is tracked separately below. This recommendation does not replace the investment
dataset checkpoint. The user requests common candle collection while applications
remain separate, smaller histories after day-trading retirement, and V1 public APIs.

Recommended flow: one configured collection owner for each provider/data product,
raw receipt capture, deterministic normalization, immutable publication, then separate
TradingFlow and Market Predictor consumers. Reuse Market Predictor's existing daily
REST collection as the initial historical publisher; this is a proposed ownership
choice, not an already completed migration. TradingFlow keeps its existing live
quotes, order updates and protection responsibilities. Do not introduce another live
stream owner or let a prediction-service outage interrupt position protection.
Deduplicate requests by security, provider/feed, adjustment, interval, session scope
and covered window. Include universe members and required benchmarks; TradingFlow
requests additional watchlist coverage through the same collection owner.

Recommended intervals and initial cache targets (engineering defaults, not measured
optimal trading parameters):

| Consumer/purpose | Candle interval | History and refresh |
| --- | --- | --- |
| TradingFlow swing setup/indicator warm-up | Daily | At least 300 completed exchange sessions, or the larger dependency-derived indicator history including lags and recursive initialization; then append completed sessions. |
| TradingFlow new swing entry-confirmation baseline | 15 minutes | At least 20 completed sessions, or the larger indicator requirement; shortlisted candidates and active orders/holdings only. Use fresh quotes separately for spread/price checks. |
| Existing TradingFlow entry/protection strategies | Explicit configured interval | Keep required 5-minute/hourly streams until each strategy's dependency and replay tests justify removal. The 15-minute baseline is not a substitution for existing strategies. |
| Market Predictor swing and 63/252-session investment | Daily | Retain the admitted multi-year training history plus feature warm-up; nightly completed-session updates for prediction. A 300-session desk cache is not a training-history limit. |
| Optional hourly confirmation | Session-aligned hourly | Derive from compatible completed 15-minute bars where parity is proven; include the shortened final bucket and early-close sessions. |

No default four-hour product: a regular US equity session is 6.5 hours, so a
session-anchored four-hour scheme leaves a 2.5-hour remainder and needs its own
tested feature semantics. Daily aligns with the current models; four-hour inputs
would require a separate feature/training comparison, not a collection-only switch.

Stop broad minute-history expansion only after identifying consumers of each job.
For a normal regular session, 390 one-minute buckets become 26 fifteen-minute
buckets (about 93.3% fewer rows for equal symbols/windows), or one daily bucket.
These are bucket-count estimates, not measured disk savings or proof that the
current nightly job stores minute data: Market Predictor's incremental collector
already requests `1Day`. Preserve raw/hash-pinned historical evidence and required
fine-grained execution replay; coarse bars cannot resolve stop/target ordering inside
a bar. Removing day-trading models does not remove live protection requirements.

Keep raw/as-traded and adjusted daily data as explicit separate products. Execute
and check absolute prices on raw values; consume the adjustment policy frozen for
each model/indicator. Never silently replace existing daily data with an aggregation
of regular-hours intraday bars: provider daily-session coverage and corporate-action
semantics must match first. Record security/symbol mapping, OHLCV, precision, interval,
feed, adjustment, currency, exchange session, bar start/end, availability and receipt
times, revision identity and source hashes. Preserve immutable revisions; exclude
unfinished/future bars at the decision cutoff and do not invent zero-volume bars for
unexplained gaps. Use the exchange calendar, including daylight saving and early closes.

Observed migration work in the independently changing TradingFlow checkout:
`WarmupModels.cs` defaults to 60 calendar days across 5m/15m/1h/1d; entry preparation
uses one calendar window for all intervals. Its backtest profile asks for 260 indicator
bars, while paper asks for 150 despite SMA200/EMA200 calculations. Split daily and
intraday windows and verify actual sufficient completed bars, rather than changing
one global number. Existing strategy configs explicitly use 5m and 1h execution.

V1 public boundary target: keep `/v1/` routes and freeze prediction JSON with
`contract_version = market_predictor.prediction.v1`; the proposed candle read/publication
API uses `contract_version = market_data.candles.v1`. A candle API may deliver small
windows directly and immutable manifest/partition references for bulk history; the
transport, exact fields, decimal representation, paging and error schema must be
frozen in the shared OpenAPI/JSON contract before implementation. These are public
wire versions, not new versions of Python/C# internal feature or model classes.
Current TradingFlow branch `unified-swing-product` accepts unversioned
`contract = market_predictor.prediction`, while the producer previously served v4; an older
merged integration receipt does not establish compatibility with that working tree.
The user explicitly ordered the producer cleanup now without compatibility. Producer
and consumer adoption are separate recorded steps until the TF owner updates its
working branch and fixtures. Reject unknown versions explicitly. Historical V1 shapes
are not supported by the current strict schema; preserve historical receipt hashes.
The future investment forecast API also belongs under V1, with explicit 63/252
session horizons; it is not currently implemented and must not return swing forecasts.

Implementation exit gates: agreed collector ownership and published V1 schema;
Python/C# fixture parity including decimals, timezone/session boundaries, revisions,
partial bars and missing coverage; sufficient indicator history per interval;
duplicate-request/restart tests; training/live daily feature parity; representative
entry/protection replay on each retained strategy timeframe; matched version/error
handling and a measured before/after collection-volume report. Until these pass,
existing collectors and version checks remain unchanged; insufficient or incompatible
data returns a specific unavailable reason, with no silent provider/timeframe fallback.
TradingFlow changes must be coordinated with its current developer; this review does
not authorize interfering with that checkout's ongoing work.

Sources checked October 3: [Alpaca bars](https://docs.alpaca.markets/us/reference/stockbars)
for native intervals/feed/adjustments; [NYSE calendar](https://www.nyse.com/trade/hours-calendars)
for regular sessions and early closes. Interval recommendations are design judgments.

## Unified Product Implementation

Investment target projection completed in `147e5bd` and pushed on main (October 3).
The additive investment package binds four holding specifications and their source
identities, verifies next-open entry and exact 63/252 session closes, prohibits ordinary
sales, and selects the horizon snapshot from the unchanged shared accounting kernel.
Corporate claims/payment timing remain kernel-owned. Net return deducts the explicit
20-bps research assumption once; comparisons use matching gross benchmark values.
Unknown valuation/availability produces unavailable supervision with specific reasons.
Result flags explicitly refuse training readiness, promotion and production admission;
source identity matching alone does not establish underlying source authority.
All four pinned swing accounting/ordinary-holding files remain byte-identical.

Verification: 16 target tests passed; 231 architecture/dependency checks passed (230
first, one temporary-directory setup error rerun successfully with writable TEMP).
Ruff clean over investment sources/tests and dependency guard; strict mypy clean on
all three investment modules. Consolidated code/ML review passed. No real data read,
training, source publication, promotion, deployment or TradingFlow operation occurred.
README now states these limits and the completed monitoring/replay behavior.

Completed checkpoint: **Canonical pre-production code cleanup**.
Implementation `af9e9b4` is pushed on `codex/v1-canonical-cleanup`; main and TradingFlow
remain unchanged. Internal schema/policy generation suffixes and ml_v3 names are gone
from active source; public communicating identities remain V1. The naming guard permits
only the existing public V1 protocols, not internal version suffixes. Vendor URLs and
original archived paths remain provider/evidence identities, not compatibility paths.
Removed the weaker Alpaca reader and duplicate training-schema declarations. Temporal
consumption now uses the actual current producer's session/date columns and final layout.
Preserved original raw data and historical collection fixture bytes. Current future-build
config bindings and synthetic served fixture were regenerated; saved research provenance
was not relabelled as current. API consumer adoption and candle-flow implementation remain
separate pending work, with exact TF changes in docs/contracts/tradingflow_handoff.md.

Verification: full suite finished with 4,869 passed, 11 skipped and eight old identity/hash
assertion failures; all eight were explained by the naming reset and corrected without
changing feature vectors, attribution/classification rules or historical raw hashes.
The final rerun of affected files plus new config/naming/continuity checks passed all
152 tests. Strict mypy passed all 97 changed source files; Ruff and diff checks passed.
Independent design review handled the two schema collisions; consolidated code/ML
review found no supported code issues. No real model fit, SPY outperformance, promotion,
deployment, data reconstruction or C# build is claimed. Full test logs are local task
artifacts, not model evidence. No full-suite rerun was needed after test-assertion fixes.

Current checkpoint: **Qualified issuer-reaction feature engineering** (`historical status`).

Completed source sub-slice in `86f7d429487934494da9c587ad9536cbf993e2dd`, pushed on
`codex/v1-canonical-cleanup`: explicit offline reconstruction into
`data/raw/swing_incremental_alpaca_canonical`. All 6,972 success units / 7,228 pages
passed current query/body/UTC-clock/pagination/counter checks; 193,248 rows are raw
unit observations including repeat captures, not unique training rows. Current offline
collector verified all 6,132 daily/revision units through 2026-10-02 with zero failed,
pending or altered units. All original 14,054 file hashes still match; compressed
archives and provider retrieval clocks are unchanged. No provider request was made.

Independent pins:
- Input inventory `data/evidence/alpaca_incremental_reconstruction/input_inventory.json`:
  `4ccc387fd3cfb0e4b928c5318f32e88d4a6ab60a1d6a9e828202bb457332f294`.
- Published `_reconstruction.json`:
  `fb0513fd5647034759a2514d791f74f1b0bbd27aabc0336882ddcf7ad74d54e4`.
- Published `status.json`:
  `da87cb5544f9759aaa715c61dd42a0cb0bae8627b3987a7a041ac8b38e94c850`.
- Canonical request identity:
  `f5b16cb5daf661afca5a9f47a0ef84224c82b8e14f86739638bc4cb159a14a5a`.

The original archive has 63 revision and 21 partial capture intents. Of these,
80 contain complete success archives; four contain none: revision intents for
September 29/30 and October 1 captured October 2 (42 units each), and partial intent
`2026-10-02T22:00:15.832848+00:00` (42 units). These 168 unobserved attempt units are
explicit unavailable evidence in the immutable report, not missing daily history or
invented retrieval clocks. The explicit `--allow-empty-capture-intents` option retains
their original hashes and excludes them from completed capture metadata. Missing daily
units, partially populated captures, changed query policies, altered pages or input
inventory changes still fail publication. Incomplete staging has no final authority.

Verification: 64 focused incremental/naming/continuity tests passed before the narrow
revision-day grouping fix; all three empty-partial/same-date-revision/mixed-revision
regressions passed afterward. Changed-file Ruff and strict mypy passed. Plan/design
review and consolidated code/ML review found two implemented fixes: bind a frozen
source-file inventory and match revision scopes by target day plus capture date.
No full-suite repeat, model fit, promotion, final-test payload access or TF operation.

Fresh historical acquisition plan also produced and independently replayed using
`configs/swing_initial_fit_raw_share_plan.toml` into
`data/research/swing_initial_fit_raw_price_requirements`. Authority pin:
`e5f81597594f85485eaa83e75cd81b5f849e20c63dc3ab3f89b3cbb85c6baf5b`;
manifest pin `7eada113427c7d74ad81b65f62a708418601e61c19ce2039090500c42afe028d`.
Its 564 daily raw-price units (551 stock windows / 13 benchmarks) cover 545 in-window
securities and 586,305 decision identities; 581,455 have ten sessions before the
initial-fit cutoff. It reads only decision/membership identity columns in
2019-07-09..2024-05-28; `outcomes_read=false`. Peak memory was 0.391 GiB. All 92 inherited
source pins matched. This is a source acquisition plan, not admitted price coverage,
features, target values or a trained model.

Bounded historical transport inspection: the existing initial-fit raw archive reports
564 observed units / 601,834 rows with `transport_receipts_required=true`. One EVRG
unit has 1,231 rows and a hash-matched 129,874-byte terminal provider body, original
retrieval `2026-09-09T14:29:02.594674+00:00`, matching request/final URLs, HTTP 200 and
no redirects. The daily verifier is `swing/datasets/history_archive.py`; intraday
transport requirements do not apply. This is sample feasibility, not full archive
admission. Next publisher must check all unit/query/body/clock evidence against the
fresh plan and rebuild separate current-schema bars/receipts, then round-trip the
strict current reader. No missing-transport or redownload claim is supported here.



Historical raw-price reconstruction completed with the canonical collector in
implementation `2216a2d` (pushed):
`data/raw/swing_initial_fit_raw_share_canonical` contains 564 observed units / 564
terminal pages / 601,834 rows. The normal current reader passed; zero failed or
unattempted units. All declared retained pages were consumed, original source hashes
still matched and the fresh plan remained unchanged before publication. Original
provider entity bytes and retrieval clocks are preserved. Derived wrappers, paths,
bars and ingestion times record this actual reconstruction, not an old capture.
Independent collection authority:
`3cd0699a090208dcd3455bb0be33cd0243ce303ce1cc3af1cd48f2f4ef758c28`;
manifest `3ca4a734da1b957f24e4c424a3a69d22f8b5631936a4e6910ac139ab86ebe4ef`;
reconstruction receipt
`d8726f10680ca6d48952085ee202becd91bf2457c2fbc36db6c21fd852924424`.
All 44 daily-history tests passed, plus 3 naming/continuity checks. Ruff and strict
mypy passed the new source. Plan/design and one consolidated code/ML review passed.
No provider requests, label/evaluation payload reads, training or TF changes.

Corrected-source sub-slice completed in `fdf2b40` (pushed). The reviewed mapping,
date intervals and seven issuer documents are unchanged; only the verified parent
source dependencies changed. Fresh raw correction plan:
`data/research/swing_symbol_correction_requirements`, authority
`fcbe0692e3983d9aa97478b5a84cb7e3a6b84e20e583926c64171b9c44e320dd`.
Fresh adjusted plan: `data/research/swing_adjusted_feature_requirements`, authority
`2dde2bd9d546e4553b3fc9dbcb23d5017b9b96aedc5b00d291bdcb9e1fed38c7`.
Current correction-policy hash:
`259e5328a150eb25f67ab51a55af908f5639744f4b9c0970741d7863bf009fa1`;
adjusted-policy hash `c63314cb9daebcee5142395e766a5932ba7854596a7707060e7ea5602f528818`.

Canonical collector reconstruction/current-reader round trips passed for both
archives: `data/raw/swing_symbol_corrected_canonical`, two terminal pages / 1,476
rows, authority `ee0f8bc3c66bb8313ef2a4bc06ae7faea36203a7a1d6e451103ae75fff2a692f`;
`data/raw/swing_corrected_feature_history_canonical`, two terminal pages / 3,020
rows, authority `62f1ebe110bcf7306dccdeb0939ed366aee23e783ce2080c9b1882721f49346e`.
Original bytes/clocks and all original source hashes remain unchanged.
New source selection `data/research/swing_symbol_corrected_sources.json` hash
`093f497847532f03e029b213b5f98efb6d2d1c8e927e80c5d78f4b30719119c4`
selects 602,709 raw rows, discards 601 incorrect ECHO parent rows and retains 259
missing sessions / 21 unusable observations. No exclusions added or prices filled;
accounting/label/promotion eligibility remains false. Historical availability is not
asserted by retrospective collection clocks. All 203 affected policy/collection/CLI/
issuer-identity/proof tests passed; changed-test Ruff and diff checks passed. One
consolidated code/ML review passed. No new production code, full-suite repeat,
provider call, label evaluation, model fit or TF operation.
Five downstream outcome/issuer/news/legacy-proof configs still bind the old policy;
they require actual reconstruction before admission, not blind hash replacement.

Standalone adjusted-source sub-slice completed in `ce1d07c` (pushed). Canonical
requirements/publication and an authority-bound shared-normalization reader now replace
the need for old combined-panel source authority in the next feature implementation.
The plan preserves all 564 parent query windows (551 stock windows / 545 securities,
13 benchmarks), recorded decision identities and reviewed FI/SATS mappings. Other
aliases remain independent; query history never becomes membership evidence.
The pinned original benchmark coverage gives XLC a June 19, 2018 start, with 264
sessions before the first decision; all others start May 29, 2018. Every benchmark
meets the existing 253-session minimum. Source reads end May 28, 2024.

Publication now stays private until every unit normalizes and source/plan hashes
still match. Partials resume in `.<output-name>.collecting`; a rejected plan remains
private in `.<plan-name>.planning`. No failed gate exposes the final authority.
Required benchmark gaps/invalid observations reject; stock gaps/invalid candles remain
explicit for per-decision feature abstentions, with no population exclusions or fills.
Actual provider retrieval and derived ingestion times remain distinct from the
established historical close-plus-15-minute feature-availability assumption.

Real Alpaca SIP collection and independent completed offline replay both passed:
`data/raw/swing_initial_fit_adjusted_canonical` has 564 observed units / 564 terminal
pages / 794,279 raw candle rows, zero failed, unattempted or wholly unavailable units.
All benchmark units passed their required session coverage and observation checks.
Stock query windows have 16,170 absent session dates and 582 unusable candles;
these counts include query warmup/alias/IPO scope, not a claim that every absent date
was an eligible membership/decision date. The feature consumer must calculate the
actual affected decisions. No prices imputed, decisions dropped or exclusion added.
Canonical normalized bars omit those unusable candles and report their dates explicitly.

Independent pins:
- Config `configs/swing_initial_fit_adjusted_history.toml`:
  `684376fbde1b83e78f208dcedf32d67c13ce02d982b4aa0b071df46878f0e24f`.
- Plan `data/research/swing_initial_fit_adjusted_price_requirements/_authority.json`:
  `2c1e5d141e5f17f93f0977c2a98e3f69aa7751425b6e23bb583ede3f08a64d73`.
- Archive authority: `e1095ace476c4bca02511b857be5b377e99dee224936751c92915150ad4dcabc`.
- Archive manifest: `a624b48016cdffaffbd89daeed9878f022353a0770b74996158320c500cb399d`.
- Archive request file: `d9ed5abda2075d6e28aef16c42e6502d3c523763587b39454312d9be3a1eae3c`.
Logs in task workspace: adjusted-live-probe.log, adjusted-live-complete.log and
adjusted-offline-final.log. The one-unit probe exited incomplete as expected;
full collection and offline replay exited zero. Collection metadata records peak
working set 0.177 GiB before final normalization; do not represent this as a measured
whole-job peak. Four-GiB runtime guards remained active.
Verification: 35 plan/publication tests after review fixes; 33 adapter tests;
277 direct collector/architecture/package/continuity tests before the narrow
publication fixes. Changed-file Ruff and strict mypy over all three source files
passed after final fixes. One consolidated code/ML review found and closed missing
benchmark normalization and publication-before-source-recheck gates; poison tests
cover each. No full-suite repeat, label/feature publication, model fit, promotion,
main merge or TF operation. Original archives remain untouched.

Canonical technical predictor integration completed in `f8d266f` (pushed).
One raw-price/identity context now serves feature construction without opening
corporate-action targets; the outcome wrapper retains all action checks. Adjusted
histories bind exact original parent windows to independent query units. Shared
relationship/replay consumers use the same bindings; obsolete combined and special
issuer readers are removed. Quarantine applies to the exact failed query group,
not every window of a security. Windows paths preserve equivalent configuration pins.

Real publication: `data/features/swing_initial_fit_technical_inputs` contains all
586,305 decisions / 551 groups / 59 months, from 2019-07-09 through 2024-05-28.
564,557 rows have usable technical inputs; 21,748 retain explicit
`technical_warmup_or_unavailable` reasons. No decisions/exclusions were added or
removed. The technical contract has 40 ordered feature names. This is feature-only
reconstruction, with training/promotion false; it does not admit labels or models.

Independent pins:
- Feature config: `1a4a3832bd20e8a8ab1aafa734e0de95fa77b7162f43be427c6f91e7aae3cbd7`.
- Price/decision config: `2be2077ae44a1cddfb02feeac139a23246f481d9159455955a956fd94ecbecdf`.
- `_request.json`: `338886cc41ec5ec97d513b48b69e36476de9c28fabf8b7e261ccb39dc41f596c`.
- `_checkpoint.json`: `04f9933ebcb9d1527dd00e4348519242cd1361b57d0e822d60507338487708ad`.
- `_manifest.json`: `c1601b6683893dcf9d3885b3c7446430dafecd96ef3e06d835c910338d016891`.

Independent output audit checked every declared/source/implementation pin, every
canonical group/month artifact hash, exact unique IDs/population agreement, parent
IDs, decision bounds and all non-null availability clocks at/before decision time.
It did not numerically replay all real feature values. Numerical reconstruction,
future-poison and tamper checks ran on the affected synthetic integration fixtures.
Final selected component checks: 59 relationship/publication/input tests (402.03s),
339 predictor/feature/architecture/CLI tests (136.50s), 68 derivation/review tests
(27.29s), and 292 memory/source/CLI/architecture tests after the runtime fix (13.37s).
These overlapping test sets are not summed. Changed-file Ruff passed; strict mypy
passed the 16 feature/consumer sources and two final runtime modules. One consolidated
review closed exact-window quarantine and Windows-pin findings; observed memory
failures justified a narrow runtime reopening, now reviewed and tested.

Resource receipt: initial assembly saved 551 groups/six months before the 85% system
limit stopped it. Cache-only retries also stopped; no guard was lowered or bypassed.
Unused Arrow allocations are now reclaimed after the unchanged RSS measurement;
native trimming is excluded from that guard path. A fresh process with
`ARROW_DEFAULT_MEMORY_POOL=system` completed the exact saved request/checkpoint.
Sampled monthly peak was 0.804 GiB, not a whole-job peak claim. Keep this allocator
setting for the next heavy jobs on this machine. All owned workers exited.
Task-workspace logs: `canonical-feature-system-pool.log` and
`canonical-feature-output-audit.log`. No full-suite repeat, model fit, main merge,
promotion, provider request or TradingFlow operation occurred in this feature slice.

Corporate-action source reconstruction completed in `b97ad60` (pushed), with
hash-bound CRLF whitespace handling fixed in `635ebdb`. New archive:
`data/raw/swing_corporate_action_sources_canonical`, 570 queries/pages and 14,352
records. All 1,712 original file hashes, response-body/metadata bytes and acquisition
clocks match; only current request-bound wrappers and reconstruction provenance are
new. Independent full output audit and normal final-path offline replay passed.
Private staging is gone. Historical announcement availability, absence of actions,
ownership and accounting eligibility remain false: this source receipt admits no labels.

Independent pins:
- Scope `configs/swing_corporate_action_sources.toml`:
  `c883953e51df99990523be95504e34c6ae7d5ff48ba20e3951d2d7e8d0262554`.
- Semantic audit: `a713f53f2169be60ddd5cbf81786be8f772127496d65a93862eb72992f643add`.
- Report file: `a08278e384c41742015e6c26c8ebca755e53a9e7c9080375c556d7717ffa22af`.
- Request file: `bc3afc9b2db3389e50764d93ab97be8b944668172603563d08076e8707aeb58b`.
- Checkpoint: `071b15d28f05ca61f33bfa6faf611765e0da83483ab950c4e3ada0622e041d22`.
- Reconstruction proof: `66d50cea01b673a35f47526fb0c65ef8c8e40bc347b45300d70e2049a387bdb9`.

Verification: 24 scope/config checks; 87 provenance/collection/evidence/CLI checks
before the final map guard; 40 producer/provenance/evidence checks after that guard;
390 affected architecture/provider/scope/CLI checks. Sets overlap, not summed.
Changed-file Ruff and strict mypy over six sources passed. One consolidated code/ML
review reported no supported P0/P1/P2 findings. Final whole-checkpoint diff check
passed after preserving the scope's exact bytes with a path-specific CRLF attribute.
No full suite, provider request, target calculation, model fit, main merge or TF work.
Task-workspace logs: corporate-action-canonical-complete.log,
corporate-action-output-audit.log and corporate-action-offline-final.log.

Target-config separation and join proof completed in `12baec5` (pushed).
`configs/swing_canonical_targets.toml` has SHA
`ec90490445c88b0d4225130903de74c6361565ec96104d18f5f6043fe15bb940`;
only the three action bindings differ. Existing feature/decision configs remain
byte-identical. The typed proof hashes every other policy field. Join/CLI require
independent target pins; predictors retain their decision pin; targets bind exact
config/source-selection/action-audit lineage. Both policies, proof and helper code
enter request/resume identity. Every target month and resumed profile compares all
canonical decision metadata and parent/context fields, not IDs/counts alone.

Verification: 28 helper tests, 13 CLI tests and exact replay-config regression pass;
410 affected integration/replay/architecture/CLI checks passed before the one review
fix. Review reproduced nullable-string versus inferred-string ID dtype rejection.
Normalize only validated nonempty text, preserving exact clocks/values; actual
canonical stamping -> target dictionary -> Parquet -> fresh/resume regression passes.
85 focused dtype/metadata checks passed; final 249 full join/direct-consumer checks
passed in 186.90s. Overlapping sets are not summed. Final changed-file Ruff, strict
mypy over three sources and staged diff check passed. Consolidated review closed
its single P2. No real target/news/model/test values, training, promotion or TF work.
Task-workspace logs: target-join-integration.log and target-join-final.log.

Real target pilot and independent audit completed; full publication remains in progress.
The July 2019 pilot in private `data/labels/.swing_initial_fit_targets.materializing`
had 7,884 rows, 6,409 complete stock/benchmark comparisons, 1,475 retained
unavailable rows, 6,630 holding specifications and zero terminal-immature rows.
Independent audit consumed the full 586,305-decision metadata projection and checked
all 5,884 source/implementation pins plus exact published metadata, null/reason flags,
component references and ten-session maturity. Numerical/specification replay then
passed for the pilot; its checkpoint was
`b9b130c7924e4ab38d3a9e6004dd789caf21d23ad8911aba36557497cc2cbaeb`.
Request file pin:
`01357a4515158ed7ef9a74a803c923b74243b70b1d4be22eea97c213dc20c2ad`.
Pilot/replay wall times 403.82/340.35 seconds; process peak working sets
0.494/0.494 GiB. These are individual-run measurements, not a full-job estimate.
No final target directory/manifest, training or model/SPY result is published.
Logs: target-canonical-pilot.log, target-canonical-pilot-audit.log and
 target-canonical-pilot-replay.log (task workspace). Correct command surface is
`market_predictor.collection_cli`; the initial research_cli invocation refused the
command before creating output or reading targets. All subsequent runs use collection_cli.

Unavailable source reasons include unresolved distribution entitlement/payment,
undated action effects, five official-scope reason hits and eight ownership-gap hits.
Reason hits overlap, so they cannot be added to count distinct unusable rows.
No source capture clocks were invented or declared missing by this audit.

October 4 continuation: the full build saved 325,758 decisions across 33 months,
July 2019-March 2022, before the system-memory guard stopped April's unfinished
calculation: 86.3% used against the unchanged 85% maximum, with 2.14 GiB free.
The target process peak was 0.495 GiB. Saved checkpoint SHA:
`580a9652d04937325bfb20fbbcb02d35c04c61f8e2f0fe880ce6fa3cf81110a3`.
Only the unbound April staging directory was removed; its file hash and removal
receipt are in task workspace `target-canonical-memory-restart.json`. Raw inputs,
saved months and source/config bytes are unchanged. With about 3.1 GiB free,
the remaining 26 months are resumed under the same guards into the same private
directory; log `target-canonical-complete-resume-absolute.log`, with stderr separate.
Use explicit `--root C:/project/market-predictor`: the current runtime resolves
relative `.` to an extra nested directory. The first restart therefore exited at
config lookup before target reads or writes; its log is
`target-canonical-complete-resume.log`. The failed memory run's log is
retained as `target-canonical-complete.log`. Full replay/audit/publication have not
run, and the final directory still does not exist. Workspace audit/finalizer review
found no supported faults; successful full native replay remains an external gate.

Latest saved state: April 2022 completed, 34 months / 335,338 decisions; checkpoint
`c470ae76508ae959a69aeed197028456acf43f1b795ac017735645e884c6d23f`.
May stopped at 85.3% system use / 2.31 GiB free; own peak remained 0.494 GiB.
The unbound May specification file was removed with the hash/receipt retained in
`target-canonical-memory-restart-may.json`. No full replay or final publication ran.
Next attempt uses workspace `run_target_build_and_replay.ps1`: one control shell,
one producer at a time, progress reads in that same shell, no separate status
processes during the build. This reduces monitoring overhead without changing any
memory limit; it is not yet evidence of a completed run. Logs are
`target-canonical-single-shell-build.log` and, only after successful full build,
`target-canonical-single-shell-full-replay.log`, each with separate stderr/runtime.

Next sub-slice frozen: materialize and independently replay initial-fit targets using
that exact new policy. All 28 checked dependency/document pins match; research and
simulation contracts parse; the QQQ receipt-based report matches its reviewed hash.
59 ordered decision months span July 2019-May 2024; selected raw price segments end
no later than May 28, 2024. Read future prices only for targets whose ten-session
horizon matures by that cutoff. Retain terminal-immature and unsupported-action rows
with null targets/reasons, all 586,305 decisions and the approved population; no fills,
extra exclusions, news/test/model reads or feature rebuild. This is retrospective
research supervision, not historical announcement availability or production admission.

The unchanged outcome producer writes its final manifest before source-context exit;
therefore run entirely in private `data/labels/.swing_initial_fit_targets.materializing`.
No production source edits are needed. Under unchanged shared-lease/5-GiB/85%-system
memory guards and `ARROW_DEFAULT_MEMORY_POOL=system`, build one month, independently
replay that month using its external checkpoint pin, then resume the remaining months.
Fully replay all 59 months and compare target/specification bytes. Independently audit
all declared source/implementation/output pins, exact monthly metadata/parent IDs,
unique 586,305 decisions, unavailable/maturity counts and false training/promotion/
managed flags. Only after successful return/source-context exit, reacquire the shared
lease, recheck all frozen input/output bytes and atomically rename to
`data/labels/swing_initial_fit_targets`. Normal final-path resume with the manifest
pin must then pass. Preserve original labels and archives. Any failed check leaves
this publication private; resource stops resume only with the exact saved checkpoint.
Source admission, numerical replay and measured availability counts close this stage;
no model fit, SPY outperformance, main merge, promotion or TradingFlow operation.
Actual news/join/training and untouched-test assessment remain subsequent checkpoints.

Read-only next-step design (not implemented): one explicitly declared research
profile per publication, plus independent saved-row reconstruction and a canonical
receipt consumed by readiness, return inputs and relationship parents. Technical
uses predictors/targets only; catalyst-full also requires its catalyst authority.
The old combined technical profile gates eligibility on news; the standalone
profile must use predictor eligibility, so peer values need not match that old
profile. Reuse numerical kernels, retain all decisions/targets/nulls/clocks, and
reject missing/surplus sources, forged receipts, stale resumes and changed bytes.
Freeze this code-only scope only after target publication/replay/audit closes.

Cleanup requirement and scope retained for review:
User clarified that V1 is the initial complete collect/clean/features/targets/train/
validate/test/API system, not the count of implementation iterations. All changes
are on a new branch; merging or a future V2 requires deliberately accepted measured
improvement, not passing software checks alone. No compatibility is required.
Scope: remove internal generation suffixes and ml_v3 naming from canonical writers,
readers, configs and owned fixtures; communicating contracts stay V1. Delete the
weaker Alpaca transport reader rather than rename it into a parallel format; use one
current panel materialization contract in temporal consumers. Preserve provider URLs,
raw bytes and historical records; old derived artifacts are not automatically admitted
after source/schema changes. No rehash may claim an unperformed dataset/model replay.
Exit gates: no versioned internal identities in active source except explicit V1
communications; no compatibility aliases; current producer/consumer round trips,
retired-format rejection, mandatory raw-page replay, affected tests/lint/types, and
one consolidated code/ML review. Record any real-data reconstruction separately.
TradingFlow changes remain its owner's task while its checkout is concurrently edited.
Do not merge this cleanup into main as if SPY outperformance had been established.

Queued checkpoint: **Investment dataset policy and source admission**.
October 3 clarification: the prior timestamp question was premature and is withdrawn.
No row-level 63/252 dataset audit established a missing-timestamp count. The new
projection's ability to reject unknown clocks is a software rule, not evidence that
the actual dataset has that defect. Repeated collection and completed swing research
must not be described as failed on that basis. The current feature acceptance audit
explicitly says technical fixed-horizon research does not require historical news
receipt proof; it uses independently usable matured labels and causal features.
Next, inspect the actual permitted source/feature/label metadata, distinguish market
or event time, retrieval time, historical availability and research label maturity,
and report exact affected fields/rows if a gap exists. Reuse established applicable
research rules; do not request blanket timestamp assumptions without measured evidence.
The user has not approved any new assumption, and this clarification changes no code,
source clock, trained artifact, admission rule or historical research result.

Concrete remaining implementation: independent feature-only/source admission and
immutable horizon-specific dataset publication; frozen investment training requests,
purged folds and artifact publication; then separately verified regressor promotion,
live feature parity, horizon-specific monitoring and forecast API. Existing swing
research artifacts and the new projection do not implement those stages.
Reuse the public return estimators/validation primitives where their contracts apply;
do not call swing-only research orchestration or relabel its non-serving artifacts.
Preserve the six-spec swing budget, exclusion rules, permitted initial-fit calendar and
sealed-evidence boundary. An investment experiment needs its own frozen budget/splits.

Other unresolved evidence: part (4b)'s raw per-lot event/cost portfolio curve and
locked-test threshold comparison; current production membership/catalyst authorities;
an approved promoted model. Raw Alpaca collection and provider credentials do not
provide these approvals or ownership/payment facts. No profitable model or full
training/API readiness is asserted. TradingFlow remains untouched.

Part (4a) completed in `4ba96f6` and pushed on main (October 3).
Reports now count matured complete decision sessions / horizon, retain verified
zero-selection exposure, and include unselected scored outcomes in equal-sector
fixed-horizon ranking. HH N-1 / NW 2N uncertainty preserves exchange-session gaps;
benchmark excess is divided by actual holding duration. Internal old shapes fail.
The invalid overlapping compounding is removed; portfolio return/drawdown are null
with explicit missing accounting status. Drift cannot authorize predictions from
these reports; measured severe losses or reversed ranks remain visible.
Verification: 390 component tests passed (statistics, performance, population, drift,
calendar, commands, registration, prediction/API, CLI, architecture and continuity).
Ruff clean on all nine changed/new Python files; strict mypy clean on three source
modules. Consolidated code/ML review passed. No full release suite, model fit, sealed
test or TradingFlow operation. Drift policy pin is
`09069c085acf87622987434a271e2f4905447696283105e5a969b01fb18527e1`.
Part (4b) remains pending raw per-lot corporate-event/cost evidence and a governed
curve adapter; adjusted monitoring prices cannot satisfy that contract.

Completed checkpoint: **Investment 63/252 target contract and shared-kernel projection**.
The approved open-ended investment cohort needs separate 63- and 252-XNYS-session
forecasts. Freeze marked holding value at the horizon, not an enforced liquidation:
stock gross return, stock net return after the explicit 20-bps prepaid research-cost
assumption, and net excess against gross SPY/QQQ/point-in-time sector holding values.
This is a new investment research choice, not inherited swing target approval. No
allocation/risk policy, production admission, profitable model or sell deadline.

Scope: add investment contracts and projection only. Reuse the unchanged accounting
kernel's existing arbitrary-length SessionSnapshots for raw-share holding specifications.
Require exact next-open entry and all 63/252 consecutive closes, unique matching
benchmark identities/intervals, consistent policy/source binding, and no ordinary
ExecutionEvent. Select snapshot[N-1]; never use the kernel's day-ten aggregate label
clock or swing target/settlement/ordinary-sale adapters. Derive latest required source
availability from horizon snapshots, keeping unknown clocks/valuations unavailable.
Corporate entitlements, payment timing, residual claims and valuation remain exclusively
owned by the shared kernel; do not duplicate their arithmetic.

Preserve every current swing implementation hash. Archived implementation snapshots
are provenance, not an executable compatibility path: changing pinned accounting files
would invalidate saved replay acceptance. The additive projection avoids that migration.
Unknown values, missing required clocks or interval/source mismatch cannot yield an
eligible training label. Research-only results cannot authorize production serving.
Exit gates: 63/252 endpoints with early closes/holidays; final clock later than day ten;
matched benchmarks; retained corporate distribution; unknown residual claim; no future
payment effect; ordinary-sale refusal; source/hash binding; unchanged pinned swing files;
targeted tests, dependency checks, Ruff/strict mypy and one code/ML review.
Failure behavior: explicit unavailable target with reasons for missing valuation/clocks;
invalid identities/intervals raise. No data collection, real fit, sealed source read,
artifact rewriting or TradingFlow changes. Dataset publication/training/admission/API
remain separate subsequent steps; this checkpoint alone is not training readiness.

Part (6) completed in `b36778f` and pushed on main (October 3).
Candidate, evaluation, model card and manifest now agree on final-fit decision end
and the maximum label availability across selection and final-access populations.
Promotion requires that clock before promotion; serving compares model and bundle.
Replay requires the unique typed row decision time strictly after label availability,
selected_for_policy, and actual XNYS opens/closes. The public v4 contract adds nullable
metadata; the producer fixture is regenerated and documented. TradingFlow was not
edited or built; acknowledgement of the optional field remains its developer's work.
Verification: 57 API/service/snapshot/continuity tests; 246 architecture/dependency/
registration/intent tests; nine replay tests; 39 inference tests; 31 synthetic trainer
tests (one optional memory test deselected).
Ruff and strict mypy passed on all eight changed source modules; one consolidated
code/ML review found no actionable issue. No real fit, source read, promotion or
deployment. The opt-in realistic memory benchmark and full release suite were not run.

Completed checkpoint: **Part (4a): overlap-aware monitoring and unavailable portfolio evidence**.
Problem: reports treat overlapping trades as sequential full-capital investments,
count daily decisions as independent holding periods, and omit unselected scored
stocks from rank evaluation. Their current drawdown must not authorize predictions.
Scope: use verified committed population and as-of outcomes; count complete matured
decision sessions (including scored zero-selection sessions) / horizon; keep a separate
minimum matured-trade count. Include all scored fixed-horizon outcomes in within-sector
Spearman correlations (at least five varied observations), equal-sector daily averages.
Persist rank usable/unavailable coverage and rank-specific effective periods. Use HH
N-1 equal-weight covariance, NW 2N Bartlett fallback; exchange-session gaps remain gaps.
For each benchmark calculate sum excess / sum actual holding sessions, with matching
ratio-estimator uncertainty; policy thresholds become explicitly per-session units
(-0.0001 warning, -0.0005 severe). Retire raw independent-group fields throughout
internal reports/drift. Internal new fields are required; old reports fail parsing.

Concrete accounting conflict with the older plan: stored maturation bars are adjusted,
whereas the admitted event ledger needs raw-share entitlements/payment/cash evidence;
current per-outcome execution costs also exceed the ledger's uniform-cost interface.
Last-close sensitivity fills are diagnostics, not admitted daily holdings marks.
Therefore remove overlapping sequential compounding; cumulative portfolio return and
drawdown are null with portfolio_curve_status=unavailable_accounting_evidence. Drift
must never become actionable from these reports. No replacement ledger, artificial
marks, accounting-policy weakening, protected-file edit or sealed-test access.
Part (4b), admitted portfolio curve, remains pending a separate evidence interface
binding every selected intent, raw holding/event evidence and per-lot costs, plus
locked-test threshold comparison. This conflict does not block statistics software.

Exit gates: hand-computed overlap/HAC cases, gaps, zero-selection and all-abstained
sessions, unselected rank inclusion, sector aggregation, wrong-signed rank, missing
fixed paths, unequal durations, consumed outcome hash binding, required curve status,
and otherwise healthy evidence unable to authorize output without portfolio evidence.
Run affected performance/drift/registration/CLI tests, Ruff and strict mypy; one
consolidated code/ML review. Rollback/failure behavior: incomplete statistics remain
unavailable; incomplete portfolio evidence always refuses actionable authorization.
TradingFlow remains untouched. No model fits, real-data activation or profit claims.

October 3 continuation freeze (current user instruction supersedes the earlier
activation-first ordering): complete pending software while real publication remains
environment_pending. TradingFlow is undergoing separate development: do not edit its
files, branches, builds or processes. Its prior migration receipt is historical.

Completed checkpoint: **Part (6): verified replay information boundary**.
Problem: replay expects retired signal names and substitutes a training decision date
for the time when all labels influencing the model became known. This can reject valid
v4 snapshots or admit a historical simulation using future information.
Scope: derive candidate decision-end and maximum label-availability timestamps across
fit, calibration, threshold-selection and promotion-test populations; bind them through
verified promotion into ModelInfo and replay. Replay uses the row decision timestamp,
selected_for_policy, and actual XNYS open/close times, including early closes. Keep the
public prediction v4 contract and pinned historical evidence unchanged; additions are
explicitly documented. No model fit, promotion, deployment or TradingFlow edits.
Invariants: the information boundary is strictly earlier than promotion and replay
decision; required metadata is hash-bound and agrees across candidate and bundle.
Exit gates: missing/mismatched/future metadata refused; equality refused; row time wins
over request time; selected positive_setup accepted; early-close and holiday fixtures;
affected candidate/promotion/serving/replay/API contract tests, Ruff and strict mypy.
Rollback: missing or old metadata refuses replay/admission, never infers an earlier
boundary from a date. Preserve old immutable artifacts; no compatibility fabrication.
One consolidated final code/ML review follows tests. Subsequent software steps are
part (4) monitoring statistics/curve/rank, demonstrated part (5) scale gaps, then separate
63/252 investment contracts, training and APIs. Real source/promotion evidence cannot
be replaced by passing software tests.

October 3 operational receipt: provider access is verified with bounded read-only
requests (Alpaca SIP daily bars/news, Finviz Elite export, SEC submissions: HTTP 200
with expected response shape). Finviz credentials are available under FINVIZ_API_KEY;
the Python production setting is FINVIZ_ELITE_AUTH. No secret value was printed or
persisted by this check, and no persistent credential configuration was changed.

The Alpaca incremental archive's prior status was paused_memory. A guarded one-unit
run succeeded; a subsequent run capped at 250 units fetched all 209 remaining units.
`data/raw/swing_incremental_alpaca/status.json` now reports complete through 2026-10-02,
zero pending units, zero failed units and zero integrity failures. The command was
`python -m market_predictor.swing.datasets.alpaca_incremental --config
configs/swing_incremental_collection.toml --through 2026-10-02 --max-units 250`.
Request pin: `40853beab759eded1326fca9b8faa5b22831119e64f56cd4714edc35d6362ab1`.
Existing archives were reused and preserved; the shared lease and original memory
guards remained enabled. This is source acquisition, not serving admission:
active_membership_authority, issuer_attribution and training_ready remain false;
retrieved-now news revisions do not prove historical first availability. No sealed SEC
files, model training/promotion, runtime deployment or serving activation was performed.

Credentials are not a continuation blocker. Next preparation must establish independently
observed current memberships and issuer/catalyst authority, then canonicalize/audit the
needed stock/benchmark inputs with truthful availability clocks and saved pins. A
current snapshot cannot be backdated to a prior decision. The publisher and consumer
software remain closed; no permissive admission or source-family change is needed.


Prior checkpoint: **Nightly live-input publisher and API v4 consumer** (software complete).
Monitoring and publisher software is verified and pushed on main. The reviewed
API v4 consumer integration is also committed and pushed on TradingFlow main. The current checkpoint
remains environment_pending for approved live inputs and promoted-release registration;
no further publisher or consumer implementation is planned absent a concrete defect.
Nightly live-input publisher freeze (September 28, user requested implementation):
publish prepared, independently pinned production canonical stock bars, benchmark bars,
point-in-time membership and a production catalyst decision authority into the existing
reader format. This is the canonical nightly writer, not a provider collector. A strict
request binds absolute input paths, data/manifest hashes, strategy hash and observation
cutoff. Reuse canonical artifact/authority verification and the shared live feature
builder; derive source watermarks from verified rows, never operator-supplied clocks.
Acquire the shared heavy-job lease before any input load and the monitoring lease for
publication. Bound aggregate file bytes, uncompressed parquet bytes and rows, with 4 GiB
memory guards. Stage an immutable generation (retaining any hash-bound external catalyst
dependency), validate it through the production
reader and feature builder, move the immutable generation, then atomically replace the
active pointer last. Errors leave the previous pointer intact; retain old generations.
Identical request retries verify and reuse the current generation; older observation
cutoffs cannot roll it back. Actual generation/activation times cannot be backdated.
Readers reject future activation even on cached generations. CLI lives only on the
production surface. No model training/promotion, provider transfer, scheduled automation,
sealed-source read, deployment or order work is authorized by this checkpoint.
Exit checks: CLI/reader round trip, parity through shared builder, identical retries,
future activation/source clocks, changed pins, research-only authority, limits, path
escapes, partial publication and reader cache invalidation; Ruff/strict mypy, component
tests, one consolidated review. Real-data registration is environment_pending until
verified current source paths/pins and a promoted release are available.

TradingFlow migration is newly authorized by the user's current request, superseding
the earlier separate-developer-only boundary for this narrow API task. Preserve its
existing dirty work. API v4 parser/fixture/reason display only, advisory isolation
unchanged; independent plan/code review and fresh focused C# build/tests required.

Publisher software is locally implemented in `ebe5dfb`, with final metadata bounds
in `cb70f81`. `publish-swing-live-inputs` is the production-only entry point. It verifies
independent input pins and production admission, derives causal source watermarks,
validates through the existing reader/shared feature builder, and activates an immutable
generation by atomic pointer replacement. Retries verify existing evidence before reuse.
No real nightly publication or session registration has been performed.

Consolidated review fixes: aggregate byte/row bounds include the independently pinned
external canonical-decision artifact; retries deeply verify copied catalyst evidence;
cached readers reject future activation; JSON/TOML metadata is capped at 1 MiB before
parsing. The generation embeds bars, membership and catalyst payloads, but a catalyst
authority may retain its original absolute, hash-pinned canonical-decision dependency.
That dependency must remain available; the generation is not universally portable.

Verification tier: component. Publisher/live-feature/registration/service/API/CLI and
package/architecture tests passed 333 tests, with one opt-in memory benchmark skipped.
Swing API and continuity checks passed nine tests. After the metadata fix, all 17
publisher tests passed; Ruff and strict mypy were clean on the affected files/modules.
Positive publisher fixtures stub the expensive feature-builder boundary; the real
builder rejects inadequate membership, and the live-feature suite verifies mathematical
batch/live parity. This is software verification, not real-source admission or promotion.
No full suite, training, collection, deployment, sealed-data read or live broker call ran.

Main integration receipt (September 28): the user explicitly authorized merging
both projects into main and pushing to their configured GitHub remotes. Market
Predictor main fast-forwarded to `bb88f75` and was pushed, including publisher commits
`ebe5dfb`/`cb70f81` and all monitoring commits. TradingFlow main includes
`e3b6734` on its remote main. On September 29 the user renewed publication approval
after the exact payload/destination questions. Both pushes succeeded: Market Predictor
through `8d6a9c6` and TradingFlow through `e3b6734`. Publication blockers are resolved;
the original TradingFlow dirty checkout and running app remain unchanged.

TradingFlow's predictor integration was isolated onto committed main, including the
required evidence records, signal display and Web/Android projections. Only
`contract_version = market_predictor.prediction.v4` is accepted; all five fixture
outcomes, null scores and distinct sector_peer_floor labels are preserved. Advisory
output remains independent of execution scoring. The isolated main checkout passed
84 freshly built focused tests and an Android build with zero warnings/errors.
Independent plan and final isolation review found no remaining actionable findings.

TradingFlow main was built in
`C:/Users/manis/Documents/Codex/2026-09-28/c/work/trading-flow-main`. Its original
`C:/project/trading_flow` checkout remains on unified-swing-product with all unrelated
uncommitted work and its running process preserved. This is a source merge, not a
runtime deployment. The earlier patch under the chat outputs is historical preparation;
the committed main implementation is now authoritative. Do not reset the dirty checkout
or apply its old patch onto main. Integrate its remaining local work separately.

Real-data activation remains `environment_pending`: no active generation was found at
`data/live/edge_rebuild/swing/active_generation.json`; approved current source paths/pins
and a promoted release have not been supplied/verified. No real nightly publication,
session registration, model promotion, broker call or deployment was performed.

Verification policy update (September 20): both repositories' `AGENTS.md` now use
targeted code checks, affected component integration/integrity/causality checks,
and full suites plus applicable expensive replay at release checkpoints only.
This supersedes older blanket full-suite checkpoint gates in this document, not
their recorded results or model admission rules. No test inventory/deletion is
requested. This update is documentation-only; use continuity checks and diff checks.
Keep C# for the desk and Python for ML. C# ownership of shared provider ingestion
is a recommendation for the next design discussion, not a completed or authorized
collector migration; the current Python collector and wire contracts are unchanged.
Policy checkpoint: Market Predictor `bbbfe46` is pushed; TradingFlow `5217fad` is
committed locally only. Two continuity checks and scoped diff/reference review
passed. Independent policy review closed with no actionable findings; no full
suite, build, model replay or test-count audit was run for this prose-only update.

Completed checkpoint: **Shared evidence consumption and swing feature preparation**.
Implementation: Market Predictor `a6c1294` pushed; TradingFlow `8525f80` local-only.
September 20 continuation: the user authorizes parallel implementation, with
independent plan, design and code reviewers. Freeze each module before editing;
integrate and commit independently owned modules sequentially. This is a component
checkpoint, not a release or permission to repeat the full suite.

Bounded workstreams and exit gates:

- TradingFlow: trusted-plan discovery, exact publication verification, idempotent
  raw receipt import and durable acknowledgement. Verify real Python/C# fixture
  parity, tamper rejection, incomplete attempts, duplicates and restart recovery.
  Existing receipt wire bytes, desk feeds and collector ownership stay unchanged.
- Market Predictor: inventory distinct relationship/reaction feature gaps and
  freeze exact module/column/source scope before implementation. Verify causal
  boundaries, unavailable-versus-zero semantics and batch/live parity for changed
  transformations. Readiness code alone does not establish historical coverage.
- Integration: consolidate independent findings, run targeted component checks,
  preserve unrelated files and protected evidence, and record each partial result.
  Failure leaves data/model admission false; rollback is a scoped Git revert, never
  mutation of original receipts or historical model pins.

Downstream order is qualified issuer-reaction publication/acceptance, its two
remaining specifications trained sequentially, funded policy evaluation, then new
prospective evidence. No re-fitting the two completed baselines, widening the six
specification budget, unseen-test claim or promotion from software tests.
Investment forecast targets: on September 25 the user approved both 63 and 252
exchange sessions as separate targets; allocation and risk budgets remain undecided. Unrelated
shared-data and swing implementation proceeds without waiting for that decision.
Plan review approved this boundary; detailed module design review precedes edits.

Frozen Python module slice (independent design approved):
`swing/contracts/return_feature_profiles.py` and
`swing/features/return_relationships.py`, with focused profile tests. Append four
ordered, raw relationships to the unchanged 120 baseline inputs: close[t-21] /
close[t-126] - 1; close[t-21] / close[t-252] - 1; current volume divided by the
mean of the preceding 20 sessions times stock-minus-SPY same-session open/close
return; and five-session stock-minus-SPY return times SPY's 60-session return.
Use full exchange-session indexing, not surviving-row offsets; the 252-lag input
needs 253 session positions. Missing history stays null with a reason. Bind
security, feed, adjustment/vintage and consumed availability clocks. One transform
serves batch/current-decision calls with unchanged baseline values and population.
This closes only transformation/local-input checks, not historical publication,
training or serving acceptance. Qualified issuer/SEC reaction needs a separate
content/availability-qualified transform; no all-null placeholder counts as done.

Frozen C# design: an explicit local import operation trusts a configured publication
root plus an independently pinned plan (a hash alone does not authenticate a
mutable root). Verify the complete existing attempt chain under the producer lock
before importing received results. An atomically published, byte-preserving inbox
bundle is its acknowledgement; receipt existence without a committed result is
not importable. Restart verifies prior bundles rather than overwriting them.
Collection completeness stays separate from imported-page count. No provider,
normalized catalog, model or trading admission side effects are permitted.

Completed execution checkpoint: **Publish historical relationship features and fit their two learners**.
Plan/design freeze is approved before implementation. Reuse the immutable
59-month, 586,305-row initial-fit baseline; publish a separate derivative with
unchanged baseline values, labels, eligibility and population plus four columns.
Separate historical provenance from newly executed source/code verification: old
readiness pins are stale, so produce fresh evidence without rewriting old receipts
or weakening their verifiers. Adapt bounded source metadata readers, then publish
and verify monthly artifacts, then use the unchanged two learners/settings and
folds with the named 124-column profile. Parallel source/publication and training
integration ownership must have an agreed interface; actual heavy runs serialize.
Exit gates include exact parent parity, causal clocks, security/basis lineage,
tamper/restart rejection, fresh readiness and sequential fits within memory limits.
Any failure leaves the candidate unavailable; no silent exclusion or source repair.
Later validation/test materialization, qualified issuer-reaction inputs, funded
policy evaluation and prospective promotion remain distinct unfinished work.

Software checkpoint `be474b5` is pushed: source publication and training integration
were implemented in parallel and passed 22 source, 104 admission/integration, and
24 CLI/continuity checks; changed-file Ruff and strict mypy on 15 modules passed.
Independent plan/design/code findings closed. Full-history execution then exposed
one concrete adapter defect before output: raw collection metadata is transitively
pinned by the panel request, not necessarily directly in its source-file map.
Reopen only this metadata reader and faithful regression fixtures. Preserve inherited
byte pins, distinct pre/post semantic identities, feed and authority checks, and
conflict rejection. Fix `ed3c125` is pushed: real metadata-only verification, 69
focused source/regression/full-chain tests, Ruff and strict mypy pass; independent
code review closed. Real publication completed 586,305 rows / 59 months. Independent
replay then exposed an all-null clock dtype loss from `.to_numpy()` assignment in
the shared transform (UTC nanoseconds versus naive seconds). Reopen only dtype
preservation and its serialization/replay tests; keep exact verification, population
and formulas unchanged. Preserve the non-admitted publication, commit the fix, and
rebuild separately. UTC dtype fix `82842e0` is pushed with 94 passing focused tests,
Ruff, strict mypy and independent review. Fresh publication and full independent
row replay now pass for all 586,305 decisions / 59 months, with unchanged original
outcomes and exact source-recomputed additions. Readiness passed with unchanged
eligibility/supervision. Independently reviewed frozen configs `a8be7cb` are pushed;
all 16 validation fits and two final models completed sequentially. Each final fit
used 314,167 rows; peak working memory was 2.471607 GiB. Exact artifact pins are in
the handoff. Independent review verified all 18 model and 16 prediction payloads;
old baseline result files remain OS access-denied, so a fresh paired comparison is
unverified. Ranking is weak and mean error exceeds zero-excess prediction. No funded
SPY outperformance, outer validation, test or promotion is claimed.

Research checkpoint: **Qualify issuer-news and SEC reaction inputs** (paused behind
monitoring correctness and the remaining retirement/replay gates).
This starts with source/feature design, not another unrestricted training search.
Inventory existing initial-fit issuer event content, SEC acceptance times, coverage,
identity and completed price/volume reaction windows. Freeze exact columns, clocks,
known-empty/unknown semantics and batch/live transformation ownership before coding.
Keep the same 586,305 decisions, original labels, weights, costs, splits and approved
exclusions. Divide source/attribution and transform/consumer work into disjoint
implementation slices with independent plan, design/ML and code review. Heavy
materialization, verification and the last two fits remain sequential. Missing SEC
or full-content evidence must be reported, never replaced by aggregate counts or
invented events. Exit: a source-backed contract, historical coverage audit, full
feature acceptance tests, immutable publication/replay and only then fresh fitting.

Next bounded delivery: **Saved news and filing content inventory**. Reuse the early
and later initial-fit issuer-news authorities plus corrected issuer-news sources;
resolve their pinned original records before classifying retained content. Read
bounded batches under the configured memory/lease rules, restricted to July 9, 2019
through the existing May 28, 2024 initial-fit cutoff and unchanged cohort identities.
Report security/ticker, year, source, source paths/hashes, record and distinct-version
counts, publication/update/availability bounds, attribution status, and counts of
provider article bodies, summary/headline fallbacks, SEC form metadata and unknown
content. Nonempty text or text differing from a headline alone does not prove a full
article. Unknown coverage is not zero news. Preserve provenance for every category;
the report establishes inventory, not semantic qualification or source admission.
Exit checks: pinned-input validation, revision/duplicate handling, cutoff exclusion,
wrong-issuer rejection and stable counts under different batch sizes. This delivery
does not download sources, change model inputs or fit models. The root baseline
manifest is now readable with its original hash; 18 nested baseline manifests are
still unreadable, which does not block this independent delivery.

Bounded implementation freeze for original-content verification: one existing
Alpaca query chunk at a time, with explicit original event-artifact and sidecar
hashes, query security/ticker, and start/cutoff. Follow only sidecar-declared raw
pages; verify byte hashes, envelope hashes, request/chunk identity and pagination.
Resolve each canonical `(event_id, raw_sha256)` against the provider article object
using the original serialization, preserving every matching page/index occurrence.
Verify issuer-query symbol, title/source/URL, publication/update clocks and canonical
content fallback parity. Label retained provider fields, not full-article completeness
or event meaning. Reject malformed/conflicting provenance; report records excluded by
publication or version-availability cutoff separately. Keep historical proxy semantics
and source-query identities separate from target-cohort attribution. Tests cover hash
tampering, duplicates/revisions, cutoff poison, wrong symbols and body/fallback states.
No final feature registration or training is authorized by this component. The
qualified reaction join additionally requires independent semantic qualification and
a frozen event-to-decision selection rule; raw-content verification cannot supply them.

Consolidated design/ML and code review (September 22) amended this freeze before
closure. Bind each chunk to exactly one self-hashed `_request.json` work unit and
recompute its chunk ID; require `include_content` and the publication-proxy policy,
so headline-only means the provider returned no body rather than an unrequested one.
Canonical availability must equal `max(published, updated)` exactly. Reproduce the
producer's acceptance filter and retained revision, including its first-seen page;
count discarded raw items by reason instead of rejecting the chunk. Restrict bounds
to the existing initial-fit issuer-news constants and apply the caller's approved
90% memory guard inside the reader. Record rows published in the window as
`included` or `version_after_cutoff` (needed for per-year aggregation); outside rows
are counts only. Paths are root-relative; whitespace-only and non-text provider
fields are labelled `blank`/`non_text`, never counted as a provider body.
Implementation pins are an explicit semantic list because the relationship closure
pinner is itself hash-bound to closed evidence. No C# consumer exists.

Component closure: `7a9334c` is pushed. Final verification: 197 targeted inventory,
CLI-surface, command/architecture-boundary and continuity tests passed in 30.49
seconds with RuntimeWarnings as errors (JUnit `.test-tmp/content-inventory-final.xml`);
changed-file Ruff and strict mypy pass. Read-only probes with the approved guard
verified five real chunks across all three archives, including the largest
initial-fit chunks (TSLA 28 pages / 1,350 rows; TSLA 130 pages / 6,456 rows in 13-18
seconds). No publication, download, cohort pass or fit ran. The design/ML and
code/.NET reviewers verified closure of every finding; three P3 closure notes
(out-of-range discarded clock, non-text fields, `chosen_field` wording) were fixed.
This closes only single-chunk verification, not the delivery's cohort inventory.

Completed slice (`177f6f3`; design reviewed and amended, September 23):
**initial-fit cohort news content inventory**. Problem: content categories exist per
query chunk, but feature design needs per-security, per-year coverage in cohort
identities, with unknown coverage and unattributable queries kept distinct from no news.

- Inputs: one hash-pinned JSON config, read only after the shared lease is held. It
  pins `configs/swing_initial_fit_monthly_news.json` (whose `sources` block is verified
  by the existing `issuer_news_preparation._source`, unchanged), the source-proven
  identity manifest (pins `identity_bridge.parquet`) and the approved population audit
  (`load_swing_research_cohort(..., source_root=root)`). No download or archive scan.
- Work units: only the three pinned ledgers (early 4,066, later 1,563, corrections 105).
  Derived observed chunks use `_source_children.json` pins re-anchored under the
  derivation's declared `repository_root` (the run fails closed elsewhere; outputs never
  contain absolute paths) at the exact `<collection>/events/<chunk>.parquet` path. The 35
  derivation-unavailable chunks (raw `observed`, reason recorded) and the 92 corrections
  chunks use artifact pins from their pinned raw collection manifests; their sidecars
  must declare that artifact and the root request hash. Their evidence levels are
  `derivation_unavailable_raw_pinned` and `artifact_pinned_sidecar_observed`, reported in
  separate columns. The 32 `observed_empty` chunks are verified from
  their saved page (envelope, request/chunk binding, terminal pagination, zero admitted
  items) as `pages_verified_empty`.
- Ledger parity for every chunk: reader canonical rows = `original_row_count`, reader
  window = `original_requested_*`, derived `row_count` = included rows, and pages,
  provider, accepted, duplicate, invalid-timestamp (clock + title), outside-window and
  symbol-mismatch counts equal the ledger's producer statistics. Corrections compare
  its original ledger directly. Coverage segments use the clipped `requested_*` window.
- Identity reuses `universe/issuer_news_identity.py` unchanged, called per chunk
  (frame cap 250,000 rows): coverage via `map_news_coverage`, articles via
  `map_news_relations` at availability time. Attributed means mapped to a cohort target,
  or an unmapped query ID that already is a cohort ID (`identity_equal`, e.g.
  GOOG/GOOGL, PARA, VTRS; 128 units). Bridged non-cohort targets and the 1,150 units
  under unbridged legacy IDs are reported with totals by archive and ID prefix, never
  assigned; unmapped rows record whether the bridge lacks the query ID or the
  availability falls outside its span.
- Outputs below a new `data/research` child: `records/` parts (every recorded article
  with archive, query and cohort identity, translation status, evidence level and New
  York publication year; about 50k-row groups); `coverage.parquet`; `security_years.parquet`
  (each retained cohort security x New York calendar year in the window: query-returned
  distinct stories on `(source family, provider_story_id)` with included before
  after-cutoff precedence, included category counts, and interval-union covered time
  split by evidence level and known-empty, plus unknown `no_proven_query` time; hours
  before a bridge span opens are unknown, so 2019 rows carry at least four hours).
  Records, coverage and unit summaries carry `derivation_status`, the provider symbol and
  `query_identity_resolution` (query identity only). `_manifest.json` holds every pin,
  implementation hash, unattributed totals in segments, days and distinct query IDs,
  `known_empty_scope` (a provider-symbol query returned nothing, not proof of no issuer
  news) and `attribution_status: not_established` (counts are stories returned by the
  query, not issuer relevance).
- Resumable leased run with the 90% guard, following `research/swing_return_training.py`:
  immutable `_request.json`, atomically replaced `_checkpoint.json` listing completed
  chunk-batch parts with hashes, resume only with an independently supplied checkpoint
  SHA256, final manifest last. An empty checkpoint follows the request and the final
  manifest is replaced atomically. Estimated 20-30+ minutes; measured bounded sample:
  122 real units in 14.5 seconds at about 190 MB process memory.
- Scoped extension of the closed reader (`7a9334c`), required by this slice's parity and
  empty-evidence checks: expose the page count and add a verified-empty entry point
  reusing its request and page verification. Existing results do not change.
- Exit tests build archives through the real producer and derivation code: every status
  and evidence level, ledger/reader parity failures, a unit crossing the cutoff,
  identity-equal attribution, unbridged legacy IDs, corrected-security isolation,
  unreached cohort securities, conservation of every work unit, overlapping coverage
  rejection, story deduplication precedence, New Year boundary, part-size and chunk-order
  stability, resume/tamper of every pin, lease/memory/atomicity.
- Out of scope: SEC form metadata and missing filing/exhibit documents (next slice),
  semantic content qualification, decision joins, features and fitting.

Slice closure: implementation `177f6f3` is pushed. Independent design/ML and
code/.NET reviewers reviewed the design and the implementation; every supported
finding is closed and both verified closure. The declined helper deduplication is
recorded at `_atomic_json` (27 local copies exist repository-wide; the reference is
hash-bound and imports estimators). Final verification: 241 targeted inventory, cohort,
CLI, boundary and continuity tests in 70.65 seconds (JUnit
`.test-tmp/cohort-inventory-final.xml`); Ruff and strict mypy on nine changed files.

Real run (leased, exit 0, 11:08-11:57 on September 23; first 17 minutes re-verified the
pinned sources): `data/research/swing_initial_fit_issuer_content_inventory`, manifest
`ab5482a01575a7a203f0e070c7f2dc93c413930f8296d448cf4aca5122b3ed21`, request
`a145341955004edfbd2a3e4a40aea52014a75707981ed9e8ae2a12fcea29b3c9`, checkpoint
`89580031a718959ab07f1d855ca5c2a184f88f86371e8e4c691fe2a8a8bfef12`, log
`data/runtime/swing_initial_fit_issuer_content_inventory.log`. Memory was not measured
continuously; the 90% guard ran throughout and never stopped the run.

- All 5,734 ledger units verified: early 4,018 sidecar-pinned, 29 derivation-unavailable,
  19 verified-empty; later 1,557 and 6; corrections 92 and 13. 469,668 article records.
- Independent reconciliation: included rows of derivation-observed chunks equal the
  derivations' own source events exactly (148,784 early; 319,787 later); the 35
  derivation-unavailable chunks add 356 + 70 rows. Bridged records outside their
  coverage segment: 0.
- Cohort coverage: 69.6% of cohort security-time has a verified query; 30.4% is
  unknown `no_proven_query`. 145 of 586 retained securities have no provable query;
  141 unbridged legacy query IDs (`sp500-historical`, `cusip`, some `cik:...:ticker`)
  hold 60,590 unattributed records. This identity gap, not missing downloads, is the
  largest coverage limit for reaction features.
- 404,467 distinct attributed stories: 62.35% carry a provider body field, 37.65% are
  headline-only, none summary-only. Body share rises from 51.7% (2019) to about 65%
  (2022-2024). Median 107 stories per security per observed year (10th percentile 51).
  These are query-returned stories, not issuer relevance or qualified content.

The unbridged legacy identity gap above was closed by the next slice; the user chose on
September 23 to prove those identities before content qualification.

September 23-24 slice (`closed`; implementation `4844b3f`, design reviewed September 23,
proofs and inventory rerun published September 24):
**legacy issuer-query identity proofs**. The user decided on September 23 to prove
these identities before content qualification. Problem: 141 legacy query IDs (113
`sp500-historical`, 26 `cusip`, 2 `cik:...:ticker`) hold 60,590 initial-fit records that
the CIK-only bridge cannot translate; 145 of 586 cohort securities have no proven query.

- Measured evidence (read-only): legacy IDs are minted as `sha256("ticker|company|spell
  start")` (`membership_history.py`) and cohort `sp500-historical` IDs as
  `json_sha256({company, ticker})` (`membership_authority.py`). The canonical company of
  the S&P event authority that the cohort membership authority pins through
  `parent_lineage.event_authority_sha256` reproduces both IDs for 110 of 113 unbridged
  `sp500-historical` IDs. OGN, PENN and POOL reproduce from the addition name (legacy) and
  the deletion name (cohort) of one spell with identical boundaries. BRK-B/BF-B legacy
  IDs embed the cohort CIK. The two archives queried different legacy files: early used
  the verified v2 file, later the v1 file (their raw requests pin 0e222a23... and
  67a8edb0...); the 140 shared IDs have identical spans, and v1 alone holds Fiserv
  (`cusip:337738108`, FI from 2019, contradicted by the pinned correction policy and by
  Alpaca transition `45fd1861`, FISV to FI on 2023-06-07).
- Review blocker, accepted: the cohort membership authority is not point-in-time for
  tickers. It carries each security's latest ticker back to 2018 (for example IR for CIK
  1699150 from 2018-05-29, while legacy IR belonged to `cusip:G8994E103`, now TT, until
  2020-03-02). The frozen `sp500_point_in_time_ticker_unique` rule would attribute
  Trane's news to Ingersoll Rand and is withdrawn; no rule attributes by comparing a
  historical ticker with that authority.
- Proof kinds, strongest first; each legacy ID resolves to at most one target:
  (1) `company_ticker_hash_reproduced`: a single-spell `sp500-historical` ID reproduces
  through the real `_security_identity_for_interval(current=empty, aliases=[])` from a
  pinned event company and ticker whose `_historical_security_id` is a target security.
  (2) `sp500_spell_events_reproduced`: the legacy ID reproduces from the addition event
  that opens its spell, the target ID from the deletion event that closes it, and the
  legacy spell and target row have identical boundaries. (3) `cik_equal`: a
  `cik:X:ticker:T` legacy ID whose target `cik:X` carries T wherever they overlap (a
  share-class guard that can only reject: Discovery's bare CIK is the DISCA line). (4)
  `cusip_chain_end_ticker_match` (weaker, kept separable): every ticker change inside the
  chain is exactly one identity-continuous Alpaca transition of that chain (pinned file
  978f87fe..., read directly: the minting parser drops old tickers ending in V as
  when-issued symbols, which is how the legacy build missed Fiserv's FISV-to-FI change),
  no other transition of that chain falls inside it, and exactly one target security
  carries the chain's last ticker at its own final instant inside the chain's last span,
  the only instant where a latest-ticker authority is true: its final row only, open at
  the authority cutoff when the span is open too, or closed where a pinned S&P deletion
  of that ticker documents the row end.
- Rejections are recorded per legacy spell, never guessed: unsupported namespace,
  multiple spells for a hash proof, no candidate, ambiguous candidates, chain contradicts
  transitions, CIK ticker differs, no membership intersection, and
  `corrected_security_uses_corrections_archive_only` (targets in the pinned correction
  policy). Every in-scope spell is either proven or rejected. Scope is every legacy ID
  that is neither a CIK-bridge source nor already a target ID.
- Proof rows: source ID, query ticker, target ID, the legacy spell, the intersection
  with each target row, availability, kind, evidence JSON (event company, action,
  effective time, source URL and document hash; CIK; or transition IDs and the matched
  instant), the earliest date sufficient cited evidence existed (defined per kind in the
  manifest), and a row hash. Availability is
  labelled `retrospective_membership_effective_proxy` (both authorities set availability
  to the effective start); proofs are research evidence only. A proof whose target
  interval overlaps a CIK-bridge row, an identity-equal legacy spell or another proof is
  an error at publication.
- Implementation: pure builder and translation in new `universe/legacy_query_identity.py`.
  `universe/issuer_news_identity.py` stays byte-identical (ten closed artifacts pin it);
  its primitives are imported, and only the relations and coverage loops are
  re-expressed as generic passes, tied to the protected functions by a synthetic unit
  differential and a recorded read-only run over the real CIK bridge, every ledger and
  every saved record. The legacy pass runs only on rows the CIK pass left
  unmapped, with status `legacy_proven`, kind and proof hash columns. A leased immutable
  publisher writes `proofs.parquet`, `rejections.parquet` and `_manifest.json` below
  `data/research`. It verifies the target authority with its canonical loader, the event
  authority with its canonical verifier and raw archive (its hash must equal the target
  authority's lineage), the identity manifest (the CIK bridge, and target-authority and
  correction pins that must equal its own), each derived archive's raw request and its
  membership file, and the transition file.
- Consumer: the cohort inventory module is extended and published from a new config
  file with the proof pin; the completed inventory and its config stay immutable.
  Resolution order is bridged, identity-equal, `proven_legacy_identity` or
  `proven_legacy_non_cohort`, outside bridge span, no proven identity. Records and
  coverage carry the proof kind and hash; security years and totals split covered days
  and stories by attribution basis. The whole inventory is rerun into a new output.
- Exit tests: real minting functions for every ID; tampered company, ticker or start;
  the IR/TT handoff on a latest-ticker target authority; a Fiserv-like chain that
  contradicts its transitions and a corrected target; a merger transition of another
  CUSIP that is not a contradiction; a non-continuous transition (FLT to CPAY) staying
  split; a name present only in an unpinned authority; the spell-event and CIK kinds;
  ambiguity; partial intersections; conflicts; differential parity with both protected
  CIK functions; translation at event time and coverage splits; resolution precedence;
  pins, lineage, lease, immutability and determinism.
- Out of scope: content qualification, features and fitting. The SEC slice follows.

Real runs (leased, exit 0, September 24). Proofs:
`data/research/swing_legacy_query_identity_proofs`, manifest
`84ffb55398f3db016ce9e0af9dea8aa9de54c559f5d9a1959cca2396d92712d3`: 165 legacy IDs
proven (110 company hash, 5 spell events, 22 CIK equality, 28 weaker CUSIP chains) in
184 rows, none partly proven; 5 rejected (CUK, FLT, FBHS/FBIN no candidate; Fiserv
chain contradicts its transitions; EchoStar corrected). Inventory rerun (03:57-04:51):
`data/research/swing_initial_fit_issuer_content_inventory_with_legacy_proofs`, config
`configs/swing_issuer_content_cohort_inventory_with_legacy_proofs.json` (SHA256
`315ae1af...`), manifest `3e4f905e8e6da4d3c0331fb358437633765211a59e94e2f6bba6d1b7962e9bda`,
request `7510280c768608da2ebef841de46b4ec2aee4754f63df17230123cb437b24e32`, checkpoint
`556d4e11e2ec9f3f3780c19826cd8d6a650264e87192a2c635ed529c00e03acb`, log
`data/runtime/swing_initial_fit_issuer_content_inventory_with_legacy_proofs.log`.

- Of the 141 unbridged query IDs, 102 now reach cohort securities, 35 reach securities
  the approved cohort excludes (19 inherited, 16 own exclusions) and 4 are rejected. 59,487 formerly
  unattributed records are legacy-proven (39,449 into cohort securities); the 1,105 left
  unproven all sit under the rejected IDs. Translated included records outside their
  coverage segment: 0.
- Cohort securities with a proven query: 441 to 543 of 586. Whole-window verified query
  time: 69.6% to 80.9%. Measured inside each security's S&P membership in the window
  (read-only check against the target authority), unknown query time falls from 14.4%
  (122,640 security-days) to 0.44% (3,716); 41 of the 43 unreached securities were never
  members in the initial-fit window and 2 members remain unreached, below the 5% bar.
- 443,916 attributed stories (61.91% provider body field, 38.09% headline-only). By
  basis: CIK bridge 380,266, identity-equal 24,201, company hash 35,842, spell events
  881, CIK equality 94, and the weaker CUSIP chains 2,632 (0.98% of window days), which
  stay separable in `security_years.parquet`.
- Legacy-proven attribution starts at the membership effective start, a retrospective
  clock; records carry each proof's evidence-complete date. Research evidence only.

Current slice (`in_progress`; design reviewed and consolidated September 24):
**SEC form inventory and filing-document collection**. Count the SEC filing metadata
already saved for every approved cohort security and New York year, list which filing
documents are saved, and (user decision September 24: collect now, leave nothing for
later) download the documents of 8-Ks carrying item 2.02, 7.01 or 8.01.

- Measured evidence (read-only, September 24): the pinned archive
  `data/external/sec_filings_20190709_20260708_v1` verifies with the canonical
  `load_sec_filing_collection` (624 issuers, 877 saved EDGAR submissions responses).
  All 689,467 canonical events reproduce their `raw_sha256` from the saved raw rows;
  no saved page listing overlapping the window was left unfetched. The events drop
  fields the raw rows keep: 8-K item codes, `primaryDocDescription`, filing `size` and
  XBRL flags. All in-window 8-Ks carry item codes; for cohort securities there are
  22,067 accessions with 2.02 (9,964), 7.01 (7,453) or 8.01 (7,716). Availability is
  acceptance plus five minutes or the next XNYS open for late submissions; EDGAR's
  acceptance clock is genuine UTC (99.68% of non-ownership filings accepted by 17:30 ET
  carry that New York day's filing date). First observation was August 2, 2026, so all SEC timing is
  retrospective research evidence. 585 of 586 cohort securities have an SEC identity
  relation (591 rows, 187 partial; 43 reviewed overrides). Acceptance is often later
  than the event: 6.2% of cohort 2.02, 21.4% of 7.01 and 36.7% of 8.01 filings are
  accepted a day or more after their report date. 91,065 in-window raw rows are forms
  the collector did not request. Saved documents: ten official-document collections
  (51 corporate-action documents) and the identity-evidence store
  `data/raw/sec_identity_evidence_20260802` (56 saved documents in two inventories whose
  61 rows include 5 lookups that found no filing); no earnings release is saved. Whole filings are large (cohort 2.02
  filings total 25.5 GB by EDGAR `size`).
- Byte-frozen: `sources/sec.py`, `sources/http.py` and `sources/official_documents.py`
  are in closed evidence's pinned import closure; they are only imported.
  `catalysts/sec_filings/collection.py` is unpinned and gains the replay.

Inventory (`inspect-sec-form-inventory`):

- Metadata recovery replays the unchanged `SecSource.fetch_cik_filing_history`, one
  issuer at a time, through an archive-backed client that serves each saved body with
  its recorded URL, clock and headers; each issuer's records must equal its canonical
  events by accession identity and `raw_sha256`, so a missing overlapping page, a
  `filingCount` mismatch or a conflicting duplicate fails. `_rows` on the served
  bodies supplies only the extra fields and totals of unrequested forms.
- Scope: events with availability in [2019-07-09 00:00 UTC, 2024-05-28 22:00 UTC],
  the same inclusive `FIRST`/`LAST_INITIAL_FIT_CUTOFF` rule as the news inventory;
  accepted-before-but-available-after events are counted separately. Each event is
  attributed to every cohort security whose pinned SEC identity relation (file equal to
  the identity alignment manifest's pin, semantic hash equal to the collection
  request's via `_relation_sha256`) covers its availability and was itself available
  by then. Share-class duplicates are flagged and never double counted in issuer
  totals. For `cik:` cohort IDs, same-CIK filings outside the relation are reported as
  `cik_identity_outside_relation`, separately from unknown identity; neither is
  attributed. Inside a relation, a zero for a requested form is a verified zero.
- Each filing row carries form, item codes, report date, acceptance and availability
  clocks with the availability rule and `retrospective` label, session position
  (pre-open, intraday, after-close, non-session), identity policy, share-class flag,
  EDGAR size, description, XBRL flags and document status. 8-K timing is defined from
  SEC availability (closure review): calendar days from report date to acceptance; XNYS
  sessions after the report date that opened before availability; whether the report
  date itself was a session that opened before availability; and a three-way label
  `post_report_sessions_precede_sec`, `report_session_may_precede_sec` (SEC data alone
  cannot tell) or `no_post_report_session_before_sec`. An 8-K filed after trading
  already occurred is a late record of an event; its first public time must come from
  other evidence, never from `report_date`. Counts use EDGAR's own form names and item
  codes split by that label; no invented families.
- Document status: official-document collections are verified from their own pinned
  `_request.json` (embedded inventory checked against `inventory_sha256`) with the
  unchanged `verify_official_document_collection`; the identity-evidence store through
  its pinned inventory CSVs and per-file hashes (`saved_without_receipt`). Matching is
  by CIK and accession: `primary_document_saved`, `other_filing_document_saved`,
  `saved_without_receipt` or `not_saved`.
- Outputs: `filings.parquet`, `security_years.parquet` (per security and New York year:
  counts by form and item code, relation-covered, CIK-outside-relation and unknown
  days, document status counts) and `_manifest.json`. The same command in `sealed` mode
  lists the later-window filings (after the initial-fit cutoff to the archive end,
  covering validation and historical test) keeping only the collector's work-list
  columns, one row per accession, with an accession count as its only statistic. One
  guarded loader serves every consumer and refuses a sealed inventory except to the
  document collector; the collection reader refuses a sealed collection, whose manifest
  holds unit counts only. The seal lifts only through a pinned record that qualification
  rules were frozen on initial-fit evidence.

Collector (`collect-sec-filing-documents`, new `catalysts/sec_filings/document_collection.py`):

- Work list: the pinned inventory's cohort 8-K and 8-K/A accessions carrying 2.02,
  7.01 or 8.01 (initial fit), and separately the sealed later-window list into its own
  store. Phase one fetches each filing's EDGAR detail page (`{accession}-index.htm`).
  It is parsed with BeautifulSoup `html.parser` (already a dependency): exactly one
  document table selected by its exact header, links inside that accession's folder
  under any filer's CIK (co-registrants), inline-viewer links resolved to their
  document; a joint filing is one unit keeping every filer CIK; its header (accepted time, filing
  date, period of report, items) must match the inventory row and its primary document
  the saved `primaryDocument`, otherwise a recorded rejection. Retrospective header
  fields (current name, SIC, address) are never point-in-time attributes.
- Phase two fetches the primary document (its text holds the item disclosures) and
  every `EX-99` exhibit of that table; never XBRL, graphics or whole-submission files.
  Each document unit is derived from a verified phase-one receipt and records the index
  body hash and parsed table row; content type is recorded, and PDF or image exhibits
  stay unqualified text. Each accession keeps its own clock; amendments are separate.
- Requests use `SecSource` with a dedicated `SecRequestGovernor` (rate and cooldowns in
  `_request.json`) and `get_bytes_with_metadata(retries=1, allow_redirects=False,
  raise_for_status=False, maximum_body_bytes=16 MiB)`. Outcomes: archived (200 and
  accepted); terminal (404/410, oversize by the client's exact error, index or header
  rejection); stop-and-resume (403/429, recorded, run ends, and a resume refuses to send
  a request before SEC's Retry-After or the configured cooldown, whichever is longer,
  has elapsed); retryable (5xx or connection errors) up to three attempts per unit;
  at most one archived attempt per unit. Receipts keep the final URL, redirect chain,
  retrieval clock and safe headers.
- Storage: immutable shards (a zip of bodies named by accession, sequence and body
  hash, plus a parquet of self-hashed receipt rows), staged then renamed; an atomically
  replaced checkpoint lists shard hashes and is the commit point: a reopen removes shard
  files and staging folders the checkpoint does not name, re-hashes shards, and final
  verification checks every member and receipt hash. An immutable self-hashed `_request.json` binds the
  inventory manifest, work-list hash and governor settings. Heavy-job lease for the
  whole run (initial fit about four hours), memory guard every 500 requests. Retrieval
  time (2026) is first observation, never historical availability; EDGAR filings are
  immutable after acceptance.
- The completion manifest lists, per selected filing and document, archived, failed or
  rejected outcomes; filings outside the selected items stay `not_requested`.

- Exit tests: inventory fixtures written through the real collector path with an
  HTTP-level fake SEC client; replay with tampered, missing, duplicated and extra raw
  rows, a `filingCount` mismatch and an unfetched overlapping page; relation clipping,
  late relation availability, partial relations, a security with none and CIK outside
  relation; share classes; window, inclusive cutoff and late-submission boundaries;
  lag and session position; item parsing; document matching including the evidence
  store and an unverifiable collection; issuer-order stability; sealed mode. Collector:
  real EDGAR detail pages from a small leased pilot (2019 without inline XBRL, 2024
  with viewer links, 8-K/A, several EX-99, none) committed as fixtures; header and
  primary-document mismatches; foreign and malformed links; each outcome state;
  shard tampering; resume; stop on 403/429; no request outside the work list in either
  phase; pins, lease and immutability.
- Out of scope: content qualification, event meaning, features and fitting.

SEC acceptance-clock finding and correction (September 25; design addendum for review):

- The full initial-fit collection (`data/raw/sec_filing_documents_initial_fit_v1`,
  stopped after 18,500 index attempts; checkpoint pins 37 verified shards) rejected
  3,033 of 17,847 fetched detail pages because the page's `Accepted` time (EDGAR's own
  New York clock) differed from the saved submissions `acceptanceDateTime`. Every
  difference is exactly the New York UTC offset (four hours in daylight time, five in
  standard time), and it is per issuer: of 520 issuers seen, 93 differ on every filing
  and 427 on none, uniformly from 2019 to 2024. For those issuers the submissions API
  labels New York wall-clock time as UTC (for example Skyworks' after-close earnings
  8-K accepted 16:04 New York time is saved as 16:04Z). Their canonical acceptance and
  availability are four to five hours too early, a look-ahead risk that the earlier
  filing-date test could not detect. Only the two SEC form inventories consumed this
  archive (no feature publication, training request or model); the SEC decision
  authority built from it inherits the error and must not be consumed uncorrected.
- Correction: a leased immutable clock-convention publication decides each issuer's
  convention from EDGAR detail pages, never from guesses: the stopped collection's
  verified pages plus, for issuers without one, the detail pages of their first and
  last in-window filings. Every compared filing must differ by zero (`utc`) or by
  exactly the New York offset at that instant (`new_york_wall_clock_labeled_utc`);
  any other difference, or a mix, fails the publication. Issuers without a page stay
  explicitly unknown and their filings are excluded from timing-dependent outputs.
- The SEC form inventory consumes the convention pin: corrected acceptance, the raw
  API value and the convention on every row, availability recomputed with the unchanged
  `conservative_sec_daily_swing_availability`, then windows, session position and
  report timing from the corrected clock. Both inventories are republished into new
  outputs; the published ones stay superseded evidence.
- The collector reads the corrected inventory, so detail-page headers match again; it
  gains bounded concurrency (a few workers sharing one governor at or below five
  requests per second) because single-threaded requests measured about 1.2 per second,
  limited by SEC response latency. Receipts stay per unit and order-independent; the
  first 403 or 429 still stops every worker and checkpoints. The stopped collection
  stays immutable evidence and is not resumed.
- Exit tests: both conventions and DST on real pages; an offset that is neither zero nor
  New York; an issuer mixing conventions; an issuer without evidence; availability and
  window changes from the corrected clock (an after-close filing no longer intraday);
  concurrency with deterministic receipts and a 403 stop under several workers.

Consolidated review decisions for the clock correction (September 25; they supersede
the bullets above where they differ):

- Measured, not assumed. EDGAR's own detail pages for HollyFrontier's 10-K, 10-Q and 8-K
  all read New York wall-clock time, matching its saved labels. The reported form-group
  mix was a 10-K accepted at 17:31 and dated that day, which EDGAR allows when
  transmission began by 17:30. Over the saved archive, the page-derived convention
  leaves fewer filings outside EDGAR's 06:00-22:00 New York weekday hours than the
  opposite reading for all 520 page-labelled issuers and for 1,619 issuer-form groups;
  458 groups are uninformative and none is contrary. Genuine out-of-hours acceptances
  exist (119 filings of page-`utc` issuers, mostly Saturday 424B2 filings), so hours are
  comparative evidence, never a hard rule. Eighteen of the 93 affected issuers stopped
  filing before 2024 and the rest file through 2026, so no rule predicts the defect.
  These read-only diagnostics counted clocks over the whole archive, including the later
  window, and read no content; the publication's in-data evidence uses only filings
  dated before the initial-fit cutoff's New York date.
- One convention per issuer, verified in every form group: a resumable, leased page
  collection fetches the detail pages of each issuer's first and last archive filing in
  every form group it has (current reports, periodic reports, ownership forms, other).
  A detail-page header carries no content, and the sealed collection fetches
  later-window pages anyway. A page whose acceptance equals the raw value read as UTC
  classifies `utc`; read as New York wall clock (a DST-ambiguous or nonexistent instant
  raises), `new_york_wall_clock_labeled_utc`. Neither, disagreeing pages within one
  issuer, or an in-data comparison contrary to the pages fails the publication. An
  issuer lacking a classified page in any of its groups stays `unknown`; its filings
  are excluded from timing outputs and counted.
- The publication records per issuer the convention, every page comparison and the
  per-group in-data counts under both readings. The form inventory corrects every row
  from the raw `acceptanceDateTime` string and keeps the raw value and convention.
- Consumers. The addendum's statement that only the two inventories consumed these
  clocks was incomplete. The SEC decision authority (`catalysts/sec_filings/
  decision_authority.py`, the `publish-edge-sec-filing-authority` command and its test)
  consumed the uncorrected events, is superseded by the corrected inventory, is pinned
  by no closed evidence and is deleted. Its artifact
  `data/canonical/sec_filing_authority_20190709_20260708_v1`, SEC-family catalyst events
  and the historical `catalyst_full` SEC columns (`source_count_sec_*`,
  `source_coverage_known_sec_*`, `sec_latest_filing_*`) stay as clock-contaminated
  historical evidence for the affected issuers and cannot feed new work. Neither
  retained model consumes an SEC, news or catalyst column (0 of 120 and 0 of 124).
  `SecFilingCollection` events keep the API's labels as raw evidence, documented as not
  a timing source. The SEC family inside the pinned catalyst authority is removed in the
  retirement's evidence re-issue step, where pinned files change with fresh evidence.
- Prospective and live timing: no live SEC timing path remains after the deletion. The
  issuer-reaction design must take each new filing's acceptance from its EDGAR detail
  page (EDGAR's own New York clock), never from `acceptanceDateTime`.
- Collector: four worker threads, each with its own HTTP session, share one governor
  at five requests per second. The first 403 or 429 stops new submissions; in-flight
  attempts finish and are recorded before the checkpoint. The corrected run writes a
  new output, `data/raw/sec_filing_documents_initial_fit_corrected_clock`. It completed on
  September 26 (manifest `b3041a07f599e5c7f6a7b86a5991864c81586ba98e68391311fa3e9e84969318`,
  a final outcome for every unit); the sealed later-window collection runs next.
- Added exit tests: a real page per convention, a DST-ambiguous raw value, pages that
  disagree, pages contradicted by in-data counts, an unfetchable group, a corrected
  after-close 8-K moving from intraday to the next open, a filing moving across the
  2024-05-28 cutoff, UTC issuers' rows unchanged apart from the new columns, and
  concurrent collection with a 403 stop.

September 21 bounded source inspection and reaction-measurement contract:

- The early saved Alpaca shard `f91f0fa1d3de638169abab12.parquet` has 18
  records; eight have text different from the title. Some retained text is article
  HTML, some is only a headline. This is a sample, not cohort-wide content coverage.
- The SEC archive manifest/footer reports 689,467 form-metadata events, 624 issuer
  coverage records and 877 saved submissions responses. Canonical SEC text is
  generated form metadata, not filing/exhibit content. Sparse saved corporate-action
  documents do contain original HTML but cannot establish full-universe coverage.
  Acceptance-plus-policy-lag is a research proxy, not historical receipt evidence.
- Implement a shared **post-event session measurement** first, without registering
  a training profile or qualifying content. Select the first official XNYS session
  whose open is strictly after event availability, before inspecting available bars.
  At-open equality selects the following session. Never skip missing selected bars.
- Return two independently nullable measurements: stock open-to-close return minus
  SPY open-to-close return; and stock volume divided by its previous 20 complete
  exchange-session volumes' mean. The latter excludes the reaction session and
  requires a positive denominator. Missing SPY does not invalidate supported volume.
- Preserve each event/decision row and its selected session boundaries. Require the
  completed session and every consumed event/identity/bar clock at or before the
  original decision. Use exact UTC nanosecond maxima and explicit missing reasons;
  reject malformed identity, duplicate bars, incompatible feed/price basis and
  contradictory clocks. Live construction rejects historical proxy semantics.
- This is availability-anchored price/volume measurement, not proof of an event's
  causal market effect. Event source/version and identity authorities remain
  caller-verified; measurement alone proves neither source nor feature admission.
- Component exit tests cover open equality and +/-1 ns, intraday/after-close news,
  holidays/DST/early closes, incomplete and missing selected sessions, lagged-volume
  gaps, independent missingness, future/pre-release poison, and batch/single parity.
  Contract and transform workers have disjoint ownership from test implementation;
  independent plan, design/ML and code reviews precede closure. Heavy runs stay serial.
- This bounded component does not complete the overall checkpoint. Cohort-wide
  content/coverage audit, content-qualified event extraction, source admission,
  final ordered feature columns, immutable replay and the last two fits still follow.
  Broad headline earnings/guidance rules and form-only SEC flags are not substitutes
  for qualified issuer content. No existing population, labels, splits or fits change.

Component closure: `8e24d45` is pushed. The shared measurement and source contract
pass 276 focused reaction, inherited relationship and continuity tests (33.41 seconds,
RuntimeWarnings treated as errors), changed-file Ruff and strict mypy. Independent
review findings are closed. Integration preserved retrospective proxy availability
instead of treating download time as a historical clock, and accepts zero-volume
sessions when the complete baseline mean is positive. No archive publication or
fits ran. Continue with original-content/version inventory and per-ticker/year
coverage; this is not completion of the final profile's admission checkpoint.

Detailed plan/design review approved implementation with these admission gates:
the original physical `feature_profile` changes explicitly to
`technical_relationships`; all other parent columns, including auxiliary clocks,
remain exact. Publication metadata exposes only the named 124-column contract.
The publisher owns immutable source/config contracts and monthly artifacts; a
separate worker owns independent row verification and readiness/training consumers;
the parent owns thin CLI/configuration integration. Current source/implementation
pins are always verified. Each historical code mismatch receives an explicit
disposition; unresolved inherited semantic discrepancies cannot be classified away
as provenance. Source metadata must prove security and price-vintage compatibility,
including corrected streams and quarantine boundaries. Added-feature missingness
must not alter eligibility or the selected fitting/scoring population.

Design adjudication: unlocated original bytes for the five previously listed code
hashes are not demonstrated data defects and do not require recovering old code
for this immutable-parent derivative. Record the explicit disposition
`historical_code_not_reexecuted_or_certified`, retaining original hashes without
claiming benign drift or equivalence. Baseline numerical correctness is inherited
from the independently pinned, accepted parent evidence, not newly established by
column preservation. Verify all consumed data/authorities and every newly executed
dependency separately; a path's historical presence never exempts its current
code from checking. Freshly validate the four additions against sources and retain
current inherited-row clock, maturity and cost checks. Actual contradictions still
block fitting. Existing historical replay verifiers and retained-model reuse gates
remain unchanged; old code snapshots are not a blanket prerequisite for this path.

Component verification now passes: 79 focused C# tests, 131 focused Python feature
tests, 14 independent feature regressions, and four fixture/real-process/continuity
checks. The real Python publication is imported by the built C# CLI; lock contention,
owner-process death, idempotent restart and tamper refusal are exercised. The first
interop run exposed stale CLI dependency metadata; a normal restore/rebuild fixed
it without changing package versions or weakening checks. Ruff and strict mypy
pass for the changed feature modules. Physical power-loss durability is unverified;
the current inbox adapter explicitly requires a Windows fixed local disk. No full
suite, provider download, model fit or admission ran at this component checkpoint.

The raw collector is complete and pushed in `94aa1c4`; the new C# raw importer
preserves its bytes and ownership. Automatic polling and normalized catalog/feed
integration are not implemented by this explicit import operation.
Existing desk feeds stay unchanged until that cutover is explicitly verified;
raw import cannot fabricate normalized evidence, model admission or order authority.
The user approved deferring remaining intraday retirement and retained-model replay
to unblock this implementation. Historical source-pin mismatches still prohibit
reuse; no admission gate is weakened. Both product cohorts share raw evidence,
not a forecast target: investment training and execution are not implemented yet.

Deferred checkpoint: **complete dedicated intraday retirement** (resumed September 25
in parallel with the SEC document collection; step 1 inventory measured, design for
steps 2-6 frozen below for independent review before code).

Step 1 inventory (read-only, September 25):

- `market_predictor/intraday/` holds 72 modules. Nine source files outside it import
  it: `commands/edge_rebuild.py` (history-collection contract, prospective SIP-session
  and broker-action collectors), `governance/outcomes/maturation.py`,
  `governance/promotion/bundle_contracts.py`, `governance/readiness/audit.py`,
  `label_reconciliation.py`, `live_features.py`, `serving/model_context.py`,
  `serving/prediction_service.py` and `strategy_research_contracts.py`; 53 test files
  import it. Eleven `configs/*intraday*`/one-minute configs exist.
- Mode-level references outside the package concentrate in about 15 files
  (`modeling/prediction_selection.py`, `serving/prediction_service.py`,
  `live_features.py`, `governance/promotion/bundle_contracts.py`,
  `evidence/readiness_authority.py`, `serving/model_context.py`, drift, readiness and
  outcome contracts, `core/prediction_contracts.py`, `serving/decision_policy.py`,
  `strategy_research_contracts.py`). Other mentions are the swing `intraday_return`
  feature, sub-daily bars or session wording and stay.
- Closed swing evidence pins `label_reconciliation.py` and `live_features.py` and, through
  their imports, `intraday/__init__.py` and `intraday/contracts/{__init__,configs,memory}.py`
  (the two relationship feature publications and the relationship model request).
  `predictor_replay` compares recorded current-implementation hashes with today's code,
  so editing those files requires fresh swing-only replay evidence published beside,
  never over, the historical pins (the approved September 20 policy).

Proposed bounded sub-slices, each with its own design/diff review and checkpoint:
(a) move the retained prospective SIP-session, broker-action and history-collection
code and the security-namespace dependency to `sources`/`evidence`/`universe`
ownership; (b) make serving, prediction selection, decision policy, bundle, readiness,
outcome and drift contracts swing-only, removing retired modes without aliases and
failing closed on incompatible historical artifacts; (c) make `live_features.py`,
`label_reconciliation.py` and the strategy research contracts swing-only, then run the
swing-only predictor and outcome replays (heavy, after the SEC download releases the
lease) and publish that evidence separately; (d) delete the intraday package, its
exclusive tests and configs once no consumer remains; (e) update documentation; (f)
reference scans, CLI/API rejection, causal/lineage and swing regression, full suite,
Ruff and strict mypy, consolidated review. All of it closes before the issuer-reaction
features and the last two fits, so they build on the final swing-only contracts.

Consolidated review decisions for the retirement (September 25; they supersede the
inventory and sub-slices above where they differ):

- Closed evidence pins more intraday-bearing files than listed:
  `modeling/strategy_contract.py` and `configs/edge_rebuild_strategy_contract.toml`
  (19 artifacts, including both model requests and all three baseline readiness
  reports), `label_paths.py` (5 relationship artifacts) and `canonical/joins.py` (14).
  `load_return_inputs` re-hashes every readiness pin and requires current bytes for
  implementation files including the strategy contract, so editing any of them makes
  both model families unloadable.
- After the user's Windows permission change the baseline's 18 unit folders are
  readable. Both runs' root manifests match their recorded hashes (`0a30ef2f...` and
  `7a699020...`), and all 18 unit manifests and 34 files per run reproduce with no extra
  files. Evidence can therefore be re-issued for both model families, as the approved
  September 20 policy requires (fresh swing-only evidence beside the historical pins);
  no pinned file stays frozen and the no-model-modes exit gate stands.
- Fresh evidence under the new code, published beside the old, must reproduce exactly:
  readiness reports (479,709 eligible and 378,037 supervised rows, 545 securities,
  1,231 sessions, model and availability columns, `profile_sha256`); a relationship
  saved-row receipt over all 586,305 rows (values, dtypes, nulls, UTC nanosecond
  clocks); `load_return_inputs` reproducing `input_decision_ids_sha256`, rows, feature
  names, folds and holdouts; every saved unit re-scored on the new-code matrices
  reproducing its saved prediction payload byte for byte; and the new strategy
  contract's swing section with the same semantic hash under a new contract version,
  after the mixed contract's bytes are archived under `configs/lineage/`. The loader
  accepts old model units only with these receipts; anything else fails closed.
- Sub-slice constraints: (a) adds modules only and leaves the seven files the SEC
  collector re-hashes on resume byte-identical until that collection's manifest exists;
  no branch switch, stash or rebase touching `src/` while it runs; every commit stays
  importable. (b) keeps the wire `mode` field (value `swing`) and `resolved_horizons`;
  a schema change runs TradingFlow's `MarketPredictorHttpClientTests` and the Python
  API contract tests. (c) owns every pinned file (the six listed, the strategy contract,
  `label_paths.py`, `canonical/joins.py`) with the fresh evidence and removes the SEC
  family from the pinned catalyst authority. It also moves `data_quality._safe_json`
  (pinned by relationship evidence) to one shared helper used by
  `catalysts/issuer_events/content_inventory.py`, which carries a byte-identical inline copy
  since the SEC clock slice removed its catalyst-to-`data_quality` import. (d) deletes
  `intraday/__init__.py` and `intraday/contracts/*` only after (c)'s evidence is published.
- A retained-evidence existence gate precedes any deletion. Kept: the prospective
  SIP-session collector and its data, minute and hourly transports, selected-session
  data, `governance/outcomes/maturation.py`, `edge_rebuild/swing_setups.py` and the
  horizon-generic label code the approved 63- and 252-session investment targets need;
  `feature_timeframe="1Hour"` changes only under an explicit new version. The commands
  `collect-edge-prospective-broker-actions` and `collect-edge-prospective-sip-session`
  stay; `train-edge-swing-broker-specialists` is swing research, not intraday, and stays.
- The reference-scan baseline for (f) adds the unpinned intraday-bearing modules
  `universe/sp500/historical_security_namespace.py`, `readiness.py`, `features.py`,
  `price.py`, `feature_store.py`, `strategy_governance.py`, `governance/drift/policy.py`,
  `serving/outcome_intents.py`, `serving/bundle.py`, `edge_rebuild/swing_setups.py`,
  `market_regime.py` and `commands/canonical_data.py`.

Sub-slice (a) design (September 25; frozen for review before code):

- Measured scope. The three retained collectors are the prospective SIP-session
  authority (`intraday/datasets/prospective_sip_session.py`), prospective broker actions
  (`intraday/datasets/prospective_broker_actions.py`) and the Alpaca bar history contract,
  plan and transport they use (`intraday/contracts/history_collection.py`,
  `intraday/datasets/history.py`, `intraday/datasets/history_collection.py`). Together they
  are 6,973 lines. Broker actions also import `bar_dataset.load_complete_intraday_bar_dataset`,
  which pulls in about 9,700 more lines of intraday bars, features and labels. None of the
  five files is pinned by closed evidence (checked against 531 manifests and the reports).
  None hashes its own code. Their last runs are `data/raw/prospective_broker_actions`
  (poll 2026-08-21, `registry_v2`) and `data/raw/prospective_sip_sessions`
  (`session_20260820_v1`). No module outside `intraday/` uses
  `prospective_analyst_revision_horizon`, `selected_session_history`, `benchmark_history` or
  `one_minute_coverage`, so they go with (d).
- Ownership. A new production package, `market_predictor/collection/`, holds raw
  provider-evidence pipelines: plans, transports and prospective source authorities that
  write immutable collections. Its allowed dependencies are core, evidence, canonical,
  sources, universe, locking and resources. `sources` cannot hold these pipelines, since
  sources may not import canonical or universe; `catalysts` may not import locking. The
  package boundary test gains the package and its allowed list.
- Moves, with neutral Python names:
  - `intraday/contracts/history_collection.py` to `collection/alpaca_bars/contracts.py`;
  - `intraday/datasets/history.py` to `collection/alpaca_bars/plan.py`;
  - `intraday/datasets/history_collection.py` to `collection/alpaca_bars/transport.py`;
  - `intraday/datasets/prospective_sip_session.py` to `collection/prospective_sip_session.py`;
  - `intraday/datasets/prospective_broker_actions.py` to `collection/prospective_broker_actions.py`.
  Identifiers already written into artifacts stay byte-identical, so every existing
  collection, poll and registry still loads. Examples are schema strings such as
  `edge_rebuild.intraday_history.v1` and `edge_rebuild.intraday_bar_dataset.v1`, and
  request keys. Python names change, for example `IntradayHistoryConfig` becomes
  `AlpacaBarHistoryConfig`. The reference scans in (f) list these data identifiers as
  permitted.
- Security namespace. Broker actions read the A4.3 bar dataset only to fix the security
  identity namespace through its parent lineage. The collector drops the full-dataset
  load, and with it every intraday bar, feature and label import, and keeps its own checks
  of that dataset's authority, manifest, request and parent-lineage hashes. The namespace
  still comes from the pinned membership authorities and
  `verify_membership_namespace_extension`. Partition bytes were never used for the
  namespace.
- Intraday modules that import a moved file switch to the new path in the same change,
  so every commit stays importable; (d) deletes them. `commands/edge_rebuild.py` imports
  from `collection`, and command names and CLI options are unchanged. Configs stay
  byte-identical. The SEC collector's re-hashed files are untouched.
- Exit tests: the moved tests run under `tests/test_collection_*.py`; the package
  boundary test covers the new package; command help works; and a read-only load of the
  real `registry_v2` and `session_20260820_v1` through the moved loaders proves format
  continuity. Checks: Ruff and strict mypy on the moved files, and the affected consumers'
  tests.

Consolidated review decisions for sub-slice (a) (September 25; both reviews had no
blockers; they supersede the design above where they differ):

- Configuration identity. Stored policy hashes (`five_minute_policy_sha256`,
  `benchmark_policy_sha256`) are `model_dump(mode="json")` hashes, so the moved models keep
  field names, nesting, defaults and types exactly. Only module, class and function names
  change. An exit test recomputes both hashes stored in `session_20260820_v1`.
- Only what the retained collectors and `commands/edge_rebuild.py` use moves.
  `build_intraday_history_plan`, `_verify_readiness_audit` and
  `verify_existing_ohlcv_identity` stay in `intraday` for (d), as do the research-only
  configs (`ExtendedSessionContextConfig`, `SelectedSessionHistoryConfig`,
  `SelectedSessionOneMinuteConfig`, `BroadIntradayHistoryConfig`) and their plan schemas.
  `collection` never imports the readiness authority. The moved `json_sha256` copy is
  replaced by `evidence.hashing.json_sha256`, which is byte-identical.
- Retained lineage inputs, which (d) and (e) must never delete or edit:
  - `configs/edge_rebuild_intraday_history.toml` and
    `configs/edge_rebuild_selected_session_benchmarks.toml` (pinned by the SIP-session
    request);
  - the metadata files `_request.json`, `_manifest.json` and `_authority.json` of
    `data/features/edge_rebuild_intraday_bar_only_causal_20260814_v1`;
  - the base membership authority `data/canonical/index_membership/sp500_memberships_20180529_20260708_v1`.

  The security namespace itself comes from the membership authorities and
  `verify_membership_namespace_extension`. The bar-dataset files remain recorded lineage
  that every broker-action poll re-derives (`intraday_bar_*` hashes and
  `security_identity_namespace_sha256`), so `registry_v2` continues rather than forking.
  (d)'s deletion scan exempts this list, which is kept in a checked allowlist.
- Boundary: `collection` joins `PRODUCTION_PACKAGES` with its own allowed-dependency test,
  and the five old module paths join `REMOVED_PRODUCTION_MODULES`.
- Importers updated in the same commit: `commands/edge_rebuild.py` and 16 intraday
  modules; 5 test files move and 10 switch imports; `docs/implementation_guide.md` paths.
- Exit tests:
  - lease-free, low-memory loads of all six polls and both registries through the moved
    loader, with the namespace function returning exactly each poll's five stored hashes;
  - a tampered request or lineage is refused;
  - the next poll's and the next session's requests, built from today's configs, reproduce
    the recorded namespace and policy hashes;
  - the four pinned intraday files stay byte-identical to the relationship receipt's hashes;
  - the ten switched intraday test files still pass.

Sub-slice (a) status: implemented in `20799b9` (see the handoff for checks and continuity).

Sub-slice (b) design (September 26; frozen for review before code):

- Measured scope. Outside `intraday/`, about 400 intraday references sit in 21 modules. The
  largest are:
  - `governance/readiness/audit.py` (10 intraday functions and classes; imports
    `intraday` and its specialist modules);
  - `serving/prediction_service.py` (`predict_intraday`, the unified path and 5 helpers;
    imports `intraday.model`);
  - `evidence/readiness_authority.py` (`_validate_current_intraday_source`);
  - `modeling/prediction_selection.py` (5 intraday selection functions and a `unified`
    policy entry).

  Smaller ones: `readiness.py` (`assess_intraday_readiness`), `serving/decision_policy.py`
  (`determine_intraday_signal`), `core/prediction_contracts.py` (`IntradayPrediction`),
  `governance/promotion/bundle_contracts.py` (`PromotedIntradayBundle`,
  `validate_intraday_schema`), `governance/outcomes/{contracts,performance,maturation}.py`,
  `governance/drift/{policy,features}.py`, `serving/{model_context,bundle,outcome_intents,
  investment_replay}.py`, `governance/promotion/bundle_verification.py`,
  `strategy_governance.py`, `features.py` and `hypothesis_registry.py`.

  None of the 21 files is pinned by closed evidence (checked against 531 manifests and the reports).
- Wire contract. `PredictionMode` and `PredictionView` become `Literal["swing"]`, and the
  request default changes from `unified` to `swing`. `IntradayPrediction`, the row's
  `intraday` field and the `timeframe="intraday"` option are removed. The prediction
  contract moves to `market_predictor.prediction.v3` and the evidence contract to its
  next version, so historical payloads fail closed on their old versions. The `mode`
  field (value `swing`) and `resolved_horizons` stay. TradingFlow's client already sends
  `mode="swing"` and validates only `mode`, `models` and `resolved_horizons`, never the
  contract version, so its `MarketPredictorHttpClientTests` run as a regression check,
  with no C# change expected.
- Serving and selection. `predict_intraday`, the unified combination
  (`determine_final_signal`, `combined_readiness`), intraday scoring, readiness and
  suppression helpers, `select_intraday_candidates` and related functions,
  `determine_intraday_signal` and `assess_intraday_readiness` are deleted, with no
  aliases. A request naming any other mode is a validation error.
- Governance. The readiness audit and readiness authority verify swing and catalyst
  sources only; the intraday source slices, proxies, fold capacity and benchmark checks
  are deleted. Bundle contracts, verification, drift, outcome contracts, performance and
  maturation keep one swing mode. Stored bundles, outcome intents or drift reports that
  declare intraday fail closed with an explicit "retired intraday artifact" error.
  `serving/model_context.py` stops importing `intraday.contracts`.
- Kept deliberately: the swing `intraday_return` feature, sub-daily bar wording, session
  labels such as the SEC `intraday` acceptance position, and horizon-generic code the
  investment targets need.
- Exit tests:
  - a request with mode `intraday` or `unified` is rejected;
  - the default is `swing`;
  - swing responses keep `mode` and `resolved_horizons`;
  - historical intraday bundles, outcome intents and drift reports are refused;
  - readiness passes on swing-only sources;
  - affected serving, governance, readiness and outcome tests pass, as do the API
    contract tests and TradingFlow's `MarketPredictorHttpClientTests`;
  - Ruff and strict mypy pass on the changed files.

Consolidated review decisions for sub-slice (b) (September 26; both reviews had no
blockers; they supersede the (b) design where they differ):

- One atomic commit. It also deletes every intraday module except the four pinned files
  (`intraday/__init__.py` and `intraday/contracts/{__init__,configs,memory}.py`), together
  with those modules' exclusive tests. Leftover intraday modules import symbols (b) removes
  (`model.py`, `evaluation`, `specialist_model.py`, `promotion.py`, `datasets/history.py`).
  The pinned files import only `intraday.contracts`, and `label_reconciliation.py`,
  `live_features.py` and `strategy_research_contracts.py` import only that too, so this
  deletion is safe before (c). (d) then only deletes the four pinned files after (c)'s
  evidence. Intraday configs are deleted only when no manifest or report outside intraday
  evidence pins them. The retained allowlist and the mixed strategy contract, which (c)
  owns, stay. Checks: whole-package strict mypy and an import of every module.
- Retired ER1 readiness tooling is deleted: `governance/readiness/audit.py`,
  `governance/readiness/contracts.py`, `evidence/readiness_authority.py` and
  `configs/prediction_data_readiness.toml`. Nothing calls the audit, and the authority's
  only reader is the deleted intraday planner. The swing benchmark guarantee is enforced
  where swing readiness actually runs, in `swing/labels/fixed_horizon_readiness.py`
  (bound SPY, QQQ and sector components within the fit boundary). The published
  `data/research/edge_rebuild_readiness_er1_20260728` stays on disk as a record.
- Wire contract. The design's claim about TradingFlow was wrong. TradingFlow requires:
  - the response fields `mode`, `predictions`, `errors`, `models`, `resolved_horizons`,
    `generated_at_utc`, `final_signal`, `readiness_status`, `request_id` and `snapshot_id`;
  - per ticker, non-null `swing`, `swing.readiness`, `swing.catalyst` and
    `swing.global_context`, plus `models.swing` with `status`, `model_type`,
    `schema_version`, `target`, `artifact_sha256` and `training_data_end`;
  - the swing fields `probability`, `decision_score`, `signal`, `rank`, `return_1d` and
    `volume_z20`.

  All of them stay. A Python test serializes a real swing `PredictionResponse` and asserts
  each is present and non-null. The contract becomes `market_predictor.prediction.v3`, and
  evidence moves from `prediction_evidence.v3` to v4. `SwingPrediction.unified_score` and
  `ReadinessInfo.intraday_bar_count` are removed. Horizons keep the generic
  session/day units (`b`, `d`); minute and hour units and the intraday aliases go.
- Drift. `drift_policy.v3` replaces the two per-view pending-age constants with one limit
  derived from the prediction's horizon (horizon sessions plus a grace period), so a
  63- or 252-session outcome is not flagged overdue. `configs/default.toml`'s
  `drift_policy_sha256` is updated, and a test ties it to the hash of
  `configs/drift_policy.toml`. The promoted-bundle schema is unchanged.
- Investment targets will get their own view and policy later rather than widening swing.
  Generic horizon parsing and cohort horizon fields stay, and nothing new hard-codes
  `10b` outside the swing route.
- Retired-intraday refusal. `mode="before"` validators on bundle `mode` and on the `view`
  of outcome intents, observations, matured outcomes and performance cohorts, plus the
  outcome repository loaders, raise an explicit retired-intraday error. Drift paths for
  intraday raise it too. Old prediction v2 and evidence v3 snapshots are refused;
  nothing of that kind exists locally. `publish-drift-assessment --mode` accepts only
  `swing`.
- Also swing-only in this commit:
  - `governance/outcomes/repository.py`, `commands/outcomes.py`, `release.py`
    (`canonical_intraday`) and `serving/snapshot_store.py`;
  - `serving/requests.py`, whose duplicate request contracts merge into the core ones;
  - `scripts/promotion_fixture.py` and its `tests/r4_fixtures.py` users.

  `feature_store.py`'s intraday path depends on the pinned `live_features.LiveMode`, so it
  moves to (c). `features.py` changes nothing: its intraday mentions are swing
  news-session and reaction features.
- Strategy ledger. `docs/strategy_execution_ledger.json` keeps its 12 intraday entries as
  historical records. `mode="intraday"` stays valid only in terminal states, and the 4
  `planned` intraday entries are closed as retired with the retirement as their blocker.
- TradingFlow follow-ups, outside this slice and recorded for the user:
  - its advisory model-direction lists recognize only the retired combined signals, so
    every current swing signal reads as neutral (`UniverseRankService.cs:130-138`);
  - its result label is hard-coded to `market_predictor.prediction.v1`
    (`MarketPredictorHttpClient.cs:336,400`).

  Both are display-only; TradingFlow's scores and orders never depend on predictor
  evidence.
- Exit tests, beyond the design's:
  - the TradingFlow field contract;
  - drift hash equality;
  - refusal of v2/v3 payloads and of intraday monitoring and outcome records;
  - `--mode intraday` rejected;
  - ledger validation;
  - whole-package strict mypy and a module import smoke;
  - `ReadinessInfo` and replay rejecting intraday.

Sub-slice (b) implementation record (September 26; `7dd6d44`, review follow-up `ea93712`,
both pushed; diff reviews by both reviewers found no blockers):

- Corrections to the decisions above, each from measured evidence:
  - Horizons are exchange-session counts only (`b`). The "keep `b` and `d`" decision is
    superseded: at HEAD swing maturation intents already required `b` and serving
    already required `10b`, so a `d` horizon could never be served or matured. Request,
    cohort, drift and store paths accept `[1-9]\d*b`; the day aliases (`tomorrow`,
    `week`) are removed with `1h`.
  - The overdue limit is exact rather than estimated: a pending prediction is overdue
    once the close of the horizon's last XNYS session after its recorded decision session
    (cohorts carry `oldest_pending_decision_session_et`), plus `pending_grace_days` (7),
    has passed. The weekday estimate (`ceil(sessions * 7/5)`) ignores holidays: after
    24 July 2025 the 252nd session closes 27 July 2026, but the estimate's limit (353 days
    plus the grace) ends 19 July, so 252-session outcomes would be flagged eight days
    before their last session even closed.
  - The performance report keeps only decisions inside its lookback window (default 60
    days). Monitoring the 63- and 252-session investment targets therefore needs a window
    longer than the horizon; the investment-target design must set it.
  - `training_data_end` in `models.swing` stays nullable: TradingFlow declares it
    `string?`, and promoted swing artifacts do not record a training end (see below).
  - Promotion evidence stays type-agnostic; product admission refuses retired types.
    Release verification refuses `canonical_intraday` releases and bundle verification
    refuses `intraday` bundles, each with an explicit retired error, even when their
    hashes and signatures verify.
- Also removed: the `curated` data source (swing always reads live inputs) and
  `ServingRoute.curated_dataset`.
- Defects found and fixed in this commit, all present at HEAD:
  - registering outcome intents failed for any snapshot whose request named a ticker
    outside the live universe (the unscored abstention has no evidence row); unscored
    tickers are now skipped and a scored prediction without its row still fails;
  - an empty intent mapping was treated as "not supplied" and recomputed;
  - the CI container smoke release could not start: it configured a `5d` swing route and,
    as the code review found, lacked the required gate and drift pins (`7dd6d44` fixed
    only the route; `ea93712` adds the pins and a test that starts the app from the
    generated config and checks CI's live 200 and ready 503);
  - `publish_serving_bundle` did not validate `mode` at run time.
- Review follow-up `ea93712` (code review: one major, ten minor; ML review: six major,
  finding 4 in part):
  - requested members dropped for incomplete market or catalyst inputs abstained as
    `out_of_universe` (present at HEAD); they now abstain as `live_inputs_incomplete`,
    the live frames carry their point-in-time tickers, and registration reports every
    unmonitored ticker by reason instead of skipping it silently;
  - prediction contracts refuse unknown fields, so retired fields are rejected rather
    than dropped; outcome loaders raise the explicit retired error for stored intraday
    records;
  - the overdue check refuses naive times, non-session decision dates and dates past the
    cached XNYS calendar's end instead of returning False;
  - the retired release fixture carries `intraday_training_evidence.v1`, and the
    pre-retirement simulation applies the old schema check rather than skipping it;
  - unused ranking, action, calibration and readiness code is deleted
    (`modeling/prediction_selection.py` keeps only the served policy; `readiness.py`
    goes); shadow hypotheses take session horizons and check the view at run time; a
    configured intraday route names the retirement;
  - added tests for superseded observation, outcome and feature-drift versions.
  Moved to (c): the intraday `feature_path` branch in `serving/bundle.py`, tied to
  `feature_store`. The whole-package import smoke stays a recorded checkpoint check
  beside strict mypy rather than a permanent test.
- Monitoring defects found by the ML review, verified in code, awaiting the user's scope
  decision (all present at HEAD for the served ten-session route unless noted):
  - an outcome that can never mature (a stock acquired or delisted mid-horizon) stays
    `pending` forever, blocks the route once overdue, then leaves the report window and
    is never counted; a terminal unresolvable outcome is missing;
  - the report window must exceed the overdue deadline plus the sampling period, not
    only the horizon (for the 63- and 252-session targets);
  - `independent_decision_groups` counts overlapping decision groups, and drawdown and
    cumulative return compound overlapping holding-period returns as if sequential
    (30 daily groups at -2% read as a 45% drawdown); excess-return thresholds do not
    scale with the horizon;
  - swing drift has no check that realized excess return rises with the served score;
  - monitoring counts only requested tickers, so its rates depend on the request mix;
    registering the whole scored cross-section per decision session, with members
    excluded for incomplete inputs counted as a coverage rate, is the complete fix.
- Defects found, not fixed, awaiting the user's scope decision (investment replay, the
  `/v1/replays/investment` endpoint):
  - promoted swing artifacts never record a training-data end, so replay refuses every
    swing snapshot with "model training-data end timestamp is missing" (at HEAD the
    field came only from the retired intraday manifest's `dataset.last_date`). The
    boundary must be when the training labels became available, not the last decision
    date: the initial fit's decisions end 2024-05-28 but their ten-session labels use
    prices through 2024-06-11. A fix records it as an exact UTC instant in the swing
    candidate and training manifest, copies it into the promoted bundle, verifies it at
    promotion, and publishes it for the retained runs as separate evidence;
  - replay's `ACTIONABLE_SIGNALS` hold the retired signal names (`bullish_watch`,
    `strong_bullish_watch`), while serving emits `positive_setup` for a selected setup.
- Pre-existing, recorded: broker-action poll `poll_20260816T070948Z` is an incomplete
  attempt and fails identity-audit replay at HEAD.

Swing monitoring and replay correctness design (September 26; frozen for review before code):

The user decided on September 26 to fix every verified monitoring and replay defect above
now, as one step before retirement sub-slice (c). Measured facts it builds on:
- No production code writes a promoted swing serving generation (`bundle.json`,
  `active_generation.json`); only tests do. Nothing is promoted or served today.
- Every file this step touches is outside all 61 implementation-pin lists, so no closed
  evidence changes. The six frozen research specifications stay untouched.
- Intents already exist for every validly scored prediction, selected or not.
- The strategy contract fixes the live exclusion ceiling at 5%
  (`data_quality.maximum_security_exclusion_fraction`).

1. Outcomes that can never mature (a stock acquired, delisted or halted mid-horizon).
   - One shared session helper in `governance/outcomes` gives the close of the Nth XNYS
     session after a decision session; drift and maturation both use it.
   - After that close plus `pending_grace_days`, maturation records a terminal
     `unresolvable` attempt with reason `unavailable_trading` (the research cohort's
     vocabulary) and the missing sessions, but only when SPY, QQQ and the primary
     benchmark have their complete path and the stock does not. If a benchmark is also
     incomplete, our collection is lagging: the outcome stays pending and drift flags it
     overdue. No return is invented for an unresolvable outcome.
   - The worker skips terminal intents. Cohorts count `unresolvable_selected_samples`
     (matured + pending + unresolvable = actionable). Drift is severe when unresolvable
     outcomes exceed 5% of resolved selected outcomes, the same ceiling serving applies.
   - Nothing leaves monitoring uncounted: the route's oldest still-pending decision is
     taken over every stored intent for the route, not only those inside the window.
2. Report window. Each route's window is aligned to maturity: a decision is included
   when its horizon's last session closes on or after `generated - lookback_days` (or
   has not closed yet). A 10-session and a 252-session route then both cover the
   outcomes that finished in the same lookback period, and pending decisions stay
   visible. Rows already carry their own window bounds.
3. Overlap-aware statistics (overlapping holdings are not independent draws).
   - `independent_decision_groups` counts the largest set of matured decision groups
     whose holding periods do not overlap: in session order, a group counts when its
     decision session is at least N sessions after the last counted one.
   - Drawdown and cumulative return use a realizable overlapping-portfolio curve, the
     standard construction for overlapping holding periods (Jegadeesh and Titman, 1993):
     capital is split into N sleeves; sleeve k takes the decision groups whose XNYS
     session index is k modulo N, so a sleeve never holds two cohorts at once; each
     sleeve compounds its groups' equal-weight mean net return, holding cash when it has
     none; the route's equity is the mean of the sleeves, measured at each group's exit.
     Drawdown is therefore realized at exits, not intra-holding.
   - Excess-return thresholds become per session (`warning_min_excess_return_per_session`
     -0.0001, `severe_...` -0.0005, the current ten-session values divided by 10) and
     are compared with the average excess return divided by N. This changes
     `drift_policy.toml` and the pin in `default.toml`.
4. Score-versus-return check (swing had none). For each matured decision group with at
   least five scored predictions, the Spearman rank correlation between the served
   probability and the realized net excess return against the sector benchmark (the
   served target's basis; barrier-hit rates are never used). Over non-overlapping groups:
   the mean and its t-statistic. Drift warns when the mean is at or below zero and is
   severe when t <= -2, once `minimum_independent_decision_groups` groups exist.
5. Monitoring population and coverage.
   - `PredictionService.predict_swing_cross_section(as_of)` scores every effective
     point-in-time member; members excluded for incomplete inputs abstain as
     `live_inputs_incomplete`. Its snapshot records scope `decision_cross_section`.
   - A production command `register-session-predictions --as-of` records that snapshot,
     its intents and observations, and a session coverage record: members, scored, and
     excluded tickers by reason.
   - The performance report counts the XNYS sessions in each window without a coverage
     record; drift is not ready when any session older than the grace period is missing.
     Request snapshots may still be registered; they deduplicate by semantic identity,
     which includes the cross-section rank, so they cannot bias the rates.
6. Investment replay.
   - The swing candidate payload from `train_swing_edge_candidate` records
     `training_decisions_end_session` and `training_labels_available_through_utc`: the
     latest decision session and the latest `label_available_at_utc` of the final-fit
     rows. Labels, not decisions, bound look-ahead: the initial fit's decisions end
     2024-05-28 but their labels use prices through 2024-06-11.
   - The promoted bundle becomes `edge_rebuild.promoted_bundle.v3` with both fields;
     verification requires them to equal the verified model payload's values.
     `ModelInfo.training_data_end` shows the decision end and a new
     `training_labels_available_through_utc` carries the instant.
   - Replay requires that instant to be before the decision time, refuses when it is
     missing, and drops the date-to-16:00 conversion. "Actionable" is the snapshot's
     `selected_for_policy`, not a list of signal names.
   - When a promotion path for the frozen research specifications is built, it must
     derive the same two values from their pinned training requests as separate
     evidence; no historical pin is rewritten.
7. Exit tests: an acquired stock resolves as unresolvable while a lagging benchmark keeps
   it pending; an old pending decision outside the window still blocks drift; the
   maturity-aligned window for 10 and 252 sessions; non-overlap counting and the sleeve
   curve against a hand-computed case (30 daily -2% groups at N=10 give about a 6%
   drawdown, not 45%); per-session thresholds; the rank-correlation gate with a
   wrong-signed model; the cross-section command with an excluded member and a missing
   session; bundle v3 refusal without or with mismatched boundaries; replay refusal at
   the boundary and acceptance after it; replay actionability from `selected_for_policy`.

Consolidated design review decisions for swing monitoring and replay (September 27;
both reviews found two blockers each; these supersede the design above where they differ,
and follow the naming rule: no versions, old records refused by strict validation):

- Scope. Every mechanism works for any horizon N, but evidence minimums are set for the
  served ten-session route. Ten independent periods at N=252 would need about ten years,
  so the 63- and 252-session evidence policy belongs to the investment-target design, as
  already decided ("Investment targets will get their own view and policy later").
- Registration never waits on drift (blockers: the cross-section would inherit the
  serving drift gate and a warming route could never collect evidence).
  - `register-session-predictions` (CLI only) scores the whole cross-section without the
    actionability gate. It still requires a verified promoted generation with
    `promoted_at_utc <= as_of` and valid live inputs, and takes the heavy-job lease and
    the admission lease.
  - Serving to clients stays fail-closed; a warming route is in shadow monitoring.
  - Every XNYS session from route activation gets a session record: `registered`
    (members, scored, abstentions by reason) or `failed` (`exclusion_ceiling_exceeded`,
    `inputs_unavailable`, `model_unavailable`, `registration_error`). It is written last,
    as the commit marker, and is idempotent per route, release and session: an identical
    rerun is accepted, a different one refused.
  - Drift is not ready on unexplained gaps (no record, older than the grace) and warns
    when registered sessions fall below a policy share of the window.
- Population. Rates, sufficiency, the curve and the rank check use only cross-section
  observations from `registered` sessions, one cross-section per route and session.
  Request snapshots stay as audit records. A third abstention reason,
  `sector_peer_floor`, covers members whose sector has too few eligible peers to rank;
  it is recorded in coverage and not counted toward the 5% input-failure ceiling.
- Snapshots record a scope, `request` or `decision_cross_section`; a cross-section
  records its as-of time, route and member-set hash instead of a ticker list, so the
  100-ticker request limit does not apply. Registration and replay handle both scopes.
- Outcome evidence (blockers: missing bars cannot tell "not collected" from "did not
  trade", and the terminal state was irreversible).
  - A new outcome-bar collection requests each pending intent's path by
    `canonical_security_id` and the point-in-time ticker per session, and keeps receipts
    of every request, including empty provider responses.
  - The managed barrier is applied to the observed consecutive prefix: a target or stop
    reached before the first missing session matures with that exit (a stopped-out loss
    is never lost).
  - After the horizon's last close plus the grace, an outcome is `unresolvable` only when
    the stock was requested for the missing sessions and came back empty, with the
    point-in-time membership change as a sub-reason. Not requested means still pending.
  - Attempts are an append log and the latest decides, so a later successful maturation
    supersedes `unresolvable`.
  - The grace has one source, the drift policy's `pending_grace_days`, which
    `mature-outcomes` reads from the pinned policy file; each attempt records the grace
    and the policy hash.
  - The unresolvable share is unresolvable / (matured + unresolvable), applied only after
    the minimum matured samples, against a drift-policy ceiling (5% for the ten-session
    route). Dropping an unknown from an equal-weight group mean implicitly gives it the
    group's mean return; the report says so and adds a diagnostic sensitivity (last
    observed close; -30% for removals that are not mergers, after Shumway 1997).
- Evidence and inference (blocker: counting only non-overlapping groups could never
  reach the minimum within the lookback).
  - Every overlapping daily group stays. Evidence is counted as effective periods,
    matured decision sessions divided by N; sufficiency needs at least ten (100 matured
    sessions for ten sessions). The report lookback defaults to 150 days, and a
    validator refuses a lookback that cannot reach the minimum for its horizon.
  - Means (excess return, rank correlation) carry Newey-West standard errors with N-1
    lags over the daily group means, the standard treatment of overlapping returns.
  - The N-sleeve curve (Jegadeesh and Titman, 1993) is marked to market daily from the
    path evidence stored with each outcome; a group joins the curve once all its selected
    outcomes are resolved.
  - Excess-return thresholds are per session of actual holding time: the sum of excess
    returns divided by the sum of holding sessions.
- Score-versus-return check. Outcomes also record the fixed-horizon net return and
  sector excess (next open to the Nth close), the served label's own basis rather than
  the managed exit. The rank correlation is computed within each sector and averaged
  per decision group, over all scored outcomes (a new path for non-selected ones); the
  mean and its Newey-West t decide: warning at mean <= 0, severe at t <= -2, after the
  minimum effective periods.
- Repository scale. Outcome records are partitioned by decision session with a pending
  index; reports read only their window's partitions plus the index, and maturation
  iterates the index. A test builds a year of cross-sections (about 121,000 intents)
  within the process budget.
- Replay. The label boundary is the latest label availability over every row that
  influenced the artifact: final fit, calibration, threshold-selection validation rows
  and the locked test rows, since promotion depends on them. The bundle requires it
  before `promoted_at_utc`; replay compares it strictly with the prediction row's
  `decision_time_utc` (not the request's as-of), exposes it in its response, and uses the
  exchange calendar's actual close instead of a fixed 16:00 (early closes). The contract
  notes that `training_data_end` is not a look-ahead boundary.
- The session helper lives in `governance/outcomes`; swing modules must not import it (a
  dependency rule is added), a parity test ties it to the pinned `holding_calendar`, and
  drift delegates to it. Pinned files (`holding_paths.py`, `barrier_and_rank.py`,
  `label_paths.py`, `research_cohort.py`, `strategy_contract.py`) are only called, never
  edited.
- Delivery in reviewable parts, each with its own diff review: (1) sessions, overdue
  and maturity windows; (2) outcome evidence: receipts, prefix resolution,
  unresolvable; (3) cross-section registration, snapshot scopes, session records and
  population; (4) inference, curve and rank check; (5) repository partitioning;
  (6) replay boundary.
- Added exit tests (beyond the design's): registration while drift is warming or not
  ready; cross-sections over 100 members through registration and replay; identical and
  conflicting session reruns; a republished live generation not double counted; a
  symbol change or collection gap never becoming unresolvable; a stop before a gap
  maturing; the ceiling with small samples; the rank check on the fixed-horizon basis;
  a repository scale budget; refusal of records in the old shapes; replay at the exact
  boundary and on an early-close day; the session-helper dependency rule.

Second-round review decisions for swing monitoring and replay (September 27; neither
review found a blocker; these complete the consolidated decisions above):

- Sufficiency. Effective periods count registered cross-section sessions whose outcomes
  have matured; a session with no selection is a valid zero-exposure period and counts.
  The lookback defaults to 180 days: every rolling window from 2019 to mid-2026 holds 120
  to 127 XNYS sessions, while a 150-day window can hold as few as 99 (measured). The
  validator requires the fewest sessions in any window, minus one for the maturation lag,
  minus the failures the policy's registration share tolerates (95%), to reach
  minimum x N. Evidence minimums are keyed by horizon with only `10b` defined; drift
  refuses any other horizon.
- Standard errors. Hansen-Hodrick (equal weights, N-1 lags), exact for N-session overlap,
  with Newey-West at 2N lags as the positive-definite fallback. Newey-West at N-1 lags
  recovers only 67% of the variance at N=10 (computed), which would make t <= -2 behave
  like t <= -1.64.
- Curve. The canonical funded ledger (`swing/evaluation/ledger.build_funded_swing_ledger`),
  the capital model the promotion evidence was selected on, replaces independent sleeves.
  It takes the matured selected outcomes and their stored daily path marks; an
  unresolvable position is held at its last observed close. The drawdown thresholds are
  checked against the locked-test ledger drawdown before part (4) closes.
- Symbol changes and cessation. The outcome-bar collection also fetches Alpaca security
  transitions (`AlpacaSource.fetch_security_transitions`) for each pending horizon and
  follows name changes. `unresolvable` needs positive cessation evidence (a merger or
  reorganization transition, or a point-in-time membership removal) together with an
  empty receipt; an empty receipt alone stays pending and overdue for operator action.
  The worker re-attempts unresolvable intents when new transitions or receipts arrive;
  attempts are ordered by append sequence; a matured outcome is never superseded.
- Population. Request snapshots are audit-only: they are never registered as intents or
  observations, and `register-outcome-intents` accepts only `decision_cross_section`
  snapshots. First-writer canonicalization then cannot prefer a request.
- Registration lease. Registration takes a dedicated monitoring-registration lease, the
  admission lease and the memory guard, not the workspace heavy-job lease, so a
  multi-day research job cannot block nightly registration.
- Session records. Route activation is the first XNYS session whose decision cutoff is at
  or after the active generation's `promoted_at_utc`, per release. An identical rerun
  compares route, release, session, member-set hash, counts by reason and the snapshot's
  content hash without `recorded_at_utc`. A `failed` record may be replaced by
  `registered` from a retry within the same session; an operator may write a late
  `failed` record with reason `not_run`.
- Identities. Cohort and report identities also bind the session-record ids and the ids
  of the deciding attempts.
- Rank check. Outcomes without a full fixed-horizon path (prefix-resolved or
  unresolvable) are excluded and its coverage is reported; sectors are weighted equally
  within a decision group. The fixed-horizon sector excess is the trainer's declared
  economic target (`future_excess_return_10d_vs_sector`, `swing_training.py`); a
  non-gating diagnostic on the managed net return (the estimator's label) keeps any
  divergence visible. The fixed-horizon fields and `holding_sessions` join the outcome
  record in part (2).
- Unresolvable. The ceiling counts selected outcomes. The sensitivity uses -30% for
  NYSE/AMEX and -55% for Nasdaq removals (Shumway and Warther, 1999), the evidenced cash
  rate for cash mergers, and -100% for worthless removals.
- Sector peer floor. Drops cascaded by an input exclusion are recorded as such and count
  toward the 5% ceiling. `sector_peer_floor` is a new public abstention reason, so the
  API becomes `market_predictor.prediction.v4` when it lands in part (3), through the
  contract change log.
- Replay keeps the model-availability check (promoted at or before the request's as-of)
  beside the label boundary, and exposes the boundary in its response.
- Repository partitioning also covers the semantic-canonical lookup path.
- Order: (1) sessions, overdue and maturity windows; (5) repository partitioning;
  (2) outcome evidence; (3) registration and population; then the nightly live-input
  publisher (the user's decision of September 27); (4) inference, curve and rank check;
  (6) replay.
- Consequence to state plainly: with fail-closed serving, a newly promoted release stays
  warming, with clients refused, for at least 100 matured sessions plus N plus the grace,
  about five and a half months, and again after every re-promotion.

Monitoring implementation record (September 27):

- Part (1) `e9526f1`, review fixes `2f05100` (both diff reviews: no blocker, no major).
  The lookback check subtracts the sessions whose outcomes may still sit within the grace
  (at most 5 in any 7 days) and the tolerated failures, computed exactly: 170 and 180 days
  pass, 150 and 160 fail. The route-wide oldest pending decision is the earlier of the
  pending index and the window's own pending decisions, so a backdated report stays
  consistent; for decisions older than the window, a backdated report sees pending state
  as of its build time (the index is current state). Swing may not import governance.
- Part (5) `a9c1b89`: records are partitioned by decision session with a pending index;
  reports open only the partitions their window reaches (a test records the opened
  sessions); maturation walks only the index. The design's "year of cross-sections" test
  became a partition-bounded read test plus a measured benchmark (see the handoff), since
  writing 121,000 intents takes about 70 minutes.
- Part (5) review fixes `157f330` (code review: two majors; ML review: minors only).
  - Registration writes the pending entry before the semantic record, and a rerun
    restores a missing entry of the canonical intent, so a crash at any write leaves the
    intent indexed (a crash-injection test covers each of the five writes).
  - Index entries take no lock file, so `pending/` holds only current entries.
  - The worker matures an intent only after its horizon's last close and records no
    attempt before it. This also protects part (2a): an early exit matured before day
    ten closed would have stored its outcome, which is immutable, without the
    fixed-horizon return.
  - The worker drops an entry whose semantic record names another intent, and leaves an
    entry whose registration has not yet written its semantic record for the rerun; its
    summary counts index entries by what happened to them.
  - Reports load each intent once per partition; the repository refuses the flat
    layout, skips plain files among sessions, checks an attempt's session against its
    intent, and retries Windows sharing refusals on reads and replacements (four pauses,
    0.75 s in all). Hypothesis decision groups must be timezone-aware decision times,
    checked at declaration and by causal shadow.
  - Part (2b) design inputs: recording an attempt only when its status or reasons change
    needs the attempt append order part (2) defines; and a target or stop reached before
    a missing session may mature only when that session was requested and came back empty
    (a receipt), because after the horizon closes a session can also be missing only
    because it was not collected yet.
- Part (2a) `c39c68a`: a target or stop reached on the stock's observed consecutive
  prefix matures with that exit; without such an exit, the outcome stays pending on the
  stock's missing sessions. Outcomes record `holding_sessions` and, when the stock's
  whole path is observed, the fixed-horizon net return and sector excess (next open to
  the Nth close after the label cost), the trainer's economic target.
- Part (2a) review fixes `1ef0a7f` (both diff reviews of `c39c68a` and `157f330`: no
  blocker, no major; the ML review confirmed the fixed-horizon return equals the
  trainer's label term by term).
  - Maturation checks only the rows it uses. The stock's path ends at the first session
    without exactly one valid bar, and benchmarks are checked at entry, at exit and, for
    the fixed-horizon return, on session N. Before, an invalid bar anywhere on the path
    held the outcome, so early exits were dropped more often than late ones.
  - A target or stop reached before a gap matures only when the gap's first session is
    proven to have no usable bar (`proven_stock_gaps`). The bars artifact proves none, so
    until part (2b)'s receipts such outcomes stay pending, and collection lag can never
    leave an early exit without its fixed-horizon return (outcomes are immutable).
  - The intent stores the ATR as a fraction of the decision close
    (`managed_risk.atr_fraction_of_latest_close`, already in every scored prediction, so
    the API is unchanged). Maturation applies it to the decision close of the bars it
    matures on, so a split or dividend adjustment after the decision no longer moves the
    stop and target. Outcomes record the fraction and that decision close.
  - Contract checks: a timeout carries a fixed-horizon return equal to its label net
    return; the exit is the (holding - 1)th session after the entry; the repository
    checks that the entry opens the session after the decision.
  - Also: a trainer-parity test (`add_exact_swing_labels`); a pending entry that vanishes
    during a registration rerun is written again; `mature-outcomes` exits 1 while
    registrations are unfinished; the outcome commands' time options accept an offset
    (Typer's default formats carry none, and the commands refuse times without one, so
    these options could never be used); the report's duplicate outcome-identity check is
    removed, since the repository checks it on load; test signing keys live in a fresh
    folder per process (a reused process ID had picked up keys under the retired schema
    name).
  - Answered without a change: maturation already refuses bars not fully adjusted
    (`_prepare_bars`).
  - Part (2b) design inputs from these reviews: a receipt proves a gap only when
    requested after the horizon's last close plus the grace; requests follow name
    changes; a response without a usable bar counts as empty; a duplicated stock row is a
    data defect, not a gap; an unusable session inside the path, with later valid bars
    and no exit before it, needs a terminal state of its own.
  - Part (3): the session record surfaces unfinished registrations, and the API's v4
    change also removes `PredictionRowEvidence.decision_atr`, which nothing reads now that
    intents take the ATR fraction (one version change for TradingFlow, not two). Part (4):
    the managed excess (net of the execution cost) and the fixed-horizon excess (net of
    the label cost) are never compared or combined, and each statistic stays on its
    stated basis.
  - Confirmation reviews of `1ef0a7f`: no blocker, no major. Their minors are fixed in
    `b872b30`: the repository binds the recorded decision close to the decision bar in the
    outcome's evidence, on record and on load; a timeout's fixed-horizon sector return
    must equal its managed sector return (the same interval); a calendar-edge failure in
    the entry check is a record conflict. The ML review found no unintended change to any
    statistic or population: every matured outcome now needs the same complete path, so
    every one carries its fixed-horizon return.
- Carried forward from the part (1) reviews:
  - Until part (4), sufficiency still counts distinct matured decision groups (10) while
    the lookback check already assumes part (4)'s effective periods (10 x N sessions);
    the check is deliberately the stricter one. Parts (1) to (4) must all land before any
    promotion.
  - Part (2) must exclude terminal `unresolvable` intents from pending, or each would
    block its route forever.
  - Part (4) decides whether non-selected pending intents gate (the rank check needs
    them), and computes outcome metrics and sufficiency only over decisions whose
    deadline has passed, so the newest edge is not conditioned on fast maturation.

Part (2b) outcome evidence: implementation design (September 27; for review before code)

It implements the consolidated decisions "Outcome evidence", "Symbol changes and
cessation" and "Unresolvable" together with the part (2a) review inputs. No real
prediction is registered or matured yet, so no stored record changes.

1. Outcome-bar collection (`collection/outcome_bars.py`; the collection layer may use
   `sources`, `evidence` and `core`, never `governance`).
   - Governance builds request units from the pending index: one unit per decision
     session and horizon, holding the point-in-time tickers of its canonical intents
     whose horizon's last close has passed, plus SPY, QQQ and their sector ETFs. A unit
     asks for daily SIP bars adjusted `all` from the decision session to the Nth session,
     with `asof` set to the decision session, so Alpaca maps each symbol to the company it
     named that day and follows its later renames. At most 50 symbols per page, following
     page tokens (`AlpacaSource.fetch_bars_page`).
   - Every page is kept byte for byte (`bodies/<sha256>.json`). Each unit gets an
     immutable, self-hashed receipt (`receipts/<decision session>/<receipt id>.json`):
     requested symbols, sessions and `asof`, URLs, status, `retrieved_at_utc`, page
     hashes, and for each requested symbol the sessions returned. A symbol absent from the
     response and one returned with an empty list are both recorded as "no bars" (the
     Alpaca reference does not say which one it sends).
   - The loader verifies every page and receipt hash and decodes daily bars into the
     maturation schema: XNYS session open and close, availability the later of the close
     plus 15 minutes and `retrieved_at_utc`, feed `sip`, adjustment `all`. Unusable bars
     (zero volume, prices out of order) are kept for the pinned validator to mark; the
     loader does not call `canonicalize_bars`, which rejects a batch with any invalid row.
   - Corporate actions: the existing byte-keeping transport is reused
     (`sources/alpaca_corporate_actions.fetch_corporate_actions_page` and
     `decode_corporate_actions_page`: one ticker per request, structural checks only, and
     `process_date` never standing in for `effective_date`). For each stock with a gap it
     asks for that ticker's actions over the path window extended by the grace, with
     receipts as for bars. Name changes, cash, stock and stock-and-cash mergers,
     reorganizations and worthless removals are read from the kept records, including the
     cash merger `rate`; a record without an `effective_date` is not admitted evidence.
   - Name changes: when a name change of a unit's ticker falls inside the path and the
     stock's bars stop there, the next collection asks for the new symbol with `asof` on
     its effective date. Bars are joined back under the intent's ticker, and the evidence
     records the symbol used for each session.
   - One command, `collect-outcome-bars`, takes a named monitoring lease (not the
     heavy-job lease, so a multi-day research job cannot block it) and the 90% memory
     guard. It never matures anything.
2. Evidence classification (`governance/outcomes/evidence.py`).
   - A receipt settles session s for symbol x when it asked for s for x and was retrieved
     at or after the horizon's last close plus the grace (`pending_grace_days` from the
     pinned drift policy). The latest settling receipt decides. Before that deadline no
     gap is proven.
   - A settled session without a usable bar, because none came back or only an unusable
     one did, is a proven gap; these form `proven_stock_gaps`.
   - A duplicated stock row is a data defect and blocks the attempt; it is never a gap.
3. Outcome states and attempts.
   - `unresolvable` is terminal until a later maturation supersedes it. After the
     deadline it requires the barrier unresolved on the usable path, the first gap
     proven, and one of:
     - cessation: a merger or reorganization naming the point-in-time ticker as acquiree,
       or its worthless removal, effective between the entry and the first gap; or a
       point-in-time membership removal effective in that interval (the sub-reason names
       the evidence);
     - `interior_gap`: a proven gap followed by usable bars (a halt). The served label
       cannot cross it, and the trainer drops such rows too.
   - A proven tail gap without cessation evidence stays pending
     (`stock_gap_without_cessation`) and turns overdue for operator action.
   - Attempts become an append log: `attempts/<key>/<sequence>.json`, the sequence taken
     under the key's lock; the latest decides. An attempt is written only when its status,
     reasons or missing intervals differ from the latest. Attempts record the grace, the
     drift policy hash and the receipt ids they used.
   - Membership removal needs the point-in-time membership authority: `mature-outcomes`
     reads it from the live-input root through its pointer. Until the publisher (after
     part 3) writes one, membership removal is unavailable evidence, never assumed.
   - The worker keeps unresolvable intents in the index and re-evaluates them every run,
     so new receipts can supersede them. `_route_oldest_pending` skips an intent whose
     latest attempt is `unresolvable`.
   - `mature-outcomes` reads the collected bars and receipts instead of a bars artifact,
     and the pinned drift policy for the grace.
4. Reporting.
   - Cohort rows add `unresolvable_selected_samples`. The unresolvable share is
     unresolvable / (matured + unresolvable) over selected outcomes whose deadline has
     passed.
   - The drift policy gains `maximum_unresolvable_share = 0.05`. After the minimum
     matured samples, a larger share makes the route not ready
     (`unresolvable_share_exceeded`).
   - A diagnostic sensitivity that never gates: the mean excess return with each
     unresolvable outcome filled by (a) its last usable close and (b) a delisting proxy:
     the cash `rate` for cash mergers; -30% for NYSE and NYSE American and -55% for Nasdaq
     removals that are not mergers (Shumway 1997; Shumway and Warther 1999); -100% for
     worthless removals; the last usable close for stock mergers and interior gaps. The
     listing exchange comes from an Alpaca asset receipt fetched for each cessation case.
5. Delivery in three parts, each with its own diff review: (2b-1) sources and collection
   with receipts and loader; (2b-2) classification, states, the attempt log, the worker
   and the command; (2b-3) reporting and drift. Tests: settled and unsettled receipts, a
   renamed symbol, an absent and an empty symbol, an unusable bar as a gap, an interior
   halt, each cessation sub-reason, attempt order and change-only writes, supersession,
   route pending exclusion, the ceiling with few samples, and the sensitivity values.

Consolidated review decisions for part (2b) (September 28; neither design review found a
blocker; these supersede the design above where they differ):

- One price basis per path (majors in both reviews). Each receipt reflects the price
  adjustments as of its own retrieval. For each decision session and symbol, the whole
  path, the decision bar included, comes from exactly one receipt: the latest one whose
  page chain completed. A newer receipt replaces the whole path, never single sessions.
  The name-change fallback is dropped: a request for FB with `asof` 2022-06-01 returned
  every session through 2022-06-14, after the rename to META on 2022-06-09 (measured
  September 28), so `asof` on the decision session already follows renames. A rename it
  did not follow would leave a gap without cessation evidence: pending, then overdue for
  operator action. A duplicate means two rows for one session within one response; it
  blocks as a data defect. Re-collection never creates duplicates.
- Settlement before the deadline (code major). A gap is proven by a complete receipt
  retrieved at least `outcome_settlement_days` (3) after the horizon's last close, a new
  drift-policy field that must be shorter than `pending_grace_days` (7). An ordinary
  cessation then becomes `unresolvable` before its outcome could turn overdue. The nightly
  order is collect, mature, report, drift.
- Unresolvable leaves pending in the same part (code major). In part (2b-2), an intent
  whose latest attempt is `unresolvable` leaves the window's pending set,
  `pending_selected_samples` and the oldest-pending fields as well as the route index scan,
  and cohort rows count it as `unresolvable_selected_samples` with the number of distinct
  securities.
- Interior gaps must be corroborated. Before a proven gap with later usable bars becomes
  `interior_gap`, one-minute bars for that session are requested with a receipt. Any bar
  means the stock traded, so the gap is a collection defect: pending, for operator action.
- Cessation evidence. The corporate-actions query filters on the provider's process date,
  so it spans process dates from the decision session to the earlier of today and the
  first gap plus 90 days, and is repeated on later runs. Evidence counts when its effective
  date lies in [decision session, horizon's last close plus the grace], inclusive. The
  acquiree or removed symbol must match the symbol in use at the first gap, after any
  followed rename.
- The ceiling applies after evidence sufficiency (the effective periods), not after the
  minimum matured samples, since outcomes of one acquired stock cluster.
- Sensitivity, still diagnostic. The -30% and -55% fills are named a stress, not a proxy,
  since removals that are not mergers are broader than Shumway's performance-related
  delistings. Stock mergers use the acquirer rate times the acquirer's close on the
  effective date (its daily bar requested with a receipt), and stock-and-cash mergers add
  the cash rate. An interior gap continues the managed path after the halt, a stop crossed
  during the halt filling at the lower of the stop and the first open after it
  (`executable_fill_price`). Each fill's benchmark runs from the entry open to the fill's
  date. The listing exchange is taken at decision time: part (3) and the live-input
  publisher record it on the intent; until then, and whenever it is unknown or OTC, the
  -55% fill applies. No asset lookup after cessation.
- Receipts keep every page's full response metadata (status, requested and final URL,
  redirect chain, retrieval time, headers, content type, body length and hash) so that
  loading rebuilds the response and re-runs the shared decoders. The rebuild helper lives
  in `evidence`. A receipt's retrieval times lie within its attempt; a non-200 response,
  a redirect or a broken page chain is recorded as a failed receipt that never settles; a
  page-count cap bounds a chain. "No bars" for a symbol is decided only by a complete
  chain. Alpaca answers `{"bars":{}}` when no requested symbol has bars and omits a symbol
  without bars from a response that has others (measured September 28 for TWTR after its
  delisting); a `"bars": null` response fails the shared decoder and so the receipt. A
  bar whose timestamp is not an XNYS session fails the receipt.
- Re-collection stops. A pending gap unit is collected daily until it settles; an
  unresolvable or overdue one weekly until 90 days past its deadline, then it is frozen
  (a manual rerun can still supersede it).
- Reuse without coupling. `sources/alpaca_corporate_actions.py` is used unchanged (one
  retry, no raise on status, so the collector records failures). The research collector
  (`swing/datasets/corporate_action_collection.py`, pinned by corrected-outcome evidence)
  is neither imported nor edited; its patterns are copied. Real pages from
  `data/raw/swing_full_cohort_corporate_actions` serve as classification fixtures.
- The monitoring lease is a new module on `locking.file_lock`, not an edit of the pinned
  `heavy_jobs.py`. Registration, collection, maturation and reporting share it with a
  bounded wait, so a nightly step waits for the previous one instead of failing.
- Added tests: receipts that differ by an adjustment factor; a settled cessation that never
  makes the route not ready; window pending without unresolvable intents; a merger
  processed after the grace; a failed or truncated chain; a no-data response; receipt or
  body tampering and a missing body; lease contention; the re-collection stop rule; the
  halt corroboration; recorded real corporate-action pages.

Part (3) registration and population: implementation design (September 28; for review
before code)

It implements the consolidated decisions "Registration never waits on drift", "Population",
"Snapshots record a scope", "Registration lease", "Session records", "Identities" and
"Sector peer floor". Measured facts it builds on:
- The live path drops a member from the model frame when it is feature-ineligible or its
  sector has fewer than `minimum_cross_section_for_ranking` (30) eligible peers
  (`serving/swing_features.py` `_model_frame`); such a member then abstains as
  `out_of_universe`, which is wrong: it is a member. Several S&P 500 sectors hold fewer
  than 30 members, so this is common, not rare.
- Members excluded for market or catalyst inputs abstain as `live_inputs_incomplete` and
  count toward the 5% ceiling (`_validate_live_security_exclusions`).

1. Scoring the cross-section (`serving/prediction_service.py`).
   - The request path and a new `predict_swing_cross_section(as_of)` share one scoring
     function. The cross-section scores every effective point-in-time member, requires a
     verified promoted generation with `promoted_at_utc <= as_of` and valid live inputs,
     and skips only the actionability (drift) gate; serving to clients stays gated.
   - Every member gets a prediction or an abstention with a reason:
     - `live_inputs_incomplete`: excluded for market or catalyst inputs (counted in the
       5% ceiling, as now);
     - `sector_peer_floor`: feature-eligible, but its sector has too few eligible peers.
       When the sector reaches the floor only with its excluded peers, the drop is
       cascaded from an input exclusion: it is recorded as `live_inputs_incomplete` and
       counts toward the ceiling;
     - `insufficient_history` (proposed): feature-ineligible for warm-up or history
       reasons (a new constituent). It is neither an input failure nor a peer-floor drop;
       the review should confirm this third reason or fold it into one of the two.
   - `out_of_universe` remains only for a requested ticker that is not a member.
2. Public contract: API `market_predictor.prediction.v4` through the append-only change
   log. It adds the new abstention reasons and removes `PredictionRowEvidence.decision_atr`,
   which nothing reads since intents take the ATR fraction. The golden fixture is
   regenerated and TradingFlow's handoff names the one change it must adopt.
3. Snapshots (`serving/snapshot_store.py`). A snapshot records `scope`: `request` (as
   now, audit-only) or `decision_cross_section`: the as-of time, the route and release,
   the member-set hash and the full response, with no request ticker list, so the
   100-ticker request limit does not apply.
4. Registration (`register-session-predictions --as-of`, production CLI only).
   - Takes the monitoring lease (shared with collection and maturation, a bounded wait),
     the admission lease and the 90% memory guard, never the heavy-job lease.
   - Scores the cross-section, records its snapshot, then its intents (scored members)
     and observations (every member, abstentions included), then the session record,
     written last as the commit marker.
   - `register-outcome-intents` accepts only `decision_cross_section` snapshots.
5. Session records (outcome repository, `session_records/<session>/<route key>.json`).
   - Status `registered` (members, scored, counts by abstention reason, the snapshot id and
     content hash, the member-set hash) or `failed` (`exclusion_ceiling_exceeded`,
     `inputs_unavailable`, `model_unavailable`, `registration_error`, and `not_run`
     written by an operator), with `recorded_at_utc` outside the identity.
   - An identical rerun is accepted, a different one refused; a `failed` record may be
     replaced by `registered` from a retry for the same session.
   - Route activation, per release, is the first XNYS session whose decision cutoff is at
     or after the generation's `promoted_at_utc`.
6. Population and drift (`governance/outcomes/performance.py`, `drift/policy.py`).
   - Rates, sufficiency, the curve and the rank check use only cross-section observations
     from `registered` sessions, one cross-section per route and session.
   - The report counts the route's sessions in the window since activation, and those with
     no record or a `failed` one. Drift is not ready when a session older than the grace
     has no record, and warns when registered sessions fall below
     `minimum_registered_session_share` (0.95) of the window.
   - Cohort and report identities also bind the session-record ids and the ids of the
     deciding attempts.
7. The live-input publisher follows this part (the user's decision). Until it exists,
   registration runs only against test generations; the listing exchange for the
   sensitivity's delisting stress joins the membership data there.
8. Delivery in three parts, each with its own diff review: (3a) cross-section scoring, the
   abstention reasons and API v4; (3b) snapshot scope, registration, session records and the
   cross-section-only intent registration; (3c) population, session coverage and drift,
   identities. Tests: a thin sector abstaining as `sector_peer_floor`; a cascaded drop
   counted in the ceiling; more than 100 members through registration and replay;
   registration while drift is warming or not ready; identical and conflicting reruns; a
   failed record replaced by a retry; a republished live generation not double counted; a
   session gap turning drift not ready; request snapshots refused by intent registration.



Part (3b) implementation `8fdb29f` (local; publication awaiting authorization): scoped
snapshots and deterministic decision ids, full-member observations, strict route and
session records, crash/retry recovery, production registration and audited not_run.
Request snapshots cannot register. Cross-section replay is not constrained by the
HTTP ticker limit. Session markers bind exact observations and intents; failed records
retain history and can retry, while committed records cannot be replaced. The local
consolidated review added model/evidence identity checks and exact session-cutoff
binding. Verification: 329 affected tests passed; after final review edits, the 24
re-affected registration/snapshot/intent/replay tests passed. Ruff clean on ten changed
Python files; strict mypy clean on eight sources. No full suite or live/provider run.
API v4 bytes unchanged from part (3a); C# migration is still separately pending.

Part (3c) implementation `116d711` (September 28, local; publication pending):
reports load only committed session inventories, preserving every member in the
denominator and rejecting missing or mismatched committed evidence. Coverage binds
each release activation to expected XNYS cutoffs and partitions registered, failed
and missing sessions. Missing sessions beyond the existing grace block actionability;
registered share below 95% warns without upgrading a warming route. Session-record
and as-of deciding-attempt ids bind cohort/report identities. Unknown unscored
metadata remains null in observations and is labelled unavailable in grouped reports.
Internal report fields are required; prior reports have no compatibility fallback.

Consolidated component review found two concrete consumer conflicts and one contract
conflict, all fixed in the same checkpoint:
- Major, `performance.py`/`repository.py`: a prior partial snapshot can own the shared
  semantic pending index. The committed session's intent could then disappear from
  the overdue check. Reporting now traverses exact committed intent ids, reuses loaded
  in-window intents and does not open old observation partitions. It checks outcome
  and attempt availability at report time. Route-wide source ids include the evidence
  consumed outside the rolling metric window; the metric population still uses the
  established maturity-aligned window. Regression tests cover a competing partial
  index and a future outcome for an older committed decision. This correctness cost
  is a sequential scan of committed intent/outcome metadata, not the old pending-only
  scan; no production-scale performance claim is made.
- Major, `serving/session_registration.py`: collection/maturation still require a
  canonical pending intent. A changed snapshot after a partial write must not commit
  unreachable intents. Registration checks existing semantic bindings before writing
  and verifies them before the final marker. Conflicting retries fail closed; the
  original immutable snapshot remains resumable without rebinding or deleting evidence.
  The new integration test verifies both the refusal and original-snapshot recovery.
- Moderate, `session_records.py`: route coverage must represent the existing supported
  swing/investment horizon syntax. It now uses the shared horizon pattern; unsupported
  drift evidence policies still fail closed. This adds no investment scorer or policy.

Verification tier: component. The reporting, drift, registration, repository, intent,
package and architecture run passed 323 tests. The subsequent reporting/CLI/maturation
run passed 88 with one invalid new fixture (its two availability clocks disagreed);
the fixture was corrected. After all final fixes, all 25 re-affected population,
registration and intent tests passed. Ruff clean on 11 changed Python files; strict
mypy clean on six source modules; diff checks clean. All test processes completed.
No full suite, training, provider collection, sealed-data read, live registration,
deployment or promotion ran. API bytes unchanged from (3a); the separately owned C#
consumer migration remains pending. The publisher does not exist yet.

Monitoring part (3a)/(3b)/(3c) is implemented and locally verified. Formal checkpoint
closure still awaits publication: automatic approval review rejected the continuity
document push to the GitHub destination without explicit user authorization. Do not
retry or work around that rejection before authorization. Local commits after
`b3c8761`, including this code receipt, remain unpublished.

Part (3c) freeze: performance reads only the exact source-id inventory of registered
session markers and fails on missing/mismatched committed rows; partial/uncommitted
rows do not enter any rate, outcome statistic or route-pending gate. Report session
coverage from each verified per-release route activation over XNYS decision cutoffs,
including registered, failed and missing sessions. Missing sessions older than the
existing pending grace block actionability; registered share below the existing 95%
policy warns without upgrading an insufficient route. Bind session-record ids and
as-of deciding attempt ids into cohort/report identities. Keep unavailable unscored
metadata null; no fabricated features or outcomes. Preserve the existing matured-window
accounting and leave estimator statistics/curve/rank redesign to part (4). Test partial
writes, exact population denominators, missing committed evidence, coverage since
promotion, failed/not_run records, grace boundaries, re-promotion isolation, as-of
attempt changes and tampering. Prior raw-record accounting fixtures must explicitly
commit synthetic sessions; production has no legacy population fallback.

Part (3a) implementation `b3c8761` (September 28, pushed): shared request/internal
scoring, complete typed membership, natural peer-floor abstentions, cascaded input
failure accounting and API v4. Empty eligible frames abstain without an estimator
call and retain the decision cutoff. Internal scoring takes admission and bypasses
only drift. The five-outcome fixture removes unused row-level `decision_atr`.
Component verification: 302 passed, 1 opt-in memory benchmark skipped; live features,
prediction API/service/snapshots, outcome intents, replay, package/architecture and
continuity tests. Ruff clean on seven changed Python files; strict mypy clean on four
source files. Local consolidated diff review found no remaining supported defect.
Existing C# binary: 38 passed, 1 producer-fixture parity failure. C# source still
consumes the retired unversioned contract. Its handoff records the v4 adoption/hash;
consumer integration remains pending with TradingFlow's owner. No full suite,
training, provider or sealed-data run; no owned worker remains. No readiness claim.

Part (3b) implementation freeze: strict request/cross-section snapshot scope;
canonical decision identity separate from audit timestamps/ids; durable route and
session records. Bind full membership, response and source identities. Identical
reruns reuse the first immutable audit snapshot; changed payloads fail. Under the
monitoring lease, write a verified route activation anchor before scoring, intents
for scored members and observations for every member, then a session commit marker
binding exact record ids. Unavailable unscored feature metadata stays null. Partial
writes are resumable but cannot authorize report population. A registered session
cannot be replaced; failures can retry. Request snapshots remain audit-only. Replay
reads either scope without constructing a >100-ticker public request. Add production
registration and an audited not_run operation against a known route anchor.
Exit tests: deterministic/reordered retries, changed payload/source rejection,
>100 members, all-abstention, drift-blocked registration, request refusal, crash
before commit, failed retry and tampering. Part (3c) consumes only committed sessions
for population/coverage/drift. Publisher, training and collection ownership are outside
this slice. The documentation push was rejected by automatic approval review pending
explicit user authorization; local work may continue and publication stays pending.

Part (3) continuation review (September 28, against `21ef677`):

This is a local, code-grounded design review, not a claim that the prior two
independent reviewers have approved this implementation design. At that review baseline no part (3) code,
API v4, session registration command or live-input publisher had landed. The review required
the following corrections in the bounded implementation designs before their code.
These corrections supersede the affected wording above. At that historical baseline the public contract
and golden fixture were v3; the implementation receipts above supersede that status.

1. Major: deterministic registration identity is underspecified. In
   `serving/snapshot_store.py`, `record` hashes `recorded_at_utc`; the response also
   includes generated time and random request/correlation ids from
   `serving/prediction_service.py::_edge_swing_response`. Excluding only the session
   record's recording time cannot make identical reruns identical. A direct call to
   `_content_sha256` with unchanged request/response and two recording times produces
   different hashes. Define a canonical decision payload, separate from audit metadata,
   for cross-section/session identity. Bind route, release, decision, sorted member
   identities, scores, abstentions and source evidence; preserve the original audit
   snapshot and its byte hash. A changed source pin or decision payload must conflict
   or resolve to the already committed session without adding observations; it must
   never silently replace that session. Test identical reruns with different wall
   clocks and request ids, changed scores/source pins, reordered member input and a
   republished live generation. Request audit snapshots may retain their existing
   per-call identity. This is part (3b), not a reopening of request snapshot storage.
2. Major: `feature_eligible == false` does not prove a genuinely short listing history.
   `_select_complete_current_cross_section` currently counts cold/ineligible members
   toward the exclusion ceiling. `swing/features/eligibility.py` also sets the flag
   false for missing warm-up sessions and unavailable sector benchmark features.
   A new constituent can have a long prior price history. Blanket reclassification as
   `insufficient_history` would exempt provider gaps from the governed 5% ceiling.
   For part (3a), fold this proposed reason into `live_inputs_incomplete` and retain
   the current ceiling behavior. A future non-failure history category needs its own
   verified coverage/listing evidence and reviewed rule. Classify peer-floor cascades
   using the full effective membership's point-in-time sector identities before
   exclusions; count the union of direct and cascaded failures once against that
   unchanged membership. Tests: short/missing stock history, missing benchmark,
   naturally thin sector, and a 30-member sector falling to 29 after one input failure.
3. Major: the design needs an explicit evidence path for unscored members and an
   all-abstaining cross-section. `_model_frame` raises when nobody meets the floor;
   `live.context` retains only scored members. `serving/outcome_intents.py` skips
   unscored abstentions and refuses an empty observation set; its observation builder
   requires feature-derived regime/cap/liquidity metadata. Therefore reusing this
   response alone cannot implement "observations for every member". Part (3a) must
   return the complete verified member identity/reason set and the actual session
   cutoff independently of the scored frame. An otherwise valid all-thin-sector
   cross-section abstains without calling an estimator on empty input; failed source
   readiness still fails. Part (3b) must bind member evidence into the snapshot and
   represent unavailable observation metadata explicitly, without fake feature times
   or invented categories. Intent creation remains restricted to scored members.
   Test all members represented exactly once, all-abstention registration, unavailable
   metadata, unchanged selection among scored members, and crash/retry before the
   final session commit marker. Part (3c) must exclude all partially written sessions.

Implementation constraint confirmed: `PredictionRequest` rejects 101 tickers, and
snapshot loading currently validates that type. The shared scorer must take an
internal cross-section context rather than construct or bypass validation on a public
request. Keep the HTTP request limit; test more than 100 members through scoring,
snapshot load and registration. Preserve promotion-time checks, admission, memory
guards and generation-change checks; only the internal monitoring path skips drift.

Verification for this review: existing continuity, snapshot and live-feature tests,
24 passed, 1 skipped (26.83 s); one warning from the deliberate non-finite payload
test. Direct read-only diagnostics confirmed recording-time hash differences and
101-ticker request rejection. These are baseline checks, not evidence that the new
exit gates pass. No training, sealed-data reads, provider requests, full suite, code
lint/types or C# checks ran. Post-edit continuity tests: 2 passed (0.15 s); `git diff --check` clean.
The skipped test is the opt-in production-scale RSS benchmark. Rollback is a scoped revert of these two continuity
documents; runtime admission remains unchanged and fail-closed.

The September 20 user instruction explicitly extends the completed HTTP/CLI and
TradingFlow cleanup to all remaining Market Predictor implementation. This is a
changed requirement, not a reopening of previously passed tests without cause.

Frozen scope and order:

1. Inventory executable consumers and hash-bound evidence. Resolve how historical
   mixed strategy contracts remain verifiable before changing their bytes.
2. Move retained observed news/SIP/history collection to source/evidence ownership;
   replace its intraday training-dataset identity dependency with a verified
   security authority. Preserve original receipt clocks and resume semantics.
3. Make current strategy, prediction, release, outcome, monitoring and research
   governance contracts swing-only. Remove retired modes rather than add aliases;
   changed semantic identities require explicit new versions and fail-closed
   admission of incompatible historical artifacts.
4. Delete dedicated intraday models, training, datasets, labels, features,
   specialist research, unused configurations and their exclusive tests only after
   retained consumers are migrated. Preserve shared swing regression coverage.
5. Update README, architecture, implementation guide, feature audit and continuity
   documents in place. Historical evidence is not an active implementation guide.
6. Run import/reference scans, CLI/API rejection tests, causal/lineage and swing
   regression tests, full pytest, Ruff and strict mypy; independently review the
   consolidated diff and push only verified checkpoints.

The user approved historical retention on September 20, limited to successfully
completed swing runs. Completion must be established from the run manifest and
artifact integrity; it is not profitability, promotion or serving admission.
Failed, interrupted and no-candidate outputs do not qualify as retained models.
Reference-bound rejection metadata remains only until its references are retired.
Preserve original qualifying manifests and model bytes; publish new verification
evidence separately under the swing-only contract, never rewrite historical pins.

Preserve raw news/candles and qualifying hash-bound swing research artifacts. The
swing open-to-close feature named `intraday_return` and minute/hourly transport
are not day-trading strategies. No broker/provider job, data deletion, model
retraining, cloud deployment or promotion is implied. Do not silently repin old
manifests, execute archived code as a fallback, or turn historical research into
current admission. Contract-migration decisions that invalidate saved swing
acceptance must be made explicit before implementation.

Exit gates: no active dedicated day-trading imports, CLI/API paths, model modes,
strategy configs or current design recommendations; retained subdaily swing
evidence and causal contracts pass regression tests; incompatible retired
artifacts fail before activation. Rollback uses the preserved Git checkpoint,
never mutation or deletion of protected evidence. Shared news collection ownership
is now authorized before the remaining cleanup; no live collector has started.

Independent design findings (September 20):

- The mixed strategy contract hashes both strategies. Removing its intraday fields
  changes identity even when swing parameters are identical. Saved swing
  publications cannot be made current by simply replacing their recorded hashes.
  User approval now permits completed swing publications as historical evidence
  and requires fresh swing-only verification before reuse. Rebuild or retrain only
  when verification cannot establish equivalent usable inputs, not automatically.
- Four otherwise unused root helpers (`intraday_confirmation`,
  `intraday_enrichment`, `intraday_catalysts`, `intraday_universe`) are pinned by
  the protected KS3 swing specialist request. Preserve their referenced bytes or
  establish the approved historical-only disposition before deleting them.
- Independent S&P membership authorities exist, but the prospective news
  collector currently verifies an intraday bar dataset to obtain its namespace.
  A new universe-owned namespace must pin membership and asset identities,
  intervals, observation cutoffs and extension lineage. No silent conversion of
  previous polling chains to the new schema.
- Ownership: Alpaca transport in `sources`; immutable receipt/attempt/generation
  and SIP-session plan verification in `evidence`; membership/security identity
  in `universe/sp500`; analyst event classification in `catalysts/issuer_events`.
  Commands compose these layers. Do not move universe/model imports wholesale
  into `sources`, whose package boundary prohibits them.
- Collector tests must cover namespace substitution, ticker reuse, future
  membership, asset conflicts, exact receipt clocks, tampering, interrupted
  publication, duplicates and resume. Preserve SIP and completed-session checks.

The approved bounded retention and unused-module slice is complete below. Consumer
migration and swing-only current-contract replay remain required before full
retirement/reuse closure, but are now deferred rather than blocking the new raw
collector. Full retirement is not complete and no production readiness is claimed.

### Completed Retention And Unused-Code Slice

Implementation `fe0ed86` is pushed. September 20 cleanup removes 14 unused
day-trading source files and 12
exclusive test files, retaining five shared ranking/calibration tests. Removal
guards prohibit their reintroduction; obsolete generated bytecode was cleared.
The independent reviewer confirmed no remaining active consumers of that slice.

`verify-retained-swing-run` checks historical return-run integrity without training,
deserialization, data mutation or admission. All 18 units and three original root
pins pass. Of 266 source pins, 256 match and ten current implementation hashes
differ; the separate report records them rather than repinning old evidence.
The older technical swing candidate and its two reports also match their pinned
manifest. Preserve these completed models; no-candidate outputs are not models.
Original artifacts, source market data and mixed historical contracts are unchanged.

Independent consolidated review closed after exact readiness-source bindings and
the conflicting old catalog reuse permission were corrected. Final verification:
4,442 tests passed, ten skipped, 268 warnings in 1,742.89 seconds. Repository Ruff
and strict mypy pass (388 source files). Log:
`data/runtime/swing-retained-cleanup-20260920.log`; JUnit:
`.test-tmp/swing-retained-cleanup-20260920.xml`. Both agents and the test process
are closed. Sampled system memory stayed below 75%; no collection/training job ran.

Remaining retirement work is unchanged: shared collector/security-namespace
migration, current swing-only strategy/serving/governance contracts, removal of
their remaining dedicated intraday consumers, and fresh numerical/causal replay.
Historical file integrity does not close those gates or grant model reuse.

### Completed Main Preservation And TradingFlow Retirement

On September 20, Market Predictor `main` was fast-forwarded to `18e07d1` (191
commits). TradingFlow's existing source work was preserved in `9d50bc1`, then its
remaining day-trading cleanup was merged and pushed on `main` as `cb747da`.
Both repositories use fresh `unified-swing-product` branches. Local settings,
runtime files, databases and market/model artifacts were excluded from Git changes.

TradingFlow's dedicated strategy/config/runtime paths are retired; remaining
design workflows were removed and shared swing regression coverage restored.
New risk reservations must be swing; durable dispatch also rejects unsupported or
missing risk horizons before a fresh entry POST. Existing broker-order adoption,
reconciliation, exits and protection remain available. Independent review found no
blocking issue. Verification: 97 focused C# tests, 1,546 full offline C# tests and
four prototype-state tests passed. The live Alpaca integration test was excluded;
no provider or broker requests ran. Both review agents are closed.

Market Predictor code and pinned evidence remain unchanged. Its internal historical
domain retirement still requires the separately reviewed contract/collector
migration, not a blanket package deletion; this checkpoint does not claim that work
complete. Shared news collection resumes next from the inventory below.

### Completed Day-Trading Command Retirement

Implementation `9c32ce1` is pushed. This bounded checkpoint removes dedicated day-trading
command adapters, not every internal historical implementation in one change.
Remove training, promotion, setup, dataset and specialist-collection commands,
including the old cross-sectional command group. Move its shared S&P collection
and event-extraction commands unchanged into `commands/sp500_sources.py`.
Restrict production feature/bundle publication to swing; reject retired model
activation before changing an active pointer. No compatibility aliases.

Preserve shared minute/hourly transports, prospective news/SIP evidence, canonical
bar availability clocks, swing `intraday_return`, raw archives, immutable artifacts
and all training code pins. Reference-bound internal domain modules remain until
their consumers can be removed or migrated without rewriting historical evidence.
TradingFlow's unrelated dirty tree is outside this checkpoint.

Exit gates: retired commands absent and fail as unknown commands; retained swing,
source and S&P commands remain; unsupported production modes/models fail before
publication or activation; independent plan/diff reviews, focused poison tests,
CLI/package checks, full pytest, Ruff and strict mypy pass. No provider or broker
job starts. Rejected activation leaves active pointers unchanged. Local publication
prechecks the promoted candidate and publishes without activation; it rechecks the
exact immutable release before activation. A concurrent source replacement may
leave rejected, nonactive evidence, which must not be silently deleted. Historical
release verification stays read-only and does not confer current admission.

Independent plan review: Pauli and Newton approved this bounded adapter scope.
Newton verified 26 training, 67 predictor-replay and 26 outcome-replay pins; none
directly binds an edited command adapter. Mixed strategy contracts stay untouched.

Implementation and verification complete: 44 removed commands,
nine deleted adapters; two shared S&P commands moved unchanged. Focused checks
passed 117 and 24 admission tests. Final full suite: 4,441 passed, ten skipped,
269 warnings in 1,743.58 seconds; session 86175 exited zero. Ruff and strict mypy
passed again (405 sources). Replacement reviewers Ramanujan and Curie completed
the interrupted diff review without blocking findings and are closed. No owned
worker remains. The saved model's three root manifests and 26 directly bound
code/config pins match their original hashes. No data, feature or model changes.

### Shared News Collection Ownership And Durable Replay

Completed bounded deliverable: connect the existing raw-news receipt contract to one
designated local collection owner and reproducible consumer import. Preserve the
existing Python/C# receipt wire format. Raw symbol queries are not issuer identity,
point-in-time feature authority, evidence of complete provider coverage, or model
admission. Do not require an intraday dataset merely to preserve HTTP observations.
One local root coordinates cooperating workers; this is not multi-host fencing.
Swing and investment consume the same raw evidence without sharing model targets.

Frozen implementation: a strict independently pinned plan defines windows, owner,
producer revision and page/attempt budgets. Record a logical transport attempt
before fetching (the HTTP client may retry internally), publish original receipt
bytes atomically, then commit the receipt-bound attempt result. Recover progress
only from verified results. Preserve indeterminate attempts after interruption;
exactly-once provider execution across a crash is not promised. Bound reads and
pagination, reject cycles, isolate request failures, and fail closed on corruption.
The collection CLI must support no-network verification of an existing run.

Exit gates: focused restart/ownership/tamper/CLI tests, existing Python/C# receipt
parity, repository lint/types/full Python regression, independent code/design
review, updated continuity. No live provider/broker jobs, training, raw-data
deletion, investment horizon invention, cloud deployment or promotion in scope.

Implementation `94aa1c4` is pushed; full verification and scoped review are complete.
TradingFlow's matching documentation is saved locally as `afaafc0`, not pushed;
its repository requires explicit authorization for GitHub synchronization.
Focused checks pass: 101 collection/transport/CLI tests, 232 CLI/package/continuity
tests, 27 existing C# receipt tests; repository Ruff and strict mypy pass (392
source files). Consolidated independent review closed after preventing HTTP
redirects before they are followed and restricting transport-failure classification
to actual request errors. Partial-write, owner-substitution and mapped-drive tests
were added. Both agents are closed. No live provider/broker request was made.
Final full regression: 4,488 passed, ten skipped, 268 warnings in 2,902.59 seconds.
Log: `data/runtime/shared-news-final-20260920.log`; JUnit:
`.test-tmp/shared-news-final-20260920.xml`. The 90% watchdog sampled a 68.51% peak;
both test and watchdog processes exited zero. An earlier background launch had
setup errors and was stopped. A subsequent direct run caught a CLI memory-rejection
exit-code regression; explicit MemoryBudgetError handling was restored before the
successful final full run. No failure was waived or counted as passing.

- Reuse `sources/news_exchange.py` and the existing Alpaca observed transport;
  preserve original response bytes, query identity and actual receipt clocks.
- Give each logical page attempt durable identity. Commit progress only after its
  receipt is durably published; resume must verify the saved identity and content.
  Failure of one ticker/window must not corrupt another or create false coverage.
- Use a single configured collector owner; TradingFlow remains owner of its fenced
  live market stream. Do not create a second collector merely to satisfy each app.
- Consumers import genuine observation provenance and pinned receipts, never
  manufactured collection-plan/attempt hashes. Existing historical availability
  limitations remain explicit; importing old news today is not old first-seen proof.
- Keep process roles and storage boundaries cloud-portable. Local coordination is
  not a distributed lease; unsupported multi-host ownership must fail explicitly.
  Actual cloud deployment remains outside this checkpoint.
- Test duplicate workers, interruption/restart, partial writes, repeated pagination,
  rate-limit failures, tampered receipts, bounded memory, clock preservation and
  Python/C# fixture parity. Do not run provider/broker jobs before the design gate.

Review starting points: `catalysts/issuer_events/alpaca_news_collection.py`,
`sources/news_exchange.py`, `evidence/news_exchange.py`, and TradingFlow's
`docs/research/evidence-repository-design.md` plus its actual collection/import
implementations. Protect pinned historical helpers; inventory remaining mixed
intraday domain dependencies before moving shared prospective source ownership.

### Completed Swing Public Boundary And News Receipt Exchange

Market Predictor implementation `1fe9533` is pushed. Both bounded deliverables below
are implemented and locally verified across Python/C#. TradingFlow's corresponding
changes were initially in its uncommitted working tree, not in that Python commit.
They are now preserved in TradingFlow `9d50bc1` under the user's September 20 merge
instruction. This changed user requirement
supersedes the previously paused intraday scope:

1. Remove intraday/unified prediction HTTP routes and intraday research catalog
   entries. Public requests accept only the existing swing horizon. TradingFlow's
   latest working-tree client already requests swing. Retain promotion, identity,
   authentication and abstention checks. No redirects or compatibility aliases.
2. Implement one bounded Alpaca raw-news HTTP receipt exchange: exact body bytes,
   recorded query identity, original receive clock and independently pinned file
   import, with shared valid/invalid Python/C# fixtures. Do not add a parallel
   normalized news/bar schema. It is a transport/import foundation, not a running
   shared collector, normalized attribution, new feature or model admission.

Exit gates: removed routes absent from OpenAPI and return 404; unsupported modes
and horizons fail before inference; research catalog exposes only swing; missing
models remain unavailable; both languages agree on fixture acceptance and causal
cutoffs; malformed/tampered evidence fails closed. Run focused tests, independent
plan and diff reviews, full Python tests, Ruff/mypy and relevant C# verification.
Preserve raw archives, immutable research artifacts, account state and all unrelated
TradingFlow working-tree changes. Do not start provider streams or broker jobs.

Implementation verification completed 2026-09-17: 4,372 Python tests passed,
ten skipped, 271 warnings; Ruff and strict mypy passed (412 source files).
TradingFlow's offline suite passed 1,482 tests, excluding its unavailable live
Alpaca integration test. Both independent reviewers closed their supported findings.
The first full Python run exposed one missed active-model inventory reference;
the inventory now retains intraday artifacts as historical evidence, without
changing their bytes or hashes. The final full run passed after that correction.
Both review agents and all owned verification processes are closed. This checkpoint
does not certify full intraday retirement, common collector operation, cloud readiness
or an admitted profitable model.

Program order after command retirement:
- Connect the evidence exchange to one designated collector per source and durable
  replay; retain TradingFlow ownership of the live market stream. Move or retire
  remaining mixed historical domain code only after its consumers/pins are audited.
- Cohort-specific portfolio lots, risk, approvals and holding policies in TradingFlow.
- Full-content news/SEC attribution, verified company relationships and cited RAG.
- Causal feature/outcome publications, sequential fits and funded SPY evaluation.
- Admitted forecast serving, candidate/holding watcher, portable process roles and
  end-to-end paper verification. Actual Azure/GCP deployment remains separate.

Both long-term forecast horizons (63 and 252 sessions) are approved as separate
targets (September 25); numerical allocation/risk budgets remain explicit user
decisions, not implementation defaults. TradingFlow's local commits `afaafc0`,
`5217fad` and `8525f80` were pushed to its feature branch on September 25. Detailed approved product rationale
is in the unified investment product proposal; this remains the sole execution plan.

## Long-Only Swing Research And Implementation Plan

Research step: **complete the bounded swing return-model comparisons** (`pending`
while the unified product boundary is implemented).
The user's combined implement/verify/train checkpoint is complete in `07963cc`
(pushed): two existing-technical specifications, 16 independent fold/scope fits
and two final research models. Four specifications still need the distinct
relationship/reaction profiles described below; they are not part of the completed
two-model delivery. Preserve the verified inputs and the approved 90% system-memory
ceiling without an independent absolute free-RAM floor.

### Completed Percentage-Only Memory Policy

Implementation `7bb805b` is pushed. Independent design/diff review is closed;
Ruff and strict mypy passed. Full verification passed 4,224 tests, with 10 skips
and 133 warnings in 1,755.08 seconds. The configured real audit passed and all
data diagnostics are unchanged. Its new pin is recorded in the handoff.
The user approved a 90% total physical-memory ceiling with no separate 2 GiB
free-RAM floor. Preserve the swing process budget of 5 GiB and unavailable-memory
measurement failures. Scope: the current readiness audit and the forthcoming
return consumer, not unrelated source collectors. Bind the percentage in the
strict request configuration. Do not modify hash-bound replay dependencies.
Verified below/at/above 90%, less than 2 GiB free while below 90%, invalid settings,
unknown measurements, process limits and source pins. Prior receipts are preserved.

### Completed Training-Input Audit

Implementation `b3b2372` is pushed. Scoped independent review, real-data
diagnostics, 4,202 full tests (10 skips), Ruff and strict mypy passed. All 58
post-format focused tests passed. The final receipt refresh completed in session
56011 after memory recovered to 66.5%; every diagnostic equals the earlier
successful report. Only the expected whitespace-only implementation pin differs.
Current verified receipt and SHA256 are in the handoff. No model fit is claimed.

Bounded scope and exit gates:
- Add one reproducible, pinned-input readiness command with monthly bounded reads,
  the workspace heavy-job lease and existing 5 GiB/system-memory guards.
- Verify frozen initial-fit sessions, exact profile populations, ordered model
  columns, feature clocks and exact ten-session label maturity. Reconcile stock
  net returns and benchmark excess without changing values or adding costs twice.
- Report feature availability separately from usable supervised labels, by month,
  year and sector. Preserve every published decision and missingness; a future
  outcome cannot change predictor eligibility. No imputation or new exclusions.
- Record the three planned research profiles separately from the two published
  datasets. News counts/sentiment are not qualified issuer/SEC reaction features.
- Update the current acceptance matrix with actual training-consumer, live and
  source gaps. A report is diagnostic, not a training/production admission token.
- Focused poison tests, independent design/diff review, Ruff, strict mypy, full
  suite, real saved-data audit and pushed implementation/documentation closure.

Out of scope: repeat source collection/replays, change immutable publications,
relax eligibility flags, fit the retired classifier/ranker or claim profitability.
Malformed/tampered data aborts without a success report. Unavailable inputs are
reported explicitly. Validation/test publications and actual fitting remain
uncompleted; ready technical profiles must not wait for blocked catalyst features
once their objective-specific training consumer and acceptance gates exist.

### Completed Technical Swing Return Models

Status: complete in `07963cc` (pushed), including real historical fitting. Bernoulli
(`01a0a07e-fb40-7031-b2d7-a9166dd328c9`) reviewed the training design read-only and
is closed. Leibniz independently closed the implementation review after all three
supported findings were fixed and tested. The choices below were frozen before
the first two technical specifications were fitted; preserve them in comparisons.

- Start with existing_technical only: the published technical_market profile's
  exact 120 inputs and clocks. Do not rename aggregate catalyst features as the
  still-unimplemented qualified issuer/SEC reaction profile.
- Use four expanding inner-development folds on the complete 1,231-session
  initial-fit calendar, with at least 503 training sessions and ten intervening
  embargo sessions. Scoring blocks are 179/179/179/181 sessions. Purge
  training rows by actual label maturity as well as session separation; freeze
  dates before inspecting target availability.
- Fit two independent scopes per learner/fold: full-universe temporal prediction,
  and security-transfer prediction with the stable 20% security holdout removed
  from both preprocessing and learner fit. Preserve the existing unseeded SHA256
  security-ID threshold assignment; the estimator seed 42 is a separate fact.
  Pin actual holdout IDs. Scope/fold refits are not extra specification searches.
- Train only on feature-eligible rows with available matured fixed-horizon
  supervision and required comparisons. Score feature-eligible rows regardless
  of outcome availability. Do not require complete model inputs or delete any
  published decision. Missing scored outcomes remain explicit in evaluation.
- Estimator encoding: fit medians on each training partition, retain
  all-empty columns using an explicit zero encoding, and append one missingness
  indicator per input. Zero does not assert observed neutrality or source coverage.
  Fit linear scaling only on training rows. Use equal total weight per fit session,
  normalized to mean row weight one, including weighted scaler statistics.
- Fixed learners: Ridge (alpha 1, intercept, lsqr, tolerance 1e-6,
  maximum 10,000 iterations) and squared-error XGBoost (hist, depth 3, 200 trees,
  learning rate 0.05, lambda 10, alpha 0, minimum child weight 1, full row/column
  sampling, one thread, seed 42). No parameter grid, early-stopping search,
  probability conversion or calibration. These frozen settings were used in every
  completed fit; no post-result parameter search was performed.
- Bind input/code/config pins, feature order, learned preprocessing, weights,
  exact folds, holdout IDs, library versions and row-level predictions in a
  distinct immutable research artifact. Reuse the pure fixed-horizon checks,
  canonical projected reads and shared lease; do not reuse the old binary/ranker
  fitter or disable its exposed-test protection. Serving/promotion remain false.

Required exit tests: future/validation/held-out poisoning leaves fit transforms
unchanged; missing inputs retain dimensions and rows; absent labels never change
input eligibility; holiday/maturity boundaries purge correctly; identity-based
holdout is stable; both scopes refit independently; costs are applied once;
tampered artifacts and out-of-range reads fail; research models cannot serve.
All scoped tests, the consolidated independent review and bounded sequential
historical fits passed. Full verification: 4,329 passed, 10 skipped; Ruff and
strict mypy passed (409 code/script files). Serialization/reload parity passed
for every real fit, followed by verification of all 18 units and 266 request pins.

Delivery scope: four folds times two independently fitted scopes times two
specifications, followed by two all-initial-fit research models. Each completed
unit is immutable and resumable with exact input/config/code/version pins and
row-level out-of-sample predictions. Regression/ranking diagnostics may use known
labels with explicit coverage; they are not funded-portfolio SPY outperformance
or promotion evidence. No outer-validation/test reads, downloads, new features,
source admission flag changes or serving integration belong to this checkpoint.

Completed artifact: `data/research/swing_technical_return_models_initial_fit`;
manifest SHA256 `0a30ef2f8e3b95cee252f8c1d1bb5be3c6b98cddda4e47d68e835ad5c1892551`.
Each final model used 314,167 eligible matured rows from 2019-07-09 through
2024-05-28. Initial-fit inputs retain all 586,305 decisions / 545 securities;
101 identities are held out only in the separate transfer fits. Peak memory was
2.406 GiB. No heavy process or reviewer remains active.

The measured result is weak, not a fit failure: temporal daily rank correlation
is 0.00449 linear / 0.01467 boosted; aggregate squared error is 1.77% / 1.99%
worse than predicting zero excess. Transfer error is also worse. Only 64.68% of
scored decisions have admitted comparison outcomes; no unavailable outcome is
filled or removed from a claimed portfolio. See the feature audit for all metrics.
No SPY outperformance, funded-policy evaluation, promotion or live readiness is
asserted. The relationship profile subsequently completed its two specifications
on September 21; qualified issuer/SEC reaction now accounts for the remaining two.

### Combined Evidence, Outcomes And Features Delivery

Status: completed; implementation `7b5d834` is pushed to `er-intraday-refactoring`.
September 14 user direction: finish news rebuilding, feature joins and full
verification without stopping between stages. No new downloads, exclusions,
strategy optimization, model fitting or promotion claims belong to this checkpoint.

Frozen exit gates:
- Exact replay of all 59 outcome months and 586,305 decisions.
- Complete monthly issuer-news publication, retaining explicit source gaps.
- Two joined profiles preserving all 586,305 initial-fit decisions each.
- Independently reviewed supported repairs, exact clock/outcome checks, repository
  Ruff, strict mypy and the complete test suite.
- Updated continuity documents and a scoped pushed checkpoint.

Completed:
- Predictor and outcome replay matched all rows, with independently verified
  receipts. Earlier implementation provenance stays immutable and is not executed.
- Compact news preparation covers 59 months / 119 source-month lineages. A
  namespace mismatch was repaired using 445 positive, interval/ticker-aware
  identity bridges, never guessed aliases or additional stock exclusions.
- Source-proven news alignment passed all 136 compact batches / 957,261 relation
  rows. It translated 854,541 relations and retained 102,720 original identities.
  These counts include repeated relation rows, not globally unique stories.
- A writer schema bug stripped UTC metadata from 2,120 compact identity clocks.
  Exact original pinned rows restored them; new writers validate all nonnull
  clocks and use UTC nanosecond schemas even for empty first slices.
- Monthly catalyst publication is complete. News is attached to 224,709 decisions,
  compared with 7,521 in the rejected mismatched generation.
- Distinct publications sharing model-input text retain the earliest available
  original instance. Same-event/source-event conflicts still fail exactly.
  This supersedes cross-publication score/relevance equality, not source integrity.
  No numeric tolerance, averaging, score rewriting or rescoring was introduced.
  The named policy is hashed and loader-enforced with no older-schema fallback.
- Both joined profiles, technical_market and catalyst_full, completed all 59
  months and 586,305 rows each; 479,709 rows are feature-eligible in each profile.
  No row was removed based on its outcome. Unknown coverage and outcomes stay null.
- Independent targeted reviews closed all supported clock and duplicate-policy
  findings. All reviewers are closed. Final Ruff and strict mypy passed. The full
  run had 4,145 passes, 10 skips and one documentation-marker failure. After a
  documentation-only correction both continuity tests passed. No source changed
  after the full run; its original failure receipt is preserved in the handoff.

Completed artifact pins and the exact final verification results are maintained
only in the active handoff. Failed generations remain protected historical evidence,
not active inputs. Do not rerun completed numerical replays or source collection.

The current artifacts are research-only initial-fit evidence. They do not certify
historical first receipt, managed-exit readiness, training admission, validation/test
publication, prospective performance or outperformance of SPY.

### Current User-Approved Dataset Direction

2026-09-08: the user permits excluding a small number of unusable whole securities
from a new research dataset instead of indefinitely repairing every corporate
action before feature engineering. This supersedes the requirement to repair all
70 old selected outcomes before any new cohort may be built. It does not permit
selectively deleting losing trades, modifying raw archives or certifying unknown
total returns. The original blocked control remains immutable historical evidence.

Immediate scope: freeze an evidence-bound whole-security research cohort, apply it
before peer features/relative labels/selection, and report cumulative exclusions and
sector/year effects. The parent modeled universe is 631 IDs: 27 are already excluded,
leaving 604. The additional eighteen failures would yield 45/631 (7.1315%), leaving
586; the 27 warm-up-only securities are not part of the denominator. The user
explicitly approved a 10% ceiling on 2026-09-08 for this research dataset. This
changes only the research restriction cap to 1,000 basis points, not the exclusion
list, live-inference policy, original universe or model/economic contract.
No reset to 18/604 or exclusion based on later performance is allowed.

This is an explicitly retrospective development restriction, not a historically
observable universe screen. Freeze all IDs and evidence before new fitting/results;
do not extend the list after seeing validation losses. Rebuild features rather than
filtering already computed ranks. Reject unknown IDs, altered lineage, cumulative
cap overflow, inconsistent split/profile cohorts and newly missing selected paths.
Benchmarks remain mandatory. Cohort acceptance is not price-basis acceptance, model
promotion or fresh prospective evidence. Full tests, independent review and a Git
checkpoint close this bounded change before dependent materialization/training.

Cohort implementation is now verified and pushed in `771c7bd`. The real audit at
`data/reports/swing_research_cohort/approved_research_population_audit.json` verifies 85
identity-only monthly projections: 853,417 parent rows, 10,971 proposed removals,
842,446 retained rows and 586 retained securities. Audit hash:
`41de559dd1c415dab60771e10fd489150853a6c2fff07e387d1024b33961efe0`.
The approved cohort hash is
`794bcf834501cbaa8ec54716e8b561a5aaf0137610fe67587f4db69f9189fca6`.
These are coverage counts, not a rebuilt feature authority. The new audit status is
`accepted_research_restriction`; accounting and promotion eligibility remain false.
The prior 5% blocked report is retained as historical evidence, not overwritten.

The original long-only SPY-relative implementation plan below remains the guide.
The retained-stock **holding-identity preflight** is verified and pushed in
`307cffe`. Its bounded scope: hash-bound cohort/parent/membership sources,
metadata-only monthly projections, exact ten-session XNYS windows, full membership
authority (including excluded competing owners), and separate initial-fit versus
full-history coverage. Exit: immutable coverage report, synthetic/tamper/lease tests,
independent code/ML review and full verification. It must not inspect numeric
outcomes, add exclusions or equate S&P removal with delisting. Missing ownership
produces an uncovered report, never a fill or a training pass. Bar availability,
stock/SPY/QQQ/sector numeric paths and total-return reconciliation remain distinct
dependent checks. Only then rebuild labels/features and the six sequential fits.

Real holding-identity preflight completed across all 85 months: 842,446 retained
decisions, 836,638 covered mature windows, 947 uncovered mature windows across
100 securities, and 4,861 terminal immature decisions. Initial fit contains
581,455 mature decisions: 580,889 covered and 566 uncovered across 60 securities.
The uncovered share is 0.1131% of mature decisions, not a whole-security exclusion
rate or missing-price count. Report: `data/reports/swing_research_cohort/holding_identity_preflight.json`,
audit SHA `8f8cdd60ca2f9c1372ceda20d04b7f90eefbfb2336a7f282f30aa97b8c7e723b`.
Peak process memory was 0.231568 GiB. No numeric outcomes/features were read.
The approved exclusion list remains fixed. Full verification: **2,502 passed,
three skipped, 132 warnings in 17m19s**; 133 focused tests pass, full tracked-Python
Ruff and strict mypy on 328 sources pass. Test peak: 351,141,888 bytes (0.327 GiB).
Independent code/ML and test reviews fixed partial-session competing ownership,
open-ended timestamp handling, continuous same-owner metadata changes and duplicate
decision IDs across partitions. All owned agents and workers are closed/exited.
This closes the preflight implementation, not the overall accounting/data step.

The next dependent implementation uses a separate **holding-observation authority**
under `swing/datasets`, bound to this cohort, original raw SIP artifacts and explicit
security/class ownership evidence. Reuse the retained raw-reader semantics in
`research/swing_transfer_sources.py`, not its frozen eleven-stock population.

September 9 bounded implementation, completed/pushed in `89d1aee`: the source-backed initial-fit holding
observation inventory for the 566 flagged decisions across 60 securities. Reproduce
the exact decision/session requirements from pinned parent identity columns before
opening numeric data. Filter raw Parquet at scan time to those required sessions;
validate using the shared outcome checker, retain missing/invalid observations,
and keep source presence, observation validity and ownership as separate fields.
No post-removal ownership authority currently exists: SEC relations inherit index
boundaries and cover-page stock facts are not validity intervals. Therefore these
observations remain diagnostic until independently effective-dated class ownership
is established; do not wire a partial repair inventory as a complete shard source.
Exit: immutable reproducible inventory, independent review, projection/future-poison,
duplicate/clock/missingness/tamper tests, full verification and real-data counts.
No new exclusions, raw modifications, model fitting or unknown-identity fills.

Observed September 9: the initial-fit inventory and independently pinned replay
reproduced 566 decisions / 60 IDs. The 1,106 unique security/ticker sessions contain
876 valid, 209 missing and 21 invalid observations; 574 sessions separately lack
ownership proof. Missing/invalid bars affect 23 IDs; 37 have valid raw bars for all
requirements. The report is `data/reports/swing_holding_observations/_manifest.json`,
audit SHA `b87e18fc2c706600c0063c606c2dd40dbba0422cfad6b6c8c1e4817242714a8a`.
Peak real-run memory 0.342045 GiB. No source download, raw change, extra exclusion,
admitted label or model fit occurred. Full suite: 2,625 passed, three skipped,
133 warnings in 22m09s; full Ruff and strict mypy on 331 sources pass. Independent
review findings were fixed and verified. All agents and worker PIDs are closed.
This inventory narrows the remaining work to effective-dated ownership and explicit
corporate-action/unavailable-trading treatment, followed by verified total-return
accounting. Valid observations alone do not close those gates.

Completed dependent scope (`58468fb`): collect and replay Alpaca corporate-action response evidence
for the initial-fit flagged ticker inventory, using the pinned holding-observation
report. Freeze process-date coverage to 2019-07-09 through 2024-05-28, all action
families, `data_quality=all`, exact symbols and bounded pagination. Keep process date,
effective/ex/payable dates and retrieval time distinct. A terminal empty response
means no provider records returned for that query, not proof of no corporate actions.
No announcement-time feature, ownership interval, cash availability, adjusted-price
conversion or model eligibility is inferred by collection. Retain encoded response
bytes, exact non-secret query, page tokens, hashes and independent failure receipts;
resume completed tickers without refetching, and verify offline. Exit gates: schema,
URL/body/token/tamper/failure-isolation tests, independent review, full verification,
real acquisition summary and pushed implementation/documentation checkpoints.
Then interpret supported events against exact required sessions through the existing
holding/outcome contracts; unsupported records remain explicit evidence gaps.

Real acquisition and pinned offline replay succeeded for all 60 tickers: 649
distinct records (621 cash dividends, 21 mergers, five name changes, two spin-offs).
Audit SHA `82dafea3055db20db9ee483800a7a22c508dbb325418b979a1cd23974a244c91`.
Sixteen mergers fall inside the affected holding windows; fourteen lack payable
dates. Collection is not ownership/accounting admission. Full suite: 2,799 passed,
three skipped, 133 warnings in 18m08s; 213 focused tests, Ruff and strict mypy on
333 source files pass. Peak collection/test memory: 0.238/0.327 GiB. All workers
exited and agents closed. No new model was fitted.

### Approved Event-Aware Accounting Change

On September 9 the user explicitly approved replacing price-only accounting with
separate tradable shares, available cash, unpaid proceeds and contingent rights.
This supersedes `verified_total_return_units_no_separate_distributions` for the
next implementation. It does not change the ten-session forecast horizon, costs,
funding order, 10% exclusion cap or fixed 45/631 list. The ABMD completion filing
establishes cash plus a nontradeable contingent right: its payout cap is neither a
valuation nor immediately reusable cash. No unsupported fact is assigned zero.

Completed code scope (`b7cdb4e`): strict typed holding/evidence contracts and one canonical lot
transition/valuation kernel consumed by labels and the existing funded ledger.
Replace terminal-mark liquidation with component reconciliation. Payment must
extinguish its matching claim exactly once; only evidenced available cash funds
purchases. Residual claims may survive horizon without being sold or extending the
forecast. Missing valuation makes NAV/return unavailable and blocks NAV-dependent
allocations. Report tradable, unpaid and contingent exposure separately; do not
invent risk weights. Unsupported event ordering, delivery, currency conversion or
fractional treatment remains a gap. Existing adjusted-price diagnostics cannot be
silently converted to admitted explicit-distribution accounting.

In scope: research/holding contracts, lot kernel, label target projection, canonical
ledger/accounting consumers and focused synthetic parity/poison tests. Out of scope:
new universe restrictions, intraday, fake real-data admission, feature materialization
or six-model fitting before the dependent evidence gates pass. Exit tests: ordinary
stock parity; cash plus unvalued CVR; payment after horizon; event after stop and
previously earned surviving claims; duplicate payments; unknown timing/valuation;
double-counted distributions; identical lot economics through label and portfolio
consumers. Consolidated code/ML review, full verification and Git closure are required.
Real source interpretation/materialization follows this shared implementation.

Verification closed September 9: 309 focused tests; full tracked-Python Ruff and
strict mypy on 336 source files; full suite 2,974 passed, three skipped, 133 warnings
in 18m16s, peak 0.329 GiB. Consolidated review fixes cover entry-tied event ambiguity,
prior-session sale cash availability, unearned post-exit events and premature
successor references. All owned agents/processes exited. Target projection and
funded accounting share the kernel; source admission remains false. No new fit ran.

Completed collector dependency (`46ab0f8`): make the existing exact-unit daily collector explicitly
price-basis aware, from transport through immutable request, receipts and replay.
Reuse its collection path; accept only supported adjustment choices and reject
cross-basis resumes. Existing adjusted archives remain unchanged. Bind a fresh
initial-fit raw-share SIP acquisition plan for the fixed cohort, SPY, QQQ and required
sector benchmarks, ending 2024-05-28. No held-out numeric observations may enter it.
This collection establishes raw price evidence, not ownership or distribution
admission. Then join independently interpreted actions/ownership into the actual
holding-specification and target materializer. Do not substitute another diagnostic
inventory for that materializer. Required tests: transport query, request identity,
wrong-basis response/resume rejection, bounded date/unit scope, offline tamper replay,
failure isolation and existing adjusted collection behavior. Fail closed on missing
evidence; do not expand the frozen exclusion list or fabricate cash/marks.

Collector verification: 53 source/collector tests plus 14 combination tests pass;
full tracked-Python Ruff and strict mypy on 336 sources pass. The complete suite
passed 2,999 tests, three skipped, 134 warnings in 19m41s, peak 0.329193 GiB.
Consolidated review found and fixed rejected transport receipt loss; the regression
replays the rejected query from retained bytes and verifies sibling resume. Retained
adjusted warm-up source replay passed for 549 units and 140,383 rows, peak 0.132084
GiB. Both agents and process PIDs25540/12624 exited. No new source collection or fit
ran. The initial-fit raw-plan and source-to-target materializer are not completed by
this transport checkpoint; they are the next bounded dependency below.

Raw-plan publication and acquisition are completed/pushed in `d49c6a4`. The plan
binds the accepted cohort, membership provenance, projected decision identities,
exact ten-session holding paths and all in-window decision sessions. Contiguous
security/ticker runs are not clipped at S&P removal. All 586 IDs remain bound;
545 occur in the initial-fit window and the 41 later entrants are not exclusions.
Published `data/reports/swing_initial_fit_raw_share_plan` has 551 stock runs plus
13 SPY/QQQ/sector ETF units covering 2019-07-09..2024-05-28, before independently
evidenced successor needs. Authority-file pin:
`d912a997af361c820745e8c850f0fd455ee068397c6d034d2b54c00a475dc22e`.

Collection and fresh offline replay completed: all 564 units returned observations,
601,834 rows, no failed/empty requests. Immutable archive:
`data/raw/swing_initial_fit_raw_share_daily`; authority-file pin:
`144cab43741f3c74308b53e9322c84ac7eeaca158d3cbd6a1ddb0d7c0fa8d244`.
Do not redownload or mutate these archives. The implementation retains the caller's
pin, parses hash-matched byte snapshots, rejects scope downgrade, and reconstructs
each unit's decision/holding/union counts and required-session digest.
358 focused tests, full tracked-Python Ruff and strict mypy 338 sources pass. Full
suite: 3,082 passed, three skipped, 134 warnings, 1,266.62s, peak 0.329632 GiB;
XML `.test-tmp/raw-plan-full.xml`. Plan peak 0.396503 GiB; collection peak 0.395340
GiB. All owned agents/processes exited. No additional security exclusion or fit ran.

Next within the accounting step: source-to-target admission and materialization.
The original parent archive had 1,134 fewer rows than 602,968 required stock/ETF sessions across
28 ticker ranges: ECHO 630, FISV 245, the other 26 ranges nine or ten each. The
correction checkpoint below replaces the two incorrect symbol selections. Resolve
exact session sets, dated ticker/class ownership and corporate-action effects rather
than filling missing prices or enlarging exclusions. Collection success is not proof
of security identity, full paths, cash availability or training-ready outcomes.
Join independently interpreted cohort/benchmark actions and genuine missing successor
evidence into actual holding specifications and targets under `swing/datasets`.
Preserve the initial-fit numeric boundary, existing accounting kernel and six-candidate
research design. Do not substitute another metadata-only report for materialization.

September 9 source-admission investigation found a concrete prerequisite: the
retained EchoStar identity (`cik:0001415404`) was queried under its 2026 ECHO symbol
for 2019-2024, when EchoStar traded as SATS and ECHO denoted Echo Global Logistics
(`cik:0001426945`). Fiserv (`cik:0000798354`) changed from FISV to FI at the June 7,
2023 open. Primary issuer/SEC records establish these dated symbol distinctions;
the existing raw archive is preserved as returned provider evidence, not relabelled.
The bounded historical-symbol correction substep before target materialization is
completed and pushed in `2bce3d7`. Frozen scope:
retain official documents, bind reviewed mapping facts and both parent authority
pins, collect only SATS for 2019-07-09..2024-05-28 and FI for 2023-06-07..2024-05-28,
then publish independently replayable correction/source references. Reuse transport
and exact-unit collection rather than downloading the unaffected stock/ETF units.
Do not alter the original plan's implementation/hash bindings, invent prices,
extend numeric scope, exclude another security or admit corporate-action accounting.
Exit: independent design/code review; exact symbol/date/source/parent scope and
tamper tests; raw source replay; full verification; real correction data and coverage
evidence; implementation push and two-document closure. Ownership of returned
classes, entitlements and target materialization remain subsequent admission work.

Observed result: exactly 1,231 SATS and 245 FI raw SIP daily observations acquired,
two successful units and no failure. The source-selection authority discards all
601 wrong-issuer parent ECHO rows and selects 602,709 observations. All 13 benchmark
artifacts are inherited and replayed without new ETF requests. Remaining original
gaps are 259 missing sessions across 26 holding tails and 21 zero-volume observations
(SBNY 10, ATVI one, INFO 10). No corrected observation is missing/invalid, no extra
security is excluded and no label/accounting/promotion eligibility is asserted.
Plan `data/reports/swing_symbol_correction_plan`, independent authority-file pin:
`da98c09a026dd1b2a5a3357a4d9b548533fda8cb77987b49830ff817dbca5e58`.
Archive `data/raw/swing_symbol_corrected_daily`, independent authority-file pin:
`fb92efc1df48adc8f03d8bfff47d9c811975428b5a5903fdc03f2567c5b47970`.
Selection `data/reports/swing_symbol_corrected_sources.json`, independent file pin:
`01153e33e8b6a161c04fde2dbea8021eeea369fef6988868c38b49f0e0c065ee`.
Primary documents and reviewed policy pins are recorded in the active handoff.
Independent design/code review is closed. Supported findings fixed request-snapshot
pinning, CLI policy/plan matching and supplied offline archive pins. Full tracked
Ruff (552 Python files), strict mypy (341 sources), and **3,191 passed / three skipped**
verify the implementation; 132 warnings, 24m56s, peak 0.332546 GiB. XML:
`.test-tmp/symbol-correction-full.xml`. All owned agents/processes are closed.
Next: construct actual event-aware holding specifications from these selected raw
sources and independently interpreted actions, then corrected targets. SATS's BSS
distribution and the remaining unavailable holding tails still need accounting.
Do not train from old ECHO-derived labels, technical features or ticker-news joins.

September 9 bounded compiler work within the accounting step is **completed** in
`07d37e6` (pushed).
Independent design review accepted a fixed-horizon raw-mark compiler: pinned selected
segments plus dated, evidenced position bindings and typed corporate-action facts
produce actual serialized `HoldingSpecification` objects and canonical kernel replay.
Entry is the next XNYS open; marks use the ten exact following closes. Missing initial
ownership/entry yields no specification. Missing later marks remain unavailable.
Incomplete action/ownership evidence suppresses reportable economics even when a
diagnostic replay has numbers. Never infer liquidation, claim marks, payment or cash
availability; capture time is not historical availability. Scope remains initial-fit
2019-07-09..2024-05-28, no additional exclusion and no kernel/correction changes.
Exit tests: source/selection tamper, wrong class and parent fallback, missing sessions,
successor bindings, claims, unknown clocks, cost-once, exact horizon, bounded memory,
lease and immutable output; one code/ML review and the full verification battery.
A real pinned-source compile must persist specifications and replay, not just counts.
Managed exits, complete action coverage, benchmark-relative targets and training
admission are not claims of this bounded compiler checkpoint.
Verification: 29 focused and 47 integrated tests pass; full tracked-Python Ruff
(557 files) and strict mypy (345 sources) pass. Final full suite: **3,220 passed,
three skipped, 132 warnings**, 1,811.42s (30m11s), peak 0.332027 GiB. XML:
`.test-tmp/holding-materialization-full.xml`; PID34188/session7134 exited.
Consolidated review is closed: unconditionally null reportable returns, persistent
gaps for actually held positions, deterministic ordering and adapter lease/mutation
tests are verified. Both reviewers are closed; no Python process remained.
Real offline compilation produced two actual SATS/FI specifications and exact kernel
replays, no materialization gaps, peak 0.398182 GiB. Source/action admission remains
unproven, not silently passed. Output `data/reports/swing_fixed_holding_demonstration.json`,
independent file SHA256 `92c288af8ba12d0dd0b84fdc28a75bc4df1f924e3a2adc7f50d616858c70cbd6`.
No source was downloaded or rewritten and no extra security was excluded. Next within
the accounting step: complete/reconcile action and class-ownership evidence, admit
source facts and integrate managed/fixed stock and benchmark outcomes. Reuse this
compiler; do not replace that work with another metadata-only inventory or a repeated
generic review of this closed implementation.

Read required sessions directly from raw artifacts and share
`validate_outcome_observations()` / `outcome_bar_lookup()`. Keep
`ownership_unresolved`, `observation_missing` and `observation_invalid` distinct.
For the initial-fit diagnostic, push the required-session/date filter into the
Parquet scan before returning numeric rows; do not reuse the transfer helper's
decode-all-then-filter behavior to inspect validation/test prices. Full-history
identity geometry remains separate from the authorized numeric diagnostic scope.
Pass admitted independent observations to `build_swing_feature_rows(outcome_bars=...)`;
do not alter decision membership, peer transforms or their feature-history loader.
The raw request spans the full modeled horizon and previously examined cases had
91 omitted combined-history sessions, but this does not prove these 100 securities'
coverage. Do not infer post-removal ownership from CIK equality, a retained ticker
or an SEC relation whose interval was copied from index membership. No broad
download, new exclusion or accounting acceptance is authorized by this preflight.

Source research checked September 8: Alpaca's
[corporate-actions endpoint](https://docs.alpaca.markets/us/reference/corporateactions-1)
provides split, dividend, spin-off, merger and name-change records, with pagination
and process-date interval semantics. Its creation/availability timing is explicitly
not guaranteed, so a historical response is not a causal announcement feature.
Validate required record fields even with `data_quality=complete`; the documented
response can still include incomplete records that have already been processed.
`sources/alpaca.py::fetch_security_transitions()` already uses that endpoint but
selects only name changes, mergers and reorganizations; it is not a distribution
or adjustment-factor authority. Reuse its transport ownership while preserving raw
response bytes and complete coverage receipts for the new accounting evidence.
The [bar reference](https://docs.alpaca.markets/us/reference/stockbars) distinguishes
split, cash-dividend and spin-off adjustments; `all` includes them, and `asof`
is documented as symbol/entity mapping, not an adjustment-vintage selector. These
documented semantics guide reconciliation; they are not per-stock price proof.

Previous committed verification: 2,422 tests passed, three skipped, 133 warnings; full Ruff and strict
mypy on 326 source files pass. Peak test-process memory was 348,155,904 bytes.
Consolidated review fixed explicit retrospective scope propagation and relative CLI
root handling. The first full run found a missing CLI inventory entry; that entry
was added and the final complete suite passed. All owned workers/agents are closed.
No raw sources, original controls, training labels or fitted models changed.

### Decision And Scope

The recommendation is a **benchmark-aware cross-sectional return model conditioned
on issuer news and the price/volume reaction already observable at decision time**.
Start with the existing ten-exchange-session horizon and point-in-time S&P universe.
Do not restart intraday, buy another data feed, change language, or launch another
repository refactor to do this research.

The economic question is: does a funded, unlevered, long-only stock portfolio end
with more money than SPY over the same calendar period, after costs? A high win rate,
positive average trade return, or AUC of 0.60 does not answer that question. A stock
that rises 1% while SPY rises 2% has positive direction but negative benchmark excess.
No paper, software design, or model provider can guarantee future outperformance.

This section supersedes the earlier four-model *work sequence* and its proposed
universal AUC gate for this new campaign. Existing artifacts and their rejection
decisions remain unchanged. Changes to labels, estimator sources, selection, or
promotion require new explicit contracts and tests before implementation; this
document does not silently modify a frozen model or authorize serving.

### What The Existing Evidence Establishes

- The retained technical authority has 853,417 rows, 604 securities, and 1,759
  sessions. This is substantial data, not 853,417 independent market histories.
  Stocks share market shocks and overlapping ten-session outcomes.
- The corrected broker-action comparison has 27,087 matched prediction rows from
  11,720 unique latest announcements per profile. The historical 113/19-event
  reductions were not evidence that the entire news archive was that small.
- Reported directional broker experiments reached worst-scope inner AUC of 0.524
  for upgrades and 0.552 for downgrades; they did not produce a candidate. This is
  evidence against those exact models, not against every news-conditioned strategy.
- Current technical labels emphasize a top sector-relative quantile of managed
  return. That is different from estimating the amount of future SPY excess.
  Subtracting the same SPY return from every stock on a date does **not** change
  their order; changing the target alone cannot create a predictive signal.
- The latest implementation checkpoint, `880f2a8`, improved outcome/drift correctness
  and passed its tests. It did not train a new model or produce economic evidence.
- The retained August 9 `data/models/swing/technical/evaluation.json` explicitly
  records `locked_test_outcomes_read=true` and `test_access_count=1`. Its reported
  July-2025 through June-2026 results are already exposed. Later specialist runs
  keeping their own locked reads at zero does not make that calendar globally fresh.
- The reviewer verified a historical inconsistency in that August 9 V11-lineage
  bundle: candidate eligibility is true despite failed selected economic gates.
  Current evaluation requires those gates; no current bypass was reproduced. Do not
  use that old bundle as evidence of a qualified current baseline or deployment.
- In that old run, classifier/regressor managed portfolios reported approximately
  +4.49%/+7.91% compounded return, yet approximate managed SPY excess per selected
  trade was -27.37/-9.27 bps. Unseen-security portfolios returned -1.58%/-0.23%.
  These are exposed historical diagnostics, not exact portfolio SPY excess and not
  performance of the current technical authority.
- A semantic conflict was reproduced and corrected in `6758671`: approximate
  managed-exit-session-close excess no longer authorizes or ranks swing candidates.
  Reporting now matches the required funded daily SPY comparison. Daily benchmark
  closes are not contemporaneous stock barrier quotes; the old numerical bias is
  unmeasured, and correcting accounting does not automatically create alpha.
- The May-2019 schedule/missing-warm-up language in the older validation protocol was
  stale and is corrected in `3b2bff5`. Current config begins decisions 2019-07-09, and the panel request records
  history beginning 2018-05-29. Do not download that history again based on stale prose.

Immediate diagnosis must distinguish six measurable causes: weak stock ranking,
return-magnitude mistakes, missing/misclassified catalyst information, exit-policy
losses, transaction costs, and benchmark exposure/cash/sector effects. These are
hypotheses until attributed from the actual ledger, not excuses for a rejected run.

### Research Basis And Limits

| Primary evidence | Relevant result | Application here and important limitation |
| --- | --- | --- |
| [Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276) | Nonlinear interactions improved return forecasts; momentum, liquidity and volatility were influential. | Use regularized and shallow tree return models as controlled comparisons. Their sample covers approximately 30,000 stocks over 1957-2016, largely monthly forecasts. It does not establish ten-day, long-only S&P alpha. |
| [AQR, Implementing Momentum](https://www.aqr.com/Insights/Research/Working-Paper/Implementing-Momentum-What-Have-We-Learned) | Examines seven years of live momentum implementation and returns after multiple real-world frictions. | Evidence that implementation matters, not a promise that a new swing model will work. This is manager-authored evidence. |
| [AQR momentum methodology](https://www.aqr.com/insights/datasets/momentum-indices-monthly) | Uses prior twelve-month performance excluding the latest month and quarterly reconstitution. | Separate slow momentum context from short-term entry/reaction features. Do not cite this as evidence for rapidly rotating ten-day positions. |
| [Acadian, Machine Learning in Quant Investing](https://www.acadian-asset.com/au/investment-insights/systematic-methods/machine-learning-in-quant-investing-revolution-or-evolution) | Demonstrates nonlinear enhancement of a financially motivated signal and emphasizes explicit research discipline. | Learn conditional relationships with a small, motivated feature set, not an unconstrained indicator search. Its case study is not our holding horizon or a replication of its live strategy. |
| [Novy-Marx and Velikov, Trading Costs](https://www.nber.org/papers/w20721) | Buy/hold separation reduces turnover; many high-turnover anomalies lose significance after costs. | Measure replacement benefit versus cost and inspect how many positions the policy churns. Evidence comes from anomaly portfolios, not a guaranteed threshold for this system. |
| [Frazzini, Israel and Moskowitz, Trading Costs of Anomalies](https://www.aqr.com/insights/research/working-paper/trading-costs-of-asset-pricing-anomalies) | Uses institutional live trades to study costs, capacity and cost-aware implementation. | Report size-dependent slippage and stress costs. Institutional execution estimates cannot simply be borrowed for a personal account. |
| [Daniel and Moskowitz, Momentum Crashes](https://www.nber.org/papers/w20439) | Momentum losses depend on panic/rebound conditions. | Test interactions with observable volatility and market rebound state. Much of this evidence concerns long-short momentum; do not transplant its crash statistics to our long-only portfolio. |
| [Savor, Stock Returns After Major Price Shocks](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/10/Stock-Returns-After-Major-Price-Shocks---May-2012---Final.pdf) | Studies 1995-2009 shocks and subsequent 5/10/20/40-trading-day responses; analyst-report-associated shocks behave differently from other shocks. | Test information type multiplied by already-observed reaction. Its event classification can use day +1 information; we must not copy that into a day-0 prediction. Evidence is not a costed long-only SPY strategy. |
| [Tetlock, All the News That's Fit to Reprint, October 2010 manuscript](https://business.columbia.edu/sites/default/files-efs/pubfiles/3099/Tetlock%20Fit%20to%20Reprint%2010%2010.pdf) | Examines over 850,000 news firm-days in 1996-2008; repeated issuer text is associated with subsequent short-horizon reversal. | Measure novelty against prior available stories, not headline tone alone. The study's five-day reversal evidence is not ten-day long-only net performance. |
| [DellaVigna and Pollet, Investor Inattention, December 2006 working paper](https://eml.berkeley.edu/~sdellavi/wp/earnfr06-12-11NewTitle.pdf) | Studies 1995-2004 earnings; attention affects delayed reactions. | Motivation for earnings reaction, not a Friday trading rule. It relies on analyst expectations and much longer delayed-return windows than our ten-session forecast. |
| [Lerman and Livnat, The New Form 8-K Disclosures, March 2008 working paper](https://pages.stern.nyu.edu/~jlivnat/f8k%20current.pdf) | Studies item-specific filing reactions in 2005-2006, including later 30/60/90-calendar-day returns. | Form occurrence alone is not positive news. Verify content and first disclosure; do not imply ten-day SPY alpha from an 8-K flag. |
| [Bailey and Lopez de Prado, Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | Selecting the best of many trials inflates apparent performance. | Record all model/policy trials and assess uncertainty using time blocks. A correction does not turn reused historical validation into fresh evidence. |

These sources motivate experiments, not advertised expected returns. Public material
from quantitative firms does not disclose a reproducible recipe for their proprietary
alpha. No claim is made that this design copies a successful fund.

### Proposed Prediction And Evaluation Contract

1. **Universe and clock.** Reuse eligible point-in-time S&P members, including former
   members. Finviz may inspect today's candidates but may not select a historical
   training population using today's winners. Score after the configured completed
   session cutoff, enter at the next exact session open, and count ten actual exchange
   sessions. News arriving after cutoff belongs to a later decision.
2. **Return target.** Add a separately named fixed-horizon total-return target:
   `stock_next_open_to_tenth_close - SPY_same_interval - stock_round_trip_cost`.
   Train magnitude as well as ordering. Preserve raw gross returns and cost columns;
   cost adjustment occurs once. QQQ and the sector ETF remain reported diagnostics,
   not three independent markets the model must simultaneously beat to satisfy a
   SPY-specific objective. Required benchmark data integrity remains strict.
3. **Managed outcomes.** Keep the existing 3 ATR target, 1.5 ATR stop, ten-session
   timeout as the control execution policy. Fixed-horizon forecasts are not falsely
   labeled managed-profit forecasts. Evaluate every candidate using the actual
   frozen exit policy. A second, preregistered research policy may remove only the
   profit cap while retaining the stop and ten-session maximum; it is an experiment,
   not an already approved live change. This tests whether early profit-taking loses
   continuation gains. Do not optimize a grid of stops or extend the holding period
   until a backtest looks good.
4. **Benchmark honesty.** A daily high/low reveals a possible barrier hit but not the
   simultaneous SPY quote. Keep exact fixed-horizon comparisons separate from
   managed-exit-time estimates. If exact intraday benchmark prices are unavailable,
   mark that trade-level comparison unavailable/approximate. Full-calendar daily NAV
   versus buy-and-hold SPY is still a valid portfolio comparison with documented fill
   assumptions; do not pretend an exit-session close is the intraday exit instant.
5. **One funded account.** Replay available cash, marked holdings, entries, exits,
   exposure, and total equity each session. Overlapping recommendations cannot each
   spend the entire account. Aggregate repeated positions, enforce funded capacity,
   include idle days and capital tied up in losing positions, and charge only actual
   transactions. Report realized and unrealized P&L. Unit-test reconciliation.
6. **Costs and distributions.** Start from the existing 20 bps round-trip estimate
   and 40 bps stress case; neither is claimed to be measured live slippage. Model gaps
   and conservative ambiguous fills. Verify split/dividend treatment for stocks and
   SPY, using either verified total-return series or explicit cash distributions,
   never both. Check adjusted prices are not mistaken for executable raw prices.
   Cash yield is explicitly zero unless a contemporaneous cash-rate source is bound.
   Results are pre-tax; account-specific taxes are not silently estimated.
7. **Attribution, not benchmark substitution.** Report full-account excess versus
   SPY, beta/exposure, cash drag, sector contribution, concentration, turnover, and
   costs. Also compare an exposure-matched SPY diagnostic and the same-universe
   deterministic control. Neither may replace buy-and-hold SPY as the headline
   benchmark. Do not add leverage, shorting, or a permanent SPY allocation to make the
   result pass. An index-core allocation would be a different approved experiment.

### Feature And Data Design

Use one global stock model, not a model per ticker or sector. Preserve the existing
technical source contract as the comparator. Inspect the actual ordered feature list
before adding anything; the following are groups, not permission to duplicate inputs.

| Group | Candidate information available before entry | Historical requirement |
| --- | --- | --- |
| Medium-term relative strength | Existing 20/60-session stock/sector/market returns; distinct six/twelve-month momentum excluding the latest month where supported | Complete price warm-up; insufficient windows excluded, not shortened silently |
| Short-term reaction/reversal | One/five-session residual move, overnight gap, close location, distance from trend, rebound after a pullback | Separate information already reflected in price from return after next-open entry |
| Volume and liquidity | Trailing dollar volume, abnormal volume against lagged history, volatility-normalized move, liquidity/size proxy | SIP feed known; current observation excluded from its baseline |
| Regime interactions | Existing SPY/QQQ/sector trend, realized volatility, market breadth, stock beta/residual return; interaction with short-term reaction | Causal rolling estimates and contemporaneous universe; no retrospective bull/bear labels |
| Issuer news | Event family/direction, age, novelty, deduplicated recurrence, issuer precision, and source coverage | Archived exact article/version, security identity, availability and cutoff; no latest edited text leaking backward |
| Event response | Observed stock-minus-market/sector reaction and abnormal volume in an explicitly defined post-release or announcement window, plus event-age interactions | Verify both start and end boundaries; no future reaction, high, closing volume, or pre-release move mislabeled post-release |
| SEC issuer events | Filing type and material content, earnings releases, financing/dilution, or business changes when verified | Accession and acceptance time, original exhibit/fact availability; no restated latest facts assigned to old dates |

Prioritize **earnings/guidance continuation** and **news-conditioned continuation
versus covered-source no-qualifying-event reversal** as event hypotheses. Broker upgrades/downgrades remain
comparators rather than the sole definition of catalyst. Negative issuer news can
help avoid a long purchase; a negative event is not an instruction to short.

An earnings "surprise" requires an estimate known before the announcement. Without
that evidence, use honestly named reported growth/guidance changes or observed price
reaction; never synthesize an analyst consensus from the later actual. SEC reporting
time is not necessarily the first earnings-announcement time. FinBERT tone alone is
not economic surprise. An LLM may assist extraction only with a frozen auditable
extractor, not use hindsight to decide whether an old headline was important.

For earnings-reaction continuation, admit reported issuer results, not previews or
calendar notices. For guidance, require an explicit issuer guidance raise/lower for
an identified fiscal period; reaffirmation and initiation are different events.
Supporting source spans must be retained. Waiting for a completed post-announcement
reaction session moves the decision and next-open entry forward. An after-close
announcement cannot already have tomorrow's reaction as an evening feature.

Freeze the reaction start as well as its end. With daily evidence, a strict
post-release reaction can use the first complete session whose open follows event
availability: same-day for a premarket release, next-session for an intraday or
after-close release. An event-day close-to-close return or gap may also be useful,
but call it **announcement-window context**, not a pure post-release move; it can
contain trading before the release. More precise intraday reaction requires exact
retained bars with starts after availability and independently accepted coverage.
Test that pre-release price moves cannot enter the post-release reaction feature.

The SEC manifest inspected in this review reports 689,467 filing events across 624
issuers and a companion 853,417-decision authority. These counts are manifest
inspection, not a fresh full replay. SEC is therefore not wholly absent. However,
current canonical form-level text does not prove original exhibit/content coverage,
EPS, guidance or 8-K item meaning; inspect retained document bytes before collection.
Both historical news and SEC timing remain retrospective research evidence where
first-observed history is unproven. Do not manufacture prospective availability.

Audit precision **and recall** on development-only article samples stratified by
event family, year, issuer and market session. Record actual source denominators,
unique announcements, duplicate reports, and attached stock decisions separately.
Include unclassified and rejected articles as well as detected events in the recall
sample; reviewing only admitted events cannot measure what the classifier missed.
Name the negative control **no qualifying event in the covered source window**, not
"no public information". Provider coverage cannot prove that no news existed
elsewhere; unclassified evidence must not automatically certify a clean control.
Ground truth must be independently source-checked; an extractor grading its own
answers is not an independent audit. Record whether labels were human-reviewed or
model-assisted and retain research-only status where admission evidence is insufficient.
Weak classification of one family must not make all other news disappear. Missing
coverage is not neutral sentiment and does not mean "no catalyst". New feature
profiles must explicitly define nulls, required-source abstention and optional-source
missingness; never silently switch the source set of a fitted model.

Global events are separate market/sector context. Start with observable market and
sector reactions already supported by the bars. GDELT/open-source flashpoint text
may enter only through a separate timestamped, full-history, preregistered ablation;
no fabricated war labels, current supply-chain maps applied backward, or ticker
attribution merely because the article mentions oil or technology. It is not a
dependency of the first bounded experiment.

Existing archives are the first source. Backfill accepted new transformations across
the complete permitted history. Download only an enumerated, genuinely missing
source/date range after checking raw archives and permissions. Missing SEC or news
coverage must be reported by ticker/year and source; it cannot be replaced with zero.
The first decision date remains 2019-07-09; source coverage before then is irrelevant
to modeled decisions. Preserve the approved 45/631 exclusion list under the cumulative
10% ceiling, affected-window handling, and disclosure of retrospective universe bias.

### Bounded Experiment

Compare two existing-stack learner families: **regularized linear return regression**
and **shallow gradient-boosted return regression**. Forecast continuous return rather
than force every estimator through a top-quantile binary target. Rank predictions by
decision date. An independently calibrated binary view may still report the chance
of positive excess, but an uncalibrated score is not a probability.

Use at most six learned specifications: those two learners crossed with these three
feature profiles. Freeze exact columns and estimator settings before inspecting new
validation outcomes. Repeated folds/refits are logged but do not license new searches.

| Profile | Purpose |
| --- | --- |
| Existing technical features | Isolate the objective/learner change from new information |
| Technical features plus distinct price/volume/regime relationships | Test incremental relationships, not another collection of redundant indicators |
| Same relationships plus qualified issuer news and SEC reaction features | Measure catalyst value on exactly matched decisions as well as overall deployable coverage |

Source-blocked profiles are reported as not trained; train ready technical profiles
without pretending they include news. Paired comparisons use identical security/date
rows, weights, labels, splits, costs, and policies. Also report each profile's actual
full-universe coverage so a highly filtered event cohort cannot hide poor opportunity
coverage. Include known negative/non-event outcomes, not only stocks that later rose.
Neither a losing deterministic baseline nor a prior rejected specialist prevents a
valid new estimator from being trained.

Control models are SPY buy-and-hold, a same-universe simple momentum rank, and the
existing technical formula on the same policy. They are not extra learned families.
The two predefined exit policies give at most twelve learned model/policy comparisons;
they count toward the complete trial record. No post-result threshold grid, seed
shopping, sector winner-picking, or retrospective best holding-period selection.

A portfolio replacement must cover its incremental cost. Freeze entry/hold separation
and capital/sector constraints before evaluation, using current constraints as the
control. Do not reduce a sector peer threshold or a risk limit simply to get trades.
Measure losses caused by those constraints first. A change to sector-relative ranking
eligibility or hard sector allocation is a separately named contract decision.

### Validation And Success

- The reconciled split config and feature audit describe initial fit
  2019-07-09 through 2024-05-28, a ten-session embargo, 252 validation sessions, and
  historical test 2025-07-01 through 2026-06-30. The protocol's obsolete May-2019
  requirement was removed in `3b2bff5`. Do not move the approved cutoff or claim
  that resolving this prose discrepancy restores holdout freshness.
- Use session-grouped rolling/expanding development folds inside the permitted fit
  range, with purging based on actual label end time and at least the ten-session
  embargo. All fitting, normalization, clipping, feature selection and calibration
  precede scored dates. Give each date controlled weight; duplicated news decisions
  must not multiply one announcement into independent evidence.
- The July-2025 through June-2026 holdout has already been evaluated by the retained
  August 9 technical run. Its results are historical/exploratory for this new design,
  not an untouched final test. Preserve all genuinely unobserved outcomes by an
  access-audited manifest; use a newly accruing prospective holdout when none exists.
  Reviewed validation remains development evidence even when called out-of-sample.
  Do not relabel old dates "untouched" after three months of experiments.
- Primary outcome: positive net CAGR difference versus SPY on the complete funded
  equity curve, with a positive mean daily active return and uncertainty measured
  from **time blocks**, not a bootstrap of individual stock rows. Use the existing
  20-session block length as the primary setting; 40-session sensitivity must be
  reported rather than cherry-picked. Report confidence bounds and effective sample
  length, including the dependence from overlapping holdings.
- For promotion, require positive base-cost excess, survival under doubled costs,
  the frozen drawdown/capacity limits, and statistical evidence of positive active
  return after accounting for the complete trial search. Freeze the exact confidence
  procedure in the first checkpoint. A positive point estimate with an interval
  spanning zero is **inconclusive**, not proof of edge and not a fabricated failure
  to collect enough rows. No claim of 0.85 AUC or guaranteed CAGR.
- Report rank correlation, top-ranked return, calibration where applicable, turnover,
  drawdown, sector/year/regime attribution and unseen-security stress. AUC is a
  diagnostic for an explicitly named binary target, not an arbitrary universal veto
  on a useful continuous-return model. Beating QQQ every year is not the SPY mandate.
- If no model passes, publish the measured reason: weak ranking before costs, costs
  consuming gross edge, exit-policy damage, concentrated exposure, or uncertain
  evidence. Do not resume unbounded tuning on the same validation window.

### Ordered Checkpoints

The user initially approved two checkpoints, then authorized continuing through
model training. The objective/evidence checkpoint is complete. The old accounting
control in `6758671` remains historical blocked evidence; it is not the current
training dataset. Corrected initial-fit outcomes and predictors were subsequently
replayed and audited, and `07963cc` completed the two technical return specifications.
The relationship profile's two specifications completed on September 21 under
`a8be7cb`; funded policy evaluation and the two issuer-reaction specifications remain
incomplete. The user-approved whole-security restriction does not waive
causal, price-basis or accounting gates. Names describe behavior, not serial numbers.

1. **Define the SPY objective and reconcile evidence (`complete`).**
   Freeze the new research objective, exact candidate/policy budget, chronological
   split, capital/cost assumptions, statistical procedure and target semantics.
   Reconcile conflicting current documents/configs without rewriting old artifacts.
   Produce a source/feature/experiment inventory and an access record for held-out
   data. Map each proposed relationship to an existing feature or a real gap.
   Owners: `configs/edge_rebuild_swing_training.toml`, swing/strategy contracts,
   temporal manifest, current feature audit, and the governing validation protocol.
   Exit: one reproducible contract and no contradictory current instructions; no
   training or locked-outcome access. Review: independent ML/economics reviewer.

2. **Reconcile returns, capital and SPY accounting (`labels verified; funded evaluation pending`).**
   Accounting code is verified/pushed (`6758671`). Its historical frozen control
   selected 30,525 stock-days and encountered 70 incomplete labels; that run remains
   blocked, not rewritten as a pass. The newer corrected initial-fit dataset has
   passed outcome/predictor replay and supervision checks. Its admitted labels
   support the completed research fits, but no complete funded portfolio result
   has been produced. Unknown selected outcomes still cannot be filtered away.
   Extend the existing label/evaluation owners with named fixed-horizon return
   evidence and a funded daily ledger. Replay retained development predictions only
   if immutable row-level prediction evidence actually exists. The inspected
   specialist rejection artifacts retain aggregates, not fitted models or prediction
   rows. Otherwise verify accounting with real-data deterministic controls and unit
   fixtures, then attribute learned-policy results from step 4's saved chronological
   out-of-fold predictions. Never score fitting data with a final fitted model and
   call it historical out-of-sample replay. Attribute gross ranking, managed exits,
   costs, cash and sector effects where row-level evidence supports it; do not call
   an accounting repair a new alpha result.
   Owners: `swing/evaluation/ledger.py`, `swing/evaluation/accounting.py`,
   `modeling/resampling.py`, `research/swing_accounting_control.py`, the existing
   label owners and retained trainer consumers. The ledger and bootstrap have one
   canonical implementation; no old definitions, compatibility re-exports or
   dependency-boundary exceptions remain.
   Exit tests: identical stock/SPY produces zero gross excess; one cost deduction;
   overlapping trades conserve cash; no negative cash/leverage; no free dividends;
   mark-to-market drawdowns; gap/collision cases; deterministic replay and tamper
   rejection. Exact managed benchmark unavailability remains visible.
   Reviewed admission boundary: accounting-code acceptance is separate from
   total-return evidence. Existing `adjustment=all` metadata does not independently
   reconcile distributions or prove executable raw fills. Until scope-matched
   source proof exists, emit price-ratio diagnostics and `price_basis_pending`,
   with economic eligibility false. Do not accept a caller's `passed=true` assertion
   as proof. No speculative bulk collection is required for this code checkpoint.

3. **Complete causal news and reaction features (`aggregate joins and technical relationships complete; issuer reaction pending`).**
   The saved monthly Alpaca attribution and aggregate joins are complete. They
   do not replace qualified issuer/SEC reaction inputs required by the remaining
   two specifications. The technical-relationship profile is published and trained.
   Audit existing Alpaca/SEC artifacts, broaden eligible issuer categories only after
   development precision/recall evidence, and backfill the accepted feature profiles
   across the existing horizon. Expose the same transforms in batch and inference.
   Owners: `catalysts`, `swing/news_source_inventory.py`, `swing/datasets`,
   `swing/features/pipeline.py`, `catalyst_aggregates.py`, `technical_relationships.py`,
   and `serving/swing_features.py` where the current owner requires it.
   Exit: per-ticker/year coverage, first/last usable news and bars, exclusions, event
   counts, feature availability, and full vertical feature acceptance matrix.
   Tests: after-close news, revised text, wrong issuer, duplicate announcements,
   known-zero versus unknown, SEC acceptance timing, future-poison and batch/live
   parity. No new feature is "done" with batch-only implementation.

4. **Train the six bounded return-model comparisons (`partially complete: four of six specifications`).**
   Existing-technical linear and boosted specifications completed in `07963cc`,
   including four chronological folds, independent transfer fits and final models.
   Relationship linear and boosted specifications subsequently completed under
   `a8be7cb`; the two issuer-reaction specifications await their distinct profile.
   Use the existing Python stack and shared data IO. Fit models sequentially with a
   workspace lease, bounded projected batches and a 5 GiB process-memory limit.
   Store every development prediction with source/feature/split/model identity.
   Owners: `swing/training/return_estimators.py`, `return_validation.py`,
   `return_artifacts.py`, `swing/contracts/return_training.py`,
   `research/swing_return_inputs.py`, `research/swing_return_training.py`, and
   the thin `commands/swing_return_training.py` adapter. Do not route continuous
   return fitting through the retained classifier/ranker.
   Exit: reproducible fits, chronological calibration, matched-profile comparisons,
   date-weighted metrics, complete failure records, and no held-out data reads.
   No more feature changes are permitted after this checkpoint's validation starts.

5. **Evaluate the frozen long-only policies (`pending`).**
   Replay each learned specification under the two preregistered exit policies and
   the same funding/cost constraints. Report net NAV versus SPY and attribution,
   source coverage, turnover/capacity, doubled costs and drawdowns. Rank candidates
   by the frozen economic criterion, not highest AUC or average winning-trade return.
   Exit: one selected candidate or `no_candidate`, all twelve or fewer comparisons
   accounted for, independent review of row-to-NAV reconciliation, and no manual
   policy adjustment. This is offline evaluation, not trading_flow execution code.

6. **Start frozen prospective validation (`pending`).**
   Freeze estimator, sources, feature order, selection/exit policy, costs and thresholds
   before producing research predictions for newly observed sessions. The exposed old
   test year is usable only as disclosed historical evaluation; it cannot substitute
   for this forward record. Start the forward prediction/outcome record immediately
   after candidate selection, not after a promotion that depends on that record.
   Preserve first-observed news, filing and bar times. Include the separately fitted
   unseen-security stress and inherited robustness diagnostics without treating them
   as additional independent calendar histories. Freeze the final assessment date,
   sample requirements and stopping rule before observing results; do not repeatedly
   test until a favorable confidence interval appears. Collection and maturation may
   progress while evidence is insufficient, without issuing actionable predictions.
   Owners: `swing/evaluation`, `governance/outcomes`, model/selection contracts and
   existing research/collection adapters. Exit: immutable forward predictions and
   matured outcomes, complete clock/source/policy identity, and a scheduled fixed
   evaluation boundary. Insufficient future data does not block earlier historical
   feature engineering or training.

7. **Evaluate once and publish verified swing serving (`pending`).**
   At the preregistered boundary, evaluate the protected forward predictions and
   compare the same economic policy. Publish a model card with net SPY difference,
   uncertainty, capacity and limitations, or an honest rejected/inconclusive result.
   Historical selection and this forward test must use the identical feature builder.
   Owners: `swing/evaluation`, `governance/outcomes`, `governance/promotion`, `serving`.
   Exit: sufficient preregistered evidence, reproducible realized/predicted differences
   and accepted promotion, with no retuning against final outcomes. No promotion from
   retrospective publication-time news alone. API returns named ten-session expected excess,
   calibrated uncertainty only where validated, benchmark, cutoff, catalyst evidence,
   source coverage, release identity and abstention reasons. No alerts or orders.

After the contract checkpoint, label/accounting work and source/feature work can run
in parallel on disjoint files with reviewer oversight. Training and heavy verification
remain sequential. Each implementation step gets a bounded design/diff review,
focused tests, the required repository verification, a Git checkpoint, and updates
to these same two continuity documents. Close reviewers and owned workers afterward.

Stop conditions are named: `source_blocked`, `contract_conflict`, `no_candidate`,
`statistically_inconclusive`, or `prospective_evidence_pending`. None is silently
converted to a pass. No new infrastructure, model architecture, or data purchase is
a substitute for diagnosing which of those states actually applies.

### Research Checkpoint Status

#### Selected Holding-Path Repair Scope

Code checkpoint **verified and pushed in `bc9dbd5`**. Full suite: **2,193 passed,
three skipped, 133 warnings, 15m46s**; Ruff and strict mypy on 320 source files pass.
The corrected source/path code, exact XNYS clocks, missing/zero-volume handling,
Timestamp parity and implementation-bound resume/complete rejection have focused
tests and independent review. A real metadata check rejected the old retained
authority; its bytes were not changed. Final report:
`.test-tmp/holding-path-verified.xml`. Agents and owned Python workers are closed.

The current panel schema is `market_predictor.swing_panel.independent_holding_paths`.
Old partitions cannot be resumed under this schema. The initial full attempt was
stopped for missing resume binding; a later full run caught an outdated label-policy
hash expectation after 1,839 passes. Both were corrected before final verification.

**Source admission remains blocked, so checkpoint two is not fully accepted.** No
repaired authority or model was produced. Forty price-complete paths are not yet
identity-reconciled; 30 need additional halt/merger/distribution evidence. Canonical
API support does not supply missing source history. Current materialization callers
still require verified post-membership source expansion. Preserve the 70-row blocked
receipt, frozen selected IDs and calendar until a new authority can be replayed.

**Official-source collection implementation complete (`e1e4803`, pushed):**
2,226 tests passed, three skipped, 133 warnings, 16m29s; full Ruff and strict mypy
on 321 source files pass. Report: `.test-tmp/official-documents-verified.xml`.
The first full attempt exposed an omitted collection-CLI inventory entry; corrected
before the final full run. The bounded review's stream-error isolation and external
attempt-path findings are fixed and tested, including relative-path offline replay.
Eight official bodies were retained: six from the ten-document holding inventory
and both AIV follow-up filings. Two initial requests failed and two were deferred;
see the feature audit for immutable report identities. No historical bar, selected
decision or model was changed. Acquisition is not financial/source admission.

**Frozen source-collection scope:** retain the official filing, exchange and
regulator documents behind the unresolved holding periods. Reuse retained evidence
before networking. A configuration names exact public URLs and document purposes;
collection stores bounded response bytes, retrieval metadata and content hashes.
Successful documents resume without another download. A failure on one source
must not erase or block unrelated documents; tampered retained evidence fails closed.
Request identity and immutable per-attempt receipts must be verifiable offline.
No collection receipt authorizes security mapping, distributions, settlement,
training or serving. Historical publication and today's retrieval remain distinct.
Exit checks: offline replay, request/body tamper, independent failure/resume,
redirect/error-body rejection, bounded network reads, dependency boundaries,
independent review, Ruff, strict typing and the complete test suite. Corporate-action
interpretation and the separately reviewed post-membership identity relation follow
only from retained evidence; the frozen stock selection and old panel stay unchanged.

**Next bounded identity work:** publish `retrospective_index_transfer_identity`
evidence for the 38 explicit transfer paths across eleven tickers. This is a
retrospective source-attribution interpretation, not an investment-rule change or
an assertion of lifetime continuity. Bind canonical membership/issuer/stock class,
the retained positive S&P transfer announcement, exact permitted holding sessions,
and an explicitly date-anchored Alpaca replay. Reuse existing filings and prices;
collect only the nine missing class-document candidates and eleven short provider
windows. A successful `asof` response alone is insufficient because provider lookup
may fall back to symbol-only resolution. Review concrete class/reuse/reorganization
contradictions, without demanding a certificate of no event for every day. Hash and
replay all sources; tamper, conflicting class, wrong interval, changed decisions or
unexplained price differences fail closed. Identity admission never clears the
separate total-return/fill gate. AIV and other action-dependent cases are outside
this 38-path interpretation. Synthetic merger redemption/distribution conventions
remain unapproved pending the asynchronous user decision; keep existing rules.

**Verified source subcheckpoint `0c5f1f7`:** the bounded replay/class-fact code is
complete and pushed; the combined retrospective identity relation is not yet
published. Source preparation projects only selected initial-fit identities and
required raw observations. The shared Alpaca decoder rejects duplicate JSON keys
and normalized-symbol collisions. Per-ticker attempts preserve exact bytes and
failures, resume verified successes and replay offline under the shared job lease.
SEC class extraction keeps issuer/context associations explicit without inventing
validity intervals. Full verification: 2,369 passed, three skipped, 133 warnings
in 16m19s; 167 focused tests, full Ruff and strict mypy on 324 source files pass.
All reviewers and owned workers are closed; measured test-process peak 0.324 GiB.

Real acquisition retained all eleven windows / 140 required ticker-sessions, with
no missing or zero-volume observation. Eight tickers match exactly (26 decisions,
101 ticker-sessions). ADS, GPS and PRGO have matching volume but 60/40/56 different
OHLC values (twelve decisions, 39 ticker-sessions). Retained sources, selected IDs,
labels and model files were not replaced. Report SHA-256:
`b8daebd9322da94bd10617d54a1a9cc4c9eb07baa4459f237d76300bc3797198`.
Nine class filings plus the LB rights announcement were acquired separately and
all ten replay offline; ADS/GPS filings were reused. All eleven classes extracted.
The handoff records their immutable archive/report paths and source diagnosis.

**Immediate bounded continuation:** independently compare those three same windows
using an explicit July 25, 2026 symbol-lookup control date, with all other query
parameters fixed. The old request did not record `asof`; the control is a hypothesis
test, not a reconstruction of July adjustment vintage. A separate immutable request
must preserve the completed replay. If the control explains ADS/GPS, compare raw
bars under both lookup dates before attributing an adjustment effect. Require exact
query/byte replay, no source mutation and explicit unresolved results; never tune
price tolerances or exclude the twelve selected decisions. Then bind the positive
S&P/class/identity evidence for the eligible paths. Identity admission cannot clear
the independent total-return/tradability/settlement gates or authorize training.

The bounded continuation of accounting repairs decision/outcome separation, not
the selected population. The source audit reproduced 40 recoverable paths across
12 tickers, requiring 91 retained sessions omitted from combined history. Thirty
other selected paths remain unavailable: nine contain zero-volume records and 21
contain absent sessions. These classifications do not establish corporate-action
proceeds, trading status or investor total returns.

- Decisions retain their original membership, sector, score and cutoff. Index
  removal does not terminate an existing holding or erase its expected label.
- Fixed and managed labels consume separate, explicitly security-identified daily
  outcome history. Align every future session to the benchmark calendar; never
  shift over a missing session or fabricate an executable zero-volume bar.
- Reuse canonical return/barrier evaluators. Preserve existing feature values and
  immutable source/panel authorities. Repaired outputs require new bound artifacts;
  no in-place patch of old labels and no compatibility implementation.
- Verify membership-removal continuation, exact missing-session behavior, duplicate
  identity rejection, zero-volume handling, future-outcome poison isolation, both
  fixed/managed paths, source reconciliation and unchanged ordinary complete paths.
- Missing source identity or corporate-action economics remains source-blocked.
  Independent distribution reconciliation remains required for admission; neither
  recovering 40 price paths nor passing fixtures authorizes training or promotion.


**Accounting implementation (`6758671`, pushed):** cash-conserving overlapping lots,
single prepaid costs, daily marked P&L, fixed idle/tail calendar, required SPY/QQQ/
sector curves, paired base/stress SPY comparisons and descriptive beta/exposure.
Exact intraday exposure-matched SPY and cash-drag decomposition remain unavailable;
closing cash weights are not presented as measured return drag. Approximate
managed-exit-close comparisons cannot admit or order candidates. Price-basis
eligibility remains false without independent total-return reconciliation.

**Retained-data result:** `data/reports/swing_accounting_control/_manifest.json`
is an immutable `blocked` receipt, not a backtest. It binds 30,525 selected IDs,
1,221 initial-fit decision sessions and 1,230 valuation sessions ending 2024-05-28.
Seventy rows have all eight fixed-return fields nonfinite (560 values). Eighteen
tickers are affected; 67 rows are at recorded membership boundaries and 17 explicitly
cross missing sessions, with overlap. Provider-level causes are not proven. The
receipt reads no validation/test outcomes and produces no funded result. Its file
SHA-256 is `f4411da442dcce28c904050045371c3e1ee20e4e1f6e9f041686e7c4f385ee9e`;
post-ownership replay produced identical bytes. Do not drop selected rows after
observing outcome availability or overwrite this evidence.

**Verification:** 340 focused tests and one skip; final complete suite **2,159
passed, three skipped, 133 warnings, 20m32s**; full Ruff and strict mypy on 319
source files; independent review; source-tamper, gap/collision, cash, cost and
dependency tests. The initial full run exposed a newly introduced dependency
violation at 1,149 passed; canonical ownership was corrected, not waived, before
the passing full rerun. Sampled test RSS stayed at or below 0.256 GiB, not a complete
peak measurement. Review agents and Python workers are closed. No real model was
trained, no provider collection started, and checkpoint three remains unstarted.

The following records the completed objective checkpoint:

Implementation `3b2bff5` completes the objective/evidence checkpoint and is pushed.
`configs/swing_research.toml` freezes six return specifications, two policies,
20/40-bps costs, funded one-tenth-NAV cohorts, delayed exit proceeds, one-sided
Bonferroni bounds over twelve comparisons, 20/40-session blocks, 20,000 resamples,
and one prospective assessment after 252 decision sessions plus ten maturation
sessions. Its SHA-256 is
`e37a72796bac9c08ad92a467d079d2a70b4d938f34e286f8dd1e0129a6e9007e`.

`audit-swing-research-evidence` verifies fifteen pinned metadata/config records,
120 ordered technical inputs and sixty recorded historical trial entries across
five manifests. Counts are not unique/lifetime experiments or raw-source replay.
The known exposed evaluation is hash-checked, never parsed for outcomes. The
retained trainer rejects its exposed final interval before panel loading or fitting.
Historical strategy/data hashes are unchanged; no source or model was rebuilt.

Two independent reviewers found no remaining supported checkpoint finding. The
final full suite passed 2,070 tests with three skips and 133 warnings in 27m08s.
Focused verification passed 39 tests; repository-wide `ruff check src tests`,
strict mypy over 315 source files, real metadata audit and diff checks passed.
Sampled test working sets were 0.21-0.25 GiB, not a measured training-memory claim.
No Python worker remained after verification.

The full static gate required mechanical import/string cleanup and deletion of
one proven unreferenced duplicate loader, `intraday/datasets/dataset_io.py`.
Its canonical implementation remains in `intraday/training/training.py`. This is
not an intraday redesign. Current A4.3 transformation identity remains unchanged;
import-only KS4 code-byte identities change, while rejected historical evidence
stays immutable. No accepted/current authority was invalidated or regenerated.
The historical sections below are retained evidence, not another active queue.

## Frozen Data Policy

1. Existing estimator artifacts retain their frozen source sets. A new model may use
   Alpaca direct ticker news, issuer-specific SEC events, or Finviz Elite ticker news
   only after that exact source set is causally backfilled across the model's complete
   decision horizon and independently ablated.
2. The SEC authority distinguishes known zero filings from unknown coverage. Finviz
   screening, global news, and sector news remain overlay/audit inputs unless a
   separately preregistered model contract proves causal estimator value.
3. Reddit and Seeking Alpha are removed and prohibited.
4. Swing decisions begin on `2019-07-09`. Earlier bars are warm-up only.
5. Features and source coverage must be available by the decision time. Unknown
   coverage is null, not zero.
6. Identity, membership, ticker changes, sectors, and benchmarks are point-in-time.
7. Alpaca market bars use SIP and `adjustment=all`; missing bars are not imputed.
8. Sparse gaps invalidate affected windows. Whole-security exclusion is capped at 5%
   of the filtered universe; benchmark failures are not waived.
9. Training and evaluation are chronological, purged, embargoed, and cost-aware.
10. One heavy process runs at a time. Swing candidate training has a 5 GiB
    hard process limit; intraday and serving workloads retain 4 GiB limits.

## Verified Artifact State

| Artifact | State | Evidence |
| --- | --- | --- |
| Swing V12 technical panel | published and replayed | 853,417 `technical_market` rows; 604 securities; 1,759 sessions |
| A3.4 broker-action comparison | corrected, published, and independently replayed | 27,087 prediction rows from 11,720 unique latest broker announcements per comparison dataset; research-only |
| Prior swing candidates | rejected evidence only | no promotion; some specialist runs left locked outcomes unopened, but August 9 technical evidence already exposed July 2025-June 2026 |
| A2 swing baseline trainer | implementation complete; no new candidate artifact | four nested technical ablations plus bounded full-feature ranker/regressor; a later governed run must publish separate statistical evidence |
| Intraday V2 | published and rejected | economically failed after costs; not serveable |
| Intraday V3 z-score lineage | invalid; prohibited | five declared cross-sectional inputs lacked a valid contemporaneous decision-cohort transformation |
| Promoted serving bundle | absent | API must fail closed |

## Model Semantics

### Swing

The active hypothesis is ten-session sector-residual momentum after a controlled
pullback and trend reclaim. Required technical evidence includes 20/60-session
relative strength, SMA50/SMA200 state and slope, pullback/reclaim state, volatility,
liquidity, capacity, SPY/QQQ context, and the point-in-time sector benchmark. The A2
baseline estimator is strictly `technical_market`; catalyst remains a confirmation
overlay. The separate A3 event-driven family may evaluate direct Alpaca ticker events.
SEC is planned as a separate issuer-specific event profile after causal
collection and attachment, preserving known zero versus unknown coverage. Finviz and
global context remain separate overlays.

Within-sector ranking has a preferred target of 50 peers and a hard floor of 30.
Every row persists the sector peer count, sector rank eligibility, whether the target
was met, and ranking reliability weight. Groups of 30-49 peers remain eligible with
weight `decision_time_sector_peer_count / 50`; groups below 30 are ineligible. Portfolio construction
targets a 20% maximum sector weight, adapts to 25% with four represented sectors and
33.3% with three, and skips sessions with fewer than three. Economic gates are
unchanged.

Entry is the next exact exchange-session open. Target, stop, timeout, costs, and all
benchmark returns use the same executable interval.

### Intraday

The active hypothesis is a thirty-minute VWAP exhaustion reversal. Evidence is built
from completed causal intraday bars with next-minute execution, exact one-minute path
labels, stock/market/sector context, and explicit abstention. Catalyst is a
confirmation or ranking overlay unless a preregistered causal ablation proves
estimator value.

Intraday V2 remains a valid rejected artifact. The later V3 z-score lineage is
invalidated and cannot be evaluated, trained, or served. A replacement intraday
cross-sectional feature contract must define one causal decision cohort, share the
same batch/live transformation, and be fully backfilled before candidate training.

## Historical Four-Model Improvement Program

The prior four governed model families are swing baseline, swing event-driven, intraday
baseline, and intraday event-driven. `ROC-AUC >= 0.60` is frozen as a validation and
later locked-test gate for their comparable binary outcome view; it is not permission
to optimize repeatedly on either split. Ranking quality, calibration, benchmark-relative net economics,
drawdown, turnover, capacity, and coverage remain co-equal promotion gates.

| Code | Descriptive checkpoint | State |
| --- | --- | --- |
| A0 | Restore research integrity | Completed |
| A1 | Verify labels and leakage controls | Completed |
| A2 | Build the technical swing baseline | Completed |
| A3 | Build catalyst-driven swing specialists | Completed; combined and directional specialists produced no candidate |
| A4 | Build the technical intraday baseline | Completed; both A4.4 hypotheses rejected |
| A5 | Build catalyst-driven intraday specialists | A5.1e completed with no candidate; prospective SIP horizon is 1/20 sessions |
| A6 | Run locked evaluation and promote qualified models | Not started |

### A0 - Restore Research Integrity (`completed`)

Problem: the current working tree contains unfinished experimental edits that bypass
economic validation, partition verification, source missingness, and governed sector
limits. Those edits invalidate any resulting metric.

In scope: restore fail-closed validation and immutable replay, preserve the 4 GiB
intraday/5 GiB swing limits, remove false zero catalyst inputs, verify the current
cross-sectional feature implementation, and checkpoint only supported code.

Out of scope: new provider collection, feature backfill, model training, locked-test
access, promotion, serving success, and trading alerts or execution.

Exit gates:

- no validation, partition, lineage, or economic-gate bypass remains;
- missing catalyst authority cannot become a numeric zero feature vector;
- training does not mutate loaded immutable dataset frames;
- focused poison tests, full tests, Ruff, strict mypy, compileall, and diff checks pass;
- active plan, feature audit, and handoff describe only verified behavior;
- implementation and documentation closure commits are pushed separately.

Rollback/failure behavior: training and serving remain blocked; no existing rejected
artifact is promoted or used as a fallback.

Completed evidence: implementation commit `e168482` restores monthly partition replay,
requires every validation scope to pass economic gates, preserves immutable input
frames during bounded sequential training, and removes five cross-sectional z-score
columns that had no valid decision-cohort implementation. Focused verification passed
145 tests with one skipped. The full suite passed 1,102 tests with two skipped; three
repository-lock permission failures passed independently with normal repository access.
Ruff, strict mypy, compileall, and staged diff checks passed.

### A1 - Verify Labels and Leakage Controls (`completed`)

- Freeze one comparable binary diagnostic for each model plus the economic training
  target: managed and exact-ten-session benchmark-relative swing return and
  30-minute managed intraday return after costs.
- Reproduce labels from immutable bars using the shared evaluator and identical stock,
  SPY, QQQ, and sector executable intervals.
- Add label shuffle, feature-time shift, future-poison, duplicate-event, overlapping
  label, and survivorship controls. Any abnormal control score blocks training.

Completed evidence: implementation commit `7b61873` removes training-time swing label
rewrites, hard-coded SEC attachment, missing-as-zero filing counts, and the event-only
sector-selection bypass. Technical and Alpaca ablations must now contain identical
published decisions and labels. Intraday label schema V2 adds QQQ on the exact stock
entry-to-managed-exit interval; missing QQQ evidence abstains. Both trainers publish
named estimator-target and after-cost stock/SPY/QQQ/sector binary diagnostics plus a
deterministic shuffled-label AUC control. Future-only label, return, and feature-time
poison tests cannot alter validation selection. The canonical suite passed 1,110 tests
with two skipped; tracked Ruff, changed-module strict mypy, compileall, and diff checks
passed.

### A2 - Build the Technical Swing Baseline (`completed`)

- Evaluate compact, evidence-backed groups sequentially: benchmark/sector residual
  momentum, volatility, liquidity, turnover, quality, profitability, investment,
  valuation, and estimate-revision data where a point-in-time authority exists.
- Use a regularized linear baseline followed by bounded tree ranker/regressor models.
- Backfill every accepted feature for every eligible decision from `2019-07-09` through
  the frozen end date before any model consumes it. Partial-period feature additions
  require an explicit specialist cohort or are rejected.

Completed evidence: implementation commit `cb2aba5` freezes four nested technical
groups: momentum/volatility, trend confirmation, pullback timing, and volume/liquidity.
It evaluates one regularized logistic candidate per group, then one full-feature
XGBoost ranker and regressor, all sequentially within the six-candidate budget. The
bundle records each estimator's exact ordered feature subset and a deterministic,
lineage-bound candidate ID. A promoted baseline selects `technical_market`; an event
specialist requires its own A3 feature contract and promotion and cannot reuse a broad
catalyst frame. Current Finviz snapshots cannot supply historical
point-in-time quality, profitability, investment, valuation, or estimate-revision
features, so those groups remain blocked rather than backfilled from present values.
No new real model was trained or promoted in A2. The canonical suite passed 1,110
tests with two skipped; tracked Ruff, changed-module strict mypy, compileall, and
governance hash replay passed.

### A3 - Build Catalyst-Driven Swing Specialists (`completed; rejected`)

1. **A3.1 - Verify the issuer-targeted event taxonomy (`completed`).** Build separate
   earnings/guidance, SEC material-event, analyst-revision, offering, M&A, regulatory,
   and product-event cohorts only when exact availability and issuer relevance verify.
2. **A3.2 - Backfill and replay historical event authorities (`completed`).** Publish immutable
   direct-issuer event, source-coverage, assignment, and cohort authorities for the
   complete development horizon. A source cannot imply coverage for a family it does
   not provide.
3. **A3.3 - Audit event precision and coverage (`completed`).** Use deterministic
   uniform samples of independent event clusters and preregistered one-sided lower
   confidence-bound gates. Report event, security, calendar, sector, source,
   abstention, unknown-coverage, issuer-error, and reviewer-agreement counts.
4. **A3.4 - Build identical-decision comparison datasets (`corrected and completed`).**
   Compare technical-only, broker-action-only, and combined inputs on the same
   predictions and outcomes. Exact ticker/time alignment replaces invalid direct hash
   matching; every excluded row carries a concrete reason.
5. **A3.5 - Define, train, and evaluate swing broker-action specialists (`completed`).**
   One rating-change specialist combines upgrades and downgrades with explicit action
   direction; a separate specialist handles new/resumed coverage. Price-target and
   generic actions remain report-only because only 55 independently aligned latest
   announcements exist. Each specialist compares technical-only, broker-action-only,
   and combined features with logistic and histogram-gradient-boosting estimators.

Completed A3.1/A3.2 evidence: commits `6ae703c`, `58ccc3d`, and `527e20f` publish the
V2 issuer-targeted classifier and immutable authority replay. Title-derived events
require a causal issuer anchor; bare ambiguous ticker words, preview/conditional deal
language, unsupported source/family pairs, and unknown coverage abstain. Two strict
historical authorities now cover the full development horizon without new network
collection:

- `2019-07-09` through `2021-07-08`: 9,018 classified events, 30,875 assignments,
  28,462 coverage rows, and 1.998 GiB observed peak memory;
- `2021-07-09` through `2026-07-08`: 26,370 classified and research-eligible events
  across 525 securities, 90,136 assignments, 18,333 coverage rows, and 2.157 GiB
  manifest-recorded peak memory.

Earnings, guidance, analyst revision, offering, merger/acquisition, regulatory
decision, and product event are admitted as Alpaca source families. SEC material
events remain `blocked_missing_source`; Alpaca coverage cannot imply SEC coverage.
Both authorities are research-only retrospective evidence and cannot authorize
production.

Completed A3.3 evidence: implementation commits `6ef0579` and `9c8aa5b` publish a
disk-backed, memory-guarded audit with uniform cryptographic cluster sampling, two
independent blind reviewers, separate adjudication, per-field agreement/kappa,
one-sided Wilson bounds, issuer-error vetoes, immutable ledger copies, strict replay,
and fail-fast ledger preflight. The older sample reviewed 1,788 inferential clusters
plus eight paired issuer diagnostics; the newer sample reviewed 1,830 inferential
clusters plus 29 diagnostics. Reviews were performed by independent Codex agents, not
human reviewers, and remain research evidence.

Both eras currently admit only broker rating actions (stored under the internal event
family code `analyst_revision`). Earnings, guidance, offering,
merger/acquisition, regulatory decision, and product event fail at least one frozen
precision, wrong-issuer, reviewer-agreement, or rule-variant gate. SEC material event
has no source-authorized population. Blocked families cannot enter A3.4 or training.

The catalyst-independent V12 base authority contains 853,417 `technical_market`
rows, 604 securities, and 1,759 sessions. The first A3.4 artifact was invalid: it joined
old event decision hashes directly to the rebuilt technical panel, so only 113 rows
from 50 announcements survived. The source data actually contains 17,401 broker
announcements and 37,372 causally covered prediction timestamps. The corrected A3.4
artifact aligns exact ticker and exact prediction timestamp, rejects conflicting CIKs,
and records every inclusion and exclusion. It publishes 27,087 prediction rows from
11,720 unique latest broker announcements in each of three exact datasets:
technical-only, broker-action-only, and technical-plus-broker-action. The internal
profile names retain `analyst_revision` for source lineage. Unknown three-day Alpaca
coverage abstains; blocked event families are absent, not zero. The artifact is
research-only and cannot serve or authorize production.

A3.5 development artifact
`data/models/swing_broker_action_specialists_dev_20260812_v4` records 12 sequential
experiments and a 0.414 GiB peak working set. Capacity passed: rating changes contain
5,841 development and 1,138 validation announcements; coverage initiations contain
2,344 and 502. Model/threshold selection uses a separate 2023-06-13 through
2024-05-28 inner window after a ten-session embargo. No experiment passed that inner
gate, so outer 2024-2025 validation and the locked test both remained unopened. Best
worst-scope inner AUC was 0.546 for rating changes and 0.506 for coverage. Broker-only
inputs were near or below chance; every candidate also failed the canonical portfolio
economic gate. No model artifact was emitted or promoted. The prior v2 report is
superseded because it selected and evaluated on one validation window and used a
simplified economic calculation.

6. **A3.6 - Evaluate upgrades and downgrades as separate swing specialists
   (`completed; no candidate`).** Reuse the corrected A3.4 identical-decision authority and existing
   governed analyst subtype classifier. Split the prior combined rating-change
   specialist into upgrade-only and downgrade-only cohorts, then compare technical-
   only, broker-action-only, and combined profiles on identical rows, folds, labels,
   costs, and security scopes. Capacity must be audited before training; no threshold,
   estimator, gate, or locked-test boundary may be changed based on A3.5 or A5.1e
   results. Coverage initiation and price-target/generic actions are out of scope.

   Implementation result:

   - Commit `82b4959` generalizes the single governed swing-specialist path to accept
     either the historical rating/coverage pair or the frozen upgrade/downgrade pair.
     The new policy changes only cohort membership; profiles, estimators, labels,
     chronological split, embargo, unseen-security assignment, costs, and gates are
     unchanged. Mixed specialist sets fail closed.
   - Upgrade capacity passes with 3,008 development, 561 chronological-validation,
     and 102 unseen-security announcements. Downgrade capacity passes with 2,833,
     577, and 111 respectively. Both include 359 development securities and all eight
     represented sectors.
   - Twelve experiments compare technical-only, broker-action-only, and combined
     profiles using logistic and histogram-gradient-boosting estimators. Upgrade best
     worst-scope inner ROC-AUC is 0.524 for combined logistic (0.533 chronological /
     0.524 unseen). Downgrade best is 0.552 for combined gradient boosting
     (0.552/0.560), only 0.002 above its technical-only control in the weaker scope.
   - No threshold passes canonical economics in both scopes and no experiment reaches
     the 0.60 AUC gate. The artifact is strict `no_development_candidate`; outer
     validation and the locked test remain unopened, no model file exists, and
     promotion remains prohibited.
   - Output
     `data/models/swing_directional_broker_action_specialists_dev_20260820_v1`
     strictly replays. Peak working set was 0.376 GiB under the 5 GiB limit. Final
     verification passed 1,463 tests with two skipped, tracked Ruff, strict mypy over
     231 source files, and tracked compilation.

### A4 - Build the Technical Intraday Baseline

1. **A4.1 - Build the SIP trade/quote source authority (`collector_complete`, full
   authority `environment_blocked`).** Add bounded,
   paginated Alpaca SIP trade and NBBO-quote clients plus resumable raw collection.
   Preserve provider timestamps, exchange/tape/condition identity, request bounds,
   page tokens, response rate-limit headers, and per-unit failures. Raw transport
   completion is not model readiness.
2. **A4.2 - Publish the one-minute microstructure authority.** Aggregate only completed
   regular-session minutes using the shared batch/live transformation. Publish
   time-weighted relative spread, time-weighted quote-size imbalance, quote-update
   count, trade count, share volume, dollar volume, and mean trade size with explicit
   availability and source coverage. Missing quotes or trades remain unavailable.
3. **A4.3 - Publish the bar-only causal technical dataset (`complete`).** This is a distinct
   governed model profile, not a degraded microstructure model. Its source set is
   limited to verified SIP/all one-minute bars, causal five-minute bars,
   point-in-time membership, SPY, QQQ, and sector ETFs. It may use volume clock,
   VWAP displacement, opening range, volatility, and exact market/sector residuals,
   but it must not contain trade-count, quote, spread, or imbalance fields. Prove
   batch/live parity, future-poison rejection, and ordered-feature hash identity.
   Cross-security ranks use explicit contemporaneous clock-time cohorts, never each
   stock's asynchronous volume-bar completion timestamp. ATR used by features,
   targets, and stops comes from the causal five-minute authority required by the
   strategy contract, not from volume bars.

   Frozen A4.3 contract:

   - Project the already verified canonical SIP/all regular-session five-minute store
     into one immutable selected-stock-session authority. This is a local projection,
     not a provider download. Retain incomplete sessions as explicit coverage metadata.
   - Continue to derive event-based volume bars from the verified selected-session
     one-minute stock collection, but decisions exist only on a pre-scheduled
     five-minute cohort clock after activation. At each fixed cohort, use the latest
     completed volume bar whose evidence was already available by the cutoff. Late
     evidence cannot move the cohort; it remains unavailable until a later scheduled
     decision. Cohorts follow exchange-session open plus the frozen 60-second
     finalization delay.
   - Set `source_feature_available_at_utc` to the latest source availability and
     `feature_available_at_utc` to the cohort cutoff. All one-minute stock, SPY, QQQ,
     and sector context is the latest exact completed minute available by that cutoff.
   - Compute `atr_14_5m` only from completed canonical five-minute bars in the same
     session. The model ATR fraction and the 2.0/1.5 ATR target/stop labels must use
     this value. Volume-bar ATR is prohibited.
   - The bar-only ordered estimator contract contains technical momentum/trend,
     volume/liquidity, session VWAP, 15-minute opening-range distance, exact
     stock/SPY/QQQ/sector returns and residuals, and timing fields. It contains no
     trade count, quote, spread, imbalance, catalyst, SEC, Finviz, or global-event
     input.
   - A later session gap cannot remove an earlier decision. Feature eligibility uses
     only evidence through the cohort cutoff; label eligibility uses only the exact
     next-minute entry and 30-minute managed outcome interval. Missing evidence
     abstains only the affected row.

   Exit gates: immutable selected five-minute projection and dataset replay; exact
   clock-cohort identity; no duplicate ticker/cohort; five-minute ATR lineage proof;
   identical batch/live ordered features; future-poison, missing-versus-zero,
   incomplete-later-session, benchmark-interval, artifact-tamper, and path-traversal
   tests; complete dataset publication under 4 GiB; no locked-test access and no model
   training in this checkpoint.

   Completion evidence: implementation commits `8a76ec1`, `e76bf8d`, and `1829fce`;
   immutable five-minute
   projection `data/canonical/edge_rebuild_selected_session_5m_bar_only_causal_20260814_v2`
   with 43,226 selected stock-sessions, 3,364,335 rows, 43,132 complete pairs, 94
   incomplete pairs retained as coverage, and no provider download; immutable dataset
   `data/features/intraday_causal_volume_bar_dataset_20260831_v2` with 794 sessions,
   501 tickers, 3,095,688 rows, and 1,365,015 eligible rows. Dataset manifest SHA-256 is
   `1f09a55489a2889b40899eff44ecb4205dba2162c3198f8def559b6e50951a58`, request SHA-256 is
   `5e8c508a4237320d6ea56205502f670244a49f713b22dbff10a336d4d2dc303a`, and
   transformation SHA-256 is
   `6fdfd0c8f07e4f7445b66d038cbd936e4459db68e087a5ddbcb30eac4795cb51`.
   Audit v2 at
   `data/reports/intraday_causal_volume_bar_dataset_20260831_v2_audit_v2.json`
   (SHA-256 `f1b21af3704317d070f479ac6a562fdb126c21d2b7da021ddf5c7b6108be97e8`)
   binds the dataset and exact projection path/authority/manifest/inventory. It reports
   zero duplicate decisions, normalized or raw source cutoff violations, incomplete
   five-minute-prefix eligibility, label-availability violations, eligible ATR defects,
   feature-hash defects, schema defects, and prohibited features. The resumed build
   predates mandatory per-invocation execution receipts, so
   `data/reports/intraday_causal_volume_bar_dataset_20260831_v2_execution_assessment.json`
   truthfully records `complete_run_memory_proven=false`; earlier invocation memory was
   not reconstructed. This operational limitation does not alter the independently
   replayed row or lineage evidence, but it cannot be cited as complete-run memory proof.
4. **A4.4 - Train separate bar-only continuation and reversion baselines (`complete; no candidate`).** Use purged,
   embargoed chronological selection and the canonical intraday portfolio evaluator.
   Do not open the future holdout unless a preregistered development candidate passes
   calibration, predictive, coverage, cost, drawdown, and benchmark-relative gates.
   Selecting the least-bad failed candidate does not authorize future-holdout access.

   Frozen A4.4 contract:

   - Train continuation and long reversion as independent hypotheses and immutable
     outputs. Continuation requires positive one-volume-bar return, positive
     twenty-minute stock return, and price at or above session VWAP. Reversion requires
     negative twenty-minute stock return, price at least 0.5 five-minute ATR below
     session VWAP, and volume-bar RSI at or below 45. These causal cohort rules are
     configuration identity, not fitted parameters.
   - Each hypothesis fits an expected-net-return opportunity estimator and a separate
     stop-hit downside estimator. Downside calibration uses only an earlier fit slice
     and a later purged calibration slice; validation outcomes cannot calibrate scores.
   - Fit excludes a stable 20% security holdout. Every candidate is evaluated on both
     chronological seen-security and contemporaneous unseen-security validation. The
     worse scope controls selection.
   - Candidate and policy grids are preregistered and bounded. Estimators run
     sequentially under the 4 GiB process limit; no GPU or parallel model fit is used.
   - Gates cover positive-net-return ROC-AUC, stop-risk ROC-AUC and calibration,
     prediction coverage, trade/session capacity, after-cost return, SPY/QQQ/sector
     excess return, rank gain, stress costs, drawdown, turnover, and fold stability.
     A failed run publishes reproducible rejection evidence but no model artifact.
   - The post-2026-07-08 future authority remains unopened unless one frozen
     development policy passes every gate. A4.4 itself cannot promote or serve.

   Result: both real-data runs completed sequentially and published strict immutable
   `no_candidate` evidence.
   - [x] Create `src/market_predictor/edge_rebuild/utils/` (`io.py`, `hashing.py`, `memory.py`, `validation.py`).
   - [x] Consolidate duplicated logic from `intraday_training.py` and `swing_training.py`.
   - [x] Restructure `intraday_training.py` into `intraday_dataset_io.py` and `intraday_types.py`.
   - [x] Refactor `swing_types.py` and `data_io.py` to import from `utils/` instead of local clones.
   - [x] Fix strict mypy and ruff linting errors.

   Continuation's best audited seen/unseen positive-return
   ROC-AUC was 0.510/0.516; long reversion's was 0.513/0.508. Stop-risk ROC-AUC was
   about 0.60 but calibration failed. Controlling scopes also failed after-cost return,
   benchmark excess, confidence-bound, trade-count, and fold-stability gates. Peak RSS
   was below 2.1 GiB. No candidate artifact was written and the future holdout remained
   unopened.

   Exit gates: exact A4.3 authority/hash replay; separate hypothesis identities;
   feature-order and source-contract binding; purged fit/calibration/validation and
   unseen-security overlap audits; deterministic rerun; future-poison and
   artifact-tamper tests; sequential real-data runs below 4 GiB; consolidated ML/code
   review; full tests, Ruff, strict mypy, compileall, implementation commit, and
   documentation closure commit.
5. **A4.5 - Add the microstructure-enhanced profile only after A4.1 and A4.2 are
   complete.** Join the independently verified one-minute microstructure authority at
   the exact completed-minute cutoff, add trade intensity, relative spread, quote-size
   imbalance, quote updates, trade count, and mean trade size, and repeat ablation and
   promotion gates under a new dataset and model identity. A partial trade/quote
   collection cannot be used by either the bar-only or enhanced profile.
6. **A4.6 - Compare profiles only on an immutable matched-ablation cohort.** The
   bar-only and microstructure-enhanced comparison must use identical decision IDs,
   labels, fold assignments, execution costs, and benchmark intervals. Report the
   broader bar-only coverage separately. Missing microstructure makes only the
   enhanced row unavailable; it cannot remove or relabel the corresponding bar-only
   decision.

Historical collection covers every selected stock-session in the source coverage
authority. End-of-session bar completeness is retained only as metadata and must not
decide whether an earlier historical decision exists. Feature eligibility is measured
only through each decision cutoff; label eligibility is measured only through its
managed outcome horizon. Later missing bars may make a label unavailable, but cannot
rewrite the live-equivalent decision cohort.

Existing evidence at A4 start: the selected-session SIP/all one-minute bar collection
contains 81,349,171 rows across 559 observed symbols, and the prior V3 dataset contains
4,173,230 rows. Neither artifact contains historical trades or NBBO quotes, so neither
can authorize microstructure features or A4 training. The invalid V3 cross-sectional
z-score lineage remains prohibited.

The A4.1 live capacity probe on 2026-07-08 found 2,425 AAPL trades and 4,172
AAPL quotes in one ordinary minute; SPY exceeded the 10,000-row quote page limit in
one minute. The local C: drive had approximately 52 GiB free. Therefore a replayable
43,226-stock-session raw tick backfill is storage-blocked locally. No quote/trade
feature may enter A4.5 until a complete immutable source authority exists. A4.3 and
A4.4 completed because its separately identified bar-only contract has no
trade/quote inputs and cannot silently acquire them.

A4.1 implementation commit `b03f4f1` publishes the bounded collector. The corrected
immutable v2 plan contains 43,226 selected stock-sessions and 86,452 jobs: 43,213 have
complete source-bar session status and 13 retain incomplete status as metadata. A real
two-invocation Alpaca SIP probe completed the first trade and quote jobs with zero
failures, 4.41 MiB on disk, and 0.350 GiB peak RSS. The probe remains
`transport_incomplete` and is not an authority. Completing all replayable raw tick jobs
still exceeds current local storage, so A4.2 and A4.5 remain blocked. The next phase is
A5 causal intraday event-cohort preflight; training remains blocked until that separate
authority passes attachment, timing, coverage, and replay checks.

### A5 - Build Catalyst-Driven Intraday Specialists

- Restrict training to verified event cohorts. Use publication regime, time since
  event, premarket gap, abnormal volume, initial 5/15-minute reaction, spread,
  liquidity, and sector concurrence.
- Catalyst remains outside the broad intraday estimator unless causal ablation passes.
  Unknown coverage causes abstention, never neutral sentiment or zero event counts.

1. **A5.1 - Publish the causal intraday event-cohort preflight (`complete; blocked`).** Bind
   the strict A4.3 decision authority to the two strict Alpaca direct-issuer event-family
   authorities. Do not use `intraday_catalysts.py`, ticker filenames, Finviz snapshots,
   prohibited sources, or publication time as a substitute for observed availability.

   Frozen A5.1 contract:

   - Source family is exactly Alpaca; relation channel is exactly `direct_issuer`; the
     only currently precision-approved event family is `analyst_revision`. Each parent
     authority and its exact hash must replay before attachment.
   - Preserve A4.3 `decision_id`, `security_id`, strategy, transformation, feature-order,
     and cost identities. Attach only events for the same `security_id` with
     `feature_available_at_utc <= decision_time_utc` in a half-open 24-hour lookback.
   - Retain known-zero coverage and unknown coverage as distinct states. Unknown,
     ambiguous, proxy-only, or post-decision evidence abstains and is never encoded as
     neutral sentiment or a zero count.
   - Reuse A4.4's stable 20% security holdout and four chronological development folds.
     Report unique event episodes separately from repeated attached decision rows.
   - Publish an immutable `eligible` or `blocked` authority containing decision
     eligibility, event attachments, coverage/split audit, request, manifest, and
     authority. Strict reload must reject changed, missing, extra, nested, overlapping,
     or non-deterministically rebuilt evidence.
   - Preflight eligibility requires observed first-seen/revision availability rather
     than `provider_publication_proxy`, zero issuer/time/hash violations, both seen and
     unseen event coverage, at least 1,000 unique episodes overall, 200 securities, 120
     fit sessions, and 1,000 rows/20 securities in each validation scope. These are
     authorization floors, not performance claims.
   - A blocked output sets `training_eligible=false`, `serving_eligible=false`, and
     `future_holdout_opened=false`, and contains no estimator. A passing preflight only
     authorizes A5.2 matched ablation; it does not authorize serving or locked-test use.

   Exit gates: deterministic strict replay; future-poison, issuer-mismatch,
   proxy-availability, unknown-coverage, duplicate, tamper, extra-file, and path-overlap
   tests; real-data preflight below 4 GiB; one consolidated ML/code review; full tests,
   tracked Ruff, strict mypy, compileall, implementation commit, and documentation
   closure commit.

   Result and evidence:

   - Implementation commit `98f7a48` publishes one canonical A5.1 path and removes
     any need to consume the retired ticker-file catalyst modules.
   - Authority:
     `data/research/edge_rebuild_intraday_event_preflight_20260815_v1`.
     Strict replay binds the exact A4.3 authority, both parent event authorities,
     every consumed artifact and child manifest, and the recursive parent inventory.
   - The parents contain 17,401 research broker-action episodes. Exact point-in-time
     `security_id` matching yields 19 unique attached episodes and 862 repeated
     event-decision pairs. No ticker-only fallback is permitted.
   - All 17,401 episodes use retrospective `provider_publication_proxy` timing and
     zero episodes have observed first-seen/revision-safe production availability.
     Retrospectively completed collection intervals remain unknown at historical
     decisions; they do not become known-zero coverage.
   - The authority is `blocked`, `training_eligible=false`,
     `serving_eligible=false`, and `future_holdout_opened=false`. It contains zero
     estimators and zero production-eligible decisions. A5.2 training and A6 locked
     evaluation are therefore prohibited.
   - Peak working set was 2.299 GiB. Final verification passed 1,348 tests with two
     skipped, tracked Ruff, strict mypy over 226 source files, compileall, strict
     real-authority replay, and consolidated independent review.

2. **Prospective broker-action observation authority (`implementation and weekday
   live validation complete; capacity collection active`).** Establish the
   only permitted path for future Alpaca broker-action evidence. Each polling run must
   archive the exact provider response and observation time, retain every distinct
   provider revision, and bind symbols to the same point-in-time S&P security identity
   namespace consumed by A4.3. Historical publication or provider-update timestamps may
   never be substituted for observation time.

   Frozen scope and invariants:

   - Source is Alpaca direct ticker news only. Reddit, Seeking Alpha, Finviz news,
     retrospective timestamp repair, model training, A5.2, serving, alerts, and order
     behavior are out of scope.
   - Every poll has one UTC observation timestamp, an immutable raw-page inventory,
     explicit successful/empty/failed coverage, and a hash-bound current membership
     parent. The current authority must reproduce A4.3's identity history through its
     cutoff and may then extend that namespace to the poll date.
   - Identity joins require one effective membership interval and one observed Alpaca
     asset identity for the symbol. Missing, ambiguous, changed, or authority-stale
     identity remains excluded with a persisted reason; ticker-only fallback is
     prohibited.
   - A provider item is versioned by provider event identity, provider update time, and
     raw-content hash. `first_seen_at_utc` is the first archived observation of that
     exact version. Repeated polls cannot rewrite it, and later revisions cannot alter
     earlier versions.
   - Canonical production eligibility requires `availability_policy=observed`, complete
     collection coverage, exact identity, and strict replay. Passing this checkpoint
     starts a prospective evidence horizon; it does not satisfy A5.1 capacity floors.
   - Collection is resumable, single-process, bounded below 4 GiB, and fail-closed on
     request, parent, raw-page, revision-chain, coverage, or identity tampering.

   Exit gates: deterministic strict replay; observed-time, revision-preservation,
   identity-change, known-zero-versus-unknown, resume, duplicate, future-poison,
   extra-file, path-traversal, and artifact-tamper tests; one real Alpaca poll when
   credentials and network are available; focused and full verification; consolidated
   review; implementation commit and documentation closure commit. Rollback is removal
   of the new command/module and its new prospective authority only; A5.1 remains
   blocked and no prior immutable authority is changed.

   Implementation result:

   - Commit `5530246` adds one poll collector and one generation publisher. Polls bind
     the strict A4.3 dataset, an extending current membership authority, exact Alpaca
     asset/news URLs and query parameters, response timing, raw bodies, every provider
     revision, immutable failed attempts, and an append-only cutoff claim/commit
     registry. Parent chains replay iteratively and must use the same authority and
     registry.
   - Wrong symbols/windows, redirects, responses before the scheduled cutoff, stale or
     changed identity, partial publication, duplicate cutoffs, lineage changes, and
     registry/raw/artifact tampering fail closed. Compaction is bounded by process
     memory and verified Parquet input size. No model or feature row is emitted.
   - Verification passed 1,373 tests with two skipped, 41 focused authority/source/CLI
     tests, tracked Ruff, strict mypy across 226 source files, compileall, and one
     consolidated two-reviewer correction pass.
   - Commit `27ab9b7` accepts only the exact live or paper Alpaca asset hosts, keeps
     news on the exact data host, and verifies the malformed 2018 Twitter/Monsanto
     source by complete semantic table grammar instead of transient page-shell hashes.
   - Poll `data/raw/prospective_broker_actions/poll_20260816T071230Z`, using stable
     registry `registry_v2`, strictly replays 503 eligible security identities, 76
     observed events, 46 observed-symbol collections, and 457 known-empty collections.
     Peak working memory was 0.328 GiB. The authority remains
     `production_ready=false`, `training_eligible=false`, and
     `serving_eligible=false`; one poll starts evidence collection but cannot train A5.2.
   - The first poll occurred on Sunday and used the latest fully closed New York
     publication date, Saturday `2026-08-15`. The weekday-safe observed membership
     authority is now complete under Step 4 below. Poll
     `data/raw/prospective_broker_actions/poll_20260817T202950Z` strictly replays 11 of
     11 batches, 503 constituents, 460 observations, and 454 exact production-identity
     events at cutoff `2026-08-17T20:30:00Z`; peak working memory was 0.379 GiB.
     This expands the prospective horizon but does not satisfy the frozen A5.1 capacity
     floors or authorize A5.2 training.
   - Final verification passed 74 focused tests and the exact tracked suite with 1,382
     passed and 2 skipped; tracked Ruff, strict mypy across 227 source files, bytecode
     compilation, strict real-authority replay, and consolidated independent review
     also passed.

3. **Current S&P membership extension (`complete`).** Extend the verified
   point-in-time S&P 500 membership authority from `2026-07-08` through
   `2026-08-15` so prospective Alpaca observations can resolve security identity.

   Frozen scope and invariants:

   - Parameterize the official S&P archive collection cutoff as an immutable request
     input; retain the frozen 83-release seed authority and the `2018-04-14` lower
     boundary.
   - Collect and archive exact official S&P release bytes through `2026-08-15`, then
     rebuild event, transition, and membership authorities through that cutoff.
   - The extension must reproduce every membership interval through `2026-07-08` from
     the existing authority. Only official events effective after that date may alter
     later membership. Announced but not-yet-effective changes remain future events.
   - The cutoff anchor must be independently observed and hash-bound. It may not be
     manufactured from the same change events it is intended to verify.
   - Model training, feature changes, serving, alerts, orders, and prospective Alpaca
     collection are out of scope. Missing or contradictory official evidence fails
     closed and leaves the existing stale-membership abstention unchanged.

   Exit gates: cutoff/resume/replay/tamper and historical-prefix tests; complete raw,
   event, transition, and membership authorities through `2026-08-15`; exact semantic
   equality with the existing authority through `2026-07-08`; focused tests, full
   suite, tracked Ruff, strict mypy, compileall, memory/process check, consolidated
   review, implementation commit/push, and documentation closure commit/push.
   Rollback removes only the new parameterization and extension artifacts; the
   `2026-07-08` authority remains immutable.

   Implementation commit `42ebe5f` is pushed. It hash-binds the configurable cutoff,
   requires the inclusive cutoff day to have closed in `America/New_York`, preserves
   the complete base membership contract before the extension boundary, rejects CIK
   conflicts and inconsistent base authorities, and shares namespace verification
   with prospective collection. Verification passed 111 focused tests, tracked Ruff,
   strict mypy across 227 source files, compileall, and the full tracked suite with
   1,374 passed and 2 skipped. Peak test-process memory stayed below 0.2 GiB.

   The early same-day `v1` artifacts remain invalid and must not be used. After the New
   York day closed, the immutable `v2` raw archive collected 109 releases; event
   authority `spglobal_events_20180414_20260815_v3` published 307 events with zero
   unresolved releases; transition authority `sp500_transitions_20180529_20260815_v2`
   published 13 transitions; and membership authority
   `sp500_memberships_20180529_20260815_v2` published 1,170 intervals across 659
   securities with five governed whole-security exclusions. Strict replay verifies the
   complete authority and exact July 8 base prefix.

4. **Weekday-safe observed S&P membership authority (`complete`).** Publish a
   causal identity authority for a prospective poll without describing an unfinished
   New York publication day as a complete historical archive.

   Frozen scope and invariants:

   - The latest fully closed official S&P archive, event authority, and membership
     authority remain immutable parents. Their historical completeness contract is not
     weakened or reused for an intraday cutoff.
   - A new observation archives exact HTTP response bytes and response times for a
     contiguous newest-to-closed-cutoff prefix of the official S&P release index and
     every newly discovered membership release. Redirects, malformed pagination,
     missing release bodies, parser failures, and responses outside the approved source
     hosts fail closed.
   - Membership state at the observation cutoff applies both already-announced
     future-effective changes from the closed event authority and newly observed
     official changes whose effective time is no later than the observation cutoff.
     Provider publication dates never replace first-observed response time for a new
     release.
   - The same run archives exact bytes for an independently observed current S&P 500
     constituent anchor. The reconstructed active ticker set and every inherited CIK
     identity must agree exactly with that anchor. This equality is also the quiet-day
     completeness gate: no new release plus an unequal anchor is an abstention, not an
     inferred no-change day.
   - The resulting membership table must preserve the complete closed-authority prefix
     and the A4.3 security namespace. It records separate closed archive, observation,
     and effective horizons; these timestamps may not be collapsed into one cutoff.
   - Prospective Alpaca polling may consume only a strictly replayed observed authority
     whose observation precedes the poll cutoff and whose effective horizon covers the
     poll. Model training, A5.2, serving, scheduling, alerts, and orders remain out of
     scope.

   Exit gates: offline strict replay from exact retained bytes; closed-prefix and A4.3
   namespace equality; already-announced future-effective, same-day observed change,
   quiet-day, stale anchor, race-window, parser failure, redirect, pagination,
   future-poison, extra-file, path-traversal, and tamper tests; prospective poll
   acceptance/rejection tests; focused and full verification; tracked Ruff, strict
   mypy, compileall, memory below 4 GiB, consolidated independent review,
   implementation commit/push, and documentation closure commit/push. A real weekday
   observation remains environment evidence and cannot be replaced by a weekend test.
   Rollback removes only the new authority and poll adapter; all closed authorities and
   the Sunday poll remain immutable.

   Implementation result:

   - Commit `a5aae9b` adds the collection-only
     `collect-edge-observed-sp500-memberships` command and a strict observed-time
     authority. It retains exact no-redirect S&P search/release bytes, a complete
     second search sweep, the current constituent anchor, SEC ticker/CIK identities,
     every release outcome, observed events, pending effective changes, and canonical
     membership intervals without altering the fully closed parent authorities.
   - Poll eligibility requires the authority observation to precede the poll, remain
     within the configured 60-300 second continuity bound, and precede the next known
     pending membership change. Authority rotation must move forward in observation
     time and retain every previously observed release outcome and event. The same
     chain rules replay during strict load. Closed authorities cannot authorize a
     weekday poll; the existing weekend replay remains valid.
   - Retained-byte publication/load tests verify exact anchor identity, future-change
     timing, quiet-day equality, full multi-page race confirmation, malformed-release
     rejection, URL/redirect policy, inventory/path/symlink rejection, tamper failure,
     per-ticker poll continuity, and strict authority-chain replay. The offline fixture
     used 500 members and 5,000 SEC identities; it is test evidence, not live market
     evidence.
   - Initial verification for commit `a5aae9b` passed 152 focused tests and the complete
     tracked suite with 1,404 passed and 2 skipped. Tracked Ruff, strict mypy across 228
     source files, compileall, and consolidated independent review passed.

   Reopened evidence on `2026-08-17`:

   - The first compliant live observation reached all configured sources but failed
     closed because SEC's 10,396-record `company_tickers.json` response omitted AEP.
     The exact SEC submissions endpoint for inherited CIK `0000004904` independently
     identifies American Electric Power and lists ticker AEP. No authority was
     published.
   - Scope is limited to a causal identity fallback for tickers absent from the SEC
     bulk map. The collector must retain the exact CIK-specific SEC submissions
     response, derive the candidate CIK from the closed membership identity when
     available, verify that the SEC response lists the expected ticker and CIK, and
     replay the fallback inventory exactly. Conflicts, missing ticker claims, extra
     fallback units, redirects, or response tampering fail closed.
   - S&P release parsing, membership transitions, Alpaca polling, features, training,
     serving, scheduling, alerts, and orders are outside this correction. Exit gates
     are focused fallback/replay/tamper tests, the existing focused authority suite,
     tracked Ruff, strict mypy, compileall, a real weekday collect/load replay, and a
     new implementation plus documentation checkpoint pushed before polling Alpaca.
   - The next compliant live attempt passed the AEP fallback and then failed closed on
     XOM: the closed authority inherited CIK `0000034088`, while both the current S&P
     constituent anchor and SEC bulk identity map identify ticker XOM with successor
     registrant CIK `0002115436`. SEC's July 1, 2026 Form 8-K confirms a one-for-one
     holding-company reorganization with the same ticker, so this is an issuer identity
     transition rather than an index addition or deletion.
   - Scope therefore also admits a same-ticker identity transition only when the
     retained current S&P anchor and retained SEC identity evidence agree on a new CIK.
     The old interval closes and the successor interval opens at the observation time,
     never at a retrospective inferred date. The complete closed-authority prefix and
     old security identity remain immutable. Duplicate active identities, ticker-set
     changes, ambiguous CIKs, or replay differences fail closed.
   - Implementation commit `a7fe60e` is pushed. It retains and strictly replays the
     exact SEC CIK-specific fallback, rejects inherited-versus-anchor CIK disagreement,
     records a causally observed same-ticker successor identity, validates every raw
     unit envelope and content-addressed body path, and binds the canonical membership
     manifest to the exact request, base authority, and observation-unit inventory.
   - Real authority
     `data/raw/index_membership/sp500_observed_20260817T203000Z_v3` strictly replays 503
     current constituents, 10,397 SEC identities including the AEP fallback, zero new
     membership releases/events, and the XOM successor CIK from observation time
     `2026-08-17T20:29:05.344531Z`. It remains a collection authority, not training or
     serving authorization.
   - The immediately following Alpaca poll
     `data/raw/prospective_broker_actions/poll_20260817T202950Z` strictly replays 11 of
     11 batches, 460 observations, and 454 exact production-identity events. Final
     verification passed 152 focused tests and the complete tracked suite with 1,417
     passed and 2 skipped; tracked Ruff, strict mypy across 228 source files,
     compileall, two-reviewer remediation, and final artifact replay passed. Peak poll
     working memory was 0.379 GiB. A5.2 remains prohibited until the prospective
     horizon satisfies the frozen capacity floors.

5. **A5.1b - Publish the prospective analyst-event horizon (`completed`).**
   Aggregate one or more strictly replayed prospective broker-action generations into
   one immutable, append-only source-capacity authority. Classify each observed
   revision with the existing frozen issuer-event policy, admit only exact-identity
   Alpaca `analyst_revision` events, preserve every revision, and count one episode per
   provider event and security. Revisions must never inflate episode capacity.

   Frozen scope and invariants:

   - Each parent generation, poll, membership authority, A4.3 namespace, and registry
     identity must strictly replay. Duplicate or overlapping polls, reversed cutoffs,
     broken parent chains, namespace changes, and cross-generation security-identity
     conflicts fail closed.
   - First-seen availability is the earliest retained provider-response observation.
     Publication or provider-update timestamps cannot replace or backdate it.
     Production availability is the later of first-seen and first exact-identity
     eligibility.
   - Issuer-company classification anchors come only from the strictly replayed
     membership authority observed before the corresponding poll. A revision without a
     causal issuer-company or explicit ticker anchor may remain retained but cannot be
     admitted as an analyst episode.
   - Publish classified revisions, admitted episodes, collection coverage, and a
     source-capacity audit with exact content hashes and strict replay. Processing must
     remain below 4 GiB and accept multiple non-overlapping generations so the existing
     60-poll per-generation bound remains intact.
   - The two-poll generation
     `data/research/prospective_broker_actions_generation_20260817_v1` is valid initial
     evidence: 536 revisions, 248 provider events, 530 exact-identity revisions, and
     184 exact-identity securities. These are raw observations, not yet classified
     analyst episodes and not training rows.
   - Training, serving, model fitting, historical timestamp repair, prospective bar
     collection, feature construction, label maturation, alerts, and orders are out of
     scope. The authority always records `training_eligible=false`,
     `serving_eligible=false`, and `future_holdout_opened=false`.

   Exit gates: deterministic strict replay; duplicate/overlap, parent-chain, identity,
   revision, timestamp-poison, issuer-attribution, artifact-tamper, extra-file,
   path-overlap, deterministic-order, and memory tests; one consolidated review; full
   tracked tests, Ruff, strict mypy, compileall, and strict replay of the real horizon.
   Rollback is deletion of only the new command/module/tests and newly published
   horizon; all parent polls and generations remain immutable.

   Implementation result:

   - Commit `fe2b4c9` adds one research-only publisher and strict loader for multiple
     chronological prospective generations. It binds every parent generation, poll,
     membership authority, security namespace, registry, preflight policy, and event
     classifier; preserves revisions; counts provider-event/security episodes; and
     keeps training, serving, and future-holdout access false.
   - Corrected real authority
     `data/research/prospective_analyst_revision_horizon_20260817_v2` strictly replays
     536 revisions from 248 provider events and two polls. Only three events qualify as
     causal exact-identity analyst episodes: AMCR, HBAN, and WDAY. Eligible-security
     count is three and source capacity remains `blocked`.
   - Every coverage row is bound to that poll's exact security identity. All persisted
     timestamps use one deterministic UTC dtype. The consolidated reviewer reproduced
     and closed a Parquet round-trip defect in mixed null/non-null previous-poll times;
     the chained-poll regression and real v2 replay now pass.
   - Peak publication memory was 0.478 GiB. Final verification passed 1,431 tests with
     two skipped, tracked Ruff, strict mypy across 229 source files, compileall, focused
     causal/tamper tests, and independent reviewer verification. No model was trained
     and A5.2 remains prohibited.

6. **A5.1c - Build prospective SIP sessions and mature outcomes
   (`implementation_complete_data_pending; 1/20 source sessions`).**
   After A5.1b closes, freeze a separate append-only authority for future SIP bars,
   A4.3-identical features, and exact 30-minute labels. It must collect the complete
   contemporaneous selection cohort plus SPY, QQQ, and sector ETFs, preserve the A4.3
   namespace, attach events only at or after observed availability, and define folds
   over the causally covered prospective cohort. A5.2 training remains prohibited
   until this authority and a rerun capacity audit pass all frozen floors.

   Ordered implementation checkpoints:

   1. **Exact SIP bar transport (`completed`).** Extend the canonical Alpaca bar-page
      response with exact bounded HTTP bytes, requested/final URL, status, retrieval
      time, safe headers, and redirect evidence. Requests must use SIP, `adjustment=all`,
      ascending order, explicit `asof`, bounded pages, and no redirects. Preserve
      backward compatibility only for test doubles; production collection must reject
      pages without transport evidence.
   2. **Immutable closed-session source authority
      (`implementation_complete_data_pending`).** Publish one
      append-only child authority per fully closed XNYS session. The first phase archives
      exact Alpaca HTTP response bytes and sidecars for SIP five-minute bars covering the
      complete membership cohort observed before that session, plus SIP one-minute bars
      for SPY, QQQ, and the sector ETFs. Strict replay must reconstruct canonical bars
      from those retained bytes, bind the exact membership parent and request identity,
      reject redirects, gaps, duplicate pages, and post-session mutation, and support
      hash-verified crash-safe resume below 4 GiB. The second phase may collect selected
      stock one-minute paths only when the target session has twenty contiguous prior
      five-minute sessions under the same causal namespace. A source-complete session is
      explicitly selection-ineligible while that warm-up is absent; stale July bars may
      not activate an August cohort. This checkpoint builds no features or labels and
      authorizes neither training nor serving.
      Benchmark one-minute grids must be complete. Full-cohort stock gaps are retained
      as explicit coverage evidence and may authorize the session only within the frozen
      whole-security exclusion ceiling of 5%; no bars are imputed, and coverage above
      that ceiling publishes status only without a parent source authority.
   3. **Causal feature authority (`not_started`).** Reproduce the twenty-session
      activation cohort and reuse the exact A4.3 volume-bar and feature transformation.
   4. **Mature outcome authority (`not_started`).** Publish exact stock/SPY/QQQ/sector
      30-minute paths only after availability; keep labels separate from features.
   5. **Matched prospective preflight (`not_started`).** Attach only observed analyst
      episodes available by decision time and evaluate the frozen capacity floors.

   The exact transport checkpoint changes only the Alpaca source contract and focused
   tests. It does not collect data, alter historical authorities, build features or
   labels, train models, or authorize serving. Exit gates are exact-byte and URL/query
   tests, redirect/status/body-integrity failures, existing collector compatibility,
   tracked tests, Ruff, strict mypy, and compileall. Rollback removes only the new bar
   transport fields and byte-fetch path.

   Exact transport result:

   - Commit `c56843e` replaces parsed-only bar fetches with a bounded exact-byte HTTP
     response. `AlpacaBarsPage` now carries raw bytes, safe headers, requested/final
     URL, status, retrieval time, and redirect evidence while retaining parsed bars for
     existing collectors.
   - Production bar requests require consolidated SIP, an explicit point-in-time
     `asof`, `adjustment=all`, ascending order, at most 50 symbols, and at most 10,000
     rows. Strict semantic URL/query validation rejects host/path/query changes,
     redirects, non-200 status, non-JSON content, naive retrieval times, and body
     hash/length/representation mismatches.
   - Focused source and collector compatibility verification passed 44 tests. The full
     repository suite passed 1,433 tests with two skipped; tracked Ruff, strict mypy
     across 229 source files, compileall, and one consolidated independent review also
     passed. No market data was downloaded and no model or authority changed.

   Closed-session source implementation result:

   - Commit `bf35a11` adds the immutable prospective session parent and the
     `collect-edge-prospective-sip-session` command. It collects exact point-in-time
     cohort five-minute SIP bars and exact SPY, QQQ, and sector-ETF one-minute SIP bars
     only after an XNYS session closes and before the next session opens.
   - Strict replay binds exact provider bytes, transport sidecars, request identity,
     membership lineage, exchange-calendar bounds, coverage, and child-authority
     fingerprints. Benchmark gaps fail the source gate; stock gaps remain explicit and
     may not exceed the frozen 5% whole-security exclusion ceiling. Resource policy is
     capped at 4 GiB and two workers.
   - Final verification passed 1,453 tests with two skipped, tracked Ruff, strict mypy
     across 230 source files, compileall, and independent review with no remaining high-
     or medium-severity finding.
   - No real source parent has been published. Collection requires a fresh observed
     membership authority after the preceding session close and before the target
     session open, followed by collection after target close plus 60 seconds. A5.1c
     remains open, and feature, outcome, preflight, training, and serving work remain
     prohibited until their preceding authorities pass.
   - The first post-close observation on `2026-08-19` failed closed because the
     independent anchor contained `VMRK` while the active lineage contained `EQR`.
     Retained official and SEC evidence identifies this as a ticker successor on the
     same CIK, not an addition/deletion. This reopens only observed-membership anchor
     reconciliation: one anchor-only ticker may replace one active-only ticker at the
     observation time only when their CIK is identical and uniquely matched. Different
     or ambiguous identities remain fatal. Exit evidence is a regression test, strict
     replay of the real observation, and the existing focused verification suite.
   - Commit `6386cb4` implements the bounded reconciliation. It activates a ticker
     successor only at observation time when one anchor-only ticker and one active-only
     ticker share one unique CIK; a pending future event, ambiguous match, or different
     CIK remains fatal. Public collection and strict replay regressions cover the path.
   - Failed immutable attempt
     `data/raw/index_membership/sp500_observed_20260819T200115Z_v4` remains failure
     evidence. Complete authority
     `data/raw/index_membership/sp500_observed_20260819T201500Z_v5` strictly replays 503
     constituents at `2026-08-19T20:06:40.779393Z`, closes `EQR`, opens `VMRK` on the
     same `cik:0000906107`, and publishes universe hash
     `5b6e68d4844f9b0baa00e517bf0b515ddc1648a730776bf9dba201cf9082b1b3`.
   - Final verification passed 1,457 tests with two skipped, tracked Ruff, strict mypy
     across 230 source files, tracked compilation, and independent re-review with no
     remaining high- or medium-severity finding.
   - Real authority
     `data/raw/prospective_sip_sessions/session_20260820_v1` now strictly replays the
     first eligible source session: 503 stocks, 39,181 five-minute stock rows, and
     5,070 one-minute benchmark rows. Twenty-two stocks are incomplete, or 4.374%,
     below the unchanged 5% ceiling; no benchmark is incomplete. Peak working memory
     was 0.289 GiB. Status is `source_complete_warmup_ineligible` because only one of
     twenty required prior five-minute sessions exists. Features, outcomes, preflight,
     training, and serving remain prohibited.
   - Pre-open authority
     `data/raw/index_membership/sp500_observed_20260821T071500Z_v7`, poll
     `data/raw/prospective_broker_actions/poll_20260821T071000Z`, generation
     `data/research/prospective_broker_actions_generation_20260821_v1`, and separate
     horizon `data/research/prospective_analyst_revision_horizon_20260821_v4` all
     strictly replay. The poll contains 451 identity-bound observations and the new
     causal chain contains 12 qualifying analyst episodes. It remains capacity-blocked.
     The August 17 and August 21 chains cannot be combined because polling was not
     contiguous; the publisher failed closed rather than manufacturing continuity.

7. **A5.1d - Correct historical intraday event identity and run matched development
   training (`completed`).** The A5.1 preflight attached events by literal
   `security_id`. Historical event authorities encode most SEC identities as
   `cik:<value>:ticker:<symbol>`, while A4.3 uses `cik:<value>`. A reproduced audit
   found 17,401 direct-issuer analyst episodes, 15,603 episodes whose ticker occurs in
   A4.3, but only 97 episodes with a literal security-ID overlap and 19 attached
   episodes. This is an identity-normalization defect, not an event-capacity result.

   Frozen correction and training contract:

   - Reconcile events and source-coverage rows to the A4.3 namespace by exact uppercase
     ticker. When both source and target identities contain CIKs, unequal CIKs fail the
     publication. Ambiguous ticker/security mappings abstain and are audited; they are
     never guessed.
   - Preserve the original event and coverage security IDs as lineage. Do not rewrite
     either parent authority. Publish one new immutable preflight authority and require
     strict replay, identity-tamper, ambiguity, future-evidence, and coverage tests.
   - Historical `provider_publication_proxy` timestamps may support research-only
     development experiments. They may not set production eligibility, open the future
     holdout, authorize serving, or satisfy prospective promotion floors.
   - Compare four explicit research families: swing technical, swing technical plus
     broker catalyst, intraday technical, and intraday technical plus broker catalyst.
     Catalyst comparisons use the same decision rows, labels, temporal folds, security
     holdout, costs, and estimator budget as their technical controls. Missing catalyst
     coverage causes abstention and is never encoded as zero.
   - Existing verified historical authorities are inputs; no provider download occurs.
     Swing and intraday jobs run sequentially under their 5 GiB and 4 GiB caps.
   - Selection uses development/validation data only. ROC-AUC is diagnostic and remains
     co-gated by calibration, after-cost benchmark-relative economics, drawdown, trade
     count, and fold stability. The post-2026-07-08 intraday holdout remains closed.
   - Exit evidence is the corrected attached-episode count, strict authority replay,
     matched technical-versus-catalyst evaluation, focused poison tests, repository
     tests, tracked Ruff, strict mypy, and compile verification. Rejected families must
     publish `no_candidate`; blocked families must publish the exact blocker.

   Implementation result:

   - Implementation commits `4cc5d4e` and `cda4c1f` correct identity attachment and add
     the research-only event-confirmed training path. Exact ticker plus CIK-compatible
     reconciliation corrects the namespace defect while
     preserving source identities. Conflicting CIKs fail publication and ambiguous
     ticker mappings abstain. Focused identity, future-evidence, immutable replay, and
     verified-parent reuse tests pass.
   - Corrected authority
     `data/research/edge_rebuild_intraday_event_preflight_20260820_v2` strictly replays
     17,401 research episodes. It attaches 1,912 unique episodes to 83,636 A4.3
     event/decision pairs, compared with 19 episodes and 862 pairs before correction.
     Production remains blocked because all historical availability is a retrospective
     provider-publication proxy.
   - Catalyst remains a confirmation/population filter, not a direct intraday model
     feature. The event-confirmed continuation population has 14,451 development rows;
     long reversion has 12,951. Both use the unchanged technical feature contract,
     four chronological folds, one-session embargo, stable 20% security holdout, and
     frozen cost/economic gates.
   - Continuation positive-return ROC-AUC is 0.513 seen / 0.509 unseen. Long-reversion
     ROC-AUC is 0.535 / 0.519, versus technical-only 0.513 / 0.508. Neither family
     passes: selected policies have negative average trade and daily returns after
     costs, profit factor below one, negative benchmark excess, weak fold stability,
     and failed stop-risk calibration. Both publish strict `no_candidate` authorities.
   - Outputs are
     `data/models/edge_rebuild_intraday_event_confirmed_continuation_dev_20260820_v1`
     and
     `data/models/edge_rebuild_intraday_event_confirmed_long_reversion_dev_20260820_v1`.
     Peak working set remained about 3.16 GiB after eliminating duplicate dataset loads;
     the 4 GiB hard cap and 3.25 GiB safety threshold were not weakened. The future
     holdout stayed closed and serving/promotion remain prohibited.
   - Final verification passed 1,456 tests with two skipped, tracked Ruff, strict mypy
     across 231 source files, and tracked compilation. One unrelated microstructure
     retry test failed on the first full-suite run and then passed in isolation, as a
     complete file, and in the clean full-suite rerun.

8. **A5.1e - Evaluate directional intraday broker-action cohorts (`completed; no candidate`).**
   Split the corrected A5.1d research cohort with the existing governed analyst-rule
   classifier. Test upgrades and downgrades as separate 30-minute research
   specialists; audit coverage initiations independently and block them when capacity
   is insufficient. Catalyst remains a cohort filter around the unchanged technical
   estimator and is not added to the model feature vector.

   Frozen contract and exit gates:

   - Reuse the immutable A4.3 dataset, corrected A5.1d preflight, exact event identity,
     four chronological folds, one-session embargo, stable unseen-security holdout,
     feature order, costs, estimators, and economic gates. No provider download occurs.
   - Admit a directional subtype only with at least 500 independent announcements,
     200 securities, 200 sessions, 100 announcements in every validation fold, and 100
     announcements represented in the unseen-security scope. These floors prevent one
     issuer, time period, or seen-security population from dominating the result.
   - Only `bare_upgrade`, `bare_downgrade`, and `coverage` are eligible subtype names.
     Price-target and generic actions remain excluded because their direction is not
     governed by this hypothesis.
   - Train eligible subtype/hypothesis combinations sequentially under the unchanged
     4 GiB process cap. Publish `no_candidate` when validation fails and an explicit
     capacity blocker when a subtype fails the frozen floors.
   - The historical publication-time proxy remains research-only. Future holdout,
     serving, promotion, and A6 stay closed regardless of development performance.
   - Exit evidence is deterministic subtype classification, capacity and future-poison
     tests, immutable parent binding, strict output replay, focused tests, full tests,
     Ruff, strict mypy, compilation, memory evidence, implementation commit, and
     documentation closure.

   Implementation result:

   - Commit `4821780` adds one strict directional-cohort path to the existing research
     trainer. Parent authorities are verified once; only classification fields are
     retained before large parent tables are released. Subtype, capacity, and parent
     hashes remain in model lineage. Unsupported or under-capacity subtypes fail before
     the A4.3 training matrix is loaded.
   - Upgrade capacity passed with 805 announcements, 32,970 attached decisions, 285
     securities, 461 sessions, 149-202 announcements per validation fold, and 180
     unseen-security announcements. Downgrade capacity passed with 860 announcements,
     31,243 decisions, 273 securities, 439 sessions, 124-200 per fold, and 173 unseen.
   - Coverage initiation was blocked without training: 245 announcements, 169
     securities, 168 sessions, 38-55 per fold, and 41 unseen-security announcements.
     Price-target and generic actions remained excluded.
   - Upgrade continuation trained on 6,709 rows and produced seen/unseen positive-net-
     return ROC-AUC of 0.492/0.531. Upgrade long reversion trained on 5,341 rows and
     produced 0.495/0.494. Downgrade continuation trained on 5,991 rows and produced
     0.510/0.485. Downgrade long reversion trained on 6,041 rows and produced
     0.506/0.529.
   - All four are strict `no_candidate` outputs. No family reached the 0.60 AUC gate.
     Positive economics in isolated scopes were based on only 14-35 unseen-security
     trades and failed in the paired seen-security scope. Other scopes had negative
     after-cost trade returns, profit factor at or below one, or failed benchmark,
     calibration, confidence-bound, and fold-stability gates.
   - Immutable outputs are the four
     `data/models/edge_rebuild_intraday_{upgrade|downgrade}_confirmed_{continuation|long_reversion}_dev_20260820_v1`
     directories. All strictly replay as `no_candidate`; future holdout and serving
     remain closed. Peak working set was 3.193 GiB under the unchanged 4 GiB hard cap
     and 3.25 GiB safety threshold.
   - Final verification passed 1,460 tests with two skipped, tracked Ruff, strict mypy
     across 231 source files, tracked compilation, real capacity replay, and strict
     replay of all four outputs.

### A6 - Run Locked Evaluation and Promote Qualified Models

- Use purged, embargoed walk-forward validation plus a stable held-out-security stress
  test within each validation period. The security test measures transfer to unseen
  symbols; it is not an additional independent time period. Swing uses the frozen
  approximately 5/1/1-year sequence; intraday uses
  the maximum causally complete 2-3-year history with frozen calendar boundaries.
- Open each locked test once for a preregistered candidate. Report ROC-AUC, PR-AUC,
  Brier/ECE, rank IC, top-quantile lift, net SPY/QQQ/sector excess return, costs,
  turnover, drawdown, capacity, regime stability, and coverage.
- A specialist passing on a narrow cohort remains a specialist. The API abstains
  outside its verified coverage.

## Promotion Gates

A model is not promoted because materialization or training succeeds. Promotion
requires:

- immutable source, feature, label, split, and model lineage;
- exact batch/live ordered-feature parity;
- chronological and unseen-security stability;
- calibration and ranking value;
- positive net economics after costs with acceptable drawdown and capacity;
- regime, sector, and market-cap stability;
- prospective shadow evidence;
- a hash-verified atomic serving bundle.

Until then, production scoring and prediction API paths fail closed. Rejected models
are audit evidence, never fallbacks.

## Completion Checklist

- [x] Remove Reddit and Seeking Alpha from the active system.
- [x] Freeze Alpaca as the only ticker catalyst estimator source.
- [x] Enforce `2019-07-09` as the first swing model decision date.
- [x] Publish and economically reject intraday V2.
- [x] Complete repository-wide verification and memory audit.
- [x] Publish and strictly replay the single-profile V12 technical swing authority.
- [x] Preserve prior failed candidates as rejection evidence, never serving fallbacks.
- [x] Record the historical Intraday V3 experiment as invalid because five declared
  cross-sectional z-score inputs lacked a causal decision-cohort implementation.
- [x] Preserve learning-to-rank as a future estimator family, not as evidence that the
  invalid V3 feature contract was repaired.
- [x] Complete A0 research-integrity recovery and push implementation commit
  `e168482` plus its documentation closure.
- [x] Complete A2 governed swing-baseline ablation and serving contracts in
  implementation commit `cb2aba5`; no performance or promotion claim was made.
- [x] Complete A3.1-A3.4 issuer-event authority, precision, and matched-ablation work.
- [x] Separate rating changes from coverage initiation, keep price-target changes
  report-only, run the chronological capacity audit, and complete all 12 frozen A3.5
  development experiments without opening the locked test. No candidate passed.
- [ ] Build and backfill the replacement A4 intraday feature authority before
  collecting a new locked holdout; the invalid V3 contract cannot be reused.
- [ ] Promote only a model that passes every gate.


## Paused Structural Repair Checkpoint

The August 24 review reopened the incomplete package refactor with reproducible
correctness and verification failures. This checkpoint changes code structure only;
model features, labels, thresholds, data authorities, and promotion state are out of
scope.

The canonical source layout is domain-based: `core`, `sources`, `evidence`,
`universe`, `catalysts`, `modeling`, `swing`, `intraday`, `governance`, `serving`,
`commands`, and `research`. Swing and intraday each own descriptive `contracts`,
`datasets`, `features`, `labels`, `training`, `evaluation`, and `live` packages.
Chronology and checkpoint labels are prohibited in active package, module, command,
test, and task names.

1. **Holdout access and shared contract repair (`completed`).**
   Use one `IntradayDevelopmentConfig`, restore constructible causal calibration
   results, repair the direct future-holdout validation path, and cover the unmocked
   path with regression tests. Future access must remain fail-closed and auditable.
   Implementation commit `99f635c` is pushed. The direct holdout path, atomic claim,
   reservation/failure evidence, registry isolation, temporary replay validation,
   shared config, and calibration construction are covered by 57 passing focused tests.
   The assigned senior reviewer accepted the bounded diff after all P1 findings were
   fixed; focused Ruff and strict mypy pass.
2. **Serialized artifact and namespace inventory (`completed`).**
   Inventory every command, manifest, model artifact, and import that depends on a
   chronology-named namespace. Retrain retained models under canonical modules or
   explicitly retire rejected artifacts before deleting their code dependencies.
   Implementation commit `3026450` is pushed. The machine-readable retention inventory
   classifies every artifact group needed by the research catalog, tracked hash-bound
   evidence, active replay claims, old serialized namespaces, and ungoverned outputs.
   The four research catalog models now use behavior-based IDs and local paths. Their
   authority, manifest, and candidate hashes are unchanged; only swing technical has a
   non-actionable research candidate, and all four remain promotion-ineligible. No
   model output was deleted because tracked evidence or incomplete regeneration
   provenance still blocks broader cleanup. Ten focused tests, touched Ruff, strict
   mypy, default-service smoke, retained-bundle hash checks, and specialist evidence
   replay checks passed. The assigned senior reviewer accepted the bounded diff.
3. **Market evidence and research package migration (`completed`).**
   Consolidate base contracts under `core`, immutable lineage under `evidence`, source
   transports under `sources`, membership and identity under `universe`, issuer and
   global events under `catalysts`, reusable estimators and validation under
   `modeling`, and non-production experiments under `research`.
   - **Cross-sectional research consolidation (`completed`).** Implementation commit
     `ade847c` removes the `market_predictor.v3` source package and replaces active
     chronology-named APIs and tests with behavior-based names. Generic contracts are
     split among `core`, `evidence`, `modeling`, and `universe`; raw S&P Global archive
     transport is under `sources/spglobal`; verified index changes and point-in-time
     membership are under `universe/sp500`; reusable validation, calibration, and
     ranking economics are under `modeling`; and candidate evaluation and development
     experiments remain under `research`. An AST guard prevents production imports of
     `research` or `commands`. The two audited rejected joblibs serialized against the
     removed namespace were deleted while their manifests were retained; no other
     model artifact was removed. Across the migrated modules and direct consumers, 330
     focused test cases passed. Ruff, strict mypy on 44 source files, CLI import smoke,
     diff checks, and post-review consumer reruns passed. The assigned senior reviewer
     accepted the final diff with no P0, P1, or P2 finding.
   - **Historical membership and security identity authority migration (`completed`).**
     Implementation commit `60cff69` moves corpus integrity to `evidence`, membership
     identity validation and SEC identity authority to `universe`, and historical S&P
     transition and membership authorities to `universe/sp500`. Every direct consumer
     now imports the semantic package; the five old modules are absent and guarded
     against reintroduction across source, tests, and scripts. A universe dependency
     allowlist enforces the current lower-layer boundary. Persisted schemas, hashes,
     locking, memory gates, and authority behavior are unchanged. Sixty authority tests
     and 106 consumer tests passed before review; four additional import-form poison
     tests passed after review remediation. Ruff, strict mypy on 15 source files,
     import smoke, diff checks, and process checks passed. The assigned senior reviewer
     accepted the final diff with no P0, P1, or P2 finding.
   - **Prospective observed-membership source and authority split (`completed`).**
     Implementation commit `5259bdb` moves provider URLs, HTTP collection, response
     validation, retained raw units, source parsing, and raw replay to
     `sources/spglobal/observed_membership_collection.py`. Observed membership lineage,
     identity reconciliation, effective-state construction, publication, and strict
     authority replay now belong to
     `universe/sp500/observed_membership_authority.py`. The universe orchestrator keeps
     the single operation lock and generates the unchanged request hash passed into
     every raw unit. Authority-root inventory and raw `objects`/`units` inventory are
     verified separately. The old combined module is absent; all consumers use the
     semantic authority package. Architecture tests prevent source-to-universe imports
     and all import forms of the removed path. Exact raw-envelope and lock-contention
     tests pass. Final checkpoint verification passed 113 affected tests, Ruff, strict
     mypy on six source files, compileall, import smoke, and diff checks. The assigned
     senior reviewer accepted the diff with no P0, P1, or P2 finding.
   - **Issuer and global catalyst authority migration (`completed`).** Move the
     remaining source-independent issuer-event, SEC-filing, global-event, and
     market-context authorities out of `edge_rebuild` into `catalysts`, `sources`, and
     `evidence` without changing source coverage, availability, or causal semantics.
     - **SEC filing evidence and decision authority (`completed`).** Implementation
       commit `9244893` creates `catalysts/sec_filings`, moves causal filing collection
       evidence to `collection.py`, and moves decision-time overlays to
       `decision_authority.py`. `sources/sec.py` remains the provider transport. The
       two old modules and edge-rebuild-prefixed tests are absent and guarded against
       reintroduction. Persisted schemas, availability, coverage missingness, raw
       replay, artifact hashes, memory limits, and command behavior are unchanged.
       Forty-four SEC, architecture, and CLI tests passed with Ruff, strict mypy on
       three source files, compileall, import smoke, zero-reference and diff checks.
       The assigned senior reviewer accepted the AST-equivalent move with no P0, P1,
       or P2 finding.
     - **GDELT collection and global-event authority (`completed`).** Implementation
       commit `5e1f65f` makes `sources/gdelt.py` the single strict provider transport,
       moves immutable canonical evidence into `catalysts/global_events/collection.py`,
       and moves the decision-time overlay into
       `catalysts/global_events/decision_authority.py`. Active APIs, commands, and
       tests use behavior names; the old modules and duplicate transport are absent.
       Provider URL identity, no-redirect behavior, request parameters, raw-response
       hashes, query/scorer policy hashes, canonical event hashes, availability,
       coverage, and persisted schema strings remain fail-closed and replay-compatible.
       Ninety-eight focused tests passed with Ruff, strict mypy on six source files,
       compileall, architecture guards, and diff checks. The assigned senior reviewer
       accepted the final diff with no P0, P1, or P2 finding.
     - **Issuer event family and precision authorities (`completed sequence`).** Move causal
       issuer evidence in four independently reviewed checkpoints without changing
       any classifier, coverage, audit, or artifact contract:
       1. **Alpaca issuer-news evidence collection and audit (`completed`).**
          Implementation commit `4f271ca` moves the immutable collector and strict
          audit from `swing` to `catalysts/issuer_events`; `sources/alpaca.py` remains
          the provider transport. Persisted schema strings are unchanged in
          `news_history_contracts.py`. Canonical symbols now belong to `core/symbols.py`
          and provider-specific mappings to `sources/provider_symbols.py`, avoiding a
          reverse catalyst dependency on the old mixed symbol module. Old modules,
          imports, and tests are absent and guarded against reintroduction. Verification
          passed 109 focused parity tests and the complete suite with 1,530 passed and
          2 skipped, plus affected-file Ruff, strict mypy on 14 source files,
          compileall, dependency/file-absence guards, and diff checks. The assigned
          senior reviewer found no P0, P1, or P2 issue.
       2. **Classification and attribution foundations (`completed`).**
          Implementation commit `2b9e195` moves reusable event-family classification,
          relevance, attribution, and attribution-history behavior to
          `catalysts/issuer_events`. The rule-variant helper has one semantic owner in
          `classification.py`; an AST guard rejects definitions, imports, or assignment
          aliases in its old precision-audit owner. Exact policy hashes, schema/version
          strings, every rule-variant branch, representative outputs, and strict replay
          remain fixed. Old modules, imports, and swing-prefixed foundation tests are
          absent and guarded. Verification passed 247 focused parity tests, 88 tests
          after reviewer fixes, 54 ownership/dependency tests, and the final complete
          suite with 1,551 passed and 2 skipped. Affected-file Ruff, strict mypy on 12
          source files, compileall, removed-module scans, diff checks, and process checks
          passed. The assigned senior reviewer accepted the final diff with no P0, P1,
          or P2 finding.
       3. **Swing catalyst decision authority (`completed`).** Implementation commit
          `9408515` moves the decision-time swing feature authority to
          `swing/features/catalyst_decision_authority.py`. All consumers use the new
          semantic path directly; the old module and test are absent and guarded.
          Persisted request, authority, manifest, lineage, decision-artifact, and
          coverage-artifact identities remain unchanged. A bidirectional architecture
          guard now prohibits imports between `swing` and `intraday`. Verification
          passed 171 focused tests with one skipped and the complete suite with 1,569
          passed and 2 skipped. Affected Ruff, strict mypy on six source files,
          compileall, removed-path scans, diff checks, and memory/process checks passed.
          The assigned senior reviewer accepted the final diff with no P0, P1, or P2
          finding.
       4. **Issuer-family evidence and horizon assignment split (`completed`).**
          Implementation commit `03f8233` preserves the two retained combined v2
          envelopes byte-for-byte while separating their runtime ownership. Strict
          structural verification and a swing-independent neutral projection identity
          belong to `evidence/issuer_family_combined_envelope.py`; neutral classified
          events, coverage, and unclassified semantic replay belong to
          `catalysts/issuer_events/family_evidence.py`; swing assignments and cohort
          replay belong to `swing/datasets/issuer_event_family_cohort.py`. Intraday now
          consumes only neutral evidence and cannot access swing assignments. A true
          persisted-authority split would change schemas and hashes, so it remains a
          separately approved data migration rather than part of this byte-preserving
          checkpoint. Verification passed 178 focused tests and the complete suite with
          1,584 passed and 2 skipped. Both retained v2 eras passed strict real-data
          replay below 2 GiB, affected-file Ruff and strict mypy passed, and the assigned
          reviewer accepted the final diff with no P0, P1, or P2 finding.
       5. **Issuer-event precision governance (`completed`).** Implementation commit
          `7ce23a0` moves deterministic sampling, blind review resolution, artifact
          integrity, and family/rule-variant admission into
          `governance/issuer_event_precision`. The old combined module is absent and
          guarded against reintroduction; command names and swing-ablation behavior
          are unchanged. Public loaders remain strict, staged publication validates
          fully rewritten final paths before atomic rename, and failure-injection tests
          prove invalid authorities are never made visible. Both retained periods
          replay with unchanged sample/audit authority hashes and 1,796/1,859 review
          rows. Verification passed 120 affected tests with one skipped, 25 final
          governance tests with one skipped, and the complete suite with 1,594 passed
          and three skipped. Affected Ruff, strict mypy, compileall, removed-path
          scans, diff checks, and process checks passed. The assigned senior reviewer
          accepted the final diff with no P0, P1, or P2 finding.
       The required dependency direction is `sources -> catalysts -> swing ->
       governance`; commands remain outer adapters.
4. **Swing and intraday package migration (`paused`).**
   Consolidate each horizon under descriptive `contracts`, `datasets`, `features`,
   `labels`, `training`, `evaluation`, and `live` packages and remove the intraday
   evaluation module/package collision. Compatibility aliases are prohibited because
   this repository is not deployed.
   - **Intraday module/package collision removal (`completed`).** Implementation
     commit `a176fbb` deletes the unreachable `intraday/contracts.py` and
     `intraday/evaluation.py` shadows. Runtime and pickle ownership remain in the
     existing canonical packages; no consumer import or artifact identity changed. A
     recursive architecture guard rejects future module/package collisions, with an
     explicit poison fixture and deleted-file assertions. Characterization freezes
     package origins, schema and feature-order hashes, label policy/hash, validators,
     pickle ownership, and deterministic evaluation outputs. Verification passed 143
     affected tests and the complete suite with 1,601 passed and three skipped. Touched
     Ruff, compileall, direct-path scans, diff checks, cleanup, and process checks
     passed. The assigned senior reviewer accepted the diff with no P0, P1, or P2
     finding.
   - **Shared strategy contract migration (`completed`).** Implementation commit
     `c408d58` moves the cross-horizon contract from `edge_rebuild` to `modeling` and
     updates every consumer directly without a compatibility alias. The schema string
     remains `edge_rebuild.strategy_contract.v2`, the active configuration SHA-256
     remains `39213ad6bd5c1f09f30065f737ffecadf05bbb0ae81b81f2ffda7a343967e972`,
     and retained artifact scans found no serialized Python owner at the removed path.
     Architecture guards reject every old import form and reintroduction of the removed
     file. Verification passed 452 affected tests with two skipped and the complete
     suite with 1,605 passed and three skipped. Ruff on the migrated authority and its
     boundary tests, strict mypy, compileall, import smoke, removed-path scans, diff
     checks, and process-memory checks passed. The assigned senior reviewer accepted
     the final diff with no P0, P1, or P2 finding.
   - **Shared mathematical primitive ownership (`completed`).** Implementation commit
     `8d42d26` moves the only horizon-neutral authority found in this pass,
     `FeatureStep` and `FeaturePipeline`, from `edge_rebuild` to
     `modeling/feature_pipeline.py`. The implementation is byte- and AST-identical;
     swing and intraday consumers import the new owner directly, no alias exists, and
     architecture tests reject every old import form and old-file reintroduction. The
     review explicitly keeps cross-sectional scaling and technical relationships out of
     `modeling`: their current policies and consumers are swing-specific, so they move
     later to `swing/features`. The mixed label module must be split during the horizon
     label migrations rather than moved wholesale. Verification passed 119 focused
     tests and the complete suite with 1,613 passed and three skipped. Touched Ruff,
     strict mypy, compileall, code-hash parity, removed-path scans, diff checks, and
     process-memory checks passed. The assigned senior reviewer accepted the final diff
     with no P0, P1, or P2 finding.
   - **Intraday history-collection contract migration (`completed`).** Implementation
     commit `09341dc` moves the complete intraday Alpaca/SIP history acquisition
     contract from `edge_rebuild/history_contracts.py` to
     `intraday/contracts/history_collection.py`. The implementation is byte- and
     AST-identical with source hash
     `6b5d3b42c73aeb40958ca01b5a35b2a821d1de46`; all collectors, intraday datasets,
     command adapters, and tests import the new owner directly. No compatibility alias
     exists. New characterization freezes all eight Pydantic owners and the six active
     configuration schema/hash pairs. Architecture guards reject every old import form
     and old-file reintroduction, while existing dependency guards prohibit provider
     sources from importing horizon code. Verification passed 171 focused tests, 56
     interrupted-refactor regression tests, and the complete suite with 1,631 passed
     and three skipped. Touched Ruff, strict mypy, compileall, CLI import/help, exact
     code-hash parity, artifact scans, diff checks, and the 4 GiB process-memory gate
     passed. The assigned senior reviewer accepted the final diff with no P0, P1, or
     P2 finding.
   - **Swing contract package and materialization ownership (`completed`).**
     Implementation commit `26c048d` converts `swing/contracts.py` byte-for-byte into
     `swing/contracts/__init__.py`, preserving the Python and pickle owner
     `market_predictor.swing.contracts`, and moves the swing materialization schema
     constants from `edge_rebuild` to `swing/contracts/materialization.py`. All
     consumers use the canonical materialization module directly; regression tests
     prohibit accidental constant aliases on the legacy materialization and training
     modules. Both moves are byte- and AST-identical, with source identities
     `36b698837a09a8cd0b23e9b48e4be291afa91727` and
     `c7add055ab12ab53d46988f89da862f0a631649a`. Characterization freezes config
     owners and pickle round trips, both materialization schemas, the 99-feature and
     53-feature profile hashes, three default config hashes, and the label-policy hash.
     Verification passed 208 affected tests with one skipped and the complete suite
     with 1,645 passed and three skipped. New-file Ruff, changed import-order Ruff,
     strict mypy, compileall, import/no-alias smoke, old-path and artifact scans, diff
     checks, and the 4 GiB process-memory gate passed. Known pre-existing unused
     re-export findings in `swing_training.py` remain assigned to Step 6. The reviewer
     found one accidental-alias P2, verified its fix, and accepted the final diff with
     no remaining P0, P1, or P2 finding.
   - **Swing technical-relationship feature ownership (`completed`).** Implementation
     commit `dd4dbcd` moves `technical_relationships.py` byte-for-byte from
     `edge_rebuild` to `swing/features`, updates both lazy pipeline consumers directly,
     and renames the characterization test for descriptive ownership. Source identity
     remains `391bac1540b6ef414dced0338b842cedc5e54bdb`; no alias exists. Tests freeze
     the new `TechnicalRelationshipSpec` owner and pickle round trip, nine-column
     output-order hash, strategy-derived specification hash, representative output
     hash, future-prefix causality, and session-boundary resets. Architecture guards
     reject all old import forms and old-file reintroduction. Verification passed 170
     affected tests with two skipped and the complete suite with 1,651 passed and three
     skipped. New-owner Ruff, changed import-order Ruff, strict mypy, compileall,
     import smoke, source parity, old-path and artifact scans, diff checks, and the 4
     GiB memory gate passed. The reviewer accepted the final diff with no P0, P1, or
     P2 finding.
   - **Swing cross-sectional feature ownership (`completed`).** Implementation commit
     `68d9893` moves `cross_sectional.py` byte-for-byte from `edge_rebuild` to
     `swing/features`, updates all three consumers with module-qualified imports, and
     renames its characterization test descriptively. Source identity remains
     `cfb54c43d06382235fd341d9a9713a5262715c4f`; no alias exists. Tests freeze the
     `CrossSectionSpec` owner and pickle round trip, suffixes, specification and emitted
     column hashes, representative output hash, future-session causality, session and
     sector isolation, winsorization, peer floors, collision handling, and empty-frame
     behavior. Verification passed 181 affected tests with two skipped and the resumed
     complete suite with 1,660 passed and three skipped. New-test and changed-import
     Ruff, strict mypy, compileall, import smoke, source parity, old-path and artifact
     scans, diff checks, and the 4 GiB memory gate passed. The moved byte-identical file
     retains its pre-existing import-spacing Ruff finding for Step 6. The reviewer
     accepted the final diff with no P0, P1, or P2 finding.
   - **Shared label outcomes and swing barrier/rank ownership (`completed`).**
     Implementation commit `61f6f4c` makes `modeling/label_outcomes.py` the only
     owner of the six integer outcome/rank constants, converts `swing/labels.py`
     byte-for-byte to `swing/labels/__init__.py`, and moves the daily barrier and
     session/sector ranking implementation to `swing/labels/barrier_and_rank.py`.
     Swing and intraday consumers use module-qualified canonical imports; no alias or
     old file remains. Tests freeze constant, specification, column, representative
     output, dtype, package, and pickle identity; append-only causality and group
     isolation remain explicit. Modeling is now guarded against absolute and relative
     imports from either horizon. Verification passed 155 direct label/boundary tests,
     133 broader regression tests with two skipped, and the complete suite with 1,681
     passed and three skipped in 21 minutes 51 seconds. Affected Ruff, strict source
     mypy, compileall, old-path scans, staged diff checks, temporary-output cleanup,
     and the 4 GiB process gate passed. The assigned reviewer verified all P2 fixes and
     approved the final diff with no remaining P0, P1, or P2 finding.
   - **Intraday causal volume-bar dataset ownership (`completed`).** Implementation
     commit `e76bf8d` moves the byte-identical implementation to
     `intraday/datasets/volume_bars.py`, renames its test descriptively, and updates all
     three dataset consumers directly. No alias or old file remains. Because the
     transformation identity hashes `bar_dataset.py` itself, the direct import change
     truthfully creates schema `market_predictor.intraday.bar_dataset_transformation.v2`
     with aggregate SHA-256
     `6fdfd0c8f07e4f7445b66d038cbd936e4459db68e087a5ddbcb30eac4795cb51`.
     Source hashing now canonicalizes only CRLF/LF so Windows publication and Linux
     replay share one identity. Tests freeze the canonical source, column,
     representative output, transformation, pickle, causality, isolation, threshold,
     remainder, eligibility, memory, publication, resume, and immutability contracts.
     The retained `edge_rebuild_intraday_bar_only_causal_20260814_v1` authority remains
     unchanged historical evidence under transformation `0da898cc...`; it is rejected
     by the current loader and cannot train or promote. Its audit, two event-preflight
     authorities, and eight retained development/rejection model bundles are likewise
     historical. Deterministic rematerialization completed at
     `intraday_causal_volume_bar_dataset_20260831_v2` with request `5e8c508a...303a` and
     the current transformation `6fdfd0c8...cb51`. Audit-remediation commit `1829fce`
     adds projection-lineage, raw-source-cutoff, and five-minute-prefix poison gates.
     Future CLI publications require separate hash-bound per-invocation execution
     evidence with aggregate process/worker memory validation and crash recovery. The
     present artifact's execution assessment remains incomplete because that telemetry
     was introduced after its resumable invocations. Verification passed 25 focused
     remediation tests and the complete suite with 1,714 passed and three skipped;
     affected Ruff, strict mypy, compileall, diff, CLI-help, and transformation checks
     passed. Independent code and ML-design reviewers reported no remaining P0, P1, or
     P2 finding. This registers a current data authority only: intraday training,
     promotion, serving, and locked-test access remain unauthorized.
   - **Canonical intraday ledger ownership (`completed`).** Supplemental commit
     `a4002ce` completes an interrupted test refactor by making
     `intraday/evaluation/ledger.py` the sole owner of position-ledger construction,
     position closing, and ledger metrics. The seven functions removed from
     `economics.py` were AST-identical duplicates; `economics.py` now owns only
     ranking diagnostics. Production gates and training coordination import the
     canonical ledger directly, and a runtime identity test prevents tests from
     exercising an unused implementation. Verification passed 173 affected tests,
     Ruff, strict mypy, compileall, and a one-owner scan. The code reviewer approved
     the final diff with no remaining P0, P1, or P2 finding. The complete isolated
     suite after both implementation commits passed 1,691 tests with three skipped in
     13 minutes 30 seconds.
   - **Intraday selected-session planning ownership (`completed`).** Implementation
     commit `d34ea25` moves the complete selected stock-session verification and
     one-minute/five-minute acquisition-plan publisher byte-for-byte from
     `edge_rebuild/selected_session_history.py` to
     `intraday/datasets/selected_session_history.py`. All five production consumers
     and both direct test consumers import the canonical owner; no compatibility alias
     or old file remains. The source Git object remains
     `2f43aa9ffd48a8a4f76f7b7bb448c204ccf7f2bd`. Tests freeze the new function and
     `SelectedSession` owners, pickle round trip, exchange-calendar and early-close
     behavior, selection lineage, one-minute alignment, and all old import forms.
     Persisted schemas, policy hashes, plan fingerprints, unit identities, and existing
     authorities are unchanged. The current volume-bar transformation remains
     `6fdfd0c8...cb51`, and its 794-session authority replays without regeneration.
     Verification passed 174 focused tests and the complete suite with 1,719 passed and
     three skipped. Affected Ruff, strict mypy, compileall, CLI help, import/old-path,
     source-parity, diff, process, and temporary-output checks passed. Repository-wide
     static results remain the recorded Step 6 baseline of 168 Ruff and 14 strict mypy
     findings in untouched files. Independent code and ML-design reviewers reported no
     remaining P0, P1, or P2 finding.
   - **Intraday canonical history materialization ownership (`completed`).**
     Implementation commit `7e96bc4` moves the complete two-pass bar shuffle,
     exchange-session segmentation, selected-session eligibility, overlapping-source
     resolution, and ticker-defect quarantine byte-for-byte from
     `edge_rebuild/history_materialization.py` to
     `intraday/datasets/history_materialization.py`. The command adapter and both test
     consumers import the canonical owner; no compatibility alias or old file remains.
     The source Git object remains `f5acc0e7de04dc00f0213a54beec1b8a5531f74a`.
     Tests freeze `SessionBounds` owner and pickle identity, all five public function
     owners, every old import form, early-close/session-segment behavior, source
     overlap, ticker quarantine, and selected-session eligibility. Persisted schemas
     remain `edge_rebuild.intraday_materialization.v1` and
     `edge_rebuild.intraday_materialization_authority.v1`; no authority regeneration
     occurred. The current volume-bar transformation remains `6fdfd0c8...cb51`, and
     its 794-session authority replays unchanged. Verification passed 200 focused tests
     and the complete suite with 1,724 passed and three skipped. Affected Ruff, strict
     mypy, compileall, CLI help, import/old-path, source-parity, diff, process, and
     temporary-output checks passed. Repository-wide static results remain the Step 6
     baseline of 168 Ruff and 14 strict mypy findings in untouched files. Independent
     code and ML-design reviews closed with no remaining P0, P1, or P2 finding.
   - **Intraday Alpaca/SIP history collection ownership (`completed`).**
     Implementation commit `01275d4` moves the complete bounded intraday collector
     byte-for-byte from `edge_rebuild/history_collection.py` to
     `intraday/datasets/history_collection.py`. All six production consumers and four
     direct test consumers import the canonical owner; no compatibility alias or old
     file remains. The source Git object remains
     `ce3c6f3132b1bf10df1f50b2a24a28aee0c6b2b4`, and tests freeze all three public
     function owners plus every old import form. All six persisted collection, unit,
     and authority schema strings remain unchanged. Retained stock collection evidence
     replays 2,116 units and 16,636,841 rows at authority `63d9d714...fd5c`; retained
     benchmark evidence replays 794 units and 4,005,350 rows at authority
     `889dcc46...dde6`. Their manifest hashes are unchanged. The derived 794-session
     volume-bar authority still replays 3,095,688 rows, including 1,365,015 eligible
     rows, under request `5e8c508a...303a` and transformation `6fdfd0c8...cb51`.
     Verification passed 242 focused tests and the complete suite with 1,729 passed
     and three skipped. Affected Ruff, strict mypy, compileall, collection CLI help,
     import/old-path, source-parity, real-authority replay, diff, process, and temporary
     output checks passed. Repository-wide static results remain the Step 6 baseline
     of 168 Ruff and 14 strict mypy findings in untouched files. Independent task,
     code, and ML/data-design reviews closed with no remaining P0, P1, or P2 finding.
   - **Selected-session one-minute coverage ownership (`completed`).**
     Implementation commit `268c62d` moves the complete one-minute coverage and
     canonical five-minute verification authority byte-for-byte from
     `edge_rebuild/one_minute_coverage.py` to
     `intraday/datasets/one_minute_coverage.py`. All five production consumers and two
     direct test consumers use the canonical owner; no compatibility alias or old file
     remains. The source Git object remains
     `c770f369f8f103d0895c4fb9d8363f29b0006ead`, and tests freeze all three public
     function owners plus every old import form. Persisted schemas remain
     `edge_rebuild.selected_session_one_minute_coverage.v2` and
     `edge_rebuild.selected_session_one_minute_coverage_authority.v2`. The retained
     authority replays ready at 43,226 stock-sessions across 502 securities, with 13
     incomplete sessions retained as metadata, zero excluded securities, a 95%
     continuity floor, and the unchanged 5% whole-security exclusion ceiling. Its
     authority remains `e18adef7...a75b` and manifest `d21c1733...c560`. The downstream
     794-session volume-bar authority still replays 3,095,688 rows, including 1,365,015
     eligible rows, under request `5e8c508a...303a` and transformation
     `6fdfd0c8...cb51`. Verification passed 249 focused tests and the complete suite
     with 1,734 passed and three skipped. Affected Ruff, strict mypy, compileall, CLI
     help, import/old-path, source-parity, retained-authority replay, diff, process, and
     temporary-output checks passed. Correcting import order in two touched files
     reduced repository-wide Ruff debt from 168 to 166; strict mypy debt remains 14
     findings in the same three untouched files. Independent task, code, and ML/data
     reviews closed with no remaining P0, P1, or P2 finding.
   - **Selected-session benchmark acquisition planning ownership (`completed`).**
     Implementation commit `9a59d6f` moves the benchmark acquisition-plan builder
     byte-for-byte from `edge_rebuild/benchmark_history.py` to
     `intraday/datasets/benchmark_history.py`. The command adapter and direct tests use
     the canonical owner; no compatibility alias or old file remains. The source Git
     object remains `1c106774cadf7fdf6c514406f72590cf0d782e62`, and tests freeze the
     function owner plus every old import form. The policy still requires SPY, QQQ,
     and all eleven sector ETFs, exact XNYS regular-session bounds, 390 normal-session
     and 210 early-close one-minute bars, SIP, `adjustment=all`, and a 4 GiB process
     limit with 0.75 GiB headroom. The retained 794-session plan replays unchanged at
     manifest `b855b250...e72`, authority `ffd0783d...0a0`, and fingerprint
     `af77d941...6616`; it contains 13 benchmarks and eight early closes. No provider
     request, artifact regeneration, training, promotion, serving, or locked-test
     access occurred. Verification passed 228 focused tests and the complete suite
     with 1,739 passed and three skipped. Compileall, CLI help, canonical/old-path
     imports, source parity, retained-plan replay, diff, process, and temporary-output
     checks passed. Repository-wide Step 6 debt remains 166 Ruff findings and 14
     strict-mypy findings in the same three untouched files. Independent task, code,
     and ML/data reviews closed with no remaining P0, P1, or P2 finding.
   - **Broad intraday five-minute acquisition planning ownership (`completed`).**
     Implementation commit `7affd05` moves the two broad-history plan functions
     byte-for-byte from `edge_rebuild/broad_intraday_history.py` to
     `intraday/datasets/broad_intraday_history.py`. The command adapter and direct
     tests use the canonical owner; no compatibility alias or old file remains. The
     source Git object stays `a014735fd60d6ba04d764804854649c752896cf8`, and tests
     freeze both function owners plus every old import form. Alpaca SIP five-minute
     bars with `adjustment=all`, exact XNYS regular-session bounds, existing-corpus
     subtraction, truncated-session replanning, explicit fund exclusion, and the
     current-snapshot limitation of the broad Finviz proxy are unchanged. The latest
     retained research plan strictly replays 814 sessions, 448,151 missing
     symbol-sessions across 579 symbols, 9,428 bounded acquisition units, and
     34,793,778 maximum expected rows at fingerprint `cc7ead87...3b8e`, manifest
     `ae454a08...5d88a`, and authority `f372997e...9e6f`. The earlier logically
     equivalent plan also replays and remains untouched. No provider request,
     artifact regeneration, training, promotion, serving, or locked-test access
     occurred. Verification passed 213 focused tests and the complete suite with
     1,744 passed and three skipped. Touched Ruff and strict mypy, compileall, CLI
     help, canonical/old-path imports, source parity, both retained-plan replays,
     diff, process, and temporary-output checks passed. Repository-wide Step 6 debt
     remains 166 Ruff findings and 14 strict-mypy findings in the same three untouched
     files. Independent task, code, and ML/data reviews closed with no remaining P0,
     P1, or P2 finding.
   - **Extended-session context acquisition planning ownership (`completed`).**
     Implementation commit `1702991` moves pre/post-market planning from
     `edge_rebuild/extended_session_context.py` to
     `intraday/datasets/extended_session_context.py` and removes the old owner without
     an alias. It also closes two independent-review findings: ER1B now binds the
     membership Parquet hash, membership-audit hash, and universe snapshot ID to the
     frozen ER1A plan; and an explicit suffix must be a non-empty canonical XNYS
     session rather than a weekend/holiday that the calendar could round forward.
     Tests freeze the function owner, all old import forms, empty-date rejection, DST
     conversion, and normal/early-close bar counts. Alpaca SIP five-minute
     `adjustment=all` acquisition, exact 04:00 ET-to-open and close-to-20:00 ET
     windows, atomic publication, and the hard separation from regular-session VWAP,
     EMA, ATR, and RVOL remain unchanged. Both retained plans and the completed
     1,925,863-row collection strictly replay without regeneration. The suffix plan
     remains 313 sessions and 6,886 units at fingerprint `e005505b...f7999`, manifest
     `8e753572...3415`, and authority `ce0842af...52d9`; collection manifest remains
     `f22147ee...5658`. Verification passed 217 focused tests and the complete suite
     with 1,755 passed and three skipped. Touched Ruff and strict mypy, compileall, CLI
     help, canonical/old-path imports, exact ER1A membership lineage, retained-plan
     and collection replay, diff, process, and temporary-output checks passed.
     Repository-wide Step 6 debt remains 166 Ruff findings and 14 strict-mypy findings
     in the same three untouched files. Independent task, code, and ML/data reviews
     closed with no remaining P0, P1, or P2 finding.
   - **Prospective closed-session SIP evidence ownership (`completed`).**
     Implementation commit `3791541` moves the closed-session collector from
     `edge_rebuild/prospective_sip_session.py` to
     `intraday/datasets/prospective_sip_session.py`; the collection command imports the
     canonical owner directly, every old import form is prohibited, and no alias
     remains. The original source Git object was
     `3ff60848019791e26ada7cb0eaee814d55e37932`. The migration also closes independent
     review findings: completed output is accepted only when the requested session,
     membership authority, policy bytes, effective configs, and benchmark set match;
     fresh configs must reconstruct exactly from their recorded policy files; the
     invocation unit limit is shared across stock and benchmark children; and the
     retained stock membership table must equal the causally active observed cohort.
     A crash after both children complete can publish the parent after the next open
     only when all immutable retrieval timestamps remain inside `[close + 60 seconds,
     next open)`; incomplete children cannot make late Alpaca requests.

     The retained `2026-08-20` authority strictly replays 503 securities, 39,181 SIP
     five-minute stock rows, all 13 required benchmark ETFs, and 5,070 SIP one-minute
     benchmark rows. Twenty-two sparse securities produce a 4.3738% exclusion rate,
     below the unchanged 5% ceiling. Status remains
     `source_complete_warmup_ineligible`; training, selection, and serving eligibility
     are false. No provider request, artifact regeneration, training, promotion,
     serving, or locked-test access occurred. Verification passed 249 focused tests
     and the complete suite with 1,772 passed and three skipped. Touched Ruff and
     strict mypy, compileall, collection CLI help, canonical and old-path imports,
     retained-authority replay, diff, process, and temporary-output checks passed.
     Repository-wide Ruff debt is now 165 findings; strict mypy debt remains 14
     findings in the same three untouched files. Independent task, Python-code, and
     ML/data-design reviews closed with no remaining P0, P1, or P2 finding.
   - **Prospective broker-action evidence ownership (`completed`).** Implementation
     commit `4a98f29` moves Alpaca polling and immutable generation publication from
     `edge_rebuild/prospective_broker_actions.py` to
     `intraday/datasets/prospective_broker_actions.py`; no alias remains and every old
     import form is prohibited. The intraday owner is intentional because collection
     binds the A4.3 security namespace and observed poll cadence. Downstream issuer
     classification remains under catalyst policy.

     Historical retained polls use a hash-only identity verifier and therefore remain
     auditable after the A4.3 transformation changed, but stale bar rows cannot enter
     training, selection, or serving. Fresh collection still requires the full current
     bar authority. Strict replay now covers exact JSON schemas, duplicate/non-finite
     values, numeric types, causal timestamps, derived counts, child eligibility,
     registry claims/commits, Windows reparse ancestry, and immutable artifact paths.
     Generation publication stages under a verified ownership marker and atomically
     renames only after complete replay; unowned staging is never deleted.

     Three retained polls replay 76, 460, and 451 observations; two retained generations
     replay 536 and 451 revisions. They remain explicitly ineligible for training and
     serving. No provider call, artifact regeneration, training, promotion, serving, or
     locked-test access occurred. Verification passed 283 focused tests and the final
     complete suite with 1,810 passed and three skipped. Touched Ruff, strict mypy,
     compileall, CLI help, import guards, retained evidence replay, diff, memory, and
     temporary-output checks passed. Independent Python and ML/data reviewers reported
     no remaining P0, P1, or P2 finding.
   - **Prospective analyst-revision horizon ownership (`completed`).** Implementation
     commit `7419df8` moves classification, episode construction, source coverage,
     capacity audit, publication, and strict replay from
     `edge_rebuild/prospective_analyst_revision_horizon.py` to
     `intraday/datasets/prospective_analyst_revision_horizon.py`. The command uses the
     canonical owner, all old import forms are prohibited, and no alias remains.

     The publisher now handles zero-news horizons, enforces pre-load and post-load
     memory checks plus bounded parent bytes and projected expansion, requires exact
     metadata and frame schemas, verifies cross-generation security identity, and
     recomputes provider-time collisions and event first-seen time across the complete
     horizon. Owned sibling staging and atomic rename prevent partial publication.
     Source-event capacity is kept separate from matched market-session capacity, and
     training and serving eligibility remain false.

     Both retained authorities replay without regeneration. The August 17 horizon has
     536 classified revisions, three episodes, and 1,006 coverage rows; the August 21
     horizon has 451 revisions, 12 episodes, and 503 coverage rows. Verification passed
     273 focused migration/integration tests, the restored 57-test intraday development
     file, and the exact full suite with 1,846 passed and three skipped. Touched Ruff,
     strict mypy, compileall, CLI help, retained replay, diff, and temporary-output
     checks passed. Independent Python and ML/data reviews closed with no remaining P0,
     P1, or P2 finding. Repository-wide debt is 166 Ruff findings and 14 strict-mypy
     findings in three untouched intraday dataset files.
   - **Prediction-data readiness governance (`completed`).** Implementation commit
     `e417960` moves readiness orchestration and contracts to
     `governance/readiness`, extracts immutable authority replay to
     `evidence/readiness_authority.py`, and moves promoted-bundle contracts and
     verification to `governance/promotion`. The research CLI now exposes the
     behavior-named `audit-prediction-data-readiness` command and uses
     `configs/prediction_data_readiness.toml`; no readiness compatibility alias
     remains.

     Current authorities bind exact swing and intraday strategy identities, XNYS
     calendar package/version, costs, folds, dimensions, exclusions, catalyst channel
     policy, availability, counts, and causal event/assignment replay. Historical-v1
     evidence remains strictly replayable but cannot authorize current planning.
     Readiness loads and reduces each large horizon independently under the 4 GiB
     process limit. Pytest discovery is fixed to `tests/`, so local data, model,
     scratch, and cache artifacts cannot change repository test collection.

     Verification passed 354 focused tests with one skip and the exact complete suite
     with 1,933 passed and three skipped in 18 minutes 41 seconds. Touched Ruff and
     strict mypy, compileall, CLI help, retained historical replay, and diff checks
     passed. Independent Python and ML/data reviewers reported no remaining P0, P1,
     or P2 finding. Repository-wide debt is 161 Ruff findings and 14 strict-mypy
     findings in the same three untouched intraday dataset files. No provider call,
     artifact regeneration, training, promotion, serving, or locked-test access
     occurred.
   - **Prediction-serving ownership (`completed`).** Implementation commit
     `fc2a32e` consolidates bundle loading, API-facing prediction contracts,
     prediction orchestration, snapshots, outcome-intent registration, investment
     replay, and swing inference under behavior-named `core`, `serving`, and `swing`
     packages. Every removed top-level or `edge_rebuild` import path is prohibited
     without an alias. Serving now rejects duplicate or non-finite JSON, reparse-point
     paths, manifest races, future model generations, stale drift evidence, policy
     identity mismatches, and incomplete requested model pairs. Swing outcomes use the
     bound ten-session triple-barrier policy; intraday outcomes retain their separate
     minute-horizon calibration and managed-outcome contract.

     Verification passed 442 focused tests with two skipped and the exact complete
     suite with 1,995 passed and three skipped. Changed-file Ruff, strict mypy over 46
     source files, compileall, CLI help, and staged diff checks passed. Independent
     architecture and ML/governance reviewers reported no P0, P1, or P2 finding.
     Repository-wide debt is 130 Ruff findings and 14 strict-mypy findings in
     `intraday/datasets/publisher.py`, `intraday/datasets/bar_dataset.py`, and
     `intraday/datasets/dataset_io.py`. No provider call, data regeneration, training,
     promotion, serving process, or locked-test access occurred.
5. **Governance, serving, and command package migration (`pending`).**
   Move readiness, promotion, drift, and outcomes to `governance`; bundle loading,
   prediction services, and API behavior to `serving`; and retain only thin CLI
   adapters in `commands`. Active modules, files, commands, and tests must use behavior
   names rather than chronological labels such as `v3` or checkpoint labels. Delete
   `v3` and `edge_rebuild` only after every implementation and consumer has migrated
   and a repository scan finds zero imports of either namespace.
   - **Outcome and drift governance ownership (`completed`).** Implementation
     commit `880f2a8` moves prediction selection to `modeling`, horizon-specific
     maturation to `swing/evaluation` and `intraday/evaluation`, and outcome,
     performance, feature-drift, and drift-policy ownership to `governance`.
     Removed top-level and duplicate swing policy modules have no aliases and are
     guarded against reintroduction. Outcome observations and matured economics are
     rebound to immutable intents on every read; swing and intraday maturation require
     exact daily and one-minute bars; execution economics remain separate from label
     economics; feature drift is bound to the promoted model's complete feature-name
     set and reference profile; and serving pins the approved drift-policy hash.
     Independent ML/governance and production-Python reviewers found seven P1 issues;
     all were fixed and both reviewers confirmed closure. Verification passed 339
     focused tests and the exact complete suite with 2,033 passed and three skipped.
     Changed-file Ruff, strict mypy over 30 source files, compileall, diff checks, and
     the final no-Python-process check passed. No provider call, data regeneration,
     training, promotion, serving process, or locked-test access occurred.
6. **Repository-wide static quality (`pending`).**
   Resolve all configured repository-wide Ruff and strict mypy findings, remove only
   reference-proven scratch or placeholder artifacts, and add architecture guards that
   prevent duplicate production namespaces from returning. The measured baseline after
   `1829fce` is 168 Ruff findings and 14 strict mypy findings across three files; none is
   in the files changed by `1829fce`. These remain repository debt, not accepted passes.
7. **Full verification and closure (`pending`).**
   Run focused tests after each task, then repository-wide Ruff, strict mypy, the full
   test suite under the configured writable runtime directory, `git diff --check`, and
   a process/memory check. Update the handoff with measured evidence only.

The paused next structural checkpoint is repository-wide static quality. Resolve the
current configured Ruff and strict-mypy debt without changing features, labels,
policies, datasets, model state, or artifact identities. The 2026-09-07 repository-wide
Ruff scan reports 106 errors; the last verified strict-mypy baseline remains 14 findings
in three intraday dataset files and must be remeasured before editing. Use independent
reviewers, one constrained Python process at a time, and close every reviewer and test
worker after the checkpoint.

The clean model-research checkpoint is implementation commit `880f2a8` plus its
documentation closure. The user's 2026-09-07 request supersedes the four-family next
step: follow the Long-Only Swing Research And Implementation Plan above. Keep causal
source authorities, chronological validation, costs and source integrity; do not
promote merely because AUC reaches 0.60. Intraday and unrelated cleanup remain paused.

Rollback is the last pushed task commit. A task is not accepted until the same senior
reviewer has inspected its bounded diff and all supported P0/P1 findings are fixed.

### Package Dependency Direction

- `core` is domain-neutral and cannot import another project package.
- `sources` and `evidence` may import `core`; immutable evidence does not import a
  provider transport.
- `universe` may import `core`, `evidence`, and `sources`.
- `catalysts` may import `core`, `evidence`, `sources`, and `universe`.
- `modeling` may import `core` and `evidence`; it cannot import a trading horizon.
- `swing` and `intraday` may import the lower layers above but cannot import each other.
- `governance` may import completed horizon contracts and lower layers.
- `serving` may import governance and completed horizon contracts; governance cannot
  import serving.
- `commands` and `research` are outer adapters. Production packages cannot import
  either, and production code cannot import tests.

An AST-based architecture test must enforce the allowed top-level production package
set, required swing/intraday subpackages, the dependency direction above, forbidden
chronology/checkpoint names, zero imports of removed namespaces, and no unexpected
top-level production modules. Import and CLI smoke tests must pass after final deletion.

## Historical Structural Refactor Evidence

- Original `swing_features.py` decoupled into `swing_pipeline_steps.py`, `swing_filters.py`, and `swing_catalyst_features.py`.
- Original `swing_training.py` orchestrated and pruned into `training/data_io.py`, `training/lgbm_models.py`, `training/swing_evaluation.py`, and `training/swing_types.py`.
- Maintained frozen contracts, mathematically exact baseline logic, strict memory budgets, and passing tests.
- Implementation commit `8e9cff6` ensures 100% strict `mypy` and `ruff` compliance with fully updated lineage tests.
