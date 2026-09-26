# Active Edge Rebuild Handoff

Status: active
Last updated: 2026-09-25
Repository: `C:\project\market-predictor`
Branch: `unified-swing-product`
Last completed implementation checkpoint: `dfd1b76` (pushed; SEC retry passes with resume), after `20799b9`
(retirement sub-slice (a), collection package), `beaa4eb` (memory-guard batch keeping) and `da5d4c3`
(per-issuer SEC acceptance clock).
Last completed model-training checkpoint: relationship run on `a8be7cb` (artifact pins below).
Baseline model-training checkpoint: `07963cc` (pushed; unchanged).
Source-collection checkpoint: `19698d6` (pushed).
The combined historical outcome/features delivery is verified and committed.

## Current State

September 25 user decisions: both investment forecast horizons, 63 and 252 exchange
sessions, are approved as separate targets (allocation and risk budgets remain
undecided). TradingFlow's three local commits (`afaafc0`, `5217fad`, `8525f80`) are
pushed to its `unified-swing-product` branch (main untouched). The remaining Market
Predictor intraday retirement runs in parallel with the long SEC download; its heavy
replay waits for the download, and it must close before the issuer-reaction features
and the last two fits so they are built on the final swing-only contracts.

SEC slice progress (in progress): implementation `319360e` is pushed after design and
closure reviews by both reviewers (no blockers; the major closure findings fixed:
availability-based 8-K timing, enforced sealing, crash-safe shard reopen, joint and
co-registrant filings, fuller receipts, cooldown-respecting resume). Targeted checks:
156 affected tests, Ruff and strict mypy on 13 files; a read-only replay reproduced
all 689,467 filings and every response hash of 624 issuers.
Publications (config `configs/swing_sec_form_inventory.json`, SHA256
`26766aba38a513b99075766a6d1d595eaa8a3889281b85763cde94f6e7bc7563`):
`data/research/swing_initial_fit_sec_form_inventory`, manifest
`3370d01aa3077eb82f836659ea46d217dc153bafbf35277424570ddba1dd14bd` (389,875 filing
rows, 387,471 accessions, 543 securities; 22,067 selected 8-K accessions; 3.3 minutes,
peak process 1.9 GB, system 85.0%); sealed `data/research/swing_later_sealed_sec_form_inventory`,
manifest `9001af4ef4e52e853164a9281cd01870aeafcab76c8176bebc7d13404df04540` (unread).
Pilot collection `data/raw/sec_filing_documents_pilot_v1`, manifest
`52f3ea2fbb76d1846e43a7bc2fea16189d882ac52b34b114fbe105466c71f0ca`: 7 detail pages and
14 documents archived, no rejection; its pages are pinned fixtures in `1cf57d6`.
The full initial-fit collection into `data/raw/sec_filing_documents_initial_fit_v1`
started 10:25 UTC and was deliberately stopped at 16:46 UTC after 18,500 index attempts
(37 verified shards pinned by its `_checkpoint.json`; lease released; not to be resumed).
It exposed a data defect: for 93 of 520 issuers observed, the saved EDGAR submissions
`acceptanceDateTime` is New York wall-clock time labelled UTC (all-or-nothing per issuer,
exactly the New York offset; 3,033 of 17,847 detail pages rejected as "acceptance time
differs"), so their canonical acceptance and availability are four to five hours early.
It also measured about 1.2 requests per second, limited by SEC latency.

Both design reviews are consolidated into the plan (September 25). SEC clock (ML review,
no blockers): the old SEC decision authority, SEC-family catalyst events and historical
`catalyst_full` SEC columns also consumed the uncorrected clocks and are recorded as
contaminated historical evidence; neither retained model uses an SEC, news or catalyst
column (0 of 120 and 0 of 124, checked in the unit manifests). The reviewer's
form-group-mix example was refuted by EDGAR's own pages (HollyFrontier's 10-K, 10-Q and
8-K all read New York time); a read-only comparison against EDGAR's weekday 06:00-22:00
hours agrees with the page labels for all 520 issuers and 1,619 informative issuer-form
groups, none contrary (a read-only session diagnostic over the whole archive's clocks;
no content read). Decisions: per-issuer convention verified by first/last pages in every
form group plus in-data agreement; delete the superseded SEC decision authority path; four
collector workers under one five-per-second governor. Retirement (code review, one
blocker): closed evidence also pins `modeling/strategy_contract.py`, the strategy
contract TOML, `label_paths.py` and `canonical/joins.py`, which both model families
re-hash on load. After the user's Windows permission change the 18 baseline unit folders
are readable; both model runs verify completely against their recorded manifests
(18 units and 34 files each, no extra files), so evidence is re-issued for both
families under the approved September 20 policy.

SEC clock correction implementation `da5d4c3` is pushed. Pure logic
`catalysts/sec_filings/acceptance_clock.py`; publisher and page collection
`research/sec_acceptance_clock.py` (commands `collect-sec-clock-pages`,
`publish-sec-acceptance-clock`); shared leased runner `research/sec_page_collection.py`.
The form inventory now requires an `acceptance_clock` config pin, re-reads every saved
submissions row under the issuer's convention (so the 2019-07-08 rows the archive missed
are included, marked `archive_event = false`), excludes and counts unknown-convention
issuers, and reports issuers whose older page for the window's first New York day was
never fetched (`window_start_coverage`). The collector runs four workers under one
five-per-second governor. The first 403/429 sets a stop signal that also ends governor
waits, so nothing is sent afterwards; a failed attempt keeps every other finished
receipt. The SEC decision authority module, command and test are deleted.

Review outcomes. ML review had no blockers. The forward SEC collector
(`sec_incremental.py`) was left as a raw-evidence producer: its label windows chain end to
end and no timing consumer reads its events; its docstring now says so. The SEC family in
the pinned catalyst authority is removed in retirement sub-slice (c), which precedes any
new feature build. Code review found one major issue: queued workers slept through SEC's
cooldown and then sent requests. It is fixed and proved by a real-governor test, which
under the old sleep never returned within its 15-second limit. The minor issues are fixed:
receipts kept on failure, an unreadable page makes its group unknown while a foreign page
fails, sessions are closed. An incomplete older-version collection cannot be resumed
because the request now binds phases and workers; the stopped
`sec_filing_documents_initial_fit_v1` was never to be resumed.

This change also fixes `test_package_dependency_boundaries`, which had been failing
unnoticed since `7a9334c` and `319360e` because it was outside their targeted sets:
catalysts imported `config.Settings`, the swing `pinned_object` and
`data_quality._safe_json`. The last is now inlined with byte-identical JSON; the code
reviewer checked 3,129 real items with 0 mismatches.

Targeted checks after the final edit: 521 tests (SEC inventory, clock, documents,
collection, incremental, both content inventories, command, CLI, package and
architecture boundaries), plus 593 earlier across the `edge_rebuild` command consumers
(only the pre-existing boundary failure). Ruff and strict mypy pass on 12 changed source
files. The full suite was not run (component checkpoint).

SEC clock page runs (archive authority SHA256
`886d727670582409a8f816ac5b5283ebd6edc0c6e06d143f26c8f691ff2bf661`, 4,957 detail pages):

- September 25 16:18 UTC: the memory guard stopped the first attempt at 90% system memory
  before any page was saved (request hash `3bf06857...`, empty checkpoint; the empty
  directory was removed). The guard then discarded an unsaved batch; `beaa4eb` now saves
  the batch and in-flight attempts before re-raising, and SEC jobs name themselves in the
  guard message.
- 20:14-21:42 UTC: `data/raw/sec_acceptance_clock_pages` completed with manifest
  `3f2c71e4298fa764d97ea9b963a7e4c398345c16de3e225b9feeb032cbb877a5`: 4,690 pages archived,
  267 exhausted after three passes. 1,147 of 1,243 retryable attempts were HTTP 503 bodies
  reading "SEC.gov is undergoing maintenance", and 96 were connection failures. First
  attempts failed 12.1%; the back-to-back later passes failed 63-71%. The gaps would have
  left 12 of 624 issuers unknown (4,850 archive filings). It stays as evidence of the
  maintenance pattern and is not the clock's input.
- `dfd1b76` spaces later passes by one and then five minutes. Units still without a final
  outcome leave a run `incomplete`, and a resume with the printed checkpoint gives them
  fresh passes, so a completed collection has a final outcome for every unit.
- Running now: `collect-sec-clock-pages` into
  `data/raw/sec_acceptance_clock_pages_spaced_retries` (log
  `data/runtime/sec_acceptance_clock_pages_spaced_retries.log`).

Next: publish the clock, write the corrected inventory config, republish both
inventories into new outputs, then run the corrected document collection into
`data/raw/sec_filing_documents_initial_fit_corrected_clock`.

Retirement sub-slice (a) `20799b9` is pushed, after both design reviews (no blockers;
consolidated in the plan).
- **What moved.** `market_predictor/collection/` now owns the retained collectors:
  - `alpaca_bars/contracts.py`, `plan.py` and `transport.py`;
  - `prospective_sip_session.py` and `prospective_broker_actions.py`;
  - `retained_inputs.py`, which lists the configs and lineage data that (d) and (e)
    must keep.
- **Identity preserved.** Config field names are unchanged, and the recorded policy
  hashes reproduce (test plus a fixture of the last session's request). Broker actions
  read only the A4.3 metadata chain.
- **What stays in intraday.** Intraday research modules keep their plan layers through
  an explicit `accepted_plan_schemas` argument. `bar_dataset.py` stays byte-identical,
  because its source is part of the recorded A4.3 transformation identity. Rewriting its
  import had broken a frozen-identity test, so the rewrite was reverted and an explicit
  re-export was added.
- **Checks.** Ruff and strict mypy on 27 changed source files. 799 affected tests ran
  (collection, intraday, CLI, package and architecture boundaries, and the
  edge_rebuild-importing swing tests). Their only failure was that frozen identity, now
  fixed; 301 tests then passed on recheck, and the SEC and command tests passed 149.
- **Real-data continuity, read-only:**
  - the three completed polls that load at HEAD also load through the moved code,
    re-deriving their six namespace and bar-lineage hashes;
  - `session_20260820_v1` loads;
  - the four pinned intraday files match the relationship receipt.
- **Pre-existing defect, unrelated to the move.** `poll_20260816T070948Z` fails strict
  replay ("prospective identity audit does not replay") at HEAD too; recorded, not
  investigated in this slice. `poll_20260815T163000Z` and `poll_20260816T070535Z` are
  unfinished attempts with no authority.

September 24 slice `4844b3f` is pushed and both of its real runs are complete: legacy
news-query identity proofs, which the user chose on September 23 to finish before
content qualification. Command `prove-legacy-query-identities`; leased immutable
publisher `research/legacy_query_identity_proofs.py`; pure builder and translation
`universe/legacy_query_identity.py`; config
`configs/swing_legacy_query_identity_proofs.json` (SHA256
`25852b8a748d95ffa33dc0ba952767ad44b4b59a3b77927a6538363d4fef43c6`). Artifact
`data/research/swing_legacy_query_identity_proofs`, manifest
`84ffb55398f3db016ce9e0af9dea8aa9de54c559f5d9a1959cca2396d92712d3`, log
`data/runtime/swing_legacy_query_identity_proofs.log`, exit 0; it is immutable.
165 legacy IDs are proven in 184 rows over 165 target securities, with no partly
proven spell: 110 `company_ticker_hash_reproduced`, 5 `sp500_spell_events_reproduced`
(OGN, PENN, POOL and post-window AMTM and SOLS), 22 `cik_equal` (BRK-B, BF-B and 20 additions
after May 2024) and 28 weaker `cusip_chain_end_ticker_match`. Five are rejected with
reasons: CUK, FLT and FBHS/FBIN have no candidate; Fiserv `cusip:337738108`
contradicts Alpaca transition `45fd1861` (FISV to FI on 2023-06-07); EchoStar is a
corrected security. All 141 previously unbridged initial-fit query IDs are covered:
137 proven, 4 rejected. The target membership authority carries latest tickers back
to 2018, so no rule attributes by comparing historical tickers with it. The minting
parser `symbol_changes_from_transitions` drops old tickers ending in V as when-issued
symbols, which is how the legacy build missed Fiserv's change; the chain check reads
the pinned transitions directly. Proof availability is the membership effective start
(`retrospective_membership_effective_proxy`); proofs are research evidence only.

Review: the senior ML and senior Python/.NET reviewers reviewed the design; one
blocker was accepted (the frozen ticker-uniqueness rule would have given Trane's IR
news to Ingersoll Rand) and replaced. Closure reviews found no blockers; the major
finding (a CUSIP match on a rejoined target's earlier row) and the minor findings
(closed-row deletion evidence, share-class guard wording, transition-file origin,
evidence-date definition, rejection-reason validation, remainder totals, inventory
clock labelling, missing differential branches and guard test) are fixed. The ML
reviewer applied both passes to the completed inventory: 59,487 of 60,592 unresolved
records become legacy-proven, the 1,105 left all belong to rejected IDs, and no new
attributed-coverage overlap appears. Verification: 204 targeted tests (new builder,
publisher, inventory, command and CLI tests plus the protected bridge tests), Ruff
and strict mypy on the eight changed Python files, and a read-only real-bridge parity
run (bridge 445 rows, SHA256 `d3ab16ed...`): the generic passes equal
`map_news_relations`/`map_news_coverage` on all 5,736 coverage segments and 469,668
saved records (1,405.7 seconds). No full suite, download or fit ran.

Open observation outside this slice, not investigated: the target authority shows
retained `cik:0001699150` (the new Ingersoll Rand) as an S&P member with ticker IR from
2018-05-29, while the legacy transitions show IR belonged to Ingersoll-Rand plc (now
Trane) until 2020-03-02, so that security's 2018-2020 cohort membership and prices may
belong to another company.

Inventory rerun with the proofs (leased, exit 0, 03:57-04:51; completed and immutable,
do not resume or rerun): config
`configs/swing_issuer_content_cohort_inventory_with_legacy_proofs.json` (SHA256
`315ae1afb7cea133e4ce27efa63b3f8b002d6a45a2aa549877b45341780b422f`), artifact
`data/research/swing_initial_fit_issuer_content_inventory_with_legacy_proofs`, manifest
`3e4f905e8e6da4d3c0331fb358437633765211a59e94e2f6bba6d1b7962e9bda`, request
`7510280c768608da2ebef841de46b4ec2aee4754f63df17230123cb437b24e32`, checkpoint
`556d4e11e2ec9f3f3780c19826cd8d6a650264e87192a2c635ed529c00e03acb`, log
`data/runtime/swing_initial_fit_issuer_content_inventory_with_legacy_proofs.log`.
All 469,668 records re-verified. Of the 141 formerly unbridged query IDs, 102 reach
cohort securities, 35 excluded securities (19 inherited, 16 cohort exclusions) and 4
are rejected; 59,487 records became
legacy-proven and the 1,105 still unproven all belong to rejected IDs; translated
included records outside their coverage segment: 0. Cohort securities with a proven
query rose from 441 to 543 of 586; whole-window verified query time from 69.6% to
80.9%. Inside each security's S&P membership in the window (read-only check against
the target authority), unknown query time fell from 14.4% (122,640 security-days) to
0.44% (3,716); 41 of the 43 unreached securities were never members in the initial-fit
window and 2 members remain unreached, below the 5% bar. 443,916 attributed stories:
61.91% provider body field, 38.09% headline-only; the weaker CUSIP-chain basis
contributes 2,632 stories (0.98% of window days) and stays separable by
`attribution_basis`. The superseded inventory
(`data/research/swing_initial_fit_issuer_content_inventory`) and its config stay
immutable historical evidence. Both reviewers are idle and hold no work; no
task-owned Python process remains.

September 23 slice `177f6f3` is pushed and its real run is complete: the initial-fit
cohort news content inventory. Command `inspect-issuer-content-cohort`; publisher
`research/issuer_content_cohort_inventory.py`; config
`configs/swing_issuer_content_cohort_inventory.json` (SHA256
`7a789f107e84da483af2b1d3e853200198866ffc6cb8afa6a77a0f9e1417c9f7`), which pins
`configs/swing_initial_fit_monthly_news.json` (sources verified by the unchanged
`issuer_news_preparation._source`), the source-proven identity manifest and the
approved population audit file. The closed single-chunk reader gained a page count and
`verify_saved_alpaca_empty_chunk` (scoped extension; earlier results unchanged).

Artifact: `data/research/swing_initial_fit_issuer_content_inventory`, manifest
`ab5482a01575a7a203f0e070c7f2dc93c413930f8296d448cf4aca5122b3ed21`, request
`a145341955004edfbd2a3e4a40aea52014a75707981ed9e8ae2a12fcea29b3c9`, checkpoint
`89580031a718959ab07f1d855ca5c2a184f88f86371e8e4c691fe2a8a8bfef12`, log
`data/runtime/swing_initial_fit_issuer_content_inventory.log`. The run held the shared
lease 11:08-11:57, exit 0; it is complete and immutable, do not resume or rerun it.
All 5,734 ledger units and 469,668 article records verified; derivation-observed
included rows equal the derivations' source events exactly (148,784 / 319,787).
69.6% of cohort security-time has a verified query and 30.4% is unknown: 145 of 586
retained securities have no provable query because 141 legacy query IDs
(`sp500-historical`, `cusip`, some `cik:...:ticker`, e.g. BRK-B/BF-B) are not in the
pinned identity bridge; 60,590 records sit under them unattributed. Of 404,467
attributed stories, 62.35% carry a provider body field and 37.65% are headline-only.
Known-empty days mean only that a provider-symbol query returned nothing. Counts are
query-returned stories, not issuer relevance, content qualification or admission.

Review and verification: senior ML and senior Python/.NET reviewers reviewed the
design (two P1s: derived-ledger column rewrite and identity-equal attribution) and the
implementation; all supported findings are closed with verified closure. 241 targeted
tests passed (JUnit `.test-tmp/cohort-inventory-final.xml`), Ruff and strict mypy on nine
files. No full suite, download, fit or publication beyond this artifact ran. Memory
was not measured continuously (90% guard throughout; samples 69-71% afterwards). No
task-owned Python process remains; both reviewers are idle and hold no work.

September 22 component `7a9334c` is pushed: `inspect-saved-issuer-content` verifies
one saved original Alpaca query chunk without downloads. Reader
`catalysts/issuer_events/content_inventory.py`; leased immutable publisher
`research/issuer_content_inventory.py`; CLI adapter `commands/issuer_content_inventory.py`.
It binds the chunk to its one self-hashed `_request.json` work unit (include_content
and publication-proxy policy required), verifies page hashes/envelopes/pagination,
reproduces the producer's acceptance filter, retained revision, first-seen page and
exact `max(published, updated)` availability, and counts discarded raw items by reason.
Rows published in the initial-fit window are `included` or `version_after_cutoff`;
outside rows are counts only. Bounds reuse `issuer_news_preparation.FIRST` and
`initial_fit_issuer_news.LAST_INITIAL_FIT_CUTOFF`; the publisher's 90% guard runs
inside the reader. Categories are provider body, summary or headline-only fields,
never proof of full articles, event meaning or issuer relevance.

Two independent reviewers (senior ML design; senior Python/.NET code) reviewed the
paused implementation in parallel; their deduplicated findings plus two lead-found
defects (unbounded window, default 85% memory guard) were fixed, and both verified
closure. No C# consumer exists (searched TradingFlow). Verification: 197 targeted
tests in 30.49 seconds (JUnit `.test-tmp/content-inventory-final.xml`), changed-file
Ruff and strict mypy. Test fixtures build chunks through the real producer functions.
Read-only probes passed on five real chunks across all three archives; the largest
initial-fit chunk (TSLA, 130 pages, 6,456 rows) took 13-18 seconds, so a cohort pass
over roughly 485k saved rows is estimated at 15-25 minutes, not measured. No
publication, cohort pass, download, full suite or fit ran. Both reviewers are closed;
no task-owned Python process remains. Memory samples ranged 78.7%-87.3%, mostly from
desktop applications; the default 85% guard failure was observed at 87.3%.

September 21 next component `8e24d45` is pushed. Separate implementation/test
workers built `swing/contracts/issuer_reaction.py` and
`swing/features/issuer_reaction.py`; plan, design/ML and code reviewers checked
the bounded scope. The shared measurement selects the first official XNYS open
strictly after event availability, never a later surviving bar. It returns
independently nullable stock-minus-SPY open/close reaction and a lagged-20-session
volume ratio, with exact UTC nanosecond availability and explicit missing reasons.
This is a low-level component, not an admitted profile, collector or model fit.
Event/identity/source qualification remains caller-owned. All three source families
must have observed semantics for live construction; historical ingestion does not
replace the declared research-proxy availability clock.

Integration corrected two implementation deviations before data use: retrospective
ingestion was included in proxy feature clocks, and every lagged volume was required
positive rather than their mean. Tests cover both. The hash wording was made exact;
a NumPy overflow-comparison warning was also removed without changing the threshold.
Final verification: 276 focused reaction/relationship-regression/continuity tests
passed in 33.41 seconds with RuntimeWarnings treated as errors; changed-file Ruff
and strict mypy passed. JUnit: `.test-tmp/issuer-reaction-final.xml`.
No full suite, provider request, archive-wide audit, publication or training ran.
Memory samples were 71.39% and 70.76%, not a continuous measurement. All six spawned
workers/reviewers and task-owned command processes are closed.

Bounded inventory: the early Alpaca sample contains both actual article HTML and
headline-only text. The broad SEC archive contains form metadata/submissions JSON,
not filing/exhibit bodies; sparse retained corporate-action HTML is not cohort-wide
content coverage. Do not infer earnings/guidance meaning from form-only text or the
existing broad headline classifier. Full source/version/content qualification and
per-ticker/year coverage remain outstanding before final feature-column freeze.

Windows access recheck on September 25, after the user took ownership and granted
read access to the baseline folders: every baseline unit is readable. Root manifest
`0a30ef2f8e3b95cee252f8c1d1bb5be3c6b98cddda4e47d68e835ad5c1892551` and relationship
root `7a699020920024ed29b5d8547724f5355fdc2202ca176f11d38f23418bd4666e` match their
recorded hashes; each run's request hash, 18 unit manifests and 34 unit files reproduce,
and no unit folder holds an unlisted file. This check did not reload models, verify
prediction payloads or establish a paired performance comparison.

September 21 implementation `be474b5` is pushed. Two parallel implementation
workers completed source publication and verification/readiness/training integration;
separate plan, design/ML and code reviewers closed supported findings. Four entry
points share the configured lease, executed package initializers are pinned, and
corrected/quarantined sources plus partial restart/tamper paths have tests. Modules
use `return_relationship_`; config is
`configs/swing_return_relationship_publication.json`. The 124-column profile keeps
the two frozen learners, splits, costs and population. Verification: 22 source tests,
104 input/readiness/verification tests (including actual fixture-to-source-replay-to-
124-column loading), 24 CLI/surface/continuity checks, Ruff and strict mypy on 15
changed modules passed. No fits ran at that software checkpoint. Full suites were not repeated.

Concrete reopening: first real materialization, session 57556 / PID 30052, ended
before creating a publication. `return_relationship_sources.py` incorrectly demanded
a direct parent source-list entry for collection metadata already transitively
byte-pinned by the verified panel request. Pre and post collections have different
wire formats and different semantic JSON hash encodings. Fix scope is that metadata
lookup only, preserving exact inherited byte pins, terminal identities, source basis
and conflict rejection. Parent implemented the fix while a worker implemented
faithful fixtures/regressions. All 69 source/regression/full-chain input tests pass
in 267.69 seconds; changed-file Ruff and strict mypy pass. Independent code review
has no remaining findings. Fix `ed3c125` is pushed; worker/reviewer are closed.
JUnit: `.test-tmp/return-relationship-transitive-final.junit.xml`. No original data,
historical hashes or learner settings were changed.

The unrelated SEC collector released the normal workspace lease before that run.
Do not kill unrelated workers or bypass the shared lease. The CLI test that previously
collided with it now uses a test-only temporary runtime. Real run memory samples were
73.97%, 71.70% and 68.13%; source inventory was 8,807 files, about 3.404 GiB, zero
missing. Session 57556 is finished, not resumable. One-group retry session 5377
completed with 1,231 rows, then full
resume session 11712 completed all 586,305 rows / 551 groups / 59 months at
`data/features/swing_return_relationships_initial_fit`, manifest
`1e60e7712cb8a59ebf789f4c19d36af0a480bb7604ee7d65dc22515cfa20dd6b`.
Independent verification session 8567 and diagnostic 24274 both finished with an
exact dtype mismatch: `available_at_momentum_126_sessions_excluding_recent_21`
is persisted `datetime64[ns, UTC]`, but all-null recomputation is `datetime64[s]`.
The builder's `.to_numpy()` assignment loses extension timestamp dtype; a small
independent probe reproduced it. No changed price/return value has been observed,
and full replay did not pass for that artifact. Preserve it unchanged as non-admitted evidence.
Fix `82842e0` preserves `.array` and explicitly constructs UTC nanosecond clocks
(the new regression caught all-NaT unit inference too). Parallel implementation
added all-null serialization/integration tests; independent code review closed.
94 tests passed in 353.95 seconds, JUnit `.test-tmp/relationship-clock-final2.xml`;
Ruff and strict mypy pass. Both agents are closed. Fresh build session 33939 finished
all 586,305 rows / 551 groups / 59 months at
`data/features/swing_return_relationships_initial_fit_utc_clocks`, manifest
`e9ec4b628a59f7925fb66a0800194f1a712d617a819d13927860d0b9ab0ef04c`,
checkpoint `e3da9949c080fea05b240a4e1e10c89f62b81e7b9db8c609d59684c88c5fffea`.
Independent verifier session 87689 passed all 586,305 rows; receipt
`data/reports/swing_return_relationships_initial_fit_utc_verification.json`, hash
`e1067950ace121d13e4b3227d7488540e03e558d205ee07668696801708d1086`.
All original columns/outcomes are exact and additions are source-replayed; original
baseline numerical correctness is still inherited, not newly replayed.
Readiness config `configs/swing_return_relationship_readiness.json`, hash
`28c0d02bab59f425cc022aab3ba156d71aedf1c1dcaa0fc1d05b5a07696eed9d`,
is schema-validated. Audit session 62308 passed; report
`data/reports/swing_return_relationships_initial_fit_readiness.json`, hash
`3d2b05293cad45b1a699c6d912552067ae5668657c4542d71c0c6b17991c73c3`.
Readiness covers 545 securities / 1,231 sessions. Original eligibility (479,709)
and available supervision (378,037) are unchanged; missing features are not a new
row filter. Training config `configs/swing_return_relationship_training.json`, hash
`95eed56075440b4eca7ce1a977bcfb4434928175389d75a877d72c0ca487119f`, differs
from baseline only in feature/profile identity and prerequisite evidence pins.
Schema validation, six profile-policy tests and independent ML config review pass;
reviewer closed. Commit `a8be7cb` is pushed. Training session 21604 completed at
`data/research/swing_relationship_return_models_initial_fit`: 16 inner temporal/
security-transfer fits and two final models, sequentially. Each final model used
314,167 rows, with 124 inputs / 248 encoded columns. Peak working memory was
2.471607 GiB; all sampled system-memory readings stayed below 90%.
Manifest `7a699020920024ed29b5d8547724f5355fdc2202ca176f11d38f23418bd4666e`,
request `37199984228033e6ed56b56ea4195cf4fabc58f9beb8818e6a105bbce6b5a3a0`,
checkpoint `5cda2a3f49270facba3f0e871e937e2b0827572365cc9c4d18fab42a3705fb5c`.
No portfolio evaluation, outer validation, historical test, serving or promotion
was performed. Independent review verified the three root pins, all 18 unit/model
hashes and 16 prediction hashes, frozen config and feature order. No artifact defect
was found. Baseline request metadata confirms matched folds, input identities and
holdout assignment; its result manifests remain OS access-denied even after scoped
read permission and elevated reads. Per-unit baseline identity/metric comparison
is therefore not independently verified. No ACL or artifact was changed.

Unweighted means of four stored fold diagnostics: temporal daily Spearman is
0.000704 linear / 0.006304 boosted; transfer is 0.005503 / -0.013526. Mean MSE is
0.00262038 / 0.00261576 temporal, versus 0.00255703 for zero-excess prediction;
transfer MSE is 0.00292185 / 0.00293655, versus 0.00286511. The models fitted
correctly, but this weak ranking/error evidence does not establish useful edge.
These means are not pooled portfolio statistics or proof of SPY outperformance.
Never repin artifacts, retrain completed fits, or tune this profile from these
results. All task-owned reviewers and training/test/data processes are closed.

Design resolution: the five historical source files' original bytes remain
unlocated, not proven benign or defective. For this accepted immutable-parent
derivative, disposition is `historical_code_not_reexecuted_or_certified`. Original
hashes remain provenance; every current executed dependency and consumed data/
authority is separately checked. Baseline numerical correctness is inherited
from the original accepted evidence, not newly replayed. Other parent columns
stay exact except explicit profile identity; four additions receive fresh numeric
source replay plus clock/missingness verification. This does not authorize old
model reuse or weaken existing replay verifiers. See the active plan's adjudication.

September 20 parallel implementation checkpoint is verified: Market Predictor
`a6c1294` is pushed; TradingFlow `8525f80` is local-only (no unrequested remote
sync). Separate workers implemented C# raw publication import and Python's four
incremental relationship features. Independent plan, design/ML and code reviews
closed supported findings; the user authorized three reviewer roles despite
TradingFlow's older two-reviewer default. Two implementation workers and the
design/code reviewers are closed. That publication/training contract was completed
in the September 21 checkpoint above. Heavy jobs remain sequential under the 90% system
memory limit; observed samples were about 70-72%, not a continuous measurement.

Verification: 79 C# receipt/import/CLI tests; 131 Python feature/integration tests;
14 independent feature regressions; four Python producer/fixture/real C# process/
continuity checks; changed-file Ruff and strict mypy pass. Review fixed baseline
120-column/160-clock compatibility, physical source/availability validation and
CLI rejection handling. The real-process test initially failed because the CLI
dependency manifest omitted Contracts; normal dependency restore/rebuild resolved
it. Sandbox restore failed network/cache access, then the scoped normal-cache
restore succeeded. No package versions changed. The final interoperability check
proves byte parity, producer lock contention, abrupt test-owner death recovery,
idempotent import and corruption refusal on this Windows local disk. Physical
power-loss and cloud/shared-filesystem durability are not established.

Completed command sessions (do not resume): C# final tests `36271`; failed restore
`82525`; successful interop run exited zero with four passes. Interop environment:
`TRADINGFLOW_NEWS_CLI_DLL=C:\project\trading_flow\src\TradingFlow.Cli\bin\Debug\net10.0\TradingFlow.Cli.dll`.
Its focused test is `tests/test_news_collection_consumer_interop.py`; feature JUnit
is `.test-tmp/return-relationships-20260920-worker/focused-reviewed.xml`. Full suites
were deliberately not repeated at this component checkpoint. No provider calls,
model fitting, broker actions, data deletion or deployment ran.

Limits: the C# operation imports explicit raw publications only, not a polling
service or normalized catalog/feed migration. The Python transform subsequently
became a verified research publication/input in the September 21 checkpoint above.
Investment forecast horizons were approved on September 25: 63 and 252 exchange
sessions, as separate targets.

Latest approved policy: localized/module changes use focused tests plus applicable
lint/types/build checks; component checkpoints add affected integration, integrity
and causality checks; release checkpoints run the full suite and applicable model
replay/expensive regressions. Both AGENTS files carry the policy, including a
documentation-only exception. Do not repeat the 4,488-case suite for these policy
edits and do not start a test-count audit or deletion project. Historical full-run
results below remain valid evidence for their original commits only.
C# remains the desk language and Python the ML language. The recommendation to
centralize shared provider ingestion in C# still needs a concrete ownership/cutover
decision; no collector was moved or disabled in this documentation change.
Verification for this policy-only change: two continuity checks and both repository
diff checks passed; a scoped independent policy review found no actionable issue
and is closed. No runtime tests beyond document checks, lint/types, builds, full
suite, model replay or test inventory ran because executable behavior is unchanged.
Policy commits: Market Predictor `bbbfe46` is pushed; TradingFlow `5217fad` is
local-only under its explicit-sync rule. Runtime implementation remains `94aa1c4`.

The latest instruction starts shared implementation for both swing and long-term
investment. Remaining intraday retirement and full retained-model replay are
explicitly deferred; ten historical source-pin mismatches remain reuse blockers,
not blockers on independent raw-news collection. No investment model or execution
admission is claimed. Its finite forecast horizons still need a separate decision.

Shared collection implementation is verified and pushed in `94aa1c4`: new modules are
`evidence/news_collection.py`, `sources/news_collection.py`,
`sources/news_collection_settings.py`, and `commands/news_collection.py` under the
package. Deployment config is `configs/shared_news_collection.toml`; collection CLI,
observed Alpaca transport, receipt adapter, focused tests and current docs changed.
TradingFlow has only a scoped evidence-repository design update, no C# code edits.
That update is local commit `afaafc0`, not pushed, following its no-unrequested-sync
rule. Both repositories remain on `unified-swing-product`; no new merge to main ran.
Existing untracked TradingFlow local settings and runtime reports are untouched.
101 focused Python, 232 CLI/package/continuity and 27 C# receipt tests passed;
repository Ruff and strict mypy (392 sources) pass. Independent review closed after
two transport fixes; both agents are closed. Final full-suite verification passed:
4,488 tests, ten skips, 268 warnings in 2,902.59 seconds (48 minutes 22 seconds).
The background-launched attempt (session 7390, owned root PID 32832) was stopped
after setup errors. Direct full run 16391 then caught a CLI memory-rejection exit
code regression after 2,195 passes; explicit MemoryBudgetError handling is restored
and all 101 focused tests passed again. That run and watchdog 98342 are finished.
Final full pytest session 43747 and watchdog 98527 both exited zero. The watchdog
sampled a 68.51% system-memory peak against the 90% limit. Log:
`data/runtime/shared-news-final-20260920.log`; JUnit:
`.test-tmp/shared-news-final-20260920.xml` (4,498 collected, zero errors/failures).
No live data collection or training started. Do not resume completed processes.

The earlier September 20 instruction requested complete intraday removal from
Market Predictor, beyond the closed public/CLI boundary. Dependency/design review
is complete on `unified-swing-product`; the user approved retention of successful
swing runs and fresh swing-only verification. The first implementation slice is
complete and pushed in `fe0ed86`. It removes
14 unused intraday source files and 12 exclusive test files, preserving five shared
ranking tests. Removal/import guards were added. An artifact-audit/implementation
agent and an independent design/code reviewer completed and are closed.
Saved swing training/readiness/replay currently bind
the mixed strategy contract; prospective news collection uses an intraday dataset
as its security namespace. Resolve these explicitly before deleting packages.
No raw data or original model/evidence bytes have changed. Shared subdaily swing
evidence remains in scope to preserve. Full package/contract retirement is not done.

Approved decision: retain only completed swing runs with persisted, integrity-checked
models. Failed/interrupted/no-candidate outputs do not qualify as reusable models.
Retain reference-bound rejection metadata only until references are consolidated.
Completion is not profitability or promotion. Fresh swing-only verification before
reuse must be separate from original receipts; no silent repinning or forced
retraining when equivalent usable inputs can be established.

KS3 disposition: its run completed, but all 24 candidates were rejected and none
accepted. Its referenced metadata remains unchanged, not current replay evidence.
The four root helper source pins no longer require executable source retention.
Original bytes are recoverable from Git commit
`a080913ce109f6e8e2107f99def2678e1ece03b7`; no snapshots were fabricated or restored.

New read-only verifier: `swing/training/retained_runs.py`, research command
`verify-retained-swing-run`. Real verification passed all 18 units and the three
independently pinned root hashes. Report:
`data/reports/swing_retained_run_integrity.json`, SHA256
`fd9eb78c3217b130b955544e241372b1094048014ac5ed503a5202f0c626137d`.
Of 266 direct source pins, 256 match and ten implementation files differ.
This is historical file integrity only; current replay, numerical equivalence and
causality remain explicitly unverified, and serving/promotion remain false.

Other inventory results: August swing candidates v1/v2/v3 are `no_candidate`;
broker-action research roots completed without development candidates. A later
permission-scoped read resolved the older model access failures: candidate_v3,
candidate_v3_fixed, candidate_v4 and swing_candidate_v12 contain no-candidate
evaluation/card evidence, not model binaries. Both retired intraday catalog entries
also contain no-candidate reports only. The swing technical catalog candidate's
externally pinned manifest and all three files (model, evaluation, model card)
pass hashes and sizes. Preserve that older completed candidate alongside the two
return models; its current-contract replay remains unverified. The swing catalyst
catalog entry has no candidate. Old catalog reuse permission is superseded pending
swing verification. No original artifact directory was deleted in this slice.

Final verification passed: 4,442 tests, ten skipped, 268 warnings in 1,742.89
seconds (session 87759, exit zero); repository Ruff and strict mypy pass, 388 source
files. Logs: `data/runtime/swing-retained-cleanup-20260920.log` and
`.test-tmp/swing-retained-cleanup-20260920.xml`. Consolidated independent review
closed after fixing two supported findings (policy source-pin coverage and obsolete
catalog reuse permission). Aristotle's implementation agent and Euclid's independent
review agent are closed. No Python/dotnet test worker remains. No provider, training,
broker or deployment process ran. System memory samples stayed below 75%.

Reviewed collector migration: reuse independent S&P membership authorities to
publish a universe-owned namespace, replacing the bar-dataset dependency for new
polls. Transport stays in `sources`, receipt/session evidence in `evidence`, stock
identity in `universe/sp500`, and analyst classification in `catalysts/issuer_events`.
No new compatibility alias or silent rewrite of old polling chains is allowed.
Exact ordering, preservation rules and exit tests are in the active plan.

Additional pre-implementation finding: raw current-source hashes differ from
saved pins in 10/26 training source/config entries, 16/67 predictor current
source/config entries, and 4/26 outcome current implementation entries. These
counts exclude historical/archive implementation mappings. The predictor and
outcome current-byte comparison gates will reject those discrepancies; the exact
training-admission impact has not been established. These discrepancies predate the
unused-source cleanup; its edits did not change those retained-run source pins.

Across these requests and the joined dataset, 23 distinct source mismatches were
found. Eighteen are confirmed newline-only; five remain unclassified:
`canonical/store.py`, `evidence/io.py`, `swing/catalyst_lineage.py`,
`swing/datasets/history_archive.py`, and
`swing/features/catalyst_decision_authority.py` (relative to the package root).
For the strategy contract, the original protected snapshot has 500 CRLF among
503 newlines and normalizes exactly to today's LF source. Original digest
`8bfd58784543aa945247be5a4c42f5830a87afbc63ce6ad0da3065dee82d9b61` differs from
current `1027c7f033874efb7ad307e958838fd055f273c3176a570150ba9618bbdd1afc`.
This explains that byte mismatch, not numerical replay equivalence or admission.

Resolve these as part of approved migration: compare exact preserved inputs,
record equivalence separately, and verify/rebuild under the new contract. Never
repin old manifests or normalize historical objects into a pass. New publication
must freeze LF source before hashing and preserve the exact executed source bytes.
This finding supersedes any assumption that all saved current-byte pins still
match merely because Git is clean. No dataset or model loss was established.
The preceding design-only checkpoint passed two continuity tests. This is historical
evidence, not verification of the current implementation changes.

September 20 preservation: Market Predictor `main` and `origin/main` were
fast-forwarded to `18e07d1` (191 commits). TradingFlow's existing source work was
committed and pushed on `main` as `9d50bc1`; local settings, runtime files and data
were excluded. Both repositories now use `unified-swing-product`. This preservation
step alone did not assert fresh verification. TradingFlow cleanup subsequently
completed and was merged/pushed on `main` as `cb747da`: remaining day-trading design
workflows removed, swing tests restored, new reservations and fresh entry dispatch
restricted to swing. Existing broker adoption, reconciliation, exits and protection
remain. Verification: 97 focused C# tests, 1,546 full offline C# tests and four
prototype-state tests passed; independent review has no blocking findings. Live
Alpaca integration was excluded; no provider/broker requests ran. Local C# report:
`.test-tmp/swing-retirement/swing-retirement.trx`. Both review agents are closed;
no task-owned test/runtime process remains. No Market Predictor source, model, raw
evidence or historical pin changed. Its two continuity-document tests passed.

The interrupted CLI-retirement implementation is verified and pushed in `9c32ce1`.
It removes 44 dedicated day-trading commands and nine
adapters, relocates unchanged S&P source handlers to `commands/sp500_sources.py`,
and adds swing-only CLI publication/activation admission in `serving/admission.py`.
Shared price/news transports, pinned domain code, raw/model artifacts and TradingFlow
files are unchanged. Historical release verification remains read-only. A concurrent
source swap may leave a rejected immutable release, never an active retired model.

Prior plan reviewers Pauli and Newton approved the bounded adapter scope; their
sessions were unavailable after restart. Replacement consolidated reviewers
Ramanujan (`01a0bac4-46d6-70f3-ae13-b8912d8476fb`) and Curie
(`01a0bac4-476f-7271-93c3-045eaa20242b`) found no blocking issues and are closed.
Ramanujan checked 336 registration/help cases and unchanged S&P handler ASTs.
Curie's nonblocking test suggestions were added: positive swing-bundle CLI
activation/rollback, wrong trust and tampered attestation rejection. The first
positive bundle test correctly failed on its stale July fixture; its clock is now
fixed in the test only, without changing runtime freshness checks.

Focused verification passed 117 tests before interruption and 24 admission tests
on resumption. Full pytest passed 4,441 tests, ten skipped, 269 warnings in
1,743.58 seconds; session 86175 exited zero. Ruff and strict mypy passed again
(405 sources). No owned Python/test worker remains. Log:
`data/runtime/swing-command-retirement-full-20260919.log`; JUnit:
`.test-tmp/swing-command-retirement-full-20260919.xml`. Memory fell from 87% to
69.5% before the run; subsequent sampled use stayed between 67.9% and 70.9%.
The saved training artifact's root manifest/request/checkpoint hashes and all
26 directly bound code/config pins match the original evidence. No source
collection or model training was started.
Post-verification continuity/CLI/architecture checks passed another 16 tests.
This is a closed command-adapter checkpoint, not full internal domain retirement.

### Previous Completed Unification Checkpoint

The first unification checkpoint is implemented, verified and pushed for Market
Predictor in `1fe9533`: swing-only public admission plus one cross-language raw-news
receipt exchange. Corresponding TradingFlow changes were locally verified and
subsequently preserved with its pre-existing work in `9d50bc1` on September 20.

Public prediction routes now admit only swing `auto`/`10b`; removed intraday/unified
routes are not aliases. Public replay is swing-only. Research catalog has only two
swing experiments. TradingFlow's latest dirty tree already removed predictor mode
selection; this task tightens its response horizon/model checks and updates fixtures.
No research regressor was installed as a promoted model. All saved data, features,
targets, outcome contracts and training inputs remain untouched.

Raw Alpaca news receipt exchange lives in `evidence/news_exchange.py` and
`sources/news_exchange.py`. C# peers are `TradingFlow.Contracts/Evidence/NewsReceipt.cs`
and `TradingFlow.Data/Evidence/Collection/NewsReceiptImporter.cs`. Exact raw body,
original receive time, query and independently pinned manifest identity are retained.
This is not shared collection scheduling, normalized admission, SEC/bar exchange or
cloud deployment. Specification: architecture section Raw News Receipt Exchange.

Two independent reviewers, Kierkegaard (`01a0ae90-c0ec-7fc3-b6f3-2f916481993d`)
and Aquinas (`01a0ae90-c14a-7492-80f9-898be724f0b3`), completed plan and scoped
code/design/ML reviews. Supported findings were fixed and both closed without
remaining findings; both agents are shut down. Later verification adjusted the
wire endpoint to the stable `alpaca.news` identifier to preserve TradingFlow's
central endpoint-resolver rule, and updated its additional ranking test fixture.

Focused Python checks passed 68; expanded C# checks passed 72 with zero build
warnings/errors. TradingFlow's full offline suite passed 1,482 tests serially,
excluding its live-provider `AlpacaCandlePipelineIntegrationTests`, which could not
complete in this environment. Its cancellation test failed under the first parallel
full run but passed isolated and in the serial full suite without code changes.
Ruff and strict mypy passed (412 sources). The first full Python run passed 4,371
tests with ten skips and found one missed inventory reference: intraday models
were still listed as active workbench artifacts in the retention inventory. Only
that inventory document changed; files and hashes remain preserved. All 73 focused
closure tests then passed. The final full run passed **4,372 tests, ten skips and
271 warnings in 1,808.26 seconds** (session 16644, exit zero);
log `data/runtime/unification-verified-full-20260917.log`, JUnit
`.test-tmp/unification-verified-full-20260917.xml`. Ruff and strict mypy passed
again after the full run. No test, Python, dotnet or TradingFlow process remains;
no collection service, saved-data model training or broker runtime was started.
Identical shared fixture SHA256:
`700611f47f133bd728f9e863af7f76c5d20ee24e31e335b06a9ca8f8f936942e`
(both copies verified). These are synthetic tests, not provider observations.

TradingFlow was previously `main` ahead of origin by ten commits, with hundreds of
uncommitted changes. The user's September 20 preservation request authorized the
source checkpoint `9d50bc1`, including the receipt files, predictor client/tests,
project references and evidence architecture. Secrets/runtime files were excluded.
No broker runtime started.

Internal mixed historical training/release/serving domain implementations remain;
CLI retirement is not full package deletion. Shared minute/hourly bars and existing
swing `intraday_return` features must remain. Open-ended investment needs a separately
approved finite forecast horizon and risk budgets.

## Verified Swing Models (Unchanged)

The requested implement/verify/train checkpoint is complete in `07963cc`.
Regularized linear regression and shallow boosted trees predict ten-session net
stock return above SPY using the existing 120 technical inputs. All 16 independent
fold/scope fits and two final research models completed sequentially in session
16163, exit zero. Each final model used 314,167 eligible matured rows over 1,022
observed training sessions. The published input still contains 586,305 decisions,
545 securities and all 1,231 sessions from July 9, 2019-May 28, 2024.
No source data was downloaded, rebuilt, removed or read from later outer splits.

Four full-calendar expanding folds use 503/682/861/1,040 training sessions before
eligibility, ten intervening sessions, and 179/179/179/181 scoring sessions.
Actual label maturity is purged independently. Preprocessing fits only on training
rows; dates receive equal total weight. Separate transfer fits exclude all 101
held-out security identities from both preprocessing and training. Final research
refits use all eligible initial-fit securities; they are not unseen-stock tests.

Completed artifact: `data/research/swing_technical_return_models_initial_fit`.
Root manifest SHA256:
`0a30ef2f8e3b95cee252f8c1d1bb5be3c6b98cddda4e47d68e835ad5c1892551`.
Request SHA256:
`ebfb9de68e24cdb8bece4b5e8581174234f6c22e0521284e5c4a3cfdf2392347`.
Checkpoint SHA256:
`fecd8733f9d812a5060ee098ab11e4e611b57d7043f66d68803e403e123ef598`.
Final model files are under each named learner's `final_refit/model.joblib`.
Every validation fit retains its own `predictions.parquet` and manifest. All 18
unit hashes, 266 source/config/code pins, maturity boundaries and transfer training
identities were independently rechecked after fitting. Serialization/reload
prediction parity passed for each real model. Peak process memory: 2.406 GiB.
Log: `data/runtime/swing_technical_return_models_initial_fit.log`.

The models trained successfully but show weak predictive signal. Aggregating
nonduplicate out-of-fold predictions with equal date weight, temporal rank
correlation is 0.00449 linear / 0.01467 boosted; squared error is 1.77% / 1.99%
worse than predicting zero excess. Transfer correlations are 0.00696 / 0.00029;
errors are 1.28% / 2.14% worse. Temporal known outcomes are 187,432 of 289,802
predictions; transfer known outcomes are 35,548 of 54,964. Unknown outcomes are
retained. These roughly 64.68%-coverage diagnostics are not a complete funded
portfolio, not proof of SPY outperformance and not grounds for promotion.
Exact metrics and remaining profile gaps are in the current feature audit.

Verification: 105 focused/integration tests, 240 CLI/package/dependency tests,
4,329 full-suite tests passed with 10 skips and 271 warnings in 1,790.40 seconds.
Full-suite log: `data/runtime/swing_return_models_full_tests.log`; JUnit:
`.test-tmp/swing-return-models-full.xml`. Ruff and strict mypy passed again after
fitting, including scripts (409 files). No source/config/test edits followed the
full suite or occurred during fitting. No training/test worker remains active.
Documentation closure also passed 242 continuity/CLI/package/architecture tests
in 17.23 seconds; JUnit `.test-tmp/swing-return-doc-closure.xml`. Closing memory
sample: 70.97% physical use, 4.55 GiB available. All owned workers and agents are
closed; the next checkpoint must not resume a nonexistent background job.

Aristotle (`01a0a0eb-4e0f-7b43-af5d-cd1ae52b8703`) implemented the isolated
estimator module and validation tests and is closed. Independent reviewer Leibniz
(`01a0a0eb-4e99-7933-9c75-0eb0fa024011`) closed design and consolidated code review.
Three supported issues were fixed and tested: pinned exact-byte artifact loading,
Python/SciPy/calendar runtime identity, and peak-memory enforcement before success.
No review agent remains active. The full-suite session 89517 exited zero before
the historical run. All three requested stages are finished, not waiting for
another training approval.

Completed run command (record of execution, not an instruction to retrain):

```powershell
.venv\Scripts\python.exe -B -u -m market_predictor.research_cli train-swing-returns --root . --config configs/swing_return_training.json --config-sha256 47155e2d6ef43e9efb6bb54621913e3df5ee0f66797153c30f71d8f69e5593cc --output data/research/swing_technical_return_models_initial_fit
```

The output is complete and immutable. Resume only
with an independently verified `_checkpoint.json` SHA256 supplied through
`--resume-checkpoint-sha256`; never overwrite artifacts or accept unpinned units.
The acceptance matrix records offline-only layers explicitly. No outer-validation,
exposed-test, live-serving or portfolio-profitability claim is authorized. Existing
raw source reception limitations remain; neither model is a news/SEC reaction model.

## Historical Checkpoint Receipts

The following completed audit/source receipts remain input provenance. Their old
test counts are historical, not the latest repository verification result.

Completed checkpoint: fixed-horizon training-input diagnostics. Implementation
`b3b2372` is pushed and the final receipt refresh passed. It binds the publication and
saved-row receipt, verifies model order, decision clocks, return arithmetic and
exact ten-session maturity, and reports separate input/supervision availability.
It does not flip any training/production flags or fit an estimator. Independent
reviewer Beauvoir (`01a0a03a-327a-7eb1-828f-93b88aaab260`) closed design/diff review
after both supported P2 findings were reproduced and fixed; the reviewer is closed.
Consolidated focused verification passed 289 tests. Ruff and strict mypy passed
on 397 sources. The real audit exited zero and published
`data/reports/swing_initial_fit_training_readiness_percentage_only.json`, SHA256
`7a468ed3e0662d8d6390ae575bee700cdd7da4ef8bb7c8ae921aed6b8eead39f`.
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

Completed policy checkpoint: the user explicitly approved a 90% physical-memory ceiling
without an independent 2 GiB free-RAM floor for swing research. Keep the 5 GiB
process budget and unknown-measurement failure. Configure the audit policy without
editing shared measurement/guard files bound into historical replay evidence.
Implementation `7bb805b` is pushed: config, strict request contract, auditor and
focused tests. The configuration SHA256 is
`8d1e2870eac8f7406376eca1af202dbc5a710d344c1e953fce80faa9a58173e4`.
Shared guard and replay dependency files are unchanged. Independent reviewer
Linnaeus (`01a0a089-93ca-75f1-b1fb-34b39bee7495`) closed design/diff review with no
outstanding findings; its suggested report/config assertions are covered.
Focused verification passed 96 tests, followed by all 46 policy/auditor tests
after the last assertion was added. Ruff and strict mypy passed on 397 sources.
The configured audit completed successfully in session 40719; log:
`data/runtime/swing_training_readiness_percentage_only.log`. New receipt:
`data/reports/swing_initial_fit_training_readiness_percentage_only.json`, SHA256
`7a468ed3e0662d8d6390ae575bee700cdd7da4ef8bb7c8ae921aed6b8eead39f`.
All diagnostics equal the prior verified report; only the three expected
config/implementation pins and explicit memory-policy record changed. The report
records 90.0%, a null absolute free-RAM floor, 5 GiB process budget and 0.75 GiB
process headroom. Prior receipts remain historical, not current-policy inputs.
Full regression session 7250 exited zero: 4,224 passed, 10 skipped and 133 warnings
in 1,755.08 seconds. No source changes followed the run. Log:
`data/runtime/swing_percentage_memory_full_tests.log`, JUnit:
`.test-tmp/swing-percentage-memory-full.xml`. No worker or reviewer remains active.
Earlier 85% failures are historical, not the requested policy.

Next-consumer design review completed read-only with Bernoulli
(`01a0a07e-fb40-7031-b2d7-a9166dd328c9`), now closed. Its concrete proposed folds,
learner settings, missingness policy, separate temporal/transfer fits, artifact
contract and risk tests are recorded under "Completed Technical Swing Return
Models" in the active plan. Implementation and fitting are now complete above. The existing
security holdout is an unseeded SHA256 security-ID threshold; keep that assignment
distinct from estimator seed 42. Do not reuse the retained classifier/ranker.

The user's three requested stages are complete:
1. News rebuild: complete, including 59 monthly catalyst authorities.
2. Technical and catalyst feature joins: complete, 59 months / 586,305 rows each.
3. Saved-row audit: passed; final regression run completed with one documentation
   status-marker failure, 4,145 passed and 10 skipped. The documentation-only
   correction passed both continuity tests; no source code changed afterward.

Those source/audit stages did not fit models. The completed return fit above is
separate and does not claim promotion or SPY outperformance. Current scope is
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

Exact next checkpoint: qualify issuer-news and SEC reaction inputs for the final
feature profile, using saved initial-fit evidence before requesting any download.
Single-chunk verification (`7a9334c`), the cohort news inventory (`177f6f3`) and the
legacy identity proofs with the inventory rerun (`4844b3f`) are complete; do not
rebuild them. Content qualification uses the rerun inventory
(`..._with_legacy_proofs`) and must keep the weaker CUSIP-chain basis separable.
Current slice: SEC form inventory and filing-document collection. Its design is
reviewed and consolidated in the plan (September 24; no blockers from either
reviewer). The user decided on September 24 to collect now and leave nothing for
later: documents of cohort 8-Ks carrying item 2.02, 7.01 or 8.01 in the initial fit
(22,067 accessions: EDGAR detail page, primary document and every EX-99 exhibit), and
the same selection after the initial-fit cutoff into a sealed store that is not read
or summarized until qualification rules are frozen on initial-fit evidence. The inventory,
collector and pilot are done (see Current State); remaining: the full initial-fit
collection, the sealed collection and slice closure.
Freeze content qualification with development-only precision and recall review before
joining the completed-session measurement into the last profile.
No final feature selection or fitting belongs to the SEC slice.
The baseline and relationship profiles are already trained (four of six frozen
specifications); never rerun or tune them. Freeze source fields, event availability,
issuer matching, completed reaction windows, exact columns and missingness before
coding the third profile. Preserve matched decisions/labels, split dates, weights,
costs and the approved exclusions. Initial fit remains July 2019-May 2024, not the
outer validation/test publications. Assign source/attribution and transformation/
consumer slices to disjoint implementation workers with independent review; heavy
runs remain sequential. Automatic raw collection ownership migration, investment
targets and trading admission remain separate decisions.

Plan review found old readiness evidence has five current implementation-pin
mismatches and is not admitted by the existing training loader. This requires a
new saved-row/source verification and readiness report, not rewriting old hashes.
The old model request's ten mismatches do not prohibit a new independently verified
derivative experiment. Preserve existing replay verifiers unchanged. The saved
combined daily source contains availability/source/ingestion metadata but its
reader drops those columns: fix the bounded adapter, do not redownload everything.
Corrected FI/SATS streams and WTW/ATVI/INFO/SBNY unavailable boundaries must remain
explicit. Missing added features stay null with reasons; no additional exclusions,
shortened warmup, date/weight/cost changes or invented source clocks.

Read `evidence/news_collection.py`, `sources/news_collection.py`,
`sources/news_collection_settings.py` and `commands/news_collection.py`, plus
TradingFlow's `NewsReceiptImporter.cs` and the raw-exchange section of its
`docs/research/evidence-repository-design.md`. Consumer integration must bind real
publication/attempt evidence, never fabricate old EvidenceSourceObservation hashes
or treat receipt time as historical article availability. Review the existing desk
REST/news-stream consumers before any cutover. No runtime migration is complete yet.
Both main branches are preserved remotely and TradingFlow cleanup is closed in
`cb747da`. Leave its untracked local settings/runtime reports alone. Shared news
ownership now precedes deferred retirement; no new provider job has started. Session 86175
exited zero; do not resume it or repeat training. CLI retirement is committed in
`9c32ce1`, including registry, adapter, admission and scoped test/doc changes.
The unused cross-sectional intraday research package and root helpers are removed;
no raw data or original model artifact is deleted. Keep shared minute/hourly transports and swing
`intraday_return`; never silently repin training evidence after moving source code.
Do not resume training merely because an HTTP or CLI boundary is complete.

Retained research follow-up: qualify and implement issuer/SEC reaction inputs for
the two remaining bounded return specifications. The relationship profile is complete.

1. Existing technical training is complete: preserve its immutable artifacts as
   the two baseline specifications. Do not retrain them or tune settings after
   inspecting these results. Keep the approved six-specification experiment cap.
2. Follow the original active plan's feature design and acceptance matrix. Map
   genuinely distinct price/volume/regime relationships to current owners; audit
   issuer/SEC causal event and reaction availability. Freeze exact columns and
   source semantics before constructing matched histories. Reuse saved evidence;
   aggregate catalyst_full is not the completed reaction profile.
3. Reuse the verified return estimator/validation/artifact modules for the two
   remaining profile/learner comparisons once their acceptance gates pass. Keep
   chronological masks, dates, holdout assignment, costs and date weights matched.
   No new exclusions or reaction-source claims can be inferred from this fit.
4. Funded evaluation of the preregistered long-only policies remains unrun. Unknown
   selected outcomes must not disappear from that evaluation. The raw archives
   cover later dates, but outer-validation/test publications are still separate
   work; the exposed historical test must never be called untouched.

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
| Approved population | data/reports/swing_research_cohort/approved_research_population_audit.json | c2a7e89b6c360f7ddd704dcd41d4fd8fd6b1b698f1bf7caab9d0de72042d410f (file); internal `audit_sha256` 41de559dd1c415dab60771e10fd489150853a6c2fff07e387d1024b33961efe0 |
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
Two technical return models are fitted; neither is promoted or proven profitable.
