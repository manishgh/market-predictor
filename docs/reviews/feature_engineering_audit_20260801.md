# Current Feature Engineering Audit

### Configurable data storage and retained scratch cleanup — November 13

Status: `implementation_complete; data_movement_deferred_by_user`.
Latest user instruction: proceed with implementation and do data movement later.
The configurable-storage code is implemented, tested, independently reviewed and
pushed in d3211ee. Do not resume copying, staging, retirement, or collector cutover
until the user asks to resume data movement. Keep the current C: data references,
code, Python and enabled collection task unchanged. No migration worker is running;
all seven C: data namespaces remain present, no staging receipt exists, and the
partial D: copy is retained. Actual D: consumer checks and deletion of C: copies
remain unperformed migration gates, not completed implementation evidence.
The 715 unresolved holding outcomes remain pending; no replay or fitting is started
by this storage-only implementation checkpoint. Existing component verification
below remains applicable: this latest change is documentation-only.

Original scope: the user requests moving large datasets/archives to D:,
keeping active code and Python on C:, preserving existing data references and the
pending holding replay, and making storage location an explicit parameter.
The user also asks to remove unnecessary unverified archives. Verification status
alone is not a deletion criterion: preserve raw inputs, published artifacts,
models and all referenced evidence; remove only proven superseded scratch.

Read-only inventory found 107,964,413,214 readable data bytes: features about
40.19 GB, research 45.23 GB, raw 18.78 GB. Four model directories deny listing;
they are preserved and excluded from any automatic migration/deletion decision.
D: is a healthy Samsung T7, exFAT, with approximately 1.95 TB free. Ordinary
C-to-D junctions fail the frozen replay's resolved-path containment and exact
C SourcePin/source-inventory comparisons; changing --root alone also fails.
Two Sol reviews independently confirmed this against the real parent and code.
The current 72eedcf export and all historical source/request/result bytes remain
immutable. The guarded copy to D:/market-predictor/data remains incomplete. The
latest sustained resume began below 80% RAM at 16:59 UTC, then stopped during
partial-copy checks at 90.09%; a later attempt stopped at 90.29%. The check at
17:12 UTC measured 92.4%, with no project Python or Robocopy worker running.
The user has been asked to pause unused build tools; observed dotnet/compiler
process working sets were about 5.5 GB in total. Do not stop unrelated workers. All C: originals remain
present; the partial D: copy is retained for --resume. There is no completed copy
receipt or staging receipt, and no C dataset originals have been retired. The
existing collector task remains enabled with its original C: action. No holding
replay was resumed. Only after the user reauthorizes data movement, and below the agreed 90% system RAM ceiling:
from the chat workspace, run C:/project/market-predictor/.venv/Scripts/python.exe
-B outputs/run_market_dataset_copy.py --resume. Then complete the migration gates
below before deleting verified C: copies or changing the collector action.

Operational helper performance review: per-entry PowerShell provider lookups in
the copy/resume and retirement checks were replaced with equivalent .NET path
and metadata calls. Both Sol implementation and independent review passed tiny
fixtures, including actual junctions and missing children below junctions; all
ancestry, file type, size, hash, volume and deletion-authority checks remain.
Copy helper SHA256: 35dc29654f45b30440830d94467ae68e6d18091bef8f978f63907ce81c1290b0.
Retirement helper SHA256: eb391aa3c10f4170ac3f4aba999414dd8056da0c9ad1d20d235acd6ec545d91d.
No live-data retirement was performed. The reduced check overhead has not yet
been measured on a completed real-data resume; memory pressure still stopped it.

Implementation d3211eef6fa9625610d6d2fb3e044de51083d71c is pushed to main.
The immutable export contains 480 source files; _export.json SHA256 is
fc72cdba5f53d9d0a9a4c9eafc754f679e219fa1b1e04b6612793b2932b90377.
Explicit --data-root now covers the remaining holding replay, investment dataset
builder, investment model trainer, ranking trainer, and incremental Alpaca/SEC
collectors. The PowerShell collection wrapper/installer accept -DataRoot. Windows
immutable publication uses same-directory no-replace rename, supported by exFAT.
No model fitting, acquisition, or holding outcome compilation was run.

Checks: the source-reader final targeted rerun passed 85 cases after fixes; all
99 collector/wrapper cases passed across initial and targeted reruns; the final
investment consumer/source-validation suite passed 92 cases. Core path tests had
31 passes and one symlink-permission skip. Ruff and changed-source strict mypy
passed (30 source modules, then 17 collector/consumer modules). These are synthetic
component tests, not evidence of improved model performance. Senior Sol reviews
covered path identity, contained child reads, shared leases, immutable publication,
retention and the guarded migration procedure.

The independent retained-C preflight passed: 21,930 inventory files, 22,412 distinct
file hashes including exports/lineage, all 33 frozen currency additions, 3,177 known
and 715 unresolved of 3,892 holdings, five exclusions, 10,469 ordered placements,
464 cohorts, 16 views and 20 bp stock cost. Receipt:
chat outputs/storage-source-preflight-c-20261113.json, SHA256
cdac785950f6d4f238b92cd13711501ce640b428226fc482daa309a3a448e5f1.
Independent helper SHA256:
383bb0b6062e9752e9ccafcad110530edf61f0f8d16b2437e773599f06b0732d.
C/D expected pin inventory SHA256:
b724cce146d57d6db980e534e8fa9a5cbcde4e91021c2066f49bfa9215edb7a3.
C/D expected cohort economics SHA256:
567bab8b5d029e9499cf052f1503e8d21dbc8a41510b9ab1051d80fbc1d50987.

Historical source plans additionally require the exact old producer bytes in
 data/evidence/storage_source_implementation_20261113/_manifest.json, SHA256
 d1415f8f38b5a4b892150a3cf749f0a642533ae6883f2208aa6a943f1c3eb409.
The three original source blobs match both their retained plan hashes and commit
8005744871745b2415ddd28e96990f0ae2c8ca80; old plans were not rewritten.

Remaining migration gates: finish per-file copy verification; temporarily stage
original C dataset paths inside data/.storage_migration_20261113 while retaining
all bytes and C implementation_snapshots; validate D with original paths absent;
run --validate-inputs-only through the new immutable producer and the independent
D preflight; compare source and economic hashes; then retire only verified staged
C files. This does not resume the paused holding replay. The existing midnight
collector is still configured for C. Quiesce it during cutover after a pinned XML
backup, then update only its action to include -DataRoot D:/market-predictor/data
and restore its enabled state; preserve schedule/settings/principal. Roll back
renames by the pinned move plan if any required storage check fails.

Completed retention cleanup: removed only the completed population export's stale
_checkpoint.json and .resume SQLite database, freeing 7,395,248,626 logical bytes.
All 1,754 published files remain; published population.sqlite and _manifest.json
were SHA256-verified before and after removal. Receipt: chat
outputs/population-resume-cleanup-20261113.json, SHA256
24585683d68e7588a69ae42d5f6f0e3d625e41f25afd6d85ccaf786a4843246b.
C: free space after cleanup was 16,305,733,632 bytes. This is scratch cleanup only;
no raw, news, financial, training, model or archived published evidence was removed.

Freeze: add an explicit data-root parameter, keeping project/code/config root and
Python on C. One shared EvidencePaths abstraction separates logical evidence
identities from physical file access. Preserve historical pin strings, hashes,
known outcomes, selections, exclusions, costs, clocks and existing interpretation.
Reject unsafe paths, unexpected links, alternate-root fallback and changed source
bytes. Keep the shared heavy-job lease and immutable producer exports on C.
Publication must stage and rename on the selected output volume. Do not monkeypatch
Path/filesystem APIs or rewrite historical manifests to hide relocation.

Order: (1) verify/remove only obsolete completed-export scratch; (2) implement
and test the path abstraction; (3) integrate all readers/writers/hash checks on
the full holding replay call graph; (4) create a new immutable producer and
independent audit; (5) copy and verify retained data before cutover/removing C
copies. Other consumers must pass their own storage checks before their datasets
move. No partial helper implementation constitutes completed migration.

Exit gates: default-path regression and alternate-root parity; stable logical
source inventories/sidecars/proofs; missing/tampered/foreign/traversal/reparse and
no-fallback tests; shared-lease contention; same-volume atomic publication; actual
retained-data source/lineage verification and independent replay comparison; full
per-file copy receipts and preserved C code/environment. Use component checks,
Ruff and strict mypy for changed packages. The existing unrelated mypy diagnostic
remains documented. Revert new code or retain original C files if any gate fails.
The initial core implementation is owned by the Sol storage engineer; root owns
integration/migration orchestration and docs; senior Sol Quant reviews the contract
and produced evidence. V1/beta, refits and financial semantics remain out of scope.

### Assistant scratch cleanup — November 13

Completed the user's request to remove unnecessary assistant-generated files.
Removed 2,037 files across 96 targets: 12,611,427 logical bytes of disposable
Python/tool caches, synthetic pytest scratch and one downloaded installer wheel.
No research manifest referenced these targets; paths, links, age and tracked files
were checked before removal. All 16 protected historical-source/replay-input
hashes matched afterward. No datasets, candles, news, models, research results,
source evidence, active tools, chat history or replay helpers were removed.
Receipt: chat outputs/assistant-scratch-cleanup-20261113.json, SHA256
9c8ea46e9b37a9095f7790fc75560be8acadbc050e2157f1f7aefec2de426aa9.
The holding replay remains paused by the user; no ML job was started.

### Disposable disk cleanup — November 12

Completed the user's disk-only housekeeping request. Removed 128 verified generated
cache/test directories (82,964 files; 1,599,703,221 logical bytes), plus 1,695
dangling uv cache files (32.9 MiB) and 308 pip download-cache files (31.4 MB).
The test directories contain synthetic fixtures, including shared publication-test
fixtures explicitly created by the repository tests. No real training datasets,
candles, news, features, financial inputs, models, research results, partial work,
historical producer sources, installed environment or referenced logs were removed.
Targets were checked for path containment, links, age, tracked files and source
manifest references before native removal. Seventeen protected artifact hashes,
including all eleven external historical producer source pins, passed afterward.
No executable source changed; Git was clean and Python 3.14.5 still starts.
The last measured C: free space was 21,293,142,016 bytes (21.3 GB).

The detailed deletion receipts and verification are retained under the chat
workspace outputs directory. Verification: disk-cleanup-verification-20261112.json,
SHA256 4ce82b03391061680f9b657b7857f0cfa1937d21cf9ccfe80b194693a5d74676. Its receipt hashes bind every removed target.
This housekeeping did not itself publish outcomes or resume the replay.
The subsequent authorized recovery status is recorded below.

### Retained dividend currency and held-stock actions — November 11

Dividend resolution completed November 12 at 10:19:16.793600 UTC after resuming
at 09:42:41.715619 UTC. All 33 requested dividend records are supported, all
4,647 accepted assertion objects are preserved as the identical ordered prefix,
and the cumulative result contains 4,680 assertions. There are zero unresolved
requested events or pending amount corrections; both existing amount corrections
retain their exact bytes. The request builder verified these source bindings.

Resolver: data/research/investment_currency_action_outcomes_20261111/dividend_resolution_72eedcf/_manifest.json
SHA256 12bc2700aa41645fb89e5e14b607fde75b1cc46f11e6c0135ca698baf972a267.
Currency assertions SHA256 72164e48dc18c71ef94cf3980a669fcf7e75948704882f3313ccca8e187b30ab.
Completed log: outputs/currency-action-dividend-resolution-20261112.log,
SHA256 baab4f52fe25fdb8d6e8fbb2fb9705b992e5007cbc910d9a6af806ae6d45bff8.

Current status: `paused_by_user`, pending available RAM. The user briefly resumed
this replay on November 12, then explicitly chose "Leave it pending for now"
after system RAM measured 94.4% with no project Python workers running. No new
replay was launched and no result was changed. Main was clean at bc181e8.
The senior Sol Quant reviewer rechecked the unchanged audit helper, runner,
holding request, completed dividend manifest and producer export hashes; both
script syntax checks passed. The audit is ready but has not run on holding results.
Resume only on a new user instruction and below the unchanged 90% RAM ceiling.

Historical user stop on November 12: the user explicitly requested stopping
and saving all work to remote main. Do not restart jobs without a new instruction.
The holding replay ran from 10:20:07.375446 to 10:40:30.324396 UTC and was
terminated at the user's request. It had reached raw-share-plan preparation
after repeated bounded memory pauses. This was a user stop, not a model failure.
Request: data/research/investment_currency_action_outcomes_20261111/inputs/request.json,
SHA256 b97cca7abf05b933f0f42ca87946a491c8834001c336c31a1505615296c1be77.
Command: resolve-remaining-investment-outcomes through the retained
outputs/run_committed_research.py, using the same immutable 72eedcf producer.
Output: data/research/investment_currency_action_outcomes_20261111/outcomes_72eedcf.
Log: outputs/currency-action-holding-replay-20261112.log.
Stopped log SHA256 bed77deff2ed41722917d9f2c1c1e8d7f62efc2b7edb686fe32e38e4b45c59ad.
Both replay worker processes exited, no Python worker remains, and the shared
lease was successfully acquired and released in a lightweight stop check.
The outcomes_72eedcf directory is absent: no new holding result was published.
Baseline remains 3,177 known / 715 unknown unique holdings out of 3,892,
101/464 complete cohorts and no available full-period comparison in any of
the 16 views. Five exclusions, selections and all previous results are unchanged.
Stop receipt: outputs/currency-action-user-stop-20261112.json, SHA256 8b3a88c5dd7592e26c61d9e494ffb828e69320fe1f86169416884600ab294dac.
The completed dividend result, holding request, held-action facts, parent result
and producer export hashes were rechecked; all four dividend result files match.
The full holding replay and independent Quant audit are still required.
Training is not the only remaining task, and the current models have not
established complete after-cost outperformance against SPY.

Implementation checkpoint: main 72eedcf4bde54a043ccb11d60a1de1070f58a79e is
committed and pushed. Immutable export has 478 source files, manifest SHA256
5484eb2d86a58798aae5457d2d3f301ec20063af8de098c0ad896fe31148190f.
Consolidated Sol code review approved the exact action/time/units/source bindings.
The component run passed 412/413 tests; the new unbound-source test was rejected
by an earlier source-identity check than intended. Its fixture now uses a distinct
unpinned source; the entire 33-test outcome file passes, so all 413 cases are covered
on the final code. Ruff passes across investment/research and the 12 affected test
files; strict mypy passes all six changed source files. The broader 85-source mypy
run reports one pre-existing unchanged swing_oof_policy_evaluation.py:507 import
export error for MAXIMUM_SYSTEM_USED_PERCENT; that unrelated source is not changed.
Historical November 11 attempt: the same resolver stopped at final source
recheck after system RAM exceeded 90% for 120.1728 seconds (92.0421% at timeout).
It published no output. Its failed log remains retained. Today's successful resolver
supersedes that execution failure without changing source bytes or weakening guards.
On explicit resume, reuse the completed dividend result and existing b97cca7a
holding request: do not rerun the dividend resolver or exclusive request builder.
Run resolve-remaining-investment-outcomes from the same 72eedcf export to the
still-absent outcomes_72eedcf directory, with a NEW exclusive log such as
outputs/currency-action-holding-replay-20261112b.log and the unchanged 90% guard.
Then pin the actual result, run the frozen independent Quant audit, summarize
remaining evidence gaps and after-cost SPY comparisons, and close the checkpoint.
V1/beta remains deferred. No new holding-recovery or performance claim is made.

The canonical held-stock input is already staged and must be reused:
data/research/investment_currency_action_outcomes_20261111/inputs/held-stock-actions.json
SHA256 57cbe45fc98070ce5aaa37cbedd6070a871c3f322ff6fe29bef913c6a850cbca.
The sibling proofs/stock-class-tsla.json is
443a454de4c46d4edea9855bdb881124d484e3b94fdc888f5af90e5030a3edbb;
proofs/stock-class-pfe.json is
46e4baea641635a8547ef95d3063716bc61270a2ae62cfebc54e6a82a092fb9d.
The real source-only check verified both facts and rechecked 164 files; its
held-stock-actions-source-preflight-20261111.json SHA256 is
b19651d344e1f948a1d5bc98dc33c2d6e0ac9f501bce14b62cdbf86c899dbbd0.
This verifies source terms, not the still-unrun full holding returns.

Operational evidence under the chat workspace outputs directory:

- currency-action-dividend-resolution-20261111.log SHA256 787f55e6b5f059cb67ecb1dcc3aa9230379f64582f56f1dfec8d9277c0697038
- run_committed_research.py SHA256 5c145c49fc0b9be4b85418ac6dfd963dec381fddebade9285dfb6946e442f8b3
- build_currency_action_outcome_request.py SHA256 eccf00d3c9bd788f561145455add17bdbc087ae63da180d24b51bfa6c328f706
- audit_investment_currency_action_outcomes.py SHA256 0e12dcc602db748c5e9e7705aaf09ff914d032d5c916ab1cf58a39dfb691b54b
- run_investment_currency_action_outcome_audit.ps1 SHA256 7b973bac9ae5299d5a955e751badd127747f90e75ac8e7acd74797b2d874b6b7
- summarize_currency_action_outcomes.py SHA256 02fd6689af2a2975bd86236b3385cf1591599be94352d89f6d71bebfd70e51e9

The senior Quant helper independently checked all 33 retained tax records and
12 exact stock-class covers while preparing the audit. That bounded source check
does not replace the still-unrun audit of produced holdings and SPY economics.

Frozen scope below is paused by the current user instruction. It covers remaining currency and
corporate-action recovery with retained data, Sol engineering and independent senior
Quant/design/code review. Canonical main only; V1/beta deferred. The previous bounded
distribution checkpoint is closed at main 5be656b2701ad003cd9cd23f09ae96b1cc72742e.

Direct parent: data/research/investment_distribution_outcomes_20261111/distributions_f4c34aa/_manifest.json
SHA256 32a2163f939c3077bb0ed8ecd444b8430367b4ddcf8b2b7312852b6056285157.
Independent audit SHA256 5e28f129f9cdb1764a80ca8eaa583d66c5e646c39bbb1352673ca56afe909e03.
Freeze all 3,177 known dictionaries, proof references and clocks, 715 unknown keys,
3,892 total keys, 10,469 ordered placements, 464 cohorts and 16 comparison views.
Exclusions stay SLB, MDT, APA, EFX and MAS with the exact inherited stable identities.
Preserve every original comparison, source byte, label/model and all prior source
authorities. No new exclusions, reranking, replacements, refits, downloads, deletions,
amount corrections, training/feature changes, API changes or later sealed SEC use.
Keep 20 bp stock cost once and gross SPY on the same interval. Incomplete and empty
cohorts and full-period results remain unavailable. This is post-observation sensitivity.

Frozen currency scope: 33 exact original cash-action IDs, consisting of 18 explicit
ordinary common-share class candidates (FOX 7, PARA 7, GOOGL 4) and 15 candidates
whose retained annual tax payload includes an unrelated documented zero-value row.
Forty unknown holdings have only these covered currency reasons; the broader exact
ID-triggered attempt set may retain independent gaps. The nine projected FOX cohort
opportunities are coverage-only diagnostics, never a return-selection criterion or
a recovery/performance promise. No GOOG capital-class extension, opaque NKE row,
positive TECH null-date amount, PARA stock-or-cash election, AVGO payable conflict,
FCX rate inference, FANG incomplete legs or BBWI/HWM alias rule is admitted.

Existing StockCompositeIdentity and canonical tax replay remain the source of truth.
For explicit classes require actual dated ownership, the held SEC common-share class
letter and independently identical class/issuer in both ICTax lookup and calculation,
exact provider CUSIP-to-tax ISIN, unique exact USD ex-date/gross amount/payment date,
and all normal raw and amount checks. No generic competing-class waiver or new mode
flag. Match issuer case/whitespace only, retain substantive words and punctuation.
Unrelated zero rows require exact structured zero/non-income/no-ex-date fields and
unique matching actual Brutto table/date/zero row with a closed full German grammar
for purchase offers, share-only exchanges, own-share repurchases or share-only mergers.
Cash alternatives, ambiguity, extra clauses, positive amounts and malformed thousands
grouping remain unsupported. These annotations never waive the separate action book.
Preserve existing accepted purchase-offer evidence locators and ordinary behavior.

Use the existing dividend-resolution entry point with the frozen 33-event request,
unchanged eleven archive pins and full retained cross-version census. Preserve the
current 4,647 assertion objects from d44edb18dd08958e34934acecea3777ba39ed755c075360815458401b74ae3c3
as an identical ordered prefix and unchanged two amount corrections 9a44ec36….
Publish only genuinely source-validated additions. The held outcome request's new
optional currency_corroborations SourcePin must be cumulative and append-only.
Positive currency evidence must bind its supplying manifest/action to the actual
holding event/mark/source references with current verification availability.

Held-stock scope: exactly two typed actions in reviewed_held_stock_actions, a new
optional SourcePin. ReviewedHeldStockActions contains tagged ReviewedHoldingSplit
and ReviewedRegularCash facts bound to stable security, ticker, original record hash,
CUSIP, dated stock-class proof, actual official source and independent tax where cash.
Stage the two exact class-proof bytes inside canonical data before consumption; no
external-path admission bypass. No hard-coded ticker behavior or generic source waiver.
TSLA: exact retained issuer 3-for-1 split, record August 17 2022, payable August 24,
ex/effective August 25 open, provider due-bill redemption August 26. Seven selected
holdings open strictly before record and mature strictly after redemption, with every
raw unadjusted SIP session. Apply one ordinary continuing-share units x3 event at ex
open through the existing evaluator; no cash or daughter position. Require exact
unique source variant and reject simultaneous/conflicting actions and boundary lots.
PFE: exact retained issuer HTTP200 history row (Regular Cash, USD0.41, ex July27 2023,
record July28, payable September5), exact reviewed provider record fab16df8…,
dated Pfizer common class and ICTax US7170811035 exact gross/payment. Seven windows
open before ex and mature after ex. Preserve original foreign:true bytes. The typed
proof establishes this ordinary nominal unpaid claim at ex under the existing label
policy; it does not infer withholding, reinvestment or spendable payment. The refused
direct press release remains excluded. All independent raw/membership gaps remain.

Lineage/orchestration: append the two optional source roles; inherited cash completion,
recipient identity, reviewed dates and parent-unit pins remain exact. Verify cumulative
currency and held-action fact additions without replacing any old object/nested pin.
Read latest failed-attempt reasons from validated prior sidecars into an in-memory
map with source provenance; do not rewrite historical unresolved rows. Retry only
unknown holdings with newly supplied exact action IDs recorded in those latest
reasons. Preserve all carried failed/unattempted rows verbatim and original ancestor
unknown mask. A recovered pair must contain the positive triggering manifest/action
evidence in its actual specification; do not misclassify it as negative irrelevance.
Retain known rows/proofs/clocks and reconstruct every ordered cohort independently.

Source discovery pins (under C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs):
residual-currency-scope.json SHA256 c411653cff54aba8ac517fa0ba81ae6be4cdac884578d15064519eeda6261263
residual-currency-request.json SHA256 f70f0811ed361eba7af9ff41c56426b04d3d801364c2ed0ffa0430376f913f10
residual-held-stock-action-source-facts-20261111.json SHA256 a3ccb1fffdc4ef9ec5dc7fd29e177cd2a8fedab945afd5bcd5a253f8ffcfc787
residual-pfe-replay-20261111.json SHA256 f7480a76f9f853d8c2ee003ef8d6e229e269fc2675ac4e371a91f0f9010eea89
The pinned TSLA/PFE source bundle also binds exact issuer/index/transport and raw
session receipts. CCL/ILMN/F/BBWI/LLY discoveries are non-admitting future leads.

Ownership: currency engineer owns dividend_currencies, dividend_resolution if needed,
currency tests and bounded input builder. Action engineer owns reviewed_held_stock_actions,
corporate_actions and label-component integration plus their tests and fact builder.
Root owns request/lineage/orchestration and their tests, authoritative docs, publication
and commits. Coordinate shared imports/signatures before writes. Senior Quant reviews
source meaning/design and independently audits the real run. A separate Sol reviewer
performs one consolidated design/code review and reviews supported corrections.

Verification tier: component. Test exact positive classes and unrelated zero rows,
class/issuer/lookup/date/payment/amount/source/table poisons, opaque/positive/cash-election
rows, wrong Swiss grouping and preserved ordinary behavior. Test held action kind,
source/index/hash/record/CUSIP/owner/date/ratio conflicts, strict window boundaries,
simultaneous actions, double split application, and existing nominal cash accounting.
Test parent cycles, source replacement/removal, altered known rows/proofs/clocks,
latest-attempt provenance, no retrigger from carried facts, exact positive evidence
bindings, fixed ordering/costs and null incomplete means. Run affected source consumers,
Ruff and strict mypy; fixture tests do not substitute for retained-data producer runs.
Commit/export first, execute normal dividend resolver then full held-outcome recovery,
and independently audit new pairs, all prior known rows/proofs and all 464 cohorts.
One heavy job at a time under the normal shared lease; system RAM strictly below90%,
5 GiB process budget/headroom and bounded120-second pause. No guard bypass.
Close/push measured results on main. Unsupported facts leave the original unknown
outcome in place; no fabricated return, reduced selection denominator or promotion.

### Additional retained distribution outcomes — November 11

Status: `complete` for the bounded retained-distribution recovery and independent
comparison; 715 of the original 935 unknown holding outcomes
still lack sufficient evidence. Across both steps, 220/935 were recovered.
This does not complete every outcome or establish model profitability.
Sol engineering, independent design/code review and senior Quant audit completed.
Canonical main only. V1/beta remains deferred.

The five stable-identity exclusions remain SLB cik:0000087347, MDT cik:0001613103,
APA cik:0001841666, EFX cik:0000033185 and MAS cik:0000062996. All 3,892 keys,
10,469 ordered selection placements, 464 monthly cohorts and 16 comparison views
are preserved. All 3,125 previously known rows, proof references and availability
clocks are unchanged. The earlier 168 pair proof bytes remain unchanged. No training,
refit, reranking, replacement, additional exclusion or data download occurred.
Historical raw data, targets and original comparisons remain preserved.

This step attempted 55 of the remaining 767 unknown keys and
recovered 52; 712 were outside the supported
new action rules. Attempted by ticker: {"DVN": 19, "MOS": 24, "WAT": 12}.
Recovered by ticker: {"DVN": 16, "MOS": 24, "WAT": 12}.
Exact retained SEC and tax evidence proves completed special cash distributions for
fresh later MOS/DVN purchases. The WAT rule proves an existing regular-way recipient
class over the entire holding, with matching source/recipient provider records and
maturity strictly before all distribution dates. Neither rule creates cash or shares,
changes a special dividend into an ordinary one, or supplies missing crossing-period
entitlement. New evidence and outcomes retain current verification clocks.

Remaining evidence census: C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-distribution-residual-census-20261111.json
SHA256 dd346b00304b16007427dc9a7218ffd77bd53a57b3d66ec1b90798c558fe87ac.
715 unique decision and horizon holdings remain unresolved across 116 stable securities and 1,845 selected placements. Counts use the latest failed attempt for 91 holdings and stored outcome reasons for 624. Each holding counts once per cause and causes overlap. The remaining gaps affect 360 holdings for USD dividend proof, 185 for corporate-action terms, 135 for cash entitlement, 109 for raw segment binding, 56 for economic dates, 40 for distribution or due-bill bounds, 18 for official exceptions, 11 for raw security identity, and one for an unsupported unit-split participant field. The original unresolved row histories also retain superseded blockers, including 138 cash-entitlement, 112 economic-date and 47 due-bill or entitlement-bound entries.

Stock returns subtract 20 bp (0.20 percentage point) round-trip cost exactly once;
SPY is gross over the same interval. Complete cohorts increased from
96 to 101 of 464. Incomplete/empty
cohorts and all 16 full-period means remain unavailable. The annual ranking candidate
has 7/29 complete held-out-stock cohorts: 1 above SPY after stock
costs and 6 below. Its quarterly counterpart has 17/29:
5 above and 12 below. These partial, overlapping monthly
63/252-session cohorts are not a funded portfolio or an untouched final test.
They cannot establish an overall SPY edge or justify model promotion.

Implementation: f4c34aae146102383ff79c108ddf1c5645beafb3.
Immutable export SHA256 6a1f01b3179be10548afe2e24e886e7fa5e3c55a34058e69bfde7d6c2aade542.
Result: data/research/investment_distribution_outcomes_20261111/distributions_f4c34aa/_manifest.json
SHA256 32a2163f939c3077bb0ed8ecd444b8430367b4ddcf8b2b7312852b6056285157; completed 2026-11-11T13:50:33.716885+00:00.
Request: C:\project\market-predictor\data\research\investment_distribution_outcomes_20261111\inputs\request.json
SHA256 45a54d55096a3903a625f8a0474339c9598f4ba8f25e6f3299ff4e42c835204e.
Completed cash facts: data/research/investment_distribution_outcomes_20261111/inputs/completed-cash-distributions.json
SHA256 0f0353d6102f6f87ee7a3f1dfa8b863e321c50fe2d77dd618bfe8fe61bc0684f.
Existing recipient authority SHA256 d770a67e113e4146c8b14555601fd08f9306f8a49b878828d171970f43950741.
Parent result: data/research/investment_remaining_outcomes_20261111/remaining_9e59bb0/_manifest.json
SHA256 996ba93d91c59bea09ca59b5a0f6c93b72e5368499f5092d64152f3bcce8af22. Reviewed dates f82330c6… and parent units 7a231d88… unchanged.

Independent senior Quant audit: C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-distribution-outcomes-independent-audit-20261111.json
SHA256 5e28f129f9cdb1764a80ca8eaa583d66c5e646c39bbb1352673ca56afe909e03; helper SHA256 8d7d8aae3d3c75ab9f8c5167ba2e81ed8c82b6ba317abfb817683365da4e0f4a.
Completed 2026-11-11T14:03:02.294168+00:00; 15351 source files rechecked,
all unchanged. Before/after inventory SHA256 327ef554c7de7a514cd5c19d3c8f037af7b38ad4dd7f68de82ccdac3fc8e5b56.
The auditor independently reconstructed every new stock/SPY pair, source semantics,
all 464 cohort calculations, exact placements, costs and unavailable-period rules.

The first audit stopped before pair replay because its helper compared the raw SEC
request-file hash with the collection's semantic request hash. The reviewed audit-only
correction verifies both independently and requires their exact manifest linkage.
Actual retained request acceptance plus four changed-request poison cases passed.
Original helper/runner bytes and failed log are preserved; producer, input and outcome
bytes were not changed. Failed log SHA256
70cd0ff5deb12a1a87556a1d7ac00f229ad607535d3efc4a27d6195cecb10301.
Focused audit regression helper SHA256
fb5e919a753b39111695aba07b455572f2bcc73480efbeb4a441fcc1230d60c2;
regression log SHA256 108dbd00338756e9ca406f592e8d756fdb3710af24aa797559e809d9487995cf.
Successful audit log SHA256 f9ed0bb8b47b2a7a80fb975d803b377637678bd2df37c822ee1264c3b36cbf63.

Component checks and final review: C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-distribution-final-checks-20261111.json
SHA256 f309220b59c119ec9245bbbba0452227367afa571a2670733909cddba6b2f111.
The initial 13-file component suite passed 430 tests and caught one serialized class-evidence shape regression. The reviewed fix restored the original seven-field public shape and added cold/warm jurisdiction-cache coverage. The final six-file affected rerun passed 292 tests in 33.31 seconds; unaffected component results remain valid. Final Ruff over 16 changed paths and strict mypy over 11 production sources passed. Exact MOS/DVN source-only replay passed after the final code change and rechecked 176 retained sources. Parent-ledger focused tests separately passed 27 cases; its bounded real-parent read checked all 168 previous pair proofs and original ordering. The independent Sol code reviewer approved the implementation and class-shape fix with no outstanding finding; senior Quant approved the source semantics and separately reviewed audit helper. Repository-wide release tests were deliberately not run for this component checkpoint.
The normal shared heavy-job lease, strictly below 90% system RAM policy, 5 GiB process
budget/headroom and bounded pressure pause remain. Exact identity cache hits still
check memory, sessions, source pins and ownership; only repeated cleanup is avoided.
No repository-wide suite, product release or production-model approval is claimed.

Actual producer log: C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-distribution-outcomes-20261111.log
SHA256 c1754caa8997c9a381afe68141f14508f1eef855bacd1e22416edeef869a70c0.
The immutable producer exited successfully at 2026-11-11T13:50:59.611940Z,
63 minutes 38.7 seconds after launch. This is an observed run duration, not a
controlled performance comparison. All reviewed input bytes remained unchanged.

| Horizon / feature profile / evaluation scope | Complete dates before → after | Unknown placements |
| --- | ---: | ---: |
| 252 / all_features / security_transfer | 9 → 9 of 29 | 28 |
| 252 / all_features / temporal | 0 → 0 of 29 | 258 |
| 252 / technical_news / security_transfer | 6 → 7 of 29 | 38 |
| 252 / technical_news / temporal | 0 → 0 of 29 | 265 |
| 252 / technical_relationships / security_transfer | 8 → 9 of 29 | 31 |
| 252 / technical_relationships / temporal | 0 → 0 of 29 | 260 |
| 252 / within_date_return_rank / security_transfer | 7 → 7 of 29 | 36 |
| 252 / within_date_return_rank / temporal | 0 → 0 of 29 | 285 |
| 63 / all_features / security_transfer | 16 → 16 of 29 | 14 |
| 63 / all_features / temporal | 0 → 0 of 29 | 141 |
| 63 / technical_news / security_transfer | 16 → 17 of 29 | 15 |
| 63 / technical_news / temporal | 0 → 0 of 29 | 148 |
| 63 / technical_relationships / security_transfer | 18 → 19 of 29 | 13 |
| 63 / technical_relationships / temporal | 0 → 0 of 29 | 140 |
| 63 / within_date_return_rank / security_transfer | 16 → 17 of 29 | 17 |
| 63 / within_date_return_rank / temporal | 0 → 0 of 29 | 156 |

Next work must address the explicit residual source gaps without changing selections
or silently discarding missing holdings. The pinned outcomes and attempts retain
each missing reason. Current unknowns cannot be counted as zero returns or omitted
from a full-period SPY comparison. V1/beta remains later than demonstrated improvement.

### Remaining retained holding outcomes — November 11

Status: `complete` for this bounded retained-action recovery step; 767
of the original 935 unknown decision/horizon holding outcomes remain unresolved.
This is not completion of every remaining outcome or model promotion.
The five stable-identity exclusions remain SLB cik:0000087347, MDT cik:0001613103,
APA cik:0001841666, EFX cik:0000033185 and MAS cik:0000062996. Canonical main only.
Sol engineering, independent design/code review and senior Quant audit completed.
V1/beta remains deferred. No training, refit, reranking, replacement, further exclusion,
data download, raw deletion or historical target/model/source overwrite occurred.

Actual recovery attempted 256 of the 935 unknown holdings and verified
168 additional stock/SPY outcomes; 679 were outside the
source-supported action rules. Failed and unattempted survivor rows remain verbatim.
All 2,957 already-known rows and clocks are unchanged. The new comparison preserves
3,892 distinct decision/horizon keys, 10,469 ordered placements, 16 views and 464
monthly cohorts. Complete cohorts increased from 89 to
96 of 464. Every incomplete cohort and all 16 full-period
means remain unavailable. attempts.json records the actual attempt scope, current
clocks and remaining reasons; outcomes.json keeps every survivor and pair proof.

New outcomes reuse the established raw stock/SPY holding compiler and accounting
kernel. Stock returns subtract the 20 bp (0.20 percentage point) round-trip cost
exactly once; SPY is gross over the identical interval. Only exact, unambiguous
same-CUSIP name-only records, existing reviewed official name/parent-spin facts,
and parent holdings strictly before all observed distribution economic dates can
remove an irrelevant-action gap. Valid raw ownership and membership are prerequisites.
Boundary/crossing ambiguity and independent cash, split, successor, class, official
or raw-source gaps remain. All 12 WAT daughter-side cases fail the parent-only rule.
Parent marked holdings are not described as simulated pre-event sales. New facts
carry current verification clocks and original provider/official receipts in the
actual holding specifications. Prior amendments use the full ancestor unknown mask.

The annual ranking candidate has 7/29 complete held-out-stock cohorts:
1 above SPY after stock costs and 6 below. Its quarterly
counterpart has 16/29 complete cohorts: 5 above and
11 below. These are partial descriptive comparisons, not an
overall success rate. Monthly 63/252-session holdings overlap; their cohort returns
are not a funded portfolio and cannot be compounded or averaged over available dates
to replace missing full-period results. No overall SPY edge is established.
This remains a post-observation exclusion sensitivity, not an untouched test.

Implementation: 9e59bb0fb3506b9ad5b7e5f18098148bf5d4bf5f (main).
Immutable producer export SHA256 7fda788652ba3180855528ef5dda68edb4deae5c8c866a18ba353022b3139361.
Result: data/research/investment_remaining_outcomes_20261111/remaining_9e59bb0/_manifest.json
SHA256 996ba93d91c59bea09ca59b5a0f6c93b72e5368499f5092d64152f3bcce8af22; completed 2026-11-11T11:45:13.166869+00:00.
Request: data/research/investment_remaining_outcomes_20261111/inputs/request.json
SHA256 a13198ddb4e8a98aefdd2555070d70f6f64ff2d71ce493b731564dc305d3c546.
Parent: data/research/investment_exclusion_evaluation_20261111/exclusions_2a1a352/_manifest.json
SHA256 dd6d83ca839ac293d903d684d18d6fb7c383e2ec5c74889a1ca9d5d6b13e6237.
Parent audit SHA256 3c81171f7c0a6884fbfb3d05dc07eaa3c0434b9395b401625095ed84c68e4cf5.
Reviewed dates f82330c6cedb57152f8904ba6c007d911cf57bf8dc923230ba017f2cb0e5328d
and parent units 7a231d88133128ed98823de9098f27157f1360e2065e792f7e9320ff4e6fd09d
remain unchanged. The original five-stock and unexcluded comparisons are preserved.

Independent senior Quant audit: C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-remaining-outcomes-independent-audit-20261111.json
SHA256 dc2adc3d2cbce9de2215b2fb1896f84b47643c5d9b10bfd04e3ea10ac9f95fa9; helper SHA256 7b91572f959ec85ee7a4f890b147d598b502ee806757bc7b9d269c15c39a5625.
Completed 2026-11-11T12:01:23.119679+00:00. It independently checked every new pair,
all original survivor rows/clocks, exact action relevance, source/export hashes,
all 464 cohort calculations, ordering, exclusions, cost and unavailable-period rules.
All 15,288 source rechecks were unchanged. The first audit refused repeated provider
references in 20 pairs; independent review proved each repeat fully identical and
authorized, with no cash or unit duplication. The corrected audit revalidates every
occurrence, requires complete equality, and checks exact mark-reference multiplicity;
11 targeted poison checks passed. No published source, outcome or return was changed.
The failed audit log SHA256 is
1696f4b8f10502fdd1b715e5fc341e125ac1ca3c6d3231f9607d0eb4e681601f;
successful audit log SHA256 is
0c703c793e7e6f3c200565abd370ea2addd30cd82cfbcb0826ba721f44021718.
Original helper/runner bytes remain preserved. The producer completed in 68 minutes
27 seconds, with log SHA256
a7366fbfc0512528f1c61dce45f994ee2874d49d615bd51801e53d577165dae3.
This elapsed duration is a run observation, not a controlled speedup measurement.

Component verification: 131 focused tests (12.78 s), plus 20 original label consumer
tests (3.75 s); Ruff 17 affected paths and strict mypy 10 source modules passed.
Final logs under chat outputs: investment-remaining-outcome-component-checks-20261111d.log
SHA256 46b910edd4ddbe9e074c735a58b0f329592953157c8913e634e3946547a99873;
investment-remaining-outcome-label-consumers-20261111.log SHA256
3a20a45540a95491b96e56a869aca662fb6713d4cbb1c590ed2d63dd558bfe7b.
Runtime source-recheck change e8f1ab2 separately passed 66 pressure/tamper/cadence
tests (3.24 s), Ruff and strict mypy. Its log SHA256 is
5119fdf95642b0d92bb912eae47ba67408864bac65fb447f7b2c53daef00dea8.
Both full source rechecks and the memory guard before every file hash are retained;
only explicit GC cadence changes (32 files / 8 MiB, with oversized-file boundaries).
The normal shared lease, strictly below 90% system RAM policy, process budget and
bounded pressure pause remain active. No full-suite/product release claim is made.

| Horizon / feature profile / evaluation scope | Complete dates before → after | Unknown placements |
| --- | ---: | ---: |
| 252 / all_features / security_transfer | 9 → 9 of 29 | 29 |
| 252 / all_features / temporal | 0 → 0 of 29 | 278 |
| 252 / technical_news / security_transfer | 6 → 6 of 29 | 40 |
| 252 / technical_news / temporal | 0 → 0 of 29 | 295 |
| 252 / technical_relationships / security_transfer | 8 → 8 of 29 | 33 |
| 252 / technical_relationships / temporal | 0 → 0 of 29 | 287 |
| 252 / within_date_return_rank / security_transfer | 7 → 7 of 29 | 38 |
| 252 / within_date_return_rank / temporal | 0 → 0 of 29 | 310 |
| 63 / all_features / security_transfer | 15 → 16 of 29 | 18 |
| 63 / all_features / temporal | 0 → 0 of 29 | 150 |
| 63 / technical_news / security_transfer | 13 → 16 of 29 | 17 |
| 63 / technical_news / temporal | 0 → 0 of 29 | 156 |
| 63 / technical_relationships / security_transfer | 17 → 18 of 29 | 15 |
| 63 / technical_relationships / temporal | 0 → 0 of 29 | 149 |
| 63 / within_date_return_rank / security_transfer | 14 → 16 of 29 | 18 |
| 63 / within_date_return_rank / temporal | 0 → 0 of 29 | 163 |

All complete held-out-stock ranking-candidate cohorts follow. The full 16-view
comparison, including controls and incomplete dates, is in comparisons.json.

| Horizon | Selection date | Stocks | Stock net | SPY gross | Difference (percentage points) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 252 | 2022-06-30 | 8 | +21.8864% | +19.5858% | +2.3006 |
| 252 | 2022-08-31 | 6 | -3.0964% | +16.4989% | -19.5953 |
| 252 | 2022-09-30 | 8 | +11.3324% | +18.5607% | -7.2283 |
| 252 | 2022-10-31 | 7 | -13.2113% | +10.0039% | -23.2152 |
| 252 | 2022-11-30 | 5 | +4.6590% | +13.9049% | -9.2460 |
| 252 | 2022-12-30 | 7 | +23.6724% | +23.6891% | -0.0167 |
| 252 | 2024-05-28 | 7 | +12.9701% | +13.4827% | -0.5125 |
| 63 | 2022-03-31 | 8 | -24.1287% | -15.5508% | -8.5779 |
| 63 | 2022-06-30 | 7 | +3.9828% | -3.2328% | +7.2156 |
| 63 | 2022-08-31 | 8 | +3.7171% | +4.1707% | -0.4536 |
| 63 | 2022-09-30 | 8 | +4.2994% | +6.4062% | -2.1068 |
| 63 | 2022-10-31 | 7 | +18.9546% | +5.7521% | +13.2025 |
| 63 | 2022-11-30 | 6 | +6.5541% | -0.6846% | +7.2388 |
| 63 | 2022-12-30 | 7 | +25.5538% | +7.3071% | +18.2467 |
| 63 | 2023-02-28 | 7 | -2.6585% | +6.6453% | -9.3038 |
| 63 | 2023-04-28 | 8 | +21.9878% | +10.5804% | +11.4074 |
| 63 | 2023-07-31 | 7 | -16.6135% | -9.6449% | -6.9686 |
| 63 | 2023-09-29 | 7 | +9.9954% | +11.8598% | -1.8644 |
| 63 | 2023-10-31 | 8 | +8.6016% | +17.1532% | -8.5516 |
| 63 | 2024-01-31 | 8 | +2.6687% | +3.5728% | -0.9041 |
| 63 | 2024-02-29 | 8 | -3.8587% | +2.9913% | -6.8500 |
| 63 | 2024-04-30 | 8 | -6.5900% | +10.2096% | -16.7996 |
| 63 | 2024-05-28 | 7 | +6.4983% | +7.1601% | -0.6618 |

Next source work must retain these exact selections and verified facts. Retained
MOS special-cash and DVN paid-distribution issuer documents are discovery evidence;
they require a separately frozen typed entitlement/completion proof before any
admission. Other unresolved dividend, split, ownership and candle gaps must stay
explicit. Do not add exclusions after inspecting returns or promote the model.

### Five-stock exclusion evaluation — November 11

Status: `complete` for the user-authorized separate exclusion evaluation. Original
comparison remains unchanged. Excluded stable identities: SLB cik:0000087347,
MDT cik:0001613103, APA cik:0001841666, EFX cik:0000033185 and MAS cik:0000062996.
The same rule removed all their selected positions, including known outcomes.
No replacements, reranking, refits, downloads, source reconstruction or API changes
occurred. Every surviving outcome dictionary, order and verification clock is reused
unchanged. Equal weights are recalculated over the smaller remaining cohort.
V1/beta stays deferred; no model promotion or fully trained-model claim is made.

This is a post-observation exclusion sensitivity. It does not establish an unbiased
investment universe or out-of-sample stock-exclusion policy. The implementation is
reusable for explicit stable-identity exclusions and does not hardcode these stocks.
Its typed request pins the parent and exclusion reasons; parent count/identity/date/
horizon/order/hash/arithmetic mismatches refuse publication. Shared cohort math now
also handles empty dates/views as unavailable. Stock net values already include
the 20 bp (0.20 percentage point) round-trip cost: it is not charged again. SPY
remains the gross benchmark on the same interval. Unknown survivors and empty dates
keep cohort returns null; any incomplete date keeps the full-period view mean null.

Actual publication removed 78 distinct decision/horizon holding keys: 51 unknown
and 27 previously verified. They account for 323 placements: 225 unknown and 98
verified. The remainder contains 3,892 distinct keys and 10,469 placements, with
935 unknown keys and 2,394 unknown placements. Coverage improves from 49/464 to
89/464 complete cohorts; all 40 newly complete cohorts are held-out-stock views.
Eleven already-complete cohorts also change composition. No cohort is empty.
All eight temporal views still have zero complete dates, and all 16 full-period
means remain null. Removing five stocks does not resolve the other missing facts.

The annual ranking candidate now has 7/29 complete held-out-stock dates: 1 beats
SPY after stock costs and 6 trail it. The quarterly candidate has 14/29 complete
dates: 4 beat SPY and 10 trail it. These are descriptive counts for the available
dates, not independent trials or a full-period success rate. No overall SPY edge
has been established. Monthly 63/252-session cohorts overlap and are not a funded
portfolio; do not compound their returns or average only complete dates as a
substitute for missing full-period results. Composition changes are not model gains.

Implementation commit: 2a1a3526ee02b984ea4ff747928de217f7e22ef6; canonical main only.
Result: data/research/investment_exclusion_evaluation_20261111/exclusions_2a1a352/_manifest.json
SHA256 dd6d83ca839ac293d903d684d18d6fb7c383e2ec5c74889a1ca9d5d6b13e6237; completed 2026-11-11T09:36:58.680648+00:00.
Request: data/research/investment_exclusion_evaluation_20261111/inputs/request.json
SHA256 4e1e452b09a19dbfd14757bd377512210a6e23a40fe4c9f9d6978a12633cea22.
Parent: data/research/investment_selected_outcomes_20261111/outcomes_bc2fb11/_manifest.json
SHA256 5a7f62df2c378eeac5ce1c1bd96b181c02cea2836e846c0b55e0fcc4482f2ccb;
parent audit SHA256 d49fa9c3fd42c48125247393932eee71c19f9eb43aa1b5e2742585fd38361224.
The new command rechecks consumed parent files, not all historical raw sources;
the earlier independently audited source lineage is reused by immutable reference.
Independent senior Sol Quant child audit: C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-exclusion-evaluation-independent-audit-20261111.json
SHA256 3c81171f7c0a6884fbfb3d05dc07eaa3c0434b9395b401625095ed84c68e4cf5, completed 2026-11-11T09:37:55.705930+00:00.
It independently reconstructed all 464 cohorts and survivor rows/clocks, exclusions,
cost arithmetic, null full-period results and source/export hashes. The arithmetic
uses no production comparison functions. Parent data and historical results remain
byte-identical. Audit helper SHA256 504fcc7f46e6334587d952ec21ce6d6bd0d2269608fcc89d1fcee58527649395.

Validation tier: component. Final focused checks passed: 28 tests in 5.12 seconds,
with no skips, using tests/test_investment_exclusion_evaluation.py and
tests/test_investment_selection_outcomes.py through the normal shared-lease wrapper
chat outputs/run_investment_component_checks.py. Ruff passed on all four changed
paths, strict mypy passed on the three changed source modules, and whitespace passed.
Final check log: chat outputs/investment-exclusion-final-checks-20261111b.log,
SHA256 f21ae8900e2fcc280fc0d809d29a5b49ae9937a04eacb763b4529df1401bfc0d.
Independent Sol design/code review approved the implementation and audit helper.
Full-suite, release, model-training, serving and promotion checks were not run;
this change derives a separate research comparison from previously audited data.

Actual execution used chat outputs/run_committed_research.py with command
evaluate-investment-exclusions, producer 2a1a3526ee02b984ea4ff747928de217f7e22ef6,
the pinned request above and output
data/research/investment_exclusion_evaluation_20261111/exclusions_2a1a352.
The immutable source export contains 472 files; _export.json SHA256
e32c9ecaae9e08057412361993d3a4b53cc37dff25415faeb0c8a0f0d619314a.
Execution log: chat outputs/investment-exclusion-evaluation-20261111.log.
The run exited 0 at 2026-11-11T09:36:59.300917+00:00. Its normal memory guard
paused at 90.13% system usage and resumed 5.14 seconds later at 89.85%; no bypass
or changed memory threshold was used. Heavy jobs remained serialized.

Independent audit entry point: chat outputs/run_investment_exclusion_evaluation_audit.ps1
with ResultManifestPath, ManifestSha256, ProducerCommit and ProducerExportSha256
bound to this published result. The reviewed helper is
chat outputs/audit_investment_exclusion_evaluation.py, SHA256
504fcc7f46e6334587d952ec21ce6d6bd0d2269608fcc89d1fcee58527649395;
runner SHA256 67f8d36530a47959d2cf0d64e7d4b5cd6a5ce2a44e5d550a2d63eb2ed6ab5522.
Inputs, outputs and logs are immutable; a future independent run needs fresh output
paths. Documentation-only closure uses reference/history and whitespace checks.

| Horizon / feature profile / evaluation scope | Complete dates before → after | Removed placements | Remaining placements | Unknown placements |
| --- | ---: | ---: | ---: | ---: |
| 252 / all_features / security_transfer | 4 → 9 of 29 | 21 | 211 | 30 |
| 252 / all_features / temporal | 0 → 0 of 29 | 32 | 1085 | 317 |
| 252 / technical_news / security_transfer | 2 → 6 of 29 | 26 | 206 | 40 |
| 252 / technical_news / temporal | 0 → 0 of 29 | 31 | 1086 | 336 |
| 252 / technical_relationships / security_transfer | 3 → 8 of 29 | 26 | 206 | 34 |
| 252 / technical_relationships / temporal | 0 → 0 of 29 | 28 | 1089 | 330 |
| 252 / within_date_return_rank / security_transfer | 3 → 7 of 29 | 17 | 215 | 38 |
| 252 / within_date_return_rank / temporal | 0 → 0 of 29 | 20 | 1097 | 346 |
| 63 / all_features / security_transfer | 8 → 15 of 29 | 15 | 217 | 21 |
| 63 / all_features / temporal | 0 → 0 of 29 | 14 | 1103 | 196 |
| 63 / technical_news / security_transfer | 8 → 13 of 29 | 16 | 216 | 20 |
| 63 / technical_news / temporal | 0 → 0 of 29 | 16 | 1101 | 214 |
| 63 / technical_relationships / security_transfer | 12 → 17 of 29 | 14 | 218 | 18 |
| 63 / technical_relationships / temporal | 0 → 0 of 29 | 17 | 1100 | 211 |
| 63 / within_date_return_rank / security_transfer | 9 → 14 of 29 | 16 | 216 | 24 |
| 63 / within_date_return_rank / temporal | 0 → 0 of 29 | 14 | 1103 | 219 |

All complete ranking-candidate cohorts on held-out stocks are listed below. The
full 16-view comparison, including controls and unavailable dates, is retained in
comparisons.json beside the result manifest; outcomes.json preserves survivor rows.

| Horizon | Selection date | Stocks retained | Stock net | SPY gross | Difference (percentage points) | Original date complete? |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 252 | 2022-06-30 | 8 | +21.8864% | +19.5858% | +2.3006 | yes |
| 252 | 2022-08-31 | 6 | -3.0964% | +16.4989% | -19.5953 | no |
| 252 | 2022-09-30 | 8 | +11.3324% | +18.5607% | -7.2283 | yes |
| 252 | 2022-10-31 | 7 | -13.2113% | +10.0039% | -23.2152 | no |
| 252 | 2022-11-30 | 5 | +4.6590% | +13.9049% | -9.2460 | no |
| 252 | 2022-12-30 | 7 | +23.6724% | +23.6891% | -0.0167 | no |
| 252 | 2024-05-28 | 7 | +12.9701% | +13.4827% | -0.5125 | yes |
| 63 | 2022-03-31 | 8 | -24.1287% | -15.5508% | -8.5779 | yes |
| 63 | 2022-06-30 | 7 | +3.9828% | -3.2328% | +7.2156 | no |
| 63 | 2022-08-31 | 8 | +3.7171% | +4.1707% | -0.4536 | yes |
| 63 | 2022-09-30 | 8 | +4.2994% | +6.4062% | -2.1068 | yes |
| 63 | 2022-10-31 | 7 | +18.9546% | +5.7521% | +13.2025 | no |
| 63 | 2022-12-30 | 7 | +25.5538% | +7.3071% | +18.2467 | yes |
| 63 | 2023-02-28 | 7 | -2.6585% | +6.6453% | -9.3038 | no |
| 63 | 2023-04-28 | 8 | +21.9878% | +10.5804% | +11.4074 | yes |
| 63 | 2023-07-31 | 7 | -16.6135% | -9.6449% | -6.9686 | yes |
| 63 | 2023-09-29 | 7 | +9.9954% | +11.8598% | -1.8644 | no |
| 63 | 2023-10-31 | 8 | +8.6016% | +17.1532% | -8.5516 | yes |
| 63 | 2024-01-31 | 8 | +2.6687% | +3.5728% | -0.9041 | yes |
| 63 | 2024-02-29 | 8 | -3.8587% | +2.9913% | -6.8500 | yes |
| 63 | 2024-05-28 | 7 | +6.4983% | +7.1601% | -0.6618 | no |

Next work remains the source/identity/action facts affecting the 935 retained
unknown keys, or a separately frozen model improvement experiment. Do not add more
exclusions after inspecting their returns or treat these partial results as grounds
to promote the model. No additional experiment or V1/beta checkpoint is active.

### Fixed selected-stock outcomes and after-cost SPY comparison — November 11

Status: `complete` for the bounded retained-source recovery and measured comparison.
Unproven investment performance and remaining source gaps are explicitly open;
this is not model promotion, a funded portfolio result or a fully trained model claim.
Code is committed/pushed on canonical main: outcome implementation 5b557be28556063f9132021417851c96c0d0d31e,
final source remedy/producer bc2fb1174364e7ecc0dc21ba627ea5294df33f63. The immutable final export
contains 471 source files. No extra checkout, downloads, exclusions, refits or API
changes occurred. V1/beta remains deferred.

The actual replay preserved all 16 views, 29 dates/view, 464 cohorts and 10,792
ordered selected placements (3,970 distinct decision/horizon holding keys).
It reused all 2,771 original verified values and clocks, recovered
213 of the original 1,199 unknown keys, and retained
986 unresolved keys without replacement, zero filling or deletion.
Recovered keys span 26 securities
and 570 selected placements across the 16 views;
unresolved keys span 132 securities.
49/464 cohorts now have every selected return verified.
The cost is 20 basis points once per stock holding; SPY is gross on the same interval.
Incomplete dates have null returns; any incomplete date leaves its full-period view
mean null. Monthly 63/252-session holding cohorts overlap: do not compound their means
or describe these as a funded monthly strategy. Inspected history remains development
research, not untouched/prospective performance. No original target, model, input,
source bytes or historical verification timestamp was changed.

Actual execution began 2026-11-11T00:59:41.120228+00:00; source verification clock
2026-11-11T01:30:24.186766+00:00; report completed 2026-11-11T03:13:19.250998+00:00.
Execution log: C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-selected-outcomes-20261111.log.
Outcome report: data/research/investment_selected_outcomes_20261111/outcomes_bc2fb11/_manifest.json
SHA256 5a7f62df2c378eeac5ce1c1bd96b181c02cea2836e846c0b55e0fcc4482f2ccb.
Request SHA256 28555f0eee9f35ec994327baee749db20a2501b9ed3880fb9edd6f928b8d848f.
Independent senior Sol Quant audit: C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-selected-outcomes-independent-audit-20261111.json
SHA256 d49fa9c3fd42c48125247393932eee71c19f9eb43aa1b5e2742585fd38361224; completed 2026-11-11T03:32:39.653666+00:00.
The independent audit reconstructed original selections, old values/clocks, every
recovered pair's raw bars/action accounting and costs, all 464 cohorts/16 views,
source amendments and immutable file hashes. Software tests alone are not this
actual-data evidence.
It rehashed 21,133 distinct evidence files, all unchanged,
and independently replayed all 19 appended identity proofs and 149 new currency
assertions from the two source passes.
The first audit attempt stopped on its canonical-root restriction for 11 historical
implementation files already pinned by the accepted dataset. Preserve failed log
chat outputs/investment-selected-outcomes-independent-audit-20261111a.log and old
helper audit_investment_selected_outcomes_3dadaf6b.py. The independently reviewed
helper correction binds exactly those 11 paths and their digests to the frozen
ranking/training-request/dataset chain, then byte-hashes and rechecks each file;
all other external paths remain rejected. Published producer artifacts were unchanged.
Successful audit helper SHA256: cb229673d88058eebcf876b5ceb9e88f8aaec99acefe79943cf49f0d2f882827.

Retained source recovery preserved all 391 original class proofs and appended 19
verified proofs (410 total), including the April 25 Microsoft cover identified by
body 8db90640757a0a7f635851caf04c37155c0c962b4607c1176e844ef9f4c3bbd3.
The common-stock parser now separates explicitly tagged debt-note rows while refusing
unknown/mixed/reused contexts. The dividend parser admits an undated zero PAYMENT
only when its exact field shape and uniquely matching purchase-offer annotation in
the selected dividend's actual Brutto table prove it is not a distribution; all
other date, amount, source and entitlement checks remain strict. No ticker or
purchase-price whitelist was introduced. The two demonstrated EL/AAP examples and
all actually admitted events were replayed from retained captures. A separate bounded
independent source audit reproduced all 60 supplemental assertions, including 49
with purchase-offer annotations, and rehashed 111 source pins; receipt
C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs/investment-selected-dividend-offer-annotations-20261111.json SHA256 768fdce34b7c9f62bc8abdfac7a755951b59a9e2c04bb29a1e33c047c0399f37.
The first dividend pass added 89 assertions to 4,498;
the supplemental pass added 60 to the preserved 4,587,
for 4647 total. 393 of
453 supplemental requested events remain unavailable. The superseded unpublished
5b557be supplemental attempt was stopped before output at 2026-11-11T00:16:05.657942+00:00;
its log is preserved, and the same frozen request ran using the final source code.

Validation: component tier, 324 focused tests for the outcome/pair/class change,
71 affected tests after the final import/type correction, and 283 final affected
tests for the tax annotation remedy; counts overlap and are not a unique-test total.
Ruff/strict mypy and whitespace passed for affected files/packages. Real score/label
metadata replay verified all placements, 2,771 values/clocks and 909 pins. The retained
MSFT parser and EL/AAP parser-level checks passed. Independent Sol design/code and
senior Quant reviews approved the supported corrections. Full-suite, release,
serving and promotion checks were deliberately not run for this research component.
Recorded execution entry points (completed; output paths are immutable and must not
be reused for a new run): chat outputs/run_investment_selection_outcomes.py with
--command resolve-investment-selection-outcomes, --producer-commit bc2fb1174364e7ecc0dc21ba627ea5294df33f63,
--request data/research/investment_selected_outcomes_20261111/inputs/outcome-request.json,
--request-sha256 28555f0eee9f35ec994327baee749db20a2501b9ed3880fb9edd6f928b8d848f,
--output data/research/investment_selected_outcomes_20261111/outcomes_bc2fb11 and
--log outputs/investment-selected-outcomes-20261111.log. The accepted independent
audit command is chat outputs/run_investment_selected_outcomes_audit.ps1
-ManifestSha256 5a7f62df2c378eeac5ce1c1bd96b181c02cea2836e846c0b55e0fcc4482f2ccb; it binds the exact producer and both source
passes. Successful log chat outputs/investment-selected-outcomes-independent-audit-20261111b.log
SHA256 fc8ed80765638a328a15b12ceb3f2ca45bdc19ec548b7bcc3c1ecff025e38e37.
Retained focused-check logs in chat outputs: investment-selected-outcome-component-checks-20261111.log
(324 passed), investment-selected-outcome-post-type-checks-20261111.log (71 passed),
investment-purchase-offer-component-checks-20261111b.log (283 passed),
investment-selected-outcome-static-checks-20261111b.log and
investment-purchase-offer-static-checks-20261111.log. Tests used the exclusive lease
wrapper run_investment_component_checks.py. Do not rerun unchanged research just to
recreate narrative evidence. Documentation closure requires reference and whitespace
checks, not another training or full-suite run.
Normal exclusive lease and below-90% RAM guard stayed active; no bypass was used.
A bounded 20-second nonblocking runtime sample showed movement from extension hash
checks into segment binding after action/currency replay. Explicit memory cleanup
occupied 371/377 samples in that slice, not a whole-run attribution. The original
producer continued unchanged; no sampling of locals or second heavy job occurred.
Profile: chat outputs/live-selected-outcomes-66372-nonblocking-20261111.raw, SHA256
37bee430689360d7f9bb51e97adfc2341b4c61d17b566513ee0a598f17ffc7e0.

| Horizon / feature profile / scope | Complete dates | Recovered placements | Unresolved placements |
| --- | ---: | ---: | ---: |
| 252 / all_features / security_transfer | 4/29 | 6 | 48 |
| 252 / all_features / temporal | 0/29 | 58 | 341 |
| 252 / technical_news / security_transfer | 2/29 | 11 | 58 |
| 252 / technical_news / temporal | 0/29 | 65 | 358 |
| 252 / technical_relationships / security_transfer | 3/29 | 15 | 51 |
| 252 / technical_relationships / temporal | 0/29 | 68 | 349 |
| 252 / within_date_return_rank / security_transfer | 3/29 | 22 | 50 |
| 252 / within_date_return_rank / temporal | 0/29 | 72 | 359 |
| 63 / all_features / security_transfer | 8/29 | 12 | 32 |
| 63 / all_features / temporal | 0/29 | 55 | 205 |
| 63 / technical_news / security_transfer | 8/29 | 12 | 32 |
| 63 / technical_news / temporal | 0/29 | 47 | 225 |
| 63 / technical_relationships / security_transfer | 12/29 | 14 | 28 |
| 63 / technical_relationships / temporal | 0/29 | 46 | 220 |
| 63 / within_date_return_rank / security_transfer | 9/29 | 20 | 35 |
| 63 / within_date_return_rank / temporal | 0/29 | 47 | 228 |

All complete dated cohorts are listed below, including unfavorable comparisons.
These are equal-weight holding returns for each stated decision date and horizon,
not a return across the entire evaluation period. No available-date average is
substituted for missing dates. A 20 bp stock round-trip cost is 0.20 percentage
points deducted once from each stock holding; SPY is the gross benchmark.

| Horizon / feature profile / scope | Decision date | Stocks | Stock net return | SPY gross return | Difference (percentage points) |
| --- | --- | ---: | ---: | ---: | ---: |
| 252 / all_features / security_transfer | 2022-06-30 | 8 | +24.3139% | +19.5858% | +4.7281 |
| 252 / all_features / security_transfer | 2022-07-29 | 8 | +13.0258% | +13.1620% | -0.1362 |
| 252 / all_features / security_transfer | 2023-01-31 | 8 | -1.0705% | +22.3644% | -23.4348 |
| 252 / all_features / security_transfer | 2024-05-28 | 8 | +17.0689% | +13.4827% | +3.5862 |
| 252 / technical_news / security_transfer | 2022-06-30 | 8 | +24.3139% | +19.5858% | +4.7281 |
| 252 / technical_news / security_transfer | 2024-05-28 | 8 | +12.5124% | +13.4827% | -0.9703 |
| 252 / technical_relationships / security_transfer | 2022-06-30 | 8 | +24.3139% | +19.5858% | +4.7281 |
| 252 / technical_relationships / security_transfer | 2022-09-30 | 8 | +7.6531% | +18.5607% | -10.9077 |
| 252 / technical_relationships / security_transfer | 2024-05-28 | 8 | +16.1239% | +13.4827% | +2.6412 |
| 252 / within_date_return_rank / security_transfer | 2022-06-30 | 8 | +21.8864% | +19.5858% | +2.3006 |
| 252 / within_date_return_rank / security_transfer | 2022-09-30 | 8 | +11.3324% | +18.5607% | -7.2283 |
| 252 / within_date_return_rank / security_transfer | 2024-05-28 | 8 | +6.3784% | +13.4827% | -7.1042 |
| 63 / all_features / security_transfer | 2022-06-30 | 8 | -0.8117% | -3.2328% | +2.4212 |
| 63 / all_features / security_transfer | 2022-12-30 | 8 | +24.2324% | +7.3071% | +16.9254 |
| 63 / all_features / security_transfer | 2023-01-31 | 8 | -14.9412% | +1.7609% | -16.7021 |
| 63 / all_features / security_transfer | 2023-03-31 | 8 | +6.1798% | +8.9466% | -2.7668 |
| 63 / all_features / security_transfer | 2023-07-31 | 8 | -17.7403% | -9.6449% | -8.0954 |
| 63 / all_features / security_transfer | 2024-02-29 | 8 | -4.5376% | +2.9913% | -7.5289 |
| 63 / all_features / security_transfer | 2024-03-28 | 8 | -7.3733% | +4.2283% | -11.6016 |
| 63 / all_features / security_transfer | 2024-04-30 | 8 | -2.2398% | +10.2096% | -12.4495 |
| 63 / technical_news / security_transfer | 2022-07-29 | 8 | -5.0343% | -6.7392% | +1.7049 |
| 63 / technical_news / security_transfer | 2022-08-31 | 8 | +5.3635% | +4.1707% | +1.1927 |
| 63 / technical_news / security_transfer | 2022-12-30 | 8 | +24.1859% | +7.3071% | +16.8788 |
| 63 / technical_news / security_transfer | 2023-01-31 | 8 | -15.9554% | +1.7609% | -17.7163 |
| 63 / technical_news / security_transfer | 2023-03-31 | 8 | +8.5453% | +8.9466% | -0.4014 |
| 63 / technical_news / security_transfer | 2023-07-31 | 8 | -20.8302% | -9.6449% | -11.1853 |
| 63 / technical_news / security_transfer | 2024-02-29 | 8 | -5.6706% | +2.9913% | -8.6619 |
| 63 / technical_news / security_transfer | 2024-03-28 | 8 | -6.8591% | +4.2283% | -11.0874 |
| 63 / technical_relationships / security_transfer | 2022-04-29 | 8 | +1.0720% | +0.0672% | +1.0048 |
| 63 / technical_relationships / security_transfer | 2022-06-30 | 8 | +0.6533% | -3.2328% | +3.8861 |
| 63 / technical_relationships / security_transfer | 2022-07-29 | 8 | -5.0343% | -6.7392% | +1.7049 |
| 63 / technical_relationships / security_transfer | 2022-09-30 | 8 | +6.6339% | +6.4062% | +0.2278 |
| 63 / technical_relationships / security_transfer | 2022-12-30 | 8 | +21.5559% | +7.3071% | +14.2488 |
| 63 / technical_relationships / security_transfer | 2023-01-31 | 8 | -12.6254% | +1.7609% | -14.3863 |
| 63 / technical_relationships / security_transfer | 2023-03-31 | 8 | +7.8821% | +8.9466% | -1.0645 |
| 63 / technical_relationships / security_transfer | 2023-07-31 | 8 | -23.4896% | -9.6449% | -13.8446 |
| 63 / technical_relationships / security_transfer | 2024-01-31 | 8 | -1.6486% | +3.5728% | -5.2214 |
| 63 / technical_relationships / security_transfer | 2024-02-29 | 8 | -5.3713% | +2.9913% | -8.3626 |
| 63 / technical_relationships / security_transfer | 2024-03-28 | 8 | -6.8591% | +4.2283% | -11.0874 |
| 63 / technical_relationships / security_transfer | 2024-04-30 | 8 | -2.2398% | +10.2096% | -12.4495 |
| 63 / within_date_return_rank / security_transfer | 2022-03-31 | 8 | -24.1287% | -15.5508% | -8.5779 |
| 63 / within_date_return_rank / security_transfer | 2022-08-31 | 8 | +3.7171% | +4.1707% | -0.4536 |
| 63 / within_date_return_rank / security_transfer | 2022-09-30 | 8 | +4.2994% | +6.4062% | -2.1068 |
| 63 / within_date_return_rank / security_transfer | 2022-12-30 | 8 | +22.6082% | +7.3071% | +15.3011 |
| 63 / within_date_return_rank / security_transfer | 2023-04-28 | 8 | +21.9878% | +10.5804% | +11.4074 |
| 63 / within_date_return_rank / security_transfer | 2023-07-31 | 8 | -16.9963% | -9.6449% | -7.3514 |
| 63 / within_date_return_rank / security_transfer | 2023-10-31 | 8 | +8.6016% | +17.1532% | -8.5516 |
| 63 / within_date_return_rank / security_transfer | 2024-01-31 | 8 | +2.6687% | +3.5728% | -0.9041 |
| 63 / within_date_return_rank / security_transfer | 2024-02-29 | 8 | -3.8587% | +2.9913% | -6.8500 |

The new annual within-date ranking candidate has 3/29 complete held-out-stock dates:
June 30, 2022: stock +21.8864%, SPY +19.5858%, difference +2.3006 percentage points;
September 30, 2022: stock +11.3324%, SPY +18.5607%, difference -7.2283 points;
May 28, 2024: stock +6.3784%, SPY +13.4827%, difference -7.1042 points.
Its quarterly view has 9/29 complete held-out-stock dates. Neither horizon has a
complete temporal date. These partial results do not establish a full-period SPY
advantage. Repeated/overlapping holdings and the outcome-availability subset cannot
be treated as independent performance observations or used to choose a winner.

Remaining gaps below count each distinct decision/horizon key once per reason;
categories overlap and must not be summed into an additional position count.

| Missing fact | Distinct unresolved decision/horizon keys affected |
| --- | ---: |
| dividend_usd_currency_unproven | 378 |
| missing_action_economic_date | 219 |
| corporate_terms_require_replay | 198 |
| ordinary_domestic_cash_entitlement_unproven | 198 |
| unbounded_entitlement_or_due_bill_dates | 120 |
| missing_or_cross_segment_raw_binding | 109 |
| raw_security_symbol_or_role_differs | 11 |
| TWTR completion 8-K Item 2.01; effective October 27, not provider October 28 | 8 |
| discovery_warner_completion Item 2.01 and WBD listing transition | 6 |
| Fortune Brands Items 1.01/2.01 completed distribution; 5.03 FBIN December 15 | 4 |
| unsupported_action_participant_field | 1 |

Concrete next source work: corroborate dividend currency (378 affected keys), prove
missing corporate-action effective dates (219), model verified merger/transition
terms (198), establish ordinary cash entitlement (198), establish due-bill windows
(120), and resolve raw segment/identity bindings (109 plus 11 identity conflicts).
The exact source requests must follow the existing immutable pair reason records.
TWTR cash settlement, Discovery/WBD and Fortune Brands successor/distribution paths
remain unproven where listed. Fresh evidence must prove exact class, date, cash and
successor terms; provider ticker labels alone cannot settle them. This checkpoint
did not authorize excluding these selected holdings or changing training features.

Proposed next evidence priorities, based only on the number of fixed cohorts with
one remaining unknown position (not forecast returns): 87 cohorts have exactly one
gap. SLB (8), MDT (8), APA (7), EFX (7) and MAS (6) account for 36 of these. Counts
are conditional opportunities, not newly recovered outcomes or a promise of success.
The Sol engineer's bounded retained-source inspection found:

- SLB: prove exact common-share gross USD distributions and foreign/withholding
  entitlement on the affected dates. Retained provider records are marked foreign
  for CUSIP 806857108, and the selected tax archives contain no SLB units. The
  domestic-only adapter refusing these events is not a demonstrated parser defect.
- MDT: establish the foreign class (CUSIP G5960L103) and independently declared gross
  USD entitlement, especially June 28, 2024 and March 23, 2023. Do not treat provider
  payment amounts as proven gross dividends. No selected MDT tax/identity unit exists.
- APA/CPE: obtain official merger and continuing-share-class evidence for April 1,
  2024. The retained provider page distinguishes CPE acquiree and APA acquirer with
  ratio 1 to 1.0425; independent exact-class proof is still required before treating
  the selected APA lot as unaffected. This is a role-binding hypothesis, not a waiver.
- EFX: obtain exact 2022 gross-USD event proof for February 18, May 24, August 17,
  September 2 and November 22. Verified tax coverage is 2023-2025 for ISIN
  US2944291051; later-year facts cannot certify the missing 2022 distributions.
- MAS: resolve the official class/identifier lookup for ISIN US5745991068 and dated
  gross distributions, including November 9, 2022 and February 22, 2023. Retained
  2020-2025 lookup results report unavailable data; no tax PAYMENT bytes exist to
  reinterpret. Use a working official lookup or independent issuer evidence.

SEE affects another eight single-gap cohorts but needs both raw ownership/segment
proof and multiple currency events; it is a compound recovery rather than one fact.
Any subsequent source/parser amendment requires a bounded frozen scope and evidence
review. These proposals do not admit any new source or change a selected position.

Remaining next work: resolve only the reported source/identity/action gaps with
provable dated evidence, preserve these same ordered selections, and rerun a new
immutable outcome overlay if new facts become available. Do not claim a full-period
SPY edge from the available-date subset or use missing positions as exclusions.
No additional model/API checkpoint is active. Exact pair gaps and every known
cohort return are in the pinned outcomes.json and comparisons.json beside the report.

### Within-date investment ranking November 11

| Component | Retained source/time/value evidence | Ordered consumer | Train/evaluate | Live/API |
| --- | --- | --- | --- | --- |
| 209 technical/news inputs | Verified admitted values and clocks, unchanged | Verified shared encoder/default regressions and independent exact 418-input replay | Verified 12 actual fits and independent saved-model replay | Not applicable: retrospective ranking only |
| Within-date rank target | Verified original net SPY excess and maturity | Verified training-only rank/tie/singleton/alignment and publisher poison checks | Verified exact train-only target hashes and independent model replay | Not applicable: no new live labels |
| Unitless ranking score | Exact full eligible roster and original masks | Verified dedicated unitless score/schema, label-only date support and unknown selections | Verified all 48 unit/16 aggregate matched rankings and retained incomplete selections | Not applicable: no return/API claim |

Measured experiment complete; the SPY investment-performance goal is not yet proven.
Implementation bc8d59394d63ae7ec45aa5cca1b66e3a07121b71 is committed/pushed on main.
The immutable export contains 470 source files; export SHA256
411f32a7ce14363e0ca37621c0e62b70d4734196657120f8d1bedd1b15068a14.
Actual retained-data execution started 2026-11-10T23:20:01.710233+00:00 and
completed 2026-11-10T23:22:26.605845+00:00. Exactly 12 new validation
models and 36 saved controls produced 48 unit and 16 aggregate comparisons.
No source rebuild, download, exclusion, old/final refit, tuning or API change occurred.

Report: data/research/investment_ranking_20261111/ranking_bc8d593/_manifest.json
SHA256 47ffb2985416aea654c8c226f9f6215c94254aaf888dda1273ae1445d1e6079a.
Request: data/research/investment_ranking_20261111/inputs/request.json
SHA256 d283a93a0f36506618f5f339db5e67fa429204205480b5c5abbf93784b35d9e0.
Independent senior Sol Quant audit: chat outputs/investment-ranking-independent-audit-20261111.json
SHA256 73514bdef15423c2d80e52c6bc19c1db234979735f532c0c55cab5c63a4bbfa8; completed 2026-11-10T23:24:05.456098+00:00.
The audit independently reconstructs training-only target vectors, 209 medians and
418 encoded inputs, replays every new saved model with zero score residual, rechecks
36 previously audited controls, and reproduces all 48/16 metrics and frozen gates.
All 2,606 independently checked file hashes remain unchanged. Actual source and fit
clocks remain in 2026; inspected 2022-2024 periods are retrospective development
evidence, not an untouched test or contemporaneously deployed prediction record.

All 89 distinct focused tests pass. The initial eight-file suite passed 87 and found
one shared synthetic temporary-directory collision. Separate fixture folders and
the targeted three-case rerun passed, including the new exact-consumed-target poison
test. Ruff on 11 changed files, strict mypy on 9 sources, whitespace checks, real
44-pin metadata smoke, independent design/code review and actual replay all passed.
Component tier only: no release, full-suite or serving checks were required.
The shared lease and strictly-below-90% RAM policy remained active. Both actual workers exited successfully, the lease is released, and the final process check found no Python workers; system RAM was 71.6%.

| Horizon and scope | Technical rank / AUC | Technical + news rank / AUC | Full rank / AUC | New rank target rank / AUC | All-control gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 252 sessions, temporal | 0.073307 / 0.555428 | 0.077703 / 0.558481 | 0.066465 / 0.549641 | 0.100744 / 0.561533 | Pass |
| 252 sessions, security_transfer | 0.017467 / 0.524225 | 0.003205 / 0.522270 | 0.002506 / 0.518687 | 0.052629 / 0.542884 | Pass |
| 63 sessions, temporal | 0.035672 / 0.515615 | 0.034117 / 0.514495 | 0.028530 / 0.511070 | 0.053825 / 0.520458 | Pass |
| 63 sessions, security_transfer | -0.009452 / 0.481836 | -0.029501 / 0.469399 | -0.012923 / 0.485104 | 0.001232 / 0.493764 | Fail |

Rank means equal-date Spearman correlation; AUC distinguishes strictly positive net
excess versus SPY. Each aggregate uses the same 29 supported dates and known rows
across all four profiles; none has a constant-score supported date. Annual temporal
uses 7,126 known/3,965 unknown among 11,091 scored decision rows; held-out-stock
transfer uses 1,285 known/815 unknown among 2,100 scored rows. Quarterly counts are
7,862 known/3,229 unknown temporal and 1,425 known/675 unknown transfer. These are
monthly stock decisions, not distinct stocks or independent investment outcomes.

The frozen annual ranking gate passes in both scopes against all three controls and
neutral discrimination. Relative to the matched 209-feature model, annual temporal
Spearman/AUC improve from 0.077703/0.558481 to 0.100744/0.561533, and held-out stocks
from 0.003205/0.522270 to 0.052629/0.542884. This supports the target-only ranking
hypothesis on this development set, not an explanation that more features are better.

Year-level limits are material. Annual temporal 2023 AUC falls from 0.545863 to
0.531063 despite slightly better rank correlation. Annual held-out 2023 correlation
remains negative (-0.047269); held-out 2024 AUC falls from 0.590674 to 0.586450.
Annual year gates pass temporal 2022/2024 and transfer 2022 only. Quarterly temporal
passes the aggregate gate; quarterly held-out AUC is 0.493764, below 0.5, so its
aggregate gate fails. Quarterly year gates pass temporal 2023/2024 and transfer
2024 only. 2024 covers January-May, not a full year. No confidence/significance claim
is made from 29 overlapping dates, and no model is automatically selected/promoted.
Pooling scores across dates produces lower annual AUC than the matched 209 control:
0.553272 versus 0.562346 temporal and 0.533639 versus 0.536523 held-out. The observed
gain concerns ordering stocks on the same date; it does not establish improved
comparison of score levels across dates or calibrated SPY-beating probabilities.

Economic coverage remains insufficient. The annual candidate selects 1,117 stock/date
positions across 29 temporal cohorts, with 431 unverified selected outcomes and zero
complete cohorts. Held-out annual selects 232 positions with 72 unverified outcomes
and only 1/29 complete cohorts. Quarterly has 275/1,117 unverified temporal selections
(0/29 complete) and 55/232 held-out selections (2/29 complete). All selected identities
remain; unavailable outcomes are not replaced with zero or removed. Full-period
selected-cohort excess returns remain null in every view. Even complete overlapping
cohort means would not establish a funded portfolio's SPY outperformance.
The only complete annual held-out candidate cohort, September 30, 2022 (eight
selected positions), has mean stock net return 11.3324% versus SPY gross 18.5607%,
or -7.2283 percentage points net excess. This is an unfavorable isolated cohort,
not an estimate for the 28 incomplete cohorts or a funded portfolio result.

Next prerequisite for a SPY-return claim: resolve the existing selected-outcome source
gaps across the fixed candidate/control rosters, then freeze a cost-aware funded SPY
comparison and an untouched/prospective evaluation. Do not tune this candidate on
these observed results or promote it based only on improved AUC. Preserve this
ranking experiment as development evidence. V1/beta remains deferred.

Verification logs are in chat outputs, rooted at
C:/Users/manis/Documents/Codex/2026-11-05/c-project-market-predictor/outputs:
- investment-ranking-component-checks-20261111.log: SHA256 2c3801eda746540b99252633184ef490059dcbde996b3c98acaef5cae32ed530.
- investment-ranking-publication-checks-20261111.log: SHA256 61e77bc6edff6b630a931eb9f630177ebe3770147b777b655911fa7affaba8a0.
- investment-ranking-final-static-smoke-20261111.log: SHA256 d85cbffbd08fc08af656b9da4734f676346320acc80231009760e2fd18c4bbf4.
- investment-ranking-actual-run-20261111.log: SHA256 ec5244b4f269726cbce15dbb398aecc15609c0ccbfde295c4cd694ae07e28b6b.

Independent audit execution log: investment-ranking-independent-audit-20261111a.log, SHA256 9b9c2029394598ce4bca3d89fd72140c4e0158fecda63b8215649071512fabd3.

Verification commands: guarded pytest -q on test_investment_ranking.py,
test_investment_ranking_publication.py, test_investment_training.py,
test_investment_diagnostics.py, test_investment_diagnostics_publication.py,
test_investment_feature_ablation_publication.py,
test_investment_forecast_adjustment_publication.py and test_investment_research.py;
targeted rerun of the two publication integrations and exact-target poison case.
Ruff covered the 11 changed Python files; strict mypy --follow-imports=silent
covered all 9 changed sources. Actual command: train-investment-ranking with
the exact request above; independent helper audit_investment_stock_selection.py
SHA256 0fff64758cc8dbcd6d2e14305a4d090e563660ec7579c9db615d8f3ccaa67ec1.

### Forecast adjustment and financial handling November 10

The source/data/availability contract remains the accepted 215-column dataset.
Only the model's six financial-column encoding and a score-only batch rule change.

| Component | Source and historical values | Ordered consumer | Training/evaluation | Live/API |
| --- | --- | --- | --- | --- |
| 209 technical/news columns | Verified retained values/clocks, unchanged | Verified exact original 418 value/indicator parity | Verified original encoding, default-model regression and actual retained-data consumers | Not applicable: retrospective comparison |
| Six clipped/native-missing financial columns | Verified retained values/clocks, raw NaNs unchanged | Verified train-only bounds, empty/constant/missing state and 424-column order | Verified 12 actual fits and independent preprocessing/model replay | Not applicable: no live transformation release |
| Centered and halved forecasts | Existing saved or new verified raw forecasts | Verified full eligible roster, outcome independence and float64 centering | Verified 36 actual derived artifacts, unchanged rankings/cohorts and independent metrics | Not applicable: requires full batch, no API work |

No fitting begins until these consumer checks and independent design/code review
pass. No source or original artifact is rewritten. The planned zero forecast mean
is over all feature-eligible stocks, including unknown outcomes, not the known subset.

Implementation 0de4474169f2fc3cf1552e99c16afdad7d2cefa0 is committed and pushed
on canonical main; its immutable export contains 468 source files. The actual run
completed at 2026-11-10T22:52:48.004934+00:00. Exactly 12 new XGBoost validation models
were fitted and 36 old validation units reused. Published 36 adjusted forecast
artifacts, 84 unit diagnostics and 28 aggregate diagnostics. No final model,
source rebuild, download, stock exclusion, selection or serving change occurred.

The 139 distinct focused tests passed after correcting missing training-lineage
fields in a synthetic reference fixture (138 passed initially; all 8 affected
publication cases passed the targeted rerun). The production checks were unchanged.
Affected Ruff (10 Python files), strict mypy (6 sources), whitespace checks and
independent consolidated Sol code review passed. Actual request/experiment/settings
and 36 saved unit manifests passed a metadata-only smoke check against 44 real pins.
Component tier: affected estimator, adjustment, publication, shared reader, training
and research tests plus actual data execution and independent replay. The repository
full suite and API checks were not run because this is not a release or promotion.

Actual report: data/research/investment_forecast_adjustment_20261110/adjustment_0de4474/_manifest.json
SHA256 24c1bfb28e35f522cdefbc62435074f80c030dadef44db61b04c989582f506b1.
Request SHA256 f95f255470455201b3c11de11a3e964449a55ec9362e9314f3dee23b078af40e.
Independent senior Quant audit: chat outputs/investment-forecast-adjustment-independent-audit-20261110.json
SHA256 6cf4bdc77c57363d6783b23bafdad12f4bf2a5394ad7acc015c35bf836cae968; completed 2026-11-10T22:55:14.097463+00:00.
All 12 new models replay exactly, with independently reconstructed 209 train-only
medians, six finite-training clipping bounds and the ordered 424-column encoding.
All 36 adjusted forecasts, within-date ranks/ties/cohorts, 84 unit and 28 aggregate
diagnostics and frozen improvement checks pass. Maximum raw forecast replay error
is zero; maximum independently computed adjustment difference is
0. All 2,642 independently
checked file hashes remain unchanged before/after. Original audit/model/source evidence is reused.

Verification logs (chat outputs): component-checks-20261110 SHA256
d00ce1649e0edd271a6cc8c9d3707f80b92e1aa3ff8aa3fa90fdb7805754677f;
publication-checks-20261110 SHA256
22f9d199b4ab09e431fefa6c90bd474006fe9ecd3754b90b42fd9bebbccdd1803;
final-static-smoke-20261110b SHA256
20402a67d4bac0fccabbc3f5aa688f81673099ea40d78f570b5a14a3797fc19f.
Each filename has the prefix investment-forecast-adjustment- and suffix .log.

The new financial encoding and forecast adjustment have completed their retrospective
source-to-model-to-evaluation checks. This verifies consumer behavior, not live/API
acceptance or investment profitability. The leading active plan records the measured
candidate comparisons and their limitations. Original data and admission flags remain.

### News and financial comparison views November 10

The user authorizes two projections of the admitted215-column matrix. No new source,
value, timestamp or missingness calculation is introduced. Preserve original2-profile
dataset contract; publish new ordered view names in the separate experiment only.

| View | Source and historical values | Ordered feature consumer | Train and evaluate | Live and API |
| --- | --- | --- | --- | --- |
| technical_news209 | Verified retained124technical plus85news values and original availability clocks | Verified exact ordered projection and common-row tests | Verified actual fits and independent saved-model/metric replay; forecast quality fails the improvement checks | Not applicable retrospective research only |
| technical_financial130 | Verified retained124technical plus6financial values and original availability clocks | Verified exact ordered projection and common-row tests | Verified actual fits and independent saved-model/metric replay; forecast quality fails the improvement checks | Not applicable retrospective research only |

Old124/215models are immutable references. Source/causal/value evidence in the closed
matrix below is reused. Missing optional features and unknown outcomes are retained;
no added source is inferred from a finite zero. No new model training begins until
projection/consumer checks and independent design/code review pass.


Implementation 433d6d43e32b6448c52257dc717f24e6a37c0b07 is committed and pushed on
canonical main; the immutable export contains 466 files. The actual run completed
at 2026-11-10T22:20:31 UTC, with 48 new validation fits, 48 reference fits reused,
96 unit reports, 32 aggregate reports, 24 fold comparisons and 8 aggregate comparisons.
No final models, data downloads, source rebuilds or serving changes were performed.
The 75 focused tests, affected Ruff and strict mypy, independent Sol design/code
review and senior Quant statistical review passed.

Report: data/research/investment_feature_ablation_20261110/ablation_433d6d4/_manifest.json
SHA256 400661d2945bc28ff2bbef6e3dd2d74eb34c1e7566b506fe37624e2c7c2cf31e.
Request SHA256 c84435c641682092a84d8778dda7abe11618c05ee714a1d11f40cd8c9ecd4266.
Independent audit: chat outputs/investment-feature-ablation-independent-audit-20261110.json
SHA256 8eb6404219dc0a9d6e1d777224d0f96c3739403f80290a173df2c463d1d0ade8.
Audit completed at 2026-11-10T22:23:29.155289+00:00; all 48 new saved models replay
exactly (maximum prediction difference zero), all 96 unit and 32 aggregate metrics,
24 fold/8 aggregate conditional comparisons, source hashes and actual clocks pass.
All 2,745 independently checked file digests were unchanged before and after the audit.
Original 48-unit audit and model bytes were reused unchanged. No model was refitted
by the audit. All source admission and promotion flags retain their original meaning.

The four-view comparison is complete. Annual XGBoost deterioration comes mainly
from the current financial block; Ridge errors increase with either group. Every
aggregate still exceeds zero-excess MSE and MAE, so feature-consumer verification
does not establish forecast quality or live/API acceptance. See the leading plan
for all measured comparisons and retained unknown-outcome counts.

### Investment research feature acceptance - November 10

The actual dataset and independent pre-fit audit passed. Implementation
30b54bfd28b9a01de4bb9abcf1dba82661966cef joins retained predictors and targets;
no source calculations, values or original availability clocks were changed.
All 74,458 rows remain. The independent audit completed on November 10 at
19:49:23 UTC and permits the frozen retrospective experiment in all 12 splits.

| Feature group | Source/time contract | Historical construction | Ordered model contract | Training/evaluation consumer | Live/serving/API |
| --- | --- | --- | --- | --- | --- |
| 124 candle technical/relationship measures | Verified exact previous-XNYS-session source identity and clocks before investment close; original 586-security peer values | Verified exact retained values; no recalculation | Verified original 124-column prefix | Verified actual row/value/clock parity, common eligibility and independent splits | Not applicable: retrospective research only |
| 85 news measures | Verified original saved clocks and explicit source/missingness semantics | Verified retained values; no window reconstruction | Verified full original 209-column order | Verified actual values/nulls/reasons and source pins | Not applicable: no live transformation or serving release |
| 6 financial measures | Verified exact security/decision/close and retained SEC availability clocks | Verified 134,076 original metric values across 22,346 financial decisions | Verified appended FEATURE_NAMES order | Verified actual identity/clock/value/missingness joins | Not applicable: historical filing reconstruction is not admitted live input |

Audit: chat outputs/investment-dataset-independent-prefit-audit-20261110.json,
SHA256 bdcf62b5a7fdacbdbf854fb5f0d6adb36390eab5b2e3212fbb9c4b4f3efbda46.
Reviewed execution helper: 9e15dd22dc21d3c2065b2da30dbcb87db7f09385ab5f8dd6eff6104652eb4c1e.
The audit checked 5,874,781 saved feature values and the same number of availability
values, all 27,189 verified target bindings, 46 original news-producer files, and
2,430 consumed file hashes before/after. All 12 frozen purged/embargoed temporal
and security-transfer splits are available. Original source admission flags remain
unchanged; the separate derived admission permits retrospective research fitting only.

Per horizon, 22,334 rows have usable features. Verified target plus feature rows:
12,834 at 252 sessions and 14,341 at 63 sessions. Missing optional values retain
nulls/reasons and use training-partition-only preprocessing. Prediction eligibility
does not depend on future outcome availability; selected unknown outcomes cannot be
removed to improve measured returns. Component, poison and publication tests passed;
latest representation regression has 36 passing tests plus affected Ruff/strict mypy.

Training and independent saved-model replay completed: 48 validation fits and 8 final
research models. Training-only preprocessing, all saved predictions, split/clock/source
bindings and independent metrics passed. All 16 aggregate MSEs exceed zero-SPY-excess
baselines; news/financial inputs worsen seven of eight paired MSE comparisons.
The completed feature consumer is verified for retrospective research; measured model
quality does not support production activation. No prospective performance, promotion,
serving or complete selected-cohort return is claimed.
Post-fit audit SHA256 b139c5e84d98c332aec3598c4751dedb35b5d4d8f2b414630604839cd94404dd.

Saved-prediction diagnostic completed November10 at20:43:44UTC using implementation
770d0db. Report SHA2562e570a4b714e0305d11909fe23ee48eeb116114be42ada16331070a92ce86981.
All48validationunits and16aggregates reproduce canonical metrics exactly, with no
source/feature/forecast changes. Independent Quant confirms material positive annual
forecast bias, modest within-date ordering and weaker security transfer. The combined
news/financial comparison does not identify separate source effects. At that diagnostic checkpoint, the
209news/130financial comparison remained proposed. It is now completed and independently
verified in the leading section; the original diagnostic evidence remains unchanged.
Historical checkpoint evidence below predates the completed training result above;
earlier references to pending fitting describe those checkpoints at their original dates.

### Full investment targets built and independently checked - November 10

The full corpus contains exactly 74,458 horizon rows for 37,229 monthly decisions.
Of 44,692 in-scope requests, 27,189 are source-validated (14,348 at 63 sessions and
12,841 at 252 sessions); 17,503 retain explicit source/comparison gaps. Another
29,766 rows are outside scope. The 606 original checkpoints/71,508 rows were retained
unchanged with explicit producer/job lineage; 25 groups/2,950 rows were newly built.
EPAM's duplicate SPY sector comparison remains unavailable for 60 eligible rows.

Independent Senior Sol Quant audit: audit_passed_with_explicit_unavailable_strata.
All 74,458 row identities/counts and 631 origin records checked; all 71,508 inherited
rows retain their original checkpoint bytes, job and producer. All 57 prior target
examples match numerically and their 70 files remain unchanged. Cleanup comparison
passes all 3,540 original rows against their actual 0038663 origin, with original
capture clocks preserved. Independent raw four-holding arithmetic covers
5 metadata-selected rows. The original first-12 lists had no usable
case for these five strata: stock_split:63, stock_split:252, stock_ordinary_cash:63, stock_ordinary_cash:252, entry_exdate:252. That primary result remains unchanged.
Audit: chat outputs/investment-full-lineage-independent-audit-0a1ca7d-20261110.json
SHA256 7b9ffe57a5b97c27b01c26c6a319eec39ffe39723dc3f0e2d4260387cb590cbe.
Helper: outputs/audit_lineage_full_investment_targets.py SHA256 3bfd877ecba3077bc4874a119b0b4c3dbc6719dfbf1a2e40fd44d6f2720f3c94.

The bounded supplemental audit covers all five originally unavailable strata using
5 additional distinct rows and
20 independent holding calculations. It selects
the first source-validated candidate in each frozen metadata order after the original
12-candidate prefix, never by return. Consumed source hashes were checked before and
after arithmetic. No supplemental strata remain unavailable. This is a sampled
arithmetic check, not a second numerical computation of every corpus row.
Additional cases: COO November 30, 2023 (63-session split), COO February 28, 2023
(252-session split), MCD November 29, 2019 (63-session ordinary cash), CI July 31,
2019 (252-session ordinary cash), and SYY December 31, 2019 (252-session entry ex-date).
These are decision dates. The audit inspected 314 checkpoint status maps and selected
the first usable candidate after 73/300/1161/229/8 checks in those respective orders.
Supplemental report: chat outputs/investment-supplemental-arithmetic-audit-20261110.json
SHA256 28e6c89b45af7fa40d42f58fac03c3ac17557d1c6fbdc250714862d0a621bf1c.
Frozen reserve: outputs/investment-supplemental-metadata-reserve-20261110.json SHA256 10674bcc1bd1570e9e10c2ee270ec67838a3375628a9168bca70faeb06061687.
Execution helper: outputs/audit_investment_reserve_arithmetic_execution.py SHA256 0075d48a69684aaeaf8539c190e76d1c8e1689852bdc5fffc3283f4076c6ce73.

Final corpus SHA256 9b70c348ab17738f5874d42c51bff6af386d94061ef383385bdd6ead5bcad4c3; implementation
0a1ca7daf924e900ac0947e6169678e93d27c41e is on main. Exact artifact/input paths,
remaining source causes and verification evidence are in the current handoff.
The 257 benchmark cash assertions and 4,241/6,717 supported stock events remain
verified, with 2,476 explicit stock-event gaps and no new amount corrections.
Of 435 stock acquisition units, 350 have at least one admitted session and 98 are
partly or wholly quarantined (80 observed); 13 units appear in both groups.
These artifact/event counts do not denote target rows or whole-stock exclusions.

Financial feature coverage remains unchanged: of 22,346 eligible decisions, 6,017
have all six recovered financial measures, 15,485 have at least one, and 6,861 have
none. Missing financial features did not remove target requests. The frozen earlier
evaluation cutoff has zero currently-available rows because verification occurred
later; all 27,189 validated targets are available by actual publication. Clocks are
not backdated. All training/promotion/production flags remain false.
Training-dataset/feature acceptance and model fitting remain separate pending work.

### QQQ event resolved and raw-source coverage measured — November 9

The original benchmark inventory now has 213/213 supported events. Actual QQQ
special-distribution accounting passed on one metadata-selected 63-session target;
the annual row retains explicit gaps. Prior 56 examples are unchanged (57 validated
integration examples in total). The raw acquisition captured 106,972
of 111,661 requested sessions, with 4,689 explicit gaps.
All eleven benchmarks have complete new raw coverage through May 30, 2025 and
passed continuous old/new-source reads with capture-clock parity. All new stock
captures remain quarantined outside usable selection. Financial feature counts
below are unchanged. Broad targets, dataset admission, investment fitting and
prediction activation remain pending; all eligibility flags are false.
See the leading current handoff for exact reports, hashes and next steps.


### Fifty-six targets and broader benchmark sources verified — November 9

XLY adds 24 source-validated targets to the preserved 32 XLV targets, covering
28 selected July 2019 stock-months across two horizons. They are integration
evidence, not an accepted investment training corpus or model performance.
The official benchmark archive now covers all 77 fund-years; exact event replay
supports 212/213 required benchmark dividends with
1 explicit gaps. Prior evidence remains unchanged.
All training/promotion/production flags stay false. Raw-price, stock-action and
financial-feature gaps remain; see the current handoff for exact evidence and
next acquisition scope. Source/HTTP/CLI checks and independent Sol Quant/code
reviews passed. The existing full financial coverage counts below are unchanged.



### Thirty-two actual investment targets verified — November 9

Implementation 13b5e6f adds explicit issuer amount correction with preserved raw
provider evidence. The sixteen frozen July 2019 Health Care decisions produced
32/32 valid historical targets (16 per 63/252-session horizon), zero unresolved,
with mandatory stock/SPY/QQQ/sector accounting. Old IDXX 63 numerics match exactly;
the annual SPY effect matches only the documented per-share dividend correction.
Final manifest SHA256 26f58c8d3921b102f8a096e8d1fb11de60350204ad37a9ff8263f11c2ca3fa03.
Actual audit SHA256 fd8d9c377312994d0a704d6268115010b3d8c5e30ce0d8d2d2d7ded967890bda.
All 208 focused tests, Ruff, strict typing and independent Sol code/Quant reviews
passed. Four of 16 decisions have all six financial features, 11 at least one,
five none. This selected source-coverage batch is not model performance evidence;
training/promotion/production remain false. Full feature counts below are unchanged.
See the current handoff for evidence paths, denominator and the next source work.


### First actual investment target pilot — November 8

The independent dividend-currency adapter and explicit selected-decision materializer
are verified under implementation 88f6327. Actual IDXX July 31, 2019 yields one
source-validated 63-session target; 252 sessions remains unavailable because SPY
June 2020 provider/issuer dividend amounts differ. Full 37,229-row roster verification
preceded selecting this one decision; no model fit or population coverage is claimed.
Final pilot manifest SHA256 3286a1456a83bd9fcc2076cdf33c203379bf0687b18e356eae7fb83b648d5a3f.
Twelve currency events passed real-source replay; 170 targeted tests, Ruff, strict
mypy and independent code/Quant reviews passed. Historical feature/target data remain
research-only: training, promotion and serving are still pending. Existing six-feature
coverage counts below are unchanged. See the active handoff for exact source pins,
the explicit amount conflict and the next bounded step.


### Broad financial coverage verified; investment labels pending — November 7

Implementation `0d01e13` fixes whole-issuer rejection caused by isolated historical
SEC fact conflicts. All 431 eligible issuers produced 22,346 monthly snapshots;
zero issuer failures remain. The prior 342 successful issuers / 17,770 snapshots
match exactly, excluding the expected feature-contract hash change. Final manifest
SHA256 is ddb5021e1daba62d9fde8fc5fc9bec7bb8170c46889f9455f5b6cd341aaadf4d at
`data/research/investment_financial_recovery_20261107/final/_manifest.json`.
All raw sources were reused; old producer/artifact bytes remain preserved.
147 focused tests, Ruff, strict mypy and independent design/Quant/code reviews pass.
Quant also checked actual manifest, coverage/parity hashes and interpretation.

Availability denominator: 22,346 eligible security-months. All six metrics exist on
6,017 rows (26.93%), at least one on 15,485 (69.30%), and none on 6,861. Source
completion must not be reported as complete feature coverage or training admission.

| Feature | Available security-months |
| --- | ---: |
| Revenue growth TTM | 13,301 |
| Operating margin TTM | 13,192 |
| Operating margin change | 10,872 |
| FCF margin TTM | 10,443 |
| CapEx / revenue TTM | 10,976 |
| Cash conversion TTM | 10,877 |

The 4,712 rejected security-bound filing versions (4,682 alias conflicts, 30 invalid
facts) span retained source history, not excluded stocks or decision rows. Causal
invalidity blocks exact inputs and derived intervals without stale fallback;
independent valid periods survive. Unknown-period issues conservatively block only
their concept. Future issues do not alter earlier snapshots. No new exclusions.

Source/feature research construction is verified across the declared population,
with explicit missingness. Historical capture remains reconstructed research.
Actual investment future-return targets, training, promotion and serving remain
pending. The November 7 memory amendment replaces the former 85% / 2 GiB rule
with a 90% ceiling and no implicit free-memory floor. No actual source-validated
target count, model fit or predictive improvement is claimed. Source validation
means required price, benchmark and action data passed checks; it is not model
approval. The older matrix below records the initial feature-kernel checkpoint.

### Investment fundamentals research checkpoint — November 6

User approved the revised long-term feature/evaluation specification. This adds a
separate research path; earlier swing source families and model columns are frozen.

| Layer | Status | Evidence / remaining requirement |
| --- | --- | --- |
| SEC source collection | verified for one real issuer | Microsoft CompanyFacts immutable response and decoded payload; universe-wide coverage not established |
| Historical availability | verified reconstruction, observed training blocked | Existing corrected SEC accession clocks; later retrieval remains distinct from historical publication |
| Historical/snapshot feature construction | verified component | Shared pure kernel and pinned batch publication; actual Microsoft historical decisions |
| Ordered model input contract | verified research contract | Six named metrics, definitions hash, constituent fact hashes, explicit null reasons |
| Training/validation consumption | blocked | No admitted complete investment training/label corpus or fitted investment candidate |
| Saved forecast evaluation | verified software only | Canonical target binding, pairing, strict fit/maturity clocks and date-aggregated metrics; no actual investment forecast metrics claimed |
| Promotion/serving/API activation | blocked | No investment model approval or qualified prospective evidence; flags remain false |
| Future poison, parity and tamper | verified unit scope | Focused feature, SEC adapter, evaluator and publication tests; fixture evidence distinct from real SEC replay |

Core features are revenue growth, operating margin/change, FCF margin, CapEx to
revenue and cash conversion. Total-debt leverage and market-cap yields are deferred
until all necessary debt/share components are supported. DCF and specialised AI
forecasts are deferred. Annual/YTD arithmetic assumes compatible accounting across
filings; exact date matching can leave fiscal-calendar edge cases unavailable.
This checkpoint does not establish improved returns, sufficiently trained models,
historically observed source collection or full financial statement coverage.


### August saved; continuation active after transient host pressure — October19

Ready-run PID104640 saved August2022 at12:08:41Z:38/59months and376,002/586,305
decisions. Host RAM rose from about73% to91.45%; pause12:12:57.594807Z,120-second
timeout at93.42%, exit1 at12:15:40.5952156Z. Its measured resident memory was556MiB
(private1382MiB), so the host-wide pressure must be distinguished from its own
resident usage. Preserve news-monthly-full-sentiment-ready.* receipts/logs.

Actual idle RAM then recovered to72%. Fresh unchanged canonical continuation
native61858/PID113380 started12:17:36Z (14:17Berlin), using saved checkpoint
3354fcd8e63f80020c79f597c4caf0c68e26f763ebfbd071e6ea7d353a2b0356.
It verifies/reuses38months and builds only missing21. Current workspace operator
work/continue_news_monthly_sentiment.py SHAba72d58ca5ce527b868ccbfad511fb1d6d9d61dfa28442a671696e3c488127c7;
only two fresh receipt-path strings differ from the reviewed resumer. AST/Ruff
passed. State/log/exit: news-monthly-full-sentiment-continue.*. Source/config/
columns/targets/four baseline fits and91% ceiling remain unchanged.

Monitor67350/PID53520 still provides current owner/RAM and sleep hold; its old
monthly-phase field refers to the preserved failed ready-run. Read the continue
state for actual feature progress. Finish via fresh news-ready-finished.json and
verify monitor release at actual work end; old finished receipts remain intact.

Prepared workspace work/freeze_news_prefit_features.py SHA
8faa1777ce88c86fb0278d938c954ef706dee0a743dc77a4ec10886ae8397d44 passed AST/Ruff
and one bounded code review, no open findings. It awaits actual final publication,
replay, readiness, fit-scope and training-config pins; records ordered209-column
missingness/clock/source/sentiment diagnostics and appends the actual prefit audit
freeze. No data job or fit has been run by this helper. Continue through all
prepared steps and both learners; do not stop at feature publication.

### User resumed with recovered RAM — October19

User instructed start now. Actual prelaunch RAM69.04% used/4.86GiB available;
checkpoint/config/ready-operator hashes matched, no heavy owner or previous ready
attempt existed. Fresh continuation native84873/PID104640 started11:57:29Z
(13:57Berlin), retaining37months/365,008decisions. Source/config/learner bytes and
91% ceiling remain unchanged. It verifies saved months and builds only missing22.
Current state/log/exit: workspace review-results/news-monthly-full-sentiment-ready.*.
Fresh monitor native67350/PID53520 uses work/monitor_news_ready_pipeline.py,
news-ready-monitor-state.json, news-ready-heartbeat.txt and news-ready-finished.json.
Original finished/failed receipts are preserved. Fresh finished receipt must be
written at actual work end and monitor release verified. Keep dependencies frozen;
continue final replay/readiness/fit-scope/feature freeze/two learners/comparisons.
Prior memory-freeing question is satisfied by observed recovered headroom. Training
is not yet started. This is active execution, not a completed-model/SPY/API claim.

### Guarded continuation also stopped; fresh receipt path ready — October19

Continuation native43093/PID95528 started07:01:32Z after idle RAM stayed about88.6%
(below the user91% ceiling). It verified all37saved months using the unchanged
request/input/code pins, then paused07:04:28.487690Z at91.0132% during construction
of the next month. It timed out120.0589seconds later at92.7227%; final closing
recheck error92.66%. It exited1 at2026-10-19T07:10:28.9861997Z, without a new month.
Actual exit receipt news-monthly-full-sentiment-resume.exit.json SHA256
28cd82cdf738ea35463dbe18a8c8e57d2a2f23419763cec49799f7f19f84b01e;
log news-monthly-full-sentiment-resume.log SHA256
9b756e35650f0ffe4853216beb17025cf7e50c37377f2fb671e987615b16295b.
Both project worker PIDs have exited and no heavy owner remains. At07:10:49Z,
idle system RAM was still88.8%. The attempted restart demonstrated that this
idle level does not leave enough allocation headroom under91%; do not repeat it
without freeing memory. User request to free about3GiB remains pending.

Progress is unchanged:37/59months through2022-07,365,008/586,305decisions.
Checkpoint remains ff60ce16591312a4cb68d1ae6a16065cc9a3d46e84359e22170777f1595de0ef.
All original source/corpus/evidence and four baseline fits remain retained.
No final59-month publication/replay or new training/evaluation result exists.

Fresh workspace operator work/resume_news_monthly_sentiment_ready.py SHA256
2bc86849925c3fde839105616bf574a820c04f8c8d00cfa41c3c226077b014c9. Only its two receipt-path strings differ
from the reviewed continuation; syntax/Ruff and exact two-path correspondence
passed. It has NOT run. Its four required external pins are the same checkpoint,
configdea87d23...1eec, Julyparity53377f59...22dde and original failed full receipt
8b8959ec...2d286, with full values retained below. It preserves all attempts and
uses the canonical publisher to verify saved months/build only the missing22.

Authorization to continue through training remains active. Memory, not additional
approval or credentials, is missing. After the user frees RAM, check actual
headroom, run the fresh operator, then final replay/readiness/fit-scope/feature
freeze/two learners and work/run_news_evaluation_campaign.py. No source rebuild,
baseline refit, API version change or TradingFlow edit is needed.
The temporary idle-sleep hold is released using news-overnight-finished.json with
status awaiting_external_memory and training_completed=false; this is not a task
completion claim. Preserve that receipt. If another unattended run needs a sleep
hold, use a fresh monitor/finished-receipt pair rather than deleting prior evidence.

### Actual memory stop after37 durable months — October19

Publisher PID103896/controller native22111 exited1 at2026-10-19T06:56:02Z.
The first pause was06:50:01.459627Z at91.0097% RAM; first120.0635-second timeout
06:52:01.523351Z at94.9622%. Input-context closing rechecks also exceeded91%;
last error93.46%. This was an actual host-memory limit, not missing credentials,
missing source data or a model-quality decision. No new training has started.

All37months through2022-07 remain durable:365,008/586,305decision rows. Exactly
37month directories match the checkpoint inventory; no completed month is removed.
Private checkpoint SHA256
ff60ce16591312a4cb68d1ae6a16065cc9a3d46e84359e22170777f1595de0ef;
request973f339a7091c3536de51690d79a613df106a2e3cd62ed1abb201d4aac2b836d.
Actual failed full-publication exit receipt workspace
review-results/news-step4-sentiment/full_publication.exit.json SHA256
8b8959ecb27585bf2ae96e90c0b22e9126cbc935fb9bfe40e19d7c8b1e92d286.
Its stderr pin32b8260ddd9266532087558831e266245327659acedc2b4ff425c0dd6406c8ae
retains the pause/timeouts and full exception chain. Controller stopped; no heavy
owner remains. At06:56:38Z, host RAM was still91.02% with the project worker gone.

NEW workspace work/resume_news_monthly_sentiment.py passed syntax/Ruff and one
bounded code review. It requires actual failure receipt, stopped controller,
external checkpoint/config hashes and the existing July parity receipt. It calls
the unchanged canonical publisher resume, verifies saved months, computes only
the remaining22months and preserves the original request/code/input pins.
Config remains dea87d23abb7e45f912e1da23db0779c62a4b8561f88f1b6712516102de71eec;
July parity53377f59e077ffe7d1e0996806805a73a6998ddbb0f670763975e251cbf22dde.
No source, runtime, schema, target, baseline model or memory ceiling changed.

An asynchronous request asks the user to free about3GiB and retain that headroom
through feature generation/training. Authorization to continue remains active;
this is not a user-requested pause. Restart after memory has recovered sufficiently
for worker allocations, then complete final replay/readiness/fit-scope/feature
freeze/two learners/comparisons. Never rebuild the completed source or baseline fits.

### Actual monthly progress — October19

At 2026-10-19T05:48:23.223375+00:00, 30/59 months through
2021-12 and 296,113/586,305 retained decision rows are durable.
Controller native22111/PID99620 and publisher PID103896 continue; no new fit yet.
Workspace state: review-results/news-monthly-full-sentiment-state.json;
controller: review-results/news-step4-sentiment/state.json. Partial checkpoint
observation is retained in news-monthly-halfway-20261019.json. This is not a final
59-month publication or replay receipt. Keep source/operator bytes frozen and
continue through replay, readiness, fit-scope proof, feature freeze, both learners
and all comparisons. RAM most recently about83%;91% limit remains unchanged.
Evaluation operator work/run_news_evaluation_campaign.py passed syntax/Ruff and
one bounded code review; it awaits the actual completed news model manifest.
It will preserve all two paired comparisons/twelve funded-policy outcomes,
including explicit unavailable valuations. No evaluation/model acceptance claimed.

### Revised news profile: software connection complete; actual feature evidence pending — October18

Completed-index consumption amendment99a753a (October19): original source
creation is unchanged. Later projection uses the identical original SQLite reader
methods, hashes every consumed index/config/code artifact at entry/exit and retains
all original source hashes separately as producer provenance. It does not claim a
fresh raw-provider replay.61 focused synthetic UNIT cases and affected static checks
passed. This changes no numerical columns, target, saved row or API version; actual
source/monthly/fit evidence is still required below.

The current third slot is technical_relationships_news:124 unchanged parent model
columns plus85 observed source-text cue statistics, not126 verified world-event
columns. Existing six-specification/twelve-policy budget and four fits remain fixed.
Code7843a14 connects exact saved209 rows to existing offline return fitting;
769ed1c connects saved-score comparison and funded evaluation. No new fit or measured
return improvement is claimed by these software changes.

| Layer | Actual status and evidence before fitting |
| --- | --- |
| Existing parent/targets | Retain original124 publication e9ec4b...ef04c, saved-row receipt e10679...1086 and original replay aa8000...cec9; no numerical reconstruction. |
| Source/cue inputs | Completed516,679-version corpus retained; source link index e9f2f209...5da8 completed516,679query copies/439,881company-link rows;11unrepresentable and76,787unassociated copies remain explicit. Original clocks remain historical_proxy; provider-wide coverage/world actors remain unknown. |
| Historical209 rows | Real July pilot found exact-copy saved-score float inequality before any durable month. The bounded per-decision identity/tolerance remedy passed focused UNIT/static/review checks; its real saved-score metadata join is running. Required next: final59-month/586,305-row output and replay receipt remain absent. Actual completion is required before fitting. |
| Model order/clocks | Exact kernel85 suffix and parent124 prefix;209 model clock keys,249 physical mappings preserved. Full replay checks values/nulls/reasons/clocks and parent parity; code verified, actual report pending. |
| Training consumption | Flat monthly reader and explicit offline scope implemented; requires real receipt, contract correspondence, readiness masks/missingness and unchanged actual fit-ID/weight proof before fits. |
| Batch/single construction | Same pure kernel; existing future/revision/copy/missingness UNIT evidence retained. Real July sampled parity and complete monthly replay still pending this run. |
| API/serving/promotion | Not applicable to permission for this explicitly authorized offline experiment. No registry/release output; all serving/promotion/source admission flags stay false. Real live-source parity and accepted-model API V1 remain separate later work. |

No embeddings or fresh sentiment model inference are claimed. News columns describe
observed linked readable documents, multiple categories/directions/statuses, saved
signed sentiment where present,1/3 calendar-day windows, age/truncation/coverage and
four fixed interactions. Missing evidence remains null; a zero category share in an
observed nonempty pool is not proof that no such world event occurred. Source/target
exclusions and model settings are unchanged. Freeze this matrix with actual hashes,
missingness and scope-proof counts before invoking the prepared training operator.

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


## News prefit feature freeze — 2026-10-21T07:29:15.068215+00:00

Pinned fact report: `data/reports/swing_news_feature_acceptance_freeze.json`. Publication SHA256: `4112d3255ef57e598ced47401cbedabf4d40200be46dd50d6490dc04dbd056ec`.

The exact 209-column order (124 inherited plus 85 news columns), availability mapping, and every column's missing count are retained in that JSON for all 586,305 decisions across 59 months. Feature-eligible rows: 479,709; complete model rows: 4,482; fixed-horizon supervision available: 378,037; complete-case supervision: 3,184. Full saved-row replay reports zero clock violations.

Coverage remains explicitly unknown/unavailable where reported; it is not inferred from empty news. Monthly source-history suppression and retained-document diagnostics, original sentiment-choice policy and story-decision resolution counts/digests remain in the report. Counts describe this saved decision population, not unique articles or provider completeness. Availability remains a historical research proxy.

The pinned proof compares all 18 units in each retained baseline bundle with unchanged populations, calendars, target masks and date-balanced weights. Ridge and shallow XGBoost settings remain unchanged; medians, all-empty zero fallback and missing indicators are fitted separately on each training partition. This records the prefit evidence for the existing initial-fit research route, without selecting features or dropping rows from missingness/outcomes. No fit or economic evaluation was run by this freeze operator. Source/model admission, serving, promotion, prospective assessment and API acceptance remain unclaimed.
