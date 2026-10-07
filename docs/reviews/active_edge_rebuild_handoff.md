# Active Edge Rebuild Handoff

### Current continuation: source reviews complete; request comparison repaired; real retry pending

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

Before 95557 ended, an independent scoped news-semantic repair was implemented/pushed
as 77ff5f5:
content_review.py and new research/issuer_candidate_derivative.py plus direct units.
The running job's 131 dependencies exclude both files and were pinned to workspace
review-results/relationship-reuse-frozen-implementation.json. Recheck that inventory
after edits; all its files/config/data remained unchanged during the run.
Candidate repair and saved-text replay are not source qualification or training
admission. Full SEC packets, fresh blind judgments and unchanged gates remain next.
Commands: derive-saved-issuer-candidates --root C:/project/market-predictor
--parent-population data/research/swing_initial_fit_issuer_content_review_population_source_bound/_manifest.json
--parent-population-sha256 3c59845761740dc1a07a2f994cceaba53354c6f9486f4baa634ebd422a6a84ce
--output data/research/swing_initial_fit_issuer_candidates_revised.
The real derivative is published; session 60410 now independently verifies the
actual manifest and then original candles. Never invent a receipt or manifest SHA.
The standalone derivative is implemented; full-packet sample/review/qualification
integration is still pending. Original qualification readers stay strict and cannot
accept the old population under the new policy; no blanket producer-pin waiver.
121 initial unit cases passed (9.64s). Final selected direct-consumer scope:
242 passed, one stale command-inventory assertion failed (179.72s). Inventory names
were added; a 23-pass rerun still exposed alphabetical order, then its corrected
single case passed (2.92s). Six-file Ruff and three-source strict typing passed.
Initial 19 strict narrowing errors are repaired with all runtime checks preserved.
One consolidated review found no further issues. All logs are unit-only evidence
in review-results/issuer-semantic-* and issuer-cli-inventory-*; actual derivative
publication is now complete as recorded above, but new source qualification remains
pending. The completed comparison's 131 hashes stayed unchanged.

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

Latest inspected usage 57% used; update this handoff again at 99% used as requested.
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
Last verified publisher implementation: `cb70f81`, following `ebe5dfb`.
Monitoring implementations: `b3c8761`, `8fdb29f`, `116d711`. All are pushed on main
through `bb88f75`; the user explicitly authorized GitHub publication and main merges.
The code changes are committed; this documentation receipt is committed separately. The continuation baseline was `21ef677`.
Part (3)'s design: `38c698e`, amended by the September 28 review in `18bdd8e`.
Earlier retirement sub-slice (b) closure: `ea93712`, after `7dd6d44`.
Last completed issuer-content implementation: `90a7f16` (pushed; candidate extraction only).
Last completed model-training checkpoint: relationship run on `a8be7cb` (artifact pins below).
Baseline model-training checkpoint: `07963cc` (pushed; unchanged).
Source-collection checkpoint: `19698d6` (pushed).
The combined historical outcome/features delivery retains its original receipts;
specific reuse conflicts require verification; blanket reconstruction is superseded.

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

## Current State

Latest implementation: `da67c5c` (pushed), following `12baec5`; following `635ebdb`/`b97ad60`. Canonical source reconstructions and
standalone adjusted collection are complete; technical predictors now also have a
complete independently audited feature-only publication.
Exact pins and the next feature dependency are below. TradingFlow remains untouched.

Canonical code cleanup completed in `af9e9b4`, pushed on
`codex/v1-canonical-cleanup`. Main remains at the pre-cleanup state; TradingFlow is
untouched. No internal generation suffixes remain in active schema/policy identities;
only explicit communicating protocols retain V1. External provider URLs and preserved
raw/historical locations are not compatibility implementations. Removed the weaker
Alpaca transport reader and duplicate training constants; temporal consumption uses
the canonical current panel producer and session-aligned clocks.

Verification receipt: full pytest run completed in 2,205 seconds with 4,869 passed,
11 skipped and eight old identity/hash assertions failing. The eight assertions were
corrected for the deliberate canonical identity reset; the historical August SIP
request fixture was restored byte-for-byte and now proves old/current identities
differ. Final rerun: all 152 tests passed across collection contracts, GDELT, issuer
attribution/classification, strategy/swing contracts, monthly preparation, continuity
and the new naming guard. Ruff/diff checks passed; strict mypy passed all 97 changed
source files. Independent design and consolidated code/ML review passed. No real-data
training, performance improvement, promotion, deployment or TF build was performed.
Logs: task workspace pytest-canonical-full.log and pytest-canonical-final.log.

October 3 candle/API review: the active plan's "Shared Candle Flow And V1 API
Recommendation" records a common collection/publication flow with separate consumers.
Recommend daily MP training/prediction and TradingFlow warm-up (300 completed sessions
or larger indicator requirement), 15m new entry baseline (20 sessions or larger
requirement, shortlist/orders/holdings only), and optional derived hourly data.
Existing TradingFlow 5m/hourly execution dependencies remain until individually
verified; no blanket minute-data deletion, four-hour switch, or live protection change.
MP incremental collection already uses daily bars. Daily/intraday history windows
must be separate; TF's 60-calendar-day default cannot initialize a 200-session average.
The plan contains source links, concrete observed defaults and migration exit gates.

The user's subsequent instruction requires cleanup first, without compatibility.
The branch producer now declares market_predictor.prediction.v1 under /v1, and its
served fixture was regenerated (SHA-256
59288471c249b0422443f0a68ce7623f4b03956826877ad1158dc79ebf1fea51).
TF's concurrently edited unified-swing-product checkout still expects unversioned
contract=market_predictor.prediction; its developer must adopt contract_version and
the V1 fixture. The candle API market_data.candles.v1 remains a proposed interface.
No TF files/builds/processes, raw downloads or collection settings changed. The active
plan now records the user's full collect-to-training-to-API pipeline and branch-based
improvement policy. Software tests do not establish a model's SPY outperformance.

Canonical reconstruction required after cleanup: saved derived training/model/feature
artifacts and current-code replay receipts are not automatically admitted by renaming
their schemas. Raw bytes and historical receipts remain intact. Concrete config chains
affected by changed identities are `configs/swing_corrected_research_features.toml`,
`configs/swing_return_relationship_publication.json`,
`configs/swing_return_relationship_readiness.json`, `configs/swing_training_readiness.json`
and `configs/swing_research_evidence.toml`: their saved parent provenance binds the
old strategy/temporal/feature/failure-fact hashes. Rebuild or explicitly verify their
canonical publications from preserved sources; do not simply rehash receipts or
claim that historic models were retrained. This is outstanding data work, not a reason
to retain compatibility readers in the current source tree.

Future-build policy links were refreshed separately: `swing_initial_fit_monthly_news.json`
binds current catalyst-lineage policy bytes; both `swing_issuer_content_cohort_inventory`
JSON configs (including `_with_legacy_proofs`) and `swing_legacy_query_identity_proofs.json`
bind the current monthly-news config. Their lightweight test checks the links and
loads the policy without reading archives. Current strategy-governance config bindings,
execution-policy pin and the current hypothesis-registry link were also refreshed;
historical artifact pins were not rewritten.

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

Current checkpoint: **Qualified issuer-reaction feature engineering** (`in progress`).

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

Historical September 28 authorization covered the completed TradingFlow API migration.
The latest user instruction supersedes it: leave TradingFlow untouched while its
separate development continues.

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

Historical part (3c) receipt `116d711` (September 28, local; publication pending):
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
consumer migration and publisher were pending at that checkpoint; see current receipt above.

Monitoring part (3a)/(3b)/(3c) is implemented and verified. The earlier publication
block described in these historical receipts was resolved by the user's explicit
merge/push instruction. See the current main integration receipt above.

Historical Part (3b) implementation `8fdb29f` (local; publication awaiting authorization): scoped
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


Work split (user decision, September 27): this developer owns `market-predictor`; a
second developer (Astra, a separate ChatGPT/Codex session) owns `trading_flow`. They
coordinate only through files, relayed by the user: the contract
`docs/contracts/prediction_api.md` with its golden fixture
`tests/fixtures/contracts/swing_prediction_response.json` (current API v4 fixture), the handoff
`docs/contracts/tradingflow_handoff.md` (paths, boundaries, tasks), and TradingFlow's
`docs/integration/market-predictor-notes.md` for acknowledgements and requests. Do not edit
`trading_flow`; record every wire change in the contract's Change log first.

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
- September 26 01:31-01:52 UTC: `data/raw/sec_acceptance_clock_pages_spaced_retries`
  completed with manifest
  `3905e8acf99fe386d9a7b9edd1fd4ed06ec163aa2486cab7f5dea4ccb9d8687b`: all 4,957 pages
  archived on the first pass, with no retries.
- Clock `data/research/sec_acceptance_clock`, manifest
  `51da245ea4227a6d6a0b18a1753d22d9290232e7a71c335e11fddfdafe69c4e1`:
  - 511 issuers `utc` (628,266 archive filings);
  - 113 `new_york_wall_clock_labeled_utc` (61,201 filings);
  - none unknown; no page disagreement or contrary in-data count.

  The 113 equals the earlier diagnostic: 93 identified from pages plus 20 from in-data
  counts.
- Corrected inventories: config `configs/swing_sec_form_inventory_corrected_clock.json`
  (SHA256 `a5f793807dbd1e907be199e5315b6a4809b35786972b63d48a055d8b516b30b4`). The first
  attempt failed on a saved row from 2000, before the exchange calendar starts;
  `cc4cfe9` skips rows that can never be available in the window.
  - Initial fit, `data/research/swing_initial_fit_sec_form_inventory_corrected_clock`,
    manifest `3ca7750689e42249d35c2cb68751b459060ac8ff694b4c2168f19878f91b4b07`, against
    the superseded counts:
    - 389,876 rows (389,875 before) and 387,472 accessions;
    - 36,536 accessions re-timed from New York labels;
    - 4 saved rows outside the archive's events, included;
    - acceptance positions: intraday 131,469 (153,868 before), after close 232,191
      (206,183), pre-open 22,515 (26,123);
    - 22,066 selected 8-K units (22,067 before);
    - one issuer with an unsaved page for 2019-07-08.
  - Sealed later window, `data/research/swing_later_sealed_sec_form_inventory_corrected_clock`,
    manifest `ae0ed0b688f009485e0f9d0a1e6b0c5ae1e1c7a3d89f648ff7de02e9a1a8807c`: 232,393
    accessions, with no rows excluded for an unknown clock.
- Initial-fit documents, September 26 06:32-10:39 UTC:
  `data/raw/sec_filing_documents_initial_fit_corrected_clock`, status `complete`, manifest
  `b3041a07f599e5c7f6a7b86a5991864c81586ba98e68391311fa3e9e84969318`, checkpoint
  `6a34c2ad833467f809a9c14155c97b9e95f458fa7a8ba5a8a1683b0446855091`:
  - all 22,066 filing indexes archived;
  - 45,160 documents archived (22,065 primary, the rest EX-99 exhibits) and 11 over the
    size limit (1 primary, 10 exhibits), recorded as such;
  - 67,226 archived attempts (67,073 HTML, 153 PDF, 11.68 GB); 86 retryable attempts all
    reached a final outcome in later passes.
- Sealed later-window collection from inventory `ae0ed0b6...` into
  `data/raw/sec_filing_documents_later_sealed_corrected_clock` (reports unit and attempt
  counts only; stays unread): the first run, 16:56-17:49 UTC, ended `incomplete` with one
  index unit lacking a final outcome, checkpoint
  `ed0effec0d6776bc4a11398087b9fe577194faad6b6c3f5734953e76c935fea4`. The first resume
  (17:50-18:27 UTC) stopped at the 90% system-memory guard with its checkpoint
  `3901c36b...` saved; the second resume (September 27 05:04-05:35 UTC) completed:
  status `complete`, manifest
  `f2a24e1d90556a5ed68adc5c36e1d649e3876cdc25b8b9b50700786b15aa8729`, checkpoint
  `e32b056529dc470da4937da60253eae6e57ca098b5db0896852dce6e1fb7c250`; all 9,246 index units
  and 19,061 document units archived, 1 over the size limit; 166 retryable attempts all
  reached a final outcome. Both SEC collections are complete; slice closure remains.
- A `test_canonical_cli` failure seen twice during the collections is explained: the
  decision build takes the heavy-job workspace lease, which the collector held (exit 75).
  The test now uses a private runtime directory.

Retirement sub-slice (b) `7dd6d44` is pushed (187 files; designs consolidated in
`c126c22`; diff review by both reviewers pending). Serving, selection, bundles, readiness,
outcomes, performance and drift are swing-only; every unpinned intraday module, its tests
and the retired ER1 readiness tooling are deleted; the four pinned intraday files stay
for (c). The plan's "Sub-slice (b) implementation record" lists the evidence-backed
corrections to the review decisions (session-only horizons, exact XNYS overdue limit,
nullable `training_data_end`), the four HEAD defects fixed, and the open replay defects.
- Verification tier: component checkpoint (shared contracts and schemas changed).
  - Affected tests from the import graph plus the config and source-scanning tests,
    58 files in eight memory-guarded batches: 1,058 passed, 1 skipped, 1 failed. The
    failure (`test_canonical_cli`'s decision pipeline) happened once while the SEC
    collection ran; its message was not kept, the batch passed three times afterwards
    (45 each), and none of its modules changed.
  - After the last edits: 16 re-affected files 148 passed; package boundaries 221
    passed; `test_drift_policy` 18 passed.
  - `mypy --strict src/market_predictor scripts`: 355 files, no issues. Ruff on the 51
    changed Python files: clean. All 321 package modules import. `git diff --check` clean.
  - TradingFlow `MarketPredictorHttpClientTests`: 18 passed (no TradingFlow change);
    build servers shut down afterwards.
  - Not run: the full suite (not a release checkpoint) and the swing-only predictor and
    outcome replays, which belong to (c).
- Diff reviews (both reviewers, no blockers) are closed by `ea93712`; the plan's
  implementation record lists each finding and its fix. Its verification (component
  checkpoint):
  - affected tests from the import graph plus the boundary and smoke tests, 30 files:
    46, 61, 288 passed; the fourth batch failed two `test_swing_live_features` tests that
    read `excluded_security_ids`, which the change had dropped; it was kept beside the new
    `excluded_tickers`, and that batch then passed 131 (1 skipped);
  - the smoke-release test fails with the reviewer's error when the fix is stashed and
    passes with it;
  - `mypy --strict src/market_predictor scripts`: 354 files, no issues; Ruff on the 24
    changed Python files clean; `git diff --check` clean;
  - a scan of every implementation-pin list (61 paths) shows no pinned file changed in
    `7dd6d44` or `ea93712`, so the sealed collector's resume re-hashes unchanged files.
- User decision (September 26): fix the ML review's verified monitoring defects and the
  investment replay defects now, as one step before (c). The design, "Swing monitoring
  and replay correctness design", is in the plan. Both design reviews (September 27)
  found two blockers each; their consolidated decisions follow the design in the plan
  and supersede it. They go back to both reviewers before any code. A draft session
  helper and the replay-boundary edits are kept aside in the session scratchpad, not in
  the tree.
- Naming (user, September 27): `43e1fd2` removes version numbers from class names and
  from every record identifier nothing persisted depends on. The user then clarified that
  the public API stays versioned: the response keeps `contract_version` =
  `market_predictor.prediction.v3` and the `/v1/` routes, restored in `2acd36c`.
- TradingFlow (Astra) had already integrated the unversioned response; the contract's
  change log is now append-only with a "correction" entry (fixture `360206c2...`, which
  also drops the version from `market_predictor.swing_outcome_policy`), and the
  TradingFlow handoff opens with the required action. Both re-reviews of the consolidated
  monitoring design found no blocker; their decisions are in the plan ("Second-round
  review decisions"), and part (1) starts next.
- Monitoring part (1) `e9526f1` (sessions, overdue, maturity-aligned windows): a shared
  XNYS session helper with a parity test against the pinned holding calendar; windows
  include a decision when its horizon's last close falls inside them or has not closed;
  the route-wide oldest pending decision blocks drift even outside the window; 180-day
  lookback default with a window check (150 days refused: its fewest sessions are 99);
  only `10b` has evidence minimums. New drift pin `c64ca881...`. Verification: 18
  import-affected test files plus the boundary and smoke tests, 433 passed; strict mypy
  (355 files) and Ruff clean. Both diff reviews found no blocker and no major; their
  minors are fixed in `2f05100` (444 affected tests passed, strict mypy and Ruff clean).
- Monitoring part (5) `a9c1b89`: session partitions and a pending index (511 affected
  tests passed, strict mypy and Ruff clean). Verification: 55 affected test files
  in seven batches, all passed except `test_canonical_cli`, whose lease clash is fixed in
  `c628aea` (it passes while the real lease is held); strict mypy (354 files) and Ruff
  clean; no pinned file changed; no stored data used a renamed identifier.
- Monitoring part (2a) `c39c68a`: a target or stop reached before a data gap matures with
  that exit; outcomes record holding sessions and the fixed-horizon net return and sector
  excess (the trainer's economic target) when the whole path is observed. 524 affected
  tests passed; strict mypy and Ruff clean.
- Part (5) diff reviews: the code review found two majors (a crash between the semantic
  record and the index entry lost the intent from the index for good, and `pending/`
  kept one lock file per intent ever pending); the ML review found the same crash window
  and minors only. All are fixed in `157f330` (details in the plan's monitoring record),
  including the worker waiting for the horizon's last close, which keeps part (2a)'s
  fixed-horizon return on early exits. Verification: the 24 import-affected test files,
  305 passed, and the two edited ones again after the last edit, 42 passed; strict mypy
  (355 files) and Ruff clean.
- Repository benchmark (one-off scripts, not tests), 480 members per session:
  - Registration writes 35 ms per intent, so a year of 121,000 intents takes about 70
    minutes to build and a nightly cross-section about 17 seconds.
  - A report over a 14-day window, after the fixes: with 10 stored sessions, 4,800 window
    intents in 5.5 s and a 71.5 MiB peak Python heap; with 30 stored sessions, 5,280
    window intents in 11.0 s and 95.4 MiB. The difference is the route-pending scan: no
    outcome matured in the benchmark, so all 14,400 intents stay indexed, against about
    5,300 (11 sessions) when maturation keeps up. Listing the index took 0.33 s and
    1.09 s.
  - Extrapolated to the 180-day default (about 62,000 window intents): about 70 seconds
    and 1.1 GiB of Python heap, within the 5 GiB process budget.
  - The first run (`a9c1b89`) reported 249 s and 397 s, but it timed the report under
    `tracemalloc`, which slows every allocation; those timings are not comparable.
- Diff reviews of `c39c68a` and `157f330`: no blocker and no major from either. Their
  findings are fixed in `1ef0a7f` (details in the plan's monitoring record): maturation
  checks only the rows it uses; an early exit before an unproven gap stays pending; the
  ATR is stored as a fraction, so later splits no longer move the stop and target; and
  the outcome contract ties timeouts and holding periods together. Verification: the
  25 import-affected test files, 318 passed, plus the dependency and architecture boundary tests, 230 passed; strict mypy (355 files) and Ruff clean.
- Confirmation reviews of `1ef0a7f`: no blocker and no major from either. Their minors
  are fixed in `b872b30` (decision close bound to the evidence's decision bar; a
  timeout's sector interval checked; calendar-edge errors reported as conflicts).
  Verification: the 25 import-affected test files, 319 passed; strict mypy (355 files)
  and Ruff clean.
- User decision (September 27): no production code publishes the swing live-input
  generation (daily bars and point-in-time memberships behind `active_generation.json`),
  so nothing can register or serve real predictions yet. The nightly live-input
  publisher is built right after part (3), so registration is tested end to end on real
  Alpaca data before parts (4) and (6).
- Part (2b) design `c87a9e4` and `057fbf5`, consolidated in `2cf26dd` after both design
  reviews (no blocker; majors on one price basis per path, settlement before the overdue
  deadline, and unresolvable intents leaving pending). Two provider facts were measured
  with read-only requests: Alpaca omits a symbol without bars (`{"bars":{}}` when none
  has any), and `asof` on the decision session follows a later rename (FB to META).
- Part (2b-1) `d23df6a`: receipted daily and one-minute bar collection and corporate-action
  collection (`collection/outcome_bars.py`, `collection/http_records.py`) and the shared
  monitoring lease (`monitoring_lease.py`). Verification: the new tests, 13 passed; the
  dependency and architecture tests, 230 passed; strict mypy (358 files) and Ruff clean.
  No existing module imports the new code yet. Both diff reviews (no blocker) and their
  confirmations are fixed in `b96c593`, `1388eb6` and `62da557`: chains anchored at the
  first page, receipts matched to the sessions they cover, bars retrieved before they were
  final not counted, daily bars stamped at New York midnight, a missing price kept as an
  unusable bar, a duplicated session a defect, a provider repeating an action across pages
  a failed receipt, and the attempt window spanning the retrieval clocks.
- Part (2b-2) `7ca95fe`: maturation from receipts (`governance/outcomes/evidence.py`,
  `collection_plan.py`, the worker and both commands). Paths come from the receipt with
  the most usable sessions; a gap is proven only by a receipt retrieved three days after
  the last close (`outcome_settlement_days`, new drift pin `1351df40...`) when no receipt
  ever returned a usable bar; cessation (mergers, worthless removals, membership
  removals, following renames) or a minute-confirmed halt makes an outcome
  `unresolvable`, which leaves the report's pending counts. Attempts are an append log
  written only on change. Verification: the 27 import- and config-affected test files,
  354 passed; strict mypy (360 files) and Ruff clean. Both diff reviews (no blocker; one
  major each, the same one: a stock that stopped trading without provider evidence could
  stay pending and block its route for good) are fixed in `1871f7e`: an audited operator
  resolution (`record-operator-outcome-resolution`, naming the operator and the evidence,
  superseded only by a later maturation) and single-intent collection past the freeze;
  worthless removals counted by a process date on or after the decision; maturation
  reading only receipts collected by its time; attempts running forward in time; a new
  drift policy writing a new attempt; settlement judged by the start of a receipt's
  retrieval; corporate actions asked nightly until the deadline. 359 passed on the same
  set; strict mypy and Ruff clean. One corrupt receipt still blocks its decision session
  (fail closed); the operator path resolves the intents it holds. Part (2b-3), the
  unresolvable ceiling and the sensitivity, must land before registration runs.
- Part (2b-3) `8f389fd`: the 5% unresolvable ceiling applied once evidence suffices
  (`maximum_unresolvable_share`, new drift pin `1d21b757...`), and diagnostic sensitivity
  fills on every entered unresolvable outcome: the last usable close and a stress value
  (a merger's cash and acquirer shares at the acquirer's close; nothing for a worthless
  removal; -55% otherwise, the listing exchange not yet recorded; the managed path past
  a halt). Merger terms were measured on stored provider records (Xilinx 1.7234 AMD
  shares, Zynga $3.50 plus 0.0406 Take-Two, FLIR $28 plus 0.0718 Teledyne), since the
  provider documents no field meanings. The confirmation reviews of `1871f7e` are closed
  here too: an operator resolution needs an overdue intent with a proven stock gap or
  blocked evidence and a well-formed operator id, unreadable evidence never undoes it,
  and provider evidence naming a reason replaces it; operator ids are not yet tied to
  authenticated principals. Verification: the 26 import- and config-affected test files,
  346 passed; strict mypy (361 files) and Ruff clean. Both diff reviews (no blocker; one
  major each) are fixed in `346e0ec`: reorganizations are matched on the company's own
  `symbol` (the provider's measured record shape: `symbol` and `stock_movements`, no
  acquiree field) and filled at the last usable close, like a stock merger whose acquirer
  has no bar, never as a loss; fills need the sector bar on their session; never-entered
  decisions are marked and stay out of the sensitivity means; an operator may resolve a
  block only for defective stock evidence, and resolutions no longer need readable
  evidence; `quarantine-outcome-receipt` moves a receipt that no longer verifies aside
  (with who and why), so its session loads again. The ceiling over deadline-passed
  decisions only belongs to part (4), with every other metric. Verification: 352 passed on
  the same set, plus the collection and dependency tests, 246 passed.
- TradingFlow follow-ups, both display-only: swing signals read as neutral in the
  advisory model-direction view; the hard-coded `market_predictor.prediction.v1` label.

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

Exact next checkpoint: source step 1 and the observed request representation
software repair are complete. Start the fresh actual retry wrapper with unchanged
A/B/config/source pins and a new 27-file inventory. Preserve original failed
session50119/PID97320 logs and exit 1. Run installed publication plus independent
verification sequentially; only those actual results and natural exit 0 close
requested step 2. No repeated source reviews or mocked operational data.
The earlier qualification/reuse reruns exited for system memory pressure. Preserve
4422/PID 87284 and 37426/PID 87492 failure receipts and artifacts; both leases gone.
Saved-config implementation b589725 and actual byte/fold proof are complete.
Publisher/verifier ownership repair c2063a8 is pushed and its full 314-case selected
unit scope passed. Actual 124 reuse retains config SHA256
95dd86ec733a70a72c0d0a2f05de78e0111b71e31fef1786d424136141cc209c.
Check current headroom first, then run only one heavy job at a time; freeze pinned
source/configs during each run. Do not relax the canonical 85%/2 GiB source guard.
Do not repeat completed export, matching, independent reviews or passed unit scopes.
Preserve failed qualification 64007, reuse 39985/diagnostic 87347, and the failed
unit attempt 60429 alongside corrected 18-case unit receipt. Intermediate real
review metrics reject both families; preserve exact judgments/gates and report
final measured results without invented events or assumed training admission.
Only a passed exact reuse comparison authorizes the current-source 124 derivative/receipt.
The reviewed original-input research route instead requires its independently passed
original-snapshot receipt and explicit reaction consumer binding. Qualified source and
verified 124 parent precede actual 126 publication, receipt/readiness
and the two remaining fits. No generic news trial, target rebuild, API-version change
or TradingFlow write. The no-synthetic CI correction is closed in 892100c/88fd60a.
Main and TradingFlow remain untouched.

Publisher verification uses `.venv/Scripts/python.exe`, `PYTHONDONTWRITEBYTECODE=1`,
writable TEMP/TMP, `-p no:cacheprovider` and a unique writable `--basetemp`:
- pytest `tests/test_live_input_publication.py tests/test_swing_live_features.py
  tests/test_session_registration.py tests/test_prediction_service.py tests/test_api.py
  tests/test_cli_surfaces.py tests/test_package_dependency_boundaries.py
  tests/test_architecture_boundaries.py`: 333 passed, one skipped.
- pytest `tests/test_swing_prediction_api.py tests/test_active_continuity_documents.py`:
  nine passed. Final publisher-only run after metadata bound: 17 passed.
- Ruff over the three changed source modules and publisher tests; strict mypy over
  live_input_publication, swing_features and session_monitoring. Final metadata fix
  repeated publisher lint/type checks successfully.
- TradingFlow snapshot: `dotnet test src/TradingFlow.Tests/TradingFlow.Tests.csproj
  --no-restore --disable-build-servers --filter
  "FullyQualifiedName~MarketPredictorHttpClientTests|FullyQualifiedName~UniverseRankServiceTests|FullyQualifiedName~PredictorEvidencePresentationTests"`:
  84 passed. `dotnet build src/TradingFlow.Mobile/TradingFlow.Mobile.csproj --no-restore
  --disable-build-servers -f net10.0-android -v minimal`: succeeded. TEMP/TMP pointed to
  writable chat work; no device install or live endpoint acceptance was performed.

Part (3c) verification commands used the project `.venv/Scripts/python.exe` with
`PYTHONDONTWRITEBYTECODE=1`, TEMP/TMP set to the chat's writable work directory and
`-p no:cacheprovider --basetemp=C:/Users/manis/Documents/Codex/2026-09-28/c/work/<unique>`:
- 323 passed: pytest `test_session_population`, `test_performance_monitoring`,
  `test_drift_policy`, `test_session_registration`, `test_outcome_repository`,
  `test_outcome_intents`, `test_package_dependency_boundaries`, `test_architecture_boundaries`.
- Additional direct-consumer verification: `test_outcome_commands`,
  `test_outcome_maturation`, and continuity tests passed in the 88-pass run described
  above. Its single new population fixture failure was corrected and reverified.
- Final 25 passed: pytest `tests/test_session_population.py`,
  `tests/test_session_registration.py`, `tests/test_outcome_intents.py` (46.41 seconds).
- Ruff `--no-cache` over all 11 changed Python files; strict mypy over performance,
  repository, session coverage, session records, drift policy and session registration.
  No full-suite or fresh C# build was run for the internal reporting checkpoint.

Historical research constraints still apply: unchanged 586,305 decisions, labels,
weights, costs, splits and approved exclusions; initial fit July 2019-May 2024.
The qualified-content source is the completed inventory with legacy identity proofs,
keeping the weaker CUSIP-chain provenance distinct. Old model/evidence pins must not
be rewritten. Decision config: `configs/swing_corrected_outcomes.toml`,
`ded3b30af1185c7ab0759fa5f81c559c37c590419751c942b61ab479e67c2348`.
Strategy config: `configs/edge_rebuild_strategy_contract.toml`,
`02a087be6b9eff4971770026ca75dce3978f8f9e2028c1f21d027daefec9c0e7`.
Heavy materialization/fits remain sequential under the shared lease and memory guard.

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
