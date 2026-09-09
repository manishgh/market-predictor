# Active Edge Rebuild Handoff

Status: active
Last updated: 2026-09-09
Repository: `C:\project\market-predictor`
Branch: `er-intraday-refactoring`
Last completed implementation commit: `46ab0f8` (pushed).

## Current State

Price-basis-aware collection and original-response replay are verified and pushed
in `46ab0f8`. New acquisitions derive raw/all from their plan, retain original HTTP
bytes/query receipts and reconcile normalized Parquet with decoded source rows.
The consumer must supply `expected_adjustment`; adjusted combination requires all.
Historical adjusted evidence without receipts is not fabricated into raw evidence.
Cross-basis resume fails before calls/writes. No new raw data or model was created.

Both agents completed and closed: Volta `01a0861c-bd36-73f0-bdec-f29e2742be27`
(source advisory/tests), Hilbert `01a0861c-bcac-7f40-a20c-9e83110e2618` (design and
consolidated review). One P2 rejected-receipt loss was fixed and regression-tested.
53 source/collector plus 14 combination tests pass; full tracked-Python Ruff and
strict mypy (336 sources) pass. Full suite: **2,999 passed, three skipped, 134
warnings**, 1,181.30s (19m41s), peak **0.329193 GiB**. PID25540/session97992 exited;
XML `.test-tmp/price-basis-full.xml`. Retained adjusted warm-up replay passed for
549 units / 140,383 rows, peak 0.132084 GiB, PID12624 exited. No owned job remains.

Event-aware accounting is implemented and verified. Untracked `.test-tmp/` contains
generated evidence and is not committed.
Historical narration remains in Git; this is the only current handoff.

The user approved separating tradable shares, available cash, unpaid proceeds and
contingent rights. Unknown marks/timing remain unavailable, never zero or a payout
cap. The approved exclusion list stays **45/631 excluded, 586 retained**, under the
**10% cumulative ceiling**. No additional exclusion is approved.

**No new candidate has been fitted or proven profitable.** Real source admission,
corrected labels, news/reaction features, six sequential fits and evaluation remain.
Intraday is paused. Alerts/execution belong to trading_flow. Prospective promotion
needs 252 new sessions plus the ten-session outcome tail; exposed history is not fresh.

## Verified Implementation

Commit `b7cdb4e`:

- `swing/contracts/holding_accounting.py`: strict evidence, events, positions,
  executions, payment, availability, marks and outcomes.
- `swing/evaluation/holding_accounting.py`: canonical lot replay and target projection.
- `swing/labels/holding_accounting.py`: stock/SPY/QQQ/sector fixed and managed targets.
- `swing/evaluation/ledger.py`: one funded loop, including explicitly unadmitted
  retained price-ratio diagnostics; no production fallback.
- `swing/evaluation/accounting.py`: component NAV and benchmark comparisons.
- `swing/labels/barrier_and_rank.py`: an evidenced early exit no longer requires
  later bars. Missing data before exit still blocks.
- `configs/swing_research.toml` and research contract: named event-aware schema,
  raw prices plus explicit entitlements, evidenced cash and residual claims.

Ten-session horizon, 20 bps prepaid cost, 40 bps stress, zero cash yield and unlevered
funding remain. Same-session sale receipts cannot fund the open; actual pre-open
corporate cash and prior-session sale receipts can. Payment extinguishes one claim.
Residual claims need evidenced marks; unknown NAV stops NAV-dependent allocations.
Benchmark distributions are retained as cash, not fictitiously reinvested.
Targets remain ineligible until independent source admission; no real data was rewritten.

Consolidated review fixed entry-tied event ambiguity, prior-session cash release,
irrelevant post-exit events and premature successor lifecycle references.
Verification after final code: **309 focused tests; full suite 2,974 passed, three
skipped, 133 warnings, 1,096.07s; full tracked-Python Ruff; strict mypy 336 files**.
Peak 0.328899 GiB. PID23576/session58923 exited; XML
`.test-tmp/event-accounting-full.xml`. All agents closed. No repeat full suite is
needed for documentation closure alone.

## Exact next checkpoint:

**Publish the pinned initial-fit raw-share acquisition plan, then collect it.**
The existing exact-unit collector's price-basis work is complete. Reuse that path;
do not create a duplicate collector or relabel adjusted prices as raw fills when
dividends are separately credited.

Read `edge_rebuild/swing_history_collection.py`,
`edge_rebuild/swing_history_acquisition.py`, `sources/alpaca.py::fetch_bars_page`.
The bounded metadata inspection found `adjustment=all` in modeled, warm-up, candidate
and transfer requests, not a complete raw or adjustment-factor authority.
This is not a claim that every file on disk was inspected.

Bind a fresh initial-fit plan/output `data/raw/swing_initial_fit_raw_share_daily`
for the fixed retained cohort, required holding sessions, SPY, QQQ and point-in-time
sector benchmarks. Numeric scope **2019-07-09 through 2024-05-28 only**. Earlier bars
are warm-up only. Do not extend the cutoff or inspect held-out numeric observations.
Use `1Day`, SIP, raw, bounded pages and explicit identity-unit `asof`. That field
maps symbols/entities, not price vintage or ownership.

Volta's metadata-only source advisory reproduced 581,455 mature initial-fit
decisions from 59 monthly partitions, 545 initial-fit security IDs / 551 ticker
pairs and 586,414 unique holding sessions. The other 41 retained IDs have no
in-window membership; they are not exclusions. The 1,231-session range requires
13 benchmarks: SPY, QQQ, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY.
The basic representation is 551 stock runs plus 13 benchmark units before any
independently evidenced successor requirements; last mature decision 2024-05-13.
These advisory counts are not a published acquisition authority. Extract the existing
acquisition publisher's staging/writer once for both planners; do not call its
warm-up-gap entry point or membership-clipping unit builder for holding tails.

Then independently interpret ownership/actions and materialize actual
`HoldingSpecification` and target rows under `swing/datasets`. Do not replace this
with another metadata-only report. The existing 60-name corporate-action collection
is not cohort/benchmark-wide coverage. Acquire only genuinely missing evidence into
new request-bound archives. Unsupported currency, fractions, delivery, payment or
valuation remains a gap. Existing adjusted sources and blocked controls stay immutable.

Next-plan exit: design review, exact cohort/session/benchmark reconstruction,
source-tamper and future-poison tests, initial-fit numeric boundary, immutable
publication/replay, focused/full tests, lint/types and real plan evidence. Collector
raw-request/basis/receipt tests are already closed; do not reopen them without new
evidence. Push implementation, then close/push both continuity documents before
dependent source-to-target code. The original Astra plan remains the guide.

## Source Authorities

All paths are repository-relative. Preserve raw data and pinned evidence.

| Authority | Path | SHA256 |
| --- | --- | --- |
| Approved cohort | data/reports/swing_research_cohort/approved_research_population_audit.json | 41de559dd1c415dab60771e10fd489150853a6c2fff07e387d1024b33961efe0 |
| Holding identity | data/reports/swing_research_cohort/holding_identity_preflight.json | 8f8cdd60ca2f9c1372ceda20d04b7f90eefbfb2336a7f282f30aa97b8c7e723b |
| Holding observations | data/reports/swing_holding_observations/_manifest.json | b87e18fc2c706600c0063c606c2dd40dbba0422cfad6b6c8c1e4817242714a8a |
| Corporate actions | data/raw/swing_holding_corporate_actions/reports/82dafea3055db20db9ee483800a7a22c508dbb325418b979a1cd23974a244c91.json | 82dafea3055db20db9ee483800a7a22c508dbb325418b979a1cd23974a244c91 |

Cohort hash: `794bcf834501cbaa8ec54716e8b561a5aaf0137610fe67587f4db69f9189fca6`.
85 monthly projections, 842,446 retained identity rows. This is retrospective
research selection, not a historically observable screen or accounting admission.

Identity preflight: 836,638 full-history covered mature windows, 947 uncovered across
100 IDs, 4,861 terminal immature. Initial fit: 581,455 mature; 580,889 covered,
**566 uncovered across 60 IDs**. These are ownership gaps, not missing-price counts.

Observation inventory: 1,106 required security/ticker sessions; 876 valid, 209 missing,
21 invalid. Separately 574 lack ownership proof. 37 IDs have all observations, not
proven total returns. The 23 with bad/missing observations: MYL, VIAB, TIF, ALXN,
ATVI, RTN, INFO, SIVB, SBNY, CXO, VAR, FLIR, KSU, TWTR, FBHS, CERN, AGN, DRE, APC,
PXD, PBCT, ABMD, MXIM. Do not automatically exclude them.
Owners: `swing/datasets/holding_observation_requirements.py`,
`holding_observation_inventory.py`, `holding_observations.py`.

Corporate actions: 60/60 acquired and independently replayed, 649 distinct records
(621 dividends, 21 mergers, five renames, two spin-offs); `collected_unreviewed`.
Process dates 2019-07-09..2024-05-28 are not effective or announcement dates.
16 mergers are in affected windows, 14 lack payable dates; all renames lack effective
dates. Match FBHS/FBIN spin-off across MBC query; reconcile NOV/HFC CUSIPs,
MYL/VTRSV/VTRS and FLIR dividend after merger. VIAB, RTN, APC, ABMD, SIVB and SBNY
lack resolving in-window merger evidence. Same-CUSIP anchors are not ownership intervals.
Collection implementation `58468fb`, closure `ec0f5b8`; existing archive resume/replay
requires independent audit pin. Interrupted first batch without a report needs
explicit recovery. Config `configs/swing_holding_corporate_actions.toml`; source
`sources/alpaca_corporate_actions.py`; dataset
`swing/datasets/corporate_action_collection.py`.
ABMD primary filing: https://www.sec.gov/Archives/edgar/data/815094/000119312522311074/d353287d8k.htm
A web read alone is not hash-bound dataset evidence.

Parent panel: `data/features/edge_rebuild_swing_technical_panel_20190709_20260708_v1`.
Request-file SHA `d0e093ce2192f8511547b5c5466ac90b892fbd44d4939b002242555a8b19a4ed`;
request identity `034163c5a890781c4b58b553191e1aefd9c7d760f1a9daff462635641b7ee122`;
manifest `891bb547cff304661153661b1c18acf57ec824e6c5c7ddd1eed7346a9a13886c`.

Adjusted source: `data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3`.
Request-file SHA `942fe53a50e35730f585bd14e72eb6305e4d84065084098c8b6b64ac0300fc37`;
request identity `1cf17ba2788167f14408de9e6163352842fb744c2d4d7b8195dc48ce60ab7a54`;
manifest `b99a1d13d9075220db30c3870bc0796b3631bf4ec0bf6424469feeeec9115d93`.
SIP daily **all-adjusted**. Request hashing uses standard JSON spacing, not compact
helper. Never relabel these observations raw.

Membership: `data/canonical/index_membership/sp500_memberships_20180529_20260708_v1/memberships.parquet`;
SHA `c70b305cb64e54bda2c5664729bcb25ed766f6f7a56d3a57e3cf5875c06406c2`.
658 IDs = 631 modeled + 27 warm-up only. Include excluded competing owners when
proving ownership. Keep membership-clipped feature history; pass independent admitted
paths to `build_swing_feature_rows(outcome_bars=...)`, not an unclipped feature loader.
The complete source is not yet admitted to materialization.

## Working Rules

Read AGENTS.md, the active plan and current feature audit before changes. Preserve
six candidates/twelve model-policy comparisons, source availability, exact XNYS
windows, costs once and frozen time splits. Old July2025..June2026 outcomes were
already exposed and remain research-only, not fresh validation.

Use `.venv/Scripts/python.exe`, explicit
`--basetemp=.test-tmp/<unique-run>` for pytest (default user temp is inaccessible).
Heavy jobs use the existing lease and run sequentially below 5 GiB. Track owned
PIDs; never kill unrelated processes or print secrets. Git writes may require normal
tool escalation; do not bypass sandbox controls.

The current feature acceptance matrix is
`docs/reviews/feature_engineering_audit_20260801.md`.
Continuity tests: `tests/test_active_continuity_documents.py`.
