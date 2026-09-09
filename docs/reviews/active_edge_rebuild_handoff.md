# Active Edge Rebuild Handoff

Status: active
Last updated: 2026-09-09
Repository: `C:\project\market-predictor`
Branch: `er-intraday-refactoring`
Last completed implementation commit: `d49c6a4` (pushed).

## Current State

Initial-fit raw-share planning, collection and offline replay are complete in
`d49c6a4`. One immutable plan reconstructs exact decision/holding sessions, complete
SPY/QQQ/sector ETF ranges, provider mapping and per-unit counts/digests from pinned
cohort/identity evidence. Replay requires an independently retained authority-file
hash. The shared collector retains original response bytes and reconstructs Parquet
from them; collection is not ownership, exact-session coverage or accounting admission.

Plan: `data/reports/swing_initial_fit_raw_share_plan`.
Authority-file pin: `d912a997af361c820745e8c850f0fd455ee068397c6d034d2b54c00a475dc22e`.
Archive: `data/raw/swing_initial_fit_raw_share_daily`.
Authority-file pin: `144cab43741f3c74308b53e9322c84ac7eeaca158d3cbd6a1ddb0d7c0fa8d244`.
Manifest-file hash: `a96c70eee46e0a4b4d0ee3c1a7d47cea40dfc7a4ec2e14903f03e786bd0ced52`.
Use those independently saved pins, never a freshly recomputed replacement after a
verification failure. These completed archives are immutable and must not be redownloaded.

564/564 units returned observations, no failed/empty units, **601,834 raw SIP daily
rows**. Separate offline CLI replay passed. Scope: 2019-07-09..2024-05-28 only,
545 in-window security IDs, 551 stock ranges and 13 benchmark ranges. The other
41 retained IDs enter later; they are not exclusions. Requirements: 586,305 decision
sessions, 586,414 holding sessions, 551 decision-only sessions, 586,965 stock union
sessions, plus 16,003 benchmark sessions. All 586 retained IDs remain bound.

**Known unresolved count differences:** 602,968 required versus 601,834 returned
rows, short by 1,134 across 28 ranges. ECHO is short 630, FISV 245; 26 other ranges
are short nine or ten. This count check does not identify exact missing dates or
prove the returned bars belong to the intended security. Investigate date-specific
symbol mapping and corporate actions before interpreting these as missing trading.
No source was imputed, no additional stock excluded and no label admitted.

Verification: 358 focused tests, full tracked-Python Ruff, strict mypy 338 sources;
full suite **3,082 passed, three skipped, 134 warnings**, 1,266.62s (21m06s), peak
0.329632 GiB. XML `.test-tmp/raw-plan-full.xml`; PID13376/session99832 exited.
Plan peak 0.396503 GiB; collection peak 0.395340 GiB. Plan PID35088/session19227,
collection PID20404/session76880 and offline replay PID31288/session69985 exited.
Consolidated review fixed pin-replacement races, scope downgrade and per-unit audits.
Review/test agents Lovelace `01a0865e-9703-7c81-b892-3e0c3b8de9ba` and Leibniz
`01a08660-3dbc-77e1-b8b2-253bdba74f4e` are closed. No owned process remains.
Untracked `.test-tmp/` is generated test evidence, not committed. Historical
checkpoint narration is retained in Git; this is the only current handoff.

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

Accounting kernel checkpoint `b7cdb4e` and transport checkpoint `46ab0f8` are closed;
the latest full suite above covers them. Do not reopen without concrete new evidence.

## Exact next checkpoint:

**Admit security-owned raw observations and corporate actions into actual holding
specifications, then materialize corrected targets.** The original Astra plan remains
the guide. Freeze this bounded design with an independent reviewer before coding.

1. Replay the saved plan/archive, compare exact session sets (not counts alone),
   and resolve date-specific symbol/class ownership, especially ECHO/FISV and
   holding tails after removal. `asof=unit_end` is a provider mapping parameter,
   not independent ownership evidence. Unchanged row counts do not prove identity.
2. Reuse the 60-name action archive and official documents; extend only genuinely
   missing cohort/ETF action and successor evidence in separate pinned archives.
   Never rewrite the completed raw-price plan or replace original observations.
3. Build actual `HoldingSpecification` and target rows under `swing/datasets`, using
   the shared accounting kernel. Match stock/SPY/QQQ/sector executable intervals,
   entitlements, payable/availability times and costs once. Unsupported delivery,
   fractions, currency, payment dates and marks remain explicit gaps, not zero.
4. Verify immutable source-to-target replay, ownership and missing-session failures,
   no held-out numeric reads, stock/benchmark accounting and bounded memory. Push
   implementation and the two-document closure before dependent feature/training work.

Numeric scope stays **2019-07-09..2024-05-28**; do not inspect held-out prices or
broaden exclusions. Last mature decision is 2024-05-13. Raw collection is complete,
not a reason to repeat downloads or produce another metadata-only substitute for
the materializer. After admission, rebuild news/reaction features, run the six
candidates sequentially, evaluate against SPY and collect fresh prospective evidence.

Read `swing/datasets/initial_fit_raw_share_plan.py`,
`edge_rebuild/swing_history_collection.py`, `swing/labels/holding_paths.py`,
`swing/contracts/holding_accounting.py`, `swing/evaluation/holding_accounting.py`,
`swing/labels/holding_accounting.py`, and `swing/datasets/corporate_action_collection.py`.

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
