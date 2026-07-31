# Active Edge Rebuild Handoff

Status: official S&P raw-source acquisition and offline event extraction are
complete and verified. Transition authority, membership reconstruction, and
Alpaca history extension have not started. Intraday corpus,
swing labels, cross-sectional scaling, causal technical relationship primitives,
and the current seven-year panel remain published and immutable. No model exists.
Nothing is running.
Last updated: 2026-07-31
Repository: `C:\project\market-predictor`
Remote: `https://github.com/manishgh/market-predictor`
Branch: `r3-lineage`
Last pushed checkpoint: `ea9fe14`

## Repository Cleanup

On 2026-07-31, obsolete generated data and legacy narratives were removed under
the new-only policy. No runtime compatibility path is supported.

- Deleted 26,602,214,748 bytes (24.775 GiB): rejected/superseded feature
  matrices, scratch work, the superseded `20260730` intraday corpus, old V2
  research outputs, and redundant June/early-July raw experiments.
- `data/` decreased from 39.603 GiB to 14.828 GiB.
- Protected seven-year Alpaca news, current daily and intraday edge-rebuild
  collections, the verified `20260731` corpus, point-in-time membership,
  Seeking Alpha cache, and reusable one-minute raw history all remain.
- The README was replaced with a current-state guide. Superseded V2/V3 plans,
  review chains, rejected unregistered model cards, and stale runbooks were
  deleted. Governance-required compact evidence and registered reference cards
  remain.
- Focused governance and architecture verification passed 17 tests. No Python
  process was left running.

The current generated feature authority is
`data/features/edge_rebuild_swing_panel_20190709_20260708_v1`. No superseded
feature directory is retained.

Read in this order:

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`, sections 8A, 8B, 8C
3. This file
4. `configs/edge_rebuild_strategy_contract.toml` — every threshold lives here

Do not infer state from chat history.

## How To Work Here

The user is a strong systems thinker and is not a machine-learning specialist.
Name the problem before the technique. Do not use jargon without unpacking it,
and do not use metaphors like "load-bearing" or "blast radius" — say what would
actually break. This was asked for repeatedly.

Do not investigate individual tickers. Record exclusions with a reason and move
on. Escalate only if the universe loses more than 5% of symbols.

Do not invent methodology. Use published methods, cite them, and name them so an
implementation can be checked against its definition. Sources are in plan
section 8B.

Update this handoff after every successful step, not only at checkpoints.

Never pipe `pytest` through `tail` or `head`. It masks the exit code, and a
failing test has already been committed that way. Redirect to a file and echo
`$?`.

Set every command's working directory explicitly to
`C:\project\market-predictor`. Do not inspect or modify `trading_flow`; it is
only the future consumer boundary for promoted prediction intelligence.

The heavy-job lease is exclusive. Running the test suite while a collection
holds it produces exit code 75 on CLI tests, which looks like failure and is not.

## Data Held, Measured From Disk

| | Swing | Intraday |
| --- | --- | --- |
| Universe | 627 securities, verified point-in-time, delisted included | 1,104 symbols, index members plus volume-screened non-index names |
| Bars | 1,084,622 daily, 2019-07-09 to 2026-07-08, 7.00 years | 38,586,501 five-minute, 2023-04-10 to 2026-07-08, 814 sessions |
| Selection | n/a | 11,340 in-play stock-sessions, median 13 per session |
| News | 714,126 rows over the full 7 years | 165,142 rows, median 204 articles per company |

Key artifacts:

- `data/canonical/swing_memberships_verified_20190709_20260708_v2.parquet`
- `data/canonical/edge_rebuild_intraday_5m_20260731/{regular,extended}/5m/`
- `data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3/bars/`
- `data/research/intraday_universe_selection_20230410_20260708_v2/`
- `data/raw/edge_rebuild_selected_session_5m_20260731` — 790/790 units,
  875,425 rows, 532 symbols, zero failures, authority present

## What Failed, And Why It Matters

The first swing strategy was rejected. It earned +0.724% gross and +0.524% net
per ten-day trade — a real edge that survives costs — but SPY excess was
-0.180%, losing to simply holding the index in four of seven years.

**The rejection was measured on a flawed test and does not mean the signal is
dead.** The population applied a ten-per-day cap ranked by *dollar volume*, so
the tested portfolio was roughly ten arbitrary qualifying S&P stocks per day. An
unranked basket of index constituents approximately reproduces the index, so
measuring its excess return mostly measured costs. It could not have passed.

Two consequences drive everything below:

1. The strategy must **rank** stocks and hold the top ones, not filter them and
   pick arbitrarily. Standard practice for cross-sectional strategies is
   learning-to-rank; the repo's own earlier V3 work already had a grouped ranker.
2. Features must be expressed **relative to other stocks at the same moment**,
   or a model learns "buy when the indicator is high", which fires on everything
   at once and reproduces the index.

## Built And Verified

`edge_rebuild/labeling.py` — two labels on every row, 14 tests.

- Barrier label: from the entry price, did the position reach its target, hit
  its stop, or survive to expiry. Entry is the next session's open so no part of
  the decision bar prices the fill. A bar whose range spans both barriers
  resolves to the **stop**, because the bar proves both prices traded but not in
  what order. A horizon running past available data is left **unresolved**, not
  labelled a timeout, which would invent an observation.
- Rank label: top fifth, bottom fifth, middle, computed inside one session and
  inside one sector. A cross-section below 50 rows yields no labels.
- The contract refuses either alone. Barrier-only reproduces the failure above.
  Rank-only assumes a position is held to expiry however far it moves against it.

`edge_rebuild/cross_sectional.py` — feature scaling, 11 tests.

- Z-score against the same timestamp's cross-section, tails winsorised first so
  one stock halving cannot dominate the mean and flatten every other score.
- Rank within the cross-section, centred at zero, immune to outliers.
- Sector-relative z-score, without which a sector-wide rally reads as stock
  selection.
- Every group is one timestamp. A test asserts that adding a later session
  leaves earlier scores bit-identical; if it ever fails, future data is leaking.

`edge_rebuild/technical_relationships.py` - causal path relationships, 7 tests.

- Five-bar RSI pivot divergence is emitted only after both following bars have
  completed. Bullish and bearish paths are tested separately.
- Normalized On-Balance Volume measures whether volume confirms or contradicts
  the price path.
- Kaufman's Efficiency Ratio separates directional movement from a noisy range;
  RSI is conditioned on both states rather than converted into threshold flags.
- Intraday grouping includes the exchange session, so all state resets
  overnight. Appending future bars leaves earlier features bit-identical.

`edge_rebuild/swing_features.py` - two-stage ranking panel, implementation
commit `fccda19`.

- Stage one is safe to run in bounded security batches. It reuses the canonical
  daily history, exact next-open labels, residual-strength components, and the
  shared technical relationship primitives.
- Target, stop, and timeout outcomes come from the existing conservative
  triple-barrier evaluator. Managed return is calculated from its actual exit
  price, not from the pre-existing timeout-close column. The outcome records
  when its exit session became available.
- Stage two requires the complete tradable population. It excludes cold and
  unresolved rows, then adds same-session z-scores, centred ranks, sector
  z-scores, and the within-sector managed-return rank label. It refuses missing
  or unexpected securities, duplicate security/session rows, and mixed feature
  profiles.
- Momentum, trend, pullback, volume, relationship, and ticker-catalyst inputs
  have named schemas. Raw news counts remain audit inputs only; the estimator
  schema exposes their normalized same-session derivatives.
- Ticker catalyst aggregates must already carry canonical event-assignment
  lineage on the input decisions. Global context uses the canonical dataset
  arguments. This builder does not perform a second, untracked news join.
- Cross-sectional outputs are assembled as contiguous `float32` blocks. This
  removed DataFrame fragmentation and avoids repeating a large schema string on
  every one of the roughly one million rows.

## Swing Panel Is Published

`data/features/edge_rebuild_swing_panel_20190709_20260708_v1` has complete
authority. It contains 883,604 rows, 627 securities, and 1,759 sessions from
2019-07-09 through 2026-07-08. Of those rows, 733,515 are feature-eligible,
869,242 have resolved barrier outcomes, and 450,074 are rank-eligible. The
source replay removed 2,897 zero-volume provider placeholders.

Stage one consists of 20 hash-verified security shards. Stage two processes
eight disjoint session-year partitions, each containing the complete security
population for every date; it is never partitioned by security. The initial
all-years pandas pass was correctly refused at 8.06 GiB peak RSS. The published
pass peaked at 1.7724 GiB under the 3.25 GiB safety threshold. Immutable replay
verified all final partition hashes. Final manifest SHA-256:
`a939de80ba7cc76821dde07ad73d0e7edab0af1cbaa7b938816ad20a9dd6b55b`.

Verification: 751 tests passed with 85 existing warnings; Ruff, strict mypy
across 193 source files, compileall, and `git diff --check` passed. No Python
worker remains.

## Swing Technical Benchmark Failed

Commit `de37067` added the immutable audit and froze its equal-weighted,
directional technical score before reading outcomes. The report at
`data/reports/edge_rebuild_swing_ordering_20190709_20260708_v1` covers 1,500
sessions and failed two of four gates. Top-decile rows averaged +14.34 bps;
bottom-decile rows averaged +32.84 bps. The session-neutral mean spread was
-18.41 bps with a -1.355 ten-session Newey-West t-statistic. Although the
median was +17.28 bps and 53.53% of sessions were positive, adverse tails made
overall ordering negative. The exact equal-weight score and any portfolio ranked
by it are rejected. The result does not reject all nonlinear or grouped
relationships in the causal features and does not prohibit estimator fitting.

The report manifest SHA-256 is
`01e5795854c64260aea377f71226bc2684bf48010587913c1a9490ab60a63f3a`.
Never tune this score against its own failed full-sample result. Retain it as a
failed comparison baseline.

## PDF-Aligned Temporal Protocol

The unchanged source PDF is retained at
`docs/references/comprehensive_quantitative_trading_model_implementation_plan_intraday_and_swing.pdf`
with SHA-256
`ea8df4ad6f3c1d17666cadff2672887b01ee46211ef11a040715f9890cb2b75b`.
The repository-specific decisions are in
`docs/model_training_validation_protocol.md`.

For swing, all point-in-time S&P 500 rows from one decision date remain in one
fold. The primary design uses repeated five-year fit plus one-year validation
windows, followed by one locked temporal test year and a ten-session
purge/embargo. A security may occur in different chronological folds. A separate
deterministic unseen-security holdout measures transfer to names absent from
fitting. The first learned sequence is one global sector-aware barrier classifier
and one decision-group ranker; sector specialists are conditional later work.

The approved swing horizon is seven usable years: five years fit, one year
validation, and one locked test year. The preceding 250 sessions are warm-up
only. The current panel is sufficient for causal diagnostics but lacks the
pre-fit warm-up and first 29 fit sessions. A shorter fit is prohibited.

The outcome-blind temporal audit is published at
`data/research/edge_rebuild_swing_temporal_manifest_20260731_v1`. It freezes
one 1,260-session fit / 252-session validation window, ten-session embargoes,
250 warm-up sessions, the 2025-07-01 through 2026-06-30 locked test, and a
separate deterministic 20% `security_id` holdout policy. The audit read only
`session_date_et` and `decision_group_id` after verifying panel authority and
partition hashes. It reports `insufficient_history`: 279 required XNYS sessions
from 2018-05-29 through 2019-07-08 are absent. Peak working set was 0.285 GiB.
The manifest SHA-256 is
`8d073839eb31e1baa734c9068c957798f1884e9bb2bbe871d80be0b82affe6cf`.

The outcome-blind acquisition planner is implemented and published at
`data/research/edge_rebuild_swing_history_acquisition_20260731_v2`. Its
manifest SHA-256 is
`6a689951035fd8b236cc23ab5530a570158eda7d1c0d09eab8867da4eec6de48`.
It verified every upstream authority and did not read model outcomes. The result
is `official_source_reacquisition_required`: all 83 retained official S&P
release files fail their declared byte hashes. They are browser-saved documents,
not byte-identical collector responses, and none is reusable as immutable source
evidence. The planner emitted zero Alpaca units and explicitly records
`blocked_until_source_reacquisition`. Peak working set was 0.230 GiB.

Official reacquisition is now complete at
`data/raw/index_membership/spglobal_official_20180414_20260708_v1`. It contains
107 of 107 releases and nine discovery pages. Offline replay resumed every unit
with zero network requests. Manifest SHA-256:
`72579c0aa8a56def5be64937bb5010ccd1625fe01cd6a115a5e68befbd4c9417`.
Raw release-set SHA-256:
`8fe52d77b96cbab4a37fef5c5ee69e5a521c51c92f4a6f7cce59fcb0a24e3bb1`.
The separate offline event authority is complete at
`data/canonical/index_membership/spglobal_events_20180414_20260708_v1`.
Manifest SHA-256:
`682ef590ea3d2bbe86e7bedae0eb08ca0f4a19202946ed793eb5420d993e94e8`.
Event-set SHA-256:
`29ad7c04bbada385b4f3464cd63b2dd6c4af45daa33392cd26a5aa357369fc87`.
All 107 releases were replayed offline: 105 parsed, two explicit no-effective
events, zero unresolved, and zero conflicts. The extractor retained 305 source
assertions as 303 unique events. Tesla's 2020-12-21 addition has three official
supporting source hashes. The event verifier replayed the raw parent and every
event artifact successfully. The public reconstruction API accepts only the raw
and event authority directories, replays both verifiers, and preserves every
supporting source URL and SHA-256.

The ER3 implementation is aligned. `edge_rebuild/setup_economics.py` classifies
every gate as `readiness` or `baseline_economics` and exposes separate
`ready_for_modeling` and `baseline_economics_passed` decisions. The old
`admitted` field, universal veto, and unregistered retired attribution script
are removed. Negative deterministic economics remain visible evidence but do
not make an otherwise causal, sufficiently large population model-unready.

Implementation commit `84afb07` is pushed. Current identities:

- strategy contract:
  `a77ccb4d635cc604b49b9c8bd5277aa281ca36afaf25abbe14c1c44d45b6460a`;
- deterministic baseline configuration:
  `761c57dd7248560da36620295e86f466c6c3daae04b8389a291c63ab185a31df`;
- retained source PDF:
  `ea8df4ad6f3c1d17666cadff2672887b01ee46211ef11a040715f9890cb2b75b`.

Verification passed 756 tests with 85 existing warnings, repository-wide Ruff,
strict mypy across 194 source files, compileall, and `git diff --check`. The
focused ER3, strategy-contract, architecture, and temporal-grouping battery
passed 71 tests. No Python process remained.

Ordering-gate verification passed 754 tests with 85 existing warnings, Ruff,
strict mypy across 194 source files, compileall, and `git diff --check`.

## Contract, As Frozen

Names are `swing` and `intraday`. No version suffixes — nothing is in
production, so there is no promoted artifact a rename could invalidate.

Swing: enter next session open; exit at target, stop, or the tenth session
close, whichever first; 3.0 / 1.5 daily ATR(14); same-bar ties resolve stop
first; hourly features over daily context; 25 positions; 20% sector cap;
sector-neutral scoring; 20 bps round trip.

Intraday: volume bars built from one-minute input, ~78 per session sized off
trailing median session volume; rolling features reset overnight; 2.0 / 1.5
five-minute ATR(14); 10 bps round trip; opening / midday / late segments.

Intraday universe: not index-restricted; average volume at least 1M shares over
20 sessions; price 8 to 500 dollars; relative volume at least 2.0 measured from
prior sessions only; at most 30 candidates per session; exchange-traded products
excluded because a fund has no issuer and a catalyst setup has nothing to
condition on — measured density is a median of 8 articles for funds against 204
for operating companies.

Validators refuse: random cross-validation, raw news counts as estimator
features, a widened experiment budget, an index-restricted intraday universe, a
stop below one average range, a daily ATR on a thirty-minute hold, clock bars
for intraday decisions, volume bars from coarser than one-minute input, rolling
windows spanning the overnight gap, and either label scheme alone.

Technical relationship semantics are frozen in the same contract: a two-bar
span on each side of the RSI pivot, 20-bar normalized OBV, and 20-bar Kaufman
Efficiency Ratio. The same contract now freezes 1% within-session
winsorization plus z-score, centred-rank, and sector-relative outputs. The
current contract hash is
`a77ccb4d635cc604b49b9c8bd5277aa281ca36afaf25abbe14c1c44d45b6460a`.
The superseded temporal-manifest contract hash
`0b9c2c71f43460f76ac61ee8bbb9e002f00633acb7c25d7b49ddec2e828e1f3c`
required three validation years and therefore exceeded the approved seven-year
swing horizon. The active contract separates one full swing validation year
from the unchanged four-fold intraday requirement.
The prior
`16709f3686ec737caa206dc1f45a80ea24f61d2dd1d18ded0b78cf978a433e38`
identity remains bound only to artifacts produced before the ER3 semantic
correction.

Data-specific exclusions are settled: remove the complete security and record
the reason and dates, then continue while no more than 5% of the filtered
point-in-time universe is lost. Refuse above 5%. Do not investigate isolated
names beyond producing that evidence. Missing SPY/sector benchmarks or
market-wide sessions are structural gaps and cannot be waived by this rule.

## Intraday Corpus Is Published

`data/canonical/edge_rebuild_intraday_5m_20260731`, authority complete, 1.5 GiB.
1,104 regular per-symbol files and 573 extended. 38,586,501 rows: 32,506,506
regular, 3,190,687 pre-market, 2,889,308 post-market, over 814 sessions.
Availability is exactly sixty seconds after each bar's end on every row in both
eras. Two isolated truncations are recorded and tolerated. The authority
manifest hash is
`f71d25ec1a98d38b75a3175a1508f8529426623857a09f255aeacc7bd19db0e0`.
All 1,677 registered file hashes replayed exactly, and a full Parquet scan found
no regular/extended segment contamination.

Supersedes `edge_rebuild_intraday_5m_20260730`, which lacked the screened
non-index names.

The build refused three times before publishing. Each refusal was a check
assuming a property the data no longer had, and each fix narrowed what the gate
asks without narrowing what it can catch:

1. **Duplicate bars.** Sources overlap by design; a symbol can be an index
   member with legacy coverage and also be selected by the screen. Identical
   deliveries now collapse to the collected copy, which carries observed rather
   than derived timestamps. Deliveries disagreeing on price or volume still
   refuse, because one instant cannot have traded at two prices.
2. **Eleven false identity breaks.** The level test compared consecutive rows,
   which assumed continuous coverage. Screened symbols are collected only on
   selected sessions, so consecutive rows can sit months apart; FTAI moved 1.09x
   on the day the check reported 3.41x. The test now applies only between
   sessions a few calendar days apart. A symbol that genuinely changed hands is
   still caught by the gap test, which caught FI independently.
3. **One false fabrication.** STRC printed a whole session at 95.99 on 15.8
   million shares, a par-pegged preferred trading heavily at a stable price. A
   frozen price now requires the absence of trades; zero volume remains the
   placeholder signature.

**The close-ratio ceiling was deliberately left at 3.0.** CAPR rose 4.71x on 33
times median volume and SRRK 4.62x on 44 times, both real news moves in the
high-beta names the broad universe exists to hold. Widening a threshold to admit
the data it just rejected is how a gate stops meaning anything. If such moves
recur, the correct answer is a volume-confirmed exemption, not a bigger number.

## Exact Next Steps

1. **Complete transition evidence.** Publish independent
   transition authority covering 2018-05-29 through 2026-07-08, bind raw-source,
   event, and transition hashes. In particular, resolve the 2019 Fox temporary
   symbols without chaining same-time transitions.
2. **Rebuild point-in-time membership.** Bind the raw, event, transition,
   cutoff-anchor, and universe hashes, then prove exact cutoff-anchor replay.
3. **Rerun the acquisition planner.** Require
   `ready_for_daily_bar_collection`; any blocker still means zero Alpaca calls.
4. **Extend swing raw history to the frozen 2,033-session target.** Collect only
   the published exact SIP/all units, then rebuild the causal panel
   with the same contracts so the 250-session warm-up does not consume the
   required five-year fit, one-year validation, and one-year locked test.
5. **Run training-only feature diagnostics.** Measure rank IC, stability,
   redundancy, missingness, and sector/regime behavior within development folds;
   never select features on the locked test.
6. **Train the bounded global models sequentially.** Compare deterministic and
   logistic baselines with a barrier classifier and decision-group ranker under
   the same folds, labels, costs, and top-k policy.
7. **Evaluate once.** Select with validation, then open the locked temporal test
   once; report session-block economics and sector/regime breakdowns. Keep the
   independent unseen-security result separate.
8. **Build the intraday equivalent separately.** Use complete sessions,
   one-minute executable paths, five-minute or volume-bar decision features,
   minute-duration purging, and overnight isolation.

## Not Started

- Hourly swing bars. The contract specifies hourly features; only daily exists.
  About 6M bars, one to two hours unattended.
- One-minute intraday bars, required before volume bars can be built. Roughly
  790 units over the in-play sessions. Time bars remain in use until then and
  this is recorded, not silently substituted.
- Any model under the PDF-aligned protocol. Nothing has been trained in this
  program.

The five-minute merge is complete and must not be rerun unless a reproducible
artifact or invariant failure is recorded.

## Verification Commands

Check real exit codes on each.

```powershell
Set-Location C:\project\market-predictor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check --no-cache .
.\.venv\Scripts\mypy.exe --strict --python-version 3.14 src\market_predictor
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short --branch
```

The local NumPy stubs use syntax newer than the project mypy 3.11 target, so the
verified strict command uses the installed Python 3.14 environment. That is an
environment fact, not permission to write Python-3.14-only source.

Last full verification after implementation commit `18bb896`: 768 tests passed
with 85 existing warnings; Ruff passed; strict mypy passed across 196 source
files; compileall and `git diff --check` passed; no Python worker remained.

## Standing Prohibitions

- No broker credential. This repository emits predictions; `trading_flow` owns
  execution, orders, positions, and sizing.
- No current screener value in a historical decision. Finviz numbers choose
  which tickers to consider and nothing more.
- Exclude a security only for unprovable point-in-time identity, never for thin
  trading or delisting.
- Extended-hours bars never reach a regular-session VWAP, moving average, ATR,
  or relative-volume denominator.
- Segment classification uses real exchange session bounds, never clock times.
  An early close at 13:00 ET would otherwise file three hours of post-market
  bars as regular session.
- No gate is weakened to make a build or an evaluation pass. A reproducible
  rejection is a valid outcome.
