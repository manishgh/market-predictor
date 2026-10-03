# Handoff To The TradingFlow Developer

Audience: the developer working in `C:\project\trading_flow` (Astra).
Counterpart: the Market Predictor developer working in `C:\project\market-predictor`.
The two developers never message each other directly; the user relays, and these files
are the shared record.

## October 3 V1 reset; consumer adoption pending

User instruction: the initial complete system is V1. Do not increment API, data or
model versions for implementation iterations. Internal identities are now descriptive
and unversioned; no compatibility aliases are required. Work is on
`codex/v1-canonical-cleanup`, not merged to main as a trading-performance improvement.

Market Predictor now serves `contract_version = market_predictor.prediction.v1`
under the existing `/v1/` routes. Current served fixture SHA-256:
`59288471c249b0422443f0a68ce7623f4b03956826877ad1158dc79ebf1fea51`.
Training/policy schema cleanup changes derived hashes, not model approval or results.

Read-only inspection found TradingFlow's separately edited `unified-swing-product`
checkout uses `JsonPropertyName("contract")` and accepts
`market_predictor.prediction`. Its developer must change the JSON property to
`contract_version`, accept V1 only, copy/pin the new fixture and run the deserializer,
version-rejection and advisory-isolation tests. Missing/incompatible responses must
remain unavailable. No TradingFlow file, process or build was changed here.

The current candle recommendation and V1 candle-publication boundary are in
`docs/active_edge_rebuild_plan.md`, section "Shared Candle Flow And V1 API
Recommendation". Daily model inputs, separate daily/entry warm-up windows and retained
strategy-specific execution intervals are recommendations, not activated settings.
No 63/252-session forecast endpoint has been implemented by this cleanup.

All entries below are historical receipts/tasks, superseded by this V1 instruction.

## Earlier October 3 producer change; historical

The user says TradingFlow is undergoing separate changes. Do not modify or build it
from this task. Market Predictor retains API v4 and adds optional model metadata
`training_labels_available_through_utc`; `training_data_end` now carries the verified
final-fit decision date. Signals/actions and existing field types are unchanged.
Replay requires the label timestamp strictly before the actual prediction-row decision,
and exposes `model_training_labels_available_through_utc` in its response.
The current producer fixture hash is
`bc1f5109be83fc2985c28b0c0ded75e0487e34a68c237a1f147babc8e117a08b`.
The TradingFlow developer must acknowledge and re-pin this fixture before relying on
the added field. No fresh C# build or consumer acceptance is claimed here.

## Historical action (2026-09-28 API v4; completed below)

Accept only `contract_version = market_predictor.prediction.v4`, retain the `/v1/`
routes, and re-pin `swing_prediction_response.json` to SHA-256 `96cbcd133b8e96253b034fabba10624be79b0d80550f043f3869905251bec522`.
The fixture now includes `T061`, a verified member abstaining as `sector_peer_floor`.
Display it as unavailable for ranking, never as out of universe or an actionable
signal. The unused row-evidence `decision_atr` field is removed; risk distances and
all scored prediction fields retain their meanings. No HTTP cross-section endpoint
is added. Acknowledge this change in the shared notes file.

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

## Historical action (2026-09-27 correction)

TradingFlow integrated the "later" change-log entry, which removed the API version.
The user then clarified that the public API stays versioned, so the current contract is
the "correction" entry: read `contract_version`, accept only
`market_predictor.prediction.v3`, and re-pin the fixture to SHA-256
`360206c2440e4fb39bac210a50b12c6d233e4f49a140fd356e5d283b034536cb`. The routes under
`/v1/` never changed. Acknowledge the correction in your notes file.

## Files

Read-only for TradingFlow (owned by Market Predictor):

| Path | What it is |
| --- | --- |
| `C:\project\market-predictor\docs\contracts\prediction_api.md` | The prediction API contract: endpoints, fields, nullability, signal/action values, abstention reasons, errors, change rules, pending changes and change log. |
| `C:\project\market-predictor\tests\fixtures\contracts\swing_prediction_response.json` | A real served response covering a selected setup (`T000`), a ranked candidate (`T059`), a member with incomplete inputs (`T060`), insufficient eligible sector peers (`T061`) and an unknown ticker (`MISSING`). Per-call values (request and correlation IDs, generation time, server path) are fixed. |
| `C:\project\market-predictor\docs\contracts\tradingflow_handoff.md` | This file. |

Owned by TradingFlow (create it; Market Predictor reads it):

| Path | What it is |
| --- | --- |
| `C:\project\trading_flow\docs\integration\market-predictor-notes.md` | Acknowledgements of contract change-log entries, questions and change requests, each dated. |

## Boundaries

- Edit only files under `C:\project\trading_flow`. Never edit, commit, run jobs in, or
  delete anything under `C:\project\market-predictor`; read it only.
- The contract belongs to Market Predictor. Do not change what TradingFlow sends or
  expects on the wire unless a Change log entry in `prediction_api.md` says so. To
  request a change, write it in your notes file and ask the user to relay it.
- Predictor output stays advisory: it cannot qualify, veto, prioritize for execution or
  authorize an order (`C:\project\trading_flow\AGENTS.md`, Candidate Discovery and
  Admission). Keep operational scores and orders independent of it.
- All of `C:\project\trading_flow\AGENTS.md` applies, in particular: no GitHub sync unless
  the user asks; preserve local secrets, `appsettings.local.json`, `tradingflow.db`,
  paper-trading state and cached candle data; stop TradingFlow runtime processes before
  code changes; verify under its Risk-Based Verification policy.
- Do not implement anything listed under Pending changes until it moves to the
  Change log.
- Versions (user rule, September 27). Only the public API is versioned (contract `market_predictor.prediction.v4`, routes
under `/v1/`). Models are not final, so model, record, class, file and other internal
names carry no version number (`V2`, `.v3`, `_v1`), in TradingFlow as in
  Market Predictor. Names bound in already-published, hash-pinned evidence stay as
  recorded.

## Tasks

1. Contract test from the golden fixture.
   - Copy the fixture into the TradingFlow test project and record its source path and
     SHA-256 (`96cbcd133b8e96253b034fabba10624be79b0d80550f043f3869905251bec522`) beside it.
   - Parse it through the production deserializer and `Validate` path of
     `MarketPredictorHttpClient` for each of the five tickers, asserting the expected
     result: `T000` and `T059` available; `T060`, `T061` and `MISSING` available with swing
     evidence, `final_signal` `abstain` and their abstention reason in `Errors`.
   - Add a local check that the copy equals the source file when
     `C:\project\market-predictor` exists, so a regenerated fixture cannot drift silently.
2. Contract version. Replace the retired unversioned contract. Read `contract_version` from the response, accept
   only `market_predictor.prediction.v4`, and treat anything else as `incompatible`.
3. Signal vocabulary. Preserve the existing `UniverseRankService` mapping:
   `positive_setup` supportive; `low_probability` opposed; `ranked_candidate`,
   `neutral` and `abstain` neutral. It stays display-only. Update synthetic test payloads
   to v4 so advisory-isolation checks exercise compatible responses.
4. Abstentions. Show `sector_peer_floor` ("too few eligible sector peers for ranking"), `live_inputs_incomplete` ("inputs incomplete at the decision")
   separately from `out_of_universe` ("not in the point-in-time universe") wherever
   predictor evidence is displayed.
5. Verification: `MarketPredictorHttpClientTests` plus the tests affected by tasks 2-4,
   and the affected project builds with nullable and analyzer checks. Record the tier,
   commands and results in TradingFlow's own handoff, then write an acknowledgement of the
   2026-09-28 API v4 Change log entry in your notes file.
