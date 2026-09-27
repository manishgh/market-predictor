# Handoff To The TradingFlow Developer

Audience: the developer working in `C:\project\trading_flow` (Astra).
Counterpart: the Market Predictor developer working in `C:\project\market-predictor`.
The two developers never message each other directly; the user relays, and these files
are the shared record.

## Action required (2026-09-27 correction)

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
| `C:\project\market-predictor\tests\fixtures\contracts\swing_prediction_response.json` | A real served response covering a selected setup (`T000`), a ranked candidate (`T059`), a member with incomplete inputs (`T060`) and an unknown ticker (`MISSING`). Per-call values (request and correlation IDs, generation time, server path) are fixed. |
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
- Versions (user rule, September 27). Only the public API is versioned (contract `market_predictor.prediction.v3`, routes
under `/v1/`). Models are not final, so model, record, class, file and other internal
names carry no version number (`V2`, `.v3`, `_v1`), in TradingFlow as in
  Market Predictor. Names bound in already-published, hash-pinned evidence stay as
  recorded.

## Tasks

1. Contract test from the golden fixture.
   - Copy the fixture into the TradingFlow test project and record its source path and
     SHA-256 (`360206c2440e4fb39bac210a50b12c6d233e4f49a140fd356e5d283b034536cb`) beside it.
   - Parse it through the production deserializer and `Validate` path of
     `MarketPredictorHttpClient` for each of the four tickers, asserting the expected
     result: `T000` and `T059` available; `T060` and `MISSING` available with swing
     evidence, `final_signal` `abstain` and their abstention reason in `Errors`.
   - Add a local check that the copy equals the source file when
     `C:\project\market-predictor` exists, so a regenerated fixture cannot drift silently.
2. Contract version. `MarketPredictorHttpClient.cs` lines 336 and 400 hard-code
   `market_predictor.prediction.v1`. Read `contract_version` from the response, accept
   only `market_predictor.prediction.v3`, and treat anything else as `incompatible`.
3. Signal vocabulary. `UniverseRankService.cs` (`SupportiveSignals`, `OpposedSignals`,
   about lines 130-150) lists retired names and the action `watch_for_entry`, so every
   current signal reads as neutral. Map the current `final_signal` values from the
   contract's Signals and actions table: `positive_setup` supportive; `low_probability`
   opposed; `ranked_candidate`, `neutral` and `abstain` neutral. It stays display-only.
4. Abstentions. Show `live_inputs_incomplete` ("inputs incomplete at the decision")
   separately from `out_of_universe` ("not in the point-in-time universe") wherever
   predictor evidence is displayed.
5. Verification: `MarketPredictorHttpClientTests` plus the tests affected by tasks 2-4,
   and the affected project builds with nullable and analyzer checks. Record the tier,
   commands and results in TradingFlow's own handoff, then write an acknowledgement of the
   2026-09-27 Change log entries in your notes file.
