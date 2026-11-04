"""Bounded adjusted predictor inputs; no source admission, labels or publication."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.joins import decisions_from_completed_bars, join_universe_membership
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.datasets.feature_history_plan import DECISION_START, NUMERIC_END, WARMUP_START
from market_predictor.swing.datasets.session_requirements import expected_ticker_sessions
from market_predictor.swing.features.eligibility import apply_sparse_session_gap_abstentions
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from market_predictor.swing.features.predictors import build_swing_predictor_history
from market_predictor.swing.features.research_join import DECISION_KEYS, _clocks, _identities

BAR_COLUMNS = ("ticker", "timeframe", "bar_start_utc", "bar_end_utc", "available_at_utc",
    "open", "high", "low", "close", "volume", "price_feed", "adjustment", "schema_version")
_CONTEXT = ("sector", "primary_benchmark", "membership_available_at_utc", "membership_effective_from_utc",
    "membership_effective_to_utc", "universe_snapshot_id")


@dataclass(frozen=True)
class AdjustedTechnicalSource:
    rows: pd.DataFrame
    availability_columns: Mapping[str, str]


def expected_adjusted_history_sessions(
    *, security_id: str, memberships: pd.DataFrame,
    sparse_missing_sessions_by_ticker: Mapping[str, Sequence[date]],
) -> tuple[date, ...]:
    """Derive required history from independently verified membership intervals.

    Supply verified original memberships and any independently recorded gap inventory,
    never infer membership from returned query rows. Corrected memberships are supplied
    separately to the feature builder; query warmup cannot extend membership.
    """
    if not {"security_id", "ticker", "effective_from_utc", "effective_to_utc"}.issubset(memberships):
        raise DataReadinessError("adjusted history requirements lack membership intervals")
    calendar = tuple(xcals.get_calendar("XNYS").sessions_in_range(WARMUP_START, NUMERIC_END).date)
    owned = memberships.loc[memberships.security_id.eq(security_id)]
    if owned.empty:
        raise DataReadinessError("adjusted history requirements have no issuer membership")
    required: set[date] = set()
    for ticker in owned.ticker.unique():
        gaps = {day for day in sparse_missing_sessions_by_ticker.get(str(ticker), ()) if WARMUP_START <= day <= NUMERIC_END}
        observed_requirement = expected_ticker_sessions(str(ticker), memberships=memberships,
            benchmark_tickers=(), benchmark_start_sessions={}, all_sessions=calendar, session_abstentions=gaps)
        issuer_requirement = expected_ticker_sessions(str(ticker), memberships=owned,
            benchmark_tickers=(), benchmark_start_sessions={}, all_sessions=calendar)
        required.update((observed_requirement | gaps) & issuer_requirement)
    if not required:
        raise DataReadinessError("adjusted history requirements have no bounded sessions")
    return tuple(sorted(required))


def _bars(frame: pd.DataFrame, *, history_start: date = WARMUP_START, history_end: date = NUMERIC_END) -> pd.DataFrame:
    if not frame.columns.is_unique or not set(BAR_COLUMNS).issubset(frame):
        raise DataReadinessError("adjusted source requires canonical daily bars")
    data = frame.loc[:, list(BAR_COLUMNS)].copy()
    if not data.timeframe.eq("1d").all() or not data.price_feed.eq("sip").all() or not data.adjustment.eq("all").all():
        raise DataReadinessError("technical history must be consistently all-adjusted SIP daily bars")
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc"):
        data[column] = _clocks(data[column], column)
        if data[column].isna().any():
            raise DataReadinessError("adjusted bar clock missing")
    data["session_date_et"] = data.bar_start_utc.dt.tz_convert("America/New_York").dt.date
    if not data.session_date_et.between(history_start, history_end).all():
        raise DataReadinessError("technical history escapes warmup/initial-fit numeric bounds")
    if data.duplicated(["ticker", "session_date_et"]).any():
        raise DataReadinessError("adjusted history duplicates a session")
    values = data.loc[:, ["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(values.to_numpy(float)).all() or values.le(0).any().any()
            or values.high.lt(values[["open", "close", "low"]].max(axis=1)).any()
            or values.low.gt(values[["open", "close", "high"]].min(axis=1)).any()
            or data.available_at_utc.lt(data.bar_end_utc).any()):
        raise DataReadinessError("adjusted history has invalid or placeholder OHLCV/clocks")
    data.loc[:, values.columns] = values
    return data.sort_values(["session_date_et", "ticker"], kind="stable").reset_index(drop=True)


def build_adjusted_technical_source(
    decisions: pd.DataFrame, adjusted_bars: pd.DataFrame, benchmark_bars: pd.DataFrame,
    memberships: pd.DataFrame, raw_decision_bars: pd.DataFrame, *, security_id: str,
    expected_history_sessions: tuple[date, ...], contract: StrategyContract,
    history_start: date = WARMUP_START, history_end: date = NUMERIC_END,
    observation_cutoff: pd.Timestamp | None = None,
) -> AdjustedTechnicalSource:
    """Build one independently source-bound issuer, retaining every parent decision.

    The caller binds provider/asof streams to security_id before this call and
    supplies corrected canonical membership and raw decision-bar identities.
    Expected sessions derive the original membership history with
    expected_adjusted_history_sessions; never supply a blanket IPO calendar.
    Missing sessions mask the configured clean warmup window using the existing
    sparse-gap policy. No missing prices are imputed or history windows spliced.
    """
    identity = _identities(decisions, "adjusted source decisions").reset_index()
    if identity.empty or set(identity.security_id) != {security_id}:
        raise DataReadinessError("adjusted source requires exactly one canonical issuer")
    identity["session_date_et"] = identity.decision_time_utc.dt.tz_convert("America/New_York").dt.date
    if history_start > history_end or history_start < WARMUP_START:
        raise DataReadinessError("technical source history bounds are invalid")
    calendar = tuple(xcals.get_calendar("XNYS").sessions_in_range(history_start, history_end).date)
    expected = expected_history_sessions
    if (not expected or tuple(sorted(set(expected))) != expected or not set(expected).issubset(calendar)
            or not identity.session_date_et.between(DECISION_START, history_end).all()
            or not set(identity.session_date_et).issubset(expected)):
        raise DataReadinessError("adjusted source expected history or decision window is invalid")
    parent = join_universe_membership(identity, memberships)
    if set(parent.security_id) != {security_id}:
        raise DataReadinessError("corrected membership differs from canonical decision identity")
    parent = parent.set_index("decision_id").loc[identity.decision_id].reset_index()
    stock = _bars(adjusted_bars, history_start=history_start, history_end=history_end)
    benchmarks = _bars(benchmark_bars, history_start=history_start, history_end=history_end)
    if observation_cutoff is not None:
        observation_cutoff = pd.Timestamp(observation_cutoff)
        if (observation_cutoff.tzinfo is None or not identity.decision_time_utc.eq(observation_cutoff).all()
                or stock.available_at_utc.gt(observation_cutoff).any()
                or benchmarks.available_at_utc.gt(observation_cutoff).any()):
            raise DataReadinessError("current forecast history must be observed by its one decision cutoff")
    if len(set(stock.ticker)) > 1:
        raise DataReadinessError("one issuer must use one coherent provider/asof stream")
    if not set(stock.session_date_et).issubset(expected):
        raise DataReadinessError("adjusted stock bars escape independently expected history")
    own_memberships = memberships.loc[memberships.security_id.eq(security_id)]
    required_benchmarks = {"SPY", "QQQ", *own_memberships.primary_benchmark.dropna()}
    if not set(benchmarks.ticker).issubset(required_benchmarks):
        raise DataReadinessError("adjusted source received unrelated benchmarks")
    rows = parent.loc[:, [*DECISION_KEYS, "session_date_et", *_CONTEXT]].copy()
    rows["feature_profile"] = "technical_market"
    rows["feature_eligible"] = False
    rows["daily_bar_count"] = 0
    rows[list(TECHNICAL_RANKING_FEATURES)] = np.nan
    rows["technical_available_at_utc"] = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
    reasons: list[tuple[str, ...]] = []
    missing_stock = set(expected) - set(stock.session_date_et)
    missing_bench = {symbol: set(expected) - set(benchmarks.loc[benchmarks.ticker.eq(symbol), "session_date_et"])
        for symbol in required_benchmarks}
    stock_ready = _gap_ready(rows, rows.ticker, {ticker: missing_stock for ticker in set(rows.ticker)}, calendar, contract)
    benchmark_ready = pd.Series(True, index=rows.index)
    for symbols in (pd.Series("SPY", index=rows.index), pd.Series("QQQ", index=rows.index), rows.primary_benchmark):
        benchmark_ready &= _gap_ready(rows, symbols, missing_bench, calendar, contract)
    for index in rows.index:
        missing = []
        if not stock_ready[index]:
            missing.append("adjusted_history_warmup_crosses_missing_session")
        if not benchmark_ready[index]:
            missing.append("adjusted_benchmark_warmup_crosses_missing_session")
        reasons.append(tuple(missing))
    if not stock.empty and not benchmarks.empty and any(not reason for reason in reasons):
        history = decisions_from_completed_bars(stock,
            mode="swing-nightly" if observation_cutoff is None else "research-bar-availability")
        if observation_cutoff is not None:
            # These are price-history positions for ONE current forecast, never
            # historical predictions. Keep the actual bar observation clocks.
            history["decision_time_utc"] = observation_cutoff
            history["decision_group_id"] = observation_cutoff.isoformat()
            history["prediction_cutoff_policy_id"] = "current_forecast_history"
        history = _history_memberships(history, own_memberships, security_id, observation_cutoff=observation_cutoff)
        history["feature_profile"] = "technical_market"
        features = build_swing_predictor_history(history, benchmarks, contract=contract).set_index("session_date_et")
        for index, row in rows.iterrows():
            if reasons[index] or row.session_date_et not in features.index:
                continue
            values = features.loc[row.session_date_et]
            rows.loc[index, list(TECHNICAL_RANKING_FEATURES)] = values.loc[list(TECHNICAL_RANKING_FEATURES)].to_numpy()
            rows.loc[index, "daily_bar_count"] = values.daily_bar_count
            rows.loc[index, "feature_eligible"] = bool(values.feature_eligible)
            dependencies = [stock.loc[stock.session_date_et.le(row.session_date_et), "available_at_utc"].max(),
                benchmarks.loc[benchmarks.session_date_et.le(row.session_date_et)
                    & benchmarks.ticker.isin(("SPY", "QQQ", row.primary_benchmark)), "available_at_utc"].max(),
                row.membership_available_at_utc]
            rows.loc[index, "technical_available_at_utc"] = max(dependencies)
    _raw_dollar_volume(rows, raw_decision_bars, reasons)
    for index, row in rows.iterrows():
        if not row.feature_eligible and not reasons[index]:
            reasons[index] = ("technical_warmup_or_unavailable",)
        if reasons[index]:
            rows.loc[index, "feature_eligible"] = False
        if pd.notna(row.technical_available_at_utc):
            if row.technical_available_at_utc > row.decision_time_utc:
                raise DataReadinessError("technical source availability exceeds canonical cutoff")
    rows["technical_missing_reasons"] = reasons
    clocks = {name: "technical_available_at_utc" for name in TECHNICAL_RANKING_FEATURES}
    clocks["dollar_volume_log"] = "raw_dollar_volume_available_at_utc"
    return AdjustedTechnicalSource(rows, clocks)


def _history_memberships(history: pd.DataFrame, memberships: pd.DataFrame, security_id: str, *,
                         observation_cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    # Validate with the canonical owner even when all warmup membership is unknown.
    join_universe_membership(history.iloc[:0], memberships)
    candidates = history.copy()
    matches = pd.Series(0, index=candidates.index)
    cutoff = candidates.decision_time_utc if observation_cutoff is None else swing_prediction_cutoffs(candidates.session_date_et)
    for record in memberships.to_dict(orient="records"):
        start = pd.Timestamp(record["effective_from_utc"])
        available = pd.Timestamp(record["available_at_utc"])
        end = record["effective_to_utc"]
        known = cutoff.ge(available) if observation_cutoff is None else pd.Series(available <= observation_cutoff, index=candidates.index)
        selected = cutoff.ge(start) & known
        if pd.notna(end):
            selected &= cutoff.lt(pd.Timestamp(end))
        matches.loc[selected] += 1
        candidates.loc[selected, "ticker"] = record["ticker"]
        if observation_cutoff is not None:
            for name in _CONTEXT:
                source = {"membership_available_at_utc": "available_at_utc", "membership_effective_from_utc": "effective_from_utc",
                          "membership_effective_to_utc": "effective_to_utc"}.get(name, name)
                candidates.loc[selected, name] = record[source]
    if matches.gt(1).any():
        raise DataReadinessError("historical issuer membership is ambiguous at a warmup cutoff")
    joined = (join_universe_membership(candidates.loc[matches.eq(1)], memberships)
              if observation_cutoff is None else candidates.loc[matches.eq(1)])
    payload = joined.loc[:, ["session_date_et", "ticker", *_CONTEXT]]
    result = history.drop(columns="ticker").merge(payload, on="session_date_et", how="left", validate="one_to_one")
    result["ticker"] = result.ticker.fillna(str(history.ticker.iloc[0]))
    result["security_id"] = security_id
    return result


def _gap_ready(rows: pd.DataFrame, tickers: pd.Series, missing: Mapping[str, set[date]],
    calendar: tuple[date, ...], contract: StrategyContract,
) -> pd.Series:
    # Calendar-only input to the shared mask; no benchmark observation is created.
    market_calendar = pd.DataFrame({"ticker": contract.labels.benchmark_market, "session_date_et": calendar})
    projection = rows.loc[:, ["ticker", "session_date_et"]].assign(ticker=tickers, feature_eligible=True, label_eligible=False)
    masked = apply_sparse_session_gap_abstentions(projection, benchmark_bars=market_calendar,
        sparse_missing_sessions_by_ticker={ticker: tuple(sorted(days)) for ticker, days in missing.items() if days},
        contract=contract)
    return masked.sparse_gap_feature_eligible


def _raw_dollar_volume(rows: pd.DataFrame, raw: pd.DataFrame, reasons: list[tuple[str, ...]]) -> None:
    required = {*DECISION_KEYS, "close", "volume", "available_at_utc", "price_feed", "adjustment"}
    if not required.issubset(raw):
        raise DataReadinessError("raw dollar volume requires exact canonical decision-bar identities")
    identity = _identities(raw, "raw dollar volume")
    parent = _identities(rows, "technical rows")
    if not set(identity.index).issubset(parent.index) or not identity.eq(parent.loc[identity.index]).all().all():
        raise DataReadinessError("raw dollar volume decision identity differs")
    raw = raw.set_index("decision_id").reindex(rows.decision_id)
    close, volume = pd.to_numeric(raw.close, errors="coerce"), pd.to_numeric(raw.volume, errors="coerce")
    available = _clocks(raw.available_at_utc, "raw dollar volume availability")
    valid = (raw.price_feed.eq("sip") & raw.adjustment.eq("raw") & close.gt(0) & volume.gt(0)
        & np.isfinite(close * volume) & available.notna()
        & available.le(pd.Series(rows.decision_time_utc.to_numpy(), index=raw.index)))
    rows["dollar_volume_log"] = np.log1p(close * volume).where(valid).to_numpy()
    rows["raw_dollar_volume_available_at_utc"] = available.where(valid).to_numpy()
    for index, good in enumerate(valid):
        if not good:
            reasons[index] = (*reasons[index], "missing_or_invalid_raw_dollar_volume")
