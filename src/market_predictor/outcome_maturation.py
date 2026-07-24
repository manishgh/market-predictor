from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TypeAlias, cast

import numpy as np
import pandas as pd

from market_predictor.label_paths import (
    evaluate_intraday_barrier_paths,
    evaluate_swing_paths,
)
from market_predictor.outcome_contracts import (
    MaturationAttemptV1,
    MaturedOutcomeV1,
    PredictionMaturationIntentV2,
    content_sha256,
)
from market_predictor.v3.errors import DataReadinessError

MaturationResult: TypeAlias = MaturationAttemptV1 | MaturedOutcomeV1
_BAR_COLUMNS = {
    "ticker",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_feed",
    "adjustment",
}


def mature_prediction(
    intent: PredictionMaturationIntentV2,
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    source_artifact_sha256: str,
) -> tuple[MaturationResult, list[dict[str, object]]]:
    observed = _aware_utc(observed_as_of)
    data = _prepare_bars(
        bars,
        observed_as_of=observed,
        source_artifact_sha256=source_artifact_sha256,
        required_price_feed=intent.price_feed,
    )
    if intent.view == "swing":
        return _mature_swing(intent, data, observed)
    return _mature_intraday(intent, data, observed)


def maturation_attempt(
    intent: PredictionMaturationIntentV2,
    *,
    observed_as_of: datetime,
    status: str,
    reasons: tuple[str, ...],
    missing_intervals: tuple[str, ...] = (),
) -> MaturationAttemptV1:
    if status not in {"pending", "blocked"}:
        raise ValueError("maturation attempt status must be pending or blocked")
    observed = _aware_utc(observed_as_of)
    base = {
        "contract_version": "market_predictor.maturation_attempt.v1",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "observed_as_of_utc": observed,
        "status": status,
        "reasons": reasons,
        "missing_intervals": missing_intervals,
    }
    return MaturationAttemptV1.model_validate({**base, "attempt_id": content_sha256(base)})


def _mature_swing(
    intent: PredictionMaturationIntentV2,
    bars: pd.DataFrame,
    observed: datetime,
) -> tuple[MaturationResult, list[dict[str, object]]]:
    policy = intent.label_policy
    _require_policy(policy, "policy", "swing_label.v2")
    horizon = _policy_int(policy, "horizon_sessions")
    spy_ticker = str(policy["broad_benchmark"]).upper()
    qqq_ticker = str(policy["growth_benchmark"]).upper()
    sessions = bars.loc[bars["ticker"].eq(spy_ticker), "session_date_et"].drop_duplicates().sort_values().tolist()
    if intent.decision_session_et not in sessions:
        return _pending(
            intent,
            observed,
            reasons=("decision_session_not_observed",),
        ), []
    decision_index = sessions.index(intent.decision_session_et)
    if decision_index + horizon >= len(sessions):
        return _pending(
            intent,
            observed,
            reasons=("horizon_not_complete",),
        ), []
    path_sessions = sessions[decision_index + 1 : decision_index + horizon + 1]
    entry_session = path_sessions[0]
    exit_session = path_sessions[-1]
    stock_path, missing = _daily_path(
        bars,
        ticker=intent.ticker,
        sessions=path_sessions,
    )
    benchmark_tickers = (spy_ticker, qqq_ticker, intent.primary_benchmark)
    benchmark_pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    for ticker in benchmark_tickers:
        entry = _one_daily_row(bars, ticker=ticker, session=entry_session)
        exit_row = _one_daily_row(bars, ticker=ticker, session=exit_session)
        if entry is None:
            missing.append(f"{ticker}:{entry_session}:entry")
        if exit_row is None:
            missing.append(f"{ticker}:{exit_session}:exit")
        if entry is not None and exit_row is not None:
            benchmark_pairs[ticker] = (entry, exit_row)
    if missing:
        return _pending(
            intent,
            observed,
            reasons=("required_bar_path_incomplete",),
            missing_intervals=tuple(sorted(missing)),
        ), []

    entry_price = float(stock_path.iloc[0]["open"])
    exit_price = float(stock_path.iloc[-1]["close"])
    _require_positive_prices(entry_price, exit_price)
    evaluated = evaluate_swing_paths(
        entry_price=np.asarray([entry_price]),
        exit_price=np.asarray([exit_price]),
        path_high=stock_path["high"].to_numpy(float)[None, :],
        path_low=stock_path["low"].to_numpy(float)[None, :],
        round_trip_cost_bps=_policy_float(policy, "round_trip_cost_bps"),
    )
    gross = float(evaluated.gross_return[0])
    net = float(evaluated.net_return[0])
    spy_return = _pair_return(benchmark_pairs[spy_ticker])
    qqq_return = _pair_return(benchmark_pairs[qqq_ticker])
    sector_return = _pair_return(benchmark_pairs[intent.primary_benchmark])
    evidence_frames = [stock_path]
    evidence_frames.extend(pd.DataFrame([entry, exit_row]) for entry, exit_row in benchmark_pairs.values())
    evidence_rows = _evidence_rows(evidence_frames)
    label_available = _max_available(evidence_frames)
    outcome = _outcome(
        intent,
        entry_time=_timestamp(stock_path.iloc[0]["bar_start_utc"]),
        exit_time=_timestamp(stock_path.iloc[-1]["bar_end_utc"]),
        label_available=label_available,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return=gross,
        net_return=net,
        mfe=float(evaluated.mfe[0]),
        mae=float(evaluated.mae[0]),
        path_outcome="positive" if net > 0 else "negative",
        opportunity_target=int(net > 0),
        downside_target=None,
        spy_return=spy_return,
        qqq_return=qqq_return,
        sector_return=sector_return,
        evidence_rows=evidence_rows,
    )
    return outcome, evidence_rows


def _mature_intraday(
    intent: PredictionMaturationIntentV2,
    bars: pd.DataFrame,
    observed: datetime,
) -> tuple[MaturationResult, list[dict[str, object]]]:
    policy = intent.label_policy
    _require_policy(policy, "policy", "intraday_label.v2")
    if intent.decision_atr is None:
        raise DataReadinessError("intraday intent has no decision ATR")
    horizon_minutes = _policy_int(policy, "horizon_minutes")
    execution_minutes = _policy_int(policy, "execution_bar_minutes")
    horizon_bars = horizon_minutes // execution_minutes
    expected_starts = [intent.decision_time_utc + timedelta(minutes=execution_minutes * offset) for offset in range(horizon_bars)]
    stock_path, missing = _intraday_path(
        bars,
        ticker=intent.ticker,
        expected_starts=expected_starts,
        session=intent.decision_session_et,
    )
    if missing:
        reason = "horizon_not_complete" if expected_starts[-1] >= observed else "required_bar_path_incomplete"
        return _pending(
            intent,
            observed,
            reasons=(reason,),
            missing_intervals=tuple(missing),
        ), []

    entry_price = float(stock_path.iloc[0]["open"])
    evaluated = evaluate_intraday_barrier_paths(
        path_open=stock_path["open"].to_numpy(float)[None, :],
        path_high=stock_path["high"].to_numpy(float)[None, :],
        path_low=stock_path["low"].to_numpy(float)[None, :],
        path_close=stock_path["close"].to_numpy(float)[None, :],
        entry_atr=np.asarray([intent.decision_atr]),
        target_atr=_policy_float(policy, "target_atr"),
        stop_atr=_policy_float(policy, "stop_atr"),
        round_trip_cost_bps=_policy_float(policy, "round_trip_cost_bps"),
    )
    path_outcome = str(evaluated.outcome[0])
    outcome_index = int(evaluated.outcome_offset[0])
    opportunity_target = int(evaluated.target_first[0])
    downside_target = int(evaluated.stop_first[0])
    active_path = stock_path.iloc[: outcome_index + 1]
    realized = float(evaluated.realized_price[0])
    gross = float(evaluated.gross_return[0])
    net = float(evaluated.net_return[0])
    entry_start = expected_starts[0]
    exit_start = expected_starts[outcome_index]
    benchmark_tickers = (
        str(policy["broad_benchmark"]).upper(),
        str(policy["growth_benchmark"]).upper(),
        intent.primary_benchmark,
    )
    benchmark_pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    benchmark_missing: list[str] = []
    for ticker in benchmark_tickers:
        entry = _one_intraday_row(bars, ticker=ticker, start=entry_start)
        exit_row = _one_intraday_row(bars, ticker=ticker, start=exit_start)
        if entry is None:
            benchmark_missing.append(f"{ticker}:{entry_start.isoformat()}:entry")
        if exit_row is None:
            benchmark_missing.append(f"{ticker}:{exit_start.isoformat()}:exit")
        if entry is not None and exit_row is not None:
            benchmark_pairs[ticker] = (entry, exit_row)
    if benchmark_missing:
        return _pending(
            intent,
            observed,
            reasons=("required_benchmark_path_incomplete",),
            missing_intervals=tuple(sorted(benchmark_missing)),
        ), []
    spy_ticker, qqq_ticker, sector_ticker = benchmark_tickers
    evidence_frames = [active_path]
    evidence_frames.extend(pd.DataFrame([entry, exit_row]) for entry, exit_row in benchmark_pairs.values())
    evidence_rows = _evidence_rows(evidence_frames)
    label_available = _max_available(evidence_frames)
    outcome = _outcome(
        intent,
        entry_time=_timestamp(active_path.iloc[0]["bar_start_utc"]),
        exit_time=_timestamp(active_path.iloc[-1]["bar_end_utc"]),
        label_available=label_available,
        entry_price=entry_price,
        exit_price=realized,
        gross_return=gross,
        net_return=net,
        mfe=float(evaluated.mfe[0]),
        mae=float(evaluated.mae[0]),
        path_outcome=path_outcome,
        opportunity_target=opportunity_target,
        downside_target=downside_target,
        spy_return=_pair_return(benchmark_pairs[spy_ticker]),
        qqq_return=_pair_return(benchmark_pairs[qqq_ticker]),
        sector_return=_pair_return(benchmark_pairs[sector_ticker]),
        evidence_rows=evidence_rows,
    )
    return outcome, evidence_rows


def _outcome(
    intent: PredictionMaturationIntentV2,
    *,
    entry_time: datetime,
    exit_time: datetime,
    label_available: datetime,
    entry_price: float,
    exit_price: float,
    gross_return: float,
    net_return: float,
    mfe: float,
    mae: float,
    path_outcome: str,
    opportunity_target: int,
    downside_target: int | None,
    spy_return: float,
    qqq_return: float,
    sector_return: float,
    evidence_rows: list[dict[str, object]],
) -> MaturedOutcomeV1:
    base = {
        "contract_version": "market_predictor.matured_outcome.v1",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "snapshot_id": intent.snapshot_id,
        "ticker": intent.ticker,
        "view": intent.view,
        "horizon": intent.horizon,
        "entry_time_utc": entry_time,
        "exit_time_utc": exit_time,
        "label_available_at_utc": label_available,
        "matured_at_utc": label_available,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross_return,
        "net_return": net_return,
        "mfe": mfe,
        "mae": mae,
        "path_outcome": path_outcome,
        "opportunity_target": opportunity_target,
        "downside_target": downside_target,
        "spy_return": spy_return,
        "qqq_return": qqq_return,
        "sector_return": sector_return,
        "excess_return_vs_spy": net_return - spy_return,
        "excess_return_vs_qqq": net_return - qqq_return,
        "excess_return_vs_sector": net_return - sector_return,
        "evidence_sha256": content_sha256(evidence_rows),
    }
    return MaturedOutcomeV1.model_validate({**base, "outcome_id": content_sha256(base)})


def _pending(
    intent: PredictionMaturationIntentV2,
    observed: datetime,
    *,
    reasons: tuple[str, ...],
    missing_intervals: tuple[str, ...] = (),
) -> MaturationAttemptV1:
    return maturation_attempt(
        intent,
        observed_as_of=observed,
        status="pending",
        reasons=reasons,
        missing_intervals=missing_intervals,
    )


def _prepare_bars(
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    source_artifact_sha256: str,
    required_price_feed: str,
) -> pd.DataFrame:
    missing = sorted(_BAR_COLUMNS.difference(bars.columns))
    if missing:
        raise DataReadinessError(f"maturation bars are missing columns: {', '.join(missing)}")
    if len(source_artifact_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_artifact_sha256):
        raise DataReadinessError("maturation source artifact identity is invalid")
    data = bars.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc"):
        data[column] = pd.to_datetime(data[column], errors="coerce", utc=True)
    if data[["bar_start_utc", "bar_end_utc", "available_at_utc"]].isna().any().any():
        raise DataReadinessError("maturation bars contain invalid timestamps")
    if "session_date_et" not in data:
        data["session_date_et"] = data["bar_start_utc"].dt.tz_convert("America/New_York").dt.date
    else:
        data["session_date_et"] = pd.to_datetime(
            data["session_date_et"],
            errors="coerce",
        ).dt.date
    if data["session_date_et"].isna().any():
        raise DataReadinessError("maturation bars contain invalid sessions")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if data[numeric].isna().any().any():
        raise DataReadinessError("maturation bars contain invalid OHLCV")
    if bool(data["available_at_utc"].lt(data["bar_end_utc"]).any()):
        raise DataReadinessError("maturation bar availability precedes bar completion")
    if bool(data["price_feed"].astype(str).str.upper().ne(required_price_feed.upper()).any()):
        raise DataReadinessError("maturation bars use an unexpected price feed")
    if bool(data["adjustment"].astype(str).str.lower().ne("all").any()):
        raise DataReadinessError("maturation bars are not fully adjusted")
    if bool(data.duplicated(["ticker", "bar_start_utc"]).any()):
        raise DataReadinessError("maturation bars contain duplicate ticker intervals")
    data = data[data["available_at_utc"].le(observed_as_of)].copy()
    data["source_artifact_sha256"] = source_artifact_sha256
    return data.sort_values(["ticker", "bar_start_utc"], kind="stable")


def _daily_path(
    bars: pd.DataFrame,
    *,
    ticker: str,
    sessions: list[object],
) -> tuple[pd.DataFrame, list[str]]:
    ticker_rows = bars[bars["ticker"].eq(ticker)]
    counts = ticker_rows.groupby("session_date_et").size()
    missing = [f"{ticker}:{session}" for session in sessions if int(counts.get(session, 0)) != 1]
    if missing:
        return pd.DataFrame(), missing
    rows = ticker_rows.set_index("session_date_et")
    selected = rows.loc[sessions].reset_index()
    return selected, []


def _intraday_path(
    bars: pd.DataFrame,
    *,
    ticker: str,
    expected_starts: list[datetime],
    session: object,
) -> tuple[pd.DataFrame, list[str]]:
    ticker_rows = bars[bars["ticker"].eq(ticker)].set_index("bar_start_utc")
    missing = [f"{ticker}:{start.isoformat()}" for start in expected_starts if pd.Timestamp(start) not in ticker_rows.index]
    if missing:
        return pd.DataFrame(), missing
    selected = ticker_rows.loc[pd.DatetimeIndex(expected_starts)].copy()
    selected.index.name = "bar_start_utc"
    selected = selected.reset_index()
    wrong_session = selected["session_date_et"].ne(session)
    if bool(wrong_session.any()):
        missing.extend(
            f"{ticker}:{start.isoformat()}:cross_session"
            for start in pd.to_datetime(
                selected.loc[wrong_session, "bar_start_utc"],
                utc=True,
            )
        )
        return pd.DataFrame(), missing
    return selected, []


def _one_daily_row(
    bars: pd.DataFrame,
    *,
    ticker: str,
    session: object,
) -> pd.Series | None:
    rows = bars[bars["ticker"].eq(ticker) & bars["session_date_et"].eq(session)]
    return rows.iloc[0] if len(rows) == 1 else None


def _one_intraday_row(
    bars: pd.DataFrame,
    *,
    ticker: str,
    start: datetime,
) -> pd.Series | None:
    rows = bars[bars["ticker"].eq(ticker) & bars["bar_start_utc"].eq(pd.Timestamp(start))]
    return rows.iloc[0] if len(rows) == 1 else None


def _pair_return(pair: tuple[pd.Series, pd.Series]) -> float:
    entry, exit_row = pair
    entry_price = float(entry["open"])
    exit_price = float(exit_row["close"])
    _require_positive_prices(entry_price, exit_price)
    return exit_price / entry_price - 1.0


def _evidence_rows(
    frames: list[pd.DataFrame],
) -> list[dict[str, object]]:
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "bar_start_utc"])
    combined = combined.sort_values(["ticker", "bar_start_utc"], kind="stable")
    columns = [
        "ticker",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "session_date_et",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "price_feed",
        "adjustment",
        "source_artifact_sha256",
    ]
    records: list[dict[str, object]] = []
    for record in combined[columns].to_dict(orient="records"):
        records.append(
            {key: (value.isoformat() if isinstance(value, (datetime, date, pd.Timestamp)) else value) for key, value in record.items()}
        )
    return records


def _max_available(frames: list[pd.DataFrame]) -> datetime:
    value = max(_timestamp(frame["available_at_utc"].max()) for frame in frames)
    return value


def _require_policy(
    policy: dict[str, object],
    field: str,
    expected: object,
) -> None:
    if policy.get(field) != expected:
        raise DataReadinessError(f"unsupported maturation label policy: {policy.get(field)!r}")


def _policy_int(policy: dict[str, object], field: str) -> int:
    value = policy.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataReadinessError(f"label policy field {field} must be an integer")
    return value


def _policy_float(policy: dict[str, object], field: str) -> float:
    value = policy.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DataReadinessError(f"label policy field {field} must be numeric")
    return float(value)


def _require_positive_prices(*values: float) -> None:
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise DataReadinessError("maturation price evidence is invalid")


def _aware_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("observed_as_of must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError("maturation timestamp is timezone-naive")
    return cast(datetime, timestamp.tz_convert("UTC").to_pydatetime())
