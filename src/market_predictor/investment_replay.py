from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

import pandas as pd

from market_predictor.config import Settings
from market_predictor.prediction_contracts import (
    InvestmentLegResult,
    InvestmentReplayRequest,
    InvestmentReplayResponse,
    ModelInfo,
)
from market_predictor.prediction_snapshot import PredictionSnapshotStore
from market_predictor.sources.alpaca import AlpacaSource

ACTIONABLE_SIGNALS = {
    "swing": {
        "bullish_watch",
        "bullish_watch_confirmed",
        "strong_bullish_watch",
        "strong_bullish_watch_confirmed",
    },
    "intraday": {"entry_candidate", "entry_candidate_confirmed"},
}
ReplayReadinessStatus = Literal["valid", "warn", "invalid"]
ReplayStatus = Literal["completed", "not_entered", "invalid"]


class ReplayPriceProvider(Protocol):
    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        timeframe: str,
    ) -> pd.DataFrame: ...


class AlpacaReplayPriceProvider:
    def __init__(self, settings: Settings) -> None:
        self.source = AlpacaSource(settings)

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        timeframe: str,
    ) -> pd.DataFrame:
        if timeframe == "1Day":
            return self.source.fetch_daily_bars(ticker, start, end)
        return self.source.fetch_intraday_bars(ticker, start, end, timeframe=timeframe)


class InvestmentReplayService:
    def __init__(
        self,
        *,
        snapshot_store: PredictionSnapshotStore,
        price_provider: ReplayPriceProvider,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.snapshot_store = snapshot_store
        self.price_provider = price_provider
        self.now = now or (lambda: datetime.now(UTC))

    def replay(self, request: InvestmentReplayRequest) -> InvestmentReplayResponse:
        prediction_request, prediction_response, _ = self.snapshot_store.load(request.snapshot_id)
        decision_time = _utc(prediction_request.as_of or prediction_response.generated_at_utc)
        evaluation_time = _utc(request.evaluation_as_of or self.now())
        ticker_row = next(
            (row for row in prediction_response.predictions if row.ticker == request.ticker),
            None,
        )
        prediction = getattr(ticker_row, request.model_view, None) if ticker_row is not None else None
        model = prediction_response.models.get(request.model_view)
        signal = prediction.signal if prediction is not None else "missing_prediction"
        readiness_status = prediction.readiness.status if prediction is not None else None

        reasons = self._validation_reasons(
            request=request,
            decision_time=decision_time,
            evaluation_time=evaluation_time,
            model=model,
            prediction_present=prediction is not None,
            prediction_readiness_status=readiness_status,
        )
        if reasons:
            return _response(
                request=request,
                decision_time=decision_time,
                evaluation_time=evaluation_time,
                signal=signal,
                readiness_status=readiness_status,
                status="invalid",
                reasons=reasons,
                model=model,
            )

        if not request.force_entry and signal not in ACTIONABLE_SIGNALS[request.model_view]:
            return _response(
                request=request,
                decision_time=decision_time,
                evaluation_time=evaluation_time,
                signal=signal,
                readiness_status=readiness_status,
                status="not_entered",
                reasons=[f"prediction signal {signal} is not an actionable {request.model_view} entry"],
                model=model,
            )

        assert model is not None
        timeframe = model.bar_timeframe or ("1Day" if request.model_view == "swing" else "5Min")
        start = decision_time - timedelta(days=2)
        fetch_end = evaluation_time + timedelta(days=2 if timeframe == "1Day" else 1)
        try:
            frames = {
                ticker: self.price_provider.fetch(ticker, start, fetch_end, timeframe=timeframe)
                for ticker in [request.ticker, "SPY", "QQQ"]
            }
        except Exception as exc:
            return _response(
                request=request,
                decision_time=decision_time,
                evaluation_time=evaluation_time,
                signal=signal,
                readiness_status=readiness_status,
                status="invalid",
                reasons=[f"price collection failed: {exc}"],
                model=model,
            )
        try:
            stock = simulate_investment_leg(
                frames[request.ticker],
                ticker=request.ticker,
                decision_time=decision_time,
                evaluation_time=evaluation_time,
                timeframe=timeframe,
                initial_capital=request.initial_capital,
                slippage_bps=request.slippage_bps,
                commission_bps=request.commission_bps,
            )
            benchmarks = {
                ticker: simulate_investment_leg(
                    frames[ticker],
                    ticker=ticker,
                    decision_time=decision_time,
                    evaluation_time=evaluation_time,
                    timeframe=timeframe,
                    initial_capital=request.initial_capital,
                    slippage_bps=request.slippage_bps,
                    commission_bps=request.commission_bps,
                    required_entry_time=stock.entry_time,
                    required_exit_time=stock.exit_time,
                )
                for ticker in ["SPY", "QQQ"]
            }
        except ValueError as exc:
            return _response(
                request=request,
                decision_time=decision_time,
                evaluation_time=evaluation_time,
                signal=signal,
                readiness_status=readiness_status,
                status="invalid",
                reasons=[str(exc)],
                model=model,
            )
        return _response(
            request=request,
            decision_time=decision_time,
            evaluation_time=evaluation_time,
            signal=signal,
            readiness_status=readiness_status,
            status="completed",
            reasons=[],
            model=model,
            stock=stock,
            benchmarks=benchmarks,
        )

    @staticmethod
    def _validation_reasons(
        *,
        request: InvestmentReplayRequest,
        decision_time: datetime,
        evaluation_time: datetime,
        model: ModelInfo | None,
        prediction_present: bool,
        prediction_readiness_status: str | None,
    ) -> list[str]:
        reasons: list[str] = []
        if evaluation_time <= decision_time:
            reasons.append("evaluation_as_of must be later than the prediction decision time")
        if not prediction_present:
            reasons.append(f"snapshot has no {request.model_view} prediction for {request.ticker}")
        elif prediction_readiness_status == "invalid":
            reasons.append("prediction data-readiness status is invalid")
        if model is None:
            reasons.append(f"snapshot has no {request.model_view} model metadata")
            return reasons
        if not model.artifact_sha256:
            reasons.append("model artifact hash is missing")
        if not model.created_at_utc:
            reasons.append("model creation timestamp is missing")
        else:
            created_at = _parse_utc(model.created_at_utc)
            if created_at > decision_time:
                reasons.append("model was created after the prediction decision time")
        training_end = _training_data_available_at(model)
        if training_end is None:
            reasons.append("model training-data end timestamp is missing")
        elif training_end > decision_time:
            reasons.append("model training data extends beyond the prediction decision time")
        return reasons


def simulate_investment_leg(
    bars: pd.DataFrame,
    *,
    ticker: str,
    decision_time: datetime,
    evaluation_time: datetime,
    timeframe: str,
    initial_capital: float,
    slippage_bps: float,
    commission_bps: float,
    required_entry_time: datetime | None = None,
    required_exit_time: datetime | None = None,
) -> InvestmentLegResult:
    prepared = _prepare_bars(bars, timeframe=timeframe)
    if prepared.empty:
        raise ValueError(f"no usable {timeframe} bars for {ticker}")
    decision = pd.Timestamp(_utc(decision_time))
    evaluation = pd.Timestamp(_utc(evaluation_time))

    if required_entry_time is None:
        entry_rows = prepared[prepared["_entry_time"] >= decision]
    else:
        required_entry = pd.Timestamp(_utc(required_entry_time))
        entry_rows = prepared[prepared["_entry_time"] == required_entry]
    if entry_rows.empty:
        raise ValueError(f"no tradable entry bar for {ticker} after the prediction time")
    entry = entry_rows.iloc[0]

    if required_exit_time is None:
        exit_rows = prepared[
            (prepared["_close_time"] <= evaluation)
            & (prepared["_close_time"] >= entry["_close_time"])
        ]
    else:
        required_exit = pd.Timestamp(_utc(required_exit_time))
        exit_rows = prepared[prepared["_close_time"] == required_exit]
    if exit_rows.empty:
        raise ValueError(f"no completed exit bar for {ticker} in the evaluation window")
    exit_row = exit_rows.iloc[-1]

    open_price = _positive_price(entry.get("open"), ticker=ticker, field="entry open")
    close_price = _positive_price(exit_row.get("close"), ticker=ticker, field="exit close")
    slippage = slippage_bps / 10_000.0
    commission = commission_bps / 10_000.0
    entry_price = open_price * (1.0 + slippage)
    exit_price = close_price * (1.0 - slippage)
    shares = initial_capital / (entry_price * (1.0 + commission))
    ending_value = shares * exit_price * (1.0 - commission)
    pnl = ending_value - initial_capital
    return InvestmentLegResult(
        ticker=ticker,
        entry_time=entry["_entry_time"].to_pydatetime(),
        entry_price=float(entry_price),
        exit_time=exit_row["_close_time"].to_pydatetime(),
        exit_price=float(exit_price),
        shares=float(shares),
        initial_capital=float(initial_capital),
        ending_value=float(ending_value),
        pnl=float(pnl),
        return_pct=float(pnl / initial_capital),
    )


def _prepare_bars(bars: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    frame = bars.copy()
    for column in ["open", "close"]:
        if column not in frame.columns:
            raise ValueError(f"price bars are missing {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if timeframe == "1Day":
        if "date" not in frame.columns:
            raise ValueError("daily price bars are missing date")
        dates = pd.to_datetime(frame["date"].astype(str).str.slice(0, 10), errors="coerce")
        local = dates.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        frame["_entry_time"] = (local + pd.Timedelta(hours=9, minutes=30)).dt.tz_convert("UTC")
        frame["_close_time"] = (local + pd.Timedelta(hours=16)).dt.tz_convert("UTC")
    else:
        timestamp_col = "timestamp" if "timestamp" in frame.columns else "date"
        timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
        duration = _bar_duration(timestamps)
        frame["_entry_time"] = timestamps
        frame["_close_time"] = timestamps + duration
    return frame.dropna(subset=["_entry_time", "_close_time", "open", "close"]).sort_values("_entry_time")


def _bar_duration(timestamps: pd.Series) -> pd.Timedelta:
    differences = timestamps.sort_values().diff()
    usable = differences[(differences > pd.Timedelta(0)) & (differences <= pd.Timedelta(hours=6))]
    if usable.empty:
        raise ValueError("cannot infer replay bar duration")
    return usable.median()


def _training_data_available_at(model: ModelInfo) -> datetime | None:
    if not model.training_data_end:
        return None
    if model.bar_timeframe == "1Day":
        date_text = str(model.training_data_end)[:10]
        local = pd.Timestamp(date_text, tz="America/New_York") + pd.Timedelta(hours=16)
        return _timestamp_datetime(local.tz_convert("UTC"))
    return _parse_utc(model.training_data_end)


def _response(
    *,
    request: InvestmentReplayRequest,
    decision_time: datetime,
    evaluation_time: datetime,
    signal: str,
    readiness_status: ReplayReadinessStatus | None,
    status: ReplayStatus,
    reasons: list[str],
    model: ModelInfo | None,
    stock: InvestmentLegResult | None = None,
    benchmarks: dict[str, InvestmentLegResult] | None = None,
) -> InvestmentReplayResponse:
    benchmark_results = benchmarks or {}
    return InvestmentReplayResponse(
        snapshot_id=request.snapshot_id,
        ticker=request.ticker,
        model_view=request.model_view,
        model_path=model.path if model else None,
        model_artifact_sha256=model.artifact_sha256 if model else None,
        model_training_data_end=model.training_data_end if model else None,
        decision_time=decision_time,
        evaluation_time=evaluation_time,
        prediction_signal=signal,
        prediction_readiness_status=readiness_status,
        status=status,
        reasons=reasons,
        stock=stock,
        benchmarks=benchmark_results,
        excess_return_vs_spy=(
            stock.return_pct - benchmark_results["SPY"].return_pct
            if stock is not None and "SPY" in benchmark_results
            else None
        ),
        excess_return_vs_qqq=(
            stock.return_pct - benchmark_results["QQQ"].return_pct
            if stock is not None and "QQQ" in benchmark_results
            else None
        ),
    )


def _parse_utc(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return _timestamp_datetime(timestamp)


def _utc(value: datetime) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return _timestamp_datetime(timestamp.tz_convert("UTC"))


def _positive_price(value: object, *, ticker: str, field: str) -> float:
    try:
        price = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} for {ticker}") from exc
    if not pd.notna(price) or price <= 0:
        raise ValueError(f"invalid {field} for {ticker}")
    return price


def _timestamp_datetime(timestamp: pd.Timestamp) -> datetime:
    converted = timestamp.to_pydatetime()
    if not isinstance(converted, datetime):
        raise ValueError("timestamp conversion did not produce a datetime")
    return converted
