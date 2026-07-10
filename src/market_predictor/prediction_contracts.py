from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


PredictionMode = Literal["swing", "intraday", "unified"]
PredictionView = Literal["swing", "intraday"]

_HORIZON_ALIASES = {
    "tomorrow": "1d",
    "next_day": "1d",
    "next-day": "1d",
    "1w": "5d",
    "week": "5d",
    "next_week": "5d",
    "next-week": "5d",
    "60m": "1h",
}


class PredictionRequest(BaseModel):
    """Typed request used by CLI, API, and tests.

    Training and collection stay outside this contract. Serving starts from
    already curated feature datasets and promoted model artifacts.
    """

    tickers: list[str] = Field(..., min_length=1)
    mode: PredictionMode = "unified"
    horizon: str = "auto"
    as_of: datetime | None = None
    swing_dataset: Path | None = None
    swing_model: Path | None = None
    intraday_dataset: Path | None = None
    intraday_model: Path | None = None
    universe: Path | None = None
    flashpoints: Path | None = None
    require_promoted: bool = True

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, tickers: list[str]) -> list[str]:
        normalized = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
        unique = list(dict.fromkeys(normalized))
        if not unique:
            raise ValueError("at least one ticker is required")
        return unique

    @field_validator("horizon")
    @classmethod
    def normalize_horizon(cls, horizon: str) -> str:
        normalized = _HORIZON_ALIASES.get(horizon.strip().lower(), horizon.strip().lower())
        if normalized == "auto" or re.fullmatch(r"[1-9]\d*(?:m|h|d|b)", normalized):
            return normalized
        raise ValueError("horizon must be auto or a positive duration such as 30m, 1h, 1d, 5d, or 12b")

    @field_validator("as_of")
    @classmethod
    def require_timezone_aware_as_of(cls, as_of: datetime | None) -> datetime | None:
        if as_of is not None and as_of.utcoffset() is None:
            raise ValueError("as_of must include an explicit UTC offset or timezone")
        return as_of


class ModelInfo(BaseModel):
    path: str
    status: str
    model_type: str | None = None
    schema_version: str | None = None
    target: str | None = None
    validation_split: str | None = None
    artifact_sha256: str | None = None
    resolved_horizon: str | None = None
    bar_timeframe: str | None = None
    created_at_utc: str | None = None
    training_data_start: str | None = None
    training_data_end: str | None = None


class ReadinessInfo(BaseModel):
    status: Literal["valid", "warn", "invalid"]
    reasons: list[str] = Field(default_factory=list)
    timeframe: Literal["daily", "intraday"] = "daily"
    daily_bar_count: int = 0
    intraday_bar_count: int = 0
    required_bar_count: int = 0
    latest_price_date: str | None = None
    price_feed: str = "unknown"
    benchmark_status: str = "unknown"
    market_context_status: str = "unknown"
    model_status: str = "unknown"
    source_status: str = "unknown"


class GlobalContextInfo(BaseModel):
    net_impact: float = 0.0
    positive_impact: float = 0.0
    negative_impact: float = 0.0
    active_flashpoints: list[str] = Field(default_factory=list)


class SwingPrediction(BaseModel):
    ticker: str
    date: str | None = None
    probability: float | None = None
    model_prediction: int | None = None
    signal: str
    rank: int | None = None
    close: float | None = None
    return_1d: float | None = None
    volume_z20: float | None = None
    news_count: float | None = None
    event_count: float | None = None
    sentiment_mean: float | None = None
    monitor_theme: str | None = None
    global_context: GlobalContextInfo = Field(default_factory=GlobalContextInfo)
    readiness: ReadinessInfo
    drivers: dict[str, float | int | str | None] = Field(default_factory=dict)


class IntradayPrediction(BaseModel):
    ticker: str
    date: str | None = None
    probability: float | None = None
    model_prediction: int | None = None
    probability_field: str | None = None
    signal: str
    rank: int | None = None
    close: float | None = None
    return_1d: float | None = None
    volume_z20: float | None = None
    rsi_14: float | None = None
    macd_signal_diff: float | None = None
    entry_stop_pct: float | None = None
    entry_target_pct: float | None = None
    readiness: ReadinessInfo
    drivers: dict[str, float | int | str | None] = Field(default_factory=dict)


class UnifiedTickerPrediction(BaseModel):
    ticker: str
    final_signal: str
    readiness_status: Literal["valid", "warn", "invalid"]
    swing: SwingPrediction | None = None
    intraday: IntradayPrediction | None = None
    errors: list[str] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: PredictionMode
    horizon: str
    resolved_horizons: dict[str, str] = Field(default_factory=dict)
    models: dict[str, ModelInfo] = Field(default_factory=dict)
    predictions: list[UnifiedTickerPrediction] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    snapshot_id: str | None = None
    snapshot_sha256: str | None = None


class InvestmentReplayRequest(BaseModel):
    snapshot_id: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    ticker: str = Field(..., min_length=1, max_length=16)
    model_view: PredictionView = "swing"
    evaluation_as_of: datetime | None = None
    initial_capital: float = Field(10_000.0, gt=0.0, le=1_000_000_000.0)
    slippage_bps: float = Field(5.0, ge=0.0, le=500.0)
    commission_bps: float = Field(0.0, ge=0.0, le=100.0)
    force_entry: bool = False

    @field_validator("ticker")
    @classmethod
    def normalize_replay_ticker(cls, ticker: str) -> str:
        normalized = ticker.strip().upper()
        if not normalized:
            raise ValueError("ticker is required")
        return normalized

    @field_validator("evaluation_as_of")
    @classmethod
    def require_timezone_aware_evaluation(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("evaluation_as_of must include an explicit UTC offset or timezone")
        return value


class InvestmentLegResult(BaseModel):
    ticker: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    shares: float
    initial_capital: float
    ending_value: float
    pnl: float
    return_pct: float


class InvestmentReplayResponse(BaseModel):
    replay_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot_id: str
    ticker: str
    model_view: PredictionView
    model_path: str | None = None
    model_artifact_sha256: str | None = None
    model_training_data_end: str | None = None
    decision_time: datetime
    evaluation_time: datetime
    prediction_signal: str
    prediction_readiness_status: Literal["valid", "warn", "invalid"] | None = None
    status: Literal["completed", "not_entered", "invalid"]
    reasons: list[str] = Field(default_factory=list)
    stock: InvestmentLegResult | None = None
    benchmarks: dict[str, InvestmentLegResult] = Field(default_factory=dict)
    excess_return_vs_spy: float | None = None
    excess_return_vs_qqq: float | None = None
