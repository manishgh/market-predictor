from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

PredictionMode = Literal["swing", "intraday", "unified"]
PredictionView = Literal["swing", "intraday"]
PredictionDataSource = Literal["curated", "live"]

PREDICTION_CONTRACT_VERSION = "market_predictor.prediction.v1"
PREDICTION_EVIDENCE_CONTRACT_VERSION = "market_predictor.prediction_evidence.v1"


class PredictionServiceError(Exception):
    """Base class for stable, non-leaking service failures."""

    code: ClassVar[str] = "prediction_service_error"
    status_code: ClassVar[int] = 500
    retryable: ClassVar[bool] = False
    public_message: ClassVar[str] = "The prediction request could not be completed."


class PredictionValidationError(PredictionServiceError):
    code = "prediction_validation_error"
    status_code = 422
    public_message = "The prediction request is invalid."


class PredictionNotFoundError(PredictionServiceError):
    code = "prediction_not_found"
    status_code = 404
    public_message = "The requested prediction resource was not found."


class PredictionConflictError(PredictionServiceError):
    code = "prediction_conflict"
    status_code = 409
    public_message = "The prediction resource is in conflict with the requested operation."


class PredictionThrottledError(PredictionServiceError):
    code = "prediction_throttled"
    status_code = 429
    retryable = True
    public_message = "Prediction capacity is temporarily exhausted."


class PredictionReadinessError(PredictionServiceError):
    code = "prediction_not_ready"
    status_code = 503
    retryable = True
    public_message = "Prediction inputs or models are not ready."


class PredictionDependencyError(PredictionServiceError):
    code = "prediction_dependency_unavailable"
    status_code = 503
    retryable = True
    public_message = "A required prediction dependency is unavailable."

_HORIZON_ALIASES = {
    "tomorrow": "1d",
    "next_day": "1d",
    "next-day": "1d",
    "1w": "5d",
    "week": "5d",
    "next_week": "5d",
    "next-week": "5d",
    "1h": "60m",
}


class PredictionRequest(BaseModel):
    """Typed request used by CLI, API, and tests.

    Training and collection stay outside this contract. Model artifacts,
    feature sources, and promotion policy are owned by the server.
    """

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(..., min_length=1)
    mode: PredictionMode = "unified"
    horizon: str = "auto"
    as_of: datetime | None = None
    correlation_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")

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


class FeatureArtifactIdentityV1(BaseModel):
    mode: PredictionView
    artifact_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    source_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_artifact_type: str | None = None
    feature_schema_version: str | None = None


class PredictionRowEvidenceV1(BaseModel):
    ticker: str
    view: PredictionView
    decision_time_utc: datetime
    feature_available_at_utc: datetime

    @field_validator("decision_time_utc", "feature_available_at_utc")
    @classmethod
    def require_aware_row_timestamp(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("prediction evidence timestamps must be timezone-aware")
        return value.astimezone(UTC)


class PredictionEvidenceV1(BaseModel):
    """Immutable identities and point-in-time evidence for one served response."""

    contract_version: Literal["market_predictor.prediction_evidence.v1"] = "market_predictor.prediction_evidence.v1"
    request_id: str = Field(..., min_length=1, max_length=128)
    correlation_id: str = Field(..., min_length=1, max_length=128)
    prediction_cutoff_utc: datetime
    row_feature_availability: list[PredictionRowEvidenceV1] = Field(default_factory=list)
    feature_artifacts: dict[str, FeatureArtifactIdentityV1] = Field(default_factory=dict)
    release_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_artifact_sha256: dict[str, str] = Field(default_factory=dict)
    source_watermarks: dict[str, dict[str, str]] = Field(default_factory=dict)
    resolved_horizons: dict[str, str] = Field(default_factory=dict)
    view_prediction_cutoffs_utc: dict[str, datetime] = Field(default_factory=dict)
    serving_policy_id: str
    serving_policy_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    identity_status: Literal["complete", "incomplete", "research_only"]
    identity_gaps: list[str] = Field(default_factory=list)

    @field_validator("prediction_cutoff_utc")
    @classmethod
    def require_aware_prediction_cutoff(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("prediction_cutoff_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("view_prediction_cutoffs_utc")
    @classmethod
    def require_aware_view_cutoffs(cls, value: dict[str, datetime]) -> dict[str, datetime]:
        if any(timestamp.utcoffset() is None for timestamp in value.values()):
            raise ValueError("view prediction cutoffs must be timezone-aware")
        return {view: timestamp.astimezone(UTC) for view, timestamp in value.items()}

    @field_validator("model_artifact_sha256")
    @classmethod
    def require_model_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in value.values()):
            raise ValueError("model artifact identities must be lowercase SHA-256 values")
        return value


class PredictionApiError(BaseModel):
    code: str
    message: str
    correlation_id: str
    retryable: bool = False


class PredictionApiErrorEnvelope(BaseModel):
    error: PredictionApiError


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


class CatalystConfirmationInfo(BaseModel):
    status: Literal["confirmed", "conflicting", "veto", "mixed", "absent"] = "absent"
    direction: Literal["positive", "negative", "mixed", "none"] = "none"
    score: float = 0.0
    event_count: int = 0
    source_diversity: int = 0
    sentiment: float = 0.0
    relevance: float = 0.0
    minutes_since_latest: float | None = None
    material_event_count: int = 0
    reasons: list[str] = Field(default_factory=list)


class SwingPrediction(BaseModel):
    ticker: str
    date: str | None = None
    probability: float | None = None
    decision_score: float | None = None
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
    catalyst: CatalystConfirmationInfo = Field(default_factory=CatalystConfirmationInfo)
    readiness: ReadinessInfo
    drivers: dict[str, float | int | str | None] = Field(default_factory=dict)


class IntradayPrediction(BaseModel):
    ticker: str
    date: str | None = None
    opportunity_probability: float | None = None
    downside_probability: float | None = None
    decision_score: float | None = None
    opportunity_prediction: int | None = None
    downside_prediction: int | None = None
    signal: str
    rank: int | None = None
    close: float | None = None
    return_15m: float | None = None
    relative_volume: float | None = None
    rsi_14: float | None = None
    macd_signal_diff: float | None = None
    entry_stop_pct: float | None = None
    entry_target_pct: float | None = None
    catalyst: CatalystConfirmationInfo = Field(default_factory=CatalystConfirmationInfo)
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
    contract_version: Literal["market_predictor.prediction.v1"] = "market_predictor.prediction.v1"
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mode: PredictionMode
    data_source: PredictionDataSource = "live"
    horizon: str
    resolved_horizons: dict[str, str] = Field(default_factory=dict)
    models: dict[str, ModelInfo] = Field(default_factory=dict)
    predictions: list[UnifiedTickerPrediction] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    evidence: PredictionEvidenceV1 | None = None
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
    generated_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
