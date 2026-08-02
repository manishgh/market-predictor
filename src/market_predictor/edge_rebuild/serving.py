"""Strict serving contracts for promoted edge-rebuild models.

This module is deliberately independent of the legacy prediction service.  It
defines the only artifacts and outputs that a future edge-rebuild API may
serve, plus a fail-closed batch/live feature parity check.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, Literal, cast, overload

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.catalyst_authority import (
    REQUIRED_MODEL_SOURCE_FAMILIES,
    TRACKED_SOURCE_FAMILIES,
)
from market_predictor.edge_rebuild.global_event_authority import (
    GLOBAL_EVENT_SOURCE_FAMILIES,
    GlobalEventAuthority,
    load_global_event_authority,
)
from market_predictor.edge_rebuild.intraday_features import (
    CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS,
)
from market_predictor.edge_rebuild.intraday_features import (
    FEATURE_SCHEMA_VERSION as INTRADAY_FEATURE_SCHEMA_VERSION,
)
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
)
from market_predictor.edge_rebuild.swing_features import (
    SWING_FEATURE_PANEL_SCHEMA,
    swing_model_feature_columns,
)
from market_predictor.edge_rebuild.swing_training import (
    MODEL_SCHEMA as SWING_CANDIDATE_MODEL_SCHEMA,
)
from market_predictor.promotion_attestation import (
    promotion_attestation_path_for,
    verify_promotion_attestation,
)
from market_predictor.resources import assert_memory_budget, process_memory_snapshot
from market_predictor.v3.errors import (
    ArtifactIntegrityError,
    DataReadinessError,
    PromotionGateError,
    SchemaMismatchError,
)

SERVING_BUNDLE_SCHEMA: Final = "edge_rebuild.promoted_bundle.v2"
PREDICTION_RESULT_SCHEMA: Final = "edge_rebuild.prediction_result.v2"
ACTIVE_GENERATION_SCHEMA: Final = "edge_rebuild.active_generation.v1"
ACTIVE_GENERATION_POINTER: Final = "active_generation.json"
GENERATION_DIRECTORY: Final = "generations"
SWING_HORIZON_SESSIONS: Final = 10
INTRADAY_HORIZON_MINUTES: Final = 30

_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
_TRACKED_SOURCE_FAMILY_SET: Final = frozenset(TRACKED_SOURCE_FAMILIES)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _PromotedBundleBase(_FrozenModel):
    schema_version: Literal["edge_rebuild.promoted_bundle.v2"]
    model_id: str = Field(min_length=1, max_length=200)
    model_status: Literal["promoted"]
    promotion_permitted: Literal[True]
    model_artifact_path: str = Field(min_length=1, max_length=500)
    model_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    promotion_evidence_path: str = Field(min_length=1, max_length=500)
    promotion_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    promotion_attestation_id: str = Field(pattern=_SHA256_PATTERN)
    promotion_gate_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    approved_by_principal_id: str = Field(min_length=1, max_length=200)
    promoted_at_utc: datetime
    feature_schema_version: str = Field(min_length=1)
    ordered_feature_columns: tuple[str, ...] = Field(min_length=1)
    ordered_feature_sha256: str = Field(pattern=_SHA256_PATTERN)
    strategy_contract_schema_version: Literal[
        "edge_rebuild.strategy_contract.v2"
    ]
    strategy_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    market_data_provider: Literal["alpaca"]
    market_data_feed: Literal["sip"]
    market_data_adjustment: Literal["all"]
    model_source_families: tuple[str, ...]
    model_source_families_sha256: str = Field(pattern=_SHA256_PATTERN)
    catalyst_overlay_source_families: tuple[str, ...]
    catalyst_overlay_source_families_sha256: str = Field(pattern=_SHA256_PATTERN)
    catalyst_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    global_context_policy: Literal["ranking_overlay"]
    global_authority_schema_version: str = Field(min_length=1)
    global_source_families: tuple[str, ...] = Field(min_length=1)
    global_source_families_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("promoted_at_utc")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("promoted_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("model_artifact_path", "promotion_evidence_path")
    @classmethod
    def validate_artifact_path_text(cls, value: str) -> str:
        if value.strip() != value or "\x00" in value:
            raise ValueError("bundle artifact paths must be trimmed and contain no NUL")
        return value

    @field_validator("ordered_feature_columns")
    @classmethod
    def validate_ordered_features(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not column or column.strip() != column for column in value):
            raise ValueError("ordered feature names must be non-empty and trimmed")
        if len(value) != len(set(value)):
            raise ValueError("ordered feature names must be unique")
        return value

    @field_validator("model_source_families", "catalyst_overlay_source_families")
    @classmethod
    def validate_source_families(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(family.strip().lower() for family in value)
        if value != normalized:
            raise ValueError("source families must be normalized lowercase values")
        if len(value) != len(set(value)):
            raise ValueError("source families must be unique")
        unknown = set(value).difference(_TRACKED_SOURCE_FAMILY_SET)
        if unknown:
            raise ValueError(f"unrecognized source families: {sorted(unknown)}")
        canonical_order = tuple(
            family for family in TRACKED_SOURCE_FAMILIES if family in value
        )
        if value != canonical_order:
            raise ValueError("source families are not in canonical authority order")
        return value

    @field_validator("global_source_families")
    @classmethod
    def validate_global_source_families(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(family.strip().lower() for family in value)
        if value != normalized or len(value) != len(set(value)):
            raise ValueError("global source families must be normalized and unique")
        unknown = set(value).difference(GLOBAL_EVENT_SOURCE_FAMILIES)
        if unknown:
            raise ValueError(f"unrecognized global source families: {sorted(unknown)}")
        canonical_order = tuple(
            family for family in GLOBAL_EVENT_SOURCE_FAMILIES if family in value
        )
        if value != canonical_order:
            raise ValueError("global source families are not in canonical authority order")
        return value

    @model_validator(mode="after")
    def validate_hash_bindings(self) -> _PromotedBundleBase:
        if self.model_artifact_path == self.promotion_evidence_path:
            raise ValueError("model and promotion evidence must be distinct artifacts")
        expected_features = ordered_values_sha256(self.ordered_feature_columns)
        if self.ordered_feature_sha256 != expected_features:
            raise ValueError("ordered feature hash does not match ordered feature columns")
        source_hashes = (
            (self.model_source_families, self.model_source_families_sha256),
            (
                self.catalyst_overlay_source_families,
                self.catalyst_overlay_source_families_sha256,
            ),
            (self.global_source_families, self.global_source_families_sha256),
        )
        if any(ordered_values_sha256(values) != digest for values, digest in source_hashes):
            raise ValueError("source-family hash does not match its ordered source contract")
        return self

    def sha256(self) -> str:
        return canonical_payload_sha256(self.model_dump(mode="json"))


class PromotedSwingBundle(_PromotedBundleBase):
    """Serving identity for a promoted ten-session swing model."""

    mode: Literal["swing"]
    strategy_id: Literal["swing"]
    horizon_sessions: Literal[10]
    feature_profile: Literal["catalyst_full"]
    catalyst_policy: Literal["required_model_feature"]

    @model_validator(mode="after")
    def validate_swing_schema(self) -> PromotedSwingBundle:
        if self.feature_schema_version != SWING_FEATURE_PANEL_SCHEMA:
            raise ValueError(
                f"swing bundle requires feature schema {SWING_FEATURE_PANEL_SCHEMA}"
            )
        if self.model_source_families != REQUIRED_MODEL_SOURCE_FAMILIES:
            raise ValueError(
                "swing estimator source contract must be exactly the historical Alpaca family"
            )
        return self


class PromotedIntradayBundle(_PromotedBundleBase):
    """Serving identity for a promoted thirty-minute intraday model."""

    mode: Literal["intraday"]
    strategy_id: Literal["intraday"]
    horizon_minutes: Literal[30]
    feature_profile: Literal["technical_market"]
    catalyst_policy: Literal["confirmation_overlay"]

    @model_validator(mode="after")
    def validate_intraday_schema(self) -> PromotedIntradayBundle:
        if self.feature_schema_version != INTRADAY_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "intraday bundle requires feature schema "
                f"{INTRADAY_FEATURE_SCHEMA_VERSION}"
            )
        if self.model_source_families:
            raise ValueError(
                "intraday estimator is technical-only; catalyst sources are overlay-only"
            )
        return self


PromotedBundle = Annotated[
    PromotedSwingBundle | PromotedIntradayBundle,
    Field(discriminator="mode"),
]
_PROMOTED_BUNDLE_ADAPTER: Final[TypeAdapter[PromotedBundle]] = TypeAdapter(
    PromotedBundle
)


class GlobalContextSnapshot(_FrozenModel):
    as_of_utc: datetime
    authority_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_coverage_complete: Literal[True]
    event_count_1d: int = Field(ge=0)
    event_count_3d: int = Field(ge=0)
    sentiment_mean_1d: float = Field(ge=-1.0, le=1.0)
    sentiment_mean_3d: float = Field(ge=-1.0, le=1.0)
    sentiment_coverage_1d: float = Field(ge=0.0, le=1.0)
    sentiment_coverage_3d: float = Field(ge=0.0, le=1.0)
    source_families: tuple[str, ...] = Field(min_length=1)

    @field_validator("as_of_utc")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("global context timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("source_families")
    @classmethod
    def validate_source_families(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not family or family.strip().lower() != family for family in value):
            raise ValueError(
                "global context source families must be non-empty normalized values"
            )
        if len(value) != len(set(value)):
            raise ValueError("global context source families must be unique")
        unknown = set(value).difference(GLOBAL_EVENT_SOURCE_FAMILIES)
        if unknown:
            raise ValueError(f"unrecognized global context sources: {sorted(unknown)}")
        canonical_order = tuple(
            family for family in GLOBAL_EVENT_SOURCE_FAMILIES if family in value
        )
        if value != canonical_order:
            raise ValueError("global context sources are not in canonical order")
        return value


class CatalystSourceSnapshot(_FrozenModel):
    source_family: str = Field(min_length=1)
    coverage_known: bool
    event_count_1d: int | None = Field(default=None, ge=0)
    event_count_3d: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_missingness(self) -> CatalystSourceSnapshot:
        normalized = self.source_family.strip().lower()
        if self.source_family != normalized or normalized not in _TRACKED_SOURCE_FAMILY_SET:
            raise ValueError("catalyst source family is not a supported normalized value")
        counts_available = self.event_count_1d is not None and self.event_count_3d is not None
        if self.coverage_known != counts_available:
            raise ValueError(
                "catalyst counts must be null exactly when source coverage is unknown"
            )
        return self


class CatalystContextSnapshot(_FrozenModel):
    as_of_utc: datetime
    authority_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_model_sources_complete: bool
    event_count_1d: int | None = Field(default=None, ge=0)
    event_count_3d: int | None = Field(default=None, ge=0)
    sentiment_mean_1d: float | None = Field(default=None, ge=-1.0, le=1.0)
    sentiment_mean_3d: float | None = Field(default=None, ge=-1.0, le=1.0)
    sentiment_coverage_1d: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_coverage_3d: float | None = Field(default=None, ge=0.0, le=1.0)
    latest_event_feature_available_at_utc: datetime | None = None
    sources: tuple[CatalystSourceSnapshot, ...] = Field(min_length=1)

    @field_validator("as_of_utc", "latest_event_feature_available_at_utc")
    @classmethod
    def require_utc_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("catalyst context timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_context(self) -> CatalystContextSnapshot:
        families = tuple(source.source_family for source in self.sources)
        canonical = tuple(family for family in TRACKED_SOURCE_FAMILIES if family in families)
        if len(families) != len(set(families)) or families != canonical:
            raise ValueError("catalyst source snapshots must be unique and canonical")
        required_known = all(
            source.coverage_known
            for source in self.sources
            if source.source_family in REQUIRED_MODEL_SOURCE_FAMILIES
        ) and set(REQUIRED_MODEL_SOURCE_FAMILIES).issubset(families)
        if self.required_model_sources_complete != required_known:
            raise ValueError("required catalyst source completeness is inconsistent")
        aggregates = (
            self.event_count_1d,
            self.event_count_3d,
            self.sentiment_mean_1d,
            self.sentiment_mean_3d,
            self.sentiment_coverage_1d,
            self.sentiment_coverage_3d,
        )
        aggregates_available = all(value is not None for value in aggregates)
        if self.required_model_sources_complete != aggregates_available:
            raise ValueError(
                "catalyst aggregates must be null exactly when required sources are incomplete"
            )
        if (
            self.latest_event_feature_available_at_utc is not None
            and self.latest_event_feature_available_at_utc > self.as_of_utc
        ):
            raise ValueError("catalyst context contains future evidence")
        return self


class BenchmarkComparison(_FrozenModel):
    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    predicted_stock_return: float
    predicted_benchmark_return: float
    predicted_excess_return: float

    @model_validator(mode="after")
    def validate_returns(self) -> BenchmarkComparison:
        values = (
            self.predicted_stock_return,
            self.predicted_benchmark_return,
            self.predicted_excess_return,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("benchmark comparison returns must be finite")
        expected = self.predicted_stock_return - self.predicted_benchmark_return
        if not np.isclose(
            self.predicted_excess_return,
            expected,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("predicted excess return is inconsistent")
        return self


AbstentionReason = Literal[
    "benchmark_unavailable",
    "catalyst_source_unavailable",
    "data_quality_failure",
    "feature_schema_mismatch",
    "feature_value_unavailable",
    "live_batch_parity_failure",
    "market_data_unavailable",
    "model_not_promoted",
    "out_of_universe",
    "stale_features",
    "strategy_contract_mismatch",
]


class PredictionResult(_FrozenModel):
    """Non-executable model intelligence returned by the edge serving core."""

    schema_version: Literal[
        "edge_rebuild.prediction_result.v2"
    ] = PREDICTION_RESULT_SCHEMA
    mode: Literal["swing", "intraday"]
    strategy_id: Literal["swing", "intraday"]
    model_id: str = Field(min_length=1, max_length=200)
    bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    as_of_utc: datetime
    horizon_value: int
    horizon_unit: Literal["sessions", "minutes"]
    status: Literal["scored", "abstained"]
    predicted_direction: Literal["up", "down", "neutral"] | None
    model_score: float | None = Field(default=None, ge=0.0, le=1.0)
    technical_score: float | None = Field(default=None, ge=0.0, le=1.0)
    catalyst_overlay_status: Literal[
        "incorporated",
        "confirmed",
        "contradicted",
        "neutral",
        "unavailable",
    ]
    catalyst_context_available: bool
    catalyst_context: CatalystContextSnapshot | None
    global_context_available: bool
    global_context: GlobalContextSnapshot | None
    benchmark_comparisons: tuple[BenchmarkComparison, ...] = ()
    abstention_reasons: tuple[AbstentionReason, ...] = ()

    @field_validator("as_of_utc")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("prediction timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_result_contract(self) -> PredictionResult:
        if self.mode != self.strategy_id:
            raise ValueError("prediction mode and strategy identity disagree")
        expected_horizon = (
            (SWING_HORIZON_SESSIONS, "sessions")
            if self.mode == "swing"
            else (INTRADAY_HORIZON_MINUTES, "minutes")
        )
        if (self.horizon_value, self.horizon_unit) != expected_horizon:
            raise ValueError(
                f"{self.mode} prediction horizon must be "
                f"{expected_horizon[0]} {expected_horizon[1]}"
            )
        if self.global_context_available != (self.global_context is not None):
            raise ValueError(
                "global_context must be null exactly when global context is unavailable"
            )
        if self.catalyst_context_available != (self.catalyst_context is not None):
            raise ValueError(
                "catalyst_context must be null exactly when catalyst context is unavailable"
            )
        if self.catalyst_context is not None and self.catalyst_context.as_of_utc > self.as_of_utc:
            raise ValueError("catalyst context cannot be newer than the prediction")
        if (
            self.global_context is not None
            and self.global_context.as_of_utc > self.as_of_utc
        ):
            raise ValueError("global context cannot be newer than the prediction")
        if len(self.abstention_reasons) != len(set(self.abstention_reasons)):
            raise ValueError("abstention reasons must be unique")
        if self.status == "scored":
            if (
                self.predicted_direction is None
                or self.model_score is None
                or self.technical_score is None
            ):
                raise ValueError(
                    "scored predictions require direction, model score, and "
                    "technical score"
                )
            if self.abstention_reasons:
                raise ValueError("scored predictions cannot carry abstention reasons")
            if not self.benchmark_comparisons:
                raise ValueError("scored predictions require benchmark comparison")
            benchmark_symbols = tuple(
                item.symbol for item in self.benchmark_comparisons
            )
            if len(benchmark_symbols) != len(set(benchmark_symbols)):
                raise ValueError("benchmark comparison symbols must be unique")
            symbols = set(benchmark_symbols)
            missing_benchmarks = {"SPY", "QQQ"}.difference(symbols)
            if missing_benchmarks:
                raise ValueError(
                    "scored predictions require SPY and QQQ comparisons; missing "
                    f"{sorted(missing_benchmarks)}"
                )
            if (
                self.mode == "swing"
                and self.catalyst_overlay_status != "incorporated"
            ):
                raise ValueError(
                    "scored swing predictions require incorporated catalyst features"
                )
            if self.mode == "swing" and (
                self.catalyst_context is None
                or not self.catalyst_context.required_model_sources_complete
            ):
                raise ValueError(
                    "scored swing predictions require complete bound catalyst context"
                )
        else:
            if not self.abstention_reasons:
                raise ValueError("abstained predictions require at least one reason")
            if (
                self.predicted_direction is not None
                or self.model_score is not None
                or self.technical_score is not None
            ):
                raise ValueError("abstained predictions cannot expose a model score")
            if self.benchmark_comparisons:
                raise ValueError("abstained predictions cannot expose benchmark forecasts")
        if self.mode == "intraday" and self.catalyst_overlay_status == "incorporated":
            raise ValueError(
                "intraday catalyst is a confirmation overlay, not a model feature"
            )
        return self


class FeatureParityReport(_FrozenModel):
    matched: Literal[True] = True
    row_count: int = Field(ge=1)
    feature_count: int = Field(ge=1)
    ordered_feature_sha256: str = Field(pattern=_SHA256_PATTERN)
    maximum_absolute_difference: float = Field(ge=0.0)
    maximum_relative_difference: float = Field(ge=0.0)
    relative_tolerance: float = Field(ge=0.0)
    absolute_tolerance: float = Field(ge=0.0)


class SwingModelScores(_FrozenModel):
    """Calibrated outputs from one promoted ten-session swing model."""

    probabilities: tuple[float, ...]
    probability_threshold: float = Field(gt=0.0, lt=1.0)

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("swing probabilities must be a non-empty finite sequence in [0, 1]")
        return values


@dataclass(frozen=True, slots=True)
class LoadedSwingModelGeneration:
    """One immutable, attested swing model generation held in memory."""

    generation_id: str
    pointer_sha256: str
    bundle: PromotedSwingBundle
    model_payload: Mapping[str, object]


class SwingModelGenerationCache:
    """Load exactly one attested generation per repository and detect rollover."""

    def __init__(
        self,
        *,
        memory_budget_gib: float,
        memory_headroom_gib: float,
    ) -> None:
        self._memory_budget_gib = memory_budget_gib
        self._memory_headroom_gib = memory_headroom_gib
        self._lock = threading.Lock()
        self._loaded: dict[
            tuple[Path, str, Path, str, str, int],
            LoadedSwingModelGeneration,
        ] = {}

    def get(
        self,
        repository: Path,
        *,
        strategy_contract: StrategyContract,
        attestation_trust_store_path: Path,
        promotion_gate_policy_sha256: str,
        maximum_model_bytes: int,
        estimated_resident_gib: float,
    ) -> LoadedSwingModelGeneration:
        root = _verified_bundle_root(repository)
        trust_store = attestation_trust_store_path.resolve(strict=True)
        cache_key = (
            root,
            strategy_contract.sha256(),
            trust_store,
            file_sha256(trust_store),
            promotion_gate_policy_sha256,
            maximum_model_bytes,
        )
        pointer = load_active_generation_pointer(root)
        cached = self._loaded.get(cache_key)
        if cached is not None and cached.pointer_sha256 == pointer["pointer_sha256"]:
            return cached
        with self._lock:
            pointer = load_active_generation_pointer(root)
            cached = self._loaded.get(cache_key)
            if cached is not None and cached.pointer_sha256 == pointer["pointer_sha256"]:
                return cached
            assert_memory_budget(
                hard_budget_gib=self._memory_budget_gib,
                headroom_gib=self._memory_headroom_gib,
                stage="before swing model generation load",
            )
            _assert_projected_rss(
                estimated_resident_gib,
                hard_budget_gib=self._memory_budget_gib,
                headroom_gib=self._memory_headroom_gib,
            )
            generation_root = _verified_generation_root(root, pointer["generation_id"])
            bundle_path = generation_root / "bundle.json"
            bundle_bytes = _read_verified_file_bytes(
                bundle_path,
                expected_sha256=pointer["bundle_file_sha256"],
                maximum_bytes=1024 * 1024,
                label="swing generation bundle",
            )
            try:
                raw = json.loads(bundle_bytes.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ArtifactIntegrityError("swing generation bundle is unreadable") from exc
            if not isinstance(raw, Mapping):
                raise SchemaMismatchError("swing generation bundle must be an object")
            bundle = cast(
                PromotedSwingBundle,
                validate_file_backed_promoted_bundle(
                    raw,
                    bundle_root=generation_root,
                    strategy_contract=strategy_contract,
                    attestation_trust_store_path=attestation_trust_store_path,
                    promotion_gate_policy_sha256=promotion_gate_policy_sha256,
                    maximum_model_bytes=maximum_model_bytes,
                    expected_mode="swing",
                ),
            )
            if bundle.sha256() != pointer["generation_id"]:
                raise ArtifactIntegrityError(
                    "active swing generation identity does not match its bundle"
                )
            model_path = _verified_bundle_artifact_path(
                generation_root,
                bundle.model_artifact_path,
                label="model",
            )
            payload = _load_joblib_from_verified_handle(
                model_path,
                expected_sha256=bundle.model_artifact_sha256,
                maximum_bytes=maximum_model_bytes,
            )
            if not isinstance(payload, Mapping):
                raise SchemaMismatchError("promoted swing model payload must be an object")
            _validate_swing_model_payload(payload, bundle)
            after = load_active_generation_pointer(root)
            if after["pointer_sha256"] != pointer["pointer_sha256"]:
                raise DataReadinessError(
                    "active swing model generation changed during verification"
                )
            loaded = LoadedSwingModelGeneration(
                generation_id=pointer["generation_id"],
                pointer_sha256=pointer["pointer_sha256"],
                bundle=bundle,
                model_payload={str(key): value for key, value in payload.items()},
            )
            self._loaded[cache_key] = loaded
            assert_memory_budget(
                hard_budget_gib=self._memory_budget_gib,
                headroom_gib=self._memory_headroom_gib,
                stage="after swing model generation load",
            )
            return loaded

    def is_current(self, repository: Path, generation: LoadedSwingModelGeneration) -> bool:
        pointer = load_active_generation_pointer(_verified_bundle_root(repository))
        return pointer["pointer_sha256"] == generation.pointer_sha256


def load_active_generation_pointer(root: Path) -> dict[str, str]:
    """Read and verify the single atomic pointer to an immutable generation."""

    path = _verified_bundle_root(root) / ACTIVE_GENERATION_POINTER
    payload = _read_json_bytes(path, label="active generation pointer")
    expected_fields = {
        "schema",
        "generation_id",
        "bundle_file_sha256",
        "previous_generation_id",
        "activated_at_utc",
        "pointer_sha256",
    }
    if set(payload) != expected_fields or payload.get("schema") != ACTIVE_GENERATION_SCHEMA:
        raise ArtifactIntegrityError("active generation pointer schema is invalid")
    unsigned = dict(payload)
    pointer_sha = str(unsigned.pop("pointer_sha256", ""))
    if canonical_payload_sha256(unsigned) != pointer_sha:
        raise ArtifactIntegrityError("active generation pointer hash is invalid")
    for field in ("generation_id", "bundle_file_sha256", "pointer_sha256"):
        value = str(payload.get(field, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ArtifactIntegrityError(f"active generation pointer {field} is invalid")
    _ = _strict_utc_datetime(payload.get("activated_at_utc"), "active generation activation")
    previous = payload.get("previous_generation_id")
    if previous is not None and (
        not isinstance(previous, str)
        or len(previous) != 64
        or any(character not in "0123456789abcdef" for character in previous)
    ):
        raise ArtifactIntegrityError("active generation previous identity is invalid")
    return {str(key): str(value) if value is not None else "" for key, value in payload.items()}


def ordered_values_sha256(values: Sequence[str]) -> str:
    """Hash an ordered string contract without platform-dependent formatting."""

    payload = json.dumps(
        list(values),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    """Hash a JSON-compatible contract using one canonical representation."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@overload
def validate_promoted_bundle(
    payload: Mapping[str, object],
    *,
    strategy_contract: StrategyContract,
    expected_mode: Literal["swing"],
) -> PromotedSwingBundle: ...


@overload
def validate_promoted_bundle(
    payload: Mapping[str, object],
    *,
    strategy_contract: StrategyContract,
    expected_mode: Literal["intraday"],
) -> PromotedIntradayBundle: ...


@overload
def validate_promoted_bundle(
    payload: Mapping[str, object],
    *,
    strategy_contract: StrategyContract,
    expected_mode: None = None,
) -> PromotedSwingBundle | PromotedIntradayBundle: ...


def validate_promoted_bundle(
    payload: Mapping[str, object],
    *,
    strategy_contract: StrategyContract,
    expected_mode: Literal["swing", "intraday"] | None = None,
) -> PromotedSwingBundle | PromotedIntradayBundle:
    """Parse and bind a promoted bundle to the active frozen strategy contract."""

    if payload.get("model_status") != "promoted" or payload.get(
        "promotion_permitted"
    ) is not True:
        raise PromotionGateError("only explicitly promoted models may be served")
    mode = payload.get("mode")
    if mode not in {"swing", "intraday"}:
        raise SchemaMismatchError("serving bundle mode must be swing or intraday")
    if expected_mode is not None and mode != expected_mode:
        raise SchemaMismatchError(
            f"expected a {expected_mode} serving bundle, received {mode}"
        )
    try:
        bundle = _PROMOTED_BUNDLE_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        horizon = "10 sessions" if mode == "swing" else "30 minutes"
        raise SchemaMismatchError(
            f"invalid {mode} promoted bundle; required horizon is {horizon}: {exc}"
        ) from exc

    if bundle.strategy_contract_schema_version != strategy_contract.schema_version:
        raise ArtifactIntegrityError("bundle strategy contract schema is stale")
    if bundle.strategy_contract_sha256 != strategy_contract.sha256():
        raise ArtifactIntegrityError("bundle does not bind the active strategy contract")
    expected_strategy_id = (
        strategy_contract.swing.strategy_id
        if bundle.mode == "swing"
        else strategy_contract.intraday.strategy_id
    )
    if bundle.strategy_id != expected_strategy_id:
        raise ArtifactIntegrityError("bundle strategy identity is stale")
    expected_features = (
        swing_model_feature_columns(contract=strategy_contract, catalyst=True)
        if bundle.mode == "swing"
        else CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS
    )
    if bundle.ordered_feature_columns != expected_features:
        raise SchemaMismatchError(
            f"{bundle.mode} bundle feature columns do not match the active "
            f"{bundle.feature_profile} estimator schema"
        )
    return bundle


def validate_file_backed_promoted_bundle(
    payload: Mapping[str, object],
    *,
    bundle_root: Path,
    strategy_contract: StrategyContract,
    attestation_trust_store_path: Path | None = None,
    promotion_gate_policy_sha256: str | None = None,
    maximum_model_bytes: int | None = None,
    maximum_evidence_bytes: int = 1024 * 1024,
    expected_mode: Literal["swing", "intraday"] | None = None,
) -> PromotedSwingBundle | PromotedIntradayBundle:
    """Validate bundle metadata, artifacts, and signed promotion authorization."""

    bundle = validate_promoted_bundle(
        payload,
        strategy_contract=strategy_contract,
        expected_mode=expected_mode,
    )
    root = _verified_bundle_root(bundle_root)
    artifacts = (
        (
            "model",
            bundle.model_artifact_path,
            bundle.model_artifact_sha256,
            maximum_model_bytes,
        ),
        (
            "promotion evidence",
            bundle.promotion_evidence_path,
            bundle.promotion_evidence_sha256,
            maximum_evidence_bytes,
        ),
    )
    for label, relative_path, expected_sha256, maximum_bytes in artifacts:
        artifact_path = _verified_bundle_artifact_path(
            root,
            relative_path,
            label=label,
        )
        before = artifact_path.stat()
        if maximum_bytes is not None and (
            maximum_bytes < 1 or before.st_size > maximum_bytes
        ):
            raise DataReadinessError(f"{label} artifact byte limit exceeded")
        observed_sha256 = file_sha256(artifact_path)
        after = artifact_path.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise ArtifactIntegrityError(
                f"{label} changed while its serving hash was being verified"
            )
        if observed_sha256 != expected_sha256:
            raise ArtifactIntegrityError(f"{label} artifact SHA256 does not verify")
    if bundle.mode != "swing":
        return bundle
    if attestation_trust_store_path is None:
        raise PromotionGateError(
            "a configured promotion attestation trust store is required"
        )
    if promotion_gate_policy_sha256 is None:
        raise PromotionGateError(
            "a configured promotion gate-policy hash is required"
        )
    if bundle.promotion_gate_policy_sha256 != promotion_gate_policy_sha256:
        raise PromotionGateError(
            "serving bundle promotion gate policy differs from configuration"
        )
    model_path = _verified_bundle_artifact_path(
        root,
        bundle.model_artifact_path,
        label="model",
    )
    evidence_path = _verified_bundle_artifact_path(
        root,
        bundle.promotion_evidence_path,
        label="promotion evidence",
    )
    if evidence_path != promotion_attestation_path_for(model_path):
        raise PromotionGateError(
            "promotion evidence must be the immutable candidate attestation"
        )
    try:
        attestation = verify_promotion_attestation(
            model_path,
            trust_store_path=attestation_trust_store_path,
        )
    except (DataReadinessError, OSError, TypeError, ValueError) as exc:
        raise PromotionGateError("promotion attestation did not verify") from exc
    candidate = attestation.get("candidate")
    approver = attestation.get("approver_principal")
    ledger = attestation.get("ledger_receipt")
    if not isinstance(candidate, Mapping) or not isinstance(approver, Mapping):
        raise PromotionGateError("promotion attestation identity is incomplete")
    if (
        candidate.get("artifact_sha256") != bundle.model_artifact_sha256
        or candidate.get("model_run_id") != bundle.model_id
        or candidate.get("model_schema_version") != SWING_CANDIDATE_MODEL_SCHEMA
        or attestation.get("attestation_id") != bundle.promotion_attestation_id
        or attestation.get("gate_config_sha256")
        != bundle.promotion_gate_policy_sha256
        or approver.get("principal_id") != bundle.approved_by_principal_id
        or attestation.get("promoted_at_utc") != bundle.promoted_at_utc.isoformat()
    ):
        raise PromotionGateError(
            "promotion attestation does not bind the served candidate identity"
        )
    if not isinstance(ledger, Mapping) or ledger.get("result") != "passed":
        raise PromotionGateError("promotion attestation does not prove passed gates")
    return bundle


def build_global_context_snapshot(
    authority: GlobalEventAuthority,
    *,
    decision_time_utc: datetime,
    authority_sha256: str,
) -> GlobalContextSnapshot:
    """Build serving context from one exact row of a verified production authority."""

    authority_path = authority.directory.resolve() / "_authority.json"
    if not authority_path.is_file() or authority_path.is_symlink():
        raise ArtifactIntegrityError("global event authority file is missing or unsafe")
    if file_sha256(authority_path) != authority_sha256:
        raise ArtifactIntegrityError("global event authority SHA256 does not verify")
    verified = load_global_event_authority(
        authority.directory,
        require_production_ready=True,
    )
    decision_time = pd.Timestamp(decision_time_utc)
    if decision_time.tzinfo is None:
        raise DataReadinessError("global context decision time must be timezone-aware")
    decision_time = decision_time.tz_convert("UTC")
    observed_times = pd.to_datetime(
        verified.decisions["decision_time_utc"],
        utc=True,
        errors="coerce",
    )
    matches = verified.decisions.loc[observed_times == decision_time]
    if len(matches) != 1:
        raise DataReadinessError(
            "global event authority requires exactly one row for the requested decision"
        )
    row = matches.iloc[0]
    for window in ("1d", "3d"):
        complete = row[f"global_source_complete_{window}"]
        if not isinstance(complete, (bool, np.bool_)) or not bool(complete):
            raise DataReadinessError(
                f"global source coverage is incomplete for the {window} window"
            )
        latest_value = row[f"global_latest_event_feature_available_at_utc_{window}"]
        if not pd.isna(latest_value):
            latest = pd.Timestamp(latest_value)
            if latest.tzinfo is None or latest.tz_convert("UTC") > decision_time:
                raise DataReadinessError("global context contains future event evidence")
    source_values = verified.manifest.get("required_historical_sources")
    if not isinstance(source_values, list) or not source_values:
        raise ArtifactIntegrityError("global authority source contract is malformed")
    source_families = tuple(str(value) for value in source_values)
    return GlobalContextSnapshot(
        as_of_utc=decision_time.to_pydatetime(),
        authority_sha256=authority_sha256,
        source_coverage_complete=True,
        event_count_1d=_authority_nonnegative_integer(row["global_event_count_1d"]),
        event_count_3d=_authority_nonnegative_integer(row["global_event_count_3d"]),
        sentiment_mean_1d=_authority_finite_float(row["global_sentiment_mean_1d"]),
        sentiment_mean_3d=_authority_finite_float(row["global_sentiment_mean_3d"]),
        sentiment_coverage_1d=_authority_finite_float(
            row["global_sentiment_coverage_1d"]
        ),
        sentiment_coverage_3d=_authority_finite_float(
            row["global_sentiment_coverage_3d"]
        ),
        source_families=source_families,
    )


def _verified_bundle_root(bundle_root: Path) -> Path:
    if bundle_root.is_symlink():
        raise ArtifactIntegrityError("immutable bundle root cannot be a symlink")
    try:
        root = bundle_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError("immutable bundle root does not exist") from exc
    if not root.is_dir():
        raise ArtifactIntegrityError("immutable bundle root must be a directory")
    return root


def _verified_bundle_artifact_path(
    root: Path,
    value: str,
    *,
    label: str,
) -> Path:
    if "\\" in value:
        raise ArtifactIntegrityError(
            f"{label} path must use canonical bundle-relative POSIX syntax"
        )
    relative = PurePosixPath(value)
    if (
        relative.as_posix() != value
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise ArtifactIntegrityError(f"{label} path escapes the immutable bundle root")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactIntegrityError(f"{label} path contains a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError(f"{label} artifact is missing") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ArtifactIntegrityError(
            f"{label} artifact is not a regular file below the immutable bundle root"
        )
    return resolved


def _verified_generation_root(root: Path, generation_id: str) -> Path:
    generations = root / GENERATION_DIRECTORY
    if generations.is_symlink():
        raise ArtifactIntegrityError("swing generation directory cannot be a symlink")
    candidate = generations / generation_id
    if candidate.is_symlink():
        raise ArtifactIntegrityError("active swing generation cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        generations_resolved = generations.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError("active swing generation is unavailable") from exc
    if not resolved.is_dir() or not resolved.is_relative_to(generations_resolved):
        raise ArtifactIntegrityError("active swing generation escapes its repository")
    return resolved


def _read_json_bytes(path: Path, *, label: str) -> dict[str, object]:
    raw = _read_verified_file_bytes(
        path,
        expected_sha256=None,
        maximum_bytes=1024 * 1024,
        label=label,
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _read_verified_file_bytes(
    path: Path,
    *,
    expected_sha256: str | None,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if path.is_symlink() or maximum_bytes < 1:
        raise ArtifactIntegrityError(f"{label} path or byte limit is invalid")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            size = int(before.st_size)
            if size < 1 or size > maximum_bytes:
                raise DataReadinessError(f"{label} byte limit exceeded")
            payload = handle.read(maximum_bytes + 1)
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError(f"{label} is unavailable") from exc
    if len(payload) != size or len(payload) > maximum_bytes:
        raise ArtifactIntegrityError(f"{label} changed while being read")
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ArtifactIntegrityError(f"{label} SHA256 does not verify")
    return payload


def _load_joblib_from_verified_handle(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> object:
    if path.is_symlink() or maximum_bytes < 1:
        raise ArtifactIntegrityError("promoted swing model path or byte limit is invalid")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            size = int(before.st_size)
            if size < 1 or size > maximum_bytes:
                raise DataReadinessError("promoted swing model byte limit exceeded")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ArtifactIntegrityError("promoted swing model SHA256 does not verify")
            handle.seek(0)
            payload = joblib.load(handle)
            after = os.fstat(handle.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ArtifactIntegrityError(
                    "promoted swing model changed during deserialization"
                )
            return payload
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError("promoted swing model is unavailable") from exc
    except Exception as exc:
        if isinstance(exc, (ArtifactIntegrityError, DataReadinessError)):
            raise
        raise ArtifactIntegrityError("promoted swing model artifact is unreadable") from exc


def _validate_swing_model_payload(
    payload: Mapping[str, object],
    bundle: PromotedSwingBundle,
) -> None:
    if payload.get("schema") != SWING_CANDIDATE_MODEL_SCHEMA:
        raise SchemaMismatchError("promoted swing model payload schema is unsupported")
    if payload.get("status") != "candidate" or payload.get("promotion_permitted") is not False:
        raise PromotionGateError(
            "served model must remain an immutable candidate authorized by attestation"
        )
    if payload.get("candidate_id") != bundle.model_id:
        raise ArtifactIntegrityError("promoted model candidate identity differs")
    if payload.get("strategy_contract_sha256") != bundle.strategy_contract_sha256:
        raise ArtifactIntegrityError("promoted model strategy contract binding differs")
    features = tuple(str(value) for value in _required_sequence(payload, "feature_columns"))
    if features != bundle.ordered_feature_columns:
        raise SchemaMismatchError("promoted model feature order differs from its bundle")
    if str(payload.get("ablation_profile", "")) != bundle.feature_profile:
        raise SchemaMismatchError("promoted model feature profile differs from its bundle")
    _finite_probability(payload.get("probability_threshold"), "probability_threshold")
    fitted = payload.get("fitted_candidate")
    if getattr(fitted, "estimator", None) is None or getattr(fitted, "calibrator", None) is None:
        raise SchemaMismatchError("promoted swing model is missing estimator or calibrator")


def _strict_utc_datetime(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ArtifactIntegrityError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactIntegrityError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _assert_projected_rss(
    estimated_resident_gib: float,
    *,
    hard_budget_gib: float,
    headroom_gib: float,
) -> None:
    if estimated_resident_gib <= 0:
        raise DataReadinessError("swing generation resident-memory estimate is invalid")
    snapshot = process_memory_snapshot()
    if snapshot is None:
        return
    current_gib = snapshot[0] / 1024**3
    if current_gib + estimated_resident_gib > hard_budget_gib - headroom_gib:
        raise DataReadinessError(
            "swing generation load would exceed the configured RSS safety threshold"
        )


def _authority_nonnegative_integer(value: object) -> int:
    numeric = _authority_finite_float(value)
    if numeric < 0 or not numeric.is_integer():
        raise DataReadinessError("global authority event count is not a nonnegative integer")
    return int(numeric)


def _authority_finite_float(value: object) -> float:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("global authority metric is not numeric") from exc
    if not np.isfinite(numeric):
        raise DataReadinessError("global authority metric is not finite")
    return numeric


def validate_ordered_feature_frame(
    frame: pd.DataFrame,
    ordered_feature_columns: Sequence[str],
    *,
    frame_name: str,
) -> np.ndarray:
    """Validate an inference frame and return its finite float64 matrix."""

    expected = tuple(ordered_feature_columns)
    if not expected or len(expected) != len(set(expected)):
        raise SchemaMismatchError("expected feature contract must be non-empty and unique")
    if frame.empty:
        raise DataReadinessError(f"{frame_name} inference feature frame is empty")
    if not frame.index.is_unique:
        raise SchemaMismatchError(f"{frame_name} feature row index must be unique")
    observed = tuple(str(column) for column in frame.columns)
    if observed != expected:
        raise SchemaMismatchError(
            f"{frame_name} feature columns differ from promoted order; "
            f"expected={list(expected)!r}, observed={list(observed)!r}"
        )
    invalid_types = [
        column
        for column in expected
        if is_bool_dtype(frame[column].dtype)
        or not is_numeric_dtype(frame[column].dtype)
    ]
    if invalid_types:
        raise SchemaMismatchError(
            f"{frame_name} contains non-numeric model features: {invalid_types}"
        )
    values = frame.loc[:, expected].to_numpy(dtype="float64", copy=False)
    if not bool(np.isfinite(values).all()):
        locations = np.argwhere(~np.isfinite(values))
        row, column = (int(value) for value in locations[0])
        raise DataReadinessError(
            f"{frame_name} contains a non-finite feature at row {row}, "
            f"column {expected[column]}"
        )
    return cast(np.ndarray[Any, np.dtype[np.float64]], values)


def score_promoted_swing_model(
    generation: LoadedSwingModelGeneration,
    *,
    feature_frame: pd.DataFrame,
) -> SwingModelScores:
    """Score with the already verified, immutable in-memory model generation."""

    bundle = generation.bundle
    payload = generation.model_payload
    _validate_swing_model_payload(payload, bundle)
    threshold = _finite_probability(payload.get("probability_threshold"), "probability_threshold")
    fitted = payload.get("fitted_candidate")
    estimator = getattr(fitted, "estimator", None)
    calibrator = getattr(fitted, "calibrator", None)
    if estimator is None or calibrator is None:
        raise SchemaMismatchError("promoted model is missing estimator or calibrator")
    matrix = validate_ordered_feature_frame(
        feature_frame,
        bundle.ordered_feature_columns,
        frame_name="promoted swing",
    ).astype("float32", copy=False)
    try:
        raw = np.asarray(estimator.predict_proba(matrix), dtype="float64")
        if raw.ndim != 2 or raw.shape != (len(matrix), 2):
            raise ValueError("estimator probability shape is invalid")
        calibrated = np.asarray(
            calibrator.predict_proba(raw[:, 1].reshape(-1, 1))[:, 1],
            dtype="float64",
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise DataReadinessError("promoted swing model could not score live features") from exc
    if calibrated.shape != (len(matrix),) or not np.isfinite(calibrated).all():
        raise DataReadinessError("promoted swing model produced invalid probabilities")
    if bool(((calibrated < 0.0) | (calibrated > 1.0)).any()):
        raise DataReadinessError("promoted swing probabilities are outside [0, 1]")
    return SwingModelScores(
        probabilities=tuple(float(value) for value in calibrated),
        probability_threshold=threshold,
    )


def _required_sequence(payload: Mapping[str, object], field: str) -> Sequence[object]:
    value = payload.get(field)
    if not isinstance(value, (list, tuple)) or not value:
        raise SchemaMismatchError(f"promoted swing model {field} is invalid")
    return value


def _finite_probability(value: object, field: str) -> float:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError(f"promoted swing model {field} is invalid") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or numeric >= 1.0:
        raise SchemaMismatchError(f"promoted swing model {field} is invalid")
    return numeric


def validate_batch_live_feature_parity(
    batch: pd.DataFrame,
    live: pd.DataFrame,
    ordered_feature_columns: Sequence[str],
    *,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> FeatureParityReport:
    """Fail when live inference features diverge from the batch implementation."""

    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("feature parity tolerances cannot be negative")
    expected = tuple(ordered_feature_columns)
    batch_values = validate_ordered_feature_frame(
        batch,
        expected,
        frame_name="batch",
    )
    live_values = validate_ordered_feature_frame(
        live,
        expected,
        frame_name="live",
    )
    if batch_values.shape != live_values.shape:
        raise SchemaMismatchError(
            "batch/live feature shapes differ; "
            f"batch={batch_values.shape}, live={live_values.shape}"
        )
    if not batch.index.equals(live.index):
        raise SchemaMismatchError("batch/live feature row identities or order differ")
    absolute_difference = np.abs(batch_values - live_values)
    denominator = np.maximum(
        np.maximum(np.abs(batch_values), np.abs(live_values)),
        np.finfo(np.float64).tiny,
    )
    relative_difference = absolute_difference / denominator
    close = np.isclose(
        batch_values,
        live_values,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    if not bool(close.all()):
        row, column = (int(value) for value in np.argwhere(~close)[0])
        raise DataReadinessError(
            "batch/live feature parity failed at "
            f"row {row}, column {expected[column]}; "
            f"batch={batch_values[row, column]!r}, "
            f"live={live_values[row, column]!r}"
        )
    return FeatureParityReport(
        row_count=int(batch_values.shape[0]),
        feature_count=int(batch_values.shape[1]),
        ordered_feature_sha256=ordered_values_sha256(expected),
        maximum_absolute_difference=(
            float(absolute_difference.max()) if absolute_difference.size else 0.0
        ),
        maximum_relative_difference=(
            float(relative_difference.max()) if relative_difference.size else 0.0
        ),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
