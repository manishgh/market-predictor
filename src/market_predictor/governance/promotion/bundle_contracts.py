"""Canonical contracts for promoted swing and intraday model bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Final, Literal, overload

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from market_predictor.catalysts.global_events.decision_authority import (
    GLOBAL_EVENT_SOURCE_FAMILIES,
)
from market_predictor.core.errors import (
    ArtifactIntegrityError,
    PromotionGateError,
    SchemaMismatchError,
)
from market_predictor.intraday.features.features import (
    CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS,
)
from market_predictor.intraday.features.features import (
    FEATURE_SCHEMA_VERSION as INTRADAY_FEATURE_SCHEMA_VERSION,
)
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.catalyst_decision_authority import (
    REQUIRED_MODEL_SOURCE_FAMILIES,
    TRACKED_SOURCE_FAMILIES,
)
from market_predictor.swing.features.panel import (
    SWING_FEATURE_PANEL_SCHEMA,
    swing_model_feature_columns,
)

SERVING_BUNDLE_SCHEMA: Final = "edge_rebuild.promoted_bundle.v2"

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
    strategy_contract_schema_version: Literal["edge_rebuild.strategy_contract.v2"]
    strategy_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
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
        canonical_order = tuple(family for family in TRACKED_SOURCE_FAMILIES if family in value)
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
        canonical_order = tuple(family for family in GLOBAL_EVENT_SOURCE_FAMILIES if family in value)
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
    """Governed identity for a promoted ten-session swing model."""

    mode: Literal["swing"]
    strategy_id: Literal["swing"]
    horizon_sessions: Literal[10]
    model_family: Literal["swing_baseline", "swing_event_driven"]
    feature_profile: Literal["technical_market", "catalyst_full"]
    catalyst_policy: Literal["confirmation_overlay", "required_model_feature"]

    @model_validator(mode="after")
    def validate_swing_schema(self) -> PromotedSwingBundle:
        if self.feature_schema_version != SWING_FEATURE_PANEL_SCHEMA:
            raise ValueError(f"swing bundle requires feature schema {SWING_FEATURE_PANEL_SCHEMA}")
        if self.model_family == "swing_baseline":
            if self.feature_profile != "technical_market":
                raise ValueError("swing baseline requires technical_market features")
            if self.catalyst_policy != "confirmation_overlay":
                raise ValueError("swing baseline requires catalyst confirmation overlay")
            if self.model_source_families:
                raise ValueError("swing baseline estimator cannot consume catalyst sources")
        elif self.feature_profile != "catalyst_full":
            raise ValueError("swing event-driven model requires catalyst_full features")
        elif self.catalyst_policy != "required_model_feature":
            raise ValueError("swing event-driven model requires catalyst model features")
        elif self.model_source_families != REQUIRED_MODEL_SOURCE_FAMILIES:
            raise ValueError("swing event-driven estimator source contract must be exactly Alpaca")
        return self


class PromotedIntradayBundle(_PromotedBundleBase):
    """Governed identity for a promoted thirty-minute intraday model."""

    mode: Literal["intraday"]
    strategy_id: Literal["intraday"]
    horizon_minutes: Literal[30]
    feature_profile: Literal["technical_market"]
    catalyst_policy: Literal["confirmation_overlay"]

    @model_validator(mode="after")
    def validate_intraday_schema(self) -> PromotedIntradayBundle:
        if self.feature_schema_version != INTRADAY_FEATURE_SCHEMA_VERSION:
            raise ValueError(f"intraday bundle requires feature schema {INTRADAY_FEATURE_SCHEMA_VERSION}")
        if self.model_source_families:
            raise ValueError("intraday estimator is technical-only; catalyst sources are overlay-only")
        return self


PromotedBundle = Annotated[
    PromotedSwingBundle | PromotedIntradayBundle,
    Field(discriminator="mode"),
]
_PROMOTED_BUNDLE_ADAPTER: Final[TypeAdapter[PromotedBundle]] = TypeAdapter(PromotedBundle)


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

    if payload.get("model_status") != "promoted" or payload.get("promotion_permitted") is not True:
        raise PromotionGateError("only explicitly promoted models may be served")
    mode = payload.get("mode")
    if mode not in {"swing", "intraday"}:
        raise SchemaMismatchError("serving bundle mode must be swing or intraday")
    if expected_mode is not None and mode != expected_mode:
        raise SchemaMismatchError(f"expected a {expected_mode} serving bundle, received {mode}")
    try:
        bundle = _PROMOTED_BUNDLE_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        horizon = "10 sessions" if mode == "swing" else "30 minutes"
        raise SchemaMismatchError(f"invalid {mode} promoted bundle; required horizon is {horizon}: {exc}") from exc

    if bundle.strategy_contract_schema_version != strategy_contract.schema_version:
        raise ArtifactIntegrityError("bundle strategy contract schema is stale")
    if bundle.strategy_contract_sha256 != strategy_contract.sha256():
        raise ArtifactIntegrityError("bundle does not bind the active strategy contract")
    expected_strategy_id = strategy_contract.swing.strategy_id if bundle.mode == "swing" else strategy_contract.intraday.strategy_id
    if bundle.strategy_id != expected_strategy_id:
        raise ArtifactIntegrityError("bundle strategy identity is stale")
    expected_features = (
        swing_model_feature_columns(
            contract=strategy_contract,
            catalyst=bundle.feature_profile == "catalyst_full",
        )
        if bundle.mode == "swing"
        else CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS
    )
    if bundle.ordered_feature_columns != expected_features:
        raise SchemaMismatchError(f"{bundle.mode} bundle feature columns do not match the active {bundle.feature_profile} estimator schema")
    return bundle
