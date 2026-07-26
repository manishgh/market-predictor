"""Causal setup eligibility and strategy-specific swing label primitives."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import (
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    EXECUTION_POLICY_ID,
    EXECUTION_POLICY_SHA256,
    round_trip_cost_fraction,
)
from market_predictor.label_paths import evaluate_intraday_barrier_paths
from market_predictor.label_policy import policy_sha256
from market_predictor.label_reconciliation import (
    label_material_sha256,
    replay_mismatch_count,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.labels import add_exact_swing_labels
from market_predictor.v3.errors import DataReadinessError

STRATEGY_LABEL_SCHEMA_VERSION = "swing.strategy_labels.v1"
STRATEGY_LABEL_ARTIFACT_SCHEMA = "swing.strategy_label_rows.v3"
STRATEGY_LABEL_BUNDLE_SCHEMA = "swing.strategy_label_bundle.v1"
StrategyFamily = Literal[
    "cross_sectional_continuation",
    "time_series_continuation",
    "catalyst_continuation",
    "short_term_reversal",
    "breakout_expansion",
    "sector_residual_continuation",
]
STRATEGY_IDS = (
    "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1",
    "SWING.TIME_SERIES_MOMENTUM.5D.V1",
    "SWING.CATALYST_DRIFT.5D.V1",
    "SWING.SHORT_TERM_REVERSAL.3D.V1",
    "SWING.BREAKOUT_EXPANSION.5D.V1",
    "SWING.SECTOR_RESIDUAL_MOMENTUM.5D.V1",
)
EXPECTED_STRATEGY_CONTRACTS = {
    "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1": (
        "cross_sectional_continuation",
        5,
        "top_quintile_spy_excess_and_positive_net",
    ),
    "SWING.TIME_SERIES_MOMENTUM.5D.V1": (
        "time_series_continuation",
        5,
        "positive_net_and_positive_spy_excess",
    ),
    "SWING.CATALYST_DRIFT.5D.V1": (
        "catalyst_continuation",
        5,
        "positive_net_spy_and_sector_excess",
    ),
    "SWING.SHORT_TERM_REVERSAL.3D.V1": (
        "short_term_reversal",
        3,
        "positive_net_and_positive_sector_excess",
    ),
    "SWING.BREAKOUT_EXPANSION.5D.V1": (
        "breakout_expansion",
        5,
        "target_before_stop_and_positive_net",
    ),
    "SWING.SECTOR_RESIDUAL_MOMENTUM.5D.V1": (
        "sector_residual_continuation",
        5,
        "positive_sector_and_spy_excess",
    ),
}
SETUP_ABSTENTION_REASONS = frozenset(
    {
        "",
        "feature_ineligible",
        "cross_section_ineligible",
        "non_sip_price_feed",
        "invalid_adjustment",
        "missing_required_feature",
        "trend_not_confirmed",
        "catalyst_coverage_unavailable",
        "direct_catalyst_not_confirmed",
        "overreaction_not_confirmed",
        "breakout_not_confirmed",
        "residual_strength_not_confirmed",
    }
)
LABEL_ABSTENTION_REASONS = frozenset(
    {
        "",
        *SETUP_ABSTENTION_REASONS,
        "label_window_not_expected",
        "missing_exact_stock_path",
        "missing_benchmark_interval",
        "missing_execution_cost_evidence",
        "invalid_barrier_prices",
    }
)
_COMMON_REQUIRED_COLUMNS = {
    "ticker",
    "security_id",
    "session_date_et",
    "decision_group_id",
    "decision_time_utc",
    "feature_available_at_utc",
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
    "primary_benchmark",
    "feature_eligible",
    "atr_pct_14",
}
_LABEL_BEARING_PREFIXES = ("future_", "target_", "path_")
_LABEL_BEARING_COLUMNS = {
    "entry_time_utc",
    "exit_time_utc",
    "entry_price",
    "exit_price",
    "label_available_at_utc",
    "label_path_exact",
    "label_eligible",
    "target_excess_rank",
}
STRATEGY_LABEL_MATERIAL_COLUMNS = (
    "strategy_id",
    "strategy_version",
    "strategy_family",
    "strategy_decision_group_id",
    "setup_eligible",
    "setup_abstention_reason",
    "setup_feature_available_at_utc",
    "catalyst_source_complete",
    "entry_time_utc",
    "exit_time_utc",
    "label_available_at_utc",
    "entry_price",
    "entry_atr_pct",
    "exit_price",
    "label_window_expected",
    "label_path_exact",
    "strategy_label_eligible",
    "label_abstention_reason",
    "strategy_outcome",
    "strategy_target",
    "strategy_gross_return",
    "strategy_execution_cost_fraction",
    "strategy_net_return",
    "strategy_spy_return",
    "strategy_qqq_return",
    "strategy_sector_return",
    "strategy_excess_return_vs_spy",
    "strategy_excess_return_vs_qqq",
    "strategy_excess_return_vs_sector",
    "strategy_mfe",
    "strategy_mae",
    "barrier_outcome",
    "barrier_outcome_session",
    "barrier_target_price",
    "barrier_stop_price",
    "barrier_realized_price",
    "breakout_failed",
    "strategy_label_policy_sha256",
    "execution_policy_sha256",
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SwingStrategySpec(FrozenModel):
    version: int = Field(ge=1)
    family: StrategyFamily
    horizon_sessions: int = Field(ge=1, le=20)
    target_rule: str = Field(min_length=1)
    required_features: tuple[str, ...]
    minimum_return_20d: float | None = None
    minimum_relative_rank: float | None = Field(default=None, ge=0, le=1)
    minimum_dist_sma_50: float | None = None
    minimum_dist_sma_200: float | None = None
    minimum_sma_200_slope_20d: float | None = None
    minimum_event_count_3d: float | None = Field(default=None, ge=0)
    minimum_sentiment_coverage_3d: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    minimum_event_relevance_mean_3d: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    minimum_dist_ema_20: float | None = None
    minimum_return_5d: float | None = None
    maximum_return_to_atr: float | None = None
    maximum_rsi_14: float | None = Field(default=None, ge=0, le=100)
    maximum_event_count_3d: float | None = Field(default=None, ge=0)
    prior_high_sessions: int | None = Field(default=None, ge=2, le=252)
    compression_sessions: int | None = Field(default=None, ge=2, le=252)
    maximum_compression_range: float | None = Field(default=None, gt=0)
    minimum_volume_ratio_20: float | None = Field(default=None, ge=0)
    minimum_close_location: float | None = Field(default=None, ge=0, le=1)
    target_atr: float | None = Field(default=None, gt=0)
    stop_atr: float | None = Field(default=None, gt=0)
    minimum_residual_return_20d: float | None = None

    @model_validator(mode="after")
    def validate_family_parameters(self) -> Self:
        if not self.required_features or len(set(self.required_features)) != len(
            self.required_features
        ):
            raise ValueError("required_features must be non-empty and unique")
        required_by_family: dict[StrategyFamily, tuple[str, ...]] = {
            "cross_sectional_continuation": (
                "minimum_return_20d",
                "minimum_relative_rank",
            ),
            "time_series_continuation": (
                "minimum_return_20d",
                "minimum_dist_sma_50",
                "minimum_dist_sma_200",
                "minimum_sma_200_slope_20d",
            ),
            "catalyst_continuation": (
                "minimum_event_count_3d",
                "minimum_sentiment_coverage_3d",
                "minimum_event_relevance_mean_3d",
                "minimum_dist_ema_20",
                "minimum_return_5d",
            ),
            "short_term_reversal": (
                "maximum_return_to_atr",
                "maximum_rsi_14",
                "maximum_event_count_3d",
            ),
            "breakout_expansion": (
                "prior_high_sessions",
                "compression_sessions",
                "maximum_compression_range",
                "minimum_volume_ratio_20",
                "minimum_close_location",
                "target_atr",
                "stop_atr",
            ),
            "sector_residual_continuation": (
                "minimum_residual_return_20d",
                "minimum_relative_rank",
            ),
        }
        missing = [
            name
            for name in required_by_family[self.family]
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"{self.family} has missing parameters: {', '.join(missing)}"
            )
        return self


class SwingStrategyLabelPolicy(FrozenModel):
    schema_version: str = STRATEGY_LABEL_SCHEMA_VERSION
    execution_policy_id: str
    execution_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_rule: Literal["next_exact_exchange_session_open"]
    exit_rule: Literal["decision_plus_horizon_session_close"]
    path_rule: Literal["all_exchange_sessions_required"]
    same_barrier_rule: Literal["stop_first"]
    maximum_process_memory_gib: float = Field(gt=0, le=4)
    memory_guard_headroom_gib: float = Field(gt=0)
    strategies: dict[str, SwingStrategySpec]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != STRATEGY_LABEL_SCHEMA_VERSION:
            raise ValueError("unsupported swing strategy label schema")
        if self.execution_policy_id != EXECUTION_POLICY_ID:
            raise ValueError("strategy labels must use the frozen execution policy ID")
        if self.execution_policy_sha256 != EXECUTION_POLICY_SHA256:
            raise ValueError(
                "strategy labels must use the frozen execution policy hash"
            )
        if set(self.strategies) != set(STRATEGY_IDS):
            raise ValueError("strategy label catalog does not match KS2")
        semantic_hashes: set[str] = set()
        for strategy_id, expected in EXPECTED_STRATEGY_CONTRACTS.items():
            spec = self.strategies[strategy_id]
            observed = (
                spec.family,
                spec.horizon_sessions,
                spec.target_rule,
            )
            if observed != expected:
                raise ValueError(
                    "strategy contract mismatch for "
                    f"{strategy_id}: {observed} != {expected}"
                )
            semantic_hashes.add(
                policy_sha256(spec.model_dump(mode="json"))
            )
        if len(semantic_hashes) != len(STRATEGY_IDS):
            raise ValueError(
                "strategy policies must have distinct semantic contracts"
            )
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory guard headroom must be below the hard budget")
        return self

    def canonical_policy(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def sha256(self) -> str:
        return policy_sha256(self.canonical_policy())

    def strategy_sha256(self, strategy_id: str) -> str:
        spec = self.strategies[strategy_id]
        return policy_sha256(
            {
                "schema_version": self.schema_version,
                "strategy_id": strategy_id,
                "spec": spec.model_dump(mode="json"),
                "entry_rule": self.entry_rule,
                "exit_rule": self.exit_rule,
                "path_rule": self.path_rule,
                "same_barrier_rule": self.same_barrier_rule,
                "execution_policy_id": self.execution_policy_id,
                "execution_policy_sha256": self.execution_policy_sha256,
            }
        )


def load_swing_strategy_label_policy(path: Path) -> SwingStrategyLabelPolicy:
    with path.open("rb") as handle:
        return SwingStrategyLabelPolicy.model_validate(tomllib.load(handle))


def swing_strategy_evaluator_sha256() -> str:
    """Bind source implementations that can change KS2 outputs."""

    source_files = (
        Path(__file__),
        Path(__file__).with_name("dataset.py"),
        Path(__file__).with_name("labels.py"),
        Path(__file__).parents[1] / "execution_policy.py",
        Path(__file__).parents[1] / "label_paths.py",
        Path(__file__).parents[1] / "label_reconciliation.py",
    )
    digest = hashlib.sha256()
    for path in source_files:
        if not path.is_file():
            raise DataReadinessError(
                f"strategy evaluator source is unavailable: {path}"
            )
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def prune_swing_strategy_label_inputs(
    frame: pd.DataFrame,
    policy: SwingStrategyLabelPolicy,
) -> pd.DataFrame:
    """Drop columns that cannot affect a registered setup or exact path."""

    _require_columns(frame, _COMMON_REQUIRED_COLUMNS, "swing feature history")
    _reject_label_bearing_input(frame)
    retained = set(
        _strategy_source_columns(
            frame,
            policy,
            STRATEGY_IDS,
        )
    )
    frame.drop(
        columns=[
            column
            for column in frame.columns
            if column not in retained
        ],
        inplace=True,
    )
    return frame


def build_strategy_setups(
    frame: pd.DataFrame,
    policy: SwingStrategyLabelPolicy,
    *,
    strategy_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build one causal setup row per strategy and source decision."""

    _require_columns(frame, _COMMON_REQUIRED_COLUMNS, "swing feature history")
    _reject_label_bearing_input(frame)
    selected_ids = _selected_strategy_ids(policy, strategy_ids)
    data = frame.loc[
        :,
        _strategy_source_columns(frame, policy, selected_ids),
    ].reset_index(drop=True).copy()
    data["_strategy_source_row_id"] = np.arange(len(data), dtype=np.int64)
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["security_id"] = data["security_id"].astype(str).str.strip()
    decision = _strict_utc(data["decision_time_utc"], "decision_time_utc")
    feature_available = _strict_utc(
        data["feature_available_at_utc"],
        "feature_available_at_utc",
    )
    if bool(feature_available.gt(decision).any()):
        raise DataReadinessError(
            "strategy setup features contain post-decision evidence"
        )
    if bool(
        data.duplicated(["security_id", "session_date_et"]).any()
    ):
        raise DataReadinessError(
            "strategy setup source has duplicate security/session rows"
        )
    data = _add_breakout_setup_features(data, policy)
    records: list[pd.DataFrame] = []
    for strategy_id in selected_ids:
        spec = policy.strategies[strategy_id]
        part = data[
            [
                "_strategy_source_row_id",
                "ticker",
                "security_id",
                "session_date_et",
                "decision_time_utc",
                "feature_available_at_utc",
                "decision_group_id",
            ]
        ].copy()
        part["strategy_id"] = strategy_id
        part["strategy_version"] = spec.version
        part["strategy_family"] = spec.family
        part["strategy_horizon_sessions"] = spec.horizon_sessions
        part["strategy_target_rule"] = spec.target_rule
        reason = pd.Series("", index=data.index, dtype="string")
        _set_reason(
            reason,
            ~data["feature_eligible"].fillna(False).astype(bool),
            "feature_ineligible",
        )
        _set_reason(
            reason,
            data["price_feed"].astype(str).str.lower().ne("sip"),
            "non_sip_price_feed",
        )
        _set_reason(
            reason,
            data["adjustment"].astype(str).str.lower().ne("all"),
            "invalid_adjustment",
        )
        missing_feature = pd.Series(False, index=data.index)
        for feature in spec.required_features:
            if feature not in data:
                missing_feature[:] = True
            else:
                missing_feature |= data[feature].isna()
        _set_reason(
            reason,
            missing_feature,
            "missing_required_feature",
        )
        requires_cross_section = spec.family in {
            "cross_sectional_continuation",
            "sector_residual_continuation",
        }
        if requires_cross_section:
            cross_section = (
                data["cross_section_eligible"].fillna(False).astype(bool)
                if "cross_section_eligible" in data
                else pd.Series(False, index=data.index)
            )
            _set_reason(
                reason,
                ~cross_section,
                "cross_section_ineligible",
            )
        if spec.family in {
            "catalyst_continuation",
            "short_term_reversal",
        }:
            _set_reason(
                reason,
                ~_bool(data, "catalyst_source_complete"),
                "catalyst_coverage_unavailable",
            )
        condition, condition_reason = _strategy_condition(
            data,
            spec,
        )
        _set_reason(reason, ~condition, condition_reason)
        part["setup_eligible"] = reason.eq("")
        part["setup_abstention_reason"] = reason
        part["setup_feature_available_at_utc"] = feature_available
        part["catalyst_source_complete"] = _bool(
            data,
            "catalyst_source_complete",
        )
        part["strategy_setup_policy_sha256"] = policy.strategy_sha256(
            strategy_id
        )
        part["strategy_decision_group_id"] = (
            part["decision_group_id"].astype(str)
            + "|"
            + strategy_id
        ).map(_sha256_text)
        part["strategy_setup_schema_version"] = (
            STRATEGY_LABEL_SCHEMA_VERSION
        )
        records.append(part)
    output = pd.concat(records, ignore_index=True)
    invalid_reasons = set(
        output["setup_abstention_reason"].astype(str)
    ).difference(SETUP_ABSTENTION_REASONS)
    if invalid_reasons:
        raise DataReadinessError(
            f"unbounded setup abstention reasons: {sorted(invalid_reasons)}"
        )
    return output.sort_values(
        ["strategy_id", "decision_time_utc", "ticker"],
        kind="stable",
    ).reset_index(drop=True)


def build_swing_strategy_labels(
    frame: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    dataset_config: SwingDatasetConfig,
    policy: SwingStrategyLabelPolicy,
    strategy_ids: Sequence[str] | None = None,
    _chunk_sessions: int | None = 126,
) -> pd.DataFrame:
    """Build exact, strategy-specific labels without consuming generic targets."""

    assert_memory_budget(
        hard_budget_gib=policy.maximum_process_memory_gib,
        headroom_gib=policy.memory_guard_headroom_gib,
        stage="strategy label build start",
    )
    selected_ids = _selected_strategy_ids(policy, strategy_ids)
    _reject_label_bearing_input(frame)
    if _chunk_sessions is not None and _chunk_sessions > 0:
        sessions = sorted(pd.unique(frame["session_date_et"]))
        if len(sessions) > _chunk_sessions:
            return _build_chunked_swing_strategy_labels(
                frame,
                benchmark_bars,
                dataset_config=dataset_config,
                policy=policy,
                strategy_ids=selected_ids,
                sessions=sessions,
                chunk_sessions=_chunk_sessions,
            )
    source = frame.loc[
        :,
        _strategy_source_columns(frame, policy, selected_ids),
    ].reset_index(drop=True).copy()
    source["_strategy_source_row_id"] = np.arange(
        len(source),
        dtype=np.int64,
    )
    setups = build_strategy_setups(
        source.drop(columns="_strategy_source_row_id"),
        policy,
        strategy_ids=selected_ids,
    )
    labelled_by_horizon: dict[int, pd.DataFrame] = {}
    path_by_horizon: dict[int, dict[str, np.ndarray]] = {}
    for horizon in sorted(
        {
            policy.strategies[strategy_id].horizon_sessions
            for strategy_id in selected_ids
        }
    ):
        horizon_config = dataset_config.model_copy(
            update={"horizon_sessions": horizon}
        )
        labelled_by_horizon[horizon] = add_exact_swing_labels(
            source,
            benchmark_bars,
            horizon_config,
        )
        path_by_horizon[horizon] = _future_path_arrays(
            source,
            horizon,
        )
    rows: list[pd.DataFrame] = []
    for strategy_id in selected_ids:
        spec = policy.strategies[strategy_id]
        labelled = labelled_by_horizon[spec.horizon_sessions].set_index(
            "_strategy_source_row_id",
            drop=False,
        )
        setup = setups.loc[
            setups["strategy_id"].eq(strategy_id)
        ].copy()
        source_ids = setup["_strategy_source_row_id"].to_numpy(dtype=int)
        base = labelled.loc[source_ids].reset_index(drop=True)
        part = setup.reset_index(drop=True)
        horizon = spec.horizon_sessions
        part["entry_time_utc"] = base["entry_time_utc"]
        part["exit_time_utc"] = base["exit_time_utc"]
        part["label_available_at_utc"] = base[
            "label_available_at_utc"
        ]
        part["entry_price"] = pd.to_numeric(
            base["entry_price"],
            errors="coerce",
        )
        part["exit_price"] = pd.to_numeric(
            base["exit_price"],
            errors="coerce",
        )
        part["label_window_expected"] = base[
            "label_window_expected"
        ].fillna(False).astype(bool)
        part["label_path_exact"] = base["label_path_exact"].fillna(
            False
        ).astype(bool)
        gross = pd.to_numeric(
            base[f"future_gross_return_{horizon}d"],
            errors="coerce",
        )
        spy_return = pd.to_numeric(
            base[f"future_spy_return_{horizon}d"],
            errors="coerce",
        )
        qqq_return = pd.to_numeric(
            base[f"future_qqq_return_{horizon}d"],
            errors="coerce",
        )
        sector_return = pd.to_numeric(
            base[f"future_sector_return_{horizon}d"],
            errors="coerce",
        )
        entry_atr_pct = pd.to_numeric(
            base["atr_pct_14"],
            errors="coerce",
        )
        part["entry_atr_pct"] = entry_atr_pct
        execution_cost = round_trip_cost_fraction(
            part["entry_price"],
            entry_atr_pct,
            policy=DEFAULT_EXECUTION_POLICY,
        )
        valid_cost_evidence = (
            pd.to_numeric(part["entry_price"], errors="coerce").gt(0)
            & entry_atr_pct.notna()
            & entry_atr_pct.ge(0)
        )
        execution_cost = execution_cost.where(valid_cost_evidence)
        net = gross - execution_cost
        part["strategy_gross_return"] = gross
        part["strategy_execution_cost_fraction"] = execution_cost
        part["strategy_net_return"] = net
        part["strategy_spy_return"] = spy_return
        part["strategy_qqq_return"] = qqq_return
        part["strategy_sector_return"] = sector_return
        part["strategy_excess_return_vs_spy"] = net - spy_return
        part["strategy_excess_return_vs_qqq"] = net - qqq_return
        part["strategy_excess_return_vs_sector"] = net - sector_return
        part["strategy_mfe"] = pd.to_numeric(
            base[f"future_mfe_{horizon}d"],
            errors="coerce",
        )
        part["strategy_mae"] = pd.to_numeric(
            base[f"future_mae_{horizon}d"],
            errors="coerce",
        )
        part["barrier_outcome"] = ""
        part["barrier_outcome_session"] = pd.Series(
            pd.NA,
            index=part.index,
            dtype="Int64",
        )
        part["barrier_target_price"] = np.nan
        part["barrier_stop_price"] = np.nan
        part["barrier_realized_price"] = np.nan
        part["breakout_failed"] = pd.Series(
            pd.NA,
            index=part.index,
            dtype="boolean",
        )
        if spec.family == "breakout_expansion":
            _add_breakout_path_labels(
                part,
                base,
                source_ids=source_ids,
                path=path_by_horizon[horizon],
                spec=spec,
                benchmark_bars=benchmark_bars,
                dataset_config=dataset_config,
            )
        gross = _num(part, "strategy_gross_return")
        spy_return = _num(part, "strategy_spy_return")
        qqq_return = _num(part, "strategy_qqq_return")
        sector_return = _num(part, "strategy_sector_return")
        execution_cost = _num(
            part,
            "strategy_execution_cost_fraction",
        )
        label_reason = part["setup_abstention_reason"].astype(
            "string"
        ).copy()
        _set_reason(
            label_reason,
            ~part["label_window_expected"],
            "label_window_not_expected",
        )
        _set_reason(
            label_reason,
            ~part["label_path_exact"] | gross.isna(),
            "missing_exact_stock_path",
        )
        _set_reason(
            label_reason,
            spy_return.isna() | qqq_return.isna() | sector_return.isna(),
            "missing_benchmark_interval",
        )
        _set_reason(
            label_reason,
            execution_cost.isna(),
            "missing_execution_cost_evidence",
        )
        if spec.family == "breakout_expansion":
            _set_reason(
                label_reason,
                part["barrier_outcome"].eq(""),
                "invalid_barrier_prices",
            )
        part["strategy_label_eligible"] = label_reason.eq("")
        part["label_abstention_reason"] = label_reason
        target, outcome = _strategy_target_and_outcome(part, spec)
        part["strategy_target"] = target.astype("Int64")
        part["strategy_outcome"] = outcome
        ineligible = ~part["strategy_label_eligible"]
        part.loc[ineligible, "strategy_target"] = pd.NA
        part.loc[ineligible, "strategy_outcome"] = ""
        part["strategy_label_policy_sha256"] = policy.strategy_sha256(
            strategy_id
        )
        part["execution_policy_id"] = policy.execution_policy_id
        part["execution_policy_sha256"] = (
            policy.execution_policy_sha256
        )
        part["strategy_label_schema_version"] = (
            STRATEGY_LABEL_ARTIFACT_SCHEMA
        )
        rows.append(part)
    output = pd.concat(rows, ignore_index=True)
    return _finalize_strategy_label_output(output, policy)


def _build_chunked_swing_strategy_labels(
    frame: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    dataset_config: SwingDatasetConfig,
    policy: SwingStrategyLabelPolicy,
    strategy_ids: Sequence[str],
    sessions: Sequence[object],
    chunk_sessions: int,
) -> pd.DataFrame:
    maximum_horizon = max(
        policy.strategies[strategy_id].horizon_sessions
        for strategy_id in strategy_ids
    )
    breakout = policy.strategies["SWING.BREAKOUT_EXPANSION.5D.V1"]
    setup_lookback = max(
        _required_int_parameter(breakout.prior_high_sessions),
        _required_int_parameter(breakout.compression_sessions),
    )
    parts: list[pd.DataFrame] = []
    for core_start in range(0, len(sessions), chunk_sessions):
        core_end = min(core_start + chunk_sessions, len(sessions))
        source_start = max(0, core_start - setup_lookback)
        source_end = min(len(sessions), core_end + maximum_horizon)
        source_sessions = set(sessions[source_start:source_end])
        core_sessions = set(sessions[core_start:core_end])
        source = frame.loc[
            frame["session_date_et"].isin(source_sessions)
        ]
        labelled = build_swing_strategy_labels(
            source,
            benchmark_bars,
            dataset_config=dataset_config,
            policy=policy,
            strategy_ids=strategy_ids,
            _chunk_sessions=None,
        )
        part = labelled.loc[
            labelled["session_date_et"].isin(core_sessions)
        ].copy()
        part.drop(
            columns=[
                "strategy_label_material_sha256",
                "strategy_label_reconciliation_sha256",
                "strategy_label_reconciliation_errors",
            ],
            inplace=True,
        )
        parts.append(part)
        labelled = None
        gc.collect()
    return _finalize_strategy_label_output(
        pd.concat(parts, ignore_index=True),
        policy,
    )


def _finalize_strategy_label_output(
    output: pd.DataFrame,
    policy: SwingStrategyLabelPolicy,
) -> pd.DataFrame:
    output.drop(
        columns="_strategy_source_row_id",
        inplace=True,
        errors="ignore",
    )
    material_hash = label_material_sha256(
        output,
        identity_columns=(
            "strategy_id",
            "security_id",
            "decision_time_utc",
        ),
        material_columns=STRATEGY_LABEL_MATERIAL_COLUMNS,
    )
    if not material_hash:
        raise DataReadinessError(
            "strategy label material hash could not be produced"
        )
    output["strategy_label_material_sha256"] = material_hash
    output["strategy_label_reconciliation_sha256"] = policy_sha256(
        {
            "schema": STRATEGY_LABEL_ARTIFACT_SCHEMA,
            "policy_sha256": policy.sha256(),
            "execution_policy_sha256": policy.execution_policy_sha256,
            "material_sha256": material_hash,
        }
    )
    output["strategy_label_reconciliation_errors"] = 0
    assert_memory_budget(
        hard_budget_gib=policy.maximum_process_memory_gib,
        headroom_gib=policy.memory_guard_headroom_gib,
        stage="strategy label build complete",
    )
    return output.sort_values(
        ["strategy_id", "decision_time_utc", "ticker"],
        kind="stable",
    ).reset_index(drop=True)


def audit_swing_strategy_labels(
    frame: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    dataset_config: SwingDatasetConfig,
    policy: SwingStrategyLabelPolicy,
    strategy_ids: Sequence[str] | None = None,
) -> CanonicalAuditReport:
    selected_ids = _selected_strategy_ids(policy, strategy_ids)
    reproduced = build_swing_strategy_labels(
        frame,
        benchmark_bars,
        dataset_config=dataset_config,
        policy=policy,
        strategy_ids=selected_ids,
    )
    replay_errors = replay_mismatch_count(
        labels,
        reproduced,
        identity_columns=(
            "strategy_id",
            "security_id",
            "decision_time_utc",
        ),
        material_columns=STRATEGY_LABEL_MATERIAL_COLUMNS,
    )
    schema_errors = int(
        set(labels["strategy_id"].astype(str)) != set(selected_ids)
    ) + int(
        set(labels["strategy_label_schema_version"].astype(str))
        != {STRATEGY_LABEL_ARTIFACT_SCHEMA}
    )
    future_setup_errors = int(
        (
            _strict_utc(
                labels["setup_feature_available_at_utc"],
                "setup_feature_available_at_utc",
            )
            > _strict_utc(labels["decision_time_utc"], "decision_time_utc")
        ).sum()
    )
    invalid_setup_reasons = int(
        (
            ~labels["setup_abstention_reason"]
            .astype(str)
            .isin(SETUP_ABSTENTION_REASONS)
        ).sum()
    )
    invalid_label_reasons = int(
        (
            ~labels["label_abstention_reason"]
            .astype(str)
            .isin(LABEL_ABSTENTION_REASONS)
        ).sum()
    )
    cost_errors = int(
        (
            labels["strategy_label_eligible"].fillna(False).astype(bool)
            & ~np.isclose(
                pd.to_numeric(
                    labels["strategy_net_return"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    labels["strategy_gross_return"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    labels["strategy_execution_cost_fraction"],
                    errors="coerce",
                ),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=False,
            )
        ).sum()
    )
    eligible = labels["strategy_label_eligible"].fillna(False).astype(bool)
    cost_evidence_errors = int(
        (
            eligible
            & (
                _num(labels, "entry_price").le(0)
                | _num(labels, "entry_atr_pct").isna()
                | _num(labels, "entry_atr_pct").lt(0)
                | _num(
                    labels,
                    "strategy_execution_cost_fraction",
                ).isna()
            )
        ).sum()
    )
    expected_target, expected_outcome = (
        _audit_strategy_target_contract(labels, policy)
    )
    target_contract_errors = int(
        (
            eligible
            & (
                pd.to_numeric(
                    labels["strategy_target"],
                    errors="coerce",
                ).ne(expected_target.astype(int))
                | labels["strategy_outcome"]
                .astype(str)
                .ne(expected_outcome)
            )
        ).sum()
    )
    catalyst_family = labels["strategy_family"].astype(str).isin(
        {"catalyst_continuation", "short_term_reversal"}
    )
    catalyst_coverage_errors = int(
        (
            labels["setup_eligible"].fillna(False).astype(bool)
            & catalyst_family
            & ~labels["catalyst_source_complete"]
            .fillna(False)
            .astype(bool)
        ).sum()
    )
    breakout_execution_errors = _breakout_execution_error_count(labels)
    checks = (
        _check(
            "strategy_label_schema",
            schema_errors,
            len(labels),
            "all frozen KS2 strategy IDs are present",
        ),
        _check(
            "strategy_setup_causality",
            future_setup_errors,
            len(labels),
            "setup evidence is available by decision time",
        ),
        _check(
            "strategy_abstention_reasons",
            invalid_setup_reasons + invalid_label_reasons,
            len(labels),
            "setup and label abstentions use bounded reason codes",
        ),
        _check(
            "strategy_cost_once",
            cost_errors + cost_evidence_errors,
            len(labels),
            "net return equals gross return minus one evidenced bound execution cost",
        ),
        _check(
            "strategy_target_contract",
            target_contract_errors,
            len(labels),
            "targets and outcomes match the frozen strategy-specific objectives",
        ),
        _check(
            "strategy_catalyst_coverage",
            catalyst_coverage_errors,
            len(labels),
            "catalyst-dependent setups require causal source coverage",
        ),
        _check(
            "strategy_breakout_execution",
            breakout_execution_errors,
            len(labels),
            "breakout economics use executable barrier fills and matching outcomes",
        ),
        _check(
            "strategy_label_replay",
            replay_errors,
            len(labels),
            "strategy labels reproduce from decision-time features and exact paths",
        ),
    )
    return CanonicalAuditReport(checks=checks)


def _audit_strategy_target_contract(
    labels: pd.DataFrame,
    policy: SwingStrategyLabelPolicy,
) -> tuple[pd.Series, pd.Series]:
    expected_target = pd.Series(False, index=labels.index)
    expected_outcome = pd.Series("", index=labels.index, dtype="string")
    eligible = labels["strategy_label_eligible"].fillna(False).astype(bool)
    net = _num(labels, "strategy_net_return")
    spy = _num(labels, "strategy_excess_return_vs_spy")
    sector = _num(labels, "strategy_excess_return_vs_sector")
    for strategy_id in STRATEGY_IDS:
        selected = labels["strategy_id"].astype(str).eq(strategy_id)
        spec = policy.strategies[strategy_id]
        if spec.family == "cross_sectional_continuation":
            rank = spy.where(eligible & selected).groupby(
                labels["strategy_decision_group_id"]
            ).rank(method="average", pct=True)
            target = (
                net.gt(0)
                & spy.gt(0)
                & rank.ge(_required_parameter(spec.minimum_relative_rank))
            )
            positive, negative = (
                "top_relative_continuation",
                "relative_lag",
            )
        elif spec.family == "time_series_continuation":
            target = net.gt(0) & spy.gt(0)
            positive, negative = "trend_continuation", "trend_failure"
        elif spec.family == "catalyst_continuation":
            target = net.gt(0) & spy.gt(0) & sector.gt(0)
            positive, negative = "catalyst_drift", "catalyst_fade"
        elif spec.family == "short_term_reversal":
            target = net.gt(0) & sector.gt(0)
            positive, negative = (
                "overreaction_reversal",
                "continued_weakness",
            )
        elif spec.family == "breakout_expansion":
            target = (
                labels["barrier_outcome"].astype(str).eq("target_first")
                & net.gt(0)
            )
            positive, negative = (
                "breakout_continuation",
                "failed_breakout",
            )
        elif spec.family == "sector_residual_continuation":
            target = sector.gt(0) & spy.gt(0)
            positive, negative = (
                "residual_continuation",
                "residual_decay",
            )
        else:
            raise AssertionError(
                f"unhandled strategy family: {spec.family}"
            )
        expected_target.loc[selected] = target.loc[selected]
        expected_outcome.loc[selected] = np.where(
            target.loc[selected],
            positive,
            negative,
        )
    return expected_target, expected_outcome


def _breakout_execution_error_count(labels: pd.DataFrame) -> int:
    selected = (
        labels["strategy_family"].astype(str).eq("breakout_expansion")
        & labels["strategy_label_eligible"].fillna(False).astype(bool)
    )
    if not bool(selected.any()):
        return 0
    part = labels.loc[selected]
    entry = _num(part, "entry_price")
    realized = _num(part, "barrier_realized_price")
    gross = _num(part, "strategy_gross_return")
    net = _num(part, "strategy_net_return")
    cost = _num(part, "strategy_execution_cost_fraction")
    expected_target = (
        part["barrier_outcome"].astype(str).eq("target_first")
        & net.gt(0)
    )
    failures = (
        ~part["barrier_outcome"]
        .astype(str)
        .isin({"target_first", "stop_first", "timeout"})
        | _num(part, "barrier_outcome_session").lt(1)
        | _num(part, "barrier_outcome_session").gt(
            _num(part, "strategy_horizon_sessions")
        )
        | ~pd.Series(
            np.isclose(
                _num(part, "exit_price"),
                realized,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=False,
            ),
            index=part.index,
        )
        | ~pd.Series(
            np.isclose(
                gross,
                realized / entry - 1.0,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=False,
            ),
            index=part.index,
        )
        | ~pd.Series(
            np.isclose(
                net,
                gross - cost,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=False,
            ),
            index=part.index,
        )
        | part["breakout_failed"]
        .fillna(True)
        .astype(bool)
        .ne(~expected_target)
    )
    return int(failures.sum())


def build_swing_strategy_label_bundle(
    frame: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    dataset_config: SwingDatasetConfig,
    policy: SwingStrategyLabelPolicy,
    out_dir: Path,
    input_hashes: Mapping[str, str],
    production_ready: bool = False,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Publish one independently audited artifact per strategy."""

    request = {
        "schema": STRATEGY_LABEL_BUNDLE_SCHEMA,
        "strategy_label_policy_sha256": policy.sha256(),
        "strategy_label_evaluator_sha256": (
            swing_strategy_evaluator_sha256()
        ),
        "execution_policy_sha256": policy.execution_policy_sha256,
        "dataset_config_sha256": policy_sha256(
            dataset_config.model_dump(mode="json")
        ),
        "inputs": dict(sorted(input_hashes.items())),
        "production_ready": production_ready,
    }
    request_sha256 = policy_sha256(request)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "_manifest.json"
    _write_or_validate_json(
        out_dir / "_request.json",
        {**request, "request_sha256": request_sha256},
    )
    artifact_dir = out_dir / "strategies"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        return _validate_completed_strategy_bundle(
            final_path,
            artifact_dir=artifact_dir,
            request_sha256=request_sha256,
            production_ready=production_ready,
        )
    records: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    skipped = 0
    for index, strategy_id in enumerate(STRATEGY_IDS, start=1):
        target = artifact_dir / (
            strategy_id.lower().replace(".", "_") + ".parquet"
        )
        labels: pd.DataFrame | None = None
        was_skipped = False
        try:
            artifact_exists = target.exists()
            manifest_exists = manifest_path_for(target).exists()
            if artifact_exists and not manifest_exists:
                target.unlink()
                artifact_exists = False
            elif manifest_exists and not artifact_exists:
                raise DataReadinessError(
                    "committed strategy label artifact data is missing: "
                    f"{target}"
                )
            if artifact_exists:
                labels, manifest = load_canonical_artifact(
                    target,
                    expected_type="swing_strategy_labels",
                    allow_research=not production_ready,
                )
                manifest_inputs = manifest.get("inputs")
                if (
                    not isinstance(manifest_inputs, dict)
                    or manifest_inputs.get(
                        "strategy_label_bundle_request_sha256"
                    )
                    != request_sha256
                    or manifest_inputs.get("strategy_id") != strategy_id
                ):
                    raise DataReadinessError(
                        f"existing strategy label lineage mismatch: {target}"
                    )
                _require_passed_strategy_audit(manifest, target)
                was_skipped = True
                skipped += 1
            else:
                labels = build_swing_strategy_labels(
                    frame,
                    benchmark_bars,
                    dataset_config=dataset_config,
                    policy=policy,
                    strategy_ids=(strategy_id,),
                )
                audit = audit_swing_strategy_labels(
                    frame,
                    benchmark_bars,
                    labels,
                    dataset_config=dataset_config,
                    policy=policy,
                    strategy_ids=(strategy_id,),
                )
                manifest = write_canonical_artifact(
                    labels,
                    target,
                    artifact_type="swing_strategy_labels",
                    audit=audit,
                    inputs={
                        **dict(input_hashes),
                        "strategy_label_bundle_request_sha256": (
                            request_sha256
                        ),
                        "strategy_label_policy_sha256": policy.sha256(),
                        "strategy_label_evaluator_sha256": str(
                            request[
                                "strategy_label_evaluator_sha256"
                            ]
                        ),
                        "strategy_id": strategy_id,
                    },
                    production_ready=production_ready,
                )
            records.append(
                {
                    "strategy_id": strategy_id,
                    "path": str(target.resolve()),
                    "sha256": str(manifest["artifact_sha256"]),
                    "rows": len(labels),
                    "setup_eligible_rows": int(
                        labels["setup_eligible"].fillna(False).sum()
                    ),
                    "label_eligible_rows": int(
                        labels["strategy_label_eligible"]
                        .fillna(False)
                        .sum()
                    ),
                    "strategy_label_material_sha256": _constant_text(
                        labels,
                        "strategy_label_material_sha256",
                    ),
                    "strategy_label_policy_sha256": policy.strategy_sha256(
                        strategy_id
                    ),
                }
            )
            if progress is not None:
                progress(
                    {
                        "index": index,
                        "total": len(STRATEGY_IDS),
                        "strategy_id": strategy_id,
                        "status": (
                            "skipped"
                            if was_skipped
                            else "observed"
                        ),
                        "rows": len(labels),
                    }
                )
        except Exception as exc:
            failures[strategy_id] = (
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            if progress is not None:
                progress(
                    {
                        "index": index,
                        "total": len(STRATEGY_IDS),
                        "strategy_id": strategy_id,
                        "status": "failed",
                        "rows": 0,
                    }
                )
        finally:
            labels = None
            gc.collect()
            release_process_memory()
            assert_memory_budget(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
                stage=f"strategy label bundle {strategy_id}",
            )
            _assert_peak_memory_budget(policy, strategy_id)
    status = (
        "complete"
        if not failures and len(records) == len(STRATEGY_IDS)
        else "incomplete"
    )
    result: dict[str, object] = {
        "schema": STRATEGY_LABEL_BUNDLE_SCHEMA,
        "request_sha256": request_sha256,
        "status": status,
        "requested_strategies": len(STRATEGY_IDS),
        "observed_strategies": len(records),
        "skipped_strategies": skipped,
        "failed_strategies": failures,
        "rows": sum(_record_int(record, "rows") for record in records),
        "setup_eligible_rows": sum(
            _record_int(record, "setup_eligible_rows")
            for record in records
        ),
        "label_eligible_rows": sum(
            _record_int(record, "label_eligible_rows")
            for record in records
        ),
        "artifacts": records,
        "memory": memory_audit(
            hard_budget_gib=policy.maximum_process_memory_gib,
            headroom_gib=policy.memory_guard_headroom_gib,
        ).to_record(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "production_ready": production_ready,
    }
    _atomic_json(out_dir / "_status.json", result)
    if status == "complete":
        _atomic_json(final_path, result)
    return result


def _validate_completed_strategy_bundle(
    final_path: Path,
    *,
    artifact_dir: Path,
    request_sha256: str,
    production_ready: bool,
) -> dict[str, object]:
    try:
        loaded = json.loads(final_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"completed strategy bundle is unreadable: {final_path}"
        ) from exc
    if (
        not isinstance(loaded, dict)
        or loaded.get("schema") != STRATEGY_LABEL_BUNDLE_SCHEMA
        or loaded.get("status") != "complete"
        or loaded.get("request_sha256") != request_sha256
        or bool(loaded.get("production_ready")) != production_ready
    ):
        raise DataReadinessError(
            f"completed strategy bundle lineage mismatch: {final_path}"
        )
    artifacts = loaded.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(
        STRATEGY_IDS
    ):
        raise DataReadinessError(
            f"completed strategy bundle artifact inventory is invalid: {final_path}"
        )
    by_strategy = {
        str(record.get("strategy_id")): record
        for record in artifacts
        if isinstance(record, dict)
    }
    if set(by_strategy) != set(STRATEGY_IDS):
        raise DataReadinessError(
            f"completed strategy bundle strategy inventory is invalid: {final_path}"
        )
    for strategy_id in STRATEGY_IDS:
        record = by_strategy[strategy_id]
        target = artifact_dir / (
            strategy_id.lower().replace(".", "_") + ".parquet"
        )
        if Path(str(record.get("path", ""))).resolve() != target.resolve():
            raise DataReadinessError(
                f"completed strategy bundle path mismatch: {strategy_id}"
            )
        labels, manifest = load_canonical_artifact(
            target,
            expected_type="swing_strategy_labels",
            allow_research=not production_ready,
        )
        manifest_inputs = manifest.get("inputs")
        if (
            not isinstance(manifest_inputs, dict)
            or manifest_inputs.get(
                "strategy_label_bundle_request_sha256"
            )
            != request_sha256
            or manifest_inputs.get("strategy_id") != strategy_id
            or str(manifest.get("artifact_sha256"))
            != str(record.get("sha256"))
            or len(labels) != _record_int(record, "rows")
            or _constant_text(
                labels,
                "strategy_label_material_sha256",
            )
            != str(record.get("strategy_label_material_sha256"))
        ):
            raise DataReadinessError(
                f"completed strategy bundle artifact mismatch: {strategy_id}"
            )
        _require_passed_strategy_audit(manifest, target)
        labels = None
        gc.collect()
        release_process_memory()
    return {str(key): value for key, value in loaded.items()}


def _assert_peak_memory_budget(
    policy: SwingStrategyLabelPolicy,
    strategy_id: str,
) -> None:
    observed = memory_audit(
        hard_budget_gib=policy.maximum_process_memory_gib,
        headroom_gib=policy.memory_guard_headroom_gib,
    )
    peak = observed.peak_working_set_gib
    if peak is None:
        raise DataReadinessError(
            "memory guard cannot verify peak working set for "
            f"{strategy_id}"
        )
    if peak > observed.safety_threshold_gib:
        raise DataReadinessError(
            "memory guard stopped strategy label bundle "
            f"{strategy_id}: peak RSS {peak:.3f} GiB exceeds "
            f"the {observed.safety_threshold_gib:.3f} GiB safety threshold"
        )


def _selected_strategy_ids(
    policy: SwingStrategyLabelPolicy,
    strategy_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = tuple(strategy_ids) if strategy_ids is not None else STRATEGY_IDS
    if not selected:
        raise DataReadinessError("at least one strategy_id is required")
    if len(selected) != len(set(selected)):
        raise DataReadinessError("strategy_ids contain duplicates")
    unknown = sorted(set(selected).difference(STRATEGY_IDS))
    if unknown:
        raise DataReadinessError(
            "unsupported strategy_ids: " + ", ".join(unknown)
        )
    missing_policy = sorted(set(selected).difference(policy.strategies))
    if missing_policy:
        raise DataReadinessError(
            "strategy policy is missing: " + ", ".join(missing_policy)
        )
    selected_set = set(selected)
    return tuple(
        strategy_id
        for strategy_id in STRATEGY_IDS
        if strategy_id in selected_set
    )


def _strategy_source_columns(
    frame: pd.DataFrame,
    policy: SwingStrategyLabelPolicy,
    strategy_ids: Sequence[str],
) -> list[str]:
    retained = {
        *_COMMON_REQUIRED_COLUMNS,
        "cross_section_eligible",
        "membership_effective_to_utc",
    }
    for strategy_id in strategy_ids:
        retained.update(policy.strategies[strategy_id].required_features)
    return [
        column
        for column in frame.columns
        if column in retained
    ]


def _constant_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame or frame.empty:
        raise DataReadinessError(
            f"strategy label artifact has no {column} value"
        )
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1 or not values[0]:
        raise DataReadinessError(
            f"strategy label artifact has non-constant {column}"
        )
    return str(values[0])


def _require_passed_strategy_audit(
    manifest: Mapping[str, object],
    path: Path,
) -> None:
    audit = manifest.get("audit")
    if not isinstance(audit, list):
        raise DataReadinessError(
            f"strategy label manifest has no audit evidence: {path}"
        )
    observed: dict[str, str] = {}
    for record in audit:
        if not isinstance(record, dict):
            raise DataReadinessError(
                f"strategy label manifest has malformed audit evidence: {path}"
            )
        name = record.get("name")
        status = record.get("status")
        if isinstance(name, str) and isinstance(status, str):
            observed[name] = status
    required = {
        "strategy_label_schema",
        "strategy_setup_causality",
        "strategy_abstention_reasons",
        "strategy_cost_once",
        "strategy_target_contract",
        "strategy_catalyst_coverage",
        "strategy_breakout_execution",
        "strategy_label_replay",
    }
    failed = sorted(
        name
        for name in required
        if observed.get(name) != "pass"
    )
    if failed:
        raise DataReadinessError(
            "strategy label manifest audit is incomplete or failed "
            f"for {path}: {', '.join(failed)}"
        )


def _record_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataReadinessError(f"bundle record has invalid {key}")
    return value


def _write_or_validate_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataReadinessError(
                f"strategy label request is unreadable: {path}"
            ) from exc
        if existing != dict(payload):
            raise DataReadinessError(
                f"strategy label request lineage mismatch: {path}"
            )
        return
    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strategy_condition(
    data: pd.DataFrame,
    spec: SwingStrategySpec,
) -> tuple[pd.Series, str]:
    if spec.family == "cross_sectional_continuation":
        return (
            _num(data, "return_20d").ge(
                _required_parameter(spec.minimum_return_20d)
            )
            & _num(data, "xs_rank_rel_return_20d_vs_sector").ge(
                _required_parameter(spec.minimum_relative_rank)
            ),
            "trend_not_confirmed",
        )
    if spec.family == "time_series_continuation":
        return (
            _num(data, "return_20d").ge(
                _required_parameter(spec.minimum_return_20d)
            )
            & _num(data, "dist_sma_50").ge(
                _required_parameter(spec.minimum_dist_sma_50)
            )
            & _num(data, "dist_sma_200").ge(
                _required_parameter(spec.minimum_dist_sma_200)
            )
            & _num(data, "sma_200_slope_20d").ge(
                _required_parameter(spec.minimum_sma_200_slope_20d)
            ),
            "trend_not_confirmed",
        )
    if spec.family == "catalyst_continuation":
        coverage = _bool(data, "catalyst_source_complete")
        catalyst = (
            _num(data, "event_count_3d").ge(
                _required_parameter(spec.minimum_event_count_3d)
            )
            & _num(data, "sentiment_coverage_3d").ge(
                _required_parameter(spec.minimum_sentiment_coverage_3d)
            )
            & _num(data, "event_relevance_mean_3d").ge(
                _required_parameter(
                    spec.minimum_event_relevance_mean_3d
                )
            )
            & _num(data, "dist_ema_20").ge(
                _required_parameter(spec.minimum_dist_ema_20)
            )
            & _num(data, "return_5d").ge(
                _required_parameter(spec.minimum_return_5d)
            )
        )
        return coverage & catalyst, "direct_catalyst_not_confirmed"
    if spec.family == "short_term_reversal":
        coverage = _bool(data, "catalyst_source_complete")
        atr = _num(data, "atr_pct_14")
        return (
            coverage
            & atr.gt(0)
            & _num(data, "return_5d").le(
                _required_parameter(spec.maximum_return_to_atr) * atr
            )
            & _num(data, "rsi_14").le(
                _required_parameter(spec.maximum_rsi_14)
            )
            & _num(data, "event_count_3d").le(
                _required_parameter(spec.maximum_event_count_3d)
            ),
            "overreaction_not_confirmed",
        )
    if spec.family == "breakout_expansion":
        return (
            _num(data, "_prior_high").notna()
            & _num(data, "_prior_compression_range").le(
                _required_parameter(spec.maximum_compression_range)
            )
            & _num(data, "close").gt(_num(data, "_prior_high"))
            & _num(data, "volume_ratio_20").ge(
                _required_parameter(spec.minimum_volume_ratio_20)
            )
            & _num(data, "close_location").ge(
                _required_parameter(spec.minimum_close_location)
            ),
            "breakout_not_confirmed",
        )
    if spec.family == "sector_residual_continuation":
        return (
            _num(data, "rel_return_20d_vs_sector").ge(
                _required_parameter(spec.minimum_residual_return_20d)
            )
            & _num(data, "xs_rank_rel_return_20d_vs_sector").ge(
                _required_parameter(spec.minimum_relative_rank)
            ),
            "residual_strength_not_confirmed",
        )
    raise AssertionError(f"unhandled strategy family: {spec.family}")


def _strategy_target_and_outcome(
    part: pd.DataFrame,
    spec: SwingStrategySpec,
) -> tuple[pd.Series, pd.Series]:
    net = _num(part, "strategy_net_return")
    spy = _num(part, "strategy_excess_return_vs_spy")
    sector = _num(part, "strategy_excess_return_vs_sector")
    if spec.family == "cross_sectional_continuation":
        eligible = part["strategy_label_eligible"].fillna(False).astype(bool)
        rank = spy.where(eligible).groupby(
            part["strategy_decision_group_id"]
        ).rank(
            method="average",
            pct=True,
        )
        target = net.gt(0) & spy.gt(0) & rank.ge(
            _required_parameter(spec.minimum_relative_rank)
        )
        return target, pd.Series(
            np.where(target, "top_relative_continuation", "relative_lag"),
            index=part.index,
        )
    if spec.family == "time_series_continuation":
        target = net.gt(0) & spy.gt(0)
        return target, pd.Series(
            np.where(target, "trend_continuation", "trend_failure"),
            index=part.index,
        )
    if spec.family == "catalyst_continuation":
        target = net.gt(0) & spy.gt(0) & sector.gt(0)
        return target, pd.Series(
            np.where(target, "catalyst_drift", "catalyst_fade"),
            index=part.index,
        )
    if spec.family == "short_term_reversal":
        target = net.gt(0) & sector.gt(0)
        return target, pd.Series(
            np.where(target, "overreaction_reversal", "continued_weakness"),
            index=part.index,
        )
    if spec.family == "breakout_expansion":
        target = part["barrier_outcome"].astype(str).eq(
            "target_first"
        ) & net.gt(0)
        return target, pd.Series(
            np.where(target, "breakout_continuation", "failed_breakout"),
            index=part.index,
        )
    if spec.family == "sector_residual_continuation":
        target = sector.gt(0) & spy.gt(0)
        return target, pd.Series(
            np.where(target, "residual_continuation", "residual_decay"),
            index=part.index,
        )
    raise AssertionError(f"unhandled strategy family: {spec.family}")


def _add_breakout_setup_features(
    data: pd.DataFrame,
    policy: SwingStrategyLabelPolicy,
) -> pd.DataFrame:
    spec = policy.strategies["SWING.BREAKOUT_EXPANSION.5D.V1"]
    prior_high_sessions = _required_int_parameter(spec.prior_high_sessions)
    compression_sessions = _required_int_parameter(
        spec.compression_sessions
    )
    ordered = data.sort_values(
        ["security_id", "session_date_et"],
        kind="stable",
    ).copy()
    grouped = ordered.groupby("security_id", sort=False)
    prior_high = grouped["high"].transform(
        lambda values: values.shift(1)
        .rolling(prior_high_sessions, min_periods=prior_high_sessions)
        .max()
    )
    compression_high = grouped["high"].transform(
        lambda values: values.shift(1)
        .rolling(compression_sessions, min_periods=compression_sessions)
        .max()
    )
    compression_low = grouped["low"].transform(
        lambda values: values.shift(1)
        .rolling(compression_sessions, min_periods=compression_sessions)
        .min()
    )
    ordered["_prior_high"] = prior_high
    ordered["_prior_compression_range"] = (
        compression_high / compression_low - 1.0
    )
    return ordered.sort_index()


def _future_path_arrays(
    source: pd.DataFrame,
    horizon: int,
) -> dict[str, np.ndarray]:
    ordered = source.sort_values(
        ["security_id", "session_date_et"],
        kind="stable",
    ).copy()
    grouped = ordered.groupby("security_id", sort=False)
    output: dict[str, np.ndarray] = {}
    for column in ("open", "high", "low", "close"):
        matrix = pd.concat(
            [
                grouped[column].shift(-offset)
                for offset in range(1, horizon + 1)
            ],
            axis=1,
        )
        aligned = pd.DataFrame(
            matrix.to_numpy(float),
            index=ordered["_strategy_source_row_id"].astype(int),
        ).reindex(source["_strategy_source_row_id"].astype(int))
        output[column] = aligned.to_numpy(float)
    for column in (
        "session_date_et",
        "bar_end_utc",
        "available_at_utc",
    ):
        matrix = pd.concat(
            [
                grouped[column].shift(-offset)
                for offset in range(1, horizon + 1)
            ],
            axis=1,
        )
        aligned = pd.DataFrame(
            matrix.to_numpy(),
            index=ordered["_strategy_source_row_id"].astype(int),
        ).reindex(source["_strategy_source_row_id"].astype(int))
        output[column] = aligned.to_numpy()
    return output


def _add_breakout_path_labels(
    part: pd.DataFrame,
    base: pd.DataFrame,
    *,
    source_ids: np.ndarray,
    path: Mapping[str, np.ndarray],
    spec: SwingStrategySpec,
    benchmark_bars: pd.DataFrame,
    dataset_config: SwingDatasetConfig,
) -> None:
    entry = _num(part, "entry_price").to_numpy(float)
    atr_dollars = (
        _num(base, "atr_pct_14").to_numpy(float)
        * _num(base, "close").to_numpy(float)
    )
    selected_path = {
        name: values[source_ids]
        for name, values in path.items()
    }
    valid = (
        np.isfinite(entry)
        & (entry > 0)
        & np.isfinite(atr_dollars)
        & (atr_dollars > 0)
        & (
            entry
            - _required_parameter(spec.stop_atr) * atr_dollars
            > 0
        )
        & np.isfinite(selected_path["open"]).all(axis=1)
        & np.isfinite(selected_path["high"]).all(axis=1)
        & np.isfinite(selected_path["low"]).all(axis=1)
        & np.isfinite(selected_path["close"]).all(axis=1)
    )
    if not bool(valid.any()):
        return
    indices = np.flatnonzero(valid)
    evaluated = evaluate_intraday_barrier_paths(
        path_open=selected_path["open"][indices],
        path_high=selected_path["high"][indices],
        path_low=selected_path["low"][indices],
        path_close=selected_path["close"][indices],
        entry_atr=atr_dollars[indices],
        target_atr=_required_parameter(spec.target_atr),
        stop_atr=_required_parameter(spec.stop_atr),
        round_trip_cost_bps=0.0,
    )
    part.loc[indices, "barrier_outcome"] = evaluated.outcome
    part.loc[indices, "barrier_outcome_session"] = (
        evaluated.outcome_offset + 1
    )
    part.loc[indices, "barrier_target_price"] = (
        evaluated.target_price
    )
    part.loc[indices, "barrier_stop_price"] = evaluated.stop_price
    part.loc[indices, "barrier_realized_price"] = (
        evaluated.realized_price
    )
    outcome_offsets = evaluated.outcome_offset
    exit_sessions = selected_path["session_date_et"][
        indices,
        outcome_offsets,
    ]
    entry_sessions = selected_path["session_date_et"][indices, 0]
    part.loc[indices, "exit_time_utc"] = pd.to_datetime(
        selected_path["bar_end_utc"][indices, outcome_offsets],
        utc=True,
    ).array
    part.loc[indices, "label_available_at_utc"] = pd.to_datetime(
        selected_path["available_at_utc"][indices, outcome_offsets],
        utc=True,
    ).array
    part.loc[indices, "exit_price"] = evaluated.realized_price
    execution_cost = _num(
        part.loc[indices],
        "strategy_execution_cost_fraction",
    ).to_numpy(float)
    barrier_net = evaluated.gross_return - execution_cost
    part.loc[indices, "strategy_gross_return"] = evaluated.gross_return
    part.loc[indices, "strategy_net_return"] = barrier_net
    part.loc[indices, "strategy_mfe"] = evaluated.mfe
    part.loc[indices, "strategy_mae"] = evaluated.mae
    spy_return = _benchmark_interval_return(
        benchmark_bars,
        np.full(len(indices), dataset_config.broad_benchmark.upper()),
        entry_sessions,
        exit_sessions,
    )
    qqq_return = _benchmark_interval_return(
        benchmark_bars,
        np.full(len(indices), dataset_config.growth_benchmark.upper()),
        entry_sessions,
        exit_sessions,
    )
    sector_return = _benchmark_interval_return(
        benchmark_bars,
        base.loc[indices, "primary_benchmark"].to_numpy(str),
        entry_sessions,
        exit_sessions,
    )
    part.loc[indices, "strategy_spy_return"] = spy_return
    part.loc[indices, "strategy_qqq_return"] = qqq_return
    part.loc[indices, "strategy_sector_return"] = sector_return
    part.loc[indices, "strategy_excess_return_vs_spy"] = (
        barrier_net - spy_return
    )
    part.loc[indices, "strategy_excess_return_vs_qqq"] = (
        barrier_net - qqq_return
    )
    part.loc[indices, "strategy_excess_return_vs_sector"] = (
        barrier_net - sector_return
    )
    part.loc[indices, "breakout_failed"] = ~(
        evaluated.target_first & (barrier_net > 0)
    )


def _benchmark_interval_return(
    benchmark_bars: pd.DataFrame,
    tickers: np.ndarray,
    entry_sessions: np.ndarray,
    exit_sessions: np.ndarray,
) -> np.ndarray:
    lookup = benchmark_bars.set_index(["ticker", "session_date_et"])
    entry_index = pd.MultiIndex.from_arrays(
        [pd.Series(tickers).astype(str).str.upper(), entry_sessions],
        names=lookup.index.names,
    )
    exit_index = pd.MultiIndex.from_arrays(
        [pd.Series(tickers).astype(str).str.upper(), exit_sessions],
        names=lookup.index.names,
    )
    entry_open = pd.to_numeric(
        lookup["open"].reindex(entry_index),
        errors="coerce",
    ).to_numpy(float)
    exit_close = pd.to_numeric(
        lookup["close"].reindex(exit_index),
        errors="coerce",
    ).to_numpy(float)
    output = np.full(len(tickers), np.nan)
    valid = (
        np.isfinite(entry_open)
        & (entry_open > 0)
        & np.isfinite(exit_close)
        & (exit_close > 0)
    )
    output[valid] = exit_close[valid] / entry_open[valid] - 1.0
    return output


def _reject_label_bearing_input(frame: pd.DataFrame) -> None:
    forbidden = sorted(
        column
        for column in frame
        if column in _LABEL_BEARING_COLUMNS
        or column.startswith(_LABEL_BEARING_PREFIXES)
    )
    if forbidden:
        raise DataReadinessError(
            "strategy setup input contains generic or future labels: "
            + ", ".join(forbidden[:20])
        )


def _set_reason(
    reason: pd.Series,
    failure: pd.Series | np.ndarray,
    code: str,
) -> None:
    mask = pd.Series(failure, index=reason.index).fillna(True).astype(bool)
    reason.loc[reason.eq("") & mask] = code


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return (
        values.astype("string")
        .str.lower()
        .isin({"true", "1", "yes", "observed_complete", "observed_empty"})
    )


def _strict_utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if bool(parsed.isna().any()):
        raise DataReadinessError(f"{name} contains invalid timestamps")
    return parsed


def _required_parameter(value: float | None) -> float:
    if value is None:
        raise AssertionError("validated strategy parameter is missing")
    return float(value)


def _required_int_parameter(value: int | None) -> int:
    if value is None:
        raise AssertionError("validated integer strategy parameter is missing")
    return int(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(
            f"{name} is missing columns: {', '.join(missing)}"
        )


def _check(
    name: str,
    failures: int,
    rows: int,
    detail: str,
) -> CanonicalAuditCheck:
    return CanonicalAuditCheck(
        name=name,
        status="pass" if failures == 0 else "fail",
        failures=failures,
        rows_checked=rows,
        detail=detail,
    )
