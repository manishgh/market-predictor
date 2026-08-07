"""Production-grade candidate training for the ten-session edge-rebuild swing strategy."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_artifact_contracts import (
    SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
    SWING_MATERIALIZATION_MANIFEST_SCHEMA,
)
from market_predictor.edge_rebuild.swing_features import (
    MANAGED_BENCHMARK_RETURN_COLUMNS,
    MANAGED_EXCESS_RETURN_COLUMNS,
    MANAGED_PATH_COST_POLICY,
    MANAGED_PATH_NET_RETURN_COLUMNS,
    MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
    SWING_CATALYST_FEATURE_PROFILE,
    SWING_FEATURE_PANEL_SCHEMA,
    SWING_FEATURE_PROFILE,
    swing_model_feature_columns,
)
from market_predictor.edge_rebuild.swing_selection import (
    EFFECTIVE_SECTOR_WEIGHT_COLUMN,
    select_constrained_swing_portfolio,
)
from market_predictor.edge_rebuild.temporal_manifest import (
    build_temporal_schedule,
    load_temporal_manifest_config,
)
from market_predictor.edge_rebuild.training.economics import (
    _daily_position_ledger,
    _economic_gate,
    _moving_block_bootstrap_mean_interval,
    _session_bootstrap,
    _session_economic_blocks,
    _stability_breakdown,
    _stability_summary,
    _year_breakdown,
)
from market_predictor.edge_rebuild.training.evaluation import (
    _calibration_bins,
    _expected_calibration_error,
    _overlap_audit,
)
from market_predictor.edge_rebuild.training.utils import (
    _finite,
    _mapping,
)
from market_predictor.edge_rebuild.training.walk_forward import (
    WalkForwardFold,
    _assert_label_purge,
    _governed_folds,
    _governed_model_sessions,
    _split_fit_calibration,
    _split_record,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

TRAINING_SCHEMA: Final = "edge_rebuild.swing_training.v4"
MODEL_SCHEMA: Final = "edge_rebuild.swing_candidate.v4"
EVALUATION_SCHEMA: Final = "edge_rebuild.swing_evaluation.v5"
MODEL_CARD_SCHEMA: Final = "edge_rebuild.swing_model_card.v5"
OUTPUT_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_candidate_authority.v4"
DECISION_START_DATE: Final = date(2019, 7, 9)
HORIZON_SESSIONS: Final = 10
ALLOWED_PROFILES: Final = (
    SWING_FEATURE_PROFILE,
    SWING_CATALYST_FEATURE_PROFILE,
)
# The learned families, per profile and per (rate, depth) point. `dual_hurdle`
# was dropped: it scored 0.452-0.462 AUC on the v12 run -- below chance -- had no
# test covering it, and its four slots pushed the grid past the contract's
# six-candidate experiment budget.
_XGB_GRID: Final = (
    ("xgbranker", "xgboost_ranker"),
    ("xgbregressor", "xgboost_regressor"),
)
_XGB_FAMILIES: Final = len(_XGB_GRID)
_MANIFEST_NAME: Final = "_manifest.json"
_AUTHORITY_NAME: Final = "_authority.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_TEXT_COLUMNS: Final = (
    "decision_id",
    "decision_group_id",
    "ticker",
    "security_id",
    "sector",
    "primary_benchmark",
    "market_regime",
)


@dataclass(frozen=True, slots=True)
class SwingTrainingConfig:
    """Frozen controls for sequential candidate fitting and temporal evaluation."""

    decision_start_date: str = "2019-07-09"
    horizon_sessions: int = 10
    calibration_fraction: float = 0.20
    minimum_calibration_sessions: int = 63
    minimum_rows: int = 100_000
    minimum_securities: int = 100
    maximum_trades_per_decision: int = 25
    probability_thresholds: tuple[float, ...] = (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
    )
    logistic_c_values: tuple[float, ...] = (1.0,)
    hgb_learning_rates: tuple[float, ...] = (0.05,)
    hgb_max_leaf_nodes: tuple[int, ...] = (15, 31)
    hgb_max_iter: int = 150
    hgb_max_bins: int = 127
    # Tree depth is a policy field rather than a literal in the grid builder. It
    # was hard-coded to (3, 5), which put half the candidate count outside the
    # budget arithmetic below and let the real grid grow to fourteen while the
    # constructor still reported six.
    xgb_max_depths: tuple[int, ...] = (3,)
    maximum_learned_candidates: int = 6
    bootstrap_samples: int = 2_000
    bootstrap_block_sessions: int = 20
    random_seed: int = 42
    expected_round_trip_cost_bps: float = 20.0
    maximum_process_memory_gib: float = 4.0
    memory_guard_headroom_gib: float = 0.75

    def __post_init__(self) -> None:
        if self.decision_start_date != DECISION_START_DATE.isoformat():
            raise ValueError("the frozen swing decision start is 2019-07-09")
        if self.horizon_sessions != HORIZON_SESSIONS:
            raise ValueError("the active swing strategy has an exact ten-session horizon")
        if not 0.10 <= self.calibration_fraction <= 0.35:
            raise ValueError("calibration_fraction must be between 0.10 and 0.35")
        if self.minimum_calibration_sessions < 20:
            raise ValueError("calibration requires at least twenty sessions")
        if self.minimum_rows < 1 or self.minimum_securities < 20:
            raise ValueError("training population minimums are invalid")
        if not 1 <= self.maximum_trades_per_decision <= 50:
            raise ValueError("maximum_trades_per_decision must be in [1, 50]")
        if not self.probability_thresholds or any(
            value <= 0.0 or value >= 1.0 for value in self.probability_thresholds
        ):
            raise ValueError("probability thresholds must be in (0, 1)")
        if tuple(sorted(set(self.probability_thresholds))) != self.probability_thresholds:
            raise ValueError("probability thresholds must be unique and ascending")
        if not self.logistic_c_values or any(value <= 0 for value in self.logistic_c_values):
            raise ValueError("logistic C values must be positive")
        if not self.hgb_learning_rates or any(value <= 0 for value in self.hgb_learning_rates):
            raise ValueError("hgb learning rates must be positive")
        if not self.hgb_max_leaf_nodes or any(value < 2 for value in self.hgb_max_leaf_nodes):
            raise ValueError("hgb max leaf nodes must be at least 2")
        if self.hgb_max_iter < 1:
            raise ValueError("hgb max iter must be at least 1")
        if not self.xgb_max_depths or any(value < 1 for value in self.xgb_max_depths):
            raise ValueError("xgb max depths must be at least 1")
        # This must equal len(_candidate_specs(self)). The previous formula
        # counted an HGB grid the builder no longer emits, so it undercounted the
        # real grid by more than half and the multiple-testing control it exists
        # to provide was not being applied.
        candidate_count = len(ALLOWED_PROFILES) * (
            len(self.logistic_c_values)
            + len(self.hgb_learning_rates) * len(self.xgb_max_depths) * _XGB_FAMILIES
        )
        if candidate_count > self.maximum_learned_candidates:
            raise ValueError("candidate grid exceeds the frozen sequential budget")
        if not 2_000 <= self.bootstrap_samples <= 10_000:
            raise ValueError("bootstrap_samples must be in [2000, 10000]")
        if not HORIZON_SESSIONS <= self.bootstrap_block_sessions <= 126:
            raise ValueError("bootstrap blocks must span 10 to 126 sessions")
        if self.expected_round_trip_cost_bps <= 0:
            raise ValueError("expected round-trip cost must be positive")
        if not 0 < self.maximum_process_memory_gib <= 5.0:
            raise ValueError("process memory hard limit must be in (0, 5] GiB")
        if not 0 < self.memory_guard_headroom_gib < self.maximum_process_memory_gib:
            raise ValueError("memory headroom must be below the hard limit")


def load_swing_training_config(path: Path) -> SwingTrainingConfig:
    """Load a complete policy; partial or unknown fields are rejected."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"swing training policy is unreadable: {path}") from exc
    payload = raw.get("training")
    if not isinstance(payload, Mapping):
        raise DataReadinessError("swing training policy requires a [training] table")
    expected = {field.name for field in fields(SwingTrainingConfig)}
    actual = {str(key) for key in payload}
    if actual != expected:
        raise DataReadinessError(
            "swing training policy fields differ; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    values = dict(payload)
    for name in (
        "probability_thresholds",
        "logistic_c_values",
        "hgb_learning_rates",
        "hgb_max_leaf_nodes",
    ):
        value = values[name]
        if not isinstance(value, list):
            raise DataReadinessError(f"swing training policy {name} must be an array")
        values[name] = tuple(value)
    try:
        return SwingTrainingConfig(**values)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("swing training policy is invalid") from exc


@dataclass(frozen=True, slots=True)
class SwingPanelBinding:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    authority_sha256: str
    request_sha256: str
    strategy_contract_sha256: str


@dataclass(frozen=True, slots=True)
class SwingProfileData:
    frame: pd.DataFrame
    profile: str
    feature_columns: tuple[str, ...]
    decision_ids_sha256: str
    panel: SwingPanelBinding


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    profile: str
    estimator_family: str
    hyperparameters: Mapping[str, float | int | str]


@dataclass(frozen=True, slots=True)
class FittedCandidate:
    estimator: Any
    calibrator: LogisticRegression
    fit_sessions: int
    calibration_sessions: int
    calibration_cutoff_utc: str


@dataclass(frozen=True, slots=True)
class SwingTrainingResult:
    output_directory: Path
    selected_candidate_id: str | None
    evaluation: Mapping[str, Any]
    model_card: Mapping[str, Any]


def load_complete_swing_feature_panel(directory: Path) -> dict[str, Any]:
    """Load training-only materialization code without polluting serving imports."""

    from market_predictor.edge_rebuild.swing_materialization import (
        load_complete_swing_feature_panel as load_materialized_panel,
    )

    return load_materialized_panel(directory)


def load_swing_panel_binding(
    directory: Path,
    *,
    strategy_contract: StrategyContract,
    config: SwingTrainingConfig,
) -> SwingPanelBinding:
    """Verify the immutable panel and bind it to the active strategy contract."""

    root = directory.resolve()
    manifest = load_complete_swing_feature_panel(root)
    final = root / "final"
    manifest_path = final / _MANIFEST_NAME
    authority_path = final / _AUTHORITY_NAME
    authority = _read_json(authority_path, "swing panel authority")
    if manifest.get("schema") != SWING_MATERIALIZATION_MANIFEST_SCHEMA:
        raise DataReadinessError("only the current edge-rebuild swing panel is accepted")
    if authority.get("schema") != SWING_MATERIALIZATION_AUTHORITY_SCHEMA:
        raise DataReadinessError("swing panel authority schema is not current")
    if authority.get("state") != "complete":
        raise DataReadinessError("swing panel authority is not complete")
    if authority.get("artifact_sha256") != file_sha256(manifest_path):
        raise DataReadinessError("swing panel authority does not bind its manifest")
    if manifest.get("strategy_contract_sha256") != strategy_contract.sha256():
        raise DataReadinessError("swing panel strategy contract differs from training")
    if manifest.get("feature_profiles") != list(ALLOWED_PROFILES):
        raise DataReadinessError("swing panel must contain the two frozen ablation profiles")
    if manifest.get("required_historical_model_sources") != ["alpaca"]:
        raise DataReadinessError("Alpaca must be the only ticker catalyst estimator source")
    model_inputs = {str(value).lower() for value in manifest.get("catalyst_model_feature_inputs", [])}
    if any(_is_unapproved_source_feature(value) for value in model_inputs):
        raise DataReadinessError("unapproved source entered the swing estimator contract")
    if str(manifest.get("first_session")) != config.decision_start_date:
        raise DataReadinessError(
            "swing panel decisions must start exactly on 2019-07-09; "
            "pre-cutoff bars may exist only in the upstream warm-up store"
        )
    if int(manifest.get("rows_per_ablation_panel", -1)) < config.minimum_rows:
        raise DataReadinessError("swing panel has too few rows for training")
    if int(manifest.get("securities", -1)) < config.minimum_securities:
        raise DataReadinessError("swing panel has too few securities for training")
    if int(manifest.get("rows", -1)) * 2 != int(manifest.get("total_ablation_rows", -2)):
        raise DataReadinessError("swing ablation populations are not row-count matched")
    request_sha256 = str(manifest.get("request_sha256", ""))
    if len(request_sha256) != 64:
        raise DataReadinessError("swing panel request hash is invalid")
    return SwingPanelBinding(
        root=root,
        manifest=manifest,
        manifest_sha256=file_sha256(manifest_path),
        authority_sha256=file_sha256(authority_path),
        request_sha256=request_sha256,
        strategy_contract_sha256=strategy_contract.sha256(),
    )


def load_swing_profile(
    binding: SwingPanelBinding,
    profile: str,
    *,
    strategy_contract: StrategyContract,
    config: SwingTrainingConfig,
    sessions: tuple[str, ...],
) -> SwingProfileData:
    """Project one ablation profile into a bounded, strictly validated frame."""

    if profile not in ALLOWED_PROFILES:
        raise DataReadinessError(f"unsupported swing profile: {profile}")
    if not sessions or len(sessions) != len(set(sessions)):
        raise DataReadinessError("swing profile requires unique governed sessions")
    requested_dates = tuple(date.fromisoformat(value) for value in sessions)
    catalyst = profile == SWING_CATALYST_FEATURE_PROFILE
    feature_columns = swing_model_feature_columns(
        contract=strategy_contract,
        catalyst=catalyst,
    )
    if any(_is_unapproved_source_feature(column) for column in feature_columns):
        raise DataReadinessError("unapproved source-specific estimator feature detected")
    required = tuple(
        dict.fromkeys(
            (
                *_TEXT_COLUMNS,
                "session_date_et",
                "decision_time_utc",
                "feature_available_at_utc",
                "label_available_at_utc",
                "membership_effective_from_utc",
                "membership_effective_to_utc",
                "membership_available_at_utc",
                "entry_time_utc",
                "barrier_exit_session_date_et",
                "barrier_label_available_at_utc",
                "horizon_sessions",
                "feature_eligible",
                "label_eligible",
                "cross_section_eligible",
                "barrier_label",
                "rank_label",
                "ranking_group_size",
                "ranking_reliability_weight",
                "sector_peer_count",
                "sector_rank_eligible",
                "sector_rank_target_met",
                "barrier_holding_sessions",
                "barrier_gross_return",
                "barrier_cost",
                "barrier_net_return",
                "future_gross_return_10d",
                "future_net_return_10d",
                "future_spy_return_10d",
                "future_qqq_return_10d",
                "future_sector_return_10d",
                "future_excess_return_10d_vs_spy",
                "future_excess_return_10d_vs_qqq",
                "future_excess_return_10d_vs_sector",
                "managed_path_eligible",
                *MANAGED_BENCHMARK_RETURN_COLUMNS,
                *MANAGED_EXCESS_RETURN_COLUMNS,
                *MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
                *MANAGED_PATH_NET_RETURN_COLUMNS,
                "swing_feature_panel_schema",
                "strategy_contract_sha256",
                *feature_columns,
            )
        )
    )
    if profile == "catalyst_full":
        required = tuple(dict.fromkeys((*required, "source_count_alpaca_3d")))
    raw_files = _mapping(binding.manifest.get("files_by_profile"), "files_by_profile").get(profile)
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError(f"swing panel has no files for {profile}")
    selected_records = _partition_records_for_sessions(raw_files, sessions)
    if not selected_records:
        raise DataReadinessError(
            f"swing profile {profile} has no partitions for governed sessions"
        )
    projected_rows = sum(int(record.get("rows", -1)) for record in selected_records)
    projected_bytes = _projected_profile_memory_bytes(
        projected_rows, len(feature_columns)
    )
    safety_bytes = int(
        (config.maximum_process_memory_gib - config.memory_guard_headroom_gib)
        * 1024**3
    )
    if projected_rows < 1 or projected_bytes > safety_bytes:
        raise DataReadinessError(
            "projected swing profile memory exceeds the configured safety threshold"
        )
    paths: list[Path] = []
    for index, record in enumerate(selected_records):
        path = _resolve_inside(binding.root / "final", record.get("path"))
        schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
        missing = sorted(set(required).difference(schema.names))
        if missing:
            raise DataReadinessError(f"swing profile partition is missing columns: {missing}")
        paths.append(path)
        _guard(config, f"swing {profile} partition {index}", peak=False)
    dataset = pds.dataset(paths, format="parquet")  # type: ignore[no-untyped-call]
    session_type = dataset.schema.field("session_date_et").type
    filter_sessions: Sequence[object]
    if pa.types.is_date(session_type):
        filter_sessions = requested_dates
    elif pa.types.is_timestamp(session_type):
        filter_sessions = tuple(pd.Timestamp(value) for value in requested_dates)
    else:
        filter_sessions = sessions
    row_filter = (
        (pds.field("feature_eligible") == True)  # type: ignore[attr-defined,no-untyped-call]  # noqa: E712
        & (pds.field("label_eligible") == True)  # type: ignore[attr-defined,no-untyped-call]  # noqa: E712
        & pds.field("session_date_et").isin(filter_sessions)  # type: ignore[attr-defined,no-untyped-call]
    )
    if profile != "catalyst_full":
        row_filter = row_filter & (pds.field("cross_section_eligible") == True) & pds.field("rank_label").is_valid()  # type: ignore[attr-defined,no-untyped-call]  # noqa: E712

    table = dataset.to_table(
        columns=list(required),
        filter=row_filter,
        use_threads=False,
    )
    if table.num_rows < 1:
        raise DataReadinessError(f"swing profile {profile} has no eligible rows")
    frame = table.to_pandas(split_blocks=True, self_destruct=True)
    del table, dataset, paths
    frame = frame.copy()

    if profile == "catalyst_full":
        import pyarrow.dataset as ds
        sec_dir = r"c:\project\market-predictor\data\canonical\sec_filing_authority_20190709_20260708_v1\partitions"
        sec_dataset = ds.dataset(sec_dir, format="parquet")  # type: ignore[no-untyped-call]
        sec_df = sec_dataset.to_table(
            columns=["decision_id", "sec_filing_count_3d"]
        ).to_pandas()
        frame = frame.drop(columns=["source_count_sec_3d"], errors="ignore")
        frame = frame.merge(sec_df, on="decision_id", how="left")
        frame["source_count_sec_3d"] = frame["sec_filing_count_3d"].fillna(0)
        frame = frame.drop(columns=["sec_filing_count_3d"])

        # Event-Driven Specialist Logic
        has_catalyst = (frame["source_count_sec_3d"].fillna(0) > 0) | (frame["source_count_alpaca_3d"].fillna(0) > 0)
        frame = frame.loc[has_catalyst].copy()

        # Redefine decision_group_id to be session_date_et (global rank across all sectors for that day)
        frame["decision_group_id"] = frame["session_date_et"].astype("string[pyarrow]")

        minimum_group = strategy_contract.labels.minimum_cross_section_for_ranking
        target_group = strategy_contract.labels.swing_target_cross_section_for_ranking

        frame["sector_peer_count"] = frame.groupby("session_date_et")["security_id"].transform("count")
        frame = frame.loc[frame["sector_peer_count"] >= minimum_group].copy()

        # Redefine rank_label globally per session (top 25% of catalysts get 1.0)
        def _global_rank_label(g: pd.Series) -> pd.Series:
            return (g.rank(pct=True) >= 0.75).astype(float)
        frame["rank_label"] = frame.groupby("session_date_et")["future_excess_return_10d_vs_sector"].transform(_global_rank_label)
        frame["ranking_group_size"] = frame["sector_peer_count"]
        frame["ranking_reliability_weight"] = np.minimum(frame["sector_peer_count"] / float(target_group), 1.0)

        frame["cross_section_eligible"] = True
        frame["sector_rank_eligible"] = True
        frame["sector_rank_target_met"] = frame["sector_peer_count"] >= target_group

    frame = _validate_profile_frame(
        frame,
        profile=profile,
        feature_columns=feature_columns,
        strategy_contract=strategy_contract,
        config=config,
    )
    observed_sessions = set(frame["session_date_et"].astype(str))
    _validate_profile_session_coverage(observed_sessions, sessions)
    release_process_memory()
    _guard(config, f"swing {profile} load", peak=True)
    decision_hash = _sequence_sha256(frame["decision_id"].astype(str))
    return SwingProfileData(
        frame=frame,
        profile=profile,
        feature_columns=feature_columns,
        decision_ids_sha256=decision_hash,
        panel=binding,
    )


def _validate_profile_session_coverage(
    observed_sessions: set[str],
    governed_sessions: tuple[str, ...],
) -> None:
    if not observed_sessions or not governed_sessions:
        raise DataReadinessError("swing profile has no governed session coverage")
    expected = list(governed_sessions)
    expected_set = set(expected)
    extra = sorted(observed_sessions.difference(expected_set))
    if extra:
        raise DataReadinessError(
            f"swing profile contains sessions outside governance: {extra[:10]}"
        )
    first_observed_index = next(
        (index for index, value in enumerate(expected) if value in observed_sessions),
        len(expected),
    )
    allowed_prefix = set(expected[:first_observed_index])
    missing = expected_set.difference(observed_sessions)
    if missing != allowed_prefix:
        raise DataReadinessError(
            "swing profile has missing governed sessions outside its initial "
            f"catalyst warm-up: {sorted(missing.difference(allowed_prefix))[:10]}"
        )
    warmup_end = date.fromisoformat(expected[0]) + timedelta(days=3)
    if any(date.fromisoformat(value) >= warmup_end for value in allowed_prefix):
        raise DataReadinessError(
            "swing profile initial eligibility gap exceeds the 3-day catalyst lookback"
        )


def _partition_records_for_sessions(
    raw_files: Sequence[object],
    sessions: tuple[str, ...],
) -> list[dict[str, Any]]:
    requested = {date.fromisoformat(value) for value in sessions}
    records: list[dict[str, Any]] = []
    for raw in raw_files:
        record = _mapping(raw, "profile partition")
        first = date.fromisoformat(str(record.get("first_session")))
        last = date.fromisoformat(str(record.get("last_session")))
        month = str(record.get("partition_month", ""))
        if (
            first > last
            or first.strftime("%Y-%m") != month
            or last.strftime("%Y-%m") != month
        ):
            raise DataReadinessError("swing profile partition bounds are invalid")
        if any(first <= value <= last for value in requested):
            records.append(record)
    return records


def _projected_profile_memory_bytes(rows: int, feature_count: int) -> int:
    if rows < 0 or feature_count < 1:
        raise ValueError("profile memory projection inputs are invalid")
    # Includes compact float32 features, required labels/path columns, Arrow
    # strings, one bounded split projection, and estimator workspace. Profiles
    # are processed sequentially; the two ablation populations are never resident
    # together.
    return rows * (feature_count * 4 + 720) * 3


def _validate_profile_frame(
    frame: pd.DataFrame,
    *,
    profile: str,
    feature_columns: tuple[str, ...],
    strategy_contract: StrategyContract,
    config: SwingTrainingConfig,
) -> pd.DataFrame:
    data = frame
    if profile == "catalyst_full":
        if len(data) < 1000 or data["security_id"].nunique() < 50:
            raise DataReadinessError(f"eligible {profile} population is below training minimums")
    else:
        if len(data) < config.minimum_rows or data["security_id"].nunique() < config.minimum_securities:
            raise DataReadinessError(f"eligible {profile} population is below training minimums")
    session = pd.to_datetime(data["session_date_et"], errors="coerce")
    if session.isna().any() or bool(session.dt.date.lt(DECISION_START_DATE).any()):
        raise DataReadinessError("eligible swing decisions precede 2019-07-09")
    data["session_date_et"] = session.dt.date.astype(str)
    if data["decision_id"].isna().any() or data["decision_id"].duplicated().any():
        raise DataReadinessError("decision_id must be complete and unique within a profile")
    if data.duplicated(["decision_group_id", "security_id"]).any():
        raise DataReadinessError("a security appears more than once in a swing decision group")
    if set(data["swing_feature_panel_schema"].astype(str)) != {SWING_FEATURE_PANEL_SCHEMA}:
        raise DataReadinessError("swing feature schema differs from the current edge rebuild")
    if set(data["strategy_contract_sha256"].astype(str)) != {strategy_contract.sha256()}:
        raise DataReadinessError("row-level strategy contract hash differs from training")
    if set(pd.to_numeric(data["horizon_sessions"], errors="coerce").dropna()) != {10}:
        raise DataReadinessError("swing labels are not exact ten-session labels")
    for column in (
        "feature_eligible",
        "label_eligible",
        "cross_section_eligible",
        "managed_path_eligible",
    ):
        if not data[column].map(_strict_bool).all():
            raise DataReadinessError(f"training rows must all satisfy {column}")
    timestamp_columns = (
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "membership_effective_from_utc",
        "membership_available_at_utc",
        "entry_time_utc",
        "barrier_label_available_at_utc",
    )
    for column in timestamp_columns:
        parsed = pd.to_datetime(data[column], utc=True, errors="coerce")
        if parsed.isna().any():
            raise DataReadinessError(f"{column} must contain valid UTC timestamps")
        data[column] = parsed
    membership_end = pd.to_datetime(
        data["membership_effective_to_utc"], utc=True, errors="coerce"
    )
    decision = data["decision_time_utc"]
    if data["feature_available_at_utc"].gt(decision).any():
        raise DataReadinessError("feature availability occurs after the decision")
    if data["membership_available_at_utc"].gt(decision).any():
        raise DataReadinessError("point-in-time membership was unavailable at decision")
    if data["membership_effective_from_utc"].gt(decision).any():
        raise DataReadinessError("membership was not yet effective at decision")
    if (membership_end.notna() & membership_end.le(decision)).any():
        raise DataReadinessError("expired membership entered the training population")
    if data["label_available_at_utc"].le(decision).any():
        raise DataReadinessError("label availability must follow the decision")
    if data["entry_time_utc"].le(decision).any():
        raise DataReadinessError("managed entry must occur strictly after the decision")
    if data["barrier_label_available_at_utc"].lt(data["entry_time_utc"]).any():
        raise DataReadinessError("managed outcome is available before entry")
    if data["barrier_label_available_at_utc"].gt(data["label_available_at_utc"]).any():
        raise DataReadinessError("published label availability precedes managed outcome")
    holding = pd.to_numeric(data["barrier_holding_sessions"], errors="coerce")
    if holding.isna().any() or holding.lt(1).any() or holding.gt(10).any():
        raise DataReadinessError("managed holding period must be in [1, 10] sessions")
    barrier_label = pd.to_numeric(data["barrier_label"], errors="coerce")
    rank_label = pd.to_numeric(data["rank_label"], errors="coerce")
    if barrier_label.isna().any() or not barrier_label.isin([-1, 0, 1]).all():
        raise DataReadinessError("managed barrier labels are invalid")
    if rank_label.isna().any() or not rank_label.isin([-1, 0, 1]).all():
        raise DataReadinessError("managed cross-sectional rank labels are invalid")
    sector_peer_count = pd.to_numeric(data["sector_peer_count"], errors="coerce")
    ranking_group_size = pd.to_numeric(data["ranking_group_size"], errors="coerce")
    reliability = pd.to_numeric(
        data["ranking_reliability_weight"],
        errors="coerce",
    )
    minimum_group = strategy_contract.labels.minimum_cross_section_for_ranking
    target_group = strategy_contract.labels.swing_target_cross_section_for_ranking
    if (
        sector_peer_count.isna().any()
        or sector_peer_count.lt(minimum_group).any()
        or ranking_group_size.isna().any()
        or ranking_group_size.lt(minimum_group).any()
    ):
        raise DataReadinessError("swing ranking peer counts are below the hard floor")
    if not data["sector_rank_eligible"].map(_strict_bool).all():
        raise DataReadinessError("eligible swing rows must pass their sector peer floor")
    target_met = data["sector_rank_target_met"].map(_strict_bool)
    if not target_met.equals(sector_peer_count.ge(target_group)):
        raise DataReadinessError("swing sector ranking target status is inconsistent")
    expected_reliability = np.minimum(
        sector_peer_count.to_numpy(dtype="float64") / float(target_group),
        1.0,
    )
    if (
        not np.isfinite(reliability.to_numpy(dtype="float64")).all()
        or not np.allclose(
            reliability.to_numpy(dtype="float64"),
            expected_reliability,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise DataReadinessError("swing ranking reliability weight is inconsistent")
    data["sector_peer_count"] = sector_peer_count.astype("int32")
    data["ranking_group_size"] = ranking_group_size.astype("int32")
    data["ranking_reliability_weight"] = reliability.astype("float32")
    relevance = pd.to_numeric(data["future_excess_return_10d_vs_sector"], errors="coerce")
    if relevance.isna().any():
        raise DataReadinessError("relevance score contains missing values")
    data = pd.concat(
        [
            data.drop(columns=["target", "relevance_score"], errors="ignore"),
            rank_label.eq(1).astype("int8").rename("target"),
            relevance.astype("float32").rename("relevance_score"),
        ],
        axis=1,
    )
    if data["target"].nunique() != 2:
        raise DataReadinessError("managed rank target must contain both classes")
    for column in feature_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        array = values.to_numpy(dtype="float64", na_value=np.nan)
        if np.isinf(array).any():
            raise DataReadinessError(f"swing model feature {column} contains infinity")
        if not np.isfinite(array).any():
            raise DataReadinessError(f"swing model feature {column} is entirely missing")
        data[column] = values.astype("float32")
    numeric = (
        "barrier_gross_return",
        "barrier_cost",
        "barrier_net_return",
        "future_gross_return_10d",
        "future_net_return_10d",
        "future_spy_return_10d",
        "future_qqq_return_10d",
        "future_sector_return_10d",
        "future_excess_return_10d_vs_spy",
        "future_excess_return_10d_vs_qqq",
        "future_excess_return_10d_vs_sector",
        *MANAGED_BENCHMARK_RETURN_COLUMNS,
        *MANAGED_EXCESS_RETURN_COLUMNS,
        *MANAGED_PATH_NET_RETURN_COLUMNS,
    )
    for column in dict.fromkeys(numeric):
        values = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DataReadinessError(f"swing economic column {column} is not finite")
        data[column] = values.astype("float64")
    for column in MANAGED_PATH_SESSION_ORDINAL_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or values.lt(1).any():
            raise DataReadinessError(f"managed path session column {column} is invalid")
        data[column] = values.astype("int32")
    cost = data["barrier_cost"]
    expected_cost = config.expected_round_trip_cost_bps / 10_000.0
    if not np.allclose(cost, expected_cost, rtol=0.0, atol=1e-12):
        raise DataReadinessError("managed swing cost differs from the frozen policy")
    if not np.allclose(
        data["barrier_net_return"], data["barrier_gross_return"] - cost,
        rtol=0.0, atol=1e-10,
    ):
        raise DataReadinessError("managed net return does not apply cost exactly once")
    if not np.allclose(
        data["future_net_return_10d"], data["future_gross_return_10d"] - cost,
        rtol=0.0, atol=1e-10,
    ):
        raise DataReadinessError("ten-session net return does not apply cost exactly once")
    for benchmark in ("spy", "qqq", "sector"):
        if not np.allclose(
            data[f"future_excess_return_10d_vs_{benchmark}"],
            data["future_net_return_10d"] - data[f"future_{benchmark}_return_10d"],
            rtol=0.0,
            atol=1e-10,
        ):
            raise DataReadinessError(
                f"ten-session {benchmark.upper()} excess return arithmetic is invalid"
            )
        if not np.allclose(
            data[f"approx_managed_exit_session_close_excess_vs_{benchmark}"],
            data["barrier_net_return"]
            - data[f"approx_managed_exit_session_close_{benchmark}_return"],
            rtol=0.0,
            atol=1e-10,
        ):
            raise DataReadinessError(
                f"approximate managed-exit-session-close {benchmark.upper()} excess arithmetic is invalid"
            )
    path = data.loc[:, list(MANAGED_PATH_NET_RETURN_COLUMNS)].to_numpy(
        dtype="float64", copy=False
    )
    if not np.allclose(path[:, -1], data["barrier_net_return"], rtol=0.0, atol=1e-10):
        raise DataReadinessError("managed mark path does not reconcile to barrier net return")
    for column in _TEXT_COLUMNS:
        data[column] = data[column].astype("string[pyarrow]")
    return data.sort_values(
        ["decision_time_utc", "decision_group_id", "security_id"], kind="stable"
    ).reset_index(drop=True)


def train_swing_edge_candidate(
    panel_authority_directory: Path,
    output_directory: Path,
    *,
    strategy_contract: StrategyContract,
    config: SwingTrainingConfig,
    temporal_policy_path: Path,
) -> SwingTrainingResult:
    """Select one candidate on validation and touch the locked final test once."""

    _guard(config, "swing training start", peak=False)
    if output_directory.exists():
        raise FileExistsError(f"immutable output already exists: {output_directory}")
    if strategy_contract.swing.horizon_sessions != HORIZON_SESSIONS:
        raise DataReadinessError("trainer accepts only the active ten-session strategy")
    if strategy_contract.swing.round_trip_cost_bps != config.expected_round_trip_cost_bps:
        raise DataReadinessError("strategy and training cost contracts differ")
    if config.maximum_trades_per_decision != strategy_contract.swing.maximum_trades_per_decision:
        raise DataReadinessError("training trade cap differs from the authoritative strategy contract")
    temporal_config = load_temporal_manifest_config(temporal_policy_path)
    if (
        strategy_contract.validation.swing_walk_forward_folds != 1
        or temporal_config.validation_embargo_expected_sessions
        != strategy_contract.validation.embargo_sessions
        or temporal_config.final_embargo_expected_sessions
        != strategy_contract.validation.embargo_sessions
        or temporal_config.label_horizon_sessions != HORIZON_SESSIONS
        or temporal_config.unseen_security_holdout_fraction
        != strategy_contract.validation.unseen_ticker_holdout_fraction
        or temporal_config.modeled_decision_start.isoformat()
        != config.decision_start_date
    ):
        raise DataReadinessError(
            "temporal manifest differs from the authoritative strategy contract"
        )
    temporal_policy_sha256 = file_sha256(temporal_policy_path)
    schedule = build_temporal_schedule(temporal_config)
    folds = _governed_folds(schedule)
    model_sessions = _governed_model_sessions(schedule)
    final_refit_sessions = tuple(
        value.isoformat() for value in schedule.final_refit_sessions
    )
    test_sessions = tuple(
        value.isoformat() for value in schedule.locked_test_sessions
    )
    final_access_sessions = tuple(sorted({*final_refit_sessions, *test_sessions}))
    binding = load_swing_panel_binding(
        panel_authority_directory,
        strategy_contract=strategy_contract,
        config=config,
    )
    config_record = asdict(config)
    config_sha256 = _json_sha256(config_record)
    specs = _candidate_specs(config)
    if len(specs) > config.maximum_learned_candidates:
        raise DataReadinessError("candidate count exceeds the frozen sequential budget")

    validation_records: list[dict[str, Any]] = []
    profile_identity: str | None = None
    split_record = _split_record(
        folds=folds,
        schedule=schedule,
        temporal_config=temporal_config,
        temporal_policy_sha256=temporal_policy_sha256,
        strategy_contract=strategy_contract,
    )
    for profile in ALLOWED_PROFILES:
        profile_data = load_swing_profile(
            binding,
            profile,
            strategy_contract=strategy_contract,
            config=config,
            sessions=model_sessions,
        )
        if profile_identity is None:
            profile_identity = profile_data.decision_ids_sha256
        elif profile_identity != profile_data.decision_ids_sha256 and profile != "catalyst_full":
            raise DataReadinessError("technical and catalyst ablations do not contain identical decisions")
        for spec in (item for item in specs if item.profile == profile):
            validation_records.append(
                _evaluate_validation_candidate(
                    spec,
                    profile_data,
                    folds,
                    config,
                    strategy_contract,
                )
            )
            release_process_memory()
            _guard(config, f"{spec.candidate_id} validation", peak=True)
        del profile_data
        release_process_memory()

    if profile_identity is None:
        raise DataReadinessError("no swing ablation profiles were evaluated")
    paired_ablation = _paired_ablation_records(validation_records, config)
    eligible_candidates = [
        record for record in validation_records if record.get("candidate_eligible") is True
    ]
    if not eligible_candidates:
        no_candidate_evaluation = {
            "schema": EVALUATION_SCHEMA,
            "status": "no_candidate",
            "promotion_permitted": False,
            "selection_basis": "validation_only",
            "test_access_count": 0,
            "locked_test_outcomes_read": False,
            "dataset": _binding_record(binding, profile_identity),
            "training_config": config_record,
            "training_config_sha256": config_sha256,
            "temporal_manifest_policy_sha256": temporal_policy_sha256,
            "split": split_record,
            "validation_candidates": validation_records,
            "paired_ablation": paired_ablation,
            "reason": "no candidate passed the frozen validation economic gates",
        }
        no_candidate_model_card = {
            "schema": MODEL_CARD_SCHEMA,
            "model_schema": MODEL_SCHEMA,
            "status": "no_candidate",
            "promotion_permitted": False,
            "candidate_id": None,
            "dataset": _binding_record(binding, profile_identity),
            "strategy_contract_sha256": strategy_contract.sha256(),
            "training_config_sha256": config_sha256,
            "temporal_manifest_policy_sha256": temporal_policy_sha256,
        }
        _publish_immutable(
            output_directory,
            None,
            no_candidate_evaluation,
            no_candidate_model_card,
        )
        return SwingTrainingResult(
            output_directory=output_directory,
            selected_candidate_id=None,
            evaluation=no_candidate_evaluation,
            model_card=no_candidate_model_card,
        )
    family_groups: dict[str, list[dict[str, Any]]] = {}
    for record in eligible_candidates:
        spec = next(s for s in specs if s.candidate_id == record["candidate_id"])
        fam = "classifier" if spec.estimator_family in ("logistic", "xgboost_ranker") else spec.estimator_family
        family_groups.setdefault(fam, []).append(record)

    selected_records = {
        fam: max(recs, key=_selection_key)
        for fam, recs in family_groups.items()
    }
    selected_specs = {
        fam: next(s for s in specs if s.candidate_id == r["candidate_id"])
        for fam, r in selected_records.items()
    }
    selected_thresholds = {
        fam: float(r["selected_probability_threshold"])
        for fam, r in selected_records.items()
    }

    reference_spec = next(iter(selected_specs.values()))
    
    # The selected profile is reloaded only after validation has frozen both
    # candidate and threshold. This is the single controlled final-test access.
    selected_data = load_swing_profile(
        binding,
        reference_spec.profile,
        strategy_contract=strategy_contract,
        config=config,
        sessions=final_access_sessions,
    )
    holdout = _security_holdout_mask(selected_data.frame, strategy_contract)
    final_test_columns = list(dict.fromkeys((
        *_evaluation_columns(),
        *selected_data.feature_columns,
    )))
    development_columns = list(dict.fromkeys((
        "decision_id",
        "security_id",
        "session_date_et",
        "decision_time_utc",
        "label_available_at_utc",
        "target",
        "barrier_net_return",
        "ranking_reliability_weight",
        *selected_data.feature_columns,
    )))
    final_test = selected_data.frame.loc[
        selected_data.frame["session_date_et"].isin(test_sessions),
        final_test_columns,
    ].copy()
    development = selected_data.frame.loc[
        selected_data.frame["session_date_et"].isin(final_refit_sessions),
        development_columns,
    ].copy()
    _assert_label_purge(development, final_test, "final development/test")

    unseen_development = selected_data.frame.loc[
        selected_data.frame["session_date_et"].isin(final_refit_sessions)
        & ~holdout,
        development_columns,
    ].copy()
    unseen_final_test = selected_data.frame.loc[
        selected_data.frame["session_date_et"].isin(test_sessions)
        & holdout,
        final_test_columns,
    ].copy()
    _assert_label_purge(
        unseen_development,
        unseen_final_test,
        "unseen-security final development/test",
    )

    fitted_models = {}
    unseen_fitted_models = {}
    temporal_final_metrics = {}
    unseen_final_metrics = {}

    for fam, spec in selected_specs.items():
        threshold = selected_thresholds[fam]
        fitted = _fit_candidate(spec, development, selected_data, config)
        probability = _predict_probability(fitted, final_test, selected_data.feature_columns)
        temporal_final_metrics[fam] = _evaluation_metrics(
            final_test,
            probability,
            threshold=threshold,
            config=config,
            strategy_contract=strategy_contract,
            session_calendar=test_sessions,
        )
        
        unseen_fitted = _fit_candidate(
            spec,
            unseen_development,
            selected_data,
            config,
        )
        unseen_probability = _predict_probability(
            unseen_fitted,
            unseen_final_test,
            selected_data.feature_columns,
        )
        unseen_final_metrics[fam] = _evaluation_metrics(
            unseen_final_test,
            unseen_probability,
            threshold=threshold,
            config=config,
            strategy_contract=strategy_contract,
            session_calendar=test_sessions,
        )
        
        fitted_models[fam] = fitted
        unseen_fitted_models[fam] = unseen_fitted

    _guard(config, "swing final test", peak=True)

    evaluation: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "status": "candidate_only",
        "promotion_permitted": False,
        "selection_basis": "validation_only",
        "selection_policy": {
            "name": "SWING_CONSERVATIVE_ECONOMICS_V2",
            "auc_used_for_selection": False,
            "ordered_key": [
                "worst_holding_aligned_SPY_QQQ_sector_excess_calendar_ci_low",
                "portfolio_daily_return_bootstrap_ci_low",
                "worst_mean_holding_aligned_SPY_QQQ_sector_excess",
                "mean_managed_net_return",
                "negative_daily_mark_to_market_drawdown",
                "negative_turnover",
                "lower_probability_threshold_tie_break",
                "simpler_estimator_tie_break",
                "technical_only_tie_break",
            ],
        },
        "test_access_count": 1,
        "locked_test_outcomes_read": True,
        "locked_test_access_policy": (
            "outcome columns loaded once only after both validation scopes passed"
        ),
        "strategy": {
            "horizon_trading_sessions": HORIZON_SESSIONS,
            "entry_reference": strategy_contract.swing.entry_reference,
            "exit_rule": strategy_contract.swing.exit_rule,
            "round_trip_cost_bps": config.expected_round_trip_cost_bps,
            "target": "top_sector_relative_quantile_of_managed_barrier_net_return",
        },
        "dataset": _binding_record(binding, profile_identity),
        "training_config": config_record,
        "training_config_sha256": config_sha256,
        "temporal_manifest_policy_sha256": temporal_policy_sha256,
        "split": split_record,
        "overlap_audit": _overlap_audit(
            selected_data.frame,
            strategy_contract=strategy_contract,
            final_refit_sessions=final_refit_sessions,
            final_test_sessions=test_sessions,
            final_embargo_sessions=tuple(
                value.isoformat() for value in schedule.final_embargo_sessions
            ),
        ),
        "validation_candidates": validation_records,
        "paired_ablation": paired_ablation,
        "selected_candidate_ids": {fam: r["candidate_id"] for fam, r in selected_records.items()},
        "selected_profile": reference_spec.profile,
        "selected_probability_thresholds": selected_thresholds,
        "selected_validation_keys": {fam: list(_selection_key(r)) for fam, r in selected_records.items()},
        "final_test": {
            "temporal_generalization_full_pit_cross_section": temporal_final_metrics,
            "unseen_security_generalization_stable_20pct": unseen_final_metrics,
        },
        "benchmark_evaluation_basis": (
            "selection uses managed stock net return plus exact fixed-ten-session "
            "SPY, QQQ, and sector excess; managed-exit-session-close benchmark "
            "comparisons are approximate diagnostics only"
        ),
        "managed_path_cost_policy": MANAGED_PATH_COST_POLICY,
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
    }
    model_card: dict[str, Any] = {
        "schema": MODEL_CARD_SCHEMA,
        "model_schema": MODEL_SCHEMA,
        "status": "candidate",
        "promotion_permitted": False,
        "candidate_id": selected_records.get("classifier", next(iter(selected_records.values())))["candidate_id"],
        "candidate_ids": {fam: r["candidate_id"] for fam, r in selected_records.items()},
        "ablation_profile": reference_spec.profile,
        "models": {
            fam: {
                "estimator_family": selected_specs[fam].estimator_family,
                "hyperparameters": dict(selected_specs[fam].hyperparameters),
                "probability_threshold": selected_thresholds[fam],
            } for fam in selected_specs
        },
        "feature_columns": list(selected_data.feature_columns),
        "feature_set_sha256": _sequence_sha256(selected_data.feature_columns),
        "training_rows": len(development),
        "training_sessions": int(development["session_date_et"].nunique()),
        "training_securities": int(development["security_id"].nunique()),
        "locked_test_rows": len(final_test),
        "locked_test_unseen_security_rows": len(unseen_final_test),
        "dataset": _binding_record(binding, profile_identity),
        "strategy_contract_sha256": strategy_contract.sha256(),
        "training_config_sha256": config_sha256,
        "temporal_manifest_policy_sha256": temporal_policy_sha256,
        "calibration_method": "platt_sigmoid_on_prior_purged_sessions",
        "final_test_opened_once": True,
        "limitations": [
            "Candidate is not promoted and must not be used for live trading.",
            "Drawdown and turnover use the panel's managed daily mark-to-market paths and overlapping cohorts.",
            "Managed-exit benchmark comparisons use exit-session closes and are approximate diagnostics only.",
            "SEC, global, and Finviz inputs are excluded pending independent causal ablation.",
        ],
    }
    payload: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "status": "candidate",
        "promotion_permitted": False,
        "candidate_id": selected_records.get("classifier", next(iter(selected_records.values())))["candidate_id"],
        "ablation_profile": reference_spec.profile,
        "feature_columns": list(selected_data.feature_columns),
        "feature_set_sha256": _sequence_sha256(selected_data.feature_columns),
        "dataset": _binding_record(binding, profile_identity),
        "training_config_sha256": config_sha256,
        "temporal_manifest_policy_sha256": temporal_policy_sha256,
        "strategy_contract_sha256": strategy_contract.sha256(),
        "fitted_models": fitted_models,
        "probability_thresholds": selected_thresholds,
    }
    _publish_immutable(
        output_directory,
        payload,
        evaluation,
        model_card,
    )
    return SwingTrainingResult(
        output_directory=output_directory,
        selected_candidate_id="multi_model_bundle",
        evaluation=evaluation,
        model_card=model_card,
    )


def _candidate_specs(config: SwingTrainingConfig) -> tuple[CandidateSpec, ...]:
    specs: list[CandidateSpec] = []
    for profile in ALLOWED_PROFILES:
        family_name = "technical_only" if profile == SWING_FEATURE_PROFILE else "technical_plus_alpaca"
        for value in config.logistic_c_values:
            specs.append(
                CandidateSpec(
                    candidate_id=f"{family_name}.logistic.c_{value:g}",
                    profile=profile,
                    estimator_family="logistic",
                    hyperparameters={"C": value, "solver": "lbfgs", "threads": 1},
                )
            )
        for rate in config.hgb_learning_rates:
            for depth in config.xgb_max_depths:
                for label, family in _XGB_GRID:
                    specs.append(
                        CandidateSpec(
                            candidate_id=f"{family_name}.{label}.lr_{rate:g}.depth_{depth}",
                            profile=profile,
                            estimator_family=family,
                            hyperparameters={
                                "learning_rate": rate,
                                "max_depth": depth,
                                "n_estimators": config.hgb_max_iter,
                                "threads": 1,
                            },
                        )
                    )
    return tuple(specs)


def _ordered_sessions(data: pd.DataFrame) -> tuple[str, ...]:
    sessions = (
        data.groupby("session_date_et", as_index=False, observed=True)["decision_time_utc"]
        .min()
        .sort_values(["decision_time_utc", "session_date_et"], kind="stable")
    )
    order = tuple(sessions["session_date_et"].astype(str))
    if len(order) != len(set(order)):
        raise DataReadinessError("exchange sessions are not unique")
    return order


def _security_holdout_mask(
    data: pd.DataFrame,
    strategy_contract: StrategyContract,
) -> pd.Series:
    fraction = strategy_contract.validation.unseen_ticker_holdout_fraction
    threshold = int(fraction * 2**64)
    identities = data["security_id"].astype(str)
    assigned = identities.map(
        lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
        < threshold
    )
    if not assigned.any() or assigned.all():
        raise DataReadinessError("stable security holdout produced an empty partition")
    return assigned.astype(bool)


def _evaluation_columns() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "decision_id",
                "decision_group_id",
                "ticker",
                "security_id",
                "sector",
                "market_regime",
                "session_date_et",
                "decision_time_utc",
                "barrier_exit_session_date_et",
                "barrier_holding_sessions",
                "target",
                "barrier_gross_return",
                "barrier_cost",
                "barrier_net_return",
                "future_excess_return_10d_vs_spy",
                "future_excess_return_10d_vs_qqq",
                "future_excess_return_10d_vs_sector",
                *MANAGED_EXCESS_RETURN_COLUMNS,
                *MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
                *MANAGED_PATH_NET_RETURN_COLUMNS,
            )
        )
    )


def _evaluate_validation_candidate(
    spec: CandidateSpec,
    profile_data: SwingProfileData,
    folds: tuple[WalkForwardFold, ...],
    config: SwingTrainingConfig,
    strategy_contract: StrategyContract,
) -> dict[str, Any]:
    predictions: dict[str, list[pd.DataFrame]] = {
        "temporal_generalization_full_pit_cross_section": [],
        "unseen_security_generalization_stable_20pct": [],
    }
    fold_records: list[dict[str, Any]] = []
    holdout = _security_holdout_mask(profile_data.frame, strategy_contract)
    for fold in folds:
        train_columns = list(dict.fromkeys((
            "decision_id",
            "session_date_et",
            "decision_time_utc",
            "decision_group_id",
            "label_available_at_utc",
            "target",
            "barrier_net_return",
            "relevance_score",
            "ranking_reliability_weight",
            *profile_data.feature_columns,
        )))
        validation_columns = list(dict.fromkeys((
            *_evaluation_columns(),
            *profile_data.feature_columns,
        )))
        scope_records: dict[str, Any] = {}
        for scope, train_mask, validation_mask in (
            (
                "temporal_generalization_full_pit_cross_section",
                profile_data.frame["session_date_et"].isin(fold.train_sessions),
                profile_data.frame["session_date_et"].isin(
                    fold.validation_sessions
                ),
            ),
            (
                "unseen_security_generalization_stable_20pct",
                profile_data.frame["session_date_et"].isin(fold.train_sessions)
                & ~holdout,
                profile_data.frame["session_date_et"].isin(
                    fold.validation_sessions
                )
                & holdout,
            ),
        ):
            train = profile_data.frame.loc[train_mask, train_columns]
            validation = profile_data.frame.loc[validation_mask, validation_columns]
            _assert_label_purge(
                train,
                validation,
                f"{scope} validation fold {fold.fold}",
            )
            fitted = _fit_candidate(spec, train, profile_data, config)
            probability = _predict_probability(
                fitted,
                validation,
                profile_data.feature_columns,
            )
            scored = validation.loc[:, list(_evaluation_columns())].copy()
            scored["__probability"] = probability
            predictions[scope].append(scored)
            scope_records[scope] = {
                "train_rows": len(train),
                "validation_rows": len(validation),
                "max_train_label_available_at_utc": _iso(
                    train["label_available_at_utc"].max()
                ),
                "min_validation_decision_time_utc": _iso(
                    validation["decision_time_utc"].min()
                ),
                "fit_sessions": fitted.fit_sessions,
                "calibration_sessions": fitted.calibration_sessions,
                "calibration_cutoff_utc": fitted.calibration_cutoff_utc,
                "target_prevalence": float(validation["target"].mean()),
                "probability_distribution": _probability_distribution(probability),
            }
            del fitted, train, validation, scored
            release_process_memory()
        fold_records.append(
            {
                "fold": fold.fold,
                "train_sessions": len(fold.train_sessions),
                "purge_sessions": len(fold.purge_sessions),
                "embargo_sessions": len(fold.embargo_sessions),
                "validation_sessions": len(fold.validation_sessions),
                "scopes": scope_records,
            }
        )
    pooled = {
        scope: pd.concat(parts, ignore_index=True)
        for scope, parts in predictions.items()
    }
    del predictions
    validation_calendar = tuple(
        session
        for fold in folds
        for session in fold.validation_sessions
    )
    threshold_records: list[dict[str, Any]] = []
    for threshold in config.probability_thresholds:
        try:
            scope_metrics = {
                scope: _evaluation_metrics(
                    frame,
                    frame["__probability"].to_numpy(dtype="float64"),
                    threshold=threshold,
                    config=config,
                    strategy_contract=strategy_contract,
                    session_calendar=validation_calendar,
                )
                for scope, frame in pooled.items()
            }
            passed = all(
                bool(metrics["economic_gate"]["passed"])
                for metrics in scope_metrics.values()
            )
            threshold_records.append({
                "probability_threshold": threshold,
                "eligible": passed,
                "reason": (
                    None
                    if passed
                    else "one or more frozen validation scopes failed economic gates"
                ),
                "scopes": scope_metrics,
            })
        except DataReadinessError as exc:
            threshold_records.append(
                {
                    "probability_threshold": threshold,
                    "eligible": False,
                    "reason": str(exc),
                }
            )
    eligible = [record for record in threshold_records if record["eligible"]]
    diagnostic = [record for record in threshold_records if "scopes" in record]
    if not diagnostic:
        return {
            "candidate_id": spec.candidate_id,
            "ablation_profile": spec.profile,
            "estimator_family": spec.estimator_family,
            "hyperparameters": dict(spec.hyperparameters),
            "folds": fold_records,
            "thresholds": threshold_records,
            "candidate_eligible": False,
            "reason": "no threshold selected enough validation trades",
        }
    selected = max(eligible or diagnostic, key=_threshold_selection_key)
    metrics = _mapping(selected.get("scopes"), "selected threshold scopes")
    for threshold_record in threshold_records:
        if threshold_record is selected:
            continue
        raw_scopes = threshold_record.get("scopes")
        if isinstance(raw_scopes, dict):
            for raw_metrics in raw_scopes.values():
                if isinstance(raw_metrics, dict):
                    raw_metrics.pop("paired_session_blocks", None)
    record: dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "ablation_profile": spec.profile,
        "estimator_family": spec.estimator_family,
        "hyperparameters": dict(spec.hyperparameters),
        "folds": fold_records,
        "thresholds": threshold_records,
        "selected_probability_threshold": float(selected["probability_threshold"]),
        "selected_validation_metrics": metrics,
        "candidate_eligible": bool(eligible),
    }
    if eligible:
        record["selection_key"] = list(_selection_key(record))
    return record


def _probability_distribution(probability: np.ndarray) -> dict[str, float]:
    if probability.ndim != 1 or probability.size < 1 or not np.isfinite(probability).all():
        raise DataReadinessError("probability diagnostics require one finite vector")
    quantiles = np.quantile(probability, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "minimum": float(probability.min()),
        "p01": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(probability.max()),
        "mean": float(probability.mean()),
    }


def _fit_candidate(
    spec: CandidateSpec,
    train: pd.DataFrame,
    profile_data: SwingProfileData,
    config: SwingTrainingConfig,
) -> FittedCandidate:
    fit_sessions, calibration_sessions = _split_fit_calibration(train, config)
    fit_mask = train["session_date_et"].isin(fit_sessions)
    calibration_mask = train["session_date_et"].isin(calibration_sessions)
    if (
        train.loc[fit_mask, "target"].nunique() != 2
        or train.loc[calibration_mask, "target"].nunique() != 2
    ):
        raise DataReadinessError("fit and calibration partitions must contain both classes")
    columns = list(profile_data.feature_columns)

    if spec.estimator_family == "xgboost_ranker":
        train_fit = train.loc[fit_mask].sort_values("decision_group_id")
        x_fit = train_fit[columns].to_numpy(dtype="float32", copy=False)
        y_fit = train_fit["relevance_score"].to_numpy(dtype="float32", copy=False)
        qid_fit = pd.factorize(train_fit["decision_group_id"])[0]
        fit_weight = train_fit.drop_duplicates(
            subset=["decision_group_id"]
        )["ranking_reliability_weight"].to_numpy(dtype="float64", copy=False)
    elif spec.estimator_family == "xgboost_regressor":
        x_fit = train.loc[fit_mask, columns].to_numpy(dtype="float32", copy=False)
        y_fit = train.loc[fit_mask, "barrier_net_return"].to_numpy(dtype="float32", copy=False)
        qid_fit = None
        fit_weight = train.loc[
            fit_mask,
            "ranking_reliability_weight",
        ].to_numpy(dtype="float64", copy=False)
    else:
        x_fit = train.loc[fit_mask, columns].to_numpy(dtype="float32", copy=False)
        y_fit = train.loc[fit_mask, "target"].to_numpy(dtype="int8", copy=False)
        qid_fit = None
        fit_weight = train.loc[
            fit_mask,
            "ranking_reliability_weight",
        ].to_numpy(dtype="float64", copy=False)
    if spec.estimator_family == "logistic":
        estimator: Any = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0.0,
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(spec.hyperparameters["C"]),
                        max_iter=500,
                        random_state=config.random_seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif spec.estimator_family == "xgboost_ranker":
        import xgboost as xgb
        constraints = tuple(
            1 if col in ("dollar_volume_log", "volume_ratio_20", "volume_z20") else 0
            for col in columns
        )
        estimator = xgb.XGBRanker(
            objective="rank:pairwise",
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_depth=int(spec.hyperparameters["max_depth"]),
            n_estimators=int(spec.hyperparameters["n_estimators"]),
            random_state=config.random_seed,
            tree_method="hist",
            monotone_constraints=constraints,
        )
    elif spec.estimator_family == "xgboost_regressor":
        import xgboost as xgb
        estimator = xgb.XGBRegressor(
            objective=_linex_objective,
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_depth=int(spec.hyperparameters["max_depth"]),
            n_estimators=int(spec.hyperparameters["n_estimators"]),
            random_state=config.random_seed,
            tree_method="hist",
            reg_lambda=10.0,
        )
    else:
        raise DataReadinessError(f"unknown swing estimator family: {spec.estimator_family}")
    if spec.estimator_family == "logistic":
        estimator.fit(x_fit, y_fit, model__sample_weight=fit_weight)
    elif spec.estimator_family == "xgboost_ranker":
        estimator.fit(x_fit, y_fit, qid=qid_fit, sample_weight=fit_weight)
    else:
        estimator.fit(x_fit, y_fit, sample_weight=fit_weight)
    del x_fit, y_fit, fit_weight
    raw = _raw_probability(
        estimator,
        train.loc[calibration_mask, columns].to_numpy(dtype="float32", copy=False),
    )
    calibrator = LogisticRegression(
        C=1.0,
        max_iter=300,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    calibrator.fit(
        raw.reshape(-1, 1),
        train.loc[calibration_mask, "target"].to_numpy(dtype="int8", copy=False),
        sample_weight=train.loc[
            calibration_mask,
            "ranking_reliability_weight",
        ].to_numpy(dtype="float64", copy=False),
    )
    return FittedCandidate(
        estimator=estimator,
        calibrator=calibrator,
        fit_sessions=len(fit_sessions),
        calibration_sessions=len(calibration_sessions),
        calibration_cutoff_utc=_iso(
            train.loc[calibration_mask, "label_available_at_utc"].max()
        ),
    )


def _linex_objective(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    residual = y_pred - y_true
    
    # LinEx parameters
    # a > 0 heavily penalizes overestimation (predicted > actual) exponentially
    a = 15.0
    b = 1.0

    # Gradient: b * a * (exp(a * residual) - 1)
    # Hessian: b * a^2 * exp(a * residual)
    
    # Clip residual to prevent overflow in exp
    clipped_residual = np.clip(residual, -1.0, 1.0)
    exp_term = np.exp(a * clipped_residual)
    
    grad = b * a * (exp_term - 1.0)
    hess = b * (a ** 2) * exp_term

    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight
    return grad, hess


def _raw_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probability = np.asarray(estimator.predict_proba(features)[:, 1], dtype="float64")
    else:
        probability = np.asarray(estimator.predict(features), dtype="float64")
    if not np.isfinite(probability).all():
        raise DataReadinessError("estimator produced non-finite probabilities")
    return probability


def _predict_probability(
    fitted: FittedCandidate,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    raw = _raw_probability(
        fitted.estimator,
        frame.loc[:, list(feature_columns)].to_numpy(dtype="float32", copy=False),
    )
    calibrated = np.asarray(
        fitted.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1],
        dtype="float64",
    )
    if (
        not np.isfinite(calibrated).all()
        or (calibrated < 0.0).any()
        or (calibrated > 1.0).any()
    ):
        raise DataReadinessError("calibrated probabilities must be finite in [0, 1]")
    return calibrated


def _evaluation_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    threshold: float,
    config: SwingTrainingConfig,
    strategy_contract: StrategyContract,
    session_calendar: tuple[str, ...],
) -> dict[str, Any]:
    if len(frame) != len(probability) or not np.isfinite(probability).all():
        raise DataReadinessError("prediction length or finiteness is invalid")
    scored = frame.copy()
    scored["__probability"] = probability
    candidates = scored.loc[scored["__probability"].ge(threshold)].copy()
    if "source_count_sec_3d" in candidates.columns:
        # Event-Driven Specialist: bypass strict sector limits
        selected = candidates.groupby("session_date_et", group_keys=False).apply(
            lambda g: g.nlargest(config.maximum_trades_per_decision, "__probability")
        )
    else:
        selected = select_constrained_swing_portfolio(
            candidates,
            maximum_trades=config.maximum_trades_per_decision,
            target_maximum_sector_weight=strategy_contract.swing.target_maximum_sector_weight,
            hard_maximum_sector_weight=strategy_contract.swing.hard_maximum_sector_weight,
            minimum_distinct_sectors=strategy_contract.swing.minimum_distinct_sectors_for_selection,
        )
    if selected.empty or selected["session_date_et"].nunique() < 2:
        raise DataReadinessError("threshold selects fewer than two independent sessions")
    selected = selected.sort_values(
        ["decision_time_utc", "decision_group_id", "security_id"], kind="stable"
    )
    target = scored["target"].to_numpy(dtype="int8", copy=False)
    has_two_classes = np.unique(target).size == 2
    base_rate = float(target.mean())
    selected_rate = float(selected["target"].mean())
    ledger = _daily_position_ledger(
        selected,
        config,
        session_calendar=session_calendar,
    )
    stress_ledger = _daily_position_ledger(
        selected,
        config,
        session_calendar=session_calendar,
        additional_round_trip_cost=(
            (strategy_contract.stress.cost_multiplier - 1.0)
            * config.expected_round_trip_cost_bps
            / 10_000.0
        ),
    )
    positive = selected.loc[selected["barrier_net_return"].gt(0), "barrier_net_return"].sum()
    negative = selected.loc[selected["barrier_net_return"].lt(0), "barrier_net_return"].sum()
    calibration_bins = _calibration_bins(target, probability)
    bootstrap = _session_bootstrap(
        selected,
        config,
        session_calendar=session_calendar,
    )
    bootstrap["portfolio_daily_return"] = _moving_block_bootstrap_mean_interval(
        np.asarray(ledger["daily_returns"], dtype="float64"),
        config.bootstrap_samples,
        config.bootstrap_block_sessions,
        config.random_seed + 10_001,
    )
    bootstrap["double_cost_portfolio_daily_return"] = (
        _moving_block_bootstrap_mean_interval(
            np.asarray(stress_ledger["daily_returns"], dtype="float64"),
            config.bootstrap_samples,
            config.bootstrap_block_sessions,
            config.random_seed + 10_002,
        )
    )
    metrics: dict[str, Any] = {
        "rows": len(scored),
        "sessions": int(scored["session_date_et"].nunique()),
        "securities": int(scored["security_id"].nunique()),
        "probability_threshold": threshold,
        "roc_auc": float(roc_auc_score(target, probability)) if has_two_classes else None,
        "pr_auc": float(average_precision_score(target, probability)) if has_two_classes else None,
        "auc_is_diagnostic_only": True,
        "brier_score": float(brier_score_loss(target, probability)),
        "expected_calibration_error": _expected_calibration_error(calibration_bins, len(scored)),
        "calibration_bins": calibration_bins,
        "base_positive_rate": base_rate,
        "selected_positive_rate": selected_rate,
        "selected_probability_lift": selected_rate / base_rate if base_rate > 0 else None,
        "selected_trade_count": len(selected),
        "selected_decision_count": int(selected["decision_group_id"].nunique()),
        "selected_average_managed_gross_return": float(selected["barrier_gross_return"].mean()),
        "selected_average_managed_net_return": float(
            selected["barrier_net_return"].mean()
        ),
        "selected_win_rate_after_costs": float(selected["barrier_net_return"].gt(0).mean()),
        "calendar_average_managed_net_return": float(
            bootstrap["calendar_average_managed_net_return"]["estimate"]
        ),
        "calendar_average_managed_exit_session_close_spy_excess": float(
            bootstrap["calendar_average_managed_exit_session_close_spy_excess"]["estimate"]
        ),
        "calendar_average_managed_exit_session_close_qqq_excess": float(
            bootstrap["calendar_average_managed_exit_session_close_qqq_excess"]["estimate"]
        ),
        "calendar_average_managed_exit_session_close_sector_excess": float(
            bootstrap["calendar_average_managed_exit_session_close_sector_excess"]["estimate"]
        ),
        "selected_average_managed_exit_session_close_spy_excess": float(
            selected["approx_managed_exit_session_close_excess_vs_spy"].mean()
        ),
        "selected_average_managed_exit_session_close_qqq_excess": float(
            selected["approx_managed_exit_session_close_excess_vs_qqq"].mean()
        ),
        "selected_average_managed_exit_session_close_sector_excess": float(
            selected["approx_managed_exit_session_close_excess_vs_sector"].mean()
        ),
        "managed_exit_benchmark_timestamp_policy": "entry_open_to_exit_session_close",
        "profit_factor_after_costs": float(positive / abs(negative)) if negative < 0 else None,
        "turnover": ledger["average_daily_turnover"],
        "daily_mark_to_market_max_drawdown_after_costs": ledger["max_drawdown"],
        "daily_mark_to_market_compounded_return": ledger["compounded_return"],
        "portfolio_daily_average_return": float(
            bootstrap["portfolio_daily_return"]["estimate"]
        ),
        "double_cost_portfolio_daily_average_return": float(
            bootstrap["double_cost_portfolio_daily_return"]["estimate"]
        ),
        "drawdown_has_daily_mark_to_market": True,
        "maximum_observed_sector_weight": ledger["maximum_sector_weight"],
        "target_maximum_sector_weight": (
            strategy_contract.swing.target_maximum_sector_weight
        ),
        "hard_maximum_sector_weight": (
            strategy_contract.swing.hard_maximum_sector_weight
        ),
        "minimum_distinct_sectors_for_selection": (
            strategy_contract.swing.minimum_distinct_sectors_for_selection
        ),
        "maximum_effective_sector_weight_limit": float(
            selected[EFFECTIVE_SECTOR_WEIGHT_COLUMN].max()
        ),
        "frozen_round_trip_cost_bps": config.expected_round_trip_cost_bps,
        "cost_deduction_count": 1,
        "by_regime": _stability_breakdown(selected, "market_regime"),
        "by_sector": _stability_breakdown(selected, "sector"),
        "by_year": _year_breakdown(selected),
        "moving_block_bootstrap_95_ci": bootstrap,
        "paired_session_blocks": _session_economic_blocks(
            selected,
            session_calendar=session_calendar,
        ),
    }
    metrics["economic_gate"] = _economic_gate(metrics, strategy_contract)
    metrics["regime_stability"] = _stability_summary(metrics["by_regime"])
    metrics["sector_stability"] = _stability_summary(metrics["by_sector"])
    return metrics


def _threshold_selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    scopes = _mapping(record.get("scopes"), "threshold scopes")
    keys = [
        _scope_economic_key(_mapping(metrics, f"{scope} metrics"))
        for scope, metrics in sorted(scopes.items())
    ]
    if not keys:
        raise DataReadinessError("threshold has no validation scopes")
    return tuple(min(key[index] for key in keys) for index in range(len(keys[0]))) + (
        -float(record["probability_threshold"]),
    )


def _scope_economic_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    bootstrap = _mapping(metrics.get("moving_block_bootstrap_95_ci"), "bootstrap")
    portfolio_ci = _mapping(
        bootstrap.get("portfolio_daily_return"), "portfolio daily CI"
    )
    spy_ci = _mapping(
        bootstrap.get("calendar_average_managed_exit_session_close_spy_excess"),
        "managed SPY CI",
    )
    qqq_ci = _mapping(
        bootstrap.get("calendar_average_managed_exit_session_close_qqq_excess"),
        "managed QQQ CI",
    )
    sector_ci = _mapping(
        bootstrap.get("calendar_average_managed_exit_session_close_sector_excess"),
        "managed sector CI",
    )
    return (
        min(_finite(spy_ci, "low"), _finite(qqq_ci, "low"), _finite(sector_ci, "low")),
        _finite(portfolio_ci, "low"),
        min(
            _finite(metrics, "calendar_average_managed_exit_session_close_spy_excess"),
            _finite(metrics, "calendar_average_managed_exit_session_close_qqq_excess"),
            _finite(metrics, "calendar_average_managed_exit_session_close_sector_excess"),
        ),
        _finite(metrics, "selected_average_managed_net_return"),
        -_finite(metrics, "daily_mark_to_market_max_drawdown_after_costs"),
        -_finite(metrics, "turnover"),
    )


def _selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    scopes = _mapping(record.get("selected_validation_metrics"), "validation scopes")
    threshold_record = {
        "probability_threshold": record.get("selected_probability_threshold"),
        "scopes": scopes,
    }
    economic = _threshold_selection_key(threshold_record)
    # Prefer the simpler logistic candidate only after all economic and risk
    # criteria tie. AUC remains diagnostic and cannot drive candidate choice.
    simplicity = 1.0 if record.get("estimator_family") == "logistic" else 0.0
    profile_simplicity = 1.0 if record.get("ablation_profile") == SWING_FEATURE_PROFILE else 0.0
    return (*economic, simplicity, profile_simplicity)


def _paired_ablation_records(
    records: Sequence[Mapping[str, Any]],
    config: SwingTrainingConfig,
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        if "selected_validation_metrics" not in record:
            continue
        key = (
            str(record.get("estimator_family")),
            json.dumps(record.get("hyperparameters"), sort_keys=True),
        )
        indexed[(str(record.get("ablation_profile")), *key)] = record
    output: list[dict[str, Any]] = []
    families = {
        (key[1], key[2]) for key in indexed if key[0] == SWING_FEATURE_PROFILE
    }
    for family, hyperparameters in sorted(families):
        technical = indexed.get((SWING_FEATURE_PROFILE, family, hyperparameters))
        catalyst = indexed.get((SWING_CATALYST_FEATURE_PROFILE, family, hyperparameters))
        if technical is None or catalyst is None:
            continue
        technical_scopes = _mapping(
            technical.get("selected_validation_metrics"),
            "technical ablation scopes",
        )
        catalyst_scopes = _mapping(
            catalyst.get("selected_validation_metrics"),
            "catalyst ablation scopes",
        )
        for scope in sorted(set(technical_scopes).intersection(catalyst_scopes)):
            technical_metrics = _mapping(
                technical_scopes[scope],
                f"technical {scope} metrics",
            )
            catalyst_metrics = _mapping(
                catalyst_scopes[scope],
                f"catalyst {scope} metrics",
            )
            left = pd.DataFrame(technical_metrics.get("paired_session_blocks", []))
            right = pd.DataFrame(catalyst_metrics.get("paired_session_blocks", []))
            if left.empty or right.empty:
                continue
            paired = left.merge(
                right,
                on="session_date_et",
                how="outer",
                suffixes=("_technical", "_catalyst"),
                validate="one_to_one",
                indicator=True,
            ).sort_values("session_date_et", kind="stable")
            economic_columns = (
                "barrier_net_return",
                "approx_managed_exit_session_close_excess_vs_spy",
                "approx_managed_exit_session_close_excess_vs_qqq",
                "approx_managed_exit_session_close_excess_vs_sector",
            )
            for column in economic_columns:
                paired[f"{column}_technical"] = pd.to_numeric(
                    paired[f"{column}_technical"], errors="coerce"
                ).fillna(0.0)
                paired[f"{column}_catalyst"] = pd.to_numeric(
                    paired[f"{column}_catalyst"], errors="coerce"
                ).fillna(0.0)
            if len(paired) < config.bootstrap_block_sessions:
                output.append({
                    "estimator_family": family,
                    "hyperparameters": json.loads(hyperparameters),
                    "scope": scope,
                    "difference": "catalyst_full_minus_technical_market",
                    "paired_sessions": len(paired),
                    "eligible": False,
                    "reason": "paired ablation has fewer sessions than one bootstrap block",
                })
                continue
            intervals: dict[str, Any] = {}
            for column in economic_columns:
                differences = (
                    paired[f"{column}_catalyst"]
                    - paired[f"{column}_technical"]
                ).to_numpy(dtype="float64")
                seed = config.random_seed + int(
                    hashlib.sha256(
                        f"paired:{family}:{hyperparameters}:{scope}:{column}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                intervals[column] = _moving_block_bootstrap_mean_interval(
                    differences,
                    config.bootstrap_samples,
                    config.bootstrap_block_sessions,
                    seed,
                )
            output.append({
                "estimator_family": family,
                "hyperparameters": json.loads(hyperparameters),
                "scope": scope,
                "difference": "catalyst_full_minus_technical_market",
                "technical_threshold": technical.get("selected_probability_threshold"),
                "catalyst_threshold": catalyst.get("selected_probability_threshold"),
                "paired_sessions": len(paired),
                "calendar_join": "full_outer_zero_for_no_position",
                "technical_only_sessions": int(
                    paired["_merge"].eq("left_only").sum()
                ),
                "catalyst_only_sessions": int(
                    paired["_merge"].eq("right_only").sum()
                ),
                "eligible": True,
                "moving_block_bootstrap_95_ci": intervals,
            })
    return output


def _publish_immutable(
    output_directory: Path,
    candidate: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    model_card: Mapping[str, Any],
) -> None:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.", dir=output_directory.parent
        )
    )
    try:
        if candidate is not None:
            joblib.dump(dict(candidate), temporary / _CANDIDATE_NAME, compress=3)
        _write_json(temporary / _EVALUATION_NAME, evaluation)
        _write_json(temporary / _MODEL_CARD_NAME, model_card)
        artifact_names = [
            *([_CANDIDATE_NAME] if candidate is not None else []),
            _EVALUATION_NAME,
            _MODEL_CARD_NAME,
        ]
        artifacts = {
            name: {
                "sha256": file_sha256(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in artifact_names
        }
        state = "candidate" if candidate is not None else "no_candidate"
        manifest = {
            "schema": MODEL_SCHEMA,
            "state": state,
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": artifacts,
        }
        _write_json(temporary / _MANIFEST_NAME, manifest)
        authority = {
            "schema": OUTPUT_AUTHORITY_SCHEMA,
            "state": state,
            "promotion_permitted": False,
            "artifact": _MANIFEST_NAME,
            "artifact_sha256": file_sha256(temporary / _MANIFEST_NAME),
        }
        _write_json(temporary / _AUTHORITY_NAME, authority)
        try:
            temporary.rename(output_directory)
        except FileExistsError:
            raise FileExistsError(
                f"immutable output already exists: {output_directory}"
            ) from None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_swing_candidate_authority(directory: Path) -> dict[str, Any]:
    """Strictly replay a candidate-only output and every artifact hash."""

    root = directory.resolve()
    manifest_path = root / _MANIFEST_NAME
    authority_path = root / _AUTHORITY_NAME
    manifest = _read_json(manifest_path, "swing candidate manifest")
    authority = _read_json(authority_path, "swing candidate authority")
    state = str(manifest.get("state"))
    if (
        manifest.get("schema") != MODEL_SCHEMA
        or state not in {"candidate", "no_candidate"}
        or manifest.get("promotion_permitted") is not False
        or authority.get("schema") != OUTPUT_AUTHORITY_SCHEMA
        or authority.get("state") != state
        or authority.get("promotion_permitted") is not False
        or authority.get("artifact") != _MANIFEST_NAME
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError("swing candidate authority does not verify")
    files = _mapping(manifest.get("files"), "candidate files")
    expected_files = {_EVALUATION_NAME, _MODEL_CARD_NAME}
    if state == "candidate":
        expected_files.add(_CANDIDATE_NAME)
    if set(files) != expected_files:
        raise DataReadinessError("swing candidate manifest has an unexpected file set")
    for name, raw in files.items():
        record = _mapping(raw, f"candidate file {name}")
        path = _resolve_inside(root, name)
        if record.get("sha256") != file_sha256(path) or int(record.get("bytes", -1)) != path.stat().st_size:
            raise DataReadinessError(f"swing candidate artifact does not verify: {name}")
    evaluation = _read_json(root / _EVALUATION_NAME, "swing evaluation")
    model_card = _read_json(root / _MODEL_CARD_NAME, "swing model card")
    if state == "no_candidate":
        if (
            evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("status") != "no_candidate"
            or evaluation.get("promotion_permitted") is not False
            or evaluation.get("test_access_count") != 0
            or model_card.get("schema") != MODEL_CARD_SCHEMA
            or model_card.get("status") != "no_candidate"
            or model_card.get("promotion_permitted") is not False
            or model_card.get("candidate_id") is not None
            or evaluation.get("dataset") != model_card.get("dataset")
            or evaluation.get("training_config_sha256")
            != model_card.get("training_config_sha256")
            or evaluation.get("temporal_manifest_policy_sha256")
            != model_card.get("temporal_manifest_policy_sha256")
        ):
            raise DataReadinessError("swing no-candidate evidence is internally inconsistent")
        return {
            "status": "no_candidate",
            "candidate_id": None,
            "manifest": manifest,
            "evaluation": evaluation,
            "model_card": model_card,
        }
    payload = joblib.load(root / _CANDIDATE_NAME)
    if not isinstance(payload, Mapping):
        raise DataReadinessError("swing candidate payload is not an object")
    identities = {"multi_model_bundle"}
    if (
        evaluation.get("schema") != EVALUATION_SCHEMA
        or evaluation.get("status") != "candidate_only"
        or evaluation.get("promotion_permitted") is not False
        or evaluation.get("test_access_count") != 1
        or model_card.get("schema") != MODEL_CARD_SCHEMA
        or model_card.get("status") != "candidate"
        or model_card.get("promotion_permitted") is not False
        or payload.get("schema") != MODEL_SCHEMA
        or payload.get("status") != "candidate"
        or payload.get("promotion_permitted") is not False
        or len(identities) != 1
        or evaluation.get("dataset") != model_card.get("dataset")
        or evaluation.get("dataset") != payload.get("dataset")
        or evaluation.get("training_config_sha256")
        != model_card.get("training_config_sha256")
        or evaluation.get("training_config_sha256")
        != payload.get("training_config_sha256")
        or evaluation.get("temporal_manifest_policy_sha256")
        != model_card.get("temporal_manifest_policy_sha256")
        or evaluation.get("temporal_manifest_policy_sha256")
        != payload.get("temporal_manifest_policy_sha256")
        or model_card.get("feature_columns") != list(payload.get("feature_columns", ()))
        or model_card.get("feature_set_sha256") != payload.get("feature_set_sha256")
    ):
        raise DataReadinessError("swing candidate evidence is internally inconsistent")
    return {
        "status": "candidate",
        "candidate_id": identities.pop(),
        "manifest": manifest,
        "evaluation": evaluation,
        "model_card": model_card,
    }


def _binding_record(binding: SwingPanelBinding, decision_ids_sha256: str) -> dict[str, Any]:
    return {
        "panel_manifest_schema": SWING_MATERIALIZATION_MANIFEST_SCHEMA,
        "panel_manifest_sha256": binding.manifest_sha256,
        "panel_authority_sha256": binding.authority_sha256,
        "panel_request_sha256": binding.request_sha256,
        "strategy_contract_sha256": binding.strategy_contract_sha256,
        "matched_ablation_decision_ids_sha256": decision_ids_sha256,
    }


def _guard(config: SwingTrainingConfig, stage: str, *, peak: bool) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
    if peak:
        assert_peak_memory_budget(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
            stage=stage,
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )



def _resolve_inside(root: Path, raw: object) -> Path:
    path = (root / str(raw)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DataReadinessError(f"artifact escapes authority root: {raw}") from exc
    if not path.is_file():
        raise DataReadinessError(f"authority artifact is missing: {path}")
    return path


def _strict_bool(value: object) -> bool:
    return value is True or isinstance(value, np.bool_) and bool(value)


def _is_unapproved_source_feature(value: str) -> bool:
    normalized = value.lower()
    return any(
        token in normalized
        for token in (
            "source_count_sec_",
            "sec_filing",
            "source_count_finviz_",
            "finviz_news",
            "global_context",
            "gdelt",
            "reddit",
            "seeking_alpha",
        )
    )



def _sequence_sha256(values: Sequence[str] | pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _iso(value: object) -> str:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return cast(str, parsed.tz_convert("UTC").isoformat())
