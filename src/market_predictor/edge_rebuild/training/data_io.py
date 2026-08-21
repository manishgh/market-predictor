"""Production-grade candidate training for the ten-session edge-rebuild swing strategy."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_artifact_contracts import (
    SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
    SWING_MATERIALIZATION_MANIFEST_SCHEMA,
)
from market_predictor.edge_rebuild.swing_features import (
    MANAGED_BENCHMARK_RETURN_COLUMNS,
    MANAGED_EXCESS_RETURN_COLUMNS,
    MANAGED_PATH_NET_RETURN_COLUMNS,
    MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
    SWING_FEATURE_PANEL_SCHEMA,
    SWING_FEATURE_PROFILE,
    swing_model_feature_columns,
)
from market_predictor.edge_rebuild.training.swing_types import (
    SwingPanelBinding,
    SwingProfileData,
    SwingTrainingConfig,
    _guard,
    _is_unapproved_source_feature,
    _read_json,
    _resolve_inside,
    _sequence_sha256,
    _strict_bool,
)
from market_predictor.edge_rebuild.training.utils import (
    _mapping,
)
from market_predictor.resources import (
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

TRAINING_SCHEMA: Final = "edge_rebuild.swing_training.v5"
MODEL_SCHEMA: Final = "edge_rebuild.swing_candidate.v5"
EVALUATION_SCHEMA: Final = "edge_rebuild.swing_evaluation.v7"
MODEL_CARD_SCHEMA: Final = "edge_rebuild.swing_model_card.v7"
OUTPUT_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_candidate_authority.v5"
SWING_BASELINE_BUNDLE_PREFIX: Final = "swing_baseline_bundle."
DECISION_START_DATE: Final = date(2019, 7, 9)
HORIZON_SESSIONS: Final = 10
ALLOWED_PROFILES: Final = (
    SWING_FEATURE_PROFILE,
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
        raise DataReadinessError("swing panel must contain only the frozen technical profile")
    if str(manifest.get("first_session")) != config.decision_start_date:
        raise DataReadinessError(
            "swing panel decisions must start exactly on 2019-07-09; "
            "pre-cutoff bars may exist only in the upstream warm-up store"
        )
    if int(manifest.get("rows", -1)) < config.minimum_rows:
        raise DataReadinessError("swing panel has too few rows for training")
    if int(manifest.get("securities", -1)) < config.minimum_securities:
        raise DataReadinessError("swing panel has too few securities for training")
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
    feature_columns = swing_model_feature_columns(
        contract=strategy_contract,
        catalyst=False,
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
    row_filter = (
        row_filter
        & (pds.field("cross_section_eligible") == True)  # type: ignore[attr-defined,no-untyped-call]  # noqa: E712
        & pds.field("rank_label").is_valid()  # type: ignore[attr-defined,no-untyped-call]
    )

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
    expected_set = set(governed_sessions)
    extra = sorted(observed_sessions.difference(expected_set))
    if extra:
        raise DataReadinessError(
            f"swing profile contains sessions outside governance: {extra[:10]}"
        )
    missing = expected_set.difference(observed_sessions)
    if missing:
        raise DataReadinessError(
            f"swing technical profile is missing governed sessions: {sorted(missing)[:10]}"
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
    # strings, one bounded split projection, and estimator workspace.
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
    if (
        len(data) < config.minimum_rows
        or data["security_id"].nunique() < config.minimum_securities
    ):
        raise DataReadinessError(
            f"eligible {profile} population is below training minimums"
        )
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


















































