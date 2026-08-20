"""Governed development experiments for swing broker-action specialists.

The module deliberately stops before the locked test. It compares technical,
broker-action, and combined features on identical event-conditioned decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final
from uuid import uuid4

import exchange_calendars as xcals
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.edge_rebuild.swing_event_ablation import (
    AUTHORITY_SCHEMA as SOURCE_AUTHORITY_SCHEMA,
)
from market_predictor.edge_rebuild.swing_event_ablation import (
    COMBINED_PROFILE,
    EVENT_PROFILE,
    TECHNICAL_PROFILE,
)
from market_predictor.edge_rebuild.swing_event_ablation import (
    MANIFEST_SCHEMA as SOURCE_MANIFEST_SCHEMA,
)
from market_predictor.edge_rebuild.swing_selection import (
    select_constrained_swing_portfolio,
)
from market_predictor.edge_rebuild.swing_training import (
    SwingTrainingConfig,
    _evaluation_columns,
    _evaluation_metrics,
    load_swing_training_config,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

POLICY_SCHEMA: Final = "market_predictor.swing_broker_action_specialists.v1"
REQUEST_SCHEMA: Final = "edge_rebuild.swing_broker_specialist_request.v1"
MANIFEST_SCHEMA: Final = "edge_rebuild.swing_broker_specialist_manifest.v1"
AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_broker_specialist_authority.v1"
CAPACITY_SCHEMA: Final = "edge_rebuild.swing_broker_specialist_capacity.v1"
MODEL_SCHEMA: Final = "edge_rebuild.swing_broker_specialist_model.v1"

RATING_SPECIALIST: Final = "rating_change"
COVERAGE_SPECIALIST: Final = "coverage_initiation"
SPECIALISTS: Final = (RATING_SPECIALIST, COVERAGE_SPECIALIST)
UPGRADE_SPECIALIST: Final = "rating_upgrade"
DOWNGRADE_SPECIALIST: Final = "rating_downgrade"
DIRECTIONAL_SPECIALISTS: Final = (UPGRADE_SPECIALIST, DOWNGRADE_SPECIALIST)
SUPPORTED_SPECIALIST_SETS: Final = frozenset({SPECIALISTS, DIRECTIONAL_SPECIALISTS})
PROFILE_MAP: Final = {
    "technical_only": TECHNICAL_PROFILE,
    "broker_action_only": EVENT_PROFILE,
    "technical_plus_broker_action": COMBINED_PROFILE,
}
ESTIMATORS: Final = ("logistic", "hist_gradient_boosting")
_XNYS: Final = xcals.get_calendar("XNYS")

_IDENTITY_COLUMNS: Final = (
    "decision_id",
    "decision_group_id",
    "ticker",
    "security_id",
    "sector",
    "market_regime",
    "session_date_et",
    "decision_time_utc",
    "feature_available_at_utc",
    "label_available_at_utc",
    "analyst_revision_latest_feature_available_at_utc",
    "analyst_revision_episode_id",
    "analyst_revision_episode_sample_weight",
    "ranking_reliability_weight",
)
_OUTCOME_COLUMNS: Final = (
    "rank_label",
    "barrier_net_return",
    "future_net_return_10d",
    "future_excess_return_10d_vs_spy",
    "future_excess_return_10d_vs_qqq",
    "future_excess_return_10d_vs_sector",
)
_SUBTYPE_COLUMNS: Final = (
    "analyst_revision_latest_is_upgrade",
    "analyst_revision_latest_is_downgrade",
    "analyst_revision_latest_is_coverage",
    "analyst_revision_latest_direction_unverified",
)


@dataclass(frozen=True, slots=True)
class BrokerSpecialistPolicy:
    source_artifact_schema: str
    source_event_family: str
    development_start: date
    development_end: date
    model_selection_training_end: date
    model_selection_embargo_start: date
    model_selection_embargo_end: date
    model_selection_validation_start: date
    validation_start: date
    validation_end: date
    locked_test_start: date
    specialists: tuple[str, ...]
    profiles: tuple[str, ...]
    estimators: tuple[str, ...]
    report_only_subtypes: tuple[str, ...]
    minimum_development_announcements: int
    minimum_validation_announcements: int
    minimum_validation_securities: int
    minimum_validation_sectors: int
    minimum_unseen_validation_announcements: int
    minimum_validation_roc_auc: float
    minimum_brier_skill: float
    maximum_expected_calibration_error: float
    minimum_selected_announcements: int
    minimum_unseen_selected_announcements: int
    probability_thresholds: tuple[float, ...]
    calibration_fraction: float
    minimum_calibration_sessions: int
    bootstrap_samples: int
    bootstrap_block_sessions: int
    unseen_security_holdout_fraction: float
    unseen_security_hash_seed: int
    random_seed: int
    maximum_process_memory_gib: float
    memory_guard_headroom_gib: float

    def __post_init__(self) -> None:
        if self.source_artifact_schema != SOURCE_MANIFEST_SCHEMA:
            raise ValueError("broker specialist source schema differs")
        if self.source_event_family != "analyst_revision":
            raise ValueError("broker specialist source family differs")
        if self.specialists not in SUPPORTED_SPECIALIST_SETS:
            raise ValueError("broker specialist definitions differ")
        if self.profiles != tuple(PROFILE_MAP):
            raise ValueError("broker specialist profiles differ")
        if self.estimators != ESTIMATORS:
            raise ValueError("broker specialist estimator matrix differs")
        if (
            self.development_start != date(2019, 7, 9)
            or self.development_end != date(2024, 5, 28)
            or self.model_selection_training_end != date(2023, 5, 26)
            or self.model_selection_embargo_start != date(2023, 5, 30)
            or self.model_selection_embargo_end != date(2023, 6, 12)
            or self.model_selection_validation_start != date(2023, 6, 13)
            or self.validation_start != date(2024, 6, 12)
            or self.validation_end != date(2025, 6, 13)
            or self.locked_test_start != date(2025, 7, 1)
        ):
            raise ValueError("broker specialist chronological split differs")
        if len(_session_calendar(self.model_selection_embargo_start, self.model_selection_embargo_end)) != 10:
            raise ValueError("broker specialist inner embargo must cover ten exchange sessions")
        if not (
            self.model_selection_training_end < self.model_selection_embargo_start
            <= self.model_selection_embargo_end < self.model_selection_validation_start
            <= self.development_end < self.validation_start
            <= self.validation_end < self.locked_test_start
        ):
            raise ValueError("broker specialist chronological ranges overlap")
        positive_counts = (
            self.minimum_development_announcements,
            self.minimum_validation_announcements,
            self.minimum_validation_securities,
            self.minimum_validation_sectors,
            self.minimum_unseen_validation_announcements,
            self.minimum_selected_announcements,
            self.minimum_unseen_selected_announcements,
            self.minimum_calibration_sessions,
            self.bootstrap_samples,
            self.bootstrap_block_sessions,
        )
        if any(value < 1 for value in positive_counts):
            raise ValueError("broker specialist count gates must be positive")
        if not 0.5 < self.minimum_validation_roc_auc <= 1.0:
            raise ValueError("broker specialist AUC gate must exceed chance")
        if not -1.0 < self.minimum_brier_skill < 1.0:
            raise ValueError("broker specialist Brier-skill gate is invalid")
        if not 0.0 < self.maximum_expected_calibration_error <= 0.25:
            raise ValueError("broker specialist calibration-error gate is invalid")
        if not 0.1 <= self.calibration_fraction <= 0.35:
            raise ValueError("broker specialist calibration fraction is invalid")
        if self.unseen_security_holdout_fraction != 0.20:
            raise ValueError("broker specialist unseen-security holdout must be 20%")
        if tuple(sorted(set(self.probability_thresholds))) != self.probability_thresholds:
            raise ValueError("broker specialist thresholds must be unique and ascending")
        if any(not 0.0 < value < 1.0 for value in self.probability_thresholds):
            raise ValueError("broker specialist thresholds must be in (0, 1)")
        if not 0 < self.maximum_process_memory_gib <= 5.0:
            raise ValueError("broker specialist memory budget exceeds 5 GiB")
        if not 0 < self.memory_guard_headroom_gib < self.maximum_process_memory_gib:
            raise ValueError("broker specialist memory headroom is invalid")


@dataclass(frozen=True, slots=True)
class _FittedModel:
    estimator: Any
    calibrator: LogisticRegression
    feature_columns: tuple[str, ...]
    train_prevalence: float


@dataclass(frozen=True, slots=True)
class _ScopePrediction:
    frame: pd.DataFrame
    probability: np.ndarray
    train_prevalence: float


def load_broker_specialist_policy(path: Path) -> BrokerSpecialistPolicy:
    """Load the complete frozen policy; partial policies fail closed."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"broker specialist policy is unreadable: {path}") from exc
    if raw.pop("schema_version", None) != POLICY_SCHEMA:
        raise DataReadinessError("broker specialist policy schema differs")
    expected = {field.name for field in fields(BrokerSpecialistPolicy)}
    if set(raw) != expected:
        raise DataReadinessError(
            "broker specialist policy fields differ; "
            f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}"
        )
    values = dict(raw)
    for name in (
        "specialists",
        "profiles",
        "estimators",
        "report_only_subtypes",
        "probability_thresholds",
    ):
        values[name] = tuple(values[name])
    try:
        return BrokerSpecialistPolicy(**values)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("broker specialist policy is invalid") from exc


def train_swing_broker_specialists(
    *,
    source_directory: Path,
    output_directory: Path,
    policy_path: Path,
    strategy_contract_path: Path,
    swing_training_policy_path: Path,
) -> Mapping[str, Any]:
    """Audit and train two development-only specialists without reading locked test."""

    policy = load_broker_specialist_policy(policy_path)
    strategy_contract = load_strategy_contract(strategy_contract_path)
    swing_training_config = load_swing_training_config(swing_training_policy_path)
    if (
        swing_training_config.expected_round_trip_cost_bps
        != strategy_contract.swing.round_trip_cost_bps
        or swing_training_config.maximum_trades_per_decision
        != strategy_contract.swing.maximum_trades_per_decision
        or policy.bootstrap_samples != swing_training_config.bootstrap_samples
        or policy.bootstrap_block_sessions != swing_training_config.bootstrap_block_sessions
    ):
        raise DataReadinessError("broker specialist and canonical swing economics differ")
    if output_directory.exists():
        raise FileExistsError(f"immutable output already exists: {output_directory}")
    source = _verify_source_hashes(
        source_directory,
        policy,
        strategy_contract=strategy_contract,
    )
    policy_record = _policy_record(policy)
    request = {
        "schema": REQUEST_SCHEMA,
        "source_directory": str(source_directory.resolve()),
        "source_manifest_sha256": file_sha256(source_directory / "_manifest.json"),
        "source_authority_sha256": file_sha256(source_directory / "_authority.json"),
        "policy": policy_record,
        "policy_file_sha256": file_sha256(policy_path),
        "strategy_contract_sha256": strategy_contract.sha256(),
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "swing_training_policy": asdict(swing_training_config),
        "swing_training_policy_file_sha256": file_sha256(swing_training_policy_path),
        "locked_test_policy": (
            "partitions beginning 2025-07 are never opened; validation selects a "
            "development candidate only"
        ),
    }
    request["request_sha256"] = _json_sha256(request)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    try:
        staging.mkdir(parents=True, exist_ok=False)
        _write_json(staging / "_request.json", request)
        subtype_rows = _load_subtype_identity_rows(source_directory, source, policy)
        capacity = _capacity_audit(subtype_rows, policy)
        capacity_path = staging / "capacity_audit.parquet"
        capacity.to_parquet(capacity_path, index=False)
        capacity_record = {
            "schema": CAPACITY_SCHEMA,
            "path": capacity_path.name,
            "sha256": file_sha256(capacity_path),
            "rows": len(capacity),
            "records": capacity.to_dict(orient="records"),
        }
        evaluations: list[dict[str, Any]] = []
        selected_models: dict[str, dict[str, Any]] = {}
        for specialist in policy.specialists:
            specialist_ids = _specialist_decision_ids(subtype_rows, specialist)
            specialist_capacity = capacity.loc[capacity["specialist"].eq(specialist)]
            capacity_passed = bool(specialist_capacity["capacity_passed"].all())
            if not capacity_passed:
                evaluations.append(
                    {
                        "specialist": specialist,
                        "status": "insufficient_capacity",
                        "locked_test_outcomes_read": False,
                        "experiments": [],
                    }
                )
                continue
            specialist_result, fitted = _run_specialist_experiments(
                source_directory=source_directory,
                source=source,
                specialist=specialist,
                decision_ids=specialist_ids,
                policy=policy,
                strategy_contract=strategy_contract,
                swing_training_config=swing_training_config,
            )
            evaluations.append(specialist_result)
            if fitted is not None:
                selected_models[specialist] = fitted
            release_process_memory()
            _guard(policy, f"broker specialist {specialist}", peak=True)
        model_records: list[dict[str, Any]] = []
        for specialist, payload in selected_models.items():
            path = staging / "models" / specialist / "candidate.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, path, compress=3)
            model_records.append(
                {
                    "specialist": specialist,
                    "path": str(path.relative_to(staging)).replace("\\", "/"),
                    "sha256": file_sha256(path),
                    "model_schema": MODEL_SCHEMA,
                }
            )
        status = (
            "development_candidate"
            if model_records
            else "no_development_candidate"
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "state": "complete",
            "status": status,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "request_sha256": request["request_sha256"],
            "source_manifest_sha256": request["source_manifest_sha256"],
            "capacity_audit": capacity_record,
            "specialists": evaluations,
            "models": model_records,
            "price_target_policy": {
                "status": "report_only",
                "reason": "insufficient independently aligned latest announcements",
                "latest_direction_unverified_announcements": int(
                    subtype_rows.loc[
                        subtype_rows["subtype"].eq("price_target_or_generic"),
                        "analyst_revision_episode_id",
                    ].nunique()
                ),
                "configured_subtypes": list(policy.report_only_subtypes),
            },
            "locked_test_outcomes_read": False,
            "promotion_permitted": False,
            "memory": memory_audit(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
            ).to_record(),
        }
        _write_json(staging / "_manifest.json", manifest)
        _write_json(
            staging / "_authority.json",
            {
                "schema": AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "request_sha256": request["request_sha256"],
                "locked_test_outcomes_read": False,
                "promotion_permitted": False,
            },
        )
        _verify_output(
            staging,
            source_directory=source_directory,
            policy_path=policy_path,
            strategy_contract_path=strategy_contract_path,
            swing_training_policy_path=swing_training_policy_path,
        )
        os.replace(staging, output_directory)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_source_hashes(
    directory: Path,
    policy: BrokerSpecialistPolicy,
    *,
    strategy_contract: StrategyContract,
) -> dict[str, Any]:
    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    request_path = directory / "_request.json"
    request = _read_json(request_path, "broker-action source request")
    manifest = _read_json(manifest_path, "broker-action source manifest")
    authority = _read_json(authority_path, "broker-action source authority")
    request_identity = dict(request)
    request_sha256 = request_identity.pop("request_sha256", None)
    if (
        request_sha256 != _json_sha256(request_identity)
        or manifest.get("schema") != policy.source_artifact_schema
        or authority.get("schema") != SOURCE_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("research_training_eligible") is not True
    ):
        raise DataReadinessError("broker-action source authority is invalid")
    _assert_source_strategy_contract(request, strategy_contract)
    _verify_source_upstream_bindings(request)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise DataReadinessError("broker-action source has no partitions")
    observed_profiles: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("broker-action source partition is malformed")
        profile, month, relative = _source_record_identity(raw)
        if month >= policy.locked_test_start.strftime("%Y-%m"):
            continue
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents or not path.is_file():
            raise DataReadinessError("broker-action source partition path is invalid")
        if file_sha256(path) != raw.get("sha256"):
            raise DataReadinessError("broker-action source partition hash changed")
        observed_profiles.add(profile)
    if observed_profiles != set(PROFILE_MAP.values()):
        raise DataReadinessError("broker-action source profiles differ")
    return manifest


def _assert_source_strategy_contract(
    request: Mapping[str, Any],
    strategy_contract: StrategyContract,
) -> None:
    if request.get("strategy_contract_sha256") != strategy_contract.sha256():
        raise DataReadinessError(
            "broker-action source strategy contract differs from specialist training"
        )


def _verify_source_upstream_bindings(request: Mapping[str, Any]) -> None:
    for key in ("event_authorities", "precision_audits"):
        records = request.get(key)
        if not isinstance(records, list) or len(records) != 2:
            raise DataReadinessError(f"broker-action source {key} bindings are invalid")
        for raw in records:
            if not isinstance(raw, Mapping):
                raise DataReadinessError(f"broker-action source {key} record is invalid")
            authority_path = Path(str(raw.get("directory", ""))) / "_authority.json"
            if not authority_path.is_file() or file_sha256(authority_path) != raw.get(
                "authority_sha256"
            ):
                raise DataReadinessError(f"broker-action source {key} binding changed")
    technical = Path(str(request.get("technical_panel_directory", ""))) / "final"
    if (
        not (technical / "_authority.json").is_file()
        or not (technical / "_manifest.json").is_file()
        or file_sha256(technical / "_authority.json")
        != request.get("technical_panel_authority_sha256")
        or file_sha256(technical / "_manifest.json")
        != request.get("technical_panel_manifest_sha256")
    ):
        raise DataReadinessError("broker-action technical-panel binding changed")


def _load_subtype_identity_rows(
    directory: Path,
    manifest: Mapping[str, Any],
    policy: BrokerSpecialistPolicy,
) -> pd.DataFrame:
    records = _profile_records(manifest, EVENT_PROFILE, policy)
    columns = [*_IDENTITY_COLUMNS, *_SUBTYPE_COLUMNS]
    frames = []
    for record in records:
        table = pq.ParquetFile(  # type: ignore[no-untyped-call]
            directory / str(record["path"])
        ).read(columns=columns)
        frames.append(table.to_pandas())
    frame = pd.concat(frames, ignore_index=True)
    frame["session_date_et"] = pd.to_datetime(frame["session_date_et"], errors="coerce")
    if frame["session_date_et"].isna().any() or bool(
        frame["session_date_et"].ge(pd.Timestamp(policy.locked_test_start)).any()
    ):
        raise DataReadinessError("locked-test rows entered broker specialist identity load")
    flags = frame.loc[:, list(_SUBTYPE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if flags.isna().any().any() or not flags.isin([0, 1]).all().all():
        raise DataReadinessError("broker-action subtype flags are invalid")
    frame["subtype"] = _classify_latest_subtypes(flags)
    if frame["decision_id"].astype(str).duplicated().any():
        raise DataReadinessError("broker-action identity rows contain duplicate decisions")
    return frame


def _classify_latest_subtypes(flags: pd.DataFrame) -> pd.Series:
    """Classify action identity separately from direction availability."""

    named_sum = flags.loc[
        :,
        [
            "analyst_revision_latest_is_upgrade",
            "analyst_revision_latest_is_downgrade",
            "analyst_revision_latest_is_coverage",
        ],
    ].sum(axis=1)
    unverified = flags["analyst_revision_latest_direction_unverified"]
    if (
        named_sum.gt(1).any()
        or (named_sum.eq(0) & unverified.ne(1)).any()
        or (named_sum.eq(1) & unverified.ne(flags["analyst_revision_latest_is_coverage"])).any()
    ):
        raise DataReadinessError("broker-action latest subtype flags are inconsistent")
    result = pd.Series("price_target_or_generic", index=flags.index, dtype="string")
    result.loc[flags["analyst_revision_latest_is_upgrade"].eq(1)] = "rating_upgrade"
    result.loc[flags["analyst_revision_latest_is_downgrade"].eq(1)] = "rating_downgrade"
    result.loc[flags["analyst_revision_latest_is_coverage"].eq(1)] = "coverage_initiation"
    return result


def _profile_records(
    manifest: Mapping[str, Any],
    profile: str,
    policy: BrokerSpecialistPolicy,
) -> list[Mapping[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise DataReadinessError("broker-action source file records are invalid")
    result: list[Mapping[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping) or raw.get("feature_profile") != profile:
            continue
        observed_profile, month, _relative = _source_record_identity(raw)
        if observed_profile != profile:
            raise DataReadinessError("broker-action source profile identity differs")
        if month >= policy.locked_test_start.strftime("%Y-%m"):
            continue
        result.append(raw)
    if not result:
        raise DataReadinessError(f"broker-action source has no development partitions for {profile}")
    return result


def _source_record_identity(raw: Mapping[str, Any]) -> tuple[str, str, str]:
    profile = str(raw.get("feature_profile", ""))
    month = str(raw.get("partition_month", ""))
    relative = str(raw.get("path", "")).replace("\\", "/")
    normalized = PurePosixPath(relative)
    expected = f"panel/feature_profile={profile}/month={month}/part.parquet"
    if (
        profile not in PROFILE_MAP.values()
        or len(month) != 7
        or normalized.is_absolute()
        or ".." in normalized.parts
        or relative != expected
    ):
        raise DataReadinessError("broker-action source partition identity is invalid")
    return profile, month, relative


def _specialist_decision_ids(frame: pd.DataFrame, specialist: str) -> set[str]:
    if specialist == RATING_SPECIALIST:
        mask = frame["subtype"].isin(["rating_upgrade", "rating_downgrade"])
    elif specialist == COVERAGE_SPECIALIST:
        mask = frame["subtype"].eq("coverage_initiation")
    elif specialist == UPGRADE_SPECIALIST:
        mask = frame["subtype"].eq("rating_upgrade")
    elif specialist == DOWNGRADE_SPECIALIST:
        mask = frame["subtype"].eq("rating_downgrade")
    else:
        raise DataReadinessError(f"unknown broker specialist: {specialist}")
    return set(frame.loc[mask, "decision_id"].astype(str))


def _capacity_audit(
    frame: pd.DataFrame,
    policy: BrokerSpecialistPolicy,
) -> pd.DataFrame:
    holdout = _unseen_security_mask(frame["security_id"].astype(str), policy)
    records: list[dict[str, Any]] = []
    for specialist in policy.specialists:
        ids = _specialist_decision_ids(frame, specialist)
        subset = frame.loc[frame["decision_id"].astype(str).isin(ids)].copy()
        for split, start, end, unseen_only in (
            ("development", policy.development_start, policy.development_end, False),
            ("validation", policy.validation_start, policy.validation_end, False),
            ("unseen_security_validation", policy.validation_start, policy.validation_end, True),
        ):
            mask = subset["session_date_et"].between(pd.Timestamp(start), pd.Timestamp(end))
            if unseen_only:
                mask &= holdout.loc[subset.index]
            part = subset.loc[mask]
            announcements = int(part["analyst_revision_episode_id"].nunique())
            securities = int(part["security_id"].nunique())
            sectors = int(part["sector"].nunique())
            if split == "development":
                passed = announcements >= policy.minimum_development_announcements
            elif split == "validation":
                passed = (
                    announcements >= policy.minimum_validation_announcements
                    and securities >= policy.minimum_validation_securities
                    and sectors >= policy.minimum_validation_sectors
                )
            else:
                passed = announcements >= policy.minimum_unseen_validation_announcements
            if specialist == RATING_SPECIALIST and not part.empty:
                passed = passed and {"rating_upgrade", "rating_downgrade"}.issubset(
                    set(part["subtype"].astype(str))
                )
            records.append(
                {
                    "specialist": specialist,
                    "split": split,
                    "rows": len(part),
                    "announcements": announcements,
                    "securities": securities,
                    "sectors": sectors,
                    "sessions": int(part["session_date_et"].nunique()),
                    "rating_up_announcements": int(
                        part.loc[part["subtype"].eq("rating_upgrade"), "analyst_revision_episode_id"].nunique()
                    ),
                    "rating_down_announcements": int(
                        part.loc[part["subtype"].eq("rating_downgrade"), "analyst_revision_episode_id"].nunique()
                    ),
                    "coverage_announcements": int(
                        part.loc[part["subtype"].eq("coverage_initiation"), "analyst_revision_episode_id"].nunique()
                    ),
                    "capacity_passed": bool(passed),
                }
            )
    return pd.DataFrame.from_records(records)


def _run_specialist_experiments(
    *,
    source_directory: Path,
    source: Mapping[str, Any],
    specialist: str,
    decision_ids: set[str],
    policy: BrokerSpecialistPolicy,
    strategy_contract: StrategyContract,
    swing_training_config: SwingTrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    experiments: list[dict[str, Any]] = []
    feature_columns_by_id: dict[str, tuple[str, ...]] = {}
    for public_profile, source_profile in PROFILE_MAP.items():
        frame, feature_columns = _load_training_profile(
            source_directory,
            source,
            source_profile=source_profile,
            decision_ids=decision_ids,
            policy=policy,
            end_date=policy.development_end,
        )
        for estimator in ESTIMATORS:
            experiment_id = f"{specialist}.{public_profile}.{estimator}"
            result = _evaluate_experiment(
                frame,
                feature_columns=feature_columns,
                estimator_family=estimator,
                experiment_id=experiment_id,
                policy=policy,
                strategy_contract=strategy_contract,
                swing_training_config=swing_training_config,
            )
            experiments.append(result)
            feature_columns_by_id[experiment_id] = feature_columns
        del frame
        release_process_memory()
        _guard(policy, f"broker specialist {specialist} {public_profile}", peak=True)
    eligible = [record for record in experiments if record["candidate_eligible"]]
    if not eligible:
        return (
            {
                "specialist": specialist,
                "status": "no_development_candidate",
                "reason": "no experiment passed frozen inner model-selection gates",
                "outer_validation_opened": False,
                "locked_test_outcomes_read": False,
                "experiments": experiments,
            },
            None,
        )
    selected = max(eligible, key=_experiment_selection_key)
    selected_id = str(selected["experiment_id"])
    refit_frame, _ = _load_training_profile(
        source_directory,
        source,
        source_profile=PROFILE_MAP[str(selected["profile"])],
        decision_ids=decision_ids,
        policy=policy,
        end_date=policy.validation_end,
    )
    development = refit_frame.loc[
        refit_frame["session_date_et"].between(
            policy.development_start.isoformat(), policy.development_end.isoformat()
        )
    ].copy()
    validation = refit_frame.loc[
        refit_frame["session_date_et"].between(
            policy.validation_start.isoformat(), policy.validation_end.isoformat()
        )
    ].copy()
    outer_scopes, fitted = _evaluate_fixed_policy(
        development,
        validation,
        feature_columns=feature_columns_by_id[selected_id],
        estimator_family=str(selected["estimator_family"]),
        threshold=float(selected["selected_probability_threshold"]),
        policy=policy,
        strategy_contract=strategy_contract,
        swing_training_config=swing_training_config,
        calendar=_session_calendar(policy.validation_start, policy.validation_end),
    )
    outer_passed, outer_reasons = _acceptance_gate(outer_scopes, policy)
    outer_record = {
        "status": "passed" if outer_passed else "failed",
        "probability_threshold": selected["selected_probability_threshold"],
        "scopes": outer_scopes,
        "reasons": outer_reasons,
        "selection_frozen_before_open": True,
    }
    if not outer_passed:
        return (
            {
                "specialist": specialist,
                "status": "no_development_candidate",
                "reason": "selected inner candidate failed untouched outer validation",
                "selected_experiment_id": selected_id,
                "selection_basis": "inner_chronological_selection_only",
                "outer_validation_opened": True,
                "outer_validation": outer_record,
                "locked_test_outcomes_read": False,
                "experiments": experiments,
            },
            None,
        )
    payload = {
        "schema": MODEL_SCHEMA,
        "status": "development_candidate",
        "promotion_permitted": False,
        "locked_test_outcomes_read": False,
        "specialist": specialist,
        "experiment_id": selected_id,
        "profile": selected["profile"],
        "estimator_family": selected["estimator_family"],
        "feature_columns": list(feature_columns_by_id[selected_id]),
        "model": fitted,
    }
    return (
        {
            "specialist": specialist,
            "status": "development_candidate",
            "selected_experiment_id": selected_id,
            "selection_basis": "inner_chronological_selection_then_one_outer_validation",
            "outer_validation_opened": True,
            "outer_validation": outer_record,
            "locked_test_outcomes_read": False,
            "experiments": experiments,
        },
        payload,
    )


def _load_training_profile(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    source_profile: str,
    decision_ids: set[str],
    policy: BrokerSpecialistPolicy,
    end_date: date,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    records = [
        record
        for record in _profile_records(manifest, source_profile, policy)
        if str(record["partition_month"]) <= end_date.strftime("%Y-%m")
    ]
    first = records[0].get("model_feature_columns")
    if not isinstance(first, list) or not first:
        raise DataReadinessError(f"broker-action profile has no features: {source_profile}")
    feature_columns = tuple(str(value) for value in first)
    required = list(
        dict.fromkeys(
            (
                *_IDENTITY_COLUMNS,
                *(column for column in _evaluation_columns() if column != "target"),
                "rank_label",
                *feature_columns,
            )
        )
    )
    frames: list[pd.DataFrame] = []
    for record in records:
        if tuple(record.get("model_feature_columns", ())) != feature_columns:
            raise DataReadinessError("broker-action model features changed by partition")
        frame = pq.ParquetFile(directory / str(record["path"])).read(  # type: ignore[no-untyped-call]
            columns=required
        ).to_pandas()
        frame = frame.loc[frame["decision_id"].astype(str).isin(decision_ids)]
        frame = frame.loc[
            pd.to_datetime(frame["session_date_et"], errors="coerce").dt.date.le(end_date)
        ]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise DataReadinessError(f"broker specialist profile is empty: {source_profile}")
    data = pd.concat(frames, ignore_index=True)
    data["session_date_et"] = pd.to_datetime(data["session_date_et"], errors="coerce").dt.date.astype(str)
    if data["decision_id"].astype(str).duplicated().any():
        raise DataReadinessError("broker specialist decisions are duplicated")
    for column in (
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "analyst_revision_latest_feature_available_at_utc",
    ):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
        if data[column].isna().any():
            raise DataReadinessError(f"broker specialist timestamp is invalid: {column}")
    if (
        data["feature_available_at_utc"].gt(data["decision_time_utc"]).any()
        or data["analyst_revision_latest_feature_available_at_utc"].gt(data["decision_time_utc"]).any()
        or data["label_available_at_utc"].le(data["decision_time_utc"]).any()
    ):
        raise DataReadinessError("broker specialist chronology is not causal")
    if pd.to_datetime(data["session_date_et"]).ge(pd.Timestamp(policy.locked_test_start)).any():
        raise DataReadinessError("locked-test outcomes entered broker specialist training")
    data["target"] = pd.to_numeric(data["rank_label"], errors="coerce").eq(1).astype("int8")
    data["sample_weight"] = (
        pd.to_numeric(data["analyst_revision_episode_sample_weight"], errors="coerce")
        * pd.to_numeric(data["ranking_reliability_weight"], errors="coerce")
    )
    if data["sample_weight"].isna().any() or data["sample_weight"].le(0).any():
        raise DataReadinessError("broker specialist sample weights are invalid")
    for column in (*_OUTCOME_COLUMNS[1:], *feature_columns):
        values = pd.to_numeric(data[column], errors="coerce")
        array = values.to_numpy(dtype="float64", na_value=np.nan)
        if np.isinf(array).any():
            raise DataReadinessError(f"broker specialist numeric column contains infinity: {column}")
        if column in _OUTCOME_COLUMNS and not np.isfinite(array).all():
            raise DataReadinessError(f"broker specialist outcome is incomplete: {column}")
        if column in feature_columns and not np.isfinite(array).any():
            raise DataReadinessError(f"broker specialist feature is entirely missing: {column}")
        data[column] = values.astype("float32")
    return data.sort_values(["decision_time_utc", "security_id"], kind="stable"), feature_columns


def _evaluate_experiment(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    estimator_family: str,
    experiment_id: str,
    policy: BrokerSpecialistPolicy,
    strategy_contract: StrategyContract,
    swing_training_config: SwingTrainingConfig,
) -> dict[str, Any]:
    training = frame.loc[
        frame["session_date_et"].between(
            policy.development_start.isoformat(), policy.model_selection_training_end.isoformat()
        )
    ].copy()
    selection = frame.loc[
        frame["session_date_et"].between(
            policy.model_selection_validation_start.isoformat(), policy.development_end.isoformat()
        )
    ].copy()
    if training["label_available_at_utc"].max() >= selection["decision_time_utc"].min():
        raise DataReadinessError("broker specialist training labels overlap model selection")
    predictions, _fitted = _fit_scope_predictions(
        training,
        selection,
        feature_columns=feature_columns,
        estimator_family=estimator_family,
        policy=policy,
    )
    thresholds: list[dict[str, Any]] = []
    for threshold in policy.probability_thresholds:
        try:
            threshold_scopes = _canonical_scope_metrics(
                predictions,
                threshold=threshold,
                policy=policy,
                strategy_contract=strategy_contract,
                swing_training_config=swing_training_config,
                calendar=_session_calendar(
                    policy.model_selection_validation_start,
                    policy.development_end,
                ),
            )
            passed, reasons = _acceptance_gate(threshold_scopes, policy)
        except DataReadinessError as exc:
            threshold_scopes = {}
            passed = False
            reasons = [str(exc)]
        thresholds.append(
            {
                "probability_threshold": threshold,
                "eligible": passed,
                "reasons": reasons,
                "scopes": threshold_scopes,
            }
        )
    eligible = [record for record in thresholds if record["eligible"]]
    diagnostic = eligible or [record for record in thresholds if record["scopes"]] or thresholds
    selected = max(diagnostic, key=_threshold_selection_key)
    profile = experiment_id.split(".", maxsplit=2)[1]
    return {
        "experiment_id": experiment_id,
        "profile": profile,
        "estimator_family": estimator_family,
        "feature_count": len(feature_columns),
        "selection_window": {
            "training_end": policy.model_selection_training_end.isoformat(),
            "embargo_start": policy.model_selection_embargo_start.isoformat(),
            "embargo_end": policy.model_selection_embargo_end.isoformat(),
            "validation_start": policy.model_selection_validation_start.isoformat(),
            "validation_end": policy.development_end.isoformat(),
        },
        "thresholds": thresholds,
        "selected_probability_threshold": selected["probability_threshold"],
        "selected_scopes": selected["scopes"],
        "candidate_eligible": bool(eligible),
        "outer_validation_opened": False,
        "locked_test_outcomes_read": False,
    }


def _fit_model(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    estimator_family: str,
    policy: BrokerSpecialistPolicy,
) -> _FittedModel:
    sessions = tuple(sorted(frame["session_date_et"].unique()))
    calibration_count = max(
        policy.minimum_calibration_sessions,
        math.ceil(len(sessions) * policy.calibration_fraction),
    )
    if calibration_count >= len(sessions):
        raise DataReadinessError("broker specialist lacks fit sessions before calibration")
    calibration_sessions = set(sessions[-calibration_count:])
    calibration_start = pd.to_datetime(frame.loc[
        frame["session_date_et"].isin(calibration_sessions), "decision_time_utc"
    ]).min()
    fit_mask = (
        ~frame["session_date_et"].isin(calibration_sessions)
        & frame["label_available_at_utc"].lt(calibration_start)
    )
    calibration_mask = frame["session_date_et"].isin(calibration_sessions)
    if frame.loc[fit_mask, "target"].nunique() != 2 or frame.loc[calibration_mask, "target"].nunique() != 2:
        raise DataReadinessError("broker specialist fit and calibration require both classes")
    if estimator_family == "logistic":
        estimator: Any = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=500,
                        random_state=policy.random_seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif estimator_family == "hist_gradient_boosting":
        estimator = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=150,
                        max_depth=3,
                        l2_regularization=10.0,
                        random_state=policy.random_seed,
                    ),
                ),
            ]
        )
    else:
        raise DataReadinessError(f"unknown broker specialist estimator: {estimator_family}")
    estimator.fit(
        frame.loc[fit_mask, list(feature_columns)].to_numpy(dtype="float32", copy=False),
        frame.loc[fit_mask, "target"].to_numpy(dtype="int8", copy=False),
        model__sample_weight=frame.loc[fit_mask, "sample_weight"].to_numpy(dtype="float64", copy=False),
    )
    raw = estimator.predict_proba(
        frame.loc[calibration_mask, list(feature_columns)].to_numpy(dtype="float32", copy=False)
    )[:, 1]
    calibrator = LogisticRegression(C=1.0, max_iter=300, random_state=policy.random_seed, solver="lbfgs")
    calibrator.fit(
        np.asarray(raw).reshape(-1, 1),
        frame.loc[calibration_mask, "target"].to_numpy(dtype="int8", copy=False),
        sample_weight=frame.loc[calibration_mask, "sample_weight"].to_numpy(dtype="float64", copy=False),
    )
    return _FittedModel(
        estimator=estimator,
        calibrator=calibrator,
        feature_columns=feature_columns,
        train_prevalence=_weighted_mean(frame.loc[fit_mask, "target"], frame.loc[fit_mask, "sample_weight"]),
    )


def _predict(fitted: _FittedModel, frame: pd.DataFrame) -> np.ndarray:
    raw = fitted.estimator.predict_proba(
        frame.loc[:, list(fitted.feature_columns)].to_numpy(dtype="float32", copy=False)
    )[:, 1]
    probability = fitted.calibrator.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]
    if not np.isfinite(probability).all():
        raise DataReadinessError("broker specialist produced non-finite probabilities")
    return np.asarray(probability, dtype="float64")


def _fit_scope_predictions(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    estimator_family: str,
    policy: BrokerSpecialistPolicy,
) -> tuple[dict[str, _ScopePrediction], _FittedModel]:
    if training["label_available_at_utc"].max() >= validation["decision_time_utc"].min():
        raise DataReadinessError("broker specialist labels overlap the evaluation scope")
    holdout_train = ~_unseen_security_mask(training["security_id"].astype(str), policy)
    holdout_validation = _unseen_security_mask(validation["security_id"].astype(str), policy)
    predictions: dict[str, _ScopePrediction] = {}
    primary_fitted: _FittedModel | None = None
    for scope, train, test in (
        ("chronological_validation", training, validation),
        (
            "unseen_security_validation",
            training.loc[holdout_train],
            validation.loc[holdout_validation],
        ),
    ):
        fitted = _fit_model(
            train,
            feature_columns=feature_columns,
            estimator_family=estimator_family,
            policy=policy,
        )
        predictions[scope] = _ScopePrediction(
            frame=test,
            probability=_predict(fitted, test),
            train_prevalence=fitted.train_prevalence,
        )
        if scope == "chronological_validation":
            primary_fitted = fitted
    if primary_fitted is None:
        raise DataReadinessError("broker specialist primary fit is missing")
    return predictions, primary_fitted


def _canonical_scope_metrics(
    predictions: Mapping[str, _ScopePrediction],
    *,
    threshold: float,
    policy: BrokerSpecialistPolicy,
    strategy_contract: StrategyContract,
    swing_training_config: SwingTrainingConfig,
    calendar: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    return {
        scope: _canonical_single_scope_metrics(
            prediction,
            threshold=threshold,
            policy=policy,
            strategy_contract=strategy_contract,
            swing_training_config=swing_training_config,
            calendar=calendar,
        )
        for scope, prediction in predictions.items()
    }


def _canonical_single_scope_metrics(
    prediction: _ScopePrediction,
    *,
    threshold: float,
    policy: BrokerSpecialistPolicy,
    strategy_contract: StrategyContract,
    swing_training_config: SwingTrainingConfig,
    calendar: tuple[str, ...],
) -> dict[str, Any]:
    frame = prediction.frame
    probability = prediction.probability
    metrics = _evaluation_metrics(
        frame,
        probability,
        threshold=threshold,
        config=swing_training_config,
        strategy_contract=strategy_contract,
        session_calendar=calendar,
    )
    target = frame["target"].to_numpy(dtype="int8", copy=False)
    weight = frame["sample_weight"].to_numpy(dtype="float64", copy=False)
    brier = float(np.average(np.square(probability - target), weights=weight))
    reference_brier = float(
        np.average(np.square(prediction.train_prevalence - target), weights=weight)
    )
    scored = frame.copy()
    scored["__probability"] = probability
    selected = select_constrained_swing_portfolio(
        scored.loc[scored["__probability"].ge(threshold)],
        maximum_trades=swing_training_config.maximum_trades_per_decision,
        target_maximum_sector_weight=strategy_contract.swing.target_maximum_sector_weight,
        hard_maximum_sector_weight=strategy_contract.swing.hard_maximum_sector_weight,
        minimum_distinct_sectors=strategy_contract.swing.minimum_distinct_sectors_for_selection,
    )
    metrics.update(
        {
            "unique_announcements": int(frame["analyst_revision_episode_id"].nunique()),
            "selected_unique_announcements": int(
                selected["analyst_revision_episode_id"].nunique()
            ),
            "episode_weighted_roc_auc": float(
                roc_auc_score(target, probability, sample_weight=weight)
            ),
            "episode_weighted_brier_score": brier,
            "episode_weighted_brier_skill_vs_train_prevalence": 1.0
            - brier / reference_brier,
            "episode_weighted_expected_calibration_error": _weighted_ece(
                target, probability, weight
            ),
            "minimum_required_roc_auc": policy.minimum_validation_roc_auc,
            "portfolio_evaluator": "canonical_swing_evaluation_metrics_v5",
        }
    )
    return metrics


def _evaluate_fixed_policy(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    estimator_family: str,
    threshold: float,
    policy: BrokerSpecialistPolicy,
    strategy_contract: StrategyContract,
    swing_training_config: SwingTrainingConfig,
    calendar: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], _FittedModel]:
    predictions, fitted = _fit_scope_predictions(
        training,
        validation,
        feature_columns=feature_columns,
        estimator_family=estimator_family,
        policy=policy,
    )
    return (
        _canonical_scope_metrics(
            predictions,
            threshold=threshold,
            policy=policy,
            strategy_contract=strategy_contract,
            swing_training_config=swing_training_config,
            calendar=calendar,
        ),
        fitted,
    )


def _acceptance_gate(
    scopes: Mapping[str, Mapping[str, Any]],
    policy: BrokerSpecialistPolicy,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for scope in ("chronological_validation", "unseen_security_validation"):
        metrics = scopes[scope]
        if float(metrics["episode_weighted_roc_auc"]) < policy.minimum_validation_roc_auc:
            reasons.append(f"{scope} ROC AUC is below {policy.minimum_validation_roc_auc:.2f}")
        if (
            float(metrics["episode_weighted_brier_skill_vs_train_prevalence"])
            <= policy.minimum_brier_skill
        ):
            reasons.append(f"{scope} Brier skill is not positive")
        if (
            float(metrics["episode_weighted_expected_calibration_error"])
            > policy.maximum_expected_calibration_error
        ):
            reasons.append(
                f"{scope} expected calibration error exceeds "
                f"{policy.maximum_expected_calibration_error:.2f}"
            )
        gate = metrics.get("economic_gate")
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            reasons.append(f"{scope} canonical economic gate failed")
    temporal = scopes["chronological_validation"]
    unseen = scopes["unseen_security_validation"]
    if int(temporal.get("selected_unique_announcements", 0)) < policy.minimum_selected_announcements:
        reasons.append("chronological validation selects too few independent announcements")
    if int(unseen.get("selected_unique_announcements", 0)) < policy.minimum_unseen_selected_announcements:
        reasons.append("unseen-security validation selects too few independent announcements")
    return not reasons, reasons


def _threshold_selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    scopes = record.get("scopes")
    if not isinstance(scopes, Mapping):
        return (-math.inf,)
    temporal = scopes.get("chronological_validation")
    unseen = scopes.get("unseen_security_validation")
    if not isinstance(temporal, Mapping) or not isinstance(unseen, Mapping):
        return (-math.inf,)
    return (
        min(
            float(temporal.get("episode_weighted_roc_auc", -math.inf)),
            float(unseen.get("episode_weighted_roc_auc", -math.inf)),
        ),
        float(temporal.get("selected_average_managed_net_return", -math.inf)),
        -float(record.get("probability_threshold", 1.0)),
    )


def _experiment_selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    scopes = record.get("selected_scopes")
    if not isinstance(scopes, Mapping):
        return (-math.inf,)
    temporal = scopes.get("chronological_validation")
    unseen = scopes.get("unseen_security_validation")
    if not isinstance(temporal, Mapping) or not isinstance(unseen, Mapping):
        return (-math.inf,)
    return (
        min(
            float(temporal["episode_weighted_roc_auc"]),
            float(unseen["episode_weighted_roc_auc"]),
        ),
        min(
            float(temporal["episode_weighted_brier_skill_vs_train_prevalence"]),
            float(unseen["episode_weighted_brier_skill_vs_train_prevalence"]),
        ),
        -float(record["feature_count"]),
    )


def _unseen_security_mask(
    identities: pd.Series,
    policy: BrokerSpecialistPolicy,
) -> pd.Series:
    threshold = int(policy.unseen_security_holdout_fraction * 2**64)
    seed = str(policy.unseen_security_hash_seed)
    return identities.map(
        lambda value: int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:16], 16) < threshold
    ).astype(bool)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    weight = pd.to_numeric(weights, errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or not np.isfinite(weight).all() or weight.sum() <= 0:
        raise DataReadinessError("weighted broker specialist metric is invalid")
    return float(np.average(numeric, weights=weight))


def _weighted_ece(target: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, 9)
    total = float(weight.sum())
    error = 0.0
    for index in range(10):
        mask = bins == index
        if not mask.any():
            continue
        bin_weight = float(weight[mask].sum())
        error += bin_weight / total * abs(
            float(np.average(target[mask], weights=weight[mask]))
            - float(np.average(probability[mask], weights=weight[mask]))
        )
    return error


def _session_calendar(start: date, end: date) -> tuple[str, ...]:
    return tuple(
        value.date().isoformat()
        for value in _XNYS.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    )


def _policy_record(policy: BrokerSpecialistPolicy) -> dict[str, Any]:
    result = asdict(policy)
    for name in (
        "development_start",
        "development_end",
        "model_selection_training_end",
        "model_selection_embargo_start",
        "model_selection_embargo_end",
        "model_selection_validation_start",
        "validation_start",
        "validation_end",
        "locked_test_start",
    ):
        result[name] = result[name].isoformat()
    for name in (
        "specialists",
        "profiles",
        "estimators",
        "report_only_subtypes",
        "probability_thresholds",
    ):
        result[name] = list(result[name])
    return result


def _guard(policy: BrokerSpecialistPolicy, stage: str, *, peak: bool) -> None:
    assert_memory_budget(
        hard_budget_gib=policy.maximum_process_memory_gib,
        headroom_gib=policy.memory_guard_headroom_gib,
        stage=stage,
    )
    if peak:
        assert_peak_memory_budget(
            hard_budget_gib=policy.maximum_process_memory_gib,
            headroom_gib=policy.memory_guard_headroom_gib,
            stage=stage,
        )


def load_swing_broker_specialist_result(
    directory: Path,
    *,
    source_directory: Path,
    policy_path: Path,
    strategy_contract_path: Path,
    swing_training_policy_path: Path,
) -> Mapping[str, Any]:
    """Strictly verify an immutable A3.5 development result."""

    return _verify_output(
        directory,
        source_directory=source_directory,
        policy_path=policy_path,
        strategy_contract_path=strategy_contract_path,
        swing_training_policy_path=swing_training_policy_path,
    )


def _verify_output(
    directory: Path,
    *,
    source_directory: Path,
    policy_path: Path,
    strategy_contract_path: Path,
    swing_training_policy_path: Path,
) -> dict[str, Any]:
    policy = load_broker_specialist_policy(policy_path)
    strategy_contract = load_strategy_contract(strategy_contract_path)
    swing_training_config = load_swing_training_config(swing_training_policy_path)
    source = _verify_source_hashes(
        source_directory,
        policy,
        strategy_contract=strategy_contract,
    )
    request = _read_json(directory / "_request.json", "broker specialist request")
    manifest_path = directory / "_manifest.json"
    authority = _read_json(directory / "_authority.json", "broker specialist authority")
    manifest = _read_json(manifest_path, "broker specialist manifest")
    _assert_output_control_flags(authority, manifest)
    request_identity = dict(request)
    recorded_request_sha256 = request_identity.pop("request_sha256", None)
    if (
        request.get("schema") != REQUEST_SCHEMA
        or recorded_request_sha256 != _json_sha256(request_identity)
        or request.get("source_manifest_sha256")
        != file_sha256(source_directory / "_manifest.json")
        or request.get("source_authority_sha256")
        != file_sha256(source_directory / "_authority.json")
        or request.get("policy") != _policy_record(policy)
        or request.get("policy_file_sha256") != file_sha256(policy_path)
        or request.get("strategy_contract_sha256") != strategy_contract.sha256()
        or request.get("strategy_contract_file_sha256")
        != file_sha256(strategy_contract_path)
        or request.get("swing_training_policy")
        != json.loads(json.dumps(asdict(swing_training_config)))
        or request.get("swing_training_policy_file_sha256")
        != file_sha256(swing_training_policy_path)
        or source.get("schema") != policy.source_artifact_schema
        or manifest.get("schema") != MANIFEST_SCHEMA
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != recorded_request_sha256
        or manifest.get("request_sha256") != recorded_request_sha256
        or manifest.get("source_manifest_sha256")
        != request.get("source_manifest_sha256")
    ):
        raise DataReadinessError("broker specialist output authority is invalid")
    capacity = manifest.get("capacity_audit")
    if not isinstance(capacity, Mapping):
        raise DataReadinessError("broker specialist capacity record is missing")
    path = _canonical_output_member(
        directory,
        capacity.get("path"),
        expected="capacity_audit.parquet",
    )
    if not path.is_file() or file_sha256(path) != capacity.get("sha256"):
        raise DataReadinessError("broker specialist capacity artifact changed")
    observed_capacity = pd.read_parquet(path)
    records = capacity.get("records")
    subtype_rows = _load_subtype_identity_rows(source_directory, source, policy)
    expected_capacity = _capacity_audit(subtype_rows, policy)
    if (
        not isinstance(records, list)
        or len(observed_capacity) != int(capacity.get("rows", -1))
        or observed_capacity.to_dict(orient="records") != records
    ):
        raise DataReadinessError("broker specialist capacity content changed")
    try:
        pd.testing.assert_frame_equal(
            observed_capacity,
            expected_capacity,
            check_exact=True,
        )
    except AssertionError as exc:
        raise DataReadinessError(
            "broker specialist capacity does not replay from source"
        ) from exc
    raw_specialists = manifest.get("specialists")
    if not isinstance(raw_specialists, list) or {
        str(item.get("specialist"))
        for item in raw_specialists
        if isinstance(item, Mapping)
    } != set(policy.specialists):
        raise DataReadinessError("broker specialist result inventory differs")
    replayed_specialists: list[dict[str, Any]] = []
    replayed_models: dict[str, dict[str, Any]] = {}
    for specialist in policy.specialists:
        specialist_capacity = expected_capacity.loc[
            expected_capacity["specialist"].eq(specialist)
        ]
        if not bool(specialist_capacity["capacity_passed"].all()):
            replayed_specialists.append(
                {
                    "specialist": specialist,
                    "status": "insufficient_capacity",
                    "locked_test_outcomes_read": False,
                    "experiments": [],
                }
            )
            continue
        replayed, payload = _run_specialist_experiments(
            source_directory=source_directory,
            source=source,
            specialist=specialist,
            decision_ids=_specialist_decision_ids(subtype_rows, specialist),
            policy=policy,
            strategy_contract=strategy_contract,
            swing_training_config=swing_training_config,
        )
        replayed_specialists.append(replayed)
        if payload is not None:
            replayed_models[specialist] = payload
    if _json_sha256(replayed_specialists) != _json_sha256(raw_specialists):
        raise DataReadinessError("broker specialist experiment evidence does not replay")
    for item in raw_specialists:
        if not isinstance(item, Mapping) or item.get("locked_test_outcomes_read") is not False:
            raise DataReadinessError("broker specialist result opened locked test")
        experiments = item.get("experiments")
        if item.get("status") == "insufficient_capacity":
            if experiments != []:
                raise DataReadinessError("capacity-blocked specialist has experiments")
        elif not isinstance(experiments, list) or len(experiments) != 6:
            raise DataReadinessError("broker specialist experiment matrix differs")
        elif any(
            not isinstance(experiment, Mapping)
            or experiment.get("locked_test_outcomes_read") is not False
            for experiment in experiments
        ):
            raise DataReadinessError("broker specialist experiment opened locked test")
    raw_models = manifest.get("models")
    if not isinstance(raw_models, list):
        raise DataReadinessError("broker specialist model records are invalid")
    expected_files = {
        "_request.json",
        "_manifest.json",
        "_authority.json",
        "capacity_audit.parquet",
    }
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("broker specialist model record is malformed")
        raw_specialist = raw.get("specialist")
        if not isinstance(raw_specialist, str) or raw_specialist not in policy.specialists:
            raise DataReadinessError("broker specialist model identity is invalid")
        expected_relative = f"models/{raw_specialist}/candidate.joblib"
        model = _canonical_output_member(
            directory,
            raw.get("path"),
            expected=expected_relative,
        )
        relative = expected_relative
        expected_files.add(relative)
        if not model.is_file() or file_sha256(model) != raw.get("sha256"):
            raise DataReadinessError("broker specialist model artifact changed")
    if manifest.get("status") == "no_development_candidate" and raw_models:
        raise DataReadinessError("rejected broker specialists cannot emit models")
    price_target = manifest.get("price_target_policy")
    if (
        not isinstance(price_target, Mapping)
        or price_target.get("status") != "report_only"
        or price_target.get("configured_subtypes") != list(policy.report_only_subtypes)
        or int(price_target.get("latest_direction_unverified_announcements", -1))
        != int(
            subtype_rows.loc[
                subtype_rows["subtype"].eq("price_target_or_generic"),
                "analyst_revision_episode_id",
            ].nunique()
        )
    ):
        raise DataReadinessError("broker specialist report-only action audit differs")
    expected_status = (
        "development_candidate" if replayed_models else "no_development_candidate"
    )
    if manifest.get("status") != expected_status or len(raw_models) != len(replayed_models):
        raise DataReadinessError("broker specialist replayed model status differs")
    with tempfile.TemporaryDirectory(
        prefix="broker-specialist-replay-",
        dir=directory.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        for specialist, payload in replayed_models.items():
            expected_model = temporary / specialist / "candidate.joblib"
            expected_model.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, expected_model, compress=3)
            matching = [
                raw
                for raw in raw_models
                if isinstance(raw, Mapping) and raw.get("specialist") == specialist
            ]
            if len(matching) != 1 or matching[0].get("sha256") != file_sha256(
                expected_model
            ):
                raise DataReadinessError("broker specialist model payload does not replay")
    observed_files = {
        str(path.relative_to(directory)).replace("\\", "/")
        for path in directory.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise DataReadinessError("broker specialist output file inventory differs")
    return manifest


def _assert_output_control_flags(
    authority: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if (
        manifest.get("state") != "complete"
        or manifest.get("locked_test_outcomes_read") is not False
        or manifest.get("promotion_permitted") is not False
        or authority.get("artifact") != "_manifest.json"
        or authority.get("locked_test_outcomes_read") is not False
        or authority.get("promotion_permitted") is not False
    ):
        raise DataReadinessError("broker specialist output control flags are invalid")


def _canonical_output_member(
    directory: Path,
    raw_relative: object,
    *,
    expected: str,
) -> Path:
    if raw_relative != expected:
        raise DataReadinessError("broker specialist output path is invalid")
    candidate = (directory / expected).resolve()
    root = directory.resolve()
    if candidate.parent != root and root not in candidate.parents:
        raise DataReadinessError("broker specialist output path escapes artifact")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} is not an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
