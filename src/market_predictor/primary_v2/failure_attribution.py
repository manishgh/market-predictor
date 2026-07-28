"""Lineage, orchestration, and publication for V2 failure attribution."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday.specialist_contracts import (
    IntradaySpecialistResearchConfig,
)
from market_predictor.intraday.specialist_experiments import (
    verify_intraday_specialist_training_bundle,
)
from market_predictor.primary_v2 import (
    failure_attribution_contracts,
    failure_attribution_metrics,
)
from market_predictor.primary_v2.contracts import (
    INTRADAY_V2_ID,
    SWING_V2_ID,
    PrimaryV2ResearchConfig,
)
from market_predictor.primary_v2.experiments import (
    _json_sha256,
    _load_complete_run,
    _load_json,
    _load_verified_source,
    primary_v2_implementation_identity,
)
from market_predictor.primary_v2.failure_attribution_contracts import (
    FAILURE_ATTRIBUTION_SCHEMA,
    FailureAttributionConfig,
    FailureAttributionStrategyConfig,
)
from market_predictor.primary_v2.failure_attribution_metrics import (
    build_cohort_evidence,
    build_replicated_viability,
    normalized_category,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.specialist_contracts import (
    SwingSpecialistResearchConfig,
)
from market_predictor.v3.errors import DataReadinessError

FAILURE_ATTRIBUTION_RUN_SCHEMA = "primary_v2.failure_attribution.run.v1"
FAILURE_ATTRIBUTION_AUTHORITY_SCHEMA = (
    "primary_v2.failure_attribution.authority.v1"
)
_VALIDATION_COLUMNS = ("validation_scope", "fold", "market_regime")


def run_primary_v2_failure_attribution(
    *,
    strategy_id: str,
    v2_run_dir: Path,
    source_dir: Path,
    out_dir: Path,
    config: FailureAttributionConfig,
    policy_path: Path,
    primary_v2_config: PrimaryV2ResearchConfig,
    primary_v2_policy_path: Path,
    swing_v1_config: SwingSpecialistResearchConfig,
    swing_v1_policy_path: Path,
    intraday_v1_config: IntradaySpecialistResearchConfig,
    intraday_v1_policy_path: Path,
) -> dict[str, object]:
    """Verify exact evidence, attribute failures, and publish immutable output."""

    strategy = config.strategy(strategy_id)
    run_root = v2_run_dir.resolve()
    run_request = _verify_primary_v2_run(
        run_root,
        strategy_id=strategy_id,
        primary_v2_config=primary_v2_config,
        primary_v2_policy_path=primary_v2_policy_path,
    )
    source, source_identity = _load_attribution_source(
        strategy_id=strategy_id,
        source_dir=source_dir,
        swing_config=swing_v1_config,
        swing_policy_path=swing_v1_policy_path,
        intraday_config=intraday_v1_config,
        intraday_policy_path=intraday_v1_policy_path,
    )
    if source_identity != run_request.get("source"):
        raise DataReadinessError(
            "failure-attribution source identity differs from the V2 run"
        )
    predictions_path = (
        run_root / strategy.baseline_candidate_id / "predictions.parquet"
    )
    validation = _load_validation_membership(
        predictions_path,
        strategy=strategy,
        validation_scopes=config.validation_scopes,
    )
    evidence_rows = _join_validation_source(
        source,
        validation,
        strategy=strategy,
        strategy_id=strategy_id,
        minimum_cost_bps=config.minimum_stamped_round_trip_cost_bps,
    )
    del source, validation
    release_process_memory()
    _assert_memory(config, "failure-attribution evidence join")

    cohort_evidence = build_cohort_evidence(
        evidence_rows,
        strategy_id=strategy_id,
        config=config,
    )
    replicated = build_replicated_viability(
        cohort_evidence,
        strategy_id=strategy_id,
        config=config,
    )
    request: dict[str, object] = {
        "schema": FAILURE_ATTRIBUTION_RUN_SCHEMA,
        "strategy_id": strategy_id,
        "failure_attribution_policy_sha256": config.sha256(),
        "failure_attribution_policy_file_sha256": file_sha256(policy_path),
        "primary_v2_policy_sha256": primary_v2_config.sha256(),
        "primary_v2_policy_file_sha256": file_sha256(primary_v2_policy_path),
        "primary_v2_run_request_sha256": _json_sha256(run_request),
        "primary_v2_run_manifest_sha256": file_sha256(
            run_root / "_manifest.json"
        ),
        "baseline_candidate_id": strategy.baseline_candidate_id,
        "baseline_predictions_sha256": file_sha256(predictions_path),
        "source": source_identity,
        "implementation": failure_attribution_implementation_identity(),
    }
    request_sha256 = _json_sha256(request)
    summary = _summary(
        evidence_rows,
        replicated=replicated,
        strategy_id=strategy_id,
        request_sha256=request_sha256,
        strategy=strategy,
        config=config,
    )
    result = _publish_audit(
        out_dir.resolve(),
        request=request,
        request_sha256=request_sha256,
        summary=summary,
        cohort_evidence=cohort_evidence,
        replicated=replicated,
    )
    del evidence_rows, cohort_evidence, replicated
    release_process_memory()
    _assert_memory(config, "failure-attribution publication")
    return result


def failure_attribution_implementation_identity() -> dict[str, object]:
    files = {
        "contracts": Path(failure_attribution_contracts.__file__).resolve(),
        "metrics": Path(failure_attribution_metrics.__file__).resolve(),
        "orchestration": Path(__file__).resolve(),
    }
    return {
        name: {"path": path.name, "sha256": file_sha256(path)}
        for name, path in sorted(files.items())
    }


def _summary(
    evidence_rows: pd.DataFrame,
    *,
    replicated: pd.DataFrame,
    strategy_id: str,
    request_sha256: str,
    strategy: FailureAttributionStrategyConfig,
    config: FailureAttributionConfig,
) -> dict[str, object]:
    viable = int(replicated["replicated_viable"].sum())
    return {
        "schema": FAILURE_ATTRIBUTION_SCHEMA,
        "strategy_id": strategy_id,
        "request_sha256": request_sha256,
        "validation_rows": len(evidence_rows),
        "validation_sessions": int(
            evidence_rows[strategy.period_column].nunique()
        ),
        "cohorts_evaluated": len(replicated),
        "replicated_viable_cohorts": viable,
        "status": (
            "replicated_viable_cohorts_found"
            if viable
            else "no_replicated_viable_cohorts"
        ),
        "model_promotion_authorized": False,
        "interpretation": (
            "A replicated cohort authorizes only a separately frozen V3 "
            "hypothesis; this audit cannot promote or retain a model."
        ),
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
    }


def _verify_primary_v2_run(
    root: Path,
    *,
    strategy_id: str,
    primary_v2_config: PrimaryV2ResearchConfig,
    primary_v2_policy_path: Path,
) -> dict[str, object]:
    request = _load_json(root / "_request.json")
    request_sha256 = _json_sha256(request)
    _load_complete_run(root, expected_request_sha256=request_sha256)
    if (
        request.get("schema") != "primary_strategy_v2.run.v1"
        or request.get("strategy_id") != strategy_id
        or request.get("policy_sha256") != primary_v2_config.sha256()
        or request.get("policy_file_sha256")
        != file_sha256(primary_v2_policy_path)
        or request.get("implementation")
        != primary_v2_implementation_identity()
    ):
        raise DataReadinessError(
            "primary V2 run differs from the current frozen V2 identity"
        )
    return request


def _load_attribution_source(
    *,
    strategy_id: str,
    source_dir: Path,
    swing_config: SwingSpecialistResearchConfig,
    swing_policy_path: Path,
    intraday_config: IntradaySpecialistResearchConfig,
    intraday_policy_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if strategy_id == SWING_V2_ID:
        return _load_verified_source(
            strategy_id=strategy_id,
            source_dir=source_dir,
            swing_config=swing_config,
            swing_policy_path=swing_policy_path,
            intraday_config=intraday_config,
            intraday_policy_path=intraday_policy_path,
        )
    if strategy_id != INTRADAY_V2_ID:
        raise DataReadinessError(
            f"unknown failure-attribution strategy: {strategy_id}"
        )
    verified = verify_intraday_specialist_training_bundle(
        source_dir,
        config=intraday_config,
        policy_path=intraday_policy_path,
    )
    source_strategy_id = "INTRADAY.VWAP_REVERSION.30M.V1"
    columns = {
        "strategy_id",
        "setup_id",
        "ticker",
        "session_date_et",
        "sector",
        "market_cap_bucket",
        "liquidity_bucket",
        "price_feed",
        "adjustment",
        "decision_time_utc",
        "label_eligible",
        "regime_risk_on",
        "regime_risk_off",
        "atr_pct",
        "path_outcome",
        "path_outcome_bar",
        "target_before_stop_30m",
        "stop_before_target_30m",
        "path_timeout_30m",
        "path_realized_return_gross_30m",
        "path_realized_return_net_30m",
        "path_mfe_30m",
        "path_mae_30m",
        "path_excess_return_30m_vs_spy",
    }
    frames = [
        pd.read_parquet(
            verified.directory / str(record["path"]),
            columns=sorted(columns),
        )
        for record in verified.strategy_files[source_strategy_id]
    ]
    frame = pd.concat(frames, ignore_index=True)
    del frames
    return frame, {
        "type": "verified_intraday_specialist_training_dataset",
        "bundle_manifest_sha256": verified.manifest_sha256,
        "dataset_fingerprint": verified.dataset_fingerprint,
        "strategy_dataset_sha256": verified.strategy_dataset_sha256[
            source_strategy_id
        ],
        "rows": len(frame),
        "policy_file_sha256": file_sha256(intraday_policy_path),
    }


def _load_validation_membership(
    path: Path,
    *,
    strategy: FailureAttributionStrategyConfig,
    validation_scopes: tuple[str, ...],
) -> pd.DataFrame:
    columns = [strategy.row_id_column, *_VALIDATION_COLUMNS]
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty:
        raise DataReadinessError("baseline V2 predictions are empty")
    if frame[strategy.row_id_column].isna().any():
        raise DataReadinessError("baseline V2 predictions contain null row IDs")
    if frame[strategy.row_id_column].astype(str).duplicated().any():
        raise DataReadinessError("baseline V2 validation row IDs are not unique")
    scopes = tuple(sorted(frame["validation_scope"].astype(str).unique()))
    if scopes != tuple(sorted(validation_scopes)):
        raise DataReadinessError(
            "baseline V2 prediction validation scopes differ from policy"
        )
    return frame.rename(
        columns={"market_regime": "validation_market_regime"}
    )


def _join_validation_source(
    source: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    strategy: FailureAttributionStrategyConfig,
    strategy_id: str,
    minimum_cost_bps: float,
) -> pd.DataFrame:
    if source[strategy.row_id_column].isna().any():
        raise DataReadinessError("source contains null validation row IDs")
    if source[strategy.row_id_column].astype(str).duplicated().any():
        raise DataReadinessError("source row IDs are not unique")
    joined = validation.merge(
        source,
        on=strategy.row_id_column,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != len(validation) or not joined["_merge"].eq("both").all():
        raise DataReadinessError(
            "baseline validation rows do not join one-to-one to exact source"
        )
    joined = joined.drop(columns="_merge")
    validation_regime = normalized_category(
        joined["validation_market_regime"]
    )
    if "market_regime" in joined:
        source_regime = normalized_category(joined["market_regime"])
        if not source_regime.eq(validation_regime).all():
            raise DataReadinessError(
                "prediction-time market regime differs from exact source"
            )
    joined["market_regime"] = validation_regime
    joined = joined.drop(columns="validation_market_regime")
    if "strategy_id" in joined and not joined["strategy_id"].eq(
        _source_strategy_id(strategy_id)
    ).all():
        raise DataReadinessError(
            "joined source contains a different strategy identity"
        )
    required_numeric = {
        strategy.gross_return_column,
        strategy.net_return_column,
        strategy.spy_excess_return_column,
        strategy.mfe_column,
        strategy.mae_column,
        strategy.volatility_column,
    }
    for column in sorted(required_numeric):
        values = pd.to_numeric(joined[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise DataReadinessError(
                f"failure-attribution source has non-finite {column}"
            )
        joined[column] = values
    cost = (
        joined[strategy.gross_return_column]
        - joined[strategy.net_return_column]
    )
    if bool(cost.lt(-1e-12).any()):
        raise DataReadinessError("gross return is below net return for some rows")
    if strategy_id == SWING_V2_ID:
        _verify_swing_cost(joined, cost=cost)
    joined["stamped_round_trip_cost_fraction"] = cost.clip(lower=0)
    if float(cost.mean()) * 10_000 + 1e-9 < minimum_cost_bps:
        raise DataReadinessError(
            "validation source average stamped cost is below policy minimum"
        )
    if joined[strategy.period_column].isna().any():
        raise DataReadinessError("validation source contains null session IDs")
    return joined


def _verify_swing_cost(
    rows: pd.DataFrame,
    *,
    cost: pd.Series,
) -> None:
    stamped = pd.to_numeric(
        rows["strategy_execution_cost_fraction"],
        errors="coerce",
    )
    if (
        not np.isfinite(stamped.to_numpy(dtype=float)).all()
        or not np.allclose(
            cost.to_numpy(dtype=float),
            stamped.to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )
    ):
        raise DataReadinessError(
            "swing gross-minus-net does not equal stamped execution cost"
        )


def _publish_audit(
    root: Path,
    *,
    request: dict[str, object],
    request_sha256: str,
    summary: dict[str, object],
    cohort_evidence: pd.DataFrame,
    replicated: pd.DataFrame,
) -> dict[str, object]:
    if root.exists():
        return _load_complete_audit(
            root,
            expected_request_sha256=request_sha256,
        )
    temporary = root.with_name(f".{root.name}.{uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        _write_json(temporary / "_request.json", request)
        cohort_evidence.to_csv(temporary / "cohort_evidence.csv", index=False)
        replicated.to_csv(
            temporary / "replicated_viability.csv",
            index=False,
        )
        _write_json(temporary / "summary.json", summary)
        artifact_names = (
            "_request.json",
            "cohort_evidence.csv",
            "replicated_viability.csv",
            "summary.json",
        )
        manifest = _manifest(
            temporary,
            request=request,
            request_sha256=request_sha256,
            summary=summary,
            artifact_names=artifact_names,
        )
        _write_json(temporary / "_manifest.json", manifest)
        _write_json(
            temporary / "_authority.json",
            {
                "schema": FAILURE_ATTRIBUTION_AUTHORITY_SCHEMA,
                "state": "complete",
                "request_sha256": request_sha256,
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(
                    temporary / "_manifest.json"
                ),
            },
        )
        temporary.replace(root)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _manifest(
    directory: Path,
    *,
    request: dict[str, object],
    request_sha256: str,
    summary: dict[str, object],
    artifact_names: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema": FAILURE_ATTRIBUTION_RUN_SCHEMA,
        "request_sha256": request_sha256,
        "strategy_id": request["strategy_id"],
        "status": summary["status"],
        "artifacts": [
            {
                "path": name,
                "bytes": (directory / name).stat().st_size,
                "sha256": file_sha256(directory / name),
            }
            for name in artifact_names
        ],
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def _load_complete_audit(
    root: Path,
    *,
    expected_request_sha256: str,
) -> dict[str, object]:
    request = _load_json(root / "_request.json")
    manifest = _load_json(root / "_manifest.json")
    authority = _load_json(root / "_authority.json")
    if (
        _json_sha256(request) != expected_request_sha256
        or authority.get("schema") != FAILURE_ATTRIBUTION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != expected_request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(root / "_manifest.json")
        or manifest.get("request_sha256") != expected_request_sha256
    ):
        raise DataReadinessError(
            "failure-attribution output lacks matching complete authority"
        )
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise DataReadinessError(
            "failure-attribution manifest has no artifacts"
        )
    expected_files = {"_authority.json", "_manifest.json"}
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise DataReadinessError(
                "failure-attribution manifest artifact is invalid"
            )
        name = str(raw.get("path", ""))
        path = root / name
        expected_files.add(name)
        if (
            Path(name).name != name
            or not path.is_file()
            or path.stat().st_size != int(cast(Any, raw.get("bytes", -1)))
            or file_sha256(path) != raw.get("sha256")
        ):
            raise DataReadinessError(
                f"failure-attribution artifact does not verify: {path}"
            )
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise DataReadinessError(
            "failure-attribution artifact file set differs from manifest"
        )
    return manifest


def _source_strategy_id(strategy_id: str) -> str:
    if strategy_id == SWING_V2_ID:
        return "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1"
    if strategy_id == INTRADAY_V2_ID:
        return "INTRADAY.VWAP_REVERSION.30M.V1"
    raise DataReadinessError(f"unknown primary V2 strategy: {strategy_id}")


def _assert_memory(
    config: FailureAttributionConfig,
    stage: str,
) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return cast(Any, value).item()
    raise TypeError(
        f"object is not JSON serializable: {type(value).__name__}"
    )
