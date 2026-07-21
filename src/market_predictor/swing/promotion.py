from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from market_predictor.registry import (
    MODEL_STATUS_CANDIDATE,
    MODEL_STATUS_PROMOTED,
    file_sha256,
    manifest_path_for,
    verify_model_artifact,
)
from market_predictor.swing.contracts import (
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
    SWING_VALIDATION_SPLIT,
    SwingPromotionConfig,
)
from market_predictor.v3.errors import DataReadinessError

if TYPE_CHECKING:
    from market_predictor.swing.model import SwingTrainingResult


@dataclass(frozen=True)
class SwingPromotionEvidence:
    metrics: dict[str, Any]
    profitability_audit: pd.DataFrame
    regime_audit: pd.DataFrame
    catalyst_audit: pd.DataFrame
    alignment_audit: pd.DataFrame
    provenance: str
    evidence_manifest: dict[str, Any] | None = None


def promote_swing_model(
    *,
    model_path: Path,
    evidence: SwingPromotionEvidence,
    config: SwingPromotionConfig | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    config = config or SwingPromotionConfig()
    metrics = evidence.metrics
    profitability_audit = evidence.profitability_audit
    regime_audit = evidence.regime_audit
    catalyst_audit = evidence.catalyst_audit
    alignment_audit = evidence.alignment_audit
    manifest = verify_model_artifact(model_path, allowed_statuses={MODEL_STATUS_CANDIDATE})
    failures: list[str] = []
    if manifest.get("model_type") != SWING_MODEL_TYPE:
        failures.append(f"model_type must be {SWING_MODEL_TYPE}")
    if manifest.get("schema_version") != SWING_MODEL_SCHEMA_VERSION:
        failures.append(f"schema_version must be {SWING_MODEL_SCHEMA_VERSION}")
    validation_split = str(metrics.get("validation_split") or manifest.get("validation_split") or "")
    if validation_split != SWING_VALIDATION_SPLIT:
        failures.append(f"validation_split must be {SWING_VALIDATION_SPLIT}")

    model_run_id = str(metrics.get("model_run_id") or "")
    manifest_run_id = str(cast(dict[str, Any], manifest.get("extra") or {}).get("model_run_id") or "")
    if not model_run_id or model_run_id != manifest_run_id:
        failures.append("metrics and manifest model_run_id must match")
    audits = {
        "profitability": profitability_audit,
        "regime": regime_audit,
        "catalyst": catalyst_audit,
        "alignment": alignment_audit,
    }
    for name, audit in audits.items():
        failures.extend(_audit_provenance_failures(name, audit, model_run_id))

    failures.extend(
        _metric_gate_failures(
            metrics,
            {
                "roc_auc": (config.min_roc_auc, "min"),
                "ticker_holdout_roc_auc": (config.min_ticker_holdout_roc_auc, "min"),
                "top_decile_lift": (config.min_top_decile_lift, "min"),
                "ticker_holdout_top_decile_lift": (config.min_ticker_holdout_lift, "min"),
                "validated_rows": (float(config.min_validated_rows), "min"),
                "tickers": (float(config.min_tickers), "min"),
            },
        )
    )
    memory = metrics.get("memory")
    peak_memory = _finite_number(memory.get("peak_working_set_gib")) if isinstance(memory, dict) else None
    if peak_memory is None or peak_memory > config.max_peak_working_set_gib:
        failures.append(
            f"peak_working_set_gib {peak_memory} exceeds or does not prove <= "
            f"{config.max_peak_working_set_gib}"
        )

    conservative = _required_first_row(profitability_audit, "profitability", failures)
    if conservative is not None:
        if str(conservative.get("phase")) != "conservative":
            failures.append("profitability first row must be the conservative aggregate")
        failures.extend(
            _row_gate_failures(
                conservative,
                {
                    "selected_trades": (float(config.min_selected_trades), "min"),
                    "avg_trade_return": (config.min_avg_trade_return, "min"),
                    "avg_excess_return_vs_spy": (config.min_avg_excess_return_vs_spy, "min"),
                    "avg_excess_return_vs_qqq": (config.min_avg_excess_return_vs_qqq, "min"),
                    "avg_excess_return_vs_sector": (config.min_avg_excess_return_vs_sector, "min"),
                    "profit_factor": (config.min_profit_factor, "min"),
                    "max_drawdown": (config.max_drawdown, "max"),
                    "return_drawdown_ratio": (config.min_return_drawdown_ratio, "min"),
                    "negative_period_rate": (config.max_negative_period_rate, "max"),
                },
                prefix="profitability",
            )
        )

    regime = _required_first_row(regime_audit, "regime", failures)
    if regime is not None:
        if str(regime.get("scope")) != "summary":
            failures.append("regime first row must be the summary")
        failures.extend(
            _row_gate_failures(
                regime,
                {
                    "regimes_present": (float(config.min_regimes), "min"),
                    "max_single_regime_share": (config.max_single_regime_share, "max"),
                },
                prefix="regime",
            )
        )

    catalyst = _required_first_row(catalyst_audit, "catalyst", failures)
    if catalyst is not None:
        if not _strict_bool(catalyst.get("has_catalyst_features")):
            failures.append("catalyst features were not present in validation")
        failures.extend(
            _row_gate_failures(
                catalyst,
                {
                    "catalyst_row_rate": (config.min_catalyst_row_rate, "min"),
                    "low_relevance_event_rate": (config.max_low_relevance_event_rate, "max"),
                },
                prefix="catalyst",
            )
        )

    alignment = _required_first_row(alignment_audit, "alignment", failures)
    if alignment is not None:
        alignment_columns = (
            "alignment_error_total",
            "events_without_feature_row",
            "missing_historical_feature_rows",
            "dates_with_news_count_mismatch",
            "future_feature_rows",
            "label_path_mismatches",
        )
        alignment_total = 0.0
        for column in alignment_columns:
            value = _finite_number(alignment.get(column))
            if value is None or value < 0:
                failures.append(f"alignment.{column} is missing or invalid")
            else:
                alignment_total += value
        if alignment_total > config.max_alignment_errors:
            failures.append(f"alignment errors {alignment_total} > {config.max_alignment_errors}")

    requested_at = datetime.now(UTC).isoformat()
    effective_report_path = report_path or model_path.with_suffix(model_path.suffix + ".promotion.json")
    report: dict[str, Any] = {
        "schema": "swing_model_promotion_report.v1",
        "model_path": str(model_path),
        "manifest_path": str(manifest_path_for(model_path)),
        "model_run_id": model_run_id,
        "evidence_provenance": evidence.provenance,
        "requested_at_utc": requested_at,
        "previous_status": manifest.get("status"),
        "passed": not failures,
        "failures": failures,
        "thresholds": config.model_dump(),
        "metrics": metrics,
    }
    if failures:
        _write_json_atomic(effective_report_path, report)
        return report

    promoted_at = datetime.now(UTC).isoformat()
    history = list(manifest.get("promotion_history") or [])
    history.append(
        {
            "status": MODEL_STATUS_PROMOTED,
            "model_run_id": model_run_id,
            "promoted_at_utc": promoted_at,
            "thresholds": config.model_dump(),
        }
    )
    manifest["status"] = MODEL_STATUS_PROMOTED
    manifest["promoted_at_utc"] = promoted_at
    manifest["promotion_history"] = history
    _write_json_atomic(manifest_path_for(model_path), manifest)
    report["new_status"] = MODEL_STATUS_PROMOTED
    _write_json_atomic(effective_report_path, report)
    return report


def write_swing_training_evidence(
    result: SwingTrainingResult,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    if not overwrite and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"swing evidence directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out_dir / "metrics.json",
        "metrics_csv": out_dir / "metrics.csv",
        "oof_predictions": out_dir / "walk_forward_predictions.parquet",
        "ticker_holdout_predictions": out_dir / "ticker_holdout_predictions.parquet",
        "profitability": out_dir / "profitability.csv",
        "regime": out_dir / "regime.csv",
        "catalyst": out_dir / "catalyst.csv",
        "alignment": out_dir / "alignment.csv",
        "folds": out_dir / "folds.csv",
    }
    _write_json_atomic(paths["metrics"], result.metrics)
    pd.DataFrame([result.metrics]).to_csv(paths["metrics_csv"], index=False)
    result.oof_predictions.to_parquet(paths["oof_predictions"], index=False)
    result.ticker_holdout_predictions.to_parquet(paths["ticker_holdout_predictions"], index=False)
    result.profitability_audit.to_csv(paths["profitability"], index=False)
    result.regime_audit.to_csv(paths["regime"], index=False)
    result.catalyst_audit.to_csv(paths["catalyst"], index=False)
    result.alignment_audit.to_csv(paths["alignment"], index=False)
    result.fold_audit.to_csv(paths["folds"], index=False)
    evidence_manifest = {
        "schema": "swing_training_evidence.v1",
        "model_run_id": result.metrics["model_run_id"],
        "model_artifact_sha256": result.manifest["artifact_sha256"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": {name: {"path": path.name, "sha256": file_sha256(path)} for name, path in paths.items()},
    }
    paths["manifest"] = out_dir / "evidence.manifest.json"
    _write_json_atomic(paths["manifest"], evidence_manifest)
    return paths


def promotion_evidence_from_result(result: SwingTrainingResult) -> SwingPromotionEvidence:
    return SwingPromotionEvidence(
        metrics=result.metrics,
        profitability_audit=result.profitability_audit,
        regime_audit=result.regime_audit,
        catalyst_audit=result.catalyst_audit,
        alignment_audit=result.alignment_audit,
        provenance="in_memory_training_result",
    )


def load_swing_training_evidence(evidence_dir: Path, model_path: Path) -> SwingPromotionEvidence:
    evidence_manifest_path = evidence_dir / "evidence.manifest.json"
    if not evidence_manifest_path.exists():
        raise DataReadinessError(f"swing evidence manifest is missing: {evidence_manifest_path}")
    try:
        loaded = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataReadinessError(f"swing evidence manifest is invalid: {evidence_manifest_path}") from exc
    if not isinstance(loaded, dict) or loaded.get("schema") != "swing_training_evidence.v1":
        raise DataReadinessError(f"unsupported swing evidence manifest: {evidence_manifest_path}")
    manifest = {str(key): value for key, value in loaded.items()}
    model_manifest = verify_model_artifact(model_path, allowed_statuses={MODEL_STATUS_CANDIDATE})
    if manifest.get("model_artifact_sha256") != model_manifest.get("artifact_sha256"):
        raise DataReadinessError("swing evidence does not belong to the candidate model")
    candidate_run_id = str(cast(dict[str, Any], model_manifest.get("extra") or {}).get("model_run_id") or "")
    if not candidate_run_id or manifest.get("model_run_id") != candidate_run_id:
        raise DataReadinessError("swing evidence run does not match the candidate model")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DataReadinessError("swing evidence manifest has no file inventory")
    required = {"metrics", "profitability", "regime", "catalyst", "alignment"}
    if missing := sorted(required.difference(files)):
        raise DataReadinessError(f"swing evidence manifest is missing files: {missing}")

    verified: dict[str, Path] = {}
    root = evidence_dir.resolve()
    for name, record in files.items():
        if not isinstance(record, dict):
            raise DataReadinessError(f"invalid swing evidence record: {name}")
        path = (evidence_dir / str(record.get("path") or "")).resolve()
        if path.parent != root or not path.is_file():
            raise DataReadinessError(f"swing evidence file is missing or outside its bundle: {name}")
        if file_sha256(path) != str(record.get("sha256") or ""):
            raise DataReadinessError(f"swing evidence integrity check failed: {name}")
        verified[str(name)] = path

    metrics_loaded = json.loads(verified["metrics"].read_text(encoding="utf-8"))
    if not isinstance(metrics_loaded, dict):
        raise DataReadinessError("swing metrics evidence must contain an object")
    metrics = {str(key): value for key, value in metrics_loaded.items()}
    if metrics.get("model_run_id") != candidate_run_id:
        raise DataReadinessError("swing metrics run does not match the candidate model")
    return SwingPromotionEvidence(
        metrics=metrics,
        profitability_audit=pd.read_csv(verified["profitability"]),
        regime_audit=pd.read_csv(verified["regime"]),
        catalyst_audit=pd.read_csv(verified["catalyst"]),
        alignment_audit=pd.read_csv(verified["alignment"]),
        provenance="hash_verified_evidence_bundle",
        evidence_manifest=manifest,
    )


def _audit_provenance_failures(name: str, audit: pd.DataFrame, model_run_id: str) -> list[str]:
    if audit.empty:
        return [f"{name} audit is empty"]
    if "model_run_id" not in audit.columns:
        return [f"{name} audit is missing model_run_id"]
    run_ids = set(audit["model_run_id"].dropna().astype(str).unique())
    if run_ids != {model_run_id}:
        return [f"{name} audit model_run_id does not match the candidate"]
    return []


def _metric_gate_failures(metrics: dict[str, Any], gates: dict[str, tuple[float, str]]) -> list[str]:
    return _value_gate_failures(metrics, gates, prefix="metrics")


def _row_gate_failures(
    row: pd.Series,
    gates: dict[str, tuple[float, str]],
    *,
    prefix: str,
) -> list[str]:
    return _value_gate_failures(row.to_dict(), gates, prefix=prefix)


def _value_gate_failures(
    values: dict[str, Any],
    gates: dict[str, tuple[float, str]],
    *,
    prefix: str,
) -> list[str]:
    failures: list[str] = []
    for name, (threshold, direction) in gates.items():
        value = _finite_number(values.get(name))
        failed = value is None or (direction == "min" and value < threshold) or (
            direction == "max" and value > threshold
        )
        if failed:
            operator = ">=" if direction == "min" else "<="
            failures.append(f"{prefix}.{name} {value} does not satisfy {operator} {threshold}")
    return failures


def _required_first_row(
    audit: pd.DataFrame,
    name: str,
    failures: list[str],
) -> pd.Series | None:
    if audit.empty:
        if f"{name} audit is empty" not in failures:
            failures.append(f"{name} audit is empty")
        return None
    return audit.iloc[0]


def _finite_number(value: object) -> float | None:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_ready(value: object) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value
