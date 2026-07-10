from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_STATUS_CANDIDATE = "candidate"
MODEL_STATUS_PROMOTED = "promoted"
MODEL_STATUS_DEPRECATED = "deprecated"

ALLOWED_MODEL_STATUSES = {MODEL_STATUS_CANDIDATE, MODEL_STATUS_PROMOTED, MODEL_STATUS_DEPRECATED}
PRODUCTION_VALIDATION_SPLIT = "date_grouped_purged_walk_forward"


def feature_schema_hash(features: list[str]) -> str:
    return _json_hash({"features": list(features)})


def dataset_fingerprint(data: pd.DataFrame, *, target_col: str, features: list[str]) -> dict[str, Any]:
    dates = pd.to_datetime(data["date"], errors="coerce") if "date" in data.columns else pd.Series(dtype="datetime64[ns]")
    target = pd.to_numeric(data[target_col], errors="coerce") if target_col in data.columns else pd.Series(dtype="float")
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "tickers": int(data["ticker"].nunique()) if "ticker" in data.columns else 0,
        "first_date": _date_value(dates.min()) if not dates.empty else None,
        "last_date": _date_value(dates.max()) if not dates.empty else None,
        "target_col": target_col,
        "positive_rate": float(target.mean()) if target.notna().any() else None,
        "feature_count": int(len(features)),
        "feature_schema_hash": feature_schema_hash(features),
    }


def write_model_manifest(
    *,
    model_path: Path,
    model_type: str,
    schema_version: str,
    target_col: str,
    features: list[str],
    training_data: pd.DataFrame,
    metrics: dict[str, Any],
    validation_split: str,
    status: str = MODEL_STATUS_CANDIDATE,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in ALLOWED_MODEL_STATUSES:
        raise ValueError(f"Invalid model status: {status}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    manifest = {
        "schema": "model_registry_manifest.v1",
        "status": status,
        "model_type": model_type,
        "schema_version": schema_version,
        "target_col": target_col,
        "artifact_path": str(model_path),
        "artifact_sha256": file_sha256(model_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_split": validation_split,
        "dataset": dataset_fingerprint(training_data, target_col=target_col, features=features),
        "metrics": _json_safe(metrics),
    }
    if extra:
        manifest["extra"] = _json_safe(extra)
    manifest_path = manifest_path_for(model_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_model_manifest(model_path: Path) -> dict[str, Any]:
    path = manifest_path_for(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing model manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def promote_model_manifest(
    *,
    model_path: Path,
    metrics: dict[str, Any],
    alignment_audit: pd.DataFrame | None = None,
    profitability_audit: pd.DataFrame | None = None,
    regime_audit: pd.DataFrame | None = None,
    catalyst_audit: pd.DataFrame | None = None,
    min_roc_auc: float = 0.65,
    min_top_decile_lift: float = 2.0,
    min_validated_rows: int = 20_000,
    min_tickers: int = 200,
    max_alignment_errors: int = 0,
    require_alignment_audit: bool = True,
    min_selected_trades: int = 100,
    min_avg_trade_return: float = 0.0,
    min_profit_factor: float = 1.05,
    max_strategy_drawdown: float = 0.25,
    min_return_drawdown_ratio: float = 0.5,
    max_negative_period_rate: float = 0.55,
    require_profitability_audit: bool = True,
    min_regime_count: int = 3,
    max_single_regime_share: float = 0.85,
    require_regime_audit: bool = True,
    max_low_relevance_event_rate: float = 0.25,
    require_catalyst_audit: bool = True,
    report_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_model_manifest(model_path)
    failures = evaluate_promotion_gates(
        model_path=model_path,
        manifest=manifest,
        metrics=metrics,
        alignment_audit=alignment_audit,
        min_roc_auc=min_roc_auc,
        min_top_decile_lift=min_top_decile_lift,
        min_validated_rows=min_validated_rows,
        min_tickers=min_tickers,
        max_alignment_errors=max_alignment_errors,
        require_alignment_audit=require_alignment_audit,
        profitability_audit=profitability_audit,
        min_selected_trades=min_selected_trades,
        min_avg_trade_return=min_avg_trade_return,
        min_profit_factor=min_profit_factor,
        max_strategy_drawdown=max_strategy_drawdown,
        min_return_drawdown_ratio=min_return_drawdown_ratio,
        max_negative_period_rate=max_negative_period_rate,
        require_profitability_audit=require_profitability_audit,
        regime_audit=regime_audit,
        min_regime_count=min_regime_count,
        max_single_regime_share=max_single_regime_share,
        require_regime_audit=require_regime_audit,
        catalyst_audit=catalyst_audit,
        max_low_relevance_event_rate=max_low_relevance_event_rate,
        require_catalyst_audit=require_catalyst_audit,
    )
    result = {
        "schema": "model_promotion_report.v1",
        "model_path": str(model_path),
        "manifest_path": str(manifest_path_for(model_path)),
        "previous_status": manifest.get("status", "unknown"),
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "min_roc_auc": min_roc_auc,
            "min_top_decile_lift": min_top_decile_lift,
            "min_validated_rows": min_validated_rows,
            "min_tickers": min_tickers,
            "max_alignment_errors": max_alignment_errors,
            "require_alignment_audit": require_alignment_audit,
            "min_selected_trades": min_selected_trades,
            "min_avg_trade_return": min_avg_trade_return,
            "min_profit_factor": min_profit_factor,
            "max_strategy_drawdown": max_strategy_drawdown,
            "min_return_drawdown_ratio": min_return_drawdown_ratio,
            "max_negative_period_rate": max_negative_period_rate,
            "require_profitability_audit": require_profitability_audit,
            "min_regime_count": min_regime_count,
            "max_single_regime_share": max_single_regime_share,
            "require_regime_audit": require_regime_audit,
            "max_low_relevance_event_rate": max_low_relevance_event_rate,
            "require_catalyst_audit": require_catalyst_audit,
        },
        "metrics": _json_safe(metrics),
    }
    if failures:
        _write_report_if_requested(report_path, result)
        return result

    promoted_at = datetime.now(timezone.utc).isoformat()
    history = list(manifest.get("promotion_history", []))
    history.append(
        {
            "status": MODEL_STATUS_PROMOTED,
            "promoted_at_utc": promoted_at,
            "thresholds": result["thresholds"],
            "metrics": _json_safe(metrics),
        }
    )
    manifest["status"] = MODEL_STATUS_PROMOTED
    manifest["promoted_at_utc"] = promoted_at
    manifest["promotion_history"] = history
    manifest_path_for(model_path).write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    result["new_status"] = MODEL_STATUS_PROMOTED
    _write_report_if_requested(report_path, result)
    return result


def evaluate_promotion_gates(
    *,
    model_path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    alignment_audit: pd.DataFrame | None,
    min_roc_auc: float,
    min_top_decile_lift: float,
    min_validated_rows: int,
    min_tickers: int,
    max_alignment_errors: int,
    require_alignment_audit: bool,
    profitability_audit: pd.DataFrame | None = None,
    min_selected_trades: int = 100,
    min_avg_trade_return: float = 0.0,
    min_profit_factor: float = 1.05,
    max_strategy_drawdown: float = 0.25,
    min_return_drawdown_ratio: float = 0.5,
    max_negative_period_rate: float = 0.55,
    require_profitability_audit: bool = True,
    regime_audit: pd.DataFrame | None = None,
    min_regime_count: int = 3,
    max_single_regime_share: float = 0.85,
    require_regime_audit: bool = True,
    catalyst_audit: pd.DataFrame | None = None,
    max_low_relevance_event_rate: float = 0.25,
    require_catalyst_audit: bool = True,
) -> list[str]:
    failures: list[str] = []
    if not model_path.exists():
        failures.append(f"model artifact is missing: {model_path}")
        return failures
    if manifest.get("status") != MODEL_STATUS_CANDIDATE:
        failures.append(f"manifest status must be candidate, found {manifest.get('status', 'unknown')}")
    actual_hash = file_sha256(model_path)
    expected_hash = str(manifest.get("artifact_sha256", ""))
    if expected_hash != actual_hash:
        failures.append("artifact SHA256 does not match manifest")
    validation_split = str(metrics.get("validation_split") or manifest.get("validation_split") or "")
    if validation_split != PRODUCTION_VALIDATION_SPLIT:
        failures.append(f"validation_split must be {PRODUCTION_VALIDATION_SPLIT}, found {validation_split or 'missing'}")
    roc_auc = _float_metric(metrics, "roc_auc")
    if roc_auc is None or roc_auc < min_roc_auc:
        failures.append(f"roc_auc {roc_auc} < {min_roc_auc}")
    top_decile_lift = _float_metric(metrics, "top_decile_lift")
    if top_decile_lift is None or top_decile_lift < min_top_decile_lift:
        failures.append(f"top_decile_lift {top_decile_lift} < {min_top_decile_lift}")
    validated_rows = _int_metric(metrics, "validated_rows")
    if validated_rows is None or validated_rows < min_validated_rows:
        failures.append(f"validated_rows {validated_rows} < {min_validated_rows}")
    tickers = _int_metric(metrics, "tickers")
    if tickers is None or tickers < min_tickers:
        failures.append(f"tickers {tickers} < {min_tickers}")
    failures.extend(_alignment_failures(alignment_audit, max_alignment_errors, require_alignment_audit))
    failures.extend(
        _profitability_failures(
            profitability_audit,
            min_selected_trades=min_selected_trades,
            min_avg_trade_return=min_avg_trade_return,
            min_profit_factor=min_profit_factor,
            max_strategy_drawdown=max_strategy_drawdown,
            min_return_drawdown_ratio=min_return_drawdown_ratio,
            max_negative_period_rate=max_negative_period_rate,
            require_profitability_audit=require_profitability_audit,
        )
    )
    failures.extend(
        _regime_failures(
            regime_audit,
            min_regime_count=min_regime_count,
            max_single_regime_share=max_single_regime_share,
            require_regime_audit=require_regime_audit,
        )
    )
    failures.extend(
        _catalyst_failures(
            catalyst_audit,
            max_low_relevance_event_rate=max_low_relevance_event_rate,
            require_catalyst_audit=require_catalyst_audit,
        )
    )
    return failures


def manifest_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _alignment_failures(
    alignment_audit: pd.DataFrame | None,
    max_alignment_errors: int,
    require_alignment_audit: bool,
) -> list[str]:
    if alignment_audit is None:
        return ["alignment audit is required but was not provided"] if require_alignment_audit else []
    failures: list[str] = []
    error_rows = 0
    if "error" in alignment_audit.columns:
        error_rows = int(alignment_audit["error"].fillna("").astype(str).str.strip().ne("").sum())
    mismatch_columns = [
        "events_without_feature_row",
        "pending_after_latest_feature_date",
        "missing_historical_feature_rows",
        "dates_with_news_count_mismatch",
    ]
    mismatch_total = 0
    for column in mismatch_columns:
        if column in alignment_audit.columns:
            mismatch_total += int(pd.to_numeric(alignment_audit[column], errors="coerce").fillna(0).sum())
    alignment_errors = error_rows + mismatch_total
    if alignment_errors > max_alignment_errors:
        failures.append(f"alignment errors {alignment_errors} > {max_alignment_errors}")
    return failures


def _profitability_failures(
    profitability_audit: pd.DataFrame | None,
    *,
    min_selected_trades: int,
    min_avg_trade_return: float,
    min_profit_factor: float,
    max_strategy_drawdown: float,
    min_return_drawdown_ratio: float,
    max_negative_period_rate: float,
    require_profitability_audit: bool,
) -> list[str]:
    if profitability_audit is None:
        return ["profitability audit is required but was not provided"] if require_profitability_audit else []
    if profitability_audit.empty:
        return ["profitability audit is empty"]
    record = profitability_audit.iloc[0].to_dict()
    failures: list[str] = []
    selected = _int_metric(record, "selected_trades")
    if selected is None or selected < min_selected_trades:
        failures.append(f"selected_trades {selected} < {min_selected_trades}")
    avg_return = _float_metric(record, "avg_trade_return")
    if avg_return is None or avg_return < min_avg_trade_return:
        failures.append(f"avg_trade_return {avg_return} < {min_avg_trade_return}")
    profit_factor = _float_metric(record, "profit_factor")
    if profit_factor is None or profit_factor < min_profit_factor:
        failures.append(f"profit_factor {profit_factor} < {min_profit_factor}")
    drawdown = _float_metric(record, "max_drawdown")
    if drawdown is None or drawdown > max_strategy_drawdown:
        failures.append(f"max_drawdown {drawdown} > {max_strategy_drawdown}")
    ratio = _float_metric(record, "return_drawdown_ratio")
    if ratio is None or ratio < min_return_drawdown_ratio:
        failures.append(f"return_drawdown_ratio {ratio} < {min_return_drawdown_ratio}")
    negative_period_rate = _float_metric(record, "negative_period_rate")
    if negative_period_rate is None or negative_period_rate > max_negative_period_rate:
        failures.append(f"negative_period_rate {negative_period_rate} > {max_negative_period_rate}")
    return failures


def _regime_failures(
    regime_audit: pd.DataFrame | None,
    *,
    min_regime_count: int,
    max_single_regime_share: float,
    require_regime_audit: bool,
) -> list[str]:
    if regime_audit is None:
        return ["market regime audit is required but was not provided"] if require_regime_audit else []
    if regime_audit.empty:
        return ["market regime audit is empty"]
    record = regime_audit.iloc[0].to_dict()
    failures: list[str] = []
    regimes = _int_metric(record, "regimes_present")
    if regimes is None or regimes < min_regime_count:
        failures.append(f"regimes_present {regimes} < {min_regime_count}")
    share = _float_metric(record, "max_single_regime_share")
    if share is None or share > max_single_regime_share:
        failures.append(f"max_single_regime_share {share} > {max_single_regime_share}")
    return failures


def _catalyst_failures(
    catalyst_audit: pd.DataFrame | None,
    *,
    max_low_relevance_event_rate: float,
    require_catalyst_audit: bool,
) -> list[str]:
    if catalyst_audit is None:
        return ["catalyst/news audit is required but was not provided"] if require_catalyst_audit else []
    if catalyst_audit.empty:
        return ["catalyst/news audit is empty"]
    record = catalyst_audit.iloc[0].to_dict()
    failures: list[str] = []
    has_catalysts = bool(record.get("has_catalyst_features"))
    if require_catalyst_audit and not has_catalysts:
        failures.append("catalyst/news features are required but were not found")
    alignment_errors = _int_metric(record, "alignment_error_total")
    if alignment_errors is not None and alignment_errors > 0:
        failures.append(f"catalyst alignment errors {alignment_errors} > 0")
    low_relevance_rate = _float_metric(record, "low_relevance_event_rate")
    if low_relevance_rate is not None and low_relevance_rate > max_low_relevance_event_rate:
        failures.append(f"low_relevance_event_rate {low_relevance_rate} > {max_low_relevance_event_rate}")
    return failures


def _float_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return converted


def _int_metric(metrics: dict[str, Any], key: str) -> int | None:
    value = _float_metric(metrics, key)
    if value is None:
        return None
    return int(value)


def _write_report_if_requested(report_path: Path | None, report: dict[str, Any]) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")


def _date_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, float) and (pd.isna(value) or value in {float("inf"), float("-inf")}):
        return None
    return value
