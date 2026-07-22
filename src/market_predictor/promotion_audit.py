from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_predictor.market_regime import add_market_regime_labels

CATALYST_COLUMNS = [
    "news_count",
    "news_count_2h",
    "news_count_1d",
    "event_count",
    "catalyst_attention_score_2h",
    "market_context_news_count",
    "market_context_news_count_2h",
    "market_context_news_count_1d",
    "market_context_intraday_shock_score_2h",
    "source_count_alpaca",
    "source_count_alpaca_1d",
    "source_count_reddit",
    "source_count_reddit_1d",
    "source_count_seeking_alpha",
    "source_count_seeking_alpha_1d",
    "source_count_sec",
    "source_count_sec_1d",
    "source_count_finviz",
    "source_count_finviz_1d",
]


@dataclass(frozen=True)
class ProfitabilityAuditConfig:
    probability_col: str = "oos_probability"
    top_fraction: float = 0.10
    min_probability: float | None = None
    max_trades_per_period: int | None = None


def build_walk_forward_profitability_audit(
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    target_col: str | None = None,
    config: ProfitabilityAuditConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate whether out-of-sample high-score rows had tradable economics."""
    cfg = config or ProfitabilityAuditConfig()
    joined = join_predictions_to_dataset(dataset=dataset, predictions=predictions, target_col=target_col)
    target = target_col or _infer_target_col(joined)
    probability_col = cfg.probability_col
    if probability_col not in joined.columns:
        raise ValueError(f"Predictions missing probability column: {probability_col}")
    return_col = _infer_return_col(joined, target)
    if return_col is None:
        raise ValueError("Could not find a realized return column for profitability audit.")
    scored = joined.dropna(subset=[probability_col, return_col]).copy()
    if scored.empty:
        return _empty_profit_summary(), scored, _empty_regime_profit_frame()

    scored = add_market_regime_labels(scored)
    probability = pd.to_numeric(scored[probability_col], errors="coerce")
    cutoff = float(probability.quantile(max(0.0, min(1.0, 1.0 - cfg.top_fraction))))
    if cfg.min_probability is not None:
        cutoff = max(cutoff, float(cfg.min_probability))
    trades = _select_trades(
        scored,
        probability_col=probability_col,
        cutoff=cutoff,
        max_trades_per_period=cfg.max_trades_per_period,
    )
    trades["selected_probability_cutoff"] = cutoff
    summary = _profit_summary(scored, trades, target_col=target, return_col=return_col, probability_col=probability_col)
    regime = _regime_profit_summary(trades, return_col=return_col, target_col=target)
    return summary, trades.reset_index(drop=True), regime


def build_market_regime_audit(
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    probability_col: str = "oos_probability",
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    frame = add_market_regime_labels(dataset)
    if frame.empty:
        return pd.DataFrame([_regime_audit_record(frame, pd.DataFrame())])
    selected = pd.DataFrame()
    if predictions is not None and not predictions.empty and probability_col in predictions.columns:
        joined = join_predictions_to_dataset(dataset=frame, predictions=predictions)
        joined = add_market_regime_labels(joined)
        probability = pd.to_numeric(joined[probability_col], errors="coerce")
        cutoff = float(probability.quantile(max(0.0, min(1.0, 1.0 - top_fraction)))) if probability.notna().any() else np.nan
        selected = joined[probability.ge(cutoff)].copy() if np.isfinite(cutoff) else pd.DataFrame()
    return pd.DataFrame([_regime_audit_record(frame, selected)])


def build_catalyst_news_audit(
    *,
    dataset: pd.DataFrame,
    alignment_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = dataset.copy()
    record: dict[str, Any] = {
        "rows": int(len(frame)),
        "tickers": int(frame["ticker"].nunique()) if "ticker" in frame.columns else 0,
        "catalyst_columns_present": int(sum(1 for col in CATALYST_COLUMNS if col in frame.columns)),
        "has_catalyst_features": bool(any(col in frame.columns for col in CATALYST_COLUMNS)),
        "rows_with_news": _positive_count(frame, "news_count"),
        "rows_with_events": _positive_count(frame, "event_count"),
        "rows_with_market_context": _positive_count(frame, "market_context_news_count"),
        "alpaca_rows": _positive_count(frame, "source_count_alpaca"),
        "reddit_rows": _positive_count(frame, "source_count_reddit"),
        "seeking_alpha_rows": _positive_count(frame, "source_count_seeking_alpha"),
        "sec_rows": _positive_count(frame, "source_count_sec"),
        "finviz_rows": _positive_count(frame, "source_count_finviz"),
        "alignment_error_total": _alignment_error_total(alignment_audit),
        "alignment_audit_rows": int(len(alignment_audit)) if alignment_audit is not None else 0,
    }
    rows = max(1, int(record["rows"]))
    record["news_row_rate"] = float(record["rows_with_news"] / rows)
    record["event_row_rate"] = float(record["rows_with_events"] / rows)
    if "event_relevance_score" in frame.columns:
        relevance = pd.to_numeric(frame["event_relevance_score"], errors="coerce")
        event_rows = relevance.notna()
        low_relevance = relevance.lt(0.5) & event_rows
        record["event_relevance_rows"] = int(event_rows.sum())
        record["low_relevance_event_rows"] = int(low_relevance.sum())
        record["low_relevance_event_rate"] = float(low_relevance.sum() / max(1, event_rows.sum()))
    else:
        record["event_relevance_rows"] = 0
        record["low_relevance_event_rows"] = 0
        record["low_relevance_event_rate"] = 0.0
    if "generic_movers_headline" in frame.columns:
        generic = pd.to_numeric(frame["generic_movers_headline"], errors="coerce").fillna(0)
        record["generic_movers_rows"] = int(generic.gt(0).sum())
        record["generic_movers_rate"] = float(generic.gt(0).sum() / rows)
    else:
        record["generic_movers_rows"] = 0
        record["generic_movers_rate"] = 0.0
    return pd.DataFrame([record])


def join_predictions_to_dataset(
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    target_col: str | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    if "ticker" not in predictions.columns or "date" not in predictions.columns:
        raise ValueError("Predictions must contain ticker and date columns.")
    target = target_col or _infer_target_col(predictions) or _infer_target_col(dataset)
    pred = predictions.copy()
    data = dataset.copy()
    pred["_audit_ticker"] = pred["ticker"].astype(str).str.upper().str.strip()
    data["_audit_ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    pred["_audit_date"] = _normalized_timestamp(pred["date"])
    data["_audit_date"] = _normalized_timestamp(data["date"])
    data = data.dropna(subset=["_audit_ticker", "_audit_date"])
    pred = pred.dropna(subset=["_audit_ticker", "_audit_date"])
    existing = set(pred.columns)
    keep = _audit_join_columns(data, target=target, existing_prediction_columns=existing)
    keep.extend(["_audit_ticker", "_audit_date"])
    keep = list(dict.fromkeys(keep))
    joined = pred.merge(
        data[keep].drop_duplicates(["_audit_ticker", "_audit_date"]),
        on=["_audit_ticker", "_audit_date"],
        how="left",
        suffixes=("", "_dataset"),
    )
    for base in ["ticker", "date"]:
        dataset_col = f"{base}_dataset"
        if dataset_col in joined.columns:
            joined[base] = joined[base].where(joined[base].notna(), joined[dataset_col])
            joined = joined.drop(columns=[dataset_col])
    return joined.drop(columns=["_audit_ticker", "_audit_date"], errors="ignore")


def _audit_join_columns(
    data: pd.DataFrame,
    *,
    target: str | None,
    existing_prediction_columns: set[str],
) -> list[str]:
    columns = ["ticker", "date", "timestamp", "_mp_session_date"]
    if target and target not in existing_prediction_columns:
        columns.append(target)
    return_col = _infer_return_col(data, target)
    if return_col:
        columns.append(return_col)
    suffix = target.rsplit("_", 1)[-1] if target and "_" in target else None
    if suffix:
        columns.extend(
            [
                f"entry_exit_outcome_{suffix}",
                f"bars_to_exit_{suffix}",
                f"max_favorable_excursion_{suffix}",
                f"max_adverse_excursion_{suffix}",
                f"target_exit_risk_{suffix}",
                f"target_timeout_positive_{suffix}",
            ]
        )
    columns.extend(
        [
            "market_regime",
            "qqq_return_1bar",
            "qqq_return_3bar",
            "qqq_return_6bar",
            "spy_return_1bar",
            "spy_return_3bar",
            "spy_return_6bar",
            "volume_z20",
            "news_count",
            "news_count_2h",
            "news_count_1d",
            "event_count",
            "source_count_alpaca",
            "source_count_seeking_alpha",
            "source_count_reddit",
            "source_count_sec",
            "source_count_finviz",
            "market_context_news_count",
            "market_context_news_count_2h",
            "market_context_intraday_shock_score_2h",
            "catalyst_attention_score_2h",
        ]
    )
    return [column for column in dict.fromkeys(columns) if column in data.columns]


def read_audit_record(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = pd.read_json(path)
        return payload if isinstance(payload, pd.DataFrame) else pd.DataFrame([payload])
    return pd.read_csv(path)


def _profit_summary(
    scored: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    target_col: str | None,
    return_col: str,
    probability_col: str,
) -> pd.DataFrame:
    returns = pd.to_numeric(trades[return_col], errors="coerce").dropna()
    gross_gain = float(returns[returns > 0].sum()) if not returns.empty else 0.0
    gross_loss = float(abs(returns[returns < 0].sum())) if not returns.empty else 0.0
    period_returns = _period_returns(trades, return_col=return_col)
    equity = period_returns.cumsum()
    drawdown = equity - equity.cummax() if not equity.empty else pd.Series(dtype="float")
    max_drawdown = float(abs(drawdown.min())) if not drawdown.empty else np.nan
    cumulative_period_return = float(period_returns.sum()) if not period_returns.empty else np.nan
    record = {
        "scored_rows": int(len(scored)),
        "selected_trades": int(len(trades)),
        "selected_fraction": float(len(trades) / max(1, len(scored))),
        "probability_col": probability_col,
        "target_col": target_col,
        "return_col": return_col,
        "probability_cutoff": float(trades["selected_probability_cutoff"].iloc[0]) if not trades.empty else np.nan,
        "avg_trade_return": float(returns.mean()) if not returns.empty else np.nan,
        "median_trade_return": float(returns.median()) if not returns.empty else np.nan,
        "win_rate": float(returns.gt(0).mean()) if not returns.empty else np.nan,
        "profit_factor": float(gross_gain / gross_loss) if gross_loss > 0 else float("inf") if gross_gain > 0 else np.nan,
        "gross_gain": gross_gain,
        "gross_loss": gross_loss,
        "avg_period_return": float(period_returns.mean()) if not period_returns.empty else np.nan,
        "worst_period_return": float(period_returns.min()) if not period_returns.empty else np.nan,
        "negative_period_rate": float(period_returns.lt(0).mean()) if not period_returns.empty else np.nan,
        "cumulative_period_return": cumulative_period_return,
        "max_drawdown": max_drawdown,
        "return_drawdown_ratio": float(cumulative_period_return / max_drawdown) if max_drawdown and max_drawdown > 0 else np.nan,
        "periods": int(len(equity)),
    }
    if target_col and target_col in trades.columns:
        record["target_hit_rate"] = float(pd.to_numeric(trades[target_col], errors="coerce").mean())
    exit_risk = _matching_col(trades, "target_exit_risk_", target_col)
    if exit_risk:
        record["stop_first_rate"] = float(pd.to_numeric(trades[exit_risk], errors="coerce").mean())
    outcome_col = _matching_col(trades, "entry_exit_outcome_", target_col)
    if outcome_col:
        outcomes = trades[outcome_col].fillna("").astype(str)
        record["target_first_rate"] = float(outcomes.eq("target_first").mean())
        record["timeout_rate"] = float(outcomes.eq("timeout").mean())
    return pd.DataFrame([record])


def _select_trades(
    scored: pd.DataFrame,
    *,
    probability_col: str,
    cutoff: float,
    max_trades_per_period: int | None,
) -> pd.DataFrame:
    candidates = scored[pd.to_numeric(scored[probability_col], errors="coerce").ge(cutoff)].copy()
    if candidates.empty or max_trades_per_period is None or max_trades_per_period <= 0:
        return candidates
    candidates["_selection_period"] = _selection_period(candidates)
    selected = (
        candidates.sort_values(["_selection_period", probability_col], ascending=[True, False])
        .groupby("_selection_period", group_keys=False)
        .head(max_trades_per_period)
        .drop(columns=["_selection_period"], errors="ignore")
    )
    return selected


def _regime_profit_summary(trades: pd.DataFrame, *, return_col: str, target_col: str | None) -> pd.DataFrame:
    if trades.empty or "market_regime" not in trades.columns:
        return _empty_regime_profit_frame()
    rows = []
    for regime, group in trades.groupby("market_regime", dropna=False):
        returns = pd.to_numeric(group[return_col], errors="coerce").dropna()
        row = {
            "market_regime": str(regime),
            "selected_trades": int(len(group)),
            "avg_trade_return": float(returns.mean()) if not returns.empty else np.nan,
            "win_rate": float(returns.gt(0).mean()) if not returns.empty else np.nan,
        }
        if target_col and target_col in group.columns:
            row["target_hit_rate"] = float(pd.to_numeric(group[target_col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("selected_trades", ascending=False).reset_index(drop=True)


def _regime_audit_record(frame: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "market_regime" not in frame.columns:
        return {
            "rows": int(len(frame)),
            "regimes_present": 0,
            "max_single_regime_share": 0.0,
            "selected_trades": int(len(selected)),
            "selected_regimes_present": 0,
        }
    counts = frame["market_regime"].fillna("unknown").value_counts()
    selected_counts = (
        selected["market_regime"].fillna("unknown").value_counts()
        if "market_regime" in selected.columns
        else pd.Series(dtype="int")
    )
    return {
        "rows": int(len(frame)),
        "regimes_present": int(counts.size),
        "risk_on_rows": int(counts.get("risk_on", 0)),
        "neutral_rows": int(counts.get("neutral", 0)),
        "risk_off_rows": int(counts.get("risk_off", 0)),
        "max_single_regime_share": float(counts.max() / max(1, counts.sum())),
        "high_volatility_rows": int(pd.to_numeric(frame.get("market_regime_high_volatility", 0), errors="coerce").fillna(0).gt(0).sum()),
        "selected_trades": int(len(selected)),
        "selected_regimes_present": int(selected_counts.size),
        "selected_risk_on_trades": int(selected_counts.get("risk_on", 0)),
        "selected_neutral_trades": int(selected_counts.get("neutral", 0)),
        "selected_risk_off_trades": int(selected_counts.get("risk_off", 0)),
    }


def _period_returns(trades: pd.DataFrame, *, return_col: str) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype="float")
    frame = trades.copy()
    frame["_audit_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[return_col] = pd.to_numeric(frame[return_col], errors="coerce")
    return frame.dropna(subset=["_audit_date", return_col]).groupby("_audit_date")[return_col].mean().sort_index()


def _selection_period(frame: pd.DataFrame) -> pd.Series:
    if "_mp_session_date" in frame.columns:
        return pd.Series(frame["_mp_session_date"], index=frame.index).astype(str)
    timestamp = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    return timestamp.dt.tz_convert("America/New_York").dt.date.astype(str)


def _infer_target_col(frame: pd.DataFrame) -> str | None:
    candidates = [
        str(col)
        for col in frame.columns
        if str(col).startswith("target_entry_success_") and not str(col).endswith("_dataset")
    ]
    if candidates:
        return sorted(candidates)[-1]
    candidates = [str(col) for col in frame.columns if str(col).startswith("target_") and frame[col].notna().any()]
    return sorted(candidates)[-1] if candidates else None


def _infer_return_col(frame: pd.DataFrame, target_col: str | None) -> str | None:
    if target_col:
        suffix = target_col.rsplit("_", 1)[-1]
        for candidate in [
            f"net_realized_return_from_entry_{suffix}",
            f"realized_return_from_entry_{suffix}",
            f"net_horizon_return_from_entry_{suffix}",
            f"horizon_return_from_entry_{suffix}",
        ]:
            if candidate in frame.columns:
                return candidate
    for prefix in [
        "net_realized_return_from_entry_",
        "realized_return_from_entry_",
        "net_horizon_return_from_entry_",
        "horizon_return_from_entry_",
    ]:
        candidates = [str(col) for col in frame.columns if str(col).startswith(prefix)]
        if candidates:
            return sorted(candidates)[-1]
    return None


def _matching_col(frame: pd.DataFrame, prefix: str, target_col: str | None) -> str | None:
    if target_col:
        suffix = target_col.rsplit("_", 1)[-1]
        candidate = f"{prefix}{suffix}"
        if candidate in frame.columns:
            return candidate
    candidates = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(candidates)[-1] if candidates else None


def _normalized_timestamp(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce", utc=True)
    return converted.dt.tz_convert(None)


def _positive_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).gt(0).sum())


def _alignment_error_total(alignment_audit: pd.DataFrame | None) -> int:
    if alignment_audit is None or alignment_audit.empty:
        return 0
    total = 0
    if "error" in alignment_audit.columns:
        total += int(alignment_audit["error"].fillna("").astype(str).str.strip().ne("").sum())
    for column in [
        "events_without_feature_row",
        "pending_after_latest_feature_date",
        "missing_historical_feature_rows",
        "dates_with_news_count_mismatch",
    ]:
        if column in alignment_audit.columns:
            total += int(pd.to_numeric(alignment_audit[column], errors="coerce").fillna(0).sum())
    return total


def _empty_profit_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scored_rows": 0,
                "selected_trades": 0,
                "avg_trade_return": np.nan,
                "win_rate": np.nan,
                "profit_factor": np.nan,
                "max_drawdown": np.nan,
            }
        ]
    )


def _empty_regime_profit_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["market_regime", "selected_trades", "avg_trade_return", "win_rate", "target_hit_rate"])
