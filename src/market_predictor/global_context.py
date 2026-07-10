from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.registry import MODEL_STATUS_PROMOTED, load_model_manifest
from market_predictor.volatile import score_volatile_frame


@dataclass(frozen=True)
class FlashpointRule:
    name: str
    family: str
    commodity_channel: str
    keywords: tuple[str, ...]
    escalation_keywords: tuple[str, ...]
    positive_themes: tuple[str, ...]
    negative_themes: tuple[str, ...]


DEFAULT_FLASHPOINT_RULES: tuple[FlashpointRule, ...] = (
    FlashpointRule(
        name="oil_chokepoint_middle_east",
        family="oil_chokepoint",
        commodity_channel="oil",
        keywords=(
            "strait of hormuz",
            "hormuz",
            "persian gulf",
            "red sea",
            "suez canal",
            "bab el-mandeb",
            "tanker",
            "oil shipment",
            "shipping lane",
        ),
        escalation_keywords=("blockade", "attack", "missile", "mine", "seizure", "closure", "disruption", "strike"),
        positive_themes=("energy_oil_gas", "defense_aerospace"),
        negative_themes=("airlines_travel", "consumer_discretionary", "high_beta_growth"),
    ),
    FlashpointRule(
        name="taiwan_semiconductor_escalation",
        family="semiconductor_supply_chain",
        commodity_channel="semiconductors",
        keywords=("taiwan", "taiwan strait", "tsmc", "china drills", "pla", "south china sea", "export controls"),
        escalation_keywords=("invasion", "blockade", "sanction", "military drill", "missile", "export ban", "restriction"),
        positive_themes=("defense_aerospace", "cybersecurity"),
        negative_themes=("semis_ai_hardware", "ai_data_centers", "high_beta_growth"),
    ),
    FlashpointRule(
        name="russia_ukraine_energy_wheat",
        family="war_energy_agriculture",
        commodity_channel="energy_wheat",
        keywords=("russia", "ukraine", "black sea", "grain corridor", "nato", "pipeline", "lng"),
        escalation_keywords=("attack", "sanction", "missile", "pipeline", "embargo", "mobilization", "drone"),
        positive_themes=("energy_oil_gas", "defense_aerospace", "agriculture_inputs"),
        negative_themes=("europe_exposed", "consumer_discretionary", "high_beta_growth"),
    ),
    FlashpointRule(
        name="rare_earth_export_controls",
        family="critical_minerals",
        commodity_channel="rare_earths",
        keywords=("rare earth", "gallium", "germanium", "lithium", "cobalt", "graphite", "critical minerals"),
        escalation_keywords=("export control", "ban", "restriction", "quota", "sanction", "tariff"),
        positive_themes=("materials_miners", "defense_aerospace"),
        negative_themes=("semis_ai_hardware", "ev_battery_supply_chain", "ai_data_centers"),
    ),
    FlashpointRule(
        name="cyber_infrastructure_attack",
        family="cyberattack",
        commodity_channel="security_risk",
        keywords=("cyberattack", "ransomware", "data breach", "critical infrastructure", "power grid", "pipeline hack"),
        escalation_keywords=("outage", "shutdown", "attack", "breach", "malware", "state-backed"),
        positive_themes=("cybersecurity",),
        negative_themes=("financials", "communication_services", "utilities", "high_beta_growth"),
    ),
)


def score_flashpoints(
    events: pd.DataFrame,
    *,
    now: datetime | None = None,
    lookback_hours: int = 48,
    rules: tuple[FlashpointRule, ...] = DEFAULT_FLASHPOINT_RULES,
) -> pd.DataFrame:
    if events.empty:
        return _empty_flashpoint_frame()
    frame = events.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    else:
        frame["timestamp"] = pd.NaT
    current = now or datetime.now(timezone.utc)
    cutoff = pd.Timestamp(current - timedelta(hours=lookback_hours))
    if frame["timestamp"].notna().any():
        frame = frame[frame["timestamp"].ge(cutoff)].copy()
    if frame.empty:
        return _empty_flashpoint_frame()
    text = _event_text(frame)
    rows: list[dict[str, Any]] = []
    for rule in rules:
        keyword_hits = text.str.contains(_keyword_pattern(rule.keywords), regex=True, na=False)
        if not keyword_hits.any():
            continue
        matched = frame[keyword_hits].copy()
        matched_text = text[keyword_hits]
        escalation_hits = matched_text.str.contains(_keyword_pattern(rule.escalation_keywords), regex=True, na=False)
        sentiment = pd.to_numeric(matched.get("sentiment_numeric", 0.0), errors="coerce").fillna(0.0)
        recent_cutoff = pd.Timestamp(current - timedelta(hours=6))
        recent_count = int(matched["timestamp"].ge(recent_cutoff).sum()) if matched["timestamp"].notna().any() else 0
        event_count = int(len(matched))
        escalation_count = int(escalation_hits.sum())
        intensity = min(1.0, event_count / 25.0 + escalation_count / 10.0 + recent_count / 10.0)
        tone_penalty = max(0.0, -float(sentiment.mean()) if len(sentiment) else 0.0)
        shock_score = min(1.0, intensity + tone_penalty * 0.25)
        rows.append(
            {
                "flashpoint": rule.name,
                "family": rule.family,
                "commodity_channel": rule.commodity_channel,
                "event_count": event_count,
                "recent_event_count_6h": recent_count,
                "escalation_event_count": escalation_count,
                "mean_sentiment": float(sentiment.mean()) if len(sentiment) else 0.0,
                "shock_score": shock_score,
                "positive_themes": ",".join(rule.positive_themes),
                "negative_themes": ",".join(rule.negative_themes),
                "latest_timestamp": matched["timestamp"].max().isoformat() if matched["timestamp"].notna().any() else None,
                "sample_headline": str(matched.iloc[0].get("title", "")),
            }
        )
    if not rows:
        return _empty_flashpoint_frame()
    return pd.DataFrame(rows).sort_values(["shock_score", "event_count"], ascending=[False, False]).reset_index(drop=True)


def build_sector_theme_monitor(
    *,
    dataset: pd.DataFrame,
    universe: pd.DataFrame,
    model_path: Path,
    flashpoints: pd.DataFrame | None = None,
    require_promoted: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = load_model_manifest(model_path)
    if require_promoted and manifest.get("status") != MODEL_STATUS_PROMOTED:
        raise ValueError(f"Model must be promoted for live monitoring; found {manifest.get('status', 'unknown')}")
    latest = dataset.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).copy()
    scored = score_volatile_frame(latest, model_path)
    universe_themes = classify_universe_themes(universe)
    scored = scored.merge(universe_themes, on="ticker", how="left")
    scored["monitor_theme"] = scored["monitor_theme"].fillna("other")
    scored["global_positive_impact"] = scored["monitor_theme"].map(lambda theme: _theme_impact(theme, flashpoints, "positive")).fillna(0.0)
    scored["global_negative_impact"] = scored["monitor_theme"].map(lambda theme: _theme_impact(theme, flashpoints, "negative")).fillna(0.0)
    scored["global_net_impact"] = scored["global_positive_impact"] - scored["global_negative_impact"]
    scored["monitor_score"] = (
        pd.to_numeric(scored["volatile_model_probability"], errors="coerce").fillna(0.0)
        + 0.10 * scored["global_net_impact"]
        + 0.02 * _numeric_feature(scored, "volume_z20").clip(lower=0.0, upper=5.0)
        + 0.01 * _numeric_feature(scored, "news_count_z30").clip(lower=0.0, upper=5.0)
    )
    scored["monitor_signal"] = scored.apply(_monitor_signal, axis=1)
    ticker_report = scored.sort_values("monitor_score", ascending=False).reset_index(drop=True)
    sector_report = _sector_report(ticker_report)
    return sector_report, ticker_report


def classify_universe_themes(universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(columns=["ticker", "monitor_theme"])
    frame = universe.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    text = (
        frame.get("sector", "").fillna("").astype(str)
        + " "
        + frame.get("industry", "").fillna("").astype(str)
        + " "
        + frame.get("company", "").fillna("").astype(str)
    ).str.lower()
    frame["monitor_theme"] = "other"
    frame.loc[text.str.contains("biotechnology|life sciences|pharmaceutical", regex=True), "monitor_theme"] = "healthcare_biotech"
    frame.loc[text.str.contains("health care equipment|health care supplies|managed health|health care provider", regex=True), "monitor_theme"] = "healthcare_devices_services"
    frame.loc[text.str.contains("semiconductor|semiconductors", regex=True), "monitor_theme"] = "semis_ai_hardware"
    frame.loc[text.str.contains("software|application software|systems software", regex=True), "monitor_theme"] = "software"
    frame.loc[text.str.contains("interactive media|communication|telecom|movies|broadcasting", regex=True), "monitor_theme"] = "communication_services"
    frame.loc[text.str.contains("data center|cloud|internet services|it consulting|technology hardware", regex=True), "monitor_theme"] = "ai_data_centers"
    frame.loc[text.str.contains("oil|gas|energy|drilling|refining|exploration", regex=True), "monitor_theme"] = "energy_oil_gas"
    frame.loc[text.str.contains("aerospace|defense", regex=True), "monitor_theme"] = "defense_aerospace"
    frame.loc[text.str.contains("airline|hotel|resort|cruise|travel", regex=True), "monitor_theme"] = "airlines_travel"
    frame.loc[text.str.contains("bank|capital markets|insurance|financial", regex=True), "monitor_theme"] = "financials"
    frame.loc[text.str.contains("electric utilities|multi-utilities|water utilities|utilities", regex=True), "monitor_theme"] = "utilities"
    return frame[["ticker", "monitor_theme"]].drop_duplicates("ticker")


def _sector_report(ticker_report: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = {
        "tickers": ("ticker", "nunique"),
        "avg_model_probability": ("volatile_model_probability", "mean"),
        "max_model_probability": ("volatile_model_probability", "max"),
        "avg_monitor_score": ("monitor_score", "mean"),
        "avg_global_net_impact": ("global_net_impact", "mean"),
        "avg_volume_z20": ("volume_z20", "mean"),
        "avg_news_count": ("news_count", "mean"),
        "top_candidates": ("ticker", lambda values: ",".join(list(values.head(8)))),
    }
    grouped = ticker_report.sort_values("monitor_score", ascending=False).groupby("monitor_theme").agg(**numeric_cols)
    grouped = grouped.reset_index().sort_values("avg_monitor_score", ascending=False)
    return grouped


def _numeric_feature(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _theme_impact(theme: str, flashpoints: pd.DataFrame | None, side: str) -> float:
    if flashpoints is None or flashpoints.empty:
        return 0.0
    column = "positive_themes" if side == "positive" else "negative_themes"
    total = 0.0
    for _, row in flashpoints.iterrows():
        themes = {item.strip() for item in str(row.get(column, "")).split(",") if item.strip()}
        if theme in themes:
            total += float(row.get("shock_score", 0.0) or 0.0)
    return min(1.0, total)


def _monitor_signal(row: pd.Series) -> str:
    prob = float(row.get("volatile_model_probability", 0.0) or 0.0)
    net = float(row.get("global_net_impact", 0.0) or 0.0)
    if prob >= 0.18 and net >= 0.15:
        return "bullish_with_global_tailwind"
    if prob >= 0.18 and net <= -0.15:
        return "model_positive_but_global_headwind"
    if prob >= 0.18:
        return "bullish_watch"
    if net <= -0.35:
        return "global_downside_risk"
    if abs(net) >= 0.20:
        return "global_two_sided_watch"
    return "neutral"


def _event_text(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for column in ["title", "summary", "text"]:
        if column in frame.columns:
            parts.append(frame[column].fillna("").astype(str))
    if not parts:
        return pd.Series([""] * len(frame), index=frame.index)
    output = parts[0]
    for part in parts[1:]:
        output = output + " " + part
    return output.str.lower()


def _keyword_pattern(values: tuple[str, ...]) -> str:
    return "|".join(re.escape(value.lower()) for value in values)


def _empty_flashpoint_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "flashpoint",
            "family",
            "commodity_channel",
            "event_count",
            "recent_event_count_6h",
            "escalation_event_count",
            "mean_sentiment",
            "shock_score",
            "positive_themes",
            "negative_themes",
            "latest_timestamp",
            "sample_headline",
        ]
    )
