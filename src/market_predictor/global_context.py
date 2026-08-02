from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd


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
    current = now or datetime.now(UTC)
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
