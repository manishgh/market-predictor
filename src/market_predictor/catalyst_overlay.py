from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

SOURCE_FAMILIES = ["alpaca", "reddit", "seeking_alpha", "sec", "finviz"]
MATERIAL_EVENT_TYPES = ["earnings", "guidance", "analyst", "ma", "fda", "contract", "offering", "insider"]
VETO_EVENT_TYPES = ["guidance", "fda", "offering"]


@dataclass(frozen=True)
class CatalystAssessment:
    status: str
    direction: str
    score: float
    event_count: int
    source_diversity: int
    sentiment: float
    relevance: float
    minutes_since_latest: float | None
    material_event_count: int
    reasons: list[str]

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def assess_catalyst_overlay(row: pd.Series, *, model_probability: float | None) -> CatalystAssessment:
    event_count = int(
        max(
            0.0,
            _first_number(
                row,
                ["news_count_2h", "event_count_1d", "event_count_3d", "news_count_1d", "event_count", "news_count"],
            ),
        )
    )
    sentiment = _first_number(
        row,
        [
            "sentiment_mean_2h",
            "latest_catalyst_sentiment",
            "sentiment_mean_1d",
            "sentiment_mean_3d",
            "sentiment_mean",
        ],
    )
    relevance = _first_number(
        row,
        [
            "event_relevance_mean_2h",
            "latest_catalyst_relevance",
            "event_relevance_mean_1d",
            "event_relevance_mean_3d",
            "event_relevance_score",
        ],
    )
    generic_count = _first_number(row, ["generic_movers_count_2h", "generic_movers_count_1d"])
    minutes_since = _optional_number(row.get("minutes_since_last_catalyst"))
    source_diversity = _source_diversity(row)
    material_event_count = int(sum(_event_count(row, event_type) for event_type in MATERIAL_EVENT_TYPES))
    veto_event_count = int(sum(_event_count(row, event_type) for event_type in VETO_EVENT_TYPES))
    reasons: list[str] = []

    if event_count <= 0:
        return CatalystAssessment(
            status="absent",
            direction="none",
            score=0.0,
            event_count=0,
            source_diversity=0,
            sentiment=0.0,
            relevance=0.0,
            minutes_since_latest=minutes_since,
            material_event_count=0,
            reasons=["no recent ticker catalyst"],
        )

    generic_ratio = min(1.0, generic_count / event_count) if event_count else 0.0
    if relevance < 0.5 or generic_ratio >= 0.75:
        reasons.append("recent catalyst evidence is low relevance or mostly generic")
        return CatalystAssessment(
            status="mixed",
            direction="mixed",
            score=0.0,
            event_count=event_count,
            source_diversity=source_diversity,
            sentiment=sentiment,
            relevance=relevance,
            minutes_since_latest=minutes_since,
            material_event_count=material_event_count,
            reasons=reasons,
        )

    direction = "positive" if sentiment >= 0.15 else "negative" if sentiment <= -0.15 else "mixed"
    evidence_weight = min(2.0, math.log1p(event_count))
    source_weight = 1.0 + min(source_diversity, 3) * 0.15
    relevance_weight = min(max(relevance, 0.0), 2.0)
    score = float(np.clip(sentiment * relevance_weight * evidence_weight * source_weight, -1.0, 1.0))
    bullish_model = model_probability is not None and model_probability >= 0.55
    bearish_model = model_probability is not None and model_probability <= 0.40

    if direction == "negative" and sentiment <= -0.35 and relevance >= 1.0 and veto_event_count > 0:
        status = "veto"
        reasons.append("strong negative material catalyst conflicts with a long entry")
    elif (bullish_model and direction == "negative") or (bearish_model and direction == "positive"):
        status = "conflicting"
        reasons.append("catalyst direction conflicts with the technical model")
    elif (bullish_model and direction == "positive") or (bearish_model and direction == "negative"):
        status = "confirmed"
        reasons.append("catalyst direction confirms the technical model")
    else:
        status = "mixed"
        reasons.append("catalyst direction is not decisive for the model state")

    if source_diversity >= 2:
        reasons.append(f"evidence appears across {source_diversity} source families")
    if material_event_count > 0:
        reasons.append(f"material event count: {material_event_count}")
    return CatalystAssessment(
        status=status,
        direction=direction,
        score=score,
        event_count=event_count,
        source_diversity=source_diversity,
        sentiment=sentiment,
        relevance=relevance,
        minutes_since_latest=minutes_since,
        material_event_count=material_event_count,
        reasons=reasons,
    )


def overlay_decision_score(model_probability: float | None, assessment: CatalystAssessment) -> float:
    probability = float(model_probability) if model_probability is not None and np.isfinite(model_probability) else 0.0
    adjustment = {
        "confirmed": 0.04,
        "conflicting": -0.08,
        "veto": -0.20,
        "mixed": 0.0,
        "absent": 0.0,
    }.get(assessment.status, 0.0)
    return float(np.clip(probability + adjustment, 0.0, 1.0))


def _source_diversity(row: pd.Series) -> int:
    present = 0
    for family in SOURCE_FAMILIES:
        value = _first_number(
            row,
            [
                f"source_count_{family}_2h",
                f"source_count_{family}_1d",
                f"source_count_{family}_3d",
                f"source_count_{family}",
            ],
        )
        present += int(value > 0)
    return present


def _event_count(row: pd.Series, event_type: str) -> float:
    return _first_number(
        row,
        [
            f"event_{event_type}_count_2h",
            f"event_{event_type}_count_1d",
            f"event_{event_type}_count",
            f"event_{event_type}",
        ],
    )


def _first_number(row: pd.Series, columns: list[str]) -> float:
    for column in columns:
        if column in row.index:
            value = _optional_number(row.get(column))
            if value is not None:
                return value
    return 0.0


def _optional_number(value: object) -> float | None:
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None
