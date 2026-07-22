"""Canonical prediction ranking, selection, and action policy.

This module is the single source of truth for how model probabilities become a
ranked, selected, and labelled trading view. Serving (``prediction_service``),
offline evaluation (``swing.evaluation`` / ``intraday.evaluation``), and
promotion all import these functions so that the policy evaluated for promotion
is byte-for-byte the policy that is served.

Scope: ranking score, deterministic selection, and action labels only.
Executable fills, slippage, participation, and capital allocation belong to
``execution_policy`` and are intentionally excluded here.

The policy is immutable and content-addressed: :data:`PREDICTION_POLICY_SHA256`
changes if and only if the declarative semantics below change, and that hash is
bound into promotion evidence and prediction snapshots as identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

PREDICTION_POLICY_ID = "market_predictor.prediction_policy.v1"

# --- Swing action thresholds (probability of the model's positive target) ---
SWING_STRONG = 0.65
SWING_WATCH = 0.55
SWING_LOW = 0.40

# --- Intraday action thresholds (opportunity / downside probabilities) ---
INTRADAY_DOWNSIDE_VETO = 0.55
INTRADAY_ENTRY = 0.70
INTRADAY_ENTRY_MAX_DOWNSIDE = 0.35
INTRADAY_WATCH = 0.55
INTRADAY_WATCH_MAX_DOWNSIDE = 0.45
INTRADAY_LOW = 0.40
INTRADAY_AVOID_DOWNSIDE = 0.50

# Default eligibility ceiling for the intraday *selected* set. This is the
# tradeable-set bound used by economics; it is stricter than the labelling veto.
INTRADAY_SELECTION_DOWNSIDE_CEILING = INTRADAY_WATCH_MAX_DOWNSIDE

# Rank sentinel for rows that cannot be scored. Sorts last under descending rank.
UNSCORABLE_SCORE = float("-inf")

_SCORE_WORK_COLUMN = "__prediction_decision_score"
DECISION_SCORE_COLUMN = "decision_score"


# --------------------------------------------------------------------------- #
# Scalar scores (identical to the serving decision score)
# --------------------------------------------------------------------------- #
def swing_decision_score(model_probability: float | None) -> float:
    """Swing ranks by model probability only. Missing probability sorts last."""

    value = finite_or_none(model_probability)
    return value if value is not None else UNSCORABLE_SCORE


def intraday_decision_score(opportunity: float | None, downside: float | None) -> float:
    """Intraday score is ``opportunity * (1 - downside)``.

    Both probabilities are required; a missing input yields the unscorable
    sentinel so the row can never outrank a genuinely scored candidate.
    """

    opp = finite_or_none(opportunity)
    down = finite_or_none(downside)
    if opp is None or down is None:
        return UNSCORABLE_SCORE
    return opp * (1.0 - down)


# --------------------------------------------------------------------------- #
# Vectorized scores (frame-aligned pd.Series)
# --------------------------------------------------------------------------- #
def swing_decision_scores(frame: pd.DataFrame, *, probability_column: str) -> pd.Series:
    values = pd.to_numeric(frame[probability_column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    return values.where(finite, other=UNSCORABLE_SCORE).astype(float)


def intraday_decision_scores(
    frame: pd.DataFrame,
    *,
    opportunity_column: str,
    downside_column: str,
) -> pd.Series:
    opp = pd.to_numeric(frame[opportunity_column], errors="coerce")
    down = pd.to_numeric(frame[downside_column], errors="coerce")
    score = opp * (1.0 - down)
    valid = np.isfinite(opp.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(
        down.to_numpy(dtype=float, na_value=np.nan)
    )
    return score.where(valid, other=UNSCORABLE_SCORE).astype(float)


def intraday_selection_eligible(
    frame: pd.DataFrame,
    *,
    downside_column: str,
    downside_ceiling: float = INTRADAY_SELECTION_DOWNSIDE_CEILING,
) -> pd.Series:
    """Rows eligible for the intraday *selected* (tradeable) set."""

    down = pd.to_numeric(frame[downside_column], errors="coerce")
    return down.le(downside_ceiling) & down.notna()


# --------------------------------------------------------------------------- #
# Deterministic selection (shared by every economics phase)
# --------------------------------------------------------------------------- #
def select_top_k_per_group(
    frame: pd.DataFrame,
    *,
    score: pd.Series,
    group_column: str,
    top_k: int,
    tie_breakers: Sequence[tuple[str, bool]],
    eligible: pd.Series | None = None,
) -> pd.DataFrame:
    """Select the top ``top_k`` rows per group by decision score.

    ``score`` and ``eligible`` must align to ``frame.index``. Ordering is
    ``score`` descending, then each ``(column, ascending)`` tie-breaker, using a
    stable sort so the result is fully deterministic. The selected frame carries
    the score in :data:`DECISION_SCORE_COLUMN`.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    working = frame.copy()
    working[_SCORE_WORK_COLUMN] = pd.Series(score, index=frame.index).astype(float)
    if eligible is not None:
        mask = pd.Series(eligible, index=frame.index).fillna(False).astype(bool)
        working = working.loc[mask]
    sort_columns = [_SCORE_WORK_COLUMN, *(column for column, _ in tie_breakers)]
    ascending = [False, *(direction for _, direction in tie_breakers)]
    ordered = working.sort_values(sort_columns, ascending=ascending, kind="stable")
    selected = ordered.groupby(group_column, sort=False).head(top_k)
    return selected.rename(columns={_SCORE_WORK_COLUMN: DECISION_SCORE_COLUMN})


# --------------------------------------------------------------------------- #
# Action labels (identical to the serving signal semantics)
# --------------------------------------------------------------------------- #
def swing_action(probability: float | None) -> str:
    value = finite_or_none(probability)
    if value is None:
        return "not_scored"
    if value >= SWING_STRONG:
        return "strong_bullish_watch"
    if value >= SWING_WATCH:
        return "bullish_watch"
    if value <= SWING_LOW:
        return "low_probability"
    return "neutral"


def intraday_action(opportunity: float | None, downside: float | None) -> str:
    opp = finite_or_none(opportunity)
    down = finite_or_none(downside)
    if opp is None or down is None:
        return "not_scored"
    if down >= INTRADAY_DOWNSIDE_VETO:
        return "avoid_entry_downside_risk"
    if opp >= INTRADAY_ENTRY and down <= INTRADAY_ENTRY_MAX_DOWNSIDE:
        return "entry_candidate"
    if opp >= INTRADAY_WATCH and down <= INTRADAY_WATCH_MAX_DOWNSIDE:
        return "watch_for_confirmation"
    if opp <= INTRADAY_LOW or down > INTRADAY_AVOID_DOWNSIDE:
        return "avoid_entry"
    return "neutral"


def finite_or_none(value: float | None) -> float | None:
    """Coerce to a finite float or ``None`` (matches serving ``_float_or_none``)."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


# --------------------------------------------------------------------------- #
# Ranking-quality metrics (decision-group aware, at the deployed k)
# --------------------------------------------------------------------------- #
def group_ranking_metrics(
    frame: pd.DataFrame,
    *,
    target_column: str,
    score: pd.Series,
    group_column: str,
    k: int,
    eligible: pd.Series | None = None,
) -> dict[str, float]:
    """Precision / lift / NDCG at ``k`` measured *within* each decision group.

    This is the deployment selection rule's own quality metric: rank each group
    by decision score, take the top ``k`` eligible rows, and compare realized
    positives to the group base rate. It is invariant to any per-group monotonic
    rescaling of scores, and a ranker that orders candidates randomly within a
    group scores ~1.0 lift regardless of global score levels.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    work = pd.DataFrame(
        {
            "__group": frame[group_column].to_numpy(),
            "__target": target,
            "__score": pd.Series(score, index=frame.index).astype(float).to_numpy(),
        }
    )
    if eligible is not None:
        mask = pd.Series(eligible, index=frame.index).fillna(False).astype(bool).to_numpy()
        work = work.loc[mask]
    work = work.dropna(subset=["__target", "__score"])

    total_selected = 0
    total_positive = 0.0
    total_expected = 0.0
    total_rows = 0
    total_group_positive = 0.0
    ndcg_values: list[float] = []
    groups_evaluated = 0
    for _, group in work.groupby("__group", sort=False):
        rows = len(group)
        if rows == 0:
            continue
        positives_in_group = float(group["__target"].sum())
        base = positives_in_group / rows
        top = group.sort_values("__score", ascending=False, kind="stable").head(k)
        selected = len(top)
        selected_positive = float(top["__target"].to_numpy(dtype=float).sum())
        total_selected += selected
        total_positive += selected_positive
        total_expected += selected * base
        total_rows += rows
        total_group_positive += positives_in_group
        ndcg = _ndcg_at_k(top["__target"].to_numpy(dtype=float), positives_in_group, k)
        if math.isfinite(ndcg):
            ndcg_values.append(ndcg)
        groups_evaluated += 1

    if total_selected == 0:
        return {
            "group_precision_at_k": float("nan"),
            "group_lift_at_k": float("nan"),
            "group_ndcg_at_k": float("nan"),
            "group_base_rate": float("nan"),
            "groups_evaluated": 0.0,
            "k": float(k),
        }
    return {
        "group_precision_at_k": total_positive / total_selected,
        "group_lift_at_k": (total_positive / total_expected if total_expected > 0 else float("nan")),
        "group_ndcg_at_k": (float(np.mean(ndcg_values)) if ndcg_values else float("nan")),
        "group_base_rate": (total_group_positive / total_rows if total_rows else float("nan")),
        "groups_evaluated": float(groups_evaluated),
        "k": float(k),
    }


def _ndcg_at_k(selected_relevance: np.ndarray, group_positives: float, k: int) -> float:
    """Binary-relevance NDCG@k for one group's selected ranking."""

    if selected_relevance.size == 0:
        return float("nan")
    ranks = np.arange(1, selected_relevance.size + 1, dtype=float)
    dcg = float(np.sum(selected_relevance / np.log2(ranks + 1.0)))
    ideal_count = int(min(k, int(group_positives)))
    if ideal_count <= 0:
        return float("nan")
    ideal_ranks = np.arange(1, ideal_count + 1, dtype=float)
    idcg = float(np.sum(1.0 / np.log2(ideal_ranks + 1.0)))
    return dcg / idcg if idcg > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Calibration metrics (shared by swing and intraday)
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    target: pd.Series,
    probability: pd.Series,
    *,
    bins: int = 10,
) -> float:
    y = pd.to_numeric(target, errors="coerce").to_numpy(float)
    p = pd.to_numeric(probability, errors="coerce").clip(0.0, 1.0).to_numpy(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    error = 0.0
    for bucket in range(bins):
        selected = assignments == bucket
        if not selected.any():
            continue
        error += float(selected.mean()) * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return error


def calibration_summary(
    target: pd.Series,
    probability: pd.Series,
    *,
    bins: int = 10,
) -> dict[str, float]:
    """ECE plus reliability-line slope/intercept and mean bias.

    A well-calibrated probability has ECE ~ 0, slope ~ 1, intercept ~ 0, and
    bias (mean predicted minus mean observed) ~ 0. A monotone transform that
    preserves ranking (and AUC) but distorts probability levels moves ECE,
    bias, and slope, so gating these rejects miscalibrated-but-discriminating scores.
    """

    y = pd.to_numeric(target, errors="coerce").to_numpy(float)
    p = pd.to_numeric(probability, errors="coerce").clip(0.0, 1.0).to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size == 0:
        return {
            "expected_calibration_error": float("nan"),
            "calibration_bias": float("nan"),
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
        }
    slope, intercept = _reliability_line(y, p, bins)
    return {
        "expected_calibration_error": expected_calibration_error(pd.Series(y), pd.Series(p), bins=bins),
        "calibration_bias": float(p.mean() - y.mean()),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _reliability_line(y: np.ndarray, p: np.ndarray, bins: int) -> tuple[float, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    xs: list[float] = []
    ys: list[float] = []
    weights: list[float] = []
    for bucket in range(bins):
        selected = assignments == bucket
        if not selected.any():
            continue
        xs.append(float(p[selected].mean()))
        ys.append(float(y[selected].mean()))
        weights.append(float(selected.sum()))
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    x = np.asarray(xs)
    observed = np.asarray(ys)
    w = np.asarray(weights)
    total = w.sum()
    mean_x = float((w * x).sum() / total)
    mean_y = float((w * observed).sum() / total)
    variance = float((w * (x - mean_x) ** 2).sum())
    if variance <= 0:
        return (float("nan"), float("nan"))
    slope = float((w * (x - mean_x) * (observed - mean_y)).sum() / variance)
    intercept = float(mean_y - slope * mean_x)
    return (slope, intercept)


# --------------------------------------------------------------------------- #
# Immutable, content-addressed policy identity
# --------------------------------------------------------------------------- #
SWING_SELECTION_TIE_BREAKERS: tuple[tuple[str, bool], ...] = (("ticker", True),)
INTRADAY_SELECTION_TIE_BREAKERS: tuple[tuple[str, bool], ...] = (
    ("intraday_downside_probability", True),
    ("ticker", True),
)

_POLICY_SPEC = {
    "policy_id": PREDICTION_POLICY_ID,
    "unscorable_score": "negative_infinity",
    "swing": {
        "decision_score": "model_probability",
        "selection": "top_k_per_decision_group_by_decision_score",
        "tie_breakers": ["decision_score:desc", "ticker:asc"],
        "action_thresholds": {
            "strong_at_or_above": SWING_STRONG,
            "watch_at_or_above": SWING_WATCH,
            "low_at_or_below": SWING_LOW,
        },
    },
    "intraday": {
        "decision_score": "opportunity_probability*(1-downside_probability)",
        "selection": "top_k_per_decision_group_by_decision_score_among_eligible",
        "eligibility": "downside_probability<=selection_downside_ceiling",
        "selection_downside_ceiling": INTRADAY_SELECTION_DOWNSIDE_CEILING,
        "tie_breakers": ["decision_score:desc", "downside_probability:asc", "ticker:asc"],
        "action_thresholds": {
            "avoid_downside_at_or_above": INTRADAY_DOWNSIDE_VETO,
            "entry_opportunity_at_or_above": INTRADAY_ENTRY,
            "entry_max_downside": INTRADAY_ENTRY_MAX_DOWNSIDE,
            "watch_opportunity_at_or_above": INTRADAY_WATCH,
            "watch_max_downside": INTRADAY_WATCH_MAX_DOWNSIDE,
            "avoid_opportunity_at_or_below": INTRADAY_LOW,
            "avoid_downside_above": INTRADAY_AVOID_DOWNSIDE,
        },
    },
}

PREDICTION_POLICY_SHA256 = hashlib.sha256(
    json.dumps(_POLICY_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def prediction_policy_identity() -> dict[str, str]:
    """Identity block recorded in promotion evidence and prediction snapshots."""

    return {
        "prediction_policy_id": PREDICTION_POLICY_ID,
        "prediction_policy_sha256": PREDICTION_POLICY_SHA256,
    }
