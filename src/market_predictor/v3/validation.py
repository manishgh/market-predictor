from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import cast

import numpy as np
import pandas as pd

from market_predictor.v3.errors import DataReadinessError


@dataclass(frozen=True, slots=True)
class V3Fold:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_sessions: int

    def audit_record(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_rows": len(self.train_indices),
            "test_rows": len(self.test_indices),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "embargo_sessions": self.embargo_sessions,
        }


@dataclass(frozen=True)
class TickerHoldoutPlan:
    holdout_tickers: frozenset[str]
    representation_audit: pd.DataFrame
    assignment_cutoff_utc: str
    ticker_summary_sha256: str


class V3PurgedWalkForwardSplit:
    """Expanding-window session split that preserves every ranking query."""

    def __init__(
        self,
        *,
        n_splits: int = 4,
        embargo_sessions: int = 1,
        min_train_sessions: int = 20,
        min_train_rows: int = 500,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if embargo_sessions < 0:
            raise ValueError("embargo_sessions cannot be negative")
        if min_train_sessions < 2 or min_train_rows < 1:
            raise ValueError("minimum training sessions and rows must be positive")
        self.n_splits = n_splits
        self.embargo_sessions = embargo_sessions
        self.min_train_sessions = min_train_sessions
        self.min_train_rows = min_train_rows

    def split(self, frame: pd.DataFrame) -> list[V3Fold]:
        required = {"session_date_et", "decision_group_id"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataReadinessError(f"V3 validation input missing columns: {', '.join(missing)}")
        sessions = pd.to_datetime(frame["session_date_et"], errors="coerce").dt.date
        if bool(sessions.isna().any()):
            raise DataReadinessError("V3 validation contains invalid session_date_et values")
        query_session_count = pd.DataFrame(
            {"query": frame["decision_group_id"].astype(str), "session": sessions}
        ).groupby("query")["session"].nunique()
        if bool(query_session_count.gt(1).any()):
            raise DataReadinessError("a decision_group_id spans multiple sessions")
        ordered = sorted(sessions.unique())
        first_test_position = self.min_train_sessions + self.embargo_sessions
        remaining = len(ordered) - first_test_position
        if remaining < self.n_splits:
            raise DataReadinessError(
                "insufficient sessions for the minimum training window, embargo, and requested folds"
            )
        fold_size = max(1, remaining // self.n_splits)
        folds: list[V3Fold] = []
        for fold_number in range(self.n_splits):
            test_start_position = first_test_position + fold_number * fold_size
            test_end_position = len(ordered) if fold_number == self.n_splits - 1 else min(
                test_start_position + fold_size,
                len(ordered),
            )
            train_end_position = test_start_position - self.embargo_sessions
            if train_end_position < 1 or test_start_position >= len(ordered):
                continue
            train_sessions = set(ordered[:train_end_position])
            test_sessions = set(ordered[test_start_position:test_end_position])
            train_indices = np.flatnonzero(sessions.isin(train_sessions).to_numpy())
            test_indices = np.flatnonzero(sessions.isin(test_sessions).to_numpy())
            if len(train_indices) < self.min_train_rows or len(test_indices) == 0:
                continue
            train_queries = set(frame.iloc[train_indices]["decision_group_id"].astype(str))
            test_queries = set(frame.iloc[test_indices]["decision_group_id"].astype(str))
            if train_queries & test_queries:
                raise DataReadinessError("ranking query was split between train and test")
            folds.append(
                V3Fold(
                    fold=fold_number,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_start=min(train_sessions),
                    train_end=max(train_sessions),
                    test_start=min(test_sessions),
                    test_end=max(test_sessions),
                    embargo_sessions=self.embargo_sessions,
                )
            )
        if len(folds) < 2:
            raise DataReadinessError("fewer than two valid V3 walk-forward folds remain")
        return folds


def deterministic_ticker_holdout(
    tickers: pd.Series,
    *,
    fraction: float = 0.2,
    seed: int = 42,
) -> frozenset[str]:
    if not 0 < fraction < 1:
        raise ValueError("ticker holdout fraction must be between zero and one")
    unique = sorted(tickers.dropna().astype(str).str.upper().str.strip().unique())
    if len(unique) < 2:
        raise DataReadinessError("ticker holdout requires at least two symbols")
    scored = sorted(unique, key=lambda ticker: hashlib.sha256(f"{seed}:{ticker}".encode()).hexdigest())
    minimum = 2 if len(scored) >= 3 else 1
    count = min(len(scored) - 1, max(minimum, round(len(scored) * fraction)))
    return frozenset(scored[:count])


def deterministic_stratified_ticker_holdout(
    frame: pd.DataFrame,
    *,
    label_columns: Sequence[str],
    fraction: float = 0.2,
    seed: int = 42,
) -> TickerHoldoutPlan:
    """Assign whole tickers using only a caller-supplied causal summary window."""

    if not 0 < fraction < 1:
        raise ValueError("ticker holdout fraction must be between zero and one")
    required = {"ticker", "decision_time_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(
            f"stratified ticker holdout input missing columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise DataReadinessError("stratified ticker holdout requires causal assignment rows")
    decision = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
    if bool(decision.isna().any()):
        raise DataReadinessError("stratified ticker holdout contains invalid decision times")

    summaries, availability = _ticker_summaries(frame, label_columns=label_columns)
    if len(summaries) < 2:
        raise DataReadinessError("ticker holdout requires at least two symbols")
    holdout_count = min(
        len(summaries) - 1,
        max(2 if len(summaries) >= 3 else 1, round(len(summaries) * fraction)),
    )
    dimensions = [
        "sector",
        "market_cap",
        "liquidity",
        "label_rate",
        "event_coverage",
    ]
    groups = _stratum_groups(summaries, dimensions, availability)
    holdout = _balanced_assignment(
        summaries,
        groups=groups,
        count=holdout_count,
        fraction=fraction,
        seed=seed,
    )
    cutoff = _utc_iso(decision.max())
    audit = _representation_audit(
        summaries,
        holdout=holdout,
        dimensions=dimensions,
        availability=availability,
        assignment_cutoff_utc=cutoff,
    )
    failed = audit[audit["required"].astype(bool) & ~audit["represented"].astype(bool)]
    if not failed.empty:
        detail = ", ".join(
            f"{row.dimension}={row.stratum}" for row in failed.itertuples(index=False)
        )
        raise DataReadinessError(f"ticker holdout cannot represent required strata: {detail}")
    summary_records = summaries.sort_values("ticker").to_dict(orient="records")
    summary_hash = hashlib.sha256(
        json.dumps(summary_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TickerHoldoutPlan(
        holdout_tickers=frozenset(holdout),
        representation_audit=audit,
        assignment_cutoff_utc=cutoff,
        ticker_summary_sha256=summary_hash,
    )


def validation_row_identities(frame: pd.DataFrame) -> pd.Series:
    required = {"ticker", "decision_group_id", "decision_time_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(f"validation identity missing columns: {', '.join(missing)}")
    decision = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
    if bool(decision.isna().any()):
        raise DataReadinessError("validation identity contains invalid decision_time_utc")
    ticker = frame["ticker"].astype(str).str.upper().str.strip()
    group = frame["decision_group_id"].astype(str).str.strip()
    identities = pd.Series(
        [
            hashlib.sha256(f"{symbol}|{query}|{timestamp.isoformat()}".encode()).hexdigest()
            for symbol, query, timestamp in zip(ticker, group, decision, strict=True)
        ],
        index=frame.index,
        dtype="string",
    )
    duplicates = identities.duplicated(keep=False)
    if bool(duplicates.any()):
        raise DataReadinessError("validation row identity is not unique")
    return identities


def identity_set_sha256(values: Iterable[object]) -> str:
    normalized = sorted(str(value) for value in values)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def causal_fold_training_indices(
    frame: pd.DataFrame,
    *,
    candidate_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, pd.Timestamp, pd.Timestamp]:
    required = {"label_available_at_utc", "decision_time_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(f"causal fold input missing columns: {', '.join(missing)}")
    if len(candidate_indices) == 0 or len(test_indices) == 0:
        raise DataReadinessError("causal fold requires non-empty train and test rows")
    label_available = pd.to_datetime(frame["label_available_at_utc"], utc=True, errors="coerce")
    decision = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
    if bool(label_available.isna().any() | decision.isna().any()):
        raise DataReadinessError("causal fold contains invalid availability timestamps")
    min_test_decision = decision.iloc[test_indices].min()
    eligible = label_available.iloc[candidate_indices].lt(min_test_decision).to_numpy()
    train_indices = candidate_indices[eligible]
    if len(train_indices) == 0:
        raise DataReadinessError("no causally mature training labels remain before the test fold")
    max_train_label = label_available.iloc[train_indices].max()
    if not max_train_label < min_test_decision:
        raise DataReadinessError(
            "max train label availability must be strictly earlier than min test decision"
        )
    return train_indices, max_train_label, min_test_decision


def _ticker_summaries(
    frame: pd.DataFrame,
    *,
    label_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, bool]]:
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    if bool(data["ticker"].eq("").any()):
        raise DataReadinessError("ticker holdout contains an empty symbol")

    sector_available = "sector" in data.columns
    cap_available = "market_cap_bucket" in data.columns or _has_valid_one_hot(
        data, "market_cap_"
    )
    liquidity_available = "liquidity_bucket" in data.columns or _has_valid_one_hot(
        data, "liquidity_"
    )
    valid_labels = [column for column in label_columns if column in data.columns]
    label_available = bool(valid_labels)
    event_columns = [column for column in data.columns if column.startswith("event_count_")]
    event_available = bool(event_columns)

    data["_sector"] = _normalized_category(data.get("sector"), data.index)
    data["_market_cap"] = _category_or_one_hot(
        data,
        category="market_cap_bucket",
        prefix="market_cap_",
    )
    data["_liquidity"] = _category_or_one_hot(
        data,
        category="liquidity_bucket",
        prefix="liquidity_",
    )
    if valid_labels:
        label_values = data[valid_labels].apply(pd.to_numeric, errors="coerce")
        data["_label_value"] = label_values.mean(axis=1)
    else:
        data["_label_value"] = np.nan
    if event_columns:
        events = data[event_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        data["_event_value"] = events.sum(axis=1).gt(0).astype(float)
    else:
        data["_event_value"] = np.nan

    records: list[dict[str, object]] = []
    for ticker, rows in data.groupby("ticker", sort=True):
        label_rate = float(rows["_label_value"].mean()) if label_available else float("nan")
        event_rate = float(rows["_event_value"].mean()) if event_available else float("nan")
        records.append(
            {
                "ticker": ticker,
                "sector": _stable_mode(rows["_sector"]),
                "market_cap": _stable_mode(rows["_market_cap"]),
                "liquidity": _stable_mode(rows["_liquidity"]),
                "label_rate": _rate_bucket(label_rate),
                "event_coverage": _coverage_bucket(event_rate),
            }
        )
    availability = {
        "sector": sector_available,
        "market_cap": cap_available,
        "liquidity": liquidity_available,
        "label_rate": label_available,
        "event_coverage": event_available,
    }
    return pd.DataFrame(records), availability


def _stratum_groups(
    summaries: pd.DataFrame,
    dimensions: Sequence[str],
    availability: dict[str, bool],
) -> dict[tuple[str, str], frozenset[str]]:
    groups: dict[tuple[str, str], frozenset[str]] = {}
    for dimension in dimensions:
        if not availability[dimension]:
            continue
        for stratum, rows in summaries.groupby(dimension, sort=True):
            tickers = frozenset(rows["ticker"].astype(str))
            if len(tickers) >= 2:
                groups[(dimension, str(stratum))] = tickers
    return groups


def _balanced_assignment(
    summaries: pd.DataFrame,
    *,
    groups: dict[tuple[str, str], frozenset[str]],
    count: int,
    fraction: float,
    seed: int,
) -> set[str]:
    tickers = sorted(summaries["ticker"].astype(str))
    order = {
        ticker: rank
        for rank, ticker in enumerate(
            sorted(tickers, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
        )
    }
    desired = {
        key: min(len(members) - 1, max(1, round(len(members) * fraction)))
        for key, members in groups.items()
    }

    def objective(selected: set[str]) -> tuple[int, int, tuple[int, ...]]:
        missing = sum(not (selected & members) or not (members - selected) for members in groups.values())
        deviation = sum(abs(len(selected & groups[key]) - target) for key, target in desired.items())
        return missing, deviation, tuple(sorted(order[ticker] for ticker in selected))

    selected: set[str] = set()
    while len(selected) < count:
        candidates = [ticker for ticker in tickers if ticker not in selected]
        selected = min((selected | {ticker} for ticker in candidates), key=objective)

    for _ in range(max(1, len(groups) * 2)):
        if objective(selected)[0] == 0:
            return selected
        trials = [
            (selected - {removed}) | {added}
            for removed in sorted(selected)
            for added in tickers
            if added not in selected
        ]
        best = min(trials, key=objective)
        if objective(best) >= objective(selected):
            break
        selected = best
    return selected


def _representation_audit(
    summaries: pd.DataFrame,
    *,
    holdout: set[str],
    dimensions: Sequence[str],
    availability: dict[str, bool],
    assignment_cutoff_utc: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dimension in dimensions:
        for stratum, rows in summaries.groupby(dimension, sort=True):
            tickers = set(rows["ticker"].astype(str))
            held = tickers & holdout
            development = tickers - holdout
            required = availability[dimension] and len(tickers) >= 2
            records.append(
                {
                    "record_type": "holdout_representation",
                    "dimension": dimension,
                    "stratum": str(stratum),
                    "available": availability[dimension],
                    "required": required,
                    "represented": bool(held and development) if required else True,
                    "total_tickers": len(tickers),
                    "holdout_tickers": len(held),
                    "development_tickers": len(development),
                    "assignment_cutoff_utc": assignment_cutoff_utc,
                }
            )
    return pd.DataFrame(records)


def _category_or_one_hot(data: pd.DataFrame, *, category: str, prefix: str) -> pd.Series:
    if category in data.columns:
        return _normalized_category(data[category], data.index)
    columns = sorted(column for column in data.columns if column.startswith(prefix))
    if not columns or not _has_valid_one_hot(data, prefix):
        return pd.Series("unknown", index=data.index, dtype="string")
    numeric = data[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    chosen = numeric.idxmax(axis=1).str.removeprefix(prefix)
    return chosen.where(numeric.max(axis=1).gt(0), "unknown").astype("string")


def _normalized_category(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series("unknown", index=index, dtype="string")
    normalized = values.fillna("unknown").astype(str).str.strip().str.lower()
    return normalized.where(normalized.ne(""), "unknown").astype("string")


def _stable_mode(values: pd.Series) -> str:
    counts = values.fillna("unknown").astype(str).value_counts()
    maximum = int(counts.max()) if not counts.empty else 0
    return sorted(counts[counts.eq(maximum)].index)[0] if maximum else "unknown"


def _rate_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 1 / 3:
        return "low"
    if value < 2 / 3:
        return "medium"
    return "high"


def _coverage_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value == 0:
        return "none"
    if value < 0.25:
        return "sparse"
    if value < 0.75:
        return "medium"
    return "high"


def _has_valid_one_hot(frame: pd.DataFrame, prefix: str) -> bool:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    if not columns:
        return False
    values = frame[columns].apply(pd.to_numeric, errors="coerce").stack().dropna().unique()
    return bool(len(values) and set(values).issubset({0, 1}))


def _utc_iso(value: pd.Timestamp | datetime) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(str, timestamp.isoformat())
