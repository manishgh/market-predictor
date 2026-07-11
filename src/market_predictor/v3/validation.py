from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

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
