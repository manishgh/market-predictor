from __future__ import annotations

import unittest
from datetime import date, timedelta

import pandas as pd

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.validation import V3PurgedWalkForwardSplit, deterministic_ticker_holdout


class V3ValidationTests(unittest.TestCase):
    def test_session_folds_preserve_queries_and_embargo(self) -> None:
        frame = _validation_frame(sessions=14, queries_per_session=3, tickers=5)
        splitter = V3PurgedWalkForwardSplit(
            n_splits=3,
            embargo_sessions=1,
            min_train_sessions=5,
            min_train_rows=20,
        )
        folds = splitter.split(frame)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            train = frame.iloc[fold.train_indices]
            test = frame.iloc[fold.test_indices]
            self.assertFalse(set(train["decision_group_id"]) & set(test["decision_group_id"]))
            train_end = pd.Timestamp(fold.train_end)
            test_start = pd.Timestamp(fold.test_start)
            self.assertGreaterEqual((test_start - train_end).days, 2)

    def test_query_spanning_sessions_fails_closed(self) -> None:
        frame = _validation_frame(sessions=10, queries_per_session=2, tickers=3)
        frame.loc[frame.index[-1], "decision_group_id"] = frame.loc[0, "decision_group_id"]
        splitter = V3PurgedWalkForwardSplit(n_splits=2, min_train_sessions=4, min_train_rows=10)
        with self.assertRaises(DataReadinessError):
            splitter.split(frame)

    def test_ticker_holdout_is_exact_and_deterministic(self) -> None:
        tickers = pd.Series([f"T{index:03d}" for index in range(20)])
        first = deterministic_ticker_holdout(tickers, fraction=0.2, seed=7)
        second = deterministic_ticker_holdout(tickers.sample(frac=1, random_state=9), fraction=0.2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)


def _validation_frame(*, sessions: int, queries_per_session: int, tickers: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2026, 1, 5)
    for session_offset in range(sessions):
        session = start + timedelta(days=session_offset)
        for query in range(queries_per_session):
            query_id = f"{session.isoformat()}T{query:02d}"
            for ticker in range(tickers):
                rows.append(
                    {
                        "ticker": f"T{ticker:03d}",
                        "session_date_et": session,
                        "decision_group_id": query_id,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
