from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.validation import (
    V3PurgedWalkForwardSplit,
    causal_fold_training_indices,
    deterministic_stratified_ticker_holdout,
    deterministic_ticker_holdout,
    validation_row_identities,
)


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

    def test_stratified_holdout_is_deterministic_and_represents_available_strata(self) -> None:
        frame = _stratification_frame()
        first = deterministic_stratified_ticker_holdout(
            frame,
            label_columns=["target"],
            fraction=0.25,
            seed=17,
        )
        second = deterministic_stratified_ticker_holdout(
            frame.sample(frac=1, random_state=9),
            label_columns=["target"],
            fraction=0.25,
            seed=17,
        )
        self.assertEqual(first.holdout_tickers, second.holdout_tickers)
        self.assertEqual(first.ticker_summary_sha256, second.ticker_summary_sha256)
        required = first.representation_audit["required"].astype(bool)
        self.assertTrue(first.representation_audit.loc[required, "represented"].astype(bool).all())
        self.assertEqual(len(first.holdout_tickers), 3)

    def test_stratified_holdout_fails_when_whole_tickers_cannot_represent_strata(self) -> None:
        decision = pd.Timestamp("2026-01-05T15:00:00Z")
        frame = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "decision_time_utc": [decision] * 3,
                "sector": ["pair_ab", "pair_ab", "singleton"],
                "market_cap_bucket": ["singleton", "pair_bc", "pair_bc"],
                "liquidity_bucket": ["pair_ac", "singleton", "pair_ac"],
            }
        )
        with self.assertRaisesRegex(DataReadinessError, "cannot represent required strata"):
            deterministic_stratified_ticker_holdout(
                frame,
                label_columns=[],
                fraction=0.34,
                seed=3,
            )

    def test_causal_fold_filters_unmatured_labels_and_persists_unique_row_ids(self) -> None:
        frame = _validation_frame(sessions=12, queries_per_session=1, tickers=3)
        decision = pd.to_datetime(frame["session_date_et"], utc=True) + pd.Timedelta(hours=15)
        frame["decision_time_utc"] = decision
        frame["label_available_at_utc"] = decision + pd.Timedelta(days=2)
        frame["row_identity"] = validation_row_identities(frame)
        self.assertTrue(frame["row_identity"].is_unique)
        splitter = V3PurgedWalkForwardSplit(
            n_splits=2,
            embargo_sessions=1,
            min_train_sessions=5,
            min_train_rows=10,
        )
        fold = splitter.split(frame)[0]
        train, max_label, min_test = causal_fold_training_indices(
            frame,
            candidate_indices=fold.train_indices,
            test_indices=fold.test_indices,
        )
        self.assertGreater(len(train), 0)
        self.assertLess(max_label, min_test)
        poisoned = frame.copy()
        poisoned.loc[fold.test_indices, "label_available_at_utc"] = pd.Timestamp(
            datetime(2030, 1, 1, tzinfo=UTC)
        )
        poisoned_train, poisoned_max, poisoned_min = causal_fold_training_indices(
            poisoned,
            candidate_indices=fold.train_indices,
            test_indices=fold.test_indices,
        )
        np.testing.assert_array_equal(train, poisoned_train)
        self.assertEqual(max_label, poisoned_max)
        self.assertEqual(min_test, poisoned_min)


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


def _stratification_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decision = pd.Timestamp("2026-01-05T15:00:00Z")
    for ticker_index in range(12):
        for observation in range(4):
            rows.append(
                {
                    "ticker": f"T{ticker_index:02d}",
                    "decision_time_utc": decision + pd.Timedelta(days=observation),
                    "sector": "technology" if ticker_index % 2 == 0 else "healthcare",
                    "market_cap_bucket": "large" if ticker_index < 6 else "mid",
                    "liquidity_bucket": "high" if ticker_index % 3 else "medium",
                    "target": int((ticker_index + observation) % 3 == 0),
                    "event_count_3d": int((ticker_index + observation) % 4 == 0),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
