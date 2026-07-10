from __future__ import annotations

from datetime import datetime, timedelta
import unittest

import pandas as pd

from market_predictor.entry_exit import (
    EntryExitLabelConfig,
    _drop_features_without_fold_training_coverage,
    _feature_allowed_for_set,
    build_entry_exit_dataset,
)


class EntryExitDatasetTests(unittest.TestCase):
    def test_labels_next_open_target_first_without_same_bar_leakage(self) -> None:
        start = datetime(2026, 1, 1, 9, 30)
        rows = [
            _row("RGTI", start + timedelta(days=0), 10.0, 10.2, 9.8, 10.0),
            _row("RGTI", start + timedelta(days=1), 10.0, 11.7, 9.6, 11.4),
            _row("RGTI", start + timedelta(days=2), 11.4, 11.6, 10.9, 11.2),
            _row("RGTI", start + timedelta(days=3), 11.2, 11.5, 10.8, 11.1),
        ]
        dataset, audit = build_entry_exit_dataset(
            pd.DataFrame(rows),
            config=EntryExitLabelConfig(
                horizon_bars=2,
                take_profit_atr=1.0,
                stop_loss_atr=1.0,
                min_rows_per_ticker=3,
                min_labeled_rows_per_ticker=1,
            ),
        )
        first = dataset.sort_values("date").iloc[0]
        self.assertEqual(first["entry_exit_outcome_2b"], "target_first")
        self.assertEqual(int(first["target_entry_success_2b"]), 1)
        self.assertEqual(int(first["target_exit_risk_2b"]), 0)
        self.assertTrue(bool(audit.loc[audit["ticker"].eq("RGTI"), "model_eligible"].iloc[0]))

    def test_labels_stop_first_as_exit_risk(self) -> None:
        start = datetime(2026, 1, 1, 9, 30)
        rows = [
            _row("MXL", start + timedelta(days=0), 20.0, 20.3, 19.7, 20.0),
            _row("MXL", start + timedelta(days=1), 20.0, 20.2, 18.4, 18.8),
            _row("MXL", start + timedelta(days=2), 18.8, 19.0, 18.3, 18.5),
            _row("MXL", start + timedelta(days=3), 18.5, 18.8, 18.2, 18.4),
        ]
        dataset, _ = build_entry_exit_dataset(
            pd.DataFrame(rows),
            config=EntryExitLabelConfig(
                horizon_bars=2,
                take_profit_atr=1.0,
                stop_loss_atr=1.0,
                min_rows_per_ticker=3,
                min_labeled_rows_per_ticker=1,
            ),
        )
        first = dataset.sort_values("date").iloc[0]
        self.assertEqual(first["entry_exit_outcome_2b"], "stop_first")
        self.assertEqual(int(first["target_entry_success_2b"]), 0)
        self.assertEqual(int(first["target_exit_risk_2b"]), 1)

    def test_readiness_removes_short_histories(self) -> None:
        start = datetime(2026, 1, 1, 9, 30)
        rows = [_row("SHORT", start + timedelta(days=idx), 10, 10.5, 9.5, 10.1) for idx in range(4)]
        dataset, audit = build_entry_exit_dataset(
            pd.DataFrame(rows),
            config=EntryExitLabelConfig(horizon_bars=2, min_rows_per_ticker=10, min_labeled_rows_per_ticker=1),
        )
        self.assertTrue(dataset.empty)
        self.assertFalse(bool(audit.loc[audit["ticker"].eq("SHORT"), "model_eligible"].iloc[0]))

    def test_intraday_labels_do_not_cross_sessions_by_default(self) -> None:
        rows = [
            _row("RGTI", datetime(2026, 1, 1, 15, 0), 10.0, 10.2, 9.8, 10.0),
            _row("RGTI", datetime(2026, 1, 1, 16, 0), 10.0, 10.1, 9.9, 10.0),
            _row("RGTI", datetime(2026, 1, 2, 9, 30), 10.0, 12.0, 9.9, 11.5),
            _row("RGTI", datetime(2026, 1, 2, 10, 30), 11.5, 11.8, 11.0, 11.6),
        ]
        dataset, _ = build_entry_exit_dataset(
            pd.DataFrame(rows),
            config=EntryExitLabelConfig(
                horizon_bars=3,
                take_profit_atr=1.0,
                stop_loss_atr=1.0,
                min_rows_per_ticker=3,
                min_labeled_rows_per_ticker=1,
                bar_kind="1h",
                allow_overnight=False,
            ),
        )

        first = dataset.sort_values("date").iloc[0]
        self.assertEqual(first["entry_exit_outcome_3b"], "timeout")
        self.assertEqual(int(first["target_entry_success_3b"]), 0)

    def test_entry_features_do_not_use_next_open_prices(self) -> None:
        start = datetime(2026, 1, 1, 9, 30)
        rows = [
            _row("RGTI", start + timedelta(days=0), 10.0, 10.2, 9.8, 10.0),
            _row("RGTI", start + timedelta(days=1), 100.0, 101.0, 99.0, 100.5),
            _row("RGTI", start + timedelta(days=2), 100.5, 101.0, 100.0, 100.7),
            _row("RGTI", start + timedelta(days=3), 100.7, 101.0, 100.0, 100.8),
        ]
        dataset, _ = build_entry_exit_dataset(
            pd.DataFrame(rows),
            config=EntryExitLabelConfig(horizon_bars=2, min_rows_per_ticker=3, min_labeled_rows_per_ticker=1),
        )

        first = dataset.sort_values("date").iloc[0]
        self.assertAlmostEqual(float(first["setup_stop_pct"]), 0.1)
        self.assertNotIn("entry_stop_pct", dataset.columns)
        self.assertNotIn("entry_target_pct", dataset.columns)

    def test_feature_set_classifies_catalyst_and_technical_features(self) -> None:
        self.assertTrue(_feature_allowed_for_set("news_count_2h", "catalyst"))
        self.assertTrue(_feature_allowed_for_set("source_count_seeking_alpha_1d", "catalyst"))
        self.assertFalse(_feature_allowed_for_set("return_1d", "catalyst"))
        self.assertTrue(_feature_allowed_for_set("return_1d", "technical"))
        self.assertFalse(_feature_allowed_for_set("market_context_news_count_1d", "technical"))

    def test_sparse_features_are_removed_when_missing_in_training_fold(self) -> None:
        frame = pd.DataFrame(
            {
                "stable": [1.0] * 8,
                "late_only": [None, None, None, None, 1.0, 1.0, 1.0, 1.0],
            }
        )
        splits = [(list(range(0, 4)), list(range(4, 6))), (list(range(0, 6)), list(range(6, 8)))]

        kept, excluded = _drop_features_without_fold_training_coverage(frame, ["stable", "late_only"], splits)

        self.assertEqual(kept, ["stable"])
        self.assertEqual(excluded, ["late_only"])


def _row(ticker: str, date: datetime, open_: float, high: float, low: float, close: float) -> dict[str, object]:
    return {
        "ticker": ticker,
        "date": date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100_000,
        "atr_14": 1.0,
        "atr_pct_14": 0.1,
    }


if __name__ == "__main__":
    unittest.main()
