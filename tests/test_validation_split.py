from __future__ import annotations

import unittest
from datetime import date, timedelta

import pandas as pd

from market_predictor.model import DateGroupedPurgedWalkForwardSplit


class DateGroupedPurgedWalkForwardSplitTests(unittest.TestCase):
    def test_keeps_same_date_out_of_both_train_and_test(self) -> None:
        groups = []
        for offset in range(30):
            groups.extend([date(2026, 1, 1) + timedelta(days=offset)] * 5)
        x = pd.DataFrame({"value": range(len(groups))})
        splitter = DateGroupedPurgedWalkForwardSplit(n_splits=3, embargo_groups=2, min_train_size=20)

        splits = list(splitter.split(x, groups=groups))

        self.assertGreaterEqual(len(splits), 2)
        group_series = pd.Series(groups)
        for train_idx, test_idx in splits:
            train_dates = set(group_series.iloc[train_idx])
            test_dates = set(group_series.iloc[test_idx])
            self.assertFalse(train_dates & test_dates)
            self.assertLess(max(train_dates), min(test_dates))
            embargo_gap = (min(test_dates) - max(train_dates)).days
            self.assertGreaterEqual(embargo_gap, 3)

    def test_requires_groups(self) -> None:
        splitter = DateGroupedPurgedWalkForwardSplit(n_splits=2, embargo_groups=1, min_train_size=2)
        with self.assertRaises(ValueError):
            list(splitter.split(pd.DataFrame({"value": [1, 2, 3]})))


if __name__ == "__main__":
    unittest.main()
