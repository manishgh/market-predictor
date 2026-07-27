from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from market_predictor.intraday.dataset import _grid_transition_valid


class IntradayGridContinuityTests(unittest.TestCase):
    def test_consecutive_exchange_sessions_are_continuous(self) -> None:
        data = pd.DataFrame(
            {
                "session_date_et": [
                    date(2026, 6, 5),
                    date(2026, 6, 8),
                ],
                "session_slot": [389, 0],
            }
        )

        valid = _grid_transition_valid(data, bars_per_session=390)

        self.assertEqual(valid.tolist(), [1, 1])

    def test_missing_exchange_session_breaks_continuity(self) -> None:
        data = pd.DataFrame(
            {
                "session_date_et": [
                    date(2026, 6, 5),
                    date(2026, 6, 9),
                ],
                "session_slot": [389, 0],
            }
        )

        valid = _grid_transition_valid(data, bars_per_session=390)

        self.assertEqual(valid.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
