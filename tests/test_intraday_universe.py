from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.intraday_universe import build_intraday_candidate_universe


class IntradayUniverseTests(unittest.TestCase):
    def test_ranks_high_volume_high_abs_change_candidates(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Company": "Alpha Semiconductor",
                    "Sector": "Technology",
                    "Industry": "Semiconductors",
                    "Country": "USA",
                    "Market Cap": "1500",
                    "Price": "25",
                    "Volume": "5000000",
                    "Change": "8.0%",
                },
                {
                    "Ticker": "BBB",
                    "Company": "Quiet Utility",
                    "Sector": "Utilities",
                    "Industry": "Utilities",
                    "Country": "USA",
                    "Market Cap": "5000",
                    "Price": "50",
                    "Volume": "700000",
                    "Change": "0.1%",
                },
                {
                    "Ticker": "CCC",
                    "Company": "Cloud Data Software",
                    "Sector": "Technology",
                    "Industry": "Software",
                    "Country": "USA",
                    "Market Cap": "2B",
                    "Price": "45",
                    "Volume": "4000000",
                    "Change": "-6.5%",
                },
            ]
        )

        result = build_intraday_candidate_universe(raw, top_n=2)

        self.assertEqual(result["ticker"].tolist(), ["AAA", "CCC"])
        self.assertIn("semis_ai_hardware", result["intraday_theme"].tolist())
        self.assertIn("software_ai_data", result["intraday_theme"].tolist())


if __name__ == "__main__":
    unittest.main()
