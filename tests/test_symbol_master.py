"""R3 P1-8: canonical symbol master resolves aliases/delistings/reuse over time."""

from __future__ import annotations

import unittest
from datetime import date

from market_predictor.canonical.symbol_master import DELISTED, SymbolMaster, SymbolRecord


def _master() -> SymbolMaster:
    return SymbolMaster(
        [
            # Rename: the same security is FB then META.
            SymbolRecord("sec-meta", "FB", date(2012, 5, 18), date(2022, 6, 9)),
            SymbolRecord("sec-meta", "META", date(2022, 6, 9), None),
            # Delisting: the interval closes.
            SymbolRecord("sec-xyz", "XYZ", date(2015, 1, 1), date(2019, 3, 1), status=DELISTED),
            # Ticker reuse: T belonged to two different securities at different times.
            SymbolRecord("sec-t-old", "T", date(2000, 1, 1), date(2010, 1, 1)),
            SymbolRecord("sec-t-new", "T", date(2011, 1, 1), None),
        ]
    )


class SymbolMasterTest(unittest.TestCase):
    def test_resolves_rename_delisting_and_reuse(self) -> None:
        master = _master()
        self.assertEqual(master.resolve("FB", date(2020, 1, 1)), "sec-meta")
        self.assertEqual(master.resolve("META", date(2023, 1, 1)), "sec-meta")
        self.assertIsNone(master.resolve("META", date(2020, 1, 1)))  # META did not exist yet
        self.assertEqual(master.resolve("XYZ", date(2018, 1, 1)), "sec-xyz")
        self.assertIsNone(master.resolve("XYZ", date(2020, 1, 1)))  # after delisting
        self.assertEqual(master.resolve("T", date(2005, 1, 1)), "sec-t-old")
        self.assertEqual(master.resolve("T", date(2015, 1, 1)), "sec-t-new")
        self.assertIsNone(master.resolve("UNKNOWN", date(2020, 1, 1)))

    def test_hash_is_deterministic(self) -> None:
        self.assertEqual(_master().sha256(), _master().sha256())

    def test_overlapping_symbol_intervals_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SymbolMaster(
                [
                    SymbolRecord("sec-a", "DUP", date(2010, 1, 1), None),
                    SymbolRecord("sec-b", "DUP", date(2015, 1, 1), None),  # two securities claim DUP at once
                ]
            )


if __name__ == "__main__":
    unittest.main()
