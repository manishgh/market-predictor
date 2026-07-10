from __future__ import annotations

import unittest

from market_predictor.symbols import PROVIDER_ALPACA, PROVIDER_SEC, PROVIDER_YAHOO, canonical_symbol, provider_symbol


class SymbolMappingTests(unittest.TestCase):
    def test_canonical_symbol_normalizes_share_class_separator(self) -> None:
        self.assertEqual(canonical_symbol("brk.b"), "BRK-B")
        self.assertEqual(canonical_symbol(" BF-B "), "BF-B")

    def test_provider_symbol_formats_share_classes(self) -> None:
        self.assertEqual(provider_symbol("BRK-B", PROVIDER_ALPACA), "BRK.B")
        self.assertEqual(provider_symbol("BRK-B", PROVIDER_YAHOO), "BRK-B")
        self.assertEqual(provider_symbol("BRK-B", PROVIDER_SEC), "BRKB")


if __name__ == "__main__":
    unittest.main()
