from __future__ import annotations

import unittest

from market_predictor.core.symbols import canonical_symbol
from market_predictor.sources.provider_symbols import (
    PROVIDER_ALPACA,
    PROVIDER_FINVIZ,
    PROVIDER_SEC,
    PROVIDER_YAHOO,
    provider_symbol,
)


class SymbolMappingTests(unittest.TestCase):
    def test_canonical_symbol_normalizes_share_class_separator(self) -> None:
        self.assertEqual(canonical_symbol("brk.b"), "BRK-B")
        self.assertEqual(canonical_symbol(" BF-B "), "BF-B")

    def test_provider_symbol_formats_share_classes(self) -> None:
        self.assertEqual(provider_symbol("BRK-B", PROVIDER_ALPACA), "BRK.B")
        self.assertEqual(provider_symbol("BRK-B", PROVIDER_FINVIZ), "BRK.B")
        self.assertEqual(provider_symbol("BRK-B", PROVIDER_YAHOO), "BRK-B")
        self.assertEqual(provider_symbol("BRK-B", PROVIDER_SEC), "BRKB")
        self.assertEqual(provider_symbol("BRK.B", "unknown"), "BRK-B")


if __name__ == "__main__":
    unittest.main()
