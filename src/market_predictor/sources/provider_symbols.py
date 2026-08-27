"""Provider-specific ticker formatting."""
from __future__ import annotations

from market_predictor.core.symbols import canonical_symbol

PROVIDER_ALPACA = "alpaca"
PROVIDER_YAHOO = "yahoo"
PROVIDER_FINVIZ = "finviz"
PROVIDER_SEC = "sec"


def provider_symbol(symbol: str, provider: str) -> str:
    canonical = canonical_symbol(symbol)
    normalized_provider = provider.strip().lower()
    if normalized_provider in {PROVIDER_ALPACA, PROVIDER_FINVIZ}:
        return canonical.replace("-", ".")
    if normalized_provider == PROVIDER_YAHOO:
        return canonical.replace("-", "-")
    if normalized_provider == PROVIDER_SEC:
        return canonical.replace("-", "")
    return canonical
