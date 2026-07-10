from __future__ import annotations


PROVIDER_ALPACA = "alpaca"
PROVIDER_YAHOO = "yahoo"
PROVIDER_FINVIZ = "finviz"
PROVIDER_SEEKING_ALPHA = "seeking_alpha"
PROVIDER_SEC = "sec"


def canonical_symbol(symbol: str) -> str:
    cleaned = str(symbol or "").upper().strip()
    return cleaned.replace(".", "-")


def provider_symbol(symbol: str, provider: str) -> str:
    canonical = canonical_symbol(symbol)
    normalized_provider = provider.strip().lower()
    if normalized_provider in {PROVIDER_ALPACA, PROVIDER_FINVIZ, PROVIDER_SEEKING_ALPHA}:
        return canonical.replace("-", ".")
    if normalized_provider == PROVIDER_YAHOO:
        return canonical.replace("-", "-")
    if normalized_provider == PROVIDER_SEC:
        return canonical.replace("-", "")
    return canonical
