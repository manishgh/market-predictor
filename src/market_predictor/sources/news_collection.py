"""Alpaca transport adapter for the bounded local raw-receipt collector."""

from __future__ import annotations

from collections.abc import Callable

import requests

from market_predictor.evidence.news_collection import NewsCollectionTransportError
from market_predictor.evidence.news_exchange import NewsPageRequest, ValidatedNewsReceipt
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.news_exchange import fetch_news_receipt


def alpaca_news_receipt_fetcher(
    source: AlpacaSource, *, producer_revision: str,
) -> Callable[[NewsPageRequest], ValidatedNewsReceipt]:
    """Keep existing transport retries inside one explicit logical attempt.

    The HTTP client wraps request failures with their original requests exception.
    Response-validation and unexpected runtime errors must not become retries.
    """
    def fetch(request: NewsPageRequest) -> ValidatedNewsReceipt:
        try:
            return fetch_news_receipt(source, request, producer_revision=producer_revision)
        except RuntimeError as exc:
            if isinstance(exc.__cause__, requests.RequestException):
                raise NewsCollectionTransportError("Alpaca observed news transport failed") from exc
            raise

    return fetch
