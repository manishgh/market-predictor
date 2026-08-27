from __future__ import annotations


def canonical_symbol(symbol: str) -> str:
    cleaned = str(symbol or "").upper().strip()
    return cleaned.replace(".", "-")


def normalized_ticker(value: str) -> str:
    ticker = value.strip().upper().replace("/", ".")
    if not ticker or len(ticker) > 16 or not all(character.isalnum() or character in ".-" for character in ticker):
        raise ValueError("ticker must be a valid normalized US-listed symbol")
    return ticker
