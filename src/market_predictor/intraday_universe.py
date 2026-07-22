from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

INTRADAY_THEME_KEYWORDS = {
    "semis_ai_hardware": ["semiconductor", "semiconductors", "electronic components", "communication equipment"],
    "software_ai_data": ["software", "cloud", "data", "analytics", "information technology", "internet content"],
    "biotech_healthcare": ["biotechnology", "therapeutics", "pharmaceutical", "diagnostics", "medical"],
    "space_aerospace_mobility": ["aerospace", "defense", "airlines", "auto manufacturers", "auto parts"],
    "crypto_fintech_high_beta": ["capital markets", "financial data", "fintech", "asset management"],
    "consumer_high_beta": ["apparel", "restaurants", "travel", "entertainment", "gambling", "retail"],
}


def build_intraday_candidate_universe(
    raw: pd.DataFrame,
    *,
    top_n: int = 200,
    min_price: float = 2.0,
    min_volume: int = 500_000,
    min_abs_change_pct: float = 0.5,
    min_market_cap_m: float = 100.0,
) -> pd.DataFrame:
    """Rank Finviz rows for intraday trading suitability.

    The score intentionally favors current activity over company size:
    absolute daily move, traded shares, dollar volume, and sector/theme
    relevance for high-beta intraday setups.
    """
    if raw.empty:
        return _empty_frame()
    frame = _normalize_finviz_frame(raw)
    required = {"ticker", "price", "volume", "change_pct", "market_cap_m"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Finviz frame missing required normalized columns: {missing}")
    frame = frame[
        frame["ticker"].notna()
        & frame["price"].ge(min_price)
        & frame["volume"].ge(min_volume)
        & frame["abs_change_pct"].ge(min_abs_change_pct)
        & frame["market_cap_m"].ge(min_market_cap_m)
    ].copy()
    if frame.empty:
        return _empty_frame()
    frame["dollar_volume_m"] = (frame["price"] * frame["volume"] / 1_000_000.0).round(3)
    frame["intraday_theme"] = frame.apply(_theme_for_row, axis=1)
    theme_bonus = frame["intraday_theme"].ne("other").astype(float) * 5.0
    price_band_bonus = frame["price"].between(2.0, 250.0).astype(float) * 4.0
    frame["intraday_candidate_score"] = (
        frame["abs_change_pct"].rank(pct=True).fillna(0.0) * 45.0
        + np.log1p(frame["volume"]).rank(pct=True).fillna(0.0) * 25.0
        + np.log1p(frame["dollar_volume_m"]).rank(pct=True).fillna(0.0) * 20.0
        + theme_bonus
        + price_band_bonus
    ).round(3)
    output = frame.sort_values(
        ["intraday_candidate_score", "abs_change_pct", "volume"],
        ascending=[False, False, False],
    ).head(top_n)
    columns = [
        "ticker",
        "company",
        "sector",
        "industry",
        "country",
        "market_cap_m",
        "price",
        "volume",
        "change_pct",
        "abs_change_pct",
        "dollar_volume_m",
        "intraday_theme",
        "intraday_candidate_score",
    ]
    return output[[column for column in columns if column in output.columns]].reset_index(drop=True)


def _normalize_finviz_frame(raw: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Ticker": "ticker",
        "Company": "company",
        "Sector": "sector",
        "Industry": "industry",
        "Country": "country",
        "Market Cap": "market_cap_m",
        "Price": "price",
        "Volume": "volume",
        "Change": "change_pct",
    }
    frame = raw.rename(columns={old: new for old, new in rename.items() if old in raw.columns}).copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame = frame[frame["ticker"].str.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", na=False)].copy()
    frame["market_cap_m"] = frame["market_cap_m"].map(_parse_market_cap_m)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["volume"] = frame["volume"].map(_parse_number)
    frame["change_pct"] = frame["change_pct"].map(_parse_percent)
    frame["abs_change_pct"] = frame["change_pct"].abs()
    for column in ["company", "sector", "industry", "country"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    return frame.drop_duplicates("ticker")


def _parse_number(value: Any) -> float:
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_percent(value: Any) -> float:
    text = str(value).replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_market_cap_m(value: Any) -> float:
    text = str(value).replace(",", "").strip()
    if not text:
        return float("nan")
    suffix = text[-1].upper()
    multiplier = 1.0
    if suffix == "B":
        multiplier = 1_000.0
        text = text[:-1]
    elif suffix == "M":
        multiplier = 1.0
        text = text[:-1]
    elif suffix == "K":
        multiplier = 0.001
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return float("nan")


def _theme_for_row(row: pd.Series) -> str:
    text = f"{row.get('sector', '')} {row.get('industry', '')} {row.get('company', '')}".lower()
    for theme, keywords in INTRADAY_THEME_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return theme
    return "other"


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "company",
            "sector",
            "industry",
            "country",
            "market_cap_m",
            "price",
            "volume",
            "change_pct",
            "abs_change_pct",
            "dollar_volume_m",
            "intraday_theme",
            "intraday_candidate_score",
        ]
    )
