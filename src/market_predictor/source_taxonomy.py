"""Lightweight source-family normalization shared by offline and serving code."""
from __future__ import annotations




def source_family_for_source(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("alpaca"):
        return "alpaca"
    if raw.startswith("sec"):
        return "sec"
    if raw.startswith("finviz"):
        return "finviz"
    return raw.split(":", 1)[0] if raw else "unknown"
