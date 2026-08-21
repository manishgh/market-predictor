from typing import Any

import pandas as pd


def _parse_date(value: Any, label: str) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value, tz="UTC")


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lower = str(value).lower()
    if lower == "true":
        return True
    elif lower == "false":
        return False
    raise ValueError(f"Invalid boolean value: {value}")
