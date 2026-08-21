from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PublishedIntradayDataset:
    """Verified A4.3 rows and the immutable identities that authorize them."""

    frame: pd.DataFrame
    root: Path
    feature_columns: tuple[str, ...]
    frozen_round_trip_cost_bps: float
    dataset_sha256: str
    manifest_sha256: str
    authority_sha256: str
    request_sha256: str
    transformation_sha256: str
    session_unit_inventory_sha256: str
    ordered_feature_sha256: str
    strategy_contract_sha256: str
