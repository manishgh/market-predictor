"""Canonical model-serving route contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ServingRoute:
    repository: Path
    attestation_trust_store: Path
    promotion_gate_policy_sha256: str = ""
    bar_timeframe: str = "unknown"
    curated_dataset: Path | None = None
    estimated_resident_gib: float = 0.5
    max_model_bytes: int = 512 * 1024 * 1024
    max_feature_bytes: int = 512 * 1024 * 1024
    max_feature_rows: int = 250_000
