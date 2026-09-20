"""Frozen initial-fit derivative publication, never model admission."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin

PUBLICATION_SCHEMA = "market_predictor.return_relationship_publication"
REQUEST_SCHEMA = "market_predictor.return_relationship_request"
ARTIFACT_TYPE = "swing_return_relationships"
PROFILE = "technical_relationships"


class ReturnRelationshipPublicationPolicy(HoldingContract):
    schema_version: Literal["market_predictor.return_relationship_publication_config"]
    parent_publication: SourcePin
    parent_saved_row_verification: SourcePin
    feature_config: SourcePin
    predictor_failure_facts: SourcePin
    strategy_contract: SourcePin
    feature_plan_snapshot: SourcePin | None = None
    source_start: Literal["2018-05-29"] = "2018-05-29"
    source_end: Literal["2024-05-28"] = "2024-05-28"
    decision_start: Literal["2019-07-09"] = "2019-07-09"
    decision_end: Literal["2024-05-28"] = "2024-05-28"
    maximum_system_used_percent: Annotated[float, Field(gt=0, le=90)] = 90.0


@dataclass(frozen=True)
class VerifiedReturnRelationshipPublication:
    request: dict[str, Any]
    manifest: dict[str, Any]
    source_files: dict[str, str]
    parent_manifest: dict[str, Any]
    parent_path: Path
    model_columns: tuple[str, ...]
    availability_columns: dict[str, str]
    months: dict[str, dict[str, Any]]
