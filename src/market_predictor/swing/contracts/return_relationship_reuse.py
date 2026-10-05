"""Explicit evidence requirements for reusing one preserved historical population."""
from __future__ import annotations

from typing import Literal

from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin


class RelationshipReusePolicy(HoldingContract):
    schema_version: Literal["market_predictor.relationship_reuse_config"]
    historical_relationship_publication: SourcePin
    historical_relationship_receipt: SourcePin
    historical_parent_publication: SourcePin
    historical_parent_receipt: SourcePin
    peer_reuse_report: SourcePin
    configuration_difference_report: SourcePin
    feature_config: SourcePin
    strategy_contract: SourcePin
    predictor_failure_facts: SourcePin | None = None
    source_start: Literal["2018-05-29"] = "2018-05-29"
    source_end: Literal["2024-05-28"] = "2024-05-28"
    decision_start: Literal["2019-07-09"] = "2019-07-09"
    decision_end: Literal["2024-05-28"] = "2024-05-28"
