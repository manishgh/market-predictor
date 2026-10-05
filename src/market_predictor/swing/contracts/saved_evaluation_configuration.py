"""Explicit archived configuration evidence for two saved research runs only."""
from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin

GitObject = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class RecoveredEvaluationConfiguration(HoldingContract):
    original_logical: SourcePin
    original_artifact: SourcePin
    current: SourcePin
    git_commit: GitObject
    git_blob: GitObject

    @model_validator(mode="after")
    def identical_recovered_bytes(self) -> Self:
        if self.original_logical.sha256 != self.original_artifact.sha256:
            raise ValueError("recovered bytes must retain the original saved configuration SHA256")
        return self


class SavedEvaluationConfigurationEvidence(HoldingContract):
    schema_version: Literal["market_predictor.saved_evaluation_configuration"]
    scope: Literal["saved_initial_fit_evaluation_only"]
    runs: tuple[SourcePin, SourcePin]
    strategy: RecoveredEvaluationConfiguration
    temporal: RecoveredEvaluationConfiguration
    source_configuration_report: SourcePin

    @model_validator(mode="after")
    def distinct_runs(self) -> Self:
        if self.runs[0].path == self.runs[1].path or self.runs[0].sha256 == self.runs[1].sha256:
            raise ValueError("saved configuration evidence requires two distinct retained runs")
        return self
