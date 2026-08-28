from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from market_predictor.modeling.feature_pipeline import FeaturePipeline


@dataclass(frozen=True)
class _AddColumn:
    name: str
    value: int

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        output = data.copy()
        output[self.name] = self.value
        return output


@dataclass(frozen=True)
class _RequireColumn:
    name: str

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.name not in data.columns:
            raise RuntimeError(f"missing pipeline column: {self.name}")
        return data


def test_feature_pipeline_has_one_modeling_owner() -> None:
    assert FeaturePipeline.__module__ == "market_predictor.modeling.feature_pipeline"


def test_feature_pipeline_runs_steps_in_declared_order() -> None:
    pipeline = FeaturePipeline(
        [
            _AddColumn("first", 1),
            _RequireColumn("first"),
            _AddColumn("second", 2),
        ]
    )

    output = pipeline.transform(pd.DataFrame({"base": [0]}))

    assert output.to_dict(orient="list") == {
        "base": [0],
        "first": [1],
        "second": [2],
    }


def test_empty_feature_pipeline_returns_the_input_object() -> None:
    frame = pd.DataFrame({"value": [1]})

    assert FeaturePipeline([]).transform(frame) is frame


def test_feature_pipeline_does_not_swallow_step_failures() -> None:
    pipeline = FeaturePipeline([_RequireColumn("missing")])

    with pytest.raises(RuntimeError, match="missing pipeline column: missing"):
        pipeline.transform(pd.DataFrame({"value": [1]}))
