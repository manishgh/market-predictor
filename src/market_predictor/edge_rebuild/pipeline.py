"""Declarative Feature Pipeline for executing feature steps consistently."""

from typing import Protocol

import pandas as pd


class FeatureStep(Protocol):
    """A single transformation step in a feature pipeline."""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the feature transformation to the dataset."""
        ...


class FeaturePipeline:
    """Executes a sequence of FeatureSteps on a DataFrame."""
    
    def __init__(self, steps: list[FeatureStep]):
        self.steps = steps

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the data through all pipeline steps sequentially."""
        for step in self.steps:
            data = step.transform(data)
        return data
