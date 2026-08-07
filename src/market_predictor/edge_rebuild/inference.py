"""Abstract inference engine for scoring models."""

from typing import Protocol

import pandas as pd


class InferenceEngine(Protocol):
    """Protocol for abstracting model scoring execution."""
    
    def predict(
        self,
        feature_frame: pd.DataFrame,
        requested_models: list[str] | None = None,
    ) -> dict[str, tuple[float, ...]]:
        """
        Score a feature frame using the loaded models.
        
        Args:
            feature_frame: The validated input features.
            requested_models: Optional list of model types to evaluate (e.g. ["classifier", "regressor"]). 
                              If None, the engine should evaluate all available primary models.
                              
        Returns:
            A mapping of evaluated model names to a tuple of probabilities or scores, 
            where the tuple length matches the number of rows in the feature_frame.
        """
        ...
