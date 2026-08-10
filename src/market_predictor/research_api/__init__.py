"""Research-only inspection and scoring workbench.

This package is deliberately separate from :mod:`market_predictor.api`, which
remains the promoted-model production boundary.
"""

from market_predictor.research_api.server import create_research_app

__all__ = ["create_research_app"]
