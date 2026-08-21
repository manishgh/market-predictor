"""Memory assertion utilities for edge_rebuild."""
from __future__ import annotations



from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
)


def guard_memory(
    stage: str,
    *,
    peak: bool,
    hard_budget_gib: float,
    headroom_gib: float,
) -> None:
    function = assert_peak_memory_budget if peak else assert_memory_budget
    function(
        hard_budget_gib=hard_budget_gib,
        headroom_gib=headroom_gib,
        stage=stage,
    )
