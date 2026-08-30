"""Horizon-neutral integer outcomes used by causal label builders."""

from __future__ import annotations

from typing import Final

TARGET_HIT: Final = 1
STOP_HIT: Final = -1
TIMEOUT: Final = 0
RANK_TOP: Final = 1
RANK_BOTTOM: Final = -1
RANK_MIDDLE: Final = 0
