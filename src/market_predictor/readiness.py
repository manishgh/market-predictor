"""Prediction data-readiness assessment.

This module implements the near-term readiness contract described in
`docs/catalyst_confirmation_architecture.md` section 0.1. It is intentionally
pure (no pandas / IO) so the gate logic can be unit-tested in isolation and
reused by any command that scores a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


VALID = "valid"
WARN = "warn"
INVALID = "invalid"

# Ordering of severity so a later gate can only downgrade, never upgrade.
_RANK = {VALID: 0, WARN: 1, INVALID: 2}

DEFAULT_MIN_DAILY_BARS_VALID = 250
DEFAULT_MIN_DAILY_BARS_USABLE = 60


@dataclass
class PredictionReadiness:
    """Result of assessing whether a scored row can be trusted."""

    status: str
    reasons: list[str] = field(default_factory=list)
    daily_bar_count: int = 0
    latest_price_date: str | None = None
    price_feed: str = "unknown"
    benchmark_status: str = "unknown"
    market_context_status: str = "unknown"

    def as_record(self) -> dict[str, object]:
        """Flatten to the v0 contract columns for a report row."""
        return {
            "data_readiness_status": self.status,
            "data_readiness_reasons": "; ".join(self.reasons),
            "daily_bar_count": int(self.daily_bar_count),
            "latest_price_date": self.latest_price_date,
            "price_feed": self.price_feed,
            "benchmark_status": self.benchmark_status,
            "market_context_status": self.market_context_status,
        }


def assess_daily_readiness(
    *,
    daily_bar_count: int,
    latest_price_date: str | None,
    price_feed: str,
    benchmark_present: bool,
    market_context_present: bool,
    min_daily_bars_valid: int = DEFAULT_MIN_DAILY_BARS_VALID,
    min_daily_bars_usable: int = DEFAULT_MIN_DAILY_BARS_USABLE,
) -> PredictionReadiness:
    """Assess readiness for a daily-technical model prediction.

    Gates follow the v0 contract: history depth, price provenance, benchmark
    availability, and market-context availability. Only the most severe status
    is returned, but every non-``valid`` gate contributes a reason.
    """
    reasons: list[str] = []
    status = VALID

    def downgrade(new_status: str, reason: str) -> None:
        nonlocal status
        reasons.append(reason)
        if _RANK[new_status] > _RANK[status]:
            status = new_status

    feed = (price_feed or "unknown").strip().lower()

    if latest_price_date is None:
        downgrade(INVALID, "no latest price date available")

    if daily_bar_count < min_daily_bars_usable:
        downgrade(
            INVALID,
            f"daily_bar_count {daily_bar_count} < {min_daily_bars_usable} usable minimum",
        )
    elif daily_bar_count < min_daily_bars_valid:
        downgrade(
            WARN,
            f"daily_bar_count {daily_bar_count} < {min_daily_bars_valid}; "
            "SMA200/52-week features unreliable",
        )

    if feed in ("none", "unknown", ""):
        downgrade(WARN, f"price feed provenance is {feed or 'unknown'}")
    elif feed != "alpaca":
        downgrade(WARN, f"price feed is {feed}, not SIP-grade; volume features degraded")

    if not benchmark_present:
        downgrade(WARN, "SPY/sector benchmark features unavailable; market treated as flat")

    if not market_context_present:
        downgrade(WARN, "market-context features unavailable; global context treated as zero")

    return PredictionReadiness(
        status=status,
        reasons=reasons,
        daily_bar_count=int(daily_bar_count),
        latest_price_date=latest_price_date,
        price_feed=feed,
        benchmark_status="present" if benchmark_present else "missing",
        market_context_status="present" if market_context_present else "missing",
    )
