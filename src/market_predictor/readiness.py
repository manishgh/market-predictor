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
DEFAULT_MIN_INTRADAY_BARS_VALID = 130
DEFAULT_MIN_INTRADAY_BARS_USABLE = 30

_FULL_COVERAGE_FEEDS = {"sip", "alpaca_sip", "consolidated", "cta_utp"}
_PARTIAL_FEEDS = {"iex"}


@dataclass
class PredictionReadiness:
    """Result of assessing whether a scored row can be trusted."""

    status: str
    reasons: list[str] = field(default_factory=list)
    timeframe: str = "daily"
    daily_bar_count: int = 0
    intraday_bar_count: int = 0
    required_bar_count: int = 0
    latest_price_date: str | None = None
    price_feed: str = "unknown"
    benchmark_status: str = "unknown"
    market_context_status: str = "unknown"
    model_status: str = "unknown"
    source_status: str = "unknown"

    def as_record(self) -> dict[str, object]:
        """Flatten to the v0 contract columns for a report row."""
        return {
            "data_readiness_status": self.status,
            "data_readiness_reasons": "; ".join(self.reasons),
            "timeframe": self.timeframe,
            "daily_bar_count": int(self.daily_bar_count),
            "intraday_bar_count": int(self.intraday_bar_count),
            "required_bar_count": int(self.required_bar_count),
            "latest_price_date": self.latest_price_date,
            "price_feed": self.price_feed,
            "benchmark_status": self.benchmark_status,
            "market_context_status": self.market_context_status,
            "model_status": self.model_status,
            "source_status": self.source_status,
        }


def assess_daily_readiness(
    *,
    daily_bar_count: int,
    latest_price_date: str | None,
    price_feed: str,
    benchmark_present: bool,
    market_context_present: bool,
    model_status: str = "promoted",
    required_sources: set[str] | None = None,
    available_sources: set[str] | None = None,
    news_candle_mismatch_count: int = 0,
    stale_cache: bool = False,
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
    elif feed in _PARTIAL_FEEDS:
        downgrade(INVALID, f"price feed is {feed}; consolidated volume features are invalid")
    elif feed not in _FULL_COVERAGE_FEEDS:
        downgrade(WARN, f"price feed tier is {feed}; SIP coverage is not proven")

    if not benchmark_present:
        downgrade(WARN, "SPY/sector benchmark features unavailable; market treated as flat")

    if not market_context_present:
        downgrade(WARN, "market-context features unavailable; global context treated as zero")

    normalized_model_status = (model_status or "unknown").strip().lower()
    if normalized_model_status != "promoted":
        downgrade(INVALID, f"model status is {normalized_model_status}; promoted model required")

    required = {source.strip().lower() for source in required_sources or set() if source.strip()}
    available = {source.strip().lower() for source in available_sources or set() if source.strip()}
    missing_sources = sorted(required - available)
    if missing_sources:
        downgrade(INVALID, f"missing required source families: {', '.join(missing_sources)}")

    if news_candle_mismatch_count > 0:
        downgrade(INVALID, f"news/candle mismatches detected: {news_candle_mismatch_count}")

    if stale_cache:
        downgrade(INVALID, "stale cache present in prediction inputs")

    return PredictionReadiness(
        status=status,
        reasons=reasons,
        timeframe="daily",
        daily_bar_count=int(daily_bar_count),
        required_bar_count=int(min_daily_bars_valid),
        latest_price_date=latest_price_date,
        price_feed=feed,
        benchmark_status="present" if benchmark_present else "missing",
        market_context_status="present" if market_context_present else "missing",
        model_status=normalized_model_status,
        source_status="missing" if missing_sources else "present",
    )


def assess_intraday_readiness(
    *,
    intraday_bar_count: int,
    latest_price_timestamp: str | None,
    price_feed: str,
    benchmark_present: bool,
    market_context_present: bool,
    model_status: str = "promoted",
    required_sources: set[str] | None = None,
    available_sources: set[str] | None = None,
    news_candle_mismatch_count: int = 0,
    stale_cache: bool = False,
    min_intraday_bars_valid: int = DEFAULT_MIN_INTRADAY_BARS_VALID,
    min_intraday_bars_usable: int = DEFAULT_MIN_INTRADAY_BARS_USABLE,
) -> PredictionReadiness:
    """Assess a short-horizon prediction without applying daily-history gates."""
    reasons: list[str] = []
    status = VALID

    def downgrade(new_status: str, reason: str) -> None:
        nonlocal status
        reasons.append(reason)
        if _RANK[new_status] > _RANK[status]:
            status = new_status

    feed = (price_feed or "unknown").strip().lower()
    if latest_price_timestamp is None:
        downgrade(INVALID, "no latest intraday price timestamp available")

    if intraday_bar_count < min_intraday_bars_usable:
        downgrade(
            INVALID,
            f"intraday_bar_count {intraday_bar_count} < {min_intraday_bars_usable} usable minimum",
        )
    elif intraday_bar_count < min_intraday_bars_valid:
        downgrade(
            WARN,
            f"intraday_bar_count {intraday_bar_count} < {min_intraday_bars_valid}; indicators not fully stabilized",
        )

    if feed in ("none", "unknown", ""):
        downgrade(WARN, f"price feed provenance is {feed or 'unknown'}")
    elif feed in _PARTIAL_FEEDS:
        downgrade(INVALID, f"price feed is {feed}; consolidated volume features are invalid")
    elif feed not in _FULL_COVERAGE_FEEDS:
        downgrade(WARN, f"price feed tier is {feed}; SIP coverage is not proven")

    if not benchmark_present:
        downgrade(WARN, "SPY/QQQ benchmark features unavailable")
    if not market_context_present:
        downgrade(WARN, "market-context features unavailable; global context treated as zero")

    normalized_model_status = (model_status or "unknown").strip().lower()
    if normalized_model_status != "promoted":
        downgrade(INVALID, f"model status is {normalized_model_status}; promoted model required")

    required = {source.strip().lower() for source in required_sources or set() if source.strip()}
    available = {source.strip().lower() for source in available_sources or set() if source.strip()}
    missing_sources = sorted(required - available)
    if missing_sources:
        downgrade(INVALID, f"missing required source families: {', '.join(missing_sources)}")
    if news_candle_mismatch_count > 0:
        downgrade(INVALID, f"news/candle mismatches detected: {news_candle_mismatch_count}")
    if stale_cache:
        downgrade(INVALID, "stale cache present in prediction inputs")

    return PredictionReadiness(
        status=status,
        reasons=reasons,
        timeframe="intraday",
        intraday_bar_count=int(intraday_bar_count),
        required_bar_count=int(min_intraday_bars_valid),
        latest_price_date=latest_price_timestamp,
        price_feed=feed,
        benchmark_status="present" if benchmark_present else "missing",
        market_context_status="present" if market_context_present else "missing",
        model_status=normalized_model_status,
        source_status="missing" if missing_sources else "present",
    )
