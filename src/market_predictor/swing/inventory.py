from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.data_quality import sanitize_events_frame
from market_predictor.features import feature_date_for_timestamp, source_family_for_source
from market_predictor.resources import assert_memory_budget, memory_audit, release_process_memory

INVENTORY_SCHEMA_VERSION = "swing.research_inventory.v1"
_EVENT_SUFFIX = "_events.parquet"
_FEATURE_SUFFIXES = {1: "_daily_1d.parquet", 5: "_daily_5d.parquet"}
_VALID_COLLECTION_STATUSES = {"observed", "observed_empty"}


class SwingResearchInventoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_daily_bars: int = Field(default=1_764, ge=250)
    minimum_feature_rows_5d: int = Field(default=1_250, ge=1)
    minimum_news_months: float = Field(default=36.0, ge=0)
    minimum_first_observed_rate: float = Field(default=0.95, ge=0, le=1)
    required_feed_type: str = "sip"
    required_event_sources: tuple[str, ...] = ("alpaca",)
    require_point_in_time_membership: bool = True
    require_source_collection_evidence: bool = True
    max_alignment_errors: int = Field(default=0, ge=0)
    max_memory_gib: float = Field(default=4.0, ge=1.0, le=4.0)
    memory_headroom_gib: float = Field(default=0.75, ge=0.5, le=2.0)

    @model_validator(mode="after")
    def validate_inventory(self) -> SwingResearchInventoryConfig:
        if self.memory_headroom_gib >= self.max_memory_gib:
            raise ValueError("memory_headroom_gib must be lower than max_memory_gib")
        if self.required_feed_type.strip().lower() == "":
            raise ValueError("required_feed_type must not be empty")
        normalized_sources = tuple(source.strip().lower() for source in self.required_event_sources)
        if any(not source for source in normalized_sources):
            raise ValueError("required_event_sources must not contain empty values")
        if len(set(normalized_sources)) != len(normalized_sources):
            raise ValueError("required_event_sources must be unique")
        return self


def build_swing_research_inventory(
    *,
    raw_event_directory: Path,
    feature_directory: Path,
    memberships_path: Path | None = None,
    source_collections_path: Path | None = None,
    config: SwingResearchInventoryConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    effective = config or SwingResearchInventoryConfig()
    _require_directory(raw_event_directory, "raw event")
    _require_directory(feature_directory, "feature")
    memberships = _read_optional_frame(memberships_path)
    collections = _read_optional_frame(source_collections_path)
    event_paths = _ticker_paths(raw_event_directory, _EVENT_SUFFIX)
    feature_paths = {
        horizon: _ticker_paths(feature_directory, suffix)
        for horizon, suffix in _FEATURE_SUFFIXES.items()
    }
    tickers = sorted(set(event_paths).union(*(set(paths) for paths in feature_paths.values())))
    if not tickers:
        raise ValueError("no ticker event or daily feature parquet files were found")

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        assert_memory_budget(
            hard_budget_gib=effective.max_memory_gib,
            headroom_gib=effective.memory_headroom_gib,
            stage=f"swing research inventory {ticker}",
        )
        event_path = event_paths.get(ticker)
        one_day_path = feature_paths[1].get(ticker)
        five_day_path = feature_paths[5].get(ticker)
        event_summary, clean_events, event_error = _safe_event_summary(event_path)
        one_day_summary, _, one_day_error = _safe_feature_summary(one_day_path, horizon=1)
        five_day_summary, five_day_features, five_day_error = _safe_feature_summary(five_day_path, horizon=5)
        audit_errors = [error for error in (event_error, one_day_error, five_day_error) if error]
        alignment = _alignment_summary(clean_events, five_day_features)
        membership = _membership_summary(memberships, ticker)
        source_collection = _collection_summary(collections, ticker, effective.required_event_sources)
        feed_type = _resolve_feed_type(one_day_summary["feed_types"], five_day_summary["feed_types"])
        technical_reasons = _technical_reasons(
            daily_bar_count=max(int(one_day_summary["daily_bar_count"]), int(five_day_summary["daily_bar_count"])),
            feature_rows_5d=int(five_day_summary["feature_rows"]),
            feed_type=feed_type,
            membership_status=str(membership["point_in_time_membership_status"]),
            config=effective,
        )
        catalyst_research_reasons = _catalyst_research_reasons(
            event_summary=event_summary,
            alignment=alignment,
            config=effective,
        )
        if audit_errors:
            technical_reasons.append("ticker_input_error")
            catalyst_research_reasons.append("ticker_input_error")
        catalyst_promotion_reasons = [
            *catalyst_research_reasons,
            *_catalyst_promotion_evidence_reasons(
                event_summary=event_summary,
                collection=source_collection,
                config=effective,
            ),
        ]
        technical_eligibility = "eligible" if not technical_reasons else "ineligible"
        catalyst_research_eligibility = "eligible" if not catalyst_research_reasons else "ineligible"
        catalyst_promotion_eligibility = "eligible" if not catalyst_promotion_reasons else "ineligible"
        if technical_reasons:
            model_eligibility = "ineligible"
        elif catalyst_promotion_reasons:
            model_eligibility = "warn"
        else:
            model_eligibility = "eligible"
        eligibility_reasons = [*(f"technical:{reason}" for reason in technical_reasons)]
        eligibility_reasons.extend(f"catalyst_research:{reason}" for reason in catalyst_research_reasons)
        eligibility_reasons.extend(
            f"catalyst_promotion:{reason}"
            for reason in catalyst_promotion_reasons
            if reason not in catalyst_research_reasons
        )

        rows.append(
            {
                "schema_version": INVENTORY_SCHEMA_VERSION,
                "ticker": ticker,
                "audit_error": ";".join(audit_errors),
                "raw_source_directory": str(raw_event_directory.resolve()),
                "event_path": str(event_path.resolve()) if event_path is not None else "",
                "first_news_date": event_summary["first_news_date"],
                "last_news_date": event_summary["last_news_date"],
                "months_covered": event_summary["months_covered"],
                "event_count_raw": event_summary["event_count_raw"],
                "event_count": event_summary["event_count"],
                "duplicate_events_removed": event_summary["duplicate_events_removed"],
                "invalid_events_removed": event_summary["invalid_events_removed"],
                "future_events_removed": event_summary["future_events_removed"],
                "source_families": event_summary["source_families"],
                "first_observed_event_rate": event_summary["first_observed_event_rate"],
                "publication_time_only_event_rate": event_summary["publication_time_only_event_rate"],
                "feature_rows_1d": one_day_summary["feature_rows"],
                "feature_rows_5d": five_day_summary["feature_rows"],
                "eligible_label_rows_5d": five_day_summary["eligible_label_rows"],
                "event_feature_rows": five_day_summary["event_feature_rows"],
                "daily_bar_count": max(
                    int(one_day_summary["daily_bar_count"]),
                    int(five_day_summary["daily_bar_count"]),
                ),
                "first_daily_bar_date": _earliest(
                    one_day_summary["first_daily_bar_date"],
                    five_day_summary["first_daily_bar_date"],
                ),
                "last_daily_bar_date": _latest(
                    one_day_summary["last_daily_bar_date"],
                    five_day_summary["last_daily_bar_date"],
                ),
                "intraday_bar_count": 0,
                "news_candle_alignment_status": alignment["news_candle_alignment_status"],
                "missing_historical_feature_rows": alignment["missing_historical_feature_rows"],
                "pending_events_after_last_feature_date": alignment["pending_events_after_last_feature_date"],
                "dates_with_news_count_mismatch": alignment["dates_with_news_count_mismatch"],
                "max_abs_news_count_diff": alignment["max_abs_news_count_diff"],
                "feed_type": feed_type,
                "cache_status": "unknown",
                **membership,
                **source_collection,
                "technical_eligibility": technical_eligibility,
                "catalyst_research_eligibility": catalyst_research_eligibility,
                "catalyst_promotion_eligibility": catalyst_promotion_eligibility,
                "model_eligibility": model_eligibility,
                "eligibility_reasons": ";".join(eligibility_reasons),
            }
        )
        del clean_events, five_day_features
        release_process_memory()

    report = pd.DataFrame(rows).sort_values("ticker", kind="stable").reset_index(drop=True)
    memory = memory_audit(
        hard_budget_gib=effective.max_memory_gib,
        headroom_gib=effective.memory_headroom_gib,
    ).to_record()
    summary: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": effective.model_dump(mode="json"),
        "inputs": {
            "raw_event_directory": _directory_identity(raw_event_directory, event_paths.values()),
            "feature_directory": _directory_identity(
                feature_directory,
                [path for paths in feature_paths.values() for path in paths.values()],
            ),
            "memberships": _file_identity(memberships_path),
            "source_collections": _file_identity(source_collections_path),
        },
        "ticker_count": len(report),
        "model_eligibility": _count_values(report["model_eligibility"]),
        "technical_eligibility": _count_values(report["technical_eligibility"]),
        "catalyst_research_eligibility": _count_values(report["catalyst_research_eligibility"]),
        "catalyst_promotion_eligibility": _count_values(report["catalyst_promotion_eligibility"]),
        "total_sanitized_events": int(report["event_count"].sum()),
        "median_news_months": _finite_median(report["months_covered"]),
        "median_daily_bars": _finite_median(report["daily_bar_count"]),
        "first_observed_event_rate": _weighted_rate(
            report["event_count"],
            report["first_observed_event_rate"],
        ),
        "tickers_with_alignment_failures": int(report["news_candle_alignment_status"].eq("fail").sum()),
        "memory": memory,
    }
    return report, summary


def _event_summary(path: Path | None) -> tuple[dict[str, Any], pd.DataFrame]:
    empty = pd.DataFrame()
    if path is None:
        return {
            "first_news_date": "",
            "last_news_date": "",
            "months_covered": 0.0,
            "event_count_raw": 0,
            "event_count": 0,
            "duplicate_events_removed": 0,
            "invalid_events_removed": 0,
            "future_events_removed": 0,
            "source_families": "",
            "first_observed_event_rate": 0.0,
            "publication_time_only_event_rate": 0.0,
        }, empty
    columns = _available_columns(
        path,
        {
            "ticker",
            "timestamp",
            "source",
            "title",
            "url",
            "summary",
            "text",
            "engagement_score",
            "engagement_comments",
            "engagement_upvote_ratio",
            "raw",
            "ingested_at_utc",
            "available_at_utc",
            "availability_policy",
        },
    )
    frame = pd.read_parquet(path, columns=columns)
    clean, verification = sanitize_events_frame(frame)
    first_observed = _first_observed_mask(clean)
    first_timestamp = clean["timestamp"].min() if not clean.empty else pd.NaT
    last_timestamp = clean["timestamp"].max() if not clean.empty else pd.NaT
    months = 0.0
    if pd.notna(first_timestamp) and pd.notna(last_timestamp):
        months = max(0.0, float((last_timestamp - first_timestamp) / pd.Timedelta(days=30.4375)))
    source_families = sorted(clean["source"].map(source_family_for_source).astype(str).unique()) if not clean.empty else []
    observed_rate = float(first_observed.mean()) if len(first_observed) else 0.0
    return {
        "first_news_date": _timestamp_text(first_timestamp),
        "last_news_date": _timestamp_text(last_timestamp),
        "months_covered": round(months, 4),
        "event_count_raw": verification.rows_in,
        "event_count": verification.rows_out,
        "duplicate_events_removed": verification.duplicate_rows_removed,
        "invalid_events_removed": verification.missing_required_rows_removed,
        "future_events_removed": verification.future_timestamp_rows,
        "source_families": ",".join(source_families),
        "first_observed_event_rate": observed_rate,
        "publication_time_only_event_rate": 1.0 - observed_rate if len(clean) else 0.0,
    }, clean


def _safe_event_summary(path: Path | None) -> tuple[dict[str, Any], pd.DataFrame, str]:
    try:
        summary, frame = _event_summary(path)
        return summary, frame, ""
    except Exception as exc:
        summary, frame = _event_summary(None)
        return summary, frame, f"events:{type(exc).__name__}:{exc}"


def _feature_summary(path: Path | None, *, horizon: int) -> tuple[dict[str, Any], pd.DataFrame]:
    if path is None:
        return {
            "feature_rows": 0,
            "eligible_label_rows": 0,
            "event_feature_rows": 0,
            "daily_bar_count": 0,
            "first_daily_bar_date": "",
            "last_daily_bar_date": "",
            "feed_types": (),
        }, pd.DataFrame()
    requested = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "news_count",
        "event_count",
        "label_eligible",
        f"future_return_{horizon}d",
        "price_feed",
        "feed_type",
    }
    columns = _available_columns(path, requested)
    frame = pd.read_parquet(path, columns=columns)
    dates = pd.to_datetime(frame.get("date"), errors="coerce")
    price_columns = [column for column in ("open", "high", "low", "close", "volume") if column in frame]
    valid_bars = frame[price_columns].notna().all(axis=1) if price_columns else pd.Series(False, index=frame.index)
    label_column = "label_eligible" if "label_eligible" in frame else f"future_return_{horizon}d"
    if label_column == "label_eligible":
        eligible_labels = frame[label_column].fillna(False).astype(bool)
    elif label_column in frame:
        eligible_labels = pd.to_numeric(frame[label_column], errors="coerce").notna()
    else:
        eligible_labels = pd.Series(False, index=frame.index)
    event_column = "event_count" if "event_count" in frame else "news_count"
    event_rows = (
        int(pd.to_numeric(frame[event_column], errors="coerce").fillna(0).gt(0).sum())
        if event_column in frame
        else 0
    )
    feed_types: set[str] = set()
    for feed_column in ("price_feed", "feed_type"):
        if feed_column in frame:
            feed_types.update(
                value
                for value in frame[feed_column].dropna().astype(str).str.strip().str.lower().unique()
                if value
            )
    return {
        "feature_rows": len(frame),
        "eligible_label_rows": int(eligible_labels.sum()),
        "event_feature_rows": event_rows,
        "daily_bar_count": int(valid_bars.sum()),
        "first_daily_bar_date": _date_text(dates.min()),
        "last_daily_bar_date": _date_text(dates.max()),
        "feed_types": tuple(sorted(feed_types)),
    }, frame


def _safe_feature_summary(path: Path | None, *, horizon: int) -> tuple[dict[str, Any], pd.DataFrame, str]:
    try:
        summary, frame = _feature_summary(path, horizon=horizon)
        return summary, frame, ""
    except Exception as exc:
        summary, frame = _feature_summary(None, horizon=horizon)
        return summary, frame, f"features_{horizon}d:{type(exc).__name__}:{exc}"


def _alignment_summary(events: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    unavailable = {
        "news_candle_alignment_status": "unavailable",
        "missing_historical_feature_rows": 0,
        "pending_events_after_last_feature_date": 0,
        "dates_with_news_count_mismatch": 0,
        "max_abs_news_count_diff": 0.0,
    }
    if events.empty or features.empty or "date" not in features:
        return unavailable
    trading_dates = sorted(
        {
            value.date()
            for value in pd.to_datetime(features["date"], errors="coerce")
            if pd.notna(value)
        }
    )
    if not trading_dates:
        return unavailable
    candidates = events["timestamp"].map(feature_date_for_timestamp)
    first_date = trading_dates[0]
    last_date = trading_dates[-1]
    pending = candidates.gt(last_date)
    before_history = candidates.lt(first_date)
    assigned_dates: list[date | None] = [
        _next_trading_date(candidate, trading_dates)
        if not is_pending and not is_before_history
        else None
        for candidate, is_pending, is_before_history in zip(candidates, pending, before_history, strict=True)
    ]
    aligned = events.loc[[value is not None for value in assigned_dates]].copy()
    aligned["date"] = [value for value in assigned_dates if value is not None]
    missing_historical = int(sum(value is None for value in assigned_dates)) - int(pending.sum())
    mismatch_count = 0
    max_difference = 0.0
    if "news_count" in features:
        grouped_events = aligned.groupby("date", observed=True).size().rename("event_count_raw")
        feature_counts = features[["date", "news_count"]].copy()
        feature_counts["date"] = pd.to_datetime(feature_counts["date"], errors="coerce").dt.date
        feature_counts = feature_counts.dropna(subset=["date"]).drop_duplicates("date").set_index("date")
        joined = grouped_events.to_frame().join(feature_counts, how="left")
        differences = joined["event_count_raw"] - pd.to_numeric(joined["news_count"], errors="coerce").fillna(0)
        mismatch_count = int(differences.ne(0).sum())
        max_difference = float(differences.abs().max()) if not differences.empty else 0.0
    errors = missing_historical + mismatch_count
    return {
        "news_candle_alignment_status": "pass" if errors == 0 else "fail",
        "missing_historical_feature_rows": missing_historical,
        "pending_events_after_last_feature_date": int(pending.sum()),
        "dates_with_news_count_mismatch": mismatch_count,
        "max_abs_news_count_diff": max_difference,
    }


def _membership_summary(frame: pd.DataFrame, ticker: str) -> dict[str, Any]:
    if frame.empty or "ticker" not in frame:
        return {
            "point_in_time_membership_status": "missing",
            "membership_intervals": 0,
            "first_membership_date": "",
            "last_membership_date": "",
        }
    rows = frame[frame["ticker"].astype(str).str.upper().eq(ticker)]
    if rows.empty:
        return {
            "point_in_time_membership_status": "missing",
            "membership_intervals": 0,
            "first_membership_date": "",
            "last_membership_date": "",
        }
    required = {"effective_from_utc", "effective_to_utc"}
    has_intervals = required.issubset(rows.columns)
    has_availability = "available_at_utc" in rows and rows["available_at_utc"].notna().all()
    status = "verified" if has_intervals and has_availability else "present_unverified"
    starts = (
        pd.to_datetime(rows["effective_from_utc"], utc=True, errors="coerce")
        if "effective_from_utc" in rows
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    ends = (
        pd.to_datetime(rows["effective_to_utc"], utc=True, errors="coerce")
        if "effective_to_utc" in rows
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    return {
        "point_in_time_membership_status": status,
        "membership_intervals": len(rows),
        "first_membership_date": _timestamp_text(starts.min()),
        "last_membership_date": _timestamp_text(ends.max()),
    }


def _collection_summary(
    frame: pd.DataFrame,
    ticker: str,
    required_sources: tuple[str, ...],
) -> dict[str, Any]:
    required = tuple(source.strip().lower() for source in required_sources)
    if frame.empty or not {"ticker", "source_family", "status"}.issubset(frame.columns):
        return {
            "source_collection_evidence_status": "missing",
            "source_collection_families": "",
            "missing_required_source_collections": ",".join(required),
        }
    rows = frame[frame["ticker"].astype(str).str.upper().eq(ticker)].copy()
    rows["source_family"] = rows["source_family"].astype(str).str.strip().str.lower()
    rows["status"] = rows["status"].astype(str).str.strip().str.lower()
    families = sorted(rows["source_family"].unique())
    missing = []
    for source in required:
        source_statuses = rows.loc[rows["source_family"].eq(source), "status"]
        if source_statuses.empty or not source_statuses.isin(_VALID_COLLECTION_STATUSES).all():
            missing.append(source)
    return {
        "source_collection_evidence_status": "verified" if not missing else "incomplete",
        "source_collection_families": ",".join(families),
        "missing_required_source_collections": ",".join(missing),
    }


def _technical_reasons(
    *,
    daily_bar_count: int,
    feature_rows_5d: int,
    feed_type: str,
    membership_status: str,
    config: SwingResearchInventoryConfig,
) -> list[str]:
    reasons: list[str] = []
    if daily_bar_count < config.minimum_daily_bars:
        reasons.append(f"daily_bars={daily_bar_count}<{config.minimum_daily_bars}")
    if feature_rows_5d < config.minimum_feature_rows_5d:
        reasons.append(f"feature_rows_5d={feature_rows_5d}<{config.minimum_feature_rows_5d}")
    if feed_type != config.required_feed_type.strip().lower():
        reasons.append(f"feed_type={feed_type or 'unknown'}!={config.required_feed_type.strip().lower()}")
    if config.require_point_in_time_membership and membership_status != "verified":
        reasons.append(f"point_in_time_membership={membership_status}")
    return reasons


def _catalyst_research_reasons(
    *,
    event_summary: dict[str, Any],
    alignment: dict[str, Any],
    config: SwingResearchInventoryConfig,
) -> list[str]:
    reasons: list[str] = []
    if float(event_summary["months_covered"]) < config.minimum_news_months:
        reasons.append(f"news_months={event_summary['months_covered']}<{config.minimum_news_months}")
    if alignment["news_candle_alignment_status"] != "pass":
        reasons.append(f"news_candle_alignment={alignment['news_candle_alignment_status']}")
    errors = int(alignment["missing_historical_feature_rows"]) + int(alignment["dates_with_news_count_mismatch"])
    if errors > config.max_alignment_errors:
        reasons.append(f"alignment_errors={errors}>{config.max_alignment_errors}")
    return reasons


def _catalyst_promotion_evidence_reasons(
    *,
    event_summary: dict[str, Any],
    collection: dict[str, Any],
    config: SwingResearchInventoryConfig,
) -> list[str]:
    reasons: list[str] = []
    if float(event_summary["first_observed_event_rate"]) < config.minimum_first_observed_rate:
        reasons.append(
            f"first_observed_rate={float(event_summary['first_observed_event_rate']):.4f}"
            f"<{config.minimum_first_observed_rate:.4f}"
        )
    if config.require_source_collection_evidence and collection["source_collection_evidence_status"] != "verified":
        reasons.append(f"source_collection_evidence={collection['source_collection_evidence_status']}")
    return reasons


def _ticker_paths(directory: Path, suffix: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        ticker = path.name[: -len(suffix)].strip().upper()
        if not ticker:
            continue
        if ticker in paths:
            raise ValueError(f"duplicate ticker file for {ticker} in {directory}")
        paths[ticker] = path
    return paths


def _read_optional_frame(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise ValueError(f"optional inventory input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported inventory input format: {path.suffix}")


def _available_columns(path: Path, requested: set[str]) -> list[str]:
    available = set(cast(Any, pq.read_schema)(path).names)
    return sorted(requested.intersection(available))


def _first_observed_mask(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=bool)
    policy = (
        events["availability_policy"].fillna("").astype(str).str.strip().str.lower()
        if "availability_policy" in events
        else pd.Series("", index=events.index)
    )
    observed_at = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns, UTC]")
    for column in ("ingested_at_utc", "available_at_utc"):
        if column in events:
            parsed = pd.to_datetime(events[column], utc=True, errors="coerce")
            observed_at = observed_at.fillna(parsed)
    return policy.eq("observed") & observed_at.notna()


def _resolve_feed_type(*groups: object) -> str:
    values: set[str] = set()
    for group in groups:
        if isinstance(group, tuple | list | set):
            values.update(str(value).strip().lower() for value in group if str(value).strip())
    if not values:
        return "unknown"
    if len(values) == 1:
        return next(iter(values))
    return "mixed:" + ",".join(sorted(values))


def _next_trading_date(candidate: date, trading_dates: list[date]) -> date | None:
    index = bisect_left(trading_dates, candidate)
    return trading_dates[index] if index < len(trading_dates) else None


def _earliest(left: object, right: object) -> str:
    values = [str(value) for value in (left, right) if str(value)]
    return min(values) if values else ""


def _latest(left: object, right: object) -> str:
    values = [str(value) for value in (left, right) if str(value)]
    return max(values) if values else ""


def _timestamp_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).isoformat())


def _date_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date().isoformat())


def _count_values(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _finite_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else None


def _weighted_rate(counts: pd.Series, rates: pd.Series) -> float:
    numeric_counts = pd.to_numeric(counts, errors="coerce").fillna(0)
    total = float(numeric_counts.sum())
    if total <= 0:
        return 0.0
    numeric_rates = pd.to_numeric(rates, errors="coerce").fillna(0)
    return float((numeric_counts * numeric_rates).sum() / total)


def _require_directory(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise ValueError(f"{name} directory does not exist: {path}")


def _file_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _directory_identity(directory: Path, paths: Any) -> dict[str, Any]:
    records = []
    for path in sorted(set(paths), key=lambda item: item.name):
        records.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(directory.resolve()),
        "file_count": len(records),
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
