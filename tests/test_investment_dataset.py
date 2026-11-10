"""Exact prior-session snapshot joins, null preservation and causal admission poisons."""
import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment import dataset as module
from market_predictor.investment.fundamentals import FEATURE_NAMES
from market_predictor.research.news_decision_features import NEWS_COLUMNS

COLUMNS = (*(f"technical_{number:03}" for number in range(124)), *NEWS_COLUMNS)
CLOCKS = {name: f"available_at_{name}" for name in COLUMNS}


def decision(security: str = "issuer:one", *, status: str = "eligible") -> dict[str, Any]:
    return {"decision_id": json_sha256({"security_id": security, "decision_session": "2024-05-28"}),
        "security_id": security, "ticker": "ONE", "cik": "0000000001", "sector": "Information Technology",
        "sector_benchmark": "XLK", "decision_session": "2024-05-28", "decision_time_utc": "2024-05-28T20:00:00Z",
        "status": status, "roster_status": status, "qualification_reasons": []}


def source_row(security: str = "issuer:one", *, session: date = date(2024, 5, 24)) -> dict[str, Any]:
    cutoff = swing_prediction_cutoffs(pd.Series([session])).iloc[0]
    return {"decision_id": f"original-swing-{security}-{session}", "security_id": security, "ticker": "ONE",
        "session_date_et": session, "decision_time_utc": cutoff, "feature_eligible": False,
        "technical_missing_reasons": json.dumps(["original_partial_technical_snapshot"]),
        **{name: np.float32(0.125 + index / 1000) for index, name in enumerate(COLUMNS)},
        **dict.fromkeys(CLOCKS.values(), cutoff)}


def source_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    for name in COLUMNS:
        frame[name] = frame[name].astype("float32")
    return frame


def joined(*, source: pd.DataFrame | None = None, rows: list[dict[str, Any]] | None = None,
    financial: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    return module.join_lagged_features(decisions=rows or [decision()], source=source_frame([source_row()]) if source is None else source,
        model_columns=COLUMNS, availability_columns=CLOCKS, financial=financial or {})


def test_exact_previous_exchange_session_values_and_both_identity_namespaces_survive() -> None:
    source = source_frame([source_row(), source_row("issuer:peer")])
    # Current investment peer set must not renormalize precomputed 586-peer features.
    source.loc[1, COLUMNS[0]] = np.float32(999)
    result = joined(source=source)
    assert module.previous_session(date(2024, 5, 28)) == date(2024, 5, 24)  # weekend + Memorial Day
    assert result.loc[0, "source_session"] == date(2024, 5, 24)
    assert result.loc[0, "source_decision_id"] == source.loc[0, "decision_id"] != result.loc[0, "decision_id"]
    assert result.loc[0, "source_cutoff_utc"] == pd.Timestamp("2024-05-24T22:00:00Z")
    assert result.loc[0, "feature_eligible"]
    assert not result.loc[0, "source_feature_eligible"]  # original flag is diagnostic only
    assert result.loc[0, "source_feature_missing_reasons"] == source.loc[0, "technical_missing_reasons"]
    pd.testing.assert_series_equal(result.loc[0, list(COLUMNS)], source.loc[0, list(COLUMNS)], check_names=False)
    assert all(result[name].dtype == source[name].dtype for name in COLUMNS)
    assert result.loc[0, list(FEATURE_NAMES)].isna().all()
    assert len(module.model_profiles(COLUMNS)["technical_relationships"]) == 124
    assert len(module.model_profiles(COLUMNS)["all_features"]) == 215


def test_absent_exact_previous_snapshot_does_not_use_older_or_same_day_source() -> None:
    source = source_frame([source_row(session=date(2024, 5, 23)), source_row(session=date(2024, 5, 28))])
    result = joined(source=source)
    assert not result.loc[0, "feature_eligible"]
    assert result.loc[0, "source_decision_id"] is None
    assert result.loc[0, "source_feature_missing_reasons"] is None
    assert result.loc[0, list(COLUMNS)].isna().all()
    assert "exact_previous_xnys_snapshot_missing" in result.loc[0, "feature_missing_reasons"]


@pytest.mark.parametrize("reason", ['[]', '["warmup", "gap"]', None])
def test_saved_technical_reason_text_or_null_is_preserved_verbatim(reason: str | None) -> None:
    source = source_frame([source_row()])
    source.loc[0, "technical_missing_reasons"] = reason
    result = joined(source=source)
    if reason is None:
        assert pd.isna(result.loc[0, "source_feature_missing_reasons"])
    else:
        assert result.loc[0, "source_feature_missing_reasons"] == reason


def test_all_baseline_missing_is_ineligible_optional_news_and_financial_nulls_are_retained() -> None:
    source = source_frame([source_row()])
    source.loc[0, list(COLUMNS[:124])] = np.nan
    result = joined(source=source)
    assert not result.loc[0, "feature_eligible"]
    assert "all_baseline_features_missing" in result.loc[0, "feature_missing_reasons"]
    source.loc[0, COLUMNS[0]] = np.float32(2)
    source.loc[0, list(NEWS_COLUMNS)] = np.nan
    result = joined(source=source)
    assert result.loc[0, "feature_eligible"]
    assert result.loc[0, list(NEWS_COLUMNS)].isna().all()
    assert result.loc[0, list(FEATURE_NAMES)].isna().all()
    assert any(name.startswith("revenue_growth_ttm:") for name in result.loc[0, "feature_missing_reasons"])


def test_outside_scope_never_becomes_feature_eligible() -> None:
    assert not joined(rows=[decision(status="out_of_scope_sector")]).loc[0, "feature_eligible"]


@pytest.mark.parametrize("day,prior,cutoff", [("2024-01-31", date(2024, 1, 30), "2024-01-30T23:00:00Z"),
    ("2024-03-11", date(2024, 3, 8), "2024-03-08T23:00:00Z"),
    ("2024-11-04", date(2024, 11, 1), "2024-11-01T22:00:00Z")])
def test_nightly_cutoff_uses_new_york_policy_across_winter_and_dst(day: str, prior: date, cutoff: str) -> None:
    import exchange_calendars as xcals

    row = decision()
    row["decision_session"] = day
    row["decision_time_utc"] = xcals.get_calendar("XNYS").session_close(day).isoformat()
    row["decision_id"] = json_sha256({"security_id": row["security_id"], "decision_session": day})
    result = joined(source=source_frame([source_row(session=prior)]), rows=[row])
    assert result.loc[0, "source_cutoff_utc"] == pd.Timestamp(cutoff)
    assert result.loc[0, "feature_eligible"]


def financial_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {"security_id": row["security_id"], "cik": row["cik"], "decision_time": row["decision_time_utc"],
        "available_at": "2024-05-02T20:00:00Z", "metrics": dict.fromkeys(FEATURE_NAMES, None),
        "missing_reasons": dict.fromkeys(FEATURE_NAMES, "original_missing_fact")}


def test_financial_six_use_exact_close_with_original_null_reason_and_precision() -> None:
    row = decision()
    snapshot = financial_snapshot(row)
    snapshot["metrics"][FEATURE_NAMES[0]] = 0.123456789123456
    result = joined(financial={row["decision_id"]: snapshot})
    assert result.loc[0, FEATURE_NAMES[0]] == snapshot["metrics"][FEATURE_NAMES[0]]
    assert result.loc[0, "financial_available_at_utc"] == pd.Timestamp(snapshot["available_at"])
    assert f"{FEATURE_NAMES[1]}:original_missing_fact" in result.loc[0, "feature_missing_reasons"]


@pytest.mark.parametrize("poison", ["duplicate_key", "duplicate_id", "future_value", "unknown_clock", "not_nightly",
    "nan_infinity", "wrong_financial_security", "wrong_financial_cutoff", "future_financial", "missing_financial_clock"])
def test_join_refuses_ambiguous_or_future_source_data(poison: str) -> None:
    row = decision()
    source = source_frame([source_row()])
    snapshot = financial_snapshot(row)
    if poison == "duplicate_key":
        source = pd.concat([source, source], ignore_index=True)
    elif poison == "duplicate_id":
        other = source_row("issuer:other")
        other["decision_id"] = source.loc[0, "decision_id"]
        source = source_frame([source_row(), other])
    elif poison == "future_value":
        source.loc[0, CLOCKS[COLUMNS[0]]] = pd.Timestamp("2024-05-28T21:00:00Z")
    elif poison == "unknown_clock":
        source.loc[0, CLOCKS[COLUMNS[0]]] = pd.NaT
    elif poison == "not_nightly":
        source.loc[0, "decision_time_utc"] = pd.Timestamp("2024-05-24T23:00:00Z")
    elif poison == "nan_infinity":
        source.loc[0, COLUMNS[0]] = np.inf
    elif poison == "wrong_financial_security":
        snapshot["security_id"] = "issuer:other"
    elif poison == "wrong_financial_cutoff":
        snapshot["decision_time"] = "2024-05-28T22:00:00Z"
    elif poison == "future_financial":
        snapshot["available_at"] = "2024-05-29T00:00:00Z"
    elif poison == "missing_financial_clock":
        snapshot["metrics"][FEATURE_NAMES[0]] = 1.0
        snapshot["available_at"] = None
    with pytest.raises(DataReadinessError):
        joined(source=source, financial={row["decision_id"]: snapshot})


def test_full_population_validation_rejects_future_lag_and_accepts_unknown_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    base = joined().iloc[0].to_dict()
    records = []
    for horizon in (63, 252):
        records.append({**base, "horizon_sessions": horizon, "target_status": "unresolved", "target_verified": False,
            "research_eligible": False, "entry_time_utc": "2024-05-29T13:30:00Z",
            "economic_maturity_at_utc": "2025-05-30T20:00:00Z", "label_available_at_utc": None,
            **dict.fromkeys(module.RETURN_COLUMNS, np.nan)})
    frame = pd.DataFrame(records)
    availability = {**CLOCKS, **dict.fromkeys(FEATURE_NAMES, "financial_available_at_utc")}
    for name in (*module.TIMESTAMPS, *availability.values()):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    module.validate_dataset_rows(frame, module.model_profiles(COLUMNS), availability, admitted_at=pd.Timestamp("2026-11-10T12:00:00Z"))
    assert frame.feature_eligible.all() and not frame.research_eligible.any()
    frame["source_cutoff_utc"] = pd.Timestamp("2024-05-23T22:00:00Z")
    with pytest.raises(DataReadinessError, match="nightly"):
        module.validate_dataset_rows(frame, module.model_profiles(COLUMNS), availability,
            admitted_at=pd.Timestamp("2026-11-10T12:00:00Z"))
