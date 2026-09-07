"""Bind a small historical replay to existing selected decisions and source bytes."""
from __future__ import annotations

import tomllib
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.labels.holding_paths import holding_calendar, validate_outcome_observations

_ET = ZoneInfo("America/New_York")
_SHA = r"^[0-9a-f]{64}$"


class TransferSecurity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9.]{0,9}$")
    security_id: str = Field(min_length=1)
    sec_cik: str = Field(pattern=r"^\d{10}$")
    decision_dates: list[date] = Field(min_length=1, max_length=30)

    @field_validator("decision_dates")
    @classmethod
    def ordered_dates(cls, values: list[date]) -> list[date]:
        if values != sorted(set(values)):
            raise ValueError("transfer decision dates must be unique and ascending")
        return values


class TransferReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["market_predictor.swing_transfer_replay"]
    control_receipt: str
    control_receipt_sha256: str = Field(pattern=_SHA)
    membership_path: str
    membership_sha256: str = Field(pattern=_SHA)
    sec_relations_path: str
    sec_relations_sha256: str = Field(pattern=_SHA)
    raw_manifest_path: str
    raw_manifest_sha256: str = Field(pattern=_SHA)
    horizon_sessions: Literal[10]
    maximum_pages_per_ticker: int = Field(strict=True, ge=1, le=3)
    expected_security_count: int = Field(strict=True, ge=1, le=50)
    expected_decision_count: int = Field(strict=True, ge=1, le=1500)
    securities: list[TransferSecurity] = Field(min_length=1, max_length=50)


def load_transfer_replay_config(path: Path) -> TransferReplayConfig:
    if path.stat().st_size > 1024 * 1024:
        raise DataReadinessError("transfer replay config is oversized")
    config = TransferReplayConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    cases = config.securities
    if (len(cases) != config.expected_security_count
            or len({item.ticker for item in cases}) != len(cases)
            or len({item.security_id for item in cases}) != len(cases)
            or sum(len(item.decision_dates) for item in cases) != config.expected_decision_count):
        raise DataReadinessError("transfer replay population counts or identities differ")
    return config


def checked_transfer_file(root: Path, relative: str, expected: str, bound: dict[str, str]) -> Path:
    path = resolve_inside_authority(root, relative)
    if file_sha256(path) != expected:
        raise DataReadinessError(f"transfer source hash mismatch: {relative}")
    bound[path.relative_to(root.resolve()).as_posix()] = expected
    return path


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024 * 1024:
        raise DataReadinessError("transfer source metadata exceeds bounded size")
    return parse_strict_json_object(path.read_bytes(), label=str(path))


def _frame(path: Path, columns: list[str]) -> pd.DataFrame:
    parquet: Any = pq
    frame: pd.DataFrame = parquet.ParquetFile(path).read(columns=columns, use_threads=False).to_pandas()
    assert_memory_budget(stage="transfer source projection", hard_budget_gib=5.0, headroom_gib=0.75)
    return frame


def _selected_records(
    root: Path, case: TransferSecurity, control: dict[str, Any], bound: dict[str, str],
) -> list[dict[str, Any]]:
    dates = {day.isoformat() for day in case.decision_dates}
    parts: list[pd.DataFrame] = []
    for month in sorted({day[:7] for day in dates}):
        records = [(path, digest) for path, digest in control["bound_files"].items()
                   if path.endswith(f"/month={month}/part.parquet") and "/feature_profile=technical_market/" in path]
        if len(records) != 1:
            raise DataReadinessError(f"transfer decision partition is not uniquely bound: {month}")
        path = checked_transfer_file(root, records[0][0], records[0][1], bound)
        frame = _frame(path, ["decision_id", "security_id", "session_date_et", "decision_time_utc"])
        frame["session_date_et"] = frame["session_date_et"].astype(str)
        parts.append(frame.loc[frame["security_id"].eq(case.security_id) & frame["session_date_et"].isin(dates)])
    selected = pd.concat(parts, ignore_index=True)
    if not selected["decision_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise DataReadinessError("transfer selected decision IDs must be nonempty strings")
    if (len(selected) != len(dates) or selected["session_date_et"].duplicated().any()
            or set(selected["session_date_et"]) != dates or selected["decision_id"].duplicated().any()
            or not selected["decision_id"].isin(control["selected_decision_ids"]).all()):
        raise DataReadinessError(f"transfer dates are not exact frozen selected decisions: {case.ticker}")
    result: list[dict[str, Any]] = []
    for row in selected.sort_values("session_date_et").itertuples(index=False):
        cutoff = pd.Timestamp(row.decision_time_utc)
        if cutoff.tzinfo is None or cutoff.tz_convert(_ET).date().isoformat() != row.session_date_et:
            raise DataReadinessError("transfer decision clock differs from its session")
        result.append({"decision_id": row.decision_id, "session_date_et": str(row.session_date_et),
                       "decision_time_utc": cutoff.isoformat()})
    return result


def _anchor(case: TransferSecurity, decisions: list[dict[str, Any]], membership: pd.DataFrame, relations: pd.DataFrame) -> None:
    for row in decisions:
        cutoff = pd.Timestamp(row["decision_time_utc"])
        for frame, label in ((membership, "membership"), (relations, "SEC issuer")):
            match = frame.loc[
                frame["security_id"].eq(case.security_id) & frame["ticker"].eq(case.ticker)
                & frame["effective_from_utc"].le(cutoff)
                & (frame["effective_to_utc"].isna() | frame["effective_to_utc"].gt(cutoff))
            ]
            if len(match) != 1 or (label == "SEC issuer" and str(match.iloc[0]["sec_cik"]) != case.sec_cik):
                raise DataReadinessError(f"transfer {label} anchor is missing or ambiguous: {case.ticker}")


def _retained_rows(
    root: Path, case: TransferSecurity, sessions: list[str], raw: dict[str, Any], bound: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    records = [record for record in raw["artifacts"] if record["ticker"] == case.ticker]
    if len(records) != 1 or records[0]["price_feed"] != "sip" or records[0]["adjustment"] != "all":
        raise DataReadinessError("transfer retained source identity or feed differs")
    record = records[0]
    path = checked_transfer_file(root, str(record["path"]).replace("\\", "/"), record["sha256"], bound)
    frame = _frame(path, ["ticker", "timeframe", "bar_start_utc", "bar_end_utc", "available_at_utc",
                          "open", "high", "low", "close", "volume", "price_feed", "adjustment"])
    frame["session_date_et"] = pd.to_datetime(frame["bar_start_utc"], utc=True).dt.tz_convert(_ET).dt.date
    frame = frame.loc[frame["session_date_et"].astype(str).isin(sessions)].copy()
    if (not frame["ticker"].eq(case.ticker).all() or not frame["price_feed"].eq("sip").all()
            or not frame["adjustment"].eq("all").all() or not frame["timeframe"].eq("1d").all()
            or frame["session_date_et"].duplicated().any()):
        raise DataReadinessError("transfer retained daily bars have conflicting identity or duplicated sessions")
    frame = validate_outcome_observations(frame)
    values = [{"session_date_et": str(row.session_date_et), **{name: float(getattr(row, name)) for name in
               ("open", "high", "low", "close", "volume")}} for row in frame.itertuples(index=False)]
    # This bounded replay concerns the previously price-complete transfer cases.
    if set(frame["session_date_et"].astype(str)) != set(sessions) or not frame["outcome_observation_valid"].all():
        raise DataReadinessError("transfer retained price-complete population no longer verifies")
    return path.relative_to(root).as_posix(), sorted(values, key=lambda row: row["session_date_et"])


def prepare_transfer_replay_request(root: Path, config_path: Path) -> dict[str, Any]:
    """Read only selected identity columns and the required retained daily bars."""
    root = root.resolve()
    config_path = config_path.resolve()
    config = load_transfer_replay_config(config_path)
    bound: dict[str, str] = {}
    checked_transfer_file(root, str(config_path), file_sha256(config_path), bound)
    control = _object(checked_transfer_file(root, config.control_receipt, config.control_receipt_sha256, bound))
    if (control.get("schema_version") != "market_predictor.swing_accounting_control.v1"
            or control.get("status") != "blocked" or control.get("validation_or_test_outcomes_read") is not False
            or control.get("scope") != "initial_fit_deterministic_control_not_out_of_sample"):
        raise DataReadinessError("transfer replay requires the frozen initial-fit blocked control")
    raw = _object(checked_transfer_file(root, config.raw_manifest_path, config.raw_manifest_sha256, bound))
    membership = _frame(checked_transfer_file(root, config.membership_path, config.membership_sha256, bound),
                        ["security_id", "ticker", "effective_from_utc", "effective_to_utc"])
    relations = _frame(checked_transfer_file(root, config.sec_relations_path, config.sec_relations_sha256, bound),
                       ["security_id", "ticker", "sec_cik", "effective_from_utc", "effective_to_utc"])
    units: list[dict[str, Any]] = []
    all_ids: list[str] = []
    for case in sorted(config.securities, key=lambda item: item.ticker):
        decisions = _selected_records(root, case, control, bound)
        _anchor(case, decisions, membership, relations)
        calendar = holding_calendar(case.decision_dates[0], case.decision_dates[-1] + timedelta(days=35))
        needed: set[date] = set()
        for day in case.decision_dates:
            if day not in calendar or day.isoformat() not in control["decision_sessions"]:
                raise DataReadinessError("transfer decision is outside the frozen exchange calendar")
            index = calendar.index(day)
            needed.update(calendar[index + 1:index + 1 + config.horizon_sessions])
        sessions = sorted(day.isoformat() for day in needed)
        if (not sessions or len(sessions) > 30 or not set(sessions).issubset(control["valuation_sessions"])
                or sessions != [day.isoformat() for day in holding_calendar(date.fromisoformat(sessions[0]),
                                                                           date.fromisoformat(sessions[-1]))]):
            raise DataReadinessError("transfer holding window must be contiguous, bounded and within initial fit")
        retained_path, retained = _retained_rows(root, case, sessions, raw, bound)
        start = datetime.combine(date.fromisoformat(sessions[0]), time.min, _ET)
        end = datetime.combine(date.fromisoformat(sessions[-1]) + timedelta(days=1), time.min, _ET) - timedelta(microseconds=1)
        unit = {
            "ticker": case.ticker, "security_id": case.security_id, "sec_cik": case.sec_cik,
            "decisions": decisions, "required_sessions": sessions, "retained_path": retained_path,
            "retained_rows": retained, "parameters": {"symbols": case.ticker, "timeframe": "1Day", "feed": "sip",
                "adjustment": "all", "sort": "asc", "limit": 10000, "start": start.isoformat(),
                "end": end.isoformat(), "asof": case.decision_dates[0].isoformat()},
        }
        unit["unit_sha256"] = json_sha256(unit)
        units.append(unit)
        all_ids.extend(row["decision_id"] for row in decisions)
    if len(set(all_ids)) != config.expected_decision_count:
        raise DataReadinessError("transfer replay decision identities are not unique")
    for path, digest in list(bound.items()):
        checked_transfer_file(root, path, digest, bound)
    return {
        "schema_version": "market_predictor.swing_transfer_replay_request",
        "config": config.model_dump(mode="json"), "bound_files": bound, "units": units,
        "maximum_pages_per_ticker": config.maximum_pages_per_ticker,
        "retry_budget_per_page": 1, "provider_end_boundary": "inclusive_last_session_end_microsecond",
        "comparison_policy": "exact_finite_OHLCV_by_session_no_replacement",
        "identity_admission": False, "accounting_eligible": False,
    }
