"""Causal investment dataset contracts and one-pass retrospective research loader."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from pathlib import Path
from typing import Any, Literal, cast

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import Field

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.path_integrity import verify_no_reparse_ancestry
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.investment.fundamentals import FEATURE_NAMES
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256, Timestamp
from market_predictor.swing.contracts.holding_materialization import SourcePin

DATASET_SCHEMA = "market_predictor.investment_dataset"
ADMISSION_SCHEMA: Literal["market_predictor.investment_dataset_admission"] = "market_predictor.investment_dataset_admission"
EXPECTED_ROWS = 74458
PROFILES = ("technical_relationships", "all_features")
TIMESTAMPS = ("decision_time_utc", "entry_time_utc", "economic_maturity_at_utc", "label_available_at_utc",
    "source_cutoff_utc", "financial_available_at_utc")
RETURN_COLUMNS = ("target_net_excess_vs_spy", "stock_net_return", "spy_gross_return", "qqq_gross_return", "sector_gross_return")
BOOL_COLUMNS = ("target_verified", "feature_eligible", "research_eligible", "source_feature_eligible")


class InvestmentDatasetRequest(HoldingContract):
    schema_name: Literal["market_predictor.investment_dataset_request"] = Field(alias="schema")
    scope: Literal["retrospective_research"]
    targets_manifest: SourcePin
    news_manifest: SourcePin
    news_verification: SourcePin
    news_producer_sources: dict[str, SourcePin]
    financial_manifest: SourcePin
    roster_manifest: SourcePin
    feature_lag: Literal["previous_xnys_session_nightly"]
    primary_horizon: Literal[252]
    diagnostic_horizon: Literal[63]
    missingness: Literal["train_partition_median_with_indicators"]
    evaluation_as_of_utc: Timestamp


class InvestmentDatasetAdmission(HoldingContract):
    schema_name: Literal["market_predictor.investment_dataset_admission"] = Field(alias="schema")
    status: Literal["admitted_for_retrospective_research"]
    scope: Literal["retrospective_research"]
    dataset: SourcePin
    request: SourcePin
    feature_contract_sha256: Sha256
    model_columns: dict[str, tuple[str, ...]]
    admitted_at_utc: Timestamp
    research_training_allowed: Literal[True]
    source_consumption_scope: Literal["saved_verified_artifacts_and_exact_producer_bytes"]
    raw_sources_rechecked: Literal[False]
    source_admission_changed: Literal[False]
    historical_first_seen_proven: Literal[False]
    promotion_eligible: Literal[False]
    production_eligible: Literal[False]
    counts_by_horizon: dict[str, dict[str, int]]
    source_files: dict[str, Sha256]
    files: dict[str, Sha256] = Field(default_factory=dict)


@dataclass(frozen=True)
class InvestmentDataset:
    features: pd.DataFrame
    metadata: pd.DataFrame
    model_columns: dict[str, tuple[str, ...]]
    source_files: dict[str, str]
    admitted_at_utc: datetime
    feature_contract_sha256: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def add_pins(root: Path, destination: dict[str, str], sources: dict[str, str], *, paths: EvidencePaths | None = None) -> None:
    for name, digest in sources.items():
        path = str((root / name).resolve() if paths is None else paths.identity(name))
        require(path not in destination or destination[path] == digest, "investment dataset source pins conflict")
        destination[path] = digest


def read_object(root: Path, pin: SourcePin, pins: dict[str, str], *, paths: EvidencePaths | None = None) -> dict[str, Any]:
    from market_predictor.research.investment_features import ArtifactReference, _object

    path = str(inside(root, pin.path) if paths is None else paths.identity(pin.path))
    require(path not in pins or pins[path] == pin.sha256, "investment dataset source pin conflicts with an earlier read")
    return _object(root, ArtifactReference(path=pin.path, sha256=pin.sha256), pins, paths=paths)


def pin_source_file(root: Path, pin: SourcePin, pins: dict[str, str], *, paths: EvidencePaths | None = None) -> None:
    path = inside(root, pin.path) if paths is None else paths.read(pin.path)
    require(file_sha256(path) == pin.sha256, f"investment source bytes changed: {pin.path}")
    add_pins(root, pins, {pin.path: pin.sha256}, paths=paths)


def recheck_sources(
    root: Path, pins: dict[str, str], *, paths: EvidencePaths | None = None,
    retained_external: frozenset[str] = frozenset(),
) -> None:
    for name, digest in pins.items():
        path = root / name if paths is None or name in retained_external else paths.read(name)
        if name in retained_external:
            path = verify_no_reparse_ancestry(path, label="retained investment source")
        require(file_sha256(path) == digest, f"investment dataset source changed: {name}")


def model_profiles(columns: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    require(len(columns) == len(set(columns)) == 209 and not set(columns).intersection(FEATURE_NAMES),
        "investment dataset requires the original ordered 209-feature news contract")
    return {"technical_relationships": columns[:124], "all_features": (*columns, *FEATURE_NAMES)}


def feature_contract(profiles: dict[str, tuple[str, ...]], availability: dict[str, str]) -> str:
    return json_sha256({"model_columns": profiles, "availability_columns": availability,
        "feature_lag": "previous_xnys_session_nightly", "financial_cutoff": "exact_investment_close",
        "eligibility": "exact_previous_snapshot_and_at_least_one_finite_baseline_feature",
        "missingness": "train_partition_median_with_indicators", "scope": "retrospective_research"})


def previous_session(day: date) -> date:
    calendar = xcals.get_calendar("XNYS")
    require(calendar.is_session(day.isoformat()), "investment dataset decision is not an XNYS session")
    return cast(date, calendar.previous_session(day.isoformat()).date())


def join_lagged_features(*, decisions: list[dict[str, Any]], source: pd.DataFrame,
    model_columns: tuple[str, ...], availability_columns: dict[str, str],
    financial: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Join exact prior-session saved values; neither aggregates nor imputes features."""
    require(source.columns.is_unique, "investment source repeats column names")
    required = {"decision_id", "security_id", "ticker", "session_date_et", "decision_time_utc", "feature_eligible",
        *model_columns, *(availability_columns[name] for name in model_columns)}
    require(required.issubset(source), "investment source row contract is incomplete")
    keys = list(zip(source.security_id, source.session_date_et, strict=True))
    require(len(keys) == len(set(keys)) and not source.decision_id.duplicated().any(), "investment source repeats exact row keys")
    lookup = dict(zip(keys, source.to_dict("records"), strict=True))
    rows = []
    calendar = xcals.get_calendar("XNYS")
    prior_days = {day: previous_session(day) for day in {date.fromisoformat(str(row["decision_session"])) for row in decisions}}
    dates = sorted(set(prior_days.values()))
    nightly_cutoffs = dict(zip(dates, swing_prediction_cutoffs(pd.Series(dates)), strict=True))
    for decision in decisions:
        day = date.fromisoformat(str(decision["decision_session"]))
        cutoff = pd.Timestamp(decision["decision_time_utc"])
        require(cutoff == calendar.session_close(day.isoformat()), "investment dataset decision differs from frozen session close")
        prior = prior_days[day]
        saved = lookup.get((decision["security_id"], prior))
        result = dict(decision)
        result["decision_session"] = day
        result.update({"source_decision_id": None, "source_security_id": None, "source_ticker": None,
            "source_session": None, "source_cutoff_utc": None, "source_feature_eligible": False,
            "source_feature_missing_reasons": None, "feature_missing_reasons": []})
        baseline_finite = 0
        for name in model_columns:
            result[name], result[availability_columns[name]] = np.nan, None
        if saved is None:
            result["feature_missing_reasons"].append("exact_previous_xnys_snapshot_missing")
        else:
            source_cutoff = pd.Timestamp(saved["decision_time_utc"])
            expected = nightly_cutoffs[prior]
            require(source_cutoff == expected and source_cutoff < cutoff,
                "investment source cutoff is not the exact previous-session nightly cutoff")
            require(type(saved["feature_eligible"]) is bool, "investment source eligibility is not a boolean")
            result.update({"source_decision_id": saved["decision_id"], "source_security_id": saved["security_id"],
                "source_ticker": saved["ticker"], "source_session": prior, "source_cutoff_utc": source_cutoff,
                "source_feature_eligible": saved["feature_eligible"],
                "source_feature_missing_reasons": saved.get("technical_missing_reasons")})
            for position, name in enumerate(model_columns):
                value, raw_clock = saved[name], saved[availability_columns[name]]
                clock = pd.Timestamp(raw_clock) if not pd.isna(raw_clock) else None
                if not pd.isna(value):
                    require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)),
                        "investment source feature is not finite numeric data")
                    require(clock is not None and clock.tzinfo is not None and clock <= source_cutoff,
                        "investment source finite feature clock is absent or after its source cutoff")
                    baseline_finite += int(position < 124)
                result[name], result[availability_columns[name]] = value, clock
                if pd.isna(value):
                    reason = saved.get(f"missing_reason_{name}")
                    result["feature_missing_reasons"].append(f"{name}:{reason or 'source_value_missing'}")
        result["feature_eligible"] = (decision.get("roster_status", decision.get("status")) == "eligible"
            and saved is not None and baseline_finite > 0)
        if saved is not None and not baseline_finite:
            result["feature_missing_reasons"].append("all_baseline_features_missing")
        snapshot = financial.get(decision["decision_id"])
        result["financial_available_at_utc"] = None
        if snapshot is not None:
            require(snapshot["security_id"] == decision["security_id"]
                and snapshot["cik"] == decision["cik"] and pd.Timestamp(snapshot["decision_time"]) == cutoff,
                "investment financial snapshot identity or exact decision cutoff differs")
            clock = None if snapshot["available_at"] is None else pd.Timestamp(snapshot["available_at"])
            require(clock is None or (clock.tzinfo is not None and clock <= cutoff),
                "investment financial snapshot is after its decision cutoff")
            result["financial_available_at_utc"] = clock
        for name in FEATURE_NAMES:
            value = None if snapshot is None else snapshot["metrics"][name]
            require(value is None or (np.isfinite(value) and snapshot is not None and snapshot["available_at"] is not None),
                "investment finite financial feature has no valid availability")
            result[name] = np.nan if value is None else value
            if value is None:
                reason = "snapshot_missing" if snapshot is None else snapshot["missing_reasons"].get(name, "metric_missing")
                result["feature_missing_reasons"].append(f"{name}:{reason}")
        rows.append(result)
    frame = pd.DataFrame(rows)
    for name in model_columns:
        dtype = source[name].dtype if pd.api.types.is_numeric_dtype(source[name]) else "float32"
        frame[name] = frame[name].astype(dtype)
    return frame


def counts_by_horizon(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {str(horizon): {"population_rows": int(len(group)), "target_verified_rows": int(group.target_verified.sum()),
        "feature_eligible_rows": int(group.feature_eligible.sum()), "research_eligible_rows": int(group.research_eligible.sum())}
        for horizon, group in frame.groupby("horizon_sessions", sort=True)}


def validate_dataset_rows(frame: pd.DataFrame, profiles: dict[str, tuple[str, ...]], availability: dict[str, str],
    *, admitted_at: datetime) -> None:
    require(set(profiles) == set(PROFILES) and len(profiles["technical_relationships"]) == 124
        and len(profiles["all_features"]) == 215 and profiles["all_features"][:124] == profiles["technical_relationships"]
        and profiles["all_features"][-6:] == FEATURE_NAMES, "investment dataset profile order differs")
    require(len(frame) == EXPECTED_ROWS and not frame.duplicated(["decision_id", "horizon_sessions"]).any(),
        "investment dataset full horizon row population differs")
    require(set(frame.horizon_sessions) == {63, 252} and frame.groupby("decision_id").size().eq(2).all(),
        "investment dataset lacks the exact two-horizon population")
    require(all(pd.api.types.is_bool_dtype(frame[name]) for name in BOOL_COLUMNS), "investment dataset eligibility dtype differs")
    for name in TIMESTAMPS:
        require(isinstance(frame[name].dtype, pd.DatetimeTZDtype), "investment dataset clocks must retain UTC aware dtypes")
        require(str(frame[name].dt.tz) == "UTC", "investment dataset clocks must remain UTC")
    has_source = frame.source_cutoff_utc.notna()
    calendar = xcals.get_calendar("XNYS")
    prior = {day: previous_session(day) for day in frame.decision_session.unique()}
    require(frame.decision_time_utc.eq(pd.to_datetime(frame.decision_session.map(
        lambda day: calendar.session_close(day.isoformat())), utc=True)).all(), "investment dataset decision clock differs")
    require(frame.loc[has_source, "source_security_id"].eq(frame.loc[has_source, "security_id"]).all()
        and frame.loc[has_source, "source_decision_id"].notna().all()
        and frame.loc[has_source, "source_session"].eq(frame.loc[has_source, "decision_session"].map(prior)).all(),
        "investment dataset source identity or exact previous session differs")
    nightly = swing_prediction_cutoffs(frame.loc[has_source, "source_session"])
    require(frame.loc[has_source, "source_cutoff_utc"].eq(nightly).all(), "investment dataset source nightly cutoff differs")
    require(frame.loc[has_source, "source_cutoff_utc"].lt(frame.loc[has_source, "decision_time_utc"]).all(),
        "investment source snapshot does not precede decision")
    for name in profiles["all_features"]:
        values = frame[name]
        require(pd.api.types.is_numeric_dtype(values) and np.isfinite(values.dropna().to_numpy()).all(),
            "investment dataset feature dtype or finite values differ")
        clock = frame[availability[name]]
        valid = values.notna()
        bound = frame.decision_time_utc if name in FEATURE_NAMES else frame.source_cutoff_utc
        require(clock.loc[valid].notna().all() and clock.loc[valid].le(bound.loc[valid]).all(),
            "investment dataset finite feature availability exceeds its cutoff")
    baseline = frame.loc[:, list(profiles["technical_relationships"])].notna().any(axis=1)
    require(frame.feature_eligible.equals(frame.roster_status.eq("eligible") & has_source & baseline),
        "investment dataset baseline eligibility differs")
    require(frame.research_eligible.equals(frame.feature_eligible & frame.target_verified),
        "investment dataset profile research eligibility differs")
    verified = frame.target_verified
    require(verified.equals(frame.target_status.eq("source_validated"))
        and frame.loc[verified, "roster_status"].eq("eligible").all(), "investment target scope/status differs")
    require(frame.loc[verified, list(RETURN_COLUMNS)].notna().all().all()
        and np.isfinite(frame.loc[verified, list(RETURN_COLUMNS)].to_numpy()).all(), "investment verified target is incomplete")
    require(frame.loc[~verified, list(RETURN_COLUMNS)].isna().all().all(), "investment unresolved target contains economics")
    require(frame.loc[verified, "label_available_at_utc"].notna().all()
        and frame.loc[verified, "label_available_at_utc"].le(admitted_at).all()
        and frame.loc[verified, "label_available_at_utc"].ge(frame.loc[verified, "economic_maturity_at_utc"]).all(),
        "investment target verification is before maturity or after admission")
    require(np.isclose(frame.loc[verified, "target_net_excess_vs_spy"],
        frame.loc[verified, "stock_net_return"] - frame.loc[verified, "spy_gross_return"], rtol=0, atol=1e-12).all(),
        "investment dataset target does not reconcile stock net against SPY gross")


def load_investment_dataset(
    *, root: Path, manifest: SourcePin, admission: SourcePin, paths: EvidencePaths | None = None,
) -> InvestmentDataset:
    """Load all rows once under the caller's lease; never import historical model code."""
    guard("investment dataset admission")
    pins: dict[str, str] = {}
    receipt = InvestmentDatasetAdmission.model_validate_json(json.dumps(read_object(root, admission, pins, paths=paths)))
    require(receipt.files == {}, "investment admission contains unexpected files")
    require(receipt.dataset == manifest and receipt.scope == "retrospective_research", "investment admission refers to another dataset")
    report = read_object(root, manifest, pins, paths=paths)
    require(report.get("schema") == DATASET_SCHEMA and report.get("status") == "complete_research_dataset"
        and report.get("request") == receipt.request.model_dump(mode="json"), "investment dataset manifest or request differs")
    parsed_request = InvestmentDatasetRequest.model_validate_json(json.dumps(read_object(root, receipt.request, pins, paths=paths)))
    require(set(report["files"]) == {"request.json"}, "investment dataset file inventory differs")
    directory = (inside(root, manifest.path) if paths is None else paths.identity(manifest.path)).parent
    published_request = SourcePin(path=str(directory / "request.json"),
        sha256=report["files"]["request.json"])
    require(InvestmentDatasetRequest.model_validate_json(
        json.dumps(read_object(root, published_request, pins, paths=paths))) == parsed_request,
        "investment published request differs from the frozen input")
    require(report["feature_lag"] == parsed_request.feature_lag and report.get("rows") == EXPECTED_ROWS,
        "investment dataset request lag or population differs")
    profiles = {key: tuple(value) for key, value in report["model_columns"].items()}
    require(profiles == receipt.model_columns and report["feature_contract_sha256"] == receipt.feature_contract_sha256
        and feature_contract(profiles, report["availability_columns"]) == receipt.feature_contract_sha256,
        "investment admission feature contract differs")
    require(receipt.source_files == report["source_files"], "investment admission source inventory differs")
    require(report.get("source_consumption_scope") == receipt.source_consumption_scope
        and report.get("raw_sources_rechecked") is False and report.get("source_admission_changed") is False,
        "investment dataset source consumption scope differs")
    add_pins(root, pins, report["source_files"], paths=paths)
    pieces = []
    for month, child in sorted(report["months"].items()):
        guard("investment dataset partition input")
        require(child["path"] == f"months/{month}.parquet", "investment dataset partition path differs")
        path = inside(directory, child["path"]) if paths is None else paths.read_child(directory, child["path"])
        require(file_sha256(path) == child["sha256"], "investment dataset partition changed")
        add_pins(root, pins, {str(directory / child["path"]): child["sha256"]}, paths=paths)
        frame = pd.read_parquet(path)
        require(len(frame) == child["rows"] and frame.decision_session.map(str).str[:7].eq(month).all()
            and json_sha256(sorted(zip(frame.decision_id, frame.horizon_sessions, strict=True))) == child["row_keys_sha256"],
            "investment dataset partition population differs")
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True)
    validate_dataset_rows(frame, profiles, report["availability_columns"], admitted_at=receipt.admitted_at_utc)
    require(parsed_request.evaluation_as_of_utc <= receipt.admitted_at_utc
        and frame.loc[frame.target_verified, "label_available_at_utc"].le(parsed_request.evaluation_as_of_utc).all(),
        "investment supervised labels were unavailable at requested evaluation cutoff")
    require(counts_by_horizon(frame) == report["counts_by_horizon"] == receipt.counts_by_horizon,
        "investment dataset admission eligibility counts differ")
    recheck_sources(root, pins, paths=paths)
    names = profiles["all_features"]
    return InvestmentDataset(frame.loc[:, list(names)], frame.drop(columns=list(names)), profiles, pins,
        receipt.admitted_at_utc, receipt.feature_contract_sha256)
