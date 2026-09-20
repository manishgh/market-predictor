"""Bounded physical reads and exact derivative row assembly."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pyarrow.dataset as pds

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS, ReturnRelationshipSources
from market_predictor.swing.datasets.predictor_abstention_derivation import _observation, _validated_prefix
from market_predictor.swing.datasets.research_feature_sources import corrected_adjusted_bars
from market_predictor.swing.datasets.return_relationship_integrity import read_object
from market_predictor.swing.datasets.return_relationship_parent import VerifiedRelationshipParent
from market_predictor.swing.datasets.return_relationship_sources import RelationshipSourceContext
from market_predictor.swing.features.adjusted_source import BAR_COLUMNS, _bars
from market_predictor.swing.features.research_join import DECISION_KEYS
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from market_predictor.swing.features.return_relationships import build_return_relationship_profile

ADDITIONS = (*RETURN_RELATIONSHIP_COLUMNS, *(f"available_at_{name}" for name in RETURN_RELATIONSHIP_COLUMNS),
    *(f"missing_reason_{name}" for name in RETURN_RELATIONSHIP_COLUMNS))
PHYSICAL_COLUMNS = (*BAR_COLUMNS, "source", "availability_policy", "ingested_at_utc")
IDENTITY_COLUMNS = [*DECISION_KEYS, "session_date_et", "parent_ticker"]


def read_combined(directory: Path, record: dict[str, Any]) -> pd.DataFrame:
    path = inside(directory, record["path"])
    sidecar = read_object(manifest_path_for(path), record["canonical_manifest_sha256"])
    if (sidecar.get("artifact_type") != "bars" or sidecar.get("artifact_sha256") != record["sha256"]
            or sidecar.get("rows") != record["rows"]
            or not set(PHYSICAL_COLUMNS).issubset(sidecar["columns"])):
        raise DataReadinessError("relationship physical source metadata is missing or substituted")
    if file_sha256(path) != record["sha256"]:
        raise DataReadinessError("relationship adjusted bar hash mismatch")
    arrow: Any = pds
    dataset = arrow.dataset(path, format="parquet")
    predicate = ((arrow.field("bar_start_utc") >= pd.Timestamp("2018-05-29", tz="UTC"))
        & (arrow.field("bar_start_utc") < pd.Timestamp("2024-05-29", tz="UTC")))
    frame: pd.DataFrame = dataset.to_table(columns=list(PHYSICAL_COLUMNS), filter=predicate, use_threads=False).to_pandas()
    if file_sha256(path) != record["sha256"] or not frame.ticker.eq(record["ticker"]).all():
        raise DataReadinessError("relationship adjusted source changed or ticker differs")
    frame["session_date_et"] = frame.bar_start_utc.dt.tz_convert("America/New_York").dt.date
    return frame


def read_stock(root: Path, context: RelationshipSourceContext, item: dict[str, Any]) -> pd.DataFrame:
    identity = item["security_id"]
    if item["kind"] == "corrected":
        frame = corrected_adjusted_bars(context.corrected_directory, item["artifact"])
        frame["session_date_et"] = frame.bar_start_utc.dt.tz_convert("America/New_York").dt.date
        for rule in context.outcome_policy.decision_corrections:
            if rule.security_id == identity:
                selection = frame.session_date_et.between(rule.first_session, rule.last_session)
                frame.loc[selection, "ticker"] = rule.ticker
    else:
        if identity in context.corrected or item["source_group"] in {"FI", "FISV", "SATS", "ECHO"}:
            raise DataReadinessError("relationship corrected issuer cannot consume old combined stream")
        frame = read_combined(context.combined_directory, item["artifact"])
    fact = next((fact for fact in context.facts.failures if fact.security_id == identity), None)
    if fact is not None:
        if item["quarantine"] != fact.model_dump(mode="json"):
            raise DataReadinessError("relationship source quarantine changed")
        if fact.first_invalid_session is None:
            frame = frame.iloc[:0].copy()
        else:
            # The original prefix validator proves the boundary using immutable observations;
            # retain physical metadata from the same original rows, not its narrow projection.
            prefix = _validated_prefix(frame, fact, _observation(root, context.facts, fact))
            frame = frame.loc[frame.session_date_et.isin(prefix.session_date_et)].copy()
    else:
        _bars(frame)
    frame["security_id"] = identity
    return frame


def read_baseline_group(parent: VerifiedRelationshipParent, item: dict[str, Any]) -> pd.DataFrame:
    paths = [str(inside(parent.path.parent, record["profiles"]["technical_market"]["path"]))
        for _, record in sorted(parent.manifest["months"].items())]
    arrow: Any = pds
    dataset = arrow.dataset(paths, format="parquet")
    columns = list(dict.fromkeys((*IDENTITY_COLUMNS, *parent.model_columns,
        *parent.availability_columns.values(), "feature_profile")))
    frame: pd.DataFrame = dataset.to_table(columns=columns,
        filter=arrow.field("security_id") == item["security_id"], use_threads=False).to_pandas()
    if item["kind"] != "corrected":
        frame = frame.loc[frame.parent_ticker.eq(item["source_group"])].copy()
    if (len(frame) != item["rows"] or frame.decision_id.duplicated().any()
            or json_sha256(sorted(frame.decision_id)) != item["decision_ids_sha256"]):
        raise DataReadinessError("relationship projected baseline group differs")
    return frame.reset_index(drop=True)


def build_group(root: Path, parent: VerifiedRelationshipParent, context: RelationshipSourceContext,
    item: dict[str, Any], sources: ReturnRelationshipSources, spy: pd.DataFrame,
    baseline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if baseline is None:
        baseline = read_baseline_group(parent, item)
    bars = read_stock(root, context, item)
    sessions = tuple(day.date() for day in xcals.get_calendar("XNYS").sessions_in_range("2018-05-29", "2024-05-28"))
    result = build_return_relationship_profile(expected_decisions=baseline.loc[:, IDENTITY_COLUMNS],
        baseline=ResearchFeaturePartition(baseline, parent.model_columns, parent.availability_columns, {}),
        stock_bars=bars, spy_bars=spy, history_sessions=sessions,
        contract=load_strategy_contract(inside(root, context.feature_policy.strategy_contract.path)), sources=sources)
    return result.rows.loc[:, [*IDENTITY_COLUMNS, *ADDITIONS]]


def assemble_month(baseline: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    if (baseline.empty or baseline.decision_id.duplicated().any() or additions.decision_id.duplicated().any()
            or set(baseline.decision_id) != set(additions.decision_id)
            or not baseline.feature_profile.eq("technical_market").all()
            or set(ADDITIONS).intersection(baseline.columns)):
        raise DataReadinessError("relationship monthly population or parent profile differs")
    aligned = additions.set_index("decision_id").loc[baseline.decision_id].reset_index()
    try:
        pd.testing.assert_frame_equal(baseline.loc[:, IDENTITY_COLUMNS].reset_index(drop=True),
            aligned.loc[:, IDENTITY_COLUMNS], check_exact=True)
    except AssertionError as error:
        raise DataReadinessError("relationship staged decision identity differs") from error
    result = baseline.copy()
    result["feature_profile"] = "technical_relationships"
    for name in ADDITIONS:
        result[name] = aligned[name].array
    assert_parent_parity(baseline, result)
    return result


def assert_parent_parity(baseline: pd.DataFrame, result: pd.DataFrame) -> None:
    if (list(result.columns) != [*baseline.columns, *ADDITIONS]
            or not baseline.feature_profile.eq("technical_market").all()
            or not result.feature_profile.eq("technical_relationships").all()):
        raise DataReadinessError("relationship row schema or profile identity differs")
    columns = list(baseline.columns.drop("feature_profile"))
    try:
        pd.testing.assert_frame_equal(baseline.loc[:, columns], result.loc[:, columns], check_exact=True, check_dtype=True)
    except AssertionError as error:
        raise DataReadinessError("relationship publication changed an inherited value, outcome, clock or row order") from error
