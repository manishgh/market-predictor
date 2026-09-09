"""Replay pinned holding requirements without decoding prices, features or outcomes."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.joins import MEMBERSHIP_VALUE_COLUMNS
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.swing.contracts.research_cohort import load_swing_research_cohort
from market_predictor.swing.labels.holding_identity import inspect_holding_membership_windows
from market_predictor.swing.labels.holding_paths import holding_calendar

IDENTITY_COLUMNS = (
    "decision_id", "security_id", "ticker", "sector", "primary_benchmark", "session_date_et", "decision_time_utc",
)
MEMBERSHIP_COLUMNS = tuple(dict.fromkeys((
    "ticker", *MEMBERSHIP_VALUE_COLUMNS, "effective_from_utc", "effective_to_utc", "available_at_utc",
)))
_METADATA_LIMIT = 8 * 1024**2


@dataclass(frozen=True)
class PreparedHoldingObservations:
    bound_files: dict[str, str]
    decisions: pd.DataFrame
    requirements: list[dict[str, Any]]
    memberships: pd.DataFrame
    raw_manifest: dict[str, Any]
    numeric_end: date


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _METADATA_LIMIT:
        raise DataReadinessError("holding requirements metadata exceeds bounded size")
    with path.open("rb") as source:
        content = source.read(_METADATA_LIMIT + 1)
    if len(content) > _METADATA_LIMIT:
        raise DataReadinessError("holding requirements metadata exceeds bounded size")
    return parse_strict_json_object(content, label=str(path))


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DataReadinessError("holding requirements requires a canonical SHA-256")
    return value


def _day(value: Any) -> date:
    if not isinstance(value, str):
        raise DataReadinessError("holding requirements dates must be canonical ISO dates")
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise DataReadinessError("holding requirements dates must be canonical ISO dates")
    return result


def _identity(payload: dict[str, Any], field: str, expected: str) -> None:
    if (payload.get(field) != _digest(expected)
            or json_sha256({key: value for key, value in payload.items() if key != field}) != expected):
        raise DataReadinessError(f"holding requirements canonical {field} differs")


def _checked(root: Path, relative: Any, expected: Any, bound: dict[str, str]) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise DataReadinessError("holding requirements source path is invalid")
    path = resolve_inside_authority(root, relative)
    digest = _digest(expected)
    key = path.relative_to(root).as_posix()
    if key in bound and bound[key] != digest:
        raise DataReadinessError("holding requirements source bindings conflict")
    if file_sha256(path) != digest:
        raise DataReadinessError(f"holding requirements source hash differs: {relative}")
    bound[key] = digest
    return path


def _bound_source(root: Path, relative: Any, bound: dict[str, str]) -> Path:
    path = resolve_inside_authority(root, relative)
    key = path.relative_to(root).as_posix()
    if key not in bound:
        raise DataReadinessError(f"holding requirements source is not bound to preflight: {relative}")
    return _checked(root, key, bound[key], bound)


def _projection(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    arrow: Any = pq
    with arrow.ParquetFile(path) as source:
        if source.metadata.num_rows > 1_000_000 or source.metadata.serialized_size > _METADATA_LIMIT:
            raise DataReadinessError("holding requirements identity projection exceeds bounded size")
        if not set(columns).issubset(source.schema_arrow.names):
            raise DataReadinessError("holding requirements identity projection is missing columns")
        frame: pd.DataFrame = source.read(columns=list(columns), use_threads=False).to_pandas()
    return frame


def _raw_inventory(
    root: Path, parent: dict[str, Any], bound: dict[str, str], *, start: date, end: date,
) -> tuple[dict[str, Any], set[str]]:
    post = parent["combined_daily_inputs"]["post_collection"]
    directory = Path(post["directory"])
    request = _object(_checked(root, str(directory / "_request.json"), post["request_file_sha256"], bound))
    manifest = _object(_checked(root, str(directory / "_manifest.json"), post["manifest_sha256"], bound))
    # The existing daily-collection schema hashes standard JSON separators.
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    if identity != _digest(post["request_identity_sha256"]) or request.get("request_sha256") != identity:
        raise DataReadinessError("holding requirements daily collection request hash differs")
    if (request.get("schema") != "swing.daily_history_collection.v1"
            or manifest.get("schema") != "swing.daily_history_manifest.v1"
            or manifest.get("request_sha256") != request["request_sha256"]
            or request.get("source") != "alpaca" or request.get("price_feed") != "sip"
            or request.get("adjustment") != "all" or request.get("timeframe") != "1d"
            or _day(request["start_date"]) != start or _day(request["end_date"]) != end):
        raise DataReadinessError("holding requirements raw source, request, feed or dates differ")
    symbols = request.get("symbols")
    artifacts = manifest.get("artifacts")
    if (not isinstance(symbols, list) or not symbols
            or any(not isinstance(value, str) or not value.strip() for value in symbols)
            or len(set(symbols)) != len(symbols) or not isinstance(artifacts, list)
            or manifest.get("artifact_count") != len(artifacts)):
        raise DataReadinessError("holding requirements raw manifest inventory differs")
    seen: set[str] = set()
    for record in artifacts:
        if (not isinstance(record, dict) or record.get("ticker") not in symbols
                or record["ticker"] in seen or record.get("price_feed") != "sip" or record.get("adjustment") != "all"):
            raise DataReadinessError("holding requirements raw artifact identity or feed differs")
        seen.add(record["ticker"])
    return manifest, set(symbols)


def _affected(preflight: dict[str, Any], expected_decisions: int, expected_securities: int) -> dict[str, dict[str, Any]]:
    records = preflight.get("affected_securities")
    if not isinstance(records, list):
        raise DataReadinessError("holding requirements affected inventory is invalid")
    selected: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise DataReadinessError("holding requirements affected identity is invalid")
        identity, count = record.get("security_id"), record.get("initial_fit_decisions")
        if (not isinstance(identity, str) or not identity.strip() or identity in seen
                or type(count) is not int or count < 0):
            raise DataReadinessError("holding requirements affected identity or count is invalid")
        seen.add(identity)
        if count:
            tickers = record.get("tickers")
            if (not isinstance(tickers, list) or not tickers
                    or any(not isinstance(value, str) or not value.strip() for value in tickers)
                    or tickers != sorted(set(tickers)) or _day(record["first_decision"]) > _day(record["last_decision"])):
                raise DataReadinessError("holding requirements affected ticker or date inventory differs")
            selected[identity] = record
    if (len(selected) != expected_securities
            or sum(row["initial_fit_decisions"] for row in selected.values()) != expected_decisions
            or preflight["initial_fit"]["uncovered_matured"] != expected_decisions):
        raise DataReadinessError("holding requirements expected initial-fit totals differ")
    return selected


def _decisions(
    root: Path, manifest_path: Path, manifest: dict[str, Any], bound: dict[str, str],
    affected: dict[str, dict[str, Any]], memberships: pd.DataFrame, *, sessions: tuple[date, ...], numeric_end: date,
) -> pd.DataFrame:
    records = manifest.get("files")
    if (not isinstance(records, list) or not records or len(records) > 240
            or manifest.get("feature_profiles") != ["technical_market"]):
        raise DataReadinessError("holding requirements requires at most 240 monthly technical partitions")
    months: set[str] = set()
    paths: set[Path] = set()
    selected: list[pd.DataFrame] = []
    seen_decisions: set[str] = set()
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("partition_month"), str)
                or re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", record["partition_month"]) is None
                or record["partition_month"] in months or record.get("feature_profile") != "technical_market"
                or not isinstance(record.get("path"), str)):
            raise DataReadinessError("holding requirements duplicate or invalid monthly partition")
        month = record["partition_month"]
        months.add(month)
        path = _bound_source(root, str(manifest_path.parent / record["path"]), bound)
        if path in paths or record.get("sha256") != bound[path.relative_to(root).as_posix()]:
            raise DataReadinessError("holding requirements duplicate path or partition hash differs")
        paths.add(path)
        relevant = {identity for identity, row in affected.items()
                    if row["first_decision"][:7] <= month <= min(row["last_decision"][:7], numeric_end.isoformat()[:7])}
        if not relevant:
            continue
        frame = _projection(path, IDENTITY_COLUMNS)
        dates = pd.to_datetime(frame.session_date_et, errors="coerce")
        if (frame.empty or frame.isna().any().any() or len(frame) != record.get("rows") or dates.isna().any()
                or not dates.dt.strftime("%Y-%m").eq(month).all()
                or str(dates.min().date()) != record.get("first_session")
                or str(dates.max().date()) != record.get("last_session")
                or dates.nunique() != record.get("sessions") or frame.security_id.nunique() != record.get("securities")
                or not frame.decision_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
                or frame.decision_id.duplicated().any() or seen_decisions.intersection(frame.decision_id)):
            raise DataReadinessError("holding requirements partition identities, counts or dates differ")
        seen_decisions.update(frame.decision_id)
        frame["session_date_et"] = dates.dt.date
        frame = frame.loc[frame.security_id.isin(relevant) & frame.session_date_et.le(numeric_end)].reset_index(drop=True)
        if frame.empty:
            continue
        checked = inspect_holding_membership_windows(
            frame, memberships, sessions=sessions, horizon_sessions=10, initial_fit_end=numeric_end,
        )
        selected.append(checked.loc[checked.development_matured & checked.uncovered_holding_sessions.gt(0)])
    if not selected:
        raise DataReadinessError("holding requirements found no initial-fit uncovered decisions")
    result = pd.concat(selected, ignore_index=True)
    actual = result.groupby("security_id").size().to_dict()
    if actual != {identity: row["initial_fit_decisions"] for identity, row in affected.items()}:
        raise DataReadinessError("holding requirements per-security initial-fit counts differ")
    if result.decision_id.duplicated().any() or result.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("holding requirements decision identities are duplicated")
    for identity, group in result.groupby("security_id"):
        entry = affected[str(identity)]
        if (not set(group.ticker).issubset(entry["tickers"])
                or group.session_date_et.min() < _day(entry["first_decision"])
                or group.session_date_et.max() > _day(entry["last_decision"])):
            raise DataReadinessError("holding requirements replay differs from affected identity dates")
    return result.loc[:, [*IDENTITY_COLUMNS, "exit_session_date_et"]].sort_values(
        ["security_id", "ticker", "session_date_et"], ignore_index=True,
    )


def prepare_holding_observation_requirements(
    root: Path, *, preflight_path: Path, preflight_sha256: str, parent_request_path: Path,
    expected_decisions: int, expected_securities: int,
) -> PreparedHoldingObservations:
    """Bind sources and reproduce mature gaps before the caller opens numeric bars.

    ``preflight_sha256`` is the canonical audit identity, not its JSON file hash.
    The caller owns the workspace lease and all numeric reads/publication.
    """
    try:
        if (type(expected_decisions) is not int or type(expected_securities) is not int
                or not 1 <= expected_securities <= expected_decisions <= 1_000_000):
            raise DataReadinessError("holding requirements expected counts must be bounded positive integers")
        root = root.resolve()
        preflight_path = resolve_inside_authority(root, str(preflight_path))
        preflight_file_sha = file_sha256(preflight_path)
        preflight = _object(preflight_path)
        _identity(preflight, "audit_sha256", preflight_sha256)
        if (preflight.get("scope") != "membership_identity_only_not_bar_or_return_admission"
                or preflight.get("status") != "uncovered_holding_identity"):
            raise DataReadinessError("holding requirements requires the uncovered identity preflight")
        request = preflight["request"]
        start, end, numeric_end = (_day(request[name]) for name in ("decision_start", "snapshot_end", "initial_fit_end"))
        if (request.get("schema") != "market_predictor.swing_holding_identity_preflight_request"
                or type(request.get("horizon_sessions")) is not int or request["horizon_sessions"] != 10
                or not start <= numeric_end <= end):
            raise DataReadinessError("holding requirements preflight dates or horizon differ")
        bound: dict[str, str] = {}
        sources = preflight.get("source_files")
        if not isinstance(sources, dict) or not sources:
            raise DataReadinessError("holding requirements preflight source inventory is missing")
        for relative, digest in sources.items():
            _checked(root, relative, digest, bound)
        cohort_path = _bound_source(root, request["cohort_path"], bound)
        cohort = load_swing_research_cohort(cohort_path, source_root=root)
        if (cohort.sha256() != request["cohort_sha256"] or preflight.get("cohort_sha256") != cohort.sha256()
                or not cohort.within_cap or any(bound.get(path) != digest for path, digest in cohort.source_files.items())):
            raise DataReadinessError("holding requirements accepted cohort binding differs")
        parent = _object(_bound_source(root, str(parent_request_path), bound))
        _identity(parent, "request_sha256", parent["request_sha256"])
        manifest_path = _bound_source(root, request["parent_manifest_path"], bound)
        manifest = _object(manifest_path)
        if (manifest.get("request_sha256") != parent["request_sha256"]
                or parent.get("decision_start_date") != start.isoformat()
                or json_sha256(parent["combined_daily_inputs"]) != cohort.combined_daily_inputs_sha256):
            raise DataReadinessError("holding requirements parent request is not bound to cohort or manifest")
        affected = _affected(preflight, expected_decisions, expected_securities)
        if not set(affected).issubset(cohort.retained_security_ids):
            raise DataReadinessError("holding requirements affected identities are not retained cohort members")
        raw_manifest, symbols = _raw_inventory(root, parent, bound, start=start, end=end)
        memberships = _projection(_bound_source(root, request["membership_path"], bound), MEMBERSHIP_COLUMNS)
        if set(memberships.security_id) != set(cohort.original_security_ids).union(cohort.warmup_only_security_ids):
            raise DataReadinessError("holding requirements requires full, unfiltered memberships including excluded owners")
        sessions = holding_calendar(start, end)
        decisions = _decisions(root, manifest_path, manifest, bound, affected, memberships, sessions=sessions, numeric_end=numeric_end)
        if len(decisions) != expected_decisions or decisions.security_id.nunique() != expected_securities:
            raise DataReadinessError("holding requirements replay totals differ")
        ordinal = {day: index for index, day in enumerate(sessions)}
        requirements: list[dict[str, Any]] = []
        for (identity, ticker), group in decisions.groupby(["security_id", "ticker"], sort=True):
            needed: set[date] = set()
            for row in group.itertuples(index=False):
                index = ordinal[row.session_date_et]
                future = sessions[index + 1:index + 11]
                if len(future) != 10 or future[-1] != row.exit_session_date_et or future[-1] > numeric_end:
                    raise DataReadinessError("holding requirements exact ten-session maturity differs")
                needed.update(future)
            if ticker not in symbols:
                raise DataReadinessError("holding requirements ticker is absent from the raw request")
            requirements.append({"security_id": str(identity), "ticker": str(ticker), "sessions": sorted(needed)})
        _checked(root, str(preflight_path), preflight_file_sha, bound)
        for relative, digest in list(bound.items()):
            _checked(root, relative, digest, bound)
        return PreparedHoldingObservations(bound, decisions, requirements, memberships, raw_manifest, numeric_end)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise DataReadinessError(f"invalid holding observation requirements: {exc}") from exc
