"""Reconstruct initial-fit raw-price requests from pinned decision identities."""
from __future__ import annotations

import hashlib
import json
import tomllib
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.joins import MEMBERSHIP_VALUE_COLUMNS, join_universe_membership
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget, release_process_memory
from market_predictor.sources.provider_symbols import PROVIDER_ALPACA, provider_symbol
from market_predictor.swing.contracts.research_cohort import Sha256, load_swing_research_cohort
from market_predictor.swing.datasets.history_plan_publication import (
    AUTHORITY_SCHEMA,
    DAILY_BAR_UNITS_FILE,
    PLAN_SCHEMA,
    UNIT_COLUMNS,
    publish_daily_history_plan,
)
from market_predictor.swing.labels.holding_paths import holding_calendar

IDENTITY_COLUMNS = (
    "decision_id", "security_id", "ticker", "sector", "primary_benchmark", "session_date_et", "decision_time_utc",
)
MEMBERSHIP_COLUMNS = tuple(dict.fromkeys((
    "ticker", *MEMBERSHIP_VALUE_COLUMNS, "effective_from_utc", "effective_to_utc", "available_at_utc",
)))
_MAX_METADATA = 8 * 1024**2


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["market_predictor.swing_initial_fit_raw_share_plan_request"] = Field(alias="schema")
    preflight_path: str = Field(min_length=1)
    preflight_sha256: Sha256
    parent_request_path: str = Field(min_length=1)


def _guard() -> None:
    assert_memory_budget(hard_budget_gib=4.0, headroom_gib=0.75, stage="initial-fit raw-share plan")
    assert_peak_memory_budget(hard_budget_gib=4.0, headroom_gib=0.75, stage="initial-fit raw-share plan")


def _object(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if path.stat().st_size > _MAX_METADATA:
        raise DataReadinessError("initial-fit plan metadata exceeds bound")
    payload = path.read_bytes()
    if len(payload) > _MAX_METADATA or expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DataReadinessError("initial-fit plan pinned metadata changed")
    return parse_strict_json_object(payload, label=str(path))


def _inside(root: Path, path: Path) -> Path:
    result = (root / path).resolve()
    if result == root or not result.is_relative_to(root):
        raise DataReadinessError("initial-fit plan path escapes repository or targets root")
    return result


def _projection(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    arrow: Any = pq
    with arrow.ParquetFile(path) as source:
        if (source.metadata.num_rows > 1_000_000 or source.metadata.serialized_size > _MAX_METADATA
                or not set(columns).issubset(source.schema_arrow.names)):
            raise DataReadinessError("initial-fit plan identity projection is invalid or unbounded")
        frame: pd.DataFrame = source.read(columns=list(columns), use_threads=False).to_pandas()
        return frame


def _recheck(root: Path, bound: dict[str, str]) -> None:
    for relative, digest in bound.items():
        if file_sha256(resolve_inside_authority(root, relative)) != digest:
            raise DataReadinessError(f"initial-fit plan source hash differs: {relative}")


def _runs(requirements: dict[tuple[str, str], set[int]], sessions: tuple[date, ...]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for (identity, ticker), values in sorted(requirements.items()):
        ordered = sorted(values)
        first = previous = ordered[0]
        for index in ordered[1:] + [len(sessions) + 1]:
            if index != previous + 1:
                records.append({"security_id": identity, "ticker": ticker, "start_date": sessions[first].isoformat(),
                    "end_date": sessions[previous].isoformat(), "role": "stock"})
                first = index
            previous = index
    return pd.DataFrame(records, columns=list(UNIT_COLUMNS))


def _reconstruct(root: Path, config: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    policy = _Request.model_validate_json(json.dumps(tomllib.loads(config.read_text(encoding="utf-8"))))
    preflight_path = resolve_inside_authority(root, policy.preflight_path)
    preflight = _object(preflight_path)
    if (preflight.get("audit_sha256") != policy.preflight_sha256
            or json_sha256({k: v for k, v in preflight.items() if k != "audit_sha256"}) != policy.preflight_sha256
            or preflight.get("scope") != "membership_identity_only_not_bar_or_return_admission"
            or preflight.get("outcome_columns_read") is not False):
        raise DataReadinessError("initial-fit plan preflight pin or scope differs")
    base = preflight["request"]
    cohort_path = resolve_inside_authority(root, base["cohort_path"])
    cohort = load_swing_research_cohort(cohort_path, source_root=root)
    if (cohort.sha256() != base["cohort_sha256"] or cohort.sha256() != preflight["cohort_sha256"]
            or not cohort.within_cap or cohort.maximum_exclusion_bps > 1000 or base["horizon_sessions"] != 10):
        raise DataReadinessError("initial-fit plan requires pinned accepted cohort and ten-session horizon")
    start, end = date.fromisoformat(base["decision_start"]), date.fromisoformat(base["initial_fit_end"])
    if not date(2019, 7, 9) <= start <= end <= date.fromisoformat(base["snapshot_end"]):
        raise DataReadinessError("initial-fit plan dates violate the frozen scope")
    sessions = holding_calendar(start, end)
    if len(sessions) <= 10:
        raise DataReadinessError("initial-fit plan has no mature ten-session decisions")
    bound = dict(preflight["source_files"])
    for name, digest in cohort.source_files.items():
        if bound.get(name) != digest:
            raise DataReadinessError("initial-fit plan cohort/preflight source binding differs")
    bound[preflight_path.relative_to(root).as_posix()] = file_sha256(preflight_path)
    bound[config.relative_to(root).as_posix()] = file_sha256(config)
    _recheck(root, bound)

    def source(relative: str) -> Path:
        path = resolve_inside_authority(root, relative)
        key = path.relative_to(root).as_posix()
        if key not in cohort.source_files or bound.get(key) != file_sha256(path):
            raise DataReadinessError(f"initial-fit plan source is not cohort-bound: {relative}")
        return path

    parent = _object(source(policy.parent_request_path))
    combined = parent["combined_daily_inputs"]
    if json_sha256(combined) != cohort.combined_daily_inputs_sha256:
        raise DataReadinessError("initial-fit plan combined parent identity differs")
    membership_path = source(base["membership_path"])
    lineage = combined["membership_authority"]
    if lineage["membership_artifact_sha256"] != file_sha256(membership_path):
        raise DataReadinessError("initial-fit plan membership lineage differs")
    memberships = _projection(membership_path, MEMBERSHIP_COLUMNS)
    if set(memberships.security_id) != set(cohort.original_security_ids).union(cohort.warmup_only_security_ids):
        raise DataReadinessError("initial-fit plan needs full membership including competing owners")
    manifest_path = source(base["parent_manifest_path"])
    parent_manifest = _object(manifest_path)
    records = parent_manifest.get("files")
    if (not isinstance(records, list) or not records or len(records) > 240
            or parent_manifest.get("feature_profiles") != ["technical_market"]):
        raise DataReadinessError("initial-fit plan requires bounded technical identity partitions")
    ordinal = {day: index for index, day in enumerate(sessions)}
    decisions: dict[tuple[str, str], set[int]] = {}
    holdings: dict[tuple[str, str], set[int]] = {}
    seen_ids: set[str] = set()
    seen_months: set[str] = set()
    inspected: list[dict[str, Any]] = []
    mature_count = 0
    in_window_count = 0
    for record in records:
        month = record["partition_month"]
        if month in seen_months or record["feature_profile"] != "technical_market":
            raise DataReadinessError("initial-fit plan duplicate or invalid partition identity")
        seen_months.add(month)
        if record["first_session"] > end.isoformat() or record["last_session"] < start.isoformat():
            continue
        path = source((manifest_path.parent / record["path"]).relative_to(root).as_posix())
        if file_sha256(path) != record["sha256"]:
            raise DataReadinessError("initial-fit plan partition hash differs")
        _guard()
        frame = _projection(path, IDENTITY_COLUMNS)
        dates = pd.to_datetime(frame.session_date_et, errors="coerce")
        if (len(frame) != record["rows"] or frame.isna().any().any() or dates.isna().any()
                or not dates.dt.strftime("%Y-%m").eq(month).all()
                or str(dates.min().date()) != record["first_session"] or str(dates.max().date()) != record["last_session"]
                or frame.decision_id.duplicated().any() or seen_ids.intersection(frame.decision_id)
                or frame.duplicated(["security_id", "session_date_et"]).any()
                or not set(frame.security_id).issubset(
                    set(cohort.original_security_ids).difference(cohort.inherited_excluded_security_ids))):
            raise DataReadinessError("initial-fit plan partition identity or row inventory differs")
        seen_ids.update(frame.decision_id)
        frame = frame.loc[dates.dt.date.between(start, end) & frame.security_id.isin(cohort.retained_security_ids)].copy()
        frame["session_date_et"] = pd.to_datetime(frame.session_date_et).dt.date
        if not frame.decision_time_utc.eq(swing_prediction_cutoffs(frame.session_date_et)).all():
            raise DataReadinessError("initial-fit plan decision cutoff differs")
        if not frame.empty:
            joined = join_universe_membership(frame, memberships).set_index("decision_id")
            expected = frame.set_index("decision_id")
            if any(not joined[name].reindex(expected.index).eq(expected[name]).all()
                    for name in ("security_id", "sector", "primary_benchmark")):
                raise DataReadinessError("initial-fit plan decision membership differs")
        in_window_count += len(frame)
        for (identity, ticker), group in frame.groupby(["security_id", "ticker"], sort=True):
            key = (str(identity), str(ticker))
            indices = group.session_date_et.map(ordinal)
            if indices.isna().any():
                raise DataReadinessError("initial-fit plan decision is not an exchange session")
            decision_set = decisions.setdefault(key, set())
            holding_set = holdings.setdefault(key, set())
            for value in indices:
                index = int(value)
                decision_set.add(index)
                if index + 10 < len(sessions):
                    mature_count += 1
                    holding_set.update(range(index + 1, index + 11))
        inspected.append({"month": month, "retained_decisions": len(frame), "sha256": record["sha256"]})
        del frame
        release_process_memory()
    if mature_count != preflight["initial_fit"]["decisions"] or not decisions:
        raise DataReadinessError("initial-fit plan mature count differs from pinned preflight")
    required = {key: values | holdings[key] for key, values in decisions.items()}
    stock_units = _runs(required, sessions)
    active = memberships.loc[memberships.security_id.isin(cohort.retained_security_ids)].copy()
    starts = pd.to_datetime(active.effective_from_utc, utc=True)
    ends = pd.to_datetime(active.effective_to_utc, utc=True)
    active = active.loc[starts.lt(pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=1)) &
        (ends.isna() | ends.gt(pd.Timestamp(start, tz="America/New_York")))]
    benchmarks = sorted({"SPY", "QQQ", *active.primary_benchmark.astype(str)})
    if any(not value or value != value.strip().upper() for value in benchmarks):
        raise DataReadinessError("initial-fit plan benchmark identity is invalid")
    benchmark_units = pd.DataFrame([{"security_id": f"benchmark:{ticker}", "ticker": ticker,
        "start_date": start.isoformat(), "end_date": end.isoformat(), "role": "benchmark"}
        for ticker in benchmarks], columns=list(UNIT_COLUMNS))
    units = pd.concat([stock_units, benchmark_units], ignore_index=True).sort_values(
        ["role", "ticker", "start_date", "security_id"], kind="stable").reset_index(drop=True)
    required_count = sum(map(len, required.values()))
    unit_audit: dict[str, Any] = {}
    expanded: dict[tuple[str, str], set[int]] = {}
    for raw in units.to_dict(orient="records"):
        unit = {str(key): str(value) for key, value in raw.items()}
        indices = set(range(bisect_left(sessions, date.fromisoformat(unit["start_date"])),
            bisect_right(sessions, date.fromisoformat(unit["end_date"]))))
        key = (unit["security_id"], unit["ticker"])
        is_stock = unit["role"] == "stock"
        d = decisions[key] & indices if is_stock else set()
        h = holdings[key] & indices if is_stock else set()
        if is_stock:
            seen = expanded.setdefault(key, set())
            if seen & indices or d | h != indices:
                raise DataReadinessError("initial-fit plan exact session-run expansion differs")
            seen.update(indices)
        unit_audit[f"swing-daily-{json_sha256(unit)[:24]}"] = {
            "identity": unit, "decision_sessions": len(d), "holding_sessions": len(h),
            "decision_and_holding_sessions": len(d | h), "benchmark_sessions": 0 if is_stock else len(indices),
            "required_sessions": len(indices),
            "required_sessions_sha256": json_sha256([sessions[index].isoformat() for index in sorted(indices)]),
        }
    if expanded != required or len(unit_audit) != len(units):
        raise DataReadinessError("initial-fit plan exact unit/session inventory differs")
    package = Path(__file__).resolve().parents[2]
    request = {"schema": PLAN_SCHEMA, "scope": "initial_fit_raw_share_acquisition", "policy": policy.model_dump(mode="json", by_alias=True),
        "source_files": bound, "membership_authority": lineage, "cohort_sha256": cohort.sha256(),
        "retained_security_ids": list(cohort.retained_security_ids), "excluded_security_ids": list(cohort.excluded_security_ids),
        "decision_start": start.isoformat(), "initial_fit_end": end.isoformat(), "horizon_sessions": 10,
        "decision_policy": "all_in_window_decisions_future_paths_only_when_mature",
        "asof_policy": "inclusive_unit_end_date_entity_mapping_not_ownership",
        "provider_symbols": {ticker: provider_symbol(ticker, PROVIDER_ALPACA) for ticker in sorted(set(units.ticker))},
        "calendar_sha256": json_sha256([day.isoformat() for day in sessions]),
        "requirements_sha256": json_sha256([{ "security_id": key[0], "ticker": key[1], "session_ordinals": sorted(values)}
            for key, values in sorted(required.items())]),
        "unit_requirements_sha256": json_sha256(unit_audit),
        "implementation_files": {name: file_sha256(package / name) for name in (
            "swing/datasets/initial_fit_raw_share_plan.py", "swing/datasets/history_plan_publication.py",
            "swing/labels/holding_paths.py", "canonical/cutoffs.py", "canonical/joins.py", "sources/provider_symbols.py")}}
    result = {"schema": PLAN_SCHEMA, "status": "ready_for_daily_history_collection", "outcomes_read": False,
        "scope": request["scope"], "missing_session_ranges": [{"first_session": start.isoformat(),
            "last_session": end.isoformat(), "sessions": len(sessions)}],
        "membership": {"universe_sha256": lineage["universe_sha256"], "parent_lineage": lineage["parent_lineage"]},
        "daily_bars": {"status": "ready", "source": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": "raw",
            "planned_units": len(units), "stock_units": len(stock_units), "benchmark_units": len(benchmark_units)},
        "requirements": {"retained_securities": len(cohort.retained_security_ids),
            "in_window_securities": len({key[0] for key in decisions}),
            "zero_requirement_security_ids": sorted(set(cohort.retained_security_ids).difference(key[0] for key in decisions)),
            "in_window_decisions": in_window_count, "mature_decisions": mature_count,
            "decision_sessions": sum(map(len, decisions.values())), "holding_sessions": sum(map(len, holdings.values())),
            "decision_only_sessions": sum(len(decisions[key] - holdings[key]) for key in decisions),
            "decision_and_holding_sessions": required_count, "benchmarks": benchmarks, "partitions": inspected},
        "unit_requirements": unit_audit,
        "ownership_admitted": False, "bar_coverage_verified": False, "accounting_eligible": False, "promotion_eligible": False}
    _recheck(root, bound)
    _guard()
    return request, result, units


def _verify_existing(output: Path, request: dict[str, Any], manifest: dict[str, Any], units: pd.DataFrame,
    *, expected_sha256: str) -> dict[str, Any]:
    stored = _object(output / "_manifest.json")
    authority = _object(output / "_authority.json", expected_sha256=expected_sha256)
    units_path = output / DAILY_BAR_UNITS_FILE
    if (_object(output / "_request.json") != request or authority != {
        "schema": AUTHORITY_SCHEMA, "state": "complete", "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(output / "_manifest.json"), "request_sha256": file_sha256(output / "_request.json"),
        "units_sha256": file_sha256(units_path), "universe_sha256": request["membership_authority"]["universe_sha256"]}):
        raise DataReadinessError("initial-fit plan replay request or authority differs")
    expected = {**manifest, "request_sha256": file_sha256(output / "_request.json"), "resources": stored.get("resources")}
    expected["daily_bars"] = {**manifest["daily_bars"], "units_artifact": {
        "path": DAILY_BAR_UNITS_FILE, "bytes": units_path.stat().st_size, "sha256": file_sha256(units_path)}}
    if stored != expected or units_path.read_bytes() != units.to_csv(index=False, lineterminator="\n").encode("utf-8"):
        raise DataReadinessError("initial-fit plan replay requirements differ")
    return stored


@contextmanager
def verified_initial_fit_raw_share_plan(root: Path, config_path: Path, output_directory: Path, *,
    expected_plan_sha256: str | None = None) -> Iterator[dict[str, Any]]:
    """Hold the workspace lease across reconstruction and optional downstream collection."""
    root = root.resolve()
    with heavy_job_lease("initial-fit-raw-share-plan", runtime_dir=_inside(root, heavy_job_runtime_dir())):
        try:
            _guard()
            config = resolve_inside_authority(root, str(config_path))
            if config.stat().st_size > 1024**2:
                raise DataReadinessError("initial-fit plan configuration exceeds bound")
            output = _inside(root, output_directory)
            if output.exists():
                if expected_plan_sha256 is None or file_sha256(output / "_authority.json") != expected_plan_sha256:
                    raise DataReadinessError("initial-fit plan requires its independent authority pin for replay")
            elif expected_plan_sha256 is not None:
                raise DataReadinessError("initial-fit plan expected pin cannot create a replacement")
            request, manifest, units = _reconstruct(root, config)
            if output.exists():
                pin = expected_plan_sha256 or ""
                manifest = _verify_existing(output, request, manifest, units, expected_sha256=pin)
            else:
                if any(resolve_inside_authority(root, path).is_relative_to(output) for path in request["source_files"]):
                    raise DataReadinessError("initial-fit plan output contains a source")
                manifest = publish_daily_history_plan(output=output, request=request, manifest=manifest, units=units)
                pin = file_sha256(output / "_authority.json")
                _verify_existing(output, request, {k: v for k, v in manifest.items() if k not in {"request_sha256", "resources"}},
                    units, expected_sha256=pin)
            if file_sha256(output / "_authority.json") != pin:
                raise DataReadinessError("initial-fit plan authority changed before use")
            yield {**manifest, "plan_sha256": pin, "provider_symbols": request["provider_symbols"]}
            if file_sha256(output / "_authority.json") != pin:
                raise DataReadinessError("initial-fit plan authority changed during use")
            _recheck(root, request["source_files"])
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise DataReadinessError(f"invalid initial-fit raw-share plan: {exc}") from exc


def run_initial_fit_raw_share_plan(root: Path, config_path: Path, output_directory: Path, *,
    expected_plan_sha256: str | None = None) -> dict[str, Any]:
    with verified_initial_fit_raw_share_plan(root, config_path, output_directory, expected_plan_sha256=expected_plan_sha256) as plan:
        return plan
