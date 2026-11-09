"""Investment-owned exact raw acquisition; captured stock candidates are quarantined.

The source-gap inventory is an independently pinned audit, not a price or ownership
authority. Reconstruct its session requirements from the frozen roster and selected
raw metadata before using the shared immutable daily transport. Public operations
own the workspace lease; the shared plan callback must not acquire another lease.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import history_archive as archive
from market_predictor.swing.datasets.history_plan_publication import (
    AUTHORITY_SCHEMA,
    PLAN_SCHEMA,
    UNIT_COLUMNS,
    publish_daily_history_plan,
)
from market_predictor.swing.labels.holding_paths import holding_calendar

SCOPE = "investment_horizon_raw_acquisition"
OUTCOME_END = date(2025, 5, 30)
FROZEN_SWING_END = date(2024, 5, 28)
QUARANTINE = "quarantined_unreviewed_provider_candidate"
_MAX_JSON = 32 * 1024**2


class InvestmentRawAcquisitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["market_predictor.investment_raw_acquisition_request"] = Field(alias="schema")
    gap_inventory: SourcePin
    source_selection: SourcePin
    roster: SourcePin
    parent_plan_authority: SourcePin
    action_query: SourcePin
    outcome_end: date


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_JSON:
        raise DataReadinessError("investment raw acquisition JSON exceeds its bound")
    return parse_strict_json_object(path.read_bytes(), label=str(path))


def _guard() -> None:
    assert_memory_budget(hard_budget_gib=4.0, headroom_gib=0.75, stage="investment raw acquisition")
    assert_peak_memory_budget(hard_budget_gib=4.0, headroom_gib=0.75, stage="investment raw acquisition")


def _inside(root: Path, path: Path) -> Path:
    result = (root / path).resolve()
    if result == root or not result.is_relative_to(root):
        raise DataReadinessError("investment raw acquisition path escapes source root")
    return result


def _recheck(root: Path, files: dict[str, str]) -> None:
    for name, digest in files.items():
        if file_sha256(resolve_inside_authority(root, name)) != digest:
            raise DataReadinessError(f"investment raw acquisition pinned source changed: {name}")


def _runs(days: set[date], calendar: tuple[date, ...]) -> list[tuple[date, date]]:
    positions = [index for index, day in enumerate(calendar) if day in days]
    result: list[tuple[date, date]] = []
    if positions:
        first = previous = positions[0]
        for index in positions[1:]:
            if index != previous + 1:
                result.append((calendar[first], calendar[previous]))
                first = index
            previous = index
        result.append((calendar[first], calendar[previous]))
    if sum(len(holding_calendar(first, last)) for first, last in result) != len(days):
        raise DataReadinessError("investment acquisition missing dates escape XNYS calendar")
    return result


def _selection_dates(selection: dict[str, Any], root: Path, bound: dict[str, str]
                     ) -> tuple[dict[str, set[date]], dict[str, dict[str, Any]]]:
    if (selection.get("schema") != "market_predictor.swing_symbol_corrected_sources"
            or selection.get("status") != "source_selection_verified" or selection.get("exclusions_added") != []
            or selection.get("audit_sha256") != json_sha256({k: v for k, v in selection.items() if k != "audit_sha256"})):
        raise DataReadinessError("investment acquisition requires unchanged verified parent source selection")
    dates: dict[str, set[date]] = defaultdict(set)
    latest: dict[str, dict[str, Any]] = {}
    for segment in selection["segments"]:
        artifact = segment["artifact"]
        identity, ticker = str(artifact["security_id"]), str(artifact["ticker"])
        first, last = date.fromisoformat(segment["first_session"]), date.fromisoformat(segment["last_session"])
        if (not first <= last <= FROZEN_SWING_END or artifact["role"] not in {"stock", "benchmark"}
                or artifact["status"] != "observed" or ticker != ticker.strip().upper()):
            raise DataReadinessError("investment acquisition parent segment identity or frozen boundary differs")
        source = _inside(root, Path(segment["archive"]))
        for path_key, hash_key in (("bars_path", "bars_sha256"), ("unit_manifest_path", "unit_manifest_sha256")):
            path = resolve_inside_authority(source, artifact[path_key])
            relative, digest = path.relative_to(root).as_posix(), str(artifact[hash_key])
            if relative in bound and bound[relative] != digest:
                raise DataReadinessError("investment acquisition parent artifact has conflicting hashes")
            bound[relative] = digest
        days = set(holding_calendar(first, last))
        missing = {date.fromisoformat(day) for day in segment["missing_sessions"]}
        if not missing.issubset(days):
            raise DataReadinessError("investment acquisition parent missing dates escape segment")
        dates[identity].update(days - missing)
        previous = latest.get(identity)
        if previous is None or segment["last_session"] > previous["last_session"]:
            latest[identity] = segment
        elif segment["last_session"] == previous["last_session"] and (
            artifact["ticker"], artifact["provider_symbol"]) != (
                previous["artifact"]["ticker"], previous["artifact"]["provider_symbol"]):
            raise DataReadinessError("investment acquisition latest provider candidate is ambiguous")
    return dates, latest


def prepare_investment_raw_acquisition(root: Path, request_path: Path
                                     ) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Reconstruct metadata-only requirements inside an already leased operation."""
    root = root.resolve()
    _guard()
    config = resolve_inside_authority(root, str(request_path))
    policy = InvestmentRawAcquisitionRequest.model_validate_json(json.dumps(_object(config)))
    if policy.outcome_end != OUTCOME_END:
        raise DataReadinessError("investment acquisition outcome end must remain May 30, 2025")
    bound: dict[str, str] = {config.relative_to(root).as_posix(): file_sha256(config)}

    def pinned(pin: SourcePin) -> dict[str, Any]:
        path = resolve_inside_authority(root, pin.path)
        if file_sha256(path) != pin.sha256:
            raise DataReadinessError("investment acquisition independent source pin differs")
        bound[path.relative_to(root).as_posix()] = pin.sha256
        return _object(path)

    gap, selection, roster = pinned(policy.gap_inventory), pinned(policy.source_selection), pinned(policy.roster)
    parent_authority, action_query = pinned(policy.parent_plan_authority), pinned(policy.action_query)
    if (gap.get("schema") != "read_only_investment_raw_tail_assessment"
            or gap.get("old_raw_end") != FROZEN_SWING_END.isoformat()
            or gap.get("last_required_end_252") != OUTCOME_END.isoformat()
            or gap.get("last_decision") != FROZEN_SWING_END.isoformat()
            or gap.get("no_prices_returns_or_financial_values_read") is not True):
        raise DataReadinessError("investment acquisition gap audit scope differs")
    for pin in (policy.source_selection, policy.roster, policy.action_query):
        relative = resolve_inside_authority(root, pin.path).relative_to(root).as_posix()
        if gap["source_pins"].get(relative) != pin.sha256:
            raise DataReadinessError("investment acquisition audit and explicit source pins differ")
    bound.update({str(name): str(digest) for name, digest in gap["source_pins"].items()})
    parent_directory = resolve_inside_authority(root, policy.parent_plan_authority.path).parent
    parent_request_path = parent_directory / "_request.json"
    parent_manifest_path = parent_directory / "_manifest.json"
    parent_request, parent_manifest = _object(parent_request_path), _object(parent_manifest_path)
    if (parent_authority.get("schema") != AUTHORITY_SCHEMA or parent_authority.get("state") != "complete"
            or parent_authority.get("artifact") != "_manifest.json"
            or parent_authority.get("artifact_sha256") != file_sha256(parent_manifest_path)
            or parent_authority.get("request_sha256") != file_sha256(parent_request_path)
            or parent_request.get("scope") != "initial_fit_raw_share_acquisition"
            or parent_request.get("initial_fit_end") != FROZEN_SWING_END.isoformat()
            or parent_manifest.get("daily_bars", {}).get("adjustment") != "raw"
            or selection.get("parent_plan_sha256") != policy.parent_plan_authority.sha256):
        raise DataReadinessError("investment acquisition parent raw authority or frozen end differs")
    for path in (parent_request_path, parent_manifest_path):
        bound[path.relative_to(root).as_posix()] = file_sha256(path)
    action_scope = action_query.get("policy", {})
    if (action_scope.get("process_start") != "2018-05-29" or action_scope.get("process_end") != "2026-09-11"
            or action_scope.get("numeric_end") != FROZEN_SWING_END.isoformat()
            or gap.get("missing_action_query_symbols") != []):
        raise DataReadinessError("investment acquisition retained action query coverage differs")
    observed, latest = _selection_dates(selection, root, bound)
    calendar = holding_calendar(date(2019, 7, 9), OUTCOME_END)
    ordinals = {day: index for index, day in enumerate(calendar)}
    required: dict[str, set[date]] = defaultdict(set)
    required_benchmarks: dict[str, set[date]] = defaultdict(set)
    eligible = [row for row in roster["rows"] if row["status"] == "eligible"]
    seen: set[str] = set()
    retained = set(parent_request["retained_security_ids"])
    excluded = set(parent_request["excluded_security_ids"])
    for row in eligible:
        identity, decision_id = str(row["security_id"]), str(row["decision_id"])
        decision = date.fromisoformat(row["decision_session"])
        if (decision_id in seen or identity not in retained or identity in excluded
                or decision not in ordinals or not date(2019, 7, 31) <= decision <= FROZEN_SWING_END
                or identity not in latest or latest[identity]["artifact"]["role"] != "stock"):
            raise DataReadinessError("investment acquisition eligible roster identity or date differs")
        seen.add(decision_id)
        index = ordinals[decision]
        days = set(calendar[index + 1:index + 253])
        if len(days) != 252:
            raise DataReadinessError("investment acquisition does not cover the exact 252-session horizon")
        required[identity].update(days)
        for symbol in ("SPY", "QQQ", str(row["sector_benchmark"])):
            required_benchmarks[symbol].update(days)
    if (len(eligible) != gap["eligible_security_months"] or len(required) != gap["eligible_unique_security_ids"]
            or sorted(required_benchmarks) != sorted(gap["benchmark_tickers"])):
        raise DataReadinessError("investment acquisition full eligible roster inventory differs")
    expected_gaps = {record["security_id"]: record for record in gap["all_horizon_metadata_coverage"]["missing_ranges"]}
    if len(expected_gaps) != len(gap["all_horizon_metadata_coverage"]["missing_ranges"]):
        raise DataReadinessError("investment acquisition gap audit repeats a security")
    rows: list[dict[str, str]] = []
    details: dict[str, Any] = {}
    provider_symbols: dict[str, str] = {}
    stock_sessions = benchmark_sessions = 0
    cautions = {item["security_id"] for item in gap["identity_warnings"]}
    earlier: list[str] = []
    for identity, wanted in sorted({**required, **{f"benchmark:{k}": v for k, v in required_benchmarks.items()}}.items()):
        if identity not in latest:
            raise DataReadinessError("investment acquisition has no retained candidate identity")
        candidate = latest[identity]
        artifact = candidate["artifact"]
        role, ticker, symbol = str(artifact["role"]), str(artifact["ticker"]), str(artifact["provider_symbol"])
        if role == "benchmark" and (identity != f"benchmark:{ticker}" or symbol != ticker):
            raise DataReadinessError("investment acquisition benchmark candidate differs")
        missing = wanted - observed[identity]
        runs = _runs(missing, calendar)
        if role == "stock":
            audit = expected_gaps.pop(identity, None)
            expected_ranges = [{"first_session": str(first), "last_session": str(last),
                "sessions": len(holding_calendar(first, last))} for first, last in runs]
            if (audit is None or audit["ranges"] != expected_ranges or audit["missing_source_session_count"] != len(missing)
                    or audit["ticker_at_latest_retained_segment"] != ticker or audit["provider_symbol_candidate"] != symbol):
                raise DataReadinessError("investment acquisition reconstructed exact stock gaps differ from audit")
            stock_sessions += len(missing)
            if any(day <= FROZEN_SWING_END for day in missing):
                earlier.append(identity)
        else:
            benchmark_sessions += len(missing)
        for first, last in runs:
            row = {"security_id": identity, "ticker": ticker, "start_date": str(first), "end_date": str(last), "role": role}
            unit_id = "swing-daily-" + json_sha256(row)[:24]
            unit_sessions = [day.isoformat() for day in holding_calendar(first, last)]
            rows.append(row)
            if ticker in provider_symbols and provider_symbols[ticker] != symbol:
                raise DataReadinessError("investment acquisition provider-symbol mapping is ambiguous")
            provider_symbols[ticker] = symbol
            details[unit_id] = {"identity": row, "provider_symbol": symbol, "asof": str(last),
                "required_sessions": len(unit_sessions), "required_sessions_sha256": json_sha256(unit_sessions),
                "candidate_parent_segment": candidate, "identity_admission": QUARANTINE if role == "stock" else "captured_only",
                "tail_identity_caution": identity in cautions, "ownership_admitted": False}
    if (expected_gaps or stock_sessions != gap["all_horizon_metadata_coverage"]["missing_stock_security_sessions_total"]
            or benchmark_sessions != gap["nominal_benchmark_sessions"]
            or stock_sessions + benchmark_sessions != gap["nominal_full_missing_stock_plus_benchmark_sessions"]):
        raise DataReadinessError("investment acquisition exact missing-session totals differ")
    units = pd.DataFrame(rows, columns=list(UNIT_COLUMNS)).sort_values(
        ["role", "ticker", "start_date", "security_id"], kind="stable").reset_index(drop=True)
    lineage = parent_request["membership_authority"]
    package = Path(__file__).resolve().parents[1]
    request = {"schema": PLAN_SCHEMA, "scope": SCOPE, "source_root": str(root),
        "request_path": config.relative_to(root).as_posix(), "policy": policy.model_dump(mode="json", by_alias=True),
        "source_files": bound, "membership_authority": lineage, "provider_symbols": provider_symbols,
        "asof_policy": "inclusive_unit_end_date_entity_mapping_not_ownership", "decision_start": "2019-07-31",
        "decision_end": FROZEN_SWING_END.isoformat(), "outcome_end": OUTCOME_END.isoformat(), "horizons": [63, 252],
        "workers": 1, "stock_identity_admission": QUARANTINE, "benchmark_identity_admission": "captured_only",
        "retained_security_ids": parent_request["retained_security_ids"], "excluded_security_ids": parent_request["excluded_security_ids"],
        "unit_requirements_sha256": json_sha256(details), "implementation_files": {
            name: file_sha256(package / name) for name in (
                "investment/raw_acquisition.py", "swing/datasets/history_archive.py",
                "swing/datasets/history_plan_publication.py", "swing/labels/holding_paths.py")}}
    manifest = {"schema": PLAN_SCHEMA, "scope": SCOPE, "status": "ready_for_daily_history_collection",
        "outcomes_read": False, "ownership_admitted": False, "accounting_eligible": False, "label_eligible": False,
        "promotion_eligible": False, "historical_availability_proven": False, "outcome_end": OUTCOME_END.isoformat(),
        "membership": {"universe_sha256": lineage["universe_sha256"], "parent_lineage": lineage["parent_lineage"]},
        "missing_session_ranges": [{"first_session": min(row["start_date"] for row in rows),
            "last_session": max(row["end_date"] for row in rows)}],
        "daily_bars": {"status": "ready", "source": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": "raw",
            "planned_units": len(units), "stock_units": int(units.role.eq("stock").sum()),
            "benchmark_units": int(units.role.eq("benchmark").sum())},
        "requirements": {"eligible_stock_months": len(eligible), "stock_security_ids": len(required),
            "stock_sessions": stock_sessions, "benchmark_sessions": benchmark_sessions,
            "tail_identity_caution_ids": sorted(cautions), "historical_gap_security_ids": sorted(earlier)},
        "unit_requirements": details}
    _recheck(root, bound)
    _guard()
    return request, manifest, units


def validate_investment_raw_acquisition_plan(*, directory: Path, request: dict[str, Any],
                                           manifest: dict[str, Any], units: pd.DataFrame) -> None:
    """Shared transport callback: independently replay exact owner requirements."""
    root = Path(request["source_root"])
    expected_request, expected_manifest, expected_units = prepare_investment_raw_acquisition(root, Path(request["request_path"]))
    actual = {key: value for key, value in manifest.items() if key not in {"request_sha256", "resources"}}
    actual["daily_bars"] = {key: value for key, value in manifest["daily_bars"].items() if key != "units_artifact"}
    if (request != expected_request or actual != expected_manifest
            or units[list(UNIT_COLUMNS)].reset_index(drop=True).equals(expected_units) is False):
        raise DataReadinessError("investment raw acquisition plan differs from exact independent requirements")


def _plan(root: Path, request_path: Path, output: Path, expected_pin: str | None) -> dict[str, Any]:
    request, manifest, units = prepare_investment_raw_acquisition(root, request_path)
    output = _inside(root, output)
    if output.exists():
        if expected_pin is None:
            raise DataReadinessError("investment acquisition existing plan needs independent authority pin")
    else:
        if expected_pin is not None:
            raise DataReadinessError("investment acquisition cannot replace a missing pinned plan")
        if any(resolve_inside_authority(root, name).is_relative_to(output) for name in request["source_files"]):
            raise DataReadinessError("investment acquisition output overlaps pinned source")
        publish_daily_history_plan(output=output, request=request, manifest=manifest, units=units)
        expected_pin = file_sha256(output / "_authority.json")
    verified = archive._load_verified_plan(output, expected_plan_authority_sha256=expected_pin)
    if verified.scope != SCOPE:
        raise DataReadinessError("investment acquisition replay scope differs")
    return {**_object(output / "_manifest.json"), "plan_authority_sha256": expected_pin}


def publish_investment_raw_acquisition_plan(root: Path, request_path: Path, output_directory: Path, *,
                                          expected_plan_authority_sha256: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    with heavy_job_lease("investment-raw-acquisition-plan", runtime_dir=_inside(root, heavy_job_runtime_dir())):
        return _plan(root, request_path, output_directory, expected_plan_authority_sha256)


def collect_investment_raw_acquisition(root: Path, request_path: Path, plan_directory: Path,
                                     output_directory: Path, *, source_factory: archive.SourceFactory,
                                     expected_plan_authority_sha256: str, maximum_units_this_run: int | None = None
                                     ) -> dict[str, Any]:
    root = root.resolve()
    with heavy_job_lease("investment-raw-acquisition", runtime_dir=_inside(root, heavy_job_runtime_dir())):
        plan, output = _inside(root, plan_directory), _inside(root, output_directory)
        _plan(root, request_path, plan, expected_plan_authority_sha256)
        request = _object(plan / "_request.json")
        if output.is_relative_to(plan) or plan.is_relative_to(output) or any(
            resolve_inside_authority(root, name).is_relative_to(output) for name in request["source_files"]):
            raise DataReadinessError("investment raw archive overlaps an immutable input")
        if not (output / "_authority.json").exists():
            archive.collect_swing_history_plan(plan_directory=plan, output_directory=output, source_factory=source_factory,
                provider_symbol_for=lambda ticker: str(request["provider_symbols"][ticker]),
                expected_plan_authority_sha256=expected_plan_authority_sha256, maximum_units_this_run=maximum_units_this_run)
        return _verify(root, request_path, plan, output, expected_plan_authority_sha256)


def _verify(root: Path, request_path: Path, plan: Path, output: Path, pin: str) -> dict[str, Any]:
    _plan(root, request_path, plan, pin)
    verified = archive._load_verified_plan(plan, expected_plan_authority_sha256=pin)
    plan_request, plan_manifest = _object(plan / "_request.json"), _object(plan / "_manifest.json")
    collection_request = _object(output / "_request.json")
    request_hash = collection_request.get("request_sha256")
    units = archive._bind_provider_symbols(verified.units, lambda ticker: str(plan_request["provider_symbols"][ticker]))
    expected = {"schema": archive.COLLECTION_SCHEMA, "plan_directory": str(plan.resolve()), "plan_schema": PLAN_SCHEMA,
        "plan_hashes": verified.hashes, "universe_sha256": verified.universe_sha256,
        "provider_unit_set_sha256": archive._provider_unit_set_sha256(units), "provider_symbols": plan_request["provider_symbols"],
        "provider": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": "raw", "transport_receipts_required": True,
        "workers": 1, "maximum_memory_gib": archive.MAXIMUM_MEMORY_GIB}
    if (collection_request != {**expected, "request_sha256": json_sha256(expected)} or request_hash != json_sha256(expected)):
        raise DataReadinessError("investment raw archive request differs from exact acquisition plan")
    complete = (output / "_authority.json").exists()
    if complete:
        archive.load_complete_swing_history_collection(output, plan_directory=plan, expected_adjustment="raw",
            expected_plan_authority_sha256=pin)
    elif (output / "_manifest.json").exists():
        raise DataReadinessError("investment raw archive has an orphan final manifest")
    files = dict(plan_request["source_files"])
    for path in (*plan.iterdir(), output / "_request.json", output / "_status.json", output / "_manifest.json", output / "_authority.json"):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_sha256(path)
    completed: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    captured = requested = 0
    actual_ends: list[str] = []
    for unit in units.to_dict(orient="records"):
        _guard()
        days = {day.isoformat() for day in holding_calendar(date.fromisoformat(unit["start_date"]), date.fromisoformat(unit["end_date"]))}
        requested += len(days)
        record = archive._load_existing_unit(output, unit, request_sha256=str(request_hash), adjustment="raw", require_transport=True)
        observed: set[str] = set()
        if record is not None:
            path = resolve_inside_authority(output, record["bars_path"])
            frame = pd.read_parquet(path, columns=["session_date"])
            if file_sha256(path) != record["bars_sha256"]:
                raise DataReadinessError("investment raw observations changed during coverage read")
            observed = set(frame.session_date.astype(str))
            if len(frame) != len(observed) or not observed.issubset(days):
                raise DataReadinessError("investment raw observations duplicate or escape exact XNYS sessions")
            captured += len(observed)
            if observed:
                actual_ends.append(max(observed))
            for key in ("bars_path", "unit_manifest_path"):
                path = resolve_inside_authority(output, record[key])
                files[path.relative_to(root).as_posix()] = file_sha256(path)
            unit_manifest = _object(resolve_inside_authority(output, record["unit_manifest_path"]))
            for page in unit_manifest["pages"]:
                for name in (page["raw_path"], page["transport"]["body_path"]):
                    path = resolve_inside_authority(output, name)
                    files[path.relative_to(root).as_posix()] = file_sha256(path)
            completed.append({**record, "archive": output.relative_to(root).as_posix(),
                "identity_admission": QUARANTINE if unit["role"] == "stock" else "captured_only",
                "ownership_admitted": False, "actual_first_session": min(observed) if observed else None,
                "actual_last_session": max(observed) if observed else None, "required_sessions": len(days),
                "missing_sessions": sorted(days - observed), "candidate_requirements": plan_manifest["unit_requirements"][unit["unit_id"]]})
        if days - observed:
            gaps.append({"unit_id": unit["unit_id"], "security_id": unit["security_id"], "role": unit["role"],
                "reason": "unit_not_completed" if record is None else "provider_observations_missing",
                "missing_sessions": sorted(days - observed)})
    _recheck(root, files)
    collection_status = _object(output / "_status.json") if (output / "_status.json").is_file() else None
    return {"schema": "market_predictor.investment_raw_acquisition_report", "scope": SCOPE,
        "archive_status": _object(output / "_manifest.json")["status"] if complete else "incomplete",
        "archive_authority_present": complete, "collection_status": collection_status,
        "status_receipt_present": collection_status is not None,
        "plan_authority_sha256": pin, "plan_directory": plan.relative_to(root).as_posix(),
        "archive_directory": output.relative_to(root).as_posix(), "outcome_end": OUTCOME_END.isoformat(),
        "plan_request_sha256": file_sha256(plan / "_request.json"), "plan_manifest_sha256": file_sha256(plan / "_manifest.json"),
        "archive_request_sha256": file_sha256(output / "_request.json"),
        "archive_authority_sha256": file_sha256(output / "_authority.json") if complete else None,
        "provider_symbols": plan_request["provider_symbols"],
        "planned_benchmark_symbols": sorted(set(units.loc[units.role.eq("benchmark"), "ticker"].astype(str))),
        "actual_source_end": max(actual_ends) if actual_ends else None, "requested_units": len(units),
        "completed_units": len(completed), "requested_sessions": requested, "captured_sessions": captured,
        "missing_sessions": requested - captured, "completed_artifacts": completed, "gaps": gaps,
        "source_files": files, "parent_pins": plan_request["policy"],
        "identity_admission": {"stock": QUARANTINE, "benchmark": "captured_only"},
        "ownership_admitted": False, "accounting_eligible": False, "label_eligible": False, "promotion_eligible": False}


def verify_investment_raw_acquisition(root: Path, request_path: Path, plan_directory: Path, output_directory: Path, *,
                                    expected_plan_authority_sha256: str) -> dict[str, Any]:
    root = root.resolve()
    with heavy_job_lease("investment-raw-acquisition-verification", runtime_dir=_inside(root, heavy_job_runtime_dir())):
        return _verify(root, request_path, _inside(root, plan_directory), _inside(root, output_directory),
            expected_plan_authority_sha256)
