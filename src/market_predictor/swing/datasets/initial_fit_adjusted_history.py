"""Independent adjusted source queries projected from accepted initial-fit identities."""
from __future__ import annotations

import argparse
import copy
import io
import json
import tomllib
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from market_predictor.canonical.store import file_sha256
from market_predictor.config import get_settings
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, resolve_inside_authority
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease, heavy_job_runtime_dir
from market_predictor.locking import file_lock
from market_predictor.resources import assert_memory_budget
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.adjusted_history_source import load_adjusted_history_source, read_adjusted_history_unit
from market_predictor.swing.datasets.feature_history_plan import (
    DECISION_START,
    NUMERIC_END,
    WARMUP_START,
    feature_history_requirements,
)
from market_predictor.swing.datasets.history_archive import (
    AlpacaSwingDailyPageSource,
    SourceFactory,
    collect_swing_history_plan,
)
from market_predictor.swing.datasets.history_plan_publication import (
    AUTHORITY_SCHEMA,
    DAILY_BAR_UNITS_FILE,
    PLAN_SCHEMA,
    UNIT_COLUMNS,
    publish_daily_history_plan,
)
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.labels.holding_paths import holding_calendar

INITIAL_FIT_ADJUSTED_HISTORY_SCOPE = "initial_fit_adjusted_history"
_EXPECTED_COUNTS = (564, 551, 545, 13)
_CORRECTIONS = {"cik:0001415404": ("ECHO", "SATS"), "cik:0000798354": ("FISV", "FI")}
_MAX_METADATA = 8 * 1024**2


class InitialFitAdjustedHistoryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["market_predictor.initial_fit_adjusted_history_policy"]
    parent_plan_authority: SourcePin
    reviewed_feature_history_policy: SourcePin
    warmup_start: date
    decision_start: date
    numeric_end: date
    adjustment: Literal["all"]
    benchmark_start_policy: Literal["inherit_pinned_contiguous_prefix"]
    minimum_benchmark_warmup_sessions: Literal[253]
    purpose: Literal["source_queries_only_preserve_parent_decision_population"]


def _policy(root: Path, config: Path, expected: str) -> InitialFitAdjustedHistoryPolicy:
    path = resolve_inside_authority(root, config)
    if path.stat().st_size > 65536 or file_sha256(path) != expected:
        raise DataReadinessError("adjusted history policy requires its independent pin")
    policy = InitialFitAdjustedHistoryPolicy.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    if (policy.warmup_start, policy.decision_start, policy.numeric_end) != (WARMUP_START, DECISION_START, NUMERIC_END):
        raise DataReadinessError("adjusted history warmup or development boundaries differ")
    return policy


def _recheck(root: Path, sources: dict[str, str]) -> None:
    for name, digest in sources.items():
        if file_sha256(resolve_inside_authority(root, name)) != digest:
            raise DataReadinessError(f"adjusted history source changed: {name}")


def _benchmark_starts(coverage: object, tickers: set[str]) -> dict[str, str]:
    """Admit only the parent's exact-window or contiguous pre-inception evidence."""
    fields = {"ticker", "action", "coverage_policy", "expected_session_count", "first_missing_session",
        "first_observed_session", "missing_session_count", "observed_session_count",
        "pre_inception_missing_session_count", "requested_first_session", "requested_last_session"}
    if not isinstance(coverage, list) or len(coverage) != len(tickers):
        raise DataReadinessError("adjusted benchmark coverage inventory differs")
    starts: dict[str, str] = {}
    for row in coverage:
        if not isinstance(row, dict) or set(row) != fields:
            raise DataReadinessError("adjusted benchmark coverage fields differ")
        ticker = row["ticker"]
        if not isinstance(ticker, str) or ticker not in tickers or ticker in starts:
            raise DataReadinessError("adjusted benchmark coverage ticker differs")
        try:
            first = date.fromisoformat(row["first_observed_session"])
            requested_first = date.fromisoformat(row["requested_first_session"])
            last = date.fromisoformat(row["requested_last_session"])
        except (ValueError, TypeError) as exc:
            raise DataReadinessError("adjusted benchmark coverage dates differ") from exc
        if (requested_first != WARMUP_START or last < NUMERIC_END or last > date(2026, 7, 8)
                or not WARMUP_START <= first < DECISION_START or row["action"] != "retain"):
            raise DataReadinessError("adjusted benchmark coverage bounds differ")
        sessions = tuple(holding_calendar(requested_first, last))
        if first not in sessions or sessions[0] != requested_first or sessions[-1] != last:
            raise DataReadinessError("adjusted benchmark coverage bounds are not sessions")
        prefix = sessions.index(first)
        counts = {"expected_session_count": len(sessions), "missing_session_count": prefix,
            "pre_inception_missing_session_count": prefix, "observed_session_count": len(sessions) - prefix}
        if (any(type(row[key]) is not int or row[key] != value for key, value in counts.items())
                or row["first_missing_session"] != (str(requested_first) if prefix else None)
                or row["coverage_policy"] not in {"exact_full_window", "contiguous_pre_inception_prefix_allowed"}
                or row["coverage_policy"] == "exact_full_window" and prefix):
            raise DataReadinessError("adjusted benchmark coverage is not an accepted contiguous prefix")
        if len(holding_calendar(max(WARMUP_START, first), DECISION_START - timedelta(days=1))) < 253:
            raise DataReadinessError("adjusted benchmark history supplies fewer than 253 pre-decision sessions")
        starts[ticker] = str(max(WARMUP_START, first))
    return starts


def initial_fit_adjusted_history_requirements(root: Path, config: Path, policy_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Read pinned identity CSV/metadata and reviewed documents, never analytical rows."""
    root = root.resolve()
    policy = _policy(root, config, policy_sha256)
    sources: dict[str, str] = {}

    def pin(path: Path, digest: str | None = None) -> Path:
        resolved = resolve_inside_authority(root, path)
        actual = file_sha256(resolved)
        if digest is not None and actual != digest:
            raise DataReadinessError("adjusted history source pin differs")
        sources[resolved.relative_to(root).as_posix()] = actual
        return resolved

    pin(config, policy_sha256)
    parent_authority_path = pin(Path(policy.parent_plan_authority.path), policy.parent_plan_authority.sha256)
    if parent_authority_path.name != "_authority.json":
        raise DataReadinessError("adjusted history parent pin must name its authority")
    parent = parent_authority_path.parent
    authority = pinned_object(parent_authority_path, policy.parent_plan_authority.sha256)
    request_path = pin(parent / "_request.json", authority["request_sha256"])
    manifest_path = pin(parent / "_manifest.json", authority["artifact_sha256"])
    original = pinned_object(request_path, authority["request_sha256"])
    original_manifest = pinned_object(manifest_path, authority["artifact_sha256"])
    membership = original.get("membership_authority")
    daily = original_manifest.get("daily_bars", {})
    artifact = daily.get("units_artifact", {})
    if (authority.get("schema") != AUTHORITY_SCHEMA or authority.get("state") != "complete"
            or authority.get("artifact") != "_manifest.json" or original.get("schema") != PLAN_SCHEMA
            or original_manifest.get("schema") != PLAN_SCHEMA
            or original.get("scope") != "initial_fit_raw_share_acquisition"
            or original_manifest.get("scope") != original["scope"]
            or original_manifest.get("status") != "ready_for_daily_history_collection"
            or original_manifest.get("outcomes_read") is not False
            or original_manifest.get("request_sha256") != authority["request_sha256"]
            or not isinstance(membership, dict) or membership.get("universe_sha256") != authority.get("universe_sha256")
            or original_manifest.get("membership") != {key: membership.get(key) for key in ("universe_sha256", "parent_lineage")}
            or daily.get("adjustment") != "raw" or daily.get("timeframe") != "1Day"
            or daily.get("price_feed") != "sip" or daily.get("source") != "alpaca" or daily.get("status") != "ready"
            or artifact.get("path") != DAILY_BAR_UNITS_FILE or artifact.get("sha256") != authority.get("units_sha256")
            or original.get("decision_start") != str(DECISION_START) or original.get("initial_fit_end") != str(NUMERIC_END)):
        raise DataReadinessError("adjusted history parent identity authority differs")
    units_path = pin(parent / DAILY_BAR_UNITS_FILE, authority["units_sha256"])
    if units_path.stat().st_size > _MAX_METADATA or units_path.stat().st_size != artifact.get("bytes"):
        raise DataReadinessError("adjusted history parent identity CSV size differs")
    parent_units = pd.read_csv(io.BytesIO(units_path.read_bytes()), dtype=str, keep_default_na=False)
    if list(parent_units.columns) != list(UNIT_COLUMNS) or parent_units.empty or parent_units.duplicated().any():
        raise DataReadinessError("adjusted history parent windows are malformed")
    stocks = parent_units.loc[parent_units.role.eq("stock")]
    benchmarks = parent_units.loc[parent_units.role.eq("benchmark")]
    counts = (len(parent_units), len(stocks), stocks.security_id.nunique(), len(benchmarks))
    retained = original.get("retained_security_ids")
    excluded = original.get("excluded_security_ids")
    if (counts != _EXPECTED_COUNTS or len(stocks) + len(benchmarks) != len(parent_units)
            or not isinstance(retained, list) or not all(isinstance(value, str) and value for value in retained)
            or len(set(retained)) != len(retained) or not set(stocks.security_id).issubset(retained)
            or not isinstance(excluded, list) or set(retained).intersection(excluded)
            or daily.get("planned_units") != counts[0] or daily.get("stock_units") != counts[1]
            or daily.get("benchmark_units") != counts[3]
            or not {"SPY", "QQQ"}.issubset(set(benchmarks.ticker))
            or original.get("asof_policy") != "inclusive_unit_end_date_entity_mapping_not_ownership"):
        raise DataReadinessError("adjusted history parent population differs")
    mapping = original.get("provider_symbols")
    if not isinstance(mapping, dict) or set(mapping) != set(parent_units.ticker):
        raise DataReadinessError("adjusted history parent provider mapping differs")
    for row in parent_units.itertuples(index=False):
        first, last = date.fromisoformat(row.start_date), date.fromisoformat(row.end_date)
        if (not DECISION_START <= first <= last <= NUMERIC_END or not row.security_id or not row.ticker
                or not isinstance(mapping[row.ticker], str) or not mapping[row.ticker].strip()
                or row.role == "benchmark" and (first, last) != (DECISION_START, NUMERIC_END)):
            raise DataReadinessError("adjusted history parent window differs")

    metadata_name = original.get("policy", {}).get("parent_request_path")
    inherited_pins = original.get("source_files", {})
    if not isinstance(metadata_name, str) or not isinstance(inherited_pins, dict):
        raise DataReadinessError("adjusted benchmark coverage lacks the accepted parent metadata pin")
    metadata_path = inside(root, metadata_name)
    metadata_pin = inherited_pins.get(metadata_path.relative_to(root).as_posix())
    if not isinstance(metadata_pin, str) or len(metadata_pin) != 64:
        raise DataReadinessError("adjusted benchmark coverage lacks the accepted parent metadata pin")
    pin(metadata_path, metadata_pin)
    parent_metadata = pinned_object(metadata_path, metadata_pin)
    coverage = parent_metadata.get("combined_daily_inputs", {}).get("benchmark_coverage")
    benchmark_starts = _benchmark_starts(coverage, set(benchmarks.ticker))

    feature_pin = policy.reviewed_feature_history_policy
    pin(Path(feature_pin.path), feature_pin.sha256)
    reviewed, _, corrected_units = feature_history_requirements(root, Path(feature_pin.path), feature_pin.sha256)
    correction = reviewed["reviewed_correction_policy"]
    if (inside(root, correction["parent_plan"]) != parent
            or correction["parent_plan_sha256"] != policy.parent_plan_authority.sha256
            or reviewed["membership_authority"] != membership
            or set(corrected_units.security_id) != set(_CORRECTIONS)):
        raise DataReadinessError("adjusted history reviewed corrections use another parent")
    pin(Path(reviewed["policy"]["correction_policy"]), reviewed["policy"]["correction_policy_sha256"])
    pin(Path(correction["document_inventory"]))
    document_root = inside(root, correction["document_archive"])
    document_files = sorted(path for path in document_root.rglob("*") if path.is_file())
    if not document_files or len(document_files) > 1000:
        raise DataReadinessError("adjusted history reviewed document inventory exceeds bound")
    for path in document_files:
        pin(path)
    units = parent_units.copy()
    window_map = []
    provider_symbols: dict[str, str] = {}
    for index, row in parent_units.iterrows():
        identity, ticker = str(row.security_id), str(row.ticker)
        query_ticker = ticker
        provider = mapping[ticker]
        if identity in _CORRECTIONS:
            original_ticker, query_ticker = _CORRECTIONS[identity]
            corrected = corrected_units.loc[corrected_units.security_id.eq(identity)]
            correction_row = next(item for item in correction["corrections"] if item["security_id"] == identity)
            if (ticker != original_ticker or len(stocks.loc[stocks.security_id.eq(identity)]) != 1
                    or len(corrected) != 1 or corrected.iloc[0].ticker != query_ticker
                    or row.start_date != str(DECISION_START) or row.end_date != str(NUMERIC_END)
                    or correction_row["parent_unit_id"] != "swing-daily-" + json_sha256(row.to_dict())[:24]):
                raise DataReadinessError("adjusted history corrected parent window differs")
            provider = reviewed["provider_symbols"][query_ticker]
        if query_ticker in provider_symbols and provider_symbols[query_ticker] != provider:
            raise DataReadinessError("adjusted history query symbol has conflicting providers")
        provider_symbols[query_ticker] = provider
        units.loc[index, "ticker"] = query_ticker
        units.loc[index, "start_date"] = benchmark_starts[ticker] if row.role == "benchmark" else str(WARMUP_START)
        window_map.append({"parent": row.to_dict(), "query": units.loc[index].to_dict()})
    if units.duplicated().any():
        raise DataReadinessError("adjusted history expansion duplicates an acquisition unit")
    sessions = tuple(day.isoformat() for day in holding_calendar(WARMUP_START, NUMERIC_END))
    package = Path(__file__).resolve().parents[2]
    request = {"schema": PLAN_SCHEMA, "scope": INITIAL_FIT_ADJUSTED_HISTORY_SCOPE,
        "source_root": str(root), "policy_path": inside(root, config).relative_to(root).as_posix(),
        "policy_sha256": policy_sha256, "policy": policy.model_dump(mode="json"), "source_files": sources,
        "protected_directories": [path.relative_to(root).as_posix() for path in (
            parent, document_root, inside(root, correction["parent_archive"]))],
        "membership_authority": membership, "retained_security_ids": retained, "excluded_security_ids": excluded,
        "parent_window_mapping": window_map, "provider_symbols": provider_symbols,
        "benchmark_start_sessions": benchmark_starts, "accepted_benchmark_coverage": coverage,
        "benchmark_coverage_source": {"path": metadata_path.relative_to(root).as_posix(), "sha256": metadata_pin},
        "reviewed_feature_history_request_sha256": json_sha256(reviewed),
        "asof_policy": "inclusive_unit_end_date_entity_mapping_not_ownership",
        "query_symbol_policy": "asof_entity_stream_label_not_historical_exchange_ticker",
        "decision_start": str(DECISION_START), "warmup_start": str(WARMUP_START), "numeric_end": str(NUMERIC_END),
        "parent_requirements_sha256": original.get("requirements_sha256"),
        "parent_unit_requirements_sha256": original.get("unit_requirements_sha256"),
        "implementation_files": {name: file_sha256(package / name) for name in (
            "swing/datasets/initial_fit_adjusted_history.py", "swing/datasets/history_plan_publication.py",
            "swing/datasets/feature_history_plan.py", "swing/labels/holding_paths.py")}}
    manifest = {"schema": PLAN_SCHEMA, "scope": INITIAL_FIT_ADJUSTED_HISTORY_SCOPE,
        "status": "ready_for_daily_history_collection", "outcomes_read": False,
        "ownership_admitted": False, "bar_coverage_verified": False, "feature_eligible": False,
        "label_eligible": False, "accounting_eligible": False, "promotion_eligible": False,
        "historical_availability_proven": False,
        "membership": {key: membership[key] for key in ("universe_sha256", "parent_lineage")},
        "missing_session_ranges": [{"first_session": str(WARMUP_START), "last_session": str(NUMERIC_END), "sessions": len(sessions)}],
        "daily_bars": {"status": "ready", "source": "alpaca", "timeframe": "1Day", "price_feed": "sip",
            "adjustment": "all", "planned_units": counts[0], "stock_units": counts[1], "benchmark_units": counts[3]},
        "in_window_stock_securities": counts[2], "required_sessions_sha256": json_sha256(sessions),
        "warmup_policy": "query_expansion_only_never_membership_or_observed_coverage"}
    _recheck(root, sources)
    return request, manifest, units


def validate_initial_fit_adjusted_history_collection_plan(*, directory: Path, request: dict[str, Any],
    manifest: dict[str, Any], units: pd.DataFrame,
) -> None:
    """Reject rehashed local changes by reconstructing independently pinned requirements."""
    expected_request, expected_manifest, expected_units = initial_fit_adjusted_history_requirements(
        Path(request["source_root"]), Path(request["policy_path"]), request["policy_sha256"])
    actual = copy.deepcopy(manifest)
    for field in ("resources", "request_sha256"):
        actual.pop(field, None)
    actual["daily_bars"].pop("units_artifact", None)
    if (request != expected_request or actual != expected_manifest or not units.loc[:, list(UNIT_COLUMNS)]
            .sort_values(list(UNIT_COLUMNS)).reset_index(drop=True)
            .equals(expected_units.sort_values(list(UNIT_COLUMNS)).reset_index(drop=True))):
        raise DataReadinessError("adjusted history plan differs from independently pinned requirements")


def _protect(root: Path, destination: Path, request: dict[str, Any], *additional: Path) -> None:
    sources = [inside(root, name) for name in request["source_files"]]
    sources.extend(inside(root, name) for name in request["protected_directories"])
    sources.extend(additional)
    if any(destination.is_relative_to(path) or path.is_relative_to(destination) for path in sources):
        raise DataReadinessError("adjusted history output overlaps an immutable input")


def publish_initial_fit_adjusted_history_plan(root: Path, config: Path, policy_sha256: str,
    output: Path,
) -> dict[str, Any]:
    root = root.resolve()
    request, manifest, units = initial_fit_adjusted_history_requirements(root, config, policy_sha256)
    destination = inside(root, output)
    _protect(root, destination, request)
    staging = destination.with_name(f".{destination.name}.planning")
    _protect(root, staging, request)
    with file_lock(destination, timeout=0.0):
        if destination.exists():
            raise DataReadinessError("adjusted history plan output must be new")
        _recheck(root, request["source_files"])
        result = publish_daily_history_plan(output=staging, request=request, manifest=manifest, units=units)
        validate_initial_fit_adjusted_history_collection_plan(directory=staging, request=request,
            manifest=result, units=units)
        _recheck(root, request["source_files"])
        if destination.exists():
            raise DataReadinessError("adjusted history plan destination appeared during publication")
        staging.rename(destination)
        return result


def _verify_normalized_archive(root: Path, plan: Path, plan_pin: str, directory: Path,
    archive_pin: str, request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    context = load_adjusted_history_source(root=root,
        plan_authority=SourcePin(path=(plan / "_authority.json").relative_to(root).as_posix(), sha256=plan_pin),
        archive_authority=SourcePin(path=(directory / "_authority.json").relative_to(root).as_posix(), sha256=archive_pin))
    diagnostics = {"stock_missing_sessions": 0, "stock_invalid_sessions": 0}
    for unit_id in context.records:
        _guard()
        unit = read_adjusted_history_unit(context, unit_id)
        if unit.record["role"] == "stock":
            diagnostics["stock_missing_sessions"] += len(unit.missing_sessions)
            diagnostics["stock_invalid_sessions"] += len(unit.invalid_sessions)
        del unit
    _recheck(root, dict(context.source_files))
    _recheck(root, request["source_files"])
    pinned_object(plan / "_authority.json", plan_pin)
    authority = pinned_object(directory / "_authority.json", archive_pin)
    return pinned_object(directory / "_manifest.json", authority["artifact_sha256"]), diagnostics


def _guard() -> None:
    assert_memory_budget(stage="initial-fit adjusted history", hard_budget_gib=4.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0)


def run_initial_fit_adjusted_history(*, mode: Literal["publish-plan", "collect", "offline"], root: Path,
    config: Path, policy_sha256: str, plan_directory: Path, output_directory: Path | None = None,
    expected_plan_sha256: str | None = None, expected_archive_sha256: str | None = None,
    maximum_units_this_run: int = 1, source_factory: SourceFactory | None = None,
) -> dict[str, Any]:
    """Serialize publication/acquisition/replay; default acquisition is one resumable unit."""
    root = root.resolve()
    if not 1 <= maximum_units_this_run <= _EXPECTED_COUNTS[0]:
        raise DataReadinessError("adjusted history maximum units is outside the bounded population")
    if mode == "publish-plan":
        if output_directory is not None or expected_plan_sha256 is not None or expected_archive_sha256 is not None:
            raise DataReadinessError("plan publication cannot collect or reuse existing authorities")
    elif output_directory is None or expected_plan_sha256 is None or mode == "offline" and expected_archive_sha256 is None:
        raise DataReadinessError("collection/replay requires independent plan and offline archive pins")
    sources: list[AlpacaSource] = []
    with heavy_job_lease("initial-fit-adjusted-history", runtime_dir=inside(root, heavy_job_runtime_dir())):
        try:
            _guard()
            plan = inside(root, plan_directory)
            if mode == "publish-plan":
                publish_initial_fit_adjusted_history_plan(root, config, policy_sha256, plan)
                return {"status": "plan_published", "plan_sha256": file_sha256(plan / "_authority.json")}
            request, _, _ = initial_fit_adjusted_history_requirements(root, config, policy_sha256)
            assert expected_plan_sha256 is not None and output_directory is not None
            authority = pinned_object(plan / "_authority.json", expected_plan_sha256)
            if pinned_object(plan / "_request.json", authority["request_sha256"]) != request:
                raise DataReadinessError("adjusted history plan differs from requested configuration")
            output = inside(root, output_directory)
            _protect(root, output, request, plan)
            staging = output.with_name(f".{output.name}.collecting")
            _protect(root, staging, request, plan)
            diagnostics: dict[str, int] = {}
            if output.exists() or mode == "offline":
                if expected_archive_sha256 is None:
                    raise DataReadinessError("completed adjusted archive requires its independent authority pin")
                result, diagnostics = _verify_normalized_archive(root, plan, expected_plan_sha256,
                    output, expected_archive_sha256, request)
            else:
                if expected_archive_sha256 is not None:
                    raise DataReadinessError("archive authority pin cannot authorize a replacement")
                staged_authority = staging / "_authority.json"
                if source_factory is None and not staged_authority.exists():
                    settings = get_settings()
                    if not settings.has_alpaca or settings.alpaca_stock_feed.strip().lower() != "sip":
                        raise DataReadinessError("configured Alpaca credentials and SIP are required")

                    def factory() -> AlpacaSwingDailyPageSource:
                        source = AlpacaSource(settings)
                        source.client.before_request = _guard
                        sources.append(source)
                        return AlpacaSwingDailyPageSource(source)

                    source_factory = factory
                if not staged_authority.exists():
                    assert source_factory is not None
                    result = collect_swing_history_plan(plan_directory=plan, output_directory=staging,
                        source_factory=source_factory, provider_symbol_for=lambda ticker: str(request["provider_symbols"][ticker]),
                        maximum_units_this_run=maximum_units_this_run, expected_plan_authority_sha256=expected_plan_sha256)
                if staged_authority.exists():
                    archive_pin = file_sha256(staged_authority)
                    result, diagnostics = _verify_normalized_archive(root, plan, expected_plan_sha256,
                        staging, archive_pin, request)
                    with file_lock(output, timeout=0.0):
                        if output.exists():
                            raise DataReadinessError("adjusted history archive destination appeared during publication")
                        staging.rename(output)
            _recheck(root, request["source_files"])
            pinned_object(plan / "_authority.json", expected_plan_sha256)
            compact = {key: result[key] for key in ("status", "requested_units", "observed_units")}
            compact.update({key: len(result[key]) for key in ("failed_units", "unattempted_units", "unavailable_units")})
            if (output / "_authority.json").is_file():
                compact["archive_sha256"] = file_sha256(output / "_authority.json")
            compact.update(diagnostics)
            return compact
        finally:
            for source in sources:
                source.client.session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pinned initial-fit adjusted Alpaca source queries")
    parser.add_argument("mode", choices=("publish-plan", "collect", "offline"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--max-units", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = run_initial_fit_adjusted_history(mode=args.mode, root=args.root, config=args.config,
            policy_sha256=args.expected_policy_sha256, plan_directory=args.plan_dir, output_directory=args.out_dir,
            expected_plan_sha256=args.expected_plan_sha256, expected_archive_sha256=args.expected_archive_sha256,
            maximum_units_this_run=args.max_units)
    except (HeavyJobBusyError, MemoryBudgetError):
        print(json.dumps({"status": "resource_busy", "source_only": True}))
        return 75
    except (DataReadinessError, ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid_or_incomplete", "error": str(exc), "source_only": True}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"plan_published", "complete", "complete_with_unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
