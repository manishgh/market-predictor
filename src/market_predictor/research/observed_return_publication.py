"""Offline observed 120/124/209-column composition through the training kernels."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.contracts import CanonicalBar
from market_predictor.canonical.cutoffs import SWING_NIGHTLY_CUTOFF
from market_predictor.canonical.joins import decisions_from_completed_bars
from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for, write_canonical_artifact
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.path_integrity import verify_no_reparse_ancestry
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.research.news_decision_publication import build_news_month
from market_predictor.research.news_source_links import open_news_source_link_index
from market_predictor.serving.swing_features import _effective_members, _expected_swing_decision_time
from market_predictor.swing.contracts.holding_materialization import SourcePin as HoldingSourcePin
from market_predictor.swing.contracts.research_cohort import load_swing_research_cohort
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.contracts.saved_evaluation_configuration import SavedEvaluationConfigurationEvidence
from market_predictor.swing.datasets.saved_evaluation_configuration import verify_saved_evaluation_configuration
from market_predictor.swing.features.adjusted_source import build_adjusted_technical_source
from market_predictor.swing.features.observed_return_publication import (
    CLOSED,
    PROFILES,
    SCHEMA,
    ObservedReturnPublicationRequest,
    ObservedSourcePin,
    bound_canonical_input,
    checked_file,
    checked_object,
    request_source_files,
)
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.features.research_join import rebuild_research_peer_features
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from market_predictor.swing.features.return_relationships import build_return_relationship_profile
from market_predictor.swing.training.return_artifacts import verify_completed_return_run
from market_predictor.universe.sp500.observed_membership_authority import load_observed_sp500_membership_authority


def _implementations(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    paths: set[Path] = set()
    for name, module in list(sys.modules.items()):
        if name != "market_predictor" and not name.startswith("market_predictor."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None or Path(filename).suffix != ".py":
            continue
        path = Path(filename).resolve(strict=True)
        if not path.is_relative_to(package) or not path.is_relative_to(root.resolve() / "src"):
            raise DataReadinessError("executed predictor module is outside the registered source package")
        paths.add(path)
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(paths)}


def _inventory(root: Path, directory: Path, *, maximum_bytes: int) -> dict[str, str]:
    result: dict[str, str] = {}
    total = 0
    for path in directory.rglob("*"):
        if path.is_file():
            verify_no_reparse_ancestry(path, label="observed membership input")
            relative = path.resolve().relative_to(root).as_posix()
            total += path.stat().st_size
            if total > maximum_bytes:
                raise DataReadinessError("membership authority inventory exceeds source budget")
            result[relative] = file_sha256(path)
    return result


def _observed_bars(frame: pd.DataFrame, *, adjustment: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    required = set(CanonicalBar.model_fields)
    if (not required.issubset(frame) or frame.empty or not frame.columns.is_unique
            or not frame.availability_policy.eq("observed").all()
            or not frame.source.eq("alpaca").all() or not frame.adjustment.eq(adjustment).all()
            or not frame.price_feed.eq("sip").all() or not frame.timeframe.eq("1d").all()):
        raise DataReadinessError("observed return prices require physical Alpaca SIP daily observed inputs")
    for row in frame.loc[:, sorted(required)].to_dict("records"):
        bar = CanonicalBar.model_validate(row)
        if bar.available_at_utc < bar.ingested_at_utc or pd.Timestamp(bar.available_at_utc) > cutoff:
            raise DataReadinessError("observed price availability must include ingestion and precede decision")
    result = frame.copy()
    result["session_date_et"] = pd.to_datetime(result.bar_start_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    return result


def _run_contracts(root: Path, request: ObservedReturnPublicationRequest, files: dict[str, str]) -> dict[str, list[str]]:
    if set(request.model_runs) != set(PROFILES):
        raise DataReadinessError("publication requires all three frozen model runs")
    names: dict[str, list[str]] = {}
    profiles = {"technical": "existing_technical", "relationships": "technical_relationships", "news": "technical_relationships_news"}
    for profile, pin in request.model_runs.items():
        run_path = checked_file(root, pin)
        if run_path.name != "_manifest.json":
            raise DataReadinessError("observed model registration requires the canonical run manifest")
        run, body = verify_completed_return_run(run_path.parent, manifest_sha256=pin.sha256)
        request_path = (Path(pin.path).parent / "_request.json").as_posix()
        run_request_pin = ObservedSourcePin(path=request_path, sha256=run["request_sha256"])
        if body["policy"]["feature_profile"] != profiles[profile]:
            raise DataReadinessError("observed publication model source/strategy contract differs")
        if body["source_files"].get(request.strategy_contract.path) != request.strategy_contract.sha256:
            if request.configuration_evidence is None:
                raise DataReadinessError("frozen strategy needs its existing naming-only correspondence evidence")
            evidence = SavedEvaluationConfigurationEvidence.model_validate_json(
                json.dumps(checked_object(root, request.configuration_evidence)))
            proof = verify_saved_evaluation_configuration(root=root,
                evidence_pin=HoldingSourcePin(**request.configuration_evidence.model_dump()),
                run_manifest=HoldingSourcePin(**pin.model_dump()),
                strategy_pin=HoldingSourcePin(**request.strategy_contract.model_dump()),
                temporal_pin=evidence.temporal.current, original_strategy_pin=evidence.strategy.original_logical,
                original_temporal_pin=evidence.temporal.original_logical, pins=files)
            if proof.provenance["all_other_parsed_values_types_and_list_order_equal"] is not True:
                raise DataReadinessError("frozen feature strategy numerical correspondence is unproven")
        if (profile == "technical"
                and body["source_files"].get(request.training_feature_request.path) != request.training_feature_request.sha256):
            raise DataReadinessError("cohort parent is not bound by the frozen technical model")
        names[profile] = body["feature_names"]
        files[request_path] = run_request_pin.sha256
    return names


def compose_observed_return_features(*, root: Path, request: ObservedReturnPublicationRequest,
        maximum_source_bytes: int = 536_870_912, maximum_source_rows: int = 2_000_000,
        maximum_snapshot_rows: int = 2_000) -> tuple[dict[str, ResearchFeaturePartition], dict[str, Any]]:
    """Caller owns the heavy-job lease; no client frames or source declarations enter."""
    guard("observed return source consumption")
    root = root.resolve()
    cutoff = pd.Timestamp(request.as_of_utc)
    decision = _expected_swing_decision_time(cutoff)
    if decision > cutoff or cutoff > pd.Timestamp(datetime.now(UTC)):
        raise DataReadinessError("observed publication requires a completed nightly decision")
    start = pd.Timestamp(request.history_start).date()
    end = decision.tz_convert("America/New_York").date()
    sessions = tuple(xcals.get_calendar("XNYS").sessions_in_range(start, end).date)
    if len(sessions) < 253:
        raise DataReadinessError("observed return history requires at least 253 exchange positions")
    files = request_source_files(request)
    for name, digest in files.items():
        checked_file(root, ObservedSourcePin(path=name, sha256=digest), maximum_bytes=maximum_source_bytes)
    names = _run_contracts(root, request, files)
    contract = load_strategy_contract(checked_file(root, request.strategy_contract))
    cohort = load_swing_research_cohort(checked_file(root, request.cohort), source_root=root)
    parent = checked_object(root, request.training_feature_request)
    if parent.get("cohort_sha256") != cohort.sha256():
        raise DataReadinessError("observed peer restriction differs from the training cohort")
    membership_directory = checked_file(root, request.membership_authority).parent
    member_request = checked_object(root, ObservedSourcePin(path=(membership_directory / "_request.json").relative_to(root).as_posix(),
        sha256=file_sha256(membership_directory / "_request.json")))
    for directory in (membership_directory, *(Path(member_request[key]).resolve() for key in
                     ("base_membership_directory", "closed_archive_directory", "closed_event_directory"))):
        files.update(_inventory(root, directory, maximum_bytes=maximum_source_bytes))
    membership = load_observed_sp500_membership_authority(membership_directory)
    if pd.Timestamp(membership.parent["observed_at_utc"]) > decision:
        raise DataReadinessError("membership was observed after this decision")
    memberships = membership.memberships.loc[membership.memberships.security_id.isin(cohort.retained_security_ids)].copy()
    members = _effective_members(memberships, decision_time=decision, contract=contract)
    if len(members) > maximum_snapshot_rows:
        raise DataReadinessError("observed decision population exceeds snapshot row budget")
    expected = stamp_canonical_decision_ids(pd.DataFrame({"security_id": [item.security_id for item in members],
        "ticker": [item.ticker for item in members], "decision_time_utc": [decision] * len(members),
        "prediction_cutoff_policy_id": SWING_NIGHTLY_CUTOFF.policy_id, "timeframe": "1d",
        "bar_start_utc": xcals.get_calendar("XNYS").session_open(pd.Timestamp(end))}))
    expected["session_date_et"] = end
    inputs = {}
    for role, adjustment in (("stock_history", "all"), ("raw_decision_bars", "raw"), ("benchmark_history", "all")):
        guard("observed return canonical source load")
        frame, pins = bound_canonical_input(root, getattr(request, role), artifact_type="bars",
            maximum_bytes=maximum_source_bytes, maximum_rows=maximum_source_rows)
        inputs[role] = _observed_bars(frame, adjustment=adjustment, cutoff=decision)
        files.update(pins)
        _, source_manifest = load_canonical_artifact(checked_file(root, getattr(request, role)),
            expected_type="bars", allow_research=True, columns=[])
        provenance = source_manifest.get("inputs")
        if not isinstance(provenance, dict) or not provenance:
            raise DataReadinessError("observed price source lacks its original collection provenance")
        for source_name, digest in provenance.items():
            path = inside(root, source_name)
            source_pin = ObservedSourcePin(path=path.relative_to(root).as_posix(), sha256=digest)
            checked_file(root, source_pin, maximum_bytes=maximum_source_bytes)
            files[source_pin.path] = digest
    stocks, raw, benchmarks = (inputs[name] for name in ("stock_history", "raw_decision_bars", "benchmark_history"))
    if "security_id" not in stocks or "security_id" not in raw:
        raise DataReadinessError("observed stock streams require source-bound canonical issuer identities")
    pieces = []
    for item in members:
        guard("observed return issuer construction")
        own = memberships.loc[memberships.security_id.eq(item.security_id)]
        expected_history = tuple(day for day in sessions if any(
            pd.Timestamp(record.effective_from_utc).tz_convert("America/New_York").date() <= day
            and (pd.isna(record.effective_to_utc) or day < pd.Timestamp(record.effective_to_utc).tz_convert("America/New_York").date())
            for record in own.itertuples()))
        stock = stocks.loc[stocks.security_id.eq(item.security_id)]
        if set(stock.ticker) - {item.ticker}:
            raise DataReadinessError("live stock history needs one coherent issuer/provider symbol stream")
        raw_current = raw.loc[raw.security_id.eq(item.security_id) & raw.session_date_et.eq(end)].copy()
        raw_current = stamp_canonical_decision_ids(decisions_from_completed_bars(raw_current, mode="swing-nightly"))
        built = build_adjusted_technical_source(expected.loc[expected.security_id.eq(item.security_id)],
            stock.loc[stock.session_date_et.isin(expected_history)],
            benchmarks.loc[benchmarks.ticker.isin(("SPY", "QQQ", *own.primary_benchmark.dropna()))
                           & benchmarks.session_date_et.isin(expected_history)],
            memberships, raw_current, security_id=item.security_id, expected_history_sessions=expected_history,
            contract=contract, history_start=start, history_end=end, observation_cutoff=decision)
        pieces.append(built.rows)
    base = pd.concat(pieces, ignore_index=True)
    peers, clocks = rebuild_research_peer_features(base, contract=contract,
        retained_security_ids=frozenset(cohort.retained_security_ids), availability_columns=built.availability_columns)
    model_columns = tuple(swing_model_feature_columns(contract=contract, catalyst=False))
    peers["forecast_eligible"] = peers.feature_eligible & peers.daily_bar_count.ge(contract.swing.minimum_warmup_sessions)
    peers["forecast_abstention_reason"] = peers.forecast_eligible.map({True: "", False: "technical_inputs_or_peer_population_unavailable"})
    technical = ResearchFeaturePartition(peers, model_columns, clocks, {})
    relationship_sources = ReturnRelationshipSources(baseline_authority_sha256=json_sha256(files),
        stock_authority_sha256=request.stock_history.sha256, spy_authority_sha256=request.benchmark_history.sha256,
        availability_semantics="observed", availability_policy_id="observed",
        availability_policy_sha256=json_sha256({"availability": "observed"}),
        price_adjustment="all", price_basis_and_vintage_id="registered_observed_sip_all_generation")
    relationships = build_return_relationship_profile(expected_decisions=expected, baseline=technical,
        stock_bars=stocks.loc[stocks.security_id.isin(expected.security_id) & stocks.session_date_et.isin(sessions)],
        spy_bars=benchmarks.loc[benchmarks.ticker.eq("SPY") & benchmarks.session_date_et.isin(sessions)],
        history_sessions=sessions, contract=contract, sources=relationship_sources, purpose="live_construction")
    with open_news_source_link_index(root=root, index=SourcePin(**request.news_source_index.model_dump())) as reader:
        if reader.availability_semantics != "observed" or reader.decision_end_utc < decision.to_pydatetime():
            raise DataReadinessError("registered news authority does not cover observed current decisions")
        news = build_news_month(parent=relationships, reader=reader, purpose="observed_live")
        files.update(reader.source_files)
        files.update(reader.implementation_files)
    profiles = {"technical": technical, "relationships": relationships, "news": news}
    for profile, part in profiles.items():
        if list(part.model_columns) != names[profile]:
            raise DataReadinessError("constructed observed profile differs from frozen feature order")
    metadata = {"source_files": files, "implementation_files": _implementations(root),
        "decision_ids_sha256": json_sha256(sorted(expected.decision_id)), "security_ids_sha256": json_sha256(sorted(expected.security_id)),
        "peer_population": {"policy": "effective_observed_membership_within_frozen_research_cohort_before_peer_scaling",
            "cohort_sha256": cohort.sha256(), "security_ids": sorted(expected.security_id),
            "contributing_security_ids": sorted(peers.loc[peers.feature_eligible, "security_id"]),
            "sectors": {member.security_id: member.sector for member in members}}}
    return profiles, metadata


def publish_observed_return_features(*, root: Path, request: ObservedReturnPublicationRequest, output_directory: Path,
        maximum_source_bytes: int = 536_870_912, maximum_source_rows: int = 2_000_000,
        maximum_snapshot_rows: int = 2_000) -> dict[str, Any]:
    """Publish a new immutable generation after actual source/kernel traversal."""
    root = root.resolve()
    output = inside(root, output_directory)
    runtime_dir = heavy_job_runtime_dir()
    runtime_dir = runtime_dir if runtime_dir.is_absolute() else root / runtime_dir
    with heavy_job_lease("publish-observed-return-features", runtime_dir=runtime_dir):
        guard("before observed return publication")
        if output.exists():
            raise FileExistsError(output)
        profiles, metadata = compose_observed_return_features(root=root, request=request,
            maximum_source_bytes=maximum_source_bytes, maximum_source_rows=maximum_source_rows,
            maximum_snapshot_rows=maximum_snapshot_rows)
        stage = output.with_name(output.name + ".partial")
        stage.mkdir(parents=True, exist_ok=False)
        records = {}
        for profile, part in profiles.items():
            frame = part.rows.copy()
            frame["feature_available_at_utc"] = frame.loc[:, list(part.availability_columns.values())].max(axis=1)
            frame["availability_semantics"] = "observed"
            frame["feature_profile"] = {"technical": "existing_technical", "relationships": "technical_relationships",
                                        "news": "technical_relationships_news"}[profile]
            path = stage / f"{profile}.parquet"
            audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="observed_canonical_return_composition", status="pass",
                failures=0, rows_checked=len(frame),
                detail="Verified registered authorities and shared frozen feature kernels; research only."),))
            write_canonical_artifact(frame, path, artifact_type="swing_return_live_features", audit=audit,
                inputs=metadata["source_files"], production_ready=False)
            sidecar = manifest_path_for(path)
            child = json.loads(sidecar.read_text(encoding="utf-8"))
            child["artifact_path"] = str(output / path.name)
            sidecar.write_text(json.dumps(child, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            saved, _ = load_canonical_artifact(path, expected_type="swing_return_live_features", allow_research=True)
            pd.testing.assert_frame_equal(saved, frame, check_exact=True)
            records[profile] = {"path": path.name, "sha256": file_sha256(path),
                "manifest_sha256": file_sha256(manifest_path_for(path)), "feature_names": list(part.model_columns)}
        for name, digest in {**metadata["source_files"], **metadata["implementation_files"]}.items():
            relative = Path(name).relative_to(root).as_posix() if Path(name).is_absolute() else name
            checked_file(root, ObservedSourcePin(path=relative, sha256=digest))
        manifest = {"schema": SCHEMA, "status": "complete_observed_research_inputs", "created_at_utc": datetime.now(UTC).isoformat(),
            "request": request.model_dump(mode="json"), "request_sha256": json_sha256(request.model_dump(mode="json")),
            "profiles": records, **metadata, **CLOSED}
        write_json_object(stage / "_manifest.json", manifest)
        stage.rename(output)
        return {**manifest, "manifest_sha256": file_sha256(output / "_manifest.json")}
