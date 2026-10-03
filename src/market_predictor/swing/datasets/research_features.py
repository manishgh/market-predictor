"""Resumable, source-bound initial-fit technical history; no target calculation."""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow.dataset as pds

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.resources import assert_memory_budget, release_process_memory
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.research_features import ResearchFeaturePolicy
from market_predictor.swing.datasets.adjusted_history_bindings import (
    AdjustedHistoryBindings,
    bind_adjusted_history_decisions,
    expected_bound_history_sessions,
    load_adjusted_history_bindings,
    read_bound_adjusted_history,
    read_bound_benchmarks,
)
from market_predictor.swing.datasets.corrected_outcomes import (
    _projection,
    load_corrected_decision_partition,
    load_corrected_outcome_policy,
    verified_corrected_price_sources,
)
from market_predictor.swing.datasets.feature_history_plan import DECISION_START, NUMERIC_END
from market_predictor.swing.datasets.initial_fit_raw_share_plan import MEMBERSHIP_COLUMNS
from market_predictor.swing.datasets.research_feature_sources import raw_dollar_volume_inputs
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.features.adjusted_source import (
    AdjustedTechnicalSource,
    build_adjusted_technical_source,
)
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from market_predictor.swing.features.research_join import DECISION_KEYS


def _guard() -> None:
    assert_memory_budget(stage="research predictor publication", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0)


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write_json_object(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _pins(root: Path, files: dict[str, str]) -> None:
    for name, digest in files.items():
        if file_sha256(inside(root, name)) != digest:
            raise DataReadinessError(f"research feature source changed: {name}")


def _audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(checks=(CanonicalAuditCheck(name="source_bound_predictor_population",
        status="pass", failures=0, rows_checked=rows,
        detail="Exact initial-fit decisions retained; indicator and raw-volume inputs independently bound; not model admission."),))


def _publish(frame: pd.DataFrame, path: Path, request_pin: str) -> dict[str, Any]:
    if path.exists() or manifest_path_for(path).exists():
        saved, manifest = load_canonical_artifact(path, expected_type="swing_research_technical_inputs", allow_research=True)
        if manifest["inputs"] != {"research_feature_request_sha256": request_pin}:
            raise DataReadinessError("uncheckpointed predictor artifact belongs to another request")
        try:
            pd.testing.assert_frame_equal(saved.reset_index(drop=True), frame.reset_index(drop=True))
        except AssertionError as error:
            raise DataReadinessError("uncheckpointed predictor artifact differs from reconstruction") from error
    else:
        write_canonical_artifact(frame, path, artifact_type="swing_research_technical_inputs", audit=_audit(len(frame)),
            inputs={"research_feature_request_sha256": request_pin}, production_ready=False)
    return {"path": path.name, "sha256": file_sha256(path), "manifest_sha256": file_sha256(manifest_path_for(path)),
        "rows": len(frame), "decision_ids_sha256": json_sha256(sorted(frame.decision_id)),
        "feature_eligible_rows": int(frame.feature_eligible.sum())}


def _verify_part(directory: Path, record: dict[str, Any], request_pin: str) -> None:
    path = inside(directory, record["path"])
    if file_sha256(path) != record["sha256"] or file_sha256(manifest_path_for(path)) != record["manifest_sha256"]:
        raise DataReadinessError("research feature partition changed")
    _, manifest = load_canonical_artifact(path, expected_type="swing_research_technical_inputs", allow_research=True, columns=[])
    if manifest["rows"] != record["rows"] or manifest["inputs"] != {"research_feature_request_sha256": request_pin}:
        raise DataReadinessError("research feature partition lineage differs")


def _check_population(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    columns = (*DECISION_KEYS, "sector", "session_date_et", "primary_benchmark")
    if (not actual.columns.is_unique or not set(columns).issubset(actual)
            or actual.decision_id.duplicated().any() or len(expected) != len(actual)
            or set(actual.decision_id) != set(expected.decision_id)):
        raise DataReadinessError("technical construction changed its frozen decision population")
    left = expected.set_index("decision_id")
    right = actual.set_index("decision_id").reindex(left.index)
    for name in columns[1:]:
        if left[name].isna().any() or right[name].isna().any() or not left[name].eq(right[name]).all():
            raise DataReadinessError(f"technical construction changed frozen {name}")


def _validate_checkpoint_population(checkpoint: dict[str, Any], decisions: pd.DataFrame) -> None:
    groups = {json_sha256([identity, symbol]) for identity, symbol in
        decisions.groupby(["security_id", "source_group"]).groups}
    months = set(decisions.session_date_et.map(lambda day: day.strftime("%Y-%m")))
    if (not set(checkpoint["groups"]).issubset(groups)
            or not set(checkpoint["months"]).issubset(months)
            or not set(checkpoint.get("failed_groups", {})).issubset(groups)):
        raise DataReadinessError("research predictor checkpoint contains a foreign population")


@dataclass(frozen=True)
class PredictorSourceContext:
    root: Path
    policy: ResearchFeaturePolicy
    source_policy: CorrectedOutcomePolicy
    sources: dict[str, Any]
    strategy: StrategyContract
    bindings: AdjustedHistoryBindings
    original_memberships: pd.DataFrame
    historical_memberships: pd.DataFrame
    decisions: pd.DataFrame
    records: list[dict[str, Any]]
    adjusted_source_files: dict[str, str]
    benchmark: pd.DataFrame


def prepare_predictor_sources(*, root: Path, policy: ResearchFeaturePolicy,
    source_policy: CorrectedOutcomePolicy, sources: dict[str, Any],
) -> PredictorSourceContext:
    """Prepare bounded inputs inside the caller's verified corrected-source lease."""
    _guard()
    strategy = load_strategy_contract(inside(root, policy.strategy_contract.path))
    original_memberships = _projection(sources["membership_path"], MEMBERSHIP_COLUMNS)
    bindings = load_adjusted_history_bindings(root=root, plan_authority=policy.adjusted_plan_authority,
        archive_authority=policy.adjusted_archive_authority)
    _pins(root, dict(bindings.source_files))
    records = [record for record in sources["manifest"]["files"]
        if record["first_session"] <= str(NUMERIC_END) and record["last_session"] >= str(DECISION_START)]
    decisions = pd.concat([load_corrected_decision_partition(root, sources, record, source_policy)
        for record in records], ignore_index=True)
    if decisions.decision_id.duplicated().any() or len(decisions) != sources["plan"]["requirements"]["in_window_decisions"]:
        raise DataReadinessError("research features lack the complete frozen decision population")
    decisions = bind_adjusted_history_decisions(decisions, bindings)
    historical_memberships = sources["memberships"].loc[
        sources["memberships"].security_id.isin(decisions.security_id)
        & pd.to_datetime(sources["memberships"].effective_from_utc, utc=True).dt.tz_convert("America/New_York").dt.date.le(NUMERIC_END)]
    needed = {"SPY", "QQQ", *decisions.primary_benchmark, *historical_memberships.primary_benchmark.dropna()}
    benchmark = read_bound_benchmarks(bindings, needed)
    return PredictorSourceContext(root, policy, source_policy, sources, strategy, bindings,
        original_memberships, historical_memberships, decisions, records, dict(bindings.source_files), benchmark)


def build_predictor_group(group: pd.DataFrame, context: PredictorSourceContext) -> AdjustedTechnicalSource:
    """One ordinary issuer construction for publication and numerical replay."""
    if group.empty or len(group[["security_id", "source_group"]].drop_duplicates()) != 1:
        raise DataReadinessError("predictor construction requires one nonempty source group")
    identity, unit_id = str(group.security_id.iloc[0]), str(group.source_group.iloc[0])
    expected_history = expected_bound_history_sessions(context.bindings, unit_id, context.original_memberships)
    bars = read_bound_adjusted_history(context.bindings, unit_id, expected_history).bars
    raw = raw_dollar_volume_inputs(root=context.root, selection=context.sources["selection"], decisions=group,
        policy=context.source_policy, policy_sha256=context.policy.outcome_source_config.sha256)
    sector_symbols = {"SPY", "QQQ", *group.primary_benchmark,
        *context.historical_memberships.loc[context.historical_memberships.security_id.eq(identity), "primary_benchmark"].dropna()}
    built = build_adjusted_technical_source(group, bars, context.benchmark.loc[context.benchmark.ticker.isin(sector_symbols)],
        context.sources["memberships"], raw, security_id=identity, expected_history_sessions=expected_history, contract=context.strategy)
    frame = built.rows
    _check_population(group, frame)
    frame["technical_missing_reasons"] = frame.technical_missing_reasons.map(json.dumps)
    frame["parent_decision_id"] = frame.decision_id.map(group.set_index("decision_id").parent_decision_id)
    return built


def predictor_implementation_files(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    names = ("swing/datasets/research_features.py", "swing/datasets/research_feature_sources.py",
        "swing/features/adjusted_source.py", "swing/features/predictors.py", "swing/dataset.py", "swing/features/pipeline.py",
        "swing/features/technical_relationships.py", "canonical/normalize.py", "canonical/joins.py", "canonical/cutoffs.py",
        "swing/contracts/research_features.py", "swing/features/panel.py", "swing/features/research_join.py",
        "swing/features/eligibility.py", "swing/datasets/corrected_outcomes.py", "swing/datasets/symbol_corrections.py",
        "swing/labels/holding_identity.py", "modeling/strategy_contract.py", "swing/datasets/session_requirements.py",
        "swing/datasets/history_archive.py", "evidence/io.py", "universe/symbol_correction_policy.py",
        "swing/datasets/initial_fit_raw_share_plan.py", "swing/datasets/adjusted_history_bindings.py",
        "swing/datasets/adjusted_history_source.py")
    return {str((package / name).relative_to(root)): file_sha256(package / name) for name in names}


def materialize_research_predictors(*, root: Path, config: Path, expected_config_sha256: str, output: Path,
    expected_checkpoint_sha256: str | None = None, maximum_groups_this_run: int | None = None,
) -> dict[str, Any]:
    """Build bounded issuer histories once, then emit complete monthly populations.

    The shared corrected-source context owns the workspace lease. All held-out
    numerical columns and all old feature/target columns remain unopened. Outputs
    are research inputs, not training-ready or promoted model artifacts.
    """
    root = root.resolve()
    config, output = inside(root, config), inside(root, output)
    if maximum_groups_this_run is not None and maximum_groups_this_run < 1:
        raise ValueError("maximum_groups_this_run must be positive")
    if file_sha256(config) != expected_config_sha256 or config.stat().st_size > 65536:
        raise DataReadinessError("research feature config requires its independent hash")
    policy = ResearchFeaturePolicy.model_validate(tomllib.loads(config.read_text(encoding="utf-8")))
    if not output.is_relative_to(root / "data/features") or output == root / "data/features":
        raise DataReadinessError("research predictors must publish beneath data/features")
    if output.exists() != (expected_checkpoint_sha256 is not None):
        raise DataReadinessError("existing predictor output requires an independent checkpoint pin")
    declared = {pin.path: pin.sha256 for pin in (getattr(policy, name) for name in type(policy).model_fields if name != "schema_version")}
    declared[config.relative_to(root).as_posix()] = expected_config_sha256
    _pins(root, declared)
    for pin in (policy.adjusted_plan_authority, policy.adjusted_archive_authority):
        source_directory = inside(root, pin.path).parent
        if output.is_relative_to(source_directory) or source_directory.is_relative_to(output):
            raise DataReadinessError("research predictor output overlaps its adjusted source")
    source_config = inside(root, policy.outcome_source_config.path)
    source_policy = load_corrected_outcome_policy(root, source_config, policy.outcome_source_config.sha256)
    with verified_corrected_price_sources(root, source_config, policy.outcome_source_config.sha256,
        source_policy) as sources:
        context = prepare_predictor_sources(root=root, policy=policy, source_policy=source_policy, sources=sources)
        decisions, records = context.decisions, context.records
        adjusted_source_files = context.adjusted_source_files
        implementation = predictor_implementation_files(root)
        request = {"schema": "market_predictor.research_predictor_request", "config_sha256": expected_config_sha256,
            "declared_source_files": declared, "source_files": sources["source_files"], "implementation_files": implementation,
            "adjusted_source_files": adjusted_source_files,
            "cohort_sha256": sources["request"]["cohort_sha256"], "decision_start": str(DECISION_START),
            "numeric_end": str(NUMERIC_END), "expected_rows": len(decisions), "historical_first_seen_proven": False,
            "retained_security_ids": sources["request"]["retained_security_ids"],
            "decision_ids_sha256": json_sha256(sorted(decisions.decision_id)),
            "adjusted_feature_columns": [name for name in TECHNICAL_RANKING_FEATURES if name != "dollar_volume_log"],
            "raw_feature_columns": ["dollar_volume_log"], "separate_symbol_streams": "independent_warmup_no_price_splice"}
        if output.exists():
            checkpoint = pinned_object(output / "_checkpoint.json", expected_checkpoint_sha256)
            if pinned_object(output / "_request.json", checkpoint["request_sha256"]) != request:
                raise DataReadinessError("research predictor resume inputs or implementation changed")
        else:
            output.mkdir(parents=True)
            write_json_object(output / "_request.json", request)
            checkpoint = {"schema": "market_predictor.research_predictors", "status": "partial_in_progress",
                "request_sha256": file_sha256(output / "_request.json"), "groups": {}, "months": {}, "failed_groups": {},
                "training_eligible": False, "promotion_eligible": False, "exclusions_added": []}
            _write(output / "_checkpoint.json", checkpoint)
        request_pin = checkpoint["request_sha256"]
        _validate_checkpoint_population(checkpoint, decisions)
        built = 0
        for (identity, symbol), group in decisions.groupby(["security_id", "source_group"], sort=True):
            _guard()
            group_key = json_sha256([identity, symbol])
            if group_key in checkpoint["groups"]:
                part = checkpoint["groups"][group_key]
                _verify_part(output / "groups", part, request_pin)
                if part["decision_ids_sha256"] != json_sha256(sorted(group.decision_id)):
                    raise DataReadinessError("resumed predictor group population differs")
                continue
            if maximum_groups_this_run is not None and built >= maximum_groups_this_run:
                return {**checkpoint, "checkpoint_sha256": file_sha256(output / "_checkpoint.json")}
            try:
                built_source = build_predictor_group(group, context)
                frame = built_source.rows
            except MemoryBudgetError:
                raise
            except (DataReadinessError, KeyError, FileNotFoundError, ValueError) as error:
                checkpoint["failed_groups"][group_key] = {"security_id": str(identity),
                    "symbol": context.bindings.windows[str(symbol)]["query"]["ticker"], "source_group": str(symbol),
                    "error_type": type(error).__name__, "reason": str(error), "rows": len(group)}
                _write(output / "_checkpoint.json", checkpoint)
                built += 1
                release_process_memory()
                continue
            _pins(root, implementation)
            part = _publish(frame, output / "groups" / f"{group_key}.parquet", request_pin)
            checkpoint["groups"][group_key] = {**part, "availability_columns": dict(built_source.availability_columns)}
            checkpoint["failed_groups"].pop(group_key, None)
            _write(output / "_checkpoint.json", checkpoint)
            built += 1
            del frame, built_source
            release_process_memory()
        if checkpoint["failed_groups"]:
            return {**checkpoint, "status": "partial_source_failures", "checkpoint_sha256": file_sha256(output / "_checkpoint.json")}
        _guard()
        arrow: Any = pds
        dataset = arrow.dataset([str(output / "groups" / part["path"]) for part in checkpoint["groups"].values()], format="parquet")
        for record in records:
            _guard()
            month = record["partition_month"]
            expected = decisions.loc[decisions.session_date_et.map(lambda day: day.strftime("%Y-%m")).eq(month)]
            if expected.empty:
                continue
            frame = dataset.to_table(filter=arrow.field("session_date_et").isin(list(expected.session_date_et.unique())),
                use_threads=False).to_pandas()
            _check_population(expected, frame)
            path = output / "months" / f"{month}.parquet"
            if month in checkpoint["months"]:
                _verify_part(output / "months", checkpoint["months"][month], request_pin)
                if checkpoint["months"][month]["decision_ids_sha256"] != json_sha256(sorted(expected.decision_id)):
                    raise DataReadinessError("resumed predictor month population differs")
            else:
                checkpoint["months"][month] = _publish(frame.sort_values("decision_id"), path, request_pin)
                _write(output / "_checkpoint.json", checkpoint)
            del frame
            release_process_memory()
        _pins(root, {**declared, **implementation, **sources["source_files"], **adjusted_source_files})
        if sum(part["rows"] for part in checkpoint["months"].values()) != len(decisions):
            raise DataReadinessError("technical monthly publication lost frozen decisions")
        result = {**checkpoint, "status": "technical_inputs_complete_research_only", "rows": len(decisions),
            "feature_eligible_rows": sum(part["feature_eligible_rows"] for part in checkpoint["months"].values())}
        final_pins = {**declared, **implementation, **sources["source_files"], **adjusted_source_files}
        checkpoint_pin = file_sha256(output / "_checkpoint.json")
    # Source context exit replays its pins before any complete authority is visible.
    # Reacquire the same workspace lease for the final, independently rechecked write.
    with heavy_job_lease("research-predictor-publication", runtime_dir=inside(root, heavy_job_runtime_dir())):
        if pinned_object(output / "_checkpoint.json", checkpoint_pin) != checkpoint:
            raise DataReadinessError("predictor checkpoint changed before final publication")
        if pinned_object(output / "_request.json", request_pin) != request:
            raise DataReadinessError("predictor request changed before final publication")
        for kind in ("groups", "months"):
            for part in checkpoint[kind].values():
                _verify_part(output / kind, part, request_pin)
        _pins(root, final_pins)
        manifest_path = output / "_manifest.json"
        if manifest_path.exists():
            if pinned_object(manifest_path, file_sha256(manifest_path)) != result:
                raise DataReadinessError("completed research predictor manifest changed")
        else:
            _write(manifest_path, result)
        return {**result, "manifest_sha256": file_sha256(output / "_manifest.json"),
            "checkpoint_sha256": checkpoint_pin}
