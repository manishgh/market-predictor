"""Hash-bound initial-fit deterministic control, never a model or source admission."""
from __future__ import annotations

import json
import os
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
from pydantic import BaseModel, ConfigDict

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.edge_rebuild.swing_training import load_swing_training_config
from market_predictor.edge_rebuild.temporal_manifest import build_temporal_schedule, load_temporal_manifest_config
from market_predictor.edge_rebuild.training.swing_types import _guard, _strict_bool
from market_predictor.evidence.hashing import json_sha256, sequence_sha256
from market_predictor.evidence.io import resolve_inside_authority, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.locking import file_lock
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.evaluation.accounting import evaluate_funded_swing_accounting
from market_predictor.swing.evaluation.ledger import swing_valuation_sessions
from market_predictor.swing.features.panel import MANAGED_PATH_NET_RETURN_COLUMNS, MANAGED_PATH_SESSION_ORDINAL_COLUMNS
from market_predictor.swing.selection import select_constrained_swing_portfolio


class _Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["market_predictor.swing_accounting_control.v1"]
    panel_directory: Literal["data/features/edge_rebuild_swing_technical_panel_20190709_20260708_v1"]
    panel_request_sha256: Literal["d0e093ce2192f8511547b5c5466ac90b892fbd44d4939b002242555a8b19a4ed"]
    panel_manifest_sha256: Literal["891bb547cff304661153661b1c18acf57ec824e6c5c7ddd1eed7346a9a13886c"]
    panel_authority_sha256: Literal["68f01408550f186dc397840459477fcc6a57e946e54473e4e542a708421ba2c4"]
    combined_manifest_sha256: Literal["c5780d2f406531ec2f6d98372577c105a60687ea72c57a7b6c739515934a7e34"]
    temporal_policy: Literal["configs/edge_rebuild_temporal_manifest.toml"]
    training_policy: Literal["configs/edge_rebuild_swing_training.toml"]
    strategy_contract: Literal["configs/edge_rebuild_strategy_contract.toml"]
    research_contract: Literal["configs/swing_research.toml"]
    initial_fit_start: Literal["2019-07-09"]
    initial_fit_end: Literal["2024-05-28"]
    score_column: Literal["return_20d_xs_rank"]


_FEATURES = [
    "decision_id", "decision_group_id", "security_id", "sector", "primary_benchmark", "session_date_et",
    "decision_time_utc", "feature_available_at_utc", "feature_eligible", "cross_section_eligible",
    "return_20d_xs_rank",
]
_FIXED = [
    "future_gross_return_10d", "future_net_return_10d", "future_spy_return_10d", "future_qqq_return_10d",
    "future_sector_return_10d", "future_excess_return_10d_vs_spy", "future_excess_return_10d_vs_qqq",
    "future_excess_return_10d_vs_sector",
]
_OUTCOMES = [
    "decision_id", "entry_time_utc", "exit_time_utc", "entry_session_date_et", "exit_session_date_et",
    "horizon_sessions", "label_available_at_utc", "barrier_label_available_at_utc",
    "barrier_exit_session_date_et", "barrier_holding_sessions", "barrier_cost", "barrier_net_return",
    "barrier_gross_return", *_FIXED, *MANAGED_PATH_SESSION_ORDINAL_COLUMNS, *MANAGED_PATH_NET_RETURN_COLUMNS,
]


def _checked_file(root: Path, relative: str, expected: str, bound: dict[str, str]) -> Path:
    path = resolve_inside_authority(root, relative)
    if file_sha256(path) != expected:
        raise DataReadinessError(f"accounting control source hash mismatch: {relative}")
    bound[str(path)] = expected
    return path


def _metadata(root: Path, relative: str, expected: str, bound: dict[str, str]) -> dict[str, Any]:
    path = _checked_file(root, relative, expected, bound)
    if path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("accounting control metadata exceeds bounded size")
    result = parse_strict_json_object(path.read_bytes(), label=relative)
    _checked_file(root, relative, expected, bound)
    return result


def _sources(root: Path, policy: _Policy, strategy: StrategyContract, bound: dict[str, str]) -> tuple[dict[str, Any], ...]:
    panel_root = root / policy.panel_directory
    request = _metadata(panel_root, "_request.json", policy.panel_request_sha256, bound)
    panel = _metadata(panel_root, "final/_manifest.json", policy.panel_manifest_sha256, bound)
    authority = _metadata(panel_root, "final/_authority.json", policy.panel_authority_sha256, bound)
    combined = _metadata(panel_root, "combined_daily/_manifest.json", policy.combined_manifest_sha256, bound)
    combined_authority = _metadata(
        panel_root, "combined_daily/_authority.json", panel["source"]["combined_daily_authority_sha256"], bound,
    )
    if (authority["state"] != "complete" or authority["artifact_sha256"] != policy.panel_manifest_sha256
            or any(item["request_sha256"] != request["request_sha256"] for item in (panel, authority))
            or any(item["strategy_contract_sha256"] != strategy.sha256() for item in (request, panel, authority))
            or combined_authority["state"] != "complete"
            or combined_authority["artifact_sha256"] != policy.combined_manifest_sha256
            or combined_authority["request_sha256"] != combined["request_sha256"]
            or json_sha256({**request["combined_daily_inputs"], "parent_materialization_request_sha256": request["request_sha256"]})
            != combined["request_sha256"]
            or request["combined_daily_inputs"]["price_feed"] != "sip"
            or request["combined_daily_inputs"]["adjustment"] != "all"):
        raise DataReadinessError("accounting control metadata chain differs")
    return panel, combined


def _calendars(root: Path, policy: _Policy) -> tuple[tuple[str, ...], tuple[str, ...]]:
    temporal = load_temporal_manifest_config(root / policy.temporal_policy)
    initial = tuple(day.isoformat() for day in build_temporal_schedule(temporal).folds[0].train_sessions)
    decisions = initial[:-10]
    valuation = swing_valuation_sessions(decisions, 10)
    if (len(initial) != 1231 or initial[0] != policy.initial_fit_start or initial[-1] != policy.initial_fit_end
            or valuation != initial[1:] or valuation[-1] > policy.initial_fit_end):
        raise DataReadinessError("control decisions and complete maturation tail must stay inside initial fit")
    return decisions, valuation


def _read_projection(path: Path, columns: list[str], sessions: tuple[str, ...], ids: list[str] | None = None) -> pd.DataFrame:
    # PyArrow's dataset API is untyped; keep that boundary out of the typed policy/ledger API.
    arrow: Any = pds
    dataset = arrow.dataset(path, format="parquet")
    if "session_date_et" in dataset.schema.names:
        kind = dataset.schema.field("session_date_et").type
        dates: Any = [date.fromisoformat(day) for day in sessions] if pa.types.is_date(kind) else list(sessions)
        if pa.types.is_timestamp(kind):
            dates = [pd.Timestamp(day) for day in sessions]
        predicate = arrow.field("session_date_et").isin(dates)
    else:
        start = pd.Timestamp(sessions[0], tz="America/New_York").tz_convert("UTC")
        end = (pd.Timestamp(sessions[-1], tz="America/New_York") + pd.Timedelta(days=1)).tz_convert("UTC")
        predicate = (arrow.field("bar_start_utc") >= start) & (arrow.field("bar_start_utc") < end)
    if ids is not None:
        predicate = predicate & arrow.field("decision_id").isin(ids)
    frame: pd.DataFrame = dataset.to_table(columns=columns, filter=predicate, use_threads=False).to_pandas()
    return frame


def _select_features(frame: pd.DataFrame, strategy: StrategyContract) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise DataReadinessError("control feature columns must be unique")
    frame = frame.loc[:, _FEATURES].copy()
    if frame["session_date_et"].isna().any():
        raise DataReadinessError("control session identity is missing")
    frame["session_date_et"] = frame["session_date_et"].astype(str)
    if frame["decision_id"].duplicated().any() or frame.duplicated(["session_date_et", "security_id"]).any():
        raise DataReadinessError("control feature decisions are duplicated")
    eligible = frame["feature_eligible"].map(_strict_bool) & frame["cross_section_eligible"].map(_strict_bool)
    candidates = frame.loc[eligible].copy()
    for name in ("decision_id", "decision_group_id", "security_id", "sector", "primary_benchmark", "session_date_et"):
        if not candidates[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise DataReadinessError(f"control candidate identity is missing: {name}")
    for group, other in (("decision_group_id", "session_date_et"), ("session_date_et", "decision_group_id")):
        if candidates.groupby(group, observed=True)[other].nunique().gt(1).any():
            raise DataReadinessError("control requires one decision cohort per session")
    decision = pd.to_datetime(candidates["decision_time_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(candidates["feature_available_at_utc"], utc=True, errors="coerce")
    if decision.isna().any() or available.isna().any() or available.gt(decision).any():
        raise DataReadinessError("control features must be available before decision")
    if not np.isfinite(pd.to_numeric(candidates["return_20d_xs_rank"], errors="coerce")).all():
        raise DataReadinessError("control score is missing; do not silently remove selected candidates")
    return select_constrained_swing_portfolio(
        candidates, score_column="return_20d_xs_rank", maximum_trades=strategy.swing.maximum_trades_per_decision,
        target_maximum_sector_weight=strategy.swing.target_maximum_sector_weight,
        hard_maximum_sector_weight=strategy.swing.hard_maximum_sector_weight,
        minimum_distinct_sectors=strategy.swing.minimum_distinct_sectors_for_selection,
    )


def _join_outcomes(selected: pd.DataFrame, outcomes: pd.DataFrame, valuation: tuple[str, ...]) -> pd.DataFrame:
    if (outcomes["decision_id"].duplicated().any()
            or set(outcomes["decision_id"]) != set(selected["decision_id"])):
        raise DataReadinessError("selected control outcome missing or duplicated; no survivor filtering permitted")
    joined = selected.merge(outcomes.loc[:, _OUTCOMES], on="decision_id", how="left", validate="one_to_one")
    nonfinite = ~np.isfinite(joined[_FIXED].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64"))
    if nonfinite.any():
        raise _IncompleteFixedOutcomes({
            "code": "selected_fixed_horizon_nonfinite",
            "selected_row_count": len(joined), "affected_row_count": int(nonfinite.any(axis=1).sum()),
            "nonfinite_value_count": int(nonfinite.sum()),
            "field_counts": {name: int(count) for name, count in zip(_FIXED, nonfinite.sum(axis=0), strict=True) if count},
            "first_decision_ids": joined.loc[nonfinite.any(axis=1), "decision_id"].head(5).tolist(),
        })
    calendar = xcals.get_calendar("XNYS")
    entry, exit_time, available, managed = [pd.to_datetime(joined[name], utc=True, errors="coerce") for name in (
        "entry_time_utc", "exit_time_utc", "label_available_at_utc", "barrier_label_available_at_utc",
    )]
    # The pinned combined-daily canonicalizer publishes each close fifteen minutes later.
    publication_delay = pd.Timedelta(minutes=15)
    for i, row in enumerate(joined.to_dict(orient="records")):
        first = calendar.session_offset(row["session_date_et"], 1)
        last = calendar.session_offset(row["session_date_et"], 10)
        expected_close = calendar.session_close(last)
        if (str(first.date()) not in valuation or str(last.date()) not in valuation
                or entry.iloc[i] != calendar.session_open(first) or exit_time.iloc[i] != expected_close
                or str(row["entry_session_date_et"]) != str(first.date())
                or str(row["exit_session_date_et"]) != str(last.date()) or row["horizon_sessions"] != 10
                or available.iloc[i] != expected_close + publication_delay
                or pd.isna(managed.iloc[i]) or managed.iloc[i] < entry.iloc[i] or managed.iloc[i] > available.iloc[i]):
            raise DataReadinessError("selected outcome violates exact next-open/tenth-close or initial-fit tail")
    return joined


class _IncompleteFixedOutcomes(DataReadinessError):
    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__("selected control fixed-horizon outcome is incomplete: " + json.dumps(details, sort_keys=True))


def _bound_report(
    root: Path, policy: _Policy, bound: dict[str, str], selected: pd.DataFrame,
    decisions: tuple[str, ...], valuation: tuple[str, ...],
) -> dict[str, Any]:
    for path_string, expected_hash in bound.items():
        if file_sha256(Path(path_string)) != expected_hash:
            raise DataReadinessError("control input changed during audit")
    ids = selected["decision_id"].tolist()
    return {
        "schema_version": policy.schema_version, "status": "price_ratio_diagnostics", "eligible": False,
        "price_basis_status": "price_basis_pending", "control": "same_universe_return_20d_xs_rank_descending",
        "scope": "initial_fit_deterministic_control_not_out_of_sample", "trained_models": 0,
        "validation_or_test_outcomes_read": False, "full_authority_replay": False,
        "bound_files": {Path(path).relative_to(root).as_posix() if Path(path).is_relative_to(root) else path: digest
                        for path, digest in sorted(bound.items())},
        "selected_decision_ids": ids, "selected_decision_ids_sha256": sequence_sha256(ids),
        "decision_sessions": list(decisions), "decision_sessions_sha256": sequence_sha256(decisions),
        "valuation_sessions": list(valuation), "valuation_sessions_sha256": sequence_sha256(valuation),
    }


def _publish(directory: Path, report: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "_manifest.json"
    with file_lock(path):
        if path.exists():
            raise DataReadinessError("accounting control output is immutable; choose a new output directory")
        temporary = directory / f".{uuid4().hex}.tmp"
        try:
            write_json_object(temporary, report)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def run_swing_accounting_control_audit(root: Path, policy_path: Path, output_directory: Path) -> dict[str, Any]:
    """Own the root-scoped heavy lease; CLI callers must not acquire it a second time."""
    root = root.resolve()
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("audit-swing-accounting-control", runtime_dir=root / runtime, config_path=policy_path):
        try:
            return _run(root, policy_path, output_directory)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise DataReadinessError(f"accounting control audit failed: {exc}") from exc


def _run(root: Path, policy_path: Path, output_directory: Path) -> dict[str, Any]:
    if policy_path.stat().st_size > 64 * 1024:
        raise DataReadinessError("control policy exceeds size bound")
    bound = {str(policy_path.resolve()): file_sha256(policy_path)}
    policy = _Policy.model_validate(tomllib.loads(policy_path.read_text(encoding="utf-8")))
    for relative in (policy.temporal_policy, policy.training_policy, policy.strategy_contract, policy.research_contract):
        path = resolve_inside_authority(root, relative)
        bound[str(path)] = file_sha256(path)
    decisions, valuation = _calendars(root, policy)  # Must precede all parquet access.
    config = load_swing_training_config(root / policy.training_policy)
    strategy = load_strategy_contract(root / policy.strategy_contract)
    research = load_swing_research_contract(root / policy.research_contract)
    research.assert_strategy_matches(strategy)
    if config.maximum_trades_per_decision != strategy.swing.maximum_trades_per_decision or config.maximum_trades_per_decision != 25:
        raise DataReadinessError("control requires the frozen 25-trade cap")
    panel, combined = _sources(root, policy, strategy, bound)
    panel_root = root / policy.panel_directory
    months = {day[:7] for day in decisions}
    records = [record for record in panel["files_by_profile"]["technical_market"] if record["partition_month"] in months]
    if len(records) != len(months) or {item["partition_month"] for item in records} != months:
        raise DataReadinessError("control requires exactly one retained partition per requested month")
    selections: list[pd.DataFrame] = []
    for record in records:
        _guard(config, "control feature projection", peak=False)
        path = _checked_file(panel_root / "final", record["path"], record["sha256"], bound)
        frame = _read_projection(path, _FEATURES, decisions)
        expected = {day for day in decisions if day[:7] == record["partition_month"]}
        if set(frame["session_date_et"].astype(str)) != expected:
            raise DataReadinessError("control feature calendar is incomplete")
        selections.append(_select_features(frame, strategy))
    selected = pd.concat(selections, ignore_index=True).sort_values(["session_date_et", "security_id", "decision_id"])
    del selections, frame
    outcomes: list[pd.DataFrame] = []
    for record in records:
        ids = selected.loc[selected["session_date_et"].str.startswith(record["partition_month"]), "decision_id"].tolist()
        if ids:
            _guard(config, "control selected outcomes", peak=False)
            path = _checked_file(panel_root / "final", record["path"], record["sha256"], bound)
            outcomes.append(_read_projection(path, _OUTCOMES, decisions, ids))
    try:
        selected = _join_outcomes(
            selected, pd.concat(outcomes, ignore_index=True) if outcomes else pd.DataFrame(columns=_OUTCOMES), valuation,
        )
    except _IncompleteFixedOutcomes as exc:
        # Only this diagnosed failure after frozen selection produces a blocked receipt.
        report = _bound_report(root, policy, bound, selected, decisions, valuation)
        report.update(status="blocked", errors=[exc.details], accounting_status="not_run", benchmark_payloads_read=False)
        report["audit_sha256"] = json_sha256(report)
        _guard(config, "blocked control report publication", peak=True)
        _publish(output_directory, report)
        return report
    needed = {"SPY", "QQQ", *selected["primary_benchmark"].astype(str)}
    bars: list[pd.DataFrame] = []
    benchmark_records = [item for item in combined["artifacts"] if item["ticker"] in needed]
    if len(benchmark_records) != len(needed) or {item["ticker"] for item in benchmark_records} != needed:
        raise DataReadinessError("control benchmark manifest is incomplete or duplicated")
    for item in benchmark_records:
        path = _checked_file(panel_root / "combined_daily", item["path"], item["sha256"], bound)
        sidecar = _metadata(path.parent, manifest_path_for(path).name, item["canonical_manifest_sha256"], bound)
        if sidecar["artifact_sha256"] != item["sha256"]:
            raise DataReadinessError("control benchmark sidecar differs")
        frame = _read_projection(path, ["ticker", "bar_start_utc", "open", "close", "price_feed", "adjustment"], valuation)
        if (not frame["ticker"].eq(item["ticker"]).all() or not frame["price_feed"].eq("sip").all()
                or not frame["adjustment"].eq("all").all()):
            raise DataReadinessError("control benchmark source identity differs")
        frame["session_date_et"] = pd.to_datetime(frame["bar_start_utc"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
        bars.append(frame)
    accounting = evaluate_funded_swing_accounting(
        selected, pd.concat(bars, ignore_index=True), config=config, strategy_contract=strategy,
        research_contract=research, session_calendar=decisions,
    )
    if accounting["eligible"] is not False or accounting["price_basis_status"] != "price_basis_pending":
        raise DataReadinessError("control cannot admit an unverified price basis")
    report = _bound_report(root, policy, bound, selected, decisions, valuation)
    report["accounting"] = accounting
    report["audit_sha256"] = json_sha256(report)
    _guard(config, "control report publication", peak=True)
    _publish(output_directory, report)
    return report
