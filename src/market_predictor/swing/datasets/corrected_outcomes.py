"""Monthly, resumable initial-fit targets from independently replayed raw sources.

No action terms are inferred. Admission is source-bounded retrospective research,
not universal absence, historical availability, training or production permission.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
import tomllib
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.resources import assert_memory_budget, memory_audit
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.contracts.holding_materialization import PositionSourceBinding
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.contracts.trade_simulation import TradeSimulationContext
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence, load_corporate_action_evidence
from market_predictor.swing.datasets.action_scope import verify_action_scope_documents
from market_predictor.swing.datasets.corporate_action_scope import SUCCESSORS, prepare_corporate_action_scope
from market_predictor.swing.datasets.corrected_outcome_admission import (
    corrected_decisions,
    corrected_memberships,
    corrected_ticker,
    reconcile_actions,
    window_gaps,
)
from market_predictor.swing.datasets.history_archive import load_complete_swing_history_collection
from market_predictor.swing.datasets.holding_raw_sources import read_bound_observations
from market_predictor.swing.datasets.initial_fit_raw_share_plan import (
    IDENTITY_COLUMNS,
    MEMBERSHIP_COLUMNS,
    verified_initial_fit_raw_share_plan,
)
from market_predictor.swing.datasets.symbol_corrected_sources import reconstruct_symbol_corrected_sources
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.evaluation.holding_accounting import project_holding_targets
from market_predictor.swing.evaluation.trade_simulation import load_trade_simulation_context
from market_predictor.swing.labels.holding_identity import membership_session_coverage
from market_predictor.swing.labels.holding_paths import holding_calendar
from market_predictor.swing.labels.ordinary_holding import OrdinaryHoldingResult, build_ordinary_holding

DECISION_COLUMNS = (*IDENTITY_COLUMNS, "timeframe", "bar_start_utc", "prediction_cutoff_policy_id")
TARGET_COLUMNS = ("fixed_horizon_gross_return", "fixed_horizon_net_return", "managed_horizon_gross_return",
    "managed_horizon_net_return", "spy_horizon_gross_return", "qqq_horizon_gross_return",
    "sector_horizon_gross_return", "spy_fixed_horizon_excess_return", "qqq_fixed_horizon_excess_return",
    "sector_fixed_horizon_excess_return")


class SpecificationStream(Protocol):
    def write(self, data: bytes, /) -> int: ...


def _guard() -> None:
    assert_memory_budget(stage="corrected outcome publication", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0)


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(_json(value))
    temporary.replace(path)


def _check(root: Path, files: dict[str, str]) -> None:
    for name, digest in files.items():
        if file_sha256(inside(root, name)) != digest:
            raise DataReadinessError(f"corrected outcome source changed: {name}")


def _check_published(output: Path, request_pin: str, months: dict[str, Any]) -> None:
    files = {"_request.json": request_pin}
    for month, record in months.items():
        files.update({f"{month}/{name}": digest for name, digest in record["files"].items()})
    _check(output, files)


def load_corrected_outcome_policy(root: Path, config: Path, expected_sha256: str) -> CorrectedOutcomePolicy:
    path = inside(root, config)
    if path.stat().st_size > 1024**2:
        raise DataReadinessError("corrected outcome configuration exceeds bound")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise DataReadinessError("corrected outcome configuration requires its external file pin")
    return CorrectedOutcomePolicy.model_validate_json(json.dumps(tomllib.loads(data.decode("utf-8"))))


def _projection(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    parquet: Any = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
    if parquet.metadata.num_rows > 1_000_000 or parquet.metadata.serialized_size > 8 * 1024**2:
        raise DataReadinessError("corrected outcome identity partition exceeds bound")
    return parquet.read(columns=list(columns)).to_pandas()


@contextmanager
def verified_corrected_research_sources(root: Path, config: Path, config_sha256: str, policy: CorrectedOutcomePolicy,
    evidence: CorporateActionEvidence) -> Iterator[dict[str, Any]]:
    """The canonical plan owns the one numerical/publication lease through yield."""
    selected = pinned_object(inside(root, policy.source_selection.path), policy.source_selection.sha256)
    correction_plan = inside(root, selected["correction_plan"])
    authority = pinned_object(correction_plan / "_authority.json", selected["correction_plan_sha256"])
    correction = pinned_object(correction_plan / "_request.json", authority["request_sha256"])
    parent = inside(root, correction["policy"]["parent_plan"])
    parent_pin = correction["policy"]["parent_plan_sha256"]
    with verified_initial_fit_raw_share_plan(root, inside(root, policy.parent_config.path), parent,
        expected_plan_sha256=parent_pin) as plan:
        _guard()
        pins = {str(inside(root, config).relative_to(root)): config_sha256}
        for pin in (policy.source_selection, policy.parent_config, policy.action_config, policy.symbol_corrections,
                policy.research_contract, policy.simulation_policy, *(w.document for w in policy.official_windows)):
            pins[inside(root, pin.path).relative_to(root).as_posix()] = pin.sha256
        _check(root, pins)
        if selected["policy_sha256"] != policy.symbol_corrections.sha256:
            raise DataReadinessError("decision corrections must share the reconstructed price correction authority")
        # Freeze metadata bytes before independent reconstruction validates them.
        # Collection request hashes are semantic, unlike plan request file hashes.
        authorities = ((correction_plan, selected["correction_plan_sha256"]), (parent, parent_pin),
            (inside(root, selected["correction_archive"]), selected["correction_archive_sha256"]),
            (inside(root, correction["policy"]["parent_archive"]), correction["policy"]["parent_archive_sha256"]))
        for directory, digest in authorities:
            archive_authority = pinned_object(directory / "_authority.json", digest)
            pins[(directory / "_authority.json").relative_to(root).as_posix()] = digest
            pins[(directory / "_manifest.json").relative_to(root).as_posix()] = archive_authority["artifact_sha256"]
            pins[(directory / "_request.json").relative_to(root).as_posix()] = file_sha256(directory / "_request.json")
            units = directory / "daily_bar_units.csv"
            if units.exists():
                pins[units.relative_to(root).as_posix()] = file_sha256(units)
        rebuilt = reconstruct_symbol_corrected_sources(root=root, correction_plan=correction_plan,
            correction_plan_sha256=selected["correction_plan_sha256"],
            correction_archive=inside(root, selected["correction_archive"]),
            correction_archive_sha256=selected["correction_archive_sha256"], loader=load_complete_swing_history_collection)
        if rebuilt != selected:
            raise DataReadinessError("corrected source selection differs from independent reconstruction")
        _check(root, pins)
        parent_authority = pinned_object(parent / "_authority.json", parent_pin)
        request = pinned_object(parent / "_request.json", parent_authority["request_sha256"])
        pins.update(request["source_files"])
        pins[(parent / "_authority.json").relative_to(root).as_posix()] = parent_pin
        pins[(parent / "_request.json").relative_to(root).as_posix()] = parent_authority["request_sha256"]
        if (len(request["retained_security_ids"]) != 586 or len(request["excluded_security_ids"]) != 45
                or request["decision_start"] != str(policy.decision_start)
                or request["initial_fit_end"] != str(policy.numerical_end)):
            raise DataReadinessError("corrected outcomes differ from the frozen 586-security initial-fit population")
        parent_policy = tomllib.loads(inside(root, policy.parent_config.path).read_text(encoding="utf-8"))
        preflight_path = inside(root, parent_policy["preflight_path"])
        preflight = pinned_object(preflight_path, pins[preflight_path.relative_to(root).as_posix()])
        base = preflight["request"]
        membership_path = inside(root, base["membership_path"])
        manifest_path = inside(root, base["parent_manifest_path"])
        manifest = pinned_object(manifest_path, pins[manifest_path.relative_to(root).as_posix()])
        memberships = _projection(membership_path, MEMBERSHIP_COLUMNS)
        memberships = corrected_memberships(memberships, policy.decision_corrections)
        scope = prepare_corporate_action_scope(root, inside(root, policy.action_config.path),
            tomllib.loads(inside(root, policy.action_config.path).read_text(encoding="utf-8")))
        expected = {s["artifact"]["provider_symbol"] for s in selected["segments"]} | SUCCESSORS
        if len(expected) != 570 or set(scope["tickers"]) != expected:
            raise DataReadinessError("corrected outcomes need all 570 full-cohort action queries")
        action_request_path = inside(root, policy.action_archive) / "_request.json"
        action_request = pinned_object(action_request_path,
            evidence.source_files[action_request_path.relative_to(root).as_posix()])
        if (action_request["request_sha256"] != evidence.request_sha256
                or action_request["policy"] != scope["policy"] or set(action_request["tickers"]) != expected):
            raise DataReadinessError("action evidence request does not bind the corrected full selection")
        correction_policy = tomllib.loads(inside(root, policy.symbol_corrections.path).read_text(encoding="utf-8"))
        reviewed_ids = {doc for rule in correction_policy["corrections"] for doc in rule["document_ids"]}
        if any(not set(rule.document_ids).issubset(reviewed_ids) for rule in policy.decision_corrections):
            raise DataReadinessError("decision correction references an unreviewed source document")
        evidence.recheck(root)
        pins.update(evidence.source_files)
        pins.update(verify_action_scope_documents(root, policy.reviewed_cash_distribution_scopes))
        for segment in selected["segments"]:
            artifact = segment["artifact"]
            path = inside(inside(root, segment["archive"]), artifact["bars_path"])
            pins[path.relative_to(root).as_posix()] = artifact["bars_sha256"]
        yield {"selection": selected, "plan": plan, "request": request, "memberships": memberships,
            "manifest": manifest, "manifest_path": manifest_path, "source_files": pins,
            "action_index": reconcile_actions(evidence, expected, policy.reviewed_cash_distribution_scopes),
            "membership_path": membership_path}
        _check(root, pins)
        evidence.recheck(root)


class MonthEngine:
    """Bounded per-unit raw frames, per-ticker ownership and per-ETF/date lot cache."""

    def __init__(self, root: Path, policy: CorrectedOutcomePolicy, config_sha256: str, sources: dict[str, Any],
        simulation: TradeSimulationContext, first: date, last: date, specifications: SpecificationStream) -> None:
        self.root, self.policy, self.config_sha256, self.sources = root, policy, config_sha256, sources
        self.simulation, self.first, self.last, self.specifications = simulation, first, last, specifications
        self.calendar = xcals.get_calendar("XNYS")
        self.sessions = holding_calendar(first, last)
        self.research_sha256 = load_swing_research_contract(inside(root, policy.research_contract.path)).sha256()
        self.segments: dict[str, list[dict[str, Any]]] = {}
        for segment in sources["selection"]["segments"]:
            self.segments.setdefault(segment["artifact"]["security_id"], []).append(segment)
        self.raw: dict[str, tuple[pd.DataFrame, dict[date, EvidenceReference]]] = {}
        self.ownership: dict[tuple[str, str], set[date]] = {}
        self.benchmarks: dict[tuple[str, date], tuple[OrdinaryHoldingResult | None, tuple[str, ...], str | None]] = {}
        self.components = 0

    def release_stock(self) -> None:
        benchmark_units = {s["artifact"]["unit_id"] for group in self.segments.values() for s in group
            if s["artifact"]["role"] == "benchmark"}
        self.raw = {key: value for key, value in self.raw.items() if key in benchmark_units}
        self.ownership.clear()

    def component(self, decision: dict[str, Any], sessions: tuple[date, ...], *, benchmark: bool = False,
        ) -> tuple[OrdinaryHoldingResult | None, tuple[str, ...], str | None]:
        identity = str(decision["security_id"])
        matches = [s for s in self.segments.get(identity, ())
            if date.fromisoformat(s["first_session"]) <= sessions[0]
            and date.fromisoformat(s["last_session"]) >= sessions[-1]]
        if len(matches) != 1:
            return None, ("missing_or_cross_segment_raw_binding",), None
        segment = matches[0]
        artifact = segment["artifact"]
        expected_role = "benchmark" if benchmark else "stock"
        if artifact["role"] != expected_role or benchmark and identity != f"benchmark:{decision['ticker']}":
            raise DataReadinessError("canonical benchmark or stock source role differs")
        if any(corrected_ticker(identity, artifact["ticker"], session, self.policy.decision_corrections)
                != decision["ticker"] for session in (decision["session_date_et"], *sessions)):
            return None, ("raw_logical_ticker_not_bound_to_corrected_decision_interval",), None
        actions, global_gaps = self.sources["action_index"]
        gaps = window_gaps(identity=identity, symbol=artifact["provider_symbol"], first=sessions[0], last=sessions[-1],
            actions=actions, official=self.policy.official_windows, global_gaps=global_gaps)
        if not benchmark:
            key = (identity, str(decision["ticker"]))
            if key not in self.ownership:
                memberships = self.sources["memberships"]
                owners = memberships.loc[memberships.ticker.eq(key[1])]
                if owners.empty:
                    self.ownership[key] = set()
                else:
                    coverage = membership_session_coverage(owners, sessions=self.sessions, security_ids=(identity,))
                    self.ownership[key] = set(coverage.loc[coverage.membership_covered].index.get_level_values(1))
            if not set(sessions).issubset(self.ownership[key]):
                gaps += ("positive_membership_ownership_missing_or_competing",)
        if gaps:
            return None, tuple(sorted(set(gaps))), None
        unit = artifact["unit_id"]
        if unit not in self.raw:
            lower = max(self.first, date.fromisoformat(segment["first_session"]))
            upper = min(self.last, date.fromisoformat(segment["last_session"]))
            reference = EvidenceReference(reference=self.policy.source_selection.path,
                artifact_sha256=self.policy.source_selection.sha256, interpretation_policy_sha256=self.config_sha256,
                record_locator=f"selected unit {unit}; ownership independently checked against pinned memberships",
                retrieved_at=self.simulation.policy_reference.retrieved_at, available_at=None)
            binding = PositionSourceBinding(position_id=f"raw:{unit}", security_id=identity, unit_id=unit,
                provider_symbol=artifact["provider_symbol"], bars_sha256=artifact["bars_sha256"],
                first_session=lower, last_session=upper, ownership_evidence=(reference,))
            self.raw[unit] = read_bound_observations(self.root, self.sources["selection"], binding,
                first=lower, last=upper, interpretation_sha256=self.config_sha256)
        observations, evidence = self.raw[unit]
        bounded = observations.loc[observations.index.isin(sessions)]
        result = build_ordinary_holding(decision={**decision, "ticker": artifact["ticker"]}, sessions=sessions,
            observations=bounded, evidence={day: evidence[day] for day in sessions if day in evidence},
            research_contract_sha256=self.research_sha256, cost_prepaid_fraction=self.policy.cost_prepaid_fraction,
            simulation=self.simulation)
        if result.outcome is None or result.targets is None or result.specification is None:
            return result, tuple(g.code for g in result.missing_reasons) or ("ordinary_replay_unavailable",), None
        component_id = str(decision["decision_id"])
        record = {"component_id": component_id, "security_id": identity, "unit_id": unit,
            "source_ticker": artifact["ticker"], "provider_symbol": artifact["provider_symbol"],
            "admission": "source_bounded_no_reported_action_positive_identity",
            "specification": result.specification.model_dump(mode="json"),
            "outcome_sha256": json_sha256(result.outcome.model_dump(mode="json"))}
        self.specifications.write(_json(record) + b"\n")
        self.components += 1
        return result, (), component_id

    def row(self, decision: dict[str, Any]) -> dict[str, Any]:
        day = decision["session_date_et"]
        sessions = tuple(s.date() for s in self.calendar.sessions_window(pd.Timestamp(day), 11)[1:])
        row = {**decision, **dict.fromkeys(TARGET_COLUMNS), "managed_exit_timestamp": None,
            "label_available_at": None, "research_label_mature_at": None,
            "stock_component_id": None, "production_eligible": False, "training_eligible": False,
            "managed_missing_reasons": "managed_policy_not_materialized"}
        stock: OrdinaryHoldingResult | None = None
        mature = sessions[-1] <= self.policy.numerical_end
        if mature:
            stock, stock_gaps, row["stock_component_id"] = self.component(decision, sessions)
        else:
            stock_gaps = ("initial_fit_terminal_immature",)
        row["stock_missing_reasons"] = json.dumps(stock_gaps)
        row["stock_source_admitted"] = not stock_gaps
        benchmark_outcomes = {}
        all_benchmarks = True
        for role, symbol in (("spy", "SPY"), ("qqq", "QQQ"), ("sector", decision["primary_benchmark"])):
            key = (str(symbol), day)
            if key not in self.benchmarks:
                identity = f"benchmark:{symbol}"
                benchmark_decision = {key: value for key, value in decision.items() if key != "decision_id"}
                benchmark_decision.update(security_id=identity, ticker=symbol, sector="benchmark")
                benchmark_decision = stamp_canonical_decision_ids(pd.DataFrame([benchmark_decision])).iloc[0].to_dict()
                self.benchmarks[key] = (self.component(benchmark_decision, sessions, benchmark=True) if mature
                    else (None, ("initial_fit_terminal_immature",), None))
            result, gaps, component_id = self.benchmarks[key]
            row[f"{role}_security_id"] = f"benchmark:{symbol}"
            row[f"{role}_component_id"] = component_id
            row[f"{role}_missing_reasons"] = json.dumps(gaps)
            all_benchmarks &= not gaps
            if not gaps and result is not None and result.outcome is not None:
                benchmark_outcomes[result.outcome.security_id] = result.outcome
                target = project_holding_targets(result.outcome, None)
                row[f"{role}_horizon_gross_return"] = target.fixed_horizon_gross_return
        if not stock_gaps and stock is not None and stock.outcome is not None:
            projected = project_holding_targets(stock.outcome, None, benchmarks=tuple(benchmark_outcomes.values()))
            row["fixed_horizon_gross_return"] = projected.fixed_horizon_gross_return
            row["fixed_horizon_net_return"] = projected.fixed_horizon_net_return
            row["label_available_at"] = projected.label_available_at
            row["research_label_mature_at"] = projected.horizon_end_timestamp
            by_identity = {b.benchmark_security_id: b for b in projected.benchmarks}
            for role in ("spy", "qqq", "sector"):
                comparison = by_identity.get(row[f"{role}_security_id"])
                if comparison is not None:
                    row[f"{role}_fixed_horizon_excess_return"] = comparison.fixed_horizon_excess_return
        row["fixed_comparisons_complete"] = not stock_gaps and all_benchmarks
        return row


def load_corrected_decision_partition(root: Path, source: dict[str, Any], record: dict[str, Any],
    policy: CorrectedOutcomePolicy) -> pd.DataFrame:
    """Project frozen initial-fit identity/context columns and retain parent IDs."""
    path = inside(root, source["manifest_path"].parent / record["path"])
    if source["source_files"].get(path.relative_to(root).as_posix()) != record["sha256"]:
        raise DataReadinessError("decision partition lacks frozen raw-plan source authority")
    frame = _projection(path, DECISION_COLUMNS)
    if len(frame) != record["rows"]:
        raise DataReadinessError("parent decision partition inventory differs")
    frame["session_date_et"] = pd.to_datetime(frame.session_date_et).dt.date
    frame = frame.loc[frame.session_date_et.between(policy.decision_start, policy.numerical_end)
        & frame.security_id.isin(source["request"]["retained_security_ids"])].copy()
    if frame.empty:
        return frame
    return corrected_decisions(frame, policy.decision_corrections).sort_values(
        ["security_id", "session_date_et", "decision_id"], kind="stable").reset_index(drop=True)


def _month(root: Path, destination: Path, frame: pd.DataFrame, policy: CorrectedOutcomePolicy,
    config_sha256: str, sources: dict[str, Any], simulation: TradeSimulationContext) -> dict[str, Any]:
    _guard()
    first = min(frame.session_date_et)
    last = min(policy.numerical_end, xcals.get_calendar("XNYS").sessions_window(
        pd.Timestamp(max(frame.session_date_et)), 11)[-1].date())
    destination.mkdir()
    rows: list[dict[str, Any]] = []
    with (destination / "specifications.jsonl.gz").open("wb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", filename="", mtime=0) as stream:
        engine = MonthEngine(root, policy, config_sha256, sources, simulation, first, last, stream)
        for _, group in frame.groupby("security_id", sort=True):
            engine.release_stock()
            for decision in group.to_dict("records"):
                if len(rows) % policy.batch_rows == 0:
                    _guard()
                rows.append(engine.row(decision))
    output = pd.DataFrame(rows)
    for column in TARGET_COLUMNS:
        output[column] = pd.array(output[column], dtype="Float64")
    for column in ("managed_exit_timestamp", "label_available_at", "research_label_mature_at"):
        output[column] = pd.to_datetime(output[column], utc=True)
    output.to_parquet(destination / "targets.parquet", index=False)
    missing_counts: Counter[str] = Counter()
    for role in ("stock", "spy", "qqq", "sector"):
        for reasons in output[f"{role}_missing_reasons"]:
            missing_counts.update(f"{role}:{reason}" for reason in json.loads(reasons))
    return {"rows": len(output), "stock_source_admitted": int(output.stock_source_admitted.sum()),
        "fixed_comparisons_complete": int(output.fixed_comparisons_complete.sum()), "specifications": engine.components,
        "missing_reason_counts": dict(sorted(missing_counts.items())),
        "parent_decision_ids_sha256": json_sha256(sorted(frame.parent_decision_id)),
        "decision_ids_sha256": json_sha256(sorted(frame.decision_id)),
        "files": {name: file_sha256(destination / name) for name in ("targets.parquet", "specifications.jsonl.gz")}}


def _implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    names = ("swing/datasets/corrected_outcomes.py", "swing/datasets/corrected_outcome_admission.py",
        "swing/contracts/corrected_outcomes.py", "swing/datasets/action_evidence.py", "swing/datasets/action_windows.py",
        "swing/datasets/corporate_action_collection.py", "swing/datasets/corporate_action_scope.py",
        "swing/contracts/holding_accounting.py", "swing/contracts/trade_simulation.py", "swing/contracts/research.py",
        "swing/labels/ordinary_holding.py", "swing/labels/holding_identity.py", "swing/datasets/holding_raw_sources.py",
        "swing/labels/holding_paths.py", "swing/evaluation/holding_accounting.py", "swing/evaluation/trade_simulation.py",
        "canonical/reconciliation.py", "canonical/cutoffs.py", "swing/datasets/action_scope.py", "sources/official_documents.py",
        "swing/datasets/history_archive.py", "swing/datasets/session_requirements.py",
        "evidence/implementation_snapshot.py", "evidence/io.py", "universe/symbol_correction_policy.py")
    return {(package / name).relative_to(root).as_posix(): file_sha256(package / name) for name in names}


def materialize_corrected_outcomes(*, root: Path, config: Path, expected_config_sha256: str, output: Path,
    expected_output_sha256: str | None = None, replay: bool = False,
    maximum_months: int | None = None) -> dict[str, Any]:
    """Build/resume monthly outputs, or independently reconstruct every pinned target/spec.

    A resume needs the external checkpoint file hash. Completed months are immutable
    and reused after hash checks; --replay recomputes each month through the same
    source admission/compiler/kernel path and compares both Parquet and specifications.
    """
    root = root.resolve()
    started = time.perf_counter()
    if maximum_months is not None and (type(maximum_months) is not int or not 1 <= maximum_months <= 60):
        raise DataReadinessError("month limit must be an explicit integer from 1 to 60")
    config, output = inside(root, config), inside(root, output)
    if not output.is_relative_to(root / "data" / "labels"):
        raise DataReadinessError("corrected outcomes must publish beneath data/labels")
    policy = load_corrected_outcome_policy(root, config, expected_config_sha256)
    if output.exists() and expected_output_sha256 is None:
        raise DataReadinessError("existing corrected outcomes require an external manifest/checkpoint pin")
    if not output.exists() and (expected_output_sha256 is not None or replay):
        raise DataReadinessError("cannot recreate missing pinned corrected outcomes")
    _guard()
    evidence = load_corporate_action_evidence(root=root, config=inside(root, policy.action_config.path),
        archive=inside(root, policy.action_archive), expected_audit_sha256=policy.action_audit_sha256)
    with verified_corrected_research_sources(root, config, expected_config_sha256, policy, evidence) as source:
        _guard()
        implementation = _implementation(root)
        lineage = {"config_sha256": expected_config_sha256, "source_files": source["source_files"],
            "implementation_files": implementation, "cohort_sha256": source["request"]["cohort_sha256"],
            "action_audit_sha256": evidence.audit_sha256, "action_request_sha256": evidence.request_sha256,
            "action_replay_metadata": evidence.replay_metadata, "source_selection_sha256": policy.source_selection.sha256}
        existing: dict[str, Any] | None = None
        if output.exists():
            pin_path = output / ("_manifest.json" if (output / "_manifest.json").exists() else "_checkpoint.json")
            existing = pinned_object(pin_path, expected_output_sha256)
            request = pinned_object(output / "_request.json", existing["request_sha256"])
            if request["lineage"] != lineage:
                raise DataReadinessError("resume/replay source or implementation identity changed")
            simulation = TradeSimulationContext.model_validate_json(json.dumps(request["simulation"]))
        else:
            simulation = load_trade_simulation_context(inside(root, policy.simulation_policy.path),
                expected_sha256=policy.simulation_policy.sha256)
            request = {"schema": "market_predictor.corrected_outcome_request", "lineage": lineage,
                "simulation": simulation.model_dump(mode="json")}
            output.mkdir(parents=True)
            _write(output / "_request.json", request)
        request_pin = file_sha256(output / "_request.json")
        records = [r for r in source["manifest"]["files"] if r["first_session"] <= str(policy.numerical_end)
            and r["last_session"] >= str(policy.decision_start)]
        months: dict[str, Any] = {} if existing is None else dict(existing["months"])
        if not set(months).issubset(r["partition_month"] for r in records):
            raise DataReadinessError("output contains non-initial-fit decision months")
        checkpoint = {"schema": "market_predictor.corrected_outcomes", "status": "partial_in_progress",
            "request_sha256": request_pin, "months": months, "training_eligible": False, "promotion_eligible": False,
            "managed_available": False, "exclusions_added": [], "authority": "source_bounded_retrospective_research",
            "action_query_participants": len(evidence.records_by_symbol),
            "global_source_admission_gaps": list(source["action_index"][1])}
        if existing is None:
            _write(output / "_checkpoint.json", checkpoint)
        completed_this_run = 0
        for record in records:
            _guard()
            month = record["partition_month"]
            frame = load_corrected_decision_partition(root, source, record, policy)
            if frame.empty:
                continue
            destination = output / month
            if month in months:
                prior = months[month]
                if (prior["rows"] != len(frame) or prior["parent_decision_ids_sha256"] != json_sha256(sorted(frame.parent_decision_id))
                        or prior["decision_ids_sha256"] != json_sha256(sorted(frame.decision_id))):
                    raise DataReadinessError("resumed decision population differs")
                _check(output, {f"{month}/{name}": digest for name, digest in prior["files"].items()})
                if not replay:
                    continue
            if maximum_months is not None and completed_this_run >= maximum_months:
                _check(root, implementation)
                _check_published(output, request_pin, months)
                complete_path = output / "_manifest.json"
                return {**checkpoint, "status": "partial_specification_replay" if replay else "partial_in_progress",
                    "rows": sum(m["rows"] for m in months.values()),
                    "stock_source_admitted": sum(m["stock_source_admitted"] for m in months.values()),
                    "fixed_comparisons_complete": sum(m["fixed_comparisons_complete"] for m in months.values()),
                    "manifest_sha256": file_sha256(complete_path) if complete_path.exists() else None,
                    "checkpoint_sha256": file_sha256(output / "_checkpoint.json"),
                    "run_wall_seconds": time.perf_counter() - started,
                    "run_resources": memory_audit(hard_budget_gib=5.0, headroom_gib=0.75).to_record()}
            staging = output / f".{month}.{uuid4().hex}.pending"
            month_started = time.perf_counter()
            result = _month(root, staging, frame, policy, expected_config_sha256, source, simulation)
            completed_this_run += 1
            if month in months:
                if result != months[month]:
                    raise DataReadinessError("corrected outcomes differ from independent source/specification replay")
                for name in result["files"]:
                    (staging / name).unlink()
                staging.rmdir()
            else:
                _check(root, implementation)
                if destination.exists():
                    # A crash between rename and checkpoint publication is recoverable
                    # only by independently regenerating and comparing the complete month.
                    _check(output, {f"{month}/{name}": digest for name, digest in result["files"].items()})
                    if {p.name for p in destination.iterdir()} != set(result["files"]):
                        raise DataReadinessError("uncheckpointed month contains unexpected files")
                    for name in result["files"]:
                        (staging / name).unlink()
                    staging.rmdir()
                else:
                    staging.rename(destination)
                evidence.recheck(root)
                months[month] = result
                _write(output / "_checkpoint.json", checkpoint)
            progress = {"month": month, "rows": result["rows"], "stock_source_admitted": result["stock_source_admitted"],
                "specifications": result["specifications"], "wall_seconds": time.perf_counter() - month_started,
                "resources": memory_audit(hard_budget_gib=5.0, headroom_gib=0.75).to_record(),
                "checkpoint_sha256": file_sha256(output / "_checkpoint.json")}
            _write(output / "_progress.json", progress)
        if sum(m["rows"] for m in months.values()) != source["plan"]["requirements"]["in_window_decisions"]:
            raise DataReadinessError("corrected outcomes lost frozen initial-fit decisions")
        _check(root, source["source_files"])
        _check(root, implementation)
        _check_published(output, request_pin, months)
        evidence.recheck(root)
        result = {**checkpoint, "status": "partial_research_outcomes", "rows": sum(m["rows"] for m in months.values()),
            "stock_source_admitted": sum(m["stock_source_admitted"] for m in months.values()),
            "fixed_comparisons_complete": sum(m["fixed_comparisons_complete"] for m in months.values())}
        if existing is not None and (output / "_manifest.json").exists():
            if result != existing:
                raise DataReadinessError("corrected outcome manifest differs from replay")
        else:
            _write(output / "_manifest.json", result)
        return {**result, "manifest_sha256": file_sha256(output / "_manifest.json")}
