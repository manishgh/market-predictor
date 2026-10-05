"""Source-bound funded-policy inputs; missing action economics remain unavailable.

The caller preloads action evidence before entering this context. The existing
corrected-source context holds the sole workspace lease through lazy selection,
caller publication and final source rechecks. Do not wrap it in another lease.
Exit selection and settlement remain in their existing kernels; returned lots
have no generated simulation payments.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingSpecification, KnownMark
from market_predictor.swing.contracts.holding_materialization import PositionSourceBinding, SourcePin
from market_predictor.swing.contracts.research import SwingResearchContract, load_swing_research_contract
from market_predictor.swing.contracts.trade_simulation import TradeSimulationContext
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.corrected_outcome_admission import corrected_ticker, window_gaps
from market_predictor.swing.datasets.corrected_outcomes import (
    load_corrected_outcome_policy,
    verified_corrected_research_sources,
)
from market_predictor.swing.datasets.holding_raw_sources import read_bound_observations
from market_predictor.swing.evaluation.holding_accounting import replay_holding
from market_predictor.swing.evaluation.ledger import swing_valuation_sessions
from market_predictor.swing.evaluation.trade_simulation import load_trade_simulation_context, simulate_ordinary_sales
from market_predictor.swing.labels.frozen_exit import DecisionAtr, ExitPolicy, FrozenExitResult, compile_frozen_exit
from market_predictor.swing.labels.holding_identity import membership_session_coverage
from market_predictor.swing.labels.holding_paths import validate_outcome_observations

START, LAST_ENTRY, END = date(2019, 7, 9), date(2024, 5, 13), date(2024, 5, 28)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


@dataclass(frozen=True)
class FundedLotInput:
    decision_id: str
    canonical_security_id: str
    specification: HoldingSpecification | None = None
    decision_atr: DecisionAtr | None = None
    exit_result: FrozenExitResult | None = None
    missing_reasons: tuple[str, ...] = ()


def unsimulated_managed_specification(result: FrozenExitResult) -> HoldingSpecification:
    """Remove only generated settlement mechanics and verify the original digest."""
    replay = result.simulation
    _require(replay is not None, "managed exit has no simulation to reverse")
    assert replay is not None
    events, marks = set(replay.generated_event_ids), set(replay.generated_mark_keys)
    spec = HoldingSpecification.model_validate({**replay.specification.model_dump(),
        "events": tuple(event for event in replay.specification.events if event.event_id not in events),
        "marks": tuple(mark for mark in replay.specification.marks if (mark.position_id, mark.mark_at) not in marks)})
    _require(json_sha256(spec.model_dump(mode="json")) == replay.input_specification_sha256,
             "removing generated settlement did not reconstruct the managed input")
    return spec


def raw_decision_atr(*, decision: dict[str, Any], observations: pd.DataFrame,
                     evidence: dict[date, EvidenceReference]) -> DecisionAtr | None:
    """Mean of fourteen raw true ranges over fifteen exact completed sessions."""
    calendar = xcals.get_calendar("XNYS")
    day = decision["session_date_et"]
    days = tuple(value.date() for value in calendar.sessions_window(pd.Timestamp(day), -15))
    _require(len(days) == 15 and days[-1] == day, "ATR window is not fifteen completed sessions")
    _require(observations.index.is_unique and set(observations.index).issubset(days), "ATR observations escape decision window")
    if any(value not in observations.index or value not in evidence for value in days):
        return None
    frame = observations.loc[list(days)]
    _require(frame.security_id.eq(decision["security_id"]).all() and frame.adjustment.eq("raw").all()
             and frame.price_feed.eq("sip").all() and frame.timeframe.eq("1Day").all(), "ATR raw identity/basis differs")
    checked = validate_outcome_observations(frame.reset_index()).set_index("session_date_et")
    if not frame.outcome_observation_valid.eq(True).all() or not checked.outcome_observation_valid.eq(True).all():
        return None
    _require(all(pd.Timestamp(frame.loc[session, "bar_end_utc"]) == calendar.session_close(session.isoformat())
                 for session in days), "ATR observation clocks differ from completed sessions")
    previous = frame.close.shift(1)
    ranges = pd.concat([frame.high - frame.low, (frame.high - previous).abs(), (frame.low - previous).abs()], axis=1).max(axis=1)
    value = float(ranges.iloc[1:].mean())
    if not 0 < value < float("inf"):
        return None
    for session in days:
        ref = evidence[session]
        _require(ref.available_at is None and ref.record_locator ==
                 f"security_id={decision['security_id']};session_date={session}"
                 and ref.retrieved_at == pd.Timestamp(frame.loc[session, "ingested_at_utc"]), "ATR row evidence differs")
    _require(len({(ref.reference, ref.artifact_sha256, ref.interpretation_policy_sha256)
                  for ref in evidence.values()}) == 1, "ATR splices source authorities")
    close = calendar.session_close(day.isoformat()).to_pydatetime()
    _require(decision["decision_time_utc"] == swing_prediction_cutoffs(pd.Series([day])).iloc[0], "ATR cutoff differs")
    reference = evidence[day].model_copy(update={"record_locator":
        f"raw ATR14 mean true range;security_id={decision['security_id']};sessions={days[0]}..{day};"
        f"observation_evidence_sha256={json_sha256([evidence[value].model_dump(mode='json') for value in days])}"})
    return DecisionAtr(decision_id=decision["decision_id"], security_id=decision["security_id"], value=value,
        observation_end_at=close, available_at=decision["decision_time_utc"], evidence=reference)


class FundedPolicyInputProvider:
    """Bounded per-request reads; neither selection nor missing-outcome filtering."""

    def __init__(self, *, root: Path, policy: CorrectedOutcomePolicy, config: SourcePin,
                 evidence: CorporateActionEvidence, sources: dict[str, Any], strategy: StrategyContract,
                 research: SwingResearchContract, simulation: TradeSimulationContext, selected: pd.DataFrame,
                 session_calendar: tuple[str, ...], exit_policy: ExitPolicy) -> None:
        self.root, self.policy, self.config, self.evidence, self.sources = root, policy, config, evidence, sources
        self.strategy, self.research, self.simulation, self.exit_policy = strategy, research, simulation, exit_policy
        self.source_files = dict(sources["source_files"])
        self.valuation_sessions = swing_valuation_sessions(session_calendar, 10)
        _require(START <= date.fromisoformat(session_calendar[0]) <= date.fromisoformat(session_calendar[-1]) <= LAST_ENTRY
                 and date.fromisoformat(self.valuation_sessions[-1]) <= END, "funded calendar escapes initial fit")
        names = {"decision_id", "security_id", "ticker", "sector", "session_date_et", "decision_time_utc", "primary_benchmark"}
        _require(selected.columns.is_unique and names.issubset(selected.columns)
                 and not selected.decision_id.duplicated().any(), "funded selected identities missing or duplicate")
        self._decisions: dict[str, dict[str, Any]] = {}
        for row in selected.to_dict("records"):
            _require(all(isinstance(row[name], str) and bool(row[name].strip()) for name in names -
                         {"session_date_et", "decision_time_utc"}), "funded selection has empty identity")
            day = date.fromisoformat(row["session_date_et"]) if isinstance(row["session_date_et"], str) else row["session_date_et"]
            _require(type(day) is date and day.isoformat() in session_calendar, "selected decision outside calendar")
            row["session_date_et"] = day
            _require(row["decision_time_utc"] == swing_prediction_cutoffs(pd.Series([day])).iloc[0], "selected cutoff differs")
            self._decisions[row["decision_id"]] = row
        self.selected_ids = tuple(self._decisions)
        self.benchmark_tickers = tuple(sorted({"SPY", "QQQ", *selected.primary_benchmark.astype(str)}))

    def recheck(self) -> None:
        self.evidence.recheck(self.root)
        for name, digest in self.source_files.items():
            _require(file_sha256(inside(self.root, name)) == digest, f"funded source changed: {name}")

    def _read(self, identity: str, ticker: str, days: tuple[date, ...], *, benchmark: bool
              ) -> tuple[pd.DataFrame, dict[date, EvidenceReference], tuple[str, ...]]:
        if days[0] < START or days[-1] > END:
            return pd.DataFrame(), {}, ("raw_window_outside_initial_fit",)
        matches = [segment for segment in self.sources["selection"]["segments"]
                   if segment["artifact"]["security_id"] == identity
                   and date.fromisoformat(segment["first_session"]) <= days[0]
                   and date.fromisoformat(segment["last_session"]) >= days[-1]]
        if len(matches) != 1:
            return pd.DataFrame(), {}, ("missing_or_cross_segment_raw_binding",)
        segment = matches[0]
        artifact = segment["artifact"]
        _require(artifact["role"] == ("benchmark" if benchmark else "stock"), "funded source role differs")
        if any(corrected_ticker(identity, artifact["ticker"], day, self.policy.decision_corrections) != ticker for day in days):
            return pd.DataFrame(), {}, ("raw_logical_ticker_not_bound_to_corrected_interval",)
        actions, global_gaps = self.sources["action_index"]
        gaps = window_gaps(identity=identity, symbol=artifact["provider_symbol"], first=days[0], last=days[-1],
            actions=actions, official=self.policy.official_windows, global_gaps=global_gaps)
        if gaps:
            return pd.DataFrame(), {}, tuple([*gaps, "action_entitlement_payment_and_residual_marks_not_admitted"])
        if not benchmark:
            owners = self.sources["memberships"].loc[self.sources["memberships"].ticker.eq(ticker)]
            coverage = membership_session_coverage(owners, sessions=days, security_ids=(identity,))
            if not bool(coverage.membership_covered.all()) or len(coverage) != len(days):
                return pd.DataFrame(), {}, ("positive_membership_ownership_missing_or_competing",)
        reference = EvidenceReference(reference=self.policy.source_selection.path,
            artifact_sha256=self.policy.source_selection.sha256, interpretation_policy_sha256=self.config.sha256,
            record_locator=f"selected raw unit {artifact['unit_id']};canonical_security_id={identity}",
            retrieved_at=self.simulation.policy_reference.retrieved_at, available_at=None)
        binding = PositionSourceBinding(position_id=f"raw:{artifact['unit_id']}", security_id=identity,
            unit_id=artifact["unit_id"], provider_symbol=artifact["provider_symbol"], bars_sha256=artifact["bars_sha256"],
            first_session=days[0], last_session=days[-1], ownership_evidence=(reference,))
        frame, refs = read_bound_observations(self.root, self.sources["selection"], binding,
            first=days[0], last=days[-1], interpretation_sha256=self.config.sha256)
        gaps = tuple(f"raw_observation_unavailable:{identity}:{day}" for day in days
                     if day not in frame.index or day not in refs or not bool(frame.loc[day, "outcome_observation_valid"]))
        return frame, refs, gaps

    def _specification(self, decision: dict[str, Any], days: tuple[date, ...], frame: pd.DataFrame,
                       refs: dict[date, EvidenceReference], *, benchmark: bool) -> HoldingSpecification:
        calendar = xcals.get_calendar("XNYS")
        ends = tuple(calendar.session_close(day.isoformat()).to_pydatetime() for day in days)
        position = f"{decision['decision_id']}:ordinary-shares"
        return HoldingSpecification(research_contract_sha256=self.research.sha256(), decision_id=decision["decision_id"],
            security_id=decision["ticker"] if benchmark else decision["security_id"], sector=decision["sector"],
            initial_position_id=position, initial_entry_price=float(frame.loc[days[0], "open"]),
            initial_entry_timestamp=calendar.session_open(days[0].isoformat()).to_pydatetime(),
            price_basis="raw_with_no_adjustment", currency="USD", entry_evidence=(refs[days[0]],),
            session_end_timestamps=ends, cost_prepaid_fraction=0.0 if benchmark else self.policy.cost_prepaid_fraction,
            policy="fixed_horizon", marks=tuple(KnownMark(position_id=position, mark_at=end,
                value_per_unit=float(frame.loc[day, "close"]), currency="USD", evidence=(refs[day],))
                for day, end in zip(days, ends, strict=True)))

    def lot(self, decision_id: str) -> FundedLotInput:
        _require(decision_id in self._decisions, "funded lot was not selected")
        row = self._decisions[decision_id]
        identity, ticker, day = row["security_id"], row["ticker"], row["session_date_et"]
        calendar = xcals.get_calendar("XNYS")
        warmup = tuple(value.date() for value in calendar.sessions_window(pd.Timestamp(day), -15))
        frame, refs, gaps = self._read(identity, ticker, warmup, benchmark=False)
        if gaps:
            return FundedLotInput(decision_id, identity, missing_reasons=tuple(f"atr:{gap}" for gap in gaps))
        atr = raw_decision_atr(decision=row, observations=frame, evidence=refs)
        if atr is None:
            return FundedLotInput(decision_id, identity, missing_reasons=("raw_decision_atr_unavailable",))
        days = tuple(value.date() for value in calendar.sessions_window(pd.Timestamp(day), 11)[1:])
        frame, refs, gaps = self._read(identity, ticker, days, benchmark=False)
        if gaps:
            return FundedLotInput(decision_id, identity, decision_atr=atr, missing_reasons=gaps)
        spec = self._specification(row, days, frame, refs, benchmark=False)
        if spec.initial_entry_price <= 1.5 * atr.value:
            return FundedLotInput(decision_id, identity, decision_atr=atr, missing_reasons=("nonpositive_raw_atr_stop",))
        result = compile_frozen_exit(specification=spec, decision_session=day, decision_time=row["decision_time_utc"],
            decision_atr=atr, observations=frame, evidence=refs, strategy=self.strategy, research=self.research,
            exit_policy=self.exit_policy, simulation=self.simulation)
        if result.simulation is None or result.missing_reasons:
            return FundedLotInput(decision_id, identity, decision_atr=atr, exit_result=result,
                missing_reasons=tuple(f"{gap.code}:{gap.reference_id}" for gap in result.missing_reasons)
                or ("managed_exit_unavailable",))
        managed = unsimulated_managed_specification(result)
        # Only ordinary-sale proceeds can remain: action-bearing paths returned
        # explicit gaps above. The canonical simulator supplies their later marks.
        ends = tuple(calendar.session_close(value).to_pydatetime() for value in self.valuation_sessions
                     if value >= days[0].isoformat())
        managed = HoldingSpecification.model_validate({**managed.model_dump(), "session_end_timestamps": ends})
        replay = simulate_ordinary_sales(managed, self.simulation)
        gaps = tuple(sorted({f"{gap.code}:{gap.reference_id}" for snapshot in replay.outcome.snapshots for gap in snapshot.gaps}))
        return FundedLotInput(decision_id, identity, managed if not gaps else None, atr, result, gaps)

    def benchmark(self, ticker: str) -> FundedLotInput:
        _require(ticker in self.benchmark_tickers, "unrequested funded benchmark")
        identity, key = f"benchmark:{ticker}", f"buy-and-hold:{ticker}"
        days = tuple(date.fromisoformat(value) for value in self.valuation_sessions)
        frame, refs, gaps = self._read(identity, ticker, days, benchmark=True)
        if gaps:
            return FundedLotInput(key, identity, missing_reasons=gaps)
        spec = self._specification({"decision_id": key, "security_id": identity, "ticker": ticker, "sector": "benchmark"},
            days, frame, refs, benchmark=True)
        outcome = replay_holding(spec)
        gaps = tuple(sorted({f"{gap.code}:{gap.reference_id}" for snapshot in outcome.snapshots for gap in snapshot.gaps}))
        return FundedLotInput(key, identity, spec if not gaps else None, missing_reasons=gaps)


@contextmanager
def verified_funded_policy_inputs(*, root: Path, target_config: SourcePin, evidence: CorporateActionEvidence,
    strategy: StrategyContract, selection_loader: Callable[[], tuple[pd.DataFrame, tuple[str, ...]]], exit_policy: ExitPolicy,
) -> Iterator[FundedPolicyInputProvider]:
    """Load selection only while the existing corrected-source lease is held.

    The callback may load pinned OOF payloads and prepare the immutable request.
    The caller must publish its report inside this context; source rechecks and
    the underlying lease exit follow that publication without a second lease.
    """
    root = root.resolve()
    policy = load_corrected_outcome_policy(root, Path(target_config.path), target_config.sha256)
    _require(evidence.audit_sha256 == policy.action_audit_sha256, "funded action evidence audit differs")
    research = load_swing_research_contract(inside(root, policy.research_contract.path))
    research.assert_strategy_matches(strategy)
    _require(exit_policy in research.exit_policies, "funded exit is outside frozen research")
    simulation = load_trade_simulation_context(inside(root, policy.simulation_policy.path),
        expected_sha256=policy.simulation_policy.sha256)
    with verified_corrected_research_sources(root, Path(target_config.path), target_config.sha256, policy, evidence) as sources:
        selected, session_calendar = selection_loader()
        provider = FundedPolicyInputProvider(root=root, policy=policy, config=target_config, evidence=evidence,
            sources=sources, strategy=strategy, research=research, simulation=simulation, selected=selected,
            session_calendar=session_calendar, exit_policy=exit_policy)
        provider.recheck()
        try:
            yield provider
        finally:
            provider.recheck()
