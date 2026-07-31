"""ER3 population-readiness and deterministic-economics baseline harness.

The harness answers two separate questions. ``ready_for_modeling`` proves that a
candidate population is causal, complete, and statistically evaluable.
``baseline_economics_passed`` records whether the unlearned comparator clears the
frozen economic floor. A negative comparator is useful evidence but cannot veto a
distinct preregistered classifier or ranker; learned-policy economics are evaluated
out of sample after fitting.

Contract of the input frame (one row per ``security_id`` / decision timestamp):

``security_id``, ``ticker``, ``session``, ``decision_time_utc``, ``gross_return``,
``cost``, ``net_return``, ``spy_excess_return``, ``sector_excess_return``, ``scope``
(``walk_forward`` or ``unseen_ticker``), ``phase`` (the non-overlapping horizon phase
index for swing, or the session-block index for intraday), plus one column per
declared concentration dimension (``sector``, ``market_regime``, ``session_segment``).

Economics semantics, aligned with ``AGENTS.md`` 4.3:

- ``cost`` is the stamped round-trip cost already deducted from ``net_return``. The
  identity ``net_return == gross_return - cost`` is verified, never recomputed, so a
  cost can never be applied twice.
- ``spy_excess_return`` and ``sector_excess_return`` are benchmark-relative returns
  measured over the identical executable interval and are already net of that same
  single cost deduction. This module never re-deducts from them.
- The cost/adverse-fill stress reuses the frozen
  :mod:`market_predictor.execution_policy` stress grid. A stress multiplier scales
  commission, half-spread, slippage, and impact together, which is exactly the
  adverse-fill widening that policy models, so the stressed economics subtract only
  the *incremental* cost ``cost * (multiplier - 1)``.

Statistical semantics, aligned with ``AGENTS.md`` 4.4:

- Row counts are not sample size. Overlapping labels are evaluated inside
  non-overlapping phases, and within a phase the confidence bound comes from a
  session-block bootstrap (:func:`session_block_mean_interval`) over whole sessions.
- Every economic baseline gate is aggregated conservatively across phases: the
  worst phase decides.

Fail-closed behavior:

- A malformed frame (missing column, unknown scope, broken cost identity, duplicate
  decision rows) raises :class:`DataReadinessError`. It never degrades to a pass.
- Anything computable but unevaluable (empty scope, missing phase, single-valued
  concentration dimension, non-finite statistic) is reported as a failing gate.
- Every gate is always evaluated and reported. Nothing short-circuits, because a
  reproducible negative baseline has to remain informative.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    execution_policy_identity,
)
from market_predictor.regime_evidence import (
    REGIME_BOOTSTRAP_ITERATIONS,
    REGIME_BOOTSTRAP_SEED,
    maximum_drawdown_from_returns,
    net_profit_factor,
    session_block_mean_interval,
)
from market_predictor.v3.errors import DataReadinessError

SETUP_ECONOMICS_SCHEMA = "edge_rebuild.setup_economics.v2"

WALK_FORWARD_SCOPE = "walk_forward"
UNSEEN_TICKER_SCOPE = "unseen_ticker"
EVALUATION_SCOPES = (WALK_FORWARD_SCOPE, UNSEEN_TICKER_SCOPE)

MANDATORY_CONCENTRATION_DIMENSIONS = ("ticker", "sector", "market_regime")
DEFAULT_CONCENTRATION_DIMENSIONS = (
    *MANDATORY_CONCENTRATION_DIMENSIONS,
    "session_segment",
)

REQUIRED_SETUP_COLUMNS = (
    "security_id",
    "ticker",
    "session",
    "decision_time_utc",
    "gross_return",
    "cost",
    "net_return",
    "spy_excess_return",
    "sector_excess_return",
    "scope",
    "phase",
)

_NUMERIC_COLUMNS = (
    "gross_return",
    "cost",
    "net_return",
    "spy_excess_return",
    "sector_excess_return",
)
_SESSION_DATE_COLUMN = "__setup_session_date"
_UNKNOWN_CATEGORY = "unknown"

_Direction = Literal["minimum", "minimum_inclusive", "maximum_inclusive"]
_GateCategory = Literal["readiness", "baseline_economics"]


class SetupEconomicsConfig(BaseModel):
    """Frozen ER3 readiness and deterministic-baseline thresholds.

    Every bound is declared so that the model cannot be weakened: the pydantic
    constraints pin each threshold at or above the plan's frozen value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SETUP_ECONOMICS_SCHEMA
    required_phases: int = Field(default=10, ge=1, le=20)
    minimum_rows_per_phase: int = Field(default=200, ge=200)
    minimum_sessions_per_phase: int = Field(default=60, ge=60)
    minimum_average_gross_return: float = Field(default=0.0, ge=0.0)
    minimum_average_net_return: float = Field(default=0.0, ge=0.0)
    minimum_average_spy_excess_return: float = Field(default=0.0, ge=0.0)
    minimum_average_sector_excess_return: float = Field(default=0.0, ge=0.0)
    minimum_net_return_block_ci_low: float = Field(default=0.0, ge=0.0)
    minimum_spy_excess_block_ci_low: float = Field(default=0.0, ge=0.0)
    minimum_stress_average_net_return: float = Field(default=0.0, ge=0.0)
    minimum_stress_average_spy_excess_return: float = Field(default=0.0, ge=0.0)
    minimum_profit_factor: float = Field(default=1.05, ge=1.05)
    maximum_drawdown: float = Field(default=0.20, gt=0.0, le=0.20)
    cost_stress_multiplier: float = Field(default=2.0, ge=2.0)
    concentration_dimensions: tuple[str, ...] = DEFAULT_CONCENTRATION_DIMENSIONS
    maximum_leave_one_out_values: int = Field(default=24, ge=2, le=100)
    bootstrap_iterations: int = Field(
        default=REGIME_BOOTSTRAP_ITERATIONS,
        ge=1_000,
        le=10_000,
    )
    bootstrap_seed: int = REGIME_BOOTSTRAP_SEED
    net_cost_identity_tolerance: float = Field(default=1e-9, gt=0.0, le=1e-6)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != SETUP_ECONOMICS_SCHEMA:
            raise ValueError("unsupported setup-economics schema")
        normalized = tuple(value.strip() for value in self.concentration_dimensions)
        if any(not value for value in normalized):
            raise ValueError("concentration dimensions must be non-empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("concentration dimensions must be unique")
        if missing := [
            value
            for value in MANDATORY_CONCENTRATION_DIMENSIONS
            if value not in normalized
        ]:
            raise ValueError(
                "concentration dimensions must include " + ", ".join(missing)
            )
        if self.cost_stress_multiplier not in DEFAULT_EXECUTION_POLICY.stress_multipliers:
            raise ValueError(
                "cost stress multiplier is not on the frozen execution-policy stress grid"
            )
        return self

    @property
    def required_columns(self) -> tuple[str, ...]:
        ordered = dict.fromkeys(
            (*REQUIRED_SETUP_COLUMNS, *self.concentration_dimensions)
        )
        return tuple(ordered)

    def spec(self) -> dict[str, Any]:
        """Content-addressable identity: the thresholds plus the cost policy they stress."""

        return {**self.model_dump(mode="json"), **execution_policy_identity()}

    def sha256(self) -> str:
        encoded = json.dumps(
            self.spec(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_SETUP_ECONOMICS_CONFIG = SetupEconomicsConfig()


@dataclass(frozen=True)
class GateResult:
    """One frozen readiness or baseline gate and its measured margin."""

    gate: str
    category: _GateCategory
    passed: bool
    measured: float
    threshold: float
    margin: float
    direction: _Direction
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "category": self.category,
            "passed": self.passed,
            "measured": self.measured,
            "threshold": self.threshold,
            "margin": self.margin,
            "direction": self.direction,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PhaseEconomics:
    """Measured economics for one non-overlapping phase of one scope."""

    phase: int
    rows: int
    session_blocks: int
    tickers: int
    securities: int
    average_gross_return: float
    average_cost: float
    average_net_return: float
    average_spy_excess_return: float
    average_sector_excess_return: float
    net_return_block_ci_low: float
    net_return_block_ci_high: float
    spy_excess_block_ci_low: float
    spy_excess_block_ci_high: float
    win_rate: float
    profit_factor: float
    maximum_drawdown: float
    stress_average_net_return: float
    stress_average_spy_excess_return: float
    stress_average_sector_excess_return: float
    stress_net_return_block_ci_low: float
    stress_maximum_drawdown: float
    bootstrap_iterations: int

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class PopulationReport:
    """Readiness and baseline evidence for one population or subset."""

    label: str
    rows: int
    session_blocks: int
    phases_present: tuple[int, ...]
    phase_economics: tuple[PhaseEconomics, ...]
    gates: tuple[GateResult, ...]
    ready_for_modeling: bool
    baseline_economics_passed: bool

    @property
    def readiness_failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            _gate_reason(gate)
            for gate in self.gates
            if gate.category == "readiness" and not gate.passed
        )

    @property
    def baseline_economic_shortfalls(self) -> tuple[str, ...]:
        return tuple(
            _gate_reason(gate)
            for gate in self.gates
            if gate.category == "baseline_economics" and not gate.passed
        )

    @property
    def failed_readiness_gates(self) -> tuple[str, ...]:
        return tuple(
            gate.gate
            for gate in self.gates
            if gate.category == "readiness" and not gate.passed
        )

    @property
    def failed_baseline_gates(self) -> tuple[str, ...]:
        return tuple(
            gate.gate
            for gate in self.gates
            if gate.category == "baseline_economics" and not gate.passed
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rows": self.rows,
            "session_blocks": self.session_blocks,
            "phases_present": list(self.phases_present),
            "phase_economics": [phase.as_dict() for phase in self.phase_economics],
            "gates": [gate.as_dict() for gate in self.gates],
            "ready_for_modeling": self.ready_for_modeling,
            "baseline_economics_passed": self.baseline_economics_passed,
            "failed_readiness_gates": list(self.failed_readiness_gates),
            "failed_baseline_gates": list(self.failed_baseline_gates),
            "readiness_failure_reasons": list(self.readiness_failure_reasons),
            "baseline_economic_shortfalls": list(self.baseline_economic_shortfalls),
        }


@dataclass(frozen=True)
class LeaveOneOutResult:
    """Economics of a scope after removing one value of a concentration dimension."""

    dimension: str
    excluded_value: str
    largest_contributor: bool
    contribution_share: float
    report: PopulationReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "excluded_value": self.excluded_value,
            "largest_contributor": self.largest_contributor,
            "contribution_share": self.contribution_share,
            "report": self.report.as_dict(),
        }


@dataclass(frozen=True)
class ScopeReport:
    """Complete readiness and baseline evidence for one validation scope."""

    scope: str
    baseline: PopulationReport
    leave_one_out: tuple[LeaveOneOutResult, ...]
    gates: tuple[GateResult, ...]
    ready_for_modeling: bool
    baseline_economics_passed: bool

    @property
    def readiness_failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            f"{self.scope}: {_gate_reason(gate)}"
            for gate in self.gates
            if gate.category == "readiness" and not gate.passed
        )

    @property
    def baseline_economic_shortfalls(self) -> tuple[str, ...]:
        return tuple(
            f"{self.scope}: {_gate_reason(gate)}"
            for gate in self.gates
            if gate.category == "baseline_economics" and not gate.passed
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "baseline": self.baseline.as_dict(),
            "leave_one_out": [result.as_dict() for result in self.leave_one_out],
            "gates": [gate.as_dict() for gate in self.gates],
            "ready_for_modeling": self.ready_for_modeling,
            "baseline_economics_passed": self.baseline_economics_passed,
            "readiness_failure_reasons": list(self.readiness_failure_reasons),
            "baseline_economic_shortfalls": list(self.baseline_economic_shortfalls),
        }


@dataclass(frozen=True)
class SetupEconomicsReport:
    """The ER3 readiness decision plus deterministic baseline evidence."""

    strategy_id: str
    schema_version: str
    config_sha256: str
    ready_for_modeling: bool
    baseline_economics_passed: bool
    scopes: tuple[ScopeReport, ...]

    def scope(self, name: str) -> ScopeReport:
        for report in self.scopes:
            if report.scope == name:
                return report
        raise KeyError(f"setup-economics report has no scope: {name}")

    def gate(self, scope: str, gate: str) -> GateResult:
        for result in self.scope(scope).gates:
            if result.gate == gate:
                return result
        raise KeyError(f"setup-economics scope {scope} has no gate: {gate}")

    @property
    def readiness_failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            reason
            for report in self.scopes
            for reason in report.readiness_failure_reasons
        )

    @property
    def baseline_economic_shortfalls(self) -> tuple[str, ...]:
        return tuple(
            reason
            for report in self.scopes
            for reason in report.baseline_economic_shortfalls
        )

    @property
    def failed_readiness_gates(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                gate.gate
                for report in self.scopes
                for gate in report.gates
                if gate.category == "readiness" and not gate.passed
            )
        )

    @property
    def failed_baseline_gates(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                gate.gate
                for report in self.scopes
                for gate in report.gates
                if gate.category == "baseline_economics" and not gate.passed
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "schema_version": self.schema_version,
            "config_sha256": self.config_sha256,
            "ready_for_modeling": self.ready_for_modeling,
            "baseline_economics_passed": self.baseline_economics_passed,
            "scopes": [report.as_dict() for report in self.scopes],
            "failed_readiness_gates": list(self.failed_readiness_gates),
            "failed_baseline_gates": list(self.failed_baseline_gates),
            "readiness_failure_reasons": list(self.readiness_failure_reasons),
            "baseline_economic_shortfalls": list(self.baseline_economic_shortfalls),
        }


def evaluate_setup_economics(
    setups: pd.DataFrame,
    *,
    strategy_id: str,
    config: SetupEconomicsConfig | None = None,
) -> SetupEconomicsReport:
    """Evaluate ER3 readiness and deterministic economics without gating ML on edge.

    Both ``walk_forward`` and ``unseen_ticker`` scopes must satisfy readiness. A
    missing scope is evaluated as empty and cannot be mistaken for model-ready.
    """

    resolved = config or DEFAULT_SETUP_ECONOMICS_CONFIG
    if not strategy_id.strip():
        raise DataReadinessError("setup-economics strategy ID is required")
    frame = _prepared_population(setups, config=resolved)
    scopes = tuple(
        _evaluate_scope(
            frame.loc[frame["scope"].eq(scope)],
            scope=scope,
            config=resolved,
        )
        for scope in EVALUATION_SCOPES
    )
    return SetupEconomicsReport(
        strategy_id=strategy_id.strip(),
        schema_version=resolved.schema_version,
        config_sha256=resolved.sha256(),
        ready_for_modeling=all(report.ready_for_modeling for report in scopes),
        baseline_economics_passed=all(
            report.baseline_economics_passed for report in scopes
        ),
        scopes=scopes,
    )


# --------------------------------------------------------------------------- #
# Input validation (fail closed on a malformed population)
# --------------------------------------------------------------------------- #
def _prepared_population(
    setups: pd.DataFrame,
    *,
    config: SetupEconomicsConfig,
) -> pd.DataFrame:
    required = config.required_columns
    if missing := sorted(set(required).difference(setups.columns)):
        raise DataReadinessError(
            "setup population is missing required columns: " + ", ".join(missing)
        )
    if setups.empty:
        raise DataReadinessError("setup population is empty")
    frame = setups.loc[:, list(required)].copy()
    _apply_numeric_columns(frame, config=config)
    _apply_identity_columns(frame, config=config)
    for dimension in config.concentration_dimensions:
        frame[dimension] = _normalized_category(frame[dimension])
    return frame


def _apply_numeric_columns(
    frame: pd.DataFrame,
    *,
    config: SetupEconomicsConfig,
) -> None:
    for column in _NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            raise DataReadinessError(
                f"setup population column is missing or non-finite: {column}"
            )
        frame[column] = values
    if bool(frame["cost"].lt(0.0).any()):
        raise DataReadinessError("setup population contains a negative stamped cost")
    residual = (frame["net_return"] - (frame["gross_return"] - frame["cost"])).abs()
    if float(residual.max()) > config.net_cost_identity_tolerance:
        raise DataReadinessError(
            "setup population violates the single-cost identity "
            "net_return == gross_return - cost"
        )


def _apply_identity_columns(
    frame: pd.DataFrame,
    *,
    config: SetupEconomicsConfig,
) -> None:
    for column in ("security_id", "ticker"):
        identity = _normalized_category(frame[column])
        if bool(identity.eq(_UNKNOWN_CATEGORY).any()):
            raise DataReadinessError(f"setup population has a blank {column}")
        frame[column] = identity

    scope = _normalized_category(frame["scope"])
    if unknown := sorted(set(scope).difference(EVALUATION_SCOPES)):
        raise DataReadinessError(
            "setup population has unknown validation scopes: " + ", ".join(unknown)
        )
    frame["scope"] = scope

    phase = pd.to_numeric(frame["phase"], errors="coerce")
    if bool(phase.isna().any()) or not bool(phase.mod(1).eq(0).all()):
        raise DataReadinessError("setup population phase index is not an integer")
    phase = phase.astype(int)
    if bool(phase.lt(0).any()) or bool(phase.ge(config.required_phases).any()):
        raise DataReadinessError(
            "setup population phase index is outside "
            f"0..{config.required_phases - 1}"
        )
    frame["phase"] = phase

    session = pd.to_datetime(frame["session"], errors="coerce")
    if bool(session.isna().any()):
        raise DataReadinessError("setup population session is not a valid date")
    frame[_SESSION_DATE_COLUMN] = session.dt.normalize()

    decision = pd.to_datetime(frame["decision_time_utc"], errors="coerce", utc=True)
    if bool(decision.isna().any()):
        raise DataReadinessError(
            "setup population decision_time_utc is not a valid UTC timestamp"
        )
    frame["decision_time_utc"] = decision
    duplicates = frame.duplicated(subset=["security_id", "decision_time_utc"])
    if bool(duplicates.any()):
        raise DataReadinessError(
            "setup population repeats a (security_id, decision_time_utc) decision "
            f"{int(duplicates.sum())} time(s)"
        )


def _normalized_category(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna(_UNKNOWN_CATEGORY).str.strip()
    return normalized.mask(normalized.eq(""), _UNKNOWN_CATEGORY).astype(str)


# --------------------------------------------------------------------------- #
# Scope evaluation
# --------------------------------------------------------------------------- #
def _evaluate_scope(
    scope_rows: pd.DataFrame,
    *,
    scope: str,
    config: SetupEconomicsConfig,
) -> ScopeReport:
    baseline = _evaluate_population(scope_rows, label="baseline", config=config)
    leave_one_out: list[LeaveOneOutResult] = []
    concentration_gates: list[GateResult] = []
    for dimension in config.concentration_dimensions:
        results, gates = _concentration_evidence(
            scope_rows,
            dimension=dimension,
            config=config,
        )
        leave_one_out.extend(results)
        concentration_gates.extend(gates)
    gates = (*baseline.gates, *concentration_gates)
    return ScopeReport(
        scope=scope,
        baseline=baseline,
        leave_one_out=tuple(leave_one_out),
        gates=gates,
        ready_for_modeling=_all_category_gates_pass(gates, "readiness"),
        baseline_economics_passed=(
            _all_category_gates_pass(gates, "readiness")
            and _all_category_gates_pass(gates, "baseline_economics")
        ),
    )


def _concentration_evidence(
    scope_rows: pd.DataFrame,
    *,
    dimension: str,
    config: SetupEconomicsConfig,
) -> tuple[tuple[LeaveOneOutResult, ...], tuple[GateResult, ...]]:
    """Leave-one-out robustness for one concentration dimension.

    The largest net contributor is always removed and re-evaluated. When the
    dimension is low-cardinality every value is removed in turn, which is what
    "no result dependent on one regime or one session segment" requires.
    """

    contributions = (
        scope_rows.groupby(dimension, sort=True, observed=True)["net_return"]
        .sum()
        .sort_index()
    )
    distinct = int(len(contributions))
    cardinality_gate = _gate_result(
        gate=f"concentration:{dimension}:distinct_values",
        category="readiness",
        measured=float(distinct),
        threshold=2.0,
        direction="minimum_inclusive",
        detail=(
            f"{dimension} has {distinct} distinct value(s); leave-one-out robustness "
            "needs at least two"
        ),
    )
    if distinct < 2:
        unevaluable = _gate_result(
            gate=f"concentration:{dimension}",
            category="baseline_economics",
            measured=float("nan"),
            threshold=0.0,
            direction="maximum_inclusive",
            detail=f"{dimension} leave-one-out is not evaluable",
        )
        return (), (cardinality_gate, unevaluable)

    total_contribution = float(contributions.sum())
    largest = str(contributions.idxmax())
    tested = (
        tuple(str(value) for value in contributions.index)
        if distinct <= config.maximum_leave_one_out_values
        else (largest,)
    )
    results: list[LeaveOneOutResult] = []
    for value in tested:
        subset = scope_rows.loc[~scope_rows[dimension].eq(value)]
        results.append(
            LeaveOneOutResult(
                dimension=dimension,
                excluded_value=value,
                largest_contributor=value == largest,
                contribution_share=(
                    float(contributions.loc[value]) / total_contribution
                    if total_contribution != 0.0
                    else float("nan")
                ),
                report=_evaluate_population(
                    subset,
                    label=f"leave_one_out:{dimension}={value}",
                    config=config,
                ),
            )
        )
    failing = tuple(
        result
        for result in results
        if not result.report.baseline_economics_passed
    )
    detail = (
        f"{dimension} leave-one-out: {len(results)} of {distinct} value(s) removed; "
        f"largest contributor {largest}"
    )
    if failing:
        detail += "; economics collapse without " + ", ".join(
            f"{result.excluded_value} ({_first_failure(result.report)})"
            for result in failing
        )
    return tuple(results), (
        cardinality_gate,
        _gate_result(
            gate=f"concentration:{dimension}",
            category="baseline_economics",
            measured=float(len(failing)),
            threshold=0.0,
            direction="maximum_inclusive",
            detail=detail,
        ),
    )


def _first_failure(report: PopulationReport) -> str:
    reasons = (
        *report.readiness_failure_reasons,
        *report.baseline_economic_shortfalls,
    )
    return reasons[0] if reasons else "no failing gate"


# --------------------------------------------------------------------------- #
# Population evaluation (one scope, or one leave-one-out subset of a scope)
# --------------------------------------------------------------------------- #
def _evaluate_population(
    rows: pd.DataFrame,
    *,
    label: str,
    config: SetupEconomicsConfig,
) -> PopulationReport:
    present = tuple(sorted({int(phase) for phase in rows["phase"].tolist()}))
    phase_economics = tuple(
        _phase_economics(
            rows.loc[rows["phase"].eq(phase)],
            phase=phase,
            config=config,
        )
        for phase in present
    )
    gates = _population_gates(
        phase_economics,
        phases_present=len(present),
        config=config,
    )
    return PopulationReport(
        label=label,
        rows=int(len(rows)),
        session_blocks=sum(phase.session_blocks for phase in phase_economics),
        phases_present=present,
        phase_economics=phase_economics,
        gates=gates,
        ready_for_modeling=_all_category_gates_pass(gates, "readiness"),
        baseline_economics_passed=(
            _all_category_gates_pass(gates, "readiness")
            and _all_category_gates_pass(gates, "baseline_economics")
        ),
    )


def _phase_economics(
    rows: pd.DataFrame,
    *,
    phase: int,
    config: SetupEconomicsConfig,
) -> PhaseEconomics:
    sessions = rows[_SESSION_DATE_COLUMN]
    net = rows["net_return"]
    spy = rows["spy_excess_return"]
    sector = rows["sector_excess_return"]
    # The stress multiplier scales the whole stamped round trip, so only the
    # incremental cost is subtracted. The base cost stays applied exactly once.
    stress_increment = rows["cost"] * (config.cost_stress_multiplier - 1.0)
    stress_net = net - stress_increment
    stress_spy = spy - stress_increment
    stress_sector = sector - stress_increment

    net_interval = _block_interval(sessions, net, config=config)
    spy_interval = _block_interval(sessions, spy, config=config)
    stress_net_interval = _block_interval(sessions, stress_net, config=config)
    return PhaseEconomics(
        phase=phase,
        rows=int(len(rows)),
        session_blocks=int(sessions.nunique()),
        tickers=int(rows["ticker"].nunique()),
        securities=int(rows["security_id"].nunique()),
        average_gross_return=float(rows["gross_return"].mean()),
        average_cost=float(rows["cost"].mean()),
        average_net_return=float(net.mean()),
        average_spy_excess_return=float(spy.mean()),
        average_sector_excess_return=float(sector.mean()),
        net_return_block_ci_low=float(net_interval["low"]),
        net_return_block_ci_high=float(net_interval["high"]),
        spy_excess_block_ci_low=float(spy_interval["low"]),
        spy_excess_block_ci_high=float(spy_interval["high"]),
        win_rate=float(net.gt(0.0).mean()),
        profit_factor=net_profit_factor(net),
        maximum_drawdown=_session_drawdown(rows, net),
        stress_average_net_return=float(stress_net.mean()),
        stress_average_spy_excess_return=float(stress_spy.mean()),
        stress_average_sector_excess_return=float(stress_sector.mean()),
        stress_net_return_block_ci_low=float(stress_net_interval["low"]),
        stress_maximum_drawdown=_session_drawdown(rows, stress_net),
        bootstrap_iterations=config.bootstrap_iterations,
    )


def _block_interval(
    sessions: pd.Series,
    values: pd.Series,
    *,
    config: SetupEconomicsConfig,
) -> dict[str, float | int]:
    """Session-block bootstrap of a row-level mean inside one phase.

    Rows are not independent: a swing label spans the whole holding horizon, so
    rows from nearby sessions share overlapping outcome windows. Two properties
    make this bound honest:

    1. It is computed *inside* one non-overlapping phase, where consecutive
       sessions are a full horizon apart and therefore do not share bars.
    2. Inside the phase it resamples whole sessions with replacement and takes the
       ratio of summed values to summed counts, so all rows decided on the same
       session move together and the effective sample size is the number of
       session blocks, not the number of rows.

    Naive per-row resampling would treat every overlapping row as new information
    and shrink the interval by roughly the square root of the overlap factor,
    which is how a population of 101,918 V2 rows that really carried about 125
    independent blocks produced bounds that looked far tighter than they were.
    """

    return session_block_mean_interval(
        sessions,
        values,
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )


def _session_drawdown(rows: pd.DataFrame, returns: pd.Series) -> float:
    session_returns = (
        returns.groupby(rows[_SESSION_DATE_COLUMN], sort=True, observed=True)
        .mean()
        .sort_index()
    )
    return maximum_drawdown_from_returns(session_returns)


def _population_gates(
    phase_economics: Sequence[PhaseEconomics],
    *,
    phases_present: int,
    config: SetupEconomicsConfig,
) -> tuple[GateResult, ...]:
    def worst(attribute: str, *, maximum: bool = False) -> float:
        return _worst(
            [float(getattr(phase, attribute)) for phase in phase_economics],
            maximum=maximum,
        )

    blocks = sum(phase.session_blocks for phase in phase_economics)
    return (
        _gate_result(
            gate="phase_coverage",
            category="readiness",
            measured=float(phases_present),
            threshold=float(config.required_phases),
            direction="minimum_inclusive",
            detail=(
                f"{phases_present} of {config.required_phases} non-overlapping "
                "phase(s) represented"
            ),
        ),
        _gate_result(
            gate="rows_per_phase",
            category="readiness",
            measured=worst("rows"),
            threshold=float(config.minimum_rows_per_phase),
            direction="minimum_inclusive",
            detail="worst phase row count",
        ),
        _gate_result(
            gate="session_blocks_per_phase",
            category="readiness",
            measured=worst("session_blocks"),
            threshold=float(config.minimum_sessions_per_phase),
            direction="minimum_inclusive",
            detail=(
                f"worst phase independent session blocks; {blocks} block(s) in total"
            ),
        ),
        _gate_result(
            gate="average_gross_return",
            measured=worst("average_gross_return"),
            threshold=config.minimum_average_gross_return,
            direction="minimum",
            detail="worst phase average gross return before costs",
        ),
        _gate_result(
            gate="average_net_return",
            measured=worst("average_net_return"),
            threshold=config.minimum_average_net_return,
            direction="minimum",
            detail="worst phase average net return after the stamped cost",
        ),
        _gate_result(
            gate="average_spy_excess_return",
            measured=worst("average_spy_excess_return"),
            threshold=config.minimum_average_spy_excess_return,
            direction="minimum",
            detail="worst phase average SPY-excess return",
        ),
        _gate_result(
            gate="average_sector_excess_return",
            measured=worst("average_sector_excess_return"),
            threshold=config.minimum_average_sector_excess_return,
            direction="minimum",
            detail="worst phase average sector-excess return",
        ),
        _gate_result(
            gate="net_return_block_ci_low",
            measured=worst("net_return_block_ci_low"),
            threshold=config.minimum_net_return_block_ci_low,
            direction="minimum",
            detail="worst phase 95% session-block lower bound for net return",
        ),
        _gate_result(
            gate="spy_excess_block_ci_low",
            measured=worst("spy_excess_block_ci_low"),
            threshold=config.minimum_spy_excess_block_ci_low,
            direction="minimum",
            detail="worst phase 95% session-block lower bound for SPY excess",
        ),
        _gate_result(
            gate="profit_factor",
            measured=worst("profit_factor"),
            threshold=config.minimum_profit_factor,
            direction="minimum_inclusive",
            detail="worst phase net profit factor",
            allow_infinite=True,
        ),
        _gate_result(
            gate="maximum_drawdown",
            measured=worst("maximum_drawdown", maximum=True),
            threshold=config.maximum_drawdown,
            direction="maximum_inclusive",
            detail="worst phase session-level maximum drawdown",
        ),
        _gate_result(
            gate="stress_average_net_return",
            measured=worst("stress_average_net_return"),
            threshold=config.minimum_stress_average_net_return,
            direction="minimum",
            detail=(
                "worst phase average net return under the frozen "
                f"{config.cost_stress_multiplier:g}x cost/adverse-fill stress"
            ),
        ),
        _gate_result(
            gate="stress_average_spy_excess_return",
            measured=worst("stress_average_spy_excess_return"),
            threshold=config.minimum_stress_average_spy_excess_return,
            direction="minimum",
            detail=(
                "worst phase average SPY excess under the frozen "
                f"{config.cost_stress_multiplier:g}x cost/adverse-fill stress"
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Gate primitives
# --------------------------------------------------------------------------- #
def _worst(values: Sequence[float], *, maximum: bool) -> float:
    if not values or any(math.isnan(value) for value in values):
        return float("nan")
    return max(values) if maximum else min(values)


def _gate_result(
    *,
    gate: str,
    category: _GateCategory = "baseline_economics",
    measured: float,
    threshold: float,
    direction: _Direction,
    detail: str,
    allow_infinite: bool = False,
) -> GateResult:
    value = float(measured)
    evaluable = math.isfinite(value) or (allow_infinite and value == math.inf)
    if not evaluable:
        return GateResult(
            gate=gate,
            category=category,
            passed=False,
            measured=value,
            threshold=threshold,
            margin=float("nan"),
            direction=direction,
            detail=f"{detail}; not evaluable, failing closed",
        )
    if direction == "minimum":
        return GateResult(
            gate=gate,
            category=category,
            passed=value > threshold,
            measured=value,
            threshold=threshold,
            margin=value - threshold,
            direction=direction,
            detail=detail,
        )
    if direction == "minimum_inclusive":
        return GateResult(
            gate=gate,
            category=category,
            passed=value >= threshold,
            measured=value,
            threshold=threshold,
            margin=value - threshold,
            direction=direction,
            detail=detail,
        )
    return GateResult(
        gate=gate,
        category=category,
        passed=value <= threshold,
        measured=value,
        threshold=threshold,
        margin=threshold - value,
        direction=direction,
        detail=detail,
    )


def _all_category_gates_pass(
    gates: Sequence[GateResult],
    category: _GateCategory,
) -> bool:
    selected = tuple(gate for gate in gates if gate.category == category)
    return bool(selected) and all(gate.passed for gate in selected)


def _gate_reason(gate: GateResult) -> str:
    comparison = {
        "minimum": "must exceed",
        "minimum_inclusive": "must be at least",
        "maximum_inclusive": "must not exceed",
    }[gate.direction]
    return (
        f"{gate.gate} {gate.measured:.10g} {comparison} {gate.threshold:.10g} "
        f"({gate.detail})"
    )
