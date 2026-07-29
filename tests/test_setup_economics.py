"""Poison tests for the ER3 deterministic setup-economics admission harness.

Every synthetic population below reproduces a real failure mode. The frames are
built in the test; no corpus is read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.edge_rebuild.setup_economics import (
    ADMISSION_SCOPES,
    UNSEEN_TICKER_SCOPE,
    WALK_FORWARD_SCOPE,
    SetupEconomicsConfig,
    SetupEconomicsReport,
    evaluate_setup_economics,
)
from market_predictor.v3.errors import DataReadinessError

PHASES = 2
SESSIONS_PER_PHASE = 60
SESSIONS = PHASES * SESSIONS_PER_PHASE
WALK_FORWARD_TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
UNSEEN_TICKERS = ("III", "JJJ", "KKK", "LLL", "MMM", "NNN", "OOO", "PPP")
STRATEGY_ID = "SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1"
TEST_CONFIG = SetupEconomicsConfig(required_phases=PHASES)


@dataclass(frozen=True)
class ReturnSpec:
    """Deterministic per-row economics for one validation scope.

    ``row_dispersion`` and ``session_shock`` are exactly zero-mean over the frame,
    so the realized sample mean equals ``net_mean`` and the tests never depend on
    a random draw. ``session_shock`` moves every row of a session together, which
    is what makes the session-block bootstrap widen while the mean stays put.
    """

    net_mean: float
    cost: float = 0.0005
    spy_drag: float = 0.0005
    sector_drag: float = 0.0005
    row_dispersion: float = 0.0
    session_shock: float = 0.0
    ticker_bonus: Mapping[str, float] = field(default_factory=dict)


def _scope_frame(
    spec: ReturnSpec,
    *,
    scope: str,
    tickers: Sequence[str],
) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-02")
    records: list[dict[str, object]] = []
    for session_index in range(SESSIONS):
        session_date = start + pd.Timedelta(days=session_index)
        shock = spec.session_shock * (
            1.0 if (session_index // PHASES) % 2 == 0 else -1.0
        )
        for ticker_index, ticker in enumerate(tickers):
            net = (
                spec.net_mean
                + spec.ticker_bonus.get(ticker, 0.0)
                + shock
                + spec.row_dispersion
                * math.sin(2 * math.pi * (ticker_index + session_index) / len(tickers))
            )
            records.append(
                {
                    "security_id": f"SEC-{ticker}",
                    "ticker": ticker,
                    "session": session_date.date().isoformat(),
                    "decision_time_utc": (
                        session_date.tz_localize("UTC") + pd.Timedelta(hours=14, minutes=30)
                    ),
                    "gross_return": net + spec.cost,
                    "cost": spec.cost,
                    "net_return": net,
                    "spy_excess_return": net - spec.spy_drag,
                    "sector_excess_return": net - spec.sector_drag,
                    "scope": scope,
                    "phase": session_index % PHASES,
                    "sector": "alpha" if ticker_index < 4 else "beta",
                    "market_regime": (
                        "risk_on" if (session_index + ticker_index) % 2 == 0 else "risk_off"
                    ),
                    "session_segment": (
                        "opening" if (ticker_index // 2) % 2 == 0 else "late"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _population(
    *,
    walk_forward: ReturnSpec,
    unseen_ticker: ReturnSpec | None = None,
) -> pd.DataFrame:
    return pd.concat(
        [
            _scope_frame(
                walk_forward,
                scope=WALK_FORWARD_SCOPE,
                tickers=WALK_FORWARD_TICKERS,
            ),
            _scope_frame(
                unseen_ticker if unseen_ticker is not None else walk_forward,
                scope=UNSEEN_TICKER_SCOPE,
                tickers=UNSEEN_TICKERS,
            ),
        ],
        ignore_index=True,
    )


def _admitting_spec() -> ReturnSpec:
    """A population that genuinely clears every frozen gate."""

    return ReturnSpec(net_mean=0.0030, cost=0.0005, row_dispersion=0.0040)


def _report(setups: pd.DataFrame) -> SetupEconomicsReport:
    return evaluate_setup_economics(setups, strategy_id=STRATEGY_ID, config=TEST_CONFIG)


# --------------------------------------------------------------------------- #
# 1. Positive gross, negative net after costs (the swing V2 shape).
# --------------------------------------------------------------------------- #
def test_costs_that_consume_a_positive_gross_edge_are_rejected() -> None:
    report = _report(_population(walk_forward=ReturnSpec(net_mean=-0.0004, cost=0.0020)))

    assert report.admitted is False
    assert "average_net_return" in report.failed_gates
    for scope in ADMISSION_SCOPES:
        assert report.gate(scope, "average_gross_return").passed is True
        assert report.gate(scope, "average_gross_return").measured > 0
        net_gate = report.gate(scope, "average_net_return")
        assert net_gate.passed is False
        assert net_gate.measured < 0
        assert net_gate.margin < 0


# --------------------------------------------------------------------------- #
# 2. Negative gross before costs (the intraday V2 shape).
# --------------------------------------------------------------------------- #
def test_negative_gross_before_costs_names_the_gross_gate() -> None:
    report = _report(_population(walk_forward=ReturnSpec(net_mean=-0.0015, cost=0.0010)))

    assert report.admitted is False
    assert "average_gross_return" in report.failed_gates
    for scope in ADMISSION_SCOPES:
        gross_gate = report.gate(scope, "average_gross_return")
        assert gross_gate.passed is False
        assert gross_gate.measured < 0
        assert any(
            reason.startswith(f"{scope}: average_gross_return")
            for reason in report.failure_reasons
        )
    # No estimator can rescue this, and the diagnosis must survive serialization.
    assert "average_gross_return" in report.as_dict()["failed_gates"]


# --------------------------------------------------------------------------- #
# 3. Walk-forward passes, unseen ticker does not. Both scopes are required.
# --------------------------------------------------------------------------- #
def test_unseen_ticker_failure_rejects_a_passing_walk_forward() -> None:
    report = _report(
        _population(
            walk_forward=_admitting_spec(),
            unseen_ticker=ReturnSpec(net_mean=-0.0002, cost=0.0005),
        )
    )

    assert report.scope(WALK_FORWARD_SCOPE).admitted is True
    assert report.scope(UNSEEN_TICKER_SCOPE).admitted is False
    assert report.admitted is False
    assert report.gate(UNSEEN_TICKER_SCOPE, "average_net_return").passed is False


# --------------------------------------------------------------------------- #
# 4. Profit concentrated in one ticker fails leave-one-out.
# --------------------------------------------------------------------------- #
def test_single_ticker_dependence_is_rejected_by_leave_one_out() -> None:
    dominated = ReturnSpec(net_mean=-0.0002, cost=0.0005)
    setups = pd.concat(
        [
            _scope_frame(
                ReturnSpec(
                    net_mean=dominated.net_mean,
                    cost=dominated.cost,
                    ticker_bonus={"AAA": 0.0200},
                ),
                scope=WALK_FORWARD_SCOPE,
                tickers=WALK_FORWARD_TICKERS,
            ),
            _scope_frame(
                ReturnSpec(
                    net_mean=dominated.net_mean,
                    cost=dominated.cost,
                    ticker_bonus={"III": 0.0200},
                ),
                scope=UNSEEN_TICKER_SCOPE,
                tickers=UNSEEN_TICKERS,
            ),
        ],
        ignore_index=True,
    )

    report = _report(setups)

    assert report.admitted is False
    for scope, dominant in (
        (WALK_FORWARD_SCOPE, "AAA"),
        (UNSEEN_TICKER_SCOPE, "III"),
    ):
        scope_report = report.scope(scope)
        # The pooled population itself looks perfectly healthy.
        assert scope_report.baseline.admitted is True
        assert report.gate(scope, "concentration:ticker").passed is False
        collapsed = [
            result
            for result in scope_report.leave_one_out
            if result.dimension == "ticker" and not result.report.admitted
        ]
        assert [result.excluded_value for result in collapsed] == [dominant]
        assert collapsed[0].largest_contributor is True
        assert collapsed[0].contribution_share > 0.9
        assert "average_net_return" in collapsed[0].report.failed_gates
    # Every leave-one-out result is reported, not only the failing one.
    tested = {
        (result.dimension, result.excluded_value)
        for result in report.scope(WALK_FORWARD_SCOPE).leave_one_out
    }
    assert len({dimension for dimension, _ in tested}) == 4
    assert len(tested) == len(WALK_FORWARD_TICKERS) + 2 + 2 + 2


# --------------------------------------------------------------------------- #
# 5. Positive mean, negative 95% session-block lower bound.
# --------------------------------------------------------------------------- #
def test_positive_mean_with_negative_session_block_lower_bound_is_rejected() -> None:
    report = _report(
        _population(
            walk_forward=ReturnSpec(
                net_mean=0.0006,
                cost=0.0003,
                spy_drag=0.0002,
                sector_drag=0.0002,
                session_shock=0.0050,
            )
        )
    )

    assert report.admitted is False
    for scope in ADMISSION_SCOPES:
        assert report.gate(scope, "average_net_return").passed is True
        assert report.gate(scope, "average_net_return").measured > 0
        bound_gate = report.gate(scope, "net_return_block_ci_low")
        assert bound_gate.passed is False
        assert bound_gate.measured < 0
        # The bound is wide because whole sessions move together, not because
        # there are few rows: every phase still carries 480 rows.
        phase = report.scope(scope).baseline.phase_economics[0]
        assert phase.rows == SESSIONS_PER_PHASE * len(WALK_FORWARD_TICKERS)
        assert phase.session_blocks == SESSIONS_PER_PHASE


def test_session_block_bound_is_wider_than_the_same_mean_without_session_shock() -> None:
    """Overlapping/co-moving rows must not be counted as independent evidence."""

    shocked = _report(
        _population(
            walk_forward=ReturnSpec(
                net_mean=0.0006,
                cost=0.0003,
                spy_drag=0.0002,
                sector_drag=0.0002,
                session_shock=0.0050,
            )
        )
    )
    steady = _report(
        _population(
            walk_forward=ReturnSpec(
                net_mean=0.0006,
                cost=0.0003,
                spy_drag=0.0002,
                sector_drag=0.0002,
                row_dispersion=0.0050,
            )
        )
    )

    assert (
        shocked.gate(WALK_FORWARD_SCOPE, "average_net_return").measured
        == pytest.approx(
            steady.gate(WALK_FORWARD_SCOPE, "average_net_return").measured
        )
    )
    assert shocked.gate(WALK_FORWARD_SCOPE, "net_return_block_ci_low").measured < 0
    assert steady.gate(WALK_FORWARD_SCOPE, "net_return_block_ci_low").measured > 0


# --------------------------------------------------------------------------- #
# 6. A population that genuinely clears every gate.
# --------------------------------------------------------------------------- #
def test_population_with_real_economics_is_admitted() -> None:
    report = _report(_population(walk_forward=_admitting_spec()))

    assert report.admitted is True, report.failure_reasons
    assert report.failure_reasons == ()
    for scope in ADMISSION_SCOPES:
        scope_report = report.scope(scope)
        assert scope_report.admitted is True
        assert scope_report.baseline.phases_present == tuple(range(PHASES))
        assert all(gate.passed for gate in scope_report.gates)
        assert all(result.report.admitted for result in scope_report.leave_one_out)
        # Every frozen gate is present, not merely non-failing.
        names = {gate.gate for gate in scope_report.gates}
        assert {
            "phase_coverage",
            "rows_per_phase",
            "session_blocks_per_phase",
            "average_gross_return",
            "average_net_return",
            "average_spy_excess_return",
            "average_sector_excess_return",
            "net_return_block_ci_low",
            "spy_excess_block_ci_low",
            "profit_factor",
            "maximum_drawdown",
            "stress_average_net_return",
            "stress_average_spy_excess_return",
            "concentration:ticker",
            "concentration:sector",
            "concentration:market_regime",
            "concentration:session_segment",
        } <= names
        assert report.gate(scope, "profit_factor").measured >= 1.05
        assert report.gate(scope, "maximum_drawdown").measured <= 0.20
        assert report.gate(scope, "stress_average_net_return").measured > 0
    assert len(report.config_sha256) == 64


# --------------------------------------------------------------------------- #
# 7. A malformed population fails closed with a clear error, never a crash.
# --------------------------------------------------------------------------- #
def test_missing_required_column_fails_closed() -> None:
    setups = _population(walk_forward=_admitting_spec()).drop(
        columns=["sector_excess_return", "market_regime"]
    )

    with pytest.raises(DataReadinessError) as failure:
        _report(setups)

    message = str(failure.value)
    assert "missing required columns" in message
    assert "market_regime" in message
    assert "sector_excess_return" in message


# --------------------------------------------------------------------------- #
# Fail-closed contract details.
# --------------------------------------------------------------------------- #
def test_absent_scope_fails_instead_of_being_skipped() -> None:
    setups = _population(walk_forward=_admitting_spec())
    setups = setups.loc[setups["scope"].eq(WALK_FORWARD_SCOPE)]

    report = _report(setups)

    assert report.admitted is False
    unseen = report.scope(UNSEEN_TICKER_SCOPE)
    assert unseen.admitted is False
    assert unseen.baseline.rows == 0
    assert all(not gate.passed for gate in unseen.baseline.gates)
    assert math.isnan(report.gate(UNSEEN_TICKER_SCOPE, "average_gross_return").measured)


def test_missing_phase_fails_the_coverage_gate() -> None:
    setups = _population(walk_forward=_admitting_spec())
    setups = setups.loc[
        ~(setups["scope"].eq(WALK_FORWARD_SCOPE) & setups["phase"].eq(1))
    ]

    report = _report(setups)

    assert report.admitted is False
    coverage = report.gate(WALK_FORWARD_SCOPE, "phase_coverage")
    assert coverage.passed is False
    assert coverage.measured == 1.0
    assert coverage.threshold == float(PHASES)


def test_double_counted_cost_violates_the_single_cost_identity() -> None:
    setups = _population(walk_forward=_admitting_spec())
    setups.loc[:, "net_return"] = setups["net_return"] - setups["cost"]

    with pytest.raises(DataReadinessError, match="single-cost identity"):
        _report(setups)


def test_repeated_decision_rows_are_rejected() -> None:
    setups = _population(walk_forward=_admitting_spec())
    setups = pd.concat([setups, setups.iloc[:1]], ignore_index=True)

    with pytest.raises(DataReadinessError, match="repeats a"):
        _report(setups)


def test_single_valued_concentration_dimension_cannot_pass() -> None:
    setups = _population(walk_forward=_admitting_spec())
    setups.loc[:, "market_regime"] = "risk_on"

    report = _report(setups)

    assert report.admitted is False
    for scope in ADMISSION_SCOPES:
        assert report.gate(scope, "concentration:market_regime:distinct_values").passed is False
        assert report.gate(scope, "concentration:market_regime").passed is False


def test_frozen_thresholds_cannot_be_weakened() -> None:
    with pytest.raises(ValidationError):
        SetupEconomicsConfig(minimum_profit_factor=1.0)
    with pytest.raises(ValidationError):
        SetupEconomicsConfig(maximum_drawdown=0.35)
    with pytest.raises(ValidationError):
        SetupEconomicsConfig(minimum_sessions_per_phase=10)
    with pytest.raises(ValidationError):
        SetupEconomicsConfig(concentration_dimensions=("ticker", "sector"))
    with pytest.raises(ValidationError):
        SetupEconomicsConfig(cost_stress_multiplier=1.25)
