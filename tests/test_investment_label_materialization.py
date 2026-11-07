"""Poison tests for dedicated horizon accounting and observed capture clocks."""
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.investment.label_materialization import (
    InvestmentSourceValidation,
    compile_investment_label,
)
from market_predictor.investment.label_sources import InvestmentObservations, InvestmentSourceReader
from market_predictor.swing.contracts.holding_accounting import (
    CorporateActionEvent,
    EvidenceReference,
    KnownMark,
    PositionLeg,
    SimulationReference,
)

CAPTURE = datetime(2026, 11, 7, tzinfo=UTC)


def reference(name: str) -> EvidenceReference:
    return EvidenceReference(reference=name, artifact_sha256="a" * 64, record_locator="retained record",
        interpretation_policy_sha256="b" * 64, retrieved_at=CAPTURE, available_at=CAPTURE)


class Reader(InvestmentSourceReader):
    def __init__(self) -> None:
        super().__init__(Path.cwd(), {"segments": []}, "b" * 64)

    def read(self, security_id: str, ticker: str, sessions: tuple[date, ...], *, benchmark: bool) -> InvestmentObservations:
        prices = pd.DataFrame({"open": [100.0] * len(sessions), "close": [110.0] * len(sessions)}, index=sessions)
        return InvestmentObservations(prices, {day: reference(f"raw/{ticker}") for day in sessions}, ticker, ())


def decision() -> dict[str, Any]:
    return {"decision_id": "test", "security_id": "issuer:test", "ticker": "TEST", "sector": "technology",
        "sector_benchmark": "XLK", "decision_session": "2024-05-28",
        "decision_time_utc": "2024-05-28T20:05:00Z"}


@pytest.mark.parametrize("horizon", [63, 252])
def test_horizon_extends_swing_boundary_without_liquidation(horizon: Any) -> None:
    row = compile_investment_label(decision=decision(), horizon=horizon, reader=Reader(),
        membership=InvestmentSourceValidation((reference("membership"),)),
        action_source_validation=lambda *_: InvestmentSourceValidation((reference("action-absence"),)))
    assert row["status"] == "source_validated"
    assert row["label_available_at_utc"] == "2026-11-07T00:00:00Z"
    assert row["economic_maturity_utc"] < row["label_available_at_utc"]
    assert len(row["specifications"]["stock"]["session_end_timestamps"]) == horizon
    assert all(not spec["events"] for spec in row["specifications"].values())
    assert row["target"]["holdings"][0]["net_return"] == pytest.approx(0.098)
    assert row["target"]["net_excess_vs_spy"] == pytest.approx(-0.002)
    assert row["training_ready"] is False


def test_unresolved_dividend_does_not_publish_price_only_return() -> None:
    row = compile_investment_label(decision=decision(), horizon=63, reader=Reader(),
        membership=InvestmentSourceValidation((reference("membership"),)),
        action_source_validation=lambda *_: InvestmentSourceValidation((reference("actions"),),
            ("dividend_payment_and_claim_valuation_unproven",)))
    assert row["status"] == "unresolved"
    assert len(row["components"]) == 4
    assert row["target"] is None and row["label_available_at_utc"] is None
    assert any("spy:dividend" in gap for gap in row["missing_reasons"])


def test_simulation_reference_cannot_become_observed_source_validation() -> None:
    simulated = SimulationReference(**{**reference("assumption").model_dump(), "available_at": None}, basis="research_assumption")
    with pytest.raises(ValueError, match="simulation provenance"):
        compile_investment_label(decision=decision(), horizon=63, reader=Reader(),
            membership=InvestmentSourceValidation((reference("membership"),)),
            action_source_validation=lambda *_: InvestmentSourceValidation((simulated,)))


def test_nominal_dividend_claim_is_retained_without_fake_payment() -> None:
    def validate_sources(identity: str, symbol: str, sessions: tuple[date, ...], position: str) -> InvestmentSourceValidation:
        calendar = xcals.get_calendar("XNYS")
        ex_day = sessions[2]
        claim = position + ":dividend"
        ref = reference("ordinary-usd-unconditional-entitlement-and-nominal-policy")
        event = CorporateActionEvent(event_id=claim, effective_at=calendar.session_open(ex_day.isoformat()).to_pydatetime(),
            order=0, evidence=(ref,), owned_position_id=position, owned_security_id=identity, treatment="distribution",
            fractional_treatment="proportional", legs=(PositionLeg(position_id=claim, kind="unpaid_proceeds",
                security_id=identity, units_per_owned_unit=2.0, currency="USD", face_value_per_unit=1.0),))
        marks = tuple(KnownMark(position_id=claim, mark_at=calendar.session_close(day.isoformat()).to_pydatetime(),
            value_per_unit=1.0, currency="USD", evidence=(ref,)) for day in sessions[2:])
        return InvestmentSourceValidation((ref,), events=(event,), non_share_marks=marks,
            valuation_basis="source_verified_entitlement_policy_valued_unpaid_receivable")

    row = compile_investment_label(decision=decision(), horizon=63, reader=Reader(),
        membership=InvestmentSourceValidation((reference("membership"),)), action_source_validation=validate_sources)
    assert row["status"] == "source_validated"
    assert row["target"]["holdings"][0]["net_return"] == pytest.approx(0.118)
    assert all([event["kind"] for event in spec["events"]] == ["corporate_action"]
        for spec in row["specifications"].values())
    assert row["components"][0]["valuation_basis"] == "source_verified_entitlement_policy_valued_unpaid_receivable"
