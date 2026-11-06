from datetime import UTC, datetime

import pytest

from market_predictor.investment.sec_facts import adapt_sec_company_facts

NOW = datetime(2025, 3, 1, tzinfo=UTC)


def payload(units=None):
    return {"cik": 1, "facts": {"us-gaap": {"Revenues": {"units": units or {"USD": [
        {"accn": "a", "start": "2024-01-01", "end": "2024-03-31", "val": 100, "form": "10-Q", "filed": "2024-05-01"}]}}}}}


def adapt(source, clocks=None, **kwargs):
    return adapt_sec_company_facts(source, security_id="s", cik="0001", source_sha256="a" * 64,
        retrieved_at=NOW, accepted_at_by_accession=clocks or {}, **kwargs)


def test_unknown_clock_skipped_not_invented_from_filed():
    result = adapt(payload())
    assert result.facts == ()
    assert "accepted_clock_unknown" in result.skipped[0]


def test_source_binding_and_explicit_observation():
    result = adapt(payload(), {"a": NOW}, source_policy="observed")
    assert result.facts[0].first_observed_at == NOW
    assert result.facts[0].source_sha256 == "a" * 64
    assert result.facts[0].cik == "0001"
    with pytest.raises(ValueError, match="CIK"):
        adapt({"cik": 2})


def test_units_and_alias_ambiguity():
    assert any("unsupported_unit" in reason for reason in adapt(payload({"EUR": []})).skipped)
    source = payload()
    source["facts"]["us-gaap"]["SalesRevenueNet"] = source["facts"]["us-gaap"]["Revenues"]
    assert len(adapt(source, {"a": NOW}).facts) == 1
    source["facts"]["us-gaap"]["SalesRevenueNet"] = {"units": {"USD": [
        {"accn": "a", "start": "2024-01-01", "end": "2024-03-31", "val": 101, "form": "10-Q"}]}}
    with pytest.raises(ValueError, match="ambiguous"):
        adapt(source, {"a": NOW})


def test_sec_cashflow_standard_annual_and_comparative_ytd_are_usable():
    from market_predictor.investment.fundamentals import build_fundamental_snapshot

    tags = {"Revenues": (400, 240, 200), "NetCashProvidedByUsedInOperatingActivities": (80, 50, 40),
            "PaymentsToAcquirePropertyPlantAndEquipment": (20, 15, 10)}
    source = {"cik": 1, "facts": {"us-gaap": {tag: {"units": {"USD": [
        {"accn": "annual", "start": "2023-01-01", "end": "2023-12-31", "val": values[0], "form": "10-K"},
        {"accn": "q2", "start": "2024-01-01", "end": "2024-06-30", "val": values[1], "form": "10-Q"},
        {"accn": "q2", "start": "2023-01-01", "end": "2023-06-30", "val": values[2], "form": "10-Q"}]}}
        for tag, values in tags.items()}}}
    decoded = adapt(source, {"annual": datetime(2024, 2, 1, tzinfo=UTC), "q2": datetime(2024, 8, 1, tzinfo=UTC)})
    result = build_fundamental_snapshot(decoded.facts, security_id="s", cik="0001", decision_time=NOW)
    assert result.metrics["fcf_margin_ttm"] == 65 / 440
    assert result.metrics["capex_to_revenue_ttm"] == 25 / 440
