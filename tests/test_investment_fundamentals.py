from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from market_predictor.investment.fundamentals import FilingFact, build_fundamental_batch, build_fundamental_snapshot

NOW = datetime(2025, 3, 1, tzinfo=UTC)


def fact(concept="revenue", start=date(2024, 1, 1), end=date(2024, 3, 31), value=100.0, **kwargs):
    return FilingFact(security_id="s", cik="1", accession="a", source_sha256="a" * 64,
        source_locator=f"{concept}:{start}:{end}", concept=concept, tag=concept,
        period_kind="duration" if start else "instant", start=start, end=end, value=value,
        published_at=NOW - timedelta(days=1), retrieved_at=NOW, **kwargs)


def snapshot(facts):
    return build_fundamental_snapshot(facts, security_id="s", cik="1", decision_time=NOW)


def year(concept="revenue", value=100.0):
    return [fact(concept, date(2024, m, 1), end, value) for m, end in
        [(1, date(2024, 3, 31)), (4, date(2024, 6, 30)), (7, date(2024, 9, 30)), (10, date(2024, 12, 31))]]


def test_future_poison_and_batch_parity():
    rows = year() + year("operating_income", -10.0)
    base = snapshot(rows)
    poison = rows[0].model_copy(update={"published_at": NOW + timedelta(days=1), "retrieved_at": NOW + timedelta(days=2), "value": 999.0})
    assert snapshot([*rows, poison]) == base
    assert build_fundamental_batch(rows, [("s", "1", NOW)]) == (base,)
    assert base.metrics["operating_margin_ttm"] == -0.1
    assert base.metrics["cash_conversion_ttm"] is None
    assert not base.training_allowed and base.research_only


def test_missing_quarter_is_not_zero():
    assert snapshot(year()[:-1]).metrics["operating_margin_ttm"] is None


def test_ytd_same_accession_normalization():
    quarters = year()
    ytd = [quarters[0]] + [fact(start=date(2024, 1, 1), end=q.end, value=float(100 * (i + 1)))
        for i, q in enumerate(quarters) if i]
    assert snapshot(ytd + year("operating_income", 10.0)).metrics["operating_margin_ttm"] == 0.1
    amended = ytd[-1].model_copy(update={"accession": "amended", "published_at": NOW, "value": 999.0})
    assert snapshot([*ytd, amended, *year("operating_income", 10.0)]).metrics["operating_margin_ttm"] is None


def test_latest_version_within_period_and_old_comparative():
    rows = year() + year("operating_income", 10.0)
    amended = rows[0].model_copy(update={"accession": "amended", "published_at": NOW, "value": 200.0})
    assert snapshot([*rows, amended]).metrics["operating_margin_ttm"] == 40 / 500
    old = fact(start=date(2020, 1, 1), end=date(2020, 3, 31), value=9999.0).model_copy(update={"published_at": NOW})
    assert snapshot([*rows, old]).metrics == snapshot(rows).metrics


def test_observed_retrieval_and_invalid_contracts():
    row = fact(availability_basis="observed", first_observed_at=NOW)
    assert build_fundamental_snapshot([row], security_id="s", cik="1", decision_time=NOW - timedelta(hours=1)).consumed_fact_sha256 == ()
    with pytest.raises(ValidationError):
        fact(concept="capex", value=-1.0)
    with pytest.raises(ValidationError):
        fact(unit="EUR")
    with pytest.raises(ValidationError):
        fact(availability_basis="observed")
    with pytest.raises(ValueError):
        build_fundamental_snapshot([], security_id="s", cik="1", decision_time=NOW.replace(tzinfo=None))


def test_unqualified_total_debt_feature_is_deferred():
    assert "net_debt_to_assets" not in snapshot([]).metrics


def test_direct_and_derived_quarter_conflict_is_missing():
    rows = year() + year("operating_income", 10.0)
    conflicting_ytd = fact(start=date(2024, 1, 1), end=date(2024, 6, 30), value=999.0)
    assert snapshot([*rows, conflicting_ytd]).metrics["operating_margin_ttm"] is None


def test_different_fiscal_boundaries_cannot_mix_even_with_same_ttm_endpoint():
    rows = year()
    for start, end in [(date(2024, 1, 2), date(2024, 3, 30)),
                       (date(2024, 3, 31), date(2024, 6, 30)),
                       (date(2024, 7, 1), date(2024, 9, 30)),
                       (date(2024, 10, 1), date(2024, 12, 31))]:
        rows.append(fact("operating_income", start, end, 10.0))
    assert snapshot(rows).metrics["operating_margin_ttm"] is None


def test_new_unresolvable_ytd_prevents_fallback_to_stale_ttm():
    rows = year() + year("operating_income", 10.0)
    rows.append(fact(start=date(2025, 1, 1), end=date(2025, 2, 15), value=100.0))
    assert snapshot(rows).metrics["operating_margin_ttm"] is None


def test_cash_conversion_preserves_negative_cash_flow_and_missing_loss_denominator():
    rows = year() + year("operating_cash_flow", -10.0) + year("net_income", 5.0)
    assert snapshot(rows).metrics["cash_conversion_ttm"] == -2.0
    rows = year() + year("operating_cash_flow", -10.0) + year("net_income", -5.0)
    assert snapshot(rows).metrics["cash_conversion_ttm"] is None


def annual_ytd_rows():
    rows = []
    for concept, annual, current, prior in [("revenue", 400.0, 240.0, 200.0),
            ("operating_cash_flow", 80.0, 50.0, 40.0), ("capex", 20.0, 15.0, 10.0)]:
        rows.append(fact(concept, date(2023, 1, 1), date(2023, 12, 31), annual).model_copy(
            update={"form": "10-K", "accession": "annual", "published_at": NOW - timedelta(days=2)}))
        rows.append(fact(concept, date(2024, 1, 1), date(2024, 6, 30), current))
        rows.append(fact(concept, date(2023, 1, 1), date(2023, 6, 30), prior))
    return rows


def test_annual_plus_comparative_ytd_cashflow_and_capex_known():
    result = snapshot(annual_ytd_rows())
    assert result.metrics["fcf_margin_ttm"] == (90 - 25) / 440
    assert result.metrics["capex_to_revenue_ttm"] == 25 / 440


def test_prior_ytd_wrong_anniversary_and_future_annual_fail_closed():
    rows = annual_ytd_rows()
    misplaced = [r.model_copy(update={"end": date(2023, 6, 29)})
        if r.concept == "operating_cash_flow" and r.end == date(2023, 6, 30) else r for r in rows]
    assert snapshot(misplaced).metrics["fcf_margin_ttm"] is None
    future = [r.model_copy(update={"published_at": NOW + timedelta(days=1), "retrieved_at": NOW + timedelta(days=2)})
        if r.form == "10-K" else r for r in rows]
    assert snapshot(future).metrics["fcf_margin_ttm"] is None


def test_direct_annual_requires_no_quarter_membership():
    rows = [fact(concept, date(2024, 1, 1), date(2024, 12, 31), value).model_copy(update={"form": "10-K"})
        for concept, value in [("revenue", 100.0), ("operating_cash_flow", 20.0), ("capex", 5.0)]]
    assert snapshot(rows).metrics["fcf_margin_ttm"] == 0.15


def test_ytd_comparative_from_unrelated_filing_is_not_subtracted():
    rows = [r.model_copy(update={"accession": "unrelated"}) if r.end == date(2023, 6, 30) else r for r in annual_ytd_rows()]
    assert snapshot(rows).metrics["fcf_margin_ttm"] is None


def test_unchanged_republication_preserves_historical_comparative_pair_and_binds_both_versions():
    rows = annual_ytd_rows()
    earlier = next(r for r in rows if r.concept == "operating_cash_flow" and r.end == date(2023, 6, 30))
    republished = earlier.model_copy(update={"accession": "later", "published_at": NOW})
    annual = next(r for r in rows if r.concept == "operating_cash_flow" and r.form == "10-K")
    later_annual = annual.model_copy(update={"accession": "annual_later", "published_at": NOW})
    result = snapshot([*rows, republished, later_annual])
    assert result.metrics["fcf_margin_ttm"] == 65 / 440
    assert {earlier.sha256(), republished.sha256(), annual.sha256(), later_annual.sha256()} <= set(result.consumed_fact_sha256)
    changed = republished.model_copy(update={"value": 41.0})
    assert snapshot([*rows, changed]).metrics["fcf_margin_ttm"] is None


def test_future_or_late_observed_history_cannot_restore_comparative_pair():
    rows = annual_ytd_rows()
    earlier = next(r for r in rows if r.concept == "operating_cash_flow" and r.end == date(2023, 6, 30))
    republished = earlier.model_copy(update={"accession": "later", "published_at": NOW})
    other = [r for r in rows if r is not earlier]
    late = earlier.model_copy(update={"availability_basis": "observed", "first_observed_at": NOW + timedelta(days=1),
        "retrieved_at": NOW + timedelta(days=2)})
    assert snapshot([*other, republished, late]).metrics["fcf_margin_ttm"] is None
    future = earlier.model_copy(update={"published_at": NOW + timedelta(days=1), "retrieved_at": NOW + timedelta(days=2)})
    assert snapshot([*other, republished, future]).metrics["fcf_margin_ttm"] is None
