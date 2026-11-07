from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from market_predictor.investment.fundamentals import FilingFact, FilingFactRejection, build_fundamental_batch, build_fundamental_snapshot

NOW = datetime(2025, 3, 1, tzinfo=UTC)


def fact(concept="revenue", start=date(2024, 1, 1), end=date(2024, 3, 31), value=100.0, **kwargs):
    return FilingFact(
        security_id="s",
        cik="1",
        accession="a",
        source_sha256="a" * 64,
        source_locator=f"{concept}:{start}:{end}",
        concept=concept,
        tag=concept,
        period_kind="duration" if start else "instant",
        start=start,
        end=end,
        value=value,
        published_at=NOW - timedelta(days=1),
        retrieved_at=NOW,
        **kwargs,
    )


def snapshot(facts):
    return build_fundamental_snapshot(facts, security_id="s", cik="1", decision_time=NOW)


def year(concept="revenue", value=100.0):
    return [
        fact(concept, date(2024, m, 1), end, value)
        for m, end in [(1, date(2024, 3, 31)), (4, date(2024, 6, 30)), (7, date(2024, 9, 30)), (10, date(2024, 12, 31))]
    ]


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
    ytd = [quarters[0]] + [fact(start=date(2024, 1, 1), end=q.end, value=float(100 * (i + 1))) for i, q in enumerate(quarters) if i]
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
    for start, end in [
        (date(2024, 1, 2), date(2024, 3, 30)),
        (date(2024, 3, 31), date(2024, 6, 30)),
        (date(2024, 7, 1), date(2024, 9, 30)),
        (date(2024, 10, 1), date(2024, 12, 31)),
    ]:
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
    for concept, annual, current, prior in [
        ("revenue", 400.0, 240.0, 200.0),
        ("operating_cash_flow", 80.0, 50.0, 40.0),
        ("capex", 20.0, 15.0, 10.0),
    ]:
        rows.append(
            fact(concept, date(2023, 1, 1), date(2023, 12, 31), annual).model_copy(
                update={"form": "10-K", "accession": "annual", "published_at": NOW - timedelta(days=2)}
            )
        )
        rows.append(fact(concept, date(2024, 1, 1), date(2024, 6, 30), current))
        rows.append(fact(concept, date(2023, 1, 1), date(2023, 6, 30), prior))
    return rows


def test_annual_plus_comparative_ytd_cashflow_and_capex_known():
    result = snapshot(annual_ytd_rows())
    assert result.metrics["fcf_margin_ttm"] == (90 - 25) / 440
    assert result.metrics["capex_to_revenue_ttm"] == 25 / 440


def test_prior_ytd_wrong_anniversary_and_future_annual_fail_closed():
    rows = annual_ytd_rows()
    misplaced = [
        r.model_copy(update={"end": date(2023, 6, 29)}) if r.concept == "operating_cash_flow" and r.end == date(2023, 6, 30) else r
        for r in rows
    ]
    assert snapshot(misplaced).metrics["fcf_margin_ttm"] is None
    future = [
        r.model_copy(update={"published_at": NOW + timedelta(days=1), "retrieved_at": NOW + timedelta(days=2)}) if r.form == "10-K" else r
        for r in rows
    ]
    assert snapshot(future).metrics["fcf_margin_ttm"] is None


def test_direct_annual_requires_no_quarter_membership():
    rows = [
        fact(concept, date(2024, 1, 1), date(2024, 12, 31), value).model_copy(update={"form": "10-K"})
        for concept, value in [("revenue", 100.0), ("operating_cash_flow", 20.0), ("capex", 5.0)]
    ]
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
    late = earlier.model_copy(
        update={"availability_basis": "observed", "first_observed_at": NOW + timedelta(days=1), "retrieved_at": NOW + timedelta(days=2)}
    )
    assert snapshot([*other, republished, late]).metrics["fcf_margin_ttm"] is None
    future = earlier.model_copy(update={"published_at": NOW + timedelta(days=1), "retrieved_at": NOW + timedelta(days=2)})
    assert snapshot([*other, republished, future]).metrics["fcf_margin_ttm"] is None


def rejected(concept="revenue", start=date(2024, 10, 1), end=date(2024, 12, 31), **updates):
    values = dict(
        security_id="s",
        cik="1",
        accession="amendment",
        form="10-Q/A",
        source_sha256="b" * 64,
        source_locator="invalid-source-row",
        concept=concept,
        tag=concept,
        period_scope="exact",
        start=start,
        end=end,
        reason="invalid_fact",
        detail="invalid latest filing value",
        published_at=NOW,
        retrieved_at=NOW,
    )
    values.update(updates)
    return FilingFactRejection(**values)


def with_rejections(rows, issues):
    return build_fundamental_snapshot(rows, security_id="s", cik="1", decision_time=NOW, rejections=issues)


def test_rejected_latest_revenue_amendment_masks_period_without_stale_ttm_fallback():
    rows = year() + year("operating_income", 10.0)
    issue = rejected()
    result = with_rejections(rows, [issue])
    assert result.metrics["operating_margin_ttm"] is None
    assert result.consumed_rejection_sha256 == (issue.sha256(),)
    assert result.available_at == NOW
    assert build_fundamental_batch(rows, [("s", "1", NOW)], rejections=[issue]) == (result,)


def test_future_rejection_does_not_change_prior_snapshot_and_later_exact_valid_resolves():
    rows = year() + year("operating_income", 10.0)
    issue = rejected(published_at=NOW + timedelta(days=1), retrieved_at=NOW + timedelta(days=2))
    assert with_rejections(rows, [issue]) == snapshot(rows)
    issue = rejected(published_at=NOW - timedelta(hours=1))
    resolved = rows[3].model_copy(update={"accession": "later-valid", "published_at": NOW})
    result = with_rejections([*rows, resolved], [issue])
    assert result.metrics["operating_margin_ttm"] == 0.1
    assert result.consumed_rejection_sha256 == ()


def test_invalid_new_revenue_endpoint_cannot_fall_back_to_complete_older_year():
    issue = rejected(start=date(2025, 1, 1), end=date(2025, 2, 15))
    result = with_rejections(year() + year("operating_income", 10.0), [issue])
    assert result.metrics["operating_margin_ttm"] is None
    assert issue.sha256() in result.consumed_rejection_sha256


def test_unknown_period_tombstone_not_cleared_by_unrelated_later_valid_period():
    rows = year() + year("operating_income", 10.0)
    issue = rejected(period_scope="concept", start=None, end=None, published_at=NOW - timedelta(days=2))
    result = with_rejections(rows, [issue])
    assert all(value is None for value in result.metrics.values())
    assert issue.sha256() in result.consumed_rejection_sha256


def test_unrelated_balance_or_old_period_rejection_does_not_disable_flow_features():
    rows = year() + year("operating_income", 10.0)
    issues = [rejected("assets", None, None, period_scope="concept"), rejected(start=date(2010, 1, 1), end=date(2010, 3, 31))]
    assert with_rejections(rows, issues) == snapshot(rows)


def test_rejected_ytd_constituent_prevents_annual_bridge_reconstruction():
    rows = annual_ytd_rows()
    issue = rejected("operating_cash_flow", date(2023, 1, 1), date(2023, 6, 30))
    result = with_rejections(rows, [issue])
    assert result.metrics["fcf_margin_ttm"] is None
    assert result.metrics["capex_to_revenue_ttm"] == 25 / 440
    assert issue.sha256() in result.consumed_rejection_sha256


def test_late_observed_rejection_is_not_available_at_publication():
    issue = rejected(
        availability_basis="observed",
        first_observed_at=NOW + timedelta(days=1),
        retrieved_at=NOW + timedelta(days=1),
        published_at=NOW - timedelta(days=1),
    )
    rows = year() + year("operating_income", 10.0)
    assert with_rejections(rows, [issue]) == snapshot(rows)


def test_rejected_annual_cannot_revive_from_four_older_quarters():
    rows = year() + year("operating_income", 10.0)
    issue = rejected(start=date(2024, 1, 1), end=date(2024, 12, 31))
    result = with_rejections(rows, [issue])
    assert result.metrics["operating_margin_ttm"] is None
    assert issue.sha256() in result.consumed_rejection_sha256


def test_rejected_quarter_does_not_block_independent_annual_direct():
    rows = [
        fact(concept, date(2024, 1, 1), date(2024, 12, 31), value).model_copy(update={"form": "10-K"})
        for concept, value in [("revenue", 1000.0), ("operating_income", 100.0)]
    ]
    issue = rejected(start=date(2024, 4, 1), end=date(2024, 6, 30))
    result = with_rejections(rows, [issue])
    assert result.metrics["operating_margin_ttm"] == 0.1
    assert result.consumed_rejection_sha256 == ()
    assert result == snapshot(rows)


def test_rejected_intermediate_quarter_cannot_revive_from_ytd_subtraction():
    rows = year() + year("operating_income", 10.0)
    rows.append(fact(start=date(2024, 1, 1), end=date(2024, 6, 30), value=200.0))
    issue = rejected(start=date(2024, 4, 1), end=date(2024, 6, 30))
    result = with_rejections(rows, [issue])
    assert result.metrics["operating_margin_ttm"] is None
    assert issue.sha256() in result.consumed_rejection_sha256


def test_rejection_outside_ttm_does_not_blanket_mask_independent_annual_paths():
    rows = [
        fact(concept, date(y, 1, 1), date(y, 12, 31), value).model_copy(update={"form": "10-K"})
        for y in (2023, 2024)
        for concept, value in [("revenue", 1000.0), ("operating_income", 100.0)]
    ]
    issue = rejected(start=date(2023, 10, 1), end=date(2023, 12, 31))
    assert with_rejections(rows, [issue]) == snapshot(rows)
