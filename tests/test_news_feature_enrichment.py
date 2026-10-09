"""Synthetic UNIT stories only; no source QA, sentiment-model or training evidence."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from market_predictor.research.news_feature_enrichment import (
    EnrichmentLimits,
    NewsFeatureEnrichment,
    add_enrichment_sample,
    enrich_news_features,
)
from market_predictor.research.news_sample_buckets import NewsSampleBuckets

_PUBLICATION = datetime(2023, 12, 3, 12, tzinfo=UTC)


def _enrich(text: str, **kwargs: Any) -> NewsFeatureEnrichment:
    arguments: dict[str, Any] = {
        "document_ref": "unit://article/1",
        "version_ref": "unit://article/1/revision/2",
        "source": "unit_news",
        "published_at_utc": _PUBLICATION,
        "available_at_utc": _PUBLICATION + timedelta(minutes=3),
        "text": text,
        "attribution_scope": "company",
    }
    arguments.update(kwargs)
    return enrich_news_features(**arguments)


@pytest.mark.parametrize(
    "text, category, direction",
    [
        ("Acme reports revenue grew.", "earnings", "up"),
        ("Acme reports earnings fell.", "earnings", "down"),
        ("Acme raises guidance.", "guidance", "up"),
        ("Acme cuts guidance.", "guidance", "down"),
        ("Acme won a supply contract.", "contracts_partnerships", "favorable"),
        ("Acme lost its supply contract.", "contracts_partnerships", "adverse"),
        ("Acme completed an acquisition.", "corporate_transactions", "unclear"),
        ("Analyst upgraded Acme.", "analyst_rating_target", "favorable"),
        ("Analyst downgraded Acme.", "analyst_rating_target", "adverse"),
        ("Acme's clinical trial met its primary endpoint.", "products_clinical", "favorable"),
        ("Acme's clinical trial failed.", "products_clinical", "adverse"),
        ("The court dismissed the lawsuit against Acme.", "legal_regulatory", "favorable"),
        ("The regulator fined Acme.", "legal_regulatory", "adverse"),
        ("Acme raises its dividend.", "capital_returns_financing", "up"),
        ("Acme cuts its dividend.", "capital_returns_financing", "down"),
        ("Acme factory production resumed.", "management_operations_security", "favorable"),
        ("Acme reports a data breach.", "management_operations_security", "adverse"),
        ("Oil prices rose.", "sector_market", "up"),
        ("Oil prices fell.", "sector_market", "down"),
        ("War disrupts the industry.", "sector_market", "adverse"),
    ],
)
def test_source_cues_cover_broad_news_without_claiming_returns(text: str, category: str, direction: str) -> None:
    result = _enrich(text)
    assert any(cue.category == category and cue.business_direction == direction for cue in result.cues)
    assert result.method == "rule_cues"
    assert result.existing_sentiment_score is None
    assert result.existing_sentiment_label is None
    assert result.to_qa_reference().confidence == "unavailable"


@pytest.mark.parametrize(
    "text, status",
    [
        ("Acme announced a new contract.", "announced"),
        ("Acme reportedly won a contract.", "rumor"),
        ("Acme plans a partnership.", "planned"),
        ("Acme did not win the contract.", "negated"),
        ("Acme contract update.", "unclear"),
        ("Acme completed a merger.", "announced"),
    ],
)
def test_status_is_separate_from_category_and_business_direction(text: str, status: str) -> None:
    result = _enrich(text)
    assert result.cues
    assert {cue.status for cue in result.cues} == {status}
    if status == "negated":
        assert {cue.business_direction for cue in result.cues} == {"unclear"}
        assert result.rule_signal_state == "unavailable"


def test_positive_earnings_negative_guidance_and_war_remain_separate() -> None:
    result = _enrich("Acme reports earnings grew but cuts guidance. War hurts the sector.")
    assert set(result.categories) == {"earnings", "guidance", "sector_market"}
    assert {cue.business_direction for cue in result.cues if cue.category == "earnings"} == {"up"}
    assert {cue.business_direction for cue in result.cues if cue.category == "guidance"} == {"down"}
    assert result.rule_signal_state == result.to_qa_reference().sentiment == "mixed"


def test_analyst_outlook_is_not_issuer_guidance() -> None:
    result = _enrich("Analyst raises Acme price target on an improved outlook.")
    assert result.categories == ("analyst_rating_target",)
    assert all(cue.actor_role == "analyst" for cue in result.cues)
    assert result.attribution_scope == "company"


def test_evidence_uses_exact_unicode_source_offsets_and_small_quotes() -> None:
    text = "🧪 Acme won a supply contract; Acme cuts guidance.\nThe regulator fined Acme."
    result = _enrich(text)
    for cue in result.cues:
        assert text[cue.start : cue.end] == cue.quote
        assert len(cue.quote) <= EnrichmentLimits().max_quote_characters
        assert cue.quote
    assert result.inspected_text_characters == result.total_text_characters == len(text)
    assert not result.truncation_reasons


def test_unknown_and_empty_text_are_preserved_without_neutral_sentiment() -> None:
    for text in ("Welcome to our company website.", ""):
        result = _enrich(text)
        assert result.categories == ("other_unresolved",)
        assert result.cues == ()
        assert result.rule_signal_state == "unavailable"
        assert result.to_qa_reference().sentiment == "unavailable"


def test_delivery_date_and_missing_fiscal_period_do_not_postpone_news() -> None:
    result = _enrich("Acme won a contract for delivery in 2028.", as_of_utc=_PUBLICATION + timedelta(minutes=4))
    assert result.categories == ("contracts_partnerships",)
    assert result.published_at_utc == _PUBLICATION
    assert result.available_at_utc == _PUBLICATION + timedelta(minutes=3)
    assert _enrich("Acme cuts guidance.").categories == ("guidance",)


def test_revised_version_after_cutoff_is_rejected_even_if_original_was_published() -> None:
    with pytest.raises(ValueError, match="not yet available"):
        _enrich("Acme raises guidance.", available_at_utc=_PUBLICATION + timedelta(days=2), as_of_utc=_PUBLICATION + timedelta(days=1))
    accepted = _enrich("Acme raises guidance.", as_of_utc=_PUBLICATION + timedelta(minutes=3))
    assert accepted.version_ref.endswith("revision/2")


@pytest.mark.parametrize(
    "changes",
    [
        {"published_at_utc": datetime(2023, 12, 3)},
        {"available_at_utc": "2023-12-03T12:03:00Z"},
        {"available_at_utc": _PUBLICATION - timedelta(seconds=1)},
        {"as_of_utc": datetime(2023, 12, 4)},
        {"existing_sentiment_score": float("nan")},
        {"existing_sentiment_score": float("inf")},
        {"existing_sentiment_score": True},
        {"existing_sentiment_label": "bullish"},
        {"existing_sentiment_label": ""},
        {"attribution_scope": "guessed_company"},
        {"document_ref": ""},
    ],
)
def test_invalid_source_metadata_and_fake_scores_reject(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _enrich("Acme won a contract.", **changes)


def test_existing_actual_sentiment_is_preserved_without_overwriting_mixed_cues() -> None:
    result = _enrich("Acme reports earnings grew but cuts guidance.", existing_sentiment_score=0.81, existing_sentiment_label="positive")
    assert result.existing_sentiment_score == 0.81
    assert result.existing_sentiment_label == "positive"
    assert result.rule_signal_state == "mixed"
    assert result.to_qa_reference().sentiment == "mixed"
    no_cues = _enrich("Company update.", existing_sentiment_score=-0.2, existing_sentiment_label="negative")
    assert no_cues.to_qa_reference().sentiment == "negative"
    assert no_cues.rule_signal_state == "unavailable"


def test_sector_scope_stays_separate_from_company_identity() -> None:
    result = _enrich("Inflation rose.", attribution_scope="market")
    assert result.attribution_scope == "market"
    assert result.categories == ("sector_market",)


def test_bounds_report_uninspected_text_capped_cues_and_quote_windows() -> None:
    text = "Acme won a contract. " * 40
    result = _enrich(text, limits=EnrichmentLimits(max_text_characters=200, max_cues=2))
    assert len(result.cues) == 2
    assert set(result.truncation_reasons) == {"text_character_limit", "cue_limit"}
    assert result.inspected_text_characters < 200 < result.total_text_characters
    windowed = _enrich("Acme won a contract " + "company " * 60, limits=EnrichmentLimits(max_quote_characters=80))
    assert "quote_window_limit" in windowed.truncation_reasons
    assert all(len(cue.quote) <= 80 for cue in windowed.cues)


def test_qa_collection_is_a_side_branch_with_real_labels_and_no_corpus_filter() -> None:
    results = [
        _enrich("Acme won a contract.", document_ref=f"unit://article/{index}", version_ref=f"unit://version/{index}")
        for index in range(20)
    ]
    sampler = NewsSampleBuckets()
    for result in results:
        add_enrichment_sample(result, sampler)
    assert len(results) == 20
    snapshot = sampler.snapshot()
    assert snapshot.records_seen == 20
    assert len(snapshot.buckets[0].references) == 3
    assert snapshot.buckets[0].key.category == "contracts_partnerships"
    assert snapshot.buckets[0].key.confidence == "unavailable"
    reverse_sampler = NewsSampleBuckets()
    for result in reversed(results):
        add_enrichment_sample(result, reverse_sampler)
    assert sampler.snapshot() == reverse_sampler.snapshot()


@pytest.mark.parametrize("changes", [{"max_cues": 0}, {"max_text_characters": True}, {"max_quote_characters": 20}])
def test_limits_reject_invalid_bounds(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(EnrichmentLimits(), **changes)


@pytest.mark.parametrize(
    "text, category",
    [
        ("Earnings discussion.", "earnings"),
        ("Guidance discussion.", "guidance"),
        ("Partnership discussion.", "contracts_partnerships"),
        ("Merger discussion.", "corporate_transactions"),
        ("Price target discussion.", "analyst_rating_target"),
        ("Clinical trial discussion.", "products_clinical"),
        ("Regulatory discussion.", "legal_regulatory"),
        ("Financing discussion.", "capital_returns_financing"),
        ("CEO discussion.", "management_operations_security"),
        ("Sector discussion.", "sector_market"),
    ],
)
def test_topic_alone_does_not_establish_an_action_or_direction(text: str, category: str) -> None:
    result = _enrich(text)
    assert result.categories == (category,)
    assert all(cue.status == "unclear" and cue.business_direction == "unclear" for cue in result.cues)
    assert result.rule_signal_state == "unavailable"


def test_trailing_clause_whitespace_is_not_reported_as_truncation() -> None:
    result = _enrich("Acme won a contract   ;   Acme cuts guidance  .")
    assert not result.truncation_reasons
    assert result.rule_signal_state == "mixed"


@pytest.mark.parametrize("amount", ["$10", "$10.50"])
def test_amount_sentence_boundary_does_not_transfer_direction(amount: str) -> None:
    result = _enrich(f"Acme raises guidance to {amount}. Beta cuts its dividend.")
    assert {cue.business_direction for cue in result.cues if cue.category == "guidance"} == {"up"}
    assert {cue.business_direction for cue in result.cues if cue.category == "capital_returns_financing"} == {"down"}
    assert all(amount in cue.quote for cue in result.cues if cue.category == "guidance")


def test_quote_chunking_preserves_negation_from_the_same_clause() -> None:
    text = "Acme did not " + "really " * 10 + "win a contract."
    result = _enrich(text, limits=EnrichmentLimits(max_quote_characters=80))
    contracts = [cue for cue in result.cues if cue.category == "contracts_partnerships"]
    assert contracts
    assert {cue.status for cue in contracts} == {"negated"}
    assert {cue.business_direction for cue in contracts} == {"unclear"}
    assert result.rule_signal_state == "unavailable"
    assert "quote_window_limit" in result.truncation_reasons
    assert all(text[cue.start : cue.end] == cue.quote and len(cue.quote) <= 80 for cue in contracts)
