"""Synthetic UNIT feature cases only; no actual source or model admission."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from market_predictor.research import news_decision_features as n

_CUT = datetime(2024, 1, 10, 22, tzinfo=UTC)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _technical(value: float | None) -> n.TechnicalValue:
    return n.TechnicalValue(value, _CUT - timedelta(hours=1) if value is not None else None, "observed")


def _decision(identity: str = "d", security: str = "A", **changes: Any) -> n.NewsDecision:
    return replace(n.NewsDecision(identity, security, _CUT, _technical(-2.0), _technical(0.25), _technical(1.5)), **changes)


def _version(
    identity: str = "v",
    *,
    story: str = "story",
    hours: float = 2,
    cues: tuple[n.DecisionCue, ...] = (n.DecisionCue("earnings", "up", "announced"),),
    score: float | None = None,
    **changes: Any,
) -> n.NewsVersion:
    clock = _CUT - timedelta(hours=hours)
    return replace(
        n.NewsVersion(
            identity,
            "alpaca",
            story,
            _sha(identity),
            _sha("source"),
            clock - timedelta(hours=1),
            clock,
            clock,
            True,
            tuple(sorted({cue.category for cue in cues})),
            cues,
            False,
            score,
            clock if score is not None else None,
            "observed",
        ),
        **changes,
    )


def _link(version: n.NewsVersion, security: str = "A", **changes: Any) -> n.VerifiedCompanyLink:
    return replace(
        n.VerifiedCompanyLink(
            version.version_id,
            security,
            version.known_at_utc,
            version.known_at_utc,
            _sha("relation"),
            "observed",
        ),
        **changes,
    )


def _coverage(source: str, **changes: Any) -> n.SourceCoverage:
    return replace(
        n.SourceCoverage(
            "A",
            source,
            _CUT - timedelta(days=4),
            _CUT,
            _CUT,
            True,
            _sha("coverage"),
            "observed",
        ),
        **changes,
    )


def _run(
    versions: tuple[n.NewsVersion, ...] = (),
    links: tuple[n.VerifiedCompanyLink, ...] | None = None,
    decisions: tuple[n.NewsDecision, ...] = (_decision(),),
    **kwargs: Any,
) -> tuple[n.DecisionNewsFeatures, ...]:
    return n.aggregate_news_decisions(
        decisions=decisions,
        versions=versions,
        links=tuple(_link(version) for version in versions) if links is None else links,
        purpose=kwargs.pop("purpose", "research_proxy"),
        **kwargs,
    )


def test_exact_85_columns_and_mixed_technical_interactions() -> None:
    earnings = _version(
        cues=(n.DecisionCue("earnings", "up", "announced"), n.DecisionCue("guidance", "down", "announced")),
        score=0.8,
        sentiment_available_at_utc=_CUT - timedelta(minutes=30),
    )
    macro = _version(
        "macro", story="separate-story", cues=(n.DecisionCue("sector_market", "adverse", "announced"),), score=-0.6, truncated=True
    )
    (result,) = _run((earnings, macro))
    values = result.as_values()
    assert len(result.columns) == len(set(result.columns)) == len(result.values) == len(result.available_at_utc) == 85
    assert result.columns[:6] == (
        "news_direct_earnings_present_1d",
        "news_direct_earnings_present_3d",
        "news_direct_earnings_up_share_3d",
        "news_direct_earnings_down_share_3d",
        "news_direct_earnings_favorable_share_3d",
        "news_direct_earnings_adverse_share_3d",
    )
    assert values["news_direct_finbert_positive_mean_3d"] == pytest.approx(0.4)
    assert values["news_direct_finbert_negative_magnitude_mean_3d"] == pytest.approx(0.3)
    assert values["news_direct_finbert_score_coverage_3d"] == 1
    assert values["news_direct_mixed_direction_share_3d"] == 0.5
    assert values["news_direct_truncated_text_share_3d"] == 0.5
    assert values["news_direct_latest_age_hours_3d"] == 2
    assert values["news_earnings_up_times_guidance_down_3d"] == 1
    assert values["news_mixed_times_relative_pullback_3d"] == -1
    assert values["news_macro_mention_adverse_times_daily_return_3d"] == 0.25
    assert values["news_positive_sentiment_times_long_trend_3d"] == pytest.approx(0.6)
    assert result.as_clocks()["news_positive_sentiment_times_long_trend_3d"] == _CUT - timedelta(minutes=30)
    assert not any((result.source_admission, result.training_eligible, result.serving_eligible, result.promotion_eligible))


def test_delayed_sentiment_cannot_delay_cue_or_reveal_score() -> None:
    version = _version(score=0.9, sentiment_available_at_utc=_CUT + timedelta(hours=1))
    (result,) = _run((version,))
    values = result.as_values()
    assert values["news_direct_earnings_present_1d"] == 1
    assert values["news_direct_finbert_score_coverage_3d"] == 0
    assert values["news_direct_finbert_positive_mean_3d"] is None
    assert values["news_direct_finbert_negative_magnitude_mean_3d"] is None
    (changed,) = _run((replace(version, sentiment_score=-0.9),))
    assert result == changed


def test_future_revision_poison_cannot_change_an_earlier_decision() -> None:
    original = _version()
    future = _version("future", hours=-1, cues=(n.DecisionCue("guidance", "down", "rumor"),), score=-1.0)
    assert _run((original, future)) == _run((original,))


@pytest.mark.parametrize("unusable", [True, False])
def test_newer_unusable_or_unlinked_revision_shadows_old_company_text(unusable: bool) -> None:
    old = _version()
    new = _version("new", hours=1)
    if unusable:
        new = replace(new, usable=False, categories=(), cues=(), cue_available_at_utc=None)
    (result,) = _run((old, new), links=(_link(old),))
    assert result.selected_version_ids == ()
    assert result.suppressed_story_count == 1
    assert result.as_values()["news_direct_earnings_present_3d"] is None


def test_future_exact_copy_does_not_delay_an_earlier_independent_version() -> None:
    proof_clock = _CUT + timedelta(hours=1)
    original = _version(copy_proof_sha256=_sha("copy"), copy_proof_available_at_utc=proof_clock)
    future = replace(original, version_id="future-copy", known_at_utc=proof_clock, cue_available_at_utc=proof_clock)
    assert _run((original, future)) == _run((original,))


def test_known_copy_with_future_proof_preserves_independent_original() -> None:
    proof_clock = _CUT + timedelta(hours=1)
    original = _version(copy_proof_sha256=_sha("copy"), copy_proof_available_at_utc=proof_clock)
    second = replace(
        original, version_id="known-copy", known_at_utc=_CUT - timedelta(hours=1), cue_available_at_utc=_CUT - timedelta(hours=1)
    )
    assert _run((original, second)) == _run((original,))


def test_cross_copy_sentiment_waits_for_equivalence_proof_without_delaying_cues() -> None:
    proof_clock = _CUT + timedelta(hours=1)
    original = _version(copy_proof_sha256=_sha("copy"), copy_proof_available_at_utc=proof_clock)
    second = replace(
        original,
        version_id="scored-copy",
        known_at_utc=_CUT - timedelta(hours=1),
        cue_available_at_utc=_CUT - timedelta(hours=1),
        sentiment_score=0.8,
        sentiment_available_at_utc=_CUT - timedelta(hours=1),
    )
    assert _run((original, second)) == _run((original,))
    available_proof = _CUT - timedelta(minutes=30)
    original = replace(original, copy_proof_available_at_utc=available_proof)
    second = replace(second, copy_proof_available_at_utc=available_proof)
    (result,) = _run((original, second))
    assert result.as_values()["news_direct_finbert_positive_mean_3d"] == 0.8
    assert result.as_clocks()["news_direct_finbert_positive_mean_3d"] == available_proof
    assert result.as_clocks()["news_direct_earnings_present_1d"] == original.cue_available_at_utc


def test_distinct_equal_clock_revisions_are_ambiguous() -> None:
    first = _version("first")
    second = _version("second", cues=(n.DecisionCue("earnings", "down", "announced"),))
    (result,) = _run((first, second))
    assert result.suppressed_story_count == 1
    assert result.selected_version_ids == ()
    assert result.as_values()["news_direct_earnings_up_share_3d"] is None


def test_proven_query_copy_dedup_preserves_company_link_union() -> None:
    first = _version(copy_proof_sha256=_sha("copy"), copy_proof_available_at_utc=_CUT - timedelta(hours=3))
    second = replace(first, version_id="other-query")
    results = _run((first, second), links=(_link(first, "A"), _link(second, "B")), decisions=(_decision("a", "A"), _decision("b", "B")))
    assert results[0].values == results[1].values
    assert results[0].as_values()["news_direct_earnings_present_1d"] == 1
    assert results[0].selected_version_ids == ("v",)
    assert results[1].selected_version_ids == ("other-query",)
    both_links = _run((first, second), links=(_link(first), _link(second)))
    assert len(both_links[0].selected_version_ids) == 1


def test_unproved_duplicate_key_rejects_instead_of_silent_dedup() -> None:
    version = _version()
    with pytest.raises(ValueError, match="query-copy proof"):
        _run((version, replace(version, version_id="duplicate")))


def test_conflicting_proven_copy_payload_rejects() -> None:
    version = _version(copy_proof_sha256=_sha("copy"), copy_proof_available_at_utc=_CUT)
    conflicting = replace(version, version_id="different-query", cues=(n.DecisionCue("earnings", "down", "announced"),))
    with pytest.raises(ValueError, match="conflicting cue/content"):
        _run((version, conflicting))


def test_distinct_sec_exhibits_and_unproven_documents_are_not_collapsed() -> None:
    first = _version("exhibit-a", source_family="sec", story="accession/exhibit-a")
    second = _version("exhibit-b", source_family="sec", story="accession/exhibit-b", source_version_sha256=first.source_version_sha256)
    (result,) = _run((first, second))
    assert result.selected_version_ids == ("exhibit-a", "exhibit-b")


def test_repeated_cues_do_not_inflate_article_direction_or_status_shares() -> None:
    positive = _version(cues=(n.DecisionCue("earnings", "up", "announced"),) * 20)
    negative = _version("negative", story="other-story", cues=(n.DecisionCue("earnings", "down", "planned"),))
    (result,) = _run((positive, negative))
    values = result.as_values()
    assert values["news_direct_earnings_up_share_3d"] == values["news_direct_earnings_down_share_3d"] == 0.5
    assert values["news_direct_announced_share_3d"] == values["news_direct_planned_share_3d"] == 0.5


def test_negation_contributes_mention_and_status_but_no_direction_or_mixedness() -> None:
    version = _version(cues=(n.DecisionCue("earnings", "up", "negated"), n.DecisionCue("earnings", "down", "announced")))
    (result,) = _run((version,))
    values = result.as_values()
    assert values["news_direct_earnings_present_1d"] == 1
    assert values["news_direct_negated_share_3d"] == 1
    assert values["news_direct_earnings_up_share_3d"] == values["news_direct_mixed_direction_share_3d"] == 0


def test_calendar_windows_are_inclusive_and_use_effective_availability() -> None:
    one = _version("one", story="one", hours=24)
    three = _version("three", story="three", hours=72)
    outside = _version("outside", story="outside", hours=72.001)
    (result,) = _run((one, three, outside))
    assert result.selected_version_ids == ("one", "three")
    assert result.as_values()["news_direct_earnings_present_1d"] == 1
    assert result.as_values()["news_direct_latest_age_hours_3d"] == 24
    late_link = _link(outside, relation_available_at_utc=_CUT - timedelta(hours=1))
    (reanchored,) = _run((outside,), links=(late_link,))
    assert reanchored.as_values()["news_direct_latest_age_hours_3d"] == 1


def test_wrong_company_or_future_link_never_becomes_global_macro_broadcast() -> None:
    macro = _version(cues=(n.DecisionCue("sector_market", "adverse", "announced"),))
    (wrong,) = _run((macro,), links=(_link(macro, "B"),))
    (future,) = _run((macro,), links=(_link(macro, identity_available_at_utc=_CUT + timedelta(minutes=1)),))
    for result in (wrong, future):
        assert result.selected_version_ids == ()
        assert result.as_values()["news_direct_sector_market_present_3d"] is None


def test_known_empty_unknown_empty_and_undefined_ratios_differ() -> None:
    (unknown,) = _run()
    (known,) = _run(coverage=(_coverage("alpaca"), _coverage("sec")))
    (partial,) = _run(coverage=(_coverage("alpaca"),))
    key = "news_direct_earnings_present_3d"
    assert unknown.as_values()[key] is partial.as_values()[key] is None
    assert known.as_values()[key] == 0
    assert known.as_values()["news_direct_earnings_up_share_3d"] is None
    assert known.as_values()["news_direct_latest_age_hours_3d"] is None
    assert known.as_values()["news_direct_finbert_positive_mean_3d"] is None
    assert partial.as_values()["news_direct_source_coverage_known_alpaca_3d"] == 1
    assert partial.as_values()["news_direct_source_coverage_known_sec_3d"] == 0


def test_complete_collection_cannot_turn_unusable_latest_content_into_known_empty() -> None:
    old = _version()
    new = replace(_version("new", hours=1), usable=False, categories=(), cues=(), cue_available_at_utc=None)
    (result,) = _run((old, new), links=(_link(old),), coverage=(_coverage("alpaca"), _coverage("sec")))
    assert result.as_values()["news_direct_earnings_present_3d"] is None
    assert result.as_values()["news_direct_source_coverage_known_alpaca_3d"] == 1


def test_coverage_requires_full_interval_and_available_proof() -> None:
    intervals = (
        _coverage("alpaca", end_utc=_CUT - timedelta(days=1)),
        _coverage("alpaca", start_utc=_CUT - timedelta(days=1)),
        _coverage("sec", available_at_utc=_CUT + timedelta(seconds=1)),
    )
    (result,) = _run(coverage=intervals)
    assert result.as_values()["news_direct_source_coverage_known_alpaca_3d"] == 1
    assert result.as_values()["news_direct_source_coverage_known_sec_3d"] == 0
    gap = replace(intervals[1], start_utc=_CUT - timedelta(days=1) + timedelta(seconds=1))
    (result,) = _run(coverage=(intervals[0], gap))
    assert result.as_values()["news_direct_source_coverage_known_alpaca_3d"] == 0


def test_batch_single_decision_and_observed_live_use_identical_numerical_kernel() -> None:
    version = _version(score=0.4)
    decisions = (_decision("first"), _decision("second", decision_time_utc=_CUT + timedelta(hours=1)))
    batch = _run((version,), decisions=decisions)
    individual = tuple(_run((version,), decisions=(decision,))[0] for decision in decisions)
    assert batch == individual
    observed = _run((version,), decisions=decisions, purpose="observed_live")
    assert [(row.values, row.available_at_utc) for row in observed] == [(row.values, row.available_at_utc) for row in batch]


@pytest.mark.parametrize("proxy", ["version", "link", "coverage", "technical"])
def test_observed_live_rejects_proxy_components(proxy: str) -> None:
    version = _version(semantics="historical_proxy" if proxy == "version" else "observed")
    link = _link(version, semantics="historical_proxy" if proxy == "link" else "observed")
    coverage = _coverage("alpaca", semantics="historical_proxy" if proxy == "coverage" else "observed")
    decision = _decision()
    if proxy == "technical":
        decision = replace(decision, return_1d_xs_z=replace(decision.return_1d_xs_z, semantics="historical_proxy"))
    with pytest.raises(ValueError, match="rejects historical proxies"):
        _run((version,), links=(link,), decisions=(decision,), coverage=(coverage,), purpose="observed_live")
    assert _run((version,), links=(link,), decisions=(decision,), coverage=(coverage,))[0].purpose == "research_proxy"


def test_missing_parent_operand_stays_null_and_future_parent_value_rejects() -> None:
    version = _version(score=0.7)
    decision = _decision(dist_sma_200_xs_z=_technical(None))
    (result,) = _run((version,), decisions=(decision,))
    assert result.as_values()["news_direct_finbert_positive_mean_3d"] == 0.7
    assert result.as_values()["news_positive_sentiment_times_long_trend_3d"] is None
    with pytest.raises(ValueError, match="after the decision cutoff"):
        _decision(return_1d_xs_z=replace(_technical(0.3), available_at_utc=_CUT + timedelta(seconds=1)))


@pytest.mark.parametrize(
    "changes",
    [
        {"known_at_utc": datetime(2024, 1, 1)},
        {"sentiment_score": float("nan")},
        {"sentiment_score": 1.01, "sentiment_available_at_utc": _CUT},
        {"categories": ("unknown",)},
        {"source_version_sha256": "not-a-hash"},
        {"copy_proof_sha256": _sha("copy")},
    ],
)
def test_invalid_source_values_reject(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _version(**changes)


def test_bounds_and_duplicate_decisions_reject() -> None:
    version = _version()
    with pytest.raises(ValueError, match="exceed bounds"):
        _run((version, _version("other", story="other")), limits=n.AggregationLimits(max_versions=1))
    with pytest.raises(ValueError, match="unique identities"):
        _run(decisions=(_decision(), _decision()))
