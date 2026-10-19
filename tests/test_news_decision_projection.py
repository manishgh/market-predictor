"""Synthetic repeat-output resolution, with real canonical temporal selection."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

from market_predictor.research import news_decision_features as kernel
from market_predictor.research import news_decision_projection as owner
from market_predictor.research.news_sentiment_copy_evidence import SentimentCopyIdentity
from tests.test_news_decision_features import _CUT, _decision, _link, _sha, _version


def _copies(left: float = .7, right: float = .700001) -> tuple[kernel.NewsVersion, kernel.NewsVersion]:
    first = _version("a", score=left, copy_proof_sha256=_sha("copy-a"), copy_proof_available_at_utc=_CUT - timedelta(hours=2))
    return first, replace(first, version_id="b", sentiment_score=right, source_proof_sha256=_sha("source-b"),
                          copy_proof_sha256=_sha("copy-b"))


def _identity(version: kernel.NewsVersion, **changes: Any) -> SentimentCopyIdentity:
    assert version.sentiment_score is not None and version.sentiment_available_at_utc is not None
    return SentimentCopyIdentity(**{**dict(source_copy_id=version.version_id, score=version.sentiment_score,
        score_available_at_utc=version.sentiment_available_at_utc, source_version_sha256=version.source_version_sha256,
        input_sha256=_sha("input"), model="ProsusAI/finbert", model_revision="unit-revision", input_mode="title_summary",
        max_length=512, availability_policy="unit-observed", label="positive" if version.sentiment_score > 0 else "negative",
        binding_sha256=_sha(version.version_id), evidence_ref=f"unit-evidence:{version.version_id}",
        source_family=version.source_family, source_document_id=version.source_document_id), **changes})


def _project(versions: tuple[kernel.NewsVersion, ...], *, lookup: Any = _identity,
             decisions: tuple[kernel.NewsDecision, ...] = (_decision(),),
             links: tuple[kernel.VerifiedCompanyLink, ...] | None = None) -> tuple[owner.NewsProjectionResult, ...]:
    return owner.project_news_decisions(decisions=decisions, versions=versions,
        links=tuple(_link(version) for version in versions) if links is None else links,
        purpose="research_proxy", identity_lookup=lookup)


def test_same_clock_small_spread_selects_exact_original_with_stable_tie() -> None:
    a, b = _copies()
    (result,) = _project((b, a))
    assert result.features.as_values()["news_direct_finbert_positive_mean_3d"] == a.sentiment_score
    assert result.features.as_values()["news_direct_earnings_present_3d"] == 1
    assert result.sentiment_resolutions[0]["selected_original_copy_id"] == "a"
    assert result.sentiment_resolutions[0]["selected_original_score"] == a.sentiment_score
    assert result.sentiment_resolutions[0]["reason"] == "numerically_equivalent_original_score_selected"
    assert result == _project((a, b))[0]
    assert not any((result.features.source_admission, result.features.training_eligible,
                    result.features.serving_eligible, result.features.promotion_eligible))


@pytest.mark.parametrize("lookup,expected", [(None, "repeat_inference_identity_unavailable"),
    (lambda version: None, "repeat_inference_identity_unavailable"),
    (lambda version: _identity(version, input_sha256=_sha(version.version_id)), "repeat_inference_identity_or_label_differs"),
    (lambda version: _identity(version, model_revision=version.version_id), "repeat_inference_identity_or_label_differs"),
    (lambda version: _identity(version, input_mode=version.version_id), "repeat_inference_identity_or_label_differs"),
    (lambda version: _identity(version, max_length=256 if version.version_id == "a" else 512),
        "repeat_inference_identity_or_label_differs"),
    (lambda version: _identity(version, availability_policy=version.version_id), "repeat_inference_identity_or_label_differs")])
def test_unknown_or_different_inference_identity_keeps_cues_but_no_sentiment(lookup: Any, expected: str) -> None:
    (result,) = _project(_copies(), lookup=lookup)
    values = result.features.as_values()
    assert values["news_direct_finbert_positive_mean_3d"] is None
    assert values["news_direct_finbert_negative_magnitude_mean_3d"] is None
    assert values["news_direct_finbert_score_coverage_3d"] == 0
    assert values["news_direct_earnings_up_share_3d"] == 1
    assert result.sentiment_resolutions[0]["reason"] == expected


@pytest.mark.parametrize("scores,reason", [((.7, .700011), "repeat_inference_score_spread_exceeds_bound"),
    ((-.000001, .000001), "repeat_inference_identity_or_label_differs")])
def test_meaningful_or_opposite_polarity_disagreement_abstains(scores: tuple[float, float], reason: str) -> None:
    (result,) = _project(_copies(*scores))
    assert result.features.as_values()["news_direct_finbert_positive_mean_3d"] is None
    assert result.sentiment_resolutions[0]["reason"] == reason


@pytest.mark.parametrize("delta,accepted", [(1e-5, True), (1.0000001e-5, False)])
def test_fixed_absolute_boundary_has_no_relative_tolerance(delta: float, accepted: bool) -> None:
    # Direct reason boundary isolates the frozen comparison from float addition rounding.
    records = [{"score": .7, "identity": {"input_sha256": "i", "model": "m", "model_revision": "r", "input_mode": "t",
        "max_length": 512, "availability_policy": "p", "label": "positive"}}] * 2
    assert (owner._equivalence_reason(records, delta) == "numerically_equivalent_original_score_selected") is accepted


@pytest.mark.parametrize("field", ["score", "known", "proof"])
def test_later_copy_score_or_proof_cannot_change_earlier_features_or_diagnostics(field: str) -> None:
    a, b = _copies()
    later = _CUT + timedelta(hours=1)
    if field == "score":
        b = replace(b, sentiment_available_at_utc=later)
    elif field == "known":
        b = replace(b, known_at_utc=later, cue_available_at_utc=later, sentiment_available_at_utc=later,
                    copy_proof_available_at_utc=later)
    else:
        b = replace(b, copy_proof_available_at_utc=later)
    def forbidden(version: kernel.NewsVersion) -> SentimentCopyIdentity:
        pytest.fail("future disagreement must not request model/input identity")
    assert _project((a, b), lookup=forbidden) == _project((a,), lookup=forbidden)
    (after,) = _project((a, b), decisions=(_decision("later", decision_time_utc=later),))
    assert len(after.sentiment_resolutions) == 1


def test_wrong_company_copy_does_not_supply_conflict_or_score() -> None:
    a, b = _copies()
    assert _project((a, b), links=(_link(a), _link(b, "B")), lookup=None) == _project((a,), lookup=None)


def test_expired_window_keeps_revision_history_but_does_not_emit_score_diagnostics() -> None:
    a, b = _copies()
    def forbidden(version: kernel.NewsVersion) -> SentimentCopyIdentity:
        pytest.fail("expired source text is not a current sentiment input")
    (result,) = _project((a, b), decisions=(_decision("late", decision_time_utc=_CUT + timedelta(days=4)),), lookup=forbidden)
    assert result.features.selected_version_ids == () and result.sentiment_resolutions == ()


def test_missing_representative_score_uses_canonical_merged_proof_clock() -> None:
    a, b = _copies()
    missing = replace(a, version_id="0", sentiment_score=None, sentiment_available_at_utc=None)
    a = replace(a, copy_proof_available_at_utc=_CUT - timedelta(minutes=30))
    b = replace(b, copy_proof_available_at_utc=_CUT - timedelta(minutes=20))
    (result,) = _project((missing, a, b))
    assert result.features.selected_version_ids == ("0",)
    assert result.features.as_values()["news_direct_finbert_positive_mean_3d"] == a.sentiment_score
    assert result.features.as_clocks()["news_direct_finbert_positive_mean_3d"] == a.copy_proof_available_at_utc
    assert result.sentiment_resolutions[0]["selected_original_copy_id"] == "a"


def test_equal_scores_and_single_eligible_outputs_need_no_new_identity_gate() -> None:
    versions = _copies(.7, .7)
    def forbidden(version: kernel.NewsVersion) -> SentimentCopyIdentity:
        pytest.fail("equal outputs keep existing semantics")
    (result,) = _project(versions, lookup=forbidden)
    (expected,) = kernel.aggregate_news_decisions(decisions=(_decision(),), versions=versions,
        links=tuple(_link(version) for version in versions), purpose="research_proxy")
    assert result.features == expected and result.sentiment_resolutions == ()


def test_future_revision_shadows_old_conflicts_only_when_known_and_batch_matches_single() -> None:
    a, b = _copies()
    later = _CUT + timedelta(hours=1)
    newer = _version("newer", hours=-1)
    decisions = (_decision(), _decision("later", decision_time_utc=later))
    versions = (a, b, newer)
    batch = _project(versions, decisions=decisions)
    assert batch == tuple(_project(versions, decisions=(decision,))[0] for decision in decisions)
    assert len(batch[0].sentiment_resolutions) == 1
    assert batch[1].sentiment_resolutions == () and batch[1].features.selected_version_ids == ("newer",)


def test_false_identity_binding_rejected_not_treated_as_missing() -> None:
    with pytest.raises(ValueError, match="exact original version"):
        _project(_copies(), lookup=lambda version: _identity(version, source_copy_id="wrong"))


def test_lexical_conflict_and_unproven_duplicate_still_rejected() -> None:
    a, b = _copies()
    with pytest.raises(ValueError, match="cue/content"):
        _project((a, replace(b, truncated=True)))
    with pytest.raises(ValueError, match="exact query-copy proof"):
        _project((a, replace(b, copy_proof_sha256=None, copy_proof_available_at_utc=None)))


def test_live_proxy_rejection_is_not_bypassed() -> None:
    a, b = _copies()
    a = replace(a, semantics="historical_proxy")
    with pytest.raises(ValueError, match="historical proxies"):
        owner.project_news_decisions(decisions=(_decision(),), versions=(a, b), links=(_link(a), _link(b)),
                                    purpose="observed_live", identity_lookup=_identity)
