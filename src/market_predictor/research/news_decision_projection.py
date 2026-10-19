"""As-of selection of original sentiment outputs for proven query copies.

The producer-pinned feature kernel remains unchanged. This consumer delegates
version, issuer and proof-clock eligibility to that kernel, then resolves only
unequal eligible repeat-inference outputs. It grants no source/model admission.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from market_predictor.research import news_decision_features as kernel

if TYPE_CHECKING:
    from market_predictor.research.news_sentiment_copy_evidence import SentimentCopyIdentity

ABSOLUTE_SCORE_SPREAD = 1e-5
POLICY = "same_input_model_label_original_asof_score_absolute_spread"
IdentityLookup = Callable[[kernel.NewsVersion], "SentimentCopyIdentity | None"]
_Key = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class NewsProjectionResult:
    features: kernel.DecisionNewsFeatures
    sentiment_resolutions: tuple[dict[str, Any], ...] = ()


def _key(version: kernel.NewsVersion) -> _Key:
    return version.source_family, version.source_document_id, version.source_version_sha256


def _without_score(version: kernel.NewsVersion) -> kernel.NewsVersion:
    return replace(version, sentiment_score=None, sentiment_available_at_utc=None)


def _evidence(version: kernel.NewsVersion, identity: SentimentCopyIdentity | None) -> dict[str, Any] | None:
    if identity is None:
        return None
    expected = {"source_copy_id": version.version_id, "source_family": version.source_family,
        "source_document_id": version.source_document_id, "source_version_sha256": version.source_version_sha256,
        "score": version.sentiment_score, "score_available_at_utc": version.sentiment_available_at_utc}
    if any(getattr(identity, name) != value for name, value in expected.items()):
        raise ValueError("sentiment-copy identity does not bind the exact original version/score/clock")
    return {name: value.isoformat() if isinstance(value, datetime) else value for name, value in asdict(identity).items()}


def _equivalence_reason(records: list[dict[str, Any]], spread: float) -> str:
    identities = [record["identity"] for record in records]
    if any(identity is None for identity in identities):
        return "repeat_inference_identity_unavailable"
    fields = ("input_sha256", "model", "model_revision", "input_mode", "max_length", "availability_policy", "label")
    signatures = {tuple(identity[name] for name in fields) for identity in identities}
    if len(signatures) != 1:
        return "repeat_inference_identity_or_label_differs"
    label = identities[0]["label"]
    scores = [record["score"] for record in records]
    if not (label == "positive" and all(score > 0 for score in scores)
            or label == "negative" and all(score < 0 for score in scores)
            or label == "neutral" and all(score == 0 for score in scores)):
        return "repeat_inference_polarity_differs"
    if spread > ABSOLUTE_SCORE_SPREAD:
        return "repeat_inference_score_spread_exceeds_bound"
    return "numerically_equivalent_original_score_selected"


def project_news_decisions(
    *, decisions: Sequence[kernel.NewsDecision], versions: Sequence[kernel.NewsVersion],
    links: Sequence[kernel.VerifiedCompanyLink], coverage: Sequence[kernel.SourceCoverage] = (),
    purpose: kernel.Purpose, limits: kernel.AggregationLimits = kernel.AggregationLimits(),
    identity_lookup: IdentityLookup | None = None,
) -> tuple[NewsProjectionResult, ...]:
    """Keep lexical features; resolve only decision-eligible unequal copy scores.

Equal scores and single eligible outputs keep the existing kernel semantics,
without a new source-identity gate. Diagnostics contain only disagreements that
the unchanged selector can actually consume at that decision.
    """
    # Validate all original lexical/proof contracts, without the old global
    # exact-score-equality assumption. No source values are changed in place.
    lexical = tuple(_without_score(version) for version in versions)
    kernel._prepare(decisions, lexical, links, coverage, purpose, limits)
    original_groups: dict[_Key, list[kernel.NewsVersion]] = {}
    for version in versions:
        original_groups.setdefault(_key(version), []).append(version)
    unequal = {key for key, copies in original_groups.items()
               if len({copy.sentiment_score for copy in copies if copy.sentiment_score is not None}) > 1}
    if not unequal:
        return tuple(NewsProjectionResult(result) for result in kernel.aggregate_news_decisions(
            decisions=decisions, versions=versions, links=links, coverage=coverage, purpose=purpose, limits=limits))
    stripped = tuple(_without_score(version) if _key(version) in unequal else version for version in versions)
    groups, link_index = kernel._prepare(decisions, stripped, links, coverage, purpose, limits)
    affected_stories = sorted({key[:2] for key in unequal})
    results = []
    for decision in decisions:
        (baseline,) = kernel.aggregate_news_decisions(decisions=(decision,), versions=stripped, links=links,
            coverage=coverage, purpose=purpose, limits=limits)
        observed = set(baseline.selected_version_ids)
        restore: dict[str, kernel.NewsVersion] = {}
        diagnostics: list[dict[str, Any]] = []
        for story_key in affected_stories:
            story = groups[story_key]
            selected, _ = kernel._select(decision, {story_key: story}, link_index)
            if not selected or selected[0].version.version_id not in observed:
                continue
            selected_key = _key(selected[0].version)
            if selected_key not in unequal:
                continue
            digest = selected_key[2]
            eligible: list[tuple[datetime, str, kernel.NewsVersion]] = []
            for copy in original_groups[selected_key]:
                if copy.sentiment_score is None:
                    continue
                # One original score at a time: the canonical selector alone
                # applies company, latest-version and merged-copy proof clocks.
                probe_story = {**story, digest: [copy if item.version_id == copy.version_id else item for item in story[digest]]}
                probe, _ = kernel._select(decision, {story_key: probe_story}, link_index)
                if probe and probe[0].sentiment_score is not None:
                    clock = probe[0].sentiment_available_at_utc
                    assert clock is not None
                    eligible.append((clock, copy.version_id, copy))
            if not eligible:
                continue
            eligible.sort(key=lambda item: (item[0], item[1]))
            winner: kernel.NewsVersion | None = eligible[0][2]
            scores = {item[2].sentiment_score for item in eligible}
            if len(scores) > 1:
                records: list[dict[str, Any]] = [{"copy_id": copy.version_id, "score": copy.sentiment_score,
                    "original_score_available_at_utc": copy.sentiment_available_at_utc.isoformat()
                        if copy.sentiment_available_at_utc is not None else None,
                    "eligible_score_available_at_utc": clock.isoformat(), "source_proof_sha256": copy.source_proof_sha256,
                    "copy_proof_sha256": copy.copy_proof_sha256,
                    "identity": _evidence(copy, identity_lookup(copy) if identity_lookup is not None else None)}
                    for clock, _, copy in eligible]
                numeric = [record["score"] for record in records]
                spread = max(numeric) - min(numeric)
                reason = _equivalence_reason(records, spread)
                if reason != "numerically_equivalent_original_score_selected":
                    winner = None
                diagnostics.append({"policy": POLICY, "absolute_spread_limit": ABSOLUTE_SCORE_SPREAD,
                    "relative_tolerance": 0.0, "decision_id": decision.decision_id, "security_id": decision.security_id,
                    "source_family": selected_key[0], "source_document_id": selected_key[1], "source_version_sha256": digest,
                    "score_spread": spread, "reason": reason, "eligible_original_outputs": records,
                    "selected_original_copy_id": winner.version_id if winner is not None else None,
                    "selected_original_score": winner.sentiment_score if winner is not None else None})
            if winner is not None:
                restore[winner.version_id] = winner
        features = baseline
        if restore:
            projected = tuple(restore.get(version.version_id, version) for version in stripped)
            (features,) = kernel.aggregate_news_decisions(decisions=(decision,), versions=projected, links=links,
                coverage=coverage, purpose=purpose, limits=limits)
        results.append(NewsProjectionResult(features, tuple(diagnostics)))
    return tuple(results)
