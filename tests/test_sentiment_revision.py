from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from market_predictor import sentiment

REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"


def _dependencies(monkeypatch: pytest.MonkeyPatch, actual: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def tokenizer(name: str, **kwargs: Any) -> object:
        calls.append({"type": "tokenizer", "name": name, **kwargs})
        return object()

    def model(name: str, **kwargs: Any) -> object:
        calls.append({"type": "model", "name": name, **kwargs})
        return SimpleNamespace(config=SimpleNamespace(_commit_hash=actual))

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=model), pipeline=lambda *args, **kwargs: object())
    monkeypatch.setattr(sentiment, "_load_optional_dependency",
        lambda name, **kwargs: fake_torch if name == "torch" else fake_transformers)
    return calls


def test_explicit_model_revision_is_requested_for_model_and_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _dependencies(monkeypatch, REVISION)
    scorer = sentiment.FinbertScorer(revision=REVISION)
    assert scorer.model_revision == REVISION
    assert len(calls) == 2
    assert all(call["revision"] == REVISION and call["local_files_only"] for call in calls)


def test_cache_with_wrong_model_revision_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _dependencies(monkeypatch, "a" * 40)
    with pytest.raises(RuntimeError, match="immutable revision"):
        sentiment.FinbertScorer(revision=REVISION)


@pytest.mark.parametrize("value", ["main", "latest", "", "a" * 39, "../snapshot"])
def test_mutable_or_invalid_revision_fails_before_loading(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid revision attempted dependency/model loading")

    monkeypatch.setattr(sentiment, "_load_optional_dependency", forbidden)
    with pytest.raises(ValueError, match="full immutable model commit"):
        sentiment.FinbertScorer(revision=value)


def test_unpinned_existing_call_still_reports_loaded_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _dependencies(monkeypatch, REVISION)
    assert sentiment.FinbertScorer().model_revision == REVISION
    assert all("revision" not in call for call in calls)
