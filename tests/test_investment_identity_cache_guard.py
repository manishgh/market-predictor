"""Every identity replay keeps its memory gate, including exact cache hits."""
from datetime import date
from typing import Any

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment import stock_identity as identity
from market_predictor.resources import MemoryBudgetError
from tests.test_investment_stock_identity import replay
from tests.test_investment_stock_identity import scope as scope


def observe(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    cleanups: list[bool] = []

    def check(stage: str, *, reclaim_unused_memory: bool = True) -> None:
        if stage == "stock identity replay":
            cleanups.append(reclaim_unused_memory)

    monkeypatch.setattr(identity, "guard", check)
    return cleanups


def test_exact_cache_hit_keeps_guard_and_replays_each_requested_session(scope, monkeypatch):
    cleanups = observe(monkeypatch)
    cache: dict[str, Any] = {}
    first = replay(scope, "2024-05-31", cache=cache)
    second = replay(scope, "2024-06-03", cache=cache)
    assert cleanups == [True, False]
    assert first.proven_sessions == (date(2024, 5, 31),)
    assert second.proven_sessions == ()
    assert second.missing_reasons == ("stock_identity_session_unproven:2024-06-03",)
    assert second.verified_at_utc >= first.verified_at_utc
    assert scope["calls"] == ["parse"]


@pytest.mark.parametrize("mode", ["no_cache", "empty_cache", "different_proof", "missing_root", "wrong_root"])
def test_only_same_root_and_exact_loaded_proof_skips_initial_cleanup(scope, monkeypatch, mode):
    cache: dict[str, Any] = {}
    replay(scope, "2024-05-31", cache=cache)
    cleanups = observe(monkeypatch)
    if mode == "no_cache":
        replay(scope, "2024-05-31")
    elif mode == "empty_cache":
        replay(scope, "2024-05-31", cache={})
    elif mode == "different_proof":
        scope["proof"] = scope["proof"].model_copy(update={
            "anchor": scope["proof"].anchor.model_copy(update={"unit_id": "different-anchor"})})
        replay(scope, "2024-05-31", cache=cache)
    elif mode == "missing_root":
        del cache["source_root"]
        replay(scope, "2024-05-31", cache=cache)
    else:
        cache["source_root"] = str(scope["root"] / "other")
        with pytest.raises(DataReadinessError, match="cannot cross source roots"):
            replay(scope, "2024-05-31", cache=cache)
    assert cleanups == [True]


def test_memory_refusal_on_cached_proof_precedes_pin_and_session_work(scope, monkeypatch):
    cache: dict[str, Any] = {}
    replay(scope, "2024-05-31", cache=cache)
    before = dict(cache)
    pins: dict[str, str] = {}
    checks = []

    def refuse(stage: str, *, reclaim_unused_memory: bool = True) -> None:
        checks.append((stage, reclaim_unused_memory))
        raise MemoryBudgetError("synthetic pressure refusal")

    monkeypatch.setattr(identity, "guard", refuse)
    monkeypatch.setattr(identity.xcals, "get_calendar", lambda *_: pytest.fail("session work before memory gate"))
    with pytest.raises(MemoryBudgetError, match="synthetic pressure refusal"):
        identity.replay_stock_identity(scope["root"], scope["proof"], sessions=(date(2024, 5, 31),),
            source_files=pins, interpretation_sha256="c" * 64, cache=cache)
    assert checks == [("stock identity replay", False)]
    assert not pins
    assert cache.keys() == before.keys()
    assert all(cache[key] is value for key, value in before.items())


def test_cached_proof_still_refuses_conflicting_source_pin(scope, monkeypatch):
    cache: dict[str, Any] = {}
    replay(scope, "2024-05-31", cache=cache)
    cleanups = observe(monkeypatch)
    name = next(iter(cache["source_files"]))
    with pytest.raises(DataReadinessError, match="cached source conflicts"):
        identity.replay_stock_identity(scope["root"], scope["proof"], sessions=(date(2024, 5, 31),),
            source_files={name: "0" * 64}, interpretation_sha256="c" * 64, cache=cache)
    assert cleanups == [False]
