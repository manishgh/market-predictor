from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.commands import news_collection as commands
from market_predictor.core.system_memory import SystemMemory
from market_predictor.evidence.news_collection import (
    NewsCollectionPlan,
    NewsCollectionWindow,
    news_collection_plan_bytes,
    news_collection_plan_sha256,
)
from market_predictor.evidence.news_exchange import validate_news_receipt


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[typer.Typer, list[str], Mock, Mock]:
    fixture = json.loads((Path(__file__).parent / "fixtures/news_receipt_exchange.json").read_text(encoding="utf-8"))["cases"][0]
    receipt = validate_news_receipt(fixture["manifest_utf8"].encode(), fixture["payload_utf8"].encode())
    plan = NewsCollectionPlan(
        producer_revision="a" * 40, windows=(NewsCollectionWindow(window_id="MSFT", request=receipt.manifest.request),),
        max_pages_per_window=3,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(news_collection_plan_bytes(plan))
    config = tmp_path / "config.toml"
    config.write_text('owner="market_predictor"\ncoordination="single_host"\nstorage_root="receipts"\n'
                      'maximum_system_memory_percent=90.0\n', encoding="utf-8")
    app = typer.Typer()
    commands.register_news_collection_commands(app)
    source = Mock()
    factory = Mock(return_value=source)
    fetch = Mock(return_value=receipt)
    monkeypatch.setattr(commands, "AlpacaSource", factory)
    monkeypatch.setattr(commands, "get_settings", Mock(return_value=Mock(has_alpaca=True)))
    monkeypatch.setattr(commands, "alpaca_news_receipt_fetcher", Mock(return_value=fetch))
    monkeypatch.setattr(commands, "system_memory_snapshot", lambda: SystemMemory(100, 30))
    args = ["--plan", str(plan_path), "--expected-plan-sha256", news_collection_plan_sha256(plan), "--config", str(config)]
    return app, args, factory, fetch


def test_collect_then_offline_preserves_bytes_and_closes_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, args, factory, fetch = _setup(tmp_path, monkeypatch)
    online = CliRunner().invoke(app, args)
    assert online.exit_code == 0, online.output
    assert json.loads(online.output)["complete"] is True
    factory.return_value.client.session.close.assert_called_once()
    offline = CliRunner().invoke(app, [*args, "--offline"])
    assert offline.exit_code == 0, offline.output
    assert json.loads(offline.output) == json.loads(online.output)
    assert factory.call_count == fetch.call_count == 1


def test_offline_missing_collection_is_incomplete_without_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, args, factory, fetch = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(commands, "get_settings", Mock(side_effect=AssertionError("offline requested credentials")))
    result = CliRunner().invoke(app, [*args, "--offline"])
    assert result.exit_code == 2, result.output
    assert not json.loads(result.output)["complete"]
    factory.assert_not_called()
    fetch.assert_not_called()


@pytest.mark.parametrize("memory", [SystemMemory(100, 10), SystemMemory(100, 1), None])
def test_memory_at_limit_or_unmeasurable_stops_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memory: SystemMemory | None,
) -> None:
    app, args, factory, fetch = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(commands, "system_memory_snapshot", lambda: memory)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2 and "memory" in result.output
    factory.assert_not_called()
    fetch.assert_not_called()


def test_failed_plan_pin_does_not_access_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, args, factory, fetch = _setup(tmp_path, monkeypatch)
    args[3] = "0" * 64
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2 and "integrity pin" in result.output
    factory.assert_not_called()
    fetch.assert_not_called()
