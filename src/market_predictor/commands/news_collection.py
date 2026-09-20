"""Collect or verify shared raw news; never train models or authorize trades."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from market_predictor.config import get_settings
from market_predictor.core.errors import MemoryBudgetError
from market_predictor.core.system_memory import system_memory_snapshot
from market_predictor.evidence.news_collection import collect_news_receipts, read_news_collection_plan
from market_predictor.evidence.news_exchange import NewsPageRequest, ValidatedNewsReceipt
from market_predictor.locking import LockTimeout
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.news_collection import alpaca_news_receipt_fetcher
from market_predictor.sources.news_collection_settings import load_news_collection_settings


def _memory_guard(maximum_percent: float) -> None:
    snapshot = system_memory_snapshot()
    if snapshot is None or snapshot.used_percent >= maximum_percent:
        raise MemoryBudgetError("Shared news collection requires measurable system memory below its configured percent limit")


def register_news_collection_commands(app: typer.Typer) -> None:
    @app.command("collect-shared-news")
    def shared_news_command(
        plan: Path = typer.Option(..., help="Canonical collection plan JSON; contains no credentials."),
        expected_plan_sha256: str = typer.Option(..., help="Independently saved plan SHA256."),
        config: Path = typer.Option(Path("configs/shared_news_collection.toml")),
        offline: bool = typer.Option(False, help="Verify saved receipts only; never contact Alpaca."),
    ) -> None:
        """Collect horizon-independent Alpaca receipts for swing and investment."""
        source: AlpacaSource | None = None
        try:
            deployment, root = load_news_collection_settings(config)
            request = read_news_collection_plan(plan, expected_plan_sha256=expected_plan_sha256)
            if request.owner != deployment.owner:
                raise ValueError("collection plan does not belong to the configured owner")
            _memory_guard(deployment.maximum_system_memory_percent)

            def fetch(page: NewsPageRequest) -> ValidatedNewsReceipt:
                nonlocal source
                _memory_guard(deployment.maximum_system_memory_percent)
                if source is None:
                    settings = get_settings()
                    if not settings.has_alpaca:
                        raise ValueError("Configured Alpaca credentials are required for online collection")
                    source = AlpacaSource(settings)
                return alpaca_news_receipt_fetcher(source, producer_revision=request.producer_revision)(page)

            report = collect_news_receipts(
                root=root, plan=request, expected_plan_sha256=expected_plan_sha256,
                fetch_page=None if offline else fetch,
            )
        except LockTimeout as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=75) from exc
        except (ValueError, OSError, RuntimeError, MemoryBudgetError) as exc:
            typer.echo(f"Shared news collection rejected: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        finally:
            if source is not None:
                source.client.session.close()
        typer.echo(json.dumps({"complete": report.complete, **asdict(report)}, default=str, sort_keys=True))
        if not report.complete:
            raise typer.Exit(code=2)
