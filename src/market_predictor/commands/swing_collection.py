from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer

from market_predictor.catalysts.issuer_events.alpaca_news_collection import (
    collect_alpaca_news_history,
)
from market_predictor.config import get_settings
from market_predictor.edge_rebuild.swing_history_collection import (
    AlpacaSwingDailyPageSource,
    SwingDailyPageSource,
    collect_swing_history_plan,
)
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.sources.alpaca import AlpacaNewsPage, AlpacaSource
from market_predictor.sources.provider_symbols import PROVIDER_ALPACA, provider_symbol


def register_swing_collection_commands(app: typer.Typer, console: Any) -> None:
    @app.command("collect-edge-rebuild-swing-history")
    @serialized_heavy_job("collect-edge-rebuild-swing-history")
    def collect_edge_rebuild_swing_history_command(
        plan_dir: Path = typer.Option(
            ...,
            help="Complete swing_history_acquisition_plan.v2 authority directory.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New or matching resumable exact-unit collection directory.",
        ),
        max_units: int | None = typer.Option(
            None,
            min=1,
            help="Optional resumable operational batch limit.",
        ),
    ) -> None:
        """Collect exact authority-bound swing daily units from Alpaca SIP."""

        settings = get_settings()
        if not settings.has_alpaca:
            raise typer.BadParameter(
                "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are required"
            )
        if settings.alpaca_stock_feed.strip().lower() != "sip":
            raise typer.BadParameter("ALPACA_STOCK_FEED must be sip")

        def source_factory() -> SwingDailyPageSource:
            return AlpacaSwingDailyPageSource(AlpacaSource(settings))

        result = collect_swing_history_plan(
            plan_directory=plan_dir,
            output_directory=out_dir,
            source_factory=source_factory,
            provider_symbol_for=lambda ticker: provider_symbol(
                ticker,
                PROVIDER_ALPACA,
            ),
            maximum_units_this_run=max_units,
        )
        console.print(
            {
                key: result[key]
                for key in (
                    "status",
                    "requested_units",
                    "terminal_units",
                    "observed_units",
                    "unavailable_units",
                    "failed_units",
                    "unattempted_units",
                    "resumed_units",
                    "stop_reason",
                )
            }
        )
        if result["status"] not in {"complete", "complete_with_unavailable"}:
            raise typer.Exit(code=2)

    @app.command("collect-alpaca-news-history")
    @serialized_heavy_job("collect-alpaca-news-history")
    def collect_alpaca_news_history_command(
        memberships: Path = typer.Option(
            ...,
            help="Hash-verified point-in-time membership artifact with security IDs.",
        ),
        start_date: str = typer.Option(
            ...,
            help="Inclusive first publication date YYYY-MM-DD.",
        ),
        end_date: str = typer.Option(
            ...,
            help="Inclusive frozen final publication date YYYY-MM-DD.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Resumable raw-page and research-event collection directory.",
        ),
        workers: int = typer.Option(
            2,
            min=1,
            max=4,
            help="Bounded independent network workers; no model work is started.",
        ),
        chunk_days: int = typer.Option(
            92,
            min=7,
            max=366,
            help="Half-open provider request chunk length.",
        ),
    ) -> None:
        """Collect publication-time-proxy Alpaca/Benzinga history immutably."""

        settings = get_settings()
        if not settings.has_alpaca:
            raise typer.BadParameter(
                "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are required"
            )
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise typer.BadParameter(
                "start-date and end-date must be YYYY-MM-DD"
            ) from exc
        source = AlpacaSource(settings)

        def fetch_page(
            symbol: str,
            start_at: datetime,
            end_at: datetime,
            page_token: str | None,
        ) -> AlpacaNewsPage:
            return source.fetch_news_page(
                symbol,
                start_at,
                end_at,
                page_token=page_token,
                include_content=True,
                limit=50,
            )

        result = collect_alpaca_news_history(
            memberships_path=memberships,
            start_date=start,
            end_date=end,
            out_dir=out_dir,
            fetch_page=fetch_page,
            provider_symbol_for=lambda ticker: provider_symbol(
                ticker,
                PROVIDER_ALPACA,
            ),
            workers=workers,
            chunk_days=chunk_days,
        )
        console.print(
            {
                "status": result.status,
                "requested_chunks": result.requested_chunks,
                "observed_chunks": result.observed_chunks,
                "empty_chunks": result.empty_chunks,
                "failed_chunks": list(result.failed_chunks),
                "skipped_chunks": result.skipped_chunks,
                "manifest": (
                    str(result.manifest_path)
                    if result.manifest_path is not None
                    else None
                ),
                "status_path": str(result.status_path),
                "production_ready": False,
                "availability_policy": "provider_publication_proxy",
            }
        )
        if result.status == "incomplete":
            raise typer.Exit(code=2)
