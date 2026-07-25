from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from market_predictor.config import get_settings
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.swing.market_history import collect_swing_daily_history
from market_predictor.symbols import PROVIDER_ALPACA, provider_symbol


def register_swing_collection_commands(app: typer.Typer, console: Any) -> None:
    @app.command("collect-swing-daily-history")
    def collect_swing_daily_history_command(
        memberships: Path = typer.Option(..., help="Point-in-time membership CSV or parquet with security IDs."),
        start_date: str = typer.Option(..., help="Inclusive first market date YYYY-MM-DD."),
        end_date: str = typer.Option(..., help="Inclusive frozen final market date YYYY-MM-DD."),
        out_dir: Path = typer.Option(..., help="Resumable collection directory; finalized manifests are immutable."),
        workers: int = typer.Option(4, min=1, max=4, help="Bounded per-symbol Alpaca workers."),
    ) -> None:
        """Collect hash-audited Alpaca SIP daily bars for historical members and benchmarks."""

        settings = get_settings()
        if not settings.has_alpaca:
            raise typer.BadParameter("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are required")
        if settings.alpaca_stock_feed.strip().lower() != "sip":
            raise typer.BadParameter("ALPACA_STOCK_FEED must be sip")
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise typer.BadParameter("start-date and end-date must be YYYY-MM-DD") from exc

        def fetch(symbol: str, start_at: datetime, end_at: datetime) -> pd.DataFrame:
            provider_ticker = provider_symbol(symbol, PROVIDER_ALPACA)
            return AlpacaSource(settings).fetch_daily_bars(provider_ticker, start_at, end_at)

        result = collect_swing_daily_history(
            memberships_path=memberships,
            start_date=start,
            end_date=end,
            out_dir=out_dir,
            fetcher=fetch,
            price_feed=settings.alpaca_stock_feed,
            workers=workers,
        )
        console.print(
            {
                "status": result.status,
                "requested_symbols": result.requested_symbols,
                "observed_symbols": result.observed_symbols,
                "unavailable_symbols": list(result.unavailable_symbols),
                "failed_symbols": list(result.failed_symbols),
                "skipped_symbols": result.skipped_symbols,
                "manifest": str(result.manifest_path) if result.manifest_path else None,
                "status_path": str(result.status_path),
            }
        )
        if result.status == "incomplete":
            raise typer.Exit(code=2)
