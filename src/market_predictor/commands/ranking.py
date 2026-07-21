from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from market_predictor.global_context import build_sector_theme_monitor


def register_ranking_commands(app: typer.Typer, console: Console) -> None:
    @app.command("rank-sector-themes")
    def rank_sector_themes(
        dataset: Path = typer.Option(
            Path("data/features/swing/latest.parquet"),
            help="Canonical swing feature dataset with latest ticker rows.",
        ),
        universe: Path = typer.Option(
            Path("data/universe/sp500_current_latest.csv"),
            help="Universe CSV with ticker, sector, industry, and company columns.",
        ),
        model: Path = typer.Option(
            Path("models/swing/promoted/swing_5d.joblib"),
            help="Promoted canonical swing model artifact.",
        ),
        flashpoints: Path | None = typer.Option(
            None,
            help="Optional flashpoint CSV from score-flashpoints.",
        ),
        sector_out: Path = typer.Option(
            Path("data/reports/sector_theme_ranking_latest.csv"),
            help="Output sector/theme ranking CSV.",
        ),
        ticker_out: Path = typer.Option(
            Path("data/reports/sector_theme_ranking_tickers_latest.csv"),
            help="Output ticker-level ranking CSV.",
        ),
        allow_candidate_model: bool = typer.Option(False, help="Allow non-promoted model for research only."),
    ) -> None:
        """Rank sectors/themes using the promoted model plus global flashpoint context."""
        if not dataset.exists():
            raise typer.BadParameter(f"Missing dataset: {dataset}")
        if not universe.exists():
            raise typer.BadParameter(f"Missing universe: {universe}")
        if not model.exists():
            raise typer.BadParameter(f"Missing model: {model}")
        feature_frame = pd.read_parquet(dataset) if dataset.suffix.lower() == ".parquet" else pd.read_csv(dataset)
        universe_frame = pd.read_csv(universe)
        flashpoint_frame = None
        if flashpoints is not None:
            if not flashpoints.exists():
                raise typer.BadParameter(f"Missing flashpoint CSV: {flashpoints}")
            flashpoint_frame = pd.read_csv(flashpoints)
        sector_report, ticker_report = build_sector_theme_monitor(
            dataset=feature_frame,
            universe=universe_frame,
            model_path=model,
            flashpoints=flashpoint_frame,
            require_promoted=not allow_candidate_model,
        )
        sector_out.parent.mkdir(parents=True, exist_ok=True)
        ticker_out.parent.mkdir(parents=True, exist_ok=True)
        sector_report.to_csv(sector_out, index=False)
        ticker_keep = [
            column
            for column in [
                "ticker",
                "date",
                "monitor_theme",
                "monitor_signal",
                "monitor_score",
                "swing_model_probability",
                "global_net_impact",
                "global_positive_impact",
                "global_negative_impact",
                "volume_z20",
                "event_count_3d",
                "return_1d",
                "sector_return_1d",
                "rel_return_1d_vs_sector",
            ]
            if column in ticker_report.columns
        ]
        ticker_report[ticker_keep].to_csv(ticker_out, index=False)
        console.print(sector_report.head(30))
        console.print(ticker_report[ticker_keep].head(50))
        console.print(f"Wrote sector/theme ranking to {sector_out}")
        console.print(f"Wrote ticker ranking to {ticker_out}")
