from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.sources.spglobal.archive import (
    ArchiveCollectionConfig,
    collect_spglobal_archive,
)
from market_predictor.universe.sp500.index_change_events import extract_spglobal_events


def register_sp500_source_commands(app: typer.Typer, console: Console) -> None:
    @app.command("collect-sp500-official-source-archive")
    @serialized_heavy_job("collect-sp500-official-source-archive")
    def collect_sp500_official_source_archive(
        source_audit: Path = typer.Option(
            ...,
            help="Frozen universe audit containing the 83 canonical release URLs.",
        ),
        source_audit_sha256: str = typer.Option(
            ...,
            help="Expected SHA-256 of the frozen source audit.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New or resumable immutable official-source archive directory.",
        ),
        cutoff_date: str = typer.Option(
            ...,
            help="Inclusive official-release discovery cutoff (YYYY-MM-DD).",
        ),
        maximum_pages: int = typer.Option(20, min=1, max=100),
        workers: int = typer.Option(1, min=1, max=2),
        retries: int = typer.Option(3, min=1, max=10),
        retry_pause_seconds: float = typer.Option(1.0, min=0.0, max=120.0),
        maximum_units_this_run: int | None = typer.Option(None, min=1),
    ) -> None:
        """Collect exact official S&P release bytes with verified resume."""

        try:
            parsed_cutoff_date = date.fromisoformat(cutoff_date)
        except ValueError as exc:
            raise typer.BadParameter(
                "must use YYYY-MM-DD",
                param_hint="--cutoff-date",
            ) from exc

        result = collect_spglobal_archive(
            source_audit_path=source_audit,
            expected_source_audit_sha256=source_audit_sha256,
            output_directory=out_dir,
            config=ArchiveCollectionConfig(
                discovery_end=parsed_cutoff_date,
                maximum_pages=maximum_pages,
                workers=workers,
                retries=retries,
                retry_pause_seconds=retry_pause_seconds,
                maximum_units_this_run=maximum_units_this_run,
            ),
        )
        console.print(
            {
                key: result[key]
                for key in (
                    "status",
                    "stop_reason",
                    "discovery_complete",
                    "requested_releases",
                    "completed_releases",
                    "resumed_releases",
                    "network_units_this_run",
                )
            }
        )
        if result["status"] != "complete":
            raise typer.Exit(code=2)

    @app.command("extract-sp500-official-events")
    @serialized_heavy_job("extract-sp500-official-events")
    def extract_sp500_official_events(
        archive_dir: Path = typer.Option(
            ...,
            help="Verified immutable official S&P raw archive.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New immutable offline event-extraction directory.",
        ),
    ) -> None:
        """Extract and reconcile S&P membership events without network access."""

        result = extract_spglobal_events(
            archive_directory=archive_dir,
            output_directory=out_dir,
        )
        console.print(
            {
                key: result[key]
                for key in (
                    "status",
                    "release_count",
                    "parsed_release_count",
                    "no_effective_event_release_count",
                    "unresolved_release_count",
                    "assertion_count",
                    "event_count",
                    "duplicate_support_count",
                    "conflict_count",
                )
            }
        )
        if result["status"] != "complete":
            raise typer.Exit(code=2)
