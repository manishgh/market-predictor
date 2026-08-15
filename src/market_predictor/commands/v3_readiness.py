from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.v3.readiness import DevelopmentReadinessConfig, audit_development_readiness
from market_predictor.v3.spglobal_archive import (
    ArchiveCollectionConfig,
    collect_spglobal_archive,
)
from market_predictor.v3.spglobal_events import extract_spglobal_events


def register_v3_readiness_commands(app: typer.Typer, console: Console) -> None:
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

    @app.command("audit-v3-development-readiness")
    def audit_readiness(
        bars: Path = typer.Option(..., help="Ticker 5-minute parquet file or dataset directory."),
        universe: Path = typer.Option(..., help="Point-in-time universe CSV or parquet."),
        benchmark_dir: Path = typer.Option(..., help="Directory containing per-symbol benchmark parquet files."),
        out: Path = typer.Option(Path("data/reports/v3_development_readiness_latest.json"), help="Readiness report JSON."),
        minimum_tickers: int = typer.Option(300, min=2, help="Minimum distinct development symbols."),
        minimum_sessions: int = typer.Option(252, min=2, help="Minimum development sessions."),
        required_benchmarks: str = typer.Option(
            "SPY,QQQ,XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY",
            help="Comma-separated exact-timestamp benchmark symbols.",
        ),
    ) -> None:
        """Gate C8 on history, PIT universe, SIP provenance, and benchmarks."""
        symbols = tuple(dict.fromkeys(item.strip().upper() for item in required_benchmarks.split(",") if item.strip()))
        report = audit_development_readiness(
            bars_path=bars,
            universe_path=universe,
            benchmark_dir=benchmark_dir,
            config=DevelopmentReadinessConfig(
                minimum_tickers=minimum_tickers,
                minimum_sessions=minimum_sessions,
                required_benchmarks=symbols,
            ),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"Wrote V3 development readiness report to {out}")
        for check in report["checks"]:
            color = "green" if check["status"] == "pass" else "red"
            console.print(f"[{color}]{check['status'].upper()}[/{color}] {check['name']}: {check['observed']}")
        if not report["ready"]:
            raise typer.Exit(code=2)
