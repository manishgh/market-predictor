from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from market_predictor.config import get_settings
from market_predictor.sources import AlpacaSource
from market_predictor.v3.readiness import DevelopmentReadinessConfig, audit_development_readiness
from market_predictor.v3.universe import (
    build_point_in_time_sp500_universe,
    collect_sp500_changes,
    symbol_changes_from_alpaca,
)


def register_v3_readiness_commands(app: typer.Typer, console: Console) -> None:
    @app.command("build-v3-sp500-point-in-time-universe")
    def build_sp500_point_in_time_universe(
        current_snapshot: Path = typer.Option(..., help="Frozen S&P 500 constituent anchor CSV or parquet."),
        start_date: str = typer.Option(..., help="First required membership date (YYYY-MM-DD)."),
        cutoff_date: str = typer.Option(..., help="Frozen final membership date (YYYY-MM-DD)."),
        out: Path = typer.Option(Path("data/universe/sp500_point_in_time.parquet"), help="Output membership parquet."),
        raw_dir: Path = typer.Option(Path("data/raw/index_membership/spglobal"), help="Official announcement evidence directory."),
        audit_out: Path = typer.Option(
            Path("data/reports/sp500_point_in_time_universe_audit.json"),
            help="Universe reconstruction audit JSON.",
        ),
        workers: int = typer.Option(6, min=1, max=16, help="Parallel announcement downloads."),
    ) -> None:
        """Build S&P membership intervals from official effective add/drop rows."""
        try:
            start = date.fromisoformat(start_date)
            cutoff = date.fromisoformat(cutoff_date)
        except ValueError as exc:
            raise typer.BadParameter("start-date and cutoff-date must use YYYY-MM-DD") from exc
        current = pd.read_parquet(current_snapshot) if current_snapshot.suffix.lower() == ".parquet" else pd.read_csv(current_snapshot)
        name_changes = AlpacaSource(get_settings()).fetch_name_changes(start, cutoff)
        name_changes_path = raw_dir / "alpaca_name_changes.parquet"
        name_changes_path.parent.mkdir(parents=True, exist_ok=True)
        name_changes.to_parquet(name_changes_path, index=False)
        changes, source_manifest = collect_sp500_changes(
            start_date=start,
            end_date=cutoff,
            raw_directory=raw_dir,
            workers=workers,
        )
        universe, audit = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=changes,
            symbol_changes=symbol_changes_from_alpaca(name_changes),
            start_date=start,
            cutoff_date=cutoff,
            anchor_source=str(current_snapshot),
        )
        audit["source_manifest"] = source_manifest
        out.parent.mkdir(parents=True, exist_ok=True)
        universe.to_parquet(out, index=False)
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"Wrote {len(universe):,} membership intervals for {universe['ticker'].nunique():,} tickers to {out}")
        console.print(f"Wrote point-in-time universe audit to {audit_out}")

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
