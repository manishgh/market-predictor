from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from market_predictor.v3.audits import build_data_audit
from market_predictor.v3.partitions import partition_development_shadow, write_shadow_partition


def register_v3_data_commands(app: typer.Typer, console: Console) -> None:
    @app.command("audit-v3-data")
    def audit_v3_data(
        bars: Path = typer.Option(..., help="Curated OHLCV CSV or parquet."),
        events: Path = typer.Option(..., help="Curated event CSV or parquet."),
        decisions: Path = typer.Option(..., help="Decision-row CSV or parquet with benchmark columns."),
        memberships: Path = typer.Option(..., help="Point-in-time universe membership CSV or parquet."),
        out: Path = typer.Option(Path("data/reports/v3_data_audit_latest.csv"), help="Audit report CSV."),
        interval_minutes: int = typer.Option(5, min=1, help="Expected intraday bar interval."),
        require_sip: bool = typer.Option(True, help="Fail volume provenance unless every bar is SIP."),
        strict: bool = typer.Option(True, help="Exit with an error when any required check fails."),
    ) -> None:
        """Audit V3 bars, events, universe membership, and benchmark coverage."""
        report = build_data_audit(
            bars=_read_frame(bars),
            events=_read_frame(events),
            decisions=_read_frame(decisions),
            memberships=_read_frame(memberships),
            interval=timedelta(minutes=interval_minutes),
            require_sip=require_sip,
        )
        audit_frame = report.to_frame()
        out.parent.mkdir(parents=True, exist_ok=True)
        audit_frame.to_csv(out, index=False)
        console.print(audit_frame)
        console.print(f"Wrote V3 data audit to {out}")
        if strict:
            report.raise_for_failure()

    @app.command("partition-v3-data")
    def partition_v3_data(
        dataset: Path = typer.Option(..., help="Frozen-schema decision dataset CSV or parquet."),
        development_out: Path = typer.Option(..., help="New development parquet; must not already exist."),
        shadow_out: Path = typer.Option(..., help="New immutable shadow parquet; must not already exist."),
    ) -> None:
        """Split development and immutable shadow rows at the frozen V3 cutoff."""
        development_manifest = development_out.with_suffix(".manifest.json")
        if development_out.exists() or development_manifest.exists():
            raise typer.BadParameter(f"Development partition already exists: {development_out}")
        frame = _read_frame(dataset)
        development, shadow = partition_development_shadow(frame)
        if development.empty or shadow.empty:
            raise typer.BadParameter("Input must contain rows on both sides of the frozen cutoff.")
        development_out.parent.mkdir(parents=True, exist_ok=True)
        development.to_parquet(development_out, index=False)
        try:
            manifest = write_shadow_partition(shadow, shadow_out)
        except Exception:
            development_out.unlink(missing_ok=True)
            raise
        console.print(f"Wrote {len(development)} development rows to {development_out}")
        console.print(f"Wrote {manifest['rows']} immutable shadow rows to {shadow_out}")


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Missing input: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter(f"Unsupported input format: {path.suffix}")
