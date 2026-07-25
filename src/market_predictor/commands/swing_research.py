from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console

from market_predictor.canonical.store import file_sha256
from market_predictor.commands.configuration import load_typed_config
from market_predictor.swing.inventory import (
    SwingResearchInventoryConfig,
    build_swing_research_inventory,
)
from market_predictor.swing.market_history_audit import audit_swing_daily_history


def register_swing_research_commands(app: typer.Typer, console: Console) -> None:
    @app.command("audit-swing-daily-history")
    def audit_swing_daily_history_command(
        memberships: Path = typer.Option(..., help="Point-in-time membership CSV or parquet."),
        collection_dir: Path = typer.Option(..., help="Finalized immutable daily-history collection."),
        out: Path = typer.Option(..., help="New interval-level coverage CSV; existing files are rejected."),
        summary_out: Path = typer.Option(..., help="New hash-bound coverage summary JSON."),
    ) -> None:
        """Replay daily-history hashes and classify every membership/session gap."""

        if out.exists():
            raise typer.BadParameter(f"daily-history audit output already exists: {out}")
        if summary_out.exists():
            raise typer.BadParameter(f"daily-history audit summary already exists: {summary_out}")
        report, summary = audit_swing_daily_history(
            memberships_path=memberships,
            collection_dir=collection_dir,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        csv_temporary = out.with_name(f".{out.name}.{uuid4().hex}.tmp")
        summary_temporary = summary_out.with_name(f".{summary_out.name}.{uuid4().hex}.tmp")
        try:
            report.to_csv(csv_temporary, index=False)
            summary["report"] = {
                "path": str(out.resolve()),
                "rows": len(report),
                "sha256": file_sha256(csv_temporary),
            }
            summary_temporary.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(csv_temporary, out)
            os.replace(summary_temporary, summary_out)
        except Exception:
            csv_temporary.unlink(missing_ok=True)
            summary_temporary.unlink(missing_ok=True)
            raise
        console.print(
            {
                "training_ready": summary["training_ready"],
                "coverage_rate": summary["coverage_rate"],
                "blocking_interval_count": summary["blocking_interval_count"],
                "excluded_interval_count": summary["excluded_interval_count"],
                "out": str(out),
                "summary_out": str(summary_out),
            }
        )

    @app.command("audit-swing-research-inventory")
    def audit_swing_research_inventory(
        raw_event_dir: Path = typer.Option(..., help="Directory containing per-ticker *_events.parquet files."),
        feature_dir: Path = typer.Option(..., help="Directory containing per-ticker daily 1D/5D feature parquets."),
        out: Path = typer.Option(..., help="New ticker-level inventory CSV; existing files are rejected."),
        summary_out: Path = typer.Option(..., help="New hash-bound aggregate inventory JSON; existing files are rejected."),
        memberships: Path | None = typer.Option(None, help="Optional point-in-time universe CSV or parquet."),
        source_collections: Path | None = typer.Option(None, help="Optional canonical source-collection CSV or parquet."),
        config_path: Path | None = typer.Option(None, "--config", help="Inventory threshold JSON or TOML config."),
    ) -> None:
        """Audit swing history, catalyst timing, provenance, and model eligibility."""

        if out.exists():
            raise typer.BadParameter(f"inventory output already exists: {out}")
        if summary_out.exists():
            raise typer.BadParameter(f"inventory summary already exists: {summary_out}")
        config = load_typed_config(config_path, SwingResearchInventoryConfig)
        try:
            report, summary = build_swing_research_inventory(
                raw_event_directory=raw_event_dir,
                feature_directory=feature_dir,
                memberships_path=memberships,
                source_collections_path=source_collections,
                config=config,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        csv_temporary = out.with_name(f".{out.name}.{uuid4().hex}.tmp")
        summary_temporary = summary_out.with_name(f".{summary_out.name}.{uuid4().hex}.tmp")
        try:
            report.to_csv(csv_temporary, index=False)
            summary["report"] = {
                "path": str(out.resolve()),
                "rows": len(report),
                "sha256": file_sha256(csv_temporary),
            }
            summary_temporary.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(csv_temporary, out)
            os.replace(summary_temporary, summary_out)
        except Exception:
            csv_temporary.unlink(missing_ok=True)
            summary_temporary.unlink(missing_ok=True)
            raise
        console.print(
            {
                "tickers": len(report),
                "model_eligibility": summary["model_eligibility"],
                "technical_eligibility": summary["technical_eligibility"],
                "catalyst_research_eligibility": summary["catalyst_research_eligibility"],
                "catalyst_promotion_eligibility": summary["catalyst_promotion_eligibility"],
                "out": str(out),
                "summary_out": str(summary_out),
            }
        )
