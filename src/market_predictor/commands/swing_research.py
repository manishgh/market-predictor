from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_canonical_bars,
    audit_universe_memberships,
)
from market_predictor.canonical.store import (
    file_sha256,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.commands.configuration import load_typed_config
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.swing.inventory import (
    SwingResearchInventoryConfig,
    build_swing_research_inventory,
)
from market_predictor.swing.market_history_audit import audit_swing_daily_history
from market_predictor.swing.panel_inputs import build_swing_market_panel_inputs


def register_swing_research_commands(app: typer.Typer, console: Console) -> None:
    @app.command("build-swing-market-panel-inputs")
    @serialized_heavy_job("build-swing-market-panel-inputs")
    def build_swing_market_panel_inputs_command(
        memberships: Path = typer.Option(..., help="Frozen point-in-time membership parquet."),
        collection_dir: Path = typer.Option(..., help="Finalized immutable SIP daily-history collection."),
        coverage_report: Path = typer.Option(..., help="Hash-bound interval coverage CSV."),
        coverage_summary: Path = typer.Option(..., help="Hash-bound coverage summary JSON."),
        stock_bars_out: Path = typer.Option(..., help="New PIT-filtered canonical stock bars parquet."),
        benchmark_bars_out: Path = typer.Option(..., help="New canonical benchmark bars parquet."),
        memberships_out: Path = typer.Option(..., help="New canonical research membership parquet."),
        audit_out: Path = typer.Option(..., help="New final bundle audit JSON."),
    ) -> None:
        """Assemble immutable leakage-safe market inputs from audited history."""

        outputs = (stock_bars_out, benchmark_bars_out, memberships_out, audit_out)
        occupied = [
            path
            for path in outputs
            if path.exists() or (path != audit_out and manifest_path_for(path).exists())
        ]
        if occupied:
            raise typer.BadParameter(f"market-panel output already exists: {occupied[0]}")
        stock_bars, benchmark_bars, canonical_memberships, audit = build_swing_market_panel_inputs(
            memberships_path=memberships,
            collection_dir=collection_dir,
            coverage_report_path=coverage_report,
            coverage_summary_path=coverage_summary,
        )
        artifact_inputs = {
            key: str(value)
            for key, value in audit["inputs"].items()
            if key.endswith("_sha256")
        }
        created: list[Path] = []
        try:
            stock_manifest = write_canonical_artifact(
                stock_bars,
                stock_bars_out,
                artifact_type="bars",
                audit=CanonicalAuditReport(
                    checks=audit_canonical_bars(stock_bars, require_sip=True)
                ),
                inputs=artifact_inputs,
                production_ready=True,
            )
            created.extend((stock_bars_out, manifest_path_for(stock_bars_out)))
            benchmark_manifest = write_canonical_artifact(
                benchmark_bars,
                benchmark_bars_out,
                artifact_type="bars",
                audit=CanonicalAuditReport(
                    checks=audit_canonical_bars(benchmark_bars, require_sip=True)
                ),
                inputs=artifact_inputs,
                production_ready=True,
            )
            created.extend((benchmark_bars_out, manifest_path_for(benchmark_bars_out)))
            membership_manifest = write_canonical_artifact(
                canonical_memberships,
                memberships_out,
                artifact_type="universe_memberships",
                audit=CanonicalAuditReport(
                    checks=audit_universe_memberships(
                        canonical_memberships,
                        require_observed=False,
                    )
                ),
                inputs=artifact_inputs,
                production_ready=False,
            )
            created.extend((memberships_out, manifest_path_for(memberships_out)))
            audit["outputs"] = {
                "stock_bars": _manifest_identity(stock_bars_out, stock_manifest),
                "benchmark_bars": _manifest_identity(benchmark_bars_out, benchmark_manifest),
                "memberships": _manifest_identity(memberships_out, membership_manifest),
            }
            audit_out.parent.mkdir(parents=True, exist_ok=True)
            temporary = audit_out.with_name(f".{audit_out.name}.{uuid4().hex}.tmp")
            temporary.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, audit_out)
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        console.print(
            {
                "training_ready": audit["training_ready"],
                "stock_rows": audit["stock_rows"],
                "stock_tickers": audit["stock_tickers"],
                "benchmark_rows": audit["benchmark_rows"],
                "membership_intervals": audit["membership_intervals"],
                "excluded_intervals": len(audit["excluded_intervals"]),
                "audit_out": str(audit_out),
            }
        )

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


def _manifest_identity(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    rows = manifest.get("rows")
    if not isinstance(rows, int):
        raise TypeError("canonical artifact manifest rows must be an integer")
    return {
        "path": str(path.resolve()),
        "sha256": str(manifest["artifact_sha256"]),
        "manifest_path": str(manifest_path_for(path).resolve()),
        "manifest_sha256": file_sha256(manifest_path_for(path)),
        "rows": rows,
        "production_ready": bool(manifest["production_ready"]),
    }
