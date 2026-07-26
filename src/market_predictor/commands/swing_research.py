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
from market_predictor.config import get_settings
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.sentiment import FinbertScorer
from market_predictor.swing.catalyst_lineage import build_catalyst_lineage
from market_predictor.swing.event_attribution_history import (
    attribute_alpaca_news_history,
)
from market_predictor.swing.inventory import (
    SwingResearchInventoryConfig,
    build_swing_research_inventory,
)
from market_predictor.swing.market_history_audit import audit_swing_daily_history
from market_predictor.swing.news_history_audit import audit_alpaca_news_history
from market_predictor.swing.panel_inputs import build_swing_market_panel_inputs
from market_predictor.swing.security_label_artifact import (
    build_security_label_artifact,
)
from market_predictor.swing.sentiment_history import score_alpaca_news_history


def register_swing_research_commands(app: typer.Typer, console: Console) -> None:
    @app.command("attribute-alpaca-news-history")
    @serialized_heavy_job("attribute-alpaca-news-history")
    def attribute_alpaca_news_history_command(
        collection_dir: Path = typer.Option(
            ...,
            help="Completed immutable Alpaca news collection.",
        ),
        collection_audit: Path = typer.Option(
            ...,
            help="Passed collection audit summary JSON.",
        ),
        business_labels: Path = typer.Option(
            ...,
            help="Canonical point-in-time business-tag assignments.",
        ),
        security_identities: Path = typer.Option(
            ...,
            help="Canonical identity/coverage artifact for all securities.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Resumable event-security relation directory.",
        ),
    ) -> None:
        """Attribute news through direct, exposure, and context channels."""

        def report_progress(payload: dict[str, object]) -> None:
            index = payload.get("index")
            total = payload.get("total")
            if not isinstance(index, int) or not isinstance(total, int):
                raise TypeError("attribution progress requires integer counters")
            if (
                index == 1
                or index == total
                or index % 25 == 0
                or payload.get("status") == "failed"
            ):
                console.print(payload)

        result = attribute_alpaca_news_history(
            collection_dir=collection_dir,
            collection_audit_path=collection_audit,
            business_labels_path=business_labels,
            security_identities_path=security_identities,
            out_dir=out_dir,
            progress=report_progress,
        )
        console.print(
            {
                "status": result["status"],
                "requested_chunks": result["requested_chunks"],
                "observed_chunks": result["observed_chunks"],
                "skipped_chunks": result["skipped_chunks"],
                "failed_chunks": result["failed_chunks"],
                "relation_rows": result["relation_rows"],
                "channel_counts": result["channel_counts"],
            }
        )
        if result["status"] != "complete":
            raise typer.Exit(code=2)

    @app.command("build-catalyst-lineage")
    @serialized_heavy_job("build-catalyst-lineage")
    def build_catalyst_lineage_command(
        collection_dir: Path = typer.Option(
            ...,
            help="Completed immutable Alpaca news collection.",
        ),
        collection_audit: Path = typer.Option(
            ...,
            help="Passed Alpaca news collection audit JSON.",
        ),
        attribution_dir: Path = typer.Option(
            ...,
            help="Completed event-attribution replay directory.",
        ),
        sentiment_dir: Path = typer.Option(
            ...,
            help="Completed FinBERT replay directory.",
        ),
        decisions: Path = typer.Option(
            ...,
            help="Hash-verified canonical decision artifact.",
        ),
        policy: Path = typer.Option(
            Path("configs/catalyst_lineage.toml"),
            help="Frozen catalyst-lineage policy.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Resumable catalyst-lineage artifact directory.",
        ),
    ) -> None:
        """Join catalyst evidence and replay exact decision assignments."""

        def report_progress(payload: dict[str, object]) -> None:
            index = payload.get("index")
            total = payload.get("total")
            if not isinstance(index, int) or not isinstance(total, int):
                raise TypeError("catalyst lineage progress requires integer counters")
            if (
                index == 1
                or index == total
                or index % 25 == 0
                or payload.get("status") == "failed"
            ):
                console.print(payload)

        result = build_catalyst_lineage(
            collection_dir=collection_dir,
            collection_audit_path=collection_audit,
            attribution_dir=attribution_dir,
            sentiment_dir=sentiment_dir,
            decisions_path=decisions,
            policy_path=policy,
            out_dir=out_dir,
            progress=report_progress,
        )
        memory = result["memory"]
        if not isinstance(memory, dict):
            raise TypeError("catalyst lineage memory evidence is malformed")
        console.print(
            {
                "status": result["status"],
                "requested_chunks": result["requested_chunks"],
                "observed_chunks": result["observed_chunks"],
                "skipped_chunks": result["skipped_chunks"],
                "failed_chunks": result["failed_chunks"],
                "relation_rows": result["relation_rows"],
                "training_eligible_rows": result["training_eligible_rows"],
                "assignment_rows": result["assignment_rows"],
                "lineage_sha256": result["lineage_sha256"],
                "peak_working_set_gib": memory["peak_working_set_gib"],
            }
        )
        if result["status"] != "complete":
            raise typer.Exit(code=2)

    @app.command("build-security-business-labels")
    def build_security_business_labels_command(
        memberships: Path = typer.Option(
            ...,
            help="Canonical point-in-time membership artifact.",
        ),
        universe: Path = typer.Option(
            ...,
            help="Point-in-time universe with company identities.",
        ),
        profiles: Path = typer.Option(
            ...,
            help="Canonical current-profile evidence artifact.",
        ),
        training_dataset: Path = typer.Option(
            ...,
            help="Frozen swing dataset defining label-eligible securities.",
        ),
        policy: Path = typer.Option(
            Path("configs/security_business_labels.toml"),
            help="Closed business-tag taxonomy and exact evidence rules.",
        ),
        assignments_out: Path = typer.Option(
            ...,
            help="New canonical point-in-time business-tag assignments.",
        ),
        coverage_out: Path = typer.Option(
            ...,
            help="New explicit training-security coverage artifact.",
        ),
        summary_out: Path = typer.Option(
            ...,
            help="New lineage-bound assignment summary JSON.",
        ),
    ) -> None:
        """Build auditable business tags without historical profile leakage."""

        outputs = (assignments_out, coverage_out)
        occupied = [
            path
            for path in outputs
            if path.exists() or manifest_path_for(path).exists()
        ]
        if occupied:
            raise typer.BadParameter(
                f"security-label output already exists: {occupied[0]}"
            )
        if summary_out.exists():
            raise typer.BadParameter(
                f"security-label summary already exists: {summary_out}"
            )
        artifact = build_security_label_artifact(
            memberships_path=memberships,
            universe_path=universe,
            profiles_path=profiles,
            training_dataset_path=training_dataset,
            policy_path=policy,
        )
        created: list[Path] = []
        try:
            assignment_manifest = write_canonical_artifact(
                artifact.assignments,
                assignments_out,
                artifact_type="security_business_labels",
                audit=artifact.audit,
                inputs=artifact.inputs,
                production_ready=False,
            )
            created.extend(
                (
                    assignments_out,
                    manifest_path_for(assignments_out),
                )
            )
            coverage_manifest = write_canonical_artifact(
                artifact.coverage,
                coverage_out,
                artifact_type="security_business_label_coverage",
                audit=artifact.audit,
                inputs={
                    **artifact.inputs,
                    "assignments_sha256": str(
                        assignment_manifest["artifact_sha256"]
                    ),
                },
                production_ready=False,
            )
            created.extend(
                (
                    coverage_out,
                    manifest_path_for(coverage_out),
                )
            )
            summary = {
                **artifact.summary,
                "outputs": {
                    "assignments": _manifest_identity(
                        assignments_out,
                        assignment_manifest,
                    ),
                    "coverage": _manifest_identity(
                        coverage_out,
                        coverage_manifest,
                    ),
                },
            }
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            temporary = summary_out.with_name(
                f".{summary_out.name}.{uuid4().hex}.tmp"
            )
            temporary.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, summary_out)
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        console.print(
            {
                "training_security_ids": artifact.summary[
                    "training_security_ids"
                ],
                "training_ticker_histories": artifact.summary[
                    "training_ticker_histories"
                ],
                "historical_assigned_security_ids": artifact.summary[
                    "historical_assigned_security_ids"
                ],
                "historical_insufficient_security_ids": artifact.summary[
                    "historical_insufficient_security_ids"
                ],
                "current_profile_assigned_security_ids": artifact.summary[
                    "current_profile_assigned_security_ids"
                ],
                "historical_exposure_training_ready": artifact.summary[
                    "historical_exposure_training_ready"
                ],
                "assignments_out": str(assignments_out),
                "coverage_out": str(coverage_out),
                "summary_out": str(summary_out),
            }
        )

    @app.command("score-alpaca-news-history")
    @serialized_heavy_job("score-alpaca-news-history")
    def score_alpaca_news_history_command(
        collection_dir: Path = typer.Option(
            ...,
            help="Completed audited Alpaca news collection.",
        ),
        collection_audit: Path = typer.Option(
            ...,
            help="Passed Alpaca news audit summary JSON.",
        ),
        universe: Path = typer.Option(
            ...,
            help="Point-in-time universe with company/sector/industry metadata.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Resumable research-only sentiment artifact directory.",
        ),
        text_mode: str = typer.Option(
            "title_summary",
            help="FinBERT input mode.",
        ),
        max_length: int = typer.Option(128, min=1, max=512),
        batch_size: int = typer.Option(32, min=1, max=256),
        torch_threads: int = typer.Option(4, min=1, max=32),
        fixed_latency_minutes: int = typer.Option(5, min=0, max=60),
    ) -> None:
        """Score audited historical events sequentially with explicit proxy timing."""

        settings = get_settings()
        scorer = FinbertScorer(
            settings.finbert_model,
            torch_num_threads=torch_threads,
            max_length=max_length,
        )

        def report_progress(payload: dict[str, object]) -> None:
            index_value = payload.get("index")
            total_value = payload.get("total")
            if not isinstance(index_value, int) or not isinstance(
                total_value,
                int,
            ):
                raise TypeError("sentiment progress requires integer counters")
            index = index_value
            total = total_value
            if (
                index == 1
                or index == total
                or index % 25 == 0
                or payload["status"] == "failed"
            ):
                console.print(payload)

        result = score_alpaca_news_history(
            collection_dir=collection_dir,
            collection_audit_path=collection_audit,
            universe_path=universe,
            out_dir=out_dir,
            scorer=scorer,
            model_name=settings.finbert_model,
            model_revision=scorer.model_revision,
            execution_device=scorer.device,
            text_mode=text_mode,
            max_length=max_length,
            batch_size=batch_size,
            fixed_latency_minutes=fixed_latency_minutes,
            progress=report_progress,
        )
        console.print(
            {
                "status": result["status"],
                "requested_chunks": result["requested_chunks"],
                "observed_chunks": result["observed_chunks"],
                "failed_chunks": result["failed_chunks"],
                "excluded_security_ids": result["excluded_security_ids"],
                "total_rows": result["total_rows"],
                "peak_working_set_gib": result["memory"][
                    "peak_working_set_gib"
                ],
            }
        )
        if result["status"] != "complete":
            raise typer.Exit(code=2)

    @app.command("audit-alpaca-news-history")
    @serialized_heavy_job("audit-alpaca-news-history")
    def audit_alpaca_news_history_command(
        collection_dir: Path = typer.Option(
            ...,
            help="Completed immutable Alpaca news-history collection.",
        ),
        out: Path = typer.Option(
            ...,
            help="New chunk-level audit CSV.",
        ),
        summary_out: Path = typer.Option(
            ...,
            help="New aggregate audit JSON.",
        ),
    ) -> None:
        """Replay raw-page, event, identity, and manifest integrity sequentially."""

        if out.exists() or summary_out.exists():
            raise typer.BadParameter(
                "Alpaca news audit outputs must not already exist"
            )
        report, summary = audit_alpaca_news_history(collection_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        csv_temporary = out.with_name(f".{out.name}.{uuid4().hex}.tmp")
        summary_temporary = summary_out.with_name(
            f".{summary_out.name}.{uuid4().hex}.tmp"
        )
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
                "passed": summary["passed"],
                "chunks": summary["requested_chunks"],
                "event_rows": summary["event_rows"],
                "pages": summary["page_count"],
                "peak_working_set_gib": summary["memory"][
                    "peak_working_set_gib"
                ],
                "out": str(out),
                "summary_out": str(summary_out),
            }
        )
        if not summary["passed"]:
            raise typer.Exit(code=2)

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
                artifact_type="memberships",
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
