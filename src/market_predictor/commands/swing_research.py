from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console

from market_predictor.canonical.store import (
    file_sha256,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events.alpaca_news_audit import (
    audit_alpaca_news_history,
)
from market_predictor.config import get_settings
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.sentiment import FinbertScorer
from market_predictor.swing.catalyst_lineage import build_catalyst_lineage
from market_predictor.swing.event_attribution_history import (
    attribute_alpaca_news_history,
)
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
        max_batch_events: int = typer.Option(
            2_048,
            min=1,
            max=16_384,
            help="Maximum events loaded across one FinBERT scorer call.",
        ),
        max_batch_shards: int = typer.Option(
            32,
            min=1,
            max=256,
            help="Maximum source chunks combined into one scorer call.",
        ),
        torch_threads: int = typer.Option(4, min=1, max=32),
        fixed_latency_minutes: int = typer.Option(5, min=0, max=60),
    ) -> None:
        """Score audited history in bounded batches with explicit proxy timing."""

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
            max_batch_events=max_batch_events,
            max_batch_shards=max_batch_shards,
            fixed_latency_minutes=fixed_latency_minutes,
            progress=report_progress,
        )
        console.print(
            {
                "status": result["status"],
                "requested_chunks": result["requested_chunks"],
                "observed_chunks": result["observed_chunks"],
                "failed_chunks": result["failed_chunks"],
                "scorer_calls": result["scorer_calls"],
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
