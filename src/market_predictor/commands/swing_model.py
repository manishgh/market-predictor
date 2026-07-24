from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.commands.configuration import load_typed_config
from market_predictor.promotion_workflow import PromotionTrustContext
from market_predictor.registry import manifest_path_for
from market_predictor.swing.contracts import (
    SwingDatasetConfig,
    SwingPromotionConfig,
    SwingTrainingConfig,
)
from market_predictor.swing.dataset import build_swing_dataset, build_swing_inference_features
from market_predictor.swing.model import train_swing_model
from market_predictor.swing.promotion import (
    load_swing_training_evidence,
    promote_swing_model,
    write_swing_training_evidence,
)


def register_swing_model_commands(app: typer.Typer, console: Console) -> None:
    @app.command("build-swing-dataset")
    def build_swing_dataset_command(
        decisions: Path = typer.Option(..., help="Hash-verified canonical decision artifact."),
        benchmark_bars: Path = typer.Option(..., help="Hash-verified SPY, QQQ, and sector daily bars."),
        global_events: Path = typer.Option(..., help="Hash-verified canonical MARKET event artifact."),
        global_source_collections: Path = typer.Option(
            ...,
            help="Hash-verified source collection states for ticker MARKET.",
        ),
        out: Path = typer.Option(..., help="Immutable canonical swing dataset parquet."),
        config_path: Path | None = typer.Option(None, "--config", help="Swing dataset JSON or TOML config."),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Build an audited point-in-time daily swing feature and label artifact."""

        decision_frame, benchmark_frame, global_event_frame, global_collection_frame = _load_swing_build_inputs(
            decisions,
            benchmark_bars,
            global_events,
            global_source_collections,
            production=production,
        )
        config = load_typed_config(config_path, SwingDatasetConfig)
        dataset, audit = build_swing_dataset(
            decision_frame,
            benchmark_frame,
            global_events=global_event_frame,
            global_source_collections=global_collection_frame,
            config=config,
        )
        inputs = {str(path): file_sha256(path) for path in (decisions, benchmark_bars, global_events, global_source_collections)}
        manifest = write_canonical_artifact(
            dataset,
            out,
            artifact_type="swing_dataset",
            audit=audit,
            inputs=inputs,
            production_ready=production,
        )
        console.print(
            {
                "rows": len(dataset),
                "eligible_rows": int(dataset["label_eligible"].fillna(False).sum()),
                "out": str(out),
                "sha256": manifest["artifact_sha256"],
            }
        )

    @app.command("build-swing-live-features")
    def build_swing_live_features_command(
        decisions: Path = typer.Option(..., help="Hash-verified canonical decision artifact."),
        benchmark_bars: Path = typer.Option(..., help="Hash-verified SPY, QQQ, and sector daily bars."),
        global_events: Path = typer.Option(..., help="Hash-verified canonical MARKET event artifact."),
        global_source_collections: Path = typer.Option(
            ...,
            help="Hash-verified source collection states for ticker MARKET.",
        ),
        out: Path = typer.Option(..., help="Immutable latest swing inference feature artifact."),
        config_path: Path | None = typer.Option(None, "--config", help="Swing dataset JSON or TOML config."),
    ) -> None:
        """Build a label-free, audited latest swing inference snapshot."""

        decision_frame, benchmark_frame, global_event_frame, global_collection_frame = _load_swing_build_inputs(
            decisions,
            benchmark_bars,
            global_events,
            global_source_collections,
            production=True,
        )
        features, audit = build_swing_inference_features(
            decision_frame,
            benchmark_frame,
            global_events=global_event_frame,
            global_source_collections=global_collection_frame,
            config=load_typed_config(config_path, SwingDatasetConfig),
        )
        inputs = {str(path): file_sha256(path) for path in (decisions, benchmark_bars, global_events, global_source_collections)}
        manifest = write_canonical_artifact(
            features,
            out,
            artifact_type="swing_inference_features",
            audit=audit,
            inputs=inputs,
            production_ready=True,
        )
        console.print(
            {
                "rows": len(features),
                "decision_time_utc": str(features["decision_time_utc"].iloc[0]),
                "out": str(out),
                "sha256": manifest["artifact_sha256"],
            }
        )

    @app.command("train-swing-model")
    def train_swing_model_command(
        dataset: Path = typer.Option(..., help="Hash-verified canonical swing dataset."),
        model_out: Path = typer.Option(..., help="New candidate model artifact path."),
        evidence_dir: Path = typer.Option(..., help="New directory for promotion evidence."),
        config_path: Path | None = typer.Option(None, "--config", help="Swing training JSON or TOML config."),
        production: bool = typer.Option(True, "--production/--research"),
        overwrite: bool = typer.Option(False, help="Explicitly replace model and evidence outputs."),
    ) -> None:
        """Train a candidate with purged walk-forward and unseen-ticker validation."""

        if not overwrite and (model_out.exists() or manifest_path_for(model_out).exists()):
            raise typer.BadParameter(f"model output already exists: {model_out}")
        if not overwrite and evidence_dir.exists() and any(evidence_dir.iterdir()):
            raise typer.BadParameter(f"evidence directory is not empty: {evidence_dir}")
        frame, manifest = load_canonical_artifact(
            dataset,
            expected_type="swing_dataset",
            allow_research=not production,
        )
        config = load_typed_config(config_path, SwingTrainingConfig)
        result = train_swing_model(
            frame,
            model_out=model_out,
            dataset_sha256=str(manifest["artifact_sha256"]),
            config=config,
            overwrite=overwrite,
        )
        evidence = write_swing_training_evidence(result, evidence_dir, overwrite=overwrite)
        console.print(
            {
                "model": str(model_out),
                "status": result.manifest["status"],
                "model_run_id": result.metrics["model_run_id"],
                "roc_auc": result.metrics["roc_auc"],
                "ticker_holdout_roc_auc": result.metrics["ticker_holdout_roc_auc"],
                "evidence": {name: str(path) for name, path in evidence.items()},
            }
        )

    @app.command("promote-swing-model")
    def promote_swing_model_command(
        model: Path = typer.Option(..., help="Candidate canonical swing model."),
        evidence_dir: Path = typer.Option(..., help="Evidence directory produced by training."),
        hypothesis_registry: Path = typer.Option(..., help="Root containing immutable hypothesis declarations."),
        hypothesis_id: str = typer.Option(..., help="Predeclared hypothesis identifier."),
        shadow_bundle: Path = typer.Option(..., help="Immutable untouched-shadow evidence bundle."),
        outcome_repository: Path = typer.Option(
            ...,
            help="Durable repository containing paired shadow intents and outcomes.",
        ),
        baseline_artifact: Path = typer.Option(
            ...,
            help="Frozen baseline model artifact declared by the hypothesis.",
        ),
        build_identity: str = typer.Option(..., help="CI/build workload identity recorded in the attestation."),
        approver_identity: str = typer.Option(..., help="Reviewer identity recorded in the attestation."),
        signing_private_key: Path = typer.Option(
            ...,
            help="Ed25519 private key controlled by the promotion workload.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted signer registry used to verify the attestation.",
        ),
        signer_id: str = typer.Option(
            ...,
            help="Trusted signer id corresponding to the private key.",
        ),
        minimum_shadow_sessions: int = typer.Option(60, min=2, help="Minimum independent shadow sessions."),
        minimum_paired_improvement_ci_low: float = typer.Option(
            0.0,
            help="Paired benchmark-excess improvement CI lower bound must be strictly above this value.",
        ),
        config_path: Path | None = typer.Option(None, "--config", help="Promotion gate JSON or TOML config."),
        report_out: Path | None = typer.Option(None, help="Optional promotion report path."),
    ) -> None:
        """Promote a candidate only when all independent production gates pass."""

        evidence = load_swing_training_evidence(evidence_dir, model)
        result = promote_swing_model(
            model_path=model,
            evidence=evidence,
            config=load_typed_config(config_path, SwingPromotionConfig),
            trust_context=PromotionTrustContext(
                hypothesis_registry_root=hypothesis_registry,
                hypothesis_id=hypothesis_id,
                shadow_bundle_path=shadow_bundle,
                outcome_repository_root=outcome_repository,
                baseline_artifact_path=baseline_artifact,
                build_identity=build_identity,
                approver_identity=approver_identity,
                signing_private_key_path=signing_private_key,
                attestation_trust_store_path=attestation_trust_store,
                signer_id=signer_id,
                minimum_shadow_sessions=minimum_shadow_sessions,
                minimum_paired_improvement_ci_low=minimum_paired_improvement_ci_low,
            ),
            report_path=report_out,
        )
        console.print(result)
        if not bool(result["passed"]):
            raise typer.Exit(code=2)


def _load_swing_build_inputs(
    decisions: Path,
    benchmark_bars: Path,
    global_events: Path,
    global_source_collections: Path,
    *,
    production: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decision_frame, _ = load_canonical_artifact(
        decisions,
        expected_type="decisions",
        allow_research=not production,
    )
    benchmark_frame, _ = load_canonical_artifact(
        benchmark_bars,
        expected_type="bars",
        allow_research=not production,
    )
    global_event_frame, _ = load_canonical_artifact(
        global_events,
        expected_type="events",
        allow_research=not production,
    )
    global_collection_frame, _ = load_canonical_artifact(
        global_source_collections,
        expected_type="source_collections",
        allow_research=not production,
    )
    return decision_frame, benchmark_frame, global_event_frame, global_collection_frame
