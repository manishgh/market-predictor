from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from market_predictor.canonical.store import (
    canonical_artifact_columns,
    file_sha256,
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.canonical.store import (
    manifest_path_for as canonical_manifest_path_for,
)
from market_predictor.commands.configuration import load_typed_config
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.promotion_identity import (
    DEFAULT_APPROVER_TOKEN_ENV,
    DEFAULT_BUILD_TOKEN_ENV,
    PromotionIdentityConfig,
    promotion_tokens_from_environment,
)
from market_predictor.promotion_workflow import PromotionTrustContext
from market_predictor.registry import manifest_path_for
from market_predictor.resources import memory_audit
from market_predictor.swing.contracts import (
    SwingDatasetConfig,
    SwingPromotionConfig,
    SwingTrainingConfig,
)
from market_predictor.swing.dataset import (
    DECISION_REQUIRED_COLUMNS,
    build_swing_dataset,
    build_swing_feature_history,
    build_swing_inference_features,
)
from market_predictor.swing.model import (
    swing_training_input_columns,
    train_swing_model,
)
from market_predictor.swing.promotion import (
    load_swing_training_evidence,
    promote_swing_model,
    write_swing_training_evidence,
)
from market_predictor.swing.strategy_labels import (
    build_swing_strategy_label_bundle,
    load_swing_strategy_label_policy,
    prune_swing_strategy_label_inputs,
)


def register_swing_model_commands(app: typer.Typer, console: Console) -> None:
    @app.command("build-swing-dataset")
    @serialized_heavy_job("build-swing-dataset")
    def build_swing_dataset_command(
        decisions: Path = typer.Option(..., help="Hash-verified canonical decision artifact."),
        benchmark_bars: Path = typer.Option(..., help="Hash-verified SPY, QQQ, and sector daily bars."),
        global_events: Path | None = typer.Option(
            None,
            help="Hash-verified MARKET events; catalyst_full only.",
        ),
        global_source_collections: Path | None = typer.Option(
            None,
            help="Hash-verified MARKET source states; catalyst_full only.",
        ),
        out: Path = typer.Option(..., help="Immutable canonical swing dataset parquet."),
        config_path: Path | None = typer.Option(None, "--config", help="Swing dataset JSON or TOML config."),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Build an audited point-in-time daily swing feature and label artifact."""

        config = load_typed_config(config_path, SwingDatasetConfig)
        decision_frame, benchmark_frame, global_event_frame, global_collection_frame = _load_swing_build_inputs(
            decisions,
            benchmark_bars,
            global_events,
            global_source_collections,
            feature_profile=config.feature_profile,
            production=production,
        )
        dataset, audit = build_swing_dataset(
            decision_frame,
            benchmark_frame,
            global_events=global_event_frame,
            global_source_collections=global_collection_frame,
            config=config,
        )
        if not audit.passed:
            console.print(
                {
                    "audit": audit.to_frame().to_dict(orient="records"),
                }
            )
        input_paths = [
            path
            for path in (
                decisions,
                benchmark_bars,
                global_events,
                global_source_collections,
            )
            if path is not None
        ]
        inputs = {str(path): file_sha256(path) for path in input_paths}
        inputs["feature_profile"] = config.feature_profile
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
                "memory": memory_audit(
                    hard_budget_gib=config.max_build_memory_gb,
                    headroom_gib=config.memory_guard_headroom_gb,
                ).to_record(),
            }
        )

    @app.command("build-swing-strategy-labels")
    @serialized_heavy_job("build-swing-strategy-labels")
    def build_swing_strategy_labels_command(
        decisions: Path = typer.Option(
            ...,
            help="Hash-verified canonical decision artifact.",
        ),
        benchmark_bars: Path = typer.Option(
            ...,
            help="Hash-verified SPY, QQQ, and sector daily bars.",
        ),
        global_events: Path | None = typer.Option(
            None,
            help="Hash-verified MARKET events; catalyst_full only.",
        ),
        global_source_collections: Path | None = typer.Option(
            None,
            help="Hash-verified MARKET source states; catalyst_full only.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="Resumable directory of immutable per-strategy artifacts.",
        ),
        config_path: Path | None = typer.Option(
            None,
            "--config",
            help="Swing dataset JSON or TOML config.",
        ),
        strategy_policy: Path = typer.Option(
            Path("configs/swing_strategy_labels.toml"),
            help="Frozen KS2 strategy-label policy.",
        ),
        production: bool = typer.Option(
            False,
            "--production/--research",
        ),
    ) -> None:
        """Build and replay distinct causal labels for each swing strategy."""

        config = load_typed_config(config_path, SwingDatasetConfig)
        policy = load_swing_strategy_label_policy(strategy_policy)
        (
            decision_frame,
            benchmark_frame,
            global_event_frame,
            global_collection_frame,
        ) = _load_swing_build_inputs(
            decisions,
            benchmark_bars,
            global_events,
            global_source_collections,
            feature_profile=config.feature_profile,
            production=production,
        )
        features, prepared_benchmarks = build_swing_feature_history(
            decision_frame,
            benchmark_frame,
            global_events=global_event_frame,
            global_source_collections=global_collection_frame,
            config=config,
        )
        prune_swing_strategy_label_inputs(features, policy)
        input_paths = [
            path
            for path in (
                decisions,
                benchmark_bars,
                global_events,
                global_source_collections,
            )
            if path is not None
        ]
        inputs = {str(path): file_sha256(path) for path in input_paths}
        for path in input_paths:
            manifest_path = canonical_manifest_path_for(path)
            inputs[str(manifest_path)] = file_sha256(manifest_path)
        inputs[str(strategy_policy)] = file_sha256(strategy_policy)
        inputs["strategy_label_policy_sha256"] = policy.sha256()
        result = build_swing_strategy_label_bundle(
            features,
            prepared_benchmarks,
            dataset_config=config,
            policy=policy,
            out_dir=out_dir,
            input_hashes=inputs,
            production_ready=production,
            progress=console.print,
        )
        console.print(result)
        if result["status"] != "complete":
            raise typer.Exit(code=2)

    @app.command("build-swing-live-features")
    @serialized_heavy_job("build-swing-live-features")
    def build_swing_live_features_command(
        decisions: Path = typer.Option(..., help="Hash-verified canonical decision artifact."),
        benchmark_bars: Path = typer.Option(..., help="Hash-verified SPY, QQQ, and sector daily bars."),
        global_events: Path | None = typer.Option(
            None,
            help="Hash-verified MARKET events; catalyst_full only.",
        ),
        global_source_collections: Path | None = typer.Option(
            None,
            help="Hash-verified MARKET source states; catalyst_full only.",
        ),
        out: Path = typer.Option(..., help="Immutable latest swing inference feature artifact."),
        config_path: Path | None = typer.Option(None, "--config", help="Swing dataset JSON or TOML config."),
    ) -> None:
        """Build a label-free, audited latest swing inference snapshot."""

        config = load_typed_config(config_path, SwingDatasetConfig)
        decision_frame, benchmark_frame, global_event_frame, global_collection_frame = _load_swing_build_inputs(
            decisions,
            benchmark_bars,
            global_events,
            global_source_collections,
            feature_profile=config.feature_profile,
            production=True,
        )
        features, audit = build_swing_inference_features(
            decision_frame,
            benchmark_frame,
            global_events=global_event_frame,
            global_source_collections=global_collection_frame,
            config=config,
        )
        input_paths = [
            path
            for path in (
                decisions,
                benchmark_bars,
                global_events,
                global_source_collections,
            )
            if path is not None
        ]
        inputs = {str(path): file_sha256(path) for path in input_paths}
        inputs["feature_profile"] = config.feature_profile
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
    @serialized_heavy_job("train-swing-model")
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
        config = load_typed_config(config_path, SwingTrainingConfig)
        frame, manifest = load_canonical_artifact(
            dataset,
            expected_type="swing_dataset",
            allow_research=not production,
            columns=swing_training_input_columns(
                canonical_artifact_columns(dataset),
                config,
            ),
        )
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
        identity_issuer: str = typer.Option(
            ...,
            help="OIDC issuer trusted for promotion identities.",
        ),
        identity_audience: str = typer.Option(
            ...,
            help="OIDC audience required for promotion identities.",
        ),
        identity_jwks: Path = typer.Option(
            ...,
            help="Deployment-owned JWKS file for promotion identity verification.",
        ),
        build_token_env: str = typer.Option(
            DEFAULT_BUILD_TOKEN_ENV,
            help="Environment variable containing the promotion.build OIDC token.",
        ),
        approver_token_env: str = typer.Option(
            DEFAULT_APPROVER_TOKEN_ENV,
            help="Environment variable containing the promotion.approve OIDC token.",
        ),
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
                identity_config=PromotionIdentityConfig(
                    issuer=identity_issuer,
                    audience=identity_audience,
                    jwks_path=identity_jwks,
                ),
                identity_tokens=promotion_tokens_from_environment(
                    build_token_env=build_token_env,
                    approver_token_env=approver_token_env,
                ),
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
    global_events: Path | None,
    global_source_collections: Path | None,
    *,
    feature_profile: str,
    production: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    decision_frame, _ = load_canonical_artifact(
        decisions,
        expected_type="decisions",
        allow_research=not production,
        columns=(
            sorted(DECISION_REQUIRED_COLUMNS | {"sector"})
            if feature_profile == "technical_market"
            else None
        ),
    )
    benchmark_frame, _ = load_canonical_artifact(
        benchmark_bars,
        expected_type="bars",
        allow_research=not production,
    )
    if feature_profile == "technical_market":
        if global_events is not None or global_source_collections is not None:
            raise typer.BadParameter(
                "technical_market rejects global event and source-collection inputs"
            )
        return decision_frame, benchmark_frame, None, None
    if global_events is None or global_source_collections is None:
        raise typer.BadParameter(
            "catalyst_full requires --global-events and --global-source-collections"
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
