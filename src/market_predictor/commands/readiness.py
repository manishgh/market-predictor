"""Prediction data-readiness command adapter."""

from pathlib import Path
from typing import Any

import typer

from market_predictor.governance.readiness.audit import (
    run_prediction_data_readiness_audit,
)
from market_predictor.governance.readiness.contracts import (
    load_prediction_data_readiness_config,
)
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.intraday.specialist_contracts import (
    load_intraday_specialist_research_config,
)


def register_readiness_commands(app: typer.Typer, console: Any) -> None:
    """Register the production data-readiness audit command."""

    @app.command("audit-prediction-data-readiness")
    @serialized_heavy_job("audit-prediction-data-readiness")
    def audit_prediction_data_readiness(
        swing_panel_dir: Path = typer.Option(...),
        swing_candidate_dir: Path = typer.Option(...),
        swing_promoted_bundle_dir: Path | None = typer.Option(None),
        promotion_attestation_trust_store: Path | None = typer.Option(None),
        promotion_gate_policy_sha256: str | None = typer.Option(None),
        intraday_training_dir: Path = typer.Option(...),
        intraday_collection_dir: Path = typer.Option(...),
        intraday_coverage_dir: Path = typer.Option(...),
        catalyst_lineage_dir: Path = typer.Option(...),
        news_source_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/prediction_data_readiness.toml")),
        swing_training_policy: Path = typer.Option(Path("configs/edge_rebuild_swing_training.toml")),
        strategy_contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
        intraday_policy: Path = typer.Option(Path("configs/intraday_specialist_research.toml")),
    ) -> None:
        """Audit source capacity and causal readiness without fitting a model."""

        result = run_prediction_data_readiness_audit(
            swing_panel_dir=swing_panel_dir,
            swing_candidate_dir=swing_candidate_dir,
            swing_promoted_bundle_dir=swing_promoted_bundle_dir,
            promotion_attestation_trust_store_path=(promotion_attestation_trust_store),
            promotion_gate_policy_sha256=promotion_gate_policy_sha256,
            intraday_training_dir=intraday_training_dir,
            intraday_collection_dir=intraday_collection_dir,
            intraday_coverage_dir=intraday_coverage_dir,
            catalyst_lineage_dir=catalyst_lineage_dir,
            news_source_dir=news_source_dir,
            out_dir=out_dir,
            config=load_prediction_data_readiness_config(policy),
            policy_path=policy,
            swing_training_policy_path=swing_training_policy,
            strategy_contract_path=strategy_contract,
            intraday_config=load_intraday_specialist_research_config(intraday_policy),
            intraday_policy_path=intraday_policy,
        )
        console.print(result)
