from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.feature_store import LiveFeatureStore, LiveFeatureStoreConfig
from market_predictor.live_features import (
    LIVE_ARTIFACT_TYPES,
    LIVE_SCHEMA_VERSIONS,
)


def register_production_commands(app: typer.Typer, console: Console) -> None:
    @app.command("serve-api")
    def serve_api(
        host: str = typer.Option("127.0.0.1", help="API bind host."),
        port: int = typer.Option(8000, help="API bind port."),
        reload: bool = typer.Option(
            False,
            help="Enable uvicorn reload for local development.",
        ),
    ) -> None:
        """Serve the typed prediction API."""

        try:
            import uvicorn
        except ImportError as exc:
            raise typer.BadParameter(
                "uvicorn is not installed; install the production dependency set"
            ) from exc
        uvicorn.run(
            "market_predictor.api:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
        )

    @app.command("publish-live-features")
    def publish_live_features(
        mode: str = typer.Option(..., help="Feature mode: swing."),
        input_path: Path = typer.Option(
            ...,
            help="Canonical inference feature artifact to publish.",
        ),
        live_dir: Path = typer.Option(
            Path("data/live"),
            help="Managed live feature root.",
        ),
    ) -> None:
        """Atomically publish an integrity-checked feature snapshot."""

        normalized_mode = mode.strip().lower()
        if normalized_mode != "swing":
            raise typer.BadParameter("mode must be swing")
        expected_type = LIVE_ARTIFACT_TYPES["swing"]
        frame, canonical_manifest = load_canonical_artifact(
            input_path,
            expected_type=expected_type,
            allow_research=False,
        )
        schema_column = "swing_feature_schema_version"
        schemas = (
            set(frame[schema_column].astype(str).unique())
            if schema_column in frame
            else set()
        )
        expected_schema = LIVE_SCHEMA_VERSIONS["swing"]
        if schemas != {expected_schema}:
            raise typer.BadParameter(
                f"canonical {normalized_mode} features do not match schema "
                f"{expected_schema}"
            )
        feeds = set(frame["price_feed"].astype(str).str.lower().str.strip().unique())
        if len(feeds) != 1:
            raise typer.BadParameter(
                "canonical inference features must contain exactly one price feed"
            )
        store = LiveFeatureStore(
            Path("."),
            LiveFeatureStoreConfig(
                swing_path=live_dir / "features/swing.parquet",
            ),
        )
        manifest = store.publish(
            "swing",
            frame,
            price_feed=next(iter(feeds)),
            feature_schema_version=expected_schema,
            source_artifact_sha256=str(canonical_manifest["artifact_sha256"]),
            source_artifact_type=expected_type,
        )
        console.print(manifest)
