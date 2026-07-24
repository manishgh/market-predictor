from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import typer
from rich.console import Console

from market_predictor.live_features import LiveMode
from market_predictor.release import (
    activate_local_release,
    load_active_local_release,
    publish_local_release,
    rollback_local_release,
    verify_local_release,
)
from market_predictor.serving_bundle import (
    activate_serving_bundle,
    load_active_serving_bundle,
    publish_serving_bundle,
    rollback_serving_bundle,
    verify_serving_bundle,
)


def register_release_commands(app: typer.Typer, console: Console) -> None:
    @app.command("publish-local-release")
    def publish_local_release_command(
        model: Path = typer.Option(..., help="Attested promoted model artifact."),
        evidence_manifest: Path = typer.Option(
            ...,
            help="Hash-bound training evidence manifest.",
        ),
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release repository root.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
        activate: bool = typer.Option(
            True,
            help="Atomically activate the release after full verification.",
        ),
    ) -> None:
        """Publish a content-addressed local release."""

        console.print(
            publish_local_release(
                release_root,
                model_path=model,
                evidence_manifest_path=evidence_manifest,
                activate=activate,
                attestation_trust_store_path=attestation_trust_store,
            )
        )

    @app.command("verify-local-release")
    def verify_local_release_command(
        release_id: str = typer.Option(..., help="Content-addressed release id."),
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release repository root.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
    ) -> None:
        """Verify every file and the promotion attestation in a local release."""

        console.print(
            verify_local_release(
                release_root,
                release_id,
                attestation_trust_store_path=attestation_trust_store,
            )
        )

    @app.command("activate-local-release")
    def activate_local_release_command(
        release_id: str = typer.Option(..., help="Verified release id to activate."),
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release repository root.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
    ) -> None:
        """Atomically move the active pointer to a verified release."""

        console.print(
            activate_local_release(
                release_root,
                release_id,
                attestation_trust_store_path=attestation_trust_store,
            )
        )

    @app.command("rollback-local-release")
    def rollback_local_release_command(
        release_id: str = typer.Option(
            ...,
            help="Previously published release id to restore.",
        ),
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release repository root.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
    ) -> None:
        """Roll back to a complete, verified prior release."""

        console.print(
            rollback_local_release(
                release_root,
                release_id,
                attestation_trust_store_path=attestation_trust_store,
            )
        )

    @app.command("show-active-local-release")
    def show_active_local_release_command(
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release repository root.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
    ) -> None:
        """Verify and show the active local release."""

        console.print(
            load_active_local_release(
                release_root,
                attestation_trust_store_path=attestation_trust_store,
            )
        )

    @app.command("publish-serving-bundle")
    def publish_serving_bundle_command(
        mode: str = typer.Option(..., help="Serving mode: swing or intraday."),
        horizon: str = typer.Option(..., help="Canonical route horizon."),
        model_release_id: str = typer.Option(
            ...,
            help="Verified immutable model-release id.",
        ),
        feature_snapshot: Path = typer.Option(
            ...,
            help="Registered live feature Parquet snapshot.",
        ),
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release and serving-bundle repository.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
        activate: bool = typer.Option(
            True,
            help="Atomically activate the bundle after complete verification.",
        ),
        generated_at: str | None = typer.Option(
            None,
            help="ISO-8601 timezone-aware bundle timestamp; defaults to current UTC.",
        ),
    ) -> None:
        """Publish one immutable model, policy, calibration, and feature bundle."""

        live_mode = _serving_mode(mode)
        console.print(
            publish_serving_bundle(
                release_root,
                mode=live_mode,
                horizon=horizon.strip().lower(),
                model_release_id=model_release_id.strip().lower(),
                feature_path=feature_snapshot,
                attestation_trust_store_path=attestation_trust_store,
                activate=activate,
                generated_at=(
                    datetime.fromisoformat(generated_at)
                    if generated_at is not None
                    else None
                ),
            )
        )

    @app.command("verify-serving-bundle")
    def verify_serving_bundle_command(
        bundle_id: str = typer.Option(..., help="Content-addressed serving-bundle id."),
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release and serving-bundle repository.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
    ) -> None:
        """Verify every transitive identity in one serving bundle."""

        console.print(
            verify_serving_bundle(
                release_root,
                bundle_id.strip().lower(),
                attestation_trust_store_path=attestation_trust_store,
            )
        )

    @app.command("activate-serving-bundle")
    def activate_serving_bundle_command(
        bundle_id: str = typer.Option(..., help="Verified serving-bundle id."),
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release and serving-bundle repository.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
    ) -> None:
        """Atomically move the active pointer to one verified serving bundle."""

        console.print(
            activate_serving_bundle(
                release_root,
                bundle_id.strip().lower(),
                attestation_trust_store_path=attestation_trust_store,
            )
        )

    @app.command("rollback-serving-bundle")
    def rollback_serving_bundle_command(
        bundle_id: str = typer.Option(
            ...,
            help="Immediately previous serving-bundle id.",
        ),
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release and serving-bundle repository.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
    ) -> None:
        """Roll back atomically to the verified immediately previous bundle."""

        console.print(
            rollback_serving_bundle(
                release_root,
                bundle_id.strip().lower(),
                attestation_trust_store_path=attestation_trust_store,
            )
        )

    @app.command("show-active-serving-bundle")
    def show_active_serving_bundle_command(
        release_root: Path = typer.Option(
            Path("data/releases"),
            help="Durable local release and serving-bundle repository.",
        ),
        attestation_trust_store: Path = typer.Option(
            ...,
            help="Trusted Ed25519 promotion signer registry.",
        ),
    ) -> None:
        """Verify and show the active atomic serving bundle."""

        console.print(
            load_active_serving_bundle(
                release_root,
                attestation_trust_store_path=attestation_trust_store,
            )
        )


def _serving_mode(value: str) -> LiveMode:
    normalized = value.strip().lower()
    if normalized not in {"swing", "intraday"}:
        raise typer.BadParameter("mode must be swing or intraday")
    return cast(LiveMode, normalized)
