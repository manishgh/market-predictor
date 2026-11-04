from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from market_predictor.release import (
    activate_local_release,
    load_active_local_release,
    publish_local_release,
    rollback_local_release,
    verify_local_release,
)
from market_predictor.serving.admission import (
    require_swing_candidate,
    require_swing_release,
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

        require_swing_candidate(model, attestation_trust_store)
        release = publish_local_release(
            release_root,
            model_path=model,
            evidence_manifest_path=evidence_manifest,
            activate=False,
            attestation_trust_store_path=attestation_trust_store,
        )
        release_id = str(release["release_id"])
        require_swing_release(release_root, release_id, attestation_trust_store)
        if activate:
            release["active_pointer"] = activate_local_release(
                release_root, release_id,
                attestation_trust_store_path=attestation_trust_store,
            )
        console.print(release)

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

        require_swing_release(release_root, release_id, attestation_trust_store)
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

        require_swing_release(release_root, release_id, attestation_trust_store)
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
