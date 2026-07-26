from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from market_predictor.strategy_governance import (
    validate_strategy_execution_ledger,
)


def register_strategy_governance_commands(app: typer.Typer, console: Any) -> None:
    @app.command("validate-strategy-execution-ledger")
    def validate_strategy_execution_ledger_command(
        ledger: Path = typer.Option(
            Path("docs/strategy_execution_ledger.json"),
            help="Authoritative strategy execution ledger.",
        ),
        repository_root: Path = typer.Option(
            Path("."),
            help="Repository root used to resolve evidence and Git identities.",
        ),
        verify_git: bool = typer.Option(
            True,
            help="Verify bound commits against local remote-tracking refs.",
        ),
    ) -> None:
        """Validate plan coverage, closure evidence, hashes, and Git bindings."""
        report = validate_strategy_execution_ledger(
            ledger,
            repository_root=repository_root,
            verify_git=verify_git,
        )
        console.print(json.dumps(report, indent=2, sort_keys=True))
