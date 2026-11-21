"""Verify the weekday prospective forward scoring PowerShell scheduler adapters."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("run_forward_research_scoring.ps1", "install_forward_research_task.ps1")


@pytest.mark.parametrize("name", SCRIPTS)
def test_forward_powershell_script_parses(name: str) -> None:
    shell = shutil.which("powershell.exe")
    if shell is None:
        pytest.skip("Windows adapter syntax is tested on Windows only")
    path = str(ROOT / "scripts" / name).replace("'", "''")
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile('{path}',"
            "[ref]$t,[ref]$e) | Out-Null; if ($e.Count) { $e | Out-String | Write-Output; exit 1 }",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_forward_runner_and_installer_contracts() -> None:
    runner = (ROOT / "scripts" / SCRIPTS[0]).read_text(encoding="utf-8")
    assert "$childExitCode = $LASTEXITCODE" in runner
    assert "record-forward-research-scores" in runner
    assert "evaluate-forward-research-scores" in runner
    assert "--skip-existing" in runner
    assert "Activate.ps1" not in runner

    installer = (ROOT / "scripts" / SCRIPTS[1]).read_text(encoding="utf-8")
    assert "SupportsShouldProcess = $true" in installer
    assert "MarketPredictorForwardScoringPostClose" in installer
    assert "-Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday" in installer
    assert "-MultipleInstances IgnoreNew" in installer
    assert "-LogonType Interactive -RunLevel Limited" in installer
