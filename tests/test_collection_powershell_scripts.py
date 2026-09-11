"""Verify the local scheduler adapters without installing or starting a task."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("run_swing_data_collection.ps1", "install_swing_collection_task.ps1")


@pytest.mark.parametrize("name", SCRIPTS)
def test_collection_script_parses(name: str) -> None:
    shell = shutil.which("powershell.exe")
    if shell is None:
        pytest.skip("Windows adapter syntax is tested on Windows only")
    path = str(ROOT / "scripts" / name).replace("'", "''")
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command",
         f"$t=$null; $e=$null; [System.Management.Automation.Language.Parser]::ParseFile('{path}',"
         "[ref]$t,[ref]$e) | Out-Null; if ($e.Count) { $e | Out-String | Write-Output; exit 1 }"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_collection_adapter_preserves_child_exit_and_separates_training() -> None:
    source = (ROOT / "scripts" / SCRIPTS[0]).read_text()
    assert "$childExitCode = $LASTEXITCODE" in source
    assert 'if ($childExitCode -ne 0) { $exitCode = $childExitCode }' in source
    assert "exit $exitCode" in source
    assert '"market_predictor.swing.datasets.alpaca_incremental"' in source
    assert "Activate.ps1" not in source
    assert "train-" not in source
    assert "score-" not in source


def test_scheduler_has_rollback_and_nonoverlap() -> None:
    source = (ROOT / "scripts" / SCRIPTS[1]).read_text()
    assert "SupportsShouldProcess = $true" in source
    assert "-MultipleInstances IgnoreNew" in source
    assert "-WorkingDirectory $root" in source
    assert "Export-ScheduledTask" in source
    assert source.index("Register-ScheduledTask -TaskName") < source.index("Unregister-ScheduledTask -TaskName")
    assert "-LogonType Interactive -RunLevel Limited" in source


def test_collection_rejects_missing_interpreter(tmp_path: Path) -> None:
    shell = shutil.which("powershell.exe")
    if shell is None:
        pytest.skip("Windows adapter execution is tested on Windows only")
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-File", str(ROOT / "scripts" / SCRIPTS[0]),
         "-ProjectDir", str(tmp_path)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode != 0
    assert "Project Python is missing" in result.stderr


@pytest.fixture
def native_exit_fixture(tmp_path: Path) -> Path:
    """A test-only native executable exercises PowerShell's stderr/exit handling."""
    shell = shutil.which("powershell.exe")
    if shell is None:
        pytest.skip("Native Windows adapter fixture")
    target = tmp_path / ".venv" / "Scripts" / "python.exe"
    target.parent.mkdir(parents=True)
    config = tmp_path / "configs"
    config.mkdir()
    for name in ("swing_incremental_collection.toml", "swing_incremental_sec.toml"):
        (config / name).write_text("# test-only wrapper fixture\n")
    program = (
        'public class ExitFixture { public static void Main() { '
        'System.Console.WriteLine("fixture-out"); System.Console.Error.WriteLine("fixture-error"); '
        'System.Environment.Exit(int.Parse(System.Environment.GetEnvironmentVariable("FIXTURE_EXIT_CODE"))); } }'
    )
    escaped = str(target).replace("'", "''")
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command",
         f"Add-Type -TypeDefinition '{program}' -OutputAssembly '{escaped}' -OutputType ConsoleApplication"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    return tmp_path


@pytest.mark.parametrize("code", [0, 2, 75])
def test_real_native_stderr_preserves_exit(native_exit_fixture: Path, code: int) -> None:
    result = subprocess.run(
        [str(shutil.which("powershell.exe")), "-NoProfile", "-NonInteractive", "-File",
         str(ROOT / "scripts" / SCRIPTS[0]), "-ProjectDir", str(native_exit_fixture), "-Source", "alpaca"],
        env={**os.environ, "FIXTURE_EXIT_CODE": str(code)},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == code, result.stdout + result.stderr
    logs = list((native_exit_fixture / "data/runtime/collection_logs").glob("*.log"))
    assert len(logs) == 1


@pytest.mark.parametrize("task_name", ["MarketPredictorLiveMidnight", "MarketPredictorSwingCollectionMidnight"])
def test_scheduler_rejects_sibling_checkout(native_exit_fixture: Path, task_name: str) -> None:
    root = native_exit_fixture
    (root / "scripts").mkdir()
    (root / "scripts/run_swing_data_collection.ps1").write_text("# test-only adapter\n")
    sibling = str(root) + "-other"
    filename = "run_live_midnight.ps1" if task_name == "MarketPredictorLiveMidnight" else "run_swing_data_collection.ps1"
    arguments = f'-File "{sibling}\\scripts\\{filename}" -ProjectDir "{sibling}"'
    command = (
        f"function Get-ScheduledTask {{ [pscustomobject]@{{TaskName='{task_name}'; TaskPath='\\'; State='Ready'; "
        f"Actions=@([pscustomobject]@{{Arguments='{arguments}'; WorkingDirectory='{sibling}'}})}} }}; "
        "function Register-ScheduledTask { throw 'registration-must-not-run' }; "
        "function Unregister-ScheduledTask { throw 'removal-must-not-run' }; "
        f"& '{ROOT / 'scripts' / SCRIPTS[1]}' -ProjectDir '{root}'"
    )
    result = subprocess.run(
        [str(shutil.which("powershell.exe")), "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode != 0
    assert "not owned by" in result.stderr
    assert "must-not-run" not in result.stderr


def test_scheduler_registration_failure_preserves_old_tasks(native_exit_fixture: Path) -> None:
    root = native_exit_fixture
    (root / "scripts").mkdir()
    (root / "scripts/run_swing_data_collection.ps1").write_text("# test-only adapter\n")
    arguments = f'-File "{root}\\scripts\\run_live_midnight.ps1" -ProjectDir "{root}"'
    command = (
        "function Get-ScheduledTask { [pscustomobject]@{TaskName='MarketPredictorLiveMidnight'; TaskPath='\\'; State='Ready'; "
        f"Actions=@([pscustomobject]@{{Arguments='{arguments}'; WorkingDirectory='{root}'}})}} }}; "
        "function Export-ScheduledTask { '<Task>fixture</Task>' }; "
        "function New-ScheduledTaskAction { [pscustomobject]@{Arguments='fixture'} }; "
        "function New-ScheduledTaskTrigger { 'fixture' }; "
        "function New-ScheduledTaskSettingsSet { 'fixture' }; "
        "function New-ScheduledTaskPrincipal { 'fixture' }; "
        "function Register-ScheduledTask { throw 'fixture-registration-failure' }; "
        "function Unregister-ScheduledTask { Write-Output 'UNEXPECTED-REMOVAL' }; "
        f"& '{ROOT / 'scripts' / SCRIPTS[1]}' -ProjectDir '{root}'"
    )
    result = subprocess.run(
        [str(shutil.which("powershell.exe")), "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode != 0
    assert "fixture-registration-failure" in result.stderr
    assert "UNEXPECTED-REMOVAL" not in result.stdout
    assert len(list((root / "data/runtime/scheduler_migrations").glob("*/*.xml"))) == 1
