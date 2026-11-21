[CmdletBinding()]
param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [string]$DataRoot,
    [string]$AsOfDate = "latest",
    [ValidateRange(1, 1000)][int]$CatalystTopK = 100
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectDir).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python is missing: $python"
}

$bundleRoot = if ($DataRoot) { (Resolve-Path -LiteralPath $DataRoot).Path } else { $root }
$bundleManifest = Join-Path $bundleRoot "data\research\research_lane\models\mid_small_beta_neutral_sec_xgb\_manifest.json"
if (-not (Test-Path -LiteralPath $bundleManifest -PathType Leaf)) {
    throw "Persisted SEC catalyst XGBoost bundle manifest is missing: $bundleManifest"
}

$logDir = Join-Path $root "data\runtime\forward_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir ("forward_scoring_{0}_{1}.log" -f (Get-Date -Format "yyyyMMddTHHmmss"), $PID)
$exitCode = 0

$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

Push-Location -LiteralPath $root
try {
    # Invoke the configured environment directly; never depend on shell activation.
    # Windows PowerShell 5.1 treats native stderr as ErrorRecord objects; preserve $LASTEXITCODE.
    $ErrorActionPreference = "Continue"

    $recordArgs = @(
        "-B", "-m", "market_predictor.research_cli",
        "record-forward-research-scores",
        "--root", $root,
        "--as-of-date", $AsOfDate,
        "--live-prospective",
        "--catalyst-top-k", "$CatalystTopK",
        "--skip-existing"
    )
    if ($DataRoot) { $recordArgs += @("--data-root", $DataRoot) }

    & $python @recordArgs 2>&1 | Tee-Object -FilePath $log -Append
    $childExitCode = $LASTEXITCODE
    if ($childExitCode -ne 0) { $exitCode = $childExitCode }

    if ($childExitCode -eq 0) {
        $evalArgs = @(
            "-B", "-m", "market_predictor.research_cli",
            "evaluate-forward-research-scores",
            "--root", $root,
            "--as-of-date", $AsOfDate
        )
        if ($DataRoot) { $evalArgs += @("--data-root", $DataRoot) }

        & $python @evalArgs 2>&1 | Tee-Object -FilePath $log -Append
        $evalExitCode = $LASTEXITCODE
        if ($evalExitCode -ne 0) { $exitCode = $evalExitCode }
    }
} finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
}
exit $exitCode
