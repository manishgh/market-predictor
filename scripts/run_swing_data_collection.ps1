[CmdletBinding()]
param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [string]$Config = "configs/swing_incremental_collection.toml",
    [string]$SecConfig = "configs/swing_incremental_sec.toml",
    [ValidateSet("all", "alpaca", "sec")][string]$Source = "all",
    [string]$Through,
    [ValidateRange(1, 1000000)][int]$MaxUnits,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectDir).Path
if (-not $Through) { $Through = [DateTime]::UtcNow.ToString("yyyy-MM-dd") }
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python is missing: $python"
}
$jobs = @()
if ($Source -in @("all", "alpaca")) {
    $jobs += @{ Module = "market_predictor.swing.datasets.alpaca_incremental"; Config = $Config; Limit = "--max-units" }
}
if ($Source -in @("all", "sec")) {
    $jobs += @{ Module = "market_predictor.swing.datasets.sec_incremental"; Config = $SecConfig; Limit = "--max-issuers" }
}
foreach ($job in $jobs) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $job.Config) -PathType Leaf)) {
        throw "Collection configuration is missing: $($job.Config)"
    }
}

$logDir = Join-Path $root "data\runtime\collection_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir ("swing_collection_{0}_{1}.log" -f (Get-Date -Format "yyyyMMddTHHmmss"), $PID)
$exitCode = 0
Push-Location -LiteralPath $root
try {
    # Invoke the configured environment directly; never depend on shell activation.
    # Windows PowerShell 5.1 treats native stderr as ErrorRecord objects. Preserve
    # the child's exit status instead of turning a diagnostic into a wrapper exit.
    $ErrorActionPreference = "Continue"
    foreach ($job in $jobs) {
        $arguments = @("-B", "-m", $job.Module, "--config", (Join-Path $root $job.Config))
        if ($Through) { $arguments += @("--through", $Through) }
        if ($PSBoundParameters.ContainsKey("MaxUnits")) { $arguments += @($job.Limit, "$MaxUnits") }
        if ($Offline) { $arguments += "--offline" }
        & $python @arguments 2>&1 | Tee-Object -FilePath $log -Append
        $childExitCode = $LASTEXITCODE
        if ($childExitCode -ne 0) { $exitCode = $childExitCode }
        # Busy/memory pressure is a machine-level pause, not an issuer failure.
        if ($childExitCode -eq 75) { break }
    }
} finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
}
exit $exitCode
