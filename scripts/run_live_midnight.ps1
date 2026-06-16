param(
    [string]$ProjectDir = "C:\project\market-predictor",
    [string]$Tickers = "",
    [int]$LookbackDays = 3,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).Path
$Cli = Join-Path $ProjectDir ".venv\Scripts\market-predictor.exe"
if (-not (Test-Path -LiteralPath $Cli)) {
    throw "market-predictor CLI not found at $Cli"
}

$LogDir = Join-Path $ProjectDir "data\live\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "live_once_$Stamp.log"

Set-Location -LiteralPath $ProjectDir

$Args = @(
    "live-once",
    "--live-dir", "data\live",
    "--lookback-days", "$LookbackDays",
    "--workers", "$Workers"
)

if ($Tickers.Trim().Length -gt 0) {
    $Args += @("--tickers", $Tickers)
}

"[$(Get-Date -Format o)] Starting market-predictor live cycle" | Tee-Object -FilePath $LogPath
"ProjectDir=$ProjectDir" | Tee-Object -FilePath $LogPath -Append
"Command=$Cli $($Args -join ' ')" | Tee-Object -FilePath $LogPath -Append

& $Cli @Args *>> $LogPath
$ExitCode = $LASTEXITCODE

"[$(Get-Date -Format o)] Finished market-predictor live cycle with exit code $ExitCode" | Tee-Object -FilePath $LogPath -Append
exit $ExitCode
