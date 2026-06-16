param(
    [string]$ProjectDir = "C:\project\market-predictor",
    [int]$MinLiveRows = 1000,
    [double]$MinAccuracy = 0.505
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
$LogPath = Join-Path $LogDir "live_train_event_$Stamp.log"

Set-Location -LiteralPath $ProjectDir

$Args = @(
    "live-train-event",
    "--live-dir", "data\live",
    "--base-dataset", "data\features\event_swing_combined_2y_clean.parquet",
    "--min-live-rows", "$MinLiveRows",
    "--min-accuracy", "$MinAccuracy",
    "--promote",
    "--max-iter", "900",
    "--learning-rate", "0.025"
)

"[$(Get-Date -Format o)] Starting guarded event-model retrain" | Tee-Object -FilePath $LogPath
"ProjectDir=$ProjectDir" | Tee-Object -FilePath $LogPath -Append
"Command=$Cli $($Args -join ' ')" | Tee-Object -FilePath $LogPath -Append

& $Cli @Args *>> $LogPath
$ExitCode = $LASTEXITCODE

"[$(Get-Date -Format o)] Finished guarded event-model retrain with exit code $ExitCode" | Tee-Object -FilePath $LogPath -Append
exit $ExitCode
